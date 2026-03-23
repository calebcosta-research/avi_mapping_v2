"""
terrain_static.py
-----------------
Downloads and co-registers all static terrain layers needed for the
Lassen gap-zone avalanche interpolation model.

Layers produced (all written to data/terrain/):
  elevation.tif       - Raw DEM, 10m, EPSG:32610 (UTM Zone 10N)
  slope.tif           - Slope in degrees, 10m
  aspect.tif          - Aspect in degrees (0=N, 90=E, 180=S, 270=W), 10m
  aspect_class.tif    - Aspect classified to 8 directions (0=N...7=NW), 10m
  elev_band.tif       - Elevation band (1=below_treeline, 2=treeline, 3=alpine), 10m
  treeline.tif        - LANDFIRE EVT-derived treeline mask (0=below, 1=above), 30m→10m
  zone_boundaries.gpkg - Official NAC forecast zone polygons (SAC + MSAC)
  gap_zone.gpkg        - Derived Lassen gap zone polygon
  terrain_stack.tif   - All layers co-registered to a single 10m GeoTIFF stack

Data sources:
  DEM + slope + aspect : USGS 3DEP 1/3 arc-sec via The National Map API
  Treeline             : LANDFIRE 2022 EVT (Existing Vegetation Type) via LANDFIRE API
  Zone boundaries      : api.avalanche.org/v2/public/products/map-layer/
  Gap zone             : Derived from SAC + MSAC zone boundaries

Lassen bounding box (WGS84):
  N: 40.70  S: 40.15  W: -121.75  E: -121.00

Usage:
    python terrain_static.py              # full pipeline
    python terrain_static.py --check      # verify existing outputs only
    python terrain_static.py --step dem   # run a single step

PhD project: Lassen gap-zone avalanche interpolation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lassen gap zone bounding box (WGS84 decimal degrees)
# Chosen to fully encompass the area between SAC's northern boundary
# and MSAC's southern boundary, with 10km buffer on each side
BBOX = {
    "north": 40.70,
    "south": 40.15,
    "west":  -121.75,
    "east":  -121.00,
}

# Elevation thresholds for the Lassen area (meters)
# These are aspect-independent first-pass thresholds; the LANDFIRE
# treeline layer will refine these in the final elev_band product
ELEV_BELOW_TREELINE_MAX = 2100   # ~6900 ft
ELEV_TREELINE_MAX       = 2700   # ~8860 ft  (above = alpine)

# Output directory
OUT_DIR = Path("data/terrain")

# Target CRS for all raster outputs
TARGET_EPSG = 32610   # UTM Zone 10N — appropriate for Lassen

# Target resolution (meters)
TARGET_RES = 10

# Aspect class mapping (index → label, used in aspect_class.tif)
ASPECT_CLASSES = {
    0: "N",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SW",
    6: "W",
    7: "NW",
}

# Elevation band mapping (value → label, used in elev_band.tif)
ELEV_BAND_CLASSES = {
    1: "below_treeline",
    2: "treeline",
    3: "alpine",
}

# NAC center IDs bounding the gap zone
BOUNDARY_CENTERS = ["SAC", "MSAC"]

# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

# USGS 3DEP TNM API — 1/3 arc-second DEM GeoTIFF
# Documentation: https://tnmaccess.nationalmap.gov/api/v1/docs
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

# LANDFIRE async job API for Existing Vegetation Type (EVT)
# Documentation: https://www.landfire.gov/data_access.php
LANDFIRE_API = (
    "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService"
    "/GPServer/LandfireProductService/submitJob"
)

# NAC zone boundary API
NAC_MAP_LAYER = "https://api.avalanche.org/v2/public/products/map-layer/{center_id}"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, retries: int = 3, delay: float = 2.0) -> dict | list:
    """GET a URL and return parsed JSON. Retries on transient errors."""
    import urllib.parse  # noqa: ensure available
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AviGapZone-PhD/0.1 (terrain-pipeline)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            log.warning("HTTP %s on attempt %d for %s", e.code, attempt + 1, url)
            if e.code in (429, 503):
                time.sleep(delay * (attempt + 1))
            else:
                raise
        except Exception as e:
            log.warning("Error on attempt %d: %s", attempt + 1, e)
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def download_file(url: str, dest: Path, label: str = "") -> Path:
    """Download a file to dest, showing progress. Skips if already exists."""
    if dest.exists():
        log.info("Already exists, skipping: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s → %s", label or url[:60], dest.name)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AviGapZone-PhD/0.1 (terrain-pipeline)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(dest, "wb") as f:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.0f}% ({downloaded // 1024} KB)", end="", flush=True)
        print()

    log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest

# ---------------------------------------------------------------------------
# Step 1: Download DEM from USGS 3DEP
# ---------------------------------------------------------------------------

def step_dem() -> Path:
    """
    Download the 1/3 arc-second DEM for the Lassen bounding box from
    USGS The National Map. Returns path to the downloaded GeoTIFF.

    The TNM API returns a list of available tiles. We download all tiles
    that intersect our bounding box and mosaic them.
    """
    import urllib.parse

    out_raw = OUT_DIR / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    dest_mosaic = OUT_DIR / "elevation_raw.tif"

    if dest_mosaic.exists():
        log.info("DEM mosaic already exists, skipping download")
        return dest_mosaic

    log.info("Querying USGS TNM API for 3DEP tiles...")
    bbox_str = f"{BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}"
    url = (
        f"{TNM_API}?"
        f"datasets=National+Elevation+Dataset+%28NED%29+1%2F3+arc-second"
        f"&bbox={bbox_str}&max=20&outputFormat=json"
    )

    data = fetch_json(url)
    items = data.get("items", [])
    log.info("Found %d DEM tiles", len(items))

    if not items:
        raise RuntimeError(
            "No DEM tiles returned from TNM API. "
            "Try downloading manually from: https://apps.nationalmap.gov/downloader/"
            f"\nBounding box: {bbox_str}"
        )

    tile_paths = []
    for i, item in enumerate(items):
        download_url = item.get("downloadURL") or item.get("urls", {}).get("TIFF")
        if not download_url:
            log.warning("No download URL for tile %d, skipping", i)
            continue
        tile_dest = out_raw / f"dem_tile_{i:02d}.tif"
        download_file(download_url, tile_dest, label=f"DEM tile {i+1}/{len(items)}")
        tile_paths.append(tile_dest)

    # Mosaic tiles if more than one
    if len(tile_paths) == 1:
        import shutil
        shutil.copy(tile_paths[0], dest_mosaic)
    else:
        log.info("Mosaicking %d tiles...", len(tile_paths))
        _mosaic_tifs(tile_paths, dest_mosaic)

    return dest_mosaic


def _mosaic_tifs(input_paths: list[Path], output_path: Path) -> None:
    """Mosaic multiple GeoTIFFs into one using rasterio."""
    import rasterio
    from rasterio.merge import merge

    datasets = [rasterio.open(p) for p in input_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)
    for ds in datasets:
        ds.close()
    log.info("Mosaic written: %s", output_path)

# ---------------------------------------------------------------------------
# Step 2: Reproject and clip DEM to target CRS + bbox
# ---------------------------------------------------------------------------

def step_reproject_dem(raw_dem: Path) -> Path:
    """
    Reproject DEM from WGS84 geographic to UTM Zone 10N (EPSG:32610)
    at 10m resolution, clipped to the Lassen bounding box.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box
    import json

    dest = OUT_DIR / "elevation.tif"
    if dest.exists():
        log.info("Reprojected DEM exists, skipping")
        return dest

    log.info("Reprojecting DEM to UTM Zone 10N at %dm...", TARGET_RES)
    target_crs = CRS.from_epsg(TARGET_EPSG)

    with rasterio.open(raw_dem) as src:
        # Clip to bbox in source CRS first
        bbox_geom = box(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])
        clipped, clip_transform = rio_mask(
            src, [bbox_geom.__geo_interface__], crop=True, nodata=-9999
        )

        # Calculate transform for target CRS at target resolution
        transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            clipped.shape[2], clipped.shape[1],
            left=BBOX["west"], bottom=BBOX["south"],
            right=BBOX["east"], top=BBOX["north"],
            resolution=TARGET_RES,
        )

        profile = src.profile.copy()
        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": -9999,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })

        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=clipped[band - 1],
                    destination=rasterio.band(dst, band),
                    src_transform=clip_transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    log.info("DEM reprojected: %s  (%d × %d px)", dest.name, width, height)
    return dest

# ---------------------------------------------------------------------------
# Step 3: Compute slope and aspect from DEM
# ---------------------------------------------------------------------------

def step_slope_aspect(dem_path: Path) -> tuple[Path, Path]:
    """
    Compute slope (degrees) and aspect (degrees, 0=N clockwise) from DEM.
    Uses the Horn (1981) gradient method — the same used by ESRI, QGIS, GDAL.

    Returns (slope_path, aspect_path).
    """
    import rasterio

    slope_path = OUT_DIR / "slope.tif"
    aspect_path = OUT_DIR / "aspect.tif"

    if slope_path.exists() and aspect_path.exists():
        log.info("Slope and aspect exist, skipping")
        return slope_path, aspect_path

    log.info("Computing slope and aspect (Horn 1981 method)...")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata or -9999
        profile = src.profile.copy()

    # Cell size in meters (from UTM transform)
    cell_x = abs(transform.a)
    cell_y = abs(transform.e)

    # Replace nodata with NaN for calculations
    dem = np.where(dem == nodata, np.nan, dem)

    # Pad array to handle edges
    padded = np.pad(dem, 1, mode="edge")

    # Horn (1981) partial derivatives
    # dz/dx: rate of change in x direction
    dzdx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) -
        (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8 * cell_x)

    # dz/dy: rate of change in y direction
    dzdy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:]) -
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8 * cell_y)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Aspect in degrees, 0 = North, clockwise
    # np.arctan2 returns angle from +x axis, counterclockwise
    # We convert to compass bearing
    # Image y-axis is inverted relative to geographic north (row 0 = top = north)
    # so negate dzdy to convert image-space to geographic-space before
    # computing compass bearing
    aspect_rad = np.arctan2(-dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_compass = (90 - aspect_deg) % 360

    # Flat areas (slope ≈ 0) get -1 aspect (no meaningful direction)
    aspect_compass = np.where(slope < 0.5, -1, aspect_compass)

    # Write slope
    slope_profile = profile.copy()
    slope_profile.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})
    slope_out = np.where(np.isnan(slope), -9999, slope).astype(np.float32)
    with rasterio.open(slope_path, "w", **slope_profile) as dst:
        dst.write(slope_out, 1)

    # Write aspect
    aspect_profile = profile.copy()
    aspect_profile.update({"dtype": "float32", "nodata": -9999, "compress": "lzw"})
    aspect_out = np.where(np.isnan(aspect_compass), -9999, aspect_compass).astype(np.float32)
    with rasterio.open(aspect_path, "w", **aspect_profile) as dst:
        dst.write(aspect_out, 1)

    log.info("Slope range: %.1f – %.1f degrees", np.nanmin(slope), np.nanmax(slope))
    log.info("Aspect written (0=N clockwise)")
    return slope_path, aspect_path


def step_aspect_class(aspect_path: Path) -> Path:
    """
    Classify continuous aspect (degrees) into 8 compass classes.

    Class encoding matches the AvalancheForecast schema:
      0=N  1=NE  2=E  3=SE  4=S  5=SW  6=W  7=NW  255=flat/no_data
    """
    import rasterio

    out_path = OUT_DIR / "aspect_class.tif"
    if out_path.exists():
        log.info("Aspect class exists, skipping")
        return out_path

    log.info("Classifying aspect into 8 compass directions...")

    with rasterio.open(aspect_path) as src:
        aspect = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata_val = src.nodata or -9999

    # 8 classes, each covering 45 degrees, centered on cardinal/intercardinal
    # N: 337.5–360 and 0–22.5
    classified = np.full(aspect.shape, 255, dtype=np.uint8)  # 255 = flat/nodata

    valid = (aspect >= 0) & (aspect != nodata_val)

    boundaries = [
        (337.5, 360.0, 0),   # N (upper)
        (0.0,   22.5,  0),   # N (lower)
        (22.5,  67.5,  1),   # NE
        (67.5,  112.5, 2),   # E
        (112.5, 157.5, 3),   # SE
        (157.5, 202.5, 4),   # S
        (202.5, 247.5, 5),   # SW
        (247.5, 292.5, 6),   # W
        (292.5, 337.5, 7),   # NW
    ]

    for lo, hi, cls in boundaries:
        mask = valid & (aspect >= lo) & (aspect < hi)
        classified[mask] = cls

    profile.update({"dtype": "uint8", "nodata": 255, "compress": "lzw"})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(classified, 1)

    # Log distribution
    for cls_id, cls_name in ASPECT_CLASSES.items():
        count = np.sum(classified == cls_id)
        log.info("  Aspect class %s (%d): %d px (%.1f%%)",
                 cls_name, cls_id, count, count / classified.size * 100)

    return out_path

# ---------------------------------------------------------------------------
# Step 4: Elevation band classification
# ---------------------------------------------------------------------------

def step_elev_band(dem_path: Path) -> Path:
    """
    Classify DEM into 3 elevation bands matching the NAC forecast schema:
      1 = below_treeline  (< ELEV_BELOW_TREELINE_MAX meters)
      2 = treeline        (ELEV_BELOW_TREELINE_MAX – ELEV_TREELINE_MAX)
      3 = alpine          (> ELEV_TREELINE_MAX)
      255 = nodata

    These thresholds are first-pass estimates for the Lassen area.
    They will be refined in step_treeline_refine() once the LANDFIRE
    EVT layer is downloaded.
    """
    import rasterio

    out_path = OUT_DIR / "elev_band.tif"
    if out_path.exists():
        log.info("Elevation band raster exists, skipping")
        return out_path

    log.info("Classifying elevation bands...")
    log.info("  Below treeline: < %dm", ELEV_BELOW_TREELINE_MAX)
    log.info("  Treeline:       %d – %dm", ELEV_BELOW_TREELINE_MAX, ELEV_TREELINE_MAX)
    log.info("  Alpine:         > %dm", ELEV_TREELINE_MAX)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata_val = src.nodata or -9999

    band = np.full(dem.shape, 255, dtype=np.uint8)
    valid = dem != nodata_val

    band[valid & (dem < ELEV_BELOW_TREELINE_MAX)] = 1
    band[valid & (dem >= ELEV_BELOW_TREELINE_MAX) & (dem < ELEV_TREELINE_MAX)] = 2
    band[valid & (dem >= ELEV_TREELINE_MAX)] = 3

    profile.update({"dtype": "uint8", "nodata": 255, "compress": "lzw"})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band, 1)

    for val, label in ELEV_BAND_CLASSES.items():
        count = np.sum(band == val)
        log.info("  %s (%d): %d px (%.1f%%)",
                 label, val, count, count / band.size * 100)

    return out_path

# ---------------------------------------------------------------------------
# Step 5: Download LANDFIRE EVT for treeline refinement
# ---------------------------------------------------------------------------

def step_landfire() -> Optional[Path]:
    """
    Download the LANDFIRE 2022 Existing Vegetation Type (EVT) layer
    for the Lassen bounding box. This gives us field-validated treeline
    boundaries rather than simple elevation thresholds.

    LANDFIRE uses an async job API — we submit, poll, then download.
    Returns path to downloaded GeoTIFF, or None if download fails
    (the pipeline can continue with elevation-based treeline as fallback).
    """
    import urllib.parse

    dest = OUT_DIR / "landfire_evt.tif"
    if dest.exists():
        log.info("LANDFIRE EVT exists, skipping")
        return dest

    log.info("Submitting LANDFIRE EVT job (async)...")

    params = build_landfire_params()
    encoded = urllib.parse.urlencode(params).encode()

    try:
        req = urllib.request.Request(
            LANDFIRE_API,
            data=encoded,
            headers={
                "User-Agent": "AviGapZone-PhD/0.1",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        log.warning("LANDFIRE job submission failed: %s", e)
        log.warning("Continuing with elevation-based treeline (fallback)")
        return None

    job_id = result.get("jobId")
    if not job_id:
        log.warning("No job ID returned from LANDFIRE API")
        return None

    log.info("LANDFIRE job ID: %s — polling for completion...", job_id)
    job_status_url = (
        f"https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/"
        f"GPServer/LandfireProductService/jobs/{job_id}?f=json"
    )

    # Poll every 15 seconds, up to 10 minutes
    for attempt in range(40):
        time.sleep(15)
        try:
            status = fetch_json(job_status_url)
            job_status = status.get("jobStatus", "")
            log.info("  LANDFIRE job status: %s (attempt %d)", job_status, attempt + 1)
            if job_status == "esriJobSucceeded":
                break
            elif job_status in ("esriJobFailed", "esriJobCancelled"):
                log.warning("LANDFIRE job failed: %s", status)
                return None
        except Exception as e:
            log.warning("Poll error: %s", e)
    else:
        log.warning("LANDFIRE job timed out — using elevation-based treeline")
        return None

    # Get result download URL
    result_url = (
        f"https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/"
        f"GPServer/LandfireProductService/jobs/{job_id}/results/Output_File?f=json"
    )
    try:
        result_data = fetch_json(result_url)
        download_url = result_data.get("value", {}).get("url")
        if not download_url:
            log.warning("No download URL in LANDFIRE result")
            return None
        download_file(download_url, dest, label="LANDFIRE EVT")
        return dest
    except Exception as e:
        log.warning("LANDFIRE download failed: %s — using elevation-based treeline", e)
        return None


def step_treeline_refine(elev_band_path: Path, evt_path: Optional[Path]) -> Path:
    """
    Refine elevation band classification using LANDFIRE EVT.
    If EVT is unavailable, returns the original elevation-based product as-is.

    LANDFIRE EVT values for forested / subalpine / alpine classes:
      Values >= 7000 and <= 7290 cover subalpine/alpine vegetation types
      (Subalpine Woodland, Alpine Sparse, Alpine Meadow, etc.)
    These pixels are reclassified to treeline or alpine bands.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    out_path = OUT_DIR / "elev_band_refined.tif"

    if evt_path is None or not evt_path.exists():
        log.info("No LANDFIRE EVT — using elevation-based band classification as-is")
        import shutil
        shutil.copy(elev_band_path, out_path)
        return out_path

    if out_path.exists():
        log.info("Refined elevation band exists, skipping")
        return out_path

    log.info("Refining treeline using LANDFIRE EVT...")

    with rasterio.open(elev_band_path) as base_src:
        base_band = base_src.read(1)
        profile = base_src.profile.copy()
        transform = base_src.transform
        crs = base_src.crs

    with rasterio.open(evt_path) as evt_src:
        # Reproject EVT to match elev_band grid
        evt_reprojected = np.zeros_like(base_band, dtype=np.int16)
        reproject(
            source=rasterio.band(evt_src, 1),
            destination=evt_reprojected,
            src_transform=evt_src.transform,
            src_crs=evt_src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

    refined = base_band.copy()

    # EVT alpine/subalpine classes (LANDFIRE 2022 codes):
    # 7190 = Rocky Mountain Subalpine Woodland
    # 7191 = Sierra Nevada Subalpine Woodland
    # 7290 = Rocky Mountain Alpine Sparse Vegetation
    # 7291 = Sierra Nevada Alpine
    # 7292 = Sierra Nevada-Cascade Alpine Meadow
    alpine_subalpine_codes = set(range(7190, 7300))
    alpine_mask = np.isin(evt_reprojected, list(alpine_subalpine_codes))

    # Where LANDFIRE says subalpine/alpine but elevation said below_treeline,
    # upgrade to treeline (conservative correction)
    refined[(alpine_mask) & (base_band == 1)] = 2

    # Where LANDFIRE says alpine and elevation also says treeline+, upgrade to alpine
    # EVT codes 7290+ are purely alpine (no trees)
    pure_alpine_codes = set(range(7290, 7300))
    pure_alpine_mask = np.isin(evt_reprojected, list(pure_alpine_codes))
    refined[(pure_alpine_mask) & (base_band == 2)] = 3

    profile.update({"dtype": "uint8", "nodata": 255, "compress": "lzw"})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(refined, 1)

    changes = np.sum(refined != base_band)
    log.info("  Refined %d pixels using LANDFIRE EVT (%.2f%% of area)",
             changes, changes / base_band.size * 100)
    return out_path

# ---------------------------------------------------------------------------
# Step 6: Download forecast zone boundaries from NAC
# ---------------------------------------------------------------------------

def step_zone_boundaries() -> tuple[Path, Path]:
    """
    Download official forecast zone GeoJSON for SAC and MSAC from
    api.avalanche.org, save as GeoPackage, and derive the gap zone polygon.

    Returns (zone_boundaries_path, gap_zone_path).
    """
    import geopandas as gpd
    from shapely.geometry import shape
    from shapely.ops import unary_union

    zone_path = OUT_DIR / "zone_boundaries.gpkg"
    gap_path = OUT_DIR / "gap_zone.gpkg"

    if zone_path.exists() and gap_path.exists():
        log.info("Zone boundaries and gap zone exist, skipping")
        return zone_path, gap_path

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []

    for center_id in BOUNDARY_CENTERS:
        url = NAC_MAP_LAYER.format(center_id=center_id)
        log.info("Fetching zone boundaries for %s...", center_id)
        try:
            geojson = fetch_json(url)
            features = geojson.get("features", [])
            log.info("  %s: %d zones", center_id, len(features))
            for feat in features:
                feat["properties"]["center_id"] = center_id
            all_features.extend(features)
        except Exception as e:
            log.warning("Could not fetch %s boundaries: %s", center_id, e)

    if not all_features:
        raise RuntimeError(
            "No zone boundaries downloaded. Check network connection or "
            "manually download from api.avalanche.org/v2/public/products/map-layer/SAC"
        )

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=TARGET_EPSG)
    gdf.to_file(zone_path, driver="GPKG", layer="forecast_zones")
    log.info("Saved %d forecast zones → %s", len(gdf), zone_path.name)

    # Derive gap zone: Lassen bbox minus all official forecast zones
    from shapely.geometry import box as shapely_box
    from pyproj import Transformer

    # Transform bbox to UTM
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{TARGET_EPSG}", always_xy=True)
    w, s = transformer.transform(BBOX["west"], BBOX["south"])
    e, n = transformer.transform(BBOX["east"], BBOX["north"])
    lassen_bbox = shapely_box(w, s, e, n)

    # Dissolve all official zones and subtract from bbox
    all_zones_union = unary_union(gdf.geometry.values)
    gap_polygon = lassen_bbox.difference(all_zones_union)

    gap_gdf = gpd.GeoDataFrame(
        [{"name": "Lassen_gap_zone", "description": "Area between SAC and MSAC with no official forecast"}],
        geometry=[gap_polygon],
        crs=f"EPSG:{TARGET_EPSG}",
    )
    gap_gdf.to_file(gap_path, driver="GPKG", layer="gap_zone")
    log.info("Gap zone area: %.0f km²", gap_polygon.area / 1e6)
    log.info("Saved gap zone → %s", gap_path.name)

    return zone_path, gap_path

# ---------------------------------------------------------------------------
# Step 7: Build terrain stack
# ---------------------------------------------------------------------------

def step_terrain_stack(
    dem_path: Path,
    slope_path: Path,
    aspect_path: Path,
    aspect_class_path: Path,
    elev_band_path: Path,
) -> Path:
    """
    Stack all terrain layers into a single multi-band GeoTIFF.
    All bands are already co-registered (same transform, same CRS, same size).

    Band order:
      1 - elevation (float32, meters)
      2 - slope (float32, degrees)
      3 - aspect (float32, degrees 0=N)
      4 - aspect_class (uint8, 0–7)
      5 - elev_band (uint8, 1–3)
    """
    import rasterio

    out_path = OUT_DIR / "terrain_stack.tif"
    if out_path.exists():
        log.info("Terrain stack exists, skipping")
        return out_path

    log.info("Building terrain stack...")

    band_configs = [
        (dem_path,          "float32", -9999, "elevation_m"),
        (slope_path,        "float32", -9999, "slope_deg"),
        (aspect_path,       "float32", -9999, "aspect_deg"),
        (aspect_class_path, "uint8",   255,   "aspect_class"),
        (elev_band_path,    "uint8",   255,   "elev_band"),
    ]

    with rasterio.open(dem_path) as ref:
        profile = ref.profile.copy()
        height, width = ref.shape

    profile.update({
        "count": len(band_configs),
        "dtype": "float32",
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        for band_idx, (src_path, dtype, nodata, name) in enumerate(band_configs, start=1):
            with rasterio.open(src_path) as src:
                data = src.read(1).astype(np.float32)
                data = np.where(data == nodata, -9999, data)
                dst.write(data, band_idx)
            dst.update_tags(band_idx, name=name)
            log.info("  Band %d: %s", band_idx, name)

    log.info("Terrain stack written: %s", out_path)
    log.info("  Shape: %d bands × %d × %d px", len(band_configs), height, width)
    log.info("  Resolution: %dm", TARGET_RES)
    return out_path

# ---------------------------------------------------------------------------
# Step 8: Avalanche terrain analysis
# ---------------------------------------------------------------------------

# Slope thresholds (degrees) for avalanche-relevant terrain.
# Below 25°: rarely produces slab avalanches.
# Above 60°: typically self-clears too frequently to build a slab.
AVI_SLOPE_MIN = 25.0
AVI_SLOPE_MAX = 60.0


def step_avi_terrain(
    slope_path: Path,
    aspect_class_path: Path,
    elev_band_path: Path,
) -> tuple[Path, Path, Path]:
    """
    Identify avalanche-relevant terrain cells and produce three outputs:

    avi_terrain_mask.tif
        Binary raster (1 = avalanche terrain, 0 = non-avi, 255 = nodata).
        Avalanche terrain is defined as AVI_SLOPE_MIN ≤ slope ≤ AVI_SLOPE_MAX
        with a valid aspect and elevation band.

    avi_terrain_cells.tif
        Cell-ID raster for avi-terrain pixels:  aspect_class_id * 10 + elev_band_id
        (values 1–73, matching the NAC aspect × elevation schema).
        Non-avi pixels = 255.

        Cell ID lookup:
          aspect_class_id : 0=N  1=NE  2=E  3=SE  4=S  5=SW  6=W  7=NW
          elev_band_id    : 1=below_treeline  2=treeline  3=alpine
          Example: SE × alpine → 3*10+3 = 33

    avi_terrain_summary.json
        Pixel counts and area (km²) per (aspect, elevation_band) cell.
        This is the primary input to the interpolation model — it tells the
        model which spatial cells exist in the gap zone and how large they are.

    Returns (mask_path, cells_path, summary_path).
    """
    import rasterio

    mask_path    = OUT_DIR / "avi_terrain_mask.tif"
    cells_path   = OUT_DIR / "avi_terrain_cells.tif"
    summary_path = OUT_DIR / "avi_terrain_summary.json"

    if mask_path.exists() and cells_path.exists() and summary_path.exists():
        log.info("Avalanche terrain layers exist, skipping")
        return mask_path, cells_path, summary_path

    log.info(
        "Identifying avalanche terrain (slope %.0f°–%.0f°)...",
        AVI_SLOPE_MIN, AVI_SLOPE_MAX,
    )

    with rasterio.open(slope_path) as src:
        slope      = src.read(1).astype(np.float32)
        nodata_slp = float(src.nodata) if src.nodata is not None else -9999.0
        profile    = src.profile.copy()

    with rasterio.open(aspect_class_path) as src:
        asp_cls    = src.read(1)   # uint8, 0–7, nodata=255

    with rasterio.open(elev_band_path) as src:
        elev_band  = src.read(1)   # uint8, 1–3, nodata=255

    # --- Build avi-terrain mask ---
    valid_slope  = (slope != nodata_slp) & np.isfinite(slope)
    valid_aspect = asp_cls != 255
    valid_elev   = (elev_band >= 1) & (elev_band <= 3)

    avi_mask = (
        valid_slope  &
        valid_aspect &
        valid_elev   &
        (slope >= AVI_SLOPE_MIN) &
        (slope <= AVI_SLOPE_MAX)
    )

    # --- Binary mask raster ---
    mask_arr = np.where(avi_mask, np.uint8(1), np.uint8(0))
    mask_profile = profile.copy()
    mask_profile.update({"dtype": "uint8", "nodata": 255, "count": 1, "compress": "lzw"})
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask_arr, 1)

    # --- Cell ID raster ---
    # Encode as aspect_class_id * 10 + elev_band_id  (range 1–73)
    cell_id = np.where(
        avi_mask,
        (asp_cls.astype(np.uint16) * 10 + elev_band.astype(np.uint16)).astype(np.uint8),
        np.uint8(255),
    )
    cells_profile = profile.copy()
    cells_profile.update({"dtype": "uint8", "nodata": 255, "count": 1, "compress": "lzw"})
    with rasterio.open(cells_path, "w", **cells_profile) as dst:
        dst.write(cell_id, 1)

    # --- Summary JSON ---
    total_avi_px   = int(np.sum(avi_mask))
    total_avi_km2  = total_avi_px * (TARGET_RES ** 2) / 1e6
    total_px       = int(avi_mask.size)
    total_km2      = total_px * (TARGET_RES ** 2) / 1e6

    summary: dict = {
        "slope_filter_deg": {"min": AVI_SLOPE_MIN, "max": AVI_SLOPE_MAX},
        "resolution_m":     TARGET_RES,
        "bbox":             BBOX,
        "total_pixels":     total_px,
        "total_area_km2":   round(total_km2, 2),
        "avi_pixels":       total_avi_px,
        "avi_area_km2":     round(total_avi_km2, 2),
        "avi_fraction":     round(total_avi_px / total_px, 4) if total_px else 0,
        "cells":            {},
    }

    log.info("  %-26s  %8s  %8s", "Cell (aspect × elev_band)", "pixels", "area km²")
    log.info("  " + "-" * 50)

    for asp_id, asp_name in ASPECT_CLASSES.items():
        for band_id, band_name in ELEV_BAND_CLASSES.items():
            cell_mask = avi_mask & (asp_cls == asp_id) & (elev_band == band_id)
            px_count  = int(np.sum(cell_mask))
            area_km2  = round(px_count * (TARGET_RES ** 2) / 1e6, 3)
            key       = f"{asp_name}_{band_name}"

            summary["cells"][key] = {
                "aspect":           asp_name,
                "elevation_band":   band_name,
                "aspect_class_id":  asp_id,
                "elev_band_id":     band_id,
                "cell_id":          asp_id * 10 + band_id,
                "pixel_count":      px_count,
                "area_km2":         area_km2,
            }

            if px_count > 0:
                log.info(
                    "  %-26s  %8d  %8.2f",
                    key, px_count, area_km2,
                )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("  " + "-" * 50)
    log.info(
        "  Total avi terrain: %d px = %.1f km² (%.1f%% of bbox)",
        total_avi_px, total_avi_km2, summary["avi_fraction"] * 100,
    )
    log.info("Cell summary → %s", summary_path)

    return mask_path, cells_path, summary_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def step_verify(outputs: dict) -> None:
    """Print a summary table of all output files."""
    print("\n" + "="*60)
    print("  TERRAIN PIPELINE — OUTPUT SUMMARY")
    print("="*60)
    expected = [
        "elevation.tif",
        "slope.tif",
        "aspect.tif",
        "aspect_class.tif",
        "elev_band.tif",
        "elev_band_refined.tif",
        "landfire_evt.tif",
        "zone_boundaries.gpkg",
        "gap_zone.gpkg",
        "terrain_stack.tif",
        "avi_terrain_mask.tif",
        "avi_terrain_cells.tif",
        "avi_terrain_summary.json",
    ]
    for fname in expected:
        path = OUT_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            status = f"OK  {size_mb:6.1f} MB"
        else:
            status = "--  MISSING"
        print(f"  {status}  {fname}")

    print("="*60)

    # Summarize terrain_stack if it exists
    stack_path = OUT_DIR / "terrain_stack.tif"
    if stack_path.exists():
        try:
            import rasterio
            with rasterio.open(stack_path) as src:
                print(f"\n  Stack CRS:        {src.crs}")
                print(f"  Stack resolution: {src.res[0]}m × {src.res[1]}m")
                print(f"  Stack dimensions: {src.width} × {src.height} px")
                print(f"  Stack bounds:     {src.bounds}")
                print(f"  Stack bands:      {src.count}")
        except ImportError:
            print("\n  (install rasterio to show stack details)")

    print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static terrain pipeline for Lassen gap zone")
    p.add_argument(
        "--step",
        choices=["dem", "reproject", "slope_aspect", "aspect_class",
                 "elev_band", "landfire", "zones", "stack", "avi_terrain",
                 "verify", "all"],
        default="all",
        help="Run a specific step (default: all)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Only verify existing outputs, do not download or process",
    )
    p.add_argument(
        "--bbox-north", type=float, default=BBOX["north"], help="Override bbox north"
    )
    p.add_argument(
        "--bbox-south", type=float, default=BBOX["south"], help="Override bbox south"
    )
    p.add_argument(
        "--bbox-west",  type=float, default=BBOX["west"],  help="Override bbox west"
    )
    p.add_argument(
        "--bbox-east",  type=float, default=BBOX["east"],  help="Override bbox east"
    )
    return p.parse_args()


def main() -> None:
    import urllib.parse  # ensure available in all paths

    args = parse_args()

    # Apply any bbox overrides
    BBOX["north"] = args.bbox_north
    BBOX["south"] = args.bbox_south
    BBOX["west"]  = args.bbox_west
    BBOX["east"]  = args.bbox_east

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        step_verify({})
        return

    outputs = {}

    step = args.step

    try:
        if step in ("dem", "all"):
            outputs["raw_dem"] = step_dem()

        if step in ("reproject", "all"):
            raw = outputs.get("raw_dem") or OUT_DIR / "elevation_raw.tif"
            outputs["dem"] = step_reproject_dem(raw)

        if step in ("slope_aspect", "all"):
            dem = outputs.get("dem") or OUT_DIR / "elevation.tif"
            outputs["slope"], outputs["aspect"] = step_slope_aspect(dem)

        if step in ("aspect_class", "all"):
            aspect = outputs.get("aspect") or OUT_DIR / "aspect.tif"
            outputs["aspect_class"] = step_aspect_class(aspect)

        if step in ("elev_band", "all"):
            dem = outputs.get("dem") or OUT_DIR / "elevation.tif"
            outputs["elev_band_raw"] = step_elev_band(dem)

        if step in ("landfire", "all"):
            evt_path = step_landfire()
            elev_band_raw = outputs.get("elev_band_raw") or OUT_DIR / "elev_band.tif"
            outputs["elev_band"] = step_treeline_refine(elev_band_raw, evt_path)

        if step in ("zones", "all"):
            outputs["zones"], outputs["gap_zone"] = step_zone_boundaries()

        if step in ("stack", "all"):
            dem        = outputs.get("dem")          or OUT_DIR / "elevation.tif"
            slope      = outputs.get("slope")        or OUT_DIR / "slope.tif"
            aspect     = outputs.get("aspect")       or OUT_DIR / "aspect.tif"
            asp_class  = outputs.get("aspect_class") or OUT_DIR / "aspect_class.tif"
            elev_band  = outputs.get("elev_band")    or OUT_DIR / "elev_band_refined.tif"
            outputs["stack"] = step_terrain_stack(dem, slope, aspect, asp_class, elev_band)

        if step in ("avi_terrain", "all"):
            slope_p    = outputs.get("slope")        or OUT_DIR / "slope.tif"
            asp_cls_p  = outputs.get("aspect_class") or OUT_DIR / "aspect_class.tif"
            elev_b_p   = outputs.get("elev_band")    or OUT_DIR / "elev_band_refined.tif"
            (
                outputs["avi_mask"],
                outputs["avi_cells"],
                outputs["avi_summary"],
            ) = step_avi_terrain(slope_p, asp_cls_p, elev_b_p)

        step_verify(outputs)
        log.info("Pipeline complete.")

    except Exception as e:
        log.error("Pipeline failed: %s", e)
        log.info("Run with --check to see which outputs exist so far")
        raise


if __name__ == "__main__":
    main()
