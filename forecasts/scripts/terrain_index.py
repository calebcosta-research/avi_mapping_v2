"""
terrain_index.py
----------------
Builds a unified terrain index across all NAC forecast zones.

For each zone:
  1. Downloads 1 arc-sec (~30m) DEM tiles from USGS 3DEP (cached per zone)
  2. Reprojects to local UTM for accurate slope/aspect calculation
  3. Filters to avalanche-relevant terrain: elevation > 762m (2500ft), slope 30-45 degrees
  4. Classifies each surviving pixel by aspect (8-class) and elevation band
  5. Vectorizes and dissolves into polygons per (aspect_class, elev_band)
  6. Clips to zone boundary, tags with center_id / zone metadata

Output:
  forecasts/output/terrain_index.geojson   — all zones merged, WGS84
  forecasts/output/terrain_cache/          — per-zone cached GeoPackages

Usage:
    python terrain_index.py                        # process all zones
    python terrain_index.py --zone 2856            # single zone (BTAC Salt River)
    python terrain_index.py --center BTAC          # all zones for one center
    python terrain_index.py --check                # report cache status
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import os
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
# Constants
# ---------------------------------------------------------------------------

ELEV_MIN_M        = 762.0    # 2500 ft — discard below-terrain pixels
SLOPE_MIN_DEG     = 30.0     # minimum avalanche slope
SLOPE_MAX_DEG     = 45.0     # maximum (above this is cliff, not slab terrain)
BBOX_BUFFER_DEG   = 0.05     # degrees of padding around zone bbox for DEM download
SIMPLIFY_TOLERANCE = 0.003   # degrees (~300m at mid-latitudes) for geometry simplification
VECTORIZE_RES_M    = 200     # downsample to this resolution (m) before vectorizing
SMOOTH_BUFFER_M    = 200     # morphological close: expand then contract to merge nearby patches

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASET = "National Elevation Dataset (NED) 1/3 arc-second"

ASPECT_CLASSES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Elevation band thresholds — will be overridden per zone based on actual DEM range
# These are reasonable defaults for continental US mountain ranges
ELEV_TREELINE_MIN  = 2400    # meters — below this is below_treeline
ELEV_ALPINE_MIN    = 3000    # meters — above this is alpine

OUT_DIR   = Path(__file__).parent.parent / "output"
CACHE_DIR = OUT_DIR / "terrain_cache"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utm_epsg(lon: float, lat: float) -> int:
    """Return EPSG code for the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def fetch_bytes(url: str, label: str = "", retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AviTerrain/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning("Retry %d for %s: %s", attempt + 1, label, e)
            time.sleep(2 ** attempt)


def fetch_json(url: str) -> dict | list:
    return json.loads(fetch_bytes(url).decode())


# ---------------------------------------------------------------------------
# DEM download
# ---------------------------------------------------------------------------

TILE_CACHE_DIR = CACHE_DIR / "tiles"


def _tile_filename(url: str) -> str:
    """Derive a stable filename from a TNM tile URL."""
    return Path(urllib.parse.urlparse(url).path).name or url.split("/")[-1]


def query_tnm_tiles(bbox: dict) -> list[str]:
    """Return list of DEM tile download URLs for the given bbox."""
    params = {
        "datasets":     TNM_DATASET,
        "bbox":         f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "max":          50,
        "outputFormat": "json",
    }
    url = TNM_API + "?" + urllib.parse.urlencode(params)
    data = fetch_json(url)
    items = data.get("items", [])
    urls = [item["downloadURL"] for item in items if item.get("downloadURL")]
    return urls


def download_dem_for_bbox(bbox: dict, cache_path: Path) -> Optional[Path]:
    """
    Download and mosaic DEM tiles for bbox. Returns path to merged GeoTIFF.
    Tiles are cached individually in TILE_CACHE_DIR so adjacent zones
    never re-download the same tile. The mosaicked zone DEM is cached at
    cache_path.
    """
    if cache_path.exists():
        return cache_path

    import rasterio
    from rasterio.merge import merge

    TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tile_urls = query_tnm_tiles(bbox)
    if not tile_urls:
        log.warning("No DEM tiles found for bbox %s", bbox)
        return None

    def _tile_valid(path: Path) -> bool:
        """Quick check that a cached tile is a readable GeoTIFF."""
        try:
            with rasterio.open(path):
                return True
        except Exception:
            return False

    tile_paths = []
    new_downloads = 0
    for url in tile_urls:
        fname = _tile_filename(url)
        tile_path = TILE_CACHE_DIR / fname
        if tile_path.exists() and _tile_valid(tile_path):
            tile_paths.append(tile_path)
        else:
            if tile_path.exists():
                log.warning("  Corrupt cached tile %s — re-downloading", fname)
                tile_path.unlink()
            try:
                data = fetch_bytes(url, label=fname)
                tile_path.write_bytes(data)
                if _tile_valid(tile_path):
                    tile_paths.append(tile_path)
                    new_downloads += 1
                else:
                    log.warning("  Downloaded tile still invalid: %s", fname)
                    tile_path.unlink(missing_ok=True)
            except Exception as e:
                log.warning("  Failed tile %s: %s", fname, e)

    cached = len(tile_urls) - new_downloads
    log.info("  DEM tiles: %d downloaded, %d from cache", new_downloads, cached)

    if not tile_paths:
        return None

    # Mosaic tiles into zone DEM — write with LZW compression to save disk
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update({"driver": "GTiff", "height": mosaic.shape[1],
                    "width": mosaic.shape[2], "transform": transform,
                    "compress": "lzw",    # ~4x smaller than uncompressed
                    "predictor": 2})      # horizontal differencing for elevation data
    for ds in datasets:
        ds.close()

    with rasterio.open(cache_path, "w", **profile) as dst:
        dst.write(mosaic)

    return cache_path


# ---------------------------------------------------------------------------
# Terrain computation
# ---------------------------------------------------------------------------

def reproject_to_utm(src_path: Path, epsg: int) -> tuple[np.ndarray, object]:
    """
    Reproject DEM to UTM. Returns (elevation_array, rasterio_profile).
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(src_path) as src:
        crs_target = f"EPSG:{epsg}"
        transform, width, height = calculate_default_transform(
            src.crs, crs_target, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update({
            "crs": crs_target,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": -9999,
        })
        elevation = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs_target,
            resampling=Resampling.bilinear,
        )

    return elevation, profile


def compute_slope_aspect(elevation: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Horn (1981) gradient method. Returns (slope_deg, aspect_deg).
    aspect_deg: 0=N, 90=E, 180=S, 270=W (clockwise from north).
    Edges are set to nodata (-1).
    """
    nodata = -9999.0
    valid = elevation != nodata

    # Pad with edge values for gradient computation
    e = np.pad(elevation, 1, mode="edge")

    dz_dx = ((e[:-2, 2:] + 2*e[1:-1, 2:] + e[2:, 2:]) -
             (e[:-2, :-2] + 2*e[1:-1, :-2] + e[2:, :-2])) / (8 * cell_size)
    dz_dy = ((e[2:, :-2] + 2*e[2:, 1:-1] + e[2:, 2:]) -
             (e[:-2, :-2] + 2*e[:-2, 1:-1] + e[:-2, 2:])) / (8 * cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Negate dz_dx so that east-facing slopes (which rise going west, dz_dx < 0)
    # correctly map to 90°. Without this, E and W are swapped.
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360   # 0-360, 0=N clockwise

    slope_deg[~valid] = -1
    aspect_deg[~valid] = -1

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def classify_aspect(aspect_deg: np.ndarray) -> np.ndarray:
    """
    Classify aspect degrees to 8-class index:
    0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    Flat/nodata → 255
    """
    out = np.full(aspect_deg.shape, 255, dtype=np.uint8)
    bins = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    classes = [1, 2, 3, 4, 5, 6, 7, 0]  # NE, E, SE, S, SW, W, NW, N
    valid = aspect_deg >= 0

    out[valid & (aspect_deg < bins[0])] = 0   # N
    out[valid & (aspect_deg >= bins[7])] = 0   # N (wraps)
    for i, (lo, hi) in enumerate(zip(bins, bins[1:])):
        out[valid & (aspect_deg >= lo) & (aspect_deg < hi)] = classes[i]

    return out


def classify_elev_band(elevation: np.ndarray,
                       treeline_min: float = ELEV_TREELINE_MIN,
                       alpine_min: float = ELEV_ALPINE_MIN) -> np.ndarray:
    """
    1 = below_treeline, 2 = treeline, 3 = alpine
    """
    band = np.ones(elevation.shape, dtype=np.uint8)
    band[elevation >= treeline_min] = 2
    band[elevation >= alpine_min]   = 3
    band[elevation <= 0]            = 0
    return band


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

def vectorize_terrain_cells(
    mask: np.ndarray,
    aspect_class: np.ndarray,
    elev_band: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    profile: dict,
    zone_geom_wgs84,
    epsg: int,
) -> Optional[object]:
    """
    Convert filtered raster cells to dissolved GeoDataFrame polygons.
    Groups contiguous pixels with same (aspect_class, elev_band) into one polygon.
    Returns GeoDataFrame in WGS84, or None if no terrain survives the filters.
    """
    import rasterio.features
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from pyproj import Transformer

    if not mask.any():
        return None

    transform = profile["transform"]
    cell_m = abs(transform.a)  # current pixel size in metres

    # Build label raster at full resolution: aspect_class * 10 + elev_band
    label_full = np.where(mask, aspect_class.astype(np.int16) * 10 + elev_band, -1).astype(np.int16)

    # Downsample to VECTORIZE_RES_M using nearest-neighbor (preserves class values).
    # This collapses the pixel count by (VECTORIZE_RES_M/cell_m)^2, making
    # shapes() tractable on large zones.
    factor = max(1, int(round(VECTORIZE_RES_M / cell_m)))
    if factor > 1:
        label = label_full[::factor, ::factor]
        from rasterio.transform import Affine
        transform = Affine(
            transform.a * factor, transform.b, transform.c,
            transform.d, transform.e * factor, transform.f,
        )
        log.info("  Downsampled label raster %dx (%.0fm -> %dm) for vectorization",
                 factor, cell_m, cell_m * factor)
    else:
        label = label_full

    # Collect shapes per label value
    rows = []
    for geom_dict, val in rasterio.features.shapes(label, transform=transform):
        if val < 0:
            continue
        ac = int(val) // 10
        eb = int(val) % 10
        if ac > 7 or eb not in (1, 2, 3):
            continue
        geom = shape(geom_dict)
        rows.append({
            "geometry":     geom,
            "aspect_class": ac,
            "elev_band":    eb,
            "label":        int(val),
        })

    if not rows:
        return None

    gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{epsg}")

    # Dissolve by (aspect_class, elev_band)
    gdf = gdf.dissolve(by=["aspect_class", "elev_band"]).reset_index()

    # Morphological close in UTM (metres): expand to merge nearby same-class patches,
    # then contract back — produces smooth, connected blobs instead of patchwork
    if SMOOTH_BUFFER_M > 0:
        gdf["geometry"] = (
            gdf["geometry"]
            .buffer(SMOOTH_BUFFER_M)
            .buffer(-SMOOTH_BUFFER_M * 0.6)
        )
        gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]

    # Reproject to WGS84
    gdf = gdf.to_crs("EPSG:4326")

    # Clip to zone boundary
    zone_gdf = gpd.GeoDataFrame(geometry=[zone_geom_wgs84], crs="EPSG:4326")
    try:
        gdf = gpd.clip(gdf, zone_gdf)
    except Exception as e:
        log.warning("  Clip failed: %s — skipping clip", e)

    if gdf.empty:
        return None

    # Simplify for web delivery
    gdf["geometry"] = gdf["geometry"].simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    gdf = gdf[~gdf["geometry"].is_empty]

    # Decode labels to human-readable names
    gdf["aspect"]    = gdf["aspect_class"].map(lambda i: ASPECT_CLASSES[i] if i < 8 else "")
    elev_band_names  = {1: "below_treeline", 2: "treeline", 3: "alpine"}
    gdf["elev_band_name"] = gdf["elev_band"].map(elev_band_names)

    return gdf


# ---------------------------------------------------------------------------
# Per-zone processing
# ---------------------------------------------------------------------------

def process_zone(feature: dict) -> Optional[object]:
    """
    Full terrain processing pipeline for one zone feature.
    Returns a GeoDataFrame with terrain polygons, or None on failure.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    props    = feature["properties"]
    zone_id  = props["zone_id"]
    center   = props["center_id"]
    zone_nm  = props["zone_name"]
    geometry = feature["geometry"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_gpkg = CACHE_DIR / f"zone_{zone_id}.gpkg"

    if cache_gpkg.exists():
        log.info("[%s] %s — loading from cache", center, zone_nm)
        return gpd.read_file(cache_gpkg)

    log.info("[%s] %s — processing...", center, zone_nm)

    geom_shape = shape(geometry)
    minx, miny, maxx, maxy = geom_shape.bounds

    # Expand bbox with buffer for DEM edge effects
    bbox = {
        "west":  minx - BBOX_BUFFER_DEG,
        "east":  maxx + BBOX_BUFFER_DEG,
        "south": miny - BBOX_BUFFER_DEG,
        "north": maxy + BBOX_BUFFER_DEG,
    }

    # Determine UTM CRS for accurate slope calculation
    centroid_lon = (minx + maxx) / 2
    centroid_lat = (miny + maxy) / 2
    epsg = utm_epsg(centroid_lon, centroid_lat)

    # Download DEM (cached as raw WGS84 GeoTIFF)
    dem_raw = CACHE_DIR / f"dem_raw_{zone_id}.tif"
    dem_path = download_dem_for_bbox(bbox, dem_raw)
    if dem_path is None:
        log.warning("[%s] %s — DEM download failed, skipping", center, zone_nm)
        return None

    # Reproject to UTM, get elevation array
    elevation, profile = reproject_to_utm(dem_path, epsg)
    cell_size = abs(profile["transform"].a)  # meters per pixel

    # Compute slope/aspect
    slope, aspect = compute_slope_aspect(elevation, cell_size)

    # Filter: elevation > 762m, slope 30-45°
    mask = (
        (elevation > ELEV_MIN_M) &
        (slope >= SLOPE_MIN_DEG) &
        (slope <= SLOPE_MAX_DEG) &
        (elevation != -9999)
    )

    pct = mask.sum() / mask.size * 100
    log.info("  Surviving pixels after filter: %d (%.1f%%)", mask.sum(), pct)

    if not mask.any():
        log.info("  No avalanche terrain in this zone — skipping")
        # Save empty file so we don't retry
        gpd.GeoDataFrame(columns=["geometry"]).to_file(cache_gpkg, driver="GPKG")
        return None

    # Classify aspect and elevation band
    aspect_class = classify_aspect(aspect)

    # Adaptive treeline thresholds based on zone's elevation range
    valid_elev = elevation[(elevation > 0) & (elevation != -9999)]
    elev_p75 = float(np.percentile(valid_elev, 75)) if valid_elev.size else ELEV_TREELINE_MIN
    elev_p90 = float(np.percentile(valid_elev, 90)) if valid_elev.size else ELEV_ALPINE_MIN
    treeline_m = max(ELEV_TREELINE_MIN, elev_p75)
    alpine_m   = max(ELEV_ALPINE_MIN,   elev_p90)
    log.info("  Elevation bands: treeline > %.0fm, alpine > %.0fm", treeline_m, alpine_m)

    elev_band = classify_elev_band(elevation, treeline_m, alpine_m)

    # Vectorize
    gdf = vectorize_terrain_cells(
        mask, aspect_class, elev_band, elevation, slope,
        profile, geom_shape, epsg
    )

    if gdf is None or gdf.empty:
        log.info("  Vectorization produced no features")
        gpd.GeoDataFrame(columns=["geometry"]).to_file(cache_gpkg, driver="GPKG")
        return None

    # Tag with zone metadata
    gdf["zone_id"]    = zone_id
    gdf["center_id"]  = center
    gdf["zone_name"]  = zone_nm
    gdf["state"]      = props.get("state", "")

    log.info("  -> %d terrain polygons", len(gdf))
    gdf.to_file(cache_gpkg, driver="GPKG")

    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_zone_boundaries() -> list[dict]:
    path = OUT_DIR / "zone_boundaries.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run zone_boundaries.py first"
        )
    with open(path) as f:
        geo = json.load(f)
    return geo["features"]


def run(zone_ids: Optional[list[int]] = None,
        center_filter: Optional[list[str]] = None,
        exclude_centers: Optional[list[str]] = None) -> None:
    import geopandas as gpd
    import pandas as pd

    features = load_zone_boundaries()

    # Apply filters
    if zone_ids:
        features = [f for f in features if f["properties"]["zone_id"] in zone_ids]
    if center_filter:
        centers = [c.upper() for c in center_filter]
        features = [f for f in features if f["properties"]["center_id"] in centers]
    if exclude_centers:
        excl = [c.upper() for c in exclude_centers]
        features = [f for f in features if f["properties"]["center_id"] not in excl]

    log.info("Processing %d zone(s)...", len(features))

    all_gdfs = []
    failed = []

    for i, feat in enumerate(features, 1):
        props = feat["properties"]
        log.info("--- Zone %d/%d: [%s] %s",
                 i, len(features), props["center_id"], props["zone_name"])
        try:
            gdf = process_zone(feat)
            if gdf is not None and not gdf.empty:
                all_gdfs.append(gdf)
        except Exception as e:
            log.error("Zone %s failed: %s", props["zone_id"], e)
            failed.append(props["zone_id"])

    if not all_gdfs:
        log.warning("No terrain data produced.")
        return

    # Merge all zones
    merged = pd.concat(all_gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs="EPSG:4326")

    out_path = OUT_DIR / "terrain_index.geojson"
    merged.to_file(out_path, driver="GeoJSON")
    log.info("Saved terrain_index.geojson — %d polygons across %d zones",
             len(merged), len(all_gdfs))

    if failed:
        log.warning("Failed zones: %s", failed)


def check_cache() -> None:
    features = load_zone_boundaries()
    cached = done = empty = 0
    for f in features:
        zid = f["properties"]["zone_id"]
        p = CACHE_DIR / f"zone_{zid}.gpkg"
        if p.exists():
            cached += 1
            import geopandas as gpd
            gdf = gpd.read_file(p)
            if gdf.empty:
                empty += 1
            else:
                done += 1
    missing = len(features) - cached
    print(f"\nCache status ({len(features)} zones total):")
    print(f"  Processed with terrain : {done}")
    print(f"  Processed, no avi terrain: {empty}")
    print(f"  Not yet processed      : {missing}")


def main():
    parser = argparse.ArgumentParser(description="Build terrain index for NAC zones")
    parser.add_argument("--zone",    type=int, nargs="+", help="Process specific zone ID(s)")
    parser.add_argument("--center",  type=str, nargs="+", help="Process one or more centers (e.g. BTAC UAC CAIC)")
    parser.add_argument("--exclude", type=str, nargs="+", help="Exclude centers (e.g. CNFAIC HPAC VAC)")
    parser.add_argument("--all",     action="store_true", help="Process all zones (skips cached)")
    parser.add_argument("--check",   action="store_true", help="Report cache status")
    args = parser.parse_args()

    if args.check:
        check_cache()
        return

    run(zone_ids=args.zone, center_filter=args.center, exclude_centers=args.exclude)


if __name__ == "__main__":
    main()
