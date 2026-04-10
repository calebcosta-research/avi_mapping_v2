"""
build_raster_tiles.py
----------------------
Converts the forecast + terrain data into XYZ PNG raster tiles for Mapbox.

Pipeline:
  1. For each zone, load the cached terrain rasters (slope mask, aspect_class, elev_band)
  2. Paint each surviving pixel with an RGBA color based on today's forecast:
       - Color  = danger level (or problem type, set via --mode)
       - Alpha  = 180 if pixel has an active avalanche problem
       - Alpha  = 60  if pixel is avi terrain but no active problem
       - Alpha  = 0   if pixel is not avi terrain
  3. Reproject painted raster to EPSG:3857 (Web Mercator) for tiling
  4. Mosaic all zones into one VRT
  5. Tile the mosaic into XYZ PNG tiles (zoom 6–13)
  6. Write to web/tiles/forecast/

The resulting tiles are served as a Mapbox raster source — perfectly fluid,
no gaps or overlaps, every pixel accounted for.

Usage:
    python build_raster_tiles.py                   # all cached zones, danger mode
    python build_raster_tiles.py --mode problem    # color by problem type
    python build_raster_tiles.py --mode likelihood
    python build_raster_tiles.py --center BTAC     # single center (for testing)
    python build_raster_tiles.py --zoom 6-12       # custom zoom range
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
OUT_DIR     = SCRIPTS_DIR.parent / "output"
CACHE_DIR   = OUT_DIR / "terrain_cache"
WEB_DIR     = SCRIPTS_DIR.parent.parent / "web"
TILES_DIR   = WEB_DIR / "tiles" / "forecast"

sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Color palettes (RGBA tuples 0-255)
# ---------------------------------------------------------------------------

DANGER_RGBA = {
    0: (136, 136, 136, 0),     # No rating — transparent
    1: (93,  184,  92, 180),   # Low       — green
    2: (255, 242,   0, 180),   # Moderate  — yellow
    3: (245, 166,  35, 200),   # Considerable — orange
    4: (208,   2,  27, 220),   # High      — red
    5: (20,   20,  20, 240),   # Extreme   — black
}

PROBLEM_RGBA = {
    'Wind Slab':            (79,  195, 247, 200),
    'Storm Slab':           (121, 134, 203, 200),
    'Wet Slab':             (240,  98, 146, 200),
    'Persistent Slab':      (255, 138, 101, 200),
    'Deep Persistent Slab': (161, 136, 127, 200),
    'Cornice':              (128, 203, 196, 200),
    'Glide Avalanche':      (255, 241, 118, 200),
    'Loose Wet':            (206, 147, 216, 200),
    'Loose Dry':            (176, 190, 197, 200),
    'default':              (100, 100, 100,  60),
}

LIKELIHOOD_RGBA = {
    'unlikely':       (93,  184,  92, 160),
    'possible':       (255, 242,   0, 180),
    'likely':         (245, 166,  35, 200),
    'very likely':    (208,   2,  27, 220),
    'almost certain': (80,    0,   0, 240),
    'default':        (100, 100, 100,  60),
}

NO_PROBLEM_ALPHA = 40   # faint overlay for avi terrain with no active problem
NOT_AVI_ALPHA    = 0    # fully transparent


# ---------------------------------------------------------------------------
# Forecast lookup (zone_id → aspect+elev → color)
# ---------------------------------------------------------------------------

def load_forecast_index(center_filter: Optional[list[str]] = None) -> dict:
    """
    Build a lookup: zone_id → list of {aspect_class, elev_band, rgba, has_problem}
    from forecast_layer.geojson.
    """
    layer_path = OUT_DIR / "forecast_layer.geojson"
    if not layer_path.exists():
        raise FileNotFoundError(
            "forecast_layer.geojson not found — run forecast_projection.py first"
        )

    with open(layer_path) as f:
        geo = json.load(f)

    index = {}
    for feat in geo["features"]:
        p = feat["properties"]
        if center_filter and p.get("center_id") not in center_filter:
            continue
        zid = p["zone_id"]
        if zid not in index:
            index[zid] = {}
        label = p["label"]  # aspect_class * 10 + elev_band
        index[zid][label] = p

    log.info("Forecast index loaded: %d zones", len(index))
    return index


def props_to_rgba(props: dict, mode: str) -> tuple[int, int, int, int]:
    """Convert forecast properties to an RGBA color tuple."""
    has_problem = props.get("has_problem", False)

    if not has_problem:
        danger = props.get("danger_level", 0)
        r, g, b, _ = DANGER_RGBA.get(danger, DANGER_RGBA[0])
        return (r, g, b, NO_PROBLEM_ALPHA)

    if mode == "danger":
        danger = props.get("danger_level", 0)
        return DANGER_RGBA.get(danger, DANGER_RGBA[0])

    if mode == "problem":
        pt = props.get("primary_problem") or "default"
        return PROBLEM_RGBA.get(pt, PROBLEM_RGBA["default"])

    if mode == "likelihood":
        lh = (props.get("primary_likelihood") or "").lower()
        return LIKELIHOOD_RGBA.get(lh, LIKELIHOOD_RGBA["default"])

    return DANGER_RGBA[0]


# ---------------------------------------------------------------------------
# Per-zone raster painting
# ---------------------------------------------------------------------------

def paint_zone(zone_id: int, forecast_index: dict, mode: str) -> Optional[Path]:
    """
    Load cached terrain for zone_id, paint RGBA forecast colors,
    save as a WGS84 GeoTIFF. Returns path or None if no data.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    dem_raw = CACHE_DIR / f"dem_raw_{zone_id}.tif"
    if not dem_raw.exists():
        return None

    zone_forecasts = forecast_index.get(zone_id, {})
    if not zone_forecasts:
        return None

    out_path = CACHE_DIR / f"painted_{zone_id}.tif"
    if out_path.exists():
        return out_path

    # We need to rebuild the terrain masks to know which pixels are avi terrain
    # and what their aspect/elev labels are. Re-run the filtering logic on the
    # cached DEM (fast — no download).
    from terrain_index import (
        reproject_to_utm, compute_slope_aspect, classify_aspect,
        classify_elev_band, utm_epsg,
        ELEV_MIN_M, SLOPE_MIN_DEG, SLOPE_MAX_DEG,
        ELEV_TREELINE_MIN, ELEV_ALPINE_MIN,
    )

    with rasterio.open(dem_raw) as src:
        centroid_lon = (src.bounds.left + src.bounds.right) / 2
        centroid_lat = (src.bounds.bottom + src.bounds.top) / 2

    epsg = utm_epsg(centroid_lon, centroid_lat)
    elevation, profile = reproject_to_utm(dem_raw, epsg)
    cell_size = abs(profile["transform"].a)

    slope, aspect = compute_slope_aspect(elevation, cell_size)

    mask = (
        (elevation > ELEV_MIN_M) &
        (slope >= SLOPE_MIN_DEG) &
        (slope <= SLOPE_MAX_DEG) &
        (elevation != -9999)
    )

    if not mask.any():
        return None

    aspect_class = classify_aspect(aspect)

    valid_elev = elevation[(elevation > 0) & (elevation != -9999)]
    elev_p75 = float(np.percentile(valid_elev, 75)) if valid_elev.size else ELEV_TREELINE_MIN
    elev_p90 = float(np.percentile(valid_elev, 90)) if valid_elev.size else ELEV_ALPINE_MIN
    treeline_m = max(ELEV_TREELINE_MIN, elev_p75)
    alpine_m   = max(ELEV_ALPINE_MIN,   elev_p90)
    elev_band  = classify_elev_band(elevation, treeline_m, alpine_m)

    # Build label array: aspect_class * 10 + elev_band (same encoding as forecast index)
    label = np.where(mask, aspect_class.astype(np.int16) * 10 + elev_band, -1).astype(np.int16)

    # Paint RGBA
    h, w = label.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)

    for lbl, props in zone_forecasts.items():
        px_mask = label == int(lbl)
        if not px_mask.any():
            continue
        r, g, b, a = props_to_rgba(props, mode)
        rgba[0, px_mask] = r
        rgba[1, px_mask] = g
        rgba[2, px_mask] = b
        rgba[3, px_mask] = a

    # Any avi terrain pixel not covered by a forecast cell gets a faint gray
    uncovered = mask & (rgba[3] == 0)
    rgba[3, uncovered] = 20

    # Reproject RGBA to WGS84 for mosaicking
    from rasterio.transform import Affine
    transform_src = profile["transform"]
    crs_src = profile["crs"]

    wgs84_transform, wgs84_w, wgs84_h = calculate_default_transform(
        crs_src, "EPSG:4326", w, h,
        left=transform_src.c,
        bottom=transform_src.f + transform_src.e * h,
        right=transform_src.c + transform_src.a * w,
        top=transform_src.f,
    )

    rgba_wgs84 = np.zeros((4, wgs84_h, wgs84_w), dtype=np.uint8)

    wgs84_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": wgs84_w,
        "height": wgs84_h,
        "count": 4,
        "crs": "EPSG:4326",
        "transform": wgs84_transform,
        "compress": "deflate",
        "photometric": "RGBA",
    }

    for band_i in range(4):
        reproject(
            source=rgba[band_i],
            destination=rgba_wgs84[band_i],
            src_transform=transform_src,
            src_crs=crs_src,
            dst_transform=wgs84_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )

    with rasterio.open(out_path, "w", **wgs84_profile) as dst:
        dst.write(rgba_wgs84)

    return out_path


# ---------------------------------------------------------------------------
# XYZ tile generation
# ---------------------------------------------------------------------------

def lonlat_to_tile(lon, lat, zoom):
    """Convert lon/lat to XYZ tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat_bounds(x, y, zoom):
    """Return (west, south, east, north) for tile x,y,z."""
    n = 2 ** zoom
    west  = x / n * 360.0 - 180.0
    east  = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def generate_tiles(painted_paths: list[Path], zoom_min: int, zoom_max: int,
                   mode: str) -> None:
    """
    Mosaic all painted zone rasters and generate XYZ PNG tiles.
    Uses rasterio windowed reads for memory efficiency.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from PIL import Image

    if not painted_paths:
        log.warning("No painted rasters to tile.")
        return

    log.info("Mosaicking %d zone rasters...", len(painted_paths))

    # Open all sources, merge
    sources = [rasterio.open(p) for p in painted_paths]
    mosaic, mosaic_transform = merge(sources, method="first")
    mosaic_crs = sources[0].crs
    for s in sources:
        s.close()

    log.info("Mosaic: %dx%d pixels", mosaic.shape[2], mosaic.shape[1])

    # Reproject to Web Mercator (EPSG:3857) for XYZ tiling
    log.info("Reprojecting to Web Mercator...")
    wm_transform, wm_w, wm_h = calculate_default_transform(
        mosaic_crs, "EPSG:3857",
        mosaic.shape[2], mosaic.shape[1],
        left=mosaic_transform.c,
        bottom=mosaic_transform.f + mosaic_transform.e * mosaic.shape[1],
        right=mosaic_transform.c + mosaic_transform.a * mosaic.shape[2],
        top=mosaic_transform.f,
    )

    mosaic_3857 = np.zeros((4, wm_h, wm_w), dtype=np.uint8)
    for band_i in range(4):
        reproject(
            source=mosaic[band_i],
            destination=mosaic_3857[band_i],
            src_transform=mosaic_transform,
            src_crs=mosaic_crs,
            dst_transform=wm_transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest,
        )

    # Calculate overall bounds in lon/lat for tile range
    bounds_left   = mosaic_transform.c
    bounds_top    = mosaic_transform.f
    bounds_right  = mosaic_transform.c + mosaic_transform.a * mosaic.shape[2]
    bounds_bottom = mosaic_transform.f + mosaic_transform.e * mosaic.shape[1]

    TILE_SIZE = 256
    total_tiles = 0

    for zoom in range(zoom_min, zoom_max + 1):
        tx_min, ty_min = lonlat_to_tile(bounds_left,  bounds_top,    zoom)
        tx_max, ty_max = lonlat_to_tile(bounds_right, bounds_bottom, zoom)
        tx_min, tx_max = min(tx_min, tx_max), max(tx_min, tx_max)
        ty_min, ty_max = min(ty_min, ty_max), max(ty_min, ty_max)

        n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
        log.info("  Zoom %d: %dx%d = %d tiles", zoom,
                 tx_max - tx_min + 1, ty_max - ty_min + 1, n_tiles)

        tile_dir = TILES_DIR / mode / str(zoom)
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Downsample mosaic to appropriate resolution for this zoom
        # At zoom Z, one tile covers 256px. World = 2^Z tiles.
        # meters_per_pixel at equator ≈ 156543 / 2^Z
        # We sample the mosaic at tile resolution using rasterio
        n = 2 ** zoom
        world_size_m = 20037508.342789244 * 2  # Web Mercator extent

        for tx in range(tx_min, tx_max + 1):
            x_dir = tile_dir / str(tx)
            x_dir.mkdir(exist_ok=True)
            for ty in range(ty_min, ty_max + 1):
                tile_path = x_dir / f"{ty}.png"
                if tile_path.exists():
                    continue

                w_lon, s_lat, e_lon, n_lat = tile_to_lonlat_bounds(tx, ty, zoom)

                # Convert tile bounds to Web Mercator pixel coords in mosaic
                def lon_to_px(lon):
                    return int((lon - bounds_left) / (mosaic_transform.a) *
                               (wm_w / mosaic.shape[2]))
                def lat_to_py(lat):
                    import math
                    # Convert lat to Web Mercator y pixel
                    lat_r = math.radians(lat)
                    merc_y = math.log(math.tan(math.pi/4 + lat_r/2))
                    # mosaic bounds in mercator
                    top_r = math.radians(bounds_top)
                    bot_r = math.radians(bounds_bottom)
                    top_merc = math.log(math.tan(math.pi/4 + top_r/2))
                    bot_merc = math.log(math.tan(math.pi/4 + bot_r/2))
                    frac = (top_merc - merc_y) / (top_merc - bot_merc)
                    return int(frac * wm_h)

                px0 = max(0, lon_to_px(w_lon))
                px1 = min(wm_w, lon_to_px(e_lon))
                py0 = max(0, lat_to_py(n_lat))
                py1 = min(wm_h, lat_to_py(s_lat))

                if px1 <= px0 or py1 <= py0:
                    continue

                chunk = mosaic_3857[:, py0:py1, px0:px1]
                if chunk.max() == 0:
                    continue   # fully transparent, skip

                # Resize to 256×256
                img_arr = np.moveaxis(chunk, 0, -1)  # HWC
                img = Image.fromarray(img_arr, mode="RGBA")
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
                img.save(tile_path, "PNG", optimize=True)
                total_tiles += 1

    log.info("Generated %d tiles -> %s", total_tiles, TILES_DIR / mode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(center_filter: Optional[list[str]] = None,
        mode: str = "danger",
        zoom_min: int = 6,
        zoom_max: int = 13) -> None:

    TILES_DIR.mkdir(parents=True, exist_ok=True)

    # Load forecast data
    forecast_index = load_forecast_index(center_filter)

    # Paint each zone
    painted = []
    zone_ids = list(forecast_index.keys())
    log.info("Painting %d zones...", len(zone_ids))

    for i, zid in enumerate(zone_ids, 1):
        log.info("  [%d/%d] zone %s", i, len(zone_ids), zid)
        try:
            # Clear old painted cache so it re-paints with new forecast/mode
            old = CACHE_DIR / f"painted_{zid}.tif"
            if old.exists():
                old.unlink()
            path = paint_zone(zid, forecast_index, mode)
            if path:
                painted.append(path)
        except Exception as e:
            log.warning("  Zone %s paint failed: %s", zid, e)

    log.info("Painted %d zones successfully", len(painted))

    # Generate tiles
    generate_tiles(painted, zoom_min, zoom_max, mode)

    # Write tile metadata for the web app
    meta = {
        "mode":     mode,
        "zoom_min": zoom_min,
        "zoom_max": zoom_max,
        "url":      f"tiles/forecast/{mode}/{{z}}/{{x}}/{{y}}.png",
    }
    (TILES_DIR / mode / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Tile metadata written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build raster forecast tiles")
    parser.add_argument("--center", nargs="+", help="Limit to center IDs")
    parser.add_argument("--mode", choices=["danger", "problem", "likelihood"],
                        default="danger", help="Color mode")
    parser.add_argument("--zoom", default="6-13",
                        help="Zoom range e.g. 6-13 (default)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore painted raster cache")
    args = parser.parse_args()

    zmin, zmax = map(int, args.zoom.split("-"))
    run(center_filter=args.center, mode=args.mode,
        zoom_min=zmin, zoom_max=zmax)
