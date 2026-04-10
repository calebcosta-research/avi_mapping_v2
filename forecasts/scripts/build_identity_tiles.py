"""
build_identity_tiles.py
-----------------------
Generates static identity-encoded raster tiles from the terrain index.

Each pixel encodes permanent physical characteristics:
  R = zone_index       (0-99, sequential index into zone_lookup.json)
  G = cell_id          (aspect_class * 10 + elev_band, range 1-73)
  B = slope_encoded    (slope degrees 30-45 mapped to 0-255)
  A = 255 if valid avi terrain, 0 if not

These tiles NEVER need to be rebuilt unless the terrain algorithm changes.
Forecast data is applied dynamically in the browser via forecast.json.

Output:
  web/tiles/identity/{z}/{x}/{y}.png   — identity-encoded tiles
  web/data/zone_lookup.json            — maps zone_index → zone metadata

Usage:
    python build_identity_tiles.py              # build all zones
    python build_identity_tiles.py --zoom 6-13  # specific zoom range
    python build_identity_tiles.py --center BTAC
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

OUT_DIR    = Path(__file__).parent.parent / "output"
WEB_DIR    = Path(__file__).parent.parent.parent / "web"
TILES_DIR  = WEB_DIR / "tiles" / "identity"
DATA_DIR   = WEB_DIR / "data"

ZOOM_MIN   = 6
ZOOM_MAX   = 13

# Slope encoding: 30-45° → 0-255
SLOPE_MIN  = 30.0
SLOPE_MAX  = 45.0


# ---------------------------------------------------------------------------
# Zone lookup — maps zone_id to a sequential 0-based index
# ---------------------------------------------------------------------------

def build_zone_lookup(features: list[dict]) -> dict:
    """
    Build a bidirectional lookup between zone_id and zone_index (0-99).
    Returns dict: zone_id (str) → {index, center_id, zone_name, state}
    """
    seen = {}
    idx = 0
    for feat in features:
        p = feat["properties"]
        zid = str(p["zone_id"])
        if zid not in seen:
            seen[zid] = {
                "index":      idx,
                "center_id":  p["center_id"],
                "zone_name":  p["zone_name"],
                "state":      p.get("state", ""),
            }
            idx += 1
    return seen


# ---------------------------------------------------------------------------
# Tile math
# ---------------------------------------------------------------------------

def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS84 for a tile."""
    n = 2 ** z
    west  = x / n * 360.0 - 180.0
    east  = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def lon_to_tile_x(lon: float, z: int) -> int:
    return int((lon + 180.0) / 360.0 * (2 ** z))


def lat_to_tile_y(lat: float, z: int) -> int:
    lat_r = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * (2 ** z))


def tiles_for_bounds(west: float, south: float, east: float, north: float,
                     z: int) -> list[tuple[int, int]]:
    """Return list of (x, y) tile coords covering the bbox at zoom z."""
    x0 = lon_to_tile_x(west,  z)
    x1 = lon_to_tile_x(east,  z)
    y0 = lat_to_tile_y(north, z)   # north = smaller y
    y1 = lat_to_tile_y(south, z)   # south = larger y
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


# ---------------------------------------------------------------------------
# Rasterize one tile
# ---------------------------------------------------------------------------

def rasterize_tile(
    z: int, x: int, y: int,
    features: list[dict],
    zone_lookup: dict,
    tile_size: int = 256,
) -> Optional[np.ndarray]:
    """
    Rasterize terrain features into a 256×256 RGBA identity tile.
    Returns None if no terrain falls within this tile.
    """
    try:
        from shapely.geometry import shape, box
        from shapely.strtree import STRtree
    except ImportError:
        log.error("shapely not installed")
        raise

    west, south, east, north = tile_bounds(z, x, y)
    tile_box = box(west, south, east, north)

    # Filter features that intersect this tile
    candidates = [f for f in features
                  if shape(f["geometry"]).intersects(tile_box)]
    if not candidates:
        return None

    # Build RGBA canvas — all zeros (transparent)
    canvas = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)

    # Pixel size in degrees
    px_w = (east  - west)  / tile_size
    px_h = (north - south) / tile_size

    for feat in candidates:
        props   = feat["properties"]
        zone_id = str(props["zone_id"])
        if zone_id not in zone_lookup:
            continue

        zone_idx  = zone_lookup[zone_id]["index"]
        cell_id   = int(props.get("label", 0))
        # Slope: if stored, decode; otherwise use midpoint of range
        slope_val = float(props.get("mean_slope", 37.5))
        slope_enc = int(np.clip((slope_val - SLOPE_MIN) / (SLOPE_MAX - SLOPE_MIN) * 255, 0, 255))

        geom = shape(feat["geometry"])
        clipped = geom.intersection(tile_box)
        if clipped.is_empty:
            continue

        # Rasterize: iterate pixels, check containment
        # For performance, only check pixels in the feature's bbox
        minx, miny, maxx, maxy = clipped.bounds
        col0 = max(0, int((minx - west)  / px_w))
        col1 = min(tile_size, int((maxx - west)  / px_w) + 1)
        row0 = max(0, int((north - maxy) / px_h))
        row1 = min(tile_size, int((north - miny) / px_h) + 1)

        for row in range(row0, row1):
            for col in range(col0, col1):
                px_lon = west  + (col + 0.5) * px_w
                px_lat = north - (row + 0.5) * px_h
                from shapely.geometry import Point
                if clipped.contains(Point(px_lon, px_lat)):
                    canvas[row, col, 0] = zone_idx    # R: zone index
                    canvas[row, col, 1] = cell_id     # G: cell_id
                    canvas[row, col, 2] = slope_enc   # B: slope
                    canvas[row, col, 3] = 255         # A: valid terrain

    if canvas[:, :, 3].max() == 0:
        return None

    return canvas


def save_tile(canvas: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGBA").save(path, "PNG", optimize=True)


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def run(zoom_min: int = ZOOM_MIN,
        zoom_max: int = ZOOM_MAX,
        center_filter: Optional[str] = None) -> None:
    terrain_path = OUT_DIR / "terrain_index.geojson"
    if not terrain_path.exists():
        log.error("terrain_index.geojson not found — run terrain_index.py first")
        sys.exit(1)

    with open(terrain_path) as f:
        terrain = json.load(f)

    features = terrain["features"]
    if center_filter:
        features = [f for f in features
                    if f["properties"]["center_id"] == center_filter.upper()]

    if not features:
        log.error("No features found")
        sys.exit(1)

    # Build zone lookup and save it
    zone_lookup = build_zone_lookup(features)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lookup_path = DATA_DIR / "zone_lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(zone_lookup, f, indent=2)
    log.info("Saved zone_lookup.json — %d zones", len(zone_lookup))

    # Get overall bounds
    all_lons = []
    all_lats = []
    for feat in features:
        from shapely.geometry import shape
        bounds = shape(feat["geometry"]).bounds
        all_lons += [bounds[0], bounds[2]]
        all_lats += [bounds[1], bounds[3]]
    bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
    log.info("Coverage bbox: W=%.2f S=%.2f E=%.2f N=%.2f", *bbox)

    total_tiles = 0
    written     = 0

    for z in range(zoom_min, zoom_max + 1):
        candidate_tiles = tiles_for_bounds(*bbox, z)
        log.info("Zoom %d: %d candidate tiles", z, len(candidate_tiles))

        for tx, ty in candidate_tiles:
            total_tiles += 1
            out_path = TILES_DIR / str(z) / str(tx) / f"{ty}.png"
            if out_path.exists():
                continue

            canvas = rasterize_tile(z, tx, ty, features, zone_lookup)
            if canvas is not None:
                save_tile(canvas, out_path)
                written += 1

        log.info("  Zoom %d complete — %d tiles written", z, written)

    log.info("Done — %d identity tiles written to %s", written, TILES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoom",   default="6-13",
                        help="Zoom range, e.g. '6-13' or '8'")
    parser.add_argument("--center", type=str, help="Filter to one center")
    args = parser.parse_args()

    if "-" in args.zoom:
        zmin, zmax = map(int, args.zoom.split("-"))
    else:
        zmin = zmax = int(args.zoom)

    run(zoom_min=zmin, zoom_max=zmax, center_filter=args.center)
