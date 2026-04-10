"""
identify_gap_zones.py
---------------------
Algorithmically identifies candidate gap zones — terrain between existing
avalanche centers that has no dedicated forecast coverage.

Strategy
--------
1. Load NAC zone boundaries (from zone_boundaries.geojson or fresh API fetch).
2. Group zones by center, compute union polygon per center → "covered" areas.
3. Compute centroid of each center's coverage polygon.
4. Run a Voronoi diagram over center centroids, clipped to western North America.
   Each Voronoi cell is the natural "sphere of influence" for that center.
5. Gap zones = terrain inside a Voronoi cell but OUTSIDE the center's actual
   forecast polygon. These are places where a center is the closest authority
   but has no formal coverage.
6. Merge adjacent small gaps, filter by area and elevation band overlap.
7. For each gap, assign the 2 nearest anchor centers and compute distance-based
   weights (inverse distance, normalized).
8. Output:
   - gap_candidates.geojson   (GeoJSON FeatureCollection of candidate gaps)
   - gap_candidates_summary   (printed table of candidates with anchor pairs)

Usage
-----
    python identify_gap_zones.py                         # uses cached boundaries
    python identify_gap_zones.py --fetch                 # fetch fresh from API
    python identify_gap_zones.py --min-area-km2 500      # filter small candidates
    python identify_gap_zones.py --out candidates.geojson

Dependencies
------------
    pip install shapely scipy numpy requests
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import (
    MultiPolygon, Point, Polygon, shape, mapping
)
from shapely.ops import unary_union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).parent
INTERP_DIR    = SCRIPTS_DIR.parent
FORECASTS_DIR = INTERP_DIR.parent / "forecasts"
BOUNDARIES_PATH = FORECASTS_DIR / "output" / "zone_boundaries.geojson"
OUTPUT_PATH     = INTERP_DIR / "zones" / "gap_candidates.geojson"

# ── Western North America bounding box ────────────────────────────────────────
# Covers all active NAC centers + margin.
WNAM_BBOX = Polygon([
    (-135, 32), (-95, 32), (-95, 65), (-135, 65), (-135, 32)
])

# Minimum area threshold in km² — filters trivial gaps (slivers, islands, etc.)
DEFAULT_MIN_AREA_KM2 = 300


# ── Geometry helpers ──────────────────────────────────────────────────────────

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two lon/lat points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def polygon_area_km2(poly: Polygon) -> float:
    """Rough area estimate in km² using equirectangular projection at centroid lat."""
    c = poly.centroid
    lat_rad = radians(c.y)
    km_per_deg_lon = 111.32 * cos(lat_rad)
    km_per_deg_lat = 111.32
    # Scale bounds
    minx, miny, maxx, maxy = poly.bounds
    return abs(maxx - minx) * km_per_deg_lon * abs(maxy - miny) * km_per_deg_lat


def voronoi_polygons(points: np.ndarray, clip: Polygon) -> list[Polygon]:
    """
    Build Voronoi regions for each input point, clipped to `clip` polygon.
    Returns list aligned with input points (same index = same region).
    Mirror points are added at the clip boundary to ensure all regions are finite.
    """
    n = len(points)
    # Add far mirror points to bound all Voronoi cells
    cx, cy = clip.centroid.x, clip.centroid.y
    far = 50.0   # degrees, well outside clip polygon
    mirrors = np.array([
        [cx - far, cy - far], [cx + far, cy - far],
        [cx - far, cy + far], [cx + far, cy + far],
        [cx,       cy - far], [cx,       cy + far],
        [cx - far, cy],       [cx + far, cy],
    ])
    all_pts = np.vstack([points, mirrors])
    vor = Voronoi(all_pts)

    regions = []
    for i in range(n):
        region_idx = vor.point_region[i]
        vertices   = vor.regions[region_idx]
        if -1 in vertices or len(vertices) == 0:
            # Infinite region — clip will handle it
            regions.append(clip)
            continue
        poly = Polygon(vor.vertices[vertices])
        clipped = poly.intersection(clip)
        regions.append(clipped if not clipped.is_empty else Polygon())

    return regions


# ── Core algorithm ────────────────────────────────────────────────────────────

def load_boundaries(path: Path, fetch: bool = False) -> dict:
    if fetch or not path.exists():
        log.info("Fetching fresh zone boundaries from NAC API...")
        sys.path.insert(0, str(FORECASTS_DIR / "scripts"))
        from zone_boundaries import fetch_zone_boundaries, save_boundaries
        data = fetch_zone_boundaries()
        save_boundaries(data, path)
        return data
    log.info("Loading cached zone boundaries from %s", path)
    with open(path) as f:
        return json.load(f)


def build_center_polygons(boundaries: dict) -> dict[str, Polygon]:
    """
    Groups zone features by center_id, merges geometries into one polygon per
    center. Returns {center_id: merged_polygon}.
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for feat in boundaries["features"]:
        cid  = feat["properties"]["center_id"]
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        groups[cid].append(geom)

    result = {}
    for cid, geoms in groups.items():
        merged = unary_union(geoms)
        if isinstance(merged, (Polygon, MultiPolygon)):
            result[cid] = merged
    log.info("Built coverage polygons for %d centers", len(result))
    return result


def identify_gaps(
    center_polygons: dict[str, Polygon],
    min_area_km2: float = DEFAULT_MIN_AREA_KM2,
) -> list[dict]:
    """
    Main gap detection algorithm.

    Returns list of gap dicts:
      {
        "geometry":     shapely Polygon,
        "area_km2":     float,
        "anchor_centers": [
            {"center_id": str, "distance_km": float, "weight": float}, ...
        ],
        "centroid": (lon, lat),
      }
    """
    # Step 1: Compute centroid per center
    center_centroids: dict[str, tuple[float, float]] = {}
    for cid, poly in center_polygons.items():
        c = poly.centroid
        center_centroids[cid] = (c.x, c.y)

    cids   = list(center_centroids.keys())
    pts    = np.array([center_centroids[c] for c in cids])

    # Step 2: Voronoi over centers, clipped to western NA
    log.info("Computing Voronoi diagram over %d center centroids...", len(cids))
    voronoi_cells = voronoi_polygons(pts, WNAM_BBOX)

    # Step 3: For each center, gap = Voronoi cell minus actual coverage polygon
    gaps = []
    covered_union = unary_union(list(center_polygons.values()))

    for i, cid in enumerate(cids):
        cell = voronoi_cells[i]
        if cell.is_empty:
            continue
        actual = center_polygons[cid]
        # Gap for this center = inside its Voronoi cell but outside its polygon
        gap = cell.difference(actual)
        # Also subtract ALL other centers' polygons (they have their own coverage)
        gap = gap.difference(covered_union)
        if gap.is_empty:
            continue
        # Decompose MultiPolygon into individual pieces
        pieces = list(gap.geoms) if isinstance(gap, MultiPolygon) else [gap]
        for piece in pieces:
            if not isinstance(piece, Polygon):
                continue
            area = polygon_area_km2(piece)
            if area < min_area_km2:
                continue
            c = piece.centroid
            gaps.append({
                "geometry":   piece,
                "area_km2":   area,
                "centroid":   (c.x, c.y),
                "voronoi_center": cid,
            })

    log.info("Found %d raw gap polygons (area >= %.0f km²)", len(gaps), min_area_km2)

    # Step 4: Assign 2 nearest anchor centers + inverse-distance weights
    for gap in gaps:
        lon, lat = gap["centroid"]
        dists = [
            (cid, haversine_km(lon, lat, cx, cy))
            for cid, (cx, cy) in center_centroids.items()
        ]
        dists.sort(key=lambda x: x[1])
        nearest = dists[:2]
        total   = sum(1.0 / d for _, d in nearest)
        gap["anchor_centers"] = [
            {
                "center_id":   cid,
                "distance_km": round(dist, 1),
                "weight":      round((1.0 / dist) / total, 3),
            }
            for cid, dist in nearest
        ]

    gaps.sort(key=lambda g: g["area_km2"], reverse=True)
    return gaps


def to_geojson(gaps: list[dict]) -> dict:
    features = []
    for i, g in enumerate(gaps):
        lon, lat = g["centroid"]
        features.append({
            "type": "Feature",
            "id":   i,
            "geometry": mapping(g["geometry"]),
            "properties": {
                "area_km2":       round(g["area_km2"], 1),
                "centroid_lon":   round(lon, 4),
                "centroid_lat":   round(lat, 4),
                "voronoi_center": g.get("voronoi_center", ""),
                "anchor_1":       g["anchor_centers"][0]["center_id"] if len(g["anchor_centers"]) > 0 else "",
                "anchor_1_dist":  g["anchor_centers"][0]["distance_km"] if len(g["anchor_centers"]) > 0 else None,
                "anchor_1_wt":    g["anchor_centers"][0]["weight"] if len(g["anchor_centers"]) > 0 else None,
                "anchor_2":       g["anchor_centers"][1]["center_id"] if len(g["anchor_centers"]) > 1 else "",
                "anchor_2_dist":  g["anchor_centers"][1]["distance_km"] if len(g["anchor_centers"]) > 1 else None,
                "anchor_2_wt":    g["anchor_centers"][1]["weight"] if len(g["anchor_centers"]) > 1 else None,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def print_summary(gaps: list[dict], top_n: int = 20) -> None:
    print(f"\n{'#':<4} {'Area km2':>10}  {'Anchor 1':>8}  {'Wt':>5}  {'Anchor 2':>8}  {'Wt':>5}  Centroid")
    print("-" * 80)
    for i, g in enumerate(gaps[:top_n]):
        lon, lat = g["centroid"]
        a1 = g["anchor_centers"][0] if len(g["anchor_centers"]) > 0 else {}
        a2 = g["anchor_centers"][1] if len(g["anchor_centers"]) > 1 else {}
        print(
            f"{i:<4} {g['area_km2']:>10.0f}  "
            f"{a1.get('center_id',''):>8}  {a1.get('weight',0):>5.2f}  "
            f"{a2.get('center_id',''):>8}  {a2.get('weight',0):>5.2f}  "
            f"({lon:.2f}, {lat:.2f})"
        )
    if len(gaps) > top_n:
        print(f"  ... and {len(gaps) - top_n} more")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Identify avalanche forecast gap zones")
    parser.add_argument("--fetch",          action="store_true",
                        help="Fetch fresh zone boundaries from NAC API")
    parser.add_argument("--min-area-km2",   type=float, default=DEFAULT_MIN_AREA_KM2,
                        help=f"Minimum gap area in km2 (default: {DEFAULT_MIN_AREA_KM2})")
    parser.add_argument("--out",            type=Path, default=OUTPUT_PATH,
                        help="Output GeoJSON path")
    parser.add_argument("--top",            type=int, default=25,
                        help="Rows to print in summary table")
    args = parser.parse_args()

    boundaries      = load_boundaries(BOUNDARIES_PATH, fetch=args.fetch)
    center_polygons = build_center_polygons(boundaries)
    gaps            = identify_gaps(center_polygons, min_area_km2=args.min_area_km2)

    geojson = to_geojson(gaps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(geojson, f)
    log.info("Wrote %d gap candidates -> %s", len(gaps), args.out)

    print_summary(gaps, top_n=args.top)


if __name__ == "__main__":
    main()
