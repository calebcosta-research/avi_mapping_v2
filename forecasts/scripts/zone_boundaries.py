"""
zone_boundaries.py
------------------
Fetches all active NAC zone boundaries and saves them as GeoJSON.

The NAC map-layer endpoint returns a FeatureCollection of MultiPolygons —
one per forecast zone. This script grabs that data, cleans up the properties,
and writes it to forecasts/output/zone_boundaries.geojson.

The geometry is in WGS84 (EPSG:4326), which is what Mapbox/Cesium expect.

Usage:
    python zone_boundaries.py                  # fetch and save
    python zone_boundaries.py --summary        # print zone counts by center
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from forecast_parser import ForecastClient, MAP_LAYER_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "output"


def fetch_zone_boundaries() -> dict:
    """
    Fetch the global map-layer FeatureCollection from NAC.
    Returns a cleaned GeoJSON FeatureCollection with:
      - geometry: original MultiPolygon in WGS84
      - properties: center_id, center_name, zone_id, zone_name,
                    state, danger_level, color, off_season
    """
    client = ForecastClient()
    log.info("Fetching zone boundaries from NAC map-layer...")
    raw = client._get_json(MAP_LAYER_URL)

    features = raw.get("features", [])
    log.info("Found %d zones", len(features))

    cleaned = []
    for f in features:
        props = f.get("properties", {})
        geometry = f.get("geometry")
        zone_id = f.get("id")

        if not geometry or zone_id is None:
            continue

        cleaned.append({
            "type": "Feature",
            "id": zone_id,
            "geometry": geometry,
            "properties": {
                "zone_id":      zone_id,
                "zone_name":    props.get("name", ""),
                "center_id":    props.get("center_id", "").upper(),
                "center_name":  props.get("center", ""),
                "state":        props.get("state", ""),
                "off_season":   bool(props.get("off_season")),
                "danger_level": props.get("danger_level"),
                "color":        props.get("color", "#cccccc"),
                "link":         props.get("link", ""),
            },
        })

    return {
        "type": "FeatureCollection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(cleaned),
        "features": cleaned,
    }


def save_boundaries(geojson: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    log.info("Saved %d zones -> %s", geojson["feature_count"], path)


def print_summary(geojson: dict) -> None:
    from collections import Counter
    centers = Counter(
        f["properties"]["center_id"]
        for f in geojson["features"]
    )
    print(f"\n{'Center':<12} {'Zones':>5}")
    print("-" * 20)
    for cid, count in sorted(centers.items()):
        print(f"{cid:<12} {count:>5}")
    off = sum(1 for f in geojson["features"] if f["properties"]["off_season"])
    print(f"\nTotal: {geojson['feature_count']} zones across {len(centers)} centers")
    print(f"Off season: {off}  |  Active: {geojson['feature_count'] - off}")


def main():
    parser = argparse.ArgumentParser(description="Fetch NAC zone boundaries")
    parser.add_argument("--summary", action="store_true",
                        help="Print zone counts by center")
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "zone_boundaries.geojson",
                        help="Output path (default: forecasts/output/zone_boundaries.geojson)")
    args = parser.parse_args()

    geojson = fetch_zone_boundaries()
    save_boundaries(geojson, args.out)

    if args.summary:
        print_summary(geojson)


if __name__ == "__main__":
    main()
