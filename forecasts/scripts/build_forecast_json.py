"""
build_forecast_json.py
-----------------------
Generates the daily forecast lookup JSON (~50KB).
This is the ONLY file that changes daily. Tiles never rebuild.

Output:
  web/data/forecast.json

Format:
  {
    "date": "2026-03-24",
    "generated_at": "2026-03-24T12:00:00Z",
    "zones": {
      "42": {                          <- zone_index (matches identity tile R channel)
        "zone_id": 2856,
        "center_id": "BTAC",
        "zone_name": "Salt River...",
        "11": {                        <- cell_id (matches identity tile G channel)
          "danger": 2,
          "danger_label": "Moderate",
          "danger_color": "#fff200",
          "problems": [
            {
              "type": "Persistent Slab",
              "likelihood": "unlikely",
              "size_min": 1.5,
              "size_max": 2.0
            }
          ]
        }
      }
    }
  }

Usage:
    python build_forecast_json.py        # fetch live forecasts
    python build_forecast_json.py --dry  # print output without saving
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from forecast_parser import ForecastClient, AvalancheForecast

OUT_DIR  = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent.parent / "web" / "data"

DANGER_COLORS = {
    1: "#5db85c",
    2: "#fff200",
    3: "#f5a623",
    4: "#d0021b",
    5: "#000000",
    0: "#cccccc",
}
DANGER_LABELS = {
    1: "Low", 2: "Moderate", 3: "Considerable",
    4: "High", 5: "Extreme", 0: "No Rating",
}

ASPECT_CLASSES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ELEV_BANDS     = {1: "below_treeline", 2: "treeline", 3: "alpine"}


def elev_band_name(s: str) -> str:
    s = s.lower().replace(" ", "_")
    if s in ("alpine",): return "alpine"
    if s in ("treeline", "near_treeline"): return "treeline"
    return "below_treeline"


def build_forecast_json(dry_run: bool = False) -> dict:
    # Load zone lookup to get zone_index mapping
    lookup_path = DATA_DIR / "zone_lookup.json"
    if not lookup_path.exists():
        log.error("zone_lookup.json not found — run build_identity_tiles.py first")
        sys.exit(1)

    with open(lookup_path) as f:
        zone_lookup = json.load(f)  # zone_id (str) → {index, center_id, ...}

    # Build reverse: index (str) → zone_id
    index_to_zid = {str(v["index"]): k for k, v in zone_lookup.items()}

    # Load terrain index to get zone/center mapping
    terrain_path = OUT_DIR / "terrain_index.geojson"
    with open(terrain_path) as f:
        terrain = json.load(f)

    # Collect unique (center_id, zone_id) pairs
    zone_pairs = {}
    for feat in terrain["features"]:
        p = feat["properties"]
        zid = str(p["zone_id"])
        if zid not in zone_pairs:
            zone_pairs[zid] = p["center_id"]

    log.info("Fetching forecasts for %d zones...", len(zone_pairs))

    client  = ForecastClient()
    output  = {
        "date":         None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zones":        {},
    }
    dates_seen = set()

    for zone_id_str, center_id in zone_pairs.items():
        zone_id = int(zone_id_str)
        if zone_id_str not in zone_lookup:
            continue
        zone_meta  = zone_lookup[zone_id_str]
        zone_index = str(zone_meta["index"])

        try:
            fc: AvalancheForecast = client.fetch_zone(center_id, zone_id)
        except Exception as e:
            log.warning("  [%s] zone %d failed: %s", center_id, zone_id, e)
            continue

        if fc.valid_for_date:
            dates_seen.add(fc.valid_for_date)

        # Build per-cell danger + problems
        cells = {}

        # Danger by elevation band → applies to all aspects in that band
        danger_by_band = {}
        for dr in fc.danger_ratings:
            band = elev_band_name(dr.elevation_band)
            danger_by_band[band] = int(dr.danger_level or 0)

        # For each (aspect_class, elev_band) cell, assign danger + problems
        for ac_idx, aspect in enumerate(ASPECT_CLASSES):
            for eb_idx, eb_name in ELEV_BANDS.items():
                cell_id = ac_idx * 10 + eb_idx

                danger = danger_by_band.get(eb_name, 0)

                # Find problems targeting this cell
                problems = []
                for prob in fc.avalanche_problems:
                    aspects_hit = [a.upper() for a in (prob.aspects or [])]
                    bands_hit   = [elev_band_name(b) for b in (prob.elevation_bands or [])]
                    if aspect.upper() in aspects_hit and eb_name in bands_hit:
                        problems.append({
                            "type":       prob.problem_type,
                            "likelihood": prob.likelihood or "unknown",
                            "size_min":   prob.size_min,
                            "size_max":   prob.size_max,
                        })

                if danger > 0 or problems:
                    cells[str(cell_id)] = {
                        "danger":       danger,
                        "danger_label": DANGER_LABELS.get(danger, "No Rating"),
                        "danger_color": DANGER_COLORS.get(danger, "#cccccc"),
                        "problems":     problems,
                    }

        output["zones"][zone_index] = {
            "zone_id":   zone_id,
            "center_id": center_id,
            "zone_name": zone_meta["zone_name"],
            **cells,
        }
        log.info("  [%s] %s — %d cells with data",
                 center_id, zone_meta["zone_name"], len(cells))

    output["date"] = sorted(dates_seen)[0] if dates_seen else None
    log.info("Forecast date: %s | %d zones loaded", output["date"], len(output["zones"]))

    if dry_run:
        print(json.dumps(output, indent=2)[:2000], "\n... (truncated)")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / "forecast.json"
        with open(out_path, "w") as f:
            json.dump(output, f)
        size_kb = out_path.stat().st_size / 1024
        log.info("Saved forecast.json — %.1f KB", size_kb)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Print without saving")
    args = parser.parse_args()
    build_forecast_json(dry_run=args.dry)
