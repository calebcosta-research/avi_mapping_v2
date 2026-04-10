"""
refresh_forecasts.py
---------------------
Daily forecast refresh — re-joins current forecasts to the existing terrain index.
Terrain is static (never re-downloaded). Only forecast data is updated.

Run this once per day (or on-demand) to keep web/data/forecast_layer.geojson current.

Usage:
    python refresh_forecasts.py                     # update all zones
    python refresh_forecasts.py --center BTAC UAC   # specific centers only
    python refresh_forecasts.py --check             # show last update time

Typical cron / Task Scheduler entry:
    0 6 * * * cd /path/to/repo && python forecasts/scripts/refresh_forecasts.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
OUT_DIR     = SCRIPTS_DIR.parent / "output"
WEB_DATA    = SCRIPTS_DIR.parent.parent / "web" / "data"

sys.path.insert(0, str(SCRIPTS_DIR))


def run(center_filter: list[str] | None = None) -> None:
    from forecast_projection import run as project_run

    terrain_path = OUT_DIR / "terrain_index.geojson"
    if not terrain_path.exists():
        log.error("terrain_index.geojson not found — run terrain_index.py first")
        sys.exit(1)

    log.info("Refreshing forecasts%s...",
             f" for {', '.join(center_filter)}" if center_filter else " (all centers)")

    project_run(center_filter=center_filter)

    # Copy to web/data for serving
    src = OUT_DIR / "forecast_layer.geojson"
    if WEB_DATA.exists():
        dst = WEB_DATA / "forecast_layer.geojson"
        shutil.copy2(src, dst)
        log.info("Copied to web/data/forecast_layer.geojson")
    else:
        log.warning("web/data/ not found — skipping copy (run from repo root?)")

    # Write a metadata file with last update time
    meta = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "centers": center_filter or "all",
    }
    (OUT_DIR / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Done. Forecast layer updated.")


def check() -> None:
    meta_path = OUT_DIR / "forecast_meta.json"
    if not meta_path.exists():
        print("No refresh history found.")
        return
    meta = json.loads(meta_path.read_text())
    print(f"Last updated : {meta['last_updated']}")
    print(f"Centers      : {meta['centers']}")

    layer = OUT_DIR / "forecast_layer.geojson"
    if layer.exists():
        with open(layer) as f:
            geo = json.load(f)
        feats = geo["features"]
        with_problem = sum(1 for f in feats if f["properties"].get("has_problem"))
        print(f"Polygons     : {len(feats)} total, {with_problem} with active problems")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily avalanche forecast refresh")
    parser.add_argument("--center", nargs="+", help="Limit to specific center IDs")
    parser.add_argument("--check",  action="store_true", help="Show last update info")
    args = parser.parse_args()

    if args.check:
        check()
    else:
        run(center_filter=args.center)
