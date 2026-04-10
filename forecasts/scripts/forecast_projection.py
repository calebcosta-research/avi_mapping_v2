"""
forecast_projection.py
-----------------------
Joins today's avalanche forecast data to the terrain index polygons.

For each polygon in terrain_index.geojson:
  - Looks up the zone's forecast (via forecast_parser)
  - Checks which avalanche problems target this polygon's (aspect, elev_band)
  - Assigns: danger_level, problem list, likelihood, size, colors

Output:
  forecasts/output/forecast_layer.geojson   — ready for Mapbox GL JS overlay

Usage:
    python forecast_projection.py                  # all zones in terrain_index
    python forecast_projection.py --center BTAC    # single center
    python forecast_projection.py --date 2026-03-23  # specific date (uses cached forecast)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from forecast_parser import ForecastClient, AvalancheForecast as ZoneForecast

OUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# Danger color scale (standard avalanche danger palette)
# ---------------------------------------------------------------------------

DANGER_COLORS = {
    1: "#5db85c",   # Low       — green
    2: "#fff200",   # Moderate  — yellow
    3: "#f5a623",   # Considerable — orange
    4: "#d0021b",   # High      — red
    5: "#000000",   # Extreme   — black
    0: "#cccccc",   # No rating — gray
}

DANGER_LABELS = {
    1: "Low", 2: "Moderate", 3: "Considerable",
    4: "High", 5: "Extreme", 0: "No Rating",
}

# Likelihood ordering for priority when multiple problems hit same polygon
LIKELIHOOD_RANK = {
    "unlikely": 1,
    "possible": 2,
    "likely": 3,
    "very likely": 4,
    "almost certain": 5,
}

# Problem type display names
PROBLEM_DISPLAY = {
    "Wind Slab":             "Wind Slab",
    "Storm Slab":            "Storm Slab",
    "Wet Slab":              "Wet Slab",
    "Persistent Slab":       "Persistent Slab",
    "Deep Persistent Slab":  "Deep Persistent Slab",
    "Cornice":               "Cornice",
    "Glide Avalanche":       "Glide Avalanche",
    "Loose Wet":             "Loose Wet",
    "Loose Dry":             "Loose Dry",
}


# ---------------------------------------------------------------------------
# Core projection logic
# ---------------------------------------------------------------------------

def elev_band_name(band: str) -> str:
    """Normalize elevation band string."""
    b = band.lower().replace(" ", "_")
    if b in ("alpine",):
        return "alpine"
    if b in ("treeline", "near_treeline", "near treeline"):
        return "treeline"
    if b in ("below_treeline", "below treeline"):
        return "below_treeline"
    return b


def problems_for_cell(forecast: ZoneForecast, aspect: str, elev_band: str) -> list[dict]:
    """
    Return all avalanche problems that target this (aspect, elev_band) cell,
    sorted by likelihood descending.
    """
    matches = []
    norm_band = elev_band_name(elev_band)

    for prob in forecast.avalanche_problems:
        aspects_hit = [a.upper() for a in (prob.aspects or [])]
        bands_hit   = [elev_band_name(b) for b in (prob.elevation_bands or [])]

        if aspect.upper() in aspects_hit and norm_band in bands_hit:
            matches.append({
                "problem_type": prob.problem_type,
                "likelihood":   prob.likelihood or "unknown",
                "size_min":     prob.size_min,
                "size_max":     prob.size_max,
                "rank":         LIKELIHOOD_RANK.get((prob.likelihood or "").lower(), 0),
            })

    matches.sort(key=lambda x: -x["rank"])
    return matches


def danger_for_band(forecast: ZoneForecast, elev_band: str) -> int:
    """Return numeric danger level (1-5) for the given elevation band."""
    norm = elev_band_name(elev_band)
    for d in forecast.danger_ratings:
        if elev_band_name(d.elevation_band) == norm:
            return d.danger_level or 0
    return 0


def project_forecast(terrain_feature: dict, forecast: ZoneForecast) -> dict:
    """
    Annotate a terrain GeoJSON feature with forecast data.
    Returns a new feature dict with forecast properties added.
    """
    props = terrain_feature["properties"].copy()
    aspect   = props["aspect"]
    elev_band = props["elev_band_name"]

    # Danger level for this elevation band
    danger = danger_for_band(forecast, elev_band)
    props["danger_level"]  = danger
    props["danger_label"]  = DANGER_LABELS.get(danger, "No Rating")
    props["danger_color"]  = DANGER_COLORS.get(danger, "#cccccc")

    # Avalanche problems targeting this cell
    problems = problems_for_cell(forecast, aspect, elev_band)
    props["problem_count"] = len(problems)
    props["has_problem"]   = len(problems) > 0

    if problems:
        # Primary (highest likelihood) problem drives the display
        primary = problems[0]
        props["primary_problem"]     = primary["problem_type"]
        props["primary_likelihood"]  = primary["likelihood"]
        props["primary_size_min"]    = primary["size_min"]
        props["primary_size_max"]    = primary["size_max"]
        # All problems as JSON string (Mapbox properties must be scalar)
        props["all_problems_json"]   = json.dumps(problems)
    else:
        props["primary_problem"]     = None
        props["primary_likelihood"]  = None
        props["primary_size_min"]    = None
        props["primary_size_max"]    = None
        props["all_problems_json"]   = "[]"

    # Forecast metadata
    props["forecast_date"]    = forecast.valid_for_date or str(date.today())
    props["bottom_line"]      = forecast.bottom_line or ""

    return {"type": "Feature",
            "geometry": terrain_feature["geometry"],
            "properties": props}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(center_filter: Optional[str] = None) -> None:
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

    # Collect unique (center_id, zone_id) pairs
    zone_keys = {}
    for feat in features:
        p = feat["properties"]
        key = (p["center_id"], p["zone_id"])
        if key not in zone_keys:
            zone_keys[key] = []
        zone_keys[key].append(feat)

    log.info("Fetching forecasts for %d zone(s)...", len(zone_keys))

    client = ForecastClient()
    output_features = []
    failed_zones = []

    for (center_id, zone_id), zone_features in zone_keys.items():
        log.info("  [%s] zone %s (%d polygons)", center_id, zone_id, len(zone_features))
        try:
            forecast = client.fetch_zone(center_id, int(zone_id))
            for feat in zone_features:
                projected = project_forecast(feat, forecast)
                output_features.append(projected)
            log.info("    -> danger: alpine=%s treeline=%s below=%s  problems=%d",
                     danger_for_band(forecast, "alpine"),
                     danger_for_band(forecast, "treeline"),
                     danger_for_band(forecast, "below_treeline"),
                     len(forecast.avalanche_problems))
        except Exception as e:
            log.warning("    Forecast fetch failed: %s — keeping terrain properties only", e)
            failed_zones.append((center_id, zone_id))
            for feat in zone_features:
                output_features.append(feat)

    out_path = OUT_DIR / "forecast_layer.geojson"
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": output_features}, f)

    problems_hit = sum(1 for f in output_features if f["properties"].get("has_problem"))
    log.info("Saved forecast_layer.geojson — %d polygons, %d with active problems",
             len(output_features), problems_hit)

    if failed_zones:
        log.warning("Zones with no forecast: %s", failed_zones)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", type=str, help="Filter to one center (e.g. BTAC)")
    args = parser.parse_args()
    run(center_filter=args.center)
