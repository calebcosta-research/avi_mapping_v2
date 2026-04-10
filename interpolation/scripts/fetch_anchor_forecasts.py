"""
fetch_anchor_forecasts.py
--------------------------
Extracts per-center structured forecast data for all anchor centers referenced
in gap_zones.json and archives it to:

    interpolation/data/anchor_forecasts/{center_id}/{YYYY-MM-DD}.json

Two modes
---------
  extract  (default)  Pull anchor forecasts from the existing R2 forecast archive
                      (data/forecasts/YYYY-MM-DD.json). Fast, no API calls needed.
                      Works on any date range already in the archive.

  fetch               Call the NAC API directly. Used for the current day before
                      the main pipeline has run, or to backfill missing dates.

Output format per file
----------------------
{
  "date":      "2026-03-26",
  "center_id": "SAC",
  "source":    "extract" | "api",
  "zones": {
    "2458": {                         <- zone_id (NAC numeric ID)
      "zone_name": "Central Sierra Nevada",
      "danger": {
        "alpine":          3,
        "treeline":        3,
        "below_treeline":  2
      },
      "problems": [
        {
          "type":            "Persistent Slab",
          "likelihood":      "likely",
          "size_min":        2.0,
          "size_max":        3.0,
          "aspects":         ["N", "NE", "NW"],
          "elevation_bands": ["alpine", "treeline"]
        }
      ]
    }
  },
  "center_summary": {
    "danger_max": {"alpine": 3, "treeline": 3, "below_treeline": 2},
    "active_problem_types": ["Persistent Slab", "Wind Slab"],
    "problem_likelihood_max": {"Persistent Slab": "likely", "Wind Slab": "possible"}
  }
}

Usage
-----
    python fetch_anchor_forecasts.py                         # extract today from R2 archive
    python fetch_anchor_forecasts.py --mode fetch            # call API directly
    python fetch_anchor_forecasts.py --date 2026-03-15       # specific date (extract only)
    python fetch_anchor_forecasts.py --date-range 2026-01-01 2026-03-26   # range (extract)
    python fetch_anchor_forecasts.py --centers SAC MSAC      # specific centers only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR    = Path(__file__).parent
INTERP_DIR     = SCRIPTS_DIR.parent
FORECASTS_DIR  = INTERP_DIR.parent / "forecasts"
WEB_DATA_DIR   = INTERP_DIR.parent / "web" / "data"

GAP_ZONES_PATH = INTERP_DIR / "zones" / "gap_zones.json"
OUTPUT_DIR     = INTERP_DIR / "data" / "anchor_forecasts"

NAC_API_BASE   = "https://api.avalanche.org/v2/public"

ELEV_BAND_NORM = {
    "alpine":         "alpine",
    "above_treeline": "alpine",
    "near_treeline":  "treeline",
    "treeline":       "treeline",
    "below_treeline": "below_treeline",
    "below treeline": "below_treeline",
}

LIKELIHOOD_ORDER = ["unlikely", "possible", "likely", "very likely", "almost certain"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gap_zones() -> dict:
    with open(GAP_ZONES_PATH) as f:
        return json.load(f)


def anchor_center_ids(gap_zones: dict) -> set[str]:
    ids: set[str] = set()
    for zone in gap_zones["zones"]:
        for anchor in zone["anchor_centers"]:
            ids.add(anchor["center_id"])
    return ids


def forecast_archive_path(d: date) -> Optional[Path]:
    """Path to the archived forecast.json for a given date."""
    p = WEB_DATA_DIR / "forecasts" / f"{d.isoformat()}.json"
    if p.exists():
        return p
    return None


def normalize_elev(band: str) -> str:
    return ELEV_BAND_NORM.get(band.lower().replace(" ", "_"), band.lower())


def max_likelihood(likelihoods: list[str]) -> str:
    ranked = [(LIKELIHOOD_ORDER.index(lh.lower()) if lh.lower() in LIKELIHOOD_ORDER else -1)
              for lh in likelihoods]
    best = max(range(len(ranked)), key=lambda i: ranked[i])
    return likelihoods[best]


# ── Extraction from forecast.json archive ─────────────────────────────────────

def extract_from_archive(forecast_json: dict, center_id: str) -> dict:
    """
    Pull all zones belonging to center_id out of a forecast.json blob.
    Reconstructs per-zone danger and problems from the cell-level data.
    """
    zones_out = {}

    for zone_index_str, zone in forecast_json.get("zones", {}).items():
        if zone.get("center_id") != center_id:
            continue

        zone_id   = str(zone.get("zone_id", zone_index_str))
        zone_name = zone.get("zone_name", "")

        # Aggregate danger and problems across all cells in this zone
        danger_by_band: dict[str, list[int]] = defaultdict(list)
        all_problems:   list[dict] = []
        seen_problems:  set[tuple] = set()

        for key, cell in zone.items():
            if not key.isdigit():
                continue
            cell_id = int(key)
            elev_raw = {1: "below_treeline", 2: "treeline", 3: "alpine"}.get(cell_id % 10)
            if elev_raw:
                danger_by_band[elev_raw].append(cell.get("danger", 0))

            for prob in cell.get("problems", []):
                sig = (prob["type"], prob.get("likelihood", ""), cell_id % 10)
                if sig not in seen_problems:
                    seen_problems.add(sig)
                    all_problems.append({
                        "type":       prob["type"],
                        "likelihood": prob.get("likelihood", "unknown"),
                        "size_min":   prob.get("size_min"),
                        "size_max":   prob.get("size_max"),
                        "_cell_id":   cell_id,
                    })

        # Collapse danger to max per elevation band
        danger = {
            band: max(vals) if vals else 0
            for band, vals in danger_by_band.items()
        }
        for band in ("alpine", "treeline", "below_treeline"):
            danger.setdefault(band, 0)

        # Collapse problems: merge cells for the same type
        merged: dict[str, dict] = {}
        for p in all_problems:
            t = p["type"]
            if t not in merged:
                merged[t] = {
                    "type":            t,
                    "likelihood":      p["likelihood"],
                    "size_min":        p["size_min"],
                    "size_max":        p["size_max"],
                    "aspects":         [],
                    "elevation_bands": [],
                }
            else:
                # Take highest likelihood
                if (LIKELIHOOD_ORDER.index(p["likelihood"].lower())
                        > LIKELIHOOD_ORDER.index(merged[t]["likelihood"].lower())):
                    merged[t]["likelihood"] = p["likelihood"]

        zones_out[zone_id] = {
            "zone_name": zone_name,
            "danger":    danger,
            "problems":  list(merged.values()),
        }

    return zones_out


def build_center_summary(zones: dict) -> dict:
    """Compute center-level aggregates across all zones for model features."""
    band_max: dict[str, int] = {"alpine": 0, "treeline": 0, "below_treeline": 0}
    prob_lh:  dict[str, list[str]] = defaultdict(list)

    for zone in zones.values():
        for band, level in zone.get("danger", {}).items():
            if band in band_max:
                band_max[band] = max(band_max[band], level)
        for prob in zone.get("problems", []):
            prob_lh[prob["type"]].append(prob["likelihood"])

    return {
        "danger_max":              band_max,
        "active_problem_types":    sorted(prob_lh.keys()),
        "problem_likelihood_max":  {t: max_likelihood(lhs) for t, lhs in prob_lh.items()},
    }


# ── Direct API fetch ──────────────────────────────────────────────────────────

def fetch_from_api(center_id: str) -> dict:
    """
    Fetch the current forecast for all zones of a center via the NAC API.
    Falls back to the forecast_parser infrastructure already in forecasts/scripts.
    """
    sys.path.insert(0, str(FORECASTS_DIR / "scripts"))
    from forecast_parser import ForecastClient
    from build_forecast_json import elev_band_name, DANGER_LABELS

    client    = ForecastClient()
    zone_list = client._get_json(
        f"https://api.avalanche.org/v2/public/products/map-layer/{center_id.lower()}"
    )
    zone_ids  = [f["id"] for f in zone_list.get("features", [])]
    log.info("  [%s] %d zones to fetch", center_id, len(zone_ids))

    zones_out = {}
    for zid in zone_ids:
        try:
            fc = client.fetch_zone(center_id, int(zid))
        except Exception as e:
            log.warning("  [%s] zone %s failed: %s", center_id, zid, e)
            continue

        danger: dict[str, int] = {}
        for dr in fc.danger_ratings:
            band = elev_band_name(dr.elevation_band)
            danger[band] = max(danger.get(band, 0), int(dr.danger_level or 0))

        problems = []
        for prob in fc.avalanche_problems:
            problems.append({
                "type":            prob.problem_type,
                "likelihood":      prob.likelihood or "unknown",
                "size_min":        prob.size_min,
                "size_max":        prob.size_max,
                "aspects":         list(prob.aspects or []),
                "elevation_bands": list(prob.elevation_bands or []),
            })

        zones_out[str(zid)] = {
            "zone_name": fc.zone_name or "",
            "danger":    danger,
            "problems":  problems,
        }

    return zones_out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_date(target_date: date, center_ids: set[str], mode: str) -> dict[str, int]:
    """Process one date. Returns {center_id: status} where status = 1 success, 0 skip, -1 error."""
    results = {}

    if mode == "extract":
        archive = forecast_archive_path(target_date)
        if archive is None:
            log.debug("No archive for %s — skipping", target_date)
            return {cid: 0 for cid in center_ids}

        with open(archive) as f:
            forecast_json = json.load(f)

        for cid in center_ids:
            out_path = OUTPUT_DIR / cid / f"{target_date.isoformat()}.json"
            if out_path.exists():
                log.debug("  [%s] %s already exists — skipping", cid, target_date)
                results[cid] = 0
                continue
            zones = extract_from_archive(forecast_json, cid)
            if not zones:
                log.warning("  [%s] no zones found in archive for %s", cid, target_date)
                results[cid] = 0
                continue
            _save(out_path, cid, target_date, "extract", zones)
            results[cid] = 1

    elif mode == "fetch":
        for cid in center_ids:
            out_path = OUTPUT_DIR / cid / f"{target_date.isoformat()}.json"
            try:
                zones = fetch_from_api(cid)
                _save(out_path, cid, target_date, "api", zones)
                results[cid] = 1
            except Exception as e:
                log.error("  [%s] API fetch failed: %s", cid, e)
                results[cid] = -1

    return results


def _save(path: Path, center_id: str, d: date, source: str, zones: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date":           d.isoformat(),
        "center_id":      center_id,
        "source":         source,
        "zones":          zones,
        "center_summary": build_center_summary(zones),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("  [%s] %s -> %d zones saved", center_id, d, len(zones))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch/extract anchor center forecasts")
    parser.add_argument("--mode",       choices=["extract", "fetch"], default="extract",
                        help="extract: from R2 archive  fetch: call NAC API directly")
    parser.add_argument("--date",       type=str, default=None,
                        help="Single date YYYY-MM-DD (default: today)")
    parser.add_argument("--date-range", nargs=2, metavar=("START", "END"),
                        help="Date range YYYY-MM-DD YYYY-MM-DD (extract mode only)")
    parser.add_argument("--centers",    nargs="+", default=None,
                        help="Specific center IDs to process (default: all from gap_zones.json)")
    args = parser.parse_args()

    gap_zones  = load_gap_zones()
    center_ids = set(args.centers) if args.centers else anchor_center_ids(gap_zones)
    log.info("Processing %d anchor centers: %s", len(center_ids), sorted(center_ids))

    if args.date_range:
        start = date.fromisoformat(args.date_range[0])
        end   = date.fromisoformat(args.date_range[1])
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    else:
        target_str = args.date or date.today().isoformat()
        dates      = [date.fromisoformat(target_str)]

    total_saved = 0
    for d in dates:
        res = process_date(d, center_ids, args.mode)
        total_saved += sum(1 for v in res.values() if v == 1)

    log.info("Done — %d files saved across %d dates", total_saved, len(dates))


if __name__ == "__main__":
    main()
