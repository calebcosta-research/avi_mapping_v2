"""
fetch_snotel.py
---------------
Fetches SNOTEL station data (SWE, snow depth, temperature, precipitation) for
stations near each gap zone and saves daily time series to:

    interpolation/data/snotel/{zone_id}/{YYYY-MM-DD}.json   (daily snapshots)
    interpolation/data/snotel/{zone_id}/history.json        (multi-day time series)

SNOTEL data is fetched via the NRCS AWDB REST API (no auth required):
    https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html

Key SNOTEL variables fetched
-----------------------------
  WTEQ  — Snow Water Equivalent (inches)
  SNWD  — Snow Depth (inches)
  TOBS  — Observed air temperature (°F)
  PREC  — Accumulated precipitation (inches)
  PRCPSA  — Precipitation (daily, inches)

Station selection
-----------------
  Uses bounding box from gap_zones.json extended by `--buffer-deg` (default 1.0°)
  to capture stations near (not just inside) each gap zone.

Usage
-----
    python fetch_snotel.py                            # today, all zones
    python fetch_snotel.py --zone lassen-ca           # specific zone
    python fetch_snotel.py --date 2026-03-15          # specific date
    python fetch_snotel.py --date-range 2026-01-01 2026-03-26  # history
    python fetch_snotel.py --buffer-deg 1.5           # wider station search
    python fetch_snotel.py --list-stations lassen-ca  # list stations, no fetch

Output format (daily snapshot)
-------------------------------
{
  "date":    "2026-03-26",
  "zone_id": "lassen-ca",
  "stations": {
    "1050:CA:SNTL": {
      "name":      "Lassen Peak",
      "elevation_ft": 8512,
      "lat":  40.4872,
      "lon": -121.5028,
      "distance_km": 4.2,
      "values": {
        "WTEQ": 42.1,
        "SNWD": 198,
        "TOBS": 28.4,
        "PREC": 315.2
      }
    }
  },
  "zone_summary": {
    "station_count": 3,
    "swe_mean_in":   38.4,
    "swe_max_in":    44.1,
    "swe_min_in":    31.2,
    "depth_mean_in": 182.0,
    "temp_mean_f":   26.8
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from math import radians, sin, cos, sqrt, atan2
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
GAP_ZONES_PATH = INTERP_DIR / "zones" / "gap_zones.json"
OUTPUT_DIR     = INTERP_DIR / "data" / "snotel"

# ── NRCS AWDB API ─────────────────────────────────────────────────────────────
AWDB_BASE      = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
SNOTEL_NETWORK = "SNTL"

VARIABLES = ["WTEQ", "SNWD", "TOBS", "PREC", "PRCPSA"]

DEFAULT_BUFFER_DEG = 1.0   # degrees lat/lon added to zone bbox for station search


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def awdb_get(endpoint: str, params: dict) -> dict | list:
    url = f"{AWDB_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Station discovery ─────────────────────────────────────────────────────────

def find_stations_in_bbox(
    min_lon: float, min_lat: float,
    max_lon: float, max_lat: float,
    network: str = SNOTEL_NETWORK,
) -> list[dict]:
    """
    Returns SNOTEL stations within the given bounding box.
    Each station dict includes stationTriplet, name, elevation, lat, lon.
    """
    try:
        data = awdb_get("stations", {
            "minLatitude":   min_lat,
            "maxLatitude":   max_lat,
            "minLongitude":  min_lon,
            "maxLongitude":  max_lon,
            "networkCds":    network,
        })
    except Exception as e:
        log.warning("Station search failed: %s", e)
        return []
    return data if isinstance(data, list) else []


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_station_data(
    station_triplet: str,
    target_date: date,
    variables: list[str] = VARIABLES,
) -> dict[str, Optional[float]]:
    """
    Fetch daily element values for one station on one date.
    Returns {element_code: value_or_None}.
    """
    date_str = target_date.isoformat()
    values: dict[str, Optional[float]] = {}

    for var in variables:
        try:
            data = awdb_get("data", {
                "stationTriplets": station_triplet,
                "elementCd":       var,
                "beginDate":       date_str,
                "endDate":         date_str,
                "duration":        "DAILY",
            })
            # Response: list of {stationTriplet, data: [{date, value}]}
            if data and isinstance(data, list) and data[0].get("data"):
                raw = data[0]["data"][0].get("value")
                values[var] = float(raw) if raw is not None else None
            else:
                values[var] = None
        except Exception as e:
            log.debug("  %s/%s fetch failed: %s", station_triplet, var, e)
            values[var] = None

    return values


# ── Zone summary ──────────────────────────────────────────────────────────────

def build_zone_summary(stations: dict) -> dict:
    swevals  = [s["values"].get("WTEQ") for s in stations.values() if s["values"].get("WTEQ") is not None]
    depthvals = [s["values"].get("SNWD") for s in stations.values() if s["values"].get("SNWD") is not None]
    tempvals = [s["values"].get("TOBS") for s in stations.values() if s["values"].get("TOBS") is not None]
    return {
        "station_count": len(stations),
        "swe_mean_in":   round(sum(swevals) / len(swevals), 2) if swevals else None,
        "swe_max_in":    round(max(swevals), 2) if swevals else None,
        "swe_min_in":    round(min(swevals), 2) if swevals else None,
        "depth_mean_in": round(sum(depthvals) / len(depthvals), 1) if depthvals else None,
        "temp_mean_f":   round(sum(tempvals) / len(tempvals), 1) if tempvals else None,
    }


# ── Per-zone pipeline ─────────────────────────────────────────────────────────

def process_zone(zone: dict, target_date: date, buffer_deg: float) -> dict:
    """Fetch SNOTEL data for all stations near a gap zone on a given date."""
    zone_id = zone["zone_id"]
    bbox    = zone["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
    centroid_lon, centroid_lat = zone["centroid"]

    # Expand bbox by buffer
    min_lon = bbox[0] - buffer_deg
    min_lat = bbox[1] - buffer_deg
    max_lon = bbox[2] + buffer_deg
    max_lat = bbox[3] + buffer_deg

    log.info("[%s] Searching SNOTEL stations in bbox [%.2f,%.2f,%.2f,%.2f]",
             zone_id, min_lon, min_lat, max_lon, max_lat)
    raw_stations = find_stations_in_bbox(min_lon, min_lat, max_lon, max_lat)
    log.info("[%s] Found %d stations", zone_id, len(raw_stations))

    stations_out: dict[str, dict] = {}
    for st in raw_stations:
        triplet   = st.get("stationTriplet", "")
        st_lat    = float(st.get("latitude", 0))
        st_lon    = float(st.get("longitude", 0))
        dist_km   = haversine_km(centroid_lon, centroid_lat, st_lon, st_lat)

        log.debug("  Fetching %s (%s)...", triplet, st.get("name", ""))
        values = fetch_station_data(triplet, target_date)

        stations_out[triplet] = {
            "name":         st.get("name", ""),
            "elevation_ft": st.get("elevation"),
            "lat":          st_lat,
            "lon":          st_lon,
            "distance_km":  round(dist_km, 1),
            "values":       values,
        }

    return {
        "date":         target_date.isoformat(),
        "zone_id":      zone_id,
        "stations":     stations_out,
        "zone_summary": build_zone_summary(stations_out),
    }


def save_snapshot(zone_id: str, target_date: date, payload: dict) -> None:
    out_dir = OUTPUT_DIR / zone_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_date.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("[%s] Saved -> %s (%d stations)", zone_id, out_path,
             len(payload.get("stations", {})))


def list_stations(zone: dict, buffer_deg: float) -> None:
    """Print stations near a zone without fetching data."""
    bbox = zone["bbox"]
    min_lon, min_lat = bbox[0] - buffer_deg, bbox[1] - buffer_deg
    max_lon, max_lat = bbox[2] + buffer_deg, bbox[3] + buffer_deg
    centroid_lon, centroid_lat = zone["centroid"]

    stations = find_stations_in_bbox(min_lon, min_lat, max_lon, max_lat)
    print(f"\nSNOTEL stations near {zone['name']} ({zone['zone_id']}):")
    print(f"  Buffer: {buffer_deg} deg — {len(stations)} stations found\n")
    print(f"  {'Triplet':<20} {'Name':<30} {'Elev ft':>8}  {'Dist km':>8}")
    print("  " + "-" * 72)
    for st in sorted(stations, key=lambda s: haversine_km(
            centroid_lon, centroid_lat,
            float(s.get("longitude", 0)), float(s.get("latitude", 0)))):
        dist = haversine_km(centroid_lon, centroid_lat,
                            float(st.get("longitude", 0)), float(st.get("latitude", 0)))
        print(f"  {st.get('stationTriplet',''):<20} {st.get('name',''):<30} "
              f"{st.get('elevation', '?'):>8}  {dist:>8.1f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch SNOTEL covariates for gap zones")
    parser.add_argument("--zone",           default=None,
                        help="Zone ID to process (default: all)")
    parser.add_argument("--date",           type=str, default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--date-range",     nargs=2, metavar=("START", "END"),
                        help="Date range YYYY-MM-DD YYYY-MM-DD")
    parser.add_argument("--buffer-deg",     type=float, default=DEFAULT_BUFFER_DEG,
                        help=f"Degrees to expand bbox for station search (default: {DEFAULT_BUFFER_DEG})")
    parser.add_argument("--list-stations",  metavar="ZONE_ID",
                        help="List SNOTEL stations near a zone and exit")
    args = parser.parse_args()

    with open(GAP_ZONES_PATH) as f:
        gap_zones = json.load(f)

    zone_map = {z["zone_id"]: z for z in gap_zones["zones"]}

    # --list-stations mode
    if args.list_stations:
        if args.list_stations not in zone_map:
            print(f"Unknown zone: {args.list_stations}")
            print(f"Available: {', '.join(zone_map)}")
            return
        list_stations(zone_map[args.list_stations], args.buffer_deg)
        return

    # Select zones
    if args.zone:
        if args.zone not in zone_map:
            log.error("Unknown zone: %s", args.zone)
            return
        zones = [zone_map[args.zone]]
    else:
        zones = gap_zones["zones"]

    # Select dates
    if args.date_range:
        start = date.fromisoformat(args.date_range[0])
        end   = date.fromisoformat(args.date_range[1])
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    else:
        target_str = args.date or date.today().isoformat()
        dates      = [date.fromisoformat(target_str)]

    log.info("Processing %d zone(s) across %d date(s)", len(zones), len(dates))

    total = 0
    for z in zones:
        for d in dates:
            out_path = OUTPUT_DIR / z["zone_id"] / f"{d.isoformat()}.json"
            if out_path.exists():
                log.debug("[%s] %s already exists — skipping", z["zone_id"], d)
                continue
            payload = process_zone(z, d, args.buffer_deg)
            save_snapshot(z["zone_id"], d, payload)
            total += 1

    log.info("Done — %d snapshot(s) saved", total)


if __name__ == "__main__":
    main()
