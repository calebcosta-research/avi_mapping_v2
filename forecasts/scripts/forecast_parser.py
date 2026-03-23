"""
forecast_parser.py
------------------
Fetches and normalizes avalanche forecast reports from the NAC public API
(api.avalanche.org/v2) into a common AvalancheForecast schema.

Supported centers (easily extensible):
  CSAW  - Central Sierra Avalanche Center
  ESAC  - Eastern Sierra Avalanche Center
  BTAC  - Bridger-Teton Avalanche Center
  ... any center_id on api.avalanche.org

Usage:
    from forecast_parser import ForecastClient, AvalancheForecast

    client = ForecastClient()
    forecast = client.fetch("CSAW")
    print(forecast.danger_ratings)
    print(forecast.avalanche_problems)

    # Fetch all centers relevant to your gap zone
    forecasts = client.fetch_multiple(["CSAW", "ESAC"])

PhD project: Lassen gap-zone interpolation
Author: <your name>
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAC_API_BASE = "https://api.avalanche.org/v2/public/products"

# Danger rose layout: 8 compass aspects × 3 elevation bands = 24 positions.
# Aspect order (clockwise from N): N NE E SE S SW W NW
# Elevation bands (inner→outer in the rose): alpine, treeline, below_treeline
#
# Rose index layout (matches NAC API array indexing):
#   indices 0–7:   alpine         (N, NE, E, SE, S, SW, W, NW)
#   indices 8–15:  treeline       (N, NE, E, SE, S, SW, W, NW)
#   indices 16–23: below_treeline (N, NE, E, SE, S, SW, W, NW)

ASPECTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ELEVATION_BANDS = ["alpine", "treeline", "below_treeline"]

DANGER_LEVEL_LABELS = {
    -1: "no_rating",
    0: "no_rating",
    1: "low",
    2: "moderate",
    3: "considerable",
    4: "high",
    5: "extreme",
}

DANGER_COLORS = {
    "no_rating": "#CCCCCC",
    "low": "#78B943",
    "moderate": "#F4C025",
    "considerable": "#F58220",
    "high": "#E3001F",
    "extreme": "#1A1A1A",
}

AVALANCHE_PROBLEM_TYPES = {
    "wind_slab",
    "storm_slab",
    "persistent_slab",
    "deep_persistent_slab",
    "wet_slab",
    "wet_loose",
    "dry_loose",
    "cornice",
    "glide",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DangerLevel(IntEnum):
    NO_RATING = -1
    LOW = 1
    MODERATE = 2
    CONSIDERABLE = 3
    HIGH = 4
    EXTREME = 5

    @classmethod
    def from_int(cls, value: int) -> "DangerLevel":
        """Safely coerce API integer to DangerLevel; returns NO_RATING on unknown."""
        try:
            return cls(value)
        except ValueError:
            return cls.NO_RATING

    @property
    def label(self) -> str:
        return DANGER_LEVEL_LABELS.get(self.value, "no_rating")

    @property
    def color(self) -> str:
        return DANGER_COLORS.get(self.label, "#CCCCCC")


@dataclass
class DangerRating:
    """Danger at a specific elevation band."""
    elevation_band: str          # alpine | treeline | below_treeline
    danger_level: DangerLevel
    valid_time: str = "all_day"  # all_day | morning | afternoon


@dataclass
class DangerRoseCell:
    """One cell in the 24-cell danger rose (aspect × elevation band)."""
    aspect: str          # N NE E SE S SW W NW
    elevation_band: str  # alpine | treeline | below_treeline
    value: int           # raw rose value (1=problem present, 0=absent; or danger 1-5)
    danger_level: Optional[DangerLevel] = None


@dataclass
class AvalancheProblem:
    """One avalanche problem entry from the forecast."""
    problem_type: str                          # e.g. "wind_slab", "persistent_slab"
    aspects: list[str] = field(default_factory=list)   # list of affected aspects
    elevation_bands: list[str] = field(default_factory=list)
    likelihood: Optional[str] = None          # unlikely | possible | likely | very_likely | almost_certain
    size_min: Optional[float] = None          # D-scale minimum (1.0–5.0)
    size_max: Optional[float] = None
    discussion: Optional[str] = None
    rose_array: list[int] = field(default_factory=list)  # raw 24-element array


@dataclass
class AvalancheForecast:
    """
    Normalized forecast from any NAC-platform avalanche center.

    This is the canonical schema for the gap-zone interpolation model and
    the map visualizer. All source-specific parsing produces this object.
    """
    center_id: str
    center_name: str
    zone_name: str
    forecast_id: int
    issued_at: datetime
    expires_at: Optional[datetime]
    valid_for_date: str                        # ISO date string YYYY-MM-DD
    bottom_line: Optional[str]

    # Structured danger by elevation band (3 rows of the danger bar)
    danger_ratings: list[DangerRating] = field(default_factory=list)

    # Full 24-cell rose, decoded
    danger_rose: list[DangerRoseCell] = field(default_factory=list)

    # Avalanche problems (0–3 per forecast)
    avalanche_problems: list[AvalancheProblem] = field(default_factory=list)

    # Raw travel advice text
    travel_advice: Optional[str] = None

    # Source metadata
    source_url: Optional[str] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def max_danger(self) -> DangerLevel:
        """Return the highest danger level across all elevation bands."""
        if not self.danger_ratings:
            return DangerLevel.NO_RATING
        return max(r.danger_level for r in self.danger_ratings)

    def danger_for_band(self, band: str) -> Optional[DangerLevel]:
        """Return DangerLevel for a specific elevation band, or None."""
        for r in self.danger_ratings:
            if r.elevation_band == band:
                return r.danger_level
        return None

    def problems_by_aspect(self, aspect: str) -> list[AvalancheProblem]:
        """Return all problems that include a given aspect."""
        return [p for p in self.avalanche_problems if aspect in p.aspects]

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON export or DataFrame construction."""
        return {
            "center_id": self.center_id,
            "center_name": self.center_name,
            "zone_name": self.zone_name,
            "forecast_id": self.forecast_id,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "valid_for_date": self.valid_for_date,
            "bottom_line": self.bottom_line,
            "max_danger": self.max_danger().label,
            "max_danger_level": int(self.max_danger()),
            "danger_ratings": [
                {
                    "elevation_band": r.elevation_band,
                    "danger_level": int(r.danger_level),
                    "danger_label": r.danger_level.label,
                    "color": r.danger_level.color,
                }
                for r in self.danger_ratings
            ],
            "avalanche_problems": [
                {
                    "problem_type": p.problem_type,
                    "aspects": p.aspects,
                    "elevation_bands": p.elevation_bands,
                    "likelihood": p.likelihood,
                    "size_min": p.size_min,
                    "size_max": p.size_max,
                    "discussion": p.discussion,
                }
                for p in self.avalanche_problems
            ],
            "travel_advice": self.travel_advice,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string to datetime (UTC-aware). Returns None on failure."""
    if not dt_str:
        return None
    try:
        # Python 3.11+ handles Z suffix natively; handle older versions too
        dt_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        logger.warning("Could not parse datetime string: %r", dt_str)
        return None


def _decode_danger_rose(rose_array: list[int]) -> list[DangerRoseCell]:
    """
    Decode a 24-element NAC danger rose array into structured DangerRoseCell objects.

    Array layout (NAC convention):
      positions  0– 7: alpine band,         N→NW clockwise
      positions  8–15: treeline band,        N→NW clockwise
      positions 16–23: below_treeline band,  N→NW clockwise
    """
    if not rose_array or len(rose_array) < 24:
        return []

    cells = []
    for band_idx, band_name in enumerate(ELEVATION_BANDS):
        for aspect_idx, aspect in enumerate(ASPECTS):
            raw_value = rose_array[band_idx * 8 + aspect_idx]
            cells.append(DangerRoseCell(
                aspect=aspect,
                elevation_band=band_name,
                value=raw_value,
            ))
    return cells


def _parse_danger_ratings(forecast_data: dict) -> list[DangerRating]:
    """
    Extract structured danger ratings (alpine / treeline / below_treeline)
    from the NAC forecast product JSON.

    The NAC API uses a `danger` list with entries shaped as:
      { "lower": int, "upper": int, "valid_day": "current|tomorrow" }
    where indices 0=alpine, 1=treeline, 2=below_treeline.
    Some centers put this under `forecast_zone[0].danger`.
    """
    danger_list = (
        forecast_data.get("danger")
        or (forecast_data.get("forecast_zone") or [{}])[0].get("danger")
        or []
    )

    ratings = []
    for i, band in enumerate(ELEVATION_BANDS):
        if i < len(danger_list):
            entry = danger_list[i]
            # Use "upper" value (afternoon); fall back to "lower" (morning)
            raw = entry.get("upper", entry.get("lower", -1))
        else:
            raw = -1
        ratings.append(DangerRating(
            elevation_band=band,
            danger_level=DangerLevel.from_int(raw),
        ))
    return ratings


def _parse_avalanche_problems(forecast_data: dict) -> list[AvalancheProblem]:
    """
    Parse avalanche problems from the NAC product JSON.

    Each problem entry typically has:
      - name (problem type string)
      - aspects (list of aspect strings or a 24-element rose array)
      - likelihood
      - size (min/max)
      - discussion (free text)
    """
    raw_problems = (
        forecast_data.get("avalanche_problems")
        or forecast_data.get("forecast_zone", [{}])[0].get("avalanche_problems")
        or []
    )

    problems = []
    for rp in raw_problems:
        # Normalize problem type string
        ptype = (rp.get("name") or rp.get("type") or "unknown").lower().strip()
        ptype = ptype.replace(" ", "_").replace("-", "_")

        # Aspect extraction — the API can give either a list of strings or a
        # 24-element rose-style array (non-zero means affected)
        raw_aspects = rp.get("aspects") or []
        if raw_aspects and isinstance(raw_aspects[0], int):
            # Rose-encoded: decode the 24-element array to aspect+band pairs
            aspects = [
                ASPECTS[i % 8]
                for i, v in enumerate(raw_aspects)
                if v > 0
            ]
            aspects = sorted(set(aspects), key=ASPECTS.index)
            elev_bands = [
                ELEVATION_BANDS[i // 8]
                for i, v in enumerate(raw_aspects)
                if v > 0
            ]
            elev_bands = sorted(set(elev_bands), key=ELEVATION_BANDS.index)
        else:
            aspects = [a.upper() for a in raw_aspects if a]
            elev_bands = rp.get("elevation_bands") or rp.get("elevations") or []
            elev_bands = [e.lower().replace(" ", "_") for e in elev_bands]

        # Likelihood
        likelihood_raw = rp.get("likelihood") or ""
        likelihood = likelihood_raw.lower().replace(" ", "_") if likelihood_raw else None

        # Size (D-scale)
        size = rp.get("size") or {}
        size_min = size.get("min") if isinstance(size, dict) else None
        size_max = size.get("max") if isinstance(size, dict) else None

        discussion = rp.get("discussion") or rp.get("problem_description") or None

        problems.append(AvalancheProblem(
            problem_type=ptype,
            aspects=aspects,
            elevation_bands=elev_bands,
            likelihood=likelihood,
            size_min=float(size_min) if size_min is not None else None,
            size_max=float(size_max) if size_max is not None else None,
            discussion=discussion,
            rose_array=raw_aspects if isinstance(raw_aspects[0] if raw_aspects else None, int) else [],
        ))

    return problems


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class ForecastClient:
    """
    Thin HTTP client for the NAC public API.

    Respects NAC's rate-limit guidance: check no more than once per 15 minutes.
    Caches the last fetched forecast per center_id in memory to avoid
    redundant requests within the same session.
    """

    def __init__(
        self,
        user_agent: str = "AviGapZone-PhD/0.1 (your@email.edu)",
        timeout: int = 15,
        cache_ttl_seconds: int = 900,  # 15 minutes — NAC recommendation
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, AvalancheForecast]] = {}

    def _is_cached(self, center_id: str) -> bool:
        if center_id not in self._cache:
            return False
        ts, _ = self._cache[center_id]
        return (time.monotonic() - ts) < self.cache_ttl

    def _get_json(self, url: str) -> dict | list:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------

    def fetch(self, center_id: str) -> AvalancheForecast:
        """
        Fetch and return a normalized AvalancheForecast for the given center.

        Pulls from in-memory cache if data is fresh (< 15 min old).
        """
        center_id = center_id.upper()

        if self._is_cached(center_id):
            logger.debug("Cache hit for %s", center_id)
            return self._cache[center_id][1]

        url = f"{NAC_API_BASE}?type=forecast&center_id={center_id}"
        logger.info("Fetching forecast from %s", url)

        raw = self._get_json(url)

        # The endpoint returns a list of forecast product objects
        if isinstance(raw, list):
            products = raw
        elif isinstance(raw, dict):
            products = raw.get("data") or raw.get("products") or [raw]
        else:
            raise ValueError(f"Unexpected API response type: {type(raw)}")

        if not products:
            raise ValueError(f"No forecast products returned for center {center_id}")

        # Take the most recent valid forecast
        product = products[0]

        forecast = self._normalize(product, center_id)
        self._cache[center_id] = (time.monotonic(), forecast)
        return forecast

    def fetch_multiple(self, center_ids: list[str]) -> dict[str, AvalancheForecast]:
        """Fetch forecasts for multiple centers. Returns {center_id: forecast}."""
        results = {}
        for cid in center_ids:
            try:
                results[cid] = self.fetch(cid)
            except Exception as exc:
                logger.error("Failed to fetch forecast for %s: %s", cid, exc)
        return results

    def fetch_map_layer(self, center_id: str) -> dict:
        """
        Fetch the GeoJSON map-layer product for a center.
        Returns the raw GeoJSON FeatureCollection (danger level by zone polygon).
        Useful for the Phase 2 map visualizer.
        """
        center_id = center_id.upper()
        url = f"{NAC_API_BASE}/map-layer/{center_id}"
        logger.info("Fetching map layer from %s", url)
        return self._get_json(url)

    # ------------------------------------------------------------------

    def _normalize(self, product: dict, center_id: str) -> AvalancheForecast:
        """Map a raw NAC product dict to AvalancheForecast."""

        # -- Identity
        center_id = center_id.upper()
        forecast_id = product.get("id") or -1
        center_name = (
            product.get("avalanche_center", {}).get("name")
            or product.get("center_name")
            or center_id
        )
        # Zone name lives in forecast_zone list or top-level
        zone_list = product.get("forecast_zone") or []
        zone_name = (
            zone_list[0].get("name") if zone_list
            else product.get("zone_name") or "Unknown zone"
        )

        # -- Timestamps
        issued_at = _parse_iso(product.get("published_time") or product.get("created_at"))
        if issued_at is None:
            issued_at = datetime.now(timezone.utc)
            logger.warning("No issued_at timestamp found; using now()")

        expires_at = _parse_iso(product.get("expires_time"))

        valid_for_date = (
            product.get("forecast_date")
            or product.get("date")
            or issued_at.date().isoformat()
        )

        # -- Bottom line
        bottom_line = (
            product.get("bottom_line")
            or product.get("hazard_discussion")
            or (zone_list[0].get("hazard_discussion") if zone_list else None)
        )

        # -- Danger ratings (alpine / treeline / below_treeline)
        danger_ratings = _parse_danger_ratings(product)

        # -- Danger rose (24-cell)
        rose_raw = (
            product.get("danger_rose")
            or (zone_list[0].get("danger_rose") if zone_list else None)
            or []
        )
        danger_rose = _decode_danger_rose(rose_raw)

        # -- Avalanche problems
        avalanche_problems = _parse_avalanche_problems(product)

        # -- Travel advice
        travel_advice = product.get("travel_advice") or None

        return AvalancheForecast(
            center_id=center_id,
            center_name=center_name,
            zone_name=zone_name,
            forecast_id=forecast_id,
            issued_at=issued_at,
            expires_at=expires_at,
            valid_for_date=str(valid_for_date),
            bottom_line=bottom_line,
            danger_ratings=danger_ratings,
            danger_rose=danger_rose,
            avalanche_problems=avalanche_problems,
            travel_advice=travel_advice,
            source_url=f"{NAC_API_BASE}?type=forecast&center_id={center_id}",
        )
