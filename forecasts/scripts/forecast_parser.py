"""
forecast_parser.py
------------------
Fetches and normalizes avalanche forecast reports from the NAC public API
(api.avalanche.org/v2) into a common AvalancheForecast schema.

Working endpoints (as of 2026-03):
  /v2/public/products/map-layer          — all zones, center-level danger summary
  /v2/public/products/map-layer/{cid}    — single-center zone list
  /v2/public/products?avalanche_center_id={cid}  — forecast list for a center
  /v2/public/product/{id}                — full forecast detail (problems, elevation danger)
  /v2/public/product?type=forecast&center_id={cid}&zone_id={zone_id}  — current forecast by zone

NOTE: /v2/public/products?type=forecast&center_id={cid} (plural, type param) returns a
PHP memory exhaustion error and is not used here.

Usage:
    from forecast_parser import ForecastClient, AvalancheForecast

    client = ForecastClient()

    # All zones from one center
    forecasts = client.fetch_center("BTAC")

    # All zones from all NAC centers
    forecasts = client.fetch_all()

    # Single zone by center + numeric zone ID (from map-layer feature id)
    forecast = client.fetch_zone("BTAC", 2852)

PhD project: Lassen gap-zone interpolation
"""

from __future__ import annotations

import logging
import re
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

NAC_API_BASE   = "https://api.avalanche.org/v2/public"
MAP_LAYER_URL  = f"{NAC_API_BASE}/products/map-layer"
PRODUCT_URL    = f"{NAC_API_BASE}/product"

ASPECTS         = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ELEVATION_BANDS = ["alpine", "treeline", "below_treeline"]

# Maps location-string words to canonical values
ASPECT_WORD_MAP = {
    "north": "N", "northeast": "NE", "east": "E", "southeast": "SE",
    "south": "S", "southwest": "SW", "west": "W", "northwest": "NW",
}
ELEV_WORD_MAP = {
    "upper": "alpine", "middle": "treeline", "lower": "below_treeline",
}

DANGER_LEVEL_LABELS = {
    -1: "no_rating", 0: "no_rating",
    1: "low", 2: "moderate", 3: "considerable", 4: "high", 5: "extreme",
}
DANGER_COLORS = {
    "no_rating": "#CCCCCC", "low": "#78B943", "moderate": "#F4C025",
    "considerable": "#F58220", "high": "#E3001F", "extreme": "#1A1A1A",
}
DANGER_INT_MAP = {v: k for k, v in DANGER_LEVEL_LABELS.items() if k >= 0}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DangerLevel(IntEnum):
    NO_RATING  = -1
    LOW        = 1
    MODERATE   = 2
    CONSIDERABLE = 3
    HIGH       = 4
    EXTREME    = 5

    @classmethod
    def from_int(cls, value) -> "DangerLevel":
        if value is None:
            return cls.NO_RATING
        try:
            return cls(int(value))
        except (ValueError, TypeError):
            return cls.NO_RATING

    @classmethod
    def from_label(cls, label: str) -> "DangerLevel":
        """Parse a danger label string like 'moderate' or 'considerable'."""
        if not label:
            return cls.NO_RATING
        key = label.lower().strip().replace(" ", "_")
        return cls.from_int(DANGER_INT_MAP.get(key, -1))

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
    value: int           # raw rose value
    danger_level: Optional[DangerLevel] = None


@dataclass
class AvalancheProblem:
    """One avalanche problem from the forecast."""
    rank: int                                  # 1 = primary, 2 = secondary, etc.
    problem_type: str                          # e.g. "Wet Slab", "Persistent Slab"
    aspects: list[str] = field(default_factory=list)        # e.g. ["N", "NE", "NW"]
    elevation_bands: list[str] = field(default_factory=list)  # e.g. ["alpine", "treeline"]
    likelihood: Optional[str] = None          # unlikely|possible|likely|very_likely|almost_certain
    size_min: Optional[float] = None          # D-scale minimum (1–5)
    size_max: Optional[float] = None
    discussion: Optional[str] = None


@dataclass
class AvalancheForecast:
    """
    Normalized forecast from any NAC-platform avalanche center.

    Canonical schema for gap-zone interpolation and map visualization.
    All source-specific parsing produces this object.
    """
    center_id: str
    center_name: str
    zone_name: str
    zone_id: int                               # numeric NAC zone ID (map-layer feature id)
    forecast_id: int
    issued_at: datetime
    expires_at: Optional[datetime]
    valid_for_date: str                        # ISO date YYYY-MM-DD

    # Danger by elevation band (alpine / treeline / below_treeline)
    danger_ratings: list[DangerRating] = field(default_factory=list)

    # Full 24-cell rose (may be empty if not provided)
    danger_rose: list[DangerRoseCell] = field(default_factory=list)

    # Avalanche problems (0–3 per forecast), sorted by rank
    avalanche_problems: list[AvalancheProblem] = field(default_factory=list)

    bottom_line: Optional[str] = None
    travel_advice: Optional[str] = None
    source_url: Optional[str] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def max_danger(self) -> DangerLevel:
        if not self.danger_ratings:
            return DangerLevel.NO_RATING
        return max(r.danger_level for r in self.danger_ratings)

    def danger_for_band(self, band: str) -> Optional[DangerLevel]:
        for r in self.danger_ratings:
            if r.elevation_band == band:
                return r.danger_level
        return None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON export or DataFrame construction."""
        return {
            "center_id":    self.center_id,
            "center_name":  self.center_name,
            "zone_name":    self.zone_name,
            "zone_id":      self.zone_id,
            "forecast_id":  self.forecast_id,
            "issued_at":    self.issued_at.isoformat(),
            "expires_at":   self.expires_at.isoformat() if self.expires_at else None,
            "valid_for_date": self.valid_for_date,
            "bottom_line":  self.bottom_line,
            "travel_advice": self.travel_advice,
            "max_danger":   self.max_danger().label,
            "max_danger_level": int(self.max_danger()),
            "danger_ratings": [
                {
                    "elevation_band": r.elevation_band,
                    "danger_level":   int(r.danger_level),
                    "danger_label":   r.danger_level.label,
                    "color":          r.danger_level.color,
                }
                for r in self.danger_ratings
            ],
            "avalanche_problems": [
                {
                    "rank":            p.rank,
                    "problem_type":    p.problem_type,
                    "aspects":         p.aspects,
                    "elevation_bands": p.elevation_bands,
                    "likelihood":      p.likelihood,
                    "size_min":        p.size_min,
                    "size_max":        p.size_max,
                    "discussion":      p.discussion,
                }
                for p in self.avalanche_problems
            ],
            "source_url":  self.source_url,
            "fetched_at":  self.fetched_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_html(html: Optional[str]) -> Optional[str]:
    """Remove HTML tags and normalize whitespace."""
    if not html:
        return None
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 string to UTC-aware datetime."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logger.warning("Could not parse datetime: %r", dt_str)
        return None


def _parse_danger_ratings(product: dict) -> list[DangerRating]:
    """
    Extract per-elevation danger ratings from a NAC product dict.

    The detail endpoint (/v2/public/product/{id}) returns:
      danger: [
        { "lower": <below_treeline>, "middle": <treeline>, "upper": <alpine>,
          "valid_day": "current" },
        { ..., "valid_day": "tomorrow" },
      ]

    Fallback: if the array has 3 entries indexed by elevation band (old format),
    parse them as alpine/treeline/below_treeline in order.
    """
    danger_list = product.get("danger") or []

    # New format: find the "current" day entry with lower/middle/upper sub-fields
    current = next(
        (d for d in danger_list if d.get("valid_day") == "current"), None
    )
    if current and any(k in current for k in ("lower", "middle", "upper")):
        return [
            DangerRating("alpine",          DangerLevel.from_int(current.get("upper"))),
            DangerRating("treeline",        DangerLevel.from_int(current.get("middle"))),
            DangerRating("below_treeline",  DangerLevel.from_int(current.get("lower"))),
        ]

    # Old / fixture format: 3 entries indexed by elevation band
    ratings = []
    for i, band in enumerate(ELEVATION_BANDS):
        if i < len(danger_list):
            entry = danger_list[i]
            raw = entry.get("upper", entry.get("lower", -1))
        else:
            raw = -1
        ratings.append(DangerRating(band, DangerLevel.from_int(raw)))
    return ratings


def _parse_location(location_list: list[str]) -> tuple[list[str], list[str]]:
    """
    Parse NAC location strings into aspects and elevation bands.

    Each string is "{aspect_word} {elevation_word}", e.g.:
      "northwest upper"  -> aspect="NW",  elevation="alpine"
      "south middle"     -> aspect="S",   elevation="treeline"
      "east lower"       -> aspect="E",   elevation="below_treeline"

    Returns (aspects, elevation_bands) — both sorted canonically.
    """
    aspects: set[str] = set()
    elevations: set[str] = set()
    for loc in (location_list or []):
        parts = loc.lower().split()
        if len(parts) == 2:
            asp_word, elev_word = parts
            if asp_word in ASPECT_WORD_MAP:
                aspects.add(ASPECT_WORD_MAP[asp_word])
            if elev_word in ELEV_WORD_MAP:
                elevations.add(ELEV_WORD_MAP[elev_word])
    return (
        sorted(aspects,    key=lambda a: ASPECTS.index(a) if a in ASPECTS else 99),
        sorted(elevations, key=lambda e: ELEVATION_BANDS.index(e) if e in ELEVATION_BANDS else 99),
    )


def _parse_avalanche_problems(product: dict) -> list[AvalancheProblem]:
    """
    Parse forecast_avalanche_problems from a NAC product detail dict.

    Each problem has:
      rank, name, likelihood, size (list of D-scale strings), location (list of
      "{aspect} {elevation}" strings), discussion (HTML).
    """
    raw_problems = product.get("forecast_avalanche_problems") or []

    problems = []
    for rp in raw_problems:
        aspects, elev_bands = _parse_location(rp.get("location") or [])

        # Size: list of D-scale strings like ["1", "2", "3"]
        size_vals = [float(s) for s in (rp.get("size") or []) if s is not None]
        size_min = min(size_vals) if size_vals else None
        size_max = max(size_vals) if size_vals else None

        likelihood_raw = rp.get("likelihood") or ""
        likelihood = likelihood_raw.lower().replace(" ", "_") if likelihood_raw else None

        problems.append(AvalancheProblem(
            rank         = rp.get("rank", len(problems) + 1),
            problem_type = rp.get("name") or "Unknown",
            aspects      = aspects,
            elevation_bands = elev_bands,
            likelihood   = likelihood,
            size_min     = size_min,
            size_max     = size_max,
            discussion   = _strip_html(rp.get("discussion")),
        ))

    problems.sort(key=lambda p: p.rank)
    return problems


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class ForecastClient:
    """
    Thin HTTP client for the NAC public API.

    All methods return AvalancheForecast objects using the canonical schema.
    In-memory caching avoids redundant calls within the same session.
    """

    def __init__(
        self,
        user_agent: str = "AviGapZone-PhD/0.1 (your@email.edu)",
        timeout: int = 20,
        cache_ttl_seconds: int = 900,     # 15 min — NAC recommendation
        request_delay: float = 0.3,       # seconds between calls to be polite
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds
        self.request_delay = request_delay
        self._cache: dict[str, tuple[float, object]] = {}

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts, _ = self._cache[key]
        return (time.monotonic() - ts) < self.cache_ttl

    def _get_json(self, url: str) -> dict | list:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_zone(self, center_id: str, zone_id: int) -> AvalancheForecast:
        """
        Fetch the current forecast for a single zone.

        Args:
            center_id: NAC center abbreviation, e.g. "BTAC"
            zone_id:   Numeric zone ID (map-layer feature id), e.g. 2852
        """
        center_id = center_id.upper()
        cache_key = f"{center_id}:{zone_id}"

        if self._is_cached(cache_key):
            logger.debug("Cache hit for %s", cache_key)
            return self._cache[cache_key][1]

        url = f"{PRODUCT_URL}?type=forecast&center_id={center_id}&zone_id={zone_id}"
        logger.info("Fetching forecast: %s", url)
        time.sleep(self.request_delay)

        product = self._get_json(url)
        if isinstance(product, list):
            if not product:
                raise ValueError(f"No forecast for {center_id} zone {zone_id}")
            product = product[0]

        if not product:
            raise ValueError(f"Empty forecast response for {center_id} zone {zone_id}")

        # Skip zones where the API returns an all-null shell (no active forecast)
        if not product.get("published_time") and not product.get("created_at"):
            raise ValueError(f"No published forecast for {center_id} zone {zone_id}")

        forecast = self._normalize(product, center_id, zone_id)
        self._cache[cache_key] = (time.monotonic(), forecast)
        return forecast

    def fetch_center(self, center_id: str) -> list[AvalancheForecast]:
        """
        Fetch forecasts for all zones in a center.

        Uses the map-layer to discover zones, then fetches each via fetch_zone.
        Returns a list sorted by zone name.
        """
        center_id = center_id.upper()
        cache_key = f"center:{center_id}"
        if self._is_cached(cache_key):
            return self._cache[cache_key][1]

        url = f"{MAP_LAYER_URL}/{center_id}"
        logger.info("Fetching zone list: %s", url)
        geo = self._get_json(url)

        forecasts = []
        for feature in geo.get("features", []):
            props = feature.get("properties", {})
            if props.get("off_season"):
                continue
            zone_id = feature.get("id")
            if zone_id is None:
                continue
            try:
                fc = self.fetch_zone(center_id, zone_id)
                fc.travel_advice = fc.travel_advice or props.get("travel_advice")
                forecasts.append(fc)
            except Exception as exc:
                logger.error("Failed fetching zone %s/%s: %s", center_id, zone_id, exc)

        forecasts.sort(key=lambda f: f.zone_name)
        self._cache[cache_key] = (time.monotonic(), forecasts)
        return forecasts

    def fetch_all(self) -> list[AvalancheForecast]:
        """
        Fetch forecasts for every active zone across all NAC centers.

        Uses the global map-layer to discover all zones, then fetches each
        individually. Returns forecasts sorted by center_id, then zone_name.
        """
        logger.info("Fetching global zone list from map-layer…")
        geo = self._get_json(MAP_LAYER_URL)

        forecasts = []
        features = geo.get("features", [])
        active = [f for f in features if not f.get("properties", {}).get("off_season")]
        logger.info("%d active zones to fetch (of %d total)", len(active), len(features))

        for feature in active:
            props = feature.get("properties", {})
            center_id = props.get("center_id", "")
            zone_id   = feature.get("id")
            if not center_id or zone_id is None:
                continue
            try:
                fc = self.fetch_zone(center_id, zone_id)
                # Backfill travel_advice from map-layer when detail endpoint omits it
                fc.travel_advice = fc.travel_advice or props.get("travel_advice")
                forecasts.append(fc)
            except Exception as exc:
                logger.error("Failed %s/%s: %s", center_id, zone_id, exc)

        forecasts.sort(key=lambda f: (f.center_id, f.zone_name))
        return forecasts

    def fetch_multiple(self, center_ids: list[str]) -> dict[str, list[AvalancheForecast]]:
        """Fetch all zone forecasts for multiple centers. Returns {center_id: [forecasts]}."""
        results = {}
        for cid in center_ids:
            try:
                results[cid.upper()] = self.fetch_center(cid)
            except Exception as exc:
                logger.error("Failed center %s: %s", cid, exc)
        return results

    # ------------------------------------------------------------------

    def _normalize(self, product: dict, center_id: str, zone_id: int) -> AvalancheForecast:
        """Map a raw NAC product dict to AvalancheForecast."""
        center_id = center_id.upper()

        forecast_id = product.get("id") or -1

        center_name = (
            (product.get("avalanche_center") or {}).get("name")
            or center_id
        )

        zone_list = product.get("forecast_zone") or []
        zone_name = zone_list[0].get("name") if zone_list else "Unknown zone"

        issued_at = _parse_iso(product.get("published_time") or product.get("created_at"))
        if issued_at is None:
            issued_at = datetime.now(timezone.utc)

        expires_at = _parse_iso(product.get("expires_time"))

        valid_for_date = (
            product.get("forecast_date")
            or product.get("start_date", "")[:10]
            or issued_at.date().isoformat()
        )

        bottom_line = _strip_html(
            product.get("bottom_line")
            or product.get("hazard_discussion")
        )

        danger_ratings = _parse_danger_ratings(product)
        avalanche_problems = _parse_avalanche_problems(product)

        travel_advice = _strip_html(product.get("travel_advice")) or None

        return AvalancheForecast(
            center_id          = center_id,
            center_name        = center_name,
            zone_name          = zone_name,
            zone_id            = zone_id,
            forecast_id        = forecast_id,
            issued_at          = issued_at,
            expires_at         = expires_at,
            valid_for_date     = str(valid_for_date),
            danger_ratings     = danger_ratings,
            danger_rose        = [],   # not in detail endpoint; use map-layer for rose
            avalanche_problems = avalanche_problems,
            bottom_line        = bottom_line,
            travel_advice      = travel_advice,
            source_url         = f"{PRODUCT_URL}?type=forecast&center_id={center_id}&zone_id={zone_id}",
        )
