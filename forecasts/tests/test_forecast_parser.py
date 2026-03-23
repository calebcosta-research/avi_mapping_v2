"""
tests/test_forecast_parser.py
-----------------------------
Unit tests for the forecast parser using fixture data.
No network calls — all tests run offline against captured API responses.

Run with:
    pytest tests/test_forecast_parser.py -v
"""

import json
import pytest
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from forecast_parser import (
    AvalancheForecast,
    AvalancheProblem,
    DangerLevel,
    DangerRating,
    ForecastClient,
    _decode_danger_rose,
    _parse_danger_ratings,
    _parse_avalanche_problems,
    _parse_iso,
    ASPECTS,
    ELEVATION_BANDS,
)

# ---------------------------------------------------------------------------
# Fixtures — representative NAC API response fragments
# ---------------------------------------------------------------------------

CSAW_PRODUCT_FIXTURE = {
    "id": 99001,
    "published_time": "2024-01-15T07:30:00Z",
    "expires_time": "2024-01-16T07:30:00Z",
    "forecast_date": "2024-01-15",
    "avalanche_center": {"name": "Sierra Avalanche Center"},
    "forecast_zone": [{"name": "Central Sierra Nevada"}],
    "bottom_line": (
        "Considerable avalanche danger exists at all elevations. "
        "A persistent slab problem on north-facing slopes remains the primary concern."
    ),
    "travel_advice": "Be cautious on slopes steeper than 35 degrees.",
    "danger": [
        {"lower": 3, "upper": 3, "valid_day": "current"},  # alpine
        {"lower": 3, "upper": 3, "valid_day": "current"},  # treeline
        {"lower": 2, "upper": 2, "valid_day": "current"},  # below_treeline
    ],
    "avalanche_problems": [
        {
            "name": "persistent slab",
            "aspects": ["N", "NE", "NW"],
            "elevation_bands": ["alpine", "treeline"],
            "likelihood": "likely",
            "size": {"min": 2.0, "max": 3.0},
            "discussion": "A layer of facets from early December persists on north aspects.",
        },
        {
            "name": "wind slab",
            "aspects": ["NE", "E", "SE"],
            "elevation_bands": ["alpine", "treeline"],
            "likelihood": "possible",
            "size": {"min": 1.5, "max": 2.5},
            "discussion": "Recent SW winds loaded leeward aspects near and above treeline.",
        },
    ],
}

ESAC_PRODUCT_FIXTURE = {
    "id": 99002,
    "published_time": "2024-01-15T08:00:00Z",
    "expires_time": "2024-01-16T08:00:00Z",
    "forecast_date": "2024-01-15",
    "avalanche_center": {"name": "Eastern Sierra Avalanche Center"},
    "forecast_zone": [{"name": "Eastern Sierra"}],
    "bottom_line": "Moderate danger. Wind slabs on east aspects near treeline.",
    "danger": [
        {"lower": 2, "upper": 3, "valid_day": "current"},  # alpine
        {"lower": 2, "upper": 2, "valid_day": "current"},  # treeline
        {"lower": 1, "upper": 1, "valid_day": "current"},  # below_treeline
    ],
    "avalanche_problems": [
        {
            "name": "wind_slab",
            "aspects": ["NE", "E", "SE"],
            "elevation_bands": ["treeline"],
            "likelihood": "possible",
            "size": {"min": 1.0, "max": 2.0},
            "discussion": "Wind slabs formed during recent storm on E-facing aspects.",
        }
    ],
}

# A 24-element danger rose: N-facing alpine has value 1, everything else 0
SAMPLE_ROSE_ARRAY = [1, 0, 0, 0, 0, 0, 0, 1,   # alpine:          N...NW
                     1, 1, 0, 0, 0, 0, 0, 1,   # treeline:        N NE...NW
                     0, 0, 0, 0, 0, 0, 0, 0]   # below_treeline:  all zero


# ---------------------------------------------------------------------------
# DangerLevel tests
# ---------------------------------------------------------------------------

class TestDangerLevel:
    def test_from_int_valid(self):
        assert DangerLevel.from_int(3) == DangerLevel.CONSIDERABLE
        assert DangerLevel.from_int(5) == DangerLevel.EXTREME

    def test_from_int_unknown(self):
        assert DangerLevel.from_int(99) == DangerLevel.NO_RATING

    def test_from_int_negative(self):
        assert DangerLevel.from_int(-1) == DangerLevel.NO_RATING

    def test_label(self):
        assert DangerLevel.CONSIDERABLE.label == "considerable"
        assert DangerLevel.LOW.label == "low"

    def test_color_returns_hex(self):
        color = DangerLevel.HIGH.color
        assert color.startswith("#")
        assert len(color) == 7


# ---------------------------------------------------------------------------
# Danger rose decoding
# ---------------------------------------------------------------------------

class TestDecodeRose:
    def test_returns_24_cells(self):
        cells = _decode_danger_rose(SAMPLE_ROSE_ARRAY)
        assert len(cells) == 24

    def test_aspect_order(self):
        cells = _decode_danger_rose(SAMPLE_ROSE_ARRAY)
        alpine_cells = [c for c in cells if c.elevation_band == "alpine"]
        assert [c.aspect for c in alpine_cells] == ASPECTS

    def test_elevation_band_assignment(self):
        cells = _decode_danger_rose(SAMPLE_ROSE_ARRAY)
        assert cells[0].elevation_band == "alpine"
        assert cells[8].elevation_band == "treeline"
        assert cells[16].elevation_band == "below_treeline"

    def test_values_correct(self):
        cells = _decode_danger_rose(SAMPLE_ROSE_ARRAY)
        # Index 0: alpine, N → value 1
        assert cells[0].value == 1
        # Index 1: alpine, NE → value 0
        assert cells[1].value == 0
        # Index 7: alpine, NW → value 1
        assert cells[7].value == 1

    def test_empty_array(self):
        assert _decode_danger_rose([]) == []

    def test_short_array(self):
        assert _decode_danger_rose([1, 2, 3]) == []


# ---------------------------------------------------------------------------
# Danger rating parsing
# ---------------------------------------------------------------------------

class TestParseDangerRatings:
    def test_three_bands_returned(self):
        ratings = _parse_danger_ratings(CSAW_PRODUCT_FIXTURE)
        assert len(ratings) == 3

    def test_band_names(self):
        ratings = _parse_danger_ratings(CSAW_PRODUCT_FIXTURE)
        bands = [r.elevation_band for r in ratings]
        assert bands == ["alpine", "treeline", "below_treeline"]

    def test_danger_levels(self):
        ratings = _parse_danger_ratings(CSAW_PRODUCT_FIXTURE)
        assert ratings[0].danger_level == DangerLevel.CONSIDERABLE  # alpine
        assert ratings[2].danger_level == DangerLevel.MODERATE      # below_treeline

    def test_uses_upper_value(self):
        # ESAC alpine: lower=2, upper=3 → should return 3 (considerable)
        ratings = _parse_danger_ratings(ESAC_PRODUCT_FIXTURE)
        assert ratings[0].danger_level == DangerLevel.CONSIDERABLE

    def test_missing_danger_returns_no_rating(self):
        ratings = _parse_danger_ratings({})
        assert all(r.danger_level == DangerLevel.NO_RATING for r in ratings)


# ---------------------------------------------------------------------------
# Avalanche problem parsing
# ---------------------------------------------------------------------------

class TestParseProblems:
    def test_problem_count(self):
        problems = _parse_avalanche_problems(CSAW_PRODUCT_FIXTURE)
        assert len(problems) == 2

    def test_problem_types_normalized(self):
        problems = _parse_avalanche_problems(CSAW_PRODUCT_FIXTURE)
        assert problems[0].problem_type == "persistent_slab"
        assert problems[1].problem_type == "wind_slab"

    def test_aspects_uppercase(self):
        problems = _parse_avalanche_problems(CSAW_PRODUCT_FIXTURE)
        assert "N" in problems[0].aspects
        assert "NE" in problems[0].aspects

    def test_elevation_bands(self):
        problems = _parse_avalanche_problems(CSAW_PRODUCT_FIXTURE)
        assert "alpine" in problems[0].elevation_bands

    def test_likelihood(self):
        problems = _parse_avalanche_problems(CSAW_PRODUCT_FIXTURE)
        assert problems[0].likelihood == "likely"
        assert problems[1].likelihood == "possible"

    def test_size_values(self):
        problems = _parse_avalanche_problems(CSAW_PRODUCT_FIXTURE)
        assert problems[0].size_min == 2.0
        assert problems[0].size_max == 3.0

    def test_no_problems(self):
        assert _parse_avalanche_problems({}) == []


# ---------------------------------------------------------------------------
# Full normalization via ForecastClient._normalize
# ---------------------------------------------------------------------------

class TestForecastNormalize:
    @pytest.fixture
    def client(self):
        return ForecastClient()

    def test_center_id_uppercased(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "csaw")
        assert fc.center_id == "CSAW"

    def test_center_name(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        assert fc.center_name == "Sierra Avalanche Center"

    def test_zone_name(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        assert fc.zone_name == "Central Sierra Nevada"

    def test_issued_at_parsed(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        assert isinstance(fc.issued_at, datetime)
        assert fc.issued_at.year == 2024

    def test_max_danger(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        assert fc.max_danger() == DangerLevel.CONSIDERABLE

    def test_danger_for_band(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        assert fc.danger_for_band("below_treeline") == DangerLevel.MODERATE
        assert fc.danger_for_band("alpine") == DangerLevel.CONSIDERABLE

    def test_problems_by_aspect(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        n_problems = fc.problems_by_aspect("N")
        assert len(n_problems) == 1
        assert n_problems[0].problem_type == "persistent_slab"

    def test_bottom_line_present(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        assert "persistent slab" in fc.bottom_line.lower()

    def test_to_dict_serializable(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        d = fc.to_dict()
        # Confirm it can round-trip through JSON
        serialized = json.dumps(d)
        rehydrated = json.loads(serialized)
        assert rehydrated["center_id"] == "CSAW"
        assert rehydrated["max_danger"] == "considerable"
        assert len(rehydrated["danger_ratings"]) == 3
        assert len(rehydrated["avalanche_problems"]) == 2

    def test_to_dict_colors_present(self, client):
        fc = client._normalize(CSAW_PRODUCT_FIXTURE, "CSAW")
        d = fc.to_dict()
        for rating in d["danger_ratings"]:
            assert "color" in rating
            assert rating["color"].startswith("#")


# ---------------------------------------------------------------------------
# datetime parsing
# ---------------------------------------------------------------------------

class TestParseIso:
    def test_z_suffix(self):
        dt = _parse_iso("2024-01-15T07:30:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_offset_format(self):
        dt = _parse_iso("2024-01-15T07:30:00+00:00")
        assert dt is not None

    def test_none_input(self):
        assert _parse_iso(None) is None

    def test_empty_string(self):
        assert _parse_iso("") is None

    def test_invalid_string(self):
        assert _parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# Schema completeness — spot-check that all required fields exist
# ---------------------------------------------------------------------------

class TestSchemaCompleteness:
    def test_all_required_fields(self):
        fc = AvalancheForecast(
            center_id="TEST",
            center_name="Test Center",
            zone_name="Test Zone",
            forecast_id=1,
            issued_at=datetime.now(timezone.utc),
            expires_at=None,
            valid_for_date="2024-01-15",
            bottom_line=None,
        )
        # Should not raise
        d = fc.to_dict()
        assert d["center_id"] == "TEST"
        assert d["max_danger"] == "no_rating"
