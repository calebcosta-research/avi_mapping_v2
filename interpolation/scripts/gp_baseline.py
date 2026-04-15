"""
gp_baseline.py
--------------
Gaussian Process baseline model for avalanche danger interpolation.

Fits an independent GP per day on anchor center observations and predicts
danger ratings at gap zone centroids. Used as a comparison baseline
against the full Hierarchical Bayesian model (Phase 2).

Model
-----
  Features:  [lon, lat, elevation_m]  (normalized)
  Target:    danger level 1-5 per elevation band (alpine / treeline / below_treeline)
  Kernel:    RBF + WhiteKernel  (length scale tuned per fit)
  Fit:       one GP per elevation band per day

Input data
----------
  interpolation/data/anchor_forecasts/{center_id}/{YYYY-MM-DD}.json
  interpolation/zones/gap_zones.json  (centroid + elevation range of each gap)

Output
------
  interpolation/data/predictions/{YYYY-MM-DD}.json

  {
    "date": "2026-03-26",
    "model": "gp_baseline",
    "predictions": {
      "lassen-ca": {
        "danger": {
          "alpine":          3.1,
          "treeline":        2.8,
          "below_treeline":  2.1
        },
        "danger_rounded": {
          "alpine":          3,
          "treeline":        3,
          "below_treeline":  2
        },
        "uncertainty": {
          "alpine":          0.6,
          "treeline":        0.5,
          "below_treeline":  0.4
        },
        "anchor_obs": {
          "SAC":  {"alpine": 3, "treeline": 3, "below_treeline": 2},
          "MSAC": {"alpine": 4, "treeline": 3, "below_treeline": 2}
        }
      }
    },
    "diagnostics": {
      "n_anchor_obs": 8,
      "date_coverage": ["SAC", "MSAC", "COAA", ...],
      "kernel_params": { "alpine": {...}, ... }
    }
  }

Usage
-----
    python gp_baseline.py                          # predict today
    python gp_baseline.py --date 2026-03-15        # specific date
    python gp_baseline.py --date-range 2026-01-01 2026-03-26
    python gp_baseline.py --date 2026-03-26 --plot # scatter plot of fit

Dependencies
------------
    pip install scikit-learn numpy
    (shapely already required for identify_gap_zones.py)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

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
ZONES_DIR      = INTERP_DIR / "zones"
ANCHOR_DIR     = INTERP_DIR / "data" / "anchor_forecasts"
OUTPUT_DIR     = INTERP_DIR / "data" / "predictions"

ELEV_BANDS = ["alpine", "treeline", "below_treeline"]

# Nominal elevations (m) used for GP features when predicting at each elevation band
BAND_ELEVATIONS = {
    "alpine":          2900,
    "treeline":        2400,
    "below_treeline":  1900,
}

# Approximate center elevations for GP features (meters).
# Used when SNOTEL elevation data isn't available.
CENTER_ELEVATIONS = {
    "SAC":   2400,
    "MSAC":  2200,
    "COAA":  1800,
    "PAC":   2000,
    "SNFAC": 2100,
    "CAIC":  2800,
    "TAC":   2600,
    "NWAC":  1900,
    "UAC":   2700,
    "GNFAC": 2500,
    "BTAC":  2300,
    "ESAC":  2600,
}

# Approximate center centroids [lon, lat] for GP spatial features
CENTER_CENTROIDS = {
    "SAC":   (-120.5, 39.0),
    "MSAC":  (-121.9, 41.3),
    "COAA":  (-122.5, 43.5),
    "PAC":   (-117.2, 45.7),
    "SNFAC": (-114.4, 43.5),
    "CAIC":  (-106.8, 39.1),
    "TAC":   (-105.6, 36.2),
    "NWAC":  (-121.5, 47.6),
    "UAC":   (-111.5, 40.7),
    "GNFAC": (-111.0, 45.7),
    "BTAC":  (-110.8, 43.5),
    "ESAC":  (-119.1, 37.5),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gap_zones() -> dict:
    with open(GAP_ZONES_PATH) as f:
        return json.load(f)


def load_anchor_obs(target_date: date) -> dict[str, dict]:
    """
    Load all available anchor center observations for a given date.
    Returns {center_id: {band: danger_level}} — only centers with data.
    """
    obs = {}
    if not ANCHOR_DIR.exists():
        return obs

    for center_dir in ANCHOR_DIR.iterdir():
        if not center_dir.is_dir():
            continue
        center_id = center_dir.name
        path = center_dir / f"{target_date.isoformat()}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        summary = data.get("center_summary", {})
        danger  = summary.get("danger_max", {})
        if any(v > 0 for v in danger.values()):
            obs[center_id] = {band: danger.get(band, 0) for band in ELEV_BANDS}

    return obs


# ── Boundary + spatial grid ───────────────────────────────────────────────────

def load_zone_boundary(zone_def: dict):
    """
    Load the park boundary polygon for a zone that has a boundary_file.
    Returns a shapely geometry or None.
    """
    boundary_file = zone_def.get("boundary_file")
    if not boundary_file:
        return None
    path = ZONES_DIR / boundary_file
    if not path.exists():
        log.warning("Boundary file not found: %s", path)
        return None
    try:
        import json as _json
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union
        with open(path) as f:
            gj = _json.load(f)
        geoms = [shapely_shape(feat["geometry"]) for feat in gj["features"]]
        return unary_union(geoms)
    except Exception as e:
        log.warning("Failed to load boundary %s: %s", path, e)
        return None


def generate_grid_cells(boundary_geom, resolution_deg: float = 0.015) -> list[dict]:
    """
    Generate a regular grid of cells within a park boundary polygon.
    Each cell is a GeoJSON-ready dict with centroid lon/lat for GP prediction.
    Only cells whose centroid falls within the boundary are kept.

    Returns list of {lon, lat, geom_coords} dicts.
    """
    try:
        from shapely.geometry import Point
    except ImportError:
        log.error("shapely required for spatial grid — pip install shapely")
        return []

    minx, miny, maxx, maxy = boundary_geom.bounds
    half = resolution_deg / 2.0
    cells = []
    y = miny + half
    while y < maxy:
        x = minx + half
        while x < maxx:
            pt = Point(x, y)
            if boundary_geom.contains(pt):
                # Build cell square coords
                coords = [
                    [x - half, y - half],
                    [x + half, y - half],
                    [x + half, y + half],
                    [x - half, y + half],
                    [x - half, y - half],
                ]
                cells.append({"lon": x, "lat": y, "coords": coords})
            x += resolution_deg
        y += resolution_deg

    log.info("Generated %d grid cells (res=%.3f°) within boundary", len(cells), resolution_deg)
    return cells


def build_grid_features(cells: list[dict], elev_m: float) -> np.ndarray:
    """Feature matrix [lon, lat, elev] for a list of grid cells at a fixed elevation."""
    return np.array([[c["lon"], c["lat"], elev_m] for c in cells], dtype=float)


def predict_spatial_grid(
    cells:      list[dict],
    band:       str,
    X_anchors_n: np.ndarray,
    y:          np.ndarray,
    mu:         np.ndarray,
    sig:        np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GP prediction at grid cell centroids for one elevation band.
    Returns (mean, std) arrays aligned with cells.
    """
    elev_m = BAND_ELEVATIONS[band]
    X_grid = build_grid_features(cells, elev_m)
    X_grid_n = (X_grid - mu) / sig
    try:
        mean, std = fit_and_predict(X_anchors_n, y, X_grid_n)
        mean = np.clip(mean, 1.0, 5.0)
    except Exception as e:
        log.warning("Spatial GP failed for band %s: %s — using IDW fallback", band, e)
        mean = np.full(len(cells), float(y.mean()))
        std  = np.full(len(cells), float(y.std()))
    return mean, std


def build_spatial_geojson(cells: list[dict], preds: dict) -> dict:
    """
    Build a GeoJSON FeatureCollection from grid cells with per-band danger predictions.
    preds = {band: (mean_arr, std_arr)}
    """
    features = []
    n = len(cells)
    for i in range(n):
        props = {}
        for band, (mean_arr, std_arr) in preds.items():
            danger_raw  = float(mean_arr[i])
            danger_int  = int(round(danger_raw))
            props[f"danger_{band}"]     = danger_raw
            props[f"danger_{band}_int"] = danger_int
            props[f"unc_{band}"]        = round(float(std_arr[i]), 2)

        # Primary display: alpine danger (or treeline if alpine is 0)
        d_alp = props.get("danger_alpine_int", 0)
        d_tl  = props.get("danger_treeline_int", 0)
        props["danger_display"] = d_alp if d_alp > 0 else d_tl

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [cells[i]["coords"]],
            },
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(center_ids: list[str]) -> np.ndarray:
    """
    Build feature matrix [lon, lat, elev_m] for a list of center IDs.
    Falls back to defaults for unknown centers.
    """
    rows = []
    for cid in center_ids:
        lon, lat = CENTER_CENTROIDS.get(cid, (0.0, 0.0))
        elev     = CENTER_ELEVATIONS.get(cid, 2000)
        rows.append([lon, lat, elev])
    return np.array(rows, dtype=float)


def build_gap_features(zones: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix for gap zone prediction points.
    Uses zone centroid + midpoint of elevation range.
    Returns (X, zone_ids).
    """
    rows     = []
    zone_ids = []
    for z in zones:
        lon, lat  = z["centroid"]
        elev_min, elev_max = z.get("elevation_range_m", [2000, 3000])
        elev_mid  = (elev_min + elev_max) / 2.0
        rows.append([lon, lat, elev_mid])
        zone_ids.append(z["zone_id"])
    return np.array(rows, dtype=float), zone_ids


def normalize_features(X_train: np.ndarray, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance normalization fitted on training data."""
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sig, (X_pred - mu) / sig


# ── GP fit + predict ──────────────────────────────────────────────────────────

def fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a GP on (X_train, y_train) and predict at X_pred.
    Returns (mean_pred, std_pred).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42,
    )
    gp.fit(X_train, y_train)
    mean, std = gp.predict(X_pred, return_std=True)
    return mean, std


def run_gp_for_date(
    target_date: date,
    gap_zones:   list[dict],
    obs:         dict[str, dict],
) -> Optional[dict]:
    """
    Run the GP model for one date.
    Returns prediction dict or None if insufficient data.
    """
    center_ids = list(obs.keys())
    if len(center_ids) < 2:
        log.warning("Only %d anchor center(s) with data for %s — skipping", len(center_ids), target_date)
        return None

    X_anchors          = build_features(center_ids)
    X_gaps, gap_ids    = build_gap_features(gap_zones)

    # Fit normalization on anchor features (reused for spatial grid predictions too)
    mu  = X_anchors.mean(axis=0)
    sig = X_anchors.std(axis=0) + 1e-8
    X_anchors_n = (X_anchors - mu) / sig
    X_gaps_n    = (X_gaps    - mu) / sig

    predictions: dict[str, dict] = {z: {"danger": {}, "uncertainty": {}, "anchor_obs": {}} for z in gap_ids}
    kernel_params: dict[str, dict] = {}

    # Per-band GP fit at centroid points
    for band in ELEV_BANDS:
        y = np.array([obs[cid][band] for cid in center_ids], dtype=float)

        if y.max() == 0:
            for z in gap_ids:
                predictions[z]["danger"][band]      = 0.0
                predictions[z]["uncertainty"][band] = 0.0
            continue

        try:
            mean, std = fit_and_predict(X_anchors_n, y, X_gaps_n)
        except Exception as e:
            log.warning("GP fit failed for band %s: %s", band, e)
            mean = np.full(len(gap_ids), float(y.mean()))
            std  = np.full(len(gap_ids), float(y.std()))

        mean = np.clip(mean, 1.0, 5.0)

        for i, zone_id in enumerate(gap_ids):
            predictions[zone_id]["danger"][band]      = round(float(mean[i]), 2)
            predictions[zone_id]["uncertainty"][band] = round(float(std[i]),  2)

        kernel_params[band] = {}

    # Rounded centroid predictions + anchor obs
    for zone_id in gap_ids:
        predictions[zone_id]["danger_rounded"] = {
            band: int(round(predictions[zone_id]["danger"][band]))
            for band in ELEV_BANDS
        }
        zone_def = next((z for z in gap_zones if z["zone_id"] == zone_id), None)
        if zone_def:
            for anchor in zone_def.get("anchor_centers", []):
                cid = anchor["center_id"]
                if cid in obs:
                    predictions[zone_id]["anchor_obs"][cid] = obs[cid]

    # Spatial grid predictions for zones with a boundary file
    for zone_def in gap_zones:
        zone_id = zone_def["zone_id"]
        if zone_id not in predictions:
            continue
        boundary_geom = load_zone_boundary(zone_def)
        if boundary_geom is None:
            continue

        resolution = zone_def.get("grid_resolution_deg", 0.015)
        cells = generate_grid_cells(boundary_geom, resolution)
        if not cells:
            continue

        grid_preds = {}
        for band in ELEV_BANDS:
            y_band = np.array([obs[cid][band] for cid in center_ids], dtype=float)
            if y_band.max() == 0:
                grid_preds[band] = (
                    np.zeros(len(cells)),
                    np.zeros(len(cells)),
                )
                continue
            mean_g, std_g = predict_spatial_grid(
                cells, band, X_anchors_n, y_band, mu, sig
            )
            grid_preds[band] = (mean_g, std_g)

        predictions[zone_id]["spatial_grid"] = build_spatial_geojson(cells, grid_preds)
        log.info("  [%s] spatial grid: %d cells", zone_id, len(cells))

    return {
        "date":        target_date.isoformat(),
        "model":       "gp_baseline",
        "predictions": predictions,
        "diagnostics": {
            "n_anchor_obs":   len(center_ids),
            "date_coverage":  sorted(center_ids),
            "kernel_params":  kernel_params,
        },
    }


# ── Output ────────────────────────────────────────────────────────────────────

def save_predictions(target_date: date, result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{target_date.isoformat()}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved predictions -> %s", path)


def print_predictions(result: dict) -> None:
    print(f"\nGP Baseline predictions for {result['date']}")
    print(f"Anchor centers: {', '.join(result['diagnostics']['date_coverage'])}\n")
    print(f"{'Zone':<25} {'Alpine':>8} {'Treeline':>10} {'Below':>8}  {'Uncertainty (alp)':>18}")
    print("-" * 75)
    for zone_id, pred in result["predictions"].items():
        dr = pred["danger_rounded"]
        unc = pred["uncertainty"]
        print(
            f"{zone_id:<25} "
            f"{dr.get('alpine', 0):>8}  "
            f"{dr.get('treeline', 0):>8}  "
            f"{dr.get('below_treeline', 0):>8}  "
            f"  ±{unc.get('alpine', 0):.2f}"
        )


def plot_fit(result: dict, obs: dict[str, dict], gap_zones: list[dict]) -> None:
    """Quick scatter plot of GP fit for visual QC (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"GP Baseline — {result['date']}")

    for ax, band in zip(axes, ELEV_BANDS):
        # Anchor obs
        for cid, danger in obs.items():
            lon, lat = CENTER_CENTROIDS.get(cid, (0, 0))
            ax.scatter(lon, lat, c=[danger[band]], vmin=1, vmax=5,
                       cmap="RdYlGn_r", s=100, marker="^", zorder=3,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(cid, (lon, lat), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)

        # Gap zone predictions
        for z in gap_zones:
            zone_id = z["zone_id"]
            pred    = result["predictions"].get(zone_id, {})
            d       = pred.get("danger_rounded", {}).get(band, 0)
            lon, lat = z["centroid"]
            ax.scatter(lon, lat, c=[d], vmin=1, vmax=5,
                       cmap="RdYlGn_r", s=200, marker="*", zorder=4,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(f"{zone_id}\n({d})", (lon, lat),
                        textcoords="offset points", xytext=(4, -10), fontsize=7)

        ax.set_title(band)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GP baseline avalanche danger interpolation")
    parser.add_argument("--date",       type=str, default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--date-range", nargs=2, metavar=("START", "END"),
                        help="Date range YYYY-MM-DD YYYY-MM-DD")
    parser.add_argument("--plot",       action="store_true",
                        help="Show matplotlib scatter plot of fit")
    parser.add_argument("--no-save",    action="store_true",
                        help="Print results but don't save to disk")
    args = parser.parse_args()

    gap_data  = load_gap_zones()
    gap_zones = gap_data["zones"]

    if args.date_range:
        start = date.fromisoformat(args.date_range[0])
        end   = date.fromisoformat(args.date_range[1])
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    else:
        dates = [date.fromisoformat(args.date or date.today().isoformat())]

    for d in dates:
        obs = load_anchor_obs(d)
        log.info("%s: %d anchor center(s) with data", d, len(obs))

        if not obs:
            log.warning("No anchor data for %s — run fetch_anchor_forecasts.py first", d)
            continue

        result = run_gp_for_date(d, gap_zones, obs)
        if result is None:
            continue

        print_predictions(result)

        if not args.no_save:
            save_predictions(d, result)

        if args.plot:
            plot_fit(result, obs, gap_zones)


if __name__ == "__main__":
    main()
