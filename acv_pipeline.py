"""
ACV subsystem pipeline — refrigerant leak localisation (car ranking).

Each case is an xlsx of 30s telemetry from 8 cars. Per-car features combine
temperature/control behaviour with fleet-relative deviations, so a leaking
car (weak cooling) stands out from the other 7 cars in the same file.
Evaluation: leave-one-case-out rank-decay score (official metric).
"""

import numpy as np
import pandas as pd

# parameter name variants across the two file formats
PARAM_ALIASES = {
    "indoor": ["Indoor Average Temperature", "Passenger Cabin Temperature Detected Value"],
    "outdoor": ["Outdoor Average Temperature", "Outside Temperature Sensor Reading",
                "Fresh Air Temperature Detected Value"],
    "cooling_ctrl": ["ACV Control Temperature (Cooling)", "Target Temperature Value"],
    "heating_ctrl": ["ACV Control Temperature (Heating)"],
    "running_mode": ["ACV Running Mode"],
    "setting_mode": ["ACV Setting Mode", "ACV Control Mode"],
    "load_halved": ["Load Halved"],
    "info_valid": ["ACV Information Valid"],
}


def _find_col(df, car, aliases):
    for a in aliases:
        col = f"Car {car} - {a}"
        if col in df.columns:
            return col
    return None


def _numeric(df, col):
    if col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna().astype(float)


def _num_mean(df, col):
    s = _numeric(df, col)
    return float(s.mean()) if len(s) else 0.0


def _num_std(df, col):
    s = _numeric(df, col)
    return float(s.std()) if len(s) > 1 else 0.0


def _entropy(df, col):
    s = df[col].astype(str)
    vc = s.value_counts(normalize=True)
    n = len(vc)
    if n <= 1:
        return 0.0, n, 1.0
    ent = float(-np.sum(vc * np.log(vc)) / np.log(n))
    return ent, n, float(vc.max())


def car_features(df, car):
    cols = {k: _find_col(df, car, v) for k, v in PARAM_ALIASES.items()}
    indoor = _numeric(df, cols["indoor"])
    cooling = _numeric(df, cols["cooling_ctrl"])
    heating = _numeric(df, cols["heating_ctrl"])
    outdoor = _numeric(df, cols["outdoor"])

    err = (indoor - cooling) if len(indoor) and len(cooling) else pd.Series(dtype=float)

    run_ent, run_n, run_top = _entropy(df, cols["running_mode"]) if cols["running_mode"] else (0.0, 0, 0.0)
    set_ent, set_n, set_top = _entropy(df, cols["setting_mode"]) if cols["setting_mode"] else (0.0, 0, 0.0)

    slope = 0.0
    if len(indoor) > 10:
        slope = float(np.polyfit(np.arange(len(indoor)), indoor.to_numpy(), 1)[0])

    feats = {
        "indoor_mean": float(indoor.mean()) if len(indoor) else 0.0,
        "indoor_std": float(indoor.std()) if len(indoor) else 0.0,
        "indoor_min": float(indoor.min()) if len(indoor) else 0.0,
        "indoor_max": float(indoor.max()) if len(indoor) else 0.0,
        "indoor_slope": slope,
        "err_mean": float(err.mean()) if len(err) else 0.0,
        "err_std": float(err.std()) if len(err) else 0.0,
        "err_max": float(err.max()) if len(err) else 0.0,
        "outdoor_mean": float(outdoor.mean()) if len(outdoor) else 0.0,
        "outdoor_std": float(outdoor.std()) if len(outdoor) else 0.0,
        "heating_mean": float(heating.mean()) if len(heating) else 0.0,
        "run_entropy": run_ent,
        "run_n_unique": float(run_n),
        "run_top1": run_top,
        "set_entropy": set_ent,
        "set_n_unique": float(set_n),
        "set_top1": set_top,
        "load_halved_mean": _num_mean(df, cols["load_halved"]),
        "info_valid_mean": _num_mean(df, cols["info_valid"]),
    }
    return feats


FEATURE_NAMES = [
    "indoor_mean", "indoor_std", "indoor_min", "indoor_max", "indoor_slope",
    "err_mean", "err_std", "err_max", "outdoor_mean", "outdoor_std",
    "heating_mean", "run_entropy", "run_n_unique", "run_top1",
    "set_entropy", "set_n_unique", "set_top1", "load_halved_mean", "info_valid_mean",
    "rel_indoor_mean", "rel_err_mean", "rel_outdoor_mean", "rel_indoor_std",
]

REL_MAP = {
    "indoor_mean": "rel_indoor_mean",
    "err_mean": "rel_err_mean",
    "outdoor_mean": "rel_outdoor_mean",
    "indoor_std": "rel_indoor_std",
}


def extract_file_features(path):
    """Return (cars, feature DataFrame) for one case file."""
    df = pd.read_excel(path)
    cars = sorted({c.split(" - ")[0].replace("Car ", "") for c in df.columns[3:] if c.startswith("Car ")})
    rows = [car_features(df, c) for c in cars]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)

    for base, rel in REL_MAP.items():
        med = X[base].median()
        X[rel] = X[base] - med
    return cars, X


def rank_decay_score(true_car, ranked_cars):
    n = len(ranked_cars)
    if true_car not in ranked_cars:
        return 0.0
    r = ranked_cars.index(true_car) + 1
    return (n - (r - 1)) / n


def run_inference(path, model, scaler):
    """Rank cars for one case file. Returns (cars ranked, proba list)."""
    cars, X = extract_file_features(path)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]
    order = np.argsort(-proba)
    ranked = [cars[i] for i in order]
    return ranked, proba
