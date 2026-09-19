"""
SHM subsystem pipeline — cumulative fatigue damage regression.

Features are a lightweight rainflow-style decomposition of each long dynamic
stress series: cycle-range histogram (log bins), damage proxies (sum of
range^m, matching Miner's rule N = C / sigma^m), signal statistics and
exceedance counts. A tree regressor is trained on log(damage) and its
prediction exponentiated, which suits the relative-error (MAPE) scoring.

Scoring: score = max(0, 1 - MAPE).
"""

import os

import numpy as np
import pandas as pd

N_RANGE_BINS = 48
RANGE_MIN = 1e-4
DAMAGE_EXPONENTS = (2.0, 3.0, 4.0, 5.0)
EXCEED_Q = (0.9, 0.99, 0.999, 0.9999)


def load_shm_file(path):
    """Return the single dynamic-stress channel as a 1-D float array."""
    df = pd.read_csv(path)
    return df.iloc[:, 0].to_numpy(dtype=float)


def _cycle_ranges(x):
    """Approximate rainflow ranges from successive extrema (half-cycles)."""
    d = np.diff(x)
    s = pd.Series(np.sign(d)).replace(0.0, np.nan).ffill().fillna(0.0).to_numpy()
    pos = np.flatnonzero(s[1:] != s[:-1])
    if len(pos) < 2:
        return np.array([0.0])
    extrema = x[pos + 1]
    return np.abs(np.diff(extrema))


def extract_features(x):
    r = _cycle_ranges(x)
    feats = [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.sqrt(np.mean(x**2))),
        float(np.max(np.abs(x))),
        float(np.ptp(x)),
        float(pd.Series(x).skew()),
        float(pd.Series(x).kurt()),
        float(np.mean(np.abs(x))),
    ]
    rmax = max(float(r.max()), RANGE_MIN)
    bins = np.logspace(np.log10(RANGE_MIN), np.log10(rmax), N_RANGE_BINS + 1)
    hist, _ = np.histogram(r, bins=bins)
    n_cycles = max(len(r), 1)
    feats.extend((hist / n_cycles).tolist())
    for m in DAMAGE_EXPONENTS:
        feats.append(float(np.log1p(np.mean(r**m))))
    ax = np.abs(x)
    for q in EXCEED_Q:
        thr = float(np.quantile(ax, q))
        feats.append(float(np.mean(ax > thr)))
    feats.append(float(np.mean(np.sign(x[1:]) != np.sign(x[:-1]))))
    return np.asarray(feats, dtype=float)


FEATURE_NAMES = (
    ["mean", "std", "rms", "max_abs", "ptp", "skew", "kurt", "mean_abs"]
    + [f"range_bin_{i}" for i in range(N_RANGE_BINS)]
    + [f"damage_m{int(m)}" for m in DAMAGE_EXPONENTS]
    + [f"exceed_{q}" for q in EXCEED_Q]
    + ["zero_cross_rate"]
)


def mape_score(true, pred):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mape = np.mean(np.abs(true - pred) / np.abs(true))
    return max(0.0, 1.0 - float(mape)), float(mape)


def build_features_df(file_paths, verbose=False):
    rows = []
    for p in file_paths:
        rows.append(extract_features(load_shm_file(p)))
        if verbose:
            print("  featurized", os.path.basename(p))
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def run_inference(folder, model, scaler, log_target=True):
    """Predict damage for every CSV in a folder. Returns a predictions DataFrame."""
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
    paths = [os.path.join(folder, f) for f in files]
    X = build_features_df(paths)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)
    if log_target:
        pred = np.expm1(pred)
    return pd.DataFrame({"file_id": files, "prediction": np.clip(pred, 0.0, None)})
