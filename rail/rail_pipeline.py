"""
Rail Corrugation subsystem pipeline — 3-class classification
(Normal / Side I / Side II) from 1-second axle-box vibration/shock recordings.

Each file: 10,000 rows x 129 cols (speed + 128 channels = 8 cars x 8 bearing
positions x {vibration, shock}). Positions 1,3,5,7 -> Side I, 2,4,6,8 -> Side II.

Features aggregate per (side, sensor type): channel RMS/peak/kurtosis/crest
plus FFT peak magnitude/frequency, so the model learns side-localised
corrugation signatures. Scoring: macro F1.
"""

import numpy as np
import pandas as pd

FS = 10_000.0
LABELS = ("Normal", "Side I", "Side II")


def load_rail_file(path):
    return pd.read_csv(path)


def _kurtosis(x):
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - x.mean()) / std) ** 4) - 3.0)


def _channel_stats(x):
    x = x.astype(float)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2))) or 1e-9
    peak = float(np.max(np.abs(x)))
    fft = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / FS)
    fft[0] = 0.0
    pk = int(np.argmax(fft))
    return {
        "rms": rms,
        "peak": peak,
        "kurt": _kurtosis(x),
        "crest": peak / rms,
        "fpeak": float(freqs[pk]),
        "fmag": float(fft[pk]),
    }


def extract_features(df):
    groups = {}
    for col in df.columns[1:]:
        try:
            head, rest = col.split(" of bearing in position ")
            pos_s, car_s = rest.split(" of car ")
            pos, car = int(pos_s), int(car_s)
        except Exception:
            continue
        typ = "vib" if col.lower().startswith("vibration") else "shock"
        side = "I" if pos % 2 == 1 else "II"
        groups.setdefault((side, typ), []).append(df[col].to_numpy())

    feats = []
    agg = []
    for (side, typ), arrs in sorted(groups.items()):
        stats = [_channel_stats(a) for a in arrs]
        vals = {
            f"{side}_{typ}_rms_mean": float(np.nanmean([s["rms"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_rms_max": float(np.nanmax([s["rms"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_peak_mean": float(np.nanmean([s["peak"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_peak_max": float(np.nanmax([s["peak"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_kurt_mean": float(np.nanmean([s["kurt"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_kurt_max": float(np.nanmax([s["kurt"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_crest_mean": float(np.nanmean([s["crest"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_crest_max": float(np.nanmax([s["crest"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_fpeak_mean": float(np.nanmean([s["fpeak"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_fpeak_max": float(np.nanmax([s["fpeak"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_fmag_mean": float(np.nanmean([s["fmag"] for s in stats])) if stats else 0.0,
            f"{side}_{typ}_fmag_max": float(np.nanmax([s["fmag"] for s in stats])) if stats else 0.0,
        }
        agg.append(vals)
    feats = []
    for v in agg:
        feats.extend(v.values())
    speed = df[df.columns[0]].to_numpy(dtype=float)
    feats.append(float(np.mean(speed)))
    feats.append(float(np.std(speed)))
    feats.append(float(np.mean(np.abs(np.diff(speed)))))
    return np.asarray(feats, dtype=float)


FEATURE_NAMES = (
    [f"{s}_{t}_{m}_{k}" for s in ("I", "II") for t in ("vib", "shock")
     for m in ("rms", "peak", "kurt", "crest", "fpeak", "fmag") for k in ("mean", "max")]
    + ["speed_mean", "speed_std", "speed_transitions"]
)


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1s = []
    for i in range(len(LABELS)):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f1s))


def build_features_df(file_paths, verbose=False):
    rows = []
    for p in file_paths:
        rows.append(extract_features(load_rail_file(p)))
        if verbose:
            print("  featurized", p.split("\\")[-1])
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_inference(folder, model, scaler):
    """Predict the label for every CSV in a folder."""
    import os

    files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".csv")
        and not f.lower().startswith("train_labels")
    )
    paths = [os.path.join(folder, f) for f in files]
    X = build_features_df(paths)
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    proba = model.predict_proba(Xs)
    rows = []
    for f, p, pr in zip(files, preds, proba):
        rows.append(
            {
                "file_id": f,
                "prediction": LABELS[int(p)],
                "confidence": round(float(pr.max()), 4),
            }
        )
    return pd.DataFrame(rows)
