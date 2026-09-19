"""
Door subsystem pipeline — segmentation + classification for the Door Fault
Diagnosis dataset (temporal segment detection problem).

Shared by train_door.py, predict.py and the Streamlit console.

Segmentation
------------
The stream is a sequence of door cycles separated by irregular time gaps
(observed gaps >= 10.2 s, intra-cycle sampling 20 ms). Primary split is on
gaps > 5 s. As a safety net, any motion-flag rise (Door is opening /
Door is closing going 0 -> 1) inside a chunk is treated as an extra cycle
boundary, which also handles back-to-back cycles with no gap.

Classification
--------------
Per-cycle features are extracted from motor current/voltage/back-EMF and the
door-position stroke profile, then classified by a GradientBoosting model.
"""

import datetime

import numpy as np
import pandas as pd

GAP_THRESHOLD_MS = 5000.0
POS_MAX = 705.0
N_STROKE_BINS = 10
MID_STROKE = (0.15, 0.85)

LABEL_NORMAL = "Normal"
LABEL_ABNORMAL = "Abnormal resistance"


def parse_time(value):
    y, mo, d, h, mi, se, ms = [int(x) for x in str(value).split("-")]
    return datetime.datetime(y, mo, d, h, mi, se, ms * 1000)


def format_time(t):
    return f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}-{t.microsecond // 1000}"


def load_stream(path_or_buf):
    df = pd.read_csv(path_or_buf)
    df["t"] = df["Datetime"].apply(parse_time)
    df["dt_ms"] = df["t"].diff().dt.total_seconds() * 1000.0
    return df.reset_index(drop=True)


def segment_stream(df):
    """Return list of (start_idx, end_idx) inclusive row spans, one per cycle."""
    dt = df["dt_ms"].to_numpy()
    opens = df["Door is opening"].to_numpy()
    closes = df["Door is closing"].to_numpy()
    n = len(df)

    chunks = []
    i0 = 0
    for i in range(1, n):
        if dt[i] > GAP_THRESHOLD_MS:
            chunks.append((i0, i - 1))
            i0 = i
    chunks.append((i0, n - 1))

    segments = []
    for c0, c1 in chunks:
        starts = [c0]
        for i in range(c0 + 1, c1 + 1):
            if (opens[i] == 1 and opens[i - 1] == 0) or (closes[i] == 1 and closes[i - 1] == 0):
                starts.append(i)
        for k, s in enumerate(starts):
            e = starts[k + 1] - 1 if k + 1 < len(starts) else c1
            segments.append((s, e))
    return segments


def _bin_means(values, progress, n_bins):
    bins = np.zeros(n_bins)
    idx = np.clip((progress * n_bins).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        bins[b] = values[m].mean() if m.any() else 0.0
    return bins


def segment_features(seg):
    """Fixed-order feature vector for one cycle segment (a DataFrame slice)."""
    cur = seg["Motor current(mA)"].to_numpy(dtype=float)
    volt = seg["Motor Voltage(10mV)"].to_numpy(dtype=float)
    emf = seg["Motor electrodynamic force"].to_numpy(dtype=float)
    pos = seg["Door leaf position"].to_numpy(dtype=float)
    op_time = seg["Door opening time(.1s)"].to_numpy(dtype=float)
    cl_time = seg["Door closing time(.1s)"].to_numpy(dtype=float)

    pos_span = pos[-1] - pos[0]
    op_open = 1.0 if pos_span > 0 else 0.0
    progress = (pos - pos[0]) / pos_span if abs(pos_span) > 1e-6 else np.zeros_like(pos)
    velocity = np.abs(np.diff(pos, prepend=pos[0]))

    mid_mask = (progress >= MID_STROKE[0]) & (progress <= MID_STROKE[1])

    feats = [
        op_open,
        float(len(seg)),
        float(np.mean(cur)), float(np.median(cur)), float(np.std(cur)),
        float(np.max(cur)), float(np.quantile(cur, 0.95)), float(np.quantile(cur, 0.10)),
        float(np.mean(volt)), float(np.std(volt)), float(np.max(volt)),
        float(np.mean(emf)), float(np.std(emf)),
        float(np.mean(velocity)), float(np.max(velocity)),
        float(np.mean(op_time)), float(np.mean(cl_time)),
        float(np.mean(cur[mid_mask])) if mid_mask.any() else float(np.mean(cur)),
        float(np.max(cur[mid_mask])) if mid_mask.any() else float(np.max(cur)),
    ]
    feats.extend(_bin_means(cur, progress, N_STROKE_BINS).tolist())
    return np.asarray(feats, dtype=float)


FEATURE_NAMES = (
    ["op_open", "n_rows"]
    + [f"cur_{s}" for s in ["mean", "median", "std", "max", "p95", "p10"]]
    + [f"volt_{s}" for s in ["mean", "std", "max"]]
    + ["emf_mean", "emf_std"]
    + ["vel_mean", "vel_max"]
    + ["op_time_mean", "cl_time_mean"]
    + ["mid_cur_mean", "mid_cur_max"]
    + [f"bin_{i}" for i in range(N_STROKE_BINS)]
)


def _as_dt(series):
    return series.apply(parse_time)


def score_iou_weighted_f1(true_df, pred_df):
    """IoU-weighted F1 exactly as described in the dataset documentation."""
    t = true_df.rename(columns={"start_time": "s", "end_time": "e"})[
        ["s", "e", "status"]
    ].reset_index(drop=True)
    p = pred_df.rename(columns={"start_time": "s", "end_time": "e"})[
        ["s", "e", "prediction"]
    ].reset_index(drop=True)
    t["s"] = _as_dt(t["s"])
    t["e"] = _as_dt(t["e"])
    p["s"] = _as_dt(p["s"])
    p["e"] = _as_dt(p["e"])

    pairs = []
    for ti, row_t in t.iterrows():
        for pi, row_p in p.iterrows():
            if row_t["status"] != row_p["prediction"]:
                continue
            inter = max(
                0.0,
                (min(row_t["e"], row_p["e"]) - max(row_t["s"], row_p["s"])).total_seconds(),
            )
            union = (
                (row_t["e"] - row_t["s"]).total_seconds()
                + (row_p["e"] - row_p["s"]).total_seconds()
                - inter
            )
            iou = inter / union if union > 0 else 0.0
            if iou > 0:
                pairs.append((iou, ti, pi))

    pairs.sort(key=lambda x: -x[0])
    used_t, used_p = set(), set()
    matches = []
    for iou, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matches.append(iou)

    sum_iou = sum(matches)
    soft_recall = sum_iou / len(t) if len(t) else 0.0
    soft_precision = sum_iou / len(p) if len(p) else 0.0
    if soft_recall + soft_precision > 0:
        score = 2 * soft_recall * soft_precision / (soft_recall + soft_precision)
    else:
        score = 0.0

    return {
        "score": score,
        "soft_recall": soft_recall,
        "soft_precision": soft_precision,
        "n_true": len(t),
        "n_pred": len(p),
        "n_matched": len(matches),
    }


def build_features_df(df, segments):
    rows = []
    for s, e in segments:
        rows.append(segment_features(df.iloc[s : e + 1]))
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def run_inference(df, model, scaler, conf_threshold=None):
    """Segment + classify a stream. Returns a predictions DataFrame."""
    segments = segment_stream(df)
    X = build_features_df(df, segments)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]
    preds = np.where(model.predict(Xs) == 1, LABEL_ABNORMAL, LABEL_NORMAL)

    rows = []
    for i, ((s, e), label, p_ab) in enumerate(zip(segments, preds, proba), start=1):
        conf = p_ab if label == LABEL_ABNORMAL else 1.0 - p_ab
        if conf_threshold is not None and 0.0 < conf < conf_threshold:
            label = LABEL_NORMAL
            conf = 1.0 - p_ab
        pos_start = float(df.loc[s, "Door leaf position"])
        pos_end = float(df.loc[e, "Door leaf position"])
        operation = "Open" if pos_end > pos_start else "Close"
        rows.append(
            {
                "segment_id": f"seg_{i:03d}",
                "start_time": format_time(df.loc[s, "t"]),
                "end_time": format_time(df.loc[e, "t"]),
                "operation": operation,
                "status": label,
                "n_rows": int(e - s + 1),
                "confidence": round(float(conf), 4),
            }
        )
    return pd.DataFrame(rows)
