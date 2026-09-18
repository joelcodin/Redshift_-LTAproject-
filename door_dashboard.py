"""
Door dashboard visuals — SVG timeline, risk histogram and a Monte Carlo
reliability simulation for the Streamlit console.

All charts are hand-built SVG so they inherit the console's dark theme.
"""

import numpy as np

import door_pipeline as dp

LABEL_ABNORMAL = dp.LABEL_ABNORMAL

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _p_abnormal(preds):
    """Per-cycle probability of abnormal resistance (model's P(abnormal))."""
    return np.where(
        preds["status"].to_numpy() == LABEL_ABNORMAL,
        preds["confidence"].to_numpy(dtype=float),
        1.0 - preds["confidence"].to_numpy(dtype=float),
    )


def _parse_series(series):
    return [dp.parse_time(s) for s in series]


def _mix(c1, c2, t):
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# 1. Cycle timeline
# ---------------------------------------------------------------------------


def timeline_svg(preds):
    W, H = 1080, 150
    t0s = _parse_series(preds["start_time"])
    t1s = _parse_series(preds["end_time"])
    t0, t1 = min(t0s), max(t1s)
    span = (t1 - t0).total_seconds() or 1.0
    pad_l, pad_r = 24, 24
    plot_w = W - pad_l - pad_r

    bars = []
    for s, e, lab, conf in zip(t0s, t1s, preds["status"], preds["confidence"]):
        x = pad_l + (s - t0).total_seconds() / span * plot_w
        w = max(3.0, (e - s).total_seconds() / span * plot_w)
        color = "#ff2d55" if lab == LABEL_ABNORMAL else "#3a3a46"
        opacity = 0.45 + 0.55 * float(conf)
        bars.append(
            f'<rect x="{x:.1f}" y="50" width="{w:.1f}" height="36" rx="7" fill="{color}" '
            f'opacity="{opacity:.2f}">'
            f"<title>{s:%H:%M:%S} → {e:%H:%M:%S} · {lab} · conf {conf:.0%}</title></rect>"
        )

    ticks = ""
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = pad_l + f * plot_w
        ticks += (
            f'<line x1="{x:.1f}" y1="104" x2="{x:.1f}" y2="108" stroke="#2e2e3a"/>'
            f'<text x="{x:.1f}" y="122" text-anchor="middle" font-size="10" '
            f'fill="#6b6b76">{span * f / 60:.0f}m</text>'
        )
    n_ab = int((preds["status"] == LABEL_ABNORMAL).sum())
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f'<line x1="{pad_l}" y1="104" x2="{W - pad_r}" y2="104" stroke="#26262e"/>'
        f"{ticks}{''.join(bars)}"
        f'<rect x="6" y="4" width="10" height="10" rx="3" fill="#ff2d55"/>'
        f'<text x="22" y="13" font-size="10" fill="#8b8b96">Abnormal ({n_ab})</text>'
        f'<rect x="120" y="4" width="10" height="10" rx="3" fill="#3a3a46"/>'
        f'<text x="136" y="13" font-size="10" fill="#8b8b96">Normal ({len(preds) - n_ab})</text>'
        "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# 2. Risk (P abnormal) histogram
# ---------------------------------------------------------------------------


def risk_hist_svg(preds):
    W, H = 420, 240
    p_ab = _p_abnormal(preds)
    counts, edges = np.histogram(p_ab, bins=10, range=(0.0, 1.0))
    max_c = max(counts.max(), 1)
    pad_l, pad_b, pad_t = 30, 24, 14
    plot_w = W - pad_l - 12
    plot_h = H - pad_t - pad_b
    bar_w = plot_w / 10

    bars = ""
    for i, c in enumerate(counts):
        x = pad_l + i * bar_w
        h = c / max_c * plot_h
        t = edges[i + 1]
        color = _mix((42, 42, 51), (255, 45, 85), t)
        bars += (
            f'<rect x="{x + 2:.1f}" y="{H - pad_b - h:.1f}" width="{bar_w - 4:.1f}" '
            f'height="{h:.1f}" rx="4" fill="{color}">'
            f"<title>{c} cycles · P(abnormal) {edges[i]:.0f}–{edges[i+1]:.0f}</title></rect>"
        )
    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        y = H - pad_b - f * plot_h
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - 12}" y2="{y:.1f}" stroke="#26262e"/>'
    xlabels = ""
    for i, e in enumerate(edges):
        x = pad_l + i * bar_w + bar_w / 2
        xlabels += (
            f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle" font-size="9" '
            f'fill="#6b6b76">{e:.0f}%</text>'
        )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f"{grid}{bars}{xlabels}"
        "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# 3. Monte Carlo reliability simulation
# ---------------------------------------------------------------------------


def run_monte_carlo(preds, n_trials=5000, horizon=1000, seed=42):
    """Simulate `horizon` future cycles `n_trials` times.

    Each simulated cycle re-samples one of the analysed cycles (with its
    model-predicted P(abnormal)) and rolls a Bernoulli trial.
    """
    rng = np.random.default_rng(seed)
    p_ab = _p_abnormal(preds)
    k = len(p_ab)

    idx = rng.integers(0, k, size=(n_trials, horizon))
    p_sample = p_ab[idx]
    faults = rng.random((n_trials, horizon)) < p_sample
    counts = faults.sum(axis=1)

    first = np.where(faults.any(axis=1), faults.argmax(axis=1), horizon)
    n_100 = faults[:, :100].sum(axis=1)

    surv_x = np.arange(0, horizon + 1, max(1, horizon // 200))
    surv_y = np.array([(first > n).mean() for n in surv_x])

    has_fault = first < horizon
    if has_fault.sum() > n_trials // 2:
        median_cycles = float(np.median(first[has_fault]))
        median_cycles_disp = f"{median_cycles:,.0f} cycles"
    else:
        median_cycles = None
        median_cycles_disp = f">{horizon:,} cycles"

    starts = _parse_series(preds["start_time"])
    if len(starts) > 1:
        period_s = float(np.median(np.diff([s.timestamp() for s in starts])))
    else:
        period_s = None
    if median_cycles and period_s:
        secs = median_cycles * period_s
        median_hours = f"{secs / 60:.1f} min" if secs < 3600 else f"{secs / 3600:,.1f} h"
    else:
        median_hours = "—"

    return {
        "n_trials": n_trials,
        "horizon": horizon,
        "counts": counts,
        "mean": float(counts.mean()),
        "p50": float(np.percentile(counts, 50)),
        "p90": float(np.percentile(counts, 90)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "p_ge1_100": float((n_100 > 0).mean()),
        "surv_x": surv_x,
        "surv_y": surv_y,
        "median_cycles": median_cycles,
        "median_cycles_disp": median_cycles_disp,
        "median_hours": median_hours,
        "period_s": period_s,
    }


def _fmt(v):
    return f"{v:,.1f}"


def mc_hist_svg(mc):
    W, H = 1080, 260
    counts = mc["counts"]
    bins = np.histogram(counts, bins=24)[1]
    hist, edges = np.histogram(counts, bins=bins)
    max_c = max(hist.max(), 1)
    pad_l, pad_b, pad_t = 34, 26, 18
    plot_w = W - pad_l - 14
    plot_h = H - pad_t - pad_b
    bar_w = plot_w / len(hist)

    bars = ""
    for i, c in enumerate(hist):
        x = pad_l + i * bar_w
        h = c / max_c * plot_h
        bars += (
            f'<rect x="{x + 1.5:.1f}" y="{H - pad_b - h:.1f}" width="{bar_w - 3:.1f}" '
            f'height="{h:.1f}" rx="3" fill="#ff2d55" opacity="0.85">'
            f"<title>{int(edges[i])}–{int(edges[i+1])} faults: {int(c)} trials</title></rect>"
        )

    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        y = H - pad_b - f * plot_h
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - 14}" y2="{y:.1f}" stroke="#26262e"/>'

    xlabels = ""
    for i, e in enumerate(edges):
        if i % 3 == 0:
            x = pad_l + i * bar_w
            xlabels += (
                f'<text x="{x:.1f}" y="{H - 8}" font-size="9" fill="#6b6b76">{int(e)}</text>'
            )

    def marker(v, color, label):
        x = pad_l + (v - edges[0]) / (edges[-1] - edges[0]) * plot_w
        return (
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{H - pad_b}" '
            f'stroke="{color}" stroke-dasharray="4 3"/>'
            f'<text x="{x + 6:.1f}" y="{pad_t + 10}" font-size="10" fill="{color}">{label}</text>'
        )

    markers = marker(mc["mean"], "#ffb020", f"mean {_fmt(mc['mean'])}") + marker(
        mc["p95"], "#ff6b35", f"P95 {_fmt(mc['p95'])}"
    )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f"{grid}{bars}{markers}{xlabels}"
        f'<text x="{pad_l}" y="{H - 8 + 14}" font-size="9" fill="#6b6b76">faults per {mc["horizon"]:,} cycles</text>'
        "</svg>"
    )
    return svg


def survival_svg(mc):
    W, H = 1080, 260
    pad_l, pad_b, pad_t = 40, 26, 18
    plot_w = W - pad_l - 14
    plot_h = H - pad_t - pad_b
    x, y = mc["surv_x"], mc["surv_y"]
    hx = mc["horizon"]

    pts = []
    for xi, yi in zip(x, y):
        px = pad_l + xi / hx * plot_w
        py = H - pad_b - yi * plot_h
        pts.append(f"{px:.1f},{py:.1f}")
    poly = " ".join(pts)
    area = f"{pad_l},{H - pad_b} " + poly + f" {pad_l + plot_w},{H - pad_b}"

    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        yg = H - pad_b - f * plot_h
        grid += (
            f'<line x1="{pad_l}" y1="{yg:.1f}" x2="{W - 14}" y2="{yg:.1f}" stroke="#26262e"/>'
            f'<text x="{pad_l - 6}" y="{yg + 3:.1f}" text-anchor="end" font-size="9" '
            f'fill="#6b6b76">{f * 100:.0f}%</text>'
        )
    xlabels = ""
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        px = pad_l + f * plot_w
        xlabels += (
            f'<text x="{px:.1f}" y="{H - 8}" text-anchor="middle" font-size="9" '
            f'fill="#6b6b76">{int(f * hx):,}</text>'
        )

    mid_y = H - pad_b - 0.5 * plot_h
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f"{grid}"
        f'<polygon points="{area}" fill="rgba(255,45,85,0.10)"/>'
        f'<polyline points="{poly}" fill="none" stroke="#ff2d55" stroke-width="2.5" stroke-linecap="round"/>'
        f'<line x1="{pad_l}" y1="{mid_y:.1f}" x2="{W - 14}" y2="{mid_y:.1f}" stroke="#ffb020" stroke-dasharray="4 3"/>'
        f'<text x="{pad_l + 8}" y="{mid_y - 6:.1f}" font-size="10" fill="#ffb020">50% survival</text>'
        f"{xlabels}"
        f'<text x="{pad_l}" y="{H - 8 + 14}" font-size="9" fill="#6b6b76">cycles ahead without a fault</text>'
        "</svg>"
    )
    return svg
