"""
Door dashboard visuals — SVG timeline, risk histogram, survival curve and a
Monte Carlo reliability simulation for the Streamlit console.

All charts are hand-built SVG sharing one instrument-panel style: dark slate
grid, signal-red accent, amber markers, mono tick labels.
"""

import datetime

import numpy as np

import door_pipeline as dp

LABEL_ABNORMAL = dp.LABEL_ABNORMAL

# Shared instrument-panel palette -------------------------------------------

GRID = "#191924"
GRID_SOFT = "#14141D"
AXIS = "#2A2A36"
MUTED = "#63636E"
RED = "#FF2D55"
RED_SOFT = "#FF6B81"
AMBER = "#FFB020"
ORANGE = "#FF6B35"
SLATE = "#2E3246"


def _pal(theme):
    if theme == "light":
        return {
            "grid": "#DDE2EC",
            "grid_soft": "#E9EDF5",
            "axis": "#C2CADA",
            "muted": "#64748B",
            "red": "#E11D48",
            "red_soft": "#E11D48",
            "amber": "#D97706",
            "orange": "#EA580C",
            "slate": "#B9C1D2",
            "mix_from": (185, 193, 209),
            "hover": "#0F172A",
        }
    return {
        "grid": GRID,
        "grid_soft": GRID_SOFT,
        "axis": AXIS,
        "muted": MUTED,
        "red": RED,
        "red_soft": RED_SOFT,
        "amber": AMBER,
        "orange": ORANGE,
        "slate": SLATE,
        "mix_from": (46, 50, 70),
        "hover": "#E7E7EF",
    }


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


def _tick_step(span_s, target=7):
    """A 'nice' tick interval (1/2/2.5/5 x 10^k seconds) giving ~target ticks."""
    if span_s <= 0:
        return 60.0
    raw = span_s / max(target, 1)
    mag = 10.0 ** int(np.floor(np.log10(raw)))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


# ---------------------------------------------------------------------------
# 1. Cycle timeline
# ---------------------------------------------------------------------------


def timeline_svg(preds, theme="dark"):
    P = _pal(theme)
    W, H = 1080, 186
    t0s = _parse_series(preds["start_time"])
    t1s = _parse_series(preds["end_time"])
    t0, t1 = min(t0s), max(t1s)
    span = (t1 - t0).total_seconds() or 1.0
    pad_l, pad_r = 28, 28
    plot_w = W - pad_l - pad_r
    band_y, band_h = 44, 36
    axis_y = 122

    def x(t):
        return pad_l + (t - t0).total_seconds() / span * plot_w

    n_ab = int((preds["status"] == LABEL_ABNORMAL).sum())
    n_nm = len(preds) - n_ab

    bars = []
    for sid, s, e, lab, conf in zip(
        preds["segment_id"], t0s, t1s, preds["status"], preds["confidence"]
    ):
        x0 = x(s)
        w = max(2.5, x(e) - x0)
        fill = P["red"] if lab == LABEL_ABNORMAL else P["slate"]
        bars.append(
            f'<rect class="cyc" x="{x0:.1f}" y="{band_y}" width="{w:.1f}" height="{band_h}" rx="3" fill="{fill}">'
            f"<title>{sid} · {s:%H:%M:%S}–{e:%H:%M:%S} · {lab} · conf {conf:.0%}</title></rect>"
        )
    bars = "".join(bars)

    step = _tick_step(span)
    start_ts = t0.timestamp()
    first = (int(start_ts // step) + 1) * step
    ticks = ""
    k = 0
    while True:
        ts = first + k * step
        if ts > t1.timestamp():
            break
        t = datetime.datetime.fromtimestamp(ts)
        px = x(t)
        ticks += (
            f'<line x1="{px:.1f}" y1="{band_y - 10}" x2="{px:.1f}" y2="{axis_y + 5}" stroke="{P["grid"]}"/>'
            f'<text x="{px:.1f}" y="{axis_y + 20}" text-anchor="middle" font-size="10" fill="{P["muted"]}">{t:%H:%M:%S}</text>'
        )
        k += 1

    gap_marks = ""
    for i in range(len(t0s) - 1):
        gap_s = (t0s[i + 1] - t1s[i]).total_seconds()
        if gap_s > dp.GAP_THRESHOLD_MS / 1000.0:
            mid = t1s[i] + datetime.timedelta(seconds=gap_s / 2)
            px = x(mid)
            gap_marks += (
                f'<line x1="{px:.1f}" y1="24" x2="{px:.1f}" y2="{band_y - 8}" stroke="{P["amber"]}" stroke-width="1.6"/>'
                f"<title>cycle gap {gap_s:.0f}s</title>"
            )

    legend = ""
    lx = pad_l
    for label, count, color in (("NORMAL", n_nm, P["slate"]), ("ABNORMAL", n_ab, P["red"])):
        legend += (
            f'<rect x="{lx}" y="12" width="9" height="9" rx="2" fill="{color}"/>'
            f'<text x="{lx + 14}" y="20" font-size="10" fill="{P["muted"]}" letter-spacing="1">{label} {count}</text>'
        )
        lx += 14 + (len(label) + len(str(count))) * 6.2 + 26
    total = (
        f'<text x="{W - pad_r}" y="20" text-anchor="end" font-size="10" fill="{P["muted"]}" '
        f'letter-spacing="1">{len(preds)} CYCLES</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f"<style>rect.cyc:hover{{stroke:{P['hover']};stroke-width:1.2;}}</style>"
        f'<rect x="{pad_l}" y="{band_y - 6}" width="{plot_w}" height="{band_h + 12}" fill="{P["grid_soft"]}" rx="4"/>'
        f'<line x1="{pad_l}" y1="{axis_y}" x2="{W - pad_r}" y2="{axis_y}" stroke="{P["axis"]}"/>'
        f"{ticks}{gap_marks}{bars}{legend}{total}"
        "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# 2. Risk (P abnormal) histogram
# ---------------------------------------------------------------------------


def risk_hist_svg(preds, theme="dark"):
    P = _pal(theme)
    W, H = 420, 248
    p_ab = _p_abnormal(preds)
    counts, edges = np.histogram(p_ab, bins=10, range=(0.0, 1.0))
    max_c = max(int(counts.max()), 1)
    pad_l, pad_b, pad_t, pad_r = 30, 26, 14, 10
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    bar_w = plot_w / 10

    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        y = H - pad_b - f * plot_h
        grid += (
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" stroke="{P["grid"]}"/>'
            f'<text x="{pad_l - 7}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="{P["muted"]}">{max_c * f:.0f}</text>'
        )

    bars = ""
    for i, c in enumerate(counts):
        x = pad_l + i * bar_w
        h = c / max_c * plot_h
        t = edges[i + 1]
        color = _mix(P["mix_from"], (255, 45, 85), t)
        bars += (
            f'<rect x="{x + 2:.1f}" y="{H - pad_b - h:.1f}" width="{bar_w - 4:.1f}" '
            f'height="{max(h, 1.5):.1f}" rx="3" fill="{color}">'
            f"<title>{int(c)} cycles · P(abnormal) {edges[i] * 100:.0f}–{edges[i + 1] * 100:.0f}%</title></rect>"
        )

    mean = float(np.mean(p_ab))
    mx = pad_l + mean * plot_w
    mean_mark = (
        f'<line x1="{mx:.1f}" y1="{pad_t}" x2="{mx:.1f}" y2="{H - pad_b}" stroke="{P["amber"]}" stroke-dasharray="3 3" stroke-width="1.4"/>'
        f'<text x="{mx:.1f}" y="{pad_t + 10}" text-anchor="middle" font-size="9" fill="{P["amber"]}">mean {mean:.2f}</text>'
    )

    xlabels = ""
    for i in range(len(edges) - 1):
        x = pad_l + (i + 0.5) * bar_w
        v = (edges[i] + edges[i + 1]) / 2 * 100
        xlabels += (
            f'<text x="{x:.1f}" y="{H - 8}" text-anchor="middle" font-size="9" fill="{P["muted"]}">{v:.0f}</text>'
        )
    xlabels += (
        f'<text x="{W - pad_r}" y="{H - 8}" text-anchor="end" font-size="9" fill="{P["muted"]}">100%</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f'<line x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" y2="{H - pad_b}" stroke="{P["axis"]}"/>'
        f"{grid}{bars}{mean_mark}{xlabels}"
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
    se = np.sqrt(np.clip(surv_y * (1.0 - surv_y), 0.0, None) / n_trials)
    surv_lo = np.clip(surv_y - 1.96 * se, 0.0, 1.0)
    surv_hi = np.clip(surv_y + 1.96 * se, 0.0, 1.0)

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
        "surv_lo": surv_lo,
        "surv_hi": surv_hi,
        "median_cycles": median_cycles,
        "median_cycles_disp": median_cycles_disp,
        "median_hours": median_hours,
        "period_s": period_s,
    }


def _fmt(v):
    return f"{v:,.1f}"


def mc_hist_svg(mc, theme="dark"):
    P = _pal(theme)
    W, H = 1080, 276
    counts = mc["counts"]
    hist, edges = np.histogram(counts, bins=24)
    max_c = max(int(hist.max()), 1)
    pad_l, pad_b, pad_t, pad_r = 40, 26, 18, 14
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b - 18
    bar_w = plot_w / len(hist)
    mean = float(mc["mean"])

    bars = ""
    for i, c in enumerate(hist):
        x = pad_l + i * bar_w
        h = c / max_c * plot_h
        fill = P["red"] if edges[i] >= mean else P["slate"]
        bars += (
            f'<rect x="{x + 1.5:.1f}" y="{H - pad_b - 18 - h:.1f}" width="{bar_w - 3:.1f}" '
            f'height="{max(h, 1.5):.1f}" rx="3" fill="{fill}" opacity="0.92">'
            f"<title>{int(edges[i])}–{int(edges[i + 1])} faults: {int(c)} trials</title></rect>"
        )

    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        y = H - pad_b - 18 - f * plot_h
        grid += (
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" stroke="{P["grid"]}"/>'
            f'<text x="{pad_l - 7}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="{P["muted"]}">{max_c * f:.0f}</text>'
        )
    grid += (
        f'<line x1="{pad_l}" y1="{H - pad_b - 18:.1f}" x2="{W - pad_r}" y2="{H - pad_b - 18:.1f}" stroke="{P["axis"]}"/>'
    )

    def marker(v, color, label, y_off):
        x = pad_l + (v - edges[0]) / (edges[-1] - edges[0]) * plot_w
        x = min(max(x, pad_l + 4), W - pad_r - 4)
        xl = min(max(x + 6, pad_l + 4), W - pad_r - 62)
        return (
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{H - pad_b - 18}" stroke="{color}" stroke-dasharray="4 3" stroke-width="1.4"/>'
            f'<text x="{xl:.1f}" y="{pad_t + 10 + y_off}" font-size="10" fill="{color}">{label}</text>'
        )

    mean_x = pad_l + (mean - edges[0]) / (edges[-1] - edges[0]) * plot_w
    p95_x = pad_l + (mc["p95"] - edges[0]) / (edges[-1] - edges[0]) * plot_w
    sep = 22 if abs(mean_x - p95_x) < 74 else 0
    markers = marker(mean, P["amber"], f"mean {_fmt(mean)}", 0) + marker(
        mc["p95"], P["orange"], f"P95 {_fmt(mc['p95'])}", sep
    )

    xlabels = ""
    for i in range(len(hist)):
        if i % 3 == 0:
            x = pad_l + (i + 0.5) * bar_w
            v = int(edges[i] + (edges[i + 1] - edges[i]) / 2)
            xlabels += (
                f'<text x="{x:.1f}" y="{H - 26}" text-anchor="middle" font-size="9" fill="{P["muted"]}">{v}</text>'
            )

    caption = (
        f'<text x="{W - pad_r}" y="{H - 8}" text-anchor="end" font-size="9" fill="{P["muted"]}" '
        f'letter-spacing="1">FAULTS PER {mc["horizon"]:,} CYCLES</text>'
    )
    trials = (
        f'<text x="{pad_l}" y="{H - 8}" font-size="9" fill="{P["muted"]}" '
        f'letter-spacing="1">N = {mc["n_trials"]:,} TRIALS</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f"{grid}{bars}{markers}{xlabels}{caption}{trials}"
        "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# 4. Single-cycle detail (current + position vs stroke progress)
# ---------------------------------------------------------------------------


def cycle_detail_svg(seg, meta, theme="dark"):
    """Motor current and door position plotted against stroke progress for one cycle."""
    P = _pal(theme)
    W, H = 560, 300
    pad_l, pad_r, pad_t, pad_b = 44, 16, 30, 26

    pos = np.nan_to_num(seg["Door leaf position"].to_numpy(dtype=float))
    cur = np.nan_to_num(seg["Motor current(mA)"].to_numpy(dtype=float))
    span = pos[-1] - pos[0]
    if abs(span) > 1e-6:
        prog = np.clip((pos - pos[0]) / span, 0.0, 1.0)
    else:
        prog = np.linspace(0, 1, len(pos))

    c_top, c_bot = pad_t, pad_t + (H - pad_t - pad_b) * 0.62
    p_top, p_bot = c_bot + 24, H - pad_b

    def mx(p):
        return pad_l + p * (W - pad_l - pad_r)

    def my(v, top, bot, vmax):
        return bot - (v / vmax) * (bot - top)

    cur_max = max(float(cur.max()), 1.0)
    pos_max = max(float(np.abs(pos).max()), 1.0)

    cur_pts = " ".join(f"{mx(p):.1f},{my(v, c_top, c_bot, cur_max):.1f}" for p, v in zip(prog, cur))
    pos_pts = " ".join(f"{mx(p):.1f},{my(v, p_top, p_bot, pos_max):.1f}" for p, v in zip(prog, pos))

    cur_area = f"{pad_l},{c_bot} " + cur_pts + f" {mx(1.0)},{c_bot}"
    pos_area = f"{pad_l},{p_bot} " + pos_pts + f" {mx(1.0)},{p_bot}"

    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        y = c_top + f * (c_bot - c_top)
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" stroke="{P["grid"]}"/>'
    for f in (0.5, 1.0):
        y = p_top + f * (p_bot - p_top)
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" stroke="{P["grid"]}"/>'
    grid += f'<line x1="{pad_l}" y1="{c_bot:.1f}" x2="{W - pad_r}" y2="{c_bot:.1f}" stroke="{P["axis"]}"/>'
    grid += f'<line x1="{pad_l}" y1="{p_bot:.1f}" x2="{W - pad_r}" y2="{p_bot:.1f}" stroke="{P["axis"]}"/>'

    xlabels = ""
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = mx(f)
        xlabels += (
            f'<text x="{x:.1f}" y="{H - 8}" text-anchor="middle" font-size="9" '
            f'fill="{P["muted"]}">{int(f * 100)}%</text>'
        )

    mid = (
        f'<rect x="{mx(0.15):.1f}" y="{c_top}" width="{mx(0.85) - mx(0.15):.1f}" '
        f'height="{c_bot - c_top:.1f}" fill="rgba(255,45,85,0.045)"/>'
    )

    status_color = P["red_soft"] if meta["status"] == LABEL_ABNORMAL else P["muted"]
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        '<defs><linearGradient id="cd-a" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#FF2D55" stop-opacity="0.22"/>'
        '<stop offset="100%" stop-color="#FF2D55" stop-opacity="0"/></linearGradient></defs>'
        f"{grid}{mid}"
        f'<polygon points="{cur_area}" fill="url(#cd-a)"/>'
        f'<polyline points="{cur_pts}" fill="none" stroke="{P["red"]}" stroke-width="2.2" stroke-linecap="round"/>'
        f'<polygon points="{pos_area}" fill="rgba(255,176,32,0.06)"/>'
        f'<polyline points="{pos_pts}" fill="none" stroke="{P["amber"]}" stroke-width="1.8" stroke-linecap="round"/>'
        f'<text x="{pad_l}" y="{c_top - 10}" font-size="10" fill="{P["red_soft"]}" letter-spacing="1">MOTOR CURRENT (mA)</text>'
        f'<text x="{pad_l}" y="{p_top - 6}" font-size="10" fill="{P["amber"]}" letter-spacing="1">DOOR LEAF POSITION</text>'
        f'<text x="{W - pad_r}" y="{c_top - 10}" text-anchor="end" font-size="10" fill="{status_color}">'
        f"{meta['status'].upper()} · CONF {float(meta['confidence']) * 100:.0f}%</text>"
        f'<text x="{pad_l - 7}" y="{c_top + (c_bot - c_top) * 0.5}" text-anchor="end" font-size="9" fill="{P["muted"]}">{cur_max:.0f}</text>'
        f'<text x="{pad_l - 7}" y="{p_top + 4}" text-anchor="end" font-size="9" fill="{P["muted"]}">{pos_max:.0f}</text>'
        f"{xlabels}"
        "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# 5. Survival curve
# ---------------------------------------------------------------------------


def survival_svg(mc, theme="dark"):
    P = _pal(theme)
    W, H = 1080, 276
    pad_l, pad_b, pad_t, pad_r = 40, 26, 18, 14
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b - 18
    x, y = mc["surv_x"], mc["surv_y"]
    lo, hi = mc["surv_lo"], mc["surv_hi"]
    hx = mc["horizon"]

    def px(v):
        return pad_l + v / hx * plot_w

    def py(v):
        return H - pad_b - 18 - v * plot_h

    def step_path(xs, ys):
        pts = [f"{px(xs[0]):.1f},{py(ys[0]):.1f}"]
        for xi, yp, yi in zip(xs[1:], ys[:-1], ys[1:]):
            pts.append(f"{px(xi):.1f},{py(yp):.1f}")
            pts.append(f"{px(xi):.1f},{py(yi):.1f}")
        return " ".join(pts)

    hi_path = step_path(x, hi)
    lo_path = step_path(x, lo)
    lo_rev = " ".join(lo_path.split(" ")[::-1])
    band = f'<polygon points="{hi_path} {lo_rev}" fill="rgba(255,45,85,0.09)"/>'

    grid = ""
    for f in (0.25, 0.5, 0.75, 1.0):
        yg = H - pad_b - 18 - f * plot_h
        grid += (
            f'<line x1="{pad_l}" y1="{yg:.1f}" x2="{W - pad_r}" y2="{yg:.1f}" stroke="{P["grid"]}"/>'
            f'<text x="{pad_l - 7}" y="{yg + 3:.1f}" text-anchor="end" font-size="9" fill="{P["muted"]}">{f * 100:.0f}%</text>'
        )
    grid += (
        f'<line x1="{pad_l}" y1="{H - pad_b - 18:.1f}" x2="{W - pad_r}" y2="{H - pad_b - 18:.1f}" stroke="{P["axis"]}"/>'
    )

    curve = (
        f'<polyline points="{step_path(x, y)}" fill="none" stroke="{P["red"]}" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    mid_y = py(0.5)
    mid = (
        f'<line x1="{pad_l}" y1="{mid_y:.1f}" x2="{W - pad_r}" y2="{mid_y:.1f}" stroke="{P["amber"]}" stroke-dasharray="4 3" stroke-width="1.3"/>'
        f'<text x="{pad_l + 8}" y="{mid_y - 6:.1f}" font-size="10" fill="{P["amber"]}">50% survival · median {mc["median_cycles_disp"]}</text>'
    )

    xlabels = ""
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        pxv = pad_l + f * plot_w
        xlabels += (
            f'<text x="{pxv:.1f}" y="{H - 26}" text-anchor="middle" font-size="9" '
            f'fill="{P["muted"]}">{int(f * hx):,}</text>'
        )

    caption = (
        f'<text x="{W - pad_r}" y="{H - 8}" text-anchor="end" font-size="9" fill="{P["muted"]}" '
        'letter-spacing="1">CYCLES AHEAD WITHOUT A FAULT</text>'
    )
    ci = (
        f'<text x="{pad_l}" y="{H - 8}" font-size="9" fill="{P["muted"]}" '
        'letter-spacing="1">± 95% CI BAND</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f"{grid}{band}{curve}{mid}{xlabels}{caption}{ci}"
        "</svg>"
    )
    return svg
