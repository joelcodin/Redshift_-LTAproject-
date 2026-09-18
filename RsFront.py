"""
Train Condition Monitoring — Frontend shell (design only)

Run with:
    streamlit run RsFront.py

Instrument console: near-black drafting-grid base with signal-red accents and
mono data type, working upload console with a prediction-model picker and
card-based results. All four subsystems run real models: Door (segment +
classify + Monte Carlo), ACV (leak localisation), Rail Corrugation (3-class)
and SHM (fatigue damage regression).
"""

import io
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import acv_pipeline as acvp
import door_dashboard as dd
import door_pipeline as dp
import rail_pipeline as rp
import shm_pipeline as sp

SUBSYSTEMS = [
    ("Door", "[01]", "CYCLE TIMING · OBSTRUCTION EVENTS"),
    ("ACV", "[02]", "REFRIGERANT LEAK LOCALISATION"),
    ("Rail Corrugation", "[03]", "AXLE-BOX VIBRATION · CORRUGATION TYPE"),
    ("SHM", "[04]", "DYNAMIC STRESS · FATIGUE DAMAGE"),
]

SUBSYSTEM_BUNDLES = {
    "Door": "door_models.joblib",
    "ACV": "acv_models.joblib",
    "Rail Corrugation": "rail_models.joblib",
    "SHM": "shm_models.joblib",
}

ICONS = {
    "activity": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "gauge": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "bar": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "pulse": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "trend": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
}


@st.cache_resource
def load_bundle(bundle_file):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), bundle_file)
    return joblib.load(path)


def run_door_inference(uploaded_file, model_name):
    bundle = load_bundle(SUBSYSTEM_BUNDLES["Door"])
    entry = bundle["models"][model_name]
    df = dp.load_stream(io.BytesIO(uploaded_file.getvalue()))
    return dp.run_inference(df, entry["model"], entry["scaler"])


def chart_frame(svg_html, height):
    """Render an SVG chart inside an iframe so it always displays."""
    return components.html(
        "<style>body{margin:0;background:transparent;}"
        "svg{width:100%;height:100%;display:block;}"
        "svg text{font-family:'IBM Plex Mono',Consolas,Menlo,monospace;}"
        "</style>" + svg_html,
        height=height,
        scrolling=False,
    )


def _theme_colors(theme):
    if theme == "light":
        return {
            "grid": "rgba(15,23,42,0.10)",
            "text": "#64748B",
            "red": "#E11D48",
            "amber": "#D97706",
            "bar_track": "rgba(15,23,42,0.08)",
        }
    return {
        "grid": "rgba(255,255,255,0.06)",
        "text": "#63636E",
        "red": "#FF2D55",
        "amber": "#FFB020",
        "bar_track": "rgba(255,255,255,0.06)",
    }


def shm_signal_svg(x, theme="dark"):
    c = _theme_colors(theme)
    n = min(1500, len(x))
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    y = x[idx]
    lim = float(np.quantile(np.abs(y), 0.999)) or 1.0
    W, H = 900, 220
    pad_l, pad_r, pad_t, pad_b = 40, 14, 18, 22
    pts = []
    for i, v in enumerate(y):
        px = pad_l + i / (n - 1) * (W - pad_l - pad_r)
        py = H - pad_b - (v + lim) / (2 * lim) * (H - pad_t - pad_b)
        pts.append(f"{px:.1f},{py:.1f}")
    poly = " ".join(pts)
    zero_y = H - pad_b - lim / (2 * lim) * (H - pad_t - pad_b)
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{W - pad_r}" y2="{zero_y:.1f}" stroke="{c["grid"]}"/>'
        f'<polyline points="{poly}" fill="none" stroke="{c["red"]}" stroke-width="1.4"/>'
        f'<text x="{pad_l}" y="{pad_t - 6}" font-size="10" fill="{c["text"]}">dynamic stress (downsampled)</text>'
        f'<text x="{pad_l - 6}" y="{pad_t + 6}" font-size="9" fill="{c["text"]}">+{lim:.2f}</text>'
        f'<text x="{pad_l - 6}" y="{H - pad_b + 2:.1f}" font-size="9" fill="{c["text"]}">-{lim:.2f}</text>'
        "</svg>"
    )
    return svg


def bars_svg(rows, theme="dark", value_fmt="{:.0f}%"):
    """rows: list of (label, fraction 0..1, highlight)."""
    c = _theme_colors(theme)
    W, H = 560, 30 + 40 * len(rows)
    pad_l, pad_r = 46, 40
    parts = []
    for i, (label, frac, hl) in enumerate(rows):
        y = 26 + i * 40
        w = max(2.0, frac * (W - pad_l - pad_r))
        color = c["red"] if hl else c["amber"]
        parts.append(
            f'<text x="0" y="{y + 13}" font-size="11" fill="{c["text"]}">{label}</text>'
            f'<rect x="{pad_l}" y="{y}" width="{W - pad_l - pad_r}" height="18" rx="4" fill="{c["bar_track"]}"/>'
            f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" height="18" rx="4" fill="{color}"/>'
            f'<text x="{W - pad_r + 8}" y="{y + 13}" font-size="10" fill="{color}">{value_fmt.format(frac * 100)}</text>'
        )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        + "".join(parts)
        + "</svg>"
    )
    return svg

# ---------------------------------------------------------------------------
# Batch (multi-file) results
# ---------------------------------------------------------------------------
def render_batch_section(subsystem, files, model_choice, theme, submitted):
    if not (submitted or st.session_state.get("ran_batch")):
        return
    st.session_state["ran_batch"] = True

    st.html(
        """
        <div class="section">
            <span class="sec-t">03 · BATCH OUTPUT</span>
            <span class="sec-line"></span>
            <span class="sec-hint">ALL FILES · AGGREGATED</span>
        </div>
        """,
    )

    bundle = load_bundle(SUBSYSTEM_BUNDLES[subsystem])
    entry = bundle["models"][model_choice]
    rows = []
    errors = []

    if subsystem == "Door":
        for f in files:
            try:
                df = dp.load_stream(io.BytesIO(f.getvalue()))
                preds = dp.run_inference(df, entry["model"], entry["scaler"])
                n_tot = len(preds)
                n_ab = int((preds["status"] == dp.LABEL_ABNORMAL).sum())
                rows.append(
                    {
                        "file": f.name,
                        "cycles": n_tot,
                        "abnormal": n_ab,
                        "rate_pct": round(n_ab / n_tot * 100, 1) if n_tot else 0.0,
                        "mean_conf": round(float(preds["confidence"].mean()), 3),
                    }
                )
            except Exception as exc:
                errors.append((f.name, str(exc)))
        if rows:
            table = pd.DataFrame(rows)
            n_ab_all = int(table["abnormal"].sum())
            st.html(
                f"""
                <div class="result-banner">
                    <span><b>{len(rows)}</b> streams analysed · <b>{n_ab_all}</b> abnormal cycles flagged across all files</span>
                </div>
                """,
            )
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label="Download batch predictions",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="door_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

    elif subsystem == "SHM":
        for f in files:
            try:
                df_in = pd.read_csv(io.BytesIO(f.getvalue()))
                x = df_in.iloc[:, 0].to_numpy(dtype=float)
                feats = pd.DataFrame([sp.extract_features(x)], columns=sp.FEATURE_NAMES)
                Xs = entry["scaler"].transform(feats)
                pred = float(entry["model"].predict(Xs)[0])
                if entry.get("log_target", True):
                    pred = float(np.expm1(pred))
                rows.append(
                    {
                        "file": f.name,
                        "damage": round(max(pred, 0.0), 6),
                        "samples": len(x),
                    }
                )
            except Exception as exc:
                errors.append((f.name, str(exc)))
        if rows:
            table = pd.DataFrame(rows)
            mean_dmg = float(table["damage"].mean())
            worst = table.loc[table["damage"].idxmax()]
            st.html(
                f"""
                <div class="result-banner">
                    <span>Mean cumulative damage <b>{mean_dmg:.4f}</b> · worst file <b>{worst['file']}</b> at <b>{worst['damage']:.4f}</b></span>
                </div>
                """,
            )
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label="Download batch predictions",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="shm_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

    elif subsystem == "Rail Corrugation":
        counts = {lab: 0 for lab in rp.LABELS}
        proba_mean = {lab: 0.0 for lab in rp.LABELS}
        n_ok = 0
        for f in files:
            try:
                df_r = rp.load_rail_file(io.BytesIO(f.getvalue()))
                feats = rp.extract_features(df_r)
                Xs = entry["scaler"].transform(
                    pd.DataFrame([feats], columns=rp.FEATURE_NAMES)
                )
                label = rp.LABELS[int(entry["model"].predict(Xs)[0])]
                proba = entry["model"].predict_proba(Xs)[0]
                counts[label] += 1
                for lab, p in zip(rp.LABELS, proba):
                    proba_mean[lab] += float(p)
                n_ok += 1
                rows.append(
                    {
                        "file": f.name,
                        "prediction": label,
                        "confidence": round(float(proba.max()), 3),
                    }
                )
            except Exception as exc:
                errors.append((f.name, str(exc)))
        if rows:
            table = pd.DataFrame(rows)
            dominant = max(rp.LABELS, key=lambda lab: counts[lab])
            mean_rows = [
                (lab, proba_mean[lab] / n_ok, lab == dominant) for lab in rp.LABELS
            ]
            st.html(
                f"""
                <div class="result-banner">
                    <span><b>{len(rows)}</b> recordings classified · most common verdict <b>{dominant.upper()}</b> ({counts[dominant]})</span>
                </div>
                """,
            )
            chart_frame(bars_svg(mean_rows, theme=theme), 30 + 40 * len(rp.LABELS))
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label="Download batch predictions",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="rail_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

    elif subsystem == "ACV":
        proba_sum = {}
        n_ok = 0
        for f in files:
            tmp_path = os.path.join(tempfile.gettempdir(), f"acv_batch_{f.name}")
            with open(tmp_path, "wb") as fh:
                fh.write(f.getvalue())
            try:
                cars, X = acvp.extract_file_features(tmp_path)
                Xs = entry["scaler"].transform(X)
                proba = entry["model"].predict_proba(Xs)[:, 1]
            except Exception as exc:
                errors.append((f.name, str(exc)))
                continue
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            for car, p in zip(cars, proba):
                proba_sum[car] = proba_sum.get(car, 0.0) + float(p)
            n_ok += 1
            order = np.argsort(-proba)
            rows.append(
                {
                    "file": f.name,
                    "top_car": f"Car {cars[int(order[0])]}",
                    "ranking": " | ".join(str(cars[int(i)]) for i in order),
                }
            )
        if rows:
            table = pd.DataFrame(rows)
            mean_proba = {car: p / n_ok for car, p in proba_sum.items()}
            ranked_list = sorted(mean_proba, key=mean_proba.get, reverse=True)
            proba_list = [mean_proba[c] for c in ranked_list]
            st.html(
                f"""
                <div class="result-banner">
                    <span>Fleet verdict — <b>CAR {ranked_list[0]}</b> · probability {proba_list[0] * 100:.0f}% across {len(rows)} cases</span>
                </div>
                """,
            )
            bar_rows = [
                (f"Car {c}", p, i == 0)
                for i, (c, p) in enumerate(zip(ranked_list, proba_list))
            ]
            chart_frame(bars_svg(bar_rows, theme=theme), 30 + 40 * len(bar_rows))
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label="Download batch rankings",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="acv_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

    for name, msg in errors:
        st.error(f"{name} — {msg}")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redshift · Train Condition Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

        :root {
            --bg: #06060B;
            --panel: #0C0C14;
            --line: rgba(255, 255, 255, 0.055);
            --line-strong: rgba(255, 255, 255, 0.10);
            --txt: #E7E7EF;
            --txt2: #9C9CA8;
            --muted: #63636E;
            --red: #FF2D55;
            --red-soft: #FF6B81;
            --amber: #FFB020;
            --input: #14141E;
            --track: #161620;
            --scroll-thumb: #1C1C28;
            --scroll-thumb-hover: #2A2A3A;
            --chip-up: #FF9AA8;
            --on-accent: #fff;
            --red-text: #FFC2CC;
            --alert-text: #FFB9C4;
            --switch-track: rgba(250, 250, 250, 0.20);
            --switch-thumb: #FAFAFA;
            --switch-on: rgba(255, 45, 85, 0.45);
            --btn-bg: rgba(255, 255, 255, 0.015);
            --btn-text: #9C9CA8;
            --btn-text-hover: #E7E7EF;
            --btn-primary-bg: rgba(255, 45, 85, 0.09);
            --red-btn-text: #FFC2CC;
        }

        html {scroll-behavior: smooth;}

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg);
            background-image:
                radial-gradient(820px 460px at 12% -8%, rgba(255, 45, 85, 0.06), transparent 60%),
                linear-gradient(rgba(255, 255, 255, 0.014) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.014) 1px, transparent 1px);
            background-size: auto, 46px 46px, 46px 46px;
            color: var(--txt2);
        }

        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            max-width: 1160px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        ::-webkit-scrollbar {width: 8px;}
        ::-webkit-scrollbar-track {background: var(--bg);}
        ::-webkit-scrollbar-thumb {background: var(--scroll-thumb);}
        ::-webkit-scrollbar-thumb:hover {background: var(--scroll-thumb-hover);}
        ::selection {background: var(--red); color: #fff;}

        @keyframes fadeUp {
            from {opacity: 0; transform: translateY(14px);}
            to {opacity: 1; transform: none;}
        }

        @keyframes blink {
            0%, 100% {opacity: 1;}
            50% {opacity: 0.35;}
        }

        /* Top bar ------------------------------------------------------ */
        .topbar {display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0 0.9rem;}
        .tb-l {display: flex; align-items: center; gap: 0.75rem;}
        .logo-mark {
            width: 28px; height: 28px; border-radius: 6px;
            background: var(--red);
            display: flex; align-items: center; justify-content: center;
            font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 15px;
            color: #0A0A12;
        }
        .tb-name {font-weight: 700; font-size: 0.9rem; letter-spacing: 0.3em; color: var(--txt);}
        .tb-div {width: 1px; height: 16px; background: var(--line-strong);}
        .tb-sub {font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 0.14em; color: var(--muted);}
        .tb-ver {
            font-family: 'IBM Plex Mono', monospace;
            border: 1px solid var(--line-strong);
            background: var(--input);
            border-radius: 999px; padding: 0.34rem 0.8rem;
            font-size: 0.6rem; font-weight: 500; letter-spacing: 0.12em;
            color: var(--muted);
        }

        /* Results grid --------------------------------------------------- */
        .bento {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 1.2rem 0 0.4rem;}
        .fcard {
            display: flex; flex-direction: column;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1.3rem 1.4rem;
            position: relative;
            animation: fadeUp 0.55s ease both;
            transition: border-color 0.2s ease;
        }
        .fcard::before {
            content: ""; position: absolute; top: -1px; left: -1px;
            width: 26px; height: 26px;
            border-top: 2px solid rgba(255, 45, 85, 0.5);
            border-left: 2px solid rgba(255, 45, 85, 0.5);
            border-top-left-radius: 12px;
        }
        .fcard:hover {border-color: rgba(255, 45, 85, 0.30);}
        .fcard-wide {grid-column: span 2;}
        .fcard-h {display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.6rem;}
        .fcard-ic {
            width: 28px; height: 28px; border-radius: 7px; flex-shrink: 0;
            background: rgba(255, 45, 85, 0.08);
            border: 1px solid rgba(255, 45, 85, 0.30);
            display: flex; align-items: center; justify-content: center;
            color: var(--red-soft);
        }
        .fcard-t {font-weight: 600; font-size: 0.95rem; color: var(--txt);}
        .fcard-s {font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-top: 0.15rem;}

        .fchart {flex: 1; display: flex; align-items: center; justify-content: center; margin: 0.4rem 0 0.2rem; min-height: 170px;}
        .fchart svg {width: 100%; height: 100%; display: block;}
        [data-testid="stCustomComponentV1"] {width: 100%;}
        [data-testid="stCustomComponentV1"] iframe {border: 0; background: transparent; width: 100%;}

        .mini-stats {
            display: flex; gap: 0.55rem; flex-wrap: wrap;
            border-top: 1px solid var(--line);
            padding-top: 0.65rem; margin-top: 0.5rem;
        }
        .chip {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.62rem; font-weight: 500; letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.34rem 0.6rem; border-radius: 6px;
            border: 1px solid var(--line);
            background: var(--input); color: var(--txt2);
        }
        .chip b {color: var(--txt); font-weight: 600;}
        .chip.up {border-color: rgba(255, 45, 85, 0.35); background: rgba(255, 45, 85, 0.06); color: var(--chip-up);}
        .chip.up b {color: var(--red-soft);}

        @media (max-width: 900px) {
            .bento {grid-template-columns: 1fr;}
            .fcard-wide {grid-column: auto;}
        }

        .tgl-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--red);
            animation: blink 2.4s ease-in-out infinite;
        }

        .stat-big {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 2.3rem; font-weight: 600; color: var(--txt);
            letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
        }
        .stat-big span {font-size: 0.72rem; font-weight: 500; color: var(--muted); margin-left: 0.35rem; letter-spacing: 0.08em; text-transform: uppercase;}
        .sum-rows {margin-top: 0.9rem;}
        .sum-row {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.55rem 0; border-top: 1px solid var(--line);
            font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        }
        .sum-row:first-child {border-top: none; padding-top: 0.1rem;}
        .sum-row span {color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.64rem;}
        .sum-row b {color: var(--txt); font-weight: 600;}
        .up-foot {
            margin-top: auto; padding-top: 0.65rem;
            display: flex; align-items: center; gap: 0.5rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.62rem; letter-spacing: 0.06em; text-transform: uppercase;
            color: var(--muted);
        }

        /* Console --------------------------------------------------------- */
        .section {display: flex; align-items: center; gap: 0.9rem; margin: 2.8rem 0 1.1rem;}
        .sec-t {font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.74rem; color: var(--txt); letter-spacing: 0.14em;}
        .sec-line {flex: 1; height: 1px; background: var(--line-strong);}
        .sec-hint {font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 0.12em; color: var(--muted);}

        .stButton > button {
            font-family: 'IBM Plex Mono', monospace;
            border-radius: 10px;
            border: 1px solid var(--line-strong);
            background: var(--btn-bg);
            color: var(--btn-text);
            font-weight: 500; font-size: 0.7rem; letter-spacing: 0.1em;
            padding: 0.95rem 0.8rem;
            transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
        }
        .stButton > button:hover {border-color: rgba(255, 107, 83, 0.5); color: var(--btn-text-hover);}
        .stButton > button[kind="primary"] {
            border: 1px solid rgba(255, 45, 85, 0.55);
            background: var(--btn-primary-bg);
            color: var(--red-btn-text);
        }
        .stButton > button[kind="primary"]:hover {border-color: var(--red-soft); color: var(--btn-text-hover);}
        .stButton > button:focus:not(:active) {box-shadow: none;}

        .sub-desc {
            margin-top: -0.05rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 9px; font-weight: 500; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--muted); text-align: center;
        }

        [data-testid="stForm"] {border: none; padding: 0; border-radius: 0;}
        [data-testid="stFormSubmitButton"] {
            width: 100% !important;
            background: var(--btn-primary-bg) !important;
            border: 1px solid rgba(255, 45, 85, 0.45) !important;
            border-radius: 10px !important;
            color: var(--red-btn-text) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.14em !important;
            padding: 0.95rem 1rem !important;
        }
        [data-testid="stFormSubmitButton"]:hover {
            background: rgba(255, 45, 85, 0.18) !important;
            color: var(--btn-text-hover) !important;
        }

        /* Model picker ----------------------------------------------------- */
        [data-testid="stSelectbox"] label p {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.62rem; font-weight: 500; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--muted);
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            font-family: 'IBM Plex Mono', monospace;
            background: var(--input);
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            color: var(--txt2);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--btn-bg);
            border: 1px dashed var(--line-strong);
            border-radius: 12px;
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {border-color: rgba(255, 45, 85, 0.5);}
        [data-testid="stFileUploaderDropzoneInstructions"] > div {
            font-family: 'IBM Plex Mono', monospace; color: var(--muted);
            font-size: 0.72rem; letter-spacing: 0.06em;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span {color: var(--red-soft);}
        [data-testid="stFileUploaderDropzoneInstructions"] small {color: var(--muted);}
        [data-testid="stFileUploaderDropzone"] button {
            font-family: 'IBM Plex Mono', monospace;
            background: var(--btn-bg);
            border: 1px solid var(--line-strong);
            border-radius: 7px;
            color: var(--btn-text);
            font-weight: 500; font-size: 0.64rem; letter-spacing: 0.06em;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {border-color: rgba(255, 45, 85, 0.45); color: var(--btn-text-hover);}

        .file-chip {
            font-family: 'IBM Plex Mono', monospace;
            display: inline-flex; align-items: center; gap: 0.55rem;
            margin: 0.9rem 0 0.7rem; padding: 0.55rem 0.9rem;
            background: var(--input);
            border: 1px solid var(--line-strong);
            border-left: 3px solid var(--red);
            border-radius: 0 8px 8px 0;
            font-size: 0.66rem; font-weight: 500; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--txt2);
        }

        [data-testid="stProgress"] > div > div {background: var(--track); border-radius: 999px; height: 6px;}
        [data-testid="stProgress"] > div > div > div > div {
            background: var(--red);
            border-radius: 999px;
        }
        [data-testid="stProgress"] p {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.64rem; font-weight: 500; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--muted);
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1.1rem 1.15rem;
            animation: fadeUp 0.45s ease both;
        }
        [data-testid="stMetricValue"] {font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: var(--txt); font-variant-numeric: tabular-nums;}
        [data-testid="stMetricLabel"] {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem; font-weight: 500;
            letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
        }

        .result-banner {
            font-family: 'IBM Plex Mono', monospace;
            display: flex; align-items: center; gap: 0.8rem;
            margin: 0.8rem 0 1rem; padding: 0.95rem 1.1rem;
            background: var(--input);
            border: 1px solid var(--line-strong);
            border-left: 3px solid var(--amber);
            border-radius: 8px;
            font-size: 0.66rem; font-weight: 500; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--txt2);
            animation: fadeUp 0.45s ease both;
        }
        .result-banner b {color: var(--amber); font-weight: 600;}

        [data-testid="stDownloadButton"] > button {
            font-family: 'IBM Plex Mono', monospace;
            border-radius: 10px;
            border: 1px solid var(--line-strong);
            background: var(--btn-bg);
            color: var(--btn-text);
            font-weight: 500; font-size: 0.7rem;
            letter-spacing: 0.1em; text-transform: uppercase;
            padding: 0.95rem 1rem;
            transition: border-color 0.15s ease, color 0.15s ease;
        }
        [data-testid="stDownloadButton"] > button:hover {border-color: rgba(255, 45, 85, 0.45); color: var(--btn-text-hover);}

        /* Helper text ---------------------------------------------------------- */
        .helper {
            margin-top: 0.55rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.64rem; letter-spacing: 0.04em; color: var(--muted);
            line-height: 1.6;
        }
        .helper em {color: var(--red-soft); font-style: normal; font-weight: 500;}

        .await {
            margin-top: 1rem; padding: 2.8rem 1.5rem; text-align: center;
            border: 1px dashed var(--line-strong);
            border-radius: 12px;
            background: var(--btn-bg);
            animation: fadeUp 0.5s ease both;
        }
        .await-t {font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.78rem; letter-spacing: 0.3em; color: var(--btn-text-hover);}
        .await-s {font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.08em; color: var(--muted); margin-top: 0.5rem; text-transform: uppercase;}

        .stCheckbox label p {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.64rem; letter-spacing: 0.1em; text-transform: uppercase;
            color: var(--txt2);
        }
        .stCheckbox label > div:not([data-testid="stWidgetLabel"]) {background: var(--switch-track);}
        .stCheckbox label > div:not([data-testid="stWidgetLabel"]) > div {background: var(--switch-thumb);}
        .stCheckbox label:has(input:checked) > div:not([data-testid="stWidgetLabel"]) {background: var(--switch-on);}
        .stCheckbox label:has(input:checked) > div:not([data-testid="stWidgetLabel"]) > div {background: var(--red);}

        /* Dataframe + alerts ------------------------------------------------------ */
        [data-testid="stDataFrame"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.35rem;
        }
        [data-testid="stDataFrame"] * {font-family: 'IBM Plex Mono', monospace !important;}
        [data-testid="stDataFrame"] [role="columnheader"] span {
            font-size: 10px !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted) !important;
        }
        [data-testid="stAlert"] {
            background: rgba(255, 45, 85, 0.07);
            border: 1px solid rgba(255, 45, 85, 0.35);
            border-left: 3px solid var(--red);
            border-radius: 8px;
            color: var(--alert-text);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
        }
    </style>
    """,
)

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.session_state.setdefault("theme", "dark")

top_l, top_r = st.columns([3.2, 1], gap="medium", vertical_alignment="center")
with top_l:
    st.html(
        """
        <div class="topbar">
            <div class="tb-l">
                <div class="logo-mark">R</div>
                <div class="tb-name">REDSHIFT</div>
                <div class="tb-div"></div>
                <div class="tb-sub">TRAIN CONDITION MONITORING</div>
            </div>
            <div class="tb-ver">CONSOLE · 2.1</div>
        </div>
        """,
    )
with top_r:
    light_mode = st.toggle("Light mode", key="theme_toggle")

theme = "light" if light_mode else "dark"
st.session_state["theme"] = theme
if theme == "light":
    st.html(
        """
        <style>
            :root {
                --bg: #F4F5FA;
                --panel: #FFFFFF;
                --line: rgba(15, 23, 42, 0.10);
                --line-strong: rgba(15, 23, 42, 0.16);
                --txt: #0F172A;
                --txt2: #475569;
                --muted: #64748B;
                --red: #E11D48;
                --red-soft: #E11D48;
                --amber: #D97706;
                --input: #FFFFFF;
                --track: #E2E6EF;
                --scroll-thumb: #C6CCD8;
                --scroll-thumb-hover: #AEB6C4;
                --chip-up: #E11D48;
                --on-accent: #0F172A;
                --red-text: #E11D48;
                --alert-text: #BE123C;
                --switch-track: rgba(15, 23, 42, 0.15);
                --switch-thumb: #FFFFFF;
                --switch-on: rgba(225, 29, 72, 0.45);
                --btn-bg: #0F172A;
                --btn-text: #E7E7EF;
                --btn-text-hover: #FFFFFF;
                --btn-primary-bg: #0F172A;
                --red-btn-text: #FF6B81;
            }
            .stApp, [data-testid="stAppViewContainer"] {
                background-image:
                    radial-gradient(820px 460px at 12% -8%, rgba(225, 29, 72, 0.05), transparent 60%);
            }
            .await-s {color: #9CA3AF;}
            [data-testid="stFileUploaderDropzone"] {border-color: rgba(15, 23, 42, 0.45);}
            [data-testid="stFileUploaderDropzoneInstructions"] > div {color: #9CA3AF;}
            [data-testid="stFileUploaderDropzoneInstructions"] small {color: #9CA3AF;}
            [data-testid="stFileUploaderDropzoneInstructions"] span {color: #FF6B81;}
        </style>
        """,
    )

# ---------------------------------------------------------------------------
# Console — subsystem selector
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="section" id="console">
        <span class="sec-t">01 · PICK A SUBSYSTEM</span>
        <span class="sec-line"></span>
        <span class="sec-hint">CHANGE ANYTIME</span>
    </div>
    """,
)

st.session_state.setdefault("subsystem", "Door")

cols = st.columns(4, gap="medium")
for col, (name, code, desc) in zip(cols, SUBSYSTEMS):
    with col:
        selected = st.session_state.subsystem == name
        if st.button(
            f"{code} {name.upper()}",
            key=f"sub_{name}",
            type="primary" if selected else "secondary",
            width="stretch",
        ):
            st.session_state.subsystem = name
            st.rerun()
        st.html(f'<div class="sub-desc">{desc}</div>')

subsystem = st.session_state.subsystem

# ---------------------------------------------------------------------------
# Console — upload + prediction flow (placeholder logic)
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="section">
        <span class="sec-t">02 · UPLOAD DATA</span>
        <span class="sec-line"></span>
        <span class="sec-hint">CSV · TXT · XLSX</span>
    </div>
    """,
)

uploaded_files = st.file_uploader(
    f"Upload {subsystem} data file(s)",
    type=["csv", "txt", "xlsx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
uploaded_files = uploaded_files or []

upload_sig = tuple((f.name, f.size) for f in uploaded_files)
if st.session_state.get("upload_sig") != upload_sig:
    st.session_state["upload_sig"] = upload_sig
    st.session_state["ran"] = False
    st.session_state["ran_batch"] = False

if not uploaded_files:
    st.html(
        """
        <div class="await">
            <div class="await-t">AWAITING STREAM</div>
            <div class="await-s">no files loaded — drop one or more CSV · TXT · XLSX exports above</div>
        </div>
        """,
    )

if uploaded_files:
    uploaded_file = uploaded_files[0]
    size_kb = uploaded_file.size / 1024
    stream_df = None
    preview_html = ""
    try:
        stream_df = dp.load_stream(io.BytesIO(uploaded_file.getvalue()))
        span_s = (stream_df["t"].iloc[-1] - stream_df["t"].iloc[0]).total_seconds()
        rate_hz = (
            1000.0 / float(stream_df["dt_ms"].iloc[1:].mean()) if len(stream_df) > 2 else 0.0
        )
        n_gaps = int((stream_df["dt_ms"] > dp.GAP_THRESHOLD_MS).sum())
        preview_html = (
            '<div class="helper" style="margin-top:0.35rem">'
            f'Rows <em>{len(stream_df):,}</em> · span <em>{span_s / 60:.1f} min</em> · '
            f'sample <em>{rate_hz:.0f} Hz</em> · gaps <em>{n_gaps}</em></div>'
        )
    except Exception:
        stream_df = None
    if len(uploaded_files) > 1:
        total_kb = sum(f.size for f in uploaded_files) / 1024
        file_list = " &nbsp;·&nbsp; ".join(f.name for f in uploaded_files)
        st.html(
            f'<div class="file-chip">[BATCH] {len(uploaded_files)} FILES &nbsp;—&nbsp; {total_kb:,.1f} KB TOTAL</div>'
            f'<div class="helper" style="margin-top:0.35rem">{file_list}</div>'
            f"{preview_html}",
        )
    else:
        st.html(
            f'<div class="file-chip">[FILE] {uploaded_file.name} &nbsp;—&nbsp; {size_kb:,.1f} KB</div>'
            f"{preview_html}",
        )

    with st.form("run_form", border=False):
        bundle = load_bundle(SUBSYSTEM_BUNDLES[subsystem])
        model_names = [bundle["best"]]
        model_choice = st.selectbox(
            "Prediction model",
            model_names,
            index=0,
            key=f"model_choice_{subsystem}",
        )
        submitted = st.form_submit_button("Run prediction", width="stretch")

    if submitted or st.session_state.get("ran"):
        st.session_state["ran"] = True

        st.html(
            """
            <div class="section" id="output">
                <span class="sec-t">03 · OUTPUT</span>
                <span class="sec-line"></span>
            </div>
            """,
        )

        if subsystem == "Door":
            door_error = None
            try:
                entry = load_bundle(SUBSYSTEM_BUNDLES["Door"])["models"][model_choice]
                df = stream_df if stream_df is not None else dp.load_stream(
                    io.BytesIO(uploaded_file.getvalue())
                )
                prog = st.progress(0, text="Segmenting the stream...")
                segs = dp.segment_stream(df)
                prog.progress(30, text=f"{len(segs)} cycles found — classifying...")
                preds = dp.run_inference(df, entry["model"], entry["scaler"])
                prog.progress(75, text="Running Monte Carlo simulation...")
                mc = dd.run_monte_carlo(preds)
                prog.progress(100, text="Done")
                prog.empty()
            except Exception as exc:
                preds = None
                mc = None
                door_error = str(exc)

            if door_error is not None:
                st.error(f"Could not analyse this file as a Door data stream — {door_error}")
            else:
                n_abnormal = int((preds["status"] == dp.LABEL_ABNORMAL).sum())
                n_total = len(preds)
                n_normal = n_total - n_abnormal
                mean_conf = float(preds["confidence"].mean())
                rate = n_abnormal / n_total * 100 if n_total else 0.0
                t_start = dp.parse_time(preds["start_time"].iloc[0])
                t_end = dp.parse_time(preds["end_time"].iloc[-1])
                dur_min = (t_end - t_start).total_seconds() / 60
                model_score = load_bundle(SUBSYSTEM_BUNDLES["Door"])["scores"].get(model_choice, None)
                model_tag = (
                    f"{model_choice} · holdout IoU-F1 {model_score:.3f}"
                    if model_score is not None
                    else model_choice
                )

                st.html(
                    f"""
                    <div class="bento">

                        <div class="fcard fcard-wide">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["activity"]}</div>
                                <div>
                                    <div class="fcard-t">Cycle timeline</div>
                                    <div class="fcard-s">each bar = one cycle · red = abnormal resistance</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.timeline_svg(preds, theme=theme), 200)
                st.html(
                    f"""
                            <div class="mini-stats">
                                <span class="chip">Cycles <b>{n_total}</b></span>
                                <span class="chip">Abnormal <b>{n_abnormal}</b></span>
                                <span class="chip">Normal <b>{n_normal}</b></span>
                                <span class="chip up">Abnormal rate <b>{rate:.0f}%</b></span>
                            </div>
                        </div>

                        <div class="fcard">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["gauge"]}</div>
                                <div>
                                    <div class="fcard-t">Analysis summary</div>
                                    <div class="fcard-s">{model_tag}</div>
                                </div>
                            </div>
                            <div class="stat-big">{n_total}<span>cycles</span></div>
                            <div class="sum-rows">
                                <div class="sum-row"><span>Abnormal</span><b style="color:var(--red-soft)">{n_abnormal}</b></div>
                                <div class="sum-row"><span>Normal</span><b>{n_normal}</b></div>
                                <div class="sum-row"><span>Mean confidence</span><b>{mean_conf * 100:.0f}%</b></div>
                                <div class="sum-row"><span>Stream length</span><b>{dur_min:.0f} min</b></div>
                            </div>
                            <div class="up-foot"><span class="tgl-dot"></span>{n_abnormal} cycles flagged for inspection</div>
                        </div>

                        <div class="fcard fcard-wide">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["bar"]}</div>
                                <div>
                                    <div class="fcard-t">Monte Carlo — fault distribution</div>
                                    <div class="fcard-s">{mc['n_trials']:,} trials × {mc['horizon']:,} future cycles</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.mc_hist_svg(mc, theme=theme), 300)
                st.html(
                    f"""
                            <div class="mini-stats">
                                <span class="chip">Mean <b>{mc['mean']:,.1f}</b></span>
                                <span class="chip">P95 <b>{mc['p95']:,.1f}</b></span>
                                <span class="chip">P99 <b>{mc['p99']:,.1f}</b></span>
                            </div>
                        </div>

                        <div class="fcard">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["pulse"]}</div>
                                <div>
                                    <div class="fcard-t">Cycle risk score</div>
                                    <div class="fcard-s">model P(abnormal) per cycle · 10 bins</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.risk_hist_svg(preds, theme=theme), 260)
                st.html(
                    f"""
                            <div class="up-foot"><span class="tgl-dot"></span>risk re-sampled in the simulation</div>
                        </div>

                        <div class="fcard fcard-wide">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["trend"]}</div>
                                <div>
                                    <div class="fcard-t">Survival curve</div>
                                    <div class="fcard-s">Probability of no fault vs cycles ahead</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.survival_svg(mc, theme=theme), 300)
                st.html(
                    f"""
                            <div class="mini-stats">
                                <span class="chip">Median cycles to fault <b>{mc['median_cycles_disp']}</b></span>
                                <span class="chip">Median time <b>{mc['median_hours']}</b></span>
                                <span class="chip up">P(≥1 in 100) <b>{mc['p_ge1_100'] * 100:.0f}%</b></span>
                            </div>
                        </div>

                        <div class="fcard">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["clock"]}</div>
                                <div>
                                    <div class="fcard-t">Forecast summary</div>
                                    <div class="fcard-s">from the Monte Carlo run</div>
                                </div>
                            </div>
                            <div class="sum-rows" style="margin-top:0.4rem">
                                <div class="sum-row"><span>Expected faults / 1,000</span><b>{mc['mean']:,.1f}</b></div>
                                <div class="sum-row"><span>P50 faults</span><b>{mc['p50']:,.1f}</b></div>
                                <div class="sum-row"><span>P95 faults</span><b>{mc['p95']:,.1f}</b></div>
                                <div class="sum-row"><span>P99 faults</span><b>{mc['p99']:,.1f}</b></div>
                                <div class="sum-row"><span>P(≥1 in 100 cycles)</span><b>{mc['p_ge1_100'] * 100:.0f}%</b></div>
                                <div class="sum-row"><span>Median time to fault</span><b>{mc['median_hours']}</b></div>
                            </div>
                            <div class="up-foot"><span class="chip">1,000 future cycles</span></div>
                        </div>

                    </div>
                    """,
                )

                st.html(
                    """
                    <div class="section">
                        <span class="sec-t">PREDICTIONS TABLE</span>
                        <span class="sec-line"></span>
                    </div>
                    """,
                )
                disp = preds.copy()
                disp["flag"] = np.where(
                    disp["status"] == dp.LABEL_ABNORMAL, "● abnormal", "—"
                )
                show_ab = st.toggle("Abnormal only", key="door_filter")
                view = disp[disp["status"] == dp.LABEL_ABNORMAL] if show_ab else disp

                def _flag_rows(row):
                    bg = (
                        "background-color: rgba(255, 45, 85, 0.06);"
                        if row["status"] == dp.LABEL_ABNORMAL
                        else ""
                    )
                    return [bg] * len(row)

                st.dataframe(
                    view.style.apply(_flag_rows, axis=1),
                    width="stretch",
                    hide_index=True,
                    height=min(340, 44 + 35 * len(view)),
                    column_config={
                        "segment_id": st.column_config.TextColumn("Segment"),
                        "start_time": st.column_config.TextColumn("Start"),
                        "end_time": st.column_config.TextColumn("End"),
                        "operation": st.column_config.TextColumn("Operation"),
                        "status": st.column_config.TextColumn("Status"),
                        "flag": st.column_config.TextColumn("Flag"),
                        "n_rows": st.column_config.NumberColumn("Rows"),
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.0f%%",
                        ),
                    },
                )

                st.html(
                    """
                    <div class="section">
                        <span class="sec-t">INSPECT A CYCLE</span>
                        <span class="sec-line"></span>
                        <span class="sec-hint">CURRENT &amp; POSITION VS STROKE</span>
                    </div>
                    """,
                )
                insp_options = [
                    f"{r.segment_id} · {r.operation} · {r.status} · conf {r.confidence * 100:.0f}%"
                    for r in preds.itertuples()
                ]
                insp_idx = st.selectbox(
                    "Cycle",
                    range(len(insp_options)),
                    format_func=lambda i: insp_options[i],
                    key="door_inspect",
                )
                sel = preds.iloc[insp_idx]
                seg_mask = (df["t"] >= dp.parse_time(sel["start_time"])) & (
                    df["t"] <= dp.parse_time(sel["end_time"])
                )
                chart_frame(dd.cycle_detail_svg(df[seg_mask], sel, theme=theme), 320)

                st.download_button(
                    label="Download predictions",
                    data=preds[
                        ["segment_id", "start_time", "end_time", "operation", "status", "n_rows"]
                    ].to_csv(index=False).encode("utf-8"),
                    file_name="door_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
        elif subsystem == "SHM":
            shm_error = None
            try:
                entry = load_bundle(SUBSYSTEM_BUNDLES["SHM"])["models"][model_choice]
                df_in = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                x = df_in.iloc[:, 0].to_numpy(dtype=float)
                feats = pd.DataFrame([sp.extract_features(x)], columns=sp.FEATURE_NAMES)
                Xs = entry["scaler"].transform(feats)
                pred = float(entry["model"].predict(Xs)[0])
                if entry.get("log_target", True):
                    pred = float(np.expm1(pred))
                damage = max(pred, 0.0)
                shm_score = load_bundle(SUBSYSTEM_BUNDLES["SHM"])["scores"].get(model_choice, None)
            except Exception as exc:
                damage = None
                shm_error = str(exc)

            if shm_error is not None:
                st.error(f"Could not analyse this file as an SHM stress stream — {shm_error}")
            else:
                tag = (
                    f"{model_choice} · CV 1−MAPE {shm_score:.3f}"
                    if shm_score is not None
                    else model_choice
                )
                st.html(
                    f"""
                    <div class="result-banner">
                        <span>Cumulative fatigue damage estimate — <b>{damage:.4f}</b>
                        · higher values mean closer to the fatigue limit.</span>
                    </div>
                    <div class="fcard">
                        <div class="fcard-h">
                            <div class="fcard-ic">{ICONS["gauge"]}</div>
                            <div>
                                <div class="fcard-t">Fatigue damage</div>
                                <div class="fcard-s">{tag}</div>
                            </div>
                        </div>
                        <div class="stat-big">{damage:.4f}<span>cumulative damage</span></div>
                        <div class="sum-rows">
                            <div class="sum-row"><span>Fatigue failure threshold</span><b>1.0000</b></div>
                            <div class="sum-row"><span>Remaining margin</span><b>{max(1.0 - damage, 0.0):.4f}</b></div>
                            <div class="sum-row"><span>Samples analysed</span><b>{len(x):,}</b></div>
                        </div>
                    </div>
                    """,
                )
                chart_frame(shm_signal_svg(x, theme=theme), 240)
                st.download_button(
                    label="Download prediction",
                    data=pd.DataFrame(
                        {"file_id": [uploaded_file.name], "prediction": [round(damage, 6)]}
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="shm_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )

        elif subsystem == "Rail Corrugation":
            rail_error = None
            try:
                entry = load_bundle(SUBSYSTEM_BUNDLES["Rail Corrugation"])["models"][model_choice]
                df_r = rp.load_rail_file(io.BytesIO(uploaded_file.getvalue()))
                feats = rp.extract_features(df_r)
                Xs = entry["scaler"].transform(
                    pd.DataFrame([feats], columns=rp.FEATURE_NAMES)
                )
                label = rp.LABELS[int(entry["model"].predict(Xs)[0])]
                proba = entry["model"].predict_proba(Xs)[0]
                conf = float(proba.max())
                rail_score = load_bundle(SUBSYSTEM_BUNDLES["Rail Corrugation"])["scores"].get(
                    model_choice, None
                )
            except Exception as exc:
                label = None
                rail_error = str(exc)

            if rail_error is not None:
                st.error(f"Could not analyse this file as a Rail Corrugation recording — {rail_error}")
            else:
                tag = (
                    f"{model_choice} · CV macro F1 {rail_score:.3f}"
                    if rail_score is not None
                    else model_choice
                )
                flag = label != "Normal"
                st.html(
                    f"""
                    <div class="result-banner">
                        <span>Classification — <b>{label.upper()}</b>
                        · confidence {conf * 100:.0f}%</span>
                    </div>
                    <div class="fcard">
                        <div class="fcard-h">
                            <div class="fcard-ic">{ICONS["pulse"]}</div>
                            <div>
                                <div class="fcard-t">Corrugation verdict</div>
                                <div class="fcard-s">{tag}</div>
                            </div>
                        </div>
                        <div class="stat-big">{label}<span></span></div>
                        <div class="sum-rows">
                            <div class="sum-row"><span>Confidence</span><b>{conf * 100:.0f}%</b></div>
                            <div class="sum-row"><span>Duration</span><b>1.0 s · 10 kHz</b></div>
                        </div>
                    </div>
                    """,
                )
                bar_rows = [(lab, float(p), lab == label) for lab, p in zip(rp.LABELS, proba)]
                chart_frame(bars_svg(bar_rows, theme=theme), 30 + 40 * 3)
                st.download_button(
                    label="Download prediction",
                    data=pd.DataFrame(
                        {"file_id": [uploaded_file.name], "prediction": [label]}
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="rail_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )

        elif subsystem == "ACV":
            acv_error = None
            ranked = None
            proba = None
            try:
                entry = load_bundle(SUBSYSTEM_BUNDLES["ACV"])["models"][model_choice]
                tmp_path = os.path.join(tempfile.gettempdir(), f"acv_upload_{uploaded_file.name}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                try:
                    ranked, proba = acvp.run_inference(tmp_path, entry["model"], entry["scaler"])
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                acv_score = load_bundle(SUBSYSTEM_BUNDLES["ACV"])["scores"].get(model_choice, None)
            except Exception as exc:
                acv_error = str(exc)

            if acv_error is not None:
                st.error(f"Could not analyse this file as an ACV case — {acv_error}")
            else:
                tag = (
                    f"{model_choice} · LOO rank-decay {acv_score:.3f}"
                    if acv_score is not None
                    else model_choice
                )
                order = np.argsort(-proba)
                ranked_list = [ranked[i] for i in order]
                proba_list = [float(proba[i]) for i in order]
                st.html(
                    f"""
                    <div class="result-banner">
                        <span>Most likely faulty car — <b>CAR {ranked_list[0]}</b>
                        · probability {proba_list[0] * 100:.0f}%</span>
                    </div>
                    <div class="fcard">
                        <div class="fcard-h">
                            <div class="fcard-ic">{ICONS["bar"]}</div>
                            <div>
                                <div class="fcard-t">Car leak-probability ranking</div>
                                <div class="fcard-s">{tag}</div>
                            </div>
                        </div>
                        <div class="sum-rows">
                            <div class="sum-row"><span>Top pick</span><b>Car {ranked_list[0]}</b></div>
                            <div class="sum-row"><span>Runner-up</span><b>Car {ranked_list[1]}</b></div>
                            <div class="sum-row"><span>Cars ranked</span><b>{len(ranked_list)}</b></div>
                        </div>
                    </div>
                    """,
                )
                rows = [(f"Car {c}", p, i == 0) for i, (c, p) in enumerate(zip(ranked_list, proba_list))]
                chart_frame(bars_svg(rows, theme=theme), 30 + 40 * len(rows))
                st.download_button(
                    label="Download ranking",
                    data=pd.DataFrame(
                        {"file_id": [uploaded_file.name], "ranked_cars": ["|".join(ranked_list)]}
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="acv_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            st.html(
                """
                <div class="result-banner">
                    <span>This subsystem is not wired to a model yet.</span>
                </div>
                """,
            )

    if len(uploaded_files) > 1:
        render_batch_section(subsystem, uploaded_files, model_choice, theme, submitted)
