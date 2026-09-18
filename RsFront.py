"""
Train Condition Monitoring — Frontend shell (design only)

Run with:
    streamlit run RsFront.py

Tech-noir console: deep violet/black base (#05050B), working upload console
with a prediction-model picker and bento-card results (timeline, Monte Carlo,
survival curve). The Door subsystem runs the real segmentation +
classification pipeline (see door_pipeline.py); other subsystems remain
placeholders.
"""

import io
import os
import time

import joblib
import streamlit as st
import streamlit.components.v1 as components

import door_dashboard as dd
import door_pipeline as dp

SUBSYSTEMS = [
    ("Door", "[01]", "CYCLE TIMING · OBSTRUCTION EVENTS"),
    ("ACV", "[02]", "AIR SUPPLY · TEMPERATURE DRIFT"),
    ("Rail Corrugation", "[03]", "WEAR PATTERN · ROUGHNESS PROFILE"),
    ("SHM", "[04]", "VIBRATION · STRUCTURAL HEALTH"),
]

ICONS = {
    "activity": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "gauge": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "bar": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "pulse": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "trend": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
}


@st.cache_resource
def load_door_models():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_models.joblib")
    return joblib.load(path)


def run_door_inference(uploaded_file, model_name):
    bundle = load_door_models()
    entry = bundle["models"][model_name]
    df = dp.load_stream(io.BytesIO(uploaded_file.getvalue()))
    return dp.run_inference(df, entry["model"], entry["scaler"])


def chart_frame(svg_html, height):
    """Render an SVG chart inside an iframe so it always displays."""
    return components.html(
        "<style>body{margin:0;background:transparent;}"
        "svg{width:100%;height:100%;display:block;}</style>" + svg_html,
        height=height,
        scrolling=False,
    )

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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html {scroll-behavior: smooth;}

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #05050B;
            background-image:
                radial-gradient(1100px 620px at 50% -12%, rgba(255, 45, 85, 0.10), transparent 60%);
            color: #a1a1aa;
        }

        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            max-width: 1160px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        ::-webkit-scrollbar {width: 10px;}
        ::-webkit-scrollbar-track {background: #05050B;}
        ::-webkit-scrollbar-thumb {background: #1c1c26; border-radius: 0;}
        ::-webkit-scrollbar-thumb:hover {background: #2a2a38;}
        ::selection {background: #e11d48; color: #fff;}

        @keyframes fadeUp {
            from {opacity: 0; transform: translateY(14px);}
            to {opacity: 1; transform: none;}
        }

        /* Top bar ------------------------------------------------------ */
        .topbar {display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 0 0.2rem;}
        .tb-l {display: flex; align-items: center; gap: 0.7rem;}
        .logo-mark {
            width: 30px; height: 30px; border-radius: 8px;
            background: linear-gradient(135deg, #ff2d55, #ff8c1a);
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 14px; color: #05050B;
            box-shadow: 0 0 14px rgba(255, 45, 85, 0.35);
        }
        .tb-name {font-weight: 700; font-size: 0.85rem; letter-spacing: 0.22em; color: #fff;}
        .tb-nav {display: flex; gap: 1.4rem;}
        .tb-nav a {
            font-size: 0.72rem; font-weight: 500; letter-spacing: 0.08em;
            text-transform: uppercase; color: #6b6b76; text-decoration: none;
            transition: color 0.15s ease;
        }
        .tb-nav a:hover {color: #ffc4c4;}
        .tb-ver {
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.02);
            border-radius: 999px; padding: 0.32rem 0.75rem;
            font-size: 0.64rem; font-weight: 500; letter-spacing: 0.1em;
            color: #71717a;
        }

        /* Bento grid (prediction results) -------------------------------- */
        .bento {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 1.2rem 0 0.4rem;}
        .fcard {
            display: flex; flex-direction: column;
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            animation: fadeUp 0.55s ease both;
            transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        }
        .fcard:hover {
            border-color: rgba(255, 45, 85, 0.28);
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.35);
        }
        .fcard-wide {grid-column: span 2;}
        .fcard-h {display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.6rem;}
        .fcard-ic {
            width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
            background: rgba(255, 45, 85, 0.10);
            border: 1px solid rgba(255, 45, 85, 0.3);
            display: flex; align-items: center; justify-content: center;
            color: #ff6b6b;
        }
        .fcard-t {font-weight: 600; font-size: 0.95rem; color: #fff;}
        .fcard-s {font-size: 0.74rem; color: #6b6b76; margin-top: 0.1rem;}

        .fchart {flex: 1; display: flex; align-items: center; justify-content: center; margin: 0.4rem 0 0.2rem; min-height: 170px;}
        .fchart svg {width: 100%; height: 100%; display: block;}
        [data-testid="stCustomComponentV1"] {width: 100%;}
        [data-testid="stCustomComponentV1"] iframe {border: 0; background: transparent; width: 100%;}

        .mini-stats {
            display: flex; gap: 0.55rem; flex-wrap: wrap;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 0.65rem; margin-top: 0.5rem;
        }
        .chip {
            font-size: 0.66rem; font-weight: 500;
            padding: 0.32rem 0.65rem; border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.03); color: #a1a1aa;
        }
        .chip b {color: #fff; font-weight: 700;}
        .chip.up {border-color: rgba(255, 45, 85, 0.35); color: #ff9a9a;}
        .chip.up b {color: #ff6b6b;}

        @media (max-width: 900px) {
            .bento {grid-template-columns: 1fr;}
            .fcard-wide {grid-column: auto;}
        }

        .tgl-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #ff2d55; box-shadow: 0 0 8px rgba(255, 45, 85, 0.7);
            animation: pulse 1.6s infinite;
        }

        .stat-big {font-size: 2.4rem; font-weight: 800; color: #fff; letter-spacing: -0.02em;}
        .stat-big span {font-size: 1rem; font-weight: 500; color: #6b6b76; margin-left: 0.2rem;}
        .sum-rows {margin-top: 0.9rem;}
        .sum-row {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.55rem 0; border-top: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.78rem;
        }
        .sum-row:first-child {border-top: none; padding-top: 0.1rem;}
        .sum-row span {color: #8b8b96;}
        .sum-row b {color: #fff; font-weight: 700;}
        .up-foot {
            margin-top: auto; padding-top: 0.65rem;
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.68rem; color: #6b6b76;
        }

        /* Console --------------------------------------------------------- */
        .section {display: flex; align-items: center; gap: 0.8rem; margin: 2.6rem 0 1.1rem;}
        .sec-t {font-weight: 700; font-size: 0.95rem; color: #fff; letter-spacing: 0.04em;}
        .sec-line {flex: 1; height: 1px; background: rgba(255, 255, 255, 0.06);}
        .sec-hint {font-size: 0.7rem; color: #6b6b76;}

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.02);
            color: #c4c4cc;
            font-weight: 600; font-size: 0.8rem; letter-spacing: 0.06em;
            padding: 0.95rem 0.8rem;
            backdrop-filter: blur(12px);
            transition: border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {border-color: rgba(255, 107, 53, 0.55); color: #fff;}
        .stButton > button[kind="primary"] {
            border: 1px solid rgba(255, 107, 53, 0.7);
            background: rgba(255, 45, 85, 0.12);
            color: #ffe4e6;
            box-shadow: 0 0 18px rgba(255, 45, 85, 0.25);
        }
        .stButton > button[kind="primary"]:hover {border-color: #ff9a9a; color: #fff;}
        .stButton > button:focus:not(:active) {box-shadow: none;}

        .sub-desc {
            margin-top: -0.15rem;
            font-size: 10.5px; font-weight: 500; letter-spacing: 0.08em;
            text-transform: uppercase; color: #6b6b76; text-align: center;
        }

        [data-testid="stForm"] {border: none; padding: 0; border-radius: 0;}
        [data-testid="stFormSubmitButton"] {
            width: 100% !important;
            background: rgba(255, 45, 85, 0.16) !important;
            border: 1px solid rgba(255, 107, 53, 0.5) !important;
            border-radius: 12px !important;
            color: #ffe4e6 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.06em !important;
            padding: 0.95rem 1rem !important;
            box-shadow: 0 0 20px rgba(255, 45, 85, 0.4) !important;
        }
        [data-testid="stFormSubmitButton"]:hover {
            background: rgba(255, 45, 85, 0.28) !important;
            color: #fff !important;
        }

        /* Model picker ----------------------------------------------------- */
        [data-testid="stSelectbox"] label p {
            font-size: 0.72rem; font-weight: 500; letter-spacing: 0.06em;
            text-transform: uppercase; color: #6b6b76;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            color: #c4c4cc;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px dashed rgba(255, 255, 255, 0.16);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            transition: border-color 0.15s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {border-color: rgba(255, 107, 53, 0.6);}
        [data-testid="stFileUploaderDropzoneInstructions"] > div {color: #8b8b96;}
        [data-testid="stFileUploaderDropzoneInstructions"] span {color: #ff9a9a;}
        [data-testid="stFileUploaderDropzoneInstructions"] small {color: #6b6b76;}
        [data-testid="stFileUploaderDropzone"] button {
            background: #1b1b24;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            color: #c4c4cc;
            font-weight: 600; font-size: 0.72rem;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {border-color: rgba(255, 107, 53, 0.55); color: #fff;}

        .file-chip {
            display: inline-flex; align-items: center; gap: 0.55rem;
            margin: 0.9rem 0 0.7rem; padding: 0.5rem 0.9rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 3px solid #ff2d55;
            border-radius: 0 10px 10px 0;
            font-size: 0.72rem; font-weight: 500; letter-spacing: 0.06em;
            text-transform: uppercase; color: #c4c4cc;
        }

        [data-testid="stProgress"] > div > div {background: #1b1b24; border-radius: 999px;}
        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #ff2d55, #ffb020);
            border-radius: 999px;
            box-shadow: 0 0 12px rgba(255, 45, 85, 0.45);
        }
        [data-testid="stProgress"] p {
            font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
            text-transform: uppercase; color: #8b8b96;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            backdrop-filter: blur(12px);
            animation: fadeUp 0.45s ease both;
        }
        [data-testid="stMetricValue"] {font-weight: 800; color: #fff;}
        [data-testid="stMetricLabel"] {
            font-size: 0.62rem; font-weight: 500;
            letter-spacing: 0.14em; text-transform: uppercase; color: #6b6b76;
        }

        .result-banner {
            display: flex; align-items: center; gap: 0.8rem;
            margin: 0.8rem 0 1rem; padding: 0.9rem 1.05rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 3px solid #ff2d55;
            border-radius: 12px;
            font-size: 0.74rem; font-weight: 500; letter-spacing: 0.05em;
            text-transform: uppercase; color: #c4c4cc;
            animation: fadeUp 0.45s ease both;
        }
        .result-banner b {color: #ff9a9a; font-weight: 700;}

        [data-testid="stDownloadButton"] > button {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.02);
            color: #c4c4cc;
            font-weight: 600; font-size: 0.8rem;
            letter-spacing: 0.06em; text-transform: uppercase;
            padding: 0.95rem 1rem;
            transition: border-color 0.15s ease, color 0.15s ease;
        }
        [data-testid="stDownloadButton"] > button:hover {border-color: rgba(255, 107, 53, 0.55); color: #fff;}

        /* Helper text ---------------------------------------------------------- */
        .helper {
            margin-top: 0.55rem; font-size: 0.76rem; color: #6b6b76;
        }
        .helper em {color: #ff9a9a; font-style: normal; font-weight: 600;}

        /* Dataframe + alerts ------------------------------------------------------ */
        [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.35rem;
        }
        [data-testid="stAlert"] {
            background: rgba(255, 45, 85, 0.08);
            border: 1px solid rgba(255, 45, 85, 0.4);
            border-radius: 12px;
            color: #ffc4c4;
        }
    </style>
    """,
)

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="topbar">
        <div class="tb-l">
            <div class="logo-mark">R</div>
            <div class="tb-name">REDSHIFT</div>
        </div>
    </div>
    """,
)

# ---------------------------------------------------------------------------
# Console — subsystem selector
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="section" id="console">
        <span class="sec-t">01 · Pick a subsystem</span>
        <span class="sec-line"></span>
        <span class="sec-hint">Tap a card — you can change it anytime</span>
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
        <span class="sec-t">02 · Upload your data</span>
        <span class="sec-line"></span>
        <span class="sec-hint">CSV · TXT · XLSX</span>
    </div>
    """,
)

uploaded_file = st.file_uploader(
    f"Upload {subsystem} data file",
    type=["csv", "txt", "xlsx"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.html(
        """
        <div class="helper">Drag &amp; drop a file above, or click <em>browse files</em>.
        Stuck? Any CSV, TXT or XLSX export from your subsystem works.</div>
        """,
    )

if uploaded_file is not None:
    size_kb = uploaded_file.size / 1024
    st.html(
        f'<div class="file-chip">[FILE] {uploaded_file.name} &nbsp;—&nbsp; {size_kb:,.1f} KB</div>',
    )

    with st.form("run_form", border=False):
        bundle = load_door_models()
        model_names = list(bundle["models"].keys())
        model_choice = st.selectbox(
            "Prediction model",
            model_names,
            index=model_names.index(bundle["best"]),
            key="door_model_choice",
        )
        submitted = st.form_submit_button("Run prediction", width="stretch")

    if submitted or st.session_state.get("ran"):
        st.session_state["ran"] = True

        st.html(
            """
            <div class="section">
                <span class="sec-t">03 · Output</span>
                <span class="sec-line"></span>
            </div>
            """,
        )

        if subsystem == "Door":
            with st.spinner("Segmenting the stream and classifying cycles..."):
                try:
                    preds = run_door_inference(uploaded_file, model_choice)
                    door_error = None
                except Exception as exc:
                    preds = None
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
                mc = dd.run_monte_carlo(preds)
                model_score = load_door_models()["scores"].get(model_choice, None)
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
                                    <div class="fcard-s">Each bar is one door cycle · red = abnormal resistance</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.timeline_svg(preds), 200)
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
                                <div class="sum-row"><span>Abnormal</span><b style="color:#ff6b6b">{n_abnormal}</b></div>
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
                chart_frame(dd.mc_hist_svg(mc), 280)
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
                                    <div class="fcard-s">Model P(abnormal) per cycle</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.risk_hist_svg(preds), 260)
                st.html(
                    f"""
                            <div class="up-foot"><span class="tgl-dot"></span>Risk re-sampled in the simulation</div>
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
                chart_frame(dd.survival_svg(mc), 280)
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
                                    <div class="fcard-s">From the Monte Carlo run</div>
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
                        <span class="sec-t">Predictions table</span>
                        <span class="sec-line"></span>
                    </div>
                    """,
                )
                st.dataframe(
                    preds,
                    width="stretch",
                    hide_index=True,
                    height=300,
                    column_config={
                        "segment_id": st.column_config.TextColumn("Segment"),
                        "start_time": st.column_config.TextColumn("Start"),
                        "end_time": st.column_config.TextColumn("End"),
                        "operation": st.column_config.TextColumn("Operation"),
                        "status": st.column_config.TextColumn("Status"),
                        "n_rows": st.column_config.NumberColumn("Rows"),
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.0f%%",
                        ),
                    },
                )

                st.download_button(
                    label="Download predictions",
                    data=preds[
                        ["segment_id", "start_time", "end_time", "operation", "status", "n_rows"]
                    ].to_csv(index=False).encode("utf-8"),
                    file_name="door_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            prog = st.progress(0, text="Running your prediction...")
            for pct in range(1, 101):
                time.sleep(0.012)
                prog.progress(pct, text=f"Running your prediction... {pct}%")
            prog.empty()

            st.html(
                """
                <div class="result-banner">
                    <span>All done — here's your <b>sample output</b>. The prediction
                    model isn't wired in yet, so these numbers are placeholders.</span>
                </div>
                """,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Health Score", "94.2")
            m2.metric("Fault Probability", "2.4 %")
            m3.metric("Remaining Life", "38 d")
            m4.metric("Confidence", "97 %")

            st.download_button(
                label="Download result",
                data="placeholder\n",
                file_name="prediction.csv",
                mime="text/csv",
                width="stretch",
            )
