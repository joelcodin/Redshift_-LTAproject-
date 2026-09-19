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
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

for _sub in ("door", "acv", "rail", "shm"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), _sub))

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
    "Door": "door/door_models.joblib",
    "ACV": "acv/acv_models.joblib",
    "Rail Corrugation": "rail/rail_models.joblib",
    "SHM": "shm/shm_models.joblib",
}

ICONS = {
    "activity": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "gauge": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "bar": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "pulse": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "trend": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
}

# ---------------------------------------------------------------------------
# i18n — English -> Simplified Chinese
# ---------------------------------------------------------------------------
ZH = {
    # topbar
    "TRAIN CONDITION MONITORING": "列车状态监测",
    "Light mode": "亮色模式",
    "TELEMETRY": "遥测",
    # sections
    "01 · PICK A SUBSYSTEM": "01 · 选择子系统",
    "CHANGE ANYTIME": "随时可切换",
    "02 · UPLOAD DATA": "02 · 上传数据",
    "03 · OUTPUT": "03 · 输出",
    "03 · BATCH OUTPUT": "03 · 批量输出",
    "ALL FILES · AGGREGATED": "所有文件 · 汇总",
    "PREDICTIONS TABLE": "预测结果表",
    "INSPECT A CYCLE": "检查单个循环",
    "CURRENT &amp; POSITION VS STROKE": "电流与位置 · 相对行程",
    # subsystems
    "Door": "车门",
    "Rail Corrugation": "钢轨波磨",
    "CYCLE TIMING · OBSTRUCTION EVENTS": "循环时序 · 卡阻事件",
    "REFRIGERANT LEAK LOCALISATION": "制冷剂泄漏定位",
    "AXLE-BOX VIBRATION · CORRUGATION TYPE": "轴箱振动 · 波磨类型",
    "DYNAMIC STRESS · FATIGUE DAMAGE": "动应力 · 疲劳损伤",
    # upload / run
    "AWAITING STREAM": "等待数据流",
    "no files loaded — drop one or more CSV · TXT · XLSX exports above": "未加载文件 — 请将 CSV · TXT · XLSX 数据文件拖入上方区域",
    "Prediction model": "预测模型",
    "Run prediction": "运行预测",
    "Abnormal only": "仅显示异常",
    "Cycle": "循环",
    "Segmenting the stream...": "正在分割数据流…",
    " cycles found — classifying...": " 个循环 — 正在分类…",
    "Running Monte Carlo simulation...": "正在运行蒙特卡洛仿真…",
    "Done": "完成",
    # shared cards
    "Cycle timeline": "循环时间线",
    "each bar = one cycle · red = abnormal resistance": "每个柱条 = 一个循环 · 红色 = 异常阻力",
    "Rows": "行数",
    "span": "跨度",
    "sample": "采样",
    "gaps": "间隔",
    "FILES": "文件",
    "TOTAL": "总计",
    "Cycles": "循环",
    "Abnormal": "异常",
    "Normal": "正常",
    "Abnormal resistance": "异常阻力",
    "Abnormal rate": "异常率",
    "Analysis summary": "分析摘要",
    "cycles": "循环",
    "Mean confidence": "平均置信度",
    "Stream length": "数据流时长",
    " cycles flagged for inspection": " 个循环已标记待检查",
    "Monte Carlo — fault distribution": "蒙特卡洛 — 故障分布",
    " trials × ": " 次试验 × ",
    " future cycles": " 个未来循环",
    "Mean": "均值",
    "Cycle risk score": "循环风险评分",
    "model P(abnormal) per cycle · 10 bins": "模型每循环 P(异常) · 10 区间",
    "risk re-sampled in the simulation": "风险在仿真中重采样",
    "Survival curve": "生存曲线",
    "Probability of no fault vs cycles ahead": "未来循环内无故障概率",
    "Median cycles to fault": "故障前循环中位数",
    "Median time": "中位时间",
    "P(≥1 in 100)": "P(≥1 / 100)",
    "Forecast summary": "预测摘要",
    "from the Monte Carlo run": "来自蒙特卡洛仿真",
    "Expected faults / 1,000": "预期故障数 / 1,000",
    "P50 faults": "P50 故障数",
    "P95 faults": "P95 故障数",
    "P99 faults": "P99 故障数",
    "P(≥1 in 100 cycles)": "P(≥1 / 100 循环)",
    "Median time to fault": "故障中位时间",
    "1,000 future cycles": "1,000 个未来循环",
    "Download predictions": "下载预测结果",
    "Download prediction": "下载预测结果",
    "Download batch predictions": "下载批量预测结果",
    "Download ranking": "下载排名",
    "Download batch rankings": "下载批量排名",
    # door table
    "Segment": "段",
    "Start": "开始",
    "End": "结束",
    "Operation": "操作",
    "Status": "状态",
    "Flag": "标记",
    "Rows": "行数",
    "Confidence": "置信度",
    "● abnormal": "● 异常",
    # shm
    "Cumulative fatigue damage estimate — ": "累积疲劳损伤估计 — ",
    " · higher values mean closer to the fatigue limit.": " · 数值越高代表越接近疲劳极限。",
    "Fatigue damage": "疲劳损伤",
    "cumulative damage": "累积损伤",
    "Fatigue failure threshold": "疲劳失效阈值",
    "Remaining margin": "剩余裕度",
    "Samples analysed": "已分析样本",
    # rail
    "Classification — ": "分类 — ",
    " · confidence ": " · 置信度 ",
    "Corrugation verdict": "波磨判定",
    "Duration": "时长",
    "Side I": "I 侧",
    "Side II": "II 侧",
    # acv
    "Most likely faulty car — ": "最可能故障车厢 — ",
    " · probability ": " · 概率 ",
    "Car leak-probability ranking": "车厢泄漏概率排名",
    "Top pick": "首选",
    "Runner-up": "次选",
    "Cars ranked": "已排序车厢",
    "Car ": "车厢 ",
    # batch
    " streams analysed · ": " 个数据流已分析 · ",
    " abnormal cycles flagged across all files": " 个异常循环被标记（全部文件）",
    "Mean cumulative damage ": "平均累积损伤 ",
    " · worst file ": " · 最差文件 ",
    " at ": "，损伤 ",
    " recordings classified · most common verdict ": " 条记录已分类 · 最常见判定 ",
    "Fleet verdict — ": "车队判定 — ",
    " · probability ": " · 概率 ",
    " across ": "（",
    " cases": " 个案例）",
    "file": "文件",
    "file_id": "文件名",
    "ranked_cars": "排名车厢",
    "segment_id": "段编号",
    "start_time": "开始时间",
    "end_time": "结束时间",
    "operation": "操作",
    "n_rows": "行数",
    "cycles": "循环",
    "abnormal": "异常",
    "rate_pct": "异常率",
    "mean_conf": "平均置信度",
    "damage": "损伤",
    "samples": "样本",
    "prediction": "预测",
    "confidence": "置信度",
    "top_car": "首选车厢",
    "ranking": "排名",
    # errors / misc
    "Could not analyse this file as a Door data stream — ": "无法将该文件作为车门数据流进行分析 — ",
    "Could not analyse this file as an ": "无法将该文件作为",
    " stress stream — ": "应力数据流进行分析 — ",
    " case — ": "案例进行分析 — ",
    "Could not analyse this file as a Rail Corrugation recording — ": "无法将该文件作为钢轨波磨记录进行分析 — ",
    "This subsystem is not wired to a model yet.": "该子系统尚未接入模型。",
    "conf": "置信度",
    "dynamic stress (downsampled)": "动应力（降采样）",
}


def T(text):
    """Translate a string when the console language is Simplified Chinese."""
    if st.session_state.get("lang") == "zh":
        return ZH.get(text, text)
    return text


def TSUB(name):
    """Translated display name for a subsystem."""
    if st.session_state.get("lang") == "zh":
        return {
            "Door": "车门",
            "ACV": "空调机组",
            "Rail Corrugation": "钢轨波磨",
            "SHM": "结构健康监测",
        }.get(name, name)
    return {
        "ACV": "Air Conditioning Unit",
        "SHM": "Structural Health Monitoring",
    }.get(name, name)


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
        f'<text x="{pad_l}" y="{pad_t - 6}" font-size="10" fill="{c["text"]}">{T("dynamic stress (downsampled)")}</text>'
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
        f"""
        <div class="section">
            <span class="sec-t">{T("03 · BATCH OUTPUT")}</span>
            <span class="sec-line"></span>
            <span class="sec-hint">{T("ALL FILES · AGGREGATED")}</span>
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
                    <span><b>{len(rows)}</b>{T(" streams analysed · ")}<b>{n_ab_all}</b>{T(" abnormal cycles flagged across all files")}</span>
                </div>
                """,
            )
            table = table.rename(columns={k: T(k) for k in table.columns})
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label=T("Download batch predictions"),
                data=table.to_csv(index=False).encode("utf-8-sig"),
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
                    <span>{T("Mean cumulative damage ")}<b>{mean_dmg:.4f}</b>{T(" · worst file ")}<b>{worst['file']}</b>{T(" at ")}<b>{worst['damage']:.4f}</b></span>
                </div>
                """,
            )
            table = table.rename(columns={k: T(k) for k in table.columns})
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label=T("Download batch predictions"),
                data=table.to_csv(index=False).encode("utf-8-sig"),
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
                (T(lab), proba_mean[lab] / n_ok, lab == dominant) for lab in rp.LABELS
            ]
            st.html(
                f"""
                <div class="result-banner">
                    <span><b>{len(rows)}</b>{T(" recordings classified · most common verdict ")}<b>{T(dominant).upper()}</b> ({counts[dominant]})</span>
                </div>
                """,
            )
            chart_frame(bars_svg(mean_rows, theme=theme), 30 + 40 * len(rp.LABELS))
            table = table.copy()
            table["prediction"] = table["prediction"].map(T)
            table = table.rename(columns={k: T(k) for k in table.columns})
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label=T("Download batch predictions"),
                data=table.to_csv(index=False).encode("utf-8-sig"),
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
                    <span>{T("Fleet verdict — ")}<b>CAR {ranked_list[0]}</b>{T(" · probability ")}{proba_list[0] * 100:.0f}%{T(" across ")}{len(rows)}{T(" cases")}</span>
                </div>
                """,
            )
            bar_rows = [
                (f"{T('Car ')}{c}", p, i == 0)
                for i, (c, p) in enumerate(zip(ranked_list, proba_list))
            ]
            chart_frame(bars_svg(bar_rows, theme=theme), 30 + 40 * len(bar_rows))
            table = table.copy()
            table["top_car"] = table["top_car"].str.replace(
                r"^Car ", T("Car "), regex=True
            )
            table = table.rename(columns={k: T(k) for k in table.columns})
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                label=T("Download batch rankings"),
                data=table.to_csv(index=False).encode("utf-8-sig"),
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
            --bg: #000000;
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
            --card-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), 0 0 28px rgba(255, 45, 85, 0.08);
            --stat-glow: 0 0 24px rgba(255, 45, 85, 0.18);
        }

        html {scroll-behavior: smooth;}

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg);
            color: var(--txt2);
        }
        .stApp {
            background-image:
                radial-gradient(620px 420px at 18% 12%, rgba(255, 45, 85, 0.10), transparent 60%),
                radial-gradient(520px 400px at 82% 88%, rgba(255, 45, 85, 0.07), transparent 60%);
            background-size: 135% 135%;
            animation: nebula 18s ease-in-out infinite alternate;
        }
        @keyframes nebula {
            from {background-position: 0% 0%, 100% 100%;}
            to {background-position: 10% 6%, 86% 94%;}
        }

        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            max-width: 1360px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        /* Side rails — fill the wide-screen borders --------------------------------- */
        .rail {
            position: fixed;
            top: 0; bottom: 0;
            width: 84px;
            z-index: 999997;
            pointer-events: none;
            display: flex; flex-direction: column;
            align-items: center;
            justify-content: space-between;
            padding: 130px 0 46px;
            font-family: 'IBM Plex Mono', monospace;
            color: var(--muted);
            font-size: 0.56rem; letter-spacing: 0.22em; text-transform: uppercase;
        }
        .rail-l {left: 0; border-right: 1px solid var(--line);}
        .rail-r {right: 0; border-left: 1px solid var(--line);}
        .rail .vtext {
            writing-mode: vertical-rl;
            letter-spacing: 0.45em;
            color: var(--muted);
        }
        .rail .vtext b {color: var(--red-soft); font-weight: 600;}
        .rail .ruler {
            flex: 1; width: 1px; margin: 26px 0;
            background: repeating-linear-gradient(to bottom, var(--line-strong) 0 6px, transparent 6px 20px);
            position: relative; overflow: hidden;
        }
        .rail .ruler::after {
            content: ""; position: absolute; left: 0; right: 0; top: -40px; height: 40px;
            background: linear-gradient(to bottom, transparent, rgba(255, 45, 85, 0.9));
            animation: railScan 4.5s linear infinite;
        }
        .rail .knot {
            display: flex; flex-direction: column; align-items: center; gap: 0.6rem;
            color: var(--muted);
        }
        .rail .knot .dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--red);
            box-shadow: 0 0 12px var(--red);
            animation: blink 2.4s ease-in-out infinite;
        }
        @keyframes railScan {to {top: 100%;}}
        @media (max-width: 1500px) {
            .rail {display: none;}
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

        @keyframes shine {to {background-position: 200% center;}}
        @keyframes logoPulse {
            0%, 100% {box-shadow: 0 0 10px rgba(255, 45, 85, 0.35);}
            50% {box-shadow: 0 0 26px rgba(255, 45, 85, 0.75);}
        }
        @keyframes sweep {to {transform: translate(-50%, -50%) rotate(360deg);}}
        @keyframes scan {to {left: 110%;}}
        @keyframes flow {to {background-position: -200% 0;}}
        @keyframes dash {to {transform: translateX(100%);}}
        @keyframes glitch {
            0%, 91%, 100% {text-shadow: none; transform: none;}
            92% {text-shadow: 2px 0 var(--red-soft), -2px 0 var(--amber); transform: translateX(1px) skewX(-4deg);}
            94% {text-shadow: -2px 0 var(--red-soft), 2px 0 var(--amber); transform: translateX(-1px);}
            96% {text-shadow: 1px 0 var(--red-soft), -1px 0 var(--amber); transform: translateX(0.5px);}
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
            animation: logoPulse 3s ease-in-out infinite;
            transition: transform 0.2s ease;
        }
        .logo-mark:hover {transform: rotate(-8deg) scale(1.08);}
        .tb-name {
            font-weight: 700; font-size: 0.9rem; letter-spacing: 0.3em;
            background: linear-gradient(90deg, var(--txt) 0%, var(--red-soft) 50%, var(--txt) 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 6s linear infinite, glitch 4.5s steps(1) infinite;
        }
        .tb-div {width: 1px; height: 16px; background: var(--line-strong);}
        .tb-sub {font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 0.14em; color: var(--muted);}
        .tb-ver {
            font-family: 'IBM Plex Mono', monospace;
            border: 1px solid var(--line-strong);
            background: var(--input);
            border-radius: 999px; padding: 0.34rem 0.8rem;
            font-size: 0.6rem; font-weight: 500; letter-spacing: 0.12em;
            color: var(--muted);
            transition: border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
        }
        .tb-ver:hover {
            border-color: rgba(255, 45, 85, 0.4);
            color: var(--txt2);
            box-shadow: 0 0 14px rgba(255, 45, 85, 0.15);
        }
        .tb-r {display: flex; align-items: center; gap: 0.6rem;}
        .made-footer {
            display: flex; align-items: center; gap: 1rem; justify-content: center;
            margin: 3.5rem 0 0.5rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.58rem; font-weight: 500; letter-spacing: 0.22em;
            color: var(--muted); text-transform: uppercase;
            animation: fadeUp 0.6s ease both;
        }
        .made-line {flex: 0 0 90px; height: 1px; background: var(--line-strong);}

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
            transition: border-color 0.2s ease, box-shadow 0.25s ease, transform 0.16s ease-out;
            transform: perspective(900px) rotateX(var(--rx, 0deg)) rotateY(var(--ry, 0deg));
            will-change: transform;
        }
        .bento > .fcard:nth-child(1) {animation-delay: 0.06s;}
        .bento > .fcard:nth-child(2) {animation-delay: 0.14s;}
        .bento > .fcard:nth-child(3) {animation-delay: 0.22s;}
        .bento > .fcard:nth-child(4) {animation-delay: 0.30s;}
        .bento > .fcard:nth-child(5) {animation-delay: 0.38s;}
        .bento > .fcard:nth-child(6) {animation-delay: 0.46s;}
        .fcard::before {
            content: ""; position: absolute; top: -1px; left: -1px;
            width: 26px; height: 26px;
            border-top: 2px solid rgba(255, 45, 85, 0.5);
            border-left: 2px solid rgba(255, 45, 85, 0.5);
            border-top-left-radius: 12px;
        }
        .fcard::after {
            content: ""; position: absolute; bottom: -1px; right: -1px;
            width: 26px; height: 26px;
            border-bottom: 2px solid rgba(255, 45, 85, 0.5);
            border-right: 2px solid rgba(255, 45, 85, 0.5);
            border-bottom-right-radius: 12px;
        }
        .fcard:hover {
            border-color: rgba(255, 45, 85, 0.30);
            box-shadow: var(--card-shadow);
            transform: perspective(900px) rotateX(var(--rx, 0deg)) rotateY(var(--ry, 0deg)) translateY(-2px);
        }
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
            text-shadow: var(--stat-glow);
        }
        .stat-big b {font-weight: 600;}
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
        .sec-t::before {
            content: ""; display: inline-block;
            width: 8px; height: 8px; margin-right: 0.65rem;
            border-radius: 2px;
            background: var(--red);
            box-shadow: 0 0 10px var(--red);
            animation: blink 2.8s ease-in-out infinite;
        }
        .sec-line {
            flex: 1; height: 1px; background: var(--line-strong);
            position: relative; overflow: hidden;
        }
        .sec-line::after {
            content: ""; position: absolute; inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 45, 85, 0.55), transparent);
            transform: translateX(-100%);
            animation: dash 3.2s linear infinite;
        }
        .sec-hint {font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 0.12em; color: var(--muted);}

        .stButton > button {
            font-family: 'IBM Plex Mono', monospace;
            border-radius: 10px;
            border: 1px solid var(--line-strong);
            background: var(--btn-bg);
            color: var(--btn-text);
            font-weight: 500; font-size: 0.7rem; letter-spacing: 0.1em;
            padding: 0.95rem 0.8rem;
            position: relative; overflow: hidden;
            transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease,
                transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.15s ease;
        }
        .stButton > button::before,
        [data-testid="stFormSubmitButton"] button::before,
        [data-testid="stDownloadButton"] button::before {
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background: radial-gradient(120px circle at var(--mx, 50%) var(--my, 50%), rgba(255, 45, 85, 0.30), transparent 65%);
            opacity: 0; transition: opacity 0.2s ease;
        }
        .stButton > button:hover::before,
        [data-testid="stFormSubmitButton"] button:hover::before,
        [data-testid="stDownloadButton"] button:hover::before {opacity: 1;}
        [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
            position: relative; overflow: hidden;
        }
        .stButton > button:hover {
            border-color: rgba(255, 107, 83, 0.5); color: var(--btn-text-hover);
            box-shadow: 0 0 18px rgba(255, 45, 85, 0.22);
            transform: translateY(-2px) scale(1.045);
        }
        .stButton > button:active {
            transform: translateY(0) scale(0.97);
            transition-duration: 0.08s;
        }
        .stButton > button[kind="primary"] {
            border: 1px solid rgba(255, 45, 85, 0.55);
            background: var(--btn-primary-bg);
            color: var(--red-btn-text);
            box-shadow: 0 0 16px rgba(255, 45, 85, 0.18);
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
            position: relative; overflow: hidden;
        }
        .file-chip::after {
            content: ""; position: absolute; top: 0; bottom: 0; left: -40%; width: 40%;
            background: linear-gradient(100deg, transparent, rgba(255, 45, 85, 0.12), transparent);
            animation: scan 2.8s linear infinite;
            pointer-events: none;
        }

        [data-testid="stProgress"] > div > div {background: var(--track); border-radius: 999px; height: 6px;}
        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, var(--red), #FF5D7A, var(--red));
            background-size: 200% 100%;
            border-radius: 999px;
            animation: flow 1.6s linear infinite;
            box-shadow: 0 0 12px rgba(255, 45, 85, 0.55);
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
            font-size: 0.66rem; font-weight:500; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--txt2);
            animation: fadeUp 0.45s ease both;
            position: relative;
            box-shadow: 0 0 22px rgba(255, 176, 32, 0.06), inset 0 0 22px rgba(255, 176, 32, 0.03);
        }
        .result-banner::before {
            content: ""; width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
            background: var(--amber);
            box-shadow: 0 0 10px var(--amber);
            animation: blink 1.8s ease-in-out infinite;
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
            position: relative; overflow: hidden;
        }
        .await::before {
            content: ""; position: absolute; left: 50%; top: 50%;
            width: 200%; aspect-ratio: 1 / 1;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: conic-gradient(from 0deg, transparent 0deg 300deg, rgba(255, 45, 85, 0.05) 330deg, rgba(255, 45, 85, 0.22) 355deg, transparent 360deg);
            animation: sweep 5s linear infinite;
            pointer-events: none;
        }
        .await > * {position: relative;}
        .await-t {font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.78rem; letter-spacing: 0.3em; color: var(--btn-text-hover);}
        .await-t::after {content: "▍"; margin-left: 0.5rem; color: var(--red-soft); animation: blink 1s steps(1) infinite;}
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
        [data-testid="stDataFrame"] [role="row"]:hover {background: rgba(255, 45, 85, 0.045);}
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
st.session_state.setdefault("lang", "en")

top_l, top_r = st.columns([3.2, 1], gap="medium", vertical_alignment="center")
with top_l:
    st.html(
        f"""
        <div class="topbar">
            <div class="tb-l">
                <div class="logo-mark">R</div>
                <div class="tb-name">REDSHIFT</div>
                <div class="tb-div"></div>
                <div class="tb-sub">{T("TRAIN CONDITION MONITORING")}</div>
            </div>
            <div class="tb-r">
                <div class="tb-ver">CONSOLE</div>
            </div>
        </div>
        """,
    )
with top_r:
    toggle_col, lang_col = st.columns([1, 1], gap="small", vertical_alignment="center")
    with toggle_col:
        light_mode = st.toggle(T("Light mode"), key="theme_toggle")
    with lang_col:
        lang_label = "中文" if st.session_state.lang == "en" else "EN"
        if st.button(lang_label, key="lang_btn", width="stretch"):
            st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"
            st.rerun()

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
                --card-shadow: 0 10px 30px rgba(15, 23, 42, 0.10), 0 0 28px rgba(225, 29, 72, 0.06);
                --stat-glow: 0 0 24px rgba(225, 29, 72, 0.10);
            }
            .await-s {color: #9CA3AF;}
            .stApp {
                background-image:
                    radial-gradient(620px 420px at 18% 12%, rgba(225, 29, 72, 0.05), transparent 60%),
                    radial-gradient(520px 400px at 82% 88%, rgba(225, 29, 72, 0.035), transparent 60%);
            }
            [data-testid="stFileUploaderDropzone"] {border-color: rgba(15, 23, 42, 0.45);}
            [data-testid="stFileUploaderDropzoneInstructions"] > div {color: #9CA3AF;}
            [data-testid="stFileUploaderDropzoneInstructions"] small {color: #9CA3AF;}
            [data-testid="stFileUploaderDropzoneInstructions"] span {color: #FF6B81;}
        </style>
        """,
    )

# ---------------------------------------------------------------------------
# Side rails — occupy the wide-screen borders
# ---------------------------------------------------------------------------
st.html(
    f"""
    <div class="rail rail-l">
        <div class="vtext">REDSHIFT · <b>{T("TRAIN CONDITION MONITORING")}</b></div>
        <div class="ruler"></div>
        <div class="knot">
            <div>TCM</div>
            <div class="dot"></div>
        </div>
    </div>
    <div class="rail rail-r">
        <div class="knot">
            <div class="dot"></div>
            <div>{T("TELEMETRY")}</div>
        </div>
        <div class="ruler"></div>
        <div class="vtext">CSV · TXT · XLSX</div>
    </div>
    """,
)

# ---------------------------------------------------------------------------
# Console — subsystem selector
# ---------------------------------------------------------------------------
st.html(
    f"""
    <div class="section" id="console">
        <span class="sec-t">{T("01 · PICK A SUBSYSTEM")}</span>
        <span class="sec-line"></span>
        <span class="sec-hint">{T("CHANGE ANYTIME")}</span>
    </div>
    """,
)

st.session_state.setdefault("subsystem", "Door")

cols = st.columns(4, gap="medium")
for col, (name, code, desc) in zip(cols, SUBSYSTEMS):
    with col:
        selected = st.session_state.subsystem == name
        if st.button(
            f"{code} {TSUB(name).upper()}",
            key=f"sub_{name}",
            type="primary" if selected else "secondary",
            width="stretch",
        ):
            st.session_state.subsystem = name
            st.rerun()
        st.html(f'<div class="sub-desc">{T(desc)}</div>')

subsystem = st.session_state.subsystem

# ---------------------------------------------------------------------------
# Console — upload + prediction flow (placeholder logic)
# ---------------------------------------------------------------------------
st.html(
    f"""
    <div class="section">
        <span class="sec-t">{T("02 · UPLOAD DATA")}</span>
        <span class="sec-line"></span>
        <span class="sec-hint">CSV · TXT · XLSX</span>
    </div>
    """,
)

uploaded_files = st.file_uploader(
    f"Upload {TSUB(subsystem)} data file(s)",
    type=["csv", "txt", "xlsx"],
    accept_multiple_files=subsystem != "ACV",
    label_visibility="collapsed",
)
if uploaded_files is None:
    uploaded_files = []
elif not isinstance(uploaded_files, list):
    uploaded_files = [uploaded_files]

upload_sig = tuple((f.name, f.size) for f in uploaded_files)
if st.session_state.get("upload_sig") != upload_sig:
    st.session_state["upload_sig"] = upload_sig
    st.session_state["ran"] = False
    st.session_state["ran_batch"] = False

if not uploaded_files:
    st.html(
        f"""
        <div class="await">
            <div class="await-t">{T("AWAITING STREAM")}</div>
            <div class="await-s">{T("no files loaded — drop one or more CSV · TXT · XLSX exports above")}</div>
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
            f'{T("Rows")} <em>{len(stream_df):,}</em> · {T("span")} <em>{span_s / 60:.1f} min</em> · '
            f'{T("sample")} <em>{rate_hz:.0f} Hz</em> · {T("gaps")} <em>{n_gaps}</em></div>'
        )
    except Exception:
        stream_df = None
    if len(uploaded_files) > 1:
        total_kb = sum(f.size for f in uploaded_files) / 1024
        file_list = " &nbsp;·&nbsp; ".join(f.name for f in uploaded_files)
        st.html(
            f'<div class="file-chip">[BATCH] {len(uploaded_files)} {T("FILES")} &nbsp;—&nbsp; {total_kb:,.1f} KB {T("TOTAL")}</div>'
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
        model_names = [bundle["best"]] + [
            m for m in bundle["models"].keys() if m != bundle["best"]
        ]
        model_key = f"model_choice_{subsystem}"
        if st.session_state.get(model_key) not in model_names:
            st.session_state.pop(model_key, None)
        model_choice = st.selectbox(
            T("Prediction model"),
            model_names,
            index=0,
            key=model_key,
        )
        submitted = st.form_submit_button(T("Run prediction"), width="stretch")

    if submitted or st.session_state.get("ran"):
        st.session_state["ran"] = True

        st.html(
            f"""
            <div class="section" id="output">
                <span class="sec-t">{T("03 · OUTPUT")}</span>
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
                prog = st.progress(0, text=T("Segmenting the stream..."))
                segs = dp.segment_stream(df)
                prog.progress(30, text=f"{len(segs)}{T(' cycles found — classifying...')}")
                preds = dp.run_inference(df, entry["model"], entry["scaler"])
                prog.progress(75, text=T("Running Monte Carlo simulation..."))
                mc = dd.run_monte_carlo(preds, lang=st.session_state.get("lang", "en"))
                prog.progress(100, text=T("Done"))
                prog.empty()
            except Exception as exc:
                preds = None
                mc = None
                door_error = str(exc)

            if door_error is not None:
                st.error(T("Could not analyse this file as a Door data stream — ") + door_error)
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
                                    <div class="fcard-t">{T("Cycle timeline")}</div>
                                    <div class="fcard-s">{T("each bar = one cycle · red = abnormal resistance")}</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.timeline_svg(preds, theme=theme, lang=st.session_state.get("lang", "en")), 200)
                st.html(
                    f"""
                            <div class="mini-stats">
                                <span class="chip">{T("Cycles")} <b>{n_total}</b></span>
                                <span class="chip">{T("Abnormal")} <b>{n_abnormal}</b></span>
                                <span class="chip">{T("Normal")} <b>{n_normal}</b></span>
                                <span class="chip up">{T("Abnormal rate")} <b>{rate:.0f}%</b></span>
                            </div>
                        </div>

                        <div class="fcard">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["gauge"]}</div>
                                <div>
                                    <div class="fcard-t">{T("Analysis summary")}</div>
                                    <div class="fcard-s">{model_tag}</div>
                                </div>
                            </div>
                            <div class="stat-big"><b class="count" data-val="{n_total}" data-dec="0">0</b><span>{T("cycles")}</span></div>
                            <div class="sum-rows">
                                <div class="sum-row"><span>{T("Abnormal")}</span><b style="color:var(--red-soft)">{n_abnormal}</b></div>
                                <div class="sum-row"><span>{T("Normal")}</span><b>{n_normal}</b></div>
                                <div class="sum-row"><span>{T("Mean confidence")}</span><b>{mean_conf * 100:.0f}%</b></div>
                                <div class="sum-row"><span>{T("Stream length")}</span><b>{dur_min:.0f} min</b></div>
                            </div>
                            <div class="up-foot"><span class="tgl-dot"></span>{n_abnormal}{T(" cycles flagged for inspection")}</div>
                        </div>

                        <div class="fcard fcard-wide">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["bar"]}</div>
                                <div>
                                    <div class="fcard-t">{T("Monte Carlo — fault distribution")}</div>
                                    <div class="fcard-s">{mc['n_trials']:,}{T(" trials × ")}{mc['horizon']:,}{T(" future cycles")}</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.mc_hist_svg(mc, theme=theme, lang=st.session_state.get("lang", "en")), 300)
                st.html(
                    f"""
                            <div class="mini-stats">
                                <span class="chip">{T("Mean")} <b>{mc['mean']:,.1f}</b></span>
                                <span class="chip">P95 <b>{mc['p95']:,.1f}</b></span>
                                <span class="chip">P99 <b>{mc['p99']:,.1f}</b></span>
                            </div>
                        </div>

                        <div class="fcard">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["pulse"]}</div>
                                <div>
                                    <div class="fcard-t">{T("Cycle risk score")}</div>
                                    <div class="fcard-s">{T("model P(abnormal) per cycle · 10 bins")}</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.risk_hist_svg(preds, theme=theme, lang=st.session_state.get("lang", "en")), 260)
                st.html(
                    f"""
                            <div class="up-foot"><span class="tgl-dot"></span>{T("risk re-sampled in the simulation")}</div>
                        </div>

                        <div class="fcard fcard-wide">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["trend"]}</div>
                                <div>
                                    <div class="fcard-t">{T("Survival curve")}</div>
                                    <div class="fcard-s">{T("Probability of no fault vs cycles ahead")}</div>
                                </div>
                            </div>
                    """,
                )
                chart_frame(dd.survival_svg(mc, theme=theme, lang=st.session_state.get("lang", "en")), 300)
                st.html(
                    f"""
                            <div class="mini-stats">
                                <span class="chip">{T("Median cycles to fault")} <b>{mc['median_cycles_disp']}</b></span>
                                <span class="chip">{T("Median time")} <b>{mc['median_hours']}</b></span>
                                <span class="chip up">{T("P(≥1 in 100)")} <b>{mc['p_ge1_100'] * 100:.0f}%</b></span>
                            </div>
                        </div>

                        <div class="fcard">
                            <div class="fcard-h">
                                <div class="fcard-ic">{ICONS["clock"]}</div>
                                <div>
                                    <div class="fcard-t">{T("Forecast summary")}</div>
                                    <div class="fcard-s">{T("from the Monte Carlo run")}</div>
                                </div>
                            </div>
                            <div class="sum-rows" style="margin-top:0.4rem">
                                <div class="sum-row"><span>{T("Expected faults / 1,000")}</span><b>{mc['mean']:,.1f}</b></div>
                                <div class="sum-row"><span>{T("P50 faults")}</span><b>{mc['p50']:,.1f}</b></div>
                                <div class="sum-row"><span>{T("P95 faults")}</span><b>{mc['p95']:,.1f}</b></div>
                                <div class="sum-row"><span>{T("P99 faults")}</span><b>{mc['p99']:,.1f}</b></div>
                                <div class="sum-row"><span>{T("P(≥1 in 100 cycles)")}</span><b>{mc['p_ge1_100'] * 100:.0f}%</b></div>
                                <div class="sum-row"><span>{T("Median time to fault")}</span><b>{mc['median_hours']}</b></div>
                            </div>
                            <div class="up-foot"><span class="chip">{T("1,000 future cycles")}</span></div>
                        </div>

                    </div>
                    """,
                )

                st.html(
                    f"""
                    <div class="section">
                        <span class="sec-t">{T("PREDICTIONS TABLE")}</span>
                        <span class="sec-line"></span>
                    </div>
                    """,
                )
                disp = preds.copy()
                disp["status"] = disp["status"].map(
                    {dp.LABEL_ABNORMAL: T("Abnormal resistance"), dp.LABEL_NORMAL: T("Normal")}
                )
                disp["flag"] = np.where(
                    disp["status"] == T("Abnormal resistance"), T("● abnormal"), "—"
                )
                show_ab = st.toggle(T("Abnormal only"), key="door_filter")
                view = disp[disp["status"] == T("Abnormal resistance")] if show_ab else disp

                def _flag_rows(row):
                    bg = (
                        "background-color: rgba(255, 45, 85, 0.06);"
                        if row["flag"] != "—"
                        else ""
                    )
                    return [bg] * len(row)

                st.dataframe(
                    view.style.apply(_flag_rows, axis=1),
                    width="stretch",
                    hide_index=True,
                    height=min(340, 44 + 35 * len(view)),
                    column_config={
                        "segment_id": st.column_config.TextColumn(T("Segment")),
                        "start_time": st.column_config.TextColumn(T("Start")),
                        "end_time": st.column_config.TextColumn(T("End")),
                        "operation": st.column_config.TextColumn(T("Operation")),
                        "status": st.column_config.TextColumn(T("Status")),
                        "flag": st.column_config.TextColumn(T("Flag")),
                        "n_rows": st.column_config.NumberColumn(T("Rows")),
                        "confidence": st.column_config.ProgressColumn(
                            T("Confidence"),
                            min_value=0.0,
                            max_value=1.0,
                            format="%.0f%%",
                        ),
                    },
                )

                st.html(
                    f"""
                    <div class="section">
                        <span class="sec-t">{T("INSPECT A CYCLE")}</span>
                        <span class="sec-line"></span>
                        <span class="sec-hint">{T("CURRENT &amp; POSITION VS STROKE")}</span>
                    </div>
                    """,
                )
                status_zh = (
                    preds["status"]
                    .map({dp.LABEL_ABNORMAL: T("Abnormal resistance"), dp.LABEL_NORMAL: T("Normal")})
                )
                insp_options = [
                    f"{r.segment_id} · {r.operation} · {status_zh[i]} · {T('conf')} {r.confidence * 100:.0f}%"
                    for i, r in enumerate(preds.itertuples())
                ]
                insp_idx = st.selectbox(
                    T("Cycle"),
                    range(len(insp_options)),
                    format_func=lambda i: insp_options[i],
                    key="door_inspect",
                )
                sel = preds.iloc[insp_idx]
                seg_mask = (df["t"] >= dp.parse_time(sel["start_time"])) & (
                    df["t"] <= dp.parse_time(sel["end_time"])
                )
                chart_frame(dd.cycle_detail_svg(df[seg_mask], sel, theme=theme, lang=st.session_state.get("lang", "en")), 320)

                st.download_button(
                    label=T("Download predictions"),
                    data=(
                        preds[
                            ["segment_id", "start_time", "end_time", "operation", "status", "n_rows"]
                        ]
                        .assign(status=lambda d: d["status"].map(
                            {dp.LABEL_ABNORMAL: T("Abnormal resistance"), dp.LABEL_NORMAL: T("Normal")}
                        ))
                        .rename(columns={k: T(k) for k in (
                            "segment_id", "start_time", "end_time", "operation", "status", "n_rows"
                        )})
                        .to_csv(index=False)
                        .encode("utf-8-sig")
                    ),
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
                st.error(
                    T("Could not analyse this file as an ") + TSUB("SHM") + T(" stress stream — ") + shm_error
                )
            else:
                tag = (
                    f"{model_choice} · CV 1−MAPE {shm_score:.3f}"
                    if shm_score is not None
                    else model_choice
                )
                st.html(
                    f"""
                    <div class="result-banner">
                        <span>{T("Cumulative fatigue damage estimate — ")}<b>{damage:.4f}</b>
                        {T(" · higher values mean closer to the fatigue limit.")}</span>
                    </div>
                    <div class="fcard">
                        <div class="fcard-h">
                            <div class="fcard-ic">{ICONS["gauge"]}</div>
                            <div>
                                <div class="fcard-t">{T("Fatigue damage")}</div>
                                <div class="fcard-s">{tag}</div>
                            </div>
                        </div>
                        <div class="stat-big"><b class="count" data-val="{damage:.4f}" data-dec="4">0.0000</b><span>{T("cumulative damage")}</span></div>
                        <div class="sum-rows">
                            <div class="sum-row"><span>{T("Fatigue failure threshold")}</span><b>1.0000</b></div>
                            <div class="sum-row"><span>{T("Remaining margin")}</span><b>{max(1.0 - damage, 0.0):.4f}</b></div>
                            <div class="sum-row"><span>{T("Samples analysed")}</span><b>{len(x):,}</b></div>
                        </div>
                    </div>
                    """,
                )
                chart_frame(shm_signal_svg(x, theme=theme), 240)
                st.download_button(
                    label=T("Download prediction"),
                    data=pd.DataFrame(
                        {T("file_id"): [uploaded_file.name], T("prediction"): [round(damage, 6)]}
                    ).to_csv(index=False).encode("utf-8-sig"),
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
                st.error(T("Could not analyse this file as a Rail Corrugation recording — ") + rail_error)
            else:
                tag = (
                    f"{model_choice} · CV macro F1 {rail_score:.3f}"
                    if rail_score is not None
                    else model_choice
                )
                label_disp = T(label)
                flag = label != "Normal"
                st.html(
                    f"""
                    <div class="result-banner">
                        <span>{T("Classification — ")}<b>{label_disp.upper()}</b>
                        {T(" · confidence ")}{conf * 100:.0f}%</span>
                    </div>
                    <div class="fcard">
                        <div class="fcard-h">
                            <div class="fcard-ic">{ICONS["pulse"]}</div>
                            <div>
                                <div class="fcard-t">{T("Corrugation verdict")}</div>
                                <div class="fcard-s">{tag}</div>
                            </div>
                        </div>
                        <div class="stat-big">{label_disp}<span></span></div>
                        <div class="sum-rows">
                            <div class="sum-row"><span>{T("Confidence")}</span><b>{conf * 100:.0f}%</b></div>
                            <div class="sum-row"><span>{T("Duration")}</span><b>1.0 s · 10 kHz</b></div>
                        </div>
                    </div>
                    """,
                )
                bar_rows = [
                    (T(lab), float(p), lab == label) for lab, p in zip(rp.LABELS, proba)
                ]
                chart_frame(bars_svg(bar_rows, theme=theme), 30 + 40 * 3)
                st.download_button(
                    label=T("Download prediction"),
                    data=pd.DataFrame(
                        {T("file_id"): [uploaded_file.name], T("prediction"): [T(label)]}
                    ).to_csv(index=False).encode("utf-8-sig"),
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
                st.error(
                    T("Could not analyse this file as an ") + TSUB("ACV") + T(" case — ") + acv_error
                )
            else:
                tag = (
                    f"{model_choice} · LOO rank-decay {acv_score:.3f}"
                    if acv_score is not None
                    else model_choice
                )
                order = np.argsort(-proba)
                ranked_list = list(ranked)
                proba_list = [float(proba[i]) for i in order]
                car_disp = lambda c: f"{T('Car ')}{c}"
                st.html(
                    f"""
                    <div class="result-banner">
                        <span>{T("Most likely faulty car — ")}<b>CAR {ranked_list[0]}</b>
                        {T(" · probability ")}{proba_list[0] * 100:.0f}%</span>
                    </div>
                    <div class="fcard">
                        <div class="fcard-h">
                            <div class="fcard-ic">{ICONS["bar"]}</div>
                            <div>
                                <div class="fcard-t">{T("Car leak-probability ranking")}</div>
                                <div class="fcard-s">{tag}</div>
                            </div>
                        </div>
                        <div class="sum-rows">
                            <div class="sum-row"><span>{T("Top pick")}</span><b>{car_disp(ranked_list[0])}</b></div>
                            <div class="sum-row"><span>{T("Runner-up")}</span><b>{car_disp(ranked_list[1])}</b></div>
                            <div class="sum-row"><span>{T("Cars ranked")}</span><b>{len(ranked_list)}</b></div>
                        </div>
                    </div>
                    """,
                )
                rows = [
                    (car_disp(c), p, i == 0)
                    for i, (c, p) in enumerate(zip(ranked_list, proba_list))
                ]
                chart_frame(bars_svg(rows, theme=theme), 30 + 40 * len(rows))
                st.download_button(
                    label=T("Download ranking"),
                    data=pd.DataFrame(
                        {T("file_id"): [uploaded_file.name], T("ranked_cars"): ["|".join(ranked_list)]}
                    ).to_csv(index=False).encode("utf-8-sig"),
                    file_name="acv_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            st.html(
                f"""
                <div class="result-banner">
                    <span>{T("This subsystem is not wired to a model yet.")}</span>
                </div>
                """,
            )

    if len(uploaded_files) > 1:
        render_batch_section(subsystem, uploaded_files, model_choice, theme, submitted)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="made-footer">
        <span class="made-line"></span>
        <span>MADE BY · JOEL · JORDAN · JIREH · ELSON</span>
        <span class="made-line"></span>
    </div>
    """,
)

# ---------------------------------------------------------------------------
# Interactive effects (spotlight buttons, 3D card tilt, stat count-up)
# ---------------------------------------------------------------------------
st.html(
    """
    <script>
    (function () {
        if (window.__rsfx) return;
        window.__rsfx = true;

        var raf = null, lastBtn = null, lastCard = null;

        document.addEventListener("mousemove", function (e) {
            if (raf) return;
            raf = requestAnimationFrame(function () {
                raf = null;

                var btn = e.target.closest("button");
                if (btn !== lastBtn) {
                    if (lastBtn) {
                        lastBtn.style.setProperty("--mx", null);
                        lastBtn.style.setProperty("--my", null);
                    }
                    lastBtn = btn;
                }
                if (btn) {
                    var br = btn.getBoundingClientRect();
                    btn.style.setProperty("--mx", (e.clientX - br.left) + "px");
                    btn.style.setProperty("--my", (e.clientY - br.top) + "px");
                }

                var card = e.target.closest(".fcard");
                if (card !== lastCard) {
                    if (lastCard) {
                        lastCard.style.setProperty("--rx", "0deg");
                        lastCard.style.setProperty("--ry", "0deg");
                    }
                    lastCard = card;
                }
                if (card) {
                    var cr = card.getBoundingClientRect();
                    var px = (e.clientX - cr.left) / cr.width - 0.5;
                    var py = (e.clientY - cr.top) / cr.height - 0.5;
                    card.style.setProperty("--rx", (py * -7).toFixed(2) + "deg");
                    card.style.setProperty("--ry", (px * 9).toFixed(2) + "deg");
                }
            });
        });

        function animCounts() {
            document.querySelectorAll(".count:not([data-counted])").forEach(function (el) {
                el.setAttribute("data-counted", "1");
                var target = parseFloat(el.getAttribute("data-val")) || 0;
                var dec = parseInt(el.getAttribute("data-dec") || "0", 10);
                var t0 = performance.now(), dur = 900;
                (function frame(now) {
                    var p = Math.min(1, (now - t0) / dur);
                    p = 1 - Math.pow(1 - p, 3);
                    var v = target * p;
                    el.textContent = dec ? v.toFixed(dec) : Math.round(v).toLocaleString("en-US");
                    if (p < 1) requestAnimationFrame(frame);
                })(t0);
            });
        }
        animCounts();
        new MutationObserver(function () { animCounts(); })
            .observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)
