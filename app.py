"""Streamlit UI — load model đã train từ train.ipynb, predict, vẽ biểu đồ."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "heart_disease_models.joblib"

FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

# Field config theo nhóm hiển thị: key -> (label, "num"|"sel", default, options, format_func)
FIELD_CONFIG = {
    "age":      ("Age (years)",            "num", 58,  (1, 120), None),
    "sex":      ("Sex",                    "sel", 1,   [0, 1],
                 lambda v: "Male" if v == 1 else "Female"),
    "trestbps": ("Resting BP (mmHg)",      "num", 130, (60, 250), None),
    "chol":     ("Cholesterol (mg/dl)",    "num", 250, (80, 700), None),
    "thalach":  ("Max Heart Rate",         "num", 150, (50, 250), None),

    "cp":       ("Chest Pain Type",        "sel", 2,   [1, 2, 3, 4],
                 lambda v: {1: "Typical angina", 2: "Atypical angina",
                            3: "Non-anginal pain", 4: "Asymptomatic"}[v]),
    "fbs":      ("Fasting Blood Sugar",    "sel", 0,   [0, 1],
                 lambda v: ">120 mg/dl" if v == 1 else "≤120 mg/dl"),
    "exang":    ("Exercise Angina",        "sel", 0,   [0, 1],
                 lambda v: "Yes" if v == 1 else "No"),
    "oldpeak":  ("ST Depression (oldpeak)", "num", 1.0, (0.0, 10.0), None),

    "restecg":  ("Resting ECG",            "sel", 1,   [0, 1, 2],
                 lambda v: {0: "Normal", 1: "ST-T abnormality",
                            2: "LV hypertrophy"}[v]),
    "slope":    ("ST Slope",               "sel", 1,   [1, 2, 3],
                 lambda v: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[v]),
    "ca":       ("Major Vessels (0-3)",    "sel", 0,   [0, 1, 2, 3], None),
    "thal":     ("Thalassemia",            "sel", 3,   [3, 6, 7],
                 lambda v: {3: "Normal", 6: "Fixed defect",
                            7: "Reversible defect"}[v]),
}

# Nhóm hiển thị — chỉ phục vụ bố cục, không ảnh hưởng thứ tự dữ liệu (FEATURES)
FIELD_GROUPS = [
    ("📋 Demographics & Vitals", ["age", "sex", "trestbps", "chol", "thalach"]),
    ("💓 Symptoms",              ["cp", "fbs", "exang", "oldpeak"]),
    ("🧪 Test Results",          ["restecg", "slope", "ca", "thal"]),
]

PINK_DISEASE  = "#E8487A"   # cảnh báo: bệnh tim
PINK_SAFE     = "#1FAE8C"   # an toàn: không bệnh tim
PINK_PRIMARY  = "#D6336C"
PINK_DARK     = "#9C2056"
PINK_SOFT_BG  = "#FFF0F5"
PINK_BORDER   = "#FFD2E4"
TEXT_MAIN     = "#3B2434"
TEXT_MUTED    = "#9C7E8E"


# ── Predict helpers (logic giữ nguyên) ─────────────────────────────────────
def transform_patient(patient: pd.DataFrame, cont_stats: dict) -> pd.DataFrame:
    """Chuyển input UI (giá trị thật) → cùng scale với CSV train."""
    p = patient.copy()
    for feat, (mean, std) in cont_stats.items():
        p[feat] = (p[feat] - mean) / std
    p["cp"]      = ((p["cp"] - 1) / 3).clip(0, 1)
    p["restecg"] = (p["restecg"] / 2).clip(0, 1)
    p["slope"]   = ((p["slope"] - 1) / 2).clip(0, 1)
    p["ca"]      = (p["ca"] / 3).clip(0, 1)
    p["thal"]    = p["thal"].map({3: 0.0, 6: 0.75, 7: 1.0}).fillna(p["thal"])
    return p


def predict_all(payload: dict, patient: pd.DataFrame) -> pd.DataFrame:
    models      = payload["models"]
    model_names = payload["model_names"]
    p = transform_patient(patient, payload["cont_stats"])

    rows = []
    for key, model in models.items():
        prob = model.predict_proba(p)[0]
        pred = int(prob[1] >= 0.5)
        rows.append({
            "model":      model_names.get(key, key),
            "prediction": "Heart Disease" if pred else "No Heart Disease",
            "confidence": prob[1] if pred else prob[0],
        })
    return pd.DataFrame(rows)


def draw_chart(preds: pd.DataFrame) -> plt.Figure:
    colors = preds["prediction"].map({"Heart Disease": PINK_DISEASE, "No Heart Disease": PINK_SAFE})

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(range(len(preds)), preds["confidence"], color=colors,
                   width=0.6, zorder=3, edgecolor="white", linewidth=1.5)

    ax.set_title("Model Predictions", fontsize=15, fontweight="bold",
                  color=PINK_DARK, pad=16)
    ax.set_ylabel("Prediction Confidence", fontsize=10.5, color=TEXT_MUTED)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels(preds["model"], rotation=-25, ha="left", fontsize=9.5, color=TEXT_MAIN)

    ax.grid(axis="y", color=PINK_BORDER, linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PINK_BORDER)
    ax.tick_params(left=False, bottom=False)
    ax.set_yticklabels([f"{t:.0%}" for t in ax.get_yticks()], color=TEXT_MUTED, fontsize=9)

    for bar, (_, row) in zip(bars, preds.iterrows()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                f"{h:.0%}", ha="center", fontsize=10.5, fontweight="bold", color=TEXT_MAIN)
        ax.text(bar.get_x() + bar.get_width() / 2, h / 2,
                row["prediction"],
                ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold", rotation=90)
    fig.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_payload() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Chưa có model! Hãy chạy train.ipynb trước (Run All)."
        )
    return joblib.load(MODEL_FILE)


def render_field(key: str) -> object:
    label, kind, default, opts, fmt = FIELD_CONFIG[key]
    if kind == "num":
        return st.number_input(
            label, min_value=opts[0], max_value=opts[1], value=default,
            step=0.1 if isinstance(default, float) else 1,
        )
    kwargs = {"format_func": fmt} if fmt else {}
    return st.selectbox(label, opts, index=opts.index(default), **kwargs)


def patient_input() -> pd.DataFrame:
    values = {}
    for group_title, keys in FIELD_GROUPS:
        st.markdown(f"**{group_title}**")
        cols = st.columns(len(keys))
        for col, key in zip(cols, keys):
            with col:
                values[key] = render_field(key)
        st.write("")
    return pd.DataFrame([values])[FEATURES]


st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT_MAIN};
}}

.stApp {{ background: {PINK_SOFT_BG}; }}

/* Header */
.app-header {{ padding: 0.4rem 0 1.4rem 0; }}
.app-title {{
    font-family: 'Poppins', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: {PINK_DARK};
    margin-bottom: 0.1rem;
}}
.app-subtitle {{
    color: {TEXT_MUTED};
    font-size: 0.98rem;
}}

/* Section headers inside form */
.stMarkdown p strong {{
    color: {PINK_DARK};
    font-family: 'Poppins', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}}

/* Card containers (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: white;
    border-radius: 18px;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1.5px solid {PINK_BORDER} !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 18px rgba(214, 51, 108, 0.07);
}}

/* Inputs */
div[data-baseweb="select"] > div, .stNumberInput input {{
    border-radius: 10px !important;
    border-color: {PINK_BORDER} !important;
}}
label[data-testid="stWidgetLabel"] p {{
    font-size: 0.84rem !important;
    color: {TEXT_MAIN} !important;
    font-weight: 500;
}}

/* Button */
.stButton>button {{
    width: 100%;
    font-weight: 600;
    font-size: 0.95rem;
    font-family: 'Poppins', sans-serif;
    letter-spacing: 0.03em;
    background: linear-gradient(135deg, {PINK_PRIMARY}, {PINK_DARK});
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    min-height: 2.6rem;
    box-shadow: 0 3px 10px rgba(214, 51, 108, 0.28);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(214, 51, 108, 0.38);
    color: white !important;
    background: linear-gradient(135deg, {PINK_DARK}, {PINK_PRIMARY});
}}
.stButton>button:active {{
    transform: translateY(0);
}}

/* Summary cards (thay st.metric) */
.summary-card {{
    background: white;
    border: 1.5px solid {PINK_BORDER};
    border-radius: 12px;
    padding: 0.65rem 0.75rem;
    min-height: 4.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.25rem;
    overflow: hidden;
}}
.summary-label {{
    font-size: 0.72rem;
    font-weight: 500;
    color: {TEXT_MUTED};
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.summary-value {{
    font-family: 'Poppins', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: {PINK_DARK};
    line-height: 1.25;
    word-break: break-word;
    overflow-wrap: break-word;
}}
.summary-value.safe {{ color: {PINK_SAFE}; }}
.summary-value.disease {{ color: {PINK_DISEASE}; }}

/* Info / error boxes */
div[data-testid="stAlert"] {{ border-radius: 14px; }}

/* Section subheaders */
h3 {{ font-family: 'Poppins', sans-serif !important; color: {TEXT_MAIN} !important; }}
</style>""", unsafe_allow_html=True)


def render_summary(label: str, value: str, css_class: str = "") -> str:
    extra = f" {css_class}" if css_class else ""
    return (
        f'<div class="summary-card">'
        f'<div class="summary-label">{label}</div>'
        f'<div class="summary-value{extra}">{value}</div>'
        f'</div>'
    )


def main() -> None:
    st.markdown(
        '<div class="app-header">'
        '<div class="app-title">🫀 Heart Disease Risk Prediction</div>'
        '<div class="app-subtitle">Nhập chỉ số bệnh nhân để xem dự đoán từ các mô hình đã train</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Patient Information")
            patient = patient_input()
            run = st.button("Predict", type="primary", use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Model Predictions")
            if run:
                try:
                    payload = load_payload()
                    preds = predict_all(payload, patient)
                    st.pyplot(draw_chart(preds))

                    best = preds.loc[preds["confidence"].idxmax()]
                    report = payload.get("report")

                    pred_class = "safe" if best["prediction"] == "No Heart Disease" else "disease"
                    best_model = report.iloc[0]["model"] if report is not None and len(report) else "—"

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(render_summary("Best Model (val)", best_model), unsafe_allow_html=True)
                    c2.markdown(render_summary("Prediction", best["prediction"], pred_class), unsafe_allow_html=True)
                    c3.markdown(render_summary("Confidence", f"{best['confidence']:.0%}"), unsafe_allow_html=True)

                    if report is not None:
                        with st.expander("Training accuracy (validation set)"):
                            st.dataframe(
                                report.assign(accuracy=report["accuracy"].map("{:.2%}".format)),
                                use_container_width=True,
                            )
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Lỗi: {e}")
            else:
                st.info("Nhập dữ liệu bệnh nhân bên trái rồi bấm **Predict** để xem kết quả.")

    st.markdown(
        f'<div style="text-align:center; color:{TEXT_MUTED}; font-size:0.8rem; '
        'margin-top:2rem;">Chỉ mang tính tham khảo — không thay thế chẩn đoán y khoa chuyên nghiệp.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()