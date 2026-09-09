from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support

from engine import (
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    ENGINE_NAMES,
    ModelBundle,
    average_inference_ms,
    build_quality_report,
    dataset_statistics,
    evaluate_all,
    evaluate_challenge_set,
    extract_entities,
    load_dataset,
    predict,
    response_for_intent,
    train_all_models,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
RESPONSES_PATH = BASE_DIR / "responses.json"
CHALLENGE_PATH = BASE_DIR / "challenge_test.json"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

st.set_page_config(
    page_title="CampusConnect University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = r"""
<style>
:root {
    --ink: #10233f;
    --muted: #64748b;
    --surface: #ffffff;
    --canvas: #f4f7fb;
    --line: #e6ecf4;
    --brand: #1d4ed8;
    --brand-deep: #102a5c;
    --success: #166534;
    --success-bg: #ecfdf5;
    --warning: #92400e;
    --warning-bg: #fffbeb;
    --danger: #991b1b;
    --danger-bg: #fef2f2;
}
html { font-size: 18px; }
html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    font-size: 1rem;
}
.stApp {
    background:
        radial-gradient(circle at 86% -8%, #dbeafe 0, transparent 27rem),
        linear-gradient(180deg, #f8fbff 0%, var(--canvas) 38rem);
    color: var(--ink);
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1400px; padding: 2.5rem 3.4rem 4rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #0b1f43 0%, #12366f 100%);
    border-right: 0;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.7rem 1rem; }
[data-testid="stSidebar"] * { color: #f8fbff; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #c8d7f1 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(226, 232, 240, 0.18); }
[data-testid="stSidebar"] [data-baseweb="input"] {
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.16); border-radius: 10px;
}
[data-testid="stSidebar"] input { color: #10233f !important; -webkit-text-fill-color: #10233f !important; caret-color: #10233f !important; }
[data-testid="stSidebar"] input::placeholder { color: #64748b !important; -webkit-text-fill-color: #64748b !important; opacity: 1; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { background: transparent; border-radius: 10px; padding: .45rem .55rem; margin-bottom: .15rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.12); }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight: 600; }
.brand-mark { display:flex; align-items:center; gap:.72rem; margin-bottom:1.55rem; }
.brand-symbol { width:2.55rem; height:2.55rem; display:grid; place-items:center; border-radius:.8rem; background:linear-gradient(145deg,#60a5fa,#2563eb); color:#fff; font-size:1.25rem; box-shadow:0 10px 22px rgba(0,0,0,.18); }
.brand-name { font-size:1.4rem; font-weight:750; letter-spacing:-.02em; }
.brand-subtitle { color:#c8d7f1; font-size:.9rem; margin-top:.16rem; }
.sidebar-label, .eyebrow { color:#5d7bb1; font-size:.82rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.sidebar-label { color:#9db7e4; margin:1.35rem 0 .5rem; }
.profile-card { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.16); padding:.9rem; border-radius:.9rem; margin-bottom:.5rem; }
.profile-name { font-weight:750; font-size:1.05rem; }
.profile-meta { color:#c8d7f1; font-size:.9rem; margin-top:.23rem; line-height:1.45; }
.hero { overflow:hidden; position:relative; padding:2.35rem 2.5rem; border-radius:1.35rem; color:#fff; background:linear-gradient(120deg,#102a5c 0%,#174fa7 58%,#2774c9 100%); box-shadow:0 18px 45px rgba(23,79,167,.22); margin-bottom:1.5rem; }
.hero::after { content:""; position:absolute; width:23rem; height:23rem; border-radius:50%; top:-14rem; right:-6rem; background:rgba(191,219,254,.16); }
.hero .eyebrow { color:#bfdbfe; margin-bottom:.65rem; }
.hero h1 { color:#fff; font-size:clamp(2rem,4vw,2.8rem); line-height:1.1; letter-spacing:-.045em; margin:0; }
.hero p { color:#dceaff; max-width:45rem; font-size:1.12rem; line-height:1.6; margin:.9rem 0 0; }
.access-badge { display:inline-flex; align-items:center; gap:.42rem; position:relative; z-index:1; margin-top:1.25rem; padding:.42rem .72rem; border:1px solid rgba(255,255,255,.25); border-radius:99px; background:rgba(255,255,255,.12); font-size:.92rem; font-weight:700; }
.access-dot { width:.45rem; height:.45rem; border-radius:50%; background:#86efac; }
.section-heading { margin:1.9rem 0 .8rem; }
.section-heading h2 { color:var(--ink); font-size:1.55rem; letter-spacing:-.025em; margin:.2rem 0; }
.section-heading p { color:var(--muted); margin:0; font-size:1.03rem; line-height:1.55; }
.metric-grid, .kpi-row { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.85rem; margin:1rem 0 1.65rem; }
.metric-card, .kpi, .info-card, .status-card { padding:1.2rem 1.25rem; background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:1rem; box-shadow:0 8px 24px rgba(15,35,64,.045); }
.metric-label, .kpi .label { color:var(--muted); font-size:.91rem; font-weight:700; }
.metric-value, .kpi .value { color:var(--ink); font-size:1.34rem; font-weight:800; letter-spacing:-.02em; margin-top:.38rem; }
.metric-detail { color:#5171a4; font-size:.9rem; line-height:1.4; margin-top:.3rem; }
.info-card h3 { margin:.15rem 0 .4rem; color:var(--ink); }
.info-card p { margin:0; color:#3c506c; line-height:1.55; }
.notice { display:flex; gap:.58rem; align-items:flex-start; background:var(--warning-bg); border:1px solid #fde68a; color:var(--warning); padding:.78rem .9rem; border-radius:.85rem; font-size:1rem; line-height:1.55; margin:1rem 0; }
.notice-icon { font-weight:900; }
[data-testid="stChatMessage"] { background:rgba(255,255,255,.88); border:1px solid var(--line); border-radius:1rem; padding:.7rem .85rem; margin-bottom:.7rem; box-shadow:0 7px 20px rgba(15,35,64,.035); }
[data-testid="stChatMessage"] p { color:#263a58; font-size:1.06rem; line-height:1.62; }
[data-testid="stChatInput"] { border:1px solid #cbd9ee; border-radius:1rem; background:#fff; box-shadow:0 10px 28px rgba(15,35,64,.09); }
[data-testid="stChatInput"] textarea { font-size:1.07rem; }
.stButton > button, [data-testid="stFormSubmitButton"] > button { border:0; border-radius:.7rem; background:linear-gradient(135deg,#1d4ed8,#2563eb); color:white; font-size:1rem; font-weight:700; min-height:3rem; transition:transform .15s ease, box-shadow .15s ease; box-shadow:0 6px 14px rgba(37,99,235,.18); }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { border:0; color:white; transform:translateY(-1px); box-shadow:0 10px 20px rgba(37,99,235,.26); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:1rem; overflow:hidden; box-shadow:0 8px 24px rgba(15,35,64,.045); }
[data-testid="stDataFrame"] [role="columnheader"], [data-testid="stDataFrame"] [role="gridcell"] { font-size:.98rem !important; }
.stCaption, [data-testid="stCaptionContainer"] p { font-size:.95rem !important; line-height:1.55; }
.engine-chip { display:inline-flex; padding:.5rem .85rem; border-radius:999px; background:#eff6ff; border:1px solid #bfdbfe; color:#174ea6; font-weight:800; }
.confidence-good, .confidence-mid, .confidence-low { display:inline-flex; padding:.22rem .52rem; border-radius:999px; font-weight:800; font-size:.84rem; }
.confidence-good { background:var(--success-bg); color:var(--success); }
.confidence-mid { background:var(--warning-bg); color:var(--warning); }
.confidence-low { background:var(--danger-bg); color:var(--danger); }
.feedback-note { padding:.6rem .75rem; background:#f8fafc; border:1px solid var(--line); border-radius:.8rem; margin-top:.55rem; color:#465a76; }
.success-box { padding:.9rem 1rem; background:var(--success-bg); color:var(--success); border:1px solid #bbf7d0; border-radius:.85rem; margin:.7rem 0; }
.warning-box { padding:.9rem 1rem; background:var(--warning-bg); color:var(--warning); border:1px solid #fde68a; border-radius:.85rem; margin:.7rem 0; }
.danger-box { padding:.9rem 1rem; background:var(--danger-bg); color:var(--danger); border:1px solid #fecaca; border-radius:.85rem; margin:.7rem 0; }
.comparison-win { padding:.8rem 1rem; background:#eff6ff; color:#174ea6; border:1px solid #bfdbfe; border-radius:.85rem; }
@media (max-width: 900px) { .metric-grid, .kpi-row { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 760px) { html{font-size:16px;} .block-container{padding:1.25rem 1rem 2.5rem;} .hero{padding:1.65rem 1.4rem;border-radius:1rem;} .metric-grid,.kpi-row{grid-template-columns:1fr;} }
</style>
"""


def load_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def file_signature(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


@st.cache_data(show_spinner=False)
def get_data(signature: str) -> Dict[str, Any]:
    _ = signature
    return load_dataset(DATASET_PATH)


@st.cache_data(show_spinner=False)
def get_responses(signature: str) -> Dict[str, List[str]]:
    _ = signature
    if not RESPONSES_PATH.exists():
        return {}
    with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_challenge_cases(signature: str) -> List[Dict[str, str]]:
    _ = signature
    if not CHALLENGE_PATH.exists():
        return []
    with open(CHALLENGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


@st.cache_resource(show_spinner=False)
def get_models(signature: str) -> ModelBundle:
    _ = signature
    return train_all_models(DATASET_PATH)


@st.cache_data(show_spinner=False)
def get_evaluations(signature: str) -> Dict[str, Any]:
    _ = signature
    return evaluate_all(get_models(signature))


@st.cache_data(show_spinner=False)
def get_challenge_evaluations(
    signature: str,
    threshold: float,
    margin: float,
) -> Dict[str, Any]:
    cases = get_challenge_cases(signature)
    models = get_models(signature)
    return {
        engine: evaluate_challenge_set(models, cases, engine, threshold, margin)
        for engine in ENGINE_NAMES
    }


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.80:
        css = "confidence-good"
    elif confidence >= 0.60:
        css = "confidence-mid"
    else:
        css = "confidence-low"
    return f'<span class="{css}">{confidence:.2%} confidence</span>'


def ensure_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("feedback_ids", set())
    st.session_state.setdefault("last_compare", None)
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("threshold", DEFAULT_FALLBACK_THRESHOLD)
    st.session_state.setdefault("margin_threshold", DEFAULT_MARGIN_THRESHOLD)


def log_feedback(
    question: str,
    engine: str,
    intent: str,
    confidence: float,
    rating: str,
    comment: str = "",
) -> None:
    file_exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "question", "engine", "intent", "confidence", "rating", "comment"]
            )
        writer.writerow(
            [datetime.now().isoformat(timespec="seconds"), question, engine, intent, f"{confidence:.6f}", rating, comment]
        )


def read_feedback() -> pd.DataFrame:
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "question", "engine", "intent", "confidence", "rating", "comment"])
    try:
        return pd.read_csv(FEEDBACK_PATH)
    except Exception:
        return pd.DataFrame()


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><div class="brand-symbol">🎓</div><div>'
            '<div class="brand-name">CampusConnect</div><div class="brand-subtitle">ML Intent Classification</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Chatbot", "Model Evaluation", "Dataset Explorer", "System Workflow", "Feedback Analytics"],
            label_visibility="collapsed",
        )
        st.markdown('<div class="sidebar-label">Machine Learning Engine</div>', unsafe_allow_html=True)
        engine = st.radio(
            "Engine",
            ENGINE_NAMES,
            index=1,
            key="engine_choice",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-label">Confidence Control</div>', unsafe_allow_html=True)
        st.session_state.threshold = st.slider(
            "Fallback threshold",
            min_value=0.40,
            max_value=0.90,
            value=float(st.session_state.threshold),
            step=0.05,
            help="Predictions below this confidence are treated as uncertain.",
        )
        st.session_state.margin_threshold = st.slider(
            "Top-1 / Top-2 margin",
            min_value=0.00,
            max_value=0.25,
            value=float(st.session_state.margin_threshold),
            step=0.01,
            help="Small probability gaps can indicate model uncertainty.",
        )

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_compare = None
            st.session_state.feedback_ids = set()
            st.rerun()

        data = get_data(file_signature(DATASET_PATH))
        stats = dataset_statistics(data)
        st.markdown(
            f'<div class="profile-card"><div class="profile-name">Development Approach</div>'
            f'<div class="profile-meta">Option 1: NLP + Machine Learning<br>'
            f'TF-IDF → Naive Bayes / SVM<br>Embedding → BiLSTM</div></div>'
            f'<div class="profile-card"><div class="profile-name">Dataset</div>'
            f'<div class="profile-meta">{stats["intents"]} intents · {stats["patterns"]} patterns<br>'
            f'Shared 80/20 stratified split</div></div>',
            unsafe_allow_html=True,
        )
    return page, engine


def render_message_details(message: Dict[str, Any]) -> None:
    if message["role"] != "assistant":
        st.write(message["content"])
        return

    st.write(message["content"])
    intent = message.get("intent", "unknown")
    confidence = float(message.get("confidence", 0.0))
    engine = message.get("engine", "")
    question = message.get("question", "")
    fallback = bool(message.get("fallback", False))

    box_class = "warning-box" if fallback else "feedback-note"
    st.markdown(
        f'<div class="{box_class}"><b>Intent:</b> {intent.replace("_", " ")} &nbsp; · &nbsp; '
        f'<b>Engine:</b> {engine} &nbsp; · &nbsp; {confidence_badge(confidence)}'
        + ("<br><b>Decision:</b> Low-confidence fallback" if fallback else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    alternatives = message.get("alternatives", [])
    if len(alternatives) > 1:
        with st.expander("Alternative intent predictions"):
            rows = [
                {"Rank": rank, "Intent": tag.replace("_", " "), "Confidence": f"{prob:.2%}"}
                for rank, (tag, prob) in enumerate(alternatives, start=1)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    entities = message.get("entities", {})
    if entities:
        with st.expander("Detected entities / data fields"):
            entity_rows = [{"Entity": key.replace("_", " ").title(), "Value": value} for key, value in entities.items()]
            st.dataframe(pd.DataFrame(entity_rows), hide_index=True, use_container_width=True)

    msg_id = message.get("id")
    if msg_id in st.session_state.feedback_ids:
        st.caption("Feedback recorded for this answer.")
        return

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("👍", key=f"up_{msg_id}"):
            log_feedback(question, engine, intent, confidence, "Helpful")
            st.session_state.feedback_ids.add(msg_id)
            st.rerun()
    with c2:
        if st.button("👎", key=f"down_{msg_id}"):
            log_feedback(question, engine, intent, confidence, "Not helpful")
            st.session_state.feedback_ids.add(msg_id)
            st.rerun()
    with c3:
        st.caption("Was this answer useful?")


def add_chat_turn(
    engine: str,
    question: str,
    intent: str,
    confidence: float,
    response: str,
    alternatives: List[tuple[str, float]],
    fallback: bool,
    entities: Dict[str, str],
) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    msg_id = f"m{len(st.session_state.messages) + 1}_{int(time.time() * 1000)}"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "engine": engine,
            "intent": intent,
            "confidence": confidence,
            "alternatives": alternatives,
            "fallback": fallback,
            "entities": entities,
            "question": question,
            "id": msg_id,
        }
    )


def render_chatbot(models: ModelBundle, data: Dict[str, Any], responses: Dict[str, List[str]], engine: str) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">University FAQ Assistant</div>'
        '<h1>Ask your university question.</h1>'
        '<p>Select an ML engine, enter a question, and the system will classify your intent, '
        'check confidence, extract useful entities, and retrieve an answer from the supplied university data.</p>'
        '<div class="access-badge"><span class="access-dot"></span> Models trained and ready</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current Engine", engine, "Selected classifier")
    with c2:
        metric_card("Fallback Threshold", f"{st.session_state.threshold:.0%}", "Adjustable in sidebar")
    with c3:
        metric_card("Margin Threshold", f"{st.session_state.margin_threshold:.0%}", "Top-1 vs Top-2")
    with c4:
        metric_card("Dataset Intents", str(len(data.get("intents", []))), "Supported topics")

    st.markdown('<div class="section-heading"><div class="eyebrow">Quick Questions</div><h2>Try the university assistant</h2><p>Use one of these examples or type your own.</p></div>', unsafe_allow_html=True)
    quick_prompts = [
        "Where is the library?",
        "When is my Data Structures exam?",
        "How much is the Diploma in Computer Science?",
        "How do I reset my university password?",
        "Where can I pay my tuition fee?",
        "Who handles scholarships?",
    ]
    cols = st.columns(3)
    for idx, prompt in enumerate(quick_prompts):
        with cols[idx % 3]:
            if st.button(prompt, key=f"quick_{idx}", use_container_width=True):
                st.session_state.pending_question = prompt

    compare_mode = st.toggle(
        "Compare all three engines for the same question",
        value=False,
        help="Runs Naive Bayes, SVM and LSTM on the exact same input.",
    )

    for message in st.session_state.messages:
        role = message["role"]
        avatar = "🎓" if role == "assistant" else "🧑‍🎓"
        with st.chat_message(role, avatar=avatar):
            render_message_details(message)

    user_prompt = st.chat_input("Ask about courses, exams, fees, library, campus, IT support...")
    pending = st.session_state.pop("pending_question", None)
    question = pending or user_prompt

    if question:
        threshold = st.session_state.threshold
        margin_threshold = st.session_state.margin_threshold
        with st.spinner(f"Running {engine} model..."):
            result = predict(models, engine, question, threshold, margin_threshold)
            entities = extract_entities(data, question)
            if result.is_fallback:
                response = (
                    "Sorry, I cannot confidently match that question to a supported university service. "
                    "Please rephrase it or ask about courses, exams, fees, student services, facilities, "
                    "library, campus locations, or IT support."
                )
                shown_intent = "unknown"
            else:
                shown_intent = result.intent
                response = response_for_intent(data, responses, result.intent, question)

            comparison = {}
            if compare_mode:
                for model_name in ENGINE_NAMES:
                    comp_result = predict(models, model_name, question, threshold, margin_threshold)
                    comp_intent = "unknown" if comp_result.is_fallback else comp_result.intent
                    comparison[model_name] = {
                        "intent": comp_intent,
                        "confidence": comp_result.confidence,
                        "fallback": comp_result.is_fallback,
                        "response": response_for_intent(data, responses, comp_intent, question)
                        if not comp_result.is_fallback else "Low-confidence fallback",
                    }

        add_chat_turn(
            engine,
            question,
            shown_intent,
            result.confidence,
            response,
            result.alternatives,
            result.is_fallback,
            entities,
        )
        st.session_state.last_compare = comparison if compare_mode else None
        st.rerun()

    if st.session_state.get("last_compare"):
        results = st.session_state.last_compare
        st.markdown('<div class="section-heading"><div class="eyebrow">Model Comparison</div><h2>Same question, three classifiers</h2><p>Use this to demonstrate how model predictions can agree or disagree.</p></div>', unsafe_allow_html=True)
        rows = []
        for model_name, item in results.items():
            rows.append(
                {
                    "Engine": model_name,
                    "Predicted Intent": item["intent"].replace("_", " "),
                    "Confidence": item["confidence"],
                    "Fallback": "Yes" if item["fallback"] else "No",
                }
            )
        comp_df = pd.DataFrame(rows)
        view_df = comp_df.copy()
        view_df["Confidence"] = view_df["Confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(view_df, hide_index=True, use_container_width=True)
        intents = {item["intent"] for item in results.values()}
        if len(intents) == 1:
            st.markdown('<div class="success-box"><b>Model agreement:</b> all three engines returned the same intent.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box"><b>Model disagreement:</b> the three engines returned different intent predictions.</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="notice"><div class="notice-icon">!</div><div><b>Important:</b> '
        'Answers come from the supplied university dataset. Verify official fees, deadlines, '
        'policies and emergency information against current university announcements.</div></div>',
        unsafe_allow_html=True,
    )


def render_evaluation(evaluations: Dict[str, Any], models: ModelBundle, data_signature: str) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Model Evaluation</div>'
        '<h1>Compare Naive Bayes, SVM and LSTM.</h1>'
        '<p>All three models use the same labelled dataset and shared stratified 80/20 hold-out test set.</p></div>',
        unsafe_allow_html=True,
    )

    rows = []
    for engine in ENGINE_NAMES:
        r = evaluations[engine]
        rows.append(
            {
                "Model": engine,
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1 Score": r.f1,
                "Macro F1": r.macro_f1,
                "Training (s)": r.training_seconds,
                "Inference (ms)": average_inference_ms(models, engine, models.x_test, repeats=3),
            }
        )
    df = pd.DataFrame(rows)
    display_df = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
    display_df["Training (s)"] = display_df["Training (s)"].map(lambda x: f"{x:.2f}")
    display_df["Inference (ms)"] = display_df["Inference (ms)"].map(lambda x: f"{x:.2f}")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    best_f1 = max(rows, key=lambda row: row["F1 Score"])
    best_accuracy = max(rows, key=lambda row: row["Accuracy"])
    best_speed = min(rows, key=lambda row: row["Inference (ms)"])

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Best Weighted F1", best_f1["Model"], f"{best_f1['F1 Score']:.2%}")
    with c2:
        metric_card("Best Accuracy", best_accuracy["Model"], f"{best_accuracy['Accuracy']:.2%}")
    with c3:
        metric_card("Fastest Inference", best_speed["Model"], f"{best_speed['Inference (ms)']:.2f} ms")

    st.markdown('<div class="section-heading"><h2>Metric comparison</h2><p>These charts summarize the held-out test performance.</p></div>', unsafe_allow_html=True)
    metric_name = st.selectbox("Chart metric", ["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df["Model"], df[metric_name])
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by model")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-heading"><h2>Confusion matrix</h2><p>Choose a model to inspect intent-level errors.</p></div>', unsafe_allow_html=True)
    chosen = st.selectbox("Model for confusion matrix", ENGINE_NAMES, key="cm_model")
    result = evaluations[chosen]
    fig, ax = plt.subplots(figsize=(15, 11))
    im = ax.imshow(result.confusion, interpolation="nearest", aspect="auto")
    ax.set_title(f"{chosen} — Confusion Matrix")
    ax.set_xlabel("Predicted Intent")
    ax.set_ylabel("True Intent")
    labels = [x.replace("_", "\n") for x in result.labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    precision, recall, f1, support = precision_recall_fscore_support(
        result.y_true,
        result.predictions,
        labels=result.labels,
        zero_division=0,
    )
    per_intent = pd.DataFrame(
        {"Intent": result.labels, "Precision": precision, "Recall": recall, "F1": f1, "Support": support}
    ).sort_values("F1")
    styled = per_intent.copy()
    for col in ["Precision", "Recall", "F1"]:
        styled[col] = styled[col].map(lambda x: f"{x:.2%}")
    st.markdown('<div class="section-heading"><h2>Per-intent performance</h2><p>Useful for explaining which university services are difficult to classify.</p></div>', unsafe_allow_html=True)
    st.dataframe(styled, hide_index=True, use_container_width=True)

    history = models.lstm.history
    if history:
        hist = pd.DataFrame(history)
        st.markdown('<div class="section-heading"><h2>LSTM training history</h2><p>Training and validation curves are retained for this model run.</p></div>', unsafe_allow_html=True)
        curve = st.selectbox("LSTM history metric", ["Loss", "Accuracy"], key="lstm_curve")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        if curve == "Loss":
            ax.plot(hist["epoch"], hist["train_loss"], label="Training loss")
            ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
            ax.set_ylabel("Loss")
        else:
            ax.plot(hist["epoch"], hist["train_accuracy"], label="Training accuracy")
            ax.plot(hist["epoch"], hist["val_accuracy"], label="Validation accuracy")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
        ax.set_xlabel("Epoch")
        ax.set_title(f"LSTM {curve} history")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    threshold = st.session_state.threshold
    margin = st.session_state.margin_threshold
    challenge = get_challenge_evaluations(data_signature, threshold, margin)
    st.markdown('<div class="section-heading"><h2>Unseen challenge set</h2><p>These hand-written paraphrases are separate from the random train/test split and are useful for testing generalization.</p></div>', unsafe_allow_html=True)
    ch_rows = []
    for engine in ENGINE_NAMES:
        ch = challenge[engine]
        ch_rows.append({
            "Model": engine,
            "Challenge Accuracy": ch["accuracy"],
            "Weighted F1": ch["f1"],
            "Fallbacks": ch["fallback_count"],
            "Total Cases": ch["total"],
        })
    ch_df = pd.DataFrame(ch_rows)
    ch_display = ch_df.copy()
    ch_display["Challenge Accuracy"] = ch_display["Challenge Accuracy"].map(lambda x: f"{x:.2%}")
    ch_display["Weighted F1"] = ch_display["Weighted F1"].map(lambda x: f"{x:.2%}")
    st.dataframe(ch_display, hide_index=True, use_container_width=True)

    chosen_ch = st.selectbox("Challenge model details", ENGINE_NAMES, key="challenge_model")
    detail_df = pd.DataFrame(challenge[chosen_ch]["rows"])
    if not detail_df.empty:
        detail_df["Confidence"] = detail_df["Confidence"].map(lambda x: f"{x:.2%}")
    st.dataframe(detail_df, hide_index=True, use_container_width=True)


def render_dataset_explorer(data: Dict[str, Any]) -> None:
    stats = dataset_statistics(data)
    quality = build_quality_report(data)
    st.markdown(
        '<div class="hero"><div class="eyebrow">Dataset Explorer</div><h1>The supplied University dataset.</h1>'
        '<p>Inspect intents, training patterns, responses, structured university records and data-quality signals.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Intents", str(stats["intents"]), "Unique intent classes")
    with c2: metric_card("Patterns", str(stats["patterns"]), "Training examples")
    with c3: metric_card("Responses", str(stats["responses"]), "Prepared response entries")
    with c4: metric_card("Avg / Intent", f"{stats['avg_patterns_per_intent']:.1f}", "Training patterns per class")

    q1, q2, q3, q4 = st.columns(4)
    with q1: metric_card("Pattern Conflicts", str(quality.conflicting_patterns), "Cross-intent duplicates")
    with q2: metric_card("Empty Patterns", str(quality.empty_patterns), "Should be zero")
    with q3: metric_card("Smallest Class", f"{quality.min_patterns}", quality.intents_under_20[0] if quality.intents_under_20 else "No class under 20")
    with q4: metric_card("Largest Class", f"{quality.max_patterns}", stats["largest_intent"])

    st.markdown('<div class="section-heading"><h2>Intent distribution</h2><p>Inspect class balance before discussing model performance.</p></div>', unsafe_allow_html=True)
    balance_df = pd.DataFrame(
        [{"Intent": k, "Patterns": v} for k, v in stats["intent_counts"].items()]
    ).sort_values("Patterns", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(balance_df["Intent"], balance_df["Patterns"])
    ax.invert_yaxis()
    ax.set_xlabel("Training patterns")
    ax.set_title("Training examples by intent")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    records = []
    for item in data["intents"]:
        records.append(
            {
                "Intent": item["tag"],
                "Training Patterns": len(item.get("patterns", [])),
                "Responses": len(item.get("responses", [])),
                "Example": item.get("patterns", [""])[0],
            }
        )
    intent_df = pd.DataFrame(records).sort_values("Training Patterns", ascending=False)
    st.dataframe(intent_df, hide_index=True, use_container_width=True)

    selected_intent = st.selectbox("Inspect intent", intent_df["Intent"].tolist())
    item = next(x for x in data["intents"] if x["tag"] == selected_intent)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="info-card"><h3>Training patterns</h3><p>Examples used by the classifiers.</p></div>', unsafe_allow_html=True)
        for pattern in item.get("patterns", []):
            st.write("•", pattern)
    with c2:
        st.markdown('<div class="info-card"><h3>Prepared responses</h3><p>Controlled answers for this intent.</p></div>', unsafe_allow_html=True)
        for response in item.get("responses", []):
            st.write("•", response)

    st.markdown('<div class="section-heading"><h2>Structured university data</h2><p>Records used by the decision/data layer.</p></div>', unsafe_allow_html=True)
    structured_keys = list(stats["structured_sections"].keys())
    if structured_keys:
        section = st.selectbox("Section", structured_keys, format_func=lambda x: x.replace("_", " ").title())
        st.dataframe(pd.DataFrame(data.get(section, [])), hide_index=True, use_container_width=True)


def render_workflow() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">System Workflow</div><h1>University Chatbot architecture.</h1>'
        '<p>This implementation follows the ML-based development approach in the assignment brief.</p></div>',
        unsafe_allow_html=True,
    )
    st.code(
        r"""
UNIVERSITY CHATBOT
        |
        v
   STREAMLIT UI
        |
        +----------------------------------+
        |                                  |
        v                                  v
 Select ML Engine                    User Question
        |                                  |
        +----------------+-----------------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      Naive Bayes       SVM            LSTM
      TF-IDF + NB      TF-IDF + SVM    Tokenize + Embedding
          |              |              |
          +--------------+--------------+
                         |
                         v
                INTENT CLASSIFICATION
                         |
                         v
               CONFIDENCE / MARGIN CHECK
                  |               |
                 High            Low
                  |               |
                  v               v
          ENTITY / DATA LOGIC   FALLBACK
                  |
                  v
        STRUCTURED DATA + responses.json
                  |
                  v
             CHATBOT ANSWER
                  |
                  v
              USER FEEDBACK
                  |
                  v
             feedback.csv
""",
        language="text",
    )

    cols = st.columns(3)
    cards = [
        ("Naive Bayes", "TF-IDF + Multinomial Naive Bayes", "Fast probabilistic baseline for text classification."),
        ("SVM", "TF-IDF + Linear SVM", "Strong traditional classifier for sparse text features."),
        ("LSTM", "Embedding + BiLSTM", "Neural sequence model that can learn word-order patterns."),
    ]
    for col, (title, tech, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="info-card"><h3>{title}</h3><p><b>{tech}</b><br>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading"><h2>Why the confidence layer?</h2><p>The classifier is allowed to refuse uncertain questions instead of always forcing one of the supported intents.</p></div>', unsafe_allow_html=True)


def render_feedback_analytics() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Feedback Analytics</div><h1>Measure user satisfaction.</h1>'
        '<p>Chatbot feedback is persisted locally as CSV so it can be reviewed and included in usability evaluation.</p></div>',
        unsafe_allow_html=True,
    )
    df = read_feedback()
    if df.empty:
        st.info("No persistent feedback has been recorded yet. Use 👍 or 👎 on chatbot answers to create the first records.")
        return

    helpful = int((df["rating"] == "Helpful").sum())
    not_helpful = int((df["rating"] == "Not helpful").sum())
    total = len(df)
    satisfaction = helpful / total if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Ratings", str(total), "Persistent CSV records")
    with c2: metric_card("Helpful", str(helpful), "Positive ratings")
    with c3: metric_card("Not Helpful", str(not_helpful), "Negative ratings")
    with c4: metric_card("Satisfaction", f"{satisfaction:.1%}", "Helpful / total")

    st.markdown('<div class="section-heading"><h2>Ratings by engine</h2></div>', unsafe_allow_html=True)
    engine_table = pd.crosstab(df["engine"], df["rating"])
    st.dataframe(engine_table, use_container_width=True)

    st.markdown('<div class="section-heading"><h2>Feedback records</h2><p>These rows can be exported for the project report.</p></div>', unsafe_allow_html=True)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button(
        "Download feedback.csv",
        data=FEEDBACK_PATH.read_bytes(),
        file_name="feedback.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    load_css()
    ensure_session_state()
    signature = file_signature(DATASET_PATH)
    data = get_data(signature)
    responses = get_responses(signature)
    for item in data.get("intents", []):
        # Dataset responses remain the source of truth when present.
        if item.get("responses"):
            responses[item["tag"]] = item["responses"]

    with st.spinner("Training Naive Bayes, SVM and LSTM models..."):
        models = get_models(signature)

    page, engine = render_sidebar()

    if page == "Chatbot":
        render_chatbot(models, data, responses, engine)
    elif page == "Model Evaluation":
        evaluations = get_evaluations(signature)
        render_evaluation(evaluations, models, signature)
    elif page == "Dataset Explorer":
        render_dataset_explorer(data)
    elif page == "System Workflow":
        render_workflow()
    else:
        render_feedback_analytics()


if __name__ == "__main__":
    main()
