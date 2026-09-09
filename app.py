
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import precision_recall_fscore_support

from engine import (
    ENGINE_NAMES,
    ModelBundle,
    dataset_statistics,
    evaluate_all,
    predict,
    response_for_intent,
    load_dataset,
    train_all_models,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
RESPONSES_PATH = BASE_DIR / "responses.json"

st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = ':root {\n                --ink: #10233f;\n                --muted: #64748b;\n                --surface: #ffffff;\n                --canvas: #f4f7fb;\n                --line: #e6ecf4;\n                --brand: #1d4ed8;\n                --brand-deep: #102a5c;\n            }\n            html { font-size: 18px; }\n            html, body, [class*="css"] {\n                font-family: Inter, "Segoe UI", Arial, sans-serif;\n                font-size: 1rem;\n            }\n            .stApp {\n                background:\n                    radial-gradient(circle at 86% -8%, #dbeafe 0, transparent 27rem),\n                    linear-gradient(180deg, #f8fbff 0%, var(--canvas) 38rem);\n                color: var(--ink);\n            }\n            #MainMenu, footer { visibility: hidden; }\n            [data-testid="stHeader"] { background: transparent; }\n            .block-container {\n                max-width: 1400px;\n                padding: 2.5rem 3.4rem 4rem;\n            }\n            [data-testid="stSidebar"] {\n                background: linear-gradient(165deg, #0b1f43 0%, #12366f 100%);\n                border-right: 0;\n            }\n            [data-testid="stSidebar"] > div:first-child { padding: 1.7rem 1rem; }\n            [data-testid="stSidebar"] * { color: #f8fbff; }\n            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {\n                color: #c8d7f1 !important;\n            }\n            [data-testid="stSidebar"] hr { border-color: rgba(226, 232, 240, 0.18); }\n            [data-testid="stSidebar"] [data-baseweb="input"] {\n                background: rgba(255, 255, 255, 0.12);\n                border: 1px solid rgba(255, 255, 255, 0.16);\n                border-radius: 10px;\n            }\n            [data-testid="stSidebar"] input {\n                color: #10233f !important;\n                -webkit-text-fill-color: #10233f !important;\n                caret-color: #10233f !important;\n            }\n            [data-testid="stSidebar"] input::placeholder {\n                color: #64748b !important;\n                -webkit-text-fill-color: #64748b !important;\n                opacity: 1;\n            }\n            [data-testid="stSidebar"] [data-testid="stRadio"] label {\n                background: transparent;\n                border-radius: 10px;\n                padding: 0.45rem 0.55rem;\n                margin-bottom: 0.15rem;\n            }\n            [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {\n                background: rgba(255, 255, 255, 0.12);\n            }\n            [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight: 600; }\n            .brand-mark {\n                display: flex;\n                align-items: center;\n                gap: 0.72rem;\n                margin-bottom: 1.55rem;\n            }\n            .brand-symbol {\n                width: 2.55rem;\n                height: 2.55rem;\n                display: grid;\n                place-items: center;\n                border-radius: 0.8rem;\n                background: linear-gradient(145deg, #60a5fa, #2563eb);\n                color: white;\n                font-size: 1.25rem;\n                box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);\n            }\n            .brand-name { font-size: 1.4rem; font-weight: 750; letter-spacing: -0.02em; }\n            .brand-subtitle { color: #c8d7f1; font-size: 0.9rem; margin-top: 0.16rem; }\n            .sidebar-label, .eyebrow {\n                color: #5d7bb1;\n                font-size: 0.82rem;\n                font-weight: 800;\n                letter-spacing: 0.12em;\n                text-transform: uppercase;\n            }\n            .sidebar-label { color: #9db7e4; margin: 1.35rem 0 0.5rem; }\n            .profile-card {\n                background: rgba(255, 255, 255, 0.1);\n                border: 1px solid rgba(255, 255, 255, 0.16);\n                padding: 0.9rem;\n                border-radius: 0.9rem;\n                margin-bottom: 0.5rem;\n            }\n            .profile-name { font-weight: 750; font-size: 1.1rem; }\n            .profile-meta { color: #c8d7f1; font-size: 0.94rem; margin-top: 0.23rem; line-height: 1.45; }\n            .hero {\n                overflow: hidden;\n                position: relative;\n                padding: 2.35rem 2.5rem;\n                border-radius: 1.35rem;\n                color: #ffffff;\n                background: linear-gradient(120deg, #102a5c 0%, #174fa7 58%, #2774c9 100%);\n                box-shadow: 0 18px 45px rgba(23, 79, 167, 0.22);\n                margin-bottom: 1.5rem;\n            }\n            .hero::after {\n                content: "";\n                position: absolute;\n                width: 23rem;\n                height: 23rem;\n                border-radius: 50%;\n                top: -14rem;\n                right: -6rem;\n                background: rgba(191, 219, 254, 0.16);\n            }\n            .hero .eyebrow { color: #bfdbfe; margin-bottom: 0.65rem; }\n            .hero h1 { color: #ffffff; font-size: clamp(2rem, 4vw, 2.8rem); line-height: 1.1; letter-spacing: -0.045em; margin: 0; }\n            .hero p { color: #dceaff; max-width: 45rem; font-size: 1.16rem; line-height: 1.65; margin: 0.9rem 0 0; }\n            .access-badge {\n                display: inline-flex;\n                align-items: center;\n                gap: 0.42rem;\n                position: relative;\n                z-index: 1;\n                margin-top: 1.25rem;\n                padding: 0.42rem 0.72rem;\n                border: 1px solid rgba(255, 255, 255, 0.25);\n                border-radius: 99px;\n                background: rgba(255, 255, 255, 0.12);\n                font-size: 0.92rem;\n                font-weight: 700;\n            }\n            .access-dot { width: 0.45rem; height: 0.45rem; border-radius: 50%; background: #86efac; }\n            .section-heading { margin: 1.9rem 0 0.8rem; }\n            .section-heading h2 { color: var(--ink); font-size: 1.55rem; letter-spacing: -0.025em; margin: 0.2rem 0; }\n            .section-heading p { color: var(--muted); margin: 0; font-size: 1.08rem; line-height: 1.55; }\n            .metric-grid {\n                display: grid;\n                grid-template-columns: repeat(3, minmax(0, 1fr));\n                gap: 0.85rem;\n                margin: 1rem 0 1.65rem;\n            }\n            .metric-card {\n                padding: 1.2rem 1.25rem;\n                background: rgba(255, 255, 255, 0.92);\n                border: 1px solid var(--line);\n                border-radius: 1rem;\n                box-shadow: 0 8px 24px rgba(15, 35, 64, 0.045);\n            }\n            .metric-label { color: var(--muted); font-size: 0.94rem; font-weight: 700; }\n            .metric-value { color: var(--ink); font-size: 1.34rem; font-weight: 800; letter-spacing: -0.02em; margin-top: 0.38rem; }\n            .metric-detail { color: #5171a4; font-size: 0.94rem; line-height: 1.4; margin-top: 0.3rem; }\n            .notice {\n                display: flex;\n                gap: 0.58rem;\n                align-items: flex-start;\n                background: #fffbeb;\n                border: 1px solid #fde68a;\n                color: #92400e;\n                padding: 0.78rem 0.9rem;\n                border-radius: 0.85rem;\n                font-size: 1rem;\n                line-height: 1.55;\n                margin: 1rem 0;\n            }\n            .notice-icon { font-weight: 900; }\n            [data-testid="stChatMessage"] {\n                background: rgba(255, 255, 255, 0.88);\n                border: 1px solid var(--line);\n                border-radius: 1rem;\n                padding: 0.7rem 0.85rem;\n                margin-bottom: 0.7rem;\n                box-shadow: 0 7px 20px rgba(15, 35, 64, 0.035);\n            }\n            [data-testid="stChatMessage"] p { color: #263a58; font-size: 1.08rem; line-height: 1.65; }\n            [data-testid="stChatInput"] {\n                border: 1px solid #cbd9ee;\n                border-radius: 1rem;\n                background: #ffffff;\n                box-shadow: 0 10px 28px rgba(15, 35, 64, 0.09);\n            }\n            [data-testid="stChatInput"] textarea { font-size: 1.08rem; }\n            .stButton > button, [data-testid="stFormSubmitButton"] > button {\n                border: 0;\n                border-radius: 0.7rem;\n                background: linear-gradient(135deg, #1d4ed8, #2563eb);\n                color: white;\n                font-size: 1rem;\n                font-weight: 700;\n                min-height: 3rem;\n                transition: transform 0.15s ease, box-shadow 0.15s ease;\n                box-shadow: 0 6px 14px rgba(37, 99, 235, 0.18);\n            }\n            .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {\n                border: 0;\n                color: white;\n                transform: translateY(-1px);\n                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.26);\n            }\n            [data-testid="stDataFrame"] {\n                border: 1px solid var(--line);\n                border-radius: 1rem;\n                overflow: hidden;\n                box-shadow: 0 8px 24px rgba(15, 35, 64, 0.045);\n            }\n            [data-testid="stDataFrame"] [role="columnheader"],\n            [data-testid="stDataFrame"] [role="gridcell"] {\n                font-size: 1rem !important;\n            }\n            .stCaption, [data-testid="stCaptionContainer"] p {\n                font-size: 0.98rem !important;\n                line-height: 1.55;\n            }\n            @media (max-width: 760px) {\n                html { font-size: 16px; }\n                .block-container { padding: 1.25rem 1rem 2.5rem; }\n                .hero { padding: 1.65rem 1.4rem; border-radius: 1rem; }\n                .metric-grid { grid-template-columns: 1fr; }\n            }\n        \n\n/* University Chatbot additions */\n.engine-chip {\n    display: inline-flex;\n    align-items: center;\n    padding: 0.38rem 0.78rem;\n    border-radius: 999px;\n    background: #dbeafe;\n    color: #1e40af;\n    font-weight: 800;\n    border: 1px solid #bfdbfe;\n}\n.model-chip {\n    display: inline-flex;\n    padding: 0.32rem 0.62rem;\n    border-radius: 999px;\n    background: #eff6ff;\n    color: #1d4ed8;\n    font-weight: 750;\n    border: 1px solid #dbeafe;\n    font-size: 0.9rem;\n}\n.confidence-good, .confidence-mid, .confidence-low {\n    display: inline-block;\n    padding: 0.3rem 0.58rem;\n    border-radius: 999px;\n    font-weight: 800;\n    font-size: 0.9rem;\n}\n.confidence-good { background: #dcfce7; color: #166534; }\n.confidence-mid { background: #fef3c7; color: #92400e; }\n.confidence-low { background: #fee2e2; color: #991b1b; }\n.info-card {\n    padding: 1rem 1.15rem;\n    background: rgba(255,255,255,0.92);\n    border: 1px solid var(--line);\n    border-radius: 1rem;\n    box-shadow: 0 8px 24px rgba(15,35,64,0.045);\n}\n.info-card h3 { color: var(--ink); margin: 0 0 0.3rem; font-size: 1.05rem; }\n.info-card p { color: var(--muted); margin: 0; line-height: 1.55; }\n.kpi-row {\n    display: grid;\n    grid-template-columns: repeat(4, minmax(0, 1fr));\n    gap: .75rem;\n    margin: 1rem 0 1.5rem;\n}\n.kpi {\n    padding: .95rem 1rem;\n    border-radius: .9rem;\n    border: 1px solid var(--line);\n    background: #fff;\n}\n.kpi .label { color: var(--muted); font-size: .83rem; font-weight: 750; }\n.kpi .value { color: var(--ink); font-size: 1.35rem; font-weight: 850; margin-top: .2rem; }\n.response-card {\n    padding: .85rem 1rem;\n    border-left: 4px solid #2563eb;\n    background: #f8fbff;\n    border-radius: .7rem;\n    margin: .6rem 0;\n}\n.response-card .title { color: #5171a4; font-size: .82rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }\n.response-card .body { color: #263a58; font-size: 1.05rem; line-height: 1.6; margin-top: .25rem; }\n.feedback-note {\n    color: var(--muted);\n    font-size: .88rem;\n    margin-top: .25rem;\n}\n@media (max-width: 900px) {\n    .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }\n}\n@media (max-width: 600px) {\n    .kpi-row { grid-template-columns: 1fr; }\n}'


def load_css() -> None:
    """Inject the app CSS directly from this file.

    Keeping CSS inside app.py avoids Streamlit displaying a separate CSS file as
    page content when the project is copied or run from another directory.
    """
    html = "<style>\n" + APP_CSS + "\n</style>"
    try:
        # st.html is the most reliable way to inject a <style> block in recent
        # Streamlit versions.
        st.html(html)
    except AttributeError:
        # Compatibility fallback for older Streamlit versions.
        st.markdown(html, unsafe_allow_html=True)


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
    with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_models(signature: str) -> ModelBundle:
    _ = signature
    return train_all_models(DATASET_PATH)


@st.cache_data(show_spinner=False)
def get_evaluations(signature: str) -> Dict[str, Any]:
    _ = signature
    models = get_models(signature)
    return evaluate_all(models)


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.80:
        css = "confidence-good"
    elif confidence >= 0.55:
        css = "confidence-mid"
    else:
        css = "confidence-low"
    return f'<span class="{css}">{confidence:.2%} confidence</span>'


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-mark">
                <div class="brand-symbol">🎓</div>
                <div>
                    <div class="brand-name">University Chatbot</div>
                    <div class="brand-subtitle">ML Intent Classification</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Chatbot", "Model Evaluation", "Dataset Explorer", "System Workflow"],
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

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback = {}
            st.session_state.last_compare = None
            st.rerun()

        data = get_data(file_signature(DATASET_PATH))
        stats = dataset_statistics(data)
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-name">Development Approach</div>
                <div class="profile-meta">
                    Option 1: NLP + Machine Learning<br>
                    TF-IDF: Naive Bayes / SVM<br>
                    Tokenization + Embedding: LSTM
                </div>
            </div>
            <div class="profile-card">
                <div class="profile-name">Dataset</div>
                <div class="profile-meta">
                    {stats["intents"]} intents · {stats["patterns"]} patterns<br>
                    Shared train/test split for fair comparison
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page, engine


def ensure_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("feedback", {})
    st.session_state.setdefault("last_compare", None)


def add_chat_turn(
    engine: str,
    user_text: str,
    intent: str,
    confidence: float,
    response: str,
    alternatives: List[tuple[str, float]],
) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    message_id = len(st.session_state.messages)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "engine": engine,
            "intent": intent,
            "confidence": confidence,
            "alternatives": alternatives,
            "id": message_id,
        }
    )


def render_message_details(message: Dict[str, Any], index: int) -> None:
    if message["role"] != "assistant":
        st.write(message["content"])
        return

    st.write(message["content"])
    intent = message.get("intent", "unknown")
    confidence = float(message.get("confidence", 0.0))
    engine = message.get("engine", "")

    st.markdown(
        f"""
        <div class="feedback-note">
            <b>Intent:</b> {intent.replace("_", " ")} &nbsp; · &nbsp;
            <b>Engine:</b> {engine} &nbsp; · &nbsp;
            {confidence_badge(confidence)}
        </div>
        """,
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

    feedback = st.session_state.feedback.get(message.get("id"))
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("👍", key=f"up_{message.get('id')}", disabled=feedback is not None):
            st.session_state.feedback[message.get("id")] = "Helpful"
            st.rerun()
    with c2:
        if st.button("👎", key=f"down_{message.get('id')}", disabled=feedback is not None):
            st.session_state.feedback[message.get("id")] = "Not helpful"
            st.rerun()
    with c3:
        if feedback:
            st.caption(f"Feedback recorded: {feedback}")


def render_chatbot(models: ModelBundle, data: Dict[str, Any], responses: Dict[str, List[str]], engine: str) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">University FAQ Assistant</div>
            <h1>Ask your university question.</h1>
            <p>
                Select an ML engine, enter a question, and the model will classify the
                student's intent before retrieving a university answer.
            </p>
            <div class="access-badge"><span class="access-dot"></span> Models trained and ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading"><div class="eyebrow">Current Engine</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<span class="engine-chip">{engine}</span>', unsafe_allow_html=True)
    compare_mode = st.toggle(
        "Compare all three engines for each question",
        value=False,
        key="compare_mode",
        help="Runs the same question through Naive Bayes, SVM and LSTM so their predictions can be compared.",
    )

    quick_prompts = [
        "Where is the library?",
        "When is my Data Structures exam?",
        "How much is the Diploma in Computer Science?",
        "How do I reset my university password?",
        "Where can I pay my tuition fee?",
        "Who handles scholarships?",
    ]
    st.markdown(
        '<div class="section-heading"><h2>Quick questions</h2><p>Use one of these or type your own question below.</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for idx, prompt in enumerate(quick_prompts):
        with cols[idx % 3]:
            if st.button(prompt, key=f"quick_{idx}", use_container_width=True):
                st.session_state.pending_question = prompt

    ensure_session_state()
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        avatar = "🎓" if role == "assistant" else "🧑‍🎓"
        with st.chat_message(role, avatar=avatar):
            render_message_details(message, idx)

    user_prompt = st.chat_input("Ask about courses, exams, fees, library, campus, IT support...")
    pending = st.session_state.pop("pending_question", None)
    question = pending or user_prompt

    if question:
        with st.spinner(f"Running {engine} model..."):
            intent, confidence, alternatives = predict(models, engine, question)
            response = response_for_intent(data, responses, intent, question)

            comparison = {}
            if compare_mode:
                for model_name in ENGINE_NAMES:
                    c_intent, c_conf, _ = predict(models, model_name, question)
                    comparison[model_name] = {
                        "intent": c_intent,
                        "confidence": c_conf,
                        "response": response_for_intent(data, responses, c_intent, question),
                    }

        add_chat_turn(engine, question, intent, confidence, response, alternatives)
        if compare_mode:
            st.session_state.last_compare = {
                "question": question,
                "results": comparison,
            }
        else:
            st.session_state.last_compare = None
        st.rerun()

    if st.session_state.get("last_compare"):
        comparison = st.session_state.last_compare
        st.markdown(
            '<div class="section-heading"><h2>Engine comparison for this question</h2>'
            '<p>The same question was processed by all three models.</p></div>',
            unsafe_allow_html=True,
        )
        rows = []
        for model_name, item in comparison["results"].items():
            rows.append(
                {
                    "Engine": model_name,
                    "Predicted Intent": item["intent"].replace("_", " "),
                    "Confidence": item["confidence"],
                }
            )
        comp_df = pd.DataFrame(rows)
        display_comp = comp_df.copy()
        display_comp["Confidence"] = display_comp["Confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_comp, hide_index=True, use_container_width=True)

        best = max(rows, key=lambda x: x["Confidence"])
        st.info(
            f"Highest confidence on this question: {best['Engine']} "
            f"({best['Predicted Intent']}, {best['Confidence']:.2%})."
        )

    st.markdown(
        """
        <div class="notice">
            <div class="notice-icon">!</div>
            <div>
                <b>Important:</b> This chatbot answers from the supplied university dataset.
                For official deadlines, fees, policies, and emergency information, verify the
                latest university announcement.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluation(evaluations: Dict[str, Any], models: ModelBundle) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Model Evaluation</div>
            <h1>Compare Naive Bayes, SVM and LSTM.</h1>
            <p>
                All three models use the same labelled dataset and shared 80/20 train-test split,
                allowing a direct intent-classification comparison.
            </p>
        </div>
        """,
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
            }
        )

    df = pd.DataFrame(rows)
    display_df = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
    display_df["Training (s)"] = display_df["Training (s)"].map(lambda x: f"{x:.2f}")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    best_model = max(rows, key=lambda x: x["F1 Score"])
    st.markdown(
        f"""
        <div class="info-card">
            <h3>Best overall intent-classification result</h3>
            <p>
                Based on weighted F1 Score in this run, <b>{best_model["Model"]}</b>
                achieved <b>{best_model["F1 Score"]:.2%}</b>.
                The winner should be reported from the actual run rather than assumed beforehand.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading"><h2>Confusion matrix</h2><p>Select a model to inspect where intents are being confused.</p></div>',
        unsafe_allow_html=True,
    )
    chosen = st.selectbox("Model", ENGINE_NAMES, key="cm_model")
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

    st.markdown(
        '<div class="section-heading"><h2>Per-intent performance</h2><p>Weighted metrics are used above; this view shows class-level F1 where useful.</p></div>',
        unsafe_allow_html=True,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        result.y_true,
        result.predictions,
        labels=result.labels,
        zero_division=0,
    )
    per_intent = pd.DataFrame(
        {
            "Intent": result.labels,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    ).sort_values("F1")
    styled = per_intent.copy()
    for col in ["Precision", "Recall", "F1"]:
        styled[col] = styled[col].map(lambda x: f"{x:.2%}")
    st.dataframe(styled, hide_index=True, use_container_width=True)

    feedback_values = list(st.session_state.get("feedback", {}).values())
    if feedback_values:
        helpful = sum(v == "Helpful" for v in feedback_values)
        not_helpful = sum(v == "Not helpful" for v in feedback_values)
        total = len(feedback_values)
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Session user feedback</h3>
                <p>
                    {helpful} helpful · {not_helpful} not helpful · {total} total responses rated.
                    This is an in-session usability signal and is separate from the held-out ML metrics above.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_explorer(data: Dict[str, Any]) -> None:
    stats = dataset_statistics(data)
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Dataset Explorer</div>
            <h1>The supplied University dataset.</h1>
            <p>Review the intent taxonomy, training examples and structured university records used by the chatbot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi"><div class="label">Intents</div><div class="value">{stats["intents"]}</div></div>
            <div class="kpi"><div class="label">Patterns</div><div class="value">{stats["patterns"]}</div></div>
            <div class="kpi"><div class="label">Responses</div><div class="value">{stats["responses"]}</div></div>
            <div class="kpi"><div class="label">Avg. Patterns / Intent</div><div class="value">{stats["avg_patterns_per_intent"]:.1f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.markdown('<div class="info-card"><h3>Training patterns</h3><p>These examples teach the model how students may express this intent.</p></div>', unsafe_allow_html=True)
        for pattern in item.get("patterns", []):
            st.write("•", pattern)
    with c2:
        st.markdown('<div class="info-card"><h3>Prepared responses</h3><p>The response layer provides controlled answers for the predicted intent.</p></div>', unsafe_allow_html=True)
        for response in item.get("responses", []):
            st.write("•", response)

    st.markdown(
        '<div class="section-heading"><h2>Structured university data</h2><p>These records support dynamic, data-backed answers for timetable, exams, fees, campus places and contacts.</p></div>',
        unsafe_allow_html=True,
    )
    section = st.selectbox(
        "Section",
        list(stats["structured_sections"].keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    value = data.get(section, [])
    st.dataframe(pd.DataFrame(value), hide_index=True, use_container_width=True)


def render_workflow() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">System Workflow</div>
            <h1>University Chatbot architecture.</h1>
            <p>This implementation follows the ML-based development approach in the assignment brief.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        r"""
UNIVERSITY CHATBOT
        |
        v
   STREAMLIT UI
        |
        +-----------------------------+
        |                             |
        v                             v
Select Engine                    User Question
        |                             |
        +-------------+---------------+
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
  Naive Bayes       SVM           LSTM
  TF-IDF            TF-IDF        Tokenization
  Classifier        Classifier    + Embedding
        |             |             |
        +-------------+-------------+
                      |
                      v
             INTENT CLASSIFICATION
                      |
                      v
              DECISION / DATA LOGIC
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Structured Data          responses.json
 (fees/exams/etc.)                |
          |                       |
          +-----------+-----------+
                      |
                      v
                CHATBOT ANSWER
                      |
                      v
                 STREAMLIT UI
        """,
        language="text",
    )

    st.markdown(
        """
        <div class="section-heading">
            <h2>Why three models?</h2>
            <p>
                Naive Bayes provides a lightweight probabilistic baseline, SVM provides a strong
                linear text-classification baseline, and LSTM provides a neural-network approach
                that can model word order. They can be trained and evaluated on the same labelled
                dataset for a fair comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cards = [
        ("Naive Bayes", "TF-IDF + Multinomial Naive Bayes", "Fast baseline for text classification."),
        ("SVM", "TF-IDF + Linear SVM", "Strong traditional classifier for sparse text features."),
        ("LSTM", "Tokenization + Embedding + BiLSTM", "Neural approach that learns sequence patterns."),
    ]
    for col, (title, tech, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{title}</h3>
                    <p><b>{tech}</b><br>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    load_css()
    ensure_session_state()

    signature = file_signature(DATASET_PATH)
    data = get_data(signature)
    responses = get_responses(signature)
    # Dataset remains the source of truth if users replace dataset.json later.
    for intent_item in data.get("intents", []):
        responses[intent_item["tag"]] = intent_item.get("responses", responses.get(intent_item["tag"], []))

    with st.spinner("Training Naive Bayes, SVM and LSTM models..."):
        models = get_models(signature)

    page, engine = render_sidebar()

    if page == "Chatbot":
        render_chatbot(models, data, responses, engine)
    elif page == "Model Evaluation":
        with st.spinner("Calculating evaluation metrics..."):
            evaluations = get_evaluations(signature)
        render_evaluation(evaluations, models)
    elif page == "Dataset Explorer":
        render_dataset_explorer(data)
    else:
        render_workflow()


if __name__ == "__main__":
    main()
