from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import precision_recall_fscore_support

from engine import (
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    ENGINE_NAMES,
    ModelBundle,
    average_inference_ms,
    build_quality_report,
    clean_text,
    dataset_statistics,
    decide_visual_response,
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
CAMPUS_MAP_PATH = BASE_DIR / "assets" / "campus_map.pdf"

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

/* Engineer Access intentionally uses a real Streamlit container so it has the same
   visual card treatment as Public Use without relying on Expander styling. */
[data-testid="stSidebar"] .st-key-engineer_access_card {
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.16);
    border-radius: .9rem;
    padding: .9rem;
    margin-top: .5rem;
    margin-bottom: .55rem;
    box-shadow: none;
}
[data-testid="stSidebar"] .st-key-engineer_access_card [data-testid="stTextInput"] > div {
    margin-top: .15rem;
}
[data-testid="stSidebar"] .st-key-engineer_access_card [data-baseweb="input"] {
    background: #ffffff !important;
    border: 1px solid rgba(255,255,255,.28) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .st-key-engineer_access_card input {
    color: #10233f !important;
    -webkit-text-fill-color: #10233f !important;
}
[data-testid="stSidebar"] .st-key-engineer_access_card input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] .st-key-engineer_access_card [data-testid="stFormSubmitButton"] > button {
    width: 100%;
    margin-top: .25rem;
}

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

.engine-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin:1rem 0 1.25rem; }
.single-engine-wrap { max-width: 860px; margin: 1rem auto 1.25rem; }
.single-selector-heading { margin: 1.6rem 0 .75rem; }
 .single-card-kicker { margin-bottom: 1rem; }
 .selected-engine-row { display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:.55rem 0; }
 .engine-tech-inline { color:#5b6f8c; font-size:.88rem; }
 .single-answer-text { color:#263a58; line-height:1.7; font-size:1.04rem; white-space:pre-line; margin:1rem 0; }
.single-selector-heading h2 { color: var(--ink); font-size: 1.35rem; margin: .15rem 0 .3rem; }
.single-selector-heading p { color: var(--muted); margin: 0; font-size: .98rem; line-height: 1.5; }
.selector-title { color: var(--ink); font-weight: 800; font-size: 1rem; margin-bottom: .3rem; }
.view-mode-chip { display:inline-flex; padding:.42rem .75rem; border-radius:999px; background:#f8fafc; color:#475569; border:1px solid var(--line); font-weight:800; font-size:.82rem; }
.engine-answer-card { background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:1.05rem; padding:1.15rem; min-height:240px; box-shadow:0 10px 28px rgba(15,35,64,.06); position:relative; overflow:hidden; }
.engine-answer-card.focused { border:2px solid #2563eb; box-shadow:0 14px 32px rgba(37,99,235,.12); }
.engine-answer-card .engine-title { font-size:1.12rem; font-weight:850; color:var(--ink); margin-bottom:.18rem; }
.engine-answer-card .engine-tech { color:#5b6f8c; font-size:.84rem; margin-bottom:.8rem; }
.engine-answer-card .answer-text { color:#263a58; line-height:1.6; font-size:1rem; white-space:pre-line; margin:.75rem 0; }
.engine-answer-card .meta-row { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin-bottom:.55rem; }
.engine-badge { display:inline-flex; padding:.24rem .55rem; border-radius:999px; background:#eff6ff; color:#174ea6; border:1px solid #bfdbfe; font-weight:800; font-size:.78rem; }
.intent-badge { display:inline-flex; padding:.24rem .55rem; border-radius:999px; background:#f8fafc; color:#475569; border:1px solid var(--line); font-weight:750; font-size:.78rem; }
.answer-status { font-size:.8rem; font-weight:800; margin-top:.25rem; }
.answer-status.ok { color:var(--success); }
.answer-status.fallback { color:var(--danger); }
#latest-ai-answer { scroll-margin-top: 90px; }
.compare-summary { padding:.8rem 1rem; background:#f8fbff; border:1px solid #dbeafe; border-radius:.9rem; margin:1rem 0; color:#35506f; }
.map-pdf-panel { margin-top:.85rem; padding:.85rem; border:1px solid #dbeafe; border-radius:.95rem; background:#f8fbff; }
.map-pdf-title { color:#174ea6; font-weight:800; font-size:.95rem; margin-bottom:.55rem; }
.map-pdf-note { color:#64748b; font-size:.82rem; line-height:1.45; margin-top:.5rem; }
@media (max-width: 1050px) { .engine-grid { grid-template-columns:1fr; } }

@media (max-width: 900px) { .metric-grid, .kpi-row { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 760px) { html{font-size:16px;} .block-container{padding:1.25rem 1rem 2.5rem;} .hero{padding:1.65rem 1.4rem;border-radius:1rem;} .metric-grid,.kpi-row{grid-template-columns:1fr;} }

.visual-support { margin:1rem 0 .25rem; padding:.85rem; border:1px solid #dbeafe; border-radius:.9rem; background:#f8fbff; }
.visual-support-title { color:#174ea6; font-weight:800; font-size:.9rem; margin-bottom:.55rem; }
.visual-support-reason { color:#64748b; font-size:.8rem; margin-top:.5rem; line-height:1.45; }
.visual-svg-wrap { width:100%; overflow:hidden; border-radius:.75rem; border:1px solid #dbeafe; background:#fff; }
.visual-data-badge { display:inline-flex; padding:.22rem .5rem; border-radius:999px; background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe; font-weight:800; font-size:.77rem; }
.visual-contact { display:grid; gap:.28rem; padding:.65rem .75rem; background:#fff; border:1px solid #e6ecf4; border-radius:.7rem; color:#334155; font-size:.88rem; }
.visual-table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:.4rem; }
.visual-table th,.visual-table td { text-align:left; padding:.45rem .5rem; border-bottom:1px solid #e6ecf4; }
.visual-table th { color:#475569; background:#f8fafc; }
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
    st.session_state.setdefault("scroll_to_answer", False)
    st.session_state.setdefault("view_mode", "Single Engine View")
    st.session_state.setdefault("single_engine", "Naive Bayes")
    st.session_state.setdefault("threshold", DEFAULT_FALLBACK_THRESHOLD)
    st.session_state.setdefault("margin_threshold", DEFAULT_MARGIN_THRESHOLD)
    st.session_state.setdefault("access_role", "Public")


def get_engineer_password() -> str:
    """Read the engineer password from Streamlit secrets or an environment variable.

    A local demo fallback is kept for assignment use. For deployment, set
    CAMPUSCONNECT_ENGINEER_PASSWORD or the engineer_password Streamlit secret.
    """
    try:
        secret_value = st.secrets.get("engineer_password")
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    return os.getenv("CAMPUSCONNECT_ENGINEER_PASSWORD", "12345678")


def is_engineer() -> bool:
    return st.session_state.get("access_role") == "Engineer"


def public_engine() -> str:
    """Public users are restricted to Naive Bayes only."""
    return "Naive Bayes"


def scroll_to_latest_answer() -> None:
    """Scroll the parent Streamlit page to the latest chatbot answer."""
    components.html(
        """
        <script>
        (() => {
            const target = window.parent.document.getElementById('latest-ai-answer');
            if (!target) return;
            setTimeout(() => {
                target.scrollIntoView({behavior: 'smooth', block: 'start'});
            }, 120);
        })();
        </script>
        """,
        height=0,
    )


def render_latest_answer_anchor() -> None:
    st.markdown('<div id="latest-ai-answer" style="height:1px;scroll-margin-top:90px;"></div>', unsafe_allow_html=True)


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


def render_sidebar() -> tuple[str, str, bool]:
    """Render role-aware navigation. Public users are hard-limited to chatbot + single view."""
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><div class="brand-symbol">🎓</div><div>'
            '<div class="brand-name">CampusConnect</div><div class="brand-subtitle">University Chatbot</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        engineer = is_engineer()

        if engineer:
            st.markdown('<div class="sidebar-label">Access</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="profile-card"><div class="profile-name">Engineer Mode</div>'
                '<div class="profile-meta">Advanced evaluation and model-comparison tools are enabled.</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
            page = st.radio(
                "Navigation",
                ["Chatbot", "Model Evaluation", "Dataset Explorer", "Feedback Analytics"],
                label_visibility="collapsed",
                key="engineer_navigation",
            )

            st.markdown('<div class="sidebar-label">Answer Display</div>', unsafe_allow_html=True)
            view_mode = st.radio(
                "Answer display mode",
                ["Compare View", "Single Engine View"],
                index=0 if st.session_state.view_mode == "Compare View" else 1,
                key="view_mode_choice",
                label_visibility="collapsed",
                help="Compare View shows all three answers. Single Engine View shows only the algorithm selected inside the answer card.",
            )
            st.session_state.view_mode = view_mode

            if st.button("Engineer logout", use_container_width=True, key="engineer_logout"):
                st.session_state.access_role = "Public"
                st.session_state.view_mode = "Single Engine View"
                st.session_state.single_engine = "Naive Bayes"
                st.session_state.pending_question = None
                st.rerun()
        else:
            # Public users cannot select pages or compare mode.
            page = "Chatbot"
            view_mode = "Single Engine View"
            st.session_state.view_mode = view_mode

            st.markdown('<div class="sidebar-label">Access</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="profile-card"><div class="profile-name">Public Use</div>'
                '<div class="profile-meta">Chatbot and Single Engine View only.</div></div>',
                unsafe_allow_html=True,
            )

        if st.button("Clear chat history", use_container_width=True, key="clear_chat_history"):
            st.session_state.messages = []
            st.session_state.last_compare = None
            st.session_state.feedback_ids = set()
            st.session_state.pending_question = None
            st.session_state.scroll_to_answer = False
            st.rerun()

        if not engineer:
            # Public users see Engineer Access as a card with the same visual treatment
            # as Public Use. It is always visible, but engineer-only pages stay hidden
            # until the password is verified.
            with st.container(key="engineer_access_card", border=True):
                st.markdown(
                    '<div class="profile-name">Engineer Access</div>'
                    '<div class="profile-meta">For engineering, evaluation and model-comparison tools.</div>',
                    unsafe_allow_html=True,
                )
                with st.form("engineer_login_form", clear_on_submit=False, border=False):
                    password = st.text_input(
                        "Engineer password",
                        type="password",
                        key="engineer_password_input",
                        placeholder="Enter engineer password",
                        label_visibility="visible",
                    )
                    submitted = st.form_submit_button("Engineer sign in", use_container_width=True)
                    if submitted:
                        if password == get_engineer_password():
                            st.session_state.access_role = "Engineer"
                            st.session_state.view_mode = "Compare View"
                            st.session_state.pending_question = None
                            st.rerun()
                        else:
                            st.error("Incorrect engineer password.")

        # Dataset / development metadata is engineer-only.
        if engineer:
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

    return page, view_mode, engineer


def _svg_visual(kind: str, target: str = "") -> str:
    """Create a compact illustrative visual directly inside app.py."""
    target = html.escape(target or "Campus")
    title_map = {"library_map":"Library", "campus_map":"Campus Location", "parking_map":"Parking", "hostel_map":"Hostel", "dining_map":"Dining"}
    title = title_map.get(kind, target)
    y_map = {"library_map":105, "campus_map":70, "parking_map":135, "hostel_map":165, "dining_map":105}
    y = y_map.get(kind,105)
    return "".join([
        f'<svg viewBox="0 0 760 270" width="100%" role="img" aria-label="Illustrative {html.escape(title)} visual">',
        '<rect x="0" y="0" width="760" height="270" rx="16" fill="#f8fbff"/>',
        '<path d="M40 210 C120 160 150 120 220 150 S360 235 430 170 S580 60 720 95" fill="none" stroke="#cbd5e1" stroke-width="26" stroke-linecap="round"/>',
        '<path d="M55 42 L705 42" stroke="#dbeafe" stroke-width="4"/><path d="M85 230 L690 230" stroke="#dbeafe" stroke-width="4"/>',
        '<rect x="70" y="72" width="110" height="58" rx="10" fill="#e0ecff" stroke="#93c5fd"/><text x="125" y="106" text-anchor="middle" font-size="16" fill="#1e3a8a">Gate 1</text>',
        '<rect x="250" y="58" width="140" height="76" rx="12" fill="#eef2ff" stroke="#a5b4fc"/><text x="320" y="102" text-anchor="middle" font-size="16" fill="#3730a3">Block M</text>',
        '<rect x="470" y="145" width="140" height="76" rx="12" fill="#eff6ff" stroke="#93c5fd"/><text x="540" y="189" text-anchor="middle" font-size="16" fill="#1d4ed8">West Campus</text>',
        f'<circle cx="430" cy="{y}" r="22" fill="#2563eb" opacity=".15"/><circle cx="430" cy="{y}" r="10" fill="#2563eb"/>',
        f'<line x1="430" y1="{y-13}" x2="430" y2="{y-38}" stroke="#2563eb" stroke-width="2"/>',
        f'<rect x="350" y="{y-76}" width="160" height="30" rx="15" fill="#102a5c"/><text x="430" y="{y-56}" text-anchor="middle" font-size="13" fill="white">{target}</text>',
        f'<text x="34" y="28" font-size="17" font-weight="700" fill="#10233f">Illustrative {html.escape(title)}</text>',
        '<text x="725" y="28" text-anchor="end" font-size="11" fill="#64748b">Not an official map</text></svg>'
    ])



def show_campus_map_pdf(
    query: str,
    title: str = "Campus Location Map",
    button_key: str | None = None,
) -> None:
    """Display the supplied campus_map.pdf without requiring the optional Streamlit PDF component."""
    if not CAMPUS_MAP_PATH.exists():
        st.warning("Campus map PDF is not available in the assets folder.")
        return

    st.markdown(
        f'<div class="map-pdf-panel"><div class="map-pdf-title">🗺️ {html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    # Use a browser-native PDF iframe instead of st.pdf().
    # This avoids the `streamlit-pdf` optional component dependency entirely.
    pdf_bytes = CAMPUS_MAP_PATH.read_bytes()
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" '
        'width="100%" height="650" loading="lazy" '
        'style="border:1px solid #dbeafe;border-radius:12px;background:#fff;"></iframe>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download Campus Map PDF",
        data=pdf_bytes,
        file_name="campus_map.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=button_key or f"campus_map_download_{abs(hash((query, title))) % 10**12}",
    )

    st.markdown(
        '<div class="map-pdf-note">The campus map PDF is displayed because this question is location/map related.</div></div>',
        unsafe_allow_html=True,
    )


def should_show_campus_map_pdf(intent: str, query: str, data: Dict[str, Any]) -> bool:
    """Return True for map/location questions that can be supported by the campus map PDF."""
    location_intents = {
        "library_location",
        "campus_location",
        "parking",
        "hostel",
        "campus_dining",
    }
    if intent in location_intents:
        return True

    q = clean_text(query)
    location_phrases = (
        "map", "location", "located", "where", "direction", "directions",
        "route", "get to", "find", "near", "block", "gate"
    )
    has_location_language = any(p in q for p in location_phrases)
    entities = extract_entities(data, query)
    has_place_entity = any(k in entities for k in ("place", "department", "gate"))
    return has_location_language and has_place_entity

def render_visual_support(visual: Dict[str, Any], data: Dict[str, Any], query: str) -> None:
    if not visual or visual.get("type") == "text":
        return
    entities = extract_entities(data, query)
    target = entities.get("place") or entities.get("department") or entities.get("programme") or visual.get("title", "")
    title = html.escape(str(visual.get("title", "Visual support")))
    reason = html.escape(str(visual.get("reason", "")))

    if visual.get("type") in {"image", "pdf"} and visual.get("needs_image"):
        show_campus_map_pdf(
            query,
            str(visual.get("title", "Campus Location Map")),
            button_key=f"map_download_visual_{abs(hash((query, str(visual.get('title', 'Campus Location Map'))))) % 10**12}",
        )
        return

    if visual.get("type") == "table" and visual.get("intent") == "course_fee_inquiry" and data.get("course_fees"):
        q = clean_text(query)
        matches=[]
        for row in data["course_fees"]:
            programme=clean_text(str(row.get("Programme", "")))
            level=clean_text(str(row.get("Level", "")))
            if (programme and programme in q) or (level and level in q):
                matches.append(row)
        if not matches: matches=data["course_fees"][:3]
        rows=''.join(f'<tr><td>{html.escape(str(r.get("Programme", "")))}</td><td>{html.escape(str(r.get("Level", "")))}</td><td>{html.escape(str(r.get("Estimated_Fee_Malaysian", "")))}</td></tr>' for r in matches[:3])
        table='<table class="visual-table"><tr><th>Programme</th><th>Level</th><th>MY Fee</th></tr>'+rows+'</table>'
        st.markdown(f'<div class="visual-support"><div class="visual-support-title">📊 {title}</div>{table}<div class="visual-support-reason">{reason}</div></div>', unsafe_allow_html=True)
        return

    if visual.get("type") == "contact" and data.get("department_contacts"):
        q=clean_text(query)
        matches=[]
        for row in data["department_contacts"]:
            name=clean_text(str(row.get("Department", "")))
            if name and name in q: matches.append(row)
        if not matches: matches=data["department_contacts"][:1]
        row=matches[0]
        st.markdown(f'<div class="visual-support"><div class="visual-support-title">👤 {title}</div><div class="visual-contact"><b>{html.escape(str(row.get("Department", "")))}</b><span>{html.escape(str(row.get("Help with", "")))}</span><span>📍 {html.escape(str(row.get("Location", "N/A")))}</span><span>☎ {html.escape(str(row.get("Phone", "N/A")))}</span></div><div class="visual-support-reason">{reason}</div></div>', unsafe_allow_html=True)


def engine_card(engine_name: str, item: Dict[str, Any], question: str, group_id: str, data: Dict[str, Any], focused: bool = False) -> None:
    title = engine_name
    tech_map = {
        "Naive Bayes": "TF-IDF + Multinomial Naive Bayes",
        "SVM": "TF-IDF + Linear SVM",
        "LSTM": "Embedding + BiLSTM",
    }
    intent = str(item.get("intent", "unknown"))
    confidence = float(item.get("confidence", 0.0))
    fallback = bool(item.get("fallback", False))
    response = str(item.get("response", ""))
    visual = item.get("visual", {}) or {}
    focused_class = " focused" if focused else ""
    status_text = "Low-confidence fallback" if fallback else "Answer generated from university data"
    status_class = "fallback" if fallback else "ok"
    visual_badge = "🗺️ Map PDF" if visual.get("needs_image") else ("📊 Data" if visual.get("type") == "table" else ("👤 Contact" if visual.get("type") == "contact" else "📝 Text"))
    visual_html = ""
    # The actual PDF is rendered with Streamlit below the HTML answer card,
    # because a native PDF viewer cannot be safely nested inside raw HTML.

    safe_response = (
        response.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
        .replace("\r", "")
    )
    st.markdown(
        f'<div class="engine-answer-card{focused_class}">'
        f'<div class="engine-title">{title}</div>'
        f'<div class="engine-tech">{tech_map.get(engine_name, "ML Intent Classifier")}</div>'
        f'<div class="meta-row">'
        f'<span class="engine-badge">{engine_name}</span>'
        f'<span class="intent-badge">{intent.replace("_", " ")}</span>'
        f'{confidence_badge(confidence)}'
        f'<span class="intent-badge">{visual_badge}</span>'
        f'</div>'
        f'<div class="answer-status {status_class}">{status_text}</div>'
        f'<div class="answer-text">{safe_response}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Streamlit-native visual components are rendered immediately below the HTML card.
    # The PDF remains visually associated with this model card.
    if visual.get("type") in {"image", "pdf"} and visual.get("needs_image"):
        show_campus_map_pdf(
            question,
            str(visual.get("title", "Campus Location Map")),
            button_key=f"map_download_{group_id}_{engine_name.lower().replace(' ', '_')}",
        )
    elif visual.get("type") != "image":
        render_visual_support(visual, data, question)

    alternatives = item.get("alternatives", [])
    if alternatives:
        with st.expander(f"{engine_name}: top predictions"):
            rows = [
                {"Rank": i, "Intent": tag.replace("_", " "), "Confidence": f"{prob:.2%}"}
                for i, (tag, prob) in enumerate(alternatives[:3], 1)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fb1, fb2 = st.columns([1, 1])
    key_base = f"{group_id}_{engine_name.lower().replace(' ', '_')}"
    already_rated = any(
        key in st.session_state.feedback_ids
        for key in (f"{key_base}_up", f"{key_base}_down")
    )
    if already_rated:
        st.caption(f"Feedback recorded for {engine_name}.")
        return
    with fb1:
        if st.button("👍", key=f"{key_base}_up", use_container_width=True, help=f"Mark {engine_name} answer as helpful"):
            log_feedback(question, engine_name, intent, confidence, "Helpful")
            st.session_state.feedback_ids.add(f"{key_base}_up")
            st.rerun()
    with fb2:
        if st.button("👎", key=f"{key_base}_down", use_container_width=True, help=f"Mark {engine_name} answer as not helpful"):
            log_feedback(question, engine_name, intent, confidence, "Not helpful")
            st.session_state.feedback_ids.add(f"{key_base}_down")
            st.rerun()


def _render_single_engine_card(
    engine_name: str,
    item: Dict[str, Any],
    question: str,
    group_id: str,
    data: Dict[str, Any],
) -> None:
    """Render the single-engine answer with the algorithm selector INSIDE the answer card."""
    tech_map = {
        "Naive Bayes": "TF-IDF + Multinomial Naive Bayes",
        "SVM": "TF-IDF + Linear SVM",
        "LSTM": "Embedding + BiLSTM",
    }
    intent = str(item.get("intent", "unknown"))
    confidence = float(item.get("confidence", 0.0))
    fallback = bool(item.get("fallback", False))
    response = str(item.get("response", ""))
    status_text = "Low-confidence fallback" if fallback else "Answer generated from university data"
    status_class = "fallback" if fallback else "ok"

    # The bordered Streamlit container is the answer card. The algorithm buttons are
    # deliberately created inside this container so they visually belong to the card.
    with st.container(border=True):
        if is_engineer():
            card_intro = (
                '<div class="single-card-kicker"><div class="eyebrow">Single Engine View</div>'
                '<h2>University chatbot answer</h2>'
                '<p>Choose the algorithm inside this answer card. The original question is kept unchanged.</p></div>'
            )
        else:
            card_intro = (
                '<div class="single-card-kicker"><div class="eyebrow">Single Engine View</div>'
                '<h2>University chatbot answer</h2>'
                '<p>Public mode uses Naive Bayes for chatbot responses.</p></div>'
            )
        st.markdown(card_intro, unsafe_allow_html=True)

        if is_engineer():
            st.markdown('<div class="selector-title">Algorithm</div>', unsafe_allow_html=True)
            current = st.session_state.single_engine
            if hasattr(st, "pills"):
                chosen = st.pills(
                    "Algorithm",
                    ENGINE_NAMES,
                    default=current,
                    selection_mode="single",
                    label_visibility="collapsed",
                    key=f"single_engine_pills_{group_id}",
                )
            else:
                chosen = st.radio(
                    "Algorithm",
                    ENGINE_NAMES,
                    index=ENGINE_NAMES.index(current),
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"single_engine_radio_{group_id}",
                )

            if chosen and chosen != st.session_state.single_engine:
                st.session_state.single_engine = chosen
                st.rerun()

            selected = st.session_state.single_engine
        else:
            # Public UI: Naive Bayes only. No algorithm selector is rendered.
            selected = public_engine()
            st.session_state.single_engine = selected
        item = st.session_state.get("last_compare", {}).get(selected, item)
        intent = str(item.get("intent", "unknown"))
        visual = item.get("visual", {}) or {}
        confidence = float(item.get("confidence", 0.0))
        fallback = bool(item.get("fallback", False))
        response = str(item.get("response", ""))
        status_text = "Low-confidence fallback" if fallback else "Answer generated from university data"
        status_class = "fallback" if fallback else "ok"

        visual_badge = "🗺️ Map PDF" if visual.get("needs_image") else ("📊 Data" if visual.get("type") == "table" else ("👤 Contact" if visual.get("type") == "contact" else "📝 Text"))
        st.markdown(
            f'<div class="selected-engine-row"><span class="engine-badge">{selected}</span>'
            f'<span class="engine-tech-inline">{tech_map.get(selected, "ML Intent Classifier")}</span>'
            f'<span class="intent-badge">{visual_badge}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="meta-row">'
            f'<span class="intent-badge">{intent.replace("_", " ")}</span>'
            f'{confidence_badge(confidence)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="answer-status {status_class}">{status_text}</div>',
            unsafe_allow_html=True,
        )
        safe_response = (
            response.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\\n", "<br>")
            .replace("\n", "<br>")
        )
        st.markdown(f'<div class="single-answer-text">{safe_response}</div>', unsafe_allow_html=True)

        visual_now = item.get("visual", {}) or {}
        if visual_now.get("type") == "pdf" and visual_now.get("needs_image"):
            show_campus_map_pdf(
                question,
                str(visual_now.get("title", "Campus Location Map")),
                button_key=f"map_download_single_{group_id}_{selected.lower().replace(' ', '_')}",
            )
        else:
            render_visual_support(visual_now, data, question)

        alternatives = item.get("alternatives", [])
        if alternatives:
            with st.expander(f"{selected}: top predictions"):
                rows = [
                    {"Rank": i, "Intent": tag.replace("_", " "), "Confidence": f"{prob:.2%}"}
                    for i, (tag, prob) in enumerate(alternatives[:3], 1)
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        key_base = f"{group_id}_{selected.lower().replace(' ', '_')}"
        already_rated = any(
            key in st.session_state.feedback_ids
            for key in (f"{key_base}_up", f"{key_base}_down")
        )
        if already_rated:
            st.caption(f"Feedback recorded for {selected}.")
        else:
            fb1, fb2 = st.columns([1, 1])
            with fb1:
                if st.button(
                    "👍",
                    key=f"{key_base}_up",
                    use_container_width=True,
                    help=f"Mark {selected} answer as helpful",
                ):
                    log_feedback(question, selected, intent, confidence, "Helpful")
                    st.session_state.feedback_ids.add(f"{key_base}_up")
                    st.rerun()
            with fb2:
                if st.button(
                    "👎",
                    key=f"{key_base}_down",
                    use_container_width=True,
                    help=f"Mark {selected} answer as not helpful",
                ):
                    log_feedback(question, selected, intent, confidence, "Not helpful")
                    st.session_state.feedback_ids.add(f"{key_base}_down")
                    st.rerun()


def render_message_details(message: Dict[str, Any], view_mode: str, selected_engine: str, data: Dict[str, Any]) -> None:
    if message["role"] != "assistant":
        st.write(message["content"])
        return

    if message.get("type") == "comparison":
        question = message.get("question", "")
        results = message.get("results", {})
        group_id = message.get("id", "comparison")

        if view_mode == "Compare View":
            st.markdown(
                '<div class="section-heading"><div class="eyebrow">Compare View</div>'
                '<h2>Same question, three model answers</h2>'
                '<p>Each engine independently classified the same user question and produced its own answer.</p></div>',
                unsafe_allow_html=True,
            )

            agree_intents = {item.get("intent", "unknown") for item in results.values()}
            if len(agree_intents) == 1:
                st.markdown(
                    f'<div class="success-box"><b>Model agreement:</b> all three engines predicted '
                    f'<b>{next(iter(agree_intents)).replace("_", " ")}</b>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="warning-box"><b>Model disagreement:</b> at least one engine predicted a different intent. '
                    'This is useful evidence for model comparison.</div>',
                    unsafe_allow_html=True,
                )

            cols = st.columns(3)
            for col, engine_name in zip(cols, ENGINE_NAMES):
                with col:
                    engine_card(engine_name, results[engine_name], question, group_id, data, False)
        else:
            # IMPORTANT: no fixed algorithm selector above the answer anymore.
            # The selector lives inside the single answer card itself.
            _render_single_engine_card(
                st.session_state.single_engine,
                results[st.session_state.single_engine],
                question,
                group_id,
                data,
            )

        entities = message.get("entities", {})
        if entities:
            with st.expander("Detected entities / data fields"):
                entity_rows = [
                    {"Entity": key.replace("_", " ").title(), "Value": value}
                    for key, value in entities.items()
                ]
                st.dataframe(pd.DataFrame(entity_rows), hide_index=True, use_container_width=True)
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

def add_comparison_turn(
    question: str,
    results: Dict[str, Dict[str, Any]],
    entities: Dict[str, str],
    focus_engine: str,
) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    group_id = f"cmp_{len(st.session_state.messages) + 1}_{int(time.time() * 1000)}"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "type": "comparison",
            "content": "Three-engine comparison",
            "question": question,
            "results": results,
            "entities": entities,
            "focus_engine": focus_engine,
            "id": group_id,
        }
    )

def render_chatbot(
    models: ModelBundle,
    data: Dict[str, Any],
    responses: Dict[str, List[str]],
    view_mode: str,
) -> None:
    if view_mode == "Compare View":
        hero_title = "Ask one question. Compare three ML answers."
        hero_desc = (
            "Every question is processed by Naive Bayes, SVM and LSTM at the same time. "
            "The three answers are displayed together so you can directly compare intent predictions, confidence and response quality."
        )
    else:
        hero_title = "Ask one question. View one ML answer."
        if is_engineer():
            hero_desc = (
                "Every question is evaluated by Naive Bayes, SVM and LSTM in the background. "
                "Switch the displayed algorithm inside the answer card without retyping the question."
            )
        else:
            hero_desc = (
                "Public mode uses Naive Bayes for university chatbot responses. "
                "The engineering model-selection controls are hidden."
            )

    st.markdown(
        f'<div class="hero"><div class="eyebrow">University FAQ Assistant</div>'
        f'<h1>{hero_title}</h1>'
        f'<p>{hero_desc}</p>'
        f'<div class="access-badge"><span class="access-dot"></span> All three models are trained and ready</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Answer View", view_mode, "Current display mode")
    with c2:
        metric_card("Fallback Threshold", f"{st.session_state.threshold:.0%}", "Applied to all models")
    with c3:
        metric_card("Margin Threshold", f"{st.session_state.margin_threshold:.0%}", "Top-1 vs Top-2")
    with c4:
        metric_card("Dataset Intents", str(len(data.get("intents", []))), "Supported topics")

    selected_engine = st.session_state.single_engine if is_engineer() else public_engine()
    if not is_engineer():
        view_mode = "Single Engine View"
        st.session_state.view_mode = view_mode
        st.session_state.single_engine = public_engine()

    if view_mode == "Compare View":
        st.markdown(
            '<div class="compare-summary"><b>Compare View is active.</b> One user question is sent to all three classifiers. '
            'Each engine returns its own intent, confidence score and university answer.</div>',
            unsafe_allow_html=True,
        )
    else:
        if is_engineer():
            single_summary = (
                '<div class="compare-summary"><b>Single Engine View is active.</b> The algorithm selector is inside each answer card. '
                'The same question is still evaluated by all three models in the background.</div>'
            )
        else:
            single_summary = (
                '<div class="compare-summary"><b>Public Single Engine View is active.</b> '
                'Naive Bayes is used for chatbot responses.</div>'
            )
        st.markdown(single_summary, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading"><div class="eyebrow">Quick Questions</div>'
        '<h2>Try the university assistant</h2><p>Use one of these examples or type your own.</p></div>',
        unsafe_allow_html=True,
    )
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
                st.session_state.scroll_to_answer = True

    last_assistant_index = max(
        (idx for idx, message in enumerate(st.session_state.messages) if message.get("role") == "assistant"),
        default=-1,
    )
    for idx, message in enumerate(st.session_state.messages):
        if idx == last_assistant_index:
            # Anchor immediately before the newest AI answer so Quick Questions can
            # smoothly scroll the viewport to the response rather than the input box.
            render_latest_answer_anchor()
        role = message["role"]
        avatar = "🎓" if role == "assistant" else "🧑‍🎓"
        with st.chat_message(role, avatar=avatar):
            render_message_details(message, view_mode, selected_engine, data)

    user_prompt = st.chat_input("Ask about courses, exams, fees, library, campus, IT support...")
    pending = st.session_state.pop("pending_question", None)
    question = pending or user_prompt

    if question:
        threshold = st.session_state.threshold
        margin_threshold = st.session_state.margin_threshold
        comparison: Dict[str, Dict[str, Any]] = {}
        entities = extract_entities(data, question)
        models_to_run = ENGINE_NAMES if is_engineer() else [public_engine()]
        spinner_text = "Running Naive Bayes, SVM and LSTM..." if is_engineer() else "Running Naive Bayes..."
        with st.spinner(spinner_text):
            for model_name in models_to_run:
                result = predict(models, model_name, question, threshold, margin_threshold)
                intent = "unknown" if result.is_fallback else result.intent
                if result.is_fallback:
                    answer = (
                        "Sorry, I cannot confidently match that question to a supported university service. "
                        "Please rephrase it or ask about courses, exams, fees, student services, facilities, "
                        "library, campus locations, or IT support."
                    )
                else:
                    answer = response_for_intent(data, responses, intent, question)

                visual = decide_visual_response(intent, question)
                if should_show_campus_map_pdf(intent, question, data):
                    visual = {
                        "needs_image": True,
                        "type": "pdf",
                        "visual_key": "campus_map_pdf",
                        "title": "Campus Location Map",
                        "reason": "This question is location/map related, so the provided campus map PDF is shown alongside the answer.",
                    }
                visual["intent"] = intent
                comparison[model_name] = {
                    "intent": intent,
                    "confidence": result.confidence,
                    "fallback": result.is_fallback,
                    "response": answer,
                    "alternatives": result.alternatives,
                    "visual": visual,
                }

        add_comparison_turn(question, comparison, entities, selected_engine)
        st.session_state.last_compare = comparison
        # The next rerun renders the new answer, then scroll_to_answer will move the viewport to it.
        st.session_state.scroll_to_answer = True if pending is not None else st.session_state.get("scroll_to_answer", False)
        st.rerun()

    if st.session_state.get("scroll_to_answer"):
        scroll_to_latest_answer()
        st.session_state.scroll_to_answer = False

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
        '<p>Inspect intent coverage, training distribution and data-quality signals.</p></div>',
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
        key="download_feedback_csv",
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

    page, view_mode, engineer = render_sidebar()

    # Server-side access guard: public sessions can never render engineer pages or Compare View.
    if not engineer:
        page = "Chatbot"
        view_mode = "Single Engine View"
        st.session_state.view_mode = "Single Engine View"

    if page == "Chatbot":
        render_chatbot(models, data, responses, view_mode)
    elif page == "Model Evaluation" and engineer:
        evaluations = get_evaluations(signature)
        render_evaluation(evaluations, models, signature)
    elif page == "Dataset Explorer" and engineer:
        render_dataset_explorer(data)
    elif page == "Feedback Analytics" and engineer:
        render_feedback_analytics()
    else:
        # Fallback guard for any unexpected navigation state.
        render_chatbot(models, data, responses, "Single Engine View")


if __name__ == "__main__":
    main()
