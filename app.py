"""
app.py
------
RAG-Powered Counterfactual Risk Validation for 10-K Filings
Main Streamlit Application — Professional Light UI
"""

import io
import logging
import os
import traceback

import pandas as pd
import streamlit as st

from edgar_utils import fetch_and_clean_10k
from rag_engine import (
    build_vector_store,
    retrieve_risk_candidates,
    format_chunks_for_prompt,
)
from counterfactual_engine import run_counterfactual_validation
from comparator import build_comparison_dataframe, compute_mode_summary

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RiskProbe — 10-K Counterfactual Validator",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f5f2ed;
    color: #1a1814;
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-image: radial-gradient(ellipse 70% 40% at 100% 0%, rgba(139, 100, 60, 0.04) 0%, transparent 60%);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #eeeae3 !important;
    border-right: 1px solid #d4cfc7 !important;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Sans', sans-serif !important; }

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background: #f5f2ed !important;
    border: 1px solid #c8c3bb !important;
    border-radius: 3px !important;
    color: #1a1814 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.83rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stNumberInput input:focus {
    border-color: #8b6440 !important;
    box-shadow: 0 0 0 2px rgba(139, 100, 64, 0.12) !important;
    outline: none !important;
}

[data-testid="stSidebar"] label {
    color: #5a5550 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #6a6560 !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
}

[data-testid="stSidebar"] h2 {
    color: #1a1814 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    margin-top: 1.25rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #d4cfc7 !important;
}

/* Slider track + thumb */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #8b6440 !important;
    border-color: #8b6440 !important;
}

/* ── HIDE CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── MAIN LAYOUT ── */
.main .block-container {
    padding: 2.5rem 3.5rem 3rem 3.5rem !important;
    max-width: 1400px !important;
}

/* ── MASTHEAD ── */
.masthead {
    padding-bottom: 1.75rem;
    border-bottom: 2px solid #1a1814;
    margin-bottom: 2rem;
}
.masthead-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #8b6440;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    color: #1a1814;
    line-height: 1;
    margin: 0 0 0.5rem 0;
    font-weight: 600;
}
.masthead-title em {
    color: #8b6440;
    font-style: italic;
    font-weight: 500;
}
.masthead-desc {
    font-size: 0.85rem;
    color: #6a6560;
    letter-spacing: 0.03em;
    max-width: 600px;
    line-height: 1.6;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8b6440;
    margin: 1.75rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #d4cfc7;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #1a1814 !important;
    color: #f5f2ed !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.75rem !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: #2e2a24 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 12px rgba(26, 24, 20, 0.2) !important;
}
.stButton > button:disabled {
    background: #d4cfc7 !important;
    color: #a09890 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    color: #1a1814 !important;
    border: 1px solid #c8c3bb !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.18s ease !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #1a1814 !important;
    background: rgba(26, 24, 20, 0.04) !important;
}

/* ── METRIC GRID ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    border: 1px solid #d4cfc7;
    border-radius: 3px;
    overflow: hidden;
    margin: 1.5rem 0;
    background: #d4cfc7;
    gap: 1px;
}
.metric-card {
    background: #f5f2ed;
    padding: 1.25rem 1rem 1rem 1rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    color: #1a1814;
    line-height: 1;
    display: block;
    font-weight: 600;
}
.metric-card .val.accent { color: #8b6440; }
.metric-card .val.muted { color: #c8c3bb; }
.metric-card .lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8a8580;
    display: block;
    margin-top: 0.35rem;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgress"] > div > div {
    background: #8b6440 !important;
}
[data-testid="stProgress"] > div {
    background: #e4e0d8 !important;
    border-radius: 0 !important;
    height: 2px !important;
}

/* ── STATUS LOG ── */
.log-wrap {
    background: #fff;
    border: 1px solid #d4cfc7;
    border-radius: 3px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.status-running {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    color: #8b6440;
    padding: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.status-done {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    color: #3a7a4a;
    padding: 0.28rem 0;
    border-left: 2px solid #3a7a4a;
    padding-left: 0.6rem;
    margin: 0.2rem 0;
}
.status-error {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    color: #b84040;
    padding: 0.28rem 0.6rem;
    border-left: 2px solid #b84040;
    margin: 0.2rem 0;
    background: rgba(184, 64, 64, 0.04);
}

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #d4cfc7 !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #8a8580 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1a1814 !important;
    border-bottom-color: #1a1814 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #1a1814 !important;
    background: transparent !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid #d4cfc7 !important;
    border-radius: 3px !important;
}
[data-testid="stDataFrame"] th {
    background: #eeeae3 !important;
    color: #5a5550 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.63rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #d4cfc7 !important;
    padding: 0.6rem 0.75rem !important;
}
[data-testid="stDataFrame"] td {
    font-size: 0.8rem !important;
    color: #2a2620 !important;
    border-color: #ede9e2 !important;
    padding: 0.55rem 0.75rem !important;
}

/* ── EXPANDERS ── */
[data-testid="stExpander"] {
    border: 1px solid #d4cfc7 !important;
    border-radius: 3px !important;
    background: #fff !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #2a2620 !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { background: #f5f2ed !important; }
[data-testid="stExpander"] .streamlit-expanderContent {
    background: #fff !important;
    padding: 0 1rem 1rem 1rem !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-size: 0.82rem !important;
    border: none !important;
}
.stSuccess { background: #eef6f0 !important; color: #2d6b3a !important; border-left: 3px solid #3a7a4a !important; }
.stWarning { background: #fdf6ec !important; color: #7a5a20 !important; border-left: 3px solid #c49040 !important; }
.stError   { background: #fdf0f0 !important; color: #8b3030 !important; border-left: 3px solid #b84040 !important; }
.stInfo    { background: #eef2f8 !important; color: #2a4a7a !important; border-left: 3px solid #4a70b0 !important; }

/* ── RISK CARDS ── */
.risk-card {
    background: #fff;
    border: 1px solid #d4cfc7;
    border-radius: 3px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.6rem;
    position: relative;
    border-left: 3px solid #8b6440;
}
.risk-card-title {
    font-size: 0.88rem;
    color: #1a1814;
    font-weight: 500;
    line-height: 1.55;
    margin-bottom: 0.7rem;
}
.risk-meta {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 0.7rem;
}
.tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border-radius: 2px;
    font-weight: 400;
}
.tag-high     { background: #fdeaea; color: #b84040; }
.tag-medium   { background: #fdf4e4; color: #8b6420; }
.tag-low      { background: #f0f0f5; color: #6a6a80; }
.tag-category { background: #eef2f8; color: #3a5a90; }
.tag-chunk    { background: #eef6f0; color: #2d6b3a; }

.evidence-block {
    background: #faf8f4;
    border-left: 2px solid #d4cfc7;
    padding: 0.55rem 0.75rem;
    margin: 0.5rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.71rem;
    color: #6a6560;
    line-height: 1.65;
    font-style: italic;
}
.cf-block {
    background: #eef6f0;
    border-left: 2px solid #a8d0b4;
    padding: 0.55rem 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.76rem;
    color: #2d5a3a;
    line-height: 1.65;
}

/* ── CHUNK PREVIEW ── */
.chunk-preview {
    background: #fff;
    border: 1px solid #d4cfc7;
    border-radius: 3px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.chunk-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #8b6440;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.chunk-text {
    font-size: 0.78rem;
    color: #5a5550;
    line-height: 1.65;
}

/* ── EMPTY STATE ── */
.empty-state {
    padding: 2.5rem;
    text-align: center;
    border: 1px dashed #d4cfc7;
    border-radius: 3px;
    background: #faf8f4;
}
.empty-state-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8c3bb;
    margin-bottom: 0.5rem;
}
.empty-state-sub {
    font-size: 0.78rem;
    color: #b0aba4;
    line-height: 1.6;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid #d4cfc7 !important;
    margin: 2rem 0 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f5f2ed; }
::-webkit-scrollbar-thumb { background: #d4cfc7; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #b8b3ab; }

/* ── SPINNER ── */
[data-testid="stSpinner"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.73rem !important;
    color: #8b6440 !important;
}

/* ── TOOLTIP OVERRIDE ── */
[data-testid="stTooltipIcon"] svg { color: #a09890 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help=(
            "Your OpenAI secret key. Required to call:\n"
            "• text-embedding-3-small — embeds document chunks into vectors\n"
            "• gpt-4o-mini — runs the counterfactual validation\n\n"
            "Get yours at platform.openai.com → API Keys."
        ),
    )

    st.markdown("## Filing")

    ticker = (
        st.text_input(
            "Ticker Symbol",
            value="AAPL",
            max_chars=10,
            help=(
                "The stock exchange ticker of the company whose 10-K you want to analyze.\n\n"
                "Examples: AAPL (Apple), MSFT (Microsoft), TSLA (Tesla), JPM (JPMorgan).\n\n"
                "Must be a valid NYSE or NASDAQ ticker with a 10-K filing on SEC EDGAR."
            ),
        )
        .strip()
        .upper()
    )

    year = st.number_input(
        "Fiscal Year",
        min_value=2015,
        max_value=2024,
        value=2023,
        step=1,
        help=(
            "The fiscal year of the 10-K filing you want to analyze.\n\n"
            "Note: A company's FY2023 10-K is typically filed in early 2024, "
            "so SEC EDGAR filings may be dated one year after the fiscal year shown here.\n\n"
            "If no filing is found, try the adjacent year (e.g., 2022 or 2024)."
        ),
    )

    st.markdown("## RAG Parameters")

    top_k = st.slider(
        "Top-K per query",
        min_value=2,
        max_value=8,
        value=3,
        help=(
            "How many document chunks to retrieve per semantic search query.\n\n"
            "The pipeline runs 12 different risk-related queries. Each query fetches "
            "this many chunks from ChromaDB.\n\n"
            "• Higher K = more evidence passed to the LLM, but higher API cost and slower.\n"
            "• Lower K = faster and cheaper, but may miss relevant passages.\n\n"
            "Recommended: 3 for most analyses. Use 5–6 for longer or complex filings."
        ),
    )

    max_chunks_to_llm = st.slider(
        "Max chunks to LLM",
        min_value=10,
        max_value=40,
        value=20,
        help=(
            "The maximum number of retrieved chunks sent to the LLM for validation.\n\n"
            "After retrieval, all chunks are deduplicated and sorted by relevance score. "
            "Only the top N are included in the LLM prompt.\n\n"
            "• Higher = more thorough analysis, but higher token cost (~$0.01–0.05 per run).\n"
            "• Lower = faster and cheaper, but may miss risks in lower-ranked chunks.\n\n"
            "Recommended: 20 (balances cost and coverage)."
        ),
    )

    st.markdown("---")
    st.markdown("""
**Mode A — Aggressive**
Any speculative or hedging language ("may", "could", "no assurance") is flagged as a risk.

**Mode B — Balanced**
Requires explicit risk terms ("risk", "exposure", "adverse") with operational context.

**Mode C — Conservative**
Only quantified or imminent risks with specific dollar amounts or active legal proceedings.

---
🔴 **Robust** — all 3 modes agree
                
🟡 **Moderate** — 2 modes agree
                
⚪ **Fragile** — aggressive mode only
    """)

# ── MASTHEAD ─────────────────────────────────────────────────────
st.markdown(
    """
<div class="masthead">
    <div class="masthead-eyebrow">⬡ &nbsp; Columbia SIPA · SIPA6674 · Spring 2026</div>
    <div class="masthead-title">Risk<em>Probe</em></div>
    <div class="masthead-desc">
        RAG-powered counterfactual validation for SEC 10-K filings. 
        Retrieves candidate evidence from the entire document, then validates each chunk 
        against three distinct risk thresholds — refusing to classify when evidence is insufficient.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── RUN BUTTON ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)

col_btn, col_status = st.columns([1, 4])
with col_btn:
    run_analysis = st.button(
        "Run Analysis →",
        type="primary",
        use_container_width=True,
        disabled=(not api_key or not ticker),
    )
with col_status:
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to begin.")
    elif not ticker:
        st.warning("Enter a stock ticker symbol.")
    else:
        st.info(
            f"Ready to analyze **{ticker}** FY**{year}** — "
            f"{max_chunks_to_llm} chunks · Top-{top_k} per query · ~2–5 min"
        )

# ── PIPELINE ──────────────────────────────────────────────────────
if run_analysis:
    if not api_key:
        st.error("API key required.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown(
        f'<div class="section-label">Pipeline — {ticker} FY{year}</div>',
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0)
    log_area = st.empty()
    log_lines = []

    def log(msg, kind="run"):
        css = (
            "status-done"
            if kind == "ok"
            else ("status-error" if kind == "err" else "status-running")
        )
        prefix = "✓" if kind == "ok" else ("✗" if kind == "err" else "›")
        log_lines.append(f'<div class="{css}">{prefix}&nbsp;&nbsp;{msg}</div>')
        log_area.markdown(
            f'<div class="log-wrap">{"".join(log_lines)}</div>',
            unsafe_allow_html=True,
        )

    # STEP 1
    log("Fetching 10-K from SEC EDGAR...")
    progress_bar.progress(5)
    try:
        with st.spinner("Downloading filing from SEC EDGAR..."):
            clean_text, filing_url = fetch_and_clean_10k(ticker, int(year))
        log(
            f"Filing acquired — {len(clean_text):,} characters &nbsp;·&nbsp; <a href='{filing_url}' target='_blank' style='color:#8b6440;text-decoration:none;border-bottom:1px solid #d4c8b4'>source ↗</a>",
            "ok",
        )
        progress_bar.progress(20)
    except Exception as e:
        log(f"Acquisition failed: {str(e)}", "err")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.stop()

    # STEP 2
    log("Chunking & embedding full document into ChromaDB...")
    progress_bar.progress(25)
    try:
        with st.spinner("Building vector index (text-embedding-3-small)..."):
            vector_store = build_vector_store(clean_text, ticker, int(year), api_key)
        est = len(clean_text) // 900
        log(f"Vector store ready — ~{est} chunks indexed with chunk_id metadata", "ok")
        progress_bar.progress(45)
    except Exception as e:
        log(f"Indexing failed: {str(e)}", "err")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.stop()

    # STEP 3
    log(
        "Running multi-query RAG retrieval across financial, operational, regulatory categories..."
    )
    progress_bar.progress(50)
    try:
        with st.spinner("Retrieving candidate evidence chunks..."):
            candidates = retrieve_risk_candidates(vector_store, top_k=top_k)
            chunks_text = format_chunks_for_prompt(
                candidates, max_chunks=max_chunks_to_llm
            )
            num_chunks = min(len(candidates), max_chunks_to_llm)
        log(
            f"Retrieved {len(candidates)} unique chunks — top {num_chunks} passed to LLM",
            "ok",
        )
        progress_bar.progress(60)
    except Exception as e:
        log(f"Retrieval failed: {str(e)}", "err")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.stop()

    # STEP 4
    log(
        "Running counterfactual validation — Mode A (Aggressive), B (Balanced), C (Conservative)..."
    )
    progress_bar.progress(65)
    try:
        with st.spinner("LLM counterfactual validation in progress (gpt-4o-mini)..."):
            mode_results = run_counterfactual_validation(
                chunks_text, num_chunks, ticker, int(year), api_key
            )
        summary = compute_mode_summary(mode_results)
        log(
            f"Validation complete — "
            f"A: {summary['Mode A (Aggressive)']} risks &nbsp;·&nbsp; "
            f"B: {summary['Mode B (Balanced)']} risks &nbsp;·&nbsp; "
            f"C: {summary['Mode C (Conservative)']} risks",
            "ok",
        )
        progress_bar.progress(88)
    except Exception as e:
        log(f"Validation failed: {str(e)}", "err")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.stop()

    # STEP 5
    log("Building comparison DataFrame...")
    try:
        df = build_comparison_dataframe(mode_results, ticker, int(year))
        progress_bar.progress(100)
        log("Analysis complete.", "ok")
    except Exception as e:
        log(f"Comparison failed: {str(e)}", "err")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.stop()

    # ── RESULTS ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div class="section-label">Results — {ticker} FY{year}</div>',
        unsafe_allow_html=True,
    )

    total_risks = len(
        df[df["risk_summary"] != "No risks extracted across any threshold mode."]
    )
    robust = len(df[df["robustness"].str.startswith("🔴 Robust")])
    moderate = len(df[df["robustness"].str.startswith("🟡")])
    fragile = len(df[df["robustness"].str.startswith("⚪")])
    cons_hits = summary.get("Mode C (Conservative)", 0)

    st.markdown(
        f"""
    <div class="metric-grid">
        <div class="metric-card">
            <span class="val">{total_risks}</span>
            <span class="lbl">Total Risks</span>
        </div>
        <div class="metric-card">
            <span class="val accent">{robust}</span>
            <span class="lbl">🔴 Robust</span>
        </div>
        <div class="metric-card">
            <span class="val">{moderate}</span>
            <span class="lbl">🟡 Moderate</span>
        </div>
        <div class="metric-card">
            <span class="val muted">{fragile}</span>
            <span class="lbl">⚪ Fragile</span>
        </div>
        <div class="metric-card">
            <span class="val accent">{cons_hits}</span>
            <span class="lbl">Conservative Hits</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── TABS ─────────────────────────────────────────────────────
    tab_all, tab_a, tab_b, tab_c, tab_chunks = st.tabs(
        [
            "All Modes",
            "A · Aggressive",
            "B · Balanced",
            "C · Conservative",
            "Evidence Chunks",
        ]
    )

    with tab_all:
        display_cols = [
            c
            for c in [
                "robustness",
                "risk_summary",
                "confidence",
                "category",
                "source_chunk_id",
                "key_evidence",
                "mode_A_Aggressive",
                "mode_B_Balanced",
                "mode_C_Conservative",
            ]
            if c in df.columns
        ]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            height=480,
            column_config={
                "robustness": st.column_config.TextColumn("Robustness", width="medium"),
                "risk_summary": st.column_config.TextColumn(
                    "Risk Summary", width="large"
                ),
                "confidence": st.column_config.TextColumn("Conf.", width="small"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "source_chunk_id": st.column_config.TextColumn(
                    "Chunk ID", width="medium"
                ),
                "key_evidence": st.column_config.TextColumn(
                    "Key Evidence", width="large"
                ),
                "mode_A_Aggressive": st.column_config.CheckboxColumn(
                    "A", width="small"
                ),
                "mode_B_Balanced": st.column_config.CheckboxColumn("B", width="small"),
                "mode_C_Conservative": st.column_config.CheckboxColumn(
                    "C", width="small"
                ),
            },
        )

    def render_mode_tab(mode_letter, mode_name):
        risks = mode_results.get(mode_letter, [])
        if not risks:
            st.markdown(
                f"""
            <div class="empty-state">
                <div class="empty-state-title">Mode {mode_letter} — No Risks Identified</div>
                <div class="empty-state-sub">
                    The retrieved chunks did not meet the {mode_name.lower()} threshold criteria.<br>
                    The LLM correctly refused to classify insufficient evidence.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            return

        for risk in risks:
            conf = risk.get("confidence", "MEDIUM").upper()
            conf_cls = (
                f"tag-{conf.lower()}"
                if conf in ("HIGH", "MEDIUM", "LOW")
                else "tag-medium"
            )
            cat = risk.get("category", "")
            st.markdown(
                f"""
            <div class="risk-card">
                <div class="risk-card-title">{risk.get("risk_summary", "N/A")}</div>
                <div class="risk-meta">
                    <span class="tag {conf_cls}">{conf} confidence</span>
                    <span class="tag tag-category">{cat}</span>
                    <span class="tag tag-chunk">{risk.get("source_chunk_id", "N/A")}</span>
                </div>
                <div class="evidence-block">"{risk.get("key_evidence", "N/A")}"</div>
                <div class="cf-block"><strong style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;font-family:'IBM Plex Mono',monospace;">Counterfactual Test</strong><br>{risk.get("counterfactual_test", "N/A")}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab_a:
        render_mode_tab("A", "Aggressive")
    with tab_b:
        render_mode_tab("B", "Balanced")
    with tab_c:
        render_mode_tab("C", "Conservative")

    with tab_chunks:
        st.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
            f"color:#8a8580;letter-spacing:0.12em;text-transform:uppercase;"
            f"margin-bottom:1rem;'>Showing 5 of {len(candidates)} retrieved chunks</p>",
            unsafe_allow_html=True,
        )
        for c in candidates[:5]:
            st.markdown(
                f"""
            <div class="chunk-preview">
                <div class="chunk-header">{c["chunk_id"]} &nbsp;·&nbsp; {c["category"].upper()} &nbsp;·&nbsp; similarity score {c["retrieval_score"]}</div>
                <div class="chunk-text">{c["text"][:420]}…</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ── EXPORT ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "Download Full CSV",
            data=buf.getvalue(),
            file_name=f"{ticker}_FY{year}_all_risks.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the complete results table including all modes, robustness labels, evidence, and counterfactual tests.",
        )
    with col2:
        robust_df = df[df["robustness"].str.startswith("🔴")]
        if not robust_df.empty:
            rb = io.StringIO()
            robust_df.to_csv(rb, index=False)
            st.download_button(
                "Download Robust Only",
                data=rb.getvalue(),
                file_name=f"{ticker}_FY{year}_robust_risks.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download only risks identified by all three modes — the highest-confidence, most defensible findings.",
            )

    # ── METHODOLOGY ──────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Methodology & Academic Notes"):
        st.markdown(f"""
**Document scope** — The entire {ticker} FY{year} 10-K was ingested including MD&A, Legal Proceedings, Financial Statements, and Notes to Financials — not limited to Item 1A.

**RAG pipeline** — The filing was split into ~1,000-character overlapping chunks using `RecursiveCharacterTextSplitter`. Each chunk was embedded with OpenAI's `text-embedding-3-small` and stored in ChromaDB with a deterministic `chunk_id`. Retrieval used 12 semantically diverse queries across financial, operational, and regulatory risk categories.

**Counterfactual validation** — `gpt-4o-mini` acts as a threshold validator, not a summarizer. For each retrieved chunk, it must: (1) check whether evidence meets the active mode's criteria, (2) cite the `source_chunk_id`, (3) provide a `counterfactual_test` explaining its reasoning, and (4) return `[]` if evidence is insufficient — preventing hallucination by design.

**Robustness classification** — Risks found in all three modes are labeled Robust (hardest to dispute). Risks found only in Aggressive mode are Fragile (dependent on loose interpretation). Conservative-only hits carry the highest per-risk confidence.
        """)

# ── FOOTER ───────────────────────────────────────────────────────
st.markdown(
    """
<div style="margin-top:4rem; padding-top:1.25rem; border-top:1px solid #d4cfc7;
     display:flex; justify-content:space-between; align-items:center;">
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#b0aba4; letter-spacing:0.15em;">
        RISKPROBE · COLUMBIA SIPA · SPRING 2026
    </span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#b0aba4; letter-spacing:0.1em;">
        LangChain · ChromaDB · OpenAI · SEC EDGAR
    </span>
</div>
""",
    unsafe_allow_html=True,
)
