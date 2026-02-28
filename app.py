"""
app.py
------
Counterfactual Risk Assistant — Main Streamlit Application

Architecture:
  1. Data Acquisition  → edgar_utils.py
  2. Chunking + RAG    → rag_engine.py
  3. LLM Validation    → counterfactual_engine.py
  4. Comparison Output → comparator.py

Run with: streamlit run app.py
"""

import io
import logging
import os
import time
import traceback

import pandas as pd
import streamlit as st

# Local modules
from edgar_utils import fetch_and_clean_10k
from rag_engine import build_vector_store, retrieve_risk_candidates, format_chunks_for_prompt
from counterfactual_engine import run_counterfactual_validation
from comparator import build_comparison_dataframe, compute_mode_summary

# ─────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Counterfactual Risk Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .mode-card {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .metric-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-step {
        padding: 0.3rem 0;
        font-size: 0.95rem;
    }
    .highlight-robust {
        color: #c0392b;
        font-weight: bold;
    }
    .highlight-fragile {
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Key input (masked)
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Required for embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)",
    )

    st.divider()
    st.markdown("## 📋 Filing Parameters")

    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        max_chars=10,
        help="NYSE/NASDAQ ticker symbol (e.g., AAPL, MSFT, TSLA)",
    ).strip().upper()

    current_year = 2024
    year = st.number_input(
        "Fiscal Year",
        min_value=2015,
        max_value=current_year,
        value=2023,
        step=1,
        help="The fiscal year of the 10-K filing (e.g., 2023 for the FY2023 10-K)",
    )

    st.divider()
    st.markdown("## 🔧 RAG Parameters")

    top_k = st.slider(
        "Top-K chunks per query",
        min_value=2,
        max_value=8,
        value=3,
        help="Number of chunks retrieved per semantic query. Higher = more evidence but higher cost.",
    )

    max_chunks_to_llm = st.slider(
        "Max chunks sent to LLM",
        min_value=10,
        max_value=40,
        value=20,
        help="Caps total chunks passed to each mode's LLM call to manage token usage.",
    )

    st.divider()
    st.markdown("## 📖 Mode Reference")
    st.markdown("""
**Mode A — Aggressive**
> Flags speculative, minor, and forward-looking risks. Highest recall, lowest precision.

**Mode B — Balanced**
> Standard extraction based on explicit risk language. Balanced recall/precision.

**Mode C — Conservative**
> Only quantified, imminent, or material risks. Lowest recall, highest precision.

---
**Robustness Interpretation:**
- 🔴 Robust = All 3 modes agree
- 🟡 Moderate = 2 modes agree  
- ⚪ Fragile = Only Mode A found it
    """)

# ─────────────────────────────────────────────────────────────────
# MAIN PANEL — HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Counterfactual Risk Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">SEC 10-K Risk Analysis via RAG + Multi-Threshold Counterfactual Validation (gpt-4o-mini)</div>',
    unsafe_allow_html=True,
)

# Architecture diagram (collapsible)
with st.expander("🏗️ Pipeline Architecture", expanded=False):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                     COUNTERFACTUAL RISK ASSISTANT                           │
    │                                                                             │
    │  INPUT: Ticker + Year                                                       │
    │     │                                                                       │
    │     ▼                                                                       │
    │  ┌──────────────────┐                                                       │
    │  │  1. DATA ACQ.    │  SEC EDGAR API → Full 10-K HTML → Cleaned Text        │
    │  │  edgar_utils.py  │  (NOT limited to Item 1A — ENTIRE DOCUMENT)           │
    │  └────────┬─────────┘                                                       │
    │           ▼                                                                 │
    │  ┌──────────────────┐                                                       │
    │  │  2. RAG INDEXING │  Chunk → Embed (text-embedding-3-small) → ChromaDB   │
    │  │  rag_engine.py   │  Each chunk tagged with chunk_id for citation         │
    │  └────────┬─────────┘                                                       │
    │           ▼                                                                 │
    │  ┌──────────────────┐                                                       │
    │  │  3. RAG RETRIEVAL│  12 semantic queries across 3 risk categories         │
    │  │  rag_engine.py   │  → Deduplicated candidate evidence chunks             │
    │  └────────┬─────────┘                                                       │
    │           ▼                                                                 │
    │  ┌────────────────────────────────────────────────────────────┐             │
    │  │  4. COUNTERFACTUAL VALIDATION (counterfactual_engine.py)  │             │
    │  │                                                            │             │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │             │
    │  │  │ Mode A       │  │ Mode B       │  │ Mode C       │    │             │
    │  │  │ AGGRESSIVE   │  │ BALANCED     │  │ CONSERVATIVE │    │             │
    │  │  │ (Speculative)│  │ (Explicit)   │  │ (Quantified) │    │             │
    │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │             │
    │  │         │                 │                  │            │             │
    │  │  LLM evaluates EACH chunk against mode threshold          │             │
    │  │  LLM REFUSES to classify if evidence is insufficient      │             │
    │  └─────────────────────────┬──────────────────────────────────             │
    │                            ▼                                               │
    │  ┌──────────────────┐                                                       │
    │  │  5. COMPARATOR   │  Aggregate → Robust vs Fragile risk labeling          │
    │  │  comparator.py   │  → Pandas DataFrame → Streamlit UI + CSV Export      │
    │  └──────────────────┘                                                       │
    └─────────────────────────────────────────────────────────────────────────────┘
    ```
    """)

st.divider()

# ─────────────────────────────────────────────────────────────────
# ANALYSIS TRIGGER
# ─────────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_analysis = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=(not api_key or not ticker),
    )

with col_info:
    if not api_key:
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to begin.")
    elif not ticker:
        st.warning("⚠️ Enter a stock ticker in the sidebar.")
    else:
        st.info(f"Ready to analyze **{ticker}** 10-K for FY**{year}**. Estimated time: 2–5 minutes.")

# ─────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────
if run_analysis:
    if not api_key:
        st.error("OpenAI API key is required.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key  # Set for LangChain components

    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown(f"### 🔄 Analyzing {ticker} FY{year} 10-K")
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        def update_status(msg: str, progress: int):
            status_placeholder.markdown(f'<div class="status-step">⏳ {msg}</div>', unsafe_allow_html=True)
            progress_bar.progress(progress)

        # ── STEP 1: DATA ACQUISITION ──────────────────────────────
        update_status("Step 1/5: Fetching 10-K from SEC EDGAR...", 5)
        try:
            with st.spinner("Downloading filing from SEC EDGAR..."):
                clean_text, filing_url = fetch_and_clean_10k(ticker, int(year))
            st.success(f"✅ **Step 1 Complete** — Fetched {len(clean_text):,} characters from [SEC EDGAR]({filing_url})")
        except Exception as e:
            st.error(f"❌ **Data Acquisition Failed**: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()

        # ── STEP 2: CHUNKING + VECTOR STORE ───────────────────────
        update_status("Step 2/5: Chunking and embedding the full 10-K...", 20)
        try:
            with st.spinner("Building ChromaDB vector store (text-embedding-3-small)..."):
                vector_store = build_vector_store(
                    clean_text=clean_text,
                    ticker=ticker,
                    year=int(year),
                    openai_api_key=api_key,
                )
            # Estimate chunk count from text length
            estimated_chunks = len(clean_text) // 900
            st.success(f"✅ **Step 2 Complete** — ~{estimated_chunks} chunks indexed in ChromaDB")
        except Exception as e:
            st.error(f"❌ **Indexing Failed**: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()

        # ── STEP 3: RAG RETRIEVAL ─────────────────────────────────
        update_status("Step 3/5: Running multi-query RAG retrieval across risk categories...", 40)
        try:
            with st.spinner("Retrieving candidate evidence chunks from vector DB..."):
                candidates = retrieve_risk_candidates(vector_store, top_k=top_k)
                chunks_text = format_chunks_for_prompt(candidates, max_chunks=max_chunks_to_llm)
                num_chunks = min(len(candidates), max_chunks_to_llm)

            st.success(
                f"✅ **Step 3 Complete** — Retrieved {len(candidates)} unique chunks "
                f"({num_chunks} sent to LLM) across financial, operational, and regulatory queries"
            )

            # Show retrieved chunks sample
            with st.expander(f"📄 Sample Retrieved Evidence Chunks (showing 3 of {num_chunks})"):
                sample_candidates = candidates[:3]
                for c in sample_candidates:
                    st.markdown(f"**{c['chunk_id']}** | Category: `{c['category']}` | Score: `{c['retrieval_score']}`")
                    st.text(c["text"][:300] + "...")
                    st.divider()

        except Exception as e:
            st.error(f"❌ **RAG Retrieval Failed**: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()

        # ── STEP 4: COUNTERFACTUAL VALIDATION ─────────────────────
        update_status("Step 4/5: Running Counterfactual Validation across 3 modes (gpt-4o-mini)...", 60)
        try:
            with st.spinner("Running LLM counterfactual validation — Modes A, B, C..."):
                mode_results = run_counterfactual_validation(
                    chunks_text=chunks_text,
                    num_chunks=num_chunks,
                    ticker=ticker,
                    year=int(year),
                    openai_api_key=api_key,
                )

            summary = compute_mode_summary(mode_results)
            st.success(
                f"✅ **Step 4 Complete** — "
                f"Mode A: {summary['Mode A (Aggressive)']} risks | "
                f"Mode B: {summary['Mode B (Balanced)']} risks | "
                f"Mode C: {summary['Mode C (Conservative)']} risks"
            )

        except Exception as e:
            st.error(f"❌ **Counterfactual Validation Failed**: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()

        # ── STEP 5: COMPARATOR & OUTPUT ───────────────────────────
        update_status("Step 5/5: Building comparison DataFrame...", 85)
        try:
            df = build_comparison_dataframe(mode_results, ticker, int(year))
            progress_bar.progress(100)
            status_placeholder.markdown(
                '<div class="status-step">✅ Analysis complete!</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ **Comparison Failed**: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # RESULTS DISPLAY
    # ─────────────────────────────────────────────────────────────
    st.markdown(f"## 📋 Results: {ticker} FY{year} Risk Analysis")

    # Summary metrics
    total_risks = len(df[df["risk_summary"] != "No risks extracted across any threshold mode."])
    robust_risks = len(df[df["robustness"].str.startswith("🔴 Robust")])
    fragile_risks = len(df[df["robustness"].str.startswith("⚪ Fragile")])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Unique Risks", total_risks)
    with col2:
        st.metric("🔴 Robust Risks", robust_risks, help="Found in ALL 3 modes")
    with col3:
        st.metric("🟡 Moderate Risks", len(df[df["robustness"].str.startswith("🟡")]))
    with col4:
        st.metric("⚪ Fragile Risks", fragile_risks, help="Found in Aggressive mode only")
    with col5:
        st.metric("Mode C (Conservative)", summary.get("Mode C (Conservative)", 0), help="Highest-confidence risks")

    st.divider()

    # Per-mode breakdown (tabs)
    tab_all, tab_a, tab_b, tab_c = st.tabs([
        "📊 Comparison (All Modes)",
        "🔶 Mode A: Aggressive",
        "🔷 Mode B: Balanced",
        "🔴 Mode C: Conservative",
    ])

    with tab_all:
        st.markdown("### Cross-Mode Risk Comparison")
        st.markdown(
            "Rows are sorted by **Robustness** (Robust → Fragile). "
            "Risks appearing in all 3 modes are the most defensible."
        )

        # Display-friendly column subset
        display_cols = [
            "robustness", "risk_summary", "confidence", "category",
            "source_chunk_id", "key_evidence",
            "mode_A_Aggressive", "mode_B_Balanced", "mode_C_Conservative",
        ]
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[display_cols],
            use_container_width=True,
            height=500,
            column_config={
                "robustness": st.column_config.TextColumn("Robustness", width="medium"),
                "risk_summary": st.column_config.TextColumn("Risk Summary", width="large"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "source_chunk_id": st.column_config.TextColumn("Source Chunk", width="medium"),
                "key_evidence": st.column_config.TextColumn("Key Evidence", width="large"),
                "mode_A_Aggressive": st.column_config.CheckboxColumn("Mode A", width="small"),
                "mode_B_Balanced": st.column_config.CheckboxColumn("Mode B", width="small"),
                "mode_C_Conservative": st.column_config.CheckboxColumn("Mode C", width="small"),
            },
        )

    def render_mode_tab(mode_letter: str, mode_name: str, color: str):
        risks = mode_results.get(mode_letter, [])
        if not risks:
            st.info(
                f"Mode {mode_letter} ({mode_name}) returned **no risks** for this filing. "
                "This indicates the retrieved chunks did not meet this threshold's criteria — "
                "the LLM correctly refused to hallucinate risks."
            )
            return

        st.markdown(f"### {mode_name} Mode — {len(risks)} Risk(s) Identified")

        for i, risk in enumerate(risks):
            with st.expander(
                f"Risk {i+1}: {risk.get('risk_summary', 'N/A')[:80]}... "
                f"[{risk.get('confidence', '?')}]"
            ):
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(f"**Source Chunk:** `{risk.get('source_chunk_id', 'N/A')}`")
                    st.markdown(f"**Category:** `{risk.get('category', 'N/A')}`")
                    st.markdown(f"**Confidence:** `{risk.get('confidence', 'N/A')}`")
                with col_r:
                    st.markdown(f"**Risk Summary:**")
                    st.info(risk.get("risk_summary", "N/A"))

                st.markdown("**Key Evidence (from chunk):**")
                st.warning(f'"{risk.get("key_evidence", "N/A")}"')

                st.markdown("**Counterfactual Test (why this meets the threshold):**")
                st.success(risk.get("counterfactual_test", "N/A"))

    with tab_a:
        render_mode_tab("A", "Aggressive", "#ff8c00")

    with tab_b:
        render_mode_tab("B", "Balanced", "#0066cc")

    with tab_c:
        render_mode_tab("C", "Conservative", "#cc0000")

    # ─────────────────────────────────────────────────────────────
    # CSV DOWNLOAD
    # ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💾 Export Results")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Full DataFrame CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Comparison (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"{ticker}_FY{year}_counterfactual_risks.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        # Robust risks only
        robust_df = df[df["robustness"].str.startswith("🔴")]
        if not robust_df.empty:
            robust_csv = io.StringIO()
            robust_df.to_csv(robust_csv, index=False)
            st.download_button(
                label="📥 Download Robust Risks Only (CSV)",
                data=robust_csv.getvalue(),
                file_name=f"{ticker}_FY{year}_robust_risks_only.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No robust risks (found in all 3 modes) to export separately.")

    # ─────────────────────────────────────────────────────────────
    # METHODOLOGY NOTE
    # ─────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📚 Methodology & Academic Notes"):
        st.markdown(f"""
        ### How This Analysis Was Produced

        **Document Scope:** The ENTIRE {ticker} FY{year} 10-K filing was ingested — not limited to Item 1A.
        This ensures risks disclosed in MD&A, financial statements, legal proceedings, and other
        sections are captured.

        **RAG Strategy:** The filing was chunked into ~1,000-character overlapping segments using
        `RecursiveCharacterTextSplitter`. Each chunk was embedded using OpenAI's
        `text-embedding-3-small` model and stored in ChromaDB. Retrieval used **12 semantically
        diverse queries** across financial, operational, and regulatory risk categories to maximize
        recall.

        **Counterfactual Validation:** Unlike a summarizer, the LLM (`gpt-4o-mini`) was instructed to
        act as a **threshold validator**. For each chunk, it must:
        1. Check whether the evidence meets the *active mode's* specific criteria
        2. Provide a `counterfactual_test` explaining WHY it meets (or fails) the threshold
        3. Output `[]` (refuse) if evidence is insufficient — preventing hallucination

        **Robustness Classification:**
        - **Robust (All Modes):** The risk is so clearly evidenced that even the Conservative mode
          (which requires quantification or materiality) identifies it.
        - **Moderate (2 Modes):** The risk is substantiated but not definitively quantified.
        - **Fragile (Aggressive Only):** The risk relies on speculative or hedging language;
          may not be actionable without further investigation.

        **Limitations:** LLM outputs are probabilistic and may contain errors. Always verify
        source chunk citations against the original filing. This tool is for research purposes only.
        """)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Counterfactual Risk Assistant | Built with Streamlit + LangChain + ChromaDB + OpenAI | "
    "Data source: SEC EDGAR (public domain)</small>",
    unsafe_allow_html=True,
)
