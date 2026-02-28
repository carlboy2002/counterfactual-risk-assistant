# Counterfactual Risk Assistant
### SEC 10-K Risk Analysis via RAG + Multi-Threshold LLM Validation

---

## Overview

This application ingests an **entire** SEC 10-K filing (not just Item 1A), uses RAG to retrieve candidate risk evidence, and applies an LLM acting as a **Counterfactual Validator** to classify risks under three distinct evidence thresholds. It explicitly refuses to hallucinate risks when the retrieved evidence is insufficient.

### Key Design Principles

| Principle | Implementation |
|---|---|
| **Non-triviality** | LLM is a threshold validator, not a summarizer. Must pass a counterfactual test and cite source chunk IDs. |
| **Full document scope** | Entire 10-K ingested, not limited to Item 1A |
| **RAG separation** | RAG only retrieves candidate chunks. LLM independently evaluates them. |
| **Failure-aware** | LLM outputs `[]` (refusal) when evidence doesn't meet threshold |
| **Structured output** | JSON with `risk_summary`, `confidence`, `source_chunk_id`, `counterfactual_test` |

---

## Architecture

```
Ticker + Year → SEC EDGAR → Full 10-K Text
                                  ↓
               Chunking (RecursiveCharacterTextSplitter)
                                  ↓
               Embeddings (text-embedding-3-small) → ChromaDB
                                  ↓
               Multi-Query RAG Retrieval (12 queries × 3 categories)
                                  ↓
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
    Mode A (Aggressive)    Mode B (Balanced)    Mode C (Conservative)
    [gpt-4o-mini]          [gpt-4o-mini]        [gpt-4o-mini]
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  ↓
              Comparator: Robust / Moderate / Fragile labeling
                                  ↓
              Streamlit DataFrame + CSV Export
```

---

## File Structure

```
counterfactual_risk_assistant/
├── app.py                    # Main Streamlit application
├── edgar_utils.py            # SEC EDGAR data acquisition & HTML cleaning
├── rag_engine.py             # Chunking, embedding, ChromaDB, retrieval
├── counterfactual_engine.py  # LLM validation with 3 mode system prompts
├── comparator.py             # Aggregation & Robust/Fragile classification
└── requirements.txt
```

---

## Quickstart for Team Members

### 1. Clone the Repo

```bash
git clone https://github.com/carlboy2002/counterfactual-risk-assistant.git
cd counterfactual-risk-assistant
```

### 2. Create a Virtual Environment

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> ⚠️ Requires **Python 3.10 or 3.11**. If your default Python is older, download 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/) first.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your SEC User-Agent

Open `edgar_utils.py` and update **line 17** with your real name and email:

```python
USER_AGENT = "Columbia University yourname@columbia.edu"
```

This is required by SEC EDGAR's access policy. Use your real institutional email.

### 5. Get an OpenAI API Key

You need an OpenAI account with access to:
- `gpt-4o-mini` (LLM validation)
- `text-embedding-3-small` (embeddings)

Get your key at [platform.openai.com](https://platform.openai.com) → API Keys.

> ⚠️ Never commit your API key to the repo. Enter it only in the Streamlit UI sidebar.

### 6. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter your OpenAI API key in the sidebar, then input a ticker and year.

---

## Usage

1. Enter your **OpenAI API Key** in the sidebar
2. Enter a **Stock Ticker** (e.g., `AAPL`, `MSFT`, `TSLA`)
3. Select a **Fiscal Year** (e.g., `2023`)
4. Click **🚀 Run Analysis**
5. Wait ~2–5 minutes for the pipeline to complete
6. Review the comparison table and download CSV

---

## The Three Threshold Modes

### Mode A — Aggressive
Classifies a chunk as a risk if it contains **any** speculative or forward-looking language ("may", "could", "no assurance"), competitive pressure mentions, or any hedging around future performance. Highest recall.

### Mode B — Balanced
Requires **explicit risk language** ("risk", "uncertainty", "exposure", "adverse effect") with some operational context. Standard financial analyst approach.

### Mode C — Conservative
Only classifies **quantified, imminent, or material risks**: specific dollar amounts, active legal proceedings, current operational disruptions described as material. The LLM must refuse if evidence doesn't meet this bar.

---

## Output Schema

Each identified risk has:

| Field | Description |
|---|---|
| `risk_summary` | 1-2 sentence risk description |
| `confidence` | HIGH / MEDIUM / LOW |
| `source_chunk_id` | Traceable to specific document chunk |
| `category` | financial / operational / regulatory |
| `key_evidence` | Direct quote (<30 words) from the chunk |
| `counterfactual_test` | LLM's reasoning for why this meets the threshold |
| `mode_A/B/C` | Boolean — which modes identified this risk |
| `robustness` | 🔴 Robust / 🟡 Moderate / ⚪ Fragile |

---

## Academic Notes

**Why this is non-trivial:** The counterfactual validation layer requires the LLM to reason about evidence thresholds, not just extract text. The `counterfactual_test` field forces the model to articulate WHY a chunk meets or fails a threshold — making its reasoning auditable. The refusal mechanism (`[]` output) is a form of constrained generation that prevents hallucination.

**Scope:** The full 10-K is ingested including MD&A, Legal Proceedings, Financial Statements, Notes to Financials, and other sections beyond Item 1A.

**RAG Design:** 12 semantically diverse queries across 3 risk categories ensure high recall across the full document. Deduplication prevents the same chunk from being counted multiple times.

---

*Data source: SEC EDGAR (public domain). This tool is for research and educational purposes only.*
