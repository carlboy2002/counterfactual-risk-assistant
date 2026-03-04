"""
comparator.py
-------------
Aggregates counterfactual validation results from all three modes into
a single Pandas DataFrame with Robust / Moderate / Fragile labeling.

Fix: The original design grouped only by exact chunk_id match. In practice,
the LLM may cite the same underlying risk using slightly different chunk_ids
across modes (e.g., capitalisation differences, trimmed strings). 

The improved strategy:
  1. First try exact chunk_id match (fast path).
  2. For unmatched risks, try fuzzy risk_summary similarity matching so that
     the same real-world risk identified by multiple modes gets merged into
     one row even if the cited chunk_id differs slightly.
"""

import re
import pandas as pd
from typing import Any


# ── SIMILARITY HELPERS ──────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings (0–1)."""
    sa = set(_normalise(a).split())
    sb = set(_normalise(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


MERGE_THRESHOLD = 0.45  # risks with >45% token overlap are considered the same risk


# ── ROBUSTNESS LABEL ────────────────────────────────────────────

def compute_robustness_label(modes_present: list[str]) -> str:
    unique = sorted(set(modes_present))
    count = len(unique)
    if count == 3:
        return "🔴 Robust (All Modes)"
    elif count == 2:
        if "A" in unique and "B" in unique:
            return "🟡 Moderate (A + B)"
        elif "B" in unique and "C" in unique:
            return "🟡 Moderate (B + C)"
        elif "A" in unique and "C" in unique:
            return "🟡 Moderate (A + C)"
        return "🟡 Moderate (2 Modes)"
    else:
        mode = unique[0] if unique else "?"
        if mode == "A":
            return "⚪ Fragile (Aggressive Only)"
        elif mode == "B":
            return "🟡 Moderate (Balanced Only)"
        elif mode == "C":
            return "🔴 Strong (Conservative Hit)"
        return "⚪ Fragile (1 Mode)"


# ── MAIN BUILD FUNCTION ─────────────────────────────────────────

def build_comparison_dataframe(
    mode_results: dict[str, list[dict[str, Any]]],
    ticker: str,
    year: int,
) -> pd.DataFrame:
    """
    Build the final comparison DataFrame.

    Merging strategy:
    - Primary key: normalised source_chunk_id (exact match after normalisation)
    - Fallback: fuzzy risk_summary overlap > MERGE_THRESHOLD
    This ensures that the same real-world risk identified by multiple modes
    gets consolidated into ONE row with all mode flags set correctly.
    """
    MODE_LABELS = {"A": "Aggressive", "B": "Balanced", "C": "Conservative"}
    MODE_PRIORITY = {"C": 3, "B": 2, "A": 1}

    if not any(mode_results.values()):
        return pd.DataFrame([{
            "risk_summary": "No risks extracted across any threshold mode.",
            "confidence": "N/A",
            "source_chunk_id": "N/A",
            "category": "N/A",
            "key_evidence": "N/A",
            "counterfactual_test": "All three modes returned empty arrays.",
            "mode_A_Aggressive": False,
            "mode_B_Balanced": False,
            "mode_C_Conservative": False,
            "robustness": "⚫ No Evidence",
            "ticker": ticker.upper(),
            "year": year,
        }])

    # Each entry: canonical risk record
    merged: list[dict] = []

    def _find_existing(risk: dict) -> int | None:
        """
        Return index of an existing merged entry that matches this risk,
        or None if no match found.
        Match criteria (in order):
          1. Exact normalised chunk_id
          2. Fuzzy summary overlap above threshold
        """
        cid_norm = _normalise(risk.get("source_chunk_id", ""))
        summary = risk.get("risk_summary", "")

        for i, entry in enumerate(merged):
            # 1. Exact chunk_id match (after normalisation)
            if cid_norm and cid_norm == _normalise(entry.get("source_chunk_id", "")):
                return i
            # 2. Fuzzy summary match
            if _token_overlap(summary, entry.get("risk_summary", "")) >= MERGE_THRESHOLD:
                return i
        return None

    for mode_letter, risks in mode_results.items():
        mode_label = MODE_LABELS[mode_letter]
        for risk in risks:
            idx = _find_existing(risk)

            if idx is None:
                # New unique risk — create entry
                entry = {
                    "risk_summary": risk.get("risk_summary", ""),
                    "confidence": risk.get("confidence", "MEDIUM"),
                    "source_chunk_id": risk.get("source_chunk_id", "unknown"),
                    "category": risk.get("category", ""),
                    "key_evidence": risk.get("key_evidence", ""),
                    "counterfactual_test": risk.get("counterfactual_test", ""),
                    "mode_A_Aggressive": False,
                    "mode_B_Balanced": False,
                    "mode_C_Conservative": False,
                    "modes_present": [mode_letter],
                }
                entry[f"mode_{mode_letter}_{mode_label}"] = True
                merged.append(entry)
            else:
                # Existing risk found — merge this mode's findings in
                merged[idx][f"mode_{mode_letter}_{mode_label}"] = True
                merged[idx]["modes_present"].append(mode_letter)

                # Higher-priority mode updates the summary fields
                existing_best = max(
                    MODE_PRIORITY.get(m, 0)
                    for m in merged[idx]["modes_present"]
                )
                if MODE_PRIORITY.get(mode_letter, 0) >= existing_best:
                    merged[idx]["risk_summary"] = risk.get(
                        "risk_summary", merged[idx]["risk_summary"]
                    )
                    merged[idx]["confidence"] = risk.get(
                        "confidence", merged[idx]["confidence"]
                    )
                    merged[idx]["counterfactual_test"] = risk.get(
                        "counterfactual_test", merged[idx]["counterfactual_test"]
                    )

    # Build DataFrame rows
    rows = []
    for entry in merged:
        modes_present = entry.pop("modes_present", [])
        robustness = compute_robustness_label(modes_present)
        rows.append({
            **entry,
            "robustness": robustness,
            "ticker": ticker.upper(),
            "year": year,
        })

    df = pd.DataFrame(rows)

    # Sort
    robustness_order = {
        "🔴 Robust (All Modes)": 0,
        "🔴 Strong (Conservative Hit)": 1,
        "🟡 Moderate (B + C)": 2,
        "🟡 Moderate (A + C)": 3,
        "🟡 Moderate (A + B)": 4,
        "🟡 Moderate (2 Modes)": 5,
        "🟡 Moderate (Balanced Only)": 6,
        "⚪ Fragile (Aggressive Only)": 7,
        "⚪ Fragile (1 Mode)": 8,
        "⚫ No Evidence": 9,
    }
    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    df["_rs"] = df["robustness"].map(lambda x: robustness_order.get(x, 9))
    df["_cs"] = df["confidence"].map(lambda x: confidence_order.get(x, 9))
    df = df.sort_values(["_rs", "category", "_cs"]).drop(columns=["_rs", "_cs"])
    df = df.reset_index(drop=True)

    return df


def compute_mode_summary(mode_results: dict[str, list[dict]]) -> dict[str, int]:
    return {
        "Mode A (Aggressive)": len(mode_results.get("A", [])),
        "Mode B (Balanced)": len(mode_results.get("B", [])),
        "Mode C (Conservative)": len(mode_results.get("C", [])),
    }


# ── REPORT GENERATOR ────────────────────────────────────────────

def generate_report_text(
    df: pd.DataFrame,
    mode_results: dict[str, list[dict]],
    ticker: str,
    year: int,
    filing_url: str = "",
) -> str:
    """
    Generate a structured plain-text report summarising the analysis.
    This is passed to the LLM in app.py for professional prose generation.
    """
    summary = compute_mode_summary(mode_results)

    robust_risks = df[df["robustness"].str.startswith("🔴 Robust")]
    conservative_hits = df[df["robustness"].str.startswith("🔴 Strong")]
    moderate_risks = df[df["robustness"].str.startswith("🟡")]
    fragile_risks = df[df["robustness"].str.startswith("⚪")]

    lines = []
    lines.append(f"TICKER: {ticker.upper()} | FISCAL YEAR: {year}")
    lines.append(f"SOURCE: {filing_url}")
    lines.append(f"Total unique risks identified: {len(df)}")
    lines.append(f"Mode A (Aggressive): {summary['Mode A (Aggressive)']} risks")
    lines.append(f"Mode B (Balanced): {summary['Mode B (Balanced)']} risks")
    lines.append(f"Mode C (Conservative): {summary['Mode C (Conservative)']} risks")
    lines.append("")

    def _add_section(label, subset):
        if subset.empty:
            return
        lines.append(f"── {label.upper()} ({len(subset)}) ──")
        for _, row in subset.iterrows():
            lines.append(f"• [{row.get('category','').upper()}] {row.get('risk_summary','')}")
            lines.append(f"  Evidence: \"{row.get('key_evidence','')}\"")
            lines.append(f"  Confidence: {row.get('confidence','')} | Chunk: {row.get('source_chunk_id','')}")
            lines.append("")

    _add_section("Robust Risks — All Three Modes Agree", robust_risks)
    _add_section("Strong Conservative Hits", conservative_hits)
    _add_section("Moderate Risks", moderate_risks)
    _add_section("Fragile Risks — Aggressive Mode Only", fragile_risks)

    return "\n".join(lines)
