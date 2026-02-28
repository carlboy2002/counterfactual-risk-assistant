"""
comparator.py
-------------
Aggregates the counterfactual validation results from all three modes into
a single Pandas DataFrame that makes cross-mode comparison explicit.

Key insight: A risk that only appears in Mode A (Aggressive) but NOT in Mode C
(Conservative) is a "Fragile Risk" — it depends on loose interpretation.
A risk that appears in ALL three modes is a "Robust Risk" — it's hard to dispute.

This comparator makes that distinction explicit in the output.
"""

import pandas as pd
from typing import Any


def compute_robustness_label(modes_present: list[str]) -> str:
    """
    Classify a risk as Robust, Moderate, or Fragile based on how many
    threshold modes identified it.

    - Robust: Identified by all 3 modes (A + B + C) — high-confidence risk
    - Moderate: Identified by 2 modes — worth noting
    - Fragile: Identified by only 1 mode — speculative or borderline
    """
    count = len(modes_present)
    if count == 3:
        return "🔴 Robust (All Modes)"
    elif count == 2:
        return "🟡 Moderate (2 Modes)"
    else:
        mode = modes_present[0] if modes_present else "?"
        if mode == "A":
            return "⚪ Fragile (Aggressive Only)"
        elif mode == "B":
            return "🟡 Moderate (Balanced Only)"
        elif mode == "C":
            return "🔴 Strong (Conservative Hit)"
        return "⚪ Fragile (1 Mode)"


def build_comparison_dataframe(
    mode_results: dict[str, list[dict[str, Any]]],
    ticker: str,
    year: int,
) -> pd.DataFrame:
    """
    Build the final comparison DataFrame from mode results.

    Strategy: Use (source_chunk_id, category) as a grouping key to identify
    which risks were found by multiple modes. This surfaces Robust vs Fragile risks.

    Returns a DataFrame with columns:
        risk_summary | confidence | source_chunk_id | category |
        key_evidence | counterfactual_test | mode_A | mode_B | mode_C |
        robustness | ticker | year
    """
    MODE_LABELS = {"A": "Aggressive", "B": "Balanced", "C": "Conservative"}

    if not any(mode_results.values()):
        # All modes returned empty — build an informative empty DataFrame
        return pd.DataFrame(
            [{
                "risk_summary": "No risks extracted across any threshold mode.",
                "confidence": "N/A",
                "source_chunk_id": "N/A",
                "category": "N/A",
                "key_evidence": "N/A",
                "counterfactual_test": "All three modes returned empty arrays — the retrieved chunks did not meet any threshold.",
                "mode_A_Aggressive": False,
                "mode_B_Balanced": False,
                "mode_C_Conservative": False,
                "robustness": "⚫ No Evidence",
                "ticker": ticker.upper(),
                "year": year,
            }]
        )

    # Index: map chunk_id → risk data per mode
    # Since multiple modes may extract the same risk (same chunk), we group by chunk_id
    chunk_risk_map: dict[str, dict] = {}

    for mode_letter, risks in mode_results.items():
        mode_label = MODE_LABELS[mode_letter]
        for risk in risks:
            cid = risk.get("source_chunk_id", "unknown")
            if cid not in chunk_risk_map:
                chunk_risk_map[cid] = {
                    "risk_summary": risk.get("risk_summary", ""),
                    "confidence": risk.get("confidence", ""),
                    "source_chunk_id": cid,
                    "category": risk.get("category", ""),
                    "key_evidence": risk.get("key_evidence", ""),
                    "counterfactual_test": risk.get("counterfactual_test", ""),
                    "mode_A_Aggressive": False,
                    "mode_B_Balanced": False,
                    "mode_C_Conservative": False,
                    "modes_present": [],
                }
            # Mark which mode found this chunk
            chunk_risk_map[cid][f"mode_{mode_letter}_{mode_label}"] = True
            chunk_risk_map[cid]["modes_present"].append(mode_letter)

            # If a higher-confidence mode found this risk, update summary fields
            # Priority: C > B > A (Conservative findings are most precise)
            mode_priority = {"C": 3, "B": 2, "A": 1}
            existing_modes = chunk_risk_map[cid]["modes_present"]
            if mode_priority.get(mode_letter, 0) >= max(
                mode_priority.get(m, 0) for m in existing_modes
            ):
                chunk_risk_map[cid]["risk_summary"] = risk.get(
                    "risk_summary", chunk_risk_map[cid]["risk_summary"]
                )
                chunk_risk_map[cid]["confidence"] = risk.get(
                    "confidence", chunk_risk_map[cid]["confidence"]
                )
                chunk_risk_map[cid]["counterfactual_test"] = risk.get(
                    "counterfactual_test", chunk_risk_map[cid]["counterfactual_test"]
                )

    # Build rows
    rows = []
    for cid, data in chunk_risk_map.items():
        modes_present = list(set(data.pop("modes_present", [])))
        robustness = compute_robustness_label(modes_present)
        row = {
            **data,
            "robustness": robustness,
            "ticker": ticker.upper(),
            "year": year,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: most robust first, then by category, then by confidence
    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    robustness_order = {
        "🔴 Robust (All Modes)": 0,
        "🟡 Moderate (2 Modes)": 1,
        "🔴 Strong (Conservative Hit)": 2,
        "🟡 Moderate (Balanced Only)": 3,
        "⚪ Fragile (Aggressive Only)": 4,
        "⚪ Fragile (1 Mode)": 5,
        "⚫ No Evidence": 6,
    }

    df["_robustness_sort"] = df["robustness"].map(lambda x: robustness_order.get(x, 9))
    df["_confidence_sort"] = df["confidence"].map(lambda x: confidence_order.get(x, 9))
    df = df.sort_values(["_robustness_sort", "category", "_confidence_sort"])
    df = df.drop(columns=["_robustness_sort", "_confidence_sort"])
    df = df.reset_index(drop=True)

    return df


def compute_mode_summary(mode_results: dict[str, list[dict]]) -> dict[str, int]:
    """Return count of risks per mode for display in the UI."""
    return {
        "Mode A (Aggressive)": len(mode_results.get("A", [])),
        "Mode B (Balanced)": len(mode_results.get("B", [])),
        "Mode C (Conservative)": len(mode_results.get("C", [])),
    }
