"""
Grading script: Macro F1 over 4 verdict classes for
Synthetic Temporal Evidence Verification.

Score range: [0.0, 1.0] where HIGHER is better.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv).
        answers: Ground truth labels (loaded from private/answers.csv).

    Returns:
        A float score in [0.0, 1.0]. Higher is better (Macro F1).
        Returns 0.0 for invalid or malformed submissions (no exception raised).
    """
    try:
        if "claim_id" not in submission.columns or "verdict" not in submission.columns:
            return 0.0

        if submission["claim_id"].duplicated().any():
            return 0.0
        if answers["claim_id"].duplicated().any():
            return 0.0

        if set(submission["claim_id"]) != set(answers["claim_id"]) or len(submission) != len(answers):
            return 0.0

        merged = answers.merge(
            submission, on="claim_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        col_true = "verdict_true" if "verdict_true" in merged.columns else "verdict"
        col_pred = "verdict_pred" if "verdict_pred" in merged.columns else "verdict"

        if merged[col_pred].isna().any():
            return 0.0

        y_true = merged[col_true].astype(str).str.strip()
        y_pred = merged[col_pred].astype(str).str.strip()

        all_classes = sorted(set(y_true.tolist()))

        f1_scores = []
        for cls in all_classes:
            tp = ((y_pred == cls) & (y_true == cls)).sum()
            fp = ((y_pred == cls) & (y_true != cls)).sum()
            fn = ((y_pred != cls) & (y_true == cls)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        if np.isnan(macro_f1):
            return 0.0
        return macro_f1

    except Exception:
        return 0.0
