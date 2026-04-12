"""
grade.py – Synthetic Corrupted Alignment Signal Recovery.

Metric: Macro F1 across 5 quality tiers (0-4).
Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if "sample_id" not in submission.columns or "quality_tier" not in submission.columns:
            return 0.0
        if "sample_id" not in answers.columns or "quality_tier" not in answers.columns:
            return 0.0

        if submission["sample_id"].duplicated().any():
            return 0.0
        if answers["sample_id"].duplicated().any():
            return 0.0

        merged = answers.merge(submission, on="sample_id", how="left",
                               suffixes=("_true", "_pred"))

        if len(merged) != len(answers):
            return 0.0
        if merged["quality_tier_pred"].isna().any():
            return 0.0

        y_true = merged["quality_tier_true"].astype(int).values
        y_pred = merged["quality_tier_pred"].astype(int).values

        classes = sorted(set(y_true))
        f1_scores = []
        for c in classes:
            tp = int(np.sum((y_pred == c) & (y_true == c)))
            fp = int(np.sum((y_pred == c) & (y_true != c)))
            fn = int(np.sum((y_pred != c) & (y_true == c)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)

        return float(np.mean(f1_scores))

    except Exception:
        return 0.0
