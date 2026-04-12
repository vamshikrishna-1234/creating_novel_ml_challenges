"""
grade.py — Synthetic Contextual Sequence Transduction

Metric: Per-character positional accuracy across all samples.
For each sample, characters are compared position-by-position.
Missing positions (shorter prediction) count as incorrect.
Extra positions (longer prediction) count as incorrect.

Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if "sample_id" not in submission.columns or "target" not in submission.columns:
            return 0.0
        if "sample_id" not in answers.columns or "target" not in answers.columns:
            return 0.0

        if submission["sample_id"].duplicated().any():
            return 0.0
        if answers["sample_id"].duplicated().any():
            return 0.0

        merged = answers.merge(
            submission, on="sample_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) != len(answers):
            return 0.0
        if merged["target_pred"].isna().any():
            return 0.0

        true_vals = merged["target_true"].astype(str).values
        pred_vals = merged["target_pred"].astype(str).values

        total_correct = 0
        total_positions = 0

        for t, p in zip(true_vals, pred_vals):
            max_len = max(len(t), len(p))
            if max_len == 0:
                continue
            correct = sum(
                1 for i in range(min(len(t), len(p))) if t[i] == p[i]
            )
            total_correct += correct
            total_positions += max_len

        if total_positions == 0:
            return 0.0

        return float(total_correct / total_positions)

    except Exception:
        return 0.0
