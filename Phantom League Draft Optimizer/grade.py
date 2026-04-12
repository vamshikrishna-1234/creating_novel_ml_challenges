"""
grade.py — Phantom League Draft Optimizer

Metric: Recall@5 — fraction of test situations where the optimal
hero_id appears among the solver's top-5 recommended picks.

Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        pick_cols = ["pick_1", "pick_2", "pick_3", "pick_4", "pick_5"]

        if "situation_id" not in submission.columns:
            return 0.0
        if not all(c in submission.columns for c in pick_cols):
            return 0.0
        if "situation_id" not in answers.columns or "best_pick" not in answers.columns:
            return 0.0

        if submission["situation_id"].duplicated().any():
            return 0.0
        if answers["situation_id"].duplicated().any():
            return 0.0

        if len(submission) != len(answers):
            return 0.0

        merged = answers.merge(
            submission, on="situation_id", how="left"
        )

        if len(merged) != len(answers):
            return 0.0
        if merged[pick_cols].isna().any().any():
            return 0.0

        true_vals = merged["best_pick"].astype(int).values
        pred_matrix = merged[pick_cols].astype(int).values

        hits = 0
        for i in range(len(true_vals)):
            if true_vals[i] in pred_matrix[i]:
                hits += 1

        return float(hits / len(true_vals))

    except Exception:
        return 0.0
