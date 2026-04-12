"""
grade.py – Synthetic Visual Feature Imputation Under Hidden Relational Constraints.

Metric: average per-property accuracy across 5 visual attributes.
Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        prop_cols = ["prop_0", "prop_1", "prop_2", "prop_3", "prop_4"]
        required = ["image_id"] + prop_cols
        if not all(c in submission.columns for c in required):
            return 0.0
        if not all(c in answers.columns for c in required):
            return 0.0

        if submission["image_id"].duplicated().any():
            return 0.0

        merged = answers.merge(submission, on="image_id", how="left", suffixes=("_true", "_pred"))

        if len(merged) != len(answers):
            return 0.0
        if merged[[c + "_pred" for c in prop_cols]].isna().any().any():
            return 0.0

        scores = []
        for p in prop_cols:
            acc = float((merged[f"{p}_true"] == merged[f"{p}_pred"]).mean())
            scores.append(acc)

        return float(sum(scores) / len(scores))

    except Exception:
        return 0.0
