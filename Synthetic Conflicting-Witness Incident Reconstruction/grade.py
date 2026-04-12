import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv)
        answers: Ground truth labels (loaded from private/answers.csv)

    Returns:
        A float score between 0.0 and 1.0. Higher is better.
        Average per-field exact-match accuracy across 6 structured fields.
    """
    try:
        FIELDS = ["actor", "action", "location", "time_period", "severity", "contributing_factor"]

        if "incident_id" not in submission.columns:
            return 0.0
        for f in FIELDS:
            if f not in submission.columns:
                return 0.0

        if submission["incident_id"].duplicated().any():
            return 0.0
        if answers["incident_id"].duplicated().any():
            return 0.0

        merged = answers.merge(
            submission, on="incident_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        if set(submission["incident_id"]) != set(answers["incident_id"]) or len(submission) != len(answers):
            return 0.0

        field_accuracies = []
        for f in FIELDS:
            col_true = f"{f}_true" if f"{f}_true" in merged.columns else f
            col_pred = f"{f}_pred" if f"{f}_pred" in merged.columns else f

            if merged[col_pred].isna().any():
                field_accuracies.append(0.0)
                continue

            true_vals = merged[col_true].astype(str).str.strip()
            pred_vals = merged[col_pred].astype(str).str.strip()
            acc = (true_vals == pred_vals).mean()
            field_accuracies.append(float(acc))

        avg_accuracy = float(np.mean(field_accuracies))
        return avg_accuracy

    except Exception:
        return 0.0
