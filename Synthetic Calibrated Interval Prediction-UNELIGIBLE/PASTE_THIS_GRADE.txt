import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv).
        answers: Ground truth labels (loaded from private/answers.csv).

    Returns:
        A float score in [0.0, 1.0]. Higher is better.
        Combined metric: 0.50 * point_accuracy + 0.30 * coverage_score + 0.20 * sharpness_score.
        Returns 0.0 for invalid or malformed submissions.
    """
    try:
        required_cols = {"sample_id", "point_estimate", "lower_90", "upper_90"}
        if not required_cols.issubset(submission.columns):
            return 0.0

        if "sample_id" not in answers.columns or "target" not in answers.columns:
            return 0.0

        if submission["sample_id"].duplicated().any():
            return 0.0
        if answers["sample_id"].duplicated().any():
            return 0.0

        if (
            set(submission["sample_id"]) != set(answers["sample_id"])
            or len(submission) != len(answers)
        ):
            return 0.0

        merged = answers.merge(
            submission, on="sample_id", how="left", suffixes=("", "_sub")
        )

        if len(merged) == 0:
            return 0.0

        for col in ["point_estimate", "lower_90", "upper_90"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            if merged[col].isna().any():
                return 0.0

        if not np.all(np.isfinite(merged["point_estimate"])):
            return 0.0
        if not np.all(np.isfinite(merged["lower_90"])):
            return 0.0
        if not np.all(np.isfinite(merged["upper_90"])):
            return 0.0

        if (merged["lower_90"] > merged["upper_90"]).any():
            return 0.0

        y_true = merged["target"].values.astype(float)
        y_pred = merged["point_estimate"].values.astype(float)
        lo = merged["lower_90"].values.astype(float)
        hi = merged["upper_90"].values.astype(float)

        baseline_mae = float(np.abs(y_true - y_true.mean()).mean())
        if baseline_mae < 1e-9:
            baseline_mae = 1.0
        mae = float(np.abs(y_true - y_pred).mean())
        point_score = max(0.0, 1.0 - mae / baseline_mae)

        in_interval = (y_true >= lo) & (y_true <= hi)
        coverage = float(in_interval.mean())
        coverage_score = max(0.0, 1.0 - 5.0 * abs(coverage - 0.90))

        widths = hi - lo
        mean_width = float(widths.mean())
        target_range = float(y_true.max() - y_true.min())
        if target_range < 1e-9:
            target_range = 1.0
        sharpness_score = max(0.0, 1.0 - mean_width / target_range)

        final = (
            0.50 * point_score
            + 0.30 * coverage_score
            + 0.20 * sharpness_score
        )

        if np.isnan(final):
            return 0.0

        return float(np.clip(final, 0.0, 1.0))

    except Exception:
        return 0.0
