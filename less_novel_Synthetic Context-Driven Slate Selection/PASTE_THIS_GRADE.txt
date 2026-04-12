import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv).
        answers: Ground truth labels (loaded from private/answers.csv).

    Returns:
        A float score in [0.0, 1.0]. Higher is better (accuracy).
        Returns 0.0 for invalid or malformed submissions.
    """
    try:
        if "query_id" not in submission.columns or "chosen_item_id" not in submission.columns:
            return 0.0

        if submission["query_id"].duplicated().any():
            return 0.0
        if answers["query_id"].duplicated().any():
            return 0.0

        if (
            set(submission["query_id"]) != set(answers["query_id"])
            or len(submission) != len(answers)
        ):
            return 0.0

        merged = answers.merge(
            submission, on="query_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        col_true = "chosen_item_id_true" if "chosen_item_id_true" in merged.columns else "chosen_item_id"
        col_pred = "chosen_item_id_pred" if "chosen_item_id_pred" in merged.columns else "chosen_item_id"

        if merged[col_pred].isna().any():
            return 0.0

        accuracy = float((merged[col_true] == merged[col_pred]).mean())
        if np.isnan(accuracy):
            return 0.0

        return accuracy

    except Exception:
        return 0.0
