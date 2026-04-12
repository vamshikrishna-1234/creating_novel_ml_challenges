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
    """
    ans_cols = answers[["session_id", "selected_candidate_id"]].copy()
    sub_cols = submission[["session_id", "selected_candidate_id"]].copy()
    merged = ans_cols.merge(
        sub_cols, on="session_id", suffixes=("_true", "_pred")
    )

    if len(merged) == 0:
        raise ValueError("No common session_ids between submission and answers")

    if merged["selected_candidate_id_pred"].isna().any():
        raise ValueError("Submission has missing predictions for some sessions")

    accuracy = (
        merged["selected_candidate_id_true"]
        == merged["selected_candidate_id_pred"]
    ).mean()

    return float(accuracy)
