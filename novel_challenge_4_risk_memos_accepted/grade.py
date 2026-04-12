"""
Grading script: scores submission using quadratic-weighted Cohen's Kappa.

Custom metric for ordinal 5-class classification. Penalizes distant
misclassifications more than close ones (tier_1 predicted as tier_5 is worse
than tier_3 predicted as tier_4).

Prev_reviews compliance:
- try/except wrapping
- merge(..., how='left')
- set(submission.id) == set(answers.id)
- Unique id enforcement (submission + answers)
- Explicit length check
- Label value validation
"""

import pandas as pd
import numpy as np


VALID_LABELS = {
    "tier_1_minor", "tier_2_moderate", "tier_3_significant",
    "tier_4_severe", "tier_5_critical",
}

LABEL_ORDER = [
    "tier_1_minor", "tier_2_moderate", "tier_3_significant",
    "tier_4_severe", "tier_5_critical",
]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth using quadratic-weighted Cohen's Kappa.

    Args:
        submission: The agent's predictions (loaded from submission.csv). Columns: id, label.
        answers: Ground truth labels (loaded from private/answers.csv). Columns: id, label.

    Returns:
        A float score in [-1, 1]. Direction: maximize. Higher = better ordinal agreement.

    Raises:
        ValueError: If the submission format is invalid.
        RuntimeError: If an unexpected error occurs.
    """
    try:
        if "id" not in submission.columns or "label" not in submission.columns:
            raise ValueError("Submission must have columns: id, label")

        if submission["id"].duplicated().any():
            raise ValueError("Submission must not contain duplicate id values")

        if answers["id"].duplicated().any():
            raise ValueError("Answers file must not contain duplicate id values")

        if len(submission) != len(answers):
            raise ValueError(
                f"Submission must have exactly {len(answers)} rows, got {len(submission)}"
            )

        sub_ids = set(submission["id"])
        ans_ids = set(answers["id"])
        if sub_ids != ans_ids:
            missing = ans_ids - sub_ids
            extra = sub_ids - ans_ids
            if missing:
                raise ValueError(f"Submission missing ids: {len(missing)} (e.g. {list(missing)[:5]})")
            if extra:
                raise ValueError(f"Submission has extra ids: {len(extra)}")

        if submission["label"].isna().any():
            raise ValueError("Submission has missing (NaN) label values")

        sub_labels = set(submission["label"].dropna().unique())
        invalid = sub_labels - VALID_LABELS
        if invalid:
            raise ValueError(f"Invalid label values: {invalid}. Must be one of {VALID_LABELS}")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["label_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        y_true = merged["label_true"].values
        y_pred = merged["label_pred"].values

        # Convert to ordinal indices for Kappa
        label_to_idx = {lab: i for i, lab in enumerate(LABEL_ORDER)}
        y_true_idx = np.array([label_to_idx[l] for l in y_true])
        y_pred_idx = np.array([label_to_idx[l] for l in y_pred])

        # Handle constant predictions (all same class): kappa is 0
        if np.ptp(y_pred_idx) == 0:
            return float(0.0)

        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(y_true_idx, y_pred_idx, weights="quadratic")

        if np.isnan(kappa):
            return float(0.0)

        return float(kappa)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
