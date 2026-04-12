"""
Grading script: scores submission using weighted macro F1.

Custom metric: macro F1 across the 4 outcome classes, weighted by class
difficulty (inverse frequency in test set). This rewards models that handle
the minority class (timeout) well, not just the majority.

Prev_reviews compliance:
- try/except wrapping
- merge(..., how='left')
- set(submission.id) == set(answers.id)
- Unique id enforcement (submission + answers)
- Explicit length check
- Label value validation (no invented labels)
"""

import pandas as pd
import numpy as np


VALID_LABELS = {"deal_accepted", "deal_rejected", "counter_proposed", "timeout"}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers using frequency-weighted macro F1.

    Args:
        submission: The agent's predictions (loaded from submission.csv). Columns: id, label.
        answers: Ground truth labels (loaded from private/answers.csv). Columns: id, label.

    Returns:
        A float score in [0, 1]. Direction: maximize. Higher = better.

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

        # Validate label values
        sub_labels = set(submission["label"].dropna().unique())
        invalid = sub_labels - VALID_LABELS
        if invalid:
            raise ValueError(f"Invalid label values: {invalid}. Must be one of {VALID_LABELS}")

        if submission["label"].isna().any():
            raise ValueError("Submission has missing (NaN) label values")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["label_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        y_true = merged["label_true"].values
        y_pred = merged["label_pred"].values

        # Compute frequency-weighted macro F1
        # Weight per class = 1 / class_frequency (inverse frequency)
        classes = sorted(VALID_LABELS)
        total = len(y_true)
        class_f1s = []
        class_weights = []

        for cls in classes:
            true_pos = np.sum((y_true == cls) & (y_pred == cls))
            false_pos = np.sum((y_true != cls) & (y_pred == cls))
            false_neg = np.sum((y_true == cls) & (y_pred != cls))

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_count = np.sum(y_true == cls)
            weight = (total / class_count) if class_count > 0 else 0.0

            class_f1s.append(f1)
            class_weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(class_weights)
        if total_weight == 0:
            return float(0.0)
        class_weights = [w / total_weight for w in class_weights]

        score = sum(f * w for f, w in zip(class_f1s, class_weights))
        return float(score)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
