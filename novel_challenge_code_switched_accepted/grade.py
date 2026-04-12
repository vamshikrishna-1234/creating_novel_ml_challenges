"""
Grading script: scores submission against private answers using RMSE.
- Wrapped in try/except so malformed submissions don't crash the pipeline.
- Uses merge(..., how='left') and validates submission ids match answers.
- Enforces unique ids and length check (prev_reviews).
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score regression predictions (resolution hours) using Root Mean Squared Error.
    Lower is better. Theoretical range: 0 (perfect) to unbounded; practical max
    is set in challenge config (e.g. 168 or 200) for display.

    Args:
        submission: Agent's CSV with columns id, prediction
        answers: Ground truth with columns id, target (from private/answers.csv)

    Returns:
        RMSE (float). Minimize.

    Raises:
        ValueError: Invalid format, missing ids, duplicates, or invalid predictions.
    """
    try:
        # Require expected columns (prev_reviews: probability columns not required = exploit; we require 'prediction')
        if "id" not in submission.columns or "prediction" not in submission.columns:
            raise ValueError("Submission must have columns: id, prediction")

        # Unique ids in submission (prev_reviews: duplicates distort RMSE)
        if submission["id"].duplicated().any():
            raise ValueError("Submission must not contain duplicate id values")

        # Unique ids in answers (many-to-one merge risk)
        if answers["id"].duplicated().any():
            raise ValueError("Answers file must not contain duplicate id values")

        # Length check: submission must have exactly one row per test sample
        if len(submission) != len(answers):
            raise ValueError(
                f"Submission must have exactly {len(answers)} rows (one per test id), got {len(submission)}"
            )

        # Must have same set of ids (prev_reviews: set(submission.id) == set(answers.id))
        sub_ids = set(submission["id"])
        ans_ids = set(answers["id"])
        if sub_ids != ans_ids:
            missing = ans_ids - sub_ids
            extra = sub_ids - ans_ids
            if missing:
                raise ValueError(f"Submission missing ids: {len(missing)} ids (e.g. {list(missing)[:5]})")
            if extra:
                raise ValueError(f"Submission has extra ids not in test set: {len(extra)} ids")

        # Merge with left join on answers so we align by id (prev_reviews: use left not inner)
        merged = answers.merge(submission, on="id", how="left")

        if merged["prediction"].isna().any():
            raise ValueError("Submission has missing (NaN) predictions for some rows")

        # Predictions must be numeric
        pred = pd.to_numeric(merged["prediction"], errors="coerce")
        if pred.isna().any():
            raise ValueError("All predictions must be numeric")

        # Optional: constrain prediction range so extreme values don't distort (prev_reviews)
        # Allow 0 to 200 (target is 0.5–168). Values outside get clipped for scoring only.
        pred = pred.clip(lower=0.0, upper=200.0)

        rmse = np.sqrt(((merged["target"] - pred) ** 2).mean())
        return float(rmse)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
