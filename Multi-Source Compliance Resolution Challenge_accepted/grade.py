"""
Grading script: per-field exact match scoring for structured verdict strings.

Verdict format:
  FACILITY:X | VIOLATIONS:Y | COUNT:Z | SEVERITY:W | ACTION:A | PENALTY:P

Scoring: For each row, compare each of the 6 fields independently.
  Field match = 1, mismatch = 0.
  Row score = (matches) / 6.
  Final score = mean of all row scores.

Direction: Maximize. Min = 0.0, Max = 1.0.

Compliance with prev_reviews:
- try/except wrapping
- merge(..., how='left')
- set(submission.id) == set(answers.id)
- Unique id enforcement
- Explicit length check
"""

import pandas as pd
import numpy as np


VERDICT_FIELDS = ["FACILITY", "VIOLATIONS", "COUNT", "SEVERITY", "ACTION", "PENALTY"]


def _parse_verdict(verdict_str: str) -> dict:
    """Parse a verdict string into a dict of field -> value."""
    if not isinstance(verdict_str, str):
        return {}
    fields = {}
    for part in verdict_str.split(" | "):
        if ":" in part:
            key, val = part.split(":", 1)
            fields[key.strip()] = val.strip()
    return fields


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission using per-field exact match on structured verdict strings.

    Args:
        submission: Agent predictions. Columns: id, verdict.
        answers: Ground truth. Columns: id, verdict.

    Returns:
        Float in [0, 1]. Direction: maximize.

    Raises:
        ValueError: If submission format is invalid.
        RuntimeError: If unexpected error occurs.
    """
    try:
        if "id" not in submission.columns or "verdict" not in submission.columns:
            raise ValueError("Submission must have columns: id, verdict")

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
                raise ValueError(f"Submission missing ids: {len(missing)}")
            if extra:
                raise ValueError(f"Submission has extra ids: {len(extra)}")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["verdict_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        row_scores = []
        for _, row in merged.iterrows():
            true_fields = _parse_verdict(row["verdict_true"])
            pred_fields = _parse_verdict(row["verdict_pred"])

            if not true_fields:
                row_scores.append(0.0)
                continue

            matches = 0
            total = len(VERDICT_FIELDS)
            for field in VERDICT_FIELDS:
                true_val = true_fields.get(field, "")
                pred_val = pred_fields.get(field, "")
                if true_val == pred_val:
                    matches += 1

            row_scores.append(matches / total)

        if not row_scores:
            return 0.0

        score = float(np.mean(row_scores))
        if np.isnan(score):
            return 0.0
        return score

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
