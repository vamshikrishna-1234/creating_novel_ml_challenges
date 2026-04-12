"""
Grading script: Average Character Error Rate (CER) for the
Synthetic Dialect Translation Challenge.

For each row, CER = edit_distance(predicted, ground_truth) / len(ground_truth).
Final score = mean CER across all rows.

Score range: [0.0, inf) where LOWER is better. Perfect = 0.0.

Uses dynamic-programming Levenshtein distance (no external deps).
"""

import pandas as pd
import numpy as np


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score using Average Character Error Rate (CER).

    Args:
        submission: Agent predictions. Columns: id, original.
        answers: Ground truth. Columns: id, original.

    Returns:
        Float >= 0.0. Direction: minimize. Perfect = 0.0.
    """
    try:
        if "id" not in submission.columns or "original" not in submission.columns:
            raise ValueError("Submission must have columns: id, original")

        if submission["id"].duplicated().any():
            raise ValueError("Submission must not contain duplicate id values")

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

        if merged["original_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        cers = []
        for _, row in merged.iterrows():
            true_str = str(row["original_true"])
            pred_str = str(row["original_pred"])
            if len(true_str) == 0:
                cer = 0.0 if len(pred_str) == 0 else 1.0
            else:
                cer = _levenshtein(pred_str, true_str) / len(true_str)
            cers.append(cer)

        if not cers:
            return 1.0

        score = float(np.mean(cers))
        if np.isnan(score):
            return 1.0
        return score

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python grade.py <submission.csv> <answers.csv>")
        sys.exit(1)
    sub = pd.read_csv(sys.argv[1])
    ans = pd.read_csv(sys.argv[2])
    score = grade(sub, ans)
    print(f"CER Score: {score:.4f}")
