"""
Grading script: Accuracy for the Ambiguous Instruction Resolution Challenge.

Compares predicted answer labels (A/B/C/D) against ground truth.
Score = fraction of correct predictions.

Score range: [0.0, 1.0] where HIGHER is better. Perfect = 1.0.
Random baseline = ~0.25.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score using classification accuracy.

    Args:
        submission: Agent predictions. Columns: id, answer.
        answers: Ground truth. Columns: id, answer.

    Returns:
        Float in [0, 1]. Direction: maximize. Perfect = 1.0.
    """
    try:
        if "id" not in submission.columns or "answer" not in submission.columns:
            raise ValueError("Submission must have columns: id, answer")

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

        if merged["answer_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        merged["answer_true"] = merged["answer_true"].astype(str).str.strip().str.upper()
        merged["answer_pred"] = merged["answer_pred"].astype(str).str.strip().str.upper()

        accuracy = (merged["answer_true"] == merged["answer_pred"]).mean()

        score = float(accuracy)
        if np.isnan(score):
            return 0.0
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
    print(f"Accuracy: {score:.4f}")
