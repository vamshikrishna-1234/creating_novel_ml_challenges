"""
Grading script: Accuracy for Latent Selection Function Inference Challenge.

Compares predicted purchased_item_id against ground truth for each session.
Score = fraction of correct predictions.

Score range: [0.0, 1.0] where HIGHER is better. Perfect = 1.0.
Random baseline ~ 0.10.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if "session_id" not in submission.columns or "purchased_item_id" not in submission.columns:
            raise ValueError("Submission must have columns: session_id, purchased_item_id")

        if submission["session_id"].duplicated().any():
            raise ValueError("Submission contains duplicate session_id values")

        if answers["session_id"].duplicated().any():
            raise ValueError("Answers contains duplicate session_id values")

        if len(submission) != len(answers):
            raise ValueError(
                f"Submission must have exactly {len(answers)} rows, got {len(submission)}"
            )

        sub_ids = set(submission["session_id"])
        ans_ids = set(answers["session_id"])
        if sub_ids != ans_ids:
            missing = ans_ids - sub_ids
            extra = sub_ids - ans_ids
            if missing:
                raise ValueError(f"Submission missing session_ids: {len(missing)}")
            if extra:
                raise ValueError(f"Submission has extra session_ids: {len(extra)}")

        merged = answers.merge(
            submission, on="session_id", how="left",
            suffixes=("_true", "_pred")
        )

        if merged["purchased_item_id_pred"].isna().any():
            raise ValueError("Submission has missing predictions after merge")

        accuracy = (
            merged["purchased_item_id_true"] == merged["purchased_item_id_pred"]
        ).mean()

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
    print(f"Accuracy: {grade(sub, ans):.4f}")
