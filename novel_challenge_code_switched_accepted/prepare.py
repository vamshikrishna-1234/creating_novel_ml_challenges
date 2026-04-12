"""
Prepare script: transforms raw data.csv into public/ and private/ splits.
Deterministic (fixed random_state). No row appears in both train and test.
Uses only libraries available in Kaggle Python Docker image.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Read raw data (single file)
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)

    required_cols = {"id", "text", "category", "target"}
    if set(df.columns) != required_cols:
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    # Deterministic split: 80% train, 20% test (no overlap)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Ensure no duplicate ids between train and test (prev_reviews: no duplicates)
    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(test_ids), "Train and test must not share any ids"

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # public: train.csv = id + features + labels
    train_df.to_csv(public / "train.csv", index=False)

    # public: test.csv = id + features only (no target)
    test_public = test_df[["id", "text", "category"]].copy()
    test_public.to_csv(public / "test.csv", index=False)

    # public: sample_submission.csv — required format example
    sample = test_df[["id"]].copy()
    sample["prediction"] = 0.0  # placeholder
    sample.to_csv(public / "sample_submission.csv", index=False)

    # private: answers.csv = id + labels only
    answers = test_df[["id", "target"]].copy()
    answers.to_csv(private / "answers.csv", index=False)
