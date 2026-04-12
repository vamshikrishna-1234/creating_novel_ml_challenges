"""
prepare.py – Synthetic Corrupted Alignment Signal Recovery.

Reads raw_data/data.csv.
Produces:
    pub/  train.csv   (sample_id, instruction, response, quality_tier)
          test.csv    (sample_id, instruction, response)
          sample_submission.csv  (sample_id, quality_tier = 2)
    priv/ answers.csv (sample_id, quality_tier)
"""

from pathlib import Path
import pandas as pd


def prepare(raw_dir="raw_data", public_dir="pub", private_dir="priv", seed=42):
    raw = Path(raw_dir)
    pub = Path(public_dir)
    priv = Path(private_dir)
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(raw / "data.csv"))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * 0.75)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    assert len(set(train_df["sample_id"]) & set(test_df["sample_id"])) == 0

    keep_train = ["sample_id", "instruction", "response", "quality_tier"]
    keep_test = ["sample_id", "instruction", "response"]

    train_df[keep_train].to_csv(str(pub / "train.csv"), index=False)
    test_df[keep_test].to_csv(str(pub / "test.csv"), index=False)

    test_df[["sample_id", "quality_tier"]].to_csv(str(priv / "answers.csv"), index=False)

    sub = test_df[["sample_id"]].copy()
    sub["quality_tier"] = 2
    sub.to_csv(str(pub / "sample_submission.csv"), index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"pub/ and priv/ written.")


if __name__ == "__main__":
    prepare()
