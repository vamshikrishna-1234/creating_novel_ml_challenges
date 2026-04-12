"""
prepare.py — Synthetic Contextual Sequence Transduction

Reads raw_data/data.csv.
Produces:
    pub/  train.csv   (sample_id, source, target)
          test.csv    (sample_id, source)
          sample_submission.csv  (sample_id, target = "aaaaaa")
    priv/ answers.csv (sample_id, target)
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

    train_df[["sample_id", "source", "target"]].to_csv(
        str(pub / "train.csv"), index=False
    )
    test_df[["sample_id", "source"]].to_csv(
        str(pub / "test.csv"), index=False
    )

    test_df[["sample_id", "target"]].to_csv(
        str(priv / "answers.csv"), index=False
    )

    sub = test_df[["sample_id"]].copy()
    sub["target"] = "aaaaaa"
    sub.to_csv(str(pub / "sample_submission.csv"), index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"pub/ and priv/ written.")


if __name__ == "__main__":
    prepare()
