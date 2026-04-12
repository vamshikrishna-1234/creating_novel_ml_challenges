"""
prepare.py — Ironhold Arena Ruling Generation v2

Splits clean data, injects 10% label noise into TRAINING data only.
Test answers remain clean. No rulebook provided.
"""

from pathlib import Path


def prepare(raw: Path, public: Path, private: Path) -> None:
    import pandas as pd
    import numpy as np

    rng = np.random.RandomState(42)

    df = pd.read_csv(str(raw / "data.csv"))

    assert df["question_id"].nunique() == len(df), "Duplicate question_ids"

    indices = np.arange(len(df))
    rng.shuffle(indices)

    split_point = int(len(df) * 0.75)
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    assert len(set(train_idx) & set(test_idx)) == 0

    train_df = df.iloc[train_idx].reset_index(drop=True).copy()
    test_df = df.iloc[test_idx].reset_index(drop=True).copy()

    assert set(train_df["question_id"]) & set(test_df["question_id"]) == set()

    outcomes = ["ALLOWED", "BLOCKED", "MODIFIED"]
    noise_rng = np.random.RandomState(99)
    noise_mask = noise_rng.random(len(train_df)) < 0.10
    for idx in train_df.index[noise_mask]:
        true_out = train_df.loc[idx, "outcome"]
        wrong = [o for o in outcomes if o != true_out]
        new_out = noise_rng.choice(wrong)
        train_df.loc[idx, "outcome"] = new_out
        train_df.loc[idx, "output"] = (
            f"<analysis>{train_df.loc[idx, 'analysis']}</analysis>\n"
            f"<outcome>{new_out}</outcome>"
        )

    train_out = train_df[["question_id", "situation", "question", "output"]].copy()
    train_out.to_csv(str(public / "train.csv"), index=False)

    test_out = test_df[["question_id", "situation", "question"]].copy()
    test_out.to_csv(str(public / "test.csv"), index=False)

    sample_sub = test_df[["question_id"]].copy()
    sample_sub["output"] = (
        "<analysis>No analysis provided.</analysis>\n"
        "<outcome>BLOCKED</outcome>"
    )
    sample_sub.to_csv(str(public / "sample_submission.csv"), index=False)

    answers = test_df[["question_id", "analysis", "outcome"]].copy()
    answers.to_csv(str(private / "answers.csv"), index=False)

    print(f"Train: {len(train_out)} rows ({noise_mask.sum()} noisy)")
    print(f"Test:  {len(test_out)} rows (clean)")
    print(f"Answers: {len(answers)} rows")


if __name__ == "__main__":
    prepare(Path("raw_data"), Path("public"), Path("private"))
