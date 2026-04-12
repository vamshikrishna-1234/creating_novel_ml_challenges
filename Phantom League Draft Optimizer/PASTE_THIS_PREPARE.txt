"""
prepare.py — Battle Arena Hero Draft Recommendation

Reads raw/ directory containing heroes.csv, maps.csv, situations.csv.
Produces:
  public/  heroes.csv, maps.csv, train.csv, test.csv, sample_submission.csv
  private/ answers.csv

Key twist: in test situations, ~40% of enemy picks are masked to -1,
simulating fog-of-war conditions. The model must learn hero interactions
from fully-visible training data and generalize to partial observability
at test time.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw)
    public = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = np.random.RandomState(seed)

    heroes = pd.read_csv(str(raw / "heroes.csv"))
    maps = pd.read_csv(str(raw / "maps.csv"))
    df = pd.read_csv(str(raw / "situations.csv"))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split = int(len(df) * 0.75)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    assert len(set(train_df["situation_id"]) & set(test_df["situation_id"])) == 0

    feature_cols = [c for c in df.columns if c != "best_pick"]

    train_df.to_csv(str(public / "train.csv"), index=False)

    test_pub = test_df[feature_cols].copy()
    enemy_cols_a = [f"team_a_pick_{i}" for i in range(1, 6)]
    enemy_cols_b = [f"team_b_pick_{i}" for i in range(1, 6)]

    for idx in test_pub.index:
        picking = test_pub.loc[idx, "picking_team"]
        if picking == "A":
            enemy_cols = enemy_cols_b
        else:
            enemy_cols = enemy_cols_a

        for col in enemy_cols:
            if test_pub.loc[idx, col] != -1 and rng.random() < 0.40:
                test_pub.loc[idx, col] = -1

    test_pub.to_csv(str(public / "test.csv"), index=False)

    heroes.to_csv(str(public / "heroes.csv"), index=False)
    maps.to_csv(str(public / "maps.csv"), index=False)

    test_df[["situation_id", "best_pick"]].to_csv(
        str(private / "answers.csv"), index=False
    )

    sub = test_df[["situation_id"]].copy()
    for i in range(1, 6):
        sub[f"pick_{i}"] = 0
    sub.to_csv(str(public / "sample_submission.csv"), index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"public/ and private/ written.")


if __name__ == "__main__":
    prepare(Path("raw_data"), Path("pub"), Path("priv"))
