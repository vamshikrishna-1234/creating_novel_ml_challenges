from pathlib import Path


def prepare(raw: Path, public: Path, private: Path) -> None:
    import random as _rnd
    import numpy as np
    import pandas as pd

    raw, public, private = Path(raw), Path(public), Path(private)

    df = pd.read_csv(raw / "data.csv")

    SPLIT_SEED = 314159
    TRAIN_FRAC = 0.75
    N = len(df)

    indices = np.arange(N)
    rng_split = np.random.RandomState(SPLIT_SEED)
    rng_split.shuffle(indices)
    split_point = int(N * TRAIN_FRAC)
    train_ids = set(indices[:split_point].tolist())
    test_ids = set(indices[split_point:].tolist())

    train_df = df.iloc[sorted(train_ids)].copy().reset_index(drop=True)
    test_df = df.iloc[sorted(test_ids)].copy().reset_index(drop=True)

    # ---- Obfuscate column names ----
    feature_cols = [c for c in df.columns if c not in ("sample_id", "yield_output", "noise_sigma")]

    rng_cols = _rnd.Random(271828)
    shuffled_cols = list(feature_cols)
    rng_cols.shuffle(shuffled_cols)
    col_map = {orig: f"F_{i:02d}" for i, orig in enumerate(shuffled_cols)}

    for old_name, new_name in col_map.items():
        train_df = train_df.rename(columns={old_name: new_name})
        test_df = test_df.rename(columns={old_name: new_name})

    train_df = train_df.rename(columns={"yield_output": "target"})
    test_df = test_df.rename(columns={"yield_output": "target"})

    feature_cols_new = sorted(col_map.values())

    # ---- Write outputs ----
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_out = train_df[["sample_id"] + feature_cols_new + ["target"]]
    train_out.to_csv(public / "train.csv", index=False)

    test_out = test_df[["sample_id"] + feature_cols_new]
    test_out.to_csv(public / "test.csv", index=False)

    answers = test_df[["sample_id", "target", "noise_sigma"]].copy()
    answers = answers.rename(columns={"noise_sigma": "true_sigma"})
    answers.to_csv(private / "answers.csv", index=False)

    # ---- Sample submission: global mean + constant interval ----
    train_mean = train_out["target"].mean()
    train_std = train_out["target"].std()
    sample_sub = test_df[["sample_id"]].copy()
    sample_sub["point_estimate"] = round(train_mean, 4)
    sample_sub["lower_90"] = round(train_mean - 2.0 * train_std, 4)
    sample_sub["upper_90"] = round(train_mean + 2.0 * train_std, 4)
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    print(f"Train samples: {len(train_out)}")
    print(f"Test samples:  {len(test_out)}")
    print(f"Features:      {len(feature_cols_new)}")
    print(f"Train target   mean={train_out['target'].mean():.3f}  std={train_out['target'].std():.3f}")


if __name__ == "__main__":
    prepare(Path("raw_data"), Path("pub"), Path("priv"))
