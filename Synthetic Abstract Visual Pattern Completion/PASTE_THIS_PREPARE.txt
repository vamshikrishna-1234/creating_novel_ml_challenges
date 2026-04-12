"""
prepare.py – Synthetic Visual Feature Imputation Under Hidden Relational Constraints.

Reads raw_data/labels.csv and raw_data/images/.
Produces:
    pub/  train/  *.png
          test/   *.png
          train.csv          (image_id, prop_0..prop_4)
          sample_submission.csv (image_id, prop_0..prop_4 filled with 0)
    priv/ answers.csv        (image_id, prop_0..prop_4)
"""

from pathlib import Path
import shutil
import pandas as pd


def prepare(raw_dir="raw_data", public_dir="pub", private_dir="priv", seed: int = 42) -> None:
    raw = Path(raw_dir)
    pub = Path(public_dir)
    priv = Path(private_dir)
    for d in [pub / "train", pub / "test", priv]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(raw / "labels.csv"))
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * 0.75)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    assert len(set(train_df["image_id"]) & set(test_df["image_id"])) == 0

    prop_cols = ["prop_0", "prop_1", "prop_2", "prop_3", "prop_4"]
    keep_cols = ["image_id"] + prop_cols

    train_df[keep_cols].to_csv(str(pub / "train.csv"), index=False)

    test_answers = test_df[keep_cols].copy()
    test_answers.to_csv(str(priv / "answers.csv"), index=False)

    sub = test_df[["image_id"]].copy()
    for c in prop_cols:
        sub[c] = 0
    sub.to_csv(str(pub / "sample_submission.csv"), index=False)

    img_src = raw / "images"
    for _, row in train_df.iterrows():
        fn = f"{int(row['image_id']):05d}.png"
        shutil.copy2(str(img_src / fn), str(pub / "train" / fn))

    for _, row in test_df.iterrows():
        fn = f"{int(row['image_id']):05d}.png"
        shutil.copy2(str(img_src / fn), str(pub / "test" / fn))

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"pub/ and priv/ written.")


if __name__ == "__main__":
    prepare()
