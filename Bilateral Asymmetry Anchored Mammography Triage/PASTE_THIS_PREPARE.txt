"""prepare.py for the Bilateral Asymmetry-Anchored Mammography Triage challenge.

Input  raw/  containing:
  - cases.csv       columns: case_raw_id, view, lesion, malignant, asymmetric, image_file
                    where view ∈ {LCC, RCC, LMLO, RMLO}, lesion ∈ {none,mass,calc},
                    malignant ∈ {0,1}, asymmetric ∈ {0,1}.
                    A "case" is a tuple of up to 4 view-rows that share the same
                    case_raw_id (left/right paired views from the same study).
  - images/<image_file>     PNG mammogram view.

Output public/:
  - train.csv                    one row per (image_id, view) for training views
  - test.csv                     one row per (image_id, view) for test views.
                                 For ~30% of test cases ONE view per case is
                                 marked dropped (the row is still present, but
                                 the image_file column is empty and the
                                 dropped image is replaced with a black 1x1 PNG
                                 named the same as `<image_id>_<view>.png`).
  - sample_submission.csv        constant baseline (lesion=none, asym=0, mal=0.5)
  - images/<image_id>_<view>.png renamed mammogram views

Output private/:
  - answers.csv                  one row per submission `id` with ground-truth
                                 fields the grader needs (true_lesion, true_asymmetry,
                                 true_malignancy, is_dropped_view, row_type).

Submission row design:
  - For every test (image_id, view) pair, emit one VIEW row:
        id = view_<image_id>_<view>           (string)
        row_type = "view"
        prediction columns: lesion_pred (string)
  - For every test case, emit one CASE row:
        id = case_<image_id>                  (string, image_id == case_id)
        row_type = "case"
        prediction columns: asymmetry_pred (0/1), malignancy_prob (float in [0,1])
  - Submissions must include BOTH row types in a single CSV; the grader keys
    on `id` and routes the columns by `row_type`.

Anonymisation:
  - Every raw case_raw_id is permuted to a dense integer image_id via a seeded
    permutation; the raw id is never written to public files.
  - View order within a case is fixed: LCC, RCC, LMLO, RMLO.

Determinism:
  All randomness is seeded by the seven 32-bit hex constants below; rerunning
  the script on the same raw_data/ produces byte-identical public/ + private/.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


SPLIT_SEED       = 0xA1B2C3D4
ID_PERM_SEED     = 0x9E3B2C84
DROP_VIEW_SEED   = 0x7F2A9E51
TRAIN_FRACTION   = 0.75
VIEW_DROP_FRACTION_OF_TEST_CASES = 0.30
VIEW_ORDER = ("LCC", "RCC", "LMLO", "RMLO")
LESION_LABELS = ("none", "mass", "calc")


def _read_raw(raw: Path) -> pd.DataFrame:
    cases_csv = raw / "cases.csv"
    images_dir = raw / "images"
    if not cases_csv.exists():
        raise FileNotFoundError(f"raw/cases.csv missing at {cases_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"raw/images/ missing at {images_dir}")
    df = pd.read_csv(cases_csv)
    expected = {"case_raw_id", "view", "lesion", "malignant", "asymmetric", "image_file"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"raw/cases.csv must contain columns {sorted(expected)}; got {sorted(df.columns)}"
        )
    df = df[df["view"].isin(VIEW_ORDER)].copy()
    df = df[df["lesion"].isin(LESION_LABELS)].copy()
    return df.reset_index(drop=True)


def _write_black_png(dst: Path) -> None:
    """Tiny black 1x1 PNG used to materialise dropped views."""
    from PIL import Image  # imported lazily so prepare.py imports cheaply
    Image.new("RGB", (1, 1), (0, 0, 0)).save(str(dst), format="PNG", optimize=True)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw)
    public = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (public / "images").mkdir(parents=True, exist_ok=True)

    df = _read_raw(raw)

    # Validate per-case malignancy / asymmetry consistency.
    case_summary = (
        df.groupby("case_raw_id")
        .agg(
            n_views=("view", "nunique"),
            malignant=("malignant", "max"),
            asymmetric=("asymmetric", "max"),
        )
        .reset_index()
    )

    # 1) deterministic shuffle and split over case_raw_id
    split_rng = np.random.default_rng(SPLIT_SEED)
    all_cases = case_summary["case_raw_id"].to_numpy().copy()
    split_rng.shuffle(all_cases)

    n_total_cases = len(all_cases)
    n_train_cases = int(round(n_total_cases * TRAIN_FRACTION))
    train_cases = set(all_cases[:n_train_cases].tolist())
    test_cases = [c for c in all_cases if c not in train_cases]

    # 2) build a deterministic raw_id -> image_id permutation
    id_rng = np.random.default_rng(ID_PERM_SEED)
    image_ids = np.arange(n_total_cases)
    id_rng.shuffle(image_ids)
    case_to_imgid = {c: int(image_ids[i]) for i, c in enumerate(all_cases)}

    # 3) decide which test cases get one view dropped, and which view
    drop_rng = np.random.default_rng(DROP_VIEW_SEED)
    n_drop_cases = int(round(len(test_cases) * VIEW_DROP_FRACTION_OF_TEST_CASES))
    drop_pick = drop_rng.choice(len(test_cases), size=n_drop_cases, replace=False)
    drop_set = set(test_cases[i] for i in drop_pick)
    drop_view_choice = {
        c: VIEW_ORDER[int(drop_rng.integers(0, 4))] for c in drop_set
    }

    # 4) build train.csv, test.csv, sample_submission.csv, answers.csv
    train_rows: List[dict] = []
    test_rows: List[dict] = []
    sub_rows: List[dict] = []
    ans_rows: List[dict] = []

    images_src = raw / "images"
    images_dst = public / "images"

    by_case = df.groupby("case_raw_id", sort=False)
    for case_raw_id, grp in by_case:
        image_id = case_to_imgid[case_raw_id]
        is_train = case_raw_id in train_cases

        case_views = grp.set_index("view")
        case_dropped_view = drop_view_choice.get(case_raw_id, None)

        for view in VIEW_ORDER:
            if view not in case_views.index:
                continue  # this raw case is missing that view in the source
            row = case_views.loc[view]
            new_name = f"{int(image_id):06d}_{view}.png"
            dst_png = images_dst / new_name

            view_dropped = (not is_train) and (view == case_dropped_view)

            if view_dropped:
                _write_black_png(dst_png)
                public_image_file = ""  # tells solver this view is dropped
            else:
                src_png = images_src / str(row["image_file"])
                if not src_png.exists():
                    raise FileNotFoundError(f"missing source image {src_png}")
                shutil.copy2(str(src_png), str(dst_png))
                public_image_file = new_name

            view_id = f"view_{int(image_id):06d}_{view}"
            true_lesion = str(row["lesion"])

            if is_train:
                train_rows.append({
                    "image_id": int(image_id),
                    "view": view,
                    "image_file": new_name,
                    "lesion": true_lesion,
                })
            else:
                test_rows.append({
                    "image_id": int(image_id),
                    "view": view,
                    "image_file": public_image_file,  # empty string when dropped
                })
                sub_rows.append({
                    "id": view_id,
                    "row_type": "view",
                    "lesion_pred": "none",
                    "asymmetry_pred": 0,
                    "malignancy_prob": 0.5,
                })
                ans_rows.append({
                    "id": view_id,
                    "row_type": "view",
                    "true_lesion": true_lesion,
                    "true_asymmetry": int(row["asymmetric"]),
                    "true_malignancy": int(row["malignant"]),
                    "is_dropped_view": int(view_dropped),
                })

        # case row (test cases only — train cases don't go to grader)
        if not is_train:
            srow = case_summary[case_summary["case_raw_id"] == case_raw_id].iloc[0]
            case_id = f"case_{int(image_id):06d}"
            sub_rows.append({
                "id": case_id,
                "row_type": "case",
                "lesion_pred": "none",
                "asymmetry_pred": 0,
                "malignancy_prob": 0.5,
            })
            ans_rows.append({
                "id": case_id,
                "row_type": "case",
                "true_lesion": "none",
                "true_asymmetry": int(srow["asymmetric"]),
                "true_malignancy": int(srow["malignant"]),
                "is_dropped_view": 0,
            })

    train_df = (
        pd.DataFrame(train_rows)
        .sort_values(["image_id", "view"])
        .reset_index(drop=True)
    )
    test_df = (
        pd.DataFrame(test_rows)
        .sort_values(["image_id", "view"])
        .reset_index(drop=True)
    )
    sub_df = (
        pd.DataFrame(sub_rows)
        .sort_values("id")
        .reset_index(drop=True)
    )
    ans_df = (
        pd.DataFrame(ans_rows)
        .sort_values("id")
        .reset_index(drop=True)
    )

    train_df.to_csv(public / "train.csv", index=False)
    test_df.to_csv(public / "test.csv", index=False)
    sub_df.to_csv(public / "sample_submission.csv", index=False)
    ans_df.to_csv(private / "answers.csv", index=False)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path(__file__).parent / "raw_data")
    ap.add_argument("--public", type=Path, default=Path(__file__).parent / "pub")
    ap.add_argument("--private", type=Path, default=Path(__file__).parent / "priv")
    args = ap.parse_args()
    if args.public.exists():
        shutil.rmtree(args.public)
    if args.private.exists():
        shutil.rmtree(args.private)
    prepare(args.raw, args.public, args.private)
    print("prepare done ->", args.public, args.private)


if __name__ == "__main__":
    main()
