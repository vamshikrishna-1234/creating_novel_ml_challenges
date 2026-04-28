# Challenge creation form — fill-in

**Platform status:** **Draft** — *Bilateral Asymmetry Anchored Mammography Triage*.

Tie this challenge to the **accepted dataset**: Bilateral Mammography View Corpus With Per-Case Asymmetry And Malignancy Labels.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Bilateral Asymmetry Anchored Mammography Triage
```

---

## 3) Problem Description

# Bilateral Asymmetry Anchored Mammography Triage

## Overview

This is a **Computer Vision / Medical Imaging** challenge that models a realistic mammographic triage workflow. Each *case* is a study consisting of up to four mammographic views — left craniocaudal (LCC), right craniocaudal (RCC), left mediolateral oblique (LMLO), right mediolateral oblique (RMLO). For every test case, your system must produce three jointly-evaluated outputs:

- **Per-view lesion type** — one of `none / mass / calc` for each available view in the case.
- **Per-case bilateral asymmetry** — a binary decision: is the breast tissue distribution materially different between the two sides beyond what density / projection alone explains?
- **Per-case calibrated malignancy probability** — a real number in `[0, 1]` denoting the likelihood the case contains malignancy.

The challenge is *anchored* on bilateral comparison because that is how the radiology workflow actually triages: a finding on one side is interpreted in the context of the contralateral side. A purely single-image classifier cannot recover the asymmetry signal; the model must reason across paired views.

A subset of test cases has **one of its four views missing** — the corresponding entry in `test.csv` has an empty `image_file`, and the renamed image on disk is a 1×1 placeholder. Your model must still emit a `lesion_pred` for that view (predicting `none` is a reasonable default), and the lesion-prediction quality on that subset is graded as a separate robustness sub-term.

### What makes this challenging

- **Paired-view comparison is required.** Per-view classifiers cannot recover bilateral asymmetry. A Siamese / multi-instance fusion across L↔R views is the natural architecture.
- **View-drop robustness.** A meaningful slice of test cases is missing one of the four views. Models that quietly assume "always 4 views" degrade badly on that slice; the subset is graded on its own with its own weight.
- **Calibrated malignancy probability.** A naïve classifier that outputs hard 0/1 or always-0.5 wins nothing on the calibration term — the malignancy term is `AUROC + ECE`. The ECE penalty zeros out around 6.7 % mean miscalibration.
- **Asymmetry is not just "lesion exists".** The asymmetry label is *case-level* and reflects whether one breast carries pathology that the contralateral side does not. A model that just predicts `asymmetry = (any view shows mass or calc)` will still miss bilateral cases and over-predict on artefactual unilateral findings.
- **No source identifiers in the public CSVs.** Image identifiers are seeded permutations of the underlying case identifiers; you cannot rejoin to any external mammography corpus by id. Pixel-level matching is also defeated because the renamed PNGs are de-identified copies.

### Intended approach

This problem is intended to be solved with a **trained vision model** that fuses features across the (up to four) views of each case. We expect strong submissions to:

- Use a pretrained ImageNet / vision-foundation backbone (CNN, ViT, or similar) and fine-tune on the 3-class lesion task per view.
- Add a **case-level fusion head** that pools features across the 2–4 available views to predict `asymmetry_pred` and `malignancy_prob`.
- Apply **temperature scaling**, **deep ensembles**, or **conformal-style abstention** on a held-out training fold to improve the malignancy calibration term.
- Train with **random view-dropout augmentation** (zero one of the four views with probability 0.2–0.4) so the model degrades gracefully when a view is missing at test time.
- Optionally use **lateral-flip Siamese encoders** so the LCC ↔ RCC and LMLO ↔ RMLO pairs share a comparison head.

### What to use

- ImageNet-pretrained or vision-foundation (CC-licensed) backbones; standard data augmentation; multi-task heads; held-out validation slices carved out of `train.csv`.
- Per-case multi-instance learning frameworks for the asymmetry + malignancy heads.
- Temperature / Platt scaling and uncertainty-aware ensembles for the calibration sub-term.

### What not to use

Using any of the approaches below is grounds for solution rejection on review, regardless of leaderboard score:

- **External mammography image archives.** Do NOT download external mammography corpora at training or inference time to recover the original per-image labels, patient identity, or laterality / view metadata that the public files do not expose. The labels must be learned from `public/` alone. Pretrained weights from natural-image / general-vision corpora are fine; pretrained weights specifically trained on a mammography dataset are not.
- **Hard-coded id → label dictionaries** or perceptual-hash matching of the released PNGs against any external public mirror to recover ground truth.
- **Hosted / closed-source API models** at any stage of training or inference (OpenAI, Anthropic, Google, Cohere, Mistral-API, xAI, etc.), including any distillation / pseudo-labelling from such teachers. Only open-weights models and self-trained pipelines are permitted.
- **Rule-only / no-ML pipelines.** A submission that simply thresholds pixel statistics or uses hand-coded if/else rules without a trained ML component does not satisfy the challenge intent and will be rejected on review.
- **Probability-vector hacks that game the calibration term.** Submitting deliberately over-smoothed probabilities to artificially lower ECE while the discrete decision remains confident, or any trick that decouples the binary decision from the reported probability mass. Calibration is part of the grade.
- **Grader / platform exploitation.** Hard-coded answer dictionaries, filesystem probes for `private/answers.csv`, attempts to inspect any column outside `public/`, or any other channel that bypasses the model's actual predictions.
- **Test-time fitting of normalisers, transforms, or learned representations on test images** (any form of test-set leakage). Fitting on training images only is fine.
- **Ensembles mixing allowed and prohibited components.** An ensemble is allowed only if every component is itself trained / used within the rules above. One prohibited component contaminates the whole ensemble.

## Evaluation

The grader computes a composite in `[0, 1]` (higher better):

```
Final = 0.30 * MacroF1_lesion
      + 0.25 * F1_asymmetry
      + 0.25 * AUROC_malignancy
      + 0.10 * max(0, 1 - 15 * ECE_malignancy)
      + 0.10 * MacroF1_lesion_dropped_views
```

- **MacroF1_lesion** — macro-F1 over the three lesion classes `{none, mass, calc}` across **all** view rows in the test set (including the dropped-view subset).
- **F1_asymmetry** — F1 on the positive class of the binary case-level asymmetry decision (positive = "asymmetric"). Includes every test case.
- **AUROC_malignancy** — area under the ROC curve of the case-level `malignancy_prob` against the true binary `malignant` label. Threshold-free.
- **ECE_malignancy** — 15-bin Expected Calibration Error of the case-level malignancy probability. Confidence = `malignancy_prob`; correctness = `(malignancy_prob >= 0.5) == true_malignancy`. Penalty slope `15 *` zeroes the term at ECE ≈ 0.067.
- **MacroF1_lesion_dropped_views** — macro-F1 over the lesion classes restricted to the rows where the public test.csv exposes the view as dropped (image_file column empty). Robustness sub-term.

Any structural failure in the submission (wrong columns, duplicate / missing `id`s, NaN, out-of-range probabilities, wrong `row_type`) makes the grader return `0.0`.

## Dataset

The prepared dataset is in `public/` with the following structure:

- `public/train.csv` — one row per (image_id, view) for training cases. Columns: `image_id` (int), `view` (one of `LCC, RCC, LMLO, RMLO`), `image_file` (string, points to `public/images/<image_file>`), `lesion` (string, one of `none, mass, calc`).
- `public/test.csv` — one row per (image_id, view) for test cases. Columns: `image_id`, `view`, `image_file`. **For dropped views the `image_file` field is an empty string** and the corresponding PNG on disk is a 1×1 black placeholder; you must still emit a per-view `lesion_pred` for that row.
- `public/sample_submission.csv` — required submission columns and one valid baseline row per test `id`.
- `public/images/<image_id>_<view>.png` — 8-bit grayscale mammogram view (for non-dropped views) or 1×1 black PNG placeholder (for dropped views).
- `private/answers.csv` — withheld; the grader uses this to score submissions.

There is no separate validation split; carve your own holdout out of `train.csv`.

## Submission

Submit a CSV at `./working/submission.csv` with exactly these columns:

- `id` — string. For view rows: `view_<image_id padded to 6>_<view>`; for case rows: `case_<image_id padded to 6>`. The set of `id`s must equal the set in `public/sample_submission.csv` exactly.
- `row_type` — string. Must equal `view` for view rows and `case` for case rows. Must match the `row_type` of the same `id` in `public/sample_submission.csv`.
- `lesion_pred` — string. One of `none / mass / calc`. **Required for view rows;** for case rows the column must be present but its value is ignored — predict `none` to keep the file valid.
- `asymmetry_pred` — int in `{0, 1}`. **Required for case rows;** values `>= 0.5` are treated as `1`. For view rows the column must be present but its value is ignored — predict `0`.
- `malignancy_prob` — float in `[0, 1]`. **Required for case rows;** for view rows the column must be present but its value is ignored — predict `0.5`.

Example (illustrative; abbreviated):

```
id,row_type,lesion_pred,asymmetry_pred,malignancy_prob
view_000042_LCC,view,mass,0,0.5
view_000042_RCC,view,none,0,0.5
view_000042_LMLO,view,mass,0,0.5
view_000042_RMLO,view,none,0,0.5
case_000042,case,none,1,0.81
```

**Requirements:**

- Exactly one row per `id` in `public/sample_submission.csv`, plus a header row.
- All five columns above must be present (case-sensitive).
- Every probability must be a finite number in `[0, 1]`. No NaNs, no infinities, no negative values, no values greater than 1.
- `id` values must be unique and equal to the set in `public/sample_submission.csv`.
- Any violation causes the grader to return 0.0.

---

## 4) Tags

**Select:** `image`, `medical`, `multimodal`, `small-data`

---

## 5) Grading Configuration

- **Grade direction:** **Maximize**
- **Theoretical minimum:** `0`
- **Theoretical maximum:** `1`

---

## 6) Grading Script

**Select:** `Custom`

```python
import numpy as np
import pandas as pd


REQUIRED_SUB_COLS = {"id", "row_type", "lesion_pred", "asymmetry_pred", "malignancy_prob"}
REQUIRED_ANS_COLS = {
    "id",
    "row_type",
    "true_lesion",
    "true_asymmetry",
    "true_malignancy",
    "is_dropped_view",
}
LESION_LABELS = ("none", "mass", "calc")
ECE_BINS = 15
ECE_PENALTY = 15.0


def _binary_f1(y_true, y_pred, positive=1):
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())
    if tp + fp == 0 and tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _macro_f1_strings(y_true, y_pred, labels):
    f1s = []
    for lab in labels:
        tp = int(((y_true == lab) & (y_pred == lab)).sum())
        fp = int(((y_true != lab) & (y_pred == lab)).sum())
        fn = int(((y_true == lab) & (y_pred != lab)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0 or len(y_true) == 0:
        return 0.0
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    n = len(y_score)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return float(max(0.0, min(1.0, (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg))))


def _ece(confidences, correctness, n_bins):
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences <= hi) if b == n_bins - 1 else (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(correctness[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Score submission vs answers (private/answers.csv). Float in [0,1]."""
    try:
        if not REQUIRED_SUB_COLS.issubset(set(submission.columns)):
            return 0.0
        if not REQUIRED_ANS_COLS.issubset(set(answers.columns)):
            return 0.0
        sub = submission[list(REQUIRED_SUB_COLS)].copy()
        ans = answers[list(REQUIRED_ANS_COLS)].copy()
        if sub["id"].duplicated().any() or ans["id"].duplicated().any():
            return 0.0
        if len(sub) != len(ans):
            return 0.0
        if set(sub["id"]) != set(ans["id"]):
            return 0.0

        merged = ans.merge(sub, on="id", how="left", suffixes=("_a", "_s"))
        if len(merged) != len(ans) or merged["row_type_s"].isna().any():
            return 0.0
        if (merged["row_type_s"].astype(str) != merged["row_type_a"].astype(str)).any():
            return 0.0
        merged["row_type"] = merged["row_type_a"]

        view_mask = merged["row_type"].astype(str) == "view"
        case_mask = merged["row_type"].astype(str) == "case"

        view_rows = merged.loc[view_mask].copy()
        if view_rows["lesion_pred"].isna().any():
            return 0.0
        view_rows["lesion_pred"] = view_rows["lesion_pred"].astype(str)
        if not view_rows["lesion_pred"].isin(LESION_LABELS).all():
            return 0.0

        y_true_lesion = view_rows["true_lesion"].astype(str).to_numpy()
        y_pred_lesion = view_rows["lesion_pred"].astype(str).to_numpy()
        is_dropped = view_rows["is_dropped_view"].astype(int).to_numpy() == 1
        macro_f1_lesion = _macro_f1_strings(y_true_lesion, y_pred_lesion, LESION_LABELS)
        macro_f1_dropped = (
            _macro_f1_strings(y_true_lesion[is_dropped], y_pred_lesion[is_dropped], LESION_LABELS)
            if is_dropped.any() else 0.0
        )

        case_rows = merged.loc[case_mask].copy()
        if case_rows["asymmetry_pred"].isna().any() or case_rows["malignancy_prob"].isna().any():
            return 0.0
        try:
            asym = pd.to_numeric(case_rows["asymmetry_pred"], errors="coerce").astype(float).to_numpy()
            mal = pd.to_numeric(case_rows["malignancy_prob"], errors="coerce").astype(float).to_numpy()
        except Exception:
            return 0.0
        if not np.all(np.isfinite(asym)) or not np.all(np.isfinite(mal)):
            return 0.0
        asym_int = (asym >= 0.5).astype(int)
        if (mal < 0.0).any() or (mal > 1.0).any():
            return 0.0

        y_true_asym = case_rows["true_asymmetry"].astype(int).to_numpy()
        y_true_mal = case_rows["true_malignancy"].astype(int).to_numpy()
        f1_asym = _binary_f1(y_true_asym, asym_int, positive=1)
        auroc_mal = _auroc(y_true_mal, mal)
        correctness = ((mal >= 0.5).astype(int) == y_true_mal).astype(float)
        ece = _ece(mal, correctness, ECE_BINS)
        cal = max(0.0, 1.0 - ECE_PENALTY * ece)

        final = 0.30 * macro_f1_lesion + 0.25 * f1_asym + 0.25 * auroc_mal + 0.10 * cal + 0.10 * macro_f1_dropped
        return float(max(0.0, min(1.0, final)))
    except Exception:
        return 0.0
```

---

## 7) Prepare Script

The raw dataset shipped with this challenge is the unaltered upstream mammography source — `cases.csv` with per-view metadata and `images/*.png` per view. `prepare.py` does ALL of the challenge-specific curation (deterministic case-level train/test split, deterministic case-id permutation, deterministic per-test-case view-drop injection on a fraction of test cases, image renaming).

```python
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
    if not cases_csv.exists() or not images_dir.exists():
        raise FileNotFoundError(f"raw/ missing cases.csv or images/ at {raw}")
    df = pd.read_csv(cases_csv)
    expected = {"case_raw_id", "view", "lesion", "malignant", "asymmetric", "image_file"}
    if not expected.issubset(set(df.columns)):
        raise ValueError("raw/cases.csv missing required columns")
    df = df[df["view"].isin(VIEW_ORDER)].copy()
    df = df[df["lesion"].isin(LESION_LABELS)].copy()
    return df.reset_index(drop=True)


def _write_black_png(dst: Path) -> None:
    from PIL import Image
    Image.new("RGB", (1, 1), (0, 0, 0)).save(str(dst), format="PNG", optimize=True)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw); public = Path(public); private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (public / "images").mkdir(parents=True, exist_ok=True)

    df = _read_raw(raw)
    case_summary = (
        df.groupby("case_raw_id")
        .agg(malignant=("malignant", "max"), asymmetric=("asymmetric", "max"))
        .reset_index()
    )

    split_rng = np.random.default_rng(SPLIT_SEED)
    all_cases = case_summary["case_raw_id"].to_numpy().copy()
    split_rng.shuffle(all_cases)
    n_total = len(all_cases)
    n_train = int(round(n_total * TRAIN_FRACTION))
    train_cases = set(all_cases[:n_train].tolist())
    test_cases = [c for c in all_cases if c not in train_cases]

    id_rng = np.random.default_rng(ID_PERM_SEED)
    image_ids = np.arange(n_total)
    id_rng.shuffle(image_ids)
    case_to_imgid = {c: int(image_ids[i]) for i, c in enumerate(all_cases)}

    drop_rng = np.random.default_rng(DROP_VIEW_SEED)
    n_drop = int(round(len(test_cases) * VIEW_DROP_FRACTION_OF_TEST_CASES))
    drop_pick = drop_rng.choice(len(test_cases), size=n_drop, replace=False)
    drop_set = set(test_cases[i] for i in drop_pick)
    drop_view_choice = {c: VIEW_ORDER[int(drop_rng.integers(0, 4))] for c in drop_set}

    train_rows: List[dict] = []
    test_rows: List[dict] = []
    sub_rows: List[dict] = []
    ans_rows: List[dict] = []
    images_src = raw / "images"
    images_dst = public / "images"

    for case_raw_id, grp in df.groupby("case_raw_id", sort=False):
        image_id = case_to_imgid[case_raw_id]
        is_train = case_raw_id in train_cases
        case_views = grp.set_index("view")
        case_dropped = drop_view_choice.get(case_raw_id, None)

        for view in VIEW_ORDER:
            if view not in case_views.index:
                continue
            row = case_views.loc[view]
            new_name = f"{int(image_id):06d}_{view}.png"
            dst_png = images_dst / new_name
            view_dropped = (not is_train) and (view == case_dropped)

            if view_dropped:
                _write_black_png(dst_png)
                public_image_file = ""
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
                    "image_id": int(image_id), "view": view,
                    "image_file": new_name, "lesion": true_lesion,
                })
            else:
                test_rows.append({
                    "image_id": int(image_id), "view": view,
                    "image_file": public_image_file,
                })
                sub_rows.append({
                    "id": view_id, "row_type": "view",
                    "lesion_pred": "none", "asymmetry_pred": 0, "malignancy_prob": 0.5,
                })
                ans_rows.append({
                    "id": view_id, "row_type": "view",
                    "true_lesion": true_lesion,
                    "true_asymmetry": int(row["asymmetric"]),
                    "true_malignancy": int(row["malignant"]),
                    "is_dropped_view": int(view_dropped),
                })

        if not is_train:
            srow = case_summary[case_summary["case_raw_id"] == case_raw_id].iloc[0]
            case_id = f"case_{int(image_id):06d}"
            sub_rows.append({
                "id": case_id, "row_type": "case",
                "lesion_pred": "none", "asymmetry_pred": 0, "malignancy_prob": 0.5,
            })
            ans_rows.append({
                "id": case_id, "row_type": "case",
                "true_lesion": "none",
                "true_asymmetry": int(srow["asymmetric"]),
                "true_malignancy": int(srow["malignant"]),
                "is_dropped_view": 0,
            })

    pd.DataFrame(train_rows).sort_values(["image_id", "view"]).reset_index(drop=True).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).sort_values(["image_id", "view"]).reset_index(drop=True).to_csv(public / "test.csv", index=False)
    pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True).to_csv(public / "sample_submission.csv", index=False)
    pd.DataFrame(ans_rows).sort_values("id").reset_index(drop=True).to_csv(private / "answers.csv", index=False)
```

---

## 8) GPU Tier

**Select:** **A10G** — standard single-GPU training of a multi-view mammography classifier. Fits in 24 GB. H100 is not required.
