# Challenge creation form — fill-in

**Platform status:** **Draft** — *Patient Linkage Verification On Chest Radiographs With Decoy Pathology Matching*.

Tie this challenge to the **accepted dataset**: Anonymised Chest Radiograph Corpus With Per-Image Pathology Pattern Annotations.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Patient Linkage Verification On Chest Radiographs With Decoy Pathology Matching
```

---

## 3) Problem Description

# Patient Linkage Verification On Chest Radiographs With Decoy Pathology Matching

## Overview

This is a **Computer Vision / Medical Imaging** challenge with two intertwined tasks per query.

For every test query (a single chest X-ray of a patient), the system is given a **gallery of 5 candidate chest X-rays** and must:

1. **Patient linkage** — rank the 5 candidates such that the candidate from the **same patient** as the query is at the top. Output a per-candidate score `cand_score_0 .. cand_score_4`; higher = more likely same-patient. Candidate order in the test file is fixed, your scores must follow that order.
2. **Pathology pattern** — predict the binary 14-class pathology vector of the *query* image: 14 binary indicators `pred_0 .. pred_13` and 14 calibrated probabilities `p_0 .. p_13`.

The challenge is named *with decoy pathology matching* because the four decoy candidates in every gallery are **selected to match the query's pathology pattern within Hamming distance ≤ 1** of the 14-class pattern. A model that solves the linkage task by simply classifying the candidates' pathology and matching to the query's predicted pathology cannot win — every candidate has roughly the same pathology fingerprint as the same-patient candidate. The only signal that disambiguates is the **patient-specific anatomical signature** — rib spacing, clavicle morphology, mediastinal silhouette, projection geometry, etc.

### What makes this challenging

- **Decoys defeat the obvious shortcut.** The pathology head and the linkage head must learn fundamentally different features. A model that tries to use pathology-vector matching for linkage will sit near-random because the decoys are pathology-matched.
- **Two heads, two graded subtasks.** The submission carries one `link` row and one `patho` row per query. The grader weights MRR (0.30) + Hits@1 (0.20) on linkage and MacroF1 (0.25) + MacroAUROC (0.15) + ECE (0.10) on pathology.
- **Multi-label pathology, not single-class.** The 14 disease classes are independent indicators. Naïve softmax / argmax thinking won't fit; the model needs sigmoid heads with proper per-class thresholding.
- **Calibrated multi-label probabilities.** ECE is computed on the per-row maximum probability vs whether the corresponding binary indicator was correct. Hard one-hot or always-confident predictions tank ECE.
- **No source identifiers in the public CSVs.** Image identifiers are seeded permutations and disease columns are anonymised integer indices `0..13`; you cannot rejoin to any external chest-X-ray corpus or recover the canonical disease names from the public files.

### Intended approach

This problem is intended to be solved with a **trained vision model**. Strong submissions are expected to:

- Train a **shared backbone** (CNN / ViT / vision-foundation, ImageNet or general-vision pretraining) with two heads:
  - a **multi-label pathology head** (14 sigmoid logits) with binary cross-entropy loss,
  - a **patient-embedding head** (e.g. a metric-learning head using triplet / contrastive / supervised-contrastive loss with same-patient positives and different-patient negatives).
- At inference, encode the query and each candidate, compute cosine similarity / dot-product, and use that as `cand_score_*`.
- Apply **temperature scaling** on a held-out training fold to drive ECE down on the pathology probabilities.
- Threshold the per-class probabilities `p_*` into the binary `pred_*` columns using a per-class operating point chosen on a held-out fold (not a global 0.5 threshold).

### What to use

- ImageNet / general-vision pretrained backbones; metric-learning losses (triplet, contrastive, supervised-contrastive); BCE with positive-class weighting for class imbalance.
- Calibration techniques (temperature scaling, Platt scaling, ensembling) for the pathology probability head.
- Held-out folds carved out of `train.csv` for threshold and temperature tuning.

### What not to use

Using any of the approaches below is grounds for solution rejection on review, regardless of leaderboard score:

- **External chest-X-ray image archives.** Do NOT download external chest-X-ray corpora at training or inference time to recover patient-level identifiers, the canonical disease-name mapping for the anonymised disease indices, the original `Image Index` filenames, or any per-image metadata that the public files do not expose. Pretrained weights from natural-image / general-vision corpora are fine; pretrained weights specifically trained on a chest-X-ray dataset are not.
- **Hard-coded id → patient dictionaries** or perceptual-hash matching of the released PNGs against any external public mirror to recover ground truth.
- **Hosted / closed-source API models** at any stage of training or inference (OpenAI, Anthropic, Google, Cohere, Mistral-API, xAI, etc.), including any distillation / pseudo-labelling from such teachers. Only open-weights models and self-trained pipelines are permitted.
- **Pathology-only linkage shortcuts.** Submitting `cand_score_k` derived from any function that purely compares pathology-vector predictions between the query and the candidates. The linkage signal is intentionally orthogonal to the pathology pattern; using only pathology will score near random on MRR / Hits@1.
- **Rule-only / no-ML pipelines.** A submission that simply thresholds pixel statistics or uses hand-coded if/else rules without a trained ML component does not satisfy the challenge intent and will be rejected on review.
- **Probability-vector hacks that game the calibration term.** Submitting deliberately-flat `p_*` values to artificially lower ECE while the discrete `pred_*` decisions remain confident, or any trick that decouples the binary decision from the reported probability mass. Calibration is part of the grade.
- **Grader / platform exploitation.** Hard-coded answer dictionaries, filesystem probes for `private/answers.csv`, attempts to inspect any column outside `public/`, or any other channel that bypasses the model's actual predictions.
- **Test-time fitting of normalisers, transforms, or learned representations on test images** (any form of test-set leakage). Fitting on training images only is fine.
- **Ensembles mixing allowed and prohibited components.** An ensemble is allowed only if every component is itself trained / used within the rules above. One prohibited component contaminates the whole ensemble.

## Evaluation

The grader computes a composite in `[0, 1]` (higher better):

```
Final = 0.30 * MRR_patient_match
      + 0.20 * Hits@1_patient_match
      + 0.25 * MacroF1_pathology_query
      + 0.15 * MacroAUROC_pathology_query
      + 0.10 * max(0, 1 - 15 * ECE_pathology_argmax)
```

- **MRR_patient_match** — mean reciprocal rank of the true same-patient candidate after sorting `cand_score_0..cand_score_4` descending (ties broken by index ascending). 1.0 if always at top, 1/5 = 0.20 if random.
- **Hits@1_patient_match** — fraction of queries where the top-ranked candidate is the true same-patient candidate.
- **MacroF1_pathology_query** — macro-F1 over the 14 binary pathology classes, computed across the `pred_*` columns of the per-query patho rows.
- **MacroAUROC_pathology_query** — mean of the 14 one-vs-rest AUROCs of `p_<i>` against the binary `true_y_<i>`.
- **ECE_pathology_argmax** — 15-bin Expected Calibration Error pooled across queries: confidence = per-row `max(p_*)`, correctness = `pred_<argmax>` matches `true_y_<argmax>`. Penalty slope `15 *` zeros the term at ECE ≈ 0.067.

Any structural failure in the submission (wrong columns, duplicate / missing `id`s, NaN, out-of-range values, wrong `row_type`, non-binary `pred_*`) makes the grader return `0.0`.

## Dataset

The prepared dataset is in `public/`:

- `public/train.csv` — one row per training image. Columns: `id` (`img_<6-digit>`), `image_file`, `patient_id` (anonymised `pid_<6-digit>`), and 14 binary columns `has_0 .. has_13`.
- `public/test_queries.csv` — one row per test query. Columns: `query_id`, `image_file`. The query's true patient_id and pathology pattern are NOT exposed.
- `public/test_candidates.csv` — one row per test query. Columns: `query_id`, `cand_id_0..4`, `cand_image_file_0..4`. The 5 candidate `image_file`s point at PNGs in `public/images/`. **One** of the 5 candidates is from the same patient as the query; the other 4 are pathology-matched decoys. Candidate order is fixed by the file — your `cand_score_<k>` columns must match this order.
- `public/sample_submission.csv` — required submission columns and a constant baseline row per test `id`.
- `public/images/<id>.png` — 8-bit grayscale chest X-ray.
- `private/answers.csv` — withheld; the grader uses this to score submissions.

There is no separate validation split; carve your own holdout out of `train.csv`.

## Submission

Submit a CSV at `./working/submission.csv` with **two row types** keyed by `id`:

- **Link rows** (one per query):
  - `id` = `link_<query_id>` (string).
  - `row_type` = `link` (string, must match exactly).
  - `cand_score_0 .. cand_score_4` = floats; higher means "more likely same patient as the query". Order corresponds to the candidate columns in `public/test_candidates.csv`.
  - The `pred_*` and `p_*` columns must be present on link rows but their values are ignored — set `pred_<i> = 0` and `p_<i> = 0.5`.
- **Patho rows** (one per query):
  - `id` = `patho_<query_id>` (string).
  - `row_type` = `patho` (string).
  - `pred_0 .. pred_13` = ints in `{0, 1}` — predicted binary indicator per disease class.
  - `p_0 .. p_13` = floats in `[0, 1]` — predicted probability per disease class. **No sum-to-1 constraint** (independent multi-label).
  - The `cand_score_*` columns must be present on patho rows but their values are ignored — set them to `0.2` (uniform) for safety.

Example (illustrative; abbreviated columns):

```
id,row_type,cand_score_0,cand_score_1,cand_score_2,cand_score_3,cand_score_4,pred_0,...,pred_13,p_0,...,p_13
link_img_000042,link,0.10,0.05,0.81,0.02,0.02,0,...,0,0.5,...,0.5
patho_img_000042,patho,0.2,0.2,0.2,0.2,0.2,1,...,0,0.91,...,0.04
```

**Requirements:**

- Exactly two rows per query (one `link_*`, one `patho_*`); the set of `id`s must equal the set in `public/sample_submission.csv`.
- All columns above must be present (case-sensitive). All numerics finite.
- `pred_*` strictly 0/1 (integer); `p_*` in `[0, 1]`; `cand_score_*` finite floats.
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


N_CANDIDATES = 5
N_DISEASES = 14
DISEASE_IDS = tuple(range(N_DISEASES))
PROB_COLS = tuple(f"p_{i}" for i in DISEASE_IDS)
PRED_COLS = tuple(f"pred_{i}" for i in DISEASE_IDS)
TRUE_COLS = tuple(f"true_y_{i}" for i in DISEASE_IDS)
CAND_COLS = tuple(f"cand_score_{i}" for i in range(N_CANDIDATES))
REQUIRED_SUB_COLS = {"id", "row_type", *CAND_COLS, *PRED_COLS, *PROB_COLS}
REQUIRED_ANS_COLS = {"id", "row_type", "true_match_idx", *TRUE_COLS}
ECE_BINS = 15
ECE_PENALTY = 15.0


def _macro_f1_binary(y_true_mat, y_pred_mat):
    f1s = []
    for j in range(y_true_mat.shape[1]):
        yt, yp = y_true_mat[:, j], y_pred_mat[:, j]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _auroc_binary(y_true, y_score):
    pos = int((y_true == 1).sum()); neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    n = len(y_score); i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return float(max(0.0, min(1.0, (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg))))


def _macro_auroc(y_true_mat, y_score_mat):
    aurocs = []
    for j in range(y_true_mat.shape[1]):
        a = _auroc_binary(y_true_mat[:, j], y_score_mat[:, j])
        if not np.isnan(a):
            aurocs.append(a)
    return float(sum(aurocs) / len(aurocs)) if aurocs else 0.0


def _ece(confidences, correctness, n_bins):
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0; n = len(confidences)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences <= hi) if b == n_bins - 1 else (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(correctness[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if not REQUIRED_SUB_COLS.issubset(set(submission.columns)):
            return 0.0
        if not REQUIRED_ANS_COLS.issubset(set(answers.columns)):
            return 0.0
        sub = submission[list(REQUIRED_SUB_COLS)].copy()
        ans = answers[list(REQUIRED_ANS_COLS)].copy()
        if sub["id"].duplicated().any() or ans["id"].duplicated().any():
            return 0.0
        if len(sub) != len(ans) or set(sub["id"]) != set(ans["id"]):
            return 0.0
        merged = ans.merge(sub, on="id", how="left", suffixes=("_a", "_s"))
        if len(merged) != len(ans) or merged["row_type_s"].isna().any():
            return 0.0
        if (merged["row_type_s"].astype(str) != merged["row_type_a"].astype(str)).any():
            return 0.0
        merged["row_type"] = merged["row_type_a"]

        link_mask = merged["row_type"].astype(str) == "link"
        patho_mask = merged["row_type"].astype(str) == "patho"

        link_rows = merged.loc[link_mask].copy()
        for c in CAND_COLS:
            try:
                link_rows[c] = pd.to_numeric(link_rows[c], errors="coerce").astype(float)
            except Exception:
                return 0.0
        cand_mat = link_rows[list(CAND_COLS)].to_numpy()
        if not np.all(np.isfinite(cand_mat)):
            return 0.0
        true_idx = link_rows["true_match_idx"].astype(int).to_numpy()
        if (true_idx < 0).any() or (true_idx >= N_CANDIDATES).any():
            return 0.0

        ranks = np.empty(len(true_idx), dtype=float)
        for r, scores in enumerate(cand_mat):
            order = np.argsort(-scores, kind="stable")
            ranks[r] = int(np.where(order == true_idx[r])[0][0]) + 1
        mrr = float((1.0 / ranks).mean()) if len(ranks) else 0.0
        hits1 = float((ranks == 1).mean()) if len(ranks) else 0.0

        patho_rows = merged.loc[patho_mask].copy()
        for c in PRED_COLS + PROB_COLS:
            try:
                patho_rows[c] = pd.to_numeric(patho_rows[c], errors="coerce")
            except Exception:
                return 0.0
        if patho_rows[list(PRED_COLS) + list(PROB_COLS)].isna().any().any():
            return 0.0
        pred_mat = patho_rows[list(PRED_COLS)].astype(float).to_numpy()
        prob_mat = patho_rows[list(PROB_COLS)].astype(float).to_numpy()
        true_mat = patho_rows[list(TRUE_COLS)].astype(int).to_numpy()
        if not np.all(np.isin(pred_mat.astype(int), (0, 1))):
            return 0.0
        if (np.abs(pred_mat - pred_mat.astype(int)) > 1e-9).any():
            return 0.0
        pred_mat = pred_mat.astype(int)
        if not np.all(np.isfinite(prob_mat)):
            return 0.0
        if (prob_mat < 0.0).any() or (prob_mat > 1.0).any():
            return 0.0

        macro_f1 = _macro_f1_binary(true_mat, pred_mat)
        macro_auroc = _macro_auroc(true_mat, prob_mat)
        argmax_idx = prob_mat.argmax(axis=1)
        max_p = prob_mat.max(axis=1)
        argmax_correct = np.array([float(pred_mat[i, j] == true_mat[i, j])
                                    for i, j in enumerate(argmax_idx)], dtype=float)
        ece = _ece(max_p, argmax_correct, ECE_BINS)
        cal = max(0.0, 1.0 - ECE_PENALTY * ece)

        final = 0.30 * mrr + 0.20 * hits1 + 0.25 * macro_f1 + 0.15 * macro_auroc + 0.10 * cal
        return float(max(0.0, min(1.0, final)))
    except Exception:
        return 0.0
```

---

## 7) Prepare Script

The raw dataset shipped with this challenge is the unaltered upstream chest-X-ray corpus — `cxr_index.csv` (with anonymised disease columns `has_0..has_13`) and `images/*.png`. `prepare.py` does ALL of the challenge-specific curation (deterministic per-patient train/test split, anonymised patient and image identifiers, deterministic decoy sampling under the pathology-Hamming-distance constraint, deterministic candidate shuffling, image renaming).

```python
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SPLIT_SEED        = 0x11335577
ID_PERM_SEED      = 0x88AABBCC
DECOY_SEED        = 0xDEED0042
PATIENT_PERM_SEED = 0x9090CAFE
TRAIN_FRACTION    = 0.65
N_DISEASES        = 14
N_CANDIDATES      = 5
N_DECOYS          = N_CANDIDATES - 1


def _read_raw(raw: Path) -> pd.DataFrame:
    csv_path = raw / "cxr_index.csv"
    img_dir = raw / "images"
    if not csv_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"raw/ missing cxr_index.csv or images/ at {raw}")
    df = pd.read_csv(csv_path)
    needed = {"raw_patient_id", "raw_image_id", "image_file"}
    needed |= {f"has_{i}" for i in range(N_DISEASES)}
    if not needed.issubset(set(df.columns)):
        raise ValueError("raw/cxr_index.csv missing required columns")
    for c in (f"has_{i}" for i in range(N_DISEASES)):
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw); public = Path(public); private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (public / "images").mkdir(parents=True, exist_ok=True)

    df = _read_raw(raw)

    split_rng = np.random.default_rng(SPLIT_SEED)
    all_patients = df["raw_patient_id"].drop_duplicates().to_numpy().copy()
    split_rng.shuffle(all_patients)
    n_train = int(round(len(all_patients) * TRAIN_FRACTION))
    train_patients = set(all_patients[:n_train].tolist())

    pid_rng = np.random.default_rng(PATIENT_PERM_SEED)
    pid_perm = np.arange(len(all_patients)); pid_rng.shuffle(pid_perm)
    raw_pid_to_anon = {p: f"pid_{int(pid_perm[i]):06d}" for i, p in enumerate(all_patients)}

    id_rng = np.random.default_rng(ID_PERM_SEED)
    img_perm = np.arange(len(df)); id_rng.shuffle(img_perm)
    raw_imgrow_to_imgid = {i: f"img_{int(img_perm[i]):06d}" for i in range(len(df))}

    df["image_id"] = [raw_imgrow_to_imgid[i] for i in range(len(df))]
    df["anon_pid"] = df["raw_patient_id"].map(raw_pid_to_anon)
    df["is_train"] = df["raw_patient_id"].isin(train_patients)

    train_rows: List[dict] = []
    images_src = raw / "images"; images_dst = public / "images"

    for _, row in df.iterrows():
        new_name = f"{row['image_id']}.png"
        src_png = images_src / str(row["image_file"])
        dst_png = images_dst / new_name
        if not src_png.exists():
            raise FileNotFoundError(f"missing source image {src_png}")
        shutil.copy2(str(src_png), str(dst_png))
        if row["is_train"]:
            r = {"id": row["image_id"], "image_file": new_name,
                 "patient_id": row["anon_pid"]}
            for i in range(N_DISEASES):
                r[f"has_{i}"] = int(row[f"has_{i}"])
            train_rows.append(r)

    pd.DataFrame(train_rows).sort_values("id").reset_index(drop=True).to_csv(public / "train.csv", index=False)

    test_df = df[~df["is_train"]].copy()
    pat_cols = [f"has_{i}" for i in range(N_DISEASES)]
    decoy_patterns = test_df[pat_cols].to_numpy().astype(int)
    decoy_rng = np.random.default_rng(DECOY_SEED)

    queries: List[dict] = []
    candidates: List[dict] = []
    sub_rows: List[dict] = []
    ans_rows: List[dict] = []
    for anon_pid, grp in test_df.groupby("anon_pid"):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("image_id").reset_index(drop=True)
        q = grp_sorted.iloc[0]; match = grp_sorted.iloc[1]
        q_pat = np.array([int(q[c]) for c in pat_cols])
        decoy_mask = (
            (test_df["anon_pid"] != anon_pid).to_numpy()
            & (test_df["image_id"] != q["image_id"]).to_numpy()
            & (test_df["image_id"] != match["image_id"]).to_numpy()
        )
        ham = np.abs(decoy_patterns - q_pat).sum(axis=1)
        decoy_mask &= (ham <= 1)
        decoy_idx_pool = np.where(decoy_mask)[0]
        if len(decoy_idx_pool) < N_DECOYS:
            decoy_mask2 = (
                (test_df["anon_pid"] != anon_pid).to_numpy()
                & (test_df["image_id"] != q["image_id"]).to_numpy()
                & (test_df["image_id"] != match["image_id"]).to_numpy()
                & (ham <= 2)
            )
            decoy_idx_pool = np.where(decoy_mask2)[0]
            if len(decoy_idx_pool) < N_DECOYS:
                continue
        decoy_pick = decoy_rng.choice(decoy_idx_pool, size=N_DECOYS, replace=False)
        cand_image_ids = [match["image_id"]] + [test_df.iloc[int(i)]["image_id"] for i in decoy_pick]
        order = np.arange(N_CANDIDATES); decoy_rng.shuffle(order)
        true_match_idx = int(np.where(order == 0)[0][0])
        ordered_cand_ids = [cand_image_ids[i] for i in order]

        query_id = q["image_id"]
        queries.append({"query_id": query_id, "image_file": f"{query_id}.png"})
        cand_row = {"query_id": query_id}
        for k, cid in enumerate(ordered_cand_ids):
            cand_row[f"cand_id_{k}"] = cid
            cand_row[f"cand_image_file_{k}"] = f"{cid}.png"
        candidates.append(cand_row)

        link_id = f"link_{query_id}"; patho_id = f"patho_{query_id}"
        for rid, rt in ((link_id, "link"), (patho_id, "patho")):
            r = {"id": rid, "row_type": rt}
            for k in range(N_CANDIDATES):
                r[f"cand_score_{k}"] = 0.2
            for i in range(N_DISEASES):
                r[f"pred_{i}"] = 0; r[f"p_{i}"] = 0.5
            sub_rows.append(r)

        ans_link = {"id": link_id, "row_type": "link", "true_match_idx": true_match_idx}
        for i in range(N_DISEASES):
            ans_link[f"true_y_{i}"] = int(q[f"has_{i}"])
        ans_rows.append(ans_link)
        ans_patho = {"id": patho_id, "row_type": "patho", "true_match_idx": 0}
        for i in range(N_DISEASES):
            ans_patho[f"true_y_{i}"] = int(q[f"has_{i}"])
        ans_rows.append(ans_patho)

    pd.DataFrame(queries).sort_values("query_id").reset_index(drop=True).to_csv(public / "test_queries.csv", index=False)
    pd.DataFrame(candidates).sort_values("query_id").reset_index(drop=True).to_csv(public / "test_candidates.csv", index=False)
    pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True).to_csv(public / "sample_submission.csv", index=False)
    pd.DataFrame(ans_rows).sort_values("id").reset_index(drop=True).to_csv(private / "answers.csv", index=False)
```

---

## 8) GPU Tier

**Select:** **A10G** — standard single-GPU training of a metric-learning + multi-label classifier on chest X-rays. H100 is not required.
