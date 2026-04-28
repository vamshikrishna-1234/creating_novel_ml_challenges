# Challenge creation form — fill-in

**Platform status:** **Draft** — *Vendor Shifted Retinal OCT Triage*.

Tie this challenge to the **accepted dataset**: Retinal OCT B-Scan Corpus With Disease And Layer-Position Annotations.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Vendor Shifted Retinal OCT Triage With Layer Coherence Sanity
```

---

## 3) Problem Description

# Vendor Shifted Retinal OCT Triage With Layer Coherence Sanity

## Overview

This is a **Computer Vision / Medical Imaging** challenge built on retinal optical coherence tomography (OCT) B-scan images. For every test image your system must produce four things at once:

- **Disease prediction** — one of `CNV / DME / DRUSEN / NORMAL` (4 classes).
- **Disease probability vector** — `p_CNV, p_DME, p_DRUSEN, p_NORMAL`, each in `[0,1]`, summing to 1 (±1e-3).
- **Layer position prediction** — a single scalar in `[0, 1]` denoting the relative vertical position of the dominant pathologic layer in the B-scan (or, for `NORMAL` images, of the inner-segment / outer-segment junction).

The test set is *not* drawn from the same distribution as training. A meaningful fraction of test images has had its photometric (and in some cases also light geometric) appearance shifted at preparation time, simulating a transition between OCT scanner vendors. Disease performance is graded separately on the in-domain slice and on the shifted slices, so a model that overfits the training photometric distribution will visibly lose marks on the latter.

The **layer-position regression** is a structural sanity term. A model that has actually learned where the retinal layers are will continue to localise them across photometric shift; a shortcut model that only memorises low-level texture statistics will not. The grader penalises layer MAE directly.

### What makes this challenging

- **Domain shift baked into test time.** The vendor-shifted and corrupted slices are individually weighted in the final score. A 4-class classifier that scores ~0.85 macro-F1 on the in-scope slice but collapses to ~0.40 on the corrupted slice loses ~9 points of final score. Robustness training is necessary.
- **Layer-position is not separable from the image content.** It cannot be predicted from the disease label alone (each disease covers a wide spread of layer positions) — the model must look at the actual B-scan.
- **Calibrated probability vector required.** The grader includes a 15-bin Expected Calibration Error term on the argmax confidence vs disease correctness pooled over all test rows. Hard one-hot vectors will tank ECE; uniform vectors will tank macro-F1. A working calibration scheme matters.
- **AUROC is computed only on the in-scope slice** so the threshold-free term cannot be gamed by a model that always predicts NORMAL.
- **No source identifiers in the public CSVs.** Image identifiers are seeded permutations of upstream content hashes; rejoining to any external retinal-OCT corpus by id is not possible.

### Intended approach

This problem is intended to be solved with a **trained vision model**. Strong submissions are expected to:

- Fine-tune a pretrained backbone (CNN / ViT / vision-foundation; ImageNet or general-purpose pretraining) on the 4-class disease task with a small auxiliary regression head for `layer_position_pred`.
- Train with **photometric augmentation** (brightness / contrast jitter, additive noise, mild blur) to bridge the gap to the shifted test slices. Geometric augmentation should be mild — OCT layers are spatially structured.
- Apply **temperature scaling** or **deep ensembles** on a held-out training fold to drive ECE down.
- Optionally use **test-time augmentation** to stabilise predictions across the shift slices.

### What to use

- ImageNet / general-vision pretrained backbones; multi-task heads; standard augmentation; held-out folds carved out of `train.csv`.
- Calibration techniques (temperature scaling, Platt scaling, ensembling).
- Test-time augmentation (averaging predictions across mild flips / crops at inference).

### What not to use

Using any of the approaches below is grounds for solution rejection on review, regardless of leaderboard score:

- **External retinal-OCT image archives.** Do NOT download external retinal-OCT corpora at training or inference time to recover the original per-image disease labels, layer-segmentation masks, or scanner metadata that the public files do not expose. Pretrained weights from natural-image / general-vision corpora are fine; pretrained weights specifically trained on a retinal-OCT dataset are not.
- **Hard-coded id → label dictionaries** or perceptual-hash matching of the released PNGs against any external public mirror to recover ground truth.
- **Hosted / closed-source API models** at any stage of training or inference (OpenAI, Anthropic, Google, Cohere, Mistral-API, xAI, etc.), including any distillation / pseudo-labelling from such teachers. Only open-weights models and self-trained pipelines are permitted.
- **Rule-only / no-ML pipelines.** A submission that simply thresholds pixel statistics without a trained ML component does not satisfy the challenge intent and will be rejected on review.
- **Probability-vector hacks that game the calibration term.** Submitting deliberately-flat probability vectors to artificially lower ECE while the discrete decision remains confident, or any trick that decouples the argmax decision from the reported probabilities. Calibration is part of the grade.
- **Grader / platform exploitation.** Hard-coded answer dictionaries, filesystem probes for `private/answers.csv`, attempts to inspect any column outside `public/`, or any other channel that bypasses the model's actual predictions.
- **Test-time fitting of normalisers, transforms, or learned representations on test images** (any form of test-set leakage). Fitting on training images only is fine.
- **Ensembles mixing allowed and prohibited components.** An ensemble is allowed only if every component is itself trained / used within the rules above. One prohibited component contaminates the whole ensemble.

## Evaluation

The grader computes a composite in `[0, 1]` (higher better):

```
Final = 0.25 * MacroF1_disease_inscope
      + 0.20 * MacroF1_disease_vendor_shifted
      + 0.15 * MacroF1_disease_corrupted
      + 0.15 * (1 - MAE_layer_position)         # clipped to [0,1]
      + 0.15 * AUROC_disease_macro_inscope     # one-vs-rest, mean over 4 classes
      + 0.10 * max(0, 1 - 15 * ECE_disease_overall)
```

- **MacroF1_disease_*** — macro-F1 over the four disease classes restricted to the named test slice.
- **MAE_layer_position** — Mean Absolute Error on the layer-position scalar over all test rows. Final layer term is `max(0, 1 - MAE)`.
- **AUROC_disease_macro_inscope** — mean of the four one-vs-rest AUROCs of the corresponding probability column against the binary indicator `true_disease == c`, computed on the in-scope slice only.
- **ECE_disease_overall** — 15-bin ECE pooled across all test rows, with confidence = `argmax(p)` and correctness = `argmax(p) == true_disease`. Penalty slope `15 *` zeros the term at ECE ≈ 0.067.

Any structural failure in the submission (wrong columns, duplicate / missing `id`s, NaN, out-of-range values, probability rows that do not sum to ~1) makes the grader return `0.0`.

## Dataset

The prepared dataset is in `public/`:

- `public/train.csv` — columns `id, image_file, disease, layer_position`. `id` is a string of the form `oct_<6-digit>`. `layer_position` is in `[0, 1]`.
- `public/test.csv` — columns `id, image_file` (no labels). The PNG referenced by `image_file` may have had photometric / mild geometric shift applied at prepare time; the `id` and `image_file` columns do not reveal which slice the image is in.
- `public/sample_submission.csv` — required submission columns and a constant baseline row per test `id`.
- `public/images/<id>.png` — 8-bit grayscale OCT B-scan.
- `private/answers.csv` — withheld; the grader uses this to score submissions.

There is no separate validation split; carve your own holdout out of `train.csv`.

## Submission

Submit a CSV at `./working/submission.csv` with exactly these columns:

- `id` — string. Must equal one `id` from `public/test.csv`. The set of `id`s must equal the set in `public/sample_submission.csv` exactly.
- `disease_pred` — string. One of `CNV, DME, DRUSEN, NORMAL`.
- `layer_position_pred` — float in `[0, 1]`.
- `p_CNV, p_DME, p_DRUSEN, p_NORMAL` — floats in `[0, 1]`. Each row's four probabilities must sum to 1 within `±1e-3`.

Example (illustrative; abbreviated):

```
id,disease_pred,layer_position_pred,p_CNV,p_DME,p_DRUSEN,p_NORMAL
oct_000042,CNV,0.32,0.75,0.10,0.10,0.05
oct_000043,NORMAL,0.55,0.05,0.05,0.10,0.80
```

**Requirements:**

- Exactly one row per `id` in `public/sample_submission.csv`, plus a header row.
- All seven columns above must be present (case-sensitive).
- All numerics finite. `layer_position_pred` and every probability in `[0, 1]`.
- `id` values unique and equal to the set in `public/sample_submission.csv`.
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


REQUIRED_SUB_COLS = {
    "id",
    "disease_pred",
    "layer_position_pred",
    "p_CNV", "p_DME", "p_DRUSEN", "p_NORMAL",
}
REQUIRED_ANS_COLS = {"id", "slice", "true_disease", "true_layer_position"}
DISEASE_LABELS = ("CNV", "DME", "DRUSEN", "NORMAL")
PROB_COLS = ("p_CNV", "p_DME", "p_DRUSEN", "p_NORMAL")
PROB_SUM_TOL = 1e-3
ECE_BINS = 15
ECE_PENALTY = 15.0


def _macro_f1(y_true, y_pred, labels):
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


def _auroc_binary(y_true, y_score):
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
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


def _macro_auroc(y_true_str, prob_matrix, labels):
    aurocs = []
    for j, lab in enumerate(labels):
        a = _auroc_binary((y_true_str == lab).astype(int), prob_matrix[:, j])
        if not np.isnan(a):
            aurocs.append(a)
    return float(sum(aurocs) / len(aurocs)) if aurocs else 0.0


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
        merged = ans.merge(sub, on="id", how="left")
        if len(merged) != len(ans):
            return 0.0
        if merged[list(REQUIRED_SUB_COLS - {"id"})].isna().any().any():
            return 0.0
        merged["disease_pred"] = merged["disease_pred"].astype(str)
        if not merged["disease_pred"].isin(DISEASE_LABELS).all():
            return 0.0
        try:
            merged["layer_position_pred"] = pd.to_numeric(merged["layer_position_pred"], errors="coerce").astype(float)
            for c in PROB_COLS:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").astype(float)
        except Exception:
            return 0.0
        if not np.all(np.isfinite(merged["layer_position_pred"].to_numpy())):
            return 0.0
        for c in PROB_COLS:
            arr = merged[c].to_numpy()
            if not np.all(np.isfinite(arr)) or (arr < 0.0).any() or (arr > 1.0).any():
                return 0.0
        prob_mat = merged[list(PROB_COLS)].to_numpy()
        if (np.abs(prob_mat.sum(axis=1) - 1.0) > PROB_SUM_TOL).any():
            return 0.0
        layer_pred = merged["layer_position_pred"].to_numpy()
        if (layer_pred < 0.0).any() or (layer_pred > 1.0).any():
            return 0.0

        y_true_disease = merged["true_disease"].astype(str).to_numpy()
        y_pred_disease = merged["disease_pred"].astype(str).to_numpy()
        y_true_layer = merged["true_layer_position"].astype(float).to_numpy()
        slice_arr = merged["slice"].astype(str).to_numpy()
        m_in = slice_arr == "inscope"
        m_vd = slice_arr == "vendor"
        m_cr = slice_arr == "corrupted"

        f1_in = _macro_f1(y_true_disease[m_in], y_pred_disease[m_in], DISEASE_LABELS) if m_in.any() else 0.0
        f1_vd = _macro_f1(y_true_disease[m_vd], y_pred_disease[m_vd], DISEASE_LABELS) if m_vd.any() else 0.0
        f1_cr = _macro_f1(y_true_disease[m_cr], y_pred_disease[m_cr], DISEASE_LABELS) if m_cr.any() else 0.0
        layer_term = max(0.0, 1.0 - float(np.mean(np.abs(y_true_layer - layer_pred))))
        auroc_in = _macro_auroc(y_true_disease[m_in], prob_mat[m_in], DISEASE_LABELS) if m_in.any() else 0.0

        max_p = prob_mat.max(axis=1)
        argmax_lbl = np.array([DISEASE_LABELS[i] for i in prob_mat.argmax(axis=1)])
        ece = _ece(max_p, (argmax_lbl == y_true_disease).astype(float), ECE_BINS)
        cal = max(0.0, 1.0 - ECE_PENALTY * ece)

        final = 0.25 * f1_in + 0.20 * f1_vd + 0.15 * f1_cr + 0.15 * layer_term + 0.15 * auroc_in + 0.10 * cal
        return float(max(0.0, min(1.0, final)))
    except Exception:
        return 0.0
```

---

## 7) Prepare Script

The raw dataset shipped with this challenge is the unaltered upstream corpus — `oct_images.csv` and `images/*.png`. `prepare.py` does ALL of the challenge-specific curation (deterministic train/test split, deterministic test-slice assignment among `{inscope, vendor, corrupted}`, deterministic image-id permutation, image renaming, and per-test-image perturbation onto disk).

```python
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


SPLIT_SEED        = 0x12345678
ID_PERM_SEED      = 0xDEADBEEF
SLICE_SEED        = 0xC0FFEE01
PERTURB_SEED      = 0xBEEFCAFE
TRAIN_FRACTION    = 0.70
SLICE_FRACTIONS   = {"inscope": 0.50, "vendor": 0.30, "corrupted": 0.20}
DISEASE_LABELS    = ("CNV", "DME", "DRUSEN", "NORMAL")


def _read_raw(raw: Path) -> pd.DataFrame:
    csv_path = raw / "oct_images.csv"
    img_dir = raw / "images"
    if not csv_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"raw/ missing oct_images.csv or images/ at {raw}")
    df = pd.read_csv(csv_path)
    df = df[df["disease"].isin(DISEASE_LABELS)].copy()
    df["layer_position"] = pd.to_numeric(df["layer_position"], errors="coerce")
    df = df.dropna(subset=["layer_position"])
    df = df[(df["layer_position"] >= 0.0) & (df["layer_position"] <= 1.0)].copy()
    return df.reset_index(drop=True)


def _apply_perturbation(src_png: Path, dst_png: Path, mode: str, rng) -> None:
    from PIL import Image, ImageEnhance, ImageFilter
    if mode == "none":
        shutil.copy2(str(src_png), str(dst_png)); return
    img = Image.open(str(src_png)).convert("L")
    if mode == "vendor":
        b = float(0.85 + rng.random() * 0.30); c = float(0.85 + rng.random() * 0.30)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        if rng.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    elif mode == "corrupted":
        b = float(0.65 + rng.random() * 0.55); c = float(0.55 + rng.random() * 0.65)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        if rng.random() < 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        import numpy as np
        arr = np.asarray(img).astype(np.float32) + rng.normal(0.0, 12.0, size=img.size[::-1]).astype(np.float32)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    img.save(str(dst_png), format="PNG", optimize=True)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw); public = Path(public); private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (public / "images").mkdir(parents=True, exist_ok=True)

    df = _read_raw(raw)
    n_total = len(df)

    split_rng = np.random.default_rng(SPLIT_SEED)
    perm = np.arange(n_total); split_rng.shuffle(perm)
    n_train = int(round(n_total * TRAIN_FRACTION))
    train_pos = set(perm[:n_train].tolist())
    test_pos = [i for i in range(n_total) if i not in train_pos]

    id_rng = np.random.default_rng(ID_PERM_SEED)
    image_ids = np.arange(n_total); id_rng.shuffle(image_ids)
    pos_to_imgid = {i: int(image_ids[i]) for i in range(n_total)}

    slice_rng = np.random.default_rng(SLICE_SEED)
    n_test = len(test_pos)
    n_in = int(round(n_test * SLICE_FRACTIONS["inscope"]))
    n_vd = int(round(n_test * SLICE_FRACTIONS["vendor"]))
    n_cr = n_test - n_in - n_vd
    test_pos_arr = np.array(test_pos); slice_rng.shuffle(test_pos_arr)
    inscope_pos = set(test_pos_arr[:n_in].tolist())
    vendor_pos = set(test_pos_arr[n_in:n_in + n_vd].tolist())
    perturb_rng = np.random.default_rng(PERTURB_SEED)

    train_rows: List[dict] = []
    test_rows: List[dict] = []
    sub_rows: List[dict] = []
    ans_rows: List[dict] = []
    images_src = raw / "images"
    images_dst = public / "images"

    for pos, row in df.iterrows():
        image_id = pos_to_imgid[pos]
        new_name = f"oct_{int(image_id):06d}.png"
        src_png = images_src / str(row["image_file"])
        if not src_png.exists():
            raise FileNotFoundError(f"missing source image {src_png}")
        dst_png = images_dst / new_name
        sample_id = f"oct_{int(image_id):06d}"
        if pos in train_pos:
            shutil.copy2(str(src_png), str(dst_png))
            train_rows.append({"id": sample_id, "image_file": new_name,
                               "disease": str(row["disease"]),
                               "layer_position": float(row["layer_position"])})
        else:
            slice_name = "inscope" if pos in inscope_pos else ("vendor" if pos in vendor_pos else "corrupted")
            mode = {"inscope": "none", "vendor": "vendor", "corrupted": "corrupted"}[slice_name]
            _apply_perturbation(src_png, dst_png, mode, perturb_rng)
            test_rows.append({"id": sample_id, "image_file": new_name})
            sub_rows.append({"id": sample_id, "disease_pred": "NORMAL",
                             "layer_position_pred": 0.5,
                             "p_CNV": 0.25, "p_DME": 0.25, "p_DRUSEN": 0.25, "p_NORMAL": 0.25})
            ans_rows.append({"id": sample_id, "slice": slice_name,
                             "true_disease": str(row["disease"]),
                             "true_layer_position": float(row["layer_position"])})

    pd.DataFrame(train_rows).sort_values("id").reset_index(drop=True).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).sort_values("id").reset_index(drop=True).to_csv(public / "test.csv", index=False)
    pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True).to_csv(public / "sample_submission.csv", index=False)
    pd.DataFrame(ans_rows).sort_values("id").reset_index(drop=True).to_csv(private / "answers.csv", index=False)
```

---

## 8) GPU Tier

**Select:** **A10G** — standard single-GPU fine-tuning of an OCT classifier with multi-task heads. H100 is not required.
