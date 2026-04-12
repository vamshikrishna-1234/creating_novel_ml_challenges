# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Industrial Reactor Monitoring Data with Heteroscedastic Noise.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Multi-Objective Confidence Calibration
```

---

## 3) Problem Description

# Synthetic Multi-Objective Confidence Calibration

## Overview

This is a **multi-objective feature engineering** challenge with a **novel compound scoring metric** that has no single dominant strategy. For each test sample, you must produce a **three-part structured output** — a central value and a lower/upper confidence band — that simultaneously optimizes three competing objectives: accuracy, calibration, and tightness.

The challenge is built on a fully synthetic tabular dataset with 30 anonymized features (`F_00` through `F_29`, a mix of floats and integer-coded categoricals). The training set (37,500 samples) includes a `target` column; the test set (12,500 samples) withholds it. What makes this task distinct from standard tabular problems is the **multi-objective scoring**: simply minimizing error is insufficient — you must also produce well-calibrated confidence bands whose coverage hits exactly 90% (penalizing both over- and under-coverage), while keeping the bands as tight as possible.

The underlying feature-to-output relationship is non-linear, involving multi-way interactions and threshold effects. The signal-to-noise ratio **varies across different regions of the feature space**: some samples are governed by strong signal, while others sit in high-variability zones. Discovering which features drive the output — and which features indicate the local noise level — requires careful feature engineering across the mixed-type feature set.

Your task: for each of the 12,500 test samples, produce a structured triple `(point_estimate, lower_90, upper_90)` that maximizes the compound score described below.

## Evaluation

Submissions are scored using a **custom compound metric** with three competing objectives:

**1. Accuracy (50% weight)**

```
baseline_deviation = mean(|target - mean(target)|)
deviation = mean(|target - point_estimate|)
accuracy_score = max(0, 1 - deviation / baseline_deviation)
```

**2. Coverage Calibration (30% weight)**

```
hit_rate = fraction of test samples where lower_90 <= target <= upper_90
coverage_score = max(0, 1 - 5 * |hit_rate - 0.90|)
```

Optimal when exactly 90% of true values fall within the submitted bands. Penalizes both over-coverage (bands too wide, wasting precision) and under-coverage (bands too narrow, missing values).

**3. Band Tightness (20% weight)**

```
mean_width = mean(upper_90 - lower_90)
value_range = max(target) - min(target)
tightness_score = max(0, 1 - mean_width / value_range)
```

Tighter bands score higher, rewarding precise confidence assessment.

**Final Score:**

```
score = 0.50 * accuracy_score + 0.30 * coverage_score + 0.20 * tightness_score
```

Score range: **[0.0, 1.0]**. **Higher is better.** The three objectives are in tension: widening bands improves coverage but hurts tightness; narrowing bands improves tightness but may miss true values.

## Dataset

- `train.csv` — 37,500 samples: sample_id (int), F_00 through F_29 (30 features, mix of float and int), target (float)
- `test.csv` — 12,500 samples: sample_id (int), F_00 through F_29 (30 features) — target withheld
- `sample_submission.csv` — 12,500 rows: sample_id (int), point_estimate (float), lower_90 (float), upper_90 (float). Shows the required submission format using a constant baseline.

### Feature Details

| Column | Type | Description |
|--------|------|-------------|
| sample_id | int | Unique sample identifier |
| F_00 | float | Anonymized feature |
| F_01 | int | Anonymized feature (integer-coded, 5 levels: 0–4) |
| F_02 – F_10 | float | Anonymized features |
| F_11 | int | Anonymized feature (integer-coded, 4 levels: 0–3) |
| F_12 | int | Anonymized feature (integer-coded, 12 levels: 0–11) |
| F_13 | float | Anonymized feature |
| F_14 | int | Anonymized feature (integer-coded, 8 levels: 0–7) |
| F_15 – F_16 | float | Anonymized features |
| F_17 | int | Anonymized feature (integer-coded, 15 levels: 0–14) |
| F_18 – F_29 | float | Anonymized features |
| target | float | Output value (train only) |

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| sample_id | int | Sample identifier from test.csv |
| point_estimate | float | Central value for this sample |
| lower_90 | float | Lower bound of 90% confidence band |
| upper_90 | float | Upper bound of 90% confidence band |

**Requirements:**
- Must contain exactly 12,500 rows (one per test sample).
- Include a header row.
- Each row must have `lower_90 <= upper_90`.
- All values must be finite numbers (no NaN or Inf).

**Example of a correctly formatted submission file:**

```
sample_id,point_estimate,lower_90,upper_90
7,3.8500,1.2000,6.5000
15,5.1200,2.4000,7.8400
22,2.0100,-0.6000,4.6200
38,4.5000,3.0000,6.0000
41,6.3300,3.5000,9.1600
```

---

## 4) Tags

**Select:** `feature-engineering`

---

## 5) Grading Configuration

- **Grade direction:** Maximize
- **Theoretical minimum:** 0
- **Theoretical maximum:** 1.0

---

## 6) Grading Script

```python
import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv).
        answers: Ground truth labels (loaded from private/answers.csv).

    Returns:
        A float score in [0.0, 1.0]. Higher is better.
        Combined metric: 0.50 * point_accuracy + 0.30 * coverage_score + 0.20 * sharpness_score.
        Returns 0.0 for invalid or malformed submissions.
    """
    try:
        required_cols = {"sample_id", "point_estimate", "lower_90", "upper_90"}
        if not required_cols.issubset(submission.columns):
            return 0.0

        if "sample_id" not in answers.columns or "target" not in answers.columns:
            return 0.0

        if submission["sample_id"].duplicated().any():
            return 0.0
        if answers["sample_id"].duplicated().any():
            return 0.0

        if (
            set(submission["sample_id"]) != set(answers["sample_id"])
            or len(submission) != len(answers)
        ):
            return 0.0

        merged = answers.merge(
            submission, on="sample_id", how="left", suffixes=("", "_sub")
        )

        if len(merged) == 0:
            return 0.0

        for col in ["point_estimate", "lower_90", "upper_90"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            if merged[col].isna().any():
                return 0.0

        if not np.all(np.isfinite(merged["point_estimate"])):
            return 0.0
        if not np.all(np.isfinite(merged["lower_90"])):
            return 0.0
        if not np.all(np.isfinite(merged["upper_90"])):
            return 0.0

        if (merged["lower_90"] > merged["upper_90"]).any():
            return 0.0

        y_true = merged["target"].values.astype(float)
        y_pred = merged["point_estimate"].values.astype(float)
        lo = merged["lower_90"].values.astype(float)
        hi = merged["upper_90"].values.astype(float)

        baseline_mae = float(np.abs(y_true - y_true.mean()).mean())
        if baseline_mae < 1e-9:
            baseline_mae = 1.0
        mae = float(np.abs(y_true - y_pred).mean())
        point_score = max(0.0, 1.0 - mae / baseline_mae)

        in_interval = (y_true >= lo) & (y_true <= hi)
        coverage = float(in_interval.mean())
        coverage_score = max(0.0, 1.0 - 5.0 * abs(coverage - 0.90))

        widths = hi - lo
        mean_width = float(widths.mean())
        target_range = float(y_true.max() - y_true.min())
        if target_range < 1e-9:
            target_range = 1.0
        sharpness_score = max(0.0, 1.0 - mean_width / target_range)

        final = (
            0.50 * point_score
            + 0.30 * coverage_score
            + 0.20 * sharpness_score
        )

        if np.isnan(final):
            return 0.0

        return float(np.clip(final, 0.0, 1.0))

    except Exception:
        return 0.0
```

---

## 7) Prepare Script

```python
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
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 12,500 rows containing columns sample_id, point_estimate, lower_90, and upper_90, where lower_90 <= upper_90 for every row and all values are finite.
- **Rationale:** Missing columns, wrong row counts, inverted bounds, or non-finite values cause the grader to return a score of zero.

### Rubric 2
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Loads and uses all 30 feature columns from train.csv without dropping rows or corrupting numeric values during preprocessing.
- **Rationale:** The dataset contains both float and integer-coded features. Incorrectly parsing or discarding columns removes signal needed for prediction.

### Rubric 3
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves a combined score above the constant-prediction baseline (~0.34) on the test set, producing sample-specific point estimates rather than a single constant value for all samples.
- **Rationale:** Submitting the training mean with a fixed band for every sample is trivially achievable and demonstrates no useful modeling.

### Rubric 4
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Criteria:** Confidence bands vary across test samples rather than being identical constant-width bands for all rows.
- **Rationale:** The compound metric penalizes both over-coverage and under-coverage. Constant-width bands cannot adapt to varying signal-to-noise ratios across different samples, limiting the achievable score.

### Rubric 5
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Does not access private answer data or test set ground-truth target values during the prediction pipeline.
- **Rationale:** Using ground-truth test targets produces inflated scores that do not reflect genuine modeling capability.
