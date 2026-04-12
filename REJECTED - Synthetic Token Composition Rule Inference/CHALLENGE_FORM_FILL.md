# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Multi-Token Interaction Sequences.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Token Composition Rule Inference
```

---

## 3) Problem Description

# Synthetic Token Composition Rule Inference

## Overview

This challenge presents a **multi-rule latent property interaction decoding** problem set in a fictional industrial material compatibility testing facility. A quality control laboratory evaluates combinations of proprietary material components — each identified only by an opaque alphanumeric code (from a pool of 40 designators T_00–T_39). For each test batch, 4–7 component codes are combined, and the facility's automated classification system assigns a **3-segment outcome code** in `X-Y-Z` format, where each segment is determined by a *different* unknown function operating on hidden latent properties of the components.

The outcome code encodes three independent assessments: the first segment (X ∈ {0,1}) captures a binary compatibility verdict, the second (Y ∈ {0–9}) reflects a graded stability index, and the third (Z ∈ {0–4}) indicates a categorical durability class. Crucially, each segment is governed by a **structurally distinct latent interaction rule** — the function producing X differs fundamentally from the functions producing Y or Z. Alongside each test batch, ambient instrumentation records (thermal drift, pressure index, concentration proxy) and a processing priority flag are logged.

**What makes this problem uniquely difficult:**

- **Unseen-combination generalization**: The test set exclusively contains component groupings absent from training. Every individual component code appears in training, but no test-set grouping was observed during training. Memorization-based approaches score near zero.
- **Multi-rule disentanglement**: The three output segments follow three structurally different hidden rules. A model must independently discover each rule and how components interact under it — a single unified mapping will fail.
- **Decoy components**: Not every component in a batch influences the outcome. Some are inert fillers included for process reasons, and identifying which codes are meaningful is part of the challenge.
- **Irrelevant instrumentation**: The numeric sensor readings and priority flag are logged per batch but may carry no predictive signal for the outcome code.
- **Unordered inputs**: Component codes within each batch record are listed in arbitrary order with no positional significance.

## Evaluation

Submissions are scored using **exact-match accuracy** on the full 3-segment outcome code string. All three segments must be correct for a match. **Higher is better.** Minimum: 0.0, Maximum: 1.0, Random baseline: ~0.01 (approximately 100 possible outcome codes).

## Dataset

- `train.csv` — 12,048 training batches: id (int), input_tokens (string, space-separated component codes), feat_1 (float), feat_2 (float), feat_3 (float), priority (int, 1–5), output (string, 3-segment outcome code in X-Y-Z format)
- `test.csv` — 3,012 test batches: same columns as train.csv except output is withheld
- `sample_submission.csv` — 3,012 rows: id (int), output (string, baseline constant prediction). Shows the required submission format with one row per test batch.

| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique batch identifier |
| input_tokens | string | Space-separated component codes (4–7 codes from the pool T_00 through T_39, unordered) |
| feat_1 | float | Thermal drift reading at time of batch processing (standard-normal distributed, mean 0, std 1) |
| feat_2 | float | Ambient pressure index during batch processing (uniformly distributed, 0–100 scale) |
| feat_3 | float | Concentration proxy measurement (positive float, exponentially distributed, mean ~2.0) |
| priority | int | Processing priority level assigned to the batch (integer 1–5) |
| output | string | 3-segment outcome code in format X-Y-Z (train only) |

## Submission

Submit a CSV file with exactly 3,012 rows (one per test batch), a header row, and two columns:

| Column | Type | Description |
|--------|------|-------------|
| id | int | Batch identifier from test.csv |
| output | string | Predicted outcome code in exact X-Y-Z format |

**Example of a correctly formatted submission file:**

id,output
12048,1-7-3
12049,0-5-2
12050,1-9-0
12051,0-3-1
...

---

## 4) Tags

**Select:** `feature-engineering`

---

## 5) Grading Configuration

- **Grade direction:** **Maximize**
- **Theoretical minimum:** `0`
- **Theoretical maximum:** `1`

---

## 6) Grading Script

**Select:** `Custom`

```python
import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv)
        answers: Ground truth labels (loaded from private/answers.csv)

    Returns:
        A float score between 0.0 and 1.0. Higher is better.
    """
    ans_cols = answers[["id", "output"]].copy()
    sub_cols = submission[["id", "output"]].copy()
    merged = ans_cols.merge(sub_cols, on="id", suffixes=("_true", "_pred"))

    if len(merged) == 0:
        raise ValueError("No common IDs between submission and answers")

    if merged["output_pred"].isna().any():
        raise ValueError("Submission has missing predictions for some IDs")

    accuracy = (
        merged["output_true"].astype(str).str.strip()
        == merged["output_pred"].astype(str).str.strip()
    ).mean()

    return float(accuracy)
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
    data = pd.read_csv(raw / "data.csv")
    data = data.sort_values("id").reset_index(drop=True)

    rng = _rnd.Random(1509602442)
    np_rng = np.random.RandomState(42)

    all_real_tokens = sorted(
        {t for inp in data["input_tokens"] for t in inp.split()}
    )

    n_real = len(all_real_tokens)
    n_decoy = 10
    all_labels = [f"T_{i:02d}" for i in range(n_real + n_decoy)]
    rng.shuffle(all_labels)

    token_rename = {
        orig: all_labels[i] for i, orig in enumerate(all_real_tokens)
    }
    decoy_pool = all_labels[n_real:]

    combos = sorted(data["input_tokens"].unique().tolist())
    rng_split = _rnd.Random(4277998674)
    rng_split.shuffle(combos)

    split_idx = int(len(combos) * 0.8)
    train_combos = list(combos[:split_idx])
    test_combos = list(combos[split_idx:])

    tokens_in_train = set()
    for combo in train_combos:
        tokens_in_train.update(combo.split())
    missing = set(all_real_tokens) - tokens_in_train

    if missing:
        for tok in sorted(missing):
            for i, combo in enumerate(test_combos):
                if tok in combo.split():
                    test_combos.append(train_combos.pop())
                    train_combos.append(test_combos.pop(i))
                    break

    train_set = set(train_combos)
    test_set = set(test_combos)

    train_df = data[data["input_tokens"].isin(train_set)].copy()
    test_df = data[data["input_tokens"].isin(test_set)].copy()
    train_df = train_df.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    def obfuscate(inp):
        tokens = inp.split()
        renamed = [token_rename[t] for t in tokens]
        n_dec = rng.randint(1, 2)
        renamed.extend(rng.sample(decoy_pool, n_dec))
        rng.shuffle(renamed)
        return " ".join(renamed)

    train_df["input_tokens"] = train_df["input_tokens"].apply(obfuscate)
    test_df["input_tokens"] = test_df["input_tokens"].apply(obfuscate)

    for df in [train_df, test_df]:
        n = len(df)
        df["feat_1"] = np_rng.normal(0, 1, n).round(3)
        df["feat_2"] = np_rng.uniform(0, 100, n).round(2)
        df["feat_3"] = np_rng.exponential(2.0, n).round(3)
        df["priority"] = np_rng.randint(1, 6, n)

    train_df = train_df.reset_index(drop=True)
    train_df["id"] = range(len(train_df))
    test_df = test_df.reset_index(drop=True)
    test_df["id"] = range(len(train_df), len(train_df) + len(test_df))

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_cols = [
        "id", "input_tokens", "feat_1", "feat_2", "feat_3", "priority",
        "output",
    ]
    test_cols = [
        "id", "input_tokens", "feat_1", "feat_2", "feat_3", "priority",
    ]

    train_df[train_cols].to_csv(public / "train.csv", index=False)
    test_df[test_cols].to_csv(public / "test.csv", index=False)

    most_common_output = test_df["output"].mode().iloc[0]
    sample = test_df[["id"]].copy()
    sample["output"] = most_common_output
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "output"]].to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly parses variable-length token sequences from the input_tokens column and handles all sequence lengths (4–7 tokens) without truncation or encoding errors.
- **Rationale:** Inputs vary in length and contain space-separated token codes. Incorrect parsing will break the entire prediction pipeline.

### Rubric 2
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 3,012 rows, columns `id` and `output`, where each output matches the X-Y-Z format.
- **Rationale:** An incorrectly formatted submission will fail the grader. The output must exactly match the specified 3-part code format.

### Rubric 3
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves exact-match accuracy meaningfully above random baseline (~0.01) on the test set.
- **Rationale:** A solution scoring at or near 0.01 indicates no meaningful learning — functionally equivalent to random prediction from the output code space.

### Rubric 4
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Evaluates intermediate results on a validation split and iterates on the approach if initial accuracy is near baseline.
- **Rationale:** Good engineering practice involves measuring progress and adjusting strategy when results are poor.

### Rubric 5
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not use test set labels or leak information from the private answers into the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces artificially inflated scores that do not reflect genuine model capability.
