# Challenge creation form — fill-in

## Title

```
Synthetic Corrupted Alignment Signal Recovery
```

## Problem Description

# Synthetic Corrupted Alignment Signal Recovery

## Overview

In reinforcement learning from human feedback (RLHF), reward model quality depends entirely on the integrity of annotator preference data. When annotation teams are heterogeneous — with inconsistent criteria, partial correlations between raters, and even adversarial contributors — the resulting training signal becomes a corrupted version of the true quality judgment.

This challenge presents 60,000 synthetic instruction-response pairs that were independently rated by a hidden ensemble of 5 annotator personas. Three personas are legitimate but weigh quality dimensions differently. One is adversarial — it systematically inverts quality ratings for a subset of topic categories. One is a noisy blend of two other personas. Two personas share partial correlations in their rating functions, making clean disentanglement difficult.

The task is to predict the **ground-truth quality tier** (0 through 4), defined as the consensus of the 3 legitimate personas only, for each test pair. The solver receives only the instruction text and response text — no numeric scores or annotator metadata are provided.

Recovering the true quality signal requires analyzing the response text to assess latent quality dimensions (factual consistency, constraint adherence, format compliance, coherence), identifying which dimensions drive the legitimate consensus, and detecting patterns of adversarial contamination across topic categories.

Approximately 15% of training tier labels are perturbed by ±1 tier (label noise), establishing an irreducible error floor.

## Evaluation

Submissions are scored using **Macro F1** across 5 quality tiers (0-4).

Higher is better. A constant prediction of tier 2 scores approximately 0.07. A random baseline scores approximately 0.20. The theoretical maximum is approximately 0.85 due to label noise.

## Dataset

After preparation, the public directory contains:

| File                   | Description                                                          |
|------------------------|----------------------------------------------------------------------|
| `train.csv`            | 45,000 labeled pairs with columns below                             |
| `test.csv`             | 15,000 unlabeled pairs (same columns minus `quality_tier`)          |
| `sample_submission.csv`| Example submission with all tiers set to 2                          |

Training and test columns:

| Column         | Type   | Description                                                   |
|----------------|--------|---------------------------------------------------------------|
| sample_id      | int    | Unique identifier                                             |
| instruction    | str    | Instruction text specifying topic, format, length, constraint |
| response       | str    | Response text with varying quality dimensions                 |
| quality_tier   | int    | Ground-truth quality tier (0–4) — **training only**           |

## Submission

Submit a CSV file with the following format:

| Column       | Type | Description                              |
|--------------|------|------------------------------------------|
| sample_id    | int  | Identifier from the test set             |
| quality_tier | int  | Predicted quality tier (0, 1, 2, 3, or 4)|

**Requirements:**
- Must contain exactly 15,000 rows (one per test sample)
- Must include a header row
- All `sample_id` values must be unique and match the test set
- `quality_tier` values must be integers in {0, 1, 2, 3, 4}

## Rubrics

### Rubric 1
- **Criteria:** Correctly loads the training CSV and parses the instruction and response text columns for downstream processing.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** The text fields are the sole source of signal for quality assessment; failing to load or parse them makes prediction impossible.

### Rubric 2
- **Criteria:** Produces a valid submission CSV with columns `sample_id` and `quality_tier`, correct row count, unique IDs matching the test set, and integer tier values in {0, 1, 2, 3, 4}.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** Malformed submissions receive a score of 0.0 from the grader; structural correctness is a prerequisite for any evaluation.

### Rubric 3
- **Criteria:** Achieves a Macro F1 exceeding 0.25 on the test set (above the random baseline of ~0.20).
- **Type:** MODELING
- **Importance:** REQUIRED
- **Rationale:** Scoring above the random baseline confirms that the model has learned at least some structure from the data rather than guessing.

### Rubric 4
- **Criteria:** Uses a language model or text feature extraction method (e.g., embeddings, TF-IDF, fine-tuned transformer) to derive features from the instruction and response text.
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Rationale:** The quality tier is determined by latent textual properties that require NLP-based feature extraction; surface-level heuristics alone cannot recover the hidden quality dimensions.

### Rubric 5
- **Criteria:** Does not use test labels or test-set statistics during training or feature engineering.
- **Type:** TRAINING
- **Importance:** UNIVERSAL
- **Rationale:** Using test information during training constitutes data leakage and produces unreliable performance estimates.

## Grading

```yaml
grading:
  method: "program"
  script: "grade.py"
  metric: "macro_f1"
  direction: "maximize"
  min_score: 0.0
  max_score: 1.0
```

## Grading Script

```python
"""
grade.py – Synthetic Corrupted Alignment Signal Recovery.

Metric: Macro F1 across 5 quality tiers (0-4).
Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if "sample_id" not in submission.columns or "quality_tier" not in submission.columns:
            return 0.0
        if "sample_id" not in answers.columns or "quality_tier" not in answers.columns:
            return 0.0

        if submission["sample_id"].duplicated().any():
            return 0.0
        if answers["sample_id"].duplicated().any():
            return 0.0

        merged = answers.merge(submission, on="sample_id", how="left",
                               suffixes=("_true", "_pred"))

        if len(merged) != len(answers):
            return 0.0
        if merged["quality_tier_pred"].isna().any():
            return 0.0

        y_true = merged["quality_tier_true"].astype(int).values
        y_pred = merged["quality_tier_pred"].astype(int).values

        classes = sorted(set(y_true))
        f1_scores = []
        for c in classes:
            tp = int(np.sum((y_pred == c) & (y_true == c)))
            fp = int(np.sum((y_pred == c) & (y_true != c)))
            fn = int(np.sum((y_pred != c) & (y_true == c)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)

        return float(np.mean(f1_scores))

    except Exception:
        return 0.0
```

## Prepare Script

```python
"""
prepare.py – Synthetic Corrupted Alignment Signal Recovery.

Reads raw_data/data.csv.
Produces:
    pub/  train.csv   (sample_id, instruction, response, quality_tier)
          test.csv    (sample_id, instruction, response)
          sample_submission.csv  (sample_id, quality_tier = 2)
    priv/ answers.csv (sample_id, quality_tier)
"""

from pathlib import Path
import pandas as pd


def prepare(raw_dir="raw_data", public_dir="pub", private_dir="priv", seed=42):
    raw = Path(raw_dir)
    pub = Path(public_dir)
    priv = Path(private_dir)
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(raw / "data.csv"))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * 0.75)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    assert len(set(train_df["sample_id"]) & set(test_df["sample_id"])) == 0

    keep_train = ["sample_id", "instruction", "response", "quality_tier"]
    keep_test = ["sample_id", "instruction", "response"]

    train_df[keep_train].to_csv(str(pub / "train.csv"), index=False)
    test_df[keep_test].to_csv(str(pub / "test.csv"), index=False)

    test_df[["sample_id", "quality_tier"]].to_csv(str(priv / "answers.csv"), index=False)

    sub = test_df[["sample_id"]].copy()
    sub["quality_tier"] = 2
    sub.to_csv(str(pub / "sample_submission.csv"), index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"pub/ and priv/ written.")


if __name__ == "__main__":
    prepare()
```
