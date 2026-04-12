# Challenge creation form — fill-in

## Title

```
Synthetic Contextual Sequence Transduction
```

## Problem Description

# Synthetic Contextual Sequence Transduction

## Overview

Sequence transduction — mapping one token sequence to another through learned rules — is a fundamental capability in language modeling. When the transformation rules are hidden, interacting, and context-dependent, recovering them from examples alone becomes a challenging induction problem.

This challenge presents a synthetic transformation system where a base token and five categorical control codes are mapped to an output character sequence. Crucially, the transformation is **context-dependent**: a preceding context token modifies how multiple transformation steps are applied. The system's rules are entirely hidden — the solver must induce them from 22,500 training examples.

The total input space contains 360,000 possible combinations (500 roots × 720 feature combinations), but only 30,000 samples exist (8.3% coverage). The model must generalize from sparse training data to unseen control code combinations.

The hidden transformation system involves 15+ interacting rules:
- **Progressive two-class harmony**: output suffix variants are selected based on the last vowel character in the current form, propagating through the entire sequence
- **Boundary assimilation**: voicing agreement between consonants is enforced at every concatenation point
- **Hiatus resolution**: adjacent vowels at boundaries trigger deletion of the first
- **Cluster breaking**: an epenthetic character is inserted to prevent 3+ consecutive consonants
- **First-layer sandhi**: the context token's final character triggers one of three modifications (lenition, nasal augmentation, or fortition) to the first appended suffix — affecting ~47% of samples
- **Second-layer sandhi**: the context token's initial character and the 5th control code jointly determine a separate modification to the voice suffix (consonant prepending, nasalization, devoicing, or no effect)
- **Voice allomorphy**: the 5th control code's suffix changes form depending on the 1st control code, creating 6 cross-position allomorphs that cannot be predicted from individual codes
- **Irregular base tokens**: 60 of 500 base tokens undergo control-code-dependent stem modifications
- **Rotational substitution**: 40 base tokens undergo full character rotation under a specific control-code pair
- **Double-irregular tokens**: 30 base tokens undergo voice-specific stem alterations (prefixing or ablaut)
- **Suppletive tokens**: 15 base tokens have completely different stems under specific control-code-pair combinations — requiring memorization, not rule application
- **Three-way portmanteau**: 3 control-code triples produce fused outputs replacing individual suffixes (active voice only)
- **Four-way portmanteau**: 5 control-code quadruples produce fused outputs replacing tense+number+case+voice

**Training mandate**: You must train a sequence model from scratch. Pre-trained weights from any language model or foundation model are prohibited. The model must be under 50 million total parameters.

Approximately 10% of training labels contain single-character noise, establishing an irreducible error floor.

## Evaluation

Submissions are scored using **per-character positional accuracy** across all test samples.

For each sample, the predicted and true target strings are compared character-by-character from the left. Missing positions (shorter prediction) and extra positions (longer prediction) both count as incorrect.

```
Accuracy = Total correct character positions / Total character positions across all samples
```

where total character positions = sum of max(len(predicted_i), len(true_i)) for each sample.

Higher is better. The placeholder baseline ("aaaaaa" for all samples) scores approximately 0.03. Copying the base token scores approximately 0.35. The theoretical maximum is approximately 0.90 due to label noise.

## Dataset

After preparation, the public directory contains:

| File                   | Description                                              |
|------------------------|----------------------------------------------------------|
| `train.csv`            | 22,500 labeled pairs with columns below                  |
| `test.csv`             | 7,500 unlabeled pairs (same columns minus `target`)      |
| `sample_submission.csv`| Example submission with all targets set to "aaaaaa"      |

Training and test columns:

| Column    | Type | Description                                                                                  |
|-----------|------|----------------------------------------------------------------------------------------------|
| sample_id | int  | Unique identifier                                                                            |
| source    | str  | Space-separated: context token, base token, then five categorical control codes              |
| target    | str  | Transformed output sequence (4–24 characters) — **training only**                            |

The source string format is: `<context> <base> <CODE1> <CODE2> <CODE3> <CODE4> <CODE5>`

Example: `ve dibnasle PRF SG GEN SBJ MID`

Control code values:
- Position 1: PRS, PST, FUT, PRF, HAB
- Position 2: SG, DU, PL
- Position 3: NOM, ACC, DAT, GEN
- Position 4: IND, SBJ, IMP
- Position 5: ACT, MID, PAS, CAU

## Submission

Submit a CSV file with the following format:

| Column    | Type | Description                          |
|-----------|------|--------------------------------------|
| sample_id | int  | Identifier from the test set         |
| target    | str  | Predicted output sequence            |

**Requirements:**
- Must contain exactly 7,500 rows (one per test sample)
- Must include a header row
- All `sample_id` values must be unique and match the test set
- `target` values must be non-empty strings

## Rubrics

### Rubric 1
- **Criteria:** All model weights must be randomly initialized. Pre-trained weights from any general-purpose language model, sequence model, or foundation model are prohibited.
- **Type:** MODELING
- **Importance:** REQUIRED
- **Rationale:** The challenge tests whether the agent can induce transformation rules from the provided training data alone; pre-trained models would bypass the core learning task entirely.

### Rubric 2
- **Criteria:** Produces a valid submission CSV with columns `sample_id` and `target`, correct row count, unique IDs matching the test set, and non-empty string predictions.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** Malformed submissions receive a score of 0.0 from the grader; structural correctness is a prerequisite for any evaluation.

### Rubric 3
- **Criteria:** Achieves per-character accuracy exceeding 0.15 on the test set (above the random baseline of ~0.04).
- **Type:** MODELING
- **Importance:** REQUIRED
- **Rationale:** Scoring above the random baseline confirms the model has learned at least some transformation structure rather than producing random outputs.

### Rubric 4
- **Criteria:** Training architecture is constrained under 50 million total parameters.
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Rationale:** Enforces efficient architecture design rather than brute-force memorization.

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
  metric: "char_accuracy"
  direction: "maximize"
  min_score: 0.0
  max_score: 1.0
```

## Grading Script

```python
"""
grade.py — Synthetic Contextual Sequence Transduction

Metric: Per-character positional accuracy across all samples.
For each sample, characters are compared position-by-position.
Missing positions (shorter prediction) count as incorrect.
Extra positions (longer prediction) count as incorrect.

Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if "sample_id" not in submission.columns or "target" not in submission.columns:
            return 0.0
        if "sample_id" not in answers.columns or "target" not in answers.columns:
            return 0.0

        if submission["sample_id"].duplicated().any():
            return 0.0
        if answers["sample_id"].duplicated().any():
            return 0.0

        merged = answers.merge(
            submission, on="sample_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) != len(answers):
            return 0.0
        if merged["target_pred"].isna().any():
            return 0.0

        true_vals = merged["target_true"].astype(str).values
        pred_vals = merged["target_pred"].astype(str).values

        total_correct = 0
        total_positions = 0

        for t, p in zip(true_vals, pred_vals):
            max_len = max(len(t), len(p))
            if max_len == 0:
                continue
            correct = sum(
                1 for i in range(min(len(t), len(p))) if t[i] == p[i]
            )
            total_correct += correct
            total_positions += max_len

        if total_positions == 0:
            return 0.0

        return float(total_correct / total_positions)

    except Exception:
        return 0.0
```

## Prepare Script

```python
"""
prepare.py — Synthetic Contextual Sequence Transduction

Reads raw_data/data.csv.
Produces:
    pub/  train.csv   (sample_id, source, target)
          test.csv    (sample_id, source)
          sample_submission.csv  (sample_id, target = "aaaaaa")
    priv/ answers.csv (sample_id, target)
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

    train_df[["sample_id", "source", "target"]].to_csv(
        str(pub / "train.csv"), index=False
    )
    test_df[["sample_id", "source"]].to_csv(
        str(pub / "test.csv"), index=False
    )

    test_df[["sample_id", "target"]].to_csv(
        str(priv / "answers.csv"), index=False
    )

    sub = test_df[["sample_id"]].copy()
    sub["target"] = "aaaaaa"
    sub.to_csv(str(pub / "sample_submission.csv"), index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"pub/ and priv/ written.")


if __name__ == "__main__":
    prepare()
```
