# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Multi-Observer Incident Report Disagreement Corpus.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Conflicting-Witness Incident Reconstruction
```

---

## 3) Problem Description

# Synthetic Conflicting-Witness Incident Reconstruction

## Overview

This is an **NLP** challenge requiring **structured information extraction from multiple conflicting free-text accounts**. The problem models a scenario where independent observers provide contradictory descriptions of the same incident, and the task is to reconstruct the ground-truth structured report by resolving field-level disagreements across observers — without knowing which observer is reliable on which field.

The dataset contains 8,000 synthetic incidents in a fictional industrial safety domain. Each incident has a ground-truth structured report consisting of 6 categorical fields. For each incident, 4–6 independent observers produced free-text statements describing what they witnessed. All entity names, actions, locations, and other values have been replaced with anonymized codes (e.g., `VA018`, `VL011`, `SRC_02`).

**The core difficulty is disagreement resolution, not information extraction.** Observers have hidden field-specific reliability profiles: the same observer may accurately report the location but misidentify the actor. Different observers disagree on 1–4 fields per incident. Statements are written in varied formats — the same information is expressed differently across observers, preventing simple template-based extraction. Statements also embed decoy entity codes within irrelevant contextual details, so that naive extraction methods pick up codes from red-herring sentences that describe unrelated entities or locations. Approximately 12% of incidents are genuinely ambiguous. In the test set, each incident has significantly fewer observers than in training (two observers are removed, leaving as few as 2 per incident), and ~8% of test statements are injected noise from fabricated observers with plausible-looking but entirely random content.

Your task: for each test incident, given only the observer statements, predict all 6 structured fields of the ground-truth incident report.

**What makes this problem uniquely challenging:**

- **Field-specific observer reliability**: the same observer can be correct about one field and wrong about another. Trusting or distrusting an entire statement is suboptimal — the model must reason at the field level.
- **No majority-rules guarantee**: in some incidents, the minority observer is the only one who was actually at the correct location or saw the correct actor. Simple majority voting across observers per field is insufficient.
- **Anonymized codes**: all values are opaque codes with no semantic content. The model cannot use external world knowledge about what actions are plausible at which locations.
- **Decoy codes in red herrings**: observer statements contain irrelevant contextual sentences that embed valid entity codes from other incidents. Extraction methods that simply collect all codes from a statement will accumulate spurious values that corrupt voting or aggregation.
- **Varied statement formats**: the same information is expressed in 8+ distinct sentence structures across observers. Pattern or template matching trained on one format fails on others.
- **Joint 6-field prediction**: errors in one field may correlate with errors in others, and all 6 fields must be predicted simultaneously.
- **Severe test-time sparsity**: test incidents have 2–4 observers (two removed vs. training's 4–6), and ~8% of test statements are injected fabricated accounts with plausible-looking random codes.
- **Irreducible ambiguity**: ~12% of incidents have genuinely uncertain ground truth, capping theoretical performance.

## Evaluation

Submissions are scored using **average per-field exact-match accuracy** across all 6 structured fields. For each field, the fraction of test incidents where the predicted value exactly matches the ground truth is computed, and the 6 per-field accuracies are averaged. **Higher is better.** Minimum: 0.0, Maximum: 1.0.

## Dataset

- `train_statements.csv` — ~31,900 training observer statements: incident_id (int), witness_idx (int), witness_role (string, coded observer role e.g. SRC_00–SRC_07), statement (string, free-text observer account)
- `test_statements.csv` — ~5,200 test observer statements: same columns, significantly fewer observers per incident than training (2–4 vs. 4–6), plus ~8% injected noise statements from fabricated observers
- `train_labels.csv` — 6,400 rows: incident_id (int), actor (string), action (string), location (string), time_period (string), severity (string), contributing_factor (string)
- `sample_submission.csv` — 1,600 rows: same columns as train_labels.csv, with baseline constant predictions. Shows the required submission format.

### Feature Details

| Column | Type | Description |
|--------|------|-------------|
| incident_id | int | Unique incident identifier |
| witness_idx | int | Observer index within this incident (0-based, 99 for noise) |
| witness_role | string | Coded observer role (SRC_00 through SRC_07) |
| statement | string | Free-text observer account containing coded entity references |
| actor | string | Coded actor identifier (1 of 30, e.g. VA003) |
| action | string | Coded action identifier (1 of 20, e.g. VA007) |
| location | string | Coded location identifier (1 of 15, e.g. VL011) |
| time_period | string | Coded time period identifier (1 of 8, e.g. VT005) |
| severity | string | Coded severity level (1 of 5, e.g. VS002) |
| contributing_factor | string | Coded contributing factor (1 of 12, e.g. VC003) |

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| incident_id | int | Incident identifier from test_statements.csv |
| actor | string | Predicted actor code |
| action | string | Predicted action code |
| location | string | Predicted location code |
| time_period | string | Predicted time period code |
| severity | string | Predicted severity code |
| contributing_factor | string | Predicted contributing factor code |

**Requirements:**
- Must contain exactly 1,600 rows (one per test incident).
- Include a header row.

**Example of a correctly formatted submission file:**

incident_id,actor,action,location,time_period,severity,contributing_factor
8,VA003,VA012,VL005,VT002,VS001,VC007
19,VA018,VA007,VL011,VT005,VS002,VC003
21,VA014,VA001,VL008,VT006,VS004,VC010
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
        Average per-field exact-match accuracy across 6 structured fields.
    """
    try:
        FIELDS = ["actor", "action", "location", "time_period", "severity", "contributing_factor"]

        if "incident_id" not in submission.columns:
            return 0.0
        for f in FIELDS:
            if f not in submission.columns:
                return 0.0

        if submission["incident_id"].duplicated().any():
            return 0.0
        if answers["incident_id"].duplicated().any():
            return 0.0

        merged = answers.merge(
            submission, on="incident_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        if set(submission["incident_id"]) != set(answers["incident_id"]) or len(submission) != len(answers):
            return 0.0

        field_accuracies = []
        for f in FIELDS:
            col_true = f"{f}_true" if f"{f}_true" in merged.columns else f
            col_pred = f"{f}_pred" if f"{f}_pred" in merged.columns else f

            if merged[col_pred].isna().any():
                field_accuracies.append(0.0)
                continue

            true_vals = merged[col_true].astype(str).str.strip()
            pred_vals = merged[col_pred].astype(str).str.strip()
            acc = (true_vals == pred_vals).mean()
            field_accuracies.append(float(acc))

        avg_accuracy = float(np.mean(field_accuracies))
        return avg_accuracy

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

    ground_truth = pd.read_csv(raw / "ground_truth.csv")
    statements = pd.read_csv(raw / "statements.csv")

    rng = _rnd.Random(161803398)
    np_rng = np.random.RandomState(88)

    FIELDS = ["actor", "action", "location", "time_period", "severity", "contributing_factor"]

    field_maps = {}
    for f in FIELDS:
        unique_vals = sorted(ground_truth[f].unique().tolist())
        n = len(unique_vals)
        codes = [f"V{f[0].upper()}{i:03d}" for i in range(n)]
        rng.shuffle(codes)
        mapping = {unique_vals[i]: codes[i] for i in range(n)}
        field_maps[f] = mapping
        ground_truth[f] = ground_truth[f].map(mapping)

    def _obfuscate_statement(text):
        result = text
        for f, mapping in field_maps.items():
            for original, coded in mapping.items():
                result = result.replace(original, coded)
        return result

    statements["statement"] = statements["statement"].apply(_obfuscate_statement)

    unique_roles = sorted(statements["witness_role"].unique().tolist())
    role_codes = [f"SRC_{i:02d}" for i in range(len(unique_roles))]
    rng.shuffle(role_codes)
    role_map = {unique_roles[i]: role_codes[i] for i in range(len(unique_roles))}
    statements["witness_role"] = statements["witness_role"].map(role_map)

    all_ids = sorted(ground_truth["incident_id"].tolist())
    rng_split = _rnd.Random(141421356)
    rng_split.shuffle(all_ids)

    split_idx = int(len(all_ids) * 0.8)
    train_ids = set(all_ids[:split_idx])
    test_ids = set(all_ids[split_idx:])

    train_gt = ground_truth[ground_truth["incident_id"].isin(train_ids)].copy()
    test_gt = ground_truth[ground_truth["incident_id"].isin(test_ids)].copy()
    train_stmts = statements[statements["incident_id"].isin(train_ids)].copy()
    test_stmts = statements[statements["incident_id"].isin(test_ids)].copy()

    test_stmts_rows = []
    for inc_id in sorted(test_ids):
        inc_stmts = test_stmts[test_stmts["incident_id"] == inc_id]
        n_drop = min(2, max(0, len(inc_stmts) - 2))
        if n_drop > 0:
            indices = sorted(inc_stmts.index.tolist())
            rng.shuffle(indices)
            drop_indices = indices[:n_drop]
            inc_stmts = inc_stmts.drop(drop_indices)
        test_stmts_rows.append(inc_stmts)
    test_stmts = pd.concat(test_stmts_rows, ignore_index=True)

    n_noise = int(len(test_stmts) * 0.08)
    noise_rows = []
    for _ in range(n_noise):
        inc_id = rng.choice(sorted(test_ids))
        noise_text = (
            f"{rng.choice(role_codes)} reported seeing "
            f"{rng.choice(list(field_maps['actor'].values()))} "
            f"{rng.choice(list(field_maps['action'].values()))} "
            f"at {rng.choice(list(field_maps['location'].values()))} "
            f"during the {rng.choice(list(field_maps['time_period'].values()))}. "
            f"Severity appeared {rng.choice(list(field_maps['severity'].values()))}. "
            f"Contributing factor was {rng.choice(list(field_maps['contributing_factor'].values()))}."
        )
        noise_rows.append({
            "incident_id": inc_id,
            "witness_idx": 99,
            "witness_role": rng.choice(role_codes),
            "statement": noise_text,
        })
    noise_df = pd.DataFrame(noise_rows)
    test_stmts = pd.concat([test_stmts, noise_df], ignore_index=True)
    test_stmts = test_stmts.sort_values(["incident_id", "witness_idx"]).reset_index(drop=True)

    train_gt = train_gt.sort_values("incident_id").reset_index(drop=True)
    test_gt = test_gt.sort_values("incident_id").reset_index(drop=True)
    train_stmts = train_stmts.sort_values(["incident_id", "witness_idx"]).reset_index(drop=True)

    sample_sub = test_gt[["incident_id"]].copy()
    for f in FIELDS:
        mode_val = train_gt[f].mode().iloc[0]
        sample_sub[f] = mode_val

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_stmts.to_csv(public / "train_statements.csv", index=False)
    train_gt[["incident_id"] + FIELDS].to_csv(public / "train_labels.csv", index=False)
    test_stmts.to_csv(public / "test_statements.csv", index=False)
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    test_gt[["incident_id"] + FIELDS].to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly parses the free-text statement column and extracts coded entity references from observer accounts without data corruption.
- **Rationale:** Statements contain coded identifiers embedded in natural language. Failure to parse them correctly makes prediction impossible.

### Rubric 2
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Handles incidents with varying numbers of observer statements (2–7 per incident in test) without crashing or producing incomplete predictions.
- **Rationale:** Different incidents have different numbers of observers. Fixed-count assumptions will fail on a subset of incidents.

### Rubric 3
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 1,600 rows, columns incident_id plus six prediction fields (actor, action, location, time_period, severity, contributing_factor), where each prediction is a valid coded identifier from the training data.
- **Rationale:** Missing columns, wrong row counts, or invalid codes will score zero.

### Rubric 4
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves average per-field accuracy above the constant-prediction baseline (~0.09) on the test set, producing field-specific predictions rather than a single constant per field.
- **Rationale:** Predicting the most common training value for every field is trivially achievable and demonstrates no useful modeling.

### Rubric 5
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Generates predictions for all 6 structured fields for every test incident, not just a subset of fields.
- **Rationale:** The evaluation averages accuracy across all 6 fields. Leaving any field blank scores zero for that field and drags down the overall score.

### Rubric 6
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Criteria:** Prediction quality does not catastrophically degrade for incidents where observer statements contain strongly conflicting claims about the same field.
- **Rationale:** Disagreement is the central challenge. A robust solution must handle conflicting accounts rather than defaulting to a fixed fallback.

### Rubric 7
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Evaluates intermediate predictions on a held-out portion of training data before generating final test predictions.
- **Rationale:** Without internal validation, there is no way to assess whether the approach is working before submission.

### Rubric 8
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not access test set ground-truth labels or leak private answer data into the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces inflated scores that do not reflect genuine capability.
