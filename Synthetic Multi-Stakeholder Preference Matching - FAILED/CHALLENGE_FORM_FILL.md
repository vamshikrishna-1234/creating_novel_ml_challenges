# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Dual-Profile Engagement Selection Logs.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Multi-Stakeholder Preference Matching
```

---

## 3) Problem Description

# Synthetic Multi-Stakeholder Preference Matching

## Overview

This challenge addresses a **dual-profile selection prediction** problem in a fictional service engagement platform. In each engagement session, a requestor is presented with a pool of 6–14 candidate profiles and selects exactly one. The selection depends on a hidden compatibility function that jointly considers requestor-side preference signals and candidate-side attribute signals — neither profile alone is sufficient to predict the outcome.

All feature identifiers are anonymized and carry no semantic meaning. Interaction-level signals (engagement intensity, dwell duration, browse depth) have been session-normalized, removing absolute scale information. Several features in the candidate, requestor, and session tables are uninformative decoys injected during data preparation. Approximately 1.4% of test requestors have no entry in the requestor profile table, requiring models to rely solely on within-session evidence for those sessions.

The fundamental challenge is **learning a latent bilateral compatibility function** from anonymized, transformed signals. Standard single-profile approaches (pure collaborative filtering, pure content-based) are insufficient because the selection depends on how requestor preferences interact with candidate attributes, not on either in isolation. The interaction signals provide partial evidence but have been normalized per-session, removing obvious shortcuts.

## Evaluation

Submissions are scored using **accuracy** — the fraction of test sessions where the predicted selected candidate matches the ground truth. **Higher is better.** Minimum: 0.0, Maximum: 1.0, Random baseline: ~0.10 (approximately 10 candidates per session).

## Dataset

- `train_sessions.csv` — 119,951 training interaction rows across 12,000 sessions: session_id (int), requestor_id (int), candidate_id (int), plus 8 anonymized signal columns (s_01 through s_08, mix of float and int)
- `test_sessions.csv` — 30,309 test interaction rows across 3,000 sessions: same columns as train_sessions.csv
- `train_labels.csv` — 12,000 rows: session_id (int), selected_candidate_id (int)
- `candidates.csv` — 500 candidates: candidate_id (int), category (string, one of 20 group codes G_00–G_19), plus 12 anonymized numeric attribute columns (a_00 through a_10 and additional codes, float)
- `requestors.csv` — ~2,957 requestors (some test requestors intentionally excluded): requestor_id (int), pg_1/pg_2/pg_3 (string, group codes representing preference categories), plus 11 anonymized numeric columns (r_00 through r_10, float)
- `sample_submission.csv` — 3,000 rows: session_id (int), selected_candidate_id (int, baseline constant prediction). Shows the required submission format.

| Column | Type | Description |
|--------|------|-------------|
| session_id | int | Unique engagement session identifier |
| requestor_id | int | Requestor conducting this session |
| candidate_id | int | Candidate profile viewed |
| s_01 | float | Per-session normalized engagement signal |
| s_02 | float | Per-session normalized dwell signal |
| s_03 | float | Per-session normalized browse depth signal |
| s_04 | int | Binary interaction flag |
| s_05 | int | Display ordering index within the session |
| s_06 | float | Auxiliary session measurement |
| s_07 | float | Auxiliary session measurement |
| s_08 | int | Auxiliary session category code |

## Submission

Submit a CSV file with exactly 3,000 rows (one per test session), a header row, and two columns:

| Column | Type | Description |
|--------|------|-------------|
| session_id | int | Session identifier from test_sessions.csv |
| selected_candidate_id | int | Predicted selected candidate_id (must be a valid candidate_id from candidates.csv) |

**Example of a correctly formatted submission file:**

session_id,selected_candidate_id
12000,42
12001,305
12002,127
12003,88
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
    ans_cols = answers[["session_id", "selected_candidate_id"]].copy()
    sub_cols = submission[["session_id", "selected_candidate_id"]].copy()
    merged = ans_cols.merge(
        sub_cols, on="session_id", suffixes=("_true", "_pred")
    )

    if len(merged) == 0:
        raise ValueError("No common session_ids between submission and answers")

    if merged["selected_candidate_id_pred"].isna().any():
        raise ValueError("Submission has missing predictions for some sessions")

    accuracy = (
        merged["selected_candidate_id_true"]
        == merged["selected_candidate_id_pred"]
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

    interactions = pd.read_csv(raw / "interactions.csv")
    candidates = pd.read_csv(raw / "candidates.csv")
    requestors = pd.read_csv(raw / "requestors.csv")
    selections = pd.read_csv(raw / "selections.csv")

    rng = _rnd.Random(987654321)
    np_rng = np.random.RandomState(99)

    all_sessions = sorted(selections["session_id"].tolist())
    rng_split = _rnd.Random(1122334455)
    rng_split.shuffle(all_sessions)

    split_idx = int(len(all_sessions) * 0.8)
    train_sids = set(all_sessions[:split_idx])
    test_sids = set(all_sessions[split_idx:])

    train_rids = set(
        interactions[interactions["session_id"].isin(train_sids)]["requestor_id"]
    )
    test_only_rids = set(
        interactions[interactions["session_id"].isin(test_sids)]["requestor_id"]
    ) - train_rids

    cold_start_rids = set()
    test_only_list = sorted(test_only_rids)
    for rid in test_only_list:
        cold_start_rids.add(rid)
        if len(cold_start_rids) >= int(len(requestors) * 0.12):
            break

    orig_cats = sorted(candidates["category"].unique().tolist())
    shuffled_cats = [f"G_{i:02d}" for i in range(len(orig_cats))]
    rng.shuffle(shuffled_cats)
    cat_map = {orig: shuffled_cats[i] for i, orig in enumerate(orig_cats)}

    candidates["category"] = candidates["category"].map(cat_map)
    for col in ["pref_cat_1", "pref_cat_2", "pref_cat_3"]:
        requestors[col] = requestors[col].map(cat_map)

    cand_feat_map = {}
    cand_orig = [c for c in candidates.columns if c.startswith("cf_")]
    cand_labels = [f"a_{i:02d}" for i in range(len(cand_orig) + 5)]
    rng.shuffle(cand_labels)
    for i, col in enumerate(sorted(cand_orig)):
        cand_feat_map[col] = cand_labels[i]
    candidates = candidates.rename(columns=cand_feat_map)
    candidates = candidates.rename(columns={"popularity": cand_labels[len(cand_orig)]})

    req_feat_map = {}
    req_orig = [c for c in requestors.columns if c.startswith("rw_")]
    req_labels = [f"r_{i:02d}" for i in range(len(req_orig) + 5)]
    rng.shuffle(req_labels)
    for i, col in enumerate(sorted(req_orig)):
        req_feat_map[col] = req_labels[i]
    requestors = requestors.rename(columns=req_feat_map)
    requestors = requestors.rename(columns={"cat_strength": req_labels[len(req_orig)]})

    int_feat_map = {
        "engagement": "s_01",
        "dwell_time": "s_02",
        "browse_depth": "s_03",
        "revisit": "s_04",
        "position": "s_05",
    }
    interactions = interactions.rename(columns=int_feat_map)

    requestors = requestors.rename(columns={
        "pref_cat_1": "pg_1", "pref_cat_2": "pg_2", "pref_cat_3": "pg_3",
    })

    n_cand = len(candidates)
    for j in range(len(cand_orig) + 1, len(cand_labels)):
        candidates[cand_labels[j]] = np_rng.normal(0, 1, n_cand).round(3)

    n_req = len(requestors)
    for j in range(len(req_orig) + 1, len(req_labels)):
        requestors[req_labels[j]] = np_rng.normal(0, 1, n_req).round(3)

    n_int = len(interactions)
    interactions["s_06"] = np_rng.uniform(0, 10, n_int).round(2)
    interactions["s_07"] = np_rng.exponential(1.5, n_int).round(3)
    interactions["s_08"] = np_rng.randint(0, 4, n_int)

    for col in ["s_01", "s_02", "s_03"]:
        group_mean = interactions.groupby("session_id")[col].transform("mean")
        group_std = interactions.groupby("session_id")[col].transform("std")
        group_std = group_std.replace(0, 1)
        interactions[col] = ((interactions[col] - group_mean) / group_std).round(4)

    train_int = interactions[interactions["session_id"].isin(train_sids)].copy()
    test_int = interactions[interactions["session_id"].isin(test_sids)].copy()
    train_int = train_int.sort_values("session_id").reset_index(drop=True)
    test_int = test_int.sort_values("session_id").reset_index(drop=True)

    train_sel = selections[selections["session_id"].isin(train_sids)].copy()
    test_sel = selections[selections["session_id"].isin(test_sids)].copy()
    train_sel = train_sel.sort_values("session_id").reset_index(drop=True)
    test_sel = test_sel.sort_values("session_id").reset_index(drop=True)

    requestors_pub = requestors[~requestors["requestor_id"].isin(cold_start_rids)].copy()
    requestors_pub = requestors_pub.reset_index(drop=True)

    mode_cand = int(train_sel["selected_candidate_id"].mode().iloc[0])
    sample_sub = test_sel[["session_id"]].copy()
    sample_sub["selected_candidate_id"] = mode_cand

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_int.to_csv(public / "train_sessions.csv", index=False)
    test_int.to_csv(public / "test_sessions.csv", index=False)
    train_sel.to_csv(public / "train_labels.csv", index=False)
    candidates.to_csv(public / "candidates.csv", index=False)
    requestors_pub.to_csv(public / "requestors.csv", index=False)
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    test_sel.to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly joins session interaction rows with candidate and requestor profile tables, handling the variable number of candidates per session (6–14 rows) without data loss or duplication.
- **Rationale:** The data spans multiple tables with one-to-many relationships. Incorrect joins corrupt the feature space and make downstream modeling unreliable.

### Rubric 2
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Handles sessions where the requestor_id has no entry in requestors.csv without crashing or dropping those sessions from the submission.
- **Rationale:** A subset of test requestors are intentionally absent from the profile table. Solutions that crash or skip these sessions will have incomplete submissions.

### Rubric 3
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 3,000 rows, columns session_id and selected_candidate_id, where each selected_candidate_id is a valid candidate from the session's candidate pool.
- **Rationale:** An incorrectly formatted or incomplete submission will fail the grader or score zero.

### Rubric 4
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves accuracy meaningfully above random baseline (~0.10) on the test set.
- **Rationale:** A solution scoring at or near 0.10 indicates no learning beyond random candidate selection from the pool.

### Rubric 5
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Evaluates intermediate predictions on a validation split before generating the final submission, and iterates on the approach if initial accuracy is near baseline.
- **Rationale:** Good engineering practice involves measuring progress on held-out data and adjusting strategy when results are poor.

### Rubric 6
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not use test set labels or leak information from the private answers into the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces artificially inflated scores that do not reflect genuine model capability.
