# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Contextual Item Selection Logs with Preference Dynamics.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Context-Driven Slate Selection
```

---

## 3) Problem Description

# Synthetic Context-Driven Slate Selection

## Overview

This challenge requires **predicting which item a user will select from a personalized candidate slate**, given the user's profile, item attributes, and contextual conditions at the time of the query. Each query presents a user with a slate of 10 candidate items under specific contextual conditions (6 anonymized context features). The task is to predict which one of the 10 candidates the user selects.

The dataset contains 400 anonymized users, 800 anonymized items, and 20,000 selection queries split into 15,000 training queries and 5,000 test queries. Users are described by 7 anonymized profile features (`UF_00` through `UF_06`), items by 10 anonymized attribute features (`IF_00` through `IF_09`), and each query includes 6 anonymized context features (`C_00` through `C_05`). All original feature names have been removed.

**What makes this problem challenging:**

- **Context-dependent preferences**: the same user may prefer entirely different items depending on the contextual conditions of the query. User preferences are not static — they shift across different context combinations. Identifying which context features drive preference shifts, and how, is central to the task.
- **Slate-specific choices**: each query has a unique slate of 10 candidates. The model must reason about relative item appeal within each specific candidate set, not just global item popularity.
- **Mixed entity features**: user profiles, item attributes, and query context must all be combined. Some features within each entity carry signal; others do not.
- **Noisy selections**: approximately 12% of selections are not preference-driven, capping theoretical accuracy at approximately 0.88.

Your task: for each of the 5,000 test queries, predict the `chosen_item_id` (one of the 10 candidate items listed for that query).

## Evaluation

Submissions are scored using **accuracy** (fraction of correctly predicted selected item IDs). **Higher is better.** Minimum: 0.0, Maximum: 1.0.

## Dataset

- `users.csv` — 400 user profiles: user_id (int), UF_00 through UF_06 (7 features, mix of float and int)
- `items.csv` — 800 item descriptions: item_id (int), IF_00 through IF_09 (10 features, mix of float and int)
- `train_queries.csv` — 15,000 training queries: query_id (int), user_id (int), C_00 through C_05 (6 context features, int)
- `train_candidates.csv` — 150,000 rows (10 per query): query_id (int), item_id (int) — lists the 10 candidate items for each training query
- `train_labels.csv` — 15,000 rows: query_id (int), chosen_item_id (int) — the item selected in each training query
- `test_queries.csv` — 5,000 test queries: query_id (int), user_id (int), C_00 through C_05 (same structure as train)
- `test_candidates.csv` — 50,000 rows (10 per query): query_id (int), item_id (int) — the 10 candidates for each test query
- `sample_submission.csv` — 5,000 rows: query_id (int), chosen_item_id (int, placeholder 0). Shows the required format.

### Feature Details

**User features (users.csv)**

| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier (0–399) |
| UF_00 | float | Anonymized user feature |
| UF_01 | int | Anonymized user feature (6 levels: 0–5) |
| UF_02 | float | Anonymized user feature |
| UF_03 | int | Anonymized user feature (4 levels: 0–3) |
| UF_04 | float | Anonymized user feature |
| UF_05 | float | Anonymized user feature |
| UF_06 | float | Anonymized user feature |

**Item features (items.csv)**

| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Unique item identifier (0–799) |
| IF_00 | float | Anonymized item feature |
| IF_01 | float | Anonymized item feature |
| IF_02 | float | Anonymized item feature |
| IF_03 | int | Anonymized item feature (5 levels: 0–4) |
| IF_04 | int | Anonymized item feature (18 levels: 0–17) |
| IF_05 | float | Anonymized item feature |
| IF_06 | float | Anonymized item feature |
| IF_07 | float | Anonymized item feature |
| IF_08 | float | Anonymized item feature |
| IF_09 | float | Anonymized item feature |

**Context features (train_queries.csv / test_queries.csv)**

| Column | Type | Description |
|--------|------|-------------|
| query_id | int | Unique query identifier |
| user_id | int | User presented with the slate |
| C_00 | int | Anonymized context feature (6 levels: 0–5) |
| C_01 | int | Anonymized context feature (4 levels: 0–3) |
| C_02 | int | Anonymized context feature (3 levels: 0–2) |
| C_03 | int | Anonymized context feature (10 levels: 0–9) |
| C_04 | int | Anonymized context feature (120 levels: 1–120) |
| C_05 | int | Anonymized context feature (5 levels: 0–4) |

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| query_id | int | Query identifier from test_queries.csv |
| chosen_item_id | int | Predicted item_id selected by the user (must be one of the 10 candidates for that query) |

**Requirements:**
- Must contain exactly 5,000 rows (one per test query).
- Include a header row.
- Each `chosen_item_id` must be a valid `item_id` from the corresponding query's candidate slate in test_candidates.csv.

**Example of a correctly formatted submission file:**

```
query_id,chosen_item_id
3,142
8,505
14,27
19,688
24,311
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
        A float score in [0.0, 1.0]. Higher is better (accuracy).
        Returns 0.0 for invalid or malformed submissions.
    """
    try:
        if "query_id" not in submission.columns or "chosen_item_id" not in submission.columns:
            return 0.0

        if submission["query_id"].duplicated().any():
            return 0.0
        if answers["query_id"].duplicated().any():
            return 0.0

        if (
            set(submission["query_id"]) != set(answers["query_id"])
            or len(submission) != len(answers)
        ):
            return 0.0

        merged = answers.merge(
            submission, on="query_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        col_true = "chosen_item_id_true" if "chosen_item_id_true" in merged.columns else "chosen_item_id"
        col_pred = "chosen_item_id_pred" if "chosen_item_id_pred" in merged.columns else "chosen_item_id"

        if merged[col_pred].isna().any():
            return 0.0

        accuracy = float((merged[col_true] == merged[col_pred]).mean())
        if np.isnan(accuracy):
            return 0.0

        return accuracy

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

    users = pd.read_csv(raw / "users.csv")
    items = pd.read_csv(raw / "items.csv")
    queries = pd.read_csv(raw / "queries.csv")

    SPLIT_SEED = 271828
    TRAIN_FRAC = 0.75

    np_rng = np.random.RandomState(SPLIT_SEED)

    # ---- Anonymize user columns ----
    u_cols = [c for c in users.columns if c != "user_id"]
    rng_u = _rnd.Random(314159)
    shuf_u = list(u_cols)
    rng_u.shuffle(shuf_u)
    u_map = {orig: f"UF_{i:02d}" for i, orig in enumerate(shuf_u)}
    users = users.rename(columns=u_map)
    users["UF_06"] = np_rng.normal(0, 1, len(users)).round(3)

    # ---- Anonymize item columns ----
    i_cols = [c for c in items.columns if c != "item_id"]
    rng_i = _rnd.Random(161803)
    shuf_i = list(i_cols)
    rng_i.shuffle(shuf_i)
    i_map = {orig: f"IF_{i:02d}" for i, orig in enumerate(shuf_i)}
    items = items.rename(columns=i_map)
    items["IF_08"] = np_rng.normal(0, 1, len(items)).round(3)
    items["IF_09"] = np_rng.uniform(0, 1, len(items)).round(3)

    # ---- Anonymize context columns ----
    ctx_orig = ["time_slot", "device_code", "day_type",
                "referral_code", "session_length", "entry_point"]
    rng_c = _rnd.Random(141421)
    shuf_c = list(ctx_orig)
    rng_c.shuffle(shuf_c)
    c_map = {orig: f"C_{i:02d}" for i, orig in enumerate(shuf_c)}
    queries = queries.rename(columns=c_map)

    # ---- Split queries 75/25 ----
    all_qids = sorted(queries["query_id"].tolist())
    rng_split = _rnd.Random(SPLIT_SEED)
    rng_split.shuffle(all_qids)
    split_pt = int(len(all_qids) * TRAIN_FRAC)
    train_qids = set(all_qids[:split_pt])
    test_qids = set(all_qids[split_pt:])

    train_q = queries[queries["query_id"].isin(train_qids)].copy()
    test_q = queries[queries["query_id"].isin(test_qids)].copy()
    train_q = train_q.sort_values("query_id").reset_index(drop=True)
    test_q = test_q.sort_values("query_id").reset_index(drop=True)

    # ---- Expand candidates into separate table ----
    def _expand(df):
        exp = df[["query_id", "candidates"]].copy()
        exp["candidates"] = exp["candidates"].astype(str).str.split("|")
        exp = exp.explode("candidates").rename(columns={"candidates": "item_id"})
        exp["item_id"] = exp["item_id"].astype(int)
        return exp.reset_index(drop=True)

    train_cands = _expand(train_q)
    test_cands = _expand(test_q)

    ctx_new = sorted(c_map.values())

    # ---- Write ----
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    users.to_csv(public / "users.csv", index=False)
    items.to_csv(public / "items.csv", index=False)

    train_q[["query_id", "user_id"] + ctx_new].to_csv(
        public / "train_queries.csv", index=False
    )
    train_cands.to_csv(public / "train_candidates.csv", index=False)
    train_q[["query_id", "chosen_item_id"]].to_csv(
        public / "train_labels.csv", index=False
    )

    test_q[["query_id", "user_id"] + ctx_new].to_csv(
        public / "test_queries.csv", index=False
    )
    test_cands.to_csv(public / "test_candidates.csv", index=False)

    sample = test_q[["query_id"]].copy()
    sample["chosen_item_id"] = 0
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_q[["query_id", "chosen_item_id"]].to_csv(
        private / "answers.csv", index=False
    )

    print(f"Train queries: {len(train_q)}")
    print(f"Test queries:  {len(test_q)}")
    print(f"Train candidates rows: {len(train_cands)}")
    print(f"Test candidates rows:  {len(test_cands)}")
    print(f"Users: {len(users)}, Items: {len(items)}")
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly loads and joins the multi-file dataset (user profiles, item attributes, query contexts, candidate slates, and training labels) without data loss or type errors.
- **Rationale:** The challenge requires combining information across 5+ CSV files with different schemas and join keys. Failure to parse or join any file correctly makes meaningful modeling impossible.

### Rubric 2
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 5,000 rows containing columns query_id and chosen_item_id, where each chosen_item_id is a valid item_id from the candidate slate of the corresponding query.
- **Rationale:** Missing columns, wrong row counts, or item IDs not in the candidate slate will score zero or produce meaningless accuracy.

### Rubric 3
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves accuracy above the random-from-slate baseline (~0.10) on the test set, producing query-specific item selections rather than a constant or random choice.
- **Rationale:** Random selection from each slate of 10 candidates yields ~10% accuracy and demonstrates no useful learning from the training data.

### Rubric 4
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Criteria:** Incorporates both user profile features and item attribute features when scoring candidate items, rather than relying solely on global item popularity.
- **Rationale:** Global item popularity achieves ~0.31 accuracy. Modeling user-item compatibility is necessary to exceed this ceiling.

### Rubric 5
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not access private answer data or test set ground-truth chosen_item_id values during the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces inflated scores that do not reflect genuine capability.
