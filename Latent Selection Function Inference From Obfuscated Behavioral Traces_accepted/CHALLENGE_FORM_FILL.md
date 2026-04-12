# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Session Behavioral Interaction Signal Logs.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Latent Selection Function Inference From Obfuscated Behavioral Traces
```

---

## 3) Problem Description

# Latent Selection Function Inference From Obfuscated Behavioral Traces

## Overview

This challenge requires **inferring a latent selection function from obfuscated multi-signal behavioral traces**. Each data record captures a user's interaction with an item during a session, but all signal names are anonymized, behavioral measurements have been statistically transformed (per-session z-scored, globally quantile-binned), and noise interactions have been injected. The task is to predict the **outcome of a hidden decision process** — which item was selected — given only the transformed signals, anonymized user profiles, and an obfuscated item catalog.

The core difficulty lies in three interacting factors that, to our knowledge, have not been combined in a single prediction task before:

1. **Dual-regime decision process** — Approximately 20% of sessions follow a "passive interest" regime where the selected item is determined entirely by latent user-item feature compatibility (not by behavioral engagement signals). The remaining ~80% follow an "active engagement" regime where behavioral signals dominate. Participants must discover and model both regimes without explicit regime labels — a latent mixture that cannot be solved by any single-strategy model.

2. **Adversarial signal obfuscation** — All feature names are anonymous codes carrying no semantic meaning. Two behavioral signals have been per-session z-score normalized (destroying absolute scale) and globally quantile-binned (discretizing continuous values). User and item features have been independently noised and renamed. Additionally, ~1–3 noise interactions are injected per session, revisit sequences are merged for ~25% of sessions, and sequential positions are perturbed for ~20% of sessions.

3. **Cold-start inference** — ~15% of test users have no profile in the user table, requiring the model to infer the decision purely from within-session behavioral evidence for those cases.

Unlike standard prediction tasks where feature semantics are known, this challenge requires jointly solving feature interpretation, regime discovery, and outcome prediction under adversarial obfuscation.

## Evaluation

Submissions are scored using **accuracy** (fraction of correctly predicted selected item IDs). **Higher is better.** Minimum: 0.0, Maximum: 1.0, Random baseline: ~0.10.

## Dataset

- `train_sessions.csv` — 12,000 training sessions (~210K rows): session_id (int), user_id (int), item_id (int), event_type (string, one of four interaction event codes), signal_1 (float, per-session normalized behavioral signal, zero-mean), signal_2 (int, binned behavioral signal 1–5), seq_pos (int, position in session, may be perturbed)
- `train_labels.csv` — 12,000 rows: session_id (int), purchased_item_id (int)
- `test_sessions.csv` — 3,000 test sessions (~53K rows): same columns as train_sessions.csv
- `items.csv` — 500 items: item_id (int), cat_code (string, anonymized category CAT_01–CAT_20), tier_code (string, anonymized tier PT_V/W/X/Y/Z), if_1/if_2/if_3 (float, anonymized numeric features)
- `users.csv` — ~2,700 users (some test users intentionally excluded): user_id (int), uf_1/uf_2/uf_3 (string, anonymized categorical features), uf_4/uf_5 (string, anonymized categorical features), uf_6/uf_7/uf_8 (float, anonymized continuous features)
- `sample_submission.csv` — 3,000 rows: session_id (int), purchased_item_id (int, placeholder value 0). Shows the required submission format with one row per test session.

## Submission

Submit a CSV file with exactly 3,000 rows (one per test session), a header row, and two columns:

| Column | Type | Description |
|--------|------|-------------|
| session_id | int | Session identifier from test_sessions.csv |
| purchased_item_id | int | Predicted purchased item_id (must be a valid item_id from items.csv) |

**Example of a correctly formatted submission file:**

session_id,purchased_item_id
3,42
8,305
14,127
19,88
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
    try:
        if "session_id" not in submission.columns or "purchased_item_id" not in submission.columns:
            raise ValueError("Submission must have columns: session_id, purchased_item_id")

        if submission["session_id"].duplicated().any():
            raise ValueError("Submission contains duplicate session_id values")

        if answers["session_id"].duplicated().any():
            raise ValueError("Answers contains duplicate session_id values")

        if len(submission) != len(answers):
            raise ValueError(
                f"Submission must have exactly {len(answers)} rows, got {len(submission)}"
            )

        sub_ids = set(submission["session_id"])
        ans_ids = set(answers["session_id"])
        if sub_ids != ans_ids:
            missing = ans_ids - sub_ids
            extra = sub_ids - ans_ids
            if missing:
                raise ValueError(f"Submission missing session_ids: {len(missing)}")
            if extra:
                raise ValueError(f"Submission has extra session_ids: {len(extra)}")

        merged = answers.merge(
            submission, on="session_id", how="left",
            suffixes=("_true", "_pred")
        )

        if merged["purchased_item_id_pred"].isna().any():
            raise ValueError("Submission has missing predictions after merge")

        accuracy = (
            merged["purchased_item_id_true"] == merged["purchased_item_id_pred"]
        ).mean()

        score = float(accuracy)
        if np.isnan(score):
            return 0.0
        return score

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Prepare Script

```python
from pathlib import Path
import hashlib
import random as _rnd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _det_seed(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def prepare(raw: Path, public: Path, private: Path) -> None:
    interactions = pd.read_csv(raw / "interactions.csv")
    items = pd.read_csv(raw / "items.csv")
    users = pd.read_csv(raw / "users.csv")
    purchases = pd.read_csv(raw / "purchases.csv")

    rng = _rnd.Random(_det_seed("prepare_42"))
    np_rng = np.random.RandomState(42)

    cat_ids = sorted(items["category"].unique())
    shuffled_cats = list(cat_ids)
    rng.shuffle(shuffled_cats)
    cat_map = {orig: f"CAT_{i:02d}" for i, orig in enumerate(shuffled_cats)}

    tier_ids = sorted(items["price_tier"].unique())
    tier_labels = ["PT_W", "PT_X", "PT_Y", "PT_Z", "PT_V"]
    rng.shuffle(tier_labels)
    tier_map = {orig: tier_labels[i] for i, orig in enumerate(tier_ids)}

    items["category"] = items["category"].map(cat_map)
    items["price_tier"] = items["price_tier"].map(tier_map)
    items["attr_1"] += np_rng.normal(0, 0.3, len(items))
    items["attr_2"] += np_rng.normal(0, 0.3, len(items))
    items["attr_3"] += np_rng.normal(0, 0.3, len(items))
    items = items.rename(columns={
        "attr_1": "if_1", "attr_2": "if_2", "attr_3": "if_3",
        "category": "cat_code", "price_tier": "tier_code",
    })
    items = items.round({"if_1": 3, "if_2": 3, "if_3": 3})

    users["pref_cat_1"] = users["pref_cat_1"].map(cat_map)
    users["pref_cat_2"] = users["pref_cat_2"].map(cat_map)
    users["pref_cat_3"] = users["pref_cat_3"].map(cat_map)
    users["tier_low"] = users["tier_low"].map(tier_map)
    users["tier_high"] = users["tier_high"].map(tier_map)
    users["pref_1"] += np_rng.normal(0, 0.4, len(users))
    users["pref_2"] += np_rng.normal(0, 0.4, len(users))
    users["pref_3"] += np_rng.normal(0, 0.4, len(users))
    users = users.rename(columns={
        "pref_cat_1": "uf_1", "pref_cat_2": "uf_2", "pref_cat_3": "uf_3",
        "tier_low": "uf_4", "tier_high": "uf_5",
        "pref_1": "uf_6", "pref_2": "uf_7", "pref_3": "uf_8",
    })
    users = users.round({"uf_6": 3, "uf_7": 3, "uf_8": 3})

    interactions["dwell_seconds"] = interactions.groupby(
        "session_id"
    )["dwell_seconds"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0
    ).round(3)

    interactions["scroll_pct"] = pd.qcut(
        interactions["scroll_pct"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    all_item_ids = items["item_id"].tolist()
    noise_rows = []
    for sid in interactions["session_id"].unique():
        sess = interactions[interactions["session_id"] == sid]
        uid = sess["user_id"].iloc[0]
        max_pos = sess["position"].max()
        n_noise = rng.randint(1, 3)
        for j in range(n_noise):
            noise_rows.append({
                "session_id": sid,
                "user_id": uid,
                "item_id": rng.choice(all_item_ids),
                "action_type": "view",
                "dwell_seconds": round(np_rng.normal(0, 0.8), 3),
                "scroll_pct": rng.randint(1, 5),
                "position": max_pos + j + 1,
            })
    noise_df = pd.DataFrame(noise_rows)
    interactions = pd.concat([interactions, noise_df], ignore_index=True)

    merge_sessions = set()
    for sid in interactions["session_id"].unique():
        if _det_seed(f"merge_{sid}") % 100 < 25:
            merge_sessions.add(sid)

    merged_parts = []
    for sid, group in interactions.groupby("session_id"):
        if sid in merge_sessions:
            agg = group.groupby("item_id").agg({
                "session_id": "first", "user_id": "first",
                "action_type": "last", "dwell_seconds": "max",
                "scroll_pct": "max", "position": "max",
            }).reset_index()
            merged_parts.append(agg)
        else:
            merged_parts.append(group)
    interactions = pd.concat(merged_parts, ignore_index=True)

    perturbed_parts = []
    for sid, group in interactions.groupby("session_id"):
        if _det_seed(f"pos_{sid}") % 100 < 20:
            group = group.copy()
            positions = group["position"].values.copy()
            n = len(positions)
            for i in range(n):
                if rng.random() < 0.3:
                    j = rng.randint(0, n - 1)
                    positions[i], positions[j] = positions[j], positions[i]
            group["position"] = positions
        perturbed_parts.append(group)
    interactions = pd.concat(perturbed_parts, ignore_index=True)

    interactions = interactions.rename(columns={
        "dwell_seconds": "signal_1",
        "scroll_pct": "signal_2",
        "action_type": "event_type",
        "position": "seq_pos",
    })

    session_ids = sorted(purchases["session_id"].unique())
    train_sids, test_sids = train_test_split(
        session_ids, test_size=0.2, random_state=42
    )
    train_sids_set = set(train_sids)
    test_sids_set = set(test_sids)

    train_ints = interactions[interactions["session_id"].isin(train_sids_set)]
    test_ints = interactions[interactions["session_id"].isin(test_sids_set)]
    train_purchases = purchases[purchases["session_id"].isin(train_sids_set)]
    test_purchases = purchases[purchases["session_id"].isin(test_sids_set)]

    test_user_ids = test_ints["user_id"].unique()
    cold_users = set()
    for uid in test_user_ids:
        if _det_seed(f"cold_{uid}") % 100 < 15:
            cold_users.add(uid)
    users_public = users[~users["user_id"].isin(cold_users)]

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_ints.to_csv(public / "train_sessions.csv", index=False)
    train_purchases.to_csv(public / "train_labels.csv", index=False)
    test_ints.to_csv(public / "test_sessions.csv", index=False)
    items.to_csv(public / "items.csv", index=False)
    users_public.to_csv(public / "users.csv", index=False)

    sample = test_purchases[["session_id"]].copy()
    sample["purchased_item_id"] = 0
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_purchases.to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly loads and parses the multi-file dataset (session interactions, item catalog, user profiles, training labels) without data loss or encoding errors.
- **Rationale:** The dataset spans multiple CSV files with mixed data types (integers, floats, categorical codes). Incorrect parsing or failed joins will produce broken predictions.

### Rubric 2
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV containing exactly 3,000 rows with columns `session_id` and `purchased_item_id`, where each `purchased_item_id` is a valid item_id.
- **Rationale:** An incorrectly formatted submission will fail the grader. The agent must produce output matching the specified schema exactly.

### Rubric 3
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves accuracy meaningfully above random baseline (~0.10) on the test set.
- **Rationale:** A solution scoring at or near 0.10 indicates no meaningful learning — functionally equivalent to random selection from each session's item set.

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
