# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Multi-Channel Ordinal Telemetry Records.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Robust Ordinal State Inference Under Multi-Channel Corruption
```

---

## 3) Problem Description

# Synthetic Robust Ordinal State Inference Under Multi-Channel Corruption

## Overview

This is an **LLM Evaluation** challenge that models the problem of **determining the true quality level of LLM-generated responses when multiple automated scoring pipelines disagree, some pipelines are unreliable, and a fraction actively produce misleading quality assessments with inflated confidence**. Unlike standard LLM evaluation benchmarks that assume a single reliable judge, this challenge requires robust ordinal inference from multiple corrupted, sparse, and adversarially manipulated scoring channels.

The dataset simulates 12,000 LLM-generated response items, each assessed by a subset of 600 independent scoring channels (automated evaluation pipelines with distinct internal configurations). Each channel produces an ordinal quality reading (0–5) accompanied by a **self-reported confidence score**, collected under one of **4 evaluation contexts** (representing different prompt categories or evaluation protocols). Channels belong to one of **5 hierarchical source groups** reflecting different pipeline architectures. A separate expert adjudication process established a **reference quality level** for each item.

Each channel's scoring fidelity depends on an unobserved latent property of the response being evaluated — creating a **channel × item-property interaction** that cannot be captured by per-channel quality metrics. This models the real-world phenomenon where an automated LLM judge may score technical content accurately but struggle with creative writing, or vice versa.

Your task: predict the reference quality level for each test item, given its anonymized feature vector, the channel profiles, and a **severely reduced set of channel readings** (2–3 per test item, compared to 5–8 per training item).

**What makes this problem uniquely challenging:**

- **Adversarial confidence manipulation**: approximately 5% of test readings are **injected spurious scores from channels not originally assigned to those items**. These corrupted readings carry **deliberately inflated confidence scores** (0.78–0.96), designed to appear more trustworthy than genuine readings. Naively trusting high-confidence readings degrades performance — the confidence signal is an **adversarial trap** that must be detected and mitigated.
- **Channel × item-property interaction**: the same scoring channel may produce accurate quality assessments for items with certain latent characteristics and severely degraded assessments for others. This non-separable interaction means per-channel reliability estimates are insufficient — the model must jointly reason about channel profiles and item feature vectors.
- **Asymmetric density transfer**: training items have 5–8 readings each while test items have only 2–3. Models must generalize from a high-density training regime to an extremely sparse test regime. Feature-space proximity between training and test items provides a potential cross-item transfer signal.
- **Multi-context evaluation variation**: readings were collected under 4 different evaluation contexts (`reading_context` 0–3). Channel scoring fidelity may vary across contexts, adding a third interaction dimension.
- **Hierarchical source structure**: channels are grouped into 5 source groups reflecting different pipeline architectures. Group membership partially predicts fidelity patterns but the relationship is non-trivial and interacts with item properties.
- **Decoy feature dimensions**: both item and channel feature vectors contain uninformative dimensions mixed with genuine ones. Feature selection is part of the challenge.
- **Irreducible reference noise**: ~8% of reference levels reflect genuine expert disagreement, capping theoretical performance.

## Evaluation

Submissions are scored using **Quadratic Weighted Kappa (QWK)**, which treats the ordinal states 0–5 as an ordered scale and penalizes predictions far from the reference more heavily than near-misses. **Higher is better.** Minimum: 0.0, Maximum: 1.0.

## Dataset

- `train_readings.csv` — 62,448 training scoring channel outputs: item_id (int), source_id (int), reading (int, ordinal quality 0–5), confidence (float, self-reported reading confidence 0.05–0.99), reading_context (int, evaluation context 0–3)
- `test_readings.csv` — ~6,300 test scoring channel outputs: same columns, only 2–3 readings per item plus ~5% injected high-confidence spurious readings
- `train_labels.csv` — 9,600 rows: item_id (int), reference_level (int, expert-adjudicated quality level 0–5)
- `items.csv` — 12,000 response items: item_id (int), plus 12 anonymized numeric feature columns (x_00 through x_11, float)
- `sources.csv` — 600 scoring channels: source_id (int), plus 11 anonymized numeric feature columns (e_00 through e_10, float), source_group (int, channel group 0–4)
- `sample_submission.csv` — 2,400 rows: item_id (int), reference_level (int, baseline prediction). Shows the required submission format.

### Feature Details

| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Unique response item identifier |
| source_id | int | Unique scoring channel identifier |
| reading | int | Ordinal quality reading produced by the channel (0–5) |
| confidence | float | Self-reported reading confidence (0.05–0.99). Caution: spurious readings carry inflated confidence. |
| reading_context | int | Evaluation context under which the reading was produced (0–3) |
| reference_level | int | Ground-truth quality level from expert adjudication (0–5, train only) |
| x_00 through x_11 | float | Anonymized item feature attributes (mix of informative and decoy) |
| e_00 through e_10 | float | Anonymized channel profile attributes (mix of informative and decoy) |
| source_group | int | Hierarchical channel group assignment (0–4) |

## Submission

Submit a CSV file with exactly 2,400 rows (one per test item), a header row, and two columns:

| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Item identifier from test_readings.csv |
| reference_level | int | Predicted quality level (integer 0–5) |

**Example of a correctly formatted submission file:**

item_id,reference_level
3,4
7,2
15,5
22,1
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
        Quadratic Weighted Kappa on ordinal reference levels.
    """
    try:
        if "item_id" not in submission.columns or "reference_level" not in submission.columns:
            return 0.0
        if submission["item_id"].duplicated().any():
            return 0.0
        if answers["item_id"].duplicated().any():
            return 0.0

        ans_cols = answers[["item_id", "reference_level"]].copy()
        sub_cols = submission[["item_id", "reference_level"]].copy()
        merged = ans_cols.merge(
            sub_cols, on="item_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        if merged["reference_level_pred"].isna().any():
            return 0.0

        if set(submission["item_id"]) != set(answers["item_id"]) or len(submission) != len(answers):
            return 0.0

        y_true = merged["reference_level_true"].astype(int).values
        y_pred = merged["reference_level_pred"].astype(int).values

        n_classes = 6
        conf = np.zeros((n_classes, n_classes), dtype=float)
        for t, p in zip(y_true, y_pred):
            t_c = min(max(int(t), 0), n_classes - 1)
            p_c = min(max(int(p), 0), n_classes - 1)
            conf[t_c, p_c] += 1.0

        w = np.zeros((n_classes, n_classes), dtype=float)
        for i in range(n_classes):
            for j in range(n_classes):
                w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

        row_sum = conf.sum(axis=1)
        col_sum = conf.sum(axis=0)
        n = conf.sum()
        if n == 0:
            return 0.0

        expected = np.outer(row_sum, col_sum) / n

        num = (w * conf).sum()
        den = (w * expected).sum()

        if den == 0:
            return 1.0

        qwk = 1.0 - num / den
        return float(max(0.0, qwk))

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

    items = pd.read_csv(raw / "items.csv")
    annotators = pd.read_csv(raw / "annotators.csv")
    annotations = pd.read_csv(raw / "annotations.csv")
    consensus = pd.read_csv(raw / "consensus.csv")

    rng = _rnd.Random(314159265)
    np_rng = np.random.RandomState(77)

    all_items = sorted(items["item_id"].tolist())
    rng_split = _rnd.Random(271828182)
    rng_split.shuffle(all_items)

    split_idx = int(len(all_items) * 0.8)
    train_ids = set(all_items[:split_idx])
    test_ids = set(all_items[split_idx:])

    item_feat_cols = sorted([c for c in items.columns if c.startswith("if_")])
    n_real_if = len(item_feat_cols)
    n_decoy_if = 4
    all_if_labels = [f"x_{i:02d}" for i in range(n_real_if + n_decoy_if)]
    rng.shuffle(all_if_labels)
    if_rename = {item_feat_cols[i]: all_if_labels[i] for i in range(n_real_if)}
    items = items.rename(columns=if_rename)
    items = items.drop(columns=["topic", "true_label"])

    n_items_total = len(items)
    for j in range(n_real_if, n_real_if + n_decoy_if):
        items[all_if_labels[j]] = np_rng.normal(0, 1, n_items_total).round(4)

    ann_feat_cols = sorted([c for c in annotators.columns if c.startswith("af_")])
    n_real_af = len(ann_feat_cols)
    n_decoy_af = 3
    all_af_labels = [f"e_{i:02d}" for i in range(n_real_af + n_decoy_af)]
    rng.shuffle(all_af_labels)
    af_rename = {ann_feat_cols[i]: all_af_labels[i] for i in range(n_real_af)}
    annotators = annotators.rename(columns=af_rename)

    n_ann_total = len(annotators)
    for j in range(n_real_af, n_real_af + n_decoy_af):
        annotators[all_af_labels[j]] = np_rng.normal(0, 1, n_ann_total).round(4)

    grp_feat = annotators[all_af_labels[1]].values
    boundaries = np.percentile(grp_feat, [20, 40, 60, 80])
    annotators["source_group"] = np.digitize(grp_feat, boundaries).astype(int)

    annotators = annotators.rename(columns={"annotator_id": "source_id"})
    annotations = annotations.rename(columns={"annotator_id": "source_id"})

    old_ids = sorted(annotators["source_id"].tolist())
    new_ids = list(range(len(old_ids)))
    rng.shuffle(new_ids)
    id_map = {old_ids[i]: new_ids[i] for i in range(len(old_ids))}
    annotators["source_id"] = annotators["source_id"].map(id_map)
    annotations["source_id"] = annotations["source_id"].map(id_map)

    annotations = annotations.rename(columns={"label": "reading"})

    train_ann = annotations[annotations["item_id"].isin(train_ids)].copy()
    test_ann_full = annotations[annotations["item_id"].isin(test_ids)].copy()

    test_ann_rows = []
    for item_id in sorted(test_ids):
        item_anns = test_ann_full[test_ann_full["item_id"] == item_id]
        n_keep = rng.randint(2, 3)
        n_keep = min(n_keep, len(item_anns))
        indices = sorted(item_anns.index.tolist())
        rng.shuffle(indices)
        kept = indices[:n_keep]
        test_ann_rows.append(item_anns.loc[kept])

    test_ann = pd.concat(test_ann_rows, ignore_index=True)

    np_rng_conf = np.random.RandomState(999)
    ann_feat_lookup = annotators.set_index("source_id")[all_af_labels[0]]

    for df_ref in [train_ann, test_ann]:
        base = df_ref["source_id"].map(ann_feat_lookup).values.astype(float)
        conf = 0.5 + 0.3 * np.tanh(base * 0.6)
        conf = conf + np_rng_conf.normal(0, 0.08, len(df_ref))
        df_ref["confidence"] = np.clip(conf, 0.05, 0.99).round(3)

    np_rng_noise = np.random.RandomState(555)
    n_noise = int(len(test_ann) * 0.05)
    noise_items = np_rng_noise.choice(sorted(test_ids), n_noise)
    noise_sources = np_rng_noise.choice(sorted(annotators["source_id"]), n_noise)
    noise_readings = np_rng_noise.randint(0, 6, n_noise)
    noise_conf = np_rng_noise.uniform(0.78, 0.96, n_noise).round(3)
    noise_df = pd.DataFrame({
        "item_id": noise_items,
        "source_id": noise_sources,
        "reading": noise_readings,
        "confidence": noise_conf,
    })
    test_ann = pd.concat([test_ann, noise_df], ignore_index=True)
    test_ann = test_ann.drop_duplicates(
        subset=["item_id", "source_id"], keep="first"
    )
    test_ann = test_ann.sort_values("item_id").reset_index(drop=True)
    train_ann = train_ann.sort_values("item_id").reset_index(drop=True)

    np_rng_ctx = np.random.RandomState(777)
    train_ann["reading_context"] = np_rng_ctx.randint(0, 4, len(train_ann))
    test_ann["reading_context"] = np_rng_ctx.randint(0, 4, len(test_ann))

    consensus = consensus.rename(columns={"consensus_label": "reference_level"})
    train_labels = consensus[consensus["item_id"].isin(train_ids)].copy()
    test_labels = consensus[consensus["item_id"].isin(test_ids)].copy()
    train_labels = train_labels.sort_values("item_id").reset_index(drop=True)
    test_labels = test_labels.sort_values("item_id").reset_index(drop=True)

    test_modes = test_ann.groupby("item_id")["reading"].agg(
        lambda x: int(x.mode().iloc[0])
    ).reset_index()
    test_modes.columns = ["item_id", "reference_level"]
    sample_sub = test_modes.sort_values("item_id").reset_index(drop=True)

    items = items.sort_values("item_id").reset_index(drop=True)
    annotators = annotators.sort_values("source_id").reset_index(drop=True)

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_ann.to_csv(public / "train_readings.csv", index=False)
    test_ann.to_csv(public / "test_readings.csv", index=False)
    train_labels.to_csv(public / "train_labels.csv", index=False)
    items.to_csv(public / "items.csv", index=False)
    annotators.to_csv(public / "sources.csv", index=False)
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    test_labels.to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly loads and links all provided CSV files without data loss, duplication, or column misalignment.
- **Rationale:** The dataset spans multiple tables with different schemas. Incorrect linking produces corrupt inputs.

### Rubric 2
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Handles variable-length inputs per item (items may have different numbers of associated readings) without crashing or producing NaN predictions.
- **Rationale:** Items have different amounts of available information. Fixed-length assumptions will cause failures.

### Rubric 3
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 2,400 rows, columns item_id and reference_level, where reference_level is an integer in [0, 5].
- **Rationale:** Incorrectly formatted or incomplete submissions will fail the grader or score zero.

### Rubric 4
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves QWK above the sample submission baseline on the test set, producing predictions that span at least 4 of the 6 ordinal levels.
- **Rationale:** A solution that predicts a single constant or near-constant value across all items demonstrates no meaningful modeling, even if QWK is non-negative.

### Rubric 5
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Incorporates information from at least two of the provided data tables (beyond just train_labels.csv) in the prediction pipeline.
- **Rationale:** Predictions based solely on readings or solely on item features ignore the multi-table structure that is central to the problem.

### Rubric 6
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Criteria:** Prediction quality does not catastrophically degrade for items where available readings disagree with each other (i.e., high variance in readings for a single item).
- **Rationale:** Items with conflicting readings are the hardest cases. A robust solution must handle disagreement rather than failing silently.

### Rubric 7
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Evaluates intermediate predictions on a held-out portion of training data before generating final test predictions.
- **Rationale:** Without internal validation, there is no way to assess whether the model is functioning correctly before submission.

### Rubric 8
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not access test set reference levels or leak private answer data into the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces inflated scores that do not reflect genuine capability.
