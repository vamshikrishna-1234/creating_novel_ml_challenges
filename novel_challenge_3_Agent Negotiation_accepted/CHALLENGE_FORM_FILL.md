# Challenge creation form — fill-in (novel_challenge_3)

Tie this challenge to the **accepted dataset**: Multi-Agent Negotiation Transcripts — Deal Outcome (Synthetic).

---

## 1) Difficulty

**Select:** **Medium**

---

## 2) Challenge Title

```
PREDICT NEXARI PROTOCOL SESSION CONVERGENCE STATE FROM OBFUSCATED MULTI-AGENT LOGS
```

---

## 3) Problem Description

```markdown
# Predict Nexari Protocol Session Convergence State from Obfuscated Multi-Agent Logs

## Overview

You are given **obfuscated interaction logs** from a simulated **autonomous agent coordination system**. In this system, two agents (tagged [BUYER] and [SELLER]) communicate through a fictional protocol called **Nexari** to resolve resource allocation disputes. Each log represents one **coordination session**.

The raw logs have undergone a **multi-stage obfuscation pipeline** that strips direct indicators of the session outcome. Your task is to reconstruct the **convergence state** — the final resolution of the session — from the degraded signal that remains. This simulates a real-world scenario where telemetry data is incomplete, partially redacted, and noisy.

**Obfuscation pipeline applied to every session:**

1. **Tail truncation:** The last 2 messages of each session are removed. The convergence state label reflects the true outcome of the *complete* session — you must predict it from incomplete context.
2. **Action-token collapse:** All typed Nexari protocol tokens (originally distinct action types) have been collapsed to a single generic `NEXO:ACTION` marker. The original action vocabulary is not recoverable.
3. **Numeric quantization:** All precise numeric parameters have been replaced with coarse-grained buckets (`<500>`, `<1500>`, `<3000>`, `<5000>`, `<5000+>`). Fine-grained parameter trajectories are lost.
4. **Positional noise:** In ~20% of sessions, intermediate messages have been shuffled. Message ordering is not always reliable.
5. **Metadata dropout:** The `sector` field (coordination domain) is blank for ~15% of sessions.

Each session converges to one of 4 terminal states:

- **deal_accepted** — both agents reached a mutually confirmed allocation (~30%)
- **deal_rejected** — one agent issued a terminal refusal (~25%)
- **counter_proposed** — the session ended with a pending unresolved revision (~28%)
- **timeout** — the session expired before either agent acted decisively (~17%, minority class)

The convergence state must be inferred from residual signals:
1. **Linguistic patterns** — phrasing in each message (willingness, resistance, delay language)
2. **Quantized parameter shifts** — how bucketed values change across messages
3. **Session structure** — message count, speaker balance, presence of system messages
4. **Domain context** — when the sector field is available

This challenge tests robustness to **information loss, noise, and missing data** — skills critical for real-world deployed ML systems.

## Evaluation

**Metric:** Inverse-frequency-weighted macro F1 (custom).

For each of the 4 convergence classes, compute class-level F1 = 2 × precision × recall / (precision + recall). Then weight each F1 by (N / n_class) where N = total samples and n_class = count for that class. Normalize weights to sum to 1. The final score is the weighted sum.

This metric penalizes models that ignore the minority class. The rarest class (timeout, ~17%) carries the highest weight.

**Grading direction:** Maximize.

**Theoretical minimum:** 0
**Theoretical maximum:** 1

## Dataset (prepared)

**In public/:**

- **train.csv** — id, transcript, num_turns, sector, label. Exactly 22,500 rows (stratified 75% split).
- **test.csv** — id, transcript, num_turns, sector. Exactly 7,500 rows. No label column.
- **sample_submission.csv** — id, label. Example format with placeholder labels.

**In private/ (not visible to solvers):** answers.csv — id, label.

**Column descriptions:**

| Column     | Type   | Description |
|------------|--------|-------------|
| id         | int    | Unique session identifier |
| transcript | string | Obfuscated interaction log. Last 2 messages removed; all protocol tokens collapsed to NEXO:ACTION; numeric parameters quantized to buckets; ~20% of sessions have shuffled intermediate messages. Messages separated by `\|\|\|`. Speaker tags: [BUYER], [SELLER], or [SYSTEM]. |
| num_turns  | int    | Number of messages in the provided (truncated) transcript. |
| sector     | string | Coordination domain. One of: energy, defense, biotech, logistics, manufacturing. **Blank for ~15% of rows.** |
| label      | string | Convergence state. One of: deal_accepted, deal_rejected, counter_proposed, timeout |

## Submission

Submit a CSV with exactly these columns:

| Column | Type   | Description |
|--------|--------|-------------|
| id     | int    | Session identifier from test.csv (one per test row) |
| label  | string | Predicted convergence state: deal_accepted, deal_rejected, counter_proposed, or timeout |

**Requirements:**

- Exactly **7,500 rows** (one per test session). No duplicate ids.
- Header row required.
- Label values must be one of the 4 valid convergence states listed above.

**Example of a correctly formatted submission (first few rows):**

```
id,label
0,deal_accepted
1,counter_proposed
2,deal_rejected
3,timeout
4,deal_accepted
...
```

(Your file must have 7,500 rows total, one per test id, with no duplicates.)
```

---

## 4) Tags

**Select:** **text**, **feature-engineering**

---

## 5) Grading Configuration

**Grade direction:** **Maximize**

**Theoretical minimum:** **0**

**Theoretical maximum:** **1**

---

## 6) Grading Script (Custom)

```python
import pandas as pd
import numpy as np


VALID_LABELS = {"deal_accepted", "deal_rejected", "counter_proposed", "timeout"}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth using frequency-weighted macro F1.

    Args:
        submission: The agent's predictions (loaded from submission.csv). Columns: id, label.
        answers: Ground truth labels (loaded from private/answers.csv). Columns: id, label.

    Returns:
        A float score in [0, 1]. Direction: maximize.

    Raises:
        ValueError: If the submission format is invalid.
        RuntimeError: If an unexpected error occurs.
    """
    try:
        if "id" not in submission.columns or "label" not in submission.columns:
            raise ValueError("Submission must have columns: id, label")

        if submission["id"].duplicated().any():
            raise ValueError("Submission must not contain duplicate id values")

        if answers["id"].duplicated().any():
            raise ValueError("Answers file must not contain duplicate id values")

        if len(submission) != len(answers):
            raise ValueError(
                f"Submission must have exactly {len(answers)} rows, got {len(submission)}"
            )

        sub_ids = set(submission["id"])
        ans_ids = set(answers["id"])
        if sub_ids != ans_ids:
            missing = ans_ids - sub_ids
            extra = sub_ids - ans_ids
            if missing:
                raise ValueError(f"Submission missing ids: {len(missing)} (e.g. {list(missing)[:5]})")
            if extra:
                raise ValueError(f"Submission has extra ids: {len(extra)}")

        sub_labels = set(submission["label"].dropna().unique())
        invalid = sub_labels - VALID_LABELS
        if invalid:
            raise ValueError(f"Invalid label values: {invalid}. Must be one of {VALID_LABELS}")

        if submission["label"].isna().any():
            raise ValueError("Submission has missing (NaN) label values")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["label_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        y_true = merged["label_true"].values
        y_pred = merged["label_pred"].values

        classes = sorted(VALID_LABELS)
        total = len(y_true)
        class_f1s = []
        class_weights = []

        for cls in classes:
            true_pos = np.sum((y_true == cls) & (y_pred == cls))
            false_pos = np.sum((y_true != cls) & (y_pred == cls))
            false_neg = np.sum((y_true == cls) & (y_pred != cls))

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_count = np.sum(y_true == cls)
            weight = (total / class_count) if class_count > 0 else 0.0

            class_f1s.append(f1)
            class_weights.append(weight)

        total_weight = sum(class_weights)
        if total_weight == 0:
            return float(0.0)
        class_weights = [w / total_weight for w in class_weights]

        score = sum(f * w for f, w in zip(class_f1s, class_weights))
        return float(score)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Data Preparation Pipeline

**Input:** raw dataset file **data.csv** (id, transcript, num_turns, sector, label).

**Script — prepare.py:**

Paste the full prepare.py from the `novel_challenge_3/prepare.py` file (contains truncation, token masking, price redaction, turn shuffling, missing value injection, and 75/25 split). Too long for inline — copy directly from the file.

Run **Run Prepare** after pasting.

---

## 8) Evaluation Rubrics

Add each via "Add Rubric":

**1** — DATA_HANDLING | REQUIRED
**Criterion:** Loads and uses the `transcript` column from train/test; does not drop it or use only `num_turns` and `sector`.
**Rationale:** The interaction log is the primary input containing phrasing patterns and turn structure. Ignoring it means the model cannot learn session-level cues critical to resolution state prediction.

**2** — DATA_HANDLING | RECOMMENDED
**Criterion:** Handles missing values in the `sector` column (~15% missing) using imputation, a dedicated "unknown" category, or a model that natively handles nulls.
**Rationale:** Sector is missing for ~15% of rows. Dropping those rows or crashing on nulls loses data and hurts performance on the minority class.

**3** — FEATURE_ENGINEERING | RECOMMENDED
**Criterion:** Extracts at least one feature from phrasing patterns in the transcript (e.g. presence of concession language, stalling phrases, refusal phrases, or turn-level sentiment).
**Rationale:** Since protocol tokens are masked to NEXO:ACTION, the English phrasing is the primary signal carrier. Models that ignore phrase-level content will underperform.

**4** — FEATURE_ENGINEERING | RECOMMENDED
**Criterion:** Engineers at least one feature from bucketed price trajectories (e.g. how price range buckets shift across turns, or the distribution of bucket types).
**Rationale:** Exact prices are redacted but bucket patterns still carry information about concession dynamics.

**5** — TRAINING | RECOMMENDED
**Criterion:** Uses a proper train/validation split or cross-validation with stratification for model selection; does not tune or select models using the test set.
**Rationale:** Class imbalance (timeout ~17%) means random splits can misrepresent minority class performance. Stratified validation matters for this task.

**6** — CODE_QUALITY | REQUIRED
**Criterion:** Submission CSV has columns `id` and `label` with exactly one row per test id, no duplicate ids, and all labels are one of the 4 valid resolution states.
**Rationale:** Grader expects this format and valid labels; wrong format or invalid labels cause grading failure.

**7** — UNIVERSAL | UNIVERSAL
**Criterion:** Does not use test set or test labels for training, feature computation, or normalization.
**Rationale:** Universal anti-leakage criterion.

---

## 9) Agent Evaluation Runs

No fill; runs on submit.

---

## Checklist

- [ ] Dataset: Multi-Agent Negotiation Transcripts — Deal Outcome (Synthetic) accepted and selected
- [ ] Difficulty: Medium
- [ ] Title: PREDICT DEAL OUTCOME FROM MULTI-AGENT NEGOTIATION TRANSCRIPTS
- [ ] Problem description: 24,000 / 6,000 rows, 4 classes, freq-weighted macro F1, maximize, min 0 max 1
- [ ] Tags: text, feature-engineering
- [ ] Grading: Maximize; min 0; max 1
- [ ] Grading script: custom freq-weighted macro F1; merge left; id/length/label validation; try/except
- [ ] Prepare: stratified split; data.csv → train/test/sample_submission/answers; Run Prepare succeeded
- [ ] 7 rubrics added (2 REQUIRED, 4 RECOMMENDED, 1 UNIVERSAL)
