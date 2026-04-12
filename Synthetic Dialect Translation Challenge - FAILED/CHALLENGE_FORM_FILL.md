# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Dialect Translation Corpus.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Dialect Translation Challenge
```

---

## 3) Problem Description

```markdown
# Synthetic Dialect Translation Challenge

## Overview

This is a **sequence-to-sequence** task requiring **fictional dialect-to-English translation**.

You are given sentences written in one of five entirely fictional dialects. Each dialect applies a unique, consistent combination of transformations to standard English — including lexical substitutions (common words replaced with invented tokens), morphological changes (altered tense/plural suffixes), phonetic respelling (consonant and vowel pattern changes), syntax reordering, and agglutination (preposition merging). Your task is to translate each dialect sentence back to its original standard English form.

**Why this requires learning from training data:**

- The dialects are completely fictional — no pre-trained language model has seen them before.
- Zero-shot or few-shot prompting will not work because the transformation rules are never explicitly stated.
- The training set provides 16,000 paired examples (dialect sentence → English original) that are the only source of information about how each dialect works.
- Each dialect applies multiple simultaneous transformation types, making simple rule extraction insufficient without substantial pattern learning.

**Input per row:**
- A **dialect** label (one of 5 dialect names or codes).
- A **transformed** sentence in the fictional dialect.

**Output per row:**
- The **original** standard English sentence.

## Evaluation

Submissions are scored using **Average Character Error Rate (CER)**.

For each row:
- CER = edit_distance(predicted, ground_truth) / len(ground_truth)

Final score = mean CER across all test rows. **Lower is better.** Range: [0.0, ∞).

**Baseline scores:**
- Constant guess: ~0.64
- Perfect: 0.0

## Dataset (prepared)

**In public/:**

- **train.csv** — id, dialect, transformed, original. Exactly 16,000 rows (stratified 80% split by dialect).
- **test.csv** — id, dialect, transformed. Exactly 4,000 rows. No original column.
- **sample_submission.csv** — id, original. Example format with placeholder predictions.

**In private/ (not visible to solvers):** answers.csv — id, original.

**Column descriptions:**

| Column      | Type   | Description |
|-------------|--------|-------------|
| id          | int    | Unique row identifier |
| dialect     | string | Fictional dialect name or code. Identifies which transformation set was applied. |
| transformed | string | The dialect-transformed sentence. This is the input to translate. May contain character-level noise, whitespace jitter, or case perturbations introduced during preparation. |
| original    | string | The original standard English sentence (target output). |

## Submission

Submit a CSV with exactly these columns:

| Column   | Type   | Description |
|----------|--------|-------------|
| id       | int    | Row identifier from test.csv |
| original | string | Predicted English translation |

**Requirements:**

- Exactly **4,000 rows** (one per test row). No duplicate ids.
- Header row required.
- The `original` column must contain the predicted English sentence.

**Example:**

```
id,original
42,the engineer examines the report
99,the analyst validated comprehensive the parameters
```
```

---

## 4) Tags

**Select:** **text**, **generative**

---

## 5) Grading Configuration

**Grade direction:** **Minimize**

**Theoretical minimum:** **0**

**Theoretical maximum:** **infinity**

---

## 6) Grading Script (Custom)

```python
import pandas as pd
import numpy as np


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score using Average Character Error Rate (CER).

    Args:
        submission: Agent predictions. Columns: id, original.
        answers: Ground truth. Columns: id, original.

    Returns:
        Float >= 0.0. Direction: minimize. Perfect = 0.0.
    """
    try:
        if "id" not in submission.columns or "original" not in submission.columns:
            raise ValueError("Submission must have columns: id, original")

        if submission["id"].duplicated().any():
            raise ValueError("Submission must not contain duplicate id values")

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
                raise ValueError(f"Submission missing ids: {len(missing)}")
            if extra:
                raise ValueError(f"Submission has extra ids: {len(extra)}")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["original_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        cers = []
        for _, row in merged.iterrows():
            true_str = str(row["original_true"])
            pred_str = str(row["original_pred"])
            if len(true_str) == 0:
                cer = 0.0 if len(pred_str) == 0 else 1.0
            else:
                cer = _levenshtein(pred_str, true_str) / len(true_str)
            cers.append(cer)

        if not cers:
            return 1.0

        score = float(np.mean(cers))
        if np.isnan(score):
            return 1.0
        return score

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Data Preparation Pipeline

**Input:** raw dataset file **data.csv** (id, dialect, transformed, original).

**Script — prepare.py:**

*(See prepare.py file — obfuscation includes: 15% character-level noise, 20% whitespace jitter, 25% case perturbation, 30% dialect label code substitution. Stratified 80/20 split by dialect.)*

Run **Run Prepare** after pasting.

---

## 8) Evaluation Rubrics

Add each via "Add Rubric":

**1** — DATA_HANDLING | REQUIRED
**Criterion:** Parses the input CSV correctly, extracting the dialect label and transformed sentence for each row.
**Rationale:** The input contains a dialect identifier and a transformed sentence that must be read as separate fields. Mishandling CSV parsing or encoding will produce incorrect inputs.

**2** — CODE_QUALITY | REQUIRED
**Criterion:** Submission CSV has columns `id` and `original` with exactly one row per test id, no duplicate ids, and original values are non-empty strings.
**Rationale:** The grader expects this exact format; missing or malformed rows will receive maximum error scores.

**3** — TRAINING | REQUIRED
**Criterion:** Builds a model that learns from the 16,000 training pairs rather than producing outputs without reference to the training data.
**Rationale:** The dialects are entirely fictional and not present in any pre-existing corpus. A solution that does not learn from the provided training examples will produce meaningless outputs.

**4** — MODELING | RECOMMENDED
**Criterion:** Produces translations that achieve a measurably lower CER than a naive constant-output baseline (~0.64).
**Rationale:** A meaningful solution must demonstrate that it has learned some structure from the data, producing outputs closer to the ground truth than a trivial guess.

**5** — DATA_HANDLING | RECOMMENDED
**Criterion:** Handles all dialect types present in the test set, including rows where the dialect label may appear in an alternate format.
**Rationale:** The test set contains rows from all five dialects, and some dialect labels are presented differently during preparation. A robust solution must not fail silently on any subset of the data.

**6** — MODELING | RECOMMENDED
**Criterion:** Produces outputs that are well-formed English sentences rather than partially transformed or garbled text.
**Rationale:** The target outputs are grammatical English sentences. Solutions that leave dialect artifacts in the output will incur higher character error rates.

**7** — UNIVERSAL | UNIVERSAL
**Criterion:** Does not use test set original sentences for training, feature computation, or model calibration.
**Rationale:** Universal anti-leakage criterion.

---

## 9) Agent Evaluation Runs

No fill; runs on submit.

---

## Checklist

- [ ] Dataset: Synthetic Dialect Translation Corpus accepted and selected
- [ ] Difficulty: Hard
- [ ] Title: Synthetic Dialect Translation Challenge
- [ ] Problem description: 16,000 / 4,000 rows, CER metric, minimize, min 0 max infinity
- [ ] Tags: text, generative
- [ ] Grading: Minimize; min 0; max infinity
- [ ] Grading script: CER via Levenshtein; merge left; id/length validation; try/except
- [ ] Prepare: char noise 15%, whitespace jitter 20%, case perturb 25%, dialect code 30%; stratified 80/20 split
- [ ] 7 rubrics added (3 REQUIRED, 3 RECOMMENDED, 1 UNIVERSAL) — no strategy hints
