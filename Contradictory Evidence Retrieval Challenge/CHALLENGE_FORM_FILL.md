# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Multi-Stance Claim Verification Passage Corpus (Synthetic).

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Contradictory Evidence Retrieval Challenge
```

---

## 3) Problem Description

```markdown
# Contradictory Evidence Retrieval Challenge

## Overview

This is a **RAG (Retrieval-Augmented Generation)** task requiring **evidence retrieval**, **contradictory reasoning**, and **structured output generation**.

You are given fictional scientific claims, each paired with a corpus of 8–15 evidence passages from fictional research institutions. Your task is to analyze the passages, determine which ones constitute genuine evidence for or against the claim, assess the overall stance, and produce a structured verdict.

**What makes this hard:**

- **Near-miss distractors:** Some passages mention the same substance or the same effect as the claim, but not both. They appear topically relevant but are not actual evidence.
- **Hedged passages:** Some passages mention the correct substance and effect but use uncertain, preliminary, or inconclusive language. These should not be counted as evidence.
- **Contradictory evidence:** Supporting and contradicting passages coexist. The model must weigh them to determine the overall stance.
- **Confidence estimation:** Confidence depends on the ratio of evidence imbalance, requiring the model to count and compare evidence types accurately.
- **Evidence ID tracking:** The model must identify the exact passage IDs of all genuine evidence passages (both supporting and contradicting), excluding distractors and hedged passages.

**Input per row:**
- A **claim** sentence asserting that a fictional substance improves a material property under specific conditions.
- A **passages** field containing 8–15 numbered passages `[P1]`, `[P2]`, etc.

**Output per row:**
A structured verdict string with 3 fields:
```
STANCE:support | EVIDENCE_IDS:2,5,7,9 | CONFIDENCE:medium
```

- **STANCE:** `support`, `contradict`, or `insufficient`
- **EVIDENCE_IDS:** Comma-separated 1-indexed passage IDs of all genuine evidence passages (both supporting and contradicting), or `none`
- **CONFIDENCE:** `high`, `medium`, or `low`

## Evaluation

Submissions are scored using **per-field exact match accuracy** across all 3 verdict fields, averaged over all test rows.

For each row:
- Parse both the predicted and true verdict into 3 fields (STANCE, EVIDENCE_IDS, CONFIDENCE).
- Compare each field: exact match = 1, mismatch = 0. For EVIDENCE_IDS, the comparison is set-based (order does not matter within the comma-separated list).
- Row score = (number of matching fields) / 3.

Final score = mean of all row scores. **Higher is better.** Range: [0.0, 1.0].

**Baseline scores:**
- Constant guess: ~0.18
- Perfect: 1.0

## Dataset (prepared)

**In public/:**

- **train.csv** — id, claim, passages, verdict. Exactly 16,000 rows (stratified 80% split by stance).
- **test.csv** — id, claim, passages. Exactly 4,000 rows. No verdict column.
- **sample_submission.csv** — id, verdict. Example format with placeholder verdicts.

**In private/ (not visible to solvers):** answers.csv — id, verdict.

**Column descriptions:**

| Column   | Type   | Description |
|----------|--------|-------------|
| id       | int    | Unique row identifier |
| claim    | string | A sentence asserting that a fictional substance improves a specific material property under specific environmental conditions. |
| passages | string | 8–17 numbered passages `[P1]`, `[P2]`, etc. Each passage is a single sentence from a fictional research institution describing experimental findings. Passages may have institution names abbreviated, years redacted, methods removed, or cross-references to other passage IDs injected. Some rows include additional hedged decoy passages added during preparation. |
| verdict  | string | 3-field structured verdict: `STANCE:<stance> | EVIDENCE_IDS:<ids> | CONFIDENCE:<conf>` |

**Verdict fields:**

| Field        | Values | Description |
|--------------|--------|-------------|
| STANCE       | support, contradict, insufficient | Overall stance of evidence toward the claim |
| EVIDENCE_IDS | comma-separated integers or "none" | IDs of all genuine evidence passages (sorted ascending) |
| CONFIDENCE   | high, medium, low | Confidence level derived from evidence balance |

## Submission

Submit a CSV with exactly these columns:

| Column  | Type   | Description |
|---------|--------|-------------|
| id      | int    | Row identifier from test.csv |
| verdict | string | Predicted 3-field verdict string |

**Requirements:**

- Exactly **4,000 rows** (one per test row). No duplicate ids.
- Header row required.
- Verdict must follow the exact format: `STANCE:<stance> | EVIDENCE_IDS:<ids> | CONFIDENCE:<conf>`

**Example:**

```
id,verdict
42,STANCE:support | EVIDENCE_IDS:2,5,7,9 | CONFIDENCE:medium
99,STANCE:contradict | EVIDENCE_IDS:1,3,6 | CONFIDENCE:high
```
```

---

## 4) Tags

**Select:** **text**, **generative**

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


VERDICT_FIELDS = ["STANCE", "EVIDENCE_IDS", "CONFIDENCE"]


def _parse_verdict(verdict_str: str) -> dict:
    if not isinstance(verdict_str, str):
        return {}
    fields = {}
    for part in verdict_str.split(" | "):
        if ":" in part:
            key, val = part.split(":", 1)
            fields[key.strip()] = val.strip()
    return fields


def _normalize_evidence_ids(ids_str: str) -> str:
    ids_str = ids_str.strip().lower()
    if ids_str in ("none", "", "n/a"):
        return "none"
    try:
        ids = sorted(int(x.strip()) for x in ids_str.split(",") if x.strip())
        return ",".join(str(i) for i in ids)
    except ValueError:
        return ids_str


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score using per-field exact match on structured verdict strings.

    Args:
        submission: Agent predictions. Columns: id, verdict.
        answers: Ground truth. Columns: id, verdict.

    Returns:
        Float in [0, 1]. Direction: maximize.
    """
    try:
        if "id" not in submission.columns or "verdict" not in submission.columns:
            raise ValueError("Submission must have columns: id, verdict")

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
                raise ValueError(f"Submission missing ids: {len(missing)}")
            if extra:
                raise ValueError(f"Submission has extra ids: {len(extra)}")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["verdict_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        row_scores = []
        for _, row in merged.iterrows():
            true_fields = _parse_verdict(row["verdict_true"])
            pred_fields = _parse_verdict(row["verdict_pred"])

            if not true_fields:
                row_scores.append(0.0)
                continue

            matches = 0
            total = len(VERDICT_FIELDS)
            for field in VERDICT_FIELDS:
                true_val = true_fields.get(field, "")
                pred_val = pred_fields.get(field, "")

                if field == "EVIDENCE_IDS":
                    true_val = _normalize_evidence_ids(true_val)
                    pred_val = _normalize_evidence_ids(pred_val)

                if true_val.strip().lower() == pred_val.strip().lower():
                    matches += 1

            row_scores.append(matches / total)

        if not row_scores:
            return 0.0

        score = float(np.mean(row_scores))
        if np.isnan(score):
            return 0.0
        return score

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Data Preparation Pipeline

**Input:** raw dataset file **data.csv** (id, claim, passages, verdict).

**Script — prepare.py:**

*(See prepare.py file — obfuscation includes: 35% passage paraphrasing via synonym substitution, 50% institution name abbreviation, 30% method citation removal, 25% year redaction, 20% cross-reference noise injection, 40% hedged decoy passage injection, full passage shuffle with evidence ID remapping. Stratified 80/20 split by stance.)*

Run **Run Prepare** after pasting.

---

## 8) Evaluation Rubrics

Add each via "Add Rubric":

**1** — DATA_HANDLING | REQUIRED
**Criterion:** Parses the claim and numbered passages from the input, correctly extracting each passage's text and its passage ID.
**Rationale:** The input contains a claim and 8–17 numbered passages that must be individually parsed. Treating passages as a single block of text loses the per-passage identity needed for evidence ID prediction.

**2** — MODELING | REQUIRED
**Criterion:** Produces a verdict string in the exact required format with all 3 fields (STANCE, EVIDENCE_IDS, CONFIDENCE) separated by ` | `.
**Rationale:** The grader parses the verdict by splitting on ` | ` and `:`. Malformed verdicts will score 0 on all fields.

**3** — TRAINING | RECOMMENDED
**Criterion:** Uses the training data to learn patterns from examples where both input and verdict are visible, rather than relying solely on zero-shot generation.
**Rationale:** The training set provides 16,000 labeled examples that reveal the relationship between passage content and verdict fields. Models that leverage this signal will outperform zero-shot approaches.

**4** — DATA_HANDLING | RECOMMENDED
**Criterion:** Handles obfuscation artifacts in passages, including abbreviated institution names, redacted years (`[YEAR]`), removed method citations, and injected cross-references (`cf. PN`).
**Rationale:** The preparation pipeline introduces these artifacts; models that are robust to them will perform better than those that rely on clean passage formatting.

**5** — UNIVERSAL | UNIVERSAL
**Criterion:** Does not use test set verdicts for training, feature computation, or model calibration.
**Rationale:** Universal anti-leakage criterion.

---

## 9) Agent Evaluation Runs

No fill; runs on submit.

---

## Checklist

- [ ] Dataset: Contradictory Evidence Retrieval Dataset (Synthetic) accepted and selected
- [ ] Difficulty: Hard
- [ ] Title: Contradictory Evidence Retrieval Challenge
- [ ] Problem description: 16,000 / 4,000 rows, 3-field verdict, per-field exact match, maximize, min 0 max 1
- [ ] Tags: text, generative
- [ ] Grading: Maximize; min 0; max 1
- [ ] Grading script: per-field exact match with evidence ID normalization; merge left; id/length validation; try/except
- [ ] Prepare: paraphrase 35%, abbreviate institutions 50%, remove methods 30%, redact years 25%, cross-references 20%, hedged decoys 40%; full shuffle; stratified 80/20 split
- [ ] 5 rubrics added (2 REQUIRED, 2 RECOMMENDED, 1 UNIVERSAL)
