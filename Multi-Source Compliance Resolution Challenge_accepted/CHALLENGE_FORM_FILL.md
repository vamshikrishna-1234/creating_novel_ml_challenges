# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Multi-Source Compliance Inspection Resolution Dataset (Synthetic).

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Multi-Source Compliance Resolution Challenge
```

---

## 3) Problem Description

```markdown
# Multi-Source Compliance Resolution Challenge

## Overview

This is a **sequence-to-sequence** task requiring **multi-document reasoning**, **rule interpretation**, and **structured output generation**.

You are given fictional environmental compliance inspection data. For each facility visit, 2–4 independent inspectors filed separate fragments describing their findings. These fragments may **contradict each other** — different inspectors may report different violations, and some fragments may be partially corrupted (`[CORRUPTED]` spans).

Each row also includes a **compliance codebook** — a set of 4–15 natural language rules specific to that row. These rules govern:

- **Contradiction resolution:** Whether a violation is confirmed if *any* inspector reports it, or only if a *majority* do.
- **Severity escalation:** Conditions that raise the severity level (e.g., violation count thresholds, facility sector, corrupted data fallback).
- **Action assignment:** Mapping from severity levels to required actions.
- **Penalty mapping:** Mapping from actions to penalty tiers, with optional caps.

Your task is to **read the inspector fragments, apply the codebook rules in order, and produce the exact 6-field structured verdict string**.

**Output format:**
```
FACILITY:KRX-0447 | VIOLATIONS:leak,emission_excess | COUNT:2 | SEVERITY:elevated | ACTION:reinspect_30d | PENALTY:tier_B
```

**Why this is hard:**
- **Multi-source contradictions:** Inspectors disagree; you must apply the correct reconciliation rule (any-flags vs majority-vote) for each violation type.
- **Variable codebook:** Rules change per row — you cannot memorize a single rule set.
- **Chained inference:** PENALTY depends on ACTION, which depends on SEVERITY, which depends on reconciled VIOLATIONS. One mistake cascades.
- **Decoy rules:** Some codebook rules reference violation types not present in any fragment — they must be correctly ignored.
- **Corrupted fragments:** Missing data triggers fallback rules that override normal severity.
- **Rule ordering:** Rules are applied sequentially; later rules can override earlier ones.

## Evaluation

Submissions are scored using **per-field exact match accuracy** across all 6 verdict fields, averaged over all test rows.

For each row:
- Parse both the predicted and true verdict into 6 fields (FACILITY, VIOLATIONS, COUNT, SEVERITY, ACTION, PENALTY).
- Compare each field: exact match = 1, mismatch = 0.
- Row score = (number of matching fields) / 6.

Final score = mean of all row scores. **Higher is better.** Range: [0.0, 1.0].

**Baseline scores:**
- All-default verdict: ~0.24
- Perfect: 1.0

## Dataset (prepared)

**In public/:**

- **train.csv** — id, input, verdict. Exactly 16,000 rows (stratified 80% split by severity).
- **test.csv** — id, input. Exactly 4,000 rows. No verdict column.
- **sample_submission.csv** — id, verdict. Example format with placeholder verdicts.

**In private/ (not visible to solvers):** answers.csv — id, verdict.

**Column descriptions:**

| Column  | Type   | Description |
|---------|--------|-------------|
| id      | int    | Unique row identifier |
| input   | string | Multi-line text: 2–4 inspector fragments + compliance codebook (4–15 rules). Rules may be shuffled, paraphrased, or include decoy rules referencing absent violation types. ~25% of fragments have injected noise sentences. |
| verdict | string | 6-field structured verdict: `FACILITY:X \| VIOLATIONS:Y \| COUNT:Z \| SEVERITY:W \| ACTION:A \| PENALTY:P` |

**Verdict fields:**

| Field      | Values | Description |
|------------|--------|-------------|
| FACILITY   | e.g. KRX-0447 | Facility ID from fragments |
| VIOLATIONS | comma-separated list or "none" | Confirmed violations after reconciliation |
| COUNT      | integer | Number of confirmed violations |
| SEVERITY   | negligible, low, moderate, elevated, high, critical | Derived severity after all escalation rules |
| ACTION     | no_action, log_only, reinspect_30d, reinspect_7d, immediate_halt, partial_shutdown, full_shutdown | Action derived from severity mapping rules |
| PENALTY    | none, tier_A, tier_B, tier_C, tier_D, tier_E | Penalty derived from action mapping rules, possibly capped |

## Submission

Submit a CSV with exactly these columns:

| Column  | Type   | Description |
|---------|--------|-------------|
| id      | int    | Row identifier from test.csv |
| verdict | string | Predicted 6-field verdict string |

**Requirements:**

- Exactly **4,000 rows** (one per test row). No duplicate ids.
- Header row required.
- Verdict must follow the exact format: `FACILITY:X | VIOLATIONS:Y | COUNT:Z | SEVERITY:W | ACTION:A | PENALTY:P`

**Example:**

```
id,verdict
42,FACILITY:KRX-0447 | VIOLATIONS:leak,emission_excess | COUNT:2 | SEVERITY:elevated | ACTION:reinspect_30d | PENALTY:tier_B
99,FACILITY:VLN-1234 | VIOLATIONS:none | COUNT:0 | SEVERITY:negligible | ACTION:no_action | PENALTY:none
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


VERDICT_FIELDS = ["FACILITY", "VIOLATIONS", "COUNT", "SEVERITY", "ACTION", "PENALTY"]


def _parse_verdict(verdict_str: str) -> dict:
    if not isinstance(verdict_str, str):
        return {}
    fields = {}
    for part in verdict_str.split(" | "):
        if ":" in part:
            key, val = part.split(":", 1)
            fields[key.strip()] = val.strip()
    return fields


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
                if true_val == pred_val:
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

**Input:** raw dataset file **data.csv** (id, input, verdict).

**Script — prepare.py:**

*(See prepare.py file — obfuscation includes: 60% rule shuffling, 40% fragment shuffling, 50% decoy rule injection, 30% rule paraphrasing, 25% noise injection into fragments. Stratified 80/20 split by severity.)*

Run **Run Prepare** after pasting.

---

## 8) Evaluation Rubrics

Add each via "Add Rubric":

**1** — DATA_HANDLING | REQUIRED
**Criterion:** Parses the multi-line input text to extract inspector fragments and codebook rules as separate structured components.
**Rationale:** The input contains two distinct sections (fragments and codebook) that must be parsed independently. Treating the entire input as flat text loses the structural information needed for rule application.

**2** — FEATURE_ENGINEERING | RECOMMENDED
**Criterion:** Extracts violation types mentioned in each inspector fragment and tracks which inspector reported which violations.
**Rationale:** Contradiction resolution rules (any-flags vs majority-vote) require knowing per-inspector violation reports, not just the union of all mentioned violations.

**3** — MODELING | RECOMMENDED
**Criterion:** Implements or learns the codebook rule application logic, applying rules sequentially to derive the verdict fields in dependency order (violations → severity → action → penalty).
**Rationale:** The chained dependency structure means fields must be derived in order; applying rules out of order or independently will produce incorrect verdicts.

**4** — DATA_HANDLING | RECOMMENDED
**Criterion:** Handles corrupted fragments (`[CORRUPTED]` spans) by detecting corruption markers and applying fallback rules from the codebook rather than ignoring or hallucinating the missing content.
**Rationale:** ~15% of fragments are corrupted; the codebook specifies explicit fallback severity rules for corrupted data that override normal derivation.

**5** — FEATURE_ENGINEERING | RECOMMENDED
**Criterion:** Identifies and correctly ignores decoy codebook rules that reference violation types not present in any inspector fragment.
**Rationale:** ~50% of prepared rows contain decoy rules; applying them would incorrectly add violations or change severity for non-existent findings.

**6** — CODE_QUALITY | REQUIRED
**Criterion:** Submission CSV has columns `id` and `verdict` with exactly one row per test id, no duplicate ids, and verdict strings follow the required 6-field pipe-delimited format.
**Rationale:** Grader expects this exact format; malformed verdicts will score 0 on all fields.

**7** — TRAINING | RECOMMENDED
**Criterion:** Uses the training data to learn or validate rule interpretation patterns, rather than relying solely on zero-shot LLM generation without reference to training examples.
**Rationale:** The variable codebook and paraphrased rules mean that learning from training examples (where both input and verdict are visible) provides signal about rule semantics and edge cases.

**8** — UNIVERSAL | UNIVERSAL
**Criterion:** Does not use test set verdicts for training, feature computation, or rule calibration.
**Rationale:** Universal anti-leakage criterion.

---

## 9) Agent Evaluation Runs

No fill; runs on submit.

---

## Checklist

- [ ] Dataset: Multi-Source Compliance Inspection Resolution Dataset (Synthetic) accepted and selected
- [ ] Difficulty: Hard
- [ ] Title: Multi-Source Compliance Resolution Challenge
- [ ] Problem description: 16,000 / 4,000 rows, 6-field verdict, per-field exact match, maximize, min 0 max 1
- [ ] Tags: text, generative
- [ ] Grading: Maximize; min 0; max 1
- [ ] Grading script: per-field exact match; merge left; id/length validation; try/except
- [ ] Prepare: shuffle rules 60%, shuffle fragments 40%, decoy rules 50%, paraphrase 30%, noise 25%; stratified 80/20 split
- [ ] 8 rubrics added (2 REQUIRED, 5 RECOMMENDED, 1 UNIVERSAL)
