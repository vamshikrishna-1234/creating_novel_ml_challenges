# Rules compliance checklist (novel_challenge_3)

Use this to confirm every rule in `rules/` is satisfied before submit.

---

## intro and datasets.txt + dataset and challenge.txt — Dataset

| Rule | Status |
|------|--------|
| Dataset description: Overview, File Structure, Features table, Notes | ✅ Your dataset has all four |
| All columns and types documented (id, transcript, num_turns, sector, label) | ✅ |
| Sufficiently many samples (30,000) | ✅ |
| Prepare script reproducible and deterministic | ✅ random_state=42, stratify |
| Synthetic: document generation process + reproducible script | ✅ generate.py in Notes |
| License/source (synthetic: no URL) | ✅ |

---

## dataset and challenge.txt — Prepare script

| Rule | Status |
|------|--------|
| train.csv = id + features + labels | ✅ id, transcript, num_turns, sector, label |
| test.csv = id + features only | ✅ id, transcript, num_turns, sector |
| sample_submission.csv in public/ | ✅ id, label (placeholder) |
| answers.csv in private/ = id + labels only | ✅ id, label |
| Deterministic (same output every run) | ✅ train_test_split(..., random_state=42) |
| No duplicate rows between train and test | ✅ assert train_ids.isdisjoint(test_ids) |
| Uses only Kaggle Docker libraries | ✅ pathlib, pandas, sklearn |
| Raw data only (no pre-split public/private uploaded) | ✅ |

---

## dataset and challenge.txt — Challenge checklist

| Rule | Status |
|------|--------|
| Problem description complete and unambiguous | ✅ Overview, Evaluation, Dataset, Submission |
| Agent could solve without clarifying questions | ✅ 4 classes, metric, format, 6k rows specified |
| Evaluation metric clearly defined (formula for custom metric) | ✅ Inverse-frequency-weighted macro F1 described |
| Submission format precisely specified | ✅ id, label; 6,000 rows; 4 valid labels |
| prepare.py produces correct splits and is deterministic | ✅ |
| grade.py scores valid submissions and handles edge cases | ✅ try/except, validation, returns float |
| 5+ initial rubrics, majority REQUIRED or RECOMMENDED | ✅ 2 REQUIRED, 4 RECOMMENDED, 1 UNIVERSAL |
| Rubrics specific to this task, not generic | ✅ Nexari, protocol tokens, resolution state, imbalance |

---

## dataset and challenge.txt — Grading script contract

| Rule | Status |
|------|--------|
| grade(submission, answers) -> float | ✅ |
| Args docstring (submission, answers) | ✅ |
| Returns docstring (float score) | ✅ |
| Raises docstring (invalid format) | ✅ |
| Validate submission format | ✅ columns, unique ids, length, label values |
| Calculate and return score | ✅ frequency-weighted macro F1, float(score) |

---

## dataset and challenge.txt — More rules (sections 1–8)

| # | Rule | Status |
|---|------|--------|
| 1 | Avoid very common datasets | ✅ Synthetic Nexari; no public benchmark |
| 2 | Avoid too trivial (agent ~0.9 or R² > 0.98) | ✅ 4-class, weighted F1; not trivial |
| 3 | Licences (synthetic: N/A) | ✅ |
| 4 | No data leakage (no test stats in prepare) | ✅ Split only; no cross-train-test stats |
| 5 | prepare.py structure (train/test/answers) | ✅ |
| 6 | Avoid vague/broken rubrics; correct importance | ✅ Measurable criteria; REQUIRED = broken without |
| 7 | Novelty ≥ 2 | ✅ Novelty 3 (Medium) |
| 8 | grade.py: try/except + set(submission.id)==set(answers.id) | ✅ |

---

## prev_reviews.txt

| Issue | Status |
|-------|--------|
| No duplicates between train and test | ✅ prepare asserts disjoint ids |
| Problem description sample counts match actual (24k / 6k) | ✅ |
| Grader requires the column used for scoring (label) | ✅ Must have id, label; invalid labels rejected |
| No same sample with different labels | ✅ One label per id; documented in dataset Notes |
| merge(..., how='left') | ✅ answers.merge(submission, on='id', how='left') |
| Formatting not all in one code block | ✅ Sections and lists used |
| Novelty (target ≥ 3) | ✅ 3/5 Medium |
| Grader: unique id submission, unique id answers, length check | ✅ |
| Grader: prediction/label constraints | ✅ VALID_LABELS check (classification) |
| prepare: anonymizing / generic feature names | ✅ id, transcript, num_turns, sector, label |
| Cheating: single-row or missing rows must not get good score | ✅ Length check + id set check; wrong/missing rows raise |

---

## Rubric 5 — Fix on platform

**Problem:** The criterion field for Rubric 5 may show "UALITY | REQUIRED **Criterion:**" at the start (copy-paste artifact).

**Fix:** Edit Rubric 5 so the **Criterion** field contains only this (no prefix):

```
Submission CSV has columns id and label with exactly one row per test id, no duplicate ids, and all labels are one of the 4 valid resolution states.
```

Delete any leading "UALITY | REQUIRED **Criterion:**" or "CODE_QUALITY | REQUIRED **Criterion:**" from the criterion text. The **Type** (Code Quality) and **Importance** (Required) are set separately; the criterion should be a short, concise description only (per rules: "Rubric criterion field used as a heading instead of a concise description" is a bad pattern to avoid).
