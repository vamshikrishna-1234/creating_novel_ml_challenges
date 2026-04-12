# Rules & Prev-Reviews Verification Checklist

Use this when submitting the dataset and challenge to ensure nothing is missed.

---

## Dataset (intro and datasets / dataset and challenge)

| Rule | Status | How we comply |
|------|--------|----------------|
| Dataset description has Overview, File Structure, Features table, Notes | ✅ | DATASET_DESCRIPTION.md |
| All columns and types documented | ✅ | id (int), text (string), category (string), target (float) |
| Sufficiently many samples | ✅ | 28,000 rows |
| Prepare script reproducible and deterministic | ✅ | prepare.py uses random_state=42 |
| Synthetic: document generation process + reproducible script | ✅ | generate.py included; Notes describe command |
| Realistic distributions and relationships | ✅ | Target 0.5–168, code-switch and category drive target |
| License / source | ✅ | Synthetic, no URL; stated in description |
| No very common dataset | ✅ | Synthetic, fictional Verani language; no overlap with public data |
| Not too trivial (R² > 0.98 / score ~0.9) | ✅ | Multi-signal target; wide distribution (std ~52) |
| Data leakage avoided | ✅ | No test stats in prepare; split by row; no target-derived features in raw |

---

## Prepare script

| Rule | Status | How we comply |
|------|--------|---------------|
| train.csv = id + features + labels | ✅ | id, text, category, target |
| test.csv = id + features only | ✅ | id, text, category |
| answers.csv = id + labels only | ✅ | id, target |
| sample_submission.csv in public/ | ✅ | id, prediction |
| Deterministic (same output every run) | ✅ | train_test_split(..., random_state=42) |
| No duplicate rows between train and test | ✅ | Assert train_ids.isdisjoint(test_ids) |
| Upload raw only (not pre-split public/private) | ✅ | We upload data.csv; platform runs prepare |

---

## Grader (prev_reviews)

| Rule | Status | How we comply |
|------|--------|---------------|
| Wrap in try/except | ✅ | grade.py has try/except; re-raises ValueError/RuntimeError |
| merge(..., how='left') | ✅ | answers.merge(submission, on='id', how='left') |
| set(submission.id) == set(answers.id) | ✅ | Explicit check; clear error if mismatch |
| Enforce unique id in submission | ✅ | submission['id'].duplicated().any() → raise |
| Enforce unique id in answers | ✅ | answers['id'].duplicated().any() → raise |
| Explicit length check | ✅ | len(submission) != len(answers) → raise |
| Prediction column required (no exploit) | ✅ | Must have column 'prediction'; no alternate column accepted |
| Reasonable prediction range | ✅ | Predictions clipped to [0, 200] for scoring (target in 0.5–168) |

---

## Problem description & challenge

| Rule | Status | How we comply |
|------|--------|---------------|
| Sample counts match actual data | ✅ | "Exactly 22,400" train, "5,600" test |
| Complete and unambiguous | ✅ | CHALLENGE_DESCRIPTION.md |
| Evaluation metric defined (RMSE) | ✅ | Formula and direction (minimize) |
| Submission format precisely specified | ✅ | id + prediction; exactly 5,600 rows |
| Not everything in one code block | ✅ | Sections and tables used; code only for formula |
| Theoretical min/max set in config | ✅ | Min 0, max e.g. 200 for RMSE |

---

## Rubrics

| Rule | Status | How we comply |
|------|--------|---------------|
| 5+ rubrics | ✅ | 7 rubrics in RUBRICS.md |
| Majority REQUIRED or RECOMMENDED | ✅ | 2 REQUIRED, 5 RECOMMENDED, 1 UNIVERSAL |
| Task-specific, not generic | ✅ | Code-switch, category, resolution hours, format |
| Not vague, not tool-specific, not "attempts to solve" | ✅ | Concrete criteria (e.g. "uses both text and category") |

---

## Prev_reviews — explicit “do not repeat”

| Issue | Avoided? |
|-------|----------|
| Duplicates between train and test | ✅ Prepare splits with no overlapping ids; assertion in place. |
| Problem description says 200k/20k but data is 50k/5k | ✅ Description says 22,400 and 5,600; matches prepare output. |
| Omitting a column gives perfect score | ✅ Grader requires `prediction`; no alternate column. |
| Same training sample with different labels | ✅ Generator: one row per id, one target; noted in dataset Notes. |
| merge(..., how='inner') | ✅ We use how='left'. |
| Everything in one code block | ✅ Description uses sections and separate blocks. |
| Novelty very low | ✅ Code-switched regression on synthetic Verani; unusual task. |
| Grader: no unique id check, no length check, no range | ✅ All added in grade.py. |
| prepare.py: no anonymizing feature names | ✅ Raw columns are generic: id, text, category, target. |

---

## Quick copy-paste for platform

**Dataset title:**  
Code-Switched Complaint Triage — Resolution Hours (Synthetic)

**Challenge title:**  
PREDICT RESOLUTION HOURS FROM CODE-SWITCHED SUPPORT COMPLAINTS

**Grading:** Minimize. Theoretical min: 0. Theoretical max: 200 (or as per platform).  
**Tags (suggested):** text, feature-engineering (and any platform-specific tags).
