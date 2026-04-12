# Final verification — rules/ and prev_reviews

Novelty: **3** (confirmed by you). Below: every condition from **rules/prev_reviews.txt** and **rules/dataset and challenge.txt** checked.

---

## prev_reviews.txt — every point

| # | Issue from prev_reviews | Status | Where it's addressed |
|---|-------------------------|--------|----------------------|
| 1 | Duplicates between train.csv and test.csv | ✅ | prepare.py: `train_test_split(..., random_state=42)`; assert `train_ids.isdisjoint(test_ids)`. Each row in exactly one split. |
| 2 | Problem description says wrong sample counts (e.g. 200k/20k vs actual 50k/5k) | ✅ | CHALLENGE_FORM_FILL: "Exactly 22,400 rows" train, "Exactly 5,600 rows" test. Matches prepare output (28k×0.8, 28k×0.2). |
| 3 | Grader allows omitting a column (e.g. probability) and getting perfect score | ✅ | grade.py: requires columns `id` and `prediction`; raises if either missing. No alternate column accepted. |
| 4 | Same training sample with different labels | ✅ | Dataset Notes: "One label per row: Each id appears exactly once with one target. No duplicate texts." Generator enforces one row per id. |
| 5 | merge(submission, on="id", how="inner") — should be left | ✅ | grade.py line 59: `answers.merge(submission, on="id", how="left")`. |
| 6 | Everything inside one code block (formatting) | ✅ | Problem description uses separate H1/H2 sections (Overview, Evaluation, Dataset, Submission), lists, and table — not one big code block. |
| 7 | Novelty very low (get to 3) | ✅ | You confirmed novelty is 3. Code-switched regression on synthetic Verani is non-generic. |
| 8 | Max score issue (unbounded RMSE / where does 10000 come from) | ✅ | Grading config: theoretical max **200** (documented as practical display cap). No arbitrary 10000. |
| 9 | Grader: no unique id in submission | ✅ | grade.py: `submission["id"].duplicated().any()` → raise. |
| 10 | Grader: no unique id in answers | ✅ | grade.py: `answers["id"].duplicated().any()` → raise. |
| 11 | Grader: no explicit length check | ✅ | grade.py: `len(submission) != len(answers)` → raise. |
| 12 | Grader: no constraints on prediction range | ✅ | grade.py: predictions clipped to [0, 200] for scoring (target 0.5–168). |
| 13 | prepare.py: no effort anonymizing raw feature names | ✅ | Raw columns are generic: id, text, category, target (no leaky or overly specific names). |
| 14 | Grader crashes on malformed submission | ✅ | grade.py: full logic in try/except; ValueError re-raised; other Exception → RuntimeError. |
| 15 | set(submission.id) == set(answers.id) | ✅ | grade.py: `sub_ids != ans_ids` check with clear errors for missing/extra ids. |

**Prev_reviews: all 15 points satisfied.**

---

## dataset and challenge.txt — Dataset

| Requirement | Status |
|-------------|--------|
| Dataset description: Overview, File Structure, Features table, Notes | ✅ DATASET_FORM_FILL.md |
| All columns and types documented | ✅ id, text, category, target with types and descriptions |
| Sufficiently many samples | ✅ 28,000 |
| Prepare script reproducible and deterministic | ✅ random_state=42 in train_test_split |
| Synthetic: document generation process + reproducible script | ✅ generate.py, command in Notes |
| Realistic distributions and relationships | ✅ Target 0.5–168, code-switch/category drive target |
| Controlled complexity | ✅ Code-switch density, priority tags, noise in generator |
| License/source (synthetic: no URL needed) | ✅ Stated in dataset description and License field |

---

## dataset and challenge.txt — Prepare script

| Requirement | Status |
|-------------|--------|
| Deterministic (same output every run) | ✅ random_state=42 |
| train.csv = id + features + labels | ✅ id, text, category, target |
| test.csv = id + features only | ✅ id, text, category |
| sample_submission.csv in public/ | ✅ id, prediction (placeholder) |
| answers.csv in private/ = id + labels only | ✅ id, target |
| Uses only Kaggle Docker libraries | ✅ pathlib, pandas, sklearn |
| No stats computed across train+test (no leakage) | ✅ Single split; no test-set stats |

---

## dataset and challenge.txt — Challenge checklist

| Requirement | Status |
|-------------|--------|
| Problem description complete and unambiguous | ✅ Overview, Evaluation, Dataset, Submission with exact row counts |
| Agent could solve without clarifying questions | ✅ Task, metric, format, and row counts specified |
| Evaluation metric clearly defined (RMSE formula) | ✅ Formula and "Minimize" in problem description |
| Submission format precisely specified | ✅ Columns id, prediction; exactly 5,600 rows; header |
| prepare.py produces correct splits and is deterministic | ✅ See above |
| grade.py scores valid submissions and handles edge cases | ✅ Validation + try/except; clear errors |
| 5+ initial rubrics, majority REQUIRED or RECOMMENDED | ✅ 7 rubrics: 2 REQUIRED, 5 RECOMMENDED, 1 UNIVERSAL |
| Rubrics specific to this task, not generic | ✅ Code-switch, category, resolution hours, UTF-8, format |

---

## dataset and challenge.txt — More rules (sections 1–8)

| # | Rule | Status |
|---|------|--------|
| 1 | Avoid very common datasets | ✅ Synthetic Verani; no public benchmark overlap |
| 2 | Avoid too trivial (agent ~0.9 or R² > 0.98) | ✅ No single trivial feature; noisy target; code-switch requires NLP |
| 3 | Licences (synthetic: N/A; no commercial restrictions) | ✅ Stated |
| 4 | Data leakage (no test stats in prepare, no target-derived features) | ✅ Split only; no cross-train-test stats; ids only for row identity |
| 5 | prepare.py structure (train/test/answers format) | ✅ See prepare section above |
| 6 | Avoid vague/broken rubrics; correct importance tags | ✅ Rubrics are measurable; REQUIRED = broken without; RECOMMENDED = task-specific |
| 7 | Novelty (≥2; you have 3) | ✅ Confirmed |
| 8 | grade.py: try/except + set(submission.id)==set(answers.id) | ✅ Done |

---

## Rubrics (dataset and challenge.txt)

| Property | Status |
|----------|--------|
| 5+ rubrics | ✅ 7 |
| Specificity (this dataset/task) | ✅ Category, code-switch, priority tags, resolution hours, UTF-8 |
| Balance (SHOULD / SHOULD NOT) | ✅ e.g. "does not tune on test set", "does not drop category" |
| Approach-neutral | ✅ No "must use XGBoost"; e.g. "at least one text-derived feature beyond BoW" |
| Discrimination (separate good from poor) | ✅ Criteria are verifiable and non-trivial |
| Not mostly UNIVERSAL | ✅ Only 1 UNIVERSAL; rest DATA_HANDLING, FEATURE_ENGINEERING, TRAINING, MODELING, CODE_QUALITY |

---

## Summary

- **prev_reviews.txt:** All 15 points satisfied (no duplicates, correct counts, required columns, left merge, formatting, novelty 3, max 200, grader validation, prepare column names).
- **dataset and challenge.txt:** Dataset description and checklist, prepare structure and determinism, challenge checklist, more rules 1–8, and rubrics all met.

If anything on the platform differs from this (e.g. different prepare/grade script or problem text), align the form with the files in `novel_challenge/` and this checklist.
