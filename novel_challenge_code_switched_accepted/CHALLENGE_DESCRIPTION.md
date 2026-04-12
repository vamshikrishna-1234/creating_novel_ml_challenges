# Challenge: Predict Resolution Hours from Code-Switched Complaints

**Challenge title (for platform):** PREDICT RESOLUTION HOURS FROM CODE-SWITCHED SUPPORT COMPLAINTS

**Difficulty:** Medium

---

## Overview

You are given **code-switched customer support complaints**: each document mixes English with tokens from a fictional language (Verani). Your task is to predict the **resolution time in hours** (continuous regression) for each complaint.

The prepared dataset provides training data (complaint text, category, and resolution hours) and test data (complaint text and category only). You must produce a CSV of predictions for each test row, one prediction per `id`. Submissions are scored with **Root Mean Squared Error (RMSE)**; **lower is better**.

Real-world context: triage systems that estimate handling time from mixed-language or code-switched tickets can improve routing and SLA planning. This benchmark uses synthetic data to avoid licensing issues while testing NLP and feature-engineering skills.

---

## Evaluation

Submissions are scored using **RMSE (Root Mean Squared Error)**. Lower is better.

**Grading direction:** Minimize.

**Formula:**

```text
RMSE = sqrt( mean( (y_true - y_pred)^2 ) )
```

**Theoretical minimum:** 0 (perfect predictions).  
**Theoretical maximum:** Unbounded; for config display you may set an upper bound (e.g. 200). The actual target range in the data is 0.5–168 hours.

---

## Dataset (prepared)

After the prepare script runs, solvers see:

**In `public/`:**

- **train.csv** — id, text, category, target. Exactly 22,400 rows (80% of 28,000).
- **test.csv** — id, text, category. Exactly 5,600 rows. No target column.
- **sample_submission.csv** — id, prediction. Example format; prediction values are placeholders.

**In `private/` (not visible to solvers):**

- **answers.csv** — id, target. Ground truth for test set only.

**Column types:** `id` (int), `text` (string), `category` (string), `target` (float, resolution hours).

---

## Submission

Submit a single CSV file with **exactly** these columns:

| Column     | Type  | Description |
|------------|-------|-------------|
| id         | int   | Row identifier from test.csv (one per test row). |
| prediction | float | Predicted resolution time in hours. |

**Requirements:**

- Must contain exactly **5,600 rows** (one per test sample). No duplicate ids.
- Must include header row.
- Column names must be `id` and `prediction`. No other columns are used for grading; extra columns are ignored but the grader requires `prediction` to exist and be numeric.

---

## Formatting note

Keep the problem description readable: use separate sections and code blocks only where needed (e.g. for the RMSE formula or submission table), not one single code block for the entire description.
