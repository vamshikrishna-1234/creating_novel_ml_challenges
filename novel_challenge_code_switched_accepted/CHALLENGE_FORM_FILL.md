# Challenge creation form — fill-in (create_template)

Copy each section below into the corresponding field on the challenge creation page. The challenge must use the **accepted dataset** (Code-Switched Complaint Triage — Resolution Hours) you already created.

---

## 1) Difficulty (Required)

**Select:** **Medium**

*(Design challenges that require using pre-trained models for significantly better payouts — Medium fits code-switched NLP with room for better models.)*

---

## 2) Challenge Title (Required)

**Paste this (clear, descriptive, capital letters):**

```
PREDICT RESOLUTION HOURS FROM CODE-SWITCHED SUPPORT COMPLAINTS
```

---

## 3) Problem Description (Required)

*The challenge prompt agents will see. Must be complete and unambiguous. Use the editor’s headings (H1, H2), lists, and code blocks as below.*

**Paste this in the Problem Description field:**

```markdown
# Predict Resolution Hours from Code-Switched Support Complaints

## Overview

You are given code-switched customer support complaints: each document mixes English with tokens from a fictional language (Verani). Your task is to predict the **resolution time in hours** (continuous regression) for each complaint.

The prepared dataset provides training data (complaint text, category, and resolution hours) and test data (complaint text and category only). You must produce a CSV of predictions for each test row, one prediction per `id`. Submissions are scored with **Root Mean Squared Error (RMSE)**; **lower is better**.

Real-world context: triage systems that estimate handling time from mixed-language or code-switched tickets can improve routing and SLA planning. This benchmark uses synthetic data to test NLP and feature-engineering skills.

## Evaluation

Submissions are scored using **RMSE (Root Mean Squared Error)**. Lower is better.

**Formula:**

RMSE = sqrt( mean( (y_true - y_pred)² ) )

**Grading direction:** Minimize.

## Dataset (prepared)

After the prepare script runs, solvers see:

**In public/:**

- **train.csv** — id, text, category, target. Exactly 22,400 rows.
- **test.csv** — id, text, category. Exactly 5,600 rows. No target column.
- **sample_submission.csv** — id, prediction. Example format; prediction values are placeholders.

**In private/ (not visible to solvers):** answers.csv — id, target. Ground truth for test set only.

**Column types:** id (int), text (string), category (string), target (float, resolution hours).

## Submission

Submit a single CSV file with exactly these columns:

| Column     | Type  | Description |
|------------|-------|-------------|
| id         | int   | Row identifier from test.csv (one per test row). |
| prediction | float | Predicted resolution time in hours. |

**Requirements:**

- Must contain exactly **5,600 rows** (one per test sample). No duplicate ids.
- Must include header row.
- Column names must be `id` and `prediction`. The grader requires `prediction` to exist and be numeric.
```

*(If the editor uses rich text instead of markdown, use H1 for the main title, H2 for Overview / Evaluation / Dataset / Submission, and bullet lists and the table as shown.)*

---

## 4) Tags

**Select one or more:** **text**, **feature-engineering**

*(Optional: small-data if your platform treats ~28k rows as small-data.)*

---

## 5) Grading Configuration (Required)

**Grade direction:** **Minimize**

**Theoretical minimum:** **0**

**Theoretical maximum:** **200**

*(RMSE has no strict upper bound; 200 is a practical display cap. Target in data is 0.5–168 hours.)*

---

## 6) Grading Script (Required)

**Choose:** **Custom** (do not use the template as-is; paste the script below).

**Paste this entire script into the Grading Script field:**

```python
import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score regression predictions (resolution hours) using Root Mean Squared Error.
    Lower is better.
    """
    try:
        if "id" not in submission.columns or "prediction" not in submission.columns:
            raise ValueError("Submission must have columns: id, prediction")

        if submission["id"].duplicated().any():
            raise ValueError("Submission must not contain duplicate id values")

        if answers["id"].duplicated().any():
            raise ValueError("Answers file must not contain duplicate id values")

        if len(submission) != len(answers):
            raise ValueError(
                f"Submission must have exactly {len(answers)} rows (one per test id), got {len(submission)}"
            )

        sub_ids = set(submission["id"])
        ans_ids = set(answers["id"])
        if sub_ids != ans_ids:
            missing = ans_ids - sub_ids
            extra = sub_ids - ans_ids
            if missing:
                raise ValueError(f"Submission missing ids: {len(missing)} ids (e.g. {list(missing)[:5]})")
            if extra:
                raise ValueError(f"Submission has extra ids not in test set: {len(extra)} ids")

        merged = answers.merge(submission, on="id", how="left")

        if merged["prediction"].isna().any():
            raise ValueError("Submission has missing (NaN) predictions for some rows")

        pred = pd.to_numeric(merged["prediction"], errors="coerce")
        if pred.isna().any():
            raise ValueError("All predictions must be numeric")

        pred = pred.clip(lower=0.0, upper=200.0)

        rmse = np.sqrt(((merged["target"] - pred) ** 2).mean())
        return float(rmse)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Data Preparation Pipeline

**Input — raw/:**  
The dataset you selected provides the raw file. Ensure the raw input is **data.csv** (single file, 4 columns: id, text, category, target). The platform will pass the contents of the dataset as the `raw` argument.

**Script — prepare.py:**  
Choose **Custom** (or “Train-test split — from a single data.csv” and then replace with the script below).

**Paste this entire script into the prepare.py editor:**

```python
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)

    required_cols = {"id", "text", "category", "target"}
    if set(df.columns) != required_cols:
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(test_ids), "Train and test must not share any ids"

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(public / "train.csv", index=False)

    test_public = test_df[["id", "text", "category"]].copy()
    test_public.to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    sample["prediction"] = 0.0
    sample.to_csv(public / "sample_submission.csv", index=False)

    answers = test_df[["id", "target"]].copy()
    answers.to_csv(private / "answers.csv", index=False)
```

**Then run “Run Prepare”** to confirm the pipeline produces public/ (train.csv, test.csv, sample_submission.csv) and private/ (answers.csv).

---

## 8) Evaluation Rubrics

*Aim for 5–10 specific criteria. Add each rubric via “Add Rubric” and fill Type, Importance, Criterion, and Rationale.*

**Rubric 1**  
- **Type:** DATA_HANDLING  
- **Importance:** REQUIRED  
- **Criterion:** Loads and uses both `text` and `category` from train/test (no dropping category).  
- **Rationale:** Category is a strong signal for resolution time; ignoring it leaves performance on the table and is effectively broken for this task.

**Rubric 2**  
- **Type:** FEATURE_ENGINEERING  
- **Importance:** RECOMMENDED  
- **Criterion:** Uses at least one text-derived feature beyond raw bag-of-words (e.g. length, code-switch ratio, presence of priority tags [P1]–[P5], or embeddings).  
- **Rationale:** The target depends on code-switch density and embedded tags; models that only use naive BoW will plateau.

**Rubric 3**  
- **Type:** TRAINING  
- **Importance:** RECOMMENDED  
- **Criterion:** Uses a proper train/validation split or cross-validation for model selection or early stopping; does not tune or select models using the test set.  
- **Rationale:** Prevents overfitting and test leakage; resolution-time distribution is wide so validation strategy matters.

**Rubric 4**  
- **Type:** MODELING  
- **Importance:** RECOMMENDED  
- **Criterion:** Uses a regression formulation (predicts continuous hours) and does not treat the task as classification (e.g. binning into classes only).  
- **Rationale:** Evaluation is RMSE on continuous values; discretizing to few classes loses information and hurts score.

**Rubric 5**  
- **Type:** CODE_QUALITY  
- **Importance:** REQUIRED  
- **Criterion:** Submission CSV has exactly columns `id` and `prediction`, with one row per test id and no duplicate ids.  
- **Rationale:** Grader expects this format; wrong format causes grading failure (broken solution).

**Rubric 6**  
- **Type:** DATA_HANDLING  
- **Importance:** RECOMMENDED  
- **Criterion:** Handles text encoding (UTF-8) and does not drop or corrupt rows with special characters or Verani tokens.  
- **Rationale:** Code-switched text contains non-ASCII tokens; broken encoding leads to missing or wrong features.

**Rubric 7**  
- **Type:** UNIVERSAL  
- **Importance:** UNIVERSAL  
- **Criterion:** Does not use test set (or test labels) for training, feature computation, or normalization.  
- **Rationale:** Universal anti-leakage criterion; applies to any supervised challenge.

---

## 9) Agent Evaluation Runs

No form fill. Agent evaluations run automatically when you submit the challenge for review.

---

## Checklist before submitting

- [ ] Difficulty: Medium  
- [ ] Challenge Title: PREDICT RESOLUTION HOURS FROM CODE-SWITCHED SUPPORT COMPLAINTS  
- [ ] Problem Description: pasted (Overview, Evaluation, Dataset, Submission; correct row counts 22,400 / 5,600)  
- [ ] Tags: text, feature-engineering  
- [ ] Grading: Minimize; min 0; max 200  
- [ ] Grading Script: full custom grade.py pasted  
- [ ] Data Preparation: raw = data.csv; prepare.py pasted; “Run Prepare” succeeded  
- [ ] Rubrics: 7 rubrics added (2 REQUIRED, 5 RECOMMENDED, 1 UNIVERSAL)  
- [ ] Challenge is tied to the accepted dataset (Code-Switched Complaint Triage — Resolution Hours)
