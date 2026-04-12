# Challenge creation form — fill-in (novel_challenge_4)

Tie this challenge to the **accepted dataset**: ORBIT Supply-Chain Risk Memos — Disruption Severity Tier (Synthetic).

---

## 1) Difficulty

**Select:** **Medium**

---

## 2) Challenge Title

```
NLP Severity Prediction from Obfuscated Supply-Chain Risk Memos
```

---

## 3) Problem Description

```markdown
# NLP Severity Prediction from Obfuscated Supply-Chain Risk Memos

## Overview

This is an **NLP** task requiring **natural language understanding** of heavily obfuscated, multi-section risk memos written in English from a simulated supply-network monitoring system. Each memo follows the **ORBIT protocol** (Operational Risk Briefing & Impact Taxonomy) — a fictional structured reporting format with typed section headers and natural language narrative.

Your goal is to build an **NLP pipeline** that reads, parses, and interprets these text documents to predict a severity label. Each memo describes a potential supply disruption and contains 2–5 ORBIT sections:

- **[ORBIT:VECTOR]** — identifies the disruption source with an anonymized risk code (`ORB-CODE`) and a narrative description. In ~55% of memos the narrative text is redacted (`[narrative redacted]`). In surviving narratives, 1–3 words may be randomly dropped.
- **[ORBIT:IMPACT]** — reports shortfall and cycle counts replaced by single tokens (`<QTY>`, `<PERIOD>`) and a masked downstream effect (`[MASKED]`).
- **[ORBIT:MITIGATION]** (sometimes present, ~50%) — describes countermeasures with a masked effectiveness rating (`[MASKED]`).
- **[ORBIT:CASCADE]** (sometimes present, ~50%) — describes propagation depth (`<DEPTH>`) and a masked rate (`[MASKED]`).
- **[ORBIT:NOTE]** (sometimes present, ~20%) — analyst notes containing tier-irrelevant boilerplate text (distractor section).

**Important preprocessing already applied:**
- The `[ORBIT:VERDICT]` section (which explicitly states the tier) has been **removed**.
- All specific risk codes (ORB-V1…V7, ORB-I1…I7, ORB-M1…M5) are **masked** to `ORB-CODE`.
- All numeric values (shortfall amounts, cycle counts, cascade depth) are replaced with **single placeholder tokens** (`<QTY>`, `<PERIOD>`, `<DEPTH>`).
- Categorical severity terms (effectiveness ratings, cascade rates, downstream impact) are **masked** to `[MASKED]`.
- ~55% of memos have the VECTOR narrative **redacted**; ~30% of surviving narratives have 1–3 words randomly **dropped**.
- ~35% of memos have their section order **shuffled**.
- ~30% of rows have the `region` column **missing** (NaN).
- ~20% of rows have the `commodity_class` column **missing** (NaN).

**Why this is hard:** Narrative templates are shared across tiers (the same phrasing can appear at any severity level), numeric values are fully masked, and most narratives are redacted. The remaining signal is subtle and distributed across multiple weak cues — no single feature determines the tier.

Your task is to predict the **disruption severity tier** for each text memo. There are 5 tiers (ordinal):

- **tier_1_minor** (~28%) — routine fluctuation, negligible downstream impact
- **tier_2_moderate** (~24%) — notable disruption, low-to-moderate downstream effect
- **tier_3_significant** (~22%) — material disruption requiring intervention
- **tier_4_severe** (~16%) — major disruption with cascading effects
- **tier_5_critical** (~10%, minority) — catastrophic failure, emergency response

Submissions are scored using **quadratic-weighted Cohen's Kappa**. This metric is designed for ordinal classification: it penalizes predictions that are far from the true tier (e.g. predicting tier_1 when truth is tier_5) much more than close misclassifications (tier_3 vs tier_4). **Higher is better.**

## Evaluation

**Metric:** Quadratic-weighted Cohen's Kappa.

Cohen's Kappa measures agreement between predicted and true labels, adjusted for chance. The quadratic weighting assigns a cost proportional to the squared distance between the predicted and true tier index (0–4), making it ideal for ordinal tasks.

**Grading direction:** Maximize.

**Theoretical minimum:** -1 (systematic disagreement, worse than chance)
**Theoretical maximum:** 1 (perfect agreement)

## Dataset (prepared)

**In public/:**

- **train.csv** — id, memo, num_sections, region, commodity_class, label. Exactly 25,600 rows (stratified 80% split).
- **test.csv** — id, memo, num_sections, region, commodity_class. Exactly 6,400 rows. No label column.
- **sample_submission.csv** — id, label. Example format with placeholder labels.

**In private/ (not visible to solvers):** answers.csv — id, label.

**Column descriptions:**

| Column          | Type   | Description |
|-----------------|--------|-------------|
| id              | int    | Unique memo identifier |
| memo            | string | Heavily obfuscated multi-section ORBIT risk memo. Risk codes masked to ORB-CODE, all numbers replaced with single tokens (<QTY>, <PERIOD>, <DEPTH>), categorical terms [MASKED], ~55% narratives redacted, ~30% of surviving narratives word-dropped, ~35% section order shuffled, distractor [ORBIT:NOTE] sections present in ~20%. |
| num_sections    | int    | Number of ORBIT sections in the memo (2–5) |
| region          | string | Fictional region. One of: norath, veldan, crymara, esthos, lunavar, torwen. ~30% missing (NaN). |
| commodity_class | string | Fictional commodity. One of: synth-ore, bio-compound, flux-crystal, plasma-stock, neuro-fiber, cryo-element, photon-alloy, gravity-mesh. ~20% missing (NaN). |
| label           | string | Severity tier. One of: tier_1_minor, tier_2_moderate, tier_3_significant, tier_4_severe, tier_5_critical |

## Submission

Submit a CSV with exactly these columns:

| Column | Type   | Description |
|--------|--------|-------------|
| id     | int    | Memo identifier from test.csv (one per test row) |
| label  | string | Predicted severity tier |

**Requirements:**

- Exactly **6,400 rows** (one per test memo). No duplicate ids.
- Header row required.
- Label values must be one of the 5 valid tiers listed above.

**Example of a correctly formatted submission (first few rows):**

```
id,label
50,tier_2_moderate
51,tier_4_severe
52,tier_1_minor
53,tier_3_significant
54,tier_5_critical
...
```

(Your file must have 6,400 rows total, one per test id, with no duplicates.)
```

---

## 4) Tags

**Select:** **text**

---

## 5) Grading Configuration

**Grade direction:** **Maximize**

**Theoretical minimum:** **-1**

**Theoretical maximum:** **1**

---

## 6) Grading Script (Custom)

```python
import pandas as pd
import numpy as np


VALID_LABELS = {
    "tier_1_minor", "tier_2_moderate", "tier_3_significant",
    "tier_4_severe", "tier_5_critical",
}

LABEL_ORDER = [
    "tier_1_minor", "tier_2_moderate", "tier_3_significant",
    "tier_4_severe", "tier_5_critical",
]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission using quadratic-weighted Cohen's Kappa.

    Args:
        submission: The agent's predictions (loaded from submission.csv). Columns: id, label.
        answers: Ground truth labels (loaded from private/answers.csv). Columns: id, label.

    Returns:
        A float score in [-1, 1]. Direction: maximize.

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

        if submission["label"].isna().any():
            raise ValueError("Submission has missing (NaN) label values")

        sub_labels = set(submission["label"].dropna().unique())
        invalid = sub_labels - VALID_LABELS
        if invalid:
            raise ValueError(f"Invalid label values: {invalid}. Must be one of {VALID_LABELS}")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["label_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        y_true = merged["label_true"].values
        y_pred = merged["label_pred"].values

        label_to_idx = {lab: i for i, lab in enumerate(LABEL_ORDER)}
        y_true_idx = np.array([label_to_idx[l] for l in y_true])
        y_pred_idx = np.array([label_to_idx[l] for l in y_pred])

        if np.ptp(y_pred_idx) == 0:
            return float(0.0)

        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(y_true_idx, y_pred_idx, weights="quadratic")

        if np.isnan(kappa):
            return float(0.0)

        return float(kappa)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Data Preparation Pipeline

**Input:** raw dataset file **data.csv** (id, memo, region, commodity_class, label).

**Script — prepare.py:**

*(See prepare.py file — updated v2 with aggressive obfuscation. Key changes from v1: all numbers → single tokens, 55% narrative redaction, 30% word-drop on survivors, 35% section shuffle, 30% missing region, 20% missing commodity, 50% NOTE section stripping.)*

Run **Run Prepare** after pasting.

---

## 8) Evaluation Rubrics

Add each via "Add Rubric":

**1** — DATA_HANDLING | REQUIRED
**Criterion:** Loads and uses the `memo` column from train/test; does not rely solely on `region`, `commodity_class`, or `num_sections`.
**Rationale:** The memo text is the primary input containing ORBIT risk codes, shortfall estimates, and cascade information. Ignoring it means the model cannot learn the structured cues critical to severity prediction.

**2** — FEATURE_ENGINEERING | RECOMMENDED
**Criterion:** Extracts structured features from the obfuscated memo (e.g. section presence patterns, narrative redaction status, [ORBIT:NOTE] distractor detection) and combines them with text features.
**Rationale:** After aggressive obfuscation (all numbers masked to single tokens, most narratives redacted), remaining signal is distributed across weak structural cues; combining multiple features is essential.

**3** — FEATURE_ENGINEERING | RECOMMENDED
**Criterion:** Handles missing metadata gracefully — ~30% of rows have region as NaN, ~20% have commodity_class as NaN. Uses imputation or treats missingness as a feature rather than dropping rows.
**Rationale:** Dropping rows with missing metadata would lose a significant fraction of training data; missingness itself may carry weak signal.

**4** — TRAINING | RECOMMENDED
**Criterion:** Uses a proper train/validation split or cross-validation with stratification for model selection; does not tune or select models using the test set.
**Rationale:** Class imbalance (tier_5_critical ~10%) means unstratified splits can misrepresent minority class performance.

**5** — CODE_QUALITY | REQUIRED
**Criterion:** Submission CSV has columns `id` and `label` with exactly one row per test id, no duplicate ids, and all labels are one of the 5 valid severity tiers.
**Rationale:** Grader expects this format and valid labels; wrong format or invalid labels cause grading failure.

**6** — DATA_HANDLING | RECOMMENDED
**Criterion:** Accounts for the class imbalance (tier_5_critical ~10% vs tier_1_minor ~28%) using techniques such as stratified sampling, class weights, oversampling, or ordinal-aware loss functions.
**Rationale:** The grading metric (quadratic-weighted Kappa) penalizes distant misclassifications; ignoring imbalance risks systematically misclassifying the minority tiers.

**7** — UNIVERSAL | UNIVERSAL
**Criterion:** Does not use test set or test labels for training, feature computation, or normalization.
**Rationale:** Universal anti-leakage criterion.

---

## 9) Agent Evaluation Runs

No fill; runs on submit.

---

## Checklist

- [ ] Dataset: ORBIT Supply-Chain Risk Memos — Disruption Severity Tier (Synthetic) accepted and selected
- [ ] Difficulty: Medium
- [ ] Title: NLP Severity Prediction from Obfuscated Supply-Chain Risk Memos
- [ ] Problem description: 25,600 / 6,400 rows, 5 tiers, quadratic-weighted Cohen's Kappa, maximize, min -1 max 1, inline submission example
- [ ] Tags: text
- [ ] Grading: Maximize; min -1; max 1
- [ ] Grading script: quadratic Cohen's Kappa (sklearn); merge left; id/length/label validation; try/except; constant predictions return 0.0
- [ ] Prepare (v2): strip VERDICT/REF/NOTE(50%); mask codes; ALL numbers→single tokens; mask categorical terms; redact ~55% narratives; word-drop ~30% survivors; shuffle ~35% sections; missing region ~30%; missing commodity ~20%; stratified 80/20 split; Run Prepare succeeded
- [ ] 7 rubrics added (2 REQUIRED, 4 RECOMMENDED, 1 UNIVERSAL)
