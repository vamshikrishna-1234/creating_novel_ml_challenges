# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Temporal Knowledge Base With Source Credibility Dynamics.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Temporal Evidence Verification
```

---

## 3) Problem Description

# Synthetic Temporal Evidence Verification

## Overview

This is a **RAG** challenge requiring **claim verification against a temporal knowledge base with dynamic source credibility, contradictions, retractions, and adversarial sources**. The problem models a scenario where factual claims must be verified using a corpus of dated documents from multiple sources, but sources disagree, some sources are unreliable, corrections may themselves be wrong, and the evidence available depends on a temporal cutoff.

The knowledge base contains 600 anonymized documents spanning the years 2010–2025, authored by 30 coded sources across 8 topic categories. Documents describe events, people, and locations in a completely fictional domain — all entity names are opaque coded identifiers (e.g., `LOC-007`, `PER-012`, `EVT-003`), so no pre-trained world knowledge can help. Some documents explicitly correct or retract earlier documents. Sources have hidden topic-dependent credibility that changes over time.

For each claim, an "as-of" year is specified. Only documents published in or before that year are admissible as evidence. The task is to predict the verification verdict: one of 4 coded classes representing the claim's truth status given the available temporal evidence.

**What makes this problem uniquely challenging:**

- **Topic-dependent source credibility**: a source may be highly reliable on one topic but consistently wrong on another. Source reliability cannot be assessed globally — it must be estimated per topic from training data patterns.
- **Temporal credibility decay**: source reliability changes over time. A source accurate in early years may become unreliable later (or vice versa). The "as-of" date determines both which documents are visible AND the effective credibility of each source at that time.
- **Temporal traps**: later documents are NOT always more reliable. Some "corrections" introduce new errors that are subsequently re-corrected by even later documents. Naive "trust the newest" strategies fail.
- **Adversarial sources**: approximately 5 sources have high apparent credibility but periodically inject factual errors. These sources appear trustworthy across most documents, making their errors hard to detect.
- **Copycat sources**: approximately 15% of sources replicate content from a hidden "parent" source. When multiple sources agree, it may reflect genuine consensus OR correlated noise from a single underlying source.
- **Sparse evidence**: approximately 10% of claims have 0–2 relevant documents, forcing verification under extreme uncertainty.
- **Multi-step reasoning**: verification requires retrieving relevant documents by matching topic and entity references, filtering by the as-of date, assessing source credibility, resolving contradictions, and producing a final verdict — a multi-layer reasoning chain.
- **Irreducible noise**: ~12% of verdicts are randomly assigned, capping theoretical Macro F1 at approximately 0.88.

Your task: for each test claim, given the claim text, its as-of year, and the full knowledge base, predict the verification verdict.

## Evaluation

Submissions are scored using **Macro F1** across the following 4 verdict classes:

| Code | Meaning |
|------|---------|
| V-A | Claim is supported by the available evidence |
| V-B | Claim is refuted by the available evidence |
| V-C | Claim is partially true (some aspects supported, others not) |
| V-D | Insufficient evidence to determine the claim's truth status |

Macro F1 computes the F1 score for each of the 4 classes independently and averages them, giving equal weight to minority classes. **Higher is better.** Minimum: 0.0, Maximum: 1.0.

## Dataset

- `knowledge_base.csv` — 600 documents: document_id (int), year (int, 2010–2025), source_name (string, coded source e.g. SRC-007), topic (string, coded category e.g. CAT-03), location (string, coded e.g. LOC-012), person (string, coded e.g. PER-005), event (string, coded e.g. EVT-009), fact_number (int), text (string, full document content with coded entity references), relevance_score (float), confidence_index (float)
- `train_claims.csv` — 3,900 training claims: claim_id (int), claim_text (string, the factual assertion with coded entity references), as_of_year (int), verdict (string, one of V-A / V-B / V-C / V-D)
- `test_claims.csv` — 1,300 test claims: claim_id (int), claim_text (string), as_of_year (int) — verdict withheld
- `sample_submission.csv` — 1,300 rows: claim_id (int), verdict (string, baseline constant prediction). Shows the required submission format.

### Feature Details

| Column | Type | Description |
|--------|------|-------------|
| document_id | int | Unique document identifier |
| year | int | Document publication year (2010–2025) |
| source_name | string | Coded source identifier (SRC-000 through SRC-029) |
| topic | string | Coded topic category (CAT-00 through CAT-07) |
| location | string | Coded location (LOC-000 through LOC-015) |
| person | string | Coded person (PER-000 through PER-015) |
| event | string | Coded event (EVT-000 through EVT-015) |
| fact_number | int | Reference number cited in the document |
| text | string | Full free-text document content with all entities replaced by coded identifiers |
| relevance_score | float | Numeric document feature (0–1 range) |
| confidence_index | float | Numeric document feature (0–1 range) |
| claim_id | int | Unique claim identifier |
| claim_text | string | The factual assertion to verify, with coded entity references |
| as_of_year | int | Temporal cutoff year — only documents published on or before this year are admissible |
| verdict | string | Coded verification verdict (one of V-A, V-B, V-C, V-D) — train only |

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| claim_id | int | Claim identifier from test_claims.csv |
| verdict | string | Predicted verdict code (one of: V-A, V-B, V-C, V-D) |

**Requirements:**
- Must contain exactly 1,300 rows (one per test claim).
- Include a header row.
- Each verdict value must be one of the 4 valid codes: `V-A`, `V-B`, `V-C`, `V-D`.

**Example of a correctly formatted submission file:**

claim_id,verdict
5,V-A
12,V-C
18,V-B
23,V-D
31,V-A
42,V-C
57,V-D
68,V-B

---

## 4) Tags

**Select:** `text`, `feature-engineering`

---

## 5) Grading Configuration

- **Grade direction:** **Maximize**
- **Theoretical minimum:** `0`
- **Theoretical maximum:** `1`

---

## 6) Grading Script

**Select:** `Custom`

```python
import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if "claim_id" not in submission.columns or "verdict" not in submission.columns:
            return 0.0

        if submission["claim_id"].duplicated().any():
            return 0.0
        if answers["claim_id"].duplicated().any():
            return 0.0

        if set(submission["claim_id"]) != set(answers["claim_id"]) or len(submission) != len(answers):
            return 0.0

        merged = answers.merge(
            submission, on="claim_id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        col_true = "verdict_true" if "verdict_true" in merged.columns else "verdict"
        col_pred = "verdict_pred" if "verdict_pred" in merged.columns else "verdict"

        if merged[col_pred].isna().any():
            return 0.0

        y_true = merged[col_true].astype(str).str.strip()
        y_pred = merged[col_pred].astype(str).str.strip()

        all_classes = sorted(set(y_true.tolist()))

        f1_scores = []
        for cls in all_classes:
            tp = ((y_pred == cls) & (y_true == cls)).sum()
            fp = ((y_pred == cls) & (y_true != cls)).sum()
            fn = ((y_pred != cls) & (y_true == cls)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        if np.isnan(macro_f1):
            return 0.0
        return macro_f1

    except Exception:
        return 0.0
```

---

## 7) Prepare Script

```python
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path) -> None:
    import random as _rnd
    import numpy as np
    import pandas as pd

    raw, public, private = Path(raw), Path(public), Path(private)

    kb = pd.read_csv(raw / "knowledge_base.csv")
    claims = pd.read_csv(raw / "claims.csv")

    rng = _rnd.Random(299792458)
    np_rng = np.random.RandomState(65)

    unique_sources = sorted(kb["source_name"].unique().tolist())
    source_codes = [f"SRC-{i:03d}" for i in range(len(unique_sources))]
    rng.shuffle(source_codes)
    source_map = {unique_sources[i]: source_codes[i] for i in range(len(unique_sources))}
    kb["source_name"] = kb["source_name"].map(source_map)

    unique_locs = sorted(kb["location"].unique().tolist())
    loc_codes = [f"LOC-{i:03d}" for i in range(len(unique_locs))]
    rng.shuffle(loc_codes)
    loc_map = {unique_locs[i]: loc_codes[i] for i in range(len(unique_locs))}
    kb["location"] = kb["location"].map(loc_map)

    unique_people = sorted(kb["person"].unique().tolist())
    person_codes = [f"PER-{i:03d}" for i in range(len(unique_people))]
    rng.shuffle(person_codes)
    person_map = {unique_people[i]: person_codes[i] for i in range(len(unique_people))}
    kb["person"] = kb["person"].map(person_map)

    unique_events = sorted(kb["event"].unique().tolist())
    event_codes = [f"EVT-{i:03d}" for i in range(len(unique_events))]
    rng.shuffle(event_codes)
    event_map = {unique_events[i]: event_codes[i] for i in range(len(unique_events))}
    kb["event"] = kb["event"].map(event_map)

    unique_topics = sorted(kb["topic"].unique().tolist())
    topic_codes = [f"CAT-{i:02d}" for i in range(len(unique_topics))]
    rng.shuffle(topic_codes)
    topic_map = {unique_topics[i]: topic_codes[i] for i in range(len(unique_topics))}
    kb["topic"] = kb["topic"].map(topic_map)

    def _obfuscate_text(text):
        result = str(text)
        for orig, code in sorted(source_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(loc_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(person_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(event_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(topic_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        return result

    kb["text"] = kb["text"].apply(_obfuscate_text)
    claims["claim_text"] = claims["claim_text"].apply(_obfuscate_text)

    verdict_map = {
        "SUPPORTED": "V-A",
        "REFUTED": "V-B",
        "PARTIALLY_TRUE": "V-C",
        "INSUFFICIENT_EVIDENCE": "V-D",
    }
    claims["verdict"] = claims["verdict"].map(verdict_map)

    kb = kb.rename(columns={"doc_id": "document_id"})
    doc_ids = sorted(kb["document_id"].tolist())
    rng_docids = _rnd.Random(173205080)
    rng_docids.shuffle(doc_ids)
    doc_id_map = {old: new for old, new in zip(sorted(kb["document_id"].tolist()), doc_ids)}
    kb["document_id"] = kb["document_id"].map(doc_id_map)
    kb = kb.sort_values("document_id").reset_index(drop=True)

    n_docs = len(kb)
    kb["relevance_score"] = np_rng.uniform(0, 1, n_docs).round(3)
    kb["confidence_index"] = np_rng.normal(0.5, 0.2, n_docs).clip(0, 1).round(3)

    all_claim_ids = sorted(claims["claim_id"].tolist())
    rng_split = _rnd.Random(141421356)
    rng_split.shuffle(all_claim_ids)

    split_idx = int(len(all_claim_ids) * 0.75)
    train_ids = set(all_claim_ids[:split_idx])
    test_ids = set(all_claim_ids[split_idx:])

    train_claims = claims[claims["claim_id"].isin(train_ids)].copy()
    test_claims = claims[claims["claim_id"].isin(test_ids)].copy()

    train_claims = train_claims.sort_values("claim_id").reset_index(drop=True)
    test_claims = test_claims.sort_values("claim_id").reset_index(drop=True)

    sample_sub = test_claims[["claim_id"]].copy()
    mode_verdict = train_claims["verdict"].mode().iloc[0]
    sample_sub["verdict"] = mode_verdict

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    kb.to_csv(public / "knowledge_base.csv", index=False)
    train_claims[["claim_id", "claim_text", "as_of_year", "verdict"]].to_csv(
        public / "train_claims.csv", index=False
    )
    test_claims[["claim_id", "claim_text", "as_of_year"]].to_csv(
        public / "test_claims.csv", index=False
    )
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    test_claims[["claim_id", "verdict"]].to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly loads and parses the knowledge_base.csv and the claim CSV files, handling the multi-file dataset structure without data corruption.
- **Rationale:** The challenge requires joining information across multiple files. Failure to load any file correctly makes prediction impossible.

### Rubric 2
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Respects the temporal constraint: for each claim, only considers documents with year <= the claim's as_of_year when forming evidence.
- **Rationale:** Using documents published after the as-of date constitutes temporal leakage and invalidates the verification task.

### Rubric 3
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 1,300 rows, columns claim_id and verdict, where each verdict is one of the valid coded labels from the training data.
- **Rationale:** Missing columns, wrong row counts, or invalid verdict codes will score zero.

### Rubric 4
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves Macro F1 above the constant-prediction baseline (~0.18) on the test set, producing class-specific predictions rather than a single constant for all claims.
- **Rationale:** Predicting the most common verdict for every claim is trivially achievable and demonstrates no useful modeling.

### Rubric 5
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Predictions span at least 3 of the 4 verdict classes in the submission, rather than predicting a single class for all test claims.
- **Rationale:** Macro F1 gives equal weight to all classes. Predicting only one class yields zero F1 on the other three classes.

### Rubric 6
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Criteria:** Prediction quality does not catastrophically degrade for claims where the as_of_year is early (2012–2015), resulting in a much smaller evidence pool.
- **Rationale:** Early-date claims have fewer available documents and higher uncertainty. A robust solution handles varying evidence density.

### Rubric 7
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Evaluates intermediate predictions on a held-out portion of training claims before generating final test predictions.
- **Rationale:** Without internal validation, there is no way to assess whether the approach generalizes before submission.

### Rubric 8
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not access test set ground-truth labels or leak private answer data into the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces inflated scores that do not reflect genuine capability.
