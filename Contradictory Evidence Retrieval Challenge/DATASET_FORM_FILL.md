# Dataset creation form — fill-in

## 1. Title (Required)

```
Multi-Stance Claim Verification Passage Corpus (Synthetic)
```

---

## 2. Description (Required)

Paste the following directly into the rich-text description editor (not inside a code block):

---

# Dataset Description

## Overview

This dataset is fully synthetic, created to benchmark retrieval-augmented generation (RAG) systems on claim verification tasks. It simulates a scientific literature review scenario: given a claim about a fictional substance's effect on a material property, a system must analyze a corpus of evidence passages and produce a structured verdict — determining the overall stance (support, contradict, or insufficient), identifying which passages constitute genuine evidence, and assessing confidence.

The dataset was designed to test whether models can distinguish genuine evidence from topically similar but irrelevant text, and whether they can weigh conflicting evidence to reach a correct conclusion. No real institutions, substances, studies, or scientific data are used — all content is procedurally generated from fictional vocabulary pools.

The passages in each row fall into five categories (not labeled in the data):
- **Support passages** — directly confirm the claim with experimental evidence.
- **Contradict passages** — directly refute the claim with experimental evidence.
- **Near-miss distractors** — mention the same substance but a different effect, or the same effect but a different substance. They appear topically relevant but do not constitute evidence for or against the specific claim.
- **Hedged passages** — mention the correct substance and effect but use uncertain, inconclusive, or preliminary language. They do not constitute reliable evidence.
- **Pure distractors** — discuss a different substance and a different effect entirely.

**Dataset Summary:**

- **Claim:** A single sentence asserting that a fictional substance improves a specific material property under specific environmental conditions.
- **Passages:** 8–15 numbered passages per row, each from a fictional institution. Passage types are intentionally mixed and unlabeled.
- **Near-miss distractors:** Share the substance OR effect with the claim (but not both), making them appear topically relevant.
- **Hedged passages:** Mention the correct substance and effect but use uncertain language (e.g., "might influence", "sample sizes were insufficient").
- **Structured verdict:** 3-field output: `STANCE:<stance> | EVIDENCE_IDS:<ids> | CONFIDENCE:<conf>`

**Current raw dataset stats:**

- Total rows: 20,000
- Columns: id, claim, passages, verdict
- Stance distribution: support ~37%, contradict ~38%, insufficient ~25%
- Confidence distribution: high ~17%, medium ~29%, low ~54%
- Passages per row: 8–15
- Evidence IDs per row: 0–8 (mean ~4.9)

## File Structure

- `data.csv` — labeled data (id, claim, passages, verdict). Single CSV file, 20,000 rows.

## Features

### Columns

| Column   | Type   | Description |
|----------|--------|-------------|
| id       | int    | Unique row identifier (0 to 19,999). |
| claim    | string | A single sentence asserting that a fictional substance improves a specific material property under specific environmental conditions. Example: "Compound-X7 improves thermal stability under high-pressure chamber conditions." The claim always names one substance (from a pool of 18), one material property (from a pool of 18), and one environment (from a pool of 12). |
| passages | string | A multi-line free-text field containing 8–15 evidence passages separated by newlines. Each passage is prefixed with a passage identifier in the format `[PN]` (e.g., `[P1]`, `[P2]`). Each passage is a single sentence describing a fictional experimental finding attributed to a fictional institution and year. Passages vary in their relevance to the claim — some directly address the claim's substance and effect, some mention only one of the two, and some discuss entirely unrelated topics. The passage text is unstructured natural language; there are no separate sub-columns for institution, year, method, etc. |
| verdict  | string | A pipe-delimited string with exactly 3 named fields separated by ` | `. Format: `STANCE:<value> | EVIDENCE_IDS:<value> | CONFIDENCE:<value>`. See field descriptions below. |

### Verdict Field Descriptions

The `verdict` column contains a single string with three fields. These are the only structured sub-components in the dataset:

| Field        | Possible Values | Description |
|--------------|-----------------|-------------|
| STANCE       | support, contradict, insufficient | Overall stance of the evidence toward the claim. "support" if more passages support than contradict; "contradict" if more contradict than support; "insufficient" if counts are tied or no genuine evidence exists. |
| EVIDENCE_IDS | Comma-separated integers, or "none" | The 1-indexed passage IDs of all passages that directly support or contradict the claim (sorted ascending). Passages that are topically similar but not direct evidence are excluded. "none" if no genuine evidence passages exist. |
| CONFIDENCE   | high, medium, low | A confidence level derived from how strongly the evidence leans one way. Higher imbalance between supporting and contradicting evidence yields higher confidence. |

## Notes

- **Synthetic origin:** All content is procedurally generated. Substance names, institution names, effects, environments, and methods are drawn from fixed fictional vocabulary pools. No real scientific data is used.
- **Passage diversity:** Each row's passage corpus intentionally mixes passages of varying relevance — from directly relevant evidence to topically similar but irrelevant text to completely unrelated content. This tests a model's ability to distinguish genuine evidence from noise.
- **Balanced distributions:** Rows are generated from 8 different evidence composition profiles to ensure diverse stance and confidence distributions across the dataset. Approximately 37% of rows have a "support" stance, 38% "contradict", and 25% "insufficient".
- **One verdict per row:** Each id appears exactly once. No duplicate rows.

---

---

## 3. Data Files (Required)

Upload `data.csv`. Single file, no zip or extra folders.

---

## 4. Source Files / Import from URL

Leave empty (synthetic).

---

## 5. License & Source

```
Synthetic dataset. Generated by the reproducible script generate.py. No third-party data. No license restrictions.
```
