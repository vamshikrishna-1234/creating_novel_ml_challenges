# Dataset creation form — fill-in

## 1. Title (Required)

```
Synthetic Multi-Token Interaction Sequences
```

---

## 2. Description (Required)

# Dataset Description

## Overview

This dataset is fully synthetic, generated programmatically using the accompanying `generate.py` script (Python, seed=42). The generation process works as follows: 30 component tokens (C_00 through C_29) are created, each assigned three integer-valued dimensional properties via `random.randint`. Combinations of 3–5 tokens are then enumerated — all 4,060 possible 3-token combinations, plus random samples of 8,000 4-token and 3,000 5-token combinations. For each combination, a deterministic 3-part output code is computed by applying one fixed function per output dimension to the properties of the tokens in the combination. The three functions operate independently (each output digit depends on a different property dimension), and they include both linear and non-linear operations.

The dataset is intended to benchmark **compositional rule inference from symbolic sequences** — a task motivated by the open problem of compositional generalization in machine learning. Unlike natural-language compositional benchmarks that depend on linguistic structure, this dataset isolates the core difficulty: learning how individual component properties interact under fixed algebraic rules, and generalizing that knowledge to predict outcomes for combinations never seen during training.

The dataset contains 15,060 rows. Each row provides an input token sequence (3–5 component codes) and its corresponding 3-part output code (format: X-Y-Z).

## File Structure

- `data.csv` — All token combination sequences with output codes (15,060 rows)

## Features

### data.csv

| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique row identifier |
| input_tokens | string | Space-separated component token codes (3–5 tokens, e.g., "C_00 C_05 C_22") |
| output | string | 3-part result code in format X-Y-Z (e.g., "1-7-3") where X∈{0,1}, Y∈{0-9}, Z∈{0-4} |

## Notes

- Each combination of tokens produces a unique, deterministic output.
- Token codes (C_00 through C_29) are arbitrary identifiers with no semantic ordering.
- The output depends on hidden properties of the component tokens, not on token identity directly.
- All data is synthetically generated with a fixed random seed (42) for reproducibility.
- No real entities or processes are represented.

---

## 3. Data Files (Required)

Upload: `data.csv` and `generate.py`.

---

## 4. License

**Synthetic data — no external license needed.**

Select: `Other`
