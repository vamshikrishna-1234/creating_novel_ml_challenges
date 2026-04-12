# Dataset: Code-Switched Complaint Triage (Synthetic)

**Title (for platform):** Code-Switched Complaint Triage — Resolution Hours (Synthetic)

---

## Overview

This dataset is **synthetically generated** (no real users or systems). It simulates customer support complaints written in a **code-switched** style: English mixed with tokens from a fictional constructed language ("Verani"). Each row is one complaint; the goal is to predict **resolution time in hours** (continuous target).

The data is produced by the included reproducible script `generate.py` (default seed 42, 28,000 rows). Relationships are designed to be learnable from text and category (e.g. priority tags [P1]–[P5], code-switch density, severity cues) but not trivial, so the benchmark has discriminative value.

**License:** Synthetic data; no third-party sources. Effectively public domain / unrestricted for use and redistribution. No URL required (generation process documented below).

---

## File Structure

- `data.csv` — single file of labeled data (id, text, category, target)

No other files in the raw dataset. The prepare script splits this into public/private splits.

---

## Features

| Column   | Type   | Description |
|----------|--------|-------------|
| id       | int    | Unique row identifier (0 to N-1). |
| text     | string | Code-switched complaint text (English + Verani tokens). May contain priority tags like [P1]–[P5]. |
| category | string | Product/area category. One of: `billing`, `authentication`, `data-pipeline`, `ui-frontend`, `api-gateway`, `storage`, `notifications`, `analytics`. |
| target   | float  | Resolution time in hours. Range 0.5–168. Two decimal places. |

---

## Notes

- **Reproducibility:** Run `python generate.py --output data.csv --seed 42 --size 28000` to regenerate. Only standard library + no external data.
- **Verani:** Fictional language tokens (e.g. shaluma, voshka, bretal) are mixed into sentences; resolution time is partly driven by code-switch density and embedded signals (priority, severity phrasing).
- **No same sample with different labels:** Each `id` appears exactly once with one `target`. No duplicate texts in the generator output.
- **Train/test split:** The challenge prepare script splits by row (e.g. 80/20) with a fixed seed so there are **no overlapping ids** between train and test.
