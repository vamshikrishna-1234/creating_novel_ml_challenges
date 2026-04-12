# Dataset creation form — fill-in

## 1. Title (Required)

```
Multi-Source Compliance Inspection Resolution Dataset (Synthetic)
```

---

## 2. Description (Required)

Paste the following directly into the rich-text description editor (not inside a code block):

---

# Dataset Description

## Overview

This dataset is fully synthetic (no real facilities, inspectors, or regulations). It represents multi-source environmental compliance inspection reports for fictional facilities. Each row contains 2–4 independent inspector fragments describing the same facility visit, a per-row compliance codebook (4–12 rules in natural language), and a structured verdict string derived by applying the codebook rules to reconcile the fragments.

The task is sequence-to-sequence: given the inspector fragments and codebook rules as input text, produce the exact 6-field structured verdict string.

**Dataset Summary:**

- **Multi-source input:** Each row has 2–4 inspector fragments that may contradict each other (different violations reported, some fragments partially corrupted).
- **Variable codebook:** Each row has a different subset of rules (from a pool of ~10 rule types) that govern contradiction resolution, severity escalation, action assignment, and penalty mapping.
- **Chained inference:** Later verdict fields depend on earlier derived fields (e.g., PENALTY depends on ACTION which depends on SEVERITY which depends on reconciled VIOLATIONS).
- **6-field structured output:** `FACILITY:X | VIOLATIONS:Y | COUNT:Z | SEVERITY:W | ACTION:A | PENALTY:P`
- **Reproducibility:** Data is produced by `generate.py` (default seed 42, 20,000 rows). Python standard library only.

**Current raw dataset stats:**

- Total rows: 20,000
- Columns: id, input, verdict
- Severity distribution: negligible ~17%, low ~12%, moderate ~12%, elevated ~14%, high ~15%, critical ~31%
- Action distribution: no_action ~28%, others ~12% each
- Input length: 725–1,946 characters

## File Structure

- `data.csv` — labeled data (id, input, verdict). Single CSV file, 20,000 rows.
- `generate.py` — reproducible Python script that generates `data.csv`. Uses only the Python standard library.

## Features

### Top-Level Columns

| Column  | Type   | Description |
|---------|--------|-------------|
| id      | int    | Unique row identifier (0 to 19,999) |
| input   | string | Multi-line text containing two sections: inspector fragments and a compliance codebook. See sub-structure below. |
| verdict | string | Pipe-delimited structured verdict with 6 named fields. See sub-structure below. |

### `input` Column Sub-Structure

The `input` column contains two concatenated sections separated by a blank line:

**Section 1 — Inspector Fragments (2–4 per row):**

Each fragment is delimited by a header line `--- Fragment N (Inspector Name) ---` followed by 1–2 lines of natural language notes. Sub-components:

| Sub-component       | Type   | Description |
|---------------------|--------|-------------|
| inspector           | string | Inspector name (e.g., "Chen", "Okafor"). One of 16 fictional names. |
| violations_found    | list of strings | Violation types mentioned in this fragment's notes (e.g., "leak", "emission_excess"). Drawn from a pool of 15 violation types. May differ across inspectors for the same facility (intentional contradictions). |
| sector              | string | Facility sector referenced in the notes. One of: alpha, beta, gamma, delta, epsilon, zeta. |
| corruption marker   | string | If present, a `[CORRUPTED]` span replaces part of the notes text, and a line `[Note: partial data corruption detected in this fragment]` is appended. ~15% of fragments are corrupted. |

**Section 2 — Compliance Codebook (4–15 rules per row):**

Begins with the header `=== Compliance Codebook ===` followed by numbered rules (`Rule 1:`, `Rule 2:`, etc.). Each rule is a single natural-language sentence describing one of the following operations:

| Rule Type                  | Description |
|----------------------------|-------------|
| Any-inspector-flags        | If any inspector reports a specific violation type, include it as confirmed. |
| Majority-vote              | Include a violation type only if a majority of inspectors report it. |
| Count escalation           | If total violation count exceeds a threshold, escalate severity by N levels. |
| Sector severity boost      | If facility sector matches a specific value, escalate severity by N levels. |
| Corrupted fallback         | If any fragment is corrupted, set severity to at least a specified level. |
| No-violations override     | If no violations remain after reconciliation, reset severity/action/penalty to defaults. |
| Violation-implies-severity | If a specific violation is confirmed, severity must be at least a specified level. |
| Severity-to-action         | If severity is at or above a threshold, set action to a specified value. |
| Action-to-penalty          | If action matches a specified value, set penalty to a specified tier. |
| Penalty cap                | Penalty cannot exceed a specified tier. |

### `verdict` Column Sub-Structure

The verdict is a pipe-delimited string with exactly 6 named fields:

| Field      | Type    | Possible Values | Description |
|------------|---------|-----------------|-------------|
| FACILITY   | string  | e.g., KRX-0447, VLN-1234 | Facility ID composed of a 3-letter prefix and 4-digit number. 16 possible prefixes. |
| VIOLATIONS | string  | Comma-separated list of violation type names, or "none" | Confirmed violations after applying reconciliation rules (any-flags / majority-vote) to inspector fragments. Sorted alphabetically. |
| COUNT      | int     | 0–7 | Number of confirmed violations (length of the VIOLATIONS list). |
| SEVERITY   | string  | negligible, low, moderate, elevated, high, critical | Severity level after applying all escalation rules (count-based, sector-based, corruption fallback, violation-implies-severity). |
| ACTION     | string  | no_action, log_only, reinspect_30d, reinspect_7d, immediate_halt, partial_shutdown, full_shutdown | Action derived from severity via severity-to-action mapping rules. Later rules override earlier ones. |
| PENALTY    | string  | none, tier_A, tier_B, tier_C, tier_D, tier_E | Penalty tier derived from action via action-to-penalty mapping rules, possibly capped by a penalty-cap rule. |

## Notes

- **Reproducibility:** Run `python generate.py --output data.csv --seed 42 --size 20000` to regenerate the dataset identically.
- **Contradictions:** Inspector fragments deliberately disagree — codebook rules specify how to reconcile (any-inspector-flags vs majority-vote).
- **Corruption:** ~15% of fragments have `[CORRUPTED]` spans; codebook may specify fallback severity for corrupted data.
- **Chained dependencies:** Rules are applied in order; severity→action→penalty is a chain where errors cascade.
- **One verdict per row:** Each id appears exactly once. No duplicate rows.

---

---

## 3. Data Files (Required)

Upload **only** `data.csv` (raw). No zip with extra parent folder.

---

## 4. Source Files / Import from URL

Leave empty (synthetic).

---

## 5. License & Source

```
Synthetic dataset. Generated by the reproducible script generate.py. No third-party data. No license restrictions.
```
