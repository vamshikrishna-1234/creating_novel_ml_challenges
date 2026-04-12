# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Synthetic Branching Procedural Execution Traces.

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Synthetic Branching Protocol Outcome Prediction
```

---

## 3) Problem Description

# Synthetic Branching Protocol Outcome Prediction

## Overview

This is a **Sequence to Sequence** challenge requiring **structured outcome prediction from procedural free-text descriptions with conditional execution paths**. The problem models a scenario where coded biochemical processing protocols — described as sequences of steps with conditional branches, side effects, and interference steps — are executed under specific initial conditions, and the task is to predict the structured 4-field execution outcome.

The dataset contains 8,000 unique protocols in a fictional processing domain. Each protocol is a directed acyclic graph of 8–15 steps described in free text using anonymized codes for reagents (e.g., `RX017`), equipment (e.g., `EQ003`), and phase states (e.g., `PH010`). Protocols contain conditional branches where the execution path depends on the current state of internal process variables. Steps have side effects that modify variables used by later conditional checks, creating non-local dependencies. Some steps are no-operations disguised as real processing steps, and some are low-impact interference steps that add noise to internal variables.

Each protocol is paired with a set of 5 initial condition variables. These variables propagate through the protocol's DAG, affecting which branches are taken and how steps transform the internal state. The same protocol text executed with different initial conditions produces different outcomes.

**What makes this problem uniquely challenging:**

- **Shuffled step ordering**: steps within each protocol description are NOT in execution order. Steps reference each other by coded step-IDs (e.g., `X32`, `X41`), and the correct execution sequence must be reconstructed from these references. Sequential reading of the protocol text does not reflect the actual execution flow.
- **Non-local side effects**: executing an early step modifies internal state variables that affect conditional branches many steps later. Predicting outcomes requires tracing variable propagation through the entire DAG.
- **Latent execution factors**: a hidden variable (not present in the data) influences branch resolution at 1–2 decision points per protocol. The same visible protocol and initial conditions can produce different outcomes depending on this unobserved factor, creating irreducible prediction uncertainty.
- **No-operation and interference steps**: some steps are disguised no-ops that produce no effect, and some are low-impact interference steps. Identifying which steps actually matter requires understanding the execution semantics.
- **Unseen test protocols**: test protocols are entirely different from training protocols. No test protocol appears in the training set. The model must generalize execution-tracing ability to new procedural structures.
- **Test-time perturbation**: ~10% of test initial conditions are slightly perturbed, and ~5% of test protocols have injected noise steps and swapped step blocks.
- **Anonymized coded terminology**: all reagent, tool, phase, step-ID, and outcome codes are opaque identifiers with no semantic content. Pre-trained domain knowledge cannot help.

Your task: for each test protocol, given the protocol text and initial condition variables, predict all 4 structured outcome fields.

## Evaluation

Submissions are scored using **average per-field exact-match accuracy** across all 4 outcome fields. For each field, the fraction of test rows where the predicted value exactly matches the ground truth is computed, and the 4 per-field accuracies are averaged. **Higher is better.** Minimum: 0.0, Maximum: 1.0.

## Dataset

- `train.csv` — 18,000 training execution records (6,000 protocols × 3 condition sets): id (int), protocol_id (int), protocol_text (string, step descriptions separated by " ||| "), iv_a through iv_e (float, initial condition variables), feat_x1/feat_x2/feat_x3 (float, numeric features), terminal_state (string), primary_product (string), byproduct_class (string), process_status (string)
- `test.csv` — 2,000 test execution records (2,000 unseen protocols × 1 condition set each): same columns as train.csv except outcome fields are withheld
- `sample_submission.csv` — 2,000 rows: id (int) plus 4 outcome fields with baseline constant predictions. Shows the required submission format.

### Feature Details

| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique execution record identifier |
| protocol_id | int | Protocol identifier (each test protocol is unique and unseen in training) |
| protocol_text | string | Free-text protocol description with shuffled step ordering, steps separated by " \|\|\| " |
| iv_a | float | Initial condition variable A |
| iv_b | float | Initial condition variable B |
| iv_c | float | Initial condition variable C |
| iv_d | float | Initial condition variable D |
| iv_e | float | Initial condition variable E |
| feat_x1 | float | Numeric feature (standard-normal distributed) |
| feat_x2 | float | Numeric feature (uniformly distributed, -2 to 2) |
| feat_x3 | float | Numeric feature (positive, exponentially distributed) |
| terminal_state | string | Final execution state (1 of 20 coded identifiers, e.g. OT019) — train only |
| primary_product | string | Primary output product (1 of 30 coded identifiers, e.g. OP011) — train only |
| byproduct_class | string | Byproduct classification (1 of 10 coded identifiers, e.g. OB007) — train only |
| process_status | string | Execution status (1 of 5 coded identifiers, e.g. OP003) — train only |

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| id | int | Row identifier from test.csv |
| terminal_state | string | Predicted terminal state code |
| primary_product | string | Predicted primary product code |
| byproduct_class | string | Predicted byproduct class code |
| process_status | string | Predicted process status code |

**Requirements:**
- Must contain exactly 2,000 rows (one per test record).
- Include a header row.

**Example of a correctly formatted submission file:**

id,terminal_state,primary_product,byproduct_class,process_status
18000,OT005,OP012,OB003,OP001
18001,OT019,OP027,OB008,OP003
18002,OT001,OP004,OB005,OP002
18003,OT011,OP018,OB001,OP004
...

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
        FIELDS = ["terminal_state", "primary_product", "byproduct_class", "process_status"]

        if "id" not in submission.columns:
            return 0.0
        for f in FIELDS:
            if f not in submission.columns:
                return 0.0

        if submission["id"].duplicated().any():
            return 0.0
        if answers["id"].duplicated().any():
            return 0.0

        if set(submission["id"]) != set(answers["id"]) or len(submission) != len(answers):
            return 0.0

        merged = answers.merge(
            submission, on="id", how="left", suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            return 0.0

        field_accuracies = []
        for f in FIELDS:
            col_true = f"{f}_true" if f"{f}_true" in merged.columns else f
            col_pred = f"{f}_pred" if f"{f}_pred" in merged.columns else f

            if merged[col_pred].isna().any():
                field_accuracies.append(0.0)
                continue

            true_vals = merged[col_true].astype(str).str.strip()
            pred_vals = merged[col_pred].astype(str).str.strip()
            acc = (true_vals == pred_vals).mean()
            field_accuracies.append(float(acc))

        avg_accuracy = float(np.mean(field_accuracies))
        if np.isnan(avg_accuracy):
            return 0.0
        return avg_accuracy

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

    data = pd.read_csv(raw / "data.csv")

    rng = _rnd.Random(271828182)
    np_rng = np.random.RandomState(77)

    OUTCOME_FIELDS = ["terminal_state", "primary_product", "byproduct_class", "process_status"]

    field_maps = {}
    for f in OUTCOME_FIELDS:
        unique_vals = sorted(data[f].unique().tolist())
        n = len(unique_vals)
        codes = [f"O{f[0].upper()}{i:03d}" for i in range(n)]
        rng.shuffle(codes)
        mapping = {unique_vals[i]: codes[i] for i in range(n)}
        field_maps[f] = mapping
        data[f] = data[f].map(mapping)

    all_reagents = sorted(set())
    all_tools = sorted(set())
    all_states = sorted(set())
    for text in data["protocol_text"]:
        for token in text.split():
            clean = token.strip(".,;:[]()!?")
            if clean.startswith("Reagent-"):
                all_reagents.append(clean)
            elif clean.startswith("Unit-"):
                all_tools.append(clean)
            elif clean.startswith("Phase-"):
                all_states.append(clean)
    all_reagents = sorted(set(all_reagents))
    all_tools = sorted(set(all_tools))
    all_states = sorted(set(all_states))

    reagent_codes = [f"RX{i:03d}" for i in range(len(all_reagents))]
    rng.shuffle(reagent_codes)
    reagent_map = {all_reagents[i]: reagent_codes[i] for i in range(len(all_reagents))}

    tool_codes = [f"EQ{i:03d}" for i in range(len(all_tools))]
    rng.shuffle(tool_codes)
    tool_map = {all_tools[i]: tool_codes[i] for i in range(len(all_tools))}

    state_codes = [f"PH{i:03d}" for i in range(len(all_states))]
    rng.shuffle(state_codes)
    state_map = {all_states[i]: state_codes[i] for i in range(len(all_states))}

    step_id_codes = [f"X{i:02d}" for i in range(50)]
    rng.shuffle(step_id_codes)
    step_id_map = {f"S{i:02d}": step_id_codes[i] for i in range(50)}

    def _obfuscate_text(text):
        result = text
        for orig, code in sorted(reagent_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(tool_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(state_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        for orig, code in sorted(step_id_map.items(), key=lambda x: -len(x[0])):
            result = result.replace(orig, code)
        return result

    data["protocol_text"] = data["protocol_text"].apply(_obfuscate_text)

    col_rename = {
        "init_var_0": "iv_a", "init_var_1": "iv_b",
        "init_var_2": "iv_c", "init_var_3": "iv_d",
        "init_var_4": "iv_e",
    }
    data = data.rename(columns=col_rename)
    iv_cols = ["iv_a", "iv_b", "iv_c", "iv_d", "iv_e"]

    n = len(data)
    data["feat_x1"] = np_rng.normal(0, 1, n).round(3)
    data["feat_x2"] = np_rng.uniform(-2, 2, n).round(3)
    data["feat_x3"] = np_rng.exponential(1.5, n).round(3)

    all_pids = sorted(data["protocol_id"].unique().tolist())
    rng_split = _rnd.Random(314159265)
    rng_split.shuffle(all_pids)

    split_idx = int(len(all_pids) * 0.75)
    train_pids = set(all_pids[:split_idx])
    test_pids = set(all_pids[split_idx:])

    train_df = data[data["protocol_id"].isin(train_pids)].copy()
    test_df = data[data["protocol_id"].isin(test_pids)].copy()

    test_rows = []
    for pid in sorted(test_pids):
        pid_rows = test_df[test_df["protocol_id"] == pid]
        keep_idx = pid_rows.index[rng.randint(0, len(pid_rows) - 1)]
        test_rows.append(pid_rows.loc[[keep_idx]])
    test_df = pd.concat(test_rows, ignore_index=True)

    n_noise = int(len(test_df) * 0.05)
    noise_indices = sorted(rng.sample(range(len(test_df)), n_noise))
    for idx in noise_indices:
        text = test_df.at[idx, "protocol_text"]
        parts = text.split(" ||| ")
        if len(parts) >= 4:
            i, j = rng.sample(range(len(parts)), 2)
            parts[i], parts[j] = parts[j], parts[i]
            inject_step = rng.choice(list(step_id_map.values()))
            inject_reagent = rng.choice(reagent_codes)
            inject_tool = rng.choice(tool_codes)
            noise_step = f"[{inject_step}] Apply {inject_reagent} using {inject_tool} as supplementary treatment."
            insert_pos = rng.randint(0, len(parts))
            parts.insert(insert_pos, noise_step)
            test_df.at[idx, "protocol_text"] = " ||| ".join(parts)

    n_perturb = int(len(test_df) * 0.10)
    perturb_indices = sorted(rng.sample(range(len(test_df)), n_perturb))
    for idx in perturb_indices:
        for col in iv_cols:
            test_df.at[idx, col] = round(test_df.at[idx, col] + np_rng.normal(0, 0.3), 3)

    train_df = train_df.sort_values(["protocol_id", "condition_set"]).reset_index(drop=True)
    train_df["id"] = range(len(train_df))

    test_df = test_df.sort_values("protocol_id").reset_index(drop=True)
    test_df["id"] = range(len(train_df), len(train_df) + len(test_df))

    common_cols = ["id", "protocol_id", "protocol_text"] + iv_cols + ["feat_x1", "feat_x2", "feat_x3"]
    train_cols = common_cols + OUTCOME_FIELDS
    test_cols = common_cols

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df[train_cols].to_csv(public / "train.csv", index=False)
    test_df[test_cols].to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    for f in OUTCOME_FIELDS:
        mode_val = train_df[f].mode().iloc[0]
        sample[f] = mode_val
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id"] + OUTCOME_FIELDS].to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Correctly loads and parses the protocol_text column, handling the " ||| " step separator without data corruption or truncation.
- **Rationale:** Protocol descriptions contain multiple steps separated by a specific delimiter. Failure to parse correctly makes any downstream processing impossible.

### Rubric 2
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Handles the multi-file dataset structure (train.csv with outcomes, test.csv without outcomes) and produces predictions for all 2,000 test rows without missing any.
- **Rationale:** Incomplete predictions for any test row score zero for that row across all 4 fields.

### Rubric 3
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV with exactly 2,000 rows, columns id plus four outcome fields (terminal_state, primary_product, byproduct_class, process_status), where each prediction is a valid coded identifier from the training data.
- **Rationale:** Missing columns, wrong row counts, or invalid codes will score zero.

### Rubric 4
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves average per-field accuracy above the constant-prediction baseline (~0.24) on the test set, producing outcome-specific predictions rather than a single constant per field.
- **Rationale:** Predicting the most common training value for every field is trivially achievable and demonstrates no useful modeling.

### Rubric 5
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Generates predictions for all 4 outcome fields for every test row, not just a subset of fields.
- **Rationale:** The evaluation averages accuracy across all 4 fields. Leaving any field blank scores zero for that field.

### Rubric 6
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Criteria:** Prediction quality does not catastrophically degrade for test protocols where the protocol_text contains conditional branch instructions referencing multiple step-IDs.
- **Rationale:** Conditional branches are the central difficulty. A robust solution must handle branching protocols rather than only working on linear step sequences.

### Rubric 7
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Evaluates intermediate predictions on a held-out portion of training data before generating final test predictions.
- **Rationale:** Without internal validation, there is no way to assess whether the approach is working before submission.

### Rubric 8
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not access test set ground-truth labels or leak private answer data into the prediction pipeline.
- **Rationale:** Using ground-truth test labels produces inflated scores that do not reflect genuine capability.
