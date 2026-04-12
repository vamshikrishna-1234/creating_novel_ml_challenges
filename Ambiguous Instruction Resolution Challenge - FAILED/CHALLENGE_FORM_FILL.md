# Challenge creation form — fill-in

Tie this challenge to the **accepted dataset**: Workplace Ambiguous Instruction Resolution Corpus (Synthetic).

---

## 1) Difficulty

**Select:** **Hard**

---

## 2) Challenge Title

```
Ambiguous Instruction Resolution From Workplace Scenarios
```

---

## 3) Problem Description

# Ambiguous Instruction Resolution From Workplace Scenarios

## Overview

This is an **NLP** challenge targeting **pragmatic ambiguity resolution** — a problem that existing NLI and reading-comprehension benchmarks largely ignore. Standard NLI tasks test entailment between sentence pairs; standard QA tasks retrieve factual spans. Neither requires resolving the kind of structural linguistic ambiguity that is pervasive in real professional communication: who does "them" refer to when multiple people are mentioned? Does "all reports and summaries" mean all of both, or all reports plus one summary? Does "reviewed the document with the errors" mean the reviewer used the errors, or the document contained the errors?

You are given workplace scenarios — each a multi-sentence description of a professional situation involving named (or anonymized) people, departments, and documents. Accompanying each scenario is an **ambiguous instruction** whose correct meaning can only be determined from the surrounding context. Your task is to select the correct interpretation from four candidate options (A, B, C, D).

The challenge covers **six distinct ambiguity categories** that co-occur in the same dataset without type labels: pronoun coreference across multiple antecedents, quantifier scope (universal vs. existential readings), prepositional-phrase attachment (instrument vs. attribute), verb-phrase ellipsis resolution, quantifier–negation interaction, and temporal event ordering from underspecified adverbial clauses. Each row is a unique entity configuration — no two rows share the same set of names, departments, or documents — so pattern memorization or lexical shortcuts cannot solve the task.

Additional obfuscation layers make the challenge harder: entity names may be replaced with anonymous codes (Person-1, Person-2), department names with generic labels (Dept-A, Dept-B), scenario sentences may be reordered, option wording paraphrased with synonyms, and irrelevant distractor sentences injected into scenarios. These transformations are applied non-uniformly across rows, so the model must be robust to varying levels of surface-form perturbation.

## Evaluation

Submissions are scored using **accuracy** (fraction of correctly predicted answer labels). **Higher is better.**

- **Minimum score:** 0.0 (no correct predictions)
- **Maximum score:** 1.0 (all predictions correct)
- **Random baseline:** ~0.25

## Dataset

The prepared data is split into:

- `train.csv` — 16,000 labeled rows
- `test.csv` — 4,000 unlabeled rows (no `answer` column)
- `sample_submission.csv` — example submission file

### Feature Details

| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique row identifier |
| scenario | string | A multi-sentence free-text workplace scenario (3–6 sentences). Describes a professional situation involving people (named or anonymized as Person-1, Person-2, etc.), departments (named or anonymized as Dept-A, Dept-B, etc.), documents, tasks, and contextual details. May contain irrelevant distractor sentences. |
| instruction | string | A single free-text sentence containing a linguistically ambiguous directive. The ambiguity may involve pronoun references, prepositional phrase attachment, elliptical constructions, quantifier–negation interactions, scope of modifiers, or temporal event ordering. |
| option_a | string | Candidate interpretation A — a full sentence restating one possible meaning of the instruction. |
| option_b | string | Candidate interpretation B — a full sentence restating one possible meaning of the instruction. |
| option_c | string | Candidate interpretation C — a full sentence restating one possible meaning of the instruction. |
| option_d | string | Candidate interpretation D — a full sentence restating one possible meaning of the instruction. |
| answer | string | (train.csv only) The correct option label: one of A, B, C, or D. |

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| id | int | Row identifier from test.csv |
| answer | string | Predicted label: one of A, B, C, or D |

**Requirements:**
- Must contain exactly 4,000 rows (one per test sample)
- Include header row
- Each `answer` value must be exactly one of: A, B, C, D

### Submission Example

id,answer
7,B
14,A
23,D
42,C
...

---

## 4) Tags

**Select:** `text`

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
        if "id" not in submission.columns or "answer" not in submission.columns:
            raise ValueError("Submission must have columns: id, answer")

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
                raise ValueError(f"Submission missing ids: {len(missing)}")
            if extra:
                raise ValueError(f"Submission has extra ids: {len(extra)}")

        merged = answers.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

        if merged["answer_pred"].isna().any():
            raise ValueError("Submission has missing predictions for some rows after merge")

        merged["answer_true"] = merged["answer_true"].astype(str).str.strip().str.upper()
        merged["answer_pred"] = merged["answer_pred"].astype(str).str.strip().str.upper()

        accuracy = (merged["answer_true"] == merged["answer_pred"]).mean()

        score = float(accuracy)
        if np.isnan(score):
            return 0.0
        return score

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Grading failed: {e}") from e
```

---

## 7) Prepare Script

```python
from pathlib import Path
import hashlib
import random as _rnd
import re

import pandas as pd
from sklearn.model_selection import train_test_split


REDHERRING_SENTENCES = [
    "The office Wi-Fi was unusually slow that morning.",
    "A facilities maintenance request had been submitted the previous week.",
    "The cafeteria was serving a new menu that day.",
    "Several employees had recently completed their annual training certification.",
    "The parking lot construction was expected to finish by month-end.",
    "A company-wide email about the updated dress code was sent on Monday.",
    "The IT department was rolling out new laptops to senior staff.",
    "The quarterly town hall was postponed due to a scheduling conflict.",
    "A fire drill had been conducted the previous afternoon.",
    "The building's elevator was under maintenance for routine inspection.",
    "New ergonomic chairs had been ordered for the open-plan area.",
    "The company newsletter featured an article about sustainability initiatives.",
]

PARAPHRASE_MAP = {
    "reviewed": ["examined", "assessed", "evaluated", "inspected"],
    "approved": ["authorized", "endorsed", "sanctioned", "greenlit"],
    "submitted": ["delivered", "filed", "turned in", "handed in"],
    "confirmed": ["verified", "acknowledged", "affirmed", "validated"],
    "completed": ["finished", "concluded", "wrapped up", "finalized"],
    "assigned": ["delegated", "allocated", "designated", "tasked"],
    "scheduled": ["arranged", "planned", "set up", "organized"],
    "prepared": ["drafted", "put together", "assembled", "compiled"],
    "discussed": ["addressed", "talked about", "deliberated on", "covered"],
    "distributed": ["circulated", "shared", "disseminated", "sent out"],
    "critical": ["crucial", "vital", "essential", "key"],
    "several": ["multiple", "a number of", "various", "numerous"],
    "together": ["jointly", "collaboratively", "as a team", "in tandem"],
    "First": ["Initially", "To begin with", "Starting off", "As a first step"],
    "Both": ["The two of them", "Each of them", "The pair"],
    "None": ["Not one", "Zero", "Not any"],
}

NAME_PATTERN = re.compile(
    r'\b(Priya|Marcus|Lena|Carlos|Yuki|Amara|Dmitri|Sofia|Kwame|Elena|'
    r'Raj|Ingrid|Hassan|Mei|Tomás|Fatima|Viktor|Chiara|Aiden|Nadia|'
    r'Kofi|Lucia|Jens|Zara|Oleg|Anya|Ravi|Sven|Layla|Esteban)\b'
)

DEPT_PATTERN = re.compile(
    r'\b(Engineering|Marketing|Operations|Finance|Legal|Research|Compliance|'
    r'Analytics|Procurement|Quality|Infrastructure|Product|Security|Support|Design)\b'
)


def _det_rng(row_id: int, salt: str) -> _rnd.Random:
    seed = int(hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest(), 16) % (2**32)
    return _rnd.Random(seed)


def _should_modify(row_id: int, salt: str, threshold: float) -> bool:
    h = hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < threshold


def _replace_names(text: str, name_map: dict) -> str:
    def replacer(m):
        return name_map.get(m.group(0), m.group(0))
    return NAME_PATTERN.sub(replacer, text)


def _replace_depts(text: str, dept_map: dict) -> str:
    def replacer(m):
        return dept_map.get(m.group(0), m.group(0))
    return DEPT_PATTERN.sub(replacer, text)


def _paraphrase(text: str, rng: _rnd.Random) -> str:
    for word, alts in PARAPHRASE_MAP.items():
        if word in text and rng.random() < 0.30:
            text = text.replace(word, rng.choice(alts), 1)
    return text


def _shuffle_scenario_sentences(scenario: str, rng: _rnd.Random) -> str:
    sentences = scenario.split('. ')
    if len(sentences) < 3:
        return scenario
    idx = rng.randint(1, len(sentences) - 2)
    sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
    return '. '.join(sentences)


def _truncate_scenario(scenario: str) -> str:
    sentences = scenario.split('. ')
    if len(sentences) > 2:
        return '. '.join(sentences[:-1]) + '.'
    return scenario


def _add_redherring(scenario: str, rng: _rnd.Random) -> str:
    herring = rng.choice(REDHERRING_SENTENCES)
    sentences = scenario.split('. ')
    if len(sentences) > 2:
        pos = rng.randint(1, len(sentences) - 1)
        sentences.insert(pos, herring)
    else:
        sentences.append(herring)
    return '. '.join(sentences)


def _obfuscate_row(row: dict, row_id: int) -> dict:
    scenario = row["scenario"]
    instruction = row["instruction"]
    options = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]

    if _should_modify(row_id, "names", 0.40):
        names_found = list(set(NAME_PATTERN.findall(scenario + " " + instruction)))
        rng = _det_rng(row_id, "names")
        rng.shuffle(names_found)
        name_map = {n: f"Person-{i+1}" for i, n in enumerate(names_found)}
        scenario = _replace_names(scenario, name_map)
        instruction = _replace_names(instruction, name_map)
        options = [_replace_names(o, name_map) for o in options]

    if _should_modify(row_id, "depts", 0.35):
        depts_found = list(set(DEPT_PATTERN.findall(scenario + " " + instruction)))
        rng = _det_rng(row_id, "depts")
        rng.shuffle(depts_found)
        dept_map = {d: f"Dept-{chr(65+i)}" for i, d in enumerate(depts_found)}
        scenario = _replace_depts(scenario, dept_map)
        instruction = _replace_depts(instruction, dept_map)
        options = [_replace_depts(o, dept_map) for o in options]

    if _should_modify(row_id, "shuffle", 0.30):
        rng = _det_rng(row_id, "shuffle")
        scenario = _shuffle_scenario_sentences(scenario, rng)

    if _should_modify(row_id, "para", 0.25):
        rng = _det_rng(row_id, "para")
        options = [_paraphrase(o, rng) for o in options]

    if _should_modify(row_id, "trunc", 0.15):
        scenario = _truncate_scenario(scenario)

    if _should_modify(row_id, "herring", 0.25):
        rng = _det_rng(row_id, "herring")
        scenario = _add_redherring(scenario, rng)

    return {
        "id": row_id,
        "scenario": scenario,
        "instruction": instruction,
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
    }


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d", "answer"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    obfuscated_rows = []
    for _, row in df.iterrows():
        ob = _obfuscate_row(row.to_dict(), int(row["id"]))
        ob["answer"] = row["answer"]
        obfuscated_rows.append(ob)

    ob_df = pd.DataFrame(obfuscated_rows)

    train_df, test_df = train_test_split(
        ob_df, test_size=0.2, random_state=42, shuffle=True, stratify=ob_df["answer"]
    )

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_cols = ["id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d", "answer"]
    test_cols = ["id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d"]

    train_df[train_cols].to_csv(public / "train.csv", index=False)
    test_df[test_cols].to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    sample["answer"] = "A"
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "answer"]].to_csv(private / "answers.csv", index=False)
```

---

## 8) Evaluation Rubrics

### Rubric 1
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Criteria:** Loads and parses the training and test CSV files correctly, handling all text columns without truncation or encoding errors.
- **Rationale:** The scenario and option columns contain long natural-language text with varied punctuation and special characters. Correct parsing is a prerequisite for any downstream processing.

### Rubric 2
- **Type:** CODE_QUALITY
- **Importance:** REQUIRED
- **Criteria:** Produces a valid submission CSV containing exactly 4,000 rows with columns `id` and `answer`, where each `answer` is one of A, B, C, or D.
- **Rationale:** An incorrectly formatted submission will fail the grader. The agent must produce output matching the specified schema.

### Rubric 3
- **Type:** MODELING
- **Importance:** REQUIRED
- **Criteria:** Achieves accuracy meaningfully above random baseline (0.25) on the test set.
- **Rationale:** A solution that scores at or near 0.25 indicates no meaningful learning has occurred and is functionally equivalent to guessing.

### Rubric 4
- **Type:** AGENT_BEHAVIOR
- **Importance:** RECOMMENDED
- **Criteria:** Iterates on the approach if initial predictions are near random baseline, rather than submitting without improvement.
- **Rationale:** Good engineering practice involves evaluating intermediate results and adjusting strategy when performance is poor.

### Rubric 5
- **Type:** CODE_QUALITY
- **Importance:** UNIVERSAL
- **Criteria:** Does not use test set labels or leak information from the private answers into the prediction pipeline.
- **Rationale:** Using ground-truth test labels would produce artificially inflated scores that don't reflect genuine model capability.
