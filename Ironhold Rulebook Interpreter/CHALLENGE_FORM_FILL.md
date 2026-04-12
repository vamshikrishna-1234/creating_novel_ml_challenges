# Challenge Form Fill

## Difficulty
Hard

## Challenge Title
Ironhold Arena Game Ruling Generation

## Tags
text, feature-engineering, generative

## Problem Description

### Overview

Ironhold Arena is a fictional tabletop strategy game with a hidden rule system governing how 8 unit types interact across 6 terrain types, 8 spells, 6 status effects, and 5 numerical attributes (level, HP, mana, strength, defense). The rules form a layered priority system with ~90 interacting rules — but **the rules are never shown to you**. You must learn them entirely from 4,125 training examples.

Given a game-state description and a question about whether a unit can perform an action (move, attack, or cast a spell), the task is to generate a **structured ruling** with two parts:

1. **Analysis** — a natural language paragraph explaining which game mechanics apply, how they interact, and why, referencing specific numbers from the situation
2. **Outcome** — the final verdict: `ALLOWED`, `BLOCKED`, or `MODIFIED`

This is a **fine-tuning challenge**. You must fine-tune an open-weights language model with **3 billion parameters or fewer** (e.g., Phi-3-mini, Llama-3.2-3B, StableLM-2) to learn the hidden rule system from examples alone. Pre-trained models cannot answer these questions without fine-tuning because:

- The game, its rules, and all mechanics are entirely fictional — nothing in any pre-training corpus describes Ironhold Arena
- The rules depend on both categorical conditions (unit type, terrain, status) AND numerical thresholds (mana costs, level gates, HP penalties, strength-vs-defense comparisons) that must be inferred from data
- No rulebook, rule IDs, or explicit rule descriptions are provided anywhere in the dataset

**Hardware**: Training and inference must run within the platform time limit on an NVIDIA H100.

### The Three Traps (Why Standard Approaches Fail)

**The Inference Gap**: The hidden rule system contains ~90 rules with 2-5 conditions each, spanning categorical AND numerical features. A model that memorizes training examples without learning the underlying rules will fail on novel feature combinations in the test set. There are too many interacting conditions to extract by brute-force pattern matching within typical agent time budgets.

**The Numbers Trap**: Rules depend on specific numerical thresholds (e.g., a spell requires minimum mana of 28, a level gate at 6, strength-vs-defense comparisons). The analysis must reference the *exact numbers from each situation* — not generic descriptions. A model that generates "the unit has insufficient mana" without citing the actual mana value and threshold scores poorly on the character-level evaluation component.

**The Noise Floor**: 10% of training labels have intentionally flipped outcomes (the analysis text may describe a BLOCKED scenario but the outcome says MODIFIED). This noise prevents perfect memorization and forces the model to learn robust patterns rather than rote input-output mappings. Theoretical maximum performance is approximately 0.90.

### Evaluation

Submissions are scored using a **composite metric** combining three components:

**Score = 0.40 × Outcome Accuracy + 0.30 × Analysis Token F1 + 0.30 × Analysis chrF**

| Component | Weight | How It's Computed |
|-----------|--------|-------------------|
| Outcome Accuracy | 40% | Exact match between predicted and true outcome (ALLOWED/BLOCKED/MODIFIED). Extracted from `<outcome>` tag. |
| Analysis Token F1 | 30% | Word-level F1 between predicted and reference analysis text. Extracted from `<analysis>` tag. Measures whether the explanation mentions the correct game mechanics and concepts. |
| Analysis chrF | 30% | Character n-gram F-score (chrF with β=2, up to 6-grams) between predicted and reference analysis. Sensitive to exact character sequences including specific numbers, unit names, and terrain references. |

- **Higher is better.** Minimum: 0.0, Maximum: 1.0.
- The grader parses `<analysis>` and `<outcome>` XML-style tags from the output string. Missing or malformed tags score 0 for that component.

**Baseline scores:**
- Always predict BLOCKED with empty analysis: ~0.18
- Correct outcome only (empty analysis): ~0.40
- Expected practical ceiling: ~0.90 (training noise limits what models can learn, but the grading scale goes to 1.0)

### Dataset

After preparation, the public directory contains:

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 4,125 | Labeled examples with 10% noisy outcomes |
| `test.csv` | 1,375 | Unlabeled: question_id, situation, question |
| `sample_submission.csv` | 1,375 | Placeholder submission with default BLOCKED outcome |

**No rulebook or rule reference is provided.** The hidden rules must be learned from training examples.

**Training columns:**

| Column | Type | Description |
|--------|------|-------------|
| question_id | int | Unique identifier |
| situation | str | Natural language game state with unit types, terrain, statuses, and specific numerical attributes (level, HP, mana, strength, defense) |
| question | str | The action being queried (e.g., "Can the Mage cast Fireball on the Guardian?") |
| output | str | Tagged ruling: `<analysis>paragraph explanation</analysis><outcome>VERDICT</outcome>` |

**Test columns:** Same as training minus the `output` column.

**Situation format example:**
> A level 7 Mage stands on Forest terrain with 34 mana and 52 HP (strength 8, defense 6). The target is a level 4 Guardian with Shielded status on Plains (78 HP, 14 strength, 17 defense), adjacent to the caster. The Mage has Burning status. The Mage attempts to cast Fireball on the Guardian.

### Submission

Submit a CSV with the following format:

| Column | Type | Description |
|--------|------|-------------|
| question_id | int | Identifier from test.csv |
| output | str | Generated ruling in tagged format |

The `output` column must contain both XML-style tags:
```
<analysis>The Mage has 34 mana, exceeding the 28 mana threshold for Fireball. Forest terrain would normally suppress fire magic, but the Mage's Burning status provides elemental affinity that overcomes the dampening. The Fireball connects with the Shielded Guardian, but the shield absorbs half the damage. The action proceeds but with altered effects.</analysis>
<outcome>MODIFIED</outcome>
```

**Requirements:**
- Must contain exactly 1,375 rows (one per test question)
- Include a header row
- All question_id values must be unique and match the test set
- Outcome must be one of: ALLOWED, BLOCKED, MODIFIED
- Missing or malformed tags score 0 for that component

## GPU Tier
H100 — LLM training & fine-tuning (required for fine-tuning a ≤3B parameter model)

## Grading Configuration

- **Grade direction**: Maximize (higher is better)
- **Theoretical minimum**: 0.0
- **Theoretical maximum**: 1.0

## Grading Script

```python
import pandas as pd
import re
from collections import Counter


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        merged = answers.merge(
            submission, on="question_id", how="left", suffixes=("_true", "_pred")
        )

        if merged["output"].isna().any():
            return 0.0

        if merged["question_id"].duplicated().any():
            return 0.0

        n = len(merged)
        if n == 0:
            return 0.0

        outcome_hits = 0
        token_f1_sum = 0.0
        chrf_sum = 0.0

        for _, row in merged.iterrows():
            pred_output = str(row["output"])
            true_outcome = str(row["outcome"])
            true_analysis = str(row["analysis"])

            pred_outcome = _extract_tag(pred_output, "outcome").strip().upper()
            if pred_outcome == true_outcome:
                outcome_hits += 1

            pred_analysis = _extract_tag(pred_output, "analysis")
            token_f1_sum += _token_f1(pred_analysis, true_analysis)
            chrf_sum += _chrf(pred_analysis, true_analysis)

        outcome_acc = outcome_hits / n
        token_f1 = token_f1_sum / n
        chrf_score = chrf_sum / n

        score = 0.40 * outcome_acc + 0.30 * token_f1 + 0.30 * chrf_score
        return float(max(0.0, min(1.0, score)))

    except Exception:
        return 0.0


def _extract_tag(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""


def _token_f1(pred: str, true: str) -> float:
    pred_tokens = set(pred.lower().split())
    true_tokens = set(true.lower().split())
    if not true_tokens and not pred_tokens:
        return 1.0
    if not true_tokens or not pred_tokens:
        return 0.0
    tp = len(pred_tokens & true_tokens)
    prec = tp / len(pred_tokens)
    rec = tp / len(true_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _chrf(pred: str, true: str, max_n: int = 6, beta: float = 2.0) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0

    pred_ngrams = Counter()
    true_ngrams = Counter()
    for n in range(1, max_n + 1):
        for i in range(len(pred) - n + 1):
            pred_ngrams[pred[i : i + n]] += 1
        for i in range(len(true) - n + 1):
            true_ngrams[true[i : i + n]] += 1

    tp = sum((pred_ngrams & true_ngrams).values())
    pred_total = sum(pred_ngrams.values())
    true_total = sum(true_ngrams.values())

    prec = tp / pred_total if pred_total > 0 else 0.0
    rec = tp / true_total if true_total > 0 else 0.0

    if prec + rec == 0:
        return 0.0

    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
```

## Evaluation Rubrics

### Rubric 1
- **Type**: MODELING
- **Importance**: REQUIRED
- **Criteria**: The solution must fine-tune an open-weights language model with 3 billion parameters or fewer. Closed-source API calls are prohibited for both training and inference.
- **Rationale**: The game's rule system is entirely fictional and absent from all pre-training corpora. The parameter cap ensures the challenge tests training methodology rather than brute-force scaling.

### Rubric 2
- **Type**: DATA_HANDLING
- **Importance**: REQUIRED
- **Criteria**: Submission output must use the XML-style tagged format (`<analysis>`, `<outcome>`) with outcome being one of ALLOWED, BLOCKED, or MODIFIED.
- **Rationale**: The automated grader parses these tags. Missing or malformed tags produce a zero score for the affected component.

### Rubric 3
- **Type**: TRAINING
- **Importance**: REQUIRED
- **Criteria**: Fine-tuning must be genuine and substantial — no trivially small learning rates, no zero-shot or one-shot style submissions that bypass actual model training.
- **Rationale**: The challenge evaluates whether a model can learn a novel rule system from examples. Submissions that skip meaningful training defeat the challenge's purpose.

### Rubric 4
- **Type**: CODE_QUALITY
- **Importance**: REQUIRED
- **Criteria**: The training and inference pipeline must be self-contained, executing within the platform time limit on an H100 GPU without internet access or external API calls.
- **Rationale**: Reproducibility and fairness require that all computation happens within the provisioned environment.

### Rubric 5
- **Type**: TRAINING
- **Importance**: RECOMMENDED
- **Criteria**: The solution should use a proper validation split to monitor generalization and avoid overfitting to the training set.
- **Rationale**: The test set contains game states not seen during training. A model that memorizes training examples without learning generalizable patterns will perform poorly on held-out data.
