# Challenge Form Fill

## Difficulty

Hard

## Challenge Title

Battle Arena Hero Draft Recommendation

## Problem Description

### Overview

In competitive 5v5 battle arenas, each match begins with a **draft phase** where two captains alternate picking heroes from a shared pool of 120 fighters following a snake order. Once a hero is claimed, it's off the board for both sides.

Picking well means more than grabbing the strongest fighter. The arena runs on hidden dynamics:

- Some hero pairings amplify each other — pair the right duo and the whole team gets stronger.
- Some heroes shut down specific opponents — pick the right counter and the enemy's star becomes dead weight.
- The map tips the scales — certain fighters thrive on certain terrain.
- Team balance matters — covering different combat roles is rewarded, while stacking one archetype is penalized.

None of these interactions are spelled out in the data. The hero stat sheets give raw numbers and three additional observable attributes. The map files give terrain specs. The real game — synergies, counters, map affinities — must be inferred from draft outcomes.

**The twist: fog of war.** Training data shows the full draft state for both teams. But in the test set, approximately 40% of the opposing team's picks are hidden (masked to -1). The solver must recommend heroes despite **incomplete information about the enemy composition** — reasoning about what the opponent likely picked based on the visible picks, the map, and the draft stage.

This creates a two-phase modeling challenge:
1. **Learn the scoring function** from fully-observed training drafts (what makes a good pick given allies, enemies, and map)
2. **Generalize under uncertainty** at test time (recommend robust picks when the enemy composition is only partially visible)

**Your task:** Given a partial draft state with fog-of-war conditions and a map, recommend the top 5 heroes to pick next from the roster of 120 available fighters.

## Evaluation

Submissions are scored using **Recall@5** — the fraction of test situations where the optimal hero appears among your 5 recommended picks.

- Picking 5 random heroes scores approximately **0.04** (5 out of 120).
- Always picking the 5 most globally popular heroes scores approximately **0.115**.
- The theoretical maximum is approximately **0.95** (some situations have near-ties due to evaluation noise, and fog-of-war masking makes some test situations genuinely ambiguous).

Higher is better.

## Dataset

After preparation, the public directory contains:

| File                   | Description                                                    |
|------------------------|----------------------------------------------------------------|
| heroes.csv             | 120 heroes with combat stats and attributes                    |
| maps.csv               | 15 maps with terrain features                                  |
| train.csv              | 33,750 draft situations with full visibility and optimal pick  |
| test.csv               | 11,250 draft situations with fog-of-war on enemy picks         |
| sample_submission.csv  | Example submission with all pick columns set to 0              |

**heroes.csv columns:**

| Column     | Type  | Description                          |
|------------|-------|--------------------------------------|
| hero_id    | int   | Unique identifier (0–119)            |
| atk        | float | Attack power                         |
| def        | float | Defense rating                       |
| spd        | float | Speed                                |
| hp         | float | Hit points                           |
| mp         | float | Mana points                          |
| rng        | float | Attack range                         |
| flux       | float | Observable attribute                 |
| resonance  | float | Observable attribute                 |
| volatility | float | Observable attribute                 |

**maps.csv columns:**

| Column     | Type  | Description                          |
|------------|-------|--------------------------------------|
| map_id     | int   | Unique identifier (0–14)             |
| terrain    | float | Terrain roughness                    |
| size       | float | Arena size                           |
| visibility | float | Sight-line clarity                   |
| elevation  | float | Vertical variation                   |

**train.csv / test.csv columns:**

| Column         | Type | Description                                                |
|----------------|------|------------------------------------------------------------|
| situation_id   | int  | Unique identifier                                          |
| map_id         | int  | Map for this match (0–14)                                  |
| pick_turn      | int  | Draft turn number (0–9)                                    |
| picking_team   | str  | "A" or "B"                                                 |
| team_a_pick_1  | int  | Hero picked by Team A in slot 1 (-1 if empty or hidden)   |
| team_a_pick_2  | int  | Hero picked by Team A in slot 2 (-1 if empty or hidden)   |
| team_a_pick_3  | int  | Hero picked by Team A in slot 3 (-1 if empty or hidden)   |
| team_a_pick_4  | int  | Hero picked by Team A in slot 4 (-1 if empty or hidden)   |
| team_a_pick_5  | int  | Hero picked by Team A in slot 5 (-1 if empty or hidden)   |
| team_b_pick_1  | int  | Hero picked by Team B in slot 1 (-1 if empty or hidden)   |
| team_b_pick_2  | int  | Hero picked by Team B in slot 2 (-1 if empty or hidden)   |
| team_b_pick_3  | int  | Hero picked by Team B in slot 3 (-1 if empty or hidden)   |
| team_b_pick_4  | int  | Hero picked by Team B in slot 4 (-1 if empty or hidden)   |
| team_b_pick_5  | int  | Hero picked by Team B in slot 5 (-1 if empty or hidden)   |
| best_pick      | int  | Optimal hero to select (0–119) — training only            |

The draft follows snake order: pick turns 0–9 correspond to A, B, B, A, A, B, B, A, A, B. A value of -1 means that roster slot is either unfilled (early in the draft) or **hidden by fog of war** (test set only). In training data, all picks are fully visible. In the test set, approximately 40% of the opposing team's existing picks are masked to -1, simulating incomplete scouting information. Heroes already picked by either team (including hidden ones) are unavailable — the optimal pick accounts for the full state, but the solver sees only partial information.

## Submission

Submit a CSV file recommending your top 5 hero picks for each test situation:

| Column       | Type | Description                                            |
|--------------|------|--------------------------------------------------------|
| situation_id | int  | Identifier from the test set                           |
| pick_1       | int  | Top recommended hero_id (0–119)                        |
| pick_2       | int  | 2nd recommended hero_id (0–119)                        |
| pick_3       | int  | 3rd recommended hero_id (0–119)                        |
| pick_4       | int  | 4th recommended hero_id (0–119)                        |
| pick_5       | int  | 5th recommended hero_id (0–119)                        |

**Requirements:**
- Must contain exactly 11,250 rows (one per test situation)
- Must include a header row
- All situation_id values must be unique and match the test set
- All pick values must be integers in the range 0–119

## Tags

feature-engineering

## Grading Configuration

- Direction: **Maximize** (Recall@5)
- Theoretical minimum: **0**
- Theoretical maximum: **1**

## Grading Script

```python
"""
grade.py — Battle Arena Hero Draft Recommendation

Metric: Recall@5 — fraction of test situations where the optimal
hero_id appears among the solver's top-5 recommended picks.

Returns 0.0 for any structurally invalid submission.
"""

import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        pick_cols = ["pick_1", "pick_2", "pick_3", "pick_4", "pick_5"]

        if "situation_id" not in submission.columns:
            return 0.0
        if not all(c in submission.columns for c in pick_cols):
            return 0.0
        if "situation_id" not in answers.columns or "best_pick" not in answers.columns:
            return 0.0

        if submission["situation_id"].duplicated().any():
            return 0.0
        if answers["situation_id"].duplicated().any():
            return 0.0

        if len(submission) != len(answers):
            return 0.0

        merged = answers.merge(
            submission, on="situation_id", how="left"
        )

        if len(merged) != len(answers):
            return 0.0
        if merged[pick_cols].isna().any().any():
            return 0.0

        true_vals = merged["best_pick"].astype(int).values
        pred_matrix = merged[pick_cols].astype(int).values

        hits = 0
        for i in range(len(true_vals)):
            if true_vals[i] in pred_matrix[i]:
                hits += 1

        return float(hits / len(true_vals))

    except Exception:
        return 0.0
```

## Prepare Script

```python
"""
prepare.py — Battle Arena Hero Draft Recommendation

Reads raw/ directory containing heroes.csv, maps.csv, situations.csv.
Produces:
  public/  heroes.csv, maps.csv, train.csv, test.csv, sample_submission.csv
  private/ answers.csv

Key twist: in test situations, ~40% of enemy picks are masked to -1,
simulating fog-of-war conditions. The model must learn hero interactions
from fully-visible training data and generalize to partial observability
at test time.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw)
    public = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = np.random.RandomState(seed)

    heroes = pd.read_csv(str(raw / "heroes.csv"))
    maps = pd.read_csv(str(raw / "maps.csv"))
    df = pd.read_csv(str(raw / "situations.csv"))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split = int(len(df) * 0.75)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    assert len(set(train_df["situation_id"]) & set(test_df["situation_id"])) == 0

    feature_cols = [c for c in df.columns if c != "best_pick"]

    train_df.to_csv(str(public / "train.csv"), index=False)

    test_pub = test_df[feature_cols].copy()
    enemy_cols_a = [f"team_a_pick_{i}" for i in range(1, 6)]
    enemy_cols_b = [f"team_b_pick_{i}" for i in range(1, 6)]

    for idx in test_pub.index:
        picking = test_pub.loc[idx, "picking_team"]
        if picking == "A":
            enemy_cols = enemy_cols_b
        else:
            enemy_cols = enemy_cols_a

        for col in enemy_cols:
            if test_pub.loc[idx, col] != -1 and rng.random() < 0.40:
                test_pub.loc[idx, col] = -1

    test_pub.to_csv(str(public / "test.csv"), index=False)

    heroes.to_csv(str(public / "heroes.csv"), index=False)
    maps.to_csv(str(public / "maps.csv"), index=False)

    test_df[["situation_id", "best_pick"]].to_csv(
        str(private / "answers.csv"), index=False
    )

    sub = test_df[["situation_id"]].copy()
    for i in range(1, 6):
        sub[f"pick_{i}"] = 0
    sub.to_csv(str(public / "sample_submission.csv"), index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"public/ and private/ written.")


if __name__ == "__main__":
    prepare(Path("raw_data"), Path("pub"), Path("priv"))
```

## Evaluation Rubrics

### Rubric 1
- **Criterion:** Loads heroes.csv, maps.csv, and train.csv correctly and joins hero/map features to draft situations for modeling.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** The three files must be linked together — hero stats inform pick quality and map features influence optimal strategy. Failing to join them leaves the model blind to the core signals.

### Rubric 2
- **Criterion:** Produces a valid submission CSV with columns `situation_id`, `pick_1`, `pick_2`, `pick_3`, `pick_4`, `pick_5`, exactly 11,250 rows, unique IDs matching the test set, and integer hero_id values in the range 0–119.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** Malformed submissions receive a score of 0.0 from the grader. Structural correctness is a prerequisite for evaluation.

### Rubric 3
- **Criterion:** Achieves a Recall@5 exceeding 0.12 on the test set (above the top-5-frequency baseline of ~0.115).
- **Type:** MODELING
- **Importance:** REQUIRED
- **Rationale:** Scoring above the naive frequency baseline confirms the model has learned at least some conditional structure from the draft context rather than always outputting the same set of heroes.

### Rubric 4
- **Criterion:** Model architecture uses fewer than 50 million total trainable parameters.
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Rationale:** The underlying game mechanics can be captured with a moderately sized model. Excessive parameterization suggests overfitting or inefficiency rather than genuine understanding of the draft dynamics.

### Rubric 5
- **Criterion:** Does not use test labels, test-set statistics, or external draft/game data during training or feature engineering.
- **Type:** TRAINING
- **Importance:** UNIVERSAL
- **Rationale:** Using test information during training constitutes data leakage and produces unreliable performance estimates. No external game data exists — the solver must learn from the provided training set only.
