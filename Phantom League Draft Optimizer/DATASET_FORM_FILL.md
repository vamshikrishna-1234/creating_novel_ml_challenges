# Dataset Form Fill

## Title

Battle Arena Hero Draft Recommendation Corpus

## Description

### Overview

This dataset captures 45,000 draft situations from a synthetic competitive 5v5 battle arena. In each match, two teams of 5 heroes are assembled through an alternating draft from a shared pool of 120 fighters. The dataset records the state of the draft at a given pick turn — which heroes each team has already locked in, which map the match is on, and whose turn it is to pick — along with the optimal next hero selection.

The 120 heroes have observable combat statistics (attack, defense, speed, hit points, mana, range) plus three additional observable attributes (flux, resonance, volatility). Hidden from the solver are each hero's **role** (tank, damage, support, assassin, controller) and **element** (fire, ice, nature, shadow, light). These hidden attributes drive a complex system of synergies, counters, role coverage bonuses, and map-dependent advantages that determine which pick is optimal in any given situation.

The 15 maps each have four observable terrain features. Hidden from the solver is each map's affinity for a particular element and role, which gives bonuses to heroes matching those traits.

The optimal pick for each situation was computed by a deterministic simulator that evaluates every available hero against the current draft state and selects the one with the highest composite score. A small amount of noise was added to the simulator's evaluations, so a handful of situations have near-ties where the "best" pick is essentially a coin flip.

The solver's goal is to recommend the top 5 hero picks for each test situation. Training data shows the full draft state for both teams. In the test set, approximately 40% of the opposing team's existing picks are masked to -1, simulating fog-of-war conditions where the enemy composition is only partially visible. The task is scored by Recall@5 — whether the optimal hero appears among the solver's 5 recommendations.

### File Structure

- `heroes.csv` — Roster of 120 heroes with combat statistics and three additional observable attributes. Each row is one hero. 120 rows, 10 columns.
- `maps.csv` — 15 arena maps with terrain characteristics. Each row is one map. 15 rows, 5 columns.
- `situations.csv` — 45,000 draft snapshots recording the partial draft state, the map, the picking team, and the optimal next hero selection. Each row is one draft situation. 45,000 rows, 15 columns.

### Features

**heroes.csv**

| Column     | Type  | Description                                      |
|------------|-------|--------------------------------------------------|
| hero_id    | int   | Unique identifier (0–119)                        |
| atk        | float | Attack power (0–1 scale)                         |
| def        | float | Defense rating (0–1 scale)                       |
| spd        | float | Speed (0–1 scale)                                |
| hp         | float | Hit points (0–1 scale)                           |
| mp         | float | Mana points (0–1 scale)                          |
| rng        | float | Attack range (0–1 scale)                         |
| flux       | float | Observable attribute (0–1 scale)                 |
| resonance  | float | Observable attribute (0–1 scale)                 |
| volatility | float | Observable attribute (0–1 scale)                 |

**maps.csv**

| Column     | Type  | Description                                      |
|------------|-------|--------------------------------------------------|
| map_id     | int   | Unique identifier (0–14)                         |
| terrain    | float | Terrain roughness (0–1 scale)                    |
| size       | float | Arena size (0–1 scale)                           |
| visibility | float | Sight-line clarity (0–1 scale)                   |
| elevation  | float | Vertical variation (0–1 scale)                   |

**situations.csv**

| Column         | Type  | Description                                              |
|----------------|-------|----------------------------------------------------------|
| situation_id   | int   | Unique identifier (0–44999)                              |
| map_id         | int   | Map where this match takes place (0–14)                  |
| pick_turn      | int   | Draft turn number (0–9, in snake draft order)            |
| picking_team   | str   | Which team is picking: "A" or "B"                        |
| team_a_pick_1  | int   | Hero already picked by Team A in slot 1 (-1 if empty)   |
| team_a_pick_2  | int   | Hero already picked by Team A in slot 2 (-1 if empty)   |
| team_a_pick_3  | int   | Hero already picked by Team A in slot 3 (-1 if empty)   |
| team_a_pick_4  | int   | Hero already picked by Team A in slot 4 (-1 if empty)   |
| team_a_pick_5  | int   | Hero already picked by Team A in slot 5 (-1 if empty)   |
| team_b_pick_1  | int   | Hero already picked by Team B in slot 1 (-1 if empty)   |
| team_b_pick_2  | int   | Hero already picked by Team B in slot 2 (-1 if empty)   |
| team_b_pick_3  | int   | Hero already picked by Team B in slot 3 (-1 if empty)   |
| team_b_pick_4  | int   | Hero already picked by Team B in slot 4 (-1 if empty)   |
| team_b_pick_5  | int   | Hero already picked by Team B in slot 5 (-1 if empty)   |
| best_pick      | int   | Optimal hero_id to pick in this situation (0–119)        |

### Notes

- The draft follows a **snake order**: A1, B1, B2, A2, A3, B3, B4, A4, A5, B5. The `pick_turn` column (0–9) corresponds to this sequence.
- A value of **-1** in any `team_X_pick_Y` column means that slot has not been filled yet. In the prepared test set, -1 can also indicate a **fog-of-war masked** enemy pick.
- Heroes that have already been picked by either team are unavailable. The `best_pick` is always a hero not already present in the full (unmasked) draft state.
- The `flux`, `resonance`, and `volatility` columns in heroes.csv are **not causally related** to draft outcomes. They are observable but do not influence the simulator's scoring.
- Each situation was sampled from a full 10-pick draft simulation. Five of the ten pick-turns per draft were randomly selected for inclusion, so the dataset contains draft states at various stages of completion.
- The optimal pick was computed by scoring every available hero on: base power, role coverage, element counters, pairwise synergies with allies, map bonuses, and anti-stacking penalties. A small Gaussian noise (σ=0.03) was added per evaluation, so in rare near-tie situations the recorded "best" pick may not be strictly optimal.
- During preparation, ~40% of the opposing team's existing picks in test situations are masked to -1, creating a train/test distribution shift that requires the model to generalize from fully-observed training data to partially-observed test conditions.
