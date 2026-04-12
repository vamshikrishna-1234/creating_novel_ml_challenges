"""
generate.py — Phantom League Draft Optimizer

Builds a synthetic fantasy battle arena with 120 heroes, 15 maps,
hidden role/element systems, synergy/counter matrices, and generates
45,000 draft situations with optimal next-pick labels.

The draft follows a 10-pick snake format:
  A1 B1 B2 A2 A3 B3 B4 A4 A5 B5

At each pick, the simulator scores every available hero for the picking
team and selects the one with the highest composite score. The composite
considers: base hero power, role coverage, element countering, pairwise
synergies, map bonuses, and anti-stacking penalties.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import itertools

SEED = 42

N_HEROES = 120
N_MAPS = 15
N_ROLES = 5
N_ELEMENTS = 5
N_DRAFTS = 9000
SITUATIONS_PER_DRAFT = 5  # sample 5 of 10 pick-turns per draft -> 45K

ROLES = ["tank", "damage", "support", "assassin", "controller"]
ELEMENTS = ["fire", "ice", "nature", "shadow", "light"]

# circular counter: fire > nature > ice > shadow > light > fire
ELEMENT_COUNTER = {
    "fire": "nature",
    "nature": "ice",
    "ice": "shadow",
    "shadow": "light",
    "light": "fire",
}
ELEMENT_COUNTERED_BY = {v: k for k, v in ELEMENT_COUNTER.items()}

DRAFT_ORDER = [
    ("A", 0), ("B", 0), ("B", 1), ("A", 1), ("A", 2),
    ("B", 2), ("B", 3), ("A", 3), ("A", 4), ("B", 4),
]


def generate_heroes(rng):
    heroes = []
    for hid in range(N_HEROES):
        role = ROLES[hid % N_ROLES]
        element = ELEMENTS[(hid // N_ROLES) % N_ELEMENTS]

        base = {
            "tank":       {"atk": 0.3, "def": 0.9, "spd": 0.3, "hp": 0.95, "mp": 0.3, "rng": 0.2},
            "damage":     {"atk": 0.9, "def": 0.3, "spd": 0.6, "hp": 0.4,  "mp": 0.5, "rng": 0.7},
            "support":    {"atk": 0.2, "def": 0.5, "spd": 0.5, "hp": 0.6,  "mp": 0.9, "rng": 0.6},
            "assassin":   {"atk": 0.8, "def": 0.2, "spd": 0.95,"hp": 0.3,  "mp": 0.4, "rng": 0.3},
            "controller": {"atk": 0.5, "def": 0.4, "spd": 0.4, "hp": 0.5,  "mp": 0.8, "rng": 0.8},
        }[role]

        stats = {}
        for s, v in base.items():
            stats[s] = np.clip(v + rng.normal(0, 0.15), 0.05, 1.0)

        decoy_1 = rng.uniform(0, 1)
        decoy_2 = rng.uniform(0, 1)
        decoy_3 = rng.uniform(0, 1)

        heroes.append({
            "hero_id": hid,
            "atk": round(stats["atk"], 3),
            "def": round(stats["def"], 3),
            "spd": round(stats["spd"], 3),
            "hp":  round(stats["hp"], 3),
            "mp":  round(stats["mp"], 3),
            "rng": round(stats["rng"], 3),
            "flux": round(decoy_1, 3),
            "resonance": round(decoy_2, 3),
            "volatility": round(decoy_3, 3),
            "_role": role,
            "_element": element,
        })
    return heroes


def generate_synergy_matrix(rng, heroes):
    n = len(heroes)
    synergy = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = heroes[i], heroes[j]
            val = 0.0
            if hi["_element"] == hj["_element"]:
                val += rng.uniform(0.05, 0.20)
            if hi["_role"] != hj["_role"]:
                val += rng.uniform(0.0, 0.08)
            if rng.random() < 0.08:
                val += rng.uniform(0.10, 0.35)
            synergy[i][j] = val
            synergy[j][i] = val
    return synergy


def generate_counter_matrix(rng, heroes):
    n = len(heroes)
    counter = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hi, hj = heroes[i], heroes[j]
            val = 0.0
            if ELEMENT_COUNTER.get(hi["_element"]) == hj["_element"]:
                val += rng.uniform(0.10, 0.30)
            if hi["_role"] == "assassin" and hj["_role"] == "support":
                val += rng.uniform(0.05, 0.15)
            if hi["_role"] == "tank" and hj["_role"] == "assassin":
                val += rng.uniform(0.05, 0.15)
            if hi["_role"] == "controller" and hj["_role"] == "damage":
                val += rng.uniform(0.05, 0.12)
            if rng.random() < 0.05:
                val += rng.uniform(0.08, 0.25)
            counter[i][j] = val
    return counter


def generate_maps(rng):
    maps = []
    for mid in range(N_MAPS):
        favored_element = ELEMENTS[mid % N_ELEMENTS]
        favored_role = ROLES[(mid * 3) % N_ROLES]
        maps.append({
            "map_id": mid,
            "terrain": round(rng.uniform(0, 1), 3),
            "size": round(rng.uniform(0, 1), 3),
            "visibility": round(rng.uniform(0, 1), 3),
            "elevation": round(rng.uniform(0, 1), 3),
            "_favored_element": favored_element,
            "_favored_role": favored_role,
        })
    return maps


def hero_power(hero):
    return (hero["atk"] + hero["def"] + hero["spd"] +
            hero["hp"] + hero["mp"] + hero["rng"]) / 6.0


def score_pick(hero_id, team_picks, enemy_picks, heroes, synergy, counter,
               map_info, available):
    hero = heroes[hero_id]

    base = hero_power(hero) * 0.25

    team_roles = [heroes[h]["_role"] for h in team_picks]
    role = hero["_role"]
    role_count = team_roles.count(role)
    if role_count == 0:
        role_bonus = 0.15
    elif role_count == 1:
        role_bonus = 0.02
    else:
        role_bonus = -0.10 * role_count

    unique_roles = len(set(team_roles + [role]))
    coverage_bonus = 0.04 * unique_roles

    syn_score = 0.0
    for ally in team_picks:
        syn_score += synergy[hero_id][ally]

    cnt_score = 0.0
    for enemy in enemy_picks:
        cnt_score += counter[hero_id][enemy]
    for enemy in enemy_picks:
        cnt_score -= counter[enemy][hero_id] * 0.3

    map_bonus = 0.0
    if hero["_element"] == map_info["_favored_element"]:
        map_bonus += 0.12
    if hero["_role"] == map_info["_favored_role"]:
        map_bonus += 0.08

    return base + role_bonus + coverage_bonus + syn_score + cnt_score + map_bonus


def simulate_draft(heroes, synergy, counter, map_info, rng):
    team_a = []
    team_b = []
    available = set(range(N_HEROES))
    pick_log = []

    for turn_idx, (team_letter, slot) in enumerate(DRAFT_ORDER):
        if team_letter == "A":
            my_team, enemy_team = team_a, team_b
        else:
            my_team, enemy_team = team_b, team_a

        best_hero = -1
        best_score = -999.0
        scores = {}

        for hid in available:
            s = score_pick(hid, my_team, enemy_team, heroes, synergy,
                           counter, map_info, available)
            noise = rng.normal(0, 0.03)
            s += noise
            scores[hid] = s
            if s > best_score:
                best_score = s
                best_hero = hid

        pick_log.append({
            "turn": turn_idx,
            "team": team_letter,
            "slot": slot,
            "best_pick": best_hero,
            "team_a": list(team_a),
            "team_b": list(team_b),
        })

        if team_letter == "A":
            team_a.append(best_hero)
        else:
            team_b.append(best_hero)
        available.discard(best_hero)

    return pick_log, team_a, team_b


def main():
    rng = np.random.RandomState(SEED)

    print("Generating heroes...")
    heroes = generate_heroes(rng)

    print("Generating synergy matrix...")
    synergy = generate_synergy_matrix(rng, heroes)

    print("Generating counter matrix...")
    counter = generate_counter_matrix(rng, heroes)

    print("Generating maps...")
    maps = generate_maps(rng)

    heroes_df = pd.DataFrame(heroes)
    public_cols = ["hero_id", "atk", "def", "spd", "hp", "mp", "rng",
                   "flux", "resonance", "volatility"]
    heroes_pub = heroes_df[public_cols]

    maps_df = pd.DataFrame(maps)
    maps_pub = maps_df[["map_id", "terrain", "size", "visibility", "elevation"]]

    print(f"Simulating {N_DRAFTS} drafts...")
    all_situations = []
    sid = 0

    for draft_idx in range(N_DRAFTS):
        map_info = maps[rng.randint(0, N_MAPS)]

        pick_log, team_a, team_b = simulate_draft(
            heroes, synergy, counter, map_info, rng
        )

        turns_to_sample = rng.choice(10, size=SITUATIONS_PER_DRAFT, replace=False)

        for turn_idx in sorted(turns_to_sample):
            entry = pick_log[turn_idx]
            ta = entry["team_a"]
            tb = entry["team_b"]

            row = {
                "situation_id": sid,
                "map_id": map_info["map_id"],
                "pick_turn": turn_idx,
                "picking_team": entry["team"],
            }
            for i in range(5):
                row[f"team_a_pick_{i+1}"] = ta[i] if i < len(ta) else -1
            for i in range(5):
                row[f"team_b_pick_{i+1}"] = tb[i] if i < len(tb) else -1

            row["best_pick"] = entry["best_pick"]
            all_situations.append(row)
            sid += 1

        if (draft_idx + 1) % 2000 == 0:
            print(f"  {draft_idx + 1}/{N_DRAFTS} drafts done...")

    df = pd.DataFrame(all_situations)

    out = Path("raw_data")
    out.mkdir(exist_ok=True)
    heroes_pub.to_csv(out / "heroes.csv", index=False)
    maps_pub.to_csv(out / "maps.csv", index=False)
    df.to_csv(out / "situations.csv", index=False)

    print(f"\nGenerated {len(df)} draft situations from {N_DRAFTS} drafts.")
    print(f"Unique heroes picked as best: {df['best_pick'].nunique()}")
    print(f"Pick turn distribution:\n{df['pick_turn'].value_counts().sort_index()}")
    print(f"\nTop-10 most recommended heroes:")
    print(df["best_pick"].value_counts().head(10))
    print(f"\nSample rows:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
