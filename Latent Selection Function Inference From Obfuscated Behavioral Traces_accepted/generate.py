"""
Session Purchase Prediction Challenge — data generator.

Simulates a fictional signal-interaction platform where users browse items in sessions.
Each session has 5-15 item interactions with behavioral signals (dwell time, scroll
depth, action type, position). Exactly one item per session is purchased.

The purchased item is determined by a multi-rule scoring system that combines
behavioral signals with user-item compatibility. ~20% of sessions are "passive
interest" where purchase is driven by preference-feature match rather than engagement.

Output files:
  - interactions.csv  (session_id, user_id, item_id, action_type, dwell_seconds, scroll_pct, position)
  - items.csv         (item_id, category, price_tier, attr_1, attr_2, attr_3)
  - users.csv         (user_id, pref_cat_1..3, tier_low, tier_high, pref_1..3)
  - purchases.csv     (session_id, purchased_item_id)
"""

from __future__ import annotations
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SEED = 42
N_ITEMS = 500
N_CATEGORIES = 20
N_USERS = 3000
N_SESSIONS = 15000
ITEMS_PER_SESSION_RANGE = (6, 14)


def generate_items(rng: random.Random) -> list[dict]:
    items = []
    for i in range(N_ITEMS):
        items.append({
            "item_id": i,
            "category": rng.randint(0, N_CATEGORIES - 1),
            "price_tier": rng.randint(1, 5),
            "attr_1": round(rng.gauss(0, 1), 3),
            "attr_2": round(rng.gauss(0, 1), 3),
            "attr_3": round(rng.gauss(0, 1), 3),
        })
    return items


def generate_users(rng: random.Random) -> list[dict]:
    users = []
    for u in range(N_USERS):
        cats = rng.sample(range(N_CATEGORIES), 3)
        bl = rng.randint(1, 3)
        bh = rng.randint(bl, 5)
        users.append({
            "user_id": u,
            "pref_cat_1": cats[0],
            "pref_cat_2": cats[1],
            "pref_cat_3": cats[2],
            "tier_low": bl,
            "tier_high": bh,
            "pref_1": round(rng.gauss(0, 1), 3),
            "pref_2": round(rng.gauss(0, 1), 3),
            "pref_3": round(rng.gauss(0, 1), 3),
        })
    return users


def _compute_purchase(session_ints: list[dict], user: dict,
                      items_dict: dict, rng: random.Random) -> int:
    """Score each item in the session, return the purchased item_id."""

    # ~20% of sessions are "passive interest" — user purchases based on
    # preference-feature match alone, ignoring behavioral engagement signals.
    # The purchased item may only have a view action (no click/cart).
    is_passive = rng.random() < 0.20

    item_agg: dict[int, dict] = {}
    for row in session_ints:
        iid = row["item_id"]
        if iid not in item_agg:
            item_agg[iid] = {
                "max_dwell": 0.0, "max_scroll": 0.0,
                "has_click": False, "has_cart": False, "has_remove": False,
                "visit_count": 0, "max_position": 0,
            }
        a = item_agg[iid]
        a["max_dwell"] = max(a["max_dwell"], row["dwell_seconds"])
        a["max_scroll"] = max(a["max_scroll"], row["scroll_pct"])
        if row["action_type"] == "click":
            a["has_click"] = True
        if row["action_type"] == "cart":
            a["has_cart"] = True
        if row["action_type"] == "remove":
            a["has_remove"] = True
        a["visit_count"] += 1
        a["max_position"] = max(a["max_position"], row["position"])

    all_dwells = [a["max_dwell"] for a in item_agg.values()]
    med_dwell = median(all_dwells) if all_dwells else 0

    user_cats = {user["pref_cat_1"], user["pref_cat_2"], user["pref_cat_3"]}

    if is_passive:
        # Passive session: score based ONLY on user-item feature compatibility
        scores = {}
        for iid, a in item_agg.items():
            item = items_dict[iid]
            s = 0.0
            if item["category"] in user_cats:
                s += 3.0
            if user["tier_low"] <= item["price_tier"] <= user["tier_high"]:
                s += 2.0
            # Continuous preference proximity: closer attr values to user prefs = better
            s += max(0, 1.0 - abs(item["attr_1"] - user["pref_1"]))
            s += max(0, 1.0 - abs(item["attr_2"] - user["pref_2"]))
            s += max(0, 1.0 - abs(item["attr_3"] - user["pref_3"]))
            # Small noise to break ties non-deterministically
            s += rng.random() * 0.01
            scores[iid] = (s, a["max_position"])
    else:
        scores = {}
        for iid, a in item_agg.items():
            s = 0.0
            item = items_dict[iid]

            if a["max_dwell"] > med_dwell:
                s += 2.0
            if a["has_click"]:
                s += 1.0
            if a["has_cart"]:
                s += 3.0
            if a["has_remove"]:
                s -= 1.5
            if a["visit_count"] >= 2:
                s += 2.0
            if a["max_scroll"] > 0.75:
                s += 1.0
            if item["category"] in user_cats:
                s += 1.5
            if user["tier_low"] <= item["price_tier"] <= user["tier_high"]:
                s += 1.0

            scores[iid] = (s, a["max_position"])

    winner = max(scores, key=lambda k: (scores[k][0], scores[k][1]))
    return winner


def generate_session(rng: random.Random, session_id: int, user: dict,
                     items: list[dict], items_by_cat: dict) -> tuple[list[dict], int]:
    """Generate one browsing session. Returns (interactions, purchased_item_id)."""
    n_items = rng.randint(*ITEMS_PER_SESSION_RANGE)
    user_cats = [user["pref_cat_1"], user["pref_cat_2"], user["pref_cat_3"]]

    selected_ids: set[int] = set()
    session_items: list[dict] = []

    for _ in range(n_items):
        if rng.random() < 0.35 and user_cats:
            cat = rng.choice(user_cats)
            pool = items_by_cat.get(cat, items)
        else:
            pool = items
        attempts = 0
        while attempts < 20:
            it = rng.choice(pool)
            if it["item_id"] not in selected_ids:
                selected_ids.add(it["item_id"])
                session_items.append(it)
                break
            attempts += 1

    if len(session_items) < 3:
        while len(session_items) < 3:
            it = rng.choice(items)
            if it["item_id"] not in selected_ids:
                selected_ids.add(it["item_id"])
                session_items.append(it)

    interactions = []
    position = 0

    for item in session_items:
        cat_match = item["category"] in set(user_cats)
        price_match = user["tier_low"] <= item["price_tier"] <= user["tier_high"]

        base_dwell = 8 + (12 if cat_match else 0) + (8 if price_match else 0)
        dwell = max(1.0, base_dwell + rng.gauss(0, 14))

        base_scroll = 0.25 + (0.22 if cat_match else 0) + (0.13 if price_match else 0)
        scroll = max(0.03, min(0.99, base_scroll + rng.gauss(0, 0.22)))

        position += 1
        interactions.append({
            "session_id": session_id, "user_id": user["user_id"],
            "item_id": item["item_id"], "action_type": "view",
            "dwell_seconds": round(dwell, 1),
            "scroll_pct": round(scroll, 2), "position": position,
        })

        click_prob = 0.25 + (0.18 if cat_match else 0) + (0.12 if price_match else 0)
        if rng.random() < click_prob:
            position += 1
            interactions.append({
                "session_id": session_id, "user_id": user["user_id"],
                "item_id": item["item_id"], "action_type": "click",
                "dwell_seconds": round(max(1.0, dwell * 1.4 + rng.gauss(0, 10)), 1),
                "scroll_pct": round(max(0.03, min(0.99, scroll + rng.gauss(0.08, 0.12))), 2),
                "position": position,
            })

        cart_prob = 0.08 + (0.09 if cat_match else 0) + (0.06 if price_match else 0)
        if rng.random() < cart_prob:
            position += 1
            interactions.append({
                "session_id": session_id, "user_id": user["user_id"],
                "item_id": item["item_id"], "action_type": "cart",
                "dwell_seconds": round(rng.uniform(2, 8), 1),
                "scroll_pct": round(rng.uniform(0.75, 0.99), 2),
                "position": position,
            })
            if rng.random() < 0.30:
                position += 1
                interactions.append({
                    "session_id": session_id, "user_id": user["user_id"],
                    "item_id": item["item_id"], "action_type": "remove",
                    "dwell_seconds": round(rng.uniform(1, 3), 1),
                    "scroll_pct": round(rng.uniform(0.08, 0.30), 2),
                    "position": position,
                })

        if rng.random() < 0.12 + (0.08 if cat_match else 0):
            position += 1
            interactions.append({
                "session_id": session_id, "user_id": user["user_id"],
                "item_id": item["item_id"], "action_type": "view",
                "dwell_seconds": round(max(1.0, dwell * 0.8 + rng.gauss(0, 8)), 1),
                "scroll_pct": round(max(0.03, min(0.99, scroll + rng.gauss(0, 0.15))), 2),
                "position": position,
            })

    items_dict = {it["item_id"]: it for it in items}
    purchased = _compute_purchase(interactions, user, items_dict, rng)
    return interactions, purchased


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sessions", type=int, default=N_SESSIONS)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    items = generate_items(rng)
    users = generate_users(rng)

    items_dict = {it["item_id"]: it for it in items}
    items_by_cat: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        items_by_cat[it["category"]].append(it)

    all_interactions = []
    purchases = []

    for sid in range(args.sessions):
        user = users[rng.randint(0, len(users) - 1)]
        ints, purchased_id = generate_session(rng, sid, user, items, items_by_cat)
        all_interactions.extend(ints)
        purchases.append({"session_id": sid, "purchased_item_id": purchased_id})

    # Write items.csv
    with open(out / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "category", "price_tier",
                                          "attr_1", "attr_2", "attr_3"])
        w.writeheader()
        w.writerows(items)

    # Write users.csv
    with open(out / "users.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "pref_cat_1", "pref_cat_2",
                                          "pref_cat_3", "tier_low", "tier_high",
                                          "pref_1", "pref_2", "pref_3"])
        w.writeheader()
        w.writerows(users)

    # Write interactions.csv
    int_fields = ["session_id", "user_id", "item_id", "action_type",
                  "dwell_seconds", "scroll_pct", "position"]
    with open(out / "interactions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=int_fields)
        w.writeheader()
        w.writerows(all_interactions)

    # Write purchases.csv
    with open(out / "purchases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "purchased_item_id"])
        w.writeheader()
        w.writerows(purchases)

    # Stats
    print(f"Items: {len(items)}")
    print(f"Users: {len(users)}")
    print(f"Sessions: {len(purchases)}")
    print(f"Total interactions: {len(all_interactions)}")
    print(f"Avg interactions/session: {len(all_interactions)/len(purchases):.1f}")

    # Purchase analysis
    p_items = [p["purchased_item_id"] for p in purchases]
    purchased_set = set()
    cart_and_purchased = 0
    click_and_purchased = 0
    for sid in range(args.sessions):
        sess_ints = [r for r in all_interactions if r["session_id"] == sid]
        pid = purchases[sid]["purchased_item_id"]
        purchased_set.add(pid)
        actions_for_pid = {r["action_type"] for r in sess_ints if r["item_id"] == pid}
        if "cart" in actions_for_pid:
            cart_and_purchased += 1
        if "click" in actions_for_pid:
            click_and_purchased += 1
        if sid >= 999:
            break

    n_checked = min(1000, args.sessions)
    print(f"\n--- Purchase signal analysis (first {n_checked} sessions) ---")
    print(f"Purchased item was carted: {cart_and_purchased}/{n_checked} ({100*cart_and_purchased/n_checked:.1f}%)")
    print(f"Purchased item was clicked: {click_and_purchased}/{n_checked} ({100*click_and_purchased/n_checked:.1f}%)")

    # Action distribution
    actions = Counter(r["action_type"] for r in all_interactions)
    print(f"\nAction distribution: {dict(sorted(actions.items()))}")

    # Items per session distribution
    sess_sizes = Counter()
    for r in all_interactions:
        pass
    sess_item_counts = defaultdict(set)
    for r in all_interactions:
        sess_item_counts[r["session_id"]].add(r["item_id"])
    sizes = [len(v) for v in sess_item_counts.values()]
    print(f"Items per session: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")


if __name__ == "__main__":
    main()
