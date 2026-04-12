"""
Prepare script: transforms raw CSVs into public/ and private/ splits.

Obfuscation pipeline:
  1. Z-score normalize dwell_seconds per session
  2. Quantile-bin scroll_pct into 5 global buckets
  3. Anonymize item categories (consistent mapping)
  4. Scramble price tier labels (consistent mapping)
  5. Rename + noise user/item feature columns
  6. Inject ~2 noise interactions per session
  7. Merge revisit interactions for ~25% of sessions
  8. Perturb position ordering for ~20% of sessions
  9. Cold-start: drop user profiles for ~15% of test session users

Split: 80/20 by session_id (deterministic).
"""

from pathlib import Path
import hashlib
import random as _rnd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _det_seed(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def prepare(raw: Path, public: Path, private: Path) -> None:
    interactions = pd.read_csv(raw / "interactions.csv")
    items = pd.read_csv(raw / "items.csv")
    users = pd.read_csv(raw / "users.csv")
    purchases = pd.read_csv(raw / "purchases.csv")

    rng = _rnd.Random(_det_seed("prepare_42"))
    np_rng = np.random.RandomState(42)

    # --- Build anonymization mappings ---
    cat_ids = sorted(items["category"].unique())
    shuffled_cats = list(cat_ids)
    rng.shuffle(shuffled_cats)
    cat_map = {orig: f"CAT_{i:02d}" for i, orig in enumerate(shuffled_cats)}

    tier_ids = sorted(items["price_tier"].unique())
    tier_labels = ["PT_W", "PT_X", "PT_Y", "PT_Z", "PT_V"]
    rng.shuffle(tier_labels)
    tier_map = {orig: tier_labels[i] for i, orig in enumerate(tier_ids)}

    # --- Obfuscate items ---
    items["category"] = items["category"].map(cat_map)
    items["price_tier"] = items["price_tier"].map(tier_map)
    items["attr_1"] += np_rng.normal(0, 0.3, len(items))
    items["attr_2"] += np_rng.normal(0, 0.3, len(items))
    items["attr_3"] += np_rng.normal(0, 0.3, len(items))
    items = items.rename(columns={
        "attr_1": "if_1", "attr_2": "if_2", "attr_3": "if_3",
        "category": "cat_code", "price_tier": "tier_code",
    })
    items = items.round({"if_1": 3, "if_2": 3, "if_3": 3})

    # --- Obfuscate users ---
    users["pref_cat_1"] = users["pref_cat_1"].map(cat_map)
    users["pref_cat_2"] = users["pref_cat_2"].map(cat_map)
    users["pref_cat_3"] = users["pref_cat_3"].map(cat_map)
    users["tier_low"] = users["tier_low"].map(tier_map)
    users["tier_high"] = users["tier_high"].map(tier_map)
    users["pref_1"] += np_rng.normal(0, 0.4, len(users))
    users["pref_2"] += np_rng.normal(0, 0.4, len(users))
    users["pref_3"] += np_rng.normal(0, 0.4, len(users))
    users = users.rename(columns={
        "pref_cat_1": "uf_1", "pref_cat_2": "uf_2", "pref_cat_3": "uf_3",
        "tier_low": "uf_4", "tier_high": "uf_5",
        "pref_1": "uf_6", "pref_2": "uf_7", "pref_3": "uf_8",
    })
    users = users.round({"uf_6": 3, "uf_7": 3, "uf_8": 3})

    # --- Obfuscate interactions ---

    # 1. Z-score normalize dwell per session
    interactions["dwell_seconds"] = interactions.groupby(
        "session_id"
    )["dwell_seconds"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0
    ).round(3)

    # 2. Quantile-bin scroll_pct into 5 global bins
    interactions["scroll_pct"] = pd.qcut(
        interactions["scroll_pct"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    # 3. Inject noise interactions (~2 per session)
    all_item_ids = items["item_id"].tolist()
    noise_rows = []
    for sid in interactions["session_id"].unique():
        sess = interactions[interactions["session_id"] == sid]
        uid = sess["user_id"].iloc[0]
        max_pos = sess["position"].max()
        n_noise = rng.randint(1, 3)
        for j in range(n_noise):
            noise_rows.append({
                "session_id": sid,
                "user_id": uid,
                "item_id": rng.choice(all_item_ids),
                "action_type": "view",
                "dwell_seconds": round(np_rng.normal(0, 0.8), 3),
                "scroll_pct": rng.randint(1, 5),
                "position": max_pos + j + 1,
            })
    noise_df = pd.DataFrame(noise_rows)
    interactions = pd.concat([interactions, noise_df], ignore_index=True)

    # 4. Merge revisits for ~25% of sessions
    merge_sessions = set()
    for sid in interactions["session_id"].unique():
        if _det_seed(f"merge_{sid}") % 100 < 25:
            merge_sessions.add(sid)

    merged_parts = []
    for sid, group in interactions.groupby("session_id"):
        if sid in merge_sessions:
            agg = group.groupby("item_id").agg({
                "session_id": "first", "user_id": "first",
                "action_type": "last", "dwell_seconds": "max",
                "scroll_pct": "max", "position": "max",
            }).reset_index()
            merged_parts.append(agg)
        else:
            merged_parts.append(group)
    interactions = pd.concat(merged_parts, ignore_index=True)

    # 5. Perturb positions for ~20% of sessions
    perturbed_parts = []
    for sid, group in interactions.groupby("session_id"):
        if _det_seed(f"pos_{sid}") % 100 < 20:
            group = group.copy()
            positions = group["position"].values.copy()
            n = len(positions)
            for i in range(n):
                if rng.random() < 0.3:
                    j = rng.randint(0, n - 1)
                    positions[i], positions[j] = positions[j], positions[i]
            group["position"] = positions
        perturbed_parts.append(group)
    interactions = pd.concat(perturbed_parts, ignore_index=True)

    # Rename interaction columns
    interactions = interactions.rename(columns={
        "dwell_seconds": "signal_1",
        "scroll_pct": "signal_2",
        "action_type": "event_type",
        "position": "seq_pos",
    })

    # --- Split sessions 80/20 ---
    session_ids = sorted(purchases["session_id"].unique())
    train_sids, test_sids = train_test_split(
        session_ids, test_size=0.2, random_state=42
    )
    train_sids_set = set(train_sids)
    test_sids_set = set(test_sids)

    train_ints = interactions[interactions["session_id"].isin(train_sids_set)]
    test_ints = interactions[interactions["session_id"].isin(test_sids_set)]
    train_purchases = purchases[purchases["session_id"].isin(train_sids_set)]
    test_purchases = purchases[purchases["session_id"].isin(test_sids_set)]

    # 6. Cold-start: drop ~15% of test session users from users table
    test_user_ids = test_ints["user_id"].unique()
    cold_users = set()
    for uid in test_user_ids:
        if _det_seed(f"cold_{uid}") % 100 < 15:
            cold_users.add(uid)
    users_public = users[~users["user_id"].isin(cold_users)]

    # --- Write outputs ---
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_ints.to_csv(public / "train_sessions.csv", index=False)
    train_purchases.to_csv(public / "train_labels.csv", index=False)
    test_ints.to_csv(public / "test_sessions.csv", index=False)
    items.to_csv(public / "items.csv", index=False)
    users_public.to_csv(public / "users.csv", index=False)

    sample = test_purchases[["session_id"]].copy()
    sample["purchased_item_id"] = 0
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_purchases.to_csv(private / "answers.csv", index=False)

    print(f"Train sessions: {len(train_sids)}, Test sessions: {len(test_sids)}")
    print(f"Train interactions: {len(train_ints)}, Test interactions: {len(test_ints)}")
    print(f"Cold-start test users dropped: {len(cold_users)}")
    print(f"Users in public file: {len(users_public)}")
    print(f"Noise interactions injected, revisits merged, positions perturbed")


if __name__ == "__main__":
    prepare()
