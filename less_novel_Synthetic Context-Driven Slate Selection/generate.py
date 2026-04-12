"""
Data generator for Synthetic Context-Driven Slate Selection.

Concept:
- 400 users, each with a hidden per-context REGIME (1 of 10) that determines
  their preference vector. The regime depends on 3 of 6 context features via
  a per-user transition matrix.
- 800 items with hidden 8-dim attribute vectors. ~15% are copycat items.
- 20,000 queries: each presents a user + 6 context features + slate of 10
  candidate items. The user selects one item from the slate based on
  regime-dependent preference matching + noise.
- ~12% of selections are purely random (irreducible noise).
- Red-herring context features: 3 of 6 context features do NOT drive regime.
- Noise item features (buzz) add no signal.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate(out_dir: Path = Path("raw_data"), seed: int = 42):
    rng = np.random.RandomState(seed)

    N_USERS = 400
    N_ITEMS = 800
    N_REGIMES = 10
    N_HIDDEN = 8
    N_QUERIES = 20_000
    SLATE_SIZE = 10
    NOISE_RATE = 0.12
    N_CTX_COMBOS = 72  # 6 time_slots * 4 devices * 3 day_types

    # ---- Regime preference vectors (10 x 8) ----
    regime_prefs = rng.randn(N_REGIMES, N_HIDDEN)
    for r in range(N_REGIMES):
        strength = rng.uniform(0.8, 2.0)
        norm = np.linalg.norm(regime_prefs[r]) + 1e-8
        regime_prefs[r] = regime_prefs[r] / norm * strength

    # ---- Item hidden attributes (800 x 8) ----
    item_hidden = rng.randn(N_ITEMS, N_HIDDEN) * 0.5
    copycat_mask = np.zeros(N_ITEMS, dtype=bool)
    for i in range(N_ITEMS):
        if rng.random() < 0.15:
            parent = rng.randint(0, N_ITEMS)
            while parent == i:
                parent = rng.randint(0, N_ITEMS)
            item_hidden[i] = item_hidden[parent] + rng.randn(N_HIDDEN) * 0.08
            copycat_mask[i] = True

    # ---- User regime transition matrices (400 x 72) ----
    user_trans = np.zeros((N_USERS, N_CTX_COMBOS), dtype=int)
    user_types = []

    for u in range(N_USERS):
        r = rng.random()
        if r > 0.80:
            base = rng.randint(0, N_REGIMES)
            user_trans[u, :] = base
            user_types.append("stable")
        elif r < 0.15:
            user_trans[u] = rng.randint(0, N_REGIMES, N_CTX_COMBOS)
            user_types.append("volatile")
        else:
            n_dom = rng.randint(2, 5)
            doms = rng.choice(N_REGIMES, n_dom, replace=False)
            for c in range(N_CTX_COMBOS):
                user_trans[u, c] = doms[c % n_dom]
            user_types.append("normal")

    # ---- Observable user features ----
    dom_regime = np.array([
        np.bincount(user_trans[u], minlength=N_REGIMES).argmax()
        for u in range(N_USERS)
    ])

    users_df = pd.DataFrame({
        "user_id": np.arange(N_USERS),
        "affinity_1": (regime_prefs[dom_regime, 0] + rng.normal(0, 0.8, N_USERS)).round(3),
        "affinity_2": (regime_prefs[dom_regime, 1] + rng.normal(0, 0.8, N_USERS)).round(3),
        "affinity_3": (regime_prefs[dom_regime, 2] + rng.normal(0, 0.8, N_USERS)).round(3),
        "segment": rng.randint(0, 6, N_USERS),
        "cohort": rng.randint(0, 4, N_USERS),
        "tenure_score": rng.uniform(0, 1, N_USERS).round(3),
    })

    # ---- Observable item features ----
    items_df = pd.DataFrame({
        "item_id": np.arange(N_ITEMS),
        "feature_a": (item_hidden[:, 0] + rng.normal(0, 0.3, N_ITEMS)).round(3),
        "feature_b": (item_hidden[:, 1] + rng.normal(0, 0.3, N_ITEMS)).round(3),
        "feature_c": (item_hidden[:, 2] + rng.normal(0, 0.3, N_ITEMS)).round(3),
        "group": ((item_hidden[:, 3] * 5 + 10 + rng.randint(0, 3, N_ITEMS)) % 20).astype(int),
        "level": ((item_hidden[:, 4] * 2 + 2.5 + rng.normal(0, 0.5, N_ITEMS)).clip(0, 4)).astype(int),
        "score_x": (item_hidden[:, 5] + rng.normal(0, 0.5, N_ITEMS)).round(3),
        "score_y": (item_hidden[:, 6] + rng.normal(0, 0.5, N_ITEMS)).round(3),
        "buzz": rng.normal(0, 1, N_ITEMS).round(3),
    })

    # ---- Precompute item-regime scores ----
    all_scores = item_hidden @ regime_prefs.T  # (N_ITEMS, N_REGIMES)

    # ---- Generate queries ----
    rows = []
    for q in range(N_QUERIES):
        uid = rng.randint(0, N_USERS)
        ts = rng.randint(0, 6)
        dev = rng.randint(0, 4)
        dt = rng.randint(0, 3)
        ref = rng.randint(0, 10)
        slen = rng.randint(1, 121)
        ep = rng.randint(0, 5)

        ctx_hash = (ts * 12 + dev * 3 + dt) % N_CTX_COMBOS
        regime = user_trans[uid, ctx_hash]

        # Build slate: 3-4 compatible + rest random
        regime_scores = all_scores[:, regime]
        top50 = np.argsort(regime_scores)[-50:]
        n_compat = rng.randint(3, 5)
        compat = rng.choice(top50, min(n_compat, len(top50)), replace=False).tolist()

        available = np.setdiff1d(np.arange(N_ITEMS), compat)
        rest = rng.choice(available, SLATE_SIZE - len(compat), replace=False).tolist()
        slate = compat + rest
        rng.shuffle(slate)

        # Selection
        slate_scores = np.array([np.dot(regime_prefs[regime], item_hidden[s]) for s in slate])
        if rng.random() < NOISE_RATE:
            chosen_idx = rng.randint(0, SLATE_SIZE)
        else:
            noisy = slate_scores + rng.gumbel(0, 0.3, SLATE_SIZE)
            chosen_idx = int(np.argmax(noisy))

        rows.append({
            "query_id": q,
            "user_id": uid,
            "time_slot": ts,
            "device_code": dev,
            "day_type": dt,
            "referral_code": ref,
            "session_length": slen,
            "entry_point": ep,
            "candidates": "|".join(str(s) for s in slate),
            "chosen_item_id": slate[chosen_idx],
        })

    queries_df = pd.DataFrame(rows)

    # ---- Save ----
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    users_df.to_csv(out_dir / "users.csv", index=False)
    items_df.to_csv(out_dir / "items.csv", index=False)
    queries_df.to_csv(out_dir / "queries.csv", index=False)

    n_vol = sum(1 for t in user_types if t == "volatile")
    n_stab = sum(1 for t in user_types if t == "stable")
    n_copy = int(copycat_mask.sum())
    print(f"Users: {N_USERS} (stable={n_stab}, volatile={n_vol})")
    print(f"Items: {N_ITEMS} (copycat={n_copy})")
    print(f"Queries: {N_QUERIES}, Slate size: {SLATE_SIZE}")
    print(f"Noise rate: {NOISE_RATE}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    generate()
