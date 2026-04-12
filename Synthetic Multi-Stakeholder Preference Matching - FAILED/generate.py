"""
Generate synthetic two-sided preference matching data.

Domain: Fictional service engagement platform where requestors browse
candidate profiles and select one per engagement session.

Hidden selection function:
  - Category alignment (discrete, non-linear)
  - Feature compatibility (dot-product of hidden weights × attributes)
  - Position bias (slight)
  - Session-type variation: normal / impulse / passive
  - Gaussian noise

Output: raw_data/ with interactions.csv, candidates.csv, requestors.csv, selections.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    rng = np.random.RandomState(42)

    N_CANDIDATES = 500
    N_REQUESTORS = 3000
    N_SESSIONS = 15000
    N_CATEGORIES = 20
    N_FEAT = 6

    # ---- Candidates (providers) ----
    cand_cats = rng.randint(0, N_CATEGORIES, N_CANDIDATES)
    cand_feats = rng.randn(N_CANDIDATES, N_FEAT)
    cand_popularity = rng.uniform(0, 1, N_CANDIDATES)

    candidates = pd.DataFrame({"candidate_id": range(N_CANDIDATES)})
    candidates["category"] = cand_cats
    for i in range(N_FEAT):
        candidates[f"cf_{i+1}"] = cand_feats[:, i].round(4)
    candidates["popularity"] = cand_popularity.round(4)

    # ---- Requestors ----
    req_pref_cats = rng.randint(0, N_CATEGORIES, (N_REQUESTORS, 3))
    req_weights = rng.randn(N_REQUESTORS, N_FEAT) * 0.8
    req_cat_strength = rng.uniform(0.5, 2.0, N_REQUESTORS)

    requestors = pd.DataFrame({"requestor_id": range(N_REQUESTORS)})
    for i in range(3):
        requestors[f"pref_cat_{i+1}"] = req_pref_cats[:, i]
    for i in range(N_FEAT):
        requestors[f"rw_{i+1}"] = req_weights[:, i].round(4)
    requestors["cat_strength"] = req_cat_strength.round(4)

    # ---- Sessions ----
    interaction_rows = []
    selection_rows = []

    for sid in range(N_SESSIONS):
        rq_id = rng.randint(0, N_REQUESTORS)
        n_cands = rng.randint(6, 15)
        pool = rng.choice(N_CANDIDATES, n_cands, replace=False)

        roll = rng.random()
        if roll < 0.15:
            stype = "passive"
        elif roll < 0.40:
            stype = "impulse"
        else:
            stype = "normal"

        rw = req_weights[rq_id]
        rc = req_pref_cats[rq_id]
        rcs = req_cat_strength[rq_id]

        pf = cand_feats[pool]
        pc = cand_cats[pool]
        pp = cand_popularity[pool]

        cat_scores = np.zeros(n_cands)
        cat_scores[pc == rc[0]] = 1.5
        mask2 = (pc == rc[1]) & (cat_scores < 0.8)
        cat_scores[mask2] = 0.8
        mask3 = (pc == rc[2]) & (cat_scores < 0.3)
        cat_scores[mask3] = 0.3

        compats = pf @ rw
        pos_bias = -0.04 * np.arange(n_cands)
        pop_bonus = pp * 0.3
        noise = rng.normal(0, 0.8, n_cands)

        if stype == "passive":
            totals = cat_scores * rcs + pop_bonus + rng.normal(0, 1.2, n_cands)
        else:
            totals = (
                cat_scores * rcs * 0.3
                + compats * 0.5
                + pop_bonus * 0.1
                + pos_bias
                + noise
            )

        if stype == "impulse":
            top_k = min(3, n_cands)
            top_idx = np.argsort(totals)[-top_k:]
            selected_idx = rng.choice(top_idx)
        else:
            selected_idx = int(np.argmax(totals))

        selected_pid = int(pool[selected_idx])

        for pos in range(n_cands):
            pid = int(pool[pos])
            eng = compats[pos] * 0.3 + cat_scores[pos] * 0.2 + rng.normal(0, 1)
            dwell = max(0.1, eng * 0.5 + rng.exponential(2))
            depth = float(np.clip(compats[pos] * 0.1 + rng.uniform(0, 1), 0, 1))
            revisit = 1 if rng.random() < 0.15 else 0

            interaction_rows.append({
                "session_id": sid,
                "requestor_id": rq_id,
                "candidate_id": pid,
                "position": pos,
                "engagement": round(eng, 3),
                "dwell_time": round(dwell, 3),
                "browse_depth": round(depth, 3),
                "revisit": revisit,
            })

        selection_rows.append({
            "session_id": sid,
            "selected_candidate_id": selected_pid,
        })

    interactions = pd.DataFrame(interaction_rows)
    selections = pd.DataFrame(selection_rows)

    out = Path("raw_data")
    out.mkdir(exist_ok=True)
    interactions.to_csv(out / "interactions.csv", index=False)
    candidates.to_csv(out / "candidates.csv", index=False)
    requestors.to_csv(out / "requestors.csv", index=False)
    selections.to_csv(out / "selections.csv", index=False)

    print(f"Interactions: {len(interactions):,} rows")
    print(f"Candidates: {len(candidates)} | Requestors: {len(requestors)}")
    print(f"Sessions: {len(selections):,}")
    print(f"Avg candidates/session: {len(interactions)/len(selections):.1f}")


if __name__ == "__main__":
    main()
