from pathlib import Path


def prepare(raw: Path, public: Path, private: Path) -> None:
    import random as _rnd
    import numpy as np
    import pandas as pd

    raw, public, private = Path(raw), Path(public), Path(private)

    users = pd.read_csv(raw / "users.csv")
    items = pd.read_csv(raw / "items.csv")
    queries = pd.read_csv(raw / "queries.csv")

    SPLIT_SEED = 271828
    TRAIN_FRAC = 0.75

    np_rng = np.random.RandomState(SPLIT_SEED)

    # ---- Anonymize user columns ----
    u_cols = [c for c in users.columns if c != "user_id"]
    rng_u = _rnd.Random(314159)
    shuf_u = list(u_cols)
    rng_u.shuffle(shuf_u)
    u_map = {orig: f"UF_{i:02d}" for i, orig in enumerate(shuf_u)}
    users = users.rename(columns=u_map)
    users["UF_06"] = np_rng.normal(0, 1, len(users)).round(3)

    # ---- Anonymize item columns ----
    i_cols = [c for c in items.columns if c != "item_id"]
    rng_i = _rnd.Random(161803)
    shuf_i = list(i_cols)
    rng_i.shuffle(shuf_i)
    i_map = {orig: f"IF_{i:02d}" for i, orig in enumerate(shuf_i)}
    items = items.rename(columns=i_map)
    items["IF_08"] = np_rng.normal(0, 1, len(items)).round(3)
    items["IF_09"] = np_rng.uniform(0, 1, len(items)).round(3)

    # ---- Anonymize context columns ----
    ctx_orig = ["time_slot", "device_code", "day_type",
                "referral_code", "session_length", "entry_point"]
    rng_c = _rnd.Random(141421)
    shuf_c = list(ctx_orig)
    rng_c.shuffle(shuf_c)
    c_map = {orig: f"C_{i:02d}" for i, orig in enumerate(shuf_c)}
    queries = queries.rename(columns=c_map)

    # ---- Split queries 75/25 ----
    all_qids = sorted(queries["query_id"].tolist())
    rng_split = _rnd.Random(SPLIT_SEED)
    rng_split.shuffle(all_qids)
    split_pt = int(len(all_qids) * TRAIN_FRAC)
    train_qids = set(all_qids[:split_pt])
    test_qids = set(all_qids[split_pt:])

    train_q = queries[queries["query_id"].isin(train_qids)].copy()
    test_q = queries[queries["query_id"].isin(test_qids)].copy()
    train_q = train_q.sort_values("query_id").reset_index(drop=True)
    test_q = test_q.sort_values("query_id").reset_index(drop=True)

    # ---- Expand candidates into separate table ----
    def _expand(df):
        exp = df[["query_id", "candidates"]].copy()
        exp["candidates"] = exp["candidates"].astype(str).str.split("|")
        exp = exp.explode("candidates").rename(columns={"candidates": "item_id"})
        exp["item_id"] = exp["item_id"].astype(int)
        return exp.reset_index(drop=True)

    train_cands = _expand(train_q)
    test_cands = _expand(test_q)

    ctx_new = sorted(c_map.values())

    # ---- Write ----
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    users.to_csv(public / "users.csv", index=False)
    items.to_csv(public / "items.csv", index=False)

    train_q[["query_id", "user_id"] + ctx_new].to_csv(
        public / "train_queries.csv", index=False
    )
    train_cands.to_csv(public / "train_candidates.csv", index=False)
    train_q[["query_id", "chosen_item_id"]].to_csv(
        public / "train_labels.csv", index=False
    )

    test_q[["query_id", "user_id"] + ctx_new].to_csv(
        public / "test_queries.csv", index=False
    )
    test_cands.to_csv(public / "test_candidates.csv", index=False)

    sample = test_q[["query_id"]].copy()
    sample["chosen_item_id"] = 0
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_q[["query_id", "chosen_item_id"]].to_csv(
        private / "answers.csv", index=False
    )

    print(f"Train queries: {len(train_q)}")
    print(f"Test queries:  {len(test_q)}")
    print(f"Train candidates rows: {len(train_cands)}")
    print(f"Test candidates rows:  {len(test_cands)}")
    print(f"Users: {len(users)}, Items: {len(items)}")


if __name__ == "__main__":
    prepare(Path("raw_data"), Path("pub"), Path("priv"))
