from pathlib import Path


def prepare(raw: Path, public: Path, private: Path) -> None:
    import random as _rnd
    import numpy as np
    import pandas as pd

    raw, public, private = Path(raw), Path(public), Path(private)

    ground_truth = pd.read_csv(raw / "ground_truth.csv")
    statements = pd.read_csv(raw / "statements.csv")

    rng = _rnd.Random(161803398)
    np_rng = np.random.RandomState(88)

    FIELDS = ["actor", "action", "location", "time_period", "severity", "contributing_factor"]

    # ---- Obfuscate field values with coded labels ----
    field_maps = {}
    for f in FIELDS:
        unique_vals = sorted(ground_truth[f].unique().tolist())
        n = len(unique_vals)
        codes = [f"V{f[0].upper()}{i:03d}" for i in range(n)]
        rng.shuffle(codes)
        mapping = {unique_vals[i]: codes[i] for i in range(n)}
        field_maps[f] = mapping
        ground_truth[f] = ground_truth[f].map(mapping)

    # Apply same mapping to statements text
    def _obfuscate_statement(text):
        result = text
        for f, mapping in field_maps.items():
            for original, coded in mapping.items():
                result = result.replace(original, coded)
        return result

    statements["statement"] = statements["statement"].apply(_obfuscate_statement)

    # ---- Obfuscate witness roles ----
    unique_roles = sorted(statements["witness_role"].unique().tolist())
    role_codes = [f"SRC_{i:02d}" for i in range(len(unique_roles))]
    rng.shuffle(role_codes)
    role_map = {unique_roles[i]: role_codes[i] for i in range(len(unique_roles))}
    statements["witness_role"] = statements["witness_role"].map(role_map)

    # ---- Split 80/20 by incident ----
    all_ids = sorted(ground_truth["incident_id"].tolist())
    rng_split = _rnd.Random(141421356)
    rng_split.shuffle(all_ids)

    split_idx = int(len(all_ids) * 0.8)
    train_ids = set(all_ids[:split_idx])
    test_ids = set(all_ids[split_idx:])

    train_gt = ground_truth[ground_truth["incident_id"].isin(train_ids)].copy()
    test_gt = ground_truth[ground_truth["incident_id"].isin(test_ids)].copy()
    train_stmts = statements[statements["incident_id"].isin(train_ids)].copy()
    test_stmts = statements[statements["incident_id"].isin(test_ids)].copy()

    test_stmts_rows = []
    for inc_id in sorted(test_ids):
        inc_stmts = test_stmts[test_stmts["incident_id"] == inc_id]
        n_drop = min(2, max(0, len(inc_stmts) - 2))
        if n_drop > 0:
            indices = sorted(inc_stmts.index.tolist())
            rng.shuffle(indices)
            drop_indices = indices[:n_drop]
            inc_stmts = inc_stmts.drop(drop_indices)
        test_stmts_rows.append(inc_stmts)
    test_stmts = pd.concat(test_stmts_rows, ignore_index=True)

    n_noise = int(len(test_stmts) * 0.08)
    noise_rows = []
    for _ in range(n_noise):
        inc_id = rng.choice(sorted(test_ids))
        noise_text = (
            f"{rng.choice(role_codes)} reported seeing "
            f"{rng.choice(list(field_maps['actor'].values()))} "
            f"{rng.choice(list(field_maps['action'].values()))} "
            f"at {rng.choice(list(field_maps['location'].values()))} "
            f"during the {rng.choice(list(field_maps['time_period'].values()))}. "
            f"Severity appeared {rng.choice(list(field_maps['severity'].values()))}. "
            f"Contributing factor was {rng.choice(list(field_maps['contributing_factor'].values()))}."
        )
        noise_rows.append({
            "incident_id": inc_id,
            "witness_idx": 99,
            "witness_role": rng.choice(role_codes),
            "statement": noise_text,
        })
    noise_df = pd.DataFrame(noise_rows)
    test_stmts = pd.concat([test_stmts, noise_df], ignore_index=True)
    test_stmts = test_stmts.sort_values(["incident_id", "witness_idx"]).reset_index(drop=True)

    # ---- Sort everything ----
    train_gt = train_gt.sort_values("incident_id").reset_index(drop=True)
    test_gt = test_gt.sort_values("incident_id").reset_index(drop=True)
    train_stmts = train_stmts.sort_values(["incident_id", "witness_idx"]).reset_index(drop=True)

    # ---- Sample submission: most frequent value per field from train ----
    sample_sub = test_gt[["incident_id"]].copy()
    for f in FIELDS:
        mode_val = train_gt[f].mode().iloc[0]
        sample_sub[f] = mode_val

    # ---- Write outputs ----
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_stmts.to_csv(public / "train_statements.csv", index=False)
    train_gt[["incident_id"] + FIELDS].to_csv(public / "train_labels.csv", index=False)
    test_stmts.to_csv(public / "test_statements.csv", index=False)
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    test_gt[["incident_id"] + FIELDS].to_csv(private / "answers.csv", index=False)
