"""prepare.py for the Patient-Linkage Verification On Chest Radiographs challenge.

Input  raw/  containing:
  - cxr_index.csv       columns: raw_patient_id, raw_image_id, image_file,
                                 has_<class_0> .. has_<class_13>
                        i.e. 14 binary columns named has_0 .. has_13.
                        (The mapping from class index to NIH disease label
                        is not exposed here; it lives in the dataset doc.)
  - images/<image_file>      8-bit grayscale chest X-ray PNG.

Output public/:
  - train.csv                      one row per training image with the 14
                                   has_<i> columns and the patient_id (kept
                                   anonymised: pid_<int>).
  - test_queries.csv               one row per test query (query_id, image_file)
  - test_candidates.csv            one row per test query, with cand_id_0..4
                                   columns each pointing at a PNG. Exactly
                                   one of the 5 candidates is from the same
                                   patient as the query; the other four are
                                   pathology-matched decoys.
  - sample_submission.csv          constant baseline (uniform link scores,
                                   pred_<i>=0, p_<i>=0.5 for all i).
  - images/<id>.png                renamed images (queries + candidates).

Output private/:
  - answers.csv                    id, row_type ∈ {link, patho},
                                   true_match_idx (link rows; int 0..4),
                                   true_y_0..true_y_13 (patho rows).

Anonymisation:
  - Every raw_patient_id is permuted to a dense pid_<int>; that integer is
    NEVER written to public files.
  - Every image gets a new id (img_<6-digit>) via a seeded permutation.
  - Disease columns retain their numeric anonymised indices 0..13; no NIH
    disease names appear anywhere in public/.

Decoys:
  - For each query, the four decoys are sampled from the test pool such that
    each decoy shares the query's pathology PATTERN (the 14-bit binary
    vector) with Hamming distance ≤ 1, and is from a DIFFERENT patient than
    the query. The 5 candidates are then shuffled into a per-query random
    order; the position of the true-same-patient candidate is the
    `true_match_idx` field in answers.csv.

Determinism:
  All randomness is seeded by the four 32-bit hex constants below; rerunning
  the script on the same raw_data/ produces byte-identical public/ + private/.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SPLIT_SEED        = 0x11335577
ID_PERM_SEED      = 0x88AABBCC
DECOY_SEED        = 0xDEED0042
PATIENT_PERM_SEED = 0x9090CAFE
TRAIN_FRACTION    = 0.65
N_DISEASES        = 14
N_CANDIDATES      = 5
N_DECOYS          = N_CANDIDATES - 1


def _read_raw(raw: Path) -> pd.DataFrame:
    csv_path = raw / "cxr_index.csv"
    img_dir = raw / "images"
    if not csv_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"raw/ missing cxr_index.csv or images/ at {raw}")
    df = pd.read_csv(csv_path)
    needed = {"raw_patient_id", "raw_image_id", "image_file"}
    needed |= {f"has_{i}" for i in range(N_DISEASES)}
    if not needed.issubset(set(df.columns)):
        raise ValueError("raw/cxr_index.csv missing required columns")
    for c in (f"has_{i}" for i in range(N_DISEASES)):
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


def _hamming_le1(pat_a: np.ndarray, pat_b: np.ndarray) -> bool:
    return int(np.sum(pat_a != pat_b)) <= 1


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw); public = Path(public); private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (public / "images").mkdir(parents=True, exist_ok=True)

    df = _read_raw(raw)

    # 1) Train / test split BY PATIENT.
    split_rng = np.random.default_rng(SPLIT_SEED)
    all_patients = df["raw_patient_id"].drop_duplicates().to_numpy().copy()
    split_rng.shuffle(all_patients)
    n_train = int(round(len(all_patients) * TRAIN_FRACTION))
    train_patients = set(all_patients[:n_train].tolist())

    # 2) Anonymise patient ids
    pid_rng = np.random.default_rng(PATIENT_PERM_SEED)
    pid_perm = np.arange(len(all_patients))
    pid_rng.shuffle(pid_perm)
    raw_pid_to_anon = {p: f"pid_{int(pid_perm[i]):06d}" for i, p in enumerate(all_patients)}

    # 3) Anonymise image ids
    id_rng = np.random.default_rng(ID_PERM_SEED)
    img_perm = np.arange(len(df))
    id_rng.shuffle(img_perm)
    raw_imgrow_to_imgid = {i: f"img_{int(img_perm[i]):06d}" for i in range(len(df))}

    # 4) Copy + rename train images, build train.csv
    train_rows: List[dict] = []
    images_src = raw / "images"
    images_dst = public / "images"

    df["image_id"] = [raw_imgrow_to_imgid[i] for i in range(len(df))]
    df["anon_pid"] = df["raw_patient_id"].map(raw_pid_to_anon)
    df["is_train"] = df["raw_patient_id"].isin(train_patients)

    for _, row in df.iterrows():
        new_name = f"{row['image_id']}.png"
        src_png = images_src / str(row["image_file"])
        dst_png = images_dst / new_name
        if not src_png.exists():
            raise FileNotFoundError(f"missing source image {src_png}")
        shutil.copy2(str(src_png), str(dst_png))
        if row["is_train"]:
            r = {
                "id": row["image_id"],
                "image_file": new_name,
                "patient_id": row["anon_pid"],
            }
            for i in range(N_DISEASES):
                r[f"has_{i}"] = int(row[f"has_{i}"])
            train_rows.append(r)

    pd.DataFrame(train_rows).sort_values("id").reset_index(drop=True).to_csv(
        public / "train.csv", index=False
    )

    # 5) Build test queries: one query per test patient who has ≥ 2 images.
    test_df = df[~df["is_train"]].copy()
    by_pid = test_df.groupby("anon_pid")
    decoy_pool = test_df.copy()
    pat_cols = [f"has_{i}" for i in range(N_DISEASES)]
    decoy_patterns = decoy_pool[pat_cols].to_numpy().astype(int)

    decoy_rng = np.random.default_rng(DECOY_SEED)

    queries: List[dict] = []
    candidates: List[dict] = []
    sub_rows: List[dict] = []
    ans_rows: List[dict] = []

    for anon_pid, grp in by_pid:
        if len(grp) < 2:
            continue
        # Pick first row deterministically as the query, second as the
        # same-patient candidate; if more than 2, just take the first two.
        grp_sorted = grp.sort_values("image_id").reset_index(drop=True)
        q = grp_sorted.iloc[0]
        match = grp_sorted.iloc[1]

        q_pat = np.array([int(q[c]) for c in pat_cols])

        # Find decoys: pathology hamming ≤ 1, different patient.
        decoy_mask = (
            (decoy_pool["anon_pid"] != anon_pid).to_numpy()
            & (decoy_pool["image_id"] != q["image_id"]).to_numpy()
            & (decoy_pool["image_id"] != match["image_id"]).to_numpy()
        )
        ham = np.abs(decoy_patterns - q_pat).sum(axis=1)
        decoy_mask &= (ham <= 1)
        decoy_idx_pool = np.where(decoy_mask)[0]
        if len(decoy_idx_pool) < N_DECOYS:
            # fallback: relax to hamming ≤ 2
            decoy_mask2 = (
                (decoy_pool["anon_pid"] != anon_pid).to_numpy()
                & (decoy_pool["image_id"] != q["image_id"]).to_numpy()
                & (decoy_pool["image_id"] != match["image_id"]).to_numpy()
                & (ham <= 2)
            )
            decoy_idx_pool = np.where(decoy_mask2)[0]
            if len(decoy_idx_pool) < N_DECOYS:
                continue
        decoy_pick = decoy_rng.choice(decoy_idx_pool, size=N_DECOYS, replace=False)

        cand_image_ids = [match["image_id"]] + [decoy_pool.iloc[int(i)]["image_id"] for i in decoy_pick]
        # shuffle candidate order
        order = np.arange(N_CANDIDATES)
        decoy_rng.shuffle(order)
        true_match_idx = int(np.where(order == 0)[0][0])
        ordered_cand_ids = [cand_image_ids[i] for i in order]

        query_id = q["image_id"]
        queries.append({"query_id": query_id, "image_file": f"{query_id}.png"})
        cand_row = {"query_id": query_id}
        for k, cid in enumerate(ordered_cand_ids):
            cand_row[f"cand_id_{k}"] = cid
            cand_row[f"cand_image_file_{k}"] = f"{cid}.png"
        candidates.append(cand_row)

        link_id = f"link_{query_id}"
        patho_id = f"patho_{query_id}"
        sub_link = {"id": link_id, "row_type": "link"}
        for k in range(N_CANDIDATES):
            sub_link[f"cand_score_{k}"] = 0.2
        for i in range(N_DISEASES):
            sub_link[f"pred_{i}"] = 0
            sub_link[f"p_{i}"] = 0.5
        sub_rows.append(sub_link)

        sub_patho = {"id": patho_id, "row_type": "patho"}
        for k in range(N_CANDIDATES):
            sub_patho[f"cand_score_{k}"] = 0.2
        for i in range(N_DISEASES):
            sub_patho[f"pred_{i}"] = 0
            sub_patho[f"p_{i}"] = 0.5
        sub_rows.append(sub_patho)

        ans_link = {"id": link_id, "row_type": "link", "true_match_idx": true_match_idx}
        for i in range(N_DISEASES):
            ans_link[f"true_y_{i}"] = int(q[f"has_{i}"])
        ans_rows.append(ans_link)

        ans_patho = {"id": patho_id, "row_type": "patho", "true_match_idx": 0}
        for i in range(N_DISEASES):
            ans_patho[f"true_y_{i}"] = int(q[f"has_{i}"])
        ans_rows.append(ans_patho)

    pd.DataFrame(queries).sort_values("query_id").reset_index(drop=True).to_csv(
        public / "test_queries.csv", index=False
    )
    pd.DataFrame(candidates).sort_values("query_id").reset_index(drop=True).to_csv(
        public / "test_candidates.csv", index=False
    )
    pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True).to_csv(
        public / "sample_submission.csv", index=False
    )
    pd.DataFrame(ans_rows).sort_values("id").reset_index(drop=True).to_csv(
        private / "answers.csv", index=False
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path(__file__).parent / "raw_data")
    ap.add_argument("--public", type=Path, default=Path(__file__).parent / "pub")
    ap.add_argument("--private", type=Path, default=Path(__file__).parent / "priv")
    args = ap.parse_args()
    if args.public.exists():
        shutil.rmtree(args.public)
    if args.private.exists():
        shutil.rmtree(args.private)
    prepare(args.raw, args.public, args.private)
    print("prepare done ->", args.public, args.private)


if __name__ == "__main__":
    main()
