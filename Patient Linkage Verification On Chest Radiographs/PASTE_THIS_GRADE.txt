"""grade.py for the Patient-Linkage Verification On Chest Radiographs challenge.

Composite score in [0, 1], higher better:

  Final = 0.30 * MRR_patient_match
        + 0.20 * Hits@1_patient_match
        + 0.25 * MacroF1_pathology_query
        + 0.15 * MacroAUROC_pathology_query
        + 0.10 * max(0, 1 - 15 * ECE_pathology_argmax)

Submission has TWO row types keyed by `id`:
  - link rows  (one per query):
      id = link_<query_id>
      row_type = "link"
      cand_score_0 .. cand_score_4 : floats; higher = more likely same-patient.
        Order corresponds to public/test_candidates.csv (cand_id_0..cand_id_4).
  - patho rows  (one per query):
      id = patho_<query_id>
      row_type = "patho"
      14 binary columns:  pred_<class>     ∈ {0, 1}     for classes 0..13
      14 prob columns:    p_<class>        ∈ [0, 1]    for classes 0..13
        (these are independent per-class probabilities, NOT softmax;
         no sum-to-1 constraint.)

Ground truth (private/answers.csv):
  id, row_type, true_match_idx (link rows; int in [0,4]),
  true_y_<class> (patho rows; int 0/1, 14 columns).

Disease classes are anonymised as integer indices 0..13; the mapping from
upstream NIH labels is held privately (in prepare.py). The grader only
needs the 14 anonymised columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


N_CANDIDATES = 5
N_DISEASES = 14
DISEASE_IDS = tuple(range(N_DISEASES))
PROB_COLS = tuple(f"p_{i}" for i in DISEASE_IDS)
PRED_COLS = tuple(f"pred_{i}" for i in DISEASE_IDS)
TRUE_COLS = tuple(f"true_y_{i}" for i in DISEASE_IDS)
CAND_COLS = tuple(f"cand_score_{i}" for i in range(N_CANDIDATES))

REQUIRED_SUB_COLS = {"id", "row_type", *CAND_COLS, *PRED_COLS, *PROB_COLS}
REQUIRED_ANS_COLS = {"id", "row_type", "true_match_idx", *TRUE_COLS}

ECE_BINS = 15
ECE_PENALTY = 15.0


def _macro_f1_binary(y_true_mat: np.ndarray, y_pred_mat: np.ndarray) -> float:
    f1s = []
    for j in range(y_true_mat.shape[1]):
        yt, yp = y_true_mat[:, j], y_pred_mat[:, j]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    n = len(y_score)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return float(max(0.0, min(1.0, (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg))))


def _macro_auroc(y_true_mat: np.ndarray, y_score_mat: np.ndarray) -> float:
    aurocs = []
    for j in range(y_true_mat.shape[1]):
        a = _auroc_binary(y_true_mat[:, j], y_score_mat[:, j])
        if not np.isnan(a):
            aurocs.append(a)
    return float(sum(aurocs) / len(aurocs)) if aurocs else 0.0


def _ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int) -> float:
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences <= hi) if b == n_bins - 1 else (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(correctness[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    try:
        if not REQUIRED_SUB_COLS.issubset(set(submission.columns)):
            return 0.0
        if not REQUIRED_ANS_COLS.issubset(set(answers.columns)):
            return 0.0

        sub = submission[list(REQUIRED_SUB_COLS)].copy()
        ans = answers[list(REQUIRED_ANS_COLS)].copy()

        if sub["id"].duplicated().any() or ans["id"].duplicated().any():
            return 0.0
        if len(sub) != len(ans) or set(sub["id"]) != set(ans["id"]):
            return 0.0

        merged = ans.merge(sub, on="id", how="left", suffixes=("_a", "_s"))
        if len(merged) != len(ans):
            return 0.0
        if merged["row_type_s"].isna().any():
            return 0.0
        if (merged["row_type_s"].astype(str) != merged["row_type_a"].astype(str)).any():
            return 0.0
        merged["row_type"] = merged["row_type_a"]

        link_mask = merged["row_type"].astype(str) == "link"
        patho_mask = merged["row_type"].astype(str) == "patho"

        # ---- LINK rows
        link_rows = merged.loc[link_mask].copy()
        for c in CAND_COLS:
            try:
                link_rows[c] = pd.to_numeric(link_rows[c], errors="coerce").astype(float)
            except Exception:
                return 0.0
        cand_mat = link_rows[list(CAND_COLS)].to_numpy()
        if not np.all(np.isfinite(cand_mat)):
            return 0.0
        true_idx = link_rows["true_match_idx"].astype(int).to_numpy()
        if (true_idx < 0).any() or (true_idx >= N_CANDIDATES).any():
            return 0.0

        # rank: descending order of cand scores; ties broken by index ascending
        # ranks[i] = position of true_idx[i] in the sorted descending order, 1-indexed
        ranks = np.empty(len(true_idx), dtype=float)
        for r, scores in enumerate(cand_mat):
            order = np.argsort(-scores, kind="stable")
            pos_of_true = int(np.where(order == true_idx[r])[0][0])
            ranks[r] = pos_of_true + 1
        mrr = float((1.0 / ranks).mean()) if len(ranks) else 0.0
        hits1 = float((ranks == 1).mean()) if len(ranks) else 0.0

        # ---- PATHO rows
        patho_rows = merged.loc[patho_mask].copy()
        for c in PRED_COLS + PROB_COLS:
            try:
                patho_rows[c] = pd.to_numeric(patho_rows[c], errors="coerce")
            except Exception:
                return 0.0
        if patho_rows[list(PRED_COLS) + list(PROB_COLS)].isna().any().any():
            return 0.0

        pred_mat = patho_rows[list(PRED_COLS)].astype(float).to_numpy()
        prob_mat = patho_rows[list(PROB_COLS)].astype(float).to_numpy()
        true_mat = patho_rows[list(TRUE_COLS)].astype(int).to_numpy()

        if not np.all(np.isin(pred_mat.astype(int), (0, 1))):
            return 0.0
        if (np.abs(pred_mat - pred_mat.astype(int)) > 1e-9).any():
            return 0.0
        pred_mat = pred_mat.astype(int)
        if not np.all(np.isfinite(prob_mat)):
            return 0.0
        if (prob_mat < 0.0).any() or (prob_mat > 1.0).any():
            return 0.0

        macro_f1 = _macro_f1_binary(true_mat, pred_mat)
        macro_auroc = _macro_auroc(true_mat, prob_mat)

        # ECE on argmax-style: per-row, take the maximum probability, treat
        # correctness as "the per-class binary prediction at that argmax matches
        # the per-class true label". This pools across queries and classes.
        argmax_idx = prob_mat.argmax(axis=1)
        max_p = prob_mat.max(axis=1)
        argmax_correct = np.zeros(len(argmax_idx), dtype=float)
        for i, j in enumerate(argmax_idx):
            argmax_correct[i] = float(pred_mat[i, j] == true_mat[i, j])
        ece = _ece(max_p, argmax_correct, ECE_BINS)
        cal = max(0.0, 1.0 - ECE_PENALTY * ece)

        final = (
            0.30 * mrr
            + 0.20 * hits1
            + 0.25 * macro_f1
            + 0.15 * macro_auroc
            + 0.10 * cal
        )
        return float(max(0.0, min(1.0, final)))
    except Exception:
        return 0.0
