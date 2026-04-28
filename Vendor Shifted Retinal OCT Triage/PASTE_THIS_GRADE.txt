"""grade.py for the Vendor-Shifted Retinal OCT Triage challenge.

Composite score in [0, 1], higher better:

  Final = 0.25 * MacroF1_disease_inscope
        + 0.20 * MacroF1_disease_vendor_shifted
        + 0.15 * MacroF1_disease_corrupted
        + 0.15 * (1 - MAE_layer_position)        # clipped to [0,1]
        + 0.15 * AUROC_disease_macro_inscope    # one-vs-rest mean
        + 0.10 * max(0, 1 - 15 * ECE_disease_overall)

Submission is a single CSV keyed by `id`:
    id                       string from public/test.csv
    disease_pred             string in {CNV, DME, DRUSEN, NORMAL}
    layer_position_pred      float  in [0, 1]
    p_CNV, p_DME, p_DRUSEN, p_NORMAL   floats summing to ~1.0 (allowed slack ±1e-3)

Answers (private/answers.csv) carries:
    id, slice ∈ {inscope, vendor, corrupted},
    true_disease, true_layer_position
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SUB_COLS = {
    "id",
    "disease_pred",
    "layer_position_pred",
    "p_CNV", "p_DME", "p_DRUSEN", "p_NORMAL",
}
REQUIRED_ANS_COLS = {"id", "slice", "true_disease", "true_layer_position"}
DISEASE_LABELS = ("CNV", "DME", "DRUSEN", "NORMAL")
PROB_COLS = ("p_CNV", "p_DME", "p_DRUSEN", "p_NORMAL")
PROB_SUM_TOL = 1e-3
ECE_BINS = 15
ECE_PENALTY = 15.0


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> float:
    f1s = []
    for lab in labels:
        tp = int(((y_true == lab) & (y_pred == lab)).sum())
        fp = int(((y_true != lab) & (y_pred == lab)).sum())
        fn = int(((y_true == lab) & (y_pred != lab)).sum())
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


def _macro_auroc(y_true_str: np.ndarray, prob_matrix: np.ndarray, labels: tuple[str, ...]) -> float:
    aurocs = []
    for j, lab in enumerate(labels):
        y_bin = (y_true_str == lab).astype(int)
        a = _auroc_binary(y_bin, prob_matrix[:, j])
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
        if len(sub) != len(ans):
            return 0.0
        if set(sub["id"]) != set(ans["id"]):
            return 0.0

        merged = ans.merge(sub, on="id", how="left")
        if len(merged) != len(ans):
            return 0.0
        if merged[list(REQUIRED_SUB_COLS - {"id"})].isna().any().any():
            return 0.0

        merged["disease_pred"] = merged["disease_pred"].astype(str)
        if not merged["disease_pred"].isin(DISEASE_LABELS).all():
            return 0.0

        try:
            merged["layer_position_pred"] = pd.to_numeric(
                merged["layer_position_pred"], errors="coerce"
            ).astype(float)
            for c in PROB_COLS:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").astype(float)
        except Exception:
            return 0.0

        if not np.all(np.isfinite(merged["layer_position_pred"].to_numpy())):
            return 0.0
        for c in PROB_COLS:
            arr = merged[c].to_numpy()
            if not np.all(np.isfinite(arr)):
                return 0.0
            if (arr < 0.0).any() or (arr > 1.0).any():
                return 0.0

        prob_mat = merged[list(PROB_COLS)].to_numpy()
        prob_sum = prob_mat.sum(axis=1)
        if (np.abs(prob_sum - 1.0) > PROB_SUM_TOL).any():
            return 0.0

        layer_pred = merged["layer_position_pred"].to_numpy()
        if (layer_pred < 0.0).any() or (layer_pred > 1.0).any():
            return 0.0

        y_true_disease = merged["true_disease"].astype(str).to_numpy()
        y_pred_disease = merged["disease_pred"].astype(str).to_numpy()
        y_true_layer = merged["true_layer_position"].astype(float).to_numpy()
        slice_arr = merged["slice"].astype(str).to_numpy()

        m_in = slice_arr == "inscope"
        m_vd = slice_arr == "vendor"
        m_cr = slice_arr == "corrupted"

        f1_in = _macro_f1(y_true_disease[m_in], y_pred_disease[m_in], DISEASE_LABELS) if m_in.any() else 0.0
        f1_vd = _macro_f1(y_true_disease[m_vd], y_pred_disease[m_vd], DISEASE_LABELS) if m_vd.any() else 0.0
        f1_cr = _macro_f1(y_true_disease[m_cr], y_pred_disease[m_cr], DISEASE_LABELS) if m_cr.any() else 0.0

        mae_layer = float(np.mean(np.abs(y_true_layer - layer_pred)))
        layer_term = max(0.0, 1.0 - mae_layer)

        if m_in.any():
            auroc_in = _macro_auroc(y_true_disease[m_in], prob_mat[m_in], DISEASE_LABELS)
        else:
            auroc_in = 0.0

        # ECE on argmax-confidence vs disease correctness across all rows.
        max_p = prob_mat.max(axis=1)
        argmax_idx = prob_mat.argmax(axis=1)
        argmax_label = np.array([DISEASE_LABELS[i] for i in argmax_idx])
        correctness = (argmax_label == y_true_disease).astype(float)
        ece_overall = _ece(max_p, correctness, ECE_BINS)
        cal = max(0.0, 1.0 - ECE_PENALTY * ece_overall)

        final = (
            0.25 * f1_in
            + 0.20 * f1_vd
            + 0.15 * f1_cr
            + 0.15 * layer_term
            + 0.15 * auroc_in
            + 0.10 * cal
        )
        return float(max(0.0, min(1.0, final)))
    except Exception:
        return 0.0
