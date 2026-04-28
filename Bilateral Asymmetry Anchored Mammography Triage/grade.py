"""grade.py for the Bilateral Asymmetry-Anchored Mammography Triage challenge.

Composite score in [0, 1], higher better:

  Final = 0.30 * MacroF1_lesion
        + 0.25 * F1_asymmetry
        + 0.25 * AUROC_malignancy
        + 0.10 * max(0, 1 - 15 * ECE_malignancy)
        + 0.10 * MacroF1_lesion_dropped_views

Where the submission has TWO row types keyed by `id`:
  - view rows  (one per (case, view) actually present in test.csv after view-drop):
      * lesion_pred ∈ {none, mass, calc}
  - case rows  (exactly one per case_id in test.csv):
      * asymmetry_pred ∈ {0, 1}
      * malignancy_prob ∈ [0, 1]

Each row's `id` is unique. Column `row_type` ∈ {view, case} disambiguates which
prediction columns are required for that row.

Any structural failure forces a return value of 0.0 (uncaught exception caught,
missing columns, NaN, duplicate ids, set mismatch, out-of-range values).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SUB_COLS = {"id", "row_type", "lesion_pred", "asymmetry_pred", "malignancy_prob"}
REQUIRED_ANS_COLS = {
    "id",
    "row_type",
    "true_lesion",
    "true_asymmetry",
    "true_malignancy",
    "is_dropped_view",
}
LESION_LABELS = ("none", "mass", "calc")
ECE_BINS = 15
ECE_PENALTY = 15.0


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())
    if tp + fp == 0 and tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _macro_f1_strings(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> float:
    f1s = []
    for lab in labels:
        tp = int(((y_true == lab) & (y_pred == lab)).sum())
        fp = int(((y_true != lab) & (y_pred == lab)).sum())
        fn = int(((y_true == lab) & (y_pred != lab)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * prec * rec / (prec + rec))
    if not f1s:
        return 0.0
    return float(sum(f1s) / len(f1s))


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return 0.0
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return 0.0
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    n = len(y_score)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(max(0.0, min(1.0, auc)))


def _ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int) -> float:
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        acc = float(correctness[mask].mean())
        conf = float(confidences[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Score submission vs answers (private/answers.csv). Float in [0,1]."""
    try:
        if not REQUIRED_SUB_COLS.issubset(set(submission.columns)):
            return 0.0
        if not REQUIRED_ANS_COLS.issubset(set(answers.columns)):
            return 0.0

        sub = submission[list(REQUIRED_SUB_COLS)].copy()
        ans = answers[list(REQUIRED_ANS_COLS)].copy()

        if sub["id"].duplicated().any():
            return 0.0
        if ans["id"].duplicated().any():
            return 0.0
        if len(sub) != len(ans):
            return 0.0
        if set(sub["id"]) != set(ans["id"]):
            return 0.0

        merged = ans.merge(sub, on="id", how="left", suffixes=("_a", "_s"))
        if len(merged) != len(ans):
            return 0.0
        if merged["row_type_s"].isna().any():
            return 0.0
        if (merged["row_type_s"].astype(str) != merged["row_type_a"].astype(str)).any():
            return 0.0
        merged["row_type"] = merged["row_type_a"]

        view_mask = merged["row_type"].astype(str) == "view"
        case_mask = merged["row_type"].astype(str) == "case"

        # ---- View rows: lesion_pred must be one of {none, mass, calc} when row_type==view
        view_rows = merged.loc[view_mask].copy()
        if view_rows["lesion_pred"].isna().any():
            return 0.0
        view_rows["lesion_pred"] = view_rows["lesion_pred"].astype(str)
        bad_lesion = ~view_rows["lesion_pred"].isin(LESION_LABELS)
        if bool(bad_lesion.any()):
            return 0.0

        # is_dropped_view marks views that exist as ANSWER rows but the public
        # test.csv tells the solver that view is dropped. The solver must still
        # emit a lesion_pred for each test row, but the dropped subset is graded
        # in MacroF1_lesion_dropped_views (robustness sub-term).
        y_true_lesion = view_rows["true_lesion"].astype(str).to_numpy()
        y_pred_lesion = view_rows["lesion_pred"].astype(str).to_numpy()
        is_dropped = view_rows["is_dropped_view"].astype(int).to_numpy() == 1

        macro_f1_lesion = _macro_f1_strings(y_true_lesion, y_pred_lesion, LESION_LABELS)
        if is_dropped.any():
            macro_f1_dropped = _macro_f1_strings(
                y_true_lesion[is_dropped], y_pred_lesion[is_dropped], LESION_LABELS
            )
        else:
            macro_f1_dropped = 0.0

        # ---- Case rows: asymmetry_pred ∈ {0,1}, malignancy_prob ∈ [0,1]
        case_rows = merged.loc[case_mask].copy()
        if case_rows["asymmetry_pred"].isna().any():
            return 0.0
        if case_rows["malignancy_prob"].isna().any():
            return 0.0

        try:
            asym_pred = pd.to_numeric(case_rows["asymmetry_pred"], errors="coerce").astype(float).to_numpy()
            mal_prob = pd.to_numeric(case_rows["malignancy_prob"], errors="coerce").astype(float).to_numpy()
        except Exception:
            return 0.0

        if not np.all(np.isfinite(asym_pred)):
            return 0.0
        if not np.all(np.isfinite(mal_prob)):
            return 0.0
        # asymmetry_pred is a discrete decision; clip and round
        asym_pred_int = np.where(asym_pred >= 0.5, 1, 0).astype(int)
        if (mal_prob < 0.0).any() or (mal_prob > 1.0).any():
            return 0.0

        y_true_asym = case_rows["true_asymmetry"].astype(int).to_numpy()
        y_true_mal = case_rows["true_malignancy"].astype(int).to_numpy()

        f1_asym = _binary_f1(y_true_asym, asym_pred_int, positive=1)
        auroc_mal = _auroc(y_true_mal, mal_prob)

        # ECE on malignancy probability (correctness = mal_prob >= 0.5 matches y_true_mal)
        argmax_mal_correct = ((mal_prob >= 0.5).astype(int) == y_true_mal).astype(float)
        ece_mal = _ece(mal_prob, argmax_mal_correct, ECE_BINS)
        calibration_term = max(0.0, 1.0 - ECE_PENALTY * ece_mal)

        final = (
            0.30 * macro_f1_lesion
            + 0.25 * f1_asym
            + 0.25 * auroc_mal
            + 0.10 * calibration_term
            + 0.10 * macro_f1_dropped
        )
        return float(max(0.0, min(1.0, final)))

    except Exception:
        return 0.0
