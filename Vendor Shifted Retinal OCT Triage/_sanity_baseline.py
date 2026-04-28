"""Organiser-only smoke test for OCT challenge."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from grade import grade


def _expect(score: float, lo: float, hi: float, tag: str) -> None:
    print(f"[sanity] {tag}: score = {score:.4f}")
    if not (lo <= score <= hi):
        raise AssertionError(f"{tag} score {score} not in [{lo}, {hi}]")


def main() -> None:
    here = Path(__file__).parent
    sub = pd.read_csv(here / "pub" / "sample_submission.csv")
    ans = pd.read_csv(here / "priv" / "answers.csv")

    score = grade(sub, ans)
    _expect(score, 0.05, 0.65, "constant baseline")

    bad = sub.copy().drop(columns=["disease_pred"])
    _expect(grade(bad, ans), 0.0, 0.0, "missing column")

    bad = sub.copy()
    bad.loc[0, "p_CNV"] = 0.99  # row no longer sums to 1
    _expect(grade(bad, ans), 0.0, 0.0, "probabilities not summing to 1")

    bad = sub.copy()
    bad.loc[0, "layer_position_pred"] = 1.5
    _expect(grade(bad, ans), 0.0, 0.0, "out-of-range layer_position")

    bad = sub.copy()
    bad.loc[0, "id"] = "oct_999999"
    _expect(grade(bad, ans), 0.0, 0.0, "id set mismatch")

    print("[sanity] all checks passed")


if __name__ == "__main__":
    main()
