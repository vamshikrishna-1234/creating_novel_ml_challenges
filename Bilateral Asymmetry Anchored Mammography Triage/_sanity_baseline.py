"""Organiser-only smoke test:
  1) load pub/sample_submission.csv  +  priv/answers.csv
  2) score the constant baseline -> non-zero, well below 0.80 ceiling
  3) score a deliberately broken submission (wrong column / extra row) -> 0.0
"""
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

    bad = sub.copy()
    bad = bad.drop(columns=["lesion_pred"])
    _expect(grade(bad, ans), 0.0, 0.0, "missing column")

    bad = sub.copy()
    bad.loc[0, "id"] = "view_999999_LCC"  # id not in answers
    _expect(grade(bad, ans), 0.0, 0.0, "id set mismatch")

    bad = sub.copy()
    bad.loc[0, "malignancy_prob"] = 1.5  # out of range
    _expect(grade(bad, ans), 0.0, 0.0, "out-of-range probability")

    print("[sanity] all checks passed")


if __name__ == "__main__":
    main()
