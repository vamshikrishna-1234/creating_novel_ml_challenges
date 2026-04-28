"""Organiser-only smoke test for Patient-Linkage challenge."""
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

    bad = sub.copy().drop(columns=["cand_score_0"])
    _expect(grade(bad, ans), 0.0, 0.0, "missing column")

    bad = sub.copy()
    bad.loc[0, "id"] = "link_does_not_exist"
    _expect(grade(bad, ans), 0.0, 0.0, "id set mismatch")

    bad = sub.copy()
    patho_idx = bad.index[bad["row_type"] == "patho"][0]
    bad.loc[patho_idx, "p_0"] = 1.5
    _expect(grade(bad, ans), 0.0, 0.0, "out-of-range prob (patho row)")

    bad = sub.copy()
    patho_idx = bad.index[bad["row_type"] == "patho"][0]
    bad.loc[patho_idx, "pred_0"] = 2  # not 0/1
    _expect(grade(bad, ans), 0.0, 0.0, "non-binary pred (patho row)")

    print("[sanity] all checks passed")


if __name__ == "__main__":
    main()
