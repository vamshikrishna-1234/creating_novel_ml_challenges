"""Organiser-only: build a tiny structurally-identical raw_data/ for smoke tests.

This is NOT shipped to participants. It writes a fake raw_data/ that has the
exact same column schema and folder layout `prepare.py` expects, so we can
verify the full pipeline (prepare.py -> sample_submission -> grade.py) without
downloading the 163 GB CBIS-DDSM source.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


N_CASES = 60
SEED = 0xCAFE


def main() -> None:
    here = Path(__file__).parent
    raw = here / "raw_data"
    if raw.exists():
        shutil.rmtree(raw)
    (raw / "images").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    rows = []
    for ci in range(N_CASES):
        pid = f"P_{ci:05d}"
        # Decide malignancy side (asymmetric pattern):
        mal_left = bool(rng.integers(0, 2))
        mal_right = (not mal_left) if rng.random() < 0.7 else bool(rng.integers(0, 2))
        asym = 1 if (mal_left ^ mal_right) else 0

        for breast, view in (
            ("LEFT", "CC"),
            ("LEFT", "MLO"),
            ("RIGHT", "CC"),
            ("RIGHT", "MLO"),
        ):
            view_code = ("L" if breast == "LEFT" else "R") + view
            mal_side = mal_left if breast == "LEFT" else mal_right
            # 60% none, 25% mass, 15% calc within malignant sides; healthy
            # contralateral side has 80% none.
            if mal_side:
                kinds = ("none", "mass", "calc")
                p = (0.10, 0.55, 0.35)
            else:
                kinds = ("none", "mass", "calc")
                p = (0.80, 0.12, 0.08)
            lesion = str(rng.choice(kinds, p=p))
            malignant = int(mal_side and lesion != "none")

            png_name = f"{pid}__{view_code}__{lesion}.png"
            arr = (rng.integers(0, 255, size=(64, 64), dtype=np.uint8))
            Image.fromarray(arr, mode="L").save(
                str(raw / "images" / png_name), format="PNG", optimize=True
            )

            rows.append({
                "case_raw_id": pid,
                "view": view_code,
                "lesion": lesion,
                "malignant": str(malignant),
                "asymmetric": str(asym),
                "image_file": png_name,
            })

    with open(raw / "cases.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_raw_id",
                "view",
                "lesion",
                "malignant",
                "asymmetric",
                "image_file",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[synth] wrote {len(rows)} rows -> {raw}")


if __name__ == "__main__":
    main()
