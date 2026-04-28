"""Organiser-only: small structurally-identical raw_data/ for smoke tests."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


N_PATIENTS = 80
N_IMAGES_PER_PATIENT = 3
N_DISEASES = 14
SEED = 0xCAFE


def main() -> None:
    here = Path(__file__).parent
    raw = here / "raw_data"
    if raw.exists():
        shutil.rmtree(raw)
    (raw / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for p in range(N_PATIENTS):
        pid = f"P{p:05d}"
        # patient pathology pattern: each disease iid Bernoulli(0.15)
        pat = rng.binomial(1, 0.15, size=N_DISEASES).astype(int)
        for k in range(N_IMAGES_PER_PATIENT):
            raw_img = f"{pid}_{k:02d}"
            arr = rng.integers(40, 220, size=(64, 64), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(
                str(raw / "images" / f"{raw_img}.png"), format="PNG", optimize=True
            )
            row = {
                "raw_patient_id": pid,
                "raw_image_id": raw_img,
                "image_file": f"{raw_img}.png",
            }
            # Slight per-image perturbation of the patient pattern
            jitter = pat.copy()
            flip_n = int(rng.integers(0, 2))
            for _ in range(flip_n):
                j = int(rng.integers(0, N_DISEASES))
                jitter[j] = 1 - jitter[j]
            for i in range(N_DISEASES):
                row[f"has_{i}"] = int(jitter[i])
            rows.append(row)

    fieldnames = ["raw_patient_id", "raw_image_id", "image_file"] + [f"has_{i}" for i in range(N_DISEASES)]
    with open(raw / "cxr_index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[synth] wrote {len(rows)} rows -> {raw}")


if __name__ == "__main__":
    main()
