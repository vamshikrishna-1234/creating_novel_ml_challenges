"""Organiser-only: small structurally-identical raw_data/ for smoke tests."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


N_IMAGES = 240  # 60 per class
SEED = 0xCAFE


def main() -> None:
    here = Path(__file__).parent
    raw = here / "raw_data"
    if raw.exists():
        shutil.rmtree(raw)
    (raw / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    classes = ("CNV", "DME", "DRUSEN", "NORMAL")
    rows = []
    for i in range(N_IMAGES):
        disease = classes[i % len(classes)]
        # synth a 64x64 image with a class-specific bright band depth
        arr = rng.integers(40, 90, size=(64, 64), dtype=np.uint8)
        band_row = {"CNV": 12, "DME": 28, "DRUSEN": 44, "NORMAL": 36}[disease]
        band_row = int(np.clip(band_row + rng.integers(-3, 4), 0, 63))
        arr[band_row:band_row + 3, :] = np.uint8(220 + rng.integers(-15, 16))
        layer_pos = float(band_row / 63.0)
        raw_id = f"{i:06d}"
        name = f"{disease}_{raw_id}.png"
        Image.fromarray(arr, mode="L").save(str(raw / "images" / name), format="PNG", optimize=True)
        rows.append({"raw_id": raw_id, "disease": disease,
                     "layer_position": f"{layer_pos:.6f}", "image_file": name})
    with open(raw / "oct_images.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_id", "disease", "layer_position", "image_file"])
        w.writeheader()
        w.writerows(rows)
    print(f"[synth] wrote {len(rows)} rows -> {raw}")


if __name__ == "__main__":
    main()
