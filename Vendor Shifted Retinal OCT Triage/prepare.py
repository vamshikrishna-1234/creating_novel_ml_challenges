"""prepare.py for the Vendor-Shifted Retinal OCT Triage challenge.

Input  raw/  containing:
  - oct_images.csv      columns: raw_id, disease, layer_position, image_file
                        disease ∈ {CNV, DME, DRUSEN, NORMAL}
                        layer_position ∈ [0.0, 1.0]  (relative depth of the
                                                     dominant pathologic layer)
  - images/<image_file> 8-bit grayscale OCT B-scan PNG.

Output public/:
  - train.csv                   id, image_file, disease, layer_position
  - test.csv                    id, image_file (no labels)
  - sample_submission.csv       constant baseline (all NORMAL, layer 0.5,
                                uniform probabilities 0.25 each)
  - images/<id>.png             renamed test image (with optional perturbation)

Output private/:
  - answers.csv                 id, slice ∈ {inscope, vendor, corrupted},
                                true_disease, true_layer_position

Three test slices are produced from the held-out split:
  inscope    : 50 % of test images, no perturbation applied at prepare time.
  vendor     : 30 % of test images, photometric / mild geometric perturbation
               applied to the image on disk; label and id are unchanged.
  corrupted  : 20 % of test images, heavier mixed perturbation applied.

The exact perturbation type / amplitude / RNG state is internal to prepare.py
and is NEVER described in the participant-facing challenge description; the
solver only sees that disease performance can degrade across the three test
slices, and must build robustness in.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SPLIT_SEED        = 0x12345678
ID_PERM_SEED      = 0xDEADBEEF
SLICE_SEED        = 0xC0FFEE01
PERTURB_SEED      = 0xBEEFCAFE
TRAIN_FRACTION    = 0.70
SLICE_FRACTIONS   = {"inscope": 0.50, "vendor": 0.30, "corrupted": 0.20}
DISEASE_LABELS    = ("CNV", "DME", "DRUSEN", "NORMAL")


def _read_raw(raw: Path) -> pd.DataFrame:
    csv_path = raw / "oct_images.csv"
    img_dir = raw / "images"
    if not csv_path.exists() or not img_dir.exists():
        raise FileNotFoundError(f"raw/ missing oct_images.csv or images/ at {raw}")
    df = pd.read_csv(csv_path)
    expected = {"raw_id", "disease", "layer_position", "image_file"}
    if not expected.issubset(set(df.columns)):
        raise ValueError("raw/oct_images.csv missing required columns")
    df = df[df["disease"].isin(DISEASE_LABELS)].copy()
    df["layer_position"] = pd.to_numeric(df["layer_position"], errors="coerce")
    df = df.dropna(subset=["layer_position"])
    df = df[(df["layer_position"] >= 0.0) & (df["layer_position"] <= 1.0)].copy()
    return df.reset_index(drop=True)


def _apply_perturbation(src_png: Path, dst_png: Path, mode: str, rng: np.random.Generator) -> None:
    """mode ∈ {none, vendor, corrupted}. Internal; not documented externally."""
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    if mode == "none":
        shutil.copy2(str(src_png), str(dst_png))
        return
    img = Image.open(str(src_png)).convert("L")
    if mode == "vendor":
        # mild photometric only
        b = float(0.85 + rng.random() * 0.30)
        c = float(0.85 + rng.random() * 0.30)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        if rng.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    elif mode == "corrupted":
        b = float(0.65 + rng.random() * 0.55)
        c = float(0.55 + rng.random() * 0.65)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        if rng.random() < 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        # additive Gaussian noise
        arr = np.asarray(img).astype(np.float32)
        noise = rng.normal(0.0, 12.0, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    img.save(str(dst_png), format="PNG", optimize=True)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw); public = Path(public); private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (public / "images").mkdir(parents=True, exist_ok=True)

    df = _read_raw(raw)
    n_total = len(df)

    split_rng = np.random.default_rng(SPLIT_SEED)
    perm = np.arange(n_total)
    split_rng.shuffle(perm)
    n_train = int(round(n_total * TRAIN_FRACTION))
    train_pos = set(perm[:n_train].tolist())
    test_pos = [i for i in range(n_total) if i not in train_pos]

    id_rng = np.random.default_rng(ID_PERM_SEED)
    image_ids = np.arange(n_total)
    id_rng.shuffle(image_ids)
    pos_to_imgid = {i: int(image_ids[i]) for i in range(n_total)}

    slice_rng = np.random.default_rng(SLICE_SEED)
    n_test = len(test_pos)
    n_in = int(round(n_test * SLICE_FRACTIONS["inscope"]))
    n_vd = int(round(n_test * SLICE_FRACTIONS["vendor"]))
    n_cr = n_test - n_in - n_vd
    test_pos_arr = np.array(test_pos)
    slice_rng.shuffle(test_pos_arr)
    inscope_pos = set(test_pos_arr[:n_in].tolist())
    vendor_pos = set(test_pos_arr[n_in:n_in + n_vd].tolist())
    corrupted_pos = set(test_pos_arr[n_in + n_vd:].tolist())

    perturb_rng = np.random.default_rng(PERTURB_SEED)

    train_rows: List[dict] = []
    test_rows: List[dict] = []
    sub_rows: List[dict] = []
    ans_rows: List[dict] = []

    images_src = raw / "images"
    images_dst = public / "images"

    for pos, row in df.iterrows():
        image_id = pos_to_imgid[pos]
        new_name = f"oct_{int(image_id):06d}.png"
        src_png = images_src / str(row["image_file"])
        if not src_png.exists():
            raise FileNotFoundError(f"missing source image {src_png}")
        dst_png = images_dst / new_name
        sample_id = f"oct_{int(image_id):06d}"

        if pos in train_pos:
            shutil.copy2(str(src_png), str(dst_png))
            train_rows.append({
                "id": sample_id,
                "image_file": new_name,
                "disease": str(row["disease"]),
                "layer_position": float(row["layer_position"]),
            })
        else:
            if pos in inscope_pos:
                slice_name = "inscope"; mode = "none"
            elif pos in vendor_pos:
                slice_name = "vendor"; mode = "vendor"
            else:
                slice_name = "corrupted"; mode = "corrupted"
            _apply_perturbation(src_png, dst_png, mode, perturb_rng)

            test_rows.append({
                "id": sample_id,
                "image_file": new_name,
            })
            sub_rows.append({
                "id": sample_id,
                "disease_pred": "NORMAL",
                "layer_position_pred": 0.5,
                "p_CNV": 0.25, "p_DME": 0.25, "p_DRUSEN": 0.25, "p_NORMAL": 0.25,
            })
            ans_rows.append({
                "id": sample_id,
                "slice": slice_name,
                "true_disease": str(row["disease"]),
                "true_layer_position": float(row["layer_position"]),
            })

    pd.DataFrame(train_rows).sort_values("id").reset_index(drop=True).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).sort_values("id").reset_index(drop=True).to_csv(public / "test.csv", index=False)
    pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True).to_csv(public / "sample_submission.csv", index=False)
    pd.DataFrame(ans_rows).sort_values("id").reset_index(drop=True).to_csv(private / "answers.csv", index=False)


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
