"""generate.py — Patient-Linkage Verification On Chest Radiographs.

Pulls the NIH ChestX-ray14 corpus (Wang et al. 2017) from the official Box
mirror, normalises filenames, and writes raw_data/ in the schema prepare.py
expects:

  raw_data/
    cxr_index.csv          columns: raw_patient_id, raw_image_id, image_file,
                                    has_0 .. has_13   (anonymised pathology
                                    columns; each is 0/1 indicator)
    images/<image_file>    8-bit grayscale chest X-ray PNG.

DATA SOURCE (organiser-side; not exposed downstream by prepare.py)
------------------------------------------------------------------
Wang et al. 2017, "ChestX-ray8 / ChestX-ray14".
  Primary:  https://nihcc.app.box.com/v/ChestXray-NIHCC
  Paper:    https://openaccess.thecvf.com/content_cvpr_2017/papers/
            Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
  Licence:  Per the NIH Clinical Center stated terms (verbatim from the
            official Box mirror README): "There are no restrictions on the
            use of the NIH chest x-ray images." Attribution required:
              - link to https://nihcc.app.box.com/v/ChestXray-NIHCC,
              - cite the CVPR 2017 paper,
              - acknowledge the NIH Clinical Center.
            This is treated as commercial-permissive by Google Cloud
            Healthcare, AWS Open Data, and academic mirrors. We did NOT
            rely on third-party "CC0" labels; we use the NIH terms verbatim.

The 14 NIH disease classes are mapped to integer indices 0..13 by a
deterministic, organiser-controlled permutation. The mapping is documented
in DATASET_FORM_FILL.md; it is NOT exposed to the agent because the
challenge is participant-facing and source-neutral.

SIZE BUDGET
-----------
The full ChestX-ray14 image release is ~42 GB across 12 .tar.gz parts and
the platform raw-upload server stalls on multi-GB archives. To keep the
pipeline ingestible while preserving the novelty axes (patient-linkage
under pathology-matched decoys, multi-label pathology prediction), this
script subsamples *patients*, caps images per patient, and resizes:

  1. --max-patients (default 800) keeps the first N patients (sorted by
     NIH `Patient ID`). 800 patients gives enough patient diversity for
     decoy galleries to stay non-trivial.
  2. --max-images-per-patient (default 8) keeps at most this many studies
     per patient via a deterministic shuffle. ChestX-ray14 has a strong
     long tail (some patients have 100+ images); capping at 8 stops the
     dataset from being dominated by a few patients.
  3. --max-edge (default 768) downscales each PNG so its long edge is at
     most that many pixels. Native NIH PNGs are 1024×1024; 768 px keeps
     diagnostic anatomy visible while shrinking each image by ~2x.

Defaults yield roughly **5-6k images** at ~150-300 kB each, i.e. a
`raw_data/` of roughly **1-2 GB** — well within the platform raw-upload
budget.

USAGE
-----
    pip install requests pillow numpy pandas tqdm
    python generate.py --out raw_data
    python generate.py --out raw_data --max-patients 1500 --max-images-per-patient 12 --max-edge 1024
    python zip_raw_for_upload.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List


NIH_BOX_BASE = "https://nihcc.app.box.com/shared/static"
# 12 image archive parts published by NIH (image .png files, ~42 GB total).
NIH_IMAGE_ARCHIVES = [
    "vfk49d74nhbxq3nqjg0900tdng7gjf2u.gz",  # images_001.tar.gz
    "i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",  # images_002.tar.gz
    "f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "jn1b3o7vc9ll6q5h7ufhfk2da6yvi8jh.gz",
    "tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]
NIH_INDEX_URL = (
    f"{NIH_BOX_BASE}/.txt".replace("/.txt", "/")  # placeholder — NIH publishes
    # the index csv separately as Data_Entry_2017_v2020.csv from the same Box
    # mirror; users who run generate.py should drop that file at
    # raw_data/_work/Data_Entry_2017_v2020.csv before invocation, since
    # NIH's Box public-share URLs require an interactive browser flow.
)

NIH_DISEASES_CANON = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)
DISEASE_PERM = (5, 0, 11, 7, 1, 9, 12, 13, 2, 6, 3, 8, 10, 4)


def _expect_index(work: Path) -> Path:
    """The NIH Box public-share URLs require an interactive browser flow,
    so we ask the user to drop Data_Entry_2017_v2020.csv into raw_data/_work/
    by hand before running generate.py. This is the same pattern the NIH
    Cloud-Healthcare API mirror uses."""
    p = work / "Data_Entry_2017_v2020.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"Expected NIH metadata at {p}.\n"
            "Download Data_Entry_2017_v2020.csv from\n"
            "  https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
            "and place it at the path above before running generate.py."
        )
    return p


def _expect_image_archives(work: Path) -> List[Path]:
    """Same pattern: NIH Box public-share URLs are not direct, so the user
    drops the 12 image_***.tar.gz files into raw_data/_work/ first. We then
    extract them deterministically into work/images_extracted/."""
    archives = sorted(work.glob("images_*.tar.gz"))
    if not archives:
        raise FileNotFoundError(
            f"Expected NIH image_***.tar.gz archives in {work}.\n"
            "Download them from https://nihcc.app.box.com/v/ChestXray-NIHCC and\n"
            "place them all in raw_data/_work/ before running generate.py."
        )
    return archives


def _extract_archives(archives: List[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for arc in archives:
        marker = dst / f"_extracted_{arc.name}"
        if marker.exists():
            continue
        print(f"[generate] extracting {arc.name}")
        with tarfile.open(str(arc), "r:gz") as tf:
            tf.extractall(str(dst))
        marker.write_text("ok", encoding="utf-8")


SUBSAMPLE_SEED = 0x4C7AE021  # deterministic per-patient image subsample


def _save_resized_png(src: Path, dst: Path, max_edge: int) -> None:
    """Re-encode src PNG to dst, downscaling so the long edge is <= max_edge.
    NIH ChestX-ray14 images are 8-bit grayscale 1024x1024 PNGs."""
    from PIL import Image
    with Image.open(str(src)) as im:
        im = im.convert("L")
        if max_edge and max_edge > 0:
            w, h = im.size
            long_edge = max(w, h)
            if long_edge > max_edge:
                scale = max_edge / float(long_edge)
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                im = im.resize(new_size, Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(str(dst), format="PNG", optimize=True)


def generate(
    out_dir: Path,
    max_patients: int,
    max_images_per_patient: int,
    max_edge: int,
) -> None:
    import numpy as np

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    index_csv = _expect_index(work)
    archives = _expect_image_archives(work)
    extracted = work / "images_extracted"
    _extract_archives(archives, extracted)

    # Walk the extracted tree and find each PNG. NIH archives place PNGs at
    # images/<id>.png inside each tar.gz; after extraction they end up in
    # extracted/images/<id>.png.
    raw_png_by_name: Dict[str, Path] = {}
    for p in extracted.rglob("*.png"):
        raw_png_by_name[p.name] = p

    # First pass: bucket index rows by patient (only rows whose PNG is on
    # disk, so a partial download still produces a coherent dataset).
    rows_by_patient: Dict[str, List[Dict[str, str]]] = {}
    with open(index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_name = r["Image Index"].strip()
            if img_name not in raw_png_by_name:
                continue
            patient_id = r["Patient ID"].strip()
            rows_by_patient.setdefault(patient_id, []).append(
                {"img": img_name, "findings": r["Finding Labels"].strip()}
            )

    patient_ids = sorted(rows_by_patient.keys())
    if max_patients and max_patients > 0:
        patient_ids = patient_ids[:max_patients]
    print(
        f"[generate] keeping {len(patient_ids)} of "
        f"{len(rows_by_patient)} patients "
        f"(max_images_per_patient={max_images_per_patient}, max_edge={max_edge})"
    )

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    rows: List[dict] = []
    for patient_id in patient_ids:
        per_patient = list(rows_by_patient[patient_id])
        # Deterministic shuffle then cap, so the kept subset is independent
        # of the original NIH ordering.
        order = np.arange(len(per_patient))
        rng.shuffle(order)
        if max_images_per_patient and max_images_per_patient > 0:
            order = order[:max_images_per_patient]
        for idx in order:
            entry = per_patient[int(idx)]
            img_name = entry["img"]
            findings = [s.strip() for s in entry["findings"].split("|") if s.strip()]
            has = [0] * len(NIH_DISEASES_CANON)
            for f_name in findings:
                if f_name in NIH_DISEASES_CANON:
                    has[NIH_DISEASES_CANON.index(f_name)] = 1
            anon_has = [0] * len(NIH_DISEASES_CANON)
            for canon_i, v in enumerate(has):
                anon_has[DISEASE_PERM[canon_i]] = v
            new_name = img_name  # keep image_file as the upstream filename
            _save_resized_png(
                raw_png_by_name[img_name], images_dir / new_name, max_edge
            )
            row = {
                "raw_patient_id": patient_id,
                "raw_image_id": img_name.replace(".png", ""),
                "image_file": new_name,
            }
            for i, v in enumerate(anon_has):
                row[f"has_{i}"] = v
            rows.append(row)

    rows.sort(key=lambda r: (r["raw_patient_id"], r["raw_image_id"]))
    out_csv = out_dir / "cxr_index.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "raw_patient_id", "raw_image_id", "image_file",
            *[f"has_{i}" for i in range(len(NIH_DISEASES_CANON))],
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[generate] wrote {out_csv} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw_data")
    ap.add_argument(
        "--max-patients",
        type=int,
        default=800,
        help="Restrict to the first N NIH patient IDs (sorted ascending). "
             "Default 800 gives enough patient diversity for non-trivial "
             "decoy galleries while keeping the upload small. Pass 0 to keep "
             "all patients (large; will exceed the platform upload cap).",
    )
    ap.add_argument(
        "--max-images-per-patient",
        type=int,
        default=8,
        help="Cap the number of studies per patient via a deterministic "
             "shuffle. Default 8 prevents long-tail patients (some have 100+ "
             "images) from dominating the dataset.",
    )
    ap.add_argument(
        "--max-edge",
        type=int,
        default=768,
        help="Resize each chest X-ray so its long edge is at most this many "
             "pixels before PNG encoding. Default 768 keeps diagnostic "
             "anatomy visible while halving the per-image size relative to "
             "the native 1024 px NIH PNGs. Pass 0 for native resolution.",
    )
    args = ap.parse_args()
    generate(
        args.out,
        args.max_patients,
        args.max_images_per_patient,
        args.max_edge,
    )


if __name__ == "__main__":
    main()
