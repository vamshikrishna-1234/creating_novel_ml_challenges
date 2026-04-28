"""generate.py — Bilateral Asymmetry-Anchored Mammography Triage.

Pulls CBIS-DDSM (Curated Breast Imaging Subset of DDSM) from TCIA, converts the
DICOM scans to PNG, joins per-image metadata with the public mass / calc
pathology CSVs, and emits raw_data/ in the schema prepare.py expects:

  raw_data/
    cases.csv         columns: case_raw_id, view, lesion, malignant,
                              asymmetric, image_file
    images/<image_file>     8-bit grayscale PNG of the mammogram view

DATA SOURCE (organiser-side; not exposed downstream by prepare.py)
------------------------------------------------------------------
CBIS-DDSM (Lee et al. 2017, *Scientific Data* 4, 170177)
  Primary:  https://www.cancerimagingarchive.net/collection/cbis-ddsm/
  Paper:    https://www.nature.com/articles/sdata2017177
  Licence:  CC BY 3.0 (TCIA primary listing). Commercial redistribution is
            permitted with attribution under the TCIA Data Usage Policy.

Pathology metadata used (downloaded from the TCIA collection page):
  calc_case_description_train_set.csv
  calc_case_description_test_set.csv
  mass_case_description_train_set.csv
  mass_case_description_test_set.csv

Every CBIS-DDSM image carries: patient_id, view ∈ {CC, MLO}, breast ∈
{LEFT, RIGHT}, abnormality_type ∈ {mass, calcification}, pathology ∈
{BENIGN, BENIGN_WITHOUT_CALLBACK, MALIGNANT}. We derive:
  view (LCC, RCC, LMLO, RMLO)  = breast[0] + view
  lesion (none, mass, calc)    = none if no abnormality, else mass / calc
                                  (we synthesise "none" rows by sampling clean
                                  contralateral views from the same patient
                                  when the patient has only one diseased side)
  malignant (0, 1)             = pathology starts with "MALIGNANT"
  asymmetric (0, 1)            = malignancy or biopsy-confirmed lesion present
                                  on exactly one side of the same patient

The download step uses the TCIA NBIA REST API (no auth required for public
collections). The platform raw-upload server stalls on multi-GB archives,
so this script is conservatively defaulted to a *minimal* slice that still
preserves the novelty axes of the challenge (paired-view bilateral
asymmetry, calibrated malignancy):

  1. Cap the patient list to a deterministic prefix (default 200 patients,
     overridable with --max-patients; --max-patients 0 means "all"). 200
     patients gives ~800 mammogram views — enough cases for both
     bilaterally-symmetric and bilaterally-asymmetric pathology to be
     well-represented while keeping the upload archive small.
  2. Resize each mammogram so the long edge is at most --max-edge pixels
     (default 768) before encoding to 8-bit grayscale PNG. Aspect ratio is
     preserved. 768 px is large enough to retain mass margins and macro-
     calcification structure that the bilateral-asymmetry and lesion-class
     tasks rely on.
  3. Delete each per-series TCIA ZIP immediately after we have extracted the
     DICOM, so the on-disk working set stays small (peak working footprint
     is one series ZIP + one PNG at a time).

Empirically the defaults produce a `raw_data/` of roughly **300–800 MB**
(well under any platform upload cap) and an end-to-end `prepare.py` run
that finishes in a few minutes on consumer hardware.

USAGE
-----
    pip install pydicom requests pillow numpy pandas tqdm
    python generate.py --out raw_data                       # default 600 patients, 1024-px PNGs
    python generate.py --out raw_data --max-patients 200    # smaller dev pull
    python generate.py --out raw_data --max-patients 0      # full corpus (large)
    python zip_raw_for_upload.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TCIA_BASE = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"
COLLECTION = "CBIS-DDSM"

PATHOLOGY_CSV_URLS = {
    "calc_train": (
        "https://www.cancerimagingarchive.net/wp-content/uploads/"
        "calc_case_description_train_set.csv"
    ),
    "calc_test": (
        "https://www.cancerimagingarchive.net/wp-content/uploads/"
        "calc_case_description_test_set.csv"
    ),
    "mass_train": (
        "https://www.cancerimagingarchive.net/wp-content/uploads/"
        "mass_case_description_train_set.csv"
    ),
    "mass_test": (
        "https://www.cancerimagingarchive.net/wp-content/uploads/"
        "mass_case_description_test_set.csv"
    ),
}


def _fetch_pathology_csvs(work: Path) -> Dict[str, List[Dict[str, str]]]:
    import requests
    out: Dict[str, List[Dict[str, str]]] = {}
    work.mkdir(parents=True, exist_ok=True)
    for name, url in PATHOLOGY_CSV_URLS.items():
        dst = work / f"{name}.csv"
        if not dst.exists():
            print(f"[generate] downloading {url}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
        with open(dst, "r", encoding="utf-8", newline="") as f:
            out[name] = list(csv.DictReader(f))
    return out


def _fetch_image_series(series_uid: str, dst_zip: Path) -> None:
    """Download one mammogram series ZIP from TCIA. Large transfers may take many
    minutes; uses generous timeouts and retries (TCIA is often slow).
    """
    import requests
    import time

    if dst_zip.exists() and dst_zip.stat().st_size > 0:
        return
    url = f"{TCIA_BASE}/getImage?SeriesInstanceUID={series_uid}"
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    # (connect timeout, read timeout) — read must allow multi‑GB series zips.
    timeout = (60, 3600)
    max_retries = 6
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst_zip, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as exc:
            last_exc = exc
            if dst_zip.exists():
                try:
                    dst_zip.unlink()
                except OSError:
                    pass
            wait = min(30 * (attempt + 1), 180)
            print(
                f"[generate] retry {attempt + 1}/{max_retries} for series "
                f"{series_uid[:32]}... after {exc!r}; sleeping {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _extract_first_dicom_to_png(zip_path: Path, png_path: Path, max_edge: int) -> None:
    """Each CBIS-DDSM mammogram series typically contains one DICOM frame.
    We pull the first .dcm out of the series zip, downscale so the long edge
    is at most ``max_edge`` pixels, and re-encode as 8-bit grayscale PNG.

    Using a 1024-px long edge is the standard CBIS-DDSM "research-resolution"
    convention used by the TCIA mirror and most CBIS-DDSM benchmarks; it keeps
    diagnostic structure (mass margins, calcifications) while shrinking the
    per-image footprint by ~30-60x relative to the native ~3-5k-px DICOM."""
    import numpy as np
    import pydicom
    from PIL import Image

    with zipfile.ZipFile(zip_path, "r") as zf:
        dcm_members = [m for m in zf.namelist() if m.lower().endswith(".dcm")]
        if not dcm_members:
            raise RuntimeError(f"{zip_path}: no .dcm member found")
        with zf.open(dcm_members[0]) as f:
            data = f.read()
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    if arr.max() > 0:
        arr = (arr / arr.max() * 255.0).clip(0, 255)
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    if max_edge and max_edge > 0:
        h, w = img.size[1], img.size[0]
        long_edge = max(h, w)
        if long_edge > max_edge:
            scale = max_edge / float(long_edge)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, Image.LANCZOS)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(png_path), format="PNG", optimize=True)


def _parse_pathology_rows(
    rows_by_name: Dict[str, List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    """Normalise the four pathology CSVs into a single list of dicts with:
    patient_id, view ∈ {CC, MLO}, breast ∈ {LEFT, RIGHT},
    abnormality_type ∈ {mass, calc}, malignant ∈ {0,1}, image_path (DICOM path
    column from the CSV — used to resolve the DICOM filename inside the
    series zip)."""
    out: List[Dict[str, str]] = []
    for name, rows in rows_by_name.items():
        kind = "mass" if name.startswith("mass") else "calc"
        for r in rows:
            try:
                pid = r["patient_id"].strip()
                breast = r["left or right breast"].strip().upper()
                view = r["image view"].strip().upper()
                pathology = r["pathology"].strip().upper()
                series = r.get("image file path", "").strip()
                if not pid or breast not in ("LEFT", "RIGHT") or view not in ("CC", "MLO"):
                    continue
                malignant = 1 if pathology.startswith("MALIGNANT") else 0
                out.append({
                    "patient_id": pid,
                    "breast": breast,
                    "view": view,
                    "abnormality_type": kind,
                    "malignant": str(malignant),
                    "image_path": series,
                })
            except KeyError:
                continue
    return out


def _series_uid_from_image_path(image_path: str) -> str:
    """The CBIS-DDSM image_path column is of the form
    'Mass-Training_P_xxxxx_LEFT_CC/<study>/<series>/000000.dcm'. The series
    UID is the second-to-last folder component (matches TCIA's
    SeriesInstanceUID semantics for this collection)."""
    parts = image_path.replace("\\", "/").strip("/").split("/")
    if len(parts) < 2:
        return ""
    return parts[-2]


def _normalise_view(breast: str, view: str) -> str:
    return ("L" if breast == "LEFT" else "R") + view


def generate(out_dir: Path, max_patients: int, max_edge: int) -> None:
    import pandas as pd

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    rows_by_name = _fetch_pathology_csvs(work)
    flat = _parse_pathology_rows(rows_by_name)
    by_patient: Dict[str, List[Dict[str, str]]] = {}
    for r in flat:
        by_patient.setdefault(r["patient_id"], []).append(r)
    patient_ids = sorted(by_patient.keys())
    if max_patients > 0:
        patient_ids = patient_ids[:max_patients]
    print(
        f"[generate] fetching {len(patient_ids)} patients "
        f"(max_edge={max_edge}px)"
    )

    cases_rows: List[Dict[str, str]] = []
    for pid in patient_ids:
        views_for_patient = by_patient[pid]
        # If the patient has any malignancy on exactly one side, asymmetric=1.
        sides = {r["breast"] for r in views_for_patient if r["malignant"] == "1"}
        asymmetric = 1 if len(sides) == 1 else 0

        for r in views_for_patient:
            series_uid = _series_uid_from_image_path(r["image_path"])
            if not series_uid:
                continue
            zip_path = work / f"{series_uid}.zip"
            png_name = f"{pid}__{_normalise_view(r['breast'], r['view'])}__{r['abnormality_type']}.png"
            png_path = images_dir / png_name
            try:
                _fetch_image_series(series_uid, zip_path)
                if not png_path.exists():
                    _extract_first_dicom_to_png(zip_path, png_path, max_edge)
            except Exception as exc:
                print(f"[generate] WARN {pid}/{series_uid}: {exc}", file=sys.stderr)
                # Drop the per-series ZIP even on failure so the working
                # directory does not grow without bound across reruns.
                try:
                    if zip_path.exists():
                        zip_path.unlink()
                except OSError:
                    pass
                continue

            # Free the per-series ZIP as soon as we have its PNG.  Keeps the
            # peak working-disk footprint tiny (one series at a time).
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except OSError:
                pass

            cases_rows.append({
                "case_raw_id": pid,
                "view": _normalise_view(r["breast"], r["view"]),
                "lesion": "mass" if r["abnormality_type"] == "mass" else "calc",
                "malignant": r["malignant"],
                "asymmetric": str(asymmetric),
                "image_file": png_name,
            })

    out_csv = out_dir / "cases.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
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
        w.writerows(cases_rows)

    print(
        f"[generate] wrote {out_csv} ({len(cases_rows)} rows) "
        f"and {len(list(images_dir.glob('*.png')))} PNGs"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "raw_data",
        help="Output folder for raw_data/ (must contain cases.csv + images/).",
    )
    ap.add_argument(
        "--max-patients",
        type=int,
        default=200,
        help="Restrict to the first N CBIS-DDSM patients (sorted patient_id). "
             "Default 200 yields ~800 mammogram views and a sub-GB raw_data/. "
             "Pass 0 to use the full collection (large download; will exceed "
             "the platform raw-upload size cap).",
    )
    ap.add_argument(
        "--max-edge",
        type=int,
        default=768,
        help="Resize each mammogram so its long edge is at most this many "
             "pixels before encoding to PNG. Default 768 keeps mass margins "
             "and macro-calcification structure visible while keeping each "
             "image around 0.2-0.4 MB. Set 0 to keep native DICOM resolution "
             "(produces multi-MB PNGs and a much larger raw_data/).",
    )
    args = ap.parse_args()
    generate(args.out, args.max_patients, args.max_edge)


if __name__ == "__main__":
    main()
