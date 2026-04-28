"""generate_from_local_dicoms.py — Bilateral Asymmetry-Anchored Mammography Triage.

Use this script if `generate.py` keeps failing on TCIA REST API calls (windows
error 10054 / connection reset). It does the **same job** as `generate.py`
(produce raw_data/cases.csv + raw_data/images/<...>.png) but **does not call
the TCIA API at all** — it expects you to have already downloaded the DICOM
images using the **TCIA NBIA Data Retriever** desktop tool (the "download"
button on the CBIS-DDSM collection page) into a local folder.

USAGE
-----
1. Install pip deps (one-time):

       pip install pydicom pillow numpy pandas requests tqdm

2. Get the **NBIA Data Retriever** desktop client and the CBIS-DDSM
   manifest:

   - Go to:  https://www.cancerimagingarchive.net/collection/cbis-ddsm/
   - Scroll to the "Data Access" section.
   - Click "Download" -> a small ".tcia" manifest file is downloaded.
   - Install **NBIA Data Retriever** for Windows from the same page (or:
     https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever ).
   - Open the .tcia manifest file in NBIA Data Retriever, accept the
     license, choose a destination folder (e.g.
     `D:/CBIS-DDSM-download`), and click "Start". The Retriever resumes
     interrupted downloads automatically and is much more reliable than
     the REST API.

   You can also use the "Browser-Based Image Download" CSV mirror on the
   collection page if you cannot run the desktop client; it produces the
   same DICOM tree.

3. After the download finishes, the destination folder will contain a
   tree like:

       D:/CBIS-DDSM-download/
         CBIS-DDSM/
           Mass-Training_P_00001_LEFT_CC/
             <study uid>/
               <series uid>/
                 000000.dcm
           Mass-Training_P_00001_LEFT_MLO/...
           Calc-Training_P_00001_LEFT_CC/...
           ...
         metadata.csv          (NBIA also drops this here)

   The exact wrapper folder ("CBIS-DDSM/") may be missing depending on
   Retriever settings — this script searches recursively for the
   `Mass-*_P_*/` and `Calc-*_P_*/` subfolders, so any nesting is fine.

4. Run:

       python generate_from_local_dicoms.py --src D:/CBIS-DDSM-download --out raw_data

   This:
     - downloads the 4 small CBIS-DDSM pathology CSVs from the TCIA wiki
       (~2 MB total; same as generate.py),
     - walks `--src` for DICOMs,
     - matches each DICOM to its pathology row by patient_id + breast + view,
     - converts to 8-bit PNG, resizes to --max-edge (default 768 px),
     - writes raw_data/cases.csv + raw_data/images/.

5. Then build the platform upload zip the same way as before:

       python zip_raw_for_upload.py --raw raw_data --out BilateralMammo_RAW_upload.zip

The output schema is **identical** to `generate.py`'s output, so all the
downstream pieces (`prepare.py`, `grade.py`, `_sanity_baseline.py`) work
unchanged.

Notes
-----
- This script is **resumable**. Re-running over the same `--out` folder
  skips PNGs that already exist on disk.
- DICOM files copied into a single CBIS-DDSM "series" folder usually
  contain a single full-resolution mammogram view; this script picks the
  largest pixel-array DICOM in the series (defensive against accidental
  scout / thumbnail series).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# We re-use the helpers in generate.py to keep the row schema 1:1.
from generate import (
    PATHOLOGY_CSV_URLS,  # noqa: F401  (kept for clarity)
    _fetch_pathology_csvs,
    _parse_pathology_rows,
    _normalise_view,
)


# CBIS-DDSM NBIA Data Retriever names each top-level folder like:
#   "Mass-Training_P_00001_LEFT_CC"
#   "Mass-Test_P_00009_RIGHT_MLO"
#   "Calc-Training_P_00077_LEFT_MLO"
#   "Calc-Test_P_00500_RIGHT_CC"
# We extract patient_id, breast, view, abnormality_type from this name.
PATIENT_FOLDER_RE = re.compile(
    r"^(?P<kind>Mass|Calc)-(?:Training|Test)_"
    r"(?P<pid>P_\d+)_"
    r"(?P<breast>LEFT|RIGHT)_"
    r"(?P<view>CC|MLO)"
    # CBIS-DDSM also has full-mammogram folders that don't have the
    # abnormality suffix; allow optional `_\d+` (lesion id) and ignore
    # cropped-ROI / mask folders ("..._1_<kind>") because we only want
    # the full mammogram for each (patient, breast, view).
    r"(?:_\d+)?"
    r"$",
    re.IGNORECASE,
)


def _find_largest_dicom(folder: Path) -> Optional[Path]:
    """Return the .dcm file with the largest pixel array in `folder` (recursive).

    CBIS-DDSM full-mammogram series usually have one DICOM, but some folders
    on disk also contain ROI mask DICOMs which are tiny. Picking the largest
    pixel array is a robust heuristic for the full mammogram view.
    """
    import pydicom

    candidates: List[Tuple[int, Path]] = []
    for p in folder.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            candidates.append((rows * cols, p))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_dicom_to_png(dicom_path: Path, png_path: Path, max_edge: int) -> None:
    """Decode a DICOM file, max-normalise to 8-bit grayscale, optionally resize,
    and save as PNG."""
    import numpy as np
    import pydicom
    from PIL import Image

    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
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


def _walk_patient_folders(src_root: Path) -> Dict[Tuple[str, str, str, str], Path]:
    """Walk `src_root` recursively and return a dict keyed by
    (patient_id, breast, view, kind) -> folder Path containing the DICOMs.

    Only "full mammogram" folders are returned; ROI mask folders (which have
    extra suffixes like `_1_<kind>` referring to a cropped ROI) are filtered
    out by `PATIENT_FOLDER_RE`.
    """
    out: Dict[Tuple[str, str, str, str], Path] = {}
    for p in src_root.rglob("*"):
        if not p.is_dir():
            continue
        m = PATIENT_FOLDER_RE.match(p.name)
        if not m:
            continue
        pid = m.group("pid")
        breast = m.group("breast").upper()
        view = m.group("view").upper()
        kind = m.group("kind").lower()
        kind_norm = "mass" if kind == "mass" else "calc"
        key = (pid, breast, view, kind_norm)
        # If a duplicate exists, prefer the one containing more DICOM files.
        if key in out:
            existing = sum(1 for _ in out[key].rglob("*.dcm"))
            new = sum(1 for _ in p.rglob("*.dcm"))
            if new <= existing:
                continue
        out[key] = p
    return out


def generate_from_local(
    src_root: Path,
    out_dir: Path,
    max_patients: int,
    max_edge: int,
) -> None:
    src_root = Path(src_root).resolve()
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    if not src_root.is_dir():
        print(f"ERROR: --src directory does not exist: {src_root}", file=sys.stderr)
        sys.exit(1)

    print(f"[generate-local] scanning {src_root}")
    folder_index = _walk_patient_folders(src_root)
    print(f"[generate-local] found {len(folder_index)} (pid, breast, view, kind) folders")
    if not folder_index:
        print(
            "ERROR: no CBIS-DDSM patient folders found under --src.\n"
            "Expected folder names like 'Mass-Training_P_00001_LEFT_CC'.",
            file=sys.stderr,
        )
        sys.exit(2)

    rows_by_name = _fetch_pathology_csvs(work)
    flat = _parse_pathology_rows(rows_by_name)
    by_patient: Dict[str, List[Dict[str, str]]] = {}
    for r in flat:
        by_patient.setdefault(r["patient_id"], []).append(r)
    patient_ids = sorted(by_patient.keys())
    if max_patients > 0:
        patient_ids = patient_ids[:max_patients]
    print(f"[generate-local] using {len(patient_ids)} patients (max_edge={max_edge}px)")

    cases_rows: List[Dict[str, str]] = []
    skipped_no_folder = 0
    skipped_no_dicom = 0
    for pid in patient_ids:
        views_for_patient = by_patient[pid]
        sides = {r["breast"] for r in views_for_patient if r["malignant"] == "1"}
        asymmetric = 1 if len(sides) == 1 else 0

        for r in views_for_patient:
            kind = "mass" if r["abnormality_type"] == "mass" else "calc"
            key = (pid, r["breast"], r["view"], kind)
            folder = folder_index.get(key)
            if folder is None:
                skipped_no_folder += 1
                continue
            png_name = f"{pid}__{_normalise_view(r['breast'], r['view'])}__{kind}.png"
            png_path = images_dir / png_name
            if not png_path.exists():
                dicom = _find_largest_dicom(folder)
                if dicom is None:
                    skipped_no_dicom += 1
                    continue
                try:
                    _extract_dicom_to_png(dicom, png_path, max_edge)
                except Exception as exc:
                    print(
                        f"[generate-local] WARN {pid} {r['breast']} {r['view']} "
                        f"{kind}: {exc}",
                        file=sys.stderr,
                    )
                    skipped_no_dicom += 1
                    continue

            cases_rows.append({
                "case_raw_id": pid,
                "view": _normalise_view(r["breast"], r["view"]),
                "lesion": kind,
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
        f"[generate-local] wrote {out_csv} ({len(cases_rows)} rows) "
        f"and {len(list(images_dir.glob('*.png')))} PNGs"
    )
    if skipped_no_folder or skipped_no_dicom:
        print(
            f"[generate-local] skipped: no folder match = {skipped_no_folder}, "
            f"no DICOM in folder = {skipped_no_dicom}",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Folder where you ran NBIA Data Retriever (it contains "
             "'Mass-Training_P_*' / 'Calc-Test_P_*' subfolders, possibly "
             "nested under 'CBIS-DDSM/').",
    )
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
             "Pass 0 to use all patients available locally.",
    )
    ap.add_argument(
        "--max-edge",
        type=int,
        default=768,
        help="Long-edge cap for PNG output. Default 768. Pass 0 for native "
             "DICOM resolution.",
    )
    args = ap.parse_args()
    generate_from_local(
        args.src,
        args.out,
        args.max_patients,
        args.max_edge,
    )


if __name__ == "__main__":
    main()
