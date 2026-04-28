"""Build BilateralMammo_RAW_upload.zip for the platform "Data Files" uploader.

The zip root must contain ONLY:
  cases.csv
  images/<...>.png

No extra parent folder, no .py, no README. prepare.py consumes these on the
platform side and emits public/ + private/.

Do NOT zip the whole challenge folder in Explorer / 7-Zip — that puts .py
files in the archive and platform validation fails.

USAGE
    python zip_raw_for_upload.py
    python zip_raw_for_upload.py --raw raw_data --out D:/uploads/BilateralMammo_RAW.zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        type=Path,
        default=Path(__file__).parent / "raw_data",
        help="Folder containing cases.csv and images/.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "BilateralMammo_RAW_upload.zip",
        help="Output zip path.",
    )
    args = ap.parse_args()
    raw: Path = args.raw.resolve()
    if not raw.is_dir():
        print(f"ERROR: missing {raw}", file=sys.stderr)
        sys.exit(1)
    cases_csv = raw / "cases.csv"
    images_dir = raw / "images"
    if not cases_csv.is_file():
        print(f"ERROR: missing {cases_csv}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"ERROR: missing {images_dir}", file=sys.stderr)
        sys.exit(1)
    out_zip: Path = args.out.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.is_file():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(cases_csv), arcname="cases.csv")
        png_files = sorted(images_dir.glob("*.png"))
        if not png_files:
            print("ERROR: images/ has no PNGs", file=sys.stderr)
            sys.exit(1)
        for p in png_files:
            zf.write(str(p), arcname=f"images/{p.name}")

    size_gb = out_zip.stat().st_size / 1e9
    print(f"OK: {out_zip}  ({len(png_files)} PNGs + cases.csv, {size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
