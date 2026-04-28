"""Build OCT_RAW_upload.zip for the platform "Data Files" uploader.

Archive root must contain ONLY:
  oct_images.csv
  images/<...>.png

Use this script (NOT Explorer / 7-Zip on the parent folder) so members sit
at the archive root.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path(__file__).parent / "raw_data")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "OCT_RAW_upload.zip")
    args = ap.parse_args()
    raw = args.raw.resolve()
    if not raw.is_dir():
        print(f"ERROR: missing {raw}", file=sys.stderr); sys.exit(1)
    csv_path = raw / "oct_images.csv"
    img_dir = raw / "images"
    if not csv_path.is_file():
        print(f"ERROR: missing {csv_path}", file=sys.stderr); sys.exit(1)
    if not img_dir.is_dir():
        print(f"ERROR: missing {img_dir}", file=sys.stderr); sys.exit(1)
    out_zip = args.out.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.is_file():
        out_zip.unlink()
    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print("ERROR: no PNGs in images/", file=sys.stderr); sys.exit(1)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(csv_path), arcname="oct_images.csv")
        for p in pngs:
            zf.write(str(p), arcname=f"images/{p.name}")
    size_gb = out_zip.stat().st_size / 1e9
    print(f"OK: {out_zip}  ({len(pngs)} PNGs + oct_images.csv, {size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
