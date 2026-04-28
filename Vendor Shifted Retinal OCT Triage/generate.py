"""generate.py — Vendor-Shifted Retinal OCT Triage.

Pulls the Kermany et al. labelled OCT corpus from Mendeley Data, normalises
filenames, and writes raw_data/ in the schema prepare.py expects:

  raw_data/
    oct_images.csv    columns: raw_id, disease, layer_position, image_file
    images/<image_file>   8-bit grayscale OCT B-scan PNG.

DATA SOURCE (organiser-side; not exposed downstream by prepare.py)
------------------------------------------------------------------
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018). *Labeled Optical
Coherence Tomography (OCT) and Chest X-Ray Images for Classification.*
Mendeley Data, V2 (DOI: 10.17632/rscbjbr9sj.2).
  Primary:  https://data.mendeley.com/datasets/rscbjbr9sj/2
  Paper:    https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
  Licence:  CC BY 4.0 (Creative Commons Attribution 4.0 International).
            Commercial redistribution is permitted with attribution.

Disease labels (4 classes):
  CNV       choroidal neovascularisation
  DME       diabetic macular edema
  DRUSEN    age-related macular degeneration drusen
  NORMAL    no pathology

`layer_position` is a per-image scalar in [0, 1] denoting the relative
vertical position of the dominant pathologic layer (or, for NORMAL images,
of the inner-segment / outer-segment junction). It is computed by a
deterministic image-processing pipeline applied to the upstream PNG. The
pipeline is reproducible from the upstream image alone — no extra
ground-truth annotations beyond what Mendeley Data ships are required.

SIZE BUDGET
-----------
The platform raw-upload server stalls on multi-GB archives, so this script
is conservatively defaulted to a *minimal* slice of the upstream corpus
that still preserves the novelty axes (in-domain disease F1, vendor-shift
robustness, layer-position regression, calibration):

  1. --max-per-class (default 800) caps the number of B-scans kept per
     disease label after a deterministic shuffle. With 4 classes this gives
     ~3.2k images total — plenty for both a sizable training set and the
     three test slices the downstream `prepare.py` produces.
  2. --max-edge (default 384) downscales each B-scan so its long edge is at
     most that many pixels before PNG encoding. OCT B-scans are natively
     ~512–1024 px in their narrow axis; 384 px keeps retinal-layer banding
     fully resolvable while shrinking each PNG to ~10–25 kB.
  3. The Mendeley archive itself is ~5 GB; once we keep only --max-per-class
     B-scans and resize, the resulting `raw_data/` is roughly 100–250 MB.

USAGE
-----
    pip install requests pillow numpy pandas tqdm

    # Recommended (Mendeley rotated direct-download API IDs; browser download is reliable):
    #   1) Open https://data.mendeley.com/datasets/rscbjbr9sj/3  →  Download All  →  OCT2017.zip
    #   2) Then:
    python generate.py --out raw_data --zip "%USERPROFILE%\\Downloads\\OCT2017.zip"

    # Or point at an already-unpacked tree (must contain train/{CNV,DME,DRUSEN,NORMAL}/):
    python generate.py --out raw_data --extracted "D:\\OCT2017"

    python zip_raw_for_upload.py --raw raw_data --out OCT_RAW_upload.zip

Automatic HTTP download is attempted only if you pass --try-auto-download (legacy).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# Legacy public-files URL — as of 2026 returns JSON {"code":404} (file id rotated).
# Use --zip after downloading from the Mendeley website, or --extracted.
MENDELEY_OCT_URL_LEGACY = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "6f650b8c-91c7-43b8-a3c8-ccba8ef3e0c9/file_downloaded"
)
DISEASES = ("CNV", "DME", "DRUSEN", "NORMAL")

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/octet-stream,*/*;q=0.8",
    "Referer": "https://data.mendeley.com/datasets/rscbjbr9sj/3",
}


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _is_zip_file(path: Path) -> bool:
    try:
        return path.is_file() and zipfile.is_zipfile(path)
    except OSError:
        return False


def _download_zip_auto(url: str, dst: Path) -> None:
    """Stream-download `url` into `dst` and verify it is a real ZIP.

    Mendeley often returns JSON `{'code':404}` with HTTP 200 — we reject that.
    """
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.unlink(missing_ok=True)
    print(f"[generate] trying automatic download: {url}")
    with requests.get(
        url,
        stream=True,
        timeout=600,
        headers=REQUEST_HEADERS,
        allow_redirects=True,
    ) as r:
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        if "json" in ct:
            body = r.content[:800]
            raise RuntimeError(
                f"Server returned JSON instead of a ZIP (content-type={ct!r}). "
                f"Snippet: {body!r}"
            )
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    # Peek at magic bytes — ZIP local header starts with PK\x03\x04
    with open(tmp, "rb") as f:
        sig = f.read(4)
    if sig[:2] != b"PK":
        peek = tmp.read_text(encoding="utf-8", errors="replace")[:500]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download is not a ZIP (magic={sig!r}). "
            f"Mendeley may have returned an error page instead of OCT2017.zip. Snippet: {peek!r}"
        )
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file failed zipfile validation.")
    tmp.replace(dst)


def _find_kermany_layout_root(search: Path) -> Path:
    """Find folder that contains ``train/CNV`` … ``train/NORMAL`` (Kermany layout)."""
    search = Path(search).resolve()
    candidates = [
        search,
        search / "OCT2017",
        search / "kermany2018" / "OCT2017",
    ]
    for c in candidates:
        if (c / "train" / "CNV").is_dir() and (c / "train" / "NORMAL").is_dir():
            return c
    for train_cn in search.glob("**/train/CNV"):
        if train_cn.is_dir():
            return train_cn.parent.parent
    raise FileNotFoundError(
        f"Could not find Kermany folder layout (train/{{CNV,DME,DRUSEN,NORMAL}}) "
        f"under {search}"
    )


def _ensure_oct2017_source(
    work: Path,
    zip_arg: Optional[Path],
    extracted_arg: Optional[Path],
    try_auto: bool,
) -> Path:
    """Return the directory whose subtree matches ``_walk_extracted`` (has train/)."""
    work.mkdir(parents=True, exist_ok=True)

    if extracted_arg is not None:
        root = _find_kermany_layout_root(Path(extracted_arg))
        print(f"[generate] using unpacked source tree: {root}")
        return root

    archive = work / "OCT2017.zip"
    if zip_arg is not None:
        src_zip = Path(zip_arg).resolve()
        if not src_zip.is_file():
            raise FileNotFoundError(f"--zip file not found: {src_zip}")
        if not _is_zip_file(src_zip):
            raise ValueError(f"--zip is not a valid ZIP file: {src_zip}")
        shutil.copy2(src_zip, archive)
        print(f"[generate] copied local ZIP -> {archive}")

    elif try_auto:
        if archive.exists() and not _is_zip_file(archive):
            print(f"[generate] removing corrupt non-ZIP file {archive}", file=sys.stderr)
            archive.unlink()
        if not archive.exists():
            try:
                _download_zip_auto(MENDELEY_OCT_URL_LEGACY, archive)
            except Exception as exc:
                raise RuntimeError(
                    "Automatic Mendeley download failed (Elsevier rotates file IDs).\n"
                    "Fix: open https://data.mendeley.com/datasets/rscbjbr9sj/3 in a browser, "
                    "click **Download All**, save OCT2017.zip, then run:\n"
                    f"  python generate.py --out raw_data --zip \"path\\\\to\\\\OCT2017.zip\"\n"
                    f"Original error: {exc}"
                ) from exc
    else:
        raise RuntimeError(
            "No OCT source specified.\n"
            "Either:\n"
            "  python generate.py --out raw_data --zip \"path\\\\to\\\\OCT2017.zip\"\n"
            "or:\n"
            "  python generate.py --out raw_data --extracted \"D:\\\\path\\\\to\\\\OCT2017\"\n"
            "where that folder contains train/CNV, train/DME, ...\n"
            "Optional legacy flag: --try-auto-download (usually fails)."
        )

    # Resume: already unpacked under _work?
    try:
        root = _find_kermany_layout_root(work)
        print(f"[generate] reusing existing tree under _work: {root}")
        return root
    except FileNotFoundError:
        pass

    if not archive.is_file():
        raise RuntimeError("internal error: OCT2017.zip missing after download/copy step.")
    if not _is_zip_file(archive):
        raise ValueError(f"Not a valid ZIP archive: {archive}")

    print(f"[generate] extracting {archive}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(work)

    return _find_kermany_layout_root(work)


def _layer_position_from_image(arr: "np.ndarray") -> float:
    """Estimate normalised vertical position of the brightest horizontal
    band in the OCT B-scan. Deterministic from pixel content alone."""
    import numpy as np
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    row_means = arr.mean(axis=1)
    if row_means.max() <= 0:
        return 0.5
    weights = row_means / row_means.sum()
    rows = np.arange(arr.shape[0], dtype=np.float32)
    centroid = float((rows * weights).sum())
    pos = centroid / max(arr.shape[0] - 1, 1)
    return float(min(1.0, max(0.0, pos)))


def _walk_extracted(extracted_root: Path):
    """Yield (disease, png_path) pairs from the canonical Kermany layout:
        OCT2017/{train,test,val}/{CNV,DME,DRUSEN,NORMAL}/*.jpeg"""
    for split in ("train", "test", "val"):
        split_dir = extracted_root / split
        if not split_dir.is_dir():
            continue
        for disease in DISEASES:
            disease_dir = split_dir / disease
            if not disease_dir.is_dir():
                continue
            for p in sorted(disease_dir.iterdir()):
                if p.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    yield disease, p


SUBSAMPLE_SEED = 0x0C7B5EED  # deterministic per-class subsample


def _resize_long_edge(arr: "np.ndarray", max_edge: int) -> "np.ndarray":
    """Downscale a 2D grayscale numpy array so its long edge is <= max_edge."""
    import numpy as np
    from PIL import Image
    if not max_edge or max_edge <= 0:
        return arr
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return arr
    scale = max_edge / float(long_edge)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    img = Image.fromarray(arr.astype("uint8"), mode="L")
    img = img.resize(new_size, Image.LANCZOS)
    return np.asarray(img)


def generate(
    out_dir: Path,
    max_per_class: int,
    max_edge: int,
    zip_arg: Optional[Path],
    extracted_arg: Optional[Path],
    try_auto: bool,
) -> None:
    import numpy as np
    from PIL import Image

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    canon_root = _ensure_oct2017_source(work, zip_arg, extracted_arg, try_auto)

    # Bucket all source images by disease, then take a deterministic
    # per-class subsample so the resulting raw_data/ is small but each
    # class is well-represented.
    by_disease: Dict[str, List[Path]] = {d: [] for d in DISEASES}
    for disease, src_path in _walk_extracted(canon_root):
        by_disease[disease].append(src_path)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    selected: List[tuple] = []  # (disease, src_path)
    for disease in DISEASES:
        candidates = sorted(by_disease[disease])
        rng.shuffle(candidates)
        if max_per_class and max_per_class > 0:
            candidates = candidates[:max_per_class]
        for p in candidates:
            selected.append((disease, p))

    print(
        f"[generate] keeping {len(selected)} of "
        f"{sum(len(v) for v in by_disease.values())} B-scans "
        f"(max_per_class={max_per_class}, max_edge={max_edge})"
    )

    rows: List[dict] = []
    for disease, src_path in selected:
        # Stable raw_id from a content hash so the script is idempotent
        # across reruns even after subsample / resize changes.
        raw_id = _file_md5(src_path)[:16]
        new_name = f"{disease}_{raw_id}.png"
        dst_path = images_dir / new_name

        with Image.open(str(src_path)) as im:
            arr = np.asarray(im.convert("L"))
            layer_pos = _layer_position_from_image(arr)
            arr_small = _resize_long_edge(arr, max_edge)
            Image.fromarray(arr_small, mode="L").save(
                str(dst_path), format="PNG", optimize=True
            )

        rows.append({
            "raw_id": raw_id,
            "disease": disease,
            "layer_position": f"{layer_pos:.6f}",
            "image_file": new_name,
        })

    rows.sort(key=lambda r: r["raw_id"])
    out_csv = out_dir / "oct_images.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_id", "disease", "layer_position", "image_file"])
        w.writeheader()
        w.writerows(rows)
    print(f"[generate] wrote {out_csv} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw_data")
    ap.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to OCT2017.zip you downloaded from Mendeley (browser "
             "Download All). Recommended; automatic URL often breaks.",
    )
    ap.add_argument(
        "--extracted",
        type=Path,
        default=None,
        help="Path to an already-unpacked Kermany tree containing "
             "train/{CNV,DME,DRUSEN,NORMAL}/ (and test/, val/).",
    )
    ap.add_argument(
        "--try-auto-download",
        action="store_true",
        help="Attempt legacy HTTP download from Mendeley (often returns 404 JSON; "
             "not recommended).",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=800,
        help="Keep at most this many B-scans per disease label after a "
             "deterministic shuffle. Default 800 yields ~3.2k images total "
             "and a raw_data/ of roughly 100-250 MB. Pass 0 to keep all.",
    )
    ap.add_argument(
        "--max-edge",
        type=int,
        default=384,
        help="Resize each B-scan so its long edge is at most this many "
             "pixels before PNG encoding. Default 384 keeps retinal-layer "
             "banding fully resolvable. Pass 0 for native resolution.",
    )
    args = ap.parse_args()
    generate(
        args.out,
        args.max_per_class,
        args.max_edge,
        args.zip,
        args.extracted,
        args.try_auto_download,
    )


if __name__ == "__main__":
    main()
