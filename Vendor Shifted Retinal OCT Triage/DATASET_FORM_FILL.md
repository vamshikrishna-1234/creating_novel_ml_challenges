# Dataset creation form — fill-in

**Platform status:** **Draft** — paired challenge *Vendor Shifted Retinal OCT Triage*.

## Title

```
Retinal OCT B-Scan Corpus With Disease And Layer-Position Annotations
```

## Description

## Overview

This dataset is a curated mirror of the **Kermany et al. 2018 labelled
optical coherence tomography (OCT) corpus** (Mendeley Data,
DOI 10.17632/rscbjbr9sj.2), repackaged into a flat schema that exposes
the original 4-class disease label per B-scan together with a derived
per-image **layer_position** scalar. Pixel content is unaltered: PNGs are
re-encoded byte-equivalent (8-bit grayscale, max-normalised) from the
upstream JPEGs, and the layer-position scalar is computed deterministically
from the image alone — no extra ground-truth annotations beyond what
Mendeley Data ships are required.

The downstream challenge uses this corpus to train a model that
simultaneously predicts disease class and layer position, and is graded
across in-domain plus two photometric-shift test slices.

## File Structure

The dataset comprises two top-level components:

```
oct_images.csv     per-image metadata
images/            folder of 8-bit grayscale OCT B-scan PNGs
```

`oct_images.csv` columns:

- `raw_id` (string): a 16-character hex content hash of the upstream image,
  stable across re-runs of `generate.py`.
- `disease` (string): one of `CNV, DME, DRUSEN, NORMAL`.
- `layer_position` (float in `[0,1]`): relative vertical position of the
  brightest horizontal band in the B-scan, computed by row-mean centroid.
- `image_file` (string): filename inside `images/`.

## Features

| Column          | Type   | Description                                          |
|-----------------|--------|------------------------------------------------------|
| raw_id          | string | 16-hex content hash of the upstream image.           |
| disease         | string | One of `CNV, DME, DRUSEN, NORMAL`.                   |
| layer_position  | float  | Normalised vertical position in `[0, 1]`.            |
| image_file      | string | PNG filename inside `images/`.                       |

## Notes

- Every PNG is a re-encoded copy of the upstream Kermany OCT-2017 JPEG
  (8-bit grayscale, max-normalised). Pixel content is unaltered.
- Each B-scan is downscaled so its long edge is at most **384 px**
  (`--max-edge` in `generate.py`) before PNG encoding. OCT B-scans natively
  span ~512–1024 px in the narrow axis; 384 px keeps retinal-layer banding
  fully resolvable while bounding each PNG to ~10–25 kB.
- Per-class coverage is capped to **800 B-scans per disease**
  (`--max-per-class 800`) by a deterministic shuffle, giving ~3.2k images
  total. That preserves enough per-class diversity for both the in-scope
  and the vendor-shift / corrupted test slices, while keeping `raw_data/`
  to roughly **100–250 MB** — small enough for the platform raw-upload
  server to ingest reliably.
- All curation is reproducible: `generate.py` in this dataset folder
  downloads the upstream archive from Mendeley Data, extracts the canonical
  `OCT2017/{train,test,val}/{CNV,DME,DRUSEN,NORMAL}/*.jpeg` layout,
  subsamples per-class, resizes, and emits `oct_images.csv` + `images/`.
- The downstream `prepare.py` performs the train / test split, permutes
  image identifiers, assigns each test image to one of the three slices
  `{inscope, vendor, corrupted}`, applies the slice's perturbation to the
  PNG on disk, and renames the files.

---

## License

**CC BY 4.0 (Creative Commons Attribution 4.0 International).** This is the
license declared by the upstream Kermany et al. corpus on Mendeley Data
(both v1 and v2). CC BY 4.0 **permits commercial redistribution with
attribution**.

License verification (organiser-side; performed before this dataset was
prepared, against primary sources):

- **Mendeley Data primary listing**:
  https://data.mendeley.com/datasets/rscbjbr9sj/2 — explicitly lists the
  dataset license as **CC BY 4.0**.
- **Original publication**: Kermany, D., Goldbaum, M., et al. (2018).
  *Identifying medical diagnoses and treatable diseases by image-based
  deep learning.* Cell, 172(5), 1122–1131.
- **CC BY 4.0 deed**: https://creativecommons.org/licenses/by/4.0/ —
  permits use, redistribution, and adaptation, including commercial,
  with attribution.
- We did not rely on third-party Kaggle license tags; we verified the
  license directly at the Mendeley Data primary listing for the original
  DOI.

## Source

- Primary source: https://data.mendeley.com/datasets/rscbjbr9sj/2
- Paper: Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S.,
  Liang, H., Baxter, S. L., et al. (2018). *Identifying medical diagnoses
  and treatable diseases by image-based deep learning.* Cell 172(5),
  1122–1131. https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
- DOI: 10.17632/rscbjbr9sj.2
- Packaging script: `generate.py` in this dataset folder.
- Downstream curation script: `prepare.py` in the paired challenge folder.
