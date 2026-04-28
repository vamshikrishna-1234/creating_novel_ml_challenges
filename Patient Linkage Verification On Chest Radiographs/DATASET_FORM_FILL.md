# Dataset creation form — fill-in

**Platform status:** **Draft** — paired challenge *Patient Linkage Verification On Chest Radiographs With Decoy Pathology Matching*.

## Title

```
Anonymised Chest Radiograph Corpus With Per-Image Pathology Pattern Annotations
```

## Description

## Overview

This dataset is a curated mirror of the **NIH ChestX-ray14** corpus
(Wang et al., CVPR 2017), repackaged into a flat schema that exposes the
per-image patient identifier and a 14-class binary pathology pattern.
Pixel content is unaltered: PNGs are byte-identical re-encodes from the
upstream `images_*.tar.gz` archives. The 14 pathology columns are
anonymised: their canonical NIH disease names are replaced with integer
indices `0..13` via a fixed organiser-side permutation that lives in
`generate.py` (constant `DISEASE_PERM`).

The downstream challenge uses this corpus to train a model that
simultaneously verifies same-patient identity across pathology-matched
candidate galleries and predicts the per-image pathology pattern.

## File Structure

The dataset comprises two top-level components:

```
cxr_index.csv     per-image metadata (one row per image)
images/           folder of 8-bit grayscale chest X-ray PNGs.
```

`cxr_index.csv` columns:

- `raw_patient_id` (string): NIH `Patient ID`.
- `raw_image_id` (string): NIH `Image Index` minus the `.png` suffix.
- `image_file` (string): filename inside `images/`.
- `has_0 .. has_13` (int 0/1): anonymised binary pathology indicators.

## Features

| Column         | Type   | Description                                                |
|----------------|--------|------------------------------------------------------------|
| raw_patient_id | string | Upstream NIH `Patient ID`.                                 |
| raw_image_id   | string | Upstream NIH `Image Index` minus `.png`.                   |
| image_file     | string | PNG filename inside `images/`.                             |
| has_0          | int    | Anonymised pathology indicator 0 (binary).                 |
| ...            | ...    | ...                                                        |
| has_13         | int    | Anonymised pathology indicator 13 (binary).                |

## Notes

- Every PNG is a re-encoded 8-bit grayscale copy of the upstream NIH
  ChestX-ray14 PNG, downscaled so its long edge is at most **768 px**
  (`--max-edge` in `generate.py`). Native NIH images are 1024×1024; 768 px
  keeps diagnostic anatomy visible while halving the per-image footprint
  to ~150–300 kB.
- Patient coverage is capped at the **first 800 NIH patients**
  (`--max-patients 800`) and **at most 8 studies per patient**
  (`--max-images-per-patient 8`) via a deterministic shuffle. The full
  ChestX-ray14 release is ~42 GB across 12 .tar.gz parts, so the long-tail
  patients (some with 100+ images) are explicitly trimmed. Defaults yield
  ~5–6k images and a `raw_data/` of roughly **1–2 GB** — small enough for
  the platform raw-upload server to ingest reliably while keeping enough
  patient diversity for non-trivial decoy galleries in the linkage task.
- The mapping from the 14 NIH disease labels to anonymised indices `0..13`
  is fixed by the `DISEASE_PERM` constant in `generate.py`. The reverse
  mapping is held organiser-side and is NOT shipped with the upload, so
  participants cannot recover the canonical NIH disease names from the
  public files.
- All curation is reproducible: `generate.py` in this dataset folder
  reads NIH's `Data_Entry_2017_v2020.csv` and the 12 `images_*.tar.gz`
  archives (which the organiser drops into `raw_data/_work/` once,
  because NIH's Box public-share URLs require an interactive browser flow),
  subsamples and resizes, then emits `cxr_index.csv` + `images/`
  deterministically.
- The downstream `prepare.py` performs the patient-level train / test
  split, anonymises both patient and image identifiers, samples
  pathology-matched decoy candidates per query under a Hamming-distance
  constraint, and renames the PNGs.
- No demographic, acquisition, or scanner metadata is included beyond the
  NIH-released image-level finding labels.

---

## License

The NIH ChestX-ray14 corpus is published with the following statement on
the official NIH Clinical Center mirror (verbatim, from the Box-hosted
README that accompanies the upstream archives):

> *"There are no restrictions on the use of the NIH chest x-ray images.
> However, the dataset has the following attribution requirements..."*

The required attribution (which we satisfy in the **Source** section
below) is:

- A link to the official NIH download site
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- Citation of Wang et al. CVPR 2017.
- Acknowledgement of the NIH Clinical Center as data provider.

This permissive notice is treated as commercial-use-permitted by Google
Cloud Healthcare, AWS Open Data, Hugging Face, and academic mirrors. We
did NOT rely on third-party "CC0 / Public Domain" labels seen on Kaggle
(those are platform-side simplifications, not the canonical NIH terms).
This dataset reproduces the unaltered NIH terms in the **Source** block,
as required.

License verification (organiser-side; performed before this dataset was
prepared, against primary sources):

- **NIH Clinical Center primary mirror (Box)**:
  https://nihcc.app.box.com/v/ChestXray-NIHCC — README explicitly states
  "There are no restrictions on the use of the NIH chest x-ray images"
  with the attribution requirements reproduced above.
- **Google Cloud Healthcare API public-dataset listing**:
  https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
  — independently confirms NIH's "no restrictions on use" statement and
  reproduces the same attribution requirements.
- **AWS Open Data Registry** also lists the dataset with the same NIH
  terms.

We comply with the NIH attribution requirement by including the citation
to Wang et al. 2017 and a link to the official NIH download mirror in
this DATASET_FORM_FILL.md and in `generate.py`. The challenge form
(participant-facing) intentionally does not name the upstream corpus, so
that the linkage / pathology-prediction tasks remain learnable from the
shipped data alone.

## Source

- Primary source: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Paper (cite this as required by the NIH attribution clause):
  Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M.
  (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and
  Benchmarks on Weakly-Supervised Classification and Localization of
  Common Thorax Diseases.* IEEE CVPR, pp. 3462–3471.
  https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
- Acknowledgement (required): NIH Clinical Center, the data provider.
- Files used:
  - `Data_Entry_2017_v2020.csv` — NIH-released per-image finding labels
    and patient identifiers.
  - 12 `images_*.tar.gz` archives at the official NIH Box mirror —
    full-resolution PNGs.
- Packaging script: `generate.py` in this dataset folder.
- Downstream curation script: `prepare.py` in the paired challenge folder.
