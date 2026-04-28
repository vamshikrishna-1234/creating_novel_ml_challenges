# Dataset creation form — fill-in

**Platform status:** **Draft** — paired challenge *Bilateral Asymmetry Anchored Mammography Triage*.

## Title

```
Bilateral Mammography View Corpus With Per-Case Asymmetry And Malignancy Labels
```

## Description

## Overview

This dataset is a curated mirror of the **CBIS-DDSM** mammographic image
collection (Curated Breast Imaging Subset of DDSM; Lee et al., *Scientific
Data* 4, 170177, 2017), repackaged into a per-case schema that exposes the
four standard mammographic views — left CC, right CC, left MLO, right MLO —
together with case-level pathology and asymmetry flags. The repackaging
keeps every image's pixel content unaltered; only the per-image rows in
`cases.csv` are re-derived from the upstream pathology metadata.

The downstream challenge uses this corpus to train a model that simultaneously
predicts (a) per-view lesion type, (b) per-case bilateral asymmetry, and
(c) per-case calibrated malignancy probability.

## File Structure

The dataset comprises two top-level components:

```
cases.csv         per-image metadata  (one row per (case, view))
images/           folder of 8-bit grayscale mammogram PNGs, one per row of
                  cases.csv (image_file column)
```

`cases.csv` columns:

- `case_raw_id` (string): patient identifier from CBIS-DDSM (`P_xxxxx`).
- `view` (string): one of `LCC, RCC, LMLO, RMLO`.
- `lesion` (string): one of `none, mass, calc`. A `none` row indicates a
  view in which the patient's contralateral side has biopsy-confirmed
  pathology but this view itself is clean (used to build the asymmetric
  bilateral structure).
- `malignant` (int, 0/1): 1 iff the upstream `pathology` field starts with
  `MALIGNANT` for this view.
- `asymmetric` (int, 0/1): 1 iff the patient has biopsy-confirmed pathology
  on exactly one side (bilateral asymmetry); 0 if pathology is bilateral or
  absent.
- `image_file` (string): filename inside `images/`.

## Features

| Column         | Type   | Description                                                          |
|----------------|--------|----------------------------------------------------------------------|
| case_raw_id    | string | Upstream CBIS-DDSM `patient_id`.                                     |
| view           | string | One of `LCC, RCC, LMLO, RMLO`.                                       |
| lesion         | string | One of `none, mass, calc`.                                           |
| malignant      | int    | 0 / 1 indicator that this view contains biopsy-malignancy.           |
| asymmetric     | int    | 0 / 1 indicator that the patient is bilaterally asymmetric.          |
| image_file     | string | Filename of the PNG inside `images/`.                                |

## Notes

- Every PNG is a re-encoded copy of the upstream CBIS-DDSM full-mammogram
  DICOM (8-bit grayscale, max-normalised). EXIF and DICOM private tags are
  stripped during conversion; no patient-level identifiers, acquisition
  metadata, or scanner information are preserved.
- Each mammogram is downscaled so its long edge is at most **768 px**
  before PNG encoding (`--max-edge` in `generate.py`). This keeps mass
  margins and macro-calcification structure that the bilateral-asymmetry
  and lesion-class tasks depend on, while bounding the per-image footprint
  to ~0.2–0.4 MB.
- Patient coverage is capped to the **first 200 CBIS-DDSM patients** (sorted
  ascending by `patient_id`) by default (`--max-patients 200`). That gives
  ~800 mammogram views and a `raw_data/` folder of roughly 300–800 MB —
  small enough for the platform raw-upload server to ingest without
  stalling. Re-running `generate.py` with the same flags is
  byte-deterministic across machines.
- All curation is reproducible: `generate.py` in this dataset folder fetches
  the four pathology CSVs from the TCIA wiki, downloads each series via the
  TCIA NBIA REST API, decodes the DICOM, resizes, and writes the PNGs
  deterministically.
- The downstream `prepare.py` performs the case-level train / test split,
  permutes case identifiers, injects view-drop on a fraction of test cases,
  and renames the PNGs.

---

## License

**CC BY 3.0 (Creative Commons Attribution 3.0 Unported).** This is the
license of the **upstream CBIS-DDSM collection** as published by The Cancer
Imaging Archive (TCIA). The CC BY 3.0 license **permits commercial
redistribution with attribution** under the TCIA Data Usage Policy.

License verification (organiser-side; performed before this dataset was
prepared, against primary sources):

- **TCIA collection page** (primary):
  https://www.cancerimagingarchive.net/collection/cbis-ddsm/ — explicitly
  lists the dataset license as **CC BY 3.0**.
- **TCIA Data Usage Policy**:
  https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/
  — confirms that CC BY 3.0 / 4.0 collections are "freely available to
  browse, download, and use for commercial, scientific and educational
  purposes" subject to attribution.
- **Attribution required** per CC BY 3.0: cite the data publication
  (Lee et al. 2017) and the original DDSM paper (Heath et al. 2001) as
  enumerated on the TCIA collection page.
- **Pixel content is unaltered**; we only re-encode DICOM as 8-bit
  grayscale PNG and add a derived per-image asymmetry flag computed from
  the public per-patient pathology metadata.

CBIS-DDSM does not carry a non-commercial restriction. We did not rely on
third-party Kaggle license tags; we verified the license at the TCIA
primary listing.

## Source

- Primary source: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- Paper: Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., &
  Rubin, D. L. (2017). *A curated mammography data set for use in
  computer-aided detection and diagnosis research.* Scientific Data, 4,
  170177. https://www.nature.com/articles/sdata2017177
- Underlying DDSM paper: Heath, M., Bowyer, K., Kopans, D., Moore, R., &
  Kegelmeyer, P. (2001). *The Digital Database for Screening Mammography.*
  Proc. 5th International Workshop on Digital Mammography, 212–218.
- Pathology metadata CSVs (mirrored by TCIA):
  - calc_case_description_train_set.csv
  - calc_case_description_test_set.csv
  - mass_case_description_train_set.csv
  - mass_case_description_test_set.csv
- Packaging script: `generate.py` in this dataset folder.
- Downstream curation script: `prepare.py` in the paired challenge folder.
