# Rules — current expectations (read this)

**Primary reference:** `challenge_template_file_creation.txt` — it tracks the **live** platform challenge-creation form.

## Rubrics (important)

**Evaluation rubrics are removed.** Do not add a “Rubrics” / “Evaluation rubrics” section to `CHALLENGE_FORM_FILL.md` or other challenge deliverables. The form no longer asks for them.

If other files in this `rules/` folder (e.g. `dataset and challenge.txt`, `intro and datasets.txt`) still mention “5+ rubrics” or rubric types, treat that as **outdated** — it is kept for historical context only.

## New / notable (challenge form)

- **“What not to use”** — list ML techniques or libraries solvers should avoid; misuse can lead to solution rejection.
- **Enforcement on invalid approaches** — rule-only solutions or approaches that do not match the challenge domain (e.g. a fine-tuning challenge solved with heuristics) may be rejected before payout. Intent is to reward genuine learning, not score-chasing shortcuts.

## Unchanged (still in effect)

- Difficulty, title (title case), full problem description, tags, grading config, theoretical min/max, custom `grade.py`, GPU tier (A10G default; H100 for LLM train/finetune).
- `prepare.py`: deterministic split into `public/` and `private/answers.csv` as documented elsewhere.

## Raw dataset upload (ZIP) — **flat at archive root**

When authors prepare the **data-files / raw upload** as a `.zip` for the platform uploader, the archive should contain the **raw source files only** at the **root of the zip** — *not* wrapped in an extra folder like `raw_data/`, the challenge repo name, or any parent directory.

- **Do:** e.g. `tiles.npz`, `tiles.csv` (or the five `*_224.npz` for MedTriage) as top-level members of the zip; optionally provide `zip_raw_for_upload.py` that uses `arcname=filename` so the zip is flat.
- **Don’t:** “Compress the `raw_data` folder” in Windows Explorer in a way that makes paths `raw_data/tiles.npz` inside the archive, or zip the whole challenge folder (adds `.py` files and breaks validation).

This matches how **MedTriage** and **UAV** ship raw data. See `rules/dataset_template_creation.txt` and `rules/challenge_template_file_creation.txt` (Data files / input—`raw/`) for the same rule.

## Dataset creation form (`DATASET_FORM_FILL.md`) — what belongs there

**Learning (repeat mistake to avoid):** Operational instructions for building the platform raw-data upload — flat ZIP layout, “archive root must contain exactly…”, naming `zip_raw_for_upload.py`, warnings about Explorer / nested folders — belong in **`rules/`** and in-repo scripts (`zip_raw_for_upload.py` docstrings), **not** in `DATASET_FORM_FILL.md`.

In the dataset form, describe **what the corpus is**: overview, logical file structure (`csv` + `images/`), columns, license, source, notes. Authors still **follow** the ZIP rules when packaging; they just do not duplicate those mechanics inside the fill-in description.

## Raw upload size budget — **stay well under the platform cap**

**Learning (repeat mistake to avoid):** the platform rejects raw-data uploads larger than **100 GB**, and an organiser disk also has to hold the materialised `raw_data/` to package it. So when `generate.py` pulls from a giant primary source (TCIA, NIH ChestX-ray, etc.), the script must shrink the corpus to a defensible-but-bounded slice **before** packaging — do not ship a `generate.py` whose default behaviour produces a 100+ GB folder.

Concrete rules every `generate.py` must satisfy:

- Have a **non-zero default** for `--max-patients` / `--max-samples` etc., chosen so the resulting `raw_data/` is at most ~tens of GB (target the upload to ≲50 GB; never plan for >90 GB).
- Resize images to a sensible long-edge cap (`--max-edge`, e.g. 1024 px for mammograms) before PNG/JPEG encoding. Native DICOM / 4 k+ images cost 30–60× more disk than the resized version and almost never improve agent task performance vs. the resized version.
- Stream-decode each per-series archive to its final image, then **delete the per-series ZIP** so peak working-disk = one series at a time.
- Document the resulting size budget in `DATASET_FORM_FILL.md` (`Notes:` section) — e.g. "default settings produce ~2.4k images, a few GB total". **Do not** advertise "163 GB if you fetch everything" as the headline number; that signal scares reviewers and may trigger "this challenge can't be uploaded" rejections.

### Minimal-but-novel sizing (the platform server stalls on multi-GB uploads)

**Learning (repeat mistake to avoid):** even within the 100 GB cap, the platform raw-upload server **stalls on multi-GB archives**. Default `generate.py` settings should target **≤ ~2 GB** for the upload, not the full upstream corpus. The novelty axes (paired-view reasoning, vendor-shift robustness, decoy linkage, etc.) almost always depend on **diversity** (number of patients / classes), not on **per-image resolution** or **total volume**. Trim aggressively along these axes and keep diversity:

- **Cap entities, not just images.** Patient-level / class-level caps (e.g. `--max-patients`, `--max-per-class`, `--max-images-per-patient`) preserve diversity while shrinking the dataset linearly.
- **Resize hard.** Medical CV tasks usually retain everything they need at long-edge **384–1024 px** (768 is a good default for X-ray / mammogram; 384 is fine for OCT B-scans). Native DICOM / 4 k+ images cost 30–60× more disk than the resized version.
- **Pick defaults that yield a sub-GB to ~2 GB upload.** Examples that worked in this repo: mammography 200 patients × 768 px ≈ ~0.3-0.8 GB; OCT 800 per class × 384 px ≈ ~0.1-0.25 GB; chest X-ray 800 patients × 8 images × 768 px ≈ ~1-2 GB.
- **Document the budget in `DATASET_FORM_FILL.md`** as the headline ("defaults yield ~5-6k images, ~1-2 GB"), not as a footnote.

If a challenge **needs** the full corpus to be valid (rare), say so explicitly in the dataset form and provide a downsampling fallback.
