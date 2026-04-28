# Set up a second machine and download challenge raw data

After someone pushes to `main`, the GitHub repo is **already updated**. You do **not** click “accept changes” anywhere unless you opened a **Pull Request** (then you merge the PR). For a normal `git push`, just use the steps below on the new laptop.

---

## 1. Get the repo on the other laptop

Install [Git](https://git-scm.com/downloads) if needed, then in **Command Prompt** or **PowerShell**:

```bat
cd %USERPROFILE%\Downloads
git clone https://github.com/vamshikrishna-1234/creating_novel_ml_challenges.git
cd creating_novel_ml_challenges
git pull
```

If you already cloned it once, only:

```bat
cd path\to\creating_novel_ml_challenges
git pull
```

---

## 2. Shared: Python (once per machine)

Use Python 3.10+ if possible.

```bat
python -m pip install --upgrade pip
```

---

## Challenge A — Bilateral Asymmetry Anchored Mammography Triage (CBIS-DDSM)

**Folder:** `Bilateral Asymmetry Anchored Mammography Triage`

**Deps:**

```bat
cd Bilateral Asymmetry Anchored Mammography Triage
python -m pip install pydicom requests pillow numpy pandas tqdm
```

### Two ways to get the DICOMs

#### **Recommended: use the TCIA website + NBIA Data Retriever** (most reliable)

This avoids the flaky TCIA REST API entirely (the source of `ConnectionResetError` / Windows error `10054`).

1. Open https://www.cancerimagingarchive.net/collection/cbis-ddsm/
2. Scroll to the **Data Access** / **Download** section. Click the **Download** button — a small **`.tcia` manifest** file is saved.
3. Install the **NBIA Data Retriever** desktop client (Windows installer is linked from the same page; the official wiki page is https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever).
4. **Open the `.tcia` file** with NBIA Data Retriever. Accept the license, pick a destination folder (e.g. `D:\CBIS-DDSM-download`), click **Start**.
   - The Retriever resumes interrupted downloads and is much more reliable than the API.
   - You can also **deselect** patient rows in the Retriever UI before clicking Start to download only a subset (e.g. 200–400 patients) to keep it small.
5. After the download finishes, in this folder run:

```bat
python generate_from_local_dicoms.py --src "D:\CBIS-DDSM-download" --out raw_data
```

That converts the local DICOMs into the same `raw_data/` schema (no API calls). It is **resumable** — re-running skips PNGs that already exist.

#### Alternative: API-based `generate.py`

```bat
python generate.py --out raw_data
```

If you see **`ConnectionResetError` / error `10054`** or many `[generate] retry` lines: TCIA's API is closing the connection. The script retries, then skips that view after repeated failure (`WARN`). You can:

- Slow it down: `python generate.py --out raw_data --sleep-between-series 12`
- Try another network (wired, hotspot, VPN on/off).
- Re-run later; existing PNGs in `raw_data/images/` are kept.
- **Or just switch to the Recommended path above** — much more reliable.

Smaller test run (API path):

```bat
python generate.py --out raw_data --max-patients 10
```

**Zip for platform upload** (after `raw_data/` exists):

```bat
python zip_raw_for_upload.py --raw raw_data --out BilateralMammo_RAW_upload.zip
```

---

## Challenge B — Vendor Shifted Retinal OCT Triage (Kermany)

**Folder:** `Vendor Shifted Retinal OCT Triage`

**Deps:**

```bat
cd Vendor Shifted Retinal OCT Triage
python -m pip install requests pillow numpy pandas tqdm
```

**Download + build `raw_data/`**

The old **direct HTTP URL** inside `generate.py` often returns **JSON 404** (Mendeley rotates file IDs). Use the browser, then point `--zip` at the file:

1. Open https://data.mendeley.com/datasets/rscbjbr9sj/3  
2. Click **Download All** and save **OCT2017.zip** (large; several GB).  
3. Run:

```bat
python generate.py --out raw_data --zip "%USERPROFILE%\Downloads\OCT2017.zip"
```

If you already unpacked the zip so you have folders `train/CNV`, `train/DME`, etc.:

```bat
python generate.py --out raw_data --extracted "D:\path\to\OCT2017"
```

Legacy automatic download (usually fails):

```bat
python generate.py --out raw_data --try-auto-download
```

**Zip:**

```bat
python zip_raw_for_upload.py --raw raw_data --out OCT_RAW_upload.zip
```

---

## Challenge C — Patient Linkage Verification On Chest Radiographs (NIH ChestX-ray14)

**Folder:** `Patient Linkage Verification On Chest Radiographs`

NIH does **not** allow fully scripted direct downloads from Box for all files. You must download manually once, then run `generate.py`.

**Deps:**

```bat
cd Patient Linkage Verification On Chest Radiographs
python -m pip install requests pillow numpy pandas tqdm
```

**Manual step (same on any PC):**

1. Open: https://nihcc.app.box.com/v/ChestXray-NIHCC  
2. Download **`Data_Entry_2017_v2020.csv`**  
3. Download all **12** `images_001.tar.gz` … `images_012.tar.gz` (large; ~42 GB total upstream — your script defaults shrink the **output** size).  
4. Create folders and copy files:

```bat
mkdir raw_data\_work
```

Put **`Data_Entry_2017_v2020.csv`** and **all** `images_*.tar.gz` files inside:

`Patient Linkage Verification On Chest Radiographs\raw_data\_work\`

**Then:**

```bat
python generate.py --out raw_data
python zip_raw_for_upload.py --raw raw_data --out CXR_RAW_upload.zip
```

---

## Smoke test without big downloads (any challenge)

Each folder has `_synth_raw_smoke.py` for a tiny fake `raw_data/` to verify `prepare.py` / `grade.py`:

```bat
python _synth_raw_smoke.py
python prepare.py
python _sanity_baseline.py
```

---

## Push this guide from your main PC (optional)

If you add or edit this file locally:

```bat
cd path\to\creating_novel_ml_challenges
git add SETUP_NEW_MACHINE_DOWNLOAD_DATA.md
git commit -m "Add second-machine download instructions"
git push origin main
```

On the other laptop: `git pull` and open `SETUP_NEW_MACHINE_DOWNLOAD_DATA.md`.
