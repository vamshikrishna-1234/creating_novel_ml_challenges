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

**Download + build `raw_data/`** (TCIA; slow, may retry often):

```bat
python generate.py --out raw_data
```

Smaller test run first:

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

**Download + build `raw_data/`** (Mendeley archive; one large zip, then extract):

```bat
python generate.py --out raw_data
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
