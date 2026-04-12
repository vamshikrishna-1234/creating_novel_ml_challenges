# Dataset Description

## Overview

Synthetic Masked Icon-Map Attribute Recovery Corpus contains 10,000 procedurally generated images of structured 3 × 4 icon maps. Each position in the map holds a small symbolic icon defined by five discrete latent generative factors. One position per image is masked (grey placeholder); its factor values must be recovered from the visual context of the remaining 11 icons. The icons are placed according to hidden generative mechanisms that may include spatial progressions, symmetries, parity alternations, and cross-factor conditional couplings.

This dataset is entirely synthetic and was created with a deterministic generation script (seed 42). No external data sources were used.

## File Structure

- `images/` — 10,000 PNG images (naming: `00000.png` … `09999.png`). Each image is a 3 × 4 icon map with one grey masked position marked "?".
- `labels.csv` — 10,000 rows, 11 columns.

## Features

| Column            | Type | Description                                                    |
|-------------------|------|----------------------------------------------------------------|
| image_id          | int  | Unique image identifier (0–9999)                               |
| blank_row         | int  | Row index of the masked position (0–2)                         |
| blank_col         | int  | Column index of the masked position (0–3)                      |
| prop_0            | int  | First latent factor of the masked icon (range 0–5)             |
| prop_1            | int  | Second latent factor of the masked icon (range 0–5)            |
| prop_2            | int  | Third latent factor of the masked icon (range 0–2)             |
| prop_3            | int  | Fourth latent factor of the masked icon (range 0–2)            |
| prop_4            | int  | Fifth latent factor of the masked icon (range 0–1)             |
| difficulty        | int  | Number of active generative mechanisms (1–3)                   |
| has_conditional   | int  | 1 if a cross-factor conditional coupling is present            |
| is_noisy          | int  | 1 if the label row was perturbed by noise                      |

## Notes

- The icon map is **3 rows × 4 columns** (12 positions). Exactly one position is masked; its location varies across images.
- Each icon is governed by 5 discrete factors with varying cardinalities (6, 6, 3, 3, 2).
- Generative mechanisms include spatial progressions, parity alternations, distribution permutations, mirror symmetries, additive modular functions, and cross-factor conditional couplings (where one factor is a deterministic function of another factor within the same icon).
- 75% of images have 3 simultaneous mechanisms; 20% have 2; 5% have 1. Approximately 35% of 3-mechanism images include a cross-factor coupling.
- 15% of label rows have 1–2 factor values randomly perturbed (label noise), establishing an irreducible error floor.
- Factors not governed by an active mechanism may contain interference: near-regularities that are deliberately violated to mislead.
