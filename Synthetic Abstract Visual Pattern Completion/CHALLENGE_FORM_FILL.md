# Synthetic Masked Icon-Map Attribute Recovery

## Overview

Each image in this challenge is a procedurally generated **icon map**: a structured 3 × 4 tableau of small symbolic icons, where exactly one icon has been masked (replaced with a grey placeholder). Every icon is characterised by five discrete latent generative factors (encoded as `prop_0` through `prop_4`). Your task is to **recover all five factor values** for the masked icon, using the visual context provided by the remaining 11 icons.

The masked position varies randomly across images. The generative process that places icons on the map mixes several mechanisms: some factors follow per-row or per-column progressions, some are governed by parity or mirror symmetries, and — critically — some factors exhibit **cross-factor conditional coupling**, meaning one factor's value is a deterministic function of another factor's value within the same icon. This coupling prevents factors from being predicted independently and requires jointly modelling multi-factor interactions.

Approximately 75% of images have three active generative mechanisms, 20% have two, and 5% have one. Factors not governed by an active mechanism may contain **interference patterns** — near-regularities that are deliberately violated to mislead. Around 15% of training labels have 1–2 factor values randomly perturbed, establishing an irreducible noise floor.

## Evaluation

Submissions are scored using **mean per-factor accuracy**: for each of the five predicted factors, the fraction of correct predictions across all test images is computed, and the final score is the average of these five accuracy values.

```python
score = mean(accuracy_prop_0, accuracy_prop_1, ..., accuracy_prop_4)
```

Higher is better. A submission predicting all zeros scores approximately 0.30 (varying by attribute cardinality). The theoretical maximum is approximately 0.85 due to label noise.

## Dataset

After preparation, the public directory contains:

| File / Directory       | Description                                              |
|------------------------|----------------------------------------------------------|
| `train/`               | 7,500 PNG images (training set)                          |
| `test/`                | 2,500 PNG images (test set, labels withheld)             |
| `train.csv`            | Training labels: `image_id, prop_0, prop_1, prop_2, prop_3, prop_4` |
| `sample_submission.csv`| Example submission with all predictions set to 0         |

Each image shows a 3 × 4 icon map with one masked position (grey, marked "?"). The five factor columns (`prop_0` through `prop_4`) are integer-valued with ranges:

| Column | Range | Cardinality |
|--------|-------|-------------|
| prop_0 | 0–5   | 6           |
| prop_1 | 0–5   | 6           |
| prop_2 | 0–2   | 3           |
| prop_3 | 0–2   | 3           |
| prop_4 | 0–1   | 2           |

The mapping between these anonymous factor indices and the actual visual properties they encode must be discovered from the training data.

## Submission

Submit a CSV file with the following format:

| Column  | Type | Description                                       |
|---------|------|---------------------------------------------------|
| image_id| int  | Image identifier from the test set                |
| prop_0  | int  | Predicted value for attribute 0 (0–5)             |
| prop_1  | int  | Predicted value for attribute 1 (0–5)             |
| prop_2  | int  | Predicted value for attribute 2 (0–2)             |
| prop_3  | int  | Predicted value for attribute 3 (0–2)             |
| prop_4  | int  | Predicted value for attribute 4 (0–1)             |

**Requirements:**
- Must contain exactly 2,500 rows (one per test image)
- Must include a header row
- All `image_id` values must be unique and match the test set
- Predicted values must be integers within the specified ranges

## Rubrics

### Rubric 1
- **Criteria:** Loads training images and correctly parses the labels CSV, associating each icon map with its five factor values.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** Incorrect data loading or misaligned image–label pairs will produce meaningless predictions for all five factors.

### Rubric 2
- **Criteria:** Produces a valid submission CSV with all required columns (`image_id, prop_0, prop_1, prop_2, prop_3, prop_4`), correct row count, unique IDs matching the test set, and integer values within the documented ranges.
- **Type:** DATA_HANDLING
- **Importance:** REQUIRED
- **Rationale:** Malformed submissions receive a score of 0.0 from the grader; structural correctness is a prerequisite for any evaluation.

### Rubric 3
- **Criteria:** Achieves a mean per-attribute accuracy exceeding 0.35 on the test set (above the all-zeros baseline of ~0.30).
- **Type:** MODELING
- **Importance:** REQUIRED
- **Rationale:** Scoring above the trivial constant baseline demonstrates that the model has extracted at least some visual signal from the icon-map images.

### Rubric 4
- **Criteria:** Uses a vision model (CNN, Vision Transformer, or similar) to extract features from the icon-map images rather than relying solely on heuristic rules or metadata.
- **Type:** MODELING
- **Importance:** RECOMMENDED
- **Rationale:** The latent factors are encoded in pixel-level iconic compositions; image-based feature extraction is necessary to decode the hidden generative mechanisms governing the map.

### Rubric 5
- **Criteria:** Does not use test labels or test-set statistics during training or feature engineering.
- **Type:** TRAINING
- **Importance:** UNIVERSAL
- **Rationale:** Using test information during training constitutes data leakage and produces unreliable performance estimates.

## Grading Configuration

```yaml
grading:
  method: "program"
  script: "grade.py"
  metric: "mean_per_attribute_accuracy"
  direction: "maximize"
```

## Grading Script

See `PASTE_THIS_GRADE.txt`.

## Prepare Script

See `PASTE_THIS_PREPARE.txt`.
