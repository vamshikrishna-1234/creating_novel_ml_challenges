# Dataset creation form — fill-in

## 1. Title (Required)

```
Synthetic Dialect Translation Corpus
```

---

## 2. Description (Required)

Paste the following directly into the rich-text description editor (not inside a code block):

---

# Dataset Description

## Overview

This dataset is fully synthetic, created to benchmark sequence-to-sequence models on a fictional dialect-to-English translation task. It simulates a scenario where text has been written in one of five invented dialects — each applying a unique combination of lexical substitutions, morphological changes, phonetic respelling, syntax reordering, and agglutination to standard English. The task is to translate each dialect sentence back to its original English form.

The dataset was designed to require learning from training examples rather than relying on pre-existing language knowledge. Since the dialects are entirely fictional with invented vocabulary and grammar rules, zero-shot approaches using pre-trained language models will not succeed — the model must learn the transformation patterns from the provided training pairs. Each dialect applies a distinct, consistent set of rules, but these rules are never explicitly provided.

No real languages, translations, or linguistic data are used. All content is procedurally generated from English sentence templates and deterministic transformation functions.

**Dataset Summary:**

- **Input:** A sentence written in one of 5 fictional dialects, plus a dialect label.
- **Output:** The original standard English sentence.
- **5 dialects:** Each applies a unique combination of lexical, morphological, phonetic, syntactic, and agglutinative transformations.
- **Deterministic:** Each dialect transformation is rule-based and consistent — the same English input always produces the same dialect output.
- **20,000 rows** with approximately equal dialect distribution (~4,000 per dialect).

**Current raw dataset stats:**

- Total rows: 20,000
- Columns: id, dialect, transformed, original
- Dialect distribution: velthari ~20%, korathi ~20%, nelvosi ~19%, drakmori ~21%, quilmari ~20%
- Average original sentence length: ~56 characters
- Average transformed sentence length: ~53 characters

## File Structure

- `data.csv` — labeled data (id, dialect, transformed, original). Single CSV file, 20,000 rows.

## Features

### Columns

| Column      | Type   | Description |
|-------------|--------|-------------|
| id          | int    | Unique row identifier (0 to 19,999). |
| dialect     | string | Name of the fictional dialect applied to this row. One of: velthari, korathi, nelvosi, drakmori, quilmari. Each dialect applies a different combination of transformation rules to the original English sentence. |
| transformed | string | The sentence after dialect transformation has been applied. This is the input that models must translate back to English. The text uses fictional vocabulary, altered morphology, and potentially reordered syntax depending on the dialect. |
| original    | string | The original standard English sentence before any dialect transformation. This is the target output that models must predict. Sentences follow common English patterns (subject-verb-object) using technical/professional vocabulary. |

### Dialect Characteristics (high-level)

The five dialects differ in which transformation types they apply. All transformations are consistent within a dialect — the same rule always applies the same way. The dialects are:

| Dialect   | Primary Transformation Types |
|-----------|------------------------------|
| velthari  | Lexical substitution + phonetic respelling (consonant and suffix changes) |
| korathi   | Morphological suffix changes + syntax reordering (SOV word order) |
| nelvosi   | Preposition agglutination (merging prepositions with following words) + lexical substitution + suffix changes |
| drakmori  | Heavy phonetic respelling + lexical substitution + adjective placement reversal |
| quilmari  | Vowel shifting + suffix changes + determiner dropping |

## Notes

- **Synthetic origin:** All content is procedurally generated. English sentences are built from templates using professional/technical vocabulary. Dialect transformations are deterministic rule-based functions.
- **No real languages:** The five dialects are entirely fictional. They do not correspond to or derive from any real language.
- **Balanced distribution:** Rows are approximately equally distributed across the five dialects.
- **One row per id:** Each id appears exactly once. No duplicate rows.

---

---

## 3. Data Files (Required)

Upload `data.csv`. Single file, no zip or extra folders.

---

## 4. Source Files / Import from URL

Leave empty (synthetic).

---

## 5. License & Source

```
Synthetic dataset. Procedurally generated. No third-party data. No license restrictions.
```
