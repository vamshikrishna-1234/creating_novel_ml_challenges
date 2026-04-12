# Dataset creation form — fill-in

## Title

```
Synthetic Contextual Sequence Transduction Corpus
```

## Description

## Overview

This dataset is fully synthetic, generated programmatically using NumPy with a fixed random seed (42) for full reproducibility. It simulates a context-sensitive symbolic transformation system where input token sequences are mapped to output character sequences through multiple interacting hidden rules.

The dataset contains 30,000 (source, target) pairs. Each source string encodes a context token, a base token, and five categorical control codes. Each target string is the transformed output produced by applying the system's hidden rules to the base token, conditioned on all five control codes and the preceding context token. The solver's goal is to train a sequence model from scratch that learns to produce correct output sequences for unseen input combinations.

The 5 control code positions produce 720 unique feature combinations per root. With 500 roots, the total input space is 360,000 — but only 30,000 samples are provided (8.3% coverage). The model must generalize from sparse training examples to unseen control code combinations.

The hidden transformation system includes: progressive two-class vowel harmony governing suffix variant selection, consonant voicing assimilation at concatenation boundaries, hiatus resolution, consonant cluster breaking via epenthetic insertion, two layers of context-dependent sandhi (the context token modifies both the first appended suffix and the voice suffix through separate rules), 6 tense-dependent voice allomorphs (the voice suffix changes form based on the tense), 60 base tokens with control-code-dependent stem modifications, 40 base tokens with character rotation under specific control-code pairs, 30 base tokens with voice-specific stem alterations, 15 base tokens with suppletive stems (completely different roots under certain tense+voice combinations), 3 three-way portmanteau patterns, and 5 four-way portmanteau patterns involving the voice dimension. Approximately 10% of target labels have a single character randomly replaced as label noise.

## File Structure

- `data.csv` — The primary dataset file containing 30,000 sequence transformation pairs. Each row maps a context-conditioned input specification to its transformed output sequence. Contains 3 columns: a unique sample identifier, the source string (context + base token + 5 control codes), and the target output sequence.

## Features

| Column    | Type | Description                                                                            |
|-----------|------|----------------------------------------------------------------------------------------|
| sample_id | int  | Unique identifier (0–29999)                                                            |
| source    | str  | Space-separated string: context token, base token, then five categorical control codes |
| target    | str  | Transformed output sequence (4–24 characters, 19-letter alphabet)                      |

## Notes

- Base tokens are generated from CV(C) syllable templates using a 19-character alphabet: 14 consonants (b, d, f, g, k, l, m, n, p, r, s, t, v, z) and 5 vowels (a, e, i, o, u). There are 500 unique base tokens of 3–8 characters.
- Context tokens are 100 unique short tokens (2–3 characters) drawn from the same alphabet.
- Five control code positions with values: position 1 (PRS, PST, FUT, PRF, HAB), position 2 (SG, DU, PL), position 3 (NOM, ACC, DAT, GEN), position 4 (IND, SBJ, IMP), position 5 (ACT, MID, PAS, CAU).
- The transformation system appends coded suffixes in a fixed order, each with two variants selected by a progressive two-class harmony rule based on the last vowel of the current form.
- First-layer sandhi: the context token's final character determines a modification applied to the first appended suffix — lenition (~16%), nasal augment (~8%), fortition (~23%), or no effect (~53%).
- Second-layer sandhi: the context token's initial character and the 5th control code jointly determine a separate modification applied to the voice suffix — consonant prepending, nasalization, devoicing, or no effect.
- Phonological rules at every concatenation boundary: vowel-vowel hiatus resolution, consonant voicing assimilation, and epenthetic vowel insertion to break 3+ consonant clusters.
- 60 base tokens undergo control-code-dependent stem modifications; 40 undergo character rotation under a specific control-code pair; 30 undergo voice-specific stem alterations (prefixing or ablaut under certain voice values); 15 have suppletive stems — completely different root forms under specific tense+voice combinations.
- 6 tense-dependent voice allomorphs: the voice suffix takes a different form depending on the active tense, creating cross-position interactions that cannot be predicted from individual control codes alone.
- 3 three-way portmanteau patterns fuse tense+number+case into a single suffix (active voice only); 5 four-way portmanteau patterns fuse tense+number+case+voice into a single suffix.
- 10% of target labels have one character randomly replaced (label noise), establishing an irreducible error floor.

---

## License

Synthetic — no license required.

## Source

Synthetically generated by the dataset author.
