"""
Prepare script (v2 — hardened): transforms raw data.csv into public/ and private/ splits.

Deterministic (fixed random_state + hashlib for row-level operations).

Obfuscation pipeline (applied to every memo):
  1.  Strip [ORBIT:VERDICT] (contains the answer) and [ORBIT:REF] (unique hash)
  2.  Mask ALL ORB risk codes (ORB-V3 → ORB-CODE, etc.)
  3.  Replace all shortfall amounts with single token <QTY>
  4.  Replace all cycle counts with single token <PERIOD>
  5.  Replace all cascade depths with single token <DEPTH>
  6.  Mask mitigation effectiveness → [MASKED]
  7.  Mask cascade propagation rate → [MASKED]
  8.  Mask downstream impact severity → [MASKED]
  9.  Redact entire VECTOR narrative in ~55% of rows
  10. Shuffle ORBIT section order in ~35% of rows
  11. Blank out region in ~30% of rows
  12. Blank out commodity_class in ~20% of rows
  13. Word-drop noise: randomly remove 1–3 words from surviving narrative (~30% of non-redacted rows)
  14. Strip [ORBIT:NOTE] distractor section entirely (~50% of the time)
"""

from pathlib import Path
import re
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


def _should_modify(row_id: int, salt: str, threshold: float) -> bool:
    h = hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < threshold


def _det_rng(row_id: int, salt: str):
    """Return a deterministic stdlib Random for this row+salt."""
    import random as _rnd
    seed = int(hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest(), 16) % (2**32)
    return _rnd.Random(seed)


# ---- Section stripping ----

def _strip_verdict_and_ref(memo: str) -> str:
    memo = re.sub(r'\[ORBIT:VERDICT\][^[]*$', '', memo).strip()
    memo = re.sub(r'\[ORBIT:VERDICT\][^[]*', '', memo).strip()
    memo = re.sub(r'\[ORBIT:REF\][^[]*$', '', memo).strip()
    memo = re.sub(r'\[ORBIT:REF\][^[]*', '', memo).strip()
    return memo


def _strip_note_section(memo: str) -> str:
    memo = re.sub(r'\[ORBIT:NOTE\][^[]*$', '', memo).strip()
    memo = re.sub(r'\[ORBIT:NOTE\][^[]*', '', memo).strip()
    return memo


# ---- Code masking ----

def _mask_risk_codes(memo: str) -> str:
    memo = re.sub(r'ORB-V\d+', 'ORB-CODE', memo)
    memo = re.sub(r'ORB-I\d+', 'ORB-CODE', memo)
    memo = re.sub(r'ORB-M\d+', 'ORB-CODE', memo)
    return memo


# ---- Numeric masking (all numbers → single placeholder) ----

def _mask_numbers(memo: str) -> str:
    memo = re.sub(r'shortfall \d+ units', 'shortfall <QTY> units', memo)
    memo = re.sub(r'over \d+ cycles', 'over <PERIOD> cycles', memo)
    memo = re.sub(r'\d+ nodes', '<DEPTH> nodes', memo)
    memo = re.sub(r'\d+ dependent nodes', '<DEPTH> dependent nodes', memo)
    memo = re.sub(r'\d+ node chain', '<DEPTH> node chain', memo)
    return memo


# ---- Categorical masking ----

_EFFECTIVENESS_TERMS = ['very-high', 'high', 'moderate', 'low', 'very-low', 'negligible']
_CASCADE_RATE_TERMS = ['slow', 'contained', 'moderate', 'steady', 'rapid',
                       'accelerating', 'exponential', 'uncontrolled']
_DOWNSTREAM_TERMS = ['negligible', 'minimal', 'low', 'moderate', 'significant',
                     'severe', 'critical', 'catastrophic']


def _mask_categorical_terms(memo: str) -> str:
    for term in _EFFECTIVENESS_TERMS:
        memo = memo.replace(f'effectiveness: {term}', 'effectiveness: [MASKED]')
    for term in _CASCADE_RATE_TERMS:
        memo = memo.replace(f'rate: {term}', 'rate: [MASKED]')
        memo = memo.replace(f'Rate: {term}', 'Rate: [MASKED]')
    for term in _DOWNSTREAM_TERMS:
        memo = memo.replace(f'effect: {term}', 'effect: [MASKED]')
    return memo


# ---- Narrative redaction ----

def _redact_vector_narrative(memo: str) -> str:
    memo = re.sub(
        r'(\[ORBIT:VECTOR\] Risk code ORB-CODE: )[^.\[]+\.',
        r'\1[narrative redacted].',
        memo,
    )
    return memo


# ---- Word-drop noise ----

def _word_drop_narrative(memo: str, row_id: int) -> str:
    """Drop 1-3 random words from the VECTOR narrative portion."""
    m = re.search(r'(\[ORBIT:VECTOR\] Risk code ORB-CODE: )([^.\[]+)(\.)', memo)
    if not m:
        return memo
    narrative = m.group(2)
    words = narrative.split()
    if len(words) <= 3:
        return memo
    rng = _det_rng(row_id, "worddrop")
    n_drop = rng.randint(1, min(3, len(words) - 2))
    drop_indices = set(rng.sample(range(len(words)), n_drop))
    kept = [w for i, w in enumerate(words) if i not in drop_indices]
    return memo[:m.start(2)] + " ".join(kept) + memo[m.end(2):]


# ---- Section shuffling ----

def _shuffle_sections(memo: str, row_id: int) -> str:
    parts = re.split(r'(?=\[ORBIT:)', memo)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return memo
    rng = _det_rng(row_id, "shuffle")
    rng.shuffle(parts)
    return ' '.join(parts)


def _count_sections(memo: str) -> int:
    return len(re.findall(r'\[ORBIT:\w+\]', memo))


# ---- Main obfuscation ----

def _obfuscate_memo(memo: str, row_id: int) -> str:
    memo = _strip_verdict_and_ref(memo)
    memo = _mask_risk_codes(memo)
    memo = _mask_numbers(memo)
    memo = _mask_categorical_terms(memo)

    # Strip NOTE distractor ~50% of the time (adds noise when present, adds noise when absent)
    if _should_modify(row_id, 'strip_note', 0.50):
        memo = _strip_note_section(memo)

    # Redact narrative in 55% of rows
    if _should_modify(row_id, 'redact_narrative', 0.55):
        memo = _redact_vector_narrative(memo)
    elif _should_modify(row_id, 'worddrop', 0.30):
        memo = _word_drop_narrative(memo, row_id)

    # Shuffle section order in 35% of rows
    if _should_modify(row_id, 'shuffle_sections', 0.35):
        memo = _shuffle_sections(memo, row_id)

    return memo


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "memo", "region", "commodity_class", "label"}
    if set(df.columns) != required_cols:
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    df = df.copy()
    df["memo"] = df.apply(lambda r: _obfuscate_memo(r["memo"], int(r["id"])), axis=1)
    df["num_sections"] = df["memo"].apply(_count_sections)

    # Blank region in ~30% of rows
    df["region"] = df.apply(
        lambda r: "" if _should_modify(int(r["id"]), "missing_region", 0.30) else r["region"],
        axis=1,
    )

    # Blank commodity_class in ~20% of rows
    df["commodity_class"] = df.apply(
        lambda r: "" if _should_modify(int(r["id"]), "missing_commodity", 0.20) else r["commodity_class"],
        axis=1,
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
    )

    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(test_ids), "Train and test must not share any ids"

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df[["id", "memo", "num_sections", "region", "commodity_class", "label"]].to_csv(
        public / "train.csv", index=False
    )
    test_df[["id", "memo", "num_sections", "region", "commodity_class"]].to_csv(
        public / "test.csv", index=False
    )

    _LABELS = ["tier_1_minor", "tier_2_moderate", "tier_3_significant",
               "tier_4_severe", "tier_5_critical"]
    sample = test_df[["id"]].copy()
    sample["label"] = [_LABELS[i % len(_LABELS)] for i in range(len(sample))]
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "label"]].to_csv(private / "answers.csv", index=False)
