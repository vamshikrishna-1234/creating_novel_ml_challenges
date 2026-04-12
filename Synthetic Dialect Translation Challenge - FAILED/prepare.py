"""
Prepare script: transforms raw data.csv into public/ and private/ splits.

Deterministic (fixed random_state + hashlib for row-level operations).

Obfuscation pipeline (applied to every row's transformed text):
  1. Random character-level noise (~15% of rows): insert/swap/drop 1 char
  2. Whitespace normalization jitter (~20% of rows): double-space or trim
  3. Dialect label obfuscation (~30% of rows): replace dialect name with
     a code (e.g., "velthari" → "DL-A") to reduce surface-level hints
  4. Case perturbation (~25% of rows): randomly capitalize 1-2 words

Split: stratified 80/20 by dialect.
"""

from pathlib import Path
import hashlib
import random as _rnd

import pandas as pd
from sklearn.model_selection import train_test_split


DIALECT_CODES = {
    "velthari": "DL-A",
    "korathi": "DL-B",
    "nelvosi": "DL-C",
    "drakmori": "DL-D",
    "quilmari": "DL-E",
}


def _det_rng(row_id: int, salt: str) -> _rnd.Random:
    seed = int(hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest(), 16) % (2**32)
    return _rnd.Random(seed)


def _should_modify(row_id: int, salt: str, threshold: float) -> bool:
    h = hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < threshold


def _char_noise(text: str, rng: _rnd.Random) -> str:
    """Insert, swap, or drop a single character."""
    if len(text) < 5:
        return text
    op = rng.choice(["insert", "swap", "drop"])
    pos = rng.randint(1, len(text) - 2)
    if op == "insert":
        char = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return text[:pos] + char + text[pos:]
    elif op == "swap" and pos < len(text) - 1:
        lst = list(text)
        lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
        return ''.join(lst)
    else:  # drop
        return text[:pos] + text[pos+1:]


def _whitespace_jitter(text: str, rng: _rnd.Random) -> str:
    """Add or remove whitespace noise."""
    words = text.split()
    if len(words) < 3:
        return text
    pos = rng.randint(0, len(words) - 2)
    words[pos] = words[pos] + "  "  # double space
    return ' '.join(words)


def _case_perturb(text: str, rng: _rnd.Random) -> str:
    """Randomly capitalize 1-2 words."""
    words = text.split()
    if len(words) < 2:
        return text
    n = rng.randint(1, min(2, len(words)))
    indices = rng.sample(range(len(words)), n)
    for idx in indices:
        words[idx] = words[idx].capitalize()
    return ' '.join(words)


def _obfuscate_row(row_id: int, transformed: str, dialect: str) -> tuple:
    text = transformed

    if _should_modify(row_id, "char_noise", 0.15):
        rng = _det_rng(row_id, "char_noise")
        text = _char_noise(text, rng)

    if _should_modify(row_id, "ws_jitter", 0.20):
        rng = _det_rng(row_id, "ws_jitter")
        text = _whitespace_jitter(text, rng)

    if _should_modify(row_id, "case_perturb", 0.25):
        rng = _det_rng(row_id, "case_perturb")
        text = _case_perturb(text, rng)

    # Dialect label obfuscation
    dial = dialect
    if _should_modify(row_id, "dial_code", 0.30):
        dial = DIALECT_CODES.get(dialect, dialect)

    return text, dial


def prepare(raw: Path = Path("."), public: Path = Path("pub"), private: Path = Path("priv")) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "dialect", "transformed", "original"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    df = df.copy()

    new_transformed = []
    new_dialect = []
    for _, row in df.iterrows():
        t, d = _obfuscate_row(int(row["id"]), row["transformed"], row["dialect"])
        new_transformed.append(t)
        new_dialect.append(d)

    df["transformed"] = new_transformed
    df["dialect"] = new_dialect

    # Stratified split by original dialect (before obfuscation of labels)
    # We need the original dialect for stratification, so re-read
    raw_df = pd.read_csv(raw_file)
    df["_strat"] = raw_df["dialect"]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["_strat"]
    )

    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df[["id", "dialect", "transformed", "original"]].to_csv(
        public / "train.csv", index=False
    )
    test_df[["id", "dialect", "transformed"]].to_csv(
        public / "test.csv", index=False
    )

    sample = test_df[["id"]].copy()
    sample["original"] = "the engineer examines the report"
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "original"]].to_csv(private / "answers.csv", index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train dialects: {dict(train_df['dialect'].value_counts())}")
    print(f"Test dialects:  {dict(test_df['dialect'].value_counts())}")


if __name__ == "__main__":
    prepare()
