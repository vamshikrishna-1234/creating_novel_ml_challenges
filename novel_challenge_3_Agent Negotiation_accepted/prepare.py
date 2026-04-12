"""
Prepare script: transforms raw data.csv into public/ and private/ splits.
Deterministic (fixed random_state). No row overlap between train and test.

Hardening strategies to prevent trivial classification:
1. Truncate last 2 turns — removes outcome signal
2. Mask NEXO protocol tokens to generic NEXO:ACTION
3. Redact exact prices to bucketed ranges
4. Shuffle middle turns for ~20% of rows (determined by id hash, fully deterministic)
5. Inject missing sector for ~15% of rows (determined by id hash, fully deterministic)
6. Use 75/25 split with seed 123
"""

from pathlib import Path
import re
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 123


def _truncate_last_n_turns(transcript, n=2):
    parts = transcript.split(" ||| ")
    if len(parts) <= n:
        return parts[0] if parts else ""
    return " ||| ".join(parts[:-n])


def _mask_protocol_tokens(transcript):
    signal_tokens = [
        "NEXO:OFFER", "NEXO:BID", "NEXO:PROPOSE",
        "NEXO:COUNTER", "NEXO:REVISE", "NEXO:ADJUST",
        "NEXO:ACCEPT", "NEXO:AGREE", "NEXO:CONFIRM",
        "NEXO:REJECT", "NEXO:DECLINE", "NEXO:REFUSE",
        "NEXO:HOLD", "NEXO:PAUSE", "NEXO:DEFER",
        "NEXO:TIMEOUT", "NEXO:EXPIRE",
    ]
    for tok in signal_tokens:
        transcript = transcript.replace(tok, "NEXO:ACTION")
    return transcript


def _redact_prices(transcript):
    def bucket(match):
        val = int(match.group(0))
        if val < 500:
            return "<500>"
        elif val < 1500:
            return "<1500>"
        elif val < 3000:
            return "<3000>"
        elif val < 5000:
            return "<5000>"
        else:
            return "<5000+>"
    return re.sub(r'\b\d{3,5}\b', bucket, transcript)


def _shuffle_middle_turns(transcript, row_id):
    """Deterministic shuffle based on row id (no shared RNG state)."""
    parts = transcript.split(" ||| ")
    if len(parts) <= 3:
        return transcript
    first = parts[0]
    last = parts[-1]
    middle = parts[1:-1]
    rng = np.random.RandomState(SEED + row_id)
    rng.shuffle(middle)
    return " ||| ".join([first] + middle + [last])


def _should_modify(row_id, salt, threshold):
    """Deterministic decision based on row id — uses md5, not Python hash()."""
    key = f"{row_id}_{salt}_{SEED}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    return (h % 1000) < int(threshold * 1000)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "transcript", "num_turns", "sector", "label"}
    if set(df.columns) != required_cols:
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    df = df.copy()

    # 1. Truncate last 2 turns
    df["transcript"] = df["transcript"].apply(lambda t: _truncate_last_n_turns(t, n=2))
    df["num_turns"] = df["transcript"].apply(lambda t: len(t.split(" ||| ")))

    # 2. Mask protocol tokens
    df["transcript"] = df["transcript"].apply(_mask_protocol_tokens)

    # 3. Redact prices
    df["transcript"] = df["transcript"].apply(_redact_prices)

    # 4. Shuffle middle turns for ~20% of rows (deterministic by id)
    shuffle_mask = df["id"].apply(lambda rid: _should_modify(rid, "shuffle", 0.20))
    for idx in df.index[shuffle_mask]:
        rid = df.at[idx, "id"]
        df.at[idx, "transcript"] = _shuffle_middle_turns(df.at[idx, "transcript"], rid)

    # 5. Missing sector for ~15% of rows (deterministic by id)
    missing_mask = df["id"].apply(lambda rid: _should_modify(rid, "missing", 0.15))
    df.loc[missing_mask, "sector"] = ""

    # 6. Split 75/25
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=SEED, shuffle=True, stratify=df["label"]
    )

    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(test_ids), "Train and test must not share any ids"

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(public / "train.csv", index=False)
    test_df[["id", "transcript", "num_turns", "sector"]].to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    sample["label"] = "deal_accepted"
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "label"]].to_csv(private / "answers.csv", index=False)
