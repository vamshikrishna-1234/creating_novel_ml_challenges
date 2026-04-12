"""
Synthetic dataset generator:
  Multi-Agent Negotiation Transcripts — Predict Deal Outcome

Domain: Two synthetic "agents" (buyer/seller) negotiate over fictional goods
using a mix of English phrases and invented protocol tokens. Each row is one
complete negotiation transcript (multi-turn dialogue). The task: predict the
outcome as one of 4 classes:
    0 = deal_accepted
    1 = deal_rejected
    2 = counter_proposed
    3 = timeout

Novelty:
  - Multi-turn dialogue structure with invented protocol tokens (no public benchmark)
  - Outcome depends on interaction patterns across turns (concession rate, protocol
    token sequences, offer gap trajectory) — not just single-turn text
  - 4-class classification scored by macro F1 weighted by per-class log-loss
    (custom composite metric)
  - Fictional protocol language ("Nexari") for offer/counter/accept/reject moves

Output: data.csv with columns [id, transcript, num_turns, sector, label]

Usage:
    python generate.py [--output data.csv] [--seed 42] [--size 30000]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SEED = 42
LABELS = ["deal_accepted", "deal_rejected", "counter_proposed", "timeout"]

# ---- Nexari protocol tokens (invented) ----
PROTO_OFFER = ["NEXO:OFFER", "NEXO:BID", "NEXO:PROPOSE"]
PROTO_COUNTER = ["NEXO:COUNTER", "NEXO:REVISE", "NEXO:ADJUST"]
PROTO_ACCEPT = ["NEXO:ACCEPT", "NEXO:AGREE", "NEXO:CONFIRM"]
PROTO_REJECT = ["NEXO:REJECT", "NEXO:DECLINE", "NEXO:REFUSE"]
PROTO_WAIT = ["NEXO:HOLD", "NEXO:PAUSE", "NEXO:DEFER"]
PROTO_META = ["NEXO:PING", "NEXO:STATUS", "NEXO:TIMEOUT", "NEXO:EXPIRE"]

# ---- Fictional goods and sectors ----
GOODS = [
    "zephyr-alloy", "crylon-fiber", "verium-crystal", "plasma-coil", "neuron-chip",
    "flux-capacitor", "tritanium-rod", "quantum-relay", "graphene-mesh", "ion-cell",
    "duranium-plate", "photon-array", "nano-weave", "helion-core", "synth-polymer",
]
SECTORS = ["energy", "defense", "biotech", "logistics", "manufacturing"]

# ---- English negotiation phrases ----
BUYER_OPEN = [
    "We're interested in purchasing {qty} units of {good}.",
    "Looking to acquire {qty} {good} for our {sector} division.",
    "We need {qty} of {good}, what's your best price?",
    "Our {sector} team requires {qty} {good} urgently.",
]
SELLER_OPEN = [
    "We can offer {good} at {price} credits per unit.",
    "Current rate for {good} is {price} credits. Volume discounts possible.",
    "Available stock of {good}: {price} credits/unit, minimum order {qty}.",
]
BUYER_COUNTER = [
    "That's above our budget. Can you do {price} credits?",
    "We'd prefer {price} per unit. Our ceiling is firm.",
    "Counter at {price}. We have alternative suppliers.",
    "{price} is our max. Let us know.",
]
SELLER_COUNTER = [
    "Best we can do is {price} credits. Margin is thin.",
    "Adjusted to {price} per unit, final offer.",
    "We'll meet you at {price}. Can't go lower on {good}.",
]
CONCESSION = [
    "We're willing to compromise.", "Let's find middle ground.",
    "Adjusting terms slightly.", "Flexibility on our end.",
]
STALL = [
    "Need to check with management.", "Let us review internally.",
    "Hold on, awaiting approval.", "This requires further discussion.",
]
CLOSE_ACCEPT = [
    "Agreed. Finalizing at {price} credits for {qty} units.",
    "Deal confirmed. {qty} {good} at {price}/unit.",
]
CLOSE_REJECT = [
    "No deal. Terms are unacceptable.", "We're walking away.",
    "Cannot proceed at this price.", "Negotiations terminated.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)


def _fmt(template: str, **kw) -> str:
    try:
        return template.format(**kw)
    except KeyError:
        return template


def generate_transcript(
    rng: random.Random,
    label_idx: int,
) -> dict:
    """Generate one multi-turn negotiation transcript with outcome label."""
    good = rng.choice(GOODS)
    sector = rng.choice(SECTORS)
    qty = rng.randint(50, 5000)
    base_price = rng.randint(100, 9000)
    seller_price = base_price
    buyer_price = int(base_price * rng.uniform(0.4, 0.85))

    label = LABELS[label_idx]

    # Number of turns depends on outcome
    if label == "deal_accepted":
        num_turns = rng.randint(4, 10)
    elif label == "deal_rejected":
        num_turns = rng.randint(3, 8)
    elif label == "counter_proposed":
        num_turns = rng.randint(6, 14)
    else:  # timeout
        num_turns = rng.randint(8, 16)

    turns = []

    # Turn 1: buyer opens
    t = _fmt(rng.choice(BUYER_OPEN), qty=qty, good=good, sector=sector, price=buyer_price)
    proto = rng.choice(PROTO_OFFER)
    turns.append(f"[BUYER] {proto} {t}")

    # Turn 2: seller responds
    t = _fmt(rng.choice(SELLER_OPEN), qty=qty, good=good, sector=sector, price=seller_price)
    proto = rng.choice(PROTO_OFFER)
    turns.append(f"[SELLER] {proto} {t}")

    # Middle turns: negotiation
    for i in range(2, num_turns - 1):
        gap = seller_price - buyer_price
        concession_rate = rng.uniform(0.05, 0.3)

        if i % 2 == 0:  # buyer turn
            if label in ("deal_accepted", "counter_proposed") and rng.random() < 0.6:
                buyer_price = int(buyer_price + gap * concession_rate)
                t = _fmt(rng.choice(BUYER_COUNTER), price=buyer_price, good=good)
                proto = rng.choice(PROTO_COUNTER)
                turns.append(f"[BUYER] {proto} {t}")
            elif label == "timeout" and rng.random() < 0.5:
                turns.append(f"[BUYER] {rng.choice(PROTO_WAIT)} {rng.choice(STALL)}")
            else:
                t = _fmt(rng.choice(BUYER_COUNTER), price=buyer_price, good=good)
                proto = rng.choice(PROTO_COUNTER if rng.random() < 0.7 else PROTO_REJECT)
                turns.append(f"[BUYER] {proto} {t}")
        else:  # seller turn
            if label in ("deal_accepted", "counter_proposed") and rng.random() < 0.5:
                seller_price = int(seller_price - gap * concession_rate)
                t = _fmt(rng.choice(SELLER_COUNTER), price=seller_price, good=good)
                proto = rng.choice(PROTO_COUNTER)
                turns.append(f"[SELLER] {proto} {t}")
            elif label == "timeout" and rng.random() < 0.4:
                turns.append(f"[SELLER] {rng.choice(PROTO_WAIT)} {rng.choice(STALL)}")
            else:
                t = _fmt(rng.choice(SELLER_COUNTER), price=seller_price, good=good)
                proto = rng.choice(PROTO_COUNTER)
                turns.append(f"[SELLER] {proto} {t}")

        # Occasional concession phrase
        if rng.random() < 0.2:
            speaker = "[BUYER]" if i % 2 == 0 else "[SELLER]"
            turns.append(f"{speaker} {rng.choice(CONCESSION)}")

    # Final 1-2 turns: NEUTRAL only — do NOT emit the outcome token (NEXO:ACCEPT/REJECT/TIMEOUT/final COUNTER).
    # The label is determined by the trajectory (concession rate, stall count, reject density) encoded in
    # the middle turns; the agent must infer resolution from the full transcript, not the last line.
    NEUTRAL_CLOSINGS = [
        "[BUYER] NEXO:STATUS Reviewing terms.",
        "[SELLER] NEXO:PING Please confirm receipt.",
        "[BUYER] NEXO:HOLD Checking with team.",
        "[SELLER] NEXO:STATUS We'll get back shortly.",
        "[BUYER] NEXO:PING Awaiting your response.",
        "[SYSTEM] NEXO:STATUS Session active.",
    ]
    turns.append(rng.choice(NEUTRAL_CLOSINGS))
    if rng.random() < 0.4:
        turns.append(rng.choice(NEUTRAL_CLOSINGS))

    transcript = " ||| ".join(turns)

    return {
        "transcript": transcript,
        "num_turns": num_turns,
        "sector": sector,
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate negotiation transcript dataset")
    parser.add_argument("--output", type=Path, default=Path("data.csv"), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--size", type=int, default=30_000, help="Total number of samples")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    # Class distribution: slight imbalance (realistic)
    # deal_accepted: 30%, deal_rejected: 25%, counter_proposed: 28%, timeout: 17%
    weights = [0.30, 0.25, 0.28, 0.17]
    cum_weights = []
    s = 0
    for w in weights:
        s += w
        cum_weights.append(s)

    rows = []
    for i in range(args.size):
        r = rng.random()
        label_idx = 0
        for j, cw in enumerate(cum_weights):
            if r <= cw:
                label_idx = j
                break
        row = generate_transcript(rng, label_idx)
        row["id"] = i
        rows.append(row)

    # Verify no duplicate transcripts
    texts = [r["transcript"] for r in rows]
    assert len(texts) == len(set(texts)), "Duplicate transcripts found"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "transcript", "num_turns", "sector", "label"])
        w.writeheader()
        w.writerows(rows)

    # Stats
    from collections import Counter
    labels = Counter(r["label"] for r in rows)
    turns = [r["num_turns"] for r in rows]
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Labels: {dict(sorted(labels.items()))}")
    print(f"  Turns: min={min(turns)}, max={max(turns)}, mean={sum(turns)/len(turns):.1f}")
    print(f"  Sectors: {dict(sorted(Counter(r['sector'] for r in rows).items()))}")


if __name__ == "__main__":
    main()
