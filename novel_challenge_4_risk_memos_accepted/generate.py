"""
Synthetic dataset generator (v2 — hardened):
  ORBIT Supply-Chain Risk Memo — Disruption Severity Tier Classification

Changes from v1:
  - Shared narrative pool: every template can appear at ANY tier (weighted).
  - 30% of narratives drawn from adjacent tier ("cross-contamination").
  - Synonym substitution layer randomises surface phrasing.
  - Section presence (MITIGATION, CASCADE) flattened to ~50% each, independent
    of tier — removes structural fingerprint.
  - Added [ORBIT:NOTE] distractor section (~40% of memos) with tier-irrelevant
    boilerplate.
  - Shortfall / cycle / depth ranges heavily overlapped across tiers.
  - Downstream impact, effectiveness, cascade rate pools overlap across tiers.

Output: data.csv with columns [id, memo, region, commodity_class, label]
Usage:
    python generate.py [--output data.csv] [--seed 42] [--size 32000]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SEED = 42

LABELS = [
    "tier_1_minor",
    "tier_2_moderate",
    "tier_3_significant",
    "tier_4_severe",
    "tier_5_critical",
]

LABEL_WEIGHTS = [0.28, 0.24, 0.22, 0.16, 0.10]

REGIONS = ["norath", "veldan", "crymara", "esthos", "lunavar", "torwen"]

COMMODITIES = [
    "synth-ore", "bio-compound", "flux-crystal", "plasma-stock",
    "neuro-fiber", "cryo-element", "photon-alloy", "gravity-mesh",
]

# ---------------------------------------------------------------------------
# Shared narrative pool — every template can appear at any tier.
# Each entry: (template, primary_tiers, weight_primary, weight_other)
#   primary_tiers: tiers where this template is most likely
#   weight_primary: selection weight when tier is in primary_tiers
#   weight_other:   selection weight when tier is NOT in primary_tiers
# ---------------------------------------------------------------------------
_NARRATIVE_POOL = [
    # Originally minor-flavoured
    ("routine fluctuation in {commodity} supply from {region} corridor",          {0, 1},    5, 2),
    ("minor delay in {commodity} shipment from {region}",                         {0},       5, 2),
    ("scheduled maintenance affecting {commodity} transit in {region} zone",      {0, 1},    5, 2),
    ("temporary {commodity} processing slowdown at {region} hub",                 {0, 1},    5, 2),
    ("{commodity} supply variance detected in {region} corridor",                 {0, 1},    4, 2),
    ("brief {commodity} throughput dip at {region} relay",                        {0},       4, 2),
    # Originally moderate-flavoured
    ("{commodity} supply route via {region} corridor partially disrupted",        {1, 2},    5, 2),
    ("disruption to {commodity} distribution in {region}",                        {1, 2},    5, 2),
    ("{commodity} processing delay at {region} node exceeding tolerance",         {1, 2},    5, 2),
    ("unexpected {commodity} quality variance reported from {region}",            {1, 2},    5, 2),
    ("{commodity} logistics deviation noted in {region} sector",                  {1, 2},    4, 2),
    # Originally significant-flavoured
    ("{commodity} supply from {region} halted for unscheduled review",            {2, 3},    5, 2),
    ("{commodity} route congestion in {region} network",                          {2, 3},    5, 2),
    ("{commodity} processing facility in {region} operating below threshold",     {2, 3},    5, 2),
    ("multi-node disruption affecting {commodity} flow through {region}",         {2, 3},    5, 2),
    ("{commodity} supply chain degradation across {region} nodes",                {2},       4, 2),
    # Originally severe-flavoured
    ("{commodity} extraction halted in {region} zone due to cascading failure",   {3, 4},    5, 1),
    ("severe {commodity} supply interruption across {region} sector",             {3, 4},    5, 1),
    ("critical path for {commodity} through {region} compromised",               {3, 4},    5, 1),
    ("{commodity} stockpile in {region} below emergency minimum",                {3, 4},    5, 1),
    ("{commodity} reserve depletion reported in {region} zone",                   {3},       4, 1),
    # Originally critical-flavoured
    ("total {commodity} supply failure in {region} — all routes severed",         {4},       5, 1),
    ("catastrophic {commodity} processing collapse at {region} primary facility", {4},       5, 1),
    ("full {region} network isolation — no {commodity} flow possible",            {4},       5, 1),
    ("emergency: {commodity} cascade failure propagating beyond {region} boundary", {4},     5, 1),
    ("{commodity} network collapse in {region} — total throughput loss",          {4},       4, 1),
]

# Synonym substitution tables applied after template rendering
_SYNONYMS = {
    "routine": ["standard", "regular", "normal", "expected"],
    "minor": ["small", "slight", "limited", "marginal"],
    "temporary": ["transient", "short-lived", "brief", "interim"],
    "disruption": ["interruption", "disturbance", "irregularity", "deviation"],
    "severe": ["acute", "serious", "pronounced", "substantial"],
    "critical": ["vital", "essential", "key", "primary"],
    "halted": ["paused", "suspended", "stopped", "ceased"],
    "failure": ["breakdown", "malfunction", "outage", "collapse"],
    "congestion": ["bottleneck", "backlog", "saturation", "overload"],
    "compromised": ["impaired", "degraded", "weakened", "undermined"],
    "catastrophic": ["devastating", "ruinous", "disastrous", "total"],
    "emergency": ["urgent", "priority-one", "immediate", "red-alert"],
    "cascading": ["propagating", "spreading", "chain-reaction", "rippling"],
    "fluctuation": ["variation", "oscillation", "shift", "change"],
    "partially": ["partly", "incompletely", "to some extent", "in part"],
    "unexpected": ["unanticipated", "unforeseen", "unplanned", "surprise"],
    "significant": ["notable", "considerable", "marked", "appreciable"],
    "processing": ["handling", "treatment", "conversion", "refinement"],
}

# Heavily overlapping numeric ranges — designed so bucket thresholds
# in prepare.py cannot cleanly separate tiers.
SHORTFALL_RANGES = {
    0: (10, 2000),
    1: (50, 3000),
    2: (100, 4000),
    3: (200, 5000),
    4: (300, 6000),
}

CYCLE_RANGES = {
    0: (1, 18),
    1: (1, 22),
    2: (2, 26),
    3: (3, 30),
    4: (4, 30),
}

CASCADE_DEPTHS = {
    0: (0, 4),
    1: (0, 5),
    2: (0, 6),
    3: (1, 7),
    4: (1, 7),
}

# Overlapping categorical pools
IMPACT_DOWNSTREAM = {
    0: ["negligible", "minimal", "low"],
    1: ["minimal", "low", "moderate"],
    2: ["low", "moderate", "significant"],
    3: ["moderate", "significant", "severe"],
    4: ["significant", "severe", "critical", "catastrophic"],
}

MITIGATION_PHRASES = [
    "partial reroute via {alt_region}",
    "emergency stockpile activated at {alt_region}",
    "secondary supplier engaged in {alt_region}",
    "temporary substitution protocol initiated",
    "demand reduction order issued",
    "cross-sector reallocation from {alt_region}",
    "priority shipment diverted from {alt_region}",
    "backup channel opened through {alt_region}",
]

MITIGATION_EFFECTIVENESS = {
    0: ["high", "very-high", "moderate"],
    1: ["moderate", "high", "low"],
    2: ["low", "moderate", "very-low"],
    3: ["very-low", "low", "negligible"],
    4: ["negligible", "very-low", "low"],
}

CASCADE_RATES = {
    0: ["slow", "contained", "moderate"],
    1: ["slow", "contained", "moderate", "steady"],
    2: ["moderate", "steady", "slow"],
    3: ["steady", "rapid", "moderate", "accelerating"],
    4: ["rapid", "accelerating", "exponential", "uncontrolled", "steady"],
}

NOTE_PHRASES = [
    "Analyst note: monitoring frequency unchanged. Next review scheduled per standard cycle.",
    "Analyst note: no policy override triggered. Continuing baseline observation.",
    "Analyst note: cross-reference with adjacent corridor pending. No action required.",
    "Analyst note: historical pattern within expected variance band.",
    "Analyst note: upstream supplier notified. Awaiting acknowledgement.",
    "Analyst note: secondary data source corroboration in progress.",
    "Analyst note: regional liaison contacted. Status update expected next cycle.",
    "Analyst note: compliance check passed. No regulatory flag raised.",
    "Analyst note: inventory reconciliation scheduled. Preliminary figures nominal.",
    "Analyst note: sensor calibration verified. Readings within tolerance.",
]

VERDICT_PHRASES = {
    0: "Assessment: disruption classified as TIER 1 — MINOR. Standard monitoring.",
    1: "Assessment: disruption classified as TIER 2 — MODERATE. Enhanced monitoring advised.",
    2: "Assessment: disruption classified as TIER 3 — SIGNIFICANT. Intervention recommended.",
    3: "Assessment: disruption classified as TIER 4 — SEVERE. Immediate action required.",
    4: "Assessment: disruption classified as TIER 5 — CRITICAL. Emergency protocol activated.",
}


def _fmt(template: str, **kw) -> str:
    try:
        return template.format(**kw)
    except KeyError:
        return template


def _pick_narrative(rng: random.Random, tier_idx: int) -> str:
    """Weighted selection from the shared narrative pool."""
    weights = []
    for _, primary_tiers, w_pri, w_oth in _NARRATIVE_POOL:
        weights.append(w_pri if tier_idx in primary_tiers else w_oth)
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return _NARRATIVE_POOL[i][0]
    return _NARRATIVE_POOL[-1][0]


def _apply_synonyms(rng: random.Random, text: str) -> str:
    """Replace ~40% of synonym-eligible words with a random synonym."""
    for word, alts in _SYNONYMS.items():
        if word in text and rng.random() < 0.40:
            text = text.replace(word, rng.choice(alts), 1)
    return text


def generate_memo(rng: random.Random, tier_idx: int) -> dict:
    region = rng.choice(REGIONS)
    commodity = rng.choice(COMMODITIES)
    alt_region = rng.choice([r for r in REGIONS if r != region])

    sections = []

    # [ORBIT:VECTOR] — always present
    narrative_tpl = _pick_narrative(rng, tier_idx)
    narrative = _fmt(narrative_tpl, commodity=commodity, region=region)
    narrative = _apply_synonyms(rng, narrative)
    v_code = f"ORB-V{rng.randint(1, 7)}"
    sections.append(f"[ORBIT:VECTOR] Risk code {v_code}: {narrative}.")

    # [ORBIT:IMPACT] — always present
    s_lo, s_hi = SHORTFALL_RANGES[tier_idx]
    shortfall = rng.randint(s_lo, s_hi)
    noise_range = (s_hi - s_lo) // 3
    shortfall = max(1, shortfall + rng.randint(-noise_range, noise_range))

    c_lo, c_hi = CYCLE_RANGES[tier_idx]
    cycles = rng.randint(c_lo, c_hi)

    i_code = f"ORB-I{rng.randint(1, 7)}"
    downstream = rng.choice(IMPACT_DOWNSTREAM[tier_idx])
    sections.append(
        f"[ORBIT:IMPACT] Estimated shortfall {shortfall} units over {cycles} cycles. "
        f"{i_code} downstream effect: {downstream}."
    )

    # [ORBIT:MITIGATION] — ~50% of all memos, independent of tier
    if rng.random() < 0.50:
        mit_phrase = _fmt(rng.choice(MITIGATION_PHRASES), alt_region=alt_region)
        mit_eff = rng.choice(MITIGATION_EFFECTIVENESS[tier_idx])
        m_code = f"ORB-M{rng.randint(1, 5)}"
        sections.append(
            f"[ORBIT:MITIGATION] Mitigation: {mit_phrase}. {m_code} effectiveness: {mit_eff}."
        )

    # [ORBIT:CASCADE] — ~50% of all memos, independent of tier
    if rng.random() < 0.50:
        d_lo, d_hi = CASCADE_DEPTHS[tier_idx]
        depth = rng.randint(d_lo, d_hi)
        depth = max(0, depth + rng.choice([-1, 0, 0, 1]))
        rate = rng.choice(CASCADE_RATES[tier_idx])
        c_templates = [
            "Cascade depth: {depth} nodes. Propagation rate: {rate}.",
            "Downstream cascade: {depth} dependent nodes affected. Rate: {rate}.",
            "Impact propagation: {depth} node chain. Estimated rate: {rate}.",
        ]
        c_phrase = _fmt(rng.choice(c_templates), depth=depth, rate=rate)
        sections.append(f"[ORBIT:CASCADE] {c_phrase}")

    # [ORBIT:NOTE] — tier-irrelevant distractor, ~40% of memos
    if rng.random() < 0.40:
        sections.append(f"[ORBIT:NOTE] {rng.choice(NOTE_PHRASES)}")

    # [ORBIT:REF] — unique reference tag (stripped by prepare.py)
    ref_hash = f"REF-{rng.getrandbits(48):012x}"
    sections.append(f"[ORBIT:REF] {ref_hash}")

    # [ORBIT:VERDICT] — stripped by prepare.py
    sections.append(f"[ORBIT:VERDICT] {VERDICT_PHRASES[tier_idx]}")

    memo = " ".join(sections)
    return {
        "memo": memo,
        "region": region,
        "commodity_class": commodity,
        "label": LABELS[tier_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ORBIT risk memo dataset (v2)")
    parser.add_argument("--output", type=Path, default=Path("data.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=32_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    cum_weights = []
    s = 0
    for w in LABEL_WEIGHTS:
        s += w
        cum_weights.append(s)

    rows = []
    for i in range(args.size):
        r = rng.random()
        tier_idx = 0
        for j, cw in enumerate(cum_weights):
            if r <= cw:
                tier_idx = j
                break
        row = generate_memo(rng, tier_idx)
        row["id"] = i
        rows.append(row)

    memos = [r["memo"] for r in rows]
    dupes = len(memos) - len(set(memos))
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate memos found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "memo", "region", "commodity_class", "label"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    labels = Counter(r["label"] for r in rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Labels: {dict(sorted(labels.items()))}")
    print(f"  Regions: {dict(sorted(Counter(r['region'] for r in rows).items()))}")
    print(f"  Commodities: {dict(sorted(Counter(r['commodity_class'] for r in rows).items()))}")
    print(f"  Memo len: min={min(len(r['memo']) for r in rows)}, max={max(len(r['memo']) for r in rows)}")


if __name__ == "__main__":
    main()
