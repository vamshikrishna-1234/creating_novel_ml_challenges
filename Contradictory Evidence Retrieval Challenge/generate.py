"""
Synthetic dataset generator:
  Contradictory Evidence Retrieval and Verdict Challenge (RAG)

Each row contains a fictional scientific claim, a corpus of 8-15 evidence
passages (some supporting, some contradicting, some near-miss distractors,
some irrelevant distractors), and a structured verdict.

Near-miss distractors share the substance OR effect with the claim but not
both — they look relevant but should not count as evidence. Hedged passages
use uncertain language and are NOT counted as evidence.

Output: data.csv with columns [id, claim, passages, verdict]
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

SUBSTANCES = [
    "Compound-X7", "Reagent-KP3", "Catalyst-M9", "Polymer-ZF2",
    "Extract-BN4", "Isotope-QR1", "Alloy-DT8", "Solvent-HV5",
    "Enzyme-WL6", "Mineral-JC0", "Protein-AG3", "Aerosol-FE7",
    "Nanogel-UT9", "Fiber-RX2", "Colloid-PS4", "Resin-YK6",
    "Emulsion-OD1", "Hydrogel-BW3",
]

EFFECTS = [
    "thermal stability", "corrosion resistance", "tensile strength",
    "electrical conductivity", "biocompatibility", "optical clarity",
    "viscosity reduction", "catalytic efficiency", "radiation shielding",
    "moisture absorption", "flame retardancy", "elasticity",
    "antimicrobial activity", "photodegradation rate", "surface adhesion",
    "thermal conductivity", "dielectric strength", "impact resistance",
]

ENVIRONMENTS = [
    "high-pressure chamber", "aqueous solution at pH 7.4",
    "vacuum conditions", "ambient atmosphere", "cryogenic storage",
    "UV exposure chamber", "saline immersion", "elevated humidity",
    "inert gas environment", "thermal cycling rig",
    "alkaline bath at pH 12", "accelerated weathering chamber",
]

INSTITUTIONS = [
    "Vellore Applied Sciences Lab", "Kessler-Brandt Institute",
    "Nakamura Materials Centre", "Okafor Regulatory Board",
    "Petrov Environmental Agency", "Lindgren Testing Facility",
    "Torres Analytical Group", "Mbeki Standards Authority",
    "Johansson Research Division", "Alvarez Compliance Unit",
    "Reyes Certification Body", "Novak Quality Bureau",
    "Chandra Polymer Research Group", "Eriksson Nanotech Lab",
]

YEARS = list(range(2017, 2026))

METHODS = [
    "spectroscopic analysis", "tensile testing", "calorimetry",
    "electron microscopy", "chromatographic separation",
    "impedance measurement", "rheological profiling",
    "X-ray diffraction", "mass spectrometry", "thermal gravimetric analysis",
    "atomic force microscopy", "dynamic mechanical analysis",
]

# ---------------------------------------------------------------------------
# Passage templates
# ---------------------------------------------------------------------------

SUPPORT_TEMPLATES = [
    "A {year} study by {inst} confirmed that {substance} exhibits improved {effect} under {env} conditions. Method: {method}.",
    "Testing at {inst} ({year}) demonstrated {substance} significantly enhances {effect} when applied in {env}. Analysis via {method}.",
    "Results from {inst} ({year}) indicate {substance} positively influences {effect} in {env} settings, validated by {method}.",
    "{inst} reported ({year}) that {substance} consistently improves {effect} across trials conducted in {env}. Measured using {method}.",
    "Experimental data from {inst} ({year}) supports the claim that {substance} increases {effect} under {env} conditions ({method}).",
    "According to {inst} ({year}), {substance} showed a statistically significant improvement in {effect} during {env} trials ({method}).",
    "A controlled experiment at {inst} ({year}) using {method} verified that {substance} enhances {effect} in {env}.",
    "Peer-reviewed findings from {inst} ({year}) confirm a positive correlation between {substance} application and {effect} in {env} ({method}).",
]

CONTRADICT_TEMPLATES = [
    "A {year} investigation by {inst} found no measurable improvement in {effect} from {substance} under {env} conditions ({method}).",
    "{inst} ({year}) reported that {substance} had negligible impact on {effect} in {env} environments. Analysis: {method}.",
    "Contrary to expectations, {inst} ({year}) observed that {substance} may reduce {effect} under {env} conditions ({method}).",
    "Testing by {inst} ({year}) showed {substance} does not significantly alter {effect} in {env} settings ({method}).",
    "Data from {inst} ({year}) contradicts the hypothesis that {substance} improves {effect} under {env} conditions ({method}).",
    "{inst} ({year}) concluded that {substance} has no reliable effect on {effect} when tested in {env} ({method}).",
    "A replication study at {inst} ({year}) failed to reproduce any {effect} improvement from {substance} in {env} ({method}).",
    "Analysis by {inst} ({year}) using {method} showed {substance} actually degrades {effect} under {env} conditions.",
]

# Near-miss: same substance, different effect (looks relevant but isn't)
NEARMISS_SUBSTANCE_TEMPLATES = [
    "{inst} ({year}) studied {substance} effects on {alt_effect} in {env} using {method}. Results were inconclusive.",
    "A {year} report from {inst} examined {substance} for {alt_effect} improvement under {env} conditions ({method}).",
    "Research at {inst} ({year}) focused on whether {substance} influences {alt_effect} in {env} settings ({method}).",
    "{inst} ({year}) tested {substance} for potential {alt_effect} enhancement in {env}. No clear trend was found ({method}).",
]

# Near-miss: same effect, different substance
NEARMISS_EFFECT_TEMPLATES = [
    "{inst} ({year}) evaluated {alt_substance} for {effect} improvement under {env} conditions ({method}).",
    "A {year} study at {inst} compared {alt_substance} performance on {effect} in {env} ({method}).",
    "Research from {inst} ({year}) explored {alt_substance} as an alternative for enhancing {effect} in {env} ({method}).",
    "{inst} ({year}) reported that {alt_substance} shows promise for {effect} under {env} conditions ({method}).",
]

# Hedged/uncertain passages — mention the right substance+effect but are
# too uncertain to count as evidence
HEDGED_TEMPLATES = [
    "Preliminary observations at {inst} ({year}) suggest {substance} might influence {effect} in {env}, but sample sizes were too small for conclusions ({method}).",
    "{inst} ({year}) noted anecdotal reports of {substance} affecting {effect} in {env}, though no formal testing was conducted.",
    "An unpublished pilot at {inst} ({year}) hinted at a possible link between {substance} and {effect} in {env}, but results were not statistically significant ({method}).",
    "{inst} ({year}) flagged {substance} as a candidate for {effect} research in {env} but did not conduct definitive experiments.",
]

# Pure distractor: different substance AND different effect
DISTRACTOR_TEMPLATES = [
    "{inst} ({year}) published findings on {alt_substance} and its role in {alt_effect} under {env} conditions ({method}).",
    "A {year} report from {inst} examined {alt_effect} in {env} environments using {method}, focusing on {alt_substance}.",
    "Research at {inst} ({year}) explored the relationship between {alt_substance} and {alt_effect} via {method}.",
    "{inst} ({year}) conducted a review of {alt_effect} measurement techniques using {method} in {env} settings.",
    "A {year} meta-analysis by {inst} surveyed {alt_substance} applications for {alt_effect} improvement ({method}).",
    "{inst} ({year}) benchmarked {method} protocols for evaluating {alt_effect} in {env} conditions.",
]

CLAIM_TEMPLATES = [
    "{substance} improves {effect} under {env} conditions.",
    "{substance} enhances {effect} when applied in {env} settings.",
    "Application of {substance} leads to increased {effect} in {env} environments.",
    "{substance} has a positive impact on {effect} under {env} conditions.",
    "Under {env} conditions, {substance} significantly boosts {effect}.",
]

# ---------------------------------------------------------------------------
# Verdict derivation
# ---------------------------------------------------------------------------

def _derive_confidence(n_support: int, n_contradict: int) -> str:
    total = n_support + n_contradict
    if total == 0:
        return "low"
    ratio = abs(n_support - n_contradict) / total
    if ratio >= 0.7:
        return "high"
    elif ratio >= 0.35:
        return "medium"
    else:
        return "low"


def _derive_stance(n_support: int, n_contradict: int) -> str:
    if n_support == 0 and n_contradict == 0:
        return "insufficient"
    if n_support > n_contradict:
        return "support"
    elif n_contradict > n_support:
        return "contradict"
    else:
        return "insufficient"


# ---------------------------------------------------------------------------
# Passage builder helpers
# ---------------------------------------------------------------------------

def _pick_other(pool, exclude, rng):
    choices = [x for x in pool if x != exclude]
    return rng.choice(choices)


def _fill_common(rng, env):
    return {
        "inst": rng.choice(INSTITUTIONS),
        "year": rng.choice(YEARS),
        "method": rng.choice(METHODS),
        "env": env,
    }


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def generate_row(rng: random.Random, row_id: int) -> dict:
    substance = rng.choice(SUBSTANCES)
    effect = rng.choice(EFFECTS)
    env = rng.choice(ENVIRONMENTS)

    claim = rng.choice(CLAIM_TEMPLATES).format(
        substance=substance, effect=effect, env=env
    )

    # --- Decide composition ---
    # We want balanced stances. Use a profile-based approach.
    profile = rng.choice([
        "strong_support", "strong_contradict", "mixed_lean_support",
        "mixed_lean_contradict", "balanced", "insufficient",
        "mostly_support", "mostly_contradict",
    ])

    if profile == "strong_support":
        n_support = rng.randint(4, 6)
        n_contradict = rng.randint(0, 1)
    elif profile == "strong_contradict":
        n_support = rng.randint(0, 1)
        n_contradict = rng.randint(4, 6)
    elif profile == "mixed_lean_support":
        n_contradict = rng.randint(1, 3)
        n_support = n_contradict + rng.randint(1, 2)
    elif profile == "mixed_lean_contradict":
        n_support = rng.randint(1, 3)
        n_contradict = n_support + rng.randint(1, 2)
    elif profile == "balanced":
        n_support = rng.randint(2, 4)
        n_contradict = n_support
    elif profile == "insufficient":
        n_support = 0
        n_contradict = 0
    elif profile == "mostly_support":
        n_support = rng.randint(3, 5)
        n_contradict = rng.randint(1, 2)
    else:  # mostly_contradict
        n_contradict = rng.randint(3, 5)
        n_support = rng.randint(1, 2)

    # Near-miss distractors (hard negatives)
    n_nearmiss = rng.randint(1, 4)
    # Hedged passages (look relevant but uncertain)
    n_hedged = rng.randint(0, 2)
    # Pure distractors
    n_distractor = rng.randint(1, 4)

    total = n_support + n_contradict + n_nearmiss + n_hedged + n_distractor

    # Clamp to 8-15
    while total < 8:
        n_distractor += 1
        total += 1
    while total > 15:
        if n_distractor > 1:
            n_distractor -= 1
        elif n_nearmiss > 1:
            n_nearmiss -= 1
        elif n_hedged > 0:
            n_hedged -= 1
        else:
            break
        total -= 1

    passages = []
    roles = []  # "support", "contradict", "nearmiss", "hedged", "distractor"

    for _ in range(n_support):
        params = _fill_common(rng, env)
        params["substance"] = substance
        params["effect"] = effect
        text = rng.choice(SUPPORT_TEMPLATES).format(**params)
        passages.append(text)
        roles.append("support")

    for _ in range(n_contradict):
        params = _fill_common(rng, env)
        params["substance"] = substance
        params["effect"] = effect
        text = rng.choice(CONTRADICT_TEMPLATES).format(**params)
        passages.append(text)
        roles.append("contradict")

    for _ in range(n_nearmiss):
        if rng.random() < 0.5:
            params = _fill_common(rng, env)
            params["substance"] = substance
            params["alt_effect"] = _pick_other(EFFECTS, effect, rng)
            text = rng.choice(NEARMISS_SUBSTANCE_TEMPLATES).format(**params)
        else:
            params = _fill_common(rng, env)
            params["alt_substance"] = _pick_other(SUBSTANCES, substance, rng)
            params["effect"] = effect
            text = rng.choice(NEARMISS_EFFECT_TEMPLATES).format(**params)
        passages.append(text)
        roles.append("nearmiss")

    for _ in range(n_hedged):
        params = _fill_common(rng, env)
        params["substance"] = substance
        params["effect"] = effect
        text = rng.choice(HEDGED_TEMPLATES).format(**params)
        passages.append(text)
        roles.append("hedged")

    for _ in range(n_distractor):
        params = _fill_common(rng, env)
        params["alt_substance"] = _pick_other(SUBSTANCES, substance, rng)
        params["alt_effect"] = _pick_other(EFFECTS, effect, rng)
        text = rng.choice(DISTRACTOR_TEMPLATES).format(**params)
        passages.append(text)
        roles.append("distractor")

    # Shuffle
    combined = list(zip(passages, roles))
    rng.shuffle(combined)
    passages, roles = zip(*combined)
    passages = list(passages)
    roles = list(roles)

    # Evidence IDs: only support + contradict count (1-indexed)
    evidence_ids = sorted([
        i + 1 for i, role in enumerate(roles)
        if role in ("support", "contradict")
    ])

    stance = _derive_stance(n_support, n_contradict)
    confidence = _derive_confidence(n_support, n_contradict)

    evidence_str = ",".join(str(eid) for eid in evidence_ids) if evidence_ids else "none"
    verdict = f"STANCE:{stance} | EVIDENCE_IDS:{evidence_str} | CONFIDENCE:{confidence}"

    passages_text = "\n".join(f"[P{i+1}] {p}" for i, p in enumerate(passages))

    return {
        "id": row_id,
        "claim": claim,
        "passages": passages_text,
        "verdict": verdict,
        "_roles": roles,
        "_n_support": n_support,
        "_n_contradict": n_contradict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=20_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [generate_row(rng, i) for i in range(args.size)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "claim", "passages", "verdict"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ["id", "claim", "passages", "verdict"]})

    stances = Counter(r["verdict"].split(" | ")[0].split(":")[1] for r in rows)
    confs = Counter(r["verdict"].split(" | ")[2].split(":")[1] for r in rows)
    n_ev = [
        0 if r["verdict"].split(" | ")[1].split(":")[1] == "none"
        else len(r["verdict"].split(" | ")[1].split(":")[1].split(","))
        for r in rows
    ]

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Stances: {dict(sorted(stances.items()))}")
    print(f"  Confidences: {dict(sorted(confs.items()))}")
    print(f"  Evidence IDs per row: min={min(n_ev)}, max={max(n_ev)}, mean={sum(n_ev)/len(n_ev):.1f}")


if __name__ == "__main__":
    main()
