"""
Prepare script: transforms raw data.csv into public/ and private/ splits.

Deterministic (fixed random_state + hashlib for row-level operations).

Obfuscation pipeline (applied to every row):
  1. Paraphrase passage text (~35% of passages) via synonym substitution
  2. Inject 1-3 "hedged near-miss" decoy passages (~40% of rows) that
     mention the claim's substance+effect but use uncertain language
  3. Truncate institution names to abbreviations (~50% of passages)
  4. Remove method citations from ~30% of passages
  5. Shuffle passage order (always) and reassign passage IDs
  6. Redact year from ~25% of passages
  7. Inject subtle cross-reference noise (~20% of passages get a
     parenthetical note referencing another passage ID)

Split: stratified 80/20 by stance field extracted from verdict.
"""

from pathlib import Path
import re
import hashlib
import random as _rnd

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Synonym pools for paraphrasing
# ---------------------------------------------------------------------------

PARAPHRASE_MAP = {
    "confirmed": ["verified", "established", "validated", "demonstrated"],
    "demonstrated": ["showed", "revealed", "established", "indicated"],
    "significantly": ["notably", "considerably", "substantially", "markedly"],
    "improved": ["enhanced", "increased", "boosted", "elevated"],
    "enhances": ["boosts", "augments", "strengthens", "amplifies"],
    "positively influences": ["favorably affects", "beneficially impacts", "constructively alters"],
    "no measurable improvement": ["no detectable enhancement", "no observable gain", "no quantifiable benefit"],
    "negligible impact": ["minimal effect", "trivial influence", "insignificant bearing"],
    "contradicts": ["refutes", "disputes", "challenges", "undermines"],
    "no reliable effect": ["no consistent impact", "no dependable influence", "no reproducible effect"],
    "statistically significant": ["statistically meaningful", "statistically notable", "statistically robust"],
    "consistently improves": ["reliably enhances", "steadily boosts", "dependably increases"],
    "may reduce": ["could diminish", "might decrease", "potentially lowers"],
    "does not significantly alter": ["fails to meaningfully change", "does not appreciably modify"],
    "failed to reproduce": ["could not replicate", "was unable to confirm", "did not reproduce"],
    "actually degrades": ["in fact worsens", "effectively diminishes", "measurably reduces"],
    "inconclusive": ["ambiguous", "indeterminate", "uncertain", "unclear"],
    "showed a statistically significant improvement": [
        "revealed a meaningful gain", "indicated a notable enhancement",
    ],
}

HEDGED_DECOY_TEMPLATES = [
    "Preliminary observations at {inst} ({year}) suggest {substance} might influence {effect} in {env}, but sample sizes were insufficient for conclusions ({method}).",
    "An unpublished pilot at {inst} ({year}) hinted at a possible link between {substance} and {effect} in {env}, but results were not statistically significant.",
    "{inst} ({year}) flagged {substance} as a candidate for {effect} research in {env} but did not conduct definitive experiments.",
    "Anecdotal reports compiled by {inst} ({year}) mention {substance} in the context of {effect} under {env} conditions, though no formal study was completed.",
]

INSTITUTIONS_SHORT = [
    "VASL", "KBI", "NMC", "ORB", "PEA", "LTF",
    "TAG", "MSA", "JRD", "ACU", "RCB", "NQB", "CPRG", "ENL",
]

INSTITUTIONS_FULL = [
    "Vellore Applied Sciences Lab", "Kessler-Brandt Institute",
    "Nakamura Materials Centre", "Okafor Regulatory Board",
    "Petrov Environmental Agency", "Lindgren Testing Facility",
    "Torres Analytical Group", "Mbeki Standards Authority",
    "Johansson Research Division", "Alvarez Compliance Unit",
    "Reyes Certification Body", "Novak Quality Bureau",
    "Chandra Polymer Research Group", "Eriksson Nanotech Lab",
]

INST_MAP = dict(zip(INSTITUTIONS_FULL, INSTITUTIONS_SHORT))

METHODS = [
    "spectroscopic analysis", "tensile testing", "calorimetry",
    "electron microscopy", "chromatographic separation",
    "impedance measurement", "rheological profiling",
    "X-ray diffraction", "mass spectrometry", "thermal gravimetric analysis",
    "atomic force microscopy", "dynamic mechanical analysis",
]

ENVIRONMENTS = [
    "high-pressure chamber", "aqueous solution at pH 7.4",
    "vacuum conditions", "ambient atmosphere", "cryogenic storage",
    "UV exposure chamber", "saline immersion", "elevated humidity",
    "inert gas environment", "thermal cycling rig",
    "alkaline bath at pH 12", "accelerated weathering chamber",
]


# ---------------------------------------------------------------------------
# Deterministic RNG helpers
# ---------------------------------------------------------------------------

def _det_rng(row_id: int, salt: str) -> _rnd.Random:
    seed = int(hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest(), 16) % (2**32)
    return _rnd.Random(seed)


def _should_modify(row_id: int, salt: str, threshold: float) -> bool:
    h = hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < threshold


# ---------------------------------------------------------------------------
# Obfuscation functions
# ---------------------------------------------------------------------------

def _paraphrase_passage(text: str, rng: _rnd.Random) -> str:
    for phrase, alts in PARAPHRASE_MAP.items():
        if phrase in text and rng.random() < 0.40:
            text = text.replace(phrase, rng.choice(alts), 1)
    return text


def _abbreviate_institutions(text: str, rng: _rnd.Random) -> str:
    for full, short in INST_MAP.items():
        if full in text and rng.random() < 0.50:
            text = text.replace(full, short, 1)
    return text


def _remove_method(text: str, rng: _rnd.Random) -> str:
    if rng.random() < 0.30:
        for method in METHODS:
            text = text.replace(f"Method: {method}.", "", 1)
            text = text.replace(f"Analysis: {method}.", "", 1)
            text = text.replace(f"Analysis via {method}.", "", 1)
            text = text.replace(f"Measured using {method}.", "", 1)
            text = text.replace(f"({method})", "", 1)
            text = text.replace(f"({method}).", ".", 1)
            text = text.replace(f"using {method} ", "", 1)
        text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def _redact_year(text: str, rng: _rnd.Random) -> str:
    if rng.random() < 0.25:
        text = re.sub(r'\b(20\d{2})\b', '[YEAR]', text)
    return text


def _add_cross_reference(text: str, passage_count: int, current_idx: int, rng: _rnd.Random) -> str:
    if rng.random() < 0.20 and passage_count > 3:
        ref_id = rng.randint(1, passage_count)
        while ref_id == current_idx + 1:
            ref_id = rng.randint(1, passage_count)
        text = text.rstrip('.') + f" (cf. P{ref_id})."
    return text


def _extract_claim_parts(claim: str):
    """Try to extract substance, effect, environment from the claim text."""
    import re as _re
    substance = None
    effect = None
    env = None

    sub_pattern = r'(Compound-X7|Reagent-KP3|Catalyst-M9|Polymer-ZF2|Extract-BN4|Isotope-QR1|Alloy-DT8|Solvent-HV5|Enzyme-WL6|Mineral-JC0|Protein-AG3|Aerosol-FE7|Nanogel-UT9|Fiber-RX2|Colloid-PS4|Resin-YK6|Emulsion-OD1|Hydrogel-BW3)'
    m = _re.search(sub_pattern, claim)
    if m:
        substance = m.group(1)

    for eff in [
        "thermal stability", "corrosion resistance", "tensile strength",
        "electrical conductivity", "biocompatibility", "optical clarity",
        "viscosity reduction", "catalytic efficiency", "radiation shielding",
        "moisture absorption", "flame retardancy", "elasticity",
        "antimicrobial activity", "photodegradation rate", "surface adhesion",
        "thermal conductivity", "dielectric strength", "impact resistance",
    ]:
        if eff in claim:
            effect = eff
            break

    for e in ENVIRONMENTS:
        if e in claim:
            env = e
            break

    return substance, effect, env


def _inject_hedged_decoys(passages_text: str, claim: str, row_id: int) -> str:
    """Add 1-2 hedged decoy passages that mention the right substance+effect
    but use uncertain language. These should NOT be counted as evidence."""
    rng = _det_rng(row_id, "hedged_decoy")
    substance, effect, env = _extract_claim_parts(claim)
    if not substance or not effect:
        return passages_text

    lines = passages_text.strip().split('\n')
    n_existing = len(lines)
    n_decoys = rng.randint(1, 2)

    for _ in range(n_decoys):
        if n_existing >= 15:
            break
        template = rng.choice(HEDGED_DECOY_TEMPLATES)
        inst = rng.choice(INSTITUTIONS_FULL)
        year = rng.choice(list(range(2017, 2026)))
        method = rng.choice(METHODS)
        text = template.format(
            substance=substance, effect=effect,
            env=env or "standard conditions",
            inst=inst, year=year, method=method,
        )
        n_existing += 1
        lines.append(f"[P{n_existing}] {text}")

    return '\n'.join(lines)


def _obfuscate_row(claim: str, passages_text: str, row_id: int) -> str:
    rng = _det_rng(row_id, "obfuscate")

    lines = passages_text.strip().split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        m = re.match(r'\[P\d+\]\s*(.*)', line)
        if not m:
            new_lines.append(line)
            continue
        ptext = m.group(1)

        prng = _det_rng(row_id * 1000 + i, "passage")
        if prng.random() < 0.35:
            ptext = _paraphrase_passage(ptext, prng)
        ptext = _abbreviate_institutions(ptext, prng)
        ptext = _remove_method(ptext, prng)
        ptext = _redact_year(ptext, prng)
        new_lines.append(ptext)

    # Shuffle passage order
    rng.shuffle(new_lines)

    # Add cross-references after shuffle
    final_lines = []
    for i, line in enumerate(new_lines):
        crng = _det_rng(row_id * 1000 + i, "crossref")
        line = _add_cross_reference(line, len(new_lines), i, crng)
        final_lines.append(f"[P{i+1}] {line}")

    result = '\n'.join(final_lines)

    # Inject hedged decoys
    if _should_modify(row_id, 'hedged_decoy', 0.40):
        result = _inject_hedged_decoys(result, claim, row_id)

    return result


def _update_verdict_evidence_ids(verdict: str, old_to_new: dict) -> str:
    """Remap evidence IDs in the verdict after passage shuffling.
    Since we shuffle and reassign, we need to track the mapping."""
    # This is handled differently — we re-derive evidence IDs from the
    # shuffled passage order, so this function is not needed.
    return verdict


# ---------------------------------------------------------------------------
# Main prepare function
# ---------------------------------------------------------------------------

def _extract_stance(verdict: str) -> str:
    m = re.search(r'STANCE:(\w+)', verdict)
    return m.group(1) if m else "unknown"


def prepare(raw: Path = Path("."), public: Path = Path("pub"), private: Path = Path("priv")) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "claim", "passages", "verdict"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    df = df.copy()

    # The verdict's EVIDENCE_IDS reference the ORIGINAL passage order.
    # After obfuscation (which shuffles passages), we need to remap them.
    # Strategy: we track which original passage index maps to which new index.

    new_passages = []
    new_verdicts = []

    for _, row in df.iterrows():
        row_id = int(row["id"])
        claim = row["claim"]
        passages_text = row["passages"]
        verdict = row["verdict"]

        # Parse original passages
        orig_lines = passages_text.strip().split('\n')
        orig_texts = []
        for line in orig_lines:
            m = re.match(r'\[P\d+\]\s*(.*)', line)
            if m:
                orig_texts.append(m.group(1))
            else:
                orig_texts.append(line)

        # Parse original evidence IDs
        parts = dict(p.strip().split(":", 1) for p in verdict.split(" | "))
        orig_eids = parts.get("EVIDENCE_IDS", "none")
        if orig_eids == "none":
            orig_eid_set = set()
        else:
            orig_eid_set = set(int(x) for x in orig_eids.split(","))

        # Mark which original indices (0-based) are evidence
        is_evidence = [((i + 1) in orig_eid_set) for i in range(len(orig_texts))]

        # Apply per-passage obfuscation (without shuffle first)
        rng = _det_rng(row_id, "obfuscate")
        obfuscated_texts = []
        for i, ptext in enumerate(orig_texts):
            prng = _det_rng(row_id * 1000 + i, "passage")
            if prng.random() < 0.35:
                ptext = _paraphrase_passage(ptext, prng)
            ptext = _abbreviate_institutions(ptext, prng)
            ptext = _remove_method(ptext, prng)
            ptext = _redact_year(ptext, prng)
            obfuscated_texts.append(ptext)

        # Shuffle: create index mapping
        indices = list(range(len(obfuscated_texts)))
        rng.shuffle(indices)

        shuffled_texts = [obfuscated_texts[idx] for idx in indices]
        shuffled_is_evidence = [is_evidence[idx] for idx in indices]

        # Add cross-references
        final_lines = []
        for i, text in enumerate(shuffled_texts):
            crng = _det_rng(row_id * 1000 + i, "crossref")
            text = _add_cross_reference(text, len(shuffled_texts), i, crng)
            final_lines.append(f"[P{i+1}] {text}")

        # Inject hedged decoys (these are NOT evidence)
        if _should_modify(row_id, 'hedged_decoy', 0.40):
            substance, effect, env = _extract_claim_parts(claim)
            if substance and effect:
                drng = _det_rng(row_id, "hedged_decoy")
                n_decoys = drng.randint(1, 2)
                for _ in range(n_decoys):
                    if len(final_lines) >= 17:
                        break
                    template = drng.choice(HEDGED_DECOY_TEMPLATES)
                    inst = drng.choice(INSTITUTIONS_FULL)
                    year = drng.choice(list(range(2017, 2026)))
                    method = drng.choice(METHODS)
                    text = template.format(
                        substance=substance, effect=effect,
                        env=env or "standard conditions",
                        inst=inst, year=year, method=method,
                    )
                    final_lines.append(f"[P{len(final_lines)+1}] {text}")
                    shuffled_is_evidence.append(False)

        # Compute new evidence IDs
        new_eids = sorted([
            i + 1 for i, ev in enumerate(shuffled_is_evidence) if ev
        ])
        new_eid_str = ",".join(str(e) for e in new_eids) if new_eids else "none"

        new_verdict = f"STANCE:{parts['STANCE']} | EVIDENCE_IDS:{new_eid_str} | CONFIDENCE:{parts['CONFIDENCE']}"

        new_passages.append('\n'.join(final_lines))
        new_verdicts.append(new_verdict)

    df["passages"] = new_passages
    df["verdict"] = new_verdicts

    # Stratified split
    df["_stance"] = df["verdict"].apply(_extract_stance)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["_stance"]
    )

    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df[["id", "claim", "passages", "verdict"]].to_csv(public / "train.csv", index=False)
    test_df[["id", "claim", "passages"]].to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    sample["verdict"] = "STANCE:support | EVIDENCE_IDS:1,2,3 | CONFIDENCE:high"
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "verdict"]].to_csv(private / "answers.csv", index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train stances: {dict(train_df['_stance'].value_counts())}")
    print(f"Test stances:  {dict(test_df['_stance'].value_counts())}")


if __name__ == "__main__":
    prepare()
