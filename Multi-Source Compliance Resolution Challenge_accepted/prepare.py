"""
Prepare script: transforms raw data.csv into public/ and private/ splits.

Deterministic (fixed random_state + hashlib for row-level operations).

Obfuscation pipeline (applied to every row's input text):
  1. Randomly reorder codebook rules (~60% of rows) — forces parsing, not positional memorization
  2. Randomly swap inspector fragment order (~40% of rows)
  3. Inject 1-2 decoy codebook rules (~50% of rows) that reference violation types
     not present in any fragment — tests whether solver ignores irrelevant rules
  4. Paraphrase some rule text (~30% of rules) using synonym substitution
  5. Add noise words to fragment notes (~25% of fragments)

Split: stratified 80/20 by severity field extracted from verdict.
"""

from pathlib import Path
import re
import hashlib
import random as _rnd

import pandas as pd
from sklearn.model_selection import train_test_split


VIOLATION_TYPES = [
    "leak", "emission_excess", "waste_overflow", "containment_breach",
    "filter_failure", "thermal_deviation", "pressure_anomaly",
    "corrosion_detected", "ventilation_fault", "runoff_contamination",
    "noise_exceedance", "particulate_spike", "chemical_spill",
    "structural_crack", "seal_degradation",
]

DECOY_RULE_TEMPLATES = [
    "If any inspector reports '{vtype}', include it as a confirmed violation.",
    "Include '{vtype}' only if a majority of inspectors report it.",
    "If '{vtype}' is a confirmed violation, severity must be at least 'moderate'.",
    "If '{vtype}' is a confirmed violation, severity must be at least 'elevated'.",
]

RULE_SYNONYMS = {
    "inspector reports": ["inspector flags", "inspector identifies", "inspector notes"],
    "include it as a confirmed violation": ["mark it as confirmed", "classify it as a violation", "record it as a confirmed finding"],
    "escalate severity": ["increase severity", "raise severity", "bump severity"],
    "set action to": ["assign action as", "change action to", "update action to"],
    "set penalty to": ["assign penalty as", "change penalty to", "update penalty to"],
    "set severity to": ["assign severity as", "change severity to", "update severity to"],
    "majority of inspectors": ["more than half of inspectors", "most inspectors", "a majority of the inspection team"],
    "If no violations remain": ["If zero violations remain", "If the violation list is empty", "When no violations are left"],
    "Penalty cannot exceed": ["Maximum penalty is", "Penalty is capped at", "Penalty must not go above"],
}

NOISE_PHRASES = [
    "Ambient conditions were within normal parameters.",
    "Equipment calibration was verified prior to inspection.",
    "Weather conditions: clear, no impact on readings.",
    "Standard protocol followed throughout.",
    "No additional personnel present during inspection.",
    "Documentation cross-referenced with facility logs.",
    "Measurement instruments last serviced within 30 days.",
]


def _det_rng(row_id: int, salt: str) -> _rnd.Random:
    seed = int(hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest(), 16) % (2**32)
    return _rnd.Random(seed)


def _should_modify(row_id: int, salt: str, threshold: float) -> bool:
    h = hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < threshold


def _extract_severity(verdict: str) -> str:
    m = re.search(r'SEVERITY:(\w+)', verdict)
    return m.group(1) if m else "unknown"


def _shuffle_codebook_rules(text: str, row_id: int) -> str:
    """Shuffle the order of rules within the codebook section."""
    codebook_match = re.search(r'(=== Compliance Codebook ===\n)(.*)', text, re.DOTALL)
    if not codebook_match:
        return text
    header = codebook_match.group(1)
    rules_text = codebook_match.group(2)
    rules = [r.strip() for r in rules_text.strip().split('\n') if r.strip()]
    rng = _det_rng(row_id, "shuffle_rules")
    rng.shuffle(rules)
    renumbered = [re.sub(r'^Rule \d+:', f'Rule {i+1}:', r) for i, r in enumerate(rules)]
    new_codebook = header + "\n".join(renumbered)
    return text[:codebook_match.start()] + new_codebook


def _shuffle_fragments(text: str, row_id: int) -> str:
    """Shuffle the order of inspector fragments."""
    parts = re.split(r'(?=--- Fragment \d+)', text)
    pre = parts[0] if not parts[0].startswith('--- Fragment') else ''
    frags = [p for p in parts if p.startswith('--- Fragment')]
    rest_idx = text.find('=== Compliance Codebook ===')
    if rest_idx == -1:
        return text
    rest = text[rest_idx:]
    rng = _det_rng(row_id, "shuffle_frags")
    rng.shuffle(frags)
    renumbered = []
    for i, f in enumerate(frags):
        renumbered.append(re.sub(r'--- Fragment \d+', f'--- Fragment {i+1}', f, count=1))
    return pre + "".join(renumbered) + "\n\n" + rest


def _inject_decoy_rules(text: str, row_id: int) -> str:
    """Add 1-2 decoy rules referencing violation types not in any fragment."""
    rng = _det_rng(row_id, "decoy")
    mentioned = set(re.findall(r'\b(' + '|'.join(VIOLATION_TYPES) + r')\b', text.split('=== Compliance Codebook ===')[0]))
    available = [v for v in VIOLATION_TYPES if v not in mentioned]
    if not available:
        return text
    n_decoys = rng.randint(1, 2)
    decoys = rng.sample(available, min(n_decoys, len(available)))

    codebook_match = re.search(r'(=== Compliance Codebook ===\n)(.*)', text, re.DOTALL)
    if not codebook_match:
        return text
    header = codebook_match.group(1)
    rules_text = codebook_match.group(2)
    rules = [r.strip() for r in rules_text.strip().split('\n') if r.strip()]

    existing_count = len(rules)
    for vtype in decoys:
        template = rng.choice(DECOY_RULE_TEMPLATES).format(vtype=vtype)
        rules.append(f"Rule {existing_count + 1}: {template}")
        existing_count += 1

    rng.shuffle(rules)
    renumbered = [re.sub(r'^Rule \d+:', f'Rule {i+1}:', r) for i, r in enumerate(rules)]
    new_codebook = header + "\n".join(renumbered)
    return text[:codebook_match.start()] + new_codebook


def _paraphrase_rules(text: str, row_id: int) -> str:
    """Apply synonym substitution to ~30% of rule phrases."""
    rng = _det_rng(row_id, "paraphrase")
    for phrase, alts in RULE_SYNONYMS.items():
        if phrase in text and rng.random() < 0.30:
            text = text.replace(phrase, rng.choice(alts), 1)
    return text


def _add_noise_to_fragments(text: str, row_id: int) -> str:
    """Inject noise phrases into ~25% of fragments."""
    rng = _det_rng(row_id, "noise")
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if line.startswith('--- Fragment') and rng.random() < 0.25:
            new_lines.append(rng.choice(NOISE_PHRASES))
    return '\n'.join(new_lines)


def _obfuscate(input_text: str, row_id: int) -> str:
    text = input_text

    if _should_modify(row_id, 'shuffle_rules', 0.60):
        text = _shuffle_codebook_rules(text, row_id)

    if _should_modify(row_id, 'shuffle_frags', 0.40):
        text = _shuffle_fragments(text, row_id)

    if _should_modify(row_id, 'decoy', 0.50):
        text = _inject_decoy_rules(text, row_id)

    text = _paraphrase_rules(text, row_id)
    text = _add_noise_to_fragments(text, row_id)

    return text


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "input", "verdict"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    df = df.copy()
    df["input"] = df.apply(lambda r: _obfuscate(r["input"], int(r["id"])), axis=1)
    df["severity"] = df["verdict"].apply(_extract_severity)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["severity"]
    )

    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(test_ids)

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_df[["id", "input", "verdict"]].to_csv(public / "train.csv", index=False)
    test_df[["id", "input"]].to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    sample["verdict"] = "FACILITY:XXX-0000 | VIOLATIONS:none | COUNT:0 | SEVERITY:negligible | ACTION:no_action | PENALTY:none"
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "verdict"]].to_csv(private / "answers.csv", index=False)
