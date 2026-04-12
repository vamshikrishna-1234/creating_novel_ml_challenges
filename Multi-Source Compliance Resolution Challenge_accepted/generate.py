"""
Synthetic dataset generator:
  Multi-Source Compliance Resolution Challenge (Seq2Seq)

Domain: Fictional environmental compliance inspections. Each row represents
a facility visit where 2-4 inspectors filed independent fragments. A
per-row codebook (subset of ~20 rules) governs how to reconcile contradictions,
handle missing data, and derive the 6-field structured verdict.

Task: Given inspector fragments + codebook rules, produce the exact verdict string.

Output: data.csv with columns [id, fragments, codebook, verdict]

Usage:
    python generate.py [--output data.csv] [--seed 42] [--size 20000]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------------------
# Fictional domain vocabulary
# ---------------------------------------------------------------------------
FACILITIES = [
    "KRX", "VLN", "QMT", "BPH", "ZRD", "FWS", "JNK", "TXL",
    "DRM", "HCG", "PSN", "WYR", "AEL", "MVO", "CGN", "UBT",
]

VIOLATION_TYPES = [
    "leak", "emission_excess", "waste_overflow", "containment_breach",
    "filter_failure", "thermal_deviation", "pressure_anomaly",
    "corrosion_detected", "ventilation_fault", "runoff_contamination",
    "noise_exceedance", "particulate_spike", "chemical_spill",
    "structural_crack", "seal_degradation",
]

SECTORS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

SEVERITY_LEVELS = ["negligible", "low", "moderate", "elevated", "high", "critical"]

ACTIONS = [
    "no_action", "log_only", "reinspect_30d", "reinspect_7d",
    "immediate_halt", "partial_shutdown", "full_shutdown",
]

PENALTY_TIERS = ["none", "tier_A", "tier_B", "tier_C", "tier_D", "tier_E"]

INSPECTOR_NAMES = [
    "Chen", "Okafor", "Petrov", "Nakamura", "Alvarez", "Johansson",
    "Mbeki", "Kowalski", "Singh", "Torres", "Lindgren", "Osei",
    "Yamamoto", "Fernandez", "Novak", "Reyes",
]

# ---------------------------------------------------------------------------
# Codebook rules — each rule is a (rule_id, NL description, function)
# The function takes the reconciled state dict and mutates it.
# ---------------------------------------------------------------------------

def _rule_any_inspector_flags(state, vtype):
    """If any inspector reports {vtype}, it is present."""
    for frag in state["_frags"]:
        if vtype in frag.get("violations_found", []):
            if vtype not in state["violations"]:
                state["violations"].append(vtype)

def _rule_majority_vote(state, vtype):
    """Include {vtype} only if majority of inspectors report it."""
    count = sum(1 for f in state["_frags"] if vtype in f.get("violations_found", []))
    total = len(state["_frags"])
    if count > total / 2:
        if vtype not in state["violations"]:
            state["violations"].append(vtype)
    else:
        if vtype in state["violations"]:
            state["violations"].remove(vtype)

def _rule_count_escalation(state, threshold, levels_up):
    """If violation count > {threshold}, escalate severity by {levels_up} level(s)."""
    if len(state["violations"]) > threshold:
        idx = SEVERITY_LEVELS.index(state["severity"])
        new_idx = min(idx + levels_up, len(SEVERITY_LEVELS) - 1)
        state["severity"] = SEVERITY_LEVELS[new_idx]

def _rule_sector_severity_boost(state, sector, levels_up):
    """If facility sector is {sector}, escalate severity by {levels_up}."""
    if state["sector"] == sector:
        idx = SEVERITY_LEVELS.index(state["severity"])
        new_idx = min(idx + levels_up, len(SEVERITY_LEVELS) - 1)
        state["severity"] = SEVERITY_LEVELS[new_idx]

def _rule_severity_to_action(state, sev, action):
    """If severity is {sev} or higher, set action to {action}."""
    if SEVERITY_LEVELS.index(state["severity"]) >= SEVERITY_LEVELS.index(sev):
        state["action"] = action

def _rule_action_to_penalty(state, action, penalty):
    """If action is {action}, set penalty to {penalty}."""
    if state["action"] == action:
        state["penalty"] = penalty

def _rule_corrupted_fallback_severity(state, fallback_sev):
    """If any fragment is [CORRUPTED], set severity to at least {fallback_sev}."""
    if any(f.get("corrupted", False) for f in state["_frags"]):
        if SEVERITY_LEVELS.index(state["severity"]) < SEVERITY_LEVELS.index(fallback_sev):
            state["severity"] = fallback_sev

def _rule_no_violations_override(state):
    """If no violations found after reconciliation, set severity=negligible, action=no_action, penalty=none."""
    if len(state["violations"]) == 0:
        state["severity"] = "negligible"
        state["action"] = "no_action"
        state["penalty"] = "none"

def _rule_violation_implies_min_severity(state, vtype, min_sev):
    """If {vtype} is present, severity must be at least {min_sev}."""
    if vtype in state["violations"]:
        if SEVERITY_LEVELS.index(state["severity"]) < SEVERITY_LEVELS.index(min_sev):
            state["severity"] = min_sev

def _rule_max_penalty_cap(state, max_penalty):
    """Penalty cannot exceed {max_penalty}."""
    if PENALTY_TIERS.index(state["penalty"]) > PENALTY_TIERS.index(max_penalty):
        state["penalty"] = max_penalty


# ---------------------------------------------------------------------------
# Rule templates — used to generate per-row codebook subsets
# ---------------------------------------------------------------------------

RULE_TEMPLATES = [
    # (template_id, NL template, builder function)
    # builder(rng) -> (nl_text, apply_fn)
    ("R_ANY", lambda rng: _build_any_rule(rng)),
    ("R_MAJ", lambda rng: _build_majority_rule(rng)),
    ("R_CNT_ESC", lambda rng: _build_count_escalation(rng)),
    ("R_SEC_BOOST", lambda rng: _build_sector_boost(rng)),
    ("R_SEV_ACT", lambda rng: _build_severity_action(rng)),
    ("R_ACT_PEN", lambda rng: _build_action_penalty(rng)),
    ("R_CORR_FB", lambda rng: _build_corrupted_fallback(rng)),
    ("R_NO_VIOL", lambda _rng: _build_no_violations()),
    ("R_VIOL_SEV", lambda rng: _build_violation_severity(rng)),
    ("R_PEN_CAP", lambda rng: _build_penalty_cap(rng)),
]

def _build_any_rule(rng):
    vtype = rng.choice(VIOLATION_TYPES)
    nl = f"If any inspector reports '{vtype}', include it as a confirmed violation."
    def apply_fn(state):
        _rule_any_inspector_flags(state, vtype)
    return nl, apply_fn

def _build_majority_rule(rng):
    vtype = rng.choice(VIOLATION_TYPES)
    nl = f"Include '{vtype}' only if a majority of inspectors report it."
    def apply_fn(state):
        _rule_majority_vote(state, vtype)
    return nl, apply_fn

def _build_count_escalation(rng):
    threshold = rng.randint(1, 4)
    levels = rng.randint(1, 2)
    nl = f"If the total violation count exceeds {threshold}, escalate severity by {levels} level(s)."
    def apply_fn(state):
        _rule_count_escalation(state, threshold, levels)
    return nl, apply_fn

def _build_sector_boost(rng):
    sector = rng.choice(SECTORS)
    levels = rng.randint(1, 2)
    nl = f"If the facility sector is '{sector}', escalate severity by {levels} level(s)."
    def apply_fn(state):
        _rule_sector_severity_boost(state, sector, levels)
    return nl, apply_fn

def _build_severity_action(rng):
    sev = rng.choice(["low", "low", "moderate", "moderate", "elevated", "high"])
    possible_actions = [a for a in ACTIONS if a != "no_action"]
    action = rng.choice(possible_actions)
    nl = f"If severity is '{sev}' or higher, set action to '{action}'."
    def apply_fn(state):
        _rule_severity_to_action(state, sev, action)
    return nl, apply_fn

def _build_action_penalty(rng):
    possible_actions = [a for a in ACTIONS if a != "no_action"]
    action = rng.choice(possible_actions)
    penalty = rng.choice([p for p in PENALTY_TIERS if p != "none"])
    nl = f"If action is '{action}', set penalty to '{penalty}'."
    def apply_fn(state):
        _rule_action_to_penalty(state, action, penalty)
    return nl, apply_fn

def _build_corrupted_fallback(rng):
    sev = rng.choice(SEVERITY_LEVELS[2:])
    nl = f"If any inspector fragment is marked [CORRUPTED], set severity to at least '{sev}'."
    def apply_fn(state):
        _rule_corrupted_fallback_severity(state, sev)
    return nl, apply_fn

def _build_no_violations():
    nl = "If no violations remain after reconciliation, set severity to 'negligible', action to 'no_action', and penalty to 'none'."
    def apply_fn(state):
        _rule_no_violations_override(state)
    return nl, apply_fn

def _build_violation_severity(rng):
    vtype = rng.choice(VIOLATION_TYPES)
    min_sev = rng.choice(SEVERITY_LEVELS[2:])
    nl = f"If '{vtype}' is a confirmed violation, severity must be at least '{min_sev}'."
    def apply_fn(state):
        _rule_violation_implies_min_severity(state, vtype, min_sev)
    return nl, apply_fn

def _build_penalty_cap(rng):
    cap = rng.choice(PENALTY_TIERS[2:])
    nl = f"Penalty cannot exceed '{cap}' for this facility class."
    def apply_fn(state):
        _rule_max_penalty_cap(state, cap)
    return nl, apply_fn


# ---------------------------------------------------------------------------
# Fragment generation
# ---------------------------------------------------------------------------

def _generate_fragment(rng, inspector, facility_id, sector, tier_violations,
                       contradict_prob, corrupt_prob):
    """Generate one inspector's fragment for a facility visit."""
    frag = {
        "inspector": inspector,
        "facility": facility_id,
        "sector": sector,
        "corrupted": False,
        "violations_found": [],
        "notes": "",
    }

    reported = []
    for vtype in tier_violations:
        if rng.random() < 0.7:
            reported.append(vtype)

    # Contradiction: sometimes report violations NOT in the ground set
    if rng.random() < contradict_prob:
        extra = rng.choice([v for v in VIOLATION_TYPES if v not in tier_violations])
        reported.append(extra)

    # Contradiction: sometimes omit a real violation
    if reported and rng.random() < contradict_prob:
        reported.pop(rng.randint(0, len(reported) - 1))

    frag["violations_found"] = sorted(set(reported))

    # Build NL notes
    if frag["violations_found"]:
        viol_str = ", ".join(frag["violations_found"])
        templates = [
            f"Inspector {inspector} observed: {viol_str} in sector {sector}.",
            f"Findings by {inspector}: detected {viol_str}. Sector: {sector}.",
            f"{inspector} report — violations identified: {viol_str}. Location: sector {sector}.",
            f"Field notes ({inspector}): {viol_str} confirmed at sector {sector} of facility {facility_id}.",
        ]
    else:
        templates = [
            f"Inspector {inspector}: no violations detected in sector {sector}.",
            f"{inspector} report — sector {sector} clear. No issues found.",
            f"Findings by {inspector}: all parameters within tolerance at sector {sector}.",
        ]
    frag["notes"] = rng.choice(templates)

    # Corruption
    if rng.random() < corrupt_prob:
        frag["corrupted"] = True
        words = frag["notes"].split()
        if len(words) > 4:
            start = rng.randint(1, len(words) - 3)
            end = min(start + rng.randint(2, 4), len(words))
            words[start:end] = ["[CORRUPTED]"]
            frag["notes"] = " ".join(words)

    return frag


def _format_fragments(frags):
    """Format fragments into the input text block."""
    parts = []
    for i, f in enumerate(frags):
        header = f"--- Fragment {i+1} (Inspector {f['inspector']}) ---"
        parts.append(header)
        parts.append(f["notes"])
        if f["corrupted"]:
            parts.append("[Note: partial data corruption detected in this fragment]")
    return "\n".join(parts)


def _format_codebook(rules_nl):
    """Format codebook rules into the input text block."""
    lines = ["=== Compliance Codebook ==="]
    for i, nl in enumerate(rules_nl):
        lines.append(f"Rule {i+1}: {nl}")
    return "\n".join(lines)


def _format_verdict(state):
    """Format the 6-field verdict string."""
    viols = ",".join(sorted(state["violations"])) if state["violations"] else "none"
    return (
        f"FACILITY:{state['facility']} | "
        f"VIOLATIONS:{viols} | "
        f"COUNT:{len(state['violations'])} | "
        f"SEVERITY:{state['severity']} | "
        f"ACTION:{state['action']} | "
        f"PENALTY:{state['penalty']}"
    )


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def generate_row(rng, row_id):
    facility_prefix = rng.choice(FACILITIES)
    facility_num = rng.randint(100, 9999)
    facility_id = f"{facility_prefix}-{facility_num:04d}"
    sector = rng.choice(SECTORS)

    # Decide ground-truth violations (1-5)
    n_violations = rng.randint(0, 5)
    tier_violations = rng.sample(VIOLATION_TYPES, min(n_violations, len(VIOLATION_TYPES)))

    # Generate 2-4 inspector fragments
    n_inspectors = rng.randint(2, 4)
    inspectors = rng.sample(INSPECTOR_NAMES, n_inspectors)

    frags = []
    for insp in inspectors:
        frag = _generate_fragment(
            rng, insp, facility_id, sector, tier_violations,
            contradict_prob=0.35, corrupt_prob=0.15,
        )
        frags.append(frag)

    # Build codebook: always include at least one of each critical rule type,
    # plus 2-4 additional random rules. Order matters — reconciliation rules
    # first, then escalation, then action/penalty mapping.
    rules_nl = []
    rules_fn = []

    # Phase 1: violation reconciliation rules (2-3 rules)
    recon_templates = [t for t in RULE_TEMPLATES if t[0] in ("R_ANY", "R_MAJ")]
    n_recon = rng.randint(2, 3)
    for _, builder in rng.sample(recon_templates * 3, n_recon):
        nl, fn = builder(rng)
        rules_nl.append(nl)
        rules_fn.append(fn)

    # Phase 2: escalation rules (1-3 rules)
    esc_templates = [t for t in RULE_TEMPLATES
                     if t[0] in ("R_CNT_ESC", "R_SEC_BOOST", "R_CORR_FB", "R_VIOL_SEV")]
    n_esc = rng.randint(1, 3)
    for _, builder in rng.sample(esc_templates * 2, n_esc):
        nl, fn = builder(rng)
        rules_nl.append(nl)
        rules_fn.append(fn)

    # Phase 3: always include a no-violations override
    nl, fn = _build_no_violations()
    rules_nl.append(nl)
    rules_fn.append(fn)

    # Phase 4: severity→action mapping (always 1-2)
    for _ in range(rng.randint(1, 2)):
        nl, fn = _build_severity_action(rng)
        rules_nl.append(nl)
        rules_fn.append(fn)

    # Phase 5: action→penalty mapping (always 2-4 to ensure coverage)
    for _ in range(rng.randint(2, 4)):
        nl, fn = _build_action_penalty(rng)
        rules_nl.append(nl)
        rules_fn.append(fn)

    # Phase 6: optional penalty cap (30% chance)
    if rng.random() < 0.30:
        nl, fn = _build_penalty_cap(rng)
        rules_nl.append(nl)
        rules_fn.append(fn)

    # Initialize state
    base_severity_idx = min(n_violations, len(SEVERITY_LEVELS) - 1)
    state = {
        "facility": facility_id,
        "sector": sector,
        "violations": list(tier_violations),
        "severity": SEVERITY_LEVELS[base_severity_idx],
        "action": "no_action",
        "penalty": "none",
        "_frags": frags,
    }

    # Apply rules in order (order matters!)
    for fn in rules_fn:
        fn(state)

    # Format outputs
    fragments_text = _format_fragments(frags)
    codebook_text = _format_codebook(rules_nl)
    verdict = _format_verdict(state)

    input_text = fragments_text + "\n\n" + codebook_text

    return {
        "id": row_id,
        "input": input_text,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate compliance resolution dataset")
    parser.add_argument("--output", type=Path, default=Path("data.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=20_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows = []
    for i in range(args.size):
        row = generate_row(rng, i)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    from collections import Counter
    n_viols = [row["verdict"].count(",") + (0 if "VIOLATIONS:none" in row["verdict"] else 1)
               for row in rows]
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Violation counts: min={min(n_viols)}, max={max(n_viols)}, "
          f"mean={sum(n_viols)/len(n_viols):.1f}")

    severities = Counter()
    actions = Counter()
    penalties = Counter()
    for row in rows:
        parts = dict(p.split(":", 1) for p in row["verdict"].split(" | "))
        severities[parts["SEVERITY"]] += 1
        actions[parts["ACTION"]] += 1
        penalties[parts["PENALTY"]] += 1

    print(f"  Severities: {dict(sorted(severities.items()))}")
    print(f"  Actions: {dict(sorted(actions.items()))}")
    print(f"  Penalties: {dict(sorted(penalties.items()))}")
    print(f"  Input len: min={min(len(r['input']) for r in rows)}, "
          f"max={max(len(r['input']) for r in rows)}")


if __name__ == "__main__":
    main()
