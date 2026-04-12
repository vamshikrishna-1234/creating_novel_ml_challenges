"""
Generate synthetic conflicting-witness incident reconstruction data.

Domain: NLP — multi-witness disagreement resolution for incident reports.

Hidden structure:
  - 8000 incidents with ground-truth structured reports (6 fields)
  - 4-6 witnesses per incident, each with a hidden reliability profile
  - Witnesses disagree on 1-3 fields based on field-specific error rates
  - Witnesses add red-herring details WITH DECOY CODES to their statements
  - Statement templates vary to prevent regex-based extraction
  - ~12% of incidents have genuinely ambiguous ground truth
"""

import numpy as np
import pandas as pd
from pathlib import Path

ACTORS = [f"Worker-{chr(65+i)}{j}" for i in range(6) for j in range(5)]  # 30
ACTIONS = [
    "dropped container", "tripped over cable", "spilled solvent",
    "left valve open", "overloaded circuit", "ignored alarm",
    "misread gauge", "skipped inspection", "removed guard",
    "used wrong tool", "entered restricted zone", "disabled sensor",
    "mixed chemicals", "bypassed lock", "exceeded load limit",
    "forgot PPE", "crossed barrier", "started machine early",
    "stored improperly", "blocked exit path",
]  # 20
LOCATIONS = [
    "Bay-1", "Bay-2", "Bay-3", "Bay-4", "Bay-5",
    "Control-Room", "Loading-Dock", "Storage-West", "Storage-East",
    "Corridor-North", "Corridor-South", "Mezzanine",
    "Roof-Access", "Basement-Pump", "Yard-Exterior",
]  # 15
TIME_PERIODS = [
    "early-morning", "mid-morning", "late-morning", "noon",
    "early-afternoon", "late-afternoon", "evening", "night",
]  # 8
SEVERITIES = ["minor", "low", "moderate", "high", "critical"]  # 5
FACTORS = [
    "fatigue", "inadequate-training", "equipment-malfunction",
    "poor-lighting", "communication-failure", "time-pressure",
    "understaffing", "weather-conditions", "procedural-gap",
    "distraction", "complacency", "supervision-lapse",
]  # 12

WITNESS_ROLES = [
    "floor-supervisor", "nearby-operator", "maintenance-tech",
    "safety-officer", "passing-worker", "control-room-operator",
    "contractor", "shift-lead",
]

STATEMENT_TEMPLATES = [
    "{intro} {actor} {action} at {location} during the {time_period}. The incident severity appeared to be {severity}. The main contributing factor seemed to be {contributing_factor}.",
    "{intro} the incident involved {actor} who {action}. This took place at {location}. The time was {time_period} and severity was assessed as {severity}. Root cause analysis pointed to {contributing_factor}.",
    "{intro} at {location} during {time_period}, {actor} {action}. I would classify severity as {severity}. In my assessment, {contributing_factor} was the primary factor.",
    "{intro} during {time_period} I observed {actor} at {location}. The individual {action}. Severity: {severity}. The underlying cause appeared to be {contributing_factor}.",
    "{intro} {action} was the event that occurred. It was {actor} at {location}. The timing was {time_period}. Severity level: {severity}. Contributing factor: {contributing_factor}.",
    "{intro} I noted that {actor} {action} in the vicinity of {location}. It was during the {time_period} period. The severity was {severity} and the contributing factor was {contributing_factor}.",
    "{intro} the event happened at {location} when {actor} {action}. Timeframe: {time_period}. My severity assessment is {severity}. I believe {contributing_factor} played a key role.",
    "{intro} {actor} was involved in an incident where they {action}. Location: {location}. Time: {time_period}. Severity appeared {severity}. Factor: {contributing_factor}.",
]

RED_HERRINGS_WITH_DECOYS = [
    "There was a strange smell in the area near {loc_decoy}.",
    "The lighting was flickering at the time, possibly related to {factor_decoy}.",
    "Someone mentioned {actor_decoy} was seen near a delivery truck outside.",
    "I heard radio chatter about {action_decoy} in a different section.",
    "The ventilation system near {loc_decoy} was making unusual noise.",
    "A forklift operated by {actor_decoy} was moving nearby but seemed unrelated.",
    "There had been a safety drill earlier that day at {loc_decoy}.",
    "The temperature was unusually warm, possibly due to {factor_decoy}.",
    "I noticed some debris on the floor near {loc_decoy}.",
    "An alarm from {loc_decoy} went off briefly, but I think it was about {action_decoy}.",
    "{actor_decoy} was on the phone in the background discussing {factor_decoy}.",
    "A new batch of materials had just arrived at {loc_decoy}.",
    "I also recall {actor_decoy} mentioning something about {factor_decoy} earlier.",
    "The conveyor belt in {loc_decoy} was running, possibly due to {action_decoy}.",
    "I overheard that {actor_decoy} had {action_decoy} sometime during {time_decoy}.",
]


def _make_statement(rng, incident, witness_role, field_errors):
    role_intros = {
        "floor-supervisor": "As the floor supervisor on duty, I observed that",
        "nearby-operator": "I was operating equipment nearby and saw that",
        "maintenance-tech": "I was performing maintenance in the area and noticed",
        "safety-officer": "During my safety rounds, I witnessed that",
        "passing-worker": "I was passing through and caught a glimpse of",
        "control-room-operator": "From the control room monitors, I could see that",
        "contractor": "I was on-site for contracted work and happened to see",
        "shift-lead": "As shift lead, I was informed and observed that",
    }
    intro = role_intros.get(witness_role, "I witnessed that")

    reported = {}
    for field in ["actor", "action", "location", "time_period", "severity", "contributing_factor"]:
        true_val = incident[field]
        if field in field_errors:
            pool = {
                "actor": ACTORS, "action": ACTIONS, "location": LOCATIONS,
                "time_period": TIME_PERIODS, "severity": SEVERITIES,
                "contributing_factor": FACTORS,
            }[field]
            alternatives = [v for v in pool if v != true_val]
            reported[field] = rng.choice(alternatives)
        else:
            reported[field] = true_val

    template = STATEMENT_TEMPLATES[rng.randint(0, len(STATEMENT_TEMPLATES))]
    main_text = template.format(
        intro=intro,
        actor=reported["actor"],
        action=reported["action"],
        location=reported["location"],
        time_period=reported["time_period"],
        severity=reported["severity"],
        contributing_factor=reported["contributing_factor"],
    )

    parts = [main_text]

    n_rh = rng.randint(1, 4)
    for _ in range(n_rh):
        rh_template = RED_HERRINGS_WITH_DECOYS[rng.randint(0, len(RED_HERRINGS_WITH_DECOYS))]
        rh_text = rh_template.format(
            loc_decoy=rng.choice(LOCATIONS),
            actor_decoy=rng.choice(ACTORS),
            action_decoy=rng.choice(ACTIONS),
            factor_decoy=rng.choice(FACTORS),
            time_decoy=rng.choice(TIME_PERIODS),
        )
        parts.append(rh_text)

    rng.shuffle(parts)
    return " ".join(parts)


def main():
    rng = np.random.RandomState(42)
    N_INCIDENTS = 8000

    incidents = []
    for i in range(N_INCIDENTS):
        incidents.append({
            "incident_id": i,
            "actor": rng.choice(ACTORS),
            "action": rng.choice(ACTIONS),
            "location": rng.choice(LOCATIONS),
            "time_period": rng.choice(TIME_PERIODS),
            "severity": rng.choice(SEVERITIES),
            "contributing_factor": rng.choice(FACTORS),
        })

    ambiguous_mask = rng.random(N_INCIDENTS) < 0.12
    for i in range(N_INCIDENTS):
        if ambiguous_mask[i]:
            fields_to_blur = rng.choice(
                ["actor", "action", "location", "severity", "contributing_factor"],
                size=rng.randint(1, 3), replace=False
            )
            for f in fields_to_blur:
                pool = {
                    "actor": ACTORS, "action": ACTIONS, "location": LOCATIONS,
                    "severity": SEVERITIES, "contributing_factor": FACTORS,
                }[f]
                if rng.random() < 0.5:
                    incidents[i][f] = rng.choice(pool)

    incidents_df = pd.DataFrame(incidents)

    ROLE_RELIABILITY = {
        "floor-supervisor":      {"actor": 0.70, "action": 0.65, "location": 0.80, "time_period": 0.75, "severity": 0.55, "contributing_factor": 0.45},
        "nearby-operator":       {"actor": 0.55, "action": 0.70, "location": 0.65, "time_period": 0.60, "severity": 0.50, "contributing_factor": 0.40},
        "maintenance-tech":      {"actor": 0.40, "action": 0.50, "location": 0.55, "time_period": 0.45, "severity": 0.65, "contributing_factor": 0.75},
        "safety-officer":        {"actor": 0.60, "action": 0.60, "location": 0.70, "time_period": 0.70, "severity": 0.75, "contributing_factor": 0.70},
        "passing-worker":        {"actor": 0.25, "action": 0.35, "location": 0.45, "time_period": 0.30, "severity": 0.20, "contributing_factor": 0.20},
        "control-room-operator": {"actor": 0.35, "action": 0.40, "location": 0.75, "time_period": 0.80, "severity": 0.45, "contributing_factor": 0.35},
        "contractor":            {"actor": 0.30, "action": 0.40, "location": 0.35, "time_period": 0.40, "severity": 0.35, "contributing_factor": 0.25},
        "shift-lead":            {"actor": 0.65, "action": 0.55, "location": 0.60, "time_period": 0.70, "severity": 0.60, "contributing_factor": 0.55},
    }

    FIELDS = ["actor", "action", "location", "time_period", "severity", "contributing_factor"]

    statement_rows = []
    for i in range(N_INCIDENTS):
        n_witnesses = rng.randint(4, 7)
        roles = rng.choice(WITNESS_ROLES, n_witnesses, replace=False)

        for w_idx, role in enumerate(roles):
            reliability = ROLE_RELIABILITY[role]

            field_errors = set()
            for f in FIELDS:
                if rng.random() > reliability[f]:
                    field_errors.add(f)

            if len(field_errors) > 4:
                field_errors = set(rng.choice(list(field_errors), 4, replace=False))

            statement = _make_statement(rng, incidents[i], role, field_errors)

            statement_rows.append({
                "incident_id": i,
                "witness_idx": w_idx,
                "witness_role": role,
                "statement": statement,
            })

    statements_df = pd.DataFrame(statement_rows)
    ground_truth = incidents_df[["incident_id"] + FIELDS].copy()

    out = Path("raw_data")
    out.mkdir(exist_ok=True)
    ground_truth.to_csv(out / "ground_truth.csv", index=False)
    statements_df.to_csv(out / "statements.csv", index=False)

    print(f"Incidents: {len(incidents_df)}")
    print(f"Statements: {len(statements_df)} ({len(statements_df)/N_INCIDENTS:.1f} per incident)")
    print(f"Ambiguous incidents: {ambiguous_mask.sum()}")
    for f in FIELDS:
        nunique = ground_truth[f].nunique()
        print(f"  {f}: {nunique} unique values")


if __name__ == "__main__":
    main()
