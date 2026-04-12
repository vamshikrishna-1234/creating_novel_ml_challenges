"""
Ambiguous Instruction Resolution Challenge — data generator.

Each row: a workplace scenario (3-5 sentences, multiple named entities),
an ambiguous instruction (pronoun/scope/attachment/ellipsis ambiguity),
and 4 candidate interpretations (one correct, three plausible distractors).

The correct interpretation is determined by contextual clues seeded into
the scenario. Each row uses a different entity configuration, so no
lookup table can solve the task.

Output: data.csv  (id, scenario, instruction, option_a, option_b, option_c, option_d, answer)

6 ambiguity types:
  1. Pronoun reference (he/she/they/it/them)
  2. Scope ambiguity ("all X and Y" — does "all" scope over both?)
  3. Attachment ambiguity ("reviewed the report with the errors")
  4. Ellipsis resolution ("Manager A approved, and B did too" — approved what?)
  5. Quantifier scope ("every team didn't submit" — universal negation vs not-all)
  6. Temporal ambiguity ("before the meeting, after the review" — ordering)
"""

from __future__ import annotations
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Priya", "Marcus", "Lena", "Carlos", "Yuki", "Amara", "Dmitri", "Sofia",
    "Kwame", "Elena", "Raj", "Ingrid", "Hassan", "Mei", "Tomás", "Fatima",
    "Viktor", "Chiara", "Aiden", "Nadia", "Kofi", "Lucia", "Jens", "Zara",
    "Oleg", "Anya", "Ravi", "Sven", "Layla", "Esteban",
]

DEPARTMENTS = [
    "Engineering", "Marketing", "Operations", "Finance", "Legal",
    "Research", "Compliance", "Analytics", "Procurement", "Quality",
    "Infrastructure", "Product", "Security", "Support", "Design",
]

DOCUMENTS = [
    "the quarterly report", "the audit summary", "the project proposal",
    "the compliance review", "the budget forecast", "the risk assessment",
    "the vendor evaluation", "the performance review", "the incident report",
    "the strategy document", "the technical specification", "the training plan",
    "the resource allocation", "the timeline update", "the stakeholder brief",
]

TASKS = [
    "submit the deliverables", "finalize the schedule", "approve the budget",
    "review the findings", "distribute the memo", "escalate the issue",
    "validate the data", "update the dashboard", "prepare the presentation",
    "sign off on the changes", "archive the records", "notify the stakeholders",
    "complete the assessment", "revise the estimates", "coordinate the rollout",
]

LOCATIONS = [
    "the main conference room", "the downtown office", "the remote hub",
    "the third-floor lab", "the executive boardroom", "the shared workspace",
    "the satellite office", "the training center", "the operations floor",
]

MEETING_TYPES = [
    "the quarterly review", "the planning session", "the status update",
    "the board presentation", "the project kickoff", "the retrospective",
    "the compliance briefing", "the vendor negotiation", "the design sprint",
]

# ---------------------------------------------------------------------------
# Ambiguity generators
# ---------------------------------------------------------------------------

def _pick_n_unique(pool, n, rng):
    return rng.sample(pool, min(n, len(pool)))


def _gen_pronoun_reference(rng):
    """Type 1: Who does 'he/she/they/them' refer to?"""
    names = _pick_n_unique(FIRST_NAMES, 3, rng)
    depts = _pick_n_unique(DEPARTMENTS, 3, rng)
    doc = rng.choice(DOCUMENTS)
    task = rng.choice(TASKS)

    # Determine genders for pronoun usage
    # Use "they" for ambiguity (works regardless of gender)
    p1, p2, p3 = names[0], names[1], names[2]
    d1, d2, d3 = depts[0], depts[1], depts[2]

    # Scenario: two people interact, then an instruction uses "they"
    # The correct referent is determined by which person was assigned the task
    correct_idx = rng.randint(0, 1)  # 0 = p1, 1 = p2

    if correct_idx == 0:
        scenario_lines = [
            f"{p1} from {d1} was assigned to {task} by end of week.",
            f"{p2} from {d2} reviewed {doc} and found several issues.",
            f"{p3} from {d3} scheduled a follow-up meeting to discuss the findings.",
            f"{p1} confirmed they would handle the revisions personally.",
            f"{p2} offered to assist but was told it was already covered.",
        ]
    else:
        scenario_lines = [
            f"{p1} from {d1} reviewed {doc} and flagged potential issues.",
            f"{p2} from {d2} was responsible for {task.replace('the ', 'the revised ')}.",
            f"{p3} from {d3} sent a reminder about the upcoming deadline.",
            f"{p2} acknowledged the deadline and confirmed they were on track.",
            f"{p1} mentioned they had completed their own review already.",
        ]

    scenario = " ".join(scenario_lines)
    instruction = f"Send them the latest version of {doc} so they can finalize it."

    correct_person = names[correct_idx]
    wrong_person = names[1 - correct_idx]

    options = [
        f"Send {doc} to {correct_person} so {correct_person} can finalize it.",
        f"Send {doc} to {wrong_person} so {wrong_person} can finalize it.",
        f"Send {doc} to {p3} so {p3} can finalize it.",
        f"Send {doc} to both {p1} and {p2} so they can finalize it together.",
    ]
    answer_idx = 0
    return scenario, instruction, options, answer_idx


def _gen_scope_ambiguity(rng):
    """Type 2: 'All X and Y' — does 'all' scope over both?"""
    names = _pick_n_unique(FIRST_NAMES, 2, rng)
    depts = _pick_n_unique(DEPARTMENTS, 3, rng)
    doc1 = rng.choice(DOCUMENTS)
    doc2 = rng.choice([d for d in DOCUMENTS if d != doc1])
    p1, p2 = names[0], names[1]

    # Correct interpretation: "all" scopes over first item only
    scope_type = rng.choice(["narrow", "wide"])

    if scope_type == "narrow":
        scenario_lines = [
            f"{p1} from {depts[0]} prepared multiple drafts of {doc1} over the past week.",
            f"{p2} from {depts[1]} produced a single version of {doc2}.",
            f"The review committee in {depts[2]} needs to see the relevant documents.",
            f"{p1} confirmed that all drafts of {doc1} are ready for review.",
            f"{p2} noted that {doc2} is also finalized.",
        ]
        instruction = f"Submit all drafts of {doc1} and {doc2} to the review committee."
        options = [
            f"Submit every draft of {doc1} plus the single {doc2} to the committee.",
            f"Submit every draft of both {doc1} and {doc2} to the committee.",
            f"Submit only the final draft of {doc1} and {doc2} to the committee.",
            f"Submit all drafts of {doc1} only; {doc2} was not requested.",
        ]
        answer_idx = 0
    else:
        scenario_lines = [
            f"{p1} from {depts[0]} created several versions of both {doc1} and {doc2}.",
            f"{p2} from {depts[1]} requested a complete archive of all document versions.",
            f"The {depts[2]} team needs every iteration for the audit trail.",
            f"{p1} confirmed that multiple drafts exist for both documents.",
            f"{p2} emphasized that nothing should be left out of the submission.",
        ]
        instruction = f"Submit all drafts of {doc1} and {doc2} to the review committee."
        options = [
            f"Submit every draft of both {doc1} and {doc2} to the committee.",
            f"Submit every draft of {doc1} plus only the final {doc2}.",
            f"Submit only the latest draft of each document.",
            f"Submit all drafts of {doc1} only; send {doc2} separately later.",
        ]
        answer_idx = 0

    scenario = " ".join(scenario_lines)
    return scenario, instruction, options, answer_idx


def _gen_attachment_ambiguity(rng):
    """Type 3: 'reviewed X with Y' — did Y accompany the review, or is Y part of X?"""
    names = _pick_n_unique(FIRST_NAMES, 3, rng)
    depts = _pick_n_unique(DEPARTMENTS, 2, rng)
    doc = rng.choice(DOCUMENTS)
    p1, p2, p3 = names[0], names[1], names[2]

    attach_type = rng.choice(["instrument", "attribute"])

    if attach_type == "instrument":
        # "with" means "together with person"
        scenario_lines = [
            f"{p1} from {depts[0]} needed help understanding {doc}.",
            f"{p2} from {depts[1]} has deep expertise in the subject area.",
            f"{p1} asked {p2} to join the review session.",
            f"{p3} noted that collaborative reviews produce better outcomes.",
            f"The two met yesterday afternoon to go through the document together.",
        ]
        instruction = f"{p1} reviewed {doc} with {p2} and found critical errors."
        options = [
            f"{p1} and {p2} reviewed {doc} together and found critical errors.",
            f"{p1} reviewed a version of {doc} that {p2} had annotated, and found critical errors.",
            f"{p1} reviewed {doc} and then separately discussed errors with {p2}.",
            f"{p2} reviewed {doc} on behalf of {p1} and found critical errors.",
        ]
        answer_idx = 0
    else:
        # "with" means "containing" (attribute of the document)
        scenario_lines = [
            f"{p1} from {depts[0]} was assigned to verify {doc}.",
            f"An earlier version of {doc} had been flagged for containing annotation marks from {p2}.",
            f"{p1} specifically requested the version that included {p2}'s comments.",
            f"{p3} confirmed that the annotated version was the one distributed.",
            f"{p1} spent the afternoon examining the flagged sections.",
        ]
        instruction = f"{p1} reviewed {doc} with {p2}'s annotations and found critical errors."
        options = [
            f"{p1} reviewed the version of {doc} containing {p2}'s annotations and found critical errors.",
            f"{p1} and {p2} reviewed {doc} together and found critical errors in the annotations.",
            f"{p1} reviewed {doc}, then added annotations for {p2}, and found critical errors.",
            f"{p2} reviewed {doc} and {p1} found critical errors in {p2}'s review.",
        ]
        answer_idx = 0

    scenario = " ".join(scenario_lines)
    return scenario, instruction, options, answer_idx


def _gen_ellipsis_resolution(rng):
    """Type 4: 'A did X, and B did too' — what did B do?"""
    names = _pick_n_unique(FIRST_NAMES, 3, rng)
    depts = _pick_n_unique(DEPARTMENTS, 3, rng)
    task1 = rng.choice(TASKS)
    task2 = rng.choice([t for t in TASKS if t != task1])
    doc = rng.choice(DOCUMENTS)
    p1, p2, p3 = names[0], names[1], names[2]

    ellipsis_type = rng.choice(["same_action", "same_target"])

    if ellipsis_type == "same_action":
        scenario_lines = [
            f"{p1} from {depts[0]} was tasked with reviewing {doc}.",
            f"{p2} from {depts[1]} was also assigned to review documents this week.",
            f"{p3} from {depts[2]} confirmed both assignments in the project tracker.",
            f"{p1} completed the review of {doc} on Tuesday.",
            f"{p2} finished their assigned review on Wednesday.",
        ]
        instruction = f"{p1} approved {doc}, and {p2} did too."
        options = [
            f"Both {p1} and {p2} approved {doc}.",
            f"{p1} approved {doc}, and {p2} approved a different document.",
            f"{p1} approved {doc}, and {p2} acknowledged the approval.",
            f"{p1} and {p2} jointly co-approved {doc} as a single action.",
        ]
        answer_idx = 0
    else:
        scenario_lines = [
            f"{p1} from {depts[0]} needed to {task1} before the deadline.",
            f"{p2} from {depts[1]} had the same task assigned independently.",
            f"{p3} from {depts[2]} was tracking completion for both.",
            f"The deadline for this task was set for Friday.",
            f"Both {p1} and {p2} were aware of the shared deadline.",
        ]
        instruction = f"{p1} managed to {task1}, and {p2} did too."
        options = [
            f"Both {p1} and {p2} managed to {task1}.",
            f"{p1} managed to {task1}, and {p2} managed to {task2}.",
            f"{p1} managed to {task1}, and {p2} helped {p1} with it.",
            f"{p1} managed to {task1} for {p2}, who also benefited.",
        ]
        answer_idx = 0

    scenario = " ".join(scenario_lines)
    return scenario, instruction, options, answer_idx


def _gen_quantifier_scope(rng):
    """Type 5: 'Every team didn't submit' — none submitted vs not all submitted."""
    names = _pick_n_unique(FIRST_NAMES, 2, rng)
    depts = _pick_n_unique(DEPARTMENTS, 4, rng)
    doc = rng.choice(DOCUMENTS)
    p1, p2 = names[0], names[1]

    quant_type = rng.choice(["universal_neg", "not_all"])

    if quant_type == "universal_neg":
        # None of them submitted
        scenario_lines = [
            f"The {depts[0]}, {depts[1]}, and {depts[2]} teams were all due to submit {doc} by Friday.",
            f"{p1} from {depts[3]} checked the submission portal on Monday morning.",
            f"The portal showed zero submissions received from any of the three teams.",
            f"{p2} confirmed that none of the teams had uploaded anything.",
            f"The deadline had passed with a completely empty submission queue.",
        ]
        instruction = f"Every team didn't submit {doc} on time."
        options = [
            f"None of the teams submitted {doc} on time.",
            f"Not all teams submitted {doc} on time, but some did.",
            f"Every team submitted {doc}, but none were on time.",
            f"The teams collectively decided not to submit {doc}.",
        ]
        answer_idx = 0
    else:
        # Not all submitted (some did, some didn't)
        scenario_lines = [
            f"The {depts[0]}, {depts[1]}, and {depts[2]} teams were all due to submit {doc} by Friday.",
            f"{p1} from {depts[3]} checked the submission portal on Monday morning.",
            f"The {depts[0]} team had submitted on time, but the other two had not.",
            f"{p2} noted that partial compliance was better than nothing.",
            f"Follow-up reminders were sent to the two teams that missed the deadline.",
        ]
        instruction = f"Not every team submitted {doc} on time."
        options = [
            f"Some teams submitted {doc} on time, but not all of them did.",
            f"None of the teams submitted {doc} on time.",
            f"Every team submitted {doc}, just not by the original deadline.",
            f"The teams that submitted {doc} did so after the deadline.",
        ]
        answer_idx = 0

    scenario = " ".join(scenario_lines)
    return scenario, instruction, options, answer_idx


def _gen_temporal_ambiguity(rng):
    """Type 6: 'before X, after Y' — ordering of events."""
    names = _pick_n_unique(FIRST_NAMES, 3, rng)
    depts = _pick_n_unique(DEPARTMENTS, 2, rng)
    meeting = rng.choice(MEETING_TYPES)
    task = rng.choice(TASKS)
    doc = rng.choice(DOCUMENTS)
    p1, p2, p3 = names[0], names[1], names[2]

    temporal_type = rng.choice(["before_first", "after_first"])

    if temporal_type == "before_first":
        scenario_lines = [
            f"{p1} from {depts[0]} needed to {task} as a prerequisite for {meeting}.",
            f"{meeting} was scheduled for Thursday at 2 PM.",
            f"{p2} from {depts[1]} reminded {p1} that the task must be done beforehand.",
            f"{p3} confirmed that no agenda items could proceed without the prerequisite.",
            f"{p1} planned to complete everything by Wednesday evening.",
        ]
        instruction = f"{p1} should {task} before {meeting} and send {doc} to {p2} afterward."
        options = [
            f"First {p1} completes the task, then {meeting} happens, then {p1} sends {doc} to {p2}.",
            f"{p1} sends {doc} to {p2} first, then completes the task, then {meeting} happens.",
            f"{p1} completes the task and sends {doc} to {p2} simultaneously before {meeting}.",
            f"{meeting} happens first, then {p1} completes the task and sends {doc} to {p2}.",
        ]
        answer_idx = 0
    else:
        scenario_lines = [
            f"{meeting} was held on Monday and produced several action items.",
            f"{p1} from {depts[0]} was assigned to {task} as a follow-up.",
            f"{p2} from {depts[1]} needed {doc} once {p1}'s work was complete.",
            f"{p3} documented that the task depended on the meeting's outcomes.",
            f"The sequence was clear: meeting first, then task, then document delivery.",
        ]
        instruction = f"After {meeting}, {p1} should {task} and then forward {doc} to {p2}."
        options = [
            f"First {meeting} occurs, then {p1} completes the task, then {p1} forwards {doc} to {p2}.",
            f"{p1} completes the task before {meeting}, then forwards {doc} to {p2} afterward.",
            f"{p1} forwards {doc} to {p2} during {meeting}, then completes the task.",
            f"{p1} completes the task and forwards {doc} to {p2} simultaneously during {meeting}.",
        ]
        answer_idx = 0

    scenario = " ".join(scenario_lines)
    return scenario, instruction, options, answer_idx


AMBIGUITY_GENERATORS = [
    _gen_pronoun_reference,
    _gen_scope_ambiguity,
    _gen_attachment_ambiguity,
    _gen_ellipsis_resolution,
    _gen_quantifier_scope,
    _gen_temporal_ambiguity,
]

AMBIGUITY_NAMES = [
    "pronoun", "scope", "attachment", "ellipsis", "quantifier", "temporal",
]


def generate_row(rng: random.Random, row_id: int) -> dict:
    gen_idx = rng.randint(0, len(AMBIGUITY_GENERATORS) - 1)
    gen = AMBIGUITY_GENERATORS[gen_idx]
    scenario, instruction, options, answer_idx = gen(rng)

    # Shuffle options (track correct answer)
    indexed = list(enumerate(options))
    rng.shuffle(indexed)
    shuffled_options = [opt for _, opt in indexed]
    new_answer_idx = next(i for i, (orig_i, _) in enumerate(indexed) if orig_i == answer_idx)

    labels = ["A", "B", "C", "D"]

    return {
        "id": row_id,
        "scenario": scenario,
        "instruction": instruction,
        "option_a": shuffled_options[0],
        "option_b": shuffled_options[1],
        "option_c": shuffled_options[2],
        "option_d": shuffled_options[3],
        "answer": labels[new_answer_idx],
        "_ambiguity_type": AMBIGUITY_NAMES[gen_idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=20_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [generate_row(rng, i) for i in range(args.size)]

    fieldnames = ["id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d", "answer"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    answers = Counter(r["answer"] for r in rows)
    types = Counter(r["_ambiguity_type"] for r in rows)
    avg_scenario = sum(len(r["scenario"]) for r in rows) / len(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Answer distribution: {dict(sorted(answers.items()))}")
    print(f"  Ambiguity types: {dict(sorted(types.items()))}")
    print(f"  Avg scenario length: {avg_scenario:.0f} chars")


if __name__ == "__main__":
    main()
