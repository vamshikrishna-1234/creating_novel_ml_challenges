"""
Generate synthetic branching protocol outcome prediction data.

Domain: Sequence to Sequence — procedural execution tracing with
conditional branches, side effects, and latent catalyst variables.

Hidden structure:
  - 8000 protocols, each a DAG of 8-15 steps with conditional branches
  - 5 latent catalyst types (NOT exposed in data) that silently alter
    which branch is taken at 1-2 decision points per protocol
  - Steps have side effects that propagate through the DAG
  - ~8% of protocols are genuinely ambiguous (multiple valid outcomes)
  - Output: 4-field structured prediction (terminal_state, primary_product,
    byproduct_class, process_status)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

TERMINAL_STATES = [f"TS-{i:02d}" for i in range(20)]
PRIMARY_PRODUCTS = [f"PP-{i:02d}" for i in range(30)]
BYPRODUCT_CLASSES = [f"BC-{i:02d}" for i in range(10)]
PROCESS_STATUSES = ["completed", "failed", "partial", "timeout", "ambiguous"]

REAGENTS = [f"Reagent-{chr(65 + i // 10)}{chr(65 + i % 10)}{j}"
            for i in range(15) for j in range(3)]  # 45 reagents
TOOLS = [f"Unit-{chr(65 + i)}{j}" for i in range(8) for j in range(3)]  # 24 tools
STATES = [f"Phase-{name}" for name in [
    "Amber", "Azure", "Crimson", "Slate", "Jade", "Onyx", "Pearl",
    "Rust", "Teal", "Violet", "Ivory", "Copper", "Silver", "Cobalt",
    "Mauve", "Coral",
]]  # 16 states
CATALYST_TYPES = ["alpha", "beta", "gamma", "delta", "epsilon"]  # 5 hidden catalysts

STEP_TEMPLATES = [
    "Add {reagent} to the mixture using {tool}.",
    "Heat the solution in {tool} until it reaches {state}.",
    "Filter the contents through {tool} and collect the {state} fraction.",
    "Combine {reagent} with the current mixture and observe for {state} transition.",
    "Centrifuge in {tool} for standard duration. Expected result: {state}.",
    "Slowly introduce {reagent} while monitoring for {state} indicators.",
    "Transfer to {tool} and allow to settle. Target phase: {state}.",
    "Apply {reagent} under controlled conditions using {tool}.",
    "Neutralize with {reagent}. Verify {state} before proceeding.",
    "Evaporate in {tool} until concentration reaches {state} threshold.",
]

CONDITION_TEMPLATES = [
    "If the mixture shows {state}, proceed to step S{target_a}; otherwise go to step S{target_b}.",
    "Check for {state} indicators. If present: S{target_a}. If absent: S{target_b}.",
    "Branch on {state} detection: positive → S{target_a}, negative → S{target_b}.",
    "Depending on whether {state} is observed, continue with S{target_a} or S{target_b}.",
]

NOOP_TEMPLATES = [
    "Record environmental readings from {tool} for quality assurance.",
    "Verify {tool} calibration. Log readings for batch record.",
    "Perform visual inspection of mixture. Note current appearance in log.",
    "Wait for standard settling period. No intervention required.",
    "Cross-reference {tool} output with batch specification sheet.",
]

INTERFERENCE_TEMPLATES = [
    "As a precaution, add trace amount of {reagent} for stabilization.",
    "Run auxiliary diagnostic on {tool}. Results are advisory only.",
    "Optional: supplement with {reagent} to adjust buffer capacity.",
]


def _det_hash(seed_str):
    return int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)


def _generate_protocol(rng, protocol_id):
    """Generate a single protocol DAG with steps and conditional branches."""
    n_steps = rng.randint(8, 16)
    catalyst = CATALYST_TYPES[rng.randint(0, 5)]

    n_conditions = rng.randint(2, 5)
    condition_steps = sorted(rng.choice(range(2, n_steps - 1), min(n_conditions, n_steps - 3), replace=False))

    n_noops = rng.randint(1, 4)
    remaining = [i for i in range(1, n_steps) if i not in condition_steps]
    noop_steps = sorted(rng.choice(remaining, min(n_noops, len(remaining)), replace=False)) if remaining else []

    n_interference = rng.randint(1, 3)
    remaining2 = [i for i in range(1, n_steps) if i not in condition_steps and i not in noop_steps]
    interf_steps = sorted(rng.choice(remaining2, min(n_interference, len(remaining2)), replace=False)) if remaining2 else []

    steps_text = []
    step_metadata = []

    for s in range(n_steps):
        step_id = f"S{s:02d}"

        if s in condition_steps:
            state = rng.choice(STATES)
            target_a = min(s + rng.randint(1, 4), n_steps - 1)
            target_b = min(s + rng.randint(1, 4), n_steps - 1)
            while target_b == target_a and n_steps > s + 2:
                target_b = min(s + rng.randint(1, 4), n_steps - 1)

            tmpl = CONDITION_TEMPLATES[rng.randint(0, len(CONDITION_TEMPLATES))]
            text = f"[{step_id}] " + tmpl.format(
                state=state,
                target_a=f"{target_a:02d}",
                target_b=f"{target_b:02d}",
            )
            step_metadata.append({
                "step_idx": s, "type": "condition", "state_check": state,
                "branch_a": target_a, "branch_b": target_b
            })

        elif s in noop_steps:
            tmpl = NOOP_TEMPLATES[rng.randint(0, len(NOOP_TEMPLATES))]
            tool = rng.choice(TOOLS)
            text = f"[{step_id}] " + tmpl.format(tool=tool)
            step_metadata.append({"step_idx": s, "type": "noop"})

        elif s in interf_steps:
            tmpl = INTERFERENCE_TEMPLATES[rng.randint(0, len(INTERFERENCE_TEMPLATES))]
            reagent = rng.choice(REAGENTS)
            tool = rng.choice(TOOLS)
            text = f"[{step_id}] " + tmpl.format(reagent=reagent, tool=tool)
            step_metadata.append({"step_idx": s, "type": "interference"})

        else:
            tmpl = STEP_TEMPLATES[rng.randint(0, len(STEP_TEMPLATES))]
            reagent = rng.choice(REAGENTS)
            tool = rng.choice(TOOLS)
            state = rng.choice(STATES)
            text = f"[{step_id}] " + tmpl.format(reagent=reagent, tool=tool, state=state)
            step_metadata.append({
                "step_idx": s, "type": "action",
                "reagent": reagent, "state_produced": state
            })

        steps_text.append(text)

    return {
        "n_steps": n_steps,
        "steps_text": steps_text,
        "step_metadata": step_metadata,
        "condition_steps": condition_steps,
        "catalyst": catalyst,
    }


def _execute_protocol(rng, protocol, initial_conditions, catalyst):
    """Simulate execution of a protocol DAG to determine the outcome."""
    n_steps = protocol["n_steps"]
    metadata = protocol["step_metadata"]
    condition_steps = protocol["condition_steps"]

    current_state_vector = list(initial_conditions)
    executed_steps = []
    step = 0
    max_iterations = n_steps * 3
    iterations = 0

    while step < n_steps and iterations < max_iterations:
        iterations += 1
        meta = metadata[step]

        if meta["type"] == "condition":
            state_check = meta["state_check"]
            state_hash = _det_hash(f"{state_check}_{catalyst}_{sum(current_state_vector):.2f}")

            catalyst_modifier = CATALYST_TYPES.index(catalyst)
            condition_val = (state_hash + int(current_state_vector[catalyst_modifier % len(current_state_vector)] * 100)) % 100

            if condition_val >= 50:
                step = meta["branch_a"]
            else:
                step = meta["branch_b"]
            executed_steps.append(("condition", meta["step_idx"], condition_val >= 50))

        elif meta["type"] == "noop":
            executed_steps.append(("noop", meta["step_idx"]))
            step += 1

        elif meta["type"] == "interference":
            effect = rng.normal(0, 0.05)
            idx = meta["step_idx"] % len(current_state_vector)
            current_state_vector[idx] += effect
            executed_steps.append(("interference", meta["step_idx"]))
            step += 1

        elif meta["type"] == "action":
            reagent = meta.get("reagent", "")
            state_prod = meta.get("state_produced", "")
            reagent_hash = _det_hash(f"{reagent}_{catalyst}")
            state_hash = _det_hash(f"{state_prod}_{catalyst}")

            for i in range(len(current_state_vector)):
                delta = ((reagent_hash >> (i * 4)) & 0xF) / 15.0 - 0.5
                current_state_vector[i] += delta * 0.3
                current_state_vector[i] = max(-5, min(5, current_state_vector[i]))

            executed_steps.append(("action", meta["step_idx"]))
            step += 1
        else:
            step += 1

    status_hash = _det_hash(f"{catalyst}_{'_'.join(str(round(v,2)) for v in current_state_vector)}_{iterations}")
    sh = status_hash % 100
    if iterations >= max_iterations:
        process_status = "timeout"
    elif len(executed_steps) < 3:
        process_status = "failed"
    elif sh < 7:
        process_status = "ambiguous"
    elif sh < 13:
        process_status = "partial"
    elif sh < 18:
        process_status = "failed"
    elif sh < 22:
        process_status = "timeout"
    else:
        process_status = "completed"

    sv_sum = sum(current_state_vector)
    sv_prod = 1.0
    for v in current_state_vector:
        sv_prod *= abs(v) + 0.1

    ts_idx = int(abs(_det_hash(f"ts_{catalyst}_{sv_sum:.3f}_{len(executed_steps)}")) % len(TERMINAL_STATES))
    pp_idx = int(abs(_det_hash(f"pp_{catalyst}_{sv_prod:.3f}_{sv_sum:.3f}")) % len(PRIMARY_PRODUCTS))
    bc_idx = int(abs(_det_hash(f"bc_{catalyst}_{sv_sum:.3f}_{process_status}")) % len(BYPRODUCT_CLASSES))

    return {
        "terminal_state": TERMINAL_STATES[ts_idx],
        "primary_product": PRIMARY_PRODUCTS[pp_idx],
        "byproduct_class": BYPRODUCT_CLASSES[bc_idx],
        "process_status": process_status,
    }


def main():
    rng = np.random.RandomState(42)
    N_PROTOCOLS = 8000
    CONDITIONS_PER_PROTO = 3

    protocols = []
    for pid in range(N_PROTOCOLS):
        protocols.append(_generate_protocol(rng, pid))

    rows = []
    for pid, proto in enumerate(protocols):
        for cond_idx in range(CONDITIONS_PER_PROTO):
            n_vars = rng.randint(3, 6)
            initial_conditions = [round(rng.normal(0, 1.5), 3) for _ in range(n_vars)]

            while len(initial_conditions) < 5:
                initial_conditions.append(0.0)
            initial_conditions = initial_conditions[:5]

            outcome = _execute_protocol(rng, proto, initial_conditions, proto["catalyst"])

            step_texts = list(proto["steps_text"])
            rng.shuffle(step_texts)
            protocol_text = " ||| ".join(step_texts)

            rows.append({
                "protocol_id": pid,
                "condition_set": cond_idx,
                "protocol_text": protocol_text,
                "init_var_0": initial_conditions[0],
                "init_var_1": initial_conditions[1],
                "init_var_2": initial_conditions[2],
                "init_var_3": initial_conditions[3],
                "init_var_4": initial_conditions[4],
                "terminal_state": outcome["terminal_state"],
                "primary_product": outcome["primary_product"],
                "byproduct_class": outcome["byproduct_class"],
                "process_status": outcome["process_status"],
            })

    df = pd.DataFrame(rows)

    out = Path("raw_data")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "data.csv", index=False)

    print(f"Total rows: {len(df)}")
    print(f"Protocols: {N_PROTOCOLS}")
    print(f"Conditions per protocol: {CONDITIONS_PER_PROTO}")
    print(f"Terminal states used: {df['terminal_state'].nunique()}")
    print(f"Primary products used: {df['primary_product'].nunique()}")
    print(f"Byproduct classes used: {df['byproduct_class'].nunique()}")
    print(f"Process status distribution:")
    print(df["process_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
