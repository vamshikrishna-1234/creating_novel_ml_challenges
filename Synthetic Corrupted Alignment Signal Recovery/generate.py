"""
Generator for Synthetic Corrupted Alignment Signal Recovery.

Creates 60K synthetic instruction-response pairs rated by 5 hidden annotator
personas (3 legitimate with partial correlations, 1 adversarial, 1 noisy).
The observed training score is the average of 3 randomly selected personas.
Ground truth = consensus of the 3 legitimate personas, quantized to 5 tiers
(0-4) at quintile boundaries for balanced classes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 60_000

# ── Vocabulary pools ─────────────────────────────────────────────

TOPICS = [
    "renewable energy systems", "urban transportation planning",
    "deep ocean exploration", "ancient agricultural techniques",
    "quantum computing fundamentals", "childhood nutrition guidelines",
    "wildfire prevention strategies", "satellite communication networks",
    "music theory and composition", "volcanic eruption prediction",
    "microplastic pollution effects", "autonomous vehicle ethics",
    "traditional textile manufacturing", "gene therapy advancements",
    "arctic ecosystem preservation", "digital currency regulation",
    "earthquake-resistant architecture", "space debris management",
    "cognitive behavioral therapy", "precision agriculture methods",
    "coral reef restoration projects", "high-speed rail economics",
    "biodegradable packaging design", "pandemic preparedness planning",
    "dark matter detection methods", "indigenous language preservation",
    "vertical farming technology", "cybersecurity threat modeling",
    "sleep disorder treatments", "desalination plant efficiency",
    "artificial photosynthesis research", "elder care robotics",
    "avalanche risk assessment", "open source software licensing",
    "pollinator decline mitigation", "undersea cable infrastructure",
    "rare earth mineral recycling", "sports injury biomechanics",
    "food fermentation science", "mental health in remote work",
]

FORMATS = ["a concise summary", "a structured comparison", "a step-by-step guide",
           "an analytical overview", "a problem-solution outline"]

LENGTHS = ["brief (2-3 sentences)", "moderate (4-6 sentences)",
           "detailed (7-10 sentences)", "comprehensive (10+ sentences)"]

CONSTRAINTS = [
    "avoids technical jargon", "includes at least one concrete example",
    "acknowledges opposing viewpoints", "cites a specific numeric statistic",
    "prioritizes practical recommendations", "uses formal academic tone",
    "addresses common misconceptions", "compares at least two approaches",
    "ends with an open question for further research",
    "highlights ethical considerations", "focuses on recent developments",
    "mentions cost or economic impact", "targets a non-expert audience",
    "incorporates a historical perspective", "emphasizes measurable outcomes",
]

INTRO_TEMPLATES = [
    "{topic} is a field that has garnered increasing attention in recent years.",
    "Understanding {topic} requires examining multiple interconnected factors.",
    "The study of {topic} has evolved significantly over the past decade.",
    "When approaching {topic}, practitioners often consider several key dimensions.",
    "Recent developments in {topic} have raised important questions.",
    "{topic} presents both opportunities and challenges for modern society.",
    "An informed perspective on {topic} begins with foundational principles.",
    "The landscape of {topic} is shaped by technological and social forces.",
]

BODY_TEMPLATES = [
    "One important aspect involves {a}, which directly affects {b}.",
    "Research indicates that {a} plays a central role, particularly when combined with {b}.",
    "A key consideration is the relationship between {a} and {b}, as noted by multiple studies.",
    "From a practical standpoint, {a} must be balanced against {b} to achieve optimal results.",
    "Experts in the field emphasize that {a} cannot be understood in isolation from {b}.",
    "The interplay of {a} and {b} creates both risks and possibilities.",
    "Evidence suggests that addressing {a} first, before tackling {b}, yields better outcomes.",
    "While {a} has received considerable attention, {b} remains underexplored.",
]

EXAMPLE_TEMPLATES = [
    "For instance, a 2024 initiative demonstrated a {pct}% improvement through targeted intervention.",
    "Consider the case where implementation costs dropped by approximately {pct}% within two years.",
    "A notable example is the {pct}% adoption rate observed in pilot programs across three regions.",
    "Data from recent trials show a {pct}% reduction in negative outcomes when best practices were followed.",
]

CLOSING_TEMPLATES = [
    "Moving forward, continued investment in this area will be essential.",
    "These findings underscore the need for a coordinated, evidence-based approach.",
    "Ultimately, the balance between innovation and caution will determine long-term success.",
    "Further research is warranted to refine these strategies and extend their applicability.",
    "The path ahead requires collaboration among researchers, policymakers, and practitioners.",
    "Whether these trends continue will depend on both technological progress and policy decisions.",
]

ASPECT_POOL = [
    "resource allocation efficiency", "stakeholder engagement levels",
    "scalability of proposed solutions", "regulatory compliance requirements",
    "long-term environmental sustainability", "cost-benefit trade-offs",
    "workforce training and adaptation", "cross-sector collaboration",
    "data-driven decision frameworks", "risk mitigation strategies",
    "public perception and trust", "infrastructure modernization",
    "equity and accessibility concerns", "supply chain resilience",
    "technological readiness levels", "monitoring and evaluation protocols",
]

JARGON_WORDS = [
    "synergistic", "paradigm-shifting", "orthogonal", "heterogeneous",
    "stochastic", "isomorphic", "asymptotic", "endogenous",
]

MISCONCEPTION_PHRASES = [
    "Contrary to popular belief, ", "A common misunderstanding is that ",
    "It is often assumed incorrectly that ", "Despite widespread belief, ",
]

ALTERNATIVES = [
    "traditional methods", "purely market-driven solutions",
    "centralized governance models", "technology-only interventions",
]

QUESTION_CLOSERS = [
    "What remains unclear is whether current approaches can scale to meet global demand.",
    "An open question persists: can these gains be maintained under shifting conditions?",
    "Future work should ask whether the observed benefits generalize across contexts.",
]

ETHICS_PHRASES = [
    "Ethical scrutiny is warranted, particularly regarding equitable access.",
    "The ethical dimension cannot be overlooked, especially concerning vulnerable populations.",
    "Responsible deployment requires addressing fairness, transparency, and accountability.",
]


# ── Text generation engine ───────────────────────────────────────

def _pick(rng, pool):
    return pool[rng.randint(0, len(pool))]


def _gen_instruction(rng, topic_idx):
    topic = TOPICS[topic_idx]
    fmt = _pick(rng, FORMATS)
    length = _pick(rng, LENGTHS)
    constraint = _pick(rng, CONSTRAINTS)
    return (f"Provide {fmt} on the topic of {topic} that is {length} "
            f"and {constraint}."), fmt, length, constraint


def _gen_response(rng, topic_idx, fmt, length, constraint,
                  factual_ok, constraint_ok, format_ok, verbose_target, coherence_target):
    topic = TOPICS[topic_idx]
    parts = []

    parts.append(_pick(rng, INTRO_TEMPLATES).format(topic=topic))

    n_body = rng.randint(1, 5)
    for _ in range(n_body):
        a1, a2 = _pick(rng, ASPECT_POOL), _pick(rng, ASPECT_POOL)
        while a2 == a1:
            a2 = _pick(rng, ASPECT_POOL)
        parts.append(_pick(rng, BODY_TEMPLATES).format(a=a1, b=a2))

    if not factual_ok:
        wrong = TOPICS[(topic_idx + rng.randint(1, len(TOPICS))) % len(TOPICS)]
        parts.append(f"Notably, {wrong} has been identified as the primary driver here.")

    if "example" in constraint or "statistic" in constraint or "numeric" in constraint:
        if constraint_ok:
            parts.append(_pick(rng, EXAMPLE_TEMPLATES).format(pct=rng.randint(8, 65)))

    if "misconception" in constraint and constraint_ok:
        parts.append(_pick(rng, MISCONCEPTION_PHRASES) + "the evidence points in a different direction.")

    if ("compare" in constraint or "comparison" in fmt) and constraint_ok:
        alt = _pick(rng, ALTERNATIVES)
        parts.append(f"Compared to {alt}, this approach offers more holistic benefits.")

    if "ethical" in constraint and constraint_ok:
        parts.append(_pick(rng, ETHICS_PHRASES))

    if "question" in constraint and constraint_ok:
        parts.append(_pick(rng, QUESTION_CLOSERS))

    if "jargon" in constraint and not constraint_ok:
        for _ in range(rng.randint(2, 5)):
            parts.append(f"The {_pick(rng, JARGON_WORDS)} implications are particularly noteworthy.")

    if "formal" in constraint and not format_ok:
        parts.append("lol honestly this stuff is wild and super cool tbh")

    if coherence_target < 0.35:
        rng.shuffle(parts)
        if len(parts) > 2:
            parts.insert(rng.randint(1, len(parts)), parts[0])

    if verbose_target > 0.65:
        for _ in range(rng.randint(2, 4)):
            parts.append(_pick(rng, BODY_TEMPLATES).format(
                a=_pick(rng, ASPECT_POOL), b=_pick(rng, ASPECT_POOL)))

    parts.append(_pick(rng, CLOSING_TEMPLATES))

    if "step-by-step" in fmt and format_ok:
        return " ".join(f"Step {i+1}: {p}" for i, p in enumerate(parts))

    return " ".join(parts)


# ── Annotator personas ───────────────────────────────────────────

ADVERSARIAL_TOPICS = set(range(0, len(TOPICS), 3))


def _persona_weights():
    """Return weight matrices for personas 0-4. Each row = persona, cols = (f, cn, fm, v, co)."""
    return {
        0: np.array([0.45, 0.30, 0.00, 0.00, 0.25]),
        1: np.array([0.10, 0.45, 0.30, 0.00, 0.15]),
        2: np.array([0.00, 0.10, 0.20, 0.15, 0.55]),
    }


def persona_score(pid, dims_arr, topic_idx, rng):
    w = _persona_weights()
    if pid <= 2:
        raw = float(np.dot(w[pid], dims_arr))
    elif pid == 3:
        if topic_idx in ADVERSARIAL_TOPICS:
            raw = 1.0 - float(np.dot(np.array([0.40, 0.35, 0.00, 0.00, 0.25]), dims_arr))
        else:
            base_w = 0.40 * w[0] + 0.60 * w[1]
            raw = float(np.dot(base_w, dims_arr))
    elif pid == 4:
        raw = float(np.dot(w[0], dims_arr)) * 0.60 + float(np.dot(w[2], dims_arr)) * 0.40
        raw += rng.uniform(-0.20, 0.20)
    else:
        raw = 0.5
    return float(np.clip(raw, 0.0, 1.0))


def consensus_score(dims_arr):
    w = _persona_weights()
    scores = [float(np.dot(w[p], dims_arr)) for p in range(3)]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


# ── Main generation ──────────────────────────────────────────────

def generate(out_dir="raw_data", seed=SEED):
    rng = np.random.RandomState(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    gt_scores = []

    for idx in range(N_SAMPLES):
        topic_idx = rng.randint(0, len(TOPICS))

        factual_ok = rng.random() > 0.40
        constraint_ok = rng.random() > 0.45
        format_ok = rng.random() > 0.35
        verbose_raw = rng.beta(2.0, 2.0)
        coherence_raw = rng.beta(2.0, 2.5)

        f_dim = rng.uniform(0.70, 1.00) if factual_ok else rng.uniform(0.00, 0.40)
        cn_dim = rng.uniform(0.65, 1.00) if constraint_ok else rng.uniform(0.00, 0.40)
        fm_dim = rng.uniform(0.70, 1.00) if format_ok else rng.uniform(0.00, 0.45)
        v_dim = float(np.clip(verbose_raw + rng.normal(0, 0.08), 0, 1))
        co_dim = float(np.clip(coherence_raw, 0, 1))

        dims = np.array([f_dim, cn_dim, fm_dim, v_dim, co_dim])

        instr, fmt, length, constraint = _gen_instruction(rng, topic_idx)
        resp = _gen_response(rng, topic_idx, fmt, length, constraint,
                             factual_ok, constraint_ok, format_ok,
                             verbose_raw, coherence_raw)

        all_scores = [persona_score(pid, dims, topic_idx, rng) for pid in range(5)]
        selected = sorted(rng.choice(5, 3, replace=False))
        observed = float(np.mean([all_scores[i] for i in selected]))

        gt = consensus_score(dims)
        gt_scores.append(gt)

        records.append({
            "sample_id": idx,
            "instruction": instr,
            "response": resp,
            "topic_id": topic_idx,
            "observed_score": round(observed, 4),
            "response_word_count": len(resp.split()),
            "topic_popularity": int(rng.poisson(50) + 10),
            "_gt_raw": gt,
        })

        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1}/{N_SAMPLES}")

    gt_arr = np.array(gt_scores)
    boundaries = np.percentile(gt_arr, [20, 40, 60, 80])
    print(f"Tier boundaries (quintiles): {boundaries}")

    for rec in records:
        gt_raw = rec.pop("_gt_raw")
        if gt_raw < boundaries[0]:
            tier = 0
        elif gt_raw < boundaries[1]:
            tier = 1
        elif gt_raw < boundaries[2]:
            tier = 2
        elif gt_raw < boundaries[3]:
            tier = 3
        else:
            tier = 4
        rec["quality_tier"] = tier

    noise_rng = np.random.RandomState(seed + 1)
    for rec in records:
        if noise_rng.random() < 0.15:
            shift = noise_rng.choice([-1, 1])
            rec["quality_tier"] = int(np.clip(rec["quality_tier"] + shift, 0, 4))

    df = pd.DataFrame(records)
    df.to_csv(str(out / "data.csv"), index=False)

    print(f"Done: {N_SAMPLES} samples written to {out / 'data.csv'}")
    print(f"Tier distribution:\n{df['quality_tier'].value_counts().sort_index()}")
    print(f"Observed score stats:\n{df['observed_score'].describe()}")
    print(f"Tier boundaries used: {list(np.round(boundaries, 4))}")


if __name__ == "__main__":
    generate()
