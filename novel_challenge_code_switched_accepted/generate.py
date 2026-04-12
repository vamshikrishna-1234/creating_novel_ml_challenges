"""
Synthetic NLP dataset generator:
  Code-Switched Complaint Triage — Predict Resolution Hours

Domain: Customer complaints written in "code-switched" style (English mixed
with tokens from a fictional constructed language called Verani). The task is
to predict the resolution time (in hours, continuous 0.5–168).

Why it's novel:
  - Code-switching in NLP is an active research area but almost zero public
    benchmarks exist for regression tasks on code-switched text.
  - The fictional second language (Verani) means no overlap with real datasets.
  - Resolution time depends on non-obvious interactions:
      * code-switch density (higher → harder to parse → longer resolution)
      * embedded priority tags (P1–P5)
      * sentiment intensity of the English portions
      * product category codes
      * complaint length (proxy for complexity)
  - A naive TF-IDF baseline will plateau; agents need feature engineering
    on code-switch ratio, tag extraction, and possibly subword embeddings.

Output: single CSV with columns [id, text, category, target]
  - text: the code-switched complaint
  - category: one of 8 product categories (string)
  - target: resolution hours (float, 0.5–168)

Usage:
    python generate.py [--output data.csv] [--seed 42] [--size 28000]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

SEED = 42

# ---------- Fictional "Verani" language tokens ----------
# Grouped by semantic role so mixing is coherent

VERANI_GREETINGS = [
    "shaluma", "verik", "toshan", "nabori", "kelash",
]
VERANI_COMPLAINTS = [
    "drakan", "voshka", "zilmek", "toruka", "pleshni", "ghariv",
    "mondrek", "sulvak", "kresha", "bivano", "yultrek", "feshka",
]
VERANI_URGENCY = [
    "zhorik", "bretal", "kashmi", "ultrek", "voshan", "dremik",
]
VERANI_PRODUCTS = [
    "fendar", "gorvex", "talumi", "reshko", "belvax", "junori",
]
VERANI_CLOSINGS = [
    "gratesh", "meriva", "toshan-vi", "kelash-un", "dashori",
]
VERANI_FILLER = [
    "ko", "na", "ve", "shi", "um", "ek", "ri", "bo", "tal", "desh",
    "plo", "fre", "gal", "niv", "osh", "zul", "krem", "biv",
]

# ---------- English complaint fragments ----------

ENGLISH_OPENERS = [
    "I need help with", "I'm having issues with", "There's a problem with",
    "Something is wrong with", "I can't use", "We're experiencing failures in",
    "Please investigate", "Urgent request regarding", "Repeated issues with",
    "This has been broken since last week:",
]

ENGLISH_DETAILS_MILD = [
    "it occasionally shows a warning", "minor glitch in the display",
    "small delay when loading", "the button sometimes doesn't respond",
    "notification comes late", "formatting looks off on mobile",
    "search results are slightly wrong", "sync takes longer than expected",
]
ENGLISH_DETAILS_MODERATE = [
    "functionality is partially broken", "data is not syncing properly",
    "the export feature fails intermittently", "users report timeouts",
    "dashboard metrics are inconsistent", "API responses are delayed",
    "login works but sessions drop", "file uploads corrupt occasionally",
]
ENGLISH_DETAILS_SEVERE = [
    "the entire system is down", "no users can access the platform",
    "critical data loss has occurred", "payment processing is failing",
    "security vulnerability detected", "database corruption confirmed",
    "production servers are unreachable", "all API endpoints return 500",
]

ENGLISH_CLOSINGS = [
    "Please resolve ASAP.", "This is affecting our operations.",
    "Kindly look into this.", "We need a fix urgently.",
    "Our team is blocked.", "Any update would be appreciated.",
    "This is a blocker for release.", "Multiple teams impacted.",
]

CATEGORIES = [
    "billing", "authentication", "data-pipeline", "ui-frontend",
    "api-gateway", "storage", "notifications", "analytics",
]

PRIORITY_TAGS = ["P1", "P2", "P3", "P4", "P5"]


def set_seed(seed: int) -> None:
    random.seed(seed)


def _verani_fragment(rng: random.Random, length: int) -> str:
    """Build a fragment of Verani text."""
    tokens: list[str] = []
    for _ in range(length):
        roll = rng.random()
        if roll < 0.25:
            tokens.append(rng.choice(VERANI_COMPLAINTS))
        elif roll < 0.4:
            tokens.append(rng.choice(VERANI_URGENCY))
        elif roll < 0.55:
            tokens.append(rng.choice(VERANI_PRODUCTS))
        elif roll < 0.7:
            tokens.append(rng.choice(VERANI_FILLER))
        else:
            tokens.append(rng.choice(VERANI_FILLER) + "-" + rng.choice(VERANI_FILLER))
    return " ".join(tokens)


def _english_detail(rng: random.Random, severity: float) -> str:
    if severity < 0.33:
        return rng.choice(ENGLISH_DETAILS_MILD)
    elif severity < 0.66:
        return rng.choice(ENGLISH_DETAILS_MODERATE)
    else:
        return rng.choice(ENGLISH_DETAILS_SEVERE)


def compute_target(
    rng: random.Random,
    severity: float,
    priority: str,
    cs_ratio: float,
    num_sentences: int,
    category: str,
) -> float:
    """
    Resolution hours from multiple interacting signals.
    Base comes from severity, modulated by priority, code-switch density,
    complaint length, and category. Noise is added so it's not perfectly
    learnable.
    """
    priority_num = int(priority[1])

    # Base: severity drives most of the target (0.5–100h range)
    base = 2.0 + severity * 90.0

    # Priority multiplier: P1 is fastest (0.4x), P5 is slowest (1.8x)
    priority_mult = 0.4 + (priority_num - 1) * 0.35

    # Code-switch penalty: higher ratio → harder to parse → longer
    cs_penalty = 1.0 + cs_ratio * 0.6

    # Length factor: more sentences → more complex
    length_factor = 1.0 + (num_sentences - 3) * 0.08

    # Category modifiers (some categories are inherently faster/slower)
    cat_mods = {
        "billing": 1.15, "authentication": 0.9, "data-pipeline": 1.3,
        "ui-frontend": 0.8, "api-gateway": 1.1, "storage": 1.25,
        "notifications": 0.75, "analytics": 1.05,
    }
    cat_mult = cat_mods.get(category, 1.0)

    raw = base * priority_mult * cs_penalty * length_factor * cat_mult

    # Add heteroscedastic noise (more noise at higher values)
    noise_scale = 0.08 * raw + 1.5
    noise = rng.gauss(0, noise_scale)
    result = raw + noise

    return round(max(0.5, min(168.0, result)), 2)


def generate_complaint(
    report_id: int,
    rng: random.Random,
) -> dict:
    """Generate one code-switched complaint with metadata."""
    severity = rng.random()
    priority = rng.choice(PRIORITY_TAGS)
    category = rng.choice(CATEGORIES)
    num_sentences = rng.randint(3, 7)

    # Code-switch ratio: fraction of tokens that are Verani (0.1–0.7)
    cs_ratio = round(rng.uniform(0.1, 0.7), 2)

    parts: list[str] = []

    # Optionally start with Verani greeting
    if rng.random() < 0.4:
        parts.append(rng.choice(VERANI_GREETINGS) + ",")

    # Priority tag (embedded naturally)
    parts.append(f"[{priority}]")

    # English opener
    parts.append(rng.choice(ENGLISH_OPENERS))

    # Mix English details and Verani fragments
    for _ in range(num_sentences):
        if rng.random() < cs_ratio:
            vlen = rng.randint(2, 5)
            parts.append(_verani_fragment(rng, vlen))
        else:
            parts.append(_english_detail(rng, severity))

        # Occasional mid-sentence switch
        if rng.random() < cs_ratio * 0.5:
            parts.append(rng.choice(VERANI_URGENCY))

    # Closing: English or Verani
    if rng.random() < 0.3:
        parts.append(rng.choice(VERANI_CLOSINGS) + ".")
    else:
        parts.append(rng.choice(ENGLISH_CLOSINGS))

    text = " ".join(parts)

    target = compute_target(rng, severity, priority, cs_ratio, num_sentences, category)

    return {
        "id": report_id,
        "text": text,
        "category": category,
        "target": target,
    }


def verify_no_duplicates(rows: list[dict]) -> None:
    texts = [r["text"] for r in rows]
    if len(texts) != len(set(texts)):
        dups = len(texts) - len(set(texts))
        raise ValueError(f"Found {dups} duplicate texts — increase vocabulary or sentence count")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate code-switched complaint triage dataset")
    parser.add_argument("--output", type=Path, default=Path("data.csv"), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--size", type=int, default=28_000, help="Total number of samples")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    rows: list[dict] = []
    for i in range(args.size):
        rows.append(generate_complaint(i, rng))

    verify_no_duplicates(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "category", "target"])
        w.writeheader()
        w.writerows(rows)

    # Print summary stats
    targets = [r["target"] for r in rows]
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  target range: {min(targets):.2f} – {max(targets):.2f}")
    print(f"  target mean:  {sum(targets)/len(targets):.2f}")
    print(f"  target std:   {(sum((t - sum(targets)/len(targets))**2 for t in targets) / len(targets))**0.5:.2f}")
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"  categories:   {dict(sorted(cats.items()))}")


if __name__ == "__main__":
    main()
