"""
Synthetic Dialect Translation Challenge — data generator.

Creates English sentences from templates, then transforms each into one of
5 fictional dialects using systematic, deterministic rule combinations.
Each dialect applies a unique mix of:
  - Lexical substitution (common words → fictional tokens)
  - Morphological changes (tense/plural markers)
  - Syntax reordering (SVO→SOV, adjective placement)
  - Phonetic respelling (consonant cluster changes)
  - Agglutination (preposition merging)

Output: data.csv  (id, dialect, transformed, original)
"""

from __future__ import annotations
import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------------------
# Base sentence templates — {slot} placeholders filled at generation time
# ---------------------------------------------------------------------------

SUBJECTS = [
    "the engineer", "the researcher", "the analyst", "the director",
    "the operator", "the inspector", "the coordinator", "the specialist",
    "the technician", "the supervisor", "the consultant", "the auditor",
    "the planner", "the designer", "the manager", "the scientist",
    "the advisor", "the architect", "the developer", "the strategist",
]

VERBS_PRESENT = [
    ("examines", "examined", "examining"),
    ("reviews", "reviewed", "reviewing"),
    ("analyzes", "analyzed", "analyzing"),
    ("monitors", "monitored", "monitoring"),
    ("evaluates", "evaluated", "evaluating"),
    ("processes", "processed", "processing"),
    ("validates", "validated", "validating"),
    ("configures", "configured", "configuring"),
    ("deploys", "deployed", "deploying"),
    ("optimizes", "optimized", "optimizing"),
    ("calibrates", "calibrated", "calibrating"),
    ("documents", "documented", "documenting"),
    ("transmits", "transmitted", "transmitting"),
    ("compiles", "compiled", "compiling"),
    ("verifies", "verified", "verifying"),
    ("measures", "measured", "measuring"),
    ("schedules", "scheduled", "scheduling"),
    ("coordinates", "coordinated", "coordinating"),
    ("implements", "implemented", "implementing"),
    ("generates", "generated", "generating"),
]

OBJECTS = [
    "the report", "the dataset", "the configuration", "the protocol",
    "the module", "the framework", "the pipeline", "the interface",
    "the specification", "the benchmark", "the output", "the schema",
    "the parameters", "the results", "the metrics", "the samples",
    "the records", "the signals", "the components", "the findings",
]

ADJECTIVES = [
    "critical", "preliminary", "comprehensive", "automated",
    "sequential", "redundant", "systematic", "experimental",
    "operational", "structural", "functional", "analytical",
    "diagnostic", "regulatory", "provisional", "integrated",
    "distributed", "concurrent", "iterative", "adaptive",
]

ADVERBS = [
    "carefully", "efficiently", "thoroughly", "rapidly",
    "systematically", "precisely", "consistently", "reliably",
    "independently", "periodically", "continuously", "accurately",
]

PREPS = [
    "in the facility", "at the station", "for the project",
    "during the review", "after the inspection", "before the deadline",
    "under the protocol", "with the equipment", "through the system",
    "across the network", "within the department", "from the archive",
]

CONJUNCTIONS_BECAUSE = [
    "because the system required updates",
    "because the deadline was approaching",
    "because the previous version failed",
    "because the standards changed recently",
    "because the client requested modifications",
    "because the audit revealed issues",
]

CONJUNCTIONS_ALTHOUGH = [
    "although the resources were limited",
    "although the timeline was tight",
    "although the data was incomplete",
    "although the conditions were suboptimal",
    "although the team was understaffed",
    "although the budget was reduced",
]

# Sentence pattern generators
def _gen_simple_sv(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    return f"{subj} {v[0]} {obj}"

def _gen_past_sv(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    return f"{subj} {v[1]} {obj}"

def _gen_adj_obj(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    adj = rng.choice(ADJECTIVES)
    obj = rng.choice(OBJECTS)
    return f"{subj} {v[0]} {adj} {obj}"

def _gen_past_adj(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    adj = rng.choice(ADJECTIVES)
    obj = rng.choice(OBJECTS)
    return f"{subj} {v[1]} {adj} {obj}"

def _gen_adv_sv(rng):
    subj = rng.choice(SUBJECTS)
    adv = rng.choice(ADVERBS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    return f"{subj} {adv} {v[0]} {obj}"

def _gen_past_adv(rng):
    subj = rng.choice(SUBJECTS)
    adv = rng.choice(ADVERBS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    return f"{subj} {adv} {v[1]} {obj}"

def _gen_prep(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    prep = rng.choice(PREPS)
    return f"{subj} {v[0]} {obj} {prep}"

def _gen_past_prep(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    prep = rng.choice(PREPS)
    return f"{subj} {v[1]} {obj} {prep}"

def _gen_because(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    reason = rng.choice(CONJUNCTIONS_BECAUSE)
    return f"{subj} {v[1]} {obj} {reason}"

def _gen_although(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    obj = rng.choice(OBJECTS)
    clause = rng.choice(CONJUNCTIONS_ALTHOUGH)
    return f"{clause} {subj} {v[1]} {obj}"

def _gen_adj_prep(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    adj = rng.choice(ADJECTIVES)
    obj = rng.choice(OBJECTS)
    prep = rng.choice(PREPS)
    return f"{subj} {v[0]} {adj} {obj} {prep}"

def _gen_past_adj_prep(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    adj = rng.choice(ADJECTIVES)
    obj = rng.choice(OBJECTS)
    prep = rng.choice(PREPS)
    return f"{subj} {v[1]} {adj} {obj} {prep}"

def _gen_adv_adj(rng):
    subj = rng.choice(SUBJECTS)
    adv = rng.choice(ADVERBS)
    v = rng.choice(VERBS_PRESENT)
    adj = rng.choice(ADJECTIVES)
    obj = rng.choice(OBJECTS)
    return f"{subj} {adv} {v[0]} {adj} {obj}"

def _gen_compound(rng):
    subj = rng.choice(SUBJECTS)
    v1 = rng.choice(VERBS_PRESENT)
    obj1 = rng.choice(OBJECTS)
    v2 = rng.choice(VERBS_PRESENT)
    obj2 = rng.choice(OBJECTS)
    return f"{subj} {v1[1]} {obj1} and {v2[1]} {obj2}"

def _gen_because_adj(rng):
    subj = rng.choice(SUBJECTS)
    v = rng.choice(VERBS_PRESENT)
    adj = rng.choice(ADJECTIVES)
    obj = rng.choice(OBJECTS)
    reason = rng.choice(CONJUNCTIONS_BECAUSE)
    return f"{subj} {v[1]} {adj} {obj} {reason}"

GENERATORS = [
    _gen_simple_sv, _gen_past_sv, _gen_adj_obj, _gen_past_adj,
    _gen_adv_sv, _gen_past_adv, _gen_prep, _gen_past_prep,
    _gen_because, _gen_although, _gen_adj_prep, _gen_past_adj_prep,
    _gen_adv_adj, _gen_compound, _gen_because_adj,
]


def generate_sentence(rng: random.Random) -> str:
    gen = rng.choice(GENERATORS)
    return gen(rng)


# ---------------------------------------------------------------------------
# Dialect transformation rules
# ---------------------------------------------------------------------------

# DIALECT A: "Velthari" — lexical + phonetic
VELTHARI_LEXICON = {
    "the": "ze", "because": "denwhy", "although": "evenwhen",
    "and": "unt", "was": "waz", "were": "wer", "is": "iz",
    "are": "ar", "for": "vor", "with": "viz", "from": "vrom",
    "that": "zat", "this": "zis", "not": "nok", "but": "buk",
    "important": "gravmark", "critical": "kritmark",
    "required": "needven", "recently": "newpast",
    "previous": "oldpast", "limited": "boundven",
    "tight": "narrowven", "incomplete": "partven",
    "suboptimal": "lowmark", "understaffed": "fewfolk",
    "reduced": "cutven", "approaching": "nearven",
}

VELTHARI_PHONETIC = [
    (r'\bth', 'z'),       # word-initial "th" → "z"
    (r'tion\b', 'shun'),  # "-tion" → "-shun"
    (r'sion\b', 'zhun'),  # "-sion" → "-zhun"
    (r'ment\b', 'mek'),   # "-ment" → "-mek"
    (r'ness\b', 'nek'),   # "-ness" → "-nek"
    (r'ful\b', 'vul'),    # "-ful" → "-vul"
    (r'ly\b', 'li'),      # "-ly" → "-li"
    (r'ing\b', 'ven'),    # "-ing" → "-ven"
]

def transform_velthari(sentence: str, rng: random.Random) -> str:
    words = sentence.split()
    result = []
    for w in words:
        low = w.lower()
        if low in VELTHARI_LEXICON:
            result.append(VELTHARI_LEXICON[low])
        else:
            transformed = w.lower()
            for pattern, repl in VELTHARI_PHONETIC:
                transformed = re.sub(pattern, repl, transformed)
            result.append(transformed)
    return ' '.join(result)


# DIALECT B: "Korathi" — SOV reorder + morphological
KORATHI_MORPH = {
    'ed': 'ek',    # past tense
    'es': 'ox',    # plural/3rd person
    's': 'xi',     # plural (word-final s after consonant)
}

def transform_korathi(sentence: str, rng: random.Random) -> str:
    words = sentence.lower().split()

    # Morphological changes
    new_words = []
    for w in words:
        if w.endswith('ed') and len(w) > 3:
            w = w[:-2] + 'ek'
        elif w.endswith('ing') and len(w) > 4:
            w = w[:-3] + 'anu'
        elif w.endswith('es') and len(w) > 3:
            w = w[:-2] + 'ox'
        elif w.endswith('ly') and len(w) > 3:
            w = w[:-2] + 'mo'
        elif w.endswith('tion') and len(w) > 5:
            w = w[:-4] + 'tanu'
        new_words.append(w)

    # SOV reorder: find verb and move object before it
    # Simple heuristic: if sentence has "subj verb obj" pattern,
    # rearrange to "subj obj verb"
    if len(new_words) >= 3:
        # Find the likely verb position (after "the X" subject)
        verb_idx = None
        for i in range(1, len(new_words)):
            if new_words[i] not in ('the', 'ze', 'a', 'an') and i > 0:
                prev = new_words[i-1] if i > 0 else ''
                if prev not in ('the', 'ze', 'a', 'an', 'and', 'unt'):
                    verb_idx = i
                    break
        if verb_idx and verb_idx < len(new_words) - 1:
            subj = new_words[:verb_idx]
            verb = new_words[verb_idx]
            rest = new_words[verb_idx+1:]
            new_words = subj + rest + [verb]

    return ' '.join(new_words)


# DIALECT C: "Nelvosi" — agglutination + lexical
NELVOSI_PREPS = {
    'in': 'n', 'at': 't', 'for': 'f', 'on': 'n',
    'to': 'tu', 'by': 'b', 'of': 'v', 'from': 'fr',
    'with': 'w', 'under': 'un', 'over': 'ov', 'through': 'thr',
    'across': 'acr', 'within': 'wn', 'during': 'dur', 'after': 'aft',
    'before': 'bef',
}

NELVOSI_LEXICON = {
    "the": "el", "because": "porke", "although": "masque",
    "and": "ey", "was": "era", "were": "eran",
    "is": "es", "are": "son", "not": "no", "but": "pero",
    "required": "requeren", "recently": "recen",
    "previous": "previo", "limited": "limiten",
    "tight": "estrech", "incomplete": "incomple",
    "suboptimal": "subopti", "understaffed": "pocogente",
    "reduced": "reducen", "approaching": "acercan",
}

def transform_nelvosi(sentence: str, rng: random.Random) -> str:
    words = sentence.lower().split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in NELVOSI_PREPS and i + 1 < len(words):
            prefix = NELVOSI_PREPS[w]
            next_w = words[i+1]
            if next_w in ('the', 'a', 'an'):
                if i + 2 < len(words):
                    result.append(prefix + words[i+2])
                    i += 3
                    continue
                else:
                    result.append(prefix + next_w)
                    i += 2
                    continue
            else:
                result.append(prefix + next_w)
                i += 2
                continue
        elif w in NELVOSI_LEXICON:
            result.append(NELVOSI_LEXICON[w])
        else:
            # Morphological: -tion → -cion, -ment → -mento, -ly → -mente
            transformed = w
            if transformed.endswith('tion'):
                transformed = transformed[:-4] + 'cion'
            elif transformed.endswith('ment'):
                transformed = transformed[:-4] + 'mento'
            elif transformed.endswith('ly'):
                transformed = transformed[:-2] + 'mente'
            elif transformed.endswith('ing'):
                transformed = transformed[:-3] + 'ando'
            elif transformed.endswith('ed') and len(transformed) > 3:
                transformed = transformed[:-2] + 'ado'
            result.append(transformed)
        i += 1
    return ' '.join(result)


# DIALECT D: "Drakmori" — reverse adjective placement + heavy phonetic
DRAKMORI_PHONETIC = [
    (r'ph', 'f'),
    (r'ck', 'k'),
    (r'ght', 't'),
    (r'ous\b', 'uz'),
    (r'ive\b', 'iv'),
    (r'ble\b', 'bl'),
    (r'ful\b', 'ful'),
    (r'ness\b', 'nis'),
    (r'ment\b', 'mnt'),
    (r'tion\b', 'tsn'),
    (r'sion\b', 'zsn'),
    (r'ally\b', 'ali'),
    (r'ical\b', 'ikl'),
    (r'ence\b', 'ens'),
    (r'ance\b', 'ans'),
    (r'ly\b', 'ly'),
    (r'ing\b', 'ng'),
    (r'ed\b', 'dt'),
]

DRAKMORI_LEXICON = {
    "the": "dra", "because": "kaus", "although": "trotz",
    "and": "und", "was": "var", "were": "varen",
    "is": "ist", "are": "sind", "not": "nit",
    "but": "dok", "for": "fur", "with": "mit",
    "from": "von", "in": "inn", "at": "bei",
    "required": "brauk", "recently": "neulik",
    "previous": "vorig", "limited": "begrenz",
    "tight": "eng", "incomplete": "unvoll",
    "suboptimal": "untergut", "understaffed": "untermann",
    "reduced": "vermind", "approaching": "nahend",
}

def transform_drakmori(sentence: str, rng: random.Random) -> str:
    words = sentence.lower().split()

    # Lexical + phonetic
    new_words = []
    for w in words:
        if w in DRAKMORI_LEXICON:
            new_words.append(DRAKMORI_LEXICON[w])
        else:
            transformed = w
            for pattern, repl in DRAKMORI_PHONETIC:
                transformed = re.sub(pattern, repl, transformed)
            new_words.append(transformed)

    # Adjective reversal: move adjectives after the noun they precede
    result = []
    i = 0
    while i < len(new_words):
        w = new_words[i]
        if w in [transform_drakmori_word(a) for a in ADJECTIVES]:
            if i + 1 < len(new_words):
                result.append(new_words[i+1])
                result.append(w)
                i += 2
                continue
        result.append(w)
        i += 1

    return ' '.join(result)

def transform_drakmori_word(w):
    if w in DRAKMORI_LEXICON:
        return DRAKMORI_LEXICON[w]
    transformed = w
    for pattern, repl in DRAKMORI_PHONETIC:
        transformed = re.sub(pattern, repl, transformed)
    return transformed


# DIALECT E: "Quilmari" — vowel shift + suffix changes + determiner drop
QUILMARI_VOWEL_MAP = str.maketrans('aeiou', 'eioua')

QUILMARI_LEXICON = {
    "the": "",  # determiner drop
    "a": "", "an": "",
    "because": "kwabec", "although": "malgre",
    "and": "ak", "was": "fui", "were": "fuir",
    "is": "esti", "are": "estir", "not": "nep",
    "but": "sed", "for": "por", "with": "kun",
    "from": "dek", "in": "en", "at": "ad",
    "required": "bezonit", "recently": "novtim",
    "previous": "antik", "limited": "limigit",
    "tight": "strikt", "incomplete": "nekomplet",
    "suboptimal": "suboptim", "understaffed": "mankhom",
    "reduced": "reduktit", "approaching": "proksimig",
}

def transform_quilmari(sentence: str, rng: random.Random) -> str:
    words = sentence.lower().split()
    result = []
    for w in words:
        if w in QUILMARI_LEXICON:
            replacement = QUILMARI_LEXICON[w]
            if replacement:
                result.append(replacement)
            # else: determiner dropped
        else:
            # Vowel shift
            transformed = w.translate(QUILMARI_VOWEL_MAP)
            # Suffix changes
            if transformed.endswith('id'):
                transformed = transformed[:-2] + 'it'
            elif transformed.endswith('ng'):
                transformed = transformed[:-2] + 'nt'
            elif transformed.endswith('ly'):
                transformed = transformed[:-2] + 'im'
            elif transformed.endswith('ed') and len(transformed) > 3:
                transformed = transformed[:-2] + 'is'
            elif transformed.endswith('tsn') or transformed.endswith('tion'):
                transformed = re.sub(r'(tsn|tion)$', 'cij', transformed)
            result.append(transformed)
    return ' '.join(result)


# ---------------------------------------------------------------------------
# Dialect registry
# ---------------------------------------------------------------------------

DIALECTS = {
    "velthari": transform_velthari,
    "korathi": transform_korathi,
    "nelvosi": transform_nelvosi,
    "drakmori": transform_drakmori,
    "quilmari": transform_quilmari,
}


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def generate_row(rng: random.Random, row_id: int) -> dict:
    original = generate_sentence(rng)
    dialect = rng.choice(list(DIALECTS.keys()))
    transform_fn = DIALECTS[dialect]
    transformed = transform_fn(original, rng)

    return {
        "id": row_id,
        "dialect": dialect,
        "transformed": transformed,
        "original": original,
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
        writer = csv.DictWriter(f, fieldnames=["id", "dialect", "transformed", "original"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    dialects = Counter(r["dialect"] for r in rows)
    avg_orig = sum(len(r["original"]) for r in rows) / len(rows)
    avg_trans = sum(len(r["transformed"]) for r in rows) / len(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Dialects: {dict(sorted(dialects.items()))}")
    print(f"  Avg original length: {avg_orig:.0f} chars")
    print(f"  Avg transformed length: {avg_trans:.0f} chars")

    # Show samples
    for dialect in sorted(DIALECTS.keys()):
        sample = next(r for r in rows if r["dialect"] == dialect)
        print(f"\n  [{dialect}]")
        print(f"    Original:    {sample['original']}")
        print(f"    Transformed: {sample['transformed']}")


if __name__ == "__main__":
    main()
