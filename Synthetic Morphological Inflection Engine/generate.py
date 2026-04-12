"""
generate.py — Synthetic Contextual Sequence Transduction (v2 — harder)

Changes from v1:
  - 5th control code: VOICE (ACT, MID, PAS, CAU) → combos per root: 720
  - Voice suffixes with tense-dependent allomorphy
  - 8 portmanteau patterns (up from 3), including cross-voice combos
  - Second-layer sandhi: context modifies voice suffix separately
  - 30 double-irregular roots (voice-specific stem changes)
  - 15 suppletive roots (completely replaced stems for specific combos)
  - Reduced to 30,000 samples (train coverage ~6% of input space)
  - Noise increased to 10%
  - Theoretical max char accuracy ~0.90

Output: raw_data/data.csv (30,000 rows)
"""

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

SEED = 42

CONSONANTS = list("bdfgklmnprstvz")
FRONT_VOWELS = list("ei")
BACK_VOWELS = list("aou")
ALL_VOWELS = set(FRONT_VOWELS + BACK_VOWELS)
ALL_CHARS = CONSONANTS + FRONT_VOWELS + BACK_VOWELS

VOICELESS = {"f", "k", "p", "s", "t"}
VOICED_STOPS = {"b", "d", "g", "v", "z"}
VOICE_MAP = {"k": "g", "t": "d", "p": "b", "s": "z", "f": "v"}
DEVOICE_MAP = {v: k for k, v in VOICE_MAP.items()}

LENITION_MAP = {"t": "s", "k": "h", "d": "z", "n": "l"}

TENSES = ["PRS", "PST", "FUT", "PRF", "HAB"]
NUMBERS = ["SG", "DU", "PL"]
CASES = ["NOM", "ACC", "DAT", "GEN"]
MOODS = ["IND", "SBJ", "IMP"]
VOICES = ["ACT", "MID", "PAS", "CAU"]

TENSE_SUFFIX = {
    "PRS": ("a", "e"),
    "PST": ("tok", "tek"),
    "FUT": ("rul", "ril"),
    "PRF": ("nash", "nesh"),
    "HAB": ("dur", "dir"),
}
NUMBER_SUFFIX = {
    "SG": ("", ""),
    "DU": ("va", "ve"),
    "PL": ("on", "en"),
}
CASE_SUFFIX = {
    "NOM": ("", ""),
    "ACC": ("ot", "et"),
    "DAT": ("ka", "ke"),
    "GEN": ("un", "in"),
}
MOOD_SUFFIX = {
    "IND": ("", ""),
    "SBJ": ("sa", "se"),
    "IMP": ("do", "de"),
}

VOICE_SUFFIX_BASE = {
    "ACT": ("", ""),
    "MID": ("lem", "lim"),
    "PAS": ("naf", "nef"),
    "CAU": ("tog", "tig"),
}

VOICE_TENSE_ALLOMORPH = {
    ("PAS", "PST"): ("nav", "niv"),
    ("PAS", "FUT"): ("nuf", "nif"),
    ("CAU", "FUT"): ("trug", "trig"),
    ("CAU", "PRF"): ("tov", "tiv"),
    ("MID", "PRF"): ("lum", "lum"),
    ("MID", "PST"): ("lam", "lam"),
}

PORTMANTEAU = {
    ("PST", "PL", "GEN"): ("tokin", "tekin"),
    ("PRF", "DU", "DAT"): ("nashvak", "neshvek"),
    ("HAB", "PL", "ACC"): ("durot", "diret"),
}

VOICE_PORTMANTEAU = {
    ("FUT", "PL", "ACC", "PAS"): ("rulonav", "riloniv"),
    ("PST", "SG", "NOM", "CAU"): ("toktog", "tektig"),
    ("PRF", "PL", "GEN", "MID"): ("nashonlim", "neshenlim"),
    ("HAB", "DU", "DAT", "PAS"): ("durvakan", "dirveken"),
    ("PRS", "PL", "ACC", "CAU"): ("onotog", "enetig"),
}

VOWEL_SHIFT = {"a": "o", "e": "i", "o": "u", "i": "e", "u": "a"}


def get_vowel_class(word):
    for ch in reversed(word):
        if ch in FRONT_VOWELS:
            return "front"
        if ch in BACK_VOWELS:
            return "back"
    return "back"


def hi(word):
    return 0 if get_vowel_class(word) == "back" else 1


def shift_last_vowel(word):
    chars = list(word)
    for i in range(len(chars) - 1, -1, -1):
        if chars[i] in VOWEL_SHIFT:
            chars[i] = VOWEL_SHIFT[chars[i]]
            return "".join(chars)
    return word


def geminate_final(word):
    if word and word[-1] not in ALL_VOWELS:
        return word + word[-1]
    return word


def voice_initial(word):
    if word and word[0] in VOICE_MAP:
        return VOICE_MAP[word[0]] + word[1:]
    elif word and word[0] in DEVOICE_MAP:
        return DEVOICE_MAP[word[0]] + word[1:]
    return word


def apply_ablaut(word):
    chars = list(word)
    for i in range(len(chars)):
        if chars[i] in VOWEL_SHIFT:
            chars[i] = VOWEL_SHIFT[chars[i]]
    return "".join(chars)


def prefix_stem(word, harmony_class):
    pfx = "u" if harmony_class == "back" else "i"
    return pfx + word


def join_morphemes(stem, suffix):
    if not suffix:
        return stem

    s = list(stem)
    x = list(suffix)

    if s and x and s[-1] in ALL_VOWELS and x[0] in ALL_VOWELS:
        s.pop()

    if s and x:
        last = s[-1]
        first = x[0]
        if last in VOICELESS and first in VOICED_STOPS and last in VOICE_MAP:
            s[-1] = VOICE_MAP[last]
        elif last in VOICED_STOPS and first in VOICELESS:
            s[-1] = DEVOICE_MAP[last]

    combined = s + x

    result = []
    cons_run = 0
    ep = "a" if get_vowel_class(stem) == "back" else "e"
    for ch in combined:
        if ch not in ALL_VOWELS:
            cons_run += 1
            if cons_run == 3:
                result.append(ep)
                cons_run = 1
        else:
            cons_run = 0
        result.append(ch)

    return "".join(result)


def get_sandhi_type(context_word):
    last = context_word[-1]
    if last in FRONT_VOWELS:
        return "lenition"
    elif last in ("m", "n"):
        return "nasal"
    elif last in VOICELESS:
        return "fortition"
    return "none"


def get_voice_sandhi(context_word, voice):
    """Second-layer sandhi: context modifies the voice suffix."""
    if voice == "ACT":
        return "none"
    first = context_word[0]
    if first in ALL_VOWELS and voice in ("MID", "PAS", "CAU"):
        return "g_prepend"
    last = context_word[-1]
    if last in ("m", "n") and voice == "PAS":
        return "nasalize"
    if first in VOICELESS and voice == "CAU":
        return "devoice_voice_suf"
    return "none"


def apply_sandhi(suffix, sandhi_type, harmony_class):
    if not suffix:
        return suffix
    if sandhi_type == "lenition":
        if suffix[0] in LENITION_MAP:
            return LENITION_MAP[suffix[0]] + suffix[1:]
    elif sandhi_type == "nasal":
        prefix = "an" if harmony_class == "back" else "en"
        return prefix + suffix
    elif sandhi_type == "fortition":
        if suffix[0] in DEVOICE_MAP:
            return DEVOICE_MAP[suffix[0]] + suffix[1:]
    return suffix


def apply_voice_sandhi(suffix, sandhi_type, harmony_class):
    """Apply second-layer sandhi to voice suffix."""
    if not suffix:
        return suffix
    if sandhi_type == "g_prepend":
        return "g" + suffix
    elif sandhi_type == "nasalize":
        if suffix[0] == "n":
            return "m" + suffix[1:]
    elif sandhi_type == "devoice_voice_suf":
        if suffix[0] in DEVOICE_MAP:
            return DEVOICE_MAP[suffix[0]] + suffix[1:]
    return suffix


def inflect(root, tense, number, case_f, mood, voice,
            is_irregular, is_ablaut, is_double_irreg,
            suppletive_map, context_word=None):
    stem = root

    if root in suppletive_map:
        key = (tense, voice)
        if key in suppletive_map[root]:
            stem = suppletive_map[root][key]

    if is_irregular:
        if tense == "PST":
            stem = shift_last_vowel(stem)
        elif tense == "PRF":
            stem = geminate_final(stem)
        elif tense == "FUT":
            stem = voice_initial(stem)

    if is_double_irreg:
        if voice == "PAS":
            stem = prefix_stem(stem, get_vowel_class(stem))
        if voice == "CAU" and tense == "PST":
            stem = apply_ablaut(stem)

    if is_ablaut and tense == "PST" and number == "PL":
        stem = apply_ablaut(stem)

    sandhi_type = get_sandhi_type(context_word) if context_word else "none"
    v_sandhi = get_voice_sandhi(context_word, voice) if context_word else "none"

    vp_key = (tense, number, case_f, voice)
    if vp_key in VOICE_PORTMANTEAU:
        suf = VOICE_PORTMANTEAU[vp_key][hi(stem)]
        suf = apply_sandhi(suf, sandhi_type, get_vowel_class(stem))
        form = join_morphemes(stem, suf)
        mood_suf = MOOD_SUFFIX[mood][hi(form)]
        form = join_morphemes(form, mood_suf)
        return form

    port_key = (tense, number, case_f)
    if port_key in PORTMANTEAU and voice == "ACT":
        suf = PORTMANTEAU[port_key][hi(stem)]
        suf = apply_sandhi(suf, sandhi_type, get_vowel_class(stem))
        form = join_morphemes(stem, suf)
        mood_suf = MOOD_SUFFIX[mood][hi(form)]
        form = join_morphemes(form, mood_suf)
        return form

    form = stem
    first_suffix = True
    for table, key in [(TENSE_SUFFIX, tense), (NUMBER_SUFFIX, number),
                       (CASE_SUFFIX, case_f)]:
        suf = table[key][hi(form)]
        if suf:
            if first_suffix:
                suf = apply_sandhi(suf, sandhi_type, get_vowel_class(form))
                first_suffix = False
            form = join_morphemes(form, suf)

    vt_key = (voice, tense)
    if vt_key in VOICE_TENSE_ALLOMORPH:
        voice_suf = VOICE_TENSE_ALLOMORPH[vt_key][hi(form)]
    else:
        voice_suf = VOICE_SUFFIX_BASE[voice][hi(form)]

    if voice_suf:
        voice_suf = apply_voice_sandhi(voice_suf, v_sandhi, get_vowel_class(form))
        if first_suffix:
            voice_suf = apply_sandhi(voice_suf, sandhi_type, get_vowel_class(form))
            first_suffix = False
        form = join_morphemes(form, voice_suf)

    mood_suf = MOOD_SUFFIX[mood][hi(form)]
    if mood_suf:
        form = join_morphemes(form, mood_suf)

    return form


def add_noise(form, rng):
    if len(form) < 2:
        return form
    chars = list(form)
    pos = rng.randint(0, len(chars))
    chars[pos] = rng.choice(ALL_CHARS)
    return "".join(chars)


def generate_roots(rng, n=500):
    roots = set()
    while len(roots) < n:
        nsyl = rng.choice([1, 2, 3], p=[0.15, 0.50, 0.35])
        r = []
        for _ in range(nsyl):
            r.append(rng.choice(CONSONANTS))
            r.append(rng.choice(FRONT_VOWELS + BACK_VOWELS))
            if rng.random() < 0.35:
                r.append(rng.choice(CONSONANTS))
        word = "".join(r)
        if 3 <= len(word) <= 8 and word not in roots:
            roots.add(word)
    return sorted(roots)


def generate_context_words(rng, n=100):
    words = set()
    while len(words) < n:
        w = [rng.choice(CONSONANTS), rng.choice(FRONT_VOWELS + BACK_VOWELS)]
        if rng.random() < 0.5:
            w.append(rng.choice(CONSONANTS))
        word = "".join(w)
        if word not in words:
            words.add(word)
    return sorted(words)


def build_suppletive_map(rng, roots, n=15):
    """15 roots get completely different stems for specific (tense, voice) combos."""
    chosen = rng.choice(roots, size=n, replace=False)
    smap = {}
    all_tv = [(t, v) for t in TENSES for v in VOICES if v != "ACT"]
    for root in chosen:
        alt_roots = generate_roots(rng, n=4)
        n_combos = rng.randint(2, 5)
        idxs = rng.choice(len(all_tv), size=n_combos, replace=False)
        smap[root] = {}
        for ci, idx in enumerate(idxs):
            smap[root][all_tv[idx]] = alt_roots[ci % len(alt_roots)]
    return smap


def main():
    rng = np.random.RandomState(SEED)

    roots = generate_roots(rng, n=500)
    context_words = generate_context_words(rng, n=100)

    irregular = set(rng.choice(roots, size=60, replace=False))
    remaining = [r for r in roots if r not in irregular]
    ablaut = set(rng.choice(remaining, size=40, replace=False))
    remaining2 = [r for r in remaining if r not in ablaut]
    double_irreg = set(rng.choice(remaining2, size=30, replace=False))

    suppletive_map = build_suppletive_map(rng, roots, n=15)

    all_combos = list(itertools.product(
        roots, TENSES, NUMBERS, CASES, MOODS, VOICES
    ))
    indices = np.arange(len(all_combos))
    rng.shuffle(indices)
    selected = [all_combos[i] for i in indices[:30000]]

    rows = []
    for i, (root, tense, number, case_f, mood, voice) in enumerate(selected):
        ctx = rng.choice(context_words)

        target = inflect(
            root, tense, number, case_f, mood, voice,
            root in irregular, root in ablaut, root in double_irreg,
            suppletive_map, context_word=ctx
        )

        if rng.random() < 0.10:
            target = add_noise(target, rng)

        source = f"{ctx} {root} {tense} {number} {case_f} {mood} {voice}"
        rows.append({"sample_id": i, "source": source, "target": target})

    df = pd.DataFrame(rows)

    out_dir = Path("raw_data")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "data.csv", index=False)

    print(f"Generated {len(df)} samples")
    print(f"Total possible combos: {len(all_combos)} "
          f"(500 roots x {len(TENSES)}T x {len(NUMBERS)}N x "
          f"{len(CASES)}C x {len(MOODS)}M x {len(VOICES)}V)")
    print(f"Coverage: {len(df)/len(all_combos)*100:.1f}%")
    print(f"Context words: {len(context_words)}")
    print(f"Unique sources: {df['source'].nunique()}")
    print(f"Unique targets: {df['target'].nunique()}")
    avg_len = df["target"].str.len().mean()
    print(f"Target length: min={df['target'].str.len().min()}, "
          f"avg={avg_len:.1f}, max={df['target'].str.len().max()}")

    sandhi_counts = {"lenition": 0, "nasal": 0, "fortition": 0, "none": 0}
    for _, row in df.iterrows():
        ctx = row["source"].split()[0]
        sandhi_counts[get_sandhi_type(ctx)] += 1
    print(f"\nSandhi distribution: {sandhi_counts}")

    voice_counts = {}
    for _, row in df.iterrows():
        v = row["source"].split()[-1]
        voice_counts[v] = voice_counts.get(v, 0) + 1
    print(f"Voice distribution: {voice_counts}")

    print(f"\nSample data:")
    print(df.head(15).to_string())


if __name__ == "__main__":
    main()
