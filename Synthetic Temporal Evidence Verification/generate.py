"""
Generate synthetic temporal evidence verification data.

Domain: RAG — claim verification against a temporal knowledge base with
source credibility decay, contradictions, and adversarial sources.

Hidden structure:
  - 600 documents about fictional nation "Velantra" dated 2010-2025
  - 30 sources with hidden topic-dependent credibility that changes over time
  - Documents may correct, retract, or partially modify earlier documents
  - ~15% of sources copy from a "parent" source (correlated noise)
  - Claims verified as SUPPORTED / REFUTED / PARTIALLY_TRUE / INSUFFICIENT_EVIDENCE
  - ~12% of verdicts have irreducible noise
"""

import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

# ---- Fictional world building ----
TOPICS = [
    "history", "geography", "politics", "economy",
    "culture", "science", "military", "education",
]

SOURCES = [f"Source-{chr(65 + i // 3)}{i % 3 + 1}" for i in range(30)]

VERDICTS = ["SUPPORTED", "REFUTED", "PARTIALLY_TRUE", "INSUFFICIENT_EVIDENCE"]

LOCATIONS = [
    "Velanthos", "Port Kessara", "Mount Drevna", "Lake Almori",
    "Starveil Plateau", "Noctis Valley", "Orinath Basin", "Cape Ithren",
    "Thornmere", "Eastgate", "Redstone District", "Sunspire",
    "Frostholm", "Coral Bay", "Windreach", "Old Quarter",
]

PEOPLE = [
    "Chancellor Marek", "Admiral Tova", "Professor Ilien", "Minister Darvo",
    "General Koreth", "Director Sanna", "Historian Brael", "Senator Voss",
    "Scientist Ulren", "Governor Thane", "Ambassador Quiris", "Elder Meska",
    "Reformer Aldis", "Commander Joval", "Scholar Penna", "Architect Rizan",
]

EVENTS = [
    "the Great Accord", "the Tidal Reforms", "the Starfall Campaign",
    "the Founding of the Maritime Council", "the Eastern Expansion",
    "the Cultural Renaissance", "the Border Treaty", "the Industrial Shift",
    "the Plague of Shadows", "the Velantra Compact", "the Mineral Rush",
    "the Naval Blockade", "the Education Decree", "the Trade Embargo",
    "the Scientific Expedition", "the Agricultural Reform",
]

CLAIM_TEMPLATES = [
    "{person} initiated {event} in {year}.",
    "The population of {location} exceeded {number} during {event}.",
    "{event} resulted in the establishment of a new institution in {location}.",
    "{person} governed {location} from {year} until the end of {event}.",
    "Research conducted in {location} by {person} contributed to {event}.",
    "The {topic} policies enacted during {event} affected {location} significantly.",
    "{location} was the primary site of {event}, overseen by {person}.",
    "According to official records, {person} opposed {event} in {year}.",
    "The economic impact of {event} on {location} was measured at {number} units.",
    "{person}'s role in {event} was documented extensively by scholars in {location}.",
]

DOC_TEMPLATES = [
    "In {year}, {source} reported that {person} was directly involved in {event} at {location}. "
    "The {topic} implications were substantial, affecting the region for years to come. "
    "{detail1} {detail2}",

    "According to {source} ({year}), the {topic} situation in {location} shifted dramatically "
    "following {event}. {person} played a central role in these developments. "
    "{detail1} {detail2}",

    "{source} published findings in {year} suggesting that {event} "
    "had a {topic}-related impact on {location}. {person} was cited as a key figure. "
    "{detail1} {detail2}",

    "A {year} report by {source} examined the {topic} consequences of {event} in {location}. "
    "The investigation revealed {person}'s involvement in the matter. "
    "{detail1} {detail2}",
]

CORRECTION_PHRASES = [
    "Contrary to earlier reports, ",
    "Correcting a previous assessment, ",
    "Upon further review, it was found that ",
    "New evidence suggests that, unlike prior claims, ",
    "Revised analysis indicates that ",
]

DETAIL_POOL = [
    "Infrastructure investments increased by a notable margin during this period.",
    "Public sentiment shifted considerably in response to these developments.",
    "Trade routes were redirected as a consequence of these changes.",
    "Diplomatic relations were strained following the incident.",
    "Technological advancements accelerated in the aftermath.",
    "Agricultural output in the region declined temporarily.",
    "Educational reforms were introduced to address the situation.",
    "Military deployments were authorized as a precautionary measure.",
    "Cultural institutions were established to commemorate the events.",
    "Environmental assessments were conducted in the affected areas.",
    "Population migration patterns were altered significantly.",
    "Financial markets reacted with increased volatility.",
]


def _det_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def main():
    rng = np.random.RandomState(42)
    N_DOCS = 600
    N_CLAIMS = 5200
    N_SOURCES = 30
    N_TOPICS = len(TOPICS)

    # ---- Source credibility profiles: topic-dependent, time-varying ----
    # source_cred[source_idx][topic_idx] = (base_cred, decay_rate)
    source_cred = {}
    for si in range(N_SOURCES):
        topic_profile = {}
        for ti in range(N_TOPICS):
            base = rng.uniform(0.3, 0.95)
            decay = rng.uniform(-0.03, 0.03)  # negative = improves over time
            topic_profile[TOPICS[ti]] = (base, decay)
        source_cred[si] = topic_profile

    # ~5 adversarial sources: high base credibility but inject errors ~20% of the time
    adversarial_sources = set(rng.choice(N_SOURCES, 5, replace=False).tolist())

    # ~15% of sources copy from a parent source (correlated noise)
    parent_map = {}
    copycat_sources = set(rng.choice(
        [s for s in range(N_SOURCES) if s not in adversarial_sources],
        size=int(N_SOURCES * 0.15), replace=False
    ).tolist())
    for cs in copycat_sources:
        possible_parents = [s for s in range(N_SOURCES) if s != cs and s not in copycat_sources]
        parent_map[cs] = rng.choice(possible_parents)

    def _get_credibility(source_idx, topic, year):
        base, decay = source_cred[source_idx][topic]
        years_from_start = year - 2010
        cred = base + decay * years_from_start
        return max(0.05, min(0.98, cred))

    # ---- Generate documents ----
    documents = []
    # Each document covers one topic, one location, one person, one event
    for doc_id in range(N_DOCS):
        year = int(rng.choice(range(2010, 2026)))
        source_idx = rng.randint(0, N_SOURCES)
        source_name = SOURCES[source_idx]
        topic = TOPICS[rng.randint(0, N_TOPICS)]
        location = LOCATIONS[rng.randint(0, len(LOCATIONS))]
        person = PEOPLE[rng.randint(0, len(PEOPLE))]
        event = EVENTS[rng.randint(0, len(EVENTS))]

        cred = _get_credibility(source_idx, topic, year)

        # Is this a correction of an earlier doc?
        is_correction = rng.random() < 0.15 and doc_id > 50
        correction_target = None
        if is_correction:
            earlier_docs = [d for d in documents if d["year"] < year and d["topic"] == topic]
            if earlier_docs:
                correction_target = rng.choice(len(earlier_docs))
                target_doc = earlier_docs[correction_target]
                # Temporal trap: ~20% of corrections are WRONG (introduce errors)
                is_wrong_correction = rng.random() < 0.20

        # Is this a retraction?
        is_retraction = rng.random() < 0.05 and doc_id > 100
        retraction_target = None
        if is_retraction and not is_correction:
            earlier_docs = [d for d in documents if d["year"] < year and d["topic"] == topic]
            if earlier_docs:
                retraction_target = rng.choice(len(earlier_docs))

        # Adversarial source: occasionally inject factual error
        has_adversarial_error = (source_idx in adversarial_sources and rng.random() < 0.20)

        # Copycat source: copy content structure from parent
        is_copycat = source_idx in copycat_sources

        detail1 = DETAIL_POOL[rng.randint(0, len(DETAIL_POOL))]
        detail2 = DETAIL_POOL[rng.randint(0, len(DETAIL_POOL))]

        tmpl = DOC_TEMPLATES[rng.randint(0, len(DOC_TEMPLATES))]
        text = tmpl.format(
            year=year, source=source_name, person=person, event=event,
            location=location, topic=topic, detail1=detail1, detail2=detail2,
        )

        if is_correction and correction_target is not None:
            prefix = CORRECTION_PHRASES[rng.randint(0, len(CORRECTION_PHRASES))]
            text = prefix + text

        if is_retraction and retraction_target is not None:
            text = f"RETRACTION NOTICE: {source_name} ({year}) hereby retracts previous claims regarding {event} in {location}. " + text

        if has_adversarial_error:
            wrong_person = PEOPLE[rng.randint(0, len(PEOPLE))]
            wrong_event = EVENTS[rng.randint(0, len(EVENTS))]
            text = text.replace(person, wrong_person, 1)
            person = wrong_person

        # Generate "fact record" for this document
        fact_number = rng.randint(100, 9999)

        documents.append({
            "doc_id": doc_id,
            "year": year,
            "source_idx": source_idx,
            "source_name": source_name,
            "topic": topic,
            "location": location,
            "person": person,
            "event": event,
            "fact_number": fact_number,
            "text": text,
            "credibility": round(cred, 3),
            "is_correction": is_correction and correction_target is not None,
            "is_retraction": is_retraction and retraction_target is not None,
            "has_adversarial_error": has_adversarial_error,
            "is_copycat": is_copycat,
        })

    docs_df = pd.DataFrame(documents)

    # ---- Generate claims and verdicts ----
    claims = []
    for claim_id in range(N_CLAIMS):
        topic = TOPICS[rng.randint(0, N_TOPICS)]
        location = LOCATIONS[rng.randint(0, len(LOCATIONS))]
        person = PEOPLE[rng.randint(0, len(PEOPLE))]
        event = EVENTS[rng.randint(0, len(EVENTS))]
        claim_year = int(rng.choice(range(2012, 2026)))
        fact_number = rng.randint(100, 9999)

        tmpl = CLAIM_TEMPLATES[rng.randint(0, len(CLAIM_TEMPLATES))]
        claim_text = tmpl.format(
            person=person, event=event, location=location,
            year=rng.randint(2010, 2026), number=rng.randint(1000, 50000),
            topic=topic,
        )

        # Relevant docs: same topic AND at least 1 of (location, person, event) match
        # AND dated before as_of_year
        match_scores = (
            (docs_df["location"] == location).astype(int) +
            (docs_df["person"] == person).astype(int) +
            (docs_df["event"] == event).astype(int)
        )
        relevant_mask = (
            (docs_df["topic"] == topic) &
            (docs_df["year"] <= claim_year) &
            (match_scores >= 1)
        )
        relevant_docs = docs_df[relevant_mask].sort_values("year", ascending=False)
        n_relevant = len(relevant_docs)

        # Strong matches (2+ entity matches) vs weak (1 match)
        strong_mask = relevant_mask & (match_scores >= 2)
        n_strong = strong_mask.sum()

        if n_relevant == 0:
            verdict = "INSUFFICIENT_EVIDENCE"
        elif n_relevant <= 2 and n_strong == 0:
            # Only weak matches — less certainty
            if rng.random() < 0.4:
                verdict = "INSUFFICIENT_EVIDENCE"
            else:
                verdict = "PARTIALLY_TRUE"
        elif n_relevant <= 2:
            top_doc = relevant_docs.iloc[0]
            if top_doc["has_adversarial_error"]:
                verdict = rng.choice(["REFUTED", "PARTIALLY_TRUE"])
            elif top_doc["is_retraction"]:
                verdict = "REFUTED"
            elif top_doc["is_correction"]:
                if top_doc["credibility"] > 0.6:
                    verdict = rng.choice(["SUPPORTED", "PARTIALLY_TRUE"])
                else:
                    verdict = "PARTIALLY_TRUE"
            elif top_doc["credibility"] > 0.7:
                verdict = "SUPPORTED"
            elif top_doc["credibility"] > 0.4:
                verdict = "PARTIALLY_TRUE"
            else:
                verdict = rng.choice(["PARTIALLY_TRUE", "INSUFFICIENT_EVIDENCE"])
        else:
            most_recent = relevant_docs.iloc[0]
            second_recent = relevant_docs.iloc[1]

            if most_recent["is_retraction"]:
                verdict = "REFUTED"
            elif most_recent["is_correction"]:
                if most_recent["has_adversarial_error"]:
                    verdict = rng.choice(["REFUTED", "PARTIALLY_TRUE"])
                elif most_recent["credibility"] > 0.7:
                    verdict = rng.choice(["SUPPORTED", "PARTIALLY_TRUE"])
                else:
                    verdict = "PARTIALLY_TRUE"
            else:
                # Weighted credibility with recency bonus
                weighted_scores = []
                for _, d in relevant_docs.iterrows():
                    recency_w = 1.0 + 0.05 * (d["year"] - 2010)
                    direction = -1.0 if (d["is_retraction"] or d["has_adversarial_error"] or d["is_correction"]) else 1.0
                    weighted_scores.append(direction * d["credibility"] * recency_w)

                avg_score = sum(weighted_scores) / len(weighted_scores)
                n_unique_sources = relevant_docs["source_idx"].nunique()
                has_copycat = relevant_docs["is_copycat"].any()

                # Penalize single-source or copycat-dominated evidence
                if n_unique_sources <= 1:
                    avg_score *= 0.6
                if has_copycat and n_unique_sources <= 2:
                    avg_score *= 0.7

                # Strong match bonus
                if n_strong > 2:
                    avg_score *= 1.1

                # Claim hash for controlled additional variability
                claim_hash = _det_hash(f"{claim_id}_{topic}_{location}_{person}") % 100

                if avg_score > 0.5 and claim_hash > 25:
                    verdict = "SUPPORTED"
                elif avg_score > 0.25 or (avg_score > 0 and claim_hash > 60):
                    verdict = "PARTIALLY_TRUE"
                elif avg_score > -0.1:
                    verdict = rng.choice(["PARTIALLY_TRUE", "REFUTED"])
                else:
                    verdict = "REFUTED"

        # Noise: ~12% of verdicts are randomly perturbed
        if rng.random() < 0.12:
            verdict = VERDICTS[rng.randint(0, len(VERDICTS))]

        claims.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "as_of_year": claim_year,
            "topic": topic,
            "n_relevant_docs": n_relevant,
            "verdict": verdict,
        })

    claims_df = pd.DataFrame(claims)

    # ---- Write outputs ----
    out = Path("raw_data")
    out.mkdir(exist_ok=True)

    # Documents KB
    kb_cols = ["doc_id", "year", "source_name", "topic", "location", "person", "event",
               "fact_number", "text"]
    docs_df[kb_cols].to_csv(out / "knowledge_base.csv", index=False)

    # Claims with verdicts
    claims_cols = ["claim_id", "claim_text", "as_of_year", "verdict"]
    claims_df[claims_cols].to_csv(out / "claims.csv", index=False)

    # Source metadata (without credibility — that's hidden)
    source_meta = []
    for si in range(N_SOURCES):
        source_meta.append({
            "source_idx": si,
            "source_name": SOURCES[si],
            "is_adversarial": si in adversarial_sources,
            "is_copycat": si in copycat_sources,
            "parent_source": SOURCES[parent_map[si]] if si in copycat_sources else "",
        })
    source_df = pd.DataFrame(source_meta)
    source_df.to_csv(out / "source_profiles.csv", index=False)

    print(f"Documents: {len(docs_df)}")
    print(f"Claims: {len(claims_df)}")
    print(f"Sources: {N_SOURCES} ({len(adversarial_sources)} adversarial, {len(copycat_sources)} copycat)")
    print(f"Verdict distribution:")
    print(claims_df["verdict"].value_counts().to_string())
    print(f"\nRelevant docs per claim stats:")
    print(claims_df["n_relevant_docs"].describe().to_string())
    print(f"Claims with 0 relevant docs: {(claims_df['n_relevant_docs'] == 0).sum()}")
    print(f"Claims with 1-2 relevant docs: {((claims_df['n_relevant_docs'] >= 1) & (claims_df['n_relevant_docs'] <= 2)).sum()}")


if __name__ == "__main__":
    main()
