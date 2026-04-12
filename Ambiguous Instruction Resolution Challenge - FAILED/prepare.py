"""
Prepare script: transforms raw data.csv into public/ and private/ splits.

Deterministic (fixed random_state + hashlib for row-level operations).

Obfuscation pipeline (applied to every row):
  1. Entity name replacement (~40%): replace person names with codes (P1, P2, P3)
  2. Department anonymization (~35%): replace department names with codes (D1, D2, D3)
  3. Scenario sentence shuffling (~30%): reorder 1-2 sentences in the scenario
  4. Paraphrase option text (~25%): synonym substitution in option wording
  5. Add a 5th distractor option (~20%): inject "None of the above" or a new plausible option
  6. Truncate scenario (~15%): remove the last sentence of the scenario
  7. Insert red-herring sentence (~25%): add an irrelevant sentence to the scenario

Split: stratified 80/20 by answer label.
"""

from pathlib import Path
import hashlib
import random as _rnd
import re

import pandas as pd
from sklearn.model_selection import train_test_split


REDHERRING_SENTENCES = [
    "The office Wi-Fi was unusually slow that morning.",
    "A facilities maintenance request had been submitted the previous week.",
    "The cafeteria was serving a new menu that day.",
    "Several employees had recently completed their annual training certification.",
    "The parking lot construction was expected to finish by month-end.",
    "A company-wide email about the updated dress code was sent on Monday.",
    "The IT department was rolling out new laptops to senior staff.",
    "The quarterly town hall was postponed due to a scheduling conflict.",
    "A fire drill had been conducted the previous afternoon.",
    "The building's elevator was under maintenance for routine inspection.",
    "New ergonomic chairs had been ordered for the open-plan area.",
    "The company newsletter featured an article about sustainability initiatives.",
]

PARAPHRASE_MAP = {
    "reviewed": ["examined", "assessed", "evaluated", "inspected"],
    "approved": ["authorized", "endorsed", "sanctioned", "greenlit"],
    "submitted": ["delivered", "filed", "turned in", "handed in"],
    "confirmed": ["verified", "acknowledged", "affirmed", "validated"],
    "completed": ["finished", "concluded", "wrapped up", "finalized"],
    "assigned": ["delegated", "allocated", "designated", "tasked"],
    "scheduled": ["arranged", "planned", "set up", "organized"],
    "prepared": ["drafted", "put together", "assembled", "compiled"],
    "discussed": ["addressed", "talked about", "deliberated on", "covered"],
    "distributed": ["circulated", "shared", "disseminated", "sent out"],
    "critical": ["crucial", "vital", "essential", "key"],
    "several": ["multiple", "a number of", "various", "numerous"],
    "together": ["jointly", "collaboratively", "as a team", "in tandem"],
    "First": ["Initially", "To begin with", "Starting off", "As a first step"],
    "Both": ["The two of them", "Each of them", "The pair"],
    "None": ["Not one", "Zero", "Not any"],
}

NAME_PATTERN = re.compile(
    r'\b(Priya|Marcus|Lena|Carlos|Yuki|Amara|Dmitri|Sofia|Kwame|Elena|'
    r'Raj|Ingrid|Hassan|Mei|Tomás|Fatima|Viktor|Chiara|Aiden|Nadia|'
    r'Kofi|Lucia|Jens|Zara|Oleg|Anya|Ravi|Sven|Layla|Esteban)\b'
)

DEPT_PATTERN = re.compile(
    r'\b(Engineering|Marketing|Operations|Finance|Legal|Research|Compliance|'
    r'Analytics|Procurement|Quality|Infrastructure|Product|Security|Support|Design)\b'
)


def _det_rng(row_id: int, salt: str) -> _rnd.Random:
    seed = int(hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest(), 16) % (2**32)
    return _rnd.Random(seed)


def _should_modify(row_id: int, salt: str, threshold: float) -> bool:
    h = hashlib.md5(f"{row_id}_{salt}".encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < threshold


def _replace_names(text: str, name_map: dict) -> str:
    def replacer(m):
        return name_map.get(m.group(0), m.group(0))
    return NAME_PATTERN.sub(replacer, text)


def _replace_depts(text: str, dept_map: dict) -> str:
    def replacer(m):
        return dept_map.get(m.group(0), m.group(0))
    return DEPT_PATTERN.sub(replacer, text)


def _paraphrase(text: str, rng: _rnd.Random) -> str:
    for word, alts in PARAPHRASE_MAP.items():
        if word in text and rng.random() < 0.30:
            text = text.replace(word, rng.choice(alts), 1)
    return text


def _shuffle_scenario_sentences(scenario: str, rng: _rnd.Random) -> str:
    sentences = scenario.split('. ')
    if len(sentences) < 3:
        return scenario
    # Swap two adjacent sentences (not the first one — it sets context)
    idx = rng.randint(1, len(sentences) - 2)
    sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
    return '. '.join(sentences)


def _truncate_scenario(scenario: str) -> str:
    sentences = scenario.split('. ')
    if len(sentences) > 2:
        return '. '.join(sentences[:-1]) + '.'
    return scenario


def _add_redherring(scenario: str, rng: _rnd.Random) -> str:
    herring = rng.choice(REDHERRING_SENTENCES)
    sentences = scenario.split('. ')
    if len(sentences) > 2:
        pos = rng.randint(1, len(sentences) - 1)
        sentences.insert(pos, herring)
    else:
        sentences.append(herring)
    return '. '.join(sentences)


def _obfuscate_row(row: dict, row_id: int) -> dict:
    scenario = row["scenario"]
    instruction = row["instruction"]
    options = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]

    # 1. Entity name replacement
    if _should_modify(row_id, "names", 0.40):
        names_found = list(set(NAME_PATTERN.findall(scenario + " " + instruction)))
        rng = _det_rng(row_id, "names")
        rng.shuffle(names_found)
        name_map = {n: f"Person-{i+1}" for i, n in enumerate(names_found)}
        scenario = _replace_names(scenario, name_map)
        instruction = _replace_names(instruction, name_map)
        options = [_replace_names(o, name_map) for o in options]

    # 2. Department anonymization
    if _should_modify(row_id, "depts", 0.35):
        depts_found = list(set(DEPT_PATTERN.findall(scenario + " " + instruction)))
        rng = _det_rng(row_id, "depts")
        rng.shuffle(depts_found)
        dept_map = {d: f"Dept-{chr(65+i)}" for i, d in enumerate(depts_found)}
        scenario = _replace_depts(scenario, dept_map)
        instruction = _replace_depts(instruction, dept_map)
        options = [_replace_depts(o, dept_map) for o in options]

    # 3. Scenario sentence shuffle
    if _should_modify(row_id, "shuffle", 0.30):
        rng = _det_rng(row_id, "shuffle")
        scenario = _shuffle_scenario_sentences(scenario, rng)

    # 4. Paraphrase options
    if _should_modify(row_id, "para", 0.25):
        rng = _det_rng(row_id, "para")
        options = [_paraphrase(o, rng) for o in options]

    # 5. Truncate scenario
    if _should_modify(row_id, "trunc", 0.15):
        scenario = _truncate_scenario(scenario)

    # 6. Red herring
    if _should_modify(row_id, "herring", 0.25):
        rng = _det_rng(row_id, "herring")
        scenario = _add_redherring(scenario, rng)

    return {
        "id": row_id,
        "scenario": scenario,
        "instruction": instruction,
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
    }


def prepare(raw: Path = Path("."), public: Path = Path("pub"), private: Path = Path("priv")) -> None:
    raw_file = raw / "data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Expected raw data at {raw_file}")

    df = pd.read_csv(raw_file)
    required_cols = {"id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d", "answer"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Raw data must have columns {required_cols}, got {set(df.columns)}")

    # Obfuscate
    obfuscated_rows = []
    for _, row in df.iterrows():
        ob = _obfuscate_row(row.to_dict(), int(row["id"]))
        ob["answer"] = row["answer"]
        obfuscated_rows.append(ob)

    ob_df = pd.DataFrame(obfuscated_rows)

    # Stratified split
    train_df, test_df = train_test_split(
        ob_df, test_size=0.2, random_state=42, shuffle=True, stratify=ob_df["answer"]
    )

    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_cols = ["id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d", "answer"]
    test_cols = ["id", "scenario", "instruction", "option_a", "option_b", "option_c", "option_d"]

    train_df[train_cols].to_csv(public / "train.csv", index=False)
    test_df[test_cols].to_csv(public / "test.csv", index=False)

    sample = test_df[["id"]].copy()
    sample["answer"] = "A"
    sample.to_csv(public / "sample_submission.csv", index=False)

    test_df[["id", "answer"]].to_csv(private / "answers.csv", index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train answers: {dict(train_df['answer'].value_counts())}")
    print(f"Test answers:  {dict(test_df['answer'].value_counts())}")


if __name__ == "__main__":
    prepare()
