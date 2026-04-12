"""
Latent Composition Rule Inference — data generator.

Creates 30 component tokens, each with 3 hidden dimensional properties.
Generates combinations of 3-5 tokens and computes output codes based on
how token properties interact through three independent composition functions.

Output files:
  - data.csv        (id, input_tokens, output)
  - properties.csv  (token_id, p1, p2, p3) — reference only, NOT for agents
"""

from __future__ import annotations
import argparse
import csv
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

SEED = 42
N_COMPONENTS = 30
SIZE_4_SAMPLE = 8000
SIZE_5_SAMPLE = 3000


def generate_components(rng: random.Random) -> list[dict]:
    comps = []
    for i in range(N_COMPONENTS):
        comps.append({
            "token_id": f"C_{i:02d}",
            "p1": rng.randint(0, 1),
            "p2": rng.randint(0, 9),
            "p3": rng.randint(0, 4),
        })
    return comps


def compute_output(combo_props: list[dict]) -> str:
    xor_val = 0
    for c in combo_props:
        xor_val ^= c["p1"]
    max_val = max(c["p2"] for c in combo_props)
    mod_val = sum(c["p3"] for c in combo_props) % 5
    return f"{xor_val}-{max_val}-{mod_val}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    components = generate_components(rng)
    comp_dict = {c["token_id"]: c for c in components}
    token_ids = [c["token_id"] for c in components]

    data = []
    row_id = 0

    for combo in combinations(token_ids, 3):
        props = [comp_dict[t] for t in combo]
        data.append({
            "id": row_id,
            "input_tokens": " ".join(combo),
            "output": compute_output(props),
        })
        row_id += 1

    all_s4 = list(combinations(token_ids, 4))
    rng.shuffle(all_s4)
    for combo in all_s4[:SIZE_4_SAMPLE]:
        props = [comp_dict[t] for t in combo]
        data.append({
            "id": row_id,
            "input_tokens": " ".join(combo),
            "output": compute_output(props),
        })
        row_id += 1

    all_s5 = list(combinations(token_ids, 5))
    rng.shuffle(all_s5)
    for combo in all_s5[:SIZE_5_SAMPLE]:
        props = [comp_dict[t] for t in combo]
        data.append({
            "id": row_id,
            "input_tokens": " ".join(combo),
            "output": compute_output(props),
        })
        row_id += 1

    with open(out / "properties.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["token_id", "p1", "p2", "p3"])
        w.writeheader()
        w.writerows(components)

    with open(out / "data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "input_tokens", "output"])
        w.writeheader()
        w.writerows(data)

    print(f"Components: {len(components)}")
    print(f"Total samples: {len(data)}")
    sizes = Counter()
    for d in data:
        sizes[len(d["input_tokens"].split())] += 1
    for k in sorted(sizes):
        print(f"  Size-{k}: {sizes[k]}")

    outputs = Counter(d["output"] for d in data)
    print(f"Unique output codes: {len(outputs)}")
    print(f"Top 5 outputs: {outputs.most_common(5)}")
    print(f"Bottom 5 outputs: {outputs.most_common()[-5:]}")

    p1_dist = Counter(c["p1"] for c in components)
    p2_dist = Counter(c["p2"] for c in components)
    p3_dist = Counter(c["p3"] for c in components)
    print(f"\nProperty distributions:")
    print(f"  p1 (binary): {dict(sorted(p1_dist.items()))}")
    print(f"  p2 (0-9):    {dict(sorted(p2_dist.items()))}")
    print(f"  p3 (0-4):    {dict(sorted(p3_dist.items()))}")


if __name__ == "__main__":
    main()
