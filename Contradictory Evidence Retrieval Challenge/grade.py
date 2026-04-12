"""
Grading script: per-field exact match scoring for the
Contradictory Evidence Retrieval and Verdict Challenge.

Verdict format:
  STANCE:<stance> | EVIDENCE_IDS:<ids> | CONFIDENCE:<conf>

Fields:
  - STANCE: support | contradict | insufficient
  - EVIDENCE_IDS: comma-separated integers or "none"
  - CONFIDENCE: high | medium | low

Scoring:
  For each row, parse both predicted and true verdict into 3 fields.
  - STANCE: exact match (0 or 1)
  - EVIDENCE_IDS: exact set match (0 or 1) — order doesn't matter
  - CONFIDENCE: exact match (0 or 1)

  Row score = mean of the 3 field scores.
  Final score = mean of all row scores.

Score range: [0.0, 1.0] where higher is better.
"""

from pathlib import Path
import csv


def _parse_verdict(verdict_str: str) -> dict | None:
    """Parse a verdict string into its component fields."""
    try:
        parts = [p.strip() for p in verdict_str.strip().split("|")]
        result = {}
        for part in parts:
            key, _, value = part.partition(":")
            result[key.strip()] = value.strip()

        if "STANCE" not in result or "EVIDENCE_IDS" not in result or "CONFIDENCE" not in result:
            return None

        return {
            "stance": result["STANCE"].lower().strip(),
            "evidence_ids": _parse_evidence_ids(result["EVIDENCE_IDS"]),
            "confidence": result["CONFIDENCE"].lower().strip(),
        }
    except Exception:
        return None


def _parse_evidence_ids(ids_str: str) -> frozenset:
    ids_str = ids_str.strip().lower()
    if ids_str in ("none", "", "n/a"):
        return frozenset()
    try:
        return frozenset(int(x.strip()) for x in ids_str.split(",") if x.strip())
    except ValueError:
        return frozenset()


def _score_row(pred_verdict: str, true_verdict: str) -> float:
    true_parsed = _parse_verdict(true_verdict)
    pred_parsed = _parse_verdict(pred_verdict)

    if true_parsed is None:
        return 0.0
    if pred_parsed is None:
        return 0.0

    stance_match = 1.0 if pred_parsed["stance"] == true_parsed["stance"] else 0.0
    evidence_match = 1.0 if pred_parsed["evidence_ids"] == true_parsed["evidence_ids"] else 0.0
    confidence_match = 1.0 if pred_parsed["confidence"] == true_parsed["confidence"] else 0.0

    return (stance_match + evidence_match + confidence_match) / 3.0


def grade(submission_path: Path, answer_path: Path) -> float:
    """Grade a submission against the answer key.

    Returns a float in [0.0, 1.0].
    """
    answers = {}
    with open(answer_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers[str(row["id"])] = row["verdict"]

    scores = []
    with open(submission_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id", ""))
            pred = row.get("verdict", "")
            true = answers.get(rid)
            if true is None:
                continue
            scores.append(_score_row(pred, true))

    if not scores:
        return 0.0

    if len(scores) < len(answers):
        n_missing = len(answers) - len(scores)
        scores.extend([0.0] * n_missing)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python grade.py <submission.csv> <answers.csv>")
        sys.exit(1)
    score = grade(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Score: {score:.4f}")
