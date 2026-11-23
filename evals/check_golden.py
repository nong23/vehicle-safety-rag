"""Check evals/golden.jsonl, the hand-written test set, for mistakes.

    python -m evals.check_golden             all checks, including the final counts
    python -m evals.check_golden --partial   while you're still writing: skips the counts

Each line of golden.jsonl is one question, as a JSON object:

    {"id": "a01", "question": "Do airbags fail to deploy in the Kia Telluride?",
     "category": "answerable", "split": "dev", "filters": {},
     "relevant_document_ids": ["complaint:11331228", "recall:23V035000"],
     "expected_points": ["owners report airbags not deploying in crashes",
                         "recall 23V035000: 2023 second-row side airbags may not deploy"],
     "should_abstain": false}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vsrag.db import get_conn
from vsrag.ingest.parse import FIRST_YEAR, LAST_YEAR, MAKES

GOLDEN = Path(__file__).parent / "golden.jsonl"
MAX_QUESTION_CHARS = 500

FIELDS: dict[str, type] = {
    "id": str,
    "question": str,
    "category": str,
    "split": str,
    "filters": dict,
    "relevant_document_ids": list,
    "expected_points": list,
    "should_abstain": bool,
}
CATEGORIES = ("answerable", "unanswerable", "tricky")
SPLITS = ("dev", "test")
# 75 questions: 45 answerable, 20 unanswerable, 10 tricky. Dev gets a third (25) with the
# same mix, for tuning; test gets the rest (50), used once for the final numbers.
COUNTS = {
    ("dev", "answerable"): 15,
    ("dev", "unanswerable"): 7,
    ("dev", "tricky"): 3,
    ("test", "answerable"): 30,
    ("test", "unanswerable"): 13,
    ("test", "tricky"): 7,
}
FILTER_TYPES: dict[str, type] = {"make": str, "model": str, "year": int, "source": str}


def load(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read one JSON object per line. Blank lines are skipped."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: not valid JSON ({error.msg})")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {number}: must be a JSON object {{...}}")
            continue
        rows.append(value)
    return rows, errors


def check(rows: list[dict[str, Any]], known_ids: set[str], partial: bool = False) -> list[str]:
    """Every problem found, as readable messages. `known_ids`: document ids that exist."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    counts = dict.fromkeys(COUNTS, 0)

    for number, row in enumerate(rows, start=1):
        where = f"row {number} ({row.get('id', 'no id')})"
        wrong = [field for field, kind in FIELDS.items() if not _is(row.get(field), kind)]
        if wrong:
            errors.append(f"{where}: missing or wrong type: {', '.join(wrong)}")
            continue

        if row["id"] in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(row["id"])

        question = row["question"].strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            errors.append(f"{where}: question must be 1-{MAX_QUESTION_CHARS} characters")
        if question.lower() in seen_questions:
            errors.append(f"{where}: duplicate question")
        seen_questions.add(question.lower())

        category, split = row["category"], row["split"]
        if category not in CATEGORIES:
            errors.append(f"{where}: category must be one of {', '.join(CATEGORIES)}")
        if split not in SPLITS:
            errors.append(f"{where}: split must be dev or test")
        if (split, category) in counts:
            counts[(split, category)] += 1

        errors += [f"{where}: {problem}" for problem in _filter_problems(row["filters"])]

        ids, points = row["relevant_document_ids"], row["expected_points"]
        if not all(isinstance(i, str) for i in ids) or not all(
            isinstance(p, str) and p.strip() for p in points
        ):
            errors.append(f"{where}: document ids and expected points must be non-empty text")
            continue
        errors += [f"{where}: unknown document id {i}" for i in ids if i not in known_ids]

        abstain = row["should_abstain"]
        if category == "answerable" and abstain:
            errors.append(f"{where}: answerable questions must have should_abstain false")
        if category == "unanswerable" and not abstain:
            errors.append(f"{where}: unanswerable questions must have should_abstain true")
        if category == "unanswerable" and (ids or points):
            errors.append(f"{where}: unanswerable questions have no relevant documents or points")
        if not abstain and not ids:
            errors.append(f"{where}: a question with an answer needs relevant_document_ids")
        if not abstain and not points:
            errors.append(f"{where}: a question with an answer needs expected_points")

    if not partial:
        for (split, category), need in COUNTS.items():
            have = counts[(split, category)]
            if have != need:
                errors.append(f"{split} / {category}: {have} questions, need {need}")
    return errors


def _is(value: object, kind: type) -> bool:
    # bool is a kind of int in Python, so True would otherwise pass as a year.
    return isinstance(value, kind) and not (kind is int and isinstance(value, bool))


def _filter_problems(filters: dict[str, Any]) -> list[str]:
    """Filters must be ones the page could send: in-scope makes and years."""
    problems = []
    for key, value in filters.items():
        if key not in FILTER_TYPES:
            problems.append(f"unknown filter {key!r} (use make, model, year, source)")
        elif not _is(value, FILTER_TYPES[key]):
            problems.append(f"filter {key} has the wrong type")
        elif key == "make" and value not in MAKES:
            problems.append(f"filter make must be one of {', '.join(sorted(MAKES))}")
        elif key == "model" and value != value.upper():
            problems.append("filter model must be uppercase, as NHTSA stores it")
        elif key == "year" and not FIRST_YEAR <= value <= LAST_YEAR:
            problems.append(f"filter year must be {FIRST_YEAR}-{LAST_YEAR}")
        elif key == "source" and value not in ("complaint", "recall"):
            problems.append("filter source must be complaint or recall")
    return problems


def summary(rows: list[dict[str, Any]]) -> str:
    """How many questions you have of each kind, next to how many are needed."""
    lines = ["            have / need"]
    for split, category in COUNTS:
        have = sum(1 for r in rows if r.get("split") == split and r.get("category") == category)
        lines.append(f"  {split:4} {category:12} {have:3} / {COUNTS[(split, category)]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partial", action="store_true", help="skip the final counts")
    parser.add_argument("path", nargs="?", type=Path, default=GOLDEN)
    args = parser.parse_args()

    rows, errors = load(args.path)
    referenced = [
        i for r in rows for i in r.get("relevant_document_ids") or [] if isinstance(i, str)
    ]
    with get_conn() as conn:
        found = conn.execute(
            "SELECT id FROM documents WHERE id = ANY(%s)", (referenced,)
        ).fetchall()
    errors += check(rows, {row[0] for row in found}, partial=args.partial)

    print(summary(rows))
    for error in errors:
        print("ERROR", error)
    print(f"\n{len(rows)} questions, {len(errors)} problems.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
