"""Review drafted test questions one at a time, reading the actual records.

    python -m evals.review_draft

For each draft it shows the question, what a correct answer should say, and the full text
of every cited document. Then you decide:

    y  approve: the documents really support it  -> added to evals/golden.jsonl
    n  reject: wrong, unclear or a bad question  -> added to evals/golden_rejected.jsonl
    s  skip for now        q  quit (your progress is saved)

To fix a draft instead of rejecting it, edit evals/golden_draft.jsonl and run this again.
"""

import json
from pathlib import Path
from typing import Any

from evals.check_golden import GOLDEN, load
from vsrag.db import get_conn

DRAFT = GOLDEN.with_name("golden_draft.jsonl")
REJECTED = GOLDEN.with_name("golden_rejected.jsonl")
MAX_BODY_CHARS = 1500


def pending(drafts: list[dict[str, Any]], done_ids: set[str]) -> list[dict[str, Any]]:
    """Drafts not yet approved or rejected."""
    return [row for row in drafts if row.get("id") not in done_ids]


def to_golden(row: dict[str, Any]) -> dict[str, Any]:
    """The draft without its reviewer notes, which don't belong in the test set."""
    return {key: value for key, value in row.items() if key != "notes"}


def _ids_in(path: Path) -> set[str]:
    return {str(row.get("id")) for row in load(path)[0]} if path.exists() else set()


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _show(row: dict[str, Any], conn: Any) -> None:
    print("\n" + "=" * 78)
    print(
        f"{row['id']}  |  {row['category']}  |  {row['split']}  |  should_abstain: "
        f"{str(row['should_abstain']).lower()}"
    )
    print(f"\nQUESTION: {row['question']}")
    if row.get("filters"):
        print(f"FILTERS: {row['filters']}")
    for point in row["expected_points"]:
        print(f"EXPECTED: {point}")
    if row.get("notes"):
        print(f"CHECK: {row['notes']}")
    for doc_id in row["relevant_document_ids"]:
        found = conn.execute(
            "SELECT title, body FROM documents WHERE id = %s", (doc_id,)
        ).fetchone()
        vehicles = conn.execute(
            "SELECT DISTINCT model_year || ' ' || make || ' ' || model FROM document_vehicles"
            " WHERE document_id = %s ORDER BY 1",
            (doc_id,),
        ).fetchall()
        names = ", ".join(v[0] for v in vehicles[:6]) + (" ..." if len(vehicles) > 6 else "")
        print(f"\n--- {doc_id}  |  {found[0] if found else 'NOT FOUND'}  |  {names}")
        if found:
            body = (
                found[1] if len(found[1]) <= MAX_BODY_CHARS else found[1][:MAX_BODY_CHARS] + " ..."
            )
            print(body)


def main() -> None:
    drafts, errors = load(DRAFT)
    for error in errors:
        print("DRAFT ERROR", error)
    todo = pending(drafts, _ids_in(GOLDEN) | _ids_in(REJECTED))
    print(f"{len(todo)} drafts left to review ({len(drafts)} in total).")

    with get_conn() as conn:
        for number, row in enumerate(todo, start=1):
            _show(row, conn)
            while True:
                choice = input(
                    f"\n[{number}/{len(todo)}] y = approve, n = reject, s = skip, q = quit: "
                )
                choice = choice.strip().lower()
                if choice in ("y", "n", "s", "q"):
                    break
            if choice == "q":
                break
            if choice == "y":
                _append(GOLDEN, to_golden(row))
            elif choice == "n":
                _append(REJECTED, row)
    print(f"\nApproved so far: {len(_ids_in(GOLDEN))}. Check them: python -m evals.check_golden")


if __name__ == "__main__":
    main()
