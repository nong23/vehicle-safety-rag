from pathlib import Path
from typing import Any

import pytest

from evals.check_golden import COUNTS, check, load
from evals.review_draft import pending, to_golden

KNOWN = {"complaint:1", "complaint:2", "recall:20V314000"}


def row(**overrides: Any) -> dict[str, Any]:
    """A valid answerable dev question; change fields with keyword arguments."""
    base: dict[str, Any] = {
        "id": "a01",
        "question": "Does the 2019 Honda Accord stall?",
        "category": "answerable",
        "split": "dev",
        "filters": {},
        "relevant_document_ids": ["complaint:1"],
        "expected_points": ["owners report stalling"],
        "should_abstain": False,
    }
    return base | overrides


def full_set() -> list[dict[str, Any]]:
    """A complete, valid 75-question set."""
    rows = []
    for (split, category), count in COUNTS.items():
        for n in range(count):
            question = row(id=f"{split}-{category}-{n}", split=split, category=category)
            question["question"] = f"Question {split} {category} {n}?"
            if category == "unanswerable":
                question |= {"relevant_document_ids": [], "expected_points": []}
                question["should_abstain"] = True
            rows.append(question)
    return rows


def test_valid_full_set_passes() -> None:
    assert check(full_set(), KNOWN) == []


def test_rejects_unknown_document_id() -> None:
    errors = check([row(relevant_document_ids=["complaint:404"])], KNOWN, partial=True)

    assert errors == ["row 1 (a01): unknown document id complaint:404"]


def test_rejects_answerable_without_relevant_documents() -> None:
    errors = check([row(relevant_document_ids=[])], KNOWN, partial=True)

    assert errors == ["row 1 (a01): a question with an answer needs relevant_document_ids"]


def test_rejects_answerable_without_expected_points() -> None:
    errors = check([row(expected_points=[])], KNOWN, partial=True)

    assert errors == ["row 1 (a01): a question with an answer needs expected_points"]


def test_rejects_wrong_category_mix() -> None:
    rows = full_set()
    rows[0]["category"] = "tricky"  # one dev answerable becomes tricky

    errors = check(rows, KNOWN)

    assert "dev / answerable: 14 questions, need 15" in errors
    assert "dev / tricky: 4 questions, need 3" in errors


def test_rejects_wrong_split_sizes() -> None:
    rows = full_set()
    rows[0]["split"] = "test"  # 24 dev, 51 test

    errors = check(rows, KNOWN)

    assert "dev / answerable: 14 questions, need 15" in errors
    assert "test / answerable: 31 questions, need 30" in errors


def test_partial_skips_the_counts() -> None:
    assert check([row()], KNOWN, partial=True) == []
    assert check([row()], KNOWN) != []


def test_unanswerable_must_abstain_and_cite_nothing() -> None:
    errors = check([row(category="unanswerable")], KNOWN, partial=True)

    assert "row 1 (a01): unanswerable questions must have should_abstain true" in errors
    assert "row 1 (a01): unanswerable questions have no relevant documents or points" in errors


def test_answerable_must_not_abstain() -> None:
    errors = check([row(should_abstain=True)], KNOWN, partial=True)

    assert errors == ["row 1 (a01): answerable questions must have should_abstain false"]


def test_tricky_can_go_either_way() -> None:
    abstains = row(category="tricky", should_abstain=True, relevant_document_ids=[])
    answers = row(id="t02", question="Other?", category="tricky")

    assert check([abstains, answers], KNOWN, partial=True) == []


def test_rejects_duplicate_ids_and_questions() -> None:
    errors = check(
        [row(), row(question="  DOES the 2019 Honda Accord stall?")], KNOWN, partial=True
    )

    assert "row 2 (a01): duplicate id" in errors
    assert "row 2 (a01): duplicate question" in errors


def test_rejects_missing_or_mistyped_fields() -> None:
    broken = row(should_abstain="no")
    del broken["filters"]

    errors = check([broken], KNOWN, partial=True)

    assert errors == ["row 1 (a01): missing or wrong type: filters, should_abstain"]


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"colour": "red"}, "unknown filter 'colour'"),
        ({"make": "MAZDA"}, "filter make must be one of"),
        ({"model": "accord"}, "filter model must be uppercase"),
        ({"year": 2015}, "filter year must be 2018-2023"),
        ({"year": True}, "filter year has the wrong type"),
        ({"source": "blog"}, "filter source must be complaint or recall"),
    ],
)
def test_rejects_filters_the_page_cant_send(filters: dict[str, Any], message: str) -> None:
    errors = check([row(filters=filters)], KNOWN, partial=True)

    assert len(errors) == 1 and message in errors[0]


def test_rejects_overlong_question() -> None:
    errors = check([row(question="a" * 501)], KNOWN, partial=True)

    assert errors == ["row 1 (a01): question must be 1-500 characters"]


def test_load_reports_the_line_of_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "a01"}\n\n{not json\n["a list"]\n', encoding="utf-8")

    rows, errors = load(path)

    assert rows == [{"id": "a01"}]
    assert errors[0].startswith("line 3: not valid JSON")
    assert errors[1] == "line 4: must be a JSON object {...}"


def test_review_skips_questions_already_decided() -> None:
    drafts = [row(id="a01"), row(id="a02"), row(id="a03")]

    assert [r["id"] for r in pending(drafts, {"a02"})] == ["a01", "a03"]


def test_approved_question_drops_the_reviewer_notes() -> None:
    assert to_golden(row(notes="read each document")) == row()
