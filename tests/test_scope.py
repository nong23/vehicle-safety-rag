from datetime import date

import psycopg
import pytest

from vsrag.db import init_schema
from vsrag.ingest.load import load_documents
from vsrag.ingest.parse import ParsedDocument, Vehicle
from vsrag.scope import check_scope, known_models
from vsrag.search import SearchFilters

MODELS = {
    "FORD": {"F-150", "ESCAPE", "EDGE"},
    "HONDA": {"ACCORD", "ACCORD HYBRID", "CR-V", "PILOT", "FIT"},
    "KIA": {"TELLURIDE"},
    "TOYOTA": {"RAV4"},
}


def scope(question: str, filters: SearchFilters | None = None):  # type: ignore[no-untyped-def]
    return check_scope(question, filters or SearchFilters(), MODELS)


@pytest.mark.parametrize(
    "question",
    ["2015 Mazda 3 brake problems", "2021 BMW X3 airbag recall", "Do Teslas catch fire?"],
)
def test_out_of_scope_make_is_rejected(question: str) -> None:
    result = scope(question)

    assert not result.in_scope
    assert result.reason is not None


def test_out_of_scope_year_is_rejected() -> None:
    assert not scope("2015 Ford Escape transmission").in_scope


def test_in_scope_vehicle_in_question_becomes_filter() -> None:
    result = scope("Do airbags fail on the 2019 Honda Accord?")

    assert result.in_scope
    assert result.filters == SearchFilters(make="HONDA", model="ACCORD", year=2019)


def test_question_with_no_vehicle_stays_in_scope() -> None:
    result = scope("What causes transmission failures?")

    assert result.in_scope
    assert result.filters == SearchFilters()


def test_model_alone_implies_make() -> None:
    assert scope("Telluride airbag problems").filters == SearchFilters(
        make="KIA", model="TELLURIDE"
    )


@pytest.mark.parametrize(
    ("question", "model"),
    [("f150 brakes", "F-150"), ("F 150 brakes", "F-150"), ("my crv stalls", "CR-V")],
)
def test_model_spelling_variants_match(question: str, model: str) -> None:
    assert scope(question).filters.model == model


def test_longest_model_name_wins() -> None:
    assert scope("2020 Accord Hybrid battery").filters.model == "ACCORD HYBRID"


def test_plural_make_counts() -> None:
    assert scope("Do Hondas have brake problems?").filters.make == "HONDA"


@pytest.mark.parametrize(
    ("question", "model"),
    [
        ("the seat belt doesn't fit", None),  # "fit" is just a word here
        ("brakes on the 2019 Pilot", "PILOT"),  # a year right before it makes it a model
        ("Honda Fit engine stalls", "FIT"),  # so does naming the make
    ],
)
def test_common_word_model_needs_make_or_year(question: str, model: str | None) -> None:
    assert scope(question).filters.model == model


@pytest.mark.parametrize("question", ["brakes failed at 2000 miles", "repair cost $2000"])
def test_quantities_are_not_years(question: str) -> None:
    result = scope(question)

    assert result.in_scope
    assert result.filters.year is None


def test_callers_filters_win_over_the_question() -> None:
    result = scope("2019 RAV4 brakes", SearchFilters(make="TOYOTA", year=2020))

    assert result.filters == SearchFilters(make="TOYOTA", model="RAV4", year=2020)


def test_out_of_range_filter_is_rejected() -> None:
    assert not scope("brakes", SearchFilters(year=2016)).in_scope


def test_known_models_come_from_the_database(conn: psycopg.Connection) -> None:
    init_schema(conn)
    vehicles = [
        Vehicle("TOYOTA", "SIENNA", 2019, "ENGINE"),
        Vehicle("TOYOTA", "REDUNDANT SIENNA", 2019, "ENGINE"),  # NHTSA's junk entries
    ]
    load_documents(
        conn,
        [
            ParsedDocument(
                id="complaint:1",
                source="complaint",
                title="Owner complaint",
                body="ENGINE STALLED.",
                search_text="ENGINE STALLED.",
                received_on=date(2021, 1, 1),
                crash=False,
                fire=False,
                injured=0,
                deaths=0,
                vehicles=vehicles,
            )
        ],
    )

    assert known_models(conn) == {"TOYOTA": {"SIENNA"}}
