import math
from datetime import date

import psycopg
import pytest

from vsrag.db import init_schema
from vsrag.embed import (
    VARIANTS,
    document_text,
    embed_missing,
    embed_query,
    ensure_index,
)
from vsrag.ingest.load import load_documents
from vsrag.ingest.parse import ParsedDocument, Vehicle

ACCORD = Vehicle("HONDA", "ACCORD", 2019, "ENGINE")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def make_doc(odino: str) -> ParsedDocument:
    return ParsedDocument(
        id=f"complaint:{odino}",
        source="complaint",
        title="Owner complaint about the 2019 HONDA ACCORD",
        body="ENGINE STALLED AT HIGHWAY SPEED.",
        search_text="2019 HONDA ACCORD ENGINE\nENGINE STALLED AT HIGHWAY SPEED.",
        received_on=date(2021, 3, 15),
        crash=False,
        fire=False,
        injured=0,
        deaths=0,
        vehicles=[ACCORD],
    )


def test_variants_are_the_three_in_the_plan() -> None:
    assert set(VARIANTS) == {"bge_plain", "bge_prefixed", "minilm_prefixed"}


def test_embed_query_returns_384_numbers() -> None:
    vector = embed_query("brakes failed", "bge_prefixed")

    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


def test_similar_meaning_scores_higher() -> None:
    question = embed_query("car shuts off while driving", "bge_prefixed")
    stall = embed_query("ENGINE STALLED AT HIGHWAY SPEED", "bge_prefixed")
    paint = embed_query("PAINT PEELING ON HOOD", "bge_prefixed")

    assert cosine(question, stall) > cosine(question, paint)


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError):
        embed_query("brakes", "not_a_variant")


def test_prefixed_text_includes_vehicle() -> None:
    text = document_text("ENGINE STALLED.", [ACCORD], prefixed=True)

    assert text == "2019 HONDA ACCORD | ENGINE\nENGINE STALLED."


def test_plain_text_is_just_the_body() -> None:
    assert document_text("ENGINE STALLED.", [ACCORD], prefixed=False) == "ENGINE STALLED."


def test_prefix_lists_each_vehicle_once_and_shortens_long_lists() -> None:
    vehicles = [Vehicle("HONDA", "ACCORD", 2019, part) for part in ("ENGINE", "STEERING")]
    vehicles += [Vehicle("HONDA", f"MODEL{i}", 2020, "ENGINE") for i in range(7)]

    prefix = document_text("BODY", vehicles, prefixed=True).splitlines()[0]

    assert prefix.count("2019 HONDA ACCORD") == 1
    assert "and 3 more" in prefix
    assert prefix.endswith("| ENGINE, STEERING")


def test_embed_missing_skips_documents_already_done(conn: psycopg.Connection) -> None:
    init_schema(conn)
    load_documents(conn, [make_doc("1"), make_doc("2")])

    assert embed_missing(conn, "bge_prefixed") == 2
    assert embed_missing(conn, "bge_prefixed") == 0


def test_embed_missing_stops_at_limit(conn: psycopg.Connection) -> None:
    init_schema(conn)
    load_documents(conn, [make_doc("1"), make_doc("2"), make_doc("3")])

    assert embed_missing(conn, "bge_prefixed", limit=2) == 2
    assert embed_missing(conn, "bge_prefixed", limit=2) == 1


def test_ensure_index_creates_one_index_per_variant(conn: psycopg.Connection) -> None:
    init_schema(conn)

    ensure_index(conn, "bge_prefixed")
    ensure_index(conn, "bge_prefixed")

    row = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'embeddings_bge_prefixed_hnsw'"
    ).fetchone()
    assert row is not None
    assert "hnsw" in row[0]
    assert "variant = 'bge_prefixed'" in row[0]
