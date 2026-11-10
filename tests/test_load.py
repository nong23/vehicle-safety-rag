from datetime import date

import psycopg

from vsrag.db import init_schema
from vsrag.ingest.load import load_documents
from vsrag.ingest.parse import ParsedDocument, Vehicle


def make_doc(
    body: str = "ENGINE STALLED.", vehicles: list[Vehicle] | None = None
) -> ParsedDocument:
    return ParsedDocument(
        id="complaint:11000001",
        source="complaint",
        title="2019 HONDA ACCORD: ENGINE",
        body=body,
        search_text=f"2019 HONDA ACCORD ENGINE\n{body}",
        received_on=date(2021, 3, 15),
        crash=False,
        fire=False,
        injured=0,
        deaths=0,
        vehicles=vehicles or [Vehicle("HONDA", "ACCORD", 2019, "ENGINE")],
    )


def count(conn: psycopg.Connection, table: str) -> int:
    row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
    assert row is not None
    return int(row[0])


def test_load_twice_does_not_duplicate(conn: psycopg.Connection) -> None:
    init_schema(conn)

    assert load_documents(conn, [make_doc()]) == 1
    assert load_documents(conn, [make_doc()]) == 1

    assert count(conn, "documents") == 1
    assert count(conn, "document_vehicles") == 1


def test_reload_replaces_changed_text_and_vehicles(conn: psycopg.Connection) -> None:
    init_schema(conn)
    load_documents(conn, [make_doc()])

    updated = make_doc(
        body="ENGINE STALLED AGAIN.",
        vehicles=[Vehicle("HONDA", "ACCORD", 2019, "POWER TRAIN")],
    )
    load_documents(conn, [updated])

    row = conn.execute("SELECT body FROM documents").fetchone()
    components = conn.execute("SELECT component FROM document_vehicles").fetchall()
    assert row == ("ENGINE STALLED AGAIN.",)
    assert components == [("POWER TRAIN",)]
