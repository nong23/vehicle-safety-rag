import psycopg

from vsrag.db import init_schema


def test_init_schema_is_repeatable(conn: psycopg.Connection) -> None:
    init_schema(conn)
    init_schema(conn)

    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    assert {"documents", "document_vehicles", "embeddings"} <= {row[0] for row in rows}


def test_tsv_is_built_from_search_text(conn: psycopg.Connection) -> None:
    init_schema(conn)
    conn.execute(
        "INSERT INTO documents (id, source, title, body, search_text)"
        " VALUES ('complaint:1', 'complaint', 'Brakes', 'body', 'BRAKES FAILED ON HIGHWAY')"
    )

    row = conn.execute(
        "SELECT id FROM documents WHERE tsv @@ websearch_to_tsquery('english', 'braking highway')"
    ).fetchone()
    assert row == ("complaint:1",)
