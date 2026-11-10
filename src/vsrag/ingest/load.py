from collections.abc import Iterable
from dataclasses import astuple

import psycopg

from vsrag.ingest.parse import ParsedDocument

UPSERT_DOCUMENT = """
    INSERT INTO documents (id, source, title, body, search_text, received_on,
                           crash, fire, injured, deaths)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        source = EXCLUDED.source, title = EXCLUDED.title, body = EXCLUDED.body,
        search_text = EXCLUDED.search_text, received_on = EXCLUDED.received_on,
        crash = EXCLUDED.crash, fire = EXCLUDED.fire,
        injured = EXCLUDED.injured, deaths = EXCLUDED.deaths
"""

INSERT_VEHICLE = """
    INSERT INTO document_vehicles (document_id, make, model, model_year, component)
    VALUES (%s, %s, %s, %s, %s)
"""

BATCH_SIZE = 1000


def load_documents(conn: psycopg.Connection, docs: Iterable[ParsedDocument]) -> int:
    """Insert new documents and update existing ones. Running it twice creates no duplicates.

    Doesn't commit: the caller decides when the changes become permanent.
    """
    loaded = 0
    batch: list[ParsedDocument] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) == BATCH_SIZE:
            loaded += _load_batch(conn, batch)
            batch = []
    if batch:
        loaded += _load_batch(conn, batch)
    return loaded


def _load_batch(conn: psycopg.Connection, batch: list[ParsedDocument]) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            UPSERT_DOCUMENT,
            [
                (
                    d.id,
                    d.source,
                    d.title,
                    d.body,
                    d.search_text,
                    d.received_on,
                    d.crash,
                    d.fire,
                    d.injured,
                    d.deaths,
                )
                for d in batch
            ],
        )
        # Replace each document's vehicle list, so vehicles NHTSA removed don't linger.
        cur.execute(
            "DELETE FROM document_vehicles WHERE document_id = ANY(%s)", ([d.id for d in batch],)
        )
        cur.executemany(INSERT_VEHICLE, [(d.id, *astuple(v)) for d in batch for v in d.vehicles])
    return len(batch)
