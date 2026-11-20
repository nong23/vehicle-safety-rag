"""Load documents' full text, for prompts and citations."""

from dataclasses import dataclass
from typing import Literal

import psycopg


@dataclass(frozen=True)
class Document:
    id: str
    source: Literal["complaint", "recall"]
    title: str
    body: str


def fetch_documents(conn: psycopg.Connection, ids: list[str]) -> list[Document]:
    """Return the documents with these ids, in the same order. Unknown ids are skipped."""
    rows = conn.execute(
        "SELECT id, source, title, body FROM documents WHERE id = ANY(%s)", (ids,)
    ).fetchall()
    by_id = {row[0]: Document(*row) for row in rows}
    return [by_id[doc_id] for doc_id in ids if doc_id in by_id]
