"""Turn documents and questions into embeddings (384 numbers that capture meaning).

Run `python -m vsrag.embed <variant>` to embed every document that doesn't have one yet.
"""

import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import psycopg
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector
from psycopg import sql

from vsrag.db import get_conn
from vsrag.ingest.parse import Vehicle

MODEL_DIR = Path("data/models")
MAX_VEHICLES_SHOWN = 5
MAX_COMPONENTS_SHOWN = 3


@dataclass(frozen=True)
class Variant:
    model_name: str
    prefixed: bool


# Experiment 2 compares bge_plain vs. bge_prefixed; Experiment 3 compares bge vs. MiniLM.
VARIANTS = {
    "bge_plain": Variant("BAAI/bge-small-en-v1.5", prefixed=False),
    "bge_prefixed": Variant("BAAI/bge-small-en-v1.5", prefixed=True),
    "minilm_prefixed": Variant("sentence-transformers/all-MiniLM-L6-v2", prefixed=True),
}


@cache
def _model(model_name: str) -> TextEmbedding:
    """Load each model once per process; loading takes a few seconds."""
    return TextEmbedding(model_name, cache_dir=str(MODEL_DIR))


def embed_query(text: str, variant: str) -> list[float]:
    model = _model(VARIANTS[variant].model_name)
    return [float(x) for x in next(iter(model.query_embed(text)))]


def document_text(body: str, vehicles: list[Vehicle], prefixed: bool) -> str:
    """The text that gets embedded: the body, optionally after a line naming the vehicles."""
    if not prefixed:
        return body

    names = list(dict.fromkeys(f"{v.model_year} {v.make} {v.model}" for v in vehicles))
    components = list(dict.fromkeys(v.component for v in vehicles))[:MAX_COMPONENTS_SHOWN]
    shown = ", ".join(names[:MAX_VEHICLES_SHOWN])
    hidden = len(names) - MAX_VEHICLES_SHOWN
    if hidden > 0:
        shown += f" and {hidden} more"
    return f"{shown} | {', '.join(components)}\n{body}"


def embed_missing(conn: psycopg.Connection, variant: str, limit: int = 512) -> int:
    """Embed up to `limit` documents that have no embedding for `variant` yet.

    Returns how many it embedded, so callers can loop until it returns 0. Doesn't commit.
    """
    settings = VARIANTS[variant]
    register_vector(conn)

    rows = conn.execute(
        """
        SELECT d.id, d.body
        FROM documents d
        WHERE NOT EXISTS (
            SELECT 1 FROM embeddings e WHERE e.document_id = d.id AND e.variant = %s
        )
        ORDER BY d.id
        LIMIT %s
        """,
        (variant, limit),
    ).fetchall()
    if not rows:
        return 0

    ids = [row[0] for row in rows]
    vehicles: dict[str, list[Vehicle]] = {doc_id: [] for doc_id in ids}
    for doc_id, make, model, year, component in conn.execute(
        "SELECT document_id, make, model, model_year, component FROM document_vehicles"
        " WHERE document_id = ANY(%s) ORDER BY document_id, model_year, make, model, component",
        (ids,),
    ):
        vehicles[doc_id].append(Vehicle(make, model, year, component))

    texts = [document_text(body, vehicles[doc_id], settings.prefixed) for doc_id, body in rows]
    vectors = _model(settings.model_name).passage_embed(texts, batch_size=64)

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO embeddings (document_id, variant, embedding) VALUES (%s, %s, %s)",
            [(doc_id, variant, vector) for doc_id, vector in zip(ids, vectors, strict=True)],
        )
    return len(rows)


def ensure_index(conn: psycopg.Connection, variant: str) -> None:
    """Create this variant's HNSW index. One index per variant, so a search for one
    variant never has to wade through another variant's vectors."""
    if variant not in VARIANTS:
        raise KeyError(variant)
    conn.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON embeddings"
            " USING hnsw (embedding vector_cosine_ops) WHERE variant = {variant}"
        ).format(
            name=sql.Identifier(f"embeddings_{variant}_hnsw"),
            variant=sql.Literal(variant),
        )
    )


def main(variant: str) -> None:
    start = time.perf_counter()
    with get_conn() as conn:
        row = conn.execute("SELECT count(*) FROM documents").fetchone()
        total = row[0] if row else 0
        done = 0
        while count := embed_missing(conn, variant):
            conn.commit()  # Commit each batch, so an interrupted run resumes where it stopped.
            done += count
            rate = done / (time.perf_counter() - start)
            print(f"{variant}: {done:,} embedded this run, {rate:.0f} docs/s, {total:,} total")
        print("Building the HNSW index...")
        ensure_index(conn, variant)
        conn.commit()
    print(f"Done: {done:,} embedded in {time.perf_counter() - start:.0f} s.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bge_prefixed")
