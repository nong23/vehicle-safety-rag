"""Find documents for a question, by keyword (full-text search) or by meaning (vector search).

Both return hits best-first. Task 5 merges the two lists with RRF.
"""

from dataclasses import dataclass
from typing import Literal

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from vsrag.embed import embed_query


@dataclass(frozen=True)
class SearchFilters:
    """Optional limits on which documents can match. None means "any".

    make and model are uppercase, as NHTSA stores them (e.g. "HONDA", "ACCORD").
    """

    make: str | None = None
    model: str | None = None
    year: int | None = None
    source: Literal["complaint", "recall"] | None = None


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    rank: int  # 1 = best match
    similarity: float | None  # cosine similarity for vector hits; None for keyword hits


# private helper filter function
def _filter_conditions(filters: SearchFilters) -> tuple[list[str], list[object]]:
    """SQL conditions for the filters that are set, plus the values for their %s blanks."""
    # Strings with %s are SQL text; each %s is a placeholder for a value.
    # psycopg sends the values separately, so Postgres treats them as data, never as SQL.

    # the params.append call adds the actual value for the placeholder in
    # the condition appended just before it
    # The values must be appended in the same order as their %s slots appear
    conditions: list[str] = []
    params: list[object] = []

    if filters.source is not None:
        conditions.append("d.source = %s")  # source is a column of documents (alias d)
        params.append(filters.source)

    vehicle_conditions: list[str] = []
    # Same pattern for make, model and year. They're columns of document_vehicles (alias v);
    # year's column is called model_year.
    if filters.make is not None:
        vehicle_conditions.append("v.make = %s")
        params.append(filters.make)
    if filters.model is not None:
        vehicle_conditions.append("v.model = %s")
        params.append(filters.model)
    if filters.year is not None:
        vehicle_conditions.append("v.model_year = %s")
        params.append(filters.year)

    if vehicle_conditions:
        conditions.append(
            "EXISTS (SELECT 1 FROM document_vehicles v WHERE v.document_id = d.id AND "
            + " AND ".join(vehicle_conditions)
            + ")"
        )
    return conditions, params


def keyword_search(
    conn: psycopg.Connection, query: str, filters: SearchFilters, k: int
) -> list[SearchHit]:
    """Return up to k documents containing the query's words, best match first."""
    # Task 4: Nathan writes this.
    conditions, params = _filter_conditions(filters)
    # joins the keyword match and any filter conditions with AND
    where = " AND ".join(["tsv @@ q"] + conditions)
    rows = conn.execute(
        # {where} pastes in the joined text above
        f"""
        SELECT d.id, ts_rank(tsv, q) AS score
        FROM documents d, websearch_to_tsquery('english', %s) q
        WHERE {where}
        ORDER BY score DESC, d.id ASC
        LIMIT %s
        """,
        # query fills the %s in FROM, params fills the %s in WHERE, and k fills the %s in LIMIT
        [query] + params + [k],
    ).fetchall()
    return [SearchHit(row[0], index, None) for index, row in enumerate(rows, start=1)]


def vector_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    variant: str,
    filters: SearchFilters,
    k: int,
) -> list[SearchHit]:
    """Return up to k documents whose `variant` embedding is closest to query_vector."""
    # Task 4: Nathan writes this.
    register_vector(conn)
    # If filters remove most nearby matches, keep searching the index (results stay sorted)
    conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")

    conditions, params = _filter_conditions(filters)
    # Only this variant's vectors, plus any filter conditions
    where = " AND ".join(["e.variant = %s"] + conditions)

    rows = conn.execute(
        # {where} pastes in the joined text above
        f"""
        SELECT d.id, e.embedding <=> %s AS distance
        FROM embeddings e JOIN documents d ON d.id = e.document_id
        WHERE {where}
        ORDER BY distance ASC, d.id ASC
        LIMIT %s
        """,
        # Slot order: vector (SELECT), variant (WHERE)
        # filter values (WHERE), k (LIMIT)
        [Vector(query_vector), variant] + params + [k],
    ).fetchall()
    # row[1] is the distance; similarity = 1 - distance, so closer = higher
    return [SearchHit(row[0], index, 1 - row[1]) for index, row in enumerate(rows, start=1)]


# --- Task 5: merge keyword and vector results with Reciprocal Rank Fusion (RRF) ---

# How many results each search fetches before fusing. More than k, so a document ranked
# low in one list but high in the other still gets a chance.
CANDIDATES_PER_SEARCH = 50


@dataclass(frozen=True)
class FusedHit:
    document_id: str
    score: float  # RRF score: sum of 1 / (rrf_k + rank) over every list the document is in
    similarity: float | None  # best vector similarity seen; None if only keyword search found it


def reciprocal_rank_fusion(result_lists: list[list[SearchHit]], rrf_k: int = 60) -> list[FusedHit]:
    """Merge ranked lists into one, best first. Ties go to the smaller document_id."""
    # Task 5: Nathan writes this.
    scores: dict[str, float] = {}
    similarities: dict[str, float] = {}

    for hits in result_lists:  # each list (keyword, vector)
        for hit in hits:  # each hit in that list
            # running total + this list's 1 / (rrf_k + rank)
            scores[hit.document_id] = scores.get(hit.document_id, 0) + 1 / (rrf_k + hit.rank)
            if hit.similarity is not None:
                # keep the larger of old and new
                similarities[hit.document_id] = max(
                    similarities.get(hit.document_id, -1.0), hit.similarity
                )
    # the negation (-scores[doc_id]) sorts in descending order of score
    # highest score first and the ties are broken by document_id
    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [FusedHit(doc_id, scores[doc_id], similarities.get(doc_id, None)) for doc_id in order]


def hybrid_search(
    conn: psycopg.Connection, query: str, variant: str, filters: SearchFilters, k: int
) -> list[FusedHit]:
    """Run keyword and vector search, merge them with RRF, and return the top k."""
    # Task 5: Nathan writes this.
    # Fetch more than k from each search: a document ranked low in one list but high in
    # the other can still make the final top k.
    keyword_results = keyword_search(conn, query, filters, CANDIDATES_PER_SEARCH)

    # The question must be embedded with the same model as the documents (the variant),
    # or the distances are meaningless.
    vector_results = vector_search(
        conn, embed_query(query, variant), variant, filters, CANDIDATES_PER_SEARCH
    )
    # RRF uses ranks, not scores: ts_rank and cosine similarity aren't on the same scale.
    fused_results = reciprocal_rank_fusion([keyword_results, vector_results])

    # Only the top k go on to the LLM (Task 6). Returns only the top k results
    return fused_results[:k]
