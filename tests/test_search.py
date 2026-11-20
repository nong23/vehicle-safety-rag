import psycopg
import pytest

from tests.sample_docs import (
    ACCORD_2019,
    BRAKES,
    DOCUMENTS,
    PAINT,
    RECALL,
    STALL,
    VARIANT,
    complaint,
)
from vsrag.db import init_schema
from vsrag.embed import embed_missing, embed_query, ensure_index
from vsrag.ingest.load import load_documents
from vsrag.ingest.parse import Vehicle
from vsrag.search import (
    SearchFilters,
    SearchHit,
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    vector_search,
)


def search(conn: psycopg.Connection, kind: str, filters: SearchFilters, k: int = 10) -> list[str]:
    """Run one kind of search with a query every document matches; return the document ids."""
    if kind == "keyword":
        hits = keyword_search(conn, "vehicle", filters, k)
    else:
        hits = vector_search(conn, embed_query("vehicle problem", VARIANT), VARIANT, filters, k)
    return [hit.document_id for hit in hits]


# --- Keyword search ---


def test_keyword_finds_exact_campaign_number(docs: psycopg.Connection) -> None:
    hits = keyword_search(docs, "18V777", SearchFilters(), k=5)

    assert [hit.document_id for hit in hits] == [RECALL]
    assert hits[0].rank == 1
    assert hits[0].similarity is None  # keyword search has no similarity score


def test_keyword_ranks_better_matches_first(conn: psycopg.Connection) -> None:
    init_schema(conn)
    vehicle = [Vehicle(*ACCORD_2019, "SERVICE BRAKES")]
    load_documents(
        conn,
        [
            complaint("complaint:10", "THE RADIO BROKE. THE BRAKES ARE FINE.", vehicle),
            complaint("complaint:11", "BRAKES FAILED. BRAKES GRIND. BRAKES SQUEAL.", vehicle),
        ],
    )

    hits = keyword_search(conn, "brakes", SearchFilters(), k=5)

    assert [hit.document_id for hit in hits] == ["complaint:11", "complaint:10"]
    assert [hit.rank for hit in hits] == [1, 2]


@pytest.mark.parametrize("query", ["???", "", "   "])
def test_keyword_punctuation_only_returns_empty(docs: psycopg.Connection, query: str) -> None:
    assert keyword_search(docs, query, SearchFilters(), k=5) == []


# --- Vector search ---


def test_vector_returns_nearest_first(docs: psycopg.Connection) -> None:
    query = embed_query("car engine shut off on the freeway", VARIANT)

    hits = vector_search(docs, query, VARIANT, SearchFilters(), k=10)

    assert hits[0].document_id == STALL
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    similarities = [hit.similarity for hit in hits]
    assert all(s is not None and 0 < s <= 1 for s in similarities)
    assert similarities == sorted(similarities, reverse=True)  # type: ignore[type-var]


def test_vector_only_searches_the_given_variant(docs: psycopg.Connection) -> None:
    embed_missing(docs, "bge_plain")  # now every document has two embeddings

    ids = search(docs, "vector", SearchFilters())

    assert sorted(ids) == sorted(doc.id for doc in DOCUMENTS)


def test_filter_matching_fewer_than_k_returns_what_exists(docs: psycopg.Connection) -> None:
    """On the real table, the HNSW index gathers the ~40 nearest vectors, *then* the filter
    runs. If none of those 40 pass the filter, you get nothing back, even though matching
    documents exist. Here the 40 is shrunk to 1, so the problem shows up with 4 documents."""
    ensure_index(docs, VARIANT)
    docs.execute("SET LOCAL enable_seqscan = off")  # make Postgres use the index, as on 49k rows
    docs.execute("SET LOCAL hnsw.ef_search = 1")
    query = embed_query("engine stalled at highway speed", VARIANT)  # nearest is the Honda

    hits = vector_search(docs, query, VARIANT, SearchFilters(make="FORD"), k=10)

    assert [hit.document_id for hit in hits] == [RECALL]


# --- Filters (each test runs once for keyword search and once for vector search) ---

KINDS = pytest.mark.parametrize("kind", ["keyword", "vector"])


@KINDS
def test_filter_limits_results_to_vehicle(docs: psycopg.Connection, kind: str) -> None:
    ids = search(docs, kind, SearchFilters(make="HONDA", model="ACCORD", year=2019))

    assert sorted(ids) == [STALL, BRAKES]


@KINDS
def test_filter_by_source(docs: psycopg.Connection, kind: str) -> None:
    assert search(docs, kind, SearchFilters(source="recall")) == [RECALL]


@KINDS
def test_filter_matching_nothing_returns_empty(docs: psycopg.Connection, kind: str) -> None:
    assert search(docs, kind, SearchFilters(make="MAZDA")) == []


@KINDS
def test_returns_at_most_k(docs: psycopg.Connection, kind: str) -> None:
    assert len(search(docs, kind, SearchFilters(), k=2)) == 2


@KINDS
def test_document_with_several_parts_appears_once(docs: psycopg.Connection, kind: str) -> None:
    ids = search(docs, kind, SearchFilters(make="HONDA"))

    assert sorted(ids) == [STALL, PAINT, BRAKES]


# --- Reciprocal rank fusion (RRF): pure Python, no database ---


def ranked(*ids: str, similarity: float | None = None) -> list[SearchHit]:
    """A ranked list of hits: the first id is rank 1, the next rank 2, and so on."""
    return [SearchHit(doc_id, rank, similarity) for rank, doc_id in enumerate(ids, start=1)]


def test_score_matches_hand_calculation() -> None:
    # "a" is rank 1 in the first list and rank 3 in the second: 1/(60+1) + 1/(60+3)
    fused = reciprocal_rank_fusion([ranked("a"), ranked("x", "y", "a")])

    assert fused[0].document_id == "a"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_k_is_a_parameter() -> None:
    # With rrf_k=0, rank 1 scores 1/(0+1) = 1.0
    assert reciprocal_rank_fusion([ranked("a")], rrf_k=0)[0].score == pytest.approx(1.0)


def test_document_in_both_lists_beats_document_in_one() -> None:
    keyword = ranked("b", "a")
    vector = ranked("c", "a")

    fused = reciprocal_rank_fusion([keyword, vector])

    # a: 1/62 + 1/62 ≈ 0.0323. b and c: 1/61 ≈ 0.0164 each (tie, so alphabetical)
    assert [hit.document_id for hit in fused] == ["a", "b", "c"]


def test_ties_break_by_document_id() -> None:
    fused = reciprocal_rank_fusion([ranked("b"), ranked("a")])

    assert [hit.document_id for hit in fused] == ["a", "b"]


def test_empty_lists_return_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_vector_similarity_is_kept_for_gate_1() -> None:
    keyword = ranked("a", "b")  # keyword hits have no similarity
    vector = [SearchHit("a", 1, 0.83)]

    # vector list first, so a later keyword hit (similarity None) must not erase 0.83
    fused = {hit.document_id: hit for hit in reciprocal_rank_fusion([vector, keyword])}

    assert fused["a"].similarity == pytest.approx(0.83)
    assert fused["b"].similarity is None  # only keyword search found it


def test_highest_similarity_is_kept() -> None:
    fused = reciprocal_rank_fusion([[SearchHit("a", 1, 0.9)], [SearchHit("a", 2, 0.7)]])

    assert fused[0].similarity == pytest.approx(0.9)


# --- Hybrid search: keyword + vector, merged with RRF ---


def test_hybrid_puts_exact_keyword_match_first(docs: psycopg.Connection) -> None:
    hits = hybrid_search(docs, "18V777", VARIANT, SearchFilters(), k=5)

    assert hits[0].document_id == RECALL
    assert all(hit.similarity is not None for hit in hits)  # vector search saw all 4


def test_hybrid_finds_by_meaning_when_no_words_match(docs: psycopg.Connection) -> None:
    # No document contains "freeway", so keyword search finds nothing; vector search still works
    hits = hybrid_search(docs, "car engine shut off on the freeway", VARIANT, SearchFilters(), k=5)

    assert hits[0].document_id == STALL


def test_hybrid_returns_at_most_k(docs: psycopg.Connection) -> None:
    assert len(hybrid_search(docs, "vehicle", VARIANT, SearchFilters(), k=2)) == 2


def test_hybrid_applies_filters(docs: psycopg.Connection) -> None:
    hits = hybrid_search(docs, "vehicle", VARIANT, SearchFilters(make="HONDA"), k=10)

    assert sorted(hit.document_id for hit in hits) == [STALL, PAINT, BRAKES]
