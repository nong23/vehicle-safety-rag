import pytest

from vsrag.answer import Claim
from vsrag.gates import claim_gate, retrieval_gate
from vsrag.search import FusedHit

# --- Gate 1: retrieval gate (before the LLM) ---


def hit(doc_id: str, similarity: float | None, score: float = 0.03) -> FusedHit:
    return FusedHit(doc_id, score, similarity)


def test_no_hits_fail_retrieval_gate() -> None:
    result = retrieval_gate([], min_similarity=0.65)

    assert not result.passed
    assert result.top_similarity is None


def test_uses_similarity_not_fusion_score() -> None:
    # A high RRF score only means "ranked first"; it says nothing about relevance.
    assert not retrieval_gate([hit("a", 0.40, score=0.9)], min_similarity=0.65).passed
    assert retrieval_gate([hit("a", 0.80, score=0.001)], min_similarity=0.65).passed


def test_best_similarity_among_hits_is_used() -> None:
    result = retrieval_gate([hit("a", 0.50), hit("b", 0.80), hit("c", 0.60)], min_similarity=0.65)

    assert result.passed
    assert result.top_similarity == pytest.approx(0.80)


def test_keyword_only_hit_with_no_similarity_is_handled() -> None:
    only_keyword = retrieval_gate([hit("a", None)], min_similarity=0.65)
    assert not only_keyword.passed  # no similarity at all, so nothing shows relevance
    assert only_keyword.top_similarity is None

    mixed = retrieval_gate([hit("a", None), hit("b", 0.70)], min_similarity=0.65)
    assert mixed.passed
    assert mixed.top_similarity == pytest.approx(0.70)


def test_threshold_boundary_is_inclusive() -> None:
    assert retrieval_gate([hit("a", 0.65)], min_similarity=0.65).passed


# --- Gate 2: claim gate (after the judge) ---

A = Claim("Owners report stalling.", ("complaint:1",))
B = Claim("Owners report fires.", ("complaint:2",))
C = Claim("Honda recalled the fuel pump.", ("recall:20V314000",))


def test_unsupported_claims_are_removed() -> None:
    result = claim_gate((A, B, C), [True, False, True], min_supported_ratio=0.5)

    assert result.passed
    assert result.kept_claims == (A, C)


def test_too_few_supported_claims_fail_the_gate() -> None:
    # 1 of 3 supported = 0.33, below 0.5: too much of the answer was made up to trust the rest
    result = claim_gate((A, B, C), [True, False, False], min_supported_ratio=0.5)

    assert not result.passed
    assert result.kept_claims == (A,)  # still reported, for the trace


def test_supported_ratio_boundary_is_inclusive() -> None:
    assert claim_gate((A, B), [True, False], min_supported_ratio=0.5).passed


def test_no_claims_fail_the_gate() -> None:
    result = claim_gate((), [], min_supported_ratio=0.5)

    assert not result.passed
    assert result.kept_claims == ()


def test_all_unsupported_fails_even_with_zero_ratio() -> None:
    # 0 of 2 >= 0.0 is true, but an answer with no supported claims says nothing
    assert not claim_gate((A, B), [False, False], min_supported_ratio=0.0).passed


def test_each_claim_needs_exactly_one_verdict() -> None:
    with pytest.raises(ValueError):
        claim_gate((A, B), [True], min_supported_ratio=0.5)
