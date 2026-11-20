from typing import Any

import psycopg

from tests.fake_llm import FakeLLM
from tests.sample_docs import BRAKES, STALL
from vsrag.config import settings
from vsrag.pipeline import PipelineConfig, ask
from vsrag.search import SearchFilters

QUESTION = "Does the Honda Accord engine stall on the highway?"
STALL_CLAIM = {
    "text": "Owners report the engine stalling at highway speed.",
    "document_ids": [STALL],
}
BRAKE_CLAIM = {"text": "Owners report the brakes failing.", "document_ids": [BRAKES]}


def scripted(claims: list[dict[str, Any]], supported: set[str] | None = None) -> FakeLLM:
    """The answer model returns `claims`; the judge supports the claims whose text is in
    `supported` (all of them if None)."""

    def reply(model: str, system: str, user: str) -> dict[str, Any]:
        if model == settings.answer_model:
            return {"claims": claims}
        claim_text = user.rsplit("Claim: ", 1)[1]
        return {"supported": supported is None or claim_text in supported}

    return FakeLLM(reply)


def test_pipeline_stops_before_llm_when_scope_fails(docs: psycopg.Connection) -> None:
    llm = scripted([STALL_CLAIM])

    result = ask(docs, "Do 2015 Mazda 3 brakes fail?", SearchFilters(), llm)

    assert result.status == "abstained"
    assert result.abstain_reason == "out_of_scope"
    assert result.message is not None
    assert llm.calls == []


def test_pipeline_stops_before_llm_when_retrieval_is_weak(docs: psycopg.Connection) -> None:
    llm = scripted([STALL_CLAIM])

    result = ask(docs, QUESTION, SearchFilters(), llm, PipelineConfig(min_similarity=0.99))

    assert result.abstain_reason == "weak_retrieval"
    assert result.trace.top_similarity is not None
    assert llm.calls == []


def test_answered_question_returns_claims_and_cited_documents(docs: psycopg.Connection) -> None:
    result = ask(docs, QUESTION, SearchFilters(), scripted([STALL_CLAIM]))

    assert result.status == "answered"
    assert result.abstain_reason is None
    assert [claim.text for claim in result.claims] == [STALL_CLAIM["text"]]
    assert [doc.id for doc in result.documents] == [STALL]
    assert result.trace.filters == SearchFilters(make="HONDA", model="ACCORD")  # from the question
    assert set(result.trace.seconds) == {"scope", "search", "answer", "judge"}


def test_documents_follow_citation_order(docs: psycopg.Connection) -> None:
    result = ask(docs, QUESTION, SearchFilters(), scripted([BRAKE_CLAIM, STALL_CLAIM]))

    assert [doc.id for doc in result.documents] == [BRAKES, STALL]


def test_no_claims_abstain_with_no_evidence(docs: psycopg.Connection) -> None:
    llm = scripted([])

    result = ask(docs, QUESTION, SearchFilters(), llm)

    assert result.abstain_reason == "no_evidence"
    assert len(llm.calls) == 1  # the answer model only: nothing for the judge to check


def test_unsupported_claims_abstain(docs: psycopg.Connection) -> None:
    result = ask(docs, QUESTION, SearchFilters(), scripted([STALL_CLAIM], supported=set()))

    assert result.abstain_reason == "unsupported_claims"
    assert result.claims == ()
    assert result.trace.verdicts == [False]


def test_unsupported_claim_is_dropped_from_an_answer(docs: psycopg.Connection) -> None:
    llm = scripted([STALL_CLAIM, BRAKE_CLAIM], supported={STALL_CLAIM["text"]})

    result = ask(docs, QUESTION, SearchFilters(), llm)  # 1 of 2 = 0.5, meets the default ratio

    assert result.status == "answered"
    assert [claim.text for claim in result.claims] == [STALL_CLAIM["text"]]
    assert [doc.id for doc in result.documents] == [STALL]
