"""Answer one question end to end.

    scope check -> hybrid search -> Gate 1 -> answer (LLM) -> judge (LLM) -> Gate 2

Each step can stop early with an abstention. The cheap checks run first, so a question we
can't answer never costs an LLM call.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import psycopg

from vsrag.answer import Claim, generate_answer
from vsrag.documents import Document, fetch_documents
from vsrag.gates import claim_gate, retrieval_gate
from vsrag.judge import judge_claim
from vsrag.llm import LLM
from vsrag.scope import check_scope, known_models
from vsrag.search import FusedHit, SearchFilters, hybrid_search

AbstainReason = Literal["out_of_scope", "weak_retrieval", "no_evidence", "unsupported_claims"]


@dataclass(frozen=True)
class PipelineConfig:
    variant: str = "bge_prefixed"
    k: int = 5  # documents sent to the LLM
    # Placeholders until Task 11 picks them from the accuracy-vs-refusal curve.
    min_similarity: float = 0.65
    min_supported_ratio: float = 0.5


DEFAULT_CONFIG = PipelineConfig()


@dataclass
class Trace:
    """What happened at each step: shown on the page, and used by the evals."""

    filters: SearchFilters | None = None
    hits: list[FusedHit] = field(default_factory=list)
    top_similarity: float | None = None
    draft_claims: tuple[Claim, ...] = ()
    verdicts: list[bool] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AskResult:
    status: Literal["answered", "abstained"]
    claims: tuple[Claim, ...]
    documents: list[Document]  # the documents the claims cite, in the order first cited
    abstain_reason: AbstainReason | None
    message: str | None  # extra detail for an abstention (the scope check's reason)
    trace: Trace


def ask(
    conn: psycopg.Connection,
    question: str,
    filters: SearchFilters,
    llm: LLM,
    config: PipelineConfig = DEFAULT_CONFIG,
    models: dict[str, set[str]] | None = None,
) -> AskResult:
    """Answer `question` with cited claims, or abstain and say why.

    `models` is known_models(conn); pass it in to skip that query on every question.
    """
    trace = Trace()

    def abstain(reason: AbstainReason, message: str | None = None) -> AskResult:
        return AskResult("abstained", (), [], reason, message, trace)

    with _timed(trace, "scope"):
        scope = check_scope(question, filters, models if models is not None else known_models(conn))
    trace.filters = scope.filters
    if not scope.in_scope:
        return abstain("out_of_scope", scope.reason)

    with _timed(trace, "search"):
        trace.hits = hybrid_search(conn, question, config.variant, scope.filters, config.k)
    gate_1 = retrieval_gate(trace.hits, config.min_similarity)
    trace.top_similarity = gate_1.top_similarity
    if not gate_1.passed:
        return abstain("weak_retrieval")

    documents = fetch_documents(conn, [hit.document_id for hit in trace.hits])
    with _timed(trace, "answer"):
        draft = generate_answer(question, documents, llm)
    trace.draft_claims = draft.claims
    if not draft.claims:
        return abstain("no_evidence")

    with _timed(trace, "judge"):
        trace.verdicts = [judge_claim(claim, documents, llm) for claim in draft.claims]
    gate_2 = claim_gate(draft.claims, trace.verdicts, config.min_supported_ratio)
    if not gate_2.passed:
        return abstain("unsupported_claims")

    # The sources to show: only documents a kept claim cites, in the order first cited.
    cited_ids = dict.fromkeys(i for claim in gate_2.kept_claims for i in claim.document_ids)
    by_id = {doc.id: doc for doc in documents}
    return AskResult(
        "answered", gate_2.kept_claims, [by_id[i] for i in cited_ids], None, None, trace
    )


@contextmanager
def _timed(trace: Trace, step: str) -> Iterator[None]:
    """Record how long the `with` block took, in seconds, under `step`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        trace.seconds[step] = round(time.perf_counter() - start, 3)
