"""The two abstention gates: when to say "not enough evidence" instead of answering.

Gate 1 (retrieval gate) runs before the LLM: if even the best match isn't close enough in
meaning, don't ask the LLM at all. Gate 2 (claim gate) runs after the judge: drop the
claims the documents don't support, and abstain if too few survive.
"""

from dataclasses import dataclass

from vsrag.answer import Claim
from vsrag.search import FusedHit


@dataclass(frozen=True)
class GateResult:
    passed: bool
    top_similarity: float | None  # best vector similarity among the hits; None if none had one


@dataclass(frozen=True)
class ClaimGateResult:
    passed: bool
    kept_claims: tuple[Claim, ...]  # the supported claims, kept even when the gate fails


def retrieval_gate(hits: list[FusedHit], min_similarity: float) -> GateResult:
    """Pass if the best vector similarity among `hits` is at least `min_similarity`."""
    # Task 7: Nathan writes this.
    # only hits that have a similarity
    # Keyword-only hits have no similarity: matching words doesn't prove the meaning matches.
    similarities = [hit.similarity for hit in hits if hit.similarity is not None]
    top = max(similarities, default=None)  # None if no hit had one
    # passed = there's a top AND it's >= min_similarity
    # "and" stops early, so top >= min_similarity never runs when top is None (that comparison crashes).
    return GateResult(top is not None and top >= min_similarity, top)


def claim_gate(
    claims: tuple[Claim, ...], verdicts: list[bool], min_supported_ratio: float
) -> ClaimGateResult:
    """Keep the claims whose verdict is True. Pass if at least one is kept and the share
    kept (kept / all claims) is at least `min_supported_ratio`."""
    # Task 7: Nathan writes this.
    # keep the claims whose verdict is True
    # strict=True: a missing verdict raises an error instead of silently skipping a claim, causing a bug.
    kept = tuple(claim for claim, verdict in zip(claims, verdicts, strict=True) if verdict)
    # at least one claim kept AND (kept / all claims) >= min_supported_ratio
    # Abstain if too few claims check out: if most were made up, the rest are suspect too.
    # "and" stops early, so with no claims we never divide by zero.
    passed = len(kept) >= 1 and len(kept) / len(claims) >= min_supported_ratio
    return ClaimGateResult(passed, kept)
