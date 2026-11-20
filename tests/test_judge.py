import pytest

from tests.fake_llm import FakeLLM
from vsrag.answer import Claim
from vsrag.config import settings
from vsrag.documents import Document
from vsrag.judge import judge_claim
from vsrag.llm import LLMOutputError

STALL = Document("complaint:1", "complaint", "Owner complaint", "ENGINE STALLED ON THE HIGHWAY.")
PAINT = Document("complaint:2", "complaint", "Owner complaint", "PAINT PEELING ON THE HOOD.")
CLAIM = Claim("Owners report the engine stalling.", ("complaint:1",))


@pytest.mark.parametrize(
    ("reply", "verdict"), [({"supported": True}, True), ({"supported": False}, False)]
)
def test_verdict_comes_from_the_judge(reply: dict[str, object], verdict: bool) -> None:
    assert judge_claim(CLAIM, [STALL, PAINT], FakeLLM(reply)) is verdict


def test_judge_only_sees_the_cited_documents() -> None:
    llm = FakeLLM({"supported": True})

    judge_claim(CLAIM, [STALL, PAINT], llm)

    assert STALL.body in llm.calls[0].user
    assert PAINT.body not in llm.calls[0].user
    assert llm.calls[0].user.endswith(f"Claim: {CLAIM.text}")


def test_claim_citing_no_given_document_is_unsupported_without_asking() -> None:
    llm = FakeLLM({"supported": True})

    assert judge_claim(Claim("Owners report stalling.", ("complaint:9",)), [STALL], llm) is False
    assert llm.calls == []


@pytest.mark.parametrize(
    "reply", [LLMOutputError("not JSON"), {}, {"supported": "true"}, {"supported": 1}]
)
def test_unclear_verdict_counts_as_unsupported(reply: dict[str, object] | Exception) -> None:
    assert judge_claim(CLAIM, [STALL], FakeLLM(reply)) is False


def test_complaint_stated_as_fact_is_unsupported_without_asking() -> None:
    llm = FakeLLM({"supported": True})

    assert judge_claim(Claim("The engine stalls.", ("complaint:1",)), [STALL], llm) is False
    assert llm.calls == []


@pytest.mark.parametrize(
    "text", ["Owners report the engine stalling.", "One driver says the engine stalled.",
             "Complaints describe stalling.", "Stalling was reported on the highway."],
)  # fmt: skip
def test_complaint_with_attribution_goes_to_the_judge(text: str) -> None:
    llm = FakeLLM({"supported": True})

    assert judge_claim(Claim(text, ("complaint:1",)), [STALL], llm) is True
    assert len(llm.calls) == 1


def test_recall_can_be_stated_as_fact() -> None:
    recall = Document("recall:20V314000", "recall", "Recall 20V314000", "Defect: fuel pump.")
    llm = FakeLLM({"supported": True})

    assert judge_claim(Claim("Honda recalled the fuel pump.", (recall.id,)), [recall], llm)


def test_uses_the_judge_model() -> None:
    llm = FakeLLM({"supported": True})

    judge_claim(CLAIM, [STALL], llm)

    assert llm.calls[0].model == settings.judge_model
    assert llm.calls[0].model != settings.answer_model  # a different model checks the work
