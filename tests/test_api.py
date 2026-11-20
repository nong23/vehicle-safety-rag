import pytest
from fastapi.testclient import TestClient

from vsrag import api
from vsrag.answer import Claim
from vsrag.documents import Document
from vsrag.llm import LLMUnavailableError
from vsrag.pipeline import AbstainReason, AskResult, Trace

COMPLAINT = Document("complaint:1", "complaint", "Owner complaint", "ENGINE STALLED.")
RECALL = Document("recall:20V314000", "recall", "Recall 20V314000 by Honda", "Fuel pump.")
ANSWERED = AskResult(
    status="answered",
    claims=(
        Claim("Owners report stalling.", ("complaint:1",)),
        Claim("Honda recalled the fuel pump.", ("recall:20V314000", "complaint:1")),
    ),
    documents=[COMPLAINT, RECALL],
    abstain_reason=None,
    message=None,
    trace=Trace(),
)


def abstained(reason: AbstainReason, message: str | None = None) -> AskResult:
    return AskResult("abstained", (), [], reason, message, Trace())


class FakeAsk:
    """Stands in for the whole pipeline: returns a canned result and records questions."""

    def __init__(self, result: AskResult | Exception) -> None:
        self.result = result
        self.questions: list[str] = []

    def __call__(self, question: str) -> AskResult:
        self.questions.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client(result: AskResult | Exception = ANSWERED) -> tuple[TestClient, FakeAsk]:
    fake = FakeAsk(result)
    return TestClient(api.create_app(fake)), fake


def test_home_page_has_a_question_form() -> None:
    page = client()[0].get("/")

    assert page.status_code == 200
    assert 'hx-post="/ask"' in page.text
    assert 'name="question"' in page.text


def test_htmx_is_served_from_the_app_not_the_internet() -> None:
    test_client = client()[0]

    assert '<script src="/static/htmx.min.js"' in test_client.get("/").text
    assert test_client.get("/static/htmx.min.js").status_code == 200


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_returns_422(question: str) -> None:
    test_client, fake = client()

    response = test_client.post("/ask", data={"question": question})

    assert response.status_code == 422
    assert "Please enter a question" in response.text
    assert fake.questions == []  # never reaches the pipeline


def test_question_over_500_characters_returns_422() -> None:
    test_client, fake = client()

    assert test_client.post("/ask", data={"question": "a" * 501}).status_code == 422
    assert test_client.post("/ask", data={"question": "a" * 500}).status_code == 200
    assert fake.questions == ["a" * 500]


def test_question_is_trimmed_before_asking() -> None:
    test_client, fake = client()

    test_client.post("/ask", data={"question": "  brakes?  "})

    assert fake.questions == ["brakes?"]


def test_ollama_down_shows_clear_error() -> None:
    response = client(LLMUnavailableError("connection refused"))[0].post(
        "/ask", data={"question": "Do Accords stall?"}
    )

    assert response.status_code == 503
    assert "Ollama" in response.text
    assert "connection refused" not in response.text  # no raw internals on the page


def test_answer_shows_claims_with_numbered_citations() -> None:
    page = client()[0].post("/ask", data={"question": "Do Accords stall?"}).text

    assert "Owners report stalling." in page
    assert 'href="#source-1"' in page and 'href="#source-2"' in page
    assert 'id="source-1"' in page and 'id="source-2"' in page


def test_document_html_is_escaped() -> None:
    sneaky = Document("complaint:1", "complaint", "Owner complaint", "<script>alert(1)</script>")
    result = AskResult(
        "answered", (Claim("Owners report it.", ("complaint:1",)),), [sneaky], None, None, Trace()
    )

    page = client(result)[0].post("/ask", data={"question": "Do Accords stall?"}).text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_complaint_citation_is_labeled_unverified() -> None:
    page = client()[0].post("/ask", data={"question": "Do Accords stall?"}).text

    assert "Owner complaint (unverified)" in page
    assert "Official recall" in page


def test_out_of_scope_shows_the_scope_reason() -> None:
    result = abstained("out_of_scope", "I only have data for Ford, Honda, Toyota, Hyundai and Kia.")

    page = client(result)[0].post("/ask", data={"question": "2015 Mazda 3?"}).text

    assert "I only have data for Ford, Honda, Toyota, Hyundai and Kia." in page


@pytest.mark.parametrize("reason", ["weak_retrieval", "no_evidence", "unsupported_claims"])
def test_abstention_says_not_enough_evidence(reason: AbstainReason) -> None:
    page = client(abstained(reason))[0].post("/ask", data={"question": "Do Accords stall?"}).text

    assert "Not enough evidence" in page


@pytest.mark.parametrize(
    ("database", "ollama", "status"), [(True, True, 200), (True, False, 503), (False, True, 503)]
)
def test_health_reports_each_dependency(
    monkeypatch: pytest.MonkeyPatch, database: bool, ollama: bool, status: int
) -> None:
    monkeypatch.setattr(api, "database_ok", lambda: database)
    monkeypatch.setattr(api, "ollama_ok", lambda: ollama)

    response = client()[0].get("/health")

    assert response.status_code == status
    assert response.json() == {
        "database": "ok" if database else "down",
        "ollama": "ok" if ollama else "down",
    }
