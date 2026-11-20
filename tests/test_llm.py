import io
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from vsrag.llm import LLMOutputError, LLMUnavailableError, OllamaLLM


class FakeServer:
    """Replaces urllib's urlopen: records requests and plays back responses in order."""

    def __init__(self, *responses: str | int | OSError) -> None:
        # a str is a reply's content; an int is an HTTP error; an OSError is a network failure
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def urlopen(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        assert isinstance(request.data, bytes)
        self.requests.append(json.loads(request.data))
        response = self.responses.pop(0)
        if isinstance(response, int):
            raise urllib.error.HTTPError(request.full_url, response, "error", Message(), None)
        if isinstance(response, OSError):
            raise response
        return io.BytesIO(json.dumps({"message": {"content": response}}).encode())


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr("vsrag.llm.time.sleep", lambda seconds: None)  # skip the retry wait
    return fake


def test_request_turns_thinking_off_and_sets_context(server: FakeServer) -> None:
    server.responses = ['{"answer": "hi"}']

    OllamaLLM("http://ollama.test").chat_json("qwen3:8b", "Be brief.", "Say hi")

    sent = server.requests[0]
    assert sent["model"] == "qwen3:8b"
    assert sent["think"] is False
    assert sent["stream"] is False
    assert sent["format"] == "json"
    assert sent["options"]["num_ctx"] == 4096
    assert sent["options"]["temperature"] == 0
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hi"},
    ]


def test_reply_is_parsed_as_json(server: FakeServer) -> None:
    server.responses = ['{"answer": "hi"}']

    assert OllamaLLM("http://ollama.test").chat_json("m", "s", "u") == {"answer": "hi"}


@pytest.mark.parametrize("content", ["not json", '["a list", "not an object"]', ""])
def test_reply_that_is_not_a_json_object_raises(server: FakeServer, content: str) -> None:
    server.responses = [content]

    with pytest.raises(LLMOutputError):
        OllamaLLM("http://ollama.test").chat_json("m", "s", "u")


def test_server_errors_are_retried_up_to_3_attempts(server: FakeServer) -> None:
    server.responses = [500, 500, '{"answer": "hi"}']  # as on Ollama's first GPU load

    assert OllamaLLM("http://ollama.test").chat_json("m", "s", "u") == {"answer": "hi"}
    assert len(server.requests) == 3


def test_third_server_error_means_unavailable(server: FakeServer) -> None:
    server.responses = [500, 500, 500]

    with pytest.raises(LLMUnavailableError):
        OllamaLLM("http://ollama.test").chat_json("m", "s", "u")
    assert len(server.requests) == 3


def test_client_error_is_not_retried(server: FakeServer) -> None:
    server.responses = [404]  # e.g. the model isn't installed: retrying won't help

    with pytest.raises(LLMUnavailableError):
        OllamaLLM("http://ollama.test").chat_json("m", "s", "u")
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError(ConnectionRefusedError()), TimeoutError("timed out")],
    ids=["not running", "too slow"],
)
def test_unreachable_ollama_means_unavailable(server: FakeServer, failure: OSError) -> None:
    server.responses = [failure]

    with pytest.raises(LLMUnavailableError):
        OllamaLLM("http://ollama.test").chat_json("m", "s", "u")
