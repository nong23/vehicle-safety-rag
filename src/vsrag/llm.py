"""Talk to a local LLM through Ollama and get JSON back.

The rest of the code depends only on the small `LLM` protocol below, so tests can pass in
a fake (tests/fake_llm.py) and never need a model running.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from vsrag.config import settings

CONTEXT_TOKENS = 4096  # 5 documents are ~600 tokens, so this leaves plenty of room
KEEP_ALIVE = "30m"  # keep the model loaded between questions; reloading costs ~10 s
TIMEOUT_SECONDS = 600  # the first request after a reboot took ~2 minutes (measured)
MAX_ATTEMPTS = 3  # for server errors (5xx); see _post
RETRY_WAIT_SECONDS = 2


class LLMOutputError(Exception):
    """The model's reply wasn't the JSON object we asked for."""


class LLMUnavailableError(Exception):
    """Ollama couldn't give a reply: not running, too slow, model missing, or failing."""


class LLM(Protocol):
    def chat_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        """Send one system and one user message; return the reply as a JSON object."""
        ...


class OllamaLLM:
    def __init__(self, url: str = settings.ollama_url) -> None:
        self.url = url.rstrip("/")

    def chat_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        reply = self._post(
            "/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Plain JSON mode, not a JSON schema: with a schema, qwen3:8b returned an
                # empty list for 2 of 6 answerable questions; plain mode answered all 6 and
                # still returned nothing for 2 unanswerable ones (measured 2026-11-19).
                # Callers check the reply's shape themselves.
                "format": "json",
                "stream": False,  # one complete reply instead of word-by-word pieces
                "think": False,  # qwen3 would otherwise write long reasoning first
                "keep_alive": KEEP_ALIVE,
                # temperature 0: always pick the most likely word, so reruns match
                "options": {"num_ctx": CONTEXT_TOKENS, "temperature": 0},
            },
        )
        return parse_json_object(reply["message"]["content"])

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    return _send(request)
                except urllib.error.HTTPError as error:
                    # 5xx = the server failed, and may work next time. Measured: on its first
                    # model load after starting, Ollama's GPU setup crashed twice ("CUDA error:
                    # shared object initialization failed") and worked on the 3rd try.
                    # 4xx = our request is wrong (e.g. the model isn't installed): no retry.
                    if error.code < 500 or attempt == MAX_ATTEMPTS:
                        raise
                    time.sleep(RETRY_WAIT_SECONDS)
            raise AssertionError("unreachable: the loop returns or raises")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            # One error type for "no usable reply", so the page can show one clear message.
            # (HTTPError is a kind of URLError, so it lands here too.)
            raise LLMUnavailableError(f"Ollama at {self.url} failed: {error}") from error


def _send(request: urllib.request.Request) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        reply: dict[str, Any] = json.load(response)
    return reply


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the model's reply, which must be a JSON object."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMOutputError(f"reply is not JSON: {text[:200]!r}") from error
    if not isinstance(value, dict):
        raise LLMOutputError(f"reply is not a JSON object: {text[:200]!r}")
    return value
