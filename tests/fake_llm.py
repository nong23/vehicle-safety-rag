"""A stand-in for Ollama, so tests run in milliseconds without a model."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Call:
    model: str
    system: str
    user: str


@dataclass
class FakeLLM:
    """Returns a canned reply (or raises a canned error) and records every call.

    `reply` can also be a function of (model, system, user), for tests where the answer
    model and the judge need different replies.
    """

    reply: dict[str, Any] | Exception | Callable[[str, str, str], dict[str, Any]]
    calls: list[Call] = field(default_factory=list)

    def chat_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        self.calls.append(Call(model, system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        if callable(self.reply):
            return self.reply(model, system, user)
        return self.reply
