"""A stand-in for Ollama, so tests run in milliseconds without a model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Call:
    model: str
    system: str
    user: str


@dataclass
class FakeLLM:
    """Returns a canned reply (or raises a canned error) and records every call."""

    reply: dict[str, Any] | Exception
    calls: list[Call] = field(default_factory=list)

    def chat_json(self, model: str, system: str, user: str) -> dict[str, Any]:
        self.calls.append(Call(model, system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply
