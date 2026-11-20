"""The web app: a page to ask a question, plus a health check.

Run it from the project folder with:  uvicorn vsrag.api:app
then open http://127.0.0.1:8000
"""

import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vsrag.config import settings
from vsrag.db import get_conn
from vsrag.llm import LLMUnavailableError, OllamaLLM
from vsrag.pipeline import AskResult, ask
from vsrag.scope import known_models
from vsrag.search import SearchFilters

HERE = Path(__file__).parent
logger = logging.getLogger(__name__)
MAX_QUESTION_CHARS = 500

SOURCE_LABELS = {"complaint": "Owner complaint (unverified)", "recall": "Official recall"}
ABSTAIN_MESSAGES = {
    "weak_retrieval": "No documents closely match this question.",
    "no_evidence": "The closest documents don't answer this question.",
    "unsupported_claims": "The draft answer wasn't supported by its sources, so it isn't shown.",
}
EMPTY_OR_LONG = f"Please enter a question of 1 to {MAX_QUESTION_CHARS} characters."
OLLAMA_DOWN = (
    "The local language model isn't responding. Make sure Ollama is running "
    "(open the Ollama app, or run `ollama serve`), then try again."
)

AskQuestion = Callable[[str], AskResult]


def create_app(ask_question: AskQuestion | None = None) -> FastAPI:
    """Build the app. Tests pass a fake `ask_question`; the real one uses the database
    and Ollama."""
    app = FastAPI(title="Vehicle Safety RAG")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")  # escapes HTML by default
    answer_question = ask_question or _real_ask_question()

    def result_page(request: Request, status_code: int = 200, **context: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "result.html", {"labels": SOURCE_LABELS, **context}, status_code=status_code
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {"max_chars": MAX_QUESTION_CHARS})

    # A plain `def` (not `async def`): FastAPI runs it in a worker thread, so a 10-second
    # LLM call doesn't freeze the server for other requests.
    @app.post("/ask", response_class=HTMLResponse)
    def ask_endpoint(request: Request, question: Annotated[str, Form()] = "") -> HTMLResponse:
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            return result_page(request, 422, error=EMPTY_OR_LONG)
        try:
            result = answer_question(question)
        except LLMUnavailableError as error:
            logger.error("Ollama unavailable: %s", error)  # the details, for the server console
            return result_page(request, 503, error=OLLAMA_DOWN)
        message = result.message or ABSTAIN_MESSAGES.get(result.abstain_reason or "")
        numbers = {doc.id: n for n, doc in enumerate(result.documents, start=1)}
        return result_page(request, result=result, message=message, numbers=numbers)

    @app.get("/health")
    def health() -> JSONResponse:
        checks = {"database": database_ok(), "ollama": ollama_ok()}
        return JSONResponse(
            {name: "ok" if ok else "down" for name, ok in checks.items()},
            status_code=200 if all(checks.values()) else 503,
        )

    return app


def _real_ask_question() -> AskQuestion:
    llm = OllamaLLM()
    models: dict[str, set[str]] = {}  # loaded on the first question, then reused

    def ask_question(question: str) -> AskResult:
        with get_conn() as conn:
            if not models:
                models.update(known_models(conn))
            return ask(conn, question, SearchFilters(), llm, models=models)

    return ask_question


def database_ok() -> bool:
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


def ollama_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{settings.ollama_url}/api/version", timeout=3):
            return True
    except OSError:
        return False


app = create_app()
