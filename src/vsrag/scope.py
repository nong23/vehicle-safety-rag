"""Decide whether a question is about vehicles we have data for, and turn any vehicle it
names into search filters.

We only loaded 5 makes and model years 2018-2023. Without this check, a question about a
2015 Mazda would be answered from whatever Ford or Toyota documents happened to match.
"""

import re
from dataclasses import dataclass, replace
from functools import cache

import psycopg

from vsrag.ingest.parse import FIRST_YEAR, LAST_YEAR, MAKES
from vsrag.search import SearchFilters

OUT_OF_SCOPE_REASON = (
    "I only have NHTSA data for Ford, Honda, Toyota, Hyundai and Kia vehicles, "
    f"model years {FIRST_YEAR}-{LAST_YEAR}."
)

# Makes we didn't load. A question naming one of these, and none of ours, is out of scope.
# Left out on purpose: GENESIS (also a Hyundai model name) and MINI ("mini van").
OTHER_MAKES = frozenset(
    {
        "ACURA", "ALFA ROMEO", "AUDI", "BMW", "BUICK", "CADILLAC", "CHEVROLET", "CHEVY",
        "CHRYSLER", "DODGE", "FIAT", "GMC", "HARLEY", "INFINITI", "JAGUAR", "JEEP",
        "KAWASAKI", "LAND ROVER", "LEXUS", "LINCOLN", "LUCID", "MAZDA", "MERCEDES",
        "MITSUBISHI", "NISSAN", "POLESTAR", "PORSCHE", "RAM", "RIVIAN", "SUBARU",
        "SUZUKI", "TESLA", "VOLKSWAGEN", "VOLVO", "VW", "YAMAHA",
    }
)  # fmt: skip

# Models that are also everyday words ("the seat belt doesn't fit"). They only count when
# the make is known or a year comes right before them ("2019 Pilot").
COMMON_WORD_MODELS = frozenset(
    {
        "86", "ACCENT", "CARNIVAL", "CLARITY", "CROWN", "EDGE", "ESCAPE", "EXPEDITION",
        "EXPLORER", "FIT", "FLEX", "FOCUS", "FORTE", "FUSION", "GT", "INSIGHT", "MAVERICK",
        "METROPOLITAN", "NAVI", "ODYSSEY", "PASSPORT", "PILOT", "RANGER", "RIO", "SOUL",
        "STINGER", "TRANSIT", "VENUE",
    }
)  # fmt: skip

# A 4-digit year, but not a quantity like "2000 miles" or "$2000".
YEAR = re.compile(
    r"(?<![$\d,.])\b(19[5-9]\d|20[0-4]\d)\b"
    r"(?!\s*(?:miles?|mi|km|rpm|lbs?|pounds|dollars|feet|ft)\b)",
    re.IGNORECASE,
)
YEAR_BEFORE = r"\b(?:19|20)\d\d\s+"


@dataclass(frozen=True)
class ScopeResult:
    in_scope: bool
    filters: SearchFilters  # the caller's filters, plus any vehicle the question names
    reason: str | None = None  # why it's out of scope, to show the user


def known_models(conn: psycopg.Connection) -> dict[str, set[str]]:
    """Every model we have documents for, by make. Skips NHTSA's "REDUNDANT ..." entries."""
    models: dict[str, set[str]] = {}
    for make, model in conn.execute(
        "SELECT DISTINCT make, model FROM document_vehicles WHERE model NOT LIKE 'REDUNDANT%'"
    ):
        models.setdefault(make, set()).add(model)
    return models


def check_scope(question: str, filters: SearchFilters, models: dict[str, set[str]]) -> ScopeResult:
    """Reject questions about vehicles we have no data for; fill filters from the question.

    Filters the caller already set (e.g. from the page's dropdowns) win over the question.
    """
    text = question.upper()
    our_makes = sorted(make for make in MAKES if _mentions(text, make))
    names_other_make = any(_mentions(text, make) for make in OTHER_MAKES)
    years = {int(year) for year in YEAR.findall(question)}
    years_in_range = sorted(year for year in years if FIRST_YEAR <= year <= LAST_YEAR)

    out_of_scope = (
        (names_other_make and not our_makes)
        or (years and not years_in_range)
        or (filters.make is not None and filters.make not in MAKES)
        or (filters.year is not None and not FIRST_YEAR <= filters.year <= LAST_YEAR)
    )
    if out_of_scope:
        return ScopeResult(False, filters, OUT_OF_SCOPE_REASON)

    make = filters.make
    if make is None and len(our_makes) == 1:
        make = our_makes[0]

    model = filters.model
    if model is None:
        found = _find_model(text, models, make)
        if found:
            found_make, model = found
            make = make or found_make

    year = filters.year
    if year is None and len(years_in_range) == 1:
        year = years_in_range[0]

    return ScopeResult(True, replace(filters, make=make, model=model, year=year))


def _mentions(text: str, name: str) -> bool:
    """Whether `name` appears as a whole word, singular or plural ("Honda", "Hondas")."""
    return re.search(rf"\b{re.escape(name)}S?\b", text) is not None


def _find_model(text: str, models: dict[str, set[str]], make: str | None) -> tuple[str, str] | None:
    """The (make, model) the question names, preferring the longest model name."""
    candidates = [
        (model_make, model)
        for model_make, names in models.items()
        if make in (None, model_make)
        for model in names
    ]
    # Longest first, so "ACCORD HYBRID" wins over "ACCORD".
    for model_make, model in sorted(candidates, key=lambda c: (-len(c[1]), c[1])):
        pattern = _model_pattern(model)
        if make is None and model in COMMON_WORD_MODELS:
            pattern = YEAR_BEFORE + pattern
        if re.search(pattern, text):
            return model_make, model
    return None


@cache
def _model_pattern(model: str) -> str:
    """Match a model name however it's typed: "F-150" also matches "F150" and "F 150"."""
    chunks = re.findall(r"[A-Z0-9]+", model)
    return r"\b" + r"[\s-]?".join(re.escape(chunk) for chunk in chunks) + r"\b"
