"""Turn NHTSA's raw complaint and recall files into documents ready to load.

The files are tab-separated, have no header row, and are encoded as latin-1. Columns are
read by position, using NHTSA's field lists:
https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt and https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt
"""

import csv
import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

MAKES = frozenset({"FORD", "HONDA", "TOYOTA", "HYUNDAI", "KIA"})
FIRST_YEAR, LAST_YEAR = 2018, 2023

COMPLAINT_COLUMN_COUNT = 51
RECALL_COLUMN_COUNT = 29

# Complaint columns. Private fields (VIN, city, state, dealer, operator) are never read.
C_ODINO, C_MAKE, C_MODEL, C_YEAR = 1, 3, 4, 5
C_CRASH, C_FIRE, C_INJURED, C_DEATHS = 6, 8, 9, 10
C_COMPONENT, C_RECEIVED, C_DESCRIPTION, C_PRODUCT_TYPE = 11, 16, 19, 45

# Recall columns.
R_CAMPAIGN, R_MAKE, R_MODEL, R_YEAR = 1, 2, 3, 4
R_COMPONENT, R_MANUFACTURER, R_RECALL_TYPE, R_RECEIVED = 6, 7, 10, 15
R_DEFECT, R_CONSEQUENCE, R_REMEDY = 19, 20, 21


@dataclass(frozen=True)
class Vehicle:
    make: str
    model: str
    model_year: int
    component: str


@dataclass
class ParsedDocument:
    id: str
    source: Literal["complaint", "recall"]
    title: str
    body: str
    search_text: str
    received_on: date | None
    crash: bool | None
    fire: bool | None
    injured: int | None
    deaths: int | None
    vehicles: list[Vehicle] = field(default_factory=list)


def parse_complaints(path: Path) -> Iterator[ParsedDocument]:
    """One document per complaint (ODINO). NHTSA repeats a complaint once per component."""
    docs: dict[str, ParsedDocument] = {}
    for row in _rows(path, COMPLAINT_COLUMN_COUNT):
        if row[C_PRODUCT_TYPE] != "V":
            continue
        vehicle = _vehicle(row[C_MAKE], row[C_MODEL], row[C_YEAR], row[C_COMPONENT])
        if vehicle is None:
            continue

        odino = row[C_ODINO].strip()
        doc = docs.get(odino)
        if doc is None:
            body = row[C_DESCRIPTION].strip()
            doc = docs[odino] = ParsedDocument(
                id=f"complaint:{odino}",
                source="complaint",
                title="",
                body=body,
                search_text="",
                received_on=_date(row[C_RECEIVED]),
                crash=row[C_CRASH] == "Y",
                fire=row[C_FIRE] == "Y",
                injured=_int(row[C_INJURED]),
                deaths=_int(row[C_DEATHS]),
            )
        if vehicle not in doc.vehicles:
            doc.vehicles.append(vehicle)

    for doc in docs.values():
        doc.title = f"Owner complaint about the {_describe(doc.vehicles)}"
        doc.search_text = f"{_vehicle_text(doc.vehicles)}\n{doc.body}"
        yield doc


def parse_recalls(path: Path) -> Iterator[ParsedDocument]:
    """One document per recall campaign (CAMPNO). NHTSA repeats it once per vehicle."""
    docs: dict[str, ParsedDocument] = {}
    for row in _rows(path, RECALL_COLUMN_COUNT):
        if row[R_RECALL_TYPE] != "V":
            continue
        vehicle = _vehicle(row[R_MAKE], row[R_MODEL], row[R_YEAR], row[R_COMPONENT])
        if vehicle is None:
            continue

        campaign = row[R_CAMPAIGN].strip()
        doc = docs.get(campaign)
        if doc is None:
            body = (
                f"Defect: {row[R_DEFECT].strip()}\n\n"
                f"Consequence: {row[R_CONSEQUENCE].strip()}\n\n"
                f"Remedy: {row[R_REMEDY].strip()}"
            )
            doc = docs[campaign] = ParsedDocument(
                id=f"recall:{campaign}",
                source="recall",
                title=f"Recall {campaign} by {row[R_MANUFACTURER].strip()}",
                body=body,
                search_text="",
                received_on=_date(row[R_RECEIVED]),
                crash=None,
                fire=None,
                injured=None,
                deaths=None,
            )
        if vehicle not in doc.vehicles:
            doc.vehicles.append(vehicle)

    for campaign, doc in docs.items():
        # People write recall numbers both ways, e.g. 18V777000 and 18V777.
        short = campaign[:-3] if len(campaign) == 9 and campaign.endswith("000") else campaign
        doc.search_text = f"{campaign} {short}\n{_vehicle_text(doc.vehicles)}\n{doc.body}"
        yield doc


def _rows(path: Path, column_count: int) -> Iterator[list[str]]:
    """Yield rows with the expected number of columns from a .txt file or a .zip holding one."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as raw:
            yield from _read(io.TextIOWrapper(raw, encoding="latin-1", newline=""), column_count)
    else:
        with path.open(encoding="latin-1", newline="") as text:
            yield from _read(text, column_count)


def _read(text: io.TextIOBase, column_count: int) -> Iterator[list[str]]:
    for row in csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) == column_count:
            yield row


def _vehicle(make: str, model: str, year: str, component: str) -> Vehicle | None:
    """Return the vehicle if it's in scope, otherwise None."""
    make = make.strip().upper()
    year = year.strip()
    if make not in MAKES or not year.isdigit() or not FIRST_YEAR <= int(year) <= LAST_YEAR:
        return None
    return Vehicle(make, model.strip().upper(), int(year), component.strip())


def _date(value: str) -> date | None:
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").date()
    except ValueError:
        return None


def _int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def _describe(vehicles: list[Vehicle]) -> str:
    """'2019 HONDA ACCORD' for one vehicle; '2019 HONDA ACCORD and 1 other' for more."""
    first = vehicles[0]
    names = {(v.model_year, v.make, v.model) for v in vehicles}
    name = f"{first.model_year} {first.make} {first.model}"
    others = len(names) - 1
    return name if others == 0 else f"{name} and {others} other{'s' if others > 1 else ''}"


def _vehicle_text(vehicles: list[Vehicle]) -> str:
    """Every vehicle and component as one line, so keyword search can match them."""
    return "; ".join(f"{v.model_year} {v.make} {v.model} {v.component}" for v in vehicles)
