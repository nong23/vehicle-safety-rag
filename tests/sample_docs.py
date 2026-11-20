"""Four small sample documents, shared by the search and pipeline tests."""

from datetime import date

from vsrag.ingest.parse import ParsedDocument, Vehicle

VARIANT = "bge_prefixed"

STALL = "complaint:1"  # 2019 Honda Accord
PAINT = "complaint:2"  # 2020 Honda Civic
BRAKES = "complaint:3"  # 2019 Honda Accord, listed under two parts
RECALL = "recall:18V777000"  # 2018 Ford F-150


def complaint(doc_id: str, body: str, vehicles: list[Vehicle]) -> ParsedDocument:
    names = " ".join(f"{v.model_year} {v.make} {v.model} {v.component}" for v in vehicles)
    return ParsedDocument(
        id=doc_id,
        source="complaint",
        title="Owner complaint",
        body=body,
        search_text=f"{names}\n{body}",
        received_on=date(2021, 3, 15),
        crash=False,
        fire=False,
        injured=0,
        deaths=0,
        vehicles=vehicles,
    )


def recall(doc_id: str, body: str, vehicles: list[Vehicle]) -> ParsedDocument:
    campaign = doc_id.removeprefix("recall:")
    return ParsedDocument(
        id=doc_id,
        source="recall",
        title=f"Recall {campaign} by Ford Motor Company",
        body=body,
        search_text=f"{campaign} {campaign[:6]}\n{body}",
        received_on=date(2018, 12, 1),
        crash=None,
        fire=None,
        injured=None,
        deaths=None,
        vehicles=vehicles,
    )


ACCORD_2019 = ("HONDA", "ACCORD", 2019)

DOCUMENTS = [
    complaint(
        STALL,
        "THE VEHICLE ENGINE STALLED AT HIGHWAY SPEED.",
        [Vehicle(*ACCORD_2019, "ENGINE")],
    ),
    complaint(
        PAINT,
        "PAINT IS PEELING OFF THE VEHICLE HOOD.",
        [Vehicle("HONDA", "CIVIC", 2020, "STRUCTURE")],
    ),
    complaint(
        BRAKES,
        "THE VEHICLE BRAKES FAILED WHEN STOPPING.",
        [Vehicle(*ACCORD_2019, "SERVICE BRAKES"), Vehicle(*ACCORD_2019, "ELECTRICAL SYSTEM")],
    ),
    recall(
        RECALL,
        "Defect: The front seat belt in the vehicle may not latch.\n\n"
        "Consequence: An unlatched seat belt increases the risk of injury in a crash.\n\n"
        "Remedy: Dealers will replace the seat belt buckle.",
        [Vehicle("FORD", "F-150", 2018, "SEAT BELTS")],
    ),
]
