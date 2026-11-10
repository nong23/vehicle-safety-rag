import re
import zipfile
from datetime import date
from pathlib import Path

from tests.rows import complaint_row, recall_row, write_rows
from vsrag.ingest.parse import Vehicle, parse_complaints, parse_recalls


def test_rows_sharing_odino_become_one_document(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "c.txt",
        [
            complaint_row(COMPDESC="ENGINE"),
            complaint_row(COMPDESC="ELECTRICAL SYSTEM"),
            complaint_row(COMPDESC="POWER TRAIN"),
        ],
    )

    docs = list(parse_complaints(path))

    assert len(docs) == 1
    assert docs[0].id == "complaint:11000001"
    assert {v.component for v in docs[0].vehicles} == {
        "ENGINE",
        "ELECTRICAL SYSTEM",
        "POWER TRAIN",
    }


def test_out_of_scope_rows_are_dropped(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "c.txt",
        [
            complaint_row(ODINO="1", MAKETXT="MAZDA"),
            complaint_row(ODINO="2", YEARTXT="2017"),
            complaint_row(ODINO="3", YEARTXT="2024"),
            complaint_row(ODINO="4", YEARTXT="9999"),
            complaint_row(ODINO="5", PROD_TYPE="T"),
            complaint_row(ODINO="6"),
        ],
    )

    assert [doc.id for doc in parse_complaints(path)] == ["complaint:6"]


def test_complaint_fields_are_parsed(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "c.txt",
        [complaint_row(CRASH="Y", FIRE="N", INJURED="2", DEATHS="0", LDATE="20210315")],
    )

    doc = next(parse_complaints(path))

    assert doc.source == "complaint"
    assert doc.received_on == date(2021, 3, 15)
    assert (doc.crash, doc.fire, doc.injured, doc.deaths) == (True, False, 2, 0)
    assert doc.body == "ENGINE STALLED ON THE HIGHWAY."
    assert doc.vehicles == [Vehicle("HONDA", "ACCORD", 2019, "ENGINE")]
    assert "2019 HONDA ACCORD" in doc.search_text


def test_private_fields_are_not_kept(tmp_path: Path) -> None:
    private = {
        "VIN": "1HGCV1F3XKA",
        "CITY": "SPRINGFIELD",
        "STATE": "WY",
        "DEALER_NAME": "SMITH HONDA",
        "DEALER_TEL": "5415550100",
        "VEHICLE_OPERATOR": "JANE DOE",
    }
    path = write_rows(tmp_path / "c.txt", [complaint_row(**private)])

    stored = repr(list(parse_complaints(path)))

    for value in private.values():
        assert value not in stored


def test_rows_with_wrong_column_count_are_skipped(tmp_path: Path) -> None:
    good = complaint_row(ODINO="7")
    path = write_rows(tmp_path / "c.txt", [good[:20], good])

    assert [doc.id for doc in parse_complaints(path)] == ["complaint:7"]


def test_reads_the_zip_nhtsa_publishes(tmp_path: Path) -> None:
    txt = write_rows(tmp_path / "c.txt", [complaint_row()])
    archive = tmp_path / "c.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(txt, "COMPLAINTS.txt")

    assert [doc.id for doc in parse_complaints(archive)] == ["complaint:11000001"]


def test_recall_rows_for_one_campaign_become_one_document(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "r.txt",
        [
            recall_row(MODELTXT="ODYSSEY"),
            recall_row(MODELTXT="HR-V"),
            recall_row(MAKETXT="ACURA", MODELTXT="MDX"),
        ],
    )

    docs = list(parse_recalls(path))

    assert len(docs) == 1
    assert docs[0].id == "recall:18V777000"
    assert {v.model for v in docs[0].vehicles} == {"ODYSSEY", "HR-V"}


def test_recall_body_combines_defect_consequence_and_fix(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "r.txt", [recall_row()])

    doc = next(parse_recalls(path))

    assert doc.source == "recall"
    assert "The rear brake calipers may leak." in doc.body
    assert "Reduced braking increases the risk of a crash." in doc.body
    assert "Dealers will bleed the brakes, free of charge." in doc.body
    assert doc.received_on == date(2018, 11, 7)
    assert (doc.crash, doc.fire, doc.injured, doc.deaths) == (None, None, None, None)


def test_recall_search_text_contains_both_campaign_forms(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "r.txt", [recall_row(CAMPNO="18V777000")])

    doc = next(parse_recalls(path))

    assert re.search(r"\b18V777000\b", doc.search_text)
    assert re.search(r"\b18V777\b", doc.search_text)


def test_out_of_scope_recall_rows_are_dropped(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "r.txt",
        [
            recall_row(CAMPNO="20E001000", RCLTYPECD="E"),
            recall_row(CAMPNO="21V001000", YEARTXT="2016"),
            recall_row(CAMPNO="21V002000"),
        ],
    )

    assert [doc.id for doc in parse_recalls(path)] == ["recall:21V002000"]
