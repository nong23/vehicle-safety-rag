"""Builds fake NHTSA rows for tests, so no test depends on the real 300 MB files.

Column positions come from NHTSA's field lists (CMPL.txt and RCL.txt), counted from 0.
"""

from pathlib import Path

COMPLAINT_COLUMNS = {
    "CMPLID": 0,
    "ODINO": 1,
    "MFR_NAME": 2,
    "MAKETXT": 3,
    "MODELTXT": 4,
    "YEARTXT": 5,
    "CRASH": 6,
    "FIRE": 8,
    "INJURED": 9,
    "DEATHS": 10,
    "COMPDESC": 11,
    "CITY": 12,
    "STATE": 13,
    "VIN": 14,
    "LDATE": 16,
    "CDESCR": 19,
    "DEALER_NAME": 40,
    "DEALER_TEL": 41,
    "PROD_TYPE": 45,
    "VEHICLE_OPERATOR": 50,
}

RECALL_COLUMNS = {
    "RECORD_ID": 0,
    "CAMPNO": 1,
    "MAKETXT": 2,
    "MODELTXT": 3,
    "YEARTXT": 4,
    "COMPNAME": 6,
    "MFGNAME": 7,
    "RCLTYPECD": 10,
    "RCDATE": 15,
    "DESC_DEFECT": 19,
    "CONEQUENCE_DEFECT": 20,
    "CORRECTIVE_ACTION": 21,
}

COMPLAINT_DEFAULTS = {
    "ODINO": "11000001",
    "MAKETXT": "HONDA",
    "MODELTXT": "ACCORD",
    "YEARTXT": "2019",
    "CRASH": "N",
    "FIRE": "N",
    "INJURED": "0",
    "DEATHS": "0",
    "COMPDESC": "ENGINE",
    "LDATE": "20210315",
    "CDESCR": "ENGINE STALLED ON THE HIGHWAY.",
    "PROD_TYPE": "V",
}

RECALL_DEFAULTS = {
    "CAMPNO": "18V777000",
    "MAKETXT": "HONDA",
    "MODELTXT": "ODYSSEY",
    "YEARTXT": "2019",
    "COMPNAME": "SERVICE BRAKES, HYDRAULIC",
    "MFGNAME": "Honda (American Honda Motor Co.)",
    "RCLTYPECD": "V",
    "RCDATE": "20181107",
    "DESC_DEFECT": "The rear brake calipers may leak.",
    "CONEQUENCE_DEFECT": "Reduced braking increases the risk of a crash.",
    "CORRECTIVE_ACTION": "Dealers will bleed the brakes, free of charge.",
}


def complaint_row(**fields: str) -> list[str]:
    row = [""] * 51
    for name, value in {**COMPLAINT_DEFAULTS, **fields}.items():
        row[COMPLAINT_COLUMNS[name]] = value
    return row


def recall_row(**fields: str) -> list[str]:
    row = [""] * 29
    for name, value in {**RECALL_DEFAULTS, **fields}.items():
        row[RECALL_COLUMNS[name]] = value
    return row


def write_rows(path: Path, rows: list[list[str]]) -> Path:
    """Write rows the way NHTSA does: tab-separated, no header, latin-1."""
    path.write_text("".join("\t".join(row) + "\r\n" for row in rows), encoding="latin-1")
    return path
