"""Download NHTSA's files and load them into the database: python -m vsrag.ingest"""

import time
from pathlib import Path

from vsrag.db import get_conn, init_schema
from vsrag.ingest.download import COMPLAINTS_URL, RECALLS_URL, download
from vsrag.ingest.load import load_documents
from vsrag.ingest.parse import parse_complaints, parse_recalls

DATA_DIR = Path("data/raw")


def main() -> None:
    start = time.perf_counter()
    print("Downloading (skipped for files already in data/raw)...")
    complaints_file = download(COMPLAINTS_URL, DATA_DIR)
    recalls_file = download(RECALLS_URL, DATA_DIR)

    with get_conn() as conn:
        init_schema(conn)
        print("Loading complaints...")
        complaints = load_documents(conn, parse_complaints(complaints_file))
        print("Loading recalls...")
        recalls = load_documents(conn, parse_recalls(recalls_file))
        conn.commit()

    seconds = time.perf_counter() - start
    print(f"Loaded {complaints:,} complaints and {recalls:,} recalls in {seconds:.0f} s.")


if __name__ == "__main__":
    main()
