from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from tests.sample_docs import DOCUMENTS, VARIANT
from vsrag.config import settings
from vsrag.db import init_schema
from vsrag.embed import embed_missing
from vsrag.ingest.load import load_documents


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Create the test database once per test run, if it doesn't exist yet."""
    url = settings.test_database_url
    name = str(conninfo_to_dict(url)["dbname"])
    if not name.endswith("_test"):
        raise RuntimeError(f"Refusing to run tests against {name!r}: name must end in _test")

    admin_url = make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin_url, autocommit=True) as admin:
        exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return url


@pytest.fixture
def conn(test_db_url: str) -> Iterator[psycopg.Connection]:
    """A connection whose changes are rolled back after each test."""
    with psycopg.connect(test_db_url) as conn:
        yield conn
        conn.rollback()


@pytest.fixture
def docs(conn: psycopg.Connection) -> psycopg.Connection:
    """The test database with the 4 documents in tests/sample_docs.py, embedded with VARIANT."""
    init_schema(conn)
    load_documents(conn, DOCUMENTS)
    embed_missing(conn, VARIANT)
    return conn
