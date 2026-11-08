from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from vsrag.config import settings


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
