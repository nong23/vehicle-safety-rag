from importlib.resources import files

import psycopg

from vsrag.config import settings


def get_conn(url: str | None = None) -> psycopg.Connection:
    """Open a connection to the app database, or to `url` if one is given."""
    return psycopg.connect(url or settings.database_url)


def init_schema(conn: psycopg.Connection) -> None:
    """Create the extension, tables and indexes. Safe to run more than once."""
    schema = files("vsrag").joinpath("schema.sql").read_bytes()
    conn.execute(schema)
