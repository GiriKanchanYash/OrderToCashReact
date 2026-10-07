import os
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector

# Always load deploy/backend/.env regardless of uvicorn working directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

_CONN_PARAMS = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
    "user": os.getenv("SNOWFLAKE_USER", ""),
    "password": os.getenv("SNOWFLAKE_PASSWORD", ""),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "CORTEX_ANALYST_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "ORDER_LENS"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "INFORMATION_MART"),
    "role": os.getenv("SNOWFLAKE_ROLE", ""),
}

DB = _CONN_PARAMS["database"]
SCHEMA = _CONN_PARAMS["schema"]
BM = f"{DB}.BUSINESS_MART"


@contextmanager
def get_connection():
    conn = snowflake.connector.connect(**{k: v for k, v in _CONN_PARAMS.items() if v})
    try:
        yield conn
    finally:
        conn.close()


def _normalize_row(row: dict) -> dict:
    """Snowflake returns unquoted aliases uppercased; normalize to lowercase for the API."""
    return {str(k).lower(): v for k, v in row.items()}


def run_query_on_conn(conn, sql: str, params: tuple | None = None) -> list[dict]:
    cur = conn.cursor(snowflake.connector.DictCursor)
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    return [_normalize_row(dict(r)) for r in rows]


def run_query(sql: str, params: tuple | None = None) -> list[dict]:
    with get_connection() as conn:
        return run_query_on_conn(conn, sql, params)


def run_execute(sql: str, params: tuple | None = None) -> None:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())


def promote_staging_to_mart(domain: str) -> None:
    """After agent writes to SAP_STG, reload the vault slice and refresh mart tables."""
    domains = {
        "payment": ("SP_LOAD_PAYMENT_DOMAIN", ["PAYMENT_DT", "AR_OPEN_ITEM_DT"]),
        "collections": (
            "SP_LOAD_COLLECTIONS_DOMAIN",
            ["COLLECTION_ACTIVITY_DT", "PTP_DT", "DISPUTE_DT"],
        ),
    }
    proc, tables = domains[domain]
    run_execute(f"CALL {DB}.RAW_VAULT.{proc}()")
    for table in tables:
        try:
            run_execute(f"ALTER DYNAMIC TABLE {DB}.{SCHEMA}.{table} REFRESH")
        except Exception:
            pass
