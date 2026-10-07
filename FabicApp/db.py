"""
Fabric Warehouse connection layer for the Order-to-Cash app.

Exposes the same surface as the Snowflake reference layer (app/db.py):
DB, SCHEMA, BM, get_connection(), run_query(), run_query_on_conn(),
run_execute(), promote_staging_to_mart() -- so the Fabric services in
FabicApp/services mirror the Snowflake services call-for-call.

Queries written in the Snowflake dialect are translated to T-SQL by
FabicApp/sql_dialect.py before execution. Hand-written T-SQL uses
run_tsql()/run_tsql_on_conn(), which skip translation.

Connection settings (.env):
  FABRIC_WAREHOUSE_SERVER / FABRIC_WAREHOUSE_DATABASE   (O2C warehouse, preferred)
  FABRIC_SQL_SERVER / FABRIC_DATABASE                   (fallback)
  FABRIC_SCHEMA            default INFORMATION_MART (app tables: SAVED_INSIGHTS, ...)
  AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET (service principal)
"""
from __future__ import annotations

import logging
import os
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pyodbc
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv

from FabicApp.sql_dialect import to_tsql

logger = logging.getLogger("o2c.fabric.db")

# SQL_COPT_SS_ACCESS_TOKEN - the ODBC connection attribute msodbcsql uses to
# accept a raw Azure AD access token instead of a UID/PWD pair.
_SQL_COPT_SS_ACCESS_TOKEN = 1256
_FABRIC_TOKEN_SCOPE = "https://database.windows.net/.default"

# Fabric's SQL analytics endpoint is unreliable with in-connection-string
# "Authentication=ActiveDirectoryServicePrincipal" over ODBC Driver 18 -- it
# frequently hangs and fails with a generic HYT00 login timeout instead of a
# clean auth error. Acquiring the AAD token ourselves via azure-identity and
# passing it as SQL_COPT_SS_ACCESS_TOKEN is the path Microsoft recommends for
# Fabric and is what reliably works.
_token_lock = threading.Lock()
_cached_credential: ClientSecretCredential | None = None
_cached_token: str | None = None
_cached_token_expires_at: float = 0.0

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# pyodbc SQLSTATE prefixes we can give a specific, actionable message for.
# See: https://learn.microsoft.com/sql/odbc/reference/appendixes/appendix-a-odbc-error-codes
_SQLSTATE_HINTS: dict[str, str] = {
    "IM002": (
        "ODBC Driver 18 for SQL Server is not installed (or not registered) on this "
        "machine. Install it: "
        "https://learn.microsoft.com/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server"
    ),
    "IM003": (
        "The ODBC driver could not be loaded. Re-install 'ODBC Driver 18 for SQL Server' "
        "for this OS/architecture."
    ),
    "08001": (
        "Could not reach the Fabric SQL endpoint (network/firewall/DNS). Confirm "
        "FABRIC_SQL_SERVER is correct, the Fabric Warehouse/Lakehouse SQL endpoint is "
        "enabled, and this machine's IP/network is allowed (VPN or workspace firewall rules)."
    ),
    "HYT00": (
        "Connection to the Fabric SQL endpoint timed out. Usually a network/firewall "
        "issue rather than bad credentials."
    ),
    "28000": (
        "Fabric rejected the login. The service principal (AZURE_CLIENT_ID/"
        "AZURE_CLIENT_SECRET/AZURE_TENANT_ID) is invalid, the secret has expired, or "
        "the app registration has not been added as a member with access to this "
        "Warehouse/Lakehouse in the Fabric workspace."
    ),
    "42000": (
        "Fabric denied the query (permissions or invalid object name). Confirm the "
        "service principal has at least Read/ReadData on this Warehouse and that "
        "FABRIC_SCHEMA / table names match what actually exists."
    ),
}


class FabricConnectionError(RuntimeError):
    """Raised when we can't open or use a connection to the Fabric SQL endpoint,
    with a human-actionable hint attached (as opposed to a raw pyodbc traceback)."""


def _hint_for(exc: pyodbc.Error) -> str:
    sqlstate = ""
    if exc.args:
        sqlstate = str(exc.args[0])
    for prefix, hint in _SQLSTATE_HINTS.items():
        if sqlstate.startswith(prefix):
            return hint
    msg = str(exc).lower()
    if "login failed" in msg or "cannot open server" in msg:
        return _SQLSTATE_HINTS["28000"]
    if "denied on the requested resource" in msg or "external policy action" in msg:
        return _SQLSTATE_HINTS["42000"]
    return "Unrecognized Fabric/ODBC error - see the underlying message for details."


def _load_conn_params() -> dict[str, str]:
    load_dotenv(_ENV_PATH, override=True)
    server = os.getenv("FABRIC_WAREHOUSE_SERVER", "").strip() or os.getenv("FABRIC_SQL_SERVER", "").strip()
    database = os.getenv("FABRIC_WAREHOUSE_DATABASE", "").strip() or os.getenv("FABRIC_DATABASE", "").strip()
    return {
        "server": server.strip('"'),
        "database": database.strip('"'),
        "schema": os.getenv("FABRIC_SCHEMA", "").strip().strip('"'),
        "tenant_id": os.getenv("AZURE_TENANT_ID", "").strip(),
        "client_id": os.getenv("AZURE_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("AZURE_CLIENT_SECRET", "").strip(),
    }


_P = _load_conn_params()
# Same names as the Snowflake reference layer (app/db.py):
#   DB     -> Fabric warehouse (Snowflake: ORDER_LENS database)
#   SCHEMA -> app/info-mart schema (Snowflake: INFORMATION_MART)
#   BM     -> business-mart views (Snowflake: ORDER_LENS.BUSINESS_MART)
DATABASE = _P["database"] or "OrderToCash_DW"
DB = DATABASE
SCHEMA = _P["schema"] or "INFORMATION_MART"
BM = f"{DB}.business_mart"
IM = SCHEMA


def _build_conn_string(p: dict[str, str]) -> str:
    missing = [k for k in ("server", "database", "tenant_id", "client_id", "client_secret") if not p.get(k)]
    if missing:
        raise RuntimeError(f"Fabric connection details missing from .env: {', '.join(missing)}")
    # No UID/PWD/Authentication= here on purpose - auth is done via an AAD
    # access token passed through attrs_before (see _get_access_token_struct).
    return (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server=tcp:{p['server']},1433;"
        f"Database={p['database']};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )


def _get_access_token_struct(p: dict[str, str]) -> bytes:
    """Returns the SQL_COPT_SS_ACCESS_TOKEN struct pyodbc's attrs_before needs,
    fetching a fresh AAD token only when the cached one is near expiry."""
    global _cached_credential, _cached_token, _cached_token_expires_at
    with _token_lock:
        now = time.time()
        if _cached_token and now < _cached_token_expires_at - 60:
            token = _cached_token
        else:
            if _cached_credential is None:
                _cached_credential = ClientSecretCredential(
                    tenant_id=p["tenant_id"],
                    client_id=p["client_id"],
                    client_secret=p["client_secret"],
                )
            try:
                aad_token = _cached_credential.get_token(_FABRIC_TOKEN_SCOPE)
            except ClientAuthenticationError as exc:
                logger.error("Azure AD token acquisition failed: %s", exc)
                raise FabricConnectionError(
                    "Azure AD rejected the service principal credentials while requesting a "
                    "Fabric access token. Check AZURE_TENANT_ID/AZURE_CLIENT_ID/"
                    f"AZURE_CLIENT_SECRET. (raw: {exc})"
                ) from exc
            token = aad_token.token
            _cached_token = token
            _cached_token_expires_at = aad_token.expires_on

    token_bytes = token.encode("utf-16-le")
    return struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)


@contextmanager
def get_connection():
    p = _load_conn_params()
    token_struct = _get_access_token_struct(p)
    try:
        conn = pyodbc.connect(
            _build_conn_string(p),
            attrs_before={_SQL_COPT_SS_ACCESS_TOKEN: token_struct},
            timeout=30,
            autocommit=True,
        )
    except pyodbc.Error as exc:
        hint = _hint_for(exc)
        logger.error(
            "Fabric connection failed (server=%s db=%s): %s | hint: %s",
            p.get("server"), p.get("database"), exc, hint,
        )
        raise FabricConnectionError(f"{hint} (raw: {exc})") from exc
    try:
        yield conn
    except pyodbc.Error as exc:
        hint = _hint_for(exc)
        logger.error("Fabric query failed: %s | hint: %s", exc, hint)
        raise FabricConnectionError(f"{hint} (raw: {exc})") from exc
    finally:
        conn.close()


def _normalize_row(row: dict) -> dict:
    return {str(k).lower(): v for k, v in row.items()}


def _execute_on_conn(conn, sql: str, params: tuple | None = None) -> list[dict]:
    cur = conn.cursor()
    try:
        cur.execute(sql, params or ())
        # Skip row-count-only result sets (e.g. from EXEC) to reach the data.
        while cur.description is None:
            if not cur.nextset():
                return []
        cols = [c[0] for c in cur.description]
        return [_normalize_row(dict(zip(cols, r))) for r in cur.fetchall()]
    finally:
        cur.close()


def run_tsql_on_conn(conn, sql: str, params: tuple | None = None) -> list[dict]:
    """Execute native T-SQL (no dialect translation). '?' placeholders."""
    return _execute_on_conn(conn, sql, params)


def run_tsql(sql: str, params: tuple | None = None) -> list[dict]:
    with get_connection() as conn:
        return _execute_on_conn(conn, sql, params)


def run_query_on_conn(conn, sql: str, params: tuple | None = None) -> list[dict]:
    """Same contract as app.db.run_query_on_conn: Snowflake-dialect SQL with
    %s placeholders in, lower-cased dict rows out."""
    return _execute_on_conn(conn, to_tsql(sql), params)


def run_query(sql: str, params: tuple | None = None) -> list[dict]:
    with get_connection() as conn:
        return run_query_on_conn(conn, sql, params)


def run_execute(sql: str, params: tuple | None = None) -> None:
    """Fabric Warehouse supports T-SQL DML/EXEC directly (autocommit)."""
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(to_tsql(sql), params or ())
        finally:
            cur.close()


def promote_staging_to_mart(domain: str) -> None:
    """Fabric counterpart of app.db.promote_staging_to_mart.

    After an agent writes to SAP_STG, run the raw-vault loader for that slice.
    Snowflake then refreshes DYNAMIC TABLEs; Fabric has no dynamic tables, so
    the INFORMATION_MART/BUSINESS_MART objects are expected to be views over
    the vault (or refreshed by the loader procedure itself). A missing loader
    procedure is logged, not raised -- same tolerance as the Snowflake code.
    """
    domains = {
        "payment": "SP_LOAD_PAYMENT_DOMAIN",
        "collections": "SP_LOAD_COLLECTIONS_DOMAIN",
    }
    proc = domains[domain]
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(f"EXEC [{DB}].[RAW_VAULT].[{proc}]")
            finally:
                cur.close()
    except Exception as exc:
        logger.warning("Fabric promote_staging_to_mart(%s): %s not run: %s", domain, proc, exc)


def qview(name: str) -> str:
    """Schema-qualified table reference. Tables are flat (no BASE_LAYER joins
    needed) so this just prefixes the configured schema."""
    return f"{DATABASE}.{SCHEMA}.{name.strip().strip(chr(34)).lower()}"


def qcol(name: str, view: str | None = None) -> str:
    """Kept for call-site compatibility with the Snowflake original's
    qcol("col") / qcol("col", "some_view") calls. Fabric/T-SQL doesn't need
    Snowflake's quoted-vs-unquoted alias workaround, so this just normalizes
    to a plain lowercase column name. The `view` argument is accepted but
    unused."""
    return name.strip().strip('"').lower()