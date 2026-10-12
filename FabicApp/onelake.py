"""
Direct writes to the OrderToCash_LK Lakehouse over OneLake.

The Lakehouse SQL analytics endpoint is read-only (INSERT/UPDATE/MERGE/DELETE fail
with error 24559), so the Fabric services keep READING Lakehouse tables through the
SQL endpoint (FabicApp.db) and WRITE them here, directly to the Delta tables.

Uses the same service principal as the SQL endpoint (AZURE_TENANT_ID / AZURE_CLIENT_ID /
AZURE_CLIENT_SECRET); it needs at least the Contributor role on the Fabric workspace.

Configuration (.env):
  FABRIC_ONELAKE_TABLES_URI = abfss://<workspace-id>@onelake.dfs.fabric.microsoft.com/<lakehouse-id>/Tables
  (a local folder path also works, which is useful for testing)

Tables are addressed as "<schema>.<table>", e.g. "raw_vault.dispute_case".
Column names are matched case-insensitively; values are converted to the table's
actual Delta column types (read from the table itself at write time).

Note: the SQL endpoint picks up new Delta commits within seconds, not instantly, so a
read issued immediately after a write may briefly return the previous state.
"""
from __future__ import annotations

import datetime as dt
import decimal
import logging
import os
import threading
import time
import uuid

try:
    import pyarrow as pa
    from deltalake import DeltaTable, QueryBuilder, write_deltalake
    _IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # keep Fabric reads working if the write packages are missing
    pa = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

import FabicApp.db  # noqa: F401  -- loads the project .env before the settings below are read

log = logging.getLogger("o2c.fabric.onelake")

TABLES_URI = os.getenv("FABRIC_ONELAKE_TABLES_URI", "").strip().strip('"').rstrip("/")
# Console logging of Lakehouse writes: "values" (default) logs every column value written,
# "summary" logs only table / operation / row counts, "off" disables write logging.
LOG_MODE = os.getenv("FABRIC_ONELAKE_LOG", "values").strip().strip('"').lower()
LAKEHOUSE_NAME = os.getenv("FABRIC_LAKEHOUSE_DATABASE", "").strip().strip('"') or "Lakehouse"

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class OneLakeWriteError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers

def _storage_options() -> dict[str, str]:
    if not TABLES_URI.startswith("abfss://"):
        return {}  # local path (tests)
    opts = {
        "azure_tenant_id": os.getenv("AZURE_TENANT_ID", "").strip().strip('"'),
        "azure_client_id": os.getenv("AZURE_CLIENT_ID", "").strip().strip('"'),
        "azure_client_secret": os.getenv("AZURE_CLIENT_SECRET", "").strip().strip('"'),
        "use_fabric_endpoint": "true",
    }
    missing = [k for k, v in opts.items() if not v]
    if missing:
        raise OneLakeWriteError(f"OneLake writes need {', '.join(missing)} in .env")
    return opts


def _uri(table: str) -> str:
    if not TABLES_URI:
        raise OneLakeWriteError(
            "FABRIC_ONELAKE_TABLES_URI is not set in .env "
            "(abfss://<workspace-id>@onelake.dfs.fabric.microsoft.com/<lakehouse-id>/Tables)"
        )
    schema, _, name = table.partition(".")
    if not schema or not name:
        raise OneLakeWriteError(f"table must be '<schema>.<table>', got {table!r}")
    return f"{TABLES_URI}/{schema}/{name}"


def _lock(table: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(table, threading.Lock())


def _open(table: str):
    if _IMPORT_ERROR is not None:
        raise OneLakeWriteError(
            f"Lakehouse writes need the deltalake and pyarrow packages "
            f"(pip install -r requirements.txt): {_IMPORT_ERROR}"
        )
    try:
        dtable = DeltaTable(_uri(table), storage_options=_storage_options())
    except OneLakeWriteError:
        raise
    except Exception as exc:
        log.error("OneLake cannot open %s.%s at %s: %s", os.getenv("FABRIC_LAKEHOUSE_DATABASE", "Lakehouse").strip().strip('"'),
                  table, _uri(table), exc)
        raise OneLakeWriteError(f"cannot open Lakehouse table {table}: {exc}") from exc
    return dtable, pa.schema(dtable.schema().to_arrow())


def _resolve(schema, table: str, names) -> dict[str, str]:
    """Map caller column names to the table's real names (case-insensitive)."""
    by_lower = {f.name.lower(): f.name for f in schema}
    out = {}
    for n in names:
        real = by_lower.get(n.lower())
        if real is None:
            raise OneLakeWriteError(f"column {n!r} does not exist in {table} (columns: {', '.join(schema.names)})")
        out[n] = real
    return out


class _Row(dict):
    """Row passed to update() callbacks; column lookups are case-insensitive."""

    def _key(self, k):
        if k in self:
            return k
        for real in self.keys():
            if real.lower() == str(k).lower():
                return real
        return k

    def __getitem__(self, k):
        return dict.__getitem__(self, self._key(k))

    def get(self, k, default=None):
        return dict.get(self, self._key(k), default)


def now_utc() -> dt.datetime:
    """Equivalent of GETDATE() on Fabric (UTC)."""
    return dt.datetime.now(dt.timezone.utc)


def today_utc() -> dt.date:
    """Equivalent of CAST(GETDATE() AS DATE) on Fabric (UTC)."""
    return now_utc().date()


def _coerce(value, typ):
    """Convert a Python value to the column's Arrow type."""
    if value is None or pa.types.is_null(typ):
        return None
    if pa.types.is_boolean(typ):
        if isinstance(value, str):
            return value.strip().upper() in ("1", "TRUE", "Y", "YES")
        return bool(value)
    if pa.types.is_integer(typ):
        return int(value)
    if pa.types.is_floating(typ):
        return float(value)
    if pa.types.is_decimal(typ):
        q = decimal.Decimal(1).scaleb(-typ.scale)
        return decimal.Decimal(str(value)).quantize(q, rounding=decimal.ROUND_HALF_UP)
    if pa.types.is_date(typ):
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        return dt.date.fromisoformat(str(value)[:10])
    if pa.types.is_timestamp(typ):
        if isinstance(value, dt.datetime):
            ts = value
        elif isinstance(value, dt.date):
            ts = dt.datetime(value.year, value.month, value.day)
        else:
            ts = dt.datetime.fromisoformat(str(value))
        if typ.tz:  # tz-aware column: treat naive values as UTC
            return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return str(value)
    return value


def _literal(value) -> str:
    """SQL literal for delete predicates (DataFusion SQL)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, decimal.Decimal)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- console logging

def _fmt(value) -> str:
    """Readable value for the console: strings quoted, NULL for None, ISO dates."""
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


def _block(pairs) -> str:
    pairs = list(pairs)
    if not pairs:
        return ""
    width = max(len(k) for k, _ in pairs)
    return "\n" + "\n".join(f"      {k.ljust(width)} = {v}" for k, v in pairs)


def _where_text(where: dict) -> str:
    return " AND ".join(f"{k} = {_fmt(v)}" for k, v in where.items())


def _version(dtable) -> str:
    try:
        dtable.update_incremental()
        return str(dtable.version())
    except Exception:
        return "?"


def _log_ok(op: str, table: str, detail: str, started: float, dtable, counts: str) -> None:
    if LOG_MODE == "off":
        return
    ms = (time.perf_counter() - started) * 1000
    body = detail if LOG_MODE == "values" else ""
    log.info("OneLake %s %s.%s%s\n      -> committed: %s | table version %s | %.0f ms",
             op, LAKEHOUSE_NAME, table, body, counts, _version(dtable), ms)


def _log_fail(op: str, table: str, detail: str, exc: Exception) -> None:
    log.error("OneLake %s %s.%s FAILED: %s%s", op, LAKEHOUSE_NAME, table, exc, detail if LOG_MODE != "off" else "")


# --------------------------------------------------------------------------- public API

def append(table: str, row: dict, *, auto_id: str | None = None, fill_now: tuple[str, ...] = ()) -> dict:
    """INSERT one row. Columns not given are NULL.

    auto_id:  column to fill with the next value (MAX + 1 for integer columns, a UUID for
              text columns) -- Delta tables have no IDENTITY / AUTOINCREMENT.
    fill_now: columns to set to the current UTC time when the table has them and the
              row does not set them (the defaults Snowflake applied, e.g. CREATED_AT).
    Returns the row as written. Logs every column value written to the console.
    """
    started = time.perf_counter()
    with _lock(table):
        dtable, schema = _open(table)
        names = _resolve(schema, table, row)
        values = {names[k]: v for k, v in row.items()}
        lower = {f.name.lower(): f.name for f in schema}
        generated = []
        for col in fill_now:
            real = lower.get(col.lower())
            if real and values.get(real) is None:
                values[real] = now_utc()
                generated.append(real)
        if auto_id:
            real = _resolve(schema, table, [auto_id])[auto_id]
            if values.get(real) is None:
                typ = schema.field(real).type
                if pa.types.is_integer(typ) or (pa.types.is_decimal(typ) and typ.scale == 0):
                    col = dtable.to_pyarrow_dataset().to_table(columns=[real]).column(real)
                    current = max((x for x in col.to_pylist() if x is not None), default=None)
                    values[real] = int(current or 0) + 1
                else:
                    values[real] = uuid.uuid4().hex
                generated.append(real)
        record = {f.name: _coerce(values.get(f.name), f.type) for f in schema}
        detail = _block(
            (name + (" (generated)" if name in generated else ""), _fmt(v)) for name, v in record.items()
        )
        try:
            write_deltalake(dtable, pa.Table.from_pylist([record], schema=schema), mode="append")
        except Exception as exc:
            _log_fail("INSERT", table, detail, exc)
            raise OneLakeWriteError(f"append to {table} failed: {exc}") from exc
        _log_ok("INSERT", table, detail, started, dtable, "1 row inserted")
        return record


def update(table: str, where: dict, changes) -> int:
    """UPDATE rows matching all `where` equalities.

    changes: dict of column -> new value, or a function(row) -> dict for values that
             depend on the current row (e.g. COALESCE(OWNER, ...), DATEDIFF from OPENED_DATE).
    Returns the number of rows updated (0 if none matched, like an UPDATE with no hits).
    Logs the WHERE condition and every changed column as old -> new.
    """
    started = time.perf_counter()
    with _lock(table):
        dtable, schema = _open(table)
        keys = _resolve(schema, table, where)
        # Look rows up with delta-rs' own SQL engine: data files rewritten by earlier
        # merges store text as string_view, which pyarrow dataset filters cannot compare.
        cond = " AND ".join(
            f'"{keys[k]}" = {_literal(_coerce(v, schema.field(keys[k]).type))}' for k, v in where.items()
        )
        current = pa.table(QueryBuilder().register("t", dtable).execute(f"SELECT * FROM t WHERE {cond}").read_all())
        if current.num_rows == 0:
            if LOG_MODE != "off":
                log.info("OneLake UPDATE %s.%s WHERE %s\n      -> no matching rows, nothing changed",
                         LAKEHOUSE_NAME, table, _where_text(where))
            return 0
        rows, seen, detail_lines = [], set(), []
        for r in current.to_pylist():
            key = tuple(r[keys[k]] for k in where)
            if key in seen:  # duplicates in the target are all updated by one source row
                continue
            seen.add(key)
            before = {f.name: _coerce(r.get(f.name), f.type) for f in schema}
            delta = changes(_Row(r)) if callable(changes) else changes
            names = _resolve(schema, table, delta)
            for k, v in delta.items():
                r[names[k]] = v
            after = {f.name: _coerce(r.get(f.name), f.type) for f in schema}
            rows.append(after)
            changed = [(c, f"{_fmt(before[c])} -> {_fmt(after[c])}") for c in after if before[c] != after[c]]
            unchanged = [names[k] for k in delta if before[names[k]] == after[names[k]]]
            if len(seen) > 1:
                detail_lines.append(("--- row", ", ".join(f"{keys[k]}={_fmt(r[keys[k]])}" for k in where)))
            detail_lines += changed or [("(no column values changed)", "")]
            if unchanged:
                detail_lines.append(("(already set, kept)", ", ".join(unchanged)))
        detail = f" WHERE {_where_text(where)} ({current.num_rows} row(s) matched)" + _block(detail_lines)
        source = pa.Table.from_pylist(rows, schema=schema)
        predicate = " AND ".join(f't."{keys[k]}" = s."{keys[k]}"' for k in where)
        try:
            (dtable.merge(source, predicate=predicate, source_alias="s", target_alias="t")
                   .when_matched_update_all()
                   .execute())
        except Exception as exc:
            _log_fail("UPDATE", table, detail, exc)
            raise OneLakeWriteError(f"update of {table} failed: {exc}") from exc
        _log_ok("UPDATE", table, detail, started, dtable, f"{current.num_rows} row(s) updated")
        return current.num_rows


def delete(table: str, where: dict) -> None:
    """DELETE rows matching all `where` equalities. Logs the condition and rows deleted."""
    started = time.perf_counter()
    with _lock(table):
        dtable, schema = _open(table)
        keys = _resolve(schema, table, where)
        predicate = " AND ".join(
            f'"{keys[k]}" = {_literal(_coerce(v, schema.field(keys[k]).type))}' for k, v in where.items()
        )
        detail = f" WHERE {_where_text(where)}"
        try:
            metrics = dtable.delete(predicate) or {}
        except Exception as exc:
            _log_fail("DELETE", table, detail, exc)
            raise OneLakeWriteError(f"delete from {table} failed: {exc}") from exc
        deleted = metrics.get("num_deleted_rows", "?")
        _log_ok("DELETE", table, detail, started, dtable, f"{deleted} row(s) deleted")