"""
Direct-to-OneLake writer for the three app-state tables that turned out to
be Lakehouse Delta tables exposed through Fabric's SQL Analytics Endpoint
(saved_insights, genie_question_history, agent_review_log) -- confirmed via
a Spark notebook test (spark.table(...).write.format("delta").mode("append")
.saveAsTable(...) succeeds; the same INSERT over the SQL endpoint fails with
SQLSTATE 42000 / error 24559, "DML statements are not supported for this
table type").

This uses the `deltalake` package (delta-rs bindings) to append/delete rows
by writing directly to the underlying Delta Lake files in OneLake -- no
Spark session needed, works from a plain Python process. Reads still go
through the existing SQL endpoint (run_query in db.py), which already works
fine for SELECT; only writes are routed through here.

REQUIRES, beyond what db.py already needs:
- Three .env entries with the full OneLake ABFS path per table:
    ONELAKE_SAVED_INSIGHTS_PATH
    ONELAKE_GENIE_QUESTION_HISTORY_PATH
    ONELAKE_AGENT_REVIEW_LOG_PATH
  Get these from the Fabric UI: open the Lakehouse, right-click the table
  under Tables, "Copy ABFS path". They look like:
    abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/<table>
  or, for a schema-enabled Lakehouse, with an extra segment:
    abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/<schema>/<table>

- The service principal (AZURE_CLIENT_ID/SECRET/TENANT_ID, already in .env)
  needs an actual Fabric WORKSPACE role (Contributor or Member) granted in
  the workspace's "Manage access" page. This is separate from -- and not
  covered by -- whatever Warehouse-level SQL permission it already has for
  the SQL Analytics Endpoint; OneLake writes go through the workspace's
  storage permission model, not the Warehouse's.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger("orderlens.onelake_writer")

_lock = threading.Lock()


class OneLakeWriteError(RuntimeError):
    """Raised when a direct OneLake write/delete fails, or required .env
    configuration (path or credentials) is missing."""


def _storage_options() -> dict[str, str]:
    tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
    client_id = os.getenv("AZURE_CLIENT_ID", "").strip()
    client_secret = os.getenv("AZURE_CLIENT_SECRET", "").strip()
    missing = [n for n, v in (("AZURE_TENANT_ID", tenant_id), ("AZURE_CLIENT_ID", client_id),
                               ("AZURE_CLIENT_SECRET", client_secret)) if not v]
    if missing:
        raise OneLakeWriteError(f"Missing .env values for OneLake auth: {', '.join(missing)}")
    return {
        "azure_tenant_id": tenant_id,
        "azure_client_id": client_id,
        "azure_client_secret": client_secret,
        # Tells object_store (delta-rs's storage layer) to use OneLake's DFS
        # endpoint semantics rather than plain ADLS Gen2 -- required for
        # abfss://...@onelake.dfs.fabric.microsoft.com paths to authenticate
        # correctly against a Fabric workspace rather than a storage account.
        "use_fabric_endpoint": "true",
    }


def _table_path(env_var: str) -> str:
    path = os.getenv(env_var, "").strip()
    if not path:
        raise OneLakeWriteError(
            f"{env_var} is not set in .env. This must be the full OneLake ABFS path "
            "for this table (Fabric UI: Lakehouse -> Tables -> right-click table -> "
            "Copy ABFS path)."
        )
    return path


def append_rows(env_var: str, rows: list[dict]) -> None:
    """Appends rows (list of plain dicts, one per row) to the Delta table at
    the ABFS path configured in env_var. Column names must exactly match the
    table's real schema (case-sensitive, e.g. INSIGHT_ID not insight_id)."""
    if not rows:
        return
    path = _table_path(env_var)
    try:
        import pyarrow as pa
        table = pa.Table.from_pylist(rows)
        with _lock:
            write_deltalake(path, table, mode="append", storage_options=_storage_options())
    except OneLakeWriteError:
        raise
    except Exception as exc:
        logger.error("OneLake append failed for %s (%s): %s", env_var, path, exc)
        raise OneLakeWriteError(f"OneLake write to {env_var} failed: {exc}") from exc


def delete_rows(env_var: str, predicate: str) -> None:
    """Deletes rows matching a SQL-style predicate, e.g. 'INSIGHT_ID = 5'.
    predicate is inlined directly (delta-rs delete() takes a raw SQL
    expression, not parameterized) -- only call this with predicates built
    from values you've already validated as the expected type (see
    delete_insight in ai_service.py, which casts to int first)."""
    path = _table_path(env_var)
    try:
        with _lock:
            dt = DeltaTable(path, storage_options=_storage_options())
            dt.delete(predicate)
    except OneLakeWriteError:
        raise
    except Exception as exc:
        logger.error("OneLake delete failed for %s (%s) predicate=%r: %s", env_var, path, predicate, exc)
        raise OneLakeWriteError(f"OneLake delete on {env_var} failed: {exc}") from exc


def next_int_id(current_max: int | None) -> int:
    """These Delta tables have no identity/auto-increment column (confirmed
    via the notebook schema dump - plain nullable IntegerType), so the app
    must compute the next id itself. current_max should come from a
    `SELECT MAX(col) FROM ...` over the existing (working) read-only SQL
    endpoint. Best-effort only: under concurrent writes from multiple app
    instances this can race and collide, same as any client-side ID
    generation without a DB-side sequence/identity -- acceptable for this
    tool's traffic level, but worth knowing about."""
    return (current_max or 0) + 1


def utcnow() -> datetime:
    """Timezone-AWARE UTC timestamp - deliberately not stripped to naive.
    A naive datetime makes PyArrow infer a timezone-less timestamp column,
    which delta-rs maps to Delta's newer "timestamp_ntz" logical type. These
    tables were created via Spark's plain TimestampType (timezone-aware,
    implied UTC), which don't have the TimestampWithoutTimezone table
    feature enabled - writing a naive/ntz column against them fails with
    "Table features must be specified, please specify: TimestampWithoutTimezone".
    Keeping tzinfo=UTC makes PyArrow produce a tz-aware timestamp column
    that matches the table's actual type instead."""
    return datetime.now(timezone.utc)