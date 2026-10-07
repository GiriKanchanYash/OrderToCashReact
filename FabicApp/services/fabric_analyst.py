"""Fabric counterpart of Snowflake Cortex Analyst for the O2C Copilot.

Snowflake: Cortex Analyst reads a semantic model (order_lens_semantic_model.yaml
on a stage) and returns {"message": {"content": [{"type": "text"}, {"type": "sql"}]}}.

Fabric: Azure OpenAI generates one read-only T-SQL statement grounded in
  1. FabicApp/Schema_model.yml (your semantic model, if you populate it), and
  2. the live column catalog of the BUSINESS_MART views (INFORMATION_SCHEMA),
and the response is returned in the *same* shape so the copied Copilot logic
in ai_service.py runs unchanged.

SECURITY: model output is untrusted. validate_sql() is the security boundary:
single SELECT/WITH statement, no DML/DDL/EXEC, only BUSINESS_MART objects.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from FabicApp import db
from FabicApp.services import azure_llm

log = logging.getLogger(__name__)

MAX_ROWS = 500
_SCHEMA_TTL = 3600
_schema_lock = threading.Lock()
_schema_cache: tuple[float, str] | None = None
_SEMANTIC_MODEL_PATH = Path(__file__).resolve().parent.parent / "Schema_model.yml"

_BLOCKED = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke|deny|into|"
    r"openrowset|openquery|opendatasource|bulk|shutdown|dbcc|backup|restore|xp_\w+|sp_\w+)\b",
    re.IGNORECASE,
)
_TABLE_REF = re.compile(
    r"\b(?:from|join)\s+((?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+))*)", re.IGNORECASE,
)
_CTE_NAME = re.compile(r"(?:with|,)\s+(\w+)\s+as\s*\(", re.IGNORECASE)


class AnalystError(RuntimeError):
    pass


def _schema_context() -> str:
    global _schema_cache
    with _schema_lock:
        if _schema_cache and time.time() - _schema_cache[0] < _SCHEMA_TTL:
            return _schema_cache[1]
    parts: list[str] = []
    try:
        model = _SEMANTIC_MODEL_PATH.read_text(encoding="utf-8").strip()
        if model:
            parts.append("SEMANTIC MODEL:\n" + model[:20000])
    except OSError:
        pass
    try:
        rows = db.run_tsql(
            """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'BUSINESS_MART'
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
        )
        by_table: dict[str, list[str]] = {}
        for r in rows:
            by_table.setdefault(str(r["table_name"]), []).append(f"{r['column_name']} {r['data_type']}")
        if by_table:
            lines = [f"{db.BM}.{t} ({', '.join(cols)})" for t, cols in by_table.items()]
            parts.append("AVAILABLE VIEWS (schema-qualify exactly as shown):\n" + "\n".join(lines))
    except Exception:
        log.warning("Could not read BUSINESS_MART catalog from Fabric", exc_info=True)
    text = "\n\n".join(parts)
    if not text:
        raise AnalystError("No schema context available (Schema_model.yml empty and catalog unreadable).")
    with _schema_lock:
        _schema_cache = (time.time(), text)
    return text


def validate_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
    if not s:
        raise AnalystError("Empty SQL.")
    if ";" in s or "--" in s or "/*" in s:
        raise AnalystError("Only a single statement without comments is allowed.")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise AnalystError("Only SELECT statements are allowed.")
    if _BLOCKED.search(s):
        raise AnalystError("Generated SQL contains a disallowed keyword.")
    ctes = {m.group(1).lower() for m in _CTE_NAME.finditer(s)}
    refs = [m.group(1) for m in _TABLE_REF.finditer(s)]
    if not refs:
        raise AnalystError("No FROM target in generated SQL.")
    for ref in refs:
        parts = [p.strip().strip("[]").upper() for p in ref.split(".")]
        if len(parts) == 1 and parts[0].lower() in ctes:
            continue
        if len(parts) < 2 or parts[-2] != "BUSINESS_MART":
            raise AnalystError(f"Generated SQL references a non-BUSINESS_MART object: {ref}")
    return s


def call_analyst(instruction: str, question: str) -> dict:
    """Return a Cortex-Analyst-shaped response: {"message": {"content": [...]}} or {"error": ...}."""
    try:
        schema = _schema_context()
        prompt = (
            "You translate Order-to-Cash (O2C / accounts receivable) questions into ONE Microsoft Fabric "
            "Warehouse T-SQL SELECT statement.\n\n"
            f"{schema}\n\n"
            "Rules:\n"
            "- Use only the views listed above, fully qualified exactly as shown.\n"
            "- T-SQL only: TOP (n) not LIMIT, IIF/CASE not IFF, CAST(GETDATE() AS DATE) for today, "
            "DATEADD/DATEDIFF/DATETRUNC for dates. Never GROUP BY ordinal or alias.\n"
            f"- Return at most {MAX_ROWS} rows. Prefer readable names (CUSTOMER_NAME) next to IDs.\n"
            "- Read-only: never INSERT/UPDATE/DELETE/MERGE/DDL/EXEC.\n"
            "- In \"text\", briefly restate how you interpreted the question. Do NOT invent figures; "
            "numbers come from executing the SQL.\n"
            'Respond as JSON: {"text": "...", "sql": "..."}. If it cannot be answered from these views, '
            'return {"text": "<why>", "sql": ""}.\n\n'
            f"{instruction}{question}"
        )
        raw = azure_llm.complete(prompt, json_mode=True, max_tokens=1500, temperature=0.0)
        data = json.loads(raw)
    except Exception as exc:
        return {"error": f"Fabric analyst failed: {exc}"}

    content: list[dict] = []
    text = str(data.get("text") or "").strip()
    if text:
        content.append({"type": "text", "text": text})
    sql = str(data.get("sql") or "").strip()
    if sql:
        try:
            content.append({"type": "sql", "statement": validate_sql(sql)})
        except AnalystError as exc:
            log.warning("Rejected analyst SQL: %s | %s", exc, sql[:300])
            return {"error": str(exc)}
    return {"message": {"role": "analyst", "content": content}}


def run_analyst_sql(conn, sql: str) -> list[dict]:
    """Execute validated analyst T-SQL with a row cap (no dialect translation)."""
    cur = conn.cursor()
    try:
        cur.execute(validate_sql(sql))
        if cur.description is None:
            return []
        cols = [c[0] for c in cur.description]
        return [{str(k).lower(): v for k, v in zip(cols, r)} for r in cur.fetchmany(MAX_ROWS)]
    finally:
        cur.close()
