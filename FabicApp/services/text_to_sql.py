"""
Natural-language -> SQL -> execution pipeline for the Copilot.

Flow: generate_sql() asks the LLM for a single T-SQL SELECT statement
grounded in the whitelisted schema catalog (schema_catalog.py) -> validate_sql()
defensively checks it before anything touches the database -> execute_sql()
runs it with a bounded row cap and query timeout.

SECURITY MODEL: the LLM's output is untrusted input, same as if a user typed
it directly into a query box. validate_sql() is the actual security boundary,
not the prompt wording - never rely on "the prompt told it not to" alone.
- Single statement only (rejects any ';' before the final one).
- Must start with SELECT or WITH (CTE) after stripping whitespace/comments.
- Hard rejects a keyword blocklist covering all DDL/DML, SELECT..INTO,
  linked-server/OS-shell escapes, and system stored procedures.
- Every FROM/JOIN target must resolve (after stripping schema/db/bracket/quote
  qualifiers) to a table in schema_catalog's whitelist, OR to a CTE name
  defined earlier in the same statement.
- Must contain an explicit row cap (TOP n / OFFSET..FETCH). If the cap
  exceeds MAX_ROWS it's clamped down; if there's no cap at all, generation
  is rejected rather than guessing where to inject one into arbitrary CTE
  structures.
"""
from __future__ import annotations

import logging
import re

import pyodbc

from app import db
from app.db import FabricConnectionError, get_connection
from app.services import schema_catalog

logger = logging.getLogger("orderlens.text_to_sql")

MAX_ROWS = 500
DEFAULT_ROWS = 200
QUERY_TIMEOUT_SECONDS = 20

_BLOCKED_KEYWORDS = [
    "insert", "update", "delete", "merge", "drop", "alter", "create",
    "truncate", "exec", "execute", "grant", "revoke", "deny", "into",
    "openrowset", "openquery", "opendatasource", "bulk", "shutdown",
    "dbcc", "backup", "restore", "xp_cmdshell", "sp_executesql",
]
_BLOCKED_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _BLOCKED_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_SP_XP_RE = re.compile(r"\b(sp_|xp_)\w+", re.IGNORECASE)
_COMMENT_RE = re.compile(r"--|/\*|\*/")
_CTE_NAME_RE = re.compile(r"(?:with|,)\s+(\w+)\s+as\s*\(", re.IGNORECASE)
_TABLE_REF_RE = re.compile(
    r"\b(?:from|join)\s+((?:\[[^\]]+\]|\"[^\"]+\"|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\"[^\"]+\"|\w+))*)",
    re.IGNORECASE,
)
_TOP_RE = re.compile(r"\btop\s*\(?\s*(\d+)\s*\)?", re.IGNORECASE)
_FETCH_RE = re.compile(r"\bfetch\s+next\s+(\d+)\s+rows\s+only\b", re.IGNORECASE)
_MAX_SQL_CHARS = 6000


class SqlGenerationError(RuntimeError):
    """The LLM failed to produce usable SQL (couldn't call the model, or
    empty/unparseable response)."""


class SqlValidationError(RuntimeError):
    """Generated SQL failed a safety check and was never sent to Fabric."""


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"```\s*$", "", t)
    return t.strip()


def _last_identifier(qualified: str) -> str:
    """'React_OrderLens.orders_header_summary' / '[x].[y]' / '"x"."y"' -> 'y' (lowercase)."""
    parts = [p.strip() for p in qualified.split(".")]
    last = parts[-1]
    last = last.strip().strip("[]").strip('"').strip()
    return last.lower()


def generate_sql(question: str, llm_complete) -> str:
    """llm_complete is injected (the caller's _llm_complete) so this module
    has no dependency on ai_service.py, avoiding a circular import."""
    schema_text = schema_catalog.schema_prompt_text(db.SCHEMA)
    prompt = f"""You are a T-SQL (Microsoft Fabric Warehouse) query generator.

{schema_text}

Rules (all mandatory):
- Output ONLY a single T-SQL SELECT statement. No markdown fences, no prose, no explanation.
- Only reference the tables and columns listed above, schema-qualified exactly as shown.
- Never use INSERT/UPDATE/DELETE/MERGE/DROP/ALTER/CREATE/TRUNCATE/EXEC/SELECT..INTO or any DDL/DML.
- Exactly one statement. No semicolons except optionally one at the very end.
- Always include an explicit row cap: either "TOP (n)" right after SELECT, or an
  ORDER BY followed by "OFFSET 0 ROWS FETCH NEXT n ROWS ONLY". Use n <= {MAX_ROWS}.
- If the question can't be answered from the tables above, output exactly: NO_QUERY

Question: {question}

SQL:"""
    try:
        raw = llm_complete(prompt)
    except Exception as exc:
        raise SqlGenerationError(f"LLM call failed: {exc}") from exc

    sql = _strip_code_fence(raw or "")
    if not sql or sql.strip().upper() == "NO_QUERY":
        raise SqlGenerationError("Model reported the question can't be answered from the available schema.")
    return sql


def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()

    if not s:
        raise SqlValidationError("Empty SQL.")
    if len(s) > _MAX_SQL_CHARS:
        raise SqlValidationError("Generated SQL is unexpectedly long - rejected as a precaution.")
    if ";" in s:
        raise SqlValidationError("Multiple statements are not allowed.")
    if _COMMENT_RE.search(s):
        raise SqlValidationError("SQL comments are not allowed in generated queries.")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise SqlValidationError("Only SELECT (or WITH ... SELECT) statements are allowed.")
    if _BLOCKED_RE.search(s):
        raise SqlValidationError("Generated SQL contains a disallowed keyword.")
    if _SP_XP_RE.search(s):
        raise SqlValidationError("Generated SQL references a system procedure - rejected.")

    cte_names = {m.group(1).lower() for m in _CTE_NAME_RE.finditer(s)}
    allowed = schema_catalog.allowed_tables()
    referenced = [_last_identifier(m.group(1)) for m in _TABLE_REF_RE.finditer(s)]
    if not referenced:
        raise SqlValidationError("No FROM/JOIN target found in generated SQL.")
    for name in referenced:
        if name not in allowed and name not in cte_names:
            raise SqlValidationError(f"Generated SQL references an unrecognized table: '{name}'.")

    top_match = _TOP_RE.search(s)
    fetch_match = _FETCH_RE.search(s)
    if not top_match and not fetch_match:
        raise SqlValidationError(
            "Generated SQL has no row cap (TOP n / OFFSET..FETCH NEXT n ROWS ONLY)."
        )
    if top_match and int(top_match.group(1)) > MAX_ROWS:
        s = s[:top_match.start(1)] + str(MAX_ROWS) + s[top_match.end(1):]
    if fetch_match and int(fetch_match.group(1)) > MAX_ROWS:
        # Recompute in case the TOP substitution above shifted offsets.
        fetch_match = _FETCH_RE.search(s)
        s = s[:fetch_match.start(1)] + str(MAX_ROWS) + s[fetch_match.end(1):]

    return s


def execute_sql(sql: str) -> list[dict]:
    with get_connection() as conn:
        try:
            conn.timeout = QUERY_TIMEOUT_SECONDS
        except (AttributeError, pyodbc.Error):
            pass
        cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS)
        return [{str(k).lower(): v for k, v in zip(cols, r)} for r in rows]


def run_text_to_sql(question: str, llm_complete) -> dict:
    """Returns {"sql": str, "rows": list[dict]} on success. Raises
    SqlGenerationError or SqlValidationError/FabricConnectionError on
    failure - the caller (ai_service.copilot_chat) is expected to catch
    these and fall back to the existing keyword-based path."""
    sql = generate_sql(question, llm_complete)
    validated = validate_sql(sql)
    try:
        rows = execute_sql(validated)
    except FabricConnectionError:
        raise
    except Exception as exc:
        logger.error("Text-to-SQL execution failed for %r: %s", validated, exc)
        raise
    return {"sql": validated, "rows": rows}