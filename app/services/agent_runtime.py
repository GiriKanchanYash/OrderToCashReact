"""Snowflake Cortex Agent runtime for Order-to-Cash agents (see sql/08_cortex_agents_and_tools.sql)."""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

import requests

from app.db import DB, get_connection, run_query, run_query_on_conn

log = logging.getLogger(__name__)

O2C_AGENTS_ENABLED = os.getenv("O2C_AGENTS_ENABLED", "true").lower() in ("1", "true", "yes")
O2C_AGENT_SCHEMA = os.getenv("O2C_AGENT_SCHEMA", "O2C_AGENT")
O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC = float(os.getenv("O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC", "90"))
O2C_COLLECTIONS_AGENT_TIMEOUT_SEC = float(os.getenv("O2C_COLLECTIONS_AGENT_TIMEOUT_SEC", "120"))
O2C_WORKBENCH_AGENT_TIMEOUT_SEC = float(os.getenv("O2C_WORKBENCH_AGENT_TIMEOUT_SEC", "90"))
O2C_AGENT_SKIP_SQL_FALLBACK = os.getenv("O2C_AGENT_SKIP_SQL_FALLBACK", "true").lower() in ("1", "true", "yes")
O2C_WORKBENCH_USE_COMPLETE = os.getenv("O2C_WORKBENCH_USE_COMPLETE", "true").lower() in ("1", "true", "yes")
O2C_WORKBENCH_COMPLETE_MODEL = os.getenv("O2C_WORKBENCH_COMPLETE_MODEL", "mistral-large2").strip() or "mistral-large2"
O2C_WORKBENCH_COMPLETE_FALLBACK_AGENT = os.getenv(
    "O2C_WORKBENCH_COMPLETE_FALLBACK_AGENT", "true",
).lower() in ("1", "true", "yes")

WORKBENCH_AGENT_KEYS = frozenset({"billing", "collections", "cash", "dispute"})

AGENT_NAMES = {
    "copilot": "O2C_COPILOT_AGENT",
    "billing": "O2C_BILLING_AGENT",
    "collections": "O2C_COLLECTIONS_AGENT",
    "cash": "O2C_CASH_APPLICATION_AGENT",
    "dispute": "O2C_DISPUTE_AGENT",
}

AGENT_LABELS = {
    "copilot": "O2C Copilot",
    "billing": "O2C Billing Exceptions Agent",
    "collections": "O2C Collections Agent",
    "cash": "O2C Cash Application Agent",
    "dispute": "O2C Dispute Agent",
}

POLICY_DOMAIN_BY_AGENT = {
    "billing": "BILLING",
    "collections": "COLLECTIONS",
    "cash": "CASH_APP",
    "dispute": "DISPUTE",
    "copilot": None,
}


def agents_enabled() -> bool:
    return O2C_AGENTS_ENABLED


def agent_fqn(agent_key: str) -> str:
    name = AGENT_NAMES.get(agent_key, agent_key)
    if "." in name:
        return name
    return f"{DB}.{O2C_AGENT_SCHEMA}.{name}"


def _agent_parts(agent_key: str) -> tuple[str, str, str]:
    fqn = agent_fqn(agent_key)
    parts = fqn.split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return DB, O2C_AGENT_SCHEMA, AGENT_NAMES.get(agent_key, agent_key)


def _extract_rows_from_json_obj(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return [dict(r) for r in obj]
        return []
    if isinstance(obj, dict):
        for key in ("data", "rows", "result", "results", "records"):
            val = obj.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return [dict(r) for r in val]
        if all(isinstance(k, str) for k in obj.keys()) and obj:
            return [dict(obj)]
    return []


def _append_text(texts: list[str], value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        texts.append(text)


def _row_looks_like_debug(row: dict) -> bool:
    keys = {str(k).lower() for k in row.keys()}
    if keys <= {"pruned_note"}:
        return True
    if "pruned_note" in keys:
        note = str(row.get("pruned_note") or row.get("PRUNED_NOTE") or "").lower()
        if "semantic model" in note or "token budget" in note:
            return True
    vals = " ".join(str(v) for v in row.values() if v is not None).lower()
    return "semantic_model_key" in vals or "left_outer" in vals


def _rows_look_like_debug(rows: list[dict] | None) -> bool:
    if not rows:
        return True
    return all(_row_looks_like_debug(r) for r in rows if isinstance(r, dict))


def _walk_sql_statements(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        if obj.get("type") == "sql":
            stmt = obj.get("statement") or obj.get("sql")
            if isinstance(stmt, str) and stmt.strip():
                out.append(stmt.strip())
        stmt = obj.get("statement")
        if isinstance(stmt, str) and "select" in stmt.lower() and stmt not in out:
            out.append(stmt.strip())
        for value in obj.values():
            _walk_sql_statements(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_sql_statements(item, out)


def _execute_agent_sql(sql_stmts: list[str]) -> tuple[list[dict], str]:
    seen: set[str] = set()
    for stmt in sql_stmts:
        normalized = stmt.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if "select" not in normalized.lower():
            continue
        try:
            rows = run_query(normalized)
            if rows and not _rows_look_like_debug(rows):
                return rows, normalized
        except Exception:
            log.warning("Failed to execute SQL extracted from agent response", exc_info=True)
    return [], ""


def _collect_text_and_rows(content: list | None, texts: list[str], rows: list[dict]) -> None:
    if not content:
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            _append_text(texts, block.get("text") or block.get("content"))
        elif btype == "thinking":
            continue
        elif btype in ("tool_result", "tool_results"):
            tr = block.get("tool_result") or block.get("tool_results") or block
            inner = tr.get("content") if isinstance(tr, dict) else None
            if isinstance(inner, list):
                for part in inner:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        _append_text(texts, part.get("text"))
                    if part.get("type") == "json":
                        rows.extend(_extract_rows_from_json_obj(part.get("json")))
                    if part.get("type") == "table":
                        rows.extend(_extract_rows_from_json_obj(part.get("table")))
                    if part.get("type") == "sql":
                        stmt = part.get("statement") or part.get("sql")
                        if isinstance(stmt, str):
                            block["_sql_stmt"] = stmt
            elif isinstance(tr, dict):
                rows.extend(_extract_rows_from_json_obj(tr.get("data") or tr.get("json")))
                _append_text(texts, tr.get("text"))
                stmt = tr.get("statement") or tr.get("sql")
                if isinstance(stmt, str):
                    tr["_sql_stmt"] = stmt
        elif btype == "json":
            rows.extend(_extract_rows_from_json_obj(block.get("json")))
        elif btype == "tool_use":
            continue
        else:
            _append_text(texts, block.get("text"))


def _walk_messages_for_text(raw: dict, texts: list[str], rows: list[dict]) -> None:
    if raw.get("role") == "assistant":
        _collect_text_and_rows(raw.get("content"), texts, rows)

    messages = raw.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                _collect_text_and_rows(msg.get("content"), texts, rows)
        if not texts:
            for msg in messages:
                if isinstance(msg, dict):
                    _collect_text_and_rows(msg.get("content"), texts, rows)

    for key in ("result", "response", "output", "data"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            _walk_messages_for_text(nested, texts, rows)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    _walk_messages_for_text(item, texts, rows)


def parse_agent_response(raw: Any) -> dict[str, Any]:
    """Normalize Cortex Agent JSON into text + optional tabular rows."""
    if raw is None:
        return {"text": "", "rows": [], "sql": "", "agent_used": True}

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw, "rows": [], "sql": "", "agent_used": True}

    texts: list[str] = []
    rows: list[dict] = []
    sql_stmts: list[str] = []

    if isinstance(raw, dict):
        if raw.get("error"):
            err = raw.get("error")
            if isinstance(err, dict):
                raise RuntimeError(err.get("message") or str(err))
            raise RuntimeError(str(err))

        _walk_messages_for_text(raw, texts, rows)
        _collect_text_and_rows(raw.get("content"), texts, rows)

        for stmt in raw.get("sql_statements") or []:
            if stmt:
                sql_stmts.append(str(stmt))
        _walk_sql_statements(raw, sql_stmts)

    # De-dupe while preserving order
    seen_sql: set[str] = set()
    unique_sql: list[str] = []
    for stmt in sql_stmts:
        if stmt not in seen_sql:
            seen_sql.add(stmt)
            unique_sql.append(stmt)
    sql_stmts = unique_sql

    if _rows_look_like_debug(rows) and sql_stmts:
        exec_rows, exec_sql = _execute_agent_sql(sql_stmts)
        if exec_rows:
            rows = exec_rows
            if exec_sql:
                sql_stmts = [exec_sql]

    # De-dupe while preserving order
    seen: set[str] = set()
    unique_texts: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    full_text = "\n\n".join(unique_texts).strip()
    desc, presc = full_text, ""
    if "**Prescriptive**:" in full_text:
        parts = full_text.split("**Prescriptive**:", 1)
        desc = parts[0].replace("**Descriptive**:", "").strip()
        presc = parts[1].strip()
    elif "**Descriptive**:" in full_text:
        desc = full_text.replace("**Descriptive**:", "").strip()

    if not full_text and rows:
        preview = "; ".join(
            f"{k}={v}" for k, v in list(rows[0].items())[:4]
        )
        full_text = f"Agent returned {len(rows)} row(s). Sample: {preview}"

    return {
        "text": full_text,
        "descriptive": desc,
        "prescriptive": presc,
        "rows": rows,
        "sql": sql_stmts[-1] if sql_stmts else "",
        "agent_used": True,
        "raw": raw if isinstance(raw, dict) else None,
    }


def _post_agent_rest(conn, database: str, schema: str, agent_name: str, body: dict) -> Any:
    token = conn.rest.token
    host = conn.host
    url = (
        f"https://{host}/api/v2/databases/{database}/schemas/{schema}"
        f"/agents/{agent_name}:run"
    )
    headers = {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=180)
    except requests.exceptions.SSLError:
        alt_host = conn.account.replace("_", "-")
        if not alt_host.endswith(".snowflakecomputing.com"):
            alt_host = f"{alt_host}.snowflakecomputing.com"
        url = (
            f"https://{alt_host}/api/v2/databases/{database}/schemas/{schema}"
            f"/agents/{agent_name}:run"
        )
        resp = requests.post(url, json=body, headers=headers, timeout=180)

    if resp.status_code >= 400:
        raise RuntimeError(f"Agent REST HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _run_agent_sql(conn, fqn: str, body: dict) -> Any:
    payload_json = json.dumps(body, ensure_ascii=False)
    rows = run_query_on_conn(
        conn,
        """
        SELECT TRY_PARSE_JSON(
            SNOWFLAKE.CORTEX.DATA_AGENT_RUN(
                %s,
                PARSE_JSON(%s),
                TRUE
            )
        ) AS resp
        """,
        (fqn, payload_json),
    )
    if not rows or rows[0].get("resp") is None:
        raise RuntimeError(f"Agent {fqn} returned empty SQL response")
    raw = rows[0]["resp"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


def run_o2c_agent(agent_key: str, user_message: str) -> dict[str, Any]:
    """Invoke a Snowflake Cortex Agent (REST preferred, SQL fallback)."""
    database, schema, agent_name = _agent_parts(agent_key)
    fqn = f"{database}.{schema}.{agent_name}"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}],
            }
        ],
        "stream": False,
    }

    errors: list[str] = []
    raw: Any = None

    with get_connection() as conn:
        try:
            raw = _post_agent_rest(conn, database, schema, agent_name, body)
        except Exception as exc:
            errors.append(f"REST: {exc}")
            log.warning("Agent REST call failed for %s", fqn, exc_info=True)
            if not O2C_AGENT_SKIP_SQL_FALLBACK:
                try:
                    raw = _run_agent_sql(conn, fqn, body)
                except Exception as exc2:
                    errors.append(f"SQL: {exc2}")
                    log.warning("Agent SQL call failed for %s", fqn, exc_info=True)

    if raw is None:
        raise RuntimeError("; ".join(errors) or f"Agent {fqn} unavailable")

    parsed = parse_agent_response(raw)
    parsed["agent_name"] = fqn
    if errors and parsed.get("text"):
        parsed["transport_note"] = "recovered_via_sql" if errors[0].startswith("REST") else None
    return parsed


def run_o2c_agent_with_timeout(
    agent_key: str,
    user_message: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    limit = O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC if timeout is None else max(5.0, float(timeout))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_o2c_agent, agent_key, user_message)
        try:
            return future.result(timeout=limit)
        except FuturesTimeout:
            raise TimeoutError(f"Snowflake agent timed out after {limit:.0f}s")


def agent_text_or_raise(result: dict[str, Any]) -> str:
    text = (result.get("text") or result.get("descriptive") or "").strip()
    if not text:
        raise RuntimeError("Agent returned no text content")
    return text


def run_workbench_complete(prompt: str, *, model: str | None = None) -> str:
    """Single Cortex COMPLETE call — fast path when ERP/policy context is already in the prompt."""
    m = (model or O2C_WORKBENCH_COMPLETE_MODEL).strip()
    rows = run_query(
        'SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS "response"',
        (m, prompt),
    )
    if not rows:
        raise RuntimeError("Cortex Complete returned no rows")
    text = str(rows[0].get("response") or rows[0].get("RESPONSE") or "").strip()
    if not text:
        raise RuntimeError("Cortex Complete returned empty text")
    return text


def _policy_id_from_chunk(chunk: str, source_file: str = "") -> str:
    text = f"{source_file} {chunk or ''}"
    match = re.search(r"\b([A-Z]{2,5}-[A-Z]{2,5}-[0-9]{2})\b", text)
    return match.group(1) if match else ""


def _search_policies_cortex(query: str, policy_domain: str | None, limit: int) -> list[dict]:
    safe_q = (query or "order to cash").replace("'", "''")[:400]
    domain = (policy_domain or "").strip().upper().replace("'", "''")
    filter_arg = ""
    if domain:
        filter_arg = f", FILTER => OBJECT_CONSTRUCT('@eq', OBJECT_CONSTRUCT('policy_domain', '{domain}'))"
    rows = run_query(f"""
        SELECT
            r.value:chunk::varchar AS chunk,
            r.value:source_file::varchar AS source_file,
            r.value:policy_domain::varchar AS policy_domain
        FROM TABLE(
            {DB}.{O2C_AGENT_SCHEMA}.POLICY_SEARCH_SVC!SEARCH(
                QUERY => '{safe_q}',
                COLUMNS => ARRAY_CONSTRUCT('chunk', 'source_file', 'policy_domain'),
                LIMIT => {limit}
                {filter_arg}
            )
        ) r
    """)
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        chunk = str(row.get("chunk") or row.get("CHUNK") or "")
        source_file = str(row.get("source_file") or row.get("SOURCE_FILE") or "")
        pid = _policy_id_from_chunk(chunk, source_file)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append({
            "policy_id": pid,
            "policy_domain": str(row.get("policy_domain") or row.get("POLICY_DOMAIN") or policy_domain or ""),
            "policy_area": "",
            "policy_title": "",
            "policy_text": chunk[:1200],
        })
    if not out:
        return []
    ids = ",".join(f"'{p['policy_id'].replace(chr(39), chr(39)*2)}'" for p in out)
    meta = run_query(f"""
        SELECT POLICY_ID, POLICY_DOMAIN, POLICY_AREA, POLICY_TITLE, POLICY_TEXT
        FROM {DB}.{O2C_AGENT_SCHEMA}.POLICY_KB
        WHERE POLICY_ID IN ({ids})
    """)
    meta_by_id = {str(r.get("POLICY_ID") or r.get("policy_id")): r for r in meta}
    for item in out:
        m = meta_by_id.get(item["policy_id"], {})
        item["policy_domain"] = str(m.get("POLICY_DOMAIN") or m.get("policy_domain") or item["policy_domain"])
        item["policy_area"] = str(m.get("POLICY_AREA") or m.get("policy_area") or "")
        item["policy_title"] = str(m.get("POLICY_TITLE") or m.get("policy_title") or item["policy_id"])
        item["policy_text"] = str(m.get("POLICY_TEXT") or m.get("policy_text") or item["policy_text"])
    return out


def _search_policies_kb(query: str, policy_domain: str | None, limit: int) -> list[dict]:
    domain = (policy_domain or "").strip().upper().replace("'", "''")
    domain_clause = f"AND POLICY_DOMAIN = '{domain}'" if domain else ""
    rows = run_query(f"""
        SELECT
            POLICY_ID AS policy_id,
            POLICY_DOMAIN AS policy_domain,
            POLICY_AREA AS policy_area,
            POLICY_TITLE AS policy_title,
            POLICY_TEXT AS policy_text
        FROM {DB}.{O2C_AGENT_SCHEMA}.POLICY_KB
        WHERE (EFFECTIVE_TO IS NULL OR EFFECTIVE_TO >= CURRENT_DATE())
        {domain_clause}
    """)
    if not rows:
        return []

    terms = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", query or "") if len(t) > 2][:8]
    scored: list[tuple[int, dict]] = []
    for row in rows:
        blob = " ".join(
            str(row.get(k) or "")
            for k in ("policy_id", "policy_title", "policy_text", "policy_area")
        ).lower()
        score = sum(2 if t in blob else 0 for t in terms)
        if domain and str(row.get("policy_domain") or "").upper() == domain:
            score += 1
        scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("policy_id") or "")))
    top = [r for s, r in scored if s > 0][:limit]
    if not top and domain:
        top = [r for _, r in scored][:limit]
    return top


def search_policies(query: str, policy_domain: str | None = None, limit: int = 5) -> list[dict]:
    """Load policy snippets from Cortex Search with POLICY_KB fallback."""
    lim = max(1, min(int(limit or 5), 10))
    try:
        rows = _search_policies_cortex(query, policy_domain, lim)
        if rows:
            return rows
    except Exception:
        log.warning("Cortex policy search failed for query=%s domain=%s", (query or "")[:80], policy_domain, exc_info=True)
    try:
        return _search_policies_kb(query, policy_domain, lim)
    except Exception:
        log.warning("POLICY_KB search failed for query=%s domain=%s", (query or "")[:80], policy_domain, exc_info=True)
        return []


def policies_for_api(policies: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in policies:
        out.append({
            "policy_id": str(p.get("policy_id") or p.get("POLICY_ID") or ""),
            "policy_domain": str(p.get("policy_domain") or p.get("POLICY_DOMAIN") or ""),
            "policy_title": str(p.get("policy_title") or p.get("POLICY_TITLE") or ""),
            "policy_text": str(p.get("policy_text") or p.get("POLICY_TEXT") or ""),
        })
    return [x for x in out if x["policy_id"]]


def format_policy_block(policies: list[dict], *, max_text: int = 220) -> str:
    lines: list[str] = []
    for p in policies:
        pid = p.get("policy_id") or p.get("POLICY_ID")
        title = p.get("policy_title") or p.get("POLICY_TITLE")
        text = str(p.get("policy_text") or p.get("POLICY_TEXT") or "")[:max_text].strip()
        if pid and text:
            lines.append(f"- [{pid}] {title}: {text}")
    return "\n".join(lines)
