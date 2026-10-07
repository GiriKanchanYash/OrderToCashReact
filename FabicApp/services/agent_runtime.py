"""Fabric agent runtime for the Order-to-Cash agents.

Mirrors app/services/agent_runtime.py (same public names, same return shapes)
so FabicApp/services/ai_service.py is a call-for-call copy of the Snowflake
service. Snowflake-only building blocks are replaced as follows:

  Snowflake                               Fabric
  --------------------------------------  -----------------------------------------
  Cortex Agent (REST :run)                Azure OpenAI chat + agent instructions
  Copilot agent's Cortex Analyst tool     fabric_analyst (validated T-SQL) + summary
  SNOWFLAKE.CORTEX.COMPLETE               Azure OpenAI chat (azure_llm.complete)
  Cortex Search over policies             POLICY_KB keyword search (same fallback
                                          the Snowflake code already uses)
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from FabicApp.db import DB, get_connection, run_query
from FabicApp.services import azure_llm, fabric_analyst

log = logging.getLogger(__name__)

O2C_AGENTS_ENABLED = os.getenv("O2C_AGENTS_ENABLED", "true").lower() in ("1", "true", "yes")
O2C_AGENT_SCHEMA = os.getenv("O2C_AGENT_SCHEMA", "O2C_AGENT")
O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC = float(os.getenv("O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC", "90"))
O2C_COLLECTIONS_AGENT_TIMEOUT_SEC = float(os.getenv("O2C_COLLECTIONS_AGENT_TIMEOUT_SEC", "120"))
O2C_WORKBENCH_AGENT_TIMEOUT_SEC = float(os.getenv("O2C_WORKBENCH_AGENT_TIMEOUT_SEC", "90"))
O2C_AGENT_SKIP_SQL_FALLBACK = os.getenv("O2C_AGENT_SKIP_SQL_FALLBACK", "true").lower() in ("1", "true", "yes")
O2C_WORKBENCH_USE_COMPLETE = os.getenv("O2C_WORKBENCH_USE_COMPLETE", "true").lower() in ("1", "true", "yes")
# Kept for parity with the Snowflake settings; on Fabric the Azure OpenAI
# deployment (AZURE_OPENAI_DEPLOYMENT) is always used.
O2C_WORKBENCH_COMPLETE_MODEL = azure_llm.deployment()
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

_AGENT_INSTRUCTIONS = {
    "copilot": (
        "You are the O2C Copilot, an accounts-receivable analyst. Answer only from the query results "
        "provided. Format exactly: **Descriptive**: what the data shows with specific numbers (start "
        "with a clear **Yes**/**No** for yes/no questions). **Prescriptive**: 4-5 bullets, each "
        "'- **[Finding]**: finding with numbers. **Action:** concrete step. **Why it matters:** impact.'"
    ),
    "billing": (
        "You are the O2C Billing Exceptions Agent. Diagnose why sales orders cannot be invoiced "
        "(credit blocks, missing goods issue, milestones) and give the release/escalation plan. "
        "Use only the ERP facts and policies supplied; cite POLICY_ID values."
    ),
    "collections": (
        "You are the O2C Collections Agent. Produce collector action plans (dunning, late fees, "
        "promise-to-pay, escalation) using only the supplied invoices, customer context and policies; "
        "cite POLICY_ID values."
    ),
    "cash": (
        "You are the O2C Cash Application Agent. Match incoming payments to open invoices, explain "
        "short-pays/unapplied cash and next steps using only the supplied data and policies; cite POLICY_ID values."
    ),
    "dispute": (
        "You are the O2C Dispute Agent. Assess deductions/disputes, recommend resolution (credit memo, "
        "rebill, reject) and customer communication using only the supplied data and policies; cite POLICY_ID values."
    ),
}
_COMMON_RULES = (
    " Never invent invoice numbers, amounts or dates. Follow the requested output format exactly. "
    "Do not describe tool calls or your reasoning process."
)


def agents_enabled() -> bool:
    return O2C_AGENTS_ENABLED


def agent_fqn(agent_key: str) -> str:
    name = AGENT_NAMES.get(agent_key, agent_key)
    if "." in name:
        return name
    return f"{DB}.{O2C_AGENT_SCHEMA}.{name}"


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


def _split_desc_presc(full_text: str) -> tuple[str, str]:
    desc, presc = full_text, ""
    if "**Prescriptive**:" in full_text:
        parts = full_text.split("**Prescriptive**:", 1)
        desc = parts[0].replace("**Descriptive**:", "").strip()
        presc = parts[1].strip()
    elif "**Descriptive**:" in full_text:
        desc = full_text.replace("**Descriptive**:", "").strip()
    return desc, presc


def _run_copilot_agent(user_message: str) -> dict[str, Any]:
    """Copilot agent = analyst tool (SQL over BUSINESS_MART) + grounded write-up.

    Like the Cortex Copilot agent's tool choice: policy questions (sent with the
    POLICY QUESTION directive and the policy excerpts) are answered from the
    excerpts alone, without the SQL tool."""
    if user_message.lstrip().upper().startswith("POLICY QUESTION"):
        text = azure_llm.complete(
            user_message,
            system="You are the O2C Copilot answering Order-to-Cash policy questions strictly from the "
                   "policy excerpts provided." + _COMMON_RULES,
        )
        return {"text": text, "rows": [], "sql": ""}
    analyst = fabric_analyst.call_analyst("", user_message)
    if "error" in analyst:
        raise RuntimeError(str(analyst["error"]))
    sql = ""
    for block in analyst.get("message", {}).get("content", []):
        if block.get("type") == "sql":
            sql = block.get("statement") or ""
    rows: list[dict] = []
    if sql:
        with get_connection() as conn:
            rows = fabric_analyst.run_analyst_sql(conn, sql)
    data = json.dumps(rows[:25], default=str, ensure_ascii=False)[:6000]
    text = azure_llm.complete(
        f"Question: {user_message}\n\nQuery results ({len(rows)} rows, first 25 shown):\n{data}",
        system=_AGENT_INSTRUCTIONS["copilot"] + _COMMON_RULES,
    )
    return {"text": text, "rows": rows, "sql": sql}


def run_o2c_agent(agent_key: str, user_message: str) -> dict[str, Any]:
    """Fabric counterpart of the Snowflake Cortex Agent call."""
    fqn = agent_fqn(agent_key)
    if agent_key == "copilot":
        out = _run_copilot_agent(user_message)
    else:
        system = _AGENT_INSTRUCTIONS.get(agent_key, _AGENT_INSTRUCTIONS["copilot"]) + _COMMON_RULES
        out = {"text": azure_llm.complete(user_message, system=system), "rows": [], "sql": ""}
    full_text = (out.get("text") or "").strip()
    desc, presc = _split_desc_presc(full_text)
    return {
        "text": full_text,
        "descriptive": desc,
        "prescriptive": presc,
        "rows": out.get("rows") or [],
        "sql": out.get("sql") or "",
        "agent_used": True,
        "raw": None,
        "agent_name": fqn,
    }


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
            raise TimeoutError(f"Fabric agent timed out after {limit:.0f}s")


def agent_text_or_raise(result: dict[str, Any]) -> str:
    text = (result.get("text") or result.get("descriptive") or "").strip()
    if not text:
        raise RuntimeError("Agent returned no text content")
    return text


def run_workbench_complete(prompt: str, *, model: str | None = None) -> str:
    """Fast path (Snowflake: single Cortex COMPLETE call) -> single Azure OpenAI call."""
    del model
    return azure_llm.complete(prompt, timeout=O2C_WORKBENCH_AGENT_TIMEOUT_SEC)


def _search_policies_kb(query: str, policy_domain: str | None, limit: int) -> list[dict]:
    import re

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
    """Load policy snippets from the Fabric POLICY_KB table (no Cortex Search on Fabric)."""
    lim = max(1, min(int(limit or 5), 10))
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
