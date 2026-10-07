from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid

import requests

from FabicApp.db import BM, DB, SCHEMA, get_connection, promote_staging_to_mart, run_query, run_query_on_conn
from FabicApp.services import azure_llm, fabric_analyst
from FabicApp.services.billing_block_catalog import billing_problem_summary, block_scenario_lines
from FabicApp.services.agent_response_format import (
    BILLING_OUTPUT_FORMAT,
    CASH_APPLICATION_OUTPUT_FORMAT,
    COLLECTIONS_OUTPUT_FORMAT,
    DISPUTE_OUTPUT_FORMAT,
    build_billing_fallback,
    build_cash_fallback,
    build_collections_fallback,
    build_dispute_fallback,
    contact_log_lines,
    customer_history_lines,
    format_invoice_context_block,
    format_order_context_block,
    invoice_lines,
    invoice_detail_lines,
    enrich_invoice_row,
    open_dispute_lines,
    policy_ref_lines,
    polish_billing_response,
    polish_cash_response,
    polish_collections_response,
    polish_dispute_response,
)
from FabicApp.services.agent_runtime import (
    agent_text_or_raise,
    agents_enabled,
    format_policy_block,
    policies_for_api,
    run_o2c_agent,
    run_o2c_agent_with_timeout,
    run_workbench_complete,
    search_policies,
    AGENT_LABELS,
    O2C_COLLECTIONS_AGENT_TIMEOUT_SEC,
    O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC,
    O2C_WORKBENCH_AGENT_TIMEOUT_SEC,
    O2C_WORKBENCH_USE_COMPLETE,
    WORKBENCH_AGENT_KEYS,
    POLICY_DOMAIN_BY_AGENT,
    _rows_look_like_debug,
)

_AGENT_CACHE: dict[str, tuple[float, object]] = {}
_AGENT_CACHE_LOCK = threading.Lock()
_AGENT_CACHE_TTL = 300
_RECOMMENDATION_CACHE_TTL = 1800
_AGENT_RECOMMENDATION_MODE = os.getenv("O2C_AGENT_RECOMMENDATION_MODE", "auto").strip().lower()

log = logging.getLogger(__name__)

WORKBENCH_AGENT_DIRECTIVE = (
    "WORKBENCH MODE — respond in one pass:\n"
    "- Do NOT call any tools (no policy_search, queue lookups, or SQL procedures).\n"
    "- All ERP facts, policies, and context are already in this message.\n"
    "- Output the final structured recommendation immediately.\n\n"
)

_WORKBENCH_CACHE_SUFFIX = "wb-live-v12"


def _val(row: dict, *keys, default=None):
    """Read a field from a Fabric row (keys may be upper or lower case)."""
    for k in keys:
        for variant in (k, str(k).lower(), str(k).upper()):
            if variant in row and row[variant] is not None:
                return row[variant]
    return default


# Fabric: the semantic model is FabicApp/Schema_model.yml (+ live BUSINESS_MART
# catalog), read by fabric_analyst. Kept for parity with the Snowflake module.
SEMANTIC_MODEL = os.getenv("FABRIC_SEMANTIC_MODEL", "FabicApp/Schema_model.yml")

_COPILOT_CHAT_CACHE_TTL = 600
_COPILOT_CHAT_CACHE: dict[str, tuple[float, dict]] = {}
_COPILOT_CHAT_CACHE_LOCK = threading.Lock()
_COPILOT_CHAT_CACHE_VERSION = "v15"

POLICY_COPILOT_DIRECTIVE = (
    "POLICY QUESTION — answer from O2C policy only.\n"
    "Format exactly two sections:\n"
    "**Descriptive:** what the policy requires (cite POLICY_ID values in bold).\n"
    "**Prescriptive:** bullet points with concrete actions. Each bullet must use:\n"
    "- **Topic:** brief policy finding.\n"
    "  **Action:** specific step for finance / collections / billing teams.\n"
    "  **Why it matters:** rationale citing POLICY_ID.\n"
    "Do NOT call o2c_kpi_analyst or data_to_chart. Use policy_search only if excerpts below are insufficient.\n\n"
)


def _cache_get(key: str) -> dict | None:
    now = time.time()
    with _COPILOT_CHAT_CACHE_LOCK:
        ent = _COPILOT_CHAT_CACHE.get(key)
        if not ent:
            return None
        ts, value = ent
        if now - ts > _COPILOT_CHAT_CACHE_TTL:
            _COPILOT_CHAT_CACHE.pop(key, None)
            return None
        return copy.deepcopy(value)


def _cache_set(key: str, value: dict) -> None:
    with _COPILOT_CHAT_CACHE_LOCK:
        _COPILOT_CHAT_CACHE[key] = (time.time(), copy.deepcopy(value))


def call_cortex_analyst(question: str, conn=None) -> dict:
    instruction = (
        "Do NOT start with 'This is our interpretation of your question.' "
        "Start directly with **Descriptive**: then **Prescriptive**:. "
        "For ANY YES/NO question: Start the Descriptive section with a clear **Yes** or **No** answer first, then explain with specific numbers. "
        "(1) **Descriptive**: What the data shows with specific numbers and evidence. "
        "(2) **Prescriptive**: Provide a header 'Here are the bullet points with specific findings, concrete actions, and explanations:' then list 4-5 SPECIFIC bullet points. "
        "Each bullet must follow this EXACT format: "
        "- **[Finding Title]**: [Specific finding with actual numbers from the data]. **Action:** [Concrete action to take]. **Why it matters:** [Business impact explanation]. "
        "Answer the following question:\n\n"
    )
    # Fabric: Azure OpenAI analyst over the BUSINESS_MART views; returns the same
    # {"message": {"content": [text, sql]}} shape as the Cortex Analyst REST API.
    # The Descriptive/Prescriptive instruction above is intentionally NOT sent:
    # like Cortex Analyst, the SQL generator never sees query results, so it must
    # not write numbers. The narrative is built from the executed rows by
    # _copilot_via_cortex_analyst / _finalize_copilot, as on Snowflake.
    del conn, instruction
    return fabric_analyst.call_analyst("", question)


def get_current_user() -> str:
    try:
        rows = run_query("SELECT CURRENT_USER() AS USERNAME")
        if rows:
            return str(rows[0].get("USERNAME", "UNKNOWN"))
    except Exception:
        log.warning("Could not fetch current user", exc_info=True)
    return "UNKNOWN"


def save_question_history(question: str, qtype: str = "custom") -> None:
    user = get_current_user()
    try:
        safe_q = question.replace("'", "''")
        run_query(f"""
            MERGE INTO {DB}.{SCHEMA}.GENIE_QUESTION_HISTORY t
            USING (SELECT '{safe_q}' AS NORMALIZED_QUERY, '{qtype}' AS TYPE, '{user}' AS "USER", 'ORDERLENS' AS PERSONA) s
            ON t.NORMALIZED_QUERY = s.NORMALIZED_QUERY AND t."USER" = s."USER"
            WHEN MATCHED THEN UPDATE SET FREQUENCY = t.FREQUENCY + 1, LAST_ASKED_AT = CURRENT_TIMESTAMP(), PERSONA = 'ORDERLENS'
            WHEN NOT MATCHED THEN INSERT (NORMALIZED_QUERY, TYPE, "USER", FREQUENCY, PERSONA)
                VALUES (s.NORMALIZED_QUERY, s.TYPE, s."USER", 1, s.PERSONA)
        """)
    except Exception:
        log.warning("Could not save question history", exc_info=True)


def load_saved_insights() -> list[dict]:
    user = get_current_user()
    try:
        return run_query(f"""
            SELECT INSIGHT_ID, TITLE, QUESTION, SQL_TEXT
            FROM {DB}.{SCHEMA}.SAVED_INSIGHTS
            WHERE CREATED_BY = '{user}'
              AND PAGE IN ('{_COPILOT_PAGE}', 'copilot')
            ORDER BY CREATED_AT DESC
            LIMIT 20
        """)
    except Exception:
        log.warning("Could not load saved insights", exc_info=True)
    return []


def save_insight(title: str, question: str, sql_text: str = "") -> None:
    user = get_current_user()
    try:
        safe_title = title.replace("'", "''")
        safe_q = question.replace("'", "''")
        safe_sql = sql_text.replace("'", "''") if sql_text else ""
        run_query(f"""
            INSERT INTO {DB}.{SCHEMA}.SAVED_INSIGHTS (CREATED_BY, PAGE, TITLE, QUESTION, SQL_TEXT)
            VALUES ('{user}', '{_COPILOT_PAGE}', '{safe_title}', '{safe_q}', '{safe_sql}')
        """)
    except Exception:
        log.warning("Could not save insight", exc_info=True)


def delete_insight(insight_id: int) -> None:
    user = get_current_user()
    try:
        run_query(f"DELETE FROM {DB}.{SCHEMA}.SAVED_INSIGHTS WHERE INSIGHT_ID = {insight_id} AND CREATED_BY = '{user}'")
    except Exception:
        log.warning("Could not delete insight", exc_info=True)


def load_frequent_questions() -> list[dict]:
    user = get_current_user()
    try:
        return run_query(f"""
            SELECT NORMALIZED_QUERY, TYPE, FREQUENCY
            FROM {DB}.{SCHEMA}.GENIE_QUESTION_HISTORY
            WHERE "USER" = '{user}'
              AND (PERSONA = 'ORDERLENS' OR PERSONA IS NULL)
              AND {_COPILOT_HISTORY_FILTER}
            ORDER BY FREQUENCY DESC, LAST_ASKED_AT DESC
            LIMIT 10
        """)
    except Exception:
        log.warning("Could not load frequent questions", exc_info=True)
    return []


def load_most_frequent_all() -> list[dict]:
    try:
        return run_query(f"""
            SELECT NORMALIZED_QUERY, TYPE, SUM(FREQUENCY) AS TOTAL_FREQ
            FROM {DB}.{SCHEMA}.GENIE_QUESTION_HISTORY
            WHERE (PERSONA = 'ORDERLENS' OR PERSONA IS NULL)
              AND {_COPILOT_HISTORY_FILTER}
            GROUP BY NORMALIZED_QUERY, TYPE
            ORDER BY TOTAL_FREQ DESC
            LIMIT 10
        """)
    except Exception:
        log.warning("Could not load most frequent questions", exc_info=True)
    return []


def cortex_complete(model: str, prompt: str) -> str:
    """Fabric counterpart of SNOWFLAKE.CORTEX.COMPLETE (model arg kept for parity)."""
    del model
    try:
        return azure_llm.complete(prompt)
    except Exception:
        log.warning("Azure OpenAI completion failed", exc_info=True)
        return ""


_COPILOT_PAGE = "orderlens_copilot"
_COPILOT_HISTORY_FILTER = """
    TYPE NOT IN ('ctb_readiness', 'part_shortages', 'supplier_performance', 'bom_completeness')
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%clear-to-build%'
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%ready to build%'
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%work order%'
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%shortage%'
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%supplier%delivery%'
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%incomplete bom%'
    AND LOWER(NORMALIZED_QUERY) NOT LIKE '%peg%'
"""

COPILOT_QUICK_ANALYSES: dict[str, dict[str, str]] = {
    "dso_overview": {
        "title": "DSO Overview",
        "desc": "Days Sales Outstanding trend by company",
        "question": "What is our current DSO and how has it trended over the last 12 months?",
    },
    "past_due_aging": {
        "title": "Past Due Aging",
        "desc": "Past-due AR breakdown by aging bucket",
        "question": "What is our past due percentage and aging bucket distribution?",
    },
    "collection_effectiveness": {
        "title": "Collection Effectiveness",
        "desc": "CEI trend and collection performance",
        "question": "What is our Collection Effectiveness Index and how is it trending?",
    },
    "cash_forecast_accuracy": {
        "title": "Cash Forecast Accuracy",
        "desc": "MAPE and forecast bias for cash predictions",
        "question": "How accurate are our cash forecasts and where is forecast bias?",
    },
}


def copilot_quick_analyses() -> list[dict]:
    return [{"key": k, **v} for k, v in COPILOT_QUICK_ANALYSES.items()]


def run_quick_analysis(key: str) -> dict:
    qa = COPILOT_QUICK_ANALYSES.get(key)
    if qa:
        save_question_history(qa["question"], key)

    result: dict = {"key": key, "metrics": {}, "rows": [], "sql": "", "descriptive": "", "prescriptive": ""}

    if key == "dso_overview":
        sql = f"""
        SELECT PERIOD_MONTH::VARCHAR AS period, COMPANY_CODE, DSO
        FROM {BM}.dso_vw
        WHERE DSO IS NOT NULL
        ORDER BY PERIOD_MONTH DESC
        LIMIT 50
        """
        rows = run_query(sql)
        latest = rows[0] if rows else {}
        dso_val = float(_val(latest, "DSO", default=0) or 0)
        result["metrics"] = {"summary": f"Latest DSO: {dso_val:.1f} days", "dso": dso_val}
        result["rows"] = rows
        result["sql"] = sql
        result["descriptive"] = f"**DSO Status:** Latest DSO is {dso_val:.1f} days across active company codes."
        result["prescriptive"] = (
            "Here are the bullet points with specific findings, concrete actions, and explanations:\n\n"
            f"- **DSO Level**: Current DSO is {dso_val:.1f} days. **Action:** Compare against industry benchmark and BPDSO target. **Why it matters:** Elevated DSO ties up working capital.\n\n"
            "- **Trend Review**: Analyze monthly DSO trend for deterioration. **Action:** Focus collections on customers driving DSO increase. **Why it matters:** Early intervention prevents cash flow gaps.\n\n"
            "- **Company Variance**: Review DSO by company code for outliers. **Action:** Escalate accounts in high-DSO entities to regional collections leads. **Why it matters:** Concentrated DSO drivers distort portfolio cash timing.\n\n"
            "- **Invoice-to-Cash Cycle**: Pair DSO with dispute and deduction rates. **Action:** Reduce billing errors that extend collection cycles. **Why it matters:** Cleaner invoicing shortens days-to-cash without harder collections tactics."
        )
        result["chart"] = {"type": "line", "xKey": "period", "yKey": "dso", "color": "#0f766e", "title": "DSO Trend"}
        return result

    if key == "past_due_aging":
        sql = f"""
        SELECT COMPANY_CODE, TOTAL_AR, PAST_DUE_AR, PAST_DUE_PCT,
               BUCKET_1_30, BUCKET_31_60, BUCKET_61_90, BUCKET_90_PLUS
        FROM {BM}.past_due_vw
        ORDER BY PAST_DUE_AR DESC
        """
        rows = run_query(sql)
        total_ar = sum(float(_val(r, "TOTAL_AR", default=0) or 0) for r in rows)
        total_past = sum(float(_val(r, "PAST_DUE_AR", default=0) or 0) for r in rows)
        past_due_pct = (total_past / total_ar * 100.0) if total_ar else 0.0
        bucket_90 = sum(float(_val(r, "BUCKET_90_PLUS", default=0) or 0) for r in rows)
        top_company = str(_val(rows[0], "COMPANY_CODE", default="")) if rows else ""
        result["metrics"] = {
            "summary": f"{past_due_pct:.1f}% past due (${total_past:,.0f})",
            "past_due_pct": past_due_pct,
            "total_ar": total_ar,
            "past_due_ar": total_past,
        }
        result["rows"] = rows
        result["sql"] = sql
        result["descriptive"] = (
            f"**Past Due:** {past_due_pct:.1f}% of open AR (${total_past:,.0f} of ${total_ar:,.0f}) is past due. "
            f"The 90+ day bucket totals ${bucket_90:,.0f}."
        )
        result["prescriptive"] = (
            "Here are the bullet points with specific findings, concrete actions, and explanations:\n\n"
            f"- **Past Due Exposure**: {past_due_pct:.1f}% of open AR is past due (${total_past:,.0f}). "
            f"**Action:** Prioritize 90+ bucket accounts for immediate outreach. **Why it matters:** Older balances have lower recovery rates.\n\n"
            f"- **Aging Concentration**: 90+ balances total ${bucket_90:,.0f}. "
            f"**Action:** Escalate dunning for 61-90 day accounts before they move to 90+. **Why it matters:** Proactive collections reduce write-off risk.\n\n"
            f"- **Company Hotspot**: Company {top_company or 'n/a'} has the highest past-due balance. "
            f"**Action:** Review collector assignment and dispute backlog for that entity. **Why it matters:** Local process gaps often drive portfolio-level past-due spikes.\n\n"
            "- **Bucket Migration**: Track weekly movement from 1-30 into 31-60 and 61-90. **Action:** Increase touch frequency on accounts crossing bucket boundaries. **Why it matters:** Early intervention prevents balances from becoming hard-to-collect."
        )
        result["chart"] = {
            "type": "bar_horizontal" if len(rows) > 6 else "bar",
            "xKey": "company_code",
            "yKey": "past_due_pct",
            "color": "#dc2626",
            "title": "Past Due % by Company",
        }
        return result

    if key == "collection_effectiveness":
        sql = f"""
        SELECT PERIOD_MONTH::VARCHAR AS period, AVG(CEI) AS cei
        FROM {BM}.cei_vw
        WHERE CEI IS NOT NULL
        GROUP BY PERIOD_MONTH
        ORDER BY PERIOD_MONTH DESC
        LIMIT 12
        """
        rows = run_query(sql)
        latest_cei = float(_val(rows[0], "CEI", "cei", default=0) or 0) if rows else 0
        result["metrics"] = {"summary": f"Latest CEI: {latest_cei:.1f}%", "cei": latest_cei}
        result["rows"] = rows
        result["sql"] = sql
        result["descriptive"] = f"**CEI:** Collection Effectiveness Index is {latest_cei:.1f}%."
        result["prescriptive"] = (
            "Here are the bullet points with specific findings, concrete actions, and explanations:\n\n"
            f"- **CEI Performance**: Current CEI is {latest_cei:.1f}%. **Action:** Target CEI above 80% through proactive follow-up on new past-due items. **Why it matters:** Higher CEI indicates effective collections processes.\n\n"
            "- **Process Optimization**: Review RPC and PTP kept rates alongside CEI. **Action:** Train collectors on right-party contact techniques. **Why it matters:** Better contact quality improves promise-to-pay conversion.\n\n"
            "- **Portfolio Segmentation**: Focus CEI improvement on high-balance segments first. **Action:** Apply differentiated strategies for strategic vs. long-tail accounts. **Why it matters:** CEI gains on large balances move cash fastest.\n\n"
            "- **Dispute Linkage**: Cross-check CEI dips with dispute volume spikes. **Action:** Resolve billing disputes within SLA to unblock payments. **Why it matters:** Disputes are a leading cause of false past-due and CEI erosion."
        )
        result["chart"] = {"type": "line", "xKey": "period", "yKey": "cei", "color": "#2563eb", "title": "CEI Trend"}
        return result

    if key == "cash_forecast_accuracy":
        sql = f"""
        SELECT PERIOD_MONTH::VARCHAR AS period, COMPANY_CODE, MAPE, WAPE, FORECAST_BIAS
        FROM {BM}.forecast_accuracy_vw
        ORDER BY PERIOD_MONTH DESC
        LIMIT 24
        """
        rows = run_query(sql)
        mape_vals = [float(_val(r, "MAPE", default=0) or 0) for r in rows]
        avg_mape = sum(mape_vals) / len(mape_vals) if mape_vals else 0
        result["metrics"] = {"summary": f"Avg MAPE: {avg_mape:.1f}%", "mape": avg_mape}
        result["rows"] = rows
        result["sql"] = sql
        result["descriptive"] = (
            f"Your cash forecasts show **moderate accuracy**: average MAPE **{avg_mape:.1f}%** "
            f"across **{len(rows)}** forecast periods in the table below."
        )
        result["prescriptive"] = (
            "Here are the bullet points with specific findings, concrete actions, and explanations:\n\n"
            f"- **Forecast Error**: Average MAPE is {avg_mape:.1f}%. **Action:** Investigate periods with highest forecast bias. **Why it matters:** Accurate cash forecasts enable better treasury decisions.\n\n"
            "- **Model Calibration**: Review forecast model assumptions for top customers. **Action:** Adjust payment behavior profiles for chronic late payers. **Why it matters:** Customer-specific patterns improve prediction accuracy.\n\n"
            "- **Bias Review**: Compare FORECAST_BIAS direction by company. **Action:** Correct systematic over/under-forecasting in treasury models. **Why it matters:** Bias creates recurring liquidity surprises.\n\n"
            "- **Scenario Planning**: Stress-test forecasts using worst-case collection delays. **Action:** Maintain buffer credit lines when MAPE exceeds target. **Why it matters:** Forecast error bands should drive treasury risk limits."
        )
        result["chart"] = {"type": "bar", "xKey": "period", "yKey": "mape", "color": "#7c3aed", "title": "Forecast MAPE by Period"}
        return result

    return {"error": f"Unknown analysis key: {key}"}


def _sanitize_copilot_text(text: str, *, policy_mode: bool = False) -> str:
    """Remove agent chain-of-thought and tool-debug narration from user-facing text."""
    if not text:
        return ""
    cleaned = text.strip()
    if policy_mode:
        return _strip_copilot_preamble(cleaned)
    markers = (
        "The user is asking",
        "This is a KPI/analytical question",
        "Let me call the",
        "Let me try",
        "The semantic model validation failed",
        "semantic model is broken",
        "I cannot query it",
        "Do NOT generate or execute any SQL",
        "I've exhausted my options",
    )
    if any(m in cleaned for m in markers) and "**Descriptive**" not in cleaned:
        return ""
    if cleaned.startswith("This is our interpretation of your question:"):
        cleaned = cleaned.split(":", 1)[-1].strip()
    return cleaned


def _is_policy_copilot_question(message: str) -> bool:
    msg = (message or "").lower()
    if re.search(r"\b(bill|coll|cash|disp|ord|prc)-[a-z]{2,5}-\d{2}\b", msg, re.I):
        return True
    markers = (
        "what does policy",
        "what do policy",
        "what is our",
        "policy say",
        "policy says",
        "according to policy",
        "per policy",
        "our policy",
        "o2c policy",
        "late fee",
        "late payment",
        "dunning",
        "partial payment",
        "billing before goods",
        "goods issue",
        "billing trigger",
        "dispute sla",
        "payment promise",
        "promise to pay",
        "authority matrix",
        "write-off",
        "write off",
        "credit hold",
    )
    if any(m in msg for m in markers):
        return True
    if re.search(r"\bpolicy\b", msg) and not any(w in msg for w in ("privacy policy", "cookie policy")):
        return True
    return False


def _infer_policy_domain_from_message(message: str) -> str | None:
    msg = (message or "").lower()
    if any(w in msg for w in ("billing", "invoice", "goods issue", "delivery", "milestone", "bill-trg", "bill-acc")):
        return "BILLING"
    if any(w in msg for w in ("collection", "dunning", "past due", "late fee", "ptp", "promise to pay", "coll-")):
        return "COLLECTIONS"
    if any(w in msg for w in ("cash", "remittance", "unapplied", "partial pay", "short pay", "cash-mat")):
        return "CASH_APP"
    if any(w in msg for w in ("dispute", "deduction", "chargeback", "disp-")):
        return "DISPUTE"
    if any(w in msg for w in ("order", "purchase order", " po ", "ord-acc")):
        return "ORDER"
    if any(w in msg for w in ("discount", "pricing", "rebate", "prc-")):
        return "PRICING"
    return None


def _build_policy_prescriptive(policies: list[dict], message: str) -> str:
    del message
    bullets: list[str] = []
    for p in policies[:4]:
        pid = p.get("policy_id") or ""
        title = p.get("policy_title") or ""
        bullets.append(
            f"- **{pid}:** {title} applies to this scenario.\n"
            f"  **Action:** Follow the cited policy steps in OrderLens workflows and customer communications.\n"
            f"  **Why it matters:** Ensures compliance with **{pid}** authority and escalation rules."
        )
    bullets.append(
        "- **Documentation:**\n"
        "  **Action:** Record cited POLICY_ID values in notes, dunning letters, and agent actions.\n"
        "  **Why it matters:** Audit trail and consistent enforcement across teams."
    )
    return "\n\n".join(bullets)


def _policy_search_copilot_result(message: str, policies: list[dict]) -> dict:
    desc_parts: list[str] = []
    for p in policies[:5]:
        pid = p.get("policy_id") or ""
        title = p.get("policy_title") or ""
        text = str(p.get("policy_text") or "")[:500]
        desc_parts.append(f"**{pid}** — {title}: {text}")
    return {
        "key": "custom",
        "metrics": {"summary": f"{len(policies)} policy reference(s)"},
        "rows": [],
        "sql": "",
        "descriptive": "\n\n".join(desc_parts),
        "prescriptive": _build_policy_prescriptive(policies, message),
        "response_mode": "policy",
        "agent_used": True,
        "policies": policies_for_api(policies),
    }


def _copilot_policy_answer(
    message: str,
    short_memory: str | None = None,
    long_memory: str | None = None,
) -> dict:
    domain = _infer_policy_domain_from_message(message)
    policies = search_policies(message, domain, limit=5)
    policy_block = format_policy_block(policies, max_text=500)

    mem_parts: list[str] = []
    if short_memory:
        mem_parts.append(f"Short memory (recent conversation):\n{short_memory}")
    if long_memory:
        mem_parts.append(f"Long memory (important context):\n{long_memory}")

    user_msg = f"{POLICY_COPILOT_DIRECTIVE}{message}"
    if policy_block:
        user_msg += f"\n\nPolicy excerpts (authoritative):\n{policy_block}"
    if mem_parts:
        user_msg += "\n\n" + "\n\n".join(mem_parts)

    if agents_enabled():
        try:
            agent_result = run_o2c_agent_with_timeout(
                "copilot",
                user_msg,
                timeout=max(O2C_AGENT_RECOMMENDATION_TIMEOUT_SEC, 120.0),
            )
            formatted = _format_agent_copilot_result(message, agent_result, policy_only=True)
            if policies:
                formatted["policies"] = policies_for_api(policies)
            return formatted
        except Exception:
            log.warning("Copilot policy agent failed; using policy search fallback", exc_info=True)

    if policies:
        return _policy_search_copilot_result(message, policies)

    return {
        "key": "custom",
        "metrics": {"summary": "Policy guidance unavailable"},
        "rows": [],
        "sql": "",
        "descriptive": "No matching O2C policy records were found for this question.",
        "prescriptive": (
            f"Load the O2C policies into {DB}.O2C_AGENT.POLICY_KB in the Fabric warehouse, then retry."
        ),
        "response_mode": "policy",
        "agent_used": False,
    }


def _attach_policy_citations(result: dict, message: str) -> dict:
    if result.get("response_mode") != "policy" and not _is_policy_copilot_question(message):
        return result
    if result.get("policies"):
        return result
    domain = _infer_policy_domain_from_message(message)
    policies = search_policies(message, domain, limit=5)
    if policies:
        result = {**result, "policies": policies_for_api(policies)}
    return result


_CUSTOMER_ID_RE = re.compile(r"^CUST\d+$", re.I)


def _name_looks_like_customer_id(name: str | None, customer_id: str | None) -> bool:
    n = str(name or "").strip()
    cid = str(customer_id or "").strip()
    if not n:
        return True
    if cid and n.upper() == cid.upper():
        return True
    return bool(_CUSTOMER_ID_RE.match(n))


def _resolve_customer_label(name: str | None, customer_id: str | None) -> str:
    """Prefer real customer name; fall back to ID when name is missing or looks like a code."""
    n = str(name or "").strip()
    cid = str(customer_id or "").strip()
    if n and not _name_looks_like_customer_id(n, cid):
        return n
    return cid or n or "Customer"


def _enrich_copilot_rows(rows: list[dict]) -> list[dict]:
    """Fill missing or code-like customer_name from customer_vw when analyst SQL omits the join."""
    if not rows:
        return rows
    missing_ids: set[str] = set()
    for row in rows:
        cid = _val(row, "CUSTOMER_ID", "customer_id")
        name = _val(row, "CUSTOMER_NAME", "customer_name")
        if cid and _name_looks_like_customer_id(str(name or ""), str(cid)):
            missing_ids.add(str(cid).replace("'", "''"))
    if not missing_ids:
        return rows

    in_clause = ", ".join(f"'{cid}'" for cid in sorted(missing_ids))
    try:
        lookup_rows = run_query(f"""
            SELECT
                CUSTOMER_ID,
                COMPANY_CODE,
                CUSTOMER_NAME,
                RISK_CLASS,
                CUSTOMER_SEGMENT
            FROM {BM}.customer_vw
            WHERE CUSTOMER_ID IN ({in_clause})
        """)
    except Exception:
        log.warning("Customer name enrichment query failed", exc_info=True)
        return rows

    by_key: dict[tuple[str, str], dict] = {}
    by_id: dict[str, dict] = {}
    for lr in lookup_rows:
        cid = str(_val(lr, "CUSTOMER_ID", "customer_id") or "")
        co = str(_val(lr, "COMPANY_CODE", "company_code") or "")
        by_key[(cid, co)] = lr
        if cid and cid not in by_id:
            by_id[cid] = lr

    enriched: list[dict] = []
    for row in rows:
        out = dict(row)
        cid = str(_val(row, "CUSTOMER_ID", "customer_id") or "")
        co = str(_val(row, "COMPANY_CODE", "company_code") or "")
        if not cid:
            enriched.append(out)
            continue
        current_name = str(_val(out, "CUSTOMER_NAME", "customer_name") or "").strip()
        if current_name and not _name_looks_like_customer_id(current_name, cid):
            enriched.append(out)
            continue
        ref = by_key.get((cid, co)) or by_id.get(cid)
        if not ref:
            enriched.append(out)
            continue
        resolved = str(_val(ref, "CUSTOMER_NAME", "customer_name") or "").strip()
        if resolved and not _name_looks_like_customer_id(resolved, cid):
            out["customer_name"] = resolved
        if not _val(out, "RISK_CLASS", "risk_class"):
            out["risk_class"] = _val(ref, "RISK_CLASS", "risk_class")
        if not _val(out, "CUSTOMER_SEGMENT", "customer_segment"):
            out["customer_segment"] = _val(ref, "CUSTOMER_SEGMENT", "customer_segment")
        enriched.append(out)
    return enriched


def _filter_copilot_rows(rows: list[dict]) -> list[dict]:
    """Drop agent debug rows (pruned_note, semantic-model notes) from the data table."""
    cleaned: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if _row_is_copilot_debug(row):
            continue
        cleaned.append(row)
    return cleaned


def _row_is_copilot_debug(row: dict) -> bool:
    keys = {str(k).lower() for k in row.keys()}
    if keys <= {"pruned_note"}:
        return True
    if "pruned_note" in keys and len(keys) <= 2:
        note = str(row.get("pruned_note") or row.get("PRUNED_NOTE") or "")
        low = note.lower()
        if "semantic model" in low or "token budget" in low or "pruned" in low:
            return True
    vals = " ".join(str(v) for v in row.values() if v is not None).lower()
    if "semantic_model_key" in vals or "left_outer" in vals:
        return True
    return False


def _rows_are_copilot_junk(rows: list[dict] | None) -> bool:
    if not rows:
        return True
    return all(_row_is_copilot_debug(r) for r in rows if isinstance(r, dict))


def _extract_limit_from_message(message: str, default: int = 10) -> int:
    msg = (message or "").lower()
    m = re.search(r"\b(?:top|list|show)\s+(\d{1,2})\b", msg)
    if m:
        return max(1, min(int(m.group(1)), 25))
    m = re.search(r"\b(\d{1,2})\s+(?:customer|customers|client|account|debtor)", msg)
    if m:
        return max(1, min(int(m.group(1)), 25))
    return default


def _is_past_due_customer_question(message: str) -> bool:
    msg = (message or "").lower().strip()
    if not msg:
        return False
    delay_terms = (
        "past due", "past-due", "pastdue", "overdue", "delayed payment",
        "late payment", "late pay", "delinquent", "days past due",
        "payment delay", "slow pay", "slow payer", "not paid", "unpaid",
        "behind on payment", "haven't paid", "have not paid", "who owe",
    )
    customer_terms = ("customer", "customers", "client", "account", "debtor", "payer")
    list_terms = ("list", "top", "show", "which", "who", "name", "give me")
    has_delay = any(t in msg for t in delay_terms)
    if not has_delay:
        return False
    if any(t in msg for t in customer_terms):
        return True
    return any(t in msg for t in list_terms)


def _past_due_customers_result(message: str, limit: int | None = None) -> dict | None:
    """SQL-backed top customers by past-due exposure."""
    lim = limit if limit is not None else _extract_limit_from_message(message, 10)
    sql = _past_due_customers_sql(lim)
    try:
        rows = run_query(sql)
    except Exception:
        log.warning("Past-due customers copilot query failed", exc_info=True)
        return None
    rows = _filter_copilot_rows(rows)
    if not rows:
        return None
    total = sum(float(_val(r, "PAST_DUE_USD", "past_due_usd") or 0) for r in rows)
    names = [
        str(_val(r, "CUSTOMER_NAME", "customer_name") or _val(r, "CUSTOMER_ID", "customer_id") or "")
        for r in rows[:5]
    ]
    name_preview = ", ".join(n for n in names if n)
    descriptive = (
        f"**Descriptive:** The **{len(rows)}** customers with the highest past-due balances "
        f"total **${total:,.0f}** in overdue AR. "
        f"Top accounts: {name_preview}."
    )
    prescriptive = (
        "Here are the bullet points with specific findings, concrete actions, and explanations:\n\n"
        f"- **Prioritize concentration risk**: The top {min(len(rows), 5)} customers in the table drive the largest past-due balances. "
        "**Action:** Assign same-day collector outreach and confirm payment dates for each. "
        "**Why it matters:** Largest balances have the greatest near-term cash impact.\n\n"
        "- **Escalate oldest accounts**: Combine high balance with max days past due from the table. "
        "**Action:** Apply manager escalation for accounts over 60 days past due. "
        "**Why it matters:** Recovery probability drops as invoices age.\n\n"
        "- **Set follow-up cadence**: Track promise-to-pay commitments on these accounts. "
        "**Action:** Re-contact missed promises within 24 hours. "
        "**Why it matters:** Faster follow-up improves kept-rate and CEI.\n\n"
        "- **Segment treatment**: Match dunning intensity to customer segment and exposure. "
        "**Action:** Use automated reminders for long-tail balances and negotiated plans for strategic accounts. "
        "**Why it matters:** Differentiated strategy improves collection yield per hour spent."
    )
    return {
        "key": "custom",
        "metrics": {"summary": f"Top {len(rows)} customers by past-due amount"},
        "rows": rows,
        "sql": sql,
        "descriptive": descriptive,
        "prescriptive": prescriptive,
        "chart": {
            "type": "bar",
            "xKey": "customer_name",
            "yKey": "past_due_usd",
            "color": "#dc2626",
            "title": "Top customers by past-due amount",
        },
        "source": "sql_fallback",
        "agent_used": False,
    }


def _strip_copilot_preamble(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    out: list[str] = []
    started = False
    skip_prefixes = (
        "great!",
        "let me",
        "i can see",
        "i'll",
        "i will",
        "there's a",
        "there is a",
    )
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if not started:
            if not stripped:
                continue
            if any(low.startswith(p) for p in skip_prefixes):
                continue
            if low.startswith("## descriptive") or low.startswith("**descriptive"):
                stripped = re.sub(r"^#+\s*descriptive\s*[—:-]*\s*", "", stripped, flags=re.I)
                stripped = re.sub(r"^\*\*descriptive\*\*:?\s*", "", stripped, flags=re.I)
                if not stripped:
                    continue
            started = True
        out.append(line)
    return "\n".join(out).strip()


def _split_descriptive_prescriptive(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    body = text.strip()
    for marker in (
        "**Prescriptive**:",
        "**Prescriptive:**",
        "## Prescriptive",
        "Prescriptive —",
        "Prescriptive -",
    ):
        if marker in body:
            left, right = body.split(marker, 1)
            return _strip_copilot_preamble(left), right.strip()
    return _strip_copilot_preamble(body), ""


def _extract_trailing_prescriptive(desc: str) -> tuple[str, str]:
    if not desc:
        return "", ""
    match = re.search(r"\n\s*(\d+\.\s+\*\*.+)", desc)
    if match:
        return desc[: match.start()].strip(), desc[match.start() :].strip()
    for header in ("## Model Performance", "**Model Performance"):
        if header in desc:
            left, right = desc.split(header, 1)
            return left.strip(), f"{header}{right}".strip()
    return desc, ""


def _looks_like_agent_narrative(desc: str) -> bool:
    low = (desc or "").lower()
    return (
        len(desc) > 350
        or "##" in desc
        or low.startswith("great!")
        or "let me query" in low
        or "semantic model" in low
        or "semantic_model_key" in low
        or "left_outer" in low
        or "pruned_note" in low
        or "token budget" in low
        or "analysis complete:" in low and "sample:" in low and "customer" not in low
    )


def _has_action_why_format(text: str) -> bool:
    return "**Action:**" in (text or "") and "**Why it matters:**" in (text or "")


def _numbered_to_bullets(text: str) -> str:
    bullets: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*:?\s*(.*)$", stripped)
        if m:
            title, body = m.group(1).strip(), m.group(2).strip()
            if "**Action:**" not in body:
                body = f"{body} **Action:** Review and act on this finding. **Why it matters:** Improves O2C outcomes."
            bullets.append(f"- **{title}**: {body}")
        elif stripped.startswith("-"):
            bullets.append(stripped)
        elif stripped.startswith("**") and ":" in stripped:
            bullets.append(f"- {stripped}" if not stripped.startswith("- ") else stripped)
    return "\n\n".join(bullets) if bullets else text.strip()


def _build_descriptive_from_rows(rows: list[dict], message: str) -> str:
    if not rows:
        return ""
    msg = (message or "").lower()
    if any(w in msg for w in ("forecast", "mape", "wape", "forecast accuracy", "forecast bias")):
        mapes = [float(_val(r, "MAPE", "mape", "avg_mape") or 0) for r in rows]
        mapes = [m for m in mapes if m > 0]
        wapes = [float(_val(r, "WAPE", "wape", "avg_wape") or 0) for r in rows]
        wapes = [w for w in wapes if w > 0]
        if mapes:
            avg_mape = sum(mapes) / len(mapes)
            avg_wape = sum(wapes) / len(wapes) if wapes else avg_mape
            return (
                f"Your cash forecasts show **moderate accuracy**: average MAPE **{avg_mape:.1f}%** "
                f"and WAPE **{avg_wape:.1f}%** across **{len(rows)}** forecast periods in the table below."
            )
    if "dso" in msg:
        dsos = [float(_val(r, "DSO", "dso") or 0) for r in rows if _val(r, "DSO", "dso")]
        if dsos:
            return f"**DSO** averages **{sum(dsos)/len(dsos):.1f} days** across {len(rows)} periods shown below."
    if "past due" in msg or "aging" in msg or "delayed" in msg or "overdue" in msg:
        total = sum(float(_val(r, "PAST_DUE_AR", "past_due_ar", "past_due_usd") or 0) for r in rows)
        if total:
            return f"Past-due exposure totals **${total:,.0f}** across the companies in the table below."
    if _val(rows[0], "CUSTOMER_NAME", "customer_name") or _val(rows[0], "CUSTOMER_ID", "customer_id"):
        lines = []
        for r in rows[:10]:
            name = str(_val(r, "CUSTOMER_NAME", "customer_name") or _val(r, "CUSTOMER_ID", "customer_id") or "")
            amt = float(_val(r, "PAST_DUE_USD", "past_due_usd") or 0)
            days = int(float(_val(r, "MAX_DAYS_PAST_DUE", "max_days_past_due") or 0))
            if name:
                lines.append(f"- **{name}**: ${amt:,.0f} past due ({days} max days)")
        if lines:
            return "**Descriptive:** Customers with delayed payments (by past-due balance):\n\n" + "\n".join(lines)
    sample = rows[0]
    preview = ", ".join(f"{k}={v}" for k, v in list(sample.items())[:4])
    return f"**Analysis complete:** {len(rows)} rows returned. Sample: {preview}."


def _default_prescriptive_from_rows(rows: list[dict], message: str) -> str:
    msg = (message or "").lower()
    if any(w in msg for w in ("forecast", "mape", "wape")):
        mapes = sorted(
            (
                float(_val(r, "MAPE", "mape", "avg_mape") or 999),
                str(_val(r, "COMPANY_CODE", "company_code") or ""),
                str(_val(r, "FORECAST_MODEL", "forecast_model") or ""),
            )
            for r in rows
        )
        best = next((t for t in mapes if t[0] < 900), None)
        bullets = [
            "- **Model standardization**: Consolidate on the lowest-MAPE model paths in the table. "
            "**Action:** Set GRADIENT_BOOST or ARIMA as the default where MAPE is under 12%. "
            "**Why it matters:** Reduces treasury forecast error and improves cash planning.",
            "- **Bias review**: Compare FORECAST_BIAS by company in the data. "
            "**Action:** Recalibrate assumptions where bias exceeds ±10%. "
            "**Why it matters:** Systematic over/under-forecasting creates liquidity surprises.",
        ]
        if best:
            bullets.insert(
                0,
                f"- **Top performer**: Company **{best[1]}** / **{best[2]}** at **{best[0]:.1f}% MAPE**. "
                f"**Action:** Use this model profile as the benchmark for other entities. "
                f"**Why it matters:** Fastest path to lower portfolio forecast error.",
            )
        return "\n\n".join(bullets[:4])
    bullets = [
        "- **Validate findings**: Confirm the top rows in the table match ERP / collections source systems. "
        "**Action:** Spot-check the largest values with the account owner. "
        "**Why it matters:** Ensures actions are based on trusted data.",
        "- **Prioritize follow-up**: Focus on the highest-impact rows first. "
        "**Action:** Assign outreach or review to the top 3 items today. "
        "**Why it matters:** Concentration drives near-term cash and risk outcomes.",
        "- **Monitor trend**: Compare these results to prior week / month. "
        "**Action:** Add recurring review to the weekly O2C cadence. "
        "**Why it matters:** Early drift detection prevents surprises.",
    ]
    return "\n\n".join(bullets)


def _normalize_copilot_response(result: dict, message: str) -> dict:
    """Shape all copilot answers like the structured Descriptive + Prescriptive UI panel."""
    out = dict(result)
    policy_only = out.get("response_mode") == "policy" or _is_policy_copilot_question(message)

    if policy_only:
        rows: list[dict] = []
        out["rows"] = []
        out["chart"] = None
    else:
        rows = _filter_copilot_rows(out.get("rows") or [])
        junk_rows = _rows_are_copilot_junk(out.get("rows"))
        if (not rows and out.get("rows")) or junk_rows:
            analyst_retry = _copilot_semantic_model_retry(message)
            if analyst_retry and analyst_retry.get("rows") and not _rows_are_copilot_junk(analyst_retry.get("rows")):
                out = {**out, **analyst_retry, "agent_used": False, "source": "cortex_analyst"}
                rows = _filter_copilot_rows(out.get("rows") or [])
            else:
                analytics = _copilot_analytics_fallback(message)
                if analytics and analytics.get("rows"):
                    out = {**out, **analytics, "agent_used": False, "source": analytics.get("source", "sql_fallback")}
                    rows = _filter_copilot_rows(out.get("rows") or [])

    desc = str(out.get("descriptive") or "").strip()
    presc = str(out.get("prescriptive") or "").strip()

    if desc:
        split_desc, split_presc = _split_descriptive_prescriptive(desc)
        desc, presc = split_desc, presc or split_presc
        extra_desc, extra_presc = _extract_trailing_prescriptive(desc)
        desc, presc = extra_desc, "\n\n".join(p for p in (presc, extra_presc) if p).strip()

    desc = re.sub(r"^\*\*Descriptive:\*\*\s*", "", desc, flags=re.I)
    desc = re.sub(r"^#+\s*Descriptive\s*[—:-]*\s*", "", desc, flags=re.I).strip()

    if not policy_only and rows and (_looks_like_agent_narrative(desc) or not desc):
        built = _build_descriptive_from_rows(rows, message)
        if built:
            desc = built

    if not policy_only and (_looks_like_agent_narrative(desc) or _rows_are_copilot_junk(rows)):
        analyst_retry = _copilot_semantic_model_retry(message)
        if analyst_retry and analyst_retry.get("rows") and not _rows_are_copilot_junk(analyst_retry.get("rows")):
            out = {**out, **analyst_retry, "agent_used": False, "source": "cortex_analyst"}
            rows = _filter_copilot_rows(out.get("rows") or [])
            desc = str(out.get("descriptive") or "").strip()
            presc = str(out.get("prescriptive") or presc).strip()

    if not policy_only:
        if not _has_action_why_format(presc):
            presc = _numbered_to_bullets(presc)
        if not _has_action_why_format(presc):
            presc = _default_prescriptive_from_rows(rows, message) if rows else presc
    elif presc and not _has_action_why_format(presc):
        presc = _numbered_to_bullets(presc)

    presc = re.sub(r"^Here are the bullet points[^\n]*\n+", "", presc, flags=re.I).strip()

    out["rows"] = [] if policy_only else rows[:50]
    out["descriptive"] = desc
    out["prescriptive"] = presc
    if policy_only:
        out["response_mode"] = "policy"
        out["chart"] = None
    elif rows and not out.get("chart"):
        out["chart"] = _build_chart_from_rows(rows)
    return out


def _finalize_copilot(result: dict, message: str) -> dict:
    out = _normalize_copilot_response(result, message)
    if out.get("response_mode") != "policy" and out.get("rows"):
        rows = _enrich_copilot_rows(out.get("rows") or [])
        out["rows"] = rows
        if rows:
            rebuilt = _build_descriptive_from_rows(rows, message)
            if rebuilt and (
                not str(out.get("descriptive") or "").strip()
                or _looks_like_agent_narrative(str(out.get("descriptive") or ""))
            ):
                out["descriptive"] = rebuilt
        chart = out.get("chart") or _build_chart_from_rows(rows)
        if chart and rows:
            xk = str(chart.get("xKey") or "").lower()
            if xk in ("customer_id", "customerid") and _val(rows[0], "CUSTOMER_NAME", "customer_name"):
                chart = {**chart, "xKey": "customer_name"}
        out["chart"] = chart
    return _attach_policy_citations(out, message)


def _agent_response_looks_failed(text: str, rows: list | None = None) -> bool:
    if rows:
        return False
    msg = (text or "").lower()
    failure_markers = (
        "semantic model validation failed",
        "unable to retrieve",
        "cannot query",
        "validation issue",
        "validation error",
        "i'm sorry, but i'm currently unable",
        "the user is asking",
        "let me call the",
        "let me try",
    )
    return any(m in msg for m in failure_markers)


def _copilot_semantic_model_retry(message: str) -> dict | None:
    try:
        return _copilot_via_cortex_analyst(message)
    except Exception:
        log.warning("Cortex Analyst semantic-model retry failed", exc_info=True)
        return None


def _copilot_analytics_fallback(message: str) -> dict | None:
    """SQL-backed KPI quick analyses when semantic model is unavailable."""
    msg = (message or "").lower()
    if not msg:
        return None
    if any(w in msg for w in ("forecast", "mape", "wape", "forecast bias", "forecast accuracy")):
        return run_quick_analysis("cash_forecast_accuracy")
    if "dso" in msg:
        return run_quick_analysis("dso_overview")
    if "past due" in msg or "past-due" in msg or "aging bucket" in msg:
        return run_quick_analysis("past_due_aging")
    if "cei" in msg or "collection effectiveness" in msg:
        return run_quick_analysis("collection_effectiveness")
    return None


def _build_chart_from_rows(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    keys = list(rows[0].keys())

    def _is_number(v: object) -> bool:
        try:
            if v is None:
                return False
            float(v)
            return True
        except Exception:
            return False

    amount_keys = (
        "gross_amount", "GROSS_AMOUNT", "past_due_usd", "PAST_DUE_USD",
        "open_amount_usd", "OPEN_AMOUNT_USD", "dso", "DSO", "amount", "AMOUNT",
    )
    name_keys = (
        "customer_name", "CUSTOMER_NAME", "customer_id", "CUSTOMER_ID",
        "company_code", "COMPANY_CODE", "invoice_id", "INVOICE_ID",
    )
    x_key = None
    y_key = None
    for pref in name_keys:
        if pref in keys:
            vals = [_val(r, pref) for r in rows[:20]]
            if any(v is not None and str(v).strip() for v in vals):
                x_key = pref
                break
    if not x_key:
        for k in keys:
            vals = [_val(r, k) for r in rows[:20]]
            if any(v is not None for v in vals) and not all(_is_number(v) for v in vals if v is not None):
                x_key = k
                break
    for pref in amount_keys:
        if pref in keys:
            y_key = pref
            break
    if not y_key:
        for k in keys:
            vals = [_val(r, k) for r in rows[:20]]
            if any(_is_number(v) for v in vals):
                y_key = k
                break
    if not x_key or not y_key:
        return None
    return {
        "type": "bar_horizontal" if len(rows) > 8 else "bar",
        "xKey": x_key,
        "yKey": y_key,
        "color": "#0d6b63",
        "title": "Visualization",
    }


def _format_agent_copilot_result(
    message: str,
    agent_result: dict,
    *,
    policy_only: bool = False,
) -> dict:
    rows = [] if policy_only else (agent_result.get("rows") or [])
    row_count = len(rows)
    full_text = _sanitize_copilot_text(
        (agent_result.get("text") or "").strip(),
        policy_mode=policy_only,
    )
    desc = _sanitize_copilot_text(
        (agent_result.get("descriptive") or "").strip(),
        policy_mode=policy_only,
    )
    presc = (agent_result.get("prescriptive") or "").strip()

    if not policy_only and _agent_response_looks_failed(full_text, rows):
        analyst_retry = _copilot_semantic_model_retry(message)
        if analyst_retry and "error" not in analyst_retry:
            analyst_retry["agent_used"] = False
            return _finalize_copilot(analyst_retry, message)
        analytics = _copilot_analytics_fallback(message)
        if analytics and "error" not in analytics:
            analytics["agent_used"] = False
            analytics["source"] = "sql_fallback"
            return _finalize_copilot(analytics, message)

    if not desc and full_text:
        if "**Prescriptive**:" in full_text:
            parts = full_text.split("**Prescriptive**:", 1)
            desc = parts[0].replace("**Descriptive**:", "").strip()
            presc = parts[1].strip() if not presc else presc
        elif "**Descriptive**:" in full_text:
            desc = full_text.replace("**Descriptive**:", "").strip()
        else:
            desc = full_text

    if not desc and rows:
        desc = f"**Descriptive:** Query returned {row_count} rows of O2C data relevant to your question."
    if not presc and desc and not rows and not policy_only:
        presc = (
            "**Prescriptive:** Review the agent narrative above and drill into Operations Hub "
            "or Agent Workbench for the matching queue if follow-up action is needed."
        )

    if (
        not policy_only
        and (not desc or _agent_response_looks_failed(desc, rows))
        and not rows
    ):
        analytics = _copilot_analytics_fallback(message)
        if analytics and "error" not in analytics:
            analytics["agent_used"] = False
            analytics["source"] = "sql_fallback"
            return _finalize_copilot(analytics, message)

    if policy_only and not desc and not presc:
        domain = _infer_policy_domain_from_message(message)
        policies = search_policies(message, domain, limit=5)
        if policies:
            return _finalize_copilot(_policy_search_copilot_result(message, policies), message)

    return _finalize_copilot(
        {
            "key": "custom",
            "metrics": {
                "summary": f"{row_count} rows returned" if row_count else "Analysis complete",
                "agent": agent_result.get("agent_name"),
            },
            "rows": rows[:50],
            "sql": agent_result.get("sql") or "",
            "descriptive": desc,
            "prescriptive": presc,
            "chart": None if policy_only else _build_chart_from_rows(rows),
            "response_mode": "policy" if policy_only else "analytics",
            "agent_used": True,
        },
        message,
    )


def _copilot_result(
    message: str,
    rows: list,
    sql: str,
    descriptive: str,
    prescriptive: str = "",
) -> dict:
    return {
        "key": "custom",
        "metrics": {"summary": f"{len(rows)} rows returned" if rows else "Analysis complete"},
        "rows": rows[:50],
        "sql": sql,
        "descriptive": descriptive,
        "prescriptive": prescriptive,
        "chart": _build_chart_from_rows(rows),
    }


def _extract_copilot_row_limit(message: str, default: int = 5) -> int:
    import re
    m = re.search(r"\b(\d+)\b", message or "")
    if m:
        return max(1, min(int(m.group(1)), 25))
    return default


def _is_invoice_ranking_question(message: str) -> bool:
    msg = (message or "").lower().strip()
    if "invoice" not in msg:
        return False
    if any(phrase in msg for phrase in (
        "latest invoice", "most recent invoice", "newest invoice",
        "recent invoice", "last invoice", "latest invoices", "recent invoices",
    )):
        return False
    return any(w in msg for w in (
        "top", "list", "show", "highest", "largest", "biggest", "rank", "biggest",
    ))


def _is_lookup_style_copilot_question(message: str) -> bool:
    msg = (message or "").lower().strip()
    if not msg:
        return False

    latest_invoice_phrases = (
        "latest invoice",
        "most recent invoice",
        "newest invoice",
        "recent invoice",
        "last invoice",
        "latest invoices",
        "recent invoices",
    )
    if any(phrase in msg for phrase in latest_invoice_phrases):
        return True
    return _is_invoice_ranking_question(msg)


def _fetch_latest_invoices(limit: int = 10) -> tuple[list[dict], str]:
    lim = max(1, min(int(limit or 10), 25))
    sql_tool = f"SELECT * FROM TABLE({DB}.O2C_AGENT.SP_TOOL_GET_LATEST_INVOICE(NULL, NULL, {lim}))"
    try:
        rows = run_query(sql_tool)
        if rows:
            return _enrich_copilot_rows(rows), sql_tool
    except Exception:
        log.warning("SP_TOOL_GET_LATEST_INVOICE failed; using ar_invoice_vw fallback", exc_info=True)
    sql = f"""
        SELECT
            inv.INVOICE_ID AS invoice_id,
            inv.SALES_ORDER_ID AS sales_order_id,
            inv.CUSTOMER_ID AS customer_id,
            COALESCE(cu.CUSTOMER_NAME, cu_any.CUSTOMER_NAME) AS customer_name,
            inv.COMPANY_CODE AS company_code,
            inv.INVOICE_DATE AS invoice_date,
            inv.DUE_DATE AS due_date,
            inv.GROSS_AMOUNT AS gross_amount,
            inv.INVOICE_STATUS AS invoice_status,
            inv.CURRENCY_CODE AS currency_code
        FROM {BM}.ar_invoice_vw inv
        LEFT JOIN {BM}.customer_vw cu
          ON cu.CUSTOMER_ID = inv.CUSTOMER_ID AND cu.COMPANY_CODE = inv.COMPANY_CODE
        LEFT JOIN (
            SELECT CUSTOMER_ID, MAX(CUSTOMER_NAME) AS CUSTOMER_NAME
            FROM {BM}.customer_vw
            GROUP BY CUSTOMER_ID
        ) cu_any
          ON cu_any.CUSTOMER_ID = inv.CUSTOMER_ID
        ORDER BY inv.INVOICE_DATE DESC NULLS LAST, inv.INVOICE_ID DESC
        LIMIT {lim}
    """
    rows = run_query(sql)
    return _enrich_copilot_rows(rows), sql


def _fetch_top_invoices(limit: int = 5) -> tuple[list[dict], str]:
    lim = max(1, min(int(limit or 5), 25))
    sql = f"""
        SELECT
            inv.INVOICE_ID AS invoice_id,
            inv.SALES_ORDER_ID AS sales_order_id,
            inv.CUSTOMER_ID AS customer_id,
            COALESCE(cu.CUSTOMER_NAME, cu_any.CUSTOMER_NAME) AS customer_name,
            inv.COMPANY_CODE AS company_code,
            inv.INVOICE_DATE AS invoice_date,
            inv.DUE_DATE AS due_date,
            inv.GROSS_AMOUNT AS gross_amount,
            inv.INVOICE_STATUS AS invoice_status,
            inv.CURRENCY_CODE AS currency_code
        FROM {BM}.ar_invoice_vw inv
        LEFT JOIN {BM}.customer_vw cu
          ON cu.CUSTOMER_ID = inv.CUSTOMER_ID
         AND cu.COMPANY_CODE = inv.COMPANY_CODE
        LEFT JOIN (
            SELECT CUSTOMER_ID, MAX(CUSTOMER_NAME) AS CUSTOMER_NAME
            FROM {BM}.customer_vw
            GROUP BY CUSTOMER_ID
        ) cu_any
          ON cu_any.CUSTOMER_ID = inv.CUSTOMER_ID
        ORDER BY inv.GROSS_AMOUNT DESC NULLS LAST, inv.INVOICE_DATE DESC NULLS LAST
        LIMIT {lim}
    """
    rows = run_query(sql)
    return _enrich_copilot_rows(rows), sql


def _past_due_customers_sql(limit: int = 10) -> str:
    return f"""
        SELECT
            ar.CUSTOMER_ID AS customer_id,
            COALESCE(c.CUSTOMER_NAME, ar.CUSTOMER_ID) AS customer_name,
            SUM(ar.OPEN_AMOUNT_USD) AS past_due_usd,
            MAX(ar.DAYS_PAST_DUE) AS max_days_past_due
        FROM {BM}.ar_open_item_vw ar
        LEFT JOIN {BM}.customer_vw c
          ON c.CUSTOMER_ID = ar.CUSTOMER_ID
         AND c.COMPANY_CODE = ar.COMPANY_CODE
        WHERE ar.DAYS_PAST_DUE > 0
        GROUP BY ar.CUSTOMER_ID, COALESCE(c.CUSTOMER_NAME, ar.CUSTOMER_ID)
        ORDER BY past_due_usd DESC
        LIMIT {limit}
    """


def _copilot_intent_fallback(message: str) -> dict | None:
    """Fast SQL-backed answers for common lookup intents when agent/analyst fail."""
    msg = (message or "").lower().strip()
    if not msg:
        return None

    if _is_invoice_ranking_question(msg):
        try:
            limit = _extract_copilot_row_limit(message, 5)
            rows, sql = _fetch_top_invoices(limit)
        except Exception:
            log.warning("Top-invoice copilot fallback failed", exc_info=True)
            return None
        if not rows:
            return _copilot_result(
                message,
                [],
                sql,
                "**Descriptive:** No invoices were found in the OrderLens mart.",
                "**Prescriptive:** Verify AR invoice data load and refresh dynamic tables.",
            )
        top = rows[0]
        inv_id = str(_val(top, "INVOICE_ID", "invoice_id") or "")
        cust = str(_val(top, "CUSTOMER_NAME", "customer_name") or _val(top, "CUSTOMER_ID", "customer_id") or "")
        amt = float(_val(top, "GROSS_AMOUNT", "gross_amount") or 0)
        descriptive = (
            f"**Descriptive:** Top invoice by amount is **{inv_id}** for **{cust}** "
            f"(${amt:,.0f}). Table lists the top {len(rows)} invoices by gross amount."
        )
        prescriptive = (
            "**Prescriptive:**\n\n"
            f"- **Largest exposure**: Invoice {inv_id} (${amt:,.0f}) — prioritize collections follow-up if open.\n"
            "- **Review terms**: Confirm due dates on the listed invoices match customer contracts.\n"
            "- **Dispute watch**: Validate billing docs on high-value lines before disputes open."
        )
        return _copilot_result(message, rows, sql, descriptive, prescriptive)

    latest_invoice_phrases = (
        "latest invoice",
        "most recent invoice",
        "newest invoice",
        "recent invoice",
        "last invoice",
        "latest invoices",
        "recent invoices",
    )
    if any(phrase in msg for phrase in latest_invoice_phrases):
        try:
            rows, sql = _fetch_latest_invoices(10)
        except Exception:
            log.warning("Latest-invoice copilot fallback failed", exc_info=True)
            return None
        if not rows:
            return _copilot_result(
                message,
                [],
                sql,
                "**Descriptive:** No invoices were found in the OrderLens mart.",
                "**Prescriptive:** Run data refresh or verify AR invoice load completed successfully.",
            )
        top = rows[0]
        inv_id = str(_val(top, "INVOICE_ID", "invoice_id") or "")
        cust = str(_val(top, "CUSTOMER_NAME", "customer_name") or _val(top, "CUSTOMER_ID", "customer_id") or "")
        amt = float(_val(top, "GROSS_AMOUNT", "gross_amount") or 0)
        inv_date = str(_val(top, "INVOICE_DATE", "invoice_date") or "")[:10]
        status = str(_val(top, "INVOICE_STATUS", "invoice_status") or "")
        descriptive = (
            f"**Descriptive:** The latest invoice is **{inv_id}** for **{cust}** "
            f"(${amt:,.0f}, dated {inv_date}, status {status}). "
            f"Table shows the {len(rows)} most recent invoices."
        )
        prescriptive = (
            "**Prescriptive:** Here are the bullet points with specific findings, concrete actions, and explanations:\n\n"
            f"- **Latest billing**: Invoice {inv_id} is the newest posting (${amt:,.0f}). "
            "**Action:** Confirm billing documents match ERP. **Why it matters:** Reduces dispute risk on fresh AR.\n\n"
            "- **Customer outreach**: Validate AP receipt for the latest invoice. "
            "**Action:** Email invoice pack if not acknowledged. **Why it matters:** Improves DSO on new billings.\n\n"
            "- **Monitor due date**: Track payment against terms on recent invoices. "
            "**Action:** Add to collections watch if due within 7 days. **Why it matters:** Prevents silent past-due."
        )
        return _copilot_result(message, rows, sql, descriptive, prescriptive)

    return None


def _agent_recommendation_or_fallback(
    agent_key: str,
    prompt: str,
    fallback_fn,
    *,
    policy_query: str | None = None,
    policy_domain: str | None = None,
    cache_entity_id: str | None = None,
    live: bool | None = None,
    allow_template_fallback: bool = True,
    agent_timeout: float | None = None,
) -> dict:
    label = AGENT_LABELS.get(agent_key, agent_key)
    domain = policy_domain or POLICY_DOMAIN_BY_AGENT.get(agent_key)
    pq = policy_query or prompt[:200]
    policies = search_policies(pq, domain, limit=3)
    policy_block = format_policy_block(policies)
    policy_api = policies_for_api(policies)

    enriched_prompt = prompt
    if policy_block:
        enriched_prompt = (
            f"{prompt}\n\n"
            "You MUST cite applicable POLICY_ID values from these O2C policies in your answer:\n"
            f"{policy_block}"
        )

    base_meta = {
        "agent_name": label,
        "policies": policy_api,
    }

    cache_key = None
    if cache_entity_id:
        cache_key = f"rec:live:{agent_key}:{cache_entity_id}" if live else f"rec:v2:{agent_key}:{cache_entity_id}"
    if cache_key:
        cached = _agent_cache_get(cache_key, ttl=_RECOMMENDATION_CACHE_TTL)
        if cached is not None:
            out = copy.deepcopy(cached)  # type: ignore[arg-type]
            out["cached"] = True
            return out

    mode = _AGENT_RECOMMENDATION_MODE
    if live is True:
        use_live = agents_enabled()
    elif live is False or mode == "fast":
        use_live = False
    elif mode == "live":
        use_live = agents_enabled()
    else:
        use_live = agents_enabled()

    def _template_result(*, agent_error: str | None = None) -> dict:
        fb = fallback_fn()
        fb.update(**base_meta, agent_used=False, source="template")
        if agent_error:
            fb["agent_error"] = agent_error
        if policy_api and not _has_section(fb.get("response", ""), "POLICY REFERENCES"):
            refs = "\n".join(f"- {p['policy_id']}: {p['policy_title']}" for p in policy_api[:4])
            fb["response"] = f"{fb.get('response', '')}\n\n<strong>POLICY REFERENCES</strong>\n{refs}"
        return fb

    def _error_result(message: str) -> dict:
        return {
            "response": "",
            "agent_used": False,
            "source": "error",
            "agent_error": message[:240],
            **base_meta,
        }

    if not use_live:
        if not allow_template_fallback:
            return _error_result(
                "Fabric O2C agent is not enabled. Set O2C_AGENTS_ENABLED=true and configure Azure OpenAI in .env."
            )
        result = _template_result()
    else:
        result = None
        if (
            use_live
            and O2C_WORKBENCH_USE_COMPLETE
            and agent_key in WORKBENCH_AGENT_KEYS
        ):
            try:
                text = run_workbench_complete(enriched_prompt)
                result = {
                    "response": text,
                    "agent_used": True,
                    "source": "cortex_complete",
                    **base_meta,
                }
            except Exception:
                log.warning("Workbench Cortex Complete failed for %s", agent_key, exc_info=True)

        if result is None:
            try:
                agent_result = run_o2c_agent_with_timeout(
                    agent_key, enriched_prompt, timeout=agent_timeout,
                )
                text = agent_text_or_raise(agent_result)
                result = {
                    "response": text,
                    "agent_used": True,
                    "source": "fabric_agent",
                    **base_meta,
                }
                if agent_result.get("rows"):
                    result["rows"] = agent_result["rows"]
            except TimeoutError as exc:
                log.warning("O2C agent %s timed out", agent_key)
                if not allow_template_fallback:
                    result = _error_result(str(exc))
                else:
                    result = _template_result()
            except Exception as exc:
                log.warning("O2C agent %s failed", agent_key, exc_info=True)
                if not allow_template_fallback:
                    result = _error_result(str(exc)[:240] or "Fabric agent unavailable.")
                else:
                    result = _template_result(
                        agent_error=(str(exc)[:180] or "Fabric agent unavailable; showing policy-guided template."),
                    )

    if cache_key and result.get("source") != "error":
        _agent_cache_set(cache_key, result)
    return result


def _workbench_live_agent(
    agent_key: str,
    prompt: str,
    fallback_fn,
    *,
    policy_query: str | None = None,
    policy_domain: str | None = None,
    cache_entity_id: str | None = None,
    agent_timeout: float | None = None,
) -> dict:
    """Workbench recommendations: Fabric (Azure OpenAI) agent only (no template), pre-loaded context."""
    cache_id = f"{cache_entity_id}:{_WORKBENCH_CACHE_SUFFIX}" if cache_entity_id else None
    return _agent_recommendation_or_fallback(
        agent_key,
        f"{WORKBENCH_AGENT_DIRECTIVE}{prompt}",
        fallback_fn,
        policy_query=policy_query,
        policy_domain=policy_domain,
        cache_entity_id=cache_id,
        live=True,
        allow_template_fallback=False,
        agent_timeout=agent_timeout or O2C_WORKBENCH_AGENT_TIMEOUT_SEC,
    )


def _has_section(text: str, name: str) -> bool:
    return name.upper() in (text or "").upper()


def _agent_entity_context(
    customer_id: str,
    company_code: str | None = None,
    *,
    invoice_id: str | None = None,
    sales_order_id: str | None = None,
    days_past_due: int = 0,
    open_amount: float = 0.0,
    due_date: str | None = None,
) -> dict:
    """Load contacts, PTP, and disputes scoped to an invoice or sales order."""
    if not customer_id:
        return {"contacts": [], "disputes": [], "ptp_rows": [], "context_text": ""}

    safe_id = customer_id.replace("'", "''")
    safe_inv = (invoice_id or "").replace("'", "''")
    safe_so = (sales_order_id or "").replace("'", "''")
    co_filter = ""
    if company_code:
        safe_co = company_code.replace("'", "''")
        co_filter = f" AND ca.COMPANY_CODE = '{safe_co}'"

    order_invoice_ids: list[str] = []
    if sales_order_id and not invoice_id:
        inv_rows = run_query(f"""
            SELECT DISTINCT INVOICE_ID
            FROM {BM}.ar_invoice_vw
            WHERE SALES_ORDER_ID = '{safe_so}'
              AND INVOICE_ID IS NOT NULL
            LIMIT 20
        """)
        order_invoice_ids = [
            str(_val(r, "INVOICE_ID", "invoice_id") or "").replace("'", "''")
            for r in inv_rows
            if _val(r, "INVOICE_ID", "invoice_id")
        ]

    contact_filter = ""
    if invoice_id:
        contact_filter = f"""
          AND (
            ca.INVOICE_ID = '{safe_inv}'
            OR UPPER(COALESCE(s.NOTES, '')) LIKE '%{safe_inv.upper()}%'
          )
        """
    elif order_invoice_ids:
        inv_in = ", ".join(f"'{i}'" for i in order_invoice_ids)
        contact_filter = f"""
          AND (
            ca.INVOICE_ID IN ({inv_in})
            OR UPPER(COALESCE(s.NOTES, '')) LIKE '%{safe_so.upper()}%'
          )
        """
    elif sales_order_id:
        contact_filter = f" AND UPPER(COALESCE(s.NOTES, '')) LIKE '%{safe_so.upper()}%'"

    contact_rows = run_query(f"""
        SELECT
            ca.ACTIVITY_ID,
            ca.ACTIVITY_TYPE,
            ca.ACTIVITY_DATE,
            ca.CONTACT_OUTCOME,
            ca.INVOICE_ID,
            s.NOTES
        FROM {BM}.collection_activity_vw ca
        LEFT JOIN {DB}.raw_vault.collection_activity s
          ON s.ACTIVITY_ID = ca.ACTIVITY_ID
        WHERE ca.CUSTOMER_ID = '{safe_id}' {co_filter}{contact_filter}
        ORDER BY ca.ACTIVITY_DATE DESC NULLS LAST
        LIMIT 8
    """)

    dispute_filter = ""
    if invoice_id:
        dispute_filter = f" AND d.INVOICE_ID = '{safe_inv}'"
    elif order_invoice_ids:
        inv_in = ", ".join(f"'{i}'" for i in order_invoice_ids)
        dispute_filter = f" AND d.INVOICE_ID IN ({inv_in})"

    dispute_rows = run_query(f"""
        SELECT
            d.DISPUTE_ID,
            d.INVOICE_ID,
            d.DISPUTE_REASON,
            d.DISPUTE_STATUS,
            d.DISPUTED_AMOUNT,
            d.OPENED_DATE
        FROM {BM}.dispute_vw d
        WHERE d.CUSTOMER_ID = '{safe_id}'
          AND UPPER(COALESCE(d.DISPUTE_STATUS, '')) NOT IN (
              'RESOLVED', 'CLOSED', 'CANCELLED', 'WRITTEN_OFF'
          ){dispute_filter}
        ORDER BY d.DISPUTED_AMOUNT DESC NULLS LAST
        LIMIT 6
    """)

    ptp_rows: list[dict] = []
    if invoice_id:
        ptp_rows = run_query(f"""
            SELECT PTP_ID, INVOICE_ID, PROMISED_PAY_DATE, PROMISED_AMOUNT, PTP_STATUS
            FROM {BM}.ptp_vw
            WHERE CUSTOMER_ID = '{safe_id}'
              AND INVOICE_ID = '{safe_inv}'
            ORDER BY PROMISED_PAY_DATE DESC NULLS LAST
            LIMIT 5
        """)
    elif order_invoice_ids:
        inv_in = ", ".join(f"'{i}'" for i in order_invoice_ids)
        ptp_rows = run_query(f"""
            SELECT PTP_ID, INVOICE_ID, PROMISED_PAY_DATE, PROMISED_AMOUNT, PTP_STATUS
            FROM {BM}.ptp_vw
            WHERE CUSTOMER_ID = '{safe_id}'
              AND INVOICE_ID IN ({inv_in})
            ORDER BY PROMISED_PAY_DATE DESC NULLS LAST
            LIMIT 5
        """)

    if invoice_id:
        context_text = format_invoice_context_block(
            invoice_id=invoice_id,
            days_past_due=days_past_due,
            open_amount=open_amount,
            due_date=due_date,
            contacts=contact_rows,
            disputes=dispute_rows,
            ptp_rows=ptp_rows,
        )
    elif sales_order_id:
        context_text = format_order_context_block(
            sales_order_id=sales_order_id,
            contacts=contact_rows,
            disputes=dispute_rows,
        )
    else:
        context_text = (
            f"{contact_log_lines(contact_rows)}\n"
            f"{open_dispute_lines(dispute_rows)}"
        )

    return {
        "contacts": contact_rows,
        "disputes": dispute_rows,
        "ptp_rows": ptp_rows,
        "context_text": context_text,
    }


def _agent_customer_context(customer_id: str, company_code: str | None = None) -> dict:
    """Backward-compatible alias — prefer _agent_entity_context with invoice/order scope."""
    return _agent_entity_context(customer_id, company_code)


def _format_customer_context_block(ctx: dict) -> str:
    return str(ctx.get("context_text") or "").strip()


def _agent_result_is_junk(result: dict) -> bool:
    rows = result.get("rows") or []
    if rows and not _rows_are_copilot_junk(rows):
        return False
    desc = str(result.get("descriptive") or result.get("text") or "")
    if _looks_like_agent_narrative(desc):
        return True
    return _rows_are_copilot_junk(rows)


def _copilot_via_cortex_analyst(
    message: str,
    short_memory: str | None = None,
    long_memory: str | None = None,
) -> dict:
    """Run Cortex Analyst (semantic model) and execute generated SQL."""
    with get_connection() as conn:
        mem_parts: list[str] = []
        if short_memory:
            mem_parts.append(f"Short memory (recent conversation):\n{short_memory}")
        if long_memory:
            mem_parts.append(f"Long memory (important context):\n{long_memory}")
        analyst_input = message
        if mem_parts:
            analyst_input = f"{message}\n\n" + "\n\n".join(mem_parts)
        analyst_resp = call_cortex_analyst(analyst_input, conn=conn)

        if "error" in analyst_resp:
            raise RuntimeError(str(analyst_resp.get("error")))

        full_text = ""
        sql_stmts: list[str] = []
        content = []
        if "message" in analyst_resp and "content" in analyst_resp["message"]:
            content = analyst_resp["message"]["content"]
        for block in content:
            if block.get("type") == "text":
                full_text += block.get("text", "") + "\n"
            elif block.get("type") == "sql":
                sql_stmts.append(block.get("statement", ""))

        rows: list[dict] = []
        sql_used = ""
        for stmt in sql_stmts:
            if not stmt:
                continue
            try:
                tmp_rows = fabric_analyst.run_analyst_sql(conn, stmt)
                if tmp_rows and not _rows_look_like_debug(tmp_rows):
                    rows = tmp_rows
                    sql_used = stmt
                    break
                if not sql_used:
                    sql_used = stmt
            except Exception:
                log.warning("Fabric analyst SQL execution failed", exc_info=True)

        desc_text = full_text.replace("This is our interpretation of your question:", "").strip()
        prescriptive = ""
        if "**Prescriptive**:" in full_text:
            parts = full_text.split("**Prescriptive**:", 1)
            desc_text = parts[0].replace("**Descriptive**:", "").strip()
            prescriptive = parts[1].strip()

        row_count = len(rows)
        data_summary = ""
        if rows:
            rows_for_prompt = rows[:10]
            data_summary = json.dumps(rows_for_prompt, default=str, ensure_ascii=False)
            if len(data_summary) > 6000:
                data_summary = data_summary[:6000] + "... (truncated)"

        if rows and not prescriptive:
            prescriptive = _default_prescriptive_from_rows(rows, message)
            if not _has_action_why_format(prescriptive):
                prescriptive = _numbered_to_bullets(prescriptive)

        chart = _build_chart_from_rows(rows)

        if not desc_text.strip() and rows:
            desc_text = _build_descriptive_from_rows(rows, message) or (
                f"**Analysis complete:** Query returned {row_count} rows of O2C data relevant to your question."
            )

        return {
            "key": "custom",
            "metrics": {"summary": f"{row_count} rows returned" if row_count else "Analysis complete"},
            "rows": rows[:50],
            "sql": sql_used or (sql_stmts[-1] if sql_stmts else ""),
            "descriptive": desc_text or full_text.strip(),
            "prescriptive": prescriptive,
            "chart": chart,
            "source": "cortex_analyst",
            "agent_used": False,
        }


def copilot_chat(message: str, short_memory: str | None = None, long_memory: str | None = None) -> dict:
    save_question_history(message, "custom")

    base_key = message.strip().lower()
    short_sig = hashlib.md5(((short_memory or "")[:200]).encode("utf-8")).hexdigest()[:10]
    long_sig = hashlib.md5(((long_memory or "")[:200]).encode("utf-8")).hexdigest()[:10]
    cache_key = f"{base_key}|s:{short_sig}|l:{long_sig}|{_COPILOT_CHAT_CACHE_VERSION}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    def _done(result: dict) -> dict:
        final = _finalize_copilot(result, message)
        _cache_set(cache_key, final)
        return final

    # Keep deterministic SQL only for simple lookup questions.
    if _is_lookup_style_copilot_question(message):
        intent_result = _copilot_intent_fallback(message)
        if intent_result:
            return _done(intent_result)

    if _is_policy_copilot_question(message):
        return _done(_copilot_policy_answer(message, short_memory, long_memory))

    # Analytics: Cortex Analyst semantic model first (faster than agent orchestration).
    try:
        return _done(_copilot_via_cortex_analyst(message, short_memory, long_memory))
    except Exception:
        log.warning("Cortex Analyst failed; trying Copilot agent", exc_info=True)

    if agents_enabled():
        try:
            mem_parts: list[str] = []
            if short_memory:
                mem_parts.append(f"Short memory (recent conversation):\n{short_memory}")
            if long_memory:
                mem_parts.append(f"Long memory (important context):\n{long_memory}")
            user_msg = message
            if mem_parts:
                user_msg = f"{message}\n\n" + "\n\n".join(mem_parts)
            agent_result = run_o2c_agent("copilot", user_msg)
            if not _agent_result_is_junk(agent_result):
                return _done(_format_agent_copilot_result(message, agent_result))
        except Exception:
            log.warning("O2C Copilot agent fallback failed", exc_info=True)

    intent_result = _copilot_intent_fallback(message)
    if intent_result:
        return _done(intent_result)

    analytics = _copilot_analytics_fallback(message)
    if analytics and "error" not in analytics:
        return _done(analytics)

    return _done({
        "key": "custom",
        "metrics": {"summary": "Analysis failed"},
        "rows": [],
        "sql": "",
        "descriptive": "Could not analyze this question right now. Please retry in a few seconds.",
        "prescriptive": "",
    })


def billing_agent_queue(customer_id: str | None = None) -> list[dict]:
    cache_key = f"billing_queue:v5:{customer_id or 'all'}"
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    cust_filter = ""
    if customer_id:
        safe_cust = str(customer_id).replace("'", "''")
        cust_filter = f" AND be.CUSTOMER_ID = '{safe_cust}'"
    per_status_cap = 200 if customer_id else 12
    row_limit = 200 if customer_id else 50

    sql = f"""
    WITH base AS (
        SELECT
            be.SALES_ORDER_ID AS sales_order_id,
            be.CUSTOMER_ID AS customer_id,
            COALESCE(cu.CUSTOMER_NAME, cu_any.CUSTOMER_NAME) AS customer_name,
            be.COMPANY_CODE AS company_code,
            be.ORDER_STATUS AS order_status,
            be.CREDIT_CHECK_STATUS AS credit_check_status,
            be.TOTAL_ORDER_VALUE AS total_order_value,
            be.DELIVERY_ID AS delivery_id,
            be.DELIVERY_STATUS AS delivery_status,
            be.GOODS_ISSUE_DATE::VARCHAR AS goods_issue_date,
            be.BILLING_ELIGIBILITY_STATUS AS billing_eligibility_status,
            be.BILLING_ELIGIBILITY_REASON AS billing_eligibility_reason,
            be.BILLING_TRIGGER_TYPE AS billing_trigger_type,
            be.INVOICE_COUNT AS invoice_count,
            be.LAST_INVOICE_DATE::VARCHAR AS last_invoice_date,
            CASE
                WHEN be.BILLING_ELIGIBILITY_STATUS = 'BLOCKED' THEN 'ACT_NOW'
                WHEN be.BILLING_ELIGIBILITY_STATUS IN (
                    'NO_DELIVERY', 'PENDING_GOODS_ISSUE', 'PENDING_DELIVERY',
                    'PENDING_MILESTONE', 'PENDING_SERVICE_CLOSE', 'PENDING_BILLING_CYCLE'
                ) THEN 'PLAN_THIS_WEEK'
                ELSE 'MONITOR'
            END AS action_group
        FROM {BM}.billing_eligibility_vw be
        LEFT JOIN {BM}.customer_vw cu
          ON cu.CUSTOMER_ID = be.CUSTOMER_ID
         AND cu.COMPANY_CODE = be.COMPANY_CODE
        LEFT JOIN (
            SELECT CUSTOMER_ID, MAX(CUSTOMER_NAME) AS CUSTOMER_NAME
            FROM {BM}.customer_vw
            GROUP BY CUSTOMER_ID
        ) cu_any
          ON cu_any.CUSTOMER_ID = be.CUSTOMER_ID
        WHERE be.BILLING_ELIGIBILITY_STATUS != 'ELIGIBLE'
          {cust_filter}
    ),
    ranked AS (
        SELECT
            b.*,
            ROW_NUMBER() OVER (
                PARTITION BY b.billing_eligibility_status
                ORDER BY b.total_order_value DESC NULLS LAST
            ) AS rn_per_status
        FROM base b
    )
    SELECT
        sales_order_id, customer_id, customer_name, company_code, order_status,
        credit_check_status, total_order_value, delivery_id, delivery_status,
        goods_issue_date, billing_eligibility_status, billing_eligibility_reason,
        billing_trigger_type, invoice_count, last_invoice_date, action_group
    FROM ranked
    WHERE rn_per_status <= {per_status_cap}
    ORDER BY
        CASE billing_eligibility_status
            WHEN 'BLOCKED' THEN 1
            WHEN 'PENDING_GOODS_ISSUE' THEN 2
            WHEN 'PENDING_MILESTONE' THEN 3
            WHEN 'NO_DELIVERY' THEN 4
            WHEN 'PENDING_DELIVERY' THEN 5
            WHEN 'PENDING_SERVICE_CLOSE' THEN 6
            WHEN 'PENDING_BILLING_CYCLE' THEN 7
            ELSE 8
        END,
        total_order_value DESC NULLS LAST
    LIMIT {row_limit}
    """
    rows = run_query(sql)
    rows = _enrich_copilot_rows(rows)
    _agent_cache_set(cache_key, rows)
    return rows


def billing_agent_recommendation(payload: dict) -> dict:
    sales_order_id = str(payload.get("sales_order_id", "") or "").strip()
    if not sales_order_id:
        return {"response": "sales_order_id is required"}

    customer = _resolve_customer_label(
        str(payload.get("customer_name") or ""),
        str(payload.get("customer_id") or ""),
    )
    customer_id = str(payload.get("customer_id") or "")
    company_code = str(payload.get("company_code") or "").strip() or None
    status = str(payload.get("billing_eligibility_status") or "UNKNOWN")
    reason = str(payload.get("billing_eligibility_reason") or "")
    order_value = float(payload.get("total_order_value") or 0)
    delivery_status = str(payload.get("delivery_status") or "n/a")
    credit_status = str(payload.get("credit_check_status") or "n/a")
    action_group = str(payload.get("action_group") or "ACT_NOW")

    customer_ctx = _agent_entity_context(
        customer_id, company_code, sales_order_id=sales_order_id,
    ) if customer_id else {}
    customer_context = _format_customer_context_block(customer_ctx) if customer_id else ""
    block_line = block_scenario_lines(
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=str(payload.get("order_status") or ""),
    )
    problem_line = billing_problem_summary(
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=str(payload.get("order_status") or ""),
    )

    policy_rows = search_policies(
        f"billing eligibility {status} {reason}",
        "BILLING",
        limit=5,
    )
    policy_api = policies_for_api(policy_rows)

    prompt = (
        f"O2C Billing Exceptions Agent — billing eligibility action plan.\n\n"
        f"Sales order: {sales_order_id}\n"
        f"Customer: {customer}\n"
        f"Order value: ${order_value:,.0f}\n"
        f"Definitive problem: {problem_line}\n"
        f"Billing eligibility status: {status}\n"
        f"Priority: {action_group.replace('_', ' ')}\n"
        f"{block_line}\n\n"
    )
    if customer_context:
        prompt += f"CUSTOMER CONTEXT (this order / its invoices only):\n{customer_context}\n\n"
    prompt += BILLING_OUTPUT_FORMAT

    def fallback() -> dict:
        base = build_billing_fallback(
            sales_order_id=sales_order_id,
            customer=customer,
            customer_id=customer_id,
            order_value=order_value,
            status=status,
            reason=reason,
            delivery_status=delivery_status,
            credit_status=credit_status,
            action_group=action_group,
            policies=policy_api,
            customer_context=customer_context,
            order_status=str(payload.get("order_status") or ""),
        )
        return {"response": base}

    result = _workbench_live_agent(
        "billing", prompt, fallback,
        policy_query=f"billing eligibility {status} {reason}",
        policy_domain="BILLING",
        cache_entity_id=sales_order_id,
    )
    if result.get("response"):
        result["response"] = polish_billing_response(
            result["response"],
            policies=result.get("policies") or policy_api,
            sales_order_id=sales_order_id,
            customer=customer,
            customer_id=customer_id,
            order_value=order_value,
            status=status,
            reason=reason,
            delivery_status=delivery_status,
            credit_status=credit_status,
            action_group=action_group,
            customer_context=customer_context,
            order_status=str(payload.get("order_status") or ""),
        )
    return result


def billing_agent_credit_escalation(payload: dict) -> dict:
    sales_order_id = str(payload.get("sales_order_id") or "").strip()
    customer = str(payload.get("customer_name") or payload.get("customer_id") or "")
    reason = str(payload.get("billing_eligibility_reason") or "Credit or order hold")
    order_value = float(payload.get("total_order_value") or 0)
    credit_status = str(payload.get("credit_check_status") or "BLOCKED")

    subject = f"Credit release required — {sales_order_id}"
    body = f"""Credit team,

Sales order {sales_order_id} for {customer} (${order_value:,.0f}) is blocked for billing.

Block reason: {reason}
Credit check status: {credit_status}

Please review exposure / payment terms and release the hold or advise on partial billing so AR can invoice.

OrderLens Billing Agent
"""
    return {"to": "credit-control@orderlens.internal", "subject": subject, "body": body}


def billing_agent_fulfillment_escalation(payload: dict) -> dict:
    sales_order_id = str(payload.get("sales_order_id") or "").strip()
    customer = str(payload.get("customer_name") or payload.get("customer_id") or "")
    delivery_status = str(payload.get("delivery_status") or "n/a")
    status = str(payload.get("billing_eligibility_status") or "")

    subject = f"Fulfillment action — goods issue / delivery for {sales_order_id}"
    body = f"""Fulfillment team,

Billing is waiting on logistics for sales order {sales_order_id} ({customer}).

Billing eligibility: {status.replace('_', ' ')}
Delivery status: {delivery_status}

Please confirm shipment completion and post goods issue (or update delivery docs) so billing can trigger the invoice.

OrderLens Billing Agent
"""
    return {"to": "fulfillment-ops@orderlens.internal", "subject": subject, "body": body}


def billing_agent_mark_reviewed(payload: dict) -> dict:
    from FabicApp.db import run_execute

    sales_order_id = str(payload.get("sales_order_id") or "").strip()
    customer_id = str(payload.get("customer_id") or "").strip()
    company_code = str(payload.get("company_code") or "1000").strip()
    notes = str(payload.get("notes") or "Billing blocker reviewed in OrderLens Billing Agent").replace("'", "''")

    if not sales_order_id:
        return {"ok": False, "message": "sales_order_id is required"}

    safe_so = sales_order_id.replace("'", "''")
    safe_cust = customer_id.replace("'", "''")
    safe_co = company_code.replace("'", "''")
    event_id = f"OSE-{uuid.uuid4().hex[:8].upper()}"

    try:
        seq_rows = run_query(f"""
            SELECT COALESCE(MAX(STATUS_SEQ), 0) + 1 AS next_seq
            FROM {DB}.raw_vault.order_status_history
            WHERE SALES_ORDER_ID = '{safe_so}'
        """)
        next_seq = int(seq_rows[0].get("next_seq") or seq_rows[0].get("NEXT_SEQ") or 1)

        run_execute(f"""
            INSERT INTO {DB}.raw_vault.order_status_history (
                STATUS_EVENT_ID, SALES_ORDER_ID, CUSTOMER_ID, COMPANY_CODE, STATUS_SEQ,
                LIFECYCLE_AREA, ORDER_STATUS, STATUS_DATE, STATUS_TIMESTAMP, CHANGED_BY,
                IS_LATEST_STATUS, NOTES, CREATED_AT, UPDATED_AT, SOURCE_SYSTEM
            ) VALUES (
                '{event_id}', '{safe_so}', '{safe_cust}', '{safe_co}', {next_seq},
                'BILLING', 'UNDER_BILLING_REVIEW', CURRENT_DATE(), CURRENT_TIMESTAMP(),
                'ORDERLENS_BILLING_AGENT', TRUE, '{notes}',
                CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'ORDERLENS_AGENT'
            )
        """)
        with _AGENT_CACHE_LOCK:
            _AGENT_CACHE.pop("billing_queue:v3", None)
        return {
            "ok": True,
            "message": f"Billing review logged for {sales_order_id} (event {event_id}).",
            "event_id": event_id,
        }
    except Exception as exc:
        log.warning("Billing mark reviewed failed", exc_info=True)
        return {"ok": False, "message": f"Failed to log billing review: {exc}"}


def proactive_dispute_queue() -> list[dict]:
    cached = _agent_cache_get("proactive_dispute_queue:v3")
    if cached is not None:
        return cached  # type: ignore[return-value]

    sql = f"""
    SELECT
        r.RISK_ID AS risk_id,
        r.INVOICE_ID AS invoice_id,
        r.SALES_ORDER_ID AS sales_order_id,
        r.CUSTOMER_ID AS customer_id,
        r.CUSTOMER_NAME AS customer_name,
        r.COMPANY_CODE AS company_code,
        r.BILLING_TRIGGER_TYPE AS billing_trigger_type,
        r.EXPOSURE_AMOUNT AS exposure_amount,
        r.PREDICTED_DISPUTE_REASON AS predicted_dispute_reason,
        r.RISK_DRIVER AS risk_driver,
        r.POLICY_REFS AS policy_refs,
        r.RISK_SCORE AS risk_score,
        r.PREDICTED_DISPUTE_DATE::VARCHAR AS predicted_dispute_date,
        r.ACTION_GROUP AS action_group,
        r.PRIORITY_SCORE AS priority_score
    FROM {BM}.proactive_dispute_risk_vw r
    ORDER BY r.PRIORITY_SCORE DESC
    LIMIT 50
    """
    try:
        rows = run_query(sql)
    except Exception:
        log.warning("proactive_dispute_risk_vw unavailable; using billing eligibility fallback", exc_info=True)
        rows = run_query(f"""
            SELECT
                'RISK-FALL-' || be.SALES_ORDER_ID AS risk_id,
                NULL AS invoice_id,
                be.SALES_ORDER_ID AS sales_order_id,
                be.CUSTOMER_ID AS customer_id,
                be.CUSTOMER_NAME AS customer_name,
                be.COMPANY_CODE AS company_code,
                coalesce(be.BILLING_TRIGGER_TYPE, 'DELIVERY_BASED') AS billing_trigger_type,
                be.TOTAL_ORDER_VALUE AS exposure_amount,
                'QUALITY' AS predicted_dispute_reason,
                be.BILLING_ELIGIBILITY_REASON AS risk_driver,
                array_construct('BILL-ACC-01') AS policy_refs,
                0.65 AS risk_score,
                dateadd(day, 14, current_date())::varchar AS predicted_dispute_date,
                'PLAN_THIS_WEEK' AS action_group,
                be.TOTAL_ORDER_VALUE * 0.65 AS priority_score
            FROM {BM}.billing_eligibility_vw be
            WHERE be.BILLING_ELIGIBILITY_STATUS IN ('BLOCKED', 'PENDING_GOODS_ISSUE', 'PENDING_MILESTONE')
            ORDER BY be.TOTAL_ORDER_VALUE DESC
            LIMIT 25
        """)

    rows = _enrich_copilot_rows(rows)
    _agent_cache_set("proactive_dispute_queue:v3", rows)
    return rows


def _dispute_predicted_recommendation(payload: dict) -> dict:
    risk_id = str(payload.get("risk_id", "") or "").strip()
    customer = str(payload.get("customer_name") or payload.get("customer_id") or "")
    reason = str(payload.get("predicted_dispute_reason") or "UNKNOWN")
    driver = str(payload.get("risk_driver") or "")
    billing_type = str(payload.get("billing_trigger_type") or "DELIVERY_BASED")
    exposure = float(payload.get("exposure_amount") or 0)
    score = float(payload.get("risk_score") or 0)
    pred_date = str(payload.get("predicted_dispute_date") or "")
    invoice_id = str(payload.get("invoice_id") or "n/a")
    sales_order_id = str(payload.get("sales_order_id") or "n/a")
    action_group = str(payload.get("action_group") or "ACT_NOW")
    policy_rows = search_policies(
        f"proactive dispute {reason} {billing_type}",
        "DISPUTE",
        limit=5,
    )
    policy_api = policies_for_api(policy_rows)

    prompt = (
        f"O2C Dispute Agent — preventive mode. Predicted {reason} dispute for {customer} "
        f"(invoice {invoice_id}, order {sales_order_id}). Billing trigger: {billing_type}. "
        f"Exposure ${exposure:,.0f}, risk score {score:.0%}, predicted by {pred_date}. "
        f"Driver: {driver}. Priority {action_group}.\n\n"
        f"{DISPUTE_OUTPUT_FORMAT}"
    )

    def fallback() -> dict:
        base = f"""<strong>SITUATION ASSESSMENT</strong>
{risk_id or 'Risk item'} for {customer}: predicted {reason} dispute on invoice {invoice_id} (${exposure:,.0f}, {score:.0%} confidence). Trigger: {billing_type.replace('_', ' ')}. Driver: {driver}. Priority: {action_group.replace('_', ' ')}.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. BILLING VALIDATION: Re-check trigger rules for {billing_type.replace('_', ' ')}. Next step: Billing analyst validates invoice lines today.
2. PROACTIVE AP CONTACT: Send invoice package with POD/pricing proof. Next step: AR emails AP within 48 hours.
3. INTERNAL HOLD: Pause dunning on related AR until evidence confirmed. Next step: Collections lead flags account today.
4. ESCALATION: Route to dispute lead if exposure exceeds threshold. Next step: Manager review before {pred_date or 'period close'}.

<strong>INVOICE CONTEXT</strong>
- Invoice: {invoice_id} · Order: {sales_order_id}
- Exposure: ${exposure:,.0f} · Predicted reason: {reason}
- Risk driver: {driver}

<strong>TIMELINE</strong>
- Today: Validate billing evidence and flag account.
- Within 48 hours: Proactive AP outreach with documentation.
- Day 7: Escalate if dispute case opens.

<strong>POLICY REFERENCES</strong>
{policy_ref_lines(policy_api)}"""
        return {"response": base}

    result = _workbench_live_agent(
        "dispute", prompt, fallback,
        policy_query=f"proactive dispute {reason} {billing_type}",
        policy_domain="DISPUTE",
        cache_entity_id=risk_id or f"pred:{invoice_id}",
    )
    if result.get("response"):
        result["response"] = polish_dispute_response(
            result["response"],
            policies=result.get("policies") or policy_api,
            dispute_id=risk_id or "predicted",
            customer_id=str(payload.get("customer_id") or ""),
            invoice_id=invoice_id,
            reason=reason,
            amount=exposure,
            status="PREDICTED",
            action_group=action_group,
        )
    return result


def proactive_dispute_recommendation(payload: dict) -> dict:
    """Backward-compatible alias — unified under O2C Dispute Agent."""
    return _dispute_predicted_recommendation(payload)


def collections_agent_queue(annual_interest_rate_pct: float = 8.0, customer_id: str | None = None) -> list[dict]:
    rate = max(1.0, min(float(annual_interest_rate_pct or 8.0), 30.0))
    cache_key = f"coll_queue:inv:v8:{rate}:{customer_id or 'all'}"
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    cust_filter = ""
    if customer_id:
        safe_cust = str(customer_id).replace("'", "''")
        cust_filter = f" AND ar.CUSTOMER_ID = '{safe_cust}'"
    tier_filter = (
        "1=1"
        if customer_id
        else """(action_group = 'ACT_NOW'        AND tier_rank <= 100)
       OR (action_group = 'PLAN_THIS_WEEK' AND tier_rank <= 60)
       OR (action_group = 'MONITOR'        AND tier_rank <= 40)"""
    )

    # Tiering (forward-looking for Plan / Monitor):
    # - ACT_NOW: already past due
    # - PLAN_THIS_WEEK: not past due, due within the next 7 days
    # - MONITOR: not past due, due 8+ days out (week after next and beyond)
    sql = f"""
    WITH rpc AS (
        SELECT COMPANY_CODE, AVG(RPC_RATE_PCT) AS avg_rpc
        FROM {BM}.rpc_vw
        GROUP BY COMPANY_CODE
    ),
    base AS (
        SELECT
            ar.INVOICE_ID AS invoice_id,
            ar.CUSTOMER_ID AS customer_id,
            COALESCE(cu.CUSTOMER_NAME, cu_any.CUSTOMER_NAME) AS customer_name,
            ar.COMPANY_CODE AS company_code,
            COALESCE(cu.CUSTOMER_SEGMENT, cu_any.CUSTOMER_SEGMENT) AS customer_segment,
            ar.OPEN_AMOUNT_USD AS open_amount_usd,
            ar.OPEN_AMOUNT_USD AS past_due_usd,
            ar.DAYS_PAST_DUE AS days_past_due,
            ar.DAYS_PAST_DUE AS max_days_past_due,
            ar.DUE_DATE::VARCHAR AS due_date,
            DATEDIFF('day', CURRENT_DATE(), ar.DUE_DATE) AS days_until_due,
            ar.AGING_BUCKET AS aging_bucket,
            1 AS open_items,
            IFF(ar.AGING_BUCKET = '90+', ar.OPEN_AMOUNT_USD, 0) AS bucket_90_plus,
            COALESCE(rpc.avg_rpc, 50) AS avg_rpc,
            COALESCE(ar.DUNNING_LEVEL, cu.DUNNING_LEVEL, cu_any.DUNNING_LEVEL, 0) AS dunning_level,
            COALESCE(cu.ACCOUNT_MANAGER, cu_any.ACCOUNT_MANAGER) AS account_manager,
            ROUND(
                ar.OPEN_AMOUNT_USD * ({rate} / 100.0) * (GREATEST(ar.DAYS_PAST_DUE, 0) / 365.0),
                2
            ) AS amount_lost_usd,
            {rate} AS annual_interest_rate_pct,
            ar.OPEN_AMOUNT_USD
                * (1 + GREATEST(ar.DAYS_PAST_DUE, 0) / 30.0)
                * (1 + (100 - COALESCE(rpc.avg_rpc, 50)) / 100.0) AS priority_score,
            CASE
                WHEN ar.DAYS_PAST_DUE > 0 THEN 'ACT_NOW'
                WHEN DATEDIFF('day', CURRENT_DATE(), ar.DUE_DATE) BETWEEN 0 AND 7 THEN 'PLAN_THIS_WEEK'
                WHEN DATEDIFF('day', CURRENT_DATE(), ar.DUE_DATE) >= 8 THEN 'MONITOR'
                ELSE 'ACT_NOW'
            END AS action_group
        FROM {BM}.ar_open_item_vw ar
        LEFT JOIN {BM}.customer_vw cu
          ON cu.CUSTOMER_ID = ar.CUSTOMER_ID
         AND cu.COMPANY_CODE = ar.COMPANY_CODE
        LEFT JOIN (
            SELECT
                CUSTOMER_ID,
                MAX(CUSTOMER_NAME) AS CUSTOMER_NAME,
                MAX(CUSTOMER_SEGMENT) AS CUSTOMER_SEGMENT,
                MAX(DUNNING_LEVEL) AS DUNNING_LEVEL,
                MAX(ACCOUNT_MANAGER) AS ACCOUNT_MANAGER
            FROM {BM}.customer_vw
            GROUP BY CUSTOMER_ID
        ) cu_any
          ON cu_any.CUSTOMER_ID = ar.CUSTOMER_ID
        LEFT JOIN rpc
          ON rpc.COMPANY_CODE = ar.COMPANY_CODE
        WHERE ar.OPEN_AMOUNT_USD > 0
          AND (
            ar.DAYS_PAST_DUE > 0
            OR DATEDIFF('day', CURRENT_DATE(), ar.DUE_DATE) BETWEEN 0 AND 120
          )
          {cust_filter}
    ),
    ranked AS (
        SELECT
            base.*,
            ROW_NUMBER() OVER (
                PARTITION BY action_group
                ORDER BY
                    CASE action_group
                        WHEN 'ACT_NOW' THEN amount_lost_usd
                        ELSE -days_until_due
                    END DESC,
                    open_amount_usd DESC
            ) AS tier_rank
        FROM base
    )
    SELECT * EXCLUDE (tier_rank)
    FROM ranked
    WHERE {tier_filter}
    ORDER BY
        CASE action_group
            WHEN 'ACT_NOW' THEN 0
            WHEN 'PLAN_THIS_WEEK' THEN 1
            ELSE 2
        END,
        CASE action_group
            WHEN 'ACT_NOW' THEN amount_lost_usd * -1
            ELSE days_until_due
        END,
        open_amount_usd DESC
    """
    rows = run_query(sql)
    rows = _enrich_copilot_rows(rows)
    _agent_cache_set(cache_key, rows)
    return rows


def _collections_invoice_sql(
    customer_id: str,
    company_code: str | None,
    limit: int,
    invoice_id: str | None = None,
) -> str:
    safe_id = customer_id.replace("'", "''")
    co_filter = ""
    if company_code:
        safe_co = company_code.replace("'", "''")
        co_filter = f" AND ar.COMPANY_CODE = '{safe_co}'"
    inv_filter = ""
    if invoice_id:
        safe_inv = invoice_id.replace("'", "''")
        inv_filter = f" AND ar.INVOICE_ID = '{safe_inv}'"
    eligibility = (
        "AND ar.OPEN_AMOUNT_USD > 0"
        if invoice_id
        else """AND ar.OPEN_AMOUNT_USD > 0
          AND (
            ar.DAYS_PAST_DUE > 0
            OR DATEDIFF('day', CURRENT_DATE(), ar.DUE_DATE) BETWEEN 1 AND 120
          )"""
    )
    return f"""
        SELECT
            ar.INVOICE_ID,
            ar.OPEN_AMOUNT_USD,
            ar.DAYS_PAST_DUE,
            ar.AGING_BUCKET,
            ar.DUE_DATE,
            ar.DUNNING_LEVEL,
            ar.IS_DISPUTED,
            ar.COMPANY_CODE,
            d.DISPUTE_ID
        FROM {BM}.ar_open_item_vw ar
        LEFT JOIN {BM}.dispute_vw d
          ON d.INVOICE_ID = ar.INVOICE_ID
         AND d.CUSTOMER_ID = ar.CUSTOMER_ID
         AND upper(coalesce(d.DISPUTE_STATUS, '')) NOT IN ('RESOLVED', 'CLOSED', 'CANCELLED')
        WHERE ar.CUSTOMER_ID = '{safe_id}' {co_filter}{inv_filter}
          {eligibility}
        ORDER BY ar.DAYS_PAST_DUE DESC NULLS LAST, ar.OPEN_AMOUNT_USD DESC
        LIMIT {int(limit)}
    """


def _fetch_collections_invoices(
    customer_id: str,
    company_code: str | None = None,
    *,
    invoice_id: str | None = None,
    limit: int = 25,
) -> tuple[list[dict], list[dict]]:
    """Return raw AR rows and enriched invoice dicts for collections UI/API."""
    if not customer_id:
        return [], []
    ar_rows = run_query(_collections_invoice_sql(customer_id, company_code, limit, invoice_id))
    if not ar_rows and company_code and not invoice_id:
        ar_rows = run_query(_collections_invoice_sql(customer_id, None, limit, invoice_id))
    invoice_rows = [enrich_invoice_row(r) for r in ar_rows]
    return ar_rows, invoice_rows


def collections_agent_recommendation(payload: dict) -> dict:
    customer_id = str(payload.get("customer_id", "") or "").strip()
    if not customer_id:
        return {"response": "customer_id is required"}

    safe_id = customer_id.replace("'", "''")
    company_code = str(payload.get("company_code") or "").strip() or None
    invoice_id = str(payload.get("invoice_id") or "").strip() or None

    cust_rows = run_query(f"""
        SELECT CUSTOMER_ID, CUSTOMER_NAME, CUSTOMER_SEGMENT, DUNNING_LEVEL, CREDIT_EXPOSURE, PAYMENT_TERMS
        FROM {BM}.customer_vw WHERE CUSTOMER_ID = '{safe_id}' LIMIT 1
    """)
    ar_rows, invoice_rows = _fetch_collections_invoices(
        customer_id, company_code, invoice_id=invoice_id, limit=25,
    )

    cust = cust_rows[0] if cust_rows else {}
    name = str(_val(cust, "CUSTOMER_NAME", "customer_name") or customer_id)
    focus_due_date: str | None = None
    if invoice_rows:
        focus = invoice_rows[0]
        total_past = float(focus.get("open_amount_usd") or 0)
        max_days = int(focus.get("days_past_due") or 0)
        focus_due_date = str(focus.get("due_date") or "") or None
        inv_block = invoice_detail_lines([focus])
        invoice_label = str(focus.get("invoice_id") or invoice_id or "")
    else:
        total_past = sum(float(_val(r, "OPEN_AMOUNT_USD", "open_amount_usd") or 0) for r in ar_rows)
        max_days = max((int(_val(r, "DAYS_PAST_DUE", "days_past_due") or 0) for r in ar_rows), default=0)
        inv_block = invoice_detail_lines(invoice_rows)
        invoice_label = invoice_id or ""
    dunning = int(_val(cust, "DUNNING_LEVEL", "dunning_level") or 0)
    action_group = str(payload.get("action_group") or "ACT_NOW")
    past_due_payload = float(payload.get("past_due_usd") or payload.get("open_amount_usd") or 0)
    if past_due_payload > 0:
        total_past = past_due_payload
    days_payload = int(float(payload.get("days_past_due") or payload.get("max_days_past_due") or 0))
    if days_payload > 0:
        max_days = days_payload
    days_until = int(float(payload.get("days_until_due") or 0))
    if invoice_rows and not days_until:
        days_until = max(int(float(invoice_rows[0].get("days_until_due") or 0)), 0)
    if not days_until and focus_due_date:
        try:
            from datetime import date
            due = date.fromisoformat(str(focus_due_date)[:10])
            days_until = max((due - date.today()).days, 0)
        except ValueError:
            pass
    is_proactive = max_days <= 0 and days_until > 0

    customer_ctx = _agent_entity_context(
        customer_id,
        company_code,
        invoice_id=invoice_label or invoice_id,
        days_past_due=max_days,
        open_amount=total_past,
        due_date=focus_due_date,
    )
    customer_context = _format_customer_context_block(customer_ctx)
    policy_query = (
        f"collections proactive reminder due in {days_until} days payment terms {action_group}"
        if is_proactive
        else f"collections dunning past due {max_days} days late fee partial payment {action_group}"
    )
    policy_rows = search_policies(policy_query, "COLLECTIONS", limit=5)
    policy_api = policies_for_api(policy_rows)

    if is_proactive:
        timing_line = f"Due in {days_until} days (proactive — not yet past due)"
        focus_line = (
            f"Focus invoice: {invoice_label} — ${total_past:,.0f} open, {timing_line}.\n"
            if invoice_label
            else ""
        )
        agent_intro = (
            "O2C Collections Agent — produce a proactive outreach plan before this invoice is due."
        )
        status_line = (
            f"Open on focus line: ${total_past:,.0f} | {timing_line} | "
            f"Dunning level: {dunning} | Priority: {action_group.replace('_', ' ')}"
        )
    else:
        focus_line = (
            f"Focus invoice: {invoice_label} — ${total_past:,.0f} open, {max_days} days past due.\n"
            if invoice_label
            else ""
        )
        agent_intro = (
            "O2C Collections Agent — produce a collector action plan for this past-due invoice line."
        )
        status_line = (
            f"Past due on focus line: ${total_past:,.0f} | Days past due: {max_days} | "
            f"Dunning level: {dunning} | Priority: {action_group.replace('_', ' ')}"
        )

    agent_prompt = (
        f"{agent_intro}\n\n"
        f"Customer: {name} ({customer_id})\n"
        f"{focus_line}"
        f"{status_line}\n\n"
        f"CUSTOMER CONTEXT (focus invoice {invoice_label or invoice_id or 'n/a'} only):\n{customer_context}\n\n"
        f"OPEN INVOICES (authoritative — repeat exactly in your OPEN INVOICES section):\n{inv_block}\n\n"
        f"{COLLECTIONS_OUTPUT_FORMAT}"
    )

    def fallback() -> dict:
        base = build_collections_fallback(
            name=name,
            customer_id=customer_id,
            total_past=total_past,
            ar_rows=ar_rows,
            max_days=max_days,
            dunning=dunning,
            action_group=action_group,
            policies=policy_api,
            customer_context=customer_context,
        )
        return {"response": base}

    result = _workbench_live_agent(
        "collections", agent_prompt, fallback,
        policy_query=policy_query,
        policy_domain="COLLECTIONS",
        cache_entity_id=f"{customer_id}:{invoice_id or 'all'}:{company_code or 'all'}",
        agent_timeout=O2C_COLLECTIONS_AGENT_TIMEOUT_SEC,
    )
    if result.get("response"):
        result["response"] = polish_collections_response(
            result["response"],
            ar_rows=ar_rows,
            policies=result.get("policies") or policy_api,
            action_group=action_group,
            customer_context=customer_context,
            invoice_rows=invoice_rows,
        )
    result["open_invoices"] = invoice_rows
    return result


def collections_agent_email_draft(payload: dict) -> dict:
    customer_id = str(payload.get("customer_id", "") or "").strip()
    name = str(payload.get("customer_name") or customer_id)
    invoice_id = str(payload.get("invoice_id") or "").strip()
    past_due = float(payload.get("past_due_usd") or payload.get("open_amount_usd") or 0)
    max_days = int(float(payload.get("days_past_due") or payload.get("max_days_past_due") or 0))
    dunning = int(float(payload.get("dunning_level") or 1))

    if invoice_id:
        subject = f"Payment Reminder - Invoice {invoice_id} (${past_due:,.0f})"
    else:
        subject = f"Payment Reminder - Past Due Balance (${past_due:,.0f})"
    if dunning >= 3:
        subject = f"URGENT: Final Notice - {invoice_id or customer_id}"

    inv_line = f"Invoice {invoice_id} " if invoice_id else "Invoices "
    body = f"""Dear {name} Accounts Payable Team,

Our records show {inv_line}with a past-due balance of ${past_due:,.2f} aged {max_days} days.

Please remit payment or contact us within 5 business days to arrange a payment plan. If any invoices are in dispute, reply with dispute reference numbers so we can resolve them promptly.

Regards,
OrderToCash Collections Team"""

    contact = str(payload.get("account_manager") or "collections@ordertocash.com")
    return {"to": contact, "subject": subject, "body": body}


def collections_agent_open_invoices(
    customer_id: str,
    company_code: str | None = None,
    annual_interest_rate_pct: float = 8.0,
) -> list[dict]:
    del annual_interest_rate_pct  # API compat; late fees follow COLL-LPF-01 in collection_policy.py
    _, invoice_rows = _fetch_collections_invoices(customer_id, company_code, limit=25)
    return invoice_rows


def collections_agent_log_ptp(payload: dict) -> dict:
    from FabicApp.db import run_execute

    customer_id = str(payload.get("customer_id", "") or "").strip()
    company_code = str(payload.get("company_code") or "1000").strip()
    amount = float(payload.get("promised_amount") or 0)
    pay_date = str(payload.get("promised_pay_date") or "").strip()[:10]
    invoice_id = str(payload.get("invoice_id") or "").strip()
    notes = str(payload.get("notes") or "").strip()
    contact_outcome = str(payload.get("contact_outcome") or "RPC").strip().upper()
    is_rpc = contact_outcome == "RPC"

    if not customer_id:
        return {"ok": False, "message": "Customer ID is required"}
    if amount <= 0:
        return {"ok": False, "message": "Promised amount must be greater than zero"}
    if not pay_date or len(pay_date) < 10:
        return {"ok": False, "message": "Promised pay date is required"}

    safe_cust = customer_id.replace("'", "''")
    safe_co = company_code.replace("'", "''")
    safe_date = pay_date.replace("'", "''")
    safe_notes = (notes or f"PTP captured via OrderToCash Agent — ${amount:,.2f} by {pay_date}").replace("'", "''")
    safe_invoice = invoice_id.replace("'", "''") if invoice_id else None

    activity_id = f"CA-{uuid.uuid4().hex[:8].upper()}"
    ptp_id = f"PTP-{uuid.uuid4().hex[:8].upper()}"

    try:
        run_execute(f"""
            INSERT INTO {DB}.raw_vault.collection_activity (
                ACTIVITY_ID, CUSTOMER_ID, COMPANY_CODE, ACTIVITY_TYPE, ACTIVITY_DATE,
                CONTACT_OUTCOME, IS_RIGHT_PARTY_CONTACT, NOTES, SOURCE_SYSTEM
            )
            VALUES (
                '{activity_id}', '{safe_cust}', '{safe_co}', 'CALL', CURRENT_DATE(),
                '{contact_outcome}', {str(is_rpc).upper()}, '{safe_notes}', 'ORDERLENS_AGENT'
            )
        """)

        invoice_sql = f"'{safe_invoice}'" if safe_invoice else "NULL"
        run_execute(f"""
            INSERT INTO {DB}.raw_vault.promised_to_pay (
                PTP_ID, CUSTOMER_ID, COMPANY_CODE, INVOICE_ID, ACTIVITY_ID,
                PROMISE_DATE, PROMISED_PAY_DATE, PROMISED_AMOUNT, CURRENCY_CODE,
                PTP_STATUS, IS_KEPT, SOURCE_SYSTEM
            )
            VALUES (
                '{ptp_id}', '{safe_cust}', '{safe_co}', {invoice_sql}, '{activity_id}',
                CURRENT_DATE(), '{safe_date}'::DATE, {amount}, 'USD',
                'OPEN', FALSE, 'ORDERLENS_AGENT'
            )
        """)
        promote_staging_to_mart("collections")
        return {
            "ok": True,
            "message": f"PTP {ptp_id} logged — ${amount:,.2f} due {pay_date}",
            "ptp_id": ptp_id,
            "activity_id": activity_id,
        }
    except Exception as exc:
        log.warning("Log PTP failed", exc_info=True)
        return {"ok": False, "message": f"Failed to log PTP: {exc}"}


def collections_agent_parse_inbound_reply(email_body: str, payload: dict) -> dict:
    """Extract PTP commitment from a customer email reply using Azure OpenAI."""
    customer_id = str(payload.get("customer_id") or "").strip()
    past_due = float(payload.get("past_due_usd") or 0)
    body = (email_body or "").strip()
    if not body:
        return {"has_ptp": False, "reason": "Empty reply"}

    prompt = f"""You are an AR collections parser. Extract promise-to-pay details from this customer email reply.

Customer: {payload.get('customer_name') or customer_id}
Past due balance: ${past_due:,.2f}

Email reply:
\"\"\"
{body[:3000]}
\"\"\"

Respond in JSON only with keys:
- has_ptp (true/false)
- promised_amount (number or null)
- promised_pay_date (YYYY-MM-DD or null)
- contact_outcome (RPC or EMAIL)
- summary (one sentence)

If the customer commits to pay by a date or amount, set has_ptp true and fill fields from the text."""

    try:
        raw = cortex_complete("llama3.1-8b", prompt)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
        else:
            parsed = {"has_ptp": False, "reason": "Could not parse model response"}
    except Exception:
        log.warning("Inbound reply parse failed", exc_info=True)
        parsed = {"has_ptp": False, "reason": "Parse error"}

    if not parsed.get("has_ptp"):
        lowered = body.lower()
        if any(w in lowered for w in ("will pay", "payment on", "remit", "promise", "commit", "by end of")):
            default_date = (datetime.date.today() + datetime.timedelta(days=14)).isoformat()
            parsed = {
                "has_ptp": True,
                "promised_amount": past_due * 0.5 if past_due else None,
                "promised_pay_date": default_date,
                "contact_outcome": "EMAIL",
                "summary": "Inferred PTP from payment commitment language in reply.",
            }
        else:
            return {**parsed, "has_ptp": False}

    amount = float(parsed.get("promised_amount") or past_due or 0)
    pay_date = str(parsed.get("promised_pay_date") or "")[:10]
    if not pay_date or len(pay_date) < 10:
        pay_date = (datetime.date.today() + datetime.timedelta(days=14)).isoformat()

    return {
        "has_ptp": True,
        "promised_amount": amount,
        "promised_pay_date": pay_date,
        "contact_outcome": str(parsed.get("contact_outcome") or "EMAIL").upper(),
        "notes": f"Auto-parsed from inbound email: {parsed.get('summary') or body[:200]}",
        "raw_reply": body,
    }


def collections_agent_process_inbound(payload: dict) -> dict:
    """Parse inbound customer reply and auto-log PTP when detected."""
    email_body = str(payload.get("email_body") or payload.get("reply_text") or "").strip()
    if not email_body:
        name = str(payload.get("customer_name") or payload.get("customer_id") or "Customer")
        past_due = float(payload.get("past_due_usd") or 0)
        max_days = int(float(payload.get("max_days_past_due") or 0))
        default_amount = round(past_due * 0.6, 2) if past_due else 1000.0
        pay_date = (datetime.date.today() + datetime.timedelta(days=10)).isoformat()
        email_body = (
            f"Hi OrderToCash Collections,\n\n"
            f"Thank you for the reminder. {name} AP confirms we received your notice regarding the "
            f"${past_due:,.2f} past-due balance ({max_days} days). We will remit ${default_amount:,.2f} "
            f"by {pay_date}. Please apply to the oldest open invoices.\n\n"
            f"Regards,\n{name} Accounts Payable"
        )

    parsed = collections_agent_parse_inbound_reply(email_body, payload)
    if not parsed.get("has_ptp"):
        return {
            "ok": False,
            "automated": False,
            "message": parsed.get("reason") or "No promise-to-pay detected in reply.",
            "parsed": parsed,
        }

    ptp_payload = {
        **payload,
        "promised_amount": parsed["promised_amount"],
        "promised_pay_date": parsed["promised_pay_date"],
        "contact_outcome": parsed.get("contact_outcome", "EMAIL"),
        "notes": parsed.get("notes"),
    }
    result = collections_agent_log_ptp(ptp_payload)
    return {
        "ok": result.get("ok", False),
        "automated": True,
        "message": result.get("message", ""),
        "parsed": parsed,
        "ptp_id": result.get("ptp_id"),
        "activity_id": result.get("activity_id"),
        "inbound_reply": email_body,
    }


def collections_agent_auto_flow(payload: dict) -> dict:
    """Fully automated demo flow: send reminder -> simulate reply -> parse -> log PTP."""
    customer_id = str(payload.get("customer_id") or "").strip()
    if not customer_id:    
        return {"ok": False, "steps": [], "message": "customer_id is required"}

    name = str(payload.get("customer_name") or customer_id)
    past_due = float(payload.get("past_due_usd") or 0)
    max_days = int(float(payload.get("max_days_past_due") or 0))
    default_amount = round(past_due * 0.6, 2) if past_due else 1000.0
    pay_date = (datetime.date.today() + datetime.timedelta(days=10)).isoformat()

    steps: list[dict] = []

    draft = collections_agent_email_draft(payload)
    steps.append({
        "step": "send_reminder",
        "status": "completed",
        "detail": f"Payment reminder drafted to {draft.get('to')}",
        "email": draft,
    })

    simulated_reply = (
        f"Hi OrderToCash Collections,\n\n"
        f"Thank you for the reminder. {name} AP confirms we received your notice regarding the "
        f"${past_due:,.2f} past-due balance ({max_days} days). We will remit ${default_amount:,.2f} "
        f"by {pay_date}. Please apply to the oldest open invoices.\n\n"
        f"Regards,\n{name} Accounts Payable"
    )
    steps.append({
        "step": "receive_reply",
        "status": "completed",
        "detail": "Inbound customer email received and queued for agent processing",
        "reply_preview": simulated_reply[:280] + "...",
    })

    parsed = collections_agent_parse_inbound_reply(simulated_reply, payload)
    steps.append({
        "step": "parse_reply",
        "status": "completed" if parsed.get("has_ptp") else "skipped",
        "detail": parsed.get("summary") or parsed.get("reason") or "Parsed reply",
        "parsed": parsed,
    })

    if not parsed.get("has_ptp"):
        return {"ok": False, "steps": steps, "message": "Automation stopped — no PTP in simulated reply."}

    ptp_result = collections_agent_log_ptp({
        **payload,
        "promised_amount": parsed["promised_amount"],
        "promised_pay_date": parsed["promised_pay_date"],
        "contact_outcome": parsed.get("contact_outcome", "EMAIL"),
        "notes": parsed.get("notes"),
    })
    steps.append({
        "step": "log_ptp",
        "status": "completed" if ptp_result.get("ok") else "failed",
        "detail": ptp_result.get("message", ""),
        "ptp_id": ptp_result.get("ptp_id"),
        "activity_id": ptp_result.get("activity_id"),
    })

    return {
        "ok": bool(ptp_result.get("ok")),
        "automated": True,
        "message": ptp_result.get("message") or "Automated collections flow completed.",
        "steps": steps,
        "simulated_reply": simulated_reply,
        "ptp_id": ptp_result.get("ptp_id"),
    }


def collections_agent_mark_contacted(payload: dict) -> dict:
    customer_id = str(payload.get("customer_id", "") or "").strip()
    company_code = str(payload.get("company_code") or "1000").strip()
    activity_type = str(payload.get("activity_type") or "CALL")
    notes = str(payload.get("notes") or "Contact logged via OrderToCash Collections Agent").replace("'", "''")

    activity_id = f"CA-{uuid.uuid4().hex[:8].upper()}"
    safe_cust = customer_id.replace("'", "''")
    safe_co = company_code.replace("'", "''")

    try:
        run_query(f"""
            INSERT INTO {DB}.raw_vault.collection_activity (
                ACTIVITY_ID, CUSTOMER_ID, COMPANY_CODE, ACTIVITY_TYPE, ACTIVITY_DATE,
                CONTACT_OUTCOME, IS_RIGHT_PARTY_CONTACT, NOTES, SOURCE_SYSTEM
            )
            VALUES (
                '{activity_id}', '{safe_cust}', '{safe_co}', '{activity_type}', CURRENT_DATE(),
                'RPC', TRUE, '{notes}', 'ORDERLENS_AGENT'
            )
        """)
        promote_staging_to_mart("collections")
        return {"ok": True, "message": f"Activity {activity_id} logged for {customer_id}", "activity_id": activity_id}
    except Exception as exc:
        log.warning("Mark contacted failed", exc_info=True)
        return {"ok": False, "message": f"Failed to log activity: {exc}"}


def collections_agent_mark_resolved(payload: dict) -> dict:
    """Mark collection case resolved — logs final activity note."""
    payload = {**payload, "notes": "Case marked resolved via OrderToCash Collections Agent", "activity_type": "CALL"}
    return collections_agent_mark_contacted(payload)


def cash_application_agent_queue(customer_id: str | None = None) -> list[dict]:
    cache_key = f"cash_queue:v5:{customer_id or 'all'}"
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    cust_filter = ""
    if customer_id:
        safe_cust = str(customer_id).replace("'", "''")
        cust_filter = f" AND p.CUSTOMER_ID = '{safe_cust}'"
    row_limit = 200 if customer_id else 50
    sql = f"""
    SELECT
        p.PAYMENT_ID AS payment_id,
        p.CUSTOMER_ID AS customer_id,
        COALESCE(cu.CUSTOMER_NAME, cu_any.CUSTOMER_NAME) AS customer_name,
        p.INVOICE_ID AS invoice_id,
        p.COMPANY_CODE AS company_code,
        p.PAYMENT_METHOD AS payment_method,
        p.PAYMENT_AMOUNT_USD AS payment_amount_usd,
        p.IS_PARTIAL AS is_partial,
        p.PAYMENT_DATE::VARCHAR AS payment_date,
        p.CLEARING_DATE::VARCHAR AS clearing_date,
        DATEDIFF('day', p.PAYMENT_DATE, CURRENT_DATE()) AS days_unapplied,
        CASE
            WHEN p.CLEARING_DATE IS NULL THEN 'UNAPPLIED'
            WHEN p.IS_PARTIAL THEN 'PARTIAL'
            ELSE 'EXCEPTION'
        END AS exception_type,
        p.PAYMENT_AMOUNT_USD * (1 + DATEDIFF('day', p.PAYMENT_DATE, CURRENT_DATE()) / 7.0) AS priority_score
    FROM {BM}.payment_vw p
    LEFT JOIN {BM}.customer_vw cu
      ON cu.CUSTOMER_ID = p.CUSTOMER_ID
     AND cu.COMPANY_CODE = p.COMPANY_CODE
    LEFT JOIN (
        SELECT CUSTOMER_ID, MAX(CUSTOMER_NAME) AS CUSTOMER_NAME
        FROM {BM}.customer_vw
        GROUP BY CUSTOMER_ID
    ) cu_any
      ON cu_any.CUSTOMER_ID = p.CUSTOMER_ID
    WHERE (p.CLEARING_DATE IS NULL OR p.IS_PARTIAL = TRUE)
      {cust_filter}
    ORDER BY priority_score DESC
    LIMIT {row_limit}
    """
    rows = run_query(sql)
    rows = _enrich_copilot_rows(rows)
    _agent_cache_set(cache_key, rows)
    return rows


def cash_application_match_candidates(customer_id: str, company_code: str | None, payment_amount: float) -> list[dict]:
    safe_cust = customer_id.replace("'", "''")
    co_filter = ""
    if company_code:
        safe_co = company_code.replace("'", "''")
        co_filter = f" AND ar.COMPANY_CODE = '{safe_co}'"
    return run_query(f"""
        SELECT
            ar.INVOICE_ID AS invoice_id,
            ar.OPEN_AMOUNT_USD AS open_amount_usd,
            ar.DAYS_PAST_DUE AS days_past_due,
            ar.AGING_BUCKET AS aging_bucket,
            ar.DUE_DATE::VARCHAR AS due_date,
            ABS(ar.OPEN_AMOUNT_USD - {float(payment_amount)}) AS amount_delta,
            CASE
                WHEN ABS(ar.OPEN_AMOUNT_USD - {float(payment_amount)}) <= 50 THEN 'EXACT'
                WHEN ar.OPEN_AMOUNT_USD > {float(payment_amount)} THEN 'UNDERPAY'
                ELSE 'OVERPAY'
            END AS match_type
        FROM {BM}.ar_open_item_vw ar
        WHERE ar.CUSTOMER_ID = '{safe_cust}' {co_filter}
        ORDER BY amount_delta ASC, ar.DAYS_PAST_DUE DESC
        LIMIT 25
    """)


def cash_application_recommendation(payload: dict) -> dict:
    payment_id = str(payload.get("payment_id", "") or "").strip()
    if not payment_id:
        return {"response": "payment_id is required"}

    safe_id = payment_id.replace("'", "''")
    rows = run_query(f"""
        SELECT PAYMENT_ID, CUSTOMER_ID, INVOICE_ID, COMPANY_CODE, PAYMENT_METHOD,
               PAYMENT_AMOUNT_USD, PAYMENT_DATE::VARCHAR AS payment_date,
               CLEARING_DATE, IS_PARTIAL
        FROM {BM}.payment_vw
        WHERE PAYMENT_ID = '{safe_id}'
        LIMIT 1
    """)
    if not rows:
        return {"response": f"Payment {payment_id} not found."}

    p = rows[0]
    customer_id = str(p.get("customer_id") or "")
    amount = float(p.get("payment_amount_usd") or 0)
    company_code = str(p.get("company_code") or "")
    exception = str(payload.get("exception_type") or "UNAPPLIED")
    days = int(float(payload.get("days_unapplied") or 0))

    candidates = cash_application_match_candidates(customer_id, company_code or None, amount)

    policy_rows = search_policies(
        f"cash application {exception} unapplied",
        "CASH_APP",
        limit=5,
    )
    policy_api = policies_for_api(policy_rows)

    match_block = "\n".join(
        f"- {c.get('invoice_id')}: ${float(c.get('open_amount_usd') or 0):,.0f} open "
        f"({c.get('match_type')}, delta ${float(c.get('amount_delta') or 0):,.0f})"
        for c in candidates[:5]
    ) or "No open invoices found for this customer."

    agent_prompt = (
        f"O2C Cash Application Agent — matching action plan.\n\n"
        f"Payment: {payment_id}\n"
        f"Customer: {customer_id}\n"
        f"Amount: ${amount:,.0f}\n"
        f"Exception: {exception}\n"
        f"Days unapplied: {days}\n\n"
        f"Candidate matches:\n{match_block}\n\n"
        f"{CASH_APPLICATION_OUTPUT_FORMAT}"
    )

    def fallback() -> dict:
        base = build_cash_fallback(
            payment_id=payment_id,
            customer_id=customer_id,
            amount=amount,
            exception=exception,
            days=days,
            candidates=candidates,
            policies=policy_api,
        )
        return {"response": base, "candidates": candidates[:10]}

    result = _workbench_live_agent(
        "cash", agent_prompt, fallback,
        policy_query=f"cash application {exception} unapplied",
        policy_domain="CASH_APP",
        cache_entity_id=payment_id,
    )
    if result.get("response"):
        result["response"] = polish_cash_response(
            result["response"],
            policies=result.get("policies") or policy_api,
            payment_id=payment_id,
            customer_id=customer_id,
            amount=amount,
            exception=exception,
            days=days,
            candidates=candidates,
        )
    if "candidates" not in result:
        result["candidates"] = candidates[:10]
    return result


def cash_application_remittance_request(payload: dict) -> dict:
    payment_id = str(payload.get("payment_id", "") or "").strip()
    customer_id = str(payload.get("customer_id", "") or "").strip()
    amount = float(payload.get("payment_amount_usd") or 0)
    pay_date = str(payload.get("payment_date") or "")[:10]

    subject = f"Remittance Advice Request — Payment {payment_id}"
    body = f"""Dear Accounts Payable,

We received a payment of ${amount:,.2f} on {pay_date or 'recent date'} (reference {payment_id}) but cannot match it to specific invoice(s).

Please provide remittance advice showing:
- Invoice number(s) being paid
- Amount applied to each invoice
- Any deductions or short-pay reasons

We will apply the cash within 1 business day of receiving your reply.

Regards,
OrderToCash Cash Application Team"""

    return {
        "to": f"{customer_id.lower()}@customer-ap.com",
        "subject": subject,
        "body": body,
    }


def cash_application_apply(payload: dict) -> dict:
    from FabicApp.db import run_execute

    payment_id = str(payload.get("payment_id", "") or "").strip()
    invoice_id = str(payload.get("invoice_id", "") or "").strip()
    notes = str(payload.get("notes") or "Applied via OrderToCash Cash Application Agent").replace("'", "''")

    if not payment_id:
        return {"ok": False, "message": "payment_id is required"}
    if not invoice_id:
        return {"ok": False, "message": "invoice_id is required"}

    safe_pay = payment_id.replace("'", "''")
    safe_inv = invoice_id.replace("'", "''")

    pay_rows = run_query(f"""
        SELECT PAYMENT_ID, CUSTOMER_ID, COMPANY_CODE, PAYMENT_AMOUNT_USD, IS_PARTIAL
        FROM {BM}.payment_vw
        WHERE PAYMENT_ID = '{safe_pay}'
        LIMIT 1
    """)
    if not pay_rows:
        return {"ok": False, "message": f"Payment {payment_id} not found"}

    pay = pay_rows[0]
    customer_id = str(pay.get("customer_id") or "")
    company_code = str(pay.get("company_code") or "")
    pay_amount = float(pay.get("payment_amount_usd") or 0)

    inv_rows = run_query(f"""
        SELECT INVOICE_ID, CUSTOMER_ID, COMPANY_CODE, OPEN_AMOUNT_USD
        FROM {BM}.ar_open_item_vw
        WHERE INVOICE_ID = '{safe_inv}'
        LIMIT 1
    """)
    if not inv_rows:
        return {"ok": False, "message": f"Open invoice {invoice_id} not found"}

    inv = inv_rows[0]
    if str(inv.get("customer_id") or "") != customer_id:
        return {"ok": False, "message": "Invoice customer does not match payment customer"}
    if company_code and str(inv.get("company_code") or "") != company_code:
        return {"ok": False, "message": "Invoice company code does not match payment"}

    open_amt = float(inv.get("open_amount_usd") or 0)
    is_partial = pay_amount < open_amt * 0.995

    try:
        run_execute(f"""
            UPDATE {DB}.raw_vault.payment
            SET INVOICE_ID = '{safe_inv}',
                CLEARING_DATE = CURRENT_DATE(),
                IS_PARTIAL = {str(is_partial).upper()},
                POSTING_TIMESTAMP = CURRENT_TIMESTAMP(),
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE PAYMENT_ID = '{safe_pay}'
        """)
        promote_staging_to_mart("payment")
        status = "partially applied" if is_partial else "fully applied"
        return {
            "ok": True,
            "message": f"Payment {payment_id} {status} to {invoice_id}. {notes}",
            "is_partial": is_partial,
        }
    except Exception as exc:
        log.warning("Cash apply failed", exc_info=True)
        return {"ok": False, "message": f"Failed to apply payment: {exc}"}


def dispute_agent_queue(customer_id: str | None = None) -> list[dict]:
    cache_key = f"dispute_queue:v5:{customer_id or 'all'}"
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    cust_filter = ""
    if customer_id:
        safe_cust = str(customer_id).replace("'", "''")
        cust_filter = f" AND d.CUSTOMER_ID = '{safe_cust}'"
    row_limit = 200 if customer_id else 50
    sql = f"""
    WITH ranked AS (
        SELECT
            d.DISPUTE_ID AS dispute_id,
            d.CUSTOMER_ID AS customer_id,
            COALESCE(cu.CUSTOMER_NAME, cu_any.CUSTOMER_NAME) AS customer_name,
            d.INVOICE_ID AS invoice_id,
            d.COMPANY_CODE AS company_code,
            d.DISPUTE_REASON AS dispute_reason,
            d.DISPUTE_STATUS AS dispute_status,
            d.DISPUTED_AMOUNT AS disputed_amount,
            d.OPENED_DATE::VARCHAR AS opened_date,
            DATEDIFF('day', d.OPENED_DATE, CURRENT_DATE()) AS resolution_days,
            d.OWNER AS owner,
            d.DISPUTED_AMOUNT * (1 + DATEDIFF('day', d.OPENED_DATE, CURRENT_DATE()) / 30.0) AS priority_score,
            CASE
                WHEN DATEDIFF('day', d.OPENED_DATE, CURRENT_DATE()) >= 30 OR d.DISPUTED_AMOUNT > 50000 THEN 'ACT_NOW'
                WHEN DATEDIFF('day', d.OPENED_DATE, CURRENT_DATE()) >= 14 THEN 'PLAN_THIS_WEEK'
                ELSE 'MONITOR'
            END AS action_group,
            ROW_NUMBER() OVER (
                PARTITION BY d.DISPUTE_ID
                ORDER BY d.DISPUTED_AMOUNT DESC NULLS LAST, d.OPENED_DATE DESC NULLS LAST
            ) AS rn
        FROM {BM}.dispute_vw d
        LEFT JOIN {BM}.customer_vw cu
          ON cu.CUSTOMER_ID = d.CUSTOMER_ID
         AND cu.COMPANY_CODE = d.COMPANY_CODE
        LEFT JOIN (
            SELECT CUSTOMER_ID, MAX(CUSTOMER_NAME) AS CUSTOMER_NAME
            FROM {BM}.customer_vw
            GROUP BY CUSTOMER_ID
        ) cu_any
          ON cu_any.CUSTOMER_ID = d.CUSTOMER_ID
        WHERE UPPER(COALESCE(d.DISPUTE_STATUS, '')) NOT IN ('RESOLVED', 'CLOSED', 'CANCELLED', 'WRITTEN_OFF')
          {cust_filter}
    )
    SELECT
        dispute_id, customer_id, customer_name, invoice_id, company_code, dispute_reason,
        dispute_status, disputed_amount, opened_date, resolution_days, owner,
        priority_score, action_group
    FROM ranked
    WHERE rn = 1
    ORDER BY priority_score DESC
    LIMIT {row_limit}
    """
    rows = run_query(sql)
    rows = _enrich_copilot_rows(rows)
    _agent_cache_set(cache_key, rows)
    return rows


def dispute_agent_recommendation(payload: dict) -> dict:
    if payload.get("risk_id"):
        return _dispute_predicted_recommendation(payload)

    dispute_id = str(payload.get("dispute_id", "") or "").strip()
    if not dispute_id:
        return {"response": "dispute_id is required", "agent_used": False, "source": "error"}

    safe_id = dispute_id.replace("'", "''")
    rows = run_query(f"""
        SELECT DISPUTE_ID, CUSTOMER_ID, INVOICE_ID, COMPANY_CODE, DISPUTE_REASON,
               DISPUTE_STATUS, DISPUTED_AMOUNT, OPENED_DATE::VARCHAR AS opened_date, OWNER
        FROM {BM}.dispute_vw
        WHERE DISPUTE_ID = '{safe_id}'
        LIMIT 1
    """)
    if not rows:
        return {"response": f"Dispute {dispute_id} not found."}

    d = rows[0]
    reason = str(d.get("dispute_reason") or "UNKNOWN")
    amount = float(d.get("disputed_amount") or 0)
    customer_id = str(d.get("customer_id") or "")
    invoice_id = str(d.get("invoice_id") or "")
    status = str(d.get("dispute_status") or "OPEN")
    action_group = str(payload.get("action_group") or "ACT_NOW")

    inv_rows = run_query(f"""
        SELECT INVOICE_ID, GROSS_AMOUNT, INVOICE_STATUS, INVOICE_DATE::VARCHAR AS invoice_date
        FROM {BM}.ar_invoice_vw
        WHERE INVOICE_ID = '{invoice_id.replace("'", "''")}'
        LIMIT 1
    """) if invoice_id else []
    inv = inv_rows[0] if inv_rows else {}

    policy_rows = search_policies(
        f"dispute resolution {reason}",
        "DISPUTE",
        limit=5,
    )
    policy_api = policies_for_api(policy_rows)

    agent_prompt = (
        f"O2C Dispute Agent — resolution action plan.\n\n"
        f"Dispute: {dispute_id}\n"
        f"Customer: {customer_id}\n"
        f"Invoice: {invoice_id}\n"
        f"Reason: {reason}\n"
        f"Amount: ${amount:,.0f}\n"
        f"Status: {status}\n"
        f"Priority: {action_group.replace('_', ' ')}\n\n"
        f"{DISPUTE_OUTPUT_FORMAT}"
    )

    def fallback() -> dict:
        base = build_dispute_fallback(
            dispute_id=dispute_id,
            customer_id=customer_id,
            invoice_id=invoice_id,
            reason=reason,
            amount=amount,
            status=status,
            action_group=action_group,
            inv=inv,
            policies=policy_api,
        )
        return {"response": base}

    result = _workbench_live_agent(
        "dispute", agent_prompt, fallback,
        policy_query=f"dispute resolution {reason}",
        policy_domain="DISPUTE",
        cache_entity_id=dispute_id,
    )
    if result.get("response"):
        result["response"] = polish_dispute_response(
            result["response"],
            policies=result.get("policies") or policy_api,
            dispute_id=dispute_id,
            customer_id=customer_id,
            invoice_id=invoice_id,
            reason=reason,
            amount=amount,
            status=status,
            inv=inv,
            action_group=action_group,
        )
    return result


def dispute_agent_preventive_outreach(payload: dict) -> dict:
    customer_id = str(payload.get("customer_id") or "").strip()
    customer = str(payload.get("customer_name") or customer_id)
    invoice_id = str(payload.get("invoice_id") or "n/a")
    reason = str(payload.get("predicted_dispute_reason") or "billing discrepancy")
    exposure = float(payload.get("exposure_amount") or 0)

    subject = f"Proactive invoice review — {invoice_id} (prevent dispute)"
    body = f"""Dear Accounts Payable,

We are proactively sharing billing documentation for invoice {invoice_id} ({customer}) to reduce the risk of a {reason.lower()} dispute.

Exposure at risk: ${exposure:,.0f}

Attached / available on request:
- Invoice detail and line items
- Proof of delivery / service completion
- Pricing and contract references

Please confirm receipt and let us know within 3 business days if anything appears incorrect before a formal dispute is opened.

Regards,
OrderToCash AR / Dispute Prevention Team"""

    return {
        "to": f"{customer_id.lower()}@customer-ap.com",
        "subject": subject,
        "body": body,
    }


def dispute_agent_request_info(payload: dict) -> dict:
    dispute_id = str(payload.get("dispute_id", "") or "").strip()
    customer_id = str(payload.get("customer_id", "") or "").strip()
    invoice_id = str(payload.get("invoice_id", "") or "").strip()
    reason = str(payload.get("dispute_reason") or "billing discrepancy")
    amount = float(payload.get("disputed_amount") or 0)

    subject = f"Dispute {dispute_id} — Information Request ({invoice_id})"
    body = f"""Dear Accounts Payable,

We are reviewing dispute {dispute_id} related to invoice {invoice_id} for ${amount:,.2f} ({reason}).

Please reply with:
- Dispute reference / claim number
- Supporting documentation (invoice copy, delivery proof, pricing agreement)
- Expected resolution or credit amount

We will respond within 3 business days once documentation is received.

Regards,
OrderToCash Dispute Resolution Team"""

    return {
        "to": f"{customer_id.lower()}@customer-ap.com",
        "subject": subject,
        "body": body,
    }


def dispute_agent_update_status(payload: dict) -> dict:
    from FabicApp.db import run_execute

    dispute_id = str(payload.get("dispute_id", "") or "").strip()
    new_status = str(payload.get("dispute_status") or "IN_REVIEW").strip().upper()
    notes = str(payload.get("notes") or "Status updated via OrderToCash Dispute Agent").replace("'", "''")
    safe_id = dispute_id.replace("'", "''")

    if not dispute_id:
        return {"ok": False, "message": "dispute_id is required"}

    try:
        run_execute(f"""
            UPDATE {DB}.raw_vault.dispute_case
            SET DISPUTE_STATUS = '{new_status}',
                OWNER = COALESCE(OWNER, 'OrderToCash Agent'),
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE DISPUTE_ID = '{safe_id}'
        """)
        promote_staging_to_mart("collections")
        return {"ok": True, "message": f"Dispute {dispute_id} moved to {new_status}. {notes}"}
    except Exception as exc:
        log.warning("Dispute status update failed", exc_info=True)
        return {"ok": False, "message": f"Failed to update dispute: {exc}"}


def dispute_agent_resolve(payload: dict) -> dict:
    from FabicApp.db import run_execute

    dispute_id = str(payload.get("dispute_id", "") or "").strip()
    resolution_type = str(payload.get("resolution_type") or "FULL").strip().upper()
    resolved_amount = float(payload.get("resolved_amount") or payload.get("disputed_amount") or 0)
    notes = str(payload.get("notes") or "Resolved via OrderToCash Dispute Agent").replace("'", "''")
    safe_id = dispute_id.replace("'", "''")

    if not dispute_id:
        return {"ok": False, "message": "dispute_id is required"}

    status = "RESOLVED"
    if resolution_type == "WRITTEN_OFF":
        status = "WRITTEN_OFF"
        resolved_amount = 0

    try:
        run_execute(f"""
            UPDATE {DB}.raw_vault.dispute_case
            SET DISPUTE_STATUS = '{status}',
                RESOLVED_AMOUNT = {resolved_amount},
                RESOLVED_DATE = CURRENT_DATE(),
                RESOLUTION_DAYS = DATEDIFF('day', OPENED_DATE, CURRENT_DATE()),
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE DISPUTE_ID = '{safe_id}'
        """)
        promote_staging_to_mart("collections")
        return {
            "ok": True,
            "message": f"Dispute {dispute_id} {status.lower()} — ${resolved_amount:,.2f}. {notes}",
        }
    except Exception as exc:
        log.warning("Dispute resolve failed", exc_info=True)
        return {"ok": False, "message": f"Failed to resolve dispute: {exc}"}


def _agent_cache_get(key: str, ttl: float | None = None) -> object | None:
    now = time.time()
    effective_ttl = _AGENT_CACHE_TTL if ttl is None else ttl
    with _AGENT_CACHE_LOCK:
        ent = _AGENT_CACHE.get(key)
        if not ent:
            return None
        ts, val = ent
        if now - ts > effective_ttl:
            _AGENT_CACHE.pop(key, None)
            return None
        return val


def _agent_cache_set(key: str, val: object) -> None:
    with _AGENT_CACHE_LOCK:
        _AGENT_CACHE[key] = (time.time(), copy.deepcopy(val))


def _queue_payload(row: dict) -> dict:
    """Normalize Fabric row keys for agent payloads."""
    out: dict = {}
    for k, v in row.items():
        out[str(k).lower()] = v
    return out


def agent_workbench_snapshot(annual_interest_rate_pct: float = 8.0) -> dict:
    rate = max(1.0, min(float(annual_interest_rate_pct or 8.0), 30.0))
    cache_key = f"wb_snapshot:{rate}"
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    bill_q = billing_agent_queue()
    coll_q = collections_agent_queue(rate)
    cash_q = cash_application_agent_queue()
    disp_q = dispute_agent_queue()
    proactive_q = proactive_dispute_queue()

    recommendations: dict[str, dict] = {}

    last_cycle = _agent_cache_get(f"wb_cycle:{rate}")

    result = {
        "billing_queue": bill_q,
        "predicted_dispute_queue": proactive_q,
        "collections_queue": coll_q,
        "cash_queue": cash_q,
        "dispute_queue": disp_q,
        "recommendations": recommendations,
        "last_autonomous_cycle": last_cycle,
        "cached_at": time.time(),
    }
    _agent_cache_set(cache_key, result)
    return result


def agent_workbench_autonomous_cycle(
    annual_interest_rate_pct: float = 8.0,
    max_collections: int = 3,
    max_cash: int = 3,
    max_disputes: int = 2,
) -> dict:
    """Run collections auto-flow, cash apply, and dispute triage without manual steps."""
    rate = max(1.0, min(float(annual_interest_rate_pct or 8.0), 30.0))
    started = time.time()
    activity: list[dict] = []

    coll_q = collections_agent_queue(rate)
    act_now = [r for r in coll_q if str(_val(r, "ACTION_GROUP", "action_group") or "").upper() == "ACT_NOW"]
    collections_runs: list[dict] = []
    for row in act_now[:max_collections]:
        payload = _queue_payload(row)
        name = str(payload.get("customer_name") or payload.get("customer_id") or "")
        result = collections_agent_auto_flow(payload)
        collections_runs.append({
            "customer_id": payload.get("customer_id"),
            "customer_name": name,
            "ok": result.get("ok"),
            "message": result.get("message"),
            "steps": result.get("steps", []),
        })
        activity.append({
            "agent": "collections",
            "action": "auto_flow",
            "target": name,
            "status": "completed" if result.get("ok") else "failed",
            "detail": str(result.get("message") or ""),
        })

    cash_q = cash_application_agent_queue()
    cash_runs: list[dict] = []
    for row in cash_q[:max_cash]:
        payload = _queue_payload(row)
        payment_id = str(payload.get("payment_id") or "")
        customer_id = str(payload.get("customer_id") or "")
        amount = float(payload.get("payment_amount_usd") or 0)
        invoice_id = str(payload.get("invoice_id") or "")
        company = str(payload.get("company_code") or "") or None

        if invoice_id:
            apply_result = cash_application_apply({**payload, "invoice_id": invoice_id})
            cash_runs.append({"payment_id": payment_id, **apply_result})
            activity.append({
                "agent": "cash",
                "action": "apply_payment",
                "target": payment_id,
                "status": "completed" if apply_result.get("ok") else "failed",
                "detail": str(apply_result.get("message") or ""),
            })
            continue

        cands = cash_application_match_candidates(customer_id, company, amount)
        exact = next((c for c in cands if str(c.get("match_type") or "").upper() == "EXACT"), None)
        if exact:
            apply_result = cash_application_apply({
                **payload,
                "invoice_id": str(exact.get("invoice_id") or ""),
                "notes": "Auto-applied by agent (exact match)",
            })
            cash_runs.append({"payment_id": payment_id, **apply_result})
            activity.append({
                "agent": "cash",
                "action": "auto_apply",
                "target": payment_id,
                "status": "completed" if apply_result.get("ok") else "failed",
                "detail": str(apply_result.get("message") or ""),
            })
        else:
            draft = cash_application_remittance_request(payload)
            cash_runs.append({
                "payment_id": payment_id,
                "ok": True,
                "message": f"Remittance request sent to {draft.get('to')}",
            })
            activity.append({
                "agent": "cash",
                "action": "request_remittance",
                "target": payment_id,
                "status": "completed",
                "detail": str(draft.get("subject") or "Remittance requested"),
            })

    disp_q = dispute_agent_queue()
    disp_act = [r for r in disp_q if str(_val(r, "ACTION_GROUP", "action_group") or "").upper() == "ACT_NOW"]
    dispute_runs: list[dict] = []
    for row in disp_act[:max_disputes]:
        payload = _queue_payload(row)
        dispute_id = str(payload.get("dispute_id") or "")
        review = dispute_agent_update_status({
            **payload,
            "dispute_status": "IN_REVIEW",
            "notes": "Auto-assigned by OrderToCash dispute agent",
        })
        dispute_runs.append({"dispute_id": dispute_id, **review})
        activity.append({
            "agent": "disputes",
            "action": "mark_in_review",
            "target": dispute_id,
            "status": "completed" if review.get("ok") else "failed",
            "detail": str(review.get("message") or ""),
        })

    with _AGENT_CACHE_LOCK:
        _AGENT_CACHE.clear()

    result = {
        "ok": True,
        "automated": True,
        "annual_interest_rate_pct": rate,
        "duration_ms": int((time.time() - started) * 1000),
        "collections_runs": collections_runs,
        "cash_runs": cash_runs,
        "dispute_runs": dispute_runs,
        "activity": activity,
        "message": (
            f"Agent cycle complete: {len(collections_runs)} collections, "
            f"{len(cash_runs)} cash, {len(dispute_runs)} disputes."
        ),
    }
    _agent_cache_set(f"wb_cycle:{rate}", result)
    return result
