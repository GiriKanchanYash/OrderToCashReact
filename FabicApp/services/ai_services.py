from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import re

from app.db import qcol, qview, run_query
from app.services import text_to_sql
from app.services import onelake_writer

from openai import AzureOpenAI

logger = logging.getLogger("orderlens.ai_service")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini").strip()

_aoai_client: AzureOpenAI | None = None


def _get_aoai_client() -> AzureOpenAI:
    global _aoai_client
    if _aoai_client is None:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY missing from .env")
        _aoai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _aoai_client


def _llm_complete(prompt: str) -> str:
    """Drop-in replacement for the Snowflake original's
    `select snowflake.cortex.complete(%s, %s) as response` call site."""
    client = _get_aoai_client()
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900,
    )
    return (resp.choices[0].message.content or "").strip()

COPILOT_OUTPUT_DIRECTIVE = (
    "Format the answer with EXACTLY these two markdown sections and nothing else before them:\n"
    "**Descriptive:** What the data shows — concrete numbers and evidence from the facts.\n"
    "**Prescriptive:** Start with 'Here are the bullet points with specific findings, concrete actions, and explanations:' "
    "then list 4-5 bullets. Each bullet MUST use this pattern:\n"
    "- **Topic:** brief finding. **Action:** specific next step. **Why it matters:** why this helps sales ops / planning.\n"
    "Do not invent metrics that are not in the facts. Start directly with **Descriptive:**.\n"
)

AGENT_OUTPUT_DIRECTIVE = (
    "WORKBENCH MODE — respond in one pass with NO tools.\n"
    "Return ONLY these HTML section headers in this exact order. "
    "Do not use inline bold except for section headers. No markdown tables, no emojis, no preamble.\n\n"
    "<strong>SITUATION ASSESSMENT</strong>\n"
    "2-3 sentences with concrete facts from the JSON (ids, customer/material, amounts, ages, reasons). "
    "State the problem clearly — do not leave this section empty.\n\n"
    "<strong>IMMEDIATE ACTIONS (ranked)</strong>\n"
    "Exactly 3 numbered lines. Each line MUST start with a real ALL-CAPS action name, then a colon, then detail + Next step.\n"
    "GOOD examples:\n"
    "1. SUPPLY CHECK: Confirm ATP and open schedule qty. Next step: Fulfillment lead diagnoses today.\n"
    "2. CUSTOMER COMMIT: Align revised delivery date. Next step: CS confirms within 48h.\n"
    "3. EXPEDITE / TRANSFER: Evaluate alternate plant or expedite PO. Next step: Planner decides today.\n"
    "NEVER write the literal words 'ACTION LABEL'. Never use Title Case action names — use ALL CAPS names.\n\n"
    "<strong>CONTEXT</strong>\n"
    "2-4 short bullets with supporting facts from the JSON. Do not leave empty.\n\n"
    "<strong>TIMELINE</strong>\n"
    "Exactly 3 bullets: Today / +48h / +7d tied to the actions above. Do not leave empty.\n"
)


def _num(row: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return default


def _str(row: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return default

def recommend(agent_key: str, context: dict) -> dict:
    """Cortex COMPLETE recommendation in O2C workbench section format."""
    names = {
        "fulfillment": "Order Lens Fulfillment Agent",
        "inventory": "Order Lens Inventory Agent",
        "forecast": "Order Lens Forecast Agent",
    }
    prompts = {
        "fulfillment": (
            f"{AGENT_OUTPUT_DIRECTIVE}\n"
            "You are the Order Lens Fulfillment Agent. Focus on backlog aging, open value, "
            "delivery risk, and backorder reasons.\n"
            f"Backlog item JSON: {json.dumps(context, default=str)[:4000]}"
        ),
        "inventory": (
            f"{AGENT_OUTPUT_DIRECTIVE}\n"
            "You are the Order Lens Inventory Agent. Focus on reorder shortage, stock vs reorder point, "
            "and replenishment lead time.\n"
            f"Inventory breach JSON: {json.dumps(context, default=str)[:4000]}"
        ),
        "forecast": (
            f"{AGENT_OUTPUT_DIRECTIVE}\n"
            "You are the Order Lens Forecast Agent. Focus on variance category, APE, and method bias.\n"
            f"Forecast outlier JSON: {json.dumps(context, default=str)[:4000]}"
        ),
    }
    prompt = prompts.get(agent_key, prompts["fulfillment"])
    fallback = _template_recommend(agent_key, context)
    fallback["agent_name"] = names.get(agent_key, "Order Lens Agent")
    try:
        text = _llm_complete(prompt)
        text = _polish_agent_response(text, agent_key, context)
        if text.strip():
            return {
                "response": text,
                "source": "azure_openai",
                "agent_used": True,
                "agent_name": names.get(agent_key, "Order Lens Agent"),
                "cached": False,
            }
    except Exception as e:
        return {
            **fallback,
            "agent_used": False,
            "agent_error": str(e),
            "source": "error",
            "response": _polish_agent_response(fallback.get("response") or "", agent_key, context),
        }
    return {
        **fallback,
        "agent_used": False,
        "response": _polish_agent_response(fallback.get("response") or "", agent_key, context),
    }


def _parse_agent_sections(text: str) -> dict[str, str]:
    """Split workbench HTML into canonical section bodies."""
    raw = _normalize_agent_response(text or "")
    parts = re.split(r"<strong>([^<]+)</strong>", raw, flags=re.I)
    sections: dict[str, str] = {}
    canon_map = {
        "SITUATION": "SITUATION ASSESSMENT",
        "SITUATION ASSESSMENT": "SITUATION ASSESSMENT",
        "IMMEDIATE ACTIONS": "IMMEDIATE ACTIONS (ranked)",
        "IMMEDIATE ACTIONS (RANKED)": "IMMEDIATE ACTIONS (ranked)",
        "ACTIONS": "IMMEDIATE ACTIONS (ranked)",
        "CONTEXT": "CONTEXT",
        "ORDER CONTEXT": "CONTEXT",
        "TIMELINE": "TIMELINE",
    }
    for i in range(1, len(parts), 2):
        title = re.sub(r"\s+", " ", (parts[i] or "").strip()).upper()
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        key = None
        for prefix, canon in canon_map.items():
            if title == prefix or title.startswith(prefix):
                key = canon
                break
        if key:
            # Prefer non-empty body if section appears twice
            if body or key not in sections:
                sections[key] = body
    return sections


def _fix_action_labels(actions_body: str) -> str:
    """Replace placeholder ACTION LABEL / Title Case with ALL-CAPS action names."""
    lines_out: list[str] = []
    for line in (actions_body or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if not m:
            lines_out.append(stripped)
            continue
        num, rest = m.group(1), m.group(2).strip()
        # Strip markdown bold around labels
        rest = re.sub(r"^\*\*(.+?)\*\*:?\s*", r"\1: ", rest)
        rest = re.sub(r"^<strong>([^<]+)</strong>:?\s*", r"\1: ", rest, flags=re.I)

        labeled = re.match(r"^([A-Za-z][A-Za-z0-9\s/&-]{1,40}):\s*(.+)$", rest)
        if labeled:
            label, detail = labeled.group(1).strip(), labeled.group(2).strip()
            if re.fullmatch(r"ACTION\s*LABELS?", label, flags=re.I) or label.lower() in {
                "action", "actions", "step", "next step",
            }:
                # Promote first phrase of detail to the label
                detail_m = re.match(
                    r"^([A-Za-z][A-Za-z0-9\s/&-]{2,40}?)\s*[–—\-:]\s*(.+)$",
                    detail,
                )
                if detail_m:
                    label = detail_m.group(1).strip()
                    detail = detail_m.group(2).strip()
                else:
                    words = detail.split()
                    label = " ".join(words[:3]) if words else "FOLLOW UP"
                    detail = " ".join(words[3:]) if len(words) > 3 else detail
            label = re.sub(r"\s+", " ", label).upper()
            if "next step:" not in detail.lower():
                detail = f"{detail} Next step: Owner executes within 48h."
            lines_out.append(f"{num}. {label}: {detail}")
        else:
            words = rest.split()
            label = " ".join(words[:3]).upper() if words else "FOLLOW UP"
            detail = " ".join(words[3:]) if len(words) > 3 else rest
            if "next step:" not in detail.lower():
                detail = f"{detail} Next step: Owner executes within 48h."
            lines_out.append(f"{num}. {label}: {detail}")
    return "\n".join(lines_out)


def _polish_agent_response(text: str, agent_key: str, context: dict) -> str:
    """Ensure all workbench sections are present, filled, and action labels are real."""
    template = _template_recommend(agent_key, context)["response"]
    tmpl_sections = _parse_agent_sections(template)
    live = _parse_agent_sections(text)
    order = [
        "SITUATION ASSESSMENT",
        "IMMEDIATE ACTIONS (ranked)",
        "CONTEXT",
        "TIMELINE",
    ]
    merged: dict[str, str] = {}
    for key in order:
        body = (live.get(key) or "").strip()
        # If situation empty but context has narrative prose, promote it
        if key == "SITUATION ASSESSMENT" and len(body) < 40:
            ctx = (live.get("CONTEXT") or "").strip()
            if len(ctx) > 60 and not ctx.lstrip().startswith("-"):
                body = ctx
        if key == "IMMEDIATE ACTIONS (ranked)":
            body = _fix_action_labels(body)
            # Require at least 2 numbered actions
            if len(re.findall(r"^\d+\.", body, flags=re.M)) < 2:
                body = tmpl_sections.get(key, body)
            else:
                body = _fix_action_labels(body)
        if len(body) < 20:
            body = tmpl_sections.get(key, body)
        merged[key] = body.strip()

    # If CONTEXT still equals situation prose, use template bullets
    if merged["CONTEXT"] and merged["CONTEXT"] == merged["SITUATION ASSESSMENT"]:
        merged["CONTEXT"] = tmpl_sections.get("CONTEXT", merged["CONTEXT"])

    parts = []
    for key in order:
        parts.append(f"<strong>{key}</strong>\n{merged[key]}")
    return "\n\n".join(parts).strip()


def _normalize_agent_response(text: str) -> str:
    """Ensure ALL-CAPS HTML section headers so AgentRecommendation colors match O2C."""
    raw = (text or "").strip()
    if not raw:
        return raw

    # Canonical O2C workbench section titles (matched case-insensitively)
    section_map = [
        (r"SITUATION(?:\s+ASSESSMENT)?", "SITUATION ASSESSMENT"),
        (r"IMMEDIATE\s+ACTIONS(?:\s*\(\s*RANKED\s*\))?|ACTIONS(?:\s*\(\s*RANKED\s*\))?", "IMMEDIATE ACTIONS (ranked)"),
        (r"(?:ORDER\s+)?CONTEXT", "CONTEXT"),
        (r"TIMELINE", "TIMELINE"),
    ]

    # Convert markdown **Section Title** only when it looks like a known header
    def _md_section(m: re.Match) -> str:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        upper = title.upper()
        for pattern, canon in section_map:
            if re.fullmatch(pattern, upper):
                return f"<strong>{canon}</strong>"
        return m.group(0)  # leave inline bold alone

    raw = re.sub(r"\*\*([^*\n]+)\*\*", _md_section, raw)

    def _fix_strong(m: re.Match) -> str:
        inner = re.sub(r"\s+", " ", m.group(1)).strip()
        upper = inner.upper()
        for pattern, canon in section_map:
            if re.fullmatch(pattern, upper):
                return f"<strong>{canon}</strong>"
        return m.group(0)

    raw = re.sub(r"<strong>([^<]+)</strong>", _fix_strong, raw, flags=re.I)

    for pattern, canon in section_map:
        raw = re.sub(
            rf"(?m)^(?:#+\s*)?(?:<strong>)?({pattern})(?:</strong>)?\s*$",
            f"<strong>{canon}</strong>",
            raw,
            flags=re.I,
        )
    # Flatten remaining inline bold so it doesn't create fake sections in the UI
    raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
    return raw


def _template_recommend(agent_key: str, context: dict) -> dict:
    if agent_key == "inventory":
        mat = context.get("material_description") or context.get("material_key") or "material"
        plant = context.get("plant_name") or "plant"
        shortage = context.get("shortage_qty") or context.get("reorder_shortage_quantity") or 0
        stock = context.get("stock_qty") or 0
        reorder = context.get("reorder_point") or 0
        lead = context.get("planned_delivery_days") or 0
        text = f"""<strong>SITUATION ASSESSMENT</strong>
{mat} at {plant} is below reorder point (stock {stock}, reorder point {reorder}, shortage ≈ {shortage}).

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. REPLENISH: Raise purchase / STO for shortage quantity. Next step: MRP controller confirms today.
2. LEAD TIME CHECK: Confirm planned delivery days ({lead}d) and expedite if ACT_NOW. Next step: Buyer updates ETA within 24h.
3. SUBSTITUTE / TRANSFER: Check alternate plants or substitute materials. Next step: Planner reviews options today.

<strong>CONTEXT</strong>
- Material / plant driven by inventory_reorder_point_breaches
- Shortage quantity: {shortage}

<strong>TIMELINE</strong>
- Today: raise replenishment request
- +48h: confirm supply commitment
- +7d: verify stock recovery vs reorder point"""
    elif agent_key == "forecast":
        mat = context.get("material_description") or context.get("material_key") or "material"
        cat = context.get("variance_category") or "High Variance"
        ape = context.get("ape") or context.get("absolute_percentage_error") or 0
        plant = context.get("plant_name") or "plant"
        period = context.get("forecast_period") or ""
        text = f"""<strong>SITUATION ASSESSMENT</strong>
Forecast outlier for {mat} at {plant}: {cat} (APE ≈ {ape}%, period {period}).

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. DEMAND REVIEW: Compare recent actuals vs forecast assumptions. Next step: Demand planner investigates today.
2. METHOD ADJUST: Revisit forecast method / bias if pattern persists. Next step: Planning updates model this week.
3. ALIGN S&OP: Align sales ops and planning on near-term demand. Next step: Joint review before next cycle.

<strong>CONTEXT</strong>
- Source: forecasting_outlier_analysis
- Variance category: {cat}

<strong>TIMELINE</strong>
- Today: investigate outlier drivers
- This week: revise plan / method
- Next cycle: monitor MAPE improvement"""
    else:
        order_id = context.get("sales_order_id") or "order"
        age = context.get("order_age_days") or 0
        value = context.get("open_value") or 0
        customer = context.get("customer_name") or "customer"
        reasons = context.get("backorder_reasons") or "n/a"
        text = f"""<strong>SITUATION ASSESSMENT</strong>
Open backlog on {order_id} for {customer} (age {age} days, open value {value}). Backorder reasons: {reasons}.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. SUPPLY CHECK: Confirm plant constraints and open schedule quantity. Next step: Fulfillment lead diagnoses today.
2. CUSTOMER COMMIT: Align on revised delivery date with customer. Next step: CS confirms within 48h.
3. BLOCK REVIEW: Escalate if delivery / billing / credit blocks exist on header. Next step: Ops clears blockers this week.

<strong>CONTEXT</strong>
- Customer: {customer}
- Backorder reason: {reasons}
- Open value: {value}
- Order age: {age} days

<strong>TIMELINE</strong>
- Today: diagnose supply constraint and confirm ATP
- +48h: commit revised date with customer
- +7d: clear open quantity or escalate shortage"""
    return {
        "response": text,
        "source": "template",
        "cached": False,
        "agent_used": False,
        "agent_name": {
            "fulfillment": "Order Lens Fulfillment Agent",
            "inventory": "Order Lens Inventory Agent",
            "forecast": "Order Lens Forecast Agent",
        }.get(agent_key, "Order Lens Agent"),
    }


def _next_int_id(table_view: str, id_column: str) -> int:
    """These Delta tables have no identity/auto-increment column (confirmed
    via notebook schema dump - plain nullable 32-bit IntegerType), so the
    app computes the next id itself from the existing (working) read-only
    SQL endpoint. NOTE: a millisecond-epoch id, as an earlier version of this
    code used, overflows a 32-bit IntegerType column - must stay a small int.
    Best-effort only: under concurrent writes from multiple app instances
    this can race and collide, same as any client-side id generation without
    a DB-side sequence - acceptable at this tool's traffic level."""
    rows = run_query(f"select MAX({id_column}) as m from {table_view}")
    current_max = (rows[0].get("m") if rows else None) or 0
    return int(current_max) + 1


def _onelake_write_or_readonly(kind: str, *args, **kwargs) -> dict:
    """kind is 'append' or 'delete' -> onelake_writer.append_rows / delete_rows.
    Degrades gracefully (ok: False) instead of raising when OneLake isn't
    configured yet or the write itself fails, so the UI doesn't 503; any
    programming error elsewhere still surfaces normally."""
    try:
        if kind == "append":
            onelake_writer.append_rows(*args, **kwargs)
        else:
            onelake_writer.delete_rows(*args, **kwargs)
        return {"ok": True}
    except onelake_writer.OneLakeWriteError as e:
        logger.warning("OneLake write skipped/failed: %s", e)
        return {"ok": False, "reason": "onelake_write_failed", "message": str(e)}

def quick_analyses() -> list[dict]:
    return [
        {
            "key": "backlog_overview",
            "title": "Backlog Overview",
            "desc": "Summarize open backlog value, aging, and top constrained orders.",
            "question": "Summarize current order backlog by open value and aging. Highlight top risk orders.",
        },
        {
            "key": "reorder_hotspots",
            "title": "Reorder Hotspots",
            "desc": "Materials and plants below reorder point.",
            "question": "Which materials and plants have the largest reorder shortages right now?",
        },
        {
            "key": "forecast_accuracy",
            "title": "Forecast Accuracy",
            "desc": "Recent forecast MAPE and outlier categories.",
            "question": "How accurate are recent demand forecasts and where are the biggest outliers?",
        },
        {
            "key": "dealer_performance",
            "title": "Dealer Performance",
            "desc": "Top dealers by order value and fulfillment rate.",
            "question": "Show top dealers by order value and call out low fulfillment performers.",
        },
    ]


def run_analysis(key: str) -> dict:
    defs = {d["key"]: d for d in quick_analyses()}
    item = defs.get(key) or {
        "key": key or "custom",
        "title": key or "Analysis",
        "question": key or "Summarize sales ops health",
    }
    result = copilot_chat(item["question"], persona="sales_ops")
    result["key"] = item["key"]
    if not result.get("metrics"):
        result["metrics"] = {"summary": item.get("title") or item["key"]}
    return result


def copilot_chat(question: str, persona: str = "sales_ops") -> dict:
    """Return O2C-shaped Descriptive (blue) + Prescriptive (green) answer."""
    q = (question or "").strip()
    if not q:
        return _finalize_copilot({
            "key": "custom",
            "metrics": {},
            "rows": [],
            "sql": "",
            "descriptive": "Ask a sales ops, inventory, or forecast question.",
            "prescriptive": (
                "- **Pick a starting point:** Use a quick analysis tile or ask about backlog, reorder, or forecast. "
                "**Action:** Click Backlog Overview. **Why it matters:** Establishes a clear baseline."
            ),
            "source": "empty",
        }, q)

    try:
        norm = " ".join(q.lower().split())
        _onelake_write_or_readonly(
            "append",
            "ONELAKE_GENIE_QUESTION_HISTORY_PATH",
            [{
                "NORMALIZED_QUERY": norm,
                "TYPE": "ANALYTICS",
                "PERSONA": persona.upper(),
                "USER_NAME": "app",
                "FREQUENCY": 1,
                "LAST_ASKED_AT": onelake_writer.utcnow(),
            }],
        )
    except Exception:
        pass

    digest = _facts_digest()

    generated_sql = ""
    t2sql_error = ""
    top_rows = None
    try:
        result = text_to_sql.run_text_to_sql(q, _llm_complete)
        generated_sql = result["sql"]
        top_rows = result["rows"]
    except text_to_sql.SqlGenerationError as e:
        # Model declined or couldn't be reached - fall back silently to the
        # keyword-based sample rows below, same as before this feature existed.
        t2sql_error = str(e)
    except text_to_sql.SqlValidationError as e:
        # Model produced SQL that failed a safety check - never executed.
        # Logged for visibility, then fall back the same way.
        logger.warning("Text-to-SQL validation rejected generated SQL for %r: %s", q, e)
        t2sql_error = str(e)
    except Exception as e:
        logger.error("Text-to-SQL pipeline failed for %r: %s", q, e)
        t2sql_error = str(e)

    if top_rows is None:
        top_rows = _sample_rows_for_question(q)

    prompt = (
        "You are Order Lens Copilot for sales operations planning.\n"
        f"{COPILOT_OUTPUT_DIRECTIVE}\n"
        f"Facts:\n{digest}\n\n"
        f"Sample rows (JSON):\n{json.dumps(top_rows[:5], default=str)[:3000]}\n\n"
        f"Question: {q}"
    )
    try:
        text = _llm_complete(prompt)
        if text.strip():
            descriptive, prescriptive = _split_desc_pres(text)
            return _finalize_copilot({
                "key": "custom",
                "metrics": {"summary": q[:80]},
                "rows": top_rows,
                "sql": generated_sql,
                "descriptive": descriptive,
                "prescriptive": prescriptive,
                "answer": text,
                "source": "azure_openai",
                "response_mode": "analytics",
                **({"text_to_sql_error": t2sql_error} if t2sql_error else {}),
            }, q)
    except Exception as e:
        return _finalize_copilot({
            "key": "custom",
            "metrics": {"summary": "fallback"},
            "rows": top_rows,
            "sql": generated_sql,
            "descriptive": _build_descriptive_from_rows(top_rows, q) or digest or f"AI unavailable ({e}).",
            "prescriptive": _default_prescriptive(top_rows, q),
            "answer": digest,
            "source": "fallback",
            "response_mode": "analytics",
            "error": str(e),
        }, q)

    return _finalize_copilot({
        "key": "custom",
        "metrics": {"summary": "digest"},
        "rows": top_rows,
        "sql": generated_sql,
        "descriptive": _build_descriptive_from_rows(top_rows, q) or digest,
        "prescriptive": _default_prescriptive(top_rows, q),
        "answer": digest,
        "source": "digest",
        "response_mode": "analytics",
    }, q)


def _split_desc_pres(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""
    for marker in (
        "**Prescriptive:**",
        "**Prescriptive**:",
        "## Prescriptive",
        "Prescriptive —",
        "Prescriptive -",
        "Prescriptive:",
    ):
        if marker in raw:
            left, right = raw.split(marker, 1)
            return _strip_descriptive_header(left), right.strip()
    # case-insensitive fallback
    lower = raw.lower()
    for marker in ("**prescriptive**", "\nprescriptive"):
        idx = lower.find(marker)
        if idx >= 0:
            return _strip_descriptive_header(raw[:idx]), raw[idx:].split(":", 1)[-1].strip()
    return _strip_descriptive_header(raw), ""


def _strip_descriptive_header(text: str) -> str:
    desc = (text or "").strip()
    desc = re.sub(r"^\*\*Descriptive:\*\*\s*", "", desc, flags=re.I)
    desc = re.sub(r"^\*\*Descriptive\*\*:?\s*", "", desc, flags=re.I)
    desc = re.sub(r"^#+\s*Descriptive\s*[—:-]*\s*", "", desc, flags=re.I)
    desc = re.sub(r"^Descriptive\s*[—:-]+\s*", "", desc, flags=re.I)
    return desc.strip()


def _has_action_why(text: str) -> bool:
    t = text or ""
    return "**Action:**" in t and "**Why it matters:**" in t


def _numbered_to_action_bullets(text: str) -> str:
    bullets: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*:?\s*(.*)$", stripped)
        if m:
            title, body = m.group(1).strip(), m.group(2).strip()
            if "**Action:**" not in body:
                body = (
                    f"{body} **Action:** Review and act on this finding. "
                    f"**Why it matters:** Improves sales ops and planning outcomes."
                )
            bullets.append(f"- **{title}:** {body}")
            continue
        m2 = re.match(r"^[-•]\s+\*\*(.+?)\*\*:?\s*(.*)$", stripped)
        if m2:
            title, body = m2.group(1).strip(), m2.group(2).strip()
            if "**Action:**" not in body:
                body = (
                    f"{body} **Action:** Follow up with the owner. "
                    f"**Why it matters:** Keeps exceptions from aging further."
                )
            bullets.append(f"- **{title}:** {body}")
            continue
        if stripped.startswith("-"):
            if "**Action:**" not in stripped:
                stripped = (
                    f"{stripped} **Action:** Assign an owner today. "
                    f"**Why it matters:** Clear ownership drives closure."
                )
            bullets.append(stripped)
        elif stripped.startswith("**") and ":" in stripped:
            bullets.append(f"- {stripped}")
    return "\n\n".join(bullets) if bullets else (text or "").strip()


def _build_descriptive_from_rows(rows: list[dict], message: str) -> str:
    if not rows:
        return ""
    msg = (message or "").lower()
    if any(w in msg for w in ("reorder", "inventory", "shortage")):
        shortages = [_num(r, "shortage_qty") for r in rows]
        total = sum(shortages)
        top = rows[0]
        return (
            f"There are **{len(rows)}** materials below reorder point in the sample, "
            f"with combined shortage quantity **{total:,.0f}**. "
            f"Largest shortage: **{_str(top, 'material_description') or _str(top, 'material_key')}** "
            f"at **{_str(top, 'plant_name')}** (shortage **{_num(top, 'shortage_qty'):,.0f}**)."
        )
    if any(w in msg for w in ("forecast", "outlier", "mape", "ape")):
        apes = [_num(r, "ape") for r in rows if _num(r, "ape") > 0]
        avg = sum(apes) / len(apes) if apes else 0
        top = rows[0]
        return (
            f"Forecast outliers show average APE **{avg:.1f}%** across **{len(rows)}** rows. "
            f"Top outlier: **{_str(top, 'material_key')}** at **{_str(top, 'plant_name')}** "
            f"({_str(top, 'variance_category')}, APE **{_num(top, 'ape'):.1f}%**)."
        )
    if "dealer" in msg:
        top = rows[0]
        return (
            f"Top dealers by order value in the sample: **{_str(top, 'dealer_name')}** "
            f"with order value **{_num(top, 'total_order_value'):,.0f}** "
            f"({_num(top, 'fulfilled_orders'):,.0f}/{_num(top, 'total_orders'):,.0f} fulfilled)."
        )
    top = rows[0]
    total = sum(_num(r, "open_value") for r in rows)
    return (
        f"Open backlog sample covers **{len(rows)}** order lines with combined open value "
        f"**{total:,.0f}**. Highest exposure: order **{_str(top, 'sales_order_id')}** "
        f"for **{_str(top, 'customer_name')}** (open **{_num(top, 'open_value'):,.0f}**, "
        f"age **{_num(top, 'order_age_days'):.0f}** days)."
    )


def _default_prescriptive(rows: list[dict], message: str) -> str:
    msg = (message or "").lower()
    if any(w in msg for w in ("reorder", "inventory", "shortage")) and rows:
        top = rows[0]
        return "\n\n".join([
            f"- **Replenish top shortage:** {_str(top, 'material_key')} at {_str(top, 'plant_name')}. "
            f"**Action:** Raise purchase/STO for shortage {_num(top, 'shortage_qty'):,.0f}. "
            f"**Why it matters:** Prevents stock-outs and protects fulfillment OTIF.",
            "- **Prioritize ACT_NOW plants:** Focus on zero/negative stock first. "
            "**Action:** Sort Agent Workbench Inventory by Act now. "
            "**Why it matters:** Concentrates scarce supply attention where risk is highest.",
            "- **Confirm lead times:** Validate planned delivery days with buyers. "
            "**Action:** Update ETA on the top 5 breaches today. "
            "**Why it matters:** Accurate ETAs improve customer commitments.",
            "- **Monitor recovery:** Track shortage qty week over week. "
            "**Action:** Add reorder breach review to the weekly planning cadence. "
            "**Why it matters:** Early drift detection prevents recurring shortages.",
        ])
    if any(w in msg for w in ("forecast", "outlier", "mape")) and rows:
        top = rows[0]
        return "\n\n".join([
            f"- **Investigate top outlier:** {_str(top, 'material_key')} ({_str(top, 'variance_category')}). "
            f"**Action:** Demand planner reviews actual vs forecast today. "
            f"**Why it matters:** High APE drives excess stock or lost sales.",
            "- **Adjust biased methods:** Revisit forecast method where error persists. "
            "**Action:** Flag methods with APE > 50% in the next S&OP. "
            "**Why it matters:** Method bias compounds across periods.",
            "- **Align sales & planning:** Confirm near-term demand signals. "
            "**Action:** Share outlier list with sales ops this week. "
            "**Why it matters:** Consensus demand reduces surprise variance.",
            "- **Track MAPE trend:** Watch accuracy after changes. "
            "**Action:** Review Forecast tab in Agent Workbench weekly. "
            "**Why it matters:** Confirms corrective actions are working.",
        ])
    if rows:
        top = rows[0]
        return "\n\n".join([
            f"- **Clear top backlog:** Order {_str(top, 'sales_order_id')} for {_str(top, 'customer_name')}. "
            f"**Action:** Confirm supply and commit a delivery date within 48h. "
            f"**Why it matters:** Highest open value drives customer service risk.",
            "- **Work Act now queue:** Age and overdue delivery first. "
            "**Action:** Open Agent Workbench → Fulfillment → Act now. "
            "**Why it matters:** Prioritization protects OTIF and backlog dollars.",
            "- **Remove blockers:** Check delivery/billing/credit blocks on open headers. "
            "**Action:** Ops clears blocks on aged open orders this week. "
            "**Why it matters:** Blocks silently stall fulfillment.",
            "- **Monitor backlog value:** Track open value trend. "
            "**Action:** Review Command Center backlog KPI daily. "
            "**Why it matters:** Early warning before aging worsens.",
        ])
    return "\n\n".join([
        "- **Start in Command Center:** Review backlog, reorder, and forecast KPIs. "
        "**Action:** Open Command Center and note red tiles. "
        "**Why it matters:** Gives a shared baseline for sales ops.",
        "- **Use Agent Workbench:** Work prioritized exceptions. "
        "**Action:** Clear Act now items in Fulfillment/Inventory/Forecast. "
        "**Why it matters:** Focuses effort where risk is highest.",
        "- **Ask a specific question:** Narrow to a plant, material, or dealer. "
        "**Action:** Rephrase with an entity name or period. "
        "**Why it matters:** Produces more actionable recommendations.",
    ])


def _finalize_copilot(result: dict, message: str) -> dict:
    """Shape answers like OrderToCash: Descriptive + Prescriptive + chart."""
    out = dict(result)
    rows = out.get("rows") or []
    desc = _strip_descriptive_header(str(out.get("descriptive") or ""))
    presc = str(out.get("prescriptive") or "").strip()

    # If descriptive still contains a Prescriptive section, split it
    if desc and not presc:
        split_desc, split_presc = _split_desc_pres(desc)
        if split_presc:
            desc, presc = split_desc, split_presc
    elif desc and "**Prescriptive" in desc:
        split_desc, split_presc = _split_desc_pres(desc)
        if split_presc:
            desc = split_desc
            presc = "\n\n".join(p for p in (presc, split_presc) if p).strip()

    presc = re.sub(r"^Here are the bullet points[^\n]*\n+", "", presc, flags=re.I).strip()
    presc = re.sub(r"^\*\*Prescriptive:\*\*\s*", "", presc, flags=re.I).strip()

    if (not desc or len(desc) < 40) and rows:
        built = _build_descriptive_from_rows(rows, message)
        if built:
            desc = built

    if not _has_action_why(presc):
        presc = _numbered_to_action_bullets(presc)
    if not _has_action_why(presc):
        presc = _default_prescriptive(rows, message)

    rows = rows[:50]
    chart_rows = _aggregate_rows_for_chart(rows, message)
    out["descriptive"] = desc
    out["prescriptive"] = presc
    out["rows"] = rows
    out["chart_rows"] = chart_rows
    out["chart"] = out.get("chart") or _build_chart_from_rows(chart_rows or rows, message)
    out["answer"] = (
        f"**Descriptive:**\n{desc}\n\n**Prescriptive:**\n{presc}" if desc or presc else out.get("answer") or ""
    )
    out["response_mode"] = out.get("response_mode") or "analytics"
    return out


def _aggregate_rows_for_chart(rows: list[dict], message: str = "") -> list[dict]:
    """Roll detail rows into a chart-friendly aggregation (top N categories)."""
    if not rows:
        return []
    msg = (message or "").lower()
    keys = set(rows[0].keys())

    def _agg(label_key: str, value_key: str, limit: int = 12) -> list[dict]:
        buckets: dict[str, float] = {}
        for r in rows:
            label = str(r.get(label_key) or "").strip() or "Unknown"
            buckets[label] = buckets.get(label, 0.0) + _num(r, value_key)
        ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [{label_key: k, value_key: v} for k, v in ranked if v]

    if any(w in msg for w in ("reorder", "inventory", "shortage")) and "shortage_qty" in keys:
        label = "material_description" if "material_description" in keys else "material_key"
        if label in keys:
            return _agg(label, "shortage_qty")
    if any(w in msg for w in ("forecast", "outlier", "mape", "ape")) and "ape" in keys:
        label = "material_description" if any(str(r.get("material_description") or "").strip() for r in rows[:5]) else "material_key"
        if label in keys:
            # Keep top outliers by APE (already row-level); de-dupe by material
            return _agg(label, "ape")
    if "dealer" in msg and "dealer_name" in keys and "total_order_value" in keys:
        return _agg("dealer_name", "total_order_value")
    if "customer_name" in keys and "open_value" in keys:
        return _agg("customer_name", "open_value")
    if "plant_name" in keys and "shortage_qty" in keys:
        return _agg("plant_name", "shortage_qty")
    return rows[:15]


def _build_chart_from_rows(rows: list[dict], message: str = "") -> dict | None:
    """Infer a bar chart config from tabular rows (Order Lens columns)."""
    if not rows:
        return None
    keys = list(rows[0].keys())
    msg = (message or "").lower()

    def _is_number(v: object) -> bool:
        try:
            if v is None or v == "":
                return False
            float(v)
            return True
        except Exception:
            return False

    # Prefer domain-specific axes so charts match the question
    preferred: list[tuple[tuple[str, ...], tuple[str, ...], str, str]] = []
    if any(w in msg for w in ("reorder", "inventory", "shortage")):
        preferred.append((
            ("material_description", "material_key", "plant_name"),
            ("shortage_qty", "reorder_shortage_quantity", "stock_qty"),
            "#be123c",
            "Top reorder shortages",
        ))
    if any(w in msg for w in ("forecast", "outlier", "mape", "ape")):
        preferred.append((
            ("material_key", "material_description", "plant_name"),
            ("ape", "absolute_percentage_error", "mape"),
            "#7c3aed",
            "Forecast outliers (APE %)",
        ))
    if "dealer" in msg:
        preferred.append((
            ("dealer_name", "dealer_key"),
            ("total_order_value", "fulfilled_orders", "total_orders"),
            "#0369a1",
            "Dealer order value",
        ))
    preferred.append((
        ("customer_name", "sales_order_id", "plant_name", "material_description", "material_key", "dealer_name"),
        ("open_value", "total_order_value", "shortage_qty", "ape", "order_age_days", "fulfilled_orders"),
        "#0d7d72",
        "Visualization",
    ))

    x_key = None
    y_key = None
    color = "#0d7d72"
    title = "Visualization"

    for name_prefs, value_prefs, pref_color, pref_title in preferred:
        x_cand = next((k for k in name_prefs if k in keys), None)
        y_cand = next((k for k in value_prefs if k in keys), None)
        if x_cand and y_cand:
            x_key, y_key, color, title = x_cand, y_cand, pref_color, pref_title
            break

    if not x_key:
        for k in keys:
            vals = [r.get(k) for r in rows[:20]]
            if any(v is not None and str(v).strip() for v in vals) and not all(
                _is_number(v) for v in vals if v is not None and str(v).strip()
            ):
                x_key = k
                break
    if not y_key:
        for k in keys:
            vals = [r.get(k) for r in rows[:20]]
            if any(_is_number(v) for v in vals):
                y_key = k
                break

    if not x_key or not y_key or x_key == y_key:
        return None

    if x_key == "material_key" and "material_description" in keys:
        if any(str(r.get("material_description") or "").strip() for r in rows[:10]):
            x_key = "material_description"

    if title == "Visualization":
        title_by_y = {
            "open_value": "Open backlog by customer",
            "total_order_value": "Order value by dealer",
            "shortage_qty": "Reorder shortage quantity",
            "ape": "Forecast APE %",
            "order_age_days": "Order age (days)",
        }
        title = title_by_y.get(y_key, f"{y_key.replace('_', ' ').title()} by {x_key.replace('_', ' ')}")

    return {
        "type": "bar_horizontal" if len(rows) > 8 else "bar",
        "xKey": x_key,
        "yKey": y_key,
        "color": color,
        "title": title,
    }


def _sample_rows_for_question(q: str) -> list[dict]:
    ql = q.lower()
    try:
        if "reorder" in ql or "inventory" in ql or "shortage" in ql:
            return run_query(f'''
                select {qcol("material_key")} as material_key,
                       {qcol("material_description")} as material_description,
                       {qcol("plant_name")} as plant_name,
                       {qcol("reorder_shortage_quantity")} as shortage_qty
                from {qview("inventory_reorder_point_breaches")}
                order by {qcol("reorder_shortage_quantity")} desc
                OFFSET 0 ROWS FETCH NEXT 15 ROWS ONLY
            ''')
        if "forecast" in ql or "outlier" in ql or "mape" in ql:
            return run_query(f'''
                select {qcol("material_key")} as material_key,
                       {qcol("material_description")} as material_description,
                       {qcol("plant_name")} as plant_name,
                       {qcol("variance_category")} as variance_category,
                       {qcol("absolute_percentage_error")} as ape
                from {qview("forecasting_outlier_analysis")}
                where {qcol("variance_category")} <> 'Normal'
                order by {qcol("absolute_percentage_error")} desc
                OFFSET 0 ROWS FETCH NEXT 15 ROWS ONLY
            ''')
        if "dealer" in ql:
            return run_query(f'''
                select {qcol("dealer_name", "dealer_order_performance")} as dealer_name,
                       {qcol("total_order_value", "dealer_order_performance")} as total_order_value,
                       {qcol("fulfilled_orders", "dealer_order_performance")} as fulfilled_orders,
                       {qcol("total_orders", "dealer_order_performance")} as total_orders
                from {qview("dealer_order_performance")}
                order by {qcol("total_order_value", "dealer_order_performance")} desc
                OFFSET 0 ROWS FETCH NEXT 15 ROWS ONLY
            ''')
        return run_query(f'''
            select {qcol("sales_order_id")} as sales_order_id,
                   {qcol("customer_name")} as customer_name,
                   {qcol("open_value")} as open_value,
                   {qcol("order_age_days")} as order_age_days
            from {qview("orders_backlog_analysis")}
            order by {qcol("open_value")} desc
            OFFSET 0 ROWS FETCH NEXT 15 ROWS ONLY
        ''')
    except Exception:
        return []

def saved_insights(persona: str | None = None) -> list[dict]:
    wh = f"where PERSONA = '{persona.replace(chr(39), chr(39)+chr(39))}'" if persona else ""
    rows = run_query(f"""
        select TOP 50 INSIGHT_ID, CREATED_AT, CREATED_BY, PERSONA, PAGE, TITLE, QUESTION, SQL_TEXT
        from {qview("saved_insights")}
        {wh}
        order by CREATED_AT desc
    """)
    # Preserve O2C-style uppercase keys expected by Copilot UI helpers
    out = []
    for r in rows:
        out.append({
            "INSIGHT_ID": r.get("insight_id"),
            "TITLE": r.get("title"),
            "QUESTION": r.get("question"),
            "SQL_TEXT": r.get("sql_text"),
            "insight_id": r.get("insight_id"),
            "title": r.get("title"),
            "question": r.get("question"),
            "sql_text": r.get("sql_text"),
        })
    return out


def save_insight(payload: dict) -> dict:
    insight_id = _next_int_id(qview("saved_insights"), "INSIGHT_ID")
    return _onelake_write_or_readonly(
        "append",
        "ONELAKE_SAVED_INSIGHTS_PATH",
        [{
            "INSIGHT_ID": insight_id,
            "CREATED_AT": onelake_writer.utcnow(),
            "CREATED_BY": payload.get("created_by") or "app",
            "PERSONA": payload.get("persona") or "sales_ops",
            "PAGE": payload.get("page") or "copilot",
            "TITLE": payload.get("title") or "Saved insight",
            "QUESTION": payload.get("question") or "",
            "VERIFIED_QUERY_NAME": None,
            "SQL_TEXT": payload.get("sql_text") or "",
            "TAGS": None,
        }],
    )


def delete_insight(insight_id: int) -> dict:
    return _onelake_write_or_readonly(
        "delete", "ONELAKE_SAVED_INSIGHTS_PATH", f"INSIGHT_ID = {int(insight_id)}"
    )


def frequent_questions(persona: str | None = None) -> list[dict]:
    wh = f"where PERSONA = '{persona.replace(chr(39), chr(39)+chr(39))}'" if persona else ""
    # append-only write model (see copilot_chat) means multiple rows can
    # exist per (query, persona) - aggregate them here rather than assuming
    # one counter row per question.
    rows = run_query(f"""
        select TOP 20 NORMALIZED_QUERY, TYPE, sum(FREQUENCY) as frequency, max(LAST_ASKED_AT) as last_asked_at
        from {qview("genie_question_history")}
        {wh}
        group by NORMALIZED_QUERY, TYPE
        order by frequency desc, last_asked_at desc
    """)
    return [
        {
            "NORMALIZED_QUERY": r.get("normalized_query"),
            "TYPE": r.get("type") or "ANALYTICS",
            "FREQUENCY": r.get("frequency") or 0,
            "normalized_query": r.get("normalized_query"),
            "question": r.get("normalized_query"),
        }
        for r in rows
    ]


def most_frequent() -> list[dict]:
    rows = run_query(f"""
        select TOP 20 NORMALIZED_QUERY, TYPE, sum(FREQUENCY) as total_freq
        from {qview("genie_question_history")}
        group by NORMALIZED_QUERY, TYPE
        order by total_freq desc
    """)
    return [
        {
            "NORMALIZED_QUERY": r.get("normalized_query"),
            "TYPE": r.get("type") or "ANALYTICS",
            "TOTAL_FREQ": r.get("total_freq") or 0,
            "normalized_query": r.get("normalized_query"),
            "total_freq": r.get("total_freq") or 0,
        }
        for r in rows
    ]