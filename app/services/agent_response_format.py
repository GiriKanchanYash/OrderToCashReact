"""Normalize Cortex agent text into OrderLens workbench sections."""
from __future__ import annotations

import re

from app.services.billing_block_catalog import billing_order_context_lines, billing_problem_summary, block_scenario_lines
from app.services.collection_policy import (
    LATE_FEE_GRACE_DAYS,
    LATE_FEE_MONTHLY_RATE,
    POLICY_LATE_FEE,
    POLICY_PARTIAL,
    expected_dunning_history,
    invoice_collection_action,
    invoice_policy_refs,
    late_fee_usd,
    partial_payment_note,
    total_due_with_late_fee,
)

_NOISE_LINE = re.compile(
    r"(?im)^(?:"
    r"let me .+|"
    r"i(?:'ll| will) .+|"
    r"the tools?(?: are| returned| failed).+|"
    r"please note that .+|"
    r"i need to .+|"
    r"my next step .+"
    r").*$"
)
_EMOJI = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"
)
_MD_TABLE_ROW = re.compile(r"^\s*\|?.+\|.+\|?\s*$")
_MD_SEP_ROW = re.compile(r"^\s*\|?[-:\s|]+\|?\s*$")
_MD_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_SECTION_BOLD = re.compile(r"^\*\*([A-Z][A-Z0-9\s/&()—-]{3,}(?:\s*\([^)]+\))?)\*\*\s*$", re.MULTILINE)
_PIPE_ONLY = re.compile(r"^\s*\|([^|]+)\|\s*$")
_HRULE = re.compile(r"^\s*---+\s*$")


def _val(row: dict, *keys, default=None):
    for k in keys:
        for variant in (k, str(k).lower(), str(k).upper()):
            if variant in row and row[variant] is not None:
                return row[variant]
    return default


def invoice_lines(ar_rows: list, limit: int = 8) -> str:
    """Legacy single-line format; prefer invoice_detail_lines with enriched rows."""
    if not ar_rows:
        return "No open invoice lines found for this customer in the mart."
    enriched = [enrich_invoice_row(r) for r in ar_rows[:limit]]
    return invoice_detail_lines(enriched, limit=limit)


def enrich_invoice_row(row: dict, **_ignored) -> dict:
    """Enrich an AR row using COLL-LPF-01 / COLL-DUN-01 policy rules."""
    days = int(_val(row, "DAYS_PAST_DUE", "days_past_due") or 0)
    amt = float(_val(row, "OPEN_AMOUNT_USD", "open_amount_usd") or 0)
    due = str(_val(row, "DUE_DATE", "due_date") or "")[:10]
    disputed = bool(
        _val(row, "DISPUTE_ID", "dispute_id")
        or _val(row, "IS_DISPUTED", "is_disputed") in (True, "TRUE", "true", 1, "1")
    )
    fee = 0.0 if disputed else late_fee_usd(amt, days)
    new_balance = round(amt + fee, 2)
    return {
        "invoice_id": str(_val(row, "INVOICE_ID", "invoice_id") or ""),
        "due_date": due,
        "open_amount_usd": amt,
        "days_past_due": days,
        "aging_bucket": str(_val(row, "AGING_BUCKET", "aging_bucket") or ""),
        "dunning_level": int(_val(row, "DUNNING_LEVEL", "dunning_level") or 0),
        "is_disputed": disputed,
        "dispute_id": str(_val(row, "DISPUTE_ID", "dispute_id") or ""),
        "recommended_action": invoice_collection_action(days, row),
        "late_fee_usd": fee,
        "total_due_with_late_fee": new_balance,
        "policy_refs": invoice_policy_refs(days, row),
        "within_grace": days > 0 and days <= LATE_FEE_GRACE_DAYS,
    }


def invoice_detail_lines(enriched_rows: list[dict], limit: int = 10) -> str:
    if not enriched_rows:
        return "No open invoice lines found for this customer in the mart."
    lines: list[str] = []
    for r in enriched_rows[:limit]:
        inv = r.get("invoice_id") or "n/a"
        amt = float(r.get("open_amount_usd") or 0)
        fee = float(r.get("late_fee_usd") or 0)
        new_balance = float(r.get("total_due_with_late_fee") or (amt + fee))
        if fee > 0:
            fee_bit = (
                f"Late fee ${fee:,.0f} ({LATE_FEE_MONTHLY_RATE * 100:.1f}%/mo after "
                f"{LATE_FEE_GRACE_DAYS}d grace, {POLICY_LATE_FEE}) | "
                f"Re-bill total: ${new_balance:,.0f}"
            )
        else:
            fee_bit = f"No late fee yet (within {LATE_FEE_GRACE_DAYS}-day grace, {POLICY_LATE_FEE})"
        lines.append(
            f"- {inv}: ${amt:,.0f} due {r.get('due_date') or 'n/a'} "
            f"({r.get('days_past_due', 0)} days past due, {r.get('aging_bucket', 'n/a')}) | "
            f"Action: {r.get('recommended_action', 'Review')} | "
            f"{fee_bit} | Policy: {r.get('policy_refs', POLICY_LATE_FEE)}"
        )
    lines.append(f"- {partial_payment_note()}")
    return "\n".join(lines)


def dunning_history_lines(
    days_past_due: int,
    due_date: str | None = None,
    *,
    has_ptp: bool = False,
    is_disputed: bool = False,
) -> str:
    """Reconstructed dunning ledger (what was due + inferred outcome + next step)."""
    history = expected_dunning_history(
        days_past_due, due_date, has_ptp=has_ptp, is_disputed=is_disputed
    )
    if not history:
        return ""
    lines: list[str] = []
    for h in history:
        marker = "[sent]" if h.get("done") else "[next]"
        lines.append(
            f"- {marker} {h['date']}: {h['label']} ({h['policy']}) — {h['status']}"
        )
    return "\n".join(lines)


def policy_ref_lines(policies: list[dict], limit: int = 4) -> str:
    if not policies:
        return "- No policy rows matched; apply standard O2C governance for this domain."
    lines: list[str] = []
    for p in policies[:limit]:
        pid = p.get("policy_id") or ""
        title = p.get("policy_title") or ""
        text = (p.get("policy_text") or "")[:160].strip()
        if pid:
            detail = f"{title}: {text}" if text and text != title else title or text
            lines.append(f"- {pid}: {detail}")
    return "\n".join(lines) if lines else "- No policy rows matched; apply standard O2C governance for this domain."


def _has_section(text: str, name: str) -> bool:
    return name.upper() in (text or "").upper()


def _section_body(text: str, name: str) -> str:
    if not text:
        return ""
    pattern = rf"(?is)<strong>\s*{re.escape(name)}\s*</strong>\s*(.*?){_NEXT_SECTION}"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


_NEXT_SECTION = r"(?=\r?\n<strong>|\Z)"


def _remove_section(text: str, name: str) -> str:
    if not _has_section(text, name):
        return text
    pattern = rf"(?is)(?:\r?\n)*<strong>\s*{re.escape(name)}\s*</strong>\s*.*?{_NEXT_SECTION}"
    return re.sub(pattern, "", text, count=1).strip()


def _section_has_content(text: str, name: str) -> bool:
    if not _has_section(text, name):
        return False
    body = _section_body(text, name)
    if not body:
        return False
    stripped = re.sub(r"[\s\-•|]", "", body)
    return len(stripped) > 0


def _force_section(text: str, name: str, content: str) -> str:
    content = (content or "").strip()
    if not content:
        return text
    header = f"<strong>{name}</strong>"
    if not _has_section(text, name):
        return f"{text.rstrip()}\n\n{header}\n{content}".strip()
    pattern = rf"(?is)(<strong>\s*{re.escape(name)}\s*</strong>\s*)(.*?){_NEXT_SECTION}"
    return re.sub(pattern, rf"\1\n{content}\n", text, count=1)

def _inject_section(text: str, name: str, content: str) -> str:
    content = (content or "").strip()
    if not content:
        return text
    header = f"<strong>{name}</strong>"
    if not _has_section(text, name):
        return f"{text.rstrip()}\n\n{header}\n{content}".strip()
    if _section_has_content(text, name):
        return text
    return _force_section(text, name, content)


def customer_history_lines(row: dict | None) -> str:
    if not row:
        return "- No customer O2C history found in mart."
    return (
        f"- Risk band: {_val(row, 'CUSTOMER_RISK_BAND', 'customer_risk_band')}\n"
        f"- Disputes: {_val(row, 'OPEN_DISPUTES', 'open_disputes')} open / "
        f"{_val(row, 'TOTAL_DISPUTES', 'total_disputes')} total\n"
        f"- Late payments: {_val(row, 'LATE_PAYMENT_COUNT', 'late_payment_count')} | "
        f"Broken PTPs: {_val(row, 'BROKEN_PTP_COUNT', 'broken_ptp_count')}\n"
        f"- Past due: ${float(_val(row, 'PAST_DUE_USD', 'past_due_usd') or 0):,.0f} "
        f"(max {_val(row, 'MAX_DAYS_PAST_DUE', 'max_days_past_due')} days)\n"
        f"- Billing blocks: credit {_val(row, 'CREDIT_BLOCK_ORDERS', 'credit_block_orders')} | "
        f"fulfillment {_val(row, 'FULFILLMENT_BLOCK_ORDERS', 'fulfillment_block_orders')}"
    )


def contact_log_lines(rows: list, limit: int = 6) -> str:
    if not rows:
        return "- No prior collection contacts logged for this customer."
    lines: list[str] = []
    for r in rows[:limit]:
        date = str(_val(r, "ACTIVITY_DATE", "activity_date") or "")[:10]
        atype = _val(r, "ACTIVITY_TYPE", "activity_type") or "CONTACT"
        outcome = _val(r, "CONTACT_OUTCOME", "contact_outcome") or "n/a"
        inv = _val(r, "INVOICE_ID", "invoice_id") or "account-level"
        notes = str(_val(r, "NOTES", "notes") or "").strip()
        note_bit = f" — {notes[:90]}" if notes else ""
        lines.append(f"- {date} {atype} ({outcome}) on {inv}{note_bit}")
    return "\n".join(lines)


def open_dispute_lines(rows: list, limit: int = 5) -> str:
    if not rows:
        return "- No open dispute cases for this customer."
    lines: list[str] = []
    for r in rows[:limit]:
        lines.append(
            f"- {_val(r, 'DISPUTE_ID', 'dispute_id')}: "
            f"{_val(r, 'DISPUTE_REASON', 'dispute_reason')} on "
            f"{_val(r, 'INVOICE_ID', 'invoice_id')} "
            f"(${float(_val(r, 'DISPUTED_AMOUNT', 'disputed_amount') or 0):,.0f}, "
            f"{_val(r, 'DISPUTE_STATUS', 'dispute_status')})"
        )
    return "\n".join(lines)


def ptp_lines(rows: list, limit: int = 4) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows[:limit]:
        pid = _val(r, "PTP_ID", "ptp_id") or "PTP"
        inv = _val(r, "INVOICE_ID", "invoice_id") or "n/a"
        amt = float(_val(r, "PROMISED_AMOUNT", "promised_amount") or 0)
        pdate = str(_val(r, "PROMISED_PAY_DATE", "promised_pay_date") or "")[:10]
        status = _val(r, "PTP_STATUS", "ptp_status") or "OPEN"
        lines.append(f"- {pid} on {inv}: ${amt:,.0f} promised by {pdate} ({status})")
    return "\n".join(lines)


def format_invoice_context_block(
    *,
    invoice_id: str,
    days_past_due: int = 0,
    open_amount: float = 0.0,
    due_date: str | None = None,
    contacts: list | None = None,
    disputes: list | None = None,
    ptp_rows: list | None = None,
) -> str:
    """Customer context scoped to one invoice — logged contacts, the derived dunning
    ledger (what was due + inferred outcome + next step), PTP, disputes, and late-fee re-bill."""
    inv = invoice_id or "this invoice"
    contact_rows = contacts or []
    dispute_rows = disputes or []
    ptp = ptp_rows or []
    disputed = bool(dispute_rows)
    has_ptp = bool(ptp)
    lines: list[str] = []

    if contact_rows:
        lines.append("Logged contacts on this invoice:")
        lines.append(contact_log_lines(contact_rows))

    ledger = dunning_history_lines(
        days_past_due, due_date, has_ptp=has_ptp, is_disputed=disputed
    )
    if ledger:
        lines.append(f"Dunning trail per COLL-DUN-01 (reconstructed from {days_past_due}-day aging):")
        lines.append(ledger)
    elif not contact_rows:
        lines.append(f"- No outreach due yet for invoice {inv} (within grace) — monitor only.")

    if ptp:
        lines.append("Promise-to-pay on this invoice:")
        lines.append(ptp_lines(ptp))
    elif days_past_due > 30 and not disputed:
        lines.append(f"- No promise-to-pay on file for invoice {inv} — secure a dated commitment on next contact.")

    if dispute_rows:
        lines.append("Open dispute on this invoice (dunning paused):")
        lines.append(open_dispute_lines(dispute_rows))

    if open_amount and days_past_due > LATE_FEE_GRACE_DAYS and not disputed:
        fee = late_fee_usd(open_amount, days_past_due)
        new_balance = total_due_with_late_fee(open_amount, days_past_due)
        lines.append(
            f"- Late fee accrued: ${fee:,.0f} on ${open_amount:,.0f} "
            f"({LATE_FEE_MONTHLY_RATE * 100:.1f}%/month after {LATE_FEE_GRACE_DAYS}-day grace, {POLICY_LATE_FEE}). "
            f"Re-bill amount to send: ${new_balance:,.0f}."
        )

    return "\n".join(lines)


def format_order_context_block(
    *,
    sales_order_id: str,
    contacts: list | None = None,
    disputes: list | None = None,
) -> str:
    """Customer context scoped to a billing sales order."""
    lines: list[str] = []
    so = sales_order_id or "this order"
    contact_rows = contacts or []
    dispute_rows = disputes or []

    if contact_rows:
        lines.append(contact_log_lines(contact_rows))
    else:
        lines.append(f"- No collection or billing contacts logged for sales order {so}.")

    if dispute_rows:
        lines.append("Open disputes on invoices for this order:")
        lines.append(open_dispute_lines(dispute_rows))

    return "\n".join(lines) if lines else f"- No invoice-specific customer context for sales order {so}."


COLLECTIONS_OUTPUT_FORMAT = f"""
Return ONLY these sections in this exact order. Use HTML headers: <strong>SECTION NAME</strong>
Rules: no markdown tables, no emojis, no preamble, no tool commentary, no "let me look up" text.
Do not use inline bold in sentences — only use <strong> for section headers on their own line.

<strong>SITUATION ASSESSMENT</strong>
2-3 sentences stating the definitive collections problem from the data (customer name, invoice id, amount, days past due, dunning level). Name the root cause and cash risk — do not speculate, ask what could be wrong, or list possibilities.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
Exactly 4 numbered lines in this pattern:
1. [ACTION NAME]: [Fact from data]. Next step: [owner + timeframe + concrete task].
2. ...
3. ...
4. ...

<strong>CUSTOMER CONTEXT</strong>
For the focus invoice ONLY, copy the provided dunning trail verbatim: each reminder/escalation that was due for the current aging, its inferred outcome (no payment / PTP / dispute pause), and the next step to send. Include any logged contacts, promise-to-pay, open dispute, and the accrued late fee with the re-bill amount. Do not summarize general customer history.

<strong>OPEN INVOICES</strong>
Per-invoice bullets with due date, days past due, recommended action, estimated late fee per COLL-LPF-01 (1.5% per month pro-rated after {LATE_FEE_GRACE_DAYS}-day grace; no fee during grace or open dispute), the re-bill total (open balance + late fee), and policy refs. Include partial-payment note per {POLICY_PARTIAL}. Copy the authoritative invoice lines provided — do not leave this section empty.

<strong>TIMELINE</strong>
3-4 bullets with dates or day offsets (Today, +48h, +7 days) tied to the actions above.

<strong>POLICY REFERENCES</strong>
Bullet list of POLICY_ID values cited and one-line why each applies.

<strong>CONFIRM WITH USER</strong>
2-3 questions asking the user to confirm before you execute any action (e.g. "Would you like me to send a payment reminder email to [Customer] for invoice [INV]?"). Do not claim actions were already taken.
""".strip()

BILLING_OUTPUT_FORMAT = """
Return ONLY these sections in this exact order. Use HTML headers: <strong>SECTION NAME</strong>
Rules: no markdown tables, no emojis, no preamble, no tool commentary, no pipe characters.
Do not use inline bold in sentences — only use <strong> for section headers on their own line.

<strong>SITUATION ASSESSMENT</strong>
2-3 sentences stating the definitive billing problem from the Reason and status fields (block type, root cause, revenue at risk). State facts — do not speculate, ask what could be wrong, or list possibilities. Omit delivery or credit fields when they are N/A or not the primary blocker.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
Exactly 4 numbered lines:
1. [ACTION NAME]: [Fact from data]. Next step: [owner + timeframe + concrete task].
2. ...
3. ...
4. ...

<strong>CUSTOMER CONTEXT</strong>
Bullet list of contacts, promise-to-pay, and disputes for THIS invoice/order only — do not summarize unrelated customer history.

<strong>ORDER CONTEXT</strong>
Bullet list with sales order id, block scenario type, customer, order value, delivery status, credit check status, and billing eligibility reason.

<strong>TIMELINE</strong>
3 bullets (Today, +48h, before period close) tied to clearing the billing blocker.

<strong>POLICY REFERENCES</strong>
Bullet list of POLICY_ID values cited and one-line why each applies.

<strong>CONFIRM WITH USER</strong>
2-3 questions asking the user to confirm before you execute any action (e.g. "Would you like me to escalate order [SO] to Credit for release review?"). Do not claim actions were already taken.
""".strip()

DISPUTE_OUTPUT_FORMAT = """
Return ONLY these sections in this exact order. Use HTML headers: <strong>SECTION NAME</strong>
Rules: no markdown tables, no emojis, no preamble, no tool commentary, no pipe characters.
Do not use inline bold in sentences — only use <strong> for section headers on their own line.

<strong>SITUATION ASSESSMENT</strong>
2-3 sentences stating the definitive dispute problem (dispute id, customer name, invoice, reason, amount, status). State the root cause and cash-at-risk — do not speculate or ask what could be wrong.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
Exactly 4 numbered lines:
1. [ACTION NAME]: [Fact from data]. Next step: [owner + timeframe + concrete task].
2. ...
3. ...
4. ...

<strong>INVOICE CONTEXT</strong>
Bullet list with invoice id, amount, status, and any supporting evidence needed.

<strong>TIMELINE</strong>
3 bullets (Today, +48h, +7 days) for resolution milestones.

<strong>POLICY REFERENCES</strong>
Bullet list of POLICY_ID values cited and one-line why each applies.

<strong>CONFIRM WITH USER</strong>
2-3 questions asking the user to confirm before you execute any action (e.g. "Would you like me to email the customer requesting dispute documentation for [DISPUTE_ID]?"). Do not claim actions were already taken.
""".strip()

CASH_APPLICATION_OUTPUT_FORMAT = """
Return ONLY these sections in this exact order. Use HTML headers: <strong>SECTION NAME</strong>
Rules: no markdown tables, no emojis, no preamble, no tool commentary, no pipe characters.
Do not use inline bold in sentences — only use <strong> for section headers on their own line.

<strong>SITUATION ASSESSMENT</strong>
2-3 sentences stating the definitive cash-application problem (payment id, customer name, amount, exception type, days unapplied). State the root cause and DSO risk — do not speculate or ask what could be wrong.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
Exactly 4 numbered lines:
1. [ACTION NAME]: [Fact from data]. Next step: [owner + timeframe + concrete task].
2. ...
3. ...
4. ...

<strong>SUGGESTED MATCHES</strong>
Bullet list of invoice_id, open amount, match type, and amount delta (use candidate data provided).

<strong>TIMELINE</strong>
3 bullets (Today, +24h, +3 days) for matching and posting.

<strong>POLICY REFERENCES</strong>
Bullet list of POLICY_ID values cited and one-line why each applies.

<strong>CONFIRM WITH USER</strong>
2-3 questions asking the user to confirm before you execute any action (e.g. "Would you like me to request remittance advice from [Customer] for payment [PAYMENT_ID]?"). Do not claim actions were already taken.
""".strip()


def _normalize_headers(text: str) -> str:
    out = _MD_HEADING.sub(r"<strong>\1</strong>", text)
    out = _SECTION_BOLD.sub(r"<strong>\1</strong>", out)
    # Inline bold breaks the workbench section parser — flatten to plain text.
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"<strong>([^<]+)</strong>", lambda m: m.group(0) if _is_section_title(m.group(1)) else m.group(1), out)
    return out


def _is_section_title(title: str) -> bool:
    core = re.sub(r"\s*\([^)]*\)\s*$", "", (title or "").strip())
    letters = [c for c in core if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.75


def _strip_noise_and_tables(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _NOISE_LINE.match(stripped):
            continue
        if "policy document search returned no matching records" in stripped.lower():
            continue
        if stripped.lower().startswith("note: policy document search"):
            continue
        if _MD_TABLE_ROW.match(line) or _MD_SEP_ROW.match(line):
            continue
        if _HRULE.match(stripped):
            continue
        pipe_only = _PIPE_ONLY.match(stripped)
        if pipe_only:
            cell = pipe_only.group(1).strip()
            if cell and not cell.replace("-", "").replace(":", "").strip():
                continue
            if cell:
                lines.append(cell)
            continue
        lines.append(line)
    out = "\n".join(lines)
    out = _EMOJI.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def polish_workbench_response(
    text: str,
    *,
    policies: list[dict],
    section_fillers: dict[str, str] | None = None,
) -> str:
    body = _strip_noise_and_tables(_normalize_headers(text or ""))
    fillers = section_fillers or {}
    for section, content in fillers.items():
        body = _inject_section(body, section, content)
    if not _section_has_content(body, "POLICY REFERENCES"):
        body = _inject_section(body, "POLICY REFERENCES", policy_ref_lines(policies))
    return body.strip()


def polish_collections_response(
    text: str,
    *,
    ar_rows: list,
    policies: list[dict],
    action_group: str = "ACT_NOW",
    customer_context: str = "",
    invoice_rows: list[dict] | None = None,
) -> str:
    inv_content = invoice_detail_lines(
        invoice_rows if invoice_rows is not None else [enrich_invoice_row(r) for r in ar_rows]
    )
    fillers: dict[str, str] = {}
    if customer_context:
        fillers["CUSTOMER CONTEXT"] = customer_context
    if action_group.upper() == "ACT_NOW" and not _section_has_content(text, "TIMELINE"):
        fillers["TIMELINE"] = (
            "- Today: Right-party contact and dunning escalation.\n"
            "- Within 48 hours: Credit hold if no PTP is logged.\n"
            "- Day 7: Escalate to external collections or legal if unresolved."
        )
    body = polish_workbench_response(text, policies=policies, section_fillers=fillers)
    return _force_section(body, "OPEN INVOICES", inv_content)


def polish_billing_response(
    text: str,
    *,
    policies: list[dict],
    sales_order_id: str,
    customer: str,
    customer_id: str,
    order_value: float,
    status: str,
    reason: str,
    delivery_status: str,
    credit_status: str,
    action_group: str = "ACT_NOW",
    customer_context: str = "",
    order_status: str = "",
) -> str:
    fillers: dict[str, str] = {}
    if customer_context:
        fillers["CUSTOMER CONTEXT"] = customer_context
    fillers["ORDER CONTEXT"] = billing_order_context_lines(
        sales_order_id=sales_order_id,
        customer=customer,
        order_value=order_value,
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=order_status,
    )
    if action_group.upper() == "ACT_NOW" and not _section_has_content(text, "TIMELINE"):
        fillers["TIMELINE"] = (
            "- Today: Validate blocker with credit and fulfillment owners.\n"
            "- Within 48 hours: Post goods issue or release credit hold.\n"
            "- Before period close: Trigger invoice creation once eligible."
        )
    return polish_workbench_response(text, policies=policies, section_fillers=fillers)


def polish_dispute_response(
    text: str,
    *,
    policies: list[dict],
    dispute_id: str,
    customer_id: str,
    invoice_id: str,
    reason: str,
    amount: float,
    status: str,
    inv: dict | None = None,
    action_group: str = "ACT_NOW",
) -> str:
    fillers: dict[str, str] = {}
    if not _section_has_content(text, "INVOICE CONTEXT"):
        if inv:
            fillers["INVOICE CONTEXT"] = (
                f"- Invoice: {inv.get('invoice_id') or invoice_id}\n"
                f"- Amount: ${float(inv.get('gross_amount') or amount):,.0f}\n"
                f"- Status: {inv.get('invoice_status') or 'n/a'}\n"
                f"- Dispute: {dispute_id} · {reason} · ${amount:,.0f} · {status}"
            )
        else:
            fillers["INVOICE CONTEXT"] = (
                f"- Dispute: {dispute_id} · Customer {customer_id}\n"
                f"- Invoice: {invoice_id or 'n/a'} · ${amount:,.0f} · {reason}"
            )
    if action_group.upper() == "ACT_NOW" and not _section_has_content(text, "TIMELINE"):
        fillers["TIMELINE"] = (
            "- Today: Pull invoice evidence and assign dispute owner.\n"
            "- Within 48 hours: Request customer documentation.\n"
            "- Day 7: Resolve, credit, or escalate per dispute policy."
        )
    return polish_workbench_response(text, policies=policies, section_fillers=fillers)


def polish_cash_response(
    text: str,
    *,
    policies: list[dict],
    payment_id: str,
    customer_id: str,
    amount: float,
    exception: str,
    days: int,
    candidates: list,
) -> str:
    fillers: dict[str, str] = {}
    if not _section_has_content(text, "SUGGESTED MATCHES"):
        if candidates:
            lines = [
                f"- {c.get('invoice_id')}: ${float(c.get('open_amount_usd') or 0):,.0f} open "
                f"({c.get('match_type')}, delta ${float(c.get('amount_delta') or 0):,.0f})"
                for c in candidates[:5]
            ]
            fillers["SUGGESTED MATCHES"] = "\n".join(lines)
        else:
            fillers["SUGGESTED MATCHES"] = "No open invoices found for this customer."
    if not _section_has_content(text, "TIMELINE"):
        fillers["TIMELINE"] = (
            "- Today: Match remittance to closest open invoices.\n"
            "- Within 24 hours: Request remittance advice if unmatched.\n"
            "- Day 3: Post clearing and notify collections."
        )
    return polish_workbench_response(text, policies=policies, section_fillers=fillers)


def build_billing_fallback(
    *,
    sales_order_id: str,
    customer: str,
    customer_id: str,
    order_value: float,
    status: str,
    reason: str,
    delivery_status: str,
    credit_status: str,
    action_group: str,
    policies: list[dict],
    customer_context: str = "",
    order_status: str = "",
) -> str:
    problem = billing_problem_summary(
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=order_status,
    )
    return f"""<strong>SITUATION ASSESSMENT</strong>
Sales order {sales_order_id} for {customer} (${order_value:,.0f}) cannot be invoiced. Problem: {problem} Priority: {action_group.replace('_', ' ')} — revenue recognition is delayed until this blocker clears.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. VALIDATE BLOCKERS: {problem} Next step: Billing clerk confirms hold reason in ERP today.
2. GOODS ISSUE: {'Post goods issue if shipment is complete' if status in ('PENDING_GOODS_ISSUE', 'NO_DELIVERY', 'PENDING_DELIVERY') else 'Confirm fulfillment status with logistics'}. Next step: Fulfillment owner validates delivery within 24 hours.
3. CREDIT RELEASE: {'Credit check is blocked — escalate for release' if (credit_status or '').upper() == 'BLOCKED' else reason} Next step: Credit analyst releases hold or adjusts order within 48 hours.
4. INVOICE TRIGGER: Once eligible, create invoice and verify AR posting to collections queue before period close.

<strong>CUSTOMER CONTEXT</strong>
{customer_context or "- No customer history available."}

<strong>ORDER CONTEXT</strong>
{billing_order_context_lines(
    sales_order_id=sales_order_id,
    customer=customer,
    order_value=order_value,
    status=status,
    reason=reason,
    credit_status=credit_status,
    delivery_status=delivery_status,
    order_status=order_status,
)}

<strong>TIMELINE</strong>
- Today: Validate blocker with credit and fulfillment owners.
- Within 48 hours: Post goods issue or release credit hold.
- Before period close: Trigger invoice creation once eligible.

<strong>POLICY REFERENCES</strong>
{policy_ref_lines(policies)}"""


def build_dispute_fallback(
    *,
    dispute_id: str,
    customer_id: str,
    invoice_id: str,
    reason: str,
    amount: float,
    status: str,
    action_group: str,
    inv: dict | None,
    policies: list[dict],
) -> str:
    inv_line = (
        f"- Invoice: {inv.get('invoice_id') or invoice_id}: "
        f"${float(inv.get('gross_amount') or amount):,.0f} ({inv.get('invoice_status') or 'n/a'})"
        if inv else f"- Invoice: {invoice_id or 'n/a'} · ${amount:,.0f}"
    )
    return f"""<strong>SITUATION ASSESSMENT</strong>
Dispute {dispute_id} for customer {customer_id} on invoice {invoice_id or 'n/a'} is {status.lower()} for ${amount:,.0f} ({reason}). Priority: {action_group.replace('_', ' ')} — unresolved disputes block cash application and inflate DSO.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. VALIDATE ROOT CAUSE: Confirm the {reason.lower()} claim against invoice line items and POD. Next step: Billing analyst pulls evidence today.
2. CUSTOMER CONTACT: Request supporting documentation within 2 business days. Next step: AR specialist emails AP with invoice package.
3. RESOLUTION PATH: {'Approve partial credit if evidence supports claim' if amount > 25000 else 'Resolve with credit memo or rebill once validated'}. Next step: Dispute owner documents decision in ERP.
4. CLOSE LOOP: Update dispute status and notify collections. Next step: Mark resolved when credit memo posts.

<strong>INVOICE CONTEXT</strong>
{inv_line}
- Dispute: {dispute_id} · {reason} · ${amount:,.0f} · {status}

<strong>TIMELINE</strong>
- Today: Pull invoice evidence and assign dispute owner.
- Within 48 hours: Request customer documentation.
- Day 7: Resolve, credit, or escalate per dispute policy.

<strong>POLICY REFERENCES</strong>
{policy_ref_lines(policies)}"""


def build_cash_fallback(
    *,
    payment_id: str,
    customer_id: str,
    amount: float,
    exception: str,
    days: int,
    candidates: list,
    policies: list[dict],
) -> str:
    match_lines = "\n".join(
        f"- {c.get('invoice_id')}: ${float(c.get('open_amount_usd') or 0):,.0f} open "
        f"({c.get('match_type')}, delta ${float(c.get('amount_delta') or 0):,.0f})"
        for c in candidates[:5]
    ) or "No open invoices found for this customer."
    return f"""<strong>SITUATION ASSESSMENT</strong>
Payment {payment_id} from {customer_id} for ${amount:,.0f} is {exception.replace('_', ' ').lower()} for {days} days. Unapplied cash delays AR clearing and distorts DSO until matched and posted.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. MATCH INVOICE: Compare remittance advice to open AR for {customer_id}. Next step: Cash app clerk starts with closest amount matches below today.
2. VALIDATE CUSTOMER: Confirm payer bank account matches {customer_id}. Next step: Review lockbox remitter name for mis-posts within 24 hours.
3. RESOLVE VARIANCE: {'Apply as partial payment and leave residual on account' if exception == 'PARTIAL' else 'Request remittance detail before posting if no match'}. Next step: Document variance reason in ERP.
4. POST CLEARING: Apply payment and verify open AR balance clears. Next step: Notify collections when posting completes.

<strong>SUGGESTED MATCHES</strong>
{match_lines}

<strong>TIMELINE</strong>
- Today: Match remittance to closest open invoices.
- Within 24 hours: Request remittance advice if unmatched.
- Day 3: Post clearing and notify collections.

<strong>POLICY REFERENCES</strong>
{policy_ref_lines(policies)}"""


def build_collections_fallback(
    *,
    name: str,
    customer_id: str,
    total_past: float,
    ar_rows: list,
    max_days: int,
    dunning: int,
    action_group: str,
    policies: list[dict],
    customer_context: str = "",
) -> str:
    escalate = "Escalate to final notice and credit-hold review" if max_days >= 60 else (
        "Send structured payment reminder with itemized open invoices"
    )
    timeline = ""
    if action_group.upper() == "ACT_NOW":
        timeline = """
<strong>TIMELINE</strong>
- Today: Senior collector RPC with AP lead; log outcome in ERP.
- Within 48 hours: Credit hold if no PTP received.
- Day 7: External collections referral for unresolved 90+ day balance."""

    return f"""<strong>SITUATION ASSESSMENT</strong>
{name} ({customer_id}) has ${total_past:,.0f} past due on the selected invoice line. Aging is {max_days} days at dunning level {dunning}. Priority: {action_group.replace('_', ' ')}. Invoice-level action required today.

<strong>IMMEDIATE ACTIONS (ranked)</strong>
1. RIGHT-PARTY CONTACT: ${total_past:,.0f} on this invoice, {max_days} days past due. Next step: Call AP decision-maker today and confirm receipt/dispute status on this invoice.
2. DUNNING ESCALATION: Dunning level {dunning} for {max_days}-day paper. Next step: {escalate} within 24 hours and CC account manager.
3. PROMISE TO PAY: No PTP on file for this invoice. Next step: Capture firm pay date and log PTP in OrderLens before end of day.
4. DISPUTE TRIAGE: Verify dispute status on this line and clear billing blockers before next contact.

<strong>CUSTOMER CONTEXT</strong>
{customer_context or "- No customer history available."}

<strong>OPEN INVOICES</strong>
{invoice_lines(ar_rows)}

{timeline}

<strong>POLICY REFERENCES</strong>
{policy_ref_lines(policies)}"""
