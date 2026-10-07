"""Collections policy constants — must match POLICY_KB and docs/o2c_order_to_cash_policy.md."""
from __future__ import annotations

from datetime import datetime, timedelta

# COLL-LPF-01
LATE_FEE_GRACE_DAYS = 15
LATE_FEE_MONTHLY_RATE = 0.015  # 1.5% per month
LATE_FEE_APR = 0.18  # 18% APR (equivalent to 1.5% × 12)

# COLL-DUN-01 / COLL-ESC-01 action thresholds (days past due)
STRUCTURED_REMINDER_DAYS = 30
ESCALATION_DAYS = 60  # COLL-ESC-01
FINAL_NOTICE_DAYS = 90

POLICY_LATE_FEE = "COLL-LPF-01"
POLICY_DUNNING = "COLL-DUN-01"
POLICY_ESCALATION = "COLL-ESC-01"
POLICY_PARTIAL = "COLL-PART-01"
POLICY_DISPUTE = "DISP-CRT-01"


def _row_val(row: dict, *keys, default=None):
    for k in keys:
        for variant in (k, str(k).lower(), str(k).upper()):
            if variant in row and row[variant] is not None:
                return row[variant]
    return default


def is_invoice_disputed(row: dict) -> bool:
    return bool(
        _row_val(row, "DISPUTE_ID", "dispute_id")
        or _row_val(row, "IS_DISPUTED", "is_disputed") in (True, "TRUE", "true", 1, "1")
    )


def late_fee_usd(open_amount: float, days_past_due: int) -> float:
    """COLL-LPF-01: 1.5% per month pro-rated on balance after 15-day grace."""
    days = max(int(days_past_due or 0), 0)
    if days <= LATE_FEE_GRACE_DAYS:
        return 0.0
    feeable_days = days - LATE_FEE_GRACE_DAYS
    return round(float(open_amount or 0) * LATE_FEE_MONTHLY_RATE * (feeable_days / 30.0), 2)


def invoice_policy_refs(days: int, row: dict) -> str:
    if is_invoice_disputed(row):
        return POLICY_DISPUTE
    refs: list[str] = []
    if days > LATE_FEE_GRACE_DAYS:
        refs.append(POLICY_LATE_FEE)
    else:
        refs.append(POLICY_DUNNING)
    if days >= ESCALATION_DAYS:
        refs.append(POLICY_ESCALATION)
    return ", ".join(refs)


def invoice_collection_action(days: int, row: dict) -> str:
    if is_invoice_disputed(row):
        return "Hold dunning — open dispute (DISP-CRT-01)"
    if days >= FINAL_NOTICE_DAYS:
        return f"ACT NOW: Final notice + assess late fee ({POLICY_LATE_FEE})"
    if days >= ESCALATION_DAYS:
        return f"Escalate per {POLICY_ESCALATION}; late fee letter ({POLICY_LATE_FEE})"
    if days >= STRUCTURED_REMINDER_DAYS:
        return f"RPC + structured reminder ({POLICY_DUNNING}); late fee applies ({POLICY_LATE_FEE})"
    if days > LATE_FEE_GRACE_DAYS:
        return f"Past grace — add late fee notice ({POLICY_LATE_FEE})"
    if days > 0:
        return f"Friendly reminder within {LATE_FEE_GRACE_DAYS}-day grace ({POLICY_DUNNING})"
    return "Not past due — monitor only"


def partial_payment_note() -> str:
    return (
        f"Partial payments: post per {POLICY_PARTIAL}; residual stays open; "
        f"late fee on unpaid balance per {POLICY_LATE_FEE} ({LATE_FEE_MONTHLY_RATE * 100:.1f}%/month after "
        f"{LATE_FEE_GRACE_DAYS}-day grace)."
    )


def total_due_with_late_fee(open_amount: float, days_past_due: int) -> float:
    """Re-billed amount = open balance + accrued late fee (COLL-LPF-01)."""
    return round(float(open_amount or 0) + late_fee_usd(open_amount, days_past_due), 2)


# Graduated dunning schedule (days past due → action) per COLL-DUN-01 / COLL-ESC-01.
DUNNING_SCHEDULE: list[tuple[int, str, str]] = [
    (1, "Friendly payment reminder (email)", POLICY_DUNNING),
    (LATE_FEE_GRACE_DAYS + 1, "Past-grace reminder + late-fee notice", POLICY_LATE_FEE),
    (STRUCTURED_REMINDER_DAYS, "Structured reminder + phone right-party contact", POLICY_DUNNING),
    (45, "Second written reminder + statement of accrued late fees", POLICY_LATE_FEE),
    (ESCALATION_DAYS, "Escalation notice + credit-hold review", POLICY_ESCALATION),
    (FINAL_NOTICE_DAYS, "Final notice — pre-legal / external collections", POLICY_ESCALATION),
]


def expected_dunning_history(
    days_past_due: int,
    due_date: str | None = None,
    *,
    has_ptp: bool = False,
    is_disputed: bool = False,
) -> list[dict]:
    """Derive the dunning steps that should already have gone out for this invoice's aging.

    We do not fabricate ERP rows — this reconstructs the expected COLL-DUN-01 cadence from
    days-past-due so collectors see what was due, the inferred outcome, and the next step.
    """
    days = max(int(days_past_due or 0), 0)
    if days <= 0:
        return []

    base = None
    if due_date:
        try:
            base = datetime.strptime(str(due_date)[:10], "%Y-%m-%d").date()
        except ValueError:
            base = None

    def _date(offset: int) -> str:
        return (base + timedelta(days=offset)).strftime("%Y-%m-%d") if base else f"due+{offset}d"

    history: list[dict] = []
    next_added = False
    for offset, label, policy in DUNNING_SCHEDULE:
        if days >= offset:
            if is_disputed:
                outcome = "issued, then paused — invoice under dispute (DISP-CRT-01)"
            elif has_ptp:
                outcome = "issued — promise-to-pay captured (monitor for kept/broken)"
            else:
                outcome = "issued — no payment or promise recorded"
            history.append({
                "offset": offset,
                "label": label,
                "policy": policy,
                "date": _date(offset),
                "status": outcome,
                "done": True,
            })
        elif not next_added:
            history.append({
                "offset": offset,
                "label": label,
                "policy": policy,
                "date": _date(offset),
                "status": "next step — schedule/send now",
                "done": False,
            })
            next_added = True
    return history
