"""Weekly O2C calendar facts: actuals for past/today, due-date + customer-behavior projections for future."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.db import BM, run_query


def _row_val(row: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return default


def parse_facts_row(row: dict) -> dict[str, float]:
    return {
        "revenue": _row_val(row, "revenue", "REVENUE"),
        "collections": _row_val(row, "collections", "COLLECTIONS"),
        "amount_due": _row_val(row, "amount_due", "AMOUNT_DUE"),
        "expected_collections": _row_val(row, "expected_collections", "EXPECTED_COLLECTIONS"),
        "early_payer_expected": _row_val(row, "early_payer_expected", "EARLY_PAYER_EXPECTED"),
        "late_payer_expected": _row_val(row, "late_payer_expected", "LATE_PAYER_EXPECTED"),
        "invoices_created": _row_val(row, "invoices_created", "INVOICES_CREATED"),
        "disputes_opened": _row_val(row, "disputes_opened", "DISPUTES_OPENED"),
        "disputes_open": _row_val(row, "disputes_open", "DISPUTES_OPEN"),
        "overdue_amount": _row_val(row, "overdue_amount", "OVERDUE_AMOUNT"),
        "overdue_invoices": _row_val(row, "overdue_invoices", "OVERDUE_INVOICES"),
        "unapplied_cash": _row_val(row, "unapplied_cash", "UNAPPLIED_CASH"),
        "short_payments": _row_val(row, "short_payments", "SHORT_PAYMENTS"),
        "promise_to_pay": _row_val(row, "promise_to_pay", "PROMISE_TO_PAY"),
        "invoice_blocked": _row_val(row, "invoice_blocked", "INVOICE_BLOCKED"),
        "revenue_at_risk": _row_val(row, "revenue_at_risk", "REVENUE_AT_RISK"),
        "payments_cleared": _row_val(row, "payments_cleared", "PAYMENTS_CLEARED"),
        "payments_cleared_cnt": _row_val(row, "payments_cleared_cnt", "PAYMENTS_CLEARED_CNT"),
        "dispute_resolution_days": _row_val(row, "dispute_resolution_days", "DISPUTE_RESOLUTION_DAYS"),
        "disputes_resolved": _row_val(row, "disputes_resolved", "DISPUTES_RESOLVED"),
        "expected_confidence": _row_val(row, "expected_confidence", "EXPECTED_CONFIDENCE", default=52.0),
        "amount_due_confidence": _row_val(row, "amount_due_confidence", "AMOUNT_DUE_CONFIDENCE", default=90.0),
    }


def build_daily_facts_sql(
    range_start: str,
    range_end: str,
    *,
    cf: str,
    df: str,
    cf_ar: str,
    df_ar: str,
    cf_pay: str,
    df_pay: str,
    blocked_join: str,
    risk_join: str,
) -> str:
    start_dt = datetime.strptime(range_start[:10], "%Y-%m-%d").date()
    end_dt = datetime.strptime(range_end[:10], "%Y-%m-%d").date()
    rowcount = max(1, (end_dt - start_dt).days + 1)
    return f"""
        with spine as (
            select dateadd(day, seq4(), '{range_start}'::date) as metric_date
            from table(generator(rowcount => {rowcount}))
        ),
        customer_pay_behavior as (
            select
                CUSTOMER_ID,
                COMPANY_CODE,
                avg(datediff(day, DUE_DATE, CLEARING_DATE)) as avg_days_from_due,
                median(datediff(day, DUE_DATE, CLEARING_DATE)) as median_days_from_due,
                count(*) as hist_cnt,
                count_if(CLEARING_DATE < DUE_DATE) as early_payment_count,
                count_if(PAID_LATE) as late_payment_count
            from {BM}.AR_CLEARED_ITEM_VW
            where DUE_DATE is not null
              and CLEARING_DATE is not null
            group by 1, 2
        ),
        global_offset as (
            select coalesce(
                median(median_days_from_due),
                avg(avg_days_from_due),
                0
            ) as default_offset
            from customer_pay_behavior
        ),
        open_ar as (
            select
                ar.AR_ITEM_ID,
                ar.CUSTOMER_ID,
                ar.COMPANY_CODE,
                ar.DUE_DATE,
                ar.OPEN_AMOUNT_USD,
                coalesce(
                    iff(cpb.hist_cnt >= 3, cpb.median_days_from_due, cpb.avg_days_from_due),
                    go.default_offset,
                    0
                ) as pay_offset_days,
                coalesce(cpb.hist_cnt, 0) as hist_cnt
            from {BM}.AR_OPEN_ITEM_VW ar
            left join customer_pay_behavior cpb
              on cpb.CUSTOMER_ID = ar.CUSTOMER_ID
             and cpb.COMPANY_CODE = ar.COMPANY_CODE
            cross join global_offset go
            where 1=1 {cf_ar}{df_ar}
        ),
        open_ar_expected as (
            select
                *,
                dateadd(
                    day,
                    iff(
                        dayofweekiso(dateadd(day, pay_offset_days::int, DUE_DATE)) in (6, 7),
                        8 - dayofweekiso(dateadd(day, pay_offset_days::int, DUE_DATE)),
                        0
                    ),
                    dateadd(day, pay_offset_days::int, DUE_DATE)
                ) as expected_pay_date,
                iff(pay_offset_days < 0, OPEN_AMOUNT_USD, 0) as early_profile_amt,
                iff(pay_offset_days > 0, OPEN_AMOUNT_USD, 0) as late_profile_amt
            from open_ar
        ),
        due_on_date as (
            select
                DUE_DATE::date as d,
                sum(OPEN_AMOUNT_USD) as due_amt,
                count(*) as due_cnt,
                90.0 as due_confidence
            from open_ar_expected
            where DUE_DATE between '{range_start}'::date and '{range_end}'::date
            group by 1
        ),
        expected_coll as (
            select
                expected_pay_date::date as d,
                sum(OPEN_AMOUNT_USD) as v,
                sum(early_profile_amt) as early_amt,
                sum(late_profile_amt) as late_amt,
                round(
                    sum(OPEN_AMOUNT_USD * (
                        case
                            when hist_cnt >= 10 then 92
                            when hist_cnt >= 5 then 85
                            when hist_cnt >= 3 then 78
                            when hist_cnt >= 1 then 65
                            else 52
                        end
                    )) / nullif(sum(OPEN_AMOUNT_USD), 0),
                    1
                ) as expected_confidence
            from open_ar_expected
            where expected_pay_date between '{range_start}'::date and '{range_end}'::date
            group by 1
        ),
        overdue_by_day as (
            select
                s.metric_date as d,
                sum(ar.OPEN_AMOUNT_USD) as amt,
                count(*) as cnt
            from spine s
            join open_ar_expected ar
              on ar.DUE_DATE < s.metric_date
            group by 1
        ),
        revenue as (
            select inv.invoice_date::date as d, sum(inv.gross_amount) as v
            from {BM}.AR_INVOICE_VW inv
            where inv.invoice_date between '{range_start}'::date and '{range_end}'::date {cf}{df}
            group by 1
        ),
        collections_actual as (
            select p.payment_date::date as d, sum(p.payment_amount_usd) as v
            from {BM}.PAYMENT_VW p
            where p.payment_date between '{range_start}'::date and '{range_end}'::date {cf_pay}{df_pay}
            group by 1
        ),
        invoices_day as (
            select inv.invoice_date::date as d, count(*) as v
            from {BM}.AR_INVOICE_VW inv
            where inv.invoice_date between '{range_start}'::date and '{range_end}'::date {cf}{df}
            group by 1
        ),
        disputes_opened as (
            select d.opened_date::date as d, count(*) as v
            from {BM}.DISPUTE_VW d
            where d.opened_date between '{range_start}'::date and '{range_end}'::date
            group by 1
        ),
        disputes_open as (
            select s.metric_date as d, count(*) as v
            from spine s
            join {BM}.DISPUTE_VW d
              on d.opened_date <= s.metric_date
             and coalesce(d.resolved_date, '9999-12-31'::date) > s.metric_date
            group by 1
        ),
        unapplied as (
            select p.payment_date::date as d, sum(p.payment_amount_usd) as v
            from {BM}.PAYMENT_VW p
            where p.clearing_date is null
              and p.payment_date between '{range_start}'::date and '{range_end}'::date {cf_pay}{df_pay}
            group by 1
        ),
        short_pay as (
            select p.payment_date::date as d, count(*) as v
            from {BM}.PAYMENT_VW p
            where p.is_partial = true
              and p.payment_date between '{range_start}'::date and '{range_end}'::date {cf_pay}{df_pay}
            group by 1
        ),
        payments_cleared as (
            select p.clearing_date::date as d, sum(p.payment_amount_usd) as v, count(*) as cnt
            from {BM}.PAYMENT_VW p
            where p.clearing_date between '{range_start}'::date and '{range_end}'::date {cf_pay}{df_pay}
            group by 1
        ),
        disputes_resolved as (
            select d.resolved_date::date as d,
                   avg(datediff('day', d.opened_date, d.resolved_date)) as avg_days,
                   count(*) as cnt
            from {BM}.DISPUTE_VW d
            where d.resolved_date between '{range_start}'::date and '{range_end}'::date
            group by 1
        ),
        ptp as (
            select ptp.promise_date::date as d, count(*) as v
            from {BM}.PTP_VW ptp
            where ptp.promise_date between '{range_start}'::date and '{range_end}'::date
            group by 1
        ),
        {blocked_join}
        {risk_join}
        daily as (
            select
                s.metric_date::varchar as metric_date,
                dayname(s.metric_date) as day_name,
                coalesce(rev.v, 0) as revenue,
                case
                    when dayofweekiso(s.metric_date) in (6, 7) then 0
                    when s.metric_date <= current_date() then coalesce(coll.v, 0)
                    else coalesce(ec.v, 0)
                end as collections,
                coalesce(dd.due_amt, 0) as amount_due,
                iff(dayofweekiso(s.metric_date) in (6, 7), 0, coalesce(ec.v, 0)) as expected_collections,
                iff(dayofweekiso(s.metric_date) in (6, 7), 0, coalesce(ec.early_amt, 0)) as early_payer_expected,
                iff(dayofweekiso(s.metric_date) in (6, 7), 0, coalesce(ec.late_amt, 0)) as late_payer_expected,
                coalesce(ec.expected_confidence, 52) as expected_confidence,
                coalesce(dd.due_confidence, 90) as amount_due_confidence,
                coalesce(inv.v, 0) as invoices_created,
                coalesce(dop.v, 0) as disputes_opened,
                coalesce(dopen.v, 0) as disputes_open,
                coalesce(od.amt, 0) as overdue_amount,
                coalesce(od.cnt, 0) as overdue_invoices,
                coalesce(ua.v, 0) as unapplied_cash,
                coalesce(sp.v, 0) as short_payments,
                coalesce(ptp.v, 0) as promise_to_pay,
                coalesce(bl.v, 0) as invoice_blocked,
                coalesce(rk.v, 0) as revenue_at_risk,
                coalesce(pc.v, 0) as payments_cleared,
                coalesce(pc.cnt, 0) as payments_cleared_cnt,
                coalesce(dr.avg_days, 0) as dispute_resolution_days,
                coalesce(dr.cnt, 0) as disputes_resolved
            from spine s
            left join revenue rev on rev.d = s.metric_date
            left join collections_actual coll on coll.d = s.metric_date
            left join due_on_date dd on dd.d = s.metric_date
            left join expected_coll ec on ec.d = s.metric_date
            left join invoices_day inv on inv.d = s.metric_date
            left join disputes_opened dop on dop.d = s.metric_date
            left join disputes_open dopen on dopen.d = s.metric_date
            left join overdue_by_day od on od.d = s.metric_date
            left join unapplied ua on ua.d = s.metric_date
            left join short_pay sp on sp.d = s.metric_date
            left join ptp on ptp.d = s.metric_date
            left join blocked bl on bl.d = s.metric_date
            left join risk rk on rk.d = s.metric_date
            left join payments_cleared pc on pc.d = s.metric_date
            left join disputes_resolved dr on dr.d = s.metric_date
            order by s.metric_date
        )
        select * from daily
    """


def fetch_daily_facts(
    range_start: str,
    range_end: str,
    *,
    cf: str,
    df: str,
    cf_ar: str,
    df_ar: str,
    cf_pay: str,
    df_pay: str,
    blocked_join: str,
    risk_join: str,
) -> list[dict[str, Any]]:
    sql = build_daily_facts_sql(
        range_start,
        range_end,
        cf=cf,
        df=df,
        cf_ar=cf_ar,
        df_ar=df_ar,
        cf_pay=cf_pay,
        df_pay=df_pay,
        blocked_join=blocked_join,
        risk_join=risk_join,
    )
    rows = run_query(sql)
    out: list[dict[str, Any]] = []
    for row in rows:
        md = str(row.get("metric_date") or row.get("METRIC_DATE"))[:10]
        out.append({"date": md, "facts": parse_facts_row(row)})
    return out
