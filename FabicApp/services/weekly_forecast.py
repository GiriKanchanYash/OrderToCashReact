"""Weekly O2C calendar facts (Fabric): actuals for past/today, due-date + customer-behavior projections for future.

Hand-written T-SQL port of app/services/weekly_forecast.py. This query uses
Snowflake features with no direct T-SQL syntax (GENERATOR/SEQ4 row spine,
aggregate MEDIAN, COUNT_IF, DAYOFWEEKISO, BOOLEAN columns), so it is ported by
hand and executed with run_tsql() (no automatic dialect translation):

  Snowflake                         Fabric T-SQL
  --------------------------------  ---------------------------------------------
  table(generator(rowcount => n))   digits x digits x digits tally (0..999)
                                    (Fabric does not support recursive CTEs)
  median(x)                         ROW_NUMBER/COUNT window median (same result)
  count_if(cond)                    SUM(CASE WHEN cond THEN 1 ELSE 0 END)
  dayofweekiso(d)                   DATEDIFF(day, '19000101', d) % 7 + 1
                                    (1900-01-01 is a Monday; independent of DATEFIRST)
  x::int  (float, rounds)           CAST(ROUND(x, 0) AS INT)
  dayname(d)  ('Mon')               LEFT(DATENAME(weekday, d), 3)
  boolean col / = true              CAST(col AS VARCHAR(5)) IN ('1', 'true', ...)
  AVG(int)  (decimal result)        AVG(CAST(x AS FLOAT))
  ORDER BY inside CTE               ORDER BY on the outer SELECT

The blocked/risk CTE fragments are supplied in T-SQL by o2c_service.get_weekly_metrics.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from FabicApp.db import BM, run_tsql


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


def _iso_dow(expr: str) -> str:
    """ISO day of week (1=Mon..7=Sun), independent of the session DATEFIRST."""
    return f"(DATEDIFF(day, CAST('19000101' AS DATE), {expr}) % 7 + 1)"


def _truthy(expr: str) -> str:
    return f"CAST({expr} AS VARCHAR(5)) IN ('1', 'true', 'TRUE', 'True')"


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
    rs = f"CAST('{range_start[:10]}' AS DATE)"
    re_ = f"CAST('{range_end[:10]}' AS DATE)"
    raw_pay = "DATEADD(day, CAST(ROUND(pay_offset_days, 0) AS INT), DUE_DATE)"
    return f"""
        with digits as (
            select n from (values (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)) v(n)
        ),
        tally as (
            select a.n + 10 * b.n + 100 * c.n as n
            from digits a cross join digits b cross join digits c
        ),
        spine as (
            select CAST(DATEADD(day, n, {rs}) AS DATE) as metric_date
            from tally
            where n < {rowcount}
        ),
        cleared as (
            select
                CUSTOMER_ID,
                COMPANY_CODE,
                DUE_DATE,
                CLEARING_DATE,
                PAID_LATE,
                DATEDIFF(day, DUE_DATE, CLEARING_DATE) as days_from_due,
                ROW_NUMBER() OVER (PARTITION BY CUSTOMER_ID, COMPANY_CODE
                                   ORDER BY DATEDIFF(day, DUE_DATE, CLEARING_DATE)) as rn,
                COUNT(*) OVER (PARTITION BY CUSTOMER_ID, COMPANY_CODE) as cnt
            from {BM}.ar_cleared_item_vw
            where DUE_DATE is not null
              and CLEARING_DATE is not null
        ),
        customer_pay_behavior as (
            select
                CUSTOMER_ID,
                COMPANY_CODE,
                AVG(CAST(days_from_due AS FLOAT)) as avg_days_from_due,
                AVG(CASE WHEN rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
                         THEN CAST(days_from_due AS FLOAT) END) as median_days_from_due,
                COUNT(*) as hist_cnt,
                SUM(CASE WHEN CLEARING_DATE < DUE_DATE THEN 1 ELSE 0 END) as early_payment_count,
                SUM(CASE WHEN {_truthy('PAID_LATE')} THEN 1 ELSE 0 END) as late_payment_count
            from cleared
            group by CUSTOMER_ID, COMPANY_CODE
        ),
        cpb_ranked as (
            select
                median_days_from_due,
                ROW_NUMBER() OVER (ORDER BY median_days_from_due) as rn,
                COUNT(*) OVER () as cnt
            from customer_pay_behavior
            where median_days_from_due is not null
        ),
        global_offset as (
            select COALESCE(
                (select AVG(median_days_from_due) from cpb_ranked
                  where rn IN ((cnt + 1) / 2, (cnt + 2) / 2)),
                (select AVG(avg_days_from_due) from customer_pay_behavior),
                0
            ) as default_offset
        ),
        open_ar as (
            select
                ar.AR_ITEM_ID,
                ar.CUSTOMER_ID,
                ar.COMPANY_CODE,
                ar.DUE_DATE,
                ar.OPEN_AMOUNT_USD,
                COALESCE(
                    IIF(cpb.hist_cnt >= 3, cpb.median_days_from_due, cpb.avg_days_from_due),
                    gofs.default_offset,
                    0
                ) as pay_offset_days,
                COALESCE(cpb.hist_cnt, 0) as hist_cnt
            from {BM}.ar_open_item_vw ar
            left join customer_pay_behavior cpb
              on cpb.CUSTOMER_ID = ar.CUSTOMER_ID
             and cpb.COMPANY_CODE = ar.COMPANY_CODE
            cross join global_offset gofs
            where 1=1 {cf_ar}{df_ar}
        ),
        open_ar_expected as (
            select
                *,
                DATEADD(
                    day,
                    IIF({_iso_dow(raw_pay)} IN (6, 7), 8 - {_iso_dow(raw_pay)}, 0),
                    {raw_pay}
                ) as expected_pay_date,
                IIF(pay_offset_days < 0, OPEN_AMOUNT_USD, 0) as early_profile_amt,
                IIF(pay_offset_days > 0, OPEN_AMOUNT_USD, 0) as late_profile_amt
            from open_ar
        ),
        due_on_date as (
            select
                CAST(DUE_DATE AS DATE) as d,
                SUM(OPEN_AMOUNT_USD) as due_amt,
                COUNT(*) as due_cnt,
                90.0 as due_confidence
            from open_ar_expected
            where DUE_DATE between {rs} and {re_}
            group by CAST(DUE_DATE AS DATE)
        ),
        expected_coll as (
            select
                CAST(expected_pay_date AS DATE) as d,
                SUM(OPEN_AMOUNT_USD) as v,
                SUM(early_profile_amt) as early_amt,
                SUM(late_profile_amt) as late_amt,
                ROUND(
                    CAST(SUM(OPEN_AMOUNT_USD * (
                        CASE
                            WHEN hist_cnt >= 10 THEN 92
                            WHEN hist_cnt >= 5 THEN 85
                            WHEN hist_cnt >= 3 THEN 78
                            WHEN hist_cnt >= 1 THEN 65
                            ELSE 52
                        END
                    )) AS FLOAT) / NULLIF(SUM(OPEN_AMOUNT_USD), 0),
                    1
                ) as expected_confidence
            from open_ar_expected
            where expected_pay_date between {rs} and {re_}
            group by CAST(expected_pay_date AS DATE)
        ),
        overdue_by_day as (
            select
                s.metric_date as d,
                SUM(ar.OPEN_AMOUNT_USD) as amt,
                COUNT(*) as cnt
            from spine s
            join open_ar_expected ar
              on ar.DUE_DATE < s.metric_date
            group by s.metric_date
        ),
        revenue as (
            select CAST(inv.INVOICE_DATE AS DATE) as d, SUM(inv.GROSS_AMOUNT) as v
            from {BM}.ar_invoice_vw inv
            where inv.INVOICE_DATE between {rs} and {re_} {cf}{df}
            group by CAST(inv.INVOICE_DATE AS DATE)
        ),
        collections_actual as (
            select CAST(p.PAYMENT_DATE AS DATE) as d, SUM(p.PAYMENT_AMOUNT_USD) as v
            from {BM}.payment_vw p
            where p.PAYMENT_DATE between {rs} and {re_} {cf_pay}{df_pay}
            group by CAST(p.PAYMENT_DATE AS DATE)
        ),
        invoices_day as (
            select CAST(inv.INVOICE_DATE AS DATE) as d, COUNT(*) as v
            from {BM}.ar_invoice_vw inv
            where inv.INVOICE_DATE between {rs} and {re_} {cf}{df}
            group by CAST(inv.INVOICE_DATE AS DATE)
        ),
        disputes_opened as (
            select CAST(d.OPENED_DATE AS DATE) as d, COUNT(*) as v
            from {BM}.dispute_vw d
            where d.OPENED_DATE between {rs} and {re_}
            group by CAST(d.OPENED_DATE AS DATE)
        ),
        disputes_open as (
            select s.metric_date as d, COUNT(*) as v
            from spine s
            join {BM}.dispute_vw d
              on d.OPENED_DATE <= s.metric_date
             and COALESCE(d.RESOLVED_DATE, CAST('9999-12-31' AS DATE)) > s.metric_date
            group by s.metric_date
        ),
        unapplied as (
            select CAST(p.PAYMENT_DATE AS DATE) as d, SUM(p.PAYMENT_AMOUNT_USD) as v
            from {BM}.payment_vw p
            where p.CLEARING_DATE is null
              and p.PAYMENT_DATE between {rs} and {re_} {cf_pay}{df_pay}
            group by CAST(p.PAYMENT_DATE AS DATE)
        ),
        short_pay as (
            select CAST(p.PAYMENT_DATE AS DATE) as d, COUNT(*) as v
            from {BM}.payment_vw p
            where {_truthy('p.IS_PARTIAL')}
              and p.PAYMENT_DATE between {rs} and {re_} {cf_pay}{df_pay}
            group by CAST(p.PAYMENT_DATE AS DATE)
        ),
        payments_cleared as (
            select CAST(p.CLEARING_DATE AS DATE) as d, SUM(p.PAYMENT_AMOUNT_USD) as v, COUNT(*) as cnt
            from {BM}.payment_vw p
            where p.CLEARING_DATE between {rs} and {re_} {cf_pay}{df_pay}
            group by CAST(p.CLEARING_DATE AS DATE)
        ),
        disputes_resolved as (
            select CAST(d.RESOLVED_DATE AS DATE) as d,
                   AVG(CAST(DATEDIFF(day, d.OPENED_DATE, d.RESOLVED_DATE) AS FLOAT)) as avg_days,
                   COUNT(*) as cnt
            from {BM}.dispute_vw d
            where d.RESOLVED_DATE between {rs} and {re_}
            group by CAST(d.RESOLVED_DATE AS DATE)
        ),
        ptp as (
            select CAST(ptp.PROMISE_DATE AS DATE) as d, COUNT(*) as v
            from {BM}.ptp_vw ptp
            where ptp.PROMISE_DATE between {rs} and {re_}
            group by CAST(ptp.PROMISE_DATE AS DATE)
        ),
        {blocked_join}
        {risk_join}
        daily as (
            select
                CONVERT(VARCHAR(10), s.metric_date, 23) as metric_date,
                LEFT(DATENAME(weekday, s.metric_date), 3) as day_name,
                COALESCE(rev.v, 0) as revenue,
                CASE
                    WHEN {_iso_dow('s.metric_date')} IN (6, 7) THEN 0
                    WHEN s.metric_date <= CAST(GETDATE() AS DATE) THEN COALESCE(coll.v, 0)
                    ELSE COALESCE(ec.v, 0)
                END as collections,
                COALESCE(dd.due_amt, 0) as amount_due,
                IIF({_iso_dow('s.metric_date')} IN (6, 7), 0, COALESCE(ec.v, 0)) as expected_collections,
                IIF({_iso_dow('s.metric_date')} IN (6, 7), 0, COALESCE(ec.early_amt, 0)) as early_payer_expected,
                IIF({_iso_dow('s.metric_date')} IN (6, 7), 0, COALESCE(ec.late_amt, 0)) as late_payer_expected,
                COALESCE(ec.expected_confidence, 52) as expected_confidence,
                COALESCE(dd.due_confidence, 90) as amount_due_confidence,
                COALESCE(inv.v, 0) as invoices_created,
                COALESCE(dop.v, 0) as disputes_opened,
                COALESCE(dopen.v, 0) as disputes_open,
                COALESCE(od.amt, 0) as overdue_amount,
                COALESCE(od.cnt, 0) as overdue_invoices,
                COALESCE(ua.v, 0) as unapplied_cash,
                COALESCE(sp.v, 0) as short_payments,
                COALESCE(ptp.v, 0) as promise_to_pay,
                COALESCE(bl.v, 0) as invoice_blocked,
                COALESCE(rk.v, 0) as revenue_at_risk,
                COALESCE(pc.v, 0) as payments_cleared,
                COALESCE(pc.cnt, 0) as payments_cleared_cnt,
                COALESCE(dr.avg_days, 0) as dispute_resolution_days,
                COALESCE(dr.cnt, 0) as disputes_resolved
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
        )
        select * from daily
        order by metric_date
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
    rows = run_tsql(sql)
    out: list[dict[str, Any]] = []
    for row in rows:
        md = str(row.get("metric_date") or row.get("METRIC_DATE"))[:10]
        out.append({"date": md, "facts": parse_facts_row(row)})
    return out
