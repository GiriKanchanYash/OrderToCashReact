from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from FabicApp.db import run_query, run_execute, run_tsql, BM
from FabicApp.services.weekly_forecast import build_daily_facts_sql

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 900


def _run_queries_parallel(specs: dict[str, str], *, max_workers: int = 8) -> dict[str, list]:
    """Run independent Fabric reads concurrently (each opens its own connection)."""
    if not specs:
        return {}
    if len(specs) == 1:
        name, sql = next(iter(specs.items()))
        return {name: run_query(sql)}

    results: dict[str, list] = {}
    workers = min(len(specs), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_query, sql): name for name, sql in specs.items()}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
    return results


def _cache_get(key: str) -> Any | None:
    now = time.time()
    with _CACHE_LOCK:
        ent = _CACHE.get(key)
        if not ent:
            return None
        ts, val = ent
        if now - ts > _CACHE_TTL:
            _CACHE.pop(key, None)
            return None
        return val


def _cache_set(key: str, val: Any) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), val)


def _comparison_from_series(rows: list[dict], *value_keys: str, n_recent: int = 3, n_prior: int = 3) -> dict:
    """Compare recent vs prior window from a monthly series."""
    keys = value_keys or ("value", "dso", "cei", "past_due_ar", "past_due_pct", "VALUE", "DSO", "CEI")
    vals: list[float] = []
    for r in rows:
        for k in keys:
            if k in r and r[k] is not None:
                try:
                    vals.append(float(r[k]))
                    break
                except (TypeError, ValueError):
                    continue

    def _block(nr: int, np: int) -> dict:
        empty = {"current": None, "prior": None, "delta": None, "delta_pct": None}
        if len(vals) < 2:
            return empty
        nr = min(nr, len(vals))
        np = min(np, max(len(vals) - nr, 1))
        if len(vals) < nr + 1:
            nr = 1
            np = 1
        curr = sum(vals[-nr:]) / nr
        prior_slice = vals[-(nr + np):-nr] if len(vals) >= nr + np else vals[:-nr]
        if not prior_slice:
            return {"current": round(curr, 2), "prior": None, "delta": None, "delta_pct": None}
        prior = sum(prior_slice) / len(prior_slice)
        delta = curr - prior
        delta_pct = (100.0 * delta / prior) if prior else None
        return {
            "current": round(curr, 2),
            "prior": round(prior, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
        }

    period = _block(n_recent, n_prior)
    return {
        "period": period,
        "quarterly": _block(3, 3),
        "yearly": _block(12, 12) if len(vals) >= 13 else _block(min(6, max(len(vals) - 1, 1)), min(6, max(len(vals) - 1, 1))),
    }


def _blocks_for_preset(period_preset: str | None) -> tuple[int, int, str]:
    preset = (period_preset or "this_quarter").lower()
    if "month" in preset:
        return 1, 1, "vs last month"
    if "quarter" in preset:
        return 3, 3, "vs last quarter"
    if "year" in preset:
        return 12, 12, "vs last year"
    return 3, 3, "vs prior period"


def _prior_period_bounds(date_from: str, date_to: str) -> tuple[str, str]:
    from datetime import datetime, timedelta

    d0 = datetime.strptime(date_from[:10], "%Y-%m-%d")
    d1 = datetime.strptime(date_to[:10], "%Y-%m-%d")
    span_days = (d1 - d0).days + 1
    prior_end = d0 - timedelta(days=1)
    prior_start = prior_end - timedelta(days=span_days - 1)
    return prior_start.strftime("%Y-%m-%d"), prior_end.strftime("%Y-%m-%d")


_KPI_LOWER_IS_BETTER = {
    "dso": True,
    "add": True,
    "past_due_pct": True,
    "past_due_ar": True,
    "cei": False,
    "ar_pct_revenue": False,
    "forecast_mape": True,
}


def _trend_from_delta(delta: float | None, delta_pct: float | None, lower_is_better: bool, label: str) -> dict:
    if delta is None:
        return {
            "direction": "flat",
            "sentiment": "neutral",
            "text": "No prior data",
            "delta": None,
            "delta_pct": None,
            "label": label,
        }
    flat_threshold = 0.05
    if abs(delta) < flat_threshold:
        return {
            "direction": "flat",
            "sentiment": "neutral",
            "text": f"No change {label}",
            "delta": 0,
            "delta_pct": 0,
            "label": label,
        }
    direction = "up" if delta > 0 else "down"
    sign = "+" if delta > 0 else ""
    pct_part = f" ({sign}{delta_pct}%)" if delta_pct is not None else ""
    return {
        "direction": direction,
        "sentiment": "good" if direction == "up" else "bad",
        "text": f"{sign}{delta:.1f}{pct_part} {label}".strip(),
        "delta": round(delta, 2),
        "delta_pct": delta_pct,
        "label": label,
    }


def _trend_from_kpis(
    current: float | None,
    prior: float | None,
    lower_is_better: bool,
    label: str,
) -> dict:
    if current is None or prior is None:
        return _trend_from_delta(None, None, lower_is_better, label)
    delta = current - prior
    delta_pct = round(100.0 * delta / prior, 1) if prior else None
    return _trend_from_delta(delta, delta_pct, lower_is_better, label)


def _scoped_prior_kpis(
    company: str | None,
    dealer_id: str | None,
    date_from: str,
    date_to: str,
) -> dict:
    """Lightweight prior-period KPI snapshot for trend badges (no charts/recursion)."""
    cache_key = f"prior_kpis:{company or 'all'}:{dealer_id or 'all'}:{date_from}:{date_to}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ar_from = _scoped_ar_from(dealer_id, date_from, date_to)
    inv_from = _scoped_invoice_from(dealer_id, date_from, date_to)

    past_due = run_query(f"""
        SELECT SUM(ar.OPEN_AMOUNT_USD) AS total_ar,
               SUM(IFF(ar.DAYS_PAST_DUE > 0, ar.OPEN_AMOUNT_USD, 0)) AS past_due_ar
        {ar_from}
    """)
    add_row = run_query(f"""
        SELECT AVG(ar.DAYS_PAST_DUE) AS add_val
        {ar_from} AND ar.DAYS_PAST_DUE > 0
    """)
    inv_90 = run_query(f"""
        SELECT SUM(inv.GROSS_AMOUNT) AS invoiced_90d
        {inv_from} AND inv.INVOICE_DATE >= DATEADD(day, -90, CURRENT_DATE())
    """)
    cei_row = run_query(f"""
        SELECT 100 * SUM(IFF(UPPER(inv.INVOICE_STATUS) = 'PAID', inv.GROSS_AMOUNT, 0))
                   / NULLIF(SUM(inv.GROSS_AMOUNT), 0) AS cei
        {inv_from} AND inv.INVOICE_DATE IS NOT NULL
    """)
    ar_pct_row = run_query(f"""
        SELECT 100 * SUM(ar.OPEN_AMOUNT_USD) / NULLIF(SUM(so.TOTAL_ORDER_VALUE), 0) AS ar_pct
        {ar_from}
    """)
    forecast = run_query(f"""
        SELECT AVG(f.MAPE) AS mape
        FROM {BM}.forecast_accuracy_vw f
        WHERE f.COMPANY_CODE IN (
            SELECT DISTINCT inv.COMPANY_CODE {inv_from} AND inv.COMPANY_CODE IS NOT NULL
        )
    """)

    pd = past_due[0] if past_due else {}
    total_ar = float(pd.get("TOTAL_AR") or pd.get("total_ar") or 0)
    past_due_ar = float(pd.get("PAST_DUE_AR") or pd.get("past_due_ar") or 0)
    invoiced_90d = float((inv_90[0] if inv_90 else {}).get("INVOICED_90D") or (inv_90[0] if inv_90 else {}).get("invoiced_90d") or 0)
    dso_proxy = (total_ar / (invoiced_90d / 90.0)) if invoiced_90d > 0 else None
    avg_add = (add_row[0] if add_row else {}).get("ADD_VAL") or (add_row[0] if add_row else {}).get("add_val")
    latest_cei = (cei_row[0] if cei_row else {}).get("CEI") or (cei_row[0] if cei_row else {}).get("cei")
    ar_pct_val = (ar_pct_row[0] if ar_pct_row else {}).get("AR_PCT") or (ar_pct_row[0] if ar_pct_row else {}).get("ar_pct")
    past_due_pct = (100.0 * past_due_ar / total_ar) if total_ar > 0 else 0.0

    result = {
        "dso": round(float(dso_proxy), 1) if dso_proxy is not None else None,
        "add": round(float(avg_add), 1) if avg_add is not None else None,
        "past_due_pct": round(past_due_pct, 1),
        "past_due_ar": past_due_ar,
        "cei": round(float(latest_cei), 1) if latest_cei is not None else None,
        "ar_pct_revenue": round(float(ar_pct_val), 1) if ar_pct_val is not None else None,
        "forecast_mape": round(float(forecast[0].get("MAPE") or forecast[0].get("mape") or 0), 1) if forecast else None,
    }
    _cache_set(cache_key, result)
    return result


def _build_kpi_trends(
    kpis: dict,
    prior_kpis: dict | None,
    series_blocks: dict[str, dict],
    compare_label: str,
) -> dict:
    trends: dict[str, dict] = {}
    metric_keys = ["dso", "add", "cei", "past_due_pct", "past_due_ar", "ar_pct_revenue", "forecast_mape"]
    for key in metric_keys:
        lower = _KPI_LOWER_IS_BETTER.get(key, True)
        if prior_kpis and prior_kpis.get(key) is not None and kpis.get(key) is not None:
            curr = float(kpis[key])
            prior = float(prior_kpis[key])
            delta = curr - prior
            delta_pct = round(100.0 * delta / prior, 1) if prior else None
            trends[key] = _trend_from_delta(delta, delta_pct, lower, compare_label)
        elif key in series_blocks:
            block = series_blocks[key].get("period") or series_blocks[key]
            trends[key] = _trend_from_delta(block.get("delta"), block.get("delta_pct"), lower, compare_label)
        else:
            trends[key] = _trend_from_delta(None, None, lower, compare_label)

    # past_due tile uses past_due_pct trend
    if "past_due_pct" in trends:
        trends["past_due"] = trends["past_due_pct"]
    return trends


def _company_filter(company: str | None, prefix: str = "") -> str:
    if not company:
        return ""
    safe = company.replace("'", "''")
    col = f"{prefix}COMPANY_CODE" if prefix else "COMPANY_CODE"
    return f" AND {col} = '{safe}'"


def _dealer_filter(dealer_id: str | None, prefix: str = "") -> str:
    if not dealer_id:
        return ""
    safe = dealer_id.replace("'", "''")
    col = f"{prefix}CUSTOMER_ID" if prefix else "CUSTOMER_ID"
    return f" AND {col} = '{safe}'"


def _status_filter(status: str | None, prefix: str = "") -> str:
    if not status:
        return ""
    safe = status.replace("'", "''").upper()
    col = f"{prefix}ORDER_STATUS" if prefix else "ORDER_STATUS"
    return f" AND UPPER({col}) = '{safe}'"


def _date_filter(date_from: str | None, date_to: str | None, col: str = "ORDER_DATE") -> str:
    parts = []
    if date_from:
        safe = date_from.replace("'", "''")[:10]
        parts.append(f" AND {col} >= '{safe}'::DATE")
    if date_to:
        safe = date_to.replace("'", "''")[:10]
        parts.append(f" AND {col} <= '{safe}'::DATE")
    return "".join(parts)


def _scoped_order_date_filter(date_from: str | None, date_to: str | None) -> str:
    return _date_filter(date_from, date_to, "COALESCE(so.ORDER_DATE, inv.INVOICE_DATE)")


def _has_ops_filters(
    dealer_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> bool:
    return bool(dealer_id or date_from or date_to)


def _scoped_ar_from(dealer_id: str | None, date_from: str | None, date_to: str | None) -> str:
    df = _dealer_filter(dealer_id, "ar.")
    dtf = _scoped_order_date_filter(date_from, date_to)
    return f"""
        FROM {BM}.ar_open_item_vw ar
        LEFT JOIN {BM}.ar_invoice_vw inv ON inv.INVOICE_ID = ar.INVOICE_ID
        LEFT JOIN {BM}.sales_order_vw so ON so.SALES_ORDER_ID = inv.SALES_ORDER_ID
        WHERE 1=1 {df}{dtf}
    """


def _scoped_invoice_from(dealer_id: str | None, date_from: str | None, date_to: str | None) -> str:
    df = _dealer_filter(dealer_id, "inv.")
    dtf = _scoped_order_date_filter(date_from, date_to)
    return f"""
        FROM {BM}.ar_invoice_vw inv
        LEFT JOIN {BM}.sales_order_vw so ON so.SALES_ORDER_ID = inv.SALES_ORDER_ID
        WHERE 1=1 {df}{dtf}
    """


def _scoped_order_from(dealer_id: str | None, date_from: str | None, date_to: str | None) -> str:
    df = _dealer_filter(dealer_id, "so.")
    dtf = _date_filter(date_from, date_to, "so.ORDER_DATE")
    return f"""
        FROM {BM}.sales_order_vw so
        WHERE 1=1 {df}{dtf}
    """


def get_filter_options() -> dict:
    cached = _cache_get("filters")
    if cached is not None:
        return cached

    q = _run_queries_parallel({
        "companies": f"""
            SELECT DISTINCT COMPANY_CODE
            FROM {BM}.sales_order_vw
            WHERE COMPANY_CODE IS NOT NULL
            ORDER BY COMPANY_CODE
        """,
        "segments": f"""
            SELECT DISTINCT CUSTOMER_SEGMENT
            FROM {BM}.customer_vw
            WHERE CUSTOMER_SEGMENT IS NOT NULL
            ORDER BY CUSTOMER_SEGMENT
        """,
        "dealers": f"""
            SELECT DISTINCT
                c.CUSTOMER_ID AS dealer_id,
                c.CUSTOMER_NAME AS dealer_name
            FROM {BM}.customer_vw c
            INNER JOIN {BM}.sales_order_vw so ON so.CUSTOMER_ID = c.CUSTOMER_ID
            WHERE c.CUSTOMER_NAME IS NOT NULL
            ORDER BY c.CUSTOMER_NAME
            LIMIT 500
        """,
    })
    companies = q["companies"]
    segments = q["segments"]
    dealers = q["dealers"]
    result = {
        "companies": [r.get("company_code") or r.get("COMPANY_CODE") for r in companies],
        "segments": [r.get("customer_segment") or r.get("CUSTOMER_SEGMENT") for r in segments],
        "dealers": [
            {
                "dealer_id": r.get("dealer_id") or r.get("DEALER_ID"),
                "dealer_name": r.get("dealer_name") or r.get("DEALER_NAME"),
            }
            for r in dealers
        ],
        "customers": [
            {
                "customer_id": r.get("dealer_id") or r.get("DEALER_ID"),
                "customer_name": r.get("dealer_name") or r.get("DEALER_NAME"),
            }
            for r in dealers
        ],
    }
    _cache_set("filters", result)
    return result


def get_summary(
    company: str | None = None,
    dealer_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    period_preset: str | None = None,
) -> dict:
    cache_key = f"summary:{company or 'all'}:{dealer_id or 'all'}:{date_from or ''}:{date_to or ''}:{period_preset or ''}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    scoped = _has_ops_filters(dealer_id, date_from, date_to)
    cf = _company_filter(company)
    add_trend_rows: list[dict] = []
    ar_pct_trend_rows: list[dict] = []
    past_due_pct_trend: list[dict] = []

    if scoped:
        ar_from = _scoped_ar_from(dealer_id, date_from, date_to)
        inv_from = _scoped_invoice_from(dealer_id, date_from, date_to)
        ord_from = _scoped_order_from(dealer_id, date_from, date_to)

        past_due = run_query(f"""
            SELECT
                SUM(ar.OPEN_AMOUNT_USD) AS total_ar,
                SUM(IFF(ar.DAYS_PAST_DUE > 0, ar.OPEN_AMOUNT_USD, 0)) AS past_due_ar,
                SUM(IFF(ar.AGING_BUCKET = '1-30', ar.OPEN_AMOUNT_USD, 0)) AS bucket_1_30,
                SUM(IFF(ar.AGING_BUCKET = '31-60', ar.OPEN_AMOUNT_USD, 0)) AS bucket_31_60,
                SUM(IFF(ar.AGING_BUCKET = '61-90', ar.OPEN_AMOUNT_USD, 0)) AS bucket_61_90,
                SUM(IFF(ar.AGING_BUCKET = '90+', ar.OPEN_AMOUNT_USD, 0)) AS bucket_90_plus
            {ar_from}
        """)

        add_row = run_query(f"""
            SELECT AVG(ar.DAYS_PAST_DUE) AS add_val
            {ar_from}
              AND ar.DAYS_PAST_DUE > 0
        """)

        inv_90 = run_query(f"""
            SELECT SUM(inv.GROSS_AMOUNT) AS invoiced_90d
            {inv_from}
              AND inv.INVOICE_DATE >= DATEADD(day, -90, CURRENT_DATE())
        """)

        dso_rows = run_query(f"""
            SELECT
                DATE_TRUNC('month', inv.INVOICE_DATE)::VARCHAR AS period,
                AVG(GREATEST(ar.DAYS_PAST_DUE, 0) + COALESCE(ar.PAYMENT_TERM_DAYS, 30)) AS dso
            {ar_from}
              AND inv.INVOICE_DATE IS NOT NULL
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 12
        """)
        dso_rows.reverse()

        cei_rows = run_query(f"""
            SELECT
                DATE_TRUNC('month', inv.INVOICE_DATE)::VARCHAR AS period,
                100 * SUM(IFF(UPPER(inv.INVOICE_STATUS) = 'PAID', inv.GROSS_AMOUNT, 0))
                    / NULLIF(SUM(inv.GROSS_AMOUNT), 0) AS cei
            {inv_from}
              AND inv.INVOICE_DATE IS NOT NULL
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 12
        """)
        cei_rows.reverse()

        ar_pct_row = run_query(f"""
            SELECT
                100 * SUM(ar.OPEN_AMOUNT_USD)
                    / NULLIF(SUM(so.TOTAL_ORDER_VALUE), 0) AS ar_pct
            {ar_from}
        """)

        forecast = run_query(f"""
            SELECT AVG(f.MAPE) AS mape
            FROM {BM}.forecast_accuracy_vw f
            WHERE f.COMPANY_CODE IN (
                SELECT DISTINCT inv.COMPANY_CODE
                {inv_from}
                  AND inv.COMPANY_CODE IS NOT NULL
            )
        """)

        order_trend = run_query(f"""
            SELECT
                DATE_TRUNC('month', so.ORDER_DATE)::VARCHAR AS period,
                COUNT(*) AS order_count,
                SUM(so.TOTAL_ORDER_VALUE) AS order_value
            {ord_from}
            GROUP BY 1
            ORDER BY 1
        """)

        invoice_by_status = run_query(f"""
            SELECT
                inv.INVOICE_STATUS AS status,
                COUNT(*) AS invoice_count,
                SUM(inv.GROSS_AMOUNT) AS gross_amount
            {inv_from}
            GROUP BY 1
            ORDER BY gross_amount DESC
        """)

        past_due_trend = run_query(f"""
            SELECT
                DATE_TRUNC('month', ar.DUE_DATE)::VARCHAR AS period,
                SUM(ar.OPEN_AMOUNT_USD) AS past_due_ar
            {ar_from}
              AND ar.DAYS_PAST_DUE > 0
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 24
        """)
        past_due_trend.reverse()

        pd = past_due[0] if past_due else {}
        total_ar = float(pd.get("TOTAL_AR") or pd.get("total_ar") or 0)
        past_due_ar = float(pd.get("PAST_DUE_AR") or pd.get("past_due_ar") or 0)
        invoiced_90d = float((inv_90[0] if inv_90 else {}).get("INVOICED_90D") or (inv_90[0] if inv_90 else {}).get("invoiced_90d") or 0)
        dso_proxy = (total_ar / (invoiced_90d / 90.0)) if invoiced_90d > 0 else None
        if dso_proxy is None and dso_rows:
            dso_proxy = dso_rows[-1].get("dso") or dso_rows[-1].get("DSO")

        avg_add = (add_row[0] if add_row else {}).get("ADD_VAL") or (add_row[0] if add_row else {}).get("add_val")
        latest_cei = cei_rows[-1].get("cei") or cei_rows[-1].get("CEI") if cei_rows else None
        ar_pct_val = (ar_pct_row[0] if ar_pct_row else {}).get("AR_PCT") or (ar_pct_row[0] if ar_pct_row else {}).get("ar_pct")
    else:
        ord_from = (
            _scoped_order_from(None, date_from, date_to)
            if (date_from or date_to)
            else f"FROM {BM}.sales_order_vw so WHERE 1=1 {cf.replace('COMPANY_CODE', 'so.COMPANY_CODE')}"
        )
        cf_ar = cf.replace("COMPANY_CODE", "ar.COMPANY_CODE")
        cf_inv = cf.replace("COMPANY_CODE", "inv.COMPANY_CODE")
        q = _run_queries_parallel({
            "dso_rows": f"""
                SELECT PERIOD_MONTH::VARCHAR AS period, AVG(DSO) AS dso
                FROM {BM}.dso_vw
                WHERE DSO IS NOT NULL {cf.replace('COMPANY_CODE', 'COMPANY_CODE')}
                GROUP BY PERIOD_MONTH
                ORDER BY PERIOD_MONTH DESC
                LIMIT 12
            """,
            "add_rows": f"""
                SELECT COMPANY_CODE, AVG_DAYS_DELINQUENT AS add_val
                FROM {BM}.add_vw
                WHERE 1=1 {cf}
            """,
            "past_due": f"""
                SELECT
                    SUM(TOTAL_AR) AS total_ar,
                    SUM(PAST_DUE_AR) AS past_due_ar,
                    AVG(PAST_DUE_PCT) AS past_due_pct,
                    SUM(BUCKET_1_30) AS bucket_1_30,
                    SUM(BUCKET_31_60) AS bucket_31_60,
                    SUM(BUCKET_61_90) AS bucket_61_90,
                    SUM(BUCKET_90_PLUS) AS bucket_90_plus
                FROM {BM}.past_due_vw
                WHERE 1=1 {cf}
            """,
            "cei_rows": f"""
                SELECT PERIOD_MONTH::VARCHAR AS period, AVG(CEI) AS cei
                FROM {BM}.cei_vw
                WHERE CEI IS NOT NULL {cf}
                GROUP BY PERIOD_MONTH
                ORDER BY PERIOD_MONTH DESC
                LIMIT 12
            """,
            "ar_pct": f"""
                SELECT AVG(AR_PCT_OF_REVENUE) AS ar_pct
                FROM {BM}.ar_pct_revenue_vw
                WHERE 1=1 {cf}
            """,
            "forecast": f"""
                SELECT AVG(MAPE) AS mape, AVG(WAPE) AS wape
                FROM {BM}.forecast_accuracy_vw
                WHERE 1=1 {cf.replace('COMPANY_CODE', 'COMPANY_CODE')}
            """,
            "order_trend": f"""
                SELECT
                    DATE_TRUNC('month', so.ORDER_DATE)::VARCHAR AS period,
                    COUNT(*) AS order_count,
                    SUM(so.TOTAL_ORDER_VALUE) AS order_value
                {ord_from}
                GROUP BY 1
                ORDER BY 1
            """,
            "invoice_by_status": f"""
                SELECT
                    inv.INVOICE_STATUS AS status,
                    COUNT(*) AS invoice_count,
                    SUM(inv.GROSS_AMOUNT) AS gross_amount
                FROM {BM}.ar_invoice_vw inv
                WHERE 1=1 {cf_inv}
                GROUP BY 1
                ORDER BY gross_amount DESC
            """,
            "past_due_trend": f"""
                SELECT
                    DATE_TRUNC('month', ar.DUE_DATE)::VARCHAR AS period,
                    SUM(ar.OPEN_AMOUNT_USD) AS past_due_ar
                FROM {BM}.ar_open_item_vw ar
                WHERE ar.DAYS_PAST_DUE > 0 {cf_ar}
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT 24
            """,
            "add_trend_rows": f"""
                SELECT DATE_TRUNC('month', ar.DUE_DATE)::VARCHAR AS period,
                       AVG(ar.DAYS_PAST_DUE) AS add_val
                FROM {BM}.ar_open_item_vw ar
                WHERE ar.DAYS_PAST_DUE > 0 {cf_ar}
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT 24
            """,
            "ar_pct_trend_rows": f"""
                SELECT PERIOD_MONTH::VARCHAR AS period, AVG(AR_PCT_OF_REVENUE) AS ar_pct
                FROM {BM}.ar_pct_revenue_vw
                WHERE AR_PCT_OF_REVENUE IS NOT NULL {cf}
                GROUP BY PERIOD_MONTH
                ORDER BY PERIOD_MONTH DESC
                LIMIT 24
            """,
            "past_due_pct_trend": f"""
                SELECT DATE_TRUNC('month', ar.DUE_DATE)::VARCHAR AS period,
                       SUM(IFF(ar.DAYS_PAST_DUE > 0, ar.OPEN_AMOUNT_USD, 0))
                           / NULLIF(SUM(ar.OPEN_AMOUNT_USD), 0) * 100 AS past_due_pct
                FROM {BM}.ar_open_item_vw ar
                WHERE 1=1 {cf_ar}
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT 24
            """,
        })
        dso_rows = q["dso_rows"]
        dso_rows.reverse()
        add_rows = q["add_rows"]
        past_due = q["past_due"]
        cei_rows = q["cei_rows"]
        cei_rows.reverse()
        ar_pct = q["ar_pct"]
        forecast = q["forecast"]
        order_trend = q["order_trend"]
        invoice_by_status = q["invoice_by_status"]
        past_due_trend = q["past_due_trend"]
        past_due_trend.reverse()
        add_trend_rows = q["add_trend_rows"]
        add_trend_rows.reverse()
        ar_pct_trend_rows = q["ar_pct_trend_rows"]
        ar_pct_trend_rows.reverse()
        past_due_pct_trend = q["past_due_pct_trend"]
        past_due_pct_trend.reverse()

        pd = past_due[0] if past_due else {}
        total_ar = float(pd.get("TOTAL_AR") or pd.get("total_ar") or 0)
        past_due_ar = float(pd.get("PAST_DUE_AR") or pd.get("past_due_ar") or 0)
        dso_proxy = dso_rows[-1].get("dso") if dso_rows else None
        vals = [
            float(r.get("ADD_VAL") or r.get("add_val") or r.get("AVG_DAYS_DELINQUENT") or 0)
            for r in add_rows
            if r.get("ADD_VAL") is not None or r.get("add_val") is not None or r.get("AVG_DAYS_DELINQUENT") is not None
        ]
        avg_add = sum(vals) / len(vals) if vals else None
        latest_cei = cei_rows[-1].get("cei") or cei_rows[-1].get("CEI") if cei_rows else None
        ar_pct_val = ar_pct[0].get("AR_PCT") or ar_pct[0].get("ar_pct") if ar_pct else None

    past_due_pct = (100.0 * past_due_ar / total_ar) if total_ar > 0 else float(
        pd.get("PAST_DUE_PCT") or pd.get("past_due_pct") or 0
    )

    kpis = {
        "dso": round(float(dso_proxy), 1) if dso_proxy is not None else None,
        "add": round(float(avg_add), 1) if avg_add is not None else None,
        "past_due_pct": round(past_due_pct, 1),
        "past_due_ar": past_due_ar,
        "total_ar": total_ar,
        "cei": round(float(latest_cei), 1) if latest_cei is not None else None,
        "ar_pct_revenue": round(float(ar_pct_val), 1) if ar_pct_val is not None else None,
        "forecast_mape": round(float(forecast[0].get("MAPE") or forecast[0].get("mape") or 0), 1) if forecast else None,
    }

    n_recent, n_prior, compare_label = _blocks_for_preset(period_preset)

    prior_kpis = None
    if date_from and date_to:
        pf, pt = _prior_period_bounds(date_from, date_to)
        prior_kpis = _scoped_prior_kpis(company, dealer_id, pf, pt)

    series_blocks = {
        "dso": _comparison_from_series(dso_rows, "dso", "DSO", "value", n_recent=n_recent, n_prior=n_prior),
        "cei": _comparison_from_series(cei_rows, "cei", "CEI", "value", n_recent=n_recent, n_prior=n_prior),
        "add": _comparison_from_series(add_trend_rows, "add_val", "ADD_VAL", "value", n_recent=n_recent, n_prior=n_prior),
        "past_due_ar": _comparison_from_series(past_due_trend, "past_due_ar", "PAST_DUE_AR", n_recent=n_recent, n_prior=n_prior),
        "past_due_pct": _comparison_from_series(
            past_due_pct_trend or past_due_trend,
            "past_due_pct", "PAST_DUE_PCT", "past_due_ar", "PAST_DUE_AR",
            n_recent=n_recent, n_prior=n_prior,
        ),
        "ar_pct_revenue": _comparison_from_series(ar_pct_trend_rows, "ar_pct", "AR_PCT", "value", n_recent=n_recent, n_prior=n_prior),
    }

    kpi_trends = _build_kpi_trends(kpis, prior_kpis, series_blocks, compare_label)

    result = {
        "updated_at": time.time(),
        "filters": {
            "dealer_id": dealer_id,
            "date_from": date_from,
            "date_to": date_to,
            "company": company,
            "period_preset": period_preset,
        },
        "kpis": kpis,
        "aging_buckets": {
            "1-30": float(pd.get("BUCKET_1_30") or pd.get("bucket_1_30") or 0),
            "31-60": float(pd.get("BUCKET_31_60") or pd.get("bucket_31_60") or 0),
            "61-90": float(pd.get("BUCKET_61_90") or pd.get("bucket_61_90") or 0),
            "90+": float(pd.get("BUCKET_90_PLUS") or pd.get("bucket_90_plus") or 0),
        },
        "dso_trend": [
            {"period": str(r.get("period") or r.get("PERIOD") or "")[:10], "value": float(r.get("dso") or r.get("DSO") or 0)}
            for r in dso_rows
        ],
        "cei_trend": [
            {"period": str(r.get("period") or r.get("PERIOD") or "")[:10], "value": float(r.get("cei") or r.get("CEI") or 0)}
            for r in cei_rows
        ],
        "order_trend": [
            {
                "period": str(r.get("period") or r.get("PERIOD") or "")[:10],
                "order_count": int(r.get("ORDER_COUNT") or r.get("order_count") or 0),
                "order_value": float(r.get("ORDER_VALUE") or r.get("order_value") or 0),
            }
            for r in order_trend
        ],
        "invoice_by_status": [
            {
                "status": str(r.get("STATUS") or r.get("status") or ""),
                "invoice_count": int(r.get("INVOICE_COUNT") or r.get("invoice_count") or 0),
                "gross_amount": float(r.get("GROSS_AMOUNT") or r.get("gross_amount") or 0),
            }
            for r in invoice_by_status
        ],
        "period_comparisons": {
            "dso": series_blocks["dso"],
            "cei": series_blocks["cei"],
            "past_due_ar": series_blocks["past_due_ar"],
            "past_due_pct": series_blocks["past_due_pct"],
        },
        "kpi_trends": kpi_trends,
        "compare_label": compare_label,
        "kpi_series_extra": _fetch_kpi_series_extra(company, dealer_id, date_from, date_to),
    }
    _cache_set(cache_key, result)
    return result


def _fetch_kpi_series_extra(
    company: str | None,
    dealer_id: str | None,
    date_from: str | None,
    date_to: str | None,
) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_add = pool.submit(get_kpi_series, "add", company, dealer_id, date_from, date_to)
        f_pd = pool.submit(get_kpi_series, "past_due", company, dealer_id, date_from, date_to)
        add_rows = f_add.result()
        past_due_rows = f_pd.result()
    return {
        "add": _serialize_kpi_series(add_rows),
        "past_due": _serialize_kpi_series(past_due_rows),
    }


def get_orders(
    company: str | None = None,
    dealer_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    order_status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    cf = _company_filter(company, "so.")
    df = _dealer_filter(dealer_id, "so.")
    dtf = _date_filter(date_from, date_to, "so.ORDER_DATE")
    sf = _status_filter(order_status, "so.")
    return run_query(f"""
        SELECT
            so.SALES_ORDER_ID AS sales_order_id,
            so.CUSTOMER_ID AS customer_id,
            cu.CUSTOMER_NAME AS customer_name,
            so.COMPANY_CODE AS company_code,
            so.ORDER_STATUS AS order_status,
            so.ORDER_DATE::VARCHAR AS order_date,
            so.TOTAL_ORDER_VALUE AS total_order_value,
            so.CURRENCY_CODE AS currency_code,
            so.CREDIT_CHECK_STATUS AS credit_check_status
        FROM {BM}.sales_order_vw so
        LEFT JOIN {BM}.customer_vw cu ON cu.CUSTOMER_ID = so.CUSTOMER_ID
        WHERE 1=1 {cf}{df}{dtf}{sf}
        ORDER BY so.ORDER_DATE DESC, so.SALES_ORDER_ID DESC
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)


def get_order_timeline(order_id: str) -> list[dict]:
    safe_id = order_id.replace("'", "''")
    return run_query(f"""
        SELECT
            SALES_ORDER_ID AS sales_order_id,
            STATUS_SEQ AS status_seq,
            LIFECYCLE_AREA AS lifecycle_area,
            ORDER_STATUS AS order_status,
            STATUS_DATE::VARCHAR AS status_date,
            STATUS_TIMESTAMP::VARCHAR AS status_timestamp,
            IS_LATEST_STATUS AS is_latest_status,
            CHANGED_BY AS changed_by,
            NOTES AS notes
        FROM {BM}.order_status_history_vw
        WHERE SALES_ORDER_ID = '{safe_id}'
        ORDER BY STATUS_SEQ
    """)


_INVOICE_SELECT = f"""
    SELECT
        inv.INVOICE_ID AS invoice_id,
        inv.SALES_ORDER_ID AS sales_order_id,
        inv.CUSTOMER_ID AS customer_id,
        cu.CUSTOMER_NAME AS customer_name,
        inv.COMPANY_CODE AS company_code,
        inv.INVOICE_DATE::VARCHAR AS invoice_date,
        inv.DUE_DATE::VARCHAR AS due_date,
        COALESCE(
            pay.PAYMENT_DATE,
            clr.CLEARING_DATE,
            IFF(UPPER(inv.INVOICE_STATUS) = 'PAID', LEAST(inv.DUE_DATE, CURRENT_DATE()), NULL)
        )::VARCHAR AS payment_date,
        inv.GROSS_AMOUNT AS gross_amount,
        inv.CURRENCY_CODE AS currency_code,
        inv.INVOICE_STATUS AS invoice_status
    FROM {{bm}}.ar_invoice_vw inv
    LEFT JOIN {{bm}}.customer_vw cu ON cu.CUSTOMER_ID = inv.CUSTOMER_ID
    LEFT JOIN (
        SELECT INVOICE_ID, MAX(PAYMENT_DATE) AS PAYMENT_DATE
        FROM {{bm}}.payment_vw
        WHERE INVOICE_ID IS NOT NULL
        GROUP BY INVOICE_ID
    ) pay ON pay.INVOICE_ID = inv.INVOICE_ID
    LEFT JOIN (
        SELECT INVOICE_ID, MAX(CLEARING_DATE) AS CLEARING_DATE
        FROM {{bm}}.ar_cleared_item_vw
        WHERE INVOICE_ID IS NOT NULL
        GROUP BY INVOICE_ID
    ) clr ON clr.INVOICE_ID = inv.INVOICE_ID
"""


def get_invoices(
    company: str | None = None,
    dealer_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    cf = _company_filter(company, "inv.")
    df = _dealer_filter(dealer_id, "inv.")
    dtf = _date_filter(date_from, date_to, "inv.INVOICE_DATE")
    return run_query(f"""
        {_INVOICE_SELECT.format(bm=BM)}
        WHERE 1=1 {cf}{df}{dtf}
        ORDER BY inv.INVOICE_DATE DESC
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)


def get_order_invoices(order_id: str) -> list[dict]:
    safe_id = order_id.replace("'", "''")
    return run_query(f"""
        {_INVOICE_SELECT.format(bm=BM)}
        WHERE inv.SALES_ORDER_ID = '{safe_id}'
        ORDER BY inv.INVOICE_DATE
    """)


def get_data_version() -> dict:
    """Lightweight fingerprint — polled by UI; full pages refetch only when this changes."""
    rows = run_query(f"""
        SELECT
            (SELECT COUNT(*) FROM {BM}.sales_order_vw) AS order_cnt,
            (SELECT COUNT(*) FROM {BM}.ar_open_item_vw) AS ar_cnt,
            (SELECT COUNT(*) FROM {BM}.ar_invoice_vw) AS inv_cnt,
            (SELECT COALESCE(MAX(STATUS_TIMESTAMP)::VARCHAR, '') FROM {BM}.order_status_history_vw) AS last_status_ts,
            (SELECT COALESCE(SUM(PAST_DUE_AR), 0) FROM {BM}.past_due_vw) AS past_due_total,
            (SELECT COALESCE(MAX(ACTIVITY_DATE)::VARCHAR, '') FROM {BM}.collection_activity_vw) AS last_collection_dt
    """)
    r = rows[0] if rows else {}
    parts = [
        str(r.get("ORDER_CNT") or r.get("order_cnt") or 0),
        str(r.get("AR_CNT") or r.get("ar_cnt") or 0),
        str(r.get("INV_CNT") or r.get("inv_cnt") or 0),
        str(r.get("LAST_STATUS_TS") or r.get("last_status_ts") or ""),
        str(r.get("PAST_DUE_TOTAL") or r.get("past_due_total") or 0),
        str(r.get("LAST_COLLECTION_DT") or r.get("last_collection_dt") or ""),
    ]
    version = "|".join(parts)
    return {"version": version, "checked_at": time.time()}


def _serialize_kpi_series(rows: list[dict]) -> list[dict]:
    return [
        {
            "period": (str(r.get("period") or r.get("PERIOD") or "")[:10] or None),
            "label": (str(r.get("label") or r.get("LABEL") or "") or None),
            "value": float(r.get("value") or r.get("VALUE") or 0),
        }
        for r in rows
    ]


def get_kpi_series(
    metric: str,
    company: str | None = None,
    dealer_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    cache_key = f"kpi_series:{metric}:{company or 'all'}:{dealer_id or 'all'}:{date_from or ''}:{date_to or ''}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    rows = _fetch_kpi_series(metric, company, dealer_id, date_from, date_to)
    _cache_set(cache_key, rows)
    return rows


def _fetch_kpi_series(
    metric: str,
    company: str | None = None,
    dealer_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    if _has_ops_filters(dealer_id, date_from, date_to):
        ar_from = _scoped_ar_from(dealer_id, date_from, date_to)
        inv_from = _scoped_invoice_from(dealer_id, date_from, date_to)
        if metric == "dso":
            return run_query(f"""
                SELECT DATE_TRUNC('month', inv.INVOICE_DATE)::VARCHAR AS period,
                       AVG(GREATEST(ar.DAYS_PAST_DUE, 0) + COALESCE(ar.PAYMENT_TERM_DAYS, 30)) AS value
                {ar_from}
                  AND inv.INVOICE_DATE IS NOT NULL
                GROUP BY 1 ORDER BY 1
            """)
        if metric == "add":
            return run_query(f"""
                SELECT ar.AGING_BUCKET AS label, AVG(ar.DAYS_PAST_DUE) AS value
                {ar_from}
                  AND ar.DAYS_PAST_DUE > 0
                GROUP BY 1 ORDER BY 1
            """)
        if metric == "cei":
            return run_query(f"""
                SELECT DATE_TRUNC('month', inv.INVOICE_DATE)::VARCHAR AS period,
                       100 * SUM(IFF(UPPER(inv.INVOICE_STATUS) = 'PAID', inv.GROSS_AMOUNT, 0))
                           / NULLIF(SUM(inv.GROSS_AMOUNT), 0) AS value
                {inv_from}
                  AND inv.INVOICE_DATE IS NOT NULL
                GROUP BY 1 ORDER BY 1
            """)
        if metric == "past_due":
            return run_query(f"""
                SELECT ar.AGING_BUCKET AS label, SUM(ar.OPEN_AMOUNT_USD) AS value
                {ar_from}
                  AND ar.DAYS_PAST_DUE > 0
                GROUP BY 1 ORDER BY 1
            """)
        return []

    cf = _company_filter(company)
    views = {
        "dso": ("dso_vw", "DSO", "PERIOD_MONTH"),
        "add": ("add_vw", "AVG_DAYS_DELINQUENT", None),
        "cei": ("cei_vw", "CEI", "PERIOD_MONTH"),
        "past_due": ("past_due_vw", "PAST_DUE_PCT", None),
    }
    if metric not in views:
        return []
    view, col, period_col = views[metric]
    if period_col:
        return run_query(f"""
            SELECT {period_col}::VARCHAR AS period, AVG({col}) AS value
            FROM {BM}.{view}
            WHERE {col} IS NOT NULL {cf}
            GROUP BY {period_col}
            ORDER BY {period_col}
        """)
    return run_query(f"""
        SELECT COMPANY_CODE AS label, {col} AS value
        FROM {BM}.{view}
        WHERE {col} IS NOT NULL {cf}
        ORDER BY COMPANY_CODE
    """)


def refresh_data() -> dict:
    """Fabric counterpart of the Snowflake refresh.

    Snowflake runs ALTER DYNAMIC TABLE ... REFRESH on the INFORMATION_MART
    dynamic tables. Fabric has no dynamic tables: if the mart is materialised
    by stored procedures, list them in FABRIC_REFRESH_PROCEDURES
    (comma-separated, schema-qualified, e.g. "RAW_VAULT.SP_REFRESH_MART");
    otherwise the mart objects are views and are already current. Either way
    the API caches are cleared, as on Snowflake.
    """
    import os

    from FabicApp.db import DB
    procs = [p.strip() for p in os.getenv("FABRIC_REFRESH_PROCEDURES", "").split(",") if p.strip()]
    for proc in procs:
        try:
            run_tsql(f"EXEC {DB}.{proc}")
        except Exception:
            pass
    with _CACHE_LOCK:
        _CACHE.clear()
    return {"ok": True, "message": "Refresh initiated for Fabric warehouse data", "version": get_data_version()["version"]}


WEEKLY_METRIC_DEFS: list[dict] = [
    {"id": "dso", "label": "DSO (Days Sales Outstanding)", "format": "days", "lower_better": True},
    {"id": "collections", "label": "Collections", "format": "currency", "lower_better": False},
    {"id": "amount_due", "label": "Amount Due (per Terms)", "format": "currency", "lower_better": False},
    {"id": "expected_collections", "label": "Expected Collections", "format": "currency", "lower_better": False},
    {"id": "early_payer_expected", "label": "Early Payer Expected", "format": "currency", "lower_better": False},
    {"id": "late_payer_expected", "label": "Late Payer Expected", "format": "currency", "lower_better": True},
    {"id": "collections_efficiency", "label": "Collections Efficiency", "format": "percent", "lower_better": False},
    {"id": "open_invoices", "label": "Open Invoices", "format": "count", "lower_better": True},
    {"id": "overdue_invoices", "label": "Overdue Invoices", "format": "count", "lower_better": True},
    {"id": "overdue_amount", "label": "Overdue Amount", "format": "currency", "lower_better": True},
    {"id": "disputes_open", "label": "Disputes Open", "format": "count", "lower_better": True},
    {"id": "dispute_resolution_time", "label": "Dispute Resolution Time", "format": "days", "lower_better": True},
    {"id": "invoice_blocked", "label": "Invoice Blocked", "format": "count", "lower_better": True},
    {"id": "revenue", "label": "Revenue", "format": "currency", "lower_better": False},
    {"id": "revenue_at_risk", "label": "Revenue at Risk", "format": "currency", "lower_better": True},
    {"id": "invoice_accuracy", "label": "Invoice-Level Accuracy", "format": "percent", "lower_better": False},
    {"id": "customer_exposure", "label": "Customer-Level Exposure", "format": "currency", "lower_better": True},
    {"id": "remittance_match_rate", "label": "Remittance Match Rate", "format": "percent", "lower_better": False},
    {"id": "unapplied_cash", "label": "Unapplied Cash", "format": "currency", "lower_better": True},
    {"id": "deductions", "label": "Deductions", "format": "currency", "lower_better": True},
    {"id": "short_payments", "label": "Short Payments", "format": "count", "lower_better": True},
    {"id": "credit_memo_pending", "label": "Credit Memo Pending", "format": "count", "lower_better": True},
    {"id": "promise_to_pay", "label": "Promise to Pay", "format": "count", "lower_better": False},
    {"id": "cash_app_cycle_time", "label": "Cash Application Cycle Time", "format": "days", "lower_better": True},
]


def _monday_of_week(anchor: str | None = None) -> str:
    from datetime import date, datetime, timedelta

    if anchor:
        d = datetime.strptime(anchor[:10], "%Y-%m-%d").date()
    else:
        d = date.today()
    monday = d - timedelta(days=d.weekday())
    return monday.strftime("%Y-%m-%d")


def _is_weekday(d) -> bool:
    return d.weekday() < 5


def _prior_weekdays(before, count: int) -> list:
    from datetime import timedelta

    out: list = []
    d = before - timedelta(days=1)
    while len(out) < count:
        if _is_weekday(d):
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def _next_weekdays(after, count: int) -> list:
    from datetime import timedelta

    out: list = []
    d = after + timedelta(days=1)
    while len(out) < count:
        if _is_weekday(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _weekly_metrics_window(anchor) -> list:
    """3 prior weekdays, anchor (weekdays only), then 3 future weekdays — no Sat/Sun."""
    from datetime import timedelta

    if _is_weekday(anchor):
        return _prior_weekdays(anchor, 3) + [anchor] + _next_weekdays(anchor, 3)

    friday = anchor - timedelta(days=anchor.weekday() - 4)
    past: list = []
    d = friday
    while len(past) < 3:
        past.insert(0, d)
        d -= timedelta(days=1)
        while not _is_weekday(d):
            d -= timedelta(days=1)
    return past + _next_weekdays(friday, 3)


def _adjacent_weekday(d, direction: int):
    from datetime import timedelta

    step = timedelta(days=1 if direction > 0 else -1)
    n = d + step
    while not _is_weekday(n):
        n += step
    return n


def _weekly_horizon_decay(day_cells: list[dict], idx: int, today) -> float:
    from datetime import datetime

    md = datetime.strptime(day_cells[idx]["date"][:10], "%Y-%m-%d").date()
    days_ahead = max(0, (md - today).days)
    if days_ahead <= 0:
        return 1.0
    return max(0.72, 1.0 - 0.025 * days_ahead)


def _weekly_mape_adjust(base_mape: float) -> float:
    return max(0.85, min(1.05, 1.0 - (float(base_mape or 12) - 12.0) / 100.0))


def _weekly_cell_confidence(
    metric_id: str,
    idx: int,
    day_cells: list[dict],
    *,
    today,
    base_mape: float,
) -> float | None:
    if not day_cells[idx].get("is_projected"):
        return None
    facts = day_cells[idx]["facts"]
    decay = _weekly_horizon_decay(day_cells, idx, today)
    mape_adj = _weekly_mape_adjust(base_mape)
    due_conf = float(facts.get("amount_due_confidence") or 90.0)
    expected_conf = float(facts.get("expected_confidence") or 52.0)

    if metric_id == "amount_due":
        base = due_conf if facts.get("amount_due", 0) > 0 else 88.0
    elif metric_id in ("expected_collections", "early_payer_expected", "late_payer_expected"):
        base = expected_conf
    elif metric_id == "collections":
        base = expected_conf
    elif metric_id == "collections_efficiency":
        base = round((due_conf + expected_conf) / 2.0, 1)
    elif metric_id == "dso":
        base = round(58.0 + (100.0 - float(base_mape or 12)) * 0.35, 1)
    elif metric_id == "revenue_at_risk":
        base = 68.0
    elif metric_id in ("open_invoices", "overdue_invoices", "overdue_amount", "disputes_open"):
        base = 62.0
    elif metric_id in ("invoice_blocked", "revenue", "invoice_accuracy"):
        base = 60.0
    else:
        base = 58.0

    return round(min(95.0, max(40.0, base * decay * mape_adj)), 1)


def _build_weekly_metrics_for_days(
    day_cells: list[dict],
    *,
    today,
    base_dso: float,
    base_cei: float,
    base_mape: float,
    open_cnt: float,
    open_amt: float,
) -> list[dict]:
    """Build metric rows (with cells) for one business-day window."""

    def _actual_metric(metric_id: str, idx: int) -> float:
        f = day_cells[idx]["facts"]
        rev = f["revenue"]
        coll = f["collections"]
        if metric_id == "dso":
            if rev > 0:
                return round(base_dso * max(0.75, min(1.35, 1.0 + 0.15 * (coll / rev - 0.85))), 1)
            return round(base_dso, 1)
        if metric_id == "collections":
            return f["collections"]
        if metric_id == "amount_due":
            return f["amount_due"]
        if metric_id == "expected_collections":
            return f["expected_collections"]
        if metric_id == "early_payer_expected":
            return f["early_payer_expected"]
        if metric_id == "late_payer_expected":
            return f["late_payer_expected"]
        if metric_id == "collections_efficiency":
            due = f.get("amount_due") or 0
            expected = f.get("expected_collections") or coll
            if day_cells[idx].get("is_projected") and due > 0:
                return round(min(99.5, max(0.0, 100.0 * expected / due)), 1)
            if rev > 0:
                daily_cei = min(99.5, max(55.0, base_cei + 12.0 * (coll / rev - 0.8)))
                return round(daily_cei, 1)
            return round(base_cei, 1)
        if metric_id == "open_invoices":
            return max(0, open_cnt + f["invoices_created"] - f["disputes_resolved"] * 0.5)
        if metric_id == "overdue_invoices":
            return f["overdue_invoices"]
        if metric_id == "overdue_amount":
            return f["overdue_amount"]
        if metric_id == "disputes_open":
            return f["disputes_open"]
        if metric_id == "dispute_resolution_time":
            if f["dispute_resolution_days"] > 0:
                return round(f["dispute_resolution_days"], 1)
            return round(18.0 + f["disputes_opened"] * 0.5, 1)
        if metric_id == "invoice_blocked":
            return f["invoice_blocked"]
        if metric_id == "revenue":
            return f["revenue"]
        if metric_id == "revenue_at_risk":
            return f["revenue_at_risk"]
        if metric_id == "invoice_accuracy":
            disputed = f["disputes_opened"]
            invoiced = max(1.0, f["invoices_created"])
            return round(max(88.0, 100.0 - (disputed / invoiced) * 100.0 * 0.4), 1)
        if metric_id == "customer_exposure":
            return open_amt
        if metric_id == "remittance_match_rate":
            cleared = f["payments_cleared_cnt"]
            if cleared > 0:
                return round(min(99.0, 85.0 + cleared * 0.8), 1)
            return round(91.0, 1)
        if metric_id == "unapplied_cash":
            return f["unapplied_cash"]
        if metric_id == "deductions":
            return f["disputes_opened"] * 2500
        if metric_id == "short_payments":
            return f["short_payments"]
        if metric_id == "credit_memo_pending":
            return max(0, f["disputes_open"] * 0.15)
        if metric_id == "promise_to_pay":
            return f["promise_to_pay"]
        if metric_id == "cash_app_cycle_time":
            if f["payments_cleared_cnt"] > 0 and f["collections"] > 0:
                return round(max(1.0, 5.0 - f["payments_cleared_cnt"] * 0.2), 1)
            return round(3.5, 1)
        return 0.0

    metrics_out: list[dict] = []
    for mdef in WEEKLY_METRIC_DEFS:
        cells: list[dict] = []
        for i, dc in enumerate(day_cells):
            value = _actual_metric(mdef["id"], i)
            prev_val = cells[-1]["value"] if cells else None
            delta_pct = None
            if prev_val not in (None, 0):
                delta_pct = round(100.0 * (value - prev_val) / abs(prev_val), 1)
            cells.append({
                "date": dc["date"],
                "day_name": dc["day_name"],
                "value": round(value, 2),
                "is_forecast": dc["is_forecast"],
                "delta_pct": delta_pct,
                "confidence_pct": _weekly_cell_confidence(
                    mdef["id"],
                    i,
                    day_cells,
                    today=today,
                    base_mape=base_mape,
                ),
            })
        metrics_out.append({**mdef, "cells": cells})
    return metrics_out


def get_weekly_metrics(
    week_start: str | None = None,
    company: str | None = None,
    dealer_id: str | None = None,
    *,
    weeks: int = 1,
) -> dict:
    """O2C calendar metrics: 3 prior weekdays, today, 3 future weekdays (no weekends)."""
    from datetime import date, datetime, timedelta

    del weeks
    today = date.today()
    if week_start:
        anchor = datetime.strptime(week_start[:10], "%Y-%m-%d").date()
    else:
        anchor = today
    window_dates = _weekly_metrics_window(anchor)
    ws = window_dates[0].strftime("%Y-%m-%d")
    we = window_dates[-1].strftime("%Y-%m-%d")
    cache_key = f"weekly:v6:{ws}:{we}:{company or 'all'}:{dealer_id or 'all'}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    cf = _company_filter(company, "inv.")
    df = _dealer_filter(dealer_id, "inv.")
    cf_ar = _company_filter(company, "ar.")
    df_ar = _dealer_filter(dealer_id, "ar.")
    cf_pay = _company_filter(company, "p.")
    df_pay = _dealer_filter(dealer_id, "p.")

    # Fabric T-SQL versions of the Snowflake CTE fragments (see weekly_forecast.py).
    risk_join = f"""
        risk as (
            select s.metric_date as d, sum(r.exposure_amount * r.risk_score) as v
            from spine s
            join {BM}.proactive_dispute_risk_vw r
              on r.predicted_dispute_date <= dateadd(day, 7, s.metric_date)
            group by s.metric_date
        ),"""
    blocked_join = f"""
        blocked as (
            select s.metric_date as d, count(*) as v
            from spine s
            join {BM}.billing_eligibility_vw be
              on be.billing_eligibility_status in ('BLOCKED', 'PENDING_GOODS_ISSUE', 'PENDING_MILESTONE')
            group by s.metric_date
        ),"""

    try:
        run_query(f"select 1 from {BM}.proactive_dispute_risk_vw limit 1")
    except Exception:
        risk_join = """
        risk as (select cast(null as date) as d, cast(0 as decimal(38, 2)) as v where 1 = 0),"""
    try:
        run_query(f"select billing_trigger_type from {BM}.billing_eligibility_vw limit 1")
    except Exception:
        blocked_join = """
        blocked as (select cast(null as date) as d, cast(0 as decimal(38, 2)) as v where 1 = 0),"""

    daily_sql = build_daily_facts_sql(
        ws,
        we,
        cf=cf,
        df=df,
        cf_ar=cf_ar,
        df_ar=df_ar,
        cf_pay=cf_pay,
        df_pay=df_pay,
        blocked_join=blocked_join,
        risk_join=risk_join,
    )
    daily = run_tsql(daily_sql)
    daily_by_date = {
        str(row.get("metric_date") or row.get("METRIC_DATE"))[:10]: row
        for row in daily
    }

    kpi_row = run_query(f"""
        select avg(dso) as dso, avg(cei) as cei, avg(mape) as mape
        from (
            select dso, null as cei, null as mape from {BM}.dso_vw where dso is not null {_company_filter(company)}
            union all
            select null, cei, null from {BM}.cei_vw where cei is not null {_company_filter(company)}
            union all
            select null, null, mape from {BM}.forecast_accuracy_vw where mape is not null {_company_filter(company)}
        )
    """)
    base = kpi_row[0] if kpi_row else {}
    base_dso = float(base.get("DSO") or base.get("dso") or 42)
    base_cei = float(base.get("CEI") or base.get("cei") or 88)
    base_mape = float(base.get("MAPE") or base.get("mape") or 12)

    ar_snap = run_query(f"""
        select count(*) as open_cnt, sum(open_amount_usd) as open_amt
        from {BM}.ar_open_item_vw ar
        where 1=1 {cf_ar}{df_ar}
    """)
    snap = ar_snap[0] if ar_snap else {}
    open_cnt = float(snap.get("OPEN_CNT") or snap.get("open_cnt") or 0)
    open_amt = float(snap.get("OPEN_AMT") or snap.get("open_amt") or 0)

    day_cells: list[dict] = []
    for i, md in enumerate(window_dates):
        row = daily_by_date.get(md.strftime("%Y-%m-%d"), {})
        is_projected = md > today
        day_cells.append({
            "date": str(md),
            "day_name": str(row.get("day_name") or row.get("DAY_NAME") or md.strftime("%a")),
            "day_index": i,
            "is_today": md == today,
            "is_yesterday": md == _adjacent_weekday(today, -1),
            "is_tomorrow": md == _adjacent_weekday(today, 1),
            "is_forecast": is_projected,
            "is_projected": is_projected,
            "is_weekend": not _is_weekday(md),
            "facts": {
                "revenue": float(row.get("revenue") or row.get("REVENUE") or 0),
                "collections": float(row.get("collections") or row.get("COLLECTIONS") or 0),
                "amount_due": float(row.get("amount_due") or row.get("AMOUNT_DUE") or 0),
                "expected_collections": float(row.get("expected_collections") or row.get("EXPECTED_COLLECTIONS") or 0),
                "early_payer_expected": float(row.get("early_payer_expected") or row.get("EARLY_PAYER_EXPECTED") or 0),
                "late_payer_expected": float(row.get("late_payer_expected") or row.get("LATE_PAYER_EXPECTED") or 0),
                "invoices_created": float(row.get("invoices_created") or row.get("INVOICES_CREATED") or 0),
                "disputes_opened": float(row.get("disputes_opened") or row.get("DISPUTES_OPENED") or 0),
                "disputes_open": float(row.get("disputes_open") or row.get("DISPUTES_OPEN") or 0),
                "overdue_amount": float(row.get("overdue_amount") or row.get("OVERDUE_AMOUNT") or 0),
                "overdue_invoices": float(row.get("overdue_invoices") or row.get("OVERDUE_INVOICES") or 0),
                "unapplied_cash": float(row.get("unapplied_cash") or row.get("UNAPPLIED_CASH") or 0),
                "short_payments": float(row.get("short_payments") or row.get("SHORT_PAYMENTS") or 0),
                "promise_to_pay": float(row.get("promise_to_pay") or row.get("PROMISE_TO_PAY") or 0),
                "invoice_blocked": float(row.get("invoice_blocked") or row.get("INVOICE_BLOCKED") or 0),
                "revenue_at_risk": float(row.get("revenue_at_risk") or row.get("REVENUE_AT_RISK") or 0),
                "payments_cleared": float(row.get("payments_cleared") or row.get("PAYMENTS_CLEARED") or 0),
                "payments_cleared_cnt": float(row.get("payments_cleared_cnt") or row.get("PAYMENTS_CLEARED_CNT") or 0),
                "dispute_resolution_days": float(row.get("dispute_resolution_days") or row.get("DISPUTE_RESOLUTION_DAYS") or 0),
                "disputes_resolved": float(row.get("disputes_resolved") or row.get("DISPUTES_RESOLVED") or 0),
                "expected_confidence": float(row.get("expected_confidence") or row.get("EXPECTED_CONFIDENCE") or 52),
                "amount_due_confidence": float(row.get("amount_due_confidence") or row.get("AMOUNT_DUE_CONFIDENCE") or 90),
            },
        })

    metrics_out = _build_weekly_metrics_for_days(
        day_cells,
        today=today,
        base_dso=base_dso,
        base_cei=base_cei,
        base_mape=base_mape,
        open_cnt=open_cnt,
        open_amt=open_amt,
    )
    week_blocks = [{
        "week_start": ws,
        "week_end": we,
        "week_index": 0,
        "is_current_week": anchor == today or (anchor <= today <= window_dates[-1]),
        "days": day_cells,
        "metrics": metrics_out,
    }]

    highlights = []
    date_to_idx = {dc["date"]: i for i, dc in enumerate(day_cells)}
    for label, metric_id, target in (
        ("yesterday", "revenue", _adjacent_weekday(today, -1)),
        ("today", "dso", today if _is_weekday(today) else None),
        ("tomorrow", "expected_collections", _adjacent_weekday(today, 1)),
    ):
        if target is None:
            continue
        target_str = str(target)
        col_idx = date_to_idx.get(target_str)
        if col_idx is None:
            continue
        m = next((x for x in metrics_out if x["id"] == metric_id), None)
        if not m:
            continue
        cell = m["cells"][col_idx]
        highlights.append({
            "label": label,
            "metric_id": metric_id,
            "metric_label": m["label"],
            "date": cell["date"],
            "value": cell["value"],
            "format": m["format"],
            "delta_pct": cell["delta_pct"],
            "confidence_pct": cell.get("confidence_pct"),
            "vs_label": "vs prior day" if col_idx > 0 else None,
        })

    primary = week_blocks[0]
    result = {
        "week_start": ws,
        "week_end": we,
        "week_count": 1,
        "weeks": week_blocks,
        "anchor_date": str(anchor),
        "window_days": len(day_cells),
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "data_as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "today": str(today),
        "company": company,
        "dealer_id": dealer_id,
        "days": primary["days"],
        "metrics": primary["metrics"],
        "highlights": highlights,
        "projection_method": "due_date_customer_behavior",
    }
    _cache_set(cache_key, result)
    return result


def _rel_num(row: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _rel_str(row: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


_RELIABILITY_ACTIONS: dict[str, str] = {
    "open_disputes": "Review open disputes in Agent Workbench",
    "broken_ptp": "Follow up on broken promises to pay in Collections",
    "late_payments": "Review payment history and dunning strategy",
    "days_past_due": "Prioritize collections outreach for overdue invoices",
    "past_due_exposure": "Assess credit exposure and collection priority",
    "billing_friction": "Resolve billing blocks in Billing Exceptions agent",
}


def _compute_reliability(row: dict) -> dict:
    open_disputes = int(_rel_num(row, "open_disputes", "OPEN_DISPUTES"))
    broken_ptp = int(_rel_num(row, "broken_ptp_count", "BROKEN_PTP_COUNT"))
    late_payments = int(_rel_num(row, "late_payment_count", "LATE_PAYMENT_COUNT"))
    max_dpd = int(_rel_num(row, "max_days_past_due", "MAX_DAYS_PAST_DUE"))
    past_due_usd = _rel_num(row, "past_due_usd", "PAST_DUE_USD")
    credit_blocks = int(_rel_num(row, "credit_block_orders", "CREDIT_BLOCK_ORDERS"))
    order_holds = int(_rel_num(row, "order_hold_count", "ORDER_HOLD_COUNT"))

    penalties: list[dict] = []

    p_disputes = min(open_disputes * 8, 24)
    if p_disputes:
        penalties.append({
            "factor": "open_disputes",
            "label": f"{open_disputes} open dispute{'s' if open_disputes != 1 else ''}",
            "penalty": p_disputes,
        })

    p_ptp = min(broken_ptp * 6, 18)
    if p_ptp:
        penalties.append({
            "factor": "broken_ptp",
            "label": f"{broken_ptp} broken promise{'s' if broken_ptp != 1 else ''} to pay",
            "penalty": p_ptp,
        })

    p_late = min(late_payments * 2, 20)
    if p_late:
        penalties.append({
            "factor": "late_payments",
            "label": f"{late_payments} late payment{'s' if late_payments != 1 else ''}",
            "penalty": p_late,
        })

    if max_dpd <= 0:
        p_dpd = 0
    elif max_dpd <= 30:
        p_dpd = 5
    elif max_dpd <= 60:
        p_dpd = 12
    else:
        p_dpd = 20
    if p_dpd:
        penalties.append({
            "factor": "days_past_due",
            "label": f"Max {max_dpd} days past due",
            "penalty": p_dpd,
        })

    p_exposure = 0
    if past_due_usd > 50_000:
        p_exposure = 10
    elif past_due_usd > 10_000:
        p_exposure = 5
    if p_exposure:
        penalties.append({
            "factor": "past_due_exposure",
            "label": f"${past_due_usd:,.0f} past due",
            "penalty": p_exposure,
        })

    p_billing = min(credit_blocks * 3 + order_holds * 2, 10)
    if p_billing:
        penalties.append({
            "factor": "billing_friction",
            "label": f"{credit_blocks} credit block(s), {order_holds} order hold(s)",
            "penalty": p_billing,
        })

    total_penalty = sum(p["penalty"] for p in penalties)
    score = max(0, min(100, round(100 - total_penalty)))

    if score >= 80:
        band = "reliable"
    elif score >= 60:
        band = "watch"
    elif score >= 40:
        band = "at_risk"
    else:
        band = "critical"

    ranked = sorted(penalties, key=lambda x: x["penalty"], reverse=True)
    insights = [p["label"] for p in ranked[:3]]
    actions = [_RELIABILITY_ACTIONS[p["factor"]] for p in ranked[:3] if p["factor"] in _RELIABILITY_ACTIONS]

    return {
        "reliability_score": score,
        "reliability_band": band,
        "total_penalty": total_penalty,
        "penalties": penalties,
        "insights": insights,
        "suggested_actions": actions,
    }


def _enrich_reliability_row(row: dict) -> dict:
    rel = _compute_reliability(row)
    return {
        "customer_id": _rel_str(row, "customer_id", "CUSTOMER_ID"),
        "customer_name": _rel_str(row, "customer_name", "CUSTOMER_NAME"),
        "company_code": _rel_str(row, "company_code", "COMPANY_CODE"),
        "customer_segment": _rel_str(row, "customer_segment", "CUSTOMER_SEGMENT"),
        "risk_class": _rel_str(row, "risk_class", "RISK_CLASS"),
        "customer_risk_band": _rel_str(row, "customer_risk_band", "CUSTOMER_RISK_BAND"),
        "total_disputes": int(_rel_num(row, "total_disputes", "TOTAL_DISPUTES")),
        "open_disputes": int(_rel_num(row, "open_disputes", "OPEN_DISPUTES")),
        "avg_resolution_days": _rel_num(row, "avg_resolution_days", "AVG_RESOLUTION_DAYS") or None,
        "late_payment_count": int(_rel_num(row, "late_payment_count", "LATE_PAYMENT_COUNT")),
        "avg_days_beyond_terms": _rel_num(row, "avg_days_beyond_terms", "AVG_DAYS_BEYOND_TERMS") or None,
        "past_due_usd": round(_rel_num(row, "past_due_usd", "PAST_DUE_USD"), 2),
        "max_days_past_due": int(_rel_num(row, "max_days_past_due", "MAX_DAYS_PAST_DUE")),
        "past_due_items": int(_rel_num(row, "past_due_items", "PAST_DUE_ITEMS")),
        "broken_ptp_count": int(_rel_num(row, "broken_ptp_count", "BROKEN_PTP_COUNT")),
        "ptp_count": int(_rel_num(row, "ptp_count", "PTP_COUNT")),
        "credit_block_orders": int(_rel_num(row, "credit_block_orders", "CREDIT_BLOCK_ORDERS")),
        "order_hold_count": int(_rel_num(row, "order_hold_count", "ORDER_HOLD_COUNT")),
        "fulfillment_block_orders": int(_rel_num(row, "fulfillment_block_orders", "FULFILLMENT_BLOCK_ORDERS")),
        **rel,
    }


def _fetch_customer_reliability_raw(
    company: str | None = None,
    customer_id: str | None = None,
    segment: str | None = None,
    risk_band: str | None = None,
) -> list[dict]:
    clauses = ["1=1"]
    if company:
        clauses.append(f"COMPANY_CODE = '{company.replace(chr(39), chr(39)+chr(39))}'")
    if customer_id:
        clauses.append(f"CUSTOMER_ID = '{customer_id.replace(chr(39), chr(39)+chr(39))}'")
    if segment:
        clauses.append(f"CUSTOMER_SEGMENT = '{segment.replace(chr(39), chr(39)+chr(39))}'")
    if risk_band:
        clauses.append(f"CUSTOMER_RISK_BAND = '{risk_band.replace(chr(39), chr(39)+chr(39))}'")

    sql = f"""
    SELECT
        CUSTOMER_ID,
        CUSTOMER_NAME,
        COMPANY_CODE,
        CUSTOMER_SEGMENT,
        RISK_CLASS,
        CUSTOMER_RISK_BAND,
        TOTAL_DISPUTES,
        OPEN_DISPUTES,
        AVG_RESOLUTION_DAYS,
        LATE_PAYMENT_COUNT,
        AVG_DAYS_BEYOND_TERMS,
        PAST_DUE_USD,
        MAX_DAYS_PAST_DUE,
        PAST_DUE_ITEMS,
        BROKEN_PTP_COUNT,
        PTP_COUNT,
        CREDIT_BLOCK_ORDERS,
        ORDER_HOLD_COUNT,
        FULFILLMENT_BLOCK_ORDERS
    FROM {BM}.customer_o2c_history_vw
    WHERE {' AND '.join(clauses)}
    """
    return run_query(sql)


def _sort_reliability_rows(rows: list[dict], sort: str) -> list[dict]:
    key_map = {
        "score": lambda r: r.get("reliability_score", 0),
        "score_desc": lambda r: -r.get("reliability_score", 0),
        "past_due": lambda r: -r.get("past_due_usd", 0),
        "disputes": lambda r: -r.get("open_disputes", 0),
        "customer": lambda r: (r.get("customer_name") or "").lower(),
    }
    fn = key_map.get(sort or "score", key_map["score"])
    return sorted(rows, key=fn)


def get_customer_reliability_list(
    company: str | None = None,
    customer_id: str | None = None,
    segment: str | None = None,
    risk_band: str | None = None,
    reliability_band: str | None = None,
    sort: str | None = "score",
    limit: int = 200,
    offset: int = 0,
) -> dict:
    cache_key = (
        f"customer_reliability_list:v1:{company}:{customer_id}:{segment}:"
        f"{risk_band}:{reliability_band}:{sort}:{limit}:{offset}"
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    raw = _fetch_customer_reliability_raw(company, customer_id, segment, risk_band)
    rows = [_enrich_reliability_row(r) for r in raw]

    if reliability_band:
        rb = reliability_band.lower().replace(" ", "_")
        rows = [r for r in rows if r.get("reliability_band") == rb]

    rows = _sort_reliability_rows(rows, sort or "score")
    total = len(rows)
    page = rows[offset: offset + limit]

    result = {"total": total, "offset": offset, "limit": limit, "rows": page}
    _cache_set(cache_key, result)
    return result


def get_customer_reliability_summary(
    company: str | None = None,
    segment: str | None = None,
) -> dict:
    cache_key = f"customer_reliability_summary:v1:{company}:{segment}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    raw = _fetch_customer_reliability_raw(company, segment=segment)
    rows = [_enrich_reliability_row(r) for r in raw]

    band_counts = {"reliable": 0, "watch": 0, "at_risk": 0, "critical": 0}
    at_risk_past_due = 0.0
    total_open_disputes = 0
    scores: list[float] = []

    for r in rows:
        band = r.get("reliability_band", "reliable")
        if band in band_counts:
            band_counts[band] += 1
        scores.append(float(r.get("reliability_score") or 0))
        total_open_disputes += int(r.get("open_disputes") or 0)
        if (r.get("reliability_score") or 100) < 60:
            at_risk_past_due += float(r.get("past_due_usd") or 0)

    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    at_risk_count = band_counts["at_risk"] + band_counts["critical"]

    result = {
        "total_customers": len(rows),
        "avg_reliability_score": avg_score,
        "customers_at_risk": at_risk_count,
        "at_risk_past_due_usd": round(at_risk_past_due, 2),
        "total_open_disputes": total_open_disputes,
        "band_counts": band_counts,
        "band_distribution": [
            {"band": "reliable", "label": "Reliable", "count": band_counts["reliable"]},
            {"band": "watch", "label": "Watch", "count": band_counts["watch"]},
            {"band": "at_risk", "label": "At risk", "count": band_counts["at_risk"]},
            {"band": "critical", "label": "Critical", "count": band_counts["critical"]},
        ],
    }
    _cache_set(cache_key, result)
    return result


def get_customer_reliability_detail(
    customer_id: str,
    company_code: str | None = None,
) -> dict | None:
    cache_key = f"customer_reliability_detail:v1:{customer_id}:{company_code}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    raw = _fetch_customer_reliability_raw(company=company_code, customer_id=customer_id)
    if not raw:
        return None

    if company_code:
        match = [r for r in raw if _rel_str(r, "company_code", "COMPANY_CODE") == company_code]
        row = match[0] if match else raw[0]
    else:
        row = raw[0]

    enriched = _enrich_reliability_row(row)
    result = {
        **enriched,
        "company_matches": len(raw),
    }
    _cache_set(cache_key, result)
    return result

