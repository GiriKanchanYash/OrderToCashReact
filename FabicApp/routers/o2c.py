from fastapi import APIRouter, Depends, HTTPException, Query

from FabicApp.auth import require_roles
from FabicApp.services import o2c_service as svc

router = APIRouter(prefix="/api/o2c", tags=["O2C"])


@router.get("/filters")
def filters():
    return svc.get_filter_options()


@router.get("/summary")
def summary(
    company: str | None = Query(None),
    dealer_id: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    period_preset: str | None = Query(None),
):
    return svc.get_summary(company, dealer_id, date_from, date_to, period_preset)


@router.get("/orders")
def orders(
    company: str | None = Query(None),
    dealer_id: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    order_status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return svc.get_orders(company, dealer_id, date_from, date_to, order_status, limit, offset)


@router.get("/orders/{order_id}/timeline")
def order_timeline(order_id: str):
    return svc.get_order_timeline(order_id)


@router.get("/orders/{order_id}/invoices")
def order_invoices(order_id: str):
    return svc.get_order_invoices(order_id)


@router.get("/data-version")
def data_version():
    return svc.get_data_version()


@router.get("/invoices")
def invoices(
    company: str | None = Query(None),
    dealer_id: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return svc.get_invoices(company, dealer_id, date_from, date_to, limit, offset)


@router.get("/kpis/{metric}")
def kpi_series(
    metric: str,
    company: str | None = Query(None),
    dealer_id: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    return svc.get_kpi_series(metric, company, dealer_id, date_from, date_to)


@router.get("/weekly-metrics")
def weekly_metrics(
    week_start: str | None = Query(None),
    company: str | None = Query(None),
    dealer_id: str | None = Query(None),
    weeks: int = Query(1, ge=1, le=6),
):
    return svc.get_weekly_metrics(week_start, company, dealer_id, weeks=weeks)


@router.get("/customer-reliability/summary")
def customer_reliability_summary(
    company: str | None = Query(None),
    segment: str | None = Query(None),
    _=Depends(require_roles("admin", "finance", "collections", "exec")),
):
    return svc.get_customer_reliability_summary(company, segment)


@router.get("/customer-reliability")
def customer_reliability_list(
    company: str | None = Query(None),
    customer_id: str | None = Query(None),
    segment: str | None = Query(None),
    risk_band: str | None = Query(None),
    reliability_band: str | None = Query(None),
    sort: str | None = Query("score"),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _=Depends(require_roles("admin", "finance", "collections", "exec")),
):
    return svc.get_customer_reliability_list(
        company, customer_id, segment, risk_band, reliability_band, sort, limit, offset,
    )


@router.get("/customer-reliability/{customer_id}")
def customer_reliability_detail(
    customer_id: str,
    company_code: str | None = Query(None),
    _=Depends(require_roles("admin", "finance", "collections", "exec")),
):
    detail = svc.get_customer_reliability_detail(customer_id, company_code)
    if detail is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return detail


@router.post("/refresh")
def refresh(_=Depends(require_roles("admin"))):
    return svc.refresh_data()
