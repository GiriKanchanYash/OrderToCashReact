from fastapi import APIRouter, Depends, Query

from app.auth import require_roles
from app.services import ai_service as svc

router = APIRouter(prefix="/api/ai", tags=["AI"])


@router.get("/copilot/quick-analyses")
def quick_analyses():
    return svc.copilot_quick_analyses()


@router.post("/copilot/run-analysis")
def run_analysis(body: dict):
    return svc.run_quick_analysis(body.get("key", ""))


@router.post("/copilot/chat")
def copilot_chat(body: dict):
    return svc.copilot_chat(
        str(body.get("message", "")),
        body.get("short_memory") if body else None,
        body.get("long_memory") if body else None,
    )


@router.get("/copilot/saved-insights")
def saved_insights():
    return svc.load_saved_insights()


@router.post("/copilot/save-insight")
def save_insight(body: dict):
    svc.save_insight(
        title=str(body.get("title", "")),
        question=str(body.get("question", "")),
        sql_text=str(body.get("sql_text", "")),
    )
    return {"ok": True}


@router.post("/copilot/delete-insight")
def delete_insight(body: dict):
    svc.delete_insight(int(body.get("insight_id", 0)))
    return {"ok": True}


@router.get("/copilot/frequent-questions")
def frequent_questions():
    return svc.load_frequent_questions()


@router.get("/copilot/most-frequent")
def most_frequent():
    return svc.load_most_frequent_all()


@router.get("/billing-agent/queue")
def billing_queue(customer_id: str | None = None):
    return svc.billing_agent_queue(customer_id)


@router.post("/billing-agent/recommend")
def billing_recommend(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.billing_agent_recommendation(body)


@router.post("/billing-agent/credit-escalation")
def billing_credit_escalation(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.billing_agent_credit_escalation(body)


@router.post("/billing-agent/fulfillment-escalation")
def billing_fulfillment_escalation(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.billing_agent_fulfillment_escalation(body)


@router.post("/billing-agent/mark-reviewed")
def billing_mark_reviewed(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.billing_agent_mark_reviewed(body)


@router.get("/collections-agent/queue")
def collections_queue(
    annual_interest_rate_pct: float = Query(8.0, ge=1.0, le=30.0),
    customer_id: str | None = None,
):
    return svc.collections_agent_queue(annual_interest_rate_pct, customer_id)


@router.post("/collections-agent/recommend")
def collections_recommend(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_recommendation(body)


@router.post("/collections-agent/email-draft")
def collections_email_draft(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_email_draft(body)


@router.get("/collections-agent/open-invoices")
def collections_open_invoices(
    customer_id: str,
    company_code: str | None = None,
    annual_interest_rate_pct: float = 8.0,
    _=Depends(require_roles("admin", "collections")),
):
    return svc.collections_agent_open_invoices(customer_id, company_code, annual_interest_rate_pct)


@router.post("/collections-agent/log-ptp")
def collections_log_ptp(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_log_ptp(body)


@router.post("/collections-agent/mark-contacted")
def collections_mark_contacted(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_mark_contacted(body)


@router.post("/collections-agent/mark-resolved")
def collections_mark_resolved(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_mark_resolved(body)


@router.post("/collections-agent/process-inbound")
def collections_process_inbound(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_process_inbound(body)


@router.post("/collections-agent/auto-flow")
def collections_auto_flow(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.collections_agent_auto_flow(body)


@router.get("/agent-workbench/snapshot")
def agent_workbench_snapshot(annual_interest_rate_pct: float = Query(8.0, ge=1.0, le=30.0)):
    return svc.agent_workbench_snapshot(annual_interest_rate_pct)


@router.post("/agent-workbench/autonomous-cycle")
def agent_autonomous_cycle(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.agent_workbench_autonomous_cycle(
        float(body.get("annual_interest_rate_pct") or 8.0),
        int(body.get("max_collections") or 3),
        int(body.get("max_cash") or 3),
        int(body.get("max_disputes") or 2),
    )


@router.get("/cash-application-agent/queue")
def cash_application_queue(customer_id: str | None = None):
    return svc.cash_application_agent_queue(customer_id)


@router.get("/cash-application-agent/match-candidates")
def cash_match_candidates(
    customer_id: str,
    payment_amount: float,
    company_code: str | None = None,
    _=Depends(require_roles("admin", "collections", "finance")),
):
    return svc.cash_application_match_candidates(customer_id, company_code, payment_amount)


@router.post("/cash-application-agent/recommend")
def cash_recommend(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.cash_application_recommendation(body)


@router.post("/cash-application-agent/remittance-request")
def cash_remittance_request(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.cash_application_remittance_request(body)


@router.post("/cash-application-agent/apply")
def cash_apply(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.cash_application_apply(body)


@router.get("/dispute-agent/queue")
def dispute_queue(customer_id: str | None = None):
    return svc.dispute_agent_queue(customer_id)


@router.post("/dispute-agent/recommend")
def dispute_recommend(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.dispute_agent_recommendation(body)


@router.post("/dispute-agent/request-info")
def dispute_request_info(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.dispute_agent_request_info(body)


@router.post("/dispute-agent/preventive-outreach")
def dispute_preventive_outreach(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.dispute_agent_preventive_outreach(body)


@router.post("/dispute-agent/update-status")
def dispute_update_status(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.dispute_agent_update_status(body)


@router.post("/dispute-agent/resolve")
def dispute_resolve(body: dict, _=Depends(require_roles("admin", "collections"))):
    return svc.dispute_agent_resolve(body)


@router.get("/dispute-agent/predicted-queue")
def dispute_predicted_queue():
    return svc.proactive_dispute_queue()


@router.get("/proactive-dispute-agent/queue")
def proactive_dispute_queue():
    return svc.proactive_dispute_queue()


@router.post("/proactive-dispute-agent/recommend")
def proactive_dispute_recommend(body: dict, _=Depends(require_roles("admin", "collections", "finance"))):
    return svc.dispute_agent_recommendation(body)
