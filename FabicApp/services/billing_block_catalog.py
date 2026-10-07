"""SAP-style billing / order block scenarios for the O2C Billing Agent."""
from __future__ import annotations

BLOCK_SCENARIOS: list[dict[str, str]] = [
    {
        "id": "CREDIT_BLOCK",
        "label": "Credit block",
        "description": "Sales order is blocked when customer exceeds credit limit or has overdue receivables.",
    },
    {
        "id": "DELIVERY_BLOCK",
        "label": "Delivery block",
        "description": "Order is blocked due to incomplete data, pending approval, or customer-specific delivery restrictions.",
    },
    {
        "id": "BILLING_BLOCK",
        "label": "Billing block",
        "description": "Sales order or delivery is blocked from invoicing due to pricing, dispute, or commercial approval pending.",
    },
    {
        "id": "PRICING_ERROR_BLOCK",
        "label": "Pricing error block",
        "description": "Order or invoice is blocked when mandatory pricing condition, tax, or discount is missing.",
    },
    {
        "id": "INCOMPLETE_ORDER_BLOCK",
        "label": "Incomplete order block",
        "description": "Sales order is blocked if required fields like customer PO, material, payment terms, or shipping data are missing.",
    },
    {
        "id": "AVAILABILITY_BLOCK",
        "label": "Availability block",
        "description": "Order cannot proceed fully when stock is unavailable or ATP confirmation fails.",
    },
    {
        "id": "CUSTOMER_MASTER_BLOCK",
        "label": "Customer master block",
        "description": "Order or invoice is blocked if the customer is flagged for central, sales, delivery, or billing block.",
    },
    {
        "id": "MATERIAL_MASTER_BLOCK",
        "label": "Material master block",
        "description": "Order is blocked if the material is blocked for sales, discontinued, or not extended to the sales area.",
    },
    {
        "id": "DELIVERY_QUANTITY_BLOCK",
        "label": "Delivery quantity block",
        "description": "Invoice is blocked when delivered quantity does not match ordered or expected quantity.",
    },
    {
        "id": "INVOICE_LIST_BLOCK",
        "label": "Invoice list/billing due block",
        "description": "Invoice is delayed when billing date, billing relevance, or copy-control settings prevent billing.",
    },
    {
        "id": "TAX_DETERMINATION_BLOCK",
        "label": "Tax determination block",
        "description": "Invoice is blocked when tax code, tax jurisdiction, or tax classification is missing.",
    },
    {
        "id": "PAYMENT_TERMS_BLOCK",
        "label": "Payment terms block",
        "description": "Order or invoice may be blocked if payment terms are invalid, missing, or require approval.",
    },
    {
        "id": "LEGAL_EXPORT_BLOCK",
        "label": "Legal/export block",
        "description": "Order is blocked due to embargo, export control, sanctioned party, or compliance checks.",
    },
    {
        "id": "RETURNS_DISPUTE_BLOCK",
        "label": "Returns/dispute block",
        "description": "Invoice or credit memo is blocked when customer raises a dispute, return, or claim requiring review.",
    },
    {
        "id": "ACCOUNTING_POSTING_BLOCK",
        "label": "Accounting posting block",
        "description": "Billing document is blocked from posting to FI due to account determination, cost center, tax, or reconciliation errors.",
    },
]

_BLOCK_BY_ID = {b["id"]: b for b in BLOCK_SCENARIOS}


def classify_billing_block(
    *,
    status: str = "",
    reason: str = "",
    credit_status: str = "",
    delivery_status: str = "",
    order_status: str = "",
) -> dict[str, str]:
    """Map ERP billing eligibility signals to a block scenario."""
    st = (status or "").upper()
    rs = (reason or "").lower()
    cr = (credit_status or "").upper()
    ds = (delivery_status or "").upper()
    os_ = (order_status or "").upper()

    if cr == "BLOCKED" or "credit" in rs and ("limit" in rs or "block" in rs or "overdue" in rs):
        return _BLOCK_BY_ID["CREDIT_BLOCK"]
    if "dispute" in rs or "return" in rs or "claim" in rs:
        return _BLOCK_BY_ID["RETURNS_DISPUTE_BLOCK"]
    if "tax" in rs:
        return _BLOCK_BY_ID["TAX_DETERMINATION_BLOCK"]
    if "pricing" in rs or "discount" in rs or "condition" in rs:
        return _BLOCK_BY_ID["PRICING_ERROR_BLOCK"]
    if "payment term" in rs:
        return _BLOCK_BY_ID["PAYMENT_TERMS_BLOCK"]
    if "export" in rs or "embargo" in rs or "sanction" in rs or "compliance" in rs:
        return _BLOCK_BY_ID["LEGAL_EXPORT_BLOCK"]
    if "account" in rs and ("post" in rs or "fi" in rs or "reconcil" in rs):
        return _BLOCK_BY_ID["ACCOUNTING_POSTING_BLOCK"]
    if "quantity" in rs or "short ship" in rs or "over ship" in rs:
        return _BLOCK_BY_ID["DELIVERY_QUANTITY_BLOCK"]
    if "material" in rs or "discontinued" in rs:
        return _BLOCK_BY_ID["MATERIAL_MASTER_BLOCK"]
    if "customer" in rs and "block" in rs:
        return _BLOCK_BY_ID["CUSTOMER_MASTER_BLOCK"]
    if "stock" in rs or "atp" in rs or "availability" in rs:
        return _BLOCK_BY_ID["AVAILABILITY_BLOCK"]
    if "po" in rs or "incomplete" in rs or "missing" in rs and "field" in rs:
        return _BLOCK_BY_ID["INCOMPLETE_ORDER_BLOCK"]
    if st in ("PENDING_BILLING_CYCLE",) or "billing date" in rs or "billing relevance" in rs:
        return _BLOCK_BY_ID["INVOICE_LIST_BLOCK"]
    if os_ == "BLOCKED":
        return _BLOCK_BY_ID["CUSTOMER_MASTER_BLOCK"]
    if st in ("NO_DELIVERY", "PENDING_DELIVERY") or "delivery" in rs and "restrict" in rs:
        return _BLOCK_BY_ID["DELIVERY_BLOCK"]
    if st in ("PENDING_GOODS_ISSUE",) or "goods issue" in rs:
        return _BLOCK_BY_ID["AVAILABILITY_BLOCK"]
    if st == "BLOCKED":
        return _BLOCK_BY_ID["BILLING_BLOCK"]
    if st in ("PENDING_MILESTONE", "PENDING_SERVICE_CLOSE"):
        return _BLOCK_BY_ID["BILLING_BLOCK"]
    return _BLOCK_BY_ID["BILLING_BLOCK"]


def block_scenario_lines(
    *,
    status: str = "",
    reason: str = "",
    credit_status: str = "",
    delivery_status: str = "",
    order_status: str = "",
) -> str:
    block = classify_billing_block(
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=order_status,
    )
    return f"- Block type: {block['label']} — {block['description']}"


def _clean_erp_status(val: str) -> str:
    v = (val or "").strip()
    if not v or v.upper() in ("N/A", "NA", "NONE", "UNKNOWN", "NULL", "—", "-"):
        return ""
    return v.replace("_", " ")


def billing_problem_summary(
    *,
    status: str = "",
    reason: str = "",
    credit_status: str = "",
    delivery_status: str = "",
    order_status: str = "",
) -> str:
    """Single definitive problem statement — omit irrelevant N/A delivery/credit fields."""
    block = classify_billing_block(
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=order_status,
    )
    rs = (reason or "").strip()
    headline = rs if rs else block["description"]
    st = (status or "").upper()
    cr = (credit_status or "").upper()
    ds = _clean_erp_status(delivery_status)

    if cr == "BLOCKED" and st == "BLOCKED":
        return f"{block['label']}: {headline}"
    if cr == "BLOCKED":
        return f"Credit block: {headline}"
    if st in ("NO_DELIVERY", "PENDING_DELIVERY", "PENDING_GOODS_ISSUE"):
        if ds:
            return f"Delivery blocker ({st.replace('_', ' ').lower()}): {headline} — {ds}"
        return f"Delivery blocker: {headline}"
    if ds and st not in ("BLOCKED",):
        return f"{block['label']}: {headline} — delivery {ds.lower()}"
    return f"{block['label']}: {headline}"


def billing_order_context_lines(
    *,
    sales_order_id: str,
    customer: str,
    order_value: float,
    status: str = "",
    reason: str = "",
    credit_status: str = "",
    delivery_status: str = "",
    order_status: str = "",
) -> str:
    """ORDER CONTEXT filler — one primary problem, no contradictory N/A fields."""
    block = classify_billing_block(
        status=status,
        reason=reason,
        credit_status=credit_status,
        delivery_status=delivery_status,
        order_status=order_status,
    )
    st = (status or "").upper()
    cr = (credit_status or "").upper()
    ds = _clean_erp_status(delivery_status)
    os_ = _clean_erp_status(order_status)
    rs = (reason or "").strip() or block["description"]

    lines = [
        f"- Sales order: {sales_order_id}",
        f"- Problem: {billing_problem_summary(status=status, reason=reason, credit_status=credit_status, delivery_status=delivery_status, order_status=order_status)}",
        f"- Customer: {customer}",
        f"- Order value: ${order_value:,.0f}",
    ]
    if cr == "BLOCKED":
        lines.append("- Credit check: blocked — release required before invoicing")
    if st in ("NO_DELIVERY", "PENDING_DELIVERY", "PENDING_GOODS_ISSUE") and ds:
        lines.append(f"- Delivery status: {ds}")
    elif ds and cr != "BLOCKED" and st not in ("BLOCKED",):
        lines.append(f"- Delivery status: {ds}")
    if os_ and os_.upper() not in ("N/A", "NA", "UNKNOWN"):
        lines.append(f"- Order status: {os_}")
    lines.append(f"- Billing eligibility: {st.replace('_', ' ').lower() or 'unknown'}")
    return "\n".join(lines)
