"""
OrderLens (Order-to-Cash) data-quality rules — mirrors Procure2Pay validity / coverage / recon pattern.

Mart views: ORDER_LENS.BUSINESS_MART.*_VW
Hub entities: ORDER_LENS.RAW_VAULT.H_*
"""

ORDER_LENS = {
    "display_name": "OrderLens",
    "validity_views": [
        ("DSO_VW", [
            {
                "description": "DSO > 365 days",
                "impact": "DSO above one year is almost always a credit-sales or AR balance error — distorts Command Center DSO KPI and working-capital narrative.",
                "column": "DSO",
                "operator": ">",
                "threshold": 365,
                "severity": "WARNING",
            },
            {
                "description": "Negative DSO",
                "impact": "Negative DSO breaks the DSO trend chart and month-over-month flux in Command Center.",
                "column": "DSO",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
            {
                "description": "DSO row with zero credit sales month",
                "impact": "Division-by-zero guard may hide months — gaps in DSO trend look like missing data.",
                "column": "CREDIT_SALES_MONTH",
                "operator": "<=",
                "threshold": 0,
                "severity": "WARNING",
            },
        ]),
        ("CEI_VW", [
            {
                "description": "CEI > 100%",
                "impact": "Collection Effectiveness above 100% is impossible — signals formula or AR roll-forward error.",
                "column": "CEI",
                "operator": ">",
                "threshold": 100,
                "severity": "ERROR",
            },
            {
                "description": "Negative CEI",
                "impact": "Negative CEI invalidates the CEI KPI card and collections effectiveness storyline.",
                "column": "CEI",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
        ]),
        ("PAST_DUE_VW", [
            {
                "description": "Past due % > 100%",
                "impact": "Past-due share above 100% is mathematically impossible — breaks Past Due AR KPI.",
                "column": "PAST_DUE_PCT",
                "operator": ">",
                "threshold": 100,
                "severity": "ERROR",
            },
            {
                "description": "Negative past due AR",
                "impact": "Negative past-due amounts corrupt aging buckets and collections queue prioritisation.",
                "column": "PAST_DUE_AR",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
            {
                "description": "Past due AR exceeds total AR",
                "impact": "Past-due greater than total AR indicates bucket aggregation or sign error.",
                "column": "PAST_DUE_AR",
                "operator": ">",
                "threshold_column": "TOTAL_AR",
                "severity": "ERROR",
            },
        ]),
        ("ADD_VW", [
            {
                "description": "Average days delinquent > 365",
                "impact": "ADD above a year inflates delinquency severity in Command Center and agent prompts.",
                "column": "AVG_DAYS_DELINQUENT",
                "operator": ">",
                "threshold": 365,
                "severity": "WARNING",
            },
            {
                "description": "Negative ADD",
                "impact": "Negative ADD implies DSO snapshot below BPDSO — inconsistent delinquency math.",
                "column": "AVG_DAYS_DELINQUENT",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
        ]),
        ("FORECAST_ACCURACY_VW", [
            {
                "description": "Forecast MAPE > 1000% (likely bad baseline)",
                "impact": "Extreme MAPE skews Forecast MAPE KPI and weekly calendar confidence scoring.",
                "column": "MAPE",
                "operator": ">",
                "threshold": 1000,
                "severity": "WARNING",
            },
            {
                "description": "Negative MAPE",
                "impact": "Negative MAPE is invalid — forecast accuracy card will show nonsense.",
                "column": "MAPE",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
        ]),
        ("AR_PCT_REVENUE_VW", [
            {
                "description": "AR % of revenue > 100%",
                "impact": "AR exceeding 100% of trailing revenue is usually a revenue denominator error.",
                "column": "AR_PCT_OF_REVENUE",
                "operator": ">",
                "threshold": 100,
                "severity": "WARNING",
            },
            {
                "description": "Negative AR % of revenue",
                "impact": "Negative ratio breaks AR % Revenue KPI tile.",
                "column": "AR_PCT_OF_REVENUE",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
        ]),
        ("BILLING_ELIGIBILITY_VW", [
            {
                "description": "Negative order value on billing eligibility row",
                "impact": "Negative order values break Billing Exceptions queue ranking and blocked-value KPI.",
                "column": "TOTAL_ORDER_VALUE",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
            {
                "description": "BLOCKED status with empty eligibility reason",
                "impact": "Agent cannot state definitive problem — user sees vague holds without root cause.",
                "column": "BILLING_ELIGIBILITY_REASON",
                "operator": "IS_NULL",
                "threshold": None,
                "filter": "BILLING_ELIGIBILITY_STATUS = 'BLOCKED'",
                "severity": "WARNING",
            },
        ]),
        ("AR_OPEN_ITEM_VW", [
            {
                "description": "Negative open amount on AR item",
                "impact": "Negative open balances corrupt past-due totals, collections queue, and Copilot rankings.",
                "column": "OPEN_AMOUNT_USD",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
            {
                "description": "Days past due > 999",
                "impact": "Extreme aging days distort Act Now / Plan / Monitor tiering in Collections agent.",
                "column": "DAYS_PAST_DUE",
                "operator": ">",
                "threshold": 999,
                "severity": "WARNING",
            },
        ]),
        ("PAYMENT_VW", [
            {
                "description": "Clearing date before payment date",
                "impact": "Payment cannot clear before receipt — timestamp or join error in cash application.",
                "column": "CLEARING_DATE",
                "operator": "<",
                "threshold_column": "PAYMENT_DATE",
                "severity": "ERROR",
            },
            {
                "description": "Negative payment amount",
                "impact": "Negative payments break Cash Application queue amounts and unapplied cash totals.",
                "column": "PAYMENT_AMOUNT_USD",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
        ]),
        ("DISPUTE_VW", [
            {
                "description": "Negative disputed amount",
                "impact": "Invalid dispute amounts break Disputes agent queue and cash-at-risk rollups.",
                "column": "DISPUTED_AMOUNT",
                "operator": "<",
                "threshold": 0,
                "severity": "ERROR",
            },
            {
                "description": "Resolved date before opened date",
                "impact": "Dispute lifecycle is backwards — timeline and aging metrics are unreliable.",
                "column": "RESOLVED_DATE",
                "operator": "<",
                "threshold_column": "OPENED_DATE",
                "severity": "ERROR",
            },
        ]),
    ],
    "coverage_views": [
        (
            "DSO_VW",
            "DSO",
            "Companies missing from DSO monthly trend.",
            "COMPANY_CODE",
            "H_COMPANY_CODE",
        ),
        (
            "CEI_VW",
            "CEI",
            "Companies excluded from Collection Effectiveness Index.",
            "COMPANY_CODE",
            "H_COMPANY_CODE",
        ),
        (
            "PAST_DUE_VW",
            "Past Due AR",
            "Companies with open AR but no past-due snapshot row.",
            "COMPANY_CODE",
            "H_COMPANY_CODE",
        ),
        (
            "COLLECTIONS_WORKLIST_VW",
            "Collections Queue",
            "Customers with past-due AR not appearing in collections worklist.",
            "CUSTOMER_ID",
            "H_CUSTOMER",
        ),
        (
            "BILLING_ELIGIBILITY_VW",
            "Billing Exceptions",
            "Blocked or pending orders missing from billing eligibility mart.",
            "SALES_ORDER_ID",
            "H_SALES_ORDER",
        ),
        (
            "CASH_APP_WORKLIST_VW",
            "Cash Application",
            "Unapplied or partial payments not in cash application worklist.",
            "PAYMENT_ID",
            "H_PAYMENT",
        ),
        (
            "DISPUTE_WORKLIST_VW",
            "Disputes",
            "Open disputes missing from dispute worklist (agent queue empty).",
            "DISPUTE_ID",
            "H_DISPUTE",
        ),
        (
            "CUSTOMER_O2C_HISTORY_VW",
            "Customer Risk Context",
            "Active customers missing O2C history enrichment for agent recommendations.",
            "CUSTOMER_ID",
            "H_CUSTOMER",
        ),
    ],
    "recon_checks": [
        {
            "view": "AR_OPEN_ITEM_VW",
            "description": "Open AR customer count in mart vs H_CUSTOMER mismatch",
            "impact": "Transformation silently dropping customers — vault has customers but dashboard AR is incomplete.",
            "sql_template": (
                "ABS((SELECT COUNT(DISTINCT CUSTOMER_ID) FROM {mart_view} "
                "WHERE CUSTOMER_ID IS NOT NULL) - "
                "(SELECT COUNT(*) FROM ORDER_LENS.RAW_VAULT.{hub}))"
            ),
            "hub": "H_CUSTOMER",
            "severity": "WARNING",
        },
        {
            "view": "AR_INVOICE_VW",
            "description": "NULL CUSTOMER_ID on invoice rows",
            "impact": "Invoices without customer attribution break Copilot top-invoice and collections joins.",
            "sql_template": "COUNT(*) FROM {mart_view} WHERE CUSTOMER_ID IS NULL",
            "hub": None,
            "severity": "ERROR",
        },
        {
            "view": "AR_OPEN_ITEM_VW",
            "description": "Due date before invoice posting date",
            "impact": "Due date preceding posting is invalid — aging buckets and dunning tiers are wrong.",
            "sql_template": "COUNT(*) FROM {mart_view} WHERE DUE_DATE < POSTING_DATE",
            "hub": None,
            "severity": "ERROR",
        },
        {
            "view": "PAYMENT_VW",
            "description": "Unapplied payments with NULL customer_id",
            "impact": "Cash cannot be routed to collections or matching — Cash Application agent blind.",
            "sql_template": (
                "COUNT(*) FROM {mart_view} "
                "WHERE CUSTOMER_ID IS NULL AND CLEARING_DATE IS NULL"
            ),
            "hub": None,
            "severity": "ERROR",
        },
        {
            "view": "BILLING_ELIGIBILITY_VW",
            "description": "Sales orders in SALES_ORDER_VW missing from billing eligibility",
            "impact": "Orders exist but never evaluated for billing — exceptions under-reported.",
            "sql_template": (
                "COUNT(*) FROM ORDER_LENS.BUSINESS_MART.SALES_ORDER_VW so "
                "LEFT JOIN {mart_view} be ON be.SALES_ORDER_ID = so.SALES_ORDER_ID "
                "WHERE be.SALES_ORDER_ID IS NULL"
            ),
            "hub": None,
            "severity": "WARNING",
        },
        {
            "view": "DISPUTE_VW",
            "description": "Open dispute with NULL invoice_id",
            "impact": "Dispute cannot be tied to AR line — Disputes agent shows incomplete context.",
            "sql_template": (
                "COUNT(*) FROM {mart_view} "
                "WHERE DISPUTE_STATUS NOT IN ('RESOLVED','CLOSED','WRITTEN_OFF') "
                "AND INVOICE_ID IS NULL"
            ),
            "hub": None,
            "severity": "ERROR",
        },
        {
            "view": "COLLECTIONS_WORKLIST_VW",
            "description": "Past-due open items not represented on collections worklist",
            "impact": "Invoices are overdue in AR_OPEN_ITEM_VW but missing from Collections agent queue.",
            "sql_template": (
                "COUNT(*) FROM ORDER_LENS.BUSINESS_MART.AR_OPEN_ITEM_VW ar "
                "LEFT JOIN {mart_view} cw "
                "  ON cw.CUSTOMER_ID = ar.CUSTOMER_ID AND cw.COMPANY_CODE = ar.COMPANY_CODE "
                "WHERE ar.DAYS_PAST_DUE > 0 AND cw.CUSTOMER_ID IS NULL"
            ),
            "hub": None,
            "severity": "WARNING",
        },
        {
            "view": "DSO_VW",
            "description": "Months with NULL DSO across all companies",
            "impact": "Entire period with no DSO — bulk ETL or credit-sales feed failure for that month.",
            "sql_template": (
                "COUNT(*) FROM (SELECT PERIOD_MONTH FROM {mart_view} "
                "GROUP BY PERIOD_MONTH HAVING COUNT(DSO) = 0 OR SUM(IFF(DSO IS NULL,1,0)) = COUNT(*))"
            ),
            "hub": None,
            "severity": "WARNING",
        },
        {
            "view": "AR_OPEN_ITEM_VW",
            "description": "Past-due flag inconsistent with aging bucket",
            "impact": "CURRENT bucket rows with positive days past due break aging chart click-through.",
            "sql_template": (
                "COUNT(*) FROM {mart_view} "
                "WHERE (DAYS_PAST_DUE > 0 AND AGING_BUCKET = 'CURRENT') "
                "OR (DAYS_PAST_DUE <= 0 AND AGING_BUCKET <> 'CURRENT')"
            ),
            "hub": None,
            "severity": "WARNING",
        },
        {
            "view": "FORECAST_ACCURACY_VW",
            "description": "Forecast periods with zero actuals across all models",
            "impact": "MAPE/WAPE undefined — weekly calendar forecast confidence degrades.",
            "sql_template": (
                "COUNT(*) FROM (SELECT PERIOD_MONTH FROM {mart_view} "
                "GROUP BY PERIOD_MONTH HAVING SUM(ABS(TOTAL_ACTUAL)) = 0 "
                "OR SUM(ABS(TOTAL_ACTUAL)) IS NULL)"
            ),
            "hub": None,
            "severity": "WARNING",
        },
    ],
}

# Alias for registry keyed by product code (matches Procure2Pay style)
ORDERTOCASH = ORDER_LENS
