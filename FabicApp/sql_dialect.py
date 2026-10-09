"""
Snowflake-dialect SQL -> Microsoft Fabric Warehouse T-SQL.

The Fabric services in FabicApp/services are faithful mirrors of the Snowflake
reference services in app/services (same functions, same queries, same response
shapes). Rather than hand-maintaining a second copy of ~150 queries, the Fabric
data layer (FabicApp/db.py) passes every query through to_tsql() before it is
sent to Fabric. sqlglot does the bulk of the dialect work (IFF->IIF,
LIMIT->TOP/OFFSET-FETCH, ::casts, DATEADD/DATEDIFF/DATE_TRUNC, QUALIFY,
NULLS LAST, MERGE, booleans), and the transforms below close the gaps where a
straight syntax translation would run but silently return *different numbers*
than Snowflake:

* AVG(int) and int/int truncate in T-SQL (Snowflake returns decimals).
* CURRENT_DATE() must stay a DATE (GETDATE() carries a time component).
* DATE_TRUNC('WEEK') starts Monday in Snowflake, Sunday in T-SQL (us_english).
* GROUP BY ordinal / GROUP BY select-alias are not valid T-SQL.
* ORDER BY inside a CTE/derived table (without TOP) is not valid T-SQL.
* Derived tables must be aliased in T-SQL.
* CAST(x AS VARCHAR) defaults to VARCHAR(30) in T-SQL, and a datetime cast to
  varchar renders as 'Oct  7 2026 10:00AM' instead of ISO; date-like values are
  converted with an ISO style so period labels match Snowflake's output.

Hand-written T-SQL (e.g. the weekly-forecast query) bypasses this module via
db.run_tsql(); see FabicApp/services/weekly_forecast.py.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

import sqlglot
from sqlglot import exp

logger = logging.getLogger("o2c.fabric.sql_dialect")

# Optional schema renames, e.g. FABRIC_SCHEMA_MAP="BUSINESS_MART=bm,SAP_STG=stg".
# Default: Fabric warehouse uses the same schema names as Snowflake.
_SCHEMA_MAP = {
    k.strip().upper(): v.strip()
    for k, v in (
        pair.split("=", 1)
        for pair in os.getenv("FABRIC_SCHEMA_MAP", "").split(",")
        if "=" in pair
    )
}

# Fabric Warehouse object names are case-sensitive. "preserve" keeps the
# Snowflake (upper-case) names as written; "lower"/"upper" force a case.
_IDENT_CASE = os.getenv("FABRIC_IDENTIFIER_CASE", "preserve").strip().lower()

_DATE_HINT = re.compile(r"(DATE|_DT$|_TS$|TIMESTAMP|MONTH|PERIOD|_AT$|WEEK|DAY$)", re.I)


def _is_date_like(node: exp.Expression) -> bool:
    """Heuristic: does this expression produce a DATE/TIMESTAMP value?"""
    if isinstance(node, (exp.CurrentDate, exp.CurrentTimestamp, exp.DateTrunc,
                         exp.TimestampTrunc, exp.DateAdd, exp.TsOrDsAdd, exp.TimestampAdd)):
        return True
    if isinstance(node, exp.Cast) and node.to.is_type(*exp.DataType.TEMPORAL_TYPES):
        return True
    if isinstance(node, (exp.Max, exp.Min, exp.Coalesce, exp.Paren)):
        return _is_date_like(node.this)
    if isinstance(node, exp.Column):
        return bool(_DATE_HINT.search(node.name or ""))
    return False


def _float(node: exp.Expression) -> exp.Expression:
    return exp.Cast(this=node, to=exp.DataType.build("FLOAT"))


def _transform(node: exp.Expression) -> exp.Expression:
    # CURRENT_DATE() -> CAST(GETDATE() AS DATE)
    if isinstance(node, exp.CurrentDate):
        return exp.Cast(this=exp.Anonymous(this="GETDATE"), to=exp.DataType.build("DATE"))

    # CURRENT_USER() -> CURRENT_USER (T-SQL niladic function, no parentheses)
    if isinstance(node, exp.CurrentUser):
        return exp.Var(this="CURRENT_USER")

    # AVG(int_col) truncates in T-SQL; Snowflake returns a decimal.
    if isinstance(node, exp.Avg):
        inner = node.this
        if not (isinstance(inner, exp.Cast) and inner.to.is_type("float")):
            node.set("this", _float(inner))
        return node

    # int / int truncates in T-SQL; Snowflake returns a decimal.
    if isinstance(node, exp.Div):
        left = node.this
        if not (isinstance(left, exp.Cast) and left.to.is_type("float")):
            node.set("this", _float(left))
        return node

    # VARIANCE -> VAR
    if isinstance(node, (exp.Variance, exp.VariancePop)):
        name = "VAR" if isinstance(node, exp.Variance) else "VARP"
        return exp.Anonymous(this=name, expressions=[node.this])

    # Snowflake weeks start Monday; T-SQL DATETRUNC(WEEK) follows DATEFIRST (Sunday).
    if isinstance(node, (exp.DateTrunc, exp.TimestampTrunc)):
        unit = node.args.get("unit")
        if unit is not None and unit.name.upper() == "WEEK":
            node.set("unit", exp.var("ISO_WEEK"))
        return node

    # CAST(x AS VARCHAR) without length -> sized; date-like -> ISO string.
    if isinstance(node, exp.Cast) and node.to.is_type("varchar", "char", "text", "nvarchar"):
        if not node.to.expressions:
            if _is_date_like(node.this):
                return exp.Anonymous(
                    this="CONVERT",
                    expressions=[exp.DataType.build("VARCHAR(30)", dialect="tsql"), node.this, exp.Literal.number(23 if _yields_pure_date(node.this) else 121)],
                )
            node.set("to", exp.DataType.build("VARCHAR(8000)", dialect="tsql"))
        return node

    # Schema remap / identifier casing on table references.
    if isinstance(node, exp.Table):
        db = node.args.get("db")
        if db is not None and db.name.upper() in _SCHEMA_MAP:
            node.set("db", exp.to_identifier(_SCHEMA_MAP[db.name.upper()]))
        return node

    return node


def _yields_pure_date(node: exp.Expression) -> bool:
    if isinstance(node, (exp.CurrentDate, exp.DateTrunc, exp.TimestampTrunc)):
        return True
    if isinstance(node, exp.Cast) and node.to.is_type("date"):
        return True
    if isinstance(node, (exp.Max, exp.Min, exp.Coalesce, exp.Paren)):
        return _yields_pure_date(node.this)
    if isinstance(node, exp.Column):
        name = (node.name or "").upper()
        return ("DATE" in name or "MONTH" in name or "PERIOD" in name) and "TIMESTAMP" not in name
    return False


def _fix_group_by(select: exp.Select) -> None:
    """GROUP BY 1 / GROUP BY <alias> -> GROUP BY <expression> (T-SQL requirement)."""
    group = select.args.get("group")
    if not group:
        return
    projections = select.expressions
    alias_map: dict[str, exp.Expression] = {}
    for p in projections:
        if isinstance(p, exp.Alias):
            alias_map[p.alias.upper()] = p.this
    new_items = []
    for item in group.expressions:
        if isinstance(item, exp.Literal) and item.is_int:
            idx = int(item.this) - 1
            if 0 <= idx < len(projections):
                proj = projections[idx]
                new_items.append((proj.this if isinstance(proj, exp.Alias) else proj).copy())
                continue
        if isinstance(item, exp.Column) and not item.table:
            target = alias_map.get(item.name.upper())
            if target is not None and not (isinstance(target, exp.Column) and target.name.upper() == item.name.upper()):
                new_items.append(target.copy())
                continue
        new_items.append(item)
    group.set("expressions", new_items)


def _is_inner_select(select: exp.Select) -> bool:
    parent = select.parent
    while parent is not None:
        if isinstance(parent, (exp.CTE, exp.Subquery, exp.In, exp.Exists)):
            return True
        if isinstance(parent, (exp.Union, exp.Intersect, exp.Except)):
            return True
        parent = parent.parent
    return False


def _fix_distinct_null_ordering(select: exp.Select) -> None:
    """SELECT DISTINCT ... ORDER BY x: sqlglot emulates Snowflake's NULLS LAST
    by adding 'CASE WHEN x IS NULL ...' to the ORDER BY, which T-SQL rejects
    with DISTINCT (error 145: ORDER BY items must appear in the select list).
    Use T-SQL's native null placement instead; non-null ordering is unchanged."""
    order = select.args.get("order")
    if not (select.args.get("distinct") and order):
        return
    for ordered in order.expressions:
        if isinstance(ordered, exp.Ordered):
            ordered.set("nulls_first", not ordered.args.get("desc"))


def _structural_fixes(tree: exp.Expression) -> None:
    alias_n = 0
    for select in list(tree.find_all(exp.Select)):
        _fix_group_by(select)
        _fix_distinct_null_ordering(select)
        # ORDER BY inside CTE / subquery without a row limit is invalid T-SQL.
        if select.args.get("order") and not (select.args.get("limit") or select.args.get("offset")) and _is_inner_select(select):
            select.set("order", None)
    # Derived tables must be aliased in T-SQL.
    for sub in list(tree.find_all(exp.Subquery)):
        if isinstance(sub.parent, (exp.From, exp.Join)) and not sub.alias:
            alias_n += 1
            sub.set("alias", exp.TableAlias(this=exp.to_identifier(f"_sq{alias_n}")))


def _apply_identifier_case(tree: exp.Expression) -> None:
    if _IDENT_CASE not in ("lower", "upper"):
        return
    fn = str.lower if _IDENT_CASE == "lower" else str.upper
    for ident in tree.find_all(exp.Identifier):
        parent = ident.parent
        if isinstance(parent, (exp.Table, exp.Column)) or (isinstance(parent, exp.Dot)):
            ident.set("this", fn(ident.this))


@lru_cache(maxsize=4096)
def to_tsql(sql: str) -> str:
    """Translate one Snowflake SQL statement to Fabric T-SQL (cached)."""
    src = sql.replace("%s", "?")
    tree = sqlglot.parse_one(src, read="snowflake")
    tree = tree.transform(_transform)
    _structural_fixes(tree)
    _apply_identifier_case(tree)
    out = tree.sql(dialect="tsql")
    if isinstance(tree, exp.Merge):
        out += ";"  # T-SQL requires MERGE to be terminated by a semicolon (error 10713)
    logger.debug("to_tsql:\n%s\n=>\n%s", sql, out)
    return out
