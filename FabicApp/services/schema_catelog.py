"""
Loads the OrderLens metadata catalog (orderlens_metadata.yaml, shipped
alongside this app) and exposes it in two forms used by text_to_sql.py:

1. A whitelist of table/column names the LLM is allowed to reference --
   built strictly from the catalog's analytical domains (customer, dealer,
   forecasting, inventory, orders, product). The catalog also has an
   `application_metadata` domain (GENIE_QUESTION_HISTORY, SAVED_INSIGHTS,
   AGENT_REVIEW_LOG) which is deliberately EXCLUDED here: those are
   app-internal operational tables, not analytical data meant for ad hoc
   natural-language querying, and their casing in the catalog (uppercase,
   inherited from the original Snowflake naming convention) doesn't match
   the actual lowercase objects living in the Fabric warehouse anyway.

2. A condensed, LLM-friendly text rendering of that same whitelist (table
   descriptions + column name/type/description) to ground SQL generation.

All table/column names are normalized to lowercase to match Fabric's actual
(case-sensitive-on-table-name) object names, confirmed via a live sys.tables
diagnostic against the warehouse -- see app/db.py's qview()/qcol() for the
same normalization applied to the rest of the app's hand-written SQL.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import yaml

logger = logging.getLogger("orderlens.schema_catalog")

# app/services/schema_catalog.py -> app/ -> project root
_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent

# Users have placed this file under a few different names/locations across
# deployments (app/orderlens_metadata.yaml, project-root orderlens.yaml,
# OrderLens.yaml, ...). Rather than fail because of a naming mismatch, try
# every combination we've actually seen, in priority order, before giving up.
_CANDIDATE_NAMES = ["orderlens_metadata.yaml", "orderlens.yaml", "OrderLens.yaml", "orderlens_metadata.yml", "orderlens.yml"]
_CANDIDATE_DIRS = [_APP_DIR, _PROJECT_ROOT]


def _resolve_yaml_path() -> Path:
    env_override = os.getenv("ORDERLENS_SCHEMA_YAML", "").strip()
    if env_override:
        p = Path(env_override).expanduser()
        if p.exists():
            return p
        raise SchemaCatalogError(
            f"ORDERLENS_SCHEMA_YAML is set to '{env_override}' but that file doesn't exist."
        )

    for d in _CANDIDATE_DIRS:
        for name in _CANDIDATE_NAMES:
            p = d / name
            if p.exists():
                return p

    # Last resort: case-insensitive glob for anything yaml-ish mentioning
    # "orderlens" in either candidate directory.
    for d in _CANDIDATE_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("*"):
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml") and "orderlens" in p.name.lower():
                return p

    tried = ", ".join(str(d / n) for d in _CANDIDATE_DIRS for n in _CANDIDATE_NAMES)
    raise SchemaCatalogError(
        "Schema catalog YAML not found. The text-to-SQL feature needs it in the app "
        f"directory or project root. Tried: {tried}. Set ORDERLENS_SCHEMA_YAML in .env "
        "to point at it explicitly if it lives somewhere else."
    )


# Domains containing app-internal / operational tables, not analytical data.
# Never exposed to text-to-SQL generation.
_EXCLUDED_DOMAINS = {"application_metadata"}

_lock = threading.Lock()
_catalog: dict | None = None


class SchemaCatalogError(RuntimeError):
    """Raised when the metadata catalog is missing or malformed."""


def _load() -> dict:
    """Parses the YAML once and caches the result. Returns:
    {
      "tables": {lowercase_table_name: {"description": str, "columns": [{"name","data_type","description"}]}},
      "allowed_tables": frozenset[str],
      "allowed_columns": {lowercase_table_name: frozenset[lowercase_column_name]},
    }
    """
    global _catalog
    with _lock:
        if _catalog is not None:
            return _catalog
        yaml_path = _resolve_yaml_path()
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise SchemaCatalogError(f"Failed to parse {yaml_path.name}: {exc}") from exc
        logger.info("Schema catalog source: %s", yaml_path)

        domains = (raw or {}).get("domains", {})
        tables: dict[str, dict] = {}
        for domain_name, domain_tables in domains.items():
            if domain_name in _EXCLUDED_DOMAINS:
                continue
            for _table_key, tmeta in (domain_tables or {}).items():
                real_name = str(tmeta.get("table_name") or _table_key).strip().lower()
                columns = []
                for c in tmeta.get("columns", []) or []:
                    cname = str(c.get("name", "")).strip().lower()
                    if not cname:
                        continue
                    columns.append({
                        "name": cname,
                        "data_type": c.get("data_type", ""),
                        "description": c.get("description", ""),
                    })
                if not columns:
                    logger.warning("Table %s in catalog has no columns - skipping", real_name)
                    continue
                tables[real_name] = {
                    "domain": domain_name,
                    "description": tmeta.get("description", ""),
                    "columns": columns,
                }

        if not tables:
            raise SchemaCatalogError(
                f"{_YAML_PATH.name} parsed but yielded zero usable analytical tables."
            )

        _catalog = {
            "tables": tables,
            "allowed_tables": frozenset(tables.keys()),
            "allowed_columns": {t: frozenset(m["name"] for m in v["columns"]) for t, v in tables.items()},
        }
        logger.info("Schema catalog loaded: %d analytical tables", len(tables))
        return _catalog


def allowed_tables() -> frozenset[str]:
    return _load()["allowed_tables"]


def allowed_columns(table: str) -> frozenset[str]:
    return _load()["allowed_columns"].get(table.strip().lower(), frozenset())


def table_meta(table: str) -> dict | None:
    return _load()["tables"].get(table.strip().lower())


def schema_prompt_text(schema_prefix: str) -> str:
    """Renders the whitelisted catalog as compact text for the LLM prompt.
    schema_prefix is the live Fabric schema (e.g. React_OrderLens) so the
    model always emits fully schema-qualified table references."""
    cat = _load()
    lines = [
        f"You may ONLY query these tables, each schema-qualified as "
        f"{schema_prefix}.<table_name> exactly as named below (all lowercase):",
        "",
    ]
    by_domain: dict[str, list[str]] = {}
    for tname, meta in cat["tables"].items():
        by_domain.setdefault(meta["domain"], []).append(tname)

    for domain in sorted(by_domain):
        lines.append(f"## {domain}")
        for tname in sorted(by_domain[domain]):
            meta = cat["tables"][tname]
            lines.append(f"### {schema_prefix}.{tname}")
            if meta["description"]:
                lines.append(f"{meta['description']}")
            for col in meta["columns"]:
                desc = f" - {col['description']}" if col["description"] else ""
                lines.append(f"  - {col['name']} ({col['data_type']}){desc}")
        lines.append("")
    return "\n".join(lines)