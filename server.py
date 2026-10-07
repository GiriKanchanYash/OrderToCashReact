"""OrderToCash entry point with data-source selection (Snowflake | Fabric).

    uvicorn server:app --port 8001

The Snowflake app (app/main.py) and the Fabric app (FabicApp/main.py) are
mounted side by side, unmodified. Each /api request is dispatched to one of
them based on the data source the user picked in the UI selector:

  1. X-Data-Source request header        ("snowflake" | "fabric")
  2. o2c_data_source cookie              (set by the UI selector)
  3. DEFAULT_DATA_SOURCE env var         (default "snowflake")

Everything else (the SPA, /assets, /health) is served by the Snowflake app,
exactly as before. /health/fabric always goes to the Fabric app.

GET /api/data-source reports the active selection and which sources loaded.
"""
from __future__ import annotations

import json
import logging
import os
from http.cookies import SimpleCookie

from app.main import app as snowflake_app

log = logging.getLogger("o2c.gateway")

SOURCES = ("snowflake", "fabric")
LABELS = {"snowflake": "Snowflake", "fabric": "Microsoft Fabric"}
COOKIE_NAME = "o2c_data_source"
HEADER_NAME = b"x-data-source"

_default = os.getenv("DEFAULT_DATA_SOURCE", "snowflake").strip().lower()
DEFAULT_SOURCE = _default if _default in SOURCES else "snowflake"

try:
    from FabicApp.main import app as fabric_app
    FABRIC_ERROR: str | None = None
except Exception as exc:  # e.g. pyodbc / ODBC driver missing: keep Snowflake running
    log.exception("Fabric data source failed to load")
    fabric_app = None
    FABRIC_ERROR = f"{type(exc).__name__}: {exc}"


def _selected_source(scope) -> str:
    headers = dict(scope.get("headers") or [])
    value = headers.get(HEADER_NAME, b"").decode("latin-1").strip().lower()
    if value in SOURCES:
        return value
    raw_cookie = headers.get(b"cookie", b"").decode("latin-1")
    if raw_cookie:
        try:
            morsel = SimpleCookie(raw_cookie).get(COOKIE_NAME)
        except Exception:
            morsel = None
        if morsel is not None and morsel.value.strip().lower() in SOURCES:
            return morsel.value.strip().lower()
    return DEFAULT_SOURCE


async def _send_json(send, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"),
        ],
    })
    await send({"type": "http.response.body", "body": body})


class DataSourceGateway:
    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        path = scope.get("path", "")
        if scope["type"] == "http" and path.rstrip("/") == "/api/data-source":
            await _send_json(send, 200, {
                "selected": _selected_source(scope),
                "default": DEFAULT_SOURCE,
                "sources": [
                    {"id": "snowflake", "label": LABELS["snowflake"], "available": True},
                    {"id": "fabric", "label": LABELS["fabric"], "available": fabric_app is not None,
                     "error": FABRIC_ERROR},
                ],
            })
            return

        use_fabric = path.startswith("/health/fabric") or (
            path.startswith("/api") and _selected_source(scope) == "fabric"
        )
        if not use_fabric:
            await snowflake_app(scope, receive, send)
            return
        if fabric_app is None:
            if scope["type"] == "http":
                await _send_json(send, 503, {"detail": f"Fabric data source is unavailable: {FABRIC_ERROR}"})
            return
        await fabric_app(scope, receive, send)


app = DataSourceGateway()
