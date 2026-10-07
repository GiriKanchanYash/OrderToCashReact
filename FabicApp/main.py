import logging
import os
import traceback

import pyodbc
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import HTTPException

from FabicApp.db import FabricConnectionError
from FabicApp.routers import o2c, ai, auth

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("OrderToCash")

app = FastAPI(title="OrderToCash API (Fabric)", version="1.0.0")

_default_origins = "http://localhost:5173,http://localhost:5174,http://localhost:3000"
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(o2c.router)
app.include_router(ai.router)
app.include_router(auth.router)


_DEBUG = os.getenv("ORDERTOCASH_DEBUG", "true").strip().lower() in {"1", "true", "yes"}


@app.exception_handler(FabricConnectionError)
async def handle_fabric_connection_error(request: Request, exc: FabricConnectionError):
    # db.py already logged the raw pyodbc error + hint with full detail.
    # This handler just decides what the client sees.
    logger.error("Fabric error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(pyodbc.Error)
async def handle_pyodbc_error(request: Request, exc: pyodbc.Error):
    # Any pyodbc.Error that reaches here bypassed db.py's translation (e.g. a
    # driver-level error thrown somewhere else). Log full detail either way.
    logger.exception("Unhandled pyodbc.Error on %s", request.url.path)
    msg = str(exc)
    lowered = msg.lower()
    if "login failed" in lowered or "cannot open server" in lowered:
        return JSONResponse(
            status_code=503,
            content={"detail": "Could not authenticate to Fabric. Check service principal credentials and Warehouse/Lakehouse permissions."},
        )
    if "denied on the requested resource" in lowered or "external policy action" in lowered:
        return JSONResponse(
            status_code=503,
            content={"detail": "Fabric denied this operation (read-only endpoint or missing Write permission). See app logs for details."},
        )
    return JSONResponse(status_code=503, content={"detail": f"Fabric database error: {msg}"})


@app.exception_handler(RuntimeError)
async def handle_runtime_error(request: Request, exc: RuntimeError):
    logger.exception("Unhandled RuntimeError on %s", request.url.path)
    msg = str(exc)
    if "Fabric connection details missing" in msg or "Fabric interactive auth requires" in msg:
        return JSONResponse(status_code=503, content={"detail": msg})
    return JSONResponse(status_code=500, content={"detail": msg})


@app.exception_handler(Exception)
async def handle_any_error(request: Request, exc: Exception):
    # Always log the full traceback server-side so it shows up in the console
    logger.exception("Unhandled error on %s", request.url.path)
    content = {"detail": str(exc) or type(exc).__name__, "type": type(exc).__name__}
    if _DEBUG:
        # Opt-in only: set ORDERTOCASH_DEBUG=false in production so raw
        # tracebacks are never sent to clients.
        content["traceback"] = traceback.format_exc().splitlines()[-15:]
    return JSONResponse(status_code=500, content=content)


@app.get("/health")
def health():
    return {"status": "ok", "app": "OrderToCash", "backend": "fabric"}


@app.get("/health/fabric")
def health_fabric():
    """Round-trips a trivial query to Fabric so connection issues surface here
    (with the same FabricConnectionError translation/logging) instead of only
    on the first real /api/o2c/* call."""
    from FabicApp.db import run_query

    run_query("SELECT 1 AS ok")
    return {"status": "ok", "backend": "fabric"}


def _resolve_static_dir() -> str | None:
    """Same lookup as the Snowflake app (Azure /home/site/wwwroot/static and local runs)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, "..", "static")),
        os.path.normpath(os.path.join(os.getcwd(), "static")),
        "/home/site/wwwroot/static",
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "index.html")):
            return path
    return None


_static_dir = _resolve_static_dir()
if _static_dir:
    _assets = os.path.join(_static_dir, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")


@app.get("/")
def spa_index():
    if _static_dir:
        index = os.path.join(_static_dir, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
    return JSONResponse(status_code=404, content={"detail": "Frontend not deployed (static/index.html missing)"})


@app.exception_handler(404)
async def spa_fallback(request: Request, exc: HTTPException):
    path = request.url.path
    if path.startswith("/api") or path.startswith("/health"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    if _static_dir:
        index = os.path.join(_static_dir, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
    return JSONResponse(status_code=404, content={"detail": "Not found"})
