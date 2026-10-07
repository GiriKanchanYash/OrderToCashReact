import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import HTTPException
from snowflake.connector.errors import ProgrammingError
from app.routers import o2c, ai, auth

app = FastAPI(title="OrderToCash API", version="1.0.0")

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


@app.exception_handler(ProgrammingError)
async def handle_snowflake_programming_error(_request, exc: ProgrammingError):
    msg = str(exc)
    if "resource monitor" in msg.lower() and "has exceeded its quota" in msg.lower():
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Snowflake warehouse quota exceeded. "
                    "Ask your Snowflake admin to resume/increase resource monitor quota "
                    "or switch to an available warehouse."
                )
            },
        )
    return JSONResponse(status_code=500, content={"detail": msg})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "OrderToCash",
        "static": bool(_static_dir),
        "static_dir": _static_dir,
    }


def _resolve_static_dir() -> str | None:
    """Find frontend build dir for Azure (/home/site/wwwroot/static) and local runs."""
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
    if path.startswith("/api") or path == "/health":
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    if _static_dir:
        index = os.path.join(_static_dir, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
    return JSONResponse(status_code=404, content={"detail": "Not found"})
