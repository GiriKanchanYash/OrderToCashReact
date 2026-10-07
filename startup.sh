#!/usr/bin/env bash
# Azure App Service (Linux) startup for OrderToCash FastAPI backend.
#
# Handles both deployment shapes:
#   1. Plain zip deploy  -> app code sits in /home/site/wwwroot
#   2. Oryx build output -> wwwroot holds output.tar.zst and Oryx extracts the
#      real app to /tmp/<id>, leaving `app` off sys.path (ModuleNotFoundError).
#
# Never simplify to a bare `uvicorn app.main:app` — it breaks in case 2.
set -e

find_app_dir() {
  for candidate in "$APP_PATH" /home/site/wwwroot; do
    if [ -n "$candidate" ] && [ -f "$candidate/app/main.py" ]; then
      echo "$candidate"
      return 0
    fi
  done
  # Oryx extracts the build output to /tmp/<id>; pick the newest match.
  for candidate in $(ls -dt /tmp/*/ 2>/dev/null); do
    if [ -f "${candidate}app/main.py" ]; then
      echo "${candidate%/}"
      return 0
    fi
  done
  return 1
}

APP_DIR="$(find_app_dir || true)"

if [ -z "$APP_DIR" ]; then
  echo "FATAL: could not locate app/main.py (checked \$APP_PATH, /home/site/wwwroot, /tmp/*)" >&2
  ls -la /home/site/wwwroot >&2 || true
  exit 1
fi

echo "==> Using APP_DIR=$APP_DIR"
cd "$APP_DIR"

# Activate a virtualenv if one was produced by Oryx.
for venv in "$APP_DIR/antenv" /home/site/wwwroot/antenv; do
  if [ -f "$venv/bin/activate" ]; then
    echo "==> Activating virtualenv $venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    break
  fi
done

# Install dependencies only when FastAPI is missing (cold start without Oryx build).
if ! python -c "import fastapi" >/dev/null 2>&1; then
  echo "==> Installing requirements"
  python -m pip install --upgrade pip -q
  pip install -r "$APP_DIR/requirements.txt" -q
fi

# Oryx rewrites PYTHONPATH to site-packages only; re-add the app root.
export PYTHONPATH="$APP_DIR:${PYTHONPATH}"

echo "==> Static dir contents:"
ls -la "$APP_DIR/static" 2>/dev/null | head -n 5 || echo "    (no static/ directory)"

# server:app wraps app.main:app (Snowflake) and FabicApp.main:app (Fabric) behind
# the UI data-source selector.
exec python -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}"
