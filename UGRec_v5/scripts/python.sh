#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UGREC_REPO="$(cd .. && pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$UGREC_REPO/.venv/bin/python" ]]; then
  exec "$UGREC_REPO/.venv/bin/python" "$@"
fi
exec uv run --project "$UGREC_REPO" python "$@"
