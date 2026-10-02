ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-config.json}"
export OUT_NAME="${OUT_NAME:-cluster}"
export PYTHON_MODULE="${PYTHON_MODULE:-}"

if [ -n "$PYTHON_MODULE" ]; then
  module load $PYTHON_MODULE
fi

ensure_env() {
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "building $ROOT/.venv (one time)"
    python3 -m venv "$ROOT/.venv" || return 1
    "$ROOT/.venv/bin/python" -m pip install --quiet --upgrade pip || return 1
    "$ROOT/.venv/bin/python" -m pip install --quiet -r "$ROOT/requirements.txt" || return 1
  fi
  PY="$ROOT/.venv/bin/python"
}

study_list() {
  case "${1:-all}" in
    all) echo "main extra" ;;
    main|extra) echo "$1" ;;
    *) echo "usage: $0 [main|extra|all]" >&2; return 1 ;;
  esac
}
