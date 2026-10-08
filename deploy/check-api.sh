#!/usr/bin/env bash
# Recorre las rutas de la API y comprueba codigo, content-type y JSON valido.
#
# Uso: bash deploy/check-api.sh BASE_URL
#   BASE_URL  nginx (http://127.0.0.1:8042) o gunicorn/uvicorn (http://127.0.0.1:8000)
#
# Imprime una linea por ruta (OK / FAIL / n/a) y sale con error si alguna falla.
# /healthz solo existe en nginx: un 404 se muestra como "n/a sin nginx".
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "uso: $0 BASE_URL" >&2
    exit 2
fi
base="${1%/}"
body="$(mktemp)"
trap 'rm -f "$body"' EXIT
failures=0
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "no se encuentra python" >&2; exit 2; }

# check RUTA STATUS_ESPERADO TIPO   (TIPO: json | csv | text | none)
check() {
    local path="$1" want="$2" kind="$3" out code ctype problem=""
    out="$(curl -s -o "$body" -w '%{http_code} %{content_type}' "$base$path" || true)"
    code="${out%% *}"
    ctype="${out#* }"
    if [ "$code" != "$want" ]; then
        problem="status $code (esperado $want)"
    elif [ "$kind" = json ] && ! "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$body" 2>/dev/null; then
        problem="JSON no valido"
    elif [ "$kind" = csv ] && [[ "$ctype" != text/csv* ]]; then
        problem="content-type $ctype (esperado text/csv)"
    fi
    if [ -n "$problem" ]; then
        echo "FAIL $path: $problem"
        failures=$((failures + 1))
    else
        echo "OK   $path ($code $ctype)"
    fi
}

# /healthz: 200 en nginx, 404 en gunicorn/uvicorn.
out="$(curl -s -o /dev/null -w '%{http_code}' "$base/healthz" || true)"
case "$out" in
    200) echo "OK   /healthz (200)" ;;
    404) echo "n/a  /healthz sin nginx" ;;
    *) echo "FAIL /healthz: status $out"; failures=$((failures + 1)) ;;
esac

check /api/v1/health 200 json
check /api/v1/manifest 200 json
check /api/v1/scopes 200 json
check /api/v1/forecast/es 200 json
check /api/v1/forecast/es/runs 200 json
check /api/v1/forecast/es/headline 200 json
check /api/v1/forecast/es/forecast/vote 200 json
check "/api/v1/forecast/es/forecast/summary?format=csv" 200 csv
check /api/v1/forecast/es-xx 400 none
check /api/v1/forecast/es/nope 404 none

if [ "$failures" -gt 0 ]; then
    echo "$failures ruta(s) con error" >&2
    exit 1
fi
echo "todas las rutas OK"
