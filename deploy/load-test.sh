#!/usr/bin/env bash
# Prueba de carga corta sobre la API y la portada.
#
# Uso: bash deploy/load-test.sh BASE_URL [CONCURRENCIA=4] [SEGUNDOS=20]
#   BASE_URL  nginx/ALB (https://<host>, http://127.0.0.1:8043) o uvicorn (http://127.0.0.1:8001)
#
# Usa `hey` si esta instalado (-q 5: 5 peticiones/s por cliente; con la concurrencia por defecto, 4, son 20 r/s,
# el limite por IP de nginx, porque todos los clientes salen de la misma IP) y, si no, `ab` (ApacheBench, sin
# limite por cliente). Mide /api/v1/forecast/es/nowcast/summary y /, imprime p95 y codigos de respuesta, y
# despues hace 50 `curl -sI` a la ruta de la API contando las cabeceras `X-Cache: HIT` (solo hay con nginx delante).
# Sale con 1 si alguna respuesta es 5xx o si la primera peticion responde 429; con 2 si falta herramienta.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "uso: $0 BASE_URL [CONCURRENCIA=4] [SEGUNDOS=20]" >&2
    exit 2
fi
base="${1%/}"
conc="${2:-4}"
secs="${3:-20}"
api="/api/v1/forecast/es/nowcast/summary"
out="$(mktemp)"
trap 'rm -f "$out"' EXIT
failures=0

first="$(curl -s -o /dev/null --max-time 20 -w '%{http_code}' "$base$api" || true)"
echo "primera peticion a $api: $first"
if [ "$first" = "429" ]; then
    echo "FAIL: la primera peticion responde 429 (limite de peticiones ya agotado)" >&2
    exit 1
fi
case "$first" in
    2??|3??) ;;
    *)
        echo "FAIL primera peticion: ${first:-sin respuesta}" >&2
        exit 1
        ;;
esac

if command -v hey >/dev/null 2>&1; then
    tool=hey
elif command -v ab >/dev/null 2>&1; then
    tool=ab
else
    echo "no se encuentra hey ni ab" >&2
    exit 2
fi

for path in "$api" "/"; do
    echo
    echo "== $tool $conc clientes, ${secs}s: $path"
    if [ "$tool" = hey ]; then
        hey -z "${secs}s" -c "$conc" -q 5 "$base$path" >"$out" 2>&1 || true
        grep -E '95% in|Requests/sec|\[[0-9]+\]' "$out" || true
        if grep -qE '^\s*\[(429|5[0-9][0-9])\]' "$out"; then
            echo "FAIL: respuestas 429 o 5xx en $path" >&2
            failures=$((failures + 1))
        fi
    else
        ab -t "$secs" -c "$conc" "$base$path" >"$out" 2>&1 || true
        grep -E '^\s*95%|Requests per second|Complete requests|Failed requests|Non-2xx' "$out" || true
        non2xx="$(sed -nE 's/^Non-2xx responses:[[:space:]]+([0-9]+).*/\1/p' "$out")"
        if [ -n "$non2xx" ] && [ "$non2xx" -gt 0 ]; then
            echo "FAIL: $non2xx respuestas no 2xx en $path" >&2
            failures=$((failures + 1))
        fi
        if ! grep -q 'Complete requests' "$out"; then
            echo "FAIL: ab no completo la prueba en $path" >&2
            failures=$((failures + 1))
        fi
    fi
done

echo
echo "== 50 cabeceras de $api"
hit=0
five=0
for _ in $(seq 1 50); do
    headers="$(curl -sI --max-time 20 "$base$api" || true)"
    code="$(printf '%s' "$headers" | head -n 1 | awk '{print $2}')"
    case "$code" in
        5*|000|"") five=$((five + 1)) ;;
    esac
    if printf '%s' "$headers" | grep -qi '^x-cache: *hit'; then
        hit=$((hit + 1))
    fi
done
echo "X-Cache: HIT en $hit de 50; respuestas 5xx o sin respuesta: $five"
if [ "$five" -gt 0 ]; then
    echo "FAIL: respuestas 5xx" >&2
    failures=$((failures + 1))
fi

if [ "$failures" -gt 0 ]; then
    exit 1
fi
echo "OK: sin 5xx"
