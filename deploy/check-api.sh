#!/usr/bin/env bash
# Recorre las rutas de la API y comprueba codigo, content-type y JSON valido.
#
# Uso: bash deploy/check-api.sh BASE_URL
#   BASE_URL  nginx (http://127.0.0.1:8042) o gunicorn/uvicorn (http://127.0.0.1:8000)
#
# Imprime una linea por ruta (OK / FAIL / n/a) y sale con error si alguna falla.
# /healthz solo existe en nginx: un 404 se muestra como "n/a sin nginx".
# Las rutas que leen la base (/casas, /api/v1/ratings, polls, results, drift y events/{scope}) se marcan
# "n/a sin base" si /api/v1/parties responde 503 "no database configured" (sin DB_HOST).
# Las que sirven lo que escribe `publish event` o `publish backtest` se marcan "n/a sin publicar" si
# responden 404 "not published" (API) o "no publicada/o" (paginas): el paquete puede no tenerlas.
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

# check RUTA STATUS_ESPERADO TIPO [PREFIJO_CONTENT_TYPE]   (TIPO: json | csv | html | none)
check() {
    local path="$1" want="$2" kind="$3" out code ctype problem=""
    out="$(curl -s --max-time 20 -o "$body" -w '%{http_code} %{content_type}' "$base$path" || true)"
    code="${out%% *}"
    ctype="${out#* }"
    if [ "$code" != "$want" ]; then
        problem="status $code (esperado $want)"
    elif [ "$kind" = json ] && ! "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$body" 2>/dev/null; then
        problem="JSON no valido"
    elif [ "$kind" = csv ] && [[ "$ctype" != text/csv* ]]; then
        problem="content-type $ctype (esperado text/csv)"
    elif [ -n "${4:-}" ] && [[ "$ctype" != $4* ]]; then
        problem="content-type $ctype (esperado $4)"
    elif [ "$kind" = html ] && [[ "$ctype" != text/html* ]]; then
        problem="content-type $ctype (esperado text/html)"
    fi
    if [ -n "$problem" ]; then
        echo "FAIL $path: $problem"
        failures=$((failures + 1))
    else
        echo "OK   $path ($code $ctype)"
    fi
}

# /healthz: 200 con cuerpo "ok" en nginx, 404 (pagina de Python) en gunicorn/uvicorn.
# Aqui se decide una sola vez si hay nginx delante (has_nginx).
has_nginx=0
out="$(curl -s --max-time 20 -o "$body" -w '%{http_code}' "$base/healthz" || true)"
case "$out" in
    200)
        if [ "$(cat "$body")" = ok ]; then
            has_nginx=1
            echo "OK   /healthz (200)"
            # IP del cliente tal como la ve nginx (X-Client-IP): detras de un ALB debe ser la publica.
            client_ip="$(curl -s -D - -o /dev/null --max-time 20 "$base/healthz" | tr -d '\r' | awk 'tolower($1) == "x-client-ip:" {print $2; exit}' || true)"
            echo "info cliente según nginx: ${client_ip:-ausente}"
            host="${base#*://}"; host="${host%%/*}"; host="${host%%:*}"
            if [[ "$host" != 127.0.0.1 && "$host" != localhost ]] \
                && [[ "$client_ip" =~ ^(127\.|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.) ]]; then
                echo "WARN cliente según nginx es privado ($client_ip): revisa set_real_ip_from y X-Forwarded-For del ALB"
            fi
        else
            echo "FAIL /healthz: 200 sin cuerpo 'ok'"; failures=$((failures + 1))
        fi ;;
    404) echo "n/a  /healthz sin nginx" ;;
    *) echo "FAIL /healthz: status $out"; failures=$((failures + 1)) ;;
esac

# check_nginx_only RUTA TIPO: /dist/ solo lo sirve nginx; sin nginx es n/a, con nginx un 404 es FAIL.
check_nginx_only() {
    if [ "$has_nginx" -eq 1 ]; then
        check "$1" 200 none "$2"
    else
        echo "n/a  $1 sin nginx"
    fi
}

# /api/v1/parties lee la base: un 503 "no database configured" decide una sola vez (has_db) que no hay
# base; cualquier otra respuesta (incluido un 503 "database unavailable") cuenta como base configurada
# y las comprobaciones de abajo la ejercitan.
has_db=1
out="$(curl -s --max-time 20 -o "$body" -w '%{http_code}' "$base/api/v1/parties" || true)"
if [ "$out" = 503 ] && grep -q 'no database configured' "$body"; then
    has_db=0
fi

# check_db_only RUTA STATUS TIPO: igual que check, pero n/a si la aplicacion no tiene base de datos.
check_db_only() {
    if [ "$has_db" -eq 1 ]; then
        check "$@"
    else
        echo "n/a  $1 sin base de datos"
    fi
}

# check_published RUTA TIPO: lo publicado con `publish event`/`backtest` puede faltar en el paquete. Un 404
# cuyo cuerpo dice "not published" (JSON de la API) o "no publicada"/"no publicado" (pagina HTML) es
# "n/a sin publicar"; cualquier otra respuesta se comprueba como check RUTA 200 TIPO. El patron tambien
# casa con el JSON `scope not published`: es inofensivo.
check_published() {
    local path="$1" kind="$2" code
    code="$(curl -s --max-time 20 -o "$body" -w '%{http_code}' "$base$path" || true)"
    if [ "$code" = 404 ] && grep -qE 'not published|no publicad[ao]' "$body"; then
        echo "n/a  $path sin publicar"
    else
        check "$path" 200 "$kind"
    fi
}

# check_header RUTA CABECERA VALOR: (HEAD con -sI; la API responde a HEAD con las cabeceras del GET) la cabecera (sin distinguir mayusculas) debe existir y valer VALOR.
check_header() {
    local path="$1" name="$2" want="$3" got
    got="$(curl -sI --max-time 20 "$base$path" | tr -d '\r' | awk -v n="$(echo "$name" | tr 'A-Z' 'a-z')" \
        'tolower($1) == n ":" {print $2; exit}' || true)"
    if [ "$got" = "$want" ]; then
        echo "OK   $path ($name: $got)"
    else
        echo "FAIL $path: cabecera $name '${got:-ausente}' (esperado $want)"
        failures=$((failures + 1))
    fi
}

check / 200 html
check "/es?mode=nowcast" 200 html
check "/es/escanos?mode=forecast" 200 html
check "/es/escanos?region=abc" 400 html
check /promedio 301 none
check /es-zz 404 html
check_db_only "/casas?scope=es" 200 html
check "/casas?id=abc" 400 html
check "/elecciones?scope=es" 200 html
check "/elecciones?date=abc" 400 html
check_published "/elecciones?scope=es&date=2023-07-23" html
check "/metodo/backtest?scope=es" 200 html
check /metodo 200 html
check /metodo/promedio 200 html
check /metodo/escanos 200 html
check /metodo/rating 200 html
check /metodo/nope 404 html
check /modelo 301 none
check /nope 404 html
check_nginx_only /dist/vendor/echarts-5.6.0.min.js application/javascript
check_nginx_only /dist/vendor/fonts/SourceSerif4-4.005-Semibold.woff2 font/woff2
check_nginx_only /dist/css/site.css text/css
check_nginx_only /dist/img/logo.svg image/svg+xml
check_nginx_only /favicon.ico image/x-icon
check /api/v1/health 200 json
check /api/v1/manifest 200 json
# Congelacion editorial: si el manifest la declara activa, la cabecera y el banner deben verse.
frozen="$(curl -s --max-time 20 "$base/api/v1/manifest" | "$PY" -c \
    "import json,sys; print(str(bool(json.load(sys.stdin)['data'].get('freeze', {}).get('active'))).lower())" 2>/dev/null || true)"
if [ "$frozen" = true ]; then
    check_header /api/v1/scopes x-freeze active
    if curl -s --max-time 20 "$base/" | grep -q freeze-banner; then
        echo "OK   / (freeze-banner)"
    else
        echo "FAIL /: sin freeze-banner con la publicacion congelada"
        failures=$((failures + 1))
    fi
else
    echo "n/a  congelación: inactiva"
fi
check /api/v1/scopes 200 json
check /api/v1/forecast/es 200 json
check /api/v1/forecast/es/runs 200 json
check /api/v1/forecast/es/headline 200 json
check /api/v1/forecast/es/forecast/vote 200 json
check "/api/v1/forecast/es/forecast/summary?format=csv" 200 csv
check /api/v1/forecast/es-xx 400 none
check /api/v1/forecast/es/nope 404 none
check_db_only "/api/v1/ratings?scope=es" 200 json
check_db_only "/api/v1/polls?scope=es&limit=5" 200 json
check_db_only /api/v1/results/es/2023-07-23 200 json
check_db_only "/api/v1/drift?scope=es" 200 json
check /api/v1/analysis/es-xx/herding 400 none
check /api/v1/analysis/es/nope 404 none
check_db_only /api/v1/events/es 200 json
check_published /api/v1/events/es/2023-07-23/pollsters json
check_published /api/v1/events/es/2023-07-23/series json
check /api/v1/events/es/2023-13-01/meta 400 none
check_published /api/v1/backtest/es/by-horizon json
check /api/v1/backtest/es/nope 404 none

if [ "$failures" -gt 0 ]; then
    echo "$failures ruta(s) con error" >&2
    exit 1
fi
echo "todas las rutas OK"
