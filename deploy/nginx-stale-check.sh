#!/usr/bin/env bash
# Comprueba con un nginx de verdad que /elecciones se sirve desde la copia caducada mientras se
# refresca por detras (proxy_cache_background_update) y que el resto de rutas siguen esperando al
# refresco, como siempre.
#
# Uso: bash deploy/nginx-stale-check.sh [nginx.conf]
#
# Adapta la configuracion como deploy/nginx-check.sh (rutas de la imagen Docker a las locales) y la
# arranca en puertos propios: nginx en el 18042 y, en el 18000, un servidor de pega que tarda 2 s
# en responder cualquier ruta, con `Cache-Control: public, max-age=5` y, en el cuerpo, cuantas
# veces le han pedido esa ruta. Tarda unos 27 s y necesita curl 7.84 o posterior (`%header{}`).
# Imprime OK o FAIL por paso y sale con 1 si alguno falla:
#
#   1. /elecciones?scope=es-md sin copia: MISS, espera al servidor (>= 2 s), cuerpo 1.
#   2. La misma, con la copia caducada: STALE, al instante (< 0,5 s), cuerpo 1.
#   3. /healthz justo detras, en el mismo `curl` que el paso 2: al instante. nginx retiene la
#      conexion hasta que acaba el refresco, asi que el bloque de /elecciones la cierra
#      (`keepalive_timeout 0`) y la peticion siguiente abre otra en vez de hacer cola.
#   4. La misma del paso 1, tres segundos despues: HIT, cuerpo 2 (el refresco por detras llego).
#   5. /elecciones/es/generales/2023-07 (el detalle va por `location /`): MISS y, caducada, EXPIRED
#      esperando al servidor (>= 2 s).
set -euo pipefail
export LC_ALL=C

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conf="${1:-$repo/deploy/docker/nginx.conf}"
port=18042
upstream=18000
base="http://127.0.0.1:$port"

for tool in nginx curl python3; do
    command -v "$tool" >/dev/null || { echo "$tool no esta instalado" >&2; exit 1; }
done

# Directorio de configuracion del nginx instalado (donde vive mime.types).
conf_path="$(nginx -V 2>&1 | grep -o -- '--conf-path=[^ ]*' | cut -d= -f2 || true)"
conf_dir="$(dirname "${conf_path:-/etc/nginx/nginx.conf}")"
if [ ! -f "$conf_dir/mime.types" ]; then
    for d in /opt/homebrew/etc/nginx /etc/nginx; do
        [ -f "$d/mime.types" ] && conf_dir="$d" && break
    done
fi
[ -f "$conf_dir/mime.types" ] || { echo "no se encuentra mime.types" >&2; exit 1; }

tmp="$(mktemp -d)"
nginx_pid=""
upstream_pid=""
cleanup() {
    [ -n "$nginx_pid" ] && kill "$nginx_pid" 2>/dev/null || true
    [ -n "$upstream_pid" ] && kill "$upstream_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/logs" "$tmp/lib"

sed \
    -e "s#/etc/nginx/mime.types#$conf_dir/mime.types#" \
    -e '/include \/etc\/nginx\/modules-enabled\/\*\.conf;/d' \
    -e "s#pid /run/nginx.pid;#pid $tmp/nginx.pid;#" \
    -e "s#/var/lib/nginx/#$tmp/lib/#" \
    -e "s#/proc/1/fd/2#$tmp/logs/error.log#" \
    -e "s#/proc/1/fd/1#$tmp/logs/access.log#" \
    -e "s#alias /app/web/dist/vendor/;#alias $repo/web/dist/vendor/;#" \
    -e "s#alias /app/web/dist/;#alias $repo/web/dist/;#" \
    -e "s#listen 8042;#listen $port;#" \
    -e "s#server 127.0.0.1:8000 #server 127.0.0.1:$upstream #" \
    "$conf" > "$tmp/nginx.conf"

cat > "$tmp/upstream.py" <<'PY'
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

counts, lock = {}, threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with lock:
            counts[self.path] = counts.get(self.path, 0) + 1
            body = str(counts[self.path]).encode()
        time.sleep(2)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'public, max-age=5')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
PY

# Como root (en un contenedor) los workers de nginx no son root: necesitan entrar en el temporal.
chmod -R a+rwX "$tmp"

python3 "$tmp/upstream.py" "$upstream" &
upstream_pid=$!
nginx -c "$tmp/nginx.conf" -p "$tmp/" &
nginx_pid=$!

ready=""
for _ in $(seq 1 50); do
    if curl -s -o /dev/null --max-time 1 "$base/healthz" \
        && curl -s -o /dev/null --max-time 5 "http://127.0.0.1:$upstream/ready"; then
        ready=yes
        break
    fi
    sleep 0.2
done
[ -n "$ready" ] || { echo "nginx o el servidor de pega no arrancan" >&2; cat "$tmp/logs/error.log" >&2 || true; exit 1; }

failed=0
# Lo que se pide de cada respuesta: estado, segundos, X-Cache y conexiones nuevas abiertas.
format='%{http_code} %{time_total} %header{x-cache} %{num_connects}\n'

lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }

# report PASO DESCRIPCION CONDICION...: OK si todas las condiciones (ya evaluadas a 0/1) valen 1.
report() {
    local step="$1" text="$2" ok=1
    shift 2
    for value in "$@"; do
        [ "$value" = 1 ] || ok=0
    done
    if [ "$ok" = 1 ]; then
        echo "OK   $step. $text"
    else
        echo "FAIL $step. $text" >&2
        failed=1
    fi
}

is() { [ "$1" = "$2" ] && echo 1 || echo 0; }
under() { lt "$1" "$2" && echo 1 || echo 0; }
over() { lt "$1" "$2" && echo 0 || echo 1; }

history="$base/elecciones?scope=es-md"
detail="$base/elecciones/es/generales/2023-07"

# 1. Sin copia: espera al servidor.
read -r status seconds cache _ <<< "$(curl -s -o "$tmp/body" -w "$format" "$history")"
report 1 "/elecciones sin copia: $cache en $seconds s, cuerpo $(cat "$tmp/body")" \
    "$(is "$status" 200)" "$(is "$cache" MISS)" "$(over "$seconds" 2)" "$(is "$(cat "$tmp/body")" 1)"

# 2 y 3. Copia caducada: la sirve al instante y la peticion siguiente no hace cola. nginx cuenta la
# vigencia en segundos enteros, asi que una copia con max-age=5 vale entre 5 y 6 s: se espera 6,5.
sleep 6.5
curl -s -o "$tmp/body" -w "$format" "$history" -o /dev/null -w "$format" "$base/healthz" > "$tmp/pair"
read -r status seconds cache _ < "$tmp/pair"
report 2 "/elecciones con la copia caducada: $cache en $seconds s, cuerpo $(cat "$tmp/body")" \
    "$(is "$status" 200)" "$(is "$cache" STALE)" "$(under "$seconds" 0.5)" "$(is "$(cat "$tmp/body")" 1)"
read -r status seconds connects <<< "$(sed -n 2p "$tmp/pair" | awk '{print $1, $2, $NF}')"
report 3 "/healthz justo detras: $seconds s, $connects conexiones nuevas" \
    "$(is "$status" 200)" "$(under "$seconds" 0.5)"

# 4. El refresco por detras ha dejado la copia nueva.
sleep 3
read -r status seconds cache _ <<< "$(curl -s -o "$tmp/body" -w "$format" "$history")"
report 4 "/elecciones tras el refresco por detras: $cache en $seconds s, cuerpo $(cat "$tmp/body")" \
    "$(is "$status" 200)" "$(is "$cache" HIT)" "$(under "$seconds" 0.5)" "$(is "$(cat "$tmp/body")" 2)"

# 5. El detalle de una eleccion va por `location /`: caducada, espera al servidor.
read -r status seconds cache _ <<< "$(curl -s -o "$tmp/body" -w "$format" "$detail")"
first="$(is "$cache" MISS)"
sleep 6.5
read -r status seconds cache _ <<< "$(curl -s -o "$tmp/body" -w "$format" "$detail")"
report 5 "el detalle de una eleccion con la copia caducada: $cache en $seconds s, cuerpo $(cat "$tmp/body")" \
    "$first" "$(is "$status" 200)" "$(is "$cache" EXPIRED)" "$(over "$seconds" 2)" "$(is "$(cat "$tmp/body")" 2)"

exit "$failed"
