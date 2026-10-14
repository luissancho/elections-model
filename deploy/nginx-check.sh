#!/usr/bin/env bash
# Comprueba la sintaxis de deploy/docker/nginx.conf con el nginx local
# (macOS Homebrew o Linux), sin arrancar nginx ni abrir puertos.
#
# Uso: bash deploy/nginx-check.sh
#
# Copia la configuracion a un directorio temporal adaptando las rutas de la
# imagen Docker (mime.types, modulos, pid, logs, cache y alias de /dist/) a las
# locales, y ejecuta `nginx -t`. Sale con el codigo de `nginx -t`.
#
# De la cache solo se reescribe el directorio padre (/var/lib/nginx -> $tmp/lib,
# que se crea); la hoja api_cache NO se crea, para que `nginx -t` haga el mismo
# mkdir que hara la imagen. Los logs van a ficheros del directorio temporal
# (macOS no tiene /proc).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conf="$repo/deploy/docker/nginx.conf"

command -v nginx >/dev/null || { echo "nginx no esta instalado" >&2; exit 1; }

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
trap 'rm -rf "$tmp"' EXIT
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
    "$conf" > "$tmp/nginx.conf"

nginx -t -c "$tmp/nginx.conf" -p "$tmp/"
