#!/usr/bin/env bash
# Prueba de humo de la imagen Docker: construye, arranca y comprueba nginx + API.
#
# Uso: bash deploy/smoke.sh [ENV_FILE]   (desde la raiz del repositorio)
#   ENV_FILE  fichero de entorno para --env-file; si se omite se usa el paquete
#             local (APP_API=api, DB_ADAPTER=PostgreSQL, S3_BUCKET vacio y
#             files/ montado en /app/files).
#
# Comprueba: imagen sin *.env, /healthz, rutas de la API (deploy/check-api.sh),
# paginas estaticas y cabeceras de cache. Avisa (WARN, sin fallar) si `docker
# logs` no tiene lineas de acceso de nginx o si `docker stop` tarda 10 s o mas.
# El contenedor se elimina siempre al terminar.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
image="elections-web:smoke"
name="elections-smoke"
base="http://127.0.0.1:8042"
env_file="${1:-}"

cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker build --platform linux/amd64 -t "$image" .

if [ -n "$env_file" ]; then
    env_args=(--env-file "$env_file")
else
    env_args=(-e APP_API=api -e DB_ADAPTER=PostgreSQL -e S3_BUCKET= -v "$PWD/files:/app/files")
fi
docker run -d --name "$name" --platform linux/amd64 -p 127.0.0.1:8042:8042 "${env_args[@]}" "$image" >/dev/null

echo "esperando a $base/healthz (hasta 90 s)..."
ready=0
for _ in $(seq 1 90); do
    if curl -fsS --max-time 20 "$base/healthz" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "FAIL: el contenedor no responde" >&2
    docker logs "$name" | tail -n 50 >&2
    exit 1
fi

envs="$(docker run --rm "$image" bash -c 'find /app -name "*.env" | wc -l' | tr -d '[:space:]')"
if [ "$envs" != "0" ]; then
    echo "FAIL: la imagen contiene $envs fichero(s) *.env" >&2
    exit 1
fi
echo "OK   imagen sin *.env"

bash deploy/check-api.sh "$base"

check_static() {
    local path="$1" want_type="$2" out
    out="$(curl -s --max-time 20 -o /dev/null -w '%{http_code} %{content_type}' "$base$path")"
    if [[ "$out" != 200\ $want_type* ]]; then
        echo "FAIL $path: $out (esperado 200 $want_type)" >&2
        exit 1
    fi
    echo "OK   $path ($out)"
}
check_static / text/html
check_static /promedio text/html
check_static /vendor/echarts-5.6.0.min.js application/javascript

check_header() {
    local path="$1" pattern="$2"
    if ! curl -sI --max-time 20 "$base$path" | grep -i '^cache-control:' | grep -qi -- "$pattern"; then
        echo "FAIL $path: Cache-Control sin '$pattern'" >&2
        exit 1
    fi
    echo "OK   $path Cache-Control contiene '$pattern'"
}
check_header /vendor/echarts-5.6.0.min.js 'max-age=31536000'
check_header /js/api.js 'no-cache'

# nginx escribe sus logs en /proc/1/fd/{1,2} (la salida del contenedor), porque
# supervisord se demoniza y su /dev/stdout es /dev/null.
# (Los logs se leen a una variable: con pipefail, `grep -q` cortaria la tuberia.)
sleep 1
logs="$(docker logs "$name" 2>&1 || true)"
if grep -qE '"(GET|HEAD) /[^ ]* HTTP/1\.1" [0-9]{3} ' <<<"$logs"; then
    echo "OK   docker logs contiene el log de acceso de nginx"
else
    echo "WARN docker logs sin lineas de acceso de nginx: revisar access_log /proc/1/fd/1 y" \
        "supervisord (nodaemon) en la imagen" >&2
fi

# init.sh deja bash como PID 1 esperando en `tail -f`, que ignora SIGTERM: docker
# stop espera los 10 s del SIGKILL. Una parada limpia exige `exec supervisord`
# en init.sh (decision de Luis), asi que esto solo avisa.
start="$(date +%s)"
docker stop "$name" >/dev/null
elapsed=$(( $(date +%s) - start ))
if [ "$elapsed" -ge 10 ]; then
    echo "WARN docker stop tardo ${elapsed} s (>= 10 s): init.sh ignora SIGTERM; hace falta" \
        "\`exec supervisord\` en init.sh para una parada limpia" >&2
else
    echo "OK   docker stop en ${elapsed} s"
fi
echo "smoke OK"
