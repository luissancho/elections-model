#!/usr/bin/env bash
# Prueba de humo de la imagen Docker: construye, arranca y comprueba nginx + API.
#
# Uso: bash deploy/smoke.sh [ENV_FILE]   (desde la raiz del repositorio)
#   ENV_FILE  fichero de entorno para --env-file; si se omite se usa el paquete
#             local (APP_API=api, DB_ADAPTER=PostgreSQL, S3_BUCKET vacio y
#             files/ montado en /app/files).
#
# Comprueba: imagen sin *.env, /healthz, rutas de la API (deploy/check-api.sh),
# paginas estaticas y que `docker stop` tarde menos de 10 s. El contenedor se
# elimina siempre al terminar.
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
    if curl -fsS "$base/healthz" >/dev/null 2>&1; then ready=1; break; fi
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
    out="$(curl -s -o /dev/null -w '%{http_code} %{content_type}' "$base$path")"
    if [[ "$out" != 200\ $want_type* ]]; then
        echo "FAIL $path: $out (esperado 200 $want_type)" >&2
        exit 1
    fi
    echo "OK   $path ($out)"
}
check_static / text/html
check_static /promedio text/html
check_static /vendor/echarts-5.6.0.min.js application/javascript

start="$(date +%s)"
docker stop "$name" >/dev/null
elapsed=$(( $(date +%s) - start ))
if [ "$elapsed" -ge 10 ]; then
    echo "FAIL: docker stop tardo ${elapsed} s (>= 10 s)" >&2
    exit 1
fi
echo "OK   docker stop en ${elapsed} s"
echo "smoke OK"
