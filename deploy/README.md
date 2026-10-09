# Despliegue de la web (fases 0-2 y páginas desde los controladores)

Guía breve para construir y probar la imagen `elections-web`. Sustituye `<tag>` por la etiqueta que uses.

## Fichero de configuración

- Toda la configuración (web, publicación y jobs) sale de `deploy/elections.env`: el usuario de PostgreSQL y
  las claves de AWS/S3 que ya usa el proyecto. Está ignorado por git y excluido de la imagen por
  `.dockerignore`; se inyecta en tiempo de ejecución con `--env-file`.
- Plantilla con todas las claves (valores vacíos): `.env.example` en la raíz del repositorio.

## Construir la imagen

Desde la raíz del repositorio:

```
docker build --platform linux/amd64 -t elections-web:<tag> .
```

La imagen copia el repositorio completo (`COPY . .`) salvo lo excluido en `.dockerignore`; `deploy/elections.env`,
`.git` y el material de claves nunca entran en ella. Servicios según variables de entorno: `APP_API` arranca
nginx + gunicorn, `APP_CRONTAB` arranca supercronic con `deploy/docker/crontab`, `APP_QUEUE` el worker;
`RUN_JOB` ejecuta un job y termina.

## Arrancar el contenedor

```
docker run -d --restart unless-stopped -p 127.0.0.1:8042:8042 --env-file deploy/elections.env elections-web:<tag>
```

## La web

Dentro del contenedor, nginx (puerto 8042) sirve solo los recursos y hace de proxy del resto:

- `/dist/` → ficheros de `web/dist/` (`css`, `js`, `vendor`, `img`), con cabeceras de seguridad.
  `/dist/vendor/` (ECharts, con la versión en el nombre) lleva caché de un año; el resto de `/dist/`
  (`.js`, `.css`) se revalida siempre. Un fichero que falta bajo `/dist/` da el 404 de nginx.
- `/` y `/api/` → gunicorn (`127.0.0.1:8000`) con `proxy_cache` de 60 s (zona `api_cache` en
  `/var/lib/nginx/api_cache`, cabecera `X-Cache`) y `limit_req` de 20 r/s por IP con ráfaga de 40
  (exceso: 429). Las páginas (`/`, `/promedio`) pasan por el mismo bloque que la API: también se
  cachean 60 s en nginx y cuentan para el mismo límite. La clave de caché es la de nginx por defecto
  (URI completa con la query string). Se probó una clave propia sobre `run`/`format` y se descartó: nginx
  y la API leen los nombres de los parámetros de forma distinta (codificación y mayúsculas), lo que
  permitía envenenar la caché. Los parámetros de más crean entradas, acotadas por `max_size=100m`,
  `inactive=10m` y `limit_req`.
- `/healthz` → `200 ok`, sin pasar por Python.

`nginx.conf` ya no tiene `root`, `index` ni `try_files`. Las páginas las renderiza Python: `Index` (`/`) y
`Promedio` (`/promedio`), controladores de `mtpy/controllers/` que heredan de `Page` y usan plantillas
Jinja2 de `web/templates/`, con los datos del paquete embebidos en la página (`initial-data`); el JS de
`web/dist/js/pages/` solo dibuja los gráficos. Una ruta desconocida fuera de `/api/` da un 404 HTML de
Python (dentro de `/api/`, el 404 JSON de la API).

Notas de operación:

- IP real: nginx toma la IP del cliente de `X-Forwarded-For` (módulo realip) cuando la petición llega
  de `127.0.0.1` o de `172.16.0.0/12` (docker-proxy). El proxy TLS de delante debe enviar
  `X-Forwarded-For`; si no, todas las peticiones parecen venir de la misma IP y el límite de 20 r/s se
  aplica a todo el sitio a la vez.
- Retirar un run (`unpublish` o `point`) tarda hasta unos 3 minutos en verse: TTL de los punteros en la
  API (60 s) + caché de nginx (60 s) + `max-age=60` en el navegador.
- Logs: `supervisord` se demoniza, así que dentro del contenedor `/dev/stdout` es `/dev/null`. nginx
  escribe en `/proc/1/fd/1` (acceso) y `/proc/1/fd/2` (errores), la salida de `init.sh` (PID 1), que es la
  del contenedor: deben verse con `docker logs` (la prueba de humo lo comprueba). Los logs de gunicorn
  siguen perdiéndose hasta que Luis adopte `nodaemon=true` / `exec supervisord` en `init.sh` (decisión
  suya).
- `docker stop` tarda 10 s: `init.sh` deja bash como PID 1 esperando en `tail -f`, que ignora SIGTERM,
  y Docker acaba con SIGKILL. Por la misma razón (`exec supervisord`), la prueba de humo solo lo avisa
  (WARN).

La API (`/api/v1`, ver `docs/web/contrato.md`) solo necesita S3 (el paquete `site/v1`): no usa la base
de datos hasta la fase 4. Variables opcionales de `deploy/elections.env`: `WEB_PREFIX` (prefijo del
paquete, por defecto `site/v1`) y `WEB_CACHE_TTL` (TTL de los punteros en segundos, por defecto 60).
Sin ningún run publicado, la API responde 503 `no bundle published yet` y las páginas muestran un 503
en HTML.

Tras arrancar el contenedor:

```
curl -s http://127.0.0.1:8042/healthz
bash deploy/check-api.sh http://127.0.0.1:8042
```

y abrir `http://127.0.0.1:8042/` y `http://127.0.0.1:8042/promedio`. `check-api.sh` decide una vez si hay
nginx a partir de `/healthz` (solo existe en nginx; sin nginx lo marca `n/a`) y comprueba `/` (200 HTML),
`/promedio?scope=es&mode=nowcast` (200 HTML), `/nope` (404 HTML), con nginx también
`/dist/vendor/echarts-5.6.0.min.js` (`application/javascript`) y `/dist/css/site.css` (`text/css`), y 10
rutas de la API (código, tipo y JSON válido); sale con error si alguna falla.

### Cómo añadir una página

1. Controlador `mtpy/controllers/<Nombre>.py` que herede de `Page`, con `active` (la entrada del menú) y
   un `index_action` que llame a `render(plantilla, **contexto)`; ver `Promedio.py`.
2. Plantilla en `web/templates/` que extienda `base.html`.
3. Entrada en `pages.ROUTES` (ruta, controlador, acción, métodos) y en `pages.PAGES` (menú), en
   `mtpy/lib/pages.py`.
4. El constructor del contexto de la página, también en `pages.py` (junto a `index_context` y
   `promedio_context`).
5. Un test en `tests/test_pages.py`.

Si la página dibuja gráficos, el módulo `web/dist/js/pages/<nombre>.js` lee `initial-data` con
`readInitial` y no hace `fetch`.

### Desarrollo sin Docker

En el Mac, Luis tiene un nginx local en el puerto 8080. Hay que añadir a su `server` las dos `location`
de `/dist/` y el proxy, con `<repo>` la ruta del repositorio:

```
location ^~ /dist/vendor/ { alias <repo>/web/dist/vendor/; expires 1y; }
location ^~ /dist/ { alias <repo>/web/dist/; expires -1; }
location / {
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_pass http://127.0.0.1:8000;
}
```

y arrancar la aplicación:

```
S3_BUCKET= python -m uvicorn api:api --port 8000
```

Con `S3_BUCKET=` vacío la API lee el paquete local `files/site/v1`. Después se abre `http://localhost:8080/`.
Sin nginx, `http://127.0.0.1:8000/` y `/promedio` se renderizan, pero los recursos de `/dist/` dan 404:
Python solo sirve las páginas y la API.

### Comprobaciones de la infraestructura

- `bash deploy/nginx-check.sh`: `nginx -t` sobre `deploy/docker/nginx.conf` con los dos `alias` de
  `/dist/` reescritos a `<repo>/web/dist/...` (no arranca nginx). Requiere nginx instalado.
- `bash deploy/smoke.sh [deploy/elections.env]`: construye la imagen, la arranca, comprueba que no hay
  `*.env` en ella, ejecuta `check-api.sh`, las cabeceras de caché de `/dist/vendor/`
  (`max-age=31536000`), `/dist/js/pages/index.js` (`no-cache`) y `/` (`max-age=60`), que `/promedio`
  devuelve HTML y que `/` contiene `initial-data`; avisa (WARN, sin fallar) si `docker logs` no tiene
  líneas de acceso de nginx o si `docker stop` tarda 10 s o más. Sin argumento usa el
  paquete local (`files/` montado, `S3_BUCKET` vacío); con `deploy/elections.env`, S3 y la base reales.
  Requiere Docker en marcha.

Estado: `nginx.conf` se verificó el 2026-10-09 con un nginx 1.29.5 local (Homebrew) en el puerto 8043
delante de uvicorn en el 8000: `/` primero `MISS` y luego `HIT`, cabeceras como las esperadas y
`check-api.sh` todo OK. La prueba de humo con Docker está sin ejecutar.

Pendiente de Luis: ejecutar `bash deploy/smoke.sh` con Docker (paquete local y `deploy/elections.env`);
añadir las `location` de `/dist/` y el proxy a su nginx local y recorrer `/` y `/promedio` con y sin JS
(el botón "Ver" debe funcionar sin JS); primera publicación a S3 y despliegue.

## Comprobar S3

Este job escribe, lee, lista y borra un fichero en `site/v1/_smoke` para detectar pronto si `s3fs==0.4.2` sigue funcionando con el bucket.

Desde el portátil:

```
set -a; . deploy/elections.env; set +a; python job.py check_s3
```

`load_dotenv` no pisa las variables ya exportadas, así que el `.env` del repositorio no interfiere.
El job se niega a probar un `files/` local (sería un falso visto bueno); para forzarlo:
`python job.py check_s3 '{"allow_local": true}'`.

Desde la imagen:

```
docker run --rm --env-file deploy/elections.env -e RUN_JOB=check_s3 elections-web:<tag>
```

Cada paso imprime `<paso> OK` o `<paso> FAIL: <error>`; si alguno falla, el job termina con error.

## Si `check_s3` falla

1. Subir `s3fs` a una versión de 2024 o posterior (con un `aiobotocore` compatible) y repetir.
2. Si sigue fallando, la fase 1 reimplementa `read_bytes`, `write_bytes`, `exists`, `listdir` y `remove` de `mtpy/core/services/s3.py` sobre `boto3`.

El primer `FAIL` es la causa raíz (los pasos posteriores fallan en cascada). Un `AccessDenied`, o un `gone FAIL` tras `remove OK`, indica que al usuario de AWS le falta `s3:DeleteObject` sobre el bucket, no un problema de `s3fs`.

## Publicar

`python job.py publish` ejecuta el modelo, escribe un run inmutable en el paquete `site/v1` de `app.fs` y
mueve los punteros (`history.json` por ámbito y `manifest.json`). Contrato de los ficheros:
`docs/web/contrato.md`.

```
python job.py publish '{"what":["forecast"],"scopes":["es"]}'
python job.py publish '{"what":["forecast"],"scopes":"all"}'
python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
python job.py publish '{"what":["unpublish"],"scopes":["es"],"run":"20261007-141503"}'
python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
```

1. Publica el pronóstico de `es` (nowcast y forecast, 1000 simulaciones) y actualiza su entrada del manifest.
2. Lo mismo para `es` y todos los ámbitos autonómicos; los que no tienen sondeos quedan `skipped`.
3. Solo reescribe el manifest; con `freeze` activa o desactiva el aviso editorial de congelación.
4. Apunta el manifest de `es` a un run anterior (no borra nada).
5. Borra un run malo, reconstruye `history.json` y, si era el último, apunta al anterior (o quita el ámbito).
6. Ensayo: escribe en `site-dry/v1` con 20 simulaciones, sin `history.json` ni manifest.

`what` admite `forecast`, `manifest`, `point` y `unpublish` (se ejecutan en ese orden); `analysis`, `event`
y `backtest` fallan con `ValueError` hasta las fases 4 y 5.

### Parámetros

| Parámetro | Por defecto | Significado |
| --- | --- | --- |
| `what` | `["forecast"]` | Acciones a ejecutar |
| `scopes` | `["es"]` | Ámbitos, o `"all"` (`es` más los ámbitos con padre) |
| `event_date` | siguiente evento del ámbito | Fecha de la elección (`YYYY-MM-DD`; si no es una fecha ISO, el ámbito queda `failed`) |
| `n_sim` | `1000` | Simulaciones por modo |
| `seed` | `42` | Semilla |
| `drange` | `6` | Ventana de sondeos del pronóstico |
| `max_fc` | `10` | Pasos máximos del pronóstico |
| `alpha` | `0.05` | Nivel de los intervalos |
| `correctors` | `null` | Cambios sobre los correctores por defecto (`house_effects`, `dispersion`, `regional_noise` activos) |
| `freeze` | `null` | Aviso del manifest: `{"active": bool, "message": str}` |
| `run` | `null` | `run_id` para `point` y `unpublish` (obligatorio en ambos) |
| `dry_run` | `false` | Escribir en `site-dry/v1` sin punteros |
| `force` | `false` | Publicar `es` dentro de la ventana LOREG |
| `today` | hoy (Europe/Madrid) | Solo para pruebas y ensayos (fija la fecha de la guarda LOREG, `YYYY-MM-DD`; a diferencia de `force`, no deja huella en `meta.freeze`) |

### Estados y resumen

Cada ámbito termina en `published`, `skipped` (sin evento próximo, sin sondeos o promedio no ajustable),
`refused` (guarda LOREG) o `failed` (cualquier otro error, incluido un fallo de validación del paquete; el
resto de ámbitos continúa). El job imprime, escribe en
`app.logger` y avisa con `alert()` un resumen de una línea por ámbito:

```
es: published 20261008-181100 (as_of 2026-10-13, 315 polls) in 121 s
es-md: skipped (<motivo>)
es: refused (es 2026-11-29: inside the LOREG window (art. 69.7); pass force=true to publish)
es: failed (<Excepción>: <mensaje>)
es: published 20261007-141503          # point y unpublish
manifest: 1 scopes, freeze off
manifest: not written (dry run)
```

Si algún ámbito falla, el job termina con error (código de salida 1) después de escribir el manifest y
el resumen; un ámbito fallido conserva su entrada anterior en el manifest.

### Publicar en local

Con `S3_BUCKET` vacío, `mtpy.run()` construye el sistema de ficheros local y el paquete queda en
`files/site/v1`. Si el `.env` de la raíz define `S3_BUCKET`, hay que anteponer `S3_BUCKET=` (vacío) al
comando, porque `load_dotenv` no pisa las variables ya exportadas y, sin ello, se publicaría en el bucket:

```
S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"]}'
```

El paquete local lee la base de datos configurada en el `.env` de la raíz.

### Publicar en S3 contra la RDS

Desde el portátil:

```
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["forecast"],"scopes":["es"]}'
```

Desde la imagen (JSON sin espacios: `init.sh` no entrecomilla):

```
docker run --rm --env-file deploy/elections.env -e RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}' elections-web:<tag>
```

Antes de la primera publicación a S3, `check_s3` debe dar `OK` en todos los pasos.

### Leer el manifest publicado

Sin la CLI de AWS:

```
set -a; . deploy/elections.env; set +a
python -c "from mtpy import mtpy; app = mtpy.run(); print(app.fs.read_bytes('site/v1/manifest.json').decode())"
```

Cada run guarda en su `meta.json` los `seconds` por paso (`init`, `fit`, `nowcast`, `forecast`, `export`,
`total`), a 1 decimal; `nowcast` y `forecast` miden solo las simulaciones y `export` todas las escrituras.

### Retirada de un run

- Volver a un run anterior sin borrar nada: `{"what":["point"],"scopes":["es"],"run":"<run_id>"}`.
- Quitar un run malo: `{"what":["unpublish"],"scopes":["es"],"run":"<run_id>"}`. Borra su carpeta,
  reconstruye `history.json` y repunta el manifest al run anterior si apuntaba a él.
- `point` y `unpublish` se ven en la web en 2 minutos como máximo: 60 s de caché de la API (por proceso) más
  60 s de la de nginx. Los runs pedidos con `?run=` son inmutables en el navegador (un año).

### Guarda LOREG

Con `scope` `es`, el job se niega a publicar (`refused`) si la elección cae en los 5 días previos o en el
propio día (LOREG art. 69.7). `{"force":true}` publica igualmente y marca el run con `meta.freeze: true`.
La fecha de hoy es la de Europe/Madrid; `today` la fija solo para pruebas y ensayos. `manifest.freeze` es otra cosa: el interruptor editorial que se fija con
`what: ["manifest"]`.

### Duraciones medidas

Primer paquete local de `es` (2026-10-08, contra la RDS, `n_sim=1000`): 121 s en total (init 35,5; ajuste
42,5; nowcast 21,0; forecast 21,5; exportación 0,8). Ensayo con `n_sim=20`: 86 s. La duración de
`scopes: "all"` contra la RDS está pendiente de medir.
