# Despliegue de la web (fases 0-2 y páginas desde los controladores)

Guía breve para construir y probar la imagen `elections-model`. Sustituye `<tag>` por la etiqueta que uses.
(Enmienda 2026-10-13, fase 6: la guía cubre ya las fases 0 a 6, incluido el despliegue en AWS ECS; el título se
conserva. El estado de cada fase está en `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`.)

Índice:

1. "Fichero de configuración", "Construir la imagen" y "Arrancar el contenedor" (Docker en local).
2. "La web": nginx, páginas, API y base de datos; "Cómo añadir una página", "Desarrollo sin Docker" y
   "Comprobaciones de la infraestructura" (`nginx-check.sh`, `smoke.sh`, `check-api.sh`).
3. "Despliegue en ECS": qué debe existir en AWS, marcadores de las plantillas de `deploy/ecs/` y procedimiento.
4. "Comprobar S3" y "Si `check_s3` falla".
5. "Publicar": qué hace cada acción y cuándo lanzarla (tabla y rutina por situación), ejemplos, parámetros,
   estados, relleno retroactivo, análisis de casas, elecciones pasadas, backtest, publicar en local, en S3 y desde
   ECS, "Veda" (runbook del 23 de noviembre y ensayo), leer el manifest, retirada de un run y "Guarda LOREG".
6. "Ensayos de operación": carga, `point`/`unpublish`, congelación y copia de la base.
7. "Duraciones medidas": publicación, carga local, arranque de un worker y memoria del job.

## Fichero de configuración

- Toda la configuración (web, publicación y jobs) sale de `deploy/elections.env`: el usuario de PostgreSQL y
  las claves de AWS/S3 que ya usa el proyecto. Está ignorado por git y excluido de la imagen por
  `.dockerignore`; se inyecta en tiempo de ejecución con `--env-file`.
- Plantilla con todas las claves (valores vacíos): `.env.example` en la raíz del repositorio.
- `GUNICORN_WORKERS` fija los workers de gunicorn (4 si se deja vacío; `init.sh` lo exporta antes de arrancar
  supervisord). En ECS, `AWS_KEY` y `AWS_SECRET` se dejan vacíos: `S3` pasa entonces `None` a `s3fs` y se usa la
  cadena de credenciales de botocore (el rol de la tarea). `deploy/elections.env` sigue siendo el entorno del
  portátil (y de `docker run` en local), no el de la web en ECS.

## Construir la imagen

Desde la raíz del repositorio:

```
docker build --platform linux/amd64 -t elections-model:<tag> .
```

La imagen copia el repositorio completo (`COPY . .`) salvo lo excluido en `.dockerignore`; `deploy/elections.env`,
`.git` y el material de claves nunca entran en ella. Servicios según variables de entorno: `APP_API` arranca
nginx + gunicorn, `APP_CRONTAB` arranca supercronic con `deploy/docker/crontab`, `APP_QUEUE` el worker;
`RUN_JOB` ejecuta un job y termina.

## Arrancar el contenedor

```
docker run -d --restart unless-stopped -p 127.0.0.1:8042:8042 --env-file deploy/elections.env elections-model:<tag>
```

## La web

Dentro del contenedor, nginx (puerto 8042) sirve solo los recursos y hace de proxy del resto:

- `/dist/` → ficheros de `web/dist/` (`css`, `js`, `vendor`, `img`), con cabeceras de seguridad.
  `/dist/vendor/` (ECharts, con la versión en el nombre, y desde la fase 7 las fuentes de `/dist/vendor/fonts/`)
  lleva caché de un año; el resto de `/dist/` (`.js`, `.css`) se revalida siempre. Un fichero que falta bajo `/dist/` da el 404 de nginx.
- `/` y `/api/` → gunicorn (`127.0.0.1:8000`) con `proxy_cache` de 60 s (zona `api_cache` en
  `/var/lib/nginx/api_cache`, cabecera `X-Cache`) y `limit_req` de 20 r/s por IP con ráfaga de 40
  (exceso: 429). Las páginas (`/`, `/{scope}`, `/{scope}/escanos`, `/elecciones`, `/casas`, `/metodo` y
  `/metodo/*`, más las redirecciones de las URL antiguas) pasan por el mismo bloque que la API: también se cachean 60 s en nginx y cuentan para el mismo límite. La clave de caché es la de nginx por defecto
  (URI completa con la query string). Se probó una clave propia sobre `run`/`format` y se descartó: nginx
  y la API leen los nombres de los parámetros de forma distinta (codificación y mayúsculas), lo que
  permitía envenenar la caché. Los parámetros de más crean entradas, acotadas por `max_size=100m`,
  `inactive=10m` y `limit_req`.
- `/healthz` → `200 ok`, sin pasar por Python. (Enmienda 2026-10-13, fase 6: también devuelve la cabecera
  `X-Client-IP` con la IP que nginx atribuye al cliente; `check-api.sh` la imprime y avisa si es privada.)

`nginx.conf` ya no tiene `root`, `index` ni `try_files`. Las páginas las renderiza Python: controladores de
`mtpy/controllers/` que heredan de `Page` y usan plantillas Jinja2 de `web/templates/`, con los datos del paquete
embebidos en la página (`initial-data`); el JS de `web/dist/js/pages/` solo dibuja los gráficos y no hace `fetch`.
Las rutas están en `pages.ROUTES` (`mtpy/lib/pages/common.py`), salvo `/`, que es el valor por defecto del router
(controlador `index`): `Router.handle` devuelve la primera que casa, así que las fijas van antes que `/{scope}`.

Mapa del sitio (enmienda 2026-10-13, fase 7; spec `docs/superpowers/specs/2026-10-13-web-fase-7-diseno-design.md`). Dos
familias de páginas:

- **De evento** (la próxima elección de un ámbito), bajo el código del ámbito del catálogo: `/{scope}` (Promedios,
  controlador `Promedios`, `promedios.html`) y `/{scope}/escanos` (Escaños, `Escanos`, `escanos.html`, con
  `?region=`). Aceptan `?mode=` (`nowcast` o `forecast`) y `?run=`, pero desde la fase 7b ninguna página pinta un
  selector de modo: sin `?mode=`, `pages.default_mode(frozen)` muestra Hoy (`nowcast`) y, mientras la publicación
  está congelada (veda), Elección (`forecast`); `?mode=` queda para enlaces y pruebas. Un código que no es del
  catálogo da 404 "Ámbito desconocido."; uno del catálogo sin pronóstico, 404 "Este ámbito no tiene pronóstico
  publicado".
- **Globales**, en el menú principal (`pages.PAGES`: Inicio, Histórico Errores, Ranking Encuestadoras, Método):
  `/` (Inicio, `Index`, `index.html`: los pronósticos publicados, uno por ámbito, y las elecciones pasadas con
  detalle), `/elecciones` (Histórico Errores, `Elecciones`; con `?date=YYYY-MM-DD`, el detalle de una elección),
  `/casas` (Ranking Encuestadoras, `Casas`; con `?id=`, el perfil de una encuestadora) y `/metodo` (`Metodo`, `metodo.html`:
  introducción, índice de las cuatro preguntas, datos, publicación y limitaciones) con sus subpáginas
  `/metodo/promedio`, `/metodo/escanos`, `/metodo/rating` y `/metodo/backtest` (la precisión del modelo, antes
  `/modelo`). Donde el contenido depende del ámbito llevan `?scope=` (`es` por defecto).

Redirecciones 301 (`Page.redirect`, desde `Redirect.py` y, para `/?scope=`, desde `Index`; `cache-control: no-store`, sin cuerpo, conservando el resto de la query):
`/?scope=X` → `/X` (`/` sin `scope` es el Inicio); `/promedio` → `/X` (`/es` sin `scope`); `/escanos` → `/X/escanos`
(con `region`); `/modelo` → `/metodo/backtest` (con `scope`, sin `mode` ni `run`). nginx no cachea los 301
(`proxy_cache_valid 200 60s`).

Cabecera, barra de evento y filtro: la cabecera (`_header.html`) lleva el nombre del sitio y el menú, sin selectores;
el banner de veda va encima. Las páginas de evento añaden la barra de evento (`_event_bar.html`, contexto
`event_bar` de `common_context`): nombre del evento ("Elecciones generales" o "Elecciones autonómicas · {nombre}"),
fecha y días que faltan (Europe/Madrid), las pestañas Promedios | Escaños (`pages.EVENT_TABS`), la línea que
explica el modo mostrado (`pages.mode_text`) y, con un run fijado, el chip del run. Desde la fase 7b la barra no
lleva formulario: ni control Hoy / Elección ni el `<details>` que lo explicaba. El título dice "Estimación a {hoy}"
(Europe/Madrid) en modo Hoy con el último run y "Estimación del {run_at}" con un run fijado anterior; en modo
Elección, "Pronóstico para el {fecha}", "Pronóstico definitivo para el {fecha}" (congelado, último run) o "Pronóstico
publicado el {run_at}" (run fijado anterior) (`pages.title_of`). `/elecciones`, `/casas` y `/metodo/backtest` llevan el filtro de ámbito
(`_scope_filter.html`: desplegable de los ámbitos publicados, GET + "Ver" sin JS, envío al cambiar con JS); el resto
de `/metodo` cita las cifras de las generales y no lo lleva. Las subpáginas del método comparten la subnavegación
`_metodo_nav.html`.

Patrón de tarjeta (`web/templates/_macros.html`, `{% import '_macros.html' as m %}`): toda sección es
`m.card_start(id, título, descripción, subtitle=None, classes='', level=2)` … `m.card_end()`, con la descripción
siempre visible; dentro, `{% call m.how() %}…{% endcall %}` es el "Cómo leerlo" (`<details>`, siempre plegado
desde la fase 7b; tres partes: qué es cada elemento, cómo leerlo y qué no dice, ≤ 120 palabras). Cada página abre con `m.headline(texto)` (una línea con cifras vivas) y las pestañas cierran con
`m.closing(pregunta, method_href(...), adelanto, enlaces)`, la tarjeta de enlace a su texto del método.

Componentes de la fase 7b (ajustes tras el recorrido de Luis, 2026-10-13):

- Terminología pública: "encuestadora/s" (nunca "casa/s"; "efectos de encuestadora") y "medio/s" (nunca
  "patrocinador/es"); siempre las siglas (`name`), con el nombre completo solo en `title=`. Las barras sin color de
  partido usan el azul neutro de `THEME.barColor`.
- Inicio: cada tarjeta es entera un enlace (`<a class="card card-link">`, con su `<h3>` dentro); la del primer
  ámbito (`.tile-hero`) es la grande y "Ver escaños" queda fuera del enlace, debajo. "Para entender el modelo" son
  tres tarjetas clicables: Histórico Errores, Ranking Encuestadoras y Método.
- Píldoras (`fieldset.pills` > `label.pill` con su casilla oculta pero enfocable): el selector de partidos o
  bloques, coloreado al estilo de la leyenda de ECharts, sin título ni botones Todos/Ninguno. La píldora marcada
  toma el color del partido (`--pill-color`) y una tinta legible calculada en el servidor (filtro `ink`,
  `pages.ink_color`: texto oscuro sobre colores claros, blanco sobre oscuros); sin `:has()` el JS pone la clase
  `is-on`. Se usan en Promedios y en la calculadora de coaliciones de Escaños; las leyendas de ECharts de Promedios
  se ocultan.
- Promedios: los controles en tres líneas, en este orden: Vista (Partidos/Bloques), Ventana y las píldoras. El
  tooltip es por sondeo (`trigger: 'item'`): nombre, fecha, muestra y las siglas en su color con su valor, de mayor a
  menor. La tabla "Sondeos del ciclo" tiene la columna "Sondeo" (`pages.poll_name`: `encuestadora / medio`, o la
  encuestadora sola, como el cargador) en lugar de Casa y Patrocinador; se pintan las primeras `TABLE_ROWS` (40)
  filas, la isla lleva todas y "Cargar más" añade las siguientes hasta que no quedan (entonces desaparece; con 40
  sondeos o menos nunca aparece). Sin JS se ven las 40 primeras y el CSV tiene todas.
- Escaños: tabla Partido, Escaños, Rango 95 %, P(escaño), P(primero), P(mayoría); los histogramas llevan la línea
  de la mediana (gruesa) y las del rango del 95 % (finas), sin el valor escrito; una sola tabla de bloques
  ("Bloques ideológicos, con todos los partidos"); la calculadora usa las mismas píldoras.
- Histórico Errores: la diana (`#errors-bullseye`) es cuadrada, de hasta 640 px y centrada en su tarjeta; la del
  detalle de una elección y la del perfil (`#bullseye-chart`) siguen en 480 px.

Glosario: `web/templates/_glosario.html` define una sola vez Error medio, Error de bloques (magnitud), Sesgo del
margen (con signo) y Herding; se incluye al pie de las tarjetas con esas columnas en el histórico, el detalle de una
elección, el ranking y el perfil de encuestadora, con `with_herding` (y `with_block_error`, por defecto verdadero) fijados
antes del `include`. Toda columna de error usa su nombre del glosario.

Fuentes: Source Serif 4 Semibold (títulos) y Source Sans 3 Regular (texto) en woff2, en `/dist/vendor/fonts/` con su
licencia (`LICENSE`, SIL OFL); 188 KB entre las dos, servidas con la caché de un año de `/dist/vendor/` y
`font-display: swap`, con la pila del sistema como respaldo. Para cambiar una fuente, cambia también el nombre del
fichero (la caché es de un año). Los tokens de color y tipografía están en `:root` de `web/dist/css/site.css`.

Datos y errores: `/elecciones` y `/metodo/backtest` leen solo el paquete (sin base): sin elecciones o sin backtest
publicados del ámbito responden 200 con un aviso; `/elecciones?date=` mal formada da 400 ("Fecha no válida.") y una
elección no publicada, 404 ("Elección no publicada."); una subpágina del método desconocida, 404 ("Página del método
no encontrada."). Una ruta desconocida fuera de `/api/` da un 404 HTML de Python (dentro de `/api/`, el 404 JSON de la
API). Toda página necesita un pronóstico publicado del ámbito elegido (la cabecera común lee su `meta`): sin ningún
paquete, 503.

Notas de operación:

- IP real: nginx toma la IP del cliente de `X-Forwarded-For` (módulo realip) cuando la petición llega
  de `127.0.0.1` o de `172.16.0.0/12` (docker-proxy). El proxy TLS de delante debe enviar
  `X-Forwarded-For`; si no, todas las peticiones parecen venir de la misma IP y el límite de 20 r/s se
  aplica a todo el sitio a la vez. (Enmienda 2026-10-13, fase 6: nginx confía ahora en `127.0.0.1` y en los
  tres rangos privados, `10.0.0.0/8`, `172.16.0.0/12` y `192.168.0.0/16`, con `real_ip_recursive on`, para
  el ALB de ECS; ver "Despliegue en ECS".)
- Retirar un run (`unpublish` o `point`) tarda hasta unos 3 minutos en verse: TTL de los punteros en la
  API (60 s) + caché de nginx (60 s) + `max-age=60` en el navegador.
- Logs: `supervisord` se demoniza, así que dentro del contenedor `/dev/stdout` es `/dev/null`. nginx
  escribe en `/proc/1/fd/1` (acceso) y `/proc/1/fd/2` (errores), la salida de `init.sh` (PID 1), que es la
  del contenedor: deben verse con `docker logs` (la prueba de humo lo comprueba). Los logs de gunicorn
  siguen perdiéndose hasta que Luis adopte `nodaemon=true` / `exec supervisord` en `init.sh` (decisión
  suya). (Enmienda 2026-10-13, fase 6: ya no se pierden: gunicorn escribe el acceso en `/proc/1/fd/1` y los
  errores en `/proc/1/fd/2`, como nginx, y arranca con `-w $GUNICORN_WORKERS`; supervisord sigue
  demonizado.)
- `docker stop` tarda 10 s: `init.sh` deja bash como PID 1 esperando en `tail -f`, que ignora SIGTERM,
  y Docker acaba con SIGKILL. Por la misma razón (`exec supervisord`), la prueba de humo solo lo avisa
  (WARN).

La API (`/api/v1`, ver `docs/web/contrato.md`) lee el paquete `site/v1` de S3 y, desde la fase 4 (enmienda
2026-10-10), también la base de datos para las rutas de tablas (`/api/v1/parties`, `/pollsters`, `/ratings`,
`/polls`, `/results`, `/house-effects`, `/herding`, `/drift` y, desde la fase 5, `/api/v1/events/{scope}`, que cruza las
elecciones de la base con las publicadas) y para la página `/casas`. Las elecciones pasadas
(`/api/v1/events/{scope}/{date}/{part}`) y el backtest (`/api/v1/backtest/{scope}/{part}`) se sirven del paquete, como
el análisis. La base se configura
con las variables `DB_*` de `deploy/elections.env` (`DB_ADAPTER`, `DB_HOST`, `DB_PORT`, `DB_USERNAME`,
`DB_PASSWORD`, `DB_DATABASE`; ver `.env.example`): sin ellas (basta con `DB_HOST` vacío), esas rutas dan 503
`no database configured` y `/casas` un 503 en HTML ("La base de datos no está disponible."); el resto del sitio
funciona igual. Con las variables pero sin base alcanzable, 503 `database unavailable` (con la traza en el
log). Cada conexión a PostgreSQL se abandona a los 5 s (`connect_timeout`); las consultas no tienen límite
salvo que se fije `DB_STATEMENT_TIMEOUT` (milisegundos). Para el contenedor de la web, pon
`DB_STATEMENT_TIMEOUT=10000` en `deploy/elections.env`: una consulta colgada da 503 `database unavailable` a los
10 s en vez de bloquear el worker; las cargas y los notebooks lo dejan vacío (enmienda 2026-10-13: en ECS va en
la definición de tarea de la web, `task-web.example.json`, y la de jobs lo deja vacío). La variable afecta a todo lo que
corra con el entorno del contenedor: si el cron (`APP_CRONTAB`) o el worker de cola comparten contenedor con
la web, también sus jobs quedan limitados a 10 s. Mientras `deploy/docker/crontab` esté vacío no hay
conflicto; si un día el contenedor ejecuta publicaciones o cargas, ejecútalas en otro contenedor con la
variable vacía (un `RUN_JOB` es siempre un `docker run` aparte: `-e DB_STATEMENT_TIMEOUT=` lo anula, porque
`-e` gana a `--env-file`). Variables opcionales de `deploy/elections.env`: `WEB_PREFIX` (prefijo del
paquete, por defecto `site/v1`) y `WEB_CACHE_TTL` (TTL de los punteros en segundos, por defecto 60).
Sin ningún run publicado, la API responde 503 `no bundle published yet` y las páginas muestran un 503
en HTML.

Tras arrancar el contenedor:

```
curl -s http://127.0.0.1:8042/healthz
bash deploy/check-api.sh http://127.0.0.1:8042
```

y abrir `http://127.0.0.1:8042/`, `/es`, `/es/escanos`, `/elecciones`, `/casas`, `/metodo` y sus cuatro subpáginas
(enmienda 2026-10-13, fase 7). `check-api.sh` decide una vez si hay nginx a partir de `/healthz` (solo existe en
nginx; sin nginx lo marca `n/a`) y si hay base a partir de `/api/v1/parties` (un 503 `no database configured` la da
por ausente y marca `n/a sin base de datos` las rutas que la necesitan). Comprueba `/` (200 HTML),
`/es?mode=nowcast` y `/es/escanos?mode=forecast` (200 HTML), `/es/escanos?region=abc` (400 HTML), `/promedio` y
`/modelo` (301), `/es-zz` (404 HTML), `/casas?scope=es` (200 HTML, necesita base), `/casas?id=abc` (400 HTML),
`/elecciones?scope=es` (200 HTML), `/elecciones?date=abc` (400 HTML), `/elecciones?scope=es&date=2023-07-23` (200 HTML
si la elección está publicada), `/metodo/backtest?scope=es`, `/metodo`, `/metodo/promedio`, `/metodo/escanos` y
`/metodo/rating` (200 HTML), `/metodo/nope` y `/nope` (404 HTML), con nginx también
`/dist/vendor/echarts-5.6.0.min.js` (`application/javascript`), `/dist/vendor/fonts/SourceSerif4-4.005-Semibold.woff2`
(`font/woff2`) y `/dist/css/site.css` (`text/css`), y 21 rutas de la API (código, tipo y JSON válido): las 10 del
paquete, `/api/v1/ratings?scope=es`, `/api/v1/polls?scope=es&limit=5`, `/api/v1/results/es/2023-07-23`,
`/api/v1/drift?scope=es` y `/api/v1/events/es` (200 JSON, necesitan base), `/api/v1/analysis/es-xx/herding` (400),
`/api/v1/analysis/es/nope` (404), `/api/v1/events/es/2023-07-23/pollsters` y `/api/v1/backtest/es/by-horizon` (200
JSON si están publicados), `/api/v1/events/es/2023-13-01/meta` (400) y `/api/v1/backtest/es/nope` (404); sale con
error si alguna falla. Lo que escriben `publish event` y `publish backtest` puede faltar en el paquete (en local,
casi siempre): `check_published` marca `n/a  RUTA sin publicar` un 404 cuyo cuerpo dice `not published` (API) o "no
publicada"/"no publicado" (páginas), y comprueba como `check RUTA 200` cualquier otra respuesta.

### Cómo añadir una página

(Enmienda 2026-10-13, fase 7.) Primero, la familia:

- **De evento** (habla de la próxima elección de un ámbito, como Promedios y Escaños): ruta `/{scope}/<nombre>` en
  `pages.ROUTES`, después de las fijas; el controlador valida el ámbito con `pages.resolve_event_scope(scope)`, pasa
  `self.state(scope)` al constructor del contexto y este llama a `common_context(state, None, event=True,
  active_suffix='/<nombre>')`. El enlace va como pestaña de la barra de evento: una entrada en `pages.EVENT_TABS`.
- **Global** (vale para todos los ámbitos, como el histórico o el ranking): ruta fija en `pages.ROUTES`, antes de
  `/{scope}`, y entrada en el menú `pages.PAGES`; el contexto llama a `common_context(state, '/<nombre>')` y, si el
  contenido depende del ámbito, la plantilla incluye `_scope_filter.html` (`?scope=`, `es` por defecto).

Después, en las dos:

1. Controlador `mtpy/controllers/<Nombre>.py` que herede de `Page`, con `active` y una acción que llame a
   `render(plantilla, **contexto)`; ver `Promedios.py` (evento) o `Elecciones.py` (global).
2. Plantilla en `web/templates/` que extienda `base.html` y monte cada sección con las macros de `_macros.html`
   (`headline`, `card_start`/`card_end`, `how` y, al final, `closing` con `method_href('<sección>')`); si muestra
   medidas de error, incluye `_glosario.html` y usa sus nombres de columna.
3. El constructor del contexto, en su propio módulo `mtpy/lib/pages/<página>.py` (como `promedios.py` o
   `elecciones.py`), reexportado en `mtpy/lib/pages/__init__.py` para que el controlador lo llame como
   `pages.<página>_context`.
4. Tests en `tests/test_pages.py`; `tests/test_web_routes.py` comprueba que todo enlace de una plantilla resuelve.
5. Un parámetro propio de la página (como `region` en Escaños) se valida en `webapi` (`check_region`) y se lee en la
   acción, fuera de `parse_state`: así no entra en los enlaces del menú ni en el estado común `scope`/`mode`/`run`.
6. Las rutas nuevas, en `deploy/check-api.sh` y `deploy/smoke.sh`.

Una subpágina del método no necesita ruta ni controlador: basta su plantilla `web/templates/metodo_<id>.html` (que
incluye `_metodo_nav.html`) y su entrada en `METODO_SECTIONS` (`mtpy/lib/pages/metodo.py`: `title` y `template`). Una
entrada con `template` a `None` no se enlaza y su ruta da 404; `method_href('<id>')` devuelve `/metodo/<id>` cuando
hay texto y `/metodo` mientras no lo hay, así que las tarjetas de cierre pueden enlazarla desde el principio. Si la
subpágina necesita datos propios, `metodo_section_context` ramifica por sección (como el backtest).

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

El `location /` envía a Python todo lo que llegue a ese `server`: hay que ponerlo en un `server` (o puerto)
que no sirva nada más, porque sustituye a cualquier `/` existente. Después, arrancar la aplicación:

```
S3_BUCKET= python -m uvicorn api:api --port 8000
```

Con `S3_BUCKET=` vacío la API lee el paquete local `files/site/v1`. Después se abre `http://localhost:8080/`.
Sin nginx, `http://127.0.0.1:8000/` y `/es` se renderizan, pero los recursos de `/dist/` (CSS, JS y fuentes) dan 404:
Python solo sirve las páginas y la API.

Contra uvicorn directamente, `curl -I` (HEAD) sobre `/` o `/promedio` da 404, porque las páginas se enrutan solo
para GET y HEAD no está mapeado; detrás de nginx el HEAD se convierte en GET (`proxy_cache_convert_head`), así que
el `curl -sI` de `smoke.sh` funciona. (Enmienda 2026-10-13, fase 6: ya no es así: el router acepta `HEAD` en
toda ruta `GET` y responde con las mismas cabeceras y sin cuerpo, también contra uvicorn; una ruta
inexistente da 404.)

### Comprobaciones de la infraestructura

- `bash deploy/nginx-check.sh`: `nginx -t` sobre `deploy/docker/nginx.conf` con los dos `alias` de
  `/dist/` reescritos a `<repo>/web/dist/...` (no arranca nginx). Requiere nginx instalado.
- `bash deploy/smoke.sh [deploy/elections.env]`: construye la imagen, la arranca, comprueba que no hay
  `*.env` en ella, ejecuta `check-api.sh`, las cabeceras de caché de `/dist/vendor/`
  (`max-age=31536000`, también de `/dist/vendor/fonts/SourceSans3-3.052-Regular.woff2`), `/dist/js/pages/promedios.js`
  (`no-cache`) y `/` (`max-age=60`), que `/`, `/es`, `/es/escanos`, `/elecciones?scope=es`,
  `/metodo/backtest?scope=es`, `/metodo`, `/metodo/promedio`, `/metodo/escanos` y `/metodo/rating` devuelven HTML y
  que `/` contiene `initial-data`; con `deploy/elections.env`, también que
  `/casas?scope=es` devuelve HTML (con el paquete local y sin base, `/casas` es un 503 y `check-api.sh` lo
  marca `n/a`, así que el humo no lo exige); avisa (WARN, sin fallar) si `docker logs` no tiene
  líneas de acceso de nginx o si `docker stop` tarda 10 s o más. Sin argumento usa el
  paquete local (`files/` montado, `S3_BUCKET` vacío); con `deploy/elections.env`, S3 y la base reales.
  Requiere Docker en marcha.

Estado: `nginx.conf` se verificó el 2026-10-09 con un nginx 1.29.5 local (Homebrew) en el puerto 8043
delante de uvicorn en el 8000: `/` primero `MISS` y luego `HIT`, cabeceras como las esperadas y
`check-api.sh` todo OK. La prueba de humo con Docker está sin ejecutar (sigue así al cierre de la fase 6,
2026-10-13: el demonio de Docker estaba parado).

Enmienda 2026-10-10 (fase 4): `check-api.sh` ganó `has_db` y comprueba `/casas`, `/api/v1/ratings` y las rutas
de `analysis`; `bash deploy/check-api.sh http://127.0.0.1:8043` contra uvicorn con la base real y un nginx
1.29.5 local dio todo OK (sin análisis publicado en local). Los pendientes de la fase 4 están en el estado al
cierre de la fase 4 de la spec.

Pendiente de Luis: ejecutar `bash deploy/smoke.sh` con Docker (paquete local y `deploy/elections.env`);
añadir las `location` de `/dist/` y el proxy a su nginx local y recorrer `/`, `/promedio` y `/escanos` con y sin JS
(el botón "Ver" debe funcionar sin JS); desplegar la imagen (enmienda 2026-10-13: el despliegue pasa a ECS, ver
"Despliegue en ECS").

Enmienda 2026-10-09: la primera publicación a S3 ya está hecha (Luis, 2026-10-08, run `20261008-202843`) y
`jinja2` está en 3.1.6 desde `68522d2`; ya no son pendientes. Fase 3: `check-api.sh` y `smoke.sh` comprueban
`/escanos`; ver el estado al cierre de la fase 3 en `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`.

Enmienda 2026-10-11 (fase 5): `check-api.sh` ganó `check_published` y comprueba `/elecciones`, `/modelo`, `/metodo` y
las rutas de `events` y `backtest`; `smoke.sh`, las tres páginas. Contra uvicorn sin nginx, con el paquete local
(`S3_BUCKET=` y `DB_HOST=` vacíos, solo el backtest de `es` publicado en local) dio todas las rutas OK: las de base
`n/a sin base de datos` y las de la elección 2023-07-23 `n/a sin publicar`. Los pendientes de la fase 5 están en
el estado al cierre de la fase 5 de la spec.

Enmienda 2026-10-13 (fase 7): `check-api.sh` comprueba las rutas de evento (`/es`, `/es/escanos`), las
redirecciones (`/promedio`, `/modelo`), `/es-zz` (404), el método y sus subpáginas (`/metodo/nope` 404) y una fuente
de `/dist/vendor/fonts/` (solo con nginx); `smoke.sh`, las páginas nuevas y la caché de un año de las fuentes.
Contra uvicorn sin nginx, con el paquete local (`S3_BUCKET=` y `DB_HOST=` vacíos), dio 32 OK y 13 `n/a` (sin nginx,
sin base, sin publicar y `congelación: inactiva`), `todas las rutas OK`. Los pendientes de la fase 7 están en el
estado al cierre de su spec.

## Despliegue en ECS

La web corre como un servicio de ECS Fargate detrás de un ALB (host raíz, sin prefijo de ruta); las publicaciones
se lanzan como tareas puntuales (ver "Publicar desde ECS"). Las plantillas están en `deploy/ecs/` (JSON puro, sin
comentarios; los valores entre `<>` son marcadores que hay que sustituir):
`task-web.example.json` (servicio web), `task-job.example.json` (jobs) y `run-task.example.json` (argumentos de
`aws ecs run-task`).

### Qué debe existir en AWS

- Un clúster de ECS `elections-model` y un repositorio de ECR `elections-model`.
- Un servicio de ECS Fargate con la definición `elections-model`, con 2 tareas como mínimo, registrado en un target
  group (puerto 8042, protocolo HTTP) con health check en `/api/v1/health`, cada 30 s y 2 fallos para marcarlo
  no sano. Detrás, un listener del ALB (HTTPS 443, host raíz) que reenvía a ese target group. (La spec de la
  fase 6 decía que una tarea basta; con dos, cada tarea tiene su caché de nginx y de la API y el despliegue no
  deja el servicio sin tareas. Lo decide Luis; el p95 de la prueba de carga es el criterio para escalar.)
- Grupos de seguridad: el del ALB (entrada 443 desde internet), el de las tareas (entrada 8042 solo desde el
  ALB) y acceso desde el de las tareas al puerto 5432 de la RDS. Las tareas necesitan salida a internet
  o endpoints de VPC para ECR, S3, CloudWatch Logs y Secrets Manager.
- Rol de tarea de lectura (`<task-role-read>`): `s3:GetObject` sobre `arn:aws:s3:::<bucket>/site/*` y
  `s3:ListBucket` sobre `arn:aws:s3:::<bucket>` (condición de prefijo `site/*`). Es el que lee la web.
- Rol de tarea de escritura (`<task-role-write>`): lo anterior más `s3:PutObject` y `s3:DeleteObject` sobre
  `site/*` y `site-dry/*`. Solo para la definición `elections-model-job`.
- Los dos roles de tarea, además, `logs:CreateLogStream` y `logs:PutLogEvents` sobre el grupo de logs
  (`arn:aws:logs:<region>:<account>:log-group:<log-group>:*`): el logger de la aplicación (`CWL_GROUP`) escribe con
  el rol de la tarea, mientras que el driver `awslogs` usa el rol de ejecución. (Enmienda 2026-10-13, revisión
  final: sin estos permisos el logger no escribe, pero un fallo de CloudWatch ya no interrumpe el job ni la web;
  va a la salida de errores del contenedor.)
- Rol de ejecución (`<execution-role>`): `AmazonECSTaskExecutionRolePolicy` (incluye `logs:CreateLogStream` y
  `logs:PutLogEvents` para `awslogs`) más `secretsmanager:GetSecretValue` sobre el secreto de `DB_PASSWORD`.
- Un grupo de logs de CloudWatch (`<log-group>`), usado por `awslogs` y por `CWL_GROUP` (logger de la aplicación).
- Un secreto con la contraseña de la base (`<secret-arn>`), que ECS inyecta como `DB_PASSWORD`.

### Marcadores de las plantillas

| Marcador | Significado |
| --- | --- |
| `<account>`, `<region>`, `<tag>` | Cuenta de AWS, región y etiqueta de la imagen en ECR |
| `<rds-host>`, `<db-user>`, `<db-name>` | Host, usuario y base de datos de la RDS (`DB_HOST`, `DB_USERNAME`, `DB_DATABASE`) |
| `<bucket>` | Bucket de S3 (`S3_BUCKET`) |
| `<log-group>` | Grupo de logs de CloudWatch (`awslogs-group` y `CWL_GROUP`) |
| `<secret-arn>` | ARN del secreto con la contraseña (`DB_PASSWORD`) |
| `<task-role-read>`, `<task-role-write>` | Roles de tarea de la web (lectura) y de los jobs (escritura) |
| `<execution-role>` | Rol de ejecución (descarga la imagen, escribe logs, lee el secreto) |
| `<subnet-id>`, `<security-group-id>` | Subred y grupo de seguridad de la tarea de job (necesita acceso a la RDS) |
| `<git-sha>` | Commit de la imagen registrada (`GIT_COMMIT` de la tarea de jobs; ver "Publicar desde ECS") |

La web usa 1 vCPU y 2 GB con `GUNICORN_WORKERS=2` y `DB_STATEMENT_TIMEOUT=10000`; el job usa 2 vCPU y 4 GB y deja
`DB_STATEMENT_TIMEOUT` vacío (sin límite). Ninguna plantilla lleva `AWS_KEY` ni `AWS_SECRET`: las credenciales
salen del rol de la tarea. `DB_ADAPTER` es `PostgreSQL`, el nombre exacto del módulo `mtpy/core/dal/PostgreSQL.py`
(enmienda 2026-10-13, revisión final: las plantillas decían `postgres` y la web no arrancaba).

### Procedimiento

```
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build --platform linux/amd64 -t elections-model:<tag> .
docker tag elections-model:<tag> <account>.dkr.ecr.<region>.amazonaws.com/elections-model:<tag>
docker push <account>.dkr.ecr.<region>.amazonaws.com/elections-model:<tag>
aws ecs register-task-definition --cli-input-json file://deploy/ecs/task-web.example.json
aws ecs update-service --cluster elections-model --service elections-model --task-definition elections-model --force-new-deployment
bash deploy/check-api.sh https://<host>
```

Copia la plantilla a un fichero local con los marcadores ya sustituidos antes de registrarla (los
`*.example.json` del repositorio no se editan con datos reales). `check-api.sh` imprime
`info cliente según nginx: <ip>`: detrás del ALB debe ser tu IP pública; si es una IP privada avisa con `WARN`.
nginx confía en `127.0.0.1` y en los rangos privados (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) para
leer `X-Forwarded-For` del ALB; así `limit_req` y los logs ven al cliente real.

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
docker run --rm --env-file deploy/elections.env -e RUN_JOB=check_s3 elections-model:<tag>
```

Cada paso imprime `<paso> OK` o `<paso> FAIL: <error>`; si alguno falla, el job termina con error.

## Si `check_s3` falla

1. Subir `s3fs` a una versión de 2024 o posterior (con un `aiobotocore` compatible) y repetir.
2. Si sigue fallando, la fase 1 reimplementa `read_bytes`, `write_bytes`, `exists`, `listdir` y `remove` de `mtpy/core/services/s3.py` sobre `boto3`.

El primer `FAIL` es la causa raíz (los pasos posteriores fallan en cascada). Un `AccessDenied`, o un `gone FAIL` tras `remove OK`, indica que al usuario de AWS le falta `s3:DeleteObject` sobre el bucket, no un problema de `s3fs`.

## Publicar

`python job.py publish` lleva al paquete `site/v1` de `app.fs` lo que la web dibuja del modelo. Solo `forecast`
produce runs: ejecuta el modelo, escribe un run inmutable y mueve los punteros (`history.json` por ámbito y
`manifest.json`). El análisis de casas, las elecciones pasadas y el backtest (`analysis`, `event`, `backtest`) no son
runs: sobrescriben su carpeta y no tocan los punteros. `manifest`, `point` y `unpublish` solo mueven punteros.
Contrato de los ficheros: `docs/web/contrato.md`. El código está en el paquete `mtpy/lib/publish/` (`common`,
`forecast`, `analysis`, `backtest`, `events` y `pointers`); el job lo usa como `publish`.

### Qué hace cada acción y cuándo lanzarla

Antes de publicar está la carga, que no es cosa de este job: los notebooks de `notebooks/data-load/` (para `es`) y
`load/run_load.py` (para los autonómicos: `--what urls polls results compute ratings`) meten en la base los sondeos y
los resultados y recalculan ahí las series derivadas (pesos, errores, deriva, efectos de casa, herding) y los
ratings de las encuestadoras. La web lee los ratings de la base en vivo (`/api/v1/ratings` y el ranking de `/casas`),
así que un recálculo de ratings se ve sin publicar nada. Todo lo demás sale del paquete y hay que publicarlo.

| Acción | Qué escribe | Cuándo lanzarla | Guarda LOREG | Punteros |
| --- | --- | --- | --- | --- |
| `forecast` | Un run inmutable por ámbito en `runs/{ámbito}/{run_id}/`: promedio de hoy, pronóstico de la elección, sondeos del ciclo y titular (páginas Promedios y Escaños, portada) | Cada vez que se cargan sondeos nuevos de un ámbito; a diario en campaña. Con `scopes: "all"` se publica un run nuevo en todos los ámbitos con sondeos (los que no tienen bastantes quedan `skipped`); con una lista, solo en esos | Sí | Mueve `latest` al run nuevo |
| `analysis` | `analysis/{ámbito}/`: efectos de casa del ciclo, herding, errores por sondeo de las elecciones pasadas y datos del estimador de error (tarjetas de `/casas`; histograma, diana y tabla de `/elecciones`) | Con cada `forecast`, en la misma llamada (`"what":["forecast","analysis"]`); también tras un recálculo de ratings y tras cargar los resultados de una elección, porque entra en los errores históricos | Sí | No |
| `event` | `events/{ámbito}/{fecha}/`: error de cada encuestadora y de cada sondeo frente al resultado, resultado oficial y lo que el modelo habría dicho a cada horizonte (detalle de `/elecciones?date=`) | Una vez por elección celebrada, con los resultados oficiales cargados (`event_date` obligatorio). De nuevo, con `"event_date":"all"`, si se corrigen sondeos de elecciones pasadas o se rehace el backtest, porque su parte "modelo" sale de `backtest/results` | No | No |
| `backtest` | `backtest/{ámbito}/`: las tablas de `backtest/results` convertidas al paquete (`/metodo/backtest`) | Tras cada ejecución de `backtest/run_backtest.py` (`--scope es` para `es`, `--scopes all` para los autonómicos), es decir, cuando cambia el método. No ejecuta el modelo ni lee la base: solo convierte ficheros | No | No |
| `manifest` | Solo `manifest.json` | Congelar y descongelar la web en la veda (`freeze`). `forecast` ya reescribe el manifest, así que sola solo hace falta para eso | No | Sí |
| `point` | Solo `manifest.json` | Volver a un run anterior sin borrar nada (`latest` nunca retrocede por sí solo) | No | Mueve `latest` al run indicado |
| `unpublish` | Borra la carpeta del run y reconstruye `history.json` | Retirar un run malo; si era `latest`, vuelve al anterior | No | Sí |

`what` admite las siete acciones y las ejecuta en ese orden; `event` exige `event_date` y no se combina con
`forecast` (las dos cosas fallan con `ValueError` antes de empezar). `backtest` y `event` pueden ir en la misma
llamada: los dos leen `backtest/results` del disco (o `source`), no el paquete.

Rutina, por situación:

- **Campaña (a diario, hasta el 23 de noviembre)**: cargar los sondeos y lanzar
  `{"what":["forecast","analysis"],"scopes":"all"}` (o la lista de ámbitos con sondeos nuevos; `es` tarda unos
  2 min). Comprobar en el resumen que cada ámbito termina `published` y mirar los "· n avisos".
- **Veda (del 24 al 28)**: no se publica nada; `manifest` con `freeze` el 23 y la noche del 29 (ver "Veda"). La
  guarda rechaza `forecast` y `analysis` en todos los ámbitos hasta el 29 incluido.
- **Tras una elección**: cargar los resultados oficiales y lanzar `event` de esa fecha (no pasa por la guarda, así
  que puede ir la misma noche); desde el día siguiente (la guarda cubre el propio día de la elección), `analysis`
  para que sus errores entren en el histórico, y `forecast` cuando haya sondeos del ciclo siguiente.
- **Tras un cambio del modelo**: `python backtest/run_backtest.py --scope es` y `--scopes all` (los autonómicos),
  después `{"what":["backtest","event"],"scopes":"all","event_date":"all"}` y por último `forecast`, para que
  `latest` lleve el modelo nuevo.
- **Tras recalcular ratings** (`run_load --what ratings` o los notebooks de carga): el ranking de `/casas` cambia
  solo; `forecast` y `analysis` para que el paquete use los ratings nuevos.
- **Incidencias**: `unpublish` para retirar un run malo o `point` para taparlo con uno anterior (ver "Retirada de
  un run"). `dry_run` ensaya `forecast`, `analysis`, `event` y `backtest` en `site-dry/v1` sin punteros (el manifest
  no se escribe); `backfill` reconstruye días pasados de la evolución (ver "Relleno retroactivo"); `force` salta la
  guarda y deja el run marcado con `meta.freeze: true`.

### Ejemplos

```
python job.py publish '{"what":["forecast"],"scopes":["es"]}'
python job.py publish '{"what":["forecast"],"scopes":"all"}'
python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
python job.py publish '{"what":["unpublish"],"scopes":["es"],"run":"20261007-141503"}'
python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-05","to":"2026-10-08"}}'
python job.py publish '{"what":["analysis"],"scopes":"all"}'
python job.py publish '{"what":["backtest"],"scopes":["es"]}'
python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"2023-07-23"}'
python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"all"}'
python job.py publish '{"what":["backtest"],"scopes":"all"}'
```

1. Publica el pronóstico de `es` (nowcast y forecast, 1000 simulaciones) y actualiza su entrada del manifest.
2. Lo mismo para `es` y todos los ámbitos autonómicos; los que no tienen sondeos quedan `skipped`.
3. Solo reescribe el manifest; con `freeze` activa o desactiva el aviso editorial de congelación.
4. Apunta el manifest de `es` a un run anterior (no borra nada).
5. Borra un run malo, reconstruye `history.json` y, si era el último, apunta al anterior (o quita el ámbito).
6. Ensayo: escribe en `site-dry/v1` con 20 simulaciones, sin `history.json` ni manifest.
7. Relleno retroactivo: un run por día del 5 al 8 de octubre de 2026, cada uno con solo los sondeos publicados
   hasta ese día (ver "Relleno retroactivo").
8. Análisis de casas de `es` y de todos los ámbitos autonómicos (enmienda 2026-10-10; ver "Análisis de casas").
9. Backtest del modelo de `es`, convertido desde `backtest/results` (enmienda 2026-10-11; ver "Backtest").
10. La elección general del 23 de julio de 2023 de `es` (enmienda 2026-10-11; ver "Elecciones pasadas").
11. Todas las elecciones `featured` de `es` anteriores a hoy.
12. El backtest de `es` y de todos los ámbitos autonómicos; los que no tienen resultados quedan `skipped`.

### Parámetros

| Parámetro | Por defecto | Significado |
| --- | --- | --- |
| `what` | `["forecast"]` | Acciones a ejecutar |
| `scopes` | `["es"]` | Ámbitos, o `"all"` (`es` más los ámbitos con padre) |
| `event_date` | siguiente evento del ámbito | Fecha de la elección (`YYYY-MM-DD`; si no es una fecha ISO, el ámbito queda `failed`). Para `event` es obligatorio y admite una cadena, una lista de fechas o `"all"` (las elecciones `featured` del ámbito anteriores a hoy); una fecha no ISO deja el ámbito `failed` (enmienda 2026-10-11) |
| `n_sim` | `1000` | Simulaciones por modo |
| `seed` | `42` | Semilla |
| `drange` | `6` | Ventana de sondeos del pronóstico |
| `max_fc` | `10` | Pasos máximos del pronóstico |
| `alpha` | `0.05` | Nivel de los intervalos |
| `correctors` | `null` | Cambios sobre los correctores por defecto (`house_effects`, `dispersion`, `regional_noise` activos) |
| `freeze` | `null` | Aviso del manifest: `{"active": bool, "message": str}` |
| `run` | `null` | `run_id` para `point` y `unpublish` (obligatorio en ambos) |
| `dry_run` | `false` | Escribir en `site-dry/v1` sin punteros |
| `force` | `false` | Publicar `forecast`/`analysis` dentro de la ventana LOREG |
| `backfill` | `null` | Relleno retroactivo (enmienda 2026-10-09): `{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}`; `to` es opcional (por defecto `from`), `from <= to` y `to` anterior a hoy (Europe/Madrid): como mucho ayer, porque el pronóstico de hoy es la publicación normal. Con `backfill`, `forecast` publica un run por día en vez de uno de hoy; exige `forecast` en `what` |
| `source` | `backtest/results` del repositorio | Carpeta de resultados del backtest que leen `backtest` y el corte del modelo de `event`: `source` para `es` y `source/{ámbito}` para los demás (enmienda 2026-10-11) |
| `today` | hoy (Europe/Madrid) | Solo para pruebas y ensayos (fija la fecha de la guarda LOREG y el tope de `backfill`, `YYYY-MM-DD`; a diferencia de `force`, no deja huella en `meta.freeze`) |
| `verbose` | `1` | Detalle de la salida por consola (niveles abajo) |

Niveles de `verbose`:

| Nivel | Qué sale |
| --- | --- |
| 0 | Solo la cabecera de la acción (`publish forecast: 18 ámbitos · run …`) y el resumen final (con "· n avisos" en cada línea) |
| 1 (por defecto) | Lo anterior más, por ámbito, `[i/n] ámbito: empezando`, la línea de resultado en el momento y los avisos del modelo en vivo (en todas las acciones que recorren ámbitos) |
| 2 | Lo anterior más las líneas de fase del pronóstico (`simulador listo`, `promedio ajustado`, `simulación de hoy`, …) |
| 3 | Lo anterior más la salida propia del modelo (`Simulator`/`Computer`, que reciben `verbose - 2`) |

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
es: published 20261005-120000 (as_of 2026-10-05, 290 polls, backfill 2026-10-05) in 118 s    # backfill, un día
es: skipped 2026-10-05 (run exists)                                                           # backfill, día ya publicado
es: failed 2026-10-06 (incomplete run exists; unpublish it first)                             # backfill, día a medias
es: published 20261007-141503          # point y unpublish
manifest: 1 scopes, freeze off
manifest: not written (dry run)
es: backtest published (28 cases, 5 elections) in 0 s                                         # backtest
<ámbito>: backtest skipped (no backtest results for <ámbito>)
es: event 2023-07-23 published (<n> polls, <n> pollsters) in <s> s                            # event, una línea por fecha
es: event 2020-01-01 skipped (2020-01-01 is not a featured election of es)
es: event all skipped (no past featured elections)
```

Con `event`, `results[ámbito]` es una lista con un resultado por fecha (`event_date`, `status`, ...).

Si algún ámbito falla, el job termina con error (código de salida 1) después de escribir el manifest y
el resumen; un ámbito fallido conserva su entrada anterior en el manifest.

Con `backfill`, `results[ámbito]` es una lista con un resultado por día (`day`, `status`, ...) y el
resumen tiene una línea por día. Si un día falla a medias (queda la carpeta del run sin `headline.json`),
al repetir se informa `failed 2026-10-05 (incomplete run exists; unpublish it first)` y el job termina con
error: hay que retirarlo antes con `unpublish` (`{"what":["unpublish"],"scopes":["es"],"run":"20261005-120000"}`)
y volver a lanzar el relleno. Los días completos se informan `skipped (run exists)`.

### Relleno retroactivo

`backfill` rellena la evolución de las publicaciones con runs "a fecha de" un día pasado:

- Un run por día del rango, con `run_id` `YYYYMMDD-120000` y `run_at` `YYYY-MM-DDT12:00:00Z`. El corte de
  sondeos es `drange`, derivado de `<día>` como hace el backtest: su límite inferior sube a los días entre
  `<día>` y `event_date`, así que solo entran los sondeos publicados hasta ese día (incluido) y los efectos de
  casa se reajustan con ellos; el simulador recibe además `limit_date=<día>`, que solo ancla `as_of` en ese
  día (no filtra sondeos). El run lleva `meta.limit_date = <día>`, la ventana efectiva en `meta.drange`
  (`[55, null]` para el 2026-10-05) y `headline.backfill = true` (y por tanto la fila de `history.json`).
- Es una reconstrucción, no lo que se publicó entonces: usa los ratings, la deriva y el catálogo de hoy y el
  evento de hoy (2026-11-29), incluso para días anteriores a la convocatoria. La web los dibuja como puntos
  huecos con "estimación retrospectiva".
- Un día cuyo run ya existe se salta (`skipped`, "run exists"): repetir el relleno es idempotente y además
  reconstruye `history.json` y la entrada del manifest (con la misma regla de `latest`), así que un reintento
  repara unos punteros que quedaron a medias. Un run a medias (sin `headline.json`) no se salta: falla.
- `latest` no retrocede: un relleno solo mueve el `latest` del ámbito si su run más nuevo tiene un `run_id`
  mayor que el actual (o no hay ninguno); si no, el manifest queda intacto, de modo que un run al que se
  volvió con `point` nunca se reactiva. Una publicación normal sigue apuntando al run recién producido.
  Excepción conocida (2026-10-09): si se vuelve con `point` a un día retrospectivo y después se repite el
  relleno de un rango que incluye un día retrospectivo posterior ya publicado, el reintento vuelve a dejar
  `latest` en ese día posterior (los reintentos reparan los punteros y no distinguen esa situación de un
  manifest a medias); en ese caso, repetir el `point`.
- `es` tarda unos 2 min por día contra la RDS. `to` tiene que ser como mucho ayer (hoy o después se
  rechaza): el pronóstico de hoy es la publicación normal, y un run a las 12:00 UTC de hoy se ordenaría de
  forma rara respecto a los runs reales de hoy.
- La guarda LOREG se aplica al momento de la publicación (el hoy real), como en cualquier publicación.
- `point`, `unpublish`, `manifest` y `dry_run` no cambian (`dry_run` con `backfill` escribe en `site-dry/v1`
  sin punteros).

### Análisis de casas

Enmienda 2026-10-10 (fase 4). `{"what":["analysis"],"scopes":"all"}` calcula, para cada ámbito, el análisis de
casas que lee la página `/casas` (efectos de casa, herding, errores de los sondeos y datos del estimador de
error) y lo escribe bajo `analysis/{ámbito}/` del paquete (formato en "Análisis" de `docs/web/contrato.md`):

- Sobrescribe el análisis anterior del ámbito (no hay runs ni histórico) y escribe `meta.json` el último: un
  análisis sin `meta.json` está incompleto y la API responde 404 `analysis not published`. Si el cálculo falla
  a medias, quedan partes nuevas junto a las viejas y manda `meta.json`.
- No toca el manifest ni los punteros, pasa por la guarda LOREG (ver "Guarda LOREG") y con `dry_run` escribe en `site-dry/v1/analysis/`.
- Cada ámbito termina `published`, `skipped` (sin sondeos del siguiente evento), `refused` (guarda LOREG) o `failed`; el resto continúa y
  el job termina con error si alguno falla. Resumen de una línea por ámbito, por ejemplo
  `es: analysis published (event 2026-11-29, <n> pollsters) in <s> s` o `es-md: analysis skipped (<motivo>)`.
- Duración estimada de ≈ 1,5 min por ámbito (el análisis de `es` tardó 44 s en el test de integración contra la
  RDS). Va bien a mano o desde la imagen, igual que `forecast`.
- Cadencia: repetirlo tras cada carga de sondeos (lo normal es en la misma llamada que `forecast`), tras cada
  recálculo de ratings (`run_load --what ratings` o los notebooks de carga) y tras cargar los resultados de una
  elección; el análisis no se actualiza solo (ver "Qué hace cada acción y cuándo lanzarla").

Publicar en local o en S3 funciona como para `forecast` (ver abajo; en local, `S3_BUCKET=` delante):

```
S3_BUCKET= python job.py publish '{"what":["analysis"],"scopes":["es"]}'
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["analysis"],"scopes":"all"}'
```

### Elecciones pasadas

Enmienda 2026-10-11 (fase 5). `{"what":["event"],"scopes":["es"],"event_date":...}` publica, para cada fecha, la
elección pasada que lee `/elecciones?date=` bajo `events/{ámbito}/{fecha}/` del paquete (formato en "Eventos" de
`docs/web/contrato.md`): el error de cada casa y de cada sondeo frente al resultado (un `Computer` de esa elección
sola, como `notebooks/PollstersEvent.ipynb`), el resultado oficial (región 0) y el corte del backtest del modelo de esa
elección (de `source`; vacío si el backtest no la tiene), con el error del sector en `meta.summary`:

- `event_date` es una fecha, una lista de fechas o `"all"` (las elecciones `featured` del ámbito anteriores a hoy).
  Lee la base (elecciones, resultados y sondeos): en local, la del `.env` de la raíz.
- Sobrescribe la publicación anterior de la fecha y escribe `meta.json` el último: una elección sin `meta.json` está
  incompleta y la API responde 404 `event not published`. Al republicar se borran los gemelos `model-*` cuyas tablas
  ya no están. No toca el manifest ni los punteros, no tiene guarda LOREG y con `dry_run` escribe en
  `site-dry/v1/events/`.
- Cada fecha termina `published`, `skipped` (no es una elección `featured` del ámbito, no tiene resultados oficiales
  o no tiene sondeos) o `failed`; el resto continúa y el job termina con error si alguna falla.
- Una elección de `es` tardó 15 s en el test de integración contra la RDS (2026-10-11).
- Cuándo relanzar: tras cargar los resultados oficiales de una elección (y tras corregir sus sondeos) y tras un
  nuevo `run_backtest`, porque el corte del modelo sale de los resultados del backtest.

### Backtest

Enmienda 2026-10-11 (fase 5). `{"what":["backtest"],"scopes":["es"]}` convierte las tablas que deja
`backtest/run_backtest.py` (`backtest/results/` para `es`, `backtest/results/{ámbito}/` para los autonómicos, o la
carpeta `source`) en `backtest/{ámbito}/` del paquete (`meta`, `metrics`, `by-horizon`, `blocks`, `shares` y `seats`,
con sus gemelos CSV; formato en "Backtest" de `docs/web/contrato.md`), que lee `/modelo`:

- **No necesita base de datos** ni ejecuta el modelo: solo lee ficheros del repositorio y tarda menos de un segundo
  por ámbito.
- Sobrescribe el backtest anterior del ámbito y escribe `meta.json` el último. No toca el manifest ni los punteros y
  con `dry_run` escribe en `site-dry/v1/backtest/`.
- Cada ámbito termina `published`, `skipped` (sin `metrics.csv` en su carpeta: `no backtest results for {ámbito}`) o
  `failed`.
- Cuándo relanzar: tras cada nueva ejecución de `backtest/run_backtest.py` (`--scope es` o `--scopes all`), y
  después `publish event` de las elecciones afectadas para refrescar su corte del modelo.

Publicar en local o en S3 funciona como para `forecast` (en local, `S3_BUCKET=` delante):

```
S3_BUCKET= python job.py publish '{"what":["backtest"],"scopes":["es"]}'
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["backtest"],"scopes":"all"}'
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"all"}'
```

### Publicar en local

Con `S3_BUCKET` vacío, `mtpy.run()` construye el sistema de ficheros local y el paquete queda en
`files/site/v1`. Si el `.env` de la raíz define `S3_BUCKET`, hay que anteponer `S3_BUCKET=` (vacío) al
comando, porque `load_dotenv` no pisa las variables ya exportadas y, sin ello, se publicaría en el bucket:

```
S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"]}'
S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-05","to":"2026-10-08"}}'
```

El paquete local lee la base de datos configurada en el `.env` de la raíz.

### Publicar en S3 contra la RDS

Desde el portátil:

```
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["forecast"],"scopes":["es"]}'
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-05","to":"2026-10-08"}}'
```

Un run retrospectivo es una reconstrucción (ratings, deriva y catálogo de hoy; solo los sondeos publicados
hasta ese día), `es` tarda unos 2 min por día, los días ya publicados se saltan y `latest` no retrocede
(ver "Relleno retroactivo").

Desde la imagen (`RUN_JOB` es el nombre del job, la primera palabra, y después el JSON, que puede llevar espacios:
`init.sh` lo pasa como un solo argumento; enmienda 2026-10-13, revisión final):

```
docker run --rm --env-file deploy/elections.env -e RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}' elections-model:<tag>
```

Antes de la primera publicación a S3, `check_s3` debe dar `OK` en todos los pasos.

### Publicar desde ECS

Registra la definición `elections-model-job` desde `deploy/ecs/task-job.example.json` (marcadores sustituidos) y lanza
la tarea con `deploy/ecs/run-task.example.json` (también con los marcadores sustituidos):

```
aws ecs register-task-definition --cli-input-json file://deploy/ecs/task-job.example.json
aws ecs run-task --cli-input-json file://deploy/ecs/run-task.example.json
```

El override de `RUN_JOB` lleva el nombre del job (la primera palabra) y después el JSON, que puede llevar espacios
(por ejemplo el mensaje de la congelación): `init.sh` pasa todo lo que sigue al primer espacio como un solo
argumento (enmienda 2026-10-13, revisión final; antes había que escribirlo sin espacios). Cambia `what`, `scopes` y
el resto de parámetros en ese valor. La salida del job está en el grupo de logs `awslogs` (`<log-group>`, flujo con
prefijo `job`). El rol de jobs (`<task-role-write>`) escribe en `site/*` y `site-dry/*`; la web, con el rol de
lectura, no puede escribir. Mide la memoria de `publish forecast all` antes de fiarte de los 4 GB de la plantilla
(la tarea 4 de la fase 6 lo mide en local). (Enmienda 2026-10-13: en local solo se midió `publish backtest`, que no
necesita la base; la memoria de `forecast` queda para la primera tarea de jobs en ECS, ver "Duraciones medidas".)

`GIT_COMMIT` de `task-job` es el sha de la imagen que registraste (`git rev-parse --short HEAD` del commit con el
que hiciste el `docker build`): la imagen no lleva `.git`, así que la procedencia de los runs publicados desde ECS
(`meta.commit`, el commit del pie de la web) sale de esa variable, con `dirty: false`. Sin ella el run queda con
`commit: null`. Actualízala cada vez que registres una imagen nueva.

**Aviso**: un `run-task` de `elections-model-job` sin el override de `RUN_JOB` arranca `init.sh` sin nada que hacer (la
definición no lleva `APP_API`, `APP_CRONTAB` ni `APP_QUEUE`) y se queda esperando en `tail -f /dev/null`, con 2 vCPU
y 4 GB facturándose, hasta que la pares (`aws ecs stop-task --cluster elections-model --task <task-id>`).

### Veda

Plan de la elección general del 2026-11-29: la veda de sondeos va del 24 al 28 (LOREG art. 69.7).

**23 de noviembre** (con los sondeos del día cargados como siempre):

1. `python job.py publish '{"what":["forecast"],"scopes":"all"}'` publica el pronóstico definitivo de todos los
   ámbitos (con `S3_BUCKET` y la RDS, como en "Publicar en S3 contra la RDS", o como tarea de ECS con ese
   `RUN_JOB`, ver "Publicar desde ECS"; lo mismo para los pasos 3 y el descongelado). Lanzarlo con margen, mucho
   antes de medianoche: la guarda lee la fecha de hoy (Europe/Madrid) al llegar a cada ámbito, así que los ámbitos
   que se publiquen pasadas las 00:00 del 24 quedan `refused` y el manifest queda mezclado (unos ámbitos con el
   run del 23 y otros con el anterior).
2. Comprobar el manifest (ver "Leer el manifest publicado").
3. `python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"<texto>"}}'` congela la web;
   `manifest.freeze.since` queda con la fecha del día. Texto sugerido: "Pronóstico definitivo, publicado el 23
   de noviembre. Durante la veda electoral no se publican sondeos nuevos ni se actualiza el pronóstico."
4. `bash deploy/check-api.sh https://<host>` comprueba el manifest y, congelado, la cabecera `x-freeze: active` y
   el banner.

**Noche del 29**: `python job.py publish '{"what":["manifest"],"freeze":{"active":false}}'` (`since` vuelve a
`null`). **Después**, con los resultados oficiales cargados, `python job.py publish
'{"what":["event"],"scopes":["es"],"event_date":"2026-11-29"}'` (`event` no pasa por la guarda).

**Sondeos durante la veda**: cargarlos es seguro. El cargador marca los de prohibición (`wban`) y los de
boca de urna (`exit`) y el modelo y la API los excluyen. No recalcular ratings ni publicar (`forecast`,
`analysis`, relleno) hasta el 30: la guarda lo rechaza en cualquier ámbito hasta el 29 incluido.

#### Ensayo

Todo en local (`S3_BUCKET=` vacío, paquete en `files/site/v1`); nada toca S3 ni la RDS pública. Repetirlo la
semana del 17 de noviembre.

1. `S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"],"today":"2026-11-25","dry_run":true}'`
   termina `refused`; lo mismo con un autonómico (`"scopes":["es-md"]`).
2. Con `"force":true` se publica en `site-dry` con `meta.freeze: true`.
3. `S3_BUCKET= python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"Ensayo"}}'` sobre
   el paquete local.
4. `S3_BUCKET= DB_HOST= uvicorn api:api --port 8001` y `bash deploy/check-api.sh http://127.0.0.1:8001`
   comprueba cabecera y banner.
5. Desactivar: `S3_BUCKET= python job.py publish '{"what":["manifest"],"freeze":{"active":false}}'`.

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
  60 s de la de nginx. Los runs pedidos con `?run=` son inmutables en el navegador (un año). (Más hasta 60 s
  del `max-age` del navegador: unos 3 minutos para un visitante que ya tenía la página, como dicen las notas
  de operación de "La web".)

### Guarda LOREG

Enmienda 2026-10-13 (fase 6). La guarda cubre todos los ámbitos: el job se niega a publicar (`refused`) un
`forecast` (incluido el relleno retroactivo) o un `analysis` si hoy cae en los 5 días previos a la elección
del ámbito o en el propio día, **o** en la misma ventana de la elección general (la de `es` de hoy o la próxima:
la ventana nacional también cubre el propio día de la elección; enmienda 2026-10-13, revisión final), porque
la prohibición de publicar sondeos (LOREG art. 69.7) es general. El motivo nombra la elección que cierra la
ventana (la propia tiene prioridad), por ejemplo `es-md 2027-05-23: inside the LOREG window (art. 69.7) of es
2026-11-29; pass force=true to publish`. En `forecast`, un ámbito sin elección próxima se salta (`skipped`, `no
upcoming event`); en `analysis` no: el análisis se calcula desde su última elección y solo lo guarda la ventana
nacional (enmienda 2026-10-13, revisión final). `event`, `backtest`, `manifest`, `point` y `unpublish` no pasan
por la guarda.

`{"force":true}` publica igualmente: el `forecast` queda marcado con `meta.freeze: true` y el `analysis` no
deja marca (su `meta` no la tiene). La fecha de hoy es la de Europe/Madrid; `today` la fija solo para pruebas y
ensayos. `manifest.freeze` es otra cosa: el interruptor editorial que se fija con `what: ["manifest"]` (ver
"Veda").

## Ensayos de operación

Cuatro ensayos antes de la semana de la elección. Los tres primeros se repiten con cada cambio de despliegue.

### Carga

`bash deploy/load-test.sh BASE_URL [CONCURRENCIA=4] [SEGUNDOS=20]` mide `/api/v1/forecast/es/nowcast/summary` y
`/` con `hey` (`-z ${SEGUNDOS}s -c $CONCURRENCIA -q 5`) o, si no está instalado, con `ab`. Imprime p95 y
códigos de respuesta, y luego hace 50 `curl -sI` a la ruta de la API contando `X-Cache: HIT`. Sale con 1 si hay
alguna 5xx o si la primera petición responde 429, y con 2 si falta la herramienta. `-q 5` limita cada cliente a 5 peticiones por segundo, pero nginx admite 20 r/s por IP y todos los clientes
salen de la misma: detrás de nginx o del ALB hay que mantener `CONCURRENCIA × 5 ≤ 20` (la concurrencia por defecto,
4, da justo 20 r/s) o esperar 429. Con `hey`, las líneas `[429]` y `[5xx]` fallan la prueba; con `ab` (sin límite por cliente) cualquier
respuesta no 2xx (`Non-2xx responses`) la falla, así que la carga debe quedar por debajo del límite por IP. Una
primera petición que no sea 2xx/3xx (host caído, 000) también sale con 1.

- Contra ECS: `bash deploy/load-test.sh https://<host>` con `hey` instalado. Todos los clientes salen de una IP,
  así que el límite por IP se aplica al conjunto. (Enmienda 2026-10-13: la concurrencia por defecto era 50, que
  supera los 20 r/s por IP y falla con 429; desde la revisión final es 4. `ab` no limita el ritmo por cliente:
  detrás de nginx o del ALB, usar `hey`.)
- Contra local: `bash deploy/load-test.sh http://127.0.0.1:8043` (nginx) o, sin caché, contra
  `S3_BUCKET= DB_HOST= uvicorn api:api --port 8001` (`http://127.0.0.1:8001`).
- Criterios de la spec: p95 de la API < 200 ms, p95 de las páginas < 500 ms, mayoría de `HIT` y sin 5xx.
- Si no se cumplen: mover a `asyncio.to_thread` las rutas que leen la base o el bundle (decisión E3), subir
  `GUNICORN_WORKERS` o la CPU de la tarea, y repetir.

### Retirada de un run (`point` y `unpublish`)

Siempre sobre una copia del paquete local, con `S3_BUCKET=` vacío (el `.env` de la raíz apunta al bucket real):

1. `cp -R files/site files/site.bak`
2. Hacen falta al menos dos runs en `files/site/v1/runs/es`. Si solo hay uno, copiar su carpeta con otro id
   y sustituir el id dentro de sus `.json` (`point` lee el `run_id` de `headline.json`).
3. `S3_BUCKET= python job.py publish '{"what":["point"],"scopes":["es"],"run":"<run antiguo>"}'`: el
   `latest` de `files/site/v1/manifest.json` pasa a ese run.
4. `S3_BUCKET= python job.py publish '{"what":["unpublish"],"scopes":["es"],"run":"<run más reciente>"}'`: se
   borra su carpeta y `runs/es/history.json` queda solo con los runs restantes.
5. Restaurar: `rm -rf files/site && mv files/site.bak files/site`, y comprobar que `latest` y las carpetas de
   `runs/es` son las de antes.

### Congelación

Ver "Veda" → "Ensayo" (guarda `refused`, `force`, `manifest.freeze` y `check-api.sh` en local). Repetirlo la
semana del 17 de noviembre.

### Copia y restauración de la base

La RDS tiene copias automáticas; esta es la copia lógica del esquema `elections` y comprueba que se restaura:

```
pg_dump -Fc -n elections -h <rds> -U <usuario> elections > elections-$(date +%F).dump
aws s3 cp elections-$(date +%F).dump s3://<bucket>/backups/
pg_restore -d <base local> --clean --if-exists elections-<fecha>.dump
python -m pytest -m integration -q
```

El criterio es que `python -m pytest -m integration -q` pase contra la base restaurada. Las credenciales de la
RDS no están en el repositorio: el ensayo lo ejecuta Luis (no se ha ejecutado en la fase 6).

## Duraciones medidas

Primer paquete local de `es` (2026-10-08, contra la RDS, `n_sim=1000`): 121 s en total (init 35,5; ajuste
42,5; nowcast 21,0; forecast 21,5; exportación 0,8). Ensayo con `n_sim=20`: 86 s. La duración de
`scopes: "all"` contra la RDS está pendiente de medir. El análisis de casas de `es` (`what: ["analysis"]`)
tardó 44 s en el test de integración contra la RDS (2026-10-10).

Ensayo de carga local (2026-10-13), `bash deploy/load-test.sh http://127.0.0.1:8001 10 5` contra
`S3_BUCKET= DB_HOST= uvicorn api:api --port 8001` (un proceso, sin nginx ni caché), con `ab`: API 30 622
peticiones, 6 111 r/s, p95 2 ms, 0 fallos; portada `/` 4 392 peticiones, 878 r/s, p95 12 ms, 0 fallos; sin 5xx.
`X-Cache: HIT` en 0 de 50 (no hay nginx delante). Sin nginx local en el puerto 8043: sin cifras de caché.
Es una referencia de la máquina de desarrollo, no de ECS; las de ECS las mide Luis con `https://<host>`.

Arranque de un worker (2026-10-13, `S3_BUCKET= DB_HOST= python -X importtime -c "import api"` y
`ru_maxrss` tras `import api`, macOS, tres ejecuciones): antes de los imports perezosos de `helpers.py` y
`dates.py` 1,55 s y 325 MB de RSS; después 0,82 s y 232 MB (`sklearn` y `statsmodels` ya no se cargan al
arrancar; `scipy` sí, a través de `dask`, que importa `mtpy.core.data` desde `PostgreSQL.py`).

`S3_BUCKET= /usr/bin/time -l python job.py publish '{"what":["backtest"],"scopes":["es"]}'` (único job sin base):
2,3 s y 360 MB de RSS máximo (343 MiB). La memoria de `publish forecast all` la mide Luis desde ECS con
`/usr/bin/time -v` (o la métrica de memoria de la tarea) para ajustar `memory` de `task-job`.
