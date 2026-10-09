<!-- Especificación de diseño aprobada por Luis el 2026-10-07 (sesión de planificación con Claude Code). -->

# Plan: interfaz web y API para publicar los resultados del modelo

## Contexto

Luis quiere publicar los resultados del modelo electoral en una web: pronósticos de voto, simulaciones de
escaños y mayorías, análisis de casas (errores, sesgos y camino hacia el rating) y eventos electorales
(errores por casa). Hoy esos resultados solo existen en notebooks (`notebooks/Polls*.ipynb`,
`notebooks/Pollsters*.ipynb`) y en ficheros locales gitignorados (`files/`).

El framework `mtpy` que envuelve el proyecto trae un sistema de API (gunicorn + uvicorn, nginx delante,
supervisord en el contenedor) que puede servir también una web. Criterios de Luis: sencillez del código,
reproducibilidad y dockerización.

Restricciones previas (memoria del proyecto): PostgreSQL es el sistema de registro y base de un servicio
de consulta; los datos son publicables (Wikipedia CC BY-SA 4.0, Infoelectoral con atribución, curación
propia); objetivo científico y divulgativo; los cambios de esquema en la base los hace Luis a mano.

## Decisiones del usuario (2026-10-07)

- Audiencia: público general, sin login; solo español.
- Interacción: consulta de resultados precalculados con filtros (ámbito, modo, agrupación, casa, evento,
  distrito). El modelo no se ejecuta en el servidor.
- Cadencia: publicación manual con un comando tras cargar y revisar sondeos. Sin cron.
- Hosting: servidor propio con Docker, evolucionando la imagen actual.
- Gráficos: Python publica datos JSON; módulos JavaScript pequeños los dibujan con una librería interactiva.
  Frontend HTML + JS moderno sin build (módulos ES nativos, librería vendorizada; ni Node ni bundler).
- API `/api/v1` con URL estables; JSON y descargas CSV.
- **Capa HTTP en `mtpy/controllers` y lógica de backend en `mtpy/lib`, como el resto del modelo.** El
  desacople de mtpy se verá más adelante; no diseñar fuera del framework.
- **Backend con PostgreSQL en producción**: los controladores consultan la base para las tablas de datos;
  las salidas del modelo se precalculan.
- Paquete precalculado entregado por **S3** vía `app.fs` (`S3_BUCKET`; en local escribe en `files/`).
- V1: `es` + ámbitos autonómicos simulables (11 hoy), solo MT, conmutador nowcast/forecast, casas, eventos
  pasados, precisión del modelo. LS/MC/DH se quedan en los notebooks.
- Páginas de evento tras cargar los resultados oficiales, a mano. Sin feed de resultados provisionales.
- Base de producción: **la RDS existente** de `deploy/elections.env`; Luis trabajará directamente contra
  ella (notebooks, `run_load`, publicar). La PostgreSQL local pasa a ser copia opcional de desarrollo.
- (2026-10-08) Imagen Docker: Luis revierte el `Dockerfile`, `supervisord.conf`, `init.sh`, el `crontab` y
  `gunicorn==20.1.0` a su versión original (quiere añadir funcionalidades y algún cron); de la fase 0 solo
  se conserva, en `.dockerignore`, la exclusión de `deploy/*.env`, `.git` y material de claves.
- (2026-10-08) Configuración única: `deploy/elections.env`, con el usuario de PostgreSQL y el usuario de
  AWS/S3 existentes, sirve para la web, la publicación y los jobs. Sin rol `web_reader` ni usuarios IAM
  nuevos por ahora.

## Hallazgos de la exploración (workflow de 10 agentes, solo lectura)

- **API de mtpy**: `mtpy.api()` devuelve un ASGI escrito a mano (`mtpy/core/api.py`). Rutas registradas
  solo en `mtpy/mtpy.py:106-107` (una: `/` → `Index`). Controladores `mtpy/controllers/<Nombre>.py`,
  clase `<Nombre>`, acciones `async <accion>_action(**params)`. Respuesta: dict → JSON (`json.dumps` con
  NaN), str/bytes → text/plain siempre, int → error JSON, list → 500 (`mtpy/core/api.py:121-151`). Sin
  estáticos, CORS ni cabeceras de caché. Alias de parámetros de ruta `[0-9a-z_\-]+` (minúsculas, sin
  puntos; `mtpy/core/api.py:181`). Query como lista de tuplas. Arranque sin base (motor perezoso); cada
  worker importa la pila científica completa.
- **nginx** (`deploy/docker/nginx.conf`): sin `root`; `try_files $uri @proxy_to_app` → proxy puro a
  gunicorn :8000; gzip solo HTML. Puerto 8042.
  (enmienda 2026-10-09: nginx sin `root` ni `try_files`; `location ^~ /dist/` y `^~ /dist/vendor/` con
  `alias` sirven los recursos y `location /` proxya páginas y API a gunicorn; ver
  `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md`)
- **Modelo**: un Simulator completo tarda 31-64 s en `es` y 13-27 s en autonómicos
  (`backtest/results/**/meta.csv`); no apto por petición. Salidas en memoria: `summary`, `probabilities`,
  `fan`, `vote_forecast`, `projection`, `dist`, `unit_summary`, `result(scenario())`, `totals`;
  `Forecaster.forecast/fc_stat/fc_series_raw/nfc_series/house_effects/dispersion`;
  `Computer.get_polls_metric/get_error_estimator_data/house_effects_summary/herding_summary`. Solo se
  persiste el promedio (`files/fc/{prefix}*.csv`) y PNG. `Simulator.mode` es un atributo llano
  (`simulator.py:158`); `horizon`, `vote_forecast` y `projection` lo siguen (`:1020-1062`), pero el
  horizonte por defecto de `run` se fija en el constructor (`:332`): pasar `horizon='deadline'` explícito.
- **Datos**: la base es imprescindible para ejecutar el modelo; los datos maestros solo existen en la base;
  `run_load.py` rechaza `es` (carga nacional por notebooks). `app.fs` abstrae `files/` o S3
  (`read_bytes`/`write_bytes`/`exists`/`listdir`, `mtpy/core/services/s3.py:77-275`); `app.data` es
  `data/` versionado. Tabla `pollsters_herding` nueva (M13, sin commit).
- **Historia del pronóstico**: no se puede regenerar fielmente (suavizado bilateral, pesos reescritos,
  ratings "de hoy", cambio de fecha del evento) → cada publicación es una instantánea inmutable con
  procedencia (patrón `backtest/run_backtest.py:59-78`, que omite el flag de árbol sucio).
- **Docker**: `.dockerignore` tiene semántica de raíz, así que `deploy/*.env` (credenciales reales), `.git`
  (127 MB), notebooks y `.superpowers/` entrarían en la imagen con `COPY . .`. Bloqueante de seguridad.
  Imagen única con build tools; supervisord sin `nodaemon`; crontab `job.py rep` inexistente.
- **Vistas de los notebooks**: V1 portada, V2 voto, V3 serie con sondeos, V4 sondeos del ciclo, V5
  efectos de casa del ciclo, V6 escaños MT, V7 distritos, V8 abanico por horizonte, V9 didáctica (fuera),
  V10 ranking, V11 perfil de casa, V12 errores del sector, V13 evento pasado, V14 precisión, V15 metodología.
- Otros: `bmap='max'` solo existe para `es`; etiquetas de evento ambiguas en `computer.py:1962` → añadir
  `event_date`; resultados oficiales llegan 25-121 días después de la elección; LOREG art. 69.7 (24 a 28
  de noviembre); SQL por formato de cadenas en `mtpy/lib/data.py` → validar toda entrada.

## Alternativas consideradas

- Sitio totalmente estático (nginx o GitHub Pages, sin Python): descartado, Luis quiere consultas en vivo
  sobre PostgreSQL y los controladores de mtpy.
- Backend fino sin base (controladores que solo sirven el paquete): descartado por Luis.
- Framework web moderno fuera de mtpy (FastAPI/Starlette): descartado por ahora; el contrato de API se
  diseña para que una migración futura no afecte al frontend.

Elegida: híbrida dentro de mtpy. Tres borradores independientes (simplicidad, contrato de datos,
operación) se han fusionado aquí; los conflictos resueltos se indican en cada apartado.

## Arquitectura

```
 portátil de Luis                                   servidor (un contenedor Docker)
 ────────────────                                   ───────────────────────────────
 notebooks / run_load ────► PostgreSQL (RDS) ◄────── controladores mtpy (gunicorn+uvicorn :8000, usuario
 python job.py publish ───► S3 site/v1/… ◄──────────┘  de solo lectura)       ▲
   Simulator + Computer → JSON/CSV + procedencia                               │ /api/ proxy + caché 60 s
                                                     nginx :8042 ── web/ estática (HTML + ES modules + ECharts)
```

Publicar: cargar sondeos como hoy → revisar → `python job.py publish '{"what":["forecast"],"scopes":"all"}'`
→ por ámbito: un Simulator, `fit_forecast` una vez, `run` nowcast y `run` forecast con la misma semilla →
exporta a `site/v1/runs/{scope}/{run_id}/` → valida → reconstruye `history.json` → al final reescribe
`manifest.json` (único puntero). La web no necesita redespliegue.

Consultar: navegador → nginx (`web/`) → `fetch('/api/v1/...')` → proxy con caché → controlador →
`mtpy/lib/webapi.py` → paquete (S3 con caché en memoria) o PostgreSQL (caché TTL) → JSON.

### Módulos nuevos y cambios

| Fichero | Papel |
|---|---|
| `mtpy/lib/bundle.py` (nuevo) | Disposición del paquete: `PREFIX='site/v1'`, `CONTRACT=1`, `run_id()`, `BundleWriter`/`BundleReader` sobre `app.fs` (`write_bytes(json.dumps(..., allow_nan=False, default=json_default))`, `read_bytes`, `exists`, `listdir`), `json_default` (numpy, Timestamp, NaN → null), `SCHEMAS` + `validate(name, obj)`. Único sitio que conoce rutas de ficheros. |
| `mtpy/lib/publish.py` (nuevo) | Exportadores puros (`export_meta`, `export_headline`, `export_series`, `export_polls`, `export_fan`, `export_house_effects`, `export_dispersion`, `export_vote`, `export_summary`, `export_dist`, `export_districts`, `export_scenario`, `export_projection`, `export_analysis`, `export_event`, `export_backtest`) y orquestadores `publish_forecast(scope, writer, ...)`, `publish_analysis`, `publish_event`, `publish_backtest`, `write_manifest`, `point`, `unpublish`. Sin estado. |
| `mtpy/jobs/Publish.py` (nuevo) | `class Publish(Job)`: parsea parámetros, itera ámbitos con aislamiento de errores, `alert()` final. Elegido frente a un script argparse porque es el mecanismo del framework, funciona con `RUN_JOB` en Docker y avisa por Pushover; la lógica vive en `publish.py`. |
| `mtpy/lib/webapi.py` (nuevo) | Capa de servicio de la API: validadores (`check_scope`, `check_date`, `check_run`, `check_mode`, `check_part`, `check_int`), `HttpError`, `TTLCache`, lectura cacheada del paquete, consultas a base (`query_parties`, `query_pollsters`, `query_pollster`, `query_ratings`, `query_polls`, `query_results`, `query_events`, `query_house_effects`, `query_herding`, `query_drift`) sobre los lectores de `mtpy/lib/data.py`, `frame_to_records`, `frame_to_csv`, tabla `ROUTES` y `add_routes(router)`. |
| `mtpy/controllers/Base.py` (nuevo) | `Base(Controller)`: query string a dict, `HttpError` → JSON de error, `cache()` (`Cache-Control`/`ETag`), `csv()` (`text/csv` + `Content-Disposition`), aviso `freeze`. |
| `mtpy/controllers/Forecast.py`, `Analysis.py`, `Data.py` (nuevos) | Un controlador por familia de rutas (paquete de pronósticos; análisis, eventos y backtest; tablas de base). Solo validan, llaman a `webapi` y devuelven dict/list/bytes. |
| `web/` (nuevo) | Multipágina: `index.html`, `promedio.html`, `escanos.html`, `casas.html`, `elecciones.html`, `modelo.html`, `metodo.html`; `js/{api,state,format,catalog,seats}.js`, `js/charts/*.js`, `js/pages/*.js`; `css/site.css`; `vendor/echarts.min.js` + licencia. |
| `deploy/smoke.sh`, `deploy/README.md` (nuevos) | Humo Docker y runbook. La configuración es el fichero existente `deploy/elections.env` (decisión del 2026-10-08). |
| `docs/web/contrato.md` (nuevo) | Copia legible de `SCHEMAS` y de la tabla de rutas. |
| `tests/test_bundle_unit.py`, `test_publish_unit.py`, `test_webapi_unit.py`, `test_api_asgi.py`, `test_web_routes.py`, `tests/integration/test_publish.py` (nuevos) | Ver Pruebas. |
| `mtpy/core/api.py` (cambia) | Ver "Cambios en el núcleo". |
| `mtpy/mtpy.py:100-111` (cambia) | `api(routes=None)`: registra `/` y además las rutas recibidas. |
| `api.py` raíz (cambia) | `api = mtpy.api(routes=webapi.ROUTES)`. |
| `mtpy/lib/computer.py:1962` (cambia) | `get_polls_metric` añade la columna `event_date` ISO junto a la etiqueta ambigua. Aditivo. |
| `config.json`, `.env.example` (cambian) | Sección `"web": {"prefix": "${WEB_PREFIX}", "cache_ttl": "${WEB_CACHE_TTL}", "freeze": "${WEB_FREEZE}"}`; vacíos → valores por defecto en código. |
| `deploy/docker/nginx.conf`, `supervisord.conf`, `init.sh`, `Dockerfile`, `.dockerignore`, `requirements.txt` (cambian) | Ver Infraestructura. |

### Paquete publicado (`site/v1/` en `app.fs`: `files/site/v1/` en local, `s3://{bucket}/site/v1/` en producción)

```
site/v1/manifest.json                                    único puntero; se reescribe al final de cada publicación
site/v1/runs/{scope}/history.json                        reconstruido desde los headline.json de los runs
site/v1/runs/{scope}/{run_id}/meta.json headline.json series.json polls.json fan.json house-effects.json dispersion.json
site/v1/runs/{scope}/{run_id}/{nowcast|forecast}/{vote,summary,dist,districts,scenario,projection}.json
site/v1/runs/{scope}/{run_id}/csv/*.csv                  gemelos CSV en formato largo
site/v1/analysis/{scope}/{meta,house-effects,herding,poll-errors,error-data}.json (+ csv/)
site/v1/events/{scope}/{date}/{meta,pollsters,polls,results,model}.json (+ csv/)
site/v1/backtest/{scope}/{meta,metrics,by-horizon,blocks,shares,seats}.json (+ csv/)
```

- `run_id` = `YYYYMMDD-HHMMSS` en UTC, uno por invocación (compartido por todos los ámbitos); cabe en el
  alias de ruta y ordena lexicográficamente. Un run nunca se reescribe.
- Sobre común de todo JSON: `{"schema": "<nombre>@1", "contract": 1, "scope", "run_id"|null, "mode"|null,
  "generated_at", "data": ...}`. Convenciones: `null` por NaN, fechas `YYYY-MM-DD`, instantes ISO con `Z`,
  partidos por `name` con catálogo (`id`, `fullname`, `color`, `block`, `regional`) en `meta.parties`,
  series largas en orientación columnar, tablas en `records`, porcentajes con 2 decimales y
  probabilidades con 3.
- `meta.json`: `run_id`, `run_at`, `commit`, `dirty` (`git status --porcelain`), `versions`, `scope`,
  `event_date`, `as_of`, `date_last`, `date_fit_last`, `horizon_max`, `n_sim`, `seed`, `drange`, `max_fc`,
  `alpha`, `correctors`, `n_polls`, `n_pollsters`, `db_polls`, `db_last_poll`, `n_seats`, `majority`,
  `parties`, `bmaps`, `smap`, `regions`, `diagnostics` (`drift_k`, `multiplier`, `ages`, `composition`,
  `clip_rate`), `seconds` por paso, `freeze`.
- `headline.json`: por modo, `pct/lo/hi/seats/seats_lo/seats_hi/p_first` por partido y `p_majority` por
  bloque `vs`; alimenta `history.json` (evolución del pronóstico publicado, no regenerable).
- Fuentes: `series` ← `fc.forecast` + `fc.fc_stat` (`cmin`/`cmax`) cortados en `date_fit_last`; `polls` ←
  `fc.fc_series_raw` (sondeos como se publicaron) + `fc.nfc_series` (resultado anterior); `fan` ←
  `sim.fan()`; `house-effects` ← `fc.house_effects`; `dispersion` ← `fc.dispersion`; por modo: `vote` ←
  `sim.vote_forecast()`, `summary` ← `sim.summary()`, `summary('vs')`, `summary('blocks')`,
  `probabilities('vs')`, `totals()`; `dist` ← `sim.dist()` (matriz n_sim × partidos, ~11 KB gz, base de
  histogramas y coaliciones en el navegador); `districts` ← `sim.unit_summary(r)` por distrito; `scenario`
  ← `sim.result(sim.scenario())`; `projection` ← `sim.projection()` (también `vs` y `blocks`).
- `analysis/{scope}` (cambia tras elecciones; se sobrescribe): `house-effects` ← `Computer.print_house_effects`
  / `house_effects_summary`; `herding` ← `herding_summary` + `get_herding`; `poll-errors` ←
  `get_polls_metric('error', bmap='vs', drange=(6, 42), n_last=1)`; `error-data` ← `get_error_estimator_data`.
- `events/{scope}/{date}` (manual tras resultados): tablas de PollstersEvent (`get_polls_metric` del evento,
  todas y última por casa), resultados región 0, filas del backtest de ese evento (sustituye la rama
  `show_fc` rota del notebook). `backtest/{scope}`: conversión de `backtest/results[/{scope}]/*.csv`.
- Retirada: `point` reescribe la entrada del ámbito en `manifest.json` a un run anterior (no destructivo);
  `unpublish` borra un run malo y reconstruye `history`. Las cachés caducan en ≤ 2 min.

### Comando de publicar

```
python job.py publish '{"what":["forecast"],"scopes":["es"]}'
python job.py publish '{"what":["forecast"],"scopes":"all"}'                 # es + autonómicos con sondeos
python job.py publish '{"what":["analysis","backtest"],"scopes":"all"}'
python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"2023-07-23"}'
python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'   # a site-dry/, sin punteros
```

Parámetros: `what`, `scopes`, `event_date` (por defecto `get_next_event_date`), `n_sim=1000`, `seed=42`,
`drange=6`, `max_fc=10`, `alpha=0.05`, `correctors`, `freeze`, `run`, `dry_run`, `force`. En Docker:
`RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}'` (JSON sin espacios: `init.sh` no entrecomilla).

Pasos por ámbito: `Simulator(scope, event_date, drange, seed, mode='nowcast', verbose=0)` →
`fit_forecast(max_fc)` → `run(split=True, random=True, n_sim)` → exporta `nowcast/` → `sim.mode='forecast'`
→ `run(..., horizon='deadline')` (misma semilla: `set_params` recrea el RNG) → exporta `forecast/` →
ficheros comunes, CSV, `meta.json` y por último `headline.json` → `validate` de cada objeto antes de
escribir → `history.json` → al terminar todos los ámbitos, `manifest.json` (los ámbitos fallidos conservan
su entrada anterior). `ValueError` de `require_forecast` (ámbito sin sondeos) → `skipped` con motivo;
cualquier otra excepción → `failed`, el resto continúa; resumen en pantalla, `app.logger` y `alert()`;
código de salida 1 si hay fallos. Guarda LOREG: con `scope='es'` y fecha dentro de los 5 días previos a
la elección se niega a publicar salvo `force`. Duración estimada: `es` 1,5-2 min; autonómico 0,5-1 min;
`all` 10-15 min desde el portátil contra la RDS (medir en la fase 1).

### Contrato de API (`/api/v1`, solo GET, JSON; `?format=csv` donde hay gemelo)

| Ruta | Fuente | Contenido |
|---|---|---|
| `/api/v1/health` | estático | `{status, contract}` sin tocar base ni S3 (HEALTHCHECK) |
| `/api/v1/manifest` | paquete | puntero, ámbitos, `freeze`, atribución |
| `/api/v1/scopes` | CSV `data/` + paquete | catálogo de ámbitos cruzado con `manifest` (`simulable`, `latest`) |
| `/api/v1/forecast/{scope}` | paquete | `meta.json` del último run o de `?run=` |
| `/api/v1/forecast/{scope}/runs` | paquete | `history.json` |
| `/api/v1/forecast/{scope}/{part}` | paquete | `part` ∈ meta, headline, series, polls, fan, house-effects, dispersion |
| `/api/v1/forecast/{scope}/{mode:str}/{part}` | paquete | `mode` ∈ nowcast, forecast; `part` ∈ vote, summary, dist, districts, scenario, projection |
| `/api/v1/analysis/{scope}/{part}` | paquete | `part` ∈ meta, house-effects, herding, poll-errors, error-data |
| `/api/v1/events/{scope}` | base + paquete | eventos del ámbito con `featured` y `has_page` |
| `/api/v1/events/{scope}/{date}/{part}` | paquete | `part` ∈ meta, pollsters, polls, results, model |
| `/api/v1/backtest/{scope}/{part}` | paquete | `part` ∈ meta, metrics, by-horizon, blocks, shares, seats |
| `/api/v1/parties` | base | `get_parties()` |
| `/api/v1/pollsters` | base | `get_pollsters()` |
| `/api/v1/pollsters/{id:num}` | base | perfil: ratings por elección (camino al rating), `pollsters_parties`, `pollsters_herding` |
| `/api/v1/ratings?scope=&event=` | base | `get_ratings` (por defecto el próximo evento) |
| `/api/v1/polls?scope=&event=&pollster=&limit=` | base | sondeos con columna por partido y pesos/errores (tope 5.000) |
| `/api/v1/results/{scope}/{date}?region=` | base | `get_event_results` (región 0 por defecto) |
| `/api/v1/house-effects?scope=`, `/herding?scope=`, `/drift?scope=` | base | tablas `pollsters_parties`, `pollsters_herding`, `drift` |

Reglas: las rutas fijas se registran antes que las genéricas (primera coincidencia gana,
`mtpy/core/api.py:256-265`); toda entrada pasa por listas cerradas antes de llegar a `mtpy/lib/data.py`
(ámbito en `get_scopes()`, fecha ISO válida, `run` con `^\d{8}-\d{6}$`, `part`/`mode` en listas, ids
enteros). Errores `{"status":"error","message"}` con 400/404/503/500. Cabeceras: `Cache-Control:
public, max-age=60` (punteros y base), `max-age=31536000, immutable` con `?run=` explícito,
`Access-Control-Allow-Origin: *` (solo GET, sin preflight). CSV `text/csv; charset=utf-8` con
`Content-Disposition: attachment`.

### Servicio y caché

- Caché por proceso en `webapi.py` (4 workers = 4 cachés): `manifest.json`, `history.json`, `analysis/*`,
  `events/*` y respuestas de base con TTL 60 s (`web.cache_ttl`); ficheros de run inmutables en LRU de
  200 entradas sin caducidad. nginx `proxy_cache` 60 s absorbe el tráfico restante.
- Consultas síncronas dentro de acciones `async` (decenas de ms); `asyncio.to_thread` solo si el p95 lo
  pide (seguro: `Controller.__init__` captura `request`/`response`; nunca leer `self.app.request` tras un
  `await`).
- Base: el contenedor usa el usuario de PostgreSQL de `deploy/elections.env` (decisión del 2026-10-08: sin
  rol de solo lectura por ahora; endurecimiento opcional más adelante).

### Cambios en el núcleo (`mtpy/core/api.py`, con tests)

1. `Response.set_content` (`:121-151`): `list`/`tuple` → JSON; `json.dumps(..., ensure_ascii=False,
   allow_nan=False, default=json_default)`; si la acción ya fijó `content-type`, `str`/`bytes` no lo
   sobrescriben (CSV). Añadir 422 y 503 a `status_codes` (`:73-82`). `Response.set_cache(seconds,
   immutable=False)` y `set_etag`.
2. `Request.query` (propiedad): dict con el primer valor de cada clave (`:37-43` devuelve tuplas).
3. `Controller.dispatch` (`:300-310`): `try/except`; `HttpError` → código y mensaje; otra excepción →
   `app.logger.error(traceback)` y 500 JSON (hoy sube hasta uvicorn sin log).
4. `Router.handle` rama `not_found` con prefijo (`:267-273`): `self.params = {}` (bug latente).
5. `Api.__call__` (`:12-22`): responder `lifespan.startup/shutdown` y volver si `scope['type'] != 'http'`
   (hoy `KeyError` en `Request.__init__`, `:33`).
6. `mtpy.api(routes=None)` en `mtpy/mtpy.py:100-111`.

Pospuesto: despachar sobre variables locales en vez del estado del router (necesario solo con
`to_thread`), soporte `HEAD`.

### Frontend (`web/`, multipágina, sin build)

- Estado en la query string (`?scope=es-md&mode=forecast&group=vs&run=...`); nginx `try_files $uri
  $uri.html $uri/ =404` da URLs limpias (`/escanos?scope=es`). Sin router JS (elegido frente a SPA por
  simplicidad).
- (enmienda 2026-10-09) Las URLs limpias y el estado ya no los resuelven nginx ni `state.js`: cada página
  es una ruta de un controlador de Python (`mtpy/controllers/`, `Page`) que lee `scope`/`mode`/`run` de la
  query string, renderiza una plantilla Jinja2 de `web/templates/` con los datos del paquete y embebe el
  contexto en `<script type="application/json" id="initial-data">`. Los controles son un formulario GET
  (ámbito, "Hoy"/"Elección", botón "Ver" sin JS); el JS solo dibuja los gráficos a partir de
  `initial-data`, sin `fetch`. `state.js`, `layout.js` y `api.js` desaparecen y los recursos pasan de
  `web/` a `web/dist/`. Ver `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md`.
- Páginas: `/` portada (V1, V2, hemiciclo, p_mayoría, evolución del `headline`), `/promedio` (V3, V4),
  `/escanos` (V6, V7, V8, calculadora de coaliciones sobre `dist`), `/casas` (V10, V5, herding; `?id=`
  perfil V11), `/elecciones` (lista; `?date=` página V13; V12), `/modelo` (V14), `/metodo` (V15, texto).
- (enmienda 2026-10-09: `api.js` y `state.js` ya no existen, ver arriba; el resto de módulos pasa a
  `web/dist/js/`.)
- Módulos: `api.js` (fetch + caché en `Map`, propaga `run`), `state.js` (query string y selectores de
  ámbito/modo rellenados desde `/api/v1/manifest`), `format.js` (`Intl` es-ES), `catalog.js` (colores y
  nombres por partido desde `meta.parties`; bloque = color del primer partido; Otros gris), `seats.js`
  (de `dist` a histogramas, cuantiles y `P(suma ≥ mayoría)` de cualquier coalición), `charts/*.js`
  (banda + línea + puntos de sondeos, barra apilada 100 %, histogramas por partido, abanico, diana,
  barras divergentes, mapa de calor casa × partido, hemiciclo), `pages/*.js`.
- Librería: **ECharts 5** vendorizada (~1 MB, ~330 KB gz; un fichero, canvas, tooltips y zoom táctil;
  cubre todas las figuras, hemiciclo con `pie` `startAngle: 180, endAngle: 0`). Elegida frente a
  Plotly.js (más pesado) y Observable Plot (más ligero pero sin zoom ni tooltips ricos); el contrato no
  cambia si se sustituye.
- Pie de cada página: run activo, "Estimación a {as_of}", "último sondeo {date_last}", commit (y "sucio"
  si procede), atribución de fuentes. Banner si `manifest.freeze.active` y, por defecto, ocultación de las
  vistas derivadas de sondeos (Luis decide si lo activa).
- Móvil: una columna < 720 px, `chart.resize()` con `ResizeObserver`, tablas con scroll horizontal.

### Infraestructura

- `.dockerignore` con semántica de Docker: `.git`, `.superpowers/`, `notebooks/`, `docs/`, `files/`,
  `log/`, `tests/`, `**/*.env`, `**/env.*`, `**/__pycache__/`, `**/.DS_Store`, `**/.ipynb_checkpoints/`,
  `*.code-workspace`; sin `*.csv` global. Además el `Dockerfile` sustituye `COPY . .` por una lista
  explícita (`api.py job.py config.json requirements.txt mtpy/ data/ web/ deploy/docker/`): aunque
  `.dockerignore` fallara, las credenciales no entran.
- `Dockerfile`: `PYTHONUNBUFFERED=1`, `pip install --no-cache-dir`, apt solo `nginx supervisor curl
  ca-certificates libpq5` (comprobar en la fase 0 que todos los pins tienen wheel; si no, mantener
  `gcc libpq-dev`), `HEALTHCHECK` sobre `/api/v1/health`, `ARG GIT_COMMIT` → `ENV`, sin supercronic ni
  crontab (job inexistente). Multietapa con venv: opcional, fase 6.
- (enmienda 2026-10-09) `deploy/docker/nginx.conf` ya no tiene `root`, `index` ni `try_files`: sirve solo
  `location ^~ /dist/vendor/` (un año) y `location ^~ /dist/` (se revalida) con `alias` a
  `/app/web/dist/...`, y `location /` (antes `/api/`) hace de proxy de las páginas y de la API a gunicorn
  con el mismo `limit_req` y `proxy_cache`. Lo que sigue es el diseño original.
- `deploy/docker/nginx.conf`: `root /app/web; index index.html`; `location = /healthz { return 200 }`;
  `location /api/` con `limit_req` (20 r/s, burst 40), `proxy_cache` (60 s, `use_stale`), cabeceras
  `X-Forwarded-*`, `proxy_pass http://app_server` sin barra final; `gzip_types` para JSON, JS, CSS, CSV y
  SVG; `/vendor/` `immutable`; `try_files $uri $uri.html $uri/ =404`; logs a stdout/stderr;
  `X-Content-Type-Options`, `Referrer-Policy`, CSP `default-src 'self'`; se eliminan
  `client_max_body_size 4G`, `proxy_buffering off` y `ssl_protocols` (TLS lo termina el proxy del
  servidor).
- `supervisord.conf`: `nodaemon=true`, `logfile=/dev/null`, programas con `autostart=%(ENV_SV_NGINX)s` /
  `%(ENV_SV_GUNICORN)s` / `%(ENV_SV_WORKER)s`, logs a `/dev/stdout`; gunicorn `-w
  %(ENV_GUNICORN_WORKERS)s` (defecto 2, cada worker carga ~400 MB) `--access-logfile - --error-logfile -
  --timeout 60`. `init.sh`: calcula `SV_*` desde `APP_API`/`APP_QUEUE`, exporta `GUNICORN_WORKERS` y hace
  `exec supervisord`; con `RUN_JOB`, `exec python /app/job.py $RUN_JOB`. Así `docker logs` muestra nginx y
  gunicorn y `docker stop` para limpio.
- Configuración: `deploy/elections.env` (existente, ignorado por git y excluido de la imagen) es el único
  fichero de entorno, para la web, la publicación y los jobs: usuario de PostgreSQL y usuario de AWS/S3
  actuales (decisión del 2026-10-08). Plantilla: `.env.example` en la raíz. Sin usuarios IAM separados por
  ahora; el usuario de AWS necesita Get/List/Put/Delete sobre `site/*` (`unpublish` y `check_s3`).
- `requirements.txt`: `gunicorn==20.1.0` → `23.0.0` (CVE-2024-1135/6827); `uvicorn==0.18.3` se mantiene
  (conserva `uvicorn.workers.UvicornWorker`). Poda de paquetes no importados: opcional, fase 6.
- Procedimiento reproducible (`deploy/README.md`): `docker build --build-arg GIT_COMMIT=$(git rev-parse
  HEAD) -t elections-web:$(git rev-parse --short HEAD) .` → `docker run -d --restart unless-stopped -p
  127.0.0.1:8042:8042 --env-file deploy/elections.env elections-web:<sha>` → `curl /healthz`,
  `/api/v1/manifest`. Sin S3: `S3_BUCKET=` y `-v $PWD/files:/app/files`. Publicar desde el portátil
  exportando `deploy/elections.env`, o `docker run --rm --env-file deploy/elections.env -e RUN_JOB=...`.

### Seguridad

- Credenciales nunca en la imagen ni en git; mover los `.env` reales fuera del repo
  (`~/.config/elections-model/`, modo 600). Rotar ahora el par de claves AWS compartido por
  `deploy/docker.env` y `deploy/elections.env`, y la contraseña de la RDS (han estado a un `docker build`
  de acabar en una imagen).
- (2026-10-08) Sin rol de solo lectura ni usuarios IAM separados: la web y la publicación usan las
  credenciales de `deploy/elections.env`. Grupo de seguridad de la RDS limitado al
  servidor y al portátil; `sslmode=require` si el adaptador lo admite (comprobar `mtpy/core/dal/PostgreSQL.py`).
- Validación en listas cerradas antes de cualquier SQL; `limit_req` en nginx; solo GET.
- Datos publicados con atribución (Wikipedia CC BY-SA 4.0; "Origen de los datos: Ministerio del
  Interior") en `manifest.attribution` y pie de página.
- Copias: backups automáticos de la RDS (14 días) y `pg_dump -Fc -n elections` semanal a S3 durante la
  campaña (recomendación; los datos maestros no se pueden reconstruir desde el repo).

### Pruebas

- (enmienda 2026-10-09) `tests/test_pages.py` cubre las páginas renderizadas por los controladores
  (contenido, cabeceras, páginas de error, plantillas con `StrictUndefined`, filtros frente a
  `format.js`, orden de las claves de `initial-data`); `tests/test_web_routes.py` pasa a comprobar que no
  hay literales `/api/v1` en el JS, que los recursos referenciados existen sin scripts en línea, que los
  imports del JS resuelven y que ECharts está fijado por versión; `deploy/smoke.sh` y `check-api.sh` usan las rutas nuevas.
- `tests/test_bundle_unit.py`: `run_id()` casa con el alias de ruta; `json_default` con `np.int64`, NaN,
  `Timestamp`, `NaT`; `validate` acepta fixtures de `tests/fixtures/bundle/` y rechaza claves ausentes.
- `tests/test_publish_unit.py` (sin base): exportadores sobre un `Simulator` reconstruido con `__new__` y
  arrays sintéticos (patrón `tests/test_simulator_unit.py:438`): esquema, `null` por NaN, cada fila de
  `dist` suma `n_seats`; `publish_forecast` con un `FakeSim` y `BundleWriter(FileSystem(tmp_path))` crea
  la disposición, escribe `headline.json` antes de los punteros, `history.json` idempotente;
  `export_meta` con `subprocess` parcheado; `point`, `unpublish`, `freeze`, guarda LOREG con fecha simulada.
- `tests/test_webapi_unit.py`: validadores (rechazan `'`, `;`, ámbitos fuera del CSV), TTL con reloj
  inyectado, LRU, `frame_to_records`, `frame_to_csv`, lectura de un paquete en `tmp_path`.
- `tests/test_api_asgi.py`: arnés ASGI de 20 líneas sin `mtpy.run()` (`App.get_()`, `app.set('fs',
  FileSystem(tmp_path))`, `set('logger', ...)`, `set('router', ...)`), scope `http` y `send` falsos:
  JSON y cabeceras, 400 en ámbito inválido, 404 en run inexistente, CSV con `content-type`, lista JSON,
  `not_found` sin arrastrar `params`, `Index` sigue devolviendo `API Home`; rutas de base con los
  `query_*` parcheados.
- `tests/test_web_routes.py`: cada literal `/api/v1/...` de `web/js/**/*.js` (plantillas `${x}`
  sustituidas) casa con un patrón de `ROUTES`; cada `<script src>` y `<link>` de `web/*.html` existe.
- `tests/integration/test_publish.py` (marcado): `publish_forecast('es', writer=FileSystem(tmp_path),
  n_sim=20, seed=42)`: esquema válido, `dist` suma 350, probabilidades en [0, 1], partidos ⊆ `params.json`,
  `vote_forecast` tras `sim.mode='forecast'` coincide con un `Simulator(mode='forecast')`; dos ejecuciones
  con la misma semilla dan `dist` idéntico y dos entradas en `history`.
- `deploy/smoke.sh`: build, run con `--env-file`, `curl` a `/healthz`, `/api/v1/health`, `/api/v1/manifest`,
  `/`, `/escanos?scope=es`; comprueba que la imagen no contiene `*.env`; `docker stop` en < 10 s.
- `pytest -m "not integration"` sin base; `FutureWarning` es error (evitar `applymap` y `groupby.apply`
  sin `include_groups` en los exportadores).

## Fases (hoy 2026-10-07; elección 2026-11-29)

| # | Fase | Entregables | Depende | Esfuerzo |
|---|---|---|---|---|
| 0 | Cimientos y seguridad | commit de los cambios M13 pendientes; `.dockerignore` y `Dockerfile` con `COPY` explícito; spike `s3fs==0.4.2` (write/read/listdir/exists contra el bucket desde la imagen; si falla, subir pin o `S3` sobre `boto3`); cambios de `mtpy/core/api.py` + `mtpy.api(routes=)` con tests; rol `web_reader` e IAM (Luis); `web.env.example`; gunicorn 23 | — | 1,5-2 d |
| 1 | Paquete y comando | `bundle.py`, `publish.py`, `jobs/Publish.py`, tests unitarios e integración; `es` publicado en `files/site/v1` y en S3; medición de duración contra la RDS | 0 | 3-4 d |
| 2 | API y sitio mínimo (primer despliegue) | `webapi.py`, `Base`/`Forecast`, nginx, supervisord e `init.sh`, `index.html`, `promedio.html`, `api.js`, `state.js`, ECharts, `smoke.sh`; producción con `es` (≈ 20-22 oct) | 1 | 3-4 d |
| 3 | Escaños, autonómicos, histórico | `escanos.html` (V6-V8, coaliciones), selector de ámbito con `scopes: all`, evolución del `headline`, CSV | 2 | 3-4 d |
| 4 | Casas | `Data` y `Analysis`, `publish analysis`, `casas.html` (ranking, perfil y camino al rating, efectos de casa, herding) | 2 | 3-4 d |
| 5 | Elecciones y modelo | `publish event` y `backtest`, `elecciones.html`, `modelo.html`, `metodo.html`; eventos `es` 2015-2023 | 4 | 2-3 d |
| 6 | Endurecimiento | prueba de carga con `proxy_cache`, ensayo de `point`/`unpublish`, ensayo de `freeze`, backups, `deploy/README.md` final, poda de requirements y multietapa opcionales; cerrar antes del 24-11 | 2-5 | 2 d |

Total ≈ 18-23 días de trabajo. Tras la fase 2 hay web pública; las fases 4 y 5 son independientes entre sí.

## Verificación

- Fase 0: `pytest -m "not integration"` en verde; `docker build` termina y `docker run --rm img ls
  /app/deploy` no lista `*.env`; spike S3 escribe y lee `site/v1/_smoke.json`.
- Fase 1: `python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'` →
  inspeccionar `files/site-dry/`; `pytest -m integration -k publish`; `vote` y `summary` coinciden con las
  tablas de PollsSimulations para la misma semilla; publicación real a S3 y lectura de `manifest.json`.
- Fase 2: `uvicorn api:api --port 8000` y `curl` de cada ruta de la tabla (JSON válido, cabeceras,
  404/400); `deploy/smoke.sh`; recorrer portada y promedio en escritorio y móvil sin errores de consola;
  `docker logs` muestra nginx y gunicorn.
- Fases 3-5: cada vista comparada con la figura equivalente del notebook (mismos números); `pytest`
  completo; `test_web_routes` en verde.
- Fase 6: 200 peticiones concurrentes a `/api/v1/forecast/es/nowcast/summary` con `X-Cache: HIT`;
  `point` a un run anterior visible en ≤ 2 min; `freeze` activado y desactivado; restauración de un
  `pg_dump` en la base local.

## Riesgos

- `s3fs 0.4.2` (2020) con `fsspec`/`botocore` actuales nunca ejercitado: spike en la fase 0; plan B `S3`
  sobre `boto3` con la misma interfaz, cambio contenido en `mtpy/core/services/s3.py`.
- RDS: puede no tener el esquema `elections` o estar desactualizada → migración inicial con `pg_dump
  --schema=elections` local y `pg_restore` (Luis); latencia portátil → RDS en cargas y publicaciones
  (medir; publicar `es` a diario y `all` semanalmente si hace falta).
- Árbol sucio (M13 sin commit): `meta.dirty` lo delata; commitear en la fase 0; etiquetar
  (`git tag web-YYYYMMDD`) antes de publicar en campaña.
- Memoria: 4 workers × pila científica ≈ 1,6 GB → `GUNICORN_WORKERS=2`.
- `as_of` posterior a hoy (`date_fit_last` = último sondeo + 10): la portada muestra "Estimación a
  {as_of}" y "último sondeo {date_last}".
- LOREG 24-28 nov: interruptor listo en la fase 6; decisión editorial de Luis.
- Plazo: 53 días; tras la fase 2 (≈ día 11) ya hay sitio público.

## Preguntas abiertas (solo Luis; no bloquean el inicio)

- Dominio y TLS: ¿qué proxy hay delante del puerto 8042 en el servidor?
- Política durante la veda: congelar, ocultar vistas de sondeos o seguir con aviso.
- Agrupación por defecto en `es` (`max` o `main`); en autonómicos solo hay `main/blocks/vs`.
- Casas destacadas en los paneles (hoy CIS, GAD3, GESOP y 40dB fijos) o por rating.
- ¿Publicar el cubo de escaños por provincia (~53 KB gz) o bastan `districts` + `scenario`?
- Diagnósticos de experto (deriva, composición, `clip_rate`) visibles o solo en `meta.json`.
- Cadencia de publicación prevista (diaria `es`, semanal `all`).
- Prefijo S3 `site/v1` en el bucket actual o bucket aparte; nombre del rol de solo lectura.
- Texto de atribución y licencia de la curación propia.

## Siguiente paso tras la aprobación

Guardar este diseño como especificación en `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`,
guardar en memoria la corrección sobre `mtpy/controllers` y `mtpy/lib`, y generar el plan de
implementación por tareas con la skill `writing-plans`, empezando por la fase 0.

## Estado al cierre de la fase 0 (2026-10-08)

Hecho en `dev` (commits 752f051 a 4d12ce4), con 192 tests unitarios en verde:

- `mtpy/core/api.py`: `Request.query`; `json_default` (numpy, NaN y NaT → `null`, `datetime64` → ISO,
  `timedelta64` rechazado); `Response.set_cache`/`set_etag`/`_set_error`; claves de cabecera en minúsculas;
  422 y 503; `set_content` con bool, int, list, tuple y content-type explícito; `HttpError`;
  `Controller.dispatch` con try/except y log; toda respuesta de error es `no-store` sin `etag` ni
  `content-disposition`; `lifespan` ASGI; `Router.handle` sin parámetros residuales.
- `mtpy/mtpy.py`: `api(routes=None)` (la fase 2 inyecta `webapi.ROUTES` desde `api.py`).
- Tests: `tests/test_api_asgi.py` (arnés ASGI sin servidor ni base, 30 tests), `tests/conftest.py`
  (fixture `fresh_app`), `tests/test_jobs_check_s3.py` (4 tests).
- `mtpy/jobs/CheckS3.py`: `python job.py check_s3` comprueba `app.fs` sobre S3 (rechaza `files/` local
  salvo `{"allow_local": true}`).
- `deploy/README.md`: construir, arrancar y comprobar S3 con `deploy/elections.env`.
- `.dockerignore`: exclusión de `deploy/*.env`, `.git` y material de claves (único cambio de imagen que
  se conserva; el resto del Dockerfile es el original de Luis, con cron).

Decisiones posteriores a la aprobación: `deploy/elections.env` es la única configuración (sin rol
`web_reader` ni usuarios IAM nuevos); imagen Docker revertida a la original. Las secciones
"Infraestructura" y "Seguridad" de arriba se leen con esas dos salvedades.

Diferido a las fases 1-2 (hallazgos menores de las revisiones): guard para `np.longdouble` en
`json_default`; política de NaN (los arrays limpian a `null`, los `float` nativos dan 500: el bundle debe
limpiar NaN antes de serializar); mover `json_default` a `mtpy/core/utils/`; `send()` fuera del `try` de
`dispatch`; docstring de `Api.__call__`; `HEAD`; crontab `job.py rep` inexistente; `APP_CRONTAB=` en
`.env.example`; HEALTHCHECK y `GIT_COMMIT` quedaron fuera con la reversión.

Pendiente de Luis antes de publicar: permisos Get/List/Put/Delete del usuario AWS de
`deploy/elections.env` sobre el bucket; ejecutar `set -a; . deploy/elections.env; set +a; python job.py
check_s3` (si falla, la fase 1 decide entre subir `s3fs` o reimplementar `S3` sobre `boto3`); confirmar
el esquema `elections` en la RDS.

Siguiente: plan de la fase 1 (`mtpy/lib/bundle.py`, `mtpy/lib/publish.py`, `mtpy/jobs/Publish.py`, tests
unitarios e integración, primer paquete de `es` en `files/site/v1` y en S3, medición de duración contra
la RDS), a partir de las secciones "Paquete publicado" y "Comando de publicar".

## Estado al cierre de la fase 1 (2026-10-08)

Hecho en `dev` (commits `30a6369..9471dfa` más el commit de documentación), con 249 tests unitarios en
verde (`python -m pytest -m "not integration" -q`: 192 de antes de la fase y 57 nuevos) y 5 de integración
en `tests/integration/test_publish.py` (`python -m pytest -m integration -k publish -v`, 247 s contra la RDS):

- `mtpy/core/utils/serialize.py`: `json_default` (reexportado desde `mtpy/core/api.py`).
- `mtpy/lib/bundle.py`: rutas, sobre, `SCHEMAS`, `validate`, `BundleReader` y `BundleWriter`.
- `mtpy/lib/publish.py`: exportadores, `publish_forecast`, `rebuild_history`, `update_manifest`, `point`,
  `unpublish`, `loreg_guard`, `resolve_scopes`, `provenance`, `ATTRIBUTION`, `DEFAULT_CORRECTORS`.
- `mtpy/jobs/Publish.py`: `python job.py publish` con `what` ∈ `forecast`, `manifest`, `point`, `unpublish`.
- Tests: `tests/fakes.py`, `tests/__init__.py`, `tests/test_serialize_unit.py`, `tests/test_bundle_unit.py`,
  `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`, `tests/integration/test_publish.py`.
- Documentación: `docs/web/contrato.md` (contrato del paquete) y la sección "Publicar" de `deploy/README.md`.

Mediciones del primer paquete LOCAL de `es` (2026-10-08, contra la RDS del `.env` de la raíz, paquete en
`files/site/v1` con `S3_BUCKET=` vacío): run `20261008-181100`, `n_sim=1000`; `seconds`: init 35,5, ajuste
42,5, nowcast 21,0, forecast 21,5, exportación 0,8, total 121,2 (la spec estimaba 1,5-2 min para `es`).
Usó 315 sondeos de 23 casas (613 sondeos del evento en la base, último 2026-10-03), `as_of` 2026-10-13,
`horizon_max` 47; la carpeta del run pesa 1,4 MB y `nowcast/dist.json` 33,7 KB. Ensayo con `n_sim=20`:
86 s, 19 JSON y 17 CSV, sin manifest. Las duraciones de `scopes: "all"` contra la RDS no se midieron.

Decisiones tomadas durante la ejecución:

1. `json_default` pasa a `mtpy/core/utils/serialize.py` (reexportado desde `mtpy/core/api.py`) con guarda
   para `np.longdouble`: los escalares de coma flotante siempre salen como `float` nativo.
2. `tests/__init__.py`: `tests` es ahora un paquete regular, porque IPython (importado por
   `mtpy/core/utils/dataviz.py`) pone en `sys.path` un paquete llamado `tests` que tapaba al paquete
   de espacio de nombres. Los dobles de prueba viven en `tests/fakes.py` (`synthetic_simulator`,
   `FakeSimulator`, `RecordingFileSystem`).
3. Detalles del contrato fijados en código: en las filas de `districts`, `seats` es la mediana de los
   escaños simulados redondeada a 1 decimal (un estadístico, como `seats_mean`); en `summary` y
   `headline`, `seats` es el titular entero (`totals()`), `null` en un bloque sin partidos. `dist` lleva
   `n_seats`, `polls` lleva `parties` y `meta` lleva `scope`.
4. `publish_forecast` no escribe punteros: el job reconstruye `runs/{scope}/history.json` de cada ámbito
   publicado y después reescribe `manifest.json` (defecto del plan hallado en el primer paquete real:
   el job solo escribía el manifest).
5. `what` ∈ `forecast`, `manifest`, `point`, `unpublish`; `analysis`, `event` y `backtest` lanzan
   `ValueError` indicando las fases 4, 5 y 5.
6. `meta.freeze` = publicado dentro de la ventana LOREG con `force`; `manifest.freeze` es el interruptor
   editorial.
7. `ATTRIBUTION` (`mtpy/lib/publish.py`) es provisional (pregunta abierta de esta spec) y, como el manifest
   conserva la atribución almacenada, cambiar la constante no alcanza a un manifest existente.
8. `provenance` ejecuta git en la raíz del repositorio; en Docker (sin `.git`) `commit` cae a la variable
   `GIT_COMMIT` o a `null`.
9. Para publicar en local con `S3_BUCKET` definido en el `.env` de la raíz hay que anteponer `S3_BUCKET=`
   (vacío): `mtpy.run()` construye entonces el sistema de ficheros `files/` (`load_dotenv` no pisa las
   variables exportadas). La publicación real al bucket es de Luis.
10. Los trailers de los commits nombran el modelo que escribió cada commit (Sonnet en los de implementación).

Pendientes de la revisión final de la rama: redondeos de tiempos en `seconds`, `attribution` no se refresca
en un manifest existente, validación de `event_date` antes del SQL de `db_stats`, docstrings y pistas de tipo.

Revisión final de la rama (2026-10-08). Una sola tanda de correcciones, con tests:

1. `point` y `unpublish` validan `scope` (`ROUTE_ALIAS_RE`) y `run_id` (`RUN_ID_RE`) antes de formar
   rutas: `unpublish(writer, 'es', '/')` borraba todos los runs del ámbito.
2. Semántica de fallo: `publish.NothingToPublish(ValueError)` solo envuelve la construcción del
   `Simulator` y `fit_forecast` (→ `skipped`); cualquier otro error, `ValueError` incluido (validación del
   paquete), es `failed` y el job sale con error.
3. `read_manifest` solo usa el manifest por defecto ante `FileNotFoundError` al leer, no con `exists`
   (fsspec devuelve `False` ante cualquier error y el manifest real se habría pisado).
4. La guarda LOREG toma la fecha de hoy en Europe/Madrid (UTC+1 fijo con aviso si falta `tzdata`) y
   acepta `datetime` además de `str` y `date`.
5. `seconds`: `nowcast` y `forecast` miden solo las simulaciones, `export` suma todas las escrituras; todo
   a 1 decimal.
6. Cada escritura del manifest refresca `attribution` desde `ATTRIBUTION` (deja sin efecto la reserva de
   la decisión 7).
7. Contrato: las filas de `fan` llevan `name` (antes `party`); `meta.regions` incluye el total del ámbito
   (id 0) y `districts.regions` solo las circunscripciones.
8. `scenario` es la simulación determinista más cercana a los escaños centrales (L1 a la mediana), no una
   al azar (docstring y contrato).
9. Docstring del módulo `publish` reescrito: exportadores puros más la orquestación que escribe, consulta
   la base y ejecuta git.
10. El job normaliza `event_date` con `date.fromisoformat(...).isoformat()` por ámbito (un valor no ISO es
    `failed`), así que `meta`, `headline` y el SQL de `db_stats` solo ven `YYYY-MM-DD`.
11. `deploy/README.md`: `today` es solo para pruebas y ensayos (sin huella en `meta.freeze`).

Tras la tanda: 262 tests unitarios en verde (`python -m pytest -m "not integration" -q`; 13 nuevos).

Pendiente de Luis: permisos AWS del bucket y `check_s3`; primera publicación a S3 contra la RDS
(`set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["forecast"],"scopes":["es"]}'`)
y medición de `scopes: "all"` (estimación 10-15 min; ver qué ámbitos quedan `skipped`); texto definitivo de
`ATTRIBUTION` y licencia de la curación propia; opcionalmente, contraste en el notebook con `n_sim=10000`.

Siguiente: plan de la fase 2 (API y sitio mínimo).

## Estado al cierre de la fase 2 (2026-10-08)

Hecho en `dev` (plan en `d6fb790`, código en `207cd46..8e0a116` (10 commits) más los commits de
documentación y el de la revisión final), con 301 tests unitarios en verde (`python -m pytest -m "not
integration" -q`: 262 del cierre de la fase 1 y 39 nuevos: `tests/test_webapi_unit.py` 10,
`tests/test_web_api.py` 22, `tests/test_web_routes.py` 4 y 3 añadidos a los de la fase 1) y los 5 de
integración sin cambios:

- `mtpy/lib/webapi.py`: constantes, validadores (`check_scope|run|mode|part|format`), `settings()`,
  `TTLCache`, `LRUCache`, `catalogue()`, `Site` (manifest, historia y ficheros de run como bytes), `site()`
  y `ROUTES`.
- `mtpy/controllers/Base.py` y `mtpy/controllers/Forecast.py`: cabeceras, caché, CSV y `x-freeze`.
- `api.py` (registro de rutas), `config.json` (sección `web`: `prefix`, `cache_ttl`) y `.env.example`
  (`WEB_PREFIX`, `WEB_CACHE_TTL`).
- `web/`: `index.html`, `promedio.html`, `css/site.css`, `js/` (`api`, `catalog`, `format`, `layout`,
  `state`, `charts/{base,bars,evolution,hemicycle,series}`, `pages/{index,promedio}`) y
  `vendor/echarts-5.6.0.min.js` con `LICENSE-echarts.txt`.
- Infraestructura: `deploy/docker/nginx.conf` (única modificación en `deploy/docker`), `deploy/check-api.sh`,
  `deploy/nginx-check.sh`, `deploy/smoke.sh`.
- Arreglos de la fase 1 (`mtpy/jobs/Publish.py`, `mtpy/lib/publish.py`): `_check_run` con `fullmatch` y
  línea `manifest: failed (...)` si falla la escritura del manifest.
- Documentación: rutas en `docs/web/contrato.md` y sección "La web" de `deploy/README.md`.

Verificado en la sesión:

- Los tests del arnés ASGI (`tests/test_web_api.py`) y el estático `tests/test_web_routes.py`.
- `S3_BUCKET= python -m uvicorn api:api --port 8000` sirviendo todas las rutas desde el paquete local
  `files/site/v1` (run `20261008-181100`), y `bash deploy/check-api.sh` contra uvicorn (todas OK).
- `check-api.sh` y 12 comprobaciones contra un nginx 1.29.5 LOCAL en el puerto 8043 con el `nginx.conf` del
  repositorio reescrito a rutas locales: `/healthz` 200; `/` y `/promedio` 200 `text/html` por
  `try_files`; `/nope` 404; `/vendor/echarts-5.6.0.min.js` `max-age=31536000`; `/js/api.js` y
  `/css/site.css` `no-cache`; `/api/v1/health` `no-store` con `X-Cache: MISS`; `/api/v1/manifest`
  `max-age=60` y `X-Cache: HIT` en la segunda petición; `?run=` → `immutable`; `/api/v1/forecast/es-xx` 400
  JSON sin tocar; cabeceras de seguridad en estáticos y API.
- `bash deploy/nginx-check.sh` (`syntax is ok`) y `node --check` en todos los módulos JS.
- Renderizado de ECharts en el servidor (SVG en Node) del gráfico de serie en los dos modos sobre el run real.

No verificado: Docker (`deploy/smoke.sh` escrito y con `bash -n` limpio, pero nunca ejecutado: el demonio no
estaba en marcha); las páginas en un navegador real (escritorio y móvil, errores de consola); el despliegue
en producción.

Decisiones tomadas durante la ejecución:

1. Modo por defecto `forecast`, con interruptor "Hoy"/"Elección". La portada muestra todos los partidos (el
   selector de grupos es de la fase 3).
2. `simulable` en `/scopes` = el ámbito tiene entrada en el manifest.
3. Sin `WEB_FREEZE`: el único interruptor editorial es `manifest.freeze`; la API añade `x-freeze: active`
   cuando está activo.
4. La API sirve los ficheros del paquete tal cual, con `ETag` (md5 de los bytes) en todo contenido servido,
   `/scopes` incluido.
5. `BLOCK_ORDER = Izquierda, Separatista, Regionalista, Derecha` para el hemiciclo.
6. En `deploy/docker` solo cambia `nginx.conf` (`Dockerfile`, `supervisord.conf`, `init.sh` y `crontab`
   de Luis, intactos). La zona de caché se llama `api_cache` (`api` chocaba con la zona de `limit_req`).
7. ECharts 5.6.0 vendorizado como `web/vendor/echarts-5.6.0.min.js` (versión en el nombre; `location ^~
   /vendor/` con un año de caché).
8. `webapi.py` lee `data/es-scopes.csv` directamente en vez de importar `mtpy.lib.data`, pero los workers
   de gunicorn siguen cargando scipy/statsmodels vía `mtpy.core.utils.helpers` (importado por
   `mtpy.core.io`): aligerarlos exige imports perezosos ahí (fase 6).
9. Los tests de la API viven en `tests/test_web_api.py` (la spec decía `test_api_asgi.py`);
   `tests/test_web_routes.py` acepta enlaces de página `/nombre` si existe `web/nombre.html`.
10. El parámetro de desarrollo `?api=` solo se acepta para `http://localhost` y `http://127.0.0.1`; las
    fechas se formatean con un array fijo de meses en español.

Diferido a la revisión final de la rama o a la fase 3: la plantilla de página duplicada entre
`pages/index.js` y `pages/promedio.js` (un `pages/common.js`), el foco de teclado que se pierde al
repintar la cabecera al cambiar de ámbito o modo, el redondeo de `p_majority` a "100 %", `scope="col"` en
las cabeceras de tabla y `caption` en la portada, el pie obsoleto tras una carga fallida, el selector de
ámbito no limitado a los publicados, las alternativas de texto de los gráficos, el ayudante común de
`part_action`/`mode_part_action`, la caché de catálogo a nivel de módulo compartida entre `App`s y
`float(cache_ttl)` ante basura (500). El foco y el redondeo de `p_majority` se resolvieron en la revisión
final (abajo); el resto queda para la fase 3.

Diferido a la fase 6 (flecos de la fase 0): soporte de `HEAD`, `send()` fuera del `try` de
`Controller.dispatch`, docstring de `Api.__call__`, y los imports perezosos de `helpers.py`/`io.py`.

Revisión final de la rama (2026-10-09), arreglos en un solo commit:

1. Caché de nginx en `/var/lib/nginx/api_cache` (antes `/var/cache/nginx/api`, cuyo padre no existe en la
   imagen); `nginx-check.sh` solo reescribe el padre y deja que `nginx -t` cree la hoja, como la imagen.
2. `set_real_ip_from` 127.0.0.1 y 172.16.0.0/12 con `real_ip_header X-Forwarded-For` (realip): el límite
   de 20 r/s vuelve a ser por cliente detrás del proxy TLS / docker-proxy.
3. Logs de nginx a `/proc/1/fd/1` y `/proc/1/fd/2` (supervisord se demoniza y su `/dev/stdout` es
   `/dev/null`); `nginx-check.sh` los reescribe a ficheros temporales.
4. `location ^~ /api/`; `gzip_proxied any` y `gzip_vary on`; CSP con `frame-ancestors`, `base-uri` y
   `object-src 'none'`; HTML con `expires -1`. La clave de caché es la de nginx por defecto (URI completa
   con la query string): se probó una clave propia sobre `run`/`format` y se descartó porque nginx y la
   API leen los nombres de los parámetros de forma distinta (codificación y mayúsculas), lo que permitía
   envenenar la caché; los parámetros de más crean entradas acotadas (`max_size`, `inactive`, `limit_req`).
5. Verificado con nginx 1.29.5 local en el 8043 y uvicorn (ver el informe de la revisión).
6. `smoke.sh`: `docker stop` ≥ 10 s pasa a WARN (`init.sh` necesita `exec supervisord`) y nueva
   comprobación (WARN) de líneas de acceso de nginx en `docker logs`.
7. `check-api.sh`: `--max-time 20` en cada `curl`.
8. Las páginas fijan el run al cargar (`?run=` o `latest` del manifest) y lo envían en todas las partes
   salvo `runs`; la URL solo lleva `run` si lo fijó el usuario.
9. `fmtProb` en `format.js`: "> 99 %" y "< 1 %" en vez de redondear a 100 % / 0 % (`p_majority` y
   `p_first`).
10. Foco de teclado: la cabecera se construye una vez por carga y `updateHeader` solo actualiza el ámbito,
    el modo y los enlaces de navegación.
11. `.grid > * { min-width: 0 }` y `defer` en el `<script>` de ECharts.
12. `check_format` antes de `resolve_run` (un `?format=` malo en un ámbito sin publicar es 400, no 404),
    con 2 tests nuevos en `tests/test_web_api.py`.
13. `contrato.md`: CORS solo en las respuestas de los controladores (no en el 404 genérico del router) y
    las páginas envían `?run=` en todas las partes.
14. `deploy/README.md`: retirada visible en hasta ~3 min, `X-Forwarded-For` obligatorio en el proxy, logs
    de nginx en `docker logs`, logs de gunicorn perdidos hasta `nodaemon`/`exec supervisord`, WARN de
    `docker stop` y `/promedio` con `http.server`.

Pendiente obligatorio: ocultar las vistas derivadas de sondeos cuando `manifest.freeze.active` (LOREG art.
69.7, difusión desde el 24-11): fase 3 o 6, decisión de Luis sobre qué se oculta; hoy solo hay banner.

Pendiente de Luis:

- Ejecutar `bash deploy/smoke.sh` (paquete local) y `bash deploy/smoke.sh deploy/elections.env` con Docker en
  marcha.
- Abrir `/` y `/promedio` en escritorio y móvil sin errores de consola. A mirar: etiquetas y texto central
  del hemiciclo a 240 px de alto, etiquetas de las barras dentro del lienzo, puntos de la evolución con un
  solo run, que `.grid > * { min-width: 0 }` (ya añadido) baste para que los gráficos encojan, tooltip de días con
  varios sondeos, la leyenda ocultando a la vez banda, puntos y proyección, el interruptor "Hoy"/"Elección"
  y el enlace CSV con `?run=`.
- Primera publicación de `es` a S3 para que producción tenga datos (enmienda 2026-10-09: hecha por Luis el
  2026-10-08, run `20261008-202843`); desplegar la imagen con
  `--env-file deploy/elections.env`; proxy y TLS delante del 8042.
- Decidir el modo por defecto, la agrupación de partidos (fase 3) y `BLOCK_ORDER`.

Siguiente: plan de la fase 3 (escaños, autonómicos, histórico).

Enmienda del 2026-10-09: páginas desde los controladores. `/` y `/promedio` pasan a renderizarse en
Python con Jinja2 y nginx solo sirve `/dist/` y hace de proxy del resto; las secciones "Frontend",
"Infraestructura" y "Pruebas" llevan la enmienda fechada. Diseño, ejecución y estado en
`docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md`. La siguiente fase (plan de la
fase 3) ya se apoya en ese patrón. Quedan superadas decisiones de la lista numerada de ese cierre:
la 7 (ECharts está ahora en `web/dist/vendor/echarts-5.6.0.min.js`, con `location ^~ /dist/vendor/`),
la 9 (`tests/test_web_routes.py` acepta los enlaces de página de `pages.ROUTES` o `/`, y las plantillas
viven en `web/templates/`) y la 10 (`?api=` desaparece con `api.js`: las páginas ya no hacen `fetch`).

## Estado al cierre de la fase 3 (2026-10-09)

Hecho en `dev` (plan en `3935d25`, `docs/superpowers/plans/2026-10-09-web-fase-3-escanos.md`, código en `4fb2811^..a0a2805`,
6 commits de código y este de documentación), con 348 tests unitarios en verde (`python -m pytest -m "not integration"
-q`: 323 del rediseño y 25 nuevos) y los 72 de integración sin cambios (no ejecutados, sin base):

- `mtpy/controllers/Escanos.py` (`Page`, `index_action` con `?region=`) y `web/templates/escanos.html`.
- `mtpy/lib/pages.py`: `chart_polls`, `seat_rows`, `block_rows`, `default_coalition`, `csv_links`,
  `escanos_context`, `resolve_region`, `district_table`, `region_rows`, entradas en `PAGES`/`ROUTES` y el menú
  "Escaños"; `table_rows` tolerante a sondeos sin fecha; `resolve_scope` con 503 sin ámbito del catálogo
  publicado.
- `mtpy/lib/webapi.py` (`check_region`, `settings()` con aviso si el TTL es basura) y
  `mtpy/controllers/Forecast.py` (`serve`).
- `web/dist/js/`: `seats.js` (matemática pura con la regla de `Stat.quantile`), `charts/{histogram,stacked,fan}.js`,
  `pages/escanos.js`, y cambios en `charts/{base,bars,evolution,series}.js`, `catalog.js` (`orderBlocks`) y
  `pages/common.js` (`wireForm`); colores en `THEME`; `role="img"` y `aria-label` en todos los gráficos.
- Pruebas: `tests/test_pages.py`, `tests/test_webapi_unit.py`, `tests/test_web_routes.py` (quita `&amp;…` de los
  `href` de plantilla) y `tests/test_js_seats.py` (paridad con `node`, se omite sin `node`).
- Scripts y documentación: `/escanos` en `deploy/check-api.sh` y `deploy/smoke.sh`, `deploy/README.md`,
  `docs/web/contrato.md`.

Decisiones de diseño (D1-D8 del plan):

1. D1. Una página nueva, `/escanos`, en el menú como "Escaños" (Portada · Promedio · Escaños), con el mismo
   estado `scope`/`mode`/`run` y los mismos controles de cabecera.
2. D2. Secciones: escaños por partido (tabla e histogramas de `dist`), bloques (tablas de `summary.blocks` y
   `summary.vs` y barra apilada), calculadora de coaliciones, circunscripciones (tabla con el escenario
   central y detalle), abanico de voto por horizonte, evolución de los escaños y descargas CSV.
3. D3. Calculadora solo en JavaScript; la selección vive en el fragmento `#coalition=PP,VOX`
   (`history.replaceState`), sin crear entradas en la caché de nginx; por defecto, los partidos del primer
   bloque de `summary.vs`; sin JS, casillas y una nota en `<noscript>`. Los cuantiles siguen la regla de
   `Stat.quantile`.
4. D4. `region` es un parámetro de query propio de `/escanos` (fuera de `parse_state` y de los enlaces del menú):
   entero de `districts.regions`; mal formado 400, desconocido 404, ausente la circunscripción con más
   escaños. Un ámbito sin circunscripciones no muestra la sección y `?region=` da 404.
5. D5. `initial-data` de `/escanos` = `{state, meta, summary, dist, fan, runs}`; `districts` y `scenario` solo
   se renderizan en HTML.
6. D6. Evolución de escaños reutilizando `charts/evolution.js` con campo (`pct` en la portada, `seats` aquí).
7. D7. Fuera de alcance: selector `group`, ocultación de vistas durante la veda (fase 6), `Page.dispatch`
   duplicado y `HEAD`, publicar `scopes: "all"` contra la RDS (Luis), el formulario sin JS con run fijado y
   cambio de ámbito que acaba en 404, el `<select>` que envía con las flechas y `X-Forwarded-For` repetido.
8. D8. Flecos incluidos: recorte del `polls` embebido en `/promedio`, `resolve_scope` sin la vuelta al orden
   del manifest, `table_rows` sin fecha, `cache_ttl` con basura, `Forecast.serve`, colores en `THEME`,
   decimales de los ticks, título "y proyección" solo en `forecast`, `caption` y `aria-label`.

Resoluciones tomadas durante la ejecución:

- Los `bmaps` de los fixtures tenían una forma distinta del contrato y se alinearon (`477d355`); las filas de
  bloque de `summary.json` pasan a `Derecha`.
- `tests/test_web_routes.py` quita `&amp;…` de los `href` de plantilla para que los enlaces de fila con
  `&region=` se resuelvan como páginas (`2cb1c0f`).
- `fan.js` sigue la clave del contrato, `name`; el run LOCAL `20261008-181100`, cuyo `fan.json` es anterior al
  cambio de `party` a `name`, muestra el abanico vacío hasta que se republique en local (`ed6ef58`).
- La línea discontinua de mayoría del histograma se dibuja solo si `starts[0] <= majority < lastStart + width`,
  porque los intervalos son semiabiertos (`ed6ef58`, `a0a2805`).
- En el test de paridad del plan, el literal `'min': 193.0` era un error aritmético y se corrigió a `194.0`
  (`ed6ef58`).

Verificado en la sesión: suite unitaria, `bash deploy/nginx-check.sh`, y `bash deploy/check-api.sh` contra un
nginx 1.29.5 local en el 8043 delante de uvicorn (commit `7002401`).

Menores aplazados: `settings()` acepta `'nan'`/`'inf'`/`cache_ttl` negativo; `chart_polls` omite la clave de un
partido ausente de un sondeo (el JS lo tolera); el `caption` compara `n_polls` sin protegerse de `None` (el
esquema lo hace entero obligatorio); líneas de más de 100 caracteres en `pages.py`/`bars.js`; el `aria-label`
del gráfico del promedio menciona la proyección también en `nowcast`; la regex del test de `aria` es de una
sola línea; la regla del `subtitle` está duplicada entre `escanos_context` e `index.html`; el pie de la
tabla `vs` fija "Derecha frente a izquierda"; ningún test comprueba los colores de bloque en `#vs-table`;
`pages.py` ronda las 1000 líneas (un módulo por página más adelante); los textos de sección de `/escanos`
(títulos, `p.about`, pies) los redactó Claude y Luis puede cambiarlos; ningún test fija la celda "–"
de una circunscripción ausente de `scenario.rows`; el intervalo de voto de una circunscripción muestra "– %"
si un extremo es nulo; `update()` al cargar sobrescribe un fragmento entrante como `#districts`; el envío GET
del formulario de circunscripción descarta el fragmento de coalición y recarga arriba; el bloque de la línea
de marca sobre una serie vacía se repite en `series.js`/`fan.js`/`stacked.js`; la leyenda del abanico sigue el
orden de filas de `fan.json`; las etiquetas de mediana y mayoría pueden solaparse en los histogramas de 160 px.

Pendiente de Luis:

- Recorrer `/escanos` en escritorio y móvil: histogramas pequeños legibles, barra apilada con etiquetas,
  calculadora (marcar y desmarcar, enlace con `#coalition=` compartido), formulario de circunscripción con y
  sin JS, abanico con la leyenda, evolución con un solo run.
- Publicar `scopes: "all"` contra la RDS y revisar cada ámbito en las tres páginas (los nombres de bloque
  distintos de `BLOCK_ORDER` quedan al final del hemiciclo y de la barra apilada).
- Decidir la política de veda (qué se oculta con `manifest.freeze.active`) antes del 2026-11-24 (fase 6) y si
  se añade el selector de agrupación `group`.
- Desplegar la imagen (la primera publicación a S3 y el pin de `jinja2` ya están resueltos).

Revisión final de la rama (2026-10-09), arreglos en un solo commit (`8ec0a1c`; 349 tests unitarios):

1. Una fila de `fan` sin `name` (el `fan.json` local es anterior al cambio `party`→`name`) lanzaba dentro de
   `renderFan` y dejaba sin dibujar la evolución: las filas y nombres sin `name` se filtran y el abanico
   queda vacío. Antes de desplegar, comprobar que el `fan.json` publicado lleva `name`
   (`/api/v1/forecast/es/fan`) o republicar; republicar también en local para ver el abanico.
2. El detalle de circunscripción listaba los partidos que no concurren en ella ("– · – % · 0 · 0–0 ·
   0 %"): `region_rows` descarta las filas con `pct` nulo y un extremo ausente del intervalo de voto se
   muestra como "–" sin unidad.
3. El formulario de circunscripción apunta a `/escanos#districts` (el envío GET conserva el fragmento y
   aterriza en la sección) y el hash `#coalition=` solo se escribe al cambiar una casilla, no al cargar;
   los enlaces profundos `#coalition=…` se siguen leyendo al cargar.
4. La tarjeta de un partido sin columna en `dist` se oculta; la leyenda del abanico sigue el orden del
   catálogo; el `aria-label` del gráfico del promedio solo menciona la proyección en `forecast`; test de
   que los enlaces de las filas de circunscripciones conservan el `run` fijado.
5. `tryDraw` en `common.js`: cada gráfico de `/escanos` se dibuja en su propio `try/catch`, de modo que una
   parte defectuosa no deja en blanco las que vienen después.

Menor aparcado tras la revisión final: un gráfico que falle dentro de `tryDraw` muestra una caja vacía y
el error en consola, sin mensaje en la línea de estado; arreglo sugerido: recoger los resultados de
`tryDraw` en `paint()` y llamar a `showError` si alguno falla. Las celdas de la tabla general para
partidos que no concurren en una circunscripción (0 escaños, rango "–") se dejan como están: el escenario
es coherente y ese 0 es real.
