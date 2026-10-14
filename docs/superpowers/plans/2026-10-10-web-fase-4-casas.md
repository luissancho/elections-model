# Web, fase 4: casas (ranking, perfil, efectos de casa y herding) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar el análisis de casas de cada ámbito (`publish analysis`), servir las tablas de la base
de datos y el análisis por la API (`/api/v1/parties`, `/pollsters`, `/pollsters/{id}`, `/ratings`,
`/polls`, `/results`, `/house-effects`, `/herding`, `/drift`, `/analysis/{scope}/{part}`), y añadir la
página `/casas` con el ranking de casas (V10), los efectos de casa del ciclo y su historia (V5), el
herding, y el perfil de cada casa con su camino al rating y su diana (V11, `?id=`).

**Architecture:** Dos fuentes, como fija la spec: las **tablas** (ratings, catálogos, sondeos,
resultados, efectos de casa, herding, deriva) se consultan en vivo en PostgreSQL desde
`mtpy/lib/webapi.py` (`query_*` sobre los lectores de `mtpy/lib/data.py`, caché TTL de 60 s, 503 sin
base) y las sirve el controlador `Data`; las **salidas del modelo** (resúmenes del `Computer`:
`print_house_effects`, `herding_summary`, `get_polls_metric`, `get_error_estimator_data`) se precalculan
con `publish analysis` en `site/v1/analysis/{scope}/` (sobrescribible, con gemelos CSV) y las sirve el
controlador `Analysis`. La página `/casas` sigue el patrón de las fases 2-3 (`Casas(Page)`,
`casas.html` y `casa.html`, `pages.casas_context`/`casa_context`, JS que solo dibuja desde
`initial-data`), tomando el ranking de la base y los efectos de casa y el herding del análisis
publicado; si el análisis no está publicado, esas secciones se omiten con un aviso, y sin base de datos
la página es un 503. Nada cambia en nginx ni en la imagen Docker (la base ya está configurada por
`deploy/elections.env`).

**Tech Stack:** Python 3.11, pandas, PostgreSQL vía los modelos de `mtpy/models/elections.py`, pytest 9
(dobles sin base; tests de integración marcados contra la RDS); Jinja2 3.1.6; módulos ES + ECharts 5.6.0;
Node 25 para `node --check` y SSR.

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Contrato de API" filas de
`analysis` y de base; "Paquete publicado" línea `analysis/{scope}`; "Comando de publicar"
`what: analysis`; "Servicio y caché"; "Frontend" `/casas`; "Pruebas"; "Preguntas abiertas": casas
destacadas; fila 4 de "Fases"). Patrón de página:
`docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md`. Contrato:
`docs/web/contrato.md`. Decisiones del modelo (memoria/`docs/analisis-2026-09`): ranking global M11,
efectos de casa M6, herding M13 informativo (no entra en el rating).

## Decisiones de diseño de la fase 4 (supuestos; Luis puede corregirlos antes de ejecutar)

- **D1. Paquete `analysis/{scope}/`** con cinco partes y gemelos CSV: `meta` (`analysis-meta@1`),
  `house-effects` (`analysis-house-effects@1`: filas de `print_house_effects()` por casa y partido/bloque:
  `n_events`, `hist`, `trend`, `cycle`, `n_cycle`), `herding` (`analysis-herding@1`: `rows` de
  `herding_summary()` y `events` de `get_herding(scope)` por ciclo), `poll-errors` (`poll-errors@1`:
  `get_polls_metric('error', bmap='vs', drange=(6, 42), n_last=1)`, con `event_date` ISO), `error-data`
  (`error-data@1`: `get_error_estimator_data()`). Se sobrescribe en cada publicación; `meta.json` se
  escribe el último y marca el análisis como completo. No toca el manifest.
- **D2. Tablas en vivo**: `webapi.query_*` llama a los lectores de `data.py` con caché TTL de 60 s por
  argumentos; sin `app.db` → 503 `no database configured`; error de base → 503 `database unavailable`
  (traza en el log). Toda entrada pasa por listas cerradas antes del SQL (`check_scope`, fecha ISO
  validada con `date.fromisoformat`, enteros con `re.fullmatch(r'\d{1,6}')`, `limit` ≤ 5000).
- **D3. `/casas`**: (1) ranking del próximo evento del ámbito (`query_ratings(scope)` → filas con
  `rating` no nulo, orden descendente): tabla (casa, rating, calidad, sondeos del ciclo, elecciones,
  error medio, error de bloques, sesgo de bloques, herding del análisis) y barras horizontales del
  rating; (2) efectos de casa (`analysis/house-effects`): mapa de calor casa × (partidos `main` +
  bloques `vs`) del `cycle`, con tabla de `hist`/`trend`/`cycle`; (3) herding (`analysis/herding`):
  tabla con `herding`, `herding_last`, `herding_p`, `herding_n`, `herding_events`. Cada casa enlaza a su
  perfil `?id=`. Todas las casas con rating se listan (sin "destacadas" fijas: la pregunta abierta de
  la spec se resuelve así; el CIS, GAD3, GESOP y 40dB del notebook eran una elección de figura).
- **D4. Perfil `/casas?id=N`** (plantilla `casa.html`, mismo controlador): cabecera (nombre, tipo,
  calidad, rating actual), camino al rating (línea del `rating` por elección de `query_pollster`), diana
  (`charts/bullseye.js`: error Derecha vs Izquierda de su último sondeo en cada elección, de
  `analysis/poll-errors`), tabla de ratings por elección, tabla de efectos de casa por elección y
  partido (`pollsters_parties`: `dev_result_c`) y tabla de herding por ciclo. `id` desconocido → 404
  "Casa no encontrada"; `id` mal formado → 400.
- **D5. Degradación**: sin base de datos, `/casas` y el perfil son 503 "La base de datos no está
  disponible"; sin `analysis/{scope}` publicado, las secciones de efectos de casa, herding y diana se
  omiten con el aviso "Análisis no publicado todavía" (la página sigue sirviendo el ranking).
- **D6. `scope`** rige el ranking y el análisis (ratings globales M11 guardados por ámbito); `mode` y
  `run` solo afectan a la cabecera común.
- **D7. Fuera de alcance**: `/api/v1/events/{scope}` (`has_page` es de la fase 5), `publish event` y
  `backtest`, páginas `/elecciones`, `/modelo`, `/metodo`; cambios en nginx, Docker o credenciales;
  ejecutar `publish analysis` contra la RDS (lo lanza Luis; el plan deja un test de integración marcado).

## Global Constraints

- PEP8 y docstring numpydoc en inglés en `mtpy/`; español en `tests/`, `docs/`, `deploy/*.md`; plantillas
  y textos visibles en español; JS: módulos ES, identificadores en inglés, sin scripts en línea ni
  `on*=`, sin `innerHTML` con datos.
- `python -m pytest -m "not integration" -q` en verde sin base (hoy 373); `FutureWarning` es error;
  los tests unitarios parchean los lectores de `data.py` (`monkeypatch.setattr(webapi, 'get_ratings',
  fake)`) y usan `FakeComputer`; los de integración van en `tests/integration/` con `pytestmark =
  pytest.mark.integration` y la fixture `app`.
- Nunca `python job.py`; nunca `mtpy.run()`/uvicorn sin `S3_BUCKET=` delante. Los tests de integración
  escriben siempre en `FileSystem(tmp_path)` (solo leen la RDS).
- Árbol limpio (rama `dev`, `f02c577` o posterior); `git add <ruta>` explícito; mensajes cortos en inglés
  con el trailer `Co-Authored-By` del modelo que escribe el commit.
- Contrato 1: sobre común (`schema`, `contract`, `scope`, `run_id` null, `mode` null, `generated_at`,
  `data`); redondeos 2 decimales (porcentajes y puntos), 3 (probabilidades y razones); `NaN` → `null`;
  fechas ISO.
- API: solo GET; `Base` pone CORS, `ETag` (md5), `Cache-Control: public, max-age=60` (nunca inmutable:
  las tablas y el análisis cambian), `x-freeze`; errores JSON `{"status":"error","message"}` con
  `no-store`; CSV `text/csv` con `Content-Disposition: attachment; filename="..."`.
- Páginas: `Page` (HTML, `max-age=60`, `ETag`), errores HTML `no-store` con texto de `pages.MESSAGES`;
  una página lee el paquete con un solo `run`; `initial` por `tojson`; `role="img"` y `aria-label` en
  cada `div.chart`.

## Review Focus

1. Sin `app.db` (tests, imagen sin credenciales): toda ruta de base y `/casas` responden 503 con el
   mensaje fijado, nunca una traza; `/api/v1/health`, el paquete y las otras páginas siguen bien (T2, T4).
2. Análisis no publicado: `/api/v1/analysis/es/herding` → 404 `analysis not published`; `/casas` sirve el
   ranking y omite las tres secciones con el aviso (T2, T4).
3. `?id=` mal formado o inexistente: 400/404 HTML `no-store` sin tocar el análisis; un `id` de una casa sin
   ratings en el ámbito muestra el perfil con tablas vacías y sin diana (T4).
4. Fechas y límites en la API: `?event=2026-13-01` → 400 `invalid date`; `?limit=9999` → 400 `invalid
   limit`; `?pollster=abc` → 400 `invalid pollster`; `?region=x` → 400 `invalid region` (T2).
5. `publish analysis` idempotente y atómico: repetirlo sobrescribe sin dejar restos; un error a mitad deja
   el `meta.json` anterior intacto porque se escribe el último, y la API sirve el análisis anterior (T1).

---

### Task 1: `publish analysis`: exportadores, esquemas, rutas del paquete y acción del job

**Files:**
- Modify: `mtpy/lib/bundle.py` (`ANALYSIS_PARTS`, `SCHEMAS`, `path_analysis`, `path_analysis_csv`,
  `BundleReader.analysis_exists`), `mtpy/lib/publish.py` (`export_analysis_*`, `publish_analysis`),
  `mtpy/jobs/Publish.py` (`what: analysis`), `tests/fakes.py` (`FakeComputer`, `synthetic_analysis`)
- Create: `tests/fixtures/bundle/analysis-meta.json`, `analysis-house-effects.json`,
  `analysis-herding.json`, `poll-errors.json`, `error-data.json`
- Test: `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`, `tests/test_bundle_unit.py`,
  `tests/integration/test_publish.py`

**Interfaces:**
- Consumes: `Computer(scope=scope, path='.').build_series()`, `comp.print_house_effects()` (índice
  `(pollster, name)`, columnas `n_events, hist, trend, cycle, n_cycle`), `comp.herding_summary()` (índice
  `pollster`), `data.get_herding(scope)`, `comp.get_polls_metric('error', bmap='vs', drange=(6, 42),
  n_last=1)` (índice `(event, pollster, days)`; comprobar que conserva `event_date` y, si no, añadirla
  como columna ISO en `get_polls_metric` sin cambiar nada más), `comp.get_error_estimator_data()`,
  `data.get_event_params` (bloques `vs` y `main` del próximo evento), `data.get_next_event_date`,
  `provenance`, `bundle.jsonable`, `BundleWriter.write_json/write_csv`.
- Produces:
  - `bundle.ANALYSIS_PARTS = ('meta', 'house-effects', 'herding', 'poll-errors', 'error-data')`;
    `bundle.ANALYSIS_SCHEMAS = {'meta': 'analysis-meta', 'house-effects': 'analysis-house-effects',
    'herding': 'analysis-herding', 'poll-errors': 'poll-errors', 'error-data': 'error-data'}` (nombre de
    esquema por parte); `SCHEMAS` gana: `analysis-meta` (`scope: str, event_date: str, events: list,
    n_pollsters: int, params: dict, commit: NULLABLE_STR, dirty: bool, versions: dict, seconds: dict`),
    `analysis-house-effects` (`parties: list, blocks: list, rows: list`), `analysis-herding` (`rows:
    list, events: list`), `poll-errors` (`blocks: list, rows: list`), `error-data` (`rows: list`).
  - `bundle.path_analysis(scope, part) -> 'site/v1/analysis/{scope}/{part}.json'`;
    `bundle.path_analysis_csv(scope, name) -> 'site/v1/analysis/{scope}/csv/{name}.csv'` (CSV:
    `house-effects`, `herding`, `herding-events`, `poll-errors`, `error-data`); ambos validan `scope`
    con `ROUTE_ALIAS_RE` y `part` con `ANALYSIS_PARTS`.
  - `BundleReader.analysis_exists(scope) -> bool`: existe `analysis/{scope}/meta.json`.
  - `publish.export_analysis_house_effects(table, parties, blocks) -> (dict, DataFrame)`: `rows` con
    `pollster, name, n_events (int), hist, trend, cycle (2 dec, null), n_cycle (int|null)`.
  - `publish.export_analysis_herding(summary, events) -> (dict, DataFrame, DataFrame)`: `rows` con
    `pollster, herding, herding_last, herding_p (3 dec), herding_n (int), herding_events (int)`; `events`
    con `event_date (ISO), pollster, n (int), ratio, p_value (3 dec)`; dos CSV.
  - `publish.export_poll_errors(frame, blocks) -> (dict, DataFrame)`: `rows` con `event_date, event,
    pollster, days (int), sample_size (int|null), weight (3 dec), error_avg, bias_avg, error_blocks,
    bias_blocks, bias (2 dec)` más una columna por bloque (2 dec); `blocks` = nombres de los bloques.
  - `publish.export_error_data(frame) -> (dict, DataFrame)`: `rows` con `date, pollster, party, regional
    (int), weeks (int), weight (3 dec), pct, error (2 dec)`.
  - `publish.export_analysis_meta(scope, event_date, events, n_pollsters, params, prov, seconds) -> dict`.
  - `publish.publish_analysis(scope, writer, computer=None, stats=None, prov=None, verbose=0) -> dict`:
    construye el `Computer` (inyectable), toma `event_date = next_event_date(scope)`, bloques del evento,
    escribe en orden `house-effects`, `herding`, `poll-errors`, `error-data` (JSON validado + CSV) y por
    último `meta`; devuelve `{'scope', 'event_date', 'n_pollsters', 'seconds'}`. `ValueError` al
    construir el `Computer` → `NothingToPublish`.
  - `Publish.run(what=['analysis'], scopes=...)`: por ámbito `published`/`skipped`/`failed` como
    `forecast` (sin guarda LOREG, sin manifest, sin `dry_run` especial: con `dry_run` escribe bajo
    `site-dry/v1/analysis/`); línea `'{scope}: analysis published (event {date}, N pollsters) in X s'`.
    `DEFERRED` pierde `analysis`.
  - `tests/fakes.py`: `synthetic_analysis()` (DataFrames sintéticos con la forma de los cuatro métodos,
    3 casas, 2 elecciones pasadas + ciclo) y `FakeComputer` (fábrica con la firma `Computer(scope=,
    path=)` cuyo `build_series()` devuelve un objeto con esos cuatro métodos y `scope`).

- [ ] **Step 1: Tests que fallan**

`tests/test_publish_unit.py`:

```python
def test_publish_analysis_writes_the_parts_and_meta_last(tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    out = publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    assert out['scope'] == 'es' and out['n_pollsters'] == 3
    written = [p for p in fs.writes if p.startswith('site/v1/analysis/es/')]
    assert written[-1].endswith('/meta.json') and len([p for p in written if p.endswith('.json')]) == 5
    assert len([p for p in written if p.endswith('.csv')]) == 5
    meta = writer.read_json(bundle.path_analysis('es', 'meta'))
    assert meta['schema'] == 'analysis-meta@1' and meta['run_id'] is None and meta['data']['event_date'] == '2026-11-29'
    he = writer.read_json(bundle.path_analysis('es', 'house-effects'))['data']
    assert he['blocks'] == ['Derecha', 'Izquierda'] and {'pollster', 'name', 'hist', 'trend', 'cycle', 'n_events', 'n_cycle'} <= set(he['rows'][0])
    herding = writer.read_json(bundle.path_analysis('es', 'herding'))['data']
    assert [r['pollster'] for r in herding['rows']] and herding['events'][0]['event_date'] == '2023-07-23'
    errors = writer.read_json(bundle.path_analysis('es', 'poll-errors'))['data']
    assert errors['rows'][0]['event_date'] == '2023-07-23' and 'Derecha' in errors['rows'][0]
    assert bundle.BundleReader(fs).analysis_exists('es')


def test_publish_analysis_overwrites_and_keeps_the_old_meta_on_failure(tmp_path, monkeypatch):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    first = writer.read_json(bundle.path_analysis('es', 'meta'))['generated_at']
    monkeypatch.setattr(publish, 'export_error_data', lambda frame: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError):
        publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    assert writer.read_json(bundle.path_analysis('es', 'meta'))['generated_at'] == first
```

(`RecordingFileSystem` ya existe en `tests/fakes.py` y anota las rutas escritas en `writes`; comprobar su
atributo real y ajustar el nombre.)

`tests/test_jobs_publish.py`:

```python
def test_analysis_action_publishes_each_scope(fresh_app, tmp_path, patched, capsys):
    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis'], scopes=['es'], fs=fs)
    assert result['es']['status'] == 'published'
    assert bundle.BundleReader(fs).analysis_exists('es')
    assert 'es: analysis published (event 2026-11-29, 3 pollsters)' in capsys.readouterr().out
```

(`patched` inyecta también `FakeComputer` en `publish.publish_analysis`; ampliar la fixture.)

`tests/test_bundle_unit.py`: los cinco fixtures nuevos validan contra sus esquemas; `path_analysis('es',
'nope')` → `ValueError`; `path_analysis('es/..', 'meta')` → `ValueError`.

`tests/integration/test_publish.py` (marcado): `publish_analysis('es', BundleWriter(FileSystem(tmp_path)))`
contra la RDS: cinco JSON válidos, `house-effects.rows` no vacío con `cycle` no nulo para alguna casa,
`herding.rows` con `herding` en (0, 5), `poll-errors.rows` con `days` entre 6 y 42.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py -q -k "analysis"`
Expected: FAIL (atributos inexistentes).

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py -q && python -m pytest -m "not integration" -q`
Expected: PASS. El test de integración lo ejecuta Luis (o el implementador con `S3_BUCKET=` delante,
`python -m pytest -m integration -k analysis -q`; solo lee la RDS).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish.py mtpy/jobs/Publish.py tests/fakes.py tests/fixtures/bundle/analysis-meta.json tests/fixtures/bundle/analysis-house-effects.json tests/fixtures/bundle/analysis-herding.json tests/fixtures/bundle/poll-errors.json tests/fixtures/bundle/error-data.json tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py tests/integration/test_publish.py
git commit -m "Publish the houses analysis: house effects, herding, poll errors and error data per scope"
```

---

### Task 2: Capa de base y de análisis en `webapi`, controladores `Data` y `Analysis`, rutas

**Files:**
- Modify: `mtpy/lib/webapi.py` (validadores, `database`, `query_*`, `frame_to_records`, `frame_to_csv`,
  `Site.analysis`, `Site.analysis_csv`, `ROUTES`), `api.py` (sin cambios si `ROUTES` crece), `tests/test_webapi_unit.py`, `tests/test_web_api.py`
- Create: `mtpy/controllers/Data.py`, `mtpy/controllers/Analysis.py`

**Interfaces:**
- Consumes: `data.get_parties`, `get_pollsters`, `get_ratings(scope)`, `get_poll_results(scope,
  event_dates)` (comprobar su salida; si no trae una columna por partido, usar el lector que sí la
  traiga y documentarlo), `get_event_results(scope, date, region_id)`, `get_house_effects(scope)`,
  `get_herding(scope)`, `get_drift(scope)`, `get_next_event_date(scope)`; `App.get_().db`;
  `bundle.ANALYSIS_PARTS`, `path_analysis`, `path_analysis_csv`.
- Produces:
  - Validadores: `check_date(value) -> str` (ISO `YYYY-MM-DD` real, 400 `invalid date`),
    `check_pollster(value) -> int` (400 `invalid pollster`), `check_limit(value, default=1000, maximum=5000)
    -> int` (400 `invalid limit`), `check_region` ya existe (400 `invalid region`).
  - `database()`: devuelve `App.get_().db` o `HttpError(503, 'no database configured')`.
  - `query_parties()`, `query_pollsters()`, `query_pollster(pollster_id, scope) -> dict`
    (`{'pollster': {...} | None, 'ratings': [...], 'house_effects': [...], 'herding': [...]}`, 404
    `pollster not found` si no está en `get_pollsters()`), `query_ratings(scope, event=None) ->
    DataFrame` (por defecto el próximo evento: la fecha máxima de `get_ratings(scope)`), `query_polls(scope,
    event=None, pollster=None, limit=1000)`, `query_results(scope, date, region=0)`,
    `query_house_effects(scope)`, `query_herding(scope)`, `query_drift(scope)`. Todas: `database()` primero,
    caché `TTLCache(settings()['cache_ttl'])` por `(nombre, args)` en `_QUERIES`, excepción de base →
    `app.logger.error(traza)` y `HttpError(503, 'database unavailable')`.
  - `frame_to_records(frame) -> list[dict]` (índice reseteado, `bundle.jsonable`, fechas ISO, NaN null);
    `frame_to_csv(frame) -> bytes` (UTF-8, `\n`, sin índice).
  - `Site.analysis(scope, part) -> bytes` y `Site.analysis_csv(scope, name) -> bytes` con caché TTL (no
    LRU: el análisis se sobrescribe); 404 `analysis not published` si falta.
  - `ROUTES` añade, antes de las genéricas de `forecast`: `('/api/v1/parties', 'data', 'parties')`,
    `('/api/v1/pollsters', 'data', 'pollsters')`, `('/api/v1/pollsters/{id:num}', 'data', 'pollster')`,
    `('/api/v1/ratings', 'data', 'ratings')`, `('/api/v1/polls', 'data', 'polls')`,
    `('/api/v1/results/{scope}/{date}', 'data', 'results')`, `('/api/v1/house-effects', 'data',
    'house_effects')`, `('/api/v1/herding', 'data', 'herding')`, `('/api/v1/drift', 'data', 'drift')`,
    `('/api/v1/analysis/{scope}/{part}', 'analysis', 'part')` (todas `['GET']`).
  - `Data(Base)`: una acción por ruta; `?scope=` por defecto `es` (`check_scope`); `?format=csv` →
    `csv_bytes(frame_to_csv(frame), '{nombre}.csv')`; JSON → `json_bytes(bundle.dumps({'contract': 1,
    ..., 'rows': records}))` (`pollster` devuelve el dict del perfil); `cache()` sin `immutable`.
  - `Analysis(Base)`: `part_action(scope, part)`: `check_scope`, `check_part(part, ANALYSIS_PARTS)`,
    `check_format`; JSON tal cual o CSV (`meta` sin gemelo → 404 `no csv for this part`; `herding`
    tiene dos CSV: `?format=csv` da `herding.csv` y `?format=csv&table=events` da `herding-events.csv`).
- Nota de registro: los literales `/api/v1/parties`, `/pollsters`, ... se registran antes de
  `/api/v1/forecast/{scope}` y no colisionan; `{id:num}` es el alias numérico del router (ver
  `mtpy/core/api.py`).

- [ ] **Step 1: Tests que fallan**

`tests/test_webapi_unit.py`: `check_date` acepta `'2026-11-29'` y rechaza `'2026-13-01'`, `'29/11/2026'`,
`''`; `check_pollster` acepta `'12'` y rechaza `'abc'`, `'-1'`; `check_limit` devuelve 1000 por defecto,
acepta `'50'`, rechaza `'0'`, `'9999'`, `'x'`; `database()` sin `app.db` → 503 `no database configured`;
`query_ratings` con `get_ratings` parcheado (DataFrame de dos eventos) devuelve solo el último evento y
cachea (segunda llamada no vuelve a invocar el lector; comprobar con un contador); `query_ratings` con un
lector que lanza `RuntimeError` → 503 `database unavailable` y una línea en `app.logger`;
`frame_to_records` convierte `Timestamp` a ISO y `NaN` a `None`.

`tests/test_web_api.py` (arnés ASGI; `fresh_app.db = object()` para simular base y los lectores
parcheados con DataFrames pequeños): `/api/v1/ratings?scope=es` → 200 JSON con `rows`, `max-age=60`,
`ETag`, CORS; `?format=csv` → `text/csv` con `Content-Disposition`; `/api/v1/pollsters/3` → perfil con
`pollster.name`; `/api/v1/pollsters/999` → 404 `pollster not found`; `/api/v1/polls?limit=9999` → 400
`invalid limit`; `/api/v1/results/es/2026-13-01` → 400 `invalid date`; `/api/v1/house-effects?scope=es-xx`
→ 400 `invalid scope`; sin `fresh_app.db` → 503 `no database configured` en `/api/v1/parties`;
`/api/v1/analysis/es/herding` → 404 `analysis not published` sin análisis y 200 con los bytes del fixture
tras `write_analysis(fs)` (ayudante nuevo en `tests/test_webapi_unit.py` que escribe los cinco fixtures);
`/api/v1/analysis/es/meta?format=csv` → 404 `no csv for this part`;
`/api/v1/analysis/es/herding?format=csv&table=events` → `herding-events.csv`;
`/api/v1/analysis/es/nope` → 404 `unknown part`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_webapi_unit.py tests/test_web_api.py -q -k "date or pollster or limit or database or query or records or analysis or ratings or polls or results or house or herding or drift or parties"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py tests/test_web_routes.py -q && python -m pytest -m "not integration" -q`
Expected: PASS (`test_web_routes` sigue exigiendo que ningún literal `/api/v1` del JS quede fuera de
`ROUTES`).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/webapi.py mtpy/controllers/Data.py mtpy/controllers/Analysis.py tests/test_webapi_unit.py tests/test_web_api.py
git commit -m "Serve the database tables and the houses analysis at /api/v1"
```

---

### Task 3: Página `/casas` en servidor: ranking, efectos de casa, herding y perfil

**Files:**
- Create: `mtpy/controllers/Casas.py`, `web/templates/casas.html`, `web/templates/casa.html`
- Modify: `mtpy/lib/pages.py` (`PAGES`, `ROUTES`, `MESSAGES`, `analysis_part`, `ranking_rows`,
  `house_effects_grid`, `herding_rows`, `profile_rows`, `casas_context`, `casa_context`),
  `web/dist/css/site.css`, `web/dist/js/pages/casas.js` (mínimo), `tests/test_pages.py`

**Interfaces:**
- Consumes: `webapi.query_ratings`, `query_pollster`, `webapi.site().analysis(scope, part)`,
  `check_pollster`, `common_context`, `api_url` (ampliar con `kind='analysis'` para
  `/api/v1/analysis/{scope}/{part}?format=csv`), filtros existentes.
- Produces:
  - `pages.PAGES` cuarta entrada `{'href': '/casas', 'label': 'Casas'}`; `ROUTES` añade `('/casas',
    'casas', 'index', ['GET'])`; `MESSAGES` añade `'no database configured'` y `'database unavailable'`
    → "La base de datos no está disponible.", `'pollster not found'` → "Casa no encontrada.",
    `'invalid pollster'` → "Casa no válida.".
  - `analysis_part(scope, name) -> dict | None`: `data` de la parte o `None` si el análisis no está
    publicado (404 `analysis not published` capturado; otros errores suben).
  - `ranking_rows(ratings: DataFrame, herding: dict | None) -> list`: filas con `rating` no nulo,
    ordenadas por `rating` desc, con `pollster_id, pollster, rating, quality, num_polls, num_events,
    error_avg, error_blocks, bias_blocks, herding` (del `rows` del análisis por `pollster`, o `None`).
  - `house_effects_grid(part: dict) -> dict`: `{'columns': parties + blocks, 'rows': [{'pollster',
    'cells': [{'cycle', 'hist', 'trend'}]}]}` en el orden del análisis, solo casas con algún `cycle` no
    nulo, ordenadas por el `cycle` del primer bloque desc.
  - `herding_rows(part: dict) -> list`: `rows` ordenadas por `herding` asc.
  - `casas_context(state, active='/casas') -> dict`: común + `ranking` (`ranking_rows`), `grid`
    (`house_effects_grid` o `None`), `herding` (lista o `None`), `analysis_missing` (bool), `csv_links`
    (`ratings` de la API de base y `house-effects`, `herding`, `poll-errors` del análisis), `initial =
    {state, meta, ranking, grid}`.
  - `casa_context(state, pollster_id, active='/casas') -> dict`: común + `profile` (`query_pollster`),
    `path` (filas `event_date, rating, rating_adj, error_blocks, bias_blocks, num_polls` ordenadas por
    fecha), `effects_table` (`{'columns': partidos, 'rows': [{'event_date', 'cells': [dev_result_c]}]}`),
    `herding_table` (filas por ciclo), `bullseye` (filas de `poll-errors` de la casa: `event, event_date,
    days` + bloques; `[]` sin análisis), `initial = {state, meta, path, bullseye, blocks}`.
  - `Casas(Page)`: `index_action` → `?id=` presente → `casa.html` con `casa_context(state,
    check_pollster(id))`; si no, `casas.html` con `casas_context(state)`.
  - `casas.html`: `section#ranking` (`table#ranking-table`: Casa (enlace `?id=`) · Rating · Calidad ·
    Sondeos · Elecciones · Error medio · Error de bloques · Sesgo de bloques · Herding; `div.chart#rating-chart`),
    `section#house-effects` (`div.chart#effects-heatmap` + `table#effects-table`, o `p.notice` con
    "Análisis no publicado todavía"), `section#herding` (`table#herding-table` o el aviso),
    `section#downloads`. `casa.html`: cabecera `h2` con el nombre y `p.muted` con tipo, calidad y rating,
    enlace "← Todas las casas", `section#path` (`div.chart#rating-path` + `table#path-table`),
    `section#bullseye` (`div.chart#bullseye-chart` o aviso), `section#effects` (`table#profile-effects`),
    `section#herding` (`table#profile-herding`). Todos los `div.chart` con `role="img"` y `aria-label`.

- [ ] **Step 1: Tests que fallan en `tests/test_pages.py`**

Con una fixture `houses(api, fresh_app, monkeypatch)` que pone `fresh_app.db = object()`, parchea
`webapi.get_ratings`/`get_pollsters`/`get_house_effects`/`get_herding` con DataFrames de tres casas y dos
eventos, y escribe el análisis con `write_analysis(fresh_app.fs)`:

```python
def test_casas_lists_the_ranking_effects_and_herding(houses):
    status, headers, body = call(houses, '/casas', query='scope=es')
    page = html(body)
    assert status == 200 and 'aria-current="page"' in page and 'id="ranking-table"' in page
    assert page.index('GAD3') < page.index('CIS')  # orden por rating desc en el fixture
    assert 'href="/casas?scope=es&amp;mode=forecast&amp;id=2"' in page
    assert 'id="effects-heatmap"' in page and 'id="herding-table"' in page
    assert 'href="/api/v1/analysis/es/herding?format=csv"' in page and 'href="/api/v1/ratings?scope=es&amp;format=csv"' in page
    data = initial(body)
    assert list(data) == ['state', 'meta', 'ranking', 'grid'] and data['ranking'][0]['pollster'] == 'GAD3'


def test_casas_without_analysis_keeps_the_ranking(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas')[2])
    assert 'id="ranking-table"' in page and 'Análisis no publicado todavía' in page
    assert 'id="effects-heatmap"' not in page and 'id="herding-table"' not in page


def test_casas_without_database_is_a_503_page(api):
    status, headers, body = call(api, '/casas')
    assert status == 503 and 'La base de datos no está disponible' in html(body) and headers['cache-control'] == 'no-store'


def test_casa_profile_shows_the_path_bullseye_and_tables(houses):
    page = html(call(houses, '/casas', query='scope=es&id=2')[2])
    assert '<h2>GAD3' in page and 'id="rating-path"' in page and 'id="bullseye-chart"' in page
    assert 'id="profile-effects"' in page and 'id="profile-herding"' in page and 'Todas las casas' in page
    data = initial(call(houses, '/casas', query='scope=es&id=2')[2])
    assert list(data) == ['state', 'meta', 'path', 'bullseye', 'blocks'] and data['path'][0]['event_date'] == '2023-07-23'


@pytest.mark.parametrize('query, status, text', [('id=abc', 400, 'Casa no válida'), ('id=999', 404, 'Casa no encontrada')])
def test_casa_profile_errors(houses, query, status, text):
    got, headers, body = call(houses, '/casas', query=query)
    assert got == status and text in html(body) and headers['cache-control'] == 'no-store'
```

Más: `test_templates_compile_with_strict_undefined` añade `casas.html` y `casa.html`;
`test_bundle_strings_are_escaped...` recorre `/casas` con un nombre de casa hostil en el fixture de
ratings; `test_page_errors_are_html_and_not_cached` añade `('/casas', 'scope=es-md', 404, ...)` con base
simulada.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "casas or casa_"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces** (módulo `casas.js` mínimo con `wireControls()` y
`mountWhenReady(() => {})`, como hizo la fase 3; CSS: `.notice`, `.heatmap` si hace falta)

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_pages.py tests/test_web_routes.py -q && python -m pytest -m "not integration" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages.py mtpy/controllers/Casas.py web/templates/casas.html web/templates/casa.html web/dist/js/pages/casas.js web/dist/css/site.css tests/test_pages.py
git commit -m "Add the houses page: ranking, house effects, herding and the house profile"
```

---

### Task 4: JavaScript: barras del ranking, mapa de calor, camino al rating y diana

**Files:**
- Create: `web/dist/js/charts/heatmap.js`, `web/dist/js/charts/bullseye.js`, `web/dist/js/charts/line.js`
- Modify: `web/dist/js/pages/casas.js` (completo)

**Interfaces:**
- Consumes: `renderBars` (`charts/bars.js`), `mountChart`, `baseOption`, `escapeHtml`, `THEME`,
  `readInitial`, `wireControls`, `mountWhenReady`, `tryDraw`, `Catalog`, `fmtNum`, `fmtDate`,
  `fmtDateShort`; `initial` de `/casas` (`ranking`, `grid`) y del perfil (`path`, `bullseye`, `blocks`).
- Produces:
  - `renderHeatmap(el, {columns, rows, formatter})`: `heatmap` de ECharts, x = `columns`, y = casas (en
    el orden dado), valor = `cycle`, `visualMap` divergente centrado en 0 (rojo negativo, azul positivo,
    `THEME`), etiqueta con el valor a 1 decimal, tooltip "casa · columna: ciclo X (historia Y, tendencia
    Z/año)"; altura según filas (`el.style.height = 24 px por fila + 80`).
  - `renderBullseye(el, rows, {blocks, labels})`: `scatter` con x = error del primer bloque, y = del
    segundo, anillos a ±2, ±5, ±10 puntos (`markLine`/círculos dibujados con `graphic` o `series` tipo
    `line` cerradas), ejes simétricos, un punto por elección etiquetado con `event`, tooltip con fecha,
    días y errores.
  - `renderLine(el, points, {formatter, name, color})`: una línea con puntos sobre eje temporal
    (`[iso_date, value]`), tooltip `fmtDate · formatter(value)`.
  - `pages/casas.js`: en `/casas`, `renderBars(#rating-chart, ranking, {value: 'rating', lo: null, hi:
    null, formatter: (x) => fmtNum(x, 1)})` sin intervalo (comprobar que `renderBars` acepta filas sin
    `lo`/`hi`; si no, pasarle claves ausentes) y `renderHeatmap(#effects-heatmap, grid)` si `grid`; en
    el perfil, `renderLine(#rating-path, path → [[event_date, rating]])` y `renderBullseye(#bullseye-chart,
    bullseye, {blocks})` si hay filas; cada gráfico en `tryDraw`.

- [ ] **Step 1: `node --check` y tests estáticos en rojo** (no hay test unitario de JS para estos
gráficos; el test estático `test_web_routes` fallará hasta que existan los módulos importados por
`casas.js`)

- [ ] **Step 2: Implementar según Interfaces**

- [ ] **Step 3: Verificar**

Run: `for f in web/dist/js/charts/heatmap.js web/dist/js/charts/bullseye.js web/dist/js/charts/line.js web/dist/js/pages/casas.js; do node --check "$f" || exit 1; done && python -m pytest tests/test_web_routes.py tests/test_pages.py -q`
Expected: PASS.

SSR en el scratchpad (patrón de las fases 3 y 3b) con datos sintéticos: el mapa de calor dibuja una celda
por casa × columna y el `visualMap` está centrado en 0; la diana dibuja un punto por elección y los
anillos; la línea dibuja los puntos del camino; `process.exit(0)` al final.

- [ ] **Step 4: Commit**

```bash
git add web/dist/js/charts/heatmap.js web/dist/js/charts/bullseye.js web/dist/js/charts/line.js web/dist/js/pages/casas.js
git commit -m "Draw the houses page: rating bars, house-effects heat map, rating path and bullseye"
```

---

### Task 5: Scripts, documentación, spec y verificación local

**Files:**
- Modify: `deploy/check-api.sh`, `deploy/smoke.sh`, `deploy/README.md`, `docs/web/contrato.md`,
  `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`

- [ ] **Step 1: Scripts**

`check-api.sh`: `check "/casas?scope=es" 200 html` y `check /api/v1/ratings?scope=es 200 json` (ambos
necesitan base; el script los marca `n/a` si `/api/v1/parties` responde 503 `no database configured`,
siguiendo el patrón de `has_nginx`), `check /api/v1/analysis/es-xx/herding 400 none`, `check
/api/v1/analysis/es/nope 404 none`. `smoke.sh`: `check_static "/casas?scope=es" text/html` solo con
`deploy/elections.env` (con el paquete local y sin base, `/casas` es 503: documentar).

- [ ] **Step 2: Documentación**

README: `python job.py publish '{"what":["analysis"],"scopes":"all"}'` (duración ≈ 1,5 min por ámbito;
repetir tras cada carga de sondeos y tras cada recálculo de ratings), la base en el contenedor (`DB_*` de
`elections.env`; sin ellas la API de base y `/casas` dan 503), páginas `Casas` (`/casas`, `?id=`),
`check-api.sh` con las rutas nuevas, "Cómo añadir una página" sin cambios. `contrato.md`: sección
"Análisis" (árbol, sobre, cinco partes y CSV, sobrescritura, `analysis not published`) y las diez rutas
nuevas con sus parámetros y errores (`invalid date`, `invalid pollster`, `invalid limit`, `pollster not
found`, `no database configured`, `database unavailable`). Spec: "Estado al cierre de la fase 4
(fecha)" con ficheros, tests, D1-D7, fallos de ejecución, menores aplazados y pendientes de Luis
(`publish analysis` real, recorrido de `/casas`, decisión sobre casas destacadas cerrada por D3).

- [ ] **Step 3: Verificación local**

Run: `python -m pytest -m "not integration" -q`; `bash deploy/nginx-check.sh`; uvicorn (`S3_BUCKET=`,
con el `.env` de la raíz, que da acceso de lectura a la RDS) + nginx local 8043:
`bash deploy/check-api.sh http://127.0.0.1:8043`; `curl` de `/casas?scope=es` (200, tabla del ranking con
casas reales; las secciones del análisis con el aviso si no se ha publicado), `/casas?id=1` (perfil del
CIS), `/api/v1/ratings?scope=es&format=csv` (CSV). Parar uvicorn y nginx al terminar.

- [ ] **Step 4: Commit**

```bash
git add deploy/check-api.sh deploy/smoke.sh deploy/README.md docs/web/contrato.md docs/superpowers/specs/2026-10-07-web-publicacion-design.md
git commit -m "Check the houses page and the data routes in the scripts and document the phase 4 closing state"
```

## Pendiente de Luis tras la fase 4

- Lanzar `publish analysis` contra la RDS para `es` (y `all`) y mirar `/casas`: ranking, mapa de calor,
  herding y el perfil de dos o tres casas (CIS, GAD3, 40dB).
- Decidir si el ranking debe limitar las casas (hoy todas las que tienen rating) y si el perfil debe
  mostrar el sesgo por partido además de por bloque.
- Cadencia: `publish analysis` tras cada recálculo de ratings (`run_load --what ratings`).

## Esfuerzo estimado

T1 1 d · T2 1 d · T3 1 d · T4 0,75 d · T5 0,5 d → 4,25 días (la spec estimaba 3-4).
