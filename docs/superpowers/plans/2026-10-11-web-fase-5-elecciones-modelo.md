# Web, fase 5: elecciones y modelo (`publish event`/`backtest`, `/elecciones`, `/modelo`, `/metodo`) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dividir `mtpy/lib/pages.py` y `mtpy/lib/webapi.py` en paquetes con un módulo por página y por
fuente (tarea aplazada al cierre de la fase 4) y, sobre esa base, publicar los eventos pasados
(`publish event`) y el backtest del modelo (`publish backtest`) en el paquete `site/v1`, servirlos por la
API (`/api/v1/events/{scope}`, `/api/v1/events/{scope}/{date}/{part}`, `/api/v1/backtest/{scope}/{part}`)
y añadir las páginas `/elecciones` (lista con el error del sector, V12, y detalle `?date=` de una elección
pasada, V13), `/modelo` (precisión del modelo por horizonte, V14) y `/metodo` (texto de la metodología,
V15).

**Architecture:** Dos refactorizaciones puras abren la fase: `mtpy/lib/webapi/` (`caches`, `checks`,
`published` —el paquete publicado—, `queries` —la base de datos—) y `mtpy/lib/pages/` (`filters`,
`common` y un módulo por página), ambas con un `__init__` que reexporta los nombres públicos para que
los controladores, `api.py` y los tests sigan importando `pages` y `webapi`. Después se repite el patrón
de la fase 4: lo **precalculado** (los eventos y el backtest) lo escribe `python job.py publish` en el
paquete (`events/{scope}/{date}/` y `backtest/{scope}/`, sobrescribibles, con `meta.json` el último y
gemelos CSV) y lo sirven controladores de solo lectura; la lista de eventos de la API cruza la tabla
`events` de la base (en vivo, `webapi.queries`) con lo publicado (`has_page`). Las páginas siguen el
patrón `Page` + plantilla Jinja2 + `initial-data` + JS que solo dibuja: `/elecciones` y `/modelo` leen
únicamente el paquete (sin base), y `/metodo` es una plantilla de texto sin JS. Nada cambia en nginx, en
Docker ni en las credenciales.

**Tech Stack:** Python 3.11, pandas, Jinja2 3.1.6, pytest 9 (dobles sin base; integración marcada contra
la RDS); módulos ES + ECharts 5.6.0 vendorizado; Node 25 solo para `node --check`.

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Paquete publicado": líneas
`events/{scope}/{date}` y `backtest/{scope}`; "Comando de publicar": `what: event`, `backtest`;
"Contrato de API": filas `events` y `backtest`; "Frontend": `/elecciones`, `/modelo`, `/metodo`;
"Hallazgos": V12-V15; "Fases": fila 5; "Estado al cierre de la fase 4": la división de `pages.py` y
`webapi.py` como preflight). Patrón de página: `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md`.
Contrato del paquete: `docs/web/contrato.md`. Backtest: `backtest/README.md` (protocolo y métricas),
`docs/analisis-2026-09/metodo-3-backtest.md`. Evento pasado: `notebooks/PollstersEvent.ipynb` (celdas 4-10).
Metodología: `docs/analisis-2026-09/metodo-*.md` y `README.md`.

## Estado de partida

Rama `dev` en `2bd5a56` (árbol limpio), 457 tests unitarios en verde (`python -m pytest -m "not integration" -q`).
`mtpy/lib/pages.py` tiene 1.484 líneas y `mtpy/lib/webapi.py` 1.510. `backtest/results/` (y
`backtest/results/{scope}/` para 17 ámbitos autonómicos) contiene la última ejecución del backtest con
`run.json`. En `tests/fakes.py` existen `FakeComputer`, `synthetic_analysis`, `RecordingFileSystem`;
en `tests/test_webapi_unit.py`, `write_bundle` y `write_analysis`. Pendientes de Luis que no bloquean:
`publish analysis` real, recorrido de `/casas`, `DB_STATEMENT_TIMEOUT` en `deploy/elections.env`.

## Decisiones de diseño de la fase 5 (supuestos; Luis puede corregirlos antes de ejecutar)

- **D1. Paquetes con fachada.** `mtpy/lib/webapi/` = `caches.py` (`TTLCache`, `LRUCache`), `checks.py`
  (validadores y catálogo), `published.py` (`settings`, `Site`, `site`: el paquete publicado) y
  `queries.py` (`database`, lectores perezosos `get_*`, `query_*`, `check_event`, `frame_to_*`: la base
  de datos); `mtpy/lib/pages/` = `filters.py` (filtros `fmt_*`), `common.py` (constantes, entorno Jinja2,
  estado, catálogo, `common_context`, `api_url`) y un módulo por página (`index`, `promedio`,
  `escanos`, `casas`, y en esta fase `elecciones`, `modelo`, `metodo`). Cada `__init__.py` reexporta
  los nombres públicos: los controladores, `api.py` y los tests siguen usando `pages.X` y `webapi.X`.
  Regla fija: **ningún submódulo se llama como un nombre exportado** (por eso `queries.py` y no
  `database.py`, `published.py` y no `site.py`): un `from .site import site` dejaría el módulo tapado
  por la función, el mismo fallo que el `__getattr__` perezoso de `controllers` en el rediseño. Los tests
  parchean los lectores en `webapi.queries` (`monkeypatch.setattr(webapi.queries, 'get_ratings', ...)`),
  que es donde los buscan los `query_*`.
- **D2. `publish backtest`** convierte las tablas de `backtest/results[/{scope}]/` (fuente `source`
  configurable, por defecto la del repositorio) en `backtest/{scope}/{meta,metrics,by-horizon,blocks,shares,seats}.json`
  con gemelos CSV; `meta` recoge `run.json` (`params`, `events`, `horizons`, `commit`, `run_at`) y la
  procedencia de la publicación. `provinces.csv` queda fuera (es el cubo provincial, no una vista).
  Sobrescribible, `meta` el último, sin manifest; un ámbito sin `metrics.csv` termina `skipped`.
- **D3. `publish event`** publica una elección pasada por ámbito y fecha (`event_date` como cadena, lista
  o `"all"` = todas las elecciones `featured` anteriores a hoy): un `Computer(scope, event_dates=[fecha])`
  como el notebook; partes `pollsters` (resumen por casa y su último sondeo), `polls` (todos los sondeos
  del evento con su error), `results` (región 0), `model` (las filas de `shares`, `seats` y `blocks` del
  backtest de esa elección, vacías si no hay backtest) y `meta` (con `summary`, el error del sector para
  V12). Sobrescribible (se relanza tras corregir datos), `meta` el último, sin manifest. Una fecha que no
  es elección `featured` del ámbito o sin resultados oficiales termina `skipped`.
- **D4. API.** `/api/v1/events/{scope}` lee la base (`events`: `date`, `name`, `featured`) y añade
  `has_page` desde el paquete (503 sin base, como las rutas de `Data`); `/api/v1/events/{scope}/{date}/{part}`
  y `/api/v1/backtest/{scope}/{part}` sirven el paquete tal cual (TTL de punteros, nunca inmutables);
  `{date}` pasa `check_date` (400 `invalid date`) y ha de estar publicada (404 `event not published`, sin
  leer la base); CSV con `?format=csv` (`meta` sin gemelo; `model` exige `table` ∈ `shares`, `seats`,
  `blocks`).
- **D5. `/elecciones`** lista las elecciones publicadas del ámbito leyendo solo el paquete (sin base; las
  publicadas son por construcción `featured` con resultado), de la más reciente a la más antigua, con el
  error del sector por elección (V12) desde `meta.summary`; `?date=` muestra el detalle (V13): resultados,
  casas con su error, diana del último sondeo por casa, sondeos y el modelo a cada horizonte; sin eventos
  publicados, aviso y 200; `?date=` mal formada 400, no publicada 404 (HTML `no-store`).
- **D6. `/modelo`** lee `backtest/{scope}`: error del promedio por horizonte frente a las tres líneas base,
  coberturas, escaños y mayorías, tabla por elección y bloques; sin backtest publicado (la mayoría de los
  autonómicos), aviso "Backtest no publicado todavía" y 200.
- **D7. `/metodo`** es una plantilla de texto en español con secciones fijas (datos, promedio, casas,
  simulación, escaños, evaluación, publicación, limitaciones) y enlaces a las demás páginas; sin JS ni
  `initial-data`; sin cifras de resultados escritas a mano (las constantes del método sí: barrera del 3 %,
  horizontes).
- **D8. `scope`** rige todo; `mode` y `run` solo afectan a la cabecera común (como D6 de la fase 4).
- **D9. Partidos antiguos.** Un partido de una elección pasada ausente del catálogo del run (`Cs`, `UP`)
  se pinta con `Catalog.__missing__` (nombre, gris de "otros"), nunca rompe la página.
- **D10. Fuera de alcance:** la veda (`freeze` en vistas; fase 6), `HEAD`, `asyncio.to_thread`, importaciones
  perezosas de `helpers.py`/`io.py`, cambios en nginx, Docker o credenciales; ejecutar `publish event`
  contra la RDS (lo lanza Luis; queda un test de integración marcado); un histograma de errores en el
  detalle de la elección (el notebook lo tiene; `charts/histogram.js` habla de escaños y no compensa
  generalizarlo ahora).

## Global Constraints

- PEP8 y docstring numpydoc en inglés en `mtpy/`; español en `tests/`, `docs/`, `deploy/*.md`; plantillas
  y textos visibles en español; JS: módulos ES, identificadores en inglés, sin scripts en línea ni `on*=`,
  sin `innerHTML` con datos; `role="img"` y `aria-label` en cada `div.chart`.
- `python -m pytest -m "not integration" -q` en verde sin base (hoy 457); `FutureWarning` es error; los
  tests unitarios parchean los lectores de `data.py` en `webapi.queries` y usan `FakeComputer`; los de
  integración van en `tests/integration/` con `pytestmark = pytest.mark.integration` y la fixture `app`, y
  escriben siempre en `FileSystem(tmp_path)` (solo leen la RDS).
- Nunca `python job.py` ni `mtpy.run()`/uvicorn sin `S3_BUCKET=` delante (el `.env` apunta al bucket y a
  la RDS reales); para las comprobaciones locales, además `DB_HOST=` vacío. Nunca tocar el nginx local de
  Luis en el 8080.
- Árbol limpio (rama `dev`, `2bd5a56` o posterior); `git add <ruta>` explícito; un commit por tarea,
  mensaje corto en inglés con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Contrato 1: sobre común (`schema`, `contract`, `scope`, `run_id` null, `mode` null, `generated_at`,
  `data`); redondeos 2 decimales (porcentajes, puntos, escaños medios), 3 (probabilidades, coberturas,
  Brier, log score, razones, pesos); enteros como `int`; `NaN` → `null`; fechas ISO; `jsonable` antes de
  validar.
- API: solo GET; `Base` pone CORS, `ETag` (md5), `Cache-Control: public, max-age=<WEB_CACHE_TTL>` (nunca
  inmutable: eventos y backtest se sobrescriben), `x-freeze`; errores JSON `{"status":"error","message"}`
  con `no-store`; CSV `text/csv` con `Content-Disposition: attachment; filename="..."`; las rutas fijas
  antes que las genéricas.
- Páginas: `Page` (HTML, `max-age`, `ETag`), errores HTML `no-store` con texto de `pages.MESSAGES`; una
  página lee el paquete con un solo `run` para la cabecera; `initial` por `tojson` solo si hay gráficos;
  el menú (`pages.PAGES`) y `pages.ROUTES` ganan una entrada por página nueva; un parámetro propio de la
  página (`date`) se valida en el `index_action`, fuera de `parse_state`.
- Refactorizaciones (T1, T2): comportamiento idéntico; ningún cambio de firma ni de texto; la suite
  existente pasa cambiando solo los destinos de `monkeypatch` y los imports de los tests.

## Review Focus

1. **Refactorización neutra (T1, T2):** la suite completa pasa; `import mtpy.lib.webapi` sigue sin cargar
   `mtpy.lib.data`, `publish` ni `simulator`; ningún submódulo queda tapado por un nombre exportado;
   `pages.TEMPLATES_DIR` sigue apuntando a `web/templates` (el `REPO_ROOT` de `common.py` tiene un nivel
   más). Tests de disposición en T1 y T2.
2. **Fechas en eventos (T5, T6):** `2023-13-01` → 400 `invalid date` / "Fecha no válida."; una fecha real
   no publicada → 404 `event not published` / "Elección no publicada." sin tocar la base ni el sistema de
   ficheros más allá del listado; `/api/v1/events/{scope}` sin base → 503 mientras `/elecciones` renderiza.
3. **Ámbito sin backtest (T3, T5, T7):** `publish backtest` → `skipped`; `/api/v1/backtest/es-cb/metrics`
   → 404 `backtest not published`; `/modelo?scope=es-cb` → 200 con aviso y sin `initial-data`.
4. **Sobrescritura atómica (T3, T4):** repetir `publish event`/`backtest` sobrescribe sin restos; un fallo a
   mitad deja el `meta.json` anterior intacto porque se escribe el último; `publish event` de una fecha sin
   resultados o no `featured` termina `skipped`, nunca `failed`.
5. **Partidos fuera del catálogo (T6):** una fila de resultados de un partido que no está en `meta.parties`
   del run se renderiza con el color de "otros" y su nombre; el detalle de la elección nunca da 500.

---

### Task 1: dividir `webapi.py` en el paquete `mtpy/lib/webapi/`

**Files:**
- Move: `mtpy/lib/webapi.py` → `mtpy/lib/webapi/queries.py` (`git mv`, conserva la historia del bloque mayor)
- Create: `mtpy/lib/webapi/__init__.py`, `mtpy/lib/webapi/caches.py`, `mtpy/lib/webapi/checks.py`,
  `mtpy/lib/webapi/published.py`
- Modify: `tests/test_webapi_unit.py`, `tests/test_web_api.py`, `tests/test_pages.py` (destinos de
  `monkeypatch`), `tests/integration/test_web_data.py` (solo si parchea)
- Test: `tests/test_webapi_unit.py`

**Interfaces:**
- Consumes: el `webapi.py` actual (bloques por líneas: 27-40 constantes; 43-68 `settings`; 71-232
  `TTLCache`, `LRUCache`; 235-519 catálogo y validadores; 522-890 `Site`, `site`; 893-1472 lectores
  perezosos, `_QUERIES`, `database_host`, `database`, `_query`, `query_*`, `check_event`, `frame_to_*`;
  1475-1510 `ROUTES`, `add_routes`).
- Produces:
  - `caches.py`: `DEFAULT_TTL`, `LRU_SIZE`, `TTLCache`, `LRUCache`.
  - `checks.py`: `DATE_RE`, `INT_RE`, `FORMATS`, `DEFAULT_LIMIT`, `MAX_LIMIT`, `ANALYSIS_PARTS`,
    `CSV_PARTS`, `MODE_PARTS`, `RUN_PARTS`, `MODES`, `_CATALOGUE`, `catalogue`, `scope_codes`,
    `check_scope`, `check_run`, `check_mode`, `check_region`, `check_date`, `check_pollster`,
    `check_limit`, `check_part`, `check_format`.
  - `published.py`: `IMMUTABLE_MAX_AGE`, `settings`, `Site`, `site`.
  - `queries.py`: `QUERY_CACHE_SIZE`, `_reader`, los diez `get_*`, `_QUERIES`, `database_host`,
    `database`, `_query`, `_rows_of`, `_ratings`, `_next_event`, `_event_dates`, `check_event`,
    `query_parties`, `query_pollsters`, `query_pollster`, `ratings_event`, `query_ratings`,
    `polls_event`, `query_polls`, `_election_dates`, `query_results`, `query_house_effects`,
    `query_herding`, `query_drift`, `frame_to_records`, `frame_to_csv`.
  - `__init__.py`: docstring del módulo actual (adaptado a los cuatro submódulos), reexportación
    explícita de todos los nombres públicos anteriores (`from .caches import ...`, etc.; nada de `*`),
    `ROUTES` y `add_routes` sin cambios. Los privados (`_QUERIES`, `_CATALOGUE`) no se reexportan.
  - Ningún módulo supera las 600 líneas. Los controladores (`Base`, `Forecast`, `Data`, `Analysis`,
    `Page`, `Casas`) y `api.py` no cambian.

- [ ] **Step 1: Test de disposición que falla**

`tests/test_webapi_unit.py`:

```python
def test_webapi_package_layout():
    """Un submódulo por papel y ninguno tapado por un nombre exportado por el paquete (D1)."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(webapi.__path__))
    assert names == ['caches', 'checks', 'published', 'queries']
    for name in names:
        assert inspect.ismodule(getattr(webapi, name)), name
    assert webapi.TTLCache.__module__ == 'mtpy.lib.webapi.caches'
    assert webapi.check_scope.__module__ == 'mtpy.lib.webapi.checks'
    assert webapi.Site.__module__ == 'mtpy.lib.webapi.published'
    assert webapi.query_ratings.__module__ == 'mtpy.lib.webapi.queries'
    assert webapi.ROUTES[0][0] == '/api/v1/health'
```

- [ ] **Step 2: Comprobar que falla**

Run: `python -m pytest tests/test_webapi_unit.py::test_webapi_package_layout -q`
Expected: FAIL (`webapi` no tiene `__path__`).

- [ ] **Step 3: Crear el paquete moviendo los bloques tal cual**

`git mv mtpy/lib/webapi.py mtpy/lib/webapi/queries.py`; recortar de `queries.py` los bloques de
`caches.py`, `checks.py` y `published.py` (contenido textual, docstrings incluidos); ajustar los imports
internos (`from .caches import TTLCache, LRUCache`, `from .checks import ...`, `from .published import
settings` en `queries.py`; `from ..core.api import HttpError`, `from .. import bundle`, `from ..bundle
import BundleReader` donde corresponda); escribir `__init__.py`. El `queries.py` sigue importando
`mtpy.lib.data` solo dentro de los proxies.

- [ ] **Step 4: Cambiar los destinos de `monkeypatch` en los tests**

```bash
sed -i '' "s/monkeypatch.setattr(webapi, 'get_/monkeypatch.setattr(webapi.queries, 'get_/g; s/webapi\._QUERIES/webapi.queries._QUERIES/g" tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py
```

En `tests/test_webapi_unit.py` (`test_settings_fall_back_to_the_default_ttl_on_garbage`): `webapi.App` →
`App` con `from mtpy.core.app import App`. Comprobar con `grep -n "webapi\.\(get_\|_QUERIES\|App\)" tests`
que no queda ninguna referencia.

- [ ] **Step 5: Suite en verde y paquete ligero**

Run: `python -m pytest -m "not integration" -q`
Expected: 458 passed (457 + la disposición), incluido `test_webapi_import_stays_light`.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/webapi tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py
git commit -m "Split webapi into a package: caches, checks, published bundle and database queries"
```

---

### Task 2: dividir `pages.py` en el paquete `mtpy/lib/pages/`

**Files:**
- Move: `mtpy/lib/pages.py` → `mtpy/lib/pages/common.py` (`git mv`)
- Create: `mtpy/lib/pages/__init__.py`, `filters.py`, `index.py`, `promedio.py`, `escanos.py`, `casas.py`
- Modify: `deploy/README.md` ("Cómo añadir una página", pasos 3-4), `tests/test_pages.py`
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: el `pages.py` actual (líneas 30-85 constantes; 88-333 filtros; 336-690 entorno, estado,
  `part`, `query_string`, `api_url`, `Catalog`, `catalog`, `block_color`, `common_context`; 692-765
  `majority_rows`, `index_context`; 768-879 `chart_polls`, `table_rows`, `promedio_context`; 882-1175
  escaños; 1178-1484 casas).
- Produces:
  - `filters.py`: `MONTHS`, `DASH`, `_DATE_RE`, `_is_missing`, `fmt_num`, `fmt_int`, `fmt_pct`,
    `fmt_prob`, `fmt_range`, `_madrid_tz`, `_parse_datetime`, `fmt_date`, `fmt_datetime`.
  - `common.py`: `REPO_ROOT` (ahora `os.path.join(os.path.dirname(__file__), '..', '..', '..')`),
    `TEMPLATES_DIR`, `SITE_TITLE`, `PAGES`, `ROUTES`, `NOT_FOUND`, `MESSAGES`, `OTHERS_COLOR`,
    `MODE_LABELS`, `_COLOR_RE`, `environment`, `parse_state`, `published_scopes`, `resolve_scope`,
    `part`, `query_string`, `api_url`, `Catalog`, `catalog`, `block_color`, `common_context`.
  - `index.py`: `majority_rows`, `index_context`. `promedio.py`: `TABLE_ROWS`, `chart_polls`,
    `table_rows`, `promedio_context`. `escanos.py`: `CSV_PARTS`, `seat_rows`, `block_rows`,
    `default_coalition`, `csv_links`, `resolve_region`, `district_table`, `region_rows`,
    `escanos_context`. `casas.py`: `MTYPE_LABELS`, `RANKING_KEYS`, `PATH_KEYS`, `HERDING_KEYS`,
    `ANALYSIS_CSV`, `analysis_part`, `ranking_rows`, `house_effects_grid`, `herding_rows`,
    `profile_rows`, `effects_table`, `casas_context`, `casa_context`.
  - `__init__.py`: docstring (adaptado), reexportación explícita de todo lo público anterior (los tests
    usan, entre otros, `pages.fmt_*`, `api_url`, `table_rows`, `environment`, `ROUTES`, `NOT_FOUND`,
    `MESSAGES`, `TEMPLATES_DIR`, `SITE_TITLE`, `OTHERS_COLOR`, `parse_state`, `*_context`, `region_rows`,
    `district_table`, `default_coalition`, `seat_rows`, `resolve_region`, `csv_links`, `ranking_rows`,
    `house_effects_grid`, `herding_rows`, `analysis_part`). `test_pages.py` parchea
    `pages.index_context` sobre el paquete y `Index` lo busca ahí (`pages.index_context(...)`): sigue
    funcionando sin cambios.
  - Los módulos de página importan de `.common` y `.filters` (`from .common import common_context, part,
    api_url, ...`); ninguno importa de `__init__`.

- [ ] **Step 1: Test de disposición que falla**

`tests/test_pages.py`:

```python
def test_pages_package_layout():
    """Un módulo por página, `common` y `filters`; ninguno tapado por un nombre exportado (D1)."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(pages.__path__))
    assert names == ['casas', 'common', 'escanos', 'filters', 'index', 'promedio']
    for name in names:
        assert inspect.ismodule(getattr(pages, name)), name
    assert pages.fmt_pct.__module__ == 'mtpy.lib.pages.filters'
    assert pages.common_context.__module__ == 'mtpy.lib.pages.common'
    assert pages.casas_context.__module__ == 'mtpy.lib.pages.casas'
    assert pages.TEMPLATES_DIR.endswith(os.path.join('web', 'templates')) and os.path.isdir(pages.TEMPLATES_DIR)
```

(T6, T7 y T8 añaden `elecciones`, `modelo` y `metodo` a la lista esperada.)

- [ ] **Step 2: Comprobar que falla**

Run: `python -m pytest tests/test_pages.py::test_pages_package_layout -q`
Expected: FAIL.

- [ ] **Step 3: Crear el paquete moviendo los bloques tal cual**

`git mv mtpy/lib/pages.py mtpy/lib/pages/common.py`; recortar los bloques en sus módulos; corregir
`REPO_ROOT`; escribir `__init__.py`. Actualizar los pasos 3-4 de "Cómo añadir una página" en
`deploy/README.md`: la entrada va en `pages.ROUTES`/`pages.PAGES` de `mtpy/lib/pages/common.py`, el
contexto en `mtpy/lib/pages/<página>.py` y su reexportación en `mtpy/lib/pages/__init__.py`.

- [ ] **Step 4: Suite en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: 459 passed (incluidos `test_templates_compile_with_strict_undefined` y
`test_unexpected_error_is_a_generic_500_page_and_is_logged`, que prueban `TEMPLATES_DIR` y el parche
sobre el paquete).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages tests/test_pages.py deploy/README.md
git commit -m "Split pages into a package: filters, common context and one module per page"
```

---

### Task 3: `publish backtest`: conversión de `backtest/results` al paquete

**Files:**
- Modify: `mtpy/lib/bundle.py` (`BACKTEST_*`, `SCHEMAS`, `path_backtest`, `path_backtest_csv`,
  `BundleReader.backtest_exists`), `mtpy/lib/publish.py`, `mtpy/jobs/Publish.py`, `tests/fakes.py`,
  `docs/web/contrato.md` (árbol y sección "Backtest")
- Create: `tests/fixtures/bundle/backtest-meta.json`, `backtest-metrics.json`, `backtest-by-horizon.json`,
  `backtest-blocks.json`, `backtest-shares.json`, `backtest-seats.json` (generados pasando
  `synthetic_backtest()` por los exportadores; sobre completo)
- Test: `tests/test_bundle_unit.py`, `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`

**Interfaces:**
- Consumes: `backtest/run_backtest.py::results_dir` (regla: `root` para `es`, `root/{scope}` para los
  demás; se replica, no se importa el script), los CSV `metrics.csv`, `by_horizon.csv`, `blocks.csv`,
  `shares.csv`, `seats.csv` y `run.json` (claves: `run_at`, `commit`, `scope`, `n_sim`, `seed`, `max_fc`,
  `nowcast_only`, `house_effects`, `industry_bias`, `dispersion`, `herding`, `composition`,
  `regional_noise`, `min_events`, `events`, `horizons`, `db_polls`, `db_last_poll`, `note`),
  `publish.provenance`, `bundle.jsonable`, `BundleWriter.write_json/write_csv`, el patrón de
  `publish_analysis` (orden de escritura, `NothingToPublish`) y `Publish._analysis_scope`.
- Produces:
  - `bundle.BACKTEST_PARTS = ('meta', 'metrics', 'by-horizon', 'blocks', 'shares', 'seats')`,
    `bundle.BACKTEST_SCHEMAS = {part: 'backtest-' + part for part in BACKTEST_PARTS}`,
    `bundle.BACKTEST_CSV = BACKTEST_PARTS[1:]`; `SCHEMAS` gana `backtest-meta` (`scope: str, run_at:
    NULLABLE_STR, commit: NULLABLE_STR, params: dict, events: list, horizons: list, n_cases: int,
    n_seats_cases: int, source: str, published: dict, seconds: dict`) y `backtest-metrics`,
    `backtest-by-horizon`, `backtest-blocks`, `backtest-shares`, `backtest-seats`, todos `{'columns':
    list, 'rows': list}`.
  - `bundle.path_backtest(scope, part) -> 'backtest/{scope}/{part}.json'`,
    `bundle.path_backtest_csv(scope, name) -> 'backtest/{scope}/csv/{name}.csv'` (misma validación que
    `_check_analysis`, generalizada en `_check_file(scope, name, names, kind)`);
    `BundleReader.backtest_exists(scope) -> bool` (existe `meta.json`).
  - `publish.BACKTEST_FILES = {'metrics': 'metrics.csv', 'by-horizon': 'by_horizon.csv', 'blocks':
    'blocks.csv', 'shares': 'shares.csv', 'seats': 'seats.csv'}`; `publish.PROB_PREFIXES = ('cov_',
    'brier', 'log_score', 'p_majority')`.
  - `publish.backtest_source(scope, root=None) -> str`: `root` o `REPO_ROOT/backtest/results`, más
    `/{scope}` salvo para `es`.
  - `publish.export_backtest_table(frame) -> (dict, DataFrame)`: `{'columns': [...], 'rows': records}`;
    columnas flotantes a 2 decimales salvo las que empiezan por `PROB_PREFIXES` (3); enteras como `int`;
    booleanas (`seats_valid`, `majority`, `main`) como `bool`; `NaN` → `null`; la `DataFrame` devuelta es
    la redondeada (gemelo CSV).
  - `publish.export_backtest_meta(scope, run, tables, source, prov, seconds) -> dict`: `run` es el
    `run.json` (o `{}`); `run_at`, `commit`, `events`, `horizons` de `run` (sin `run.json`, `events` y
    `horizons` son los valores únicos ordenados de `metrics`); `params` = las claves restantes de `run`
    menos `run_at`, `commit`, `scope`, `events`, `horizons`, `note`; `n_cases = len(metrics)`,
    `n_seats_cases = int(metrics.seats_valid.sum())`; `source` = ruta relativa a `REPO_ROOT` (absoluta si
    está fuera); `published` = `prov` (`commit`, `dirty`, `versions` de la publicación).
  - `publish.publish_backtest(scope, writer, source=None, prov=None) -> dict`: sin `metrics.csv` en
    `backtest_source(scope, source)` → `NothingToPublish('no backtest results for {scope}')`; lee
    `run.json` si existe; escribe las cinco tablas en el orden de `BACKTEST_CSV` (JSON validado + CSV) y
    `meta` el último; devuelve `{'scope', 'n_cases', 'n_events', 'seconds'}`.
  - `Publish.WHAT = ('forecast', 'analysis', 'event', 'backtest', 'manifest', 'point', 'unpublish')`
    (orden de ejecución); `DEFERRED` desaparece con su rama; `run(..., source=None)`;
    `_backtest_scope(scope, writer, source)` como `_analysis_scope`; línea `'{scope}: backtest published
    ({n_cases} cases, {n_events} elections) in {s} s'` / `'{scope}: backtest skipped ({reason})'` /
    `'{scope}: backtest failed ({reason})'`; `dry_run` escribe bajo `site-dry/v1/backtest/`.
  - `tests/fakes.py`: `synthetic_backtest() -> SimpleNamespace(run, metrics, by_horizon, blocks, shares,
    seats)` con elecciones `['2019-11-10', '2023-07-23']` × horizontes `[6, 30]` (4 casos; `seats_valid`
    falso en `2019-11-10 × 30`), partidos `NAMES`, bloques `vs`, decimales de sobra y algún `NaN`; valores
    fijados por los tests: `by_horizon` fila `horizon 6`: `n_cases 2, n_seats_cases 2, mae_shares 1.6896,
    cov_shares95 0.9567, brier_vs 0.036605`; `seats` primera fila: `event_date 2019-11-10, horizon 6,
    party PSOE, official 120, model 118, mean 118.432, seats_valid True`; última fila de `seats` con
    `lo50 NaN`; `run` = el dict de `run.json` del repositorio con `events`/`horizons` sintéticos.
    `write_backtest_source(root, scope='es') -> str` escribe los cinco CSV (`by_horizon.csv`) y `run.json`
    en `backtest_source(scope, root)` y devuelve esa carpeta.

- [ ] **Step 1: Tests que fallan**

`tests/test_publish_unit.py`:

```python
def test_publish_backtest_writes_the_parts_and_meta_last(tmp_path):
    write_backtest_source(str(tmp_path / 'bt'))
    fs = RecordingFileSystem(str(tmp_path / 'site'))
    writer = bundle.BundleWriter(fs)
    out = publish.publish_backtest('es', writer, source=str(tmp_path / 'bt'), prov=fake_prov)
    assert out['scope'] == 'es' and out['n_cases'] == 4 and out['n_events'] == 2
    written = [p for p in fs.written if p.startswith('site/v1/backtest/es/')]
    assert written[-1].endswith('/meta.json')
    assert sum(p.endswith('.json') for p in written) == 6 and sum(p.endswith('.csv') for p in written) == 5
    meta = writer.read_json(bundle.path_backtest('es', 'meta'))
    assert meta['schema'] == 'backtest-meta@1' and meta['run_id'] is None and meta['mode'] is None
    assert meta['data']['events'] == ['2019-11-10', '2023-07-23'] and meta['data']['horizons'] == [6, 30]
    assert meta['data']['params']['n_sim'] == 500 and 'note' not in meta['data']['params']
    assert meta['data']['n_seats_cases'] == 3 and meta['data']['published'] == fake_prov()
    hz = writer.read_json(bundle.path_backtest('es', 'by-horizon'))['data']
    assert hz['columns'][:3] == ['horizon', 'n_cases', 'n_seats_cases']
    assert hz['rows'][0] == {**hz['rows'][0], 'horizon': 6, 'mae_shares': 1.69, 'cov_shares95': 0.957, 'brier_vs': 0.037}
    seats = writer.read_json(bundle.path_backtest('es', 'seats'))['data']['rows']
    assert seats[0]['official'] == 120 and seats[0]['mean'] == 118.43 and seats[0]['seats_valid'] is True
    assert seats[-1]['lo50'] is None
    assert bundle.BundleReader(fs).backtest_exists('es')


def test_publish_backtest_without_results_is_nothing_to_publish(tmp_path):
    with pytest.raises(publish.NothingToPublish, match='no backtest results for es-cb'):
        publish.publish_backtest('es-cb', make_writer(tmp_path), source=str(tmp_path / 'empty'))


def test_publish_backtest_overwrites_and_keeps_the_old_meta_on_failure(tmp_path, monkeypatch):
    # como test_publish_analysis_overwrites_and_keeps_the_old_meta_on_failure, rompiendo
    # publish.export_backtest_table en su quinta llamada (seats) y comprobando generated_at de meta
    ...


def test_publish_backtest_converts_the_repository_results(tmp_path):
    """La conversión real de `backtest/results` (datos del repositorio, sin base): seis JSON válidos."""
    root = os.path.join(publish.REPO_ROOT, 'backtest', 'results')
    if not os.path.isfile(os.path.join(root, 'metrics.csv')):
        pytest.skip('backtest/results no está en el repositorio')
    writer = make_writer(tmp_path)
    out = publish.publish_backtest('es', writer, prov=fake_prov)
    meta = writer.read_json(bundle.path_backtest('es', 'meta'))['data']
    assert out['n_events'] == len(meta['events']) == 5 and meta['source'] == 'backtest/results'
    assert sorted(meta['horizons']) == [6, 14, 30, 60, 90, 180]
```

`tests/test_jobs_publish.py` (en `patched`, parchear además `publish.provenance` con un dict fijo):

```python
def test_backtest_action_publishes_each_scope(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish
    write_backtest_source(str(tmp_path / 'bt'))
    fs = FileSystem(str(tmp_path / 'site'))
    result = Publish().run(what=['backtest'], scopes=['es', 'es-cb'], fs=fs, source=str(tmp_path / 'bt'))
    assert result['es']['status'] == 'published' and result['es-cb'] == {'status': 'skipped', 'reason': 'no backtest results for es-cb'}
    out = capsys.readouterr().out
    assert 'es: backtest published (4 cases, 2 elections) in ' in out and 'es-cb: backtest skipped (no backtest results for es-cb)' in out
    assert bundle.BundleReader(fs).backtest_exists('es') and 'manifest' not in out
```

`test_deferred_and_unknown_what_are_rejected` pasa a `test_unknown_what_is_rejected` sin el caso `phase 5`.

`tests/test_bundle_unit.py`: los seis fixtures validan; `path_backtest('es', 'nope')` y
`path_backtest('es/..', 'meta')` → `ValueError`; `path_backtest_csv('es', 'meta')` → `ValueError`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py -q -k backtest`
Expected: FAIL (atributos inexistentes).

- [ ] **Step 3: Implementar según Interfaces** (y generar los fixtures con los exportadores)

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde.

- [ ] **Step 5: Contrato**

`docs/web/contrato.md`: línea `backtest/{scope}/` en el árbol; sección "Backtest" (partes, `columns`/`rows`,
redondeos, origen `backtest/README.md` para las métricas, gemelos CSV, sobrescritura, `meta` último).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish.py mtpy/jobs/Publish.py tests/fakes.py tests/fixtures/bundle tests/test_bundle_unit.py tests/test_publish_unit.py tests/test_jobs_publish.py docs/web/contrato.md
git commit -m "Publish the model backtest into the bundle (publish backtest)"
```

---

### Task 4: `publish event`: una elección pasada en el paquete

**Files:**
- Modify: `mtpy/lib/bundle.py` (`EVENT_*`, `DATE_RE`, `SCHEMAS`, `path_event`, `path_event_csv`,
  `BundleReader.list_events`, `event_exists`), `mtpy/lib/publish.py`, `mtpy/lib/data.py` (`get_events`),
  `mtpy/jobs/Publish.py`, `tests/fakes.py`, `docs/web/contrato.md` (sección "Eventos")
- Create: `tests/fixtures/bundle/event-meta.json`, `event-pollsters.json`, `event-polls.json`,
  `event-results.json`, `event-model.json`
- Test: `tests/test_bundle_unit.py`, `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`,
  `tests/integration/test_publish.py`

**Interfaces:**
- Consumes: `Computer(scope=, event_dates=[date], path='.', verbose=).build_series()` y
  `comp.get_polls_metric('error', bmap='vs')` (índice `(event, pollster, days)`, columnas `event_date,
  sample_size, proc_sample, weight, error_avg, bias_avg, error_blocks, bias_blocks, error_within,
  bias_within, bias` + un signo por bloque; todos los sondeos del evento como el notebook, celdas 4-5),
  `comp.event_params[date]['bmaps']` (`vs`, `main`), `data.get_event_results(scope, date, 0)` (`party_id,
  party, votes, pct, seats`), `data.get_pollsters()` (`id`, `name`), `models.Events` (`date, scope, name,
  featured`), `publish.export_poll_errors(frame, blocks)`, `publish.export_backtest_table` y
  `backtest_source` (T3), `publish.provenance`, `publish.event_dates` ISO (`ISO_DAY_RE`).
- Produces:
  - `data.get_events(scope) -> DataFrame` (`date`, `scope`, `name`, `featured`) =
    `Events().get_results(query={'filters': ["scope = '{}'".format(scope)]}, formatted=True)`.
  - `bundle.EVENT_PARTS = ('meta', 'pollsters', 'polls', 'results', 'model')`, `EVENT_SCHEMAS = {part:
    'event-' + part}`, `EVENT_CSV = ('pollsters', 'polls', 'results', 'model-shares', 'model-seats',
    'model-blocks')`, `bundle.DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')`; `path_event(scope, date,
    part) -> 'events/{scope}/{date}/{part}.json'`, `path_event_csv(scope, date, name) ->
    'events/{scope}/{date}/csv/{name}.csv'` (`date` ha de casar con `DATE_RE`, si no `ValueError`);
    `BundleReader.list_events(scope) -> list[str]` (fechas ascendentes con `meta.json`; `[]` sin
    carpeta), `BundleReader.event_exists(scope, date) -> bool`.
  - `SCHEMAS`: `event-meta` (`scope: str, event_date: str, name: NULLABLE_STR, n_polls: int,
    n_pollsters: int, blocks: dict, main: list, summary: dict, has_model: bool, horizons: list, commit:
    NULLABLE_STR, dirty: bool, versions: dict, seconds: dict`), `event-pollsters` (`blocks: list, rows:
    list`), `event-polls` (`blocks: list, rows: list`), `event-results` (`n_seats: int, rows: list`),
    `event-model` (`horizons: list, shares: list, seats: list, blocks: list`).
  - `publish.EVENT_POLLSTER_COLUMNS = ['pollster_id', 'pollster', 'n_polls', 'sample_size', 'error_avg',
    'error_blocks', 'bias_blocks', 'last_days', 'last_error_avg', 'last_error_blocks', 'last_bias_blocks']`
    + un `last_{bloque}` por bloque (desviación con signo del último sondeo).
  - `publish.export_event_polls(frame, blocks, ids) -> (dict, DataFrame)`: filas de
    `export_poll_errors` más `pollster_id` (entero o `null`, de `ids` = `{name: id}`), ordenadas por
    `days` ascendente (el más reciente primero).
  - `publish.export_event_pollsters(frame, blocks, ids) -> (dict, DataFrame)`: una fila por casa:
    `n_polls` (recuento), `sample_size` (mediana, `int`), `error_avg`, `error_blocks`, `bias_blocks`
    (medias, 2 dec), `last_*` de la fila con menor `days`; orden `error_blocks` ascendente, nulos al final.
  - `publish.export_event_results(frame) -> (dict, DataFrame)`: filas `party_id, party, votes (int), pct
    (2 dec), seats (int)` por votos descendente; `n_seats` = suma de `seats`.
  - `publish.export_event_model(source, event_date) -> (dict, dict[str, DataFrame])`: de
    `shares.csv`, `seats.csv`, `blocks.csv` de `source` (si existen) las filas con ese `event_date`,
    redondeadas como `export_backtest_table`; `horizons` ascendentes; sin fuente, listas vacías y
    `horizons: []`; los CSV se llaman `model-shares`, `model-seats`, `model-blocks`.
  - `publish.event_summary(frame, blocks) -> dict`: `n_polls`, `n_pollsters`, medias ponderadas por
    `weight` de `error_avg`, `error_blocks`, `bias_blocks` (2 dec) y `bias` = `{bloque: media ponderada
    de su desviación con signo}`.
  - `publish.export_event_meta(scope, event_date, name, blocks, main, summary, model, prov, seconds) -> dict`
    (`has_model = bool(model['shares'])`, `horizons = model['horizons']`).
  - `publish.event_dates_to_publish(scope, event_date, events=None, today=None) -> list[str]`: `'all'`
    → fechas `featured` del ámbito anteriores a `today` (Europe/Madrid, `_today`); cadena → `[cadena]`;
    lista → lista; cada fecha ha de ser ISO real (si no, `ValueError`).
  - `publish.publish_event(scope, event_date, writer, computer=None, events=None, results=None,
    pollsters=None, source=None, prov=None, verbose=0) -> dict`: la fecha no es evento `featured` del
    ámbito → `NothingToPublish('{date} is not a featured election of {scope}')`; resultados de la región
    0 vacíos → `NothingToPublish('no official results for {scope} {date}')`; `ValueError` del `Computer`
    o tabla de sondeos vacía → `NothingToPublish`; escribe `pollsters`, `polls`, `results`, `model` (JSON
    validado + CSV) y `meta` el último; devuelve `{'scope', 'event_date', 'n_polls', 'n_pollsters',
    'has_model', 'seconds'}`.
  - `Publish.run(what=['event'], scopes=..., event_date=<str|list|'all'>, source=None)`: sin
    `event_date` → `ValueError('publish: "event_date" is required for event')`; `event` junto a
    `forecast` → `ValueError('publish: "event" cannot be combined with "forecast"')`; `results[scope]` =
    lista de resultados por fecha (`status` `published` con `event_date`, `n_polls`, `n_pollsters`,
    `has_model`, `seconds`; `skipped`/`failed` con `reason` y `event_date`); líneas `'{scope}: event
    {date} published ({n} polls, {k} pollsters) in {s} s'` / `'{scope}: event {date} skipped ({reason})'`
    / `'{scope}: event {date} failed ({reason})'`; `dry_run` escribe bajo `site-dry/v1/events/`.
  - `tests/fakes.py`: `synthetic_event() -> SimpleNamespace(events, results, pollsters)`: `events` con
    `('2019-11-10', 'es', 'Generales 2019 (noviembre)', True)`, `('2023-07-23', 'es', 'Generales 2023',
    True)`, `('2026-11-29', 'es', None, False)`; `results` (columnas `date, scope, region_id, party_id,
    party, votes, pct, seats`) solo de `2023-07-23` y región 0: una fila por `NAMES` más `('Cs', 0.13,
    0 escaños)`, con `pct` de más decimales y `seats` que suman 350; `pollsters` con `id`/`name` de `HOUSES` (`CIS` 1, `GAD3` 2, `40dB` 3). `FakeComputer`
    acepta `event_dates` (lo anota en `calls`); su `get_polls_metric` ya devuelve los tres sondeos de
    `2023-07-23` (`GAD3` días 7, `40dB` días 8, `CIS` días 13).

- [ ] **Step 1: Tests que fallan**

`tests/test_publish_unit.py`:

```python
def event_readers(ev):
    return dict(events=lambda scope: ev.events, results=lambda scope, date, region: ev.results.loc[ev.results.date == date],
                pollsters=lambda: ev.pollsters)


def test_publish_event_writes_the_parts_and_meta_last(tmp_path):
    ev = synthetic_event()
    write_backtest_source(str(tmp_path / 'bt'))
    fs = RecordingFileSystem(str(tmp_path / 'site'))
    writer = bundle.BundleWriter(fs)
    out = publish.publish_event('es', '2023-07-23', writer, computer=FakeComputer(), source=str(tmp_path / 'bt'),
                                prov=fake_prov, **event_readers(ev))
    assert out['n_polls'] == 3 and out['n_pollsters'] == 3 and out['has_model'] is True
    written = [p for p in fs.written if p.startswith('site/v1/events/es/2023-07-23/')]
    assert written[-1].endswith('/meta.json') and sum(p.endswith('.json') for p in written) == 5 and sum(p.endswith('.csv') for p in written) == 6
    meta = writer.read_json(bundle.path_event('es', '2023-07-23', 'meta'))
    assert meta['schema'] == 'event-meta@1' and meta['data']['name'] == 'Generales 2023' and meta['data']['horizons'] == [6, 30]
    assert set(meta['data']['summary']) == {'n_polls', 'n_pollsters', 'error_avg', 'error_blocks', 'bias_blocks', 'bias'}
    assert list(meta['data']['blocks']) == ['Derecha', 'Izquierda'] and meta['data']['main'] == list(BMAPS['main'])
    houses = writer.read_json(bundle.path_event('es', '2023-07-23', 'pollsters'))['data']['rows']
    assert [r['pollster'] for r in houses] == ['GAD3', '40dB', 'CIS']
    assert houses[0] == {**houses[0], 'pollster_id': 2, 'n_polls': 1, 'last_days': 7, 'last_Derecha': 0.68}
    polls = writer.read_json(bundle.path_event('es', '2023-07-23', 'polls'))['data']['rows']
    assert [r['days'] for r in polls] == [7, 8, 13] and polls[1]['pollster_id'] == 3
    results = writer.read_json(bundle.path_event('es', '2023-07-23', 'results'))['data']
    assert results['n_seats'] == 350 and results['rows'][-1]['party'] == 'Cs' and results['rows'][0]['votes'] >= results['rows'][1]['votes']
    model = writer.read_json(bundle.path_event('es', '2023-07-23', 'model'))['data']
    assert {r['event_date'] for r in model['shares']} == {'2023-07-23'} and model['horizons'] == [6, 30]
    assert bundle.BundleReader(fs).list_events('es') == ['2023-07-23']


def test_publish_event_without_backtest_has_an_empty_model(tmp_path):
    ... has_model False, model == {'horizons': [], 'shares': [], 'seats': [], 'blocks': []}


@pytest.mark.parametrize('date, reason', [('2019-11-10', 'no official results for es 2019-11-10'),
                                          ('2026-11-29', '2026-11-29 is not a featured election of es'),
                                          ('2020-01-01', '2020-01-01 is not a featured election of es')])
def test_publish_event_skips_what_cannot_be_published(tmp_path, date, reason):
    with pytest.raises(publish.NothingToPublish, match=reason): ...


def test_publish_event_overwrites_and_keeps_the_old_meta_on_failure(tmp_path, monkeypatch):
    ... (patrón del análisis; romper `publish.export_event_model`)


def test_event_dates_to_publish():
    ev = synthetic_event()
    assert publish.event_dates_to_publish('es', 'all', events=lambda scope: ev.events, today='2026-10-11') == ['2019-11-10', '2023-07-23']
    assert publish.event_dates_to_publish('es', '2023-07-23') == ['2023-07-23']
    with pytest.raises(ValueError):
        publish.event_dates_to_publish('es', '2023-13-01')
```

`tests/test_jobs_publish.py` (`patched` envuelve `publish.publish_event` con `FakeComputer` y los
lectores de `synthetic_event`, como hace con el análisis):

```python
def test_event_action_publishes_each_date(fresh_app, tmp_path, patched, capsys):
    fs = FileSystem(str(tmp_path / 'site'))
    result = Publish().run(what=['event'], scopes=['es'], event_date=['2023-07-23', '2019-11-10'], fs=fs)
    assert [r['status'] for r in result['es']] == ['published', 'skipped']
    out = capsys.readouterr().out
    assert 'es: event 2023-07-23 published (3 polls, 3 pollsters) in ' in out
    assert 'es: event 2019-11-10 skipped (no official results for es 2019-11-10)' in out
    assert bundle.BundleReader(fs).event_exists('es', '2023-07-23') and 'manifest' not in out


def test_event_requires_a_date_and_excludes_forecast(fresh_app, tmp_path):
    with pytest.raises(ValueError, match='event_date'):
        Publish().run(what=['event'], fs=FileSystem(str(tmp_path)))
    with pytest.raises(ValueError, match='cannot be combined'):
        Publish().run(what=['event', 'forecast'], event_date='2023-07-23', fs=FileSystem(str(tmp_path)))
```

`tests/test_bundle_unit.py`: los cinco fixtures validan; `path_event('es', '2023-7-23', 'meta')`,
`path_event('es', '2023-07-23', 'nope')` y `path_event_csv('es', '2023-07-23', 'meta')` → `ValueError`.

`tests/integration/test_publish.py` (marcado): `publish_event('es', '2023-07-23', BundleWriter(FileSystem(tmp_path)))`
contra la RDS: cinco JSON válidos, `results.n_seats == 350`, `polls.rows` no vacío con `days >= 0`, toda
`pollster` de `polls` está en `pollsters`, `has_model` igual a la existencia de `backtest/results/shares.csv`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py -q -k event`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces** (fixtures generados con los exportadores)

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde.

- [ ] **Step 5: Contrato**

`docs/web/contrato.md`: línea `events/{scope}/{date}/` en el árbol; sección "Eventos" (las cinco partes y
sus columnas, `summary` de `meta`, `last_*`, `model` como corte del backtest, gemelos CSV, sobrescritura).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish.py mtpy/lib/data.py mtpy/jobs/Publish.py tests/fakes.py tests/fixtures/bundle tests/test_bundle_unit.py tests/test_publish_unit.py tests/test_jobs_publish.py tests/integration/test_publish.py docs/web/contrato.md
git commit -m "Publish past elections into the bundle (publish event)"
```

---

### Task 5: API: `/api/v1/events/{scope}`, `/api/v1/events/{scope}/{date}/{part}`, `/api/v1/backtest/{scope}/{part}`

**Files:**
- Modify: `mtpy/lib/webapi/checks.py` (`EVENT_PARTS`, `BACKTEST_PARTS`), `mtpy/lib/webapi/published.py`
  (`Site.events`, `event`, `event_csv`, `backtest`, `backtest_csv`), `mtpy/lib/webapi/queries.py`
  (`get_events`, `query_events`), `mtpy/lib/webapi/__init__.py` (rutas y reexportaciones),
  `mtpy/lib/pages/common.py` (`api_url`, `MESSAGES`), `docs/web/contrato.md` (tabla de rutas),
  `tests/test_webapi_unit.py` (`write_event`, `write_backtest`), `tests/test_web_api.py`, `tests/test_pages.py`
- Create: `mtpy/controllers/Events.py`, `mtpy/controllers/Backtest.py`

**Interfaces:**
- Consumes: `Site._analysis_bytes` (patrón de caché de punteros para ficheros sobrescribibles),
  `Site.scopes` (cruce catálogo/manifest), `Data.serve`, `Analysis.part_action`, `bundle.path_event*`,
  `path_backtest*`, `BundleReader.list_events` (T4), `data.get_events` (T4), `check_date`, `check_part`,
  `check_format`.
- Produces:
  - `checks.EVENT_PARTS = bundle.EVENT_PARTS`, `checks.BACKTEST_PARTS = bundle.BACKTEST_PARTS`
    (reexportados por el paquete).
  - `Site._mutable_bytes(kind, path, message) -> bytes` generaliza `_analysis_bytes` (clave `(kind,
    path)` en `pointers`, 404 `message` si falta); `Site.events(scope) -> list[str]` (clave `('events',
    scope)`, `reader.list_events`); `Site.event(scope, date, part) -> bytes` y `Site.event_csv(scope,
    date, name)` (404 `event not published`); `Site.backtest(scope, part)` y `Site.backtest_csv(scope,
    name)` (404 `backtest not published`).
  - `queries.get_events = _reader('get_events')`; `queries.query_events(scope) -> DataFrame` con clave
    `('events_table', scope)`.
  - `Events(Data)`: `index_action(scope)`: `check_scope`; tabla con `date` (ISO), `name`, `featured`
    (`bool`), `has_page` (`date in site().events(scope)`), ascendente por fecha; `self.serve(frame,
    'events-' + scope, scope=scope)`. `part_action(scope, date, part)`: `check_scope`, `check_date`,
    `check_part(part, EVENT_PARTS)`, `check_format`, `notice_freeze`; `date not in site().events(scope)`
    → 404 `event not published` antes de leer; CSV: `meta` → 404 `no csv for this part`; `model` exige
    `table` ∈ (`shares`, `seats`, `blocks`) y el nombre es `model-{table}`; en las demás partes `table`
    ha de estar ausente; fichero `{scope}-{date}-{name}.csv`; JSON tal cual con `json_bytes`.
  - `Backtest(Base)`: `part_action(scope, part)` como `Analysis.part_action` con `BACKTEST_PARTS`;
    CSV `{scope}-backtest-{name}.csv`; `meta` sin gemelo.
  - `ROUTES` gana, tras la ruta de `analysis`: `('/api/v1/events/{scope}', 'events', 'index', ['GET'])`,
    `('/api/v1/events/{scope}/{date}/{part}', 'events', 'part', ['GET'])`, `('/api/v1/backtest/{scope}/{part}',
    'backtest', 'part', ['GET'])`.
  - `pages.api_url(scope, part, ..., kind=, table=, date=None)`: `kind='events'` →
    `/api/v1/events/{scope}[?format=]`; `kind='event'` (exige `date`) →
    `/api/v1/events/{scope}/{date}/{part}[?format=&table=]`; `kind='backtest'` →
    `/api/v1/backtest/{scope}/{part}[?format=]`.
  - `pages.MESSAGES` gana `'invalid date': 'Fecha no válida.'`, `'event not published': 'Elección no
    publicada.'`, `'backtest not published': 'Backtest no publicado.'`.
  - `tests/test_webapi_unit.py`: `write_event(fs, scope='es', date='2023-07-23', model=True)` escribe los
    cinco fixtures (con `event_date` y `scope` sustituidos en cada sobre y en `meta.data`; con
    `model=False`, `model` con listas vacías y `has_model` falso) y seis CSV mínimos;
    `write_backtest(fs, scope='es')` escribe los seis fixtures y cinco CSV.

- [ ] **Step 1: Tests que fallan**

`tests/test_webapi_unit.py`:

```python
def test_site_events_and_backtest_read_the_published_files(site_app):
    fs = site_app.fs
    write_event(fs); write_event(fs, date='2019-11-10', model=False); write_backtest(fs)
    s = webapi.site()
    assert s.events('es') == ['2019-11-10', '2023-07-23'] and s.events('es-cb') == []
    assert json.loads(s.event('es', '2023-07-23', 'pollsters'))['schema'] == 'event-pollsters@1'
    assert s.event_csv('es', '2023-07-23', 'model-seats').startswith(b'event_date')
    assert json.loads(s.backtest('es', 'by-horizon'))['data']['rows'][0]['horizon'] == 6
    for call, message in ((lambda: s.event('es', '2020-01-01', 'meta'), 'event not published'),
                          (lambda: s.backtest('es-cb', 'metrics'), 'backtest not published')):
        with pytest.raises(HttpError) as err:
            call()
        assert err.value.status == 404 and err.value.message == message
```

`tests/test_web_api.py`:

```python
def test_events_list_crosses_the_database_with_the_bundle(db_api, fresh_app, tmp_path):
    # `fake_readers` parchea además webapi.queries.get_events con synthetic_event().events; write_event(fs)
    status, headers, body = call(db_api, '/api/v1/events/es')
    assert status == 200 and [r['date'] for r in body['rows']] == ['2019-11-10', '2023-07-23', '2026-11-29']
    assert [r['has_page'] for r in body['rows']] == [False, True, False] and body['rows'][2]['featured'] is False
    status, headers, _ = call(db_api, '/api/v1/events/es', query='format=csv')
    assert headers['content-disposition'] == 'attachment; filename="events-es.csv"'


def test_events_list_without_database_is_a_503(api):
    status, _, body = call(api, '/api/v1/events/es')
    assert status == 503 and body['message'] == 'no database configured'


def test_event_parts_and_csv(api, fresh_app):
    # sin base: write_event(fs)
    status, headers, body = call(api, '/api/v1/events/es/2023-07-23/pollsters')
    assert status == 200 and body['schema'] == 'event-pollsters@1' and 'immutable' not in headers['cache-control']
    status, headers, _ = call(api, '/api/v1/events/es/2023-07-23/model', query='format=csv&table=seats')
    assert headers['content-type'].startswith('text/csv') and 'es-2023-07-23-model-seats.csv' in headers['content-disposition']
    status, headers, _ = call(api, '/api/v1/events/es/2023-07-23/polls', query='format=csv')
    assert 'es-2023-07-23-polls.csv' in headers['content-disposition']


def test_backtest_parts_and_csv(api, fresh_app):
    # write_backtest(fs): /api/v1/backtest/es/metrics 200 'backtest-metrics@1'; ?format=csv → es-backtest-metrics.csv
    ...


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/events/es/2023-13-01/meta', '', 400, 'invalid date'),
    ('/api/v1/events/es/2019-11-10/meta', '', 404, 'event not published'),
    ('/api/v1/events/es/2023-07-23/nope', '', 404, 'unknown part'),
    ('/api/v1/events/es/2023-07-23/meta', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/events/es/2023-07-23/model', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/events/es/2023-07-23/polls', 'format=csv&table=seats', 404, 'no csv for this part'),
    ('/api/v1/events/es-xx', '', 400, 'invalid scope'),
    ('/api/v1/backtest/es-cb/metrics', '', 404, 'backtest not published'),
    ('/api/v1/backtest/es/meta', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/backtest/es/nope', '', 404, 'unknown part'),
])
def test_events_and_backtest_errors_are_json_and_not_cached(api, fresh_app, path, query, status, message):
    write_event(fresh_app.fs)
    write_backtest(fresh_app.fs)
    st, headers, body = call(api, path, query=query)
    assert st == status and body == {'status': 'error', 'message': message} and headers['cache-control'] == 'no-store'
```

`tests/test_pages.py`:

```python
def test_api_url_covers_events_and_backtest():
    assert pages.api_url('es', 'pollsters', kind='event', date='2023-07-23', fmt='csv') == '/api/v1/events/es/2023-07-23/pollsters?format=csv'
    assert pages.api_url('es', 'model', kind='event', date='2023-07-23', fmt='csv', table='seats') == '/api/v1/events/es/2023-07-23/model?format=csv&table=seats'
    assert pages.api_url('es', 'metrics', kind='backtest', fmt='csv') == '/api/v1/backtest/es/metrics?format=csv'
    assert pages.api_url('es', 'events', kind='events') == '/api/v1/events/es'
```

`tests/test_web_routes.py` ya comprueba que los literales de las plantillas casan con `ROUTES`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py -q -k "events or event_parts or backtest or api_url"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde.

- [ ] **Step 5: Contrato**

`docs/web/contrato.md`: sección "Rutas de eventos y backtest (fase 5)" con la tabla de las tres rutas,
parámetros, CSV y los errores nuevos (`event not published`, `backtest not published`); la enmienda
2026-10-10 de la spec deja de decir que `events` y `backtest` siguen en la fase 5 (se corrige en T9).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/webapi mtpy/lib/pages/common.py mtpy/controllers/Events.py mtpy/controllers/Backtest.py tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py docs/web/contrato.md
git commit -m "Serve the published events and backtest through the API"
```

---

### Task 6: página `/elecciones` (lista con el error del sector y detalle `?date=`)

**Files:**
- Create: `mtpy/lib/pages/elecciones.py`, `mtpy/controllers/Elecciones.py`, `web/templates/elecciones.html`,
  `web/templates/eleccion.html`, `web/dist/js/pages/elecciones.js`
- Modify: `mtpy/lib/pages/common.py` (`PAGES`, `ROUTES`), `mtpy/lib/pages/__init__.py`,
  `web/dist/css/site.css` (solo si hace falta una regla nueva), `tests/test_pages.py`
- Test: `tests/test_pages.py`, `tests/test_web_routes.py` (estático, ya existe)

**Interfaces:**
- Consumes: `common_context`, `catalog`, `Catalog.__missing__`, `check_date`, `site().events/event`
  (T5), `api_url(kind='event')`, `casas.analysis_part` (patrón), `charts/bars.js::renderBars(el, rows,
  {value, formatter, colors})`, `charts/bullseye.js::renderBullseye(el, rows, {blocks})` (filas con un
  valor por bloque, como `casa_context`), `charts/line.js::renderLine(el, points, {formatter, name})`,
  `pages/common.js::{readInitial, wireControls, tryDraw, mountWhenReady}`, la plantilla `casa.html`
  (cabecera con datos, tablas, descargas) como modelo.
- Produces:
  - `pages.elecciones.event_part(scope, date, name) -> Optional[dict]` (`data` de la parte; `None` en 404).
  - `published_events(scope) -> list[dict]`: un dict por fecha publicada, de la más reciente a la más
    antigua: `date`, `name`, `n_polls`, `n_pollsters`, `error_avg`, `error_blocks`, `bias_blocks`, `bias`
    (dict por bloque), `has_model` (todo de `meta`).
  - `elecciones_context(state, active='/elecciones') -> dict`: `common_context` + `events` (cada uno con
    `href = '/elecciones' + query + '&date=' + date`), `blocks` (del `meta` más reciente; `[]` sin
    eventos), `initial = {'state', 'sector': [{'date', 'error_avg', 'error_blocks', 'bias_blocks'}]
    ascendente, 'blocks'}`.
  - `resolve_event(scope, date) -> str`: `check_date` (400) y pertenencia a `site().events(scope)` (404
    `event not published`).
  - `eleccion_context(state, date, active='/elecciones') -> dict`: `event` (`meta`), `results_rows`
    (`party`, `fullname`, `color` por `catalog(meta)`, `pct`, `seats`; en el orden publicado), `n_seats`,
    `pollster_rows` (como se publican), `poll_rows`, `blocks`, `horizon_tables` (lista por horizonte
    ascendente de `{'horizon', 'shares': filas de los partidos `main` con `name, official, model, error =
    model - official`, 'seats': ídem con `official, model`, 'blocks': filas de `blocks` con `block,
    official_seats, model_seats, p_majority, majority`}`), `has_model`, `csv_links` (seis: `pollsters`,
    `polls`, `results`, `model-shares`, `model-seats`, `model-blocks`, con etiqueta en español),
    `initial = {'state', 'meta', 'pollsters': [{'pollster', 'error_blocks', 'last_error_blocks'}],
    'bullseye': [{'pollster', 'days', <bloque>: last_<bloque>}], 'blocks'}`.
  - `Elecciones(Page)`: `active = '/elecciones'`; con `?date=` renderiza `eleccion.html`, si no
    `elecciones.html`.
  - `common.PAGES` gana `{'href': '/elecciones', 'label': 'Elecciones'}` tras Casas; `common.ROUTES`
    gana `('/elecciones', 'elecciones', 'index', ['GET'])`; `__init__` reexporta `event_part`,
    `published_events`, `resolve_event`, `elecciones_context`, `eleccion_context`.
  - `elecciones.html`: tarjeta "Elecciones publicadas" (tabla: elección —enlace con `href`—, sondeos,
    casas, error medio, error de bloques, sesgo de bloques, modelo "sí"/"—"), tarjeta "Error del sector"
    con `div.chart#sector-chart` y un texto "Qué es esto" (el error medio de los sondeos frente al
    resultado en cada elección pasada; el sesgo de bloques, con signo, dice hacia qué bloque se
    equivocaron en conjunto); sin eventos, aviso "Todavía no hay elecciones publicadas en este ámbito." y
    sin gráficos.
  - `eleccion.html`: `<h2>` con la fecha (`|date`) y el nombre; tarjetas "Resultados" (tabla con muestra
    de color, partido, %, escaños; pie "n_seats escaños"), "Casas" (`div.chart#error-chart` + tabla
    `pollster_rows`; el nombre enlaza a `/casas{{ query }}&id=` cuando hay `pollster_id`), "Último
    sondeo de cada casa" (`div.chart#bullseye-chart`), "Sondeos" (tabla `poll_rows`: casa, días, muestra,
    error medio, error de bloques, sesgo de bloques, un valor por bloque), "El modelo a cada horizonte"
    (una tabla por horizonte; sin backtest, aviso "Sin backtest para esta elección."), "Descargas".
  - `pages/elecciones.js`: lista → `renderLine(#sector-chart, puntos [date, error_avg], {name: 'Error
    medio del sector'})`; detalle → `renderBars(#error-chart, filas {name: pollster, value:
    error_blocks})` y `renderBullseye(#bullseye-chart, bullseye, {blocks})`; todo por `tryDraw` y solo si
    el contenedor existe y hay datos.

- [ ] **Step 1: Tests que fallan**

`tests/test_pages.py` (fixture `events_site`: `api` + `write_event(fs)` + `write_event(fs,
date='2019-11-10', model=False)`; actualizar la lista de `test_pages_package_layout` con `'elecciones'`):

```python
def test_elecciones_lists_the_published_events_newest_first(events_site):
    status, headers, html = call(events_site, '/elecciones', query='scope=es')
    assert status == 200 and html.index('date=2023-07-23') < html.index('date=2019-11-10')
    assert '/elecciones?scope=es&amp;mode=forecast&amp;date=2023-07-23' in html
    initial = json.loads(re.search(r'id="initial-data">(.*?)</script>', html, re.S).group(1))
    assert [row['date'] for row in initial['sector']] == ['2019-11-10', '2023-07-23'] and initial['blocks'] == ['Derecha', 'Izquierda']


def test_elecciones_without_events_shows_a_notice(api):
    status, _, html = call(api, '/elecciones', query='scope=es')
    assert status == 200 and 'Todavía no hay elecciones publicadas' in html and 'initial-data' not in html


def test_eleccion_detail_renders_results_pollsters_bullseye_and_model(events_site):
    status, _, html = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    assert status == 200 and 'Generales 2023' in html
    assert html.count('/api/v1/events/es/2023-07-23/') == 6
    assert html.index('GAD3') < html.index('40dB') < html.index('CIS')
    assert '/casas?scope=es&amp;mode=forecast&amp;id=2' in html
    assert 'Cs' in html and pages.OTHERS_COLOR in html  # partido fuera del catálogo del run (D9)
    initial = ...
    assert set(initial) == {'state', 'meta', 'pollsters', 'bullseye', 'blocks'} and initial['bullseye'][0]['Derecha'] == 0.68


def test_eleccion_without_model_shows_a_notice(events_site):
    status, _, html = call(events_site, '/elecciones', query='scope=es&date=2019-11-10')
    assert status == 200 and 'Sin backtest para esta elección' in html


@pytest.mark.parametrize('query, status, text', [('scope=es&date=2023-13-01', 400, 'Fecha no válida.'),
                                                 ('scope=es&date=2020-01-01', 404, 'Elección no publicada.')])
def test_eleccion_errors(events_site, query, status, text):
    st, headers, html = call(events_site, '/elecciones', query=query)
    assert st == status and text in html and headers['cache-control'] == 'no-store'


def test_published_events_and_horizon_tables():
    # funciones puras sobre los fixtures: orden descendente; horizon_tables con error = model - official (2 dec)
    ...
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "eleccion or layout"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces**; `node --check web/dist/js/pages/elecciones.js`

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde, incluido `tests/test_web_routes.py` (recursos, imports, `aria-label`).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages mtpy/controllers/Elecciones.py web/templates/elecciones.html web/templates/eleccion.html web/dist/js/pages/elecciones.js web/dist/css/site.css tests/test_pages.py
git commit -m "Add the elections page: published events, sector error and the detail of one election"
```

---

### Task 7: página `/modelo` (precisión del modelo por horizonte)

**Files:**
- Create: `mtpy/lib/pages/modelo.py`, `mtpy/controllers/Modelo.py`, `web/templates/modelo.html`,
  `web/dist/js/pages/modelo.js`
- Modify: `web/dist/js/charts/line.js` (`renderLines`), `mtpy/lib/pages/common.py` (`PAGES`, `ROUTES`),
  `mtpy/lib/pages/__init__.py`, `tests/test_pages.py`

**Interfaces:**
- Consumes: `site().backtest` (T5), `api_url(kind='backtest')`, `charts/line.js::renderLine` (ECharts
  `line` con `THEME`), `backtest/README.md` (significado de cada métrica para los textos).
- Produces:
  - `pages.modelo.HORIZON_COLUMNS = ('horizon', 'n_cases', 'n_seats_cases', 'mae_shares',
    'mae_last_poll', 'mae_mean_4w', 'mae_prev_result', 'cov_shares50', 'cov_shares80', 'cov_shares95',
    'mae_seats', 'cov_seats95', 'crps_seats', 'brier_vs', 'mae_seats_h', 'cov_shares95_h', 'brier_vs_h')`,
    `CASE_COLUMNS = ('event_date', 'horizon', 'mae_shares', 'mae_last_poll', 'mae_seats',
    'cov_shares95', 'brier_vs', 'seats_valid')`, `BLOCK_COLUMNS = ('event_date', 'horizon', 'block',
    'official_seats', 'model_seats', 'p_majority', 'majority')`.
  - `backtest_part(scope, name) -> Optional[dict]`; `project(rows, columns) -> list[dict]` (filas con solo
    esas claves, `None` si faltan); `case_rows(rows) -> list[dict]` (orden: `event_date` descendente,
    `horizon` ascendente; vale para `metrics` y para `blocks`).
  - `modelo_context(state, active='/modelo') -> dict`: `published` (hay `meta`); si lo hay: `backtest`
    (`meta`), `horizon_rows` (`by-horizon` proyectado, horizonte ascendente), `case_rows` (`metrics`:
    fecha descendente, horizonte ascendente), `block_rows` (`blocks`, mismo orden), `csv_links` (cinco,
    `metrics`, `by-horizon`, `blocks`, `shares`, `seats`), `initial = {'state', 'horizons', 'mae':
    {'model', 'last_poll', 'mean_4w', 'prev_result'}, 'coverage': {'50', '80', '95'}, 'seats': {'mae',
    'mae_h', 'cov95'}}` (listas alineadas con `horizons`); si no: solo `published = False` (sin `initial`).
  - `Modelo(Page)`; `common.PAGES` gana `{'href': '/modelo', 'label': 'Modelo'}`; `common.ROUTES` gana
    `('/modelo', 'modelo', 'index', ['GET'])`; `__init__` reexporta `backtest_part`, `project`,
    `modelo_context`.
  - `charts/line.js::renderLines(el, series, {formatter, yMax, references})`: `series = [{name, points:
    [[x, y], ...], color?, dashed?}]` sobre un eje de categorías con la unión ordenada de las `x`
    (horizontes en días); `references` = valores con línea horizontal discontinua (`markLine`)
    etiquetada; devuelve la instancia; `renderLine` no cambia.
  - `modelo.html`: tarjetas "Error del promedio por horizonte" (`div.chart#mae-chart` + tabla con
    `horizon`, `n_cases`, `mae_shares`, `mae_last_poll`, `mae_mean_4w`, `mae_prev_result`), "Cobertura
    de los intervalos" (`div.chart#coverage-chart` + tabla `cov_shares50/80/95`), "Escaños y mayorías"
    (tabla `mae_seats`, `cov_seats95`, `crps_seats`, `brier_vs`, `mae_seats_h`, `brier_vs_h`), "Por
    elección" (`case_rows`), "Bloques" (`block_rows`), "Qué es esto" (qué evalúa el backtest, fuera de
    muestra, nowcast frente a horizonte `_h`, las tres líneas base, lectura de las coberturas), "Parámetros"
    (`n_sim`, `seed`, `max_fc`, elecciones, horizontes, `commit`, `run_at`, `source`), "Descargas"; sin
    backtest, aviso "Backtest no publicado todavía para este ámbito." y nada más.
  - `pages/modelo.js`: `renderLines(#mae-chart, cuatro series)` y `renderLines(#coverage-chart, tres
    series, {references: [0.5, 0.8, 0.95], yMax: 1})`.

- [ ] **Step 1: Tests que fallan**

`tests/test_pages.py` (fixture `backtest_site`: `api` + `write_backtest(fs)`; `'modelo'` en la lista de
`test_pages_package_layout`):

```python
def test_modelo_renders_the_backtest_tables_and_initial(backtest_site):
    status, _, html = call(backtest_site, '/modelo', query='scope=es')
    assert status == 200 and '1,69' in html and html.count('/api/v1/backtest/es/') == 5
    initial = ...
    assert initial['horizons'] == [6, 30] and initial['mae']['model'][0] == 1.69 and initial['coverage']['95'][0] == 0.957
    assert list(initial['seats']) == ['mae', 'mae_h', 'cov95']


def test_modelo_without_backtest_shows_a_notice(api):
    status, _, html = call(api, '/modelo', query='scope=es')
    assert status == 200 and 'Backtest no publicado todavía' in html and 'initial-data' not in html


def test_modelo_rows_are_ordered_and_projected():
    rows = pages.project(fixture_data('backtest-metrics')['data']['rows'], pages.modelo.CASE_COLUMNS)
    ordered = pages.modelo.case_rows(rows)
    assert [(r['event_date'], r['horizon']) for r in ordered] == [('2023-07-23', 6), ('2023-07-23', 30), ('2019-11-10', 6), ('2019-11-10', 30)]
    assert set(ordered[0]) == set(pages.modelo.CASE_COLUMNS)
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "modelo or layout"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces**; `node --check web/dist/js/charts/line.js web/dist/js/pages/modelo.js`

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages mtpy/controllers/Modelo.py web/templates/modelo.html web/dist/js/pages/modelo.js web/dist/js/charts/line.js tests/test_pages.py
git commit -m "Add the model page: backtest accuracy by horizon, coverage, seats and cases"
```

---

### Task 8: página `/metodo` (texto de la metodología)

**Files:**
- Create: `mtpy/lib/pages/metodo.py`, `mtpy/controllers/Metodo.py`, `web/templates/metodo.html`
- Modify: `mtpy/lib/pages/common.py` (`PAGES`, `ROUTES`), `mtpy/lib/pages/__init__.py`, `tests/test_pages.py`

**Interfaces:**
- Consumes: `common_context`; fuentes del texto: `README.md` del repositorio, `backtest/README.md`,
  `docs/analisis-2026-09/metodo-1-otros-barrera.md` (barrera del 3 %, bloque "otros"),
  `metodo-5-nowcast-horizonte.md` (nowcast, deriva), `metodo-6-house-effects.md`, `metodo-7-composicion.md`
  (sorteo independiente por defecto), `metodo-8-edad-partido.md`, `metodo-9-oscilaciones-autonomicas.md`,
  `metodo-10-error-estandar-cluster.md`, `metodo-11-ambitos-autonomicos.md` (ratings globales),
  `metodo-13-herding.md`, `metodo-3-backtest.md`; `publish.ATTRIBUTION` (fuentes y licencia).
- Produces:
  - `pages.metodo.SECTIONS = (('datos', 'Los datos'), ('promedio', 'El promedio'), ('casas', 'Las
    casas'), ('simulacion', 'La simulación'), ('escanos', 'Los escaños'), ('evaluacion', 'La
    evaluación'), ('publicacion', 'La publicación'), ('limitaciones', 'Limitaciones'))`;
    `metodo_context(state, active='/metodo') -> dict` = `common_context` + `sections` (lista de
    `{'id', 'title'}`), sin `initial`.
  - `Metodo(Page)`; `common.PAGES` gana `{'href': '/metodo', 'label': 'Método'}`; `common.ROUTES` gana
    `('/metodo', 'metodo', 'index', ['GET'])`; `__init__` reexporta `SECTIONS`, `metodo_context`.
  - `metodo.html`: `<nav>` con un enlace `#id` por sección y una `<section id="{{ id }}" class="card">`
    por entrada de `sections`, cada una con `<h2>` y como mucho dos párrafos en español, nivel de lector
    no especialista. Contenido fijado: **datos** (sondeos de Wikipedia con su licencia, resultados del
    Ministerio del Interior, catálogo de partidos y bloques por ámbito, curación propia); **promedio**
    (regresión local ponderada; pesos por tamaño de muestra, por sobre-representación de una casa y por
    su rating; resta de los efectos de casa del ciclo; error estándar con cluster por casa; herding
    informativo); **casas** (rating por ámbito a partir del sesgo frente al resultado en elecciones
    anteriores y una calidad previa; enlace a `/casas`); **simulación** (1.000 elecciones simuladas con el
    error histórico de los sondeos según el horizonte, deriva de la opinión hasta la elección —"Hoy"
    frente a "Elección"—, deriva mayor en los partidos jóvenes, sorteo independiente por defecto,
    oscilaciones autonómicas y provinciales; probabilidades de mayoría); **escaños** (swing proporcional
    por circunscripción, D'Hondt, barrera del 3 %, herencia de partidos; enlace a `/escanos`);
    **evaluación** (backtest fuera de muestra por horizonte, enlace a `/modelo`; elecciones pasadas,
    enlace a `/elecciones`); **publicación** (runs inmutables con su fecha y commit, runs retrospectivos
    marcados, `/api/v1` y CSV, atribución y licencia); **limitaciones** (pocas elecciones para calibrar,
    intervalos estrechos a más de dos semanas, sucesión de partidos, veda). Los enlaces internos llevan
    `{{ query }}`. Ninguna cifra de resultados escrita a mano; sin `<script>` propio.

- [ ] **Step 1: Tests que fallan**

`tests/test_pages.py` (`'metodo'` en la lista de `test_pages_package_layout`):

```python
def test_metodo_renders_the_sections_without_js(api):
    status, headers, html = call(api, '/metodo', query='scope=es')
    assert status == 200 and 'initial-data' not in html and '<script type="module"' not in html
    for section_id, title in pages.SECTIONS:
        assert 'id="{}"'.format(section_id) in html and title in html
    for href in ('/modelo?scope=es&amp;mode=forecast', '/casas?scope=es&amp;mode=forecast', '/elecciones?scope=es&amp;mode=forecast', '/escanos?scope=es&amp;mode=forecast'):
        assert href in html


def test_metodo_is_in_the_menu_of_every_page(api):
    status, _, html = call(api, '/')
    assert '>Método</a>' in html and '>Modelo</a>' in html and '>Elecciones</a>' in html
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "metodo or layout or menu"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces** (escribir el texto a partir de las fuentes citadas)

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages mtpy/controllers/Metodo.py web/templates/metodo.html tests/test_pages.py
git commit -m "Add the method page"
```

---

### Task 9: scripts de comprobación, README, contrato y estado de cierre

**Files:**
- Modify: `deploy/check-api.sh`, `deploy/smoke.sh`, `deploy/README.md` ("La web", "Publicar",
  "Parámetros", "Estados y resumen"), `docs/web/contrato.md` (cabecera y árbol, repaso final),
  `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Estado al cierre de la fase 5";
  enmiendas en la fila 5 de "Fases" y en la nota 2026-10-10 de "Contrato de API"),
  `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md` (nota sobre los paquetes en la
  tabla de componentes)

**Interfaces:**
- Consumes: `check` / `check_db_only` de `deploy/check-api.sh`; `check_static` de `deploy/smoke.sh`; las
  secciones "Análisis de casas" y "Publicar en local" del README como modelo.
- Produces:
  - `deploy/check-api.sh`: `check_published RUTA TIPO` (200 → OK; 404 cuyo cuerpo contiene `not
    published` → `n/a  RUTA sin publicar`; otro → FAIL) y las comprobaciones `check "/elecciones?scope=es"
    200 html`, `check "/elecciones?date=abc" 400 html`, `check_published "/elecciones?scope=es&date=2023-07-23"
    html`, `check "/modelo?scope=es" 200 html`, `check /metodo 200 html`, `check_db_only /api/v1/events/es
    200 json`, `check_published /api/v1/events/es/2023-07-23/pollsters json`, `check
    /api/v1/events/es/2023-13-01/meta 400 none`, `check_published /api/v1/backtest/es/by-horizon json`,
    `check /api/v1/backtest/es/nope 404 none`.
  - `deploy/smoke.sh`: `check_static "/elecciones?scope=es" text/html`, `"/modelo?scope=es"`, `/metodo`.
  - `deploy/README.md`: las tres páginas en "La web"; en "Publicar", los ejemplos
    `python job.py publish '{"what":["backtest"],"scopes":["es"]}'`,
    `'{"what":["event"],"scopes":["es"],"event_date":"2023-07-23"}'`,
    `'{"what":["event"],"scopes":["es"],"event_date":"all"}'`, `'{"what":["backtest"],"scopes":"all"}'`;
    `what` con el orden completo; parámetros `event_date` (para `event`: cadena, lista o `all`) y
    `source`; líneas de resumen de `event` y `backtest`; subsecciones "Elecciones pasadas" y "Backtest"
    (qué se publica, sobrescritura, cuándo relanzar: tras cargar resultados oficiales o tras un nuevo
    `run_backtest`; `publish backtest` no necesita base de datos).
  - Spec principal: "Estado al cierre de la fase 5" con commits, recuento de tests, D1-D10, resoluciones
    tomadas, menores aplazados y "Pendiente de Luis" (lanzar `publish event` para `es` `all` y los
    autonómicos con resultados, `publish backtest` `all`, recorrer `/elecciones`, `/modelo` y `/metodo`,
    revisar el texto de `/metodo`, decidir si el histograma de errores del notebook entra, la veda en la
    fase 6); corregir en "Contrato de API" la nota "events y backtest siguen en la fase 5" y marcar la
    fila 5 de "Fases" como hecha.

- [ ] **Step 1: Scripts y documentación**

- [ ] **Step 2: Comprobación local sin base y sin S3**

```bash
S3_BUCKET= python job.py publish '{"what":["backtest"],"scopes":["es"]}'     # escribe files/site/v1/backtest/es/ desde backtest/results
bash deploy/nginx-check.sh
S3_BUCKET= DB_HOST= uvicorn api:api --port 8000 &    # paquete local; las rutas de base quedan n/a
bash deploy/check-api.sh http://127.0.0.1:8000
kill %1
```

Expected: `check-api.sh` sin FAIL (`/modelo`, `/metodo`, `/elecciones` 200; backtest 200; eventos `n/a`
sin publicar; rutas de base `n/a sin base de datos`). `python -m pytest -m "not integration" -q` en verde.

- [ ] **Step 3: Commit**

```bash
git add deploy/check-api.sh deploy/smoke.sh deploy/README.md docs/web/contrato.md docs/superpowers/specs
git commit -m "Document phase 5: events, backtest, the three pages and the closing state"
```
