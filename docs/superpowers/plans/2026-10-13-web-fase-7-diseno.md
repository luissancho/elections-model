# Web, fase 7: modelo de eventos, diseño, explicaciones y método — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir la web en un sitio centrado en eventos (Inicio que elige el evento; Promedios y Escaños por
ámbito en `/{scope}` y `/{scope}/escanos`; Histórico de errores, Rating de encuestadoras y Método como páginas
globales con filtro de ámbito), con un diseño profesional de tarjetas con título, descripción y "Cómo leerlo",
un glosario único de medidas de error, un histórico con filtros, histograma y diana, y un método en cuatro
subpáginas (tres textos y el backtest), antes de la semana del 2026-11-17.

**Architecture:** Sin cambios de arquitectura: controlador `Page` + plantilla Jinja2 + isla JSON `#initial-data` +
JS que solo dibuja con ECharts; nginx sirve solo `/dist/` y no se toca; el paquete publicado, `publish`, el
contrato y `/api/v1` no cambian. Lo que cambia es el enrutado de las páginas (rutas con parámetro `{scope}`
registradas después de las fijas, redirecciones 301 desde las URL antiguas), el estado común (`mode` por
defecto Hoy, Elección con la web congelada; el ámbito viene de la ruta en las páginas de evento y de `?scope=`
en las globales), las plantillas (cabecera sin selectores, barra de evento, filtro de ámbito, macros de tarjeta,
glosario), el CSS (tokens, fuentes propias en `/dist/vendor/fonts/`, fondo pastel, tarjetas) y el JS (módulos
puros nuevos `blocks.js` y `errors.js` con tests de node, histograma y diana generalizados).

**Tech Stack:** Python 3.11, Jinja2 3.1.6 (StrictUndefined, autoescape), pytest 9 (`python -m pytest -m "not
integration" -q`), ECharts 5.6 vendorizado, JavaScript ES modules sin build (node 25 solo para `node --check` y
los tests de paridad, que se saltan sin node), CSS con tokens en `:root`, woff2 (Source Serif 4 y Source Sans 3,
SIL OFL), MathML para las fórmulas en bloque.

**Spec:** `docs/superpowers/specs/2026-10-13-web-fase-7-diseno-design.md` (autoridad: "Decisiones de Luis"
1-14, "Diseño" completo, "Decisiones de los puntos abiertos" 1-6, "Revisión del constructor" 1-5). Spec
general: `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Estado al cierre de la fase 6").
Contrato: `docs/web/contrato.md`. Runbooks: `deploy/README.md`. Fuentes del método: `docs/analisis-2026-09/`.

## Global Constraints

- Patrón intacto: `Page` + Jinja2 + `#initial-data` + JS que solo dibuja; sin `fetch` ni literal `/api/v1` en el
  JS (`tests/test_web_routes.py`); `StrictUndefined`; filtros de `pages/filters.py`; textos en español.
- nginx, Docker, ECS, `publish`, el paquete, `docs/web/contrato.md` (salvo el párrafo de rutas de páginas) y
  `/api/v1` no cambian. Un dato que el paquete o el catálogo no tengan: la tarea se detiene y se consulta.
- CSP actual (`default-src 'self'`): fuentes solo desde `/dist/vendor/fonts/`; sin Google Fonts, CDN ni KaTeX.
- Las páginas de evento exigen un pronóstico publicado del ámbito y son las únicas que entienden `mode` y
  `run`; las globales llevan `?scope=` (`es` por defecto) y exigen un pronóstico publicado del ámbito filtrado.
- Modo por defecto `nowcast` ("Hoy"); `forecast` por defecto solo con `manifest.freeze.active`.
- Presupuesto orientativo de `/es` ≈ 500 KB de HTML sin comprimir; recortes autorizados, en este orden: series
  con un decimal en la isla, `history.runs` y `summary` a los campos dibujados, filas de la tabla de sondeos.
  Nunca partidos ni bandas de `series`. Fuentes ≤ 2 ficheros woff2 y ≤ 150 KB en total.
- Bloques con sus nombres del catálogo (Izquierda, Derecha, Regionalista, Separatista).
- Glosario único: "Error medio" (`error_avg`, sin signo), "Error de bloques" (`|error_blocks|`, calculado en
  las páginas como `error_blocks_abs`), "Sesgo del margen" (`error_blocks` con signo; positivo, más margen al
  primer bloque). "Sesgo de bloques" y "Desviación hacia {bloque}" desaparecen del HTML público (quedan en CSV y API).
- Toda afirmación de los textos del método se contrasta con el código o con `docs/analisis-2026-09/` y el
  informe de la tarea lista las afirmaciones comprobadas; ninguna cifra que el paquete o el catálogo puedan dar
  va escrita a mano.
- Una subpágina del método sin texto no se enlaza (ni en el índice ni en las tarjetas de cierre).
- Suite unitaria en verde al cerrar cada tarea (568 al empezar), `node --check` de cada JS tocado, `bash -n` de
  cada script tocado. Trabajo en `dev` sin worktree, como en las fases 1-6; Luis integra.

## Review Focus

1. `/ES`, `/es/` o `/es//escanos`: el alias por defecto del router solo admite minúsculas sin barra final, así
   que deben dar la página 404 de Python, no un 500 ni una redirección. Test en la tarea 1.
2. Un ámbito del manifest cuyo último run no tenga `headline` (publicación a medias): el Inicio debe pintar su
   tarjeta con guiones, no caer en 500 ni esconder los demás ámbitos. Test en la tarea 2.
3. Un hash `#vista=bloques` en un ámbito cuyo `bmaps` no tiene `blocks` (autonómico con solo `vs`): el control de
   vista se oculta y el hash se ignora; el gráfico se dibuja por partidos. Test de node en la tarea 3 y test de
   página con el ámbito sintético `es-md`.
4. Un hash `#eleccion=2011-11-20` que nombra una elección del análisis pero no publicada, o una casa que no
   existe: el filtro vuelve a "Todas" sin error. Test de node en la tarea 6.
5. La barra de evento el día de la elección y después (`days_left` 0 o negativo): "Jornada electoral" y
   "Celebradas hace {n} días", nunca "faltan -3 días". Test en la tarea 1.

---

## Estado de partida

Rama `dev` en `f89dfc5` (spec de la fase 7), árbol limpio, 568 tests unitarios en verde. Hechos que las tareas
usan:

- `mtpy.api(routes=pages.ROUTES + webapi.ROUTES, not_found=pages.NOT_FOUND)` registra `/` → controlador
  `index` siempre primero y después las rutas en orden; `Router.handle` prueba las rutas en orden de registro y
  devuelve la primera que casa; `{name}` usa el alias `''` = `[0-9a-z_\-]+` y `{name:str}` = `[a-z]+`; los
  parámetros llegan a la acción como kwargs (`index_action(self, scope)`).
- `Page.state()` = `pages.parse_state(self.request.query)`; `Page.render(template, **context)` pone
  `text/html`, ETag y `public, max-age=60`; `Page.error_page` y `error_result` hacen el HTML de error con
  `MESSAGES`. El núcleo no tiene ayuda de redirección: `Response.set_status_code(301)` + `set_header('location', …)`.
- `pages.common`: `PAGES`, `ROUTES`, `NOT_FOUND`, `MESSAGES`, `parse_state(query)` (`mode` por defecto
  `'forecast'`), `published_scopes()`, `resolve_scope(scope)`, `part(scope, run, name, mode)`,
  `query_string(state)` (`?scope=&mode=[&run=]`), `api_url(...)`, `Catalog`, `catalog(meta)`,
  `block_color(block, meta, parties)`, `common_context(state, active)` (`site_title`, `pages`, `state`
  con `pinned`, `query`, `scopes`, `manifest`, `freeze`, `frozen`, `meta`, `catalog`, `mode_label`,
  `title_suffix`), `bundle_part(read, *args)`.
- `pages.index.index_context` (headline del modo, `majority_rows`, `initial` = state, meta, headline, vote,
  summary, runs); `pages.promedio.promedio_context` (`chart_polls`, `table_rows`, `TABLE_ROWS = 40`,
  `initial` = state, meta, series, polls, projection, vote); `pages.escanos.escanos_context(state, region,
  active)`; `pages.casas.casas_context`/`casa_context` (`RANKING_KEYS`, `PATH_KEYS`, `profile_rows`);
  `pages.elecciones` (`published_events`, `with_abs`, `elecciones_context`, `eleccion_context`,
  `horizon_tables`, `resolve_event`); `pages.modelo.modelo_context` (`HORIZON_COLUMNS` incluye `mae_seats` y
  `mae_seats_h`); `pages.metodo.metodo_context` (`SECTIONS` de 8 anclas).
- `webapi`: `check_scope` (400 `invalid scope`), `check_mode`, `check_run`, `check_date`, `check_pollster`,
  `scope_codes()`, `catalogue()` (DataFrame de `data/es-scopes.csv` indexado por `scode`, columnas `name`,
  `parent`, `threshold`, `threshold_scope`, `rating_weight`, `seats`), `site()` con `manifest_data()`,
  `latest_run(scope)`, `history(scope)`, `run_file`, `analysis(scope, part)`, `events(scope)` (lista de fechas),
  `event(scope, date, part)`, `backtest(scope, part)`, `scopes()` (catálogo cruzado con el manifest).
- Plantillas: `base.html` (banner, `_header.html`, `#status`, `_footer.html`, isla `initial`), `_header.html`
  (h1, nav de `pages`, formulario `#controls` con `scope`, radios `mode`, `run` oculto y botón `Ver.nojs-only`),
  `index.html`, `promedio.html` (`#series-window` 6m/all), `escanos.html`, `casas.html`, `casa.html`,
  `elecciones.html`, `eleccion.html`, `modelo.html`, `metodo.html`, `error.html`.
- JS: `pages/common.js` (`readInitial`, `wireControls`, `wireForm(id)`, `mapNames`, `showError`, `tryDraw`,
  `mountWhenReady`), `range.js` (`WINDOW_DAYS = {'6m': 180}`, `DEFAULT_WINDOW = '6m'`, `lastWindow`,
  `seriesWindow`), `charts/series.js` (`renderSeries(el, {series, polls, projection, parties, selected,
  colors, fullnames, asOf, when, window, anchor})`, `setWindow(chart, choice, {series, projection, anchor})`),
  `charts/histogram.js` (`renderHistogram(el, values, {color, majority, median, lo, hi, width, formatter})`,
  etiquetas "escaños"), `charts/bullseye.js` (`renderBullseye(el, rows, {blocks})`, puntos grises con etiqueta
  `row.event`), `charts/line.js` (`renderLine`, `renderLines(el, series, {formatter, yMin, yMax,
  references})`), `charts/bars.js`, `charts/evolution.js`, `charts/hemicycle.js`, `charts/stacked.js`,
  `charts/fan.js`, `charts/heatmap.js`, `seats.js` (`histogram(values, width)`, `binWidth`, `summarize`,
  `parseCoalition`, `coalitionHash`), `catalog.js` (`Catalog`, `BLOCK_ORDER`), `format.js`.
- Tests: `tests/test_pages.py` (fixture `api` con `write_bundle` del paquete de fixtures; `call(api, path,
  query=...)` → `(status, headers, body)`; `html(body)`, `initial(body)`; `fake_houses`, `houses`,
  `events_site`; `test_pages_package_layout` enumera los módulos de `pages`; `test_regional_scope_renders_every_page`
  publica `es-md`), `tests/test_webapi_unit.py` (`RUN = '20261008-120000'`, `fixture_data`, `write_bundle`,
  `write_analysis`, `write_event`, `write_backtest`, `use_database`), `tests/test_js_range.py` (patrón de test
  de node: `node --input-type=module -e` importando el módulo por URI), `tests/test_web_routes.py` (enlaces de
  plantilla resuelven contra `pages.ROUTES`; sin `/api/v1` en el JS; recursos referenciados existen).
- Fixtures del paquete en `tests/fixtures/bundle/`: `manifest.json` (`es`, `event_date` 2026-11-29),
  `headline.json` (`nowcast` y `forecast` con `parties` y `p_majority`), `meta.json`, `series.json`,
  `polls.json`, `projection.json` (`groups.parties`, `groups.vs`, `groups.blocks`), `poll-errors.json` (filas
  con `event_date`, `event`, `pollster`, `days`, `error_avg`, `error_blocks`, `bias_blocks` y columnas
  `Derecha`/`Izquierda`), `event-*.json`, `backtest-*.json` (`by-horizon` con `mae_seats` y `mae_seats_h`).
- Datos reales (base, solo lectura, 2026-10-13): `es` tiene 16 elecciones con sondeos desde 1977 y 11 `featured`
  (1993-2023); `es-cb` tiene 2019-05-26 con sondeos pero no `featured`.
- `deploy/check-api.sh` comprueba `/`, `/promedio?scope=es&mode=nowcast`, `/escanos?scope=es&mode=forecast`,
  `/escanos?region=abc` 400, `/casas?scope=es`, `/casas?id=abc` 400, `/elecciones?scope=es`,
  `/elecciones?date=abc` 400, `/elecciones?scope=es&date=2023-07-23`, `/modelo?scope=es`, `/metodo`, `/nope`
  404; `deploy/smoke.sh` comprueba `/promedio`, `/escanos`, `/elecciones?scope=es`, `/modelo?scope=es`,
  `/metodo` y `/casas?scope=es`.

## Decisiones de ejecución (supuestos del plan; Luis puede corregirlos)

- **E1.** El Inicio es el controlador `Index` y la plantilla `index.html` (reescrita), porque `/` → `index` lo
  registra siempre `mtpy.api`. Promedios es un controlador nuevo `Promedios` con `promedios.html`;
  `pages/index.py` y `pages/promedio.py` se funden en `pages/promedios.py` (`majority_rows`, `chart_polls`,
  `table_rows`, `TABLE_ROWS` se mudan allí); `pages/inicio.py` es nuevo; `pages/modelo.py` pasa a
  `pages/backtest.py` con `backtest_context`. Los controladores `Promedio.py` y `Modelo.py` desaparecen; un
  controlador `Redirect.py` atiende `/promedio`, `/escanos` y `/modelo`.
- **E2.** `/{scope}` y `/{scope}/escanos` usan el alias por defecto y se registran al final de `pages.ROUTES`,
  después de las rutas fijas; el controlador valida con `pages.resolve_event_scope(raw)`: un código que no está
  en `webapi.scope_codes()` da 404 `unknown scope` ("Ámbito desconocido."); uno del catálogo sin pronóstico da el
  404 `scope not published` de hoy. `/metodo/{section:str}` valida contra `METODO_SECTIONS`; desconocida, 404
  `section not found` ("Página del método no encontrada.").
- **E3.** `parse_state(query, scope=None)` devuelve `mode: None` cuando no viene; `common_context` lo resuelve
  con `default_mode(frozen)` y el `state` que llega a las plantillas lleva siempre el modo resuelto.
- **E4.** La redirección es `Page.redirect(location)`: 301, `location`, `cache-control: no-store`, cuerpo vacío.
  La query se reconstruye con `urlencode` quitando solo `scope` (y `mode`/`run` donde la página destino no los
  entiende).
- **E5.** Fuentes: Source Serif 4 (semibold, títulos) y Source Sans 3 (regular, texto) en woff2 estático de las
  releases de `github.com/adobe-fonts`, con su `LICENSE` (OFL) en `/dist/vendor/fonts/`; la negrita del texto se
  sintetiza. Si la descarga no es posible en el entorno de la tarea, se dejan los tokens apuntando a la pila
  del sistema y el informe lo dice.
- **E6.** `renderHistogram` gana `unit` ("escaños" por defecto), `label` (función de etiqueta por bin) y
  `markLabel` (texto de la línea sólida; por defecto la mediana formateada); los histogramas de escaños no
  cambian. `renderBullseye` gana `groupOf`, `colors` (por grupo), `legend` y `labels` (etiquetas de punto, true
  por defecto); las dianas de hoy no cambian.
- **E7.** Los filtros del histórico y los controles del promedio viven en el hash y se escriben con
  `history.replaceState` solo al cambiar, como `#coalition=`; la lógica pura va en `errors.js` y `blocks.js`.
- **E8.** La vista Bloques dibuja la proyección publicada (`projection.groups.blocks`) y calcula solo la
  banda histórica (spec, "Revisión del constructor" 3).
- **E9.** Los textos del método son HTML en sus plantillas (no Markdown), con `<section>` por h2, un
  `<nav>` de anclas, fórmulas inline con `<var>`/`<sub>`/`<sup>` y en bloque con `<math>`; cada texto
  declara su entrada en `METODO_SECTIONS` solo cuando existe (tareas 9-11), así que antes de escribirse no
  se enlaza.
- **E10.** `/metodo` conserva el texto actual de las secciones `datos`, `publicacion` y `limitaciones` (veda
  reescrita) y añade la introducción y el índice; el resto del texto actual (`promedio`, `casas`,
  `simulacion`, `escanos`, `evaluacion`) se retira cuando su texto nuevo entra (tareas 9-11) y, mientras,
  queda bajo "Cómo funciona el modelo" en la portada.
- **E11.** Orden de ejecución = orden de las tareas (prioridad de Luis: historia, explicaciones, método).
  Las tareas 9, 10 y 11 pueden quedar para después si el plazo aprieta; la 12 cierra lo que haya.
- **E12.** El nombre de evento es "Elecciones generales" para `es` y "Elecciones autonómicas · {name}" para
  los demás, con `name` del catálogo; los días que faltan se cuentan con la fecha de hoy en Europe/Madrid
  (`filters._madrid_tz`).

---

### Task 1: Rutas de evento, estado común, redirecciones y cabecera

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`PAGES`, `EVENT_TABS`, `ROUTES`, `MESSAGES`, `parse_state`,
  `resolve_event_scope`, `default_mode`, `event_query`, `scope_query`, `days_left`, `event_title`,
  `event_info`, `title_of`, `common_context`)
- Modify: `mtpy/lib/pages/__init__.py` (reexportar los nombres nuevos)
- Modify: `mtpy/controllers/Page.py` (`redirect`, `state` con `scope` de ruta)
- Create: `mtpy/controllers/Promedios.py`, `mtpy/controllers/Redirect.py`
- Modify: `mtpy/controllers/Index.py`, `Escanos.py`, `Casas.py`, `Elecciones.py`, `Metodo.py`
- Delete: `mtpy/controllers/Promedio.py`, `mtpy/controllers/Modelo.py`
- Modify: `mtpy/lib/pages/escanos.py` (`active`), `mtpy/lib/pages/metodo.py` (ruta `backtest`, ver abajo)
- Rename: `mtpy/lib/pages/modelo.py` → `mtpy/lib/pages/backtest.py` (`backtest_context`, `active='/metodo'`);
  `web/templates/modelo.html` → `web/templates/metodo_backtest.html`; `web/templates/index.html` →
  `web/templates/promedios.html` (el Inicio nuevo llega en la tarea 2 como `index.html`)
- Create: `web/templates/_event_bar.html`, `web/templates/_scope_filter.html`
- Modify: `web/templates/base.html`, `_header.html`, `_footer.html`, `promedios.html`, `escanos.html`,
  `casas.html`, `casa.html`, `elecciones.html`, `eleccion.html`, `metodo.html`, `metodo_backtest.html`
- Modify: `web/dist/js/pages/common.js` (`wireControls`), `web/dist/js/pages/modelo.js` → renombrar a
  `web/dist/js/pages/metodo.js` (el actual `metodo.js` de 4 líneas se sustituye), `web/dist/css/site.css`
  (solo `form.js .nojs-only { display: none }` y una regla mínima de la barra)
- Modify: `deploy/check-api.sh`, `deploy/smoke.sh`
- Test: `tests/test_pages.py`, `tests/test_web_routes.py`

**Interfaces:**
- Produces (`mtpy/lib/pages/common.py`):
  - `PAGES = ({'href': '/', 'label': 'Inicio'}, {'href': '/elecciones', 'label': 'Histórico de errores'}, {'href': '/casas', 'label': 'Rating de encuestadoras'}, {'href': '/metodo', 'label': 'Método'})`
  - `EVENT_TABS = ({'suffix': '', 'label': 'Promedios'}, {'suffix': '/escanos', 'label': 'Escaños'})`
  - `ROUTES` en este orden: `('/promedio', 'redirect', 'promedio', ['GET'])`, `('/escanos', 'redirect', 'escanos', ['GET'])`, `('/modelo', 'redirect', 'modelo', ['GET'])`, `('/casas', 'casas', 'index', ['GET'])`, `('/elecciones', 'elecciones', 'index', ['GET'])`, `('/metodo', 'metodo', 'index', ['GET'])`, `('/metodo/{section:str}', 'metodo', 'section', ['GET'])`, `('/{scope}', 'promedios', 'index', ['GET'])`, `('/{scope}/escanos', 'escanos', 'index', ['GET'])`
  - `MESSAGES` + `'unknown scope': 'Ámbito desconocido.'`, `'section not found': 'Página del método no encontrada.'`
  - `parse_state(query: dict, scope: Optional[str] = None) -> dict` → `{'scope', 'mode' (None si no viene), 'run'}`; `scope` dado (ruta) manda sobre la query.
  - `resolve_event_scope(raw: str) -> str` (E2).
  - `default_mode(frozen: bool) -> str` → `'forecast'` si `frozen`, si no `'nowcast'`.
  - `event_query(state: dict) -> str` → `'?mode=nowcast'` (+ `&run=…` si `pinned`); `scope_query(scope: str) -> str` → `'?scope=es'`.
  - `days_left(event_date: str, today: Optional[date] = None) -> Optional[int]` (hoy en Europe/Madrid).
  - `event_title(scope: str, scopes: list) -> str` (E12; `scopes` = filas de `published_scopes()`).
  - `event_info(scope: str, meta: dict, active_suffix: str, state: dict, today=None, scopes=None) -> dict` → `{'title', 'date', 'days_left', 'days_text', 'href', 'form_action', 'tabs': [{'href', 'label', 'active'}]}`; `scopes` por defecto `published_scopes()`; `days_text`: `'Faltan {n} días'` (n > 1), `'Falta 1 día'`, `'Jornada electoral'` (0), `'Celebradas hace {n} días'` (n < 0); `tabs[i].href = '/' + scope + suffix + event_query(state)`; `form_action = '/' + scope + active_suffix` (destino del formulario del modo).
  - En `mtpy/lib/pages/metodo.py` (ya en esta tarea, con la forma definitiva): `METODO_SECTIONS: dict` ordenado `{'promedio': {'title': '¿Cómo promediamos las encuestas?', 'template': None}, 'escanos': {'title': '¿Cómo simulamos los escaños?', 'template': None}, 'rating': {'title': '¿Cómo calculamos el rating de las encuestadoras?', 'template': None}, 'backtest': {'title': '¿Cuánto acierta el modelo?', 'template': 'metodo_backtest.html'}}`; `subpages(active: Optional[str]) -> list` → `[{'id', 'title', 'href': '/metodo/'+id, 'active', 'available': template is not None}]`; `method_href(section: str) -> str` → `'/metodo/'+section` si `available`, si no `'/metodo'` (lo usan las tarjetas de cierre y la barra de evento). Las tareas 9-11 rellenan `template`.
  - `title_of(mode: str, meta: dict, frozen: bool, pinned: bool, latest: str) -> str`: `nowcast` → `'Estimación a {as_of}'`; `forecast` y `frozen` y `meta['run_id'] == latest` → `'Pronóstico definitivo para el {event_date}'`; `forecast` y `pinned` y `meta['run_id'] != latest` → `'Pronóstico publicado el {run_at}'` (fecha); si no `'Pronóstico para el {event_date}'`.
  - `common_context(state: dict, active: Optional[str], event: bool = False, active_suffix: str = '') -> dict`: lo de hoy más `is_event` (bool), `event` (`event_info` o `None`), `query` (= `event_query` en evento, `scope_query` en globales), `scope_filter` (`published_scopes()`; las globales lo pintan), `mode` resuelto en `state['mode']`, `mode_text` (frase de la opción activa, ver plantilla), `title_suffix` por `title_of`. `pages[i].active` es `False` para todas cuando `active is None`.
- Produces (`mtpy/controllers/Page.py`): `redirect(self, location: str) -> str` (E4) y `state(self, scope: Optional[str] = None) -> dict`.
- Produces (`mtpy/controllers/Redirect.py`): `Redirect(Page)` con `promedio_action`, `escanos_action`, `modelo_action`; destino con `resolve_scope(query.get('scope'))` → `/{scope}`, `/{scope}/escanos` (conservando `mode`, `run`, `region`), `/metodo/backtest` (conservando `scope`).
- Consumes: `webapi.scope_codes`, `check_scope`, `check_mode`, `check_run`, `site().manifest_data()`, `site().latest_run`, `filters._madrid_tz`, `fmt_date`.

- [ ] **Step 1: Escribir los tests de estado y rutas que fallan**

En `tests/test_pages.py`, sustituir `test_home_is_rendered_html_with_cache_headers`,
`test_nowcast_and_pinned_run_are_reflected`, `test_promedio_*`, `test_escanos_*` (las llamadas pasan de
`/promedio?scope=es` a `/es` y de `/escanos?scope=es` a `/es/escanos`; `test_regional_scope_renders_every_page`
recorre `('/es-md', '/es-md/escanos')` y comprueba `Elecciones autonómicas · Madrid` en vez de `<option
value="es-md" selected>`) y añadir:

```python
def test_event_pages_live_under_the_scope_and_default_to_nowcast(api):
    status, headers, body = call(api, '/es')
    page = html(body)
    assert status == 200 and headers['cache-control'] == 'public, max-age=60'
    assert '<title>Pronóstico electoral · Promedios</title>' in page
    assert 'Estimación a' in page and 'id="event-bar"' in page and 'Elecciones generales' in page
    assert 'name="mode" value="nowcast" checked' in page.replace('\n', ' ')
    assert 'href="/es/escanos?mode=nowcast"' in page and 'id="controls"' not in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    status, _, body = call(api, '/es/escanos', query='mode=forecast&run=' + RUN)
    assert status == 200 and 'Pronóstico para el' in html(body) and 'name="run" value="' + RUN + '"' in html(body)


def test_frozen_site_defaults_to_the_forecast_and_an_older_pinned_run_is_not_definitive(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    write_bundle(fresh_app.fs, run_id='20261001-120000', manifest=False)
    publish.update_manifest(writer, {'es': publish.latest_entry(fixture_data('headline') | {'run_id': RUN})},
                            freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    assert 'Pronóstico definitivo para el' in html(call(api, '/es')[2])
    page = html(call(api, '/es', query='run=20261001-120000&mode=forecast')[2])
    assert 'Pronóstico publicado el' in page and 'definitivo' not in page


@pytest.mark.parametrize('path, query, location', [
    ('/', 'scope=es&mode=nowcast', '/es?mode=nowcast'),
    ('/promedio', 'scope=es&mode=forecast&run=' + RUN, '/es?mode=forecast&run=' + RUN),
    ('/promedio', '', '/es'),
    ('/escanos', 'scope=es&region=28', '/es/escanos?region=28'),
    ('/modelo', 'scope=es', '/metodo/backtest?scope=es'),
    ('/modelo', '', '/metodo/backtest'),
])
def test_old_urls_redirect_permanently(api, path, query, location):
    status, headers, body = call(api, path, query=query)
    assert status == 301 and headers['location'] == location and headers['cache-control'] == 'no-store'
    assert body == b''


@pytest.mark.parametrize('path', ['/es-zz', '/ES', '/es/', '/es//escanos', '/healthz', '/metodo/nope'])
def test_unknown_scopes_and_sections_are_html_404s(api, path):
    status, headers, body = call(api, path)
    assert status == 404 and headers['content-type'].startswith('text/html')


def test_unpublished_catalogue_scope_is_a_404(api):
    status, _, body = call(api, '/es-md')
    assert status == 404 and 'no tiene pronóstico publicado' in html(body)


def test_global_pages_carry_the_scope_filter_and_no_event_bar(api):
    for path in ('/elecciones', '/metodo/backtest'):
        page = html(call(api, path)[2])
        assert 'id="scope-filter"' in page and 'id="event-bar"' not in page and 'name="mode"' not in page, path
    landing = html(call(api, '/metodo')[2])
    assert 'id="scope-filter"' not in landing and 'id="event-bar"' not in landing
    assert '<option value="es" selected>' in html(call(api, '/elecciones', query='scope=es')[2])


def test_menu_has_the_four_global_entries(api):
    page = html(call(api, '/es')[2])
    for label in ('Inicio', 'Histórico de errores', 'Rating de encuestadoras', 'Método'):
        assert '>' + label + '</a>' in page
    assert 'Portada' not in page and 'aria-current="page"' not in page


def test_event_bar_days_text():
    from datetime import date
    assert pages.days_left('2026-11-29', today=date(2026, 10, 13)) == 47
    meta = {'event_date': '2026-11-29', 'as_of': '2026-10-13', 'run_id': RUN, 'run_at': '2026-10-13T12:00:00Z'}
    state = {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    scopes = [{'code': 'es', 'name': 'España'}, {'code': 'es-md', 'name': 'Madrid'}]
    info = pages.event_info('es', meta, '', state, today=date(2026, 10, 13), scopes=scopes)
    assert info['title'] == 'Elecciones generales' and info['days_text'] == 'Faltan 47 días'
    assert info['tabs'] == [{'href': '/es?mode=nowcast', 'label': 'Promedios', 'active': True},
                            {'href': '/es/escanos?mode=nowcast', 'label': 'Escaños', 'active': False}]
    assert pages.event_info('es', meta, '', state, today=date(2026, 11, 29), scopes=scopes)['days_text'] == 'Jornada electoral'
    assert pages.event_info('es', meta, '', state, today=date(2026, 12, 2), scopes=scopes)['days_text'] == 'Celebradas hace 3 días'
    assert pages.event_title('es-md', scopes) == 'Elecciones autonómicas · Madrid'


def test_title_of_rules():
    meta = {'as_of': '2026-10-13', 'event_date': '2026-11-29', 'run_id': 'B', 'run_at': '2026-10-13T12:00:00Z'}
    assert pages.title_of('nowcast', meta, True, True, 'B') == 'Estimación a 13 de octubre de 2026'
    assert pages.title_of('forecast', meta, True, False, 'B') == 'Pronóstico definitivo para el 29 de noviembre de 2026'
    assert pages.title_of('forecast', meta, True, True, 'C') == 'Pronóstico publicado el 13 de octubre de 2026'
    assert pages.title_of('forecast', meta, False, True, 'C') == 'Pronóstico publicado el 13 de octubre de 2026'
    assert pages.title_of('forecast', meta, False, False, 'B') == 'Pronóstico para el 29 de noviembre de 2026'
```

(`fmt_date` ya pinta "13 de octubre de 2026"; ajustar el literal si el filtro usa otro formato.) En
`test_pages_package_layout`, la lista pasa a `['backtest', 'casas', 'common', 'elecciones', 'escanos', 'filters',
'index', 'metodo', 'promedio']` (la tarea 2 añade `inicio` y la 3 cambia `index`/`promedio` por `promedios`).

- [ ] **Step 2: Ejecutar los tests y ver que fallan**

Run: `python -m pytest tests/test_pages.py -q -x`
Expected: FAIL (`/es` → 404, `pages.days_left` no existe).

- [ ] **Step 3: Implementar el estado y las rutas en `pages/common.py`**

Firmas de "Interfaces". `common_context` resuelve `scope` con `state['scope']` (ya validado por
`resolve_event_scope` en evento, o `resolve_scope` en globales), `mode = state['mode'] or default_mode(frozen)`,
`latest = site().latest_run(scope)`, `title_suffix = title_of(mode, meta, frozen, pinned, latest)`;
`event = event_info(...)` solo con `event=True`; `query` según la familia. `mode_text`:
Hoy → `'Si las elecciones fueran hoy: el promedio de sondeos leído el {as_of}, con los sondeos publicados hasta el {date_last}.'`;
Elección → `'Lo que puede pasar el {event_date}: añade la deriva de la opinión en los {n} días que faltan, así que los intervalos son más anchos.'`
(`n` = `days_left` o, si no hay fecha, se omite la coletilla). Reexportar en `pages/__init__.py`.

- [ ] **Step 4: Controladores**

`Page.redirect(location)` (E4) y `Page.state(scope=None)` → `pages.parse_state(self.request.query, scope)`. El `<details>` de la barra
de evento y las tarjetas de cierre de las tareas 3, 5, 6 y 7 enlazan con `pages.method_href(sección)`.
`Promedios(Page)`: `active = None`, `index_action(self, scope)` → `scope = pages.resolve_event_scope(scope)`;
`pages.promedios_context` no existe todavía: en esta tarea renderiza `promedios.html` con
`pages.index_context(self.state(scope), active=None, event=True)` (añadir a `index_context`, `promedio_context`,
`escanos_context` los parámetros `event: bool = False, active_suffix: str = ''` que pasan a `common_context`).
`Escanos.index_action(self, scope)` igual con `active_suffix='/escanos'`. `Index.index_action`: con `scope` en
la query → `self.redirect('/' + resolve_scope(check_scope(scope)) + resto)`; sin él, por ahora
`index_context(self.state(), active='/')` sobre `promedios.html` (la tarea 2 lo sustituye). `Redirect` según
"Interfaces". `Metodo.section_action(self, section)`: `section` ∉ `pages.METODO_SECTIONS` → `HttpError(404,
'section not found')`, igual que una sección cuyo `template` es `None` (las tres de texto hasta las tareas
9-11); `'backtest'` → `render('metodo_backtest.html', **pages.backtest_context(self.state(), active='/metodo'))`.
`Casas` y `Elecciones` no cambian salvo `active`.

- [ ] **Step 5: Plantillas**

`_header.html`: solo h1 + nav (sin formulario). `_event_bar.html` (incluido por `base.html` cuando `is_event`):

```html
<div id="event-bar">
  <div class="event-title"><h2>{{ event.title }}</h2><p class="muted">{{ event.date|date }} · {{ event.days_text }}</p></div>
  <nav class="event-tabs" aria-label="Secciones del evento">{% for tab in event.tabs %}<a href="{{ tab.href }}"{% if tab.active %} aria-current="page"{% endif %}>{{ tab.label }}</a>{% endfor %}</nav>
  <form id="event-controls" class="controls" method="get" action="{{ event.tabs|selectattr('active')|map(attribute='href')|first|replace(query, '') }}" autocomplete="off">
    <fieldset id="mode"><legend>Ver</legend>
      <label><input type="radio" name="mode" value="nowcast"{% if state.mode == 'nowcast' %} checked{% endif %}> Hoy</label>
      <label><input type="radio" name="mode" value="forecast"{% if state.mode == 'forecast' %} checked{% endif %}> Elección</label>
    </fieldset>
    {% if state.pinned %}<input type="hidden" name="run" value="{{ state.run }}"><span class="chip">Run del {{ meta.run_at|date }}</span>{% endif %}
    <button type="submit" class="nojs-only">Ver</button>
  </form>
  <p class="mode-text">{{ mode_text }}</p>
  <details class="how"><summary>¿Qué cambia entre Hoy y Elección?</summary><p>…</p></details>
</div>
```

(El `action` puede calcularse en el contexto como `event.form_action`; es más simple: añadirlo a
`event_info`.) El párrafo del `<details>`: qué comparten (el mismo promedio), qué añade Elección (la deriva
hasta la votación, medida en ciclos pasados), cuál mirar para qué, y un enlace a `/metodo/escanos` solo si la
sección existe (`'escanos' in metodo_sections`; hasta la tarea 10, a `/metodo`). `_scope_filter.html` (incluido
por `elecciones.html`, `casas.html`, `casa.html`, `metodo_backtest.html`, debajo de la cabecera):

```html
<form id="scope-filter" class="controls" method="get" action="{{ filter_action }}" autocomplete="off">
  <label>Ámbito <select name="scope">{% for s in scope_filter %}<option value="{{ s.code }}"{% if s.code == state.scope %} selected{% endif %}>{{ s.name }}</option>{% endfor %}</select></label>
  <button type="submit" class="nojs-only">Ver</button>
</form>
```

`filter_action` = la ruta de la página (`active` o, en el backtest, `/metodo/backtest`). Todos los enlaces
internos de las plantillas pasan de `{{ query }}` a: dentro del evento `/{{ state.scope }}{{ query }}` y
`/{{ state.scope }}/escanos{{ query }}`; hacia las globales `/elecciones{{ scope_query }}` (añadir
`scope_query` al contexto común) y `/casas{{ scope_query }}&amp;id=…`; `_footer.html` añade `<a
href="/metodo">Método</a>`. `base.html`: `<title>` de Promedios "Promedios", de Escaños "Escaños" (los
`{% block title %}`), del backtest "Cuánto acierta el modelo".

- [ ] **Step 6: JS y scripts**

`common.js`: `wireControls()` pasa a `wireForm('event-controls'); wireForm('scope-filter');` (y `pageshow`
como hoy); `site.css`: `.controls.js .nojs-only` → `form.js .nojs-only { display: none; }`. Renombrar
`pages/modelo.js` a `pages/metodo.js` (mismo contenido; el `metodo.js` actual desaparece) y actualizar
`metodo_backtest.html`. `check-api.sh`: `/promedio?scope=es&mode=nowcast` → `check "/es?mode=nowcast" 200 html`,
`/escanos?scope=es&mode=forecast` → `"/es/escanos?mode=forecast"`, `/escanos?region=abc` →
`"/es/escanos?region=abc"`, `/modelo?scope=es` → `"/metodo/backtest?scope=es"`, y añadir `check /promedio 301
none`, `check /es-zz 404 html` (la función `check` acepta el código; añadir el tipo `none` si no lo acepta).
`smoke.sh`: `/promedio` → `/es`, `/escanos` → `/es/escanos`, `/modelo?scope=es` → `/metodo/backtest?scope=es`.

- [ ] **Step 7: Ejecutar la suite, `node --check` y `bash -n`**

Run: `python -m pytest -m "not integration" -q && node --check web/dist/js/pages/common.js && bash -n deploy/check-api.sh deploy/smoke.sh`
Expected: todo en verde (los tests de `test_web_routes.py` siguen pasando: `/es` casa con `/{scope}`).

- [ ] **Step 8: Commit**

```bash
git add -A mtpy/lib/pages mtpy/controllers web/templates web/dist/js web/dist/css deploy/check-api.sh deploy/smoke.sh tests
git commit -m "Route the event pages under the scope, default to the nowcast and redirect the old URLs"
```

---

### Task 2: Inicio, el índice de eventos

**Files:**
- Create: `mtpy/lib/pages/inicio.py`
- Modify: `mtpy/lib/pages/__init__.py`, `mtpy/controllers/Index.py`
- Create: `web/templates/index.html` (nuevo contenido: el Inicio)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces (`mtpy/lib/pages/inicio.py`):
  - `event_cards(scopes: list, today=None) -> list`: una por fila de `published_scopes()` (orden: `es` primera, el resto por `event_date` ascendente); cada una `{'code', 'title' (event_title), 'event_date', 'days_text', 'run_at', 'frozen', 'href': '/'+code, 'href_seats': '/'+code+'/escanos', 'leader': {'name', 'fullname', 'color', 'pct', 'seats'} | None, 'majority': {'block', 'p', 'color'} | None}`; `leader` = la fila de `headline['nowcast']['parties']` con mayor `pct`; `majority` = la clave de `p_majority` con mayor valor; sin `headline` legible (`bundle_part` del run → `None`), ambos `None`.
  - `past_events(scopes: list) -> list`: para cada ámbito publicado, `published_events(code)` reducido a `{'scope', 'scope_name', 'date', 'name', 'href': '/elecciones?scope=…&date=…'}`, todas juntas de la más reciente a la más antigua.
  - `inicio_context(state: dict) -> dict`: `common_context(state, active='/')` (ámbito por defecto) más `cards`, `past` y `intro` (la frase: "Un modelo que promedia las encuestas y simula los escaños de cada elección: elige una para ver en qué situación está.").
- Consumes: `pages.common` (tarea 1), `pages.elecciones.published_events`, `bundle_part`, `site().run_file`.

- [ ] **Step 1: Tests que fallan**

```python
def test_home_lists_the_published_events_with_their_headline(api, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    status, headers, body = call(api, '/')
    page = html(body)
    assert status == 200 and '<title>Pronóstico electoral · Inicio</title>' in page
    assert page.index('Elecciones generales') < page.index('Elecciones autonómicas · Madrid')
    assert 'href="/es"' in page and 'href="/es/escanos"' in page and 'href="/es-md"' in page
    leader = max(fixture_data('headline')['nowcast']['parties'], key=lambda row: row['pct'])
    assert leader['name'] in page and 'de probabilidad de mayoría absoluta' in page
    assert 'id="event-bar"' not in page and 'id="scope-filter"' not in page and 'initial-data' not in page


def test_home_tolerates_a_scope_whose_run_has_no_headline(api, fresh_app, tmp_path):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    os.remove(os.path.join(str(tmp_path), bundle.path_part('es-md', '20261009-120000', 'headline')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/')
    assert status == 200 and 'Elecciones autonómicas · Madrid' in html(body)


def test_home_lists_the_past_elections_with_a_detail_link(events_site):
    page = html(call(events_site, '/')[2])
    assert page.index('2023') < page.index('2019') and 'href="/elecciones?scope=es&amp;date=2023-07-23"' in page
```

- [ ] **Step 2: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_pages.py -q -k home` → FAIL.

- [ ] **Step 3: Implementar `pages/inicio.py` y `index.html`**

Plantilla: `{% block title %}Inicio{% endblock %}`; párrafo `intro`; sección "Próximas elecciones" con una
tarjeta grande (`class="card card-hero"`) para la primera y `grid` para el resto; cada tarjeta: h2 con el
título, fecha + `days_text`, "{fullname} va primero con el {pct} % y {seats} escaños", "{block}: {p} de
probabilidad de mayoría absoluta", "Última publicación: {run_at|date}" (+ " · congelado" si `frozen`), enlaces
"Ver promedios" y "Escaños"; con `leader` nulo, guiones. Sección "Elecciones pasadas" (lista `past`, oculta si
vacía) y tarjeta final con tres enlaces (Histórico, Rating, Método) con una frase cada uno. Sin `initial`, sin
`<script type="module">` (la isla no se escribe porque `initial` no está en el contexto). `Index.index_action`
renderiza `index.html` con `pages.inicio_context(self.state())` cuando no hay `scope` en la query.

- [ ] **Step 4: Suite en verde** — Run: `python -m pytest -m "not integration" -q` → PASS (actualizar
`test_pages_package_layout` con `inicio`).

- [ ] **Step 5: Commit** — `git commit -m "Add the home page: the index of events and past elections"`

---

### Task 3: Promedios, la primera pestaña del evento

**Files:**
- Create: `mtpy/lib/pages/promedios.py` (funde `index.py` y `promedio.py`); Delete: `mtpy/lib/pages/index.py`,
  `mtpy/lib/pages/promedio.py`; Modify: `mtpy/lib/pages/__init__.py`, `mtpy/lib/pages/inicio.py` (importa
  `majority_rows` de `promedios`), `mtpy/controllers/Promedios.py`
- Modify: `web/templates/promedios.html`
- Create: `web/dist/js/blocks.js`, `web/dist/js/pages/promedios.js`; Delete: `web/dist/js/pages/index.js`,
  `web/dist/js/pages/promedio.js`
- Modify: `web/dist/js/range.js`, `web/dist/js/charts/series.js`
- Test: `tests/test_pages.py`, `tests/test_js_range.py`, create `tests/test_js_blocks.py`

**Interfaces:**
- Produces (`mtpy/lib/pages/promedios.py`): `TABLE_ROWS = 40`, `majority_rows`, `chart_polls` (añade
  `pollster_id` a las claves conservadas), `table_rows`, `pollster_options(polls: list) -> list`
  (`[{'name', 'n'}]` por número de sondeos descendente, empate alfabético), `round_series(series: dict,
  digits: int = 1) -> dict`, `promedios_context(state: dict) -> dict`: `common_context(state, None,
  event=True)` más `headline` (del modo), `majority`, `table_parties`, `table_rows`, `n_polls`, `csv_href`,
  `pollsters` (`pollster_options`), `blocks` (nombres de `meta.bmaps.blocks`, lista vacía si no hay),
  `main` (`meta.bmaps.main` presentes en `series.parties`), `headline_text` (titular, ver abajo) e `initial` =
  `{state, meta, headline, series (round_series), polls (chart_polls), projection, vote, summary, runs}`.
- Produces (`web/dist/js/range.js`): `WINDOW_DAYS = {'3m': 90, '6m': 180, '1a': 365}`, `DEFAULT_WINDOW = 'all'`.
- Produces (`web/dist/js/blocks.js`, puro):
  - `blockSeries(series, blocks)` → `{dates, parties: Object.keys(blocks), mean, lo, hi}`; por fecha y bloque: `mean` = suma de las medias no nulas de sus partidos (null si ninguna), `lo/hi` = `mean ∓ sqrt(Σ((hi_i − lo_i)/2)²)` sobre los partidos con banda.
  - `blockPolls(polls, blocks)` → `{parties, polls: [{date, pollster, pollster_id, <bloque>: suma}]}` (un sondeo sin ningún partido del bloque no lleva la clave).
  - `blockProjection(projection)` → `{dates, groups: {parties: projection.groups.blocks}}` o `null`.
  - `parseSeriesHash(hash, {parties, blocks, windows})` → `{window, view, selected}`: `window` ∈ `windows` (si no, `'all'`), `view` ∈ `{'partidos','bloques'}` (`'bloques'` solo si `blocks.length`), `selected` = nombres del hash presentes en la lista de la vista (null si ninguno válido = por defecto).
  - `seriesHash({window, view, selected})` → `'#ventana=…&vista=…&sel=A,B'` (`encodeURIComponent` por nombre).
- Consumes: `renderSeries`, `setWindow` (series.js), `Catalog`, `wireControls`, `mountWhenReady`,
  `renderBars`, `renderHemicycle`, `renderEvolution`.

- [ ] **Step 1: Test de node de `blocks.js` y de `range.js` que fallan**

`tests/test_js_blocks.py` con el patrón de `test_js_range.py` (`run_js(expr)` importando `blocks.js`):

```python
SERIES = {'dates': ['2026-01-01', '2026-01-02'], 'parties': ['PP', 'VOX', 'PSOE'],
          'mean': {'PP': [30, 31], 'VOX': [10, None], 'PSOE': [28, 28]},
          'lo': {'PP': [28, 29], 'VOX': [8, None], 'PSOE': [26, 26]}, 'hi': {'PP': [32, 33], 'VOX': [12, None], 'PSOE': [30, 30]}}
BLOCKS = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}

def test_block_series_sums_means_and_adds_half_widths_in_quadrature():
    out = run_js('b.blockSeries(series, blocks)')
    assert out['parties'] == ['Derecha', 'Izquierda'] and out['mean']['Derecha'] == [40, 31]
    assert out['lo']['Derecha'] == [40 - (8 ** 0.5), 29] and out['hi']['Derecha'][0] == 40 + (8 ** 0.5)
    assert run_js('b.blockSeries({dates: [], parties: [], mean: {}, lo: {}, hi: {}}, blocks)')['mean']['Derecha'] == []

def test_block_polls_and_projection():
    polls = {'parties': ['PP', 'VOX', 'PSOE'], 'polls': [{'date': '2026-01-01', 'pollster': 'A', 'PP': 30, 'VOX': 10, 'PSOE': 28}, {'date': '2026-01-02', 'pollster': 'B', 'PSOE': 27}]}
    out = run_js('b.blockPolls(polls, blocks)')
    assert out['polls'][0]['Derecha'] == 40 and 'Derecha' not in out['polls'][1] and out['polls'][1]['Izquierda'] == 27
    assert run_js('b.blockProjection({dates: ["d"], groups: {parties: {}, blocks: {names: ["Derecha"], mean: {}, lo: {}, hi: {}}}})')['groups']['parties']['names'] == ['Derecha']
    assert run_js('b.blockProjection({dates: ["d"], groups: {parties: {}}})') is None

def test_series_hash_round_trip_and_fallbacks():
    ctx = '{parties: ["PP", "PSOE"], blocks: ["Derecha"], windows: ["3m", "6m", "1a", "all"]}'
    assert run_js(f'b.parseSeriesHash("#ventana=6m&vista=bloques&sel=Derecha", {ctx})') == {'window': '6m', 'view': 'bloques', 'selected': ['Derecha']}
    assert run_js(f'b.parseSeriesHash("#ventana=9m&vista=bloques&sel=Nadie", {{parties: ["PP"], blocks: [], windows: ["all"]}})') == {'window': 'all', 'view': 'partidos', 'selected': None}
    assert run_js('b.seriesHash({window: "1a", view: "partidos", selected: ["PP", "PSOE"]})') == '#ventana=1a&vista=partidos&sel=PP%2CPSOE' or run_js('b.seriesHash({window: "1a", view: "partidos", selected: ["PP", "PSOE"]})') == '#ventana=1a&vista=partidos&sel=PP,PSOE'
```

(Fijar una de las dos codificaciones de `sel` al implementar y dejar una sola aserción.) En
`test_js_range.py`: `r.WINDOW_DAYS == {'3m': 90, '6m': 180, '1a': 365}`, `r.DEFAULT_WINDOW == 'all'` y
`seriesWindow('1a', …)` = 365 días antes del ancla.

- [ ] **Step 2: Tests de página que fallan**

```python
def test_promedios_orders_the_story_and_carries_the_controls(api):
    page = html(call(api, '/es')[2])
    order = [page.index(marker) for marker in ('id="series"', 'id="headline"', 'id="vote-chart"', 'id="hemicycle"',
                                              'id="evolution"', 'id="pollster"', 'id="polls"', 'id="closing"')]
    assert order == sorted(order)
    assert 'name="window" value="all" checked' in page.replace('\n', ' ') and 'value="3m"' in page and 'value="1a"' in page
    assert 'name="view" value="partidos"' in page and 'name="view" value="bloques"' in page
    assert 'id="pollster-select"' in page and 'href="/metodo' in page
    data = initial(call(api, '/es')[2])
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs'}
    assert all(abs(v * 10 - round(v * 10)) < 1e-9 for v in data['series']['mean']['PP'] if v is not None)
    assert 'pollster_id' in data['polls']['polls'][0]


def test_promedios_hides_the_blocks_view_without_blocks(api, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    meta = fixture_data('meta') | {'scope': 'es-md'}
    meta['bmaps'] = {k: v for k, v in meta['bmaps'].items() if k not in ('max', 'blocks')}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es-md', '20261009-120000', 'meta'), 'meta', meta,
                                                 'es-md', run_id='20261009-120000')
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/es-md')[2])
    assert 'name="view"' not in page and 'id="series-parties"' in page


def test_promedios_headline_sentence_uses_the_mode(api):
    assert 'Si hoy hubiera elecciones' in html(call(api, '/es')[2])
    assert 'El 29 de noviembre de 2026' in html(call(api, '/es', query='mode=forecast')[2])
```

- [ ] **Step 3: Implementar `pages/promedios.py`**

Titular (`headline_text`): Hoy → `'Si hoy hubiera elecciones, {fullname} ganaría con el {pct} % del voto y {seats} escaños; {bloque} tiene un {p} de probabilidad de mayoría absoluta.'`;
Elección → `'El {event_date}, {fullname} ganaría …'` (mismo resto); con la web congelada y el run último,
prefijo "Pronóstico definitivo: ". `{p}` con `fmt_prob`. Firmas de "Interfaces"; `inicio.py` importa
`majority_rows` desde `promedios`.

- [ ] **Step 4: Plantilla `promedios.html`** (orden de la spec "Promedios", tarjetas con `id`
`series`, `headline`, `charts` (voto + escaños), `hemicycle`/`majority`, `evolution`, `pollster`, `polls`,
`closing`): controles del promedio en una fila sobre el gráfico:

```html
<fieldset id="series-window" class="chart-window" hidden><legend>Ventana</legend> 3 meses (3m) · 6 meses (6m) · 1 año (1a) · Todo el ciclo (all, checked)</fieldset>
<fieldset id="series-view" class="chart-window" hidden><legend>Vista</legend> Partidos (partidos, checked) · Bloques (bloques)</fieldset>  {# solo si blocks #}
<fieldset id="series-parties" class="chart-parties" hidden><legend>Partidos</legend> una casilla por partido de series.parties con su punto de color, checked si está en main; <button type="button" data-all>Todos</button> <button type="button" data-none>Ninguno</button></fieldset>
<fieldset id="series-blocks" class="chart-parties" hidden><legend>Bloques</legend> una casilla por bloque, todas marcadas</fieldset>
```

Tarjeta `pollster`: `<label>Casa <select id="pollster-select">` con `pollsters` (la primera seleccionada) y
`<div class="chart chart-tall" id="pollster-chart">`. Tarjeta `closing`: "¿Cómo promediamos las encuestas?"
(enlace a `/metodo/promedio` si la sección existe, si no a `/metodo`) y los dos enlaces globales
"Cómo acertaron las encuestas en las elecciones anteriores" → `/elecciones{{ scope_query }}` y "Las casas que
sondean este ámbito" → `/casas{{ scope_query }}`. Las descripciones y "Cómo leerlo" se escriben en la tarea 4
(aquí basta con los párrafos "Qué es esto" actuales).

- [ ] **Step 5: `promedios.js`**

Lee la isla; estado inicial = `parseSeriesHash(location.hash, …)`; `paint()` dibuja el gráfico principal
con `renderSeries` (vista partidos: `series`, `polls`, `projection`, `parties = series.parties`, `selected`;
vista bloques: `blockSeries`, `blockPolls`, `blockProjection`, `parties = blocks`, colores
`catalog.blockColor`), `window` del estado y `anchor = meta.date_last || meta.as_of`; el gráfico de la casa
con los mismos argumentos y `polls` filtrados por `pollster_id` (o `pollster`) de `#pollster-select`; las
casillas cambian `selected` y redibujan; la ventana llama a `setWindow` en los dos gráficos; cada cambio escribe
`seriesHash`. Los demás gráficos como en `index.js`. Los `fieldset` se muestran (`hidden = false`) al montar.

- [ ] **Step 6: Suite, node y tamaño**

Run: `python -m pytest -m "not integration" -q && for f in web/dist/js/blocks.js web/dist/js/range.js web/dist/js/pages/promedios.js web/dist/js/charts/series.js; do node --check $f; done`
Expected: PASS. Medir con el paquete local: `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8001 &`,
`curl -s http://127.0.0.1:8001/es | wc -c` (anotar en el informe; recortar `history.runs`/`summary` si supera
~500 KB), parar uvicorn.

- [ ] **Step 7: Commit** — `git commit -m "Merge the home and average pages into Promedios with window, party and block views and a house chart"`

---

### Task 4: Lenguaje visual: tokens, fuentes, tarjetas y "Cómo leerlo" en todas las páginas

**Files:**
- Modify: `web/dist/css/site.css` (reescritura)
- Create: `web/dist/vendor/fonts/SourceSerif4-Semibold.woff2`, `web/dist/vendor/fonts/SourceSans3-Regular.woff2`,
  `web/dist/vendor/fonts/LICENSE` (E5)
- Create: `web/templates/_macros.html`
- Modify: `web/templates/base.html` (`<meta name="description">`, bloque `description`), `index.html`,
  `promedios.html`, `escanos.html`, `elecciones.html`, `eleccion.html`, `casas.html`, `casa.html`,
  `metodo.html`, `metodo_backtest.html`, `error.html`, `_event_bar.html`, `_scope_filter.html`
- Modify: `web/dist/js/charts/base.js` (`THEME.fontFamily` con la fuente del texto; animación cero bajo
  `prefers-reduced-motion`)
- Test: `tests/test_pages.py`, `tests/test_web_routes.py` (los recursos referenciados existen: las fuentes)

**Interfaces:**
- Produces (`web/templates/_macros.html`):
  - `{% macro card_start(id, title, lead, subtitle=None, classes='') %}` → `<section class="card {{ classes }}" id="{{ id }}"><header class="card-head"><h2>{{ title }}{% if subtitle %} <span class="mode-subtitle">{{ subtitle }}</span>{% endif %}</h2><p class="lead">{{ lead }}</p></header>`
  - `{% macro card_end() %}` → `</section>`
  - `{% macro how(open=False) %}` (con `caller()`): `<details class="how"{% if open %} open{% endif %}><summary>Cómo leerlo</summary><div class="how-body">{{ caller() }}</div></details>`
  - `{% macro headline(text) %}` → `<p class="headline">{{ text }}</p>` (el titular de página, fuera de tarjeta)
  - `{% macro closing(title, href, teaser, links=[]) %}` → la tarjeta de cierre: h2 con el enlace `title`, `teaser` y una lista de `links` (`{'href', 'label'}`)
- CSS: tokens `--color-bg` (pastel cálido), `--color-card: #fff`, `--color-text`, `--color-muted` (contraste AA
  sobre `--color-bg`), `--color-line`, `--color-accent`, `--font-head: 'Source Serif 4', Georgia, serif`,
  `--font-body: 'Source Sans 3', system-ui, sans-serif`, `--max-width: 1100px`; `@font-face` con
  `font-display: swap` y `src: url('/dist/vendor/fonts/…woff2')`; cabecera (nombre a la izquierda, nav a la
  derecha, fila con scroll horizontal en móvil), `#event-bar` (título, fecha, pestañas, controles a la derecha,
  apilado en móvil), `#scope-filter` discreto, `.card`/`.card-head`/`.lead`/`.how`/`.headline`/`.chip`,
  `.card-hero`, tablas con primera columna `position: sticky` en móvil, `.chart` 360/240 px, objetivos táctiles
  ≥ 40 px, `@media (prefers-reduced-motion: reduce)`.

- [ ] **Step 1: Tests que fallan**

```python
def test_every_page_uses_the_card_pattern_and_the_fonts(events_site):
    for path in ('/', '/es', '/es/escanos', '/elecciones', '/elecciones?scope=es&date=2023-07-23', '/metodo'):
        page = html(call(events_site, path.split('?')[0], query=path.partition('?')[2])[2])
        assert 'Qué es esto' not in page and '<meta name="description"' in page, path
        if path not in ('/', '/metodo'):
            assert 'class="how"' in page and 'class="lead"' in page, path


def test_fonts_are_vendored_and_declared():
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert "url('/dist/vendor/fonts/" in css and 'font-display: swap' in css
    assert os.path.isfile(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'vendor', 'fonts', 'LICENSE'))
```

(Si E5 acaba en pila del sistema, el segundo test comprueba solo que no hay URL externa en el CSS.)

- [ ] **Step 2: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_pages.py -q -k "card_pattern or fonts"` → FAIL.

- [ ] **Step 3: Fuentes** — descargar de las releases de `adobe-fonts/source-serif` y `adobe-fonts/source-sans`
los woff2 estáticos (Semibold de la serif, Regular de la sans), copiar `LICENSE.md` como `LICENSE`,
comprobar `du -k` ≤ 150 KB entre las dos.

- [ ] **Step 4: CSS y macros** — reescribir `site.css` con los tokens y reglas de "Interfaces"; crear
`_macros.html`; `base.html` importa las macros (`{% import '_macros.html' as m %}` en cada plantilla que las use)
y añade `<meta name="description" content="{% block description %}Pronóstico electoral: promedio de encuestas y simulación de escaños.{% endblock %}">`.

- [ ] **Step 5: Convertir todas las plantillas al patrón** — cada `<section class="card">` pasa a
`m.card_start(...)`/`m.card_end()` con `lead` = la primera frase del "Qué es esto" actual reescrita como
descripción (qué muestra y de dónde sale) y el resto dentro de `{% call m.how(open=loop.first) %}` (abierto
solo en la primera tarjeta de cada página); desaparecen los párrafos "Qué es esto:". Las tarjetas de cierre
usan `m.closing`. La tarjeta `#event-bar` y el filtro adoptan las clases nuevas. `error.html` con el mismo
aspecto. Textos nuevos solo donde la spec los fija en tareas posteriores.

- [ ] **Step 6: Comprobación en navegador local** — `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8001`
y recorrer `/`, `/es`, `/es/escanos`, `/elecciones`, `/metodo/backtest` a 1280 px y a 360 px (sin
desplazamiento horizontal fuera de `table-wrap`); anotar en el informe lo visto (los gráficos ECharts no se
comprueban aquí más allá de que se dibujen).

- [ ] **Step 7: Suite y node** — Run: `python -m pytest -m "not integration" -q && node --check web/dist/js/charts/base.js` → PASS.

- [ ] **Step 8: Commit** — `git commit -m "Restyle the site: tokens, vendored fonts, cards with lead and how-to-read, event bar"`

---

### Task 5: Escaños, reexplicado

**Files:**
- Modify: `web/templates/escanos.html`, `mtpy/lib/pages/escanos.py` (`headline_text`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces: `escanos_context` añade `headline_text`: `'{bloque} es el bloque con más probabilidad de mayoría absoluta ({p}); la mayoría está en {majority} escaños de {n_seats}.'` (bloque = mayor `p_majority` de `summary.blocks`; Hoy/Elección por el `title_suffix`).

- [ ] **Step 1: Test que falla**

```python
def test_escanos_explains_the_fan_and_the_evolution_and_closes_with_the_method(api):
    page = html(call(api, '/es/escanos')[2])
    assert 'bloque con más probabilidad de mayoría absoluta' in page
    fan = page[page.index('id="fan"'):page.index('id="evolution"')]
    assert 'deriva' in fan and 'horizonte 0' in fan and 'no es una predicción de trayectoria' in fan
    evolution = page[page.index('id="evolution"'):page.index('id="downloads"')]
    assert 'run' in evolution and 'huecos' in evolution and 'no es una serie de sondeos' in evolution
    assert '¿Cómo simulamos los escaños?' in page and 'href="/elecciones?scope=es"' in page and 'href="/casas?scope=es"' in page
```

- [ ] **Step 2: Ejecutar y ver que falla** — Run: `python -m pytest tests/test_pages.py -q -k escanos_explains` → FAIL.

- [ ] **Step 3: Escribir los textos** — para cada tarjeta (`parties`, `blocks`, `coalition`, `districts`,
`fan`, `evolution`, `downloads`): `lead` de una o dos frases y "Cómo leerlo" en tres partes (qué es cada
elemento; cómo interpretarlo; qué no dice), ≤ 120 palabras, sin jerga ni cifras a mano. Abanico: cada
columna es "a cuántos días de la elección"; el horizonte 0 es hoy; el abanico se abre porque la deriva de la
opinión crece con los días; la columna de la derecha es el pronóstico del modo Elección; no es una predicción
de trayectoria sino de incertidumbre. Evolución: qué es un run, qué es un punto hueco, por qué cambian los
puntos (sondeos nuevos, no cambios del modelo), no es una serie de sondeos. Titular con `m.headline`. Cierre
con `m.closing('¿Cómo simulamos los escaños?', href, teaser, links)` (href a `/metodo/escanos` si existe la
sección; si no, `/metodo`).

- [ ] **Step 4: Suite** — Run: `python -m pytest -m "not integration" -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Explain every card of the seats page and close it with the method link"`

---

### Task 6: Histórico de errores: filtros, histograma, diana, semanas, glosario y detalle

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py`, `mtpy/lib/pages/__init__.py`
- Create: `web/templates/_glosario.html`, `web/dist/js/errors.js`
- Modify: `web/templates/elecciones.html`, `web/templates/eleccion.html`, `web/dist/js/pages/elecciones.js`,
  `web/dist/js/charts/histogram.js`, `web/dist/js/charts/bullseye.js`
- Test: `tests/test_pages.py`, create `tests/test_js_errors.py`

**Interfaces:**
- Produces (`mtpy/lib/pages/elecciones.py`):
  - `error_rows(scope: str, dates: list) -> Optional[list]`: filas de `analysis/{scope}/poll-errors` con `event_date in dates`, pasadas por `with_abs`; `None` sin análisis.
  - `elecciones_context` añade `analysis_missing`, `errors` (`error_rows` o `[]`), `error_blocks` (los `blocks` de `poll-errors`), `filter_events` (`[{'date', 'name'}]` de `events`), `filter_pollsters` (nombres únicos de `errors`, alfabético), `headline_text` (`'En las {n} elecciones analizadas, el último sondeo de cada casa se desvió de media {x} puntos en el margen entre bloques.'`, `x` = media de `error_blocks_abs`; sin análisis, `'{n} elecciones analizadas.'`), y en `initial`: `errors`, `error_blocks`, `events` (`filter_events`). Columnas de la lista: Elección, Sondeos, Casas, Error medio, Error de bloques, Sesgo del margen, Modelo.
  - `eleccion_context` añade a `initial` `poll_errors`: `[{'error_avg', 'error_blocks_abs'}]` por sondeo de `polls.rows`; las tablas pierden "Sesgo de bloques".
- Produces (`web/dist/js/errors.js`, puro):
  - `filterRows(rows, {event = null, pollster = null})` → filas con `event_date === event` (si no null) y `pollster === pollster` (si no null).
  - `meanOf(rows, measure)` → media de los valores finitos o `null`.
  - `weekOf(days)` → `Math.floor((days + 1) / 7)`.
  - `weeksTable(rows, measure)` → `{weeks: [1, 2, …] (presentes, ascendente), rows: [{event, event_date, cells: {week: {n, mean}}}, …, {event: 'Todas', event_date: null, cells}]}` (elecciones de la más reciente a la más antigua).
  - `parseErrorsHash(hash, {events, pollsters})` → `{event, pollster, measure}` (`event` ∈ `events` o null, `pollster` ∈ `pollsters` o null, `measure` ∈ `{'error_blocks_abs','error_avg'}`, por defecto `'error_blocks_abs'`).
  - `errorsHash({event, pollster, measure})` → `'#eleccion=…&casa=…&medida=…'` (solo las claves no nulas; `medida` solo si no es la de por defecto).
- Produces (`charts/histogram.js`): opciones nuevas `unit = 'escaños'`, `label = null` (función `(start, step) => texto`), `markLabel = null` (E6). (`charts/bullseye.js`): `groupOf = null`, `colors = {}`, `legend = false`, `labels = true` (E6); con `groupOf`, una serie `scatter` por grupo (nombre = grupo, color = `colors[grupo]`), leyenda con `type: 'scroll'`.
- Consumes: `renderHistogram`, `renderBullseye`, `renderLine`, `renderBars`, `wireControls`, `tryDraw`, `mountWhenReady`, `_macros.html`.

- [ ] **Step 1: Test de node de `errors.js` que falla** (`tests/test_js_errors.py`, patrón de `test_js_range.py`):

```python
ROWS = [
    {'event': '2023-23J', 'event_date': '2023-07-23', 'pollster': 'A', 'days': 7, 'error_avg': 1.0, 'error_blocks_abs': 2.0},
    {'event': '2023-23J', 'event_date': '2023-07-23', 'pollster': 'B', 'days': 13, 'error_avg': 3.0, 'error_blocks_abs': 4.0},
    {'event': '2019-10N', 'event_date': '2019-11-10', 'pollster': 'A', 'days': 20, 'error_avg': 2.0, 'error_blocks_abs': None},
]

def test_filter_mean_and_weeks():
    assert len(run_js('e.filterRows(rows, {event: "2023-07-23"})')) == 2
    assert len(run_js('e.filterRows(rows, {pollster: "A"})')) == 2
    assert run_js('e.meanOf(rows, "error_blocks_abs")') == 3.0 and run_js('e.meanOf([], "error_avg")') is None
    assert run_js('e.weekOf(6)') == 1 and run_js('e.weekOf(13)') == 2 and run_js('e.weekOf(42)') == 6
    table = run_js('e.weeksTable(rows, "error_avg")')
    assert table['weeks'] == [1, 2, 3] and table['rows'][0]['event'] == '2023-23J' and table['rows'][-1]['event'] == 'Todas'
    assert table['rows'][-1]['cells']['1'] == {'n': 1, 'mean': 1.0}

def test_errors_hash_round_trip_and_unknown_values_fall_back():
    ctx = '{events: ["2023-07-23", "2019-11-10"], pollsters: ["A", "B"]}'
    assert run_js(f'e.parseErrorsHash("#eleccion=2019-11-10&casa=B&medida=error_avg", {ctx})') == {'event': '2019-11-10', 'pollster': 'B', 'measure': 'error_avg'}
    assert run_js(f'e.parseErrorsHash("#eleccion=2011-11-20&casa=Z&medida=x", {ctx})') == {'event': None, 'pollster': None, 'measure': 'error_blocks_abs'}
    assert run_js('e.errorsHash({event: "2019-11-10", pollster: null, measure: "error_blocks_abs"})') == '#eleccion=2019-11-10'
    assert run_js('e.errorsHash({event: null, pollster: null, measure: "error_blocks_abs"})') == ''
```

- [ ] **Step 2: Tests de página que fallan**

```python
@pytest.fixture
def history_site(events_site, fresh_app):
    write_analysis(fresh_app.fs)
    return events_site


def test_elecciones_only_counts_the_published_elections(history_site, fresh_app):
    errors = fixture_data('poll-errors')
    errors['rows'].append(dict(errors['rows'][0], event_date='2011-11-20', event='2011-20N'))
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es')
    data = initial(call(history_site, '/elecciones')[2])
    assert {row['event_date'] for row in data['errors']} <= {'2023-07-23', '2019-11-10'}
    assert [e['date'] for e in data['events']] == ['2023-07-23', '2019-11-10'] and data['error_blocks'] == ['Derecha', 'Izquierda']
    page = html(call(history_site, '/elecciones')[2])
    assert 'En las 2 elecciones analizadas' in page and 'id="errors-filter"' in page and 'id="errors-histogram"' in page
    assert 'id="errors-bullseye"' in page and 'id="weeks-table"' in page and 'Sesgo de bloques' not in page and 'Desviación hacia' not in page
    assert 'id="glosario"' in page and '¿Cómo calculamos el rating de las encuestadoras?' in page


def test_elecciones_without_analysis_keeps_the_list(events_site):
    page = html(call(events_site, '/elecciones')[2])
    assert 'Análisis no publicado todavía' in page and 'id="events-table"' in page and '2 elecciones analizadas' in page


def test_eleccion_detail_has_the_error_histogram_and_the_glossary(events_site):
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    page = html(body)
    assert 'id="poll-histogram"' in page and 'id="glosario"' in page and 'Sesgo de bloques' not in page
    assert initial(body)['poll_errors'][0].keys() >= {'error_avg', 'error_blocks_abs'}
```

- [ ] **Step 3: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_js_errors.py tests/test_pages.py -q -k "errors or elecciones or eleccion"` → FAIL.

- [ ] **Step 4: Implementar `errors.js`, las opciones de `histogram.js` y `bullseye.js`** (firmas de "Interfaces").

- [ ] **Step 5: Implementar `elecciones.py` y las plantillas**

`_glosario.html`: `<details class="how" id="glosario"><summary>Qué mide cada columna</summary><dl>` con las tres
medidas (y herding donde `with_herding` sea verdadero) con los textos del glosario de la spec ("Global
Constraints"). `elecciones.html`, orden: `m.headline`, tarjeta `filters` (`<form id="errors-filter">` con
`<select name="eleccion">` (Todas + `filter_events`) y `<select name="casa">` (Todas + `filter_pollsters`),
contador `<span id="errors-count">`), `histogram` (radios `medida`: Error de bloques / Error medio;
`<div class="chart" id="errors-histogram">`), `bullseye` (`#errors-bullseye`), `weeks` (`<table
id="weeks-table">` vacía que el JS rellena con `weeksTable`, y un `<noscript>` "La tabla necesita
JavaScript", como la calculadora de coaliciones), `sector` (hoy),
`events` (hoy, columnas nuevas), glosario, `m.closing`. Sin análisis: aviso "Análisis no publicado todavía." en
`filters`, `histogram`, `bullseye` y `weeks`. `eleccion.html`: tarjeta `poll-histogram` tras `pollsters`
(radios de medida, `renderHistogram` con `unit: 'puntos'`, `width: 1`, `markLabel: 'Media: …'`), glosario,
columnas sin "Sesgo de bloques".

- [ ] **Step 6: `elecciones.js`** — en la lista: estado de `parseErrorsHash`; `apply()` filtra, pinta el
contador "N sondeos de M", el histograma (`values` = `measure` de las filas, `width: 1`, `unit: 'puntos'`,
`label: (s) => `${s} a ${s + 1} puntos``, `markLabel: 'Media: ' + fmtNum(mean, 1)`), la diana (`groupOf: row =>
row.event`, `colors` por elección de una escala neutra fija de 8 tonos, `legend: true`, `labels: false`) y la
tabla por semanas (`weeksTable`); cada cambio escribe `errorsHash`. En el detalle: el histograma de
`poll_errors` y lo de hoy.

- [ ] **Step 7: Suite y node** — Run: `python -m pytest -m "not integration" -q && for f in web/dist/js/errors.js web/dist/js/pages/elecciones.js web/dist/js/charts/histogram.js web/dist/js/charts/bullseye.js; do node --check $f; done` → PASS (los histogramas de Escaños no cambian: `test_escanos_*` siguen en verde).

- [ ] **Step 8: Commit** — `git commit -m "Turn the elections page into the error history: filters, histogram, bullseye, weeks table and glossary"`

---

### Task 7: Rating de encuestadoras y perfil con la convención única

**Files:**
- Modify: `mtpy/lib/pages/casas.py`, `web/templates/casas.html`, `web/templates/casa.html`
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces (`mtpy/lib/pages/casas.py`): `profile_rows` sin cambios; `casa_context['path']` filas con
  `error_blocks_abs` (vía `elecciones.with_abs`) y `casas_context['headline_text']` =
  `'{n} casas con rating para la próxima elección; la mejor valorada es {casa} ({rating}).'`;
  `casa_context['headline_text']` = `'{casa}: rating {rating} de 100, calidad previa {quality}.'`.

- [ ] **Step 1: Tests que fallan**

```python
def test_casas_uses_the_glossary_and_drops_the_block_bias(houses):
    page = html(call(houses, '/casas')[2])
    assert 'casas con rating' in page and 'id="glosario"' in page and 'Sesgo de bloques' not in page
    assert '¿Cómo calculamos el rating de las encuestadoras?' in page and 'id="scope-filter"' in page


def test_casa_profile_shows_the_magnitude_and_the_signed_margin(houses):
    page = html(call(houses, '/casas', query='id=1')[2])
    assert '<th scope="col">Error de bloques</th>' in page and '<th scope="col">Sesgo del margen</th>' in page
    assert 'Sesgo de bloques' not in page and 'miden lo mismo' not in page and 'id="glosario"' in page
```

- [ ] **Step 2: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_pages.py -q -k "casas_uses or casa_profile_shows"` → FAIL.

- [ ] **Step 3: Implementar** — columnas del ranking: Casa, Rating, Calidad, Sondeos, Elecciones, Error medio,
Error de bloques, Herding; camino al rating: Elección, Rating, Rating ajustado, Error de bloques
(`error_blocks_abs`), Sesgo del margen (`error_blocks`), Sondeos (comprobar en `query_pollster`/`query_ratings`
el signo de `error_blocks` de la base antes de etiquetar; anotarlo en el informe). Textos `lead`/"Cómo leerlo"
de cada tarjeta (ranking, efectos de casa, herding, descargas; camino, diana, efectos por elección, herding
por ciclo) y `m.closing` con `/metodo/rating` (o `/metodo` hasta la tarea 11). Glosario con `with_herding`.

- [ ] **Step 4: Suite** — Run: `python -m pytest -m "not integration" -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Align the pollster pages with the error glossary and explain every card"`

---

### Task 8: Método: portada, subnavegación, backtest con gráfico de escaños y cifras vivas

**Files:**
- Modify: `mtpy/lib/pages/metodo.py`, `mtpy/lib/pages/backtest.py`, `mtpy/lib/pages/__init__.py`,
  `mtpy/controllers/Metodo.py`
- Create: `web/templates/_metodo_nav.html`
- Modify: `web/templates/metodo.html` (portada), `web/templates/metodo_backtest.html`,
  `web/dist/js/pages/metodo.js`
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `METODO_SECTIONS`, `subpages`, `method_href` (tarea 1).
- Produces (`mtpy/lib/pages/metodo.py`):
  - `thresholds_rows() -> list` → por fila del catálogo: `{'code', 'name', 'threshold' (float|None), 'threshold_scope' (float|None), 'rating_weight' (float), 'seats'}`.
  - `national_figures() -> dict` → `{'n_sim': meta.n_sim del último run de es (None sin run), 'coverage': [{'horizon', 'cov95'}] del by-horizon de es (lista vacía sin backtest), 'events': backtest.meta.events, 'horizons': backtest.meta.horizons}`.
  - `metodo_context(state) -> dict`: `common_context(state, '/metodo')` más `subpages(None)`, `figures` (`national_figures`), `thresholds` (`thresholds_rows`) y `sections` (las tres transversales: datos, publicacion, limitaciones) y, hasta que entren los textos, `legacy` = True (E10).
  - `metodo_section_context(state, section) -> dict`: 404 `section not found` si `section` no está o no tiene `template`; común + `subpages(section)` + `figures` + `thresholds` + `section`; para `'backtest'`, además `backtest_context` (que pasa a aceptar `active='/metodo'` y a incluir `initial['seats'] = {'mae': series('mae_seats'), 'mae_h': series('mae_seats_h')}`).
- Consumes: `webapi.catalogue()`, `site().latest_run`, `part`, `bundle_part`, `backtest_part`.

- [ ] **Step 1: Tests que fallan**

```python
def test_metodo_landing_has_the_index_and_only_links_available_sections(api):
    page = html(call(api, '/metodo')[2])
    assert '¿Cómo promediamos las encuestas?' in page and 'href="/metodo/backtest"' in page
    assert 'href="/metodo/promedio"' not in page  # sin texto todavía (tarea 9 lo activa)
    assert 'id="datos"' in page and 'id="publicacion"' in page and 'id="limitaciones"' in page
    assert 'se congela' in page and 'fuera de España' in page
    assert '1.000 simulaciones' not in page and 'id="scope-filter"' not in page


def test_thresholds_and_figures_come_from_the_catalogue_and_the_bundle(api, fresh_app):
    write_backtest(fresh_app.fs)
    rows = {row['code']: row for row in pages.thresholds_rows()}
    assert rows['es']['threshold'] == 3 and rows['es-cn']['threshold_scope'] == 4 and rows['es-md']['rating_weight'] == 0.5
    figures = pages.national_figures()
    assert figures['n_sim'] == fixture_data('meta')['n_sim'] and figures['coverage'][0]['horizon'] == min(fixture_data('backtest-meta')['horizons'])


def test_backtest_subpage_has_the_seats_chart_and_the_subnav(api, fresh_app):
    write_backtest(fresh_app.fs)
    status, _, body = call(api, '/metodo/backtest', query='scope=es')
    page = html(body)
    assert status == 200 and 'id="seats-chart"' in page and 'id="metodo-nav"' in page and 'id="scope-filter"' in page
    assert initial(body)['seats'].keys() == {'mae', 'mae_h'}
```

- [ ] **Step 2: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_pages.py -q -k "metodo or thresholds or backtest_subpage"` → FAIL.

- [ ] **Step 3: Implementar `metodo.py`, `backtest.py`, el controlador y las plantillas**

`metodo.html`: `m.headline` ("El modelo tiene dos partes: un promedio de encuestas y una simulación de
escaños; estas páginas cuentan cómo funcionan y cuánto aciertan."), introducción de dos párrafos, índice de
cuatro tarjetas (título como enlace si `available`, si no en texto con "próximamente"), secciones `datos`,
`publicacion`, `limitaciones` (texto actual; veda reescrita: "La publicación se congela al empezar la veda,
cinco días antes de la votación, con el pronóstico definitivo{% if frozen and freeze.since %}: desde el
{{ freeze.since|date }}{% endif %}; los sondeos publicados fuera de España durante la veda se cargan en la base
pero el modelo no los usa ni la web los muestra." sin ninguna fecha escrita a mano), y,
mientras `legacy`, la sección "Cómo funciona el modelo" con el texto actual de `promedio`, `casas`,
`simulacion`, `escanos`, `evaluacion` sin las "1.000" (usar `figures.n_sim|int`) y sin la frase de calibración
a mano (usar `figures.coverage`). `_metodo_nav.html` (`<nav id="metodo-nav">` con `subpages`).
`metodo_backtest.html`: `_metodo_nav`, `_scope_filter`, las tarjetas de `modelo.html` con el patrón, nueva
tarjeta `seats` con `<div class="chart" id="seats-chart">` y la tabla "Escaños y mayorías". `metodo.js`:
`renderLines(seatsEl, [{name: 'Modo Hoy', color: COLORS.model, points}, {name: 'Modo Elección', color: '#7f3c8d', dashed: true, points}], {formatter: x => fmtNum(x, 1)})`.

- [ ] **Step 4: Suite y node** — Run: `python -m pytest -m "not integration" -q && node --check web/dist/js/pages/metodo.js` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Give the method a landing page, a subnavigation and the backtest subpage with a seats-error chart"`

---

### Task 9: Texto «¿Cómo promediamos las encuestas?»

**Files:**
- Create: `web/templates/metodo_promedio.html`
- Modify: `mtpy/lib/pages/metodo.py` (`METODO_SECTIONS['promedio']['template']`, retirar de `legacy` las
  secciones `promedio` y la parte de pesos de `casas`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `metodo_section_context` (`figures`, `subpages`, `query`), `_metodo_nav.html`, `_macros.html`.
- Fuentes: `docs/analisis-2026-09/forecaster.md`, `computer-weights.md`, `metodo-6-house-effects.md`,
  `metodo-10-error-estandar-cluster.md`, `metodo-13-herding.md`, `metodo-4-sucesion-encuestas.md`;
  código: `mtpy/lib/forecaster.py` (`set_disp_params`, `fit_dispersion`, `fit_house_effects`),
  `mtpy/lib/computer.py` (pesos).

- [ ] **Step 1: Test que falla**

```python
def test_metodo_promedio_text(api):
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and '<title>Pronóstico electoral · ¿Cómo promediamos las encuestas?</title>' in page
    for anchor in ('que-entra', 'peso', 'efectos-de-casa', 'regresion', 'banda', 'voto-estimado'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'id="metodo-nav"' in page and '<math' in page and 'En resumen' in page
    assert 'href="/metodo/promedio"' in html(call(api, '/metodo')[2])
```

- [ ] **Step 2: Ejecutar y ver que falla** — Run: `python -m pytest tests/test_pages.py -q -k metodo_promedio` → FAIL.

- [ ] **Step 3: Escribir el texto** (1.200-2.000 palabras, prosa, h2 con esas anclas): "En resumen" (3-4 líneas);
`que-entra` (qué sondeos entran y cuáles no: `wban`/`exit`, encargados por partidos a casas reales sí, firmados
por un partido no; sucesión de partidos); `peso` (muestra con rendimiento decreciente, sobrerrepresentación y
series diarias, rating, dispersión y herding: M12 y su variante simétrica activa por defecto; una fórmula del
peso en `<math>`); `efectos-de-casa` (medida en el ciclo, prior histórico, recentrado entre casas; los gráficos
muestran los sondeos tal como se publicaron); `regresion` (regresión local ponderada: qué es, ancho de la
ventana); `banda` (error estándar con cluster por casa; qué mide y qué no); `voto-estimado` (lectura en `as_of`,
`figures.n_sim` sorteos no se cuentan aquí: enlazar a escaños). Cada afirmación comprobada en el código o en los
docs; el informe de la tarea lista "afirmación → fichero:línea o doc". Activar `template` en `METODO_SECTIONS`.

- [ ] **Step 4: Suite** — Run: `python -m pytest -m "not integration" -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Write the method text on how the polls are averaged"`

---

### Task 10: Texto «¿Cómo simulamos los escaños?»

**Files:**
- Create: `web/templates/metodo_escanos.html`
- Modify: `mtpy/lib/pages/metodo.py` (`template`, retirar `simulacion`/`escanos` de `legacy`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `figures` (`n_sim`, `coverage`), `thresholds`, `_metodo_nav.html`.
- Fuentes: `simulator.md`, `stat.md`, `metodo-1-otros-barrera.md`, `metodo-2-resumenes.md`,
  `metodo-5-nowcast-horizonte.md`, `metodo-7-composicion.md`, `metodo-8-edad-partido.md`,
  `metodo-9-oscilaciones-autonomicas.md`; código: `mtpy/lib/simulator.py`, `mtpy/lib/data.py`
  (`resolve_thresholds`).

- [ ] **Step 1: Test que falla**

```python
def test_metodo_escanos_text(api):
    page = html(call(api, '/metodo/escanos')[2])
    for anchor in ('sorteo', 'hoy-y-eleccion', 'circunscripciones', 'dhondt', 'barreras', 'cifras'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'id="thresholds-table"' in page and 'Canarias' in page and 'Comunidad Valenciana' in page
    assert str(fixture_data('meta')['n_sim']) in page.replace('.', '') and '1.000 simulaciones' not in page
    assert '<math' in page and 'Ejemplo' in page
```

- [ ] **Step 2: Ejecutar y ver que falla** — Run: `python -m pytest tests/test_pages.py -q -k metodo_escanos` → FAIL.

- [ ] **Step 3: Escribir el texto**: "En resumen"; `sorteo` (del promedio al voto: error del promedio, error
histórico de la última semana, `figures.n_sim` simulaciones, sorteos independientes y por qué el conjunto está
desactivado); `hoy-y-eleccion` (deriva: crece con los días, proporcional al nivel, mayor en partidos jóvenes,
con la fórmula); `circunscripciones` (swing proporcional, herencias, oscilaciones autonómicas y provinciales);
`dhondt` (ejemplo resuelto con una tabla de 5 escaños y 3 partidos); `barreras` (tabla `#thresholds-table` de
`thresholds`: Ámbito, Barrera, Barrera de circunscripción, con nota de simplificaciones: Canarias (doble barrera
`threshold`/`threshold_scope`), Comunidad Valenciana (solo autonómica), Extremadura, y que cada elección pasada
usa la barrera de su momento); `cifras` (mediana, intervalo 95 %, probabilidades, escenario central). Comprobar
cada afirmación; informe con la lista.

- [ ] **Step 4: Suite** — Run: `python -m pytest -m "not integration" -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Write the method text on how the seats are simulated"`

---

### Task 11: Texto «¿Cómo calculamos el rating de las encuestadoras?»

**Files:**
- Create: `web/templates/metodo_rating.html`
- Modify: `mtpy/lib/pages/metodo.py` (`template`, retirar `casas`/`evaluacion` de `legacy`; `legacy` queda
  falso y la sección "Cómo funciona el modelo" desaparece de la portada)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `thresholds` (para `rating_weight`), `_metodo_nav.html`, enlaces a `/casas{{ scope_query }}`.
- Fuentes: `computer-ratings.md`, `computer-estimators.md`, `metodo-6-house-effects.md` (M6b),
  `metodo-11-ambitos-autonomicos.md` (ratings transversales), `metodo-13-herding.md`; código:
  `mtpy/lib/computer.py` (`compute_ratings`, `herding_summary`).

- [ ] **Step 1: Test que falla**

```python
def test_metodo_rating_text(api):
    page = html(call(api, '/metodo/rating')[2])
    for anchor in ('que-se-mide', 'calidad-previa', 'rating-final', 'ranking-global', 'herding'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'la mitad' in page or '50 %' in page  # rating_weight 0,5 leído del catálogo
    assert '<math' in page
    landing = html(call(api, '/metodo')[2])
    assert 'Cómo funciona el modelo' not in landing and 'href="/metodo/rating"' in landing
```

- [ ] **Step 2: Ejecutar y ver que falla** — Run: `python -m pytest tests/test_pages.py -q -k metodo_rating` → FAIL.

- [ ] **Step 3: Escribir el texto**: "En resumen"; `que-se-mide` (sesgo entre bloques y dentro del bloque,
frente al sector de la misma semana y distancia; por qué un fallo compartido no penaliza); `calidad-previa`
(qué es, cómo pesa más con menos sondeos, fórmula de la mezcla); `rating-final` (rating ajustado y final, 0-100,
cómo entra en el peso); `ranking-global` (todas las elecciones con resultado, estrictamente anteriores; las
autonómicas con `rating_weight` del catálogo: "pesan la mitad" calculado de `thresholds`); `herding` (la
medida, por qué es información y no entra en el rating, qué hace en el peso). Comprobar cada afirmación;
informe con la lista. Quitar `legacy` de la portada.

- [ ] **Step 4: Suite** — Run: `python -m pytest -m "not integration" -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Write the method text on how the pollster ratings are computed"`

---

### Task 12: Cierre: documentación, scripts, verificación final

**Files:**
- Modify: `deploy/README.md` ("La web", "Cómo añadir una página"), `docs/web/contrato.md` (párrafo de rutas
  de páginas, l. 402-413), `deploy/check-api.sh`, `deploy/smoke.sh` (rutas del método y del histórico),
  `docs/superpowers/specs/2026-10-13-web-fase-7-diseno-design.md` ("Estado al cierre"),
  `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` (fila 7 de "Fases" y nota de cierre)
- Test: ninguno nuevo; verificación

- [ ] **Step 1: README y contrato** — "La web": el mapa nuevo (Inicio, `/{scope}`, `/{scope}/escanos`,
globales con `?scope=`, `/metodo/*`), las redirecciones, la barra de evento, el filtro, el patrón de tarjeta,
el glosario, las fuentes en `/dist/vendor/fonts/`; "Cómo añadir una página": familia (evento o global), dónde
va el enlace, `METODO_SECTIONS` para una subpágina del método. Contrato: solo el párrafo de rutas de páginas.

- [ ] **Step 2: Scripts** — `check-api.sh`: añadir `check /metodo/promedio 200 html`, `/metodo/escanos`,
`/metodo/rating`, `check "/metodo/backtest?scope=es" 200 html`, `check /metodo/nope 404 html`, `check /es-zz 404
html`, `check "/elecciones?scope=es" 200 html` (ya está) y `check /modelo 301 none`; `smoke.sh` igual.
`bash -n` de los dos.

- [ ] **Step 3: Verificación final** — `python -m pytest -m "not integration" -q` (anotar el número);
`for f in $(git ls-files 'web/dist/js/**/*.js' 'web/dist/js/*.js'); do node --check $f; done`;
`S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8001 &` + `bash deploy/check-api.sh
http://127.0.0.1:8001` (todas OK o `n/a`); `curl -s http://127.0.0.1:8001/es | wc -c` (presupuesto); recorrido
en navegador de las siete rutas a 1280 y 360 px; parar uvicorn; `bash deploy/nginx-check.sh` (sin cambios en
nginx, pero confirma que sigue válido).

- [ ] **Step 4: Estado al cierre** — en la spec de la fase 7: commits, tests, decisiones E1-E12 y las
resoluciones tomadas, lo verificado y lo no verificado (Docker, producción, navegador real en móvil), menores
aplazados, pendientes de Luis (desplegar antes del 17-11; nombre y logo; publicar `analysis` y `event` de los
ámbitos que falten para que el histórico tenga datos; comprobar las fuentes en producción).

- [ ] **Step 5: Commit** — `git commit -m "Document phase 7: event model, cards, glossary, method subpages and the closing state"`
