# Página del evento unificada («Encuestas») Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fundir las pestañas Promedios (`/{scope}`) y Escaños (`/{scope}/escanos`) en una sola página
«Encuestas» en `/{scope}`, con índice de secciones, sin duplicados y con 301 desde las URL antiguas.

**Architecture:** `promedios.html` absorbe las tarjetas de `escanos.html`, y `promedios_context` compone el
contexto de escaños con una función nueva, `seats_context`, en `lib/pages/escanos.py`. El JS de escaños pasa a
`pages/seats-wire.js`, que importa `promedios.js`. Un módulo nuevo, `url-state.js`, reparte el fragmento de la
URL entre los controles y el ancla de la sección. `/{scope}/escanos` pasa a ser una acción de `Redirect`.

**Tech Stack:** Python 3 (controladores propios sobre ASGI, Jinja2), JS en módulos ES sin build (ECharts 5.6
vendorizado), pytest. Los tests de JS se ejecutan con `node` desde pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-pagina-evento-unificada-design.md` (commit b67fcfa). Léelo antes
de empezar: el plan argumenta desde él.

## Global Constraints

- Orden de la página: KPIs + carrusel → `#promedio` → `#escanos` → `#evolucion` → `#territorio` → `#tabla` → cierre.
- Índice de la barra: «Promedio · Escaños · Evolución · Territorio · Tabla», anclas sin título visible.
- Nombre corto «Encuestas» (barra inferior del móvil); `<title>`: `Demoscopio · Encuestas, promedios y escaños · {event_bar.title}`. El prefijo `Demoscopio · ` lo pone `base.html` en todas las páginas. *Decisión del plan:* se mantiene.
- Barra inferior del móvil: 4 entradas, Inicio · Encuestas · Histórico · Ranking.
- Tabla de partidos: Partido · % Voto · vs {año} · Escaños · vs {año} · Rango 95% · P(escaño) · P(primero) · P(mayoría). Tabla de bloques: Bloque · % Voto · vs · Escaños · vs · Rango 95% · P(mayoría).
- Redacción pública: «sólo» con tilde; encuestadora o casa, nunca «empresa»; «actualización» para los runs, nunca «publicación»; ningún texto menciona «pestaña». Lo comprueba `assert_public_wording`.
- Identificadores de código en inglés neutro (los ids de ancla públicos sí van en español: `promedio`, `escanos`, `evolucion`, `territorio`, `tabla`).
- Los tests nunca usan `out=None`, ni nada que escriba en `backtest/results`.
- Luis commitea en paralelo en este mismo árbol. Antes de cada commit, mirar `git status` y `git log -3` y hacer `git add` sólo de los ficheros propios (nunca `-a`). Nunca reescribir sus commits.
- Suite: `python -m pytest -m "not integration" -q`. Hoy está en verde; tiene que acabar en verde.

## Review Focus

1. **Enlace viejo a la calculadora** `/es/escanos#coalition=PP,VOX`: debe acabar en `/es` con PP y VOX marcados y la página en Escaños. Lo cubren el test del `Location` sin `#` (tarea 4) y los de `mergeHash`/`sectionTarget` (tarea 1).
2. **Dos controles escribiendo el fragmento**: cambiar la ventana del promedio después de marcar una coalición no puede borrar `coalition=`, ni al revés, ni el ancla suelta. Tests de la tarea 1.
3. **Ámbito sin circunscripciones o sin sondeos** (runs autonómicos antiguos): la página carga, y el índice no ofrece «Territorio» ni «Tabla» si no existen esas secciones. Test en la tarea 2.
4. **Run fijado y modo** (`?mode=nowcast&run=…`): el 301, el formulario de circunscripción y los enlaces de las filas los conservan. Tests en las tareas 2 y 4.
5. **`region` inválida en la portada del ámbito**: `/es?region=abc` → 400 y `/es?region=999` → 404, como hoy en Escaños. Un `region` válido sin JS abre el detalle de esa circunscripción. Test en la tarea 2.

---

### Task 0: Medir el peso de partida (sin commit)

- [ ] **Step 1:** Con el árbol sin tocar, arranca la web local contra el bucket publicado, como en las comprobaciones de navegador anteriores (`.env` local). Guarda los bytes con gzip de `GET /es` y de `GET /es/escanos`, ambas `mode=forecast`, sobre el último run de producción de `es` (10.000 simulaciones). Ejemplo: `curl -s --compressed -o /dev/null -w '%{size_download}' -H 'Accept-Encoding: gzip' …`, o `gzip -c | wc -c` sobre el HTML. Si no hay acceso al bucket, usa el run local más reciente y anótalo.
- [ ] **Step 2:** Apunta las dos cifras y el `run_id` en `docs/superpowers/plans/2026-11-04-pagina-evento-unificada-ledger.md`. La tarea 5 las compara.

### Task 1: Estado compartido en el fragmento de la URL

**Files:**
- Create: `web/dist/js/url-state.js`
- Modify: `web/dist/js/blocks.js` (exporta `SERIES_KEYS`), `web/dist/js/seats.js` (exporta `COALITION_KEYS`), `web/dist/js/pages/series-wire.js:114-118` y `web/dist/js/pages/escanos.js:~130` (los dos `replaceState` pasan a usar `mergeHash`)
- Test: `tests/test_js_url_state.py` (mismo arnés `node` que `tests/test_js_blocks.py`)

**Interfaces:**
- Produces (`url-state.js`, funciones puras sin DOM):
  - `mergeHash(current: string, fragment: string, keys: string[]) -> string`: quita de `current` los pares cuya clave está en `keys` y añade al final los pares de `fragment`. Conserva en su orden el token suelto (sin `=`) y los demás pares. Devuelve `''` si no queda nada y, si queda algo, la cadena con `#`.
  - `withSection(current: string, id: string|null) -> string`: pone `id` como único token suelto, al principio, y conserva los pares.
  - `sectionTarget(search: string, hash: string, ids: string[]) -> string|null`: devuelve el `seccion` de la query si está en `ids`; si no, el primer token suelto del fragmento que esté en `ids`; si no, `null`.
  - `dropParam(search: string, name: string) -> string`: quita el parámetro de la query. Devuelve `''` si no queda ninguno, o la cadena con `?`.
- Produces: `SERIES_KEYS = ['ventana', 'vista', 'sel', 'mostrar']` (en `blocks.js`) y `COALITION_KEYS = ['coalition']` (en `seats.js`).

- [ ] **Step 1: Tests que fallan** en `tests/test_js_url_state.py`:
  - `mergeHash('#escanos&coalition=PP,VOX', '#ventana=1a&vista=partidos', SERIES_KEYS) == '#escanos&coalition=PP,VOX&ventana=1a&vista=partidos'`
  - `mergeHash('#escanos&coalition=PP&ventana=all', '#ventana=3m', SERIES_KEYS) == '#escanos&coalition=PP&ventana=3m'`
  - `mergeHash('#coalition=PP', '', COALITION_KEYS) == ''` y `mergeHash('', '#coalition=PP', COALITION_KEYS) == '#coalition=PP'`
  - `withSection('#ventana=1a', 'escanos') == '#escanos&ventana=1a'`; `withSection('#tabla&ventana=1a', 'escanos') == '#escanos&ventana=1a'`; `withSection('#tabla', null) == ''`
  - `sectionTarget('?mode=forecast&seccion=escanos', '#coalition=PP', IDS) == 'escanos'`; `sectionTarget('', '#territorio&ventana=1a', IDS) == 'territorio'`; `sectionTarget('?seccion=nope', '#coalition=PP', IDS) is None`
  - `dropParam('?mode=forecast&seccion=escanos', 'seccion') == '?mode=forecast'`; `dropParam('?seccion=escanos', 'seccion') == ''`
  - Los lectores existentes siguen sirviendo: `parseSeriesHash('#escanos&ventana=1a', …).window == '1a'` y `parseCoalition('#escanos&coalition=PP,VOX', ['PP','PSOE','VOX']) == ['PP','VOX']`
- [ ] **Step 2:** `python -m pytest tests/test_js_url_state.py -q` → FAIL (el módulo no existe).
- [ ] **Step 3:** Implementa `url-state.js` y exporta las dos constantes de claves. Los dos `replaceState` pasan a escribir `pathname + search + mergeHash(window.location.hash, <su fragmento>, <sus claves>)`.
- [ ] **Step 4:** `python -m pytest tests/test_js_url_state.py tests/test_js_blocks.py tests/test_js_seats.py tests/test_web_routes.py -q` → PASS.
- [ ] **Step 5:** Commit: `web: estado compartido del fragmento de la URL (url-state.js) para los controles de la página del evento`.

### Task 2: Contexto y plantilla de la página unida

**Files:**
- Modify: `mtpy/lib/pages/escanos.py` (nueva `seats_context`; `escanos_context` se queda como envoltorio fino hasta la tarea 4), `mtpy/lib/pages/promedios.py` (`result_rows`, `block_result_rows`, `promedios_context`), `mtpy/lib/pages/common.py` (`EVENT_SECTIONS`, `event_info`), `mtpy/controllers/Promedios.py` (pasa `region`), `web/templates/promedios.html`, `web/templates/_event_bar.html`, `mtpy/lib/pages/__init__.py` (exporta `seats_context`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces: `seats_context(context: dict, region=None) -> dict` en `escanos.py`. Recibe el contexto común ya resuelto de un evento y devuelve las claves que añade la parte de escaños: `summary`, `calculator`, `csv_links`, `district_table`, `regions`, `territory_levels`, `region`, `region_rows`, `inherited`, y `initial_extra` = `{'dist', 'fan', 'districts', 'scenario'}`. Lanza los mismos `HttpError` que hoy por `region` (400 inválida, 404 desconocida).
- Produces: `promedios_context(state: dict, region=None) -> dict`. `initial` tiene exactamente estas claves, en este orden: `['state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'dist_href', 'fan', 'districts', 'scenario', 'runs', 'cards', 'deltas', 'catalog', 'inks']`. `inks` es la unión de las dos de hoy. **`dist` no va en el JSON inicial** (decisión de Luis, 2026-11-04, por el peso). `dist_href` es la URL JSON de la parte `dist` del run y del modo mostrados: `api_url(scope, 'dist', mode=mode, run=run)`, la misma ruta que ya usa `csv_links`, pero sin `fmt`. `seats_context` devuelve `dist_href` en `initial_extra`, en lugar de `dist`. El contexto del servidor sigue leyendo `dist` para los nombres de la calculadora (`calculator`).
- Produces: cada fila de `result_rows` añade `p_seats`, `p_first` y `p_majority`; cada fila de `block_result_rows` añade `p_majority` (`None` en Otros). `seat_rows` y `block_rows` de `escanos.py` dejan de usarse en la plantilla.
- Produces: `EVENT_SECTIONS = (('promedio', 'Promedio'), ('escanos', 'Escaños'), ('evolucion', 'Evolución'), ('territorio', 'Territorio'), ('tabla', 'Tabla'))`. `event_info(...)['sections']` es una lista de `{'id', 'label', 'href': '#' + id}`. `promedios_context` quita `territorio` si no hay `district_table` y `tabla` si `table_total` es 0. La clave `tabs` desaparece; `EVENT_TABS` sigue existiendo hasta la tarea 4 para que `/es/escanos` no se rompa.

**Estructura de `promedios.html`** (ids nuevos en negrita; el resto se reutiliza tal cual):
- KPIs y carrusel, fuera de las secciones.
- **`<section id="promedio">`**: tarjeta `series`.
- **`<section id="escanos">`**:
  - `hemi-grid` con `hemicycle-card`, que lleva el único selector `#results-view` (`rview`), y la tarjeta **`seats`**. `seats` se titula «Resultados» y su descripción es «Voto y escaños previstos de cada partido o bloque, con su rango del 95% y sus probabilidades» + « y cuánto cambian frente a {año}» si hay elección previa. Dentro lleva `#seats-parties` → `table#parties-table` y `#seats-blocks` (hidden) → `table#blocks-table`, con las columnas de Global Constraints, más `#ridge-toggle` y `#ridge-chart`.
  - Tarjeta `coalition`.
  - Desaparecen `#seats-bar`, `#seats-download`, `parties-card`, `blocks-card` y `blocks-result-table`.
- **`<section id="evolucion">`**: tarjeta **`evolution`**, que sustituye a `evolution-card`, con un fieldset nuevo **`#evolution-metric`** (radios `evmetric`: `pct` «Voto», marcado, y `seats` «Escaños») antes de `#evolution-view`. Después, la tarjeta `pollster`.
- **`<section id="territorio">`** (sólo con `district_table`): tarjeta `districts`. El formulario lleva `action="/{scope}"` y los ocultos `mode`, `run` (si está fijado) y `seccion=territorio`. Los enlaces de las filas apuntan a `/{scope}{query}&region=N&seccion=territorio`.
- **`<section id="tabla">`**: tarjetas `polls`, `fan` y `downloads`.
- Un solo `m.closing`: título «¿Cómo promediamos las encuestas y simulamos los escaños?», href `method_href('promedio')`. Sus enlaces: `method_href('escanos')` «¿Cómo simulamos los escaños?», `/elecciones{scope_query}` y `/casas{scope_query}`.
- `{% block title %}`: `Encuestas, promedios y escaños · {{ event_bar.title }}`. La descripción se escribe en la misma línea (ver la spec).
- Un solo script: `/dist/js/pages/promedios.js`. El cableado de escaños llega en la tarea 3; hasta entonces, las tarjetas de escaños se ven sin JS.
- Los «Cómo leerlo» se reescriben sin «pestaña». Las tarjetas `seats` y `evolution` juntan el texto de sus dos antecesoras y explican las dos métricas y las P(*). En `downloads`, la frase de los sondeos pasa a «Los sondeos se descargan en CSV desde la tabla «Todas las encuestas».».
- `_event_bar.html`: un `<nav class="seg lg event-index" aria-label="Secciones de la página">` con un `<a href="#id">` por cada entrada de `sections`, sin `aria-current`. Las secciones llevan `scroll-margin-top` igual al alto de la barra fija (CSS en `web/dist/css`).

- [ ] **Step 1: Tests que fallan** en `tests/test_pages.py` (nuevos; reutiliza `call`, `html`, `initial`, `write_districts`):
  - `test_event_page_merges_promedios_and_escanos_in_order`: en `/es?mode=forecast`, los índices de `id="promedio"`, `id="escanos"`, `id="evolucion"`, `id="territorio"` y `id="tabla"` van en orden. Dentro de cada sección están las tarjetas que le tocan, en el orden de arriba. `id="seats-bar"`, `id="blocks-result-table"`, `id="parties-card"` y la palabra `pestaña` no aparecen.
  - `test_event_page_title_and_index`: `<title>Demoscopio · Encuestas, promedios y escaños · Generales` está en la página. La barra lleva los cinco `href="#…"` con sus etiquetas y ya no lleva `href="/es/escanos`.
  - `test_event_page_results_table_columns`: las cabeceras de `parties-table` son exactamente las 9 de Global Constraints, y las de `blocks-table` las 7.
  - `test_event_page_initial_keys`: `list(initial(body)) ==` la lista de Interfaces, `'dist' not in data`, y `data['dist_href'] == '/api/v1/forecast/es/forecast/dist?run=' + RUN` (o la forma exacta que produzca `api_url`; el test la fija). Las píldoras de la calculadora siguen saliendo del servidor (`name="coalition" value="PP" checked`).
  - `test_event_page_index_skips_missing_sections`: sin circunscripciones (el escenario de `test_escanos_without_districts_hides_the_section`), no salen ni `id="territorio"` ni `href="#territorio"`.
  - `test_event_page_region`: `/es?mode=forecast&region=8` marca `<option value="8" selected>`; el formulario lleva `name="seccion" value="territorio"` y el `mode`; con `run=` fijado, también `run`. `region=abc` → 400 y `region=999` → 404 (parametrizado como `test_escanos_region_errors`).
  - `test_event_page_texts`: el closing nombra los dos métodos. `evolution` explica las dos métricas («hueco», «no es una serie de sondeos»). `assert_public_wording` pasa sobre la página.
- [ ] **Step 2:** `python -m pytest tests/test_pages.py -q -k event_page` → FAIL.
- [ ] **Step 3:** Implementa `seats_context`, las columnas P(*) en las filas, `EVENT_SECTIONS` y `sections`, `promedios_context(state, region)`, el controlador y la plantilla.
- [ ] **Step 4:** `python -m pytest tests/test_pages.py -q` → los tests nuevos en verde. Los de Promedios que fijan lo que se ha quitado o renombrado (el título «Promedios», `parties-card`, `blocks-result-table`, `tabs`) se actualizan al contrato nuevo en este mismo paso. `/es/escanos` comparte `_event_bar.html`, así que sus tests que fijan la pestaña activa (`aria-current` en `href="/es/escanos…"`) se borran ya, porque esa página desaparece en la tarea 4. El resto de sus tests siguen en verde. Después, `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5:** Commit: `web: la página del evento junta Promedios y Escaños por secciones (plantilla y contexto)`.

### Task 3: El JS de la página unida

**Files:**
- Create: `web/dist/js/pages/seats-wire.js` (sale de `escanos.js`: `partyBar`, `wireCoalition`, `wireTerritory`, la parte de distribuciones de `wireSeats`, el abanico)
- Modify: `web/dist/js/pages/promedios.js`, `web/dist/js/pages/evolution-wire.js`, `web/dist/js/territory.js`, `web/dist/js/pages/territory-dom.js`
- Test: `tests/test_js_evolution.py`, `tests/test_js_territory.py`, `tests/test_web_routes.py` (imports y aria-labels)

**Interfaces:**
- Consumes: `mergeHash`, `withSection`, `sectionTarget` y `dropParam` (tarea 1); los ids de la tarea 2.
- Produces: `wireSeatsPage(initial: object, catalog: Catalog, view: () => 'partidos'|'bloques') -> {redraw: (view) => void}` en `seats-wire.js`. Dibuja la calculadora, el territorio, el detalle de circunscripción, el abanico y las distribuciones. `redraw` lo llama el selector `#results-view` de `promedios.js`, que ya cambia el hemiciclo y ahora también muestra `#seats-parties` o `#seats-blocks` y redibuja las distribuciones si están abiertas. **`dist` llega aparte** (decisión de Luis, 2026-11-04): `wireSeatsPage` dibuja en el acto lo que no
  necesita las simulaciones (territorio, detalle de circunscripción, abanico) y, después del primer dibujo, pide
  `initial.dist_href` con `fetch`. Mientras tanto, la calculadora enseña «Cargando las simulaciones…» en
  `#coalition-result`, y «Ver las distribuciones» y los botones de descarga de esas dos tarjetas esperan. Si la
  petición falla, el mismo sitio dice «No se han podido cargar las simulaciones.», y el resto de la página sigue
  funcionando. Se pide una sola vez por carga de página.
- Produces: `loadDist(href: string, fetchImpl = fetch) -> Promise<object>` en `seats-wire.js` (o en `seats.js` si
  así se puede probar sin DOM): rechaza la promesa si la respuesta no es OK. La prueba usa un `fetchImpl` falso.
- Produces: `wireEvolution(initial, catalog, {parties, metrics})`, con `metrics = {pct: {field: 'pct', formatter: fmtPct, total: 100}, seats: {field: 'seats', formatter: fmtInt, total: nSeats}}`. Si hay `#evolution-metric` y más de una métrica, el selector cambia gráfico, tabla de cambios y nombre del fichero de descarga (`…-evolucion-…` para voto, `…-evolucion-escanos-…` para escaños), y vuelve a la selección por defecto. Con una sola métrica funciona como hoy.
- Produces: `regionRows(districts: object, regionId: number) -> object[]` en `territory.js`, espejo de `region_rows` de Python (quita filas sin `pct`; ordena por `seats` y luego `pct`, ambos descendentes, de forma estable). También `renderRegion(table: HTMLTableElement, rows: object[], region: {name, seats}, catalog)` en `territory-dom.js`, que reescribe el caption («{name} · {seats} escaños») y el tbody con los formatos de la plantilla (`fmtPct`, `fmtRange`, `fmtProb`).

- [ ] **Step 1: Tests que fallan:**
  - `tests/test_js_territory.py::test_region_rows_mirror_python`: con un `districts` de tres filas en la región 8 (una con `pct: null`), devuelve las dos con voto, en el orden esperado. Añade un caso de empate en `seats` que se desempata por `pct`.
  - `test_load_dist` (en el fichero de tests del módulo donde viva `loadDist`): con un `fetchImpl` falso que
    devuelve `{ok: true, json: () => ({parties: ['PP']})}`, la promesa resuelve ese objeto. Con `{ok: false,
    status: 503}`, la promesa se rechaza.
  - `tests/test_js_evolution.py::test_wire_evolution_switches_metric`: comprueba con el arnés de DOM mínimo que ya usa el fichero o, si no lo hay, con la función pura que elige `field`/`total`/`formatter` por métrica. Con `metrics` de dos entradas, al cambiar a `seats` las vistas usan `field: 'seats'` y `total: nSeats`.
- [ ] **Step 2:** `python -m pytest tests/test_js_territory.py tests/test_js_evolution.py -q` → FAIL.
- [ ] **Step 3:** Implementa `regionRows` + `renderRegion` y las métricas de `wireEvolution`. Después, `seats-wire.js` y `promedios.js`:
  - `promedios.js` llama a `wireSeatsPage` y a `wireEvolution` con las dos métricas. Los `parties` son los de `initial.series.parties`, como hoy.
  - Al cargar, `sectionTarget(location.search, location.hash, ids)` hace `scrollIntoView` de la sección y quita `seccion` con `replaceState(pathname + dropParam(search, 'seccion') + hash)`.
  - Los enlaces del índice se interceptan: `scrollIntoView` + `replaceState` con `withSection`. Sin JS siguen siendo anclas normales.
  - El formulario de circunscripción ya no se envía con JS. Al cambiar el select, o al pulsar el enlace de una fila de provincia, se llama a `renderRegion`, se escribe `region=N` en la query con `replaceState` (conservando el fragmento) y se hace scroll a `#region-table`.
  - Se quitan `wireForm('district-form')` y el dibujo de `#seats-bar`.
- [ ] **Step 4:** `python -m pytest tests/test_js_territory.py tests/test_js_evolution.py tests/test_web_routes.py -q` → PASS (los imports resuelven y los charts tienen aria-label).
- [ ] **Step 5:** Comprobación en el navegador (headless Chrome contra la app local, la técnica de ASGI + CDP ya usada en este repo), en escritorio y a 390 px:
  - los cinco gráficos de escaños dibujan;
  - el selector Partidos/Bloques mueve hemiciclo, tabla y distribuciones;
  - el selector Voto/Escaños de la evolución funciona;
  - `#coalition=PP,VOX` + cambiar la ventana del promedio deja las dos cosas en la URL;
  - elegir una circunscripción no recarga la página;
  - consola sin errores.
  Haz capturas y anota en el ledger lo que se haya resuelto en la maquetación de `hemi-grid`.
- [ ] **Step 6:** Commit: `web: el JS de escaños (seats-wire.js) se carga en la página del evento, con la evolución Voto/Escaños y el detalle de circunscripción sin recarga`.

### Task 4: Navegación, redirecciones y retirada de Escaños

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`ROUTES`, `TABBAR`, `TABBAR_ACTIVE`, `tabbar`, `common_context`, `event_info`; borra `EVENT_TABS`), `mtpy/controllers/Redirect.py`, `mtpy/lib/pages/inicio.py:117`, `web/templates/index.html` (botón «Escaños»), `web/templates/metodo_escanos.html:764`, `deploy/check-api.sh:131-132`, `deploy/smoke.sh:10,67`, `deploy/README.md:79,98,117`
- Delete: `mtpy/controllers/Escanos.py`, `web/templates/escanos.html`, `web/dist/js/pages/escanos.js`, `escanos_context`, y `seat_rows`/`block_rows` si ya no los usa nadie
- Test: `tests/test_pages.py`, `tests/test_pages_ejemplo.py`, `tests/test_web_routes.py`

**Interfaces:**
- Produces: la ruta `('/{scope}/escanos', 'redirect', 'seats', ['GET'])` → `Redirect.seats_action(scope)`. Valida el ámbito con `resolve_event_scope` (404 si no está publicado) y responde 301 a `/{scope}` + la query original + `seccion=territorio` si trae `region`, o `seccion=escanos` si no. El `Location` no lleva nunca `#`. `escanos_action` (el `/escanos` antiguo) va directo al mismo destino.
- Produces: `TABBAR = inicio · {'key': 'event', 'label': 'Encuestas', 'icon': 'line'} · historico · ranking`; `tabbar(scope, active, event, query)`; `common_context(state, active, event=False)`; `event_info(scope, meta, state, today=None, scopes=None)`. Se quita `active_suffix` de las tres.
- Produces: `href_seats` de Inicio = `'/' + code + '#escanos'`; Método enlaza `/es#escanos`.

- [ ] **Step 1: Tests que fallan:**
  - En `test_old_urls_redirect_permanently`, la fila de `/escanos` pasa a `('/escanos', 'scope=es&region=28', '/es?region=28&seccion=territorio')`. Se añaden estos casos:
    - `('/es/escanos', 'mode=forecast', '/es?mode=forecast&seccion=escanos')`
    - `('/es/escanos', '', '/es?seccion=escanos')`
    - `('/es/escanos', 'mode=nowcast&run=' + RUN + '&region=8', '/es?mode=nowcast&run=' + RUN + '&region=8&seccion=territorio')`
    - y se comprueba `'#' not in headers['location']`.
  - `/es-zz/escanos` → 404 HTML.
  - `test_tabbar_has_four_entries`: en `/es`, la barra inferior tiene 4 enlaces (Inicio, Encuestas, Histórico, Ranking) y Encuestas lleva `aria-current="page"`. En `/casas`, Ranking es la activa.
  - Inicio: el botón del hero apunta a `/es#escanos`. Método escaños: `href="/es#escanos"`.
- [ ] **Step 2:** `python -m pytest tests/test_pages.py -q -k "redirect or tabbar or inicio"` → FAIL.
- [ ] **Step 3:** Implementa las rutas, `Redirect`, la barra inferior y los enlaces; borra Escaños.
- [ ] **Step 4:** Migra los tests que hoy piden `/es/escanos`: 83 apariciones en `test_pages.py`, 5 en `test_pages_ejemplo.py` y 5 en `test_web_routes.py`.
  - Los que fijan contenido que se conserva piden ahora `/es` y buscan por los ids nuevos: `seats`, `parties-table`, `blocks-table`, `coalition`, `evolution`, `districts`, `fan` y `downloads`.
  - Los que fijan lo que se quita a propósito (`seats-bar`, `seats-download`, la pestaña activa, `escanos.js`, el título «Escaños») se borran o se reescriben contra el contrato nuevo.
  - Ningún test puede quedarse comparando con el contrato viejo.
- [ ] **Step 5:** `python -m pytest -m "not integration" -q` → PASS. Actualiza `deploy/check-api.sh` (`/es/escanos?mode=forecast` → 301 y `/es?region=abc` → 400), `deploy/smoke.sh` (quita `/es/escanos` de las páginas estáticas) y `deploy/README.md` (rutas y controladores).
- [ ] **Step 6:** Commit: `web: /{scope}/escanos redirige a la página del evento; barra inferior de 4 entradas («Encuestas»)`.

### Task 5: Peso, revisión final y cierre de la spec

- [ ] **Step 1:** Repite la medición de la tarea 0 sobre `/es?mode=forecast` con el mismo run y apúntala en el ledger. **Si el gzip pasa del doble que la Promedios de partida, para y cuéntaselo a Luis** con las dos cifras y la salida prevista (pedir `dist` a la API al abrir la calculadora o las distribuciones). No la implementes sin su visto bueno.
- [ ] **Step 2:** Haz una pasada en el navegador por `/es` en los dos modos y por un ámbito autonómico (`/es-md`), en escritorio y a 390 px:
  - el índice salta bien y no queda tapado por la barra fija;
  - `/es/escanos#coalition=PP,VOX` acaba en Escaños con PP y VOX marcados;
  - sin JS, el formulario de circunscripción funciona.
- [ ] **Step 3:** Añade a la spec una sección «Cierre» con los commits, el peso antes/después y lo que se haya decidido en la maquetación. Commit: `docs: cierre de la página del evento unificada`.
