# Web fase 7d (Histórico Errores de todos los ámbitos y ranking único) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir "Histórico Errores" en el histórico de todas las elecciones publicadas, con filtros encadenados (ámbito, elección, encuestadora, años, medida), cifras con comparación y un gráfico de evolución por elección; y hacer único el ranking de encuestadoras.

**Architecture:** El contexto de la página junta en el servidor las filas de `poll-errors` de todos los ámbitos con análisis y las elecciones publicadas; la isla las lleva al navegador y `errors.js` (puro, con paridad en node) filtra, agrega y calcula complementos y tendencia; `pages/elecciones.js` dibuja (histograma, diana, semanas, tabla y el nuevo `charts/trend.js`). El ranking de `/casas` lee siempre los ratings de `es`.

**Tech Stack:** Python 3 (Jinja2), JS ES modules sin build, ECharts 5.6 vendido, pytest + node por subproceso.

**Spec:** `docs/superpowers/specs/2026-10-14-web-fase-7d-historico-design.md`

## Global Constraints

- Textos en español; siglas de partido; "encuestadora/s"; docstrings numpy/JSDoc en inglés; CSP `default-src 'self'` (sin scripts ni estilos inline); controles solo-JS renderizados `hidden` y mostrados por el JS; DOM con `createElement`/`textContent`.
- Suite `python -m pytest -m "not integration" -q` en verde (730 al empezar), sin avisos nuevos; tests de paridad en node para todo lo que entre en `errors.js`.
- Nunca ejecutar `job.py`, `mtpy.run()` ni uvicorn sin `S3_BUCKET=` y `DB_HOST=` en el entorno.
- Commits en inglés sin prefijo `feat:`, con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; `git add` solo de los ficheros tocados.
- Hash del histórico: `#ambito=&eleccion=&casa=&desde=&hasta=&medida=`, claves solo cuando están fijadas; la elección se identifica por `{ámbito}:{fecha}` (`es:2023-07-23`) en el `select`, en el hash y en las agrupaciones; un hash antiguo `#eleccion=2023-07-23` se resuelve a la única elección de esa fecha en el ámbito elegido (o en `es`) y, si no es única, se ignora.
- `?scope=` sigue pasando por `common_context` (404 sin pronóstico, como hoy); el filtro de ámbito del histórico vive en el hash; `selected_scope` sale del valor crudo de la consulta.
- Las decisiones 1-6 de la spec ("Decisiones del controlador tras la revisión del constructor") obligan a todas las tareas.
- Las filas del histórico llevan solo: `scope`, `scope_name`, `year`, `event_date`, `event`, `pollster`, `days`, `error_avg`, `error_blocks`, `error_blocks_abs`, `bias` y las columnas de los dos bloques `vs` (`Derecha`, `Izquierda`).

## Review Focus

1. Un ámbito con análisis publicado pero sin elecciones publicadas (o al revés) no aporta filas ni aparece en el filtro (test en la tarea 2).
2. `?scope=` de un ámbito del catálogo sin análisis: el filtro arranca en "Todos" sin error (test en la tarea 2).
3. Años con `desde > hasta` en el hash: se intercambian; años fuera de los presentes: se ignoran (test en la tarea 3).
4. Una selección sin filas: cifras "—", contador "0 sondeos de 0 elecciones", gráficos vacíos sin error en consola, tabla por semanas con solo la cabecera (tests en las tareas 3 y 4).
5. Filas sin las dos columnas de bloque (un ámbito cuyo `poll-errors` no trae `Derecha`/`Izquierda`): fuera de la diana, dentro del resto (test en la tarea 3).

---

### Task 1: ranking único en `/casas`

**Files:**
- Modify: `mtpy/lib/pages/casas.py` (`casas_context`), `web/templates/casas.html` (tarjeta `ranking`, enlace CSV), `tests/test_pages.py` (`test_casas_lists_the_ranking_effects_and_herding` y vecinos)

**Interfaces:**
- Produces: `RATING_SCOPE = 'es'` en `casas.py`; `casas_context` llama `query_ratings(RATING_SCOPE)` para `ratings`, `ranking`, `pollster_ids` y `headline_text`; el enlace CSV del ranking es `api_url(RATING_SCOPE, 'ratings', fmt='csv', kind='data')` con etiqueta "Ratings de la próxima elección general"; clave nueva `ranking_event` (la `event_date` mayor de `ratings`, cadena ISO, o `None` sin filas) y `ranking_query` (`scope_query(RATING_SCOPE)`, para que las filas del ranking enlacen al perfil con `scope=es`). Efectos de encuestadora, herding y sus enlaces de perfil siguen con el ámbito elegido.
- Produces (plantilla): `card_start('ranking', 'Ranking de encuestadoras', 'La valoración de cada encuestadora de cara a la próxima elección general, calculada con sus sondeos de todas las elecciones anteriores, en todos los ámbitos; no depende del ámbito elegido arriba: cuanto más alta, más pesan sus sondeos en el promedio.', subtitle)` con `subtitle` = `'· elección general del ' ~ ranking_event|date` cuando hay fecha.

- [ ] **Step 1: Test que falla** (en `tests/test_pages.py`, con la fixture `houses`):

```python
def test_casas_ranking_is_the_national_one_whatever_the_scope(houses):
    national = html(call(houses, '/casas', query='scope=es')[2])
    madrid = html(call(houses, '/casas', query='scope=es-md')[2])
    table = lambda page: re.search(r'<table id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert table(national) == table(madrid)
    assert 'no depende del ámbito elegido arriba' in madrid and 'elección general del' in madrid
    assert 'href="/api/v1/ratings?scope=es&amp;format=csv"' in madrid and 'Ratings de la próxima elección general' in madrid
    ranking = table(madrid)
    assert 'href="/casas?scope=es&amp;id=' in ranking and 'href="/casas?scope=es-md&amp;id=' not in ranking
    effects = madrid[madrid.index('id="house-effects"'):madrid.index('id="herding"')]
    assert 'scope=es-md' in effects or 'id=' not in effects  # los efectos siguen con el ámbito elegido
```

(Si la fixture `houses` solo publica `es`, `scope=es-md` necesita el manifest con `es-md`: `write_bundle(fresh_app.fs, scope='es-md', run_id=…, manifest=True)` como en `test_promedios_hides_the_blocks_view_without_blocks`; el ranking sale de la base simulada por la fixture, que debe responder a `query_ratings('es')` también con `scope=es-md`.)

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_pages.py -q -k "national_one_whatever"` → FAIL.
- [ ] **Step 3: Implementar** contexto y plantilla; adaptar los tests que fijen la entradilla antigua ("de cara a la próxima elección del ámbito").
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Show one national ranking of pollsters whatever the scope"`.

---

### Task 2: datos y plantilla del histórico de todos los ámbitos

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py` (nuevas `history_scopes`, `history_events`, `history_rows`; `history_headline`; `elecciones_context`), `web/templates/elecciones.html` (sin `_scope_filter.html`; tarjeta `filters` con los cinco controles; columna Ámbito en `#events-table`; medida fuera del histograma; tarjeta de cierre), `tests/test_pages.py`
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces (`elecciones.py`): `HISTORY_COLUMNS = ('scope', 'scope_name', 'year', 'event_date', 'event', 'pollster', 'days', 'error_avg', 'error_blocks', 'error_blocks_abs', 'bias')`; `history_scopes() -> list[dict]` (`{'code', 'name', 'analysis': bool}`, catálogo en orden, solo con `site().events(code)` no vacío; `analysis` = `analysis_part(code, 'meta')` publicado); `history_events(scopes: list) -> list[dict]` (`published_events(code)` + `scope`, `scope_name`, `href`, juntas por `date` descendente y, a igual fecha, por orden del catálogo); `history_rows(scopes: list, events: list) -> list[dict]` (`error_rows(code, fechas del ámbito)` + `scope`, `scope_name`, `year = int(event_date[:4])`, con `HISTORY_COLUMNS` más las columnas de bloque de `poll-errors.blocks` del ámbito); `history_rows` solo recorre los ámbitos con `analysis`; `history_headline(events: list, errors: list | None) -> str`: "En las N elecciones analizadas (G generales y A autonómicas), el último sondeo de cada encuestadora se desvió de media X puntos en el margen entre bloques." (general = `scope == 'es'`; con una sola clase, "N elecciones analizadas (todas generales)"/"(todas autonómicas)"; sin filas, "N elecciones analizadas."). Contexto: `events`, `scopes`, `errors`, `error_blocks` (los bloques comunes a los ámbitos con análisis, en el orden del primero), `filter_events` (`{'key': 'scope:date', 'date', 'name', 'scope', 'scope_name'}`), `filter_pollsters`, `years`, `analysis_missing` (ningún ámbito con análisis), `headline_text`, `selected_scope` (el `scope` crudo de la consulta si está entre los `scopes` con análisis, si no `None`) e `initial` = `{'state', 'scopes', 'events': filter_events, 'errors', 'error_blocks', 'years', 'selected_scope'}`.
- Produces (plantilla): sin `{% include '_scope_filter.html' %}`; tarjeta `filters`: `<form id="errors-filter" class="series-controls" hidden>` con `select name="ambito"` ("Todos" + `scopes`, `selected` el `selected_scope`), `select name="eleccion"` ("Todas" + `filter_events` con `value` = `key` (`es:2023-07-23`), `data-scope` y etiqueta "{fecha} · {nombre} · {ámbito}"), `select name="casa"`, `select name="desde"` y `select name="hasta"` (los `years`, el primero y el último `selected`), `<fieldset id="errors-measure" class="seg">` (los radios de medida, que salen de la tarjeta del histograma) y `<span id="errors-count">`; `#events-table` con `<th scope="col">Ámbito</th>` tras "Elección" y `<tr data-scope="…" data-date="…" data-year="…">` por fila; un `<tfoot>`/pie `<p id="events-count">` "M elecciones"; cierre: enlace "Ranking Encuestadoras" a `/casas` (sin ámbito).

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`; fixture nueva `history_two_scopes` que parte de `history_site` y añade `write_bundle(fs, scope='es-md', …, manifest=True)`, `write_event(fs, scope='es-md', date='2021-05-04', model=False)` y `write_analysis(fs, scope='es-md')`, limpiando las cachés como los tests vecinos):

```python
def test_elecciones_merges_every_scope_with_analysis(history_two_scopes):
    body = call(history_two_scopes, '/elecciones')[2]
    page = html(body).replace('\n', ' ')
    assert 'id="scope-filter"' not in page and '<th scope="col">Ámbito</th>' in page
    assert re.search(r'<select name="ambito">\s*<option value="" selected>Todos</option>', page)
    for name in ('eleccion', 'casa', 'desde', 'hasta'):
        assert 'name="{}"'.format(name) in page
    assert 'id="errors-measure"' in page and page.index('id="errors-measure"') < page.index('id="histogram"')
    assert '3 elecciones analizadas (2 generales y 1 autonómicas)' in page
    data = initial(body)
    assert [s['code'] for s in data['scopes']] == ['es', 'es-md'] and data['selected_scope'] is None
    assert {row['scope'] for row in data['errors']} == {'es', 'es-md'} and all(isinstance(row['year'], int) for row in data['errors'])
    assert set(data['errors'][0]) == set(pages.HISTORY_COLUMNS) | set(data['error_blocks'])
    assert [e['scope'] for e in data['events']][:3] == ['es', 'es-md', 'es'] and data['years'] == sorted(data['years'])


def test_elecciones_scope_query_preselects_the_scope_or_falls_back(history_two_scopes):
    page = html(call(history_two_scopes, '/elecciones', query='scope=es-md')[2]).replace('\n', ' ')
    assert re.search(r'<option value="es-md" selected>Madrid</option>', page)
    assert initial(call(history_two_scopes, '/elecciones', query='scope=es-md')[2])['selected_scope'] == 'es-md'
    page = html(call(history_two_scopes, '/elecciones', query='scope=es-an')[2]).replace('\n', ' ')  # sin análisis
    assert re.search(r'<option value="" selected>Todos</option>', page)


def test_history_helpers_skip_scopes_without_events_or_analysis(history_two_scopes, fresh_app):
    # es-cb: análisis sin elecciones; es-ar: elecciones sin análisis → ninguno entra
    ...  # publicar con write_analysis / write_event y comprobar `pages.history_scopes()` == [es, es-md]
```

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py -q -k "merges_every_scope or preselects or history_helpers"` → FAIL.
- [ ] **Step 3: Implementar**; adaptar `test_elecciones_lists_the_published_events_newest_first`, `test_elecciones_only_counts_the_published_elections`, `test_elecciones_without_analysis_keeps_the_list`, `test_elecciones_history_context_filters_and_headline` y los que fijen `id="scope-filter"` en `/elecciones` o el texto antiguo del titular.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Merge the error history of every scope with analysis into one page"`.

---

### Task 3: `errors.js` con ámbito, años, complementos y tendencia

**Files:**
- Modify: `web/dist/js/errors.js`, `tests/test_js_errors.py`

**Interfaces:**
- Produces: `eventKey(row) -> string` (`${row.scope}:${row.event_date}`); `filterRows(rows, {scope = null, event = null, pollster = null, from = null, to = null})` (`scope` código o `null`; `event` una clave `scope:date` o `null`; `from`/`to` años enteros inclusivos o `null`; con `from > to` se intercambian); `yearsOf(rows) -> number[]` (años presentes, ascendente); `eventsOf(rows, measure) -> object[]` (una entrada por elección de las filas: `{event_date, event, scope, scope_name, n, mean}` con `mean` la media de `measure` sobre valores finitos, `null` sin ninguno; por fecha ascendente); `complements(rows, state, measure) -> object[]` (una entrada por filtro activo entre `scope`, `event`, `pollster` y años (`from`/`to` distintos de los extremos de `yearsOf(rows)`), en ese orden: `{key, label, n, mean}` con `label` ∈ "Otros ámbitos", "Otras elecciones", "Otras encuestadoras", "Otros años", sobre las filas que pasan todos los demás filtros y no ese; un complemento sin filas se omite); `trendPoints(rows, measure, {window = 5}) -> [iso_date, value][]` (por elección ascendente, la media de las últimas `window` elecciones ponderada por su `n`, incluida la propia); `restPoints(rows, pollster, measure) -> [iso_date, value][]` (la media de las demás encuestadoras en las elecciones donde la encuestadora tiene fila); `parseErrorsHash(hash, {scopes, events, pollsters, years})` → `{scope, event, pollster, from, to, measure}` (`events` son claves `scope:date`; un `eleccion=` sin ámbito (hash antiguo) se resuelve a la única clave con esa fecha en el ámbito del hash o en `es`, o a `null`; `from`/`to` `null` cuando son los extremos o no están entre `years`); `errorsHash(state)` (claves `ambito`, `eleccion`, `casa`, `desde`, `hasta`, `medida`); `weeksTable(rows, measure)` agrupa por `eventKey` y etiqueta cada fila "{event} · {scope_name}".

- [ ] **Step 1: Tests que fallan** (en `tests/test_js_errors.py`, con un fixture de 6 filas: `es` 2019 y 2023 (CIS y GAD3), `es-md` 2021 (CIS y GAD3), una fila de `es-md` sin `Derecha`/`Izquierda`):
  - `filterRows(rows, {scope: 'es'})` → 4; `{from: 2021, to: 2023}` → las de 2021 y 2023; `{from: 2023, to: 2019}` → igual que `{from: 2019, to: 2023}`; `{pollster: 'CIS', scope: 'es-md'}` → 1.
  - `yearsOf(rows) == [2019, 2021, 2023]`; `eventsOf(rows, 'error_avg')` → 3 entradas ascendentes con `n` y `mean` (comprobar una a mano).
  - `complements(rows, {pollster: 'CIS', from: 2021, to: 2023}, 'error_avg')` → dos entradas, "Otras encuestadoras" (GAD3 en 2021-2023) y "Otros años" (CIS en 2019), con `n` y `mean` a mano; sin filtros → `[]`.
  - `trendPoints` con 3 elecciones y `window: 2` → la segunda es la media ponderada de las dos primeras (a mano).
  - `restPoints(rows, 'CIS', 'error_avg')` → las medias de GAD3 en las elecciones del CIS.
  - Hash: `parseErrorsHash('#ambito=es-md&desde=2021&hasta=2023&casa=CIS', …)` → `{scope: 'es-md', from: 2021, to: 2023, pollster: 'CIS', event: null, measure: 'error_blocks_abs'}`; `'#eleccion=es:2023-07-23'` → `event: 'es:2023-07-23'`; `'#eleccion=2023-07-23&casa=GAD3&medida=error_avg'` (hash antiguo) → `event: 'es:2023-07-23'` cuando solo `es` tiene esa fecha, `null` si también la tiene `es-md`; `desde=2030` → `from: null`; `errorsHash` es la inversa y omite los extremos.
  - `filterRows(rows, {event: 'es-md:2023-07-23'})` no devuelve las filas de `es` de ese día.
- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_js_errors.py -q` → FAIL.
- [ ] **Step 3: Implementar** en `errors.js` (JSDoc en inglés).
- [ ] **Step 4: Tests en verde**: `python -m pytest tests/test_js_errors.py -q` → PASS; `node --check web/dist/js/errors.js`.
- [ ] **Step 5: Commit**: `git commit -m "Filter the error history by scope and years and compute complements and trends"`.

---

### Task 4: la página: filtros encadenados, cifras, tendencia, diana por ámbito, semanas y tabla

**Files:**
- Create: `web/dist/js/charts/trend.js`
- Modify: `web/dist/js/pages/elecciones.js` (`wireHistory`), `web/templates/elecciones.html` (tarjetas `figures` y `trend`; `sector` desaparece), `web/dist/css/site.css` (`.figures`), `tests/test_pages.py`
- Keep: `web/dist/js/charts/line.js` (lo usa el perfil de encuestadora)

**Interfaces:**
- Produces (`charts/trend.js`): `renderTrend(el, {series, trend, rest}, {formatter, colors, unit})`: `series` = `[{name (ámbito), points: [{date, value, n, label, href}]}]` dibujadas como scatter sobre eje de tiempo (tamaño `6 + 12 · n / nMax`, color `colors[name]`), `trend` = `[date, value]` (línea del acento, 2 px) y `rest` = `[date, value]` o `null` (línea gris discontinua "Resto de encuestadoras"); leyenda con los ámbitos y las líneas; tooltip con `label` ("{fecha} · {nombre} · {ámbito}"), "media X · N sondeos"; clic en un punto navega a `href`.
- Produces (`elecciones.js`, `wireHistory`): estado `{scope, event, pollster, from, to, measure}` desde el hash (`parseErrorsHash`), con `selected_scope` de la isla como valor inicial de `scope` cuando el hash no lo trae; al cambiar ámbito o años, las opciones de elección y encuestadora se recalculan (`select` rellenados con nodos del DOM; valores que desaparecen → `null`); `apply()` → `filterRows` → cifras (`#figures`: Sondeos, Elecciones, medida media, Sesgo del margen; y una línea por `complements`), tendencia (`eventsOf` por ámbito + `trendPoints` + `restPoints`), histograma, diana (`groupOf` = ámbito con `scope === null`, elección con ámbito; colores: acento para `es` y `EVENT_COLORS` para el resto / por elección), semanas (cap de 15 elecciones con la nota), tabla de elecciones (oculta las filas por `data-scope`/`data-date`/`data-year`; `#events-count`), contador; `writeHash()` tras cada cambio.
- Produces (plantilla): `card_start('figures', 'Cifras de la selección', 'Cuántos sondeos y elecciones entran con los filtros de arriba, cuánto se equivocaron de media y hacia qué lado; debajo, lo mismo para lo que queda fuera de cada filtro.')` con `<dl class="figures" id="figures">` (cuatro `div` con `dt`/`dd`) y `<ul id="complements">`; `card_start('trend', 'Evolución del error por elección', 'El error medio de los sondeos de cada elección a lo largo del tiempo, con la tendencia de las últimas cinco elecciones; con una encuestadora elegida, frente al resto.')` con `#trend-chart` y su "Cómo leerlo"; la tarjeta `sector` y su CSS desaparecen; textos "Cómo leerlo" de filtros y diana actualizados (ámbito, años, color por ámbito con "Todos").
- CSS: `.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: var(--space-3); }`, `.figures dd { font-family: var(--font-head); font-size: 1.6rem; margin: 0; }`, `.figures dt { color: var(--color-muted); font-size: 0.85rem; }`.

- [ ] **Step 1: Tests que fallan** (`tests/test_pages.py`): `/elecciones` tiene `id="figures"`, `id="complements"`, `id="trend-chart"`, no tiene `id="sector-chart"`, el orden `filters` < `figures` < `trend` < `histogram` < `bullseye` < `weeks` < `events`; y en `tests/test_js_errors.py` nada nuevo (el DOM se comprueba a mano).
- [ ] **Step 2: Ejecutar y ver que fallan** → FAIL.
- [ ] **Step 3: Implementar**; comprobar en headless Chrome (como en 7c: `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8011` más el proxy estático de `/dist/`) sobre un paquete local con dos ámbitos si el local no los tiene (escribir con `write_analysis`/`write_event` en un `files/` temporal o describir la limitación en el informe).
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS; `node --check` de los JS tocados.
- [ ] **Step 5: Commit**: `git commit -m "Chain the history filters and add the figures and the trend by election"`.

---

### Task 5: cierre

- [ ] **Step 1**: "Estado al cierre" de la spec 7d (commits, tests, pendientes de Luis: recorrido visual; publicar `analysis` en los ámbitos que falten (Andalucía) para que entren en el histórico); una línea en `deploy/README.md` (componentes: histórico de todos los ámbitos y ranking único) y en `docs/web/contrato.md` solo si algo cambia (no debería).
- [ ] **Step 2**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 3: Commit**: `git commit -m "Close web phase 7d in its spec"`.
