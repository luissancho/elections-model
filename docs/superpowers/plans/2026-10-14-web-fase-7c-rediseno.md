# Web fase 7c (rediseño, bloque central, tablas de encuestas y de territorio) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar a la web el estilo gris de la maqueta aprobada con el contenedor fluido, simplificar el bloque central de Promedios a tres tarjetas, y sustituir las tablas de sondeos y de circunscripciones por dos tablas filtrables al estilo de Objetivo176.

**Architecture:** Mismo patrón de siempre: contexto en `mtpy/lib/pages/*.py`, plantillas Jinja2 en `web/templates/`, datos en la isla `#initial-data`, y módulos JS que solo dibujan (`web/dist/js/pages/*.js`) apoyados en módulos puros (`web/dist/js/*.js`) con tests de paridad en node (`tests/test_js_*.py`). Las tablas nuevas salen del servidor con sus primeras filas y el navegador las rehace con filtros, orden y paginación. La publicación solo añade `community` y `votes` a cada circunscripción.

**Tech Stack:** Python 3 (Jinja2, pandas), JS ES modules sin build, ECharts 5.6 vendido, pytest + node por subproceso.

**Spec:** `docs/superpowers/specs/2026-10-14-web-fase-7c-rediseno-design.md`

## Global Constraints

- Textos de usuario en español; "encuestadora/s" y "medio/s"; siglas de partido siempre; docstrings numpy en inglés.
- CSP `default-src 'self'`: nada de `script` ni `style` inline; los controles que solo funcionan con JS salen del servidor con `hidden` y el JS los muestra.
- `python -m pytest -m "not integration" -q` en verde (697 al empezar) y sin avisos nuevos; los tests de node se saltan sin `node` (patrón de `tests/test_js_polls.py`).
- Nunca ejecutar `job.py`, `mtpy.run()` ni uvicorn sin `S3_BUCKET=` y `DB_HOST=` en el entorno (el `.env` apunta al bucket y a la base reales); los tests unitarios no lo necesitan.
- Commits en inglés, sin prefijo `feat:`, con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. El árbol tiene cambios sin commit de Luis (marca, cabecera, `site.css`, `common.py`…): no revertirlos; se añaden al commit los ficheros que la tarea toca, con `git add <rutas>`, aunque lleven cambios suyos (Luis lo aceptó).
- Hash del promedio `#ventana=&vista=&sel=` sin cambiar de formato (`web/dist/js/blocks.js`).
- Tokens de la maqueta (spec D1): `--color-bg: #f4f5f8`, `--color-card: #fff`, `--color-text: #1a1c22`, `--color-muted: #5a5f72`, `--color-faint: #6b7186`, `--color-line: rgba(0,0,0,.08)`, `--color-line-strong: rgba(0,0,0,.15)`, `--color-soft: #eceef2`, `--color-accent: #1d4f86`, `--radius: 12px`, `--shadow: 0 2px 8px rgba(0,0,0,.06)`.

## Review Focus

1. Run antiguo sin `community`/`votes` en `districts.regions` (todo lo publicado hasta hoy): la tabla por territorio se dibuja en nivel Provincia y el conmutador CCAA no aparece ni falla (test en la tarea 6).
2. Sondeo con `sample_size`, `sponsor` o el % de un partido a `null`: la fila muestra "—", la media lo ignora y el líder de la fila se calcula solo con los valores presentes (tests en la tarea 5).
3. Ámbito de una sola circunscripción (`es-md`): la tabla por territorio es una fila más TOTAL, sin CCAA, y `summary` sin bloques no rompe las tarjetas de resultados (tests en las tareas 3 y 6).
4. Partido sin escaños en ninguna circunscripción: en modo Escaños no tiene columna, en % Voto sí (test en la tarea 6).
5. Hash con `sel` de otra vista (nombres de bloque con `vista=partidos`): la leyenda ignora los nombres desconocidos y el gráfico se dibuja (test en la tarea 4).

---

### Task 1: `community` y `votes` en `districts.regions` de la publicación

**Files:**
- Modify: `mtpy/lib/data.py` (tras `get_districts`), `mtpy/lib/publish/forecast.py:120-145` (`export_districts`) y la llamada a `export_districts` dentro de `publish_forecast` (busca `export_districts(` en el mismo fichero), `docs/web/contrato.md:195-203` (apartado `districts`), `.gitignore` (añadir `.superpowers/`)
- Test: `tests/test_publish_unit.py:70-80`

**Interfaces:**
- Produces: `data.get_district_communities(scope: str) -> dict[int, Optional[dict]]`: `{region_id: {'code': int, 'name': str} | None}` para cada circunscripción de `get_districts(scope)`; `None` cuando `reg_code` coincide con el `ine_code` del propio ámbito en `get_scopes()` (los autonómicos); el nombre sale de `es-regions.csv` (`App.get_().data.read_csv('es-regions.csv')`, columnas `code, region, population`) por `reg_code == code`.
- Produces: `export_districts(sim, communities: Optional[dict] = None)`: cada `regions[i]` lleva además `'community': communities.get(id) if communities else None` y `'votes': int(sim.reg_totals.loc[id, 'votes'])`. `publish_forecast` le pasa `communities=get_district_communities(sim.scope)` (import perezoso dentro de la función si `mtpy.lib.data` crea un ciclo de imports).

- [ ] **Step 1: Tests que fallan** (en `tests/test_publish_unit.py`, junto a `test_export_districts_excludes_the_total_region`):

```python
def test_export_districts_carries_the_community_and_the_votes():
    communities = {28: {'code': 13, 'name': 'Madrid'}, 8: {'code': 9, 'name': 'Cataluña'}}
    data = valid('districts', publish.export_districts(synthetic_simulator(), communities=communities)[0])
    assert data['regions'][0] == {'id': 28, 'name': 'Madrid', 'seats': 6, 'community': {'code': 13, 'name': 'Madrid'}, 'votes': 600}
    assert data['regions'][1]['community'] == {'code': 9, 'name': 'Cataluña'} and data['regions'][1]['votes'] == 400
    bare = valid('districts', publish.export_districts(synthetic_simulator())[0])
    assert bare['regions'][0]['community'] is None and bare['regions'][0]['votes'] == 600


def test_district_communities_come_from_the_catalogue_and_are_null_in_their_own_scope(fresh_app):
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    national = data.get_district_communities('es')
    assert len(national) == 52 and national[1] == {'code': 16, 'name': 'País Vasco'} and national[28] == {'code': 13, 'name': 'Madrid'}
    assert data.get_district_communities('es-pv') == {1: None, 20: None, 48: None}
```

(`fresh_app`, `FileSystem` y `ROOT` como en `tests/test_pages.py::test_without_a_published_catalogue_scope_the_home_is_a_503_page`; `data` es `mtpy.lib.data`.)

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_publish_unit.py -q -k "community or communities"` → FAIL (`TypeError`/`AttributeError`).
- [ ] **Step 3: Implementar** `get_district_communities` en `mtpy/lib/data.py` y el parámetro `communities` de `export_districts`; pasar las comunidades desde `publish_forecast`; enmienda en `docs/web/contrato.md` (`regions` pasa a `{id, name, seats, community, votes}`, con una línea de "enmienda 2026-10-14 (fase 7c)" que diga que los runs anteriores no los tienen); `.superpowers/` en `.gitignore`.
- [ ] **Step 4: Tests en verde**: `python -m pytest tests/test_publish_unit.py -q` → PASS; `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git add mtpy/lib/data.py mtpy/lib/publish/forecast.py docs/web/contrato.md tests/test_publish_unit.py .gitignore && git commit -m "Publish the community and the votes of each district"`

---

### Task 2: pie, barra del evento sin la frase de modo y Promedios sin titular

**Files:**
- Modify: `web/templates/_footer.html`, `web/templates/_event_bar.html` (quitar `<p class="mode-text">`), `web/templates/promedios.html:12-14` (quitar `m.headline`), `mtpy/lib/pages/common.py` (`mode_text` y su clave en `common_context`, líneas ~525 y ~779), `mtpy/lib/pages/__init__.py:21` (export de `mode_text`), `mtpy/lib/pages/promedios.py` (`headline_text`, `DEFINITIVE_PREFIX` y la clave `headline_text` del contexto; comprueba con `grep -rn headline_text mtpy web tests` que solo Promedios lo usa: el de Escaños es `escanos_headline_text` y se queda), `web/dist/css/site.css` (`.mode-text` y el área `mode` de `#event-bar`, también en el bloque móvil)
- Test: `tests/test_pages.py` (`test_mode_text_explains_how_to_read_the_mode_shown`, `test_promedios_headline_sentence_uses_the_mode`, `test_headline_text_names_the_leader_by_its_short_name_and_the_likeliest_majority`, `test_promedios_frozen_latest_forecast_is_definitive`: borrar los que solo comprueban el titular o `mode_text`; si el último comprueba también el título de la tarjeta "Pronóstico definitivo", conservar esa parte)

**Interfaces:**
- Produces: pie `<p>Publicación: {{ meta.run_id }} ({{ meta.run_at|date }}) · Último sondeo: {{ meta.date_last|date }}{% if frozen and freeze.since %} · congelado desde {{ freeze.since|date }}{% endif %}</p>`; `common_context` sin `mode_text`; `promedios_context` sin `headline_text`.

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`):

```python
def test_footer_names_the_publication_and_the_last_poll(api):
    page = html(call(api, '/es')[2])
    assert 'Publicación: {} ('.format(RUN) in page and '· Último sondeo: ' in page
    assert 'commit' not in page and 'promedio leído' not in page and '(sucio)' not in page


def test_event_bar_and_promedios_have_no_mode_sentence_nor_headline(api):
    page = html(call(api, '/es')[2])
    assert 'mode-text' not in page and 'Si las elecciones fueran hoy' not in page
    assert 'class="headline"' not in page and 'Si hoy hubiera elecciones, ' not in page
    assert 'class="headline"' in html(call(api, '/es/escanos')[2])  # el titular de Escaños se queda
```

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py -q -k "footer_names or no_mode_sentence"` → FAIL.
- [ ] **Step 3: Implementar** las plantillas, retirar `mode_text` y `headline_text` con sus tests y el CSS de `.mode-text`.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Name the publication in the footer and drop the mode sentence and the headline"` (con `git add` de los ficheros tocados).

---

### Task 3: bloque central de Promedios en tres tarjetas

**Files:**
- Modify: `web/templates/promedios.html:59-143` (tarjetas `headline`, `vote`, `seats`, `hemicycle-card`, `majority-card`), `mtpy/lib/pages/promedios.py` (`majority_rows` se retira; nuevas `result_rows`, `block_result_rows`; contexto), `web/dist/js/pages/promedios.js:20-61` (`paintResults`), `web/dist/css/site.css` (`.hemi-grid`; retirar `.majority-*`)
- Delete: `web/dist/js/charts/bars.js` si `grep -rn "charts/bars" web/dist/js` no da otro usuario
- Test: `tests/test_pages.py` (`test_promedios_orders_the_story_and_carries_the_controls`, y los que nombren `vote-chart`, `seats-chart`, `majority` o `headline-table`)

**Interfaces:**
- Produces (`mtpy/lib/pages/promedios.py`): `BLOCK_ORDER = ('Derecha', 'Izquierda', 'Regionalista', 'Separatista')`; `result_rows(summary: dict) -> list[dict]` con `{'name', 'pct', 'seats', 'p_first'}` por fila de `summary['parties']`, en su orden, valores sin formatear (la plantilla usa los filtros `pct`, `int`, `prob` ya existentes); `block_result_rows(summary: dict, meta: dict, parties: Catalog) -> list[dict]` con `{'name', 'pct', 'seats', 'p_majority', 'color'}` (`color` = `block_color(name, meta, parties)`) para las filas de `summary['blocks']` cuyo `seats` no es `None`, primero las de `BLOCK_ORDER` en ese orden y después las demás en el suyo. Contexto: `result_rows`, `block_result_rows`; `majority` desaparece.
- Produces (plantilla): `<div class="hemi-grid">` con `card_start('hemicycle-card', 'Hemiciclo', …, subtitle, 'hemi-card')` (gráfico `#hemicycle`), `card_start('parties-card', 'Resultados por partido', …)` (tabla `#parties-table`, cabeceras `Partido`, `% Voto`, `Escaños`, `P(primero)`) y `card_start('blocks-card', 'Resultados por bloque', …)` (tabla `#blocks-result-table`, cabeceras `Bloque`, `% Voto`, `Escaños`, `P(mayoría)`); entradillas y "Cómo leerlo" con los textos de la spec D3. La rejilla va justo después de la tarjeta `series`.
- Produces (CSS): `.hemi-grid { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: var(--space-4); }`, `.hemi-card { grid-row: 1 / 3; display: flex; flex-direction: column; }`, `.hemi-card .chart { flex: 1; min-height: 380px; }`, y en el bloque móvil (`max-width: 719px`) una columna con `.hemi-card { grid-row: auto; }`; entre 720 y 900 px también una columna (nueva `@media (max-width: 899px)`).
- `paintResults` deja de dibujar barras y mayoría: solo hemiciclo y evolución.

- [ ] **Step 1: Tests que fallan**:

```python
def test_result_rows_and_block_result_rows_follow_the_summary_and_the_block_order():
    summary = {'parties': [{'name': 'PP', 'pct': 32.66, 'seats': 139, 'p_first': 0.95, 'p_majority': 0.0}],
               'blocks': [{'name': 'Izquierda', 'pct': 35.4, 'seats': 117, 'p_majority': 0.0},
                          {'name': 'Derecha', 'pct': 52.5, 'seats': 203, 'p_majority': 0.995},
                          {'name': 'Separatista', 'pct': None, 'seats': None, 'p_majority': None}]}
    assert pages.result_rows(summary) == [{'name': 'PP', 'pct': 32.66, 'seats': 139, 'p_first': 0.95}]
    rows = pages.block_result_rows(summary, fixture_data('meta'), pages.Catalog(fixture_data('meta')))
    assert [r['name'] for r in rows] == ['Derecha', 'Izquierda'] and rows[0]['p_majority'] == 0.995 and rows[0]['color'].startswith('#')
    assert pages.result_rows({}) == [] and pages.block_result_rows({}, fixture_data('meta'), pages.Catalog(fixture_data('meta'))) == []


def test_promedios_central_block_is_three_cards_without_ranges(api):
    page = html(call(api, '/es')[2])
    order = [page.index(m) for m in ('id="series"', 'id="hemicycle-card"', 'id="parties-card"', 'id="blocks-card"', 'id="evolution-card"')]
    assert order == sorted(order)
    heads = re.findall(r'<th scope="col">([^<]*)</th>', page[page.index('id="parties-table"'):page.index('id="blocks-result-table"')])
    assert heads == ['Partido', '% Voto', 'Escaños', 'P(primero)']
    assert re.findall(r'<th scope="col">([^<]*)</th>', page[page.index('id="blocks-result-table"'):page.index('id="evolution-card"')]) == ['Bloque', '% Voto', 'Escaños', 'P(mayoría)']
    for gone in ('id="vote-chart"', 'id="seats-chart"', 'id="majority"', 'id="headline-table"', 'Rango 95'):
        assert gone not in page
```

(Si `Catalog`/`fixture_data` no están importados en ese test, importarlos como hacen los tests vecinos.)

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py -q -k "result_rows or central_block"` → FAIL.
- [ ] **Step 3: Implementar** contexto, plantilla, CSS y `paintResults`; adaptar `test_promedios_orders_the_story_and_carries_the_controls` (marcadores `id="hemicycle-card"`, `id="parties-card"`, `id="blocks-card"` en vez de `id="headline"`, `id="vote-chart"`, `id="hemicycle"`) y borrar `bars.js` si queda huérfano.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Replace the central block of Promedios with the hemicycle and two result tables"`.

---

### Task 4: leyenda como selector de partidos y controles propios del gráfico de encuestadora

**Files:**
- Modify: `web/templates/promedios.html` (tarjeta `series`: quitar `#series-parties` y `#series-blocks`; tarjeta `pollster`: añadir `#pollster-view` y `#pollster-window`; "Cómo leerlo" de ambas), `web/dist/js/pages/promedios.js:100-250` (`wireSeries`), `mtpy/lib/pages/promedios.py` (claves `main` y `block_colors` del contexto, si quedan sin uso)
- Test: `tests/test_pages.py` (`test_promedios_controls_are_hidden_without_js_and_check_the_main_parties`, `test_promedios_hides_the_blocks_view_without_blocks`, `test_promedios_controls_in_three_rows_with_pills_and_the_polls_table_by_name`, `test_promedios_pills_carry_the_colour_and_an_accessible_legend`, `test_promedios_pills_carry_the_ink_of_their_colour`: las comprobaciones de pills pasan a `/es/escanos` (calculadora de coaliciones, `#coalition-parties`) o se borran)

**Interfaces:**
- Produces (plantilla): dentro de `#pollster-controls` (que sigue `hidden`), y antes del `<label>Encuestadora`, `<fieldset id="pollster-view" class="chart-window">` (solo con `blocks`; radios `name="pview"` `partidos` checked / `bloques`) y `<fieldset id="pollster-window" class="chart-window">` (radios `name="pwindow"` `all` checked, `1a`, `6m`, `3m`, mismas etiquetas que `#series-window`).
- Produces (`promedios.js`): el gráfico del promedio se dibuja con `legend: true`; `chart.on('legendselectchanged', (event) => …)` fija `state.selected` a los nombres de `view().parties` con `event.selected[name] !== false` (o `null` si están todos) y llama a `writeHash()`. Al dibujar, `selected: state.selected` filtrado a los nombres de la vista (los desconocidos se ignoran). El gráfico de la encuestadora tiene su propio estado `{view, window}` (inicial: el del promedio al cargar), sus propios listeners y `legend: true`; no escribe el hash. Se retiran `markPills`, `syncBoxes`, `applySelection` y `boxes` de este módulo (`markPills` sigue en `common.js` para Escaños).
- Textos: "Cómo leerlo" del promedio añade la frase "Pulsa un partido de la leyenda para ocultarlo o mostrarlo."; el de la encuestadora cambia "Sigue la ventana, la vista y los partidos del gráfico del promedio." por "Tiene su propia ventana y vista; pulsa la leyenda para mostrar u ocultar partidos."

- [ ] **Step 1: Tests que fallan**:

```python
def test_promedios_legend_is_the_party_selector_and_the_house_chart_has_its_own_controls(api):
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    assert 'id="series-parties"' not in page and 'id="series-blocks"' not in page and 'name="party"' not in page
    assert re.search(r'<fieldset id="pollster-view" class="[^"]*">', page) and re.search(r'<fieldset id="pollster-window" class="[^"]*">', page)
    assert 'name="pview" value="partidos" checked' in page and 'name="pwindow" value="all" checked' in page and 'name="pwindow" value="3m"' in page
    assert page.index('id="pollster-window"') < page.index('id="pollster-select"')
    assert 'Pulsa un partido de la leyenda' in page and 'Tiene su propia ventana y vista' in page
```

Y en `tests/test_js_blocks.py` (paridad del hash, Review Focus 5): `parseSeriesHash('#vista=partidos&sel=Derecha,PP', {parties: ['PP', 'PSOE'], blocks: ['Derecha'], windows: [...]})` devuelve `selected == ['PP']` (si ya lo hace, el test lo fija; si no, `parseSeriesHash` filtra por los nombres de la vista).

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py tests/test_js_blocks.py -q -k "legend_is_the_party_selector or sel"` → FAIL.
- [ ] **Step 3: Implementar** plantilla y `wireSeries`; adaptar y mover los tests de pills.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS. Comprobación manual del controlador después (no la hagas tú).
- [ ] **Step 5: Commit**: `git commit -m "Select the parties of the average from the legend and give the house chart its own controls"`.

---

### Task 5: "Todas las encuestas"

**Files:**
- Create: `web/dist/js/polls-table.js` (sustituye al actual; borrar `nextRows`, `captionText`, `TABLE_STEP`)
- Modify: `mtpy/lib/pages/promedios.py` (`chart_polls`, `table_rows` → `polls_table`, `TABLE_ROWS` → `PAGE_SIZE = 15`, nueva `poll_months`, contexto), `web/templates/promedios.html` (tarjeta `polls`), `web/dist/js/pages/promedios.js` (`wirePollsTable`), `web/dist/css/site.css` (`.bar`, `.dd`, `.et`, `.pag`; retirar `#polls-more` y la regla de `#polls-table th:nth-child(2)`), `web/dist/js/format.js` si hace falta una fecha numérica
- Test: `tests/test_js_polls.py`, `tests/test_pages.py` (`test_promedios_has_the_series_chart_the_polls_table_and_csv_link`, `test_table_rows_are_newest_first_and_formatted`, `test_table_rows_tolerate_a_poll_without_date`, `test_promedio_caption_counts_the_polls`, `test_promedios_loads_more_polls_when_the_cycle_has_more_than_forty`, `test_promedios_initial_polls_carry_only_the_chart_columns`, `test_promedios_orders_the_story_and_carries_the_controls` (claves de `initial`))

**Interfaces:**
- Produces (`promedios.py`): `chart_polls` conserva además `sponsor`, `start_date` y `end_date` (cuando el sondeo los tiene). `poll_months(polls: list) -> list[dict]`: `{'value': 'YYYY-MM', 'label': 'Ago 2023'}` por cada mes con sondeos, ascendente (abreviaturas `ene feb mar abr may jun jul ago sep oct nov dic` con mayúscula inicial). `polls_table(polls: list, parties: list, colors: dict, n: Optional[int] = PAGE_SIZE) -> dict`: `{'rows': [...], 'mean': {'n': int, 'cells': [...]}, 'total': int}`; cada fila `{'pollster': str, 'sponsor': str | None, 'date': 'dd/mm/aa', 'fieldwork': 'Trabajo de campo: {fmt_date(start)} – {fmt_date(end)}' | None, 'sample': fmt_int(sample_size), 'cells': [{'value': fmt_num(v, 1) | DASH, 'leader': bool, 'color': colors[name]}]}`; las `n` más recientes por fecha (empate: la publicada después primero, como hoy); `leader` = valor igual al máximo de la fila entre los presentes (varios si empatan; ninguno si no hay valores); `mean` sobre esas `n` filas, celdas con el mismo formato y líder. Contexto: `table` (el dict de `polls_table`), `table_parties` (= `polls['parties']`, todos), `pollsters` (ya existe), `months` (`poll_months`), `csv_href`, `table_total`; desaparece `initial.table`.
- Produces (plantilla, tarjeta `polls`, título "Todas las encuestas", entradilla de la spec D5): `<div class="bar" id="polls-controls" hidden>` con `<details class="dd" id="polls-parties"><summary>Partidos (<span data-count>N</span>/N)</summary><div class="dd-panel"><button type="button" data-all>Seleccionar todos</button><button type="button" data-none>Quitar todos</button>{% for name in table_parties %}<label class="chip" style="--pill-color: …; --pill-ink: …"><input type="checkbox" name="party" value="{{ name }}" checked> {{ name }}</label>{% endfor %}</div></details>`, lo mismo para `#polls-pollsters` (`name="pollster"`, "Seleccionar todas"/"Quitar todas", chips neutros, opciones de `pollsters`), `<label>Desde <select id="polls-from">…</select></label>`, `<label>Hasta <select id="polls-to">…</select></label>` (opciones de `months`, `from` el primero y `to` el último `selected`), `<span class="bar-end"><span id="polls-count">{{ table_total }} sondeos</span> · <a href="{{ csv_href }}">Descargar CSV</a></span>`. Tabla `<div class="et-wrap"><table class="et" id="polls-table"><colgroup>…<thead><tr><th class="f l" data-key="pollster">Encuestadora</th><th data-key="date" aria-sort="descending">Fecha</th><th data-key="sample_size">Muestra</th><th class="p" data-key="PP" title="Partido Popular"><i class="pd" style="background: …"></i>PP</th>…</tr></thead><tbody>` (15 filas: `<td class="f l"><b>{{ pollster }}</b><small>{{ sponsor or '—' }}</small></td><td title="{{ fieldwork }}">{{ date }}</td><td>{{ sample }}</td><td class="p mx" style="background: {{ color }}24">33,4</td>` / `<td class="p">…</td>` / `<td class="p na">—</td>`) `<tfoot><tr><td class="f l"><b>Media simple</b><small>{{ mean.n }} sondeos</small></td><td></td><td></td>…</tr></tfoot></table></div><div class="pag" id="polls-pag" hidden><button type="button" data-page="-1">‹ Más recientes</button><span id="polls-range"></span><button type="button" data-page="1">Más antiguos ›</button></div>` y la pista `<p class="scroll-hint">Desliza para ver más →</p>`.
- Produces (`web/dist/js/polls-table.js`, puro): `PAGE_SIZE = 15`; `monthOf(poll) -> 'YYYY-MM' | null`; `monthsOf(polls) -> string[]` (ascendente, únicos); `monthLabel('YYYY-MM') -> 'Ago 2023'`; `shortDate(iso) -> 'dd/mm/aa'` (`'–'` sin fecha); `filterPolls(polls, {pollsters, from, to}) -> polls` (`pollsters`: array de nombres o `null` = todos; `from`/`to` meses inclusivos o `null`); `sortPolls(polls, key, asc) -> polls` (`key` ∈ `'date' | 'pollster' | 'sample_size' |` nombre de partido; los `null` al final en ambos sentidos; desempate: fecha descendente; `pollster` por `localeCompare('es')`); `pageOf(rows, page, size = PAGE_SIZE) -> rows`; `pageLabel(page, shown, total) -> '1–15 de 315'`; `meanOf(rows, parties) -> {name: number | null}` (media sin ponderar de los presentes, 1 decimal); `leaderOf(values) -> string[]` (nombres con el máximo entre los presentes, `[]` si ninguno).
- Produces (`promedios.js`): `wirePollsTable(initial, catalog)`: estado `{parties, pollsters, from, to, sort: {key: 'date', asc: false}, page: 0}`; reconstruye `thead` (columnas de `parties` marcadas), `tbody` (página actual) y `tfoot` (media de todas las filtradas) con nodos del DOM (`textContent`, nunca HTML con datos); clic en `th[data-key]` ordena (misma clave: invierte; `aria-sort` solo en la activa); cambios de filtro vuelven a la página 0; actualiza `#polls-count` ("N sondeos" de las filtradas), los contadores de los `summary` y las etiquetas de paginación ("‹ Más recientes/Más antiguos ›" solo cuando la clave es `date` descendente; si no, "‹ Anteriores/Siguientes ›"); los `details` se cierran al pulsar fuera; `data-all`/`data-none` marcan o desmarcan todos dejando al menos uno (si se quitan todos, se vuelve a marcar el primero); muestra `#polls-controls` y `#polls-pag` (cuando hay más de una página).

- [ ] **Step 1: Tests que fallan** (`tests/test_pages.py`):

```python
def test_polls_table_marks_the_leader_formats_the_values_and_averages_the_rows():
    polls = [
        {'date': '2026-10-01', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1010, 'start_date': '2026-09-28', 'end_date': '2026-09-30', 'PP': 31.1, 'PSOE': 26.1},
        {'date': '2026-09-04', 'pollster': 'CIS', 'sponsor': None, 'sample_size': None, 'PP': 25.5, 'PSOE': 31.0},
        {'date': '2026-09-02', 'pollster': '40dB', 'sponsor': 'Prisa', 'sample_size': 2000, 'PP': None, 'PSOE': None},
    ]
    table = pages.polls_table(polls, ['PP', 'PSOE'], {'PP': '#1d84ce', 'PSOE': '#ef1c27'}, n=2)
    assert table['total'] == 3 and [r['pollster'] for r in table['rows']] == ['GAD3', 'CIS']
    first = table['rows'][0]
    assert first['date'] == '01/10/26' and first['sponsor'] == 'ABC' and first['sample'] == '1.010'
    assert first['fieldwork'] == 'Trabajo de campo: 28 sep 2026 – 30 sep 2026'
    assert first['cells'] == [{'value': '31,1', 'leader': True, 'color': '#1d84ce'}, {'value': '26,1', 'leader': False, 'color': '#ef1c27'}]
    assert table['rows'][1]['sample'] == '–' and table['rows'][1]['fieldwork'] is None and table['rows'][1]['cells'][1]['leader']
    assert table['mean'] == {'n': 2, 'cells': [{'value': '28,3', 'leader': False, 'color': '#1d84ce'}, {'value': '28,6', 'leader': True, 'color': '#ef1c27'}]}
    empty = pages.polls_table(polls[2:], ['PP', 'PSOE'], {'PP': '#1', 'PSOE': '#2'})
    assert empty['rows'][0]['cells'][0] == {'value': '–', 'leader': False, 'color': '#1'} and empty['mean']['cells'][0]['value'] == '–'
    assert pages.poll_months(polls) == [{'value': '2026-09', 'label': 'Sep 2026'}, {'value': '2026-10', 'label': 'Oct 2026'}]


def test_promedios_polls_table_is_server_rendered_with_fifteen_rows_and_hidden_controls(api):
    body = call(api, '/es')[2]
    page = html(body).replace('\n', ' ')
    assert 'Todas las encuestas' in page and 'Sondeos del ciclo' not in page and 'id="polls-more"' not in page
    assert re.search(r'<div class="bar" id="polls-controls" hidden>', page) and 'id="polls-from"' in page and 'id="polls-to"' in page
    assert 'Seleccionar todos' in page and 'Seleccionar todas' in page
    tbody = page[page.index('<tbody>', page.index('id="polls-table"')):page.index('</tbody>', page.index('id="polls-table"'))]
    assert tbody.count('<tr>') == min(15, FIXTURE_POLLS) and 'Media simple' in page and 'class="p mx"' in page
    assert 'Descargar CSV' in page and 'Desliza para ver más' in page
    data = initial(body)
    assert 'table' not in data and {'sponsor', 'date', 'pollster'} <= set(data['polls']['polls'][0])
```

(`FIXTURE_POLLS` = número de sondeos del fixture `polls` de `tests/fixtures/bundle`; cuéntalo y pon el entero.) En `tests/test_js_polls.py`, con `run_js` sobre un fixture de 3 sondeos (los del test anterior): `p.monthsOf(polls) == ['2026-09', '2026-10']`, `p.monthLabel('2023-08') == 'Ago 2023'`, `p.shortDate('2026-10-01') == '01/10/26'`, `p.filterPolls(polls, {pollsters: ['CIS'], from: null, to: null}).length == 1`, `p.filterPolls(polls, {pollsters: null, from: '2026-10', to: '2026-10'}).length == 1`, `p.sortPolls(polls, 'PP', false).map(x => x.pollster) == ['GAD3', 'CIS', '40dB']` (el `null` al final) y `asc` → `['CIS', 'GAD3', '40dB']`, `p.pageOf([1,2,3,4], 1, 3) == [4]`, `p.pageLabel(0, 15, 315) == '1–15 de 315'`, `p.meanOf(polls, ['PP', 'PSOE']) == {PP: 28.3, PSOE: 28.6}`, `p.leaderOf({PP: 31.1, PSOE: 31.1, VOX: null}) == ['PP', 'PSOE']` y `p.leaderOf({PP: null}) == []`.

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py tests/test_js_polls.py -q -k "polls_table or server_rendered or monthsOf or leaderOf"` → FAIL.
- [ ] **Step 3: Implementar** módulo puro, contexto, plantilla, `wirePollsTable` y CSS (`.et` de la maqueta: `table-layout: fixed`, `min-width: 980px`, primera columna `position: sticky; left: 0`, cabeceras en mayúsculas pequeñas, `.mx` en negrita, `tfoot` con fondo `--color-soft`; `.dd-panel` absoluto bajo su `summary`, con `max-width: 420px`, `max-height: 280px`, `overflow-y: auto`, chips en línea y `z-index`; `.pag` con los botones a los lados). Borrar los tests de `table_rows`, `captionText`/`nextRows` y "Cargar más"; adaptar los de las claves de `initial` y de `chart_polls`.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Replace the polls of the cycle with a filterable, sortable and paginated table"`.

---

### Task 6: "Resultados por territorio"

**Files:**
- Create: `web/dist/js/territory.js`
- Modify: `mtpy/lib/pages/escanos.py` (`escanos_context`: `initial` con `districts` y `scenario`; nueva `has_communities`), `web/templates/escanos.html:119-155` (título, controles, tabla, pista), `web/dist/js/pages/escanos.js` (`wireTerritory`), `web/dist/css/site.css` (`.geo`, `.scroll-hint`)
- Test: `tests/test_js_seats.py` (o nuevo `tests/test_js_territory.py` con el mismo patrón), `tests/test_pages.py` (`test_escanos_is_rendered_with_tables_links_and_initial`, `test_escanos_districts_default_to_the_largest_and_link_the_rows`)

**Interfaces:**
- Produces (`escanos.py`): `has_communities(regions: list) -> bool` (al menos dos `community` distintas no nulas); contexto `territory_levels: bool`; `initial` añade `'districts': districts` y `'scenario': scenario`; `district_table` no cambia (sigue dando la tabla del servidor: nivel Provincia, modo Escaños).
- Produces (plantilla, tarjeta `districts`): título "Resultados por territorio", entradilla "El reparto de escaños y el voto de cada circunscripción en el escenario central y, para la que elijas, el detalle con sus intervalos."; `<div class="series-controls" id="territory-controls" hidden>` con `<fieldset id="territory-level" class="chart-window">` (solo con `territory_levels`; radios `name="level"`: `ccaa` checked "CCAA", `prov` "Provincia") y `<fieldset id="territory-mode" class="chart-window">` (radios `name="tmode"`: `pct` "% Voto", `esc` checked "Escaños"); la tabla actual pasa a `<table id="district-table" class="geo">` con `<th scope="col" data-key="name">Circunscripción</th>`, `<th scope="col" data-key="total">Total</th>` como última columna (antes de las de partido hoy va "Escaños"; pasa al final como "Total") y `<th scope="col" class="p" data-key="PP">` con el punto de color; la fila `Total` pasa a `<tr class="total"><td>TOTAL</td>…`; tras la tabla, `<p class="scroll-hint">Desliza para ver más →</p>`; "Cómo leerlo" con la frase nueva de la spec D6.
- Produces (`web/dist/js/territory.js`, puro): `territoryRows(districts, scenario, summary, {level = 'prov', mode = 'esc'}) -> {parties: string[], rows: [{id: number | null, name: string, values: {name: number | null}, total: number | null, ranges: {name: string | null}}], total: {values: {name: number}, total: number | string}}`. Escaños por provincia = fila de `scenario.rows` de ese `region_id` (`scenario.parties` da el orden); % Voto = `pct` de `districts.rows`; `ranges` = `'lo–hi'` de `seats_lo`/`seats_hi` (solo en `esc`); `total` de la fila = `seats` de la región (`esc`) o `'100 %'` (`pct`). Nivel `ccaa`: agrupa por `region.community.code` (nombre `community.name`), suma escaños, pondera `pct` por `region.votes` (sin `votes` o sin `community`, cae a `prov`), `id: null`. `parties`: en `esc`, los de `scenario.parties` con algún escaño en alguna fila, por total descendente; en `pct`, todos los de `districts.parties` por su `pct` en `summary.parties` descendente. `total.values`: en `esc`, la fila `region_id === 0` de `scenario` si existe o la suma de las filas; en `pct`, `pct` de `summary.parties`. `winnerOf(values) -> string[]` (máximo > 0; empates incluidos; `[]` si nada); `columnRange(rows, name) -> {min, max}` sobre valores > 0; `tintAlpha(value, min, max) -> number` (0 cuando no hay valor; `0.18` si `max === min`; lineal entre 0 y 0.18); `sortTerritory(rows, key, asc) -> rows` (`'name'` por `localeCompare('es')`, `'total'` o un partido por valor, `null` al final).
- Produces (`escanos.js`): `wireTerritory(initial, catalog)`: estado `{level: territory_levels ? 'ccaa' : 'prov', mode: 'esc', sort: {key: 'name', asc: true}}`; reconstruye `thead`/`tbody` de `#district-table` con nodos del DOM: cabecera de territorio "CCAA"/"Provincia" según el nivel; celdas `<td><span class="cell [is-win]" style="background: {color}{alpha hex}" title="{range}">valor</span></td>` ("—" sin valor; `pct` con 1 decimal y " %"); punto del ganador (`<span class="tdot">`) tras el nombre; en nivel `prov` el nombre es `<a href="/{scope}/escanos?mode={mode}[&run={run}]&region={id}#districts">`; fila TOTAL; clic en `th[data-key]` ordena; muestra `#territory-controls`.
- Produces (CSS): `.geo` (mínimo 900 px en `.et-wrap`, cabeceras como `.et`, `.cell` con `border-radius: 6px; padding: 3px 6px; min-width: 54px`, `.cell.is-win { font-weight: 800; box-shadow: inset 0 0 0 1.5px rgba(0,0,0,.18); }`, `.tdot` y `tr.total td` con fondo `--color-soft`), `.scroll-hint { font-size: 0.8rem; color: var(--color-faint); }`.

- [ ] **Step 1: Tests que fallan**. Node (fixture: `districts` con regiones `28 Madrid` (`seats` 6, `votes` 600, `community` `{code: 13, name: 'Madrid'}`), `8 Barcelona` (4, 400, `{code: 9, name: 'Cataluña'}`), `17 Girona` (2, 200, `{code: 9, name: 'Cataluña'}`); `pct` PP 40/30/20, PSOE 35/30/20, VOX 10/5/0; `scenario.parties = ['PP', 'PSOE', 'VOX']`, filas `0: [7, 5, 0]`, `28: [3, 3, 0]`, `8: [2, 2, 0]`, `17: [2, 0, 0]`; `summary.parties` con `pct` PP 33, PSOE 30, VOX 7):
  - `territoryRows(..., {level: 'prov', mode: 'esc'})`: `parties == ['PP', 'PSOE']` (VOX sin escaños fuera), fila Madrid `values == {PP: 3, PSOE: 3}`, `total == 6`, `ranges.PP` es `'lo–hi'`; `total.values == {PP: 7, PSOE: 5}`.
  - `{level: 'ccaa', mode: 'pct'}`: dos filas; Cataluña `values.PP` ≈ `(30·400 + 20·200)/600 = 26.7` (1 decimal), `total == '100 %'`, `id === null`; `parties == ['PP', 'PSOE', 'VOX']`.
  - `{level: 'ccaa', mode: 'esc'}`: Cataluña `values == {PP: 4, PSOE: 2}`, `total == 6`.
  - Sin `community`/`votes` (fixture sin esos campos) y `level: 'ccaa'`: devuelve las tres provincias (cae a `prov`).
  - `winnerOf({PP: 3, PSOE: 3, VOX: 0}) == ['PP', 'PSOE']`, `winnerOf({PP: null}) == []`; `tintAlpha(3, 2, 4) == 0.09`, `tintAlpha(5, 5, 5) == 0.18`, `tintAlpha(null, 0, 1) == 0`; `sortTerritory(rows, 'PP', false)` pone Madrid primero y `sortTerritory(rows, 'name', true)` Barcelona primero.
  - Python (`tests/test_pages.py`):

```python
def test_escanos_territory_table_has_hidden_controls_and_the_island_carries_the_districts(api, fresh_app):
    write_districts(fresh_app.fs)
    body = call(api, '/es/escanos', query='mode=forecast')[2]
    page = html(body).replace('\n', ' ')
    assert 'Resultados por territorio' in page and 'Circunscripciones</h2>' not in page
    assert re.search(r'<div class="series-controls" id="territory-controls" hidden>', page)
    assert 'name="tmode" value="esc" checked' in page and 'name="level"' not in page  # el fixture no trae comunidades
    assert '<table id="district-table" class="geo">' in page and 'data-key="total"' in page and 'TOTAL' in page
    assert 'Desliza para ver más' in page and 'id="district-form"' in page and 'id="region-table"' in page
    data = initial(body)
    assert {'districts', 'scenario'} <= set(data) and data['districts']['regions'][0]['id'] == 28
    with_communities = {**DISTRICTS, 'regions': [dict(r, community={'code': 13 if r['id'] == 28 else 9, 'name': 'X'}, votes=100) for r in DISTRICTS['regions']]}
    write_districts(fresh_app.fs, districts=with_communities)
    page = html(call(api, '/es/escanos', query='mode=forecast')[2]).replace('\n', ' ')
    assert 'name="level" value="ccaa" checked' in page and 'name="level" value="prov"' in page
```

(`write_districts` y `DISTRICTS` ya existen en `tests/test_pages.py`; si `write_districts` cachea el manifest, limpiarlo como hacen los tests vecinos.)

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py tests/test_js_territory.py -q -k "territory"` → FAIL.
- [ ] **Step 3: Implementar** módulo puro, contexto, plantilla, `wireTerritory` y CSS; adaptar `test_escanos_is_rendered_with_tables_links_and_initial` (claves de `initial`) y `test_escanos_districts_default_to_the_largest_and_link_the_rows` (la columna "Escaños" pasa a "Total" al final; `<td>Total</td>` pasa a `<td>TOTAL</td>`).
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Replace the districts table with the results by territory"`.

---

### Task 7: estilo visual nuevo y contenedor fluido

**Files:**
- Modify: `web/dist/css/site.css` (tokens, base, `main`, `.grid`, `.card`, `.card-head`, `.chart-window` → `.seg`, `.pills`/`.pill`, tablas, `#event-bar`, `#site-footer`, bloque móvil), `web/dist/js/charts/base.js` (`THEME`: color de ejes y texto a los tokens nuevos), plantillas con `class="chart-window"` (`promedios.html`, `escanos.html` y cualquier otra que dé `grep -rn "chart-window" web/templates`) → `class="seg"`, `web/templates/_macros.html` solo si la barra del título necesita un elemento (no: va en `.card-head h2::before`/`h3::before`)
- Test: `tests/test_pages.py` (junto a `test_fonts_are_vendored_and_declared`)

**Interfaces:**
- Produces (CSS): los tokens de "Global Constraints"; `body { background: var(--color-bg); }`; `main { max-width: none; padding: var(--space-5) 32px 40px; }` (16 px en el bloque móvil); `--band-pad: 32px` (16 px en móvil) y se retira `--max-width`; `.grid { grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }` (`.grid-three` igual con 320 px); `.card { padding: 20px 22px; border-radius: var(--radius); box-shadow: var(--shadow); }`; `.card-head h2, .card-head h3 { display: flex; align-items: center; gap: 10px; }` con `::before { content: ""; width: 5px; height: 20px; border-radius: 3px; background: var(--color-accent); }`; `.seg { display: inline-flex; background: var(--color-soft); border-radius: 999px; padding: 3px; gap: 2px; border: 0; }`, `.seg legend { position: absolute; … }` (visualmente oculta, como `.sr-only`), `.seg label { border-radius: 999px; padding: 6px 14px; font-weight: 600; font-size: 0.85rem; color: var(--color-muted); cursor: pointer; }`, `.seg label:has(input:checked), .seg label.is-on { background: var(--color-accent); color: #fff; }` (dos reglas, como hoy con `.pill`), `.seg input { … }` (oculto como `.pill input`), `.seg[hidden] { display: none; }`; `.pill` conserva su contrato (`--pill-color`, `--pill-ink`, `is-on`) con borde de 1 px y radio 999; tablas: `th { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--color-faint); font-weight: 600; }`, `tr:hover td { background: #f6f8fb; }`; prosa: `.lead, .how-body, .card p, #glosario dd { max-width: 72ch; }`; `#event-bar` sin el área `mode` (`grid-template-areas: "title run" "tabs run"` o equivalente) y las pestañas `.event-tabs` como `.seg` (fondo `--color-soft`, activa en acento); `#site-footer` sobre `--color-bg` con borde superior `--color-line`.
- Produces (`base.js`): `THEME.textColor = '#1a1c22'`, `THEME.axisColor` (o el campo equivalente de los ejes) `rgba(0,0,0,.15)`, fondo transparente; `barColor` se mantiene.
- Produces (plantillas): todo `class="chart-window"` pasa a `class="seg"`; las pestañas del evento y los filtros de las páginas globales (`_scope_filter.html`, filtros de `elecciones.html`/`casas.html` si son radios) usan `.seg`.

- [ ] **Step 1: Tests que fallan**:

```python
def test_site_css_has_the_grey_tokens_the_fluid_container_and_the_segments():
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    for token in ('--color-bg: #f4f5f8;', '--color-card: #fff;', '--color-soft: #eceef2;', '--radius: 12px;', '--shadow: 0 2px 8px rgba(0,0,0,.06);', '--color-faint:'):
        assert token in css, token
    assert '--max-width' not in css and 'repeat(auto-fit, minmax(420px, 1fr))' in css
    assert '.seg {' in css and '.seg label.is-on' in css and 'max-width: 72ch' in css and '.pill {' in css


def test_controls_are_segments_on_every_page(api):
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    assert 'chart-window' not in page and re.search(r'<fieldset id="series-window" class="seg" hidden>', page)
    assert re.search(r'<fieldset id="pollster-window" class="seg">', page)
    assert 'chart-window' not in html(call(api, '/es/escanos')[2])
```

- [ ] **Step 2: Ejecutar y ver que fallan**: `python -m pytest tests/test_pages.py -q -k "grey_tokens or segments_on_every_page"` → FAIL.
- [ ] **Step 3: Implementar** el CSS, `base.js` y el renombrado en las plantillas; adaptar los tests que fijen `chart-window` (`test_promedios_controls_are_hidden_without_js…`, `test_window_options_run_from_the_whole_cycle_to_three_months`, los de la tarea 4 y 6) y comprobar con `grep -rn "chart-window\|--max-width\|mode-text" web mtpy tests` que no queda nada.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS. Además `bash deploy/nginx-check.sh` no aplica (sin nginx); el controlador hará la comprobación visual en local.
- [ ] **Step 5: Commit**: `git commit -m "Restyle the site with the grey background, white cards, segments and a fluid container"`.

---

### Task 8: cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-10-14-web-fase-7c-rediseno-design.md` ("Estado al cierre de la fase 7c"), `deploy/README.md` (solo si algún paso de operación cambia: `districts.regions` nuevos campos → una línea en "Publicar" / "Contrato")

- [ ] **Step 1: Estado al cierre**: lista de commits de la fase, número de tests, qué quedó fuera (nada previsto) y las comprobaciones manuales pendientes de Luis (recorrido visual en escritorio y móvil; `publish forecast` de `es` para que el run lleve `community`/`votes` y aparezca el nivel CCAA).
- [ ] **Step 2: Suite completa**: `python -m pytest -m "not integration" -q` → PASS, avisos no superiores a 54.
- [ ] **Step 3: Commit**: `git commit -m "Close web phase 7c in its spec"`.
