# Evolución de las publicaciones (un punto por día) · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rehacer la tarjeta «Evolución de las publicaciones» de Promedios y Escaños como un gráfico hermano del promedio: un punto por día ligado por línea, altura `chart-tall`, vista Partidos/Bloques con pastillas, eje que sigue a lo marcado y descarga.

**Architecture:** `charts/evolution.js` se reescribe como opción ECharts pura (`dailyRuns`, `evolutionOption`, `evolutionImageOption`) sin tocar `series.js`; `blocks.js` gana `blockRuns` (bloques sumados en el navegador, como `blockPolls`); un cableado compartido `pages/evolution-wire.js` (vista, pastillas, descarga) lo usan `promedios.js` y `escanos.js`. En Python sólo cambia `escanos.py` (bloques y tintas para las pastillas). Nada se republica.

**Tech Stack:** ECharts 5.6 (vanilla ES modules bajo `web/dist/js`), Jinja2 (`web/templates`), pytest con `node` para los módulos JS (`tests/test_js_*.py`, se saltan sin `node`).

**Spec:** `docs/superpowers/specs/2026-10-30-web-evolucion-publicaciones-design.md`

## Global Constraints

- Identificadores en inglés neutro; textos públicos en castellano con las reglas de `assert_public_wording` («sólo» con tilde, encuestadora/medio, ningún «Cómo leerlo» abierto).
- Un punto por día: agrupar por `as_of` (a falta de él, los 10 primeros caracteres de `run_at`); en vivo gana a `backfill`; entre varios del mismo tipo, el último `run_at`.
- Eje vertical: mínimo `floor(min)`, máximo `max(ceil(max), floor(min) + 1)`; sin cero fijo; marcas enteras en escaños (`minInterval: 1`).
- Se dibuja siempre el titular `nowcast`; subtítulos sin modo: «· voto de cada publicación» (Promedios) y «· escaños de cada publicación» (Escaños).
- Por defecto marcados los partidos de `bmaps.main` (los del promedio); todos si no hay. Bloques: todos marcados; Otros = resto hasta 100 (hasta la cámara en escaños), sólo cuando todos los demás bloques tienen valor.
- Sin selector de ventana, sin hash. Deslizador de zoom abajo siempre.
- Ids compartidos por las dos páginas: `evolution-chart`, `evolution-legend`, `evolution-view` (radios `evview`), `evolution-download`. Las tarjetas conservan sus ids (`evolution-card` en Promedios, `evolution` en Escaños).
- Commits sólo de los ficheros de cada tarea (`git add` nominal, nunca `-a`): Luis hace commits en paralelo en el mismo árbol; comprobar `git status` antes de cada commit y no reescribir los suyos.

## Review Focus

- Ámbito sin bloques (`bmaps.blocks` ausente o lista): sin selector Vista, sólo partidos, sin error. Prueba en la tarea 3 (`evolutionViews(...).bloques === null`).
- Un partido presente en runs antiguos pero no en el último (SUMAR → FA): desaparece del gráfico sin inventar ceros; un run sin valor de un partido no produce punto. Prueba en la tarea 2 (`VOX` con 3 puntos de 4 runs).
- Relleno y run en vivo el mismo día con el relleno a las 12:00 y el en vivo antes: gana el en vivo. Prueba en la tarea 2 (`dailyRuns`).
- Un único run (ámbito recién publicado): eje de `[floor, floor + 1]` y un punto; lista vacía de runs: opción sin series y sin excepción. Pruebas en la tarea 2.
- Run sin `as_of` (titulares antiguos): día de `run_at`. Prueba en la tarea 2.

---

### Task 1: `blockRuns` en `blocks.js`

**Files:**
- Modify: `web/dist/js/blocks.js` (junto a `blockPolls`, líneas 93-127; `restOf` líneas 30-32)
- Test: `tests/test_js_blocks.py`

**Interfaces:**
- Consumes: `REST_BLOCK`, `present`, `restOf` del propio módulo.
- Produces: `blockRuns(runs, blocks, field = 'pct', total = 100) → object[]`: copia de cada run con `nowcast.parties` sustituido por las filas de bloque `{name, [field]: suma}`; un bloque sin ningún partido presente no tiene fila; `REST_BLOCK` (si el mapa lo tiene) es `total − Σ` de los demás, sólo cuando todos tienen fila. `run_id`, `run_at`, `as_of`, `backfill` y `n_polls` se conservan.

- [ ] **Step 1: Escribir las pruebas que fallan** en `tests/test_js_blocks.py` (añadir `RUNS` a las constantes y a `run_js` como `runs`):

```python
RUNS = [{'run_id': 'a', 'run_at': '2026-10-05T12:00:00Z', 'as_of': '2026-10-05', 'backfill': True, 'n_polls': 311,
         'nowcast': {'parties': [{'name': 'PP', 'pct': 32.7, 'seats': 140}, {'name': 'VOX', 'pct': 17.9, 'seats': 63},
                                 {'name': 'PSOE', 'pct': 26.4, 'seats': 107}]}},
        {'run_id': 'b', 'run_at': '2026-10-06T09:30:00Z', 'as_of': '2026-10-06',
         'nowcast': {'parties': [{'name': 'PP', 'pct': 32.6, 'seats': 139}, {'name': 'PSOE', 'pct': 26.5, 'seats': 108}]}}]
BLOCKS_REST = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE'], 'Otros': []}


def test_block_runs_sum_the_parties_of_each_block_and_the_rest_to_the_total():
    rows = run_js('b.blockRuns(runs, %s, "pct", 100).map((run) => run.nowcast.parties)' % json.dumps(BLOCKS_REST))
    assert rows[0] == [{'name': 'Derecha', 'pct': pytest.approx(50.6)}, {'name': 'Izquierda', 'pct': 26.4},
                       {'name': 'Otros', 'pct': pytest.approx(23.0)}]
    seats = run_js('b.blockRuns(runs, %s, "seats", 350).map((run) => run.nowcast.parties)' % json.dumps(BLOCKS_REST))
    assert seats[0] == [{'name': 'Derecha', 'seats': 203}, {'name': 'Izquierda', 'seats': 107}, {'name': 'Otros', 'seats': 40}]


def test_block_runs_keep_the_run_fields_and_sum_only_the_parties_present():
    runs = run_js('b.blockRuns(runs, %s, "pct")' % json.dumps(BLOCKS_REST))
    assert [r['run_id'] for r in runs] == ['a', 'b'] and runs[0]['backfill'] is True and runs[0]['n_polls'] == 311
    # VOX is missing from the second run: Derecha is PP alone, as `blockPolls` does
    assert runs[1]['nowcast']['parties'][0] == {'name': 'Derecha', 'pct': 32.6}


def test_block_runs_without_rest_block_and_with_an_empty_block():
    rows = run_js('b.blockRuns(runs, %s, "pct").map((run) => run.nowcast.parties)' % json.dumps({'Derecha': ['PP', 'VOX'], 'Sep': ['ERC']}))
    assert rows[0] == [{'name': 'Derecha', 'pct': pytest.approx(50.6)}]  # Sep has no party in the run: no row, no rest
    with_rest = run_js('b.blockRuns(runs, %s, "pct").map((run) => run.nowcast.parties)' % json.dumps({'Derecha': ['PP'], 'Sep': ['ERC'], 'Otros': []}))
    assert with_rest[0] == [{'name': 'Derecha', 'pct': 32.7}]  # a block without row: no rest either
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_js_blocks.py -k block_runs -v`
Expected: FAIL (`b.blockRuns is not a function`).

- [ ] **Step 3: Implementar `blockRuns(runs, blocks, field = 'pct', total = 100)` en `web/dist/js/blocks.js`**, con docstring JSDoc en el estilo de `blockPolls`. Generalizar `restOf(others, total = 100)` (sin cambiar sus usos actuales).

- [ ] **Step 4: Comprobar que pasan todas las de bloques**

Run: `pytest tests/test_js_blocks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/blocks.js tests/test_js_blocks.py
git commit -m "web: bloques de los runs publicados (blockRuns) para la evolución"
```

---

### Task 2: `charts/evolution.js` reescrito

**Files:**
- Rewrite: `web/dist/js/charts/evolution.js`
- Modify: `web/dist/js/pages/promedios.js:96-98`, `web/dist/js/pages/escanos.js:346-348` (llamadas mínimas a la nueva firma, para que las páginas sigan funcionando hasta la tarea 4)
- Test: `tests/test_js_evolution.py` (nuevo)

**Interfaces:**
- Consumes: `axisLabelStyle`, `baseOption`, `escapeHtml`, `mountChart`, `THEME`, `tooltipStyle` (`charts/base.js`); `dayTime`, `isoDay`, `axisPctLabel` (`charts/series.js`); `OTHERS_COLOR` (`catalog.js`); `fmtDate`, `fmtDateShort`, `fmtDateTime`, `fmtPct` (`format.js`).
- Produces:
  - `runDay(run) → string`: `run.as_of || String(run.run_at).slice(0, 10)`.
  - `dailyRuns(runs) → object[]`: un run por día (regla de las Global Constraints), ascendente por día; los objetos originales, sin copiar.
  - `evolutionPoints(runs, parties, field = 'pct') → {name, points: Array<[time, value, index]>}[]`: `time = dayTime(runDay(run))`, `index` la posición del run en `runs`; los runs sin el partido o con valor nulo se saltan.
  - `evolutionBounds({min, max}) → [number|null, number|null]`: la regla del eje; `[null, null]` si alguno no es finito.
  - `evolutionOption({runs, parties, selected = null, colors = {}, fullnames = {}, field = 'pct', formatter = fmtPct, legend = 'hidden'}) → object`: `runs` ya diarios. Series `line` por partido en el orden de `parties`, `data` de `evolutionPoints` con los puntos de runs `backfill` como `{value, symbolSize: 8, itemStyle: {color: '#fff', borderColor, borderWidth: 2}}` y los demás como arrays; `showSymbol: true`, `symbolSize: 6`, `lineStyle: {width: 2}`, `emphasis: {focus: 'series'}`. `legend`: con `'hidden'`, `{show: false, selectedMode: true, type: 'scroll', top: 0, data: parties, selected}` (`selected[name] = !selected || selected.includes(name)`); con `true`, `show: true`. `grid: {left: 8, right: 24, top: 24, bottom: 56, containLabel: true}`. `xAxis` time con `minInterval` de un día, etiquetas `fmtDateShort(isoDay(value))`, sin `splitLine`. `yAxis` value con `min`/`max` funciones sobre `evolutionBounds`, `splitLine` del tema, etiquetas `axisPctLabel` para `pct` y `formatter(value)` para otros campos, `minInterval: 1` cuando `field === 'seats'`. Tooltip `trigger: 'axis'`, `confine: true`, cabecera `fmtDateTime(run.run_at)` para un run en vivo y `fmtDate(runDay(run)) + ' · estimación retrospectiva'` para un relleno, filas de mayor a menor `marker + fullname + ': ' + formatter(value)`. `dataZoom: [{type: 'slider', bottom: 8, filterMode: 'filter', labelFormatter: (value) => fmtDate(isoDay(value))}]`.
  - `evolutionImageOption(data, zoom = null) → object`: la opción con `legend: {show: true, type: 'plain', selectedMode: false, top: 0, data: shown}` (`shown` los nombres de `selected`, o todos), `grid.top: 40`, `grid.bottom: 8` y `dataZoom: [{type: 'inside', filterMode: 'filter', disabled: true, ...zoom}]` (`zoom` `{startValue, endValue}` o nada).
  - `renderEvolution(el, data) → chart`: `mountChart(el, evolutionOption(data))`.

- [ ] **Step 1: Escribir las pruebas que fallan** en `tests/test_js_evolution.py`, con el arnés de `tests/test_js_charts.py` (carga ECharts en `window` y los módulos `e` = `charts/evolution.js`, `b` = `blocks.js`) y este fijo:

```python
RUNS = [
    {'run_id': 'b05', 'run_at': '2026-10-05T12:00:00Z', 'as_of': '2026-10-05', 'backfill': True,
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.71, 'seats': 140}, {'name': 'PSOE', 'pct': 26.37, 'seats': 107}, {'name': 'VOX', 'pct': 17.93, 'seats': 63}]}},
    {'run_id': 'b06', 'run_at': '2026-10-06T12:00:00Z', 'as_of': '2026-10-06', 'backfill': True,
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.64, 'seats': 139}, {'name': 'PSOE', 'pct': 26.41, 'seats': 107}]}},
    {'run_id': 'l06', 'run_at': '2026-10-06T09:30:00Z', 'as_of': '2026-10-06',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.6, 'seats': 139}, {'name': 'PSOE', 'pct': 26.45, 'seats': 108}, {'name': 'VOX', 'pct': 17.95, 'seats': 63}]}},
    {'run_id': 'l07a', 'run_at': '2026-10-07T08:00:00Z', 'as_of': '2026-10-07',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.5, 'seats': 138}, {'name': 'PSOE', 'pct': 26.5, 'seats': 108}, {'name': 'VOX', 'pct': 18.0, 'seats': 64}]}},
    {'run_id': 'l07b', 'run_at': '2026-10-07T18:00:00Z', 'as_of': '2026-10-07',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.4, 'seats': 138}, {'name': 'PSOE', 'pct': 26.55, 'seats': 108}, {'name': 'VOX', 'pct': 18.1, 'seats': 64}]}},
    {'run_id': 'x08', 'run_at': '2026-10-08T23:30:00Z',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.3, 'seats': 137}, {'name': 'PSOE', 'pct': 26.6, 'seats': 109}, {'name': 'VOX', 'pct': None, 'seats': None}]}},
]
PARTIES = ['PP', 'PSOE', 'VOX']
DATA = {'runs': None, 'parties': PARTIES, 'selected': ['PP', 'PSOE'], 'colors': {'PP': '#1e5aa8', 'PSOE': '#e3001b', 'VOX': '#63be21'}}


def option(**over):
    """`evolutionOption` sobre los runs diarios del fijo, con `over` encima de `DATA`."""
    return run_js('e.evolutionOption({...%s, runs: e.dailyRuns(runs), ...%s})' % (json.dumps(DATA), json.dumps(over)))


def test_daily_runs_keep_one_run_per_day_live_over_backfill_and_the_last_live():
    assert run_js('e.dailyRuns(runs).map((run) => run.run_id)') == ['b05', 'l06', 'l07b', 'x08']
    assert run_js('e.runDay(runs[5])') == '2026-10-08'  # without `as_of`: the day of `run_at`
    assert run_js('e.dailyRuns([])') == []


def test_evolution_points_skip_the_runs_without_the_party_and_carry_the_run_index():
    points = run_js('e.evolutionPoints(e.dailyRuns(runs), %s, "pct")' % json.dumps(PARTIES))
    assert [p['name'] for p in points] == PARTIES
    assert [p[1] for p in points[0]['points']] == [32.71, 32.6, 32.4, 32.3] and [p[2] for p in points[0]['points']] == [0, 1, 2, 3]
    assert [p[1] for p in points[2]['points']] == [17.93, 17.95, 18.1]  # VOX: null in the last run
    assert run_js('e.evolutionPoints(e.dailyRuns(runs), ["PP"], "seats")[0].points.map((p) => p[1])') == [140, 139, 138, 137]


def test_evolution_bounds_round_to_whole_units_around_the_visible_data():
    assert run_js('e.evolutionBounds({min: 31.53, max: 32.71})') == [31, 33]
    assert run_js('e.evolutionBounds({min: 8, max: 8})') == [8, 9]
    assert run_js('e.evolutionBounds({min: 32, max: 33})') == [32, 33]
    assert run_js('e.evolutionBounds({min: Infinity, max: -Infinity})') == [None, None]


def test_evolution_option_draws_a_line_per_party_with_hollow_retrospective_points():
    opt = option()
    assert [s['name'] for s in opt['series']] == PARTIES and all(s['type'] == 'line' for s in opt['series'])
    first = opt['series'][0]['data'][0]
    assert first['value'][1] == 32.71 and first['itemStyle'] == {'color': '#fff', 'borderColor': '#1e5aa8', 'borderWidth': 2}
    assert opt['series'][0]['data'][1][1:] == [32.6, 1]  # a live point is a plain `[time, value, index]`
    assert opt['legend'] == {'show': False, 'selectedMode': True, 'type': 'scroll', 'top': 0, 'data': PARTIES,
                             'selected': {'PP': True, 'PSOE': True, 'VOX': False}}
    assert opt['dataZoom'][0]['type'] == 'slider' and opt['xAxis']['type'] == 'time'
    assert run_js('typeof e.evolutionOption({...%s, runs: e.dailyRuns(runs)}).yAxis.min' % json.dumps(DATA)) == 'function'
    assert 'minInterval' not in opt['yAxis'] and option(field='seats')['yAxis']['minInterval'] == 1
    assert run_js('e.evolutionOption({runs: [], parties: []}).series') == []


def test_evolution_image_option_shows_a_plain_legend_of_the_names_shown_and_freezes_the_zoom():
    img = run_js('e.evolutionImageOption({...%s, runs: e.dailyRuns(runs)}, {startValue: 1, endValue: 2})' % json.dumps(DATA))
    assert img['legend'] == {'show': True, 'type': 'plain', 'selectedMode': False, 'top': 0, 'data': ['PP', 'PSOE']}
    assert img['dataZoom'] == [{'type': 'inside', 'filterMode': 'filter', 'disabled': True, 'startValue': 1, 'endValue': 2}]
    assert img['grid']['bottom'] == 8 and img['grid']['top'] == 40
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_js_evolution.py -v`
Expected: FAIL (`e.dailyRuns is not a function`).

- [ ] **Step 3: Reescribir `web/dist/js/charts/evolution.js`** con las funciones de Interfaces (cabecera del módulo y JSDoc en inglés, como `series.js`). El tooltip lee el run por el índice del punto: `runs[params[0].value[2]]`.

- [ ] **Step 4: Adaptar las dos llamadas de las páginas a la nueva firma** (sin pastillas todavía): en `promedios.js` `renderEvolution(document.getElementById('evolution'), {runs: dailyRuns(runs.runs), parties: current.parties.map((party) => party.name), colors, fullnames})` y en `escanos.js` lo mismo con `field: 'seats', formatter: fmtInt` sobre `#seats-evolution`.

- [ ] **Step 5: Comprobar que pasan, y la suite JS entera**

Run: `pytest tests/test_js_evolution.py tests/test_js_charts.py tests/test_js_blocks.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/dist/js/charts/evolution.js web/dist/js/pages/promedios.js web/dist/js/pages/escanos.js tests/test_js_evolution.py
git commit -m "web: evolución de las publicaciones con un punto por día y eje ajustado"
```

---

### Task 3: `pages/evolution-wire.js`

**Files:**
- Create: `web/dist/js/pages/evolution-wire.js`
- Test: `tests/test_js_evolution.py` (añadir `ew` = `pages/evolution-wire.js` y `c` = `catalog.js` al arnés)

**Interfaces:**
- Consumes: `dailyRuns`, `evolutionImageOption`, `renderEvolution` (tarea 2); `blockRuns` (tarea 1); `chosenNames`, `fillLegend` (`pages/series-wire.js`); `markLegend` (`pages/legend-dom.js`); `markSegments`, `mapNames`, `showError`, `tryDraw` (`pages/common.js`); `wireDownload` (`share.js`); `Catalog` (`catalog.js`); `fmtPct` (`format.js`).
- Produces:
  - `evolutionViews(runs, catalog, {field = 'pct', total = 100} = {}) → {partidos, bloques}`: `partidos = {runs: dailyRuns(runs), parties, colors, fullnames, defaults}` con `parties` = `catalog.order` de los nombres con valor de `field` en el `nowcast` del último run diario, `colors` del catálogo, `fullnames` `catalog.fullname`, `defaults` = los nombres de `parties`, en su orden, que están en `bmaps.main` (null si no queda ninguno); `bloques` = null salvo que `catalog.bmaps.blocks` sea un objeto no vacío, y entonces `{runs: blockRuns(daily, blocks, field, total), parties: catalog.orderBlocks(keys), colors: catalog.blockColor, fullnames: identidad, defaults: null}`.
  - `wireEvolution(initial, catalog, {field = 'pct', formatter = fmtPct, total = 100} = {})`: busca `#evolution-chart` (sin él no hace nada), `#evolution-legend`, `#evolution-view`, `#evolution-download`; estado `{view: 'partidos', selected: null}`; dibuja con `renderEvolution(el, dataOf(state))` donde `dataOf` devuelve `{runs, parties, selected: state.selected ?? defaults, colors, fullnames, field, formatter, legend: 'hidden'}` de la vista; la primera vez escucha `legendselectchanged` (`state.selected = chosenNames(view, event.selected)`, `markLegend`); `fillLegend(legendEl, view, state.selected, chart, initial.inks || {})` tras cada dibujo; radios `evview` marcados según el estado, `markSegments()`, `viewSet.hidden = false` y en `change` cambia la vista, anula la selección y redibuja; `wireDownload(button, () => evolutionImageOption(dataOf(state), chartZoom(chart)), showError)`.

- [ ] **Step 1: Escribir las pruebas que fallan** (mismo fichero, mismo fijo; `META = {'parties': [{'name': 'PP', 'color': '#1e5aa8', 'fullname': 'Partido Popular', 'block': 'Derecha'}, {'name': 'VOX', 'color': '#63be21', 'block': 'Derecha'}, {'name': 'PSOE', 'color': '#e3001b', 'block': 'Izquierda'}], 'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE'], 'Otros': []}}}`):

```python
def test_evolution_views_take_the_parties_of_the_last_run_in_catalogue_order_with_the_main_ones_by_default():
    views = run_js('ew.evolutionViews(runs, new c.Catalog(%s))' % json.dumps(META))
    assert views['partidos']['parties'] == ['PSOE', 'PP'] and views['partidos']['defaults'] == ['PSOE', 'PP']  # VOX is null in the last run
    assert views['partidos']['fullnames'] == {'PSOE': 'PSOE', 'PP': 'Partido Popular'} and views['partidos']['colors']['PP'] == '#1e5aa8'
    assert [r['run_id'] for r in views['partidos']['runs']] == ['b05', 'l06', 'l07b', 'x08']
    assert views['bloques']['parties'] == ['Izquierda', 'Otros', 'Derecha'] and views['bloques']['defaults'] is None
    assert views['bloques']['colors'] == {'Izquierda': '#e3001b', 'Otros': '#9e9e9e', 'Derecha': '#1e5aa8'}
    assert views['bloques']['runs'][0]['nowcast']['parties'][0] == {'name': 'Derecha', 'pct': pytest.approx(50.64)}


def test_evolution_views_without_blocks_or_main_parties():
    meta = {**META, 'bmaps': {'blocks': []}}
    views = run_js('ew.evolutionViews(runs, new c.Catalog(%s))' % json.dumps(meta))
    assert views['bloques'] is None and views['partidos']['defaults'] is None
    seats = run_js('ew.evolutionViews(runs, new c.Catalog(%s), {field: "seats", total: 350}).bloques.runs[0].nowcast.parties' % json.dumps(META))
    assert seats == [{'name': 'Derecha', 'seats': 203}, {'name': 'Izquierda', 'seats': 107}, {'name': 'Otros', 'seats': 40}]
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_js_evolution.py -k views -v`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Crear `web/dist/js/pages/evolution-wire.js`** con `evolutionViews` y `wireEvolution` (cabecera y JSDoc como `series-wire.js`; `wireEvolution` sigue la forma de `wireHouse` en `promedios.js:113-195`, sin ventana ni selector de casa).

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_js_evolution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/pages/evolution-wire.js tests/test_js_evolution.py
git commit -m "web: cableado compartido de la evolución (vista, pastillas y descarga)"
```

---

### Task 4: Plantillas, Python de Escaños y páginas

**Files:**
- Modify: `web/templates/promedios.html:186-196`, `web/templates/escanos.html:278-292`
- Modify: `mtpy/lib/pages/escanos.py:338-390` (contexto `blocks` e `inks`)
- Modify: `web/dist/js/pages/promedios.js` (quitar la llamada de la tarea 2 y llamar a `wireEvolution`), `web/dist/js/pages/escanos.js` (ídem con `{field: 'seats', formatter: fmtInt, total: nSeats}`)
- Test: `tests/test_pages.py` (líneas 677, 683, 688-694, 893-897)

**Interfaces:**
- Consumes: `wireEvolution` (tarea 3); macros `card_start(..., tools=m.download_button(...))`, `how()`; `ink_color`, `block_color` (`pages/common.py`).
- Produces: en Escaños `context['blocks']` (lista de nombres de `meta.bmaps.blocks` si es un dict, `[]` si no) e `initial['inks']` (`{name: ink_color(parties[name]['color'])}` para cada partido de `meta['parties']` más `{block: ink_color(block_color(block, meta, parties))}` por bloque), última clave de `initial`.

- [ ] **Step 1: Escribir las pruebas de página que fallan** (sustituir `test_evolution_cards_explain_the_retrospective_points` y retocar las otras tres):

```python
def test_evolution_cards_share_the_markup_and_explain_the_revision(api):
    for path, pill in (('/es', 'voto de cada publicación'), ('/es/escanos', 'escaños de cada publicación')):
        for query in ('', 'mode=forecast'):
            page = html(call(api, path, query=query)[2])
            card = page[page.index('Evolución de las publicaciones'):page.index('id="evolution-chart"')]
            assert '<span class="pill">%s</span>' % pill in card
            assert 'de cada publicación' not in card.replace(pill, '')  # no mode in the subtitle: both modes publish the same figures
            assert 'id="evolution-download"' in card and 'id="evolution-view"' in card and 'name="evview"' in card
            assert 'id="evolution-legend"' in card and '<div class="chart chart-tall" id="evolution-chart"' in page
            text = re.sub(r'\s+', ' ', page[page.index('id="evolution-chart"'):])
            text = text[:text.index('</section>')]
            assert 'estimaciones retrospectivas' in text and 'revisa el pasado' in text and 'no se revisan' in text
            assert 'Si un día hubo varias publicaciones, se muestra la última' in text
            assert_public_wording(page, path)
    assert 'hasta completar la cámara' in html(call(api, '/es/escanos')[2]) and 'hasta 100' in html(call(api, '/es')[2])
```

Y: en la línea 677 `'id="evolution-chart"'` en vez de `'id="seats-evolution"'`; en la 683 la lista de claves termina en `'deltas', 'inks'`; en la 688-694 el tramo `evolution` se corta en `id="downloads"` como ahora y sigue exigiendo `'run'`, `'huecos'` y `'no es una serie de sondeos'`.

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_pages.py -k "evolution or escanos_is_rendered" -v`
Expected: FAIL.

- [ ] **Step 3: Plantillas.** En las dos tarjetas: `card_start` con el nuevo subtítulo y `tools=m.download_button('evolution-download', event_bar.title ~ ' · ' ~ event_bar.date|date, 'Evolución de las publicaciones · ' ~ title_suffix, site_host ~ ' · ' ~ author_handle, site_title|lower ~ '-evolucion-' ~ state.scope ~ '-' ~ meta.run_at[:10] ~ '.png')` (`-evolucion-escanos-` en Escaños); debajo `<div class="series-controls">` con, sólo `{% if blocks %}`, `<fieldset id="evolution-view" class="seg" hidden><legend>Vista</legend>` y los radios `evview` `partidos` (checked) y `bloques`; `<div class="legend" id="evolution-legend" role="group" aria-label="Partidos y bloques del gráfico" hidden></div>`; `<div class="chart chart-tall" id="evolution-chart" role="img" aria-label="Evolución de las publicaciones: …">`. Texto de «Cómo leerlo» de Promedios (Escaños igual con «los escaños de cada partido», «Cada publicación del modelo es un run: una ejecución completa con los sondeos conocidos ese día.» como primera frase y «hasta completar la cámara»):

```html
<p>
  Cada punto es lo que el modelo publicó ese día: el voto de cada partido con los sondeos conocidos hasta
  entonces. Si un día hubo varias publicaciones, se muestra la última. Los puntos huecos son estimaciones
  retrospectivas: el modelo de hoy con los sondeos publicados hasta aquel día.
</p>
<p>
  Esto no es una serie de sondeos ni la curva del promedio. La curva del promedio revisa el pasado: el valor de
  cada día usa también los sondeos que llegaron después, así que cambia con cada sondeo nuevo. Estos puntos
  no se revisan: cada uno es lo que se publicó ese día. Por eso pueden no coincidir.
</p>
<p>
  Los puntos se mueven de un día a otro porque entran sondeos nuevos, no porque cambie el modelo: mira la
  tendencia, no cada salto. Pulsa un partido de la leyenda para ocultarlo o mostrarlo; el eje vertical se
  ajusta a lo que queda a la vista, así que con pocos partidos marcados se ven mejor los cambios pequeños.
  En la vista Bloques, cada bloque suma sus partidos y Otros es lo que falta hasta 100.
</p>
```

- [ ] **Step 4: Python.** En `escanos_context` añadir `context['blocks']` e `initial['inks']` como dice Interfaces (importar `block_color` e `ink_color` de `.common`).

- [ ] **Step 5: Páginas.** `promedios.js`: quitar la llamada de la tarea 2 en `paintResults` y, en el mismo sitio, `tryDraw(() => wireEvolution(initial, catalog), 'the evolution')`. `escanos.js`: ídem con `wireEvolution(initial, catalog, {field: 'seats', formatter: fmtInt, total: nSeats})`. Quitar los imports que queden sin uso.

- [ ] **Step 6: Suite completa**

Run: `pytest -q`
Expected: todo en verde (los `test_js_*` pasan con `node`).

- [ ] **Step 7: Commit**

```bash
git add web/templates/promedios.html web/templates/escanos.html mtpy/lib/pages/escanos.py web/dist/js/pages/promedios.js web/dist/js/pages/escanos.js tests/test_pages.py
git commit -m "web: tarjetas de evolución con vista, pastillas, descarga y texto de la revisión"
```

---

### Task 5: Comprobación en navegador y cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-10-30-web-evolucion-publicaciones-design.md` (nota de cierre)

- [ ] **Step 1: Levantar la web local** contra el bundle local (`files/site`, un solo run de `es`) o, si Luis tiene su gunicorn en el 8000, usarlo; abrir `/es` y `/es/escanos` con Chrome headless (CDP) o a mano y comprobar: un punto por día, pastillas que amplían el eje (PP solo: eje de dos puntos), vista Bloques con Izquierda/Otros/Derecha, tooltip por día con la etiqueta retrospectiva, descarga de la imagen con la leyenda de lo marcado, y la consola sin errores.

- [ ] **Step 2: Cerrar la spec** con una sección «Estado al cierre» (commits, pruebas, lo comprobado en navegador y lo que queda a Luis: integrar dev y desplegar; nada que republicar).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-10-30-web-evolucion-publicaciones-design.md
git commit -m "docs: cierre de la evolución de las publicaciones"
```
