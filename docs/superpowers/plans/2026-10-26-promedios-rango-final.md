# Promedios: rango del 29-N como barra al final, sin banda en el histórico — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quitar la banda del promedio y la franja de la proyección del gráfico de series y dibujar, en Elección, el
rango del 95 % de cada partido el día de la votación como una barra vertical con realce al pasar.

**Architecture:** Todo en el navegador: `seriesOption` deja de emitir las series de banda y añade una serie `custom`
por partido en la última fecha de la proyección (`[when, lo, hi, mean]`), con `emphasis.focus: 'series'` para que la
barra del partido quede al frente y las demás se atenúen; el tooltip del histórico pierde el paréntesis del
intervalo. Las partes publicadas no cambian. Los textos de cuatro plantillas dejan de hablar de la banda.

**Tech Stack:** ECharts 5.6 (módulos ES en `web/dist/js`, probados con `node` en `tests/test_js_*.py`), Jinja2,
pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-promedios-rango-final-design.md`

## Global Constraints

- Redacción pública: «sólo» (solamente) con tilde; «encuestadora» o «casa», nunca «empresa» ni «patrocinador»;
  ningún `<details class="how" open`; `assert_public_wording` lo comprueba.
- Las partes `series` y `projection` y la API no cambian; `bandSeries` sigue exportado para `weights.js`.
- Comentarios y JSDoc en inglés; textos públicos en español. Los tests de JS se saltan sin `node`: correrlos con
  `node` disponible (`shutil.which('node')`).
- Sin datos nuevos ni republicación.

## Review Focus

- Un run en Hoy (`projection.dates` de un solo día) o una elección pasada (sin proyección): ni barras ni bandas; el
  gráfico sigue dibujando líneas, sondeos, rombos y marcas (Tarea 2).
- La capa «Sondeos» no dibuja barras; «Promedio» y «Ambos» sí (Tarea 2).
- La vista Bloques: una barra por bloque desde `blockProjection`, con el color del bloque (Tarea 2).
- Un partido sin proyección (`groups.lo[name]` ausente o con `null` al final): sin barra, sin error (Tarea 2).
- El tooltip del histórico ya no enseña «(lo–hi)», pero el del ejemplo de /metodo/promedio (`weights.js`) lo
  conserva porque pasa `lo`/`hi` (Tarea 1).

---

### Task 1: `meanTooltipLine` sin paréntesis cuando no hay límites y con cabecera libre

**Files:**
- Modify: `web/dist/js/polls-tooltip.js:60-73`
- Test: `tests/test_js_polls.py:48-52`

**Interfaces:**
- Produces: `meanTooltipLine(name, mean, lo, hi, date, color, header = null)`: `header` es `true` («Proyección»,
  compatibilidad), una cadena (cabecera literal) o `null`; sin `lo`/`hi` el texto es `PP: 33,1% · 13 oct 2026`.

- [ ] **Step 1: Cambiar el test para que falle**

```python
def test_mean_tooltip_line_shows_the_average_its_interval_and_the_date():
    line = run_js('s.meanTooltipLine("PP", 33.1, 31.0, 35.3, "2026-10-13", "#1")')
    assert line.endswith('PP: 33,1% (31,0–35,3) · 13 oct 2026') and 'background-color:#1' in line
    # without bounds there is no parenthesis (the history of the series chart, whose band is no longer drawn)
    bare = run_js('s.meanTooltipLine("PP", 33.1, null, null, "2026-10-13", "#1")')
    assert bare.endswith('PP: 33,1% · 13 oct 2026') and '(' not in bare
    projected = run_js('s.meanTooltipLine("PP", 33.1, null, null, "2026-11-29", "#1", true)')
    assert projected.startswith('Proyección<br>') and projected.endswith('PP: 33,1% · 29 nov 2026')
    election = run_js('s.meanTooltipLine("PP", 31.4, 25.9, 37.0, "2026-11-29", "#1", "Elección")')
    assert election.startswith('Elección<br>') and election.endswith('PP: 31,4% (25,9–37,0) · 29 nov 2026')
```

- [ ] **Step 2: Verificar que falla**

Run: `python -m pytest tests/test_js_polls.py -q -k mean_tooltip`
Expected: FAIL (`bare` termina en `(–) · 13 oct 2026`; `election` empieza por `Proyección<br>`).

- [ ] **Step 3: Implementar en `polls-tooltip.js`**

`header === true ? 'Proyección' : header` como cabecera; el paréntesis sólo cuando `lo` y `hi` no son `null`/`undefined`.
JSDoc actualizado (`@param {boolean|string|null} header`).

- [ ] **Step 4: Verificar que pasa**

Run: `python -m pytest tests/test_js_polls.py -q`
Expected: PASS.

---

### Task 2: Sin bandas, con barras del rango en la elección

**Files:**
- Modify: `web/dist/js/charts/series.js` (constantes `:13-23`, JSDoc `:150-172`, cuerpo de `seriesOption` `:173-300`)
- Test: `tests/test_js_charts.py:211-317`

**Interfaces:**
- Consumes: `meanTooltipLine` de la Tarea 1 (`header` «Elección»); `blockProjection` de `blocks.js` (ya devuelve
  `{dates, groups: {parties: …}}` para los bloques).
- Produces: en la opción, por partido con proyección de más de un día y capa de promedio, una serie
  `{type: 'custom', name, color, z: 6, data: [[dayTime(lastDate), lo, hi, mean]], encode: {x: 0, y: [1, 2, 3]},
  emphasis: {focus: 'series', blurScope: 'coordinateSystem'}, blur: {…}}` cuyo `renderItem` dibuja la barra
  (`lineWidth: 4`, `lineCap: 'round'`, `opacity: RANGE_OPACITY = 0.45`, al frente y opacidad 1 en `emphasis`,
  opacidad 0,15 en `blur`) y el punto (`r: 4`, `fill: '#fff'`, borde del color). Ninguna serie de banda en la
  opción. Papel `range` en el tooltip.

- [ ] **Step 1: Actualizar y añadir tests para que fallen**

En `series_roles`, la rama de banda pasa a detectar la barra: `elif entry['type'] == 'custom': roles.add('range')`
(la de `lineStyle.opacity == 0` se elimina). En `test_series_layers_draw_the_polls_the_average_or_both`:
`everything = ['marks', 'mean', 'poll', 'projection', 'range', 'result']`, el caso sin proyección
`['marks', 'mean', 'poll']` y «promedio» `['marks', 'mean', 'projection', 'range', 'result']`. Test nuevo:

```python
def test_series_draws_the_election_range_as_a_bar_instead_of_the_bands():
    """Luis, 2026-10-26: fuera la banda del promedio y la franja de la proyección; en Elección, una barra por partido
    con el rango del 95 % del último día de la proyección y el punto del voto central, al frente al pasar."""
    option = run_js('s.seriesOption(%s)' % (LAYER_ARGS % "'ambos'"))
    assert not [s for s in option['series'] if s.get('lineStyle', {}).get('opacity') == 0]  # no bands at all
    bars = [s for s in option['series'] if s['type'] == 'custom']
    last = run_js('s.dayTime("2026-10-03")')
    assert [(s['name'], s['data'], s['z'], s['color']) for s in bars] == [('PP', [[last, 30, 33, 31.5]], 6, '#1d84ce')]
    assert bars[0]['encode'] == {'x': 0, 'y': [1, 2, 3]}
    assert bars[0]['emphasis'] == {'focus': 'series', 'blurScope': 'coordinateSystem'}
    # the bar is drawn after the lines, so it sits over them; hidden with the polls-only layer and without projection
    assert option['series'].index(bars[0]) > max(i for i, s in enumerate(option['series']) if s['type'] == 'line' and s['name'] == 'PP' and s.get('lineStyle', {}).get('type') != 'dashed')
    assert not [s for s in run_js('s.seriesOption(%s)' % (LAYER_ARGS % "'sondeos'"))['series'] if s['type'] == 'custom']
    assert not [s for s in run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))['series'] if s['type'] == 'custom']
    # the tooltip of the bar: «Elección» and the figure with its range at the election day
    box = run_js("(() => { const o = s.seriesOption(%s); const i = o.series.findIndex((x) => x.type === 'custom'); "
                 "return o.tooltip.formatter({seriesIndex: i, seriesName: 'PP', dataIndex: 0, data: o.series[i].data[0]}); })()" % (LAYER_ARGS % "'ambos'"))
    assert box.startswith('Elección<br>') and box.endswith('PP: 31,5% (30,0–33,0) · 3 oct 2026')
    # the history keeps its value and date, without the interval of the average
    line = run_js("(() => { const o = s.seriesOption(%s); return o.tooltip.formatter({seriesIndex: 0, seriesName: 'PP', dataIndex: 0}); })()" % (SERIES_ARGS % "'hidden'"))
    assert line.endswith('PP: 30,0% · 1 oct 2026') and '(' not in line


def test_series_range_bar_renders_a_rounded_line_and_a_white_point():
    shape = run_js("(() => { const o = s.seriesOption(%s); const bar = o.series.find((x) => x.type === 'custom'); "
                   "const api = {value: (k) => [1, 30, 33, 31.5][k], coord: ([x, y]) => [x === 1 ? 100 : 0, y === 0 ? 0 : 400 - y * 10], style: () => ({})}; "
                   "const g = bar.renderItem({}, api); return {type: g.type, kinds: g.children.map((c) => c.type), "
                   "line: g.children[0].shape, lineStyle: g.children[0].style, emphasis: g.children[0].emphasis, blur: g.children[0].blur, "
                   "point: g.children[1].shape, pointStyle: g.children[1].style}; })()" % (LAYER_ARGS % "'ambos'"))
    assert shape['type'] == 'group' and shape['kinds'] == ['line', 'circle']
    assert shape['line'] == {'x1': 100, 'y1': 100, 'x2': 100, 'y2': 70}  # lo 30 → 100, hi 33 → 70
    assert shape['lineStyle'] == {'stroke': '#1d84ce', 'lineWidth': 4, 'lineCap': 'round', 'opacity': 0.45}
    assert shape['emphasis'] == {'style': {'opacity': 1}} and shape['blur'] == {'style': {'opacity': 0.15}}
    assert shape['point'] == {'cx': 100, 'cy': 85, 'r': 4} and shape['pointStyle'] == {'fill': '#fff', 'stroke': '#1d84ce', 'lineWidth': 2}
```

Y en `test_series_views_carry_the_result_of_each_view`, un caso con proyección de bloques:

```python
    with_projection = {**initial, 'projection': {'dates': ['2026-10-01', '2026-10-02'], 'groups': {
        'parties': {'mean': {'PP': [30, 30]}, 'lo': {'PP': [28, 27]}, 'hi': {'PP': [32, 33]}},
        'blocks': {'mean': {'Derecha': [40, 40]}, 'lo': {'Derecha': [37, 36]}, 'hi': {'Derecha': [43, 44]}}}}}
    bloques = run_js('sw.seriesViews(%s, new c.Catalog(%s.catalog)).bloques' % (json.dumps(with_projection), json.dumps(initial)))
    assert bloques['projection']['groups']['parties']['lo'] == {'Derecha': [37, 36]}
```

- [ ] **Step 2: Verificar que fallan**

Run: `python -m pytest tests/test_js_charts.py -q -k "series"`
Expected: FAIL (`series_roles` encuentra `band`; no hay series `custom`; el tooltip del histórico lleva paréntesis).

- [ ] **Step 3: Implementar en `series.js`**

Quitar `BAND_OPACITY`/`PROJECTION_OPACITY` y las dos llamadas a `bandSeries` (la función se conserva y se exporta
para `weights.js`); añadir `RANGE_OPACITY = 0.45`, `RANGE_BLUR_OPACITY = 0.15`; función `rangeBarSeries(name, time,
lo, hi, mean, color)` con el `renderItem` descrito en Interfaces (la `api.coord` da `x` de `[time, 0]` e `y` de
`[0, valor]`); en `seriesOption`, `ranges = !withProjection || !withAverage ? [] : parties.flatMap(...)` sólo para
los partidos con `lo`, `hi` y `mean` finitos en el último día, con `withRole('range', …)`; `allSeries = [...means,
...points, ...projected, ...ranges, ...results, markSeries]`; en `tooltip`, el papel `mean` pasa `null, null` como
límites y `range` devuelve `meanTooltipLine(name, mean, lo, hi, lastDate, color(name), 'Elección')`. JSDoc del
módulo y de `seriesOption` sin «band», con la barra.

- [ ] **Step 4: Verificar que pasa**

Run: `python -m pytest tests/test_js_charts.py tests/test_js_polls.py tests/test_js_histogram.py -q`
Expected: PASS.

---

### Task 3: Los textos dejan de hablar de la banda

**Files:**
- Modify: `web/templates/promedios.html:49-66, 222-226`, `web/templates/eleccion.html:151, 156-172`,
  `web/templates/metodo_escanos.html:56-57, 159, 168-170`, `web/templates/metodo_promedio.html:21, 40-47, 651-660`
- Test: `tests/test_pages.py:572` y nuevos; `tests/test_pages_ejemplo.py` (anclas de /metodo/promedio sin cambio: el
  id `banda` se conserva)

**Interfaces:**
- Produces: textos públicos exactos de abajo; ningún «banda» en `promedios.html` ni `eleccion.html`.

- [ ] **Step 1: Cambiar y añadir tests para que fallen**

En `tests/test_pages.py:572`: `'banda de incertidumbre y la proyección"'` → `'aria-label="Promedio de sondeos y la proyección"'`.
Test nuevo junto a él:

```python
def test_series_cards_explain_the_election_bar_and_not_a_band(api, events_site):
    """Luis, 2026-10-26: la banda del promedio ya no se dibuja; en Elección, la barra del final es el rango del 29-N."""
    page = html(call(api, '/es')[2])
    text = re.sub(r'\s+', ' ', page)
    assert 'banda' not in text.lower()
    assert 'la barra del final es el rango del 95% de cada partido ese día: pasa por el punto para ver las cifras' in text
    assert 'La línea dice dónde está el consenso de las encuestadoras; la barra del final, dónde puede estar el voto el día de la elección.' in text
    assert 'la línea es el promedio de todas' in text
    assert_public_wording(page, '/es')
    detail = re.sub(r'\s+', ' ', html(call(events_site, '/elecciones/es/generales/2023-07')[2]))
    assert 'banda' not in detail.lower() and 'La línea dice dónde estaba el consenso de las encuestadoras, no la incertidumbre del resultado' in detail
    method = re.sub(r'\s+', ' ', html(call(api, '/metodo/promedio')[2]))
    assert 'Qué significa la precisión del promedio' in method and 'href="#banda">La precisión del promedio</a>' in method
    assert 'ya no la dibujan como una banda' in method
    seats = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'el de su banda' not in seats and 'es la precisión del promedio' in seats
```

(Si `events_site` no sirve el detalle de 2023 con esa ruta, usar la que ya usan los tests del detalle.)

- [ ] **Step 2: Verificar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "series_cards or aria or chart_ids"`
Expected: FAIL por los textos antiguos.

- [ ] **Step 3: Escribir los textos**

`promedios.html`: aria-label `"Promedio de sondeos{{ projection_label }}"` (con `projection_label` = ` y la proyección`
en Elección, ya definido); «Cómo leerlo» de la serie:

```
Cada punto es un sondeo publicado (al pasar por encima, o al tocarlo, se ve entero); la línea es el promedio
ponderado del modelo. El selector Sondeos / Promedio / Ambos deja a la vista sólo los puntos, sólo la línea, o las
dos cosas, para leer mejor cada una en el móvil. En el modo Elección, la línea discontinua lleva el voto estimado
hasta el día de la votación y la barra del final es el rango del 95% de cada partido ese día: pasa por el punto
para ver las cifras. En la vista Bloques, cada bloque suma sus partidos. Pulsa un partido de la leyenda para
ocultarlo o mostrarlo.
```

y el último párrafo: «La línea dice dónde está el consenso de las encuestadoras; la barra del final, dónde puede
estar el voto el día de la elección.» Tarjeta de la encuestadora: «Los puntos son sólo los sondeos de la
encuestadora elegida; la línea es el promedio de todas.»

`eleccion.html`: aria-label «Promedio de sondeos del ciclo, los sondeos y el resultado oficial»; «la línea es el
promedio ponderado del modelo. El selector Sondeos / Promedio / Ambos deja a la vista sólo los puntos, sólo la
línea, o las dos cosas.»; y «La línea dice dónde estaba el consenso de las encuestadoras, no la incertidumbre del
resultado; y no recoge lo que cambió en la última semana.»

`metodo_escanos.html`: «<var>s</var> es el error del propio promedio: dice cuánto puede equivocarse la línea al
resumir los sondeos.»; en la tarjeta de los tres errores «<var>s</var> es la precisión del promedio» y «Por eso vale
varias veces el error del promedio: ese error dice cuánto puede equivocarse la línea al resumir los sondeos;
<var>e</var>, cuánto se han equivocado los sondeos frente a las urnas.»

`metodo_promedio.html`: navegación `<a href="#banda">La precisión del promedio</a>`; intro: «…calculamos una
tendencia para cada día, con una medida de la precisión de ese cálculo.» y «Por eso la precisión de la curva y el
intervalo de nuestra estimación electoral responden a preguntas diferentes.»; la tarjeta `banda` se titula «Qué
significa la precisión del promedio» con la entradilla «La precisión con la que estimamos la tendencia que
describen las encuestas, no cuánto se parecerá al resultado.» y, tras su primer párrafo, uno nuevo:

```
Desde octubre de 2026 los gráficos de la web ya no la dibujan como una banda alrededor de la curva: se confundía
con el margen del pronóstico, que es mucho mayor. Es la primera de las tres piezas de ese margen, la <var>s</var>
de <a href="{{ method_href('escanos') }}#errores">¿Cómo simulamos los escaños?</a>; el ejemplo de más abajo sí la
dibuja, porque ahí es lo que se explica.
```

El resto de la tarjeta y las notas conservan «banda» como término técnico.

- [ ] **Step 4: Verificar que pasa**

Run: `python -m pytest tests/test_pages.py tests/test_pages_ejemplo.py -q`
Expected: PASS, incluido `assert_public_wording`.

---

### Task 4: Cierre

- [ ] **Step 1: Suite completa**

Run: `python -m pytest -q -m "not integration" -p no:cacheprovider`
Expected: todo en verde (1352 antes de este plan más los nuevos; los de JS requieren `node`).

- [ ] **Step 2: Cerrar la spec**

Estado: «construida el 2026-10-26 en `dev`, sin commits (los hace Luis)» y las desviaciones si las hubo.
