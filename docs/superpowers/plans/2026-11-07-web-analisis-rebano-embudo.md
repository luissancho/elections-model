# El embudo por semanas en la pregunta 2 del rebaño — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Volver a dibujar en la tarjeta 2 de `/analisis/rebano` la dispersión por tramos de semanas, una línea de color por elección general desde 2008, la media discontinua y 2026 en azul, con la medida de la versión 2.

**Architecture:** El estudio gana un bloque `weeks` (tramos de semanas sobre las comparaciones entre pares de casas, `stats.by_buckets`), publicado como parte `weeks` con su CSV; la página lo entrega en la isla `initial` y `web/dist/js/charts/weeks.js` (recuperado de git c3aa20e y adaptado) lo dibuja con ECharts; la descarga de la tarjeta lo redibuja sola.

**Tech Stack:** Python 3.11, pandas, numpy; Jinja2; ECharts 5.6 (vendored), html2canvas (card-image.js); pytest, node para los tests de JS.

**Spec:** `docs/superpowers/specs/2026-11-07-web-analisis-rebano-embudo-design.md` (leer entera; el contexto de la versión 2 está en `docs/superpowers/specs/2026-11-06-web-analisis-rebano-v2-design.md`).

## Global Constraints

- Textos públicos: «sólo» adverbio con tilde; «encuestadora» y «casa», nunca «empresa»; nunca «se copian», «copiar», «mirar al vecino», «permite el azar», ni explicaciones de intenciones.
- Identificadores en inglés; PEP8 ≤ 120; docstring numpy en toda función Python; `import mtpy.lib.publish` no debe cargar `mtpy.lib.studies.herd`.
- Parámetros exactos: `"weeks": {"scope": "es", "since": "2008-01-01", "min_pairs": 2, "buckets": [[0, 2], [3, 4], [5, 7], [8, 11], [12, 17], [18, 26], [27, 52]]}`; un tramo `[lo, hi]` cubre `lo·7 ≤ días < (hi + 1)·7`; `label` del tramo `'{lo}-{hi}'`.
- Ningún recuento, veredicto ni subtítulo cambia; los ficheros ya publicados (`meta`, `summary`, `elections`, `current`) no cambian de forma.
- Luis commitea en paralelo en el checkout principal: `git status` y `git log -3` antes de cada commit; sólo los ficheros de la tarea, por su nombre; nunca `git add -A` ni `commit -a`. Cada mensaje termina con `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Nada se publica en S3 ni se escribe en el bundle real desde el plan.

## Review Focus

- Un tramo con dos parejas de sondeos que comparten un sondeo (A-B y A-C) cuenta como 2 parejas y se dibuja: test en la tarea 1.
- La elección en curso de otro ámbito (`live_scope` ≠ `weeks.scope`) no entra en `weeks.rows`: test en la tarea 1.
- Sin ninguna elección pasada del ámbito desde `since`, `weeks.rows` y `mean` van vacíos y la página no pinta el gráfico ni dice «2026»: tests en las tareas 1 y 3.
- Una elección pasada con cociente en un solo tramo no rompe la media ni el gráfico (`connectNulls`): test en la tarea 3 (JS).
- La imagen descargable de la tarjeta 2 incluye el gráfico (ECharts redibujado por `card-image.js`): comprobación en navegador en la tarea 3.

---

### Task 1: Tramos de semanas en el estudio

**Files:**
- Modify: `mtpy/lib/studies/herd/stats.py` (añadir `bucket_of`, `by_buckets`)
- Modify: `mtpy/lib/studies/herd/study.py` (`HerdStudy.weeks`, `_weeks`, `compute_herd`)
- Modify: `data/rebano/params.json`, `data/README.md:32`
- Test: `tests/test_herd_stats.py`, `tests/test_herd_study.py`

**Interfaces:**
- Consumes: `comparisons(polls, names, params)` y `dispersion(rows)` de `stats.py`; `Election` (`scope`, `event_date`, `label`, `current`, `polls`, `names`) de `data.py`.
- Produces:
  - `bucket_of(days: float, buckets: list) -> Optional[tuple[int, int]]`: el `(lo, hi)` cuyo rango `lo·7 ≤ days < (hi + 1)·7` contiene `days`; `None` fuera de todos.
  - `by_buckets(rows: pd.DataFrame, buckets: list, min_pairs: int) -> list[dict]`: una entrada por tramo en el orden de `buckets`: `{lo, hi, pairs, comparisons, ratio}` con `pairs` = parejas distintas `(poll_i, poll_j)` del tramo, `comparisons` = filas y `ratio` = `dispersion` sin redondear, `None` si `pairs < min_pairs`.
  - `HerdStudy.weeks: dict` = `{'buckets': [{lo, hi, label}], 'rows': [{event_date, label, current, lo, hi, pairs, comparisons, ratio}], 'mean': [{lo, hi, k, ratio}]}`; `buckets` de más lejano a más cercano, en el orden de `params['weeks']['buckets']` invertido (27-52 primero); `rows` por elección (pasadas por fecha, la actual la última) y tramo en ese mismo orden; `ratio` a 2 decimales; `mean` sobre las pasadas con cociente, media sin redondear y luego a 2, `k` = cuántas.

- [ ] **Step 1: Write the failing tests**

`tests/test_herd_stats.py`:
```python
BUCKETS = [[0, 2], [3, 4], [5, 7], [8, 11], [12, 17], [18, 26], [27, 52]]

def test_bucket_of_edges():
    assert [bucket_of(d, BUCKETS) for d in (0, 20, 20.5, 21, 34.5, 35, 55, 56, 83, 84, 125, 126, 188, 189, 365, 400)] == [
        (0, 2), (0, 2), (0, 2), (3, 4), (3, 4), (5, 7), (5, 7), (8, 11), (8, 11), (12, 17), (12, 17), (18, 26),
        (18, 26), (27, 52), (27, 52), None]

def test_by_buckets_counts_distinct_poll_pairs():
    rows = pd.DataFrame({'poll_i': ['a', 'a', 'a', 'a', 'c'], 'poll_j': ['b', 'b', 'c', 'c', 'd'],
                         'days': [10.0, 10.0, 12.0, 12.0, 30.0], 'd2': [4.0, 0.0, 4.0, 0.0, 9.0], 'v': [2.0] * 5})
    out = by_buckets(rows, BUCKETS, 2)
    assert [(b['lo'], b['hi'], b['pairs'], b['comparisons']) for b in out[:3]] == [(0, 2, 2, 4), (3, 4, 1, 1), (5, 7, 0, 0)]
    assert out[0]['ratio'] == pytest.approx(1.0) and out[1]['ratio'] is None and out[2]['ratio'] is None
```

`tests/test_herd_study.py` (sobre `write_herd_bundle`, helper `compute`; cifras a mano: en 2019 las parejas A50-B48 y A40-B38 caen en 5-7 con Σd² 18 y Σv 16,131 → 1,06, y A15-B13 y A4-B2 en 0-2 con Σd² 2 → 0,35; en 2023 A10-B8, A3-B1 y B8-A3 caen en 0-2 con Σd² 27 y Σv 24,197 → 1,06; en 2026 A100-B98 y A90-B88 caen en 12-17 con Σd² 32 y Σv 16,204 → 1,41; A200-B198, A100-B97, A150-B148, A120-B118, A40-B38 y A30-B28 son parejas solas en su tramo):
```python
def test_compute_herd_weeks(fresh_app, tmp_path):
    weeks = compute(tmp_path).weeks
    assert [b['label'] for b in weeks['buckets']] == ['27-52', '18-26', '12-17', '8-11', '5-7', '3-4', '0-2']
    rows = {(r['label'], r['lo']): r for r in weeks['rows']}
    assert {r['label'] for r in weeks['rows']} == {'Generales 2019', 'Generales 2023', 'Generales 2026'}
    assert (rows[('Generales 2019', 5)]['pairs'], rows[('Generales 2019', 5)]['ratio']) == (2, 1.06)
    assert rows[('Generales 2019', 0)]['ratio'] == 0.35
    assert rows[('Generales 2019', 27)]['pairs'] == 1 and rows[('Generales 2019', 27)]['ratio'] is None
    assert (rows[('Generales 2023', 0)]['pairs'], rows[('Generales 2023', 0)]['ratio']) == (3, 1.06)
    assert rows[('Generales 2026', 12)]['current'] is True and rows[('Generales 2026', 12)]['ratio'] == 1.41
    assert rows[('Generales 2026', 3)]['ratio'] is None
    mean = {m['lo']: (m['k'], m['ratio']) for m in weeks['mean']}
    assert mean[0] == (2, 0.7) and mean[5] == (1, 1.06) and mean[27] == (0, None)
    assert len(weeks['rows']) == 3 * 7

def test_compute_herd_weeks_without_elections_of_the_scope(fresh_app, tmp_path):
    weeks = compute(tmp_path, params={'weeks': {'scope': 'es', 'since': '2030-01-01', 'min_pairs': 2,
                                                'buckets': [[0, 2], [27, 52]]}}).weeks
    assert [r['label'] for r in weeks['rows']] == ['Generales 2026'] * 2 and weeks['mean'] == [
        {'lo': 27, 'hi': 52, 'k': 0, 'ratio': None}, {'lo': 0, 'hi': 2, 'k': 0, 'ratio': None}]

def test_compute_herd_weeks_skips_a_current_election_of_another_scope(fresh_app, tmp_path):
    weeks = compute(tmp_path, params={'weeks': {'scope': 'es-md', 'since': '2000-01-01', 'min_pairs': 2,
                                                'buckets': [[0, 2]]}}).weeks
    assert [r['label'] for r in weeks['rows']] == ['Madrid 2021'] and not any(r['current'] for r in weeks['rows'])
```
(`compute` del test acepta `params`; si no, añadirle el argumento.)

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_herd_stats.py tests/test_herd_study.py -k "bucket or weeks"`
Expected: FAIL (ImportError / AttributeError)

- [ ] **Step 3: Implement** `bucket_of`, `by_buckets` (`stats.py`), `_weeks(elections, current, params) -> dict` y el campo `weeks` de `HerdStudy` (`study.py`), los parámetros y la fila del README. Las comparaciones de cada elección se calculan una sola vez por elección (reutilizar `comparisons` donde `_results` ya las obtiene, o calcularlas en `compute_herd` y pasarlas).

- [ ] **Step 4: Run the herd tests**

Run: `python -m pytest -q tests/test_herd_stats.py tests/test_herd_data.py tests/test_herd_study.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/stats.py mtpy/lib/studies/herd/study.py data/rebano/params.json data/README.md tests/test_herd_stats.py tests/test_herd_study.py
git commit -m "Rebaño: dispersión por tramos de semanas (generales desde 2008 y la elección en curso)"
```

### Task 2: La parte `weeks` publicada

**Files:**
- Modify: `mtpy/lib/bundle.py:89,100,110,194-200` (declaraciones y `SCHEMAS['rebano-weeks']`)
- Modify: `mtpy/lib/publish/herd.py` (`export_herd_weeks`), `mtpy/lib/publish/studies.py:614` (exportador), `mtpy/lib/publish/__init__.py` (re-export), `mtpy/lib/pages/analisis.py:64` (`CSV_LABELS`)
- Create: `tests/fixtures/bundle/rebano-weeks.json`
- Modify: `docs/web/contrato.md:545-557,888`
- Test: `tests/test_publish_herd.py` (+ los tests que listan las partes de `rebano`: `tests/test_bundle_unit.py`, `tests/test_web_api.py`, `tests/test_webapi_unit.py`)

**Interfaces:**
- Consumes: `HerdStudy.weeks` (tarea 1).
- Produces: `STUDY_PARTS['rebano'] = ('meta', 'summary', 'elections', 'current', 'weeks')`; `STUDY_CSV['rebano'] = ('elections', 'weeks')`; `STUDY_CSV_OF['rebano'] = {'elections': ('elections',), 'weeks': ('weeks',)}`; `SCHEMAS['rebano-weeks'] = {'buckets': list, 'rows': list, 'mean': list}`; `export_herd_weeks(study) -> tuple[dict, dict]` = `(study.weeks limpio, {'weeks': frame})` con el frame en forma larga: las filas más las de la media (`event_date` nulo, `label` «Media», `current` falso, `pairs` y `comparisons` nulos, `ratio`), columnas `WEEK_COLUMNS = ['event_date', 'label', 'current', 'lo', 'hi', 'pairs', 'comparisons', 'ratio']`; `CSV_LABELS['rebano']['weeks'] = 'Separación entre casas por elección y tramo de semanas (generales desde 2008 y la elección en curso)'`.
- Fixture `rebano-weeks.json`: los 7 tramos, las 7 generales pasadas con las cifras del spike (2015 3,03→1,36; 2016 sin 27-52 y 0,70 en 0-2; abr. 2019 1,51→1,33; …) y 2026 con cociente de 27-52 a 3-4 (1,01, 1,05, 1,28, 1,05, 1,11, 1,06) y `null` en 0-2; `mean` 1,65 / 1,31 / 1,42 / 1,42 / 1,35 / 1,19 / 1,07 con sus `k`.

- [ ] **Step 1: Write the failing tests** en `tests/test_publish_herd.py`

```python
def test_export_herd_weeks_validates_and_flattens(fresh_app, tmp_path):
    study = compute(tmp_path)
    data, frames = publish.STUDY_EXPORTERS['rebano']['weeks'](study)
    bundle.validate('rebano-weeks', data)            # el validador que usan los demás tests del fichero
    frame = frames['weeks']
    assert list(frame.columns) == WEEK_COLUMNS and len(frame) == len(data['rows']) + len(data['mean'])
    assert (frame.label == 'Media').sum() == 7 and frame[frame.label == 'Media'].event_date.isna().all()

def test_publish_study_rebano_writes_weeks(fresh_app, tmp_path):
    # tras publish_study: studies/rebano/weeks.json y csv/weeks.csv existen; STUDY_PARTS termina en 'weeks'
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_publish_herd.py`
Expected: FAIL

- [ ] **Step 3: Implement** las declaraciones, el esquema, el exportador, el re-export, el `CSV_LABELS`, el fixture y las dos tablas de `contrato.md`.

- [ ] **Step 4: Run the publish, bundle and web tests**

Run: `python -m pytest -q tests/test_publish_herd.py tests/test_publish_studies.py tests/test_bundle_unit.py tests/test_web_api.py tests/test_webapi_unit.py tests/test_pipelines_publish.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish/herd.py mtpy/lib/publish/studies.py mtpy/lib/publish/__init__.py mtpy/lib/pages/analisis.py tests/fixtures/bundle/rebano-weeks.json docs/web/contrato.md tests/test_publish_herd.py <tests tocados>
git commit -m "Rebaño: parte weeks publicada con su CSV"
```

### Task 3: El gráfico y la tarjeta

**Files:**
- Create: `web/dist/js/charts/weeks.js` (partir de `git show c3aa20e:web/dist/js/charts/weeks.js`)
- Rewrite: `web/dist/js/pages/analisis-rebano.js`
- Modify: `web/templates/analisis_rebano.html:99-108` (tarjeta 2), `mtpy/lib/pages/herd.py` (`weeks_text`), `mtpy/lib/pages/analisis.py` (`herd_context`: `initial`, `weeks_text`, `weeks_since`)
- Modify: `web/dist/css/site.css` (altura de `#weeks-chart`, ~320 px; 260 en el móvil)
- Test: `tests/test_js_charts.py` (volver a declarar `WEEKS_JS` y `wk` en `run_js`), `tests/test_pages.py`

**Interfaces:**
- Consumes: la parte `weeks` (tarea 2; fixture `rebano-weeks.json`), `study_part(herd.HERD, 'weeks')`.
- Produces:
  - `weeksOption(data, {current = THEME.accentColor, mean = THEME.textColor, reference = 1, yMax = 3, palette = PALETTE} = {})` en `weeks.js`, con `PALETTE = ['#e07b39', '#2a9d8f', '#8e44ad', '#c0392b', '#7f8c3a', '#d4a017', '#5d6d7e']`; series: una por elección pasada en el orden de las filas (`name` = `label`, `lineStyle.color` = `palette[i % n]`, `width` 1.6, `connectNulls`, puntos), después «Media de las {n}» (negro discontinuo, con `markLine` en `reference` rotulada «lo que cabría esperar sólo por el azar» y `markArea` con dos zonas: `[{xAxis: '5-7'}, {xAxis: '3-4'}]` rotulada «3-8 semanas» y `[{xAxis: '0-2'}, {xAxis: '0-2'}]` rotulada «tres últimas», sin banda bajo 1), y la actual la última (azul, `width` 3.5, `endLabel` «{label} · {valor}»); `xAxis.data` = etiquetas de los tramos, `xAxis.name` «semanas antes de la elección»; `yAxis` de 0 a `max(yMax, techo a 0,5)` de 0,5 en 0,5; tooltip por tramo «{label}: {valor} ({pairs} parejas)». `renderWeeks(el, data, opts)` monta con `mountChart`.
  - `herd.weeks_text(weeks: Optional[dict]) -> Optional[str]`: «La media de las {k} generales pasadas pasa de {ratio del primer tramo con media} a más de {lo} semanas a {ratio del último} en las tres últimas{; {label actual} va en {ratio} a {lo}-{hi} semanas}.» (`None` sin ninguna media; la cláusula de la actual sólo con un cociente actual, el del tramo más cercano con cociente).
  - `herd_context`: `initial = {'weeks': parte o {}, 'state': context['state']}`, `weeks_text`, `weeks_rows` (bool: hay filas de pasadas con cociente), `weeks_since` (año de la primera fila pasada).
  - Plantilla, tarjeta 2, entre `.lead` y `.herd-grid`, sólo con `weeks_rows`: `<p class="ctitle">Separación entre casas por semanas hasta la elección</p>`, `<p class="csub">Generales desde {{ weeks_since }}{% if has_run %} y {{ current.label }}, que se completa cada día{% endif %} · pares de sondeos de casas distintas con 7 días o menos de diferencia</p>`, `<div class="chart" id="weeks-chart" role="img" aria-label="{{ weeks_text }}"></div>` y el `.tnote` literal de la spec D3 (sin la frase de 2026 cuando no hay elección en curso).

- [ ] **Step 1: Write the failing tests**

`tests/test_js_charts.py` (datos pequeños a mano, dos tramos, una pasada con un `null`, la actual):
```python
def test_weeks_option_colours_each_election_and_marks_the_stretches():
    option = run_js('wk.weeksOption(%s)' % json.dumps(data))
    names = [s['name'] for s in option['series']]
    assert names == ['Generales 2016', 'Generales 2023', 'Media de las 2', 'Generales 2026']
    assert option['series'][0]['lineStyle']['color'] == '#e07b39' and option['series'][1]['lineStyle']['color'] == '#2a9d8f'
    assert option['series'][0]['connectNulls'] is True and option['series'][3]['lineStyle']['width'] == 3.5
    assert option['series'][2]['lineStyle']['type'] == 'dashed' and option['series'][2]['markLine']['data'][0]['yAxis'] == 1
    areas = option['series'][2]['markArea']['data']
    assert [a[0]['xAxis'] for a in areas] == ['5-7', '0-2'] and 'markArea' not in option['series'][3]
    assert option['xAxis']['data'] == ['27-52', '5-7', '3-4', '0-2'] and option['yAxis']['interval'] == 0.5
```

`tests/test_pages.py`:
```python
def test_herd_page_draws_the_weeks_chart_before_the_squares(api, fresh_app, monkeypatch):
    herd_today(monkeypatch, date(2026, 11, 6))
    publish_study(fresh_app.fs, study='rebano')
    status, _, body = call(api, '/analisis/rebano'); page = html(body)
    chart = page.index('id="weeks-chart"')
    assert chart < page.index('class="herd-squares"') and 'Separación entre casas por semanas hasta la elección' in page
    assert initial(body)['weeks']['buckets'][0]['label'] == '27-52'
    text = re.sub(r'\s+', ' ', visible_text(page))
    assert 'Generales desde 2008 y Generales 2026, que se completa cada día' in text
    assert 'La media de las 7 generales pasadas pasa de 1,65 a más de 27 semanas a 1,07 en las tres últimas; Generales 2026 va en 1,06 a 3-4 semanas.' in page
    assert 'Un tramo con menos de dos parejas de sondeos no se dibuja.' in text

def test_herd_page_without_weeks_rows_has_no_chart(api, fresh_app):
    publish_study(fresh_app.fs, study='rebano', parts={'weeks': {'rows': [], 'mean': []}})
    page = html(call(api, '/analisis/rebano')[2])
    assert 'weeks-chart' not in page and 'se completa cada día' not in page

def test_herd_weeks_text_without_a_current_election():
    # weeks of the fixture without the current rows → the sentence ends at «en las tres últimas.»
```
(`assert_public_wording` sobre la página, como en `test_herd_page_structure_and_wording`.)

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_js_charts.py -k weeks tests/test_pages.py -k "herd or rebano"`
Expected: FAIL

- [ ] **Step 3: Implement** `weeks.js`, el script de la página, `weeks_text`, el contexto, la plantilla y la altura en CSS.

- [ ] **Step 4: Run the page and JS tests, then the full suite**

Run: `python -m pytest -q tests/test_js_charts.py tests/test_pages.py`, después `python -m pytest -q`
Expected: todo en verde (base 2294 passed, 81 skipped)

- [ ] **Step 5: Check in the browser** con el patrón del scratchpad de la sesión (`check_app.py`: la web con el bundle de fixtures y el estudio publicado en un puerto local; `check_dl.mjs`: Chrome headless por CDP que abre `/analisis/rebano`, captura la tarjeta 2 a 1300 y 390 px y pulsa su botón de descarga guardando el PNG). Mirar las capturas y el PNG descargado: líneas de color, leyenda, media discontinua, 2026 azul con su etiqueta, las dos zonas, y que el gráfico está en la imagen descargada. Guardar los ficheros en el espacio de trabajo del plan y anotar en el informe lo visto.

- [ ] **Step 6: Commit**

```bash
git add web/dist/js/charts/weeks.js web/dist/js/pages/analisis-rebano.js web/templates/analisis_rebano.html mtpy/lib/pages/herd.py mtpy/lib/pages/analisis.py web/dist/css/site.css tests/test_js_charts.py tests/test_pages.py
git commit -m "Rebaño: el embudo por semanas en la pregunta 2 (una línea de color por elección, la media y 2026)"
```

### Task 4: Aceptación con datos reales y cierre

**Files:**
- Script en el scratchpad de la sesión (no se commitea).
- Modify: `docs/superpowers/specs/2026-11-07-web-analisis-rebano-embudo-design.md` (sección «Cierre»)

- [ ] **Step 1: Compute against the real bundle without publishing** (sólo lectura, como `v2_accept.py` del scratchpad): `compute_herd(...).weeks`.
Expected: `mean` ≈ 1,65 / 1,31 / 1,42 / 1,42 / 1,35 / 1,19 / 1,07 (de 27-52 a 0-2; diferencias de ±0,02 por el bundle del día se explican); 2026 con cociente de 27-52 a 3-4 y `None` en 0-2; 7 pasadas; tiempo total por debajo de 60 s.

- [ ] **Step 2: Render the page with the real parts** (patrón `v4_parts.py` + test temporal borrado después) y leer el texto de la tarjeta 2 y el aria-label con las cifras reales; captura a 1300 px.

- [ ] **Step 3: Write the «Cierre»** (qué existe, cifras reales, decisiones de la ejecución, lo no verificado) y commit:

```bash
git add docs/superpowers/specs/2026-11-07-web-analisis-rebano-embudo-design.md
git commit -m "Spec embudo del rebaño: cierre"
```
