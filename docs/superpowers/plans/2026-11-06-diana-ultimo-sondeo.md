# Diana del perfil sin tope de días y gráfico «frente a las demás» — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La diana del perfil de una encuestadora muestra su último sondeo en cada elección analizada, sin el tope de 42 días (hueco cuando es más antiguo), y a su lado el gráfico de puntos «frente a la media de las demás casas»; el histórico, la tarjeta del CIS y el estudio del sesgo no cambian ni una cifra.

**Architecture:** La parte `poll-errors` se publica sin tope superior de días (`drange (6, None)`); la ventana de la recta final pasa a una constante única `POLL_ERRORS_WINDOW = (6, 42)` en `mtpy/lib/bundle.py` que la publicación usa como mínimo y las páginas y el estudio como tope (`days <= 42`). El perfil usa todas las filas, marca `late` las de más de 42 días y añade a cada una la media del error de las demás casas en el bloque de izquierdas (`others`, `n_others`). En el navegador, `bullseyeOption` y `scatterOption` ganan `hollowOf` y `bullseye-wire.js` dibuja los dos gráficos con los mismos filtros.

**Tech Stack:** Python 3 (pandas, Jinja2), pytest; ES modules en `web/dist/js` con ECharts 5.6, probados con `node` desde `tests/test_js_charts.py`.

**Spec:** `docs/superpowers/specs/2026-11-06-diana-ultimo-sondeo-design.md`

## Global Constraints

- Identificadores en inglés neutro; textos públicos en español con «sólo» con tilde y «encuestadora/casa», nunca «empresa» (`assert_public_wording` en los tests de páginas).
- PEP8 y docstrings numpy en toda función nueva o tocada; JSDoc en el JS.
- El límite inferior de 6 días no cambia; `n_last = 1` no cambia; `drop_ctypes` por defecto (`wban`, `exit`) no cambia.
- Ninguna cifra de `/elecciones`, de la tarjeta «Un ejemplo: el CIS desde 2018» de `/metodo/promedio` ni del estudio `/analisis/sesgo-cis` puede cambiar: sus textos dicen «entre 6 y 42 días» y siguen siendo ciertos.
- No tocar la página de cada elección (`/elecciones/{scope}/{date}`): lee su parte `polls`, no la del análisis.
- Luis commitea en paralelo en el mismo árbol: `git status` y `git log -1` antes de cada commit; nunca reescribir sus commits. Mensajes de commit en español, con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Los tests nunca escriben en `backtest/results` ni con `out=None`.

## Review Focus

1. Una fila de `poll-errors` sin `days` (parte vieja o escrita a mano): el filtro de la recta final la deja fuera sin fallar; cubierto en la Tarea 2 (`final_stretch_rows`).
2. Una casa cuya única fila en una elección es `late` y sin otras casas en esa elección: `others` es `None`, `n_others` 0, y el gráfico de puntos no pinta ese punto pero la diana sí; cubierto en la Tarea 4 y la Tarea 5 (`scatterOption` ignora coordenadas no finitas).
3. Las filas `late` de las demás casas no entran en su media (la media es de la recta final): cubierto en la Tarea 4.
4. El perfil de una casa sin análisis o sin bloques (`es-an` con `blocks: []`): la tarjeta sigue sin diana y sin gráfico de puntos, sin error de plantilla; cubierto en la Tarea 6 (`houses_without_analysis`).
5. La página del backtest, que comparte `_bullseye.html` sin el gráfico de puntos: el cableado no falla cuando falta `#consensus-chart`; cubierto en la Tarea 6.

---

### Task 1: La ventana en una constante y la parte sin tope

**Files:**
- Modify: `mtpy/lib/bundle.py` (junto a `ANALYSIS_PARTS`, línea 69)
- Modify: `mtpy/lib/publish/analysis.py:34` y `:342`
- Modify: `docs/web/contrato.md:313`
- Test: `tests/test_publish_unit.py:925-940`

**Interfaces:**
- Produces: `bundle.POLL_ERRORS_WINDOW: tuple[int, int] = (6, 42)`; `publish.analysis.POLL_ERRORS_PARAMS == {'bmap': 'vs', 'drange': (6, None), 'n_last': 1}`; `meta.params.drange == [6, None]`.

- [ ] **Step 1: Cambiar el test existente**

En `test_publish_analysis_records_its_parameters_and_the_database` (línea 927): el docstring dice `drange=(6, None)`; la llamada esperada pasa a `{'drange': (6, None), 'n_last': 1, 'featured': 'results'}` y `meta['params']['drange'] == [6, None]`. Añadir al final:

```python
    assert bundle.POLL_ERRORS_WINDOW == (6, 42) and publish.POLL_ERRORS_PARAMS['drange'] == (bundle.POLL_ERRORS_WINDOW[0], None)
```

- [ ] **Step 2: Correr el test y verlo fallar**

Run: `pytest tests/test_publish_unit.py::test_publish_analysis_records_its_parameters_and_the_database -q`
Expected: FAIL (`(6, 42) != (6, None)`).

- [ ] **Step 3: Implementar**

En `mtpy/lib/bundle.py`, tras `ANALYSIS_PARTS`:

```python
# The final stretch of an election, in days before it: the pages and studies that speak of «the last poll of
# each house before the election» keep the rows of `poll-errors` within it (`days <= 42`); the part itself is
# published with its lower bound only (the ban: `wban` and `exit` polls are dropped apart), so the profile of a
# house can show its last poll of every election it polled (Luis, 2026-11-06).
POLL_ERRORS_WINDOW = (6, 42)
```

En `publish/analysis.py`: `POLL_ERRORS_PARAMS = {'bmap': 'vs', 'drange': (bundle.POLL_ERRORS_WINDOW[0], None), 'n_last': 1}` (importar `POLL_ERRORS_WINDOW` desde `..bundle`). La línea 342 (`'drange': list(...)`) sigue valiendo: `[6, None]`.

En `docs/web/contrato.md:313`: `drange=(6, None)` y una frase: «sin tope superior desde 2026-11-06: el último sondeo de cada casa en cada elección, aunque tenga más de 42 días; las páginas de la recta final filtran `days <= 42` (`bundle.POLL_ERRORS_WINDOW`)».

- [ ] **Step 4: Correr los tests de publicación**

Run: `pytest tests/test_publish_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish/analysis.py docs/web/contrato.md tests/test_publish_unit.py docs/superpowers/specs/2026-11-06-diana-ultimo-sondeo-design.md docs/superpowers/plans/2026-11-06-diana-ultimo-sondeo.md
git commit -m "poll-errors sin tope de días: ventana de la recta final en bundle.POLL_ERRORS_WINDOW, drange (6, None) y contrato"
```

---

### Task 2: El histórico y la tarjeta del CIS se quedan en la recta final

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py` (nueva función junto a `history_rows`, línea 319; `elecciones_context`, línea 513)
- Modify: `mtpy/lib/pages/metodo.py:380-384`
- Modify: `mtpy/lib/pages/__init__.py` (exportar `final_stretch_rows`)
- Test: `tests/test_pages.py` (junto a `test_elecciones_only_counts_the_published_elections`, línea 1886), `tests/test_pages_ejemplo.py` (junto a la línea 312)

**Interfaces:**
- Consumes: `bundle.POLL_ERRORS_WINDOW` (Tarea 1).
- Produces: `pages.elecciones.final_stretch_rows(rows: list) -> list`: las filas cuyo `days` es un número `<= POLL_ERRORS_WINDOW[1]`; una fila sin `days` queda fuera.

- [ ] **Step 1: Tests que fallan**

En `tests/test_pages.py`:

```python
def test_elecciones_history_keeps_only_the_final_stretch(history_site, fresh_app):
    """La parte `poll-errors` trae desde 2026-11-06 el último sondeo de cada casa aunque tenga más de 42 días; el
    histórico (filtros, histograma, diana y semanas) sigue hablando de la recta final."""
    errors = fixture_data('poll-errors')
    errors['rows'].append(dict(errors['rows'][0], pollster='SW Demoscopia', days=387))
    errors['rows'].append(dict(errors['rows'][0], pollster='Sin días', days=None))
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es')
    data = initial(call(history_site, '/elecciones')[2])
    assert [row['pollster'] for row in data['errors']] == ['GAD3', 'Sigma Dos']
    assert pages.elecciones.final_stretch_rows([{'days': 42}, {'days': 43}, {'days': None}, {}]) == [{'days': 42}]
```

En `tests/test_pages_ejemplo.py`, tras `test_house_example_leaves_out_the_elections_without_the_block_errors`:

```python
def test_house_example_keeps_only_the_final_stretch(house_site, fresh_app):
    """Un sondeo del CIS de más de 42 días no cuenta en el ejemplo: su texto dice «entre 6 y 42 días»."""
    errors = fixture_data('poll-errors')
    errors['rows'] = [dict(row, days=60) if row['pollster'] == 'CIS' and row['event_date'] == '2019-11-10' else row
                      for row in errors['rows']]
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es')
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    assert [row['event_date'] for row in metodo.house_effects_example()['rows']] == ['2023-07-23']
```

(Comprobar primero con `grep -n "CIS" tests/test_pages_ejemplo.py | head` qué fixture de `poll-errors` usa `house_site`, `HOUSE_ROWS`, y adaptar el nombre de la fixture y la fecha a las filas que realmente tiene el CIS; el test de la línea 273 espera `['2023-07-23', '2019-11-10']`.)

- [ ] **Step 2: Verlos fallar**

Run: `pytest tests/test_pages.py::test_elecciones_history_keeps_only_the_final_stretch tests/test_pages_ejemplo.py::test_house_example_keeps_only_the_final_stretch -q`
Expected: FAIL (`AttributeError: final_stretch_rows` / la fila de 387 días aparece).

- [ ] **Step 3: Implementar**

`final_stretch_rows(rows)` en `pages/elecciones.py` con docstring numpy; `elecciones_context`: `errors = final_stretch_rows(history_rows(scopes, events, parts))`; `house_effects_example`: envolver `history_rows(...)` en `final_stretch_rows`. Actualizar la frase del docstring de `error_rows` («the filters, the histogram, the bullseye and the weeks table speak of the same elections as the list») con «and of the final stretch only».

- [ ] **Step 4: Correr las páginas**

Run: `pytest tests/test_pages.py tests/test_pages_ejemplo.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/elecciones.py mtpy/lib/pages/metodo.py mtpy/lib/pages/__init__.py tests/test_pages.py tests/test_pages_ejemplo.py
git commit -m "Histórico y ejemplo del CIS: sólo la recta final (days <= 42) de poll-errors"
```

---

### Task 3: El estudio del sesgo del CIS se queda en la recta final

**Files:**
- Modify: `mtpy/lib/studies/lean/data.py:97-100`
- Test: `tests/test_lean_data.py` (tras `test_read_error_rows_keeps_a_missing_block_as_none`)

**Interfaces:**
- Consumes: `bundle.POLL_ERRORS_WINDOW`.
- Produces: `read_error_rows` devuelve sólo filas con `days` numérico `<= 42` (sin cambio de firma).

- [ ] **Step 1: Test que falla**

```python
def test_read_error_rows_keeps_only_the_final_stretch(catalogue, tmp_path):
    """La parte trae desde 2026-11-06 sondeos de más de 42 días; el estudio dice «entre 6 y 42 días» y los deja fuera."""
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    part = bundle.path_analysis('es', 'poll-errors')
    envelope = bundle.BundleReader(fs).read_json(part)
    envelope['data']['rows'].append(dict(envelope['data']['rows'][0], pollster='SW Demoscopia', days=387))
    bundle.BundleWriter(fs).write_json(part, 'poll-errors', envelope['data'], 'es')
    rows, _ = data.read_error_rows(bundle.BundleReader(fs), catalogue)
    assert 'SW Demoscopia' not in {row['pollster'] for row in rows}
```

- [ ] **Step 2: Verlo fallar**

Run: `pytest tests/test_lean_data.py::test_read_error_rows_keeps_only_the_final_stretch -q`
Expected: FAIL.

- [ ] **Step 3: Implementar**

En `read_error_rows`, al construir `scope_rows`, añadir la condición `isinstance(row.get('days'), (int, float)) and row['days'] <= POLL_ERRORS_WINDOW[1]` (importar de `...bundle`). Una frase en el docstring.

- [ ] **Step 4: Correr el estudio**

Run: `pytest tests/test_lean_data.py tests/test_lean_stats.py tests/test_lean_study.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/lean/data.py tests/test_lean_data.py
git commit -m "Estudio del sesgo del CIS: sólo la recta final de poll-errors"
```

---

### Task 4: Los datos del perfil: `late`, `others` y `n_others`

**Files:**
- Modify: `mtpy/lib/pages/casas.py` (`casa_context`, líneas 866-872 y 908-912; docstring)
- Test: `tests/test_pages.py:1564` y un test nuevo junto a `test_casa_profile_tables_and_bullseye`

**Interfaces:**
- Consumes: `bundle.POLL_ERRORS_WINDOW`, `final_stretch_rows` (Tarea 2), `table_blocks` (ya importado).
- Produces: cada fila de `context['bullseye']` (y de `initial['bullseye']`) lleva además `late: bool` (`days > 42`), `others: float | None` (media, a 2 decimales, del error en el bloque `left` de las demás casas de esa misma elección cuyas filas pasan `final_stretch_rows`; `None` sin ninguna) y `n_others: int`; `context['left']` y `initial['left']` = `table_blocks(blocks)[0]` (`None` sin bloques). Orden de `initial`: `state, meta, path, today, bullseye, blocks, left, scopes`.

- [ ] **Step 1: Tests**

Cambiar la aserción de la línea 1564 a:

```python
    assert data['bullseye'] == [{'event': '2023-23J', 'event_date': '2023-07-23', 'year': 2023, 'days': 7, 'scope': 'es',
                                 'name': 'Generales 2023', 'Derecha': 2.81, 'Izquierda': -0.24,
                                 'late': False, 'others': -0.51, 'n_others': 1}]
    assert data['left'] == 'Izquierda'
```

y la de la línea 1501 (`list(data)`) a `['state', 'meta', 'path', 'today', 'bullseye', 'blocks', 'left', 'scopes']`. Test nuevo:

```python
def test_casa_profile_bullseye_keeps_the_late_polls_and_averages_the_final_stretch_of_the_others(houses, fresh_app):
    """Desde 2026-11-06 la diana trae el último sondeo de cada elección aunque tenga más de 42 días (`late`); la media
    de las demás casas (`others`) sólo cuenta sus sondeos de la recta final."""
    errors = fixture_data('poll-errors')
    errors['rows'][0]['days'] = 60          # GAD3, tardío
    errors['rows'][1]['days'] = 50          # Sigma Dos, tardío: no entra en la media de las demás
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es')
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    [row] = initial(call(houses, '/casas/gad3')[2])['bullseye']
    assert row['late'] is True and row['days'] == 60 and row['others'] is None and row['n_others'] == 0
    [row] = initial(call(houses, '/casas/sigma-dos')[2])['bullseye']
    assert row['late'] is True and row['others'] is None
```

(Si `/casas/sigma-dos` no existe en la fixture de casas, dejar sólo GAD3; comprobar los slugs con `grep -n "sigma" tests/test_pages.py | head`.)

- [ ] **Step 2: Verlos fallar**

Run: `pytest tests/test_pages.py -q -k "casa_profile"`
Expected: FAIL por las claves nuevas.

- [ ] **Step 3: Implementar en `casa_context`**

Tras construir `by_election` (toda fila de `errors`, ya agrupada por `(scope, event_date)`): `left = table_blocks(blocks)[0] if blocks else None`; para cada fila de `bullseye`, `rest = [r[left] for r in final_stretch_rows(by_election[...]) if r.get('pollster') != name and r.get(left) is not None]` (vacío sin `left`); `row['others'] = round(sum(rest) / len(rest), 2)` o `None`; `row['n_others'] = len(rest)`; `row['late'] = row['days'] is not None and row['days'] > POLL_ERRORS_WINDOW[1]`. `context['left'] = left` e `initial['left']` detrás de `blocks`. Documentar las tres claves y `left` en el docstring (párrafo de `bullseye`).

- [ ] **Step 4: Correr las páginas**

Run: `pytest tests/test_pages.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/casas.py tests/test_pages.py
git commit -m "Perfil: filas de la diana con late, others y n_others; initial.left"
```

---

### Task 5: `hollowOf` en la diana y en la nube de puntos

**Files:**
- Modify: `web/dist/js/charts/bullseye.js` (`pointSeries`, `bullseyeOption`, JSDoc)
- Modify: `web/dist/js/charts/scatter.js` (`scatterOption`, JSDoc)
- Test: `tests/test_js_charts.py` (junto a `test_bullseye_draws_the_groups_on_top_over_the_others`, línea 105, y a `test_scatter_diagonal_and_fixed_axes`, línea 937)

**Interfaces:**
- Produces: `bullseyeOption(rows, {..., hollowOf = null})` y `scatterOption(rows, {..., hollowOf = null})`: `hollowOf(row) => boolean`; un punto hueco lleva `itemStyle: {color: 'transparent', borderColor: <color de su serie o de colorOf(row)>, borderWidth: 1.5}` (más la `opacity` de `alphaOf` en la diana). Los puntos no huecos no cambian. La gemela de etiquetas de la diana no cambia.

- [ ] **Step 1: Tests que fallan**

```python
def test_bullseye_draws_the_late_polls_hollow():
    """Un sondeo de más de 42 días antes (`late`, perfil desde 2026-11-06) va hueco: relleno transparente y borde del
    color de su serie; su etiqueta y su tooltip no cambian."""
    rows = [dict(ROWS[0], late=True), ROWS[1]]
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], groupOf: (r) => r.scope, '
                    'colors: {es: "#1d4f86"}, hollowOf: (r) => Boolean(r.late), alphaOf: () => 0.5})', rows=rows)
    points, labels = scatter(option)
    hollow, plain = points[0]['data']
    assert hollow['itemStyle'] == {'color': 'transparent', 'borderColor': '#1d4f86', 'borderWidth': 1.5, 'opacity': 0.5}
    assert plain['itemStyle'] == {'opacity': 0.5}
    assert [p['row']['event'] for p in labels[0]['data']] == ['2023-23J', '2019-10N'] and 'itemStyle' not in labels[0]['data'][0]


def test_scatter_draws_the_hollow_points_with_their_colour_as_border():
    rows = [{'x': 1, 'y': 2, 'name': 'a', 'late': True}, {'x': 2, 'y': 1, 'name': 'b'}]
    option = run_js('sc.scatterOption(rows, {x: "x", y: "y", name: "name", colorOf: () => "#c0672e", hollowOf: (r) => Boolean(r.late)})', rows=rows)
    hollow, plain = option['series'][0]['data']
    assert hollow['itemStyle'] == {'color': 'transparent', 'borderColor': '#c0672e', 'borderWidth': 1.5}
    assert plain['itemStyle'] == {'color': '#c0672e'}
```

- [ ] **Step 2: Verlos fallar**

Run: `pytest tests/test_js_charts.py -q -k "hollow"`
Expected: FAIL.

- [ ] **Step 3: Implementar**

`bullseye.js`: `styled(row)` devuelve también el `itemStyle` hueco cuando `hollowOf && hollowOf(row)`; para ello `pointSeries` necesita el color de la serie: pasar `hollowOf` a `pointSeries` y resolver el borde allí (`style.color`), fusionando con la `opacity` de `alphaOf`. `scatter.js`: en `data.map`, `itemStyle` hueco con `borderColor: colorOf ? colorOf(row) : THEME.accentColor`. Documentar `hollowOf` en los JSDoc de las dos opciones.

- [ ] **Step 4: Correr los tests JS**

Run: `pytest tests/test_js_charts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/bullseye.js web/dist/js/charts/scatter.js tests/test_js_charts.py
git commit -m "Gráficos: opción hollowOf en la diana y en la nube de puntos"
```

---

### Task 6: La tarjeta del perfil: dos gráficos, filtros, descarga y textos

**Files:**
- Modify: `web/templates/_bullseye.html`, `web/templates/casa.html:126-145`
- Modify: `web/dist/js/pages/bullseye-wire.js`, `web/dist/js/pages/casas.js:57`
- Modify: `web/dist/css/site.css` (junto a la línea 689)
- Test: `tests/test_pages.py` (`test_casa_profile_tables_and_bullseye`, `test_casa_profile_bullseye_offers_a_download_button_with_the_filters_in_js`, `test_casa_profile_without_analysis_has_no_bullseye`, y los del backtest que usan `_bullseye.html`: `grep -n "bullseye" tests/test_pages.py | grep -i backtest`)

**Interfaces:**
- Consumes: `initial.bullseye[].late/others/n_others`, `initial.left` (Tarea 4); `hollowOf` (Tarea 5); `wireDownload(button, optionOf, onError, {charts: () => [{option: (box) => echartsOption, height}, ...], chartHeight, texts})` de `share.js`.
- Produces: `wireBullseye(bullseye, blocks, {left = null} = {})`; la plantilla `_bullseye.html` admite `bv.consensus` (`{'house': nombre, 'left': bloque}` o ausente): con él, el gráfico `#bullseye-chart` y un `#consensus-chart` (con `data-house`) van en un `.grid2` con `.ctitle`/`.csub` cada uno; sin él, todo como hasta ahora (backtest).

- [ ] **Step 1: Tests**

En `test_casa_profile_tables_and_bullseye` añadir:

```python
    assert 'id="consensus-chart"' in page and 'data-house="GAD3"' in page
    grid = page[page.index('id="bullseye-chart"') - 400:page.index('id="consensus-chart"')]
    assert 'class="grid2"' in grid and 'Frente a la media de las demás encuestadoras en la misma elección' in page
    assert 'Un punto hueco es un sondeo de más de seis semanas antes' in page
```

En `test_casa_profile_without_analysis_has_no_bullseye`: `assert 'consensus-chart' not in page`. En los tests del backtest que incluyen la diana del modelo: `assert 'consensus-chart' not in page` (añadir en uno). Test nuevo:

```python
def test_casa_profile_card_texts_name_the_two_charts(houses):
    page = html(call(houses, '/casas/gad3')[2])
    assert 'Su último sondeo, por bloques' in page
    assert 'Horizontal: la media de las demás casas en sus últimos sondeos de las seis semanas finales. Vertical: GAD3.' in page
    assert_public_wording(page)
```

(Usar el helper de redacción que ya usen los tests de `casa.html`: `grep -n "assert_public_wording\|def html" tests/test_pages.py | head`.)

- [ ] **Step 2: Verlos fallar**

Run: `pytest tests/test_pages.py -q -k "casa_profile"`
Expected: FAIL.

- [ ] **Step 3: Plantillas y textos**

`casa.html`: en `bv` añadir `'consensus': {'house': profile.pollster.name, 'left': left}` sólo cuando `left`; subtítulo de la tarjeta: «Cuánto falló su último sondeo antes de cada elección analizada, de cualquier ámbito: a la izquierda, en cada bloque; a la derecha, en el bloque de izquierdas frente a la media de las demás encuestadoras en esa misma elección. Elige un ámbito o unos años para ver sólo esas elecciones.» «Cómo leerlo»: a la frase de la diana añadir «Un punto hueco es un sondeo de más de seis semanas antes de la votación: el último que publicó, aunque no fuera de la recta final.» y un párrafo nuevo: «En el gráfico de la derecha cada punto es también una elección: en horizontal, la media del error de las demás encuestadoras en el bloque de izquierdas con sus últimos sondeos de las seis semanas finales; en vertical, el de la encuestadora. Sobre la diagonal, dio a ese bloque más voto que las demás; debajo, menos.»

`_bullseye.html`: si `bv.consensus`, envolver en `<div class="grid2">` dos columnas: la primera con `<p class="ctitle">Su último sondeo, por bloques</p>`, `<p class="csub">Horizontal, error en {{ bv.table_blocks[1] }}; vertical, en {{ bv.table_blocks[0] }}, en puntos. Un punto hueco es un sondeo de más de seis semanas antes.</p>` y el `#bullseye-chart` (sin la clase `big`); la segunda con `<p class="ctitle">Frente a la media de las demás encuestadoras en la misma elección</p>`, `<p class="csub">Error en {{ bv.consensus.left }}, cada punto una elección. Horizontal: la media de las demás casas en sus últimos sondeos de las seis semanas finales. Vertical: {{ bv.consensus.house }}. Sobre la diagonal, dio a {{ bv.consensus.left }} más voto que las demás.</p>` y `<div class="chart" id="consensus-chart" role="img" aria-label="Error de {{ bv.consensus.house }} en {{ bv.consensus.left }} frente a la media de las demás encuestadoras, por elección" data-house="{{ bv.consensus.house }}"></div>`. Sin `bv.consensus`, el bloque actual sin cambios. Comprobar en `_bullseye.html` el orden de los ejes: la diana pinta `blocks` como se publican (Derecha en x), la tabla `table_blocks` (Izquierda primero); el `csub` de la diana usa el orden de los ejes (`bv.blocks` si la plantilla lo recibe; si no, añadirlo a `bv` en `casa.html`).

CSS, junto a la línea 689: `#bullseye .grid2 .chart { height: auto; aspect-ratio: 1 / 1; max-width: 480px; margin: 0 auto; }`.

- [ ] **Step 4: Cableado**

`bullseye-wire.js`: `wireBullseye(bullseye, blocks, {left = null} = {})`; `options.hollowOf = (row) => Boolean(row.late)`; `const consensus = document.getElementById('consensus-chart')`; `scatterOptions(rows)`: filas con `Number.isFinite(row.others) && Number.isFinite(row[left])`, `L = Math.max(5, Math.ceil(max(|others|, |row[left]|) / 5) * 5)`, `scatterOption(rows, {x: 'others', y: left, name: 'label', labelled: rows.map(r => r.label), diagonal: true, axes: {xMin: -L, xMax: L, yMin: -L, yMax: L, interval: 5}, colorOf: row => row.scope === 'es' ? THEME.accentColor : REGIONAL_COLOR, hollowOf: row => Boolean(row.late), xName: 'Media de las demás casas, error en ' + left, xLabel: 'demás casas', yLabel: consensus.dataset.house})` con `label = `${row.event}-${row.scope}`` añadido a cada fila (como `labelOf` de la diana). `draw(rows)` dibuja también la nube cuando `consensus && left` (`tryDraw(..., 'the house against the others')`). Descarga: `wireDownload(button, null, showError, {chartHeight: BULLSEYE_IMAGE_HEIGHT, charts: () => consensus && left ? [{option: () => bullseyeOption(current.rows, options), height: BULLSEYE_IMAGE_HEIGHT}, {option: () => scatterOption(...current...), height: BULLSEYE_IMAGE_HEIGHT}] : [{option: () => bullseyeOption(current.rows, options), height: BULLSEYE_IMAGE_HEIGHT}], texts})` (leer `wireDownload` en `share.js:429` y la forma de `charts`/`imageLayout` antes; si `optionOf` no admite `null` con `charts`, pasar la de la diana). `casas.js`: `const {..., left = null} = readInitial(); wireBullseye(bullseye, blocks, {left});`. Importar `scatterOption` en el wire.

- [ ] **Step 5: Correr las páginas y los tests JS**

Run: `pytest tests/test_pages.py tests/test_js_charts.py -q`
Expected: PASS.

- [ ] **Step 6: Comprobar en el navegador**

Arrancar la web local como en `project-web-descargas-imagenes` (scripts `check_app.py` + `check.mjs` en la memoria) o con el `run` de la app; abrir `/casas/sw-demoscopia` (o la casa que tenga el bundle local) y comprobar: dos gráficos lado a lado en escritorio, apilados a 700 px, puntos huecos donde `late`, los filtros redibujan los dos, la descarga produce una imagen con los dos gráficos, la página del backtest sigue mostrando su diana sin el segundo gráfico. Sin bundle local republicado, los puntos `late` no existirán aún: comprobar al menos que nada falla y que el gráfico de puntos sale.

- [ ] **Step 7: Commit**

```bash
git add web/templates/_bullseye.html web/templates/casa.html web/dist/js/pages/bullseye-wire.js web/dist/js/pages/casas.js web/dist/css/site.css tests/test_pages.py
git commit -m "Perfil: diana y «frente a las demás casas» lado a lado, puntos huecos para los sondeos tardíos, descarga con los dos"
```

---

### Task 7: Verificación final y entrega

**Files:**
- Modify: memoria del proyecto (`project-web-eleccion-promedio-modelo.md` o fichero nuevo `project-diana-ultimo-sondeo.md` en `/Users/luiss/.claude/projects/-Users-luiss-HD-Proyectos-Code-elections-model/memory/` + línea en `MEMORY.md`)

- [ ] **Step 1: Suite completa**

Run: `pytest -q -x`
Expected: todo PASS (unos 2.300 tests; los de integración se saltan sin base).

- [ ] **Step 2: `git status` limpio salvo lo de Luis; `git log --oneline -8` con los seis commits**

- [ ] **Step 3: Memoria y recap para Luis**

Guardar en memoria: lo construido, que la web no cambia hasta republicar el análisis (`python job.py publish '{"what":["analysis"],"scopes":"all"}'` o el pipeline `update`), y que `es-an`, `es-ar`, `es-cl` y `es-ex` siguen sin bloques (esas elecciones no entran en la diana ni en el gráfico nuevo hasta republicarlos con bloques). En el recap: qué republicar y en qué orden, y qué páginas no cambian.
