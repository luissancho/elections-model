# Tabla «Todas las encuestas»: fechas de campo, columna «Peso» y desglose — plan de implementación

> **Revertido en parte el 2026-11-07.** Luis retiró la columna «Peso» y su tooltip el mismo día («prefiero no incluirlo»), y con ellos la enmienda de `polls`/`event-cycle` que sólo les servía (D2, D3 y D4). En `dev` quedan la celda «Fecha» con los días de campo (D1, commit 0e4148c) y el redondeo de `fmt_num` de la revisión final; el resto lo deshace el commit de reversión que sigue a d1f0e1e.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La celda «Fecha» de la tabla de sondeos enseña los días del trabajo de campo, una columna «Peso» con barra y cifra sigue al rating, y al pasar por la cifra un tooltip desglosa el peso por factores; las partes `polls` y `event-cycle` publican los factores.

**Architecture:** La tabla se pinta dos veces con el mismo texto: el servidor (`pages/promedios.py` + `web/templates/_polls_table.html`) renderiza las 15 filas recientes y el navegador (`web/dist/js/polls-table.js`, `pages/polls-dom.js`, `pages/polls-wire.js`) repinta al filtrar. Cada helper nuevo tiene un gemelo en Python y en JS (`fieldwork_label`/`fieldworkLabel`, `weight_breakdown`/`weightBreakdown`, `bar_width`/`barWidth`). El desglose llega como cuatro columnas nuevas de `POLL_COLUMNS` (`publish/forecast.py`), que el ciclo de una elección pasada rellena a medias (`publish/events.py`).

**Tech Stack:** Python 3 (pandas, Jinja2), JavaScript ES modules sin dependencias en `web/dist/js`, pruebas con pytest y `node` (`tests/test_js_polls.py`, se saltan sin `node`).

**Spec:** `docs/superpowers/specs/2026-11-07-web-tabla-sondeos-campo-peso-design.md`

## Global Constraints

- PEP8, nombres en inglés neutro (`weight_disp`, no calcos del español), docstring numpy en toda función nueva.
- Redacción pública en español: «encuestadora» o «casa», nunca «empresa»; «sólo» con tilde; lo comprueba `assert_public_wording`.
- Meses de la celda con mayúscula y «Sep» (`MONTH_LABELS` de `promedios.py`, `MONTHS` de `polls-table.js`); guión corto «–» (U+2013) sin espacios.
- El DOM del navegador se construye con `createElement`/`textContent`, nunca HTML con datos; el servidor y el navegador escriben el mismo texto y las mismas clases.
- Los runs ya publicados no traen las columnas nuevas: toda función tolera la clave ausente o nula.
- `weight` de `event-cycle` no cambia de definición (`weight_over · weight_sample`).
- Pruebas: nunca escribir en `backtest/results` ni con `out=None`; Luis commitea en paralelo, mirar `git status` antes de cada commit.

## Review Focus

- Una fecha con hora (`'2026-10-30T00:00:00'`) o una fecha ISO inválida en `start_date`: la primera se etiqueta, la segunda cuenta como ausente (Task 1).
- `weight_disp` nulo, cero o ausente con `weight_sample` presente: «Muestra» es `weight_sample` sin dividir, sin excepción ni `Infinity` (Tasks 3 y 4).
- Todos los sondeos con peso 0 o nulo: sin barras y sin fallo, la cifra sigue (Tasks 3 y 4).
- Parte antigua sin las claves de los factores: columna Peso con barra y cifra, celda sin `title` (Tasks 3 y 4).
- Orden por Peso con nulos y ceros: nulos al final en los dos sentidos, los ceros ordenan como números (Task 4).

---

### Task 1: Fechas de campo en la celda «Fecha»

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`short_date` → `fieldwork_label`, `polls_table`)
- Modify: `mtpy/lib/pages/__init__.py:49-51` (exporta `fieldwork_label`, deja de exportar `short_date`)
- Modify: `web/dist/js/polls-table.js` (`shortDate` → `fieldworkLabel`)
- Modify: `web/dist/js/pages/polls-dom.js` (import, `renderBody`, título de la cabecera «Fecha» en `renderHead`)
- Modify: `web/templates/_polls_table.html` (cabecera «Fecha» con `title`)
- Modify: `web/dist/css/site.css:770` (`col.c-d` a 104 px)
- Test: `tests/test_pages.py` (560, 585, 650, 2249, 5425 y una prueba nueva), `tests/test_js_polls.py` (93 y una nueva)

**Interfaces:**
- Produces: `fieldwork_label(start, end, date=None) -> str` en `promedios.py` (fechas ISO, `datetime.date` o `None`, como `_parse_datetime`); `fieldworkLabel(start, end, date = null)` exportada por `polls-table.js`.

- [ ] **Step 1: Escribir la prueba Python de la etiqueta**

En `tests/test_pages.py`, junto a las de `polls_table`:

```python
def test_fieldwork_label_writes_the_days_of_the_fieldwork_without_the_year():
    assert pages.fieldwork_label('2026-10-29', '2026-10-30') == '29–30 Oct'
    assert pages.fieldwork_label('2026-09-01', '2026-11-01') == '1 Sep–1 Nov'
    assert pages.fieldwork_label('2026-10-30', '2026-10-30') == '30 Oct'
    assert pages.fieldwork_label('2026-12-28', '2027-01-03') == '28 Dic–3 Ene'
    assert pages.fieldwork_label(None, '2026-10-30') == '30 Oct' and pages.fieldwork_label('2026-10-28', None) == '28 Oct'
    assert pages.fieldwork_label(None, None, '2026-10-30') == '30 Oct'
    assert pages.fieldwork_label(None, None, '2026-10-30T00:00:00') == '30 Oct'
    assert pages.fieldwork_label('nope', None, None) == '–' and pages.fieldwork_label(None, None, None) == '–'
```

Y actualizar las aserciones existentes: línea 560 `first['date'] == '28–30 Sep'`; línea 585 `[('CIS', '1 Oct'), ('GAD3', '1 Oct'), ('CIS', '5 Sep'), ('–', '–')]`; línea 650 la cabecera pasa a `<th scope="col" data-key="date" aria-sort="descending" title="Días del trabajo de campo del sondeo"><button type="button" class="sort">Fecha</button></th>`; línea 2249 `'12–15 Jul' in page`; línea 5425 `rows[0].index('1 Oct')`.

- [ ] **Step 2: Comprobar que falla**

Run: `pytest tests/test_pages.py -k "fieldwork_label or polls_table or eleccion_detail_has_the_polls_table or rating_badge_between" -q`
Expected: FAIL (`fieldwork_label` no existe; las fechas siguen en `dd/mm/aa`).

- [ ] **Step 3: Implementar `fieldwork_label` y usarla en `polls_table`**

En `promedios.py`, `def fieldwork_label(start, end, date=None) -> str`: parsea las tres con `_parse_datetime`; con inicio y fin válidos escribe `'{d} {Mes}'` si es el mismo día, `'{d1}–{d2} {Mes}'` si es el mismo mes y año, y `'{d1} {Mes1}–{d2} {Mes2}'` en otro caso, con `MONTH_LABELS`; con una sola válida, esa; sin ninguna, `date`; sin nada válido, `DASH`. `polls_table` escribe `'date': fieldwork_label(start, end, poll.get('date'))` y su docstring dice «`'date'` (los días del campo, `fieldwork_label`)». Borrar `short_date` y quitarla de `pages/__init__.py`; exportar `fieldwork_label`.

- [ ] **Step 4: Cabecera y anchura**

En `_polls_table.html` la cabecera «Fecha» gana `title="Días del trabajo de campo del sondeo"` (antes de `><button`, tras `aria-sort`). En `site.css`, `.et col.c-d { width: 104px; }` con el comentario «la etiqueta más larga, «30 Sep–31 Oct»».

- [ ] **Step 5: Correr las pruebas Python**

Run: `pytest tests/test_pages.py tests/test_pages_ejemplo.py -q`
Expected: PASS.

- [ ] **Step 6: Escribir la prueba JS**

En `tests/test_js_polls.py`, sustituir la aserción de `shortDate` (línea 93) por:

```python
def test_fieldwork_label_of_the_polls_table_matches_the_server():
    assert table_js('p.fieldworkLabel("2026-10-29", "2026-10-30")') == '29–30 Oct'
    assert table_js('p.fieldworkLabel("2026-09-01", "2026-11-01")') == '1 Sep–1 Nov'
    assert table_js('p.fieldworkLabel("2026-10-30", "2026-10-30")') == '30 Oct'
    assert table_js('p.fieldworkLabel("2026-12-28", "2027-01-03")') == '28 Dic–3 Ene'
    assert table_js('p.fieldworkLabel(null, "2026-10-30")') == '30 Oct' and table_js('p.fieldworkLabel("2026-10-28", null)') == '28 Oct'
    assert table_js('p.fieldworkLabel(null, null, "2026-10-30T00:00:00")') == '30 Oct'
    assert table_js('p.fieldworkLabel("nope", null, null)') == '–' and table_js('p.fieldworkLabel(undefined, undefined)') == '–'
```

- [ ] **Step 7: Comprobar que falla**

Run: `pytest tests/test_js_polls.py -k fieldwork -q`
Expected: FAIL (`fieldworkLabel` no está exportada).

- [ ] **Step 8: Implementar `fieldworkLabel` y usarla en el DOM**

En `polls-table.js`, `export function fieldworkLabel(start, end, date = null)` con las mismas reglas; parsear con la expresión `/^(\d{4})-(\d{2})-(\d{2})/` (como hacía `shortDate`, sin `Date`), un mes fuera de 1-12 cuenta como ausente. Borrar `shortDate`. En `polls-dom.js`: importar `fieldworkLabel`, `const date = el('td', fieldworkLabel(poll.start_date, poll.end_date, poll.date));` y en `renderHead` dar a la columna `date` el título `'Días del trabajo de campo del sondeo'` (constante `DATE_TITLE`, como `RATING_TITLE`).

- [ ] **Step 9: Correr las pruebas JS**

Run: `pytest tests/test_js_polls.py -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git status
git add mtpy/lib/pages/promedios.py mtpy/lib/pages/__init__.py web/dist/js/polls-table.js web/dist/js/pages/polls-dom.js web/templates/_polls_table.html web/dist/css/site.css tests/test_pages.py tests/test_js_polls.py
git commit -m "Tabla de sondeos: la celda Fecha enseña los días del trabajo de campo («29–30 Oct», «1 Sep–1 Nov»), en el servidor y en el navegador

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Enmienda de `polls` y `event-cycle`: los factores del peso

**Files:**
- Modify: `mtpy/lib/publish/forecast.py:32-33` (`POLL_COLUMNS`), `export_poll_rows` (~línea 403), `export_polls` (~línea 416)
- Modify: `mtpy/lib/publish/events.py` (`export_event_cycle`, ~línea 100)
- Modify: `docs/web/contrato.md:58,137-152,436-442`
- Test: `tests/test_publish_unit.py:197-240,1588-1596`

**Interfaces:**
- Produces: `POLL_COLUMNS` con 14 nombres, los cuatro últimos `weight_sample, weight_disp, weight_over, weight_rating`; `WEIGHT_FACTORS = ('weight_sample', 'weight_disp', 'weight_over', 'weight_rating')` en `publish/forecast.py` (exportada por `mtpy.lib.publish`). Cada registro de `polls` y de `event-cycle` lleva las cuatro claves, con número a 3 decimales o `null`.

- [ ] **Step 1: Escribir las pruebas**

En `test_export_polls_keeps_the_published_figures_and_the_previous_result` añadir:

```python
    assert publish.POLL_COLUMNS[-4:] == ['weight_sample', 'weight_disp', 'weight_over', 'weight_rating']
    assert [p['weight_disp'] for p in data['polls']] == [0.95, 1.0, 0.95, 1.0, 1.0, 0.95]  # `dispersion.factor` por casa
    assert first['weight_sample'] is None and first['weight_over'] is None and first['weight_rating'] is None
```

Prueba nueva:

```python
def test_export_polls_publishes_the_weight_factors_of_the_series_and_one_without_dispersion():
    sim = synthetic_simulator()
    raw = sim.model.fc_series_raw
    sim.model.fc_series_raw = raw.assign(weight_over=1.0, weight_sample=raw['weight'] * 1.23456, weight_rating=0.5)
    data = valid('polls', publish.export_polls(sim.model, NAMES)[0], mode=None)
    first = data['polls'][0]
    assert (first['weight_sample'], first['weight_disp'], first['weight_over'], first['weight_rating']) == (1.481, 0.95, 1.0, 0.5)
    sim.model.dispersion = None
    data = valid('polls', publish.export_polls(sim.model, NAMES)[0], mode=None)
    assert {p['weight_disp'] for p in data['polls']} == {1.0}
```

En `test_export_event_cycle_keeps_the_shape_of_the_run_polls_and_skips_unknown_parties` añadir:

```python
    cis = data['polls'][0]
    assert (cis['weight'], cis['weight_sample'], cis['weight_over']) == (0.568, 0.7, 0.811)
    assert cis['weight_disp'] is None and cis['weight_rating'] is None
    rated, _ = publish.export_event_cycle(synthetic_cycle().assign(weight_rating=2.0), ['PP'])
    assert rated['polls'][0]['weight_rating'] is None and rated['polls'][0]['weight'] == 0.568
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_publish_unit.py -k "export_polls or export_event_cycle" -q`
Expected: FAIL (`KeyError: 'weight_disp'`).

- [ ] **Step 3: Implementar la enmienda**

`POLL_COLUMNS` gana las cuatro columnas al final; `WEIGHT_FACTORS` las nombra. En `export_poll_rows`, redondear `['weight', *WEIGHT_FACTORS]` con `astype(float).round(PROB_DECIMALS)` (una columna ausente, creada a `None`, queda nula). En `export_polls`, antes de `export_poll_rows`: `polls = fc.fc_series_raw.reset_index()` y `polls['weight_disp'] = polls['pollster_id'].map(dispersion['factor']).fillna(1.0)` cuando `getattr(fc, 'dispersion', None)` es un DataFrame con `factor`, si no `1.0`. En `export_event_cycle`, tras calcular `weight`: `table['weight_disp'] = None` y `table['weight_rating'] = None` (sobrescribe el `weight_rating` del frame). Añadir `WEIGHT_FACTORS` al `from .forecast import ...` de `mtpy/lib/publish/__init__.py`.

- [ ] **Step 4: Correr las pruebas de publicación**

Run: `pytest tests/test_publish_unit.py tests/test_bundle_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Contrato**

En `docs/web/contrato.md`: en `### polls` la lista pasa a 14 columnas y una «Enmienda 2026-11-07» explica `weight_sample` (con la dispersión de la casa dentro, como en `weights`), `weight_disp` (factor de dispersión de la casa, 1 sin él), `weight_over`, `weight_rating`, `weight = weight_sample · weight_over · weight_rating`, la muestra sin dispersión `weight_sample / weight_disp`, y que los runs anteriores no las tienen; en «Convenciones» (línea 58) los factores van a 3 decimales como `weight`; en el bullet de `cycle` (línea 436), `weight_over` y `weight_sample` por separado y `weight_disp`/`weight_rating` nulos (enmienda 2026-11-07).

- [ ] **Step 6: Commit**

```bash
git status
git add mtpy/lib/publish/forecast.py mtpy/lib/publish/events.py mtpy/lib/publish/__init__.py docs/web/contrato.md tests/test_publish_unit.py
git commit -m "Partes polls y event-cycle: los cuatro factores del peso de cada sondeo (weight_sample, weight_disp, weight_over, weight_rating; enmienda 2026-11-07 del contrato)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Columna «Peso» y desglose en el servidor

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`chart_polls`, `polls_table`, nuevo `weight_breakdown`)
- Modify: `mtpy/lib/pages/__init__.py` (exporta `weight_breakdown`)
- Modify: `web/templates/_polls_table.html` (colgroup, cabecera, celda, pie), `web/dist/css/site.css` (`col.c-w`)
- Modify: `web/templates/promedios.html:472-480`, `web/templates/eleccion.html:442-450` (párrafos de lectura)
- Modify: `tests/fixtures/bundle/event-cycle.json` (factores de los tres sondeos)
- Test: `tests/test_pages.py` (534-536, 549-575, 5405-5430, 5433-5441 y pruebas nuevas)

**Interfaces:**
- Consumes: claves `weight`, `weight_sample`, `weight_disp`, `weight_over`, `weight_rating` de los registros (Task 2); `bar_width(x, vmin, vmax)` y `fmt_num(x, 2)` de `filters.py`.
- Produces: `weight_breakdown(poll: dict) -> Optional[str]` y `WEIGHT_TITLE` (ambos exportados por `pages/__init__.py`); filas de `polls_table` con `'weight'` (str), `'weight_bar'` (float, 0-100 a un decimal), `'weight_title'` (str o `None`), `'weight_zero'` (bool); `chart_polls` con las cinco claves del peso; texto de la cabecera `WEIGHT_TITLE` (constante en `promedios.py`, la misma cadena en la plantilla y en el JS de Task 4): «Peso del sondeo en el promedio: muestra × dispersión × frecuencia × rating de la encuestadora, sin la cercanía al día estimado. La barra va de 0 al mayor peso del ciclo.»

- [ ] **Step 1: Pruebas de `weight_breakdown`**

```python
def test_weight_breakdown_lists_the_factors_the_poll_carries():
    full = {'weight_sample': 1.339, 'weight_disp': 0.95, 'weight_over': 1.0, 'weight_rating': 2.4}
    assert pages.weight_breakdown(full) == 'Muestra ×1,41\nDispersión ×0,95\nFrecuencia ×1,00\nRating ×2,40'
    cycle = {'weight_sample': 0.7, 'weight_disp': None, 'weight_over': 0.811, 'weight_rating': None}
    assert pages.weight_breakdown(cycle) == 'Muestra ×0,70\nFrecuencia ×0,81'
    assert pages.weight_breakdown({'weight_sample': 1.2, 'weight_over': 0.0, 'weight_rating': 1.5}) == 'Muestra ×1,20\nFrecuencia ×0,00\nRating ×1,50'
    assert pages.weight_breakdown({'weight_sample': 0.5, 'weight_disp': 0}) == 'Muestra ×0,50\nDispersión ×0,00'
    assert pages.weight_breakdown({}) is None and pages.weight_breakdown({'weight': 1.7}) is None
    assert pages.weight_breakdown({'weight_sample': 'x', 'weight_disp': True}) is None
```

- [ ] **Step 2: Pruebas de `polls_table` y `chart_polls`**

En `test_polls_table_marks_the_leader_formats_the_values_and_averages_all_the_polls`, dar a los sondeos `'weight': 2.0` (GAD3, más los cuatro factores de `full`), `'weight': 0.5` (CIS, sin factores) y `'weight': 4.0` (40dB) y añadir:

```python
    assert first['weight'] == '2,00' and first['weight_bar'] == 50.0 and first['weight_zero'] is False  # la barra sobre todos los sondeos
    assert first['weight_title'] == 'Muestra ×1,41\nDispersión ×0,95\nFrecuencia ×1,00\nRating ×2,40'
    assert second['weight'] == '0,50' and second['weight_bar'] == 12.5 and second['weight_title'] is None
    assert empty['rows'][0]['weight'] == '4,00' and empty['rows'][0]['weight_bar'] == 100.0
    zero = pages.polls_table([{'date': '2026-10-01', 'weight': 0, 'PP': 1.0}, {'date': '2026-10-02', 'PP': 2.0}], ['PP'], {})
    assert [(r['weight'], r['weight_bar'], r['weight_zero']) for r in zero['rows']] == [('–', 0.0, False), ('0,00', 0.0, True)]
```

En `test_promedios_initial_polls_carry_the_chart_and_table_columns`, `extra` gana `'weight', 'weight_sample', 'weight_disp', 'weight_over', 'weight_rating'`.

- [ ] **Step 3: Comprobar que fallan**

Run: `pytest tests/test_pages.py -k "weight_breakdown or polls_table_marks or initial_polls_carry" -q`
Expected: FAIL.

- [ ] **Step 4: Implementar en `promedios.py`**

`weight_breakdown(poll)`: factores en orden («Muestra», «Dispersión», «Frecuencia», «Rating») con valor finito no booleano (`_is_missing` y `isinstance(bool)`); «Muestra» es `weight_sample / weight_disp` si `weight_disp` es un número `> 0`, si no `weight_sample`; líneas `'{} ×{}'.format(label, fmt_num(value, 2))` unidas por `'\n'`; `None` sin factores. `chart_polls`: `keys` gana `'weight', 'weight_sample', 'weight_disp', 'weight_over', 'weight_rating'` tras `'rating'` (docstring al día). `polls_table`: `wmax` = mayor `weight` numérico y `> 0` de `polls` (o `None`); por fila `weight = poll.get('weight')`, `'weight': fmt_num(weight, 2)`, `'weight_bar': bar_width(weight, 0, wmax) if wmax else 0.0`, `'weight_title': weight_breakdown(poll)`, `'weight_zero': weight == 0` (sólo números); docstring al día. Exportar `weight_breakdown` en `pages/__init__.py`.

- [ ] **Step 5: Correr**

Run: `pytest tests/test_pages.py -k "weight_breakdown or polls_table_marks or initial_polls_carry" -q`
Expected: PASS.

- [ ] **Step 6: Pruebas de la página**

En `test_promedios_polls_table_shows_the_rating_badge_between_the_house_and_the_date_and_links_the_profile`, los tres sondeos ganan `{'weight': 3.384, 'weight_sample': 1.339, 'weight_disp': 0.95, 'weight_over': 1.0, 'weight_rating': 2.4}`, `{'weight': 0.31, 'weight_sample': 0.31, 'weight_disp': 1.0, 'weight_over': 1.0, 'weight_rating': 1.0}` y `{'weight': 0}`; la aserción del `colgroup` pasa a `'<col class="c-f"><col class="c-r"><col class="c-w"><col class="c-d"><col class="c-s">'` y se añaden:

```python
    assert '<th scope="col" data-key="weight" title="{}"><button type="button" class="sort">Peso</button></th>'.format(pages.WEIGHT_TITLE) in head
    assert ('<td class="cell-bar" title="Muestra ×1,41 Dispersión ×0,95 Frecuencia ×1,00 Rating ×2,40">'
            '<i class="fill w" style="left: 0; width: 100.0%"></i><span>3,38</span></td>') in rows[0]  # el \n del title es un espacio en `page`
    assert rows[0].index('class="rt"') < rows[0].index('cell-bar') < rows[0].index('1 Oct')
    assert '<i class="fill w" style="left: 0; width: 9.2%"></i><span>0,31</span>' in rows[1]
    assert '<td class="cell-bar na"><i class="fill w" style="left: 0; width: 0.0%"></i><span>0,00</span></td>' in rows[2]
    assert initial(body)['polls']['polls'][0]['weight_disp'] == 0.95
    assert 'mayor peso del ciclo' in page and 'los días del trabajo de campo' in page
```

y `foot.count('<td></td>') == 4` (comentario: «rating, peso, fecha y muestra»). En `tests/fixtures/bundle/event-cycle.json`, `columns` pasa a las 14 de `POLL_COLUMNS` y cada sondeo gana `weight_sample`/`weight_over` (CIS 0.7/0.811, 40dB 0.9/0.903, GAD3 1.1/1.122) y `weight_disp`/`weight_rating` a `null`. En `test_eleccion_cycle_table_shows_the_rating_each_house_arrived_with_and_links_its_profile`:

```python
    assert 'title="Muestra ×1,10 Frecuencia ×1,12"' in tbody and '<span>1,24</span>' in tbody
    assert 'width: 100.0%' in tbody and 'width: 46.0%' in tbody  # GAD3 1,235 es el mayor; CIS 0,568
    assert data[0]['weight_over'] == 0.811 and 'el rating de entonces va aparte' in page
```

Prueba nueva, con el fixture tal cual (sin rating, sin peso ni factores):

```python
def test_promedios_polls_table_without_weights_shows_a_dash_in_the_weight_cell(api):
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    row = re.findall(r'<tr>(.*?)</tr>', polls_tbody(page))[0]
    assert row.count('<td class="na">—</td>') == 2 and 'cell-bar' not in row  # rating y peso
    assert row.rindex('<td class="na">—</td>') < row.index('5 Oct')
    foot = page[page.index('<tfoot>'):page.index('</tfoot>')]
    assert foot.count('<td></td>') == 4
```

- [ ] **Step 7: Comprobar que fallan**

Run: `pytest tests/test_pages.py -k "rating_badge_between or cycle_table_shows or polls_table" -q`
Expected: FAIL.

- [ ] **Step 8: Plantilla, CSS y párrafos**

`_polls_table.html`: `<col class="c-w">` tras `c-r`; cabecera `<th scope="col" data-key="weight" title="…"><button type="button" class="sort">Peso</button></th>` tras el rating, con el texto de `WEIGHT_TITLE` escrito literal en la plantilla, como ya está el del rating (la prueba comprueba que coincide con la constante); celda: `{% if row.weight == '–' %}<td class="na">—</td>{% else %}<td class="cell-bar{% if row.weight_zero %} na{% endif %}"{% if row.weight_title %} title="{{ row.weight_title }}"{% endif %}><i class="fill w" style="left: 0; width: {{ row.weight_bar }}%"></i><span>{{ row.weight }}</span></td>{% endif %}`; un `<td></td>` más en la fila de la media y en las fijas. `site.css`: `.et col.c-w { width: 96px; }`. Párrafos: en los dos primeros «su rating, la fecha de publicación, la muestra» pasa a «su rating, su peso, los días del trabajo de campo, la muestra»; `promedios.html` añade el párrafo «El peso es lo que el sondeo pesa en el promedio del modelo: el producto de su muestra, de la dispersión y la frecuencia de su encuestadora y de su rating, los cuatro factores del método, sin la cercanía al día estimado, que cambia cada día. La barra va de 0 al mayor peso del ciclo; al pasar por la cifra se ve el desglose.»; `eleccion.html` añade «El peso del ciclo recoge la muestra y la frecuencia de cada sondeo; el rating de entonces va aparte, en su columna. La barra va de 0 al mayor peso del ciclo; al pasar por la cifra se ve el desglose.»

- [ ] **Step 9: Correr todo lo de páginas**

Run: `pytest tests/test_pages.py tests/test_pages_ejemplo.py tests/test_web_api.py tests/test_webapi_unit.py -q`
Expected: PASS (incluida `assert_public_wording`).

- [ ] **Step 10: Commit**

```bash
git status
git add mtpy/lib/pages/promedios.py mtpy/lib/pages/__init__.py web/templates/_polls_table.html web/templates/promedios.html web/templates/eleccion.html web/dist/css/site.css tests/fixtures/bundle/event-cycle.json tests/test_pages.py
git commit -m "Tabla de sondeos: columna Peso tras el rating (barra de 0 al mayor peso del ciclo y cifra a dos decimales) con el desglose por factores en el tooltip, en el servidor

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Columna «Peso» y desglose en el navegador

**Files:**
- Modify: `web/dist/js/format.js` (`barWidth`), `web/dist/js/polls-table.js` (`maxWeight`, `weightBreakdown`, `sortPolls`), `web/dist/js/pages/polls-dom.js` (`FIXED_COLUMNS`, `WEIGHT_TITLE`, `weightCell`, `renderBody`, `renderFoot`), `web/dist/js/pages/polls-wire.js` (`FIXED_KEYS`, `wmax`)
- Test: `tests/test_js_polls.py` (`TABLE_POLLS`, 150-160, 189-197 y pruebas nuevas)

**Interfaces:**
- Consumes: las claves del peso de `chart_polls` (Task 3); `fmtNum(x, 2)` de `format.js`.
- Produces: `barWidth(x, vmin, vmax)` en `format.js` (espejo de `bar_width`: 0-100 a un decimal, 0 si falta o `vmin === vmax`); `maxWeight(polls)` (mayor `weight` finito `> 0`, o `null`) y `weightBreakdown(poll)` (string o `null`, el mismo texto que Python) en `polls-table.js`; `renderBody(tbody, polls, parties, catalog, mode = 'pct', wmax = null)`.

- [ ] **Step 1: Escribir las pruebas**

`TABLE_POLLS` gana `'weight': 2.0` (GAD3, más `weight_sample: 1.339, weight_disp: 0.95, weight_over: 1.0, weight_rating: 2.4`), `'weight': 0.5` (CIS) y nada en 40dB. Pruebas:

```python
def test_weight_helpers_of_the_polls_table_match_the_server():
    assert table_js('p.maxWeight(POLLS)') == 2.0 and run_js('p.maxWeight([{weight: 0}, {weight: null}, {}])') is None
    assert table_js('p.weightBreakdown(POLLS[0])') == 'Muestra ×1,41\nDispersión ×0,95\nFrecuencia ×1,00\nRating ×2,40'
    assert run_js('p.weightBreakdown({weight_sample: 0.7, weight_disp: null, weight_over: 0.811, weight_rating: null})') == 'Muestra ×0,70\nFrecuencia ×0,81'
    assert run_js('p.weightBreakdown({weight_sample: 0.5, weight_disp: 0})') == 'Muestra ×0,50\nDispersión ×0,00'
    assert run_js('p.weightBreakdown({weight: 1.7})') is None and run_js('p.weightBreakdown({weight_sample: "x"})') is None
    assert table_js('p.sortPolls(POLLS, "weight", false).map(x => x.pollster)') == ['GAD3', 'CIS', '40dB']
    assert table_js('p.sortPolls(POLLS, "weight", true).map(x => x.pollster)') == ['CIS', 'GAD3', '40dB']
    zeros = [{'pollster': 'a', 'date': '2026-01-01', 'weight': 0}, {'pollster': 'b', 'date': '2026-01-02', 'weight': 1}, {'pollster': 'c', 'date': '2026-01-03'}]
    assert run_js('p.sortPolls({}, "weight", true).map(x => x.pollster)'.format(json.dumps(zeros))) == ['a', 'b', 'c']


def test_bar_width_mirrors_the_server():
    script = 'import * as f from "{}"; console.log(JSON.stringify([f.barWidth(0.31, 0, 3.384), f.barWidth(5, 0, 3), f.barWidth(null, 0, 1), f.barWidth(1, 2, 2)]));'
    out = subprocess.run(['node', '--input-type=module', '-e', script.format(pathlib.Path(ROOT, 'web', 'dist', 'js', 'format.js').as_uri())], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [9.2, 100, 0, 0]
```

En `test_polls_dom_fixed_columns_and_rating_colours_follow_the_server`, `fixed == [['pollster', 'Encuestadora'], ['rating', 'Rating (0-100)'], ['weight', 'Peso'], ['date', 'Fecha'], ['sample_size', 'Muestra']]` y, exportando también `d.WEIGHT_TITLE`, `title == pages.WEIGHT_TITLE`.

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_js_polls.py -k "weight_helpers or bar_width or fixed_columns" -q`
Expected: FAIL.

- [ ] **Step 3: Implementar**

`format.js`: `export function barWidth(x, vmin = 0, vmax = 100)` con la fracción recortada a [0, 1] y `Math.round(1000 * fraction) / 10`. `polls-table.js`: `maxWeight`, `weightBreakdown` (mismas reglas que Python: valor numérico finito, «Muestra» dividida sólo con `weight_disp > 0`, `fmtNum(value, 2)`, líneas unidas por `'\n'`) y `'weight'` en la lista de claves fijas de `sortPolls`. `polls-dom.js`: `FIXED_COLUMNS` con `['weight', 'Peso']` tras el rating, `export const WEIGHT_TITLE` (la cadena de Task 3) puesto en `renderHead` como el del rating; `weightCell(poll, wmax)`: sin número, `td.na` «—»; si no, `td.cell-bar` (más `na` con peso 0), `title` si hay desglose, `i.fill.w` con `style.left = '0'` y `style.width = barWidth(weight, 0, wmax) + '%'` (0 sin `wmax`) y `span` con `fmtNum(weight, 2)`; `renderBody(..., mode = 'pct', wmax = null)` inserta la celda tras `ratingCell`; `renderFoot` pone cuatro `el('td')` vacíos en la media y en las fijas. `polls-wire.js`: `FIXED_KEYS` gana `'weight'`; `const wmax = maxWeight(polls);` una vez en `wirePollsTable` y `renderBody(table.tBodies[0], shown, parties, catalog, state.mode, wmax)`.

- [ ] **Step 4: Correr**

Run: `pytest tests/test_js_polls.py tests/test_js_share.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add web/dist/js/format.js web/dist/js/polls-table.js web/dist/js/pages/polls-dom.js web/dist/js/pages/polls-wire.js tests/test_js_polls.py
git commit -m "Tabla de sondeos: la columna Peso y su desglose también en el navegador (orden por peso, barra sobre todos los sondeos del ciclo, cuatro celdas vacías en el pie)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Comprobación en el navegador y suite completa

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `pytest -q`
Expected: PASS, sin avisos nuevos.

- [ ] **Step 2: Arrancar la web local y pasear las dos páginas**

Con la skill `run` (o `deploy/check-api.sh` como referencia del puerto), abrir `/es` y `/elecciones/es/generales/2023-07` con el bundle local y comprobar: la celda Fecha con «29–30 Oct» y «1 Sep–1 Nov» (la última de Sigma Dos y la de SocioMétrica), la columna Peso entre el rating y la fecha con barra y cifra, el mayor peso con la barra llena, las de peso 0 en gris, orden por Peso en los dos sentidos, filtros y paginación repintan igual que el servidor (comparar la primera fila antes y después de pulsar «Más antiguos» y volver), la cabecera Peso y la cabecera Fecha con su tooltip. Con el bundle local anterior a la enmienda no hay desglose: comprobar que la celda no lleva `title` y nada falla en la consola. Las descargas de imágenes de la página siguen funcionando.

- [ ] **Step 3: Anotar el resultado**

Decir qué se vio, con la fecha del run local; sin commit.
