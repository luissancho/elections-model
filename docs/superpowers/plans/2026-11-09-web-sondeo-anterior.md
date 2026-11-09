# Detalle de un sondeo: cambio frente a la entrega anterior — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La página de un sondeo compara cada partido con la entrega anterior de la misma casa y medio en el ciclo: flecha de cambio junto a cada barra de la tarjeta (y por tanto en la imagen), columnas «Anterior» y «Cambio» en «Voto y escaños», cambio de escaños si las dos entregas los publicaron, nota y pie que nombran la entrega anterior.

**Architecture:** `previous_poll` (en `mtpy/lib/pages/sondeos.py`) busca la entrega anterior entre los sondeos del run; `vote_rows` y `sondeo_context` añaden la comparación al contexto y a la isla `initial`. En el navegador, `voteRows` (en `poll-card.js`) añade `change` a cada fila y `renderBars` (en `charts/bars.js`) gana `labelExtra`, un segundo texto coloreado tras la cifra de cada barra, que la imagen de la tarjeta recoge al redibujar el gráfico.

**Tech Stack:** Python 3 (Jinja2), pytest; ES modules con ECharts 5.6 probados con `node` desde `tests/test_js_*.py`.

**Spec:** `docs/superpowers/specs/2026-11-09-web-sondeo-anterior-design.md` (amplía `docs/superpowers/specs/2026-11-09-web-detalle-sondeo-design.md`).

## Global Constraints

- Entrega anterior = misma casa (`pollster_id`) y mismo medio (`(a or None) == (b or None)` sobre `sponsor`), `date[:10]` estrictamente anterior; la de fecha más reciente; empate de fecha: la que aparece más tarde en la parte `polls`.
- Flechas con el formato de la web: `fmtDelta(x, 1)` en JS y `m.badge(x, 1)` en Jinja («▲ 1,2», «▼ 0,4», «–» si redondea a cero); colores de `.dlt`: sube `#2ca02c`, baja `#d62728`, sin cambio el gris apagado de `THEME.mutedColor`.
- La fila «Otros» nunca lleva flecha ni cambio.
- Textos públicos en español con «sólo» con tilde, «encuestadora/casa/medio», nunca «empresa» ni «patrocinador»; la `date` de un sondeo es su último día de campo: nunca «publicado».
- Identificadores en inglés neutro; docstrings numpy; JSDoc en el JS (en inglés, sangría de 2 espacios, como el resto de módulos salvo `poll-card.js`, que sigue su estilo actual).
- No tocar `charts/series.js`, `share.js`, `pages/promedios.js`, `pages/series-wire.js`; `renderBars` sin `labelExtra` debe dibujar exactamente lo mismo que hoy.
- Tests con el `pytest` de pyenv (Python 3.11), nunca `python3`. Commits en español con el trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git status --short` y `git log -1 --oneline` antes de cada commit; sólo los ficheros de la tarea.

## Review Focus

1. Dos entregas de la misma casa el mismo día para el mismo medio (una oleada corregida): la anterior es la previa en la parte, nunca la propia; cubierto en la Tarea 1 (`previous_poll`, empate).
2. Una entrega anterior que no listaba un partido nuevo (entra UP, SALF): ese partido no lleva flecha ni cambio; cubierto en las Tareas 1 y 2.
3. Misma casa para otro medio el día anterior: no cuenta; cubierto en la Tarea 1.
4. El partido más votado con flecha: la etiqueta compuesta no se sale del gráfico ni de la imagen; cubierto en la Tarea 2 (margen del eje con `labelExtra`) y en la comprobación en Chrome de la Tarea 3.
5. Escaños en una sola de las dos entregas: sin cambio de escaños, la columna muestra sólo la horquilla; cubierto en la Tarea 1.

---

### Task 1: La entrega anterior en el contexto y en la página

**Files:**
- Modify: `mtpy/lib/pages/sondeos.py` (`vote_rows` ~173-211; `sondeo_context` ~458-590), `mtpy/lib/pages/__init__.py` (re-exportar `previous_poll`)
- Modify: `web/templates/sondeo.html` (nota de la tarjeta ~33; tabla `#vote-table` ~55-85 y su «Cómo leerlo»; «Cómo leerlo» de la tarjeta)
- Test: `tests/test_pages.py` (sección «Detalle de un sondeo»; el ayudante `write_detail_polls` y `DETAIL_POLLS`)

**Interfaces:**
- Produces: `previous_poll(polls: list, poll: dict) -> Optional[dict]`; `vote_rows(poll, parties, average, colors, effects=None, previous=None)` con, además de lo actual, `prev` (cifra de la anterior o `None`), `change` (`round(value - prev, 1)` o `None`) y `seats_change` (diferencia de puntos medios `round((lo+hi)/2)` de las dos horquillas, o `None` si falta alguna); contexto `previous` = `{'date': fmt_date, 'href': poll_detail_href(scope, prev) + query, 'title': poll_title(prev)}` o `None`; isla `initial['previous']` = `{'date': prev['date'][:10], 'title': poll_title(prev), 'values': {partido: cifra}}` o `None` (los partidos de `poll_parties` con cifra en la anterior).

- [ ] **Step 1: Tests que fallan**

```python
PREV_POLLS = [
    {'date': '2026-10-05', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': 'El Mundo', 'PP': 33.0},
    {'date': '2026-10-05', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'PP': 31.0},
    {'date': '2026-09-20', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'PP': 30.0},
    {'date': '2026-09-28', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'PP': 30.5},
    {'date': '2026-09-28', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'PP': 30.8},
    {'date': '2026-10-04', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': 'ABC', 'PP': 29.0},
    {'date': '2026-10-04', 'pollster_id': 3, 'pollster': 'GAD3', 'sponsor': None, 'PP': 28.0},
]


def test_previous_poll_is_the_last_earlier_delivery_of_the_same_house_and_sponsor():
    poll = PREV_POLLS[1]
    assert pages.previous_poll(PREV_POLLS, poll)['PP'] == 30.8  # same day of 28 sep: the later one in the part
    assert pages.previous_poll(PREV_POLLS, PREV_POLLS[0]) is None  # El Mundo: no earlier delivery for that sponsor
    assert pages.previous_poll(PREV_POLLS, PREV_POLLS[2]) is None  # the first of the cycle
    assert pages.previous_poll(PREV_POLLS, PREV_POLLS[4])['PP'] == 30.5  # same date earlier in the part
    assert pages.previous_poll(PREV_POLLS, PREV_POLLS[3])['PP'] == 30.0
    assert pages.previous_poll([], poll) is None


def test_vote_rows_compare_with_the_previous_delivery():
    poll = {'PP': 32.6, 'PSOE': 25.6, 'UP': 1.2, 'seats': {'PP': [136, 138], 'PSOE': [99, 101]}}
    previous = {'PP': 31.4, 'PSOE': 25.6, 'seats': {'PP': [130, 134]}}
    rows = pages.vote_rows(poll, ['PP', 'PSOE', 'UP'], {}, {}, previous=previous)
    assert [(r['name'], r['prev'], r['change'], r['seats_change']) for r in rows] == [
        ('PP', 31.4, 1.2, 5), ('PSOE', 25.6, 0.0, None), ('UP', None, None, None)]
    assert all(r['prev'] is None and r['change'] is None for r in pages.vote_rows(poll, ['PP'], {}, {}))
```

En los tests de la página: añadir a `DETAIL_POLLS` una entrega anterior de Sigma Dos sin medio (`'date': '2026-09-28', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'sample_size': 1500, 'rating': 71.2, 'weight': 1.0, 'PP': 32.1, 'seats': {'PP': [146, 148]}`) y comprobar en `/es/sondeos/2026-10-05/sigma-dos`: cabecera `<th scope="col">Anterior</th>` y `<th scope="col">Cambio</th>`; `32,1` y `▲ 1,4` en la fila del PP; el cambio de escaños `▲ 4` junto a `150–152`; pie con `href="/es/sondeos/2026-09-28/sigma-dos?mode=forecast"` y «28 sept 2026»; nota de la tarjeta «Flechas: frente al sondeo de Sigma Dos del 28 sept 2026.»; `initial['previous'] == {'date': '2026-09-28', 'title': 'Sigma Dos', 'values': {'PP': 32.1}}`. En `/es/sondeos/2026-09-24/sigma-dos/el-mundo` (sin anterior de ese medio): sin columnas «Anterior»/«Cambio», nota «Primer sondeo de Sigma Dos para El Mundo en este ciclo.», `initial['previous'] is None`. Ajustar las aserciones existentes que cuenten los sondeos de la casa en la isla (ahora hay uno más). Usar el formato de fecha que dé `fmt_date` («sept» con el de hoy).

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_pages.py -q -k "previous or vote_rows or sondeo_page"` → FAIL.

- [ ] **Step 3: Implementar** `previous_poll`, la ampliación de `vote_rows` y de `sondeo_context` (la anterior sale de `polls['polls']`, todos los del run, no sólo de los de la casa ya filtrados — es lo mismo, pero el filtro de casa+medio es el de `previous_poll`); en la plantilla: la nota de la tarjeta (`… del día.{% if previous %} Flechas: frente al sondeo de {{ previous.title }} del {{ previous.date }}.{% else %} Primer sondeo de {{ poll_title }} en este ciclo.{% endif %}` más la frase de los escaños), las dos columnas sólo con `previous` (`{{ row.prev|num(1) }}`, `{{ m.badge(row.change, 1) }}`), en la columna de escaños `{{ row.seats or '–' }}{% if row.seats_change is not none %} {{ m.badge(row.seats_change, 0) }}{% endif %}`, el pie «Anterior: <a href="{{ previous.href }}">sondeo del {{ previous.date }}</a>.» y una frase en cada «Cómo leerlo» («Las flechas comparan con la entrega anterior de la misma casa para el mismo medio en este ciclo: verde si sube, roja si baja.»).

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_pages.py -q` → PASS.

- [ ] **Step 5: Commit** — «Sondeo: comparación con la entrega anterior de la misma casa y medio (contexto, tabla y nota)».

---

### Task 2: Flecha de cambio junto a cada barra

**Files:**
- Modify: `web/dist/js/charts/bars.js` (`renderBars`, el `renderItem` de la serie `custom` ~70-110 y el `max` del eje ~115)
- Modify: `web/dist/js/poll-card.js` (`voteRows`)
- Test: `tests/test_js_charts.py` (junto a `test_bars_tooltip_names_the_interval_with_interval_text`), `tests/test_js_poll_card.py`

**Interfaces:**
- Produces: `renderBars(el, rows, {..., labelExtra = null})`: `labelExtra(row, index)` recibe la fila **de entrada** (no la normalizada) y devuelve `{text, color}` o `null`; cuando devuelve texto, el `renderItem` dibuja tras la cifra un segundo `text` de zrender con ese texto, ese color, peso 700 y la misma fuente, separado 6 px (medir el ancho de la cifra con `echarts.format.getTextRect` o, si no está disponible, con `zrender`'s `getBoundingRect` del primer texto — o usar un único texto con `rich`: `{v|31,9 %} {d|▲ 1,2}` con el color en el estilo `d`; elección del constructor, la que se pueda probar en node); con `labelExtra`, el `max` del eje usa `top * 1.3` en lugar de `top * 1.15` para dejar sitio. Sin `labelExtra`, la opción es idéntica a la de hoy. Para poder probarlo en node, extraer la construcción de la opción en `barsOption(rows, opts)` exportada, que `renderBars` monta (sin cambiar su firma).
- Produces: `voteRows(poll, parties, average = {}, previous = null)`: cada fila con cifra en las dos entregas lleva `change = Math.round((value - prev) * 10) / 10`; las demás, `change: null`; «Otros», `change: null`.

- [ ] **Step 1: Tests que fallan**

```python
def test_bars_label_extra_writes_a_coloured_change_after_the_value():
    rows = [{'name': 'PP', 'value': 31.9, 'change': 1.2}, {'name': 'Otros', 'value': 14.0, 'change': None}]
    plain = run_js('JSON.stringify(ba.barsOption(%s, {value: "value"}))' % json.dumps(rows))
    # with labelExtra the custom series' renderItem draws a second text (or a rich label) for PP only, and the axis
    # leaves more room; without it the option is the one of today (pin the max of the axis of both)
```

(Concretar las aserciones según la forma elegida: llamar al `renderItem` de la serie `custom` con un `api` falso — `value(i)`, `coord([x, y])`, `size([0, 1])` — como hace el test de `intervalText` con el `formatter`, y comprobar que la fila del PP produce el texto «▲ 1,2» con `fill: '#2ca02c'` y la de «Otros» no; y que `xAxis.max` es mayor con `labelExtra` que sin él.)

En `tests/test_js_poll_card.py`:

```python
def test_vote_rows_carry_the_change_against_the_previous_delivery():
    rows = run_js('pc.voteRows({PP: 32.6, PSOE: 25.6, UP: 1.2}, ["PP", "PSOE", "UP"], {}, {PP: 31.4, PSOE: 25.6})')
    assert [(r['name'], r['change']) for r in rows] == [('PP', 1.2), ('PSOE', 0), ('UP', None), ('Otros', None)]
    assert all(r['change'] is None for r in run_js('pc.voteRows({PP: 32.6}, ["PP"], {})'))
```

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_js_charts.py tests/test_js_poll_card.py -q -k "label_extra or previous_delivery"` → FAIL.

- [ ] **Step 3: Implementar** con las firmas de arriba (JSDoc de `labelExtra` y de `barsOption`).

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_js_charts.py tests/test_js_poll_card.py -q` → PASS (los tests de barras existentes, sin cambios).

- [ ] **Step 5: Commit** — «Barras: flecha de cambio tras la cifra (labelExtra); voteRows con el cambio frente a la entrega anterior».

---

### Task 3: La tarjeta dibuja las flechas y comprobación en Chrome

**Files:**
- Modify: `web/dist/js/pages/sondeo.js` (la llamada a `voteRows` y a `renderBars`, ~37-49)
- Test: comprobación en Chrome con los guiones del scratchpad (`/private/tmp/claude-501/-Users-luiss-HD-Proyectos-Code-elections-model/8096bee8-e449-4a82-b975-bb04ffe6aec0/scratchpad/t10/` y `t11/`: lanzador contra el paquete publicado en S3 en sólo lectura y script CDP de la imagen)

**Interfaces:**
- Consumes: `initial.previous` (Tarea 1), `voteRows(..., previous)` y `renderBars(..., {labelExtra})` (Tarea 2), `fmtDelta` de `format.js`, `THEME.mutedColor` de `charts/base.js`.

- [ ] **Step 1:** `voteRows(initial.poll, parties, initial.average || {}, initial.previous ? initial.previous.values : null)` y `labelExtra: (row) => (row.change === null || row.change === undefined ? null : {text: fmtDelta(row.change, 1), color: …})` con verde/rojo/gris según el signo y el redondeo (`fmtDelta` devuelve «–» cuando redondea a cero → gris).

- [ ] **Step 2:** `pytest tests/test_pages.py tests/test_js_charts.py tests/test_js_poll_card.py tests/test_js_card_image.py -q` → PASS.

- [ ] **Step 3: Chrome.** Con el lanzador contra S3, elegir en `/es` un sondeo cuya casa y medio tengan una entrega anterior en el ciclo (y otro que sea el primero de su casa y medio): capturas a 1300 y 390 px y la imagen de la tarjeta por CDP desde las dos anchuras. Comprobar: flechas verdes/rojas tras la cifra, sin flecha en «Otros», la flecha del partido más votado dentro del gráfico, la nota con la fecha de la anterior, la tabla con «Anterior» y «Cambio» y el pie con enlace; imagen idéntica desde las dos anchuras. Sin `Uncaught` en la consola. Parar el servidor al acabar.

- [ ] **Step 4:** `pytest -q -x` → PASS.

- [ ] **Step 5: Commit** — «Sondeo: flechas de cambio frente a la entrega anterior en la tarjeta».
