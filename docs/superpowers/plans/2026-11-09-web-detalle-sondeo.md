# Detalle de un sondeo y su tarjeta para compartir — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cada sondeo del run de un ámbito tiene su página (`/{scope}/sondeos/{fecha}/{casa}[/{medio}]`) con una tarjeta descargable como una sola imagen (casa, medio, campo, muestra, barras de voto con el promedio del día, hemiciclo con los escaños de la casa, rating, peso y distancia al promedio) y, debajo, el detalle: voto frente al promedio, los factores del peso, la casa en este ciclo con el sondeo resaltado y la ficha; cada fila de «Todas las encuestas» enlaza a la suya.

**Architecture:** La parte `polls` del contrato gana los tres factores del peso y la fuente (`POLL_COLUMNS`, 14 columnas; el `Forecaster` conserva `url`). Un módulo nuevo `mtpy/lib/pages/sondeos.py` resuelve el sondeo por fecha y slugs dentro del `polls.json` del run, compara con `series.json` el día del sondeo y arma el contexto de `web/templates/sondeo.html`, servida por `controllers/Sondeos.py` como página del evento (misma barra, `mode` y `run`). En el navegador, `poll-card.js` (lógica pura) y `pages/sondeo.js` dibujan las barras (`bars.js`), el hemiciclo con rótulos de horquilla (`hemicycle.js` gana `labels` y `rest`) y el gráfico de la casa (`series.js`, sin tocarlo, con un anillo sobre el sondeo); la tarjeta se descarga con `card-image.js` y el gráfico con `share.js`.

**Tech Stack:** Python 3 (pandas, Jinja2), pytest; ES modules en `web/dist/js` con ECharts 5.6 y html2canvas 1.4.1 vendorizado, probados con `node` desde `tests/test_js_*.py`.

**Spec:** `docs/superpowers/specs/2026-11-09-web-detalle-sondeo-design.md`

## Global Constraints

- Identificadores en inglés neutro; textos públicos en español con «sólo» con tilde, «encuestadora/casa» y «medio», nunca «empresa» ni «patrocinador» visible (`assert_public_wording` de `tests/test_pages.py`).
- PEP8 y docstrings numpy en toda función Python nueva o tocada; JSDoc en el JS. Sin FastAPI ni scripts sueltos.
- **No tocar** los módulos de JS ni las plantillas que Luis tiene con cambios sin commit (2026-11-09): `share.js`, `charts/series.js`, `blocks.js`, `polls-tooltip.js`, `pages/promedios.js`, `pages/series-wire.js`, `pages/legend-dom.js`, `pages/evolution-wire.js`, `pages/elecciones.js`, `promedios.html`, `eleccion.html`. Lo que haga falta de ellos se usa desde fuera (importando sus funciones). Dos ficheros de este plan también están en su lista (`web/dist/css/site.css`, `tests/test_js_charts.py`): antes de empezar la Tarea 5 y la Tarea 6, `git status --short` sobre ellos; si siguen modificados por Luis, el commit de la tarea incluye sólo los trozos propios (revisar `git diff <fichero>` y, de ser necesario, aplicar los trozos propios con `git apply --cached` de un parche recortado), nunca los suyos.
- La `date` de un sondeo es el último día de su trabajo de campo (`loader.py`: `date = end_date`): la página nunca dice «publicado»; habla de «trabajo de campo» y de «fecha del sondeo».
- Sólo los escaños que publicó la propia encuestadora (`seats` de la parte `polls`): ningún reparto del modelo por sondeo.
- `html2canvas` 1.4.1 no entiende `color-mix()`: el CSS nuevo usa colores planos o `rgba`.
- Los runs anteriores a la enmienda no traen `weight_over`, `weight_sample`, `weight_rating` ni `url`: la página los muestra con guiones y una fuente genérica, nunca falla.
- Luis commitea en paralelo en el mismo árbol: `git status` y `git log -1` antes de cada commit; nunca reescribir sus commits; añadir sólo los ficheros de la tarea. Mensajes de commit en español con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Los tests nunca escriben en `backtest/results` ni llaman a nada con `out=None`.

## Review Focus

1. Un sondeo cuyo `sponsor` da un slug vacío (`'—'`, `'?'`): la URL no lleva segmento de medio y `find_poll` lo encuentra por la casa y la fecha; cubierto en la Tarea 2 (`poll_detail_href`, `find_poll`).
2. Un sondeo con `seats` que no suman la cámara (horquillas, o una casa que no reparte todos los escaños): el hemiciclo completa con «Sin asignar» y nunca pasa de la cámara en el centro; cubierto en la Tarea 6 (`rest`) y la Tarea 7 (`pollSeats`).
3. Un sondeo cuya fecha es anterior al primer día de `series` (un run con sondeos de antes del arranque del promedio): sin promedio del día, la tabla pone guiones, la distancia es un guion y las barras no llevan marca; cubierto en la Tarea 2 (`series_at`), la Tarea 4 (`vote_rows`, `distance_to_average`) y la Tarea 7 (`voteRows`).
4. Un run sin la parte `weights` o sin `house-effects` (relleno antiguo), y un sondeo de peso 0 (oleada solapada de un tracking): la tarjeta del peso sólo enseña los factores, nunca dice «fuera de la ventana» a un sondeo de peso 0, y la tabla no tiene columna de efecto de casa; cubierto en la Tarea 4 (`weight_card`) y la Tarea 5 (página sin esas partes).
5. La descarga de la tarjeta desde un móvil (contenedor del hemiciclo de ~350 px): el hemiciclo sale al tamaño de la imagen de 1200 px, no al de la pantalla; cubierto en la Tarea 6 (`__imageOption`) y la Tarea 8 (`card-image.js`), y comprobado en la Tarea 10.
6. Un ámbito autonómico con `?run=` anclado: los enlaces de la página (volver, índice de la barra, perfil) conservan `mode` y `run`, y el índice apunta a `/{scope}` y no a anclas de esta página; cubierto en la Tarea 5.

---

### Task 1: Contrato — los tres factores del peso y la fuente en `polls`

**Files:**
- Modify: `mtpy/lib/publish/forecast.py:32-33` (`POLL_COLUMNS`) y `:403` (redondeo de los pesos)
- Modify: `mtpy/lib/forecaster.py:578-583` (`data_cols`) y antes de `:620` (la selección `series.set_index(...)[data_cols + party_names]`)
- Modify: `docs/web/contrato.md:143-148`
- Test: `tests/test_publish_unit.py` (tras `test_export_poll_rows_builds_the_table_at_once_without_fragmenting_it`, línea 241), `tests/test_forecaster_unit.py` (tras `test_fc_series_raw_keeps_the_published_polls_after_house_effects`, línea 220; el `_weights_builder` de la línea ~530)

**Interfaces:**
- Produces: `publish.POLL_COLUMNS == ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'sample_size', 'mtype', 'rating', 'weight', 'weight_over', 'weight_sample', 'weight_rating', 'url']`; cada registro de `polls.json` (y de `event-cycle`) lleva las cuatro claves nuevas, los pesos a 3 decimales y `null` cuando la serie no las trae; `Forecaster.build_series` deja `url` en `series` y `series_raw` (`None` cuando los sondeos no la traen).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_publish_unit.py`:

```python
def test_export_poll_rows_publishes_the_weight_factors_and_the_source():
    """Enmienda 2026-11-09: `polls` lleva los tres factores del peso (a 3 decimales) y la fuente; lo que la serie no
    trae sale nulo, así que un sondeo sin `url` ni factores se publica igual."""
    polls = pd.DataFrame({'date': pd.to_datetime(['2026-10-01', '2026-10-02']), 'pollster_id': [1, 2], 'sponsor_id': [None, 3],
                          'weight': [0.5, 0.25], 'weight_over': [1.0, 0.5], 'weight_sample': [0.51234, 0.5],
                          'weight_rating': [1.0, 1.0], 'url': ['https://x.test/a', None], 'PP': [30.0, 31.0]})
    records, frame = publish.forecast.export_poll_rows(polls, ['PP'])
    assert publish.POLL_COLUMNS == ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'sample_size',
                                    'mtype', 'rating', 'weight', 'weight_over', 'weight_sample', 'weight_rating', 'url']
    assert records[0]['weight_over'] == 1.0 and records[0]['weight_sample'] == 0.512 and records[0]['url'] == 'https://x.test/a'
    assert records[1]['url'] is None and records[1]['weight_rating'] == 1.0
    assert list(frame.columns)[:14] == publish.POLL_COLUMNS
    bare = pd.DataFrame({'date': pd.to_datetime(['2026-10-01']), 'pollster_id': [1], 'sponsor_id': [None], 'weight': [0.5], 'PP': [30.0]})
    records, _ = publish.forecast.export_poll_rows(bare, ['PP'])
    assert pd.isna(records[0]['weight_over']) and pd.isna(records[0]['weight_rating']) and records[0]['url'] is None
```

En `tests/test_forecaster_unit.py`: en `_weights_builder`, dar a `polls` una columna `url` (`'https://a.test/{pid}'` por casa) y añadir:

```python
def test_build_series_keeps_the_source_url_of_the_polls(monkeypatch):
    """La serie conserva la fuente de cada sondeo (`url`) para publicarla; los resultados oficiales no tienen."""
    build, _ = _weights_builder(monkeypatch)
    series = build(None).series_raw
    assert series.loc[series.pollster == 'A', 'url'].eq('https://a.test/1').all()
    assert series.loc[series.pollster.isnull(), 'url'].isnull().all()
```

- [ ] **Step 2: Correr los tests y verlos fallar**

Run: `pytest tests/test_publish_unit.py::test_export_poll_rows_publishes_the_weight_factors_and_the_source tests/test_forecaster_unit.py::test_build_series_keeps_the_source_url_of_the_polls -q`
Expected: FAIL (`POLL_COLUMNS` de 10 y `KeyError: 'url'`).

- [ ] **Step 3: Implementar**

`publish/forecast.py`: `POLL_COLUMNS` con las cuatro columnas nuevas tras `weight`; en `export_poll_rows`, redondear `('weight', 'weight_over', 'weight_sample', 'weight_rating')` con `astype(float).round(PROB_DECIMALS)` (la columna ausente ya llega como `None` y queda `NaN`, que el sobre escribe como `null`). Actualizar el docstring de `export_poll_rows`.

`forecaster.py`: `'url'` en `data_cols` (tras `'sponsor'`); justo antes de la selección `[data_cols + party_names]`, `if 'url' not in series.columns: series['url'] = None` (los sondeos de los tests y de las pruebas de afinado no la traen). Una frase en el docstring de `fc_series_raw`.

`docs/web/contrato.md:143-148`: la lista de columnas con las 14 y «(14 columnas)»; añadir «Enmienda 2026-11-09: `weight_over`, `weight_sample` y `weight_rating` (a 3 decimales; `weight` es su producto) y `url`, la fuente del sondeo tal como la guarda la tabla `polls` (`null` sin ella); los runs anteriores no las traen y la página de un sondeo enseña entonces sólo el peso total.»

- [ ] **Step 4: Correr los tests de publicación y del Forecaster**

Run: `pytest tests/test_publish_unit.py tests/test_forecaster_unit.py tests/test_bundle_unit.py tests/test_pipelines_publish.py -q`
Expected: PASS (los tests que comparan con `publish.POLL_COLUMNS` siguen valiendo).

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add mtpy/lib/publish/forecast.py mtpy/lib/forecaster.py docs/web/contrato.md tests/test_publish_unit.py tests/test_forecaster_unit.py
git commit -m "Contrato polls: factores del peso y fuente de cada sondeo

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: La URL del sondeo, su resolutor y el promedio del día

**Files:**
- Create: `mtpy/lib/pages/sondeos.py`
- Modify: `mtpy/lib/pages/slugs.py` (tras `pollster_href`, línea 191), `mtpy/lib/pages/common.py:85-104` (`MESSAGES`), `mtpy/lib/pages/__init__.py` (re-exportar `poll_detail_href`, `find_poll`, `series_at`)
- Test: `tests/test_pages.py` (sección nueva al final: «Detalle de un sondeo»)

**Interfaces:**
- Produces: `slugs.poll_detail_href(scope: str, poll: dict) -> Optional[str]`; `sondeos.find_poll(polls: list, date: str, pollster: str, sponsor: Optional[str] = None) -> dict` (lanza `HttpError(404, 'poll not found')`); `sondeos.series_at(series: dict, date: str) -> dict`; `MESSAGES['poll not found'] == 'Sondeo no encontrado.'`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_poll_detail_href_builds_the_route_from_the_poll():
    poll = {'date': '2026-11-09', 'pollster_id': 12, 'pollster': 'NC Report', 'sponsor': 'La Razón'}
    assert pages.poll_detail_href('es', poll) == '/es/sondeos/2026-11-09/nc-report/la-razon'
    assert pages.poll_detail_href('es-md', poll | {'sponsor': None}) == '/es-md/sondeos/2026-11-09/nc-report'
    assert pages.poll_detail_href('es', poll | {'sponsor': '—'}) == '/es/sondeos/2026-11-09/nc-report'  # a sponsor without a slug
    assert pages.poll_detail_href('es', poll | {'pollster': '???'}) == '/es/sondeos/2026-11-09/12/la-razon'
    assert pages.poll_detail_href('es', poll | {'date': '2026-11-09T00:00:00'}) == '/es/sondeos/2026-11-09/nc-report/la-razon'
    assert pages.poll_detail_href('es', poll | {'date': None}) is None
    assert pages.poll_detail_href('es', {'date': '2026-11-09', 'pollster': None, 'pollster_id': None}) is None


POLLS_OF_A_DAY = [
    {'date': '2026-10-01', 'pollster_id': 3, 'pollster': 'GAD3', 'sponsor': 'ABC', 'PP': 33.0},
    {'date': '2026-10-01', 'pollster_id': 3, 'pollster': 'GAD3', 'sponsor': 'El Mundo', 'PP': 34.0},
    {'date': '2026-10-01', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'PP': 32.0},
    {'date': '2026-10-02', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': 'El Mundo', 'PP': 31.0},
    {'date': '2026-10-03', 'pollster_id': 9, 'pollster': 'Casa', 'sponsor': '—', 'PP': 30.0},
]


def test_find_poll_matches_the_date_the_house_and_the_sponsor():
    assert pages.find_poll(POLLS_OF_A_DAY, '2026-10-01', 'gad3', 'abc')['PP'] == 33.0
    assert pages.find_poll(POLLS_OF_A_DAY, '2026-10-01', 'sigma-dos')['PP'] == 32.0
    assert pages.find_poll(POLLS_OF_A_DAY, '2026-10-02', 'sigma-dos')['PP'] == 31.0  # the only one of the house that day
    assert pages.find_poll(POLLS_OF_A_DAY, '2026-10-03', 'casa')['PP'] == 30.0  # a sponsor without a slug
    assert pages.find_poll(POLLS_OF_A_DAY, '2026-10-01', '1')['PP'] == 32.0  # the id stands for a house without a slug
    assert pages.MESSAGES['poll not found'] == 'Sondeo no encontrado.'
    for args in (('2026-10-01', 'gad3'), ('2026-10-01', 'gad3', 'la-razon'), ('2026-10-03', 'gad3'), ('2026-10-01', 'nadie'),
                 ('2026-10-02', 'sigma-dos', 'el-pais')):
        with pytest.raises(HttpError) as err:
            pages.find_poll(POLLS_OF_A_DAY, *args)
        assert (err.value.status, err.value.message) == (404, 'poll not found')


def test_series_at_takes_the_average_of_the_last_day_up_to_the_date():
    series = {'dates': ['2026-10-01', '2026-10-02', '2026-10-05'], 'parties': ['PP', 'PSOE'],
              'mean': {'PP': [30.0, 30.5, 31.0], 'PSOE': [28.0, None, 27.0]}, 'lo': {}, 'hi': {}}
    assert pages.series_at(series, '2026-10-02') == {'PP': 30.5, 'PSOE': None}
    assert pages.series_at(series, '2026-10-04') == {'PP': 30.5, 'PSOE': None}
    assert pages.series_at(series, '2026-10-09') == {'PP': 31.0, 'PSOE': 27.0}
    assert pages.series_at(series, '2026-09-30') == {}
    assert pages.series_at({'dates': [], 'parties': [], 'mean': {}}, '2026-10-01') == {}
```

(`HttpError` expone `status` y `message`: comprobarlo en `mtpy/core/api.py:416` y ajustar los nombres de los atributos si son otros.)

- [ ] **Step 2: Correr los tests y verlos fallar**

Run: `pytest tests/test_pages.py -q -k "poll_detail_href or find_poll or series_at"`
Expected: FAIL (`AttributeError: module has no attribute poll_detail_href`).

- [ ] **Step 3: Implementar**

`slugs.poll_detail_href(scope, poll)`: `date = str(poll.get('date') or '')[:10]`, válida si cumple `^\d{4}-\d{2}-\d{2}$`; `house = slugify(poll.get('pollster')) or (str(int(pollster_id)) si no falta)`; sin fecha o sin casa, `None`; `sponsor = slugify(poll.get('sponsor') or '')`; `'/{scope}/sondeos/{date}/{house}'` + `'/' + sponsor` si no está vacío. Una constante `POLLS_PATH = 'sondeos'` junto a `PROFILE_PATH`.

`sondeos.py` (docstring de módulo: la página de un sondeo): `find_poll` filtra por `str(date)[:10] == date`, segmento de casa en `{slugify(pollster), str(pollster_id)}` y, con `sponsor`, `slugify(sponsor or '') == sponsor`; sin `sponsor`, primero los sondeos de la casa ese día con slug de medio vacío y, si no hay ninguno, todos los de la casa ese día; con exactamente un candidato lo devuelve, si no lanza el 404. `series_at`: el mayor índice `i` con `dates[i] <= date`; `{name: mean[name][i] si existe, else None}` para `series['parties']`; `{}` sin tal día.

`common.MESSAGES['poll not found'] = 'Sondeo no encontrado.'`. Re-exportar en `pages/__init__.py` (`from .sondeos import find_poll, series_at` y `poll_detail_href` en la línea de `slugs`).

- [ ] **Step 4: Correr los tests y verlos pasar**

Run: `pytest tests/test_pages.py -q -k "poll_detail_href or find_poll or series_at"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add mtpy/lib/pages/sondeos.py mtpy/lib/pages/slugs.py mtpy/lib/pages/common.py mtpy/lib/pages/__init__.py tests/test_pages.py
git commit -m "Página de un sondeo: URL, resolutor y promedio del día

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Cada fila de «Todas las encuestas» enlaza a su sondeo

**Files:**
- Modify: `mtpy/lib/pages/promedios.py:313-342` (`chart_polls`), `:431-503` (`polls_table`), `:776` y `:795` (sus llamadas en `promedios_context`)
- Modify: `web/templates/_polls_table.html` (la celda Fecha, `<td{% if row.fieldwork %} title=...>{{ row.date }}</td>`)
- Modify: `web/dist/js/pages/polls-dom.js:146-158` (`renderBody`, la celda `date`)
- Test: `tests/test_pages.py:531-540` (`test_promedios_initial_polls_carry_the_chart_and_table_columns`) y un test nuevo junto a `test_promedios_polls_table_renders_fifteen_of_many_polls_and_the_island_carries_them_all` (línea ~684)

**Interfaces:**
- Consumes: `pages.poll_detail_href` (Tarea 2).
- Produces: `chart_polls(polls: dict, scope: Optional[str] = None) -> dict`, cada sondeo con `detail_href` (`poll_detail_href(scope, poll)`; `None` sin `scope`); `polls_table(polls, parties, colors, n=PAGE_SIZE, scope=None) -> dict`, cada fila con `detail_href`. La página de una elección pasada (`elecciones.py:1214, 1244`) no pasa `scope`: sus filas no enlazan.

- [ ] **Step 1: Escribir los tests que fallan**

En `test_promedios_initial_polls_carry_the_chart_and_table_columns`, añadir `'detail_href'` al conjunto exigido y, dentro del bucle, `assert poll['detail_href'] == pages.poll_detail_href('es', poll)`. Test nuevo:

```python
def test_promedios_polls_table_links_each_poll_to_its_page(api, fresh_app, events_site):
    polls = fixture_data('polls')
    base = polls['polls'][0]
    polls['polls'] = [base | {'date': '2026-08-02', 'sponsor': 'M1', 'start_date': '2026-08-01', 'end_date': '2026-08-02'}]
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', polls, 'es', run_id=RUN)
    webapi.site().files.clear()
    tbody = polls_tbody(html(call(api, '/es')[2]).replace('\n', ' '))
    assert ('<td title="Trabajo de campo: 1 ago 2026 – 2 ago 2026"><a href="/es/sondeos/2026-08-02/sigma-dos/m1">1–2 Ago</a></td>'
            in tbody)
    # the polls of a past election have no page: their date cell is plain text
    past = polls_tbody(html(call(events_site, '/elecciones/es/generales/2023-07')[2]).replace('\n', ' '))
    assert '/sondeos/' not in past and '<td>' in past
```

(Comprobar con el test de la elección pasada de la línea 2020 qué ruta y fixture sirven la tabla de 2023; si `events_site` no basta para la tabla del ciclo, usar la fixture que la escriba.)

- [ ] **Step 2: Correr los tests y verlos fallar**

Run: `pytest tests/test_pages.py -q -k "polls_table_links or initial_polls_carry"`
Expected: FAIL (`detail_href` ausente).

- [ ] **Step 3: Implementar**

`chart_polls(polls, scope=None)`: añadir `'detail_href': poll_detail_href(scope, poll) if scope else None` a cada registro (docstring). `polls_table(..., scope=None)`: `'detail_href'` en cada fila. En `promedios_context`, pasar `scope` a las dos llamadas. Plantilla: `<td{% if row.fieldwork %} title="{{ row.fieldwork }}"{% endif %}>{% if row.detail_href %}<a href="{{ row.detail_href }}">{{ row.date }}</a>{% else %}{{ row.date }}{% endif %}</td>`. `polls-dom.js` `renderBody`: la celda `date` recibe un `<a href=poll.detail_href>` con el texto cuando `poll.detail_href` existe, y el texto a secas si no; el `title` del campo se conserva en el `td`.

- [ ] **Step 4: Correr los tests de páginas**

Run: `pytest tests/test_pages.py tests/test_js_polls.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add mtpy/lib/pages/promedios.py web/templates/_polls_table.html web/dist/js/pages/polls-dom.js tests/test_pages.py
git commit -m "Todas las encuestas: la fecha de cada sondeo enlaza a su página

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Las piezas del contexto del sondeo

**Files:**
- Modify: `mtpy/lib/pages/sondeos.py` (Tarea 2), `mtpy/lib/pages/__init__.py` (re-exportar las funciones nuevas)
- Test: `tests/test_pages.py` (sección «Detalle de un sondeo»)

**Interfaces:**
- Consumes: `filters.fmt_num`, `fmt_int`, `fmt_date`, `gradient`, `DASH`; `promedios.RATING_COLORS`, `fieldwork_label`; `casas.mtype_label`; `slugs.pollster_href`.
- Produces (todas en `sondeos.py`):
  - `seats_text(seats_range) -> Optional[str]`: `[136, 138]` → `'136–138'`, `[100, 100]` → `'100'`, vacío o `None` → `None`.
  - `vote_rows(poll: dict, parties: list, average: dict, colors: dict, effects: Optional[dict] = None) -> list`: una fila por partido con valor numérico en el sondeo, por valor descendente (empates en el orden de `parties`): `{'name', 'color', 'value' (float), 'avg' (float|None), 'diff' (round(value − avg, 1)|None), 'effect' (effects[name]|None), 'seats' (seats_text|None)}`.
  - `distance_to_average(poll: dict, parties: list, average: dict) -> Optional[float]`: media de |sondeo − promedio| a 1 decimal sobre los partidos con las dos cifras; `None` sin ninguno.
  - `weight_card(poll: dict, weights: Optional[dict]) -> dict`: `{'factors': [{'label': 'Solape', 'value'}, {'label': 'Muestra', ...}, {'label': 'Rating', ...}, {'label': 'Peso', ...}] (fmt_num a 2 decimales; el guion si falta), 'has_part': bool, 'today': {'date', 'kernel', 'weight', 'party'} | None, 'status': str | None, 'overlap': bool}`: `today` sale de las filas de `weights['rows']` con la misma `date`, `pollster_id` y medio (`(a or None) == (b or None)`), la del partido con más voto en el sondeo; `status` es `None` sin parte, `'today'` con fila, `'zero'` cuando `weight` es 0 o `None` (`overlap` True si además `weight_over` es 0), `'outside'` sin fila y con `date` anterior al primer día de `weights['line'][p]['dates']` de algún partido, `'absent'` sin fila en otro caso. La plantilla pone una frase por `status` (Tarea 5).
  - `poll_kpis(poll: dict, distance: Optional[float]) -> list` para `m.kpis`: `[{'label': 'Rating', 'value': fmt_num(rating, 1), 'unit': '', 'color': color de gradient(rating, 0, 100, RATING_COLORS) o None}, {'label': 'Peso en el promedio', 'value': fmt_num(weight, 2), 'unit': ''}, {'label': 'Muestra', 'value': fmt_int(sample_size), 'unit': 'entrevistas'}, {'label': 'Distancia al promedio', 'value': fmt_num(distance, 1), 'unit': 'pts'}]`.
  - `poll_title(poll: dict) -> str`: `'NC Report para La Razón'`, o la casa sola.
  - `fieldwork_text(poll: dict) -> Optional[str]`: `'29 oct 2026 – 30 oct 2026'`, una sola fecha si coinciden o sólo se conoce una, `None` sin campo (la plantilla pone entonces «Fecha {date}»: la `date` es el último día de campo, nunca «publicado»).
  - `sheet_rows(poll: dict, house_href: str, event_label: str, run_text: str) -> list`: `[{'label', 'text', 'href'}]` en este orden: Encuestadora (href al perfil), Medio (`'—'` sin él), Tipo (`mtype_label`), Trabajo de campo (`fieldwork_text`, o `fmt_date(date)` sin campo), Muestra (`fmt_int`), Rating (`fmt_num(…, 1)`), Peso (`fmt_num(…, 2)`), Fuente (`href` = `url` y `text` = el host de la URL si la hay; si no, `text` = `'Wikipedia, lista de sondeos'` y `href` None), Elección (`event_label`), Actualización del modelo (`run_text`).

- [ ] **Step 1: Escribir los tests que fallan**

```python
POLL = {'date': '2026-11-09', 'pollster_id': 12, 'pollster': 'NC Report', 'sponsor': 'La Razón', 'start_date': '2026-11-05',
        'end_date': '2026-11-07', 'sample_size': 1000, 'mtype': 'cati', 'rating': 64.23, 'weight': 0.8412, 'weight_over': 1.0,
        'weight_sample': 0.8412, 'weight_rating': 1.0, 'url': 'https://www.larazon.es/encuesta', 'PP': 32.6, 'PSOE': 25.6,
        'VOX': 19.3, 'ERC': None, 'seats': {'PP': [136, 138], 'PSOE': [99, 101], 'VOX': [67, 69]}}
AVERAGE = {'PP': 31.9, 'PSOE': 26.4, 'VOX': 18.0, 'ERC': 2.0}
COLORS = {'PP': '#1d84ce', 'PSOE': '#ef1c27', 'VOX': '#63be21', 'ERC': '#ffb232'}


def test_vote_rows_order_the_parties_of_the_poll_and_compare_them_with_the_average():
    rows = pages.vote_rows(POLL, ['PSOE', 'PP', 'VOX', 'ERC'], AVERAGE, COLORS, effects={'PP': 0.6})
    assert [row['name'] for row in rows] == ['PP', 'PSOE', 'VOX']  # ERC has no value in the poll
    assert rows[0] == {'name': 'PP', 'color': '#1d84ce', 'value': 32.6, 'avg': 31.9, 'diff': 0.7, 'effect': 0.6, 'seats': '136–138'}
    assert rows[1]['diff'] == -0.8 and rows[1]['effect'] is None
    bare = pages.vote_rows(POLL | {'seats': {'PP': [136, 136]}}, ['PP', 'PSOE'], {}, COLORS)
    assert bare[0]['avg'] is None and bare[0]['diff'] is None and bare[0]['seats'] == '136' and bare[1]['seats'] is None
    assert pages.seats_text([]) is None and pages.seats_text(None) is None


def test_distance_to_average_is_the_mean_absolute_gap_over_the_shared_parties():
    assert pages.distance_to_average(POLL, ['PP', 'PSOE', 'VOX', 'ERC'], AVERAGE) == 0.9  # (0.7 + 0.8 + 1.3) / 3
    assert pages.distance_to_average(POLL, ['PP'], {}) is None


def test_weight_card_shows_the_factors_and_the_kernel_of_the_day():
    weights = {'date': '2026-11-09', 'parties': ['PP', 'PSOE'], 'rows': [
        {'name': 'PSOE', 'date': '2026-11-09', 'pollster_id': 12, 'sponsor': 'La Razón', 'weight_kernel': 0.97, 'weight': 0.816},
        {'name': 'PP', 'date': '2026-11-09', 'pollster_id': 12, 'sponsor': 'La Razón', 'weight_kernel': 0.98, 'weight': 0.824},
        {'name': 'PP', 'date': '2026-11-09', 'pollster_id': 12, 'sponsor': None, 'weight_kernel': 0.5, 'weight': 0.4},
    ], 'line': {'PP': {'dates': ['2026-09-20', '2026-11-09'], 'mean': [30.0, 32.0], 'lo': [29.0, 31.0], 'hi': [31.0, 33.0]}},
        'kernel': {}, 'stats': {}}
    card = pages.weight_card(POLL, weights)
    assert [(f['label'], f['value']) for f in card['factors']] == [('Solape', '1,00'), ('Muestra', '0,84'), ('Rating', '1,00'), ('Peso', '0,84')]
    assert card['has_part'] and card['status'] == 'today' and not card['overlap']
    assert card['today'] == {'date': '2026-11-09', 'kernel': '0,98', 'weight': '0,82', 'party': 'PP'}
    old = pages.weight_card({k: v for k, v in POLL.items() if not k.startswith('weight_')} | {'date': '2026-08-01'}, weights)
    assert [f['value'] for f in old['factors']] == ['–', '–', '–', '0,84'] and old['today'] is None and old['status'] == 'outside'
    inside = pages.weight_card(POLL | {'date': '2026-10-20'}, weights)
    assert inside['today'] is None and inside['status'] == 'absent'
    zero = pages.weight_card(POLL | {'date': '2026-11-09', 'weight': 0.0, 'weight_over': 0.0}, weights)
    assert zero['status'] == 'zero' and zero['overlap'] and zero['today'] is None  # the rows of a poll of weight 0 are not looked up
    assert pages.weight_card(POLL | {'weight': 0.0, 'weight_over': 1.0}, weights)['overlap'] is False
    assert pages.weight_card(POLL, None) == {'factors': card['factors'], 'has_part': False, 'today': None, 'status': None, 'overlap': False}


def test_poll_kpis_title_fieldwork_and_sheet():
    kpis = pages.poll_kpis(POLL, 0.9)
    assert [(k['label'], k['value'], k['unit']) for k in kpis] == [('Rating', '64,2', ''), ('Peso en el promedio', '0,84', ''),
                                                                 ('Muestra', '1.000', 'entrevistas'), ('Distancia al promedio', '0,9', 'pts')]
    assert kpis[0]['color'] == pages.gradient(64.23, 0, 100, pages.RATING_COLORS)
    assert pages.poll_kpis(POLL | {'rating': None}, None)[0]['color'] is None and pages.poll_kpis(POLL, None)[3]['value'] == '–'
    assert pages.poll_title(POLL) == 'NC Report para La Razón' and pages.poll_title(POLL | {'sponsor': None}) == 'NC Report'
    assert pages.fieldwork_text(POLL) == '5 nov 2026 – 7 nov 2026'
    assert pages.fieldwork_text(POLL | {'end_date': None}) == '5 nov 2026' and pages.fieldwork_text({'date': '2026-11-09'}) is None
    sheet = pages.sheet_rows(POLL, '/casas/nc-report', 'Elecciones generales · 29 nov 2026', 'Actualizado el 9 nov 2026')
    assert [row['label'] for row in sheet] == ['Encuestadora', 'Medio', 'Tipo', 'Trabajo de campo', 'Muestra', 'Rating', 'Peso', 'Fuente',
                                               'Elección', 'Actualización del modelo']
    assert sheet[0] == {'label': 'Encuestadora', 'text': 'NC Report', 'href': '/casas/nc-report'}
    assert sheet[3]['text'] == '5 nov 2026 – 7 nov 2026'
    assert sheet[7] == {'label': 'Fuente', 'text': 'www.larazon.es', 'href': 'https://www.larazon.es/encuesta'}
    assert pages.sheet_rows(POLL | {'url': None}, '/casas/nc-report', 'E', 'R')[7] == {'label': 'Fuente', 'text': 'Wikipedia, lista de sondeos', 'href': None}
    assert pages.sheet_rows({'date': '2026-11-09', 'pollster': 'X'}, '/casas/x', 'E', 'R')[3]['text'] == '9 nov 2026'  # the poll's date is its last fieldwork day
```

- [ ] **Step 2: Correr los tests y verlos fallar**

Run: `pytest tests/test_pages.py -q -k "vote_rows or distance_to_average or weight_card or poll_kpis"`
Expected: FAIL (atributos ausentes).

- [ ] **Step 3: Implementar las funciones en `sondeos.py` con las firmas de arriba** y re-exportarlas en `pages/__init__.py`. `mtype_label` se importa de `.casas`; `RATING_COLORS`, `fieldwork_label` de `.promedios` (cuidado con los imports circulares: `promedios` no importa `sondeos`).

- [ ] **Step 4: Correr los tests y verlos pasar**

Run: `pytest tests/test_pages.py -q -k "vote_rows or distance_to_average or weight_card or poll_kpis"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add mtpy/lib/pages/sondeos.py mtpy/lib/pages/__init__.py tests/test_pages.py
git commit -m "Página de un sondeo: filas del voto, peso, cifras y ficha

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Contexto, controlador, ruta y plantilla

**Files:**
- Modify: `mtpy/lib/pages/sondeos.py` (`sondeo_context`), `mtpy/lib/pages/common.py:67-83` (`ROUTES`, dos rutas tras `/{scope}/escanos`), `mtpy/lib/pages/__init__.py`
- Create: `mtpy/controllers/Sondeos.py`, `web/templates/sondeo.html`
- Modify: `web/dist/css/site.css` (junto a `.hemi-grid`, línea 678, y en el bloque móvil de la línea 1052)
- Test: `tests/test_pages.py` (sección «Detalle de un sondeo»)

**Interfaces:**
- Consumes: Tareas 2 y 4; `common_context`, `part`, `bundle_part`, `event_query`, `round_series`, `chart_polls` (Tarea 3), `catalog`, `ink_color`, `block_color`, `pollster_href`, `check_date` (`webapi`), `resolve_event_scope`.
- Produces: `sondeos.sondeo_context(state: dict, date: str, pollster: str, sponsor: Optional[str] = None) -> dict` con, además de `common_context(state, None, event=True)` (su `event_bar['sections'][i]['href']` reescrito como `'/' + scope + query + '#' + id`): `poll`, `poll_title`, `house` (`{'name', 'href', 'type'}`), `fieldwork` (`fieldwork_text`, `None` sin campo), `poll_date` (`fmt_date(date)`, el último día de campo), `event_label` (`event_bar.title ~ ' · ' ~ fecha`), `kpis`, `vote_rows`, `has_seats`, `has_effects`, `average_date` (el día de `series` usado, `None` sin él), `weight` (`weight_card`), `weights_date` (`weights['date']`, `None` sin la parte), `sheet`, `back_href` (`'/' + scope + query + '#tabla'`), `detail_href`, `image_file` (`'demoscopio-sondeo-{scope}-{date}-{slug casa}[-{slug medio}].png'`), `image_sign` (`site_host ~ ' · ' ~ author_handle`), `chart_file` (`'demoscopio-encuestadora-{scope}-{slug casa}-{date}.png'`) e `initial` (`state`, `meta`, `series` (`round_series`), `polls` (`chart_polls` sólo con los sondeos de la casa: mismo `pollster_id`, o mismo `pollster` si falta), `projection` y `vote` (partes del modo), `catalog` (`{'parties': [...], 'bmaps'}` como en `promedios_context`), `inks`, `poll` (`date`, `pollster_id`, `sponsor`, `pollster` y los porcentajes), `average`, `seats` (o `None`), `n_seats`, `majority`). Errores: 404 `poll not found` sin parte `polls` o sin sondeo; los de `common_context`.
- Controlador `Sondeos(Page)`, `active = None`, `async def detail_action(self, scope, date, pollster, sponsor=None)`: `code = pages.resolve_event_scope(scope)`; `check_date(date)` (400 `invalid date`); `context = pages.sondeo_context(self.state(code), date, pollster, sponsor)`; `self.render('sondeo.html', canonical=webapi.settings()['site_url'] + context['detail_href'], **context)`.
- Rutas: `('/{scope}/sondeos/{date}/{pollster}', 'sondeos', 'detail', ['GET'])` y `('/{scope}/sondeos/{date}/{pollster}/{sponsor}', 'sondeos', 'detail', ['GET'])`.

- [ ] **Step 1: Escribir los tests que fallan**

Un ayudante y los tests de la página (los sondeos de la casa 1 comparten `pollster_id` con la parte `weights` del fixture):

```python
DETAIL_POLLS = [
    {'date': '2026-10-05', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'start_date': '2026-10-01', 'end_date': '2026-10-04',
     'sample_size': 2000, 'mtype': 'cati', 'rating': 71.2, 'weight': 1.265, 'weight_over': 1.0, 'weight_sample': 1.265,
     'weight_rating': 1.0, 'url': 'https://www.elmundo.es/sondeo', 'PP': 33.5, 'seats': {'PP': [150, 152]}},
    {'date': '2026-09-24', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': 'El Mundo', 'sample_size': 1000, 'rating': 71.2,
     'weight': 0.8, 'PP': 32.0},
    {'date': '2026-10-01', 'pollster_id': 3, 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1200, 'rating': 80.0, 'weight': 0.9, 'PP': 32.4},
    {'date': '2026-10-01', 'pollster_id': 3, 'pollster': 'GAD3', 'sponsor': 'El Mundo', 'sample_size': 1200, 'rating': 80.0, 'weight': 0.9, 'PP': 32.9},
    {'date': '2026-10-03', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': 'Tracking', 'sample_size': 500, 'rating': 71.2,
     'weight': 0.0, 'weight_over': 0.0, 'weight_sample': 0.6, 'weight_rating': 1.0, 'PP': 33.0},
]


def write_detail_polls(fs, polls=DETAIL_POLLS, scope='es', run_id=RUN):
    """La parte `polls` del run con `polls` (sólo el PP, como el fixture), una serie de tres días alrededor de ellos y una
    parte `weights` leída el 8 de octubre cuya recta empieza el 26 de septiembre, con la fila del sondeo del día 5."""
    writer = bundle.BundleWriter(fs)
    part = fixture_data('polls') | {'polls': polls}
    writer.write_json(bundle.path_part(scope, run_id, 'polls'), 'polls', part, scope, run_id=run_id)
    series = {'dates': ['2026-09-20', '2026-10-01', '2026-10-05'], 'parties': ['PP'], 'mean': {'PP': [31.0, 31.5, 32.0]},
              'lo': {'PP': [30.0, 30.5, 31.0]}, 'hi': {'PP': [32.0, 32.5, 33.0]}}
    writer.write_json(bundle.path_part(scope, run_id, 'series'), 'series', series, scope, run_id=run_id)
    weights = fixture_data('weights') | {
        'date': '2026-10-08', 'parties': ['PP'],
        'rows': [{'name': 'PP', 'date': '2026-10-05', 'pollster_id': 1, 'pollster': 'Sigma Dos', 'sponsor': None, 'sample_size': 2000,
                  'pct': 33.5, 'pct_adj': 33.2, 'weight': 1.24, 'weight_kernel': 0.98, 'weight_over': 1.0, 'weight_sample': 1.265,
                  'weight_rating': 1.0}],
        'line': {'PP': {'dates': ['2026-09-26', '2026-10-08'], 'mean': [31.2, 32.0], 'lo': [30.2, 31.0], 'hi': [32.2, 33.0]}},
    }
    writer.write_json(bundle.path_part(scope, run_id, 'weights'), 'weights', weights, scope, run_id=run_id)
    webapi.site().files.clear()
    return writer


def test_sondeo_page_renders_the_card_the_detail_and_the_sheet(api, fresh_app):
    write_detail_polls(fresh_app.fs)
    status, headers, body = call(api, '/es/sondeos/2026-10-05/sigma-dos')
    page = html(body)
    assert status == 200 and headers['cache-control'] == 'public, max-age=60'
    assert '<title>Demoscopio · Sigma Dos · 5 oct 2026 · Elecciones generales 2026</title>' in page
    assert '<link rel="canonical" href="https://demoscopio.test/es/sondeos/2026-10-05/sigma-dos">' in page
    assert 'id="event-bar"' in page and 'href="/es?mode=forecast#promedio"' in page  # the index goes back to the event page
    assert '<a href="/es?mode=forecast#tabla">← Todas las encuestas</a>' in page
    for card in ('poll-card', 'vote', 'weight', 'house', 'sheet', 'closing'):
        assert 'id="%s"' % card in page
    assert 'class="card shot poll-card"' in page and 'data-file="demoscopio-sondeo-es-2026-10-05-sigma-dos.png"' in page
    assert 'id="poll-bars"' in page and 'id="poll-hemi"' in page and 'Escaños según Sigma Dos' in page
    assert 'Trabajo de campo 1 oct 2026 – 4 oct 2026' in page and '2.000' in page and 'ublicad' not in page  # the date is the last fieldwork day
    assert pages.mtype_label('cati') in page
    kpis = page[page.index('id="poll-card"'):page.index('id="vote"')]
    assert '>Rating<' in kpis and '>71,2<' in kpis and '>Distancia al promedio<' in kpis and '>1,5<' in kpis  # 33.5 − 32.0
    vote = page[page.index('id="vote"'):page.index('id="weight"')]
    assert '<th scope="col">Escaños</th>' in vote and '150–152' in vote and '>33,5<' in vote and '>32,0<' in vote
    assert 'Promedio del 5 oct 2026' in vote
    weight = page[page.index('id="weight"'):page.index('id="house"')]
    assert '>Solape<' in weight and '>1,27<' in weight and 'En el promedio del 8 oct 2026' in weight
    sheet = page[page.index('id="sheet"'):page.index('id="closing"')]
    assert '<a href="/casas/sigma-dos">Sigma Dos</a>' in sheet and 'href="https://www.elmundo.es/sondeo"' in sheet
    assert '<a href="/casas/sigma-dos"' in page and 'id="house-chart"' in page
    assert 'data-file="demoscopio-encuestadora-es-sigma-dos-2026-10-05.png"' in page
    assert_public_wording(page, 'sondeo')
    data = initial(body)
    assert set(data) == {'state', 'meta', 'series', 'polls', 'projection', 'vote', 'catalog', 'inks', 'poll', 'average', 'seats',
                         'n_seats', 'majority'}
    assert [p['date'] for p in data['polls']['polls']] == ['2026-10-05', '2026-09-24', '2026-10-03']  # only the house's polls, in the order of the part
    assert data['poll']['PP'] == 33.5 and data['average'] == {'PP': 32.0} and data['seats'] == {'PP': [150, 152]}
    assert data['n_seats'] == 350 and data['majority'] == 176


def test_sondeo_page_without_seats_factors_or_parts_degrades(api, fresh_app):
    write_detail_polls(fresh_app.fs)
    page = html(call(api, '/es/sondeos/2026-09-24/sigma-dos/el-mundo')[2])
    assert 'Sigma Dos para El Mundo' in page and 'id="poll-hemi"' not in page and 'Sigma Dos no publicó escaños con este sondeo' in page
    vote = page[page.index('id="vote"'):page.index('id="weight"')]
    assert '<th scope="col">Escaños</th>' not in vote and 'Promedio del 20 sep 2026' in vote
    weight = page[page.index('id="weight"'):page.index('id="house"')]
    assert weight.count('>–<') >= 3 and 'ya no pesa en el promedio del 8 oct 2026' in weight  # before the line starts (26 sep)
    assert 'Wikipedia, lista de sondeos' in page and initial(call(api, '/es/sondeos/2026-09-24/sigma-dos/el-mundo')[2])['seats'] is None
    # inside the window but without a row: no claim about the window
    inside = html(call(api, '/es/sondeos/2026-10-01/gad3/abc')[2])
    assert 'no entra en la recta del promedio del 8 oct 2026' in inside and 'ventana' not in inside[inside.index('id="weight"'):inside.index('id="house"')]
    # a poll of weight 0 (an overlapping wave): never «fuera de la ventana»
    zero = html(call(api, '/es/sondeos/2026-10-03/sigma-dos/tracking')[2])
    assert 'no entra en el promedio: su peso es 0' in zero and 'se solapa con otro sondeo de la misma casa' in zero and 'ventana' not in zero[zero.index('id="weight"'):zero.index('id="house"')]
```

Y un run sin las partes `weights` ni `house-effects` (escribir el paquete en otro `run_id` sin ellas, con `write_bundle(..., manifest=False)` más el borrado de los dos ficheros si el `FileSystem` lo permite —comprobar `mtpy/core/io.py`—, o un `write_bundle` propio que omita esas dos partes): la página responde 200 con `?run=<ese run>`, no lleva «En el promedio del», «ya no pesa» ni «no entra en», y la tabla del voto no tiene la columna «Efecto de casa».

```python
def test_sondeo_page_errors_and_pinned_run(api, fresh_app):
    write_detail_polls(fresh_app.fs)
    assert call(api, '/es/sondeos/2026-10-01/gad3')[0] == 404  # two sponsors that day
    status, _, body = call(api, '/es/sondeos/2020-01-01/nadie')
    assert status == 404 and 'Sondeo no encontrado.' in html(body)
    assert call(api, '/es/sondeos/abc/gad3')[0] == 400 and call(api, '/es-zz/sondeos/2026-10-01/gad3')[0] == 404
    status, _, body = call(api, '/es/sondeos/2026-10-01/gad3/abc', query='mode=nowcast&run=' + RUN)
    page = html(body)
    assert status == 200 and 'GAD3 para ABC' in page
    assert 'href="/es?mode=nowcast&amp;run=%s#tabla"' % RUN in page and 'href="/es?mode=nowcast&amp;run=%s#escanos"' % RUN in page
    assert '<link rel="canonical" href="https://demoscopio.test/es/sondeos/2026-10-01/gad3/abc">' in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': True}
```

(El `site_url` de los tests: copiar el que usen los tests de canonical existentes, por ejemplo `test_casa_profile...`, en lugar de `https://demoscopio.test` si es otro.)

- [ ] **Step 2: Correr los tests y verlos fallar**

Run: `pytest tests/test_pages.py -q -k sondeo_page`
Expected: FAIL (404 `Página no encontrada.`: la ruta no existe).

- [ ] **Step 3: Implementar `sondeo_context`, el controlador, las rutas y la plantilla**

`sondeo_context`: `context = common_context(state, None, event=True)`; `polls = bundle_part(site().run_file, scope, run, 'polls')` → 404 `poll not found` si `None`; `poll = find_poll(...)`; `series = part(scope, run, 'series')`; `average = series_at(series, date)`; `average_date` = el día usado (devolverlo con una variante o calcularlo igual que `series_at`); `weights = bundle_part(..., 'weights')`; `effects` = `{row['name']: row['effect']}` de `bundle_part(..., 'house-effects')` para el `pollster_id` (`None` sin parte o sin filas); `colors` del catálogo de la página; las partes `projection` y `vote` del modo (`part(scope, run, 'projection', mode=mode)`); `house_polls` por `pollster_id`; el resto, con las funciones de la Tarea 4. `run_text` de la ficha: `'Actualizado el ' ~ fmt_date(meta.run_at)` más `' (run anclado)'` si `pinned`. Reescribir los `href` de `event_bar['sections']`.

`controllers/Sondeos.py` como `Casas.py` (docstrings numpy). Rutas en `common.ROUTES` tras `('/{scope}/escanos', ...)`.

`sondeo.html` (extiende `base.html`; `{% block title %}{{ poll_title }} · {{ poll_date }} · {{ event_bar.title }}{% if event_bar.year %} {{ event_bar.year }}{% endif %}{% endblock %}`; descripción según el diseño («con trabajo de campo hasta el {poll_date}»); `{% block scripts %}` con `<script type="module" src="/dist/js/pages/sondeo.js">`), `#content.stack`:
1. `<p class="back"><a href="{{ back_href }}">← Todas las encuestas</a></p>`.
2. `m.card_start('poll-card', poll_title, lead, classes='shot poll-card')` con `lead` = «Trabajo de campo {fieldwork} · {muestra} entrevistas · {tipo}» (sin campo, «Fecha {poll_date} · …»; la palabra «publicado» no aparece), `m.card_image_button(image_file, image_sign)` justo después, `<p class="pc-event">{{ event_label }}</p>`, `<div class="poll-grid{% if not has_seats %} single{% endif %}">` con `<div class="chart poll-bars" id="poll-bars" role="img" aria-label="Voto por partido según {{ house.name }}" style="height: {{ 34 * vote_rows|length + 60 }}px"></div>` y, con `has_seats`, `<div class="poll-seats"><p class="csub">Escaños según {{ house.name }}</p><div class="chart hemi" id="poll-hemi" role="img" aria-label="..."></div></div>`; sin `has_seats`, `<p class="muted">{{ house.name }} no publicó escaños con este sondeo.</p>`; `{{ m.kpis(kpis) }}`; «Cómo leerlo» (barras = el sondeo, la marca gris = el promedio del día; los escaños son los que publicó la casa, no un reparto nuestro; la distancia es la media de las diferencias).
3. `m.card_start('vote', 'Voto y escaños', ...)`: tabla `#vote-table` (Partido con `.pn` y `.dot` · Sondeo · Promedio del día · Diferencia (`m.badge(row.diff, 1)`) · Efecto de casa (sólo con `has_effects`) · Escaños (sólo con `has_seats`)); pie `<p class="muted">Promedio del {{ average_date|date }}…</p>` cuando hay promedio; «Cómo leerlo».
4. `m.card_start('weight', 'Peso en el promedio', ...)`: `m.kpis(weight.factors)` (adaptar: `label`/`value`/`unit ''`) y, sólo con `weight.has_part`, un `<p class="muted">` según `weight.status`: `today` → «En el promedio del {today.date|date}: kernel {today.kernel} · peso {today.weight} ({today.party}).»; `zero` → «Este sondeo no entra en el promedio: su peso es 0» más «: se solapa con otro sondeo de la misma casa publicado pocos días antes.» cuando `weight.overlap`, o «.» si no; `outside` → «Este sondeo ya no pesa en el promedio del {weights_date|date}: su fecha está fuera de la ventana del kernel.»; `absent` → «Este sondeo no entra en la recta del promedio del {weights_date|date}.» (`weights_date` va en el contexto). «Cómo leerlo» con F(solape), F(muestra), F(rating), F(kernel) y el enlace `method_href('promedio')`.
5. `m.card_start('house', house.name ~ ' en este ciclo', ..., tools=m.download_button('house-download', house.name ~ ' · ' ~ event_label, 'frente al promedio · ' ~ title_suffix, image_sign, chart_file))`: los controles de `#pollster-controls` de `promedios.html` (ventana `house-window`/`hwindow`, vista `house-view`/`hview` con `blocks`), `<div class="legend" id="house-legend" ... hidden>`, `<div class="chart chart-tall" id="house-chart">`, «Cómo leerlo» (el anillo es este sondeo), enlaces al perfil y al ranking.
6. `m.card_start('sheet', 'Ficha', ...)`: `<table id="sheet-table">` de dos columnas con `sheet` (`<a href rel="noopener">` cuando hay `href`).
7. `m.closing('¿Cómo pesamos cada sondeo?', method_href('promedio'), ..., [{'href': back_href, 'label': 'Todas las encuestas'}, {'href': house.href, 'label': 'Perfil de ' ~ house.name}])`.

CSS: `.poll-card .poll-grid { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: var(--space-4); align-items: center; }`, `.poll-grid.single { grid-template-columns: 1fr; }`, `.poll-card .pc-event { margin: -4px 0 var(--space-3); font-size: 13.5px; color: var(--color-muted); }`, `.poll-card .poll-bars { height: auto; }`, `.poll-seats .csub { margin: 0 0 4px; text-align: center; }`; en el bloque móvil (línea 1052) `.poll-grid { grid-template-columns: 1fr; }`. Colores planos (nada de `color-mix()`).

- [ ] **Step 4: Correr los tests de páginas**

Run: `pytest tests/test_pages.py -q`
Expected: PASS (incluidos los tres de `sondeo_page` y `test_public_wording_uses_encuestadora_and_medio_and_short_names`; añadir a su lista de rutas `('/es/sondeos/2026-10-05/sigma-dos', '')` si la fixture de ese test escribe los sondeos, o dejarla como está).

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add mtpy/lib/pages/sondeos.py mtpy/lib/pages/common.py mtpy/lib/pages/__init__.py mtpy/controllers/Sondeos.py web/templates/sondeo.html web/dist/css/site.css tests/test_pages.py
git commit -m "Página de un sondeo: contexto, controlador, ruta y plantilla

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: El hemiciclo con rótulos de horquilla y el arco «Sin asignar»

**Files:**
- Modify: `web/dist/js/charts/hemicycle.js:24-36` (`hemicycleData`), `:44-48` (`hemicycleLabel`), `:60-75` (`hemicycleOption`), `:167` (`renderHemicycle`, sólo el JSDoc)
- Test: `tests/test_js_charts.py` (junto a `test_hemicycle_writes_the_seat_delta_under_the_seats`, línea 172)

**Interfaces:**
- Produces: `hemicycleData(seats, {colors, order, nSeats, deltas, labels = null})`: cada ítem lleva `text: labels[name] ?? null`; `hemicycleLabel(params)` escribe `{seats|<text>}` cuando `params.data.text` es una cadena, si no el valor; `hemicycleOption(seats, {..., labels = null, rest = null})`: con `rest > 0`, un último ítem `{name: 'Sin asignar', value: rest, itemStyle: {color: '#d8dbe3'}, label: {show: false}, text: null, delta: null}` en la primera serie (fuera de `order`), y el total del centro sigue siendo `nSeats`. Sin `labels` ni `rest`, la opción no cambia. `renderHemicycle(el, seats, opts)` deja además `el.__imageOption = (width, height) => hemicycleOption(seats, {...opts, width, height})`, la opción al tamaño que le dé la imagen de una tarjeta (`card-image.js`, Tarea 8); no se prueba en node (necesita el DOM), se comprueba en la Tarea 10.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_hemicycle_labels_the_arcs_with_a_range_and_fills_the_unassigned_seats():
    """La tarjeta de un sondeo: los arcos se rotulan con la horquilla de la casa y un arco gris completa la cámara."""
    assert run_js('h.hemicycleLabel({name: "PP", value: 137, data: {text: "136–138", delta: null}})') == '{name|PP}\n{seats|136–138}'
    assert run_js('h.hemicycleLabel({name: "PP", value: 137, data: {text: null}})') == '{name|PP}\n{seats|137}'
    data = run_js('h.hemicycleData({PP: 137, PSOE: 100}, {nSeats: 350, labels: {PP: "136–138"}})')
    assert [item['text'] for item in data] == ['136–138', None]
    option = run_js('h.hemicycleOption({PP: 137, PSOE: 100}, {nSeats: 350, majority: 176, width: 600, height: 330, '
                    'labels: {PP: "136–138", PSOE: "99–101"}, rest: 113})')
    items = option['series'][0]['data']
    assert [(item['name'], item['value']) for item in items] == [('PP', 137), ('PSOE', 100), ('Sin asignar', 113)]
    assert items[2]['itemStyle']['color'] == '#d8dbe3' and items[2]['label'] == {'show': False}
    assert any(g.get('style', {}).get('text') == '350' for g in option['graphic'])
    plain = run_js('h.hemicycleOption({PP: 137, PSOE: 100}, {nSeats: 350, majority: 176, width: 600, height: 330})')
    assert len(plain['series'][0]['data']) == 2 and 'text' in plain['series'][0]['data'][0] and plain['series'][0]['data'][0]['text'] is None
```

- [ ] **Step 2: Correr el test y verlo fallar**

Run: `pytest tests/test_js_charts.py::test_hemicycle_labels_the_arcs_with_a_range_and_fills_the_unassigned_seats -q`
Expected: FAIL (`text` ausente, dos ítems).

- [ ] **Step 3: Implementar** en `hemicycle.js` con las firmas de arriba (JSDoc de `labels`, `rest` y `__imageOption`; el tooltip del arco gris dice «Sin asignar: n escaños»; `renderHemicycle` asigna `el.__imageOption` antes de devolver el gráfico).

- [ ] **Step 4: Correr los tests de gráficos**

Run: `pytest tests/test_js_charts.py -q`
Expected: PASS (los tests antiguos del hemiciclo no cambian).

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add web/dist/js/charts/hemicycle.js tests/test_js_charts.py
git commit -m "Hemiciclo: rótulos de horquilla y arco de escaños sin asignar

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `poll-card.js`, la lógica pura de la tarjeta

**Files:**
- Create: `web/dist/js/poll-card.js`
- Test: `tests/test_js_poll_card.py` (nuevo, calcado de `tests/test_js_card_image.py`: carga ECharts, define `window`, importa el módulo como `pc`)

**Interfaces:**
- Consumes: `dayTime` de `charts/series.js` (sólo importado).
- Produces:
  - `voteRows(poll, parties, average = {}) -> [{name, value, avg}]`: los partidos con valor numérico en `poll`, por valor descendente (empates en el orden de `parties`), `avg` el promedio o `null`; al final `{name: 'Otros', value: rest, avg: null}` cuando `rest = round((100 − suma) * 10) / 10 ≥ 0.5`.
  - `pollSeats(seats, nSeats) -> {seats: {name: midpoint}, labels: {name: text}, rest: number|null}`: `midpoint = Math.round((lo + hi) / 2)`, `text = lo === hi ? String(lo) : `${lo}–${hi}``; `rest = nSeats − suma` cuando es > 0, si no `null`; con `seats` vacío o `null`, `{seats: {}, labels: {}, rest: null}`.
  - `highlightSeries(poll, parties, colors) -> object[]`: por partido con valor, `{type: 'scatter', name, id: `highlight:${name}`, silent: true, z: 6, symbol: 'circle', symbolSize: 16, itemStyle: {color: 'transparent', borderColor: colors[name] || OTHERS_COLOR, borderWidth: 3}, tooltip: {show: false}, data: [[dayTime(poll.date), poll[name]]]}` (el mismo `name` que la línea, para que la leyenda lo oculte con ella).

- [ ] **Step 1: Escribir los tests que fallan**

```python
POLL = {'date': '2026-11-09', 'PP': 32.6, 'PSOE': 25.6, 'VOX': 19.3, 'ERC': None}


def test_vote_rows_sort_the_parties_and_add_the_rest():
    rows = run_js('pc.voteRows(%s, ["PSOE", "PP", "VOX", "ERC"], {PP: 31.9, VOX: 18.0})' % json.dumps(POLL))
    assert rows == [{'name': 'PP', 'value': 32.6, 'avg': 31.9}, {'name': 'PSOE', 'value': 25.6, 'avg': None},
                    {'name': 'VOX', 'value': 19.3, 'avg': 18.0}, {'name': 'Otros', 'value': 22.5, 'avg': None}]
    full = run_js('pc.voteRows({PP: 60.0, PSOE: 39.7}, ["PP", "PSOE"])')
    assert [row['name'] for row in full] == ['PP', 'PSOE']  # a rest under 0.5 is not shown
    assert run_js('pc.voteRows({PP: 30, PSOE: 30}, ["PSOE", "PP"])')[0]['name'] == 'PSOE'  # ties keep the party order


def test_poll_seats_take_the_midpoints_label_the_ranges_and_count_the_rest():
    seats = run_js('pc.pollSeats({PP: [136, 138], PSOE: [100, 100], VOX: [67, 69]}, 350)')
    assert seats == {'seats': {'PP': 137, 'PSOE': 100, 'VOX': 68}, 'labels': {'PP': '136–138', 'PSOE': '100', 'VOX': '67–69'}, 'rest': 45}
    assert run_js('pc.pollSeats({PP: [200, 200], PSOE: [160, 160]}, 350)')['rest'] is None  # over the chamber: nothing to fill
    assert run_js('pc.pollSeats(null, 350)') == {'seats': {}, 'labels': {}, 'rest': None}


def test_highlight_series_ring_each_party_of_the_poll():
    series = run_js('pc.highlightSeries(%s, ["PP", "ERC"], {PP: "#1d84ce"})' % json.dumps(POLL))
    assert len(series) == 1 and series[0]['name'] == 'PP' and series[0]['id'] == 'highlight:PP' and series[0]['silent'] is True
    assert series[0]['itemStyle'] == {'color': 'transparent', 'borderColor': '#1d84ce', 'borderWidth': 3}
    assert series[0]['data'][0][1] == 32.6 and series[0]['tooltip'] == {'show': False}
```

- [ ] **Step 2: Correr los tests y verlos fallar**

Run: `pytest tests/test_js_poll_card.py -q`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Implementar `web/dist/js/poll-card.js`** con las tres funciones y su JSDoc (`OTHERS_COLOR` de `catalog.js`).

- [ ] **Step 4: Correr los tests y verlos pasar**

Run: `pytest tests/test_js_poll_card.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add web/dist/js/poll-card.js tests/test_js_poll_card.py
git commit -m "poll-card.js: filas del voto, escaños de la casa y anillo del sondeo

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `card-image.js` redibuja cada gráfico con la opción del tamaño de la imagen

**Files:**
- Modify: `web/dist/js/card-image.js:46-55` (`chartOptions`), `:70-90` (`prepareClone`)
- Test: `tests/test_js_card_image.py` (tras `test_chart_options_take_the_mounted_charts_with_an_id`)

**Interfaces:**
- Consumes: `el.__imageOption` (Tarea 6).
- Produces: `chartOptions(card) -> [{id, option, imageOption}]` con `imageOption` la función `el.__imageOption` del contenedor o `null`; `imageOptionOf(chart, width, height) -> object` (nueva, exportada): `chart.imageOption ? chart.imageOption(width, height) : chart.option`; `prepareClone` dibuja cada gráfico con `imageOptionOf(chart, width, height)` al tamaño medido en la copia. Las páginas de los análisis, cuyos gráficos no tienen `__imageOption`, no cambian.

- [ ] **Step 1: Escribir el test que falla**

```python
def test_chart_options_carry_the_image_option_of_a_chart_and_image_option_of_picks_it():
    """El hemiciclo de la tarjeta de un sondeo se redibuja al tamaño de la imagen con `el.__imageOption`; los gráficos sin
    ella siguen con la opción de la pantalla."""
    card = ('{querySelectorAll: () => [{id: "a", __chart: {getOption: () => ({series: [1]})}, __imageOption: (w, h) => ({w, h})}, '
            '{id: "b", __chart: {getOption: () => ({x: 2})}}]}')
    assert run_js('c.chartOptions(%s).map((chart) => [chart.id, typeof chart.imageOption])' % card) == [['a', 'function'], ['b', 'object']]
    assert run_js('c.chartOptions(%s)[1].imageOption' % card) is None
    assert run_js('c.imageOptionOf({option: {a: 1}, imageOption: (w, h) => ({w, h})}, 500, 300)') == {'w': 500, 'h': 300}
    assert run_js('c.imageOptionOf({option: {a: 1}, imageOption: null}, 500, 300)') == {'a': 1}
```

- [ ] **Step 2: Correr el test y verlo fallar**

Run: `pytest tests/test_js_card_image.py -q`
Expected: FAIL (`imageOption` ausente, `imageOptionOf` no exportada).

- [ ] **Step 3: Implementar** en `card-image.js`: `chartOptions` añade `imageOption: typeof el.__imageOption === 'function' ? el.__imageOption : null`; `imageOptionOf` exportada; en `prepareClone`, `chartDataUrl(imageOptionOf(chart, width, height), {width, height, scale})`. Actualizar los JSDoc y el comentario de cabecera (un gráfico que calcula su geometría en píxeles deja `__imageOption`).

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_js_card_image.py tests/test_pages.py -q -k "card_image or study_cards"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add web/dist/js/card-image.js tests/test_js_card_image.py
git commit -m "card-image: la opción de cada gráfico al tamaño de la imagen

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `pages/sondeo.js`: barras, hemiciclo, la casa en el ciclo y las descargas

**Files:**
- Create: `web/dist/js/pages/sondeo.js`
- Modify: `web/templates/sondeo.html` (sólo si el cableado necesita un `id` o un atributo más)
- Test: `tests/test_pages.py` (una aserción más en `test_sondeo_page_renders_the_card_the_detail_and_the_sheet`: `'/dist/js/pages/sondeo.js' in page`)

**Interfaces:**
- Consumes: `readInitial`, `mountWhenReady`, `wireControls`, `markSegments`, `showError`, `tryDraw`, `mapNames` (`pages/common.js`); `Catalog` (`catalog.js`); `renderBars` (`charts/bars.js`); `renderHemicycle` (`charts/hemicycle.js`, Tarea 6); `seriesOption`, `seriesImageOption`, `setWindow`, `chartZoom` (`charts/series.js`); `mountChart` (`charts/base.js`); `seriesViews`, `chosenNames`, `fillLegend`, `WINDOWS` (`pages/series-wire.js`); `markLegend` (`pages/legend-dom.js`); `voteRows`, `pollSeats`, `highlightSeries` (Tarea 7); `wireDownload`, `fileSlug` (`share.js`); `wireCardImages` (`card-image.js`); `fmtPct` (`format.js`).
- Produces: la página dibujada: `#poll-bars` con `renderBars(el, voteRows(poll, parties, average), {value: 'value', lo: 'avg', hi: 'avg', colors, fullnames, formatter: (x) => fmtPct(x)})` (los colores del catálogo y `OTHERS_COLOR` para «Otros»); `#poll-hemi` con `renderHemicycle(el, seats, {colors, order: (names) => catalog.order(names), nSeats, majority, fullnames, labels, rest})` de `pollSeats(initial.seats, initial.n_seats)`; `#house-chart` como el `wireHouse` de `promedios.js:140-215` con la casa fija (sin `select`), `state = {view: 'partidos', window: 'all', selected: null, layers: 'ambos'}`, `dataOf` como el de `promedios.js:238-255` (`asOf: meta.as_of`, `when: vote.when`, `legend: 'hidden'`), dibujado con `mountChart(el, {...seriesOption(data), series: [...seriesOption(data).series, ...highlightSeries(poll, current.parties, current.colors)]})` (una función `optionWithRing(data)`), los controles `#house-window`/`#house-view`, la leyenda `#house-legend`, y `wireDownload(#house-download, () => ({...seriesImageOption(data, chartZoom(chart)), series: [...]}), showError)` (si `share.js` ya acepta `caption` al construir, pasar la de la vista como hace `promedios.js:185`); `wireCardImages(document, showError)` para la tarjeta.

- [ ] **Step 1: Añadir la aserción del módulo al test de la Tarea 5 y verla fallar**

Run: `pytest tests/test_pages.py::test_sondeo_page_renders_the_card_the_detail_and_the_sheet -q`
Expected: FAIL si la plantilla aún no carga el módulo (si la Tarea 5 ya lo puso, pasar a los pasos siguientes).

- [ ] **Step 2: Escribir `web/dist/js/pages/sondeo.js`** (cabecera de comentario como la de `promedios.js`; `wireControls(); mountWhenReady(paint);`; cada gráfico dentro de `tryDraw` para que un fallo no pare los demás).

- [ ] **Step 3: Comprobación en node de que el módulo carga**

Run: `node --input-type=module -e 'import {createRequire} from "node:module"; const echarts = createRequire(import.meta.url)("./web/dist/vendor/echarts-5.6.0.min.js"); globalThis.window = {echarts}; globalThis.document = {getElementById: () => null, querySelectorAll: () => [], addEventListener: () => {}, readyState: "loading"}; await import("./web/dist/js/pages/sondeo.js"); console.log("ok")'`
Expected: `ok` (sin errores de importación; si `mountWhenReady` necesita más del DOM, basta con que la importación resuelva).

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_pages.py tests/test_js_poll_card.py tests/test_js_charts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short && git log -1 --oneline
git add web/dist/js/pages/sondeo.js web/templates/sondeo.html tests/test_pages.py
git commit -m "Página de un sondeo: barras, hemiciclo, la casa en el ciclo y descargas

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Comprobación en el navegador y la prueba de humo

**Files:**
- Modify: `deploy/check-api.sh:134` (dos `check` tras `check /es-zz 404 html`)
- Scratchpad (fuera del repo): `check_app.py` (envuelve `api` de `api.py` sirviendo `/dist/` desde `web/dist`, como la comprobación del 2026-10-27) y el guion de Chrome

**Interfaces:**
- Consumes: todo lo anterior; el paquete local `files/site/v1` (`S3_BUCKET=` vacío) con un run de `es` que tenga sondeos con `seats`.

- [ ] **Step 1: La prueba de humo**

En `deploy/check-api.sh`, tras la línea 134: `check "/es/sondeos/abc/nadie" 400 html` y `check "/es/sondeos/2020-01-01/nadie" 404 html`. Run: `bash -n deploy/check-api.sh`. Expected: sin errores de sintaxis.

- [ ] **Step 2: Arrancar la web local**

Run (desde la raíz, con el `.env` local): `S3_BUCKET= python -m uvicorn check_app:app --port 8000` donde `check_app.py` del scratchpad importa `api` de `api.py` y sirve `/dist/*` desde `web/dist` (un ASGI mínimo que lee el fichero y pone el `content-type` por extensión). Expected: `curl -s http://127.0.0.1:8000/es | grep -c '/sondeos/'` > 0.

- [ ] **Step 3: Capturas con Chrome sin cabeza**

Tomar de `/es` el primer enlace `/es/sondeos/...` de un sondeo con escaños (buscar en `files/site/v1/runs/es/<run>/polls.json` uno con `seats`). Run: `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --window-size=1300,2400 --screenshot=/private/tmp/claude-501/.../scratchpad/sondeo-desktop.png "http://127.0.0.1:8000/es/sondeos/<fecha>/<casa>"` y otra con `--window-size=390,2400` (`sondeo-mobile.png`); una tercera de un sondeo sin escaños. Abrir las capturas (Read) y comprobar: la tarjeta con dos columnas en escritorio y apiladas en el móvil, barras con la marca del promedio, hemiciclo con rótulos de horquilla y «Sin asignar» si procede, fila de cifras, tabla del voto, peso, gráfico de la casa con el anillo, ficha. Sin errores en la consola: añadir `--enable-logging=stderr --v=0` y revisar que no haya `Uncaught`.

- [ ] **Step 4: El PNG de la tarjeta**

Con Chrome arrancado con `--remote-debugging-port=9222` y la página abierta, un guion de node en el scratchpad (WebSocket nativo de node 22+ contra el `webSocketDebuggerUrl` de `http://127.0.0.1:9222/json`) que ejecute en la página, con `Runtime.evaluate` (`awaitPromise: true`), `import('/dist/js/card-image.js').then(async (m) => { const blob = await m.cardImage(document.getElementById('poll-card'), {sign: 'demoscopio.es · @luis_sancho'}); return new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); }); })`, guarde el PNG y compruebe con `python3 -c "from PIL import Image; ..."` (o `file`) que mide 2400 px de ancho. Abrirlo (Read) y comprobar la tarjeta entera con la firma. Repetir con Chrome a `--window-size=390,2400`: el PNG debe medir también 2400 px y su hemiciclo tener el mismo tamaño que en el de escritorio (es lo que prueba `__imageOption`). Si el guion no sale en media hora, dejarlo anotado en el informe final: Luis hace esa comprobación a mano pulsando el botón en el ordenador y en el móvil.

- [ ] **Step 5: Toda la batería**

Run: `pytest -q -x`
Expected: PASS (ningún test antiguo roto; los de JS se saltan sin `node`).

- [ ] **Step 6: Commit**

```bash
git status --short && git log -1 --oneline
git add deploy/check-api.sh
git commit -m "check-api: la página de un sondeo

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Tras el plan (pasos de Luis)

1. `update`/`compute` + `publish` de los ámbitos para que los runs lleven las cuatro columnas nuevas de `polls`.
2. Pasear dos o tres sondeos (con y sin escaños) en el ordenador y en el móvil, y probar las dos descargas.
3. Imagen y `update-service` en ECS; `bash deploy/check-api.sh https://demoscopio.es`.
