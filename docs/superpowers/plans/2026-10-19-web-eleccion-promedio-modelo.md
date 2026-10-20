# Web: el promedio de sondeos y el modelo en el detalle de una elección — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** En el detalle de una elección pasada, la curva del promedio de sondeos del ciclo con los sondeos como puntos y el resultado oficial al final, y lo que el modelo dijo a 6 días de la elección (voto y escaños) junto a las encuestas y en una tarjeta compacta por horizonte.

**Architecture:** El backtest guarda la serie diaria del ajuste del horizonte más corto de cada elección (`series.csv`, fuera de git); `publish event` la corta y la publica como parte nueva `series` de la elección (schema `event-series@1`, gemelo CSV, ruta de la API automática). La página lee la parte, la embebe en la isla `#initial-data` y la dibuja con el gráfico de series de Promedios, cuyo cableado se extrae a un módulo compartido y gana la marca del resultado oficial. Las cifras a 6 días salen de la parte `model` ya publicada: dos filas fijas en el pie de «Todas las encuestas» y la tarjeta «Lo que dijo el modelo» con un selector de horizonte.

**Tech Stack:** Python 3.11 (pandas, Jinja2), ECharts 5.6 en JS vanilla (módulos ES), pytest (`pytest` del shim de pyenv) y node 25 para los módulos JS.

**Spec:** `docs/superpowers/specs/2026-10-19-web-eleccion-promedio-modelo-design.md`

## Global Constraints

- Rama `dev`, **árbol limpio al empezar**: Luis tiene sin commit un lote propio (catálogo de partidos: `mtpy/lib/publish/parties.py`, `tests/fixtures/bundle/parties.json` y cambios en `bundle.py`, `pages/__init__.py`, `pages/common.py`, `pages/promedios.py`, `publish/__init__.py`, `webapi/published.py`, `jobs/Publish.py`, `promedios.js`, `test_pages.py`, `test_publish_unit.py`, `test_webapi_unit.py`, `test_jobs_publish.py`, `README.md`, `data/params.json`, `docs/web/contrato.md`) que este plan también toca. Antes de la tarea 1, ese lote tiene que estar commiteado (o guardado) por Luis; el plan no lo integra ni lo deshace. Cada commit añade solo los ficheros de su tarea con `git add <rutas>`; nunca `git add -A`.
- Tests: `pytest -q` (pyenv, Python 3.11); `pytest.ini` convierte `FutureWarning` en error. Hoy recoge 1144 tests (con el lote de Luis); todos en verde antes de empezar. Los tests JS corren con `node` (se saltan sin él) con los arneses de `tests/test_js_charts.py`, `tests/test_js_blocks.py` y `tests/test_js_polls.py`.
- Nombres técnicos fijos (spec D1–D3, P1–P3): tabla del backtest `series.csv` con columnas `event_date, horizon, date, name, mean, lo, hi`; `export_series(fc, names, end=None)`; `case_series(sim, event_date, horizon)`; `shortest_horizon(series)`; parte `series`, schema `event-series` (`event-series@1`) con `{'horizon': int, 'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict}`; gemelo `series.csv` (`date, name, mean, lo, hi`); `export_event_series(source, event_date)`; en la página `model_last(model, event_date)`, `fixed_rows(last, results_rows, parties, colors, event_date)`, `horizon_tables(model)` con la forma nueva; ids HTML `series`, `series-view`, `series-window`, `series-legend`, `series-chart`, `model`, `model-horizon`, `model-table-{h}`, `model-blocks-{h}` y filas `tr.fixed` del pie; JS: opción `result` y `asOfLabel` de `seriesOption`, `axisDatesOf(series, projection, result)`, `setWindow(chart, choice, {series, projection, anchor, result})`, `resultTooltipLine(name, value, date, color)`, `blockResult(result, blocks)`, módulo `web/dist/js/pages/series-wire.js` con `seriesViews`, `chosenNames`, `fillLegend` y `wireAverage(views, state, render, {anchor, inks})`, `fixedRows(modelLast, result, seats, parties, mode)`, `renderFoot(tfoot, count, means, parties, catalog, mode, fixed)`, `wirePollsTable(polls, catalog, {fixed})`.
- Redondeos: el paquete a `PCT_DECIMALS` (2); la isla de la página redondea la serie a 1 decimal (`round_series`), como Promedios. Fechas ISO `YYYY-MM-DD`; en pantalla `fmt_date` («23 jul 2023») en el servidor y `fmtDate` en el navegador, que coinciden.
- Las cifras de escaños y de mayoría son las de la ejecución a horizonte (`model_h`, `model_seats_h`, `p_majority_h`) cuando la fila las trae y no son nulas; si no, las del nowcast (`model`, `model_seats`, `p_majority`). El voto es siempre `model` (spec P2).
- Textos públicos en castellano, en el tono de `eleccion.html` (tres párrafos de «Cómo leerlo»: qué es, cómo leerlo, qué no es). Sin `script` inline ni recursos externos (CSP `default-src 'self'`); la página se lee sin JavaScript (solo los gráficos y los selectores lo necesitan; sin él se ve el primer horizonte y la vista % de voto).
- Docstrings numpydoc en todas las funciones nuevas (AGENTS.md); comentarios y JSDoc en inglés, como el resto del código.

## Review Focus

1. Una elección publicada antes de esta fase (tiene `model` pero no `series.json`): la página sale con 200, la tarjeta del promedio enseña el aviso y lo demás funciona → test en la tarea 5.
2. Un backtest `nowcast_only` (filas sin `model_h`, `model_seats_h`, `p_majority_h`): `horizon_tables` y `model_last` caen al nowcast sin romper → test en la tarea 4.
3. Un partido de la serie o de la tabla que no tiene resultado (o al revés): las filas fijas ponen «–», el resultado solo marca los partidos con cifra y el catálogo del navegador tiene a todos → tests en las tareas 4, 5 y 8.
4. La ventana «3 meses» con el resultado 6 días después del final de la serie: el eje y la ventana incluyen la fecha del resultado, así que el rombo no queda fuera → test en la tarea 6.
5. Escaños no válidos en el horizonte más corto (`seats_valid` falso): la fila «Promedio de sondeos» en la vista Escaños va toda con «—» y la tarjeta del modelo enseña su aviso → tests en las tareas 4, 5 y 8.

---

### Task 1: La serie en el backtest

**Files:**
- Modify: `mtpy/lib/publish/forecast.py:246-275` (`export_series`)
- Modify: `mtpy/lib/backtest.py` (`run_case` 163-342, `run_backtest` 421-480; nueva `case_series` y `shortest_horizon`)
- Modify: `.gitignore`, `backtest/README.md:66`, `deploy/README.md:584-600`
- Test: `tests/test_publish_unit.py` (junto a `test_export_series_stops_at_the_last_fitted_day`, línea 156), `tests/test_backtest_unit.py` (`fake_results` 139, `test_backtest_scope_writes_the_tables_and_the_run_record` 160), `tests/integration/test_backtest.py:8-27`

**Interfaces:**
- Consumes: `export_series(fc, names)` (forecast.py), `synthetic_simulator()` y `synthetic_backtest()` de `tests/fakes.py` (`sim.model` con `forecast`/`fc_stat`, `sim.as_of = Timestamp('2026-10-05')`, `sim.params['names'] == NAMES`; `BACKTEST_EVENTS = ['2019-11-10', '2023-07-23']`, `BACKTEST_HORIZONS = [6, 30]`).
- Produces: `export_series(fc, names, end=None) -> (dict, DataFrame)`; `case_series(sim, event_date: str, horizon: int) -> DataFrame` con columnas `['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi']`; `shortest_horizon(series: DataFrame) -> DataFrame`; `run_case(...)['series']`; `run_backtest(...)['series']` (solo el horizonte más corto por elección); `backtest_scope` escribe `series.csv` en la carpeta del ámbito.

- [ ] **Step 1: Test de `export_series` con `end`** (en `tests/test_publish_unit.py`, tras `test_export_series_without_statistics_gives_null_bounds`)

```python
def test_export_series_cuts_at_the_end_given():
    """`end` corta la serie ese día incluido; un `end` posterior al último día ajustado no cambia nada."""
    import pandas as pd
    sim = synthetic_simulator()
    data, frame = publish.export_series(sim.model, NAMES, end='2026-10-03')
    assert data['dates'][0] == '2026-09-01' and data['dates'][-1] == '2026-10-03' and len(data['dates']) == 33
    assert frame.shape[0] == 33 * 3 and frame['date'].max() == '2026-10-03'
    assert publish.export_series(sim.model, NAMES, end=pd.Timestamp('2026-12-31'))[0]['dates'][-1] == '2026-10-05'
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_publish_unit.py::test_export_series_cuts_at_the_end_given -q`
Expected: FAIL con `TypeError: export_series() got an unexpected keyword argument 'end'`

- [ ] **Step 3: Implementar `export_series(fc, names, end=None)`** en `forecast.py`

`end` (str o Timestamp, opcional): último día exportado; por defecto `fc.date_fit_last` (o el máximo del índice sin él). El corte es `min(end, por defecto)` cuando se da `end`: `table = fc.forecast.loc[:end, names]`. Docstring con el parámetro.

- [ ] **Step 4: Test de `case_series`** (en `tests/test_backtest_unit.py`, antes de `fake_results`)

```python
def test_case_series_is_the_fitted_series_up_to_the_anchor():
    """La serie de un caso es la del Forecaster del simulador hasta `as_of` incluido, con la elección y el horizonte delante."""
    import pandas as pd
    from mtpy.lib import backtest
    from tests.fakes import NAMES, synthetic_simulator
    sim = synthetic_simulator()
    sim.as_of = pd.Timestamp('2026-10-03')
    frame = backtest.case_series(sim, '2026-11-29', 57)
    assert list(frame.columns) == ['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi']
    assert frame.shape[0] == 33 * 3 and frame['date'].max() == '2026-10-03' and set(frame['name']) == set(NAMES)
    assert (frame['event_date'] == '2026-11-29').all() and (frame['horizon'] == 57).all()
```

- [ ] **Step 5: Ejecutar y ver que falla**

Run: `pytest tests/test_backtest_unit.py::test_case_series_is_the_fitted_series_up_to_the_anchor -q`
Expected: FAIL con `AttributeError: module 'mtpy.lib.backtest' has no attribute 'case_series'`

- [ ] **Step 6: Implementar `case_series(sim, event_date, horizon)`** en `backtest.py` y llamarla desde `run_case`

Importación perezosa dentro de la función (`from .publish.forecast import export_series`), como hace `backtest_scope` con `provenance`: así `backtest` no importa `publish` al cargarse. `_, frame = export_series(sim.model, sim.params['names'], end=sim.as_of)`; `frame.insert(0, 'horizon', int(horizon))`; `frame.insert(0, 'event_date', event_date)`. En `run_case`, el diccionario devuelto gana `'series': case_series(sim, event_date, horizon)` (calculada tras `fit_forecast`; el docstring de `run_case` la describe junto a `shares`, `seats`...).

- [ ] **Step 7: Ejecutar y ver que pasa**

Run: `pytest tests/test_backtest_unit.py::test_case_series_is_the_fitted_series_up_to_the_anchor tests/test_publish_unit.py::test_export_series_cuts_at_the_end_given -q`
Expected: PASS

- [ ] **Step 8: Test de `run_backtest` con el horizonte más corto** (en `tests/test_backtest_unit.py`, tras `test_run_backtest_passes_min_events_to_every_case`)

```python
def test_run_backtest_keeps_the_series_of_the_shortest_horizon_of_each_election(monkeypatch):
    """De las series de los casos solo queda la del horizonte más corto que corrió en cada elección (el 6, o el 30
    si el caso a 6 días se omitió); las demás tablas siguen igual."""
    import warnings
    import pandas as pd
    from mtpy.lib import backtest
    from tests.fakes import BACKTEST_EVENTS, BACKTEST_HORIZONS, synthetic_backtest
    tables = synthetic_backtest()

    def case(scope, event_date, horizon, **kwargs):
        if (event_date, horizon) == ('2019-11-10', 6):
            raise ValueError('no usable polls')
        pick = lambda t: t.loc[(t['event_date'] == event_date) & (t['horizon'] == horizon)].reset_index(drop=True)
        series = pd.DataFrame({'event_date': [event_date] * 2, 'horizon': [horizon] * 2, 'date': ['2023-07-01', '2023-07-02'],
                               'name': ['PP', 'PP'], 'mean': [30.0, 30.5], 'lo': [29.0, 29.5], 'hi': [31.0, 31.5]})
        return {'shares': pick(tables.shares), 'seats': pick(tables.seats), 'blocks': pick(tables.blocks),
                'provinces': pd.DataFrame(), 'meta': pd.DataFrame([{'event_date': event_date, 'horizon': horizon}]),
                'series': series}

    monkeypatch.setattr(backtest, 'run_case', case)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = backtest.run_backtest(scope='es', events=BACKTEST_EVENTS, horizons=BACKTEST_HORIZONS)
    kept = out['series'][['event_date', 'horizon']].drop_duplicates()
    assert sorted(map(tuple, kept.itertuples(index=False))) == [('2019-11-10', 30), ('2023-07-23', 6)]
    assert out['series'].shape[0] == 4 and out['metrics'].shape[0] == 3
    assert list(backtest.shortest_horizon(pd.DataFrame(columns=out['series'].columns)).columns) == list(out['series'].columns)
```

- [ ] **Step 9: Ejecutar y ver que falla**

Run: `pytest tests/test_backtest_unit.py::test_run_backtest_keeps_the_series_of_the_shortest_horizon_of_each_election -q`
Expected: FAIL con `KeyError: 'series'`

- [ ] **Step 10: Implementar `shortest_horizon(series)` y usarla en `run_backtest`**

`shortest_horizon`: de una tabla con `event_date` y `horizon`, las filas cuyo `horizon` es el mínimo de su `event_date` (una tabla vacía vuelve vacía con sus columnas). En `run_backtest`, `parts` gana `'series': []` (cada caso aporta `case['series']`) y, tras concatenar, `out['series'] = shortest_horizon(out['series'])`. Docstring de `run_backtest`: `series` en la lista de tablas devueltas.

- [ ] **Step 11: Actualizar `fake_results` y el test de `backtest_scope`**

En `fake_results()` añadir `'series': pd.DataFrame(columns=['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi'])`. En `test_backtest_scope_writes_the_tables_and_the_run_record`, el docstring pasa a «sus ocho tablas» y el listado esperado de la carpeta gana `'series.csv'` (en orden alfabético, entre `'seats.csv'` y `'shares.csv'`).

- [ ] **Step 12: Ejecutar los tests del backtest y de publicación**

Run: `pytest tests/test_backtest_unit.py tests/test_jobs_backtest.py tests/test_publish_unit.py -q`
Expected: PASS

- [ ] **Step 13: `.gitignore` y documentación**

- `.gitignore`: línea `backtest/results/**/series.csv` bajo `files/` con el comentario `# la serie diaria del backtest (derivada y pesada; ver backtest/README.md)`.
- `backtest/README.md:66`: añadir `series.csv` a la lista («`series.csv` (una fila por día y partido: la serie diaria del promedio, con su banda, del ajuste del horizonte más corto de cada elección; fuera de git por tamaño, la lee `publish event`)»).
- `deploy/README.md`, «Backtest del modelo»: «las siete tablas» → «las ocho tablas (`shares`, `seats`, `blocks`, `provinces`, `series`, `meta`, `metrics`, `by_horizon`)» y una frase: `series.csv` no va en git; `publish event` la necesita en la máquina donde corrió el backtest.
- `tests/integration/test_backtest.py::test_run_case_2023_30_days`: al final, dos asserts (Luis los corre contra la base):

```python
    series = case['series']
    assert list(series.columns) == ['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi'] and (series['horizon'] == 30).all()
    assert series['date'].max() == meta.loc[0, 'as_of'] and set(series['name']) == set(shares['party'])
```

- [ ] **Step 14: Commit**

```bash
git add mtpy/lib/publish/forecast.py mtpy/lib/backtest.py .gitignore backtest/README.md deploy/README.md tests/test_publish_unit.py tests/test_backtest_unit.py tests/integration/test_backtest.py
git commit -m "Keep the daily series of the shortest horizon of each election in the backtest"
```

---

### Task 2: La parte `series` de la elección en el paquete

**Files:**
- Modify: `mtpy/lib/bundle.py:69-71` (`EVENT_PARTS`, `EVENT_CSV`) y `SCHEMAS` (junto a `'event-model'`, ~línea 131)
- Create: `tests/fixtures/bundle/event-series.json`
- Modify: `tests/test_bundle_unit.py:116-125`, `tests/test_webapi_unit.py:59-85` (`write_event`) y `646-660`, `deploy/check-api.sh:184`, `docs/web/contrato.md` (tabla de eventos 346-353, viñetas 364-382, ruta `events/{scope}/{date}/{part}` 655-660)

**Interfaces:**
- Produces: `bundle.EVENT_PARTS == ('meta', 'pollsters', 'polls', 'cycle', 'results', 'model', 'series')`; `bundle.EVENT_CSV` termina en `'series'`; `bundle.SCHEMAS['event-series'] == {'horizon': int, 'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict}`; `write_event(fs, scope='es', date='2023-07-23', model=True, series=None)` (`series` por defecto igual a `model`).

- [ ] **Step 1: Actualizar el test de rutas del paquete** (`test_event_paths_follow_the_published_layout_and_reject_bad_input`)

La tupla esperada de `EVENT_PARTS` gana `'series'` al final y se añade:

```python
    assert bundle.path_event_csv('es', '2023-07-23', 'series') == 'events/es/2023-07-23/csv/series.csv'
    assert bundle.SCHEMAS['event-series'] == {'horizon': int, 'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict}
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_bundle_unit.py::test_event_paths_follow_the_published_layout_and_reject_bad_input -q`
Expected: FAIL en la primera aserción (la tupla no tiene `'series'`)

- [ ] **Step 3: Implementar en `bundle.py`** las tres constantes del bloque Interfaces y el fixture

`tests/fixtures/bundle/event-series.json`, con el sobre de `event-cycle.json` (`run_id` y `mode` nulos, `scope` `es`, `generated_at` `2026-10-19T12:00:00Z`) y:

```json
"data": {"horizon": 6, "dates": ["2023-07-15", "2023-07-16", "2023-07-17"], "parties": ["PP", "PSOE", "VOX"],
         "mean": {"PP": [32.2, 32.3, 32.41], "PSOE": [31.1, 31.0, 30.99], "VOX": [12.8, 12.9, 13.01]},
         "lo": {"PP": [31.1, 31.2, 31.31], "PSOE": [null, 29.9, 29.89], "VOX": [11.7, 11.8, 11.91]},
         "hi": {"PP": [33.3, 33.4, 33.51], "PSOE": [32.2, 32.1, 32.09], "VOX": [13.9, 14.0, 14.11]}}
```

(el último día coincide con `shares.model` a 6 días del fixture `event-model`: 32.41, 30.99, 13.01; el `null` de `PSOE.lo` ejercita los huecos de la banda). `test_validate_accepts_every_fixture_and_rejects_a_missing_key` lo valida solo.

- [ ] **Step 4: Ejecutar los tests del paquete**

Run: `pytest tests/test_bundle_unit.py -q`
Expected: PASS

- [ ] **Step 5: `write_event` con la parte y su CSV, y el test del lector** (`tests/test_webapi_unit.py`)

`write_event(fs, scope='es', date='2023-07-23', model=True, series=None)`: `series = model if series is None else series`; con `series`, `tables` gana `'series': pd.DataFrame({'date': ['2023-07-17'], 'name': ['PP'], 'mean': [32.41]})` y el bucle de partes escribe `event-series` desde su fixture; sin ella, salta `'series'` (como `model=False` deja `model-*` sin gemelos). Docstring: «las seis partes… y, con `series`, la serie del promedio». En `test_site_events_and_backtest_read_the_published_files` añadir:

```python
    assert json.loads(s.event('es', '2023-07-23', 'series'))['data']['horizon'] == 6
    assert s.event_csv('es', '2023-07-23', 'series').startswith(b'date')
    with pytest.raises(HttpError) as missing:
        s.event('es', '2019-11-10', 'series')
    assert missing.value.status == 404 and missing.value.message == 'event not published'
```

- [ ] **Step 6: Ejecutar los tests de la API y de las páginas**

Run: `pytest tests/test_webapi_unit.py tests/test_pages.py tests/test_api_asgi.py -q`
Expected: PASS (las páginas no leen la parte todavía; `write_event` escribe una parte más que nadie lee)

- [ ] **Step 7: Prueba de humo y contrato**

- `deploy/check-api.sh`: tras la línea de `pollsters`, `check_published /api/v1/events/es/2023-07-23/series json`.
- `docs/web/contrato.md`: fila `| series | event-series@1 | horizon, dates, parties, mean, lo, hi |` en la tabla de eventos; viñeta: «`series` es la serie diaria del promedio del modelo del horizonte más corto del backtest (`series.csv` de `backtest/results`, normalmente a 6 días), hasta el día en que se congeló: `horizon` y, con la forma de la parte `series` de un run, `dates`, `parties`, `mean`, `lo`, `hi` (2 decimales, `null` en los huecos). Solo existe si el backtest guardó la serie: una elección puede tener `model` y no `series`»; en el párrafo de gemelos, `series` entre los que lo tienen («al republicar sin serie se borran `series.json` y `series.csv`»); en la tabla de rutas, `part` y `nombre` ganan `series`.

- [ ] **Step 8: Commit**

```bash
git add mtpy/lib/bundle.py tests/fixtures/bundle/event-series.json tests/test_bundle_unit.py tests/test_webapi_unit.py deploy/check-api.sh docs/web/contrato.md
git commit -m "Add the series part of a past election to the bundle: schema, CSV twin, fixture and contract"
```

---

### Task 3: `export_event_series` y la publicación de la elección

**Files:**
- Modify: `mtpy/lib/publish/events.py` (nueva `export_event_series` tras `export_event_model`; `publish_event` 304-420)
- Modify: `mtpy/lib/publish/__init__.py:41-45`
- Modify: `deploy/README.md:889-915` («Elecciones pasadas»)
- Test: `tests/test_publish_unit.py` (junto a los tests de `publish_event`, 917-1003)

**Interfaces:**
- Consumes: `backtest_source(scope, source)`, `round_cols`, `PCT_DECIMALS`, `bundle.path_event`, `bundle.path_event_csv`, `writer.exists(name)`, `writer.remove(name)`; helpers de test `synthetic_event`, `write_backtest_source(root)` (devuelve la carpeta), `FakeComputer`, `event_readers`, `fake_prov`, `FileSystem`.
- Produces: `export_event_series(source: Optional[str], event_date: str) -> tuple`: `(data, frame)` con `data` para `event-series` y `frame` el gemelo (`date, name, mean, lo, hi`), o `(None, None)` sin fichero o sin filas de la elección; `publish_event(...)` escribe la parte tras `model` y antes de `meta`, borra la antigua cuando no hay serie y devuelve además `'has_series': bool`; `seconds['series']`.

- [ ] **Step 1: Test del exportador**

```python
def write_series_source(folder):
    """Escribe en `folder` un `series.csv` del backtest: 2023-07-23 a 6 días (tres días, PP y PSOE, un hueco en la banda
    de PSOE el primer día, decimales de sobra el último) y 2019-11-10 a 30 días (un día)."""
    rows = [
        ('2023-07-23', 6, '2023-07-15', 'PP', 32.2, 31.1, 33.3), ('2023-07-23', 6, '2023-07-16', 'PP', 32.3, 31.2, 33.4),
        ('2023-07-23', 6, '2023-07-17', 'PP', 32.41234, 31.31234, 33.51234),
        ('2023-07-23', 6, '2023-07-15', 'PSOE', 31.1, None, 32.2), ('2023-07-23', 6, '2023-07-16', 'PSOE', 31.0, 29.9, 32.1),
        ('2023-07-23', 6, '2023-07-17', 'PSOE', 30.98765, 29.88765, 32.08765),
        ('2019-11-10', 30, '2019-11-04', 'PP', 20.5, 19.4, 21.6),
    ]
    pd.DataFrame(rows, columns=['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi']).to_csv(
        os.path.join(folder, 'series.csv'), index=False)


def test_export_event_series_pivots_the_series_of_the_election(tmp_path):
    write_series_source(str(tmp_path))
    data, frame = publish.export_event_series(str(tmp_path), '2023-07-23')
    assert set(data) == set(bundle.SCHEMAS['event-series']) and data['horizon'] == 6
    assert data['dates'] == ['2023-07-15', '2023-07-16', '2023-07-17'] and data['parties'] == ['PP', 'PSOE']
    assert data['mean']['PP'] == [32.2, 32.3, 32.41] and data['lo']['PSOE'] == [None, 29.9, 29.89] and data['hi']['PSOE'][-1] == 32.09
    assert list(frame.columns) == ['date', 'name', 'mean', 'lo', 'hi'] and frame.shape[0] == 6 and pd.isna(frame['lo'].iloc[3])
    assert publish.export_event_series(str(tmp_path), '2015-12-20') == (None, None)
    assert publish.export_event_series(str(tmp_path / 'none'), '2023-07-23') == (None, None)
    assert publish.export_event_series(None, '2023-07-23') == (None, None)
```

(el `frame` va por partido y fecha: PP sus tres días y después PSOE, como `export_series`).

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_publish_unit.py::test_export_event_series_pivots_the_series_of_the_election -q`
Expected: FAIL con `AttributeError: ... has no attribute 'export_event_series'`

- [ ] **Step 3: Implementar `export_event_series`** en `events.py` y exportarla en `publish/__init__.py`

Lee `os.path.join(source, 'series.csv')` si `source` y el fichero existen; filtra `event_date` (`astype(str) ==`); vacío → `(None, None)`. Si quedaran varios horizontes, se queda con el mínimo. `round_cols(table, pct=['mean', 'lo', 'hi'])`; `dates` ordenadas y únicas; `parties` por primera aparición; `mean`/`lo`/`hi` por partido alineados a `dates` (`None` donde falta); `frame` = las columnas del gemelo ordenadas por partido (orden de `parties`) y fecha.

- [ ] **Step 4: Test de `publish_event` con y sin serie**

```python
def test_publish_event_writes_the_series_and_removes_it_when_the_backtest_lacks_it(tmp_path):
    ev = synthetic_event()
    folder = write_backtest_source(str(tmp_path / 'bt'))
    write_series_source(folder)
    fs = RecordingFileSystem(str(tmp_path / 'site'))
    writer = bundle.BundleWriter(fs)
    args = dict(computer=FakeComputer(), source=str(tmp_path / 'bt'), prov=fake_prov, **event_readers(ev))
    out = publish.publish_event('es', '2023-07-23', writer, **args)
    part = writer.read_json(bundle.path_event('es', '2023-07-23', 'series'))
    assert out['has_series'] is True and 'series' in out['seconds']
    assert part['schema'] == 'event-series@1' and part['data']['horizon'] == 6 and part['data']['dates'][-1] == '2023-07-17'
    assert writer.exists(bundle.path_event_csv('es', '2023-07-23', 'series'))
    written = [p for p in fs.written if p.startswith('site/v1/events/es/2023-07-23/')]
    assert written[-1].endswith('/meta.json') and written.index(next(p for p in written if p.endswith('/series.json'))) > written.index(next(p for p in written if p.endswith('/model.json')))
    os.remove(os.path.join(folder, 'series.csv'))
    out = publish.publish_event('es', '2023-07-23', writer, **args)
    assert out['has_series'] is False
    assert not writer.exists(bundle.path_event(scope='es', date='2023-07-23', part='series'))
    assert not writer.exists(bundle.path_event_csv('es', '2023-07-23', 'series'))
    assert writer.read_json(bundle.path_event('es', '2023-07-23', 'meta'))['data']['has_model'] is True
```

(`write_backtest_source` sigue sin escribir `series.csv`, así que `test_publish_event_writes_the_parts_and_meta_last` conserva sus recuentos de 6 JSON y 7 CSV; añadirle `assert not writer.exists(bundle.path_event('es', '2023-07-23', 'series'))`).

- [ ] **Step 5: Ejecutar y ver que falla**

Run: `pytest tests/test_publish_unit.py::test_publish_event_writes_the_series_and_removes_it_when_the_backtest_lacks_it -q`
Expected: FAIL con `KeyError: 'has_series'` (o la lectura de `series.json`)

- [ ] **Step 6: Implementar en `publish_event`**

Tras el borrado de los gemelos `model-*`: `series_data, series_frame = export_event_series(backtest_source(scope, source), event_date)`; con datos, `write('series', series_data, {'series': series_frame})`; sin ellos, `writer.remove(name)` de `bundle.path_event(scope, event_date, 'series')` y de `bundle.path_event_csv(scope, event_date, 'series')` cuando `writer.exists(name)`. `seconds['series'] = lap()` antes del total; el dict devuelto gana `'has_series'`. Docstring: la parte en el orden de escritura y el borrado.

- [ ] **Step 7: Ejecutar los tests de publicación y del job**

Run: `pytest tests/test_publish_unit.py tests/test_jobs_publish.py -q`
Expected: PASS

- [ ] **Step 8: Documentación y commit**

`deploy/README.md` «Elecciones pasadas»: en la lista de lo que publica, «la serie diaria del promedio del modelo del horizonte más corto del backtest (parte `series`, desde 2026-10-19: la curva «Promedio de sondeos» del detalle), solo si `backtest/results` tiene `series.csv` (fuera de git: publicar desde la máquina que corrió el backtest)»; en «Cuándo relanzar», «y una vez con `"event_date":"all"` en cada ámbito tras desplegar esta fase y correr el backtest, para que las elecciones tengan la parte `series`».

```bash
git add mtpy/lib/publish/events.py mtpy/lib/publish/__init__.py deploy/README.md tests/test_publish_unit.py
git commit -m "Publish the daily series of the model as a part of each past election"
```

---

### Task 4: Los datos de la página: `model_last`, `fixed_rows` y las tablas por horizonte

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py` (`EVENT_CSV` 22-28, `horizon_tables` 737-780; nuevas `model_last` y `fixed_rows`)
- Modify: `mtpy/lib/pages/__init__.py` (exportar `fixed_rows`, `model_last`)
- Test: `tests/test_pages.py` (`test_published_events_and_horizon_tables` 1748, `test_horizon_tables_flag_the_horizons_whose_seats_are_not_valid` 1769; tests nuevos junto a ellos)

**Interfaces:**
- Consumes: `_leaders` y `DASH` de `pages.promedios`/`pages.filters`, `fmt_num`, `fmt_date`, `OTHERS_COLOR`.
- Produces:
  - `EVENT_CSV`: tuplas `(name, table, label, needs)` con `needs` en `(None, 'model', 'series')`; entrada nueva `('series', None, 'Promedio de sondeos del modelo a {h} días', 'series')` tras `'results'`.
  - `model_last(model: dict, event_date: str) -> Optional[dict]`: `None` sin horizontes; si no `{'horizon': h, 'date': (elección − h días) ISO, 'pct': {partido: shares.model}, 'seats': {partido: seats.model_h o model}, 'seats_valid': bool}` del horizonte mínimo, sobre todas las filas del horizonte (no solo `main`).
  - `fixed_rows(last: Optional[dict], results_rows: list, parties: list, colors: dict, event_date: str) -> list`: `[]` sin `last`; si no dos filas `{'label', 'small', 'title', 'cells'}` con las celdas de `polls_table` (`{'value', 'leader', 'color'}`).
  - `horizon_tables(model: dict) -> list`: por horizonte ascendente `{'horizon', 'seats_valid', 'rows': [{'name', 'official', 'model', 'error', 'official_seats', 'model_seats'}] (partidos `main` de `shares`, los escaños por nombre o `None`), 'blocks': [{'block', 'official_seats', 'model_seats', 'p_majority', 'majority'}]}` con `model_seats`/`p_majority` de la ejecución a horizonte cuando la fila trae `_h` no nulo.

- [ ] **Step 1: Tests de los helpers** (en `tests/test_pages.py`, tras `test_horizon_tables_flag_the_horizons_whose_seats_are_not_valid`)

```python
def test_model_last_takes_the_shortest_horizon_and_the_horizon_run_seats():
    model = {'horizons': [30, 6],
             'shares': [{'horizon': h, 'party': p, 'main': p != 'Cs', 'official': 33.0, 'model': m}
                        for h in (6, 30) for p, m in (('PP', 32.4 if h == 6 else 31.0), ('Cs', 1.2))],
             'seats': [{'horizon': h, 'party': 'PP', 'main': True, 'official': 137, 'model': 140, 'model_h': 142 if h == 6 else None,
                        'seats_valid': True} for h in (6, 30)],
             'blocks': []}
    assert pages.model_last(model, '2023-07-23') == {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.4, 'Cs': 1.2},
                                                     'seats': {'PP': 142}, 'seats_valid': True}
    nowcast = {**model, 'horizons': [30]}
    assert pages.model_last(nowcast, '2023-07-23')['seats'] == {'PP': 140} and pages.model_last(nowcast, '2023-07-23')['horizon'] == 30
    assert pages.model_last({'horizons': [], 'shares': [], 'seats': [], 'blocks': []}, '2023-07-23') is None


def test_fixed_rows_of_the_polls_table_footer():
    last = {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.4, 'PSOE': 31.0}, 'seats': {'PP': 140}, 'seats_valid': True}
    results = [{'party': 'PP', 'pct': 33.06, 'seats': 160}, {'party': 'PSOE', 'pct': 31.44, 'seats': 150}]
    colors = {'PP': '#1d84ce', 'PSOE': '#e4001c'}
    rows = pages.fixed_rows(last, results, ['PP', 'PSOE', 'VOX'], colors, '2023-07-23')
    assert [(row['label'], row['small']) for row in rows] == [('Promedio de sondeos', 'modelo · a 6 días'), ('Resultado', '23 jul 2023')]
    assert rows[0]['title'] == 'Congelado el 17 jul 2023: solo los sondeos publicados hasta entonces' and rows[1]['title'] is None
    assert [cell['value'] for cell in rows[0]['cells']] == ['32,4', '31,0', '–'] and [cell['leader'] for cell in rows[0]['cells']] == [True, False, False]
    assert [cell['value'] for cell in rows[1]['cells']] == ['33,1', '31,4', '–'] and rows[1]['cells'][2]['color'] == pages.OTHERS_COLOR
    assert pages.fixed_rows(None, results, ['PP'], colors, '2023-07-23') == []


def test_horizon_tables_merge_vote_and_seats_and_prefer_the_horizon_run():
    model = {'horizons': [6],
             'shares': [{'horizon': 6, 'party': 'PP', 'main': True, 'official': 33.05, 'model': 32.41},
                        {'horizon': 6, 'party': 'Cs', 'main': False, 'official': 1.0, 'model': 1.5}],
             'seats': [{'horizon': 6, 'party': 'PP', 'main': True, 'official': 137, 'model': 140, 'model_h': 142, 'seats_valid': True}],
             'blocks': [{'horizon': 6, 'block': 'Derecha', 'official_seats': 169, 'model_seats': 171, 'model_seats_h': 173,
                         'p_majority': 0.4, 'p_majority_h': 0.45, 'majority': False, 'seats_valid': True}]}
    [table] = pages.horizon_tables(model)
    assert table['rows'] == [{'name': 'PP', 'official': 33.05, 'model': 32.41, 'error': round(32.41 - 33.05, 2), 'official_seats': 137, 'model_seats': 142}]
    assert table['blocks'] == [{'block': 'Derecha', 'official_seats': 169, 'model_seats': 173, 'p_majority': 0.45, 'majority': False}]
    model['seats'][0]['model_h'] = None
    del model['blocks'][0]['model_seats_h'], model['blocks'][0]['p_majority_h']
    [table] = pages.horizon_tables(model)
    assert table['rows'][0]['model_seats'] == 140 and table['blocks'][0]['model_seats'] == 171 and table['blocks'][0]['p_majority'] == 0.4
```

Y en `test_published_events_and_horizon_tables` sustituir las tres líneas sobre `tables` por:

```python
    assert [table['horizon'] for table in tables] == [6, 30]
    pp = tables[0]['rows'][0]
    assert pp == {'name': 'PP', 'official': 33.05, 'model': 32.41, 'error': round(32.41 - 33.05, 2), 'official_seats': 137, 'model_seats': 140}
    assert set(tables[0]['blocks'][0]) == {'block', 'official_seats', 'model_seats', 'p_majority', 'majority'}
```

y en `test_horizon_tables_flag_the_horizons_whose_seats_are_not_valid` la última línea pasa a `tables[1]['rows'][0]['error'] == -1.0`.

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_pages.py -k "model_last or fixed_rows or horizon_tables" -q`
Expected: FAIL (`model_last`/`fixed_rows` no existen; `horizon_tables` devuelve `shares`/`seats`)

- [ ] **Step 3: Implementar los tres helpers y `EVENT_CSV`** en `elecciones.py`; exportarlos en `pages/__init__.py`

`fixed_rows`: celdas con `fmt_num(value, 1)`, `_leaders` y `colors.get(name, OTHERS_COLOR)`, exactamente como la media de `polls_table`; `small` de la primera fila `'modelo · a {h} días'`, `title` `'Congelado el {fmt_date(last["date"])}: solo los sondeos publicados hasta entonces'`; la segunda `fmt_date(event_date)` y `title` `None`. El voto de la primera es `last['pct']`, el de la segunda el `pct` de `results_rows` por partido. `horizon_tables`: en cada fila de escaños y de bloques, `value = row['x_h'] if row.get('x_h') is not None else row.get('x')`. Los enlaces de `eleccion_context` se adaptan en la tarea 5 (hasta entonces, `needs == 'model'` equivale al antiguo `True`: cambiar la comprensión a `if needs is None or (needs == 'model' and has_model) or (needs == 'series' and False)` para que los tests sigan en verde; la tarea 5 la deja definitiva).

- [ ] **Step 4: Ejecutar las páginas**

Run: `pytest tests/test_pages.py -q`
Expected: PASS salvo las aserciones de la plantilla del modelo que la tarea 5 rehace (`test_eleccion_detail_replaces_the_seats_of_an_invalid_horizon_by_a_notice` y la de orden de tarjetas pueden fallar ahora porque la plantilla sigue leyendo `table.shares`): si fallan, adaptar la plantilla de forma mínima en esta tarea (`table.rows` en lugar de `table.shares`/`table.seats`) sin cambiar ids ni textos, y volver a correr hasta PASS completo.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/elecciones.py mtpy/lib/pages/__init__.py web/templates/eleccion.html tests/test_pages.py
git commit -m "Compute what the model said at the shortest horizon and the fixed rows of the polls table"
```

---

### Task 5: El contexto del detalle y las plantillas

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py` (`eleccion_context` 782-912, docstring incluido; `chart_parties`)
- Modify: `mtpy/lib/pages/promedios.py` (`promedios_context`: `context['fixed_rows'] = []` junto a `context['table']`, línea 652)
- Modify: `web/templates/eleccion.html` (tarjeta nueva tras `</div>` de `.hemi-grid`; tarjeta `model` rehecha y movida tras la nueva; «Cómo leerlo» de `polls`; `downloads`)
- Modify: `web/templates/_polls_table.html` (pie)
- Test: `tests/test_pages.py` (`test_eleccion_detail_renders_results_last_polls_bullseye_and_model` 1534, `test_eleccion_without_model_shows_a_notice` 1734, `test_eleccion_detail_replaces_the_seats_of_an_invalid_horizon_by_a_notice` 1781; tests nuevos junto a `test_eleccion_detail_has_the_polls_table_of_the_cycle` 1625)

**Interfaces:**
- Consumes: tarea 4; `round_series`, `ink_color`, `block_color`, `party_blocks`, `chart_parties`; `write_event(fs, ..., series=…)` de la tarea 2 (`events_site` publica 2023 con modelo y serie y 2019 sin ninguno).
- Produces en el contexto: `has_series`, `series_horizon` (el de la serie, o el de `model_last`), `series_blocks` (nombres de bloque de `initial.catalog.bmaps.blocks`, vacío sin bloques), `model_last`, `fixed_rows`, `horizon_tables` (tarea 4), `csv_links` con la serie solo si `has_series`; en `initial`: `series` (redondeada a 1 decimal), `result` `{'date', 'values': {partido: pct}}` (solo partidos con `pct`), `model_last`, `inks`, `results.rows` con `pct`, `catalog.bmaps` `{'main', 'blocks'}`.

- [ ] **Step 1: Tests de la página**

Actualizar `test_eleccion_detail_renders_results_last_polls_bullseye_and_model`:
- `page.count('/api/v1/events/es/2023-07-23/') == 8 + 2` (la descarga de la serie);
- `set(data)` gana `'series', 'result', 'model_last', 'inks'`;
- `data['results']['rows'][0] == {'name': 'PP', 'seats': 160, 'pct': 33.06}` (y las otras filas con su `pct`);
- la cadena de orden pasa a `id="event" < hemi-grid < hemicycle-card < results < id="series" < id="model" < class="split" < last-errors < last-split < bullseye < last < glosario < poll-histogram < polls < downloads`.

Actualizar `test_eleccion_without_model_shows_a_notice`: además `'id="series"' not in page and 'id="model-horizon"' not in page and 'tr class="fixed"' not in page`.

Actualizar `test_eleccion_detail_replaces_the_seats_of_an_invalid_horizon_by_a_notice`: ids `model-table-6`, `model-table-30`, `model-blocks-6` presentes, `model-blocks-30` ausente, el aviso una vez, y `'id="model-shares-6"' not in page`.

Tests nuevos:

```python
def test_eleccion_detail_draws_the_average_with_the_result_and_what_the_model_said(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body).replace('\n', ' ')
    assert status == 200 and 'congelado a 6 días de la elección' in page
    assert 'id="series-chart"' in page and 'id="series-window"' in page and 'id="series-legend"' in page and 'name="view"' in page
    assert 'href="/api/v1/events/es/2023-07-23/series?format=csv"' in page and 'Promedio de sondeos del modelo a 6 días' in page
    assert 'no está publicado todavía: hace falta' not in page
    # la tarjeta del modelo: selector de horizonte (6 marcado), una tabla por horizonte (la de 30 oculta) y la de bloques
    assert '<fieldset id="model-horizon" class="seg" hidden>' in page and '<input type="radio" name="horizon" value="6" checked>' in page
    assert 'data-horizon="6">' in page and 'data-horizon="30" hidden>' in page
    table = re.search(r'<table id="model-table-6">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col">(.*?)</th>', table) == ['Partido', 'Promedio (%)', 'Resultado (%)', 'Error', 'Escaños modelo', 'Escaños reales']
    assert table.index('<td>PP</td>') < table.index('<td>32,4</td>') < table.index('<td><strong>140</strong></td>')
    assert 'class="cell-shade"' in table and '0,6</td>' in table  # el error, 32,41 − 33,05, en su celda sombreada
    assert 'id="model-blocks-6"' in page and 'id="model-shares-6"' not in page and 'Lo que dijo el modelo' in page
    # las dos filas fijas del pie, tras la media simple
    foot = page[page.index('<tfoot>'):page.index('</tfoot>')]
    assert foot.index('Media simple') < foot.index('<b>Promedio de sondeos</b><small>modelo · a 6 días</small>') < foot.index('<b>Resultado</b><small>23 jul 2023</small>')
    assert foot.count('<tr class="fixed"') == 2 and 'Congelado el 17 jul 2023' in foot and '>32,4</td>' in foot  # el líder, sombreado
    data = initial(body)
    assert data['series']['horizon'] == 6 and data['series']['dates'][-1] == '2023-07-17' and data['series']['mean']['PP'][-1] == 32.4
    assert data['result']['date'] == '2023-07-23' and data['result']['values']['PP'] == 33.06
    assert data['model_last'] == {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.41, 'PSOE': 30.99, 'VOX': 13.01},
                                  'seats': {'PP': 140, 'PSOE': 118, 'VOX': 30}, 'seats_valid': True}
    assert data['catalog']['bmaps'] == {'main': ['PP', 'PSOE', 'VOX'], 'blocks': {'Derecha': ['PP', 'VOX', 'Cs'], 'Izquierda': ['PSOE']}}
    assert set(data['inks']) >= {'PP', 'PSOE', 'VOX', 'Derecha', 'Izquierda'}


def test_eleccion_detail_with_model_but_without_series_shows_a_notice(events_site, monkeypatch):
    real = pages.elecciones.event_part
    monkeypatch.setattr(pages.elecciones, 'event_part', lambda scope, date, name: None if name == 'series' else real(scope, date, name))
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and 'El promedio de este ciclo no está publicado todavía' in page
    assert 'id="series"' in page and 'id="series-chart"' not in page and page.count('/api/v1/events/es/2023-07-23/') == 7 + 2
    data = initial(body)
    assert 'series' not in data and data['model_last']['horizon'] == 6 and 'tr class="fixed"' in page


def test_promedios_polls_table_has_no_fixed_rows(api):
    page = html(call(api, '/es')[2])
    assert 'Media simple' in page and 'tr class="fixed"' not in page
```

(en `test_eleccion_detail_draws_...`, el `pct` del resultado de `Cs` no se asume: el fixture `event-results` lo trae con `seats` 0; `result.values` lleva a `Cs` solo si su `pct` no es nulo).

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_pages.py -k "eleccion or promedios_polls_table_has_no" -q`
Expected: FAIL (sin tarjeta `series`, sin `model-horizon`, sin filas fijas, isla sin `series`)

- [ ] **Step 3: Implementar el contexto** en `eleccion_context`

- `series = event_part(scope, date, 'series')`; `last = model_last(model, date) if has_model else None`; `series_horizon = (series or {}).get('horizon') or (last or {}).get('horizon')`.
- Bloques de la curva: `names = chart_parties(catalog, table_parties, [resultados], (series or {}).get('parties') or [])`; `blocks_map` = inversa de `party_blocks(names, params, meta.get('blocks'), catalog)` (`{bloque: [partidos en el orden de names]}`, sin los `None`); `context['series_blocks'] = list(blocks_map)`.
- `context['fixed_rows'] = fixed_rows(last, results_rows, table_parties, colors, date)`; `context['has_series']`, `context['series_horizon']`, `context['model_last'] = last`.
- Enlaces: `[{'label': label.format(h=series_horizon), 'href': ...} for name, table, label, needs in EVENT_CSV if needs is None or (needs == 'model' and has_model) or (needs == 'series' and has_series)]`.
- `initial`: `results.rows` con `'pct': row.get('pct')`; `'result': {'date': date, 'values': {row['party']: row['pct'] for row in results_rows if row.get('pct') is not None}}`; `'model_last': last`; `'inks': {name: ink_color(catalog[name]['color']) for name in names} | {block: ink_color(block_color(block, context['meta'], catalog)) for block in blocks_map}`; `'catalog': {'parties': [...names...], 'bmaps': {'main': list(meta.get('main') or []), 'blocks': blocks_map}}`; con serie, `'series': round_series(series)`. Docstring de `eleccion_context` al día.
- `promedios_context`: `context['fixed_rows'] = []`.

- [ ] **Step 4: Implementar las plantillas**

`_polls_table.html`, dentro de `<tfoot>` tras la fila de la media:

```jinja
              {% for row in fixed_rows %}
              <tr class="fixed">
                <td class="f l"{% if row.title %} title="{{ row.title }}"{% endif %}><b>{{ row.label }}</b><small>{{ row.small }}</small></td>
                <td></td>
                <td></td>
                <td></td>
                {% for cell in row.cells %}
                {% if cell.leader %}<td class="p mx" style="background: {{ cell.color }}24">{{ cell.value }}</td>
                {% elif cell.value == '–' %}<td class="p na">—</td>
                {% else %}<td class="p">{{ cell.value }}</td>{% endif %}
                {% endfor %}
              </tr>
              {% endfor %}
```

y el comentario de cabecera del parcial cita `fixed_rows`.

`eleccion.html`:
- Tras `</div>` de `.hemi-grid`, `{% if has_model %}` la tarjeta `series` (`m.card_start('series', 'Promedio de sondeos', 'La intención de voto de cada partido a lo largo del ciclo según el promedio ponderado del modelo, congelado a ' ~ series_horizon|int ~ ' días de la elección, y el resultado oficial al final.')`). Con `has_series`: los controles de Promedios (`<div class="series-controls">` con `#series-view` solo `{% if series_blocks %}`, `#series-window` con las cuatro ventanas, ambos `hidden`), `<div class="legend" id="series-legend" role="group" aria-label="Partidos y bloques del gráfico" hidden></div>`, `<div class="chart chart-tall" id="series-chart" role="img" aria-label="Promedio de sondeos del ciclo con su banda, los sondeos y el resultado oficial"></div>` y el `noscript`; sin ella, `<p class="notice">El promedio de este ciclo no está publicado todavía: hace falta volver a publicar la elección con un backtest que guarde la serie.</p>`. «Cómo leerlo», tres párrafos: (1) «Cada punto es un sondeo publicado (al pasar por encima, o al tocarlo, se ve entero); la línea es el promedio ponderado del modelo y la banda, su intervalo de confianza. El punto marcado del final, en la fecha de la elección, es el resultado oficial de cada partido.» (2) «El hueco entre el final de la línea y el resultado es la veda: la línea se congeló a {h} días de votar, el último día con sondeos. Pulsa un partido de la leyenda para ocultarlo o mostrarlo; en la vista Bloques, cada bloque suma sus partidos.» (3) «La banda dice dónde estaba el consenso de las encuestadoras, no la incertidumbre del resultado; y la línea no recoge lo que cambió en la última semana.»
- La tarjeta `model`, justo después, sustituye a «El modelo a cada horizonte»: título «Lo que dijo el modelo», texto «El voto y los escaños que el modelo habría publicado a cada distancia de la elección, frente al resultado; de entrada, a {primer horizonte} días, el último día con sondeos.»; `<fieldset id="model-horizon" class="seg" hidden><legend>Horizonte</legend>` con un `<label><input type="radio" name="horizon" value="{{ h }}"{% if loop.first %} checked{% endif %}> A {{ h }} días</label>` por tabla; por tabla `<div class="horizon" data-horizon="{{ h }}"{% if not loop.first %} hidden{% endif %}>` con `<table id="model-table-{{ h }}">` (caption «Voto y escaños del modelo a {h} días de la elección, frente al resultado.»; cabeceras `Partido | Promedio (%) | Resultado (%) | Error | Escaños modelo | Escaños reales`; filas `<td>{{ row.name }}</td><td>{{ row.model|num(1) }}</td><td>{{ row.official|num(1) }}</td><td class="cell-shade" style="background-color: {{ row.error|shade_around(0, 8, '#d62728', '#2ca02c') }}">{{ row.error|num(1) }}</td>` y las dos de escaños `<td><strong>{{ row.model_seats|int }}</strong></td><td>{{ row.official_seats|int }}</td>` cuando `table.seats_valid`, si no `<td class="na">—</td>` dos veces), después la tabla `model-blocks-{h}` de hoy (misma estructura) si `seats_valid`, si no el aviso de hoy «Escaños no proyectables en esta elección…». Sin `horizon_tables`, el aviso «Sin backtest para esta elección.» como hoy. «Cómo leerlo»: (1) «Cada fila es un partido principal: el voto del promedio del modelo a esa distancia de la elección y el oficial, su diferencia (promedio menos resultado, en puntos: rojo si el modelo se quedó corto y verde si se pasó, más intenso cuanto mayor, hasta 8) y los escaños que proyectaba frente a los reales. Debajo, cada bloque con sus escaños y la probabilidad de mayoría absoluta que le daba el modelo.» (2) «El selector cambia la distancia: a cada horizonte el modelo se calcula solo con los sondeos publicados hasta entonces. Los escaños y la probabilidad son los de la ejecución que proyecta la opinión hasta el día de la elección.» (3) «Una probabilidad no se acierta ni se falla en una sola elección: una baja que se cumple no es por sí sola un error.»
- Tarjeta `polls`, «Cómo leerlo»: `{% if fixed_rows %}` un párrafo más al final: «Las dos últimas filas del pie no son sondeos: «Promedio de sondeos» es lo que el modelo decía a {h} días de la elección, el último día con sondeos, y «Resultado», el oficial; los filtros no las tocan. En la vista Escaños, la primera pasa a los escaños que proyectaba el modelo.»
- Tarjeta `downloads`: «el modelo a cada horizonte» → «lo que dijo el modelo{% if has_series %} y el promedio del ciclo{% endif %}».
- `{% block description %}`: sin cambios.

- [ ] **Step 5: Ejecutar las páginas**

Run: `pytest tests/test_pages.py tests/test_webapi_unit.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/pages/elecciones.py mtpy/lib/pages/promedios.py web/templates/eleccion.html web/templates/_polls_table.html tests/test_pages.py
git commit -m "Show the poll average of the cycle and what the model said in the detail of a past election"
```

---

### Task 6: El resultado oficial en el gráfico de series y por bloques

**Files:**
- Modify: `web/dist/js/charts/series.js` (`axisDatesOf` 107-111, `setWindow` 120-128, `seriesOption` 146-296)
- Modify: `web/dist/js/polls-tooltip.js` (nueva `resultTooltipLine`, tras `meanTooltipLine`)
- Modify: `web/dist/js/blocks.js` (nueva `blockResult`, tras `blockProjection`)
- Test: `tests/test_js_charts.py` (tras `test_series_hidden_legend_keeps_the_component_and_the_theme`), `tests/test_js_blocks.py`, `tests/test_js_polls.py` (tras `test_mean_tooltip_line_shows_the_average_its_interval_and_the_date`)

**Interfaces:**
- Produces: `axisDatesOf(series, projection, result = null)` añade `result.date` al final cuando es posterior a la última fecha; `setWindow(chart, choice, {series, projection, anchor = null, result = null})`; `seriesOption({..., result = null, asOfLabel = 'Estimación'})` dibuja por partido con cifra en `result.values` una serie `scatter` (`symbol: 'diamond'`, `symbolSize: 12`, `z: 5`, `itemStyle: {borderColor: '#fff', borderWidth: 1.5}`, un punto `[dayTime(result.date), valor]`, rol `result` para el tooltip) y la marca «Elección» en `result.date`; la marca de `asOf` lleva `asOfLabel`; `resultTooltipLine(name, value, date, color)` → HTML de dos líneas: «Resultado oficial · {fmtDate(date)}» y «{name} {fmtPct(value, 1)}», con el marcador de color; `blockResult(result, blocks)` → `{date, values: {bloque: suma de sus partidos presentes}}` (un bloque sin ningún partido en `values` no tiene clave).

- [ ] **Step 1: Tests JS**

En `tests/test_js_charts.py`:

```python
RESULT_ARGS = SERIES_ARGS.replace("legend: %s", "legend: 'hidden', asOfLabel: 'A 6 días', result: {date: '2026-10-08', values: {PP: 31.5}}")


def test_series_result_draws_a_diamond_at_the_election_and_extends_the_axis():
    option = run_js('s.seriesOption(%s)' % RESULT_ARGS)
    election = run_js('s.dayTime("2026-10-08")')
    diamonds = [s for s in option['series'] if s['type'] == 'scatter' and s.get('symbol') == 'diamond']
    assert [(s['name'], s['data'], s['symbolSize'], s['z']) for s in diamonds] == [('PP', [[election, 31.5]], 12, 5)]
    assert diamonds[0]['itemStyle'] == {'borderColor': '#fff', 'borderWidth': 1.5}
    marks = option['series'][-1]['markLine']['data']
    assert [m['label']['formatter'] for m in marks] == ['A 6 días', 'Elección'] and marks[1]['xAxis'] == election
    assert option['dataZoom'][0]['endValue'] == election
    plain = run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))
    assert not [s for s in plain['series'] if s.get('symbol') == 'diamond']
    assert [m['label']['formatter'] for m in plain['series'][-1]['markLine']['data']] == ['Estimación']
    assert run_js('s.axisDatesOf({dates: ["2026-10-01", "2026-10-02"]}, null, {date: "2026-10-08", values: {}})') == ['2026-10-01', '2026-10-02', '2026-10-08']
    assert run_js('s.axisDatesOf({dates: ["2026-10-01", "2026-10-02"]}, null, {date: "2026-10-02", values: {}})') == ['2026-10-01', '2026-10-02']


def test_series_window_of_three_months_keeps_the_result_in_view():
    option = run_js('s.seriesOption(%s)' % RESULT_ARGS.replace("window: 'all'", "window: '3m', anchor: '2026-10-08'"))
    assert option['dataZoom'][0]['endValue'] == run_js('s.dayTime("2026-10-08")')
```

En `tests/test_js_blocks.py`:

```python
def test_block_result_sums_the_parties_of_each_block():
    assert run_js('b.blockResult({date: "2026-01-10", values: {PP: 30, VOX: 10, PSOE: 28}}, blocks)') == {'date': '2026-01-10', 'values': {'Derecha': 40, 'Izquierda': 28}}
    assert run_js('b.blockResult({date: "2026-01-10", values: {PSOE: 28}}, blocks)') == {'date': '2026-01-10', 'values': {'Izquierda': 28}}
```

En `tests/test_js_polls.py`:

```python
def test_result_tooltip_line_names_the_official_result_and_the_date():
    line = run_js('s.resultTooltipLine("PP", 33.06, "2023-07-23", "#1d84ce")')
    assert 'Resultado oficial' in line and '23 jul 2023' in line and '#1d84ce' in line and line.index('<br>') < line.index('PP')
    assert '33,1' in line
```

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_js_charts.py tests/test_js_blocks.py tests/test_js_polls.py -k "result or three_months" -q`
Expected: FAIL (sin rombos, sin `blockResult`, sin `resultTooltipLine`)

- [ ] **Step 3: Implementar** en `series.js`, `polls-tooltip.js` y `blocks.js`

En `seriesOption`: las series del resultado van tras `projected` y antes de `markSeries`; el tooltip devuelve `resultTooltipLine(name, result.values[name], result.date, color(name))` para el rol `result`; `axisDates = axisDatesOf(series, projection, result)`; la marca «Elección» se añade si `result` (además del caso con proyección de hoy). JSDoc de `seriesOption` con `result` y `asOfLabel`. `blockResult` sigue el estilo de `blockPolls` (`present`). `polls-tooltip.js` importa `fmtDate` y `fmtPct` de `format.js` si no los tiene.

- [ ] **Step 4: Ejecutar los tests JS**

Run: `pytest tests/test_js_charts.py tests/test_js_blocks.py tests/test_js_polls.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/series.js web/dist/js/polls-tooltip.js web/dist/js/blocks.js tests/test_js_charts.py tests/test_js_blocks.py tests/test_js_polls.py
git commit -m "Mark the official result on the series chart and sum it by block"
```

---

### Task 7: El cableado compartido de la curva y los selectores del detalle

**Files:**
- Create: `web/dist/js/pages/series-wire.js`
- Modify: `web/dist/js/pages/promedios.js` (quitar `seriesViews`, `chosenNames`, `fillLegend`, `wireAverage` 101-250; importarlos; `wireSeries` 318-360 y `wireHouse`)
- Modify: `web/dist/js/pages/elecciones.js` (nuevas `wireAverageCard` y `wireModelHorizons`; `paint`)
- Test: `tests/test_js_charts.py` (arnés `run_js`: añadir el módulo como `sw`)

**Interfaces:**
- Consumes: tarea 6 (`result` en `seriesOption`/`setWindow`, `blockResult`); `parseSeriesHash`, `seriesHash`, `WINDOW_DAYS`, `renderSeries`, `markLegend`, `renderLegend`, `markSegments`, `tryDraw`.
- Produces: `series-wire.js` exporta `WINDOWS`, `seriesViews(initial, catalog)` (como hoy, y cada vista gana `result`: `initial.result || null` en `partidos`, `blockResult(initial.result, blocks)` en `bloques`, `null` sin resultado), `chosenNames(current, legendSelected)`, `fillLegend(legendEl, current, selected, chart, inks)` y `wireAverage(views, state, render, {anchor, inks = {}})` (la de Promedios, que ya no lee `initial`: el ancla llega por parámetro y `setWindow` recibe `result: current.result`). Promedios: `wireSeries` llama `wireAverage(views, state, render, {anchor, inks: initial.inks || {}})` y `wireHouse` importa `chosenNames`/`fillLegend`. Detalle: `wireAverageCard(initial, catalog)` y `wireModelHorizons()`.

- [ ] **Step 1: Test de paridad de las vistas** (en `tests/test_js_charts.py`; el arnés `run_js` importa además `web/dist/js/pages/series-wire.js` como `sw` y `web/dist/js/catalog.js` como `c`)

```python
def test_series_views_carry_the_result_of_each_view():
    initial = {'series': {'dates': ['2026-10-01'], 'parties': ['PP', 'VOX', 'PSOE'], 'mean': {'PP': [30], 'VOX': [10], 'PSOE': [28]},
                          'lo': {'PP': [29], 'VOX': [9], 'PSOE': [27]}, 'hi': {'PP': [31], 'VOX': [11], 'PSOE': [29]}},
               'polls': {'parties': ['PP', 'VOX', 'PSOE'], 'polls': []}, 'projection': None,
               'result': {'date': '2026-10-08', 'values': {'PP': 31, 'PSOE': 27.5}},
               'catalog': {'parties': [{'name': n, 'color': '#000000', 'fullname': n, 'block': None} for n in ('PP', 'VOX', 'PSOE')],
                           'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}}}}
    views = run_js('sw.seriesViews(%s, new c.Catalog(%s.catalog))' % (json.dumps(initial), json.dumps(initial)))
    assert views['partidos']['result'] == initial['result'] and views['partidos']['defaults'] == ['PP', 'PSOE']
    assert views['bloques']['result'] == {'date': '2026-10-08', 'values': {'Derecha': 31, 'Izquierda': 27.5}}
    assert views['bloques']['parties'] == ['Derecha', 'Izquierda'] and views['bloques']['projection'] is None
    assert run_js('sw.seriesViews(%s, new c.Catalog({}))' % json.dumps({**initial, 'result': None}))['partidos']['result'] is None
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_js_charts.py::test_series_views_carry_the_result_of_each_view -q`
Expected: FAIL (el módulo no existe)

- [ ] **Step 3: Crear `series-wire.js` y adaptar Promedios**

Mover las cuatro funciones y `WINDOWS`/`BLOCKS_VIEW` tal cual, con sus JSDoc, cambiando solo lo del bloque Interfaces. Promedios conserva su comportamiento y su hash. Comprobar que `import * as sw` carga en node con `globalThis.window = {echarts}` (ningún acceso al DOM al cargar el módulo).

- [ ] **Step 4: Ejecutar los tests JS**

Run: `pytest tests/test_js_charts.py tests/test_js_blocks.py tests/test_js_polls.py -q`
Expected: PASS

- [ ] **Step 5: Cablear el detalle** en `elecciones.js`

```js
function wireAverageCard(initial, catalog) {
  if (!initial.series || !document.getElementById('series-chart')) {
    return;
  }
  const views = seriesViews(initial, catalog);
  const when = initial.result ? initial.result.date : initial.meta.event_date;
  const state = parseSeriesHash(window.location.hash, {
    parties: views.partidos.parties, blocks: views.bloques ? views.bloques.parties : [], windows: WINDOWS,
  });
  const render = (el, pollsOfChart, chartState) => {
    const current = views[chartState.view];
    const dates = current.series.dates;
    return renderSeries(el, {
      series: current.series, polls: pollsOfChart, projection: null, result: current.result,
      parties: current.parties, selected: chartState.selected ?? current.defaults, colors: current.colors,
      asOf: dates[dates.length - 1], asOfLabel: `A ${initial.series.horizon} días`, when,
      window: chartState.window, anchor: when, legend: 'hidden',
    });
  };
  wireAverage(views, state, render, {anchor: when, inks: initial.inks || {}});
}
```

`wireModelHorizons()`: con `#model-horizon`, lo muestra y, al cambiar un `input[name=horizon]`, oculta cada `#model .horizon[data-horizon]` cuyo `data-horizon` no sea el valor y llama `markSegments(fieldset)`. En `paint`, tras `wireHemicycle`: `tryDraw(() => wireAverageCard(initial, catalog), 'the poll average')` y `wireModelHorizons()`. El comentario de cabecera del módulo cita la curva y el selector.

- [ ] **Step 6: Comprobación en el navegador** (sin test automático: el cableado es DOM)

Arrancar `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8011` contra el paquete local (`files/`) con una elección que tenga `series.json`; si el paquete local no la tiene, escribirla con `write_event` de `tests/test_webapi_unit.py` sobre un `FileSystem` apuntado a `files/` temporal (o publicar `event` en local tras un backtest). Abrir `/elecciones/es/generales/2023-07`: la curva con puntos y rombos, el hueco de la veda, la marca «A 6 días», la ventana «3 meses» con los rombos a la vista, la leyenda de píldoras, la vista Bloques, el selector de horizonte y `#`-hash como en Promedios; en `/es`, Promedios igual que antes. Consola sin errores. Anotar el resultado en el informe de la tarea.

- [ ] **Step 7: Commit**

```bash
git add web/dist/js/pages/series-wire.js web/dist/js/pages/promedios.js web/dist/js/pages/elecciones.js tests/test_js_charts.py
git commit -m "Share the wiring of the poll average and draw it with the result in the detail of an election"
```

---

### Task 8: Las filas fijas del pie en el navegador

**Files:**
- Modify: `web/dist/js/polls-table.js` (nueva `fixedRows`, tras `meanOf`)
- Modify: `web/dist/js/pages/polls-dom.js:165-181` (`renderFoot`)
- Modify: `web/dist/js/pages/polls-wire.js` (`wirePollsTable` 50-143)
- Modify: `web/dist/js/pages/elecciones.js` (la llamada a `wirePollsTable` en `paint`)
- Test: `tests/test_js_polls.py` (tras `test_seats_view_uses_the_pollsters_projections`)

**Interfaces:**
- Consumes: `valueCells`, `fmtInt`, `fmtDate`; `initial.model_last`, `initial.result`, `initial.results.rows` (tarea 5).
- Produces: `fixedRows(modelLast, result, seats, parties, mode = 'pct')` → `[]` sin `modelLast`; si no dos filas `{label, small, title, values, texts}`: `values` por partido de `parties` (`null` si falta), `texts` nulo en `pct` y `{name: fmtInt(valor)}` en `esc`; en `esc` la primera fila usa `modelLast.seats` solo si `seats_valid` (si no, todo `null`); `renderFoot(tfoot, count, means, parties, catalog, mode = 'pct', fixed = [])` reconstruye la fila de la media y añade una `tr.fixed` por fila (primera celda `td.f.l` con `b` y `small` y `title` si lo hay, tres celdas vacías, `valueCells(row.values, parties, catalog, row.texts)`); `wirePollsTable(polls, catalog, {fixed = null} = {})` con `fixed = {modelLast, result, seats}` y `renderFoot(..., fixed ? fixedRows(fixed.modelLast, fixed.result, fixed.seats, parties, state.mode) : [])`.

- [ ] **Step 1: Test de `fixedRows`**

```python
def test_fixed_rows_of_the_footer_follow_the_mode_and_the_validity_of_the_seats():
    last = {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.4, 'PSOE': 31.0}, 'seats': {'PP': 140, 'PSOE': 118}, 'seats_valid': True}
    result = {'date': '2023-07-23', 'values': {'PP': 33.1, 'PSOE': 31.4}}
    seats = {'PP': 137, 'PSOE': 121}
    args = '{}, {}, {}, ["PP", "PSOE", "VOX"]'.format(json.dumps(last), json.dumps(result), json.dumps(seats))
    rows = run_js('p.fixedRows({})'.format(args))
    assert rows == [
        {'label': 'Promedio de sondeos', 'small': 'modelo · a 6 días', 'title': 'Congelado el 17 jul 2023: solo los sondeos publicados hasta entonces',
         'values': {'PP': 32.4, 'PSOE': 31.0, 'VOX': None}, 'texts': None},
        {'label': 'Resultado', 'small': '23 jul 2023', 'title': None, 'values': {'PP': 33.1, 'PSOE': 31.4, 'VOX': None}, 'texts': None}]
    esc = run_js('p.fixedRows({}, "esc")'.format(args))
    assert esc[0]['values'] == {'PP': 140, 'PSOE': 118, 'VOX': None} and {k: v for k, v in esc[0]['texts'].items() if k != 'VOX'} == {'PP': '140', 'PSOE': '118'}
    assert esc[1]['values'] == {'PP': 137, 'PSOE': 121, 'VOX': None}
    invalid = run_js('p.fixedRows({}, "esc")'.format(args.replace('"seats_valid": true', '"seats_valid": false')))
    assert invalid[0]['values'] == {'PP': None, 'PSOE': None, 'VOX': None}
    assert run_js('p.fixedRows(null, {}, {}, ["PP"])'.format(json.dumps(result), json.dumps(seats))) == []
```

(`fmtInt(null)` devuelve el guion de `format.js`; `valueCells` ya pinta «—» para un valor que no es número, así que el texto de VOX no se asevera).

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_js_polls.py::test_fixed_rows_of_the_footer_follow_the_mode_and_the_validity_of_the_seats -q`
Expected: FAIL (`p.fixedRows is not a function`)

- [ ] **Step 3: Implementar** `fixedRows`, `renderFoot` con `fixed`, `wirePollsTable` con `{fixed}` y la llamada del detalle

En `elecciones.js`: `wirePollsTable(initial.polls.polls, catalog, {fixed: initial.model_last ? {modelLast: initial.model_last, result: initial.result, seats: Object.fromEntries((initial.results.rows || []).map((row) => [row.name, row.seats]))} : null})`. Promedios sigue llamando `wirePollsTable(polls, catalog)` sin cambios. JSDoc de las tres funciones.

- [ ] **Step 4: Ejecutar los tests JS y las páginas**

Run: `pytest tests/test_js_polls.py tests/test_pages.py -q`
Expected: PASS

- [ ] **Step 5: Comprobación en el navegador**

Con el servidor del paso 6 de la tarea 7: en el detalle, filtrar por encuestadora y por meses (las dos filas fijas no cambian), cambiar a Escaños (la primera fila pasa a los escaños del modelo, «—» en todas si los escaños no son válidos) y ordenar por una cabecera (el pie se mantiene). Consola sin errores.

- [ ] **Step 6: Commit**

```bash
git add web/dist/js/polls-table.js web/dist/js/pages/polls-dom.js web/dist/js/pages/polls-wire.js web/dist/js/pages/elecciones.js tests/test_js_polls.py
git commit -m "Keep the model and the result as fixed rows of the polls table in every view"
```

---

### Task 9: Comprobación de conjunto y cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-10-19-web-eleccion-promedio-modelo-design.md` (estado)
- Modify: `deploy/README.md` (si algún texto de las tareas 1 y 3 quedó por ajustar)

- [ ] **Step 1: Suite completa**

Run: `pytest -q`
Expected: PASS; el recuento sube de 1144 a 1144 + los tests nuevos (13: 3 en la tarea 1, 1 en la 2, 2 en la 3, 3 en la 4, 3 en la 5, 3 en la 6, 1 en la 7, 1 en la 8 → ajustar la cifra al número real y anotarla en la spec).

- [ ] **Step 2: Prueba de humo de rutas** (si hay paquete local con la elección publicada)

Run: `bash deploy/check-api.sh` contra el servidor local, o la petición directa `curl -s localhost:8011/api/v1/events/es/2023-07-23/series | head -c 200`
Expected: JSON con `"schema": "event-series@1"` (o «n/a … sin publicar» si la elección local no tiene la parte).

- [ ] **Step 3: Cerrar la spec**

En la cabecera de la spec: «Estado: cerrada el {fecha} en `dev` (plan `docs/superpowers/plans/2026-10-19-web-eleccion-promedio-modelo.md`, {n} commits desde {hash}; {tests} tests)». Y una sección final «Pendiente de Luis»: correr el backtest (`python job.py backtest '{"scopes":"all"}'`), republicar las elecciones (`python job.py publish '{"what":["event"],"scopes":"all","event_date":"all"}'`), desplegar, y el test de integración `test_run_case_2023_30_days` contra la base.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-10-19-web-eleccion-promedio-modelo-design.md deploy/README.md
git commit -m "Close the spec of the poll average and the model in the detail of an election"
```
