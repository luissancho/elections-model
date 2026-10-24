# Titular de escaños = reparto central: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La predicción de escaños pasa a ser el reparto central (D'Hondt en cada circunscripción sobre el voto central, sin ruido), igual en nowcast y forecast; provincias, titular y bloques coinciden por construcción y la web deja de llamar «mediana» a la predicción.

**Architecture:** Un método nuevo del simulador, `central_result()`, hace una pasada `build_frame → build_umat → simulate` con el ruido apagado y la guarda en caché; `totals()` la devuelve por defecto (`method='central'`) y todo lo que cuelga de `totals` (resumen, bloques, titular, histórico, backtest) la hereda. Las partes `scenario` y `districts` exportan ese reparto, y la web sólo cambia textos y las marcas de los gráficos (crestas e histograma) para señalar el reparto en vez de la mediana.

**Tech Stack:** Python 3.11, pandas/numpy, pytest; Jinja2; ES modules en `web/dist/js` probados con `node` (los tests JS se saltan sin `node`).

**Spec:** `docs/superpowers/specs/2026-10-23-titular-reparto-central-design.md`

## Global Constraints

- El reparto se calcula con `params['random']` puesto a `False` y restaurado a mano, nunca con `set_params` (reconstruye `rng` y `rng_swing` en cada llamada con argumentos). No debe tocar `rng`, `rng_swing`, `frames`, `units` ni `results`.
- Con `split=False` no hay reparto: `central_result()` devuelve `None` y `totals()` cae a `method='median'` (comportamiento actual).
- El contrato del bundle sigue en 1. `SCHEMAS['scenario']` pierde la clave `simulation`; los runs antiguos (que la traen) siguen validando porque `validate` admite claves de más.
- `tests/fixtures/bundle/scenario.json` y el `SCENARIO` de `tests/test_pages.py` conservan `simulation` a propósito: son runs antiguos.
- Redacción pública: nosotros, «sólo» con tilde, encuestadora/casa; ningún texto de `escanos.html`, `metodo_escanos.html`, `escanos.js` ni `seats.js` llama mediana al titular. Nombres de código en inglés neutro.
- Luis hace commits en paralelo en el mismo árbol: antes de cada commit, `git status` y añadir sólo los ficheros de la tarea.
- Cada commit termina con la línea `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Suite completa al final: `pytest -q` (hoy 1 134+ tests; los JS necesitan `node`).

## Review Focus

- Barrera de ámbito (Canarias, Valencia): el reparto central debe pasar `scope_shares` y `threshold_scope` como `simulate` ya hace; un partido por debajo de la barrera provincial no entra. Test en la Tarea 1 con `sim.threshold = 20.0` (VOX queda a 0 y las sumas se mantienen).
- Modo de regresión (`split=False`): `totals()` y `summary()` no pueden fallar; vuelven a la mediana expandida. Test en la Tarea 1.
- Runs antiguos publicados con `simulation`: deben validar y renderizar sin cambios. Tests en las Tareas 2 y 4.
- Bloque sin partidos en el reparto (`Nadie`): `summary('vs')` sigue dando `seats` nulo y no rompe la suma. Test existente `test_export_summary_writes_null_for_a_block_without_parties` (Tarea 2 lo mantiene).
- Crestas de un bloque sin cifra central (por ejemplo un bloque del `bmap` sin fila en `summary.blocks`): el rótulo vuelve a «Mediana» y la marca gruesa a la mediana. Test en la Tarea 5.

---

### Task 1: `central_result()` y `totals(method='central')` en el simulador

**Files:**
- Modify: `mtpy/lib/simulator.py` (`__init__` junto a `self.results = None`; `set_params` 364-411; `totals` 826-856; `scenario` 857-866; `summary` 876-925; `summarize_seats` docstring; `unit_summary` 2049-2075; `run` 2144-2180)
- Modify: `tests/fakes.py:89-138` (`synthetic_simulator`)
- Modify: `tests/integration/test_model.py:32,148-151,200,255,414`, `tests/integration/test_regional_model.py:105`
- Test: `tests/test_simulator_unit.py`

**Interfaces:**
- Produces: `Simulator.central_result() -> Optional[pd.DataFrame]` (índice como `result()`: nombres de región, `name='region'`; columnas `params['names']`; enteros; `None` con `split=False` o sin `forecast`). Atributo `self.central_table` (caché, `None` hasta calcular; no `self.central`, que es el método estático `Simulator.central`).
- Produces: `Simulator.totals(sort=False, method='central') -> pd.Series` (índice `params['names']`, enteros; `method` en `('central', 'median', 'mean')`).
- Produces: `Simulator.summary(names=None, alpha=None, method='central')`; `Simulator.unit_summary(region, alpha=None)` con columnas `['pct', 'pct_lo', 'pct_hi', 'seats', 'seats_median', 'seats_mean', 'seats_lo', 'seats_hi', 'p_seats']`.
- Produces: `tests.fakes.synthetic_simulator` capaz de ejecutar `run()`, `build_umat`, `simulate` y `central_result()` (reparto nacional PP 5, PSOE 4, VOX 1; Madrid 3/2/1; Barcelona 2/2/0).

- [ ] **Step 1: Ampliar el fake para que la pasada real corra sin base de datos**

En `tests/fakes.py::synthetic_simulator`, tras `sim.industry_bias, sim.composition, sim.smap = ...`, añadir:

```python
    sim.threshold, sim.threshold_scope = 3.0, None
    sim.v2swing, sim.rng, sim.rng_swing, sim.region_groups, sim.central_table = None, np.random.default_rng(42), None, {}, None
    # Resultados previos por región (la fila 0 es la media ponderada por `prev_totals.votes`): la base del swing
    prev = pd.DataFrame({'PP': [38., 42., 32.], 'PSOE': [32., 28., 38.], 'VOX': [14., 15., 12.]},
                        index=pd.Index(REGIONS, name='region_id'))
    sim.prev_results = pd.concat({'votes': prev * 10, 'pct': prev, 'seats': prev * 0}, axis=1)
```

El `smap` del fake (`{'UP': [{'agg': ['UP', 'MP']}]}`) no cambia: sin `type` ni `names`, `apply_source_rules` lo ignora, y `test_publish_unit.py:354` lo fija.

- [ ] **Step 2: Escribir los tests que fallan**

En `tests/test_simulator_unit.py`, al final, con `from tests.fakes import NAMES, REGIONS, synthetic_simulator`:

```python
def test_central_result_is_a_dhondt_allocation_of_the_central_vote():
    sim = synthetic_simulator()
    table = sim.central_result()
    assert list(table.columns) == NAMES and list(table.index) == ['es', 'Madrid', 'Barcelona']
    assert table.loc['es'].tolist() == [5, 4, 1]
    assert table.loc['Madrid'].tolist() == [3, 2, 1] and table.loc['Barcelona'].tolist() == [2, 2, 0]
    assert table.loc['es'].equals(table.loc[['Madrid', 'Barcelona']].sum())
    assert table.dtypes.eq(int).all()


def test_totals_is_the_central_allocation_and_keeps_the_old_statistics():
    sim = synthetic_simulator()
    assert sim.totals().tolist() == [5, 4, 1] and sim.totals().index.tolist() == NAMES
    assert sim.totals(sort=True).index.tolist() == ['PP', 'PSOE', 'VOX']
    expanded = Simulator.round_proportional(Simulator.central(sim.dist()), sim.n_seats)
    assert sim.totals(method='median').tolist() == expanded.loc[NAMES].tolist()
    assert int(sim.totals(method='mean').sum()) == 10
    with pytest.raises(ValueError):
        sim.totals(method='mode')


def test_central_result_leaves_the_simulations_and_the_generators_alone():
    a, b = synthetic_simulator(), synthetic_simulator()
    a.params['n_sim'] = b.params['n_sim'] = 5
    a.run(reset=False)
    b.totals()  # la pasada determinista entre medias no debe mover la secuencia aleatoria
    b.run(reset=False)
    assert np.array_equal(a.results, b.results) and b.params['random'] is True and b.rng is not None
    assert b.central_table is None  # `run` vacía la caché


def test_central_result_is_the_same_in_both_modes_and_cached():
    sim = synthetic_simulator()
    first = sim.central_result()
    assert sim.central_result() is not first and sim.central_result().equals(first)
    sim.mode = 'forecast'
    sim.params['horizon'] = 'deadline'
    assert sim.central_result().equals(first)
    sim.set_params(n_sim=7)
    assert sim.central_table is None


def test_central_result_applies_the_threshold_and_regression_mode_falls_back_to_the_median():
    sim = synthetic_simulator()
    sim.threshold = 20.0
    table = sim.central_result()
    assert (table['VOX'] == 0).all() and table.loc['es'].sum() == 10 and table.loc['Madrid'].sum() == 6
    flat = synthetic_simulator()
    flat.params['split'] = False
    assert flat.central_result() is None
    assert flat.totals().tolist() == flat.totals(method='median').tolist()


def test_summary_headline_is_the_central_allocation_and_blocks_add_up():
    sim = synthetic_simulator()
    parties = sim.summary()
    assert parties['seats'].tolist() == [5, 4, 1] and 'seats_median' in parties.columns
    blocks = sim.summary('vs')
    assert blocks.loc['Derecha', 'seats'] == 6 and blocks.loc['Izquierda', 'seats'] == 4


def test_unit_summary_carries_the_central_seats_and_the_median():
    sim = synthetic_simulator()
    madrid = sim.unit_summary(28)
    assert list(madrid.columns) == ['pct', 'pct_lo', 'pct_hi', 'seats', 'seats_median', 'seats_mean', 'seats_lo', 'seats_hi', 'p_seats']
    assert madrid['seats'].tolist() == [3, 2, 1]
    assert madrid['seats_median'].tolist() == pd.DataFrame(sim.results[:, 1, :], columns=NAMES).median().tolist()
```

- [ ] **Step 3: Comprobar que fallan**

Run: `pytest tests/test_simulator_unit.py -q -k "central or totals_is or summary_headline or unit_summary_carries"`
Expected: FAIL (`AttributeError: 'Simulator' object has no attribute 'central_result'` y afines).

- [ ] **Step 4: Implementar en `mtpy/lib/simulator.py`**

- `__init__`: `self.central_table = None` junto a `self.results = None`.
- `set_params`: tras el `return self` temprano (sin argumentos y con `params`), poner `self.central_table = None`.
- `run`: `self.central_table = None` justo después de `self.set_params(...)`.
- `central_result(self) -> Optional[pd.DataFrame]`: si `not self.params['split']` o `self.forecast is None`, `None`; si `self.central_table is not None`, `self.central_table.copy()`; si no, guardar `self.params['random']`, ponerlo a `False`, dentro de `try/finally` hacer `frame = self.build_frame()`, `umat = self.build_umat(frame)`, `result = self.simulate(frame, umat)`, restaurar el flag; construir el DataFrame como `result()` (índice de nombres de región, columnas `params['names']`, `dtype=int` tras `round`), guardarlo en `self.central_table` y devolver una copia. Dentro de la clase, llamar al método estático como `Simulator.central(...)` sólo si hiciera falta; `self.central(...)` sigue siendo el método. Docstring: «reparto central: D'Hondt en cada circunscripción sobre el voto central (la media del forecast proyectada con el swing proporcional, sin ruido ni deriva); el titular publicado; igual en nowcast y forecast».
- `totals(self, sort=False, method='central')`: validar `method` en `('central', 'median', 'mean')` (ValueError como `central`); con `'central'`, `table = self.central_result()`; si es `None`, seguir como `'median'`; si no, `ds = table.iloc[self.params['regions'].index(self.default_region)].astype(int)`; el resto (`sort`, `.loc[names]`) igual. Docstring: titular = reparto central; `'median'`/`'mean'` son la central expandida con `round_proportional`.
- `summary(..., method='central')`: `seats = self.totals(method=method)`; a `summarize_seats` pasar `method='median' if method == 'central' else method` (sólo cuenta sin `headline`).
- `summarize_seats` docstring: «`seats` (titular: el `headline` dado, o la central expandida sin él)».
- `unit_summary`: `central = self.central_result()`; `seats_central = central.iloc[rloc] if central is not None else seats.median()`; devolver las columnas en el orden del Interfaces, con `'seats': seats_central` y `'seats_median': seats.median()`.
- `scenario()` y `closest_simulation` se quedan; docstring de `scenario`: «ya no se publica (ver `central_result`)».

- [ ] **Step 5: Comprobar que pasan, y el resto del fichero también**

Run: `pytest tests/test_simulator_unit.py tests/test_publish_unit.py tests/test_jobs_publish.py -q`
Expected: los nuevos PASS; fallan sólo `test_export_districts_*` (tipo de `seats`) y `test_export_scenario_is_a_real_simulation`, que la Tarea 2 cambia.

- [ ] **Step 6: Adaptar los tests de integración (no se ejecutan sin base)**

En `tests/integration/test_model.py` líneas 200, 255 y 414 (`REGRESSION_TOTALS`, «la secuencia aleatoria no cambia»): `sim26.totals()` → `sim26.totals(method='median')`. Líneas 32, 148-151 y `test_regional_model.py:105` comprueban sumas: no cambian. Añadir en `test_totals_sum_to_n_seats_both_methods`: `assert int(sim26.totals(method='central').sum()) == 350`.

- [ ] **Step 7: Commit**

```bash
git status
git add mtpy/lib/simulator.py tests/fakes.py tests/test_simulator_unit.py tests/integration/test_model.py
git commit -m "Simulador: el titular es el reparto central (central_result, totals method='central')

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Publicar el reparto en `scenario` y `districts`

**Files:**
- Modify: `mtpy/lib/publish/forecast.py` (`SUMMARY_COLS` 46; `export_districts` 126-160; `export_scenario` 161-192)
- Modify: `mtpy/lib/bundle.py:112` (`SCHEMAS['scenario']`)
- Modify: `mtpy/lib/pages/escanos.py:19` (`CSV_PARTS`)
- Test: `tests/test_publish_unit.py:70-82,125-132`, `tests/test_pages.py:668`

**Interfaces:**
- Consumes: `Simulator.central_result()`, `Simulator.unit_summary()` (Tarea 1).
- Produces: parte `scenario` `{'parties': list, 'rows': list}` (filas `region_id`, `region`, `seats`, sin `simulation`); parte `districts` con `rows[*].seats` entero y `rows[*].seats_median` flotante; etiqueta CSV «Reparto central».

- [ ] **Step 1: Cambiar los tests**

En `tests/test_publish_unit.py` sustituir `test_export_scenario_is_a_real_simulation` por:

```python
def test_export_scenario_is_the_central_allocation_and_old_runs_still_validate():
    sim = synthetic_simulator()
    data, frame = publish.export_scenario(sim)
    data = valid('scenario', data)
    assert 'simulation' not in data
    assert [r['region_id'] for r in data['rows']] == REGIONS and data['rows'][0]['region'] == 'es'
    assert [r['seats'] for r in data['rows']] == [[5, 4, 1], [3, 2, 1], [2, 2, 0]]
    assert list(frame.columns) == ['region_id', 'region'] + NAMES
    old = valid('scenario', {'simulation': 0, 'parties': NAMES, 'rows': data['rows']})
    assert old['simulation'] == 0


def test_districts_seats_are_the_central_allocation_of_the_scenario():
    sim = synthetic_simulator()
    scenario = valid('scenario', publish.export_scenario(sim)[0])
    districts = valid('districts', publish.export_districts(sim)[0])
    summary = valid('summary', publish.export_summary(sim)[0])
    by_region = {r['region_id']: dict(zip(scenario['parties'], r['seats'])) for r in scenario['rows']}
    assert all(by_region[row['region_id']][row['name']] == row['seats'] for row in districts['rows'])
    assert summary['totals'] == by_region[0]
```

En `test_export_districts_*` (línea 75-82): el conjunto de claves de la fila añade `'seats_median'` y la última aserción pasa a `assert all(isinstance(r['seats'], int) for r in data['rows'])` más `assert all(isinstance(r['seats_median'], float) for r in data['rows'])`.

En `tests/test_pages.py:668`: `'Escenario central'` → `'Reparto central'`.

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_publish_unit.py -q -k "scenario or districts" && pytest tests/test_pages.py -q -k csv_links`
Expected: FAIL (`KeyError`/`assert 'simulation' not in data`, tipo de `seats`, etiqueta).

- [ ] **Step 3: Implementar**

- `mtpy/lib/bundle.py:112`: `'scenario': {'parties': list, 'rows': list}`.
- `export_scenario(sim)`: `table = sim.central_result()`; mismas filas y CSV; `data = {'parties': parties, 'rows': rows}`. Docstring: reparto central, igual en los dos modos.
- `SUMMARY_COLS = ('pct', 'pct_lo', 'pct_hi', 'seats', 'seats_median', 'seats_mean', 'seats_lo', 'seats_hi', 'p_seats')`; en `export_districts`, `round_cols(..., seats=(*SEATS_COLS, 'seats_median'))` y `seats` a entero (`int(row['seats'])`) en cada fila; columnas del CSV = `['region_id', 'region', 'name', *SUMMARY_COLS]`.
- `mtpy/lib/pages/escanos.py:19`: `('scenario', 'Reparto central', True)`.

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py tests/test_pages.py -q`
Expected: PASS (salvo `test_escanos_majority_calculator_draws_the_bar_above_the_histogram`, que la Tarea 4 cambia).

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/lib/publish/forecast.py mtpy/lib/bundle.py mtpy/lib/pages/escanos.py tests/test_publish_unit.py tests/test_pages.py
git commit -m "Publicación: scenario y districts exportan el reparto central

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Resumen provincial del backtest con el reparto

**Files:**
- Modify: `mtpy/lib/backtest.py:120-153` (`_provinces`)
- Test: `tests/test_backtest_unit.py`

**Interfaces:**
- Consumes: `Simulator.central_result()` (Tarea 1).
- Produces: filas de `_provinces` con `model_seats` (reparto central de la provincia) y `median_seats` (mediana de las simulaciones).

- [ ] **Step 1: Escribir el test que falla**

```python
def test_provinces_report_the_central_allocation_and_keep_the_median(monkeypatch):
    from mtpy.lib import backtest
    from tests.fakes import NAMES, synthetic_simulator
    sim = synthetic_simulator()
    empty = pd.DataFrame(index=pd.Index([], name='region_id'))
    monkeypatch.setattr(backtest, 'official_provinces', lambda scope, event_date: (empty, empty))
    rows = backtest._provinces(sim, 'es', '2026-11-29', 30, {'PP'}).set_index(['region', 'party'])
    assert rows.loc[(28, 'PP'), 'model_seats'] == 3 and rows.loc[(8, 'VOX'), 'model_seats'] == 0
    medians = pd.DataFrame(sim.results[:, 1, :], columns=NAMES).median()
    assert rows.loc[(28, 'PSOE'), 'median_seats'] == medians['PSOE']
```

- [ ] **Step 2: Comprobar que falla**

Run: `pytest tests/test_backtest_unit.py -q -k provinces_report`
Expected: FAIL (`KeyError: 'median_seats'`).

- [ ] **Step 3: Implementar**

En `_provinces`, antes del bucle `central = sim.central_result()`; en cada fila `'model_seats': float(central.loc[sim.region_names.get(r, r), n]) if central is not None else float(np.median(seats))` y `'median_seats': float(np.median(seats))`. Docstring: «seats (reparto central, mediana, intervalos, CRPS)».

- [ ] **Step 4: Comprobar que pasa**

Run: `pytest tests/test_backtest_unit.py tests/test_jobs_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/lib/backtest.py tests/test_backtest_unit.py
git commit -m "Backtest: el resumen provincial usa el reparto central y guarda la mediana

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Textos públicos de Escaños y Método

**Files:**
- Modify: `web/templates/escanos.html` (líneas 91-105 «Cómo leerlo» del reparto; 128-141 «Cómo leerlo» de la calculadora; 163 pie de la tabla; 240-253 «Cómo leerlo» del territorio)
- Modify: `web/templates/metodo_escanos.html:244-260` («Qué es cada cifra»)
- Test: `tests/test_pages.py:2371` (`test_metodo_escanos_text`), `3165-3176` (`test_escanos_majority_calculator_draws_the_bar_above_the_histogram`), más un test nuevo

**Interfaces:**
- Consumes: nada nuevo (el contexto de la página ya trae `summary`, `district_table`, `region_rows`).

- [ ] **Step 1: Cambiar los tests**

En `test_escanos_majority_calculator_draws_the_bar_above_the_histogram` sustituir la aserción de «la suma de esas medianas…» por:

```python
    assert 'La franja de arriba y la línea del histograma son el reparto central' in re.sub(r'\s+', ' ', card)
    assert 'mediana' not in card.lower()
```

Añadir:

```python
def test_escanos_texts_call_the_headline_the_central_allocation(api, fresh_app):
    write_districts(fresh_app.fs)
    page = re.sub(r'\s+', ' ', html(call(api, '/es/escanos')[2]))
    assert 'Reparto central: D'Hondt en cada circunscripción sobre el voto central; cada fila suma sus escaños.' in page
    assert 'la línea gruesa el reparto central' in page and 'la línea gruesa la mediana' not in page
    assert 'la simulación más cercana a la mediana' not in page and 'Escenario central' not in page
    assert 'la suma de las circunscripciones es el titular' in page
    assert 'Reparto central' in page  # etiqueta del CSV


def test_metodo_escanos_explains_the_central_allocation(api):
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert '<strong>Los escaños de cada partido</strong> son su reparto central' in text
    assert 'el mismo titular en el nowcast y en el forecast' in text
    assert 'a veces saca escaño y a veces no' in text
    assert 'son su mediana' not in text and 'El escenario central' not in text
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_pages.py -q -k "central_allocation or majority_calculator_draws or metodo_escanos"`
Expected: FAIL.

- [ ] **Step 3: Reescribir los textos**

- `escanos.html:163` (caption): «Reparto central: D'Hondt en cada circunscripción sobre el voto central; cada fila suma sus escaños. Pasa el ratón para ver el rango del 95%.»
- `escanos.html:91-105`: «La barra reparte los escaños del reparto central…» (ya); en el párrafo de las curvas, «la línea gruesa el reparto central».
- `escanos.html:128-141`: el primer párrafo se queda; el tercero pasa a decir que la franja de arriba y la línea del histograma son el reparto central (la suma de los escaños de cada partido, la misma cifra que la barra y que los bloques en el resto de la web) y que el rango del 95% y la probabilidad salen de las simulaciones, sumadas una a una; desaparece la explicación de por qué la mediana no coincide.
- `escanos.html:240-253`: «La tabla reparte los escaños de cada circunscripción con D'Hondt sobre el voto central: la suma de las circunscripciones es el titular de la página y la de los partidos de cada bloque, la cifra del bloque. Al elegir una circunscripción, el detalle muestra ese reparto con el rango del 95% y la probabilidad de escaño de las simulaciones.» Mantener los párrafos de los intervalos y de las CCAA; el último pasa a «Es un reparto del voto central, no la predicción cerrada de ninguna circunscripción: en las pequeñas, pocos votos cambian un escaño.»
- `metodo_escanos.html:246-248`: «<strong>Los escaños de cada partido</strong> son su reparto central: proyectamos el promedio de sondeos a cada circunscripción con el swing proporcional, sin sorteo, y aplicamos D'Hondt y la barrera. La suma de las circunscripciones es el titular y los bloques suman sus partidos. Es el mismo titular en el nowcast y en el forecast: la deriva sólo abre los rangos.» Nueva viñeta tras la de las probabilidades: «<strong>Dónde cae el reparto</strong> dentro del rango: en un partido que a veces saca escaño y a veces no, el reparto central puede quedar lejos del centro de su rango.» La viñeta «El escenario central» desaparece.
- Comprobar que `_glosario.html` no define «escenario central» (hoy no).

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_pages.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add web/templates/escanos.html web/templates/metodo_escanos.html tests/test_pages.py
git commit -m "Escaños y Método: los textos explican el reparto central

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: JS: crestas, calculadora y pies de la tabla marcan el reparto

**Files:**
- Modify: `web/dist/js/kde.js:107-141,151-165` (`ridgeRows`, `ridgeData`, `ridgeBlocks`)
- Modify: `web/dist/js/charts/ridge.js:98,131-132` (marca y rótulo)
- Modify: `web/dist/js/seats.js:97-112` (`coalitionCaption`)
- Modify: `web/dist/js/pages/escanos.js:60-115` (`wireCoalition`), `117-122` (`CAPTION`), `213,222` (crestas)
- Test: `tests/test_js_kde.py`, `tests/test_js_charts.py`, `tests/test_js_seats.py:63-68`

**Interfaces:**
- Produces: `ridgeRows(series, {majority, points, minWidth, range, central = {}})`: cada fila lleva `central` (`central[name]` si es número, si no `null`); `ridgeData` y `ridgeBlocks` pasan `opts` tal cual.
- Produces: `renderRidge`: marca gruesa en `row.central ?? row.median`; rótulo «Reparto central: N» con `central`, «Mediana: N» sin él.
- Produces: `coalitionCaption(stats, central)` → `[{label: 'Simulaciones'}, {label: 'Reparto central', value: fmtInt(central)}, {label: 'Rango 95%'}]`.

- [ ] **Step 1: Cambiar y añadir tests**

`tests/test_js_seats.py:63-68`:

```python
def test_coalition_caption_gives_the_simulations_the_central_seats_and_the_range():
    caption = run_js('s.coalitionCaption({n: 1000, median: 201, lo: 181, hi: 220.5, pMajority: 0.992}, 202)')
    assert caption == [{'label': 'Simulaciones', 'value': '1.000'}, {'label': 'Reparto central', 'value': '202'},
                       {'label': 'Rango 95%', 'value': '181–221'}]
```

`tests/test_js_kde.py`, junto a `test_ridge_rows_scale_each_row_to_its_peak_over_its_own_range` (misma forma de llamada):

```python
def test_ridge_rows_carry_the_central_seats_when_given():
    rows = run_js('k.ridgeData(dist, ["PP", "PSOE"], {central: {PP: 141}})')
    assert rows[0]['central'] == 141 and rows[1]['central'] is None
```

`tests/test_js_charts.py`, con un ayudante `ridge_option(rows_js, args)` calcado de `histogram_option` (línea 603): `renderRidge` recibe un contenedor falso `{__chart: {setOption: (o) => { got = o; }}, style: {}, clientWidth: 0}` (usa `el.style.height`, `el.clientWidth` y `mountChart`, que reutiliza `__chart`). Filas mínimas: `[{name: 'PP', x0: 130, x1: 150, grid: [130, 140, 150], y: [0, 1, 0], median: 139.5, lo: 133, hi: 150, central: 141}, {name: 'PSOE', x0: 95, x1: 115, grid: [95, 105, 115], y: [0, 1, 0], median: 105, lo: 100, hi: 112, central: null}]`.

```python
def test_ridge_marks_and_labels_the_central_seats_and_falls_back_to_the_median():
    option = ridge_option(ROWS_JS, '{colors: {PP: "#1d4f86"}, majority: 176}')
    thick = [[m['xAxis'] for m in s['markLine']['data'] if m['lineStyle'].get('width') == 2] for s in option['series']]
    assert thick == [[141], [105]]
    texts = [g['style']['text'] for g in option['graphic'] if g['type'] == 'text']
    assert texts[0].startswith('{b|PP}\nReparto central: {b|141}') and texts[1].startswith('{b|PSOE}\nMediana: {b|105}')
```

(Si `graphic` no es una lista plana de textos en la opción, localizar los rótulos como lo hace `renderRidge` y ajustar la extracción, no la aserción.)

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_js_seats.py tests/test_js_kde.py tests/test_js_charts.py -q -k "caption or central"`
Expected: FAIL.

- [ ] **Step 3: Implementar**

- `kde.js::ridgeRows`: opción `central = {}`; en cada fila `central: typeof central[name] === 'number' ? central[name] : null`.
- `ridge.js`: `const mark = row.central ?? row.median;` para la marca gruesa; rótulo `${row.central !== null && row.central !== undefined ? 'Reparto central' : 'Mediana'}: {b|...}`; comentario de cabecera y JSDoc actualizados.
- `seats.js::coalitionCaption(stats, central)`: segunda línea «Reparto central».
- `escanos.js`: `CAPTION.esc` = «Reparto central: D'Hondt en cada circunscripción sobre el voto central; cada fila suma sus escaños. Pasa el ratón para ver el rango del 95%.», `CAPTION.ccaa` = «Reparto central: D'Hondt en cada circunscripción sobre el voto central; cada fila suma los escaños de las provincias de la comunidad.»; en `wireCoalition`, `renderHistogram(..., {median: central, markLabel: \`Reparto central: ${fmtInt(central)}\`, caption: coalitionCaption(stats, central), ...})`; en `wireSeats`, `ridgeData(dist, parties.names, {majority, range, central: Object.fromEntries(parties.rows.map((row) => [row.name, row.seats]))})` y `ridgeBlocks(..., {majority, range, central: Object.fromEntries(blockRows.map((row) => [row.name, row.seats]))})`. Comentarios de `seats.js:80-101` y de `partyBar` dejan de hablar de «central scenario» como simulación.

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_js_seats.py tests/test_js_kde.py tests/test_js_charts.py tests/test_js_territory.py tests/test_js_histogram.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add web/dist/js/kde.js web/dist/js/charts/ridge.js web/dist/js/seats.js web/dist/js/pages/escanos.js tests/test_js_seats.py tests/test_js_kde.py tests/test_js_charts.py
git commit -m "Escaños (JS): crestas, calculadora y tabla marcan el reparto central

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Suite completa, spec cerrada y lista para Luis

**Files:**
- Modify: `docs/superpowers/specs/2026-10-23-titular-reparto-central-design.md` (estado)

- [ ] **Step 1: Suite completa**

Run: `pytest -q`
Expected: todo PASS (los JS se saltan sin `node`; si `node` está, también PASS).

- [ ] **Step 2: Cerrar la spec**

Estado: «construido el 2026-10-23 (plan `docs/superpowers/plans/2026-10-23-titular-reparto-central.md`)». Debajo, la lista de tareas de Luis en orden: backtest + `publish backtest`; `publish event` con `event_date: all`; backfill y run nuevo; desplegar la web; fijar las cifras del reparto en los tests de integración cuando corra contra la RDS.

- [ ] **Step 3: Commit**

```bash
git status
git add docs/superpowers/specs/2026-10-23-titular-reparto-central-design.md
git commit -m "Spec titular reparto central: construido

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
