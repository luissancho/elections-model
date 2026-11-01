# Calibrar la incertidumbre del resultado (parte B) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que las bandas y las probabilidades del modelo cumplan lo que prometen a cualquier horizonte: un diagnóstico de fiabilidad (PIT) en el backtest y en la web, el error terminal medido sobre el propio promedio, el crecimiento con el horizonte medido igual, colas con un parámetro propio y la correlación entre partidos reevaluada.

**Architecture:** Cinco pasos en orden, cada uno con su backtest completo antes / después (spec 17): el diagnóstico no toca el modelo (Task 1); después una tabla nueva `elections.average_errors` y su estimador (Tasks 2-3), un factor de escala de la deriva (Task 4), los grados de libertad fijos de la t (Task 5) y la composición (Task 6). Todo entra por parámetro del `Simulator` (`error_model`, `drift_scale`, `tail_dof`), enhebrado por el backtest y su job; con los valores por defecto el backtest reproduce `backtest/results/` (spec 19). La web gana una tarjeta (Task 7).

**Tech Stack:** Python 3.11, pandas, numpy, scipy (`student_t`), pytest; PostgreSQL (tabla nueva, sólo en integración); echarts en la web.

**Spec:** `docs/superpowers/specs/2026-11-01-sondeos-futuros-y-calibracion-design.md` (parte B, decisiones 11-17, 18, 19, y «Decisiones tras la revisión» 1, 2, 3, 7).

## Global Constraints

- Con `error_model='polls'`, `drift_scale=None` (= 1) y `tail_dof=None`, el backtest reproduce `backtest/results/` exactamente, misma semilla (spec 19); `tests/integration/test_backtest.py` lo comprueba en un caso.
- Fuera de muestra: el estimador del error del promedio y el factor de escala de la deriva de un caso se ajustan sólo con las elecciones anteriores a la evaluada, del ámbito estimador (`est_scope`, regla de M11); la tabla se calcula con la configuración viva del Simulator (efectos de casa, muestra efectiva y rebaño activados).
- El PIT se resume en tres poblaciones separadas (generales, autonómicas, provincias), cada elección con el mismo peso dentro de la suya; deciden las dos primeras al nivel de partido (spec 11).
- Criterio de cada paso (spec 17): coberturas y PIT más cerca del nominal donde faltaban **sin** empeorar donde ya cubrían, y log-score y CRPS no peores más allá del ruido; cada decisión con su tabla antes / después en `docs/analisis-2026-09/metodo-15-calibracion.md`.
- Ningún valor por defecto del modelo vivo cambia en este plan salvo en el paso de decisión de cada task, en commit aparte, antes del 20-11 o después del 29-N; si la parte A cambia el promedio antes del 13-11, la tabla y los backtests de B se rehacen sobre el promedio nuevo (spec, regla de encaje).
- Redacción pública con las reglas de Luis (`assert_public_wording`); textos de la web en el tono de `/metodo` (nosotros, todo a la vista).
- Tests unitarios sin base de datos; mensajes de commit en español, sin atribución de modelo.

## Review Focus

- Un caso con `nowcast_only` (sin ejecución `_h`): `calibration.csv` lleva sólo las filas `mode='now'` y nada falla (Task 1).
- Un partido principal sin resultado oficial en la elección (no aparece en `events_results`): su PIT es NaN y no entra en ninguna caja (Task 1).
- Un caso cuyo ámbito estimador no tiene filas en `average_errors` (los primeros ciclos, o una base sin la tabla): `error_model='average'` cae a `'polls'` con un aviso y `meta` lo registra (Task 3).
- Un factor de escala sin filas a 30 días o más, o fuera de [1, 4]: queda en 1 con un aviso y `meta` lo registra (Task 4).
- `tail_dof` igual a 0, negativo o no entero: `ValueError` al construir el Simulator (Task 5).

---

### Task 1: PIT por caso, `calibration.csv` y el resumen por poblaciones

**Files:**
- Modify: `mtpy/lib/backtest.py:119-130` (`_quantile_cols`), `:238-428` (`run_case`), `:525-598` (`run_backtest`), `:817-900` (`backtest_scope`, `RESULT_FILES`)
- Test: `tests/test_backtest_unit.py`

**Interfaces:**
- Produces: `backtest.pit(samples: np.ndarray, y: float) -> float`: fracción de muestras estrictamente menores que `y` más la mitad de las iguales; NaN si `y` no es finito.
- Produces: `backtest.QUANTILES = [0.05, 0.10, ..., 0.95]` y `_quantile_cols` añade `q05`…`q95` (más los `lo`/`hi` de hoy), con el mismo sufijo `_h`.
- Produces: en `run_case`, `calibration`: DataFrame con una fila por `(event_date, horizon, kind, mode, party[, region_id])`, `kind` ∈ `{'shares', 'seats', 'prov_shares', 'prov_seats'}`, `mode` ∈ `{'now', 'h'}` (las `_h` sólo con la ejecución a horizonte; las provinciales sólo `now` y con escaños válidos), columnas `main`, `official`, `model`, `pit`. `run_backtest` concatena `calibration`; `backtest_scope` escribe `calibration.csv` (`RESULT_FILES` lo incluye).
- Produces: `backtest.summarize_calibration(calibration: pd.DataFrame, bins: int = 10, groups: dict[str, tuple[int, int]] = HORIZON_GROUPS) -> pd.DataFrame`: una fila por `(population, kind, mode, group, bin)` con `expected` (1 / bins), `observed` (fracción de PIT en la caja, cada elección con peso 1 / sus celdas), `n_cells`, `n_events`; `population` la pone el llamador en una columna `population` del frame (`'generales'`, `'autonomicas'`; las filas `prov_*` van a `'provincias'`); `HORIZON_GROUPS = {'6-14': (6, 14), '30-60': (30, 60), '90-180': (90, 180)}`.
- Produces: `backtest.majority_reliability(blocks: pd.DataFrame, bins: tuple = (0, .1, .3, .5, .7, .9, 1.)) -> pd.DataFrame`: una fila por caja de `p_majority` (y `p_majority_h` si existe, columna `mode`): `p_mean`, `observed` (fracción de `majority`), `n`.

- [ ] **Step 1: Tests**

```python
def test_pit_counts_the_mass_below_the_outcome_with_half_the_ties():
    assert pit(np.array([1., 2., 3., 4.]), 2.5) == pytest.approx(0.5)
    assert pit(np.array([1., 2., 2., 4.]), 2.) == pytest.approx(0.5)   # 1 por debajo + 2 empates / 2 = 2 de 4
    assert np.isnan(pit(np.array([1., 2.]), np.nan))

def test_quantile_cols_carry_the_five_percent_grid():
    cols = _quantile_cols(np.arange(1., 101.))
    assert cols['lo95'] < cols['q05'] < cols['q50'] < cols['q95'] < cols['hi95'] and 'q50_h' in _quantile_cols(np.arange(3.), suffix='_h')

def test_summarize_calibration_weights_each_election_equally_and_bins_by_horizon_group():
    cal = pd.DataFrame({
        'population': ['generales'] * 6, 'kind': ['shares'] * 6, 'mode': ['now'] * 6,
        'event_date': ['2023-07-23'] * 4 + ['2019-11-10'] * 2, 'horizon': [6, 6, 6, 6, 14, 14],
        'pit': [0.05, 0.05, 0.05, 0.05, 0.95, 0.95], 'official': 1., 'model': 1., 'main': True, 'party': list('ABCDEF')})
    out = summarize_calibration(cal, bins=10)
    row = out.query("group == '6-14'")
    assert row.loc[row.bin == 0, 'observed'].iloc[0] == pytest.approx(0.5)   # 2023 pesa 1/2 aunque tenga 4 celdas
    assert row.loc[row.bin == 9, 'observed'].iloc[0] == pytest.approx(0.5)
    assert row['n_events'].iloc[0] == 2 and row['n_cells'].iloc[0] == 6 and out['expected'].eq(0.1).all()

def test_summarize_calibration_drops_nan_pit_and_keeps_only_now_without_h():
    cal = pd.DataFrame({'population': ['generales'] * 2, 'kind': ['shares'] * 2, 'mode': ['now'] * 2, 'event_date': ['2023-07-23'] * 2,
                        'horizon': [6, 6], 'pit': [0.5, np.nan], 'official': [1., np.nan], 'model': 1., 'main': True, 'party': ['A', 'B']})
    out = summarize_calibration(cal)
    assert out['n_cells'].iloc[0] == 1 and set(out['mode']) == {'now'}

def test_majority_reliability_bins_probabilities():
    blocks = pd.DataFrame({'p_majority': [0.05, 0.95, 0.6], 'majority': [False, True, False], 'seats_valid': True})
    out = majority_reliability(blocks)
    assert out.loc[out.n > 0, 'observed'].tolist() == [0., 0., 1.] and out['mode'].eq('now').all()
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_backtest_unit.py -v -k "pit or quantile_cols or calibration or reliability"`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar `pit`, `QUANTILES`, las filas de `calibration` en `run_case` (cuotas y escaños de `now['shares']`, `now['dist']`, `fwd[...]`; celdas provinciales desde `_provinces`, que ya tiene las muestras por celda), `summarize_calibration` y `majority_reliability`**

`summarize_calibration` descarta los PIT no finitos, asigna cada celda a su grupo de horizonte, pesa cada celda por `1 / n_cells(event_date, population, kind, mode, group)` y suma pesos por caja (`bin = min(int(pit · bins), bins − 1)`).

- [ ] **Step 4: Ejecutar los tests y la guarda de regresión**

Run: `pytest tests/test_backtest_unit.py tests/test_pipelines_backtest.py tests/test_jobs_backtest.py -q`
Expected: PASS. Con base: `pytest tests/integration/test_backtest.py -q` (`shares.csv` y `metrics.csv` del caso de prueba iguales que antes salvo las columnas nuevas).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/backtest.py tests/test_backtest_unit.py
git commit -m "backtest: PIT por celda, cuantiles de 5 en 5, calibration.csv y resumen por poblaciones y grupos de horizonte"
```

---

### Task 2: La tabla `elections.average_errors`

**Files:**
- Modify: `mtpy/models/elections.py:196-220` (junto a `Drift`), `mtpy/lib/data.py:1105-1160` (junto a `get_drift` / `save_drift_data`)
- Modify: `mtpy/lib/computer.py:2387-2460` (junto a `compute_drift` / `get_drift_data`)
- Modify: `mtpy/pipelines/compute/Compute.py:27-60`, `mtpy/jobs/Compute.py` (docstring y paso)
- Test: `tests/test_computer_unit.py`, `tests/test_pipelines_compute.py`, `tests/integration/test_average_errors.py`

**Interfaces:**
- Produces: modelo `AverageErrors` (`table = 'elections.average_errors'`, `key = ['event_date', 'event_scope', 'party_id', 'horizon']`, columnas `party`, `main` (bool), `level` (el promedio, `num [10, 4]`), `official`, `error` (modelo − oficial), `err` (`s`, error estándar de la regresión local en el ancla), `n_polls` (`int`), `as_of` (`dtd`)).
- Produces: `data.get_average_errors(scope: str = 'es', event_dates: Optional[list[str]] = None) -> pd.DataFrame` (frame vacío con las columnas si la tabla no existe) y `data.save_average_errors_data(data: pd.DataFrame) -> int` (reemplaza las filas de las mismas elecciones; crea la tabla si falta).
- Produces: `Computer.get_average_error_data(horizons: tuple[int, ...] = (6, 14, 30, 60, 90, 180), min_polls: int = 5, max_fc: int = 10, house_effects: bool = True, dispersion: bool = True, herding: bool = True, events: Optional[list[str]] = None) -> pd.DataFrame`: para cada elección `featured` del Computer (o `events`) con resultado oficial y cada horizonte, un `Forecaster(scope, event_date, drange=horizon, house_effects=..., dispersion=..., disp_params={'herding': ...})` con `fit_forecast(names=main, max_fc=max_fc, fillna=True)`, ancla `as_of = min(event − horizon, date_fit_last)` (la regla del Simulator), y una fila por partido principal: `level` = promedio en `as_of`, `err` = `fc_stat[party].loc[as_of]['err']`, `official` de `backtest.official_results`, `error = level − official`. Sin sondeos suficientes el caso se salta con aviso.
- Produces: `Computer.compute_average_errors(save: bool = False, **kwargs) -> pd.DataFrame` y `Computer.load_average_errors() -> pd.DataFrame` (las filas de `self.event_dates`), como `compute_drift` / `load_drift`.
- Produces: paso `average_errors` del pipeline `compute` tras `drift` (`compute_task`), omitido con `dry_run`.

- [ ] **Step 1: Test unitario de la forma de la tabla con un Forecaster sustituido**

```python
def test_get_average_error_data_builds_one_row_per_main_party_and_horizon(monkeypatch):
    c = _computer_without_db(monkeypatch, events=['2023-07-23'], main={'2023-07-23': ['PP', 'PSOE']})
    monkeypatch.setattr(computer_module, 'Forecaster', lambda **kw: _fake_forecaster(level={'PP': 33.5, 'PSOE': 28.2}, err={'PP': 0.4, 'PSOE': 0.5},
                                                                                  date_fit_last=pd.Timestamp('2023-07-23') - pd.Timedelta(days=kw['drange'])))
    monkeypatch.setattr(backtest_module, 'official_results', lambda scope, d: pd.DataFrame({'pct': [33.06, 31.68]}, index=['PP', 'PSOE']))
    df = c.get_average_error_data(horizons=(6, 30))
    assert df.shape[0] == 4 and set(df.columns) >= {'event_date', 'event_scope', 'party_id', 'party', 'horizon', 'main', 'level', 'official', 'error', 'err', 'n_polls', 'as_of'}
    row = df.query("party == 'PSOE' and horizon == 6").iloc[0]
    assert row['error'] == pytest.approx(28.2 - 31.68) and row['err'] == pytest.approx(0.5) and row['as_of'] == pd.Timestamp('2023-07-17')
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_computer_unit.py -v -k average_error_data`
Expected: FAIL con `AttributeError`

- [ ] **Step 3: Implementar el modelo, las funciones de `data.py`, `get_average_error_data`, `compute_average_errors`, `load_average_errors` y el paso del pipeline**

- [ ] **Step 4: Test del pipeline (el paso se llama tras `drift` y no con `dry_run`), calcado de los tests de `compute_drift` en `tests/test_pipelines_compute.py`**

- [ ] **Step 5: Test de integración (con base)**

```python
@pytest.mark.integration
def test_compute_average_errors_2023_matches_the_backtest_shares(db):
    df = Computer(scope='es').build_series().compute_average_errors(events=['2023-07-23'], horizons=(6,))
    ref = pd.read_csv('backtest/results/shares.csv').query("event_date == '2023-07-23' and horizon == 6 and main")
    merged = df.merge(ref, on='party')
    assert np.allclose(merged['level'], merged['model'], atol=0.05)   # mismo promedio que el backtest, misma configuración
```

- [ ] **Step 6: Ejecutar**

Run: `pytest tests/test_computer_unit.py tests/test_pipelines_compute.py -q && pytest tests/integration/test_average_errors.py -q`
Expected: PASS (integración `SKIPPED` sin base)

- [ ] **Step 7: Commit**

```bash
git add mtpy/models/elections.py mtpy/lib/data.py mtpy/lib/computer.py mtpy/pipelines/compute/Compute.py mtpy/jobs/Compute.py tests/test_computer_unit.py tests/test_pipelines_compute.py tests/integration/test_average_errors.py
git commit -m "computer: tabla average_errors (error del promedio nowcast por elección, partido y horizonte) y paso del compute"
```

---

### Task 3: El estimador del error del promedio y `error_model` en el Simulator y el backtest

**Files:**
- Modify: `mtpy/lib/computer.py` (junto a `get_error_estimator`, `:2259-2300`)
- Modify: `mtpy/lib/simulator.py:99-200` (constructor), `:438-456` (`fit_error_estimator`), `:1100-1130` (`_fan`), `:2016-2027` (`build_frame`), `meta` / `diagnostics`
- Modify: `mtpy/lib/backtest.py` (`run_case`, `run_backtest`, `backtest_scope`: parámetro `error_model`, `meta.csv` y `run.json`), `mtpy/jobs/Backtest.py`, `mtpy/pipelines/backtest/Backtest.py`, `backtest/run_backtest.py` (`--error-model`)
- Test: `tests/test_computer_unit.py`, `tests/test_simulator_unit.py`, `tests/test_jobs_backtest.py`

**Interfaces:**
- Produces: `Computer.get_average_error_estimator(horizon: int = 6, net: bool = True) -> Optional[AverageErrorEstimator]`: ajusta `|e_net|` (`e_net² = max(error² − err², 0)` con `net`; `|error|` sin él) sobre `['pct', 'regional']` con `LeastSquaresEstimator` (pesos 1) en las filas de `load_average_errors()` del horizonte dado; `None` con menos de `MIN_ROWS` = 12 filas (aviso).
- Produces: `AverageErrorEstimator` (en `computer.py`): envuelve el `LeastSquaresEstimator` con `predict(p: np.ndarray) -> np.ndarray` que acepta la matriz de tres columnas `[pct, regional, weeks]` que hoy recibe `v2err` y usa las dos primeras; atributos `n_rows`, `horizon`, `net`, `events` (fechas que lo ajustan).
- Produces: `Simulator(..., error_model: str = 'polls')`: con `'average'`, `fit_error_estimator` devuelve `self.computer.get_average_error_estimator(horizon=self.AVERAGE_ERROR_HORIZON)` (`AVERAGE_ERROR_HORIZON = 6`) y, si es `None`, cae a `'polls'` con `warnings.warn`; `self.error_model_used` ∈ `{'polls', 'average'}`; `error_events()` devuelve las elecciones del estimador en los dos modos; `meta`/`diagnostics` del run y `meta.csv` del backtest llevan `error_model` (el usado).
- Produces: `run_case(..., error_model='polls')`, `run_backtest`, `backtest_scope` (en `run.json`), job `backtest` y `--error-model polls|average` en el script.

- [ ] **Step 1: Tests**

```python
def test_average_error_estimator_fits_the_net_error_on_level_and_regional(monkeypatch):
    c = Computer.__new__(Computer)
    rows = pd.DataFrame({'event_date': ['2019-04-28'] * 6 + ['2019-11-10'] * 6, 'horizon': 6, 'party': list('ABCDEF') * 2,
                         'level': [30, 20, 10, 5, 3, 2] * 2, 'regional': [0, 0, 0, 0, 1, 1] * 2,
                         'error': [2.5, 2.0, 1.0, 0.6, 0.4, 0.3] * 2, 'err': [0.5] * 12, 'main': True})
    monkeypatch.setattr(c, 'load_average_errors', lambda: rows)
    est = c.get_average_error_estimator(horizon=6)
    assert est.n_rows == 12 and est.horizon == 6 and est.net
    assert est.predict(np.array([[30., 0., 1.]]))[0] == pytest.approx(np.sqrt(2.5 ** 2 - 0.5 ** 2), abs=0.3)
    assert est.predict(np.array([[30., 0., 1.]]))[0] > est.predict(np.array([[5., 0., 1.]]))[0]

def test_average_error_estimator_needs_twelve_rows(monkeypatch):
    c = Computer.__new__(Computer)
    monkeypatch.setattr(c, 'load_average_errors', lambda: pd.DataFrame(columns=['event_date', 'horizon', 'party', 'level', 'regional', 'error', 'err', 'main']))
    with pytest.warns(UserWarning, match='average_errors'):
        assert c.get_average_error_estimator() is None

def test_simulator_error_model_average_falls_back_to_polls_without_rows(fake_simulator):
    sim = fake_simulator(error_model='average', average_estimator=None)
    assert sim.error_model_used == 'polls' and sim.meta()['diagnostics']['error_model'] == 'polls'

def test_backtest_job_passes_error_model_to_the_runner(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), error_model='average', verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['error_model'] == 'average'
    assert json.load(open(tmp_path / 'run.json'))['error_model'] == 'average'
```

(`fake_simulator` construye un `Simulator` con `__new__` y los atributos mínimos, como `tests/fakes.py`.)

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_computer_unit.py tests/test_simulator_unit.py tests/test_jobs_backtest.py -v -k "average_error or error_model"`
Expected: FAIL

- [ ] **Step 3: Implementar el estimador, `error_model` en el Simulator y su enhebrado en el backtest, el job y el script**

- [ ] **Step 4: Ejecutar la suite y la guarda de regresión**

Run: `pytest -q`; con base, `pytest tests/integration/test_backtest.py -q` con `error_model='polls'` reproduce el caso.
Expected: PASS

- [ ] **Step 5: Backtest completo con `error_model='average'` (con base, ≈ 15 min `es` + autonómicas) y tabla antes / después en el método**

Run: `python backtest/run_backtest.py --error-model average --out backtest/results_avg` y `--scopes all`
Expected: coberturas, PIT por población (de `calibration.csv` con `summarize_calibration`), log-score y CRPS frente a `backtest/results/`; decisión según el criterio 17 anotada en `metodo-15-calibracion.md`. Esperado: `e` baja (el error del promedio es menor que el de un sondeo suelto) y a 6-14 días las bandas se estrechan sin perder cobertura; a 30+ días faltará anchura hasta Task 4.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/computer.py mtpy/lib/simulator.py mtpy/lib/backtest.py mtpy/jobs/Backtest.py mtpy/pipelines/backtest/Backtest.py backtest/run_backtest.py tests docs/analisis-2026-09/metodo-15-calibracion.md
git commit -m "simulator: error_model='average' (error terminal del propio promedio, neto de s) y su backtest"
```

---

### Task 4: El crecimiento con el horizonte: `drift_scale`

**Files:**
- Modify: `mtpy/lib/computer.py:60-200` (`DriftEstimator`), `:2457-2500` (`get_drift_estimator`)
- Modify: `mtpy/lib/simulator.py` (constructor `drift_scale`, `v2drift`, `meta`), `mtpy/lib/backtest.py` (`run_case` / `meta.csv` / `run.json`), job y script (`--drift-scale auto|1.0|...`)
- Test: `tests/test_computer_unit.py`, `tests/test_simulator_unit.py`

**Interfaces:**
- Produces: `DriftEstimator(..., scale: float = 1.)`; `var` y `sigma` multiplican por `scale`; atributo `scale_rows` (filas usadas, 0 si no se ajustó).
- Produces: `DriftEstimator.fit_scale(errors: pd.DataFrame, k: float, d_min: int = 30, d_ref: int = 6, bounds: tuple = (1., 4.)) -> tuple[float, int]`: con la tabla de `average_errors` (filas `event_date`, `party`, `horizon`, `level`, `error`, `err`), para cada `(event_date, party)` con fila en `d_ref` y en un `d ≥ d_min`: `excess = (error_d² − err_d²) − (error_ref² − err_ref²)` y `expected = k · (d − d_ref) · level²` (nivel en puntos / 100 si `k` está en esa escala: la misma convención que `var`); `scale = Σ excess / Σ expected` recortado a `bounds`; devuelve `(scale, n_rows)`; `(1., 0)` sin filas, con aviso; con el recorte, aviso que dice el valor sin recortar.
- Produces: `Computer.get_drift_estimator(..., scale: Optional[str | float] = None)`: `'auto'` ajusta la escala con `load_average_errors()` (las elecciones anteriores a las del Computer, por construcción); un número la fija; `None` deja 1.
- Produces: `Simulator(..., drift_scale: Optional[str | float] = None)` → `get_drift_estimator(scale=drift_scale)`; `meta`/`diagnostics` y `meta.csv` llevan `drift_scale` (el valor usado) y `drift_scale_rows`.

- [ ] **Step 1: Tests**

```python
def test_fit_scale_recovers_a_known_multiplier():
    k, rows = 4e-5, []
    for ev in ['2019-04-28', '2019-11-10', '2023-07-23']:
        for party, level in [('A', 30.), ('B', 20.), ('C', 10.)]:
            rows.append({'event_date': ev, 'party': party, 'horizon': 6, 'level': level, 'error': 1.0, 'err': 0.})
            for d in (30, 60, 90):
                extra = 2.0 * k * (d - 6) * level ** 2   # varianza extra en puntos² con escala 2 (`var` = k · d · level², nivel en puntos)
                rows.append({'event_date': ev, 'party': party, 'horizon': d, 'level': level, 'error': np.sqrt(1.0 + extra), 'err': 0.})
    scale, n = DriftEstimator.fit_scale(pd.DataFrame(rows), k=k)
    assert scale == pytest.approx(2.0, rel=1e-6) and n == 27

def test_fit_scale_without_rows_or_out_of_bounds_falls_back_with_a_warning():
    empty = pd.DataFrame(columns=['event_date', 'party', 'horizon', 'level', 'error', 'err'])
    with pytest.warns(UserWarning):
        assert DriftEstimator.fit_scale(empty, k=4e-5) == (1., 0)
    rows = pd.DataFrame({'event_date': ['x'] * 2, 'party': ['A'] * 2, 'horizon': [6, 90], 'level': [30., 30.], 'error': [1., 1.], 'err': [0., 0.]})
    with pytest.warns(UserWarning, match='recort'):
        scale, _ = DriftEstimator.fit_scale(rows, k=4e-5)   # sin exceso: 0, recortado a 1
    assert scale == 1.

def test_drift_var_is_multiplied_by_the_scale():
    base = DriftEstimator(k=4e-5)
    scaled = DriftEstimator(k=4e-5, scale=2.)
    assert scaled.var(90, 30.) == pytest.approx(2. * base.var(90, 30.))
```

(Las unidades de `level` en `var` son las del código actual: comprobar `DriftEstimator.var` y usar la misma convención en `fit_scale`; el test fija la relación, no la unidad.)

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_computer_unit.py -v -k "fit_scale or drift_var"`
Expected: FAIL

- [ ] **Step 3: Implementar `scale`, `fit_scale`, `get_drift_estimator(scale=...)`, `drift_scale` en el Simulator y su enhebrado en backtest, job y script**

- [ ] **Step 4: Suite y guarda de regresión**

Run: `pytest -q`; con base, `tests/integration/test_backtest.py` con `drift_scale=None`.
Expected: PASS

- [ ] **Step 5: Backtest completo con `error_model='average'` y `drift_scale='auto'`; tabla antes / después (coberturas y PIT por grupo de horizonte y población, CRPS `_h`, log-score `_h`) en el método; decisión por el criterio 17**

Run: `python backtest/run_backtest.py --error-model average --drift-scale auto --out backtest/results_avg_scale` (y `--scopes all`)

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/computer.py mtpy/lib/simulator.py mtpy/lib/backtest.py mtpy/jobs/Backtest.py mtpy/pipelines/backtest/Backtest.py backtest/run_backtest.py tests docs/analisis-2026-09/metodo-15-calibracion.md
git commit -m "simulator: drift_scale, el crecimiento del error del promedio con el horizonte medido en average_errors"
```

---

### Task 5: Las colas: `tail_dof`

**Files:**
- Modify: `mtpy/lib/simulator.py:99-200` (constructor), `:1100-1130` (`_fan`), `:2035` (`build_frame`), `:1843-1870` (`draw_correlated_t`, sin cambio de firma), `meta`
- Modify: `mtpy/lib/backtest.py`, `mtpy/jobs/Backtest.py`, `mtpy/pipelines/backtest/Backtest.py`, `backtest/run_backtest.py` (`--tail-dof`)
- Test: `tests/test_simulator_unit.py`, `tests/test_jobs_backtest.py`

**Interfaces:**
- Produces: `Simulator(..., tail_dof: Optional[int] = None)`: con un entero ≥ 2, los grados de libertad de la t del sorteo (`build_frame`) y del abanico (`_fan`) son ese valor para todos los partidos; con `None`, `nobs − 2` (mínimo 3) como hoy; `ValueError` con 0, negativo o no entero. `meta`/`diagnostics` y `meta.csv` llevan `tail_dof`.
- Produces: `run_case(..., tail_dof=None)` y el resto del enhebrado; `--tail-dof` en el script.

- [ ] **Step 1: Tests**

```python
def test_tail_dof_fixes_the_degrees_of_freedom_of_the_draw_and_the_fan(fake_simulator):
    sim = fake_simulator(tail_dof=5)
    frame = sim.build_frame()
    assert (frame.loc[frame['pct'] > 0, 'dof'] == 5).all()          # `build_frame` guarda `dof` en el frame (columna nueva de `cols_frame`)
    assert (sim._fan([0])['dof'] == 5).all()
    assert fake_simulator(tail_dof=None).build_frame()['dof'].min() >= 3

@pytest.mark.parametrize('bad', [0, -1, 2.5, 'x'])
def test_tail_dof_rejects_bad_values(fake_simulator, bad):
    with pytest.raises(ValueError, match='tail_dof'):
        fake_simulator(tail_dof=bad)
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_simulator_unit.py -v -k tail_dof`
Expected: FAIL

- [ ] **Step 3: Implementar `tail_dof` (columna `dof` en `cols_frame`), su validación y el enhebrado**

- [ ] **Step 4: Suite y guarda de regresión**

Run: `pytest -q`; con base, `tests/integration/test_backtest.py` con `tail_dof=None`.
Expected: PASS

- [ ] **Step 5: Backtests con `tail_dof` ∈ {3, 5, 8, 15, None} sobre la configuración de Task 4; tabla de log-score (voto y mayorías), CRPS y PIT por población; elección dejando fuera cada elección por turno (la tabla 5 × 5 de candidato × elección excluida, con el candidato que gana en cada una); con diferencias dentro del ruido, 5, y se dice (spec 14)**

Run: `for d in 3 5 8 15; do python backtest/run_backtest.py --error-model average --drift-scale auto --tail-dof $d --out backtest/results_dof$d; done`

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/simulator.py mtpy/lib/backtest.py mtpy/jobs/Backtest.py mtpy/pipelines/backtest/Backtest.py backtest/run_backtest.py tests docs/analisis-2026-09/metodo-15-calibracion.md
git commit -m "simulator: tail_dof, grados de libertad fijos de la t del sorteo, y su comparación en el backtest"
```

---

### Task 6: La correlación entre partidos, reevaluada

**Files:**
- Modify: `docs/analisis-2026-09/metodo-15-calibracion.md`

- [ ] **Step 1: Backtest con `--composition auto` sobre la configuración de Task 5 (escala y colas nuevas), `es` y autonómicas**

Run: `python backtest/run_backtest.py --error-model average --drift-scale auto --tail-dof <elegido> --composition auto --out backtest/results_comp2`
Expected: tabla frente a la misma configuración sin composición: CRPS de escaños, Brier y log-score de mayorías a 6-14 días (el criterio de M7: entra si no empeora ahí), coberturas de escaños y PIT de escaños por grupo de horizonte.

- [ ] **Step 2: Decisión anotada en el método (activar `composition='auto'` por defecto o dejarlo disponible), con fecha**

---

### Task 7: Decisión sobre los valores por defecto, publicación y la tarjeta «Fiabilidad»

**Files:**
- Modify (sólo si el criterio 17 lo respalda, en commit aparte): `mtpy/lib/simulator.py` (defaults `error_model`, `drift_scale`, `tail_dof`, `composition`), `backtest/results/` reejecutado, constantes de `tests/integration/test_model.py`
- Modify: `mtpy/lib/bundle.py:70-72` (`BACKTEST_PARTS` + `'calibration'`, esquema `backtest-calibration`), `mtpy/lib/publish/backtest.py:15-20` (`BACKTEST_FILES['calibration'] = 'calibration.csv'`; `export_backtest_meta` añade `error_model`, `drift_scale`, `tail_dof` de `run.json` a `params`), `docs/web/contrato.md` (enmienda)
- Modify: `mtpy/lib/pages/backtest.py` (`reliability_context`), `web/templates/metodo_backtest.html` (tarjeta `fiabilidad` tras `coverage`), `web/dist/js/pages/metodo.js` (dibujo), `web/dist/js/charts/bars.js` o `grouped.js` (barras con línea de referencia; si ninguno la admite, opción `references` como en `line.js`)
- Test: `tests/test_bundle_unit.py`, `tests/test_publish*.py` (backtest), `tests/test_pages.py`, `tests/test_js_charts.py`; fixture `tests/fixtures/bundle/backtest-calibration.json`

**Interfaces:**
- Produces: parte `calibration` del backtest de cada ámbito: `rows` de `calibration.csv` (`event_date`, `horizon`, `kind`, `mode`, `party`, `region_id`, `main`, `official`, `model`, `pit`), con su CSV gemelo; `meta.params` con los tres parámetros nuevos.
- Produces: `pages.backtest.reliability_context(scopes: list) -> dict`: lee la parte `calibration` y la parte `blocks` de todos los ámbitos con backtest publicado (como `model_record` lee `event_part`), asigna `population` (`'generales'` si `scope == 'es'`, `'autonomicas'` si no; `prov_*` → `'provincias'`), llama a `summarize_calibration` y `majority_reliability`, y devuelve `{'pit': {population: {group: [observed × 10]}}, 'majority': [{'p_mean', 'observed', 'n'}], 'n_events': {population: n}, 'mode': 'h'}` (modo Elección; el nowcast sólo a 6-14). `backtest_context` lo añade a `context['reliability']` e `initial['reliability']`.
- Produces: tarjeta `fiabilidad` en `/metodo/backtest`: título «Fiabilidad de las bandas», entradilla «Si las bandas dicen la verdad, el resultado cae en cada décima de la banda una de cada diez veces; las barras muestran cuántas veces cayó en cada una.», un gráfico por población (tres paneles o un selector) con las diez barras por grupo de horizonte y la línea de 0,10, y debajo el diagrama de fiabilidad de las mayorías (probabilidad dada frente a frecuencia observada, con la diagonal). «Cómo leerlo» en dos párrafos: (1) barras altas en los extremos, bandas demasiado estrechas; altas en el centro, demasiado anchas; planas, calibradas; (2) qué no dice: no mide si el centro acierta, sólo si la anchura es la que debe; y las provincias informan del reparto territorial, no del voto nacional. Sin tarjeta cuando ningún ámbito trae la parte (bundles anteriores).

- [ ] **Step 1: Tests de bundle y publicación (la parte nueva se escribe y se lee; `meta.params` lleva los parámetros), de página (tarjeta con el fixture; oculta sin la parte; `assert_public_wording`) y del gráfico (`test_js_charts.py`: la opción de barras con referencia construye diez barras y una línea)**

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_bundle_unit.py tests/test_pages.py tests/test_js_charts.py -q -k "calibration or fiabilidad or reliability"`
Expected: FAIL

- [ ] **Step 3: Implementar la parte, el contexto, la plantilla y el gráfico**

- [ ] **Step 4: Suite completa y Chrome sin cabeza sobre el paquete de pruebas (como en las specs anteriores): tarjeta renderizada sin errores de consola**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Decisión sobre los valores por defecto (criterio 17, antes del 20-11): si entra, commit aparte con los defaults, `backtest/results/` reejecutado con la configuración nueva y `compute '{"what":["compute"]}'` en producción para rellenar `average_errors` (la web lee el estimador de la base en vivo); si no, los parámetros quedan documentados como opciones y la fecha de revisión es tras el 29-N**

- [ ] **Step 6: Documentación y commit**

`docs/analisis-2026-09/metodo-15-calibracion.md` (problema, método, las tablas de las Tasks 3-6, decisión con fecha, salvedades: pocas generales, estimador sin autonómicas, `s` neto, unidades), `backtest/README.md` (`calibration.csv`, PIT, poblaciones, parámetros nuevos), `deploy/README.md` (parámetros del job `backtest` y paso `average_errors` del `compute`), `docs/web/contrato.md` (parte `calibration`, `meta.params`), `README.md` (una línea en `compute`).

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish/backtest.py mtpy/lib/pages/backtest.py web/templates/metodo_backtest.html web/dist/js tests docs backtest/README.md deploy/README.md README.md
git commit -m "web: tarjeta Fiabilidad en /metodo/backtest (PIT por población y mayorías) y parte calibration del backtest; método 15"
```
