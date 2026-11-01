# Afinar el promedio con los sondeos futuros (parte A) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un arnés que ajusta el promedio en cada corte de cada ciclo con sondeos, predice los sondeos de los k días siguientes, puntúa cada configuración del promedio y busca la mejor por coordenadas, con ciclos de comprobación reservados; como job `tune`, con sus tablas y su documento de método.

**Architecture:** Un módulo `mtpy/lib/tuning.py` gemelo de `backtest.py`: funciones puras (cortes, predicción con efecto de casa, puntuación, diagnósticos, descenso por coordenadas) y orquestación por ámbito (`run_cut`, `run_tuning`, `tuning_scope`). El Forecaster gana tres cosas pequeñas: pesos recalculados en memoria (`weight_overrides`), un tope del ancho de banda de frontera (`bw_cap`) y la lectura del promedio en una fecha con el efecto de casa de una casa sumado. Un job `tune` y su pipeline, calcados del `backtest`, escriben `backtest/tuning/{scope}/`.

**Tech Stack:** Python 3.11, pandas, numpy, pytest; la infraestructura de jobs de MTpy (`SequenceJob`, `ScopePipeline`); PostgreSQL sólo en los tests de integración.

**Spec:** `docs/superpowers/specs/2026-11-01-sondeos-futuros-y-calibracion-design.md` (parte A, decisiones 1-10, 18, 19).

## Global Constraints

- Fuera de muestra siempre: un corte sólo ve sondeos publicados hasta el corte (`drange`); la búsqueda usa las generales hasta 2019-11 y las autonómicas anteriores a 2020; comprobación en 2023, el ciclo 2026 y las autonómicas desde 2020 (spec A7).
- Ningún cambio en los valores por defecto del modelo vivo en este plan: toda configuración entra por parámetro; los defaults sólo cambian con el criterio A8, en un commit aparte, antes del 20-11 o después del 29-N.
- Cortes cada 7 días en los 365 días anteriores a la elección (hasta hoy en los ciclos en curso); un corte con menos de 5 sondeos útiles en los 90 días anteriores se descarta (A2).
- Horizontes de prueba k = 7 y k = 28 días, puntuados por separado (A4).
- Rejilla de A5 (sin `ol_max` ni `prior_events`; `tau` y `herd_prior` en su lugar); descenso por coordenadas, elección por RMSE a k = 7, guardas: k = 28 no empeora más del 2 % y la rugosidad no pasa de 1,5 veces la de la configuración actual sobre los mismos cortes (A4, A6).
- Dos fases (A6): fase 1, el suavizado, con efectos de casa, muestra efectiva y rebaño desactivados, dos vueltas, cortes cada 7 días; fase 2, las contracciones, con todo activado, una vuelta, cortes cada 14 días y sólo los ciclos desde 2015, en ECS por ciclos; comprobación final de `best` frente a `default` con todo activado.
- Sondeos objetivo (A3): los que entran en el promedio y además con `weight_over` > 0 en la base (sin oleadas solapadas de trackings).
- Con `weight_overrides=None`, `bw_cap=None` y los parámetros actuales, el Forecaster y el backtest reproducen `backtest/results/` exactamente (spec 19).
- Tests unitarios sin base de datos; los de integración se saltan sin ella (`tests/integration/`).
- Redacción pública con las reglas de Luis (`assert_public_wording`) en lo que llegue a la web; este plan no toca la web.
- Commits sin atribución de modelo en el mensaje; mensajes en español como los del repositorio (`tuning: ...`, `forecaster: ...`).

## Review Focus

- Un corte sin sondeos futuros en `(cut, cut + k]` (ciclo que termina, veda): `score_polls` devuelve una fila con `n_polls` = 0 y métricas NaN, nunca una excepción (Task 3).
- Un sondeo futuro de una casa sin efecto de casa estimado (menos de `min_polls` en el corte): la predicción suma 0, no NaN (Task 2).
- Un partido principal del ciclo sin valor en un sondeo futuro (la casa no lo pregunta): esa celda se omite y las demás del sondeo cuentan (Task 3).
- Un parámetro de la rejilla con un valor inválido (`bw_cap` ≤ 1, `wspan` ≤ 0): `expand_grid` lo rechaza con `ValueError` antes de ajustar nada (Task 5).
- Un ámbito sin ciclos con sondeos suficientes: `tuning_scope` lanza `NothingToTune` y el pipeline lo deja `skipped` sin escribir (Task 7).

---

### Task 1: Pesos recalculados en memoria y tope del ancho de frontera en el Forecaster

**Files:**
- Modify: `mtpy/lib/forecaster.py:99-200` (constructor), `mtpy/lib/forecaster.py:440-500` (`build_series`), `mtpy/lib/forecaster.py:268-300` (`set_reg_params`)
- Modify: `mtpy/lib/computer.py:850-1000` (`compute_weights`: extraer la lógica de muestra y sobrepublicación), `mtpy/lib/computer.py:1957-2010` (`poll_overweights`)
- Modify: `mtpy/core/utils/stat.py:465-520` (`Kernel.get_bw_adaptive`)
- Test: `tests/test_forecaster_unit.py`, `tests/test_computer_unit.py`, `tests/test_stat_unit.py` (crear si no existe)

**Interfaces:**
- Produces: `Computer.reweight(polls: pd.DataFrame, wspan: int, ol_max: int, ol_dev: int = 3, drop_mtypes: Optional[list[str]] = None) -> pd.DataFrame` (método de clase, sin base: recibe las filas de `get_poll_series` con `sample_size`, `mtype`, `ctype`, `start_date`, `end_date`, `event_date`, `pollster`, `pollster_id`, `sponsor_id`, `date`; devuelve un frame con el mismo índice y las columnas `proc_sample`, `weight_sample`, `weight_over`).
- Produces: `Forecaster(..., weight_overrides: Optional[pd.DataFrame] = None)`: frame indexado por `(date, pollster_id, sponsor_id)` con alguna de las columnas `weight_over`, `weight_sample`, `weight_rating`; `build_series` sustituye esas columnas en las filas que coinciden antes de calcular `weight`.
- Produces: `reg_params['bw_cap']: Optional[float]` (por defecto `None`): con un número, `Kernel.get_bw_adaptive` recorta el ancho adaptativo de cada punto a `bw_cap · bw` (el piloto fijo).
- Produces: `disp_params['herd_prior']: float` (por defecto `disp_params['n_prior']`, 5): la contracción `n / (n + herd_prior)` del ratio de rebaño (`Forecaster.shrink_ratio` en `measure_herding` / `dispersion_factor`), separada de la de la muestra efectiva, que sigue con `n_prior`. Con los defaults nada cambia.

- [ ] **Step 1: Test de `Computer.reweight` sobre sondeos sintéticos**

```python
def test_reweight_recomputes_sample_and_overlap_weights():
    polls = _polls_frame()  # 4 sondeos de 2 casas, muestras 1000, 4000, NaN, 9000; dos de la misma casa con 2 días entre fechas
    w = Computer.reweight(polls, wspan=4, ol_max=5000, ol_dev=3)
    assert list(w.columns) == ['proc_sample', 'weight_sample', 'weight_over']
    assert w['proc_sample'].max() <= 5000                      # tope
    assert w.loc[polls['sample_size'].isna(), 'proc_sample'].iloc[0] == 1000  # mediana de la casa
    assert w['weight_sample'].between(0, 3).all()
    assert w.loc[polls.pollster_id == 1, 'weight_over'].tolist() == [0.5, 0.5]  # dos en 4 días: 1/2 cada uno
    w2 = Computer.reweight(polls, wspan=1, ol_max=5000)
    assert w2.loc[polls.pollster_id == 1, 'weight_over'].tolist() == [1.0, 1.0]
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_computer_unit.py::test_reweight_recomputes_sample_and_overlap_weights -v`
Expected: FAIL con `AttributeError: type object 'Computer' has no attribute 'reweight'`

- [ ] **Step 3: Extraer de `compute_weights` la lógica de `proc_sample`, `weight_sample` y `weight_over` a `Computer.reweight` (classmethod) y hacer que `compute_weights` la llame**

`poll_overweights` pasa a leer `wspan` de un argumento (`poll_overweights(x, wspan)`), y el método de instancia lo envuelve con `self.wspan`. `reweight` calcula las medianas por `(event_date, pollster)`, por `pollster` y global igual que hoy, winsoriza con `Stat(..., outliers='group', n_dev=ol_dev)` y agrupa las sobrepublicaciones por `(event_date, pollster_id, wtype)` con `apply_agg_func`.

- [ ] **Step 4: Ejecutar el test y la suite del Computer**

Run: `pytest tests/test_computer_unit.py -v -k "reweight or weights"`
Expected: PASS

- [ ] **Step 5: Test de `weight_overrides` en el Forecaster (función pura, sin base)**

```python
def test_apply_weight_overrides_replaces_only_the_given_cells_of_polls():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2026-01-08'), 1, 0), (pd.Timestamp('2026-01-09'), 2, 0), (pd.Timestamp('2026-02-01'), np.nan, np.nan)],
                                    names=['date', 'pollster_id', 'sponsor_id'])
    series = pd.DataFrame({'pollster': ['A', 'B', None], 'weight_over': [1., 1., np.nan], 'weight_sample': [1.2, 0.8, np.nan], 'weight_rating': [1., 1., np.nan]}, index=idx)
    over = pd.DataFrame({'weight_over': [0.25]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp('2026-01-08'), 1, 0)], names=idx.names))
    out = Forecaster.apply_weight_overrides(series, over)
    assert out.loc[(pd.Timestamp('2026-01-08'), 1, 0), 'weight_over'] == 0.25
    assert out.loc[(pd.Timestamp('2026-01-09'), 2, 0), 'weight_over'] == 1.0 and out['weight_sample'].tolist()[:2] == [1.2, 0.8]
    assert series.loc[(pd.Timestamp('2026-01-08'), 1, 0), 'weight_over'] == 1.0   # copia, no in situ
```

- [ ] **Step 6: Ejecutar y ver que falla**

Run: `pytest tests/test_forecaster_unit.py::test_apply_weight_overrides_replaces_only_the_given_cells_of_polls -v`
Expected: FAIL con `AttributeError`

- [ ] **Step 7: Implementar `Forecaster.apply_weight_overrides(series: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame` (estática) y `weight_overrides` en `Forecaster.__init__` y `build_series`**

En `build_series`, tras concatenar sondeos y resultados e indexar por `(date, pollster_id, sponsor_id)`, y antes de `series['weight'] = ...`: si `self.weight_overrides is not None`, `series = self.apply_weight_overrides(series, self.weight_overrides)` (sólo las columnas presentes en `overrides` entre `weight_over`, `weight_sample`, `weight_rating`, sólo en filas con `pollster`). El `Simulator` enhebra el argumento (`Simulator(..., weight_overrides=None)` → Forecaster), como hace con `weight_ratings`.

- [ ] **Step 8: Test del tope de frontera del kernel**

```python
def test_adaptive_bandwidth_cap_limits_the_boundary():
    x = np.concatenate([np.linspace(0, 100, 60), [101, 102]])   # densidad que cae al final
    k = Kernel(x, bw=10., bw_type='adaptive', n_iter=3)
    free = k.get_bw_adaptive(p=np.array([50., 102.]))
    capped = Kernel(x, bw=10., bw_type='adaptive', n_iter=3, bw_cap=1.5).get_bw_adaptive(p=np.array([50., 102.]))
    assert free[1] > 15.                       # la frontera se ensancha sin tope
    assert capped[1] == pytest.approx(15.)      # con tope: 1,5 × piloto
    assert capped[0] == pytest.approx(free[0])  # el interior no cambia
```

- [ ] **Step 9: Ejecutar y ver que falla**

Run: `pytest tests/test_stat_unit.py::test_adaptive_bandwidth_cap_limits_the_boundary -v`
Expected: FAIL con `TypeError: ... unexpected keyword argument 'bw_cap'`

- [ ] **Step 10: Implementar `bw_cap` en `Kernel` y enhebrarlo desde `LocalKernelEstimator` y `Forecaster.set_reg_params`**

`Kernel.__init__(..., bw_cap: Optional[float] = None)`; en `get_bw_adaptive`, tras cada iteración, `kbw = np.minimum(kbw, bw_cap * bw)` cuando `bw_cap` no es `None`. `set_reg_params` añade `'bw_cap': reg_params.get('bw_cap', None)` y `LocalKernelEstimator` lo pasa al `Kernel`. Con `None` nada cambia.

- [ ] **Step 11: Test y cambio de `herd_prior`**

```python
def test_herd_prior_defaults_to_n_prior_and_shrinks_the_herding_ratio_alone():
    fc = Forecaster.__new__(Forecaster)
    assert fc.set_disp_params({'n_prior': 7.})['herd_prior'] == 7.
    assert fc.set_disp_params({'n_prior': 5., 'herd_prior': 2.})['herd_prior'] == 2.
    # ratio 0,5 medido con n = 10: con herd_prior 2 se contrae a 1 − (1 − 0,5) · 10 / 12; con n_prior 5 (muestra efectiva) no cambia
    assert Forecaster.shrink_ratio(0.5, 10, 2.) == pytest.approx(1 - 0.5 * 10 / 12)
```

(Si `shrink_ratio` tiene otra firma, el test usa la real; lo que se prueba es que el rebaño lee `herd_prior` y la dispersión `n_prior`.)

- [ ] **Step 12: Ejecutar los tests y la guarda de regresión**

Run: `pytest tests/test_stat_unit.py tests/test_forecaster_unit.py tests/test_computer_unit.py -q`
Expected: PASS. Con base: `pytest tests/integration/test_model.py -q` reproduce las constantes de referencia (el `weight` de cada sondeo no cambia con `weight_overrides=None`, `bw_cap=None` y `herd_prior` = `n_prior`).

- [ ] **Step 13: Commit**

```bash
git add mtpy/lib/forecaster.py mtpy/lib/computer.py mtpy/lib/simulator.py mtpy/core/utils/stat.py tests/test_forecaster_unit.py tests/test_computer_unit.py tests/test_stat_unit.py
git commit -m "forecaster: pesos recalculados en memoria (weight_overrides), tope del ancho de frontera (bw_cap), herd_prior y Computer.reweight sin base"
```

---

### Task 2: Leer el promedio de un corte en la fecha de un sondeo, con su efecto de casa

**Files:**
- Modify: `mtpy/lib/forecaster.py:700-760` (junto a `get_forecast`)
- Create: `mtpy/lib/tuning.py` (cabecera del módulo y `predict_polls`)
- Test: `tests/test_tuning_unit.py`

**Interfaces:**
- Produces: `Forecaster.house_effect_of(pollster_id: int, name: str) -> float`: el `effect` de `house_effects` para esa casa y serie, 0 si no hay efectos ajustados o la casa no está.
- Produces: `tuning.predict_polls(fc: Forecaster, polls: pd.DataFrame, names: list[str]) -> pd.DataFrame`: `polls` son los sondeos futuros (filas con `date`, `pollster_id`, `sponsor_id`, `pollster`, `sample_size` y una columna por partido, cifras publicadas); devuelve una fila por sondeo y partido de `names` con valor: `date`, `pollster_id`, `pollster`, `sample_size`, `party`, `published`, `predicted` (promedio en `min(date, date_fit_last + max_fc)` más el efecto de casa), `residual` (= published − predicted).

- [ ] **Step 1: Tests de la predicción**

```python
def test_predict_polls_reads_the_average_at_the_poll_date_and_adds_the_house_effect():
    fc = _fake_fitted_forecaster(effects={(1, 'PP'): 2.0})  # promedio PP 30 plano hasta 2026-03-10 + 10 días; PSOE 20
    polls = pd.DataFrame({'date': [pd.Timestamp('2026-03-12')], 'pollster_id': [1], 'sponsor_id': [0], 'pollster': ['A'],
                          'sample_size': [1000], 'PP': [33.], 'PSOE': [19.]})
    out = predict_polls(fc, polls, ['PP', 'PSOE'])
    assert out.loc[out.party == 'PP', 'predicted'].iloc[0] == pytest.approx(32.)   # 30 + efecto 2
    assert out.loc[out.party == 'PP', 'residual'].iloc[0] == pytest.approx(1.)
    assert out.loc[out.party == 'PSOE', 'predicted'].iloc[0] == pytest.approx(20.)  # sin efecto: suma 0

def test_predict_polls_holds_the_last_value_beyond_the_prolongation_and_skips_missing_parties():
    fc = _fake_fitted_forecaster(effects={})
    polls = pd.DataFrame({'date': [pd.Timestamp('2026-04-30')], 'pollster_id': [2], 'sponsor_id': [0], 'pollster': ['B'],
                          'sample_size': [np.nan], 'PP': [31.], 'PSOE': [np.nan]})
    out = predict_polls(fc, polls, ['PP', 'PSOE'])
    assert out['party'].tolist() == ['PP']                      # PSOE sin cifra: omitido
    assert out['predicted'].iloc[0] == pytest.approx(30.)       # último valor del promedio
```

`_fake_fitted_forecaster` construye un `Forecaster.__new__` con `forecast` (índice diario, columnas `PP`, `PSOE`, `-`), `date_fit_last`, `max_fc` = 10, `house_effects` (frame indexado por `(pollster_id, name)` con `effect`) y `names`, sin base, como `tests/fakes.synthetic_forecaster`.

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_tuning_unit.py -v -k predict_polls`
Expected: FAIL con `ModuleNotFoundError: No module named 'mtpy.lib.tuning'`

- [ ] **Step 3: Implementar `Forecaster.house_effect_of` y `tuning.predict_polls`**

`predict_polls` lee `fc.forecast[name].ffill()` en `min(date, date_fit_last + max_fc)` (la recta prolongada existe hasta ahí; después, el último valor) y suma `fc.house_effect_of(pollster_id, name)`.

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `pytest tests/test_tuning_unit.py -v -k predict_polls`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py mtpy/lib/tuning.py tests/test_tuning_unit.py
git commit -m "tuning: predicción de los sondeos futuros de un corte con el efecto de casa sumado"
```

---

### Task 3: Puntuación y diagnósticos

**Files:**
- Modify: `mtpy/lib/tuning.py`
- Test: `tests/test_tuning_unit.py`

**Interfaces:**
- Produces: `tuning.score_polls(pred: pd.DataFrame, k: int) -> dict`: sobre la salida de `predict_polls` (ya filtrada al horizonte k): `{'k', 'n_polls', 'n_cells', 'rmse', 'mae', 'skill', 'rmse_by_party': dict}`; `skill = 1 − mse / floor` con `floor` = media de `p (1 − p) / n` en puntos² (`p` la cifra publicada / 100, `n` la muestra, `n` ausente → mediana de las presentes); NaN con `n_cells` = 0.
- Produces: `tuning.autocorrelation(pred: pd.DataFrame) -> float`: correlación de orden 1 de los residuos ordenados por fecha dentro de un mismo partido (NaN con menos de 3 residuos).
- Produces: `tuning.roughness(forecast: pd.DataFrame, names: list[str]) -> float`: media sobre `names` del cambio absoluto medio de un día al siguiente de la serie ajustada, en puntos por día (sobre los días con valor ajustado, sin el relleno). Se juzga relativa a la de la configuración actual (spec A4): la guarda vive en `coordinate_descent` (Task 5).

- [ ] **Step 1: Tests**

```python
def test_score_polls_rmse_mae_skill_and_by_party():
    pred = pd.DataFrame({'party': ['PP', 'PP', 'PSOE'], 'published': [32., 28., 20.], 'predicted': [30., 30., 20.],
                         'residual': [2., -2., 0.], 'sample_size': [1000., 1000., np.nan], 'date': pd.to_datetime(['2026-01-02'] * 3)})
    s = score_polls(pred, k=7)
    assert s['k'] == 7 and s['n_cells'] == 3 and s['n_polls'] == 1
    assert s['rmse'] == pytest.approx(np.sqrt(8 / 3)) and s['mae'] == pytest.approx(4 / 3)
    assert s['rmse_by_party']['PP'] == pytest.approx(2.) and s['rmse_by_party']['PSOE'] == 0.
    assert s['skill'] < 1

def test_score_polls_without_cells_is_nan_not_an_error():
    s = score_polls(pd.DataFrame(columns=['party', 'published', 'predicted', 'residual', 'sample_size', 'date']), k=28)
    assert s['n_cells'] == 0 and np.isnan(s['rmse']) and np.isnan(s['skill'])

def test_autocorrelation_and_roughness():
    pred = pd.DataFrame({'party': ['PP'] * 4, 'date': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04']),
                         'residual': [1., 1., -1., -1.]})
    assert autocorrelation(pred) > 0
    days = pd.date_range('2026-01-01', periods=5, freq='D')
    assert roughness(pd.DataFrame({'PP': [30., 30., 30., 30., 30.]}, index=days), ['PP']) == 0.
    assert roughness(pd.DataFrame({'PP': [30., 31., 30., 31., 30.]}, index=days), ['PP']) == pytest.approx(1.)
    assert roughness(pd.DataFrame({'PP': [np.nan, 30., 30.5, np.nan, np.nan]}, index=days), ['PP']) == pytest.approx(0.5)  # sin el relleno
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_tuning_unit.py -v -k "score_polls or autocorrelation"`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar `score_polls`, `autocorrelation` y `roughness`**

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `pytest tests/test_tuning_unit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/tuning.py tests/test_tuning_unit.py
git commit -m "tuning: puntuación de los sondeos futuros (RMSE, MAE, skill) y diagnósticos de retardo y rugosidad"
```

---

### Task 4: Cortes de un ciclo y un corte completo (`run_cut`)

**Files:**
- Modify: `mtpy/lib/tuning.py`
- Modify: `mtpy/lib/data.py:726-760` (`get_poll_series`: aceptar `drange=(a, b)` con `a` = 0 para «hasta b días antes», ver nota)
- Test: `tests/test_tuning_unit.py`, `tests/integration/test_tuning.py`

**Interfaces:**
- Produces: `tuning.cycle_cuts(event_date: str, poll_dates: pd.Series, cut_days: int = 7, window: int = 365, min_polls: int = 5, lookback: int = 90, today: Optional[str] = None) -> list[pd.Timestamp]`: fechas de corte desde `event_date − window` cada `cut_days` hasta `min(event_date − 1, today)`, descartando las que tienen menos de `min_polls` fechas de sondeo en `(cut − lookback, cut]`.
- Produces: `tuning.future_polls(scope: str, event_date: str, cut: pd.Timestamp, k: int, drop_mtypes: list[str]) -> pd.DataFrame`: los sondeos del ciclo con fecha en `(cut, cut + k]` que entran en el modelo (`computed`, sin `drop_mtypes`) **y con `weight_over` > 0 en la base** (sin las oleadas solapadas de los trackings, spec A3), cifras publicadas por partido (columnas de `get_poll_series` con `drange=(days(event − cut − k), days(event − cut) − 1)`).
- Produces: `tuning.run_cut(scope: str, event_date: str, cut: pd.Timestamp, config: dict, horizons: tuple[int, ...] = (7, 28), max_fc: Optional[int] = None, verbose: int = 0) -> dict`: ajusta `Forecaster(scope, event_date, drange=days(event − cut), reg_params=config['reg_params'], house_effects=config['house_effects'], he_params=config['he_params'], disp_params=config['disp_params'], weight_overrides=...)`, `fit_forecast(max_fc=config['max_fc'])`, y devuelve `{'meta': {...}, 'polls': DataFrame (predicciones de todos los k, columna 'k'), 'scores': [score por k], 'autocorr', 'roughness', 'seconds'}`. Un `config` es el dict que produce Task 5 (`expand_grid`); `config['weights']` = `{'wspan', 'ol_max'}` dispara `Computer.reweight` sobre los sondeos del corte y los pasa como `weight_overrides`.

Nota sobre `get_poll_series`: hoy `drange=(a, b)` con `a` = 0 cae en la rama `elif drange[1]` (`t.days <= b`), que es justo «hasta b días antes»; `future_polls` lo usa así y no hace falta tocar `data.py` salvo que la revisión del constructor encuentre lo contrario. `t.days` son los días entre el sondeo y la elección.

- [ ] **Step 1: Test de `cycle_cuts` (sin base)**

```python
def test_cycle_cuts_weekly_window_and_min_polls():
    dates = pd.Series(pd.to_datetime(['2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-06-15',
                                       '2023-07-01', '2023-07-10', '2023-07-15']))
    cuts = cycle_cuts('2023-07-23', dates, cut_days=7, window=365, min_polls=2, lookback=90)
    assert cuts[0] >= pd.Timestamp('2022-07-23') and cuts[-1] <= pd.Timestamp('2023-07-22')
    assert all((b - a).days == 7 for a, b in zip(cuts, cuts[1:]))
    assert pd.Timestamp('2022-10-01') not in cuts            # sin sondeos en los 90 días anteriores
    assert cuts[-1] in {pd.Timestamp('2023-07-22'), pd.Timestamp('2023-07-21'), pd.Timestamp('2023-07-20'),
                        pd.Timestamp('2023-07-19'), pd.Timestamp('2023-07-18'), pd.Timestamp('2023-07-17'), pd.Timestamp('2023-07-16')}

def test_cycle_cuts_of_a_running_cycle_stop_today():
    dates = pd.Series(pd.date_range('2026-01-01', '2026-10-30', freq='7D'))
    cuts = cycle_cuts('2026-11-29', dates, today='2026-11-01')
    assert cuts[-1] <= pd.Timestamp('2026-11-01')
```

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_tuning_unit.py -v -k cycle_cuts`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar `cycle_cuts`, `future_polls` y `run_cut`**

`run_cut` calcula `drange = (event − cut).days`; con `config['weights']`, obtiene los sondeos del corte con `get_poll_series(scope, [event_date], drange=drange, drop_mtypes=...)`, llama a `Computer.reweight` y pasa el resultado como `weight_overrides`. Los `names` son `bmaps['main']` del evento (`get_event_params`), y las predicciones se calculan una vez con el mayor k y se etiquetan por k según `date − cut`.

- [ ] **Step 4: Test de integración de un corte (con base; se salta sin ella)**

```python
@pytest.mark.integration
def test_run_cut_2023_30_days_predicts_the_next_week_within_poll_noise(db):
    out = run_cut('es', '2023-07-23', pd.Timestamp('2023-06-23'), config=default_config(), horizons=(7, 28))
    s7 = next(s for s in out['scores'] if s['k'] == 7)
    assert s7['n_polls'] >= 5 and 1.0 < s7['rmse'] < 4.0
    assert out['meta']['n_polls'] > 100 and out['meta']['drange'] == 30
```

- [ ] **Step 5: Ejecutar**

Run: `pytest tests/test_tuning_unit.py -q && pytest tests/integration/test_tuning.py -q`
Expected: PASS (el de integración `SKIPPED` sin base)

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/tuning.py tests/test_tuning_unit.py tests/integration/test_tuning.py
git commit -m "tuning: cortes semanales de un ciclo, sondeos futuros y run_cut con pesos recalculados"
```

---

### Task 5: Configuraciones, rejilla y descenso por coordenadas

**Files:**
- Modify: `mtpy/lib/tuning.py`
- Test: `tests/test_tuning_unit.py`

**Interfaces:**
- Produces: `tuning.default_config() -> dict`: `{'reg_params': {}, 'house_effects': True, 'he_params': {}, 'dispersion': True, 'herding': True, 'disp_params': {}, 'weights': None, 'max_fc': 10, 'label': 'default'}` (los valores por defecto del modelo vivo; `{}` deja a cada `set_*_params` sus defaults).
- Produces: `tuning.GRID: dict[str, list]` con las claves `bw` (`[('fixed', 'isj'), ('fixed', 30), ('fixed', 45), ('fixed', 60), ('fixed', 90), ('adaptive', 1), ('adaptive', 2), ('adaptive', 3)]`), `bw_cap` (`[None, 1.0, 1.5, 2.0]`), `poly_deg` (`[0, 1]`), `max_fc` (`[0, 3, 10]`), `wspan` (`[2, 4, 7, 14]`), `tau` (`[0.04, 0.08, 0.16]`), `he_n_iter` (`[1, 3]`), `herd_prior` (`[2., 5., 10.]`) — spec A5 tras la revisión (sin `ol_max` ni `prior_events`).
- Produces: `tuning.ORDER_SMOOTHING = ['bw', 'bw_cap', 'poly_deg', 'max_fc', 'wspan']` y `tuning.ORDER_SHRINKAGE = ['tau', 'he_n_iter', 'herd_prior']`.
- Produces: `tuning.apply_setting(config: dict, key: str, value) -> dict`: copia de `config` con el parámetro de la rejilla aplicado donde vive (`bw` → `reg_params['bw_type']` y `reg_params['bw']` o `reg_params['bw_kwargs']['n_iter']`; `bw_cap` → `reg_params['bw_cap']`; `poly_deg` → `reg_params`; `wspan` → `weights` (`{'wspan': v}`); `tau`/`he_n_iter` → `he_params` (`he_n_iter` como `n_iter`); `herd_prior` → `disp_params`; `max_fc`), y `label` = `'{key}={value}'`; `ValueError` con un valor fuera de la rejilla o inválido (`bw_cap` ≤ 1 salvo `None`, `wspan` ≤ 0).
- Produces: `tuning.coordinate_descent(evaluate: Callable[[dict], dict], start: dict, grid: dict, order: list[str], rounds: int = 2, tolerance: float = 0.02, roughness_ratio: float = 1.5, setter=apply_setting) -> tuple[dict, pd.DataFrame]`: `evaluate(config)` devuelve `{'rmse7', 'rmse28', 'roughness'}`; evalúa primero `start` (la referencia: `rmse28_ref`, `roughness_ref`); en cada vuelta y para cada clave de `order`, prueba cada valor y se queda con el menor `rmse7` entre los que cumplen `rmse28 ≤ (1 + tolerance) · rmse28` de la configuración vigente y `roughness ≤ roughness_ratio · roughness_ref`; devuelve la configuración final y la tabla de todas las evaluaciones (`round`, `key`, `value`, `rmse7`, `rmse28`, `roughness`, `chosen`).

- [ ] **Step 1: Tests**

```python
def test_apply_setting_puts_each_parameter_where_it_lives():
    c = apply_setting(default_config(), 'bw', ('fixed', 45))
    assert c['reg_params'] == {'bw_type': 'fixed', 'bw': 45} and c['label'] == "bw=('fixed', 45)"
    c = apply_setting(default_config(), 'bw', ('adaptive', 2))
    assert c['reg_params']['bw_type'] == 'adaptive' and c['reg_params']['bw_kwargs']['n_iter'] == 2
    assert apply_setting(default_config(), 'wspan', 7)['weights'] == {'wspan': 7}
    assert apply_setting(default_config(), 'tau', 0.16)['he_params'] == {'tau': 0.16}
    assert apply_setting(default_config(), 'he_n_iter', 1)['he_params'] == {'n_iter': 1}
    assert apply_setting(default_config(), 'herd_prior', 10.)['disp_params'] == {'herd_prior': 10.}
    with pytest.raises(ValueError):
        apply_setting(default_config(), 'bw_cap', 0.5)
    with pytest.raises(ValueError):
        apply_setting(default_config(), 'wspan', 0)

def test_coordinate_descent_picks_the_best_allowed_value_per_key_and_records_every_trial():
    def evaluate(c):
        a, b = c.get('a', 0), c.get('b', 0)
        return {'rmse7': (a - 2) ** 2 + (b - 20) ** 2 / 100, 'rmse28': 1.0 if b != 30 else 2.0, 'roughness': 0.02 if a != 3 else 0.05}
    def setter(c, k, v):
        return {**c, k: v}
    best, trials = coordinate_descent(evaluate, {'a': 0, 'b': 0}, {'a': [0, 1, 2, 3], 'b': [0, 20, 30]}, ['a', 'b'], rounds=2, setter=setter)
    assert best['a'] == 2 and best['b'] == 20
    assert set(trials.columns) >= {'round', 'key', 'value', 'rmse7', 'rmse28', 'roughness', 'chosen'}
    assert trials.loc[(trials.key == 'a') & (trials.value == 3), 'chosen'].eq(False).all()   # rugosidad 2,5 × la de partida: nunca elegido
    assert trials.loc[(trials.key == 'b') & (trials.value == 30), 'chosen'].eq(False).all()  # k = 28 empeora más del 2 %
```

(`coordinate_descent` acepta `setter` para los tests; por defecto `apply_setting`. La referencia de rugosidad es la configuración de partida, evaluada antes de la primera vuelta.)

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_tuning_unit.py -v -k "apply_setting or coordinate_descent"`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar `default_config`, `GRID`, `apply_setting` y `coordinate_descent`**

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `pytest tests/test_tuning_unit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/tuning.py tests/test_tuning_unit.py
git commit -m "tuning: configuraciones, rejilla fijada y descenso por coordenadas con guardas de k=28 y rugosidad"
```

---

### Task 6: Orquestación por ámbito: `run_tuning`, `tuning_scope` y sus tablas

**Files:**
- Modify: `mtpy/lib/tuning.py`
- Test: `tests/test_tuning_unit.py`

**Interfaces:**
- Consumes: `backtest.results_fs`, `backtest.results_path`, `backtest.describe_source`, `publish.common.provenance`, `backtest.poll_stats` (mismo `run.json`).
- Produces: `tuning.TUNING_PREFIX = 'backtest/tuning'`, `tuning.TUNING_ROOT` (`backtest/tuning` del repositorio), `tuning.TUNING_FILES = ('cuts.csv', 'polls.csv', 'scores.csv', 'trials.csv', 'run.json')`.
- Produces: `tuning.search_cycles(scope: str, since: str = '2008-01-01', min_polls: int = 10, today: Optional[str] = None) -> dict[str, list[str]]`: `{'search': [...], 'check': [...]}`: las elecciones `featured` del ámbito con ≥ `min_polls` sondeos (`get_event_dates(scope, min_polls=..., featured=True)`) más la próxima elección del ámbito (ciclo en curso); `check` son las generales desde 2020 y las autonómicas desde 2020, el resto `search` (spec A7).
- Produces: `tuning.run_tuning(scope: str, configs: list[dict], cycles: Optional[dict] = None, cut_days: int = 7, horizons: tuple[int, ...] = (7, 28), max_fc: Optional[int] = None, runner: Optional[Callable] = None, verbose: int = 0) -> dict`: ejecuta `run_cut` (o `runner`) para cada configuración, ciclo y corte, y devuelve `{'cuts': DataFrame (una fila por config, ciclo y corte: n_polls, seconds), 'polls': DataFrame (una fila por config, corte, sondeo y partido), 'scores': DataFrame (una fila por config, partición search/check, ciclo y k: rmse, mae, skill, n_cells, autocorr, roughness medias ponderadas por n_cells)}`.
- Produces: `tuning.run_search(scope: str, phase: str = 'smoothing' | 'shrinkage' | 'check', start: Optional[dict] = None, grid: dict = GRID, rounds: Optional[int] = None, cut_days: Optional[int] = None, **kwargs) -> dict`: envuelve `coordinate_descent` con `evaluate(config) = run_tuning(scope, [config], cycles=search_only, ...)` → `{'rmse7', 'rmse28', 'roughness'}` sobre la partición `search`. `'smoothing'`: `ORDER_SMOOTHING`, dos vueltas, cortes cada 7 días, sobre `start` con `house_effects`, `dispersion` y `herding` a `False` (spec A6, fase 1); `'shrinkage'`: `ORDER_SHRINKAGE`, una vuelta, cortes cada 14 días, sólo los ciclos de `search` desde 2015, sobre `start` (el `best` de la fase 1) con todo a `True`; `'check'`: sin búsqueda, evalúa `start` (con todo a `True`) y `default_config()` en la partición `check`. Devuelve `{'best': config, 'trials': DataFrame, 'scores': DataFrame}` (`trials` vacío en `check`).
- Produces: `tuning.tuning_scope(scope: str, out: Optional[str] = None, mode: str = 'smoothing' | 'shrinkage' | 'check' | 'evaluate', configs: Optional[list[dict]] = None, start: Optional[dict] = None, grid: Optional[dict] = None, cycles=None, cut_days=None, horizons=(7, 28), rounds=None, verbose=0, runner=None, stats=None, prov=None) -> dict`: los tres primeros modos llaman a `run_search(scope, phase=mode, start=start or best.json de la fase anterior en la carpeta, ...)`; `'evaluate'` llama a `run_tuning(scope, configs, ...)` tal cual. Escribe `TUNING_FILES` en `results_fs(out, scope)` bajo `TUNING_PREFIX` (en `smoothing` y `shrinkage`, también `best.json`, que la fase siguiente lee), devuelve `{'n_configs', 'n_cuts', 'cycles', 'best' (en las búsquedas), 'folder', 'seconds'}`; `NothingToTune` sin ciclos.

- [ ] **Step 1: Tests con un `runner` falso (sin base)**

```python
class FakeCut:
    """`run_cut` sin modelo: RMSE que depende de la configuración y del ciclo, para probar la agregación y la búsqueda."""
    def __init__(self): self.calls = []; self.configs = []
    def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
        self.calls.append((event_date, cut, config['label'])); self.configs.append((event_date, cut, config))
        base = 2.0 if config['reg_params'].get('bw') == 45 else 2.5
        return {'meta': {'n_polls': 50, 'drange': 30}, 'polls': _polls_rows(config['label'], cut),
                'scores': [{'k': k, 'n_polls': 4, 'n_cells': 12, 'rmse': base + k / 100, 'mae': base, 'skill': 0.3, 'rmse_by_party': {}} for k in horizons],
                'autocorr': 0.1, 'roughness': 0.4, 'seconds': 0.0}

def test_run_tuning_aggregates_scores_by_partition_cycle_and_k(monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23'), pd.Timestamp('2023-06-30')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    cycles = {'search': ['2019-11-10'], 'check': ['2023-07-23']}
    out = run_tuning('es', [default_config()], cycles=cycles, runner=FakeCut())
    assert set(out['scores']['partition']) == {'search', 'check'} and set(out['scores']['k']) == {7, 28}
    assert out['cuts'].shape[0] == 4 and (out['cuts']['config'] == 'default').all()
    row = out['scores'].query("partition == 'check' and k == 7").iloc[0]
    assert row['rmse'] == pytest.approx(2.57) and row['n_cells'] == 24

def test_run_search_smoothing_prefers_the_fixed_45_day_bandwidth_with_effects_off(monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2019-10-11')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2019-09-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    runner = FakeCut()
    out = run_search('es', phase='smoothing', grid={'bw': [('fixed', 'isj'), ('fixed', 45)]}, runner=runner)
    assert out['best']['reg_params']['bw'] == 45
    assert all(not c['house_effects'] and not c['dispersion'] and not c['herding'] for _, _, c in runner.configs)  # fase 1: todo apagado
    assert out['best']['house_effects'] is False

def test_run_search_check_evaluates_start_and_default_with_everything_on(monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    runner = FakeCut()
    start = apply_setting(default_config(), 'bw', ('fixed', 45))
    out = run_search('es', phase='check', start=start, runner=runner)
    assert set(out['scores']['partition']) == {'check'}
    assert set(out['scores']['config']) == {'default', "bw=('fixed', 45)"}
    assert all(c['house_effects'] and c['dispersion'] and c['herding'] for _, _, c in runner.configs)

def test_tuning_scope_writes_the_tables_and_run_json(fresh_app, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    figures = tuning_scope('es-md', out=str(tmp_path), mode='evaluate', configs=[default_config()], runner=FakeCut(),
                           stats=lambda scope: {'db_polls': 7, 'db_last_poll': '2026-10-03'},
                           prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    assert figures['n_configs'] == 1 and figures['n_cuts'] == 2 and figures['folder'] == str(tmp_path / 'es-md')
    for name in ('cuts.csv', 'polls.csv', 'scores.csv', 'run.json'):
        assert (tmp_path / 'es-md' / name).exists()
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['scope'] == 'es-md' and run['mode'] == 'evaluate' and run['commit'] == 'abc1234' and run['configs'] == ['default']

def test_tuning_scope_shrinkage_reads_the_best_of_the_smoothing_phase(fresh_app, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10', '2023-07-23'], 'check': []})
    (tmp_path / 'es-md').mkdir()
    json.dump(apply_setting(default_config(), 'bw', ('fixed', 45)), open(tmp_path / 'es-md' / 'best.json', 'w'))
    runner = FakeCut()
    tuning_scope('es-md', out=str(tmp_path), mode='shrinkage', runner=runner, grid={'herd_prior': [2., 5.]},
                 stats=lambda scope: {}, prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    assert all(c['reg_params']['bw'] == 45 and c['house_effects'] for _, _, c in runner.configs)   # parte del best y lo enciende todo
    assert all(cut.year >= 2015 for _, cut, _ in runner.configs)                                     # sólo ciclos desde 2015

def test_tuning_scope_without_cycles_raises_nothing_to_tune(fresh_app, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': [], 'check': []})
    with pytest.raises(NothingToTune):
        tuning_scope('es-xx', out=str(tmp_path), mode='evaluate', configs=[default_config()], runner=FakeCut())
```

`poll_dates(scope, event_date)` es el helper (una consulta) que da las fechas de los sondeos del ciclo para `cycle_cuts`; los tests lo parchean.

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_tuning_unit.py -v -k "run_tuning or run_search or tuning_scope"`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar `search_cycles`, `poll_dates`, `run_tuning`, `run_search`, `tuning_scope`, `NothingToTune`**

`tuning_scope` escribe con `results_fs(out, scope)` y la carpeta `TUNING_PREFIX` (sin `out`) o la raíz (con `out`, `es` en la raíz y los demás en `{scope}/`, como el backtest); `run.json` lleva `scope`, `mode`, `configs` (etiquetas), `cycles`, `cut_days`, `horizons`, `rounds`, `commit`, `dirty`, `run_at`, `db_polls`, `db_last_poll`, `seconds`. En `smoothing` y `shrinkage`, `trials.csv` es la tabla del descenso y `best.json` la configuración elegida (la de `shrinkage` sobrescribe la de `smoothing`; cada fase conserva su `trials-{fase}.csv`). Las configuraciones de la fase 1 llevan `house_effects`, `dispersion` y `herding` a `False` sólo durante la búsqueda: `best.json` se guarda con los tres a `True`, que es como lo lee la fase 2 y la comprobación.

- [ ] **Step 4: Ejecutar y ver que pasa**

Run: `pytest tests/test_tuning_unit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/tuning.py tests/test_tuning_unit.py
git commit -m "tuning: run_tuning, run_search y tuning_scope con sus tablas y run.json"
```

---

### Task 7: Job `tune`, pipeline y envoltorio de línea de órdenes

**Files:**
- Create: `mtpy/jobs/Tune.py`, `mtpy/pipelines/tune/__init__.py`, `mtpy/pipelines/tune/Tune.py`, `backtest/run_tuning.py`
- Test: `tests/test_pipelines_tune.py`, `tests/test_jobs_tune.py`, `tests/test_launchers.py` (si lista los jobs conocidos)

**Interfaces:**
- Consumes: `tuning.tuning_scope`, `tuning.default_config`, `tuning.apply_setting`, `tuning.NothingToTune`; `ScopePipeline` (`mtpy/pipelines/base.py`), `SequenceJob` (`mtpy/core/worker.py`), `resolve_scopes` (`mtpy/lib/publish`).
- Produces: `Tune(SequenceJob)` con `sequence = {'tune': ['tune']}`, `default_action = 'run'`, `chunk_noun = 'scopes'`, `run(scopes=('es',), mode='smoothing', configs=None, cycles=None, cut_days=None, horizons=(7, 28), rounds=None, out=None, verbose=1, runner=None, stats=None, stop_on_error=None, action='run', **kwargs)`; `mode` ∈ `smoothing | shrinkage | check | evaluate`; `configs` (sólo `evaluate`) en JSON es una lista de `{'key': value}` de la rejilla (cada uno aplicado sobre `default_config()` con `apply_setting`) o `['default']`; `cycles` (opcional) restringe los ciclos, que es como se reparte la fase 2 en ECS (una tarea por ciclo o grupo de ciclos, con la misma carpeta `out` en S3 y `trials-shrinkage-{ciclo}.csv` que un `mode='merge'` posterior junta: ver Task 8).
- Produces: `pipelines.tune.Tune(ScopePipeline)` con `pipes = {'run': {'Setup': ['prepare'], 'Load': ['evaluate']}}`, cabecera `tune: n ámbitos · modo · folder`, línea `{scope}: tune completed (n configs, n cuts) in s`, `skipped` con `NothingToTune`.
- Produces: `backtest/run_tuning.py` con `--scope/--scopes`, `--mode smoothing|shrinkage|check|evaluate`, `--config key=value ...`, `--cycles ...`, `--cut-days`, `--horizons`, `--rounds`, `--out` (por defecto `backtest/tuning` del repositorio).

- [ ] **Step 1: Tests del pipeline y del job, calcados de `tests/test_pipelines_backtest.py` y `tests/test_jobs_backtest.py`**

```python
def test_tune_run_writes_the_folder_and_builds_its_line(fresh_app, patched, tmp_path, capsys):
    p = evaluate(tmp_path=tmp_path)   # Tune().run('run', chunk='es-md', out=..., runner=FakeScope(), stats=stats, mode='evaluate', configs=['default'], verbose=0)
    assert p.result['status'] == 'completed' and p.result['n_configs'] == 1
    assert p.line == 'es-md: tune completed (1 configs, 2 cuts) in 0 s'
    assert 'tune: 2 ámbitos · evaluate · {}'.format(tmp_path) in capsys.readouterr().out

def test_tune_skips_a_scope_without_cycles(fresh_app, patched, tmp_path):
    p = evaluate(tmp_path=tmp_path, scope='es-xx')
    assert p.result['status'] == 'skipped' and 'no cycles' in p.result['reason']

def test_tune_job_runs_every_scope_and_parses_configs(fresh_app, patched, tmp_path):
    runner = FakeScope()
    results = Tune().run(scopes='all', out=str(tmp_path), mode='evaluate', configs=[{'bw': ['fixed', 45]}, 'default'], verbose=0, runner=runner)
    assert [c['scope'] for c in runner.calls] == ['es', 'es-md']
    assert [c['label'] for c in runner.calls[0]['configs']] == ["bw=('fixed', 45)", 'default']
    assert results['es']['status'] == 'completed'
```

(`FakeScope` sustituye a `tuning.tuning_scope`: registra los argumentos y devuelve las cifras; `patched` parchea `resolve_scopes` como en los tests del backtest.)

- [ ] **Step 2: Ejecutar y ver que falla**

Run: `pytest tests/test_pipelines_tune.py tests/test_jobs_tune.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar el pipeline, el job y el envoltorio**

El job normaliza `configs` (`'default'` → `default_config()`; `{'bw': ['fixed', 45]}` → `apply_setting(default_config(), 'bw', ('fixed', 45))`; listas JSON a tuplas) y `horizons` (lista → tupla). `run_tuning.py` traduce las opciones al job, como `run_backtest.py`, e imprime al final la carpeta escrita y, en `search`, la configuración elegida y las dos filas de `scores.csv` de la partición `check` (`default` y `best`).

- [ ] **Step 4: Ejecutar los tests y la suite completa**

Run: `pytest -q`
Expected: PASS (los de integración `SKIPPED`)

- [ ] **Step 5: Commit**

```bash
git add mtpy/jobs/Tune.py mtpy/pipelines/tune backtest/run_tuning.py tests/test_pipelines_tune.py tests/test_jobs_tune.py
git commit -m "jobs: job tune (search y evaluate por ámbitos) y backtest/run_tuning.py"
```

---

### Task 8: Medida de coste y primera ejecución (con base de datos)

**Files:**
- Create: `backtest/tuning/.gitignore` (ignorar `polls.csv`, grande; versionar `scores.csv`, `trials.csv`, `cuts.csv`, `best.json`, `run.json`)
- Modify: `docs/analisis-2026-09/metodo-14-sondeos-futuros.md` (crear, secciones Problema / Método / Coste)

- [ ] **Step 1: Medir el coste de un corte en `es`, con todo apagado y con todo encendido**

Run: `python backtest/run_tuning.py --scope es --mode evaluate --config default --cut-days 28 --horizons 7 28` y lo mismo con `--config house_effects=false dispersion=false herding=false`
Expected: `cuts.csv` con los segundos por corte en cada caso; anotar en el método las dos medias y la extrapolación (fase 1: 700 cortes × 25 configuraciones; fase 2: 350 cortes × 8) frente a la estimación de la spec A10 (segundos y 35-55 s). Si la fase 1 supera las 15 h en el portátil, también va a ECS.

- [ ] **Step 2: Fase 1 (suavizado) en los ciclos de `search`**

Run: `python backtest/run_tuning.py --scope es --mode smoothing`
Expected: `backtest/tuning/trials-smoothing.csv` y `best.json`.

- [ ] **Step 3: Fase 2 (contracciones) en ECS, repartida por ciclos, y `check`**

Run: una tarea por ciclo de `search` desde 2015: `python deploy/ecs/job.py tune '{"scopes":["es"],"mode":"shrinkage","cycles":["2019-11-10"]}' --follow` (y las demás); después `python job.py tune '{"scopes":["es"],"mode":"merge"}'` (junta los `trials-shrinkage-*.csv` de S3, repite el descenso sobre la tabla junta y escribe `best.json`), `pull`, y `python backtest/run_tuning.py --scope es --mode check`.
Expected: `scores.csv` con `default` y `best` en la partición `check` (2023 y 2026), con todo activado. (`mode='merge'` y la acción `pull`/`push` del job `tune` se añaden en este paso, calcadas de las del backtest, con sus tests en `tests/test_jobs_tune.py`.)

- [ ] **Step 4: Registrar las tablas en el método y commitear**

```bash
git add backtest/tuning docs/analisis-2026-09/metodo-14-sondeos-futuros.md mtpy/jobs/Tune.py mtpy/pipelines/tune tests/test_jobs_tune.py
git commit -m "tuning: fases de suavizado y contracciones en los ciclos de búsqueda, merge por ciclos y medida de coste (método 14)"
```

---

### Task 9: Decisión sobre los valores por defecto (criterio A8) y documentación

**Files:**
- Modify: `docs/analisis-2026-09/metodo-14-sondeos-futuros.md` (Resultados, Backtest antes / después, Decisión, Salvedades)
- Modify: `backtest/README.md` (sección «Sondeos futuros» con el protocolo, las tablas y el job), `deploy/README.md` (job `tune` junto a «Backtest del modelo»), `README.md` (entrada del job `tune` en «Jobs»)
- Modify (sólo si se cumple A8): `mtpy/lib/forecaster.py:268-300` y los `set_*_params` con los nuevos valores por defecto; `backtest/results/` reejecutado

- [ ] **Step 1: Backtest de elecciones con la configuración candidata**

Run: `python backtest/run_backtest.py --config backtest/tuning/best.json --out backtest/results_tuned` (añadir `--config` al script y `config` al job `backtest`, que lo enhebra a `run_case` → `Simulator(reg_params, he_params, disp_params, weight_overrides)`; test en `tests/test_jobs_backtest.py`: el job pasa `config` al runner).
Expected: `backtest/results_tuned/by_horizon.csv` comparable con `backtest/results/by_horizon.csv`.

- [ ] **Step 2: Tabla antes / después en el método: MAE de voto, CRPS de escaños, Brier, cobertura 95 por horizonte y casos mejor / peor**

Criterio A8: mejora del RMSE a k = 7 en `check` **y** el backtest no empeora MAE, CRPS ni Brier más de un 2 % ni baja la cobertura del 95 %.

- [ ] **Step 3: Si A8 se cumple y es antes del 20-11: cambiar los defaults en un commit aparte, reejecutar `backtest/results/` y actualizar las constantes de `tests/integration/test_model.py`; si no, dejar `best.json` como opción documentada y anotar la fecha de la siguiente revisión (tras el 29-N)**

- [ ] **Step 4: Suite completa y commit de la documentación**

Run: `pytest -q`
Expected: PASS

```bash
git add docs/analisis-2026-09/metodo-14-sondeos-futuros.md backtest/README.md deploy/README.md README.md
git commit -m "docs: método 14, sondeos futuros: protocolo, resultados y decisión sobre los valores por defecto"
```
