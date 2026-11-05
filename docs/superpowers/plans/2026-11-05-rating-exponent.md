# Banda del rating en el peso del promedio (`rating_exponent`) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir si ensanchar (o estrechar) la banda entre la peor y la mejor encuestadora en el peso del promedio mejora el acierto del modelo frente al resultado, con un exponente γ sobre el peso por rating, y publicar la curva en el método 16.

**Architecture:** Un parámetro `rating_exponent` del Forecaster (aplicado al `weight_rating` de los sondeos tras el relleno por calidad y los overrides), enhebrado por Simulator, backtest, pipeline, job y CLI como `flat_ratings`; un script commiteado `backtest/compare_results.py` sobre funciones puras de `mtpy/lib/backtest.py` que empareja dos carpetas de resultados; una curva del promedio simple por γ en `rating_check.py`; cinco corridas encadenadas en el portátil y la sección 16b del método.

**Tech Stack:** Python 3 / pandas / numpy, pytest; `backtest/run_backtest.py`; sin cambios en la web.

**Spec:** `docs/superpowers/specs/2026-11-05-rating-exponent-design.md`

## Global Constraints

- `rating_exponent` es un float ≥ 0 y finito, 1.0 por defecto en todas las firmas; γ = 1 deja el modelo exactamente como está; negativos o no finitos → `ValueError('rating_exponent: expected a finite number >= 0')`. El modelo vivo no lo recibe (rige el default de la firma).
- Se aplica en `Forecaster.build_series` sólo a las filas de sondeos (`pollster` no nulo), después del relleno por calidad y después de `apply_weight_overrides`, antes de `weight = weight_over · weight_sample · weight_rating`; los resultados oficiales no se tocan; `weight_ratings='flat'` se conserva (con γ cualquiera, 1 ** γ = 1).
- Rejilla de la spec: γ ∈ {0, 0.5, 1, 1.5, 2, 3}; carpetas `backtest/results_exp{γ}` (`results_exp0.5`, `results_exp1`, …), ignoradas por git (`backtest/results_*/`); `n_sim` 1000, semilla 42; una corrida sólo vale sin `OperationalError` en el log, con 28 casos en `es` y 66 en los cinco autonómicos.
- Métricas del juicio: `mae_shares`, `crps_seats`, `brier_vs` (menor es mejor) y `cov_shares95` (mayor es mejor) de `metrics.csv`, emparejadas por (`scope`, `event_date`, `horizon`); poblaciones `generales` (`scope == 'es'`), `autonomicas` (el resto) y `all`; diferencia = candidata − referencia, porcentaje sobre la referencia; «mejor / peor» cuenta los casos en que la candidata gana / pierde, empates aparte.
- Regla para cambiar el default (spec, decisión 6): un γ ≠ 1 mejora `mae_shares` y `crps_seats` más del 2 % en las dos poblaciones, con más casos mejor que peor, sin empeorar `brier_vs` más del 2 % ni bajar `cov_shares95`; si no, nada cambia.
- Nombres de código en inglés neutro; docstrings numpy; docs y commits en español sin identificadores de modelo; `git status` y `git log` antes de cada commit (Luis commitea en paralelo: sus cambios sin commit en `deploy/README.md`, `web/dist/css/site.css`, `mtpy/lib/pages/common.py` y `tests/test_pages.py` no se tocan ni se añaden; `git add` siempre con rutas explícitas); tests nunca con `out=None` ni escribiendo en `backtest/results`.

## Review Focus

1. Un sondeo de una casa **sin fila de rating** (relleno por calidad) también se eleva a γ; con γ = 0 pesa 1 como las demás (Task 1, `test_rating_exponent_raises_every_poll_weight`).
2. Un `weight_overrides` que trae `weight_rating` se eleva a γ igual que los pesos guardados (Task 1, `test_rating_exponent_applies_after_weight_overrides`).
3. Un caso presente en una carpeta y ausente en la otra queda fuera del emparejamiento sin romper las medias (Task 3, `test_pair_results_drops_unpaired_cases`).
4. La cobertura cuenta «mejor» cuando sube, al revés que los errores (Task 3, `test_compare_paired_counts_coverage_upwards`).
5. `--rating-exponent` acepta decimales y rechaza negativos antes de evaluar nada, en el job y en la librería (Task 2, `test_job_backtest_rejects_a_negative_rating_exponent`).

---

### Task 1: `Forecaster(rating_exponent=1.0)` y su paso por el Simulator

**Files:**
- Modify: `mtpy/lib/forecaster.py` (firma de `__init__` ≈ línea 112, docstring ≈ 174, `build_series` ≈ 554-567)
- Modify: `mtpy/lib/simulator.py` (firma ≈ 74, docstring ≈ 183, atributo ≈ 220, construcción del Forecaster ≈ 286)
- Test: `tests/test_forecaster_unit.py` (helper `_weights_builder` ≈ 518-553: añadir el parámetro `rating_exponent=1.0` a `build`; nuevos tests junto a `test_flat_weight_ratings_sets_every_poll_to_one` ≈ 575), `tests/test_simulator_unit.py` (un test de paso al Forecaster con el patrón que ya use el fichero para `weight_ratings` o `disp_params`)

**Interfaces:**
- Produces: `Forecaster(..., rating_exponent: float = 1.0)`; atributo `self.rating_exponent`; `build_series` eleva `weight_rating` de las filas de sondeos a `rating_exponent` tras el relleno y los overrides; `Simulator(..., rating_exponent: float = 1.0)` lo guarda en `self.rating_exponent` y lo pasa al Forecaster. La validación (float finito ≥ 0, mensaje exacto de las Global Constraints) en `Forecaster.__init__` y en `Simulator.__init__`.

- [ ] **Step 1: Tests que fallan**

```python
def test_rating_exponent_raises_every_poll_weight(monkeypatch):
    """γ = 2 eleva al cuadrado el peso por rating de cada sondeo (también el relleno por calidad de B);
    los resultados oficiales quedan como con γ = 1; γ = 0 pone 1 en todos."""
    build, pollsters = _weights_builder(monkeypatch)
    base = build(None).series_raw
    out = build(None, rating_exponent=2.0).series_raw
    polls, base_polls = out.loc[out.pollster.notnull()], base.loc[base.pollster.notnull()]
    assert polls['weight_rating'].tolist() == pytest.approx((base_polls['weight_rating'] ** 2).tolist())
    assert polls['weight'].tolist() == pytest.approx((base_polls['weight_over'] * base_polls['weight_sample'] * base_polls['weight_rating'] ** 2).tolist())
    pd.testing.assert_frame_equal(out.loc[out.pollster.isnull()], base.loc[base.pollster.isnull()])
    flat = build(None, rating_exponent=0.0).series_raw
    assert flat.loc[flat.pollster.notnull(), 'weight_rating'].tolist() == [1.0, 1.0, 1.0]


def test_rating_exponent_applies_after_weight_overrides(monkeypatch):
    """Un override de `weight_rating` = 4 con γ = 0,5 deja 2."""
    build, pollsters = _weights_builder(monkeypatch)
    overrides = pd.DataFrame({'weight_rating': [4.0]}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2023-07-10'), 1, 0)], names=['date', 'pollster_id', 'sponsor_id']))
    out = build(None, weight_overrides=overrides, rating_exponent=0.5).series_raw
    assert out.loc[(out.pollster_id == 1) & (out.date == '2023-07-10'), 'weight_rating'].iloc[0] == pytest.approx(2.0)


@pytest.mark.parametrize('value', [-1.0, float('nan'), float('inf')])
def test_rating_exponent_rejects_negative_or_non_finite(monkeypatch, value):
    build, _ = _weights_builder(monkeypatch)
    with pytest.raises(ValueError, match='rating_exponent: expected a finite number >= 0'):
        build(None, rating_exponent=value)
```

(El helper `build` construye el Forecaster con `__new__` y fija atributos: añade `fc.rating_exponent = rating_exponent` **validado con la misma función** que usa `__init__`, para que el test del rechazo pase por ella; una función de módulo `check_rating_exponent(value) -> float` en `forecaster.py` sirve a las dos clases.) En `tests/test_simulator_unit.py`: `test_simulator_passes_rating_exponent_to_the_forecaster` (monkeypatch de `Forecaster` por un fake que anota los kwargs; `Simulator(..., rating_exponent=1.5)` → `kwargs['rating_exponent'] == 1.5`; y el default 1.0), y `Simulator(rating_exponent=-1)` → `ValueError`.

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_forecaster_unit.py tests/test_simulator_unit.py -q -k rating_exponent`
Expected: FAIL (`TypeError: unexpected keyword argument 'rating_exponent'`).

- [ ] **Step 3: Implementar**: `check_rating_exponent`, el parámetro y la docstring en las dos clases, la línea de `build_series` (`series.loc[series.pollster.notnull(), 'weight_rating'] = series.loc[..., 'weight_rating'] ** self.rating_exponent`, colocada tras `apply_weight_overrides` y antes del producto), el paso en `Simulator`.

- [ ] **Step 4: Ejecutar los tests**

Run: `pytest tests/test_forecaster_unit.py tests/test_simulator_unit.py -q`
Expected: PASS, sin avisos nuevos.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py mtpy/lib/simulator.py tests/test_forecaster_unit.py tests/test_simulator_unit.py
git commit -m "forecaster y simulator: rating_exponent eleva el peso por rating de cada sondeo (1 por defecto, 0 = plano), tras el relleno por calidad y los overrides"
```

### Task 2: `rating_exponent` en el backtest, el pipeline, el job, el CLI y la documentación

**Files:**
- Modify: `mtpy/lib/backtest.py` (`run_case` firma ≈ 366-392 y fila `meta` ≈ 590; `run_backtest` ≈ 815-866; `backtest_scope` firma ≈ 1134, docstring ≈ 1166, `params` ≈ 1198-1206)
- Modify: `mtpy/pipelines/backtest/Backtest.py:52-90`, `mtpy/jobs/Backtest.py:84-217`, `backtest/run_backtest.py` (uso en la docstring y `--rating-exponent` junto a `--flat-ratings`)
- Modify: `backtest/README.md` (párrafo tras «Rating plano (fase C)», ≈ línea 47), `deploy/README.md` (fila tras `flat_ratings`, ≈ 871), `README.md` (frase tras la de `flat_ratings`, ≈ 121)
- Test: `tests/test_pipelines_backtest.py` (junto a los tests `flat` ≈ 173-190), `tests/test_jobs_backtest.py` (≈ 255-275), `tests/test_backtest_unit.py` (pins de las claves de `run.json` ≈ 271 y de la llamada del CLI ≈ 467: añadir `rating_exponent`)

**Interfaces:**
- Consumes: Task 1 (`Simulator(rating_exponent=…)`).
- Produces: `run_case(..., rating_exponent: float = 1.0)` → `Simulator(rating_exponent=…)` y columna `rating_exponent` (float) en la fila `meta`; `run_backtest(..., rating_exponent=1.0)`; `backtest_scope(..., rating_exponent=1.0)` lo mete en `params` (así va a `run_backtest` y a `run.json`); el pipeline y el job lo enhebran (el job lo valida con `check_rating_exponent` antes de `resolve_scopes`); CLI `--rating-exponent` (`type=float`, default 1.0).

- [ ] **Step 1: Tests que fallan**

```python
# tests/test_pipelines_backtest.py
def test_backtest_rating_exponent_passes_through_and_records_it(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    evaluate(runner, tmp_path=tmp_path, events=['2023-05-28'], horizons=[6], rating_exponent=1.5)
    assert runner.calls[0]['rating_exponent'] == 1.5
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['rating_exponent'] == 1.5 and run['flat_ratings'] is False
```

Y el default: en `test_backtest_passes_every_parameter_through_and_records_it` (≈ 97) el dict esperado de `runner.calls[0]` gana `'rating_exponent': 1.0`. En `tests/test_jobs_backtest.py`: `test_job_backtest_passes_rating_exponent` (`runner.calls[0]['rating_exponent'] == 2.0`) y `test_job_backtest_rejects_a_negative_rating_exponent` (`ValueError` con el mensaje de las Global Constraints y `runner.calls == []`). En `tests/test_backtest_unit.py`: los pins ≈ 271 y ≈ 467 con la clave nueva (`'rating_exponent': 1.0`).

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py tests/test_backtest_unit.py -q -k "rating_exponent or every_parameter or run_json or cli"`
Expected: FAIL.

- [ ] **Step 3: Implementar** el enhebrado (firmas, docstrings, `params`, fila `meta` con `'rating_exponent': float(rating_exponent)`, validación del job, CLI) y documentar: `backtest/README.md` («Banda del rating (fase C2a): `rating_exponent` (`--rating-exponent`) eleva el peso por rating de cada sondeo a γ: 1 es el modelo tal cual, 0 el rating plano, γ > 1 ensancha la banda; registrado en `run.json` y en `meta.csv`; las corridas de la rejilla viven en `backtest/results_exp{γ}` y se leen en el método 16, sección «¿Y con otra banda?»»), `deploy/README.md` (fila `| rating_exponent | 1.0 | Exponente del peso por rating de cada sondeo (fase C2a, método 16); 0 equivale a flat_ratings |`), `README.md` (una frase).

- [ ] **Step 4: Ejecutar los tests del backtest**

Run: `pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py tests/test_backtest_unit.py tests/test_launchers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/backtest.py mtpy/pipelines/backtest/Backtest.py mtpy/jobs/Backtest.py backtest/run_backtest.py backtest/README.md deploy/README.md README.md tests/test_pipelines_backtest.py tests/test_jobs_backtest.py tests/test_backtest_unit.py
git commit -m "backtest: rating_exponent (--rating-exponent) hasta el Simulator, validado en el job, registrado en run.json y meta.csv; documentación"
```

### Task 3: Emparejar dos carpetas de resultados: `pair_results`, `compare_paired` y `backtest/compare_results.py`

**Files:**
- Modify: `mtpy/lib/backtest.py` (funciones puras nuevas junto a `summarize_calibration` ≈ 675; constantes `COMPARE_METRICS`, `COMPARE_KEYS`)
- Create: `backtest/compare_results.py`
- Test: `tests/test_backtest_unit.py` (frames diminutos construidos a mano), `tests/test_launchers.py` (el script, como se prueban `run_backtest.py`/`run_tuning.py` allí)

**Interfaces:**
- Produces:
  - `COMPARE_METRICS = ('mae_shares', 'crps_seats', 'brier_vs', 'cov_shares95')`, `HIGHER_IS_BETTER = ('cov_shares95',)`, `COMPARE_KEYS = ('scope', 'event_date', 'horizon')`.
  - `load_results(folder: str, scopes: Sequence[str]) -> pd.DataFrame`: concatena `metrics.csv` de cada ámbito (`es` en la raíz de `folder`, los demás en `folder/{scope}`) con una columna `scope`; un ámbito sin `metrics.csv` se salta con un aviso por `warnings.warn`.
  - `pair_results(a: pd.DataFrame, b: pd.DataFrame, metrics=COMPARE_METRICS, keys=COMPARE_KEYS) -> pd.DataFrame`: inner join por `keys`; una fila por caso con `{metric}_a` y `{metric}_b`; los casos presentes sólo en una de las dos quedan fuera.
  - `compare_paired(paired: pd.DataFrame, metrics=COMPARE_METRICS) -> pd.DataFrame`: una fila por (`population`, `metric`) para `generales`, `autonomicas` y `all` (en ese orden; población por `scope == 'es'`): `n` (casos con las dos métricas finitas), `mean_a`, `mean_b`, `diff` (= `mean_b − mean_a`), `pct` (= `100 · diff / mean_a`, NaN si `mean_a == 0`), `better`, `worse`, `ties` (b frente a a; en `HIGHER_IS_BETTER` gana la mayor); poblaciones sin casos con `n = 0` y NaN.
  - `compare_markdown(table: pd.DataFrame, label_a: str, label_b: str) -> str`: la tabla en markdown con el formato del método 16 (población, métrica, casos, A, B, diferencia y %, mejor / peor (empates)); 4 decimales en las medias, 1 en el %.
  - Script: `python backtest/compare_results.py A B [--scopes es es-md …] [--label-a …] [--label-b …]` imprime la tabla markdown por stdout; `--scopes` por defecto `es es-md es-an es-ct es-vc es-ga`.

- [ ] **Step 1: Tests que fallan**

```python
def test_pair_results_drops_unpaired_cases():
    a = pd.DataFrame({'scope': ['es', 'es', 'es-md'], 'event_date': ['2023-07-23'] * 3, 'horizon': [6, 30, 6],
                      'mae_shares': [2.0, 2.5, 3.0], 'crps_seats': [7., 8., 2.], 'brier_vs': [.1, .2, .0], 'cov_shares95': [.9, .8, .9]})
    b = a.iloc[:2].assign(mae_shares=[1.9, 2.6])
    paired = pair_results(a, b)
    assert len(paired) == 2 and paired['mae_shares_b'].tolist() == [1.9, 2.6]


def test_compare_paired_counts_coverage_upwards():
    paired = pd.DataFrame({'scope': ['es', 'es', 'es-md'], 'event_date': ['2023-07-23'] * 3, 'horizon': [6, 30, 6],
                           'mae_shares_a': [2.0, 2.0, 3.0], 'mae_shares_b': [1.0, 2.0, 4.0],
                           'crps_seats_a': [np.nan, 8., 2.], 'crps_seats_b': [7., 8., 2.],
                           'brier_vs_a': [.1, .2, .0], 'brier_vs_b': [.1, .2, .0],
                           'cov_shares95_a': [.9, .8, .9], 'cov_shares95_b': [.95, .8, .9]})
    table = compare_paired(paired).set_index(['population', 'metric'])
    assert table.loc[('all', 'mae_shares'), ['n', 'better', 'worse', 'ties']].tolist() == [3, 1, 1, 1]
    assert table.loc[('all', 'mae_shares'), 'pct'] == pytest.approx(0.0)        # 7/3 vs 7/3
    assert table.loc[('all', 'crps_seats'), 'n'] == 2                           # el NaN no cuenta
    assert table.loc[('generales', 'cov_shares95'), ['better', 'worse']].tolist() == [1, 0]
    assert table.loc[('autonomicas', 'mae_shares'), ['n', 'worse']].tolist() == [1, 1]


def test_compare_markdown_has_one_row_per_population_and_metric():
    md = compare_markdown(compare_paired(paired), 'γ = 1', 'γ = 2')   # el `paired` del test anterior, como constante del módulo
    assert md.count('| MAE de voto') == 3 and md.count('| Cobertura del 95 %') == 3
    assert '| Generales | MAE de voto (puntos) | 2 |' in md and 'γ = 2' in md.splitlines()[0]
```

`tests/test_launchers.py`: el script con dos carpetas temporales (`metrics.csv` de `es` en la raíz y de `es-md` en subcarpeta) imprime una tabla con `| Todas | ` y sale con 0; una carpeta sin `metrics.csv` de un ámbito pedido avisa y sigue.

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_backtest_unit.py tests/test_launchers.py -q -k "pair_results or compare"`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implementar** las funciones y el script (envoltorio fino: `argparse`, `load_results` × 2, `pair_results`, `compare_paired`, `compare_markdown`, `print`). Nombres públicos de las métricas en `compare_markdown`: `mae_shares` «MAE de voto (puntos)», `crps_seats` «CRPS de escaños», `brier_vs` «Brier de mayorías», `cov_shares95` «Cobertura del 95 %»; poblaciones «Generales», «Autonómicas», «Todas».

- [ ] **Step 4: Comprobar con las carpetas reales de la fase C**

Run: `python backtest/compare_results.py backtest/results_rated backtest/results_flat --label-a "con rating" --label-b "plano"`
Expected: la fila «Generales | MAE de voto» con 28 casos, 2,3737 / 2,3926, +0,8 % y 15 / 13 (plano peor en 15: en el método 16 la cuenta está desde el lado del plano, 13 / 15); «Autonómicas | MAE de voto» 66 casos, 2,5102 / 2,5208, +0,4 %.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/backtest.py backtest/compare_results.py tests/test_backtest_unit.py tests/test_launchers.py
git commit -m "backtest: pair_results, compare_paired y compare_markdown con el script compare_results.py que empareja dos carpetas de resultados por caso y resume por población (sustituye al script de scratch de la fase C)"
```

### Task 4: La curva del promedio simple por γ (`rating_check.simple_average_curve`)

**Files:**
- Modify: `mtpy/lib/rating_check.py` (junto a `summarize_simple_average` ≈ 514)
- Test: `tests/test_rating_check_unit.py` (con el `POLLS` del fichero)

**Interfaces:**
- Consumes: `simple_average_errors(polls, min_pollsters)` y `summarize_simple_average(elections, replicas, seed)`.
- Produces: `simple_average_curve(polls: pd.DataFrame, exponents: Sequence[float], reference: float = 1.0, replicas: int = REPLICAS, seed: int = SEED) -> pd.DataFrame`: para cada γ, `simple_average_errors(polls.assign(weight_rating=polls['weight_rating'] ** γ))`; una fila por (`exponent`, `population`) con `n`, `mae` (la media de `mae_rated` de ese γ), `diff` (media por elección de `mae_γ − mae_reference`, emparejada por `event_scope`, `event_date`), `lo`, `hi` (bootstrap de elecciones, semilla derivada por γ), `better` (elecciones con `diff < 0`), `worse`; la fila de `reference` lleva `diff` 0 y `better = worse = 0`.

- [ ] **Step 1: Tests que fallan**

```python
def test_simple_average_curve_reference_and_flat_rows():
    curve = simple_average_curve(POLLS, [0.0, 1.0, 2.0], replicas=50).set_index(['exponent', 'population'])
    elections = simple_average_errors(POLLS)
    assert curve.loc[(1.0, 'all'), 'diff'] == 0 and curve.loc[(1.0, 'all'), 'mae'] == pytest.approx(elections['mae_rated'].mean())
    assert curve.loc[(0.0, 'all'), 'mae'] == pytest.approx(elections['mae_flat'].mean())
    assert curve.loc[(0.0, 'all'), 'diff'] == pytest.approx((elections['mae_flat'] - elections['mae_rated']).mean())
    assert set(curve.index.get_level_values('population')) == set(POPULATIONS)
    row = curve.loc[(2.0, 'all')]
    assert row['lo'] <= row['diff'] <= row['hi'] and row['better'] + row['worse'] <= row['n']
```

- [ ] **Step 2: Ejecutarlo y ver que falla**

Run: `pytest tests/test_rating_check_unit.py -q -k curve`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implementar** `simple_average_curve` (docstring numpy; reutiliza el bootstrap de elecciones de `summarize_simple_average` extrayendo, si hace falta, un helper `_election_bootstrap(diffs, replicas, seed)` que las dos usen).

- [ ] **Step 4: Ejecutar los tests**

Run: `pytest tests/test_rating_check_unit.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/rating_check.py tests/test_rating_check_unit.py
git commit -m "rating_check: simple_average_curve, el promedio simple de la última campaña con el peso por rating elevado a cada γ de una rejilla, frente a γ = 1"
```

### Task 5: Las corridas y el cribado (las hace el orquestador)

**Files:**
- Create (ignoradas por git): `backtest/results_exp1`, `results_exp0.5`, `results_exp1.5`, `results_exp2`, `results_exp3` (y `results_exp0` si hace falta)
- Create: `.superpowers/sdd/2026-11-05-rating-exponent/backtests.md`, `figures.md`

**Interfaces:**
- Consumes: Tasks 2-4.

- [ ] **Step 1: Cribado 1b (minutos)**: script de scratch: `Computer('es').build_series().rating_check_data('all')` (≈ 4 min, sólo lectura) y `simple_average_curve(data['polls'], [0, .5, 1, 1.5, 2, 3])`; la tabla a `figures.md`.

- [ ] **Step 2: Lanzar la cadena** (γ = 1 primero, para comprobar la base cuanto antes), en segundo plano, log en el espacio del plan:

```bash
nohup sh -c 'for g in 1 0.5 1.5 2 3; do python backtest/run_backtest.py --scope es --rating-exponent $g --out backtest/results_exp$g && python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --rating-exponent $g --out backtest/results_exp$g || exit 1; done' > .superpowers/sdd/2026-11-05-rating-exponent/backtests.log 2>&1 &
```

- [ ] **Step 3: Comprobar la base** en cuanto acabe γ = 1: `python backtest/compare_results.py backtest/results_rated backtest/results_exp1` debe dar diferencia 0 y 0 / 0 en todo; si es así, `results_flat` vale como γ = 0; si no, añadir γ = 0 a la cadena y anotar por qué cambió la base.

- [ ] **Step 4: Al terminar**: `grep -c OperationalError` en el log (0), 28 + 66 casos por carpeta, `run.json` con `rating_exponent` = γ; `compare_results.py results_exp1 results_exp{γ}` para cada γ (y `results_flat` o `results_exp0`), tablas a `figures.md`.

### Task 6: Método 16b, cierre de la spec y memoria

**Files:**
- Modify: `docs/analisis-2026-09/metodo-16-rating.md` (sección nueva «¿Y con otra banda? (fase C2a, {fecha})» tras «¿Mejora el promedio?»; «Decisión (propuesta)» revisada), `docs/superpowers/specs/2026-11-05-rating-exponent-design.md` (sección «Resultados» y estado), `backtest/README.md` (enlace a la sección)

**Interfaces:**
- Consumes: `figures.md` de la Task 5.

- [ ] **Step 1: Escribir la sección**: la rejilla y la banda que da cada γ en el ciclo en curso (0,58-2,25 a γ = 1), la curva del promedio simple (tabla por γ y población: MAE, diferencia frente a γ = 1 con IC, mejor / peor), la curva del backtest (por γ: las cuatro métricas en las dos poblaciones con % y mejor / peor), la lectura y la decisión según la regla 6 de la spec (lo esperable: curva plana dentro del ±2 % → nada cambia, «la anchura de la banda es cosmética»; si un γ pasa la regla, el cambio de default es un commit aparte y de Luis), salvedades (cinco generales; la referencia γ = 1 rehecha; Brier de 2023).

- [ ] **Step 2: Cerrar la spec** («Resultados» con las dos curvas resumidas y la decisión; estado «CONSTRUIDA en dev»), enlazar desde `backtest/README.md`, suite completa (`pytest tests -q --ignore=tests/integration`; los pins de `tests/integration/test_model.py` son de Luis).

- [ ] **Step 3: Commit**

```bash
git add docs/analisis-2026-09/metodo-16-rating.md docs/superpowers/specs/2026-11-05-rating-exponent-design.md backtest/README.md
git commit -m "método 16b: la banda del rating en el peso del promedio (curvas del promedio simple y del backtest por γ) y cierre de la spec"
```
