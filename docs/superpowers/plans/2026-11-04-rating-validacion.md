# Fase C: validar el rating de encuestadoras — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir fuera de muestra si el rating de encuestadoras mejora el promedio y si discrimina a las casas, y publicar la respuesta con cifras vivas en la tarjeta «Cómo lo comprobamos» de /metodo/rating.

**Architecture:** Una opción `flat_ratings` del backtest (el Forecaster acepta `weight_ratings='flat'`) para la prueba 1a; un módulo puro `mtpy/lib/rating_check.py` con las medidas de las pruebas 1b y 2 sobre dos frames que reúne `Computer.rating_check_data()`; una parte `rating-check` del análisis nacional que `publish analysis` escribe sólo con `rating_check=True`; y la tarjeta rehecha en `web/templates/metodo_rating.html` con su contexto en `pages/metodo.py` y su gráfico en `web/dist/js/pages/metodo.js`. Las corridas del backtest y las cifras van al método 16.

**Tech Stack:** Python 3 / pandas / numpy / scipy (`norm`), Jinja2, ECharts (módulos ES de `web/dist/js`), pytest; `node` para los tests de JS.

**Spec:** `docs/superpowers/specs/2026-11-04-rating-validacion-design.md`

## Global Constraints

- Nada cambia en el modelo vivo ni en el ranking: `weight_ratings='flat'` y `flat_ratings` son inertes por defecto; ningún parámetro del rating se toca.
- Sin escrituras en la RDS, S3 ni ECS: los backtests escriben en `backtest/results_flat` y `backtest/results_rated` (ignoradas por git: `backtest/results_*/`); la publicación real la lanza Luis. Los tests nunca con `out=None` en el backtest ni escriben en `backtest/results`.
- Sin cambios de esquema en la base.
- Nombres de código en inglés neutro (`rating_check`, `flat_ratings`, `d_obs`, `d_prev`); textos públicos en español con las reglas de Luis: «sólo» con tilde, encuestadora/casa nunca empresa, «actualización» para los runs; los comprueba `assert_public_wording`.
- Una elección es un ámbito y una fecha (`event_scope`, `event_date`); dos autonómicas el mismo día son dos elecciones.
- Poblaciones: `generales` = `event_scope == 'es'`, `autonomicas` = el resto, `all` = todas. Constantes del módulo: `WINDOW_DAYS = 30`, `MIN_POLLSTERS = 3`, `MIN_EVENT_POLLS = 10`, `EVIDENCE_SPLIT = 2.0` (`num_events_w`), `REPLICAS = 2000`, `SEED = 0`, `QUARTERS = 4`.
- Escalas: dentro de los frames el diferencial va en log odds ratio (LOR) y `error_avg` en puntos de voto; `rating`, `rating_adj`, `quality` en 0-1. En la parte publicada y en la nube el diferencial va ×100 (la escala «Diferencial» del ranking) y los ratings ×100.
- Commits en `dev`, uno por tarea como poco, en español, sin identificadores de modelo; Luis commitea en paralelo: `git status` y `git log` antes de cada commit, nunca reescribir los suyos.
- Documentación con cada tarea que la necesita: docstrings numpy, `backtest/README.md`, `deploy/README.md`, `README.md`, `docs/web/contrato.md`.

## Review Focus

1. Una casa con sondeos en la elección e pero **sin fila de rating** en e (casa nueva): no es un par de la prueba 2 y en la prueba 1b lleva su calidad previa normalizada, como el Forecaster (Task 5, `test_rating_check_data_keeps_a_new_house_out_of_the_pairs_and_in_the_polls`).
2. Un par con `num_events_w == 0` (sólo prior): entra en la correlación y en el skill (con `rating == quality`), queda fuera de la pendiente (`d_prev` NaN) (Task 4, `test_discrimination_slope_skips_prior_only_pairs`).
3. Dos autonómicas **el mismo día**: dos pares distintos por casa, y el rating de cada una no ve a la otra (Task 5, `test_rating_check_data_keeps_same_day_elections_apart`).
4. `publish analysis` **sin** `rating_check` conserva la parte anterior y `meta.params.rating_check` es `false`; con `rating_check` en un ámbito autonómico no escribe nada (Task 6, `test_publish_analysis_keeps_the_previous_rating_check_when_not_requested`).
5. `weight_ratings='flat'` pone 1 también a las casas **sin fila de rating** (que sin él llevarían su calidad previa) y no toca los resultados oficiales (Task 1, `test_flat_weight_ratings_sets_every_poll_to_one`).

---

### Task 1: `Forecaster(weight_ratings='flat')`

**Files:**
- Modify: `mtpy/lib/forecaster.py` (`__init__` docstring del parámetro `weight_ratings` ≈ líneas 174-181; `build_series` ≈ líneas 552-557)
- Test: `tests/test_forecaster_unit.py` (junto a `test_weight_ratings_override_replaces_the_stored_weight`, ≈ línea 555; helper `_weights_builder`)

**Interfaces:**
- Produces: `Forecaster(weight_ratings: Optional[pd.Series | Literal['flat']])`; con `'flat'`, `build_series` deja `weight_rating = 1.0` en todas las filas de sondeos (las de `pollster` no nulo) antes de calcular `weight`; los resultados oficiales no se tocan; con una `Series` o `None` nada cambia. `Simulator` lo pasa tal cual (no requiere cambios: `weight_ratings=self.weight_ratings`).

- [ ] **Step 1: Escribir el test que falla**

```python
def test_flat_weight_ratings_sets_every_poll_to_one(monkeypatch):
    """`weight_ratings='flat'`: 1 en todos los sondeos (también en la casa B, que sin él llevaría su calidad previa);
    los resultados oficiales no se tocan y `weight` = over · sample."""
    build, pollsters = _weights_builder(monkeypatch)
    series = build('flat')
    polls = series.loc[series.pollster.notnull()]
    assert polls['weight_rating'].tolist() == [1.0, 1.0, 1.0]
    assert polls['weight'].tolist() == pytest.approx([0.5, 0.5, 3.0])
    results = series.loc[series.pollster.isnull()]
    assert results['weight_rating'].isna().all() or (results['weight_rating'] == 1.0).all()
```

Nota: comprueba primero qué `weight_rating` tienen hoy las filas de resultados en `build(None)` (la línea `fillna(... map(weight_quality))` las deja NaN porque no tienen `pollster_id`); el test fija que con `'flat'` siguen igual que con `None`.

- [ ] **Step 2: Ejecutarlo y ver que falla**

Run: `pytest tests/test_forecaster_unit.py::test_flat_weight_ratings_sets_every_poll_to_one -v`
Expected: FAIL (`AttributeError`/`TypeError` en `override_weight_rating` al hacer `.rename` sobre una `str`).

- [ ] **Step 3: Implementar en `build_series`**

En `mtpy/lib/forecaster.py`, donde hoy `if self.weight_ratings is not None: ... override_weight_rating`, añadir la rama `elif`/`if isinstance(self.weight_ratings, str)`: si es `'flat'`, `series.loc[series.pollster.notnull(), 'weight_rating'] = 1.0` y no aplicar el `fillna` por calidad a esas filas (ya no son NaN); cualquier otra cadena → `ValueError("weight_ratings: expected a Series, None or 'flat'")`. Actualizar la docstring del parámetro.

- [ ] **Step 4: Ejecutar los tests del Forecaster**

Run: `pytest tests/test_forecaster_unit.py -q`
Expected: PASS (los existentes de `weight_ratings` con `Series` siguen en verde).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py tests/test_forecaster_unit.py
git commit -m "forecaster: weight_ratings='flat' pone a 1 el peso por rating de todos los sondeos (rating plano para el backtest de la fase C)"
```

### Task 2: `flat_ratings` en el backtest, el job, el lanzador y la documentación

**Files:**
- Modify: `mtpy/lib/backtest.py` (`run_case` fila `meta` ≈ línea 560-592; `backtest_scope` firma ≈ 1125-1135, cuerpo ≈ 1188-1212)
- Modify: `mtpy/pipelines/backtest/Backtest.py:52-90` (`evaluate_task`)
- Modify: `mtpy/jobs/Backtest.py:84-215` (`run`, docstring de parámetros)
- Modify: `backtest/run_backtest.py` (docstring de uso línea 8, `--flat-ratings` junto a `--recompute-ratings` línea 67, llamada a `Backtest().run` línea 93)
- Modify: `backtest/README.md` (párrafo de parámetros del método 15, ≈ línea 45: añadir uno de la fase C), `deploy/README.md` tabla del backtest (≈ línea 859, fila `flat_ratings`), `README.md` sección `backtest` (una frase)
- Test: `tests/test_pipelines_backtest.py` (junto a `test_backtest_passes_every_parameter_through_and_records_it`, línea 97), `tests/test_jobs_backtest.py`

**Interfaces:**
- Consumes: Task 1 (`weight_ratings='flat'`).
- Produces: `backtest_scope(..., recompute_ratings=False, flat_ratings=False, ...)`: con `flat_ratings` pasa `weight_ratings='flat'` a `run_backtest` (y por tanto a `run_case` → `Simulator`); `run.json` lleva `flat_ratings` (bool); `meta.csv` una columna `flat_ratings` (bool) en cada caso; `ValueError('backtest: flat_ratings and recompute_ratings are exclusive')` antes de evaluar nada cuando los dos son `True`. Job: `Backtest().run(..., flat_ratings=False)` con la misma validación; CLI `--flat-ratings`.

- [ ] **Step 1: Tests que fallan (pipeline y job)**

En `tests/test_pipelines_backtest.py`, con el `FakeRunner` y el helper `evaluate` del fichero:

```python
def test_backtest_flat_ratings_passes_flat_and_records_it(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    evaluate(runner, tmp_path=tmp_path, events=['2023-05-28'], horizons=[6], flat_ratings=True)
    assert runner.calls[0]['weight_ratings'] == 'flat'
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['flat_ratings'] is True and run['recompute_ratings'] is False


def test_backtest_flat_and_recompute_ratings_are_exclusive(fresh_app, patched, tmp_path):
    with pytest.raises(ValueError, match='flat_ratings and recompute_ratings are exclusive'):
        evaluate(FakeRunner(), tmp_path=tmp_path, flat_ratings=True, recompute_ratings=True)
```

En `tests/test_jobs_backtest.py`, siguiendo el patrón de los tests de parámetros del job (`Backtest().run(...)` con `runner`): `test_job_backtest_rejects_flat_with_recompute_ratings` (mismo `ValueError`, antes de cualquier llamada al runner) y `test_job_backtest_passes_flat_ratings` (`runner.calls[0]['weight_ratings'] == 'flat'`).

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py -q -k flat`
Expected: FAIL (`TypeError: unexpected keyword argument 'flat_ratings'`).

- [ ] **Step 3: Implementar**

- `backtest_scope`: parámetro `flat_ratings: bool = False`; la validación de exclusividad al principio (antes de `default_events`); `weight_ratings = 'flat' if flat_ratings else (rating_overrides(scope) if recompute_ratings else None)`; `record[...]['flat_ratings'] = flat_ratings`.
- `run_case`: en la fila `meta`, `'flat_ratings': isinstance(weight_ratings, str)`.
- Pipeline y job: enhebrar `flat_ratings` como `recompute_ratings` (firma, docstring, `super().run(...)`); el job valida la exclusividad antes de `resolve_scopes`.
- CLI: `parser.add_argument('--flat-ratings', action='store_true', help='weigh every poll with rating 1 (the average without the rating, phase C); exclusive with --recompute-ratings')` y `flat_ratings=args.flat_ratings` en la llamada.

- [ ] **Step 4: Ejecutar los tests del backtest**

Run: `pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py tests/test_backtest_unit.py tests/test_launchers.py -q`
Expected: PASS.

- [ ] **Step 5: Documentar**

`backtest/README.md`: tras el párrafo del método 15 (≈ línea 45), un párrafo «Rating plano (fase C)»: `flat_ratings` / `--flat-ratings` pone a 1 el peso por rating de todos los sondeos (también de las casas sin rating, que sin él llevan su calidad previa), excluyente con `--recompute-ratings`, registrado en `run.json` y en `meta.csv`; las corridas de la fase C viven en `backtest/results_flat` y `backtest/results_rated` y se leen en `docs/analisis-2026-09/metodo-16-rating.md`. `deploy/README.md`: fila `| flat_ratings | false | Pesa todos los sondeos con rating 1 (fase C, método 16); excluyente con recompute_ratings |`. `README.md`: una frase en la sección `backtest` remitiendo a `backtest/README.md`.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/backtest.py mtpy/pipelines/backtest/Backtest.py mtpy/jobs/Backtest.py backtest/run_backtest.py backtest/README.md deploy/README.md README.md tests/test_pipelines_backtest.py tests/test_jobs_backtest.py
git commit -m "backtest: opción flat_ratings (--flat-ratings) que corre el modelo con rating plano, excluyente con recompute_ratings, registrada en run.json y meta.csv; documentación"
```

### Task 3: Lanzar los backtests de la prueba 1a (en segundo plano, lo hace el orquestador)

**Files:**
- Create (ignorados por git): `backtest/results_flat/`, `backtest/results_rated/` y sus `{scope}/`
- Create: `.superpowers/sdd/2026-11-04-rating-validacion/backtests.md` (hora de lanzamiento, comandos, PID, estado)

**Interfaces:**
- Consumes: Task 2 (`--flat-ratings`, `--out`).
- Produces: las doce carpetas de resultados (`es` y `es-md es-an es-ct es-vc es-ga`, planas y con rating) que lee la Task 8.

- [ ] **Step 1: Comprobar que nada escribe en `backtest/results`**

Run: `git status --short backtest/results | head` y `ls backtest/results_flat backtest/results_rated 2>&1`
Expected: sin cambios en `backtest/results`; las dos carpetas no existen aún.

- [ ] **Step 2: Lanzar las dos corridas de `es` en segundo plano, una tras otra, y después las autonómicas**

```bash
nohup sh -c 'python backtest/run_backtest.py --scope es --out backtest/results_rated && python backtest/run_backtest.py --scope es --flat-ratings --out backtest/results_flat && python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --out backtest/results_rated && python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --flat-ratings --out backtest/results_flat' > backtest/results_flat.log 2>&1 &
```

(Parámetros por defecto: `n_sim` 1000, `seed` 42, mismos en las cuatro. Coste esperado: ≈ 1 h por corrida de `es`, 1-2 h cada tanda autonómica.) Anotar en `backtests.md` el comando, la hora y el PID; no esperar aquí: seguir con la Task 4.

- [ ] **Step 3 (al final, antes de la Task 8): comprobar que terminaron**

Run: `tail -5 backtest/results_flat.log; ls backtest/results_flat backtest/results_rated; python -c "import json;print(json.load(open('backtest/results_flat/run.json'))['flat_ratings'])"`
Expected: cuatro corridas completadas, `run.json` de `results_flat` con `flat_ratings: true` y el de `results_rated` con `false`, las cinco carpetas autonómicas en cada una.

### Task 4: `mtpy/lib/rating_check.py`, las medidas puras

**Files:**
- Create: `mtpy/lib/rating_check.py`
- Test: `tests/test_rating_check_unit.py`

**Interfaces:**
- Consumes: nada del código nuevo; frames construidos a mano en los tests con estas columnas.
- Produces (todas funciones puras, sin base ni ficheros):
  - Constantes `WINDOW_DAYS`, `MIN_POLLSTERS`, `MIN_EVENT_POLLS`, `EVIDENCE_SPLIT`, `REPLICAS`, `SEED`, `QUARTERS`, `POPULATIONS = ('all', 'generales', 'autonomicas')`.
  - `population_of(scope: str) -> str` (`'generales'` si `scope == 'es'`, si no `'autonomicas'`).
  - **Frame `pairs`** (lo produce la Task 5): columnas `event_scope` (str), `event_date` (str ISO), `population`, `pollster_id` (int), `pollster` (str), `d_obs` (float, LOR), `error_avg` (float, puntos), `n_polls` (int), `n_event_polls` (int, sondeos de la elección en la bolsa, todas las casas), `rating` (0-1), `rating_adj` (0-1), `d_prev` (LOR, NaN si `num_events_w == 0`), `num_events_w` (float), `quality` (0-1), `weight_rating` (float).
  - **Frame `polls`** (lo produce la Task 5): una fila por sondeo y partido principal: `event_scope`, `event_date`, `population`, `pollster_id`, `pollster`, `party`, `error` (puntos, cifra − resultado), `weight_over`, `weight_sample`, `weight_rating`.
  - `bootstrap_ci(values_by_house: dict[int, np.ndarray] | pd.DataFrame, statistic: Callable[[pd.DataFrame], float], replicas: int = REPLICAS, seed: int = SEED) -> tuple[float, float]`: IC 95 % remuestreando casas (`pollster_id`) con reemplazo y conservando todos sus pares; `np.random.default_rng(seed)`.
  - `spearman(pairs: pd.DataFrame, tau: float) -> dict`: `{'value', 'lo', 'hi', 'n'}` de la correlación de rango entre `rating` y `d_obs` (scipy `spearmanr`), IC por `bootstrap_ci`; `tau` sólo documental (no se usa aquí).
  - `slope(pairs: pd.DataFrame) -> dict`: `{'value', 'lo', 'hi', 'n'}` de la pendiente de `d_obs` sobre `d_prev` (regresión con constante, `np.polyfit(d_prev, d_obs, 1)[0]`), sólo pares con `d_prev` finito; `n` = esos pares; con `n < 3`, `value` NaN y sin IC.
  - `skill(pairs: pd.DataFrame, tau: float) -> dict`: `{'mse_none', 'mse_prior', 'mse_rating', 'skill_prior', 'skill_rating', 'n'}` con `pred_none = 0`, `pred_prior = -tau * norm.ppf(clip(quality, 1e-3, 1 - 1e-3))`, `pred_rating = -tau * norm.ppf(clip(rating, ...))`; `skill_x = 1 - mse_x / mse_none`.
  - `quarters(pairs: pd.DataFrame) -> list[dict]`: por elección con ≥ `QUARTERS` casas, cuartos del `rating` (rango descendente, `pd.qcut` sobre el rango, el 1 es el mejor); salida `[{'population', 'quarter' (1..4), 'error_avg' (media en puntos), 'n'}]` para `all`, `generales` y `autonomicas`, en ese orden y cuartos ascendentes.
  - `discrimination(pairs: pd.DataFrame, tau: float, replicas: int = REPLICAS, seed: int = SEED) -> dict`: `{population: {'n_pairs', 'n_pollsters', 'n_events', 'spearman', 'spearman_evidence': {'high': ..., 'low': ...} (pares con `num_events_w >= EVIDENCE_SPLIT` / menos), 'slope', 'slope_polled' (sólo `n_event_polls >= MIN_EVENT_POLLS`), 'skill'}}` para las tres poblaciones (`all` = el frame entero); `first`/`last` (fechas mínima y máxima) en la clave `'dates'` del nivel superior.
  - `simple_average_errors(polls: pd.DataFrame, min_pollsters: int = MIN_POLLSTERS) -> pd.DataFrame`: una fila por elección (`event_scope`, `event_date`, `population`, `n_polls`, `n_pollsters`, `mae_flat`, `mae_rated`, `diff` = `mae_rated − mae_flat`): por partido, el error del promedio ponderado `Σ w·error / Σ w` con `w_flat = weight_over · weight_sample` y `w_rated = w_flat · weight_rating`; MAE = media de los valores absolutos sobre los partidos de la elección; elecciones con menos de `min_pollsters` casas distintas se descartan.
  - `summarize_simple_average(elections: pd.DataFrame, replicas: int = REPLICAS, seed: int = SEED) -> dict`: `{population: {'n', 'mae_flat', 'mae_rated', 'diff', 'lo', 'hi', 'better', 'worse'}}` con la media de `diff`, IC por bootstrap **de elecciones** (no de casas) y el recuento de elecciones con `diff < 0` (`better`, el rating ayudó) y `> 0` (`worse`).
  - `scatter_rows(pairs: pd.DataFrame) -> list[dict]`: `[{'scope', 'event_date', 'pollster', 'd_prev', 'd_obs'}]` con los dos diferenciales ×100 y redondeados a 2 decimales, sólo pares con `d_prev` finito, y `fit_line(pairs) -> list | None` (`[[x0, y0], [x1, y1]]` de la recta de `slope` en los extremos de `d_prev` ×100).
  - `rating_check_summary(pairs, polls, tau, sigma, params: dict) -> dict`: el `data` de la parte: `{'summary': discrimination(...) y en cada población la clave 'average' con summarize_simple_average(...)[population], 'quarters': quarters(pairs), 'scatter': scatter_rows(pairs), 'fit': fit_line(pairs), 'dates': {...}, 'params': {**params, 'tau', 'sigma', 'k': (sigma / tau) ** 2, 'window_days', 'min_pollsters', 'min_event_polls', 'evidence_split', 'replicas', 'seed', 'quarters'}}`; todos los floats redondeados a 4 decimales (los ratings y `d_*` ya ×100 donde se publiquen).

- [ ] **Step 1: Tests que fallan**

Construir en el test dos frames pequeños a mano: `pairs` con 3 elecciones (`es` 2019-11-10, `es-md` 2021-05-04, `es-cl` 2021-05-04, el mismo día) y 4 casas A-D con `rating` 0.8/0.6/0.4/0.2 y `d_obs` que crecen con el rating invertido (correlación −1 perfecta), B con `num_events_w = 0` y `d_prev` NaN; `polls` con 2 partidos y pesos sencillos.

```python
def test_discrimination_perfect_inverse_order_gives_minus_one():
    out = discrimination(PAIRS, tau=.01, replicas=50)
    assert out['all']['spearman']['value'] == pytest.approx(-1.0)
    assert out['all']['n_pairs'] == 12 and out['all']['n_events'] == 3 and out['all']['n_pollsters'] == 4
    assert out['generales']['n_events'] == 1 and out['autonomicas']['n_events'] == 2


def test_discrimination_slope_skips_prior_only_pairs():
    out = discrimination(PAIRS, tau=.01, replicas=50)
    assert out['all']['slope']['n'] == 9          # B (d_prev NaN) queda fuera
    assert out['all']['skill']['n'] == 12          # pero entra en el skill con rating == quality


def test_skill_of_a_perfect_predictor_is_one():
    pairs = PAIRS.assign(d_obs=lambda df: -.01 * norm.ppf(df['rating']))
    assert skill(pairs, tau=.01)['skill_rating'] == pytest.approx(1.0)


def test_quarters_rank_inside_each_election():
    rows = quarters(PAIRS)
    assert [r['quarter'] for r in rows if r['population'] == 'all'] == [1, 2, 3, 4]
    assert rows[0]['n'] == 3                        # la casa A es el cuarto 1 en las tres elecciones


def test_simple_average_errors_drops_small_elections_and_pairs_the_two_weights():
    out = simple_average_errors(POLLS)
    assert set(out['event_date']) == {'2019-11-10', '2021-05-04'} and (out['n_pollsters'] >= 3).all()
    row = out.set_index(['event_scope', 'event_date']).loc[('es', '2019-11-10')]
    assert row['mae_flat'] == pytest.approx(EXPECTED_FLAT) and row['mae_rated'] == pytest.approx(EXPECTED_RATED)


def test_summarize_simple_average_counts_better_and_worse():
    out = summarize_simple_average(simple_average_errors(POLLS), replicas=50)
    assert out['all']['better'] + out['all']['worse'] <= out['all']['n'] and out['all']['lo'] <= out['all']['diff'] <= out['all']['hi']


def test_rating_check_summary_has_the_part_shape_and_scaled_scatter():
    data = rating_check_summary(PAIRS, POLLS, tau=.01, sigma=.014, params={})
    assert set(data) == {'summary', 'quarters', 'scatter', 'fit', 'dates', 'params'}
    assert data['params']['k'] == pytest.approx(1.96) and data['scatter'][0]['d_prev'] == pytest.approx(PAIRS['d_prev'].dropna().iloc[0] * 100, abs=.01)
    assert all('average' in data['summary'][p] for p in POPULATIONS)
```

`EXPECTED_FLAT`/`EXPECTED_RATED` se calculan a mano en el test a partir de `POLLS` (dos partidos, tres casas).

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_rating_check_unit.py -q`
Expected: FAIL (`ModuleNotFoundError: mtpy.lib.rating_check`).

- [ ] **Step 3: Implementar `mtpy/lib/rating_check.py`** con las firmas de arriba y una docstring de módulo que remita a la spec y al método 16. Bootstrap: una semilla nueva por medida derivada de `SEED` (`seed + i` con `i` fijo por medida), como en la prueba por criterio del 2026-10-20.

- [ ] **Step 4: Ejecutar los tests**

Run: `pytest tests/test_rating_check_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/rating_check.py tests/test_rating_check_unit.py
git commit -m "rating_check: medidas puras de la validación del rating (correlación de rango, pendiente de calibración, skill frente a nada/prior/rating, cuartos, promedio simple con y sin rating, resumen de la parte)"
```

### Task 5: `Computer.rating_check_data()`: reunir pares y sondeos

**Files:**
- Modify: `mtpy/lib/computer.py` (`rating_pool` ≈ líneas 1618-1656: extraer `scope_computers`; método nuevo `rating_check_data` junto a `compute_ratings`)
- Test: `tests/test_computer_unit.py` (helpers `bare_computer`, `pool`, línea 188-222)
- Create (scratch, fuera del repo): un script de comprobación con datos reales

**Interfaces:**
- Consumes: `rating_check.population_of`; `data.get_ratings(scope)`; `Computer._rating_polls`, `_event_deviations`, `rating_pool`, `get_polls_metric`, `event_params`.
- Produces:
  - `Computer.scope_computers(scopes: list[str]) -> dict[str, 'Computer']`: el bucle de `rating_pool` (el propio `self` para su ámbito, un `Computer(...).build_series()` con los mismos filtros para los demás; los ámbitos sin sondeos se saltan); `rating_pool` pasa a usarlo y devuelve exactamente lo de hoy.
  - `Computer.rating_check_data(scopes: str | list[str] = 'all', rating_weights: Optional[dict] = None, ratings: Optional[Callable[[str], pd.DataFrame]] = None, window_days: int = WINDOW_DAYS) -> dict`: `{'pairs': DataFrame, 'polls': DataFrame, 'n_scopes': int, 'seconds': float}` con los frames de la Task 4. `scopes`/`rating_weights` como en `compute_ratings` (ámbitos con `rating_weight > 0`). `ratings(scope)` inyectable (por defecto `data.get_ratings`), leído por ámbito: filas (`event_date`, `pollster_id`) con `rating`, `rating_adj`, `bias_dev_adj`, `num_events_w`, `quality`, `weight_rating` **en las escalas guardadas**, que el método deshace: `rating`, `rating_adj` ÷ 100, `bias_dev_adj` → LOR con `bias_to_lor`, `quality` ÷ 100 si viene en 0-100 (comprobar en el Step 3).
  - `pairs`: de `self._event_deviations(self._rating_polls(pool, weights))` (una fila por casa y elección: `d_obs = bias_dev`), cruzado por (`event_scope`, `event_date`, `pollster_id`) con las filas de ratings de ese ámbito y fecha (**inner**: una casa sin fila no es un par); `error_avg` = media de `error_avg` de sus sondeos en la elección ponderada con el mismo `weight_poll`, ×100; `n_polls`, `n_event_polls` desde la bolsa; `d_prev = bias_dev_adj` de la fila, NaN donde `num_events_w == 0`. Sólo elecciones `featured` con resultado (las de la bolsa). Las casas de `drop_mtypes` fuera (no tienen rating).
  - `polls`: por ámbito, `comp.get_polls_metric('error', featured='results', drange=(0, window_days), bmap=None)` reducido a los partidos `main` de cada elección (`comp.event_params[dt]['bmaps']['main']`, con la sucesión que ya aplica el Computer), a formato largo con `error` ×100 (puntos), sólo sondeos `computed`, sin `drop_mtypes` ni `drop_ctypes`, con `weight_over > 0`; `weight_rating` = el de la fila de ratings de (ámbito, elección, casa) o, sin fila, `log1p(q) / log1p(mean q)` con `q` la calidad previa (0-1) de las casas de esa elección, como `Forecaster.build_series`.

- [ ] **Step 1: Tests que fallan**

Con `bare_computer()` y `pool(...)` de `tests/test_computer_unit.py` (añadiendo a las filas de `pool` una columna `computed=True`, `weight_rating` y `error_avg` ya están), y `ratings` inyectado como función que devuelve un frame por ámbito:

```python
def test_rating_check_data_keeps_same_day_elections_apart(monkeypatch):
    """es-md y es-cl el 2021-05-04: dos pares por casa; el rating de cada par es el de su propio ámbito."""
    c = bare_computer()
    rows = [('es-md', '2021-05-04', 'A'), ('es-cl', '2021-05-04', 'A'), ('es-md', '2021-05-04', 'B')]
    monkeypatch.setattr(c, 'rating_pool', lambda scopes: pool(rows))
    monkeypatch.setattr(c, 'scope_computers', lambda scopes: {})
    out = c.rating_check_data(scopes=['es-md', 'es-cl'], rating_weights={'es-md': .5, 'es-cl': .5},
                              ratings=lambda scope: RATINGS[scope])
    pairs = out['pairs'].set_index(['event_scope', 'pollster'])
    assert len(pairs) == 3 and pairs.loc[('es-md', 'A'), 'rating'] != pairs.loc[('es-cl', 'A'), 'rating']
    assert pairs['population'].unique().tolist() == ['autonomicas']


def test_rating_check_data_keeps_a_new_house_out_of_the_pairs_and_in_the_polls(monkeypatch):
    """La casa C sondea 2019 sin fila de rating: no es un par; en `polls` lleva la calidad previa normalizada."""
    ...
    assert 'C' not in out['pairs']['pollster'].tolist()
    c_rows = out['polls'].loc[out['polls']['pollster'] == 'C']
    assert c_rows['weight_rating'].iloc[0] == pytest.approx(np.log1p(.6) / np.log1p(QUALITY_MEAN))


def test_rating_check_data_undoes_the_stored_scales(monkeypatch):
    """`rating` 77,1 → 0,771; `bias_dev_adj` −0,84 (escala bias) → `bias_to_lor`; `d_prev` NaN con `num_events_w` 0."""
```

(`polls` se alimenta en los tests con un `get_polls_metric` falso por ámbito que devuelve un frame con `event_date`, `pollster`, `pollster_id`, `days`, `weight_over`, `weight_sample`, `computed`, `mtype`, `ctype` y una columna por partido con el error en fracción.)

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_computer_unit.py -q -k rating_check`
Expected: FAIL (`AttributeError: rating_check_data`).

- [ ] **Step 3: Comprobar las escalas guardadas con una lectura real (sólo lectura)**

Run (con el `.env` local): `python -c "from mtpy import mtpy; mtpy.run(); from mtpy.lib.data import get_ratings; r = get_ratings('es', '2023-07-23'); print(r.loc[r.pollster == 'GAD3', ['rating', 'rating_adj', 'bias_dev_adj', 'quality', 'num_events_w', 'weight_rating']])"`
Expected: `rating` en 0-100 (p. ej. 7x,x), `quality` en 0-100 o 0-1 (anotar cuál), `bias_dev_adj` en escala bias (≈ −1…+1). Fijar la conversión del método a lo observado y documentarla en la docstring.

- [ ] **Step 4: Implementar `scope_computers` y `rating_check_data`**; `rating_pool` reescrito sobre `scope_computers` sin cambiar su salida.

- [ ] **Step 5: Ejecutar los tests del Computer**

Run: `pytest tests/test_computer_unit.py -q`
Expected: PASS (incluidos `test_the_rating_pool_keeps_the_polls_of_the_excluded_pollsters` y los de `compute_ratings`).

- [ ] **Step 6: Medir con datos reales (sólo lectura) y cruzar con un cálculo independiente**

Script en el scratchpad de la sesión: `Computer(scope='es', verbose=0).build_series().rating_check_data('all')`; imprimir `n_scopes`, `seconds`, `len(pairs)`, `len(polls)`, pares por población, y `rating_check_summary(...)['summary']['all']` (Spearman, pendiente, skill). Segundo script independiente (sin `rating_check.py`): la correlación de rango de `rating` antes de e y `d_obs` recalculada desde `get_ratings` y `_event_deviations` directamente; debe coincidir al tercer decimal. Anotar cifras y segundos en `.superpowers/sdd/2026-11-04-rating-validacion/figures.md` (las lee la Task 8). Si `seconds` > 180, anotarlo: la parte sigue publicándose sólo bajo demanda (D5), no hace falta más.

- [ ] **Step 7: Commit**

```bash
git add mtpy/lib/computer.py tests/test_computer_unit.py
git commit -m "computer: rating_check_data reúne los pares casa-elección (rating antes de e frente a desviación en e) y los sondeos de la última campaña con los dos pesos; scope_computers extraído de rating_pool"
```

### Task 6: Parte `rating-check` del análisis nacional, job y API

**Files:**
- Modify: `mtpy/lib/bundle.py` (`ANALYSIS_PARTS`, `ANALYSIS_SCHEMAS`, `ANALYSIS_CSV` líneas 66-69; `SCHEMAS` ≈ línea 128)
- Modify: `mtpy/lib/publish/analysis.py` (`export_rating_check` nuevo; `publish_analysis(scope, writer, computer=None, stats=None, prov=None, verbose=0, rating_check=False)`, cuerpo ≈ líneas 262-280; `export_analysis_meta(..., rating_check: Optional[bool] = None)`)
- Modify: `mtpy/pipelines/publish/Analysis.py:48-80` (`publish_task(..., rating_check=False)`), `mtpy/jobs/Publish.py` (`run(..., rating_check=False)`, docstring, `super().run(...)`)
- Modify: `mtpy/controllers/Analysis.py:39-50` (`table=elections` para `rating-check`), docstring
- Modify: `docs/web/contrato.md` (tabla de partes ≈ línea 297, lista de CSV ≈ 314-318, tabla de la API ≈ línea 704), `README.md` (sección `publish`: parámetro `rating_check` y el procedimiento tras una elección)
- Create: `tests/fixtures/bundle/analysis-rating-check.json`
- Test: `tests/test_bundle_unit.py:57`, `tests/test_publish_unit.py` (junto a `test_publish_analysis_records_its_parameters_and_the_database`), `tests/fakes.py` (`FakeComputer.rating_check_data`), `tests/test_pipelines_publish.py`, `tests/test_jobs_publish.py`, `tests/test_web_api.py` (`test_analysis_parts_and_csv`), `tests/test_webapi_unit.py::write_analysis`

**Interfaces:**
- Consumes: `rating_check.rating_check_summary`, `simple_average_errors`; `Computer.rating_check_data` (Task 5); `bias_dev_tau`/`bias_dev_sigma` del Computer.
- Produces:
  - `bundle.ANALYSIS_PARTS = ('meta', 'house-effects', 'herding', 'poll-errors', 'error-data', 'rating-check')`, `ANALYSIS_SCHEMAS['rating-check'] = 'analysis-rating-check'`, `SCHEMAS['analysis-rating-check'] = {'summary': dict, 'quarters': list, 'scatter': list, 'fit': NULLABLE_LIST, 'dates': dict, 'params': dict}` (`NULLABLE_LIST` definido junto a `NULLABLE_STR` en `bundle.py` si no existe: lista o `None`), `ANALYSIS_CSV += ('rating-check', 'rating-check-elections')`.
  - `publish.export_rating_check(data: dict, pairs: pd.DataFrame, elections: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]`: el `data` validado y los dos CSV: `rating-check` (los pares, columnas `event_scope, event_date, population, pollster, rating, rating_adj, d_prev, d_obs, error_avg, num_events_w, quality, n_polls, n_event_polls`, diferenciales ×100 y ratings ×100 a 2 decimales) y `rating-check-elections` (`event_scope, event_date, population, n_polls, n_pollsters, mae_flat, mae_rated, diff`, 3 decimales).
  - `publish_analysis(..., rating_check=False)`: con `rating_check` y `scope == NATIONAL_SCOPE`, tras `error-data` y antes de `meta`: `data = comp.rating_check_data('all')`, `summary = rating_check_summary(data['pairs'], data['polls'], comp.bias_dev_tau, comp.bias_dev_sigma, {'n_scopes': data['n_scopes']})`, `write('rating-check', ...)`, `seconds['rating_check']`; con `rating_check` en otro ámbito: no escribe la parte ni falla, `meta.params.rating_check = False`; sin `rating_check`: no toca la parte existente (ni la borra) y `meta.params.rating_check = False`; `meta.params.rating_check = True` sólo cuando se escribió. El resultado del método añade `'rating_check': bool`.
  - Pipeline/job: `rating_check` enhebrado hasta `publish_analysis`; la línea de resumen del pipeline añade «; rating-check» cuando se escribió.
  - API: `GET /api/v1/analysis/es/rating-check` (JSON), `?format=csv` → `es-rating-check.csv`, `?format=csv&table=elections` → `es-rating-check-elections.csv`; `table=events` sigue sólo para `herding`; 404 `no csv for this part` en el resto.

- [ ] **Step 1: Tests que fallan**

```python
# tests/test_bundle_unit.py
assert bundle.ANALYSIS_PARTS == ('meta', 'house-effects', 'herding', 'poll-errors', 'error-data', 'rating-check')

# tests/test_publish_unit.py
def test_publish_analysis_writes_rating_check_only_when_asked_and_only_nationally(tmp_path):
    factory = FakeComputer()
    writer = make_writer(tmp_path)
    out = publish.publish_analysis('es', writer, computer=factory, stats=fake_stats, prov=fake_prov, rating_check=True)
    assert out['rating_check'] is True and ('rating_check_data', 'all') in [c[:2] for c in factory.calls]
    part = writer.read_json(bundle.path_analysis('es', 'rating-check'))
    assert part['schema'] == 'analysis-rating-check@1' and set(part['data']) == {'summary', 'quarters', 'scatter', 'fit', 'dates', 'params'}
    assert (tmp_path / 'site/v1/analysis/es/csv/rating-check.csv').exists() and (tmp_path / 'site/v1/analysis/es/csv/rating-check-elections.csv').exists()
    meta = writer.read_json(bundle.path_analysis('es', 'meta'))['data']
    assert meta['params']['rating_check'] is True and 'rating_check' in meta['seconds']
    out = publish.publish_analysis('es-md', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov, rating_check=True)
    assert out['rating_check'] is False and not writer.exists(bundle.path_analysis('es-md', 'rating-check'))


def test_publish_analysis_keeps_the_previous_rating_check_when_not_requested(tmp_path):
    writer = make_writer(tmp_path)
    publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov, rating_check=True)
    before = writer.read_json(bundle.path_analysis('es', 'rating-check'))
    publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    assert writer.read_json(bundle.path_analysis('es', 'rating-check')) == before
    assert writer.read_json(bundle.path_analysis('es', 'meta'))['data']['params']['rating_check'] is False
```

`FakeComputer.rating_check_data(scopes='all', **kw)` anota la llamada y devuelve `pairs`/`polls` sintéticos (3 casas × 2 elecciones; reutilizar las constantes de `tests/test_rating_check_unit.py` moviéndolas a `tests/fakes.py` si conviene) y `bias_dev_tau = .01`, `bias_dev_sigma = .014` en el namespace.

Pipeline (`tests/test_pipelines_publish.py`): `analysis(fs, 'es', rating_check=True)` → `[c[2] for c in patched if c[1] == 'analysis'][0] == {'verbose': 0, 'rating_check': True}` y la línea acaba en `; rating-check`. Job (`tests/test_jobs_publish.py`): `Publish().run(what=['analysis'], scopes=['es'], rating_check=True, ...)` llega al pipeline. API (`tests/test_web_api.py::test_analysis_parts_and_csv`): tres llamadas nuevas (JSON, `format=csv`, `format=csv&table=elections`) con `write_analysis` ampliado a la parte y sus dos CSV; `table=events` en `rating-check` → 404.

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_bundle_unit.py tests/test_publish_unit.py tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_web_api.py -q -k "rating_check or analysis_parts or ANALYSIS"`
Expected: FAIL.

- [ ] **Step 3: Implementar** bundle, `export_rating_check`, `publish_analysis`, `export_analysis_meta` (clave `rating_check` en `params`), pipeline, job, controlador y fixture `analysis-rating-check.json` (con el sobre del bundle como `analysis-herding.json` y un `data` mínimo coherente con el esquema: un par, un cuarto, `fit` nulo).

- [ ] **Step 4: Ejecutar la suite de publicación y API**

Run: `pytest tests/test_bundle_unit.py tests/test_publish_unit.py tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_jobs_update.py tests/test_web_api.py tests/test_webapi_unit.py -q`
Expected: PASS (`update` no cambia: nunca pide la parte).

- [ ] **Step 5: Documentar** `docs/web/contrato.md` (fila `rating-check` | `analysis-rating-check@1` | `summary`, `quarters`, `scatter`, `fit`, `dates`, `params`; bullet con las columnas de los dos CSV; `meta.params.rating_check`; la parte existe sólo en `es` y sólo con `rating_check`; en la tabla de la API: `table=elections` sólo CSV de `rating-check`) y `README.md` (sección `publish`: `rating_check`, y el procedimiento tras una elección: `compute '{"what":["compute","ratings"],"scopes":"all"}'` → `publish '{"what":["analysis"],"scopes":["es"],"rating_check":true}'`).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish/analysis.py mtpy/pipelines/publish/Analysis.py mtpy/jobs/Publish.py mtpy/controllers/Analysis.py docs/web/contrato.md README.md tests/fixtures/bundle/analysis-rating-check.json tests/fakes.py tests/test_bundle_unit.py tests/test_publish_unit.py tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_web_api.py tests/test_webapi_unit.py
git commit -m "publish: parte rating-check del análisis nacional (sólo con rating_check en el job publish), con sus dos CSV y la ruta table=elections; contrato y README"
```

### Task 7: La tarjeta «Cómo lo comprobamos» con cifras vivas

**Files:**
- Modify: `mtpy/lib/pages/metodo.py` (`rating_check_context()` nuevo; `metodo_section_context` rama `rating` ≈ línea 515)
- Modify: `web/templates/metodo_rating.html` (tarjeta `comprobacion` ≈ líneas 612-654; `note_keys` línea 10 → añadir `'comprobacion'` al final; nota nueva tras la de `rebano` ≈ línea 831)
- Modify: `web/dist/js/pages/metodo.js` (nube y descarga de la tarjeta)
- Test: `tests/test_pages.py` (el test de /metodo/rating que afirma `rating-backtest-table` ≈ líneas 2786-2812; nuevo `test_metodo_rating_check_card`), `tests/test_js_charts.py` (opción de la nube)

**Interfaces:**
- Consumes: parte `rating-check` (Task 6) vía `analysis_part(NATIONAL_SCOPE, 'rating-check')`; `api_url` para los CSV; `site().cached`.
- Produces: `rating_check_context() -> Optional[dict]` (caché por TTL: `site().cached(('rating-check',), ...)`; `None` sin parte): `kpis` (lista para `m.kpis`: pares, elecciones «G + A», correlación de rango con IC en `support`, pendiente con IC, skill del rating en %), `summary` (la de la parte), `quarters` (tabla por población: cuarto 1 y 4, `error_avg` a 2 decimales, `n`), `average` (por población: `mae_flat`, `mae_rated`, `diff` con IC, `better`/`worse`, `n`), `dates`, `params`, `csv` (`{'pairs': url, 'elections': url}`), `initial` (`{'scatter': rows, 'fit': fit, 'axes': {'xMin', 'xMax', 'yMin', 'yMax', 'interval'} calculados simétricos sobre los dos diferenciales, redondeados a múltiplos de 5}`). El contexto de la página expone `rating_check` y, con parte, `initial = {'state': ..., 'rating_check': initial}`.
- Plantilla: la tarjeta `comprobacion` rehecha (lead de la spec D6): párrafo del principio (rating con las elecciones anteriores, también en las pruebas); `{% if rating_check %}` KPIs → nube `<div class="chart" id="rating-check-chart">` con `tools=m.download_button('rating-check-download', 'Cómo lo comprobamos', 'Diferencial previsto frente a observado', site_host ~ ' · ' ~ author_handle, 'rating-comprobacion')` en `card_start` → tabla de cuartos `id="rating-check-quarters"` → tabla del promedio con y sin rating `id="rating-check-average"` → párrafo del backtest del modelo vivo (D1, texto fijo con fecha; placeholder «{fecha}» hasta la Task 8) → «Cómo leerlo» (`m.how`) → enlaces a los dos CSV; `{% else %}` un párrafo: la comprobación se publica con la siguiente actualización del análisis. Desaparecen la frase «no pronostica nada: es un peso», la tabla `rating-backtest-table` y el párrafo de las correlaciones 0,04 / 0,02. Nota `comprobacion` en las notas: pares, `d_obs` como evidencia contraída por sondeo y la salvedad de β, Spearman, pendiente, skill (`−τ·Φ⁻¹`), cuartos, ventana de 30 días y mínimo de 3 casas, bootstrap por casa / por elección, 2.000 réplicas.
- JS: en `paint()`, si `data.rating_check`, `renderScatter(el, rows, {x: 'd_prev', y: 'd_obs', name: 'pollster', fit, diagonal: true, axes, xName: 'Diferencial previsto', xLabel: 'Previsto', yLabel: 'Observado'})` (comprobar la firma de `renderScatter` en `charts/scatter.js`) y `wireDownload(document.getElementById('rating-check-download'), () => scatterOption(rows, opts), ...)` como `bullseye-wire.js`.

- [ ] **Step 1: Tests que fallan**

Actualizar el test existente de /metodo/rating: quitar las aserciones de `rating-backtest-table` y `<td>7,23</td><td>7,27</td>`; las claves de las notas pasan a `['error', 'consenso', 'calidad', 'evidencia', 'rating', 'rebano', 'comprobacion']` (7 notas). Añadir:

```python
def test_metodo_rating_check_card_shows_the_live_figures(api, fresh_app):
    """Sin la parte, la tarjeta anuncia la comprobación; con ella, KPIs, nube, cuartos, promedio y los dos CSV."""
    page = html(call(api, '/metodo/rating')[2])
    assert 'id="comprobacion"' in page and 'no pronostica nada' not in page and 'rating-backtest-table' not in page
    write_analysis(fresh_app.fs)            # con la parte rating-check del fixture
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/metodo/rating')[2])
    text = visible_text(page)
    assert 'id="rating-check-chart"' in page and 'id="rating-check-quarters"' in page and 'id="rating-check-average"' in page
    assert 'Correlación de rango' in text and 'Pendiente' in text
    assert '/api/v1/analysis/es/rating-check?format=csv' in page and 'table=elections' in page
    initial = json.loads(re.search(r'id="initial-data">(.*?)</script>', page).group(1))
    assert initial['rating_check']['scatter'][0]['pollster'] and initial['rating_check']['axes']['xMin'] <= 0
    assert_public_wording(page, '/metodo/rating')
```

Y en `tests/test_js_charts.py`, un test de `scatterOption` con `diagonal: true` y `fit` (dos series extra `line`) si no existe ya uno equivalente.

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `pytest tests/test_pages.py -q -k "metodo_rating" && pytest tests/test_js_charts.py -q -k scatter`
Expected: FAIL.

- [ ] **Step 3: Implementar** contexto, plantilla (con los textos provisionales, sin cifras inventadas: las frases de «Cómo leerlo» describen qué mide cada cifra), nota, JS y descarga.

- [ ] **Step 4: Ejecutar las pruebas de páginas y JS**

Run: `pytest tests/test_pages.py tests/test_js_charts.py tests/test_js_notes.py -q`
Expected: PASS (incluido `test_public_wording_uses_encuestadora_and_medio_and_short_names`).

- [ ] **Step 5: Comprobación en navegador con el fixture**

Run: el arnés del scratchpad (`check_app.py`, `BUNDLE=fixture`, puerto 8013) + captura de /metodo/rating en escritorio y móvil; comprobar nube, botón de descarga (PNG generado) y tablas. Anotar las capturas en el ledger.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/pages/metodo.py web/templates/metodo_rating.html web/dist/js/pages/metodo.js tests/test_pages.py tests/test_js_charts.py
git commit -m "web: «Cómo lo comprobamos» de /metodo/rating con cifras vivas de la parte rating-check (KPIs, nube previsto/observado con descarga, cuartos, promedio con y sin rating, nota), sin la tabla fija ni la frase «no pronostica nada»"
```

### Task 8: Resultados: método 16, textos definitivos de la tarjeta y cierre de la spec

**Files:**
- Create: `docs/analisis-2026-09/metodo-16-rating.md`
- Modify: `web/templates/metodo_rating.html` (párrafo del backtest del modelo vivo con cifras y fecha; frases de «Cómo leerlo» que dependen de un signo, con `{% if %}` sobre `rating_check.summary`), `docs/superpowers/specs/2026-11-04-rating-validacion-design.md` (sección «Resultados» y estado), `docs/analisis-2026-09/README.md` (una línea al método 16), `backtest/README.md` (enlace), memoria del proyecto
- Test: `tests/test_pages.py` (las frases condicionales con los dos signos, forzando `summary` con `monkeypatch` sobre `rating_check_context`)

**Interfaces:**
- Consumes: Task 3 (carpetas `results_flat`/`results_rated`), Task 5 Step 6 (`figures.md`), Task 7.

- [ ] **Step 1: Comprobar que los backtests terminaron** (Task 3, Step 3).

- [ ] **Step 2: Tablas de la prueba 1a**

Script en el scratchpad: leer `by_horizon.csv` y `metrics.csv` de `results_rated` y `results_flat` (es y los cinco), emparejar por (`event_date`, `horizon`): diferencias de `mae_shares`, `crps_seats`, `brier_vs`, `cov_shares95` en % y en escaños/puntos, recuento mejor / peor por métrica, y la comparación con `mae_last_poll` / `mae_mean_4w`. Criterio 17 (±2 %, cobertura no menor). Volcar las tablas en markdown.

- [ ] **Step 3: Cifras de las pruebas 1b y 2 con datos reales**

Run: `Computer('es').build_series().rating_check_data('all')` + `rating_check_summary(...)` (o la parte publicada en `site-dry` con `publish '{"what":["analysis"],"scopes":["es"],"rating_check":true,"dry_run":true}'`, que no escribe en el sitio real). Tablas por población: Spearman (IC), pendiente (IC) y pendiente en elecciones con ≥ 10 sondeos, skill de prior y rating, cuartos, promedio con y sin rating (IC, mejor / peor). Cruzar con el script independiente de la Task 5 Step 6.

- [ ] **Step 4: Escribir `metodo-16-rating.md`**: problema, método (1a, 1b, 2 con fórmulas y escalas), resultados (las tablas de los Steps 2-3), lectura, decisión sobre C2 (con qué juez y qué parámetros, o por qué no), salvedades (pocas generales; autonómicas desde 2019 en 1a; `d_obs` contraída por sondeo y la atenuación de β; el consenso de cada fecha incluye a la propia casa; referencia emparejada frente a `backtest/results`), reproducción (comandos). Enlazar desde `docs/analisis-2026-09/README.md` y `backtest/README.md`.

- [ ] **Step 5: Textos definitivos de la tarjeta**: el párrafo del backtest del modelo vivo con las cifras y la fecha de la corrida; las frases de «Cómo leerlo» que dependen del signo (correlación negativa / nula / positiva; pendiente; el promedio mejora / no cambia / empeora) escritas con `{% if %}` sobre `rating_check.summary.all` y umbrales explícitos (IC que no cruza el 0; ±2 % para el promedio), de modo que el 29-N no deje un texto falso. Test: con `monkeypatch` sobre `pages.metodo.rating_check_context` devolver dos resúmenes (uno con IC negativo, otro que cruza el 0) y afirmar la frase de cada caso.

- [ ] **Step 6: Suite completa y redacción**

Run: `pytest -q -x`
Expected: PASS (si falla `test_nowcast_regression_seed_42`/pins de `test_model.py` por la base, es lo ya conocido y de Luis: anotarlo, no refijar).

- [ ] **Step 7: Cerrar la spec** (sección «Resultados» con las tablas resumidas y la decisión sobre C2; estado «CONSTRUIDA en dev»), actualizar la memoria del proyecto (`project-rating-validacion-fase-c.md`) y commit.

```bash
git add docs/analisis-2026-09/metodo-16-rating.md docs/analisis-2026-09/README.md backtest/README.md web/templates/metodo_rating.html tests/test_pages.py docs/superpowers/specs/2026-11-04-rating-validacion-design.md
git commit -m "método 16: validación del rating (rating plano en el backtest con referencia emparejada, promedio simple con y sin rating, discriminación y pendiente de calibración), textos definitivos de la tarjeta y cierre de la spec"
```

- [ ] **Step 8: Pasos de Luis (no se automatizan)**: `publish '{"what":["analysis"],"scopes":["es"],"rating_check":true}'` contra el sitio real, revisar /metodo/rating con el bundle real (`check_app_s3.py`), integrar `dev` y desplegar antes del 17-11 o tras el 29-N; tras el 29-N: `compute '{"what":["compute","ratings"],"scopes":"all"}'` y de nuevo `publish analysis` con `rating_check`.
