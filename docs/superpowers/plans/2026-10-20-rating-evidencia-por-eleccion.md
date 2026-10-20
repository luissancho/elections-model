# Rating por elección, criterio «Implantación» y análisis autonómicos sin ciclo: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** El rating cuenta la evidencia por elecciones, la calidad previa gana el criterio «Implantación» (generales + estándares) en lugar de «Identidad» y «Estándares», el ranking muestra las generales de cada encuestadora con un filtro, y el análisis de los ámbitos recién votados se publica.

**Architecture:** El rating sigue en `Computer.pollster_ratings`/`rate_events` (`mtpy/lib/computer.py`), ahora agregando por elección antes de promediar y recibiendo la calidad previa de cada elección puntuada desde `mtpy/lib/quality.py`, que pasa a calcularla a una fecha con las generales de cada casa (`data.get_national_events`). La web lee las columnas nuevas (`num_national_events`, `num_events_w`) de `pollsters_ratings`. El backtest gana una opción para recalcular los ratings en memoria e inyectar `weight_rating` en el `Forecaster`, sin escribir en la RDS.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, Jinja2, JS de módulos ES en `web/dist/js` (tests con `node`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-rating-evidencia-por-eleccion-design.md` (aprobada, commit `f16b120`). Leerla entera antes de empezar: las fórmulas y cifras de D2 y D3 mandan.

## Global Constraints

- Rama `dev`. Luis hace commits en paralelo en el mismo árbol (`git commit -a`): antes de cada commit, `git status` y `git log -3`; añadir solo los ficheros de la tarea (`git add <rutas>`), nunca `-a`, nunca reescribir commits suyos.
- Ninguna escritura en la RDS ni en S3: nada de `save=True`, `save_catalogues()`, jobs `compute`/`publish` contra la base o S3. Las comprobaciones contra la base son de solo lectura. Para comprobar `publish`, escritor sobre el sistema de archivos local (`bundle.BundleWriter(FileSystem(tmp))`), nunca con `S3_BUCKET`.
- `backtest/results/` y los ficheros que escriben los jobs de Luis no se tocan ni se revierten. Las salidas del backtest de esta tarea van a `files/backtest/rating-por-eleccion/{antes,solo-d2,despues}`.
- Constantes: `bias_dev_tau = 0.01`, `bias_dev_sigma = 0.014` (nuevo), `k = (bias_dev_sigma / bias_dev_tau) ** 2` (≈ 1,96), `pos_decay = 0.5`, `year_decay = 0.9`, `rating_weight` del catálogo sin cambios (autonómicas 0,5), escala de calidad `per_point = 8`, `floor = 5`, rúbrica de 10 puntos: `fieldwork` 0-3, `transparency` 0-3, `independence` 0-2, `implantation` 0-2.
- Palabras públicas: «encuestadora» (nunca «casa» en /casas: `test_houses_pages_use_encuestadora`), «Confianza» (nombre público de la calidad previa), «Implantación» (criterio), «Generales» (columna y filtro). Textos en español; código, docstrings y comentarios en inglés, como el código de alrededor.
- PEP8, docstring numpy en cada función nueva o cambiada, mismo estilo que el fichero (AGENTS.md).
- Toda la suite (`pytest -q`, 1134 tests hoy) en verde al final de cada tarea.
- Las columnas nuevas de `elections.pollsters_ratings` (`num_events_w`, `num_national_events`) las crea Luis a mano; el código solo las declara en el modelo.

## Review Focus

1. Una encuestadora con muchos sondeos en una sola elección no puede acumular más evidencia que esa elección: `num_events_w` = peso de la elección y error típico previo σ (Task 3).
2. El criterio a la fecha no mira al futuro: una general celebrada el mismo día o después de la elección puntuada no cuenta; tampoco la propia elección general que se puntúa (Task 4 y Task 5).
3. Un ámbito que publicó antes un `house-effects` y ahora no tiene ciclo no deja el fichero viejo: se borra el JSON y su CSV, sin error si no existían (Task 2).
4. El filtro «Solo con experiencia en generales» y «Cargar más» no se pisan: «Cargar más» nunca destapa filas filtradas y apagar el filtro vuelve a las primeras 50 (Task 6).
5. Una encuestadora sin evaluar y con generales vale 8, no 5, y sin generales 5 (Task 4); una evaluada con `identity = 0` muestra la transparencia a 0 con la razón (Task 7).

---

### Task 1: Backtest con ratings recalculados en memoria, y corrida «antes»

**Files:**
- Modify: `mtpy/lib/forecaster.py` (constructor y construcción de la serie, ≈ líneas 94 y 453-459)
- Modify: `mtpy/lib/simulator.py` (constructor, paso a `Forecaster` ≈ línea 217)
- Modify: `mtpy/lib/backtest.py` (`run_case`, `run_backtest`, `backtest_scope`)
- Modify: `mtpy/jobs/Backtest.py` (`run`), `backtest/run_backtest.py` (CLI)
- Test: `tests/test_forecaster_unit.py`, `tests/test_backtest_unit.py`

**Interfaces:**
- Produces: `Forecaster(..., weight_ratings: Optional[pd.Series] = None)`, serie indexada por `(event_date, pollster_id)` con el `weight_rating` de cada casa en cada elección; si se pasa, sustituye al `weight_rating` de la tabla `polls` en todos los sondeos (los que no estén en el índice quedan nulos y caen al relleno por calidad, como hoy). `Simulator(..., weight_ratings=None)` lo pasa al `Forecaster`. `backtest.rating_overrides(scope: str) -> pd.Series`: `Computer(scope=scope, verbose=0).build_series()`, `compute_ratings(save=False, scopes='all')` y devuelve `comp.ratings['weight_rating']` (índice `event_date`, `pollster_id`). `run_case(..., weight_ratings=None)`, `backtest_scope(..., recompute_ratings: bool = False)` (calcula `rating_overrides` una vez por ámbito y lo pasa a cada caso), `Backtest.run(..., recompute_ratings=False)`, CLI `--recompute-ratings`. `run.json` registra `recompute_ratings`.

- [ ] **Step 1: Write the failing tests**

En `tests/test_forecaster_unit.py`, `test_weight_ratings_override_replaces_the_stored_weight`: construir la serie con el camino que ya usan los tests del fichero para un `Forecaster` sin base (fakes de `get_poll_series`/`get_pollsters`), pasar `weight_ratings = pd.Series([2.0], index=pd.MultiIndex.from_tuples([(event_date, 1)], names=['event_date', 'pollster_id']))` y comprobar que los sondeos de la casa 1 tienen `weight_rating == 2.0` y `weight == weight_over * weight_sample * 2.0`, y que los de una casa ausente del índice tienen el relleno por calidad (`log1p(q) / log1p(mean q)`).

En `tests/test_backtest_unit.py`, `test_backtest_scope_recomputes_the_ratings_once_and_passes_them_to_every_case`: con `runner` falso que registra sus kwargs y `monkeypatch` de `backtest.rating_overrides` que cuenta llamadas y devuelve una serie fija, `backtest_scope('es', out=tmp, recompute_ratings=True, runner=..., stats=..., prov=...)` llama una vez a `rating_overrides('es')`, cada caso recibe esa serie como `weight_ratings` y `run.json` tiene `recompute_ratings: true`; con `recompute_ratings=False` no se llama y los casos reciben `weight_ratings=None`.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_forecaster_unit.py::test_weight_ratings_override_replaces_the_stored_weight tests/test_backtest_unit.py::test_backtest_scope_recomputes_the_ratings_once_and_passes_them_to_every_case -v`
Expected: FAIL (argumento desconocido `weight_ratings` / `recompute_ratings`).

- [ ] **Step 3: Implement the parameters and the plumbing listed in Interfaces**

En el `Forecaster`, la sustitución va justo antes del `fillna` por calidad (línea ≈ 458): `series['weight_rating'] = pd.MultiIndex.from_frame(series[['event_date', 'pollster_id']]).map(weight_ratings)` solo en las filas de sondeos (las de resultados oficiales no llevan `weight_rating`). Documentar en el docstring que es para el backtest y que el rating de cada fila es el de su `event_date`, calculado solo con elecciones anteriores (no mira al futuro).

- [ ] **Step 4: Run the tests to verify they pass, then the whole suite**

Run: `pytest tests/test_forecaster_unit.py tests/test_backtest_unit.py -q && pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py mtpy/lib/simulator.py mtpy/lib/backtest.py mtpy/jobs/Backtest.py backtest/run_backtest.py tests/test_forecaster_unit.py tests/test_backtest_unit.py
git commit -m "Backtest: opción --recompute-ratings, ratings recalculados en memoria e inyectados en el Forecaster"
```

- [ ] **Step 6: Launch the «antes» run in the background (código actual del rating)**

Run (en segundo plano, lectura de la RDS): `python backtest/run_backtest.py --scope es --recompute-ratings --out files/backtest/rating-por-eleccion/antes`
Cronometrar; si una corrida de `es` tarda menos de 1 h, lanzar también `--scopes all --recompute-ratings --out files/backtest/rating-por-eleccion/antes`. Anotar en el informe final qué se corrió y cuánto tardó. No esperar a que acabe para seguir con la Task 2, pero no empezar la Task 3 (que cambia el algoritmo) hasta que la corrida haya cargado sus ratings: `rating_overrides` corre al principio de cada ámbito, así que basta con que el proceso haya pasado ese punto en todos los ámbitos lanzados (log de progreso), o esperar a que termine.

---

### Task 2: `publish analysis` sin sondeos en el ciclo actual (D1)

**Files:**
- Modify: `mtpy/lib/publish/analysis.py` (`publish_analysis`, `export_analysis_meta`)
- Modify: `mtpy/jobs/Publish.py` (docstring de `_analysis_scope`)
- Test: `tests/test_publish_unit.py`, `tests/test_pages.py`

**Interfaces:**
- Produces: `meta.json` del análisis con la clave nueva `house_effects: bool` (no requerida en `bundle.SCHEMAS['analysis-meta']`, para que los `meta` viejos sigan validando); `export_analysis_meta(..., house_effects: bool = True)`.

- [ ] **Step 1: Write the failing tests**

Sustituir `test_publish_analysis_without_polls_in_the_current_event_is_nothing_to_publish` por `test_publish_analysis_without_polls_in_the_current_event_publishes_the_past`: `FakeComputer(fail_effects=ValueError('No polls for es-cb 2027-05-23: nothing to forecast'))` publica `herding`, `poll-errors`, `error-data` y `meta` (4 JSON, sus CSV), no escribe `house-effects`, y `meta.data.house_effects is False`.

Añadir `test_publish_analysis_without_cycle_removes_a_stale_house_effects`: publicar primero con `FakeComputer()` (escribe `house-effects.json` y `csv/house-effects.csv`), luego con `fail_effects`: ambos ficheros ya no existen y `meta.house_effects is False`. Y que con el ámbito sin ficheros previos no falla.

Mantener `test_publish_analysis_turns_a_computer_value_error_into_nothing_to_publish` (error al construir: sigue siendo `NothingToPublish` y no se escribe nada). Añadir a `test_publish_analysis_records_its_parameters_and_the_database` que `meta['house_effects'] is True`.

En `tests/test_pages.py`, `test_casas_with_analysis_without_house_effects_shows_the_notice`: con un bundle cuyo análisis del ámbito tiene `meta` y `herding` pero no `house-effects`, `/casas?scope=…` responde 200 y la tarjeta de efectos muestra el aviso que ya muestra sin análisis (reutilizar la fixture de análisis existente borrando la parte).

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_publish_unit.py -k "publish_analysis" tests/test_pages.py::test_casas_with_analysis_without_house_effects_shows_the_notice -v`
Expected: FAIL (hoy salta `NothingToPublish`).

- [ ] **Step 3: Implement**

En `publish_analysis`: el `try` que convierte `ValueError` en `NothingToPublish` cubre solo la construcción del `Computer` y la lectura de `event_dates`/`bmaps`; `print_house_effects` va en su propio `try`: con `ValueError`, `table = None`. Con `table` no nulo se escribe `house-effects` como hoy; con `None` se borran `bundle.path_analysis(scope, 'house-effects')` y `bundle.path_analysis_csv(scope, 'house-effects')` con `writer.remove` (tolerando que no existan) y `seconds['house_effects']` no se registra. `n_pollsters` sale solo del rebaño cuando no hay efectos. Actualizar el docstring (Raises y orden de partes). La tarjeta de /casas ya trata `effects is None`; si el test de páginas falla, ajustar `casas_context` para que `analysis_part(scope, 'house-effects')` ausente dé el aviso.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `pytest tests/test_publish_unit.py tests/test_pages.py tests/test_jobs_publish.py -q && pytest -q`
Expected: PASS.

- [ ] **Step 5: Check against the real data, locally**

Run un script de comprobación (scratchpad, no en el repo) que llame a `publish.publish_analysis('es-an', bundle.BundleWriter(FileSystem(tmpdir)))` con la app iniciada (`from mtpy import mtpy; mtpy.run()`, lectura de la RDS) y lo mismo para `es-ar`, `es-cl` y `es-ex`.
Expected: cada uno devuelve un dict con `event_date` y escribe `poll-errors.json` con filas de CENTRA en `es-an` (2022-06-19 y 2026-05-17).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/publish/analysis.py mtpy/jobs/Publish.py tests/test_publish_unit.py tests/test_pages.py
git commit -m "publish analysis: publica las elecciones pasadas aunque el ciclo actual no tenga sondeos"
```

---

### Task 3: Rating con la evidencia agrupada por elección (D2)

**Files:**
- Modify: `mtpy/lib/computer.py` (`__init__` ≈ 360-475, `compute_ratings` docstring ≈ 1229-1290, `pollster_ratings` ≈ 1565-1685, nuevo `measure_event_sigma`)
- Modify: `mtpy/models/elections.py` (`PollstersRatings.meta`: `'num_events_w': 'num'`)
- Test: `tests/test_computer_unit.py`

**Interfaces:**
- Consumes: el pool de `rating_pool` y los pesos de `poll_rating_weights` (sin cambios).
- Produces: `Computer(..., bias_dev_sigma: Optional[float] = .014)`; `min_polls` desaparece del constructor y de los docstrings (solo lo usa el rating; el `min_polls` local de la función de la línea ≈ 2116 no se toca). `pollster_ratings` devuelve además `num_events_w` (internamente, antes del reescalado). `Computer.measure_event_sigma(self, pool: pd.DataFrame, rating_weights: Optional[dict] = None, min_events: int = 3) -> float` en LOR. `rparams` de `compute_ratings` y `rate_events` incluyen `num_events_w` justo después de `num_polls_w`.

- [ ] **Step 1: Write the failing tests**

En `bare_computer()` sustituir `c.min_polls` por `c.bias_dev_sigma = .014`. Tests nuevos (el pool de `pool()` tiene `bias_dev_adj` en LOR, `weight_over = weight_sample = 1`):

```python
def test_polls_of_one_election_count_as_one_election():
    """D2: A publica tres sondeos en una sola elección; su evidencia es esa elección, no tres sondeos."""
    c = bare_computer()
    rows = [('es', '2019-11-10', 'A', 10), ('es', '2019-11-10', 'A', 40), ('es', '2019-11-10', 'A', 70),
            ('es', '2019-11-10', 'B', 10)]
    out = c.rate_events(pool(rows), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    assert out.loc[1, 'num_events_w'] == pytest.approx(1.0)
    assert out.loc[1, 'num_polls'] == 3
```

(`pool()` acepta un cuarto elemento opcional en cada tupla, los días antes de la elección de ese sondeo, que manda sobre `days`; las llamadas existentes no cambian.)

```python
def test_rating_follows_the_grouped_formula():
    c = bare_computer()
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    tau, sigma = .01, .014
    se = sigma / 1.0                                  # one election: n_eff = 1
    d_post = .01 * tau ** 2 / (tau ** 2 + se ** 2)    # A deviates +0.01 (worse than the consensus)
    err_post = (1 / tau ** 2 + 1 / se ** 2) ** -.5    # `bayes_adjust` stores the posterior standard deviation
    r_adj = norm.cdf(-d_post / tau)
    k = (sigma / tau) ** 2
    assert out.loc[1, 'bias_dev_err'] == pytest.approx(err_post)
    assert out.loc[1, 'bias_dev_adj'] == pytest.approx(d_post)
    assert out.loc[1, 'rating'] == pytest.approx((1.0 * r_adj + k * .6) / (1.0 + k))
```

```python
def test_more_elections_shrink_the_standard_error():
    """Tres elecciones del mismo peso: n_eff = 3 y se = sigma / sqrt(3)."""
```
con A en `es-md 2021-05-04`, `es-cl 2021-05-04` y `es-ar 2021-05-04` (pesos de ámbito 1, mismo año): `num_events_w == 3` y `bias_dev_err == (1 / .01**2 + 3 / .014**2) ** -.5` (posterior con `se = .014 / sqrt(3)`).

```python
def test_measure_event_sigma_matches_the_weighted_spread():
```
pool con una casa en 3 elecciones con desviaciones .00, .01, .02 y otra en 2 elecciones: `measure_event_sigma(pool, {...: 1.})` es la desviación típica ponderada (pesos `W_e`) de `d_e` alrededor de la media de su casa, solo con la primera casa (≥ 3 elecciones): `sqrt(mean([.01², 0, .01²]))`.

Los tests existentes del bloque M11 se mantienen; si alguno compara `num_polls_w`, sigue valiendo.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_computer_unit.py -k "rating or sigma or election" -v`
Expected: FAIL (`num_events_w` no existe, `bias_dev_err` sigue siendo el de sondeos).

- [ ] **Step 3: Implement the grouped algorithm in `pollster_ratings`**

Pasos 1-8 de D2 de la spec, en este orden, en LOR:
- `w = weight_over * weight_sample * weight_pos`; el filtro `weight >= 1e-2` sobre el peso total de hoy se mantiene.
- Por casa y elección (`event_key`): `d_e = Σ w·bias_dev_adj / Σ w`; `W_e = weight_year * weight_scope` (constante en la elección).
- Por casa: `d = Σ W_e d_e / Σ W_e`, `num_events_w = Σ W_e`, `n_eff = (Σ W_e)² / Σ W_e²`, `bias_dev_err = bias_dev_sigma / sqrt(n_eff)`.
- `bayes_adjust` con media 0 y `std = bias_dev_tau` sobre `(d, bias_dev_err)` → `bias_dev_adj`, `bias_dev_err` posteriores (igual que hoy).
- `rating_adj = Φ(−bias_dev_adj / τ)`, relleno con `quality` sin evidencia.
- `k = (bias_dev_sigma / bias_dev_tau) ** 2`; `rating = (num_events_w·rating_adj + k·quality) / (num_events_w + k)`.
- `num_polls_w` sigue siendo `Σ` del peso total de los sondeos (informativo). Las medias descriptivas (`error_*`, `bias_*`) siguen por sondeo con el peso total, como hoy.

Reescribir los docstrings de `compute_ratings` (pasos 1-5) y `pollster_ratings`, y el comentario de la mezcla, con la nueva unidad (elecciones), σ medido 0,0137 → 0,014 y k ≈ 1,96. `measure_event_sigma` reutiliza la misma agregación `d_e`/`W_e` (función auxiliar privada compartida, sin duplicar).

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `pytest tests/test_computer_unit.py -q && pytest -q`
Expected: PASS. Si `tests/test_model.py` (regresión fijada [141, 107, 62, 8]) se ejecuta en este entorno y falla por el cambio de `weight_rating`, NO re-fijarlo aquí: se re-fija en la Task 8 con la cifra final.

- [ ] **Step 5: Check σ against the real pool (read-only)**

Run (scratchpad): `Computer(scope='es').build_series()`, `pool = comp.rating_pool(scopes con peso > 0)` filtrado a `event_date < 2026-11-29`, `comp.measure_event_sigma(pool, weights)`.
Expected: ≈ 0,0137 (la medición de la spec). Si difiere en más de 0,001, parar e informar antes de seguir.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/computer.py mtpy/models/elections.py tests/test_computer_unit.py
git commit -m "Rating: evidencia por elección (σ entre elecciones, k = (σ/τ)²) y num_events_w"
```

- [ ] **Step 7: Launch the «solo D2» backtest in the background**

Run: `python backtest/run_backtest.py --scope es --recompute-ratings --out files/backtest/rating-por-eleccion/solo-d2` (y los autonómicos si se corrieron en la Task 1). Misma regla que la Task 1: no empezar la Task 5 hasta que haya cargado sus ratings.

---

### Task 4: Rúbrica con «Implantación» y calidad a una fecha (D3, sin rating)

**Files:**
- Modify: `data/pollsters-quality-criteria.json`
- Modify: `mtpy/lib/quality.py`
- Modify: `mtpy/lib/data.py` (`get_national_events` nueva, `save_catalogues`)
- Test: `tests/test_quality_unit.py`, `tests/test_data_files.py`, `tests/test_data_unit.py`

**Interfaces:**
- Produces (en `mtpy/lib/quality.py`, sin base de datos):
  - `EVIDENCE_KEYS = ('identity', 'standards')`: columnas del CSV que no son criterios pero se validan (0-1, fuente si vale 1).
  - `evaluated_keys(criteria) -> list`: criterios sin `computed` (los que el CSV puntúa: `fieldwork`, `transparency`, `independence`).
  - `manual_points(fieldwork, transparency, independence, identity) -> int`: `fieldwork + (transparency if identity == 1 else 0) + independence`; acepta escalares o Series (vectorizado).
  - `implantation_points(n_national_events, standards) -> int`: `(n_national_events > 0) * (1 + standards)`; escalares o Series.
  - `count_national_events(national_events: pd.DataFrame, before: Optional[pd.Timestamp] = None) -> pd.Series`: índice `pollster_id`, número de fechas distintas de `national_events` (`pollster_id`, `event_date`) estrictamente anteriores a `before` (todas con `None` o `NaT`).
  - `prior_quality(pollster_ids, evaluations: pd.DataFrame, scale: dict, n_national_events: pd.Series) -> pd.DataFrame`: índice `pollster_id`, columnas `num_national_events`, `manual`, `implantation`, `points`, `quality`; las no evaluadas tienen `manual = 0` y `standards = 0`; `quality = max(floor, per_point * points)`.
  - `quality_breakdown(record: Optional[dict], criteria: dict) -> dict`: desglose de una fila de la réplica (`pollsters_quality`, cuyo `points` es el total de hoy): `{'rows': [...], 'manual', 'implantation', 'general': bool, 'transparency_void': bool}`; lo usan el perfil y la API (Task 7).
  - `read_quality` deja de calcular `points`/`quality` y añade `manual` (int).
  - `pollsters_quality_values(pollsters, quality, scale, n_national_events) -> pd.Series` (nuevo argumento obligatorio `n_national_events`).
- Produces (en `mtpy/lib/data.py`): `get_national_events() -> pd.DataFrame` con `pollster_id`, `event_date`, una fila por casa y general (`event_scope = 'es'`) con algún sondeo de `bias` no nulo y `ctype` distinto de `wban` y `exit` (nulo cuenta), sin filtro de `mtype`. Mismo patrón que `get_pollster_events` (`Polls().get_results(query=dict(filters=[...]))`).

- [ ] **Step 1: Write the failing tests**

`tests/test_quality_unit.py` (con el `App` de prueba que ya usa el fichero):
- `test_criteria_are_four_and_sum_ten`: claves `['fieldwork', 'transparency', 'independence', 'implantation']`, `max_points == 10`, `implantation` con `max 2`, `computed == 'general'` y niveles 0-2 con los textos de la spec; `evaluated_keys == ['fieldwork', 'transparency', 'independence']`.
- `test_identity_voids_transparency`: `manual_points(3, 2, 1, identity=0) == 4` y `manual_points(3, 2, 1, identity=1) == 6`.
- `test_implantation_needs_a_general_and_standards_on_top`: `(0, 1) -> 0`, `(1, 0) -> 1`, `(3, 1) -> 2`.
- `test_count_national_events_ignores_same_day_and_later_elections`: eventos de la casa 1 en 2019-11-10 y 2023-07-23; `before=2023-07-23` → 1; `before=2023-07-24` → 2; `before=None` → 2; casa sin filas → ausente (el llamador rellena 0).
- `test_prior_quality_values`: CENTRA (7 puntos de la rúbrica vieja: campo 3, transparencia 3, independencia 0, identidad 1, estándares 0) sin generales → 48; con una general → 56; GfK (campo 3, transparencia 1, independencia 2, identidad 1, estándares 1) sin generales → 48, con generales → 64; una no evaluada sin generales → 5, con generales → 8.
- `test_read_quality_validates_identity_and_standards_as_evidence`: un CSV con `standards = 2` falla con un mensaje que nombra `standards`; con `identity = 1` y fuente vacía falla nombrando `identity_source`.
- `test_quality_breakdown_splits_implantation`: fila de la réplica con `points` total; devuelve `implantation`, `national` y `transparency_void` correctos.

`tests/test_data_files.py`: si comprueba la estructura del JSON de criterios o las sumas del CSV, actualizarlo a la rúbrica nueva (el CSV no cambia de columnas).

`tests/test_data_unit.py`: `test_get_national_events_filters_scope_bias_and_ctype` con el fake de modelos del fichero: filas `es`/`es-md`, `bias` nulo, `ctype` `wban`/`exit`/nulo y `mtype` `online`; salen solo las de `es`, `bias` no nulo y `ctype` ni `wban` ni `exit`, incluida la `online`, sin duplicados por casa y fecha.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_quality_unit.py tests/test_data_files.py tests/test_data_unit.py -v`
Expected: FAIL.

- [ ] **Step 3: Edit `data/pollsters-quality-criteria.json`**

Quitar `identity` y `standards`. En `transparency`, niveles 1-3 «…, publicada por una entidad identificable…» y nota `"note": "Sin una entidad identificable detrás (razón social o institución, sede, responsables, web), la transparencia cuenta 0."`. Añadir al final:

```json
"implantation": {
  "name": "Implantación",
  "max": 2,
  "computed": "national",
  "note": "Las elecciones generales se cuentan con los sondeos publicados, a la fecha de cada elección; los estándares se comprueban con fuente.",
  "levels": {
    "0": "Solo ha sondeado elecciones autonómicas o locales.",
    "1": "Ha sondeado al menos una elección general.",
    "2": "Ha sondeado al menos una elección general y es miembro de Insights+Analytics España o de ESOMAR, o está certificada ISO 20252."
  }
}
```

`load_criteria` acepta `computed` (solo el valor `"national"`) y `note` (texto) opcionales y falla con cualquier otro valor de `computed`.

- [ ] **Step 4: Implement `quality.py` and `data.py` per Interfaces**

`save_catalogues`: `n_national_events = count_national_events(get_national_events(), pd.Timestamp(date.today()))`; la réplica guarda en `points` y `quality` los valores de hoy de `prior_quality` (las columnas de la réplica no cambian) y `pollsters.quality` sale de `pollsters_quality_values(..., n_national_events)`. El docstring del módulo `quality.py` sigue diciendo que no toca la base: las generales llegan como argumento.

- [ ] **Step 5: Run the tests, then the whole suite**

Run: `pytest tests/test_quality_unit.py tests/test_data_files.py tests/test_data_unit.py -q && pytest -q`
Expected: PASS (los tests de páginas que lean `points` del perfil pueden fallar: si fallan solo por el desglose, se arreglan en la Task 7; dejarlos en verde aquí con el mínimo cambio y anotarlo).

- [ ] **Step 6: Check the quality table against the spec (read-only)**

Run (scratchpad): `prior_quality` de las 72 evaluadas con `count_national_events(get_national_events(), hoy)`.
Expected: 43 con generales y exactamente las 31 variaciones de la tabla de D3 de la spec (CENTRA 56 → 48, GfK 64 → 48, ODEC 32 → 16, KeyData y GI Internacional 8 → 16, …). Cualquier diferencia se informa antes de seguir.

- [ ] **Step 7: Commit**

```bash
git add data/pollsters-quality-criteria.json mtpy/lib/quality.py mtpy/lib/data.py tests/test_quality_unit.py tests/test_data_files.py tests/test_data_unit.py
git commit -m "Calidad previa: criterio Implantación (generales + estándares) e identidad como condición de la transparencia"
```

---

### Task 5: El rating usa la calidad previa a la fecha de cada elección y guarda `num_national_events`

**Files:**
- Modify: `mtpy/lib/computer.py` (`compute_ratings`, `rate_events`, `pollster_ratings`)
- Modify: `mtpy/models/elections.py` (`PollstersRatings.meta`: `'num_national_events': ['int', 2]`)
- Test: `tests/test_computer_unit.py`

**Interfaces:**
- Consumes: `quality.count_national_events`, `quality.prior_quality`, `quality.read_quality`, `quality.load_criteria` (Task 4); `data.get_national_events` (Task 4).
- Produces: `rate_events(pool, targets, rating_weights, pollsters, prior: Optional[Callable[[pd.Timestamp], pd.DataFrame]] = None)`; `prior(target)` devuelve el `prior_quality` de esa fecha (`NaT` = todas las generales) y `rate_events` lo pasa a `pollster_ratings` como `pollsters` con `quality` y `num_national_events` de esa fecha. Sin `prior`, se usa `pollsters['quality']` y `num_national_events` sale nulo (los tests M11 existentes no cambian). `rparams` incluye `num_national_events` tras `num_events_w`. `compute_ratings` construye `prior` una vez: `get_national_events()`, `read_quality(load_criteria())`, la escala, y una clausura que llama a `prior_quality(pollsters['id'], evaluations, scale, count_national_events(events, target))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prior_quality_is_taken_at_the_date_of_each_target():
    """D3: la calidad de una elección puntuada solo cuenta las generales anteriores a ella."""
```
con `prior` falso que devuelve `quality` 0,40 antes de 2023-07-24 y 0,56 después, y `num_national_events` 0 / 1: puntuar `[JULY23, pd.Timestamp('2026-11-29')]`; la fila de 2023-07-23 lleva `quality == .40` y `num_national_events == 0`, la de 2026-11-29 `.56` y `1`, y la casa sin evidencia tiene `rating == quality` en cada fecha.

```python
def test_compute_ratings_builds_the_prior_from_the_national_events(monkeypatch):
```
con `monkeypatch` de `computer.get_national_events`, `computer.read_quality` y `computer.load_criteria` (fakes mínimos) y `rate_events` interceptado: el `prior` que recibe, llamado con 2023-07-23, no cuenta la general de ese día y llamado con `NaT` cuenta todas.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_computer_unit.py -k "prior" -v`
Expected: FAIL.

- [ ] **Step 3: Implement per Interfaces**

En `pollster_ratings`, `quality` y `num_national_events` salen de las columnas del `pollsters` recibido; `num_national_events` se reescala como entero (no entra en las columnas que se multiplican por 100). Docstrings de `compute_ratings` y `rate_events`: la calidad previa es la de la fecha de cada elección puntuada.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `pytest tests/test_computer_unit.py -q && pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/computer.py mtpy/models/elections.py tests/test_computer_unit.py
git commit -m "Rating: calidad previa a la fecha de cada elección y num_national_events"
```

---

### Task 6: Columna «Generales» y filtro en el ranking de /casas

**Files:**
- Modify: `mtpy/lib/pages/casas.py` (`RANKING_KEYS`, `RANKING_COLUMNS`, `COLUMN_LABELS`, descripciones de `table_columns`, `ranking_rows`)
- Modify: `web/templates/casas.html` (colgroup, celda, `data-national-events`, el control)
- Create: `web/dist/js/ranking.js`
- Modify: `web/dist/js/pages/common.js` (`wireMore`), `web/dist/js/pages/casas.js`
- Test: `tests/test_pages.py`, `tests/test_js_ranking.py` (nuevo)

**Interfaces:**
- Consumes: `num_national_events` en las filas de `query_ratings` (columna de `pollsters_ratings`, Task 5).
- Produces: `ranking.js` exporta `rankingVisibility(generals: number[], onlyNationalEvents: boolean, shown: number) -> {visible: boolean[], filtered: boolean[], more: boolean}`: `filtered[i]` = fila excluida por el filtro (`onlyNationalEvents && generals[i] === 0`), `visible` = las primeras `shown` no filtradas, `more` = quedan no filtradas sin mostrar. `wireMore` deja de destapar filas con `data-filtered`.

- [ ] **Step 1: Write the failing tests**

`tests/test_js_ranking.py` (patrón de `tests/test_js_polls.py`, se salta sin `node`):
- `test_ranking_visibility_without_filter_shows_the_first_rows`: `rankingVisibility([0, 2, 0, 1], false, 2)` → `visible [true, true, false, false]`, `filtered` todo `false`, `more true`.
- `test_ranking_visibility_with_filter_skips_rows_without_national_events`: `rankingVisibility([0, 2, 0, 1, 3], true, 2)` → `filtered [true, false, true, false, false]`, `visible [false, true, false, true, false]`, `more true`.
- `test_ranking_visibility_with_filter_and_nothing_left`: `([0, 1], true, 50)` → `more false`.

`tests/test_pages.py`:
- `test_casas_ranking_shows_the_generales_column_after_elecciones`: el orden de cabeceras del ranking es `Encuestadora, Rating, Confianza, Elecciones, Generales, Sondeos, Diferencial, Error medio, Sesgo bloques, Rebaño` y cada fila lleva `data-national-events` con su número (añadir `num_national_events` a la fixture `houses`).
- `test_casas_ranking_has_the_generales_filter`: hay `<input type="checkbox" id="ranking-national-events">` con la etiqueta «Solo con experiencia en generales», sin marcar.
- Ajustar `test_casas_glossary_explains_every_ranking_column_in_order` con la entrada de «Generales»: «En cuántas elecciones generales ha publicado sondeos antes de la próxima. Es el dato con el que se calcula el criterio «Implantación» de la confianza.».

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_js_ranking.py tests/test_pages.py -k "ranking or glossary" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Control encima de la tabla, dentro de `<div class="series-controls">` como en `casa.html`: `<label class="check"><input type="checkbox" id="ranking-national-events"> Solo con experiencia en generales</label>`, oculto sin JS (`hidden`, lo muestra el JS como hace `wireMore` con su botón). En `casas.js`, al cambiar el control: aplicar `rankingVisibility` con `shown = data-batch` del botón, poner `hidden` y `data-filtered` según el resultado y el `hidden` del botón según `more`. `wireMore` selecciona `tbody tr[hidden]:not([data-filtered])`. El estado del filtro no se guarda (la página no tiene estado de ranking que guardar). Actualizar el párrafo «Cómo leerlo» del ranking con una frase sobre la columna y el filtro.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `pytest tests/test_js_ranking.py tests/test_pages.py -q && pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/casas.py web/templates/casas.html web/dist/js/ranking.js web/dist/js/pages/common.js web/dist/js/pages/casas.js tests/test_pages.py tests/test_js_ranking.py
git commit -m "/casas: columna Generales y filtro de experiencia en generales en el ranking"
```

---

### Task 7: Tarjeta «Confianza» del perfil, rúbrica del método y API con «Implantación»

**Files:**
- Modify: `mtpy/lib/pages/casas.py` (`quality_card`)
- Modify: `web/templates/casa.html` (tarjeta de la calidad previa)
- Modify: `mtpy/lib/pages/metodo.py` (constructor de la rúbrica ≈ línea 320), `web/templates/metodo_rating.html` (lista de criterios ≈ línea 110)
- Modify: `mtpy/lib/webapi/queries.py` (`query_pollster`: desglose)
- Test: `tests/test_pages.py`, `tests/test_webapi_unit.py`, `tests/test_web_api.py`

**Interfaces:**
- Consumes: `quality.quality_breakdown`, `quality.evaluated_keys` (Task 4).
- Produces: `quality_card` con una fila por criterio de la rúbrica nueva; la fila de `implantation` lleva `national: bool` y `parts` de la fuente de estándares; la de `transparency`, `void: bool` cuando `identity = 0`. `query_pollster(...)['quality']` añade `implantation` y `general`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pages.py`:
- `test_profile_confidence_card_shows_implantation_with_its_two_parts`: perfil de una casa evaluada con generales y estándares: la tarjeta tiene la fila «Implantación 2 de 2» con «calculado a partir de los sondeos publicados» para las generales y el enlace de la fuente de estándares; una sin generales muestra «Implantación 0 de 2» con el nivel 0.
- `test_profile_confidence_card_voids_transparency_without_identity`: con `identity = 0` y `transparency = 1`, la fila «Transparencia 0 de 3» explica «Sin una entidad identificable detrás…».
- `test_metodo_rubric_lists_four_criteria_and_their_notes`: /metodo/rating muestra Trabajo de campo, Transparencia, Independencia e Implantación, con las dos notas del JSON, y no muestra «Identidad» ni «Estándares» como criterios.

`tests/test_webapi_unit.py` / `tests/test_web_api.py`: `/api/v1/pollsters/{id}` incluye `quality.implantation` y `quality.national` (ajustar el test existente del desglose).

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_pages.py -k "confidence or rubric" tests/test_webapi_unit.py tests/test_web_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement with `quality_breakdown`**

`quality_card` y `query_pollster` usan `quality_breakdown` (sin duplicar la cuenta). El texto del KPI sigue siendo «{points} de 10 puntos» con el total de la réplica.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/casas.py web/templates/casa.html mtpy/lib/pages/metodo.py web/templates/metodo_rating.html mtpy/lib/webapi/queries.py tests/test_pages.py tests/test_webapi_unit.py tests/test_web_api.py
git commit -m "Perfil, método y API: criterio Implantación y condición de identidad"
```

---

### Task 8: Texto de /metodo/rating, backtest «después», verificación y Resultados

**Files:**
- Modify: `web/templates/metodo_rating.html` (párrafo de k ≈ línea 146 y la explicación de la evidencia)
- Modify: `docs/superpowers/specs/2026-10-20-rating-evidencia-por-eleccion-design.md` (sección Resultados)
- Modify (si se mueve): `tests/test_model.py` (regresión fijada)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: todo lo anterior; las salidas `antes` y `solo-d2` de las Tasks 1 y 3.

- [ ] **Step 1: Write the failing test**

`test_metodo_rating_explains_evidence_by_election`: /metodo/rating contiene «por elecciones», «k», el valor «1,96» o «2 elecciones», «0,014» y la frase de que la calidad previa es un prior de criterios observables y no un predictor validado; ya no contiene «la calidad previa pesa como tres sondeos».

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_pages.py::test_metodo_rating_explains_evidence_by_election -v`
Expected: FAIL.

- [ ] **Step 3: Rewrite the text (D4 de la spec)**

Contenido: la evidencia se cuenta por elecciones; por qué (σ entre elecciones 0,0137 > τ 0,01 > dispersión de medias 0,006); k = (σ/τ)² ≈ 2 elecciones; ejemplo: dos autonómicas (peso 0,5 cada una) suman una elección general de evidencia; el criterio «Implantación» y por qué no se bajó el peso de las autonómicas (tabla de error de último sondeo de la spec); la calidad previa como prior de criterios observables, con la tabla de la prueba por criterio. Cada cifra del texto debe coincidir con el código (`bias_dev_sigma`, `bias_dev_tau`) o con una medición reproducida en el Step 5; si una cifra de la spec no se reproduce, el texto usa la nueva y se anota en Resultados.

- [ ] **Step 4: Run the tests**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 5: Run the «después» backtest and the read-only checks**

Run: `python backtest/run_backtest.py --scope es --recompute-ratings --out files/backtest/rating-por-eleccion/despues` (y autonómicos si se corrieron antes, con las mismas opciones).
Run (scratchpad, solo lectura): `Computer(scope='es').build_series().compute_ratings(save=False, scopes='all')`; ranking de la fila de 2026-11-29 de las casas con `num_polls > 0` (posición, rating, `num_events`, `num_national_events`, `num_events_w`); repetir la prueba por criterio de la spec (desviación centrada por elección, bootstrap por casa, 2000 réplicas, semilla 0) con la rúbrica nueva.
Expected: CENTRA fuera del primer puesto y por debajo de GAD3; `num_national_events` de CENTRA 0 y de GAD3 ≥ 6.

- [ ] **Step 6: Re-pin the model regression if it moved**

Run: `pytest tests/test_model.py -q` (si se ejecuta en este entorno; necesita la base).
Si falla solo por el cambio de ratings, re-fijar con los escaños nuevos y anotar antiguos y nuevos en Resultados.

- [ ] **Step 7: Fill «Resultados» in the spec**

Tabla `antes` / `solo-d2` / `después` por horizonte de `mae_shares`, `mae_seats`, `crps_seats`, `brier_vs` y `cov_shares95` (de `by_horizon.csv`), y lo mismo para los autonómicos si se corrieron; las 15 primeras del ranking nuevo; la tabla de calidad recalculada si difiere de D3; σ medido; la regresión. Una frase de lectura: si empeora de forma clara y consistente, decirlo.

- [ ] **Step 8: Run the whole suite and commit**

Run: `pytest -q`
Expected: PASS.

```bash
git add web/templates/metodo_rating.html tests/test_pages.py docs/superpowers/specs/2026-10-20-rating-evidencia-por-eleccion-design.md
git add tests/test_model.py   # solo si se re-fijó
git commit -m "Método del rating por elección, backtest antes/después y resultados en la spec"
```

---

## Pasos de Luis tras la integración

Los de la spec («Pasos de Luis tras la integración»): `ALTER TABLE` de `num_events_w` y `num_national_events`, `save_catalogues()`, `compute` de ratings con `scopes: all`, `publish analysis` con `scopes: all` y lo demás, y revisión de /casas, el perfil de CENTRA y /metodo/rating.
