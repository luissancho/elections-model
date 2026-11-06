# Análisis del rebaño, versión 2 — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rehacer el estudio `rebano` y su página `/analisis/rebano` para contestar, con todas las elecciones publicadas, si las encuestadoras se comportan como un rebaño y si el efecto crece al acercarse la votación.

**Architecture:** El paquete `mtpy/lib/studies/herd/` se reescribe: `stats.py` (comparaciones entre pares de sondeos de casas distintas, dispersión, comparación de tramos, permutación, signo), `data.py` (lectura multiámbito del bundle y del run en curso de `es`) y `study.py` (`compute_herd`). `mtpy/lib/publish/herd.py` exporta cuatro partes (`meta`, `summary`, `elections`, `current`); `mtpy/lib/pages/herd.py` y la plantilla las convierten en una página sin selectores, con los gráficos en HTML y CSS.

**Tech Stack:** Python 3.11, pandas, numpy, scipy.stats (`binom`), Jinja2, CSS del sitio; pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-web-analisis-rebano-v2-design.md` (léela entera antes de empezar: este plan argumenta desde ella).

## Global Constraints

- Textos públicos: «sólo» adverbio con tilde; «encuestadora» y «casa», nunca «empresa» (`assert_public_wording` de `tests/test_pages.py`).
- Nunca «se copian», «copiar», «mirar al vecino», ni ninguna explicación de intenciones (indecisos, prudencia…): la página dice lo que pasa, no por qué.
- Identificadores en inglés neutro; los textos en español.
- PEP8 con líneas de hasta 120 caracteres; docstring estilo numpy en toda función (patrón del paquete actual).
- Parámetros exactos (`data/rebano/params.json`): `scopes "all"`, `since null`, `live_scope "es"`, `max_days 365`, `pair_days 7`, `periods {"final": [0, 21], "mid": [21, 56], "far": [56, 365]}`, `min_rows 3`, `n_series 4`, `exclude ["CIS"]`, `permutations 2000`, `seed 0`.
- Tramos semiabiertos: `lo <= días < hi`, salvo `far`, que incluye 365.
- Nada de publicar en S3: los `publish` contra el bundle real los lanza Luis.
- Luis hace commits en paralelo en el mismo árbol: antes de cada commit, `git status` y `git log -3`; añadir sólo los ficheros de la tarea, por su nombre (nunca `git add -A` ni `commit -a`); nunca reescribir sus commits.
- Cada mensaje de commit termina con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- `import mtpy.lib.publish` no debe cargar `mtpy.lib.studies.herd` (test `test_importing_publish_does_not_load_the_study_package`): el paquete se importa dentro de `compute_for_publish`.

## Review Focus

- Una pareja de casas con `Σd² = 0` en el tramo A (dos casas con cifras idénticas): `R_a = 0`, el cociente sería infinito. La elección no entra en esa comparación (`compare` devuelve `None`). Test en la tarea 1.
- Un sondeo sin cifra en una serie (NaN): esa serie no genera comparación, y las demás del mismo par sí. Test en la tarea 1.
- El run en curso de `es` cuya elección ya tiene `results` publicado: `current` es `None` y no se cuenta dos veces. Test en la tarea 2.
- Una elección sin `results`, o un ámbito del manifiesto sin elecciones: se omite con su motivo en `inputs` y sin error. Test en la tarea 2.
- La página con una comparación sin elecciones (`n = 0`, por ejemplo sin `far_mid`): veredicto «SIN DATOS», sin cuadritos y sin excepción. Test en la tarea 4.

---

### Task 1: Las medidas (`stats.py`)

**Files:**
- Rewrite: `mtpy/lib/studies/herd/stats.py` (todo lo de la versión 1 se borra: residuos frente a la curva, consenso local, `house_ratio`, `accuracy`…)
- Rewrite: `tests/test_herd_stats.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `COMPARISON_COLUMNS = ['pair', 'poll_i', 'poll_j', 'days', 'period', 'name', 'd2', 'v']`
  - `period_of(days: float, periods: dict) -> Optional[str]`: `'final'|'mid'|'far'`; `None` fuera de los tramos.
  - `comparisons(polls: pd.DataFrame, names: list[str], params: dict) -> pd.DataFrame`. La entrada tiene las columnas `poll`, `pollster`, `days` (int), `n` (float) y una float por serie de `names`. La salida tiene `COMPARISON_COLUMNS`, una fila por par de sondeos de casas distintas con `|days_i − days_j| <= pair_days` y serie con las dos cifras finitas. `pair` = las dos casas ordenadas y unidas por `|`; `days` = media de los dos; `period` = `period_of(days)`; `d2 = (x_i − x_j)²`; `v = 1e4·p(1−p)(1/n_i + 1/n_j)` con `p = (x_i + x_j)/200` acotado a [0,005, 0,995].
  - `dispersion(rows: pd.DataFrame) -> Optional[float]`: `sqrt(Σd²/Σv)`; `None` sin filas.
  - `compare(rows: pd.DataFrame, a: str, b: str, min_rows: int) -> Optional[dict]`: `{pairs, rows_a, rows_b, r_a, r_b, change, falls}` sin redondear, sobre las parejas con `>= min_rows` filas en `a` y en `b`. Devuelve `None` sin parejas o con `r_a == 0`.
  - `permutation_p(rows, a, b, min_rows, permutations, seed) -> Optional[float]`: las mismas parejas que `compare`; baraja las etiquetas a/b dentro de cada pareja; estadístico `(Σd²_b/Σv_b)/(Σd²_a/Σv_a)`; `p = P(permutado <= observado)`. Vectorizado con numpy (una matriz permutaciones × filas por pareja, sin bucle de Python por permutación). `None` cuando `compare` es `None`.
  - `sign_test(n: int, k: int) -> Optional[dict]`: se conserva tal cual (`{n, k, p}`, `binom.sf(k−1, n, 0.5)`, tres cifras significativas).
  - `count(results: list[dict], alpha: float = 0.05) -> dict`: `{n, k, sign_p, clear, median_change}` sobre dicts con `falls`, `p` y `change`. `k` = los que tienen `falls`; `sign_p` = `sign_test(n, k)['p']`; `clear` = los que tienen `p < alpha`; `median_change` = mediana de `change`. Sin resultados: `{n: 0, k: 0, sign_p: None, clear: 0, median_change: None}`.

- [ ] **Step 1: Write the failing tests** in `tests/test_herd_stats.py`

```python
PARAMS = {'pair_days': 7, 'periods': {'final': [0, 21], 'mid': [21, 56], 'far': [56, 365]}}

def test_period_of_half_open():
    p = PARAMS['periods']
    assert [period_of(d, p) for d in (0, 20.5, 21, 55.5, 56, 365, 365.5)] == [
        'final', 'final', 'mid', 'mid', 'far', 'far', None]

def test_comparisons_pairs_other_houses_within_a_week():
    polls = pd.DataFrame({'poll': ['a', 'b', 'c', 'd'], 'pollster': ['A', 'B', 'A', 'C'], 'days': [30, 25, 27, 40],
                          'n': [1000.0] * 4, 'PP': [30.0, 34.0, 31.0, 30.0], 'PSOE': [25.0, np.nan, 25.0, 25.0]})
    rows = comparisons(polls, ['PP', 'PSOE'], PARAMS)
    # a-b and b-c (A|B); a-c is the same house; d is 10+ days away from b and 13 from c, 10 from a
    assert sorted(rows[['poll_i', 'poll_j']].apply(tuple, axis=1)) == [('a', 'b'), ('b', 'c')]
    ab = rows[(rows.poll_i == 'a') & (rows.poll_j == 'b')].iloc[0]
    assert (ab.pair, ab.name, ab.days, ab.period, ab.d2) == ('A|B', 'PP', 27.5, 'mid', 16.0)
    assert ab.v == pytest.approx(4.352)          # 1e4 · 0,32 · 0,68 · 2/1000
    assert set(rows.name) == {'PP'}             # PSOE missing in b: no PSOE comparison

def test_compare_and_dispersion_on_the_same_house_pairs():
    rows = pd.DataFrame({'pair': ['A|B'] * 6 + ['A|C'] * 2, 'period': ['far'] * 3 + ['final'] * 3 + ['final'] * 2,
                         'd2': [4.0] * 3 + [1.0] * 3 + [9.0] * 2, 'v': [2.0] * 8})
    out = compare(rows, 'far', 'final', 3)
    assert out['pairs'] == 1 and (out['rows_a'], out['rows_b']) == (3, 3)
    assert out['r_a'] == pytest.approx(2 ** 0.5) and out['r_b'] == pytest.approx(0.5 ** 0.5)
    assert out['change'] == pytest.approx(-0.5) and out['falls'] is True
    assert dispersion(rows[rows.period == 'far']) == pytest.approx(2 ** 0.5) and dispersion(rows.iloc[:0]) is None
    assert compare(rows, 'far', 'final', 4) is None

def test_compare_without_differences_in_a_is_none():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'period': ['far'] * 3 + ['final'] * 3,
                         'd2': [0.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    assert compare(rows, 'far', 'final', 3) is None and permutation_p(rows, 'far', 'final', 3, 200, 0) is None

def test_permutation_p_small_exact_case():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'period': ['far'] * 3 + ['final'] * 3,
                         'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    p = permutation_p(rows, 'far', 'final', 3, 2000, 0)
    assert abs(p - 0.05) < 0.015                 # exact: 1 of the C(6,3) = 20 arrangements
    assert permutation_p(rows, 'far', 'final', 3, 2000, 0) == p

def test_count_and_sign():
    results = [{'falls': True, 'p': 0.01, 'change': -0.5}, {'falls': True, 'p': 0.3, 'change': -0.1},
               {'falls': True, 'p': 0.2, 'change': -0.2}, {'falls': False, 'p': 0.9, 'change': 0.4}]
    assert count(results) == {'n': 4, 'k': 3, 'sign_p': 0.312, 'clear': 1, 'median_change': pytest.approx(-0.15)}
    assert count([]) == {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}
```

(`sign_p` 0,312 = `_sig3(5/16)`.)

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_herd_stats.py`
Expected: FAIL (ImportError: `period_of`, `comparisons`… not defined)

- [ ] **Step 3: Implement the interfaces above in `mtpy/lib/studies/herd/stats.py`**, keeping `_sig3` and `_is_number`; `scipy.stats` imported inside the functions. The module docstring describes the new measure.

- [ ] **Step 4: Run them to see them pass**

Run: `python -m pytest -q tests/test_herd_stats.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/stats.py tests/test_herd_stats.py
git commit -m "Rebaño v2: las medidas (comparaciones entre pares de casas, tramos, permutación y signo)"
```

### Task 2: Lectura y cálculo (`data.py`, `study.py`, parámetros, bundle sintético)

**Files:**
- Rewrite: `mtpy/lib/studies/herd/data.py`, `mtpy/lib/studies/herd/study.py`
- Rewrite: `data/rebano/params.json`; Modify: `data/README.md:32` (la fila de `rebano/params.json`: «ámbitos, ventana de días, días entre los dos sondeos de un par, tramos, mínimo de comparaciones por pareja, series principales, casas fuera, permutaciones y semilla»)
- Modify: `tests/fakes.py` (sustituir `write_herd_bundle` y sus constantes `HERD_*`)
- Rewrite: `tests/test_herd_data.py`, `tests/test_herd_study.py`

**Interfaces:**
- Consumes: Task 1 (`comparisons`, `compare`, `permutation_p`, `count`); `mtpy.lib.studies.labels.election_label`; `mtpy.lib.studies.ban.data.list_scopes`; `mtpy.lib.mtypes.DROP_MTYPES`; `data/es-scopes.csv` (`scode`, `name`).
- Produces:
  - `load_params(data: FileSystem) -> dict`; `MissingInput(Exception)`.
  - `@dataclass Election`: `scope`, `event_date`, `label`, `kind` (`'generales'` para `es`, `'autonómicas'` para el resto), `names` (list), `polls` (DataFrame con `poll`, `pollster`, `days`, `n` y las series), `current` (bool), `as_of` (str o None).
  - `prepare_polls(records: list, parties: list, event_date: str, params: dict) -> tuple[pd.DataFrame, list]`. Primero filtra por la ventana (fin de campo `end_date`, o `date` sin ella, a 0-`max_days` días) y quita los `mtype` de `DROP_MTYPES`. Sobre esos sondeos calcula `names = main_series(...)` (las `n_series` de mayor media; el CIS y los sondeos sin muestra cuentan aquí, como en el spike). Después se queda con `sample_size > 0`, `weight > 0` y `pollster` fuera de `exclude`. `poll` es el índice del sondeo en `records`, como str.
  - `read_elections(reader, params, names) -> tuple[list[Election], dict]`: todos los ámbitos de `list_scopes(reader)` y todas sus fechas (`>= since` cuando `since` no es nulo) con `cycle` y `results`. `inputs = {scope: {'read': [dates], 'skipped': {date: 'no cycle'|'no results'}}}`. `label = election_label(scope, names.get(scope, scope), date, month=<otra fecha del ámbito con el mismo año>)`. Orden por fecha, con `es` primero en la misma fecha.
  - `read_current(reader, params, names) -> tuple[Optional[Election], dict]`: el run `manifest.scopes[live_scope].latest`, con `polls` del run y `as_of` del `meta` del run. `None` sin manifiesto o sin run (`{'skipped': 'no manifest'|'no run'}`), o si su elección ya tiene `results` (`{'skipped': 'election published'}`). Si no, `{'run_id', 'as_of', 'event_date'}`.
  - `COMPARISONS = {'far_final': ('far', 'final'), 'far_mid': ('far', 'mid'), 'mid_final': ('mid', 'final')}` en `study.py`.
  - `@dataclass HerdStudy`: `params`, `election` (`{scope, date, days_left, as_of}` o None), `summary`, `elections` (list de filas), `current` (dict o None), `inputs`.
  - `compute_herd(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> HerdStudy`. Lanza `MissingInput('events')` sin ninguna elección pasada.
    - Filas de `elections`: una por elección pasada y comparación con resultado. Claves `scope, event_date, label, kind, comparison, pairs, rows_a, rows_b, r_a, r_b, change, falls, p`. `r_a`, `r_b` y `change` van redondeados a 2 decimales; `falls` se calcula sin redondear; `p` con `_sig3`.
    - `summary`:
      - `n_elections`: elecciones pasadas con al menos una fila.
      - `n_scopes` y `since_year` (año de la primera de esas elecciones, como str).
      - `comparisons = {key: count(filas de key)}`, con `median_change` redondeado a 2.
      - `kinds = {'generales': count(...), 'autonómicas': count(...)}` sobre las filas de `far_final`.
    - `current`, para la elección en curso:
      - `{scope, event_date, label, days_left, as_of}`;
      - `windows = {'mid': event − (mid[1] − 1) días, 'final': event − (final[1] − 1) días}` en ISO;
      - `far_mid` y `mid_final`, con la forma de una fila de `elections` (sin `kind` ni `comparison`), o `None`.

- [ ] **Step 1: Rewrite `write_herd_bundle(fs)` in `tests/fakes.py`.** Todos los sondeos llevan `sample_size 1000`, `weight 1`, `mtype None`, `start_date = end_date = date = elección − días`, partidos `['PP', 'PSOE']` y la columna `POLL_COLUMNS` de `mtpy.lib.publish`. Escribe `meta`, `cycle` y `results` de cada elección pasada. Los sondeos son `(casa, días, PP, PSOE)`:
  - `es/2019-11-10`: A 200 30 25, B 198 34 25, A 100 30 25, B 97 34 25 (lejos); A 50 30 25, B 48 33 25, A 40 30 25, B 38 33 25 (3-8 semanas); A 15 31 25, B 13 32 25, A 4 31 25, B 2 32 25 (final); CIS 14 20 40; un sondeo `mtype 'aggr'` de la casa Agg, 12 50 10.
  - `es/2023-07-23`: A 150 30 25, B 148 31 25, A 120 30 25, B 118 31 25; A 10 30 25, B 8 33 25, A 3 30 25, B 1 33 25.
  - `es-md/2021-05-04`: A 16 40 20, D 14 41 20, A 6 40 20, D 4 41 20 (sólo tramo final).
  - `es-md/2023-05-28`: un sondeo y `cycle` sin `results` (debe saltarse con `'no results'`).
  - Run en curso `HERD_RUN` de `es`, elección `2026-11-29` y `as_of 2026-11-05`, con `meta`, `polls`, manifiesto con `latest` y `headline` como hoy: A 100 30 25, B 98 34 25, A 90 30 25, B 88 34 25, A 40 30 25, B 38 32 25, A 30 30 25, B 28 32 25.
  - El manifiesto lista `es` y `es-md`. Devuelve `{'events': [(scope, date)...], 'run_id': HERD_RUN}`.

- [ ] **Step 2: Write the failing tests**

`tests/test_herd_data.py`:
```python
def test_prepare_polls_filters_and_main_series(): ...
    # from a cycle of es/2019-11-10: CIS and Agg out of the frame; names == ['PP', 'PSOE']; 12 polls; days ints
def test_read_elections_every_scope_with_results(fresh_app, tmp_path):
    # labels ['Generales 2019', 'Madrid 2021', 'Generales 2023'] in date order;
    # inputs['es-md']['skipped'] == {'2023-05-28': 'no results'}
def test_read_current_none_once_the_election_is_published(fresh_app, tmp_path):
    # publish es/2026-11-29 cycle+results → (None, {'skipped': 'election published'})
```

`tests/test_herd_study.py` (cifras calculadas a mano con el fixture):
```python
def test_compute_herd_rows(fresh_app, tmp_path):
    rows = {(r['label'], r['comparison']): r for r in compute(tmp_path).elections}
    assert set(rows) == {('Generales 2019', 'far_final'), ('Generales 2019', 'far_mid'),
                         ('Generales 2019', 'mid_final'), ('Generales 2023', 'far_final')}
    g19 = rows[('Generales 2019', 'far_final')]
    assert (g19['pairs'], g19['rows_a'], g19['rows_b'], g19['r_a'], g19['r_b'], g19['change'], g19['falls']) == (
        1, 4, 4, 1.41, 0.35, -0.75, True)
    assert abs(g19['p'] - 0.214) < 0.03           # exact 15/70
    assert rows[('Generales 2019', 'far_mid')]['change'] == -0.25
    assert rows[('Generales 2019', 'mid_final')]['change'] == -0.67
    g23 = rows[('Generales 2023', 'far_final')]
    assert (g23['r_a'], g23['r_b'], g23['change'], g23['falls'], g23['kind']) == (0.35, 1.06, 1.99, False, 'generales')

def test_compute_herd_summary(fresh_app, tmp_path):
    s = compute(tmp_path).summary
    assert (s['n_elections'], s['n_scopes'], s['since_year']) == (2, 1, '2019')
    assert s['comparisons']['far_final'] == {'n': 2, 'k': 1, 'sign_p': 0.75, 'clear': 0, 'median_change': 0.62}
    assert s['comparisons']['far_mid'] == {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.25}
    assert s['comparisons']['mid_final'] == {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.67}
    # exact permutation p: far_mid 23/70, mid_final 15/70: neither is clear
    assert s['kinds']['autonómicas'] == {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}

def test_compute_herd_current(fresh_app, tmp_path):
    study = compute(tmp_path)
    c = study.current
    assert (c['label'], c['days_left'], c['windows']) == ('Generales 2026', 24,
                                                           {'mid': '2026-10-05', 'final': '2026-11-09'})
    assert c['far_mid']['change'] == -0.5 and c['mid_final'] is None
    assert study.election == {'scope': 'es', 'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05'}

def test_compute_herd_without_elections_raises(fresh_app, tmp_path): ...  # MissingInput('events')
```

(Cuentas: en 2019, lejos `Σd² 32` y `Σv 16,204`; tramo final `Σd² 2` y `Σv 16,131`. En 2023, lejos `2 / 15,979` y final `18 / 16,131`. En el run, 3-8 semanas `8 / 16,056`. Madrid 2021 no tiene tramo lejos, así que no entra.)

- [ ] **Step 3: Run them to see them fail**

Run: `python -m pytest -q tests/test_herd_data.py tests/test_herd_study.py`
Expected: FAIL

- [ ] **Step 4: Implement `data.py`, `study.py` and `params.json`** with the interfaces above. Lo que en estos dos módulos sólo usaba la versión 1 (curvas, `read_herding`, semanas, casas) se borra.

- [ ] **Step 5: Run them to see them pass**

Run: `python -m pytest -q tests/test_herd_stats.py tests/test_herd_data.py tests/test_herd_study.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/studies/herd/data.py mtpy/lib/studies/herd/study.py data/rebano/params.json data/README.md tests/fakes.py tests/test_herd_data.py tests/test_herd_study.py
git commit -m "Rebaño v2: lectura de todas las elecciones publicadas y cálculo del estudio"
```

### Task 3: Partes publicadas (`bundle.py`, `publish/herd.py`, `publish/studies.py`, fixtures)

**Files:**
- Modify: `mtpy/lib/bundle.py:89,94,100,110,194-203` (declaraciones y `SCHEMAS` de `rebano`)
- Rewrite: `mtpy/lib/publish/herd.py`
- Modify: `mtpy/lib/publish/studies.py:37-39` (título y subtítulo), `:616-617` (exportadores)
- Modify: `mtpy/lib/pages/analisis.py:64-68` (`CSV_LABELS['rebano']`)
- Check: `docs/web/contrato.md:471` sólo nombra los estudios (no sus partes) y `deploy/README.md` no cita `rebano`: no cambian salvo que el constructor encuentre una lista de partes de `rebano`
- Fixtures: delete `tests/fixtures/bundle/rebano-cycles.json`, `rebano-weeks.json` and `rebano-houses.json`; create `rebano-elections.json` and `rebano-current.json`; rewrite `rebano-summary.json` and `rebano-meta.json`.
- Rewrite: `tests/test_publish_herd.py`; Modify: `tests/test_publish_studies.py` / `tests/test_bundle_unit.py` / `tests/test_webapi_unit.py` only where they list the old parts of `rebano`.

**Interfaces:**
- Consumes: Task 2 (`HerdStudy`, `compute_herd`).
- Produces:
  - `STUDY_PARTS['rebano'] = ('meta', 'summary', 'elections', 'current')`, `STUDY_CSV['rebano'] = ('elections',)`, `STUDY_CSV_OF['rebano'] = {'elections': ('elections',)}`.
  - `SCHEMAS`:
    - `'rebano-meta'`: `study`, `title`, `subtitle`, `scope` (NULLABLE_STR), `election` (dict/None), `params`, `inputs`, `sources`, `commit`, `dirty`, `versions`, `seconds`;
    - `'rebano-summary'`: `n_elections` int, `n_scopes` int, `since_year` NULLABLE_STR, `comparisons` dict, `kinds` dict;
    - `'rebano-elections'`: `rows` list;
    - `'rebano-current'`: `current` (dict/None).
    - La versión de esquema es la global `CONTRACT` del bundle, que no se toca. Lo de «versión 2» de la spec D1 queda en cambiar las claves.
  - `export_herd_summary(study)`, `export_herd_elections(study) -> ({'rows': rows}, {'elections': frame})`, `export_herd_current(study) -> ({'current': ...}, {})`, `export_herd_meta(study, ...)` (misma firma que hoy), y `_subtitle(summary: dict) -> str`.
  - `STUDY_TITLES['rebano'] = ('¿Se comportan las encuestadoras como un rebaño?', <plantilla>)`; `CSV_LABELS['rebano'] = {'elections': 'Comparaciones por elección: las mismas parejas de casas en cada tramo'}`.
  - `HERD_SOURCES`: sin `/casas` ni el CIS; con «Histórico de elecciones» (`/elecciones`), «Método del promedio» (`/metodo/promedio`) y «Análisis» (`/analisis`).

- [ ] **Step 1: Write the failing tests** in `tests/test_publish_herd.py`

```python
SIG = {'far_final': {'n': 23, 'k': 19, 'sign_p': 0.0013, 'clear': 3, 'median_change': -0.15},
       'far_mid': {'n': 17, 'k': 9, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.1},
       'mid_final': {'n': 29, 'k': 23, 'sign_p': 0.0012, 'clear': 2, 'median_change': -0.15}}

def test_subtitle_when_the_final_weeks_are_significant_and_the_middle_is_not():
    s = {'n_elections': 33, 'since_year': '1977', 'comparisons': SIG}
    assert _subtitle(s) == ('En 19 de las 23 elecciones con datos desde 1977, las mismas encuestadoras se parecen más '
                            'entre sí en las tres últimas semanas que lejos de la votación: por azar pasaría una vez de '
                            'cada 769. Hasta tres semanas antes, no: 9 de 17, como una moneda.')

def test_subtitle_without_significance():
    s = {'n_elections': 2, 'since_year': '2019', 'comparisons': {
        'far_final': {'n': 2, 'k': 1, 'sign_p': 0.75, 'clear': 0, 'median_change': 0.62},
        'far_mid': {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.25},
        'mid_final': {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.67}}}
    assert _subtitle(s) == ('En 1 de las 2 elecciones con datos desde 2019, las mismas encuestadoras se parecen más '
                            'entre sí en las tres últimas semanas que lejos de la votación, una proporción que podría '
                            'deberse al azar.')

def test_export_herd_parts_validate(fresh_app, tmp_path):
    # compute on write_herd_bundle; each exporter's data validates against bundle.SCHEMAS[...] (bundle.validate);
    # the elections frame has the 13 row keys as columns and 4 rows; meta['title'] is the new title;
    # meta['election'] == study.election; 'copi' not in meta['subtitle'].lower()

def test_publish_study_rebano_writes_the_four_parts(fresh_app, tmp_path):
    # publish_study('rebano', writer, data=..., compute=lambda d, w: compute_herd(reader, data)) writes
    # summary, elections, current, meta and csv/elections.csv
```

«Una vez de cada 769» sale de `round(1 / sign_p)` con el `sign_p` sin redondear de `count` (`binom.sf(18, 23, .5)`), así que redondear `sign_p` no lo cambia. Si `1/p` pasa de 1.000, se escribe con el separador de miles del sitio (`fmt_int`, sin importar `pages` desde `publish`: copiar el formato). Las demás ramas (sólo `far_final` significativa sin datos de `far_mid`; `mid_final` no significativa) las redacta el constructor con este tono; las dos de arriba son literales.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_publish_herd.py`
Expected: FAIL

- [ ] **Step 3: Implement** the declarations, schemas, exporters, title and `CSV_LABELS`; rewrite the fixtures with figures like the real ones:
  - `summary`: `SIG` above, `n_elections 33`, `n_scopes 12`, `since_year '1977'` and `kinds` with `generales {n 10, k 8}` and `autonómicas {n 13, k 11}`.
  - `elections`: `far_final` rows Generales 2015 −0,45, Madrid 2021 −0,31, Generales 2016 −0,28, Generales 2023 +0,08 and Cataluña 2021 +6,23; one `far_mid` row and one `mid_final` row.
  - `current`: `es` 2026-11-29, `days_left 24`, `as_of 2026-11-05`, `windows {'mid': '2026-10-05', 'final': '2026-11-09'}`, `far_mid` with `change −0.05` and `pairs 6`, `mid_final null`.
  - `meta`: the first subtitle above.

- [ ] **Step 4: Run the publish and bundle tests**

Run: `python -m pytest -q tests/test_publish_herd.py tests/test_publish_studies.py tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_pipelines_publish.py`
Expected: PASS

- [ ] **Step 5: Commit** (los fixtures borrados con `git rm`)

```bash
git rm tests/fixtures/bundle/rebano-cycles.json tests/fixtures/bundle/rebano-weeks.json tests/fixtures/bundle/rebano-houses.json
git add mtpy/lib/bundle.py mtpy/lib/publish/herd.py mtpy/lib/publish/studies.py mtpy/lib/pages/analisis.py tests/fixtures/bundle/rebano-*.json tests/test_publish_herd.py <los tests tocados>
git commit -m "Rebaño v2: partes publicadas (summary, elections, current) y subtítulo"
```

`publish_study` no se toca: los ficheros viejos de S3 (`studies/rebano/{cycles,weeks,houses}.json` y sus CSV) quedan huérfanos y se anotan para Luis en la tarea 5.

### Task 4: Vista de la página (`pages/herd.py`, `analisis.herd_context`)

**Files:**
- Rewrite: `mtpy/lib/pages/herd.py` (fuera todas las funciones de la versión 1)
- Modify: `mtpy/lib/pages/analisis.py:672-732` (`herd_context`)
- Test: `tests/test_pages.py` (sustituir los tests `test_herd_*` de la versión 1, l. 5661-5800)

**Interfaces:**
- Consumes: las partes de la tarea 3 (sus fixtures); `pages.herd.today()` (se conserva, inyectable con `monkeypatch`).
- Produces (todas toleran `None` y partes vacías):
  - `chance_text(p: Optional[float]) -> Optional[str]`: `None` si no hay `p`; `'menos de 1 entre 1.000'` si `p < 0.001`; `'como una moneda'` si `p > 0.2`; en otro caso `'1 entre {fmt_int(round(1/p))}'`.
  - `verdicts(summary) -> dict`: `{'q1': {'label', 'tone'}, 'q2': {...}}`.
    - `q1` (`far_final`): «SÍ, EN CONJUNTO»/`q1` si `k > n/2` y `sign_p < 0.05`; «NO»/`q3` si `k <= n/2`; «NO SE PUEDE DECIR»/`q2` en otro caso; «SIN DATOS»/`q2` con `n == 0`.
    - `q2`: «SÍ, EN LAS TRES ÚLTIMAS SEMANAS»/`q1` si `mid_final` es significativa y `far_mid` no; «SÍ, DESDE ANTES»/`q1` si lo son las dos; «NO»/`q3` si `mid_final` no lo es; «SIN DATOS»/`q2` con `mid_final.n == 0`.
    - «Significativa» quiere decir `k > n/2` y `sign_p < 0.05`.
  - `flags(meta, current) -> dict`: `has_run`, `held` (hoy ≥ día de la elección), `days_left` (desde `today()`), `year`, `election_day` («29 de noviembre»), `veda_date` (elección − 6 días), `after_veda`, `final_start` («9 de noviembre», de `windows.final`) y `final_started` (hoy ≥ esa fecha).
  - `kpis(summary, current, flags) -> list[dict]`: tres filas para `m.kpis`, con los textos de la spec D4.
    - KPI 3 con run: `value` = `change` de `mid_final` en %; sin él, el de `far_mid`; sin ninguno, `DASH`. El `support` dice qué comparación es y, antes del tramo final, «el tramo final empieza el {final_start}».
    - KPI 3 sin run: «Generales y autonómicas», `'{gen} y {aut}'` de `kinds`.
  - `bars(rows: list) -> list[dict]`: las filas `far_final` ordenadas de menor a mayor `change`, cada una con `label`, `text` (porcentaje con signo, coma decimal y el menos tipográfico «−»: `'−45 %'`, `'+623 %'`), `side` (`'down'`/`'up'`) y `width` (`min(100, |change|·100)`, en % de media anchura).
  - `squares(comparison: dict) -> dict`: `{'k', 'n', 'cells': [True]*k + [False]*(n−k), 'chance': chance_text(sign_p)}`.
  - `live_text(current, flags) -> Optional[str]`: la frase de la tarjeta 3 según la spec D4. Antes del tramo final, «El tramo de las tres últimas semanas empieza el 9 de noviembre.»; durante, «Por ahora, las mismas parejas de casas difieren un 12 % menos que entre 3 y 8 semanas antes.»; celebrada, «Elección celebrada: entra en la cuenta con el escrutinio.»; sin run, `None`.
  - Contexto de `herd_context`: `common_context` + `study_meta`, `summary`, `current`, `verdicts`, `kpis`, `bars`, `squares` (`{'far_mid': …, 'mid_final': …}`), `live`, `csv_links`, `sources`, `published_at` y las claves de `flags`. Sin `initial` (la página no dibuja con JavaScript).

- [ ] **Step 1: Write the failing tests** (funciones puras con dicts a mano, como los tests de la versión 1)

```python
def test_herd_chance_text():
    t = pages.herd.chance_text
    assert [t(None), t(0.0005), t(0.0013), t(0.05), t(0.5)] == [
        None, 'menos de 1 entre 1.000', '1 entre 769', '1 entre 20', 'como una moneda']

def test_herd_verdicts_branches():
    v = pages.herd.verdicts
    sig = {'n': 23, 'k': 19, 'sign_p': 0.0013}; flat = {'n': 17, 'k': 9, 'sign_p': 0.5}
    assert v({'comparisons': {'far_final': sig, 'far_mid': flat, 'mid_final': sig}}) == {
        'q1': {'label': 'SÍ, EN CONJUNTO', 'tone': 'q1'}, 'q2': {'label': 'SÍ, EN LAS TRES ÚLTIMAS SEMANAS', 'tone': 'q1'}}
    assert v({'comparisons': {'far_final': flat, 'far_mid': sig, 'mid_final': sig}})['q2']['label'] == 'SÍ, DESDE ANTES'
    assert v({'comparisons': {'far_final': {'n': 9, 'k': 6, 'sign_p': 0.25}, 'far_mid': flat, 'mid_final': flat}}) == {
        'q1': {'label': 'NO SE PUEDE DECIR', 'tone': 'q2'}, 'q2': {'label': 'NO', 'tone': 'q3'}}
    empty = {'n': 0, 'k': 0, 'sign_p': None}
    assert v({'comparisons': {'far_final': empty, 'far_mid': empty, 'mid_final': empty}}) == {
        'q1': {'label': 'SIN DATOS', 'tone': 'q2'}, 'q2': {'label': 'SIN DATOS', 'tone': 'q2'}}

def test_herd_bars_sorted_and_clipped():
    rows = fixture_data('rebano-elections')['rows']
    b = pages.herd.bars(rows)
    assert [x['label'] for x in b][:2] == ['Generales 2015', 'Madrid 2021'] and b[-1]['label'] == 'Cataluña 2021'
    assert (b[0]['text'], b[0]['side'], b[0]['width']) == ('−45 %', 'down', 45)
    assert (b[-1]['text'], b[-1]['side'], b[-1]['width']) == ('+623 %', 'up', 100)

def test_herd_squares():
    s = pages.herd.squares({'n': 4, 'k': 3, 'sign_p': 0.3125})
    assert s == {'k': 3, 'n': 4, 'cells': [True, True, True, False], 'chance': 'como una moneda'}

def test_herd_live_text_states(monkeypatch):
    # with fixture current: today 2026-11-06 → 'El tramo de las tres últimas semanas empieza el 9 de noviembre.';
    # today 2026-11-12 and mid_final {'change': -0.12, ...} → '… difieren un 12 % menos …';
    # today 2026-11-30 → 'Elección celebrada: entra en la cuenta con el escrutinio.'; current None → None
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_pages.py -k herd`
Expected: FAIL

- [ ] **Step 3: Implement `pages/herd.py` and `herd_context`** with the interfaces above. Reuse `fmt_int`, `fmt_num`, `DASH`, `_madrid_tz`, `MONTHS_LONG` and `_get` as the current module does.

- [ ] **Step 4: Run them to see them pass**

Run: `python -m pytest -q tests/test_pages.py -k herd`
Expected: PASS for the new function tests (the old page tests are rewritten in Task 5: skip or delete them here, so the suite does not depend on the old template)

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/herd.py mtpy/lib/pages/analisis.py tests/test_pages.py
git commit -m "Rebaño v2: vista de la página (veredictos, KPIs, barras, cuadritos, directo)"
```

### Task 5: Plantilla, estilos y limpieza

**Files:**
- Rewrite: `web/templates/analisis_rebano.html`
- Modify: `web/dist/css/site.css` (clases nuevas, por ejemplo `.herd-bars` y `.herd-squares`, con colores de la paleta existente: «se acercan» y «se separan»)
- Rewrite: `web/dist/js/pages/analisis-rebano.js` (sólo `wireCardImages(document, showError)`)
- Delete: `web/dist/js/charts/weeks.js`. Modify: `web/dist/js/charts/matrix.js` (quitar la opción `center`, que sólo usaba este estudio)
- Modify: `tests/test_js_charts.py:35,841-900` (quitar los tests de `weeks.js`, del `herdText` y de `matrixOption` con `center`)
- Modify: `web/templates/metodo_rating.html:612-615` («Si las encuestadoras se acercan en la recta final, elección a elección, está en <a href="/analisis/rebano">¿Se comportan las encuestadoras como un rebaño?</a>.») y su test (`test_metodo_rating_herding_card_links_to_rebano_only_when_published`)
- Test: `tests/test_pages.py` (tests de página de `/analisis/rebano`)

**Interfaces:**
- Consumes: el contexto de la tarea 4.
- Produces: la página. Cada tarjeta es una `section.card.shot` con el botón `shot(slug)` de la plantilla actual. Los slugs son `resumen`, `pregunta-1`, `pregunta-2`, `directo`, `como` y `datos`.

- [ ] **Step 1: Write the failing page tests** (con `publish_study(fresh_app.fs, study='rebano')` y los fixtures de la tarea 3)

```python
def test_herd_page_structure_and_wording(api, fresh_app, monkeypatch):
    herd_today(monkeypatch, date(2026, 11, 6))
    publish_study(fresh_app.fs, study='rebano')
    status, _, body = call(api, '/analisis/rebano'); page = html(body)
    assert status == 200 and '<h1 class="study-title">¿Se comportan las encuestadoras como un rebaño?</h1>' in page
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page) == [
        'SÍ, EN CONJUNTO', 'SÍ, EN LAS TRES ÚLTIMAS SEMANAS']
    assert page.count('<div class="kpi">') == 3
    text = re.sub(r'\s+', ' ', visible_text(page))
    assert '19 de 23' in text and '23 de 29' in text and '9 de 17' in text and 'como una moneda' in text
    assert '1 entre 769' in text and 'sólo en 3 la bajada es clara por sí sola' in text
    assert 'El tramo de las tres últimas semanas empieza el 9 de noviembre.' in text
    assert page.count('class="pill brand" href="/api/v1/studies/rebano/') == 1
    assert 'id="weeks-chart"' not in page and 'matrix' not in page and 'initial-data' not in page
    assert_public_wording(page, '/analisis/rebano')
    low = text.lower()
    assert all(word not in low for word in ('copi', 'vecino', 'empresa', 'indecis', 'rebaño por casa'))

def test_herd_page_bars_and_squares(api, fresh_app):
    # 5 bars in the order of pages.herd.bars; the last one carries '+623 %' and a width of 100%;
    # the mid_final block has 29 squares, 23 of them filled; the far_mid block 17 and 9

def test_herd_page_live_card_states(api, fresh_app, monkeypatch):
    # 2026-11-30: pill 'elección celebrada, pendiente del escrutinio' and the held sentence;
    # meta election None and current None: no card #directo, pill 'Evidencia histórica',
    # KPI 3 'Generales y autonómicas' with '10 y 13'

def test_herd_page_without_far_mid_elections(api, fresh_app):
    # summary.comparisons.far_mid = {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}:
    # status 200, verdict 2 is 'SÍ, EN LAS TRES ÚLTIMAS SEMANAS' or 'SIN DATOS' per pages.herd.verdicts, no squares
    # block for far_mid, and no 'None' in the visible text
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_pages.py -k "herd or rebano"`
Expected: FAIL

- [ ] **Step 3: Implement the template** following spec D4, in this order: cabecera con píldoras, título y subtítulo; KPIs; tarjeta 1 (entradilla, barras, caja de conclusión); tarjeta 2 (dos bloques de cuadritos y conclusión); tarjeta 3 «Generales {año}, en directo» sólo con run; «Cómo lo medimos»; «Datos y método» (fuentes, parámetros vivos de `meta.params`, la permutación y el signo en una frase cada uno, enlace CSV y fecha). Barras y cuadritos en HTML con anchos en `style="width: N%"`. Después, los estilos, el script, la limpieza de JS y el enlace de `/metodo/rating`.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: todo en verde (la base era de 2295 passed y 81 skipped; cambia el número de tests, no deben aparecer fallos)

- [ ] **Step 5: Check the page in a browser.** Arrancar la web local con los fixtures, o con las partes calculadas en la tarea 6 cargadas en un bundle de prueba, y mirar escritorio y móvil: barras recortadas, cuadritos en dos columnas que pasan a una en el móvil, y la descarga de cada tarjeta.

- [ ] **Step 6: Commit**

```bash
git rm web/dist/js/charts/weeks.js
git add web/templates/analisis_rebano.html web/dist/css/site.css web/dist/js/pages/analisis-rebano.js web/dist/js/charts/matrix.js web/templates/metodo_rating.html tests/test_pages.py tests/test_js_charts.py
git commit -m "Rebaño v2: la página nueva («¿Se comportan las encuestadoras como un rebaño?»)"
```

### Task 6: Aceptación con datos reales y cierre

**Files:**
- Script de comprobación en el scratchpad de la sesión (no se commitea).
- Modify: `docs/superpowers/specs/2026-11-06-web-analisis-rebano-v2-design.md` (sección «Cierre»)
- Modify: memoria `project-analisis-rebano.md`

- [ ] **Step 1: Compute against Luis's bundle without publishing.** Usar `mtpy.run()` para el `app.fs` y llamar a `compute_herd(BundleReader(app.fs, prefix=bundle.PREFIX), FileSystem(app.get('datapath')))`, cronometrando con `time.perf_counter`.
Expected: `summary.comparisons`: `far_final` 19 de 23, `far_mid` 9 de 17 y `mid_final` 23 de 29, con `sign_p` 0,0013 / 0,5 / 0,0012. Si el bundle ha cambiado desde el 2026-11-06 y difieren, explicar de dónde sale cada diferencia. `clear` de `far_final` en 3 (±1 por las 2.000 permutaciones). Tiempo por debajo de 120 s.

- [ ] **Step 2: Render the page with those parts** (patrón `herd_parts.py` + `test_zz_tmp_live.py` del scratchpad: exportar las partes a JSON, publicarlas en un `fresh_app` de test y volcar el texto visible) y leer todos los textos con las cifras reales.

- [ ] **Step 3: Write the «Cierre»** de la spec (qué existe, decisiones tomadas en la ejecución, lo no verificado) y actualizar la memoria. Pasos de Luis:
  - `python job.py publish '{"what":["study"],"study":"rebano"}'`;
  - borrar de su S3 `studies/rebano/{cycles,weeks,houses}.json` y `studies/rebano/csv/{cycles,weeks,houses}.csv`;
  - desplegar la imagen para que el `update` diario lo republique;
  - tras el 29-N, `publish event` de `es/2026-11-29` y una republicación.

- [ ] **Step 4: Commit the spec**

```bash
git add docs/superpowers/specs/2026-11-06-web-analisis-rebano-v2-design.md
git commit -m "Spec rebaño v2: cierre"
```
