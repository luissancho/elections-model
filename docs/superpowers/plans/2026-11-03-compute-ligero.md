# Compute ligero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el compute que sigue a una carga de sondeos recalcule sólo lo que esos sondeos cambian (modo `light`: pesos y rebaño), que el pipeline elija solo entre `light` y `full` según lo pendiente, y que `average_errors` sea una acción aparte del job `compute`.

**Architecture:** `compute_scope` gana un parámetro `mode` (`light`/`full`) y `average_errors` sale a `average_errors_scope`; una función pura `resolve_mode` aplica la regla `auto` (`full` con `overwrite` o con una elección pendiente que tiene resultado oficial, `light` si no) con las fechas de `pending_events` y de la nueva `data.result_dates`. El pipeline `compute.Compute` resuelve el modo en `prepare_task` (después del salto por «nada pendiente», que sigue dependiendo sólo de `overwrite`) y lo dice en la línea de resumen; un pipeline hermano `AverageErrors` y la acción `average_errors` del job `compute` cubren la tabla del método 15. `update` no cambia de secuencia: con `auto` hace `light` a diario.

**Tech Stack:** Python 3.11, pandas, el framework MTpy del repositorio (`SequenceJob`, `Pipeline`, `Model`), pytest con las fábricas falsas de `tests/`.

**Spec:** `docs/superpowers/specs/2026-11-03-compute-ligero-design.md`

## Global Constraints

- Identificadores en inglés (`mode`, `light`, `full`, `auto`, `mode_reason`, `result_dates`, `average_errors_scope`, `resolve_mode`, `check_mode`); cabeceras de consola en castellano, estados y razones en inglés (`skipped (nothing loaded since the last compute)`, `polls pending`, `results pending`, `overwrite`, `requested`).
- Ningún test toca la base del `.env` ni escribe en `backtest/results`; todo acceso a la base se sustituye por fakes o `monkeypatch`. Los jobs contra la base (compute, average_errors) los lanza Luis.
- `mtpy/core/worker.py` no se toca. `job.py`, `RUN_JOB` y `deploy/ecs/job.py` no cambian de interfaz. Sin cambios de esquema.
- `overwrite` sigue siendo el único interruptor que evita el salto por «nada pendiente»; `mode` sólo elige las fases.
- Luis commitea en paralelo en el mismo árbol: antes de cada commit, `git status` y `git add` sólo de los ficheros de la tarea; nunca `git add -A` ni `git commit -a`; no tocar ficheros que él tenga modificados.
- Docstrings en inglés en el código (estilo numpy, como los vecinos, líneas de 120); README y spec en castellano; «sólo» con tilde en los textos en castellano.
- Los commits terminan con `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Suite completa al final de cada tarea: `pytest tests -q -m "not integration"` (o la marca que use `pytest.ini`; comprobar con `grep -n integration pytest.ini setup.cfg pyproject.toml`).

## Review Focus

1. Primer compute tras el `ALTER` de las marcas (todo pendiente, incluidas las elecciones con resultado): el modo ha de ser `full`, no `light`. Test en la tarea 3 (`test_compute_results_pending_makes_it_full`).
2. `mode` forzado en un ámbito sin nada pendiente y sin `overwrite`: `skipped`, no computa (el salto va antes que el modo). Test en la tarea 3 (`test_a_forced_mode_does_not_avoid_the_skip`).
3. Un fallo de la consulta de resultados (`result_dates`) ha de ser un resultado `failed` del ámbito, no una excepción que tire el job. Test en la tarea 3 (`test_a_failing_result_dates_is_a_failed_result`).
4. `light` no llama a `update_featured`, deja `featured` nulo y la línea no dice «featured» ni «dry run» cuando escribió. Tests en la tarea 2 (`test_compute_scope_light_runs_weights_and_herding_only`) y 3 (`test_compute_series_line_and_result`).
5. `update` con un `mode` inválido ha de fallar antes de cargar nada (no después de media hora de compute). Test en la tarea 4 (`test_update_rejects_an_unknown_mode_before_loading`).

---

### Task 1: `data.result_dates` y `get_events` sobre ella

**Files:**
- Modify: `mtpy/lib/data.py:904-927` (`get_events`) y una función nueva debajo de `pending_events` (línea 975)
- Test: `tests/test_data_marks.py`

**Interfaces:**
- Consumes: `EventsData` (modelo, ya importado en `data.py`), `EventsData.get_results(query: dict, formatted: bool)`.
- Produces: `data.result_dates(scope: str) -> list[str]`: fechas `YYYY-MM-DD` ordenadas y sin repetir de las elecciones del ámbito con resultado oficial, leídas con `EventsData().get_results(query=dict(filters=["scope = '<scope>'", 'region_id = 0', 'votes IS NOT NULL']), formatted=True)` (la consulta que hoy hace `get_events` por dentro). `get_events` pasa a calcular `has_results = events['date'].isin(pd.to_datetime(result_dates(scope)))`.

- [ ] **Step 1: Escribir los tests que fallan** (añadir al final de `tests/test_data_marks.py`; `FakeEvents` del fichero sirve también de `EventsData` falso)

```python
def test_result_dates_reads_the_whole_scope_rows_with_votes(monkeypatch):
    fake = FakeEvents(rows=[{'date': pd.Timestamp('2023-07-23')}, {'date': pd.Timestamp('2019-11-10')},
                            {'date': pd.Timestamp('2023-07-23')}])
    monkeypatch.setattr(data, 'EventsData', lambda: fake)
    assert data.result_dates('es') == ['2019-11-10', '2023-07-23']
    assert fake.queries == [{'filters': ["scope = 'es'", 'region_id = 0', 'votes IS NOT NULL']}]


def test_result_dates_without_rows_is_empty(monkeypatch):
    monkeypatch.setattr(data, 'EventsData', lambda: FakeEvents())
    assert data.result_dates('es-xx') == []


def test_get_events_flags_the_results_through_result_dates(monkeypatch):
    fake = FakeEvents(rows=[{'date': pd.Timestamp('2023-07-23'), 'scope': 'es', 'name': 'Julio 2023', 'featured': True,
                             'loaded_at': None, 'computed_at': None},
                            {'date': pd.Timestamp('2026-11-29'), 'scope': 'es', 'name': 'Próximas', 'featured': False,
                             'loaded_at': None, 'computed_at': None}])
    monkeypatch.setattr(data, 'Events', lambda: fake)
    asked = []
    monkeypatch.setattr(data, 'result_dates', lambda scope: asked.append(scope) or ['2023-07-23'])
    events = data.get_events('es')
    assert asked == ['es'] and list(events['has_results']) == [True, False]  # fechas contra fechas, no cadenas
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_data_marks.py -q -k "result_dates or flags_the_results"`
Expected: FAIL con `AttributeError: module 'mtpy.lib.data' has no attribute 'result_dates'` (los dos primeros) y el tercero por `has_results` (hoy consulta `EventsData` directamente).

- [ ] **Step 3: Implementar `result_dates(scope: str) -> list[str]` en `mtpy/lib/data.py` y hacer que `get_events` la use**

Docstring numpy en inglés («Elections of a scope with an official result: a row of the whole scope with votes in `events_data`»). `get_events` deja de consultar `EventsData` y hace `events['has_results'] = events['date'].isin(pd.to_datetime(result_dates(scope)))`.

- [ ] **Step 4: Comprobar que pasan, con los tests vecinos de `update_featured`**

Run: `pytest tests/test_data_marks.py -q`
Expected: PASS (todos).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/data.py tests/test_data_marks.py
git commit -m "data: result_dates(scope), las elecciones del ámbito con resultado oficial, y get_events la usa para has_results (fechas contra fechas)"
```

---

### Task 2: Librería: modos de `compute_scope`, `average_errors_scope`, `resolve_mode` y `check_mode`

**Files:**
- Modify: `mtpy/lib/compute.py` (docstring del módulo, imports línea 18, constantes 21-24, `compute_scope` 57-147; función nueva junto a `rate_scope`)
- Modify: `load/run_load.py:284-292` (sólo el docstring de `compute_scope`, que ya no nombra `average_errors`)
- Test: `tests/test_compute_unit.py`

**Interfaces:**
- Consumes: `Computer` (fases `compute_weights(save, overwrite)`, `compute_errors`, `compute_deviations`, `compute_drift`, `compute_house_effects`, `compute_herding`, `compute_average_errors(save)`), `update_featured`, `db_now`, `mark_computed`, `pending_events`, `result_dates` (tarea 1).
- Produces (en `mtpy/lib/compute.py`):
  - `COMPUTE_STEPS = ('weights', 'featured', 'errors', 'deviations', 'drift', 'house_effects', 'herding')`.
  - `MODES = {'light': ('weights', 'herding'), 'full': COMPUTE_STEPS}`; `AUTO = 'auto'`; `AVERAGE_ERRORS_STEP = 'average_errors'`; `RATINGS_STEP = 'ratings'`.
  - `check_mode(mode: str) -> str`: devuelve `mode` si es `auto`, `light` o `full`; si no, `ValueError('compute: unknown mode "{}" (expected auto, light or full)')`.
  - `resolve_mode(mode: str, overwrite: bool, pending: Optional[list[str]], results: Optional[list[str]]) -> tuple[str, str]`: `check_mode` y después la tabla de la spec: modo forzado → `(mode, 'requested')`; `overwrite` → `('full', 'overwrite')`; `set(pending or ()) & set(results or ())` no vacío → `('full', 'results pending')`; si no → `('light', 'polls pending')`.
  - `compute_scope(scope, save=True, verbose=0, progress=None, computer=None, featured=None, mark=None, clock=None, mode='full') -> dict`: `mode` fuera de `MODES` → `ValueError('compute: unknown mode "{}" (expected light or full)')`. `full`: la secuencia de hoy sin `average_errors`. `light`: `comp = fresh()`, `weights` (con `overwrite=True`) y, con `save`, `comp = fresh()`, `herding`, `mark(scope, started_at)`; `featured` no se llama. El resultado lleva `n_polls`, `n_events`, `featured` (`None` en `light` o sin `save`), `mode` y `seconds` (sólo las fases que corrieron, más `total`).
  - `average_errors_scope(scope, save=True, verbose=0, progress=None, computer=None) -> dict`: `computer(scope=scope, verbose=verbose).build_series()` y `_phase(AVERAGE_ERRORS_STEP, lambda: comp.compute_average_errors(save=save), ...)`; devuelve `n_rows` (`int(df.shape[0])`), `n_events` (`int(df['event_date'].nunique())`, 0 con el frame vacío) y `seconds` (`average_errors`, `total`). No marca `computed_at`.
  - `compute.py` reexporta `result_dates` junto a `pending_events` (línea 18, mismo `# noqa: F401`).

- [ ] **Step 1: Adaptar los tests existentes y escribir los nuevos en `tests/test_compute_unit.py`**

En `test_compute_scope_runs_the_sequence_of_the_loader_and_reports_each_phase` quitar `'average_errors'` de las dos listas y del `set` de `seconds`, y añadir `assert result['mode'] == 'full'`. En `test_compute_scope_dry_run_stops_after_the_deviations_and_saves_nothing` añadir `and result['mode'] == 'full'`. Tests nuevos:

```python
def test_compute_scope_light_runs_weights_and_herding_only():
    factory, featured, steps, marks = FakeComputer(), [], [], []
    result = compute.compute_scope('es', save=True, verbose=1, mode='light',
                                   progress=lambda step, seconds: steps.append(step), computer=factory,
                                   featured=lambda scope: featured.append(scope) or 7,
                                   mark=lambda scope, at: marks.append((scope, at)), clock=lambda: 'T0')
    # una pasada de pesos sobre el ámbito entero (copia a cada sondeo el destacado de su elección) y, con una serie
    # fresca que lee esos pesos, el rebaño; sin destacados: exigen resultado oficial y en auto el light sólo sale sin él
    assert phases(factory) == ['weights', 'herding'] and steps == ['weights', 'herding'] and featured == []
    assert [call for call in factory.calls if call[0] == 'weights'] == [('weights', True, ('overwrite', True))]
    assert factory.inits == [('es', 1)] * 2 and marks == [('es', 'T0')]
    assert result['mode'] == 'light' and result['featured'] is None and result['n_polls'] == 2
    assert set(result['seconds']) == {'weights', 'herding', 'total'}
    assert tuple(dict.fromkeys(steps)) == compute.MODES['light']


def test_compute_scope_light_dry_run_only_weighs_and_does_not_mark():
    factory, steps, marks = FakeComputer(), [], []
    result = compute.compute_scope('es-md', save=False, mode='light', progress=lambda step, seconds: steps.append(step),
                                   computer=factory, featured=lambda scope: 0,
                                   mark=lambda scope, at: marks.append(scope), clock=lambda: marks.append('clock'))
    assert phases(factory) == ['weights'] and steps == ['weights'] and marks == []
    assert factory.calls[1] == ('weights', False, ('overwrite', True)) and result['mode'] == 'light'


def test_compute_scope_rejects_an_unknown_mode_before_building_anything():
    factory = FakeComputer()
    with pytest.raises(ValueError, match='compute: unknown mode "auto" \\(expected light or full\\)'):
        compute.compute_scope('es', save=True, mode='auto', computer=factory, featured=lambda scope: 0,
                              mark=lambda scope, at: None, clock=lambda: 'T0')
    assert factory.inits == []


def test_average_errors_scope_fits_the_table_once_and_never_marks(monkeypatch):
    marks = []
    monkeypatch.setattr(compute, 'mark_computed', lambda scope, at: marks.append(scope))
    monkeypatch.setattr(compute, 'db_now', lambda: marks.append('clock'))
    factory, steps = FakeComputer(), []
    result = compute.average_errors_scope('es', save=True, verbose=2, progress=lambda step, seconds: steps.append(step),
                                          computer=factory)
    assert factory.calls == [('build_series',), ('average_errors', True)] and factory.inits == [('es', 2)]
    assert steps == ['average_errors'] and set(result['seconds']) == {'average_errors', 'total'}
    assert result['n_rows'] == 2 and result['n_events'] == 1 and marks == []
    result = compute.average_errors_scope('es', save=False, computer=factory)
    assert factory.calls[-1] == ('average_errors', False) and result['n_rows'] == 2  # en seco calcula sin guardar


def test_resolve_mode_applies_the_auto_rule():
    assert compute.resolve_mode('light', True, None, None) == ('light', 'requested')
    assert compute.resolve_mode('full', False, [], []) == ('full', 'requested')
    assert compute.resolve_mode('auto', True, None, None) == ('full', 'overwrite')
    assert compute.resolve_mode('auto', False, ['2023-07-23', '2026-11-29'], ['2019-11-10', '2023-07-23']) == ('full', 'results pending')
    assert compute.resolve_mode('auto', False, ['2026-11-29'], ['2019-11-10', '2023-07-23']) == ('light', 'polls pending')
    assert compute.resolve_mode('auto', False, [], ['2023-07-23']) == ('light', 'polls pending')
    with pytest.raises(ValueError, match='compute: unknown mode "x" \\(expected auto, light or full\\)'):
        compute.resolve_mode('x', False, [], [])


def test_check_mode_accepts_the_three_modes_and_nothing_else():
    assert [compute.check_mode(m) for m in ('auto', 'light', 'full')] == ['auto', 'light', 'full']
    with pytest.raises(ValueError, match='compute: unknown mode "Light" \\(expected auto, light or full\\)'):
        compute.check_mode('Light')
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_compute_unit.py -q`
Expected: FAIL (el primero por `'average_errors'` en las fases; los nuevos por `TypeError: unexpected keyword 'mode'` y `AttributeError` de `average_errors_scope`, `resolve_mode`, `check_mode`).

- [ ] **Step 3: Implementar en `mtpy/lib/compute.py`**

Las constantes y las cuatro funciones de **Produces**. Reescribir el docstring del módulo y el de `compute_scope` (fases por modo; `light` sin `featured` porque `update_featured` exige resultado oficial; en seco sólo pesos; la marca con el instante de inicio en los dos modos) y el de `average_errors_scope` (su prior de efectos de casa es la tabla que haya en la base: lanzarlo tras un compute `full`; no marca). En `load/run_load.py` el docstring de `compute_scope` se queda en las siete fases (ya no nombra los errores del promedio) y dice que es el modo `full`.

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_compute_unit.py tests/test_loader_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/compute.py load/run_load.py tests/test_compute_unit.py
git commit -m "compute: modos light (pesos y rebaño) y full (sin average_errors) de compute_scope, average_errors_scope como fase aparte sin marca, resolve_mode con la regla auto (full con overwrite o con una elección pendiente con resultado) y check_mode"
```

---

### Task 3: Pipelines `compute.Compute` (modo) y `compute.AverageErrors`

**Files:**
- Modify: `mtpy/pipelines/compute/Compute.py` (ACTIONS línea 10, docstring de la clase, `start` 36-40, `prepare_task` 42-64, `compute_task` 66-86, `unit` 88-91, `line_of` 93-112)
- Create: `mtpy/pipelines/compute/AverageErrors.py`
- Test: `tests/test_pipelines_compute.py`

**Interfaces:**
- Consumes: `compute.compute_scope(..., mode)`, `compute.average_errors_scope`, `compute.resolve_mode`, `compute.pending_events`, `compute.result_dates` (tarea 2).
- Produces:
  - `ACTIONS = {'catalogues': 'catálogos', 'compute': 'series', 'ratings': 'ratings', 'average_errors': 'errores del promedio'}`.
  - `Compute.start(self, dry_run=False, overwrite=False, mode='auto', **kwargs)`: cabecera `compute series: n ámbitos[ · overwrite][ · light | · full][ · dry run]` (los dos primeros sólo para `ACTION == 'compute'`; el modo sólo si `mode != 'auto'`); inicializa `self.mode = self.mode_reason = None`.
  - `Compute.prepare_task(self, chunk=None, position=(1, 1), overwrite=False, mode='auto', pending=None, results=None, **kwargs)`: para `ACTION == 'compute'`: sin `overwrite`, `pending(chunk)` (default `compute.pending_events`) vacío → `self.end({'status': 'skipped', 'action': 'series', 'reason': NOTHING_PENDING, 'warnings': []})`; después `results(chunk)` (default `compute.result_dates`) sólo si `mode == 'auto'`, sin `overwrite` y con pendientes; `self.mode, self.mode_reason = compute.resolve_mode(mode, overwrite, pending_dates, result_dates_)`. Cualquier excepción de las consultas o de `resolve_mode` → `self.end(self.failed(chunk, e, action='series', warnings=[]))`.
  - `Compute.unit(self, chunk, dry_run, computer, featured) -> dict`: `{'dry_run': dry_run, 'mode_reason': self.mode_reason, **compute.compute_scope(chunk, save=not dry_run, verbose=self.model_verbose, progress=self.progress, computer=computer, featured=featured, mode=self.mode)}`.
  - `Compute.line_of`: para `series`: `'{}: series computed ({}, {}: {}) in {:.0f} s'` con `result['mode']`, `result['mode_reason']` y las partes `'{n_polls} polls'`, `'{n_events} events'`, `'{featured} featured'` (sólo si `featured` no es `None`) y `'dry run'` (sólo si `result['dry_run']`), unidas por `', '`. Para `errores del promedio`: `'{}: average errors computed ({} rows, {} events{}) in {:.0f} s'` con `', dry run'` si `result['dry_run']`.
  - `AverageErrors(Compute)`: `label = ACTIONS['average_errors']`, `ACTION = 'average_errors'`, `unit` → `{'dry_run': dry_run, **compute.average_errors_scope(chunk, save=not dry_run, verbose=self.model_verbose, progress=self.progress, computer=computer)}`. Nunca salta (hereda el `return` temprano de `prepare_task`).

- [ ] **Step 1: Adaptar el helper y los tests existentes de `tests/test_pipelines_compute.py`**

`series()` añade `'results': lambda scope: ['2023-07-23']` a los defaults (junto a `pending`), de modo que el caso por defecto es el `light` de la rutina diaria. Cambios:

- `test_compute_overwrite_computes_anyway_and_says_so_in_the_header`: `factory.inits == [('es', 0)] * 3` se mantiene (full) y añadir `assert p.line == 'es: series computed (full, overwrite: 2 polls, 2 events, 7 featured) in 0 s'`.
- `test_compute_series_line_and_result`: `p.line == 'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s'`, `p.result['mode'] == 'light'`, `p.result['mode_reason'] == 'polls pending'`, `p.result['featured'] is None`.
- `test_compute_dry_run_saves_nothing_and_says_so`: `p.line == 'es: series computed (light, polls pending: 2 polls, 2 events, dry run) in 0 s'`.
- `test_compute_verbosity_levels`: a `verbose=2` con el default sólo salen `pesos` y `rebaño` (`'errores ('` no); con `overwrite=True` salen las siete etiquetas (sin `errores del promedio`); añadir `series(FakeComputer(), pipeline=AverageErrors, position=(1, 1), verbose=2)` y `assert '[1/1] es: errores del promedio (' in out`.
- `test_compute_reports_the_model_warnings_as_notes`: el aviso se lanza en `compute_weights` (se llama en los dos modos) y la línea es `'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s · 1 aviso'`.
- `test_compute_failure_is_a_failed_result_with_its_reason`: `FakeComputer(fail_at='herding', fail_scope='es')` y `boom herding`.

- [ ] **Step 2: Escribir los tests nuevos**

```python
def test_compute_results_pending_makes_it_full(fresh_app):
    factory = FakeComputer()
    p = series(factory, pending=lambda scope: ['2023-07-23', '2026-11-29'])  # todo pendiente: tras el ALTER
    assert p.result['mode'] == 'full' and p.result['mode_reason'] == 'results pending'
    assert factory.inits == [('es', 0)] * 3
    assert p.line == 'es: series computed (full, results pending: 2 polls, 2 events, 7 featured) in 0 s'


def test_compute_consults_the_results_only_when_the_mode_is_auto(fresh_app, capsys):
    asked = []

    def results(scope):
        asked.append(scope)
        return []

    series(FakeComputer(), results=results)
    series(FakeComputer(), results=results, overwrite=True)
    series(FakeComputer(), results=results, mode='full')
    series(FakeComputer(), results=results, pending=lambda scope: [])
    assert asked == ['es']  # sólo la primera: auto, sin overwrite y con pendientes


def test_a_forced_mode_sets_the_header_and_the_reason(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, mode='full', position=(1, 1))
    assert p.result['mode_reason'] == 'requested' and factory.inits == [('es', 0)] * 3
    assert p.line == 'es: series computed (full, requested: 2 polls, 2 events, 7 featured) in 0 s'
    assert 'compute series: 1 ámbito · full' in capsys.readouterr().out
    p = series(FakeComputer(), mode='light', position=(1, 1), pending=lambda scope: ['2023-07-23'], dry_run=True)
    assert p.line == 'es: series computed (light, requested: 2 polls, 2 events, dry run) in 0 s'
    assert 'compute series: 1 ámbito · light · dry run' in capsys.readouterr().out


def test_a_forced_mode_does_not_avoid_the_skip(fresh_app):
    factory = FakeComputer()
    p = series(factory, mode='full', pending=lambda scope: [])
    assert p.result['status'] == 'skipped' and factory.inits == []


def test_an_unknown_mode_is_a_failed_result(fresh_app):
    p = series(FakeComputer(), mode='x')
    assert p.result['status'] == 'failed' and p.result['reason'] == 'ValueError: compute: unknown mode "x" (expected auto, light or full)'


def test_a_failing_result_dates_is_a_failed_result(fresh_app):
    def broken(scope):
        raise RuntimeError('db down')

    factory = FakeComputer()
    p = series(factory, results=broken)
    assert p.result == {'status': 'failed', 'reason': 'RuntimeError: db down', 'action': 'series', 'warnings': []}
    assert factory.inits == []


def test_the_real_result_dates_is_the_default(fresh_app, monkeypatch):
    asked = []
    monkeypatch.setattr(compute_lib, 'result_dates', lambda scope: asked.append(scope) or ['2023-07-23'])
    p = Compute()
    p.run('update', chunk='es-cl', position=(1, 1), computer=FakeComputer(), featured=lambda scope: 0, verbose=0,
          pending=lambda scope: ['2027-05-23'])
    assert asked == ['es-cl'] and p.result['mode'] == 'light'


def test_average_errors_line_result_and_header(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, pipeline=AverageErrors, pending=lambda scope: [])  # nunca salta
    assert p.result['status'] == 'computed' and p.result['action'] == 'errores del promedio'
    assert p.result['n_rows'] == 2 and p.result['n_events'] == 1 and 'mode' not in p.result
    assert p.line == 'es: average errors computed (2 rows, 1 events) in 0 s'
    assert factory.calls == [('build_series',), ('average_errors', True)]
    header = capsys.readouterr().out
    assert 'compute errores del promedio: 2 ámbitos' in header and 'overwrite' not in header
    p = series(FakeComputer(), pipeline=AverageErrors, position=(1, 1), dry_run=True, overwrite=True)
    assert p.line == 'es: average errors computed (2 rows, 1 events, dry run) in 0 s'
    assert 'compute errores del promedio: 1 ámbito · dry run' in capsys.readouterr().out


def test_average_errors_failure_names_the_action(fresh_app):
    p = series(FakeComputer(fail_at='average_errors'), pipeline=AverageErrors)
    assert p.line == 'es: errores del promedio failed (RuntimeError: boom average_errors)'
```

Importar `AverageErrors` en la cabecera del fichero (`from mtpy.pipelines.compute.AverageErrors import AverageErrors`).

- [ ] **Step 3: Comprobar que fallan**

Run: `pytest tests/test_pipelines_compute.py -q`
Expected: FAIL (`ImportError` de `AverageErrors` al cargar el módulo; tras crear un fichero vacío con la clase, las líneas y los modos).

- [ ] **Step 4: Implementar `Compute.py` y crear `AverageErrors.py`**

Según **Produces**. En `prepare_task`, el salto por pendientes va antes que `results` y que `resolve_mode`; el `try` abarca las dos consultas y `resolve_mode`. El docstring de la clase describe la regla `auto` y dice que `mode` no evita el salto.

- [ ] **Step 5: Comprobar que pasan**

Run: `pytest tests/test_pipelines_compute.py tests/test_pipelines_base.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mtpy/pipelines/compute/Compute.py mtpy/pipelines/compute/AverageErrors.py tests/test_pipelines_compute.py
git commit -m "pipelines/compute: Compute resuelve el modo (auto, light o full) tras el salto por nada pendiente, con result_dates inyectable como results, lo dice en la cabecera cuando se fuerza y en la línea de resumen con su razón; pipeline AverageErrors para la acción average_errors"
```

---

### Task 4: Jobs `compute` (acción `average_errors`, `mode`) y `update` (`mode` validado)

**Files:**
- Modify: `mtpy/jobs/Compute.py` (docstring, `sequence`/`WHAT` 48-52, `run` 54-119)
- Modify: `mtpy/jobs/Update.py` (docstring, `run` 55-120)
- Modify: `mtpy/jobs/Backtest.py:127-131` (docstring: la tabla la rellena la acción `average_errors` del job `compute`)
- Test: `tests/test_jobs_compute.py`, `tests/test_jobs_update.py`

**Interfaces:**
- Consumes: `compute.check_mode` (tarea 2); los pipelines de la tarea 3.
- Produces:
  - `Compute.WHAT = ('catalogues', 'compute', 'ratings', 'average_errors')`, `sequence = {'compute': ['catalogues', 'compute', 'ratings', 'average_errors']}`.
  - `Compute.run(self, what=('compute', 'ratings'), scopes=('es',), overwrite=False, mode='auto', dry_run=False, verbose=1, computer=None, featured=None, catalogues=None, pending=None, results=None, stop_on_error=None, **kwargs)`: `check_mode(mode)` antes de `resolve_scopes`; pasa `mode=mode, results=results` a `super().run`; la variable local `results = {}` del final se renombra `outcome`. El resultado de un ámbito es el de la última acción que corrió sobre él (`average_errors` si se pidió).
  - `Update.run(self, scopes=('es',), overwrite=False, mode='auto', dry_run=False, verbose=1, fs=None, stop_on_error=None, **kwargs)`: `check_mode(mode)` justo tras los rechazos de `event_date` y `what`, antes de `fs` y `resolve_scopes`; pasa `mode=mode` a `super().run`.

- [ ] **Step 1: Adaptar los helpers y los tests existentes**

`tests/test_jobs_compute.py`: `run()` añade `'results': lambda scope: ['2023-07-23']` a los defaults. Cambios:

- `test_compute_skips_the_scopes_with_nothing_pending_unless_overwrite`: `[scope for scope, _ in factory.inits] == ['es'] * 2 and featured == []` (light); con `overwrite`, `['es'] * 3 + ['es-md'] * 3` se mantiene.
- `test_compute_runs_the_series_of_every_scope_and_then_the_ratings`: `['es'] * 2 + ['es-md'] * 2 + ['es', 'es-md']`, `featured == []`, la línea `'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s'`.
- `test_compute_isolates_a_failing_scope_and_fails_at_the_end` y `test_compute_stop_on_error_stops_before_the_ratings`: `fail_at='herding'` (`errors` no corre en `light`), mensajes `boom herding`; en el segundo `['es'] * 2` se mantiene (pesos y rebaño).
- `test_compute_dry_run_saves_nothing_and_says_so`: línea `'es: series computed (light, polls pending: 2 polls, 2 events, dry run) in 0 s'`.
- `test_compute_verbosity_levels`: a `verbose=2` con el default, `pesos`, `rebaño` y `ratings` presentes y `'errores ('` ausente; añadir una ejecución con `overwrite=True` que compruebe las siete etiquetas.
- `test_compute_reports_the_model_warnings_as_notes`: aviso en `compute_weights`; línea light.

`tests/test_jobs_update.py`: `run()` gana `results=None` y pasa `results=results or (lambda scope: ['2023-07-23'])`. Cambios: en `test_update_runs_the_three_groups_on_es_with_one_alert`, `'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s'` y `[scope for scope, _ in computer.inits] == ['es'] * 3` (uno de la carga y dos del `light`).

- [ ] **Step 2: Escribir los tests nuevos**

En `tests/test_jobs_compute.py`:

```python
def test_compute_average_errors_is_its_own_action_and_runs_last(fresh_app, scopes, capsys):
    factory = FakeComputer()
    results, featured = run(factory, what=['average_errors', 'compute'], scopes=['es'], verbose=0)
    # la serie (light: dos Computer) y después los errores del promedio (uno), aunque el JSON los liste al revés
    assert [scope for scope, _ in factory.inits] == ['es'] * 3 and phases(factory)[-1] == 'average_errors'
    assert results['es']['action'] == 'errores del promedio' and results['es']['n_rows'] == 2
    out = capsys.readouterr().out
    assert out.index('compute series: 1 ámbito') < out.index('compute errores del promedio: 1 ámbito')
    assert 'es: average errors computed (2 rows, 1 events) in 0 s' in out
    factory = FakeComputer()
    results, _ = run(factory, what='average_errors', scopes=['es'], verbose=0, pending=lambda scope: [])
    assert phases(factory) == ['average_errors'] and results['es']['status'] == 'computed'  # nunca salta


def test_compute_mode_full_and_an_unknown_mode(fresh_app, scopes, capsys):
    factory = FakeComputer()
    results, featured = run(factory, what='compute', scopes=['es'], mode='full', verbose=0)
    assert [scope for scope, _ in factory.inits] == ['es'] * 3 and featured == ['es']
    out = capsys.readouterr().out
    assert 'compute series: 1 ámbito · full' in out
    assert 'es: series computed (full, requested: 2 polls, 2 events, 7 featured) in 0 s' in out
    factory = FakeComputer()
    with pytest.raises(ValueError, match='compute: unknown mode "x" \\(expected auto, light or full\\)'):
        run(factory, what='compute', scopes=['es'], mode='x')
    assert factory.inits == []
```

Importar `phases` junto a `FakeComputer` (`from tests.test_compute_unit import FakeComputer, phases`).

En `tests/test_jobs_update.py`:

```python
def test_update_computes_full_when_a_pending_election_has_results(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'], pending=lambda scope: ['2023-07-23', '2026-11-29'])
    assert [scope for scope, _ in computer.inits] == ['es'] * 4 and results['es']['compute']['mode'] == 'full'
    assert 'es: series computed (full, results pending: 2 polls, 2 events, 7 featured) in 0 s' in capsys.readouterr().out


def test_update_passes_a_forced_mode_to_the_compute(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'], mode='full')
    out = capsys.readouterr().out
    assert 'compute series: 1 ámbito · full' in out and results['es']['compute']['mode_reason'] == 'requested'


def test_update_rejects_an_unknown_mode_before_loading(fresh_app, tmp_path, scopes, events):
    loader = FakeLoader()
    with pytest.raises(ValueError, match='compute: unknown mode "x" \\(expected auto, light or full\\)'):
        run(tmp_path, loader=loader, scopes=['es'], mode='x')
    assert loader.calls == []
```

(Comprobar en `tests/test_load_unit.py` el atributo de `FakeLoader` que anota las llamadas; si no es `calls`, usar el que sea.)

- [ ] **Step 3: Comprobar que fallan**

Run: `pytest tests/test_jobs_compute.py tests/test_jobs_update.py -q`
Expected: FAIL (`unknown what "average_errors"`, `mode` no aceptado o no validado, líneas).

- [ ] **Step 4: Implementar los dos jobs y el docstring de `Backtest.py`**

Según **Produces**. Docstring de `Compute`: fases por modo, la regla `auto`, `mode` no evita el salto (para recalcular sin nada pendiente, `overwrite`), `average_errors` como acción aparte (coste ≈ 37 min en `es`, sólo la leen las opciones del método 15; lanzarla tras un compute `full`), ejemplos `python job.py compute '{"what":["average_errors"],"scopes":"all"}'` y `python job.py compute '{"what":["compute"],"scopes":["es"],"mode":"full"}'`. Docstring de `Update`: la actualización diaria hace `light`; `full` con `overwrite` o con un resultado pendiente; `"mode": "full"` para un día con sondeos nuevos en el que se quiere la serie entera, sin nada pendiente se salta igual; ejemplo `python job.py update '{"scopes":["es"],"mode":"full"}'`.

- [ ] **Step 5: Comprobar que pasan y la suite entera**

Run: `pytest tests/test_jobs_compute.py tests/test_jobs_update.py -q` y después `pytest tests -q -m "not integration"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mtpy/jobs/Compute.py mtpy/jobs/Update.py mtpy/jobs/Backtest.py tests/test_jobs_compute.py tests/test_jobs_update.py
git commit -m "jobs/compute y update: parámetro mode (auto, light o full) validado antes de empezar y pasado a los pipelines con results inyectable; average_errors como acción propia de compute, la última; update hace light a diario y full con overwrite o con un resultado pendiente"
```

---

### Task 5: Documentación

**Files:**
- Modify: `README.md:36-70` (secciones `update` y `compute`)
- Modify: `deploy/README.md:655-755` («Calcular las series y los ratings»), `1577-1672` («Actualizar»), `1733-` («Duraciones medidas»)
- Modify: `docs/analisis-2026-09/metodo-15-calibracion.md:26,120` (y cualquier otra frase «el job `compute` tiene que rellenar»: `grep -n "job \`compute\`" docs/analisis-2026-09/metodo-15-calibracion.md`)
- Modify: `docs/superpowers/specs/2026-11-03-job-update-design.md:1-8` (enmienda)
- Modify: `docs/superpowers/specs/2026-11-03-compute-ligero-design.md` (estado: construida, desviaciones)

**Interfaces:** ninguna de código.

- [ ] **Step 1: `README.md`**

Sección `compute`: fila `what` con `compute` (fases por modo: `light` pesos y rebaño; `full` pesos, destacados, pesos, errores, desviaciones, deriva, efectos de casa y rebaño), `ratings` y `average_errors` (la tabla del método 15, ≈ 37 min en `es`, sólo la leen `error_model='average'` y `drift_scale='auto'`); fila nueva `mode` (`auto` por defecto: `full` con `overwrite` o cuando una elección pendiente tiene resultado, `light` si lo pendiente son sólo sondeos; `light`/`full` fuerzan las fases pero no evitan el salto); el párrafo de las líneas (`series computed (light, polls pending: ...)`); ejemplos `python job.py compute '{"what":["average_errors"],"scopes":"all"}'` y `python job.py compute '{"what":["compute"],"scopes":["es"],"mode":"full"}'`. Sección `update`: una frase («a diario la serie va en modo `light`, pesos y rebaño; `full` con `overwrite` o cuando una elección con resultado quedó pendiente; `"mode": "full"` fuerza la serie entera cuando hay algo pendiente») y el ejemplo `python job.py update '{"scopes":["es"],"mode":"full"}'`.

- [ ] **Step 2: `deploy/README.md`**

«Calcular las series y los ratings»: la tabla de acciones con la fila `compute` reescrita (fases por modo y cuándo sale cada uno) y una fila nueva `average_errors` (qué escribe, cuándo lanzarla: para las opciones del método 15, tras un compute `full`); el orden fijo de `what` con las cuatro acciones; ejemplos (añadir el de `average_errors` y el de `mode`); parámetros (fila `mode`); niveles de `verbose` (las fases de `light` a nivel 2); «Estados y resumen» con las líneas nuevas (`series computed (light, polls pending: …)`, `(full, results pending: …)`, `(full, overwrite: …)`, `(light, requested: …, dry run)`, `average errors computed (59 rows, 14 events) in 2200 s`); el párrafo final de `average_errors` reescrito como acción (ya no es la última fase de `compute`). «Actualizar»: punto 2 con el modo (`light` a diario, `full` con `overwrite` o con un resultado pendiente), parámetros (fila `mode`, con la salvedad de que no evita el salto), ejemplo de salida con la línea `light`, ejemplo `"mode":"full"`. «Duraciones medidas»: un párrafo con la estimación del `light` de `es` (una pasada de pesos sobre el ámbito entero y un Forecaster por ciclo para el rebaño: del orden de cinco minutos desde el portátil; pendiente de medir por Luis) frente a los ≈ 45 min del compute entero con `average_errors` (medidos el 2026-11-03).

- [ ] **Step 3: Método 15, spec del `update` y cierre de esta spec**

`metodo-15-calibracion.md`: donde dice que el job `compute` rellena `average_errors`, decir `python job.py compute '{"what":["average_errors"],"scopes":...}'` (acción aparte desde el 2026-11-03; el compute diario ya no la calcula). `2026-11-03-job-update-design.md`: tras la línea «Fecha…», un párrafo «Enmienda 2026-11-03: el compute de la serie tiene dos modos (`light`, pesos y rebaño, el de la actualización diaria; `full`) y `average_errors` es una acción aparte del job `compute`; la regla del salto no cambia. Ver `docs/superpowers/specs/2026-11-03-compute-ligero-design.md`.» Esta spec: estado «construida el 2026-11-03 (plan …)» y una línea de desviaciones (ninguna, o las que haya).

- [ ] **Step 4: Comprobar la redacción pública y la suite**

Run: `grep -n "solo " README.md deploy/README.md | grep -v "sólo" | head` (ha de salir vacío o sólo casos ajenos) y `pytest tests -q -m "not integration"`
Expected: sin «solo» sin tilde nuevo; PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md deploy/README.md docs/analisis-2026-09/metodo-15-calibracion.md docs/superpowers/specs/2026-11-03-job-update-design.md docs/superpowers/specs/2026-11-03-compute-ligero-design.md docs/superpowers/plans/2026-11-03-compute-ligero.md
git commit -m "docs: compute con modos light y full y average_errors como acción aparte en los READMEs, el método 15 y la spec del job update; spec y plan del compute ligero"
```
