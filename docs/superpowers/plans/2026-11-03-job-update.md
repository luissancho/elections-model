# Job `update` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un job `update` que encadene carga, compute y publicación de los ámbitos pedidos en una sola tarea, saltando el compute de los ámbitos sin nada cargado desde su último compute, con dos marcas de tiempo nuevas en `elections.events`.

**Architecture:** Dos columnas en `events` (`loaded_at`, que escriben los cargadores; `computed_at`, que escribe `compute_scope` con el instante de inicio) y un helper `pending_events(scope)` que las compara. El pipeline `compute.Compute` salta el ámbito cuando no hay nada pendiente salvo con `overwrite`; el job `compute` expone `overwrite` y el nuevo job `update` es un `SequenceJob` de los tres grupos (`load.polls`, `compute.compute`, `publish.forecast/analysis/parties`) que para al primer fallo.

**Tech Stack:** Python 3.11, pandas, el framework MTpy del repositorio (`SequenceJob`, `Pipeline`, `Model`), pytest con las fábricas falsas de `tests/`.

**Spec:** `docs/superpowers/specs/2026-11-03-job-update-design.md`

## Global Constraints

- El `ALTER TABLE` lo hace Luis a mano; el código sólo actualiza el modelo y escribe en las columnas: `ALTER TABLE elections.events ADD COLUMN loaded_at timestamp, ADD COLUMN computed_at timestamp;`.
- Identificadores en inglés; cabeceras de consola en castellano, estados y razones en inglés (`skipped (nothing loaded since the last compute)`).
- Ningún test toca la base del `.env` ni escribe en `backtest/results`; todo acceso a la base se sustituye por fakes o `monkeypatch`.
- `mtpy/core/worker.py` no se toca. `job.py`, `RUN_JOB` y `deploy/ecs/job.py` no cambian de interfaz.
- Luis commitea en paralelo en el mismo árbol: antes de cada commit, `git status` y `git add` sólo de los ficheros de la tarea; nunca `git add -A` ni `git commit -a`; `data/wikipedia/wp-maps.json` está modificado por él y no se toca.
- Docstrings en inglés en el código (estilo numpy, como los vecinos); README y spec en castellano. «sólo» con tilde en los textos públicos.
- Los commits terminan con `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. Un ámbito recién añadido al catálogo sin fila en `events` (ni elección próxima): `pending_events` devuelve lista vacía y el compute lo salta con su razón; `mark_loaded` no marca nada y no lanza. Test en la tarea 1 (`test_pending_events_without_rows_is_empty`).
2. `update` con `scopes: "all"`: un ámbito sin artículo de sondeos termina `skipped` en la carga (no `failed`), no para la cadena con el default `stop_on_error = True`, y sigue hasta publicar. Test en la tarea 6 (`test_update_all_skips_a_scope_without_article_and_goes_on`).
3. `update` con `dry_run`: la carga no guarda ni marca, el compute sin nada pendiente se salta también en seco, la publicación escribe bajo `site-dry`. Test en la tarea 6 (`test_update_dry_run_touches_nothing`).
4. `compute` sin `overwrite` tras el `ALTER` (todas las marcas nulas): `pending_events` devuelve todas las elecciones (regla «`computed_at` nulo es pendiente»), así que la primera ejecución computa y marca. Test en la tarea 1 (`test_pending_events_treats_a_null_computed_at_as_pending`).
5. `update` con `event_date` en el JSON (hábito de `load`): se rechaza antes de tocar nada, con el mensaje que remite a `load`. Test en la tarea 6 (`test_update_rejects_event_date_and_years_with_overwrite`).

---

### Task 1: Modelo y helpers de las marcas (`data.py`)

**Files:**
- Modify: `mtpy/models/elections.py` (clase `Events`, línea 391)
- Modify: `mtpy/lib/data.py` (junto a `get_events`, línea 904, y `update_featured`, línea 1317)
- Create: `tests/test_data_marks.py`

**Interfaces:**
- Consumes: `Events` (modelo), `Events.execute(query: str)`, `Events.get_var(query: str)`, `Events.get_results(query: dict, formatted: bool)`.
- Produces (en `mtpy/lib/data.py`):
  - `mark_loaded(scope: str, event_date: str) -> None`: ejecuta `UPDATE elections.events SET loaded_at = now() WHERE scope = '<scope>' AND date = '<event_date>'`.
  - `db_now() -> datetime`: `Events().get_var(query='SELECT now()')`.
  - `mark_computed(scope: str, started_at: datetime) -> None`: ejecuta `UPDATE elections.events SET computed_at = '<started_at>' WHERE scope = '<scope>'`, con `started_at.strftime('%Y-%m-%d %H:%M:%S.%f')`.
  - `pending_events(scope: str) -> list[str]`: fechas ISO (`YYYY-MM-DD`, ordenadas) de las elecciones del ámbito con `computed_at` nulo o `loaded_at > computed_at`, leídas con `Events().get_results(query={'columns': ['date'], 'filters': ["scope = '<scope>'", '(computed_at IS NULL OR loaded_at > computed_at)']}, formatted=True)`.
  - `update_featured` sigue devolviendo `int`; su frame conserva `loaded_at` y `computed_at` tal como los lee `get_events`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
"""Tests de las marcas de carga y compute de `events` (`mtpy.lib.data`), con el modelo sustituido."""
from datetime import datetime

import pandas as pd
import pytest

from mtpy.lib import data


class FakeEvents:
    """`Events` sin base: anota las consultas ejecutadas y devuelve las filas y el reloj dados."""
    table = 'elections.events'

    def __init__(self, rows=None, now=datetime(2026, 11, 3, 10, 0, 0, 123456)):
        self.executed, self.queries, self.upserted = [], [], []
        self.rows, self.now = rows if rows is not None else [], now

    def execute(self, query):
        self.executed.append(query)

    def get_var(self, query=None):
        self.queries.append(query)
        return self.now

    def get_results(self, query=None, formatted=False):
        self.queries.append(query)
        return pd.DataFrame(self.rows, columns=['date'] if not self.rows else None)

    def format_data(self, df, **kwargs):
        return df

    def upsert(self, df):
        self.upserted.append(df.copy())
        return len(df)


@pytest.fixture
def events(monkeypatch):
    fake = FakeEvents()
    monkeypatch.setattr(data, 'Events', lambda: fake)
    return fake


def test_mark_loaded_stamps_the_event_with_the_database_clock(events):
    data.mark_loaded('es', '2026-11-29')
    assert events.executed == ["UPDATE elections.events SET loaded_at = now() WHERE scope = 'es' AND date = '2026-11-29'"]


def test_db_now_reads_the_clock_of_the_database(events):
    assert data.db_now() == datetime(2026, 11, 3, 10, 0, 0, 123456) and events.queries == ['SELECT now()']


def test_mark_computed_stamps_every_event_of_the_scope_with_the_start_instant(events):
    data.mark_computed('es-md', datetime(2026, 11, 3, 10, 0, 0, 123456))
    assert events.executed == ["UPDATE elections.events SET computed_at = '2026-11-03 10:00:00.123456' WHERE scope = 'es-md'"]


def test_pending_events_filters_in_the_database_and_returns_iso_dates(monkeypatch):
    fake = FakeEvents(rows=[{'date': pd.Timestamp('2026-11-29')}, {'date': pd.Timestamp('2023-07-23')}])
    monkeypatch.setattr(data, 'Events', lambda: fake)
    assert data.pending_events('es') == ['2023-07-23', '2026-11-29']
    assert fake.queries == [{'columns': ['date'],
                             'filters': ["scope = 'es'", '(computed_at IS NULL OR loaded_at > computed_at)']}]


def test_pending_events_without_rows_is_empty(events):
    assert data.pending_events('es-xx') == []


def test_pending_events_treats_a_null_computed_at_as_pending():
    # la regla va en el filtro SQL: `computed_at IS NULL` entra aunque `loaded_at` sea nulo (tras el ALTER todo
    # está pendiente); la comprobación es sobre el texto del filtro, que es lo que ve la base
    assert '(computed_at IS NULL OR loaded_at > computed_at)' in data.PENDING_FILTER


def test_update_featured_keeps_the_marks_of_each_event(monkeypatch):
    fake = FakeEvents()
    monkeypatch.setattr(data, 'Events', lambda: fake)
    monkeypatch.setattr(data, 'get_events', lambda scope: pd.DataFrame({
        'date': pd.to_datetime(['2023-07-23', '2026-11-29']), 'scope': 'es', 'name': ['Julio 2023', 'Próximas'],
        'featured': [False, False], 'has_results': [True, False],
        'loaded_at': pd.to_datetime(['2026-11-01 09:00:00', '2026-11-03 09:00:00']),
        'computed_at': pd.to_datetime(['2026-11-02 09:00:00', None])}))

    class FakePolls:
        def get_results(self, query=None, formatted=False):
            return pd.DataFrame({'event_date': pd.to_datetime(['2023-07-23'] * 10), 'computed': True,
                                 'weight_over': 1.0, 'mtype': None, 'ctype': None})

    monkeypatch.setattr(data, 'Polls', FakePolls)
    assert data.update_featured('es') == 1
    saved = fake.upserted[0]
    assert list(saved['featured']) == [True, False] and 'has_results' not in saved.columns
    assert list(saved['loaded_at']) == list(pd.to_datetime(['2026-11-01 09:00:00', '2026-11-03 09:00:00']))
    assert saved['computed_at'].iloc[0] == pd.Timestamp('2026-11-02 09:00:00') and pd.isna(saved['computed_at'].iloc[1])
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_data_marks.py -q`
Expected: FAIL con `AttributeError: module 'mtpy.lib.data' has no attribute 'mark_loaded'` (y análogos).

- [ ] **Step 3: Añadir las columnas al modelo `Events`**

En `mtpy/models/elections.py`, clase `Events`: docstring (una elección de un ámbito; `featured` lo pone `update_featured`; `loaded_at` es la última escritura de sondeos o resultados de la elección por un cargador, `computed_at` el instante de inicio del último `compute_scope` del ámbito; cualquier escritor que haga upsert de la tabla ha de arrastrar las tres columnas que no escribe, porque `format_data` repone como nulas las que faltan) y en `meta`: `'loaded_at': 'dts'`, `'computed_at': 'dts'` tras `'featured'`.

- [ ] **Step 4: Implementar los helpers en `mtpy/lib/data.py`**

Junto a `get_events`: `PENDING_FILTER = '(computed_at IS NULL OR loaded_at > computed_at)'` (constante de módulo), `mark_loaded`, `db_now`, `mark_computed` y `pending_events` con las firmas y el SQL del bloque Interfaces. `from datetime import datetime` ya no hace falta importar `date` aparte si se amplía el import existente. En `update_featured` no cambia nada salvo el docstring: las marcas llegan en `get_events` y van en el frame del upsert (el test lo fija); si el frame las perdiera por el `drop(columns='has_results')`, no las quites.

- [ ] **Step 5: Comprobar que pasan y que nada se rompe**

Run: `pytest tests/test_data_marks.py tests/test_data_unit.py tests/test_mtypes.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git status --short
git add mtpy/models/elections.py mtpy/lib/data.py tests/test_data_marks.py
git commit -m "data y modelos: marcas loaded_at y computed_at en events (mark_loaded con now() de la base, db_now, mark_computed con el instante de inicio, pending_events con el filtro en SQL: computed_at nulo o loaded_at posterior); update_featured arrastra las marcas en su upsert, porque format_data repone como nulas las columnas ausentes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Los cargadores marcan `loaded_at`

**Files:**
- Modify: `mtpy/lib/loader.py` (`InfoElectoralLoader.save_results` ~429, `WikipediaLoader.save_polls` 1122, `WikipediaResultsLoader.save_event` 1427 y `save_results` 1444; import de `.data` línea 18)
- Test: `tests/test_loader_marks.py` (nuevo)

**Interfaces:**
- Consumes: `data.mark_loaded(scope, event_date)` (tarea 1), importado en `loader.py` como `from .data import get_districts, get_scopes, mark_loaded`.
- Produces: ningún nombre nuevo. `WikipediaLoader.save_polls(overwrite)` llama a `mark_loaded(self.scope, self.event_date)` tras `vacuum()` cuando `self.polls` tiene filas; `WikipediaResultsLoader.save_results()` y `InfoElectoralLoader.save_results()` igual cuando `self.results` tiene filas; `WikipediaResultsLoader.save_event()` conserva `loaded_at` y `computed_at` de la fila actual como conserva `featured`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
"""Tests de las marcas de carga de los cargadores (`mtpy.lib.loader`): instancias desnudas, sin base ni red."""
import pandas as pd

from mtpy.lib import loader as module
from mtpy.lib.loader import InfoElectoralLoader, WikipediaLoader, WikipediaResultsLoader


class FakeTable:
    """Modelo de tabla sin base: acepta la secuencia de guardado y anota los upserts."""
    table = 'elections.t'

    def __init__(self, current=None):
        self.current, self.upserted, self.executed = current, [], []

    def execute(self, query):
        self.executed.append(query)

    def stage_write(self, df, compression=None):
        pass

    def upsert(self, df=None):
        self.upserted.append(df)
        return 3

    def stage_clean(self):
        pass

    def vacuum(self):
        pass

    def get_results(self, query=None, formatted=False):
        return self.current if self.current is not None else pd.DataFrame()

    def format_data(self, df, **kwargs):
        return df


def bare(cls, scope='es', event_date='2026-11-29'):
    obj = cls.__new__(cls)
    obj.scope, obj.event_date, obj.verbose = scope, event_date, 0
    obj.filters = ["scope = '{}'".format(scope), "date = '{}'".format(event_date)]
    return obj


def marks(monkeypatch):
    seen = []
    monkeypatch.setattr(module, 'mark_loaded', lambda scope, event_date: seen.append((scope, event_date)))
    return seen


def test_wikipedia_loader_marks_the_event_after_saving_polls(monkeypatch):
    seen = marks(monkeypatch)
    wiki = bare(WikipediaLoader)
    wiki.m_polls = FakeTable()
    wiki.polls = pd.DataFrame({'x': [1, 2]})
    wiki.save_polls(overwrite=False)
    assert seen == [('es', '2026-11-29')] and len(wiki.m_polls.upserted) == 1


def test_wikipedia_loader_without_polls_saves_nothing_and_marks_nothing(monkeypatch):
    seen = marks(monkeypatch)
    wiki = bare(WikipediaLoader)
    wiki.m_polls = FakeTable()
    wiki.polls = pd.DataFrame()
    wiki.save_polls(overwrite=True)
    assert seen == [] and wiki.m_polls.upserted == []


def test_results_loaders_mark_the_event_only_when_they_write_results(monkeypatch):
    seen = marks(monkeypatch)
    wp = bare(WikipediaResultsLoader, 'es-md', '2023-05-28')
    wp.m_results = FakeTable()
    wp.results = pd.DataFrame({'party_id': [1]})
    wp.save_results()
    wp.results = pd.DataFrame()
    wp.save_results()
    ie = bare(InfoElectoralLoader, 'es', '2023-07-23')
    ie.m_results = FakeTable()
    ie.results = pd.DataFrame({'party_id': [1]})
    ie.save_results()
    ie.results = pd.DataFrame()
    ie.save_results()
    assert seen == [('es-md', '2023-05-28'), ('es', '2023-07-23')]


def test_save_event_carries_over_featured_and_the_marks_of_the_current_row():
    wp = bare(WikipediaResultsLoader, 'es', '2023-07-23')
    wp.event = pd.DataFrame([{'date': pd.Timestamp('2023-07-23'), 'scope': 'es', 'name': 'Elecciones España Julio 2023',
                              'featured': False}])
    wp.m_events = FakeTable(current=pd.DataFrame([{
        'date': pd.Timestamp('2023-07-23'), 'scope': 'es', 'name': 'viejo', 'featured': True,
        'loaded_at': pd.Timestamp('2026-11-01 09:00:00'), 'computed_at': pd.Timestamp('2026-11-02 09:00:00')}]))
    wp.save_event()
    saved = wp.m_events.upserted[0].iloc[0]
    assert saved['name'] == 'Elecciones España Julio 2023' and bool(saved['featured']) is True
    assert saved['loaded_at'] == pd.Timestamp('2026-11-01 09:00:00') and saved['computed_at'] == pd.Timestamp('2026-11-02 09:00:00')


def test_save_event_of_a_new_row_carries_no_marks():
    wp = bare(WikipediaResultsLoader, 'es', '2030-01-01')
    wp.event = pd.DataFrame([{'date': pd.Timestamp('2030-01-01'), 'scope': 'es', 'name': 'Próximas', 'featured': False}])
    wp.m_events = FakeTable(current=pd.DataFrame())
    wp.save_event()
    saved = wp.m_events.upserted[0]
    assert 'loaded_at' not in saved.columns and 'computed_at' not in saved.columns
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_loader_marks.py -q`
Expected: FAIL (`AttributeError: module 'mtpy.lib.loader' has no attribute 'mark_loaded'` y, en `save_event`, `KeyError: 'loaded_at'`).

- [ ] **Step 3: Implementar las marcas en los tres cargadores**

En `loader.py`: importar `mark_loaded` de `.data`; en `WikipediaLoader.save_polls`, tras `self.m_polls.vacuum()` y antes del `print`, `mark_loaded(self.scope, self.event_date)` (el `return` temprano con `polls` vacío ya evita la marca); en `InfoElectoralLoader.save_results`, igual tras `vacuum()`; en `WikipediaResultsLoader.save_results`, llamar a `_save` y marcar sólo si `self.results` tiene filas; en `save_event`, junto a `event['featured'] = ...`, copiar `loaded_at` y `computed_at` de `current` cuando hay fila actual (y no añadir las columnas cuando no la hay). Docstrings de los cuatro métodos con la marca.

- [ ] **Step 4: Comprobar que pasan y que los tests de los cargadores siguen verdes**

Run: `pytest tests/test_loader_marks.py tests/test_loader_unit.py tests/test_load_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git add mtpy/lib/loader.py tests/test_loader_marks.py
git commit -m "loader: los cargadores marcan loaded_at en events al guardar sondeos (WikipediaLoader.save_polls) o resultados oficiales (WikipediaResultsLoader e InfoElectoralLoader.save_results), sólo cuando escriben filas; save_event arrastra loaded_at y computed_at de la fila actual como arrastra featured

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `compute_scope` marca `computed_at` con el instante de inicio

**Files:**
- Modify: `mtpy/lib/compute.py` (imports línea 18 y `compute_scope` líneas 56-135)
- Test: `tests/test_compute_unit.py` (añadir tras `test_compute_scope_lets_a_failure_propagate`, y ampliar `test_the_real_computer_and_featured_updater_are_the_defaults`)

**Interfaces:**
- Consumes: `data.db_now()`, `data.mark_computed(scope, started_at)`, `data.pending_events(scope)` (tarea 1), importados en `compute.py` con `from .data import db_now, mark_computed, pending_events, save_catalogues, update_featured` (`pending_events` se reexporta para el pipeline de la tarea 4).
- Produces: `compute_scope(scope, save=True, verbose=0, progress=None, computer=None, featured=None, mark=None, clock=None) -> dict`. Con `save`, `started_at = clock()` antes de construir la serie y `mark(scope, started_at)` después de `average_errors`; `clock` es `db_now` y `mark` es `mark_computed` por defecto. Sin `save`, ni `clock` ni `mark` se llaman. El resultado no cambia de claves.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_compute_scope_marks_the_scope_with_the_instant_it_started():
    marks, clocks = [], []
    compute.compute_scope('es-md', save=True, computer=FakeComputer(), featured=lambda scope: 0,
                          mark=lambda scope, at: marks.append((scope, at)), clock=lambda: clocks.append('T0') or 'T0')
    assert clocks == ['T0'] and marks == [('es-md', 'T0')]


def test_compute_scope_dry_run_and_failure_do_not_mark():
    marks = []
    compute.compute_scope('es', save=False, computer=FakeComputer(), featured=lambda scope: 0,
                          mark=lambda scope, at: marks.append(scope), clock=lambda: marks.append('clock'))
    with pytest.raises(RuntimeError, match='boom herding'):
        compute.compute_scope('es', save=True, computer=FakeComputer(fail_at='herding'), featured=lambda scope: 0,
                              mark=lambda scope, at: marks.append(scope), clock=lambda: 'T0')
    assert marks == []
```

Y en `test_the_real_computer_and_featured_updater_are_the_defaults`, además de los `monkeypatch` actuales: `monkeypatch.setattr(compute, 'db_now', lambda: 'T0')` y `monkeypatch.setattr(compute, 'mark_computed', lambda scope, at: marks.append((scope, at)))`, con `assert marks == [('es-cl', 'T0')]` tras el `compute_scope` con `save=True`.

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_compute_unit.py -q -k "marks or do_not_mark or defaults"`
Expected: FAIL con `TypeError: compute_scope() got an unexpected keyword argument 'mark'`.

- [ ] **Step 3: Implementar en `compute_scope`**

Firma y comportamiento del bloque Interfaces; el docstring documenta `mark`, `clock` y por qué el instante es el de inicio (lo cargado mientras corre el compute no entró en la serie y debe seguir pendiente). La marca no es una fase de `COMPUTE_STEPS` ni entra en `seconds`.

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_compute_unit.py tests/test_pipelines_compute.py tests/test_jobs_compute.py -q`
Expected: PASS (los pipelines y el job aún no pasan `pending`; eso llega en las tareas 4 y 5).

- [ ] **Step 5: Commit**

```bash
git status --short
git add mtpy/lib/compute.py tests/test_compute_unit.py
git commit -m "compute: compute_scope marca computed_at de todas las elecciones del ámbito al terminar, con el instante de inicio leído de la base (clock=db_now, mark=mark_computed, inyectables); ni en seco ni tras un fallo

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: El pipeline `compute.Compute` salta el ámbito sin nada pendiente

**Files:**
- Modify: `mtpy/pipelines/compute/Compute.py` (`start` líneas 27-31, `prepare_task` 33-35, docstring de la clase)
- Test: `tests/test_pipelines_compute.py` (helper `series` y tests nuevos)

**Interfaces:**
- Consumes: `compute.pending_events(scope) -> list[str]` (reexportado en `mtpy/lib/compute.py`, tarea 3).
- Produces: `Compute.start(self, dry_run=False, overwrite=False, **kwargs)` con la cabecera `compute <acción>: n ámbitos[ · overwrite][ · dry run]`; `Compute.prepare_task(self, chunk=None, position=(1, 1), overwrite=False, pending=None, **kwargs)`: tras `start_progress`, si `self.ACTION == 'compute'` y no `overwrite`, llama a `(pending or compute.pending_events)(chunk)` y, con lista vacía, `self.end({'status': 'skipped', 'action': 'series', 'reason': 'nothing loaded since the last compute', 'warnings': []})`. `Ratings` hereda y nunca salta (`ACTION == 'ratings'`). Constante de módulo `NOTHING_PENDING = 'nothing loaded since the last compute'`.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_pipelines_compute.py`, el helper `series` pasa `pending` por defecto para que los tests existentes no toquen la base:

```python
def series(factory, scope='es', pipeline=Compute, position=(1, 2), **kwargs):
    p = pipeline()
    p.run('update', chunk=scope, position=position, computer=factory, featured=lambda scope: 7,
          **{'verbose': 0, 'pending': lambda scope: ['2026-11-29'], **kwargs})
    return p
```

Tests nuevos:

```python
def test_compute_skips_a_scope_with_nothing_pending(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, pending=lambda scope: [])
    assert p.result == {'status': 'skipped', 'action': 'series', 'reason': 'nothing loaded since the last compute',
                        'warnings': []}
    assert p.line == 'es: series skipped (nothing loaded since the last compute)' and factory.inits == []


def test_compute_overwrite_computes_anyway_and_says_so_in_the_header(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, pending=lambda scope: [], overwrite=True)
    assert p.result['status'] == 'computed' and factory.inits == [('es', 0)] * 3
    assert 'compute series: 2 ámbitos · overwrite' in capsys.readouterr().out


def test_ratings_never_skip(fresh_app):
    factory = FakeComputer()
    p = series(factory, 'es-md', Ratings, pending=lambda scope: [])
    assert p.result['status'] == 'computed' and p.result['action'] == 'ratings'


def test_the_real_pending_events_is_the_default(fresh_app, monkeypatch):
    from mtpy.lib import compute as compute_lib  # el pipeline llama a `compute.pending_events` en el momento

    asked = []
    monkeypatch.setattr(compute_lib, 'pending_events', lambda scope: asked.append(scope) or [])
    p = Compute()
    p.run('update', chunk='es-cl', position=(1, 1), computer=FakeComputer(), featured=lambda scope: 0, verbose=0)
    assert asked == ['es-cl'] and p.result['status'] == 'skipped'
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_pipelines_compute.py -q`
Expected: FAIL en los cuatro tests nuevos (`result['status'] == 'computed'`, cabecera sin `· overwrite`); los demás PASS.

- [ ] **Step 3: Implementar el salto y la cabecera**

Como describe el bloque Interfaces. Docstring de la clase: el salto, `overwrite`, `pending` inyectable y que `Ratings` no salta. `dry_run` no cambia el criterio.

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_pipelines_compute.py tests/test_worker_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git add mtpy/pipelines/compute/Compute.py tests/test_pipelines_compute.py
git commit -m "pipelines/compute: Compute salta el ámbito sin nada cargado desde su último compute (pending_events, inyectable como pending) con el estado skipped y su razón, salvo con overwrite, que además va en la cabecera; Ratings nunca salta

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `overwrite` en el job `compute` y su documentación

**Files:**
- Modify: `mtpy/jobs/Compute.py` (docstring de la clase, `run` líneas 51-105)
- Modify: `deploy/README.md` (sección «Calcular las series y los ratings», líneas 655-746: tabla de acciones, parámetros, estados)
- Modify: `README.md` (subsección «`compute`: series de sondeos y ratings», líneas 35-54)
- Test: `tests/test_jobs_compute.py` (helper `run` y tests nuevos)

**Interfaces:**
- Consumes: `Compute.prepare_task(..., overwrite, pending)` (tarea 4).
- Produces: `Compute.run(self, what=('compute', 'ratings'), scopes=('es',), overwrite=False, dry_run=False, verbose=1, computer=None, featured=None, catalogues=None, pending=None, stop_on_error=None, **kwargs) -> dict`, que pasa `overwrite=overwrite` y `pending=pending` a `super().run`. El resultado de un ámbito saltado es el diccionario `skipped` de la tarea 4.

- [ ] **Step 1: Escribir los tests que fallan**

Helper `run` de `tests/test_jobs_compute.py` con `pending` por defecto:

```python
def run(factory, featured=None, **kwargs):
    seen = []
    featured = featured if featured is not None else (lambda scope: seen.append(scope) or 7)
    return Compute().run(computer=factory, featured=featured,
                         **{'pending': lambda scope: ['2026-11-29'], **kwargs}), seen
```

Tests nuevos:

```python
def test_compute_skips_the_scopes_with_nothing_pending_unless_overwrite(fresh_app, scopes, capsys):
    factory = FakeComputer()
    results, featured = run(factory, what='compute', scopes='all', verbose=0,
                            pending=lambda scope: ['2026-11-29'] if scope == 'es' else [])
    assert [scope for scope, _ in factory.inits] == ['es'] * 3 and featured == ['es']
    assert results['es-md'] == {'status': 'skipped', 'action': 'series',
                                'reason': 'nothing loaded since the last compute', 'warnings': []}
    out = capsys.readouterr().out
    assert 'es-md: series skipped (nothing loaded since the last compute)' in out and 'compute series: 2 ámbitos' in out
    factory = FakeComputer()
    results, _ = run(factory, what='compute', scopes='all', overwrite=True, verbose=0, pending=lambda scope: [])
    assert [scope for scope, _ in factory.inits] == ['es'] * 3 + ['es-md'] * 3
    assert 'compute series: 2 ámbitos · overwrite' in capsys.readouterr().out


def test_compute_skipped_scope_still_gets_its_ratings(fresh_app, scopes):
    factory = FakeComputer()
    results, _ = run(factory, what=['compute', 'ratings'], scopes=['es'], verbose=0, pending=lambda scope: [])
    assert [scope for scope, _ in factory.inits] == ['es'] and results['es']['action'] == 'ratings'
```

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_jobs_compute.py -q`
Expected: FAIL con `TypeError: run() got an unexpected keyword argument 'pending'` (o `overwrite`).

- [ ] **Step 3: Implementar `overwrite` y `pending` en el job**

Firma del bloque Interfaces; docstring de la clase (párrafo: con `overwrite` falso, por defecto, `compute` sólo corre en los ámbitos con algo cargado desde su último compute, según `loaded_at` y `computed_at` de `events`, y el ámbito saltado termina `skipped` con su razón; `catalogues` y `ratings` no cambian; tras un cambio de código o de parámetros hay que pasar `"overwrite": true`) y de `run` (parámetros `overwrite` y `pending`); ejemplo nuevo `python job.py compute '{"what":["compute"],"scopes":"all","overwrite":true}'`.

- [ ] **Step 4: Comprobar que pasan**

Run: `pytest tests/test_jobs_compute.py tests/test_launchers.py -q`
Expected: PASS.

- [ ] **Step 5: Documentar en los README**

`deploy/README.md`, sección «Calcular las series y los ratings»: en la fila `compute` de la tabla de acciones, que sólo corre en los ámbitos con algo cargado desde su último compute (marcas `loaded_at`/`computed_at` de `events`, que escriben los cargadores y el propio compute) salvo con `overwrite`; en «Ejemplos», el ejemplo con `overwrite` y su explicación (recalcular tras un cambio de código o de `params.json`); en «Parámetros», la fila `| overwrite | false | Computar la serie de todos los ámbitos aunque no tengan nada cargado desde su último compute (tras un cambio de código o de parámetros); sin él, el ámbito termina skipped |`; en «Estados y resumen», la línea `es-md: series skipped (nothing loaded since the last compute)` con su explicación, y la nota de que tras el `ALTER` todas las elecciones cuentan como pendientes y el primer `compute` computa todo una vez. `README.md`, subsección `compute`: una frase con el salto y `overwrite`, y el ejemplo.

- [ ] **Step 6: Commit**

```bash
git status --short
git add mtpy/jobs/Compute.py tests/test_jobs_compute.py deploy/README.md README.md
git commit -m "jobs/compute: overwrite (por defecto falso: la serie sólo corre en los ámbitos con algo cargado desde su último compute, el resto termina skipped; catalogues y ratings no cambian) y pending inyectable; READMEs con el criterio, el parámetro, el estado y el primer compute tras el ALTER

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: El job `update`

**Files:**
- Rewrite: `mtpy/jobs/Update.py` (hoy la secuencia de ejemplo de MTpy: `ancla`, `meteo`)
- Create: `tests/test_jobs_update.py`
- Modify: `tests/test_ecs_job.py` (una aserción junto a la línea 116)
- Modify: `deploy/README.md` (sección nueva «## Actualizar: carga, series y publicación de una vez» antes de «## Ensayos de operación», línea 1565; «Jobs desde ECS», línea 1377, un ejemplo)
- Modify: `README.md` (subsección «### `update`: carga, series y publicación de una vez» antes de «### `compute`», y la frase de `stop_on_error` del párrafo de «## Jobs»)

**Interfaces:**
- Consumes: `load.Polls` (acciones `update`/`rebulk`, kwargs `force`, `years`, `dry_run`, `skip_missing`, `loader`, `computer`, `catalogues`), `compute.Compute` (`overwrite`, `pending`, `computer`, `featured`, `dry_run`), `publish.Forecast`/`Analysis`/`Parties` (`fs`, `dry_run`, `today`, `force`, `n_sim`, `seed`, `drange`, `max_fc`, `alpha`, `correctors`), `publish.resolve_scopes`, `load.YEARS_WITH_OVERWRITE`.
- Produces: clase `Update(SequenceJob)` con `sequence = {'load': ['polls'], 'compute': ['compute'], 'publish': ['forecast', 'analysis', 'parties']}`, `default_action = 'update'`, `stop_on_error = True`, `chunk_noun = 'scopes'` y `run(self, scopes=('es',), overwrite=False, dry_run=False, verbose=1, fs=None, stop_on_error=None, **kwargs) -> dict`:
  1. `if 'event_date' in kwargs: raise ValueError('update: "event_date" is not accepted (use load)')`.
  2. `years = kwargs.get('years')`; `if years and overwrite: raise ValueError(YEARS_WITH_OVERWRITE)`; `kwargs['years'] = [str(y) for y in years]` cuando viene.
  3. `fs = fs if fs is not None else self.app.fs`; sin `fs`, `RuntimeError('update: no file system configured (set S3_BUCKET or run with a local files/ root)')`.
  4. `targets = resolve_scopes(scopes)`; vacío, `ValueError('update: no scopes to run')`.
  5. `super().run(select=None, action='rebulk' if overwrite else 'update', chunks=targets, stop_on_error=stop_on_error, overwrite=overwrite, dry_run=dry_run, verbose=verbose, fs=fs, skip_missing=scopes == 'all', **kwargs)`.
  6. Devuelve `{scope: {'polls': r, 'compute': r, 'forecast': r, 'analysis': r}, 'parties': r}` a partir de `self.results` (clave `(group.pipeline, chunk)`: el nombre del pipeline es la parte tras el punto; `chunk` `None` es `parties`).
  Imports: `from ..core.worker import SequenceJob`, `from ..lib.load import YEARS_WITH_OVERWRITE`, `from ..lib.publish import resolve_scopes`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
"""Tests del job `update` (`mtpy/jobs/Update.py`): la cadena load → compute → publish con las fábricas falsas de los
tres grupos y la base sustituida."""
import importlib

import pytest

from mtpy.core.io import FileSystem
from tests.publish_fixtures import patched  # noqa: F401
from tests.test_compute_unit import FakeComputer
from tests.test_load_unit import FakeCatalogues, FakeLoader

module = importlib.import_module('mtpy.jobs.Update')
Update = module.Update

EVENTS = {'es': '2026-11-29', 'es-md': '2027-05-23'}


@pytest.fixture
def scopes(monkeypatch):
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: EVENTS.get(scope))
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: event_date != '2027-05-23')


@pytest.fixture
def alerts(monkeypatch):
    seen = []
    monkeypatch.setattr(Update, 'alert', lambda self, msg: seen.append(msg))
    return seen


def run(tmp_path, loader=None, computer=None, pending=None, **kwargs):
    computer = computer or FakeComputer()
    results = Update().run(fs=FileSystem(str(tmp_path)), loader=loader or FakeLoader(), computer=computer,
                           catalogues=FakeCatalogues(), featured=lambda scope: 7,
                           pending=pending or (lambda scope: ['2026-11-29']), **{'verbose': 0, 'n_sim': 20, **kwargs})
    return results, computer


def test_update_runs_the_three_groups_on_es_with_one_alert(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'])
    out = capsys.readouterr().out
    assert 'load polls: 1 ámbito' in out and 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s' in out
    assert 'compute series: 1 ámbito' in out and 'es: series computed (2 polls, 2 events, 7 featured) in 0 s' in out
    assert 'publish forecast: 1 ámbito' in out and 'es: published' in out and 'es: analysis published' in out
    assert 'parties: ' in out and 'manifest: 1 scopes, freeze off' in out
    # un Computer de los pesos de la carga (2 sondeos guardados) y tres del compute
    assert [scope for scope, _ in computer.inits] == ['es'] * 4
    assert set(results) == {'es', 'parties'} and set(results['es']) == {'polls', 'compute', 'forecast', 'analysis'}
    assert results['es']['forecast']['status'] == 'published' and results['parties']['status'] == 'published'
    assert len(alerts) == 1 and alerts[0].startswith('Update job executed in')
    assert patched[0][3]['n_sim'] == 20
    assert (tmp_path / 'site' / 'v1' / 'manifest.json').exists()


def test_update_skips_the_compute_of_a_scope_with_nothing_pending(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'], pending=lambda scope: [])
    assert computer.inits == [('es', 0)] and results['es']['compute']['status'] == 'skipped'  # sólo los pesos de la carga
    assert 'es: series skipped (nothing loaded since the last compute)' in capsys.readouterr().out
    assert results['es']['forecast']['status'] == 'published'


def test_update_overwrite_reloads_the_cycle_and_computes_anyway(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    loader = FakeLoader()
    results, computer = run(tmp_path, loader=loader, scopes=['es'], overwrite=True, pending=lambda scope: [])
    out = capsys.readouterr().out
    assert 'load polls: 1 ámbito · overwrite' in out and ('select_series', True) in loader.calls
    assert 'compute series: 1 ámbito · overwrite' in out and [scope for scope, _ in computer.inits] == ['es'] * 4


def test_update_stops_at_the_first_failure_and_publishes_nothing(fresh_app, tmp_path, scopes, events, patched, alerts):
    with pytest.raises(RuntimeError, match=r'update: failed scopes: es \(polls\)'):
        run(tmp_path, loader=FakeLoader(missing={'pollsters': ['Casa Nueva']}), scopes=['es'])
    assert not (tmp_path / 'site').exists() and len(alerts) == 1 and 'with 1 failure' in alerts[0]
    with pytest.raises(RuntimeError, match=r'update: failed scopes: es \(series\)'):
        run(tmp_path, computer=FakeComputer(fail_at='herding'), scopes=['es'])
    assert not (tmp_path / 'site').exists()


def test_update_without_stop_on_error_publishes_what_it_has(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    with pytest.raises(RuntimeError, match=r'update: failed scopes: es \(polls\)'):
        run(tmp_path, loader=FakeLoader(missing={'pollsters': ['Casa Nueva']}), scopes=['es'], stop_on_error=False,
            pending=lambda scope: [])
    out = capsys.readouterr().out
    assert 'es: polls failed (' in out and 'es: series skipped (' in out and 'es: published' in out
    assert (tmp_path / 'site' / 'v1' / 'manifest.json').exists()


def test_update_all_skips_a_scope_without_article_and_goes_on(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, _ = run(tmp_path, scopes='all')
    out = capsys.readouterr().out
    assert 'es-md: polls skipped (no polls article for 2027-05-23)' in out
    assert results['es-md']['polls']['status'] == 'skipped' and results['es']['forecast']['status'] == 'published'
    assert results['es-md']['forecast']['status'] == 'skipped'  # sin evento próximo en `patched`


def test_update_dry_run_touches_nothing(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    loader = FakeLoader()
    results, computer = run(tmp_path, loader=loader, scopes=['es'], dry_run=True, pending=lambda scope: [])
    out = capsys.readouterr().out
    assert 'es: polls checked (5 read, 2 new, 2026-11-29, dry run) in 0 s' in out
    assert computer.inits == [] and 'es: series skipped (' in out
    assert (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es').exists() and not (tmp_path / 'site').exists()


def test_update_rejects_event_date_and_years_with_overwrite(fresh_app, tmp_path, scopes, events):
    with pytest.raises(ValueError, match='update: "event_date" is not accepted'):
        run(tmp_path, scopes=['es'], event_date='2023-07-23')
    with pytest.raises(ValueError, match='"years" cannot be combined with "overwrite"'):
        run(tmp_path, scopes=['es'], years=[2026], overwrite=True)


def test_update_fails_early_without_a_file_system_or_scopes(fresh_app, scopes, events):
    fresh_app.set('fs', None)
    with pytest.raises(RuntimeError, match='update: no file system configured'):
        Update().run(scopes=['es'])
    with pytest.raises(ValueError, match='update: no scopes to run'):
        Update().run(scopes=[], fs=object())
```

En `tests/test_ecs_job.py`, junto a la aserción de `load` de la línea 116:
`assert ecs_job.run_job_value('update', '{"scopes":["es"],"n_sim":10000}') == 'update {"scopes":["es"],"n_sim":10000}'`.

- [ ] **Step 2: Comprobar que fallan**

Run: `pytest tests/test_jobs_update.py tests/test_ecs_job.py -q`
Expected: los de `update` FAIL (`TypeError`/`ValueError: update: unknown select` o kwargs inesperados del `Update` viejo); el de ECS PASS (el lanzador ya deja pasar `update`).

- [ ] **Step 3: Escribir `mtpy/jobs/Update.py`**

La clase del bloque Interfaces, con docstring de clase al estilo de `Publish`: qué encadena, el salto del compute por las marcas, `overwrite` como único interruptor de rehacerlo todo, para al primer fallo (y `stop_on_error` falso para seguir), `force` llega a la carga y a la publicación a la vez (para dar de alta casas, `load` con `force` aparte; en `update`, `force` es publicar dentro de la veda), `event_date` rechazado, ejemplos (`python job.py update '{"scopes":["es"],"n_sim":10000,"verbose":3}'`, `'{"scopes":"all"}'`, `'{"scopes":["es"],"overwrite":true}'`, `'{"scopes":["es"],"n_sim":20,"dry_run":true}'`) y `RUN_JOB='update {"scopes":["es"],"n_sim":10000}'`. El resultado reagrupado, en el docstring de `run`.

- [ ] **Step 4: Comprobar que pasan y la batería entera**

Run: `pytest tests/test_jobs_update.py tests/test_ecs_job.py -q` y después `pytest -q -x`
Expected: PASS en ambos.

- [ ] **Step 5: Documentar en los README**

`deploy/README.md`, sección nueva «## Actualizar: carga, series y publicación de una vez» con la estructura de las otras («Qué hace y cuándo lanzarlo», «Ejemplos», «Parámetros», «Estados y resumen»): los tres grupos y su orden, el salto del compute por las marcas, `overwrite`, para al primer fallo y qué pasa con `stop_on_error` falso, `force` doble, `event_date` rechazado, la tabla de parámetros (`scopes`, `overwrite`, `dry_run`, `verbose`, `stop_on_error`, `force`, `n_sim`, `seed`, `drange`, `max_fc`, `alpha`, `today`, `years`), los pasos de Luis (el `ALTER`, desplegar la imagen, el primer `update` sin `overwrite` computa todo una vez) y Docker/ECS. En «Jobs desde ECS», `update` en la lista de jobs y el ejemplo `python deploy/ecs/job.py update '{"scopes":["es"],"n_sim":10000}' --follow`. `README.md`: la subsección `update` (tres líneas y dos ejemplos) y, en el párrafo de «## Jobs», que `update` para al primer fallo por defecto.

- [ ] **Step 6: Commit**

```bash
git status --short
git add mtpy/jobs/Update.py tests/test_jobs_update.py tests/test_ecs_job.py deploy/README.md README.md
git commit -m "jobs/update: carga, compute y publicación de los ámbitos pedidos en una sola tarea (load.polls, compute.compute, publish.forecast/analysis/parties), con el compute saltado donde no hay nada cargado desde el último compute, overwrite como único interruptor de rehacerlo todo, parada al primer fallo por defecto, event_date rechazado y comprobación del bucket al empezar; READMEs con la sección del job, el ejemplo de ECS y los pasos del ALTER

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Cierre: batería completa, spec y revisión

**Files:**
- Modify: `docs/superpowers/specs/2026-11-03-job-update-design.md` (línea de estado y, si hubo, «Desviaciones de la construcción»)

- [ ] **Step 1: Batería completa**

Run: `pytest -q`
Expected: PASS, con el número de tests anterior más los nuevos (≈ 25 más).

- [ ] **Step 2: Cerrar la spec**

Estado: `aprobada por Luis el 2026-11-03; construida el 2026-11-03 (plan docs/superpowers/plans/2026-11-03-job-update.md)`. Añadir un párrafo «Desviaciones de la construcción» sólo si el constructor se apartó de alguna decisión (p. ej. nombres de helpers distintos de los del plan), y «Pendiente de Luis»: el `ALTER TABLE`, desplegar la imagen, el primer `update` sin `overwrite`.

- [ ] **Step 3: Commit**

```bash
git status --short
git add docs/superpowers/specs/2026-11-03-job-update-design.md docs/superpowers/plans/2026-11-03-job-update.md
git commit -m "docs: spec del job update cerrada (construida) y su plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
