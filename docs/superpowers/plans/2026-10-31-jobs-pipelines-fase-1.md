# Jobs como pipelines de MTpy (fase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Los cuatro jobs de elecciones (`load`, `compute`, `backtest`, `publish`) pasan a ser `SequenceJob` de MTpy
sobre un `Pipeline` por acción (`mtpy/pipelines/<job>/<Accion>.py`), con un único aviso final por Pushover (fallos
delante, recortado a 1.024 caracteres, nunca lanza), salida distinta de cero cuando algo falló y el parámetro
`stop_on_error`; los comandos y runbooks de hoy siguen valiendo tal cual.

**Architecture:** El núcleo de MTpy (`mtpy/core/worker.py`) gana, de forma compatible hacia atrás, `stop_on_error`,
`select` con `grupo.pipeline`, `Pipeline.chunked`, `result`/`line` por chunk, el fin anticipado del chunk
(`Pipeline.end`), los ganchos `start`/`finish`, la posición del chunk y el resumen final. Cada pipeline de elecciones
hereda de `ScopePipeline` (`mtpy/pipelines/base.py`: `ScopeReporter` + `Pipeline`) y mueve a sus tareas el cuerpo
del `_x_scope` del job de hoy: la unidad de trabajo sigue siendo la función por ámbito de `mtpy/lib`. Los jobs
quedan finos: validan lo que cruza pipelines, traducen `what`/`scopes` a `select`/`chunks`, llaman a
`super().run(...)` y devuelven los resultados con la forma documentada hoy.

**Tech Stack:** Python 3.11 (pyenv `3.11.9`), pandas, pytest (`python -m pytest`), MTpy (`mtpy/core`). Sin base de
datos: todos los tests de este plan corren con `fresh_app` y las fábricas de `tests/fakes.py`.

**Spec:** `docs/superpowers/specs/2026-10-31-jobs-pipelines-ecs-design.md` (secciones «Diseño de la fase 1»,
«Núcleo», «Tests»).

## Global Constraints

- `mtpy/core/worker.py` es código del framework: todo cambio compatible hacia atrás. Sin fallos, ningún
  `SequenceJob` existente (`Update`, `Pipeline`) ni los pipelines `events/` cambian de comportamiento; el aviso de
  éxito sigue siendo `'{name} job executed in {delta} :-)'`.
- Estructura MTpy: `pipes = {acción: {etapa: [tareas]}}`, etapas `Setup`, `Extract`, `Transform`, `Load`, `Build`,
  `Set`, tareas `<tarea>_task(**kwargs)`; `SequenceJob.sequence = {grupo: [pipelines en snake_case]}`.
- Interfaz de `job.py` y `RUN_JOB` sin cambios: nombre del job y un JSON. `init.sh` no se toca en esta fase.
- Los pipelines llaman a las unidades de trabajo por atributo del paquete (`publish.publish_forecast(...)`,
  `from ...lib import publish`), para que los `monkeypatch` de los tests sigan valiendo.
- Las fábricas inyectables de los tests (`computer`, `loader`, `params_loader`, `catalogues`, `featured`, `fs`,
  `runner`, `stats`, `simulator`) llegan por `**kwargs` a las tareas.
- Identificadores en inglés; textos de consola y del resumen exactamente los de hoy (líneas `es: published …`,
  cabeceras `publish forecast: 2 ámbitos · run …`, `· n avisos`); docstrings de las clases `Publish`, `Load`,
  `Compute` y `Backtest` conservados (un test comprueba el de `Publish`).
- Verbosidad 0-3 como hoy: las líneas de progreso (`ScopeReporter`, `ScopeProgress` de `mtpy/jobs/progress.py`,
  que no se mueve) viven dentro de las tareas.
- Los pipelines `events/` (`Events.py`, `EventsResults.py`) no se tocan.
- Hoy hay 1.557 tests en verde; al cerrar la fase, todos en verde (`python -m pytest -q`).
- Luis hace commits en paralelo con `git commit -a`: antes de cada commit, `git status` y añadir sólo los ficheros
  de la tarea; nunca reescribir sus commits.

## Review Focus

- `stop_on_error` que llega como cadena desde un JSON mal escrito (`"stop_on_error": "false"`): `bool('false')` es
  `True`; sólo se acepta un booleano, cualquier otra cosa es `ValueError` antes de empezar. Test en la Tarea 3.
- `SIGTERM` en mitad de un chunk en modo continuar: no debe seguir con el siguiente chunk (es `BaseException`, no
  `Exception`), avisa con la línea `interrumpido en …` y propaga. Test en la Tarea 3 (con `JobInterrupted` lanzado
  por un pipeline falso) y manejador en la Tarea 4.
- Un pipeline cuyo `start` revienta (p. ej. `Forecast` sin sistema de ficheros): en modo continuar cada chunk
  queda `failed` con la razón, nunca un bucle silencioso. Test en la Tarea 3.
- Aviso de más de 1.024 caracteres (`publish forecast all`): Pushover recibe el texto recortado con la última línea
  `Job.TRUNCATED`, el logger el texto entero, y un 4xx de Pushover va a `logger.error` sin tumbar el job. Tests en
  la Tarea 2.
- `scopes` que resuelve a una lista vacía con un pipeline por ámbito: MTpy ejecutaría el pipeline una vez con
  `chunk=None`; los cuatro jobs fallan antes con `ValueError('<job>: no scopes to run')`. Tests en las Tareas 5-8.

---

### Task 1: `Pipeline` del núcleo: tolerancia, `chunked`, `result`/`line`, `start`/`finish`, fin anticipado

**Files:**
- Modify: `mtpy/core/app.py` (clase `Core`, 138-141)
- Modify: `mtpy/core/worker.py` (`Pipeline` 274-518: `__init__` 329-357, `run` 395-518)
- Test: `tests/test_worker_unit.py`

**Interfaces:**
- Produces: `Core.log(self, level: str, message: str) -> None` en `mtpy/core/app.py`: llama a
  `getattr(self.app.logger, level)(message)` y no hace nada si `self.app.logger is None`.
- Produces, en `mtpy/core/worker.py`:
  - `class PipelineEnd(Exception)` con atributo `result: dict`.
  - `Pipeline.chunked: bool = True`, `Pipeline.label: Optional[str] = None` (atributos de clase).
  - `Pipeline.result: Optional[dict | list]` y `Pipeline.line: Optional[str | list[str]]`, puestos a `None` al
    empezar cada `run`; `Pipeline._started: bool`.
  - `Pipeline.start(self, **kwargs) -> None`: gancho vacío; `run` lo llama una sola vez, en la primera
    ejecución, después de fijar `self._action` y antes de la primera etapa, con los kwargs de esa ejecución
    menos `chunk` (sí lleva `position`).
  - `Pipeline.finish(self, **kwargs) -> Optional[str]`: gancho vacío que devuelve `None`; devolver una cadena
    significa «fallo con esta etiqueta» (quien lo llama la añade a los fallos).
  - `Pipeline.end(self, result: dict) -> NoReturn`: `raise PipelineEnd(result)`.
  - `Pipeline.summary(self, chunk, **kwargs) -> Optional[str | list[str]]`: gancho que devuelve `None`.
  - `Pipeline.failure_name(self, chunk) -> str`: `'{chunk} ({label})'` si hay chunk y `label`; si no, `chunk`;
    sin chunk, `label` o `self.name`.
  - `Pipeline.run(action=None, **kwargs)`: como hoy más: pone `result`/`line` a `None`; llama a `start` la primera
    vez; captura `PipelineEnd` (guarda `result`, salta las etapas restantes y registra
    `'Pipeline {name}.{action} ended: {status} ({reason})'`); al final, si `self.result is not None` y
    `self.line is None`, `self.line = self.summary(chunk, **kwargs)`; todos sus `self.app.logger.info(...)` pasan
    a `self.log('info', ...)`; la cabecera de `Load` sin filas y sin `target` es `'--> Load...'`;
    `self.compression` sale de `getattr(getattr(self.app.config, 'app', None), 'compression', None)` (sin
    `config`, `None`).

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_worker_unit.py` (conservar el test que ya hay):

```python
import pytest

from mtpy.core.worker import Pipeline, PipelineEnd


class Counter(Pipeline):
    """Pipeline sin modelos: anota las tareas, termina antes de tiempo en el chunk `skip` y rompe en `boom`."""
    label = 'count'

    def __init__(self):
        super().__init__()
        self.pipes = {'update': {'Setup': ['check'], 'Load': ['work'], 'Set': ['close']}}
        self.calls, self.started, self.finished, self.kwargs = [], 0, 0, None

    def start(self, **kwargs):
        self.started += 1
        self.kwargs = kwargs

    def check_task(self, chunk=None, **kwargs):
        self.calls.append(('check', chunk))
        if chunk == 'skip':
            self.end({'status': 'skipped', 'reason': 'nothing'})

    def work_task(self, chunk=None, **kwargs):
        self.calls.append(('work', chunk))
        if chunk == 'boom':
            raise KeyError('x')
        self.result = {'status': 'failed' if chunk == 'bad' else 'done', 'chunk': chunk}

    def close_task(self, chunk=None, **kwargs):
        self.calls.append(('close', chunk))

    def summary(self, chunk, **kwargs):
        return '{}: {}'.format(chunk, self.result['status'])

    def finish(self, **kwargs):
        self.finished += 1
        self.line = 'closed'


def test_pipeline_runs_without_config_nor_logger(fresh_app):
    p = Counter()
    p.run('update', chunk='a', position=(1, 2), extra=1)
    assert p.compression is None and p.result == {'status': 'done', 'chunk': 'a'} and p.line == 'a: done'
    assert p.calls == [('check', 'a'), ('work', 'a'), ('close', 'a')]


def test_pipeline_starts_once_with_the_kwargs_of_the_first_run_but_the_chunk(fresh_app):
    p = Counter()
    p.run('update', chunk='a', position=(1, 2), extra=1)
    p.run('update', chunk='b', position=(2, 2), extra=1)
    assert p.started == 1 and p.kwargs == {'position': (1, 2), 'extra': 1}


def test_pipeline_end_keeps_the_result_and_skips_the_remaining_stages(fresh_app):
    p = Counter()
    p.run('update', chunk='skip')
    assert p.result == {'status': 'skipped', 'reason': 'nothing'} and p.line == 'skip: skipped'
    assert p.calls == [('check', 'skip')]


def test_pipeline_resets_result_and_line_on_each_run(fresh_app):
    p = Counter()
    p.run('update', chunk='a')
    with pytest.raises(KeyError):
        p.run('update', chunk='boom')
    assert p.result is None and p.line is None


def test_pipeline_keeps_a_line_set_by_a_task(fresh_app):
    class Lined(Counter):
        def close_task(self, chunk=None, **kwargs):
            self.line = 'mine'
    p = Lined()
    p.run('update', chunk='a')
    assert p.line == 'mine'


def test_failure_name_uses_the_label(fresh_app):
    p = Counter()
    assert p.failure_name('es') == 'es (count)' and p.failure_name(None) == 'count'
    p.label = None
    assert p.failure_name('es') == 'es' and p.failure_name(None) == p.name


def test_default_hooks_do_nothing(fresh_app):
    class Plain(Pipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': []}}
    p = Plain()
    p.run('update')
    assert p.result is None and p.line is None and p.finish() is None and p.chunked is True
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_worker_unit.py -q`
Expected: FAIL (`AttributeError: 'NoneType' object has no attribute 'app'` en `Pipeline.__init__`, `ImportError`
de `PipelineEnd`).

- [ ] **Step 3: Implementar `Core.log` y los cambios de `Pipeline`**

`Core.log` en `mtpy/core/app.py`. En `worker.py`: `PipelineEnd`, los atributos de clase, la tolerancia de
`__init__`, el `start` perezoso, el `try/except PipelineEnd` alrededor del bloque de etapas de `run`, el
`summary` al final, `failure_name`, `end`, `finish`, `summary`, y `self.log('info', ...)` en lugar de
`self.app.logger.info(...)` sólo en `run` (los `debug` de las tareas por defecto, que ya exigen un `target`, quedan
como están). Docstrings numpy en los métodos nuevos.

- [ ] **Step 4: Comprobar que pasan y que nada más se rompe**

Run: `python -m pytest tests/test_worker_unit.py tests/test_jobs_progress.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/core/app.py mtpy/core/worker.py tests/test_worker_unit.py
git commit -m "core: Pipeline con result/line, start/finish, fin anticipado y tolerancia sin config ni logger"
```

---

### Task 2: `Job.alert` recortado y a prueba de Pushover, `Job.dispatch` avisa de un reventón, `JobInterrupted`, `Worker` resistente

**Files:**
- Modify: `mtpy/core/worker.py` (`Worker.handle_message` 114-128, `Job` 131-188)
- Test: `tests/test_worker_unit.py`

**Interfaces:**
- Produces, en `mtpy/core/worker.py`:
  - `class JobInterrupted(BaseException)` (la señal de parada convertida en excepción; no es `Exception` para que
    ningún bucle de chunks la trague).
  - `Job.ALERT_LIMIT = 1024`, `Job.TRUNCATED = '… (resumen recortado; el texto completo está en el log)'`,
    `Job.alerted: bool` (instancia, `False` en `__init__` y al empezar `dispatch`).
  - `Job.alert(self, msg: str) -> None`: el texto entero a `self.log('info', msg)`; a Pushover
    `msg` si cabe en `ALERT_LIMIT`, si no `msg[:ALERT_LIMIT - len(TRUNCATED) - 1].rstrip() + '\n' + TRUNCATED`;
    `ps.send` dentro de `try/except Exception` → `self.log('error', 'alert: {Type}: {e}')`; pone
    `self.alerted = True`. Sin `ps` ni logger no hace nada.
  - `Job.dispatch(**kwargs)`: envuelve `self.run(**self.params)` en `try/except BaseException as e`: si no
    `self.alerted`, `self.log('error', '{name} job failed')` y `self.alert('{name} job interrupted: {e}')` cuando
    `e` es `JobInterrupted` o `KeyboardInterrupt`, si no `self.alert('{name} job failed: {Type}: {e}')`; después
    `raise`. `after_dispatch` sólo si `run` no lanzó.
  - `Worker.handle_message(msg)`: `job.dispatch(**params)` dentro de `try/except Exception as e` →
    `self.log('error', 'job {name} failed: {e}')` y sigue (una `BaseException` propaga).

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_worker_unit.py`:

```python
import json
import types

from mtpy.core.worker import Job, JobInterrupted, Worker


class Recorder:
    """Pushover falso: guarda los mensajes o rompe."""
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, msg, **kwargs):
        if self.fail:
            raise RuntimeError('pushover down')
        self.sent.append(msg)


def quiet_logger(infos=None, errors=None):
    return types.SimpleNamespace(info=(infos if infos is not None else []).append,
                                 error=(errors if errors is not None else []).append)


class Noop(Job):
    def run(self, **kwargs):
        pass


def test_alert_truncates_pushover_to_1024_and_logs_the_whole_text(fresh_app):
    ps, infos = Recorder(), []
    fresh_app.set('ps', ps)
    fresh_app.set('logger', quiet_logger(infos))
    text = '\n'.join('line {:03d} {}'.format(i, 'x' * 40) for i in range(40))
    job = Noop()
    job.alert(text)
    assert infos == [text] and len(ps.sent[0]) <= Job.ALERT_LIMIT and ps.sent[0].endswith('\n' + Job.TRUNCATED)
    assert ps.sent[0].startswith('line 000') and job.alerted
    job.alert('short')
    assert ps.sent[1] == 'short'


def test_alert_never_raises_when_pushover_fails(fresh_app):
    errors = []
    fresh_app.set('ps', Recorder(fail=True))
    fresh_app.set('logger', quiet_logger(errors=errors))
    Noop().alert('hi')
    assert len(errors) == 1 and 'pushover down' in errors[0]


def test_alert_without_logger_nor_pushover_is_silent(fresh_app):
    Noop().alert('hi')


def test_dispatch_alerts_when_run_raises_and_reraises(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Boom(Job):
        def run(self, **kwargs):
            raise ValueError('bad params')
    with pytest.raises(ValueError, match='bad params'):
        Boom().dispatch(x=1)
    assert ps.sent == ['Boom job failed: ValueError: bad params']


def test_dispatch_does_not_alert_twice(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Alerted(Job):
        def run(self, **kwargs):
            self.alert('summary')
            raise RuntimeError('failed')
    with pytest.raises(RuntimeError):
        Alerted().dispatch()
    assert ps.sent == ['summary']


def test_dispatch_names_an_interruption(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Stopped(Job):
        def run(self, **kwargs):
            raise JobInterrupted('SIGTERM')
    with pytest.raises(JobInterrupted):
        Stopped().dispatch()
    assert ps.sent == ['Stopped job interrupted: SIGTERM']


def test_worker_logs_a_failing_job_and_goes_on(fresh_app):
    errors = []
    fresh_app.set('logger', quiet_logger(errors=errors))

    class Boom(Job):
        def run(self, **kwargs):
            raise ValueError('x')
    worker = Worker.__new__(Worker)
    worker.app = fresh_app
    worker.namespace = types.SimpleNamespace(Boom=Boom)
    worker.handle_message(json.dumps({'job': 'boom', 'params': {}}))
    assert any('Boom' in e and 'x' in e for e in errors)
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_worker_unit.py -q`
Expected: FAIL (`ImportError: JobInterrupted`, `AttributeError` de `ALERT_LIMIT`).

- [ ] **Step 3: Implementar**

`JobInterrupted`, `alert`, `dispatch` y `handle_message` como en Interfaces. `Job.__init__` pone
`self.alerted = False`.

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_worker_unit.py tests/test_jobs_check_s3.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/core/worker.py tests/test_worker_unit.py
git commit -m "core: Job.alert recortado y sin excepciones, dispatch avisa del reventón, Worker sigue tras un job fallido"
```

---

### Task 3: `SequenceJob`: `select` con `grupo.pipeline`, `stop_on_error`, chunks, posición, ganchos, resumen y salida

**Files:**
- Modify: `mtpy/core/worker.py` (`SequenceJob` 191-271)
- Test: `tests/test_worker_unit.py`

**Interfaces:**
- Produces, en `mtpy/core/worker.py`:
  - `SequenceJob.stop_on_error: bool = True`, `SequenceJob.chunk_noun: str = 'chunks'` (atributos de clase);
    `self.results: dict[tuple[str, Optional[str]], dict | list]` (clave `('grupo.pipeline', chunk)`, en orden
    de ejecución) y `self.lines: list[str]`, vacíos en `__init__` y al empezar `run`.
  - `SequenceJob.select_pipelines(self, select) -> list[tuple[str, str, str]]`: `(grupo, pipeline, 'grupo.Camel')`
    en el orden de `sequence`; `select` es `None`/`''` (todo), una cadena separada por comas o una lista; una
    entrada es `grupo` (todos los suyos) o `grupo.pipeline`; una entrada desconocida →
    `ValueError('{job}: unknown select "{entry}"')` (`job` = `self.name.lower()`).
  - `SequenceJob.run(self, select=None, action=None, chunks=None, stop_on_error=None, **kwargs) -> dict`:
    - `stop_on_error` debe ser `None` o `bool`, si no `ValueError('{job}: stop_on_error must be true or false')`.
    - Por cada pipeline del plan: una instancia; si `pipeline.chunked` y hay chunks, una ejecución por chunk con
      `position=(i, n)` (desde 1); si no, una sola con `chunk=None, position=(1, 1)`. Cada ejecución:
      `pipeline.run(action, chunk=chunk, position=position, **kwargs)` dentro de `try/except Exception as e`:
      con excepción, `self.log('error', ...)` (traceback), `result = {'status': 'failed', 'reason': '{Type}: {e}'}` y
      `line = '{chunk o grupo.pipeline}: failed ({reason})'`; sin ella, `result`/`line` del pipeline. Si
      `result is not None` va a `self.results[(key, chunk)]`; `line` (cadena o lista) va a `self.lines`. Un
      resultado es fallo si es un dict con `status == 'failed'` o una lista con algún elemento así. Fallo →
      `failures.append(pipeline.failure_name(chunk))`; en modo parar, tras el `finish` del pipeline (abajo),
      relanza la excepción del chunk o, si fue por estado, `RuntimeError(mensaje final)`.
    - Tras el último chunk de cada pipeline que llegó a ejecutarse (también los `chunked = False`), una vez:
      `pipeline.result = pipeline.line = None`; `name = pipeline.finish(**kwargs)` en `try/except Exception as e`
      (→ `self.log('error', ...)`, `name = key`, `pipeline.line = '{key}: finish failed ({Type}: {e})'`); si
      `pipeline.line` → `self.lines`; si `name` → `failures.append(name)` (y en modo parar relanza
      `RuntimeError`).
    - Resumen (`_summary(tstart, failures, interrupted=None)`): cabecera `'{name} job executed in {strfdelta(delta)} :-)'`
      sin fallos, `'{name} job executed in {delta} with {n} failure(s)'` con fallos (`1 failure`, `2 failures`),
      `'{name} job interrupted after {delta}'` si interrumpido; después las líneas de los chunks y `finish` fallidos
      (`self._failed_lines: list[str]`, anotadas en el momento en que se producen) y luego las demás, en su
      orden. Si `self.app.verbose` es falso, cada línea de `self.lines` se imprime con `print` antes del aviso (con
      verbosidad el logger ya la saca por consola). Luego `self.alert('\n'.join(...))`.
    - Todo el bucle va en `try/except BaseException as e`: si `e` no es `Exception` (interrupción),
      `self.lines.append('interrumpido en {key}{.chunk}: {e}')` y esa línea cuenta como fallida; en todo caso
      resumen con `interrupted=e` cuando no es `Exception`, y `raise`.
    - Al acabar sin interrupción: resumen; si `failures`,
      `raise RuntimeError('{job}: failed {chunk_noun}: {", ".join(failures)}')`; si no, `return self.results`.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_worker_unit.py` (usa `Counter`, `Recorder` y `quiet_logger` de las tareas 1 y 2):

```python
from mtpy.core.worker import SequenceJob


class Alpha(Counter):
    def work_task(self, chunk=None, **kwargs):
        if chunk == 'stop':
            raise JobInterrupted('SIGTERM')
        if chunk == 'many':
            self.result = [{'status': 'done'}, {'status': 'failed', 'reason': 'second'}]
            return
        super().work_task(chunk=chunk, **kwargs)

    def summary(self, chunk, **kwargs):
        if isinstance(self.result, list):
            return ['{}: {}'.format(chunk, r['status']) for r in self.result]
        return super().summary(chunk, **kwargs)


class Beta(Counter):
    chunked = False
    label = 'beta'

    def finish(self, **kwargs):
        self.finished += 1
        if kwargs.get('break_finish'):
            self.line = 'beta: finish failed (boom)'
            return 'beta-close'
        self.line = 'beta closed'


class Gamma(Counter):
    label = None

    def start(self, **kwargs):
        if kwargs.get('break_start'):
            raise RuntimeError('no fs')
        super().start(**kwargs)


class Seq(SequenceJob):
    sequence = {'grp': ['alpha', 'beta'], 'other': ['gamma']}
    stop_on_error = False
    chunk_noun = 'scopes'

    def __init__(self):
        super().__init__()
        self.made = []
        make = lambda cls: (lambda: self.made.append(cls()) or self.made[-1])
        self.namespace = types.SimpleNamespace(**{'grp.Alpha': make(Alpha), 'grp.Beta': make(Beta),
                                                  'other.Gamma': make(Gamma)})


def test_select_accepts_groups_and_group_dot_pipeline_in_sequence_order(fresh_app):
    job = Seq()
    assert job.select_pipelines('other,grp.beta') == [('grp', 'beta', 'grp.Beta'), ('other', 'gamma', 'other.Gamma')]
    assert [p for _, p, _ in job.select_pipelines(None)] == ['alpha', 'beta', 'gamma']
    assert job.select_pipelines(['grp']) == [('grp', 'alpha', 'grp.Alpha'), ('grp', 'beta', 'grp.Beta')]
    with pytest.raises(ValueError, match='seq: unknown select "grp.nope"'):
        job.select_pipelines('grp.nope')


def test_continue_mode_runs_every_chunk_and_reports_at_the_end(fresh_app, capsys):
    ps, infos = Recorder(), []
    fresh_app.set('ps', ps)
    fresh_app.set('logger', quiet_logger(infos))
    job = Seq()
    with pytest.raises(RuntimeError, match=r'seq: failed scopes: bad \(count\), boom \(count\)$'):
        job.run(select='grp', chunks=['a', 'bad', 'boom', 'c'])
    alpha, beta = job.made
    assert [c for _, c in alpha.calls if _ == 'work'] == ['a', 'bad', 'boom', 'c']
    assert beta.calls == [('check', None), ('work', None), ('close', None)] and beta.kwargs['position'] == (1, 1)
    assert alpha.finished == beta.finished == 1
    assert job.results[('grp.alpha', 'a')] == {'status': 'done', 'chunk': 'a'}
    assert job.results[('grp.alpha', 'boom')] == {'status': 'failed', 'reason': "KeyError: 'x'"}
    assert ('grp.beta', None) in job.results
    assert job.lines[:5] == ['a: done', 'bad: failed', "boom: failed (KeyError: 'x')", 'c: done', 'closed']
    text = ps.sent[-1].splitlines()
    assert text[0].startswith('Seq job executed in ') and text[0].endswith(' with 2 failures')
    assert text[1:3] == ['bad: failed', "boom: failed (KeyError: 'x')"] and text[3] == 'a: done'
    out = capsys.readouterr().out
    assert 'bad: failed' in out and out.index('a: done') < out.index('bad: failed')  # verbose falso: se imprime


def test_every_pipeline_of_the_plan_gets_every_chunk_with_its_position(fresh_app):
    job = Seq()
    job.run(chunks=['a', 'c'])
    alpha, beta, gamma = job.made
    assert [c for _, c in gamma.calls if _ == 'work'] == ['a', 'c'] and gamma.kwargs['position'] == (1, 2)
    assert job.results[('other.gamma', 'c')]['status'] == 'done' and gamma.failure_name('c') == 'c'


def test_summary_is_not_printed_when_the_logger_echoes_the_console(fresh_app, capsys):
    fresh_app.set('verbose', 1)
    Seq().run(select='grp.alpha', chunks=['a'])
    assert 'a: done' not in capsys.readouterr().out


def test_stop_mode_finishes_the_pipeline_alerts_and_reraises(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)
    job = Seq()
    with pytest.raises(KeyError):
        job.run(chunks=['a', 'boom', 'c'], stop_on_error=True)
    alpha = job.made[0]
    assert len(job.made) == 1 and alpha.finished == 1
    assert [c for _, c in alpha.calls if _ == 'work'] == ['a', 'boom']
    assert ps.sent[-1].splitlines()[0].endswith(' with 1 failure')
    job = Seq()
    with pytest.raises(RuntimeError, match=r'seq: failed scopes: bad \(count\)'):
        job.run(chunks=['bad'], stop_on_error=True)


def test_stop_on_error_must_be_a_boolean(fresh_app):
    with pytest.raises(ValueError, match='stop_on_error must be true or false'):
        Seq().run(chunks=['a'], stop_on_error='false')


def test_a_finish_failure_is_a_failure_of_the_job_with_its_own_line(fresh_app):
    job = Seq()
    with pytest.raises(RuntimeError, match='seq: failed scopes: beta-close'):
        job.run(select='grp.beta', break_finish=True)
    assert job.lines == ['None: done', 'beta: finish failed (boom)']


def test_a_finish_exception_is_logged_and_counted(fresh_app):
    class Broken(Beta):
        def finish(self, **kwargs):
            raise OSError('s3 down')
    job = Seq()
    job.namespace = types.SimpleNamespace(**{'grp.Beta': Broken})
    with pytest.raises(RuntimeError, match='seq: failed scopes: grp.beta'):
        job.run(select='grp.beta')
    assert job.lines[-1] == 'grp.beta: finish failed (OSError: s3 down)'


def test_list_results_fail_when_any_item_failed(fresh_app):
    job = Seq()
    with pytest.raises(RuntimeError, match=r'many \(count\)'):
        job.run(select='grp.alpha', chunks=['many'])
    assert job.lines == ['many: done', 'many: failed', 'closed']


def test_a_pipeline_whose_start_raises_fails_every_chunk(fresh_app):
    job = Seq()
    with pytest.raises(RuntimeError, match='seq: failed scopes: a, b'):
        job.run(select='other', chunks=['a', 'b'], break_start=True)
    assert job.lines[:2] == ['a: failed (RuntimeError: no fs)', 'b: failed (RuntimeError: no fs)']


def test_an_interruption_alerts_and_propagates_without_running_more_chunks(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)
    job = Seq()
    with pytest.raises(JobInterrupted):
        job.run(chunks=['a', 'stop', 'c'])
    alpha = job.made[0]
    assert [c for _, c in alpha.calls if _ == 'work'] == ['a', 'stop'] and len(job.made) == 1
    text = ps.sent[-1].splitlines()
    assert text[0].startswith('Seq job interrupted after ') and text[1] == 'interrumpido en grp.alpha.stop: SIGTERM'
    assert job.alerted


def test_existing_sequence_jobs_keep_their_behaviour(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Plain(Pipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': []}}

    class Old(SequenceJob):
        sequence = {'grp': ['plain']}

        def __init__(self):
            super().__init__()
            self.namespace = types.SimpleNamespace(**{'grp.Plain': Plain})
    job = Old()
    assert job.stop_on_error is True and job.run(chunks='a, b') == {}
    assert len(ps.sent) == 1 and ps.sent[0].startswith('Old job executed in ') and ps.sent[0].endswith(' :-)')
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_worker_unit.py -q`
Expected: FAIL (`AttributeError: select_pipelines`, firmas).

- [ ] **Step 3: Implementar `SequenceJob`**

Reescribir `SequenceJob.run` según Interfaces, con `select_pipelines`, `_run_chunk`, `_finish` y `_summary` como
métodos privados. `strfdelta(delta)` como hoy (segundos enteros).

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_worker_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/core/worker.py tests/test_worker_unit.py
git commit -m "core: SequenceJob con select grupo.pipeline, stop_on_error, chunked, posición, ganchos y resumen final"
```

---

### Task 4: Lanzadores: `job.py` por `dispatch` con salida 1 y `SIGTERM`, `jobs/Pipeline.py` arreglado, `pipeline.py` con `finish`

**Files:**
- Modify: `job.py` (1-28), `pipeline.py` (1-33), `mtpy/jobs/Pipeline.py` (1-19)
- Test: `tests/test_launchers.py` (nuevo)

**Interfaces:**
- Produces: `job.main(name: str, params: Optional[str], namespace=None) -> int`: `mtpy.run()`, instala
  `job.interrupt` como manejador de `SIGTERM` (`job.interrupt(signum, frame)` lanza
  `JobInterrupted('SIGTERM')`), resuelve la clase con `to_camel` en `namespace or jobs`, `job.dispatch(**params)`
  (`{}` sin JSON); devuelve `0`; con `JobInterrupted` escribe `'{name}: interrupted ({e})'` en stderr y devuelve
  `1`; con cualquier `Exception`, `traceback.print_exc()` y `1`. `__main__` hace `sys.exit(main(...))`.
- Produces: `mtpy.jobs.Pipeline.Pipeline.run(select=None, action=None, chunks=None, **kwargs)`: sin `select` o sin
  punto → `ValueError('pipeline: "select" must be "group.pipeline"')`; con él, `self.sequence = {group: [name]}`
  (nombre en snake_case tal cual llega) y `super().run(action=action, chunks=chunks, **kwargs)`.
- Produces: `pipeline.main(name: str, action, chunks: list, namespace=None) -> int`: como hoy más
  `position=(i, n)` en cada `run`, `pipeline.finish()` al acabar y `print` de cada línea de `pipeline.line`
  acumulada (cadena o lista); devuelve `0`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_launchers.py`:

```python
"""Tests de los lanzadores de la raíz (`job.py`, `pipeline.py`) y del job `Pipeline`, sin base ni Pushover."""
import signal
import types

import pytest

import job
import pipeline
from mtpy.core.worker import Job, JobInterrupted, Pipeline as CorePipeline, SequenceJob


@pytest.fixture
def booted(monkeypatch, fresh_app):
    monkeypatch.setattr(job.mtpy, 'run', lambda: fresh_app)
    monkeypatch.setattr(pipeline.mtpy, 'run', lambda: fresh_app)
    saved = signal.getsignal(signal.SIGTERM)
    yield fresh_app
    signal.signal(signal.SIGTERM, saved)


def test_job_main_dispatches_the_json_and_returns_0(booted):
    seen = []

    class FakeJob(Job):
        def dispatch(self, **kwargs):
            seen.append(kwargs)
    assert job.main('fake_job', '{"a": 1}', namespace=types.SimpleNamespace(FakeJob=FakeJob)) == 0
    assert job.main('fake_job', None, namespace=types.SimpleNamespace(FakeJob=FakeJob)) == 0
    assert seen == [{'a': 1}, {}]


def test_job_main_returns_1_with_the_traceback_when_the_job_fails(booted, capsys):
    class Boom(Job):
        def run(self, **kwargs):
            raise ValueError('bad')
    assert job.main('boom', '{}', namespace=types.SimpleNamespace(Boom=Boom)) == 1
    assert 'ValueError: bad' in capsys.readouterr().err


def test_job_main_installs_a_sigterm_handler_that_interrupts(booted, capsys):
    class Waits(Job):
        def run(self, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert job.main('waits', '{}', namespace=types.SimpleNamespace(Waits=Waits)) == 1
    assert 'waits: interrupted (SIGTERM)' in capsys.readouterr().err
    with pytest.raises(JobInterrupted):
        job.interrupt(signal.SIGTERM, None)


def test_pipeline_job_requires_group_dot_pipeline(fresh_app):
    from mtpy.jobs.Pipeline import Pipeline
    with pytest.raises(ValueError, match='group.pipeline'):
        Pipeline().run(select='forecast')
    ran = []

    class Fake(CorePipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': ['mark']}}

        def mark_task(self, chunk=None, **kwargs):
            ran.append(chunk)
    launcher = Pipeline()
    launcher.namespace = types.SimpleNamespace(**{'grp.Fake': Fake})
    launcher.run(select='grp.fake', chunks='a,b')
    assert launcher.sequence == {'grp': ['fake']} and ran == ['a', 'b']


def test_pipeline_main_runs_each_chunk_with_its_position_finishes_and_prints_the_lines(booted, capsys):
    seen = []

    class Fake(CorePipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': ['mark']}}

        def mark_task(self, chunk=None, position=None, **kwargs):
            seen.append((chunk, position))
            self.result, self.line = {'status': 'done'}, '{}: done'.format(chunk)

        def finish(self, **kwargs):
            seen.append('finish')
    assert pipeline.main('grp.fake', 'update', ['a', 'b'], namespace=types.SimpleNamespace(**{'grp.Fake': Fake})) == 0
    assert seen == [('a', (1, 2)), ('b', (2, 2)), 'finish']
    assert capsys.readouterr().out.splitlines()[-2:] == ['a: done', 'b: done']
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_launchers.py -q`
Expected: FAIL (`main()` sin `namespace`, `interrupt` no existe, `module` no definido).

- [ ] **Step 3: Implementar los tres lanzadores**

Según Interfaces. En `job.py`, `interrupt` es una función de módulo; `main` acumula los `line` de cada chunk en
`pipeline.py` (cadena → una línea, lista → varias).

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_launchers.py tests/test_worker_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add job.py pipeline.py mtpy/jobs/Pipeline.py tests/test_launchers.py
git commit -m "jobs: job.py despacha con salida 1 y SIGTERM, pipeline.py cierra con finish, job Pipeline exige grupo.pipeline"
```

---

### Task 5: Grupo `load` (`Polls`, `Params`), `ScopePipeline` y el job `Load` como `SequenceJob`

**Files:**
- Create: `mtpy/pipelines/base.py`, `mtpy/pipelines/load/__init__.py` (vacío), `mtpy/pipelines/load/Polls.py`,
  `mtpy/pipelines/load/Params.py`
- Modify: `mtpy/jobs/Load.py` (todo; hoy `run` 59-140, `_scope` 180-228, `_line` 230-253, `_scope_params`
  255-292, `_line_params` 294-302, `STEPS` 8-9)
- Test: `tests/test_pipelines_load.py` (nuevo), `tests/test_jobs_load.py`

**Interfaces:**
- Produces: `mtpy/pipelines/base.py`:
  ```python
  class ScopePipeline(ScopeReporter, Pipeline):
      """Pipeline que trabaja ámbito a ámbito: progreso en vivo, avisos recogidos y fallos como resultados."""
      steps = None            # etiquetas de las fases (`ScopeProgress`)
      def start(self, verbose=1, position=(1, 1), **kwargs) -> None   # self.verbose, self.model_verbose = max(0, verbose - 2), self.total = position[1], self.progress = None
      def progress_of(self, chunk, position=(1, 1), day=None) -> ScopeProgress   # self._progress(chunk, *position, day, self.steps)
      def failed(self, scope, e: Exception, **extra) -> dict    # self._log_error(scope); {'status': 'failed', 'reason': '{Type}: {e}', **extra}
      def line_of(self, chunk, result, **kwargs) -> str          # abstracto: la línea de resumen de un resultado
      def summary(self, chunk, **kwargs)                         # line = self.line_of(chunk, self.result, **kwargs); si self.progress: self.progress.done(line); return line
  ```
  `ScopeReporter` viene de `mtpy/jobs/progress.py`; `self.name` es el del `Pipeline` (`'load.Polls'`).
- Produces: `load.Polls(ScopePipeline)`: `label = 'polls'`, `steps = STEPS` (el dict de `jobs/Load.py:8-9`,
  movido aquí), `pipes = {'update': {'Setup': ['resolve_event'], 'Load': ['load']}, 'rebulk': <igual>}`.
  `start(..., force=False, dry_run=False, **kw)` imprime la cabecera
  `'load polls: {n ámbitos}{ · overwrite}{ · force}{ · dry run}'` (`self._say`, nivel 0); `overwrite` es
  `self._action == 'rebulk'`, aquí y en la tarea `load` (el job no pasa `overwrite`, pasa la acción).
  `resolve_event_task(chunk, position, event_date=None, skip_missing=False, **kw)`:
  `self.progress = self.progress_of(chunk, position)`, `progress.start()`, `self.event = event_date or
  next_event_date(chunk)` en `try/except Exception as e` → `self.end({**self.failed(chunk, e), 'action': 'polls', 'warnings': []})`;
  sin evento → `self.end({'status': 'skipped' if skip_missing else 'failed', 'action': 'polls', 'reason': 'no upcoming event', 'warnings': []})`.
  `load_task(chunk, force=False, years=None, dry_run=False, loader=None, computer=None, catalogues=None, skip_missing=False, **kw)`:
  el cuerpo de `Load._scope` 209-228 con `load.load_polls(..., overwrite=self._action == 'rebulk', ...)` (`verbose=self.model_verbose`,
  `report=lambda text, level=1: self._say(text, level)`), la conversión a `skipped` con `skip_missing` y
  `NO_POLLS_ARTICLE`, `result['warnings'] = notes`; deja `self.result`. `line_of` = `Load._line` 230-253.
- Produces: `load.Params(ScopePipeline)`: `label = 'params'`, `steps = STEPS`,
  `pipes = {'update': {'Setup': ['prepare'], 'Load': ['load']}}`; cabecera `'load params: {n ámbitos}{ · dry run}'`;
  `prepare_task` crea el progreso y lo arranca; `load_task(chunk, dry_run=False, params_loader=None, skip_missing=False, **kw)`
  = `Load._scope_params` 280-292 con `(params_loader or load.load_params)(...)`; `line_of` = `_line_params`.
- Produces: `mtpy.jobs.Load.Load(SequenceJob)`: `sequence = {'load': ['polls', 'params']}`,
  `default_action = 'update'`, `stop_on_error = False`, `chunk_noun = 'scopes'`, `WHAT = ('polls', 'params')`;
  `run(self, what=('polls',), scopes=('es',), event_date=None, overwrite=False, force=False, years=None, dry_run=False, verbose=1, loader=None, computer=None, catalogues=None, params_loader=None, stop_on_error=None, **kwargs) -> dict`:
  validación de `what` y de `years`/`overwrite` como hoy (mismos mensajes), `years` a cadenas,
  `targets = resolve_scopes(scopes)` (import de módulo, como hoy, para el `monkeypatch`), vacío →
  `ValueError('load: no scopes to run')`; `super().run(select=['load.' + w for w in whats], action='rebulk' if overwrite else 'update', chunks=targets, stop_on_error=stop_on_error, event_date=..., force=..., years=..., dry_run=..., verbose=..., loader=..., computer=..., catalogues=..., params_loader=..., skip_missing=scopes == 'all')`;
  devuelve `{scope: {what: result}}` a partir de `self.results` (clave `('load.polls', scope)` → `what = 'polls'`),
  aplanado a `{scope: result}` con un solo `what`. Docstring de clase y de `run` conservados y actualizados
  (`stop_on_error`).

- [ ] **Step 1: Escribir los tests de los pipelines**

`tests/test_pipelines_load.py`, con `FakeLoader`, `FakeComputer`, `FakeCatalogues` de `tests/test_load_unit.py` y
`FakeParamsLoader` movido desde `tests/test_jobs_load.py:172-183` (se importa de aquí en los dos ficheros):

```python
"""Tests de los pipelines `load.Polls` y `load.Params` ejecutados sueltos, sin base."""
import pytest

from mtpy.pipelines.load.Params import Params
from mtpy.pipelines.load.Polls import Polls
from tests.test_load_unit import FakeCatalogues, FakeComputer, FakeLoader

EVENTS = {'es': '2026-11-29', 'es-md': '2027-05-23'}


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: EVENTS.get(scope))
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: event_date != '2027-05-23')


def polls(scope, position=(1, 1), action='update', **kwargs):
    p = Polls()
    p.run(action, chunk=scope, position=position, loader=FakeLoader(), computer=FakeComputer(),
          catalogues=FakeCatalogues(), verbose=0, **kwargs)
    return p


def test_polls_loads_a_scope_and_builds_its_line(fresh_app, events, capsys):
    p = polls('es', position=(1, 2))
    assert p.result['status'] == 'loaded' and p.result['action'] == 'polls' and p.result['warnings'] == []
    assert p.line == 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s'
    assert 'load polls: 2 ámbitos' in capsys.readouterr().out


def test_polls_rebulk_overwrites_and_flags_the_header(fresh_app, events, capsys):
    p = polls('es', action='rebulk', force=True, dry_run=True)
    assert p.line == 'es: polls checked (5 read, 5 new, 2026-11-29, overwrite, dry run) in 0 s'
    assert 'load polls: 1 ámbito · overwrite · force · dry run' in capsys.readouterr().out


def test_polls_without_article_fails_or_skips_with_all(fresh_app, events):
    assert polls('es-md').result == {'status': 'failed', 'action': 'polls', 'reason': 'no polls article for 2027-05-23', 'warnings': []}
    p = polls('es-md', skip_missing=True)
    assert p.result['status'] == 'skipped' and p.line == 'es-md: polls skipped (no polls article for 2027-05-23)'


def test_polls_without_upcoming_event_ends_in_setup(fresh_app, events):
    p = polls('es-xx')
    assert p.result == {'status': 'failed', 'action': 'polls', 'reason': 'no upcoming event', 'warnings': []}
    assert polls('es-xx', skip_missing=True).result['status'] == 'skipped'


def test_polls_event_lookup_error_fails_the_scope(fresh_app, events, monkeypatch):
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: {}['x'])
    p = polls('es')
    assert p.result['status'] == 'failed' and p.result['reason'] == "KeyError: 'x'" and 'failed' in p.line


def test_polls_failure_name_is_scope_and_action(fresh_app):
    assert Polls().failure_name('es') == 'es (polls)' and Params().failure_name('es') == 'es (params)'


def test_params_loads_a_scope_and_builds_its_line(fresh_app, capsys):
    loader = FakeParamsLoader()
    p = Params()
    p.run('update', chunk='es', position=(1, 2), params_loader=loader, verbose=1)
    assert loader.calls == [('es', True, 0)] and p.result['status'] == 'loaded'
    assert p.line == 'es: params loaded (3 elections, 2 changed, 1 skipped) in 0 s'
    out = capsys.readouterr().out
    assert 'load params: 2 ámbitos' in out and 'es 2023-07-23: bmaps.vs null -> {}' in out
```

Completar con los escenarios de `tests/test_jobs_load.py` que son del pipeline, llamando al pipeline directamente:
`test_load_years_are_passed_as_strings` (el pipeline recibe `years=['2026']`), `test_load_unmapped_names_fail_the_scope_and_force_adds_them`,
`test_load_line_counts_added_pollsters_and_sponsors_to_add`, `test_load_verbosity_levels`,
`test_load_isolates_a_failing_scope_and_reports_the_exception` (sólo la parte del resultado `failed` y la línea),
`test_load_params_dry_run_checks_and_flags_the_header`, `test_load_params_scope_without_elections_is_skipped_with_all_and_fails_when_named`
(resultado), `test_load_params_isolates_an_exception` (resultado). Mismos nombres con prefijo `test_polls_`/`test_params_`
y las mismas aserciones sobre `p.result`, `p.line` y la salida.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pipelines_load.py -q`
Expected: FAIL (`ModuleNotFoundError: mtpy.pipelines.load`).

- [ ] **Step 3: Implementar `base.py`, `Polls.py`, `Params.py`**

Según Interfaces, moviendo los cuerpos de `mtpy/jobs/Load.py`. `STEPS` queda en `Polls.py` y `Params.py` lo
importa de allí.

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_load.py -q`
Expected: PASS.

- [ ] **Step 5: Reescribir los tests del job**

`tests/test_jobs_load.py` conserva, adaptados, los tests de lo que es del job: `test_load_runs_every_scope_and_prints_the_summary`,
`test_load_scope_named_explicitly_without_article_fails`, `test_load_returns_the_result_of_each_scope`,
`test_load_header_flags_and_overwrite_line`, `test_load_event_date_parameter_and_scope_without_upcoming_event`,
`test_load_refuses_years_with_overwrite`, `test_load_unknown_action`, `test_load_params_runs_every_scope_and_prints_the_summary`,
`test_load_both_actions_run_polls_first` y `test_load_isolates_a_failing_scope_and_reports_the_exception` (el
`RuntimeError` final y las dos líneas). La fixture `events` pasa a parchear
`'mtpy.pipelines.load.Polls.next_event_date'`; `module.NO_EVENTS` pasa a `mtpy.lib.load.NO_EVENTS`. Añadir:

```python
def test_load_translates_what_and_scopes_and_returns_by_action(fresh_app, scopes, events):
    loader = FakeParamsLoader()
    out = run(FakeLoader(), what=['polls', 'params'], scopes=['es'], params_loader=loader)
    assert set(out['es']) == {'polls', 'params'} and out['es']['polls']['status'] == 'loaded'


def test_load_stop_on_error_stops_at_the_first_failed_scope(fresh_app, scopes, events, monkeypatch):
    loader = FakeParamsLoader(fail={'es': KeyError('boom')})
    with pytest.raises(RuntimeError, match=r'load: failed scopes: es \(params\)'):
        Load().run(what='params', scopes='all', params_loader=loader, stop_on_error=True)
    assert [call[0] for call in loader.calls] == ['es']


def test_load_refuses_an_empty_scope_list(fresh_app, scopes, events):
    with pytest.raises(ValueError, match='load: no scopes to run'):
        Load().run(scopes=[])


def test_load_alert_puts_the_failures_first(fresh_app, scopes, events):
    sent = []
    fresh_app.set('ps', types.SimpleNamespace(send=lambda msg, **kw: sent.append(msg)))
    with pytest.raises(RuntimeError):
        Load().run(what='params', scopes='all', params_loader=FakeParamsLoader(fail={'es-md': KeyError('boom')}))
    lines = sent[-1].splitlines()
    assert lines[0].endswith('with 1 failure') and lines[1].startswith('es-md: params failed') and lines[2].startswith('es: params loaded')
```

- [ ] **Step 6: Reescribir `mtpy/jobs/Load.py` como `SequenceJob`**

Según Interfaces. El docstring de clase conserva los ejemplos y gana una frase sobre `stop_on_error`.

- [ ] **Step 7: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_load.py tests/test_jobs_load.py tests/test_load_unit.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git status
git add mtpy/pipelines/base.py mtpy/pipelines/load mtpy/jobs/Load.py tests/test_pipelines_load.py tests/test_jobs_load.py
git commit -m "load: pipelines load.Polls y load.Params y el job como SequenceJob"
```

---

### Task 6: Grupo `compute` (`Catalogues`, `Compute`, `Ratings`) y el job `Compute`

**Files:**
- Create: `mtpy/pipelines/compute/__init__.py`, `Catalogues.py`, `Compute.py`, `Ratings.py`
- Modify: `mtpy/jobs/Compute.py` (todo; hoy `run` 53-140, `_scope` 142-181, `_catalogues` 183-206, `_line` 208-227,
  `STEPS`/`ACTIONS`/`CATALOGUES` 8-13)
- Test: `tests/test_pipelines_compute.py` (nuevo), `tests/test_jobs_compute.py`

**Interfaces:**
- Produces: `compute.Catalogues(ScopePipeline)`: `chunked = False`, `label = 'catálogos'`,
  `pipes = {'update': {'Load': ['replicate']}}`; cabecera `'compute catálogos{ · dry run}'`;
  `replicate_task(dry_run=False, catalogues=None, **kw)` = `Compute._catalogues` 199-206 con
  `(catalogues or replicate_catalogues)(save=not dry_run)`; `failure_name(chunk)` devuelve
  `'catalogues (catálogos)'`; `line_of('catalogues', result)` = las ramas `catálogos` de `Compute._line`.
- Produces: `compute.Compute(ScopePipeline)`: `label = 'series'`, `steps = STEPS` (dict de `jobs/Compute.py:8-9`,
  aquí), `ACTION = 'compute'`, `pipes = {'update': {'Setup': ['prepare'], 'Load': ['compute']}}`; cabecera
  `'compute series: {n ámbitos}{ · dry run}'`; `prepare_task(chunk, position, **kw)` arranca el progreso;
  `compute_task(chunk, dry_run=False, computer=None, featured=None, **kw)` = `Compute._scope` 167-181 con
  `compute_scope(...)`; `line_of` = ramas `series` de `_line`. `compute.Ratings(Compute)`: `label = 'ratings'`,
  cabecera `'compute ratings: …'`, la tarea llama a `rate_scope(scope, save, verbose, progress, computer)`;
  `line_of` rama `ratings`. Ambos conservan `'action': 'series' | 'ratings'` en el resultado.
- Produces: `mtpy.jobs.Compute.Compute(SequenceJob)`: `sequence = {'compute': ['catalogues', 'compute', 'ratings']}`,
  `default_action = 'update'`, `stop_on_error = False`, `chunk_noun = 'scopes'`, `WHAT`;
  `run(self, what=('compute', 'ratings'), scopes=('es',), dry_run=False, verbose=1, computer=None, featured=None, catalogues=None, stop_on_error=None, **kwargs) -> dict`:
  validación de `what` (mismo mensaje), `targets = resolve_scopes(scopes)` (import de módulo), vacío →
  `ValueError('compute: no scopes to run')`, `super().run(select=['compute.' + w for w in whats], chunks=targets, ...)`;
  devuelve `{scope: result}` (último que corrió sobre el ámbito) y `results['catalogues']` del pipeline
  `Catalogues` (clave `('compute.catalogues', None)`).

- [ ] **Step 1: Escribir los tests de los pipelines**

`tests/test_pipelines_compute.py` con `FakeComputer` de `tests/test_compute_unit.py`: los escenarios de
`tests/test_jobs_compute.py` que son del pipeline, ejecutándolo suelto (`Compute().run('update', chunk='es',
position=(1, 2), computer=factory, featured=..., verbose=0)`): `test_compute_series_line_and_result`
(`p.line == 'es: series computed (2 polls, 2 events, 7 featured) in 0 s'`, `p.result['action'] == 'series'`),
`test_ratings_line_and_result`, `test_compute_failure_is_a_failed_result_with_its_reason`
(`FakeComputer(fail_at='errors', fail_scope='es')` → `'es: series failed (RuntimeError: boom errors)'`),
`test_compute_dry_run_saves_nothing_and_says_so`, `test_compute_verbosity_levels`,
`test_compute_reports_the_model_warnings_as_notes`, `test_catalogues_runs_once_and_dry_run_only_checks`
(`Catalogues().run('update', chunk=None, catalogues=fake)` → `p.result['status'] == 'computed'`,
`p.line.startswith('catalogues: saved (')`, `p.failure_name(None) == 'catalogues (catálogos)'`,
`Catalogues.chunked is False`), `test_catalogues_failure_result`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pipelines_compute.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar los tres pipelines**

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_compute.py -q`
Expected: PASS.

- [ ] **Step 5: Reescribir los tests del job**

`tests/test_jobs_compute.py` conserva `test_compute_runs_the_series_of_every_scope_and_then_the_ratings`,
`test_compute_isolates_a_failing_scope_and_fails_at_the_end`, `test_compute_ratings_only_and_unknown_actions`,
`test_compute_catalogues_runs_first_once_and_dry_run_only_checks`, `test_compute_catalogues_failure_fails_the_job_at_the_end`
(con `match=r'catalogues \(catálogos\)'`), y añade:

```python
def test_compute_stop_on_error_stops_before_the_ratings(fresh_app, scopes):
    factory = FakeComputer(fail_at='errors', fail_scope='es')
    with pytest.raises(RuntimeError, match=r'compute: failed scopes: es \(series\)'):
        run(factory, what=['compute', 'ratings'], scopes=['es', 'es-md'], stop_on_error=True)
    assert [scope for scope, _ in factory.inits] == ['es'] * 3  # ni es-md ni los ratings


def test_compute_refuses_an_empty_scope_list(fresh_app, scopes):
    with pytest.raises(ValueError, match='compute: no scopes to run'):
        run(FakeComputer(), scopes=[])
```

- [ ] **Step 6: Reescribir `mtpy/jobs/Compute.py`**

- [ ] **Step 7: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_compute.py tests/test_jobs_compute.py tests/test_compute_unit.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git status
git add mtpy/pipelines/compute mtpy/jobs/Compute.py tests/test_pipelines_compute.py tests/test_jobs_compute.py
git commit -m "compute: pipelines Catalogues, Compute y Ratings y el job como SequenceJob"
```

---

### Task 7: Grupo `backtest` (`Backtest`, acción `run`) y el job `Backtest`

**Files:**
- Create: `mtpy/pipelines/backtest/__init__.py`, `mtpy/pipelines/backtest/Backtest.py`
- Modify: `mtpy/jobs/Backtest.py` (todo; hoy `run` 42-142, `_composition` 144-172, `_shown` 174-180, `_scope`
  182-215, `_line` 217-225)
- Test: `tests/test_pipelines_backtest.py` (nuevo), `tests/test_jobs_backtest.py`

**Interfaces:**
- Produces: `backtest.Backtest(ScopePipeline)`: `label = None` (la etiqueta de fallo es el ámbito, como hoy
  `backtest: failed scopes: es-ar`), `pipes = {'run': {'Setup': ['prepare'], 'Load': ['evaluate']}}`;
  `start(..., n_sim=1000, out=None, **kw)` imprime `'backtest: {n ámbitos} · {n_sim} simulaciones · {shown(out)}'`
  (`shown` = `Backtest._shown`, función de módulo); `prepare_task` arranca el progreso;
  `evaluate_task(chunk, out=None, events=None, horizons=None, n_sim=1000, seed=42, max_fc=10, nowcast_only=False, house_effects=True, industry_bias=False, dispersion=True, herding=True, composition=None, regional_noise=True, min_events=None, recompute_ratings=False, runner=None, stats=None, **kw)`
  = `Backtest._scope` 204-215 (`verbose=max(0, self.verbose - 1)`), y si el ámbito se completó y
  `self.verbose >= 1`, `print(table.round(3).to_string(index=False), flush=True)`; `line_of` = `_line`.
- Produces: `mtpy.jobs.Backtest.Backtest(SequenceJob)`: `sequence = {'backtest': ['backtest']}`,
  `default_action = 'run'`, `stop_on_error = False`, `chunk_noun = 'scopes'`;
  `run(self, scopes=('es',), events=None, horizons=None, n_sim=1000, seed=42, max_fc=10, nowcast_only=False, house_effects=True, industry_bias=False, dispersion=True, herding=True, composition=None, regional_noise=True, min_events=None, recompute_ratings=False, out=None, verbose=1, runner=None, stats=None, stop_on_error=None, **kwargs) -> dict`:
  `composition = self._composition(composition)` (sigue siendo el `staticmethod` del job: falla antes de
  empezar), `targets = resolve_scopes(scopes)` (import de módulo), vacío → `ValueError('backtest: no scopes to run')`,
  `super().run(select='backtest', chunks=targets, ...)`; devuelve `{scope: result}`.

- [ ] **Step 1: Escribir los tests del pipeline**

`tests/test_pipelines_backtest.py` con `FakeRunner`, `patched` (parchea `mtpy.lib.backtest.default_events` y
`common.provenance`, ya no `module.resolve_scopes`) y `stats` de `tests/test_jobs_backtest.py`:
`test_backtest_run_writes_the_folder_and_builds_its_line` (`Backtest().run('run', chunk='es-md', position=(2, 2),
out=str(tmp_path), n_sim=20, verbose=0, runner=runner, stats=stats)` → `p.result['folder'] == str(tmp_path / 'es-md')`,
`p.line == 'es-md: backtest completed (4 cases, 2 events, 2 horizons) in 0 s'`, cabecera
`'backtest: 2 ámbitos · 20 simulaciones · {tmp_path}'`), `test_backtest_skips_a_scope_without_elections_or_cases`,
`test_backtest_failure_result_and_line` (`"es-ar: failed (KeyError: 'x')"`, `p.failure_name('es-ar') == 'es-ar'`),
`test_backtest_verbosity_levels_and_the_horizon_table`, `test_backtest_reports_the_skipped_cases_as_notes`,
`test_backtest_passes_every_parameter_through_and_records_it`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pipelines_backtest.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementar el pipeline**

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Reescribir los tests del job**

`tests/test_jobs_backtest.py` conserva `test_backtest_runs_every_scope_and_writes_its_folder`,
`test_backtest_skips_and_isolates_scopes_and_fails_at_the_end`, `test_backtest_normalises_composition`,
`test_backtest_rejects_a_bad_composition_before_running` y añade
`test_backtest_stop_on_error_stops_at_the_first_failure` (`scopes=['es-ar', 'es']`, `stop_on_error=True` →
`runner.calls` sólo `es-ar`, `RuntimeError` match `'backtest: failed scopes: es-ar'`) y
`test_backtest_refuses_an_empty_scope_list` (`scopes=[]` → `ValueError` match `'backtest: no scopes to run'`).

- [ ] **Step 6: Reescribir `mtpy/jobs/Backtest.py`**

- [ ] **Step 7: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py tests/test_backtest_unit.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git status
git add mtpy/pipelines/backtest mtpy/jobs/Backtest.py tests/test_pipelines_backtest.py tests/test_jobs_backtest.py
git commit -m "backtest: pipeline backtest.Backtest (acción run) y el job como SequenceJob"
```

---

### Task 8: Grupo `publish` (1/2): `Forecast` con `finish`, `Parties`, `Manifest` y el job `Publish`

**Files:**
- Create: `mtpy/pipelines/publish/__init__.py`, `Forecast.py`, `Parties.py`, `Manifest.py`
- Modify: `mtpy/jobs/Publish.py` (`run` 103-337; se mueven `_forecast_runs` 343-419, `_manifest_entry` 421-455,
  `_forecast_scope` 457-541, `_remove_run` 543-564, `_write_manifest` 751-774, `_write_parties` 776-798,
  `_manifest_line` 800-804, `_line` 840-858; `STEPS` 9-11)
- Test: `tests/test_pipelines_publish.py` (nuevo), `tests/test_jobs_publish.py`

**Interfaces:**
- Produces: `publish.Forecast(ScopePipeline)`: `label = 'forecast'`, `steps = STEPS`,
  `RUN_EXISTS`/`RUN_INCOMPLETE` (los de `Publish`), `pipes = {'publish': {'Load': ['publish']}}`.
  `start(self, fs=None, dry_run=False, today=None, days=None, replace=False, event_date=None, force=False, n_sim=1000, seed=42, drange=6, max_fc=10, alpha=0.05, correctors=None, **kw)`:
  `self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)`,
  `self.run_id = bundle.run_id()`, `self.plan` (`Publish.run` 234-238), `self.params` (239-240 con
  `verbose=self.model_verbose`), `self.national` (226-230), `self.entries = {}`, cabecera (243-247 con
  `self.total`). `publish_task(chunk, position, **kw)` = `_forecast_runs` 392-419 (con `_forecast_scope`,
  `_remove_run`, `_manifest_entry` como métodos del pipeline, sin cambios de lógica), llamando
  `progress.done(self.line_of(chunk, result_del_día))` por día; deja `self.result` = el único resultado sin
  `days`, la lista con ellos; `self.entries[chunk] = entry` si no es `None`. `summary(chunk, **kw)` devuelve
  `self.line_of(...)` del resultado (la cadena, o la lista con `days`) sin volver a llamar a `progress.done`
  (ya se llamó por día). `line_of` = `Publish._line` 840-858 (con `day`). `finish(self, dry_run=False, **kw) -> Optional[str]`:
  con `dry_run`, `self.line = 'manifest: not written (dry run)'`; si `self.entries`, `_write_manifest` 751-774
  reescrito: `self.line = manifest_line(publish.update_manifest(self.writer, scopes=self.entries))` y, con
  excepción, `self._log_error('manifest')`, `self.line = 'manifest: failed ({Type}: {e})'` y `return 'manifest'`;
  sin entradas, nada. `manifest_line(data)` función de módulo en `Manifest.py` (= `_manifest_line`), importada.
- Produces: `publish.Parties(ScopePipeline)`: `chunked = False`, `label = 'parties'`,
  `pipes = {'publish': {'Load': ['write']}}`; `start(fs=None, dry_run=False, **kw)` crea el writer;
  `write_task` = `_write_parties` 790-798: `self.result = {'status': 'published', 'n_parties': n}` o
  `self.failed('parties', e)`; `line_of` → `'parties: {n partidos}'` / `'parties: failed (...)'`.
- Produces: `publish.Manifest(ScopePipeline)`: `chunked = False`, `label = 'manifest'`,
  `pipes = {'publish': {'Load': ['write']}}`; `write_task(dry_run=False, freeze=None, **kw)`: con `dry_run`,
  `self.result = {'status': 'skipped', 'reason': 'dry run'}` y línea `'manifest: not written (dry run)'`; si no,
  `publish.update_manifest(self.writer, freeze=freeze)` → `self.result = {'status': 'published', **data}` y
  línea `manifest_line(data)`; excepción → `self.failed('manifest', e)` y `'manifest: failed (...)'`.
- Produces: `mtpy.jobs.Publish.Publish(SequenceJob)`: `sequence = {'publish': ['forecast', 'analysis', 'event', 'backtest', 'study', 'parties', 'manifest', 'point', 'unpublish']}`,
  `default_action = 'publish'`, `stop_on_error = False`, `chunk_noun = 'scopes'`, `WHAT`; `run` con la firma de
  hoy más `stop_on_error=None`: las validaciones 193-212 tal cual (mismos mensajes y orden), la del sistema de
  ficheros 214-218 (`fs = fs if fs is not None else self.app.fs`), `select = ['publish.' + w for w in whats]` más
  `'publish.parties'` si `forecast` o `analysis` están en `whats`, `targets = publish.resolve_scopes(scopes)`
  (vacío → `ValueError('publish: no scopes to run')`), `super().run(select=select, chunks=targets, stop_on_error=..., fs=fs, dry_run=..., today=..., days=days, replace=replace, event_date=..., force=..., n_sim=..., seed=..., drange=..., max_fc=..., alpha=..., correctors=..., freeze=..., run=run, source=..., study=..., register=..., verbose=...)`;
  devuelve `{scope: result}` de las claves `('publish.<what>', scope)` con `what` en
  `('forecast', 'analysis', 'event', 'backtest', 'point', 'unpublish')` (el último gana) y `results[study]` de
  `('publish.study', None)`; `parties` y `manifest` no entran. Docstring de clase íntegro (+ `stop_on_error`).
  En esta tarea el job sólo tiene `forecast`, `parties` y `manifest` operativos; los demás pipelines llegan en
  la Tarea 9 (hasta entonces `select` con ellos falla al importar el módulo: los tests que los usan se
  ejecutan en la Tarea 9).

- [ ] **Step 1: Escribir los tests de los pipelines**

`tests/test_pipelines_publish.py`, reutilizando la fixture `patched`, `entry` y `fixed_prov` de
`tests/test_jobs_publish.py` (moverlas a `tests/publish_fixtures.py` e importarlas en los dos ficheros):

```python
"""Tests de los pipelines del grupo `publish` ejecutados sueltos, sin base."""
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.pipelines.publish.Forecast import Forecast
from mtpy.pipelines.publish.Manifest import Manifest
from mtpy.pipelines.publish.Parties import Parties
from tests.publish_fixtures import patched  # noqa: F401


def forecast(fs, scope, position=(1, 1), **kwargs):
    p = Forecast()
    p.run('publish', chunk=scope, position=position, fs=fs, verbose=0, **kwargs)
    return p


def test_forecast_publishes_a_scope_and_keeps_its_manifest_entry_for_finish(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es', position=(1, 3))
    assert p.result['status'] == 'published' and p.result['run_id'] == p.run_id and 'entry' in p.result
    assert p.line.startswith('es: published {} (as_of '.format(p.run_id)) and p.entries == {'es': p.result['entry']}
    assert not (tmp_path / 'site' / 'v1' / 'manifest.json').exists()
    assert p.finish(fs=fs) is None and p.line == 'manifest: 1 scopes, freeze off'
    assert publish.read_manifest(bundle.BundleReader(fs))['scopes']['es']['latest'] == p.run_id


def test_forecast_statuses_skipped_failed_refused(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    assert forecast(fs, 'es-md').result['status'] == 'skipped' and forecast(fs, 'es-md').line == 'es-md: skipped (no upcoming event)'
    assert forecast(fs, 'es-cb').line.startswith('es-cb: skipped (No polls')
    assert forecast(fs, 'es-ar').line == "es-ar: failed (KeyError: 'x')" and forecast(fs, 'es-ar').failure_name('es-ar') == 'es-ar (forecast)'
    p = forecast(fs, 'es', today='2026-11-26')
    assert p.result['status'] == 'refused' and 'LOREG' in p.result['reason'] and p.entries == {}
    p.line = None
    assert p.finish(fs=fs) is None and p.line is None  # sin entradas no hay línea de manifest


def test_forecast_dry_run_writes_apart_and_finish_says_so(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es', dry_run=True)
    assert (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es').exists() and p.finish(fs=fs, dry_run=True) is None
    assert p.line == 'manifest: not written (dry run)'


def test_forecast_finish_failure_is_its_own_failure(fresh_app, tmp_path, patched, monkeypatch):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es')
    monkeypatch.setattr(publish, 'update_manifest', lambda *a, **k: (_ for _ in ()).throw(OSError('s3 down')))
    assert p.finish(fs=fs) == 'manifest' and p.line == 'manifest: failed (OSError: s3 down)'


def test_forecast_backfill_gives_one_result_and_one_line_per_day(fresh_app, tmp_path, patched, capsys):
    fs = FileSystem(str(tmp_path))
    days = publish.backfill_days({'from': '2026-10-05', 'to': '2026-10-06'}, '2026-10-09')
    p = forecast(fs, 'es', days=days, today='2026-10-09')
    assert [r['status'] for r in p.result] == ['published', 'published'] and [r['day'] for r in p.result] == days
    assert len(p.line) == 2 and p.line[0].startswith('es: published 20261005-120000') and 'backfill 2026-10-05' in p.line[0]
    assert 'publish forecast: 1 ámbito · backfill 2026-10-05..2026-10-06 (2 días)' in capsys.readouterr().out


def test_parties_and_manifest_run_once(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = Parties()
    p.run('publish', chunk=None, position=(1, 1), fs=fs)
    assert p.result == {'status': 'published', 'n_parties': 5} and p.line == 'parties: 5 partidos' and p.chunked is False
    assert p.failure_name(None) == 'parties'
    m = Manifest()
    m.run('publish', chunk=None, position=(1, 1), fs=fs, freeze={'active': True, 'message': 'Veda'})
    assert m.result['status'] == 'published' and m.line == 'manifest: 0 scopes, freeze on'
    m.run('publish', chunk=None, position=(1, 1), fs=fs, dry_run=True)
    assert m.line == 'manifest: not written (dry run)'
```

Añadir, ejecutando el pipeline suelto, los escenarios de `tests/test_jobs_publish.py` que son de `Forecast`:
`test_forecast_passes_today_to_the_publication`, `test_loreg_window_refuses_es_unless_forced`,
`test_national_window_covers_the_election_day`, `test_forecast_rebuilds_the_history_of_each_published_scope`,
`test_a_value_error_other_than_nothing_to_publish_fails_the_scope`, `test_event_date_is_normalised_or_fails_the_scope`,
`test_backfill_respects_a_pointed_run`, `test_backfill_failed_day_is_reported_until_unpublished` (resultados y
líneas de cada día; el `unpublish` intermedio con `publish.unpublish` directo), `test_backfill_replace_*` (los
cuatro), `test_forecast_collects_the_model_warnings_into_the_result`, `test_forecast_keeps_the_warnings_of_a_skipped_or_failed_scope`,
`test_forecast_without_model_warnings_has_none`, `test_verbose_*` (las cuatro, sobre la salida del pipeline),
`test_forecast_prints_its_progress_live_before_the_summary` (hasta `[2/2] es-cb: skipped`), `test_backfill_progress_names_the_day`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pipelines_publish.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementar `Forecast.py`, `Parties.py`, `Manifest.py`**

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_publish.py -q`
Expected: PASS.

- [ ] **Step 5: Reescribir los tests del job (parte 1)**

`tests/test_jobs_publish.py` conserva, adaptados a `publish_fixtures`, los de extremo a extremo que cruzan
pipelines o leen el manifest tras el job: `test_publish_isolates_scopes_and_keeps_previous_manifest_entries`,
`test_dry_run_writes_apart_and_leaves_the_pointers_alone`, `test_unknown_what_is_rejected`,
`test_failed_forecast_is_not_hidden_by_a_later_successful_action`, `test_error_resolving_the_event_date_only_fails_that_scope`,
`test_manifest_write_failure_keeps_the_summary`, `test_backfill_publishes_one_run_per_day_and_never_moves_latest_back`,
`test_backfill_alone_points_latest_to_its_newest_day`, `test_backfill_range_errors_fail_before_touching_the_bundle`,
`test_backfill_needs_forecast_in_what`, `test_backfill_retry_repairs_the_history_and_the_manifest`,
`test_backfill_replace_must_be_a_boolean`, `test_publish_class_docstring_survives_and_mentions_the_levels`,
`test_parties_action_writes_the_catalogue_next_to_the_manifest`, `test_forecast_and_analysis_refresh_the_catalogue_but_the_pointers_do_not`
(la parte de `analysis` pasa a la Tarea 9), `test_parties_dry_run_writes_apart_and_a_failure_fails_the_job`, los
tres `test_progress_*` con logger y `test_logger_echoes_console_only_with_a_console_stream_handler`. En
`test_forecast_prints_its_progress_live_before_the_summary` la aserción del aviso pasa a
`any('es: published' in m for m in infos)` (el aviso lleva cabecera). Añadir:

```python
def test_publish_stop_on_error_stops_at_the_first_failed_scope_and_still_writes_the_manifest(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish
    fs = FileSystem(str(tmp_path))
    with pytest.raises(RuntimeError, match=r'publish: failed scopes: es-ar \(forecast\)'):
        Publish().run(what='forecast', scopes=['es', 'es-ar', 'es-cb'], fs=fs, stop_on_error=True)
    assert [c[0] for c in patched] == ['es', 'es-ar']
    assert publish.read_manifest(bundle.BundleReader(fs))['scopes']['es']['latest'] == patched[0][1]


def test_publish_refuses_an_empty_scope_list(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish
    with pytest.raises(ValueError, match='publish: no scopes to run'):
        Publish().run(what='forecast', scopes=[], fs=FileSystem(str(tmp_path)))


def test_publish_alert_has_the_header_and_the_failures_first(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish
    sent = []
    fresh_app.set('ps', types.SimpleNamespace(send=lambda msg, **kw: sent.append(msg)))
    with pytest.raises(RuntimeError):
        Publish().run(what='forecast', scopes='all', fs=FileSystem(str(tmp_path)))
    lines = sent[-1].splitlines()
    assert lines[0].startswith('Publish job executed in ') and lines[0].endswith(' with 1 failure')
    assert lines[1] == "es-ar: failed (KeyError: 'x')" and lines[2].startswith('es: published')
    assert 'manifest: 1 scopes, freeze off' in lines and 'parties: 5 partidos' in lines
```

- [ ] **Step 6: Reescribir `mtpy/jobs/Publish.py` como `SequenceJob`**

Según Interfaces; los métodos de `analysis`, `event`, `backtest`, `study`, `point`/`unpublish` del job de hoy se
borran aquí (sus pipelines llegan en la Tarea 9), así que los tests del job que los usan se marcan
`@pytest.mark.skip(reason='Tarea 9')` sólo hasta la tarea siguiente.

- [ ] **Step 7: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_publish_unit.py -q`
Expected: PASS (con los `skip` de la Tarea 9).

- [ ] **Step 8: Commit**

```bash
git status
git add mtpy/pipelines/publish mtpy/jobs/Publish.py tests/publish_fixtures.py tests/test_pipelines_publish.py tests/test_jobs_publish.py
git commit -m "publish: pipelines Forecast (manifest en finish), Parties y Manifest y el job como SequenceJob"
```

---

### Task 9: Grupo `publish` (2/2): `Analysis`, `Event`, `Backtest`, `Study`, `Point`, `Unpublish`

**Files:**
- Create: `mtpy/pipelines/publish/Analysis.py`, `Event.py`, `Backtest.py`, `Study.py`, `Point.py`,
  `Unpublish.py`
- Modify: `mtpy/jobs/Publish.py` (quitar los `skip`; los cuerpos salen de la versión previa a la Tarea 8:
  `_analysis_scope` 566-606, `_event_scope` 608-656, `_backtest_scope` 658-686, `_study_scope` 688-718,
  `_pointer_scope` 720-749, `_analysis_line` 806-812, `_event_line` 814-820, `_backtest_line` 822-828,
  `_study_line` 830-838; `git show HEAD~1:mtpy/jobs/Publish.py` los enseña)
- Test: `tests/test_pipelines_publish.py`, `tests/test_jobs_publish.py`

**Interfaces:**
- Produces: `publish.Analysis(ScopePipeline)`: `label = 'analysis'`, `pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}`;
  `start(fs, dry_run, today, **kw)` crea el writer y `self.national`; `prepare_task` arranca el progreso;
  `publish_task(chunk, today=None, force=False, **kw)` = `_analysis_scope`; `line_of` = `_analysis_line`.
- Produces: `publish.Event(ScopePipeline)`: `label = 'event'`, `pipes = {'publish': {'Setup': ['resolve_dates'], 'Load': ['publish']}}`;
  `resolve_dates_task(chunk, position, event_date=None, today=None, **kw)` = 635-641 (`self.days`; los dos
  casos de un solo resultado terminan con `self.end([...])`); `publish_task(chunk, source=None, **kw)` = 643-656
  con `progress.done(self.line_of(chunk, result))` por fecha; `self.result` = lista; `summary` devuelve la
  lista de líneas sin volver a `done`; `line_of` = `_event_line`. (`source` sigue siendo una carpeta local en esta
  fase; la fase 2 lo cambia.)
- Produces: `publish.Backtest(ScopePipeline)`: `label = 'backtest'`, Setup `prepare`, Load `publish` =
  `_backtest_scope`; `line_of` = `_backtest_line`.
- Produces: `publish.Study(ScopePipeline)`: `chunked = False`, `label = 'study'`,
  `pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}`; `start(..., study=None, **kw)` guarda
  `self.study`; `prepare_task` arranca `self.progress_of(self.study, (1, 1))`; `publish_task(register=False, force=False, **kw)`
  = `_study_scope`; `failure_name(chunk)` → `'{study} (study)'`; `line_of(chunk, result)` = `_study_line(self.study, result)`.
- Produces: `publish.Point(ScopePipeline)`: `label = 'point'`, `pipes = {'publish': {'Load': ['move']}}` (sin
  progreso, como hoy); `move_task(chunk, run=None, **kw)` = rama `point` de `_pointer_scope`;
  `line_of(chunk, result, run=None, **kw)` = `_line(scope, result, run)`. `publish.Unpublish(Point)`:
  `label = 'unpublish'`, rama `unpublish`.

- [ ] **Step 1: Escribir los tests de los pipelines**

Añadir a `tests/test_pipelines_publish.py`, ejecutando cada pipeline suelto con `patched` y `fixed_prov`:
`test_analysis_statuses_and_lines` (de `test_analysis_action_publishes_each_scope`,
`test_analysis_of_a_scope_without_polls_in_its_current_event_is_published_without_effects`,
`test_analysis_of_a_scope_without_upcoming_event_is_published_unless_in_the_national_window`,
`test_national_window_refuses_regional_forecast_and_analysis_but_not_events` parte `analysis`),
`test_event_publishes_each_date_and_fails_a_date_without_stopping_the_others` (de `test_event_action_publishes_each_date`,
`test_event_action_fails_a_date_without_stopping_the_others`, `test_event_all_and_invalid_list_through_the_job`:
`p.result` lista y `p.line` lista), `test_event_without_dates_ends_in_setup`, `test_event_dry_run_writes_apart`,
`test_backtest_publishes_and_skips` (de `test_backtest_action_publishes_each_scope`, con `source=str(tmp_path / 'bt')`),
`test_backtest_dry_run_and_failure`, `test_study_statuses_and_failure_name` (de `test_publish_study_runs_the_study_action`:
`Study().run('publish', chunk=None, position=(1, 1), fs=fs, study='voto-cera', register=True)` →
`p.failure_name(None) == 'voto-cera (study)'`), `test_point_and_unpublish_lines` (de `test_manifest_point_and_unpublish_actions`:
`Point().run('publish', chunk='es', fs=fs, run=run_id)` → `p.line == 'es: published {run_id}'`; un run inexistente →
`p.result['status'] == 'failed'`; `Unpublish` → `p.result == {'status': 'published', 'removed': run_id, 'latest': None}`).

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pipelines_publish.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar los seis pipelines**

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_publish.py -q`
Expected: PASS.

- [ ] **Step 5: Reactivar y adaptar los tests del job**

Quitar los `skip` de la Tarea 8. Conservar en `tests/test_jobs_publish.py`: `test_manifest_point_and_unpublish_actions`,
`test_analysis_isolates_scopes_and_dry_run_writes_apart`, `test_analysis_prints_its_scope_lines_live_at_verbose_1`,
`test_backtest_dry_run_writes_apart_and_a_failure_fails_the_job` (el `RuntimeError` match `r'es \(backtest\)'`),
`test_event_needs_its_date`, `test_event_requires_a_date_and_excludes_forecast`, `test_publish_study_runs_the_study_action`
(el `ValueError` y los `RuntimeError`), `test_national_window_refuses_regional_forecast_and_analysis_but_not_events`
y la parte `analysis` de `test_forecast_and_analysis_refresh_the_catalogue_but_the_pointers_do_not`. Añadir:

```python
def test_publish_results_are_keyed_by_scope_and_study(fresh_app, tmp_path, patched, monkeypatch):
    from mtpy.jobs.Publish import Publish
    monkeypatch.setattr(publish, 'publish_study', lambda study, writer, **kw: {'study': study, 'registered_at': None, 'checked': False, 'seconds': {'total': 1.0}})
    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis', 'study'], scopes=['es'], study='voto-cera', fs=fs)
    assert set(result) == {'es', 'voto-cera'} and result['es']['status'] == 'published'
```

- [ ] **Step 6: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_publish_unit.py tests/test_publish_studies.py -q`
Expected: PASS, sin `skip`.

- [ ] **Step 7: Commit**

```bash
git status
git add mtpy/pipelines/publish mtpy/jobs/Publish.py tests/test_pipelines_publish.py tests/test_jobs_publish.py
git commit -m "publish: pipelines Analysis, Event, Backtest, Study, Point y Unpublish"
```

---

### Task 10: Documentación, suite completa y cierre de la fase

**Files:**
- Modify: `deploy/README.md` (tablas «Parámetros» de «Cargar los sondeos» 590-600, «Calcular las series y los
  ratings» 675-690, «Backtest del modelo» 754-775 y «Publicar» 910-932; párrafos «Estados y resumen» 612-634,
  693-713, 783-803 y 943-985), `README.md` («Jobs» 16-30)
- Modify: `mtpy/jobs/progress.py` (docstring de módulo: ahora lo usan los pipelines)
- Test: `tests/test_deploy_files.py` (sin cambios), suite completa

- [ ] **Step 1: Documentar en `deploy/README.md`**

En cada tabla de parámetros una fila `| stop_on_error | false | Parar al primer ámbito (o acción) que falla en vez
de seguir con todos e informar al final; el manifest de los ámbitos ya publicados se escribe igualmente |`. En
cada «Estados y resumen», sustituir la frase «El resumen va también al logger y al servicio de avisos…» por un
párrafo común: el aviso lleva una cabecera (`<Job> job executed in <duración> :-)` o `… with n failures`), después
las líneas de los ámbitos fallidos y luego las demás; Pushover recibe como mucho 1.024 caracteres (la última
línea dice que está recortado y remite al log, que lleva el texto entero); un fallo de Pushover no interrumpe el
job; el proceso termina con salida 1 si algún ámbito falló, si el job reventó antes de empezar o si lo pararon
(`SIGTERM`: línea `interrumpido en …` y aviso). Añadir al final de la sección «Publicar» un apartado «Un pipeline
suelto»: `python job.py pipeline '{"select":"publish.backtest","chunks":"es,es-md"}'` y
`python pipeline.py publish.backtest publish es es-md`, con la lista de los quince pipelines y sus grupos (el
lanzador suelto no valida lo que cruza pipelines ni resuelve `"all"`: los ámbitos van nombrados).

- [ ] **Step 2: Documentar en `README.md`**

En «Jobs», una frase: cada acción es un pipeline de `mtpy/pipelines/<job>/` y los jobs son `SequenceJob` de
MTpy; `stop_on_error` y el aviso final.

- [ ] **Step 3: Suite completa**

Run: `python -m pytest -q`
Expected: todos en verde (≥ 1.557 más los nuevos); sin `skip` nuevos.

- [ ] **Step 4: Comprobación manual sin base**

Run: `set -a; . deploy/elections.env; set +a; S3_BUCKET= python job.py publish '{"what":["backtest"],"scopes":["es"],"source":"backtest/results"}'; echo "exit $?"`
(`resolve_scopes` consulta el catálogo de la RDS; el paquete se escribe en `files/site/v1`)
Expected: cabecera, `[1/1] es: empezando`, `es: backtest published (30 cases, 5 elections) in … s`, aviso
`Publish job executed in … :-)` en el log (`log/info.log`), `exit 0`.

Run: lo mismo con `"source":"/tmp/empty"`
Expected: `es: backtest skipped (no backtest results for es)` y `exit 0` (`skipped` no es fallo).

- [ ] **Step 5: Commit y cierre**

```bash
git status
git add deploy/README.md README.md mtpy/jobs/progress.py
git commit -m "docs: stop_on_error, aviso final y pipelines sueltos en los runbooks de los jobs"
```

Anotar en la spec (`docs/superpowers/specs/2026-10-31-jobs-pipelines-ecs-design.md`, línea de estado) «Fase 1
construida el <fecha> (<primer commit>..<último>, N tests)».
