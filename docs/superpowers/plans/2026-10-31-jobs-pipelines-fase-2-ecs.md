# Tarea de jobs en ECS y backtest en S3 (fase 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lanzar cualquier job desde el portátil como tarea puntual de ECS Fargate con un comando
(`python deploy/ecs/job.py <job> '<json>' [--follow]`, Python con boto3), con los logs una sola vez en CloudWatch, el aviso de Pushover al
final y el código de salida del job; y que el backtest corra en ECS con sus tablas en S3, de donde las leen
`publish backtest` y `publish event` sin reconstruir la imagen.

**Architecture:** Las tablas del backtest pasan a leerse y escribirse por la API de `FileSystem` (local o `S3`)
bajo `backtest/results/` del sistema de ficheros de la app; `out`/`source` siguen aceptando una carpeta local. El
pipeline `backtest.Backtest` gana `push` y `pull` para sembrar S3 y bajar la copia que se commitea. La definición
de tarea de jobs se deriva de la viva del servicio web (`deploy/ecs/job.py --register`) y el `run-task` toma la red del
despliegue actual; `init.sh` hace `exec` del job y rechaza un contenedor sin nada que hacer; `GIT_COMMIT` se
hornea en la imagen.

**Tech Stack:** Python 3.11, pandas, `mtpy.core.io.FileSystem` / `mtpy.core.services.s3.S3` (s3fs), bash (`init.sh`),
boto3 (lanzador `deploy/ecs/job.py`, con clientes falsos en los tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-31-jobs-pipelines-ecs-design.md` (secciones «Diseño de la fase 2»:
«Infraestructura real», «Qué corre en ECS», «Backtest en S3», puntos 1-5).

## Global Constraints

- Depende de la fase 1 (plan `2026-10-31-jobs-pipelines-fase-1.md`): el pipeline `backtest.Backtest`, el job
  `Backtest` como `SequenceJob` y la salida distinta de cero de `job.py`.
- Infraestructura real (leída el 2026-10-31): clúster `elections-model-cluster`, servicio `elections-model-service`,
  contenedor `Main`, configuración por `environmentFiles` desde `s3://elections-s3-bucket/elections.env` (sin
  `environment` ni `secrets` en la definición), roles `ecsTaskExecutionRole`, logs `awslogs` en
  `/ecs/elections-model`, subredes públicas con `assignPublicIp: ENABLED`. No se crea infraestructura ni se
  tocan secretos: el lanzador sólo registra la definición de jobs y lanza tareas.
- `CWL_GROUP` sigue vacío en el fichero de entorno (el driver `awslogs` ya captura stdout); `APP_VERBOSE=1` hace
  que el logger escriba por consola, así que cada línea sale una vez.
- `load params` no corre en ECS: el lanzador lo rechaza; el código de los jobs no sabe nada de ECS.
- `backtest/results` sigue en git (menos `series.csv`); `pull` es un paso del flujo tras cada backtest en ECS.
- `tests/test_deploy_files.py` vigila `deploy/ecs/`, `init.sh` y `supervisord.conf`: cambia con ellos.
- `deploy/ecs.env` queda ignorado por git y por `.dockerignore` (ya lo cubre `*.env`); se commitea
  `deploy/ecs.env.example`.
- Identificadores en inglés; textos de consola en castellano o como los de hoy; README en castellano.
- Luis hace commits en paralelo: `git status` antes de cada commit y añadir sólo los ficheros de la tarea.

## Pendientes heredados de la revisión final de la fase 1

- `job.py`: un nombre de job desconocido o un JSON mal formado revientan antes de construir el job, con salida 1 pero
  sin aviso de Pushover (hoy el aviso sale de `Job.dispatch`). En ECS sin `--follow` el aviso es la única señal:
  mover `getattr`/`json.loads` dentro del `try` de `main` y avisar por `app.ps` cuando no hay instancia de job.
  Entra en la Tarea 3 (junto a `init.sh`), con un test en `tests/test_launchers.py`.
- `backtest '{"action":"pull"}'` hoy ignora `action` por `**kwargs` y corre un backtest entero: la validación de
  `action` de la Tarea 2 es obligatoria antes de documentar `push`/`pull`.

## Review Focus

- `push`/`pull` con un `series.csv` que falta en el origen (clon fresco, o S3 sin serie): el destino no debe
  conservar una serie de otro backtest; de los nueve ficheros, se copian los que existen y se borran en el
  destino los que no existen en el origen. Test en la Tarea 2.
- `publish backtest`/`publish event` en ECS para un ámbito sin tablas en S3: `skipped`
  (`no backtest results for …`) y el evento sin lámina del modelo, nunca una excepción de s3fs. Test en la Tarea 1.
- El valor de `RUN_JOB` con espacios y comillas (`"message":"Veda electoral"`): llega intacto al override del
  `run-task` y de ahí a `init.sh`. Test en la Tarea 4 (`test_launch_builds_the_run_task_from_the_service_network`).
- `run-task` que no arranca (sin capacidad, subred mal): la respuesta trae `failures` y ninguna tarea; el lanzador
  imprime la razón y termina con 1 en vez de esperar un log que no llegará. Test en la Tarea 4.
- `init.sh` con `exec`: el código de salida del job es el del contenedor (3 → 3), y sin `RUN_JOB` ni servicios el
  contenedor termina con 1 en vez de facturar en `tail -f`. Test en la Tarea 3.

---

### Task 1: Las tablas del backtest por `FileSystem`: `results_fs`, escritor y lectores

**Files:**
- Modify: `mtpy/lib/backtest.py` (`RESULTS_ROOT` 33, `results_dir` 615-633, `backtest_scope` 659-750)
- Modify: `mtpy/lib/publish/backtest.py` (`backtest_source` 28-45, `export_backtest_meta` 86-121,
  `publish_backtest` 124-180), `mtpy/lib/publish/events.py` (`export_event_model` 169-203,
  `export_event_series` 205-248, `publish_event` 458-471)
- Test: `tests/test_backtest_unit.py` (195-241, 286-312), `tests/test_publish_unit.py` (969-1002, 1070-1080,
  1204-1213), `tests/fakes.py` (`write_backtest_source` 284-295, sin cambios de firma)

**Interfaces:**
- Produces, en `mtpy/lib/backtest.py`:
  - `RESULTS_PREFIX = 'backtest/results'` (carpeta en el sistema de ficheros de la app) y
    `RESULT_FILES = ('shares.csv', 'seats.csv', 'blocks.csv', 'provinces.csv', 'series.csv', 'meta.csv', 'metrics.csv', 'by_horizon.csv', 'run.json')`.
  - `results_fs(root: Optional[str], scope: str) -> tuple[FileSystem, str]`: con `root`,
    `(FileSystem(os.path.abspath(root)), '' if scope == 'es' else scope)`; sin él,
    `(App.get_().fs, RESULTS_PREFIX if scope == 'es' else RESULTS_PREFIX + '/' + scope)`; sin `app.fs` →
    `RuntimeError('backtest: no file system configured (set S3_BUCKET or run with a local files/ root)')`.
  - `results_path(folder: str, name: str) -> str`: `name` si `folder == ''`, si no `folder + '/' + name`.
  - `describe_source(fs: FileSystem, folder: str) -> str`: `'s3://{fs.bucket}/{folder}'` si `fs` es `S3`; si
    no, la ruta absoluta (`fs.path` con `folder == ''`, `fs.get_path(folder)` si no), relativa a la raíz del
    repositorio cuando está dentro (la regla de `Backtest._shown` de hoy).
  - `backtest_scope(scope, out=None, ...)`: misma firma; `fs, folder = results_fs(out, scope)`; cada tabla con
    `fs.write_csv(frame, results_path(folder, key + '.csv'))`, `run.json` con
    `fs.write_bytes(json.dumps(record, indent=1).encode('utf-8'), results_path(folder, 'run.json'))`;
    devuelve `'folder': describe_source(fs, folder)`. `results_dir` y `RESULTS_ROOT` se quedan (los usa
    `backtest/run_backtest.py` por su `--out` por defecto).
- Produces, en `mtpy/lib/publish/backtest.py`: `backtest_source(scope: str, root: Optional[str] = None) -> tuple[FileSystem, str]`
  = `backtest.results_fs(root, scope)` (import perezoso como hoy); `publish_backtest` lee con
  `fs.exists`/`fs.read_csv`/`fs.read_bytes` (el `run.json` con `json.loads(...decode('utf-8'))`) y pasa
  `describe_source(fs, folder)` a `export_backtest_meta`, que toma `source` tal cual (ya no aplica la regla
  del repositorio).
- Produces, en `mtpy/lib/publish/events.py`: `export_event_model(source, event_date)` y
  `export_event_series(source, event_date)` aceptan `source` como `None`, una carpeta local (`str`, envuelta en
  `(FileSystem(os.path.abspath(source)), '')`) o `(fs, folder)`; leen con `fs.exists` y `fs.read_csv`.
  `publish_event` sigue pasando `backtest_source(scope, source)`.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_backtest_unit.py`, sustituir `test_results_dir_keeps_es_at_the_root` por:

```python
def test_results_fs_keeps_es_at_the_root_locally_and_under_the_prefix_in_the_app_fs(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    from mtpy.lib import backtest, publish
    fs, folder = backtest.results_fs(str(tmp_path), 'es')
    assert isinstance(fs, FileSystem) and fs.path == str(tmp_path) and folder == ''
    assert backtest.results_fs(str(tmp_path), 'es-md')[1] == 'es-md'
    assert backtest.results_path('', 'metrics.csv') == 'metrics.csv' and backtest.results_path('es-md', 'run.json') == 'es-md/run.json'
    with pytest.raises(RuntimeError, match='no file system configured'):
        backtest.results_fs(None, 'es')
    fresh_app.set('fs', FileSystem(str(tmp_path / 'files')))
    fs, folder = backtest.results_fs(None, 'es-md')
    assert fs is fresh_app.fs and folder == 'backtest/results/es-md' and backtest.results_fs(None, 'es')[1] == 'backtest/results'
    assert publish.backtest_source('es-md', 'x')[1] == 'es-md' and publish.backtest_source('es')[1] == 'backtest/results'
    assert backtest.describe_source(fs, folder) == str(tmp_path / 'files' / 'backtest' / 'results' / 'es-md')
    assert backtest.describe_source(FileSystem(backtest.RESULTS_ROOT), '') == 'backtest/results'


def test_describe_source_names_the_bucket_of_an_s3_file_system():
    from mtpy.core.services.s3 import S3
    from mtpy.lib import backtest
    fs = S3.__new__(S3)
    fs.bucket = 'elections-s3-bucket'
    assert backtest.describe_source(fs, 'backtest/results/es-md') == 's3://elections-s3-bucket/backtest/results/es-md'


def test_backtest_scope_writes_into_the_app_fs_without_out(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    from mtpy.lib import backtest
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    figures = backtest.backtest_scope('es-md', events=['2023-05-28'], runner=lambda **kwargs: fake_results(),
                                      stats=lambda scope: {'db_polls': 1, 'db_last_poll': None},
                                      prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    folder = tmp_path / 'backtest' / 'results' / 'es-md'
    assert (folder / 'metrics.csv').exists() and (folder / 'run.json').exists() and figures['folder'] == str(folder)
```

Los `test_backtest_scope_*` con `out=str(tmp_path)` no cambian. En `tests/test_publish_unit.py`:
`test_backtest_source_follows_the_backtest_script` se borra (lo cubre el de arriba);
`test_publish_backtest_converts_the_repository_results` pasa `source=root`; añadir:

```python
def test_publish_backtest_and_event_read_the_app_fs_without_source(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    fresh_app.set('fs', FileSystem(str(tmp_path / 'files')))
    write_backtest_source(str(tmp_path / 'files' / 'backtest' / 'results'))
    writer = make_writer(tmp_path)
    out = publish.publish_backtest('es', writer, prov=fake_prov)
    assert out['n_cases'] == 4 and writer.read_json(bundle.path_backtest('es', 'meta'))['data']['source'] == str(tmp_path / 'files' / 'backtest' / 'results')
    with pytest.raises(publish.NothingToPublish, match='no backtest results for es-cb'):
        publish.publish_backtest('es-cb', writer, prov=fake_prov)
    ev = synthetic_event()
    result = publish.publish_event('es-cb', '2023-07-23', writer, computer=FakeComputer(), prov=fake_prov, **event_readers(ev))
    assert result['has_model'] is False and result['has_series'] is False


def test_export_event_model_accepts_a_file_system_and_a_folder(tmp_path):
    from mtpy.core.io import FileSystem
    folder = write_backtest_source(str(tmp_path / 'bt'))
    by_path, _ = publish.export_event_model(folder, '2023-07-23')
    by_fs, _ = publish.export_event_model((FileSystem(str(tmp_path / 'bt')), ''), '2023-07-23')
    assert by_path == by_fs and by_fs['horizons'] == [6, 30]
    assert publish.export_event_series((FileSystem(str(tmp_path / 'bt')), ''), '2023-07-23') == (None, None)
```

(`event_readers` y `synthetic_event` ya existen en ese fichero.)

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_backtest_unit.py tests/test_publish_unit.py -q -k "results_fs or describe_source or app_fs or accepts_a_file_system"`
Expected: FAIL (`AttributeError: results_fs`).

- [ ] **Step 3: Implementar escritor y lectores**

Según Interfaces. `App` se importa en `backtest.py` desde `..core.app` (perezoso dentro de `results_fs`, para
no arrastrar el arranque).

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_backtest_unit.py tests/test_publish_unit.py tests/test_pipelines_backtest.py tests/test_pipelines_publish.py tests/test_jobs_backtest.py tests/test_jobs_publish.py -q`
Expected: PASS (los tests de los jobs y pipelines siguen pasando `out`/`source` locales).

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/lib/backtest.py mtpy/lib/publish/backtest.py mtpy/lib/publish/events.py tests/test_backtest_unit.py tests/test_publish_unit.py
git commit -m "backtest: tablas por FileSystem (app.fs + backtest/results por defecto, carpeta local con out/source)"
```

---

### Task 2: Acciones `push` y `pull` del pipeline `backtest.Backtest` y `action` en el job

**Files:**
- Modify: `mtpy/lib/backtest.py` (nueva `copy_results`), `mtpy/pipelines/backtest/Backtest.py`,
  `mtpy/jobs/Backtest.py`
- Test: `tests/test_pipelines_backtest.py`, `tests/test_jobs_backtest.py`

**Interfaces:**
- Produces: `backtest.copy_results(scope: str, src: tuple[FileSystem, str], dst: tuple[FileSystem, str]) -> dict`:
  si el origen no tiene `metrics.csv` → `NothingToEvaluate('no backtest results for {scope} in {describe_source(*src)}')`;
  por cada nombre de `RESULT_FILES`: si existe en el origen, `dst_fs.write_bytes(src_fs.read_bytes(...), ...)`;
  si no existe en el origen pero sí en el destino, `dst_fs.remove(...)`; devuelve
  `{'copied': [nombres], 'removed': [nombres]}`.
- Produces: `backtest.Backtest` (pipeline): `pipes` gana `'push': {'Setup': ['prepare'], 'Load': ['push']}` y
  `'pull': {'Setup': ['prepare'], 'Load': ['pull']}`. `start` imprime, según `self._action`:
  `run` → la cabecera de hoy con `describe_source(*results_fs(out, 'es'))` en vez de `shown(out)`;
  `push` → `'backtest push: {n ámbitos} · {local} → {remote}'`; `pull` → `'backtest pull: {n ámbitos} · {remote} → {local}'`,
  con `local = describe_source(*results_fs(out or RESULTS_ROOT, 'es'))` y `remote = describe_source(*results_fs(None, 'es'))`.
  `push_task(chunk, out=None, **kw)`: `copy_results(chunk, results_fs(out or RESULTS_ROOT, chunk), results_fs(None, chunk))`
  → `self.result = {'status': 'pushed', 'n_files': len(copied), 'n_removed': len(removed), 'folder': describe_source(*destino), 'warnings': []}`;
  `NothingToEvaluate` → `skipped` con la razón; otra excepción → `self.failed(chunk, e, warnings=[])`.
  `pull_task`: lo mismo con origen y destino cambiados y `status` `'pulled'`. `line_of` para esos estados:
  `'{scope}: backtest {pushed|pulled} ({n} files{, m removed}) to {folder}'`; `skipped`/`failed` como hoy.
- Produces: `mtpy.jobs.Backtest.Backtest.run(..., action='run', ...)`: `action` fuera de `('run', 'push', 'pull')`
  → `ValueError('backtest: unknown action "{action}" (expected run, push or pull)')`; se pasa a `super().run(action=action, ...)`.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_pipelines_backtest.py`:

```python
def seeded(tmp_path, scope='es-md'):
    """Una carpeta local con los nueve ficheros de `scope` (por `backtest_scope` con el runner falso)."""
    from mtpy.lib import backtest
    from tests.test_backtest_unit import fake_results
    local = tmp_path / 'local'
    backtest.backtest_scope(scope, out=str(local), events=['2023-05-28'], runner=lambda **kwargs: fake_results(),
                            stats=stats, prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    return local


def test_push_copies_the_nine_files_and_removes_the_stale_ones(fresh_app, tmp_path, capsys):
    from mtpy.core.io import FileSystem
    remote = tmp_path / 'remote'
    fresh_app.set('fs', FileSystem(str(remote)))
    local = seeded(tmp_path)
    (local / 'es-md' / 'series.csv').unlink()
    stale = remote / 'backtest' / 'results' / 'es-md'
    stale.mkdir(parents=True)
    (stale / 'series.csv').write_text('old')
    p = Backtest()
    p.run('push', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    assert p.result['status'] == 'pushed' and p.result['n_files'] == 8 and p.result['n_removed'] == 1
    assert p.line == 'es-md: backtest pushed (8 files, 1 removed) to {}'.format(stale)
    assert sorted(f.name for f in stale.iterdir()) == ['blocks.csv', 'by_horizon.csv', 'meta.csv', 'metrics.csv', 'provinces.csv', 'run.json', 'seats.csv', 'shares.csv']
    assert 'backtest push: 1 ámbito · {} → {}'.format(local, remote / 'backtest' / 'results') in capsys.readouterr().out


def test_pull_brings_the_files_back_to_the_local_copy(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    remote = tmp_path / 'remote'
    fresh_app.set('fs', FileSystem(str(remote)))
    local = seeded(tmp_path)
    Backtest().run('push', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    (local / 'es-md' / 'metrics.csv').write_text('changed')
    p = Backtest()
    p.run('pull', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    assert p.result['status'] == 'pulled' and p.result['n_files'] == 9 and p.result['n_removed'] == 0
    assert p.line == 'es-md: backtest pulled (9 files) to {}'.format(local / 'es-md')
    assert (local / 'es-md' / 'metrics.csv').read_text() != 'changed'


def test_push_without_local_results_is_skipped_and_touches_nothing(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    fresh_app.set('fs', FileSystem(str(tmp_path / 'remote')))
    p = Backtest()
    p.run('push', chunk='es-xx', position=(1, 1), out=str(tmp_path / 'local'), verbose=0)
    assert p.result['status'] == 'skipped' and p.line == 'es-xx: skipped (no backtest results for es-xx in {})'.format(tmp_path / 'local' / 'es-xx')
    assert not (tmp_path / 'remote').exists()
```

En `tests/test_jobs_backtest.py`:

```python
def test_backtest_action_must_be_run_push_or_pull(fresh_app, patched, tmp_path):
    with pytest.raises(ValueError, match='backtest: unknown action "sync"'):
        Backtest().run(scopes=['es'], action='sync', out=str(tmp_path))


def test_backtest_pull_through_the_job(fresh_app, patched, tmp_path):
    from mtpy.core.io import FileSystem
    from tests.test_pipelines_backtest import seeded
    fresh_app.set('fs', FileSystem(str(tmp_path / 'remote')))
    local = seeded(tmp_path, 'es')
    Backtest().run(scopes=['es'], action='push', out=str(local), verbose=0)
    results = Backtest().run(scopes=['es'], action='pull', out=str(tmp_path / 'copy'), verbose=0)
    assert results['es']['status'] == 'pulled' and (tmp_path / 'copy' / 'run.json').exists()
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementar `copy_results`, las acciones y `action` en el job**

Actualizar los docstrings del pipeline y del job (ejemplos `python job.py backtest '{"action":"push","scopes":"all"}'`
y `'{"action":"pull","scopes":"all"}'`).

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pipelines_backtest.py tests/test_jobs_backtest.py tests/test_backtest_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add mtpy/lib/backtest.py mtpy/pipelines/backtest/Backtest.py mtpy/jobs/Backtest.py tests/test_pipelines_backtest.py tests/test_jobs_backtest.py
git commit -m "backtest: acciones push y pull entre la copia del repositorio y el sistema de ficheros de la app"
```

---

### Task 3: `Dockerfile` con `GIT_COMMIT`, `init.sh` con `exec` y salida 1, plantillas de ECS como referencia

**Files:**
- Modify: `Dockerfile` (tras `WORKDIR ${APP_PATH}`, 53), `deploy/docker/init.sh`,
  `deploy/ecs/task-job.example.json`, `deploy/ecs/task-web.example.json`, `deploy/ecs/run-task.example.json`
- Test: `tests/test_deploy_files.py` (22-59)

**Interfaces:**
- `Dockerfile`: `ARG GIT_COMMIT=` y `ENV GIT_COMMIT=${GIT_COMMIT}` justo después de `WORKDIR` (antes de `COPY . .`
  no hace falta: es una variable de entorno de la imagen).
- `deploy/docker/init.sh`: la rama `RUN_JOB` termina en
  `exec python /app/job.py "$job_name" ${job_params:+"$job_params"}`; en la otra rama, si `SERVICES` está vacío,
  `echo "Nothing to run: set RUN_JOB, APP_API, APP_CRONTAB or APP_QUEUE" >&2` y `exit 1`; con servicios, como
  hoy (`supervisord`, `tail -f /dev/null`).
- `task-job.example.json` = lo que registra `deploy/ecs/job.py --register` (Tarea 4): `family` `elections-model-job`,
  `networkMode` `awsvpc`, `requiresCompatibilities` `["FARGATE"]`, `cpu` `"2048"`, `memory` `"4096"`,
  `runtimePlatform` X86_64/LINUX, `taskRoleArn` `<task-role>`, `executionRoleArn` `<execution-role>`, un
  contenedor `Main` con `image` `<account>.dkr.ecr.<region>.amazonaws.com/elections-model:latest`,
  `essential` true, `environment` `[{"name": "RUN_JOB", "value": ""}]`,
  `environmentFiles` `[{"value": "arn:aws:s3:::<bucket>/elections.env", "type": "s3"}]`, sin `secrets` ni
  `portMappings`, `logConfiguration` awslogs (`awslogs-group` `/ecs/elections-model`, `awslogs-region`
  `<region>`, `awslogs-stream-prefix` `job`).
- `task-web.example.json` = la forma de la definición viva: `family` `elections-model-cluster-elections-model-service`,
  `cpu` `"1024"`, `memory` `"2048"`, contenedor `Main`, `environmentFiles` igual, sin `environment` ni
  `secrets`, `portMappings` 8042, prefijo de logs `ecs`.
- `run-task.example.json` = lo que construye el lanzador: `cluster` `elections-model-cluster`,
  `taskDefinition` `elections-model-job`, `launchType` `FARGATE`, `count` 1,
  `networkConfiguration.awsvpcConfiguration` con `subnets` `["<subnet-id>", "<subnet-id>", "<subnet-id>"]`,
  `securityGroups` `["<security-group-id>"]`, `assignPublicIp` `ENABLED`;
  `overrides.containerOverrides[0]` con `name` `Main` y el `RUN_JOB` de la llamada diaria de `es`
  (`publish {"what":["forecast","analysis"],"scopes":["es"],"n_sim":10000}`).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_deploy_files.py`, reescribir `test_ecs_templates_are_valid_and_consistent`:

```python
def test_ecs_templates_describe_the_live_infrastructure():
    """Las plantillas son JSON puro y reflejan la infraestructura real: configuración por fichero de entorno en S3,
    contenedor `Main`, definición de jobs derivada de la web (2 vCPU, 4 GB, sin puertos, prefijo `job`)."""
    web = load('deploy/ecs/task-web.example.json'); job = load('deploy/ecs/task-job.example.json'); run = load('deploy/ecs/run-task.example.json')
    for template in (web, job):
        container = template['containerDefinitions'][0]
        assert container['name'] == 'Main' and 'secrets' not in container
        assert container['environmentFiles'] == [{'value': 'arn:aws:s3:::<bucket>/elections.env', 'type': 's3'}]
        assert container['image'].endswith('.amazonaws.com/elections-model:latest')
        assert container['logConfiguration']['options']['awslogs-group'] == '/ecs/elections-model'
    assert 'environment' not in web['containerDefinitions'][0] and web['containerDefinitions'][0]['portMappings'][0]['containerPort'] == 8042
    assert web['family'] == 'elections-model-cluster-elections-model-service' and web['cpu'] == '1024'
    assert job['family'] == 'elections-model-job' and job['cpu'] == '2048' and job['memory'] == '4096'
    assert 'portMappings' not in job['containerDefinitions'][0]
    assert job['containerDefinitions'][0]['environment'] == [{'name': 'RUN_JOB', 'value': ''}]
    assert job['containerDefinitions'][0]['logConfiguration']['options']['awslogs-stream-prefix'] == 'job'
    assert run['cluster'] == 'elections-model-cluster' and run['taskDefinition'] == 'elections-model-job'
    assert run['networkConfiguration']['awsvpcConfiguration']['assignPublicIp'] == 'ENABLED'
    override = run['overrides']['containerOverrides'][0]
    assert override['name'] == 'Main' and override['environment'][0]['name'] == 'RUN_JOB'
    assert override['environment'][0]['value'].startswith('publish {"what":["forecast","analysis"]')


def test_dockerfile_bakes_the_commit():
    dockerfile = read('Dockerfile')
    assert 'ARG GIT_COMMIT=' in dockerfile and 'ENV GIT_COMMIT=${GIT_COMMIT}' in dockerfile
```

Sustituir `test_init_passes_the_run_job_parameters_as_one_argument` por:

```python
def fake_python(tmp_path, code=0):
    """Un `python` falso en el PATH que anota sus argumentos y termina con `code`."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'python'
    script.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "{}"\nexit {}\n'.format(tmp_path / 'args', code))
    script.chmod(0o755)
    return bin_dir


def run_init(tmp_path, env):
    import os
    bin_dir = fake_python(tmp_path, code=3)
    full = {'PATH': '{}:{}'.format(bin_dir, os.environ['PATH']), **env}
    return subprocess.run(['bash', 'deploy/docker/init.sh'], cwd=ROOT, env=full, capture_output=True, text=True)


def test_init_execs_the_job_with_the_json_as_one_argument_and_returns_its_exit_code(tmp_path):
    init = read('deploy/docker/init.sh')
    assert 'exec python /app/job.py "$job_name" ${job_params:+"$job_params"}' in init
    done = run_init(tmp_path, {'RUN_JOB': 'publish {"what":["manifest"],"freeze":{"active":true,"message":"Veda electoral"}}'})
    assert done.returncode == 3
    assert (tmp_path / 'args').read_text().splitlines() == ['/app/job.py', 'publish', '{"what":["manifest"],"freeze":{"active":true,"message":"Veda electoral"}}']


def test_init_refuses_an_idle_container(tmp_path):
    done = run_init(tmp_path, {})
    assert done.returncode == 1 and 'Nothing to run' in done.stderr
    assert subprocess.run(['bash', '-n', 'deploy/docker/init.sh'], cwd=ROOT).returncode == 0
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_deploy_files.py -q`
Expected: FAIL.

- [ ] **Step 3: Editar `Dockerfile`, `init.sh` y las tres plantillas**

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_deploy_files.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add Dockerfile deploy/docker/init.sh deploy/ecs tests/test_deploy_files.py
git commit -m "deploy: GIT_COMMIT en la imagen, init.sh con exec y salida 1, plantillas de ECS como la infraestructura real"
```

---

### Task 4: Lanzador `deploy/ecs/job.py` (Python con boto3) con `--register` y `--follow`

Decidido por Luis el 2026-10-31: lanzador en Python con boto3, fichero `deploy/ecs/job.py` (sustituye al
`deploy/run-job.sh` de la spec). No importa `mtpy` (corre en el portátil con las credenciales de la CLI por la
cadena normal de boto3) y toda la lógica es inyectable con clientes falsos, así que los tests no necesitan red ni
un `aws` de mentira.

**Files:**
- Create: `deploy/ecs/job.py`, `deploy/ecs.env.example`
- Test: `tests/test_ecs_job.py` (nuevo)

**Interfaces:**
- `deploy/ecs.env.example` (claves con comentario de una línea): `ECS_CLUSTER=elections-model-cluster`,
  `ECS_SERVICE=elections-model-service`, `ECS_TASK_FAMILY=elections-model-job`, `AWS_REGION=eu-west-1`,
  `ECS_LOG_GROUP=/ecs/elections-model`, y vacías para sobreescribir la red del servicio: `ECS_SUBNETS=`
  (separadas por comas), `ECS_SECURITY_GROUPS=`, `ECS_PUBLIC_IP=` (`ENABLED`/`DISABLED`). `AWS_PROFILE` se deja a la
  cadena de boto3 (variable de entorno del shell).
- `deploy/ecs/job.py` (sólo `argparse`, `json`, `os`, `sys`, `time` y `boto3`, importado dentro de `clients`):
  - `class UsageError(Exception)`, `class ConfigError(Exception)`, `class LaunchError(Exception)`.
  - `load_config(path: str) -> dict`: `KEY=VALUE` por línea, líneas vacías y `#` ignoradas, valores con `strip`;
    sin fichero, `ConfigError('no ecs.env at {path} (copy deploy/ecs.env.example)')`.
  - `run_job_value(job: Optional[str], params: Optional[str]) -> str`: job vacío →
    `UsageError('job name is required')`; `params` que no parsea con `json.loads` → `UsageError('invalid JSON: {e}')`;
    `job == 'load'` con `'params'` en `what` (cadena o lista) →
    `UsageError('load params runs on the laptop (it rewrites data/params.json)')`; devuelve `'{job} {params}'`
    con el JSON tal cual se recibió (sin reformatear) o `job` solo.
  - `service_network(ecs, cluster: str, service: str) -> dict`: `ecs.describe_services(cluster=cluster, services=[service])`
    → `services[0]['deployments'][0]['networkConfiguration']['awsvpcConfiguration']`.
  - `network_from(config: dict, live: dict) -> dict`: copia de `live` con `subnets` (`ECS_SUBNETS` partida por
    comas), `securityGroups` (`ECS_SECURITY_GROUPS`) y `assignPublicIp` (`ECS_PUBLIC_IP`) sobreescritos cuando la
    clave no está vacía.
  - `run_task_request(config: dict, network: dict, run_job: str) -> dict`: los kwargs de `ecs.run_task`:
    `cluster`, `taskDefinition` (`ECS_TASK_FAMILY`, por defecto `elections-model-job`), `launchType='FARGATE'`,
    `count=1`, `networkConfiguration={'awsvpcConfiguration': network}`,
    `overrides={'containerOverrides': [{'name': 'Main', 'environment': [{'name': 'RUN_JOB', 'value': run_job}]}]}`.
  - `launch(ecs, request: dict) -> str`: `ecs.run_task(**request)`; con `failures` no vacío o sin `tasks` →
    `LaunchError('{reason}: {detail}')` (`detail` puede ser `None`); devuelve `tasks[0]['taskArn']`.
  - `task_id(arn: str) -> str`: el último segmento del ARN.
  - `log_stream(task_arn: str) -> str`: `'job/Main/{id}'`.
  - `follow(ecs, logs, cluster: str, task_arn: str, log_group: str, poll: float = 15, sleep=time.sleep, out=print) -> int`:
    bucle: imprime los eventos nuevos del flujo con `logs.filter_log_events(logGroupName=log_group, logStreamNames=[stream], startTime=last + 1)`
    (paginando con `nextToken`; `logs.exceptions.ResourceNotFoundException` mientras el flujo no existe se ignora);
    `ecs.describe_tasks(cluster=cluster, tasks=[task_arn])`; cuando `tasks[0]['lastStatus'] == 'STOPPED'`: vuelca
    el log una última vez, imprime `stoppedReason` y devuelve `tasks[0]['containers'][0].get('exitCode')`, o `1`
    si no hay código; si no, `sleep(poll)` y repite.
  - `derive_job_definition(web: dict, family: str) -> dict`: a partir de la definición viva (`describe_task_definition(...)['taskDefinition']`):
    `family`, `cpu='2048'`, `memory='4096'`; conserva `networkMode`, `requiresCompatibilities`, `runtimePlatform`,
    `taskRoleArn`, `executionRoleArn`; el contenedor conserva `name`, `image`, `essential`, `environmentFiles`,
    `logConfiguration` (con `awslogs-stream-prefix` `job`), lleva `environment=[{'name': 'RUN_JOB', 'value': ''}]` y
    pierde `portMappings`; se quitan `taskDefinitionArn`, `revision`, `status`, `requiresAttributes`,
    `compatibilities`, `registeredAt`, `registeredBy`.
  - `register(ecs, cluster: str, service: str, family: str) -> str`: `describe_services` → `services[0]['taskDefinition']`
    → `describe_task_definition(taskDefinition=arn)` → `derive_job_definition` → `register_task_definition(**definition)`
    → `'{family}:{revision}'`.
  - `clients(region: Optional[str]) -> tuple`: `(boto3.client('ecs', region_name=region), boto3.client('logs', region_name=region))`
    (el único sitio que importa `boto3`).
  - `main(argv: list, ecs=None, logs=None, out=print, err=None) -> int`: `argparse` con `job` y `params` posicionales
    opcionales, `--follow`, `--register`; config de `ECS_ENV_FILE` o `deploy/ecs.env` (junto a `deploy/`, resuelto
    desde `__file__`); `UsageError`/`ConfigError` → mensaje en stderr + uso, salida 2; `--register` → imprime
    `registered {family:revision}`, salida 0; lanzamiento: `run_job_value`, red (`service_network` + `network_from`),
    `launch` (`LaunchError` → stderr, salida 1), imprime `task {id}`, `aws logs tail {log_group} --log-stream-names {stream} --follow --since 1h`
    y `https://{region}.console.aws.amazon.com/ecs/v2/clusters/{cluster}/tasks/{id}`; sin `--follow` salida 0; con
    `--follow` devuelve lo que devuelva `follow`. Los clientes se crean con `clients(config.get('AWS_REGION'))` sólo
    cuando no se inyectan. `if __name__ == '__main__': sys.exit(main(sys.argv[1:]))`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_ecs_job.py` (el módulo se carga por ruta porque `deploy/ecs/` no es un paquete y el nombre `job`
choca con el `job.py` de la raíz):

```python
"""Tests del lanzador `deploy/ecs/job.py` con clientes de boto3 falsos (sin red, sin mtpy)."""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('ecs_job', ROOT / 'deploy' / 'ecs' / 'job.py')
ecs_job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ecs_job)

LIVE_TASKDEF = {
    'taskDefinitionArn': 'arn:aws:ecs:eu-west-1:1:task-definition/elections-model-cluster-elections-model-service:33',
    'family': 'elections-model-cluster-elections-model-service', 'revision': 33, 'status': 'ACTIVE',
    'networkMode': 'awsvpc', 'requiresCompatibilities': ['FARGATE'], 'cpu': '1024', 'memory': '2048',
    'runtimePlatform': {'cpuArchitecture': 'X86_64', 'operatingSystemFamily': 'LINUX'},
    'taskRoleArn': 'arn:aws:iam::1:role/ecsTaskExecutionRole', 'executionRoleArn': 'arn:aws:iam::1:role/ecsTaskExecutionRole',
    'requiresAttributes': [{'name': 'x'}], 'compatibilities': ['EC2', 'FARGATE'], 'registeredAt': 1.0, 'registeredBy': 'x',
    'containerDefinitions': [{'name': 'Main', 'image': '1.dkr.ecr.eu-west-1.amazonaws.com/elections-model:latest',
                              'essential': True, 'portMappings': [{'containerPort': 8042, 'protocol': 'tcp'}],
                              'environment': [], 'environmentFiles': [{'value': 'arn:aws:s3:::elections-s3-bucket/elections.env', 'type': 's3'}],
                              'logConfiguration': {'logDriver': 'awslogs', 'options': {'awslogs-group': '/ecs/elections-model', 'awslogs-region': 'eu-west-1', 'awslogs-stream-prefix': 'ecs'}}}]}
NETWORK = {'subnets': ['subnet-1', 'subnet-2'], 'securityGroups': ['sg-1'], 'assignPublicIp': 'ENABLED'}


class ResourceNotFound(Exception):
    pass


class FakeEcs:
    """Cliente ECS falso: anota las llamadas; `fail` simula un run-task sin capacidad."""
    def __init__(self, fail=False):
        self.calls, self.fail, self.statuses = [], fail, ['RUNNING', 'STOPPED']

    def describe_services(self, **kw):
        self.calls.append(('describe_services', kw))
        return {'services': [{'taskDefinition': LIVE_TASKDEF['taskDefinitionArn'],
                              'deployments': [{'networkConfiguration': {'awsvpcConfiguration': dict(NETWORK)}}]}]}

    def run_task(self, **kw):
        self.calls.append(('run_task', kw))
        if self.fail:
            return {'tasks': [], 'failures': [{'arn': 'x', 'reason': 'RESOURCE:CPU', 'detail': None}]}
        return {'tasks': [{'taskArn': 'arn:aws:ecs:eu-west-1:1:task/c/abc123'}], 'failures': []}

    def describe_tasks(self, **kw):
        self.calls.append(('describe_tasks', kw))
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {'tasks': [{'lastStatus': status, 'stoppedReason': 'Essential container in task exited',
                           'containers': [{'exitCode': 3}]}]}

    def describe_task_definition(self, **kw):
        self.calls.append(('describe_task_definition', kw))
        return {'taskDefinition': json.loads(json.dumps(LIVE_TASKDEF))}

    def register_task_definition(self, **kw):
        self.calls.append(('register_task_definition', kw))
        return {'taskDefinition': {'family': kw['family'], 'revision': 2}}


class FakeLogs:
    """Cliente de logs falso: el flujo no existe en la primera llamada, luego devuelve dos eventos."""
    exceptions = types.SimpleNamespace(ResourceNotFoundException=ResourceNotFound)

    def __init__(self):
        self.calls = []

    def filter_log_events(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            raise ResourceNotFound('no stream yet')
        if len(self.calls) == 2:
            return {'events': [{'timestamp': 10, 'message': 'Run job [backtest]...'}, {'timestamp': 11, 'message': 'es: backtest completed'}]}
        return {'events': []}


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / 'ecs.env'
    path.write_text('# cluster\nECS_CLUSTER=elections-model-cluster\nECS_SERVICE=elections-model-service\nAWS_REGION=eu-west-1\nECS_LOG_GROUP=/ecs/elections-model\nECS_SUBNETS=\n')
    monkeypatch.setenv('ECS_ENV_FILE', str(path))
    return path


def run(argv, ecs=None, logs=None):
    out, err = io.StringIO(), io.StringIO()
    code = ecs_job.main(argv, ecs=ecs if ecs is not None else FakeEcs(), logs=logs if logs is not None else FakeLogs(), out=out.write, err=err.write)
    return code, out.getvalue(), err.getvalue()


def test_module_loads_without_mtpy_nor_boto3():
    assert 'mtpy' not in sys.modules and not any(name.startswith('mtpy.') for name in sys.modules)
    assert 'boto3' not in sys.modules or sys.modules['boto3'] is not None  # boto3 only inside clients()
    source = (ROOT / 'deploy' / 'ecs' / 'job.py').read_text()
    assert 'import mtpy' not in source and 'from mtpy' not in source
    assert source.count('import boto3') == 1 and 'def clients(' in source


def test_run_job_value_refuses_empty_bad_json_and_load_params():
    with pytest.raises(ecs_job.UsageError, match='job name is required'):
        ecs_job.run_job_value('', None)
    with pytest.raises(ecs_job.UsageError, match='invalid JSON'):
        ecs_job.run_job_value('publish', '{not json')
    with pytest.raises(ecs_job.UsageError, match='load params runs on the laptop'):
        ecs_job.run_job_value('load', '{"what":["polls","params"],"scopes":"all"}')
    with pytest.raises(ecs_job.UsageError, match='load params runs on the laptop'):
        ecs_job.run_job_value('load', '{"what":"params"}')
    assert ecs_job.run_job_value('backtest', None) == 'backtest'
    assert ecs_job.run_job_value('load', '{"scopes":["es"]}') == 'load {"scopes":["es"]}'


def test_usage_errors_exit_2_without_launching(env_file):
    ecs = FakeEcs()
    assert run([], ecs)[0] == 2 and 'usage' in run([], ecs)[2].lower()
    code, _, err = run(['load', '{"what":["params"]}'], ecs)
    assert code == 2 and 'load params runs on the laptop' in err
    assert run(['publish', '{bad'], ecs)[0] == 2
    assert not [c for c in ecs.calls if c[0] == 'run_task']


def test_missing_env_file_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv('ECS_ENV_FILE', str(tmp_path / 'none.env'))
    code, _, err = run(['backtest'])
    assert code == 2 and 'ecs.env.example' in err


def test_launch_builds_the_run_task_from_the_service_network(env_file):
    ecs = FakeEcs()
    params = '{"what":["manifest"],"freeze":{"active":true,"message":"Veda electoral"}}'
    code, out, _ = run(['publish', params], ecs)
    assert code == 0, out
    request = next(kw for name, kw in ecs.calls if name == 'run_task')
    assert request['cluster'] == 'elections-model-cluster' and request['taskDefinition'] == 'elections-model-job'
    assert request['launchType'] == 'FARGATE' and request['count'] == 1
    assert request['networkConfiguration'] == {'awsvpcConfiguration': NETWORK}
    override = request['overrides']['containerOverrides'][0]
    assert override['name'] == 'Main' and override['environment'] == [{'name': 'RUN_JOB', 'value': 'publish ' + params}]
    assert 'task abc123' in out and 'aws logs tail /ecs/elections-model --log-stream-names job/Main/abc123' in out
    assert 'console.aws.amazon.com/ecs/v2/clusters/elections-model-cluster/tasks/abc123' in out


def test_env_file_overrides_the_network(env_file, monkeypatch):
    env_file.write_text(env_file.read_text() + 'ECS_SUBNETS=subnet-9\nECS_PUBLIC_IP=DISABLED\n')
    ecs = FakeEcs()
    assert run(['backtest', '{"scopes":"all"}'], ecs)[0] == 0
    request = next(kw for name, kw in ecs.calls if name == 'run_task')
    assert request['networkConfiguration']['awsvpcConfiguration'] == {'subnets': ['subnet-9'], 'securityGroups': ['sg-1'], 'assignPublicIp': 'DISABLED'}
    assert request['overrides']['containerOverrides'][0]['environment'][0]['value'] == 'backtest {"scopes":"all"}'


def test_placement_failure_exits_1_with_the_reason(env_file):
    code, out, err = run(['backtest'], FakeEcs(fail=True))
    assert code == 1 and 'RESOURCE:CPU' in err


def test_follow_prints_the_log_and_exits_with_the_container_exit_code(env_file, monkeypatch):
    ecs, logs = FakeEcs(), FakeLogs()
    monkeypatch.setattr(ecs_job.time, 'sleep', lambda s: None)
    code, out, _ = run(['backtest', '{"scopes":["es"]}', '--follow'], ecs, logs)
    assert code == 3
    assert 'Run job [backtest]...' in out and 'es: backtest completed' in out and 'Essential container in task exited' in out
    assert logs.calls[0]['logStreamNames'] == ['job/Main/abc123'] and logs.calls[0]['logGroupName'] == '/ecs/elections-model'
    assert [name for name, _ in ecs.calls if name == 'describe_tasks'] == ['describe_tasks', 'describe_tasks']


def test_register_derives_the_job_definition_from_the_live_web_one(env_file):
    ecs = FakeEcs()
    code, out, _ = run(['--register'], ecs)
    assert code == 0 and 'registered elections-model-job:2' in out
    definition = next(kw for name, kw in ecs.calls if name == 'register_task_definition')
    assert definition['family'] == 'elections-model-job' and definition['cpu'] == '2048' and definition['memory'] == '4096'
    assert definition['taskRoleArn'].endswith('ecsTaskExecutionRole') and definition['requiresCompatibilities'] == ['FARGATE']
    assert not {'taskDefinitionArn', 'revision', 'status', 'requiresAttributes', 'compatibilities', 'registeredAt', 'registeredBy'} & set(definition)
    container = definition['containerDefinitions'][0]
    assert container['name'] == 'Main' and 'portMappings' not in container
    assert container['environment'] == [{'name': 'RUN_JOB', 'value': ''}]
    assert container['environmentFiles'] == LIVE_TASKDEF['containerDefinitions'][0]['environmentFiles']
    assert container['logConfiguration']['options']['awslogs-stream-prefix'] == 'job'
    assert not [c for c in ecs.calls if c[0] == 'run_task']
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_ecs_job.py -q`
Expected: FAIL (`deploy/ecs/job.py` no existe).

- [ ] **Step 3: Escribir `deploy/ecs/job.py` y `deploy/ecs.env.example`**

Según Interfaces; docstring de módulo con el uso y los tres comandos de ejemplo. El fichero no es ejecutable
directo (`python deploy/ecs/job.py ...`); no hace falta `chmod`.

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_ecs_job.py tests/test_deploy_files.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git status
git add deploy/ecs/job.py deploy/ecs.env.example tests/test_ecs_job.py
git commit -m "deploy: deploy/ecs/job.py lanza un job como tarea de ECS (--register, --follow) desde el portátil"
```

---

### Task 5: README: «Jobs desde ECS», flujo del backtest, veda y campaña con el lanzador

**Files:**
- Modify: `deploy/README.md` («Fichero de configuración» 26-36, «Construir la imagen» 37-48, «Despliegue en ECS»
  423-491, «Backtest del modelo» 715-803, «Publicar» rutina 836-868, «Backtest» 1078-1101, «Publicar desde ECS»
  1187-1215, «Veda» 1216-1256, «Duraciones medidas» 1355-)
- Modify: `README.md` («Jobs» 16-30)
- Modify: `docs/superpowers/specs/2026-10-31-jobs-pipelines-ecs-design.md` (línea de estado)

- [ ] **Step 1: «Despliegue en ECS» y «Construir la imagen»**

Reescribir «Despliegue en ECS» con la infraestructura real (nombres, `environmentFiles` desde
`s3://<bucket>/elections.env`, el mismo rol de tarea y de ejecución, logs `/ecs/elections-model`, subredes
públicas con IP pública, sin NAT); «Qué debe existir en AWS» reducido a lo que hay; la tabla de marcadores queda
con `<account>`, `<region>`, `<bucket>`, `<task-role>`, `<execution-role>`, `<subnet-id>`,
`<security-group-id>`; «Procedimiento» con
`docker build --platform linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) -t elections-model:<tag> .`
(también en «Construir la imagen») y la nota de que `AWS_KEY`/`AWS_SECRET` del fichero de entorno son con las
que la app escribe en S3. «Fichero de configuración» añade `deploy/ecs.env` (clúster y servicio del lanzador,
plantilla `deploy/ecs.env.example`).

- [ ] **Step 2: «Jobs desde ECS» en lugar de «Publicar desde ECS»**

Contenido: `python deploy/ecs/job.py --register` la primera vez y tras cada imagen nueva; `python deploy/ecs/job.py publish '{"what":["forecast","analysis"],"scopes":["es"],"n_sim":10000}' --follow`;
qué imprime (id, `aws logs tail`, enlace), dónde está el log (grupo `/ecs/elections-model`, flujo
`job/Main/<id>`, una sola copia: `CWL_GROUP` vacío), el aviso de Pushover (cabecera + fallos delante, 1.024
caracteres), códigos de salida (0 / 1; con `--follow` el del contenedor), `stop_on_error`, publicaciones una
tras otra (el manifest es un leer-modificar-escribir sin bloqueo), `aws ecs stop-task` deja un run incompleto
que se retira con `unpublish`, qué corre en ECS y qué no (`load params` en el portátil, rechazado por el
lanzador), coste del orden de 10 céntimos por hora de tarea (2 vCPU, 4 GB, eu-west-1), y que la memoria de
`publish forecast es` con 10.000 simulaciones se mide en la primera tarea.

- [ ] **Step 3: Backtest en S3**

En «Backtest del modelo»: las tablas viven en `backtest/results/` del sistema de ficheros de la app (S3 en ECS
y en el portátil con `.env`; `files/` sin bucket), `out` sigue siendo una carpeta local; acciones `push` y
`pull` con sus ejemplos; el flujo tras un cambio del método: `python deploy/ecs/job.py backtest '{"scopes":"all"}'` →
`python job.py backtest '{"action":"pull","scopes":"all"}'` → revisar `by_horizon.csv`, commit →
`python deploy/ecs/job.py publish '{"what":["backtest","event"],"scopes":"all","event_date":"all"}'`; la siembra inicial
`python job.py backtest '{"action":"push","scopes":"all"}'` (paso de Luis). En «Publicar» › «Backtest» y en la
rutina «Tras un cambio del modelo», `source` pasa a ser opcional (por defecto el sistema de ficheros de la
app) y `series.csv` ya no obliga a publicar desde la máquina del backtest. Quitar la frase de que `publish
backtest` «solo lee ficheros del repositorio».

- [ ] **Step 4: «Veda» y «Campaña» apuntando al lanzador**

Los pasos 1 y 3 de «Veda», el descongelado y las dos llamadas diarias de «Campaña» con `deploy/ecs/job.py` (y la
forma local como alternativa). En «Duraciones medidas», una línea para que Luis anote la primera tarea de ECS
(duración y memoria).

- [ ] **Step 5: `README.md` y cierre**

En «Jobs» del `README.md`: una línea con `deploy/ecs/job.py` y el enlace a `deploy/README.md`. En la spec,
línea de estado: «Fase 2 construida el <fecha> (<commits>); pasos de Luis: `--register`, `backtest push`, medir
memoria».

Run: `python -m pytest -q`
Expected: todos en verde.

```bash
git status
git add deploy/README.md README.md docs/superpowers/specs/2026-10-31-jobs-pipelines-ecs-design.md
git commit -m "docs: jobs desde ECS con deploy/ecs/job.py, backtest en S3 y veda/campaña con el lanzador"
```
