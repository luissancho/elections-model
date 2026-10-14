# Web de resultados, fase 1 (paquete y comando `publish`): plan de implementación

> **Estado:** completado el 2026-10-08 en `dev` (commits 30a6369..9471dfa más el commit de documentación). Ver el
> estado de cierre en la spec.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un comando `python job.py publish` que ejecuta el modelo MT para un ámbito, exporta sus salidas a
un paquete inmutable de JSON y CSV en `app.fs` (`files/site/v1/` en local, `s3://{bucket}/site/v1/` en
producción) con procedencia, y mantiene los punteros (`history.json`, `manifest.json`) que la API de la
fase 2 leerá.

**Architecture:** tres módulos nuevos sin tocar el modelo: `mtpy/lib/bundle.py` (disposición de ficheros,
sobre JSON, esquemas y lector/escritor sobre `app.fs`), `mtpy/lib/publish.py` (exportadores puros sobre
un `Simulator` ya ejecutado y orquestadores `publish_forecast`, `rebuild_history`, `update_manifest`,
`point`, `unpublish`) y `mtpy/jobs/Publish.py` (el job del framework: parámetros, aislamiento de errores
por ámbito, resumen y `alert()`). Los tests unitarios ejercitan los métodos reales de salida del
`Simulator` sobre un objeto reconstruido con `__new__` y arrays sintéticos; una prueba de integración
publica `es` con `n_sim=20` contra la base.

**Tech Stack:** Python 3.11 (`python` del venv), pandas 2.2.2, numpy 2.4.4, mtpy (`app.fs`: `FileSystem`
local o `S3` sobre `s3fs==0.4.2`), pytest 9.

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`, secciones "Paquete publicado",
"Comando de publicar", "Pruebas", fase 1 de "Fases", "Verificación" y "Estado al cierre de la fase 0".

## Global Constraints

- PEP8 y docstring en toda función, clase y método nuevos (AGENTS.md). Docstrings y comentarios en inglés
  dentro de `mtpy/`, estilo numpydoc (`Parameters` / `Returns`); en español en `tests/`, `docs/` y
  `deploy/README.md`.
- `pytest -m "not integration"` no arranca `mtpy.run()` ni toca la base (hoy 192 tests en verde);
  `pytest.ini` convierte `FutureWarning` en error: en los exportadores nada de `applymap` ni de
  `groupby.apply` sin `include_groups=False`.
- Árbol limpio al empezar (`git status --short` vacío, rama `dev`). Cada commit añade solo los ficheros de
  su tarea con `git add <ruta>`; nunca `-A`, `.` ni `-a`. Mensajes cortos en inglés terminados en
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Valores fijados por la spec: `PREFIX = 'site/v1'`; `dry_run` escribe en `site-dry/v1` y no toca
  punteros; `CONTRACT = 1`; `run_id` = `YYYYMMDD-HHMMSS` en UTC, uno por invocación; sobre común
  `{"schema": "<nombre>@1", "contract": 1, "scope", "run_id"|null, "mode"|null, "generated_at", "data"}`;
  `null` por NaN; fechas `YYYY-MM-DD`; instantes `YYYY-MM-DDTHH:MM:SSZ`; porcentajes con 2 decimales,
  probabilidades con 3, estadísticos de escaños con 1; series largas en columnas, tablas en `records`.
- Parámetros por defecto del comando: `n_sim=1000`, `seed=42`, `drange=6`, `max_fc=10`, `alpha=0.05`,
  correctores `house_effects=True, dispersion=True, regional_noise=True, industry_bias=False,
  composition=None` (los del notebook `PollsSimulations`). Por ámbito: `Simulator(mode='nowcast')` →
  `fit_forecast(names=sim.params['names'], max_fc, fillna=True)` → `run(split=True, random=True, n_sim)`
  → `sim.mode = 'forecast'` → `run(split=True, random=True, n_sim, horizon='deadline')`.
- Un run nunca se reescribe: el escritor rehúsa si `runs/{scope}/{run_id}/` ya existe. `headline.json`
  es el último fichero del run (un run sin él está incompleto y se ignora).
- Guarda LOREG: `scope == 'es'` y hoy dentro de los 5 días previos a `event_date` (ambos inclusive) →
  rehúsa salvo `force`.
- Fase 1 solo implementa `what` ∈ `forecast`, `manifest`, `point`, `unpublish`; `analysis`, `event` y
  `backtest` lanzan `ValueError` que nombra la fase en que llegan (4, 5 y 5).
- Nada de esta fase cambia `simulator.py`, `forecaster.py` ni `computer.py`; de `mtpy/core/api.py` solo
  se mueve `json_default` (pendiente de la fase 0) a `mtpy/core/utils/serialize.py`, dejándolo
  reexportado.
- Credenciales: las publicaciones a S3 y contra la RDS usan `deploy/elections.env` exportado en la
  shell (`set -a; . deploy/elections.env; set +a`); nunca se copia a ficheros versionados.

## Review Focus

1. Ámbito sin evento próximo (`get_next_event_date` → `None`) o sin sondeos (`ValueError` de
   `require_polls`/`require_forecast`): queda `skipped` con motivo, los demás ámbitos siguen y
   `manifest.json` conserva su entrada anterior (tests en Tarea 7).
2. NaN en `float` nativos de las salidas (`p_first` de un bloque sin partidos, `herd_ratio`, `prior`):
   el paquete escribe `null`, nunca un `ValueError` de `allow_nan=False` (tests en Tareas 2, 4 y 5).
3. Directorio de run incompleto por una publicación interrumpida (sin `headline.json`): `list_runs` lo
   excluye, `history.json` lo ignora y `point` lo rechaza (tests en Tareas 3 y 6).
4. `unpublish` del run al que apunta el manifest: el manifest reapunta al último run restante o retira
   el ámbito; sin runs, `history.json` queda con lista vacía (tests en Tarea 6).
5. Reintento con el mismo `run_id` (dos invocaciones en el mismo segundo UTC, o un `RUN_JOB` repetido):
   `begin_run` lanza `FileExistsError` antes de ejecutar el modelo (tests en Tareas 3 y 6).

---

### Task 1: `json_default` a `mtpy/core/utils/serialize.py`, con `np.longdouble`

**Files:**
- Create: `mtpy/core/utils/serialize.py`, `tests/test_serialize_unit.py`
- Modify: `mtpy/core/api.py:16-110` (quitar `_nan_to_none` y `json_default`; importar)

**Interfaces:**
- Produces: `nan_to_none(value) -> Any` (la `_nan_to_none` actual, pública) y `json_default(obj) -> Any`
  con las ramas de la fase 0 más una: dentro de la rama `np.generic`, un `np.floating` devuelve
  `nan_to_none(float(obj))` (en Linux x86-64 `np.longdouble.item()` devuelve otro `longdouble`, que
  `json` no serializa).
- Produces: `mtpy.core.api.json_default` sigue existiendo (`from .utils.serialize import json_default`),
  así que `Response.set_content` y los tests de la fase 0 no cambian.

- [ ] **Step 1: Crear `tests/test_serialize_unit.py`**

```python
"""Tests de `mtpy/core/utils/serialize.py`: conversiones a JSON de valores numpy y pandas."""
import datetime
import json

import numpy as np
import pandas as pd
import pytest

from mtpy.core.utils.serialize import json_default, nan_to_none


def test_longdouble_becomes_a_native_float():
    value = json_default(np.longdouble('1.5'))
    assert type(value) is float and value == 1.5
    assert json_default(np.longdouble('nan')) is None
    assert json.dumps({'x': np.longdouble('2.25')}, default=json_default) == '{"x": 2.25}'


def test_numpy_scalars_arrays_and_dates():
    assert json_default(np.int64(3)) == 3 and type(json_default(np.int64(3))) is int
    assert json_default(np.array([1.0, np.nan])) == [1.0, None]
    assert json_default(pd.Timestamp('2026-10-08')) == '2026-10-08'
    assert json_default(pd.Timestamp('2026-10-08 12:30')) == '2026-10-08T12:30:00'
    assert json_default(pd.NaT) is None and json_default(pd.NA) is None
    assert json_default(datetime.date(2026, 10, 8)) == '2026-10-08'
    assert sorted(json_default({1, 2})) == [1, 2]


def test_nan_to_none_is_recursive_and_timedelta_is_rejected():
    assert nan_to_none([1.0, [float('nan'), 2.0]]) == [1.0, [None, 2.0]]
    with pytest.raises(TypeError):
        json_default(np.timedelta64(1, 'D'))


def test_core_api_still_exports_json_default():
    from mtpy.core import api

    assert api.json_default is json_default
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_serialize_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'mtpy.core.utils.serialize'`.

- [ ] **Step 3: Crear el módulo y recortar `mtpy/core/api.py`**

Mover `_nan_to_none` (renombrada `nan_to_none`) y `json_default` con sus docstrings a
`mtpy/core/utils/serialize.py` (imports: `datetime`, `math`, `numpy`, `pandas`). En la rama
`isinstance(obj, np.generic)`, antes de `obj.item()`: `if isinstance(obj, np.floating): return
nan_to_none(float(obj))`. En `api.py`, sustituir ambas definiciones por `from .utils.serialize import
json_default` y eliminar los imports que queden sin uso (`math`, y `datetime`/`numpy`/`pandas` si ya no
se usan en el módulo).

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_serialize_unit.py tests/test_api_asgi.py -v`
Expected: PASS (4 nuevos + los 30 de la fase 0).

- [ ] **Step 5: Comprobar la suite unitaria completa**

Run: `python -m pytest -m "not integration" -q`
Expected: 196 passed.

- [ ] **Step 6: Commit**

```bash
git add mtpy/core/utils/serialize.py mtpy/core/api.py tests/test_serialize_unit.py
git commit -m "$(printf 'Move json_default to core utils and accept longdouble\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 2: `mtpy/lib/bundle.py`: disposición, sobre JSON, esquemas y `validate`

**Files:**
- Create: `mtpy/lib/bundle.py`, `tests/test_bundle_unit.py`, `tests/fixtures/bundle/<esquema>.json` (15 ficheros)

**Interfaces:**
- Produces (constantes): `PREFIX = 'site/v1'`, `DRY_PREFIX = 'site-dry/v1'`, `CONTRACT = 1`,
  `MODES = ('nowcast', 'forecast')`, `RUN_PARTS = ('meta', 'headline', 'series', 'polls', 'fan',
  'house-effects', 'dispersion')`, `MODE_PARTS = ('vote', 'summary', 'dist', 'districts', 'scenario',
  'projection')`, `RUN_ID_RE = re.compile(r'^\d{8}-\d{6}$')`, `ROUTE_ALIAS_RE =
  re.compile(r'^[0-9a-z_\-]+$')` (el alias de parámetro de ruta de `mtpy/core/api.py`).
- Produces: `run_id(now: Optional[datetime] = None) -> str` (`%Y%m%d-%H%M%S` en UTC; `now` consciente de
  zona o `None` → `datetime.now(timezone.utc)`; un `now` naive se toma como UTC) e `iso_utc(moment:
  datetime) -> str` (`YYYY-MM-DDTHH:MM:SSZ`, convirtiendo a UTC).
- Produces: `jsonable(obj) -> Any`, recursiva: `dict` → `dict` con claves `str` (una clave numpy pasa
  antes por `json_default`); `list`/`tuple` → `list`; `float` NaN o ±inf → `None`; `bool`, `int`, `str`
  y `None` tal cual; `pd.DataFrame`/`pd.Series` → `TypeError('export frames as records or columns
  before serialising')`; cualquier otro tipo → `json_default(obj)` y recursión sobre el resultado.
- Produces: `dumps(obj) -> bytes` = `json.dumps(jsonable(obj), ensure_ascii=False, allow_nan=False,
  separators=(',', ':')).encode('utf-8')` y `loads(raw: bytes) -> Any`.
- Produces: `envelope(schema: str, data, scope: Optional[str], run_id: Optional[str] = None, mode:
  Optional[str] = None, generated_at: Optional[str] = None) -> dict` con las claves del sobre en ese
  orden; `generated_at` por defecto `iso_utc(datetime.now(timezone.utc))`.
- Produces (rutas relativas al prefijo; único sitio que las conoce): `path_manifest() ->
  'manifest.json'`, `path_runs(scope) -> 'runs/{scope}'`, `path_history(scope) ->
  'runs/{scope}/history.json'`, `path_run(scope, run_id) -> 'runs/{scope}/{run_id}'`, `path_part(scope,
  run_id, part, mode=None) -> 'runs/{scope}/{run_id}/{part}.json'` o `'.../{mode}/{part}.json'`,
  `path_csv(scope, run_id, name, mode=None) -> 'runs/{scope}/{run_id}/csv/{name}.csv'` o
  `'.../csv/{mode}-{name}.csv'`.
- Produces: `SCHEMAS` y `validate(name: str, obj: dict) -> dict`:

```python
NULLABLE_STR = (str, type(None))
NULLABLE_INT = (int, type(None))
SCHEMAS = {
    'manifest': {'contract': int, 'updated_at': NULLABLE_STR, 'scopes': dict, 'freeze': dict, 'attribution': dict},
    'history': {'scope': str, 'runs': list},
    'meta': {
        'run_id': str, 'run_at': str, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'scope': str,
        'event_date': str, 'as_of': str, 'date_last': str, 'date_fit_last': NULLABLE_STR, 'horizon_max': int,
        'n_sim': int, 'seed': NULLABLE_INT, 'drange': list, 'max_fc': int, 'alpha': float, 'correctors': dict,
        'n_polls': int, 'n_pollsters': int, 'db_polls': NULLABLE_INT, 'db_last_poll': NULLABLE_STR,
        'n_seats': int, 'majority': int, 'parties': list, 'bmaps': dict, 'smap': dict, 'regions': list,
        'diagnostics': dict, 'seconds': dict, 'freeze': bool,
    },
    'headline': {'run_id': str, 'run_at': str, 'event_date': str, 'as_of': str, 'date_last': str, 'n_polls': int,
                 'nowcast': dict, 'forecast': dict},
    'series': {'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict},
    'polls': {'parties': list, 'columns': list, 'polls': list, 'results': list},
    'fan': {'horizons': list, 'rows': list},
    'house-effects': {'rows': list},
    'dispersion': {'rows': list},
    'vote': {'horizon': int, 'when': str, 'rows': list},
    'summary': {'n_seats': int, 'majority': int, 'parties': list, 'vs': list, 'blocks': list, 'p_majority': dict,
                'totals': dict},
    'dist': {'n_seats': int, 'parties': list, 'seats': list},
    'districts': {'parties': list, 'regions': list, 'rows': list},
    'scenario': {'simulation': int, 'parties': list, 'rows': list},
    'projection': {'dates': list, 'groups': dict},
}
```

  `validate` comprueba, en este orden y con `ValueError` cuyo mensaje empieza por `'{name}: '`: `name`
  está en `SCHEMAS` (`'{name}: unknown schema'`); `obj['schema'] == f'{name}@{CONTRACT}'` (`'{name}:
  schema {got} != {expected}'`); `obj['contract'] == CONTRACT`; `scope` es `str` o `None` (`'{name}:
  invalid scope'`); `run_id` es `None` o casa con `RUN_ID_RE` (`'{name}: invalid run_id'`); `mode` es
  `None` o está en `MODES`, y no es `None` cuando `name in MODE_PARTS` (`'{name}: invalid mode'`);
  `generated_at` es `str`; `data` es `dict`; claves requeridas presentes (`'{name}: missing keys
  [...]'`, ordenadas) y de tipo correcto (`'{name}: wrong type for {key}'`; `isinstance` contra el tipo
  o tupla de la tabla). Devuelve `obj`.

**Contrato de datos** (forma de `data` de cada esquema; la referencia de las Tareas 4-5 y de
`docs/web/contrato.md`):

| Esquema | `data` |
|---|---|
| `manifest` (scope y run_id `null`) | `{contract: 1, updated_at, scopes: {scope: {latest, run_at, event_date, as_of, date_last, n_polls}}, freeze: {active: bool, message: str\|null}, attribution: {polls, results, model}}` |
| `history` (run_id `null`) | `{scope, runs: [<data de headline>, ...]}` en orden ascendente de `run_id` |
| `meta` | las claves de `SCHEMAS['meta']`; `parties: [{name, id, fullname, color, block, regional}]`, `regions: [{id, name, seats}]`, `drange: [int, int\|null]`, `correctors: {house_effects, dispersion, regional_noise, industry_bias, composition}`, `diagnostics: {drift_k, multiplier, ages: {name: años}, composition, clip_rate: {nowcast, forecast}}`, `seconds: {init, fit, nowcast, forecast, export, total}`, `freeze`: publicado dentro de la ventana LOREG con `force` |
| `headline` | `{run_id, run_at, event_date, as_of, date_last, n_polls, nowcast: {parties: [{name, pct, lo, hi, seats, seats_lo, seats_hi, p_first}], p_majority: {bloque: p}}, forecast: {...}}` |
| `series` | `{dates: [YYYY-MM-DD], parties: [name], mean: {name: [num\|null]}, lo: {...}, hi: {...}}` |
| `polls` | `{parties, columns: POLL_COLUMNS, polls: [{<columns>, <name>: pct\|null}], results: [{date, <name>: pct}]}` |
| `fan` | `{horizons: [int], rows: [{party, horizon, mean, sd, lo, hi}]}` |
| `house-effects` | `{rows: [{pollster_id, name, pollster, n, w, level, dev, dev_err, prior, prior_err, effect, effect_err, center}]}` |
| `dispersion` | `{rows: [{pollster_id, pollster, n, ss_obs, ss_exp, ratio_raw, ratio, factor, herd_ratio}]}` |
| `vote` (mode) | `{horizon: int, when: YYYY-MM-DD, rows: [{name, pct, sd, lo, hi}]}` |
| `summary` (mode) | `{n_seats, majority, parties: [{name, pct, pct_mean, pct_lo, pct_hi, seats, seats_mean, seats_median, seats_lo, seats_hi, seats_min, seats_max, p_seats, p_majority, p_first}], vs: [...], blocks: [...], p_majority: {bloque: p}, totals: {name: seats}}` |
| `dist` (mode) | `{n_seats, parties, seats: [[int por partido], ...]}`, una fila por simulación |
| `districts` (mode) | `{parties, regions: [{id, name, seats}], rows: [{region_id, region, name, pct, pct_lo, pct_hi, seats, seats_mean, seats_lo, seats_hi, p_seats}]}` sin la región total |
| `scenario` (mode) | `{simulation: int, parties, rows: [{region_id, region, seats: [int por partido]}]}`, con la fila total (`region_id` 0) primero |
| `projection` (mode) | `{dates: [YYYY-MM-DD], groups: {parties\|vs\|blocks: {names: [...], mean: {name: [num]}, lo: {...}, hi: {...}}}}` |

- [ ] **Step 1: Crear `tests/test_bundle_unit.py`**

```python
"""Tests de `mtpy/lib/bundle.py`: disposición del paquete, sobre JSON y validación de esquemas."""
import copy
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from mtpy.lib import bundle

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'bundle')


def load_fixture(name):
    with open(os.path.join(FIXTURES, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def test_run_id_is_utc_and_fits_the_route_alias():
    rid = bundle.run_id(datetime(2026, 10, 8, 14, 15, 3, tzinfo=timezone.utc))
    assert rid == '20261008-141503'
    assert bundle.RUN_ID_RE.match(rid) and bundle.ROUTE_ALIAS_RE.match(rid)
    assert bundle.run_id(datetime(2026, 10, 8, 16, 0, 0, tzinfo=timezone(timedelta(hours=2)))) == '20261008-140000'
    assert bundle.RUN_ID_RE.match(bundle.run_id())
    assert bundle.iso_utc(datetime(2026, 10, 8, 14, 15, 3, tzinfo=timezone.utc)) == '2026-10-08T14:15:03Z'


def test_jsonable_cleans_nan_and_numpy_at_any_depth():
    value = {'a': np.int64(3), 'b': [float('nan'), np.float64(1.5), float('inf')], 'c': {np.int64(28): pd.Timestamp('2026-10-08')},
             'd': np.array([1.0, np.nan]), 'e': pd.NaT, 'f': (1, 2), 'g': True}
    assert bundle.jsonable(value) == {'a': 3, 'b': [None, 1.5, None], 'c': {'28': '2026-10-08'}, 'd': [1.0, None], 'e': None,
                                      'f': [1, 2], 'g': True}
    assert bundle.loads(bundle.dumps({'x': float('nan')})) == {'x': None}
    assert bundle.dumps({'a': 1}) == b'{"a":1}'
    with pytest.raises(TypeError, match='records'):
        bundle.jsonable({'df': pd.DataFrame({'a': [1]})})


def test_paths_follow_the_published_layout():
    assert bundle.path_manifest() == 'manifest.json'
    assert bundle.path_history('es') == 'runs/es/history.json'
    assert bundle.path_run('es-md', '20261008-141503') == 'runs/es-md/20261008-141503'
    assert bundle.path_part('es', '20261008-141503', 'meta') == 'runs/es/20261008-141503/meta.json'
    assert bundle.path_part('es', '20261008-141503', 'vote', mode='nowcast') == 'runs/es/20261008-141503/nowcast/vote.json'
    assert bundle.path_csv('es', '20261008-141503', 'series') == 'runs/es/20261008-141503/csv/series.csv'
    assert bundle.path_csv('es', '20261008-141503', 'vote', mode='forecast') == 'runs/es/20261008-141503/csv/forecast-vote.csv'


def test_envelope_has_the_common_keys():
    env = bundle.envelope('vote', {'x': 1}, 'es', run_id='20261008-141503', mode='nowcast', generated_at='2026-10-08T14:15:03Z')
    assert env == {'schema': 'vote@1', 'contract': 1, 'scope': 'es', 'run_id': '20261008-141503', 'mode': 'nowcast',
                   'generated_at': '2026-10-08T14:15:03Z', 'data': {'x': 1}}
    assert bundle.envelope('manifest', {}, None)['generated_at'].endswith('Z')


@pytest.mark.parametrize('name', sorted(bundle.SCHEMAS))
def test_validate_accepts_every_fixture_and_rejects_a_missing_key(name):
    obj = load_fixture(name)
    assert bundle.validate(name, obj) is obj
    broken = copy.deepcopy(obj)
    key = next(iter(bundle.SCHEMAS[name]))
    del broken['data'][key]
    with pytest.raises(ValueError, match='missing keys'):
        bundle.validate(name, broken)


def test_validate_checks_the_envelope():
    obj = load_fixture('vote')
    with pytest.raises(ValueError, match='schema'):
        bundle.validate('summary', obj)
    with pytest.raises(ValueError, match='run_id'):
        bundle.validate('vote', {**obj, 'run_id': '2026-10-08'})
    with pytest.raises(ValueError, match='mode'):
        bundle.validate('vote', {**obj, 'mode': None})
    with pytest.raises(ValueError, match='wrong type for horizon'):
        bundle.validate('vote', {**obj, 'data': {**obj['data'], 'horizon': '0'}})
    with pytest.raises(ValueError, match='unknown schema'):
        bundle.validate('nope', obj)
```

(`from datetime import datetime, timedelta, timezone` en el import.)

- [ ] **Step 2: Crear los 15 fixtures en `tests/fixtures/bundle/`**

Un fichero por esquema, `<nombre>.json`, con el sobre completo y la forma de `data` de la tabla
"Contrato de datos" con **un** elemento por lista o diccionario (un partido `PP`, una fecha, un bloque
`Derecha`, una región `Madrid`). Sobre: `generated_at` `"2026-10-08T12:00:00Z"`; `scope` `"es"` salvo
`manifest` (`null`); `run_id` `"20261008-120000"` salvo `manifest` e `history` (`null`); `mode`
`"nowcast"` en los seis de `MODE_PARTS` y `null` en el resto. `history.runs` contiene el `data` del
fixture de `headline`; `manifest.scopes.es.latest` es `"20261008-120000"`.

- [ ] **Step 3: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_bundle_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'mtpy.lib.bundle'` en la recogida.

- [ ] **Step 4: Implementar `mtpy/lib/bundle.py`**

Constantes, `run_id`, `iso_utc`, `jsonable`, `dumps`, `loads`, `envelope`, las seis funciones `path_*`,
`SCHEMAS` y `validate` según Interfaces. `jsonable` importa `json_default` de
`..core.utils.serialize`; `math.isnan`/`math.isinf` para los `float`. Docstring de módulo que describe
la disposición del paquete (el árbol de la spec) y el sobre.

- [ ] **Step 5: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_bundle_unit.py -v`
Expected: PASS (20 tests: 5 + 15 parametrizados).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/bundle.py tests/test_bundle_unit.py tests/fixtures/bundle
git commit -m "$(printf 'Add bundle layout, JSON envelope and schema validation\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 3: `BundleReader` y `BundleWriter` sobre `app.fs`

**Files:**
- Modify: `mtpy/lib/bundle.py`
- Test: `tests/test_bundle_unit.py`

**Interfaces:**
- Produces:

```python
class BundleReader:
    def __init__(self, fs: FileSystem, prefix: str = PREFIX): ...
    def path(self, name: str) -> str            # f'{prefix}/{name}'
    def exists(self, name: str) -> bool         # fs.exists(self.path(name))
    def read_json(self, name: str) -> dict      # loads(fs.read_bytes(...))
    def list_runs(self, scope: str) -> list[str]
```

  `list_runs`: nombres de `fs.listdir(path_runs(scope))` que casan con `RUN_ID_RE` y cuyo
  `path_part(scope, rid, 'headline')` existe, ordenados; `[]` si el directorio no existe
  (`FileNotFoundError` capturado; el `FileSystem` local y `s3fs` lo lanzan).

```python
class BundleWriter(BundleReader):
    def __init__(self, fs: FileSystem, prefix: str = PREFIX, clock: Optional[Callable[[], datetime]] = None): ...
    def now(self) -> datetime                   # clock() o datetime.now(timezone.utc)
    def begin_run(self, scope: str, run_id: str) -> None   # FileExistsError si exists(path_run(...))
    def write_json(self, name: str, schema: str, data, scope: Optional[str], run_id: Optional[str] = None,
                   mode: Optional[str] = None) -> str
    def write_csv(self, name: str, frame: pd.DataFrame) -> str
    def remove(self, name: str) -> None         # fs.remove(self.path(name)); recursivo en ambos fs
```

  `write_json`: `env = jsonable(envelope(schema, data, scope, run_id, mode, iso_utc(self.now())))`,
  `validate(schema, env)` (un `ValueError` sale sin escribir nada), `fs.write_bytes(dumps(env),
  self.path(name))`, devuelve `name`. `write_csv`: `frame.to_csv(index=False,
  lineterminator='\n').encode('utf-8')`.

- [ ] **Step 1: Añadir los tests a `tests/test_bundle_unit.py`**

```python
from mtpy.core.io import FileSystem


def fixed_clock():
    return datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_writer_round_trip_and_csv(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)), clock=fixed_clock)
    data = load_fixture('vote')['data']
    name = writer.write_json(bundle.path_part('es', '20261008-120000', 'vote', 'nowcast'), 'vote', data, 'es',
                             run_id='20261008-120000', mode='nowcast')
    assert name == 'runs/es/20261008-120000/nowcast/vote.json'
    assert (tmp_path / 'site' / 'v1' / name).exists()
    env = bundle.BundleReader(FileSystem(str(tmp_path))).read_json(name)
    assert env['generated_at'] == '2026-10-08T12:00:00Z' and env['data'] == data
    writer.write_csv(bundle.path_csv('es', '20261008-120000', 'vote', 'nowcast'), pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}))
    assert (tmp_path / 'site' / 'v1' / 'runs' / 'es' / '20261008-120000' / 'csv' / 'nowcast-vote.csv').read_bytes() == b'a,b\n1,x\n2,y\n'


def test_writer_validates_before_writing(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    with pytest.raises(ValueError, match='missing keys'):
        writer.write_json('runs/es/20261008-120000/nowcast/vote.json', 'vote', {'horizon': 0}, 'es', run_id='20261008-120000', mode='nowcast')
    assert not (tmp_path / 'site').exists()


def test_list_runs_keeps_only_complete_runs(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)), prefix=bundle.DRY_PREFIX)
    assert writer.list_runs('es') == []
    headline = load_fixture('headline')['data']
    for rid in ('20261008-120001', '20261007-090000'):
        writer.write_json(bundle.path_part('es', rid, 'headline'), 'headline', {**headline, 'run_id': rid}, 'es', run_id=rid)
    writer.write_json(bundle.path_part('es', '20261008-130000', 'meta'), 'meta', load_fixture('meta')['data'], 'es', run_id='20261008-130000')
    (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es' / 'history.json').write_text('{}')
    assert writer.list_runs('es') == ['20261007-090000', '20261008-120001']
    assert writer.exists('runs/es/20261008-130000/meta.json')


def test_begin_run_refuses_an_existing_run(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    writer.begin_run('es', '20261008-120000')
    writer.write_json(bundle.path_part('es', '20261008-120000', 'meta'), 'meta', load_fixture('meta')['data'], 'es', run_id='20261008-120000')
    with pytest.raises(FileExistsError):
        writer.begin_run('es', '20261008-120000')
    writer.remove(bundle.path_run('es', '20261008-120000'))
    writer.begin_run('es', '20261008-120000')
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_bundle_unit.py -v -k "writer or list_runs or begin_run"`
Expected: `AttributeError: module 'mtpy.lib.bundle' has no attribute 'BundleWriter'` (4 tests).

- [ ] **Step 3: Implementar las dos clases en `mtpy/lib/bundle.py`**

Según Interfaces; `from ..core.io import FileSystem` solo para la anotación de tipos.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_bundle_unit.py -v`
Expected: PASS (24 tests).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py tests/test_bundle_unit.py
git commit -m "$(printf 'Add BundleReader and BundleWriter over app.fs\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 4: Exportadores por modo sobre un `Simulator` sintético

**Files:**
- Create: `mtpy/lib/publish.py`, `tests/fakes.py`, `tests/test_publish_unit.py`

**Interfaces:**
- Consumes (métodos reales del `Simulator`, `mtpy/lib/simulator.py`): `vote_forecast()` (índice
  `party`; `pct, sd, lo, hi, horizon`), `summary(names=None)` (índice partido o bloque; las 15 columnas
  del contrato), `probabilities('vs')` (Series por bloque), `totals()` (Series de escaños enteros),
  `dist()` (n_sim × partidos, `int`), `unit_summary(region_id)` (`pct, pct_lo, pct_hi, seats,
  seats_mean, seats_lo, seats_hi, p_seats`), `scenario()` (int), `result(i)` (índice nombre de región,
  columnas partidos), `projection(names)` (índice fecha, columnas `(stat, serie)`), `horizon`, `when`,
  `n_seats`, `params['names']`, `params['regions']`, `default_region`, `region_names`, `reg_totals`.
- Produces en `tests/fakes.py`: `synthetic_forecaster(date_last='2026-10-01', date_fit_last='2026-10-05')
  -> SimpleNamespace` y `synthetic_simulator(mode='nowcast', n_sim=200, seed=0) -> Simulator` (Step 1);
  las Tareas 5-7 añaden `FakeSimulator` y `RecordingFileSystem`.
- Produces en `mtpy/lib/publish.py`:
  - `PCT_DECIMALS = 2`, `PROB_DECIMALS = 3`, `SEATS_DECIMALS = 1`.
  - `records(frame: pd.DataFrame, index_name: str = 'name') -> list[dict]`:
    `frame.rename_axis(index_name).reset_index().to_dict(orient='records')` (los NaN los limpia
    `bundle.jsonable` al escribir).
  - `round_cols(frame: pd.DataFrame, pct=(), prob=(), seats=()) -> pd.DataFrame`: redondea las columnas
    presentes de cada grupo a `PCT_DECIMALS`, `PROB_DECIMALS` y `SEATS_DECIMALS`.
  - `export_vote(sim) -> tuple[dict, pd.DataFrame]`, `export_summary(sim)`, `export_dist(sim)`,
    `export_districts(sim)`, `export_scenario(sim)`, `export_projection(sim)`: cada una devuelve
    `(data, frame)` con `data` según el contrato de la Tarea 2 y `frame` su gemelo CSV: `vote` → la tabla
    de filas; `summary` → las tres tablas concatenadas con una columna `group` (`parties`/`vs`/`blocks`)
    delante; `dist` → `sim.dist()`; `districts` → la tabla de filas; `scenario` → `result(i)` con
    columnas `region_id`, `region` y los partidos; `projection` → tabla larga `date, group, name, mean,
    lo, hi`. Redondeo: `pct*`/`mean`/`lo`/`hi`/`sd` a 2, `p_*` a 3, `seats_mean/median/lo/hi/min/max` a
    1, `seats` entero.
  - `headline_mode(sim) -> dict`: `{'parties': [{name, pct, lo, hi, seats, seats_lo, seats_hi,
    p_first}], 'p_majority': {bloque: p}}` combinando `vote_forecast()` (`pct, lo, hi`), `summary()`
    (`seats, seats_lo, seats_hi, p_first`) y `probabilities('vs')`, en el orden de `vote_forecast()`.

- [ ] **Step 1: Crear `tests/fakes.py` con el simulador sintético**

```python
"""Dobles de prueba de la publicación: un Simulator sintético que ejercita sus métodos reales de salida sin base."""
import types

import numpy as np
import pandas as pd

from mtpy.lib.simulator import Simulator

NAMES = ['PP', 'PSOE', 'VOX']
REGIONS = [0, 28, 8]  # total del ámbito, Madrid, Barcelona
SEATS = {0: 10, 28: 6, 8: 4}
BMAPS = {'main': NAMES, 'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']},
         'vs': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}}
COLORS = {'PP': '#1d84ce', 'PSOE': '#ef1c27', 'VOX': '#63be21'}


def synthetic_forecaster(date_last='2026-10-01', date_fit_last='2026-10-05'):
    """Salidas de un Forecaster ajustado: promedio y estadísticos diarios, sondeos, resultado anterior,
    efectos de casa y dispersión, con NaN donde el modelo real los deja."""
    dates = pd.date_range('2026-08-01', '2026-10-20', freq='D', name='date')
    forecast = pd.DataFrame({'PP': 40., 'PSOE': 30., 'VOX': 15.}, index=dates)
    forecast.loc[dates < '2026-09-01'] = np.nan  # antes del primer sondeo no hay promedio
    forecast['-'] = 100. - forecast[NAMES].sum(axis=1, min_count=1)
    fitted = (dates >= '2026-09-01') & (dates <= date_fit_last)
    fc_stat = pd.DataFrame({
        n: [{'mean': forecast.loc[d, n], 'cmin': forecast.loc[d, n] - 2., 'cmax': forecast.loc[d, n] + 2.,
             'err': 1., 'nobs': 40., 'neff': 30.} if ok else None for d, ok in zip(dates, fitted)]
        for n in NAMES
    }, index=dates)
    polls = pd.DataFrame({
        'date': pd.to_datetime(['2026-09-05', '2026-09-12', '2026-09-19', '2026-09-26', '2026-09-30', date_last]),
        'pollster_id': [1, 2, 1, 3, 2, 1], 'sponsor_id': [0, 0, 0, 5, 0, 0],
        'pollster': ['CIS', 'GAD3', 'CIS', '40dB', 'GAD3', 'CIS'], 'sponsor': [None, None, None, 'El País', None, None],
        'start_date': pd.to_datetime(['2026-09-01', '2026-09-08', '2026-09-15', '2026-09-22', '2026-09-26', '2026-09-28']),
        'end_date': pd.to_datetime(['2026-09-04', '2026-09-11', '2026-09-18', '2026-09-25', '2026-09-29', date_last]),
        'sample_size': [4000, 1000, 4000, 2000, 1000, 4000], 'mtype': ['cati', 'cati', 'cati', 'online', 'cati', 'cati'],
        'rating': [0.8, 0.9, 0.8, 0.7, 0.9, 0.8], 'weight': [1.2, 0.9, 1.2, 0.8, 0.9, 1.2],
        'PP': [41., 39.5, 40.2, 40., 39.8, 40.5], 'PSOE': [29., 31., 30.1, 30., 30.5, 29.5], 'VOX': [15., 14.5, np.nan, 15.5, 15., 14.8],
    }).set_index('date')
    polls['-'] = 100. - polls[NAMES].sum(axis=1, min_count=1)
    results = pd.DataFrame({'date': [pd.Timestamp('2023-07-23')], 'pollster': [None], 'PP': [33.1], 'PSOE': [31.7],
                            'VOX': [12.4], '-': [22.8]}).set_index('date')
    house_effects = pd.DataFrame({
        'pollster': ['CIS', 'CIS', 'GAD3', 'GAD3'], 'n': [3, 3, 2, 2], 'w': [1.2, 1.2, 0.9, 0.9], 'level': [40., 30., 40., 30.],
        'dev': [0.5, -0.4, -0.3, 0.6], 'dev_err': [0.3, 0.3, 0.4, 0.4], 'prior': [0.4, np.nan, -0.2, 0.5],
        'prior_err': [0.5, np.nan, 0.5, 0.5], 'effect': [0.45, -0.4, -0.25, 0.55], 'effect_err': [0.25, 0.3, 0.3, 0.3],
        'center': [0.1, 0.1, 0.1, 0.1],
    }, index=pd.MultiIndex.from_tuples([(1, 'PP'), (1, 'PSOE'), (2, 'PP'), (2, 'PSOE')], names=['pollster_id', 'name']))
    dispersion = pd.DataFrame({
        'pollster': ['CIS', 'GAD3', '40dB'], 'n': [3, 2, 1], 'ss_obs': [1.1, 0.8, np.nan], 'ss_exp': [1.0, 1.0, np.nan],
        'ratio_raw': [1.1, 0.8, np.nan], 'ratio': [1.05, 0.9, 1.], 'factor': [0.95, 1., 1.], 'herd_ratio': [np.nan, 0.85, np.nan],
    }, index=pd.Index([1, 2, 3], name='pollster_id'))
    return types.SimpleNamespace(
        forecast=forecast, fc_stat=fc_stat, fc_series_raw=polls, nfc_series=results, house_effects=house_effects,
        dispersion=dispersion, date_last=pd.Timestamp(date_last), date_fit_last=pd.Timestamp(date_fit_last),
        bmaps=BMAPS, colors=COLORS, names=NAMES,
    )


def synthetic_simulator(mode='nowcast', n_sim=200, seed=0):
    """Simulator reconstruido con `__new__` y arrays sintéticos (patrón `test_simulator_unit.py:438`):
    `summary`, `dist`, `unit_summary`, `vote_forecast`, `projection`... son los métodos reales."""
    rng = np.random.default_rng(seed)
    sim = Simulator.__new__(Simulator)
    sim.scope, sim.event_date, sim.mode, sim.seed, sim.verbose = 'es', '2026-11-29', mode, 42, 0
    sim.drange, sim.alpha = (6, None), 0.05
    sim.names, sim.default_region, sim.regions = NAMES, 0, REGIONS
    sim.region_names = {0: 'es', 28: 'Madrid', 8: 'Barcelona'}
    sim.reg_totals = pd.DataFrame({'votes': [1000, 600, 400], 'seats': [10, 6, 4]}, index=pd.Index(REGIONS, name='region_id'))
    sim.params = {'n_sim': n_sim, 'split': True, 'random': True, 'names': NAMES, 'regions': REGIONS,
                  'horizon': None, 'date_prior': 'historical'}
    sim.cols_forecast = ['mean', 'regional', 'err', 'nobs', 'error']
    sim.cols_frame = sim.cols_forecast + ['pct', 'pct_err', 'drift', 'std_err', 'rand', 'vpred']
    sim.cols_unit = ['prev_pct', 'vpred_pct']
    sim.forecast = pd.DataFrame({'mean': [40., 30., 15., 15.], 'regional': [0, 0, 0, 0], 'err': [1.5, 1.2, 1.0, np.nan],
                                 'nobs': [40., 40., 30., np.nan], 'error': [2.0, 1.8, 1.5, np.nan]}, index=NAMES + ['-'])
    sim.as_of, sim.deadline, sim.horizon_max = pd.Timestamp('2026-10-05'), pd.Timestamp('2026-11-29'), 55
    sim.ages = pd.Series({'PP': 40., 'PSOE': 40., 'VOX': 12.}, name='age')
    sim.v2err = sim.v2drift = sim.v2seats = None
    sim.composition_ratio, sim.composition_rho = 1.0, 0.
    sim.house_effects = sim.dispersion = sim.regional_noise = True
    sim.industry_bias, sim.composition, sim.smap = False, None, {'UP': [{'agg': ['UP', 'MP']}]}
    sim.parties = pd.DataFrame({'id': [1, 2, 3], 'name': NAMES, 'fullname': ['Partido Popular', 'PSOE', 'Vox'],
                                'color': [COLORS[n] for n in NAMES], 'block': ['Derecha', 'Izquierda', 'Derecha'],
                                'regional': [0, 0, 0]})
    sim.model = synthetic_forecaster()

    # Simulaciones: cuotas nacionales, cuotas y escaños por circunscripción; el total (región 0) es la suma
    vpred = sim.cols_frame.index('vpred')
    shares = rng.normal([40., 30., 15.], [2., 2., 1.5], size=(n_sim, 3)).clip(1.)
    others = (100. - shares.sum(axis=1)).clip(0.)
    frames = np.zeros((n_sim, 4, len(sim.cols_frame)))
    frames[:, :3, vpred], frames[:, 3, vpred] = shares, others
    units = np.zeros((n_sim, 3, 4, 2))
    results = np.zeros((n_sim, 3, 3), dtype=int)
    units[:, 0, :3, 1], units[:, 0, 3, 1] = shares, others
    for loc, region in enumerate(REGIONS[1:], start=1):
        local = (shares + rng.normal(0., 1., size=(n_sim, 3))).clip(1.)
        units[:, loc, :3, 1], units[:, loc, 3, 1] = local, (100. - local.sum(axis=1)).clip(0.)
        results[:, loc] = np.stack([rng.multinomial(SEATS[region], p / p.sum()) for p in local])
    results[:, 0] = results[:, 1:].sum(axis=1)
    sim.frames, sim.units, sim.results = frames.round(2), units.round(2), results
    sim.horizons = np.zeros(n_sim, dtype=int)
    return sim
```

- [ ] **Step 2: Crear `tests/test_publish_unit.py` con los tests de los exportadores por modo**

```python
"""Tests de `mtpy/lib/publish.py` sin base de datos: exportadores sobre el Simulator sintético de `tests/fakes.py`."""
import math

import numpy as np
import pandas as pd
import pytest

from mtpy.lib import bundle, publish
from tests.fakes import NAMES, REGIONS, synthetic_simulator


def valid(name, data, mode='nowcast'):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` y vuelve sin NaN."""
    env = bundle.jsonable(bundle.envelope(name, data, 'es', run_id='20261008-120000', mode=mode,
                                          generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(name, env)['data']


def test_export_vote_rows_are_rounded_and_ordered():
    data, frame = publish.export_vote(synthetic_simulator())
    data = valid('vote', data)
    assert data['horizon'] == 0 and data['when'] == '2026-10-05'
    assert [r['name'] for r in data['rows']] == NAMES
    row = data['rows'][0]
    assert row['pct'] == 40.0 and row['lo'] <= row['pct'] <= row['hi']
    assert all(round(r['pct'], 2) == r['pct'] for r in data['rows'])
    assert list(frame.columns) == ['name', 'pct', 'sd', 'lo', 'hi']


def test_export_summary_groups_and_probabilities():
    sim = synthetic_simulator()
    data, frame = publish.export_summary(sim)
    data = valid('summary', data)
    assert data['n_seats'] == 10 and data['majority'] == 6
    assert [r['name'] for r in data['parties']] == NAMES
    assert sum(data['totals'].values()) == 10
    assert set(data['p_majority']) == {'Derecha', 'Izquierda'}
    assert all(0 <= p <= 1 for p in data['p_majority'].values())
    assert set(data['parties'][0]) == {'name', 'pct', 'pct_mean', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_median',
                                       'seats_lo', 'seats_hi', 'seats_min', 'seats_max', 'p_seats', 'p_majority', 'p_first'}
    assert all(round(r['p_first'], 3) == r['p_first'] for r in data['parties'])
    assert sorted(frame['group'].unique()) == ['blocks', 'parties', 'vs']


def test_export_summary_writes_null_for_a_block_without_parties():
    sim = synthetic_simulator()
    sim.model.bmaps = {**sim.model.bmaps, 'vs': {'Derecha': ['PP', 'VOX'], 'Nadie': ['UP']}}
    data, _ = publish.export_summary(sim)
    data = valid('summary', data)
    nobody = [r for r in data['vs'] if r['name'] == 'Nadie'][0]
    assert nobody['p_first'] is None and nobody['seats_mean'] is None
    assert data['p_majority']['Nadie'] is None


def test_export_dist_rows_sum_the_chamber():
    data, frame = publish.export_dist(synthetic_simulator(n_sim=50))
    data = valid('dist', data)
    assert data['parties'] == NAMES and len(data['seats']) == 50
    assert all(sum(row) == 10 for row in data['seats'])
    assert frame.shape == (50, 3)


def test_export_districts_excludes_the_total_region():
    data, frame = publish.export_districts(synthetic_simulator())
    data = valid('districts', data)
    assert [r['id'] for r in data['regions']] == [28, 8]
    assert data['regions'][0] == {'id': 28, 'name': 'Madrid', 'seats': 6}
    assert len(data['rows']) == 2 * len(NAMES)
    assert set(data['rows'][0]) == {'region_id', 'region', 'name', 'pct', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_lo',
                                    'seats_hi', 'p_seats'}
    assert frame.shape[0] == 6


def test_export_scenario_is_a_real_simulation():
    sim = synthetic_simulator()
    data, frame = publish.export_scenario(sim)
    data = valid('scenario', data)
    assert data['simulation'] == sim.scenario()
    assert [r['region_id'] for r in data['rows']] == REGIONS and data['rows'][0]['region'] == 'es'
    assert data['rows'][0]['seats'] == [a + b for a, b in zip(data['rows'][1]['seats'], data['rows'][2]['seats'])]
    assert list(frame.columns) == ['region_id', 'region'] + NAMES


def test_export_projection_follows_the_mode():
    data, frame = publish.export_projection(synthetic_simulator())
    data = valid('projection', data)
    assert data['dates'] == ['2026-10-05'] and set(data['groups']) == {'parties', 'vs', 'blocks'}
    assert data['groups']['parties']['names'] == NAMES and len(data['groups']['parties']['mean']['PP']) == 1
    assert data['groups']['vs']['mean']['Derecha'][0] == pytest.approx(55.0, abs=0.01)
    data, frame = publish.export_projection(synthetic_simulator(mode='forecast'))
    data = valid('projection', data, mode='forecast')
    assert len(data['dates']) == 56 and data['dates'][-1] == '2026-11-29'
    assert list(frame.columns) == ['date', 'group', 'name', 'mean', 'lo', 'hi'] and frame.shape[0] == 56 * (3 + 2 + 2)


def test_headline_mode_combines_vote_and_seats():
    sim = synthetic_simulator(mode='forecast')
    head = bundle.jsonable(publish.headline_mode(sim))
    assert [p['name'] for p in head['parties']] == NAMES
    assert set(head['parties'][0]) == {'name', 'pct', 'lo', 'hi', 'seats', 'seats_lo', 'seats_hi', 'p_first'}
    assert sum(p['seats'] for p in head['parties']) == 10
    assert head['parties'][0]['hi'] > publish.export_vote(synthetic_simulator())[0]['rows'][0]['hi'] - 1e-9
    assert set(head['p_majority']) == {'Derecha', 'Izquierda'}
```

- [ ] **Step 3: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_publish_unit.py -v`
Expected: `ModuleNotFoundError: No module named 'mtpy.lib.publish'` en la recogida. (Si el fallo es
`tests.fakes`, confirmar que `tests/conftest.py` sigue insertando la raíz del repositorio en `sys.path`.)

- [ ] **Step 4: Implementar los exportadores por modo en `mtpy/lib/publish.py`**

Docstring de módulo (exportadores puros sobre un `Simulator` ya ejecutado; nada de estado). Imports:
`numpy`, `pandas`, `from . import bundle`. Las firmas de Interfaces. `export_districts` itera
`[r for r in sim.params['regions'] if r != sim.default_region]`; `export_scenario` alinea `region_id`
con las filas de `result(i)` por posición (`sim.params['regions']`); `export_projection` llama a
`sim.projection(names)` con `None`, `'vs'` y `'blocks'`, y las fechas salen de `index.strftime('%Y-%m-%d')`.

- [ ] **Step 5: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_publish_unit.py -v`
Expected: PASS (8 tests), sin `FutureWarning`.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/publish.py tests/fakes.py tests/test_publish_unit.py
git commit -m "$(printf 'Add per-mode exporters of the simulation to publish.py\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 5: Exportadores del ciclo, `export_meta`, `provenance` y `export_headline`

**Files:**
- Modify: `mtpy/lib/publish.py`
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Consumes (del `Forecaster`, `sim.model`): `forecast` (índice diario `date`, columnas partidos y `'-'`),
  `fc_stat` (mismo índice; celdas `dict` con `cmin`/`cmax` o `None`), `fc_series_raw` (índice `date`;
  columnas `pollster_id, sponsor_id, pollster, sponsor, start_date, end_date, sample_size, mtype,
  rating, weight` y partidos), `nfc_series` (fila del resultado anterior), `house_effects` (índice
  `(pollster_id, name)` o `None`), `dispersion` (índice `pollster_id` o `None`), `date_last`,
  `date_fit_last`, `bmaps`. Del `Simulator`: `fan()`, `ages`, `composition_ratio`, `v2drift` (`k`,
  `multiplier` o `None`), `clip_rate()`, `parties` (catálogo), `forecast['regional']`, `smap`, `regions`,
  `region_names`, `reg_totals`, `as_of`, `horizon_max`, `drange`, `alpha`, `seed`, `event_date`,
  `house_effects`, `dispersion`, `regional_noise`, `industry_bias`, `composition`.
- Produces:
  - `POLL_COLUMNS = ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'sample_size',
    'mtype', 'rating', 'weight']`, `HE_COLUMNS` y `DISP_COLUMNS` (las columnas de las filas del contrato).
  - `export_series(fc, names: list[str]) -> tuple[dict, pd.DataFrame]`: filas de `fc.forecast` hasta
    `fc.date_fit_last` (o la última si es `None`) y sin las anteriores al primer valor (`dropna(how='all')`
    sobre `names`); `lo`/`hi` de `cmin`/`cmax` de `fc.fc_stat` (`None` donde la celda no es `dict`);
    `frame` largo `date, name, mean, lo, hi`.
  - `export_polls(fc, names) -> tuple[dict, pd.DataFrame]`: `fc.fc_series_raw.reset_index()` con
    `POLL_COLUMNS + names` (pct a 2, `weight` a 3; las fechas las convierte `jsonable`); `results` =
    `fc.nfc_series.reset_index()[['date'] + names]`; `frame` = la tabla de sondeos.
  - `export_fan(sim)`, `export_house_effects(fc)`, `export_dispersion(fc) -> tuple[dict, pd.DataFrame]`:
    `fan` con `mean, sd, lo, hi` a 2 y `horizons` ordenados; con `house_effects`/`dispersion` `None`,
    `rows: []` y un `frame` vacío con las columnas del contrato; `herd_ratio` ausente → columna de `None`;
    redondeo a 2 (`house-effects`) y 3 (`dispersion`), salvo `n` y `w`.
  - `n_polls(fc) -> tuple[int, int]`: filas de `fc_series_raw` y `pollster_id.nunique()`.
  - `provenance(cwd: Optional[str] = None) -> dict`: `{'commit': sha corto de `git rev-parse --short
    HEAD` | `os.getenv('GIT_COMMIT')` | None, 'dirty': bool (`git status --porcelain` no vacío; `False`
    si git falla), 'versions': {'python': platform.python_version(), y
    `importlib.metadata.version(p)` para 'numpy', 'pandas', 'scipy', 'statsmodels'}}` con
    `subprocess.check_output(..., cwd=cwd, stderr=subprocess.DEVNULL)` dentro de `try/except Exception`.
  - `export_meta(sim, run_id: str, run_at: str, n_sim: int, max_fc: int, correctors: dict, seconds: dict,
    clip_rate: dict, db_polls: Optional[int], db_last_poll: Optional[str], prov: dict, freeze: bool) ->
    dict` con las claves de `SCHEMAS['meta']`: `parties` del catálogo `sim.parties` filtrado y ordenado
    por `sim.names` con `regional = int(sim.forecast.loc[name, 'regional'])`; `regions` de `sim.regions`
    con `region_names` y `reg_totals['seats']`; `drange` lista; `diagnostics` con `drift_k`/`multiplier`
    de `sim.v2drift` (o `None`), `ages` a 2 decimales, `composition = sim.composition_ratio`;
    `majority = n_seats // 2 + 1`; `n_polls`/`n_pollsters` de `n_polls(sim.model)`.
  - `export_headline(sim, run_id: str, run_at: str, nowcast: dict, forecast: dict) -> dict`:
    `{run_id, run_at, event_date, as_of, date_last, n_polls, nowcast, forecast}`.

- [ ] **Step 1: Añadir los tests a `tests/test_publish_unit.py`**

```python
def test_export_series_stops_at_the_last_fitted_day():
    sim = synthetic_simulator()
    data, frame = publish.export_series(sim.model, NAMES)
    data = valid('series', data, mode=None)
    assert data['dates'][0] == '2026-09-01' and data['dates'][-1] == '2026-10-05'
    assert data['parties'] == NAMES and len(data['mean']['PP']) == len(data['dates'])
    assert data['lo']['PP'][0] == 38.0 and data['hi']['PP'][0] == 42.0
    assert list(frame.columns) == ['date', 'name', 'mean', 'lo', 'hi'] and frame.shape[0] == 35 * 3


def test_export_series_without_statistics_gives_null_bounds():
    sim = synthetic_simulator()
    sim.model.fc_stat['VOX'] = None
    data, _ = publish.export_series(sim.model, NAMES)
    assert valid('series', data, mode=None)['lo']['VOX'] == [None] * 35


def test_export_polls_keeps_the_published_figures_and_the_previous_result():
    sim = synthetic_simulator()
    data, frame = publish.export_polls(sim.model, NAMES)
    data = valid('polls', data, mode=None)
    assert data['columns'] == publish.POLL_COLUMNS and len(data['polls']) == 6
    first = data['polls'][0]
    assert first['date'] == '2026-09-05' and first['pollster'] == 'CIS' and first['sponsor'] is None and first['PP'] == 41.0
    assert data['polls'][2]['VOX'] is None
    assert data['results'] == [{'date': '2023-07-23', 'PP': 33.1, 'PSOE': 31.7, 'VOX': 12.4}]
    assert list(frame.columns) == publish.POLL_COLUMNS + NAMES


def test_export_fan_house_effects_and_dispersion():
    sim = synthetic_simulator()
    fan, _ = publish.export_fan(sim)
    fan = valid('fan', fan, mode=None)
    assert fan['horizons'] == [0, 7, 14, 30, 55] and len(fan['rows']) == 5 * 3
    he, frame = publish.export_house_effects(sim.model)
    he = valid('house-effects', he, mode=None)
    assert len(he['rows']) == 4 and he['rows'][1]['prior'] is None and he['rows'][0]['name'] == 'PP'
    assert list(frame.columns) == publish.HE_COLUMNS
    disp, frame = publish.export_dispersion(sim.model)
    disp = valid('dispersion', disp, mode=None)
    assert [r['pollster_id'] for r in disp['rows']] == [1, 2, 3] and disp['rows'][0]['herd_ratio'] is None
    sim.model.house_effects = sim.model.dispersion = None
    assert publish.export_house_effects(sim.model)[0] == {'rows': []}
    assert list(publish.export_dispersion(sim.model)[1].columns) == publish.DISP_COLUMNS


def test_provenance_reads_git_or_the_image_env(monkeypatch):
    outputs = iter([b'abc1234\n', b' M mtpy/lib/publish.py\n'])
    monkeypatch.setattr(publish.subprocess, 'check_output', lambda *a, **k: next(outputs))
    prov = publish.provenance()
    assert prov['commit'] == 'abc1234' and prov['dirty'] is True
    assert set(prov['versions']) == {'python', 'numpy', 'pandas', 'scipy', 'statsmodels'}

    def boom(*a, **k):
        raise OSError('no git')
    monkeypatch.setattr(publish.subprocess, 'check_output', boom)
    monkeypatch.setenv('GIT_COMMIT', 'deadbee')
    prov = publish.provenance()
    assert prov['commit'] == 'deadbee' and prov['dirty'] is False


def test_export_meta_and_headline():
    sim = synthetic_simulator()
    prov = {'commit': 'abc1234', 'dirty': False, 'versions': {'python': '3.11.9'}}
    meta = publish.export_meta(sim, '20261008-120000', '2026-10-08T12:00:00Z', n_sim=200, max_fc=10,
                               correctors=publish.DEFAULT_CORRECTORS, seconds={'init': 1.0, 'fit': 2.0, 'nowcast': 3.0,
                               'forecast': 3.0, 'export': 1.0, 'total': 10.0}, clip_rate={'nowcast': 0.0, 'forecast': 0.01},
                               db_polls=7, db_last_poll='2026-10-01', prov=prov, freeze=False)
    meta = valid('meta', meta, mode=None)
    assert meta['run_id'] == '20261008-120000' and meta['commit'] == 'abc1234' and meta['dirty'] is False
    assert meta['as_of'] == '2026-10-05' and meta['date_last'] == '2026-10-01' and meta['horizon_max'] == 55
    assert meta['drange'] == [6, None] and meta['n_polls'] == 6 and meta['n_pollsters'] == 3
    assert meta['n_seats'] == 10 and meta['majority'] == 6
    assert meta['parties'][0] == {'name': 'PP', 'id': 1, 'fullname': 'Partido Popular', 'color': '#1d84ce', 'block': 'Derecha', 'regional': 0}
    assert meta['regions'][1] == {'id': 28, 'name': 'Madrid', 'seats': 6}
    assert meta['diagnostics'] == {'drift_k': None, 'multiplier': None, 'ages': {'PP': 40.0, 'PSOE': 40.0, 'VOX': 12.0},
                                   'composition': 1.0, 'clip_rate': {'nowcast': 0.0, 'forecast': 0.01}}
    assert meta['smap'] == {'UP': [{'agg': ['UP', 'MP']}]} and 'vs' in meta['bmaps']

    head = publish.export_headline(sim, '20261008-120000', '2026-10-08T12:00:00Z', {'parties': [], 'p_majority': {}},
                                   {'parties': [], 'p_majority': {}})
    head = valid('headline', head, mode=None)
    assert head['as_of'] == '2026-10-05' and head['n_polls'] == 6 and head['event_date'] == '2026-11-29'
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_publish_unit.py -v -k "series or polls or fan or provenance or meta"`
Expected: `AttributeError: module 'mtpy.lib.publish' has no attribute 'export_series'` y análogos (6 tests).

- [ ] **Step 3: Implementar en `mtpy/lib/publish.py`**

Según Interfaces (`import os, platform, subprocess`, `from importlib import metadata`). `DEFAULT_CORRECTORS`
(valores de Global Constraints) se define aquí porque `export_meta` lo recibe; la Tarea 6 lo reutiliza.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_publish_unit.py -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish.py tests/test_publish_unit.py
git commit -m "$(printf 'Add cycle exporters, meta and headline to publish.py\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 6: Orquestación: `publish_forecast`, `history`, `manifest`, `point`, `unpublish` y guarda LOREG

**Files:**
- Modify: `mtpy/lib/publish.py`, `tests/fakes.py`
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Consumes: `Simulator` (`mtpy/lib/simulator.py`), `get_scopes`, `get_next_event_date`
  (`mtpy/lib/data.py`), `Polls` (`mtpy/models/elections.py`), `BundleWriter` (Tarea 3), exportadores
  (Tareas 4-5).
- Produces en `tests/fakes.py`:

```python
class FakeSimulator:
    """Fábrica con la firma `Simulator(scope, event_date, **kwargs)`: devuelve el simulador sintético con
    `fit_forecast` y `run` inertes y anota las llamadas; con `fail`, lanza esa excepción al construir."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, scope, event_date, **kwargs):
        if self.fail is not None:
            raise self.fail
        sim = synthetic_simulator(mode=kwargs.get('mode', 'nowcast'))
        sim.scope, sim.event_date = scope, event_date
        self.calls.append(('init', scope, event_date, kwargs))

        def fit_forecast(**kw):
            self.calls.append(('fit_forecast', kw))
            return sim.forecast

        def run(**kw):
            self.calls.append(('run', sim.mode, kw))
            return sim

        sim.fit_forecast, sim.run = fit_forecast, run
        return sim


class RecordingFileSystem(FileSystem):
    """FileSystem local que anota el orden de las escrituras."""

    def __init__(self, path):
        super().__init__(path)
        self.written = []

    def write_bytes(self, content, name):
        self.written.append(name)
        return super().write_bytes(content, name)
```

  (`from mtpy.core.io import FileSystem` en `tests/fakes.py`.)
- Produces en `mtpy/lib/publish.py`:
  - `ATTRIBUTION = {'polls': 'Sondeos: Wikipedia, CC BY-SA 4.0', 'results': 'Origen de los datos:
    Ministerio del Interior', 'model': 'Modelo y curación: Luis Sancho'}` (texto provisional; pregunta
    abierta de la spec).
  - `class PublishRefused(Exception)`.
  - `loreg_guard(scope: str, event_date: str, today: Optional[str | date] = None, force: bool = False) ->
    bool`: `today` por defecto `datetime.now(timezone.utc).date()`; `in_window = scope == 'es' and 0 <=
    (event - today).days <= 5`; si `in_window and not force` → `PublishRefused('es {event_date}: inside
    the LOREG window (art. 69.7); pass force=true to publish')`; devuelve `in_window`.
  - `resolve_scopes(scopes: str | Sequence[str], catalogue: Optional[pd.DataFrame] = None) -> list[str]`:
    `'all'` → `['es']` más los índices del catálogo con `parent` no nulo, en su orden; otro `str` →
    `[scopes]`; secuencia → lista; cualquier ámbito fuera del índice → `ValueError('publish: unknown
    scopes [...]')` con los nombres; `catalogue` por defecto `get_scopes()`.
  - `next_event_date(scope: str) -> Optional[str]`: `get_next_event_date(scope)`.
  - `db_stats(scope: str, event_date: str) -> tuple[Optional[int], Optional[str]]`:
    `Polls().get_results(query=dict(filters=["event_scope = '{scope}'", "event_date = '{event_date}'"]),
    formatted=True)` → `(n_filas, max(date) 'YYYY-MM-DD')`, `(0, None)` sin filas.
  - `MODE_EXPORTERS = (('vote', export_vote), ('summary', export_summary), ('dist', export_dist),
    ('districts', export_districts), ('scenario', export_scenario), ('projection', export_projection))`.
  - `publish_mode(sim, writer: BundleWriter, scope: str, run_id: str) -> dict`: para cada exportador,
    `write_json(path_part(scope, run_id, part, sim.mode), part, data, scope, run_id, sim.mode)` y
    `write_csv(path_csv(scope, run_id, part, sim.mode), frame)`; devuelve `headline_mode(sim)`.
  - `publish_forecast(scope: str, writer: BundleWriter, run_id: str, event_date: str, n_sim: int = 1000,
    seed: int = 42, drange: int | tuple = 6, max_fc: int = 10, alpha: float = 0.05, correctors:
    Optional[dict] = None, freeze: bool = False, verbose: int = 0, simulator: Optional[Callable] = None,
    stats: Optional[Callable] = None, prov: Optional[Callable] = None) -> dict`. Orden: `writer.begin_run`
    → `run_at = iso_utc(writer.now())` → `sim = (simulator or Simulator)(scope=scope,
    event_date=event_date, drange=drange, alpha=alpha, seed=seed, mode='nowcast', verbose=verbose,
    path='.', **{**DEFAULT_CORRECTORS, **(correctors or {})})` → `sim.fit_forecast(names=sim.params['names'],
    max_fc=max_fc, fillna=True)` → `sim.run(split=True, random=True, n_sim=n_sim)` → `clip['nowcast'] =
    sim.clip_rate()` → `heads['nowcast'] = publish_mode(...)` → `sim.mode = 'forecast'` →
    `sim.run(split=True, random=True, n_sim=n_sim, horizon='deadline')` → `clip['forecast']` →
    `heads['forecast'] = publish_mode(...)` → `series`, `polls`, `fan`, `house-effects`, `dispersion`
    (JSON + CSV) → `(stats or db_stats)(scope, event_date)` → `meta` (con `seconds` medidos con
    `time.perf_counter()` por paso: `init`, `fit`, `nowcast`, `forecast`, `export`, `total`) →
    `headline = export_headline(...)` **el último fichero**. Devuelve `{'scope', 'entry':
    latest_entry(headline), 'seconds'}`.
  - `latest_entry(headline: dict) -> dict`: `{'latest': run_id, 'run_at', 'event_date', 'as_of',
    'date_last', 'n_polls'}`.
  - `rebuild_history(writer, scope) -> dict`: `{'scope': scope, 'runs': [read_json(headline)['data'] de
    cada run de writer.list_runs(scope)]}`; lo escribe en `path_history(scope)` con `run_id=None`.
  - `read_manifest(reader: BundleReader) -> dict`: el `data` existente o `{'contract': CONTRACT,
    'updated_at': None, 'scopes': {}, 'freeze': {'active': False, 'message': None}, 'attribution':
    ATTRIBUTION}`.
  - `update_manifest(writer, scopes: Optional[dict[str, dict]] = None, freeze: Optional[dict] = None) ->
    dict`: fusiona `scopes` sobre los existentes, sustituye `freeze` si se da (`ValueError` si no trae
    `active: bool`; `message` por defecto `None`), `updated_at = iso_utc(writer.now())`, escribe
    `path_manifest()` con `scope=None`; devuelve el `data`.
  - `point(writer, scope, run_id) -> dict`: `ValueError('{scope}: run {run_id} not found')` si no existe
    `path_part(scope, run_id, 'headline')`; `update_manifest(writer, {scope:
    latest_entry(headline['data'])})`.
  - `unpublish(writer, scope, run_id) -> dict`: `ValueError('{scope}: run {run_id} not found')` si no
    existe `path_run`; `writer.remove(path_run(...))`; `rebuild_history`; si
    `manifest.scopes[scope]['latest'] == run_id`,
    reapunta al último de `list_runs(scope)` o elimina la entrada del ámbito; devuelve `{'removed':
    run_id, 'latest': nuevo | None}`.

- [ ] **Step 1: Añadir los dobles a `tests/fakes.py` y los tests a `tests/test_publish_unit.py`**

```python
from datetime import date, datetime, timezone

from mtpy.core.io import FileSystem
from tests.fakes import FakeSimulator, RecordingFileSystem

RUN = '20261008-120000'


def fixed_clock():
    return datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def make_writer(tmp_path, fs=None):
    return bundle.BundleWriter(fs or FileSystem(str(tmp_path)), clock=fixed_clock)


def fake_stats(scope, event_date):
    return 7, '2026-10-01'


def fake_prov():
    return {'commit': 'abc1234', 'dirty': False, 'versions': {'python': '3.11.9'}}


def publish_es(tmp_path, run_id=RUN, fs=None, factory=None):
    writer = make_writer(tmp_path, fs)
    factory = factory or FakeSimulator()
    result = publish.publish_forecast('es', writer, run_id, '2026-11-29', n_sim=50, simulator=factory, stats=fake_stats, prov=fake_prov)
    return writer, factory, result


def test_publish_forecast_writes_the_whole_layout_in_order(tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    writer, factory, result = publish_es(tmp_path, fs=fs)
    root = 'site/v1/runs/es/' + RUN + '/'
    expected = {root + p + '.json' for p in bundle.RUN_PARTS} | {root + 'csv/' + p + '.csv' for p in bundle.RUN_PARTS[2:]}
    for mode in bundle.MODES:
        expected |= {root + mode + '/' + p + '.json' for p in bundle.MODE_PARTS}
        expected |= {root + 'csv/' + mode + '-' + p + '.csv' for p in bundle.MODE_PARTS}
    assert set(fs.written) == expected
    assert fs.written[-1] == root + 'headline.json' and fs.written[-2] == root + 'meta.json'
    assert [c[0] for c in factory.calls] == ['init', 'fit_forecast', 'run', 'run']
    assert factory.calls[0][3]['mode'] == 'nowcast' and factory.calls[0][3]['house_effects'] is True
    assert factory.calls[1][1] == {'names': NAMES, 'max_fc': 10, 'fillna': True}
    assert factory.calls[2][1:] == ('nowcast', {'split': True, 'random': True, 'n_sim': 50})
    assert factory.calls[3][1:] == ('forecast', {'split': True, 'random': True, 'n_sim': 50, 'horizon': 'deadline'})
    assert result['entry'] == {'latest': RUN, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
                               'date_last': '2026-10-01', 'n_polls': 6}
    assert set(result['seconds']) == {'init', 'fit', 'nowcast', 'forecast', 'export', 'total'}
    meta = writer.read_json(bundle.path_part('es', RUN, 'meta'))['data']
    assert meta['n_sim'] == 50 and meta['db_polls'] == 7 and meta['commit'] == 'abc1234' and meta['freeze'] is False
    assert set(meta['diagnostics']['clip_rate']) == {'nowcast', 'forecast'}
    for name in fs.written:
        if name.endswith('.json'):
            env = bundle.loads(fs.read_bytes(name))
            bundle.validate(env['schema'].split('@')[0], env)
    assert not writer.exists(bundle.path_history('es')) and not writer.exists(bundle.path_manifest())


def test_publish_forecast_refuses_to_rewrite_a_run(tmp_path):
    publish_es(tmp_path)
    factory = FakeSimulator()
    with pytest.raises(FileExistsError):
        publish_es(tmp_path, factory=factory)
    assert factory.calls == []


def test_history_is_rebuilt_from_complete_runs_and_is_idempotent(tmp_path):
    writer, _, _ = publish_es(tmp_path)
    publish_es(tmp_path, run_id='20261009-120000')
    writer.write_json(bundle.path_part('es', '20261010-120000', 'meta'), 'meta',
                      writer.read_json(bundle.path_part('es', RUN, 'meta'))['data'], 'es', run_id='20261010-120000')
    history = publish.rebuild_history(writer, 'es')
    assert [r['run_id'] for r in history['runs']] == [RUN, '20261009-120000']
    assert set(history['runs'][0]) >= {'nowcast', 'forecast', 'as_of', 'n_polls'}
    first = (tmp_path / 'site' / 'v1' / 'runs' / 'es' / 'history.json').read_bytes()
    publish.rebuild_history(writer, 'es')
    assert (tmp_path / 'site' / 'v1' / 'runs' / 'es' / 'history.json').read_bytes() == first


def test_manifest_merges_scopes_and_freeze(tmp_path):
    writer = make_writer(tmp_path)
    assert publish.read_manifest(writer)['scopes'] == {} and publish.read_manifest(writer)['freeze'] == {'active': False, 'message': None}
    publish.update_manifest(writer, {'es': {'latest': RUN}, 'es-md': {'latest': RUN}})
    data = publish.update_manifest(writer, {'es': {'latest': '20261009-120000'}})
    assert data['scopes'] == {'es': {'latest': '20261009-120000'}, 'es-md': {'latest': RUN}}
    assert data['updated_at'] == '2026-10-08T12:00:00Z' and data['attribution'] == publish.ATTRIBUTION
    data = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda electoral'})
    assert data['freeze'] == {'active': True, 'message': 'Veda electoral'} and len(data['scopes']) == 2
    assert publish.update_manifest(writer, freeze={'active': False})['freeze'] == {'active': False, 'message': None}
    with pytest.raises(ValueError, match='active'):
        publish.update_manifest(writer, freeze={'message': 'x'})
    bundle.validate('manifest', writer.read_json(bundle.path_manifest()))


def test_point_and_unpublish(tmp_path):
    writer, _, first = publish_es(tmp_path)
    _, _, second = publish_es(tmp_path, run_id='20261009-120000')
    publish.rebuild_history(writer, 'es')
    publish.update_manifest(writer, {'es': second['entry']})
    with pytest.raises(ValueError, match='not found'):
        publish.point(writer, 'es', '20261010-120000')
    assert publish.point(writer, 'es', RUN)['scopes']['es'] == first['entry']
    publish.update_manifest(writer, {'es': second['entry'], 'es-md': {'latest': RUN}})
    out = publish.unpublish(writer, 'es', '20261009-120000')
    assert out == {'removed': '20261009-120000', 'latest': RUN}
    assert not writer.exists(bundle.path_run('es', '20261009-120000'))
    manifest = publish.read_manifest(writer)
    assert manifest['scopes']['es'] == first['entry'] and manifest['scopes']['es-md'] == {'latest': RUN}
    assert [r['run_id'] for r in writer.read_json(bundle.path_history('es'))['data']['runs']] == [RUN]
    assert publish.unpublish(writer, 'es', RUN) == {'removed': RUN, 'latest': None}
    assert 'es' not in publish.read_manifest(writer)['scopes']
    assert writer.read_json(bundle.path_history('es'))['data']['runs'] == []
    with pytest.raises(ValueError, match='not found'):
        publish.unpublish(writer, 'es', RUN)


def test_loreg_guard_only_blocks_es_in_the_five_days_before_the_election():
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-23') is False
    assert publish.loreg_guard('es-md', '2026-11-29', today='2026-11-27') is False
    with pytest.raises(publish.PublishRefused, match='LOREG'):
        publish.loreg_guard('es', '2026-11-29', today=date(2026, 11, 24))
    with pytest.raises(publish.PublishRefused):
        publish.loreg_guard('es', '2026-11-29', today='2026-11-29')
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-28', force=True) is True
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-30') is False


def test_resolve_scopes_uses_the_catalogue():
    catalogue = pd.DataFrame({'parent': [None, 'es', 'es']}, index=pd.Index(['es', 'es-an', 'es-md'], name='scode'))
    assert publish.resolve_scopes('all', catalogue) == ['es', 'es-an', 'es-md']
    assert publish.resolve_scopes('es-md', catalogue) == ['es-md']
    assert publish.resolve_scopes(['es-md', 'es'], catalogue) == ['es-md', 'es']
    with pytest.raises(ValueError, match='es-xx'):
        publish.resolve_scopes(['es-xx'], catalogue)
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_publish_unit.py -v -k "publish_forecast or history or manifest or point or loreg or resolve"`
Expected: `ImportError: cannot import name 'FakeSimulator' from 'tests.fakes'` en la recogida; tras
añadir los dobles, `AttributeError: module 'mtpy.lib.publish' has no attribute 'publish_forecast'`
(7 tests).

- [ ] **Step 3: Implementar la orquestación en `mtpy/lib/publish.py`**

Según Interfaces. `from .simulator import Simulator`, `from .data import get_scopes,
get_next_event_date`, `from ..models.elections import Polls` (importaciones de módulo: `simulator.py`
ya es una dependencia de `mtpy.lib`). `import time` para `perf_counter`.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_bundle_unit.py -v`
Expected: PASS (21 + 24 tests).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish.py tests/fakes.py tests/test_publish_unit.py
git commit -m "$(printf 'Add publish_forecast, history, manifest, point and unpublish\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 7: Job `Publish`

**Files:**
- Create: `mtpy/jobs/Publish.py`, `tests/test_jobs_publish.py`

**Interfaces:**
- Consumes: `publish.resolve_scopes`, `publish.next_event_date`, `publish.loreg_guard`,
  `publish.publish_forecast`, `publish.update_manifest`, `publish.point`, `publish.unpublish`,
  `publish.PublishRefused` (siempre como atributos del módulo, `from ..lib import publish`, para que los
  tests los sustituyan con `monkeypatch`); `bundle.BundleWriter`, `bundle.run_id`, `bundle.PREFIX`,
  `bundle.DRY_PREFIX`; `Job` y `Job.alert` (`mtpy/core/worker.py`).
- Produces:

```python
class Publish(Job):
    WHAT = ('forecast', 'manifest', 'point', 'unpublish')
    DEFERRED = {'analysis': 4, 'event': 5, 'backtest': 5}

    def run(self, what=('forecast',), scopes=('es',), event_date=None, n_sim=1000, seed=42, drange=6, max_fc=10,
            alpha=0.05, correctors=None, freeze=None, run=None, dry_run=False, force=False, fs=None, today=None,
            verbose=0, **kwargs) -> dict
```

  Comportamiento: `what` `str` → `[what]`; un valor de `DEFERRED` → `ValueError('publish: "{what}"
  arrives in phase {n}')`; otro desconocido → `ValueError('publish: unknown what ...')`. `writer =
  BundleWriter(fs if fs is not None else self.app.fs, prefix=DRY_PREFIX if dry_run else PREFIX)`
  (`RuntimeError('publish: no file system configured (set S3_BUCKET or run with a local files/
  root)')` si no hay ninguno); `run_id = bundle.run_id()` una vez. Resultado `{scope: {'status':
  'published'|'skipped'|'refused'|'failed', ...}}`:
  - `forecast`: `for scope in publish.resolve_scopes(scopes)`: `date = event_date or
    publish.next_event_date(scope)`; `None` → `skipped` con `reason 'no upcoming event'`; `in_window =
    publish.loreg_guard(scope, date, today, force)` → `PublishRefused` → `refused` con `reason = str(e)`;
    `publish.publish_forecast(scope, writer, run_id, date, n_sim=n_sim, seed=seed, drange=drange,
    max_fc=max_fc, alpha=alpha, correctors=correctors, freeze=in_window, verbose=verbose)` →
    `published` con `run_id`, `entry`, `seconds`; `ValueError` → `skipped` con `reason = str(e)`; otra
    excepción → `failed` con `reason '{Type}: {e}'` y `self.app.logger.error(traceback)` si hay logger.
    Al acabar, si no es `dry_run` y hay publicados, `publish.update_manifest(writer, {scope: entry})`
    (los ámbitos fallidos conservan su entrada).
  - `manifest`: `publish.update_manifest(writer, freeze=freeze)`.
  - `point` / `unpublish`: `ValueError('publish: "run" is required')` sin `run`; por ámbito,
    `publish.point(writer, scope, run)` → `{'status': 'published', 'latest': run}` /
    `publish.unpublish(writer, scope, run)` → `{'status': 'published', **resultado}`; un `ValueError`
    deja el ámbito `failed`.
  - Resumen: una línea por ámbito (`'{scope}: published {run_id} (as_of {as_of}, {n_polls} polls) in
    {total:.0f} s'`, `'{scope}: skipped ({reason})'`, `'{scope}: refused ({reason})'`, `'{scope}: failed
    ({reason})'`) y una del manifest (`'manifest: {n} scopes, freeze {on|off}'` o `'manifest: not written
    (dry run)'`), impresas con `print` y, si `self.app.logger is not None`, enviadas con `self.alert('\n'.join(lines))`.
    Si algún ámbito está `failed`, al final `raise RuntimeError('publish: failed scopes: ...')` (código
    de salida 1 en `job.py`). Devuelve el dict de resultados.

- [ ] **Step 1: Crear `tests/test_jobs_publish.py`**

```python
"""Tests del job `publish` (`mtpy/jobs/Publish.py`) con la publicación y la base sustituidas."""
import types

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish


def entry(scope, run_id):
    return {'latest': run_id, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
            'date_last': '2026-10-01', 'n_polls': 6}


@pytest.fixture
def patched(monkeypatch):
    """`resolve_scopes`, `next_event_date` y `publish_forecast` sin base: `es` publica, `es-md` no tiene
    evento, `es-cb` no tiene sondeos y `es-ar` rompe."""
    calls = []

    def fake_forecast(scope, writer, run_id, event_date, **kwargs):
        calls.append((scope, run_id, event_date, kwargs))
        if scope == 'es-cb':
            raise ValueError('No polls for es-cb 2027-05-23: nothing to forecast')
        if scope == 'es-ar':
            raise KeyError('x')
        writer.write_json(bundle.path_part(scope, run_id, 'headline'), 'headline',
                          {**entry(scope, run_id), 'run_id': run_id, 'nowcast': {}, 'forecast': {}}, scope, run_id=run_id)
        return {'scope': scope, 'entry': entry(scope, run_id), 'seconds': {'total': 12.3}}

    monkeypatch.setattr(publish, 'resolve_scopes', lambda scopes, catalogue=None: ['es', 'es-md', 'es-cb', 'es-ar'] if scopes == 'all' else list(scopes))
    monkeypatch.setattr(publish, 'next_event_date', lambda scope: {'es': '2026-11-29', 'es-cb': '2027-05-23', 'es-ar': '2026-02-08'}.get(scope))
    monkeypatch.setattr(publish, 'publish_forecast', fake_forecast)
    return calls


def test_publish_isolates_scopes_and_keeps_previous_manifest_entries(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    publish.update_manifest(bundle.BundleWriter(fs), {'es-ar': entry('es-ar', '20261001-000000')})
    errors = []
    fresh_app.set('logger', types.SimpleNamespace(error=errors.append, info=lambda m: None))
    with pytest.raises(RuntimeError, match='es-ar'):
        Publish().run(what='forecast', scopes='all', fs=fs, n_sim=20)
    out = capsys.readouterr().out
    assert 'es: published' in out and 'es-md: skipped (no upcoming event)' in out
    assert 'es-cb: skipped (No polls' in out and "es-ar: failed (KeyError: 'x')" in out
    assert 'manifest: 2 scopes, freeze off' in out
    assert len(errors) == 1 and 'KeyError' in errors[0]
    assert [c[0] for c in patched] == ['es', 'es-cb', 'es-ar'] and patched[0][3]['n_sim'] == 20
    manifest = publish.read_manifest(bundle.BundleReader(fs))
    assert manifest['scopes']['es']['latest'] == patched[0][1] and manifest['scopes']['es-ar']['latest'] == '20261001-000000'


def test_dry_run_writes_apart_and_leaves_the_pointers_alone(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    result = Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)), dry_run=True)
    assert result['es']['status'] == 'published'
    assert (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es').exists()
    assert not (tmp_path / 'site').exists()


def test_loreg_window_refuses_es_unless_forced(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, today='2026-11-26')
    assert result['es']['status'] == 'refused' and 'LOREG' in result['es']['reason'] and patched == []
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, today='2026-11-26', force=True)
    assert result['es']['status'] == 'published' and patched[0][3]['freeze'] is True


def test_manifest_point_and_unpublish_actions(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    run_id = patched[0][1]
    Publish().run(what=['manifest'], fs=fs, freeze={'active': True, 'message': 'Veda'})
    assert publish.read_manifest(bundle.BundleReader(fs))['freeze']['active'] is True
    with pytest.raises(ValueError, match='run'):
        Publish().run(what=['point'], scopes=['es'], fs=fs)
    assert Publish().run(what=['point'], scopes=['es'], fs=fs, run=run_id)['es']['status'] == 'published'
    with pytest.raises(RuntimeError):
        Publish().run(what=['point'], scopes=['es'], fs=fs, run='20200101-000000')
    assert Publish().run(what=['unpublish'], scopes=['es'], fs=fs, run=run_id)['es'] == {'status': 'published', 'removed': run_id, 'latest': None}
    assert 'es' not in publish.read_manifest(bundle.BundleReader(fs))['scopes']


def test_deferred_and_unknown_what_are_rejected(fresh_app, tmp_path):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='phase 4'):
        Publish().run(what=['analysis'], fs=FileSystem(str(tmp_path)))
    with pytest.raises(ValueError, match='unknown'):
        Publish().run(what='nope', fs=FileSystem(str(tmp_path)))
    with pytest.raises(RuntimeError, match='file system'):
        Publish().run(what=['manifest'])
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_jobs_publish.py -v`
Expected: `ModuleNotFoundError: No module named 'mtpy.jobs.Publish'` (5 tests).

- [ ] **Step 3: Implementar `mtpy/jobs/Publish.py`**

Según Interfaces; patrón de `mtpy/jobs/CheckS3.py` (`from ..core.worker import Job`, `self.app.fs`,
`print` + logger opcional). Docstring de clase con los ejemplos de invocación de la spec ("Comando de
publicar"). `import traceback`.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_jobs_publish.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Comprobar la suite unitaria completa**

Run: `python -m pytest -m "not integration" -q`
Expected: 246 passed (192 + 4 + 24 + 21 + 5), sin warnings nuevos de `FutureWarning`.

- [ ] **Step 6: Commit**

```bash
git add mtpy/jobs/Publish.py tests/test_jobs_publish.py
git commit -m "$(printf 'Add publish job\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 8: Prueba de integración y primer paquete local de `es`

Precondición: base de datos accesible con el `.env` de la raíz (o `deploy/elections.env` exportado) y
`files/params.json`; si la fixture `app` se salta por falta de base, anotar el resultado y seguir con la
Tarea 9 (los pasos 4-6 son los de Luis en ese caso).

**Files:**
- Create: `tests/integration/test_publish.py`

**Interfaces:**
- Consumes: `publish.publish_forecast`, `bundle.BundleWriter`, `Simulator`, `data/params.json`.

- [ ] **Step 1: Crear `tests/integration/test_publish.py`**

```python
"""Tests de integración de la publicación: `publish_forecast('es')` con `n_sim=20` contra la base."""
import json
import os

import numpy as np
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish

pytestmark = pytest.mark.integration

EVENT = '2026-11-29'
RUNS = ('20261008-120000', '20261008-120001')


@pytest.fixture(scope='module')
def published(app, tmp_path_factory):
    """Dos publicaciones de `es` con la misma semilla en un paquete temporal; devuelve el escritor."""
    writer = bundle.BundleWriter(FileSystem(str(tmp_path_factory.mktemp('bundle'))))
    for run_id in RUNS:
        publish.publish_forecast('es', writer, run_id, EVENT, n_sim=20, seed=42)
    publish.rebuild_history(writer, 'es')
    return writer


def read(writer, part, mode=None, run_id=RUNS[0]):
    return writer.read_json(bundle.path_part('es', run_id, part, mode))['data']


def test_every_file_validates(published):
    root = os.path.join(published.fs.path, 'site', 'v1')
    count = 0
    for folder, _, files in os.walk(root):
        for name in files:
            if name.endswith('.json'):
                with open(os.path.join(folder, name), 'rb') as fh:
                    env = bundle.loads(fh.read())
                bundle.validate(env['schema'].split('@')[0], env)
                count += 1
    assert count == 2 * (len(bundle.RUN_PARTS) + 2 * len(bundle.MODE_PARTS)) + 1


def test_seats_sum_350_and_probabilities_are_bounded(published):
    for mode in bundle.MODES:
        dist = read(published, 'dist', mode)
        assert dist['n_seats'] == 350 and all(sum(row) == 350 for row in dist['seats']) and len(dist['seats']) == 20
        summary = read(published, 'summary', mode)
        assert sum(summary['totals'].values()) == 350
        assert all(0 <= r[k] <= 1 for r in summary['parties'] for k in ('p_seats', 'p_majority', 'p_first') if r[k] is not None)


def test_parties_belong_to_the_event(published, app):
    with open(os.path.join(app.datapath, 'params.json'), encoding='utf-8') as fh:
        event_parties = set(json.load(fh)['es'][EVENT]['parties']['event'])
    meta = read(published, 'meta')
    assert {p['name'] for p in meta['parties']} <= event_parties
    assert meta['n_seats'] == 350 and meta['event_date'] == EVENT and meta['n_polls'] > 0


def test_forecast_vote_matches_a_forecast_mode_simulator(published):
    from mtpy.lib.simulator import Simulator

    sim = Simulator(scope='es', event_date=EVENT, drange=6, alpha=0.05, seed=42, mode='forecast', verbose=0, path='.')
    sim.fit_forecast(names=sim.params['names'], max_fc=10, fillna=True)
    expected = sim.vote_forecast().round(2)
    rows = {r['name']: r for r in read(published, 'vote', 'forecast')['rows']}
    assert set(rows) == set(expected.index)
    for name, row in expected.iterrows():
        assert rows[name]['pct'] == pytest.approx(row['pct']) and rows[name]['hi'] == pytest.approx(row['hi'])
    assert read(published, 'vote', 'forecast')['horizon'] == sim.horizon_max


def test_same_seed_gives_the_same_dist_and_two_history_entries(published):
    first = read(published, 'dist', 'nowcast', RUNS[0])['seats']
    second = read(published, 'dist', 'nowcast', RUNS[1])['seats']
    assert np.array_equal(first, second)
    history = published.read_json(bundle.path_history('es'))['data']
    assert [r['run_id'] for r in history['runs']] == list(RUNS)
```

- [ ] **Step 2: Ejecutar la prueba de integración**

Run: `python -m pytest -m integration -k publish -v`
Expected: 5 PASS (unos 2-4 minutos: cuatro simulaciones de 20 pasadas y un `Simulator` extra). Si la
fixture `app` se salta (`Base de datos no disponible`), anotarlo en el resumen y dejar la ejecución a
Luis (Tareas de Luis).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_publish.py
git commit -m "$(printf 'Add publish integration test\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

- [ ] **Step 4: Ensayo general con `dry_run`**

Run: `python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}' && find files/site-dry -type f | sort`
Expected: línea `es: published <run_id> (as_of <fecha>, <n> polls) in <s> s` y `manifest: not written
(dry run)`; 7 JSON de ciclo, 12 JSON por modo y 17 CSV bajo `files/site-dry/v1/runs/es/<run_id>/`; sin
`files/site-dry/v1/manifest.json`. Borrar después `files/site-dry/` (está fuera de git).

- [ ] **Step 5: Primer paquete real en `files/site/v1` y medición local**

Run: `python job.py publish '{"what":["forecast"],"scopes":["es"]}' && python -c "import json; print(json.dumps(json.load(open('files/site/v1/manifest.json'))['data']['scopes'], indent=1))"`
Expected: `es: published ...` con `n_sim=1000`; el manifest apunta al `run_id` recién creado; anotar
los `seconds` de `files/site/v1/runs/es/<run_id>/meta.json` (`init`, `fit`, `nowcast`, `forecast`,
`export`, `total`) para la Tarea 9.

- [ ] **Step 6: Contraste con el notebook (opcional, mismo `n_sim`)**

Con `n_sim` igual al del notebook `PollsSimulations` (10000) y `mode='forecast'`, la tabla
`sim.summary()` del notebook y `forecast/summary.json → parties` deben coincidir en `pct`, `seats` y
`p_majority` para la misma semilla (42) y la misma base. Es la comprobación de "Verificación, fase 1" de
la spec; si no coincide, la diferencia está en los parámetros de la llamada, no en el paquete.

---

### Task 9: Documentación del contrato, runbook de publicación y cierre de la fase

**Files:**
- Create: `docs/web/contrato.md`
- Modify: `deploy/README.md`, `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` (añadir
  "Estado al cierre de la fase 1")

- [ ] **Step 1: Escribir `docs/web/contrato.md`**

En español: el árbol del paquete (`site/v1/...`), el sobre común, la tabla "Contrato de datos" de la
Tarea 2 (una sección por esquema con sus claves y su fuente en el modelo: `vote` ← `vote_forecast()`,
`summary` ← `summary()`/`summary('vs')`/`summary('blocks')`/`probabilities('vs')`/`totals()`, `dist` ←
`dist()`, `districts` ← `unit_summary(r)`, `scenario` ← `result(scenario())`, `projection` ←
`projection()`, `series` ← `Forecaster.forecast`/`fc_stat` hasta `date_fit_last`, `polls` ←
`fc_series_raw`/`nfc_series`, `fan` ← `fan()`, `house-effects` ← `Forecaster.house_effects`,
`dispersion` ← `Forecaster.dispersion`), las convenciones (redondeos, `null`, fechas, inmutabilidad de
los runs, `headline.json` último), `history.json` y `manifest.json`, y un apartado final "Rutas de la
API: fase 2". Enlazar desde el docstring de `bundle.py`.

- [ ] **Step 2: Añadir la sección "Publicar" a `deploy/README.md`**

Tras "Comprobar S3": los seis ejemplos de la spec ("Comando de publicar") con una línea de explicación
cada uno; parámetros y valores por defecto; desde el portátil contra la RDS y S3 (`set -a; .
deploy/elections.env; set +a; python job.py publish ...`); desde la imagen (`docker run --rm --env-file
deploy/elections.env -e RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}' elections-web:<tag>`,
JSON sin espacios); leer el manifest publicado sin `aws` CLI:

```
set -a; . deploy/elections.env; set +a
python -c "from mtpy import mtpy; app = mtpy.run(); print(app.fs.read_bytes('site/v1/manifest.json').decode())"
```

Y el procedimiento de retirada (`point` a un run anterior; `unpublish` de un run malo; las cachés de la
fase 2 caducan en ≤ 2 min) y la guarda LOREG (`force`). Cambiar el título a "Despliegue de la web
(fases 0-1)".

- [ ] **Step 3: Añadir "Estado al cierre de la fase 1" a la spec**

Tras "Estado al cierre de la fase 0": ficheros creados, número de tests, los `seconds` medidos en la
Tarea 8 (local y, cuando Luis lo ejecute, contra la RDS), decisiones de esta fase (`json_default` en
`mtpy/core/utils/serialize.py`; `run_id` compartido; `headline.json` último; `what` diferidos a las
fases 4-5; `ATTRIBUTION` provisional; `meta.freeze` = publicado dentro de la ventana LOREG) y lo que
queda para Luis (abajo).

- [ ] **Step 4: Comprobar que `pytest` sigue en verde y commit**

Run: `python -m pytest -m "not integration" -q`
Expected: 246 passed.

```bash
git add docs/web/contrato.md deploy/README.md docs/superpowers/specs/2026-10-07-web-publicacion-design.md
git commit -m "$(printf 'Document the bundle contract and the publish command\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Tareas de Luis (requieren `deploy/elections.env`; no bloquean las Tareas 1-9)

- [ ] Pendientes de la fase 0: permisos `s3:GetObject`, `ListBucket`, `PutObject` y `DeleteObject` del
  usuario de AWS sobre el bucket; `set -a; . deploy/elections.env; set +a; python job.py check_s3`
  (si falla, aplicar la regla de `deploy/README.md` antes de publicar a S3); confirmar el esquema
  `elections` en la RDS.
- [ ] Primera publicación real a S3 contra la RDS y medición de duración:
  `set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["forecast"],"scopes":["es"]}'`,
  leer `site/v1/manifest.json` con el comando del README y anotar los `seconds` de `meta.json` en la
  spec (estimación de la spec: `es` 1,5-2 min). Después `scopes: "all"` para medir los autonómicos
  (estimación 10-15 min) y ver qué ámbitos quedan `skipped` por falta de sondeos.
- [ ] Si la integración (Tarea 8, Step 2) se saltó por falta de base en la sesión de ejecución,
  ejecutarla: `python -m pytest -m integration -k publish -v`.
- [ ] Decidir el texto definitivo de `ATTRIBUTION` (`mtpy/lib/publish.py`) y la licencia de la
  curación propia (pregunta abierta de la spec).
