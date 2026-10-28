# Relleno automático de `params.json` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Una acción `params` del job `load` que reescribe, para cada elección de cada ámbito, `parties.event`,
`parties.polls`, `bmaps.main`, `bmaps.blocks` y `bmaps.vs` de `data/params.json`, derivados de la matriz del
ciclo y del bloque de cada partido, conservando el resto del fichero.

**Architecture:** La derivación se saca de `get_event_params` (`mtpy/lib/data.py`) a una función pura
`derive_event_params(df, parties, smap)` sobre la matriz de `get_event_dmat`, usada por el respaldo en caliente y
por la acción nueva. El fichero se lee y escribe con `read_params`/`write_params`/`format_params` (también en
`data.py`), con un formateador propio que respeta el estilo del fichero. La acción `load_params` vive en
`mtpy/lib/load.py` (función de ámbito con fábricas inyectables, como `load_polls`) y el mando en
`mtpy/jobs/Load.py`, que gana `'params'` en `WHAT`.

**Tech Stack:** Python 3.11, pandas, pytest (`python -m pytest`), PostgreSQL remoto sólo para la ejecución final.

**Spec:** `docs/superpowers/specs/2026-10-28-params-autofill-design.md`

## Global Constraints

- Nombres de código en inglés neutro; docstrings en inglés como los vecinos de `data.py` y `load.py`; textos de
  consola en inglés como las líneas de `load polls` (`es: polls loaded (...)`).
- Sólo se reescriben las cinco claves `parties.event`, `parties.polls`, `bmaps.main`, `bmaps.blocks`, `bmaps.vs`.
  Cualquier otra clave de la elección (`smap`, `threshold`, `threshold_scope`, `bmaps.max`, `bmaps.min`...) se
  conserva con su valor y su posición. Las elecciones y ámbitos no tocados quedan byte a byte como estaban salvo
  por el reformateo del fichero completo.
- Bloques: exactamente `('Derecha', 'Izquierda', 'Separatista', 'Regionalista')` del campo `block` de la tabla
  `parties`; `vs` son las entradas `Derecha` e `Izquierda` de `blocks`. Nunca entra el partido comodín de bloque
  `Otros` ni un partido sin bloque.
- `main`: partidos de `event` cuyo `count` de la matriz (fila `(0, 'count')`, porcentaje de sondeos) es `> 95`.
- Orden de los partidos: el de las columnas de la matriz de `get_event_dmat` (resultado descendente y después
  media de sondeos descendente). Ninguna lista se reordena.
- `elections` sin columnas en la matriz (ni sondeos ni resultado) se saltan: su entrada del fichero, si existe,
  no se toca.
- No se toca `data/sesgo-cis/params.json`, `data/cera/params.json`, ni los cuadernos.
- El fichero termina en una sola línea nueva, sangría de 4 espacios, `ensure_ascii=False`.
- Las pruebas no necesitan base de datos: `python -m pytest tests/test_data_unit.py tests/test_load_unit.py
  tests/test_jobs_load.py` (hoy 40 en verde) deben seguir pasando.

## Review Focus

1. Un partido del catálogo con `block` nulo o distinto de los cuatro (p. ej. `Otros`, id 528) aparece en `polls`
   pero en ningún bloque, sin excepción (Task 1, `test_derive_event_params_ignores_parties_without_a_block`).
2. Una regla de `smap` que no es un dict con `type` (`[]` de partido nuevo, lista de reglas, regla sin `names`)
   no rompe la derivación del `event` futuro (Task 1, `test_derive_event_params_future_election_drops_agg_names`).
3. Una elección sin sondeos pero con resultado (Asturias 2019) tiene `event` y `polls` iguales y `main` vacío,
   sin `KeyError` por la fila `count` (Task 1, `test_derive_event_params_result_only_election`).
4. Una elección presente en el fichero pero sin datos en la base no se borra ni se vacía (Task 3,
   `test_load_params_skips_an_election_without_matrix_and_keeps_its_entry`).
5. Un ámbito sin entrada en el fichero se crea con sus elecciones por fecha descendente (Task 3,
   `test_load_params_creates_a_missing_scope_with_dates_descending`).

---

### Task 1: Derivación pura en `data.py`

**Files:**
- Modify: `mtpy/lib/data.py:243-343` (`get_event_params`)
- Test: `tests/test_data_unit.py`

**Interfaces:**
- Produces:
  - `BLOCKS = ('Derecha', 'Izquierda', 'Separatista', 'Regionalista')` y `VS_BLOCKS = ('Derecha', 'Izquierda')`
    en `data.py`.
  - `derive_event_params(df: pd.DataFrame, parties: pd.DataFrame, smap: Optional[dict] = None) -> dict`:
    `{'parties': {'event': [...], 'polls': [...]}, 'bmaps': {'main': [...], 'blocks': {...}, 'vs': {...}}}`.
    `df` es la matriz de `get_event_dmat` (índice `(days, pollster)`, filas `(0, 'count')`, `(0, 'mean')`,
    `(0, 'std')`, `(0, 'result')`); `parties` tiene columnas `name` y `block`.
  - `agg_names(smap: Optional[dict]) -> list[str]`: nombres absorbidos por reglas `{'type': 'agg', 'names': [...]}`
    (una regla puede ser dict, lista de dicts o `[]`; sin `names` se ignora), sin duplicados, en orden de aparición.
  - `derive_scope_params(scope: str, event_dates: Optional[list[str]] = None, overrides: Optional[dict] = None)
    -> dict[str, dict]`: por fecha, `derive_event_params(df, parties, smap=(overrides or {}).get(dt, {}).get('smap'))
    | {'smap': {new party: []}}` (la lógica de partidos nuevos que hoy vive en `get_event_params`, intacta). Hace
    las lecturas de base (`get_event_dates`, `get_poll_series`, `get_event_series`, `get_parties`) que hoy hace
    `get_event_params`.
  - `get_event_params` queda como: lee el fichero si `path` no es `None`, llama a `derive_scope_params(scope,
    event_dates, overrides=params del ámbito)` y devuelve `{dt: derived | params[dt]}` (misma firma y mismo
    comportamiento de `is_single`).

- [ ] **Step 1: Write the failing tests** en `tests/test_data_unit.py`

```python
from mtpy.lib.data import agg_names, derive_event_params, get_event_dmat


def _matrix(polls_by_party, results_by_party, dates=('2023-07-23',)):
    """Matriz de un ciclo: `polls_by_party` {name: [valores por sondeo]} y `results_by_party` {name: pct o nan}."""
    names = list(dict.fromkeys(list(results_by_party) + list(polls_by_party)))
    n = max((len(v) for v in polls_by_party.values()), default=0)
    index = pd.MultiIndex.from_arrays([[dates[0]] * n, list(range(1, n + 1)), ['H{}'.format(i) for i in range(n)]],
                                      names=['event_date', 'days', 'pollster'])
    polls = pd.DataFrame({name: polls_by_party.get(name, [float('nan')] * n) for name in names}, index=index, dtype=float)
    events = pd.DataFrame({name: [results_by_party.get(name, float('nan'))] for name in names}, index=pd.Index([dates[0]], name='date'))
    parties = pd.DataFrame({'name': names})
    return get_event_dmat('es', dates[0], polls, events, parties)


PARTIES = pd.DataFrame({'name': ['PP', 'PSOE', 'VOX', 'SUMAR', 'Cs', 'ERC', 'PNV', '-', 'NEW'],
                        'block': ['Derecha', 'Izquierda', 'Derecha', 'Izquierda', 'Derecha', 'Separatista',
                                  'Regionalista', 'Otros', None]})


def test_derive_event_params_past_election():
    df = _matrix({'PP': [33, 34, 32, 33], 'PSOE': [30, 29, 31, 30], 'VOX': [12, 13, 12, 12], 'SUMAR': [12, 11, 12, 11],
                  'Cs': [1, float('nan'), float('nan'), float('nan')], 'ERC': [2, 2, 2, 2], 'PNV': [1, 1, 1, 1]},
                 {'PP': 33.1, 'PSOE': 31.7, 'VOX': 12.4, 'SUMAR': 12.3, 'ERC': 1.9, 'PNV': 1.1, 'Cs': 0.7})
    out = derive_event_params(df, PARTIES)
    assert out['parties']['event'] == ['PP', 'PSOE', 'VOX', 'SUMAR', 'ERC', 'PNV', 'Cs']
    assert out['parties']['polls'] == out['parties']['event']
    assert out['bmaps']['main'] == ['PP', 'PSOE', 'VOX', 'SUMAR', 'ERC', 'PNV']  # Cs está en 1 de 4 sondeos
    assert out['bmaps']['blocks'] == {'Derecha': ['PP', 'VOX', 'Cs'], 'Izquierda': ['PSOE', 'SUMAR'],
                                      'Separatista': ['ERC'], 'Regionalista': ['PNV']}
    assert out['bmaps']['vs'] == {'Derecha': ['PP', 'VOX', 'Cs'], 'Izquierda': ['PSOE', 'SUMAR']}
    assert list(out['bmaps']) == ['main', 'blocks', 'vs'] and list(out['parties']) == ['event', 'polls']


def test_derive_event_params_future_election_drops_agg_names():
    nan = float('nan')
    df = _matrix({'PP': [33, 34], 'PSOE': [30, 29], 'SUMAR': [12, 11], 'VOX': [12, nan], 'Cs': [1, 1]},
                 {'PP': nan, 'PSOE': nan, 'SUMAR': nan, 'VOX': nan, 'Cs': nan}, dates=('2026-11-29',))
    smap = {'PSOE': {'type': 'agg', 'names': ['SUMAR']}, 'VOX': [{'type': 'sub', 'names': ['PP']},
            {'type': 'agg', 'names': ['Cs']}], 'NEW': [], 'PP': {'type': 'sub'}}
    assert agg_names(smap) == ['SUMAR', 'Cs']
    out = derive_event_params(df, PARTIES, smap=smap)
    assert out['parties']['polls'] == ['PP', 'PSOE', 'VOX', 'SUMAR', 'Cs']  # por media del ciclo: VOX 12 > SUMAR 11,5
    assert out['parties']['event'] == ['PP', 'PSOE', 'VOX']
    assert out['bmaps']['main'] == ['PP', 'PSOE']  # VOX está en 1 de 2 sondeos
    assert out['bmaps']['blocks'] == {'Derecha': ['PP', 'VOX', 'Cs'], 'Izquierda': ['PSOE', 'SUMAR']}
    assert out['bmaps']['vs'] == out['bmaps']['blocks']


def test_derive_event_params_ignores_parties_without_a_block():
    df = _matrix({'PP': [33, 34], '-': [3, 4], 'NEW': [1, 1]}, {'PP': 33.0, '-': 3.5, 'NEW': 1.0})
    out = derive_event_params(df, PARTIES)
    assert out['parties']['polls'] == ['PP', '-', 'NEW']
    assert out['bmaps']['blocks'] == {'Derecha': ['PP']} and out['bmaps']['vs'] == {'Derecha': ['PP']}


def test_derive_event_params_result_only_election():
    df = _matrix({}, {'PSOE': 35.3, 'PP': 17.5})
    out = derive_event_params(df, PARTIES)
    assert out['parties'] == {'event': ['PSOE', 'PP'], 'polls': ['PSOE', 'PP']}
    assert out['bmaps'] == {'main': [], 'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE']},
                            'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}


def test_derive_event_params_empty_matrix():
    df = _matrix({}, {'PSOE': float('nan')}, dates=('2030-06-16',))
    assert derive_event_params(df, PARTIES) == {'parties': {'event': [], 'polls': []},
                                                'bmaps': {'main': [], 'blocks': {}, 'vs': {}}}
```

Nota: `_matrix` con `{}` de sondeos y un resultado `nan` debe dar una matriz sin columnas (es el caso de
`test_event_dmat_of_an_upcoming_event_without_polls_is_empty`); si `get_event_dmat` reindexa por `names` y deja
la columna, ajusta `_matrix` para pasar `polls` sin columnas en ese caso, no la función.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_data_unit.py -q -k derive`
Expected: FAIL con `ImportError: cannot import name 'agg_names'`.

- [ ] **Step 3: Implement `BLOCKS`, `VS_BLOCKS`, `agg_names`, `derive_event_params` y `derive_scope_params` en `mtpy/lib/data.py`, y reescribe `get_event_params` sobre ellas**

`derive_event_params`: `polls = df.columns.tolist()`; `result = df.loc[(0, 'result')].squeeze(axis=0)` si hay
columnas (con una sola columna `squeeze` da escalar: usa `df.loc[(0, 'result')].iloc[0]` o `.T`); `event` =
índices con resultado no nulo si hay alguno, si no `[p for p in polls if p not in agg_names(smap)]`; `main` =
`[p for p in event if count[p] > 95]` con `count = df.loc[(0, 'count')].iloc[0]` (NaN compara falso);
`block_of = dict(zip(parties['name'], parties['block']))`; `blocks = {b: [p for p in polls if block_of.get(p) == b]
for b in BLOCKS}` sin los vacíos; `vs = {b: blocks[b] for b in VS_BLOCKS if b in blocks}`. `derive_scope_params`
es el cuerpo actual de `get_event_params` (lecturas de base, bucle de fechas, `new_parties`) devolviendo
`{dt: derive_event_params(...) | {'smap': {p: [] for p in new_parties}}}`. Mantén los docstrings numpy.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_data_unit.py -q`
Expected: todo en verde (los tests previos del fichero también).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/data.py tests/test_data_unit.py
git commit -m "data: derivación pura de parties y bmaps de una elección (vs con el bloque completo)"
```

---

### Task 2: Lectura, formato y escritura de `params.json`

**Files:**
- Modify: `mtpy/lib/data.py` (junto a `get_event_params`)
- Test: `tests/test_data_unit.py`

**Interfaces:**
- Produces, en `data.py`:
  - `PARAMS_FILE = 'params.json'`.
  - `read_params() -> dict`: `json.loads(App.get_().data.read(PARAMS_FILE))`.
  - `format_params(data: dict) -> str`: el texto del fichero.
  - `write_params(data: dict) -> str`: `App.get_().data.write(format_params(data), PARAMS_FILE)`; devuelve la
    ruta.
- `get_event_params` pasa a leer con `read_params()`.

Reglas de `format_params` (reproducen el estilo actual del fichero):
- Sangría de 4 espacios; `json.dumps(value, ensure_ascii=False)` para escalares y para todo lo que se escribe
  en línea; `null`, `true`, `false` como JSON.
- Un dict se escribe expandido (una clave por línea, `"clave": valor`, coma entre entradas) salvo: los valores
  de las claves de `smap` (profundidad 4: ámbito → elección → `smap` → partido), que van en línea sean dict o
  lista; y un dict vacío, `{}`.
- Una lista va en línea (`["PP", "VOX"]`) salvo las de profundidad 4 bajo `parties` o `bmaps` (`event`, `polls`,
  `max`, `main` y cualquier otra lista directa de esas secciones), que van en bloque: `"main": [` + salto, los
  elementos unidos por `", "` repartidos en líneas de a lo sumo 100 columnas contadas con la sangría (sangría
  +4), coma al final de cada línea menos la última, y `]` en la sangría de la clave. Una lista vacía va `[]`.
- El texto termina en `}` y una única línea nueva.

- [ ] **Step 1: Write the failing tests** en `tests/test_data_unit.py`

```python
import json

from mtpy.lib.data import format_params, read_params, write_params

SAMPLE = {'es': {'2026-11-29': {
    'parties': {'event': ['PP', 'PSOE'], 'polls': ['PP', 'PSOE', 'SUMAR']},
    'bmaps': {'min': {'PP': 'PP', 'VOX+SALF': ['VOX', 'SALF']}, 'main': ['PP', 'PSOE'],
              'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE', 'SUMAR']}, 'vs': {'Derecha': ['PP']}},
    'smap': {'FA': {'type': 'agg', 'names': ['SUMAR']}, 'MP': [{'type': 'sub', 'names': ['UP']}, {'type': 'agg', 'names': ['C']}],
             'UP': []}},
    '2011-05-22': {'threshold': None, 'threshold_scope': 5}}}

EXPECTED = '''{
    "es": {
        "2026-11-29": {
            "parties": {
                "event": [
                    "PP", "PSOE"
                ],
                "polls": [
                    "PP", "PSOE", "SUMAR"
                ]
            },
            "bmaps": {
                "min": {
                    "PP": "PP",
                    "VOX+SALF": ["VOX", "SALF"]
                },
                "main": [
                    "PP", "PSOE"
                ],
                "blocks": {
                    "Derecha": ["PP"],
                    "Izquierda": ["PSOE", "SUMAR"]
                },
                "vs": {
                    "Derecha": ["PP"]
                }
            },
            "smap": {
                "FA": {"type": "agg", "names": ["SUMAR"]},
                "MP": [{"type": "sub", "names": ["UP"]}, {"type": "agg", "names": ["C"]}],
                "UP": []
            }
        },
        "2011-05-22": {
            "threshold": null,
            "threshold_scope": 5
        }
    }
}
'''


def test_format_params_reproduces_the_file_style():
    assert format_params(SAMPLE) == EXPECTED


def test_format_params_wraps_long_lists_at_100_columns():
    names = ['PARTIDO{:02d}'.format(i) for i in range(20)]
    text = format_params({'es': {'2023-07-23': {'parties': {'event': names, 'polls': []}}}})
    lines = text.split('\n')
    body = [line for line in lines if line.startswith(' ' * 20 + '"PARTIDO')]
    assert len(body) > 1 and all(len(line) <= 100 for line in body)
    assert all(line.endswith(',') for line in body[:-1]) and not body[-1].endswith(',')
    assert json.loads(text)['es']['2023-07-23']['parties'] == {'event': names, 'polls': []}


def test_format_params_round_trips_the_repository_file():
    data = json.loads(open('data/params.json', encoding='utf-8').read())
    assert json.loads(format_params(data)) == data


def test_read_and_write_params_use_the_data_file_system(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    fresh_app.set('data', FileSystem(str(tmp_path)))
    path = write_params(SAMPLE)
    assert path == str(tmp_path / 'params.json') and open(path, encoding='utf-8').read() == EXPECTED
    assert read_params() == SAMPLE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_data_unit.py -q -k params`
Expected: FAIL con `ImportError: cannot import name 'format_params'`.

- [ ] **Step 3: Implement `PARAMS_FILE`, `read_params`, `format_params`, `write_params` en `mtpy/lib/data.py`; `get_event_params` lee con `read_params()`**

`format_params` como función recursiva `_format(value, depth, indent, inline_rules)` que recibe la ruta de
claves para saber si está bajo `smap` (profundidad 3) o bajo `parties`/`bmaps` (profundidad 3) y aplica las
reglas de arriba. El salto de línea de cada bloque de lista: acumula elementos en la línea actual mientras
`len(indent + línea + ', ' + elemento + ',') <= 100`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_data_unit.py -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/data.py tests/test_data_unit.py
git commit -m "data: lectura, formato y escritura de params.json"
```

---

### Task 3: `load_params` en `mtpy/lib/load.py`

**Files:**
- Modify: `mtpy/lib/load.py`
- Test: `tests/test_load_unit.py`

**Interfaces:**
- Consumes: `derive_scope_params`, `read_params`, `write_params` de Task 1 y 2 (importados en `load.py` desde
  `.data`).
- Produces:
  - `PARAMS_STEPS = ('read', 'derive', 'save')` y `PARAMS_KEYS = (('parties', 'event'), ('parties', 'polls'),
    ('bmaps', 'main'), ('bmaps', 'blocks'), ('bmaps', 'vs'))`.
  - `NO_EVENTS = 'no elections for {}'` (razón de un ámbito sin elecciones en la base).
  - `merge_event_params(entry: dict, derived: dict) -> dict`: la entrada de la elección reescrita: claves
    `parties` y `bmaps` primero (en ese orden) y después las demás claves de `entry` en su orden; dentro de
    `parties`, `event` y `polls`; dentro de `bmaps`, primero las claves de `entry['bmaps']` que no son
    `main`/`blocks`/`vs` en su orden (`max`, `min`...) y luego `main`, `blocks`, `vs`. Con `entry` vacío devuelve
    `{'parties': ..., 'bmaps': ...}`.
  - `changed_keys(entry: dict, merged: dict) -> list[str]`: las rutas `'parties.event'`... de `PARAMS_KEYS` cuyo
    valor difiere (una clave ausente en `entry` cuenta como cambiada).
  - `load_params(scope: str, save: bool = True, verbose: int = 0, progress: Optional[Callable] = None,
    report: Optional[Callable] = None, deriver: Optional[Callable] = None, reader: Optional[Callable] = None,
    writer: Optional[Callable] = None) -> dict`:
    - `deriver(scope, overrides)` → `derive_scope_params(scope, overrides=overrides)` por defecto; `reader()` →
      `read_params`; `writer(data)` → `write_params`.
    - Fases con `_phase` (como `load_polls`): `read` (lee el fichero y toma `overrides = data.get(scope, {})`),
      `derive` (llama a `deriver`; una elección cuya derivación tiene `parties.polls == []` y
      `parties.event == []` se salta), `save` (sólo con `save`).
    - Si `deriver` devuelve `{}` (ámbito sin elecciones) lanza `ValueError(NO_EVENTS.format(scope))`.
    - Las elecciones nuevas del ámbito se insertan y el dict del ámbito se reordena por fecha descendente
      (`sorted(..., reverse=True)`); las elecciones del fichero sin derivación quedan tal cual. Un ámbito
      ausente del fichero se crea al final del dict de ámbitos.
    - `report(text, level)` con `level=1` por cada clave cambiada: `'{scope} {date}: {key} {old} -> {new}'`, con
      `old`/`new` como `json.dumps(..., ensure_ascii=False)` (`None` cuando falta).
    - Devuelve `{'status': 'loaded' | 'checked', 'n_events', 'n_changed', 'n_skipped', 'changed': {date: [keys]},
      'skipped': [dates], 'seconds': {fase: s, 'total': s}}`.

- [ ] **Step 1: Write the failing tests** en `tests/test_load_unit.py`

```python
from mtpy.lib.load import NO_EVENTS, changed_keys, load_params, merge_event_params

DERIVED = {'parties': {'event': ['PP', 'PSOE'], 'polls': ['PP', 'PSOE', 'SUMAR']},
           'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE', 'SUMAR']},
                     'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE', 'SUMAR']}}, 'smap': {'SUMAR': []}}
FILE = {'es': {'2026-11-29': {'parties': {'event': ['PP'], 'polls': ['PP']},
                              'bmaps': {'max': ['PP'], 'min': {'PP': 'PP'}, 'main': ['PP', 'PSOE'], 'blocks': {}, 'vs': {}},
                              'smap': {'FA': {'type': 'agg', 'names': ['SUMAR']}}},
               '2011-05-22': {'threshold': None, 'threshold_scope': 5}},
        'es-pv': {'2024-04-21': {'smap': {}}}}


class FakeParamsStore:
    def __init__(self, data):
        self.data = json.loads(json.dumps(data))
        self.written = None

    def read(self):
        return json.loads(json.dumps(self.data))

    def write(self, data):
        self.written = data


def test_merge_event_params_keeps_manual_keys_and_their_order():
    merged = merge_event_params(FILE['es']['2026-11-29'], DERIVED)
    assert list(merged) == ['parties', 'bmaps', 'smap']
    assert list(merged['bmaps']) == ['max', 'min', 'main', 'blocks', 'vs']
    assert merged['bmaps']['max'] == ['PP'] and merged['smap'] == FILE['es']['2026-11-29']['smap']
    assert merged['parties'] == DERIVED['parties'] and merged['bmaps']['vs'] == DERIVED['bmaps']['vs']
    assert list(merge_event_params({'threshold': 3}, DERIVED)) == ['parties', 'bmaps', 'threshold']
    assert changed_keys(FILE['es']['2026-11-29'], merged) == ['parties.event', 'parties.polls', 'bmaps.blocks', 'bmaps.vs']
    assert changed_keys({}, merged) == ['parties.event', 'parties.polls', 'bmaps.main', 'bmaps.blocks', 'bmaps.vs']


def test_load_params_rewrites_the_scope_and_reports_each_change():
    store = FakeParamsStore(FILE)
    seen = []
    derived = {'2026-11-29': DERIVED, '2023-07-23': DERIVED}

    def deriver(scope, overrides):
        seen.append((scope, overrides))
        return derived

    out = load_params('es', deriver=deriver, reader=store.read, writer=store.write,
                      report=lambda text, level=1: seen.append((level, text)))
    assert seen[0] == ('es', FILE['es'])
    assert out['status'] == 'loaded' and out['n_events'] == 2 and out['n_changed'] == 2 and out['n_skipped'] == 0
    assert out['changed'] == {'2026-11-29': ['parties.event', 'parties.polls', 'bmaps.blocks', 'bmaps.vs'],
                              '2023-07-23': ['parties.event', 'parties.polls', 'bmaps.main', 'bmaps.blocks', 'bmaps.vs']}
    assert set(out['seconds']) == {'read', 'derive', 'save', 'total'}
    assert list(store.written['es']) == ['2026-11-29', '2023-07-23', '2011-05-22']
    assert store.written['es']['2011-05-22'] == FILE['es']['2011-05-22'] and store.written['es-pv'] == FILE['es-pv']
    assert store.written['es']['2023-07-23'] == {'parties': DERIVED['parties'], 'bmaps': DERIVED['bmaps']}
    assert (1, 'es 2023-07-23: bmaps.main null -> ["PP", "PSOE"]') in seen
    assert (1, 'es 2026-11-29: parties.event ["PP"] -> ["PP", "PSOE"]') in seen


def test_load_params_dry_run_writes_nothing():
    store = FakeParamsStore(FILE)
    out = load_params('es', save=False, deriver=lambda scope, overrides: {'2026-11-29': DERIVED},
                      reader=store.read, writer=store.write)
    assert out['status'] == 'checked' and store.written is None and 'save' not in out['seconds']


def test_load_params_skips_an_election_without_matrix_and_keeps_its_entry():
    store = FakeParamsStore(FILE)
    empty = {'parties': {'event': [], 'polls': []}, 'bmaps': {'main': [], 'blocks': {}, 'vs': {}}, 'smap': {}}
    out = load_params('es', deriver=lambda scope, overrides: {'2026-11-29': empty, '2011-05-22': empty},
                      reader=store.read, writer=store.write)
    assert out['n_skipped'] == 2 and out['skipped'] == ['2026-11-29', '2011-05-22'] and out['n_changed'] == 0
    assert store.written['es'] == FILE['es']


def test_load_params_creates_a_missing_scope_with_dates_descending():
    store = FakeParamsStore(FILE)
    load_params('es-md', deriver=lambda scope, overrides: {'2023-05-28': DERIVED, '2027-05-23': DERIVED},
                reader=store.read, writer=store.write)
    assert list(store.written) == ['es', 'es-pv', 'es-md']
    assert list(store.written['es-md']) == ['2027-05-23', '2023-05-28']
    assert store.written['es-md']['2027-05-23'] == {'parties': DERIVED['parties'], 'bmaps': DERIVED['bmaps']}


def test_load_params_fails_without_elections():
    with pytest.raises(ValueError, match=NO_EVENTS.format('es-xx')):
        load_params('es-xx', deriver=lambda scope, overrides: {}, reader=lambda: {}, writer=lambda data: None)
```

Comprueba que `test_load_unit.py` ya importa `json` y `pytest`; añádelos si no.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_load_unit.py -q -k params`
Expected: FAIL con `ImportError: cannot import name 'NO_EVENTS'`.

- [ ] **Step 3: Implement `PARAMS_STEPS`, `PARAMS_KEYS`, `NO_EVENTS`, `merge_event_params`, `changed_keys` y `load_params` en `mtpy/lib/load.py`**

Sigue la estructura de `load_polls`: `seconds = {}`, cada fase con `_phase(...)` y `progress(step, seconds)` si
hay `progress`; `report` sólo si no es `None`. La derivación de una elección se recorta a `parties` y `bmaps`
antes de `merge_event_params` (el `smap` derivado de partidos nuevos no se escribe: lo manual manda).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_load_unit.py -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/load.py tests/test_load_unit.py
git commit -m "load: load_params reescribe parties y bmaps de cada elección en params.json"
```

---

### Task 4: Acción `params` del job `load`

**Files:**
- Modify: `mtpy/jobs/Load.py`
- Test: `tests/test_jobs_load.py`

**Interfaces:**
- Consumes: `load_params`, `NO_EVENTS`, `PARAMS_STEPS` de Task 3.
- Produces:
  - `WHAT = ('polls', 'params')`; `STEPS` gana `'derive': 'derivación'` (las claves `read` y `save` ya existen).
  - `run(..., params_loader=None, ...)`: fábrica inyectable de `load_params` (la real por defecto). Las acciones
    se ejecutan en el orden de `whats`, cada una con su cabecera (`load params: 2 ámbitos · dry run`) y su bucle
    de ámbitos; el resumen final imprime las líneas de todas las acciones y el `RuntimeError` final nombra
    `scope (action)` como hoy. Valor de retorno: con una sola acción, `{scope: result}` como hoy (los tests
    actuales de `polls` no cambian); con varias, `{scope: {action: result}}`.
  - `_scope_params(scope, dry_run, verbose, progress, params_loader, skip_missing=False) -> dict`: como `_scope`,
    con `action='params'`; `event_date`, `overwrite`, `force` y `years` no le afectan; un `ValueError` con razón
    `NO_EVENTS.format(scope)` pasa a `skipped` con `skip_missing`.
  - `_line_params(scope, result) -> str`: `'{scope}: params loaded ({n_events} elections, {n_changed} changed,
    {n_skipped} skipped) in {total:.0f} s'`; en seco `params checked (..., dry run)`; fallos y saltos como
    `_line`.

- [ ] **Step 1: Write the failing tests** en `tests/test_jobs_load.py`

```python
PARAMS_RESULT = {'status': 'loaded', 'n_events': 3, 'n_changed': 2, 'n_skipped': 1, 'changed': {}, 'skipped': [],
                 'seconds': {'read': 0.1, 'derive': 0.2, 'save': 0.1, 'total': 0.4}}


class FakeParamsLoader:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def __call__(self, scope, save=True, verbose=0, progress=None, report=None, **kwargs):
        self.calls.append((scope, save, verbose))
        if scope in self.fail:
            raise self.fail[scope]
        if report is not None:
            report('{} 2023-07-23: bmaps.vs null -> {{}}'.format(scope), 1)
        return dict(PARAMS_RESULT, status='loaded' if save else 'checked')


def test_load_params_runs_every_scope_and_prints_the_summary(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader()
    out = Load().run(what='params', scopes='all', params_loader=loader, verbose=1)
    text = capsys.readouterr().out
    assert [call[0] for call in loader.calls] == ['es', 'es-md'] and all(call[1] for call in loader.calls)
    assert 'load params: 2 ámbitos' in text
    assert 'es: params loaded (3 elections, 2 changed, 1 skipped) in 0 s' in text
    assert 'es 2023-07-23: bmaps.vs null -> {}' in text
    assert out['es']['status'] == 'loaded'


def test_load_params_dry_run_checks_and_flags_the_header(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader()
    Load().run(what='params', scopes=['es'], dry_run=True, params_loader=loader)
    text = capsys.readouterr().out
    assert loader.calls == [('es', False, 0)]
    assert 'load params: 1 ámbito · dry run' in text and 'es: params checked (3 elections, 2 changed, 1 skipped, dry run)' in text


def test_load_params_scope_without_elections_is_skipped_with_all_and_fails_when_named(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader(fail={'es-md': ValueError(module.NO_EVENTS.format('es-md'))})
    out = Load().run(what='params', scopes='all', params_loader=loader)
    assert out['es-md']['status'] == 'skipped' and 'es-md: params skipped (no elections for es-md)' in capsys.readouterr().out
    with pytest.raises(RuntimeError, match=r'es-md \(params\)'):
        Load().run(what='params', scopes=['es-md'], params_loader=loader)


def test_load_params_isolates_an_exception(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader(fail={'es': KeyError('boom')})
    with pytest.raises(RuntimeError, match=r'es \(params\)'):
        Load().run(what='params', scopes='all', params_loader=loader)
    text = capsys.readouterr().out
    assert "es: params failed (KeyError: 'boom')" in text and 'es-md: params loaded' in text


def test_load_both_actions_run_polls_first(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader()
    run(FakeLoader(), what=['polls', 'params'], scopes=['es'], params_loader=loader)
    text = capsys.readouterr().out
    assert text.index('load polls: 1 ámbito') < text.index('load params: 1 ámbito') and loader.calls == [('es', True, 0)]
```

El test `test_load_unknown_action` existente sigue valiendo: comprueba que su mensaje esperado liste
`polls, params` si fija la lista; ajústalo si hace falta.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_jobs_load.py -q -k params`
Expected: FAIL con `ValueError: load: unknown what "params"`.

- [ ] **Step 3: Implement the action in `mtpy/jobs/Load.py`**

Extrae el bucle de ámbitos de `run` a `_run_polls(targets, ...)` y añade `_run_params(targets, dry_run, verbose,
params_loader, skip_missing)`, cada uno devolviendo `(results, failures, lines)`; `run` encadena las acciones
pedidas, imprime todas las líneas, alerta y lanza el `RuntimeError` conjunto. Actualiza el docstring de la clase
(párrafo de `params`: qué reescribe, qué conserva, `dry_run` enseña cada clave que cambiaría) y el ejemplo de
línea de comandos `python job.py load '{"what":"params","scopes":"all"}'`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs_load.py tests/test_load_unit.py tests/test_data_unit.py -q`
Expected: todo en verde.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q -x --ignore=tests/integration`
Expected: todo en verde (hoy 1391 tests más los nuevos).

- [ ] **Step 6: Commit**

```bash
git add mtpy/jobs/Load.py tests/test_jobs_load.py
git commit -m "jobs: acción params del job load"
```

---

### Task 5: Ejecución real y revisión del diff

**Files:**
- Modify: `data/params.json` (lo reescribe el job; Luis lo revisa y lo integra él)
- Modify: `docs/superpowers/specs/2026-10-28-params-autofill-design.md` (estado)

- [ ] **Step 1: Dry run contra la base**

Run: `python job.py load '{"what":"params","scopes":"all","dry_run":true,"verbose":1}' 2>&1 | grep -v Warning`
Expected: cabecera `load params: 14 ámbitos · dry run`, una línea por clave que cambia y una línea
`params checked` por ámbito; ningún `failed`. Si algún ámbito falla, arregla la causa antes de seguir (no es
un fallo de datos aceptable: todos tienen elecciones).

- [ ] **Step 2: Ejecución real**

Run: `python job.py load '{"what":"params","scopes":"all","verbose":1}' 2>&1 | grep -v Warning`
Expected: `params loaded` en los 14 ámbitos; `git diff --stat data/params.json` muestra el fichero cambiado.

- [ ] **Step 3: Comprobaciones del resultado**

Run: `python -m pytest tests/test_data_files.py -q` (el fichero sigue siendo válido) y
`python - <<'EOF'` que carga `data/params.json` y comprueba, para `es` `2026-11-29`, que `smap`, `bmaps.max` y
`bmaps.min` son idénticos a `git show HEAD:data/params.json`, y que en todo el fichero `vs[b] == blocks[b]`
para `Derecha` e `Izquierda`.
Expected: sin diferencias en las claves manuales; igualdad `vs`/`blocks` en todas las elecciones.

- [ ] **Step 4: Informe para Luis (sin commit del fichero de datos)**

Resume para Luis, con cifras: elecciones escritas por ámbito, y para `es` 2026 y 2023 las diferencias frente a
sus listas manuales (al menos: `main` de 2026 pasa de 6 a 4 partidos porque UP y SALF no llegan al 95 % de los
sondeos; `polls`/`event` de 2026 ganan AC y ADELANTE y 2023 gana EV y TE si la base los trae así). No hacer
`git commit` de `data/params.json`: lo revisa e integra él. Cierra el estado del spec («construido en dev el
2026-10-28, fichero pendiente de revisión de Luis») y haz commit sólo del spec.

```bash
git add docs/superpowers/specs/2026-10-28-params-autofill-design.md
git commit -m "spec: relleno automático de params.json construido"
```
