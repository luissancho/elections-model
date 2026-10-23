# Compromís aparte y Podemos sin Cataluña: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el modelo del 29-N simule a Compromís en sus tres provincias con la base de Sumar de 2023, no le fabrique a Podemos una base en Cataluña sin vaciar la de FA, y aplique en cada carga las correcciones manuales de sondeos guardadas en un CSV curado.

**Architecture:** La aplicación de las reglas del `smap` sobre las bases provinciales sale de `Simulator.build_umat` a un método estático puro, donde una regla `sub` con `regions` hereda solo en las provincias que quedan y recalcula las filas nacionales. El loader de Wikipedia lee `data/polls/{scope}/{event_date}.csv` al final de `build_series` y lo aplica con una función pura sobre los frames `polls` y `results`. `params.json` recupera el bloque de Compromís y añade la exclusión catalana de Podemos.

**Tech Stack:** Python 3.11 (pyenv `3.11.9`), pandas, numpy, pytest (`python -m pytest`), PostgreSQL remoto solo para la comprobación final.

**Spec:** `docs/superpowers/specs/2026-10-23-compromis-podemos-reglas-design.md`

## Global Constraints

- `pytest.ini` convierte `FutureWarning` en error: nada de `pd.concat` con frames vacíos ni `groupby` sin `observed`.
- Identificadores en inglés; textos y mensajes públicos en español; mensajes de error y docstrings siguen el estilo del fichero donde viven (inglés en `mtpy/lib`).
- Luis hace commits en paralelo en el mismo árbol: `git status` y `git log -3` antes de cada commit; nunca reescribir sus commits ni revertir ficheros que escriben sus jobs (`files/`).
- Cada commit termina con la línea `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Los tests unitarios no tocan la base ni la red (`bare_loader` de `tests/test_loader_unit.py`; funciones puras en `tests/test_simulator_unit.py`).
- Valores pinados por el diseño: Compromís `agg SUMAR` en `["3", "12", "46"]`; Podemos `sub FA` con `{"exclude": ["8", "17", "25", "43"]}`; CSV inicial Celeste Tel 2026-09-21 / Onda Cero (SUMAR 5.1; COMPROMIS 1.3, 3 escaños) y SocioMétrica 2026-10-11 / El Español (SUMAR 4.9; COMPROMIS 0.9, 3 escaños).

## Review Focus

1. Un sondeo sin patrocinador (`sponsor_id` 0) y una fila del CSV con `sponsor` vacío deben casar; si no, la corrección se reporta como no encontrada y Sumar sigue contando a Compromís. Test en la Tarea 2 (SocioMétrica sin patrocinador).
2. `date` llega del CSV como texto y el frame de sondeos lo lleva como `Timestamp`: comparar sin normalizar no casa nada. Test en la Tarea 2 (el fixture del CSV usa texto).
3. Una regla `sub` regional cuando la fuente ya tiene base 0 en una provincia que queda (Valencia tras la exclusión de FA): Podemos debe recibir 0 allí y no restarse nada. Test en la Tarea 1.
4. Dos filas del CSV para el mismo sondeo y partido: aplicar la última en silencio esconde una errata. `ValueError`. Test en la Tarea 2.
5. Una fila que quita (`pct` vacío) un partido que el sondeo no lista no es error ni se reporta. Test en la Tarea 2.

---

### Task 1: `Simulator.apply_source_rules` y la herencia regional

**Files:**
- Modify: `mtpy/lib/simulator.py:1842-1899` (`build_umat`, desde `prev_pcts = self.prev_results['pct'].fillna(0)` hasta el final del bucle de reglas)
- Test: `tests/test_simulator_unit.py`

**Interfaces:**
- Consumes: nada nuevo.
- Produces: `Simulator.apply_source_rules(prev_pcts: pd.DataFrame, smap: dict[str, list[dict[str, Any]]], frame: pd.DataFrame, default_region: int, reg_votes: pd.Series) -> pd.DataFrame`, método estático. `prev_pcts` indexado por `region_id` (el total del ámbito incluido) con una columna por partido y 0 donde no hay resultado; `smap` con las reglas ya normalizadas a listas (como las deja `Simulator.__init__`); `frame` indexado por partido con la columna `vpred`; `reg_votes` votos válidos por `region_id`. Devuelve una copia con las reglas aplicadas; no modifica la entrada.

- [ ] **Step 1: Escribir los tests, que fallan**

Añadir a `tests/test_simulator_unit.py`:

```python
@pytest.fixture
def source_rules_case():
    regions = pd.Index([0, 8, 28, 46], name='region_id')  # total, Barcelona, Madrid, Valencia
    prev = pd.DataFrame({'SUMAR': [15., 20., 10., 15.], 'FA': 0., 'UP': 0., 'COMPROMIS': 0., 'PP': [30., 25., 35., 30.]}, index=regions)
    votes = pd.Series([100., 40., 40., 20.], index=regions)
    frame = pd.DataFrame({'vpred': [6., 3., 1., 30.]}, index=['FA', 'UP', 'COMPROMIS', 'PP'])
    return prev, votes, frame


def test_source_rules_agg_national_sums_the_sources(source_rules_case):
    prev, votes, frame = source_rules_case
    out = Simulator.apply_source_rules(prev, {'FA': [{'type': 'agg', 'names': ['SUMAR']}]}, frame, 0, votes)
    assert out['FA'].tolist() == [15., 20., 10., 15.]
    assert out['SUMAR'].tolist() == [15., 20., 10., 15.]
    assert prev['FA'].tolist() == [0., 0., 0., 0.]  # la entrada no se modifica


def test_source_rules_agg_regional_keeps_only_the_listed_provinces(source_rules_case):
    prev, votes, frame = source_rules_case
    out = Simulator.apply_source_rules(prev, {'COMPROMIS': [{'type': 'agg', 'names': ['SUMAR'], 'regions': ['46']}]}, frame, 0, votes)
    assert out['COMPROMIS'].tolist() == pytest.approx([3.0, 0., 0., 15.], abs=0.01)


def test_source_rules_sub_national_splits_in_proportion_to_the_forecasts(source_rules_case):
    prev, votes, frame = source_rules_case
    smap = {'FA': [{'type': 'agg', 'names': ['SUMAR']}], 'UP': [{'type': 'sub', 'names': ['FA']}]}
    out = Simulator.apply_source_rules(prev, smap, frame, 0, votes)
    assert out['UP'].tolist() == pytest.approx([5.0, 6.67, 3.33, 5.0], abs=0.01)
    assert out['FA'].tolist() == pytest.approx([10.0, 13.33, 6.67, 10.0], abs=0.01)


def test_source_rules_sub_regional_inherits_only_where_the_party_runs(source_rules_case):
    # Podemos no concurre en Barcelona (8): FA conserva allí la base entera y las filas nacionales se
    # recalculan desde las provincias (ponderadas por votos), no restando en la fila nacional
    prev, votes, frame = source_rules_case
    smap = {'FA': [{'type': 'agg', 'names': ['SUMAR']}],
            'UP': [{'type': 'sub', 'names': ['FA'], 'regions': {'exclude': ['8']}}]}
    out = Simulator.apply_source_rules(prev, smap, frame, 0, votes)
    assert out['UP'].tolist() == pytest.approx([2.33, 0., 3.33, 5.0], abs=0.01)
    assert out['FA'].tolist() == pytest.approx([12.67, 20., 6.67, 10.0], abs=0.01)
    assert (out['FA'] + out['UP']).loc[[8, 28, 46]].tolist() == pytest.approx([20., 10., 15.], abs=0.01)


def test_source_rules_sub_regional_with_an_empty_source_province(source_rules_case):
    # FA excluida de Valencia (46) y Podemos de Barcelona (8): en Valencia no hay nada que heredar
    prev, votes, frame = source_rules_case
    smap = {'FA': [{'type': 'agg', 'names': ['SUMAR']}, {'type': 'agg', 'names': [], 'regions': {'exclude': ['46']}}],
            'UP': [{'type': 'sub', 'names': ['FA'], 'regions': {'exclude': ['8']}}]}
    out = Simulator.apply_source_rules(prev, smap, frame, 0, votes)
    assert out['UP'].tolist() == pytest.approx([1.33, 0., 3.33, 0.], abs=0.01)
    assert out['FA'].tolist() == pytest.approx([10.67, 20., 6.67, 0.], abs=0.01)


def test_source_rules_split_shares_the_previous_vote_equally(source_rules_case):
    prev, votes, frame = source_rules_case
    out = Simulator.apply_source_rules(prev, {'SUMAR': [{'type': 'split', 'names': ['FA', 'UP']}]}, frame, 0, votes)
    assert out['FA'].tolist() == [7.5, 10., 5., 7.5]
    assert out['UP'].tolist() == [7.5, 10., 5., 7.5]
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_simulator_unit.py -k source_rules -v`
Expected: 6 FAIL con `AttributeError: ... has no attribute 'apply_source_rules'`

- [ ] **Step 3: Implementar `apply_source_rules` en `mtpy/lib/simulator.py`**

Método estático junto a `poll_blocks`, con docstring. Mueve a él, tal cual, el bucle de reglas de `build_umat` (tipos `agg`, `sub`, `split` y el bloque `regions`), con estos cambios:

- Trabaja sobre `prev_pcts.copy()`.
- Si la regla trae `regions`, calcula antes de aplicar el tipo una máscara booleana `kept` indexada como `prev_pcts`: la fila `default_region` siempre `True`; con `{'exclude': [...]}` `False` en esas provincias; con una lista, `False` fuera de ellas (códigos como `int`, igual que hoy).
- `sub`: la parte heredada `np.round(prev_v * fc_k / (fc_k + fc_v), 2)` se pone a 0 donde `kept` es `False` antes de asignarla a `key` y de restar `inherit / len(names)` a cada fuente. Si la regla trae `regions`, tras la resta se recalcula la fila nacional de `key` y de cada fuente como `np.round((prev_pcts.loc[provinces, col] * reg_votes.loc[provinces]).sum() / reg_votes.loc[default_region], 2)`, con `provinces` las filas distintas de `default_region`.
- El bloque `regions` posterior (poner a 0 fuera de `kept` y recalcular la fila nacional de `key`) se mantiene para todos los tipos.

Después `build_umat` queda con:

```python
prev_pcts = self.apply_source_rules(
    self.prev_results['pct'].fillna(0), self.smap, frame, self.default_region, self.prev_totals['votes']
)
names = self.params['names']
prev_pcts = prev_pcts.where(prev_pcts > 0, np.nan)[names]
```

y sigue igual desde ahí.

- [ ] **Step 4: Comprobar que pasan y que nada más se rompe**

Run: `python -m pytest tests/test_simulator_unit.py -v` y después `python -m pytest -q`
Expected: los 6 nuevos PASS; el resto como en `master` (anotar cualquier fallo preexistente con su nombre).

- [ ] **Step 5: Commit**

```bash
git status && git log --oneline -3
git add mtpy/lib/simulator.py tests/test_simulator_unit.py
git commit -m "Simulator: las reglas sub regionales heredan solo donde el partido concurre

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: CSV curado de correcciones de sondeos

**Files:**
- Modify: `mtpy/lib/loader.py` (función de módulo nueva junto a `scope_colmap`/`table_grid`; `WikipediaLoader.__init__` tras `self.sponsors_missing = []`; `WikipediaLoader.build_series` entre `results = pd.DataFrame(results)` y el `set_index(...).loc[...]` que sigue)
- Modify: `load/run_load.py:244-248` (`load_polls`)
- Create: `data/polls/es/2026-11-29.csv`
- Test: `tests/test_loader_unit.py`

**Interfaces:**
- Consumes: nada de la Tarea 1.
- Produces: `apply_poll_fixes(polls: pd.DataFrame, results: pd.DataFrame, fixes: pd.DataFrame, parties: dict[str, int], pollsters: dict[str, int], sponsors: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]` en `mtpy/lib/loader.py`; atributo `WikipediaLoader.fixes_missing: list[str]`; clave `fixes_missing` en el informe de `load_polls`.

Semántica de `apply_poll_fixes` (del diseño):
- `fixes` es el CSV tal como lo devuelve `read_csv`: columnas `pollster, date, sponsor, party, pct, seats, note`; `date` texto `YYYY-MM-DD`; `sponsor` vacío es «sin patrocinador» (`sponsor_id` 0).
- El sondeo se identifica por `(date, pollster_id, sponsor_id)` en `polls`; la fila de resultado por eso más `party_id`.
- `pct` con valor: fija `pct` en la fila existente o crea la fila (copiando `event_date`, `event_scope`, `date`, `pollster_id`, `sponsor_id` del sondeo; `party_id` y `party` del catálogo). `seats` con valor fija `seats`, `seats_min` y `seats_max` al mismo entero; vacío no toca los escaños existentes y deja `None` en una fila nueva.
- `pct` vacío: elimina la fila del partido si existe; si no existe, nada.
- `polls['parties']` se recalcula como el número de filas de `results` de cada sondeo.
- `party`, `pollster` o `sponsor` no vacío que no esté en su diccionario: `ValueError` que nombra la columna y el valor. Dos filas con el mismo sondeo y partido: `ValueError` con «duplicate».
- Filas cuyo sondeo no está en `polls`: no se aplican y se devuelven como `'{pollster} {date}'` en la lista, con la fecha en `%Y-%m-%d` y sin repetir un sondeo con varias filas.
- No modifica los frames de entrada.

- [ ] **Step 1: Escribir los tests de la función pura, que fallan**

Añadir a `tests/test_loader_unit.py` (importar `apply_poll_fixes` de `mtpy.lib.loader`):

```python
def poll_fix_frames():
    key = {'event_date': pd.Timestamp('2026-11-29'), 'event_scope': 'es'}
    celeste = {**key, 'date': pd.Timestamp('2026-09-21'), 'pollster_id': 7, 'sponsor_id': 3}
    socio = {**key, 'date': pd.Timestamp('2026-10-11'), 'pollster_id': 9, 'sponsor_id': 0}
    polls = pd.DataFrame([
        {**celeste, 'pollster': 'Celeste Tel', 'sponsor': 'Onda Cero', 'parties': 2},
        {**socio, 'pollster': 'SocioMétrica', 'sponsor': None, 'parties': 1},
    ])
    empty = {'seats': None, 'seats_min': None, 'seats_max': None}
    results = pd.DataFrame([
        {**celeste, 'party_id': 5, 'party': 'SUMAR', 'pct': 6.4, 'seats': 7, 'seats_min': 6, 'seats_max': 8},
        {**celeste, 'party_id': 1, 'party': 'PP', 'pct': 33.0, **empty},
        {**socio, 'party_id': 5, 'party': 'SUMAR', 'pct': 5.8, **empty},
    ])
    ids = ({'SUMAR': 5, 'COMPROMIS': 401, 'PP': 1}, {'Celeste Tel': 7, 'SocioMétrica': 9}, {'Onda Cero': 3, 'El Español': 4})
    return polls, results, ids


def poll_fixes(rows):
    return pd.DataFrame(rows, columns=['pollster', 'date', 'sponsor', 'party', 'pct', 'seats', 'note'])


def test_poll_fixes_set_and_create_rows():
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([
        ['Celeste Tel', '2026-09-21', 'Onda Cero', 'SUMAR', 5.1, None, 'Compromís aparte'],
        ['Celeste Tel', '2026-09-21', 'Onda Cero', 'COMPROMIS', 1.3, 3, None],
        ['SocioMétrica', '2026-10-11', None, 'SUMAR', 4.9, None, None],
        ['SocioMétrica', '2026-10-11', None, 'COMPROMIS', 0.9, 3, None],
    ])
    out_polls, out_results, missing = apply_poll_fixes(polls, results, fixes, *ids)
    assert missing == []
    rows = out_results.set_index(['pollster_id', 'party'])
    assert rows.loc[(7, 'SUMAR'), ['pct', 'seats', 'seats_min', 'seats_max']].tolist() == [5.1, 7, 6, 8]
    assert rows.loc[(7, 'COMPROMIS'), ['pct', 'seats', 'seats_min', 'seats_max', 'party_id']].tolist() == [1.3, 3, 3, 3, 401]
    assert rows.loc[(7, 'COMPROMIS'), ['event_scope', 'sponsor_id']].tolist() == ['es', 3]
    assert rows.loc[(9, 'SUMAR'), 'pct'] == 4.9 and rows.loc[(9, 'COMPROMIS'), 'pct'] == 0.9
    assert out_polls.set_index('pollster_id')['parties'].to_dict() == {7: 3, 9: 2}
    assert results.shape[0] == 3 and polls['parties'].tolist() == [2, 1]  # entradas intactas


def test_poll_fixes_remove_a_party_and_report_unread_polls():
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([
        ['Celeste Tel', '2026-09-21', 'Onda Cero', 'PP', None, None, 'fuera'],
        ['SocioMétrica', '2026-10-11', None, 'COMPROMIS', None, None, 'no estaba: nada que quitar'],
        ['Celeste Tel', '2026-10-30', 'Onda Cero', 'SUMAR', 5.0, None, 'sondeo no leído'],
    ])
    out_polls, out_results, missing = apply_poll_fixes(polls, results, fixes, *ids)
    assert missing == ['Celeste Tel 2026-10-30']
    assert out_results['party'].tolist() == ['SUMAR', 'SUMAR']
    assert out_polls.set_index('pollster_id')['parties'].to_dict() == {7: 1, 9: 1}


@pytest.mark.parametrize('row, match', [
    (['Celeste Tel', '2026-09-21', 'Onda Cero', 'COMPROMÍS', 1.3, 3, None], 'party'),
    (['Celeste-Tel', '2026-09-21', 'Onda Cero', 'SUMAR', 5.1, None, None], 'pollster'),
    (['Celeste Tel', '2026-09-21', 'Onda Zero', 'SUMAR', 5.1, None, None], 'sponsor'),
])
def test_poll_fixes_reject_unknown_names(row, match):
    polls, results, ids = poll_fix_frames()
    with pytest.raises(ValueError, match=match):
        apply_poll_fixes(polls, results, poll_fixes([row]), *ids)


def test_poll_fixes_reject_duplicates():
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([['Celeste Tel', '2026-09-21', 'Onda Cero', 'SUMAR', 5.1, None, None]] * 2)
    with pytest.raises(ValueError, match='duplicate'):
        apply_poll_fixes(polls, results, fixes, *ids)
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_loader_unit.py -k poll_fixes -v`
Expected: FAIL en el import (`ImportError: cannot import name 'apply_poll_fixes'`)

- [ ] **Step 3: Implementar `apply_poll_fixes` en `mtpy/lib/loader.py`**

Función de módulo con la firma y la semántica de arriba, docstring en inglés como las vecinas. Normalizar `fixes` primero (`date` a `Timestamp`, `sponsor`/`seats`/`pct` nulos con `pd.isnull`), validar nombres y duplicados antes de tocar nada, y construir las filas nuevas con las mismas columnas que `results` (`pd.concat` solo con frames no vacíos).

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_loader_unit.py -k poll_fixes -v`
Expected: 6 PASS

- [ ] **Step 5: Escribir el test del cableado en `build_series`, que falla**

```python
def test_build_series_applies_the_curated_poll_fixes(monkeypatch):
    import types
    from mtpy.lib import loader as loader_module

    loader = bare_loader('es')
    loader.event_date, loader.params, loader.exclude, loader.verbose = '2026-11-29', {}, None, 0
    loader.pollsters = pd.DataFrame({'id': [7], 'mtype': ['cati']})
    loader.m_polls = types.SimpleNamespace(key=['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id'],
                                           format_data=lambda df, **kwargs: df)
    loader.m_results = types.SimpleNamespace(key=loader.m_polls.key + ['party_id'], format_data=lambda df, **kwargs: df)
    loader.data = [{
        'pollster_id': 7, 'pollster': 'Celeste Tel', 'sponsor_id': 3, 'sponsor': 'Onda Cero', 'name': 'Celeste Tel / Onda Cero',
        'date': pd.Timestamp('2026-09-21'), 'start_date': pd.Timestamp('2026-09-16'), 'end_date': pd.Timestamp('2026-09-21'),
        'sample_size': 1100, 'ctype': None,
        'results': [{'party_id': 5, 'party': 'SUMAR', 'pct': 6.4, 'seats': None, 'seats_min': None, 'seats_max': None}],
    }]
    fixes = pd.DataFrame({'pollster': ['Celeste Tel'], 'date': ['2026-09-21'], 'sponsor': ['Onda Cero'],
                          'party': ['SUMAR'], 'pct': [5.1], 'seats': [None], 'note': [None]})
    names = []
    loader.app = types.SimpleNamespace(data=types.SimpleNamespace(
        exists=lambda name: names.append(name) or name == 'polls/es/2026-11-29.csv', read_csv=lambda name: fixes))
    calls = []

    def fake_apply(polls, results, fixes, parties, pollsters, sponsors):
        calls.append((fixes, parties, pollsters, sponsors))
        return polls, results, ['Celeste Tel 2026-10-30']

    monkeypatch.setattr(loader_module, 'apply_poll_fixes', fake_apply)
    loader.build_series()

    assert names == ['polls/es/2026-11-29.csv']
    assert calls[0][0] is fixes and calls[0][1] is loader.parties_idmap
    assert loader.fixes_missing == ['Celeste Tel 2026-10-30']
```

- [ ] **Step 6: Comprobar que falla**

Run: `python -m pytest tests/test_loader_unit.py::test_build_series_applies_the_curated_poll_fixes -v`
Expected: FAIL (`AttributeError` en `self.app` o `fixes_missing`, según el orden)

- [ ] **Step 7: Cablear el loader y el informe de `run_load`**

- `WikipediaLoader.__init__`: `self.fixes_missing = []` junto a los otros `*_missing`.
- `build_series`, justo tras `results = pd.DataFrame(results)`:

```python
# Curated corrections of the event (`data/polls/{scope}/{event_date}.csv`), e.g. a party given apart in a
# footnote while its share is folded into another column
fname = 'polls/{}/{}.csv'.format(self.scope, self.event_date)
self.fixes_missing = []
if self.app.data.exists(fname):
    polls, results, self.fixes_missing = apply_poll_fixes(
        polls, results, self.app.data.read_csv(fname), self.parties_idmap, self.pollsters_idmap, self.sponsors_idmap
    )
    if self.verbose > 0 and len(self.fixes_missing) > 0:
        print('Poll fixes without a poll read: {}'.format(', '.join(self.fixes_missing)))
```

  Si `bare_loader` necesita más atributos de los que fija el test, añadirlos al test, no relajar el loader.
- `load/run_load.py`, `load_polls`: tras `loader.build_series()`, `out['fixes_missing'] = top(loader.fixes_missing)`.

- [ ] **Step 8: Comprobar que pasa y que `run_load` sigue bien**

Run: `python -m pytest tests/test_loader_unit.py -v`
Expected: todo PASS

- [ ] **Step 9: Crear `data/polls/es/2026-11-29.csv`**

```csv
pollster,date,sponsor,party,pct,seats,note
Celeste Tel,2026-09-21,Onda Cero,SUMAR,5.1,,Compromís aparte: la cifra de Sumar en Wikipedia (6.4) incluía 1.3 de Compromís
Celeste Tel,2026-09-21,Onda Cero,COMPROMIS,1.3,3,
SocioMétrica,2026-10-11,El Español,SUMAR,4.9,,Compromís aparte: la cifra de Sumar en Wikipedia (5.8) incluía 0.9 de Compromís
SocioMétrica,2026-10-11,El Español,COMPROMIS,0.9,3,
```

Añadir a `tests/test_data_files.py` un test que lea todos los `data/polls/*/*.csv` y compruebe las columnas exactas, que `date` parsea como fecha, que `party` no está vacío y que no hay pares `(pollster, date, sponsor, party)` repetidos:

```python
def test_poll_fixes_files_are_well_formed():
    import glob
    files = glob.glob(os.path.join(DATA, 'polls', '*', '*.csv'))
    assert files
    for fname in files:
        df = pd.read_csv(fname)
        assert df.columns.tolist() == ['pollster', 'date', 'sponsor', 'party', 'pct', 'seats', 'note'], fname
        assert pd.to_datetime(df['date'], format='%Y-%m-%d').notnull().all(), fname
        assert df['party'].notnull().all() and df['pollster'].notnull().all(), fname
        keys = df[['pollster', 'date', 'sponsor', 'party']].fillna('')
        assert not keys.duplicated().any(), fname
```

Run: `python -m pytest tests/test_data_files.py -v`
Expected: PASS

- [ ] **Step 10: Suite completa y commit**

Run: `python -m pytest -q`
Expected: como tras la Tarea 1, más los nuevos.

```bash
git status && git log --oneline -3
git add mtpy/lib/loader.py load/run_load.py data/polls/es/2026-11-29.csv tests/test_loader_unit.py tests/test_data_files.py
git commit -m "Loader: correcciones curadas de sondeos en data/polls, con Compromís aparte en Celeste-Tel y SocioMétrica

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `params.json`, Compromís y Podemos

**Files:**
- Modify: `data/params.json` (bloque `es` → `2026-11-29`)
- Test: `tests/test_data_files.py`

**Interfaces:**
- Consumes: la herencia regional de la Tarea 1 (sin ella, la regla de Podemos vaciaría la base de FA en Cataluña).
- Produces: las reglas que lee `Simulator` vía `get_event_params(..., path=...)`.

- [ ] **Step 1: Escribir el test, que falla**

```python
def test_params_2026_general_election_line_up():
    conf = json.load(open(os.path.join(DATA, 'params.json')))['es']['2026-11-29']
    assert conf['parties']['event'][-1] == 'COMPROMIS'
    assert conf['bmaps']['blocks']['Regionalista'] == ['PNV', 'CC', 'BNG', 'COMPROMIS']
    assert conf['smap']['COMPROMIS'] == {'type': 'agg', 'names': ['SUMAR'], 'regions': ['3', '12', '46']}
    assert conf['smap']['UP'] == {'type': 'sub', 'names': ['FA'], 'regions': {'exclude': ['8', '17', '25', '43']}}
    assert conf['smap']['FA'][1]['regions'] == {'exclude': ['3', '12', '46']}
```

Run: `python -m pytest tests/test_data_files.py::test_params_2026_general_election_line_up -v`
Expected: FAIL en la primera aserción

- [ ] **Step 2: Editar `data/params.json`**

En `es/2026-11-29`: añadir `"COMPROMIS"` al final de `parties.event`; `bmaps.blocks.Regionalista` a `["PNV", "CC", "BNG", "COMPROMIS"]`; en `smap`, `"COMPROMIS": {"type": "agg", "names": ["SUMAR"], "regions": ["3", "12", "46"]}` y `"UP": {"type": "sub", "names": ["FA"], "regions": {"exclude": ["8", "17", "25", "43"]}}`. Nada más cambia (ni `parties.polls`, ni `bmaps.min/max/main/vs`, ni FA, PP, SALF). Conservar el formato del fichero.

- [ ] **Step 3: Comprobar que pasa, con la validación de códigos de provincia**

Run: `python -m pytest tests/test_data_files.py -v`
Expected: PASS, incluido `test_params_regions_are_valid_province_codes`

- [ ] **Step 4: Commit**

```bash
git status && git log --oneline -3
git add data/params.json tests/test_data_files.py
git commit -m "Params 29-N: Compromís aparte en sus provincias y Podemos sin base en Cataluña

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Comprobación contra la base

**Files:**
- Ninguno del repo. Script temporal en el scratchpad de la sesión.

**Interfaces:**
- Consumes: todo lo anterior; la base remota del `.env` (lectura únicamente).

- [ ] **Step 1: Pasada determinista antes de la recarga de Luis**

Con la base actual (Compromís tiene un solo sondeo con cifra, así que su promedio no se ajusta y su parte se renormaliza; sirve para Cataluña y para las bases valencianas):

```python
import warnings; warnings.filterwarnings('ignore')
from mtpy import mtpy; mtpy.run()
from mtpy.lib.simulator import Simulator

sim = Simulator(scope='es', event_date='2026-11-29', seed=1, verbose=0)
sim.run(n_sim=1, split=True, random=False)
seats = sim.result(0)
print(seats.loc[['Barcelona', 'Gerona', 'Lérida', 'Tarragona'], ['FA', 'UP', 'ERC']])
print(seats.loc[['Alicante', 'Castellón', 'Valencia'], ['FA', 'UP', 'COMPROMIS']])
print(sim.unit(0, 'Barcelona').loc[['FA', 'UP'], 'prev_pct'])
print(sim.unit(0, 'Valencia').loc[['FA', 'UP', 'COMPROMIS'], 'prev_pct'])
```

Run: `cd` al repo, `set -a; . ./.env; set +a` y el script con el Python de pyenv.
Expected: UP con 0 escaños en las cuatro provincias catalanas; `prev_pct` de FA en Barcelona 15.22 y de UP 0; FA con 0 escaños y `prev_pct` 0 en Alicante, Castellón y Valencia; `prev_pct` de COMPROMIS en Valencia 16.9. Anotar los escaños de FA en Barcelona.

- [ ] **Step 2: Suite completa**

Run: `python -m pytest -q`
Expected: solo los fallos preexistentes anotados en la Tarea 1, si los hay.

- [ ] **Step 3: Pasada completa tras la recarga de Luis**

Cuando Luis haya recargado los sondeos del ciclo con `overwrite` (las correcciones se aplican solas) y pasado `PollsCompute`, repetir el script del paso 1 y además:

```python
print(sim.model.series.loc[sim.model.series['COMPROMIS'].notnull(), ['pollster', 'COMPROMIS', 'FA']])
print(sim.summary().loc[['FA', 'UP', 'COMPROMIS'], ['pct', 'seats', 'seats_lo', 'seats_hi']])
```

Expected: tres sondeos con cifra de Compromís fuera de los `online` (Celeste Tel 1.3, SocioMétrica 0.9, Sigma Dos 0.9) y en esos dos primeros FA igual a 5.1 y 4.9; Compromís ajustado, con escaños solo en Alicante, Castellón y Valencia y del orden de 2-3 en total; FA sin escaños allí; Podemos sin escaños en Cataluña. Si Compromís no se ajusta, revisar que las dos filas corregidas existen en `polls_results` y que sus sondeos tienen `computed` a `True`.

- [ ] **Step 4: Cierre**

Anotar en la spec (`Estado`) la fecha de la comprobación y los escaños observados. Sin commit de ficheros del scratchpad.
