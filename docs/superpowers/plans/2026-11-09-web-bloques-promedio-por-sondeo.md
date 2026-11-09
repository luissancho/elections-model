# Vista Bloques ajustada sobre la suma por sondeo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La línea de Izquierda y Derecha en la vista Bloques del promedio sale de un ajuste sobre lo que suman sus partidos en cada sondeo (sin escalón cuando entra un partido), y una línea discontinua marca la entrada de cada partido nuevo.

**Architecture:** El `Forecaster` gana `fit_groups` (ajuste de un grupo sobre la suma por sondeo, sin tocar `forecast`) y una regla pura de entradas (`party_entries` en `utils`). La publicación del run y el backtest añaden `blocks` y `first_polls` a la parte `series` (y `kind` al CSV); el evento los lee del `series.csv` del backtest. La web los usa cuando vienen y suma partidos cuando no.

**Tech Stack:** Python 3.11 (pandas, numpy, pytest), JS ES modules + ECharts 5.6 (tests con `node`), Jinja2.

**Spec:** `docs/superpowers/specs/2026-11-09-web-bloques-promedio-por-sondeo-design.md`

## Global Constraints

- Python: `~/.pyenv/versions/3.11.9/bin/python -m pytest` (el `python3` del sistema no tiene pandas).
- Identificadores en inglés neutro; textos públicos en español con «sólo» con tilde, «encuestadora»/«casa», nunca «empresa» ni «patrocinador» (`assert_public_wording`).
- Docstrings en todas las funciones (AGENTS.md), PEP8, líneas ≤ 120.
- Bloques publicados: sólo `Izquierda` y `Derecha` (la salida de `web_blocks` sin `REST_BLOCK`). `Otros` se sigue calculando en el navegador como 100 menos los dos.
- Umbral de entrada: `min_days = 90` días desde el primer sondeo del ciclo.
- Contrato del bundle sigue en 1; claves nuevas `blocks` (`{'parties': [...], 'mean': {}, 'lo': {}, 'hi': {}}`) y `first_polls` (`{nombre: 'YYYY-MM-DD'}`) en `series` y `event-series`.
- CSV de la serie (run, backtest y evento): columnas `date, name, kind, mean, lo, hi` (`kind` ∈ `party`, `block`), filas de bloque detrás de las de partido. En el backtest: `event_date, horizon, date, name, kind, mean, lo, hi`. Una fila sin `kind` (o `NaN`) cuenta como `party`.
- Redondeo: `PCT_DECIMALS` (2), igual que los partidos.
- Tests nunca con `out=None` ni escribiendo en `backtest/results` o `files/` reales (rutas `tmp_path`).
- Luis commitea en paralelo en el mismo árbol: antes de cada commit, `git status` y añadir sólo los ficheros de la tarea por ruta (nunca `git add -A` ni `commit -a`).
- Commits terminan con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Bloque con un miembro sin sondeos suficientes** (UP en un ámbito autonómico): el bloque se ajusta igual con lo que listan los demás; si el grupo entero tiene menos de 3 sondeos, no hay columna y la web suma partidos para ese bloque. Test en la Tarea 1 y en la Tarea 5.
- **Run antiguo sin `blocks`/`first_polls`** (servido con `?run=`): la web dibuja como hoy, sin marcas, y la página conserva la frase vieja. Test en las Tareas 5 y 6.
- **Serie del bloque más corta que la de los partidos** (fecha final): se ajusta con `fc.max_fc` y se corta en `end`; mismo `dates` que los partidos. Test en la Tarea 3.
- **Partido cuya etiqueta heredada es anterior** (FA hereda la fecha de SUMAR): no se marca. Test en la Tarea 2.
- **Elección pasada cuyo backtest no trae filas `block`**: `blocks` vacío y `first_polls` igualmente calculado; la web suma y sí dibuja marcas. Test en la Tarea 4.

---

### Task 1: `Forecaster.fit_groups`

**Files:**
- Modify: `mtpy/lib/forecaster.py` (`fit`, hacia la línea 686; nuevo método detrás de `fit_forecast`)
- Test: `tests/test_forecaster_unit.py`

**Interfaces:**
- Produces: `Forecaster.fit_groups(groups: dict[str, list[str]], max_fc: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]` — `(forecast, fc_stat)` indexados por `self.fc_index`, una columna por grupo ajustado (los de menos de 3 sondeos útiles no aparecen); `fc_stat` lleva un dict por celda como `self.fc_stat` (`mean`, `cmin`, `cmax`, `err`, `nobs`, `neff`) o `None`. No muta `self.forecast`, `self.fc_stat` ni `self.series`.
- Produces (privado): `Forecaster._fit_column(df: pd.DataFrame, values: pd.Series, max_fc: int, ret_stat: bool)` — el cuerpo actual de `fit` desde el filtro de sondeos; `fit(name, ...)` pasa a ser `self._fit_column(self.fc_series, self.fc_series[name], max_fc, ret_stat)`. Salida de `fit` idéntica a la de hoy.

- [ ] **Step 1: Write the failing tests** (usar `_synthetic_forecaster`, que tiene `PP` y `PSOE`)

```python
def test_fit_groups_of_one_member_matches_the_fit_of_the_party():
    fc, _, _ = _synthetic_forecaster()
    fc.fit_forecast(names=['PP'], max_fc=0)
    forecast, stat = fc.fit_groups({'Solo': ['PP']}, max_fc=0)
    pd.testing.assert_series_equal(forecast['Solo'].dropna(), fc.forecast['PP'].dropna().astype(float), check_names=False)
    day = forecast['Solo'].dropna().index[100]
    assert stat.loc[day, 'Solo']['cmin'] == pytest.approx(fc.fc_stat.loc[day, 'PP']['cmin'])


def test_fit_groups_does_not_touch_the_forecast_of_the_parties():
    fc, _, _ = _synthetic_forecaster()
    fc.fit_forecast(names=['PP', 'PSOE'], max_fc=0)
    before, stat_before = fc.forecast.copy(), fc.fc_stat.copy()
    forecast, _ = fc.fit_groups({'Todos': ['PP', 'PSOE']}, max_fc=0)
    pd.testing.assert_frame_equal(fc.forecast, before)
    assert fc.fc_stat.equals(stat_before) and 'Todos' not in fc.forecast


def test_fit_groups_has_no_step_when_a_member_enters_mid_cycle():
    """Un miembro que las casas empiezan a listar a mitad de ciclo, restando lo mismo a otro: la suma por sondeo no
    cambia, así que el grupo no salta el día de la entrada (la suma de los ajustes sí)."""
    fc, _, _ = _synthetic_forecaster(noise={1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01})
    birth = pd.Timestamp('2026-06-01')
    dates = fc.series.index.get_level_values('date')
    fc.series['NEW'] = np.where(dates >= birth, 4., np.nan)
    fc.series.loc[dates >= birth, 'PP'] -= 4.
    fc.names = ['PP', 'PSOE', 'NEW']
    fc.he_enabled = False
    fc.fit_forecast(names=['PP', 'NEW'], max_fc=0)
    forecast, _ = fc.fit_groups({'Derecha': ['PP', 'NEW']}, max_fc=0)
    first = fc.series.index.get_level_values('date')[fc.series['NEW'].notnull().to_numpy()].min()
    prev = first - pd.Timedelta(days=1)
    assert abs(forecast.loc[first, 'Derecha'] - forecast.loc[prev, 'Derecha']) < 0.3
    summed = fc.forecast[['PP', 'NEW']].sum(axis=1, min_count=1)
    assert summed.loc[first] - summed.loc[prev] > 1.          # el artefacto de hoy


def test_fit_groups_skips_a_group_with_fewer_than_three_polls():
    fc, _, _ = _synthetic_forecaster()
    fc.series['RARE'] = np.nan
    fc.series.iloc[:2, fc.series.columns.get_loc('RARE')] = 1.
    fc.names = ['PP', 'PSOE', 'RARE']
    with pytest.warns(UserWarning):
        forecast, stat = fc.fit_groups({'Raro': ['RARE'], 'Solo': ['PP']}, max_fc=0)
    assert list(forecast.columns) == ['Solo'] and list(stat.columns) == ['Solo']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_forecaster_unit.py -k fit_groups -v`
Expected: FAIL con `AttributeError: 'Forecaster' object has no attribute 'fit_groups'`.

- [ ] **Step 3: Implement `_fit_column` and `fit_groups` in `mtpy/lib/forecaster.py`**

`fit_groups`: si `self.disp_enabled` y no hay `dispersion`, o `self.he_enabled` y no hay `house_effects`, ajusta antes como `fit_forecast` (para que `fc_series` vaya neto). Por grupo: miembros presentes en `self.fc_series.columns`; `values = self.fc_series[members].sum(axis=1, min_count=1)`; `res = self._fit_column(self.fc_series, values, max_fc, ret_stat=True)`; `None` → aviso (el mismo texto que `fit`, con el nombre del grupo) y se salta. Con `ret_stat=True`, `self.alpha` como en `fit`.

- [ ] **Step 4: Run tests to verify they pass, plus the whole Forecaster suite**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_forecaster_unit.py -q`
Expected: todo PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py tests/test_forecaster_unit.py
git commit -m "Forecaster.fit_groups: ajuste de un bloque sobre la suma de sus partidos en cada sondeo"
```

---

### Task 2: Regla de entradas (`party_entries`) y `Forecaster.cycle_births`

**Files:**
- Modify: `mtpy/lib/utils.py` (junto a `web_blocks`)
- Modify: `mtpy/lib/forecaster.py` (método detrás de `fit_groups`)
- Test: `tests/test_utils.py`, `tests/test_forecaster_unit.py`

**Interfaces:**
- Produces: `utils.party_entries(first_polls: pd.Series, members: list[str], labels: dict[str, list[str]], min_days: int = 90) -> dict[str, str]` — `first_polls` es la primera fecha de sondeo por etiqueta de sondeo (índice: etiqueta, valores `datetime64`). Inicio del ciclo = `first_polls.min()`. Para cada miembro, su fecha es el mínimo sobre `labels.get(member, [member])` presentes en `first_polls`; se incluye si `(fecha − inicio).days >= min_days`. Devuelve `{miembro: 'YYYY-MM-DD'}` ordenado por fecha; `{}` si `first_polls` está vacía.
- Produces: `utils.first_poll_dates(polls: pd.DataFrame, labels: list[str]) -> pd.Series` — para cada etiqueta con columna en `polls`, la menor fecha (nivel o columna `date`) con valor no nulo; las que nunca aparecen no salen.
- Produces: `Forecaster.cycle_births(groups: dict[str, list[str]], min_days: int = 90) -> dict[str, str]` — `party_entries(self.party_first_polls, miembros de groups, {n: list(self.blocks.loc[n, 'parties']) for n in miembros si n in self.blocks.index}, min_days)`.

- [ ] **Step 1: Write the failing tests**

En `tests/test_utils.py`:

```python
def test_party_entries_marks_only_the_parties_born_mid_cycle_with_their_inherited_labels():
    from mtpy.lib.utils import party_entries
    first = pd.Series(pd.to_datetime({'PP': '2023-08-07', 'VOX': '2023-08-07', 'SUMAR': '2023-08-07',
                                      'FA': '2025-05-01', 'UP': '2023-12-15', 'SALF': '2024-06-06',
                                      'PSOE': '2023-08-20'}))
    labels = {'FA': ['FA', 'SUMAR'], 'PP': ['PP', 'UPN']}
    out = party_entries(first, ['PSOE', 'FA', 'UP', 'PP', 'VOX', 'SALF'], labels)
    assert out == {'UP': '2023-12-15', 'SALF': '2024-06-06'} and list(out) == ['UP', 'SALF']
    assert party_entries(first, ['UP'], {}, min_days=200) == {}
    assert party_entries(pd.Series(dtype='datetime64[ns]'), ['UP'], {}) == {}


def test_first_poll_dates_reads_the_first_non_null_of_each_label():
    from mtpy.lib.utils import first_poll_dates
    polls = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-01', '2024-03-01']),
                          'UP': [None, None, 2.], 'PP': [30., 31., 32.]}).set_index('date')
    out = first_poll_dates(polls, ['UP', 'PP', 'CUP'])
    assert out.to_dict() == {'UP': pd.Timestamp('2024-03-01'), 'PP': pd.Timestamp('2024-01-01')}
```

En `tests/test_forecaster_unit.py`:

```python
def test_cycle_births_uses_the_poll_labels_of_each_simulated_name():
    fc = Forecaster.__new__(Forecaster)
    fc.party_first_polls = pd.Series(pd.to_datetime({'PSOE': '2023-08-07', 'SUMAR': '2023-08-07', 'FA': '2025-05-01',
                                                     'UP': '2023-12-15'}))
    fc.blocks = pd.DataFrame({'parties': [['PSOE'], ['FA', 'SUMAR'], ['UP']]}, index=['PSOE', 'FA', 'UP'])
    assert fc.cycle_births({'Izquierda': ['PSOE', 'FA', 'UP']}) == {'UP': '2023-12-15'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_utils.py tests/test_forecaster_unit.py -k "party_entries or first_poll_dates or cycle_births" -v`
Expected: FAIL con `ImportError` / `AttributeError`.

- [ ] **Step 3: Implement `party_entries`, `first_poll_dates` (utils) and `cycle_births` (Forecaster)**

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_utils.py tests/test_forecaster_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/utils.py mtpy/lib/forecaster.py tests/test_utils.py tests/test_forecaster_unit.py
git commit -m "Regla de entrada de partidos a mitad de ciclo (party_entries, cycle_births)"
```

---

### Task 3: Parte `series` del run y del backtest con bloques y entradas

**Files:**
- Modify: `mtpy/lib/publish/forecast.py` (`export_series`, `publish_forecast` hacia la línea 920)
- Modify: `mtpy/lib/bundle.py:136` (esquema `series`)
- Modify: `mtpy/lib/backtest.py` (`case_series`, línea 348)
- Modify: `tests/fakes.py` (`synthetic_forecaster`: `max_fc`, `fit_groups`, `cycle_births`)
- Modify: `tests/fixtures/bundle/series.json` (claves nuevas)
- Modify: `docs/web/contrato.md` (sección `### \`series\``)
- Test: `tests/test_publish_unit.py`, `tests/test_backtest_unit.py`

**Interfaces:**
- Consumes: `Forecaster.fit_groups`, `Forecaster.cycle_births` (Tareas 1 y 2), `utils.web_blocks`, `utils.REST_BLOCK`.
- Produces: `export_series(fc, names, end=None, blocks: Optional[dict] = None) -> tuple[dict, pd.DataFrame]`. `blocks` es `{bloque: [nombres]}`; si trae `REST_BLOCK`, se ignora ese bloque. `data` gana `blocks` (`parties`, `mean`, `lo`, `hi`, alineados a `dates`, `None` en huecos, redondeo `PCT_DECIMALS`) y `first_polls` (`fc.cycle_births(grupos)`); sin `blocks` (o `None`), `{'parties': [], 'mean': {}, 'lo': {}, 'hi': {}}` y `{}`. Los bloques se ajustan con `fc.fit_groups(grupos, max_fc=fc.max_fc)`. El frame: `date, name, kind, mean, lo, hi`.
- Produces: helper `series_blocks(sim) -> Optional[dict]` en `publish/forecast.py`: `_web_bmap(sim)` sin `REST_BLOCK` (o `None`), usado por `publish_forecast` y por `case_series`.
- Produces: `case_series` devuelve `event_date, horizon, date, name, kind, mean, lo, hi`.

- [ ] **Step 1: Extend the fake `synthetic_forecaster`**

En `tests/fakes.py`, añadir `max_fc=10` y dos funciones al `SimpleNamespace`: `fit_groups(groups, max_fc=0)` devuelve `(frame, stat)` con, por grupo, la suma constante de las columnas de `forecast` de sus miembros en los días ajustados (`fitted`) y `NaN` fuera, y `stat` con dicts `cmin = mean − 1`, `cmax = mean + 1`; anota `('fit_groups', max_fc)` en `calls` (lista nueva del namespace). `cycle_births(groups, min_days=90)` devuelve `{'VOX': '2026-09-19'}` si `'VOX'` está en algún grupo, si no `{}`.

- [ ] **Step 2: Write the failing tests** (en `tests/test_publish_unit.py`, junto a los de `export_series`)

```python
def test_export_series_publishes_the_fitted_blocks_and_the_entries():
    sim = synthetic_simulator()
    blocks = {'Izquierda': ['PSOE'], 'Derecha': ['PP', 'VOX'], 'Otros': []}
    data, frame = publish.export_series(sim.model, NAMES, end='2026-10-03', blocks=blocks)
    data = valid('series', data, mode=None)
    assert data['blocks']['parties'] == ['Izquierda', 'Derecha']
    assert data['blocks']['mean']['Derecha'] == [55.0] * 33 and data['blocks']['lo']['Derecha'][0] == 54.0
    assert len(data['blocks']['mean']['Izquierda']) == len(data['dates'])
    assert data['first_polls'] == {'VOX': '2026-09-19'} and sim.model.calls[-1] == ('fit_groups', 10)
    assert list(frame.columns) == ['date', 'name', 'kind', 'mean', 'lo', 'hi']
    assert frame.shape[0] == 33 * 3 + 33 * 2 and list(frame['kind'].unique()) == ['party', 'block']


def test_export_series_without_blocks_has_empty_blocks_and_no_entries():
    data, frame = publish.export_series(synthetic_simulator().model, NAMES)
    data = valid('series', data, mode=None)
    assert data['blocks'] == {'parties': [], 'mean': {}, 'lo': {}, 'hi': {}} and data['first_polls'] == {}
    assert set(frame['kind']) == {'party'}
```

Actualizar `test_export_series_stops_at_the_last_fitted_day` a `list(frame.columns) == ['date', 'name', 'kind', 'mean', 'lo', 'hi']`. En `test_publish_forecast_reads_the_average_today_and_cuts_the_series_at_the_anchor`, añadir `assert series['blocks']['parties'] == ['Izquierda', 'Derecha']` y que `series['blocks']['mean']` tenga la longitud de `series['dates']` (si el `FakeSimulator` de esa prueba no trae `fit_groups`, añadírselo con la misma regla que el fake de forecaster).

En `tests/test_backtest_unit.py`, `test_case_series_is_the_fitted_series_up_to_the_anchor`: columnas `['event_date', 'horizon', 'date', 'name', 'kind', 'mean', 'lo', 'hi']`, filas `33 * 3 + 33 * 2`, `set(frame.loc[frame.kind == 'block', 'name']) == {'Izquierda', 'Derecha'}`. En `fake_results`, añadir `kind` a las columnas de `series`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_publish_unit.py tests/test_backtest_unit.py -k "series" -v`
Expected: FAIL (`blocks` no esperado / columnas distintas).

- [ ] **Step 4: Implement**

- `export_series` y `series_blocks` como en Interfaces; `publish_forecast` pasa `blocks=series_blocks(sim)`.
- `case_series` pasa `blocks=series_blocks(sim)` (import perezoso como el de `export_series`).
- `bundle.SCHEMAS['series']`: `+ 'blocks': dict, 'first_polls': dict`.
- `tests/fixtures/bundle/series.json`: añadir `"blocks": {"parties": [], "mean": {}, "lo": {}, "hi": {}}, "first_polls": {}` a `data`.
- `docs/web/contrato.md`, sección `series`: enmienda 2026-11-09 con `blocks` (Izquierda y Derecha ajustados sobre la suma de sus partidos en cada sondeo; `Otros` lo calcula la web), `first_polls` (partidos de Izquierda/Derecha que entran en las encuestas ≥ 90 días después del primer sondeo del ciclo, con las etiquetas heredadas del `smap`) y la columna `kind` del CSV. El contrato sigue en 1.

- [ ] **Step 5: Run tests to verify they pass, plus publish, backtest, jobs and webapi suites**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_publish_unit.py tests/test_backtest_unit.py tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_webapi_unit.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/publish/forecast.py mtpy/lib/bundle.py mtpy/lib/backtest.py tests/fakes.py tests/fixtures/bundle/series.json docs/web/contrato.md tests/test_publish_unit.py tests/test_backtest_unit.py
git commit -m "Parte series: bloques ajustados sobre la suma por sondeo y entradas de partidos (run y backtest)"
```

---

### Task 4: Parte `event-series` con bloques y entradas

**Files:**
- Modify: `mtpy/lib/publish/events.py` (`export_event_series` línea 290, `publish_event` hacia la línea 543)
- Modify: `mtpy/lib/bundle.py:171` (esquema `event-series`)
- Modify: `tests/fixtures/bundle/event-series.json`
- Modify: `docs/web/contrato.md` (lista de partes del evento, `series`)
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Consumes: `utils.party_entries`, `utils.first_poll_dates`, `utils.web_blocks`, `Simulator.poll_blocks` (estática; import perezoso de `..simulator`).
- Produces: `export_event_series(source, event_date, first_polls: Optional[dict] = None) -> tuple`. Las filas con `kind == 'block'` van a `data['blocks']` (misma forma que el run; vacío sin ellas); `parties`/`mean`/`lo`/`hi` sólo con las filas `party` (o sin `kind`/`NaN`). `data['first_polls'] = first_polls or {}`. El frame lleva `kind` (relleno con `party` donde falte), partidos primero.
- Produces: `event_entries(comp, event_date: str, polls: pd.DataFrame) -> dict[str, str]` en `publish/events.py`: `names = event_params[event_date]['parties']['event']`, `smap = event_params[event_date].get('smap', {})`; miembros = `web_blocks(bmaps['vs'], names)` sin `REST_BLOCK` (sin bloques → `{}`); `labels = Simulator.poll_blocks(names, smap)`; `party_entries(first_poll_dates(polls, todas las etiquetas), miembros, labels)`. Cualquier clave ausente en `event_params` → `{}`. `publish_event` lo llama con `cycle` (los sondeos que ya lee).

- [ ] **Step 1: Write the failing tests**

```python
def test_export_event_series_splits_parties_and_blocks_and_carries_the_entries(tmp_path):
    write_series_source(str(tmp_path))
    path = tmp_path / 'series.csv'
    table = pd.read_csv(path)
    table['kind'] = 'party'
    blocks = table.loc[table['name'] == 'PP'].assign(name='Derecha', kind='block')
    pd.concat([table, blocks]).to_csv(path, index=False)
    data, frame = publish.export_event_series(str(tmp_path), '2023-07-23', first_polls={'SALF': '2023-03-01'})
    assert data['parties'] == ['PP', 'PSOE'] and data['blocks']['parties'] == ['Derecha']
    assert data['blocks']['mean']['Derecha'] == [32.2, 32.3, 32.41] and data['first_polls'] == {'SALF': '2023-03-01'}
    assert list(frame.columns) == ['date', 'name', 'kind', 'mean', 'lo', 'hi'] and frame['kind'].iloc[-1] == 'block'


def test_event_entries_reads_the_simulated_parties_and_their_labels():
    polls = pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-06-01']), 'PP': [30., 31.], 'PSOE': [25., 26.],
                          'SUMAR': [12., 11.], 'UP': [None, 2.]}).set_index('date')
    comp = types.SimpleNamespace(event_params={'2023-07-23': {
        'parties': {'event': ['PP', 'PSOE', 'FA', 'UP']}, 'smap': {'FA': {'type': 'agg', 'names': ['SUMAR']}},
        'bmaps': {'vs': {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP']}}}})
    assert publish.event_entries(comp, '2023-07-23', polls) == {'UP': '2023-06-01'}
    bare = types.SimpleNamespace(event_params={'2023-07-23': {'bmaps': {'vs': {}}}})
    assert publish.event_entries(bare, '2023-07-23', polls) == {}
```

Actualizar `test_export_event_series_pivots_the_series_of_the_election` (CSV sin `kind`): columnas `['date', 'name', 'kind', 'mean', 'lo', 'hi']`, `set(frame['kind']) == {'party'}`, `data['blocks']['parties'] == []`, `data['first_polls'] == {}`. En `test_publish_event_writes_the_series_and_removes_it_when_the_backtest_lacks_it`, `part['data']['first_polls'] == {}` (el `FakeComputer` no trae `parties`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_publish_unit.py -k "event_series or event_entries or publish_event_writes_the_series" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `export_event_series`, `event_entries`, la llamada en `publish_event`, el esquema `event-series` (`+ 'blocks': dict, 'first_polls': dict`), el fixture `event-series.json` y la enmienda del contrato. Exportar `event_entries` desde `mtpy/lib/publish/__init__.py` como el resto.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_publish_unit.py tests/test_pipelines_publish.py tests/test_webapi_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish/events.py mtpy/lib/publish/__init__.py mtpy/lib/bundle.py tests/fixtures/bundle/event-series.json docs/web/contrato.md tests/test_publish_unit.py
git commit -m "Parte series del evento: bloques del backtest y entradas de partidos"
```

---

### Task 5: Web — la vista Bloques lee los bloques ajustados y marca las entradas

**Files:**
- Modify: `web/dist/js/blocks.js` (`blockSeries`)
- Modify: `web/dist/js/charts/series.js` (`seriesOption`, serie de marcas hacia la línea 386)
- Modify: `web/dist/js/pages/series-wire.js` (`seriesViews`)
- Modify: `web/dist/js/pages/promedios.js` (`dataOf`), `web/dist/js/pages/elecciones.js` (`dataOf` de `wireAverageCard`)
- Test: `tests/test_js_blocks.py`, `tests/test_js_charts.py`

**Interfaces:**
- Consumes: `series.blocks`, `series.first_polls` (Tareas 3 y 4).
- Produces: `blockSeries(series, blocks)` — para cada bloque distinto de `REST_BLOCK` con `series.blocks.mean[bloque]`, toma `mean`/`lo`/`hi` de ahí (`null` → `null`); los demás, como hoy. `Otros` igual que hoy sobre los resultantes.
- Produces: `seriesOption({..., marks = []})` — cada `{date, label}` añade a la serie de marcas un `markAt(dayTime(date), label)` con la etiqueta en `position: 'start'` (al pie de la línea, para no pisar «Estimación» ni «Elección»).
- Produces: `seriesViews(...).bloques.marks` = `Object.entries(series.first_polls || {})` ordenadas por fecha → `{date, label: \`Entra ${name}\`}`; `partidos.marks` = `[]`. `dataOf` de las dos páginas pasa `marks: current.marks || []`.

- [ ] **Step 1: Write the failing tests**

En `tests/test_js_blocks.py`:

```python
FITTED = dict(SERIES, blocks={'parties': ['Derecha'], 'mean': {'Derecha': [39, 39.5]},
                              'lo': {'Derecha': [38, 38.5]}, 'hi': {'Derecha': [40, None]}})


def test_block_series_reads_the_fitted_blocks_and_sums_the_rest():
    out = run_js('b.blockSeries(%s, %s)' % (json.dumps(FITTED), json.dumps(REST)))
    assert out['mean']['Derecha'] == [39, 39.5] and out['hi']['Derecha'] == [40, None]
    assert out['mean']['Izquierda'] == [28, 28]                      # sin ajuste publicado: la suma
    assert out['mean']['Otros'] == [100 - 28 - 39, 100 - 28 - 39.5]
    assert run_js('b.blockSeries(series, blocks)')['mean']['Derecha'] == [40, 31]   # run antiguo
```

En `tests/test_js_charts.py`:

```python
def test_series_draws_the_entries_as_dashed_marks_at_the_foot():
    args = BREAKDOWN_ARGS.replace("breakdown: {", "marks: [{date: '2026-10-01', label: 'Entra SALF'}], breakdown: {")
    option = run_js('s.seriesOption(%s)' % args)
    marks = next(x for x in option['series'] if x['name'] == 'marks')['markLine']['data']
    entry = next(m for m in marks if m['label']['formatter'] == 'Entra SALF')
    assert entry['label']['position'] == 'start' and len(marks) == 2


def test_series_views_mark_the_entries_only_in_the_blocks_view():
    initial = {'series': {'dates': ['2026-10-01'], 'parties': ['PP', 'PSOE'], 'mean': {'PP': [30], 'PSOE': [28]},
                          'lo': {}, 'hi': {}, 'first_polls': {'SALF': '2024-06-06', 'UP': '2023-12-15'}},
               'polls': {'parties': [], 'polls': []}, 'projection': None,
               'catalog': {'parties': [{'name': n, 'color': '#000000', 'fullname': n, 'block': None} for n in ('PP', 'PSOE')],
                           'bmaps': {'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}}}
    views = run_js('sw.seriesViews(%s, new c.Catalog(%s.catalog))' % (json.dumps(initial), json.dumps(initial)))
    assert views['bloques']['marks'] == [{'date': '2023-12-15', 'label': 'Entra UP'}, {'date': '2024-06-06', 'label': 'Entra SALF'}]
    assert views['partidos']['marks'] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_js_blocks.py tests/test_js_charts.py -k "fitted or entries" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** los cambios de Interfaces en los cinco ficheros JS, con JSDoc actualizado.

- [ ] **Step 4: Run the JS suites**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_js_blocks.py tests/test_js_charts.py tests/test_js_share.py tests/test_js_evolution.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/blocks.js web/dist/js/charts/series.js web/dist/js/pages/series-wire.js web/dist/js/pages/promedios.js web/dist/js/pages/elecciones.js tests/test_js_blocks.py tests/test_js_charts.py
git commit -m "Vista Bloques: línea del bloque ajustada cuando el run la trae y marcas de entrada de partidos"
```

---

### Task 6: Páginas — redondeo de los bloques y texto según la serie

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`round_series` línea 564; contexto hacia la línea 834)
- Modify: `mtpy/lib/pages/elecciones.py` (contexto hacia las líneas 1206 y 1247)
- Modify: `web/templates/promedios.html:64-66` (sólo esta; la línea 304, Evolución, no se toca)
- Modify: `web/templates/eleccion.html:167`
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces: `round_series` redondea también `series['blocks']['mean'|'lo'|'hi']` y conserva `first_polls`.
- Produces: `fitted_blocks(series: Optional[dict]) -> bool` en `promedios.py` (`bool(((series or {}).get('blocks') or {}).get('parties'))`); variable de plantilla `fitted_blocks` en `/{scope}` y en el detalle de una elección.

Texto con `fitted_blocks` (promedios; en `eleccion.html`, la misma idea en una frase, sin la última cláusula del tooltip si no cabe):

> En la vista Bloques, la línea de cada bloque es el promedio de lo que suman sus partidos en cada sondeo, no la suma de sus promedios: cuando un partido entra en las encuestas a mitad de ciclo, una línea discontinua lo marca, y durante unos meses el desglose del bloque al pasar por la línea puede no cuadrar con ella. Los partidos de cada bloque se leen bajo la leyenda; Otros es lo que falta hasta 100.

Sin `fitted_blocks`, la frase de hoy tal cual.

- [ ] **Step 1: Write the failing tests**

```python
def test_round_series_rounds_the_fitted_blocks_and_keeps_the_entries():
    series = {'dates': ['d'], 'parties': ['PP'], 'mean': {'PP': [30.123]}, 'lo': {}, 'hi': {},
              'blocks': {'parties': ['Derecha'], 'mean': {'Derecha': [45.678]}, 'lo': {'Derecha': [None]}, 'hi': {'Derecha': [46.111]}},
              'first_polls': {'SALF': '2024-06-06'}}
    out = pages.promedios.round_series(series)
    assert out['blocks']['mean']['Derecha'] == [45.68] and out['blocks']['lo']['Derecha'] == [None]
    assert out['first_polls'] == {'SALF': '2024-06-06'} and series['blocks']['mean']['Derecha'] == [45.678]


def test_blocks_paragraph_follows_the_published_series(api, fresh_app):
    page = re.sub(r'\s+', ' ', html(call(api, '/es')[2]))
    assert 'cada bloque suma sus partidos' in page and 'no la suma de sus promedios' not in page
    series = fixture_data('series')
    series['blocks'] = {'parties': ['Derecha'], 'mean': {'Derecha': [40.]}, 'lo': {'Derecha': [39.]}, 'hi': {'Derecha': [41.]}}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es', RUN, 'series'), 'series', series, 'es', run_id=RUN)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/es')[2])
    text = re.sub(r'\s+', ' ', page)
    assert 'no la suma de sus promedios' in text and 'una línea discontinua lo marca' in text
    assert text.count('cada bloque suma sus partidos') == 1          # la de la Evolución sigue
    assert_public_wording(page, '/es')
```

Y el homólogo del detalle de una elección con `events_site` (reescribir `event-series.json` de `2023-07-23` con `blocks` no vacío y comprobar `no la suma de sus promedios` en `/elecciones/es/generales/2023-07`; sin bloques, `cada bloque suma sus partidos`). Si `RUN`, `fixture_data` o `webapi` no están importados en `test_pages.py`, tomarlos de `tests/test_webapi_unit.py` como hace `test_promedios_hides_the_blocks_view_without_blocks`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_pages.py -k "round_series_rounds_the_fitted or blocks_paragraph" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `round_series`, `fitted_blocks`, las variables de contexto y los dos `{% if fitted_blocks %}` en las plantillas.

- [ ] **Step 4: Run the page suites and the whole test suite**

Run: `~/.pyenv/versions/3.11.9/bin/python -m pytest tests/test_pages.py tests/test_pages_ejemplo.py -q && ~/.pyenv/versions/3.11.9/bin/python -m pytest -q -x`
Expected: todo PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/promedios.py mtpy/lib/pages/elecciones.py web/templates/promedios.html web/templates/eleccion.html tests/test_pages.py
git commit -m "Páginas: bloques ajustados redondeados y texto de la vista Bloques según la serie publicada"
```

---

### Task 7: Comprobación con datos reales (sin publicar)

**Files:** ninguno del repo; script en el scratchpad de la sesión.

- [ ] **Step 1:** Con la base de datos de `.env`, construir `Simulator(scope='es', event_date='2026-11-29', mode='nowcast', path='.')`, `fit_forecast(names=sim.params['names'], max_fc=10, fillna=True)` y llamar a `export_series(sim.model, names, end=sim.as_of, blocks=series_blocks(sim))`. Sin escribir en el bundle.
- [ ] **Step 2:** Comprobar e informar: `first_polls == {'UP': '2023-12-15', 'SALF': '2024-06-06'}` (o las fechas que den los sondeos actuales), y el salto diario máximo de `blocks.mean` de Derecha e Izquierda alrededor de esas fechas (< 0,3 puntos), frente al de la suma de partidos (≈ 2 y 4 puntos).
- [ ] **Step 3:** Si la conexión cuelga (RDS), parar y decirlo; no reintentar en bucle.
