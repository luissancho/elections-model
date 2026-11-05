# Análisis «¿Se copian las encuestadoras en la recta final?» Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a third study, `rebano`, computed from the already-published parts of the bundle, republished with every `update`, and its live page `/analisis/rebano` with the approved mockup's structure, live and conditioned texts and two charts.

**Architecture:** A light package `mtpy/lib/studies/herd/` reads the `cycle`/`series`/`pollsters` parts of every general election, the `polls`/`series` of the latest national run and the `herding` analysis through a `BundleReader`, measures the dispersion of the polls around the published average curve by weeks to the election and the herding of each house in the final window, and `mtpy/lib/publish/herd.py` exports five parts. A new non-chunked `studies` action of the `publish` job republishes the studies flagged `live` and closes the `update` chain. The page follows `/analisis/sesgo-cis`: view models in `pages/herd.py`, a Jinja template, a page script drawing a new `charts/weeks.js` and the matrix module with a ratio scale.

**Tech Stack:** Python 3.11, pandas, numpy, scipy (`chi2`, `binom`), Jinja2, pytest; vanilla ES modules on ECharts 5 tested with `node`.

**Spec:** `docs/superpowers/specs/2026-11-05-web-analisis-rebano-design.md` (approved 2026-11-05; mockup `docs/superpowers/specs/2026-11-05-web-analisis-rebano-mockup.html`).

## Global Constraints

- No database access anywhere in the study: inputs come only from the bundle (`BundleReader(fs, prefix=bundle.PREFIX)`, also in a dry run) and from `data/rebano/params.json`.
- `voto-cera` and `sesgo-cis` keep their parts, schemas, CSV, registration and tests; `tests/test_publish_studies.py` and `tests/test_publish_lean.py` stay green.
- `mtpy.lib.publish` stays light: the study package and `scipy` are imported inside functions (`test_importing_publish_does_not_load_the_study_package` gains `mtpy.lib.studies.herd`); `pages` and `webapi` never import `publish` or the model.
- Code identifiers in English; Spanish only in public texts. Public wording: «sólo» with accent, «encuestadora»/«casa», never «empresa», «actualización» for the model runs (never «publicación»), quotes «»; `assert_public_wording` passes on `/analisis/rebano`.
- Every figure of the page comes from the published parts (never typed into the template); every strong sentence is conditioned as spec D6 says and has a neutral fallback.
- Parameters (spec D4, in `data/rebano/params.json`): `scope "es"`, `since "2008-01-01"`, `max_days 365`, `window_days 56`, `final_days 21`, `local_window 10`, `n_series 4`, `min_polls 3`, `exclude ["CIS"]`, `buckets [[0, 1], [2, 3], [4, 5], [6, 7], [8, 11], [12, 17], [18, 26], [27, 52]]`.
- Day windows (plan ruling, consistent with the week buckets `lo·7 … hi·7+6`): `last` = days `0..window_days−1` (0-55), `before` = days `window_days..max_days` (56-365), `final` = days `0..final_days−1` (0-20). Days = election date − `end_date` (or `date` without it).
- Polls that enter: `sample_size > 0` and `weight > 0`; houses of `exclude` are out of every pooled figure and reported apart.
- Rounding: ratios, errors and means to 2 decimals; p values with three significant figures (`float(f'{p:.3g}')`, as `publish/lean.py`).
- Fluid layout, charts mounted with `mountChart`, every chart module exports its `*Option` for `tests/test_js_charts.py`.
- Git: Luis commits in parallel in the same tree. Before each commit run `git status --short` and `git log -1`, add only the task's files by path, never `git commit -a`, never rewrite history. Commit messages in the repo's style («Área: qué cambia»), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run tests with `python -m pytest -q` from the repo root; JS tests need `node` on PATH (`tests/test_js_charts.py` skips without it).

## Review Focus

1. The current cycle with no poll in the final window yet (the page goes live weeks before the first one): `houses_now` is `{n: 0, k: 0, herd: [], pending: [...]}`, `current.last.ratio` is `None`, the verdict of P3 is «NINGUNA» and the KPI 3 shows «0 de 0» — Task 4 `test_compute_herd_with_an_empty_final_window`, Task 7 `test_herd_page_without_houses_in_the_window`.
2. A poll whose `end_date` falls after the last day of the curve (fieldwork after the backtest anchor, or the run's `as_of`): the residual uses the last day of the curve before it, never a crash — Task 3 `test_curve_level_falls_back_to_the_last_day_before`.
3. A poll with a null share in one of the four main series: that pair is skipped, the poll's other series count — Task 3 `test_residuals_skip_only_the_missing_series`.
4. The election of the current run gets published with results (after the 29-N): `current` is `None`, the cycle enters `cycles` from its `cycle` part with `last_error_*` filled, and the page shows it as a past cycle — Task 4 `test_compute_herd_moves_the_current_cycle_to_the_past_once_published`.
5. The `herding` analysis part missing, or without a row for a house and cycle: `h_cycle` is `None` and the page's «En el ciclo entero» sentence names only the houses that have it — Task 2 `test_read_herding_without_the_part`, Task 7 `test_herd_page_names_only_the_houses_with_a_cycle_ratio`.

---

### Task 1: Study registry, schemas, fixtures and CSV labels

**Files:**
- Modify: `mtpy/lib/bundle.py:79-93` (`STUDIES`, `STUDY_PARTS`, `STUDY_SCHEMAS`, `STUDY_CSV`, `STUDY_CSV_OF`), `:164-172` (`SCHEMAS`), module docstring tree
- Modify: `mtpy/lib/pages/analisis.py:41-61` (`CSV_LABELS`)
- Create: the five fixtures `rebano-meta.json`, `rebano-summary.json`, `rebano-cycles.json`, `rebano-weeks.json`, `rebano-houses.json` in the folder `tests.fakes.bundle_fixture` reads (next to `sesgo-cis-*.json`)
- Modify: `tests/test_bundle_unit.py:84-105` (registry test)
- Test: `tests/test_bundle_unit.py`, `tests/test_webapi_unit.py`, `tests/test_web_api.py`

**Interfaces:**
- Produces: `bundle.STUDIES == ('voto-cera', 'sesgo-cis', 'rebano')`; `STUDY_PARTS['rebano'] == ('meta', 'summary', 'cycles', 'weeks', 'houses')`; `STUDY_SCHEMAS['rebano'][part] == 'rebano-' + part`; `STUDY_CSV['rebano'] == ('cycles', 'weeks', 'houses')`; `STUDY_CSV_OF['rebano'] == {'cycles': ('cycles',), 'weeks': ('weeks',), 'houses': ('houses',)}`; `SCHEMAS` entries: `'rebano-meta': {'study': str, 'title': str, 'subtitle': str, 'scope': str, 'election': (dict, type(None)), 'params': dict, 'inputs': dict, 'sources': list, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'seconds': dict}`, `'rebano-summary': {'n_cycles': int, 'k_narrowed': int, 'narrowed': list, 'mean_before': (float, type(None)), 'mean_last': (float, type(None)), 'mean_final': (float, type(None)), 'k_below_one': int, 'below_one': list, 'sign': (dict, type(None)), 'current': (dict, type(None)), 'houses_now': (dict, type(None)), 'cis': (dict, type(None)), 'accuracy': (dict, type(None))}`, `'rebano-cycles': {'rows': list}`, `'rebano-weeks': {'buckets': list, 'rows': list, 'mean': list}`, `'rebano-houses': {'rows': list}`; `pages.analisis.CSV_LABELS['rebano'] == {'cycles': 'Dispersión por ciclo: resto del ciclo, últimas 8 y últimas 3 semanas', 'weeks': 'Dispersión por ciclo y semanas hasta la elección', 'houses': 'Rebaño por casa en las últimas 8 semanas de cada ciclo'}`.
- Fixture content (the mockup's figures, so the page tests read like the mockup): `summary` with `n_cycles 7, k_narrowed 7, narrowed` the seven labels, `mean_before 1.68, mean_last 1.2, mean_final 1.05, k_below_one 3, below_one ['2008', '2016', '2019 nov.'], sign {n: 7, k: 7, p: 0.00781}, current {label: '2026', before: {k: 72, ratio: 1.06}, last: {k: 24, ratio: 1.06}, final: {k: 0, ratio: null}, cycle: {k: 96, ratio: 1.06}, days_left: 24, rank_before: 2, like_final: true}, houses_now {n: 3, k: 3, herd: [{pollster: 'SocioMétrica', h: 0.36}, {pollster: 'NC Report', h: 0.44}, {pollster: 'Sigma Dos', h: 0.57}], pending: [{pollster: 'GAD3', k: 2}, {pollster: '40dB', k: 1}]}, cis {current: {k: 1, ratio: 10.7}, cycle: {k: 10, ratio: 8.3}, own: {k: 1, h: null, p: null}}, accuracy {n: 20, herd: {k: 12, error_avg: 2.1, error_blocks: 2.6}, rest: {k: 8, error_avg: 2.3, error_blocks: 3.1}}`; `cycles.rows` the eight rows of the mockup table (fields of spec D3, `consensus 'local'` for 2008 and 2011); `weeks` with the eight buckets (`label` «27-52 sem.» … «0-1 sem.»), the rows of the mockup's `CYCLES`/`NOW` arrays and `mean`; `houses.rows` the mockup matrix cells (`h_cycle` only for the 2026 rows and the 2023 ones; `last_error_*` null for 2026); `meta` with `election {date: '2026-11-29', days_left: 24, as_of: '2026-11-05'}` and `params` of the Global Constraints.

- [ ] **Step 1: Write the failing tests**

In `tests/test_bundle_unit.py` extend the registry test (`:84-105`):

```python
assert bundle.STUDIES == ('voto-cera', 'sesgo-cis', 'rebano')
assert bundle.STUDY_PARTS['rebano'] == ('meta', 'summary', 'cycles', 'weeks', 'houses')
assert bundle.STUDY_SCHEMAS['rebano'] == {part: 'rebano-' + part for part in bundle.STUDY_PARTS['rebano']}
assert bundle.STUDY_CSV['rebano'] == ('cycles', 'weeks', 'houses')
assert bundle.path_study('rebano', 'weeks') == 'studies/rebano/weeks.json'
assert bundle.path_study_csv('rebano', 'houses') == 'studies/rebano/csv/houses.csv'
with pytest.raises(ValueError):
    bundle.path_study('rebano', 'matrix')
```

The parametrised `test_study_schemas_validate_the_fixtures` (`:283`) covers the five new fixtures by itself. In `tests/test_web_api.py` add `test_study_routes_serve_rebano_parts_and_csv`: `write_study(fs, study='rebano')`, then `GET /api/v1/studies/rebano/weeks` is 200 with `data.buckets` of length 8, `GET /api/v1/studies/rebano/houses?format=csv` is 200 as `rebano-houses.csv`, and `GET /api/v1/studies/rebano/matrix` is 404.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_web_api.py -q -k "registry or fixtures or rebano"`
Expected: FAIL (`STUDIES` has two entries; fixtures missing).

- [ ] **Step 3: Declare the study and write the fixtures**

Add the entries of the Interfaces block to `mtpy/lib/bundle.py` (constants and `SCHEMAS`; update the docstring tree with `studies/rebano/…`), the five fixture files, and `CSV_LABELS['rebano']` in `pages/analisis.py`. Note `election` of `rebano-meta` is nullable (a dict or `None`): write the schema tuple as `(dict, type(None))`.

- [ ] **Step 4: Run the affected suites**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_web_api.py tests/test_webapi_unit.py tests/test_pages.py -q`
Expected: all PASS (`write_study(fs, study='rebano')` works through the registry; the `/analisis/rebano` page is still 404 because `pages.analisis.STUDIES` has no template for it: fine until Task 7).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/pages/analisis.py tests/test_bundle_unit.py tests/test_web_api.py <the five fixtures>
git commit -m "Estudios: registro, esquemas y fixtures del estudio rebano"
```

---

### Task 2: Parameters and the readers of the inputs (`studies/herd/data.py`)

**Files:**
- Create: `mtpy/lib/studies/herd/__init__.py` (docstring: the herding of the houses in the final stretch, from the published parts), `mtpy/lib/studies/herd/data.py`
- Create: `data/rebano/params.json`; Modify: `data/README.md` (one table row: `rebano/params.json` | Parámetros del estudio del rebaño en la recta final (ámbito, ciclos desde, ventanas en días, consenso local, series principales, mínimo de sondeos por casa, casas aparte, cubos de semanas) | Curación propia.)
- Create: `tests/fakes.py::write_herd_bundle` (append), `tests/test_herd_data.py`
- Test: `tests/test_herd_data.py`, `tests/test_data_files.py` (if it enumerates `data/` files, add the new one)

**Interfaces:**
- Produces: `PARAMS_FILE = 'rebano/params.json'`; `load_params(data: FileSystem) -> dict`; `MissingInput(Exception)` with `.part` (same shape as `studies/lean/data.py`); `Cycle` dataclass (`event_date: str`, `label: str`, `current: bool`, `polls: pd.DataFrame` with `date, pollster_id, pollster, start_date, end_date, sample_size, weight` plus one float column per party, `parties: list[str]`, `curve: Optional[pd.DataFrame]` indexed by `Timestamp` with one column per party, `errors: dict[str, dict]` by pollster with `last_error_avg`, `last_error_blocks` (empty without the `pollsters` part), `inputs: dict`); `read_cycles(reader: BundleReader, params: dict) -> tuple[list[Cycle], dict]` (past cycles ascending by date and `inputs` keyed by date: `{'cycle': generated_at, 'series': generated_at | None, 'pollsters': generated_at | None}` or `{'skipped': 'no cycle part'}`); `is_published(reader, scope, date) -> bool` (an event folder with `results.json`); `read_current(reader: BundleReader, params: dict) -> tuple[Optional[Cycle], dict]` (the `latest` run of `manifest.scopes[scope]`: `None` and `{'skipped': reason}` without manifest, without the scope, or when `is_published(meta.event_date)`; else the `Cycle` with `current=True`, `curve` from the run `series`, `errors` empty, `inputs {'run_id', 'as_of', 'event_date'}`); `read_herding(reader, scope) -> tuple[dict, dict]` (`{(event_date, pollster): {'n', 'ratio', 'p_value'}}` from `analysis/{scope}/herding.json` `events`, and `{'generated_at'}` or `{'skipped': 'no herding part'}`); `cycle_label(event_date, dates) -> str` («2023», «2019 abr.» / «2019 nov.» when two dates share the year, using `MONTHS` of `studies/lean/data.py`: import it).
- `write_herd_bundle(fs) -> dict` in `tests/fakes.py` writes, for scope `es`: the events `2019-11-10` (parts `meta`, `cycle`, `results`, `pollsters`; no `series`) and `2023-07-23` (the same plus `series`), the manifest with `scopes.es.latest = '20261105-120000'` and that run's `meta` (`event_date '2026-11-29'`, `as_of '2026-11-05'`), `polls` and `series`, and `analysis/es/herding.json`. Every poll has `weight 1`, `sample_size 1000` and `start_date = end_date` unless said otherwise; `date = end_date`. Shares are the curve level plus the residual: the curves are constant, PP 33 / PSOE 30 / VOX 13 / SUMAR 12 (2023 and 2019 cycle parts, parties `['PP', 'PSOE', 'VOX', 'SUMAR', 'Cs']` with `Cs 1.5` in every poll) and PP 33 / PSOE 30 / VOX 13 / FA 12 (run, parties `['PP', 'PSOE', 'VOX', 'FA', 'Cs']`). 2023 `series`: dates `2022-07-23..2023-07-17`; run `series`: `2025-11-05..2026-11-05`. Residual vectors `(PP, PSOE, VOX, SUMAR)` of the 2023 polls: GAD3 2023-03-01 `(+2, −2, +2, −2)`, NC Report 2023-04-15 `(+2, −2, −2, +2)`, Sigma Dos 2023-05-20 `(−2, +2, +2, −2)`, GAD3 2023-06-10 `(+1, −1, +1, −1)`, NC Report 2023-06-15 `(+3, −3, −3, +3)`, NC Report 2023-06-25 `(+1, −1, −1, +1)`, GAD3 2023-07-05 `(+2, −2, +1, −1)`, Sigma Dos 2023-07-12 `(−1, +1, +1, −1)`, GAD3 2023-07-17 `(+1, −1, +1, −1)`, NC Report 2023-07-17 `(−2, +2, +2, −2)`; CIS 2023-02-01 `(+3, −3, +1, −1)` and CIS 2023-07-13 `(+4, −4, +2, −2)`, both `sample_size 4000`; `Panel X` 2023-07-10 `(0, 0, 0, 0)` with `sample_size null` and `weight 0.5`; `Celeste Tel` 2023-07-01 `(+1, −1, +1, −1)` with `weight 0`. 2019 polls (no curve): 2019-08-01 GAD3 `PP 36, PSOE 27, VOX 16, SUMAR 9`, NC Report `30, 33, 10, 15`, Sigma Dos `33, 30, 13, 12`; 2019-11-04 GAD3 `34, 29, 14, 11`, NC Report `32, 31, 12, 13`, Sigma Dos `33, 30, 13, 12`, CIS `40, 25, 13, 12` (`sample_size 4000`). Run polls (residuals over the run curve): SocioMétrica 2026-06-01 `(+1, −1, +1, −1)`, Sigma Dos 2026-08-15 `(+1, −1, −1, +1)`, SocioMétrica 2026-10-10, 2026-10-20 and 2026-11-02 all `(+1, −1, +1, −1)`, Sigma Dos 2026-10-25 `(+1, −1, −1, +1)`, CIS 2026-10-30 `(+5, −5, +1, −1)` with `sample_size 4000`. `pollsters` rows (`pollster, last_error_avg, last_error_blocks`): 2023 GAD3 `1.5, −2.0`, NC Report `2.5, 1.0`, Sigma Dos `2.0, 0.5`, CIS `4.0, −6.0`; 2019 GAD3 `1.0, 0.5`, NC Report `3.0, −1.0`, Sigma Dos `2.0, 1.5`. `herding.events`: `{'event_date': '2026-11-29', 'pollster': 'SocioMétrica', 'n': 10, 'ratio': 0.78, 'p_value': 0.12}` and `{'event_date': '2023-07-23', 'pollster': 'GAD3', 'n': 8, 'ratio': 0.61, 'p_value': 0.05}`. Returns `{'dates': ['2019-11-10', '2023-07-23'], 'run_id': '20261105-120000'}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_params_reads_the_repository_file():
    params = data.load_params(FileSystem(os.path.join(ROOT, 'data')))
    assert params['scope'] == 'es' and params['since'] == '2008-01-01' and params['max_days'] == 365
    assert (params['window_days'], params['final_days'], params['local_window']) == (56, 21, 10)
    assert (params['n_series'], params['min_polls'], params['exclude']) == (4, 3, ['CIS'])
    assert params['buckets'] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 11], [12, 17], [18, 26], [27, 52]]

def test_read_cycles_reads_the_published_generals(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs)
    cycles, inputs = data.read_cycles(bundle.BundleReader(fs), PARAMS)
    assert [c.event_date for c in cycles] == ['2019-11-10', '2023-07-23'] and [c.label for c in cycles] == ['2019', '2023']
    assert cycles[0].curve is None and cycles[1].curve.index.min() == pd.Timestamp('2022-07-23')
    assert cycles[1].polls.shape[0] == 14 and cycles[1].errors['GAD3'] == {'last_error_avg': 1.5, 'last_error_blocks': -2.0}
    assert cycles[0].errors['NC Report']['last_error_blocks'] == -1.0
    assert inputs['2023-07-23']['series'] and inputs['2019-11-10']['series'] is None

def test_read_cycles_skips_an_event_without_the_cycle_part(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs)
    fs.remove(bundle.path_event('es', '2019-11-10', 'cycle'))
    cycles, inputs = data.read_cycles(bundle.BundleReader(fs), PARAMS)
    assert [c.event_date for c in cycles] == ['2023-07-23'] and inputs['2019-11-10'] == {'skipped': 'no cycle part'}

def test_read_current_reads_the_latest_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs)
    current, inputs = data.read_current(bundle.BundleReader(fs), PARAMS)
    assert current.current and current.event_date == '2026-11-29' and current.label == '2026'
    assert current.polls.shape[0] == 7 and current.curve.index.max() == pd.Timestamp('2026-11-05') and current.errors == {}
    assert inputs == {'run_id': '20261105-120000', 'as_of': '2026-11-05', 'event_date': '2026-11-29'}

def test_read_current_is_none_without_a_run_or_once_the_election_is_published(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs)
    writer = bundle.BundleWriter(fs)   # the election of the run published with results
    writer.write_json(bundle.path_event('es', '2026-11-29', 'results'), 'event-results', {'n_seats': 350, 'rows': []}, 'es')
    assert data.read_current(bundle.BundleReader(fs), PARAMS) == (None, {'skipped': 'election published'})
    fs.remove(bundle.path_manifest())
    assert data.read_current(bundle.BundleReader(fs), PARAMS) == (None, {'skipped': 'no manifest'})

def test_read_herding_without_the_part(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs)
    rows, inputs = data.read_herding(bundle.BundleReader(fs), 'es')
    assert rows[('2023-07-23', 'GAD3')] == {'n': 8, 'ratio': 0.61, 'p_value': 0.05} and inputs['generated_at']
    fs.remove(bundle.path_analysis('es', 'herding'))
    assert data.read_herding(bundle.BundleReader(fs), 'es') == ({}, {'skipped': 'no herding part'})

def test_cycle_label_adds_the_month_only_when_the_year_repeats():
    dates = ['2019-04-28', '2019-11-10', '2023-07-23']
    assert [data.cycle_label(d, dates) for d in dates] == ['2019 abr.', '2019 nov.', '2023']
```

(`PARAMS` is `load_params` of the repository `data/`; `fs.remove` as `mtpy/core/io.py` names it, check.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_herd_data.py -q`
Expected: FAIL with `ModuleNotFoundError: mtpy.lib.studies.herd`.

- [ ] **Step 3: Write `data/rebano/params.json`, its README row, `write_herd_bundle` and `data.py`**

Signatures of the Interfaces block. `read_cycles` keeps the dates `>= params['since']` of `reader.list_events(scope)` with a `cycle` part; `polls` columns cast: `sample_size` float with `NaN` for null, shares float. `read_current` reads `manifest.scopes[scope]['latest']` through `reader.read_json(bundle.path_manifest())` and the run parts with `bundle.path_part(scope, run_id, part)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_herd_data.py tests/test_data_files.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/__init__.py mtpy/lib/studies/herd/data.py data/rebano/params.json data/README.md tests/fakes.py tests/test_herd_data.py
git commit -m "Estudio rebano: parámetros y lectura de ciclos, run actual y rebaño desde el paquete"
```

---

### Task 3: The measure (`studies/herd/stats.py`)

**Files:**
- Create: `mtpy/lib/studies/herd/stats.py`, `tests/test_herd_stats.py`

**Interfaces:**
- Produces: `main_series(polls: pd.DataFrame, parties: list[str], n_series: int) -> list[str]` (the `n_series` parties with the highest mean share over the polls, ties by `parties` order); `curve_level(curve: pd.DataFrame, name: str, when: pd.Timestamp) -> Optional[float]` (the value at `when`; if `when` is after the last date, the last date; if before the first, `None`); `local_level(polls: pd.DataFrame, index, name: str, window: int, exclude: list[str]) -> Optional[float]` (mean share of the polls of other houses, not in `exclude`, with `end` within ±`window` days; `None` with fewer than 2); `residual_rows(cycle: Cycle, names: list[str], election: pd.Timestamp, params: dict) -> pd.DataFrame` with columns `poll` (row index), `pollster`, `days` (int), `name`, `resid`, `var` (`1e4 · p(1−p) / n`, `p` the level in [0.005, 0.995]), one row per poll with sample and weight > 0 and per series with a finite share and level; `dispersion(rows: pd.DataFrame) -> dict` (`{'k': distinct polls, 'ratio': sqrt(mean(resid²) / mean(var)) rounded to 2, or None with k 0}`); `in_days(rows, lo, hi)` (inclusive filter on `days`); `house_ratio(rows: pd.DataFrame) -> dict` (`{'k', 'h', 'p'}` of one house: per series `ss_obs = Σ(r − r̄)²`, `ss_exp = (1 − 1/k_s) · Σ var`, summed over series, `h = sqrt(ss_obs / ss_exp)`, `p = chi2.cdf(h² · (k − 1), k − 1)`; `h` and `p` `None` with `k < 2`); `sign_test(n: int, k: int) -> Optional[dict]` (`{'n', 'k', 'p': binom.sf(k − 1, n, 0.5)}`, `None` with n 0); `accuracy(rows: list[dict]) -> Optional[dict]` (rows of `houses` with `last_error_avg` not None, split by `h < 1`: `{'n', 'herd': {'k', 'error_avg', 'error_blocks'}, 'rest': {...}}` with means of `last_error_avg` and `abs(last_error_blocks)`; `None` when a group is empty).

- [ ] **Step 1: Write the failing tests**

Hand-computed on the Task 2 bundle (constant curve, `n 1000`: the variances of the four series are `2.211, 2.1, 1.131, 1.056`, mean `1.6245`; with `n 4000`, mean `0.40613`):

```python
def test_dispersion_of_the_2023_windows(herd_2023):   # fixture: residual_rows of the 2023 cycle without the CIS
    rows = herd_2023
    assert stats.dispersion(stats.in_days(rows, 56, 365)) == {'k': 3, 'ratio': 1.57}
    assert stats.dispersion(stats.in_days(rows, 0, 55)) == {'k': 7, 'ratio': 1.31}
    assert stats.dispersion(stats.in_days(rows, 0, 20)) == {'k': 4, 'ratio': 1.14}
    assert stats.dispersion(stats.in_days(rows, 189, 365)) == {'k': 0, 'ratio': None}

def test_polls_without_sample_or_weight_are_out(herd_2023):
    assert set(herd_2023['pollster']) == {'GAD3', 'NC Report', 'Sigma Dos'}   # Panel X and Celeste Tel never enter; the CIS is filtered by the caller

def test_house_ratio_of_gad3_and_nc_report_in_the_final_window(herd_2023):
    last = stats.in_days(herd_2023, 0, 55)
    assert stats.house_ratio(last[last['pollster'] == 'GAD3']) == {'k': 3, 'h': 0.32, 'p': 0.0975}
    assert stats.house_ratio(last[last['pollster'] == 'NC Report']) == {'k': 3, 'h': 1.97, 'p': 0.98}
    assert stats.house_ratio(last[last['pollster'] == 'Sigma Dos']) == {'k': 1, 'h': None, 'p': None}

def test_cis_apart_against_the_curve(herd_2023_cis):   # residual_rows of the CIS alone
    assert stats.dispersion(stats.in_days(herd_2023_cis, 0, 55)) == {'k': 1, 'ratio': 4.96}
    assert stats.dispersion(herd_2023_cis) == {'k': 2, 'ratio': 4.3}

def test_local_consensus_excludes_the_house_and_the_cis(herd_2019):   # the 2019 cycle has no curve
    assert stats.dispersion(stats.in_days(herd_2019, 0, 55)) == {'k': 3, 'ratio': 0.96}
    assert stats.dispersion(stats.in_days(herd_2019, 56, 365)) == {'k': 3, 'ratio': 2.88}

def test_curve_level_falls_back_to_the_last_day_before():
    curve = pd.DataFrame({'PP': [30.0, 31.0]}, index=pd.to_datetime(['2023-07-01', '2023-07-02']))
    assert stats.curve_level(curve, 'PP', pd.Timestamp('2023-07-09')) == 31.0
    assert stats.curve_level(curve, 'PP', pd.Timestamp('2023-06-30')) is None

def test_residuals_skip_only_the_missing_series(cycle_with_a_null_share):   # one poll with VOX null
    rows = stats.residual_rows(cycle_with_a_null_share, ['PP', 'PSOE', 'VOX', 'SUMAR'], ELECTION, PARAMS)
    assert rows.loc[rows['poll'] == 0, 'name'].tolist() == ['PP', 'PSOE', 'SUMAR']

def test_main_series_picks_the_four_largest():
    assert stats.main_series(polls_2023, ['PP', 'PSOE', 'VOX', 'SUMAR', 'Cs'], 4) == ['PP', 'PSOE', 'VOX', 'SUMAR']

def test_sign_test_and_accuracy():
    assert stats.sign_test(7, 7) == {'n': 7, 'k': 7, 'p': 0.00781} and stats.sign_test(2, 2)['p'] == 0.25 and stats.sign_test(0, 0) is None
    rows = [{'h': 0.32, 'last_error_avg': 1.5, 'last_error_blocks': -2.0}, {'h': 1.97, 'last_error_avg': 2.5, 'last_error_blocks': 1.0},
            {'h': 0.5, 'last_error_avg': None, 'last_error_blocks': None}]
    assert stats.accuracy(rows) == {'n': 2, 'herd': {'k': 1, 'error_avg': 1.5, 'error_blocks': 2.0}, 'rest': {'k': 1, 'error_avg': 2.5, 'error_blocks': 1.0}}
    assert stats.accuracy(rows[:1]) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_herd_stats.py -q`
Expected: FAIL with `ImportError` (no `stats`).

- [ ] **Step 3: Implement `stats.py`**

Signatures of the Interfaces block; `scipy.stats.chi2` and `binom` imported inside the functions. The `p` of `house_ratio` and `sign_test` with three significant figures.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_herd_stats.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/stats.py tests/test_herd_stats.py
git commit -m "Estudio rebano: la medida de dispersión frente a la curva, el rebaño por casa y las pruebas"
```

---

### Task 4: The study (`studies/herd/study.py`)

**Files:**
- Create: `mtpy/lib/studies/herd/study.py`, `tests/test_herd_study.py`

**Interfaces:**
- Consumes: Task 2 readers and Task 3 statistics.
- Produces: `HerdStudy` dataclass with `params: dict`, `election: Optional[dict]` (`{date, days_left, as_of}`), `summary: dict`, `cycles: list[dict]`, `weeks: dict` (`buckets`, `rows`, `mean`), `houses: list[dict]`, `inputs: dict` (`cycles` by date, `current`, `herding`), exactly the fields of spec D3; `compute_herd(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> HerdStudy` (raises `MissingInput('events/es')` without past cycles). Rules: every ratio is rounded to 2 decimals only when it enters a row; the means (`mean_before`, `mean_last`, `mean_final`, `weeks.mean`) and the comparisons are computed over the unrounded ratios and rounded last (the synthetic `mean_last` is 1,1353: from the rounded rows it would sit on the 1,135 boundary); `rank_before` = 1 + number of past cycles whose rounded `before.ratio` is lower than the current rounded one; `like_final = current.cycle.ratio <= mean_last` (unrounded); `houses` rows sorted by `event_date` then `h` ascending with `None` last; `pending` = houses not in `exclude` with `1 <= k < min_polls` in the window; `cis` of a cycle: `last`/`cycle` dispersions of the excluded houses' rows and `own` = `house_ratio` of those rows in the window; `weeks.mean` over the past cycles with a ratio in the bucket.

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_herd_on_the_synthetic_bundle(fresh_app, tmp_path):
    study = compute(tmp_path)   # write_herd_bundle + compute_herd(BundleReader(fs), FileSystem(ROOT/data))
    s = study.summary
    assert (s['n_cycles'], s['k_narrowed'], s['narrowed']) == (2, 2, ['2019', '2023'])
    assert (s['mean_before'], s['mean_last'], s['mean_final']) == (2.23, 1.14, 1.05)
    assert (s['k_below_one'], s['below_one'], s['sign']) == (1, ['2019'], {'n': 2, 'k': 2, 'p': 0.25})
    assert s['current'] == {'label': '2026', 'before': {'k': 2, 'ratio': 0.78}, 'last': {'k': 4, 'ratio': 0.78}, 'final': {'k': 0, 'ratio': None},
                            'cycle': {'k': 6, 'ratio': 0.78}, 'days_left': 24, 'rank_before': 1, 'like_final': True}
    assert s['houses_now'] == {'n': 1, 'k': 1, 'herd': [{'pollster': 'SocioMétrica', 'h': 0.0}], 'pending': [{'pollster': 'Sigma Dos', 'k': 1}]}
    assert s['cis'] == {'current': {'k': 1, 'ratio': 5.66}, 'cycle': {'k': 1, 'ratio': 5.66}, 'own': {'k': 1, 'h': None, 'p': None}}
    assert s['accuracy'] == {'n': 2, 'herd': {'k': 1, 'error_avg': 1.5, 'error_blocks': 2.0}, 'rest': {'k': 1, 'error_avg': 2.5, 'error_blocks': 1.0}}
    assert study.election == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05'}

def test_compute_herd_cycle_rows(fresh_app, tmp_path):
    rows = {row['event_date']: row for row in compute(tmp_path).cycles}
    r23 = rows['2023-07-23']
    assert (r23['label'], r23['current'], r23['names'], r23['consensus']) == ('2023', False, ['PP', 'PSOE', 'VOX', 'SUMAR'], 'curve')
    assert (r23['polls'], r23['with_sample'], r23['houses']) == (14, 10, 3)
    assert (r23['before'], r23['last'], r23['final'], r23['change']) == ({'k': 3, 'ratio': 1.57}, {'k': 7, 'ratio': 1.31}, {'k': 4, 'ratio': 1.14}, -0.26)
    assert r23['cis'] == {'last': {'k': 1, 'ratio': 4.96}, 'cycle': {'k': 2, 'ratio': 4.3}, 'own': {'k': 1, 'h': None, 'p': None}}
    assert rows['2019-11-10']['consensus'] == 'local' and rows['2019-11-10']['last'] == {'k': 3, 'ratio': 0.96}
    assert rows['2026-11-29']['current'] and rows['2026-11-29']['names'] == ['PP', 'PSOE', 'VOX', 'FA']

def test_compute_herd_weeks_and_houses(fresh_app, tmp_path):
    study = compute(tmp_path)
    assert [b['label'] for b in study.weeks['buckets']] == ['27-52 sem.', '18-26 sem.', '12-17 sem.', '8-11 sem.', '6-7 sem.', '4-5 sem.', '2-3 sem.', '0-1 sem.']
    w23 = {(r['lo'], r['hi']): (r['k'], r['ratio']) for r in study.weeks['rows'] if r['event_date'] == '2023-07-23'}
    assert w23[(0, 1)] == (3, 1.11) and w23[(4, 5)] == (2, 1.75) and w23[(6, 7)] == (1, 0.78) and w23[(27, 52)] == (0, None)
    mean = {(r['lo'], r['hi']): (r['k'], r['ratio']) for r in study.weeks['mean']}
    assert mean[(0, 1)] == (2, 1.04) and mean[(12, 17)] == (2, 2.23) and mean[(27, 52)] == (0, None)
    houses = [(h['event_date'], h['pollster'], h['k'], h['h'], h['p'], h['h_cycle'], h['last_error_avg']) for h in study.houses]
    assert houses == [('2023-07-23', 'GAD3', 3, 0.32, 0.0975, 0.61, 1.5), ('2023-07-23', 'NC Report', 3, 1.97, 0.98, None, 2.5),
                      ('2026-11-29', 'SocioMétrica', 3, 0.0, 0.0, 0.78, None)]
    assert study.houses[0]['slug'] == 'gad3' and study.houses[0]['cis'] is False

def test_compute_herd_with_an_empty_final_window(fresh_app, tmp_path):
    # the run polls of the window removed: the current cycle has only its two early polls
    study = compute(tmp_path, drop_window=True)
    assert study.summary['current']['last'] == {'k': 0, 'ratio': None}
    assert study.summary['houses_now'] == {'n': 0, 'k': 0, 'herd': [], 'pending': []}
    assert study.summary['cis']['current'] == {'k': 0, 'ratio': None}

def test_compute_herd_without_a_run_and_without_cycles(fresh_app, tmp_path):
    study = compute(tmp_path, without_manifest=True)
    assert study.summary['current'] is None and study.summary['houses_now'] is None and study.election is None
    assert study.inputs['current'] == {'skipped': 'no manifest'} and study.summary['n_cycles'] == 2
    with pytest.raises(MissingInput) as info:
        compute(tmp_path, without_events=True)
    assert info.value.part == 'events/es'

def test_compute_herd_moves_the_current_cycle_to_the_past_once_published(fresh_app, tmp_path):
    # the event 2026-11-29 published with results and a cycle part copied from the run polls
    study = compute(tmp_path, publish_current=True)
    assert study.summary['current'] is None and [r['event_date'] for r in study.cycles][-1] == '2026-11-29'
    assert study.summary['n_cycles'] == 3 and study.cycles[-1]['current'] is False
```

(`compute(tmp_path, **changes)` is a test helper that writes the bundle, applies the change and calls `compute_herd`; `publish_current` writes `events/es/2026-11-29/{meta,cycle,results,pollsters}` with the run's polls and a `pollsters` row for SocioMétrica `2.0, 1.0`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_herd_study.py -q`
Expected: FAIL (`study` module missing).

- [ ] **Step 3: Implement `study.py`**

`compute_herd` reads params (`load_params` updated with the overrides), the cycles, the current run and the herding; for each cycle: the four series, the residual rows (`exclude` apart), the three windows, the week buckets, the houses (`house_ratio` per house with `k >= min_polls`, the excluded houses included with `cis: True`), `h_cycle` from the herding rows, `last_error_*` from `cycle.errors`; then the summary of spec D3.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_herd_study.py tests/test_herd_stats.py tests/test_herd_data.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/study.py tests/test_herd_study.py
git commit -m "Estudio rebano: el estudio entero desde las partes publicadas (ciclos, semanas, casas, resumen)"
```

---

### Task 5: Exporters and the publish registry (`publish/herd.py`)

**Files:**
- Create: `mtpy/lib/publish/herd.py`, `tests/test_publish_herd.py`
- Modify: `mtpy/lib/publish/studies.py:21-27` (`STUDY_TITLES`), `:565-583` (imports, `STUDY_EXPORTERS`, `STUDY_META`, `STUDY_RUNNERS` with the new key `live`), `mtpy/lib/publish/__init__.py` (re-exports)
- Modify: `tests/test_publish_studies.py:186-196` (module list and registry assertions), `tests/test_publish_unit.py:1551` (package module names)

**Interfaces:**
- Produces: `STUDY_TITLES['rebano'] = ('¿Se copian las encuestadoras en la recta final?', <subtitle template>)`, the template with the placeholders `{k}`, `{n}`, `{current}` (the 2026 clause or an empty string) rendered by `export_herd_meta`: «En las {n_words} generales desde {since_year}, los sondeos se han parecido más entre sí cuanto más cerca estaba la elección{final_clause}.{current}» where `n_words` is `k == n` ? «{n}» : «{k} de las {n}», `final_clause` = «: en las últimas tres semanas, tanto como si todas las casas sacaran sus muestras de la misma urna» only when `mean_final <= 1.1`, and `current` = « En {year} ya van así desde el principio, y {k_now} de las {n_now} casas con sondeos suficientes publican cifras que oscilan menos de lo que permite el azar de sus muestras.» only with `current.like_final` and `houses_now.n > 0` (the exact composition is the builder's; these are the conditions). `compute_for_publish(data, writer) -> HerdStudy`; `export_herd_summary(study) -> (data, {})`; `export_herd_cycles(study) -> (data, {'cycles': frame})` (frame flattened: `before_k, before_ratio, last_k, last_ratio, final_k, final_ratio, cis_last_k, cis_last_ratio, cis_cycle_k, cis_cycle_ratio, cis_own_k, cis_own_h, cis_own_p`, `names` joined by `/`); `export_herd_weeks(study) -> (data, {'weeks': frame})` (long rows plus the `mean` rows with `event_date` null and `label 'Media'`); `export_herd_houses(study) -> (data, {'houses': frame})`; `export_herd_meta(study, study_id='rebano', registration=None, checked_at=None, prov=None, seconds=None) -> (data, {})` with `sources` = `HERD_SOURCES` (`/elecciones`, `/casas`, `/metodo/rating#herding`, `/analisis/sesgo-cis`, `/analisis`); `STUDY_RUNNERS['rebano'] = {'compute': compute_for_publish, 'registrable': False, 'live': True}`, `STUDY_RUNNERS['sesgo-cis']['live'] = True`, `STUDY_RUNNERS['voto-cera']['live'] = False`.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_herd_parts_validate(study):   # fixture: compute_herd on write_herd_bundle, as test_publish_lean does
    for part in bundle.STUDY_PARTS['rebano'][1:]:
        data, frames = publish.STUDY_EXPORTERS['rebano'][part](study)
        valid(bundle.STUDY_SCHEMAS['rebano'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['rebano'][part])
    meta, frames = publish.export_herd_meta(study, prov=fake_prov())
    valid('rebano-meta', meta) and frames == {}
    # synthetic bundle: k = n = 2, mean_final 1.05 <= 1.1 (the «misma urna» clause), like_final and one measurable house
    assert meta['subtitle'].startswith('En las 2 generales desde 2008, los sondeos se han parecido más entre sí')
    assert 'misma urna' in meta['subtitle'] and 'En 2026 ya van así desde el principio, y 1 de las 1 casas' in meta['subtitle']
    assert meta['election'] == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05'} and meta['params']['window_days'] == 56
    assert [s['url'] for s in meta['sources']][:2] == ['/elecciones', '/casas']

def test_export_herd_meta_drops_the_conditioned_clauses(fresh_app, tmp_path):
    # final_days 1: no poll ends on election day, mean_final is None and the «misma urna» clause goes; without a
    # manifest there is no current cycle and the 2026 sentence goes too
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs); fs.remove(bundle.path_manifest())
    study = compute_herd(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'final_days': 1})
    meta, _ = publish.export_herd_meta(study, prov=fake_prov())
    assert 'misma urna' not in meta['subtitle'] and 'ya van así' not in meta['subtitle'] and meta['election'] is None

def test_export_herd_frames_follow_the_study(study):
    data, frames = publish.STUDY_EXPORTERS['rebano']['cycles'](study)
    assert list(frames['cycles']['before_ratio']) == [r['before']['ratio'] for r in data['rows']]
    data, frames = publish.STUDY_EXPORTERS['rebano']['weeks'](study)
    assert len(frames['weeks']) == len(data['rows']) + len(data['mean'])
    data, frames = publish.STUDY_EXPORTERS['rebano']['houses'](study)
    assert list(frames['houses']['pollster']) == [h['pollster'] for h in data['rows']]

def test_publish_study_rebano_writes_the_parts_and_the_manifest(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_herd_bundle(fs)
    out = publish.publish_study('rebano', make_writer(fs), prov=fake_prov)
    assert out['study'] == 'rebano' and out['registered_at'] is None and out['checked'] is False
    reader = bundle.BundleReader(fs)
    assert reader.read_json(bundle.path_study('rebano', 'summary'))['data']['k_narrowed'] == 2
    assert reader.read_json(bundle.path_manifest())['data']['studies']['rebano'] == {'published_at': ANY, 'registered_at': None, 'checked': False}
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('rebano', make_writer(fs), register=True, prov=fake_prov)

def test_study_runners_know_which_studies_are_live():
    assert {s: r['live'] for s, r in publish.STUDY_RUNNERS.items()} == {'voto-cera': False, 'sesgo-cis': True, 'rebano': True}
```

In `tests/test_publish_studies.py` add `'mtpy.lib.studies.herd'` to the modules that must not load on `import mtpy.lib.publish`; in `tests/test_publish_unit.py:1551` add `'herd'` to the expected module names.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_herd.py tests/test_publish_studies.py tests/test_publish_unit.py -q -k "herd or runners or study_package or names"`
Expected: FAIL (`KeyError: 'rebano'`, missing module).

- [ ] **Step 3: Implement `publish/herd.py` and register it**

Pattern of `publish/lean.py` (`_rounded`, `_significant`); the subtitle rendered from `study.summary`. `STUDY_RUNNERS` gains `live` on the three studies.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_publish_herd.py tests/test_publish_studies.py tests/test_publish_lean.py tests/test_publish_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish/herd.py mtpy/lib/publish/studies.py mtpy/lib/publish/__init__.py tests/test_publish_herd.py tests/test_publish_studies.py tests/test_publish_unit.py
git commit -m "Estudio rebano: exportadores, meta con la entradilla viva y registro del estudio (clave live)"
```

---

### Task 6: The `studies` action of `publish` and its place in `update`

**Files:**
- Create: `mtpy/pipelines/publish/Studies.py`
- Modify: `mtpy/jobs/Publish.py:89-92` (`WHAT`, `sequence`), `:35-41` and `:106-111` (docstrings), `:225-226` (`select`), `:238-245` (results keyed by study); `mtpy/jobs/Update.py:59` (`sequence['publish']`), class docstring
- Modify: `tests/test_jobs_publish.py`, `tests/test_pipelines_publish.py`, `tests/test_jobs_update.py:60-70`
- Modify: `deploy/README.md` («### Estudios»: the `studies` action and the `rebano` paragraph; the `update` section: the studies step), `docs/web/contrato.md` («Estudios»: `rebano` parts table, routes row `:807`, the `live` flag)

**Interfaces:**
- Produces: `Studies(PublishPipeline)` with `chunked = False`, `label = 'studies'`, `pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}`; `publish_task` calls `publish.publish_study(study, self.writer)` for every `study` of `bundle.STUDIES` with `publish.STUDY_RUNNERS[study]['live']`, in that order, collecting `self.result = {'status': 'published' | 'failed', 'studies': {study: {'status', 'seconds' | 'reason'}}}` (`failed` when any study failed; a study's exception never raises; `register` is never passed); `line_of` → `'studies: rebano published in 3 s, sesgo-cis published in 2 s'` or `'studies: sesgo-cis failed (KeyError: ...)'` per study joined by `, `; `failure_name` → `'studies'`. `Publish.WHAT = ('forecast', 'analysis', 'event', 'backtest', 'study', 'studies', 'parties', 'manifest', 'point', 'unpublish')` (same order in `sequence`), results keyed `'studies'`; `Update.sequence = {'load': ['polls'], 'compute': ['compute'], 'publish': ['forecast', 'analysis', 'parties', 'studies']}`, results gain `'studies'`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipelines_publish.py`:

```python
def test_studies_publishes_the_live_studies_in_order(fresh_app, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(publish, 'publish_study', lambda study, writer, **kw: (calls.append(study) or {'study': study, 'registered_at': None, 'checked': False, 'seconds': {'total': 2.0}}))
    p = Studies(); p.run('publish', chunk=None, position=(1, 1), fs=FileSystem(str(tmp_path)), verbose=0)
    assert calls == ['sesgo-cis', 'rebano'] and p.result['status'] == 'published'
    assert p.line == 'studies: sesgo-cis published in 2 s, rebano published in 2 s'

def test_studies_goes_on_after_a_failed_study(fresh_app, tmp_path, monkeypatch):
    def fake(study, writer, **kw):
        if study == 'sesgo-cis':
            raise KeyError('x')
        return {'study': study, 'registered_at': None, 'checked': False, 'seconds': {'total': 1.0}}
    monkeypatch.setattr(publish, 'publish_study', fake)
    p = Studies(); p.run('publish', chunk=None, position=(1, 1), fs=FileSystem(str(tmp_path)), verbose=0)
    assert p.result['status'] == 'failed' and p.result['studies']['rebano']['status'] == 'published'
    assert p.line == "studies: sesgo-cis failed (KeyError: 'x'), rebano published in 1 s" and p.failure_name(None) == 'studies'
```

In `tests/test_jobs_publish.py` add `test_publish_studies_action_runs_once_and_needs_no_study_slug`: `Publish().run(what=['studies'], fs=fs)` with `publish_study` patched returns `{'studies': {'status': 'published', ...}}`, does not resolve scopes (pattern of `test_manifest_parties_and_study_do_not_resolve_scopes`), and `what=['analysis', 'studies']` runs `studies` after `analysis` (order of the printed headers). In `tests/test_jobs_update.py::test_update_runs_the_three_groups_on_es_with_one_alert` patch `publish.publish_study` as above and assert `'studies: ' in out`, `set(results) == {'es', 'parties', 'studies'}` and that `studies` runs after `parties` in `out`; add `test_update_publishes_the_catalogue_even_when_a_study_fails`: `publish_study` raising for every study, default `stop_on_error`, the job raises `RuntimeError` at the end but `results['parties']['status'] == 'published'`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_jobs_update.py -q -k studies`
Expected: FAIL (`ImportError: Studies`, unknown what).

- [ ] **Step 3: Implement `Studies.py`, wire `Publish` and `Update`, write the docs**

`Studies` from the pattern of `Study.py` (progress `start_progress('studies', (1, 1))`). In `Publish.run`, `studies` needs no `study` argument and is not chunked. In `deploy/README.md` add after the «Sesgo del CIS» paragraph: «**Rebaño en la recta final.** `{"what":["study"],"study":"rebano"}` calcula «¿Se copian las encuestadoras en la recta final?» (`mtpy/lib/studies/herd/`) desde los sondeos de cada ciclo de las generales (`cycle`), las curvas publicadas del promedio (`series` de la elección y del run `latest`), los errores por casa (`pollsters`) y el rebaño del análisis nacional (`herding`); sin base de datos ni registro. **`studies`** republica, una vez por job y sin ámbito, los estudios vivos (`sesgo-cis` y `rebano`, los de `live` en `publish.STUDY_RUNNERS`), y es el último paso de `update`: la tarea diaria recalcula la página con el run del día.» plus the commands `S3_BUCKET= python job.py publish '{"what":["study"],"study":"rebano"}'` and `'{"what":["analysis","studies"],"scopes":["es"]}'`. In `docs/web/contrato.md` add the `### rebano` table (five parts with their required keys of Task 1) and the routes row.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_jobs_update.py tests/test_worker_unit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/pipelines/publish/Studies.py mtpy/jobs/Publish.py mtpy/jobs/Update.py tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_jobs_update.py deploy/README.md docs/web/contrato.md
git commit -m "Publish: acción studies (estudios vivos, una vez por job), último paso de update; documentación del estudio rebano"
```

---

### Task 7: The page `/analisis/rebano` (view models, context, template)

**Files:**
- Create: `mtpy/lib/pages/herd.py`, `web/templates/analisis_rebano.html`
- Modify: `mtpy/lib/pages/analisis.py:22-24` (`STUDIES`), `:19` (import), `:652-653` (`CONTEXTS`, `herd_context` next to `lean_context`)
- Modify: `tests/test_pages.py` (package layout list; new tests next to `test_lean_page_structure_and_wording` `:5503`)

**Interfaces:**
- Consumes: the five parts through `study_part('rebano', part)`; `pages.filters` (`fmt_num`, `fmt_int`, `fmt_prob`, `delta_text`); `pages.slugs.slugify`/`PROFILE_PATH` for the house links; `_csv_links('rebano')`, `_published_at('rebano')`.
- Produces in `pages/herd.py`: `HERD = 'rebano'`; `verdict_rows(summary: Optional[dict]) -> list` of three `{'label', 'tone', 'question'}`: P1 «¿Se acercan los sondeos cuando llega la elección?» → `('SÍ', 'q1')` when `k_narrowed == n_cycles > 0`, else `(f'EN {k} DE {n}', 'q2')`; P2 «¿Más de lo que permite el azar de sus muestras?» → `('SÍ', 'q1')` if `k_below_one == n_cycles > 0`, `('NO', 'q3')` if `k_below_one == 0`, else `(f'EN {k} DE {n}', 'q2')`; P3 «¿Quién va en rebaño en {year}?» (year of `meta.election.date`, «este año» without it) → `('NINGUNA', 'qb')` with `houses_now.k == 0` or no run, `('1 CASA', 'qb')`, `(f'{k} CASAS', 'qb')`; `kpis(summary, meta) -> list` of three `m.kpis` rows (labels of spec D6; KPI 2 and 3 fall back to the last past cycle without a run, label «Dispersión de {label}, últimas 8 semanas»); `tests(summary) -> list` of the three P2 boxes (`{'key', 'label', 'value', 'text', 'hi'}`; box 3 «1 entre {round(1/p)}» or «menos de 1 entre 1.000» when `p < 1e-3`, «–» without `sign`); `cis_rows(cycles) -> list`; `houses_now_rows(summary, houses) -> list` (the current cycle's `houses` rows with `href`, `bar` width = `min(1, max(0, 1 − h)) · 100`); `pending_text(summary) -> Optional[str]` («GAD3 (2), 40dB (1)»); `cycle_ratio_text(houses, summary) -> Optional[str]` (the «En el ciclo entero …» sentence over the current houses with `h_cycle`, `None` when none has it); `matrix(houses, cycles) -> dict` (`{'events': [{'label', 'short', 'current'}], 'houses': [str], 'cells': [[x, y, h]]}`, rows ordered as spec D6, columns in `cycles` order; `short` = the label); `repeaters(houses) -> list[str]` (houses with two or more cycles with `h < 1`, by count desc then name); `accuracy_text(summary) -> Optional[str]`; `island(parts) -> dict` (`{'weeks': parts['weeks'], 'matrix': matrix(...)}`). In `pages/analisis.py`: `herd_context(state) -> dict` with `study_meta, summary, cycles, verdicts, kpis, tests, cis_rows, houses_now, pending, cycle_ratio, matrix_rows (for the no-JS table), repeaters, accuracy, csv_links, sources, published_at, initial` and the template flags of spec D6 (`like_final`, `all_narrowed`, `final_like_chance = mean_final <= 1.1`, `veda_date = election.date − 6 days`, `has_run`).
- Template `analisis_rebano.html`: the structure of spec D6 with the mockup texts; charts `#weeks-chart` (`class="chart"`) and `#matrix-chart`; `data-initial` island; `m.card_image_button` per card via a `shot(slug)` macro as in `analisis_sesgo_cis.html`; page script `/dist/js/pages/analisis-rebano.js`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pages.py` (helpers `publish_study(fs, study='rebano', meta=None, parts=None)`, `call`, `html`, `visible_text` as the lean tests use them; if the file's `publish_study` does not take `meta`/`parts`, use `tests.test_webapi_unit.write_study`, which does):

```python
def test_herd_page_structure_and_wording(api, fresh_app):
    publish_study(fresh_app.fs, study='rebano')
    status, _, body = call(api, '/analisis/rebano'); page = html(body)
    assert status == 200 and '<h1 class="study-title">¿Se copian las encuestadoras en la recta final?</h1>' in page
    assert '/dist/js/pages/analisis-rebano.js' in page and 'id="initial-data"' in page
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page) == ['SÍ', 'EN 3 DE 7', '3 CASAS']
    assert page.count('<div class="kpi">') == 3 and '7 de 7' in page and 'Casas en rebaño en las últimas 8 semanas' in page
    for chart in ('weeks-chart', 'matrix-chart'):
        assert re.search(r'<div class="chart[^"]*" id="%s" role="img" aria-label="[^"]+"></div>' % chart, page), chart
    assert page.count('class="pill brand" href="/api/v1/studies/rebano/') == 3
    assert 'ya como una recta final' in page and 'misma urna' in page and 'hasta la veda del 23 de noviembre' in page
    assert 'SocioMétrica' in page and '/casas/sociometrica' in page and 'aún sin sondeos suficientes' in page
    assert 'NC Report' in visible_text(page) and 'Las de siempre' in page
    assert_public_wording(page, '/analisis/rebano')
    assert 'empresa' not in visible_text(page).lower() and 'publicación' not in visible_text(page).lower() and 'max-width' not in page

def test_herd_page_conditions_follow_the_figures(api, fresh_app):
    summary = fixture_data('rebano-summary') | {'k_narrowed': 5, 'narrowed': ['2008', '2011', '2015', '2016', '2023'], 'mean_final': 1.3,
                                                 'k_below_one': 0, 'below_one': [], 'current': fixture_data('rebano-summary')['current'] | {'like_final': False, 'rank_before': 3}}
    publish_study(fresh_app.fs, study='rebano', parts={'summary': summary})
    page = html(call(api, '/analisis/rebano')[2])
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page) == ['EN 5 DE 7', 'NO', '3 CASAS']
    assert 'En 5 de los 7 ciclos' in page and 'En los 7 ciclos' not in page
    assert 'misma urna' not in page and 'ya como una recta final' not in page and '3.º más bajo' in page

def test_herd_page_without_houses_in_the_window(api, fresh_app):
    summary = fixture_data('rebano-summary') | {'houses_now': {'n': 0, 'k': 0, 'herd': [], 'pending': [{'pollster': 'GAD3', 'k': 2}]}}
    publish_study(fresh_app.fs, study='rebano', parts={'summary': summary})
    page = html(call(api, '/analisis/rebano')[2])
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page)[2] == 'NINGUNA' and '0 de 0' in page and 'GAD3 (2)' in page

def test_herd_page_without_a_run_speaks_of_the_past_cycles(api, fresh_app):
    summary = fixture_data('rebano-summary') | {'current': None, 'houses_now': None, 'cis': None}
    publish_study(fresh_app.fs, study='rebano', meta={'election': None}, parts={'summary': summary})
    status, _, body = call(api, '/analisis/rebano'); page = html(body)
    assert status == 200 and 'Evidencia histórica · 7 generales' in page and 'a 24 días' not in page

def test_herd_page_names_only_the_houses_with_a_cycle_ratio(api, fresh_app):
    houses = fixture_data('rebano-houses')
    for row in houses['rows']:
        if row['pollster'] == 'NC Report':
            row['h_cycle'] = None
    publish_study(fresh_app.fs, study='rebano', parts={'houses': houses})
    text = visible_text(html(call(api, '/analisis/rebano')[2]))
    sentence = text[text.index('En el ciclo entero'):][:200]
    assert 'SocioMétrica' in sentence and 'NC Report' not in sentence

def test_analisis_index_lists_rebano_when_published(api, fresh_app):
    publish_study(fresh_app.fs, study='rebano')
    page = html(call(api, '/analisis')[2])
    assert '/analisis/rebano' in page and '¿Se copian las encuestadoras en la recta final?' in page
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pages.py -q -k "herd or rebano"`
Expected: FAIL (404: no template registered).

- [ ] **Step 3: Write `pages/herd.py`, register the study, write the template**

Texts from the mockup v1 with the live variables and the conditions of spec D6; «Cómo leerlo» (`m.how`) in P1 and P3; the no-JS matrix as a table (`matrix_rows`) under the chart container, as the mockup draws it, with the six shade classes `h1..h6` by `h` (< 0,5 · 0,5-0,7 · 0,7-0,9 · 0,9-1,2 · 1,2-1,6 · > 1,6). Add `'herd'` to `test_pages_package_layout`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pages.py -q`
Expected: PASS (`assert_public_wording` included).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/herd.py mtpy/lib/pages/analisis.py web/templates/analisis_rebano.html tests/test_pages.py
git commit -m "Web: página /analisis/rebano con textos vivos condicionados, tablas y la isla de los gráficos"
```

---

### Task 8: The charts (`charts/weeks.js`, the ratio scale of `matrix.js`) and the page script

**Files:**
- Create: `web/dist/js/charts/weeks.js`, `web/dist/js/pages/analisis-rebano.js`
- Modify: `web/dist/js/charts/matrix.js:54-105` (`matrixOption` options), `tests/test_js_charts.py` (new tests next to `:825`)
- Modify: `web/dist/css/site.css` only if the page needs a rule the lean page does not have (the `.now` row, the six `h*` shades, the legend)

**Interfaces:**
- Produces: `weeksOption({buckets, rows, mean}, {current = '#1d4f86', mean: meanColor = THEME.textColor, reference = 1, yMax = 3} = {}) -> option` in `charts/weeks.js`: category x axis of `buckets[].label` (name «semanas antes de la elección»), y axis `0..max(yMax, ceil(max ratio · 2) / 2)` step 0.5 with `fmtNum(x, 2)` labels; one `line` series per past cycle (grey `THEME.mutedColor` shades, width 1.4, `connectNulls: true`, symbol 4), the `Media de los {n}` dashed black series (`type: 'dashed'`, width 2.5) carrying `markLine` at `reference` (red `#ce3b3b`, label «lo que permite el azar de las muestras») and `markArea` `0..reference` (`rgba(206,59,59,.06)`), and the current cycle series (name `rows` label of the current, width 3.5, symbol 7, `endLabel` «{label} · {ratio}»); tooltip by axis listing every series with `fmtNum(v, 2)`; `renderWeeks(el, data, opts)`. `matrixOption(data, {cap, positive, negative, center = null, caps = {below: 0.5, above: 1}, neutral = [0.9, 1.2], label = false, cellText = null})`: with `center`, the shade is red (`positive`) below `center` with alpha by `(center − value) / caps.below`, blue (`negative`) above by `(value − center) / caps.above`, and `#f4f5f8` inside `neutral`; `label` writes `fmtNum(value, 2)` inside the cell; `cellText(house, event, value)` replaces the tooltip text. Page script `analisis-rebano.js`: `drawWeeks(data.weeks)` into `#weeks-chart`, `drawMatrix(data.matrix)` into `#matrix-chart` with `renderMatrix(el, matrix, {center: 1, label: true, cellText: (house, event, h) => \`${house} · ${event.label}: rebaño ${fmtNum(h, 2)}\`})`, `mountWhenReady(paint)`, `wireCardImages(document, showError)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_js_charts.py` (the `run_js` helper and module aliases of the file; bind `wk` to `charts/weeks.js`):

```python
def test_weeks_option_draws_the_cycles_the_mean_and_the_current_one_over_the_chance_line():
    data = {'buckets': [{'lo': 27, 'hi': 52, 'label': '27-52 sem.'}, {'lo': 0, 'hi': 1, 'label': '0-1 sem.'}],
            'rows': [{'event_date': '2023-07-23', 'label': '2023', 'current': False, 'lo': 27, 'hi': 52, 'k': 1, 'ratio': 1.31},
                     {'event_date': '2023-07-23', 'label': '2023', 'current': False, 'lo': 0, 'hi': 1, 'k': 3, 'ratio': 1.11},
                     {'event_date': '2026-11-29', 'label': '2026', 'current': True, 'lo': 27, 'hi': 52, 'k': 2, 'ratio': 1.01},
                     {'event_date': '2026-11-29', 'label': '2026', 'current': True, 'lo': 0, 'hi': 1, 'k': 0, 'ratio': None}],
            'mean': [{'lo': 27, 'hi': 52, 'k': 1, 'ratio': 1.31}, {'lo': 0, 'hi': 1, 'k': 1, 'ratio': 1.11}]}
    option = run_js('wk.weeksOption(%s)' % json.dumps(data))
    names = [s['name'] for s in option['series']]
    assert names == ['2023', 'Media de los 1', '2026'] and option['xAxis']['data'] == ['27-52 sem.', '0-1 sem.']
    assert option['series'][2]['data'] == [1.01, None] and option['series'][2]['lineStyle']['width'] == 3.5
    assert option['series'][1]['markLine']['data'][0]['yAxis'] == 1 and option['series'][1]['markArea']['data'][0][1]['yAxis'] == 1
    assert option['yAxis']['max'] == 3 and option['yAxis']['interval'] == 0.5

def test_matrix_option_with_a_centre_shades_around_one_and_labels_the_cells():
    option = run_js('mx.matrixOption({events: [{short: "2023", label: "2023"}], houses: ["GAD3", "NC Report", "IMOP"], cells: [[0, 0, 0.32], [0, 1, 1.97], [0, 2, 1.0]]}, {center: 1, label: true})')
    colors = [cell['itemStyle']['color'] for cell in option['series'][0]['data']]
    assert colors[0].startswith('rgba(214,39,40') and colors[1].startswith('rgba(29,79,134') and colors[2] == '#f4f5f8'
    assert option['series'][0]['label']['show'] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_js_charts.py -q -k "weeks or centre"`
Expected: FAIL (module missing; `center` ignored).

- [ ] **Step 3: Implement `weeks.js`, the matrix options and the page script**

Keep `matrixOption` byte-compatible without `center` (the lean page and its test at `:825` unchanged). Check the page with the fixture bundle in a headless browser as the previous studies did (`check_app.py` + `check.mjs` of the scratchpad of 2026-10-26, or `node --check` plus the SSR of `tests/test_js_charts.py`); note what could not be verified in the task report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_js_charts.py tests/test_js_card_image.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/weeks.js web/dist/js/charts/matrix.js web/dist/js/pages/analisis-rebano.js web/dist/css/site.css tests/test_js_charts.py
git commit -m "Web: gráfico de dispersión por semanas, matriz con escala en torno a 1 y script de /analisis/rebano"
```

---

### Task 9: Método link, closing state, memory and the full suite

**Files:**
- Modify: `mtpy/lib/pages/metodo.py` (context flag `herd_published`, next to `lean_published` `:777-819`), `web/templates/metodo_rating.html:605-610` (the closing sentence of «El rebaño, aparte»)
- Modify: `docs/superpowers/specs/2026-11-05-web-analisis-rebano-design.md` («Cierre» section), `tests/test_pages.py` (one assertion on `/metodo/rating`)
- Test: the whole suite

- [ ] **Step 1: Write the failing test**

In `tests/test_pages.py`, next to the `/metodo/rating` tests (`:3061`): with `publish_study(fresh_app.fs, study='rebano')` the page contains `<a href="/analisis/rebano">¿Se copian las encuestadoras en la recta final?</a>` inside the `herding` card; without the study it does not contain `/analisis/rebano`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_pages.py -q -k "rating and rebano"`
Expected: FAIL.

- [ ] **Step 3: Add the flag and the sentence**

`context['herd_published'] = bundle_part(site().study, 'rebano', 'meta') is not None`; in the template, after «Los detalles están en …»: `{% if herd_published %}<p>Cómo se acercan las encuestadoras en la recta final, ciclo a ciclo y en vivo, está en <a href="/analisis/rebano">¿Se copian las encuestadoras en la recta final?</a>.</p>{% endif %}`.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (the count before this plan on dev was 2055 passed; report the new total).

- [ ] **Step 5: Write the closing state and commit**

Append to the spec a «Cierre (fecha)» section: what exists (files per task), the rulings (day windows 0-55/56-365/0-20, `charts/weeks.js` instead of extending `line.js`, the subtitle composition, `studies` after `parties`), what was not verified (browser, S3, ECS) and Luis's list: `python job.py publish '{"what":["study"],"study":"rebano"}'` against his S3 (after the daily `update` has run at least once with this code, or `publish '{"what":["analysis","studies"],"scopes":["es"]}'`), walk `/analisis/rebano` (desktop and phone: the matrix scroll, the download buttons), `/analisis`, `/metodo/rating`, deploy the image so the ECS `update` task republishes the study daily, and after the 29-N: `publish event` of `es/2026-11-29` then `publish '{"what":["studies"]}'`.

```bash
git add mtpy/lib/pages/metodo.py web/templates/metodo_rating.html tests/test_pages.py docs/superpowers/specs/2026-11-05-web-analisis-rebano-design.md
git commit -m "Método: enlace al análisis del rebaño; cierre de la spec del estudio rebano"
```
