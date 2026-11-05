# Análisis «El mercado de la fruta en Andorra» Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a study `andorra` (the polls published abroad during the five-day ban, «clandestinos», against the last legal poll of each house and against the frozen average) computed from the published parts of the bundle, with two new inputs the existing jobs produce (an event part `ban`, a backtest table `ban.csv`), and its page `/analisis/andorra` with the approved mockup v2.

**Architecture:** `publish_event` gains a part `ban` (the `wban` polls of a past election with their stored errors, plus the last legal poll of each of those houses); the backtest gains an optional extra case per election at horizon 0 with the ban polls inside, written to `ban.csv` and exported into the event's `model` part under the key `ban`, while the `Forecaster` gets an explicit `drop_ctypes` filter. A light package `mtpy/lib/studies/ban/` reads events of every scope through a `BundleReader`, pairs each house's last clandestine poll with its last legal one, builds the clandestine consensus and the party cells of the national elections with a model, and `mtpy/lib/publish/ban.py` exports seven parts. The page follows `/analisis/sesgo-cis`: view models in `pages/ban.py`, a Jinja template, a page script drawing a new `charts/dumbbell.js`.

**Tech Stack:** Python 3.11, pandas, numpy, scipy (`binomtest`), Jinja2, pytest; vanilla ES modules on ECharts 5 tested with `node`.

**Spec:** `docs/superpowers/specs/2026-11-05-web-analisis-andorra-design.md` (approved 2026-11-05; mockup `docs/superpowers/specs/2026-11-05-web-analisis-andorra-mockup.html`, v2).

## Global Constraints

- No database access in the study or the page: inputs come only from the bundle (`BundleReader(fs, prefix=bundle.PREFIX)`, also in a dry run) and from `data/andorra/params.json`.
- The legal guard (spec D6): the part `ban` is written only by `publish_event`, which refuses an election without official results; the study never reads the `polls` of a run; the card «El 29-N» shows nothing of the current election until it is published with results. `WEB_DROP_CTYPES` and `Computer.drop_ctypes` do not change.
- The backtest published today does not move: `ban.csv` never enters `metrics`, `by_horizon`, `calibration`, `shares` or `series`; horizon 0 is never added to `model.horizons` or `meta.horizons` of an event; with `ban=False` the results of `run_backtest` are byte-identical to today's.
- `voto-cera`, `sesgo-cis` (and `rebano` if it has landed) keep their parts, schemas, CSV, registration and tests. `bundle.STUDIES` gains `'andorra'` **at the end of the tuple as it stands** when the task runs.
- `mtpy.lib.publish` stays light: the study package and `scipy` are imported inside functions (`test_importing_publish_does_not_load_the_study_package` gains `mtpy.lib.studies.ban`); `pages` and `webapi` never import `publish` or the model.
- Code identifiers in English; Spanish only in public texts. Public wording: «sólo» with accent, «encuestadora»/«casa», never «empresa», «actualización» for the model runs, quotes «» (also «clandestinos»); the polls are «sondeos clandestinos» (never «de la veda»), their mean «consenso clandestino», «veda» names only the period; `assert_public_wording` passes on `/analisis/andorra`.
- Every figure of the page comes from the published parts (never typed into the template); every strong sentence is conditioned as spec D7 says and has a neutral fallback. The four paragraphs of the header («La ley», «Por qué existen», «Cómo funcionan», «Qué hacemos con ellos») and the three sources are fixed template text copied from the mockup v2.
- Parameters (spec D4, `data/andorra/params.json`): `national "es"`, `since "1996-01-01"`, `chart_since "2015-01-01"`, `max_days 42`, `frozen_horizon 6`, `min_pairs 3`, `big_sample 5000`, `exclude_mtypes_pairs ["aggr"]`, `exclude_mtypes_consensus ["aggr", "online"]`, `half 0.5`.
- Measures (spec D4): a poll's error is its stored `error_avg`; a pair is a house's last clandestine poll (max `date`, last row on ties) with its row of `ban.legal` when `legal.days <= max_days`, `d = error_ban − error_legal`, `closer = d < 0`, `big = sample_size >= big_sample`; the sign test is `binomtest(k, n − ties, 0.5, alternative='greater').pvalue`; the consensus is the simple mean per main party of the last clandestine poll of each house not in `exclude_mtypes_consensus`; cells use `|consensus − official| < |frozen − official|` (closer), `sign(move) == sign(gap)` with `|move| > 0.05` (direction), `sign(consensus − official) != sign(frozen − official)` (crossed), `closer and (d0 − d1) / d0 >= half` (half).
- Rounding: errors, differences and means to 2 decimals; vote shares to 1; p values with three significant figures (`float(f'{p:.3g}')`, as `publish/lean.py`). Dates ISO.
- Fluid layout, charts mounted with `mountChart`, every chart module exports its `*Option` for `tests/test_js_charts.py`.
- Git: Luis commits in parallel in the same tree. Before each commit run `git status --short` and `git log -1`, add only the task's files by path, never `git commit -a`, never rewrite history. Commit messages in the repo's style («Área: qué cambia»), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run tests with `python -m pytest -q` from the repo root; JS tests need `node` on PATH (`tests/test_js_charts.py` skips without it). Never run a job against the database or S3 from a task.

## Review Focus

1. An election whose `ban` part has polls but an empty `legal` (no house has a legal poll) or whose houses' legal polls are older than `max_days`: it counts in `n_ban` and `elections`, forms no pair, and the page's «Por elección» table shows it with «–» — Task 6 `test_pairs_skip_a_legal_poll_older_than_the_window`, Task 8 `test_andorra_page_shows_an_election_without_pairs`.
2. The folder of the current election exists with `meta` and `ban` but without `results` (a dry run, or a half publication): the study ignores it, `meta.next.open` is false and the card stays closed — Task 6 `test_compute_ban_ignores_an_election_without_results`, Task 8 `test_andorra_page_keeps_the_card_closed_without_results`.
3. A clandestine poll with a null share in a main party: the consensus of that party averages the polls that have it, the cell is still computed, and a poll with no main party at all does not count in the consensus — Task 6 `test_consensus_averages_only_the_polls_with_the_party`.
4. The `model` part with `ban` rows but no row at the frozen horizon (a backtest run with other horizons): the election has no model block (`model: null`), no cells, and `error_with` is not reported alone — Task 6 `test_read_event_without_the_frozen_horizon_has_no_model`.
5. No national election with a model at all (every `model` part empty): `summary.elections` is `None`, `cells.rows` is empty, the page renders P2 without a verdict and the KPI 1 becomes «Elecciones con sondeos clandestinos» — Task 6 `test_compute_ban_without_any_model`, Task 8 `test_andorra_page_without_a_model_has_no_verdict_in_p2`.

---

### Task 1: Registry, schemas, fixtures and CSV labels

**Files:**
- Modify: `mtpy/lib/bundle.py:66-93` (`EVENT_PARTS`, `EVENT_CSV`, `STUDIES`, `STUDY_PARTS`, `STUDY_SCHEMAS`, `STUDY_CSV`, `STUDY_CSV_OF`), `:141-172` (`SCHEMAS`), module docstring tree
- Modify: `mtpy/lib/pages/analisis.py:41-61` (`CSV_LABELS`)
- Create: `tests/fixtures/bundle/andorra-meta.json`, `andorra-summary.json`, `andorra-pairs.json`, `andorra-elections.json`, `andorra-cells.json`, `andorra-houses.json`, `andorra-latest.json`
- Modify: `tests/test_bundle_unit.py:84-105` (registry test; the parametrised validation at `:283` picks the new fixtures up by itself)
- Test: `tests/test_bundle_unit.py`, `tests/test_webapi_unit.py`, `tests/test_web_api.py`

**Interfaces:**
- Produces: `bundle.EVENT_PARTS == ('meta', 'pollsters', 'polls', 'cycle', 'ban', 'results', 'model', 'series')`; `bundle.EVENT_CSV` gains `'ban'`, `'ban-legal'` (after `'cycle'`) and `'model-ban'` (after `'model-blocks'`); `SCHEMAS['event-ban'] = {'parties': list, 'columns': list, 'polls': list, 'legal': list}`; `SCHEMAS['event-model']` unchanged; `bundle.STUDIES` ends with `'andorra'`; `STUDY_PARTS['andorra'] == ('meta', 'summary', 'pairs', 'elections', 'cells', 'houses', 'latest')`; `STUDY_SCHEMAS['andorra'][part] == 'andorra-' + part`; `STUDY_CSV['andorra'] == ('pairs', 'elections', 'cells', 'houses', 'latest-polls')`; `STUDY_CSV_OF['andorra'] == {'pairs': ('pairs',), 'elections': ('elections',), 'cells': ('cells',), 'houses': ('houses',), 'latest': ('latest-polls',)}`.
- `SCHEMAS` entries: `'andorra-meta': {'study': str, 'title': str, 'subtitle': str, 'params': dict, 'inputs': dict, 'next': (dict, type(None)), 'sources': list, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'seconds': dict}`; `'andorra-summary': {'pairs': dict, 'elections': (dict, type(None)), 'cells': dict, 'big': dict, 'exception': (dict, type(None))}`; `'andorra-pairs': {'rows': list}`; `'andorra-elections': {'rows': list, 'period': dict}`; `'andorra-cells': {'rows': list}`; `'andorra-houses': {'rows': list, 'big_rows': list}`; `'andorra-latest': {'scope': str, 'event_date': str, 'label': str, 'parties': list, 'polls': list, 'consensus': (dict, type(None)), 'frozen': (dict, type(None)), 'official': dict, 'error': dict}`.
- `pages.analisis.CSV_LABELS['andorra'] == {'pairs': 'Pares casa a casa: último legal frente a sondeo clandestino', 'elections': 'Resumen por elección', 'cells': 'Cifra a cifra: promedio congelado, consenso clandestino y resultado', 'houses': 'Casas y sondeos de la jornada', 'latest-polls': 'Sondeos clandestinos de la última elección'}`.
- Fixture content = the figures of the mockup v2, so the page tests read like the mockup: `summary.pairs {n: 116, k: 81, ties: 0, share: 0.7, mean_d: -0.36, p: 1.17e-05, generals: {n: 33, k: 24, mean_d: -0.5, n_events: 10}, regional: {n: 83, k: 57, mean_d: -0.3, n_events: 40}, by_scope: [{scope: 'es', n: 33, k: 24, mean_d: -0.5}, {scope: 'es-ct', n: 17, k: 6, mean_d: -0.03}, {scope: 'es-an', n: 9, k: 8, mean_d: -0.73}]}`, `elections {n: 5, k_beaten: 4, mean_legal: 2.0, mean_frozen: 1.7, mean_consensus: 1.21, mean_with: 1.38, k_with: 4}`, `cells {n: 24, k_closer: 19, k_direction: 21, k_crossed: 2, k_half: 6, missed: [{label: '2015', party: 'PSOE'}, {label: '2019 abr.', party: 'PSOE'}, {label: '2019 abr.', party: 'Vox'}, {label: '2019 nov.', party: 'PSOE'}, {label: '2019 nov.', party: 'UP'}]}`, `big {threshold: 5000, n: 12, k: 11, mean_d: -0.65}`, `exception {scope: 'es-ct', n: 17, k: 6}`; `pairs.rows` the 25 national pairs of the mockup's `PAIRS2` array (fields of spec D5, `national true`, `big` for the five with ≥ 5000) plus two regional ones; `elections.rows` the five generals with `model` and `period {since: '1996-01-01', k: 7, n: 8, mean_d: -1.1}`; `cells.rows` the 24 rows of the mockup table; `houses.rows` the 11 houses of the mockup and `big_rows` its 12 rows; `latest` = 2023 with its 8 polls; `meta` with `next {event_date: '2026-11-29', open: false}`, `params` of the Global Constraints, `sources` the three articles (`label`, `url`, `date`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_bundle_unit.py` extend the registry test (`:84-105`):

```python
assert bundle.EVENT_PARTS == ('meta', 'pollsters', 'polls', 'cycle', 'ban', 'results', 'model', 'series')
assert 'ban' in bundle.EVENT_CSV and 'ban-legal' in bundle.EVENT_CSV and 'model-ban' in bundle.EVENT_CSV
assert bundle.SCHEMAS['event-ban'] == {'parties': list, 'columns': list, 'polls': list, 'legal': list}
assert bundle.STUDIES[-1] == 'andorra'
assert bundle.STUDY_PARTS['andorra'] == ('meta', 'summary', 'pairs', 'elections', 'cells', 'houses', 'latest')
assert bundle.STUDY_SCHEMAS['andorra'] == {part: 'andorra-' + part for part in bundle.STUDY_PARTS['andorra']}
assert bundle.STUDY_CSV['andorra'] == ('pairs', 'elections', 'cells', 'houses', 'latest-polls')
assert bundle.path_study('andorra', 'pairs') == 'studies/andorra/pairs.json'
assert bundle.path_event('es', '2023-07-23', 'ban') == 'events/es/2023-07-23/ban.json'
```

Update the `STUDIES == ('voto-cera', 'sesgo-cis')` assertion (`:87`) to the tuple as it stands plus `'andorra'`. In `tests/test_webapi_unit.py` add `test_csv_labels_cover_every_csv_of_andorra`: `set(pages.analisis.CSV_LABELS['andorra']) == set(bundle.STUDY_CSV['andorra'])`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_webapi_unit.py -q -k "registry or csv_labels or schema"`
Expected: FAIL (`EVENT_PARTS` lacks `ban`, `'andorra'` not in `STUDIES`, fixtures missing)

- [ ] **Step 3: Declare the parts and schemas and write the fixtures**

Edit `bundle.py` as the Interfaces block says; keep the docstring tree of the module in step (`ban.json`, `csv/ban.csv`, `csv/ban-legal.csv`, `csv/model-ban.csv`, `studies/andorra/`). Write the seven fixtures with the figures above (every float with at most 2 decimals except `p`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_web_api.py -q`
Expected: PASS (the parametrised validation covers the seven fixtures)

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/pages/analisis.py tests/fixtures/bundle/andorra-*.json tests/test_bundle_unit.py tests/test_webapi_unit.py
git commit -m "Bundle: parte de evento ban, estudio andorra (partes, esquemas, CSV) y fixtures"
```

---

### Task 2: The event part `ban`

**Files:**
- Modify: `mtpy/lib/publish/events.py` (`export_event_ban` new after `export_event_cycle`; `publish_event:368-530`; `export_event_meta:293-331`)
- Modify: `mtpy/lib/publish/__init__.py:19-22` (export `export_event_ban`, `BAN_COLUMNS`)
- Modify: `tests/fakes.py:338-353` (`synthetic_cycle`), `:384-430` (`FakeComputer.filter_polls`), new `synthetic_ban()`, `synthetic_legal()`
- Modify: `tests/test_publish_unit.py:1291-1330` (publish_event tests), new tests
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Produces: `BAN_COLUMNS = POLL_COLUMNS + ['days', 'error_avg', 'error_blocks', 'bias_blocks']` (in `events.py`); `export_event_ban(polls: pd.DataFrame, legal: pd.DataFrame, names: Sequence[str]) -> tuple[dict, dict[str, pd.DataFrame]]` returning `({'parties': [...], 'columns': BAN_COLUMNS, 'polls': [...], 'legal': [...]}, {'ban': frame, 'ban-legal': frame})`: `polls` are the rows of `polls` through `export_poll_rows` (ascending `date`, `pollster_id`) with the four extra columns (`days` int, errors rounded to `PCT_DECIMALS`, null when missing); `legal` the same shape, one row per `pollster_id` (the row with the fewest `days`; ties: the latest `date`), only for the `pollster_id` present in `polls`. `publish_event` writes `ban` after `cycle` from `comp.filter_polls('results', drop_ctypes=['exit'])` restricted to `ctype == 'wban'` and `comp.filter_polls('results', drop_mtypes=[], drop_ctypes=['wban', 'exit'])`; `export_event_meta(..., n_ban: int = 0)` writes `n_ban` into the meta; the returned dict of `publish_event` gains `'n_ban'`.
- Fixtures: `synthetic_ban()` = three rows with the index and columns of `synthetic_cycle()` (`ctype 'wban'`): GAD3 (id 2) at `2023-07-22` (`days 1`, `sample_size 10004`, `error_avg 2.1`, `error_blocks 6.96`, `bias_blocks 15.16`, PP 35.5 PSOE 28.6 VOX 12.7), GAD3 at `2023-07-21` (`days 2`, `error_avg 2.3`), ElectoPanel (id 4, `mtype 'online'`, sponsor `Electomanía` id 30) at `2023-07-21` (`days 2`, `sample_size 1250`, `error_avg 1.6`); `synthetic_legal()` = `synthetic_cycle()` plus one ElectoPanel row at `2023-07-14` (`days 9`, `error_avg 1.4`). `FakeComputer.filter_polls(featured=True, **kw)` returns `synthetic_cycle()` + `synthetic_ban()` when `kw.get('drop_ctypes') == ['exit']`, `synthetic_legal()` when `kw.get('drop_mtypes') == []`, and `synthetic_cycle()` otherwise (calls still recorded).

- [ ] **Step 1: Write the failing tests**

In `tests/test_publish_unit.py`:

```python
def test_export_event_ban_keeps_the_poll_shape_and_the_last_legal_of_each_house():
    data, frames = publish.export_event_ban(synthetic_ban().reset_index(), synthetic_legal().reset_index(), NAMES)
    assert data['columns'] == publish.BAN_COLUMNS and data['parties'] == list(NAMES)
    assert [(p['pollster'], p['days']) for p in data['polls']] == [('GAD3', 2), ('ElectoPanel', 2), ('GAD3', 1)]
    assert data['polls'][2]['error_avg'] == 2.1 and data['polls'][2]['sample_size'] == 10004
    assert [(p['pollster'], p['days']) for p in data['legal']] == [('GAD3', 7), ('ElectoPanel', 9)]  # 40dB and CIS have no clandestine poll
    assert set(frames) == {'ban', 'ban-legal'} and list(frames['ban'].columns[:len(publish.BAN_COLUMNS)]) == publish.BAN_COLUMNS

def test_export_event_ban_without_polls_is_empty():
    data, frames = publish.export_event_ban(synthetic_ban().iloc[:0].reset_index(), synthetic_legal().reset_index(), NAMES)
    assert data['polls'] == [] and data['legal'] == [] and frames['ban'].empty and frames['ban-legal'].empty

def test_publish_event_writes_the_ban_part_and_counts_it_in_meta(tmp_path):
    ...  # as test_publish_event_writes_the_parts_and_meta_last: the written names contain 'ban' between 'cycle' and 'results'
    ban = read('events/es/2023-07-23/ban')['data']
    assert len(ban['polls']) == 3 and {row['pollster'] for row in ban['legal']} == {'GAD3', 'ElectoPanel'}
    assert read('events/es/2023-07-23/meta')['data']['n_ban'] == 3 and out['n_ban'] == 3
    assert exists('events/es/2023-07-23/csv/ban.csv') and exists('events/es/2023-07-23/csv/ban-legal.csv')
    assert ('filter_polls', 'results', {'drop_ctypes': ['exit']}) in comp.calls and ('filter_polls', 'results', {'drop_mtypes': [], 'drop_ctypes': ['wban', 'exit']}) in comp.calls
```

Keep `test_publish_event_writes_the_parts_and_meta_last` green by updating its expected order of written parts.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_unit.py -q -k "ban"`
Expected: FAIL with `AttributeError: export_event_ban`

- [ ] **Step 3: Implement `export_event_ban`, the fixtures and the publication**

`export_event_ban` builds both tables with `export_poll_rows` on the frames reindexed with the extra columns (`weight` as in `export_event_cycle`), then appends the four extras to each record; the `legal` frame is reduced with `sort_values(['days', 'date'], ascending=[True, False]).drop_duplicates('pollster_id')` restricted to `polls['pollster_id'].unique()`. In `publish_event`, after `cycle`: `write('ban', data, {'ban': frame, 'ban-legal': legal_frame})`; `n_ban = len(data['polls'])` into `export_event_meta`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_publish_unit.py tests/test_web_api.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish/events.py mtpy/lib/publish/__init__.py tests/fakes.py tests/test_publish_unit.py
git commit -m "Publish event: parte ban con los sondeos clandestinos, sus errores y el último legal de cada casa"
```

---

### Task 3: `drop_ctypes` through the Forecaster, the Simulator and the backtest case

**Files:**
- Modify: `mtpy/lib/forecaster.py:126-150` (`__init__`), `:447-460` (`load_polls`)
- Modify: `mtpy/lib/simulator.py:47-75` (`__init__`), `:280-298` (the `Forecaster(...)` call)
- Modify: `mtpy/lib/backtest.py:366-390` (`run_case`), `:955-980` (`run_backtest`)
- Test: `tests/test_forecaster_unit.py`, `tests/test_simulator_unit.py`, `tests/test_backtest_unit.py:715-770` (the `recording` fixture records the Simulator kwargs)

**Interfaces:**
- Produces: `Forecaster(..., drop_ctypes: Optional[list[str]] = ['wban', 'exit'])` storing `self.drop_ctypes` (a list; `None` → `[]`) and passing `drop_contexts=self.drop_ctypes` to `get_poll_series` in `load_polls`; `Simulator(..., drop_ctypes: Optional[list[str]] = ['wban', 'exit'])` passing it to the `Forecaster`; `run_case(..., drop_ctypes: Optional[list[str]] = None)` passing it to the `Simulator` **only when not `None`** (so every existing call builds the Simulator exactly as before); `run_backtest(..., drop_ctypes=None)` passing it to every case.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecaster_unit.py
def test_forecaster_drops_the_ban_and_exit_polls_by_default(monkeypatch):
    seen = {}
    def get_poll_series(scope, event_dates, drange=None, drop_mtypes=None, drop_contexts=None, **kw):
        seen['drop_contexts'] = drop_contexts; return <the frame the existing load_polls tests use>
    monkeypatch.setattr(forecaster, 'get_poll_series', get_poll_series)
    Forecaster('es', '2023-07-23').load_polls()
    assert seen['drop_contexts'] == ['wban', 'exit']
    Forecaster('es', '2023-07-23', drop_ctypes=['exit']).load_polls()
    assert seen['drop_contexts'] == ['exit']
    assert Forecaster('es', '2023-07-23', drop_ctypes=None).drop_ctypes == []

# tests/test_backtest_unit.py
def test_run_case_passes_drop_ctypes_to_the_simulator(recording):
    backtest.run_case('es', '2023-07-23', 0, n_sim=5, nowcast_only=True, drop_ctypes=['exit'])
    assert recording.kwargs['drop_ctypes'] == ['exit']

def test_run_case_without_drop_ctypes_builds_the_simulator_as_before(recording):
    backtest.run_case('es', '2023-07-23', 6, n_sim=5, nowcast_only=True)
    assert 'drop_ctypes' not in recording.kwargs
```

In `tests/test_simulator_unit.py` add `test_simulator_passes_drop_ctypes_to_the_forecaster` with the pattern of `_StopAfterForecaster` (`:893`): the Forecaster receives `drop_ctypes=['exit']` when the Simulator is built with it, and `['wban', 'exit']` by default.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_forecaster_unit.py tests/test_simulator_unit.py tests/test_backtest_unit.py -q -k "drop_ctypes"`
Expected: FAIL with `TypeError: unexpected keyword argument 'drop_ctypes'`

- [ ] **Step 3: Thread the parameter through the four signatures**

Document it in each docstring: «`drop_ctypes`: poll contexts left out (`wban`, published abroad during the ban, and `exit` polls); the default makes explicit what the horizon and the freeze guaranteed».

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_forecaster_unit.py tests/test_simulator_unit.py tests/test_backtest_unit.py tests/test_tuning_unit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py mtpy/lib/simulator.py mtpy/lib/backtest.py tests/test_forecaster_unit.py tests/test_simulator_unit.py tests/test_backtest_unit.py
git commit -m "Forecaster: drop_ctypes explícito (wban y exit fuera por defecto) hasta el Simulator y run_case"
```

---

### Task 4: The backtest table `ban.csv`

**Files:**
- Modify: `mtpy/lib/backtest.py:41-43` (`RESULT_FILES`), `:955-1035` (`run_backtest`), `:1255-1380` (`backtest_scope`), new `has_ban_polls(scope: str, event_date: str) -> bool`
- Modify: `mtpy/jobs/Backtest.py:85-110` (`run`, docstring with the example `python job.py backtest '{"scopes":["es"],"ban":true}'`), `mtpy/pipelines/backtest/Backtest.py:52-90` (`evaluate_task`)
- Modify: `tests/test_backtest_unit.py:189-198` (`fake_results`), `tests/test_pipelines_backtest.py:97-124`, `:125-160` (push counts)
- Test: `tests/test_backtest_unit.py`, `tests/test_pipelines_backtest.py`

**Interfaces:**
- Produces: `run_backtest(..., ban: bool = False)`: with `ban`, after the horizons of each election, if `has_ban_polls(scope, event_date)`, runs `run_case(scope, event_date, 0, nowcast_only=True, drop_ctypes=['exit'], <same n_sim, seed, max_fc, min_polls, house_effects, industry_bias, dispersion, herding, composition, regional_noise, min_events, error_model, drift_scale, tail_dof, weight_ratings, config, rating_exponent>)` and appends its `shares` rows (`horizon == 0`) to `out['ban']` only; an exception is warned and recorded in `meta` with `horizon 0` like any skipped case; the key `'ban'` exists in the result **only when `ban` is true** (an empty frame with the `shares` columns if every case was skipped). `has_ban_polls` reads `get_poll_series(scope, [event_date], drange=(0, 5))` and answers whether any row has `ctype == 'wban'` (add `ctype` to the columns that query returns if it lacks it; the builder checks `data.get_poll_series`). `backtest_scope(..., ban: bool = False)` passes it, writes `ban.csv` through the existing `results.items()` loop, removes a stale `ban.csv` of the folder when the result has no `'ban'`, and records `'ban': ban` in `run.json`. `RESULT_FILES` gains `'ban.csv'` (so `copy_results` pushes, pulls and removes it). The job and the pipeline pass `ban` through.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backtest_unit.py
def test_run_backtest_adds_the_ban_case_apart_from_the_tables(monkeypatch):
    calls = []
    def fake_case(scope, event_date, horizon, **kw):
        calls.append((horizon, kw.get('nowcast_only'), kw.get('drop_ctypes')))
        return <a case dict as the existing fakes of run_case build, with shares rows carrying `horizon`>
    monkeypatch.setattr(backtest, 'run_case', fake_case)
    monkeypatch.setattr(backtest, 'has_ban_polls', lambda scope, date: date == '2023-07-23')
    out = backtest.run_backtest(scope='es', events=['2019-11-10', '2023-07-23'], horizons=[6], ban=True)
    assert calls == [(6, False, None), (6, False, None), (0, True, ['exit'])]
    assert set(out['ban']['event_date']) == {'2023-07-23'} and set(out['ban']['horizon']) == {0}
    assert set(out['metrics']['horizon']) == {6} and set(out['shares']['horizon']) == {6} and set(out['series']['horizon']) <= {6}

def test_run_backtest_without_ban_has_no_ban_table(monkeypatch):
    ...; out = backtest.run_backtest(scope='es', events=['2023-07-23'], horizons=[6]); assert 'ban' not in out

def test_backtest_scope_writes_ban_csv_and_removes_a_stale_one(tmp_path):
    results = fake_results() | {'ban': fake_results()['shares'].assign(horizon=0)}
    backtest.backtest_scope('es-md', out=str(tmp_path), events=['2023-05-28'], ban=True, runner=lambda **kw: results, stats=..., prov=...)
    assert (tmp_path / 'es-md' / 'ban.csv').exists() and json.load(open(tmp_path / 'es-md' / 'run.json'))['ban'] is True
    backtest.backtest_scope('es-md', out=str(tmp_path), events=['2023-05-28'], runner=lambda **kw: fake_results(), stats=..., prov=...)
    assert not (tmp_path / 'es-md' / 'ban.csv').exists() and json.load(open(tmp_path / 'es-md' / 'run.json'))['ban'] is False
```

In `tests/test_pipelines_backtest.py` extend `test_backtest_passes_every_parameter_through_and_records_it` with `ban=True` in the call, `'ban': True` in `runner.calls[0]` and `run['ban'] is True`; update the push/pull tests to the new count of `RESULT_FILES` (ten files plus `run.json`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_backtest_unit.py tests/test_pipelines_backtest.py -q -k "ban or parameter_through or push"`
Expected: FAIL with `TypeError: unexpected keyword argument 'ban'`

- [ ] **Step 3: Implement the ban case, the file and the parameter**

`has_ban_polls` lives next to `default_events`. In `run_backtest` the loop over horizons is untouched; the ban case is a separate block after it. In `backtest_scope`, `params` gains `ban` (so `run.json` records it through `**params`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_backtest_unit.py tests/test_pipelines_backtest.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/backtest.py mtpy/jobs/Backtest.py mtpy/pipelines/backtest/Backtest.py tests/test_backtest_unit.py tests/test_pipelines_backtest.py
git commit -m "Backtest: caso ban por elección (horizonte 0 con los sondeos clandestinos) en ban.csv, aparte de las métricas"
```

---

### Task 5: The `ban` rows of the event `model` part

**Files:**
- Modify: `mtpy/lib/publish/events.py:183-219` (`export_event_model`), new `EVENT_BAN_FILE = 'ban.csv'`
- Modify: `tests/fakes.py:306-320` (`write_backtest_source(root, scope='es', ban=False)`)
- Test: `tests/test_publish_unit.py:1278-1340`

**Interfaces:**
- Produces: `export_event_model(source, event_date)` returns `data` with the extra key `'ban'` (rows of `ban.csv` of that election through `export_backtest_table`, as `shares`) and `frames['model-ban']` **only when** the source has `ban.csv` with rows of the election; `data['horizons']` ignores the ban rows. `write_backtest_source(..., ban=True)` writes `ban.csv` = `synthetic_backtest().shares` of `2023-07-23` with `horizon 0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_event_model_exports_the_ban_rows_apart_from_the_horizons(tmp_path):
    folder = write_backtest_source(str(tmp_path / 'bt'), ban=True)
    data, frames = publish.export_event_model(folder, '2023-07-23')
    assert data['horizons'] == list(BACKTEST_HORIZONS) and {r['horizon'] for r in data['ban']} == {0}
    assert 'model-ban' in frames and {r['party'] for r in data['ban']} == set(NAMES)
    data, frames = publish.export_event_model(folder, '2019-11-10')
    assert 'ban' not in data and 'model-ban' not in frames

def test_publish_event_retires_a_stale_model_ban_csv(tmp_path):
    ...  # publish with ban=True, then with a source without ban.csv: 'events/es/2023-07-23/csv/model-ban.csv' disappears and model.json has no 'ban'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_unit.py -q -k "model_ban or ban_rows"`
Expected: FAIL (`'ban' not in data`)

- [ ] **Step 3: Implement**

After the `EVENT_MODEL_PARTS` loop: if the source exists and has `EVENT_BAN_FILE`, slice it by `event_date`; when the slice has rows, `data['ban'], frames['model-ban'] = export_backtest_table(slice)` (rows and frame). The stale-CSV loop of `publish_event` already covers `model-ban` (it is in `EVENT_CSV` since Task 1).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_publish_unit.py tests/test_pages.py -q -k "event or model"`
Expected: PASS (the election page keeps reading the shortest horizon)

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish/events.py tests/fakes.py tests/test_publish_unit.py
git commit -m "Publish event: la parte model gana las filas ban del backtest (horizonte 0), fuera de horizons"
```

---

### Task 6: The study package `mtpy/lib/studies/ban/`

**Files:**
- Create: `mtpy/lib/studies/ban/__init__.py`, `data.py`, `stats.py`, `study.py`; `mtpy/lib/studies/labels.py`; `data/andorra/params.json`
- Modify: `mtpy/lib/studies/lean/data.py:17-80` (import `MONTHS`, `election_code`, `election_label` from `..labels`; `_short`/`_label` become aliases so `tests/test_lean_data.py` stays green), `data/README.md:31` (row for `andorra/params.json`)
- Modify: `tests/fakes.py` (new `write_ban_bundle(fs)`)
- Create: `tests/test_ban_data.py`, `tests/test_ban_stats.py`, `tests/test_ban_study.py`
- Test: those three plus `tests/test_lean_data.py`

**Interfaces:**
- `labels.py`: `MONTHS`, `election_code(scope, event_date, month=False) -> str` («GEN 23», «GEN 19N»), `election_label(scope, name, event_date, month=False) -> str` («Generales 2023», «Generales nov. 2019»), moved verbatim from `lean/data.py`, plus `short_year_label(event_date, repeated: bool) -> str` («2023», «2019 abr.», the labels of the mockup: `MONTHS[m][:3] + '.'`).
- `data.py`: `PARAMS_FILE = 'andorra/params.json'`; `load_params(data: FileSystem) -> dict`; `class MissingInput(Exception)` (as lean's); `@dataclass Event(scope: str, event_date: str, label: str, name: Optional[str], national: bool, main: list, polls: pd.DataFrame, legal: pd.DataFrame, pollsters: Optional[pd.DataFrame], official: Optional[pd.Series], frozen: Optional[pd.DataFrame], with_ban: Optional[pd.DataFrame])` (`frozen`/`with_ban`: the `shares` rows of the frozen horizon / the `ban` rows, restricted to `main`, indexed by `party` with `model` and `official`; `frozen` is `None` without rows at `params.frozen_horizon`; `with_ban` is `None` when `frozen` is `None` or `model` lacks `ban`); `list_scopes(reader) -> list[str]` (keys of `manifest.scopes`, `national` first); `read_event(reader, scope, date, params, repeated: bool) -> Optional[Event]` (`None` and a reason when `results` or `ban` is missing); `read_events(reader, params) -> tuple[list[Event], dict]` (events with `event_date >= since` of every scope, chronological, plus `inputs = {scope: {'read': [...], 'skipped': {date: reason}}}`); `read_next(reader, params) -> Optional[dict]` (`{'event_date', 'open'}` from the latest national run's `meta.event_date` and the existence of `events/es/{date}/results`).
- `stats.py`: `pair_rows(event: Event, params) -> list[dict]` (fields of spec D5 `pairs` **without `slug`**: the study is below the web layer, the parts carry `pollster_id` and `pollster`, and the page adds the profile href with `pages.slugs` as `pages/lean.py:house_href` does; the same for `houses.rows` and `latest.polls`); `sign_test(k: int, n: int) -> float`; `consensus(event, params) -> Optional[dict[str, float]]`; `cell_rows(event, consensus: dict, params) -> list[dict]`; `election_row(event, pairs: list[dict], consensus, params) -> dict` (with `model` block or `None`; `model.error_legal` = mean of `pollsters.last_error_avg` where `last_days <= max_days`, `None` without `pollsters`); `house_rows(pairs, params) -> list[dict]`; `big_rows(pairs, params) -> list[dict]`; `summarize(events, pairs, elections, cells, params) -> dict` (the `summary` of spec D5: `pairs`, `elections`, `cells`, `big`, `exception`; `missed` = cells with `closer` false).
- `study.py`: `@dataclass BanStudy(params, inputs, next, summary, pairs, elections, period, cells, houses, big, latest)`; `compute_ban(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> BanStudy`; raises `MissingInput('pairs')` when no pair exists; `latest` = the last national event (chronological) with `polls` rows: `{scope, event_date, label, parties: main, polls: [...], consensus, frozen: {party: model} or None, official: {party: pct}, error: {legal, frozen, consensus, with}}`.
- Fixture `write_ban_bundle(fs) -> dict`: manifest with `scopes` `es` (latest run with `meta.event_date '2026-11-29'`) and `es-md`; events (each with `meta` `{name, main, blocks, featured}`, `ban`, `pollsters`, `results`, and `model` where said): **es/2019-11-10** main `['PSOE', 'PP', 'VOX']`, official 28.0/20.8/15.1, `model` horizon 6 rows 27.4/20.3/13.9 and `ban` rows 27.3/19.9/14.8; `ban.polls`: GAD3 (id 2, 13000, `2019-11-09`, days 1, error_avg 0.86, 27.0/20.1/16.3), GAD3 (`2019-11-10`, days 0, error_avg 0.80, 27.3/20.1/16.0), ElectoPanel (id 4, online, 1250, `2019-11-08`, days 2, error_avg 0.67, 27.1/20.4/15.5); `ban.legal`: GAD3 (days 10, error_avg 0.77), ElectoPanel (days 6, error_avg 0.61); `pollsters`: GAD3 (last_days 10, last_error_avg 0.77), 40dB (last_days 12, 1.11). **es/2023-07-23** main `['PP', 'PSOE', 'VOX', 'SUMAR']`, official 33.1/31.7/12.4/12.3, `model` horizon 6 rows 34.2/28.3/13.1/13.1 (no `ban`); `ban.polls`: 40dB (id 3, 2000, `2023-07-21`, days 2, error_avg 1.71, 31.7/29.4/13.8/13.8), GAD3 (10004, `2023-07-22`, days 1, error_avg 2.10, 35.5/28.6/12.7/11.9), YouGov (id 9, mtype aggr, `2023-07-18`, days 5, error_avg 1.5, 33.0/29.0/13.0/13.0); `ban.legal`: 40dB (days 7, error_avg 1.45), GAD3 (days 6, 2.73), YouGov (days 8, 1.6); `pollsters`: 40dB (7, 1.45), GAD3 (6, 2.73), CIS (13, 1.9). **es-md/2021-05-04** main `['PP', 'MM', 'PSOE']`, official 44.7/17.0/16.9, no `model`; `ban.polls`: GAD3 (8500, days 1, error_avg 0.97), Sigma Dos (id 5, 1000, days 1, error_avg 1.20); `ban.legal`: GAD3 (days 60, error_avg 1.72), Sigma Dos (days 8, 1.50); `pollsters` both (last_days 60 and 8). **es/2026-11-29**: `meta` and `ban` (one GAD3 poll) **without** `results`. Returns the hand-computed figures the tests assert.
- Hand-computed expectations (params of the Global Constraints): pairs = 2019 GAD3 `d +0.03` (`big`), 2019 ElectoPanel `+0.06`, 2023 40dB `+0.26`, 2023 GAD3 `−0.63` (`big`), es-md Sigma Dos `−0.30`; es-md GAD3 has no pair (legal 60 days); YouGov no pair (`aggr`). `summary.pairs`: `n 5, k 2, ties 0, share 0.4, mean_d −0.12, p 0.812, generals {n 4, k 1, mean_d −0.07, n_events 2}, regional {n 1, k 1, mean_d −0.3, n_events 1}`. 2019 consensus (ElectoPanel out): `PSOE 27.3, PP 20.1, VOX 16.0`; cells 2019: PSOE closer false (direction false), PP closer false, VOX closer true + direction + crossed, half false; `error_frozen 0.77`, `error_consensus 0.77`, `beaten false`, `error_with 0.63`, `beaten_with true`, `error_legal 0.94`. 2023 consensus (YouGov out): `PP 33.6, PSOE 29.0, VOX 13.25, SUMAR 12.85`; cells: PP closer+direction+half, PSOE closer+direction, VOX neither, SUMAR closer+direction; `error_frozen 1.5`, `error_consensus 1.15`, `beaten true`, `error_with None`, `error_legal 2.03`. `summary.elections {n 2, k_beaten 1, mean_legal 1.48, mean_frozen 1.13, mean_consensus 0.96, mean_with 0.63, k_with 1}` (the summary means are computed from the unrounded election figures: 2019 `error_legal` 0.94 exactly, 2023 2.0267; `error_frozen` 0.7667 and 1.5); `summary.cells {n 7, k_closer 4, k_direction 4, k_crossed 1, k_half 1}`; `big {threshold 5000, n 2, k 1, mean_d −0.3}`; `exception None`; `houses.rows == []` with `min_pairs 3` and `[{GAD3: n 2, k 1, share 0.5, mean_d −0.3}]` with `min_pairs 2`; `period {since '1996-01-01', k 0, n 0, mean_d None}`; `next {event_date '2026-11-29', open False}`; `latest.event_date == '2023-07-23'` with 3 polls; `inputs['es']['skipped']['2026-11-29'] == 'no results'`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ban_data.py`: `test_labels_moved_from_lean_keep_their_output` (`election_label('es', None, '2019-11-10', month=True) == 'Generales nov. 2019'`, `short_year_label('2019-04-28', True) == '2019 abr.'`), `test_read_events_reads_every_scope_and_skips_the_election_without_results` (three events, `inputs` as above), `test_read_event_without_the_frozen_horizon_has_no_model`, `test_read_next_reports_the_run_election_and_whether_it_has_results`, `test_compute_ban_ignores_an_election_without_results` (no `2026-11-29` anywhere in the study's parts).
`tests/test_ban_stats.py`: `test_pair_rows_pair_each_house_with_its_last_legal_within_the_window` (the five pairs and their `d`, `closer`, `big`), `test_pairs_skip_a_legal_poll_older_than_the_window`, `test_pairs_leave_aggregators_out_but_keep_panels`, `test_sign_test_is_the_binomial_of_the_wins` (`sign_test(2, 5) == pytest.approx(0.8125)`), `test_consensus_leaves_out_the_panel_and_the_aggregator`, `test_consensus_averages_only_the_polls_with_the_party` (a poll with `VOX` null), `test_cell_rows_flag_closer_direction_crossed_and_half` (the 2019 and 2023 cells above), `test_election_row_with_and_without_model`, `test_house_rows_need_min_pairs`, `test_big_rows_take_the_pairs_over_the_threshold`.
`tests/test_ban_study.py`: `test_compute_ban_summary_matches_the_hand_figures`, `test_compute_ban_without_any_pair_raises_missing_input`, `test_compute_ban_without_any_model` (`summary.elections is None`, `cells == []`), `test_latest_is_the_last_national_election_with_clandestine_polls`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ban_data.py tests/test_ban_stats.py tests/test_ban_study.py -q`
Expected: FAIL with `ModuleNotFoundError: mtpy.lib.studies.ban`

- [ ] **Step 3: Implement the package, the labels module, the params file and the fixture**

`labels.py` first (run `tests/test_lean_data.py` after moving the helpers). `read_event` reads `meta`, `results`, `ban`, `pollsters`, `model` with `reader.exists`/`read_json`; frames with `pd.DataFrame(part['polls'])`. `pair_rows` sorts the clandestine polls by `date` then row order and keeps the last per `pollster_id`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ban_data.py tests/test_ban_stats.py tests/test_ban_study.py tests/test_lean_data.py tests/test_lean_study.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/ban mtpy/lib/studies/labels.py mtpy/lib/studies/lean/data.py data/andorra/params.json data/README.md tests/fakes.py tests/test_ban_data.py tests/test_ban_stats.py tests/test_ban_study.py
git commit -m "Estudio andorra: paquete studies/ban (pares, consenso clandestino, celdas, guarda legal) y etiquetas comunes"
```

---

### Task 7: Publication of the study

**Files:**
- Create: `mtpy/lib/publish/ban.py`
- Modify: `mtpy/lib/publish/studies.py:21-60` (`STUDY_TITLES`), `:571-585` (`STUDY_EXPORTERS`, `STUDY_META`, `STUDY_RUNNERS`), `mtpy/lib/publish/__init__.py` (exports), `mtpy/jobs/Publish.py:35-40,77` (docstring and example)
- Modify: `deploy/README.md:1335-1380` («### Estudios»), `:1309-1334` («### Backtest»: the `ban` flag), `docs/web/contrato.md:395-453` (events: the part `ban`, the key `ban` of `model`), `:454-520` (the study's parts), `:764-830` (routes)
- Create: `tests/test_publish_ban.py`
- Test: `tests/test_publish_ban.py`, `tests/test_publish_studies.py`

**Interfaces:**
- `STUDY_TITLES['andorra'] = ('El mercado de la fruta en Andorra: ¿aciertan más los sondeos «clandestinos» de última hora?', 'Los cinco días anteriores a la votación la ley prohíbe publicar sondeos en España. Se publican fuera, y en las redes circulan como precios de la fruta. En las {n} generales desde 2015, el consenso de esos sondeos estuvo más cerca del resultado que el promedio congelado en {k}, y quedó más cerca del resultado en {kc} de las {nc} cifras de partido.')` (the subtitle without a model: `'… En {np} pares casa a casa desde 1996, el sondeo clandestino estuvo más cerca del resultado que el último legal en {kp}.'`).
- `publish/ban.py`: `compute_for_publish(data, writer) -> BanStudy`; `export_ban_summary(study)`, `export_ban_pairs(study)`, `export_ban_elections(study)`, `export_ban_cells(study)`, `export_ban_houses(study)`, `export_ban_latest(study)`, `export_ban_meta(study, study_id='andorra', registration=None, checked_at=None, prov=None, seconds=None)`, each `-> tuple[dict, dict[str, pd.DataFrame]]` with the CSV twins of `STUDY_CSV_OF['andorra']` (`elections` CSV flat: `model_error_legal`, `model_error_frozen`, …); `BAN_SOURCES` = the three articles (`{'label': '«La \'frutería de Andorra\' y las encuestas prohibidas: por qué es ilegal difundirlas», El Confidencial, 18-12-2017', 'url': ...}`, Onda Cero 6-11-2019 «Resultados del sondeo electoral de Andorra: notable ascenso del 'brócoli' y descalabro de la 'naranja'», Heraldo de Aragón 5-2-2026 «Cómo saltarse la prohibición de publicar encuestas con el 'ppollo', el brócoli y los frutos rojos») plus `/elecciones`, `/metodo`, `/analisis`. `STUDY_RUNNERS['andorra'] = {'compute': compute_for_publish, 'registrable': False}` (plus `'live': True` if the key exists in the file at the time).

- [ ] **Step 1: Write the failing tests**

`tests/test_publish_ban.py` (pattern of `tests/test_publish_lean.py`): `test_publish_study_andorra_writes_the_seven_parts_and_five_csv` (with `write_ban_bundle`; the manifest entry `{published_at, registered_at: None, checked: False}`; `meta.next == {'event_date': '2026-11-29', 'open': False}`; `summary.pairs.p == 0.812`), `test_publish_study_andorra_refuses_register`, `test_export_ban_meta_formats_the_subtitle_with_the_live_figures`. In `tests/test_publish_studies.py` the set equality at `:196` and `test_importing_publish_does_not_load_the_study_package` gain the package.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_ban.py tests/test_publish_studies.py -q`
Expected: FAIL (`KeyError: 'andorra'`)

- [ ] **Step 3: Implement the exporters and the registration; update the docs**

Contract: the table of event parts gains `ban` (`event-ban@1`: `parties`, `columns`, `polls`, `legal`), the key `ban` of `model`, the study `andorra` (seven parts) and its routes; deploy README: «**Mercado de la fruta.** `{"what":["study"],"study":"andorra"}` …», the `ban` flag of the backtest and the post-election steps of spec D3/review 3.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_publish_ban.py tests/test_publish_studies.py tests/test_publish_lean.py tests/test_jobs_publish.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish/ban.py mtpy/lib/publish/studies.py mtpy/lib/publish/__init__.py mtpy/jobs/Publish.py deploy/README.md docs/web/contrato.md tests/test_publish_ban.py tests/test_publish_studies.py
git commit -m "Publish study andorra: exportadores, registro, documentación del contrato y del despliegue"
```

---

### Task 8: The page `/analisis/andorra`

**Files:**
- Create: `mtpy/lib/pages/ban.py`, `web/templates/analisis_andorra.html`
- Modify: `mtpy/lib/pages/analisis.py:23-25` (`STUDIES`), `:652-653` (`CONTEXTS`), `mtpy/lib/pages/__init__.py` (export `ban_context`), `tests/test_pages.py:1835-1847` (`'ban'` in the layout list), `:3496-3509` (wording list gains `('/analisis/andorra', '')`)
- Test: `tests/test_pages.py` (new tests, with `publish_study(fs, study='andorra')` and the Task 1 fixtures)

**Interfaces:**
- `pages/ban.py`: `BAN = 'andorra'`; `verdicts(summary) -> list[dict]` (three `{question, label, tone}` of spec D7: P1 «SÍ»/«NO»/«A MEDIAS», P2 «EN {k} DE {n}»/«SÍ»/«NO» or `None` label without model, P3 «LOS GRANDES»/«{k} DE {n} GRANDES»/best house); `kpis(summary) -> list[dict]`; `tests(summary) -> list[dict]` (the three boxes of P2); `election_rows(elections, summary, params) -> list[dict]` (the generals since `chart_since`, the `period` row, then the three sums); `cell_groups(cells) -> list[dict]` (per election: `label`, `rows` with the movement text and class: `okl`/`kol`/`ok`/`ko`/`mut` as the mockup; «clavado» when `|consensus − official| < 0.05`); `model_rows(elections) -> list[dict]` plus the «Media» row; `house_rows(houses) -> list[dict]` with `href` from `pages.slugs` by `pollster_id`; `big_rows(houses) -> list[dict]`; `latest_card(latest, next) -> dict` (`open` = `next.open and latest.event_date == next.event_date`, `veda_from` = `event_date − 6` days, `year`, `day_month`); `island(pairs, params) -> dict` (`{'pairs': [{label, error_legal, error_ban}]}` for `national` rows with `event_date >= chart_since`, oldest election first, `d` ascending inside); `ban_context(state) -> dict` (in `analisis.py`, the pattern of `lean_context`: `study_meta, summary, verdicts, kpis, tests, elections, exception, cells, model_rows, houses, big_rows, latest, sources, csv_links, initial, published_at`).
- Template ids and classes of the mockup v2 (`#study-head`, `.intro`, `#resumen`, `#p1`, `#p2`, `#p3`, `#veintinueve`, `#datos`, `#c-dumb`, `.locked`, `.mv`); macros `m.card_start`, `m.kpis`, `m.how`, `m.card_image_button` as in `analisis_sesgo_cis.html`; `{% block scripts %}` loads `/dist/js/pages/analisis-andorra.js`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pages.py`:

```python
def test_andorra_page_renders_the_mockup_figures(events_site, fresh_app):
    publish_study(fresh_app.fs, study='andorra')
    status, _, body = call(events_site, '/analisis/andorra'); page = html(body)
    assert status == 200 and 'El mercado de la fruta en Andorra' in page and 'sondeos de la veda' not in page
    assert [l for _, l in verdict_pills(page)][:3] == ['SÍ', 'EN 4 DE 5', 'LOS GRANDES']
    assert '81 de 116' in page and '19 de 24' in page and '4 de 5' in page
    assert 'La ley.' in page and 'elconfidencial.com' in page
    assert initial(body)['pairs'][0]['label'] == 'GESOP · 2015'

def test_andorra_page_keeps_the_card_closed_without_results(events_site, fresh_app):
    ...  # meta.next.open false: '.locked' present, no poll of 2026 in the page, pill «Se completa con 2026»

def test_andorra_page_opens_the_card_with_the_published_election(events_site, fresh_app):
    ...  # fixtures with next {event_date '2023-07-23', open true}: the card lists the 8 polls of `latest` and the three rows Consenso clandestino · Promedio congelado · Resultado

def test_andorra_page_shows_an_election_without_pairs(...)      # elections row with n_pairs 0 → «–» cells
def test_andorra_page_without_a_model_has_no_verdict_in_p2(...) # summary.elections None → no P2 pill, KPI 1 «Elecciones con sondeos clandestinos»
def test_analisis_index_lists_andorra_after_the_others(...)
```

Add `('/analisis/andorra', '')` to the wording test and `'ban'` to `test_pages_package_layout`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pages.py -q -k "andorra or package_layout or public_wording"`
Expected: FAIL (404 and layout mismatch)

- [ ] **Step 3: Implement the view models, the context and the template**

Copy the texts of the mockup v2 into the template, each figure from the context; the header paragraphs verbatim; the sources line from `meta.sources`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pages.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/ban.py mtpy/lib/pages/analisis.py mtpy/lib/pages/__init__.py web/templates/analisis_andorra.html tests/test_pages.py
git commit -m "Web: página /analisis/andorra (modelos de vista pages/ban.py, plantilla con textos condicionados, tarjeta del 29-N cerrada)"
```

---

### Task 9: The dumbbell chart and the page script

**Files:**
- Create: `web/dist/js/charts/dumbbell.js`, `web/dist/js/pages/analisis-andorra.js`
- Modify: `tests/test_js_charts.py:36-62` (harness: `DUMBBELL_JS` and alias `db`), new tests
- Test: `tests/test_js_charts.py`

**Interfaces:**
- `dumbbell.js`: `export function dumbbellOption(rows, opts = {})` with `rows = [{label, error_legal, error_ban}]` and `opts = {max?: number, rowHeight?: number}`; `export function renderDumbbell(el, rows, opts)` (`mountChart(el, dumbbellOption(rows, opts))`, sets `el.style.height` to `max(320, rows.length * 22) px`). Option: `yAxis` category, `inverse: true`, `data` = labels in the given order; `xAxis` value `min 0`, `max` = `opts.max` or the max of both errors rounded up to 0.5 and at least 3.5, `interval 0.5`, `name 'error medio por partido (puntos)'`, `nameLocation 'middle'`, `nameGap 28`, `grid.bottom 48`; one `custom` series whose `data` is `[[error_legal, error_ban], …]` and whose rendered item colours (stored in `itemStyle` per datum for the tests: `{color: THEME.successColor or '#2e9e5b' if error_ban < error_legal, '#ce3b3b' if greater, THEME grey if equal}`); tooltip text «{label} · último legal {a} → clandestino {b} (afinó|empeoró {|d|})» with `fmtNum(x, 2)`.
- `analisis-andorra.js`: pattern of `analisis-sesgo-cis.js` (`readInitial`, `tryDraw(() => renderDumbbell(el, initial.pairs))`, `mountWhenReady`, `wireCardImages`).

- [ ] **Step 1: Write the failing tests**

```python
def test_dumbbell_draws_one_custom_series_with_a_colour_per_row():
    rows = [{'label': 'GESOP · 2015', 'error_legal': 2.71, 'error_ban': 1.57}, {'label': 'IMOP · 19 abr.', 'error_legal': 1.19, 'error_ban': 1.95}, {'label': 'X', 'error_legal': 1.0, 'error_ban': 1.0}]
    option = run_js('db.dumbbellOption(rows)', rows=rows)
    assert option['yAxis']['inverse'] is True and option['yAxis']['data'] == ['GESOP · 2015', 'IMOP · 19 abr.', 'X']
    assert option['xAxis']['min'] == 0 and option['xAxis']['max'] == 3.5 and option['xAxis']['name'] == 'error medio por partido (puntos)'
    series = option['series']; assert len(series) == 1 and series[0]['type'] == 'custom'
    colours = [d['itemStyle']['color'] for d in series[0]['data']]
    assert colours[0] != colours[1] and colours[2] not in (colours[0], colours[1])

def test_dumbbell_axis_grows_with_the_errors():
    option = run_js('db.dumbbellOption(rows)', rows=[{'label': 'A', 'error_legal': 4.2, 'error_ban': 3.9}])
    assert option['xAxis']['max'] == 4.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_js_charts.py -q -k dumbbell`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the module and the page script**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_js_charts.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/dumbbell.js web/dist/js/pages/analisis-andorra.js tests/test_js_charts.py
git commit -m "Web: gráfico dumbbell (antes → después por casa) y script de la página de Andorra"
```

---

### Task 10: Método link, docs, full suite and closing

**Files:**
- Modify: `mtpy/lib/pages/metodo.py:777-820` (`andorra_published`, as `lean_published`), `web/templates/metodo.html:88-95` (the «Veda» bullet gains «Si aciertan más que los legales, elección a elección, está en <a href="/analisis/andorra">El mercado de la fruta en Andorra</a>.» when `andorra_published`)
- Modify: `docs/superpowers/specs/2026-11-05-web-analisis-andorra-design.md` («Cierre» section)
- Test: `tests/test_pages.py` (`test_metodo_links_the_andorra_study_only_when_published`)

- [ ] **Step 1: Write the failing test**

```python
def test_metodo_links_the_andorra_study_only_when_published(events_site, fresh_app):
    assert 'href="/analisis/andorra"' not in html(call(events_site, '/metodo')[2])
    publish_study(fresh_app.fs, study='andorra')
    page = html(call(events_site, '/metodo')[2])
    assert 'href="/analisis/andorra"' in page and 'El mercado de la fruta en Andorra' in page
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_pages.py -q -k metodo_links_the_andorra`
Expected: FAIL (the link is never there)

- [ ] **Step 3: Implement `andorra_published` in `pages/metodo.py` (as `lean_published`, `:819`) and the sentence in `metodo.html`**

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS (count the new tests against the previous total; no new warnings)

- [ ] **Step 5: Write the closing in the spec and commit**

«Cierre» in the spec: deviations from the spec (if any), the list for Luis in this order — `backtest '{"scopes":["es"],"ban":true}'` on the laptop, `backtest push`, `publish '{"what":["event"],"scopes":"all","event_date":"all"}'`, `publish '{"what":["study"],"study":"andorra"}'`, walk `/analisis`, `/analisis/andorra` (desktop and mobile: the dumbbell with 25 rows, the cells table, the closed card), `/metodo`, deploy; after the 29-N, the steps of review 3.

```bash
git add mtpy/lib/pages/metodo.py web/templates/metodo.html tests/test_pages.py docs/superpowers/specs/2026-11-05-web-analisis-andorra-design.md
git commit -m "Método: enlace al análisis del mercado de la fruta; cierre de la spec"
```
