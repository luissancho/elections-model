# Análisis «El sesgo del CIS de Tezanos» Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a second study, `sesgo-cis`, computed from the already-published parts of the bundle, and its page `/analisis/sesgo-cis` with the approved mockup's structure, live figures and charts.

**Architecture:** The study registry of the bundle and the publisher becomes per-study (dicts keyed by slug) so `voto-cera` keeps its parts, schemas, CSV and registration untouched. A light package `mtpy/lib/studies/lean/` reads `poll-errors` of every analysed scope, the event `es/2023-07-23` and the latest national run through a `BundleReader`, computes the six parts, and `mtpy/lib/publish/lean.py` exports them. The page follows `/analisis/voto-cera`: a context builder in `pages/analisis.py` with view models in `pages/lean.py`, a Jinja template with the texts and tables, and a page script that draws the charts from the `initial` island with the existing chart modules plus three new ones.

**Tech Stack:** Python 3.11, pandas, scipy (already in `requirements.txt`), Jinja2, pytest; vanilla ES modules on ECharts 5 tested with `node`.

**Spec:** `docs/superpowers/specs/2026-10-24-web-analisis-sesgo-cis-design.md` (approved 2026-10-24; the approved mockup is `docs/superpowers/specs/2026-10-24-web-analisis-sesgo-cis-mockup.html`).

## Global Constraints

- No database access anywhere in the study: inputs come only from the bundle (`BundleReader(fs, prefix=bundle.PREFIX)`) and from `data/` (`es-scopes.csv`, `sesgo-cis/params.json`).
- `voto-cera` keeps its behaviour, parts, schema names (`study-*`), CSV names, registration and tests; `tests/test_publish_studies.py` must stay green.
- `mtpy.lib.publish` stays light: the study packages and `scipy` are imported inside functions (`test_importing_publish_does_not_load_the_study_package`); `pages` and `webapi` never import `publish` or the model.
- Code identifiers in English; Spanish only in public texts. Public wording: «sólo» with accent, «encuestadora»/«casa», never «empresa», quotes «», `%` glued to the figure («31%»), `assert_public_wording` passes on `/analisis/sesgo-cis`.
- Every figure of the page comes from the published parts (never typed into the template), except the context facts of the spec («Hechos verificados»).
- Signs: `Izquierda`/`Derecha` are poll minus result in points; `lean = Izquierda − Derecha` (positive = towards the left); the published `error_blocks` is `Derecha − Izquierda` and is not used for `lean`.
- Periods: with Tezanos `event_date >= '2018-09-01'` (label «2018 – 2026»); without, `'2008-01-01' <= event_date < '2017-01-01'` (label «2008 – 2016»). Ranking and matrix: houses with `>= 10` elections with Tezanos **and** polls in 2026 (`active_from '2026-01-01'`). 2023: last poll per house with `days <= 60`.
- Fluid layout: no `max-width` on the page; charts mount with `mountChart` (ResizeObserver); the two period bullseyes and the 2023 one are square by CSS class.
- Git: Luis commits in parallel in the same tree. Before each commit run `git status --short` and `git log -1`, add only the task's files by path, never `git commit -a`, never rewrite history. Commit messages in the repo's style («Área: qué cambia»), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run tests with `python -m pytest -q` from the repo root; JS tests need `node` on PATH (`tests/test_js_charts.py` skips without it).

## Review Focus

1. A scope whose `poll-errors` part has only one block column (no `Derecha`): rows must keep `Derecha: None`, `lean: None`, be counted in the left statistics and skipped in `n_lean`/`k_lean` (pinned in Task 3, `test_summarize_period_skips_rows_without_the_right_block`).
2. An election where the CIS has a row but no other house, or the other way round: no consensus row for it, but it still counts in `n`/`k_left` and in `industry` only when three or more other houses exist (Task 3, `test_consensus_and_industry_need_other_houses`).
3. A `poll-errors` row whose election has no published event folder: dropped, exactly as `/casas/cis` and Método do (Task 2, `test_read_error_rows_keeps_only_published_events`).
4. No manifest, or a manifest without a national run: the active houses come only from the 2026 elections of `poll-errors` and `meta.inputs.active_source` says so; the study still publishes (Task 2, `test_active_pollsters_without_a_national_run`).
5. The study published but the CERA one not, and the reverse: `/analisis`, the home band and `/api/v1/studies` list exactly the published ones; the unpublished page is 404 (Task 8, `test_analisis_lists_each_study_only_when_published`).

---

### Task 1: Study registry per study (bundle, API, pages, test helper)

**Files:**
- Modify: `mtpy/lib/bundle.py:72-79` (constants), `:134-146` (schemas), `:510-561` (`path_study`, `path_study_csv`), module docstring tree `:8-29`
- Modify: `mtpy/lib/webapi/checks.py:22-23`, `mtpy/controllers/Studies.py:60-76`
- Modify: `mtpy/lib/pages/analisis.py:39-50` (`CSV_LABELS`), `:440-460` (`_csv_links`)
- Modify: `tests/test_webapi_unit.py:59-72` (`write_study`), `tests/test_bundle_unit.py:84-100`, `:247-250`
- Test: `tests/test_bundle_unit.py`, `tests/test_web_api.py`, `tests/test_pages.py`

**Interfaces:**
- Produces: `bundle.STUDIES == ('voto-cera', 'sesgo-cis')`; `bundle.STUDY_PARTS: dict[str, tuple]`, `bundle.STUDY_SCHEMAS: dict[str, dict[str, str]]`, `bundle.STUDY_CSV: dict[str, tuple]`, `bundle.STUDY_CSV_OF: dict[str, dict[str, tuple]]`, all keyed by slug; `STUDY_PARTS['sesgo-cis'] == ('meta', 'summary', 'elections', 'houses', 'matrix', 'election')`; `STUDY_SCHEMAS['sesgo-cis'][part] == 'sesgo-cis-' + part`; `STUDY_CSV['sesgo-cis'] == ('elections', 'houses', 'matrix', 'election-houses')`; `STUDY_CSV_OF['sesgo-cis'] == {'elections': ('elections',), 'houses': ('houses',), 'matrix': ('matrix',), 'election': ('election-houses',)}`; `bundle.SCHEMAS` gains the six `sesgo-cis-*` entries with the required keys of Task 5; `pages.analisis.CSV_LABELS: dict[str, dict[str, str]]` keyed by slug; `write_study(fs, study='voto-cera', meta=None, parts=None)` writes the parts and CSV of `study`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_bundle_unit.py` replace `test_study_paths_follow_the_published_layout_and_reject_bad_input` with:

```python
def test_study_registry_is_per_study():
    assert bundle.STUDIES == ('voto-cera', 'sesgo-cis')
    assert bundle.STUDY_PARTS['voto-cera'] == ('meta', 'census', 'participation', 'vote', 'placement', 'prediction', 'sensitivity', 'check')
    assert bundle.STUDY_PARTS['sesgo-cis'] == ('meta', 'summary', 'elections', 'houses', 'matrix', 'election')
    assert bundle.STUDY_SCHEMAS['voto-cera'] == {part: 'study-' + part for part in bundle.STUDY_PARTS['voto-cera']}
    assert bundle.STUDY_SCHEMAS['sesgo-cis'] == {part: 'sesgo-cis-' + part for part in bundle.STUDY_PARTS['sesgo-cis']}
    for study in bundle.STUDIES:
        assert all(name in bundle.SCHEMAS for name in bundle.STUDY_SCHEMAS[study].values())
        listed = [name for names in bundle.STUDY_CSV_OF[study].values() for name in names]
        assert sorted(listed) == sorted(bundle.STUDY_CSV[study]) and set(bundle.STUDY_CSV_OF[study]) <= set(bundle.STUDY_PARTS[study])
    assert bundle.STUDY_CSV['sesgo-cis'] == ('elections', 'houses', 'matrix', 'election-houses')
    assert bundle.path_study('sesgo-cis', 'matrix') == 'studies/sesgo-cis/matrix.json'
    assert bundle.path_study_csv('sesgo-cis', 'election-houses') == 'studies/sesgo-cis/csv/election-houses.csv'
    assert bundle.path_study('voto-cera', 'meta') == 'studies/voto-cera/meta.json'
    assert bundle.path_study_csv('voto-cera', 'last-seat') == 'studies/voto-cera/csv/last-seat.csv'
    for call in (lambda: bundle.path_study('nope', 'meta'), lambda: bundle.path_study('voto-cera', 'summary'),
                 lambda: bundle.path_study('sesgo-cis', 'census'), lambda: bundle.path_study('Voto-Cera', 'meta'),
                 lambda: bundle.path_study_csv('sesgo-cis', 'check'), lambda: bundle.path_study_csv('voto-cera', 'houses')):
        with pytest.raises(ValueError):
            call()
```

Parametrise `test_study_schemas_validate_the_fixtures` over `[(study, part) for study in bundle.STUDIES for part in bundle.STUDY_PARTS[study]]` loading `bundle.STUDY_SCHEMAS[study][part]` (the `sesgo-cis-*` fixtures arrive in Task 5; until then mark those cases `pytest.importorskip`-free but expect them to fail: keep the parametrisation and let Task 5 turn them green).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bundle_unit.py -q -k "registry or fixtures"`
Expected: FAIL (`STUDIES == ('voto-cera',)`, `STUDY_PARTS` is a tuple).

- [ ] **Step 3: Make the registry per study in `mtpy/lib/bundle.py`**

Replace the five constants at `:72-79` with the dicts of the Interfaces block; add to `SCHEMAS` the six entries (required keys, nullable where noted): `'sesgo-cis-meta': {'study': str, 'title': str, 'subtitle': str, 'pollster': str, 'periods': dict, 'blocks': list, 'params': dict, 'inputs': dict, 'sources': list, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'seconds': dict}`, `'sesgo-cis-summary': {'with': dict, 'without': dict, 'election': dict}`, `'sesgo-cis-elections': {'rows': list}`, `'sesgo-cis-houses': {'rows': list}`, `'sesgo-cis-matrix': {'events': list, 'houses': list, 'cells': list}`, `'sesgo-cis-election': {'scope': str, 'date': str, 'name': NULLABLE_STR, 'blocks': dict, 'max_days': int, 'result': dict, 'cis': dict, 'consensus': dict, 'houses': list}`. In `path_study`/`path_study_csv` validate with `STUDY_PARTS[study]` / `STUDY_CSV[study]` after `_check_study`. Update the docstring tree (`studies/{study}/…`).

- [ ] **Step 4: Update the consumers**

`mtpy/lib/webapi/checks.py:23`: keep `STUDY_PARTS = bundle.STUDY_PARTS` (now a dict). `mtpy/controllers/Studies.py`: `check_part(part, STUDY_PARTS[study])` after `check_study(study)`; `names = bundle.STUDY_CSV_OF[study].get(part, ())`. `mtpy/lib/pages/analisis.py`: `CSV_LABELS = {'voto-cera': {…the eleven labels…}, 'sesgo-cis': {'elections': 'Las elecciones del CIS, con y sin Tezanos', 'houses': 'Ranking de encuestadoras desde 2018', 'matrix': 'Matriz casa por elección', 'election-houses': 'Últimos sondeos de las generales de 2023 por bloques'}}`; `_csv_links(study)` iterates `bundle.STUDY_CSV[study]` with `bundle.STUDY_CSV_OF[study]` and `CSV_LABELS[study][name]`. `tests/test_webapi_unit.py::write_study`: iterate `bundle.STUDY_CSV[study]`, `bundle.STUDY_PARTS[study]`, `bundle.STUDY_SCHEMAS[study]`. Grep `STUDY_PARTS\b|STUDY_CSV\b|STUDY_CSV_OF|STUDY_SCHEMAS|CSV_LABELS` over `mtpy/ tests/ docs/` and fix every remaining tuple/dict use (including `test_cera_page_…datos` counting the eleven CSV pills and `tests/test_publish_studies.py::test_export_study_parts_validate`, which must index by `'voto-cera'`).

- [ ] **Step 5: Run the affected suites**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_web_api.py tests/test_webapi_unit.py tests/test_pages.py tests/test_publish_studies.py -q`
Expected: all PASS except the six `test_study_schemas_validate_the_fixtures[sesgo-cis-*]` cases (fixtures arrive in Task 5).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/webapi/checks.py mtpy/controllers/Studies.py mtpy/lib/pages/analisis.py tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py tests/test_publish_studies.py
git commit -m "Estudios: partes, esquemas y CSV declarados por estudio; sesgo-cis en el registro"
```

---

### Task 2: Reading the inputs from the bundle (`studies/lean/data.py`)

**Files:**
- Create: `mtpy/lib/studies/lean/__init__.py` (docstring: the study of the lean of a pollster from the published parts), `mtpy/lib/studies/lean/data.py`
- Create: `data/sesgo-cis/params.json`; Modify: `data/README.md` (one table row)
- Create: `tests/fakes.py::write_lean_bundle` (append), `tests/test_lean_data.py`

**Interfaces:**
- Produces: `MissingInput(Exception)` with `.part`; `PARAMS_FILE = 'sesgo-cis/params.json'`; `load_params(data: FileSystem) -> dict` (the JSON, keys `pollster`, `since`, `control` (`[from, to]`), `blocks`, `min_elections`, `max_days`, `active_from`, `election` (`{scope, date}`), `labels` (`{with: '2018 – 2026', without: '2008 – 2016'}`)); `read_error_rows(reader: BundleReader, catalogue: pd.DataFrame) -> tuple[list[dict], dict]`: rows of every analysed scope with published events (fields of `poll-errors` plus `scope`, `scope_name`, `label` («{scope_name} {year}», «Generales {year}» for `es`), `short` («AN 18», «GEN 23»), `year: int`, `lean: float | None`) and `inputs` (`{scope: {'run_id', 'generated_at'} | {'skipped': reason}}`), in catalogue order then `event_date`; `read_election(reader, scope, date) -> dict` (`{'meta', 'polls', 'results'}` data dicts; raises `MissingInput`); `active_pollsters(reader, rows, active_from: str) -> tuple[set[str], str]` (the set and `'run+events'` or `'events'`); `write_lean_bundle(fs, *, since='2018-09-01') -> dict` in `tests/fakes.py` writing a synthetic bundle (scopes `es` and `es-an`: analysis `meta` + `poll-errors` with the rows below, one event folder `meta.json` per election date, the event `es/2023-07-23` with `meta`, `polls`, `results`, a manifest with `scopes.es.latest` and the run's `polls.json`) and returning the expected figures used by the tests: `cis_es_dates` (`['2008-03-09', '2011-11-20', '2015-12-20', '2016-06-26', '2019-04-28', '2019-11-10', '2023-07-23']`), `industry` (`{'n_events': 2, 'left': 0, 'right': 2, 'mean': -1.43}`: only es 2023 and es-an 2026 have three or more other houses), `election` (the dict `read_election` returns for `es`/`2023-07-23`), `summary_election` (`{cis_dist, consensus_dist, closer, rank, n}` computed by hand: five houses within 60 days) and `houses` (`['GAD3', 'Sigma Dos', 'CIS']` with `min_elections 3`).

Synthetic rows (all `days` 10, `sample_size` 1000, `weight` 1): CIS with Tezanos: es 2019-04-28 (Izq +0.1, Der −0.1), es 2019-11-10 (+6.4, −6.1), es 2023-07-23 (+3.1, −2.8), es-an 2018-12-02 (+12.6, −9.5), es-an 2022-06-19 (+3.1, −4.9), es-an 2026-05-17 (−0.5, −1.5); CIS without: es 2008-03-09 (−1.6, −1.2), es 2011-11-20 (+0.4, +2.0), es 2015-12-20 (−6.3, +5.0), es 2016-06-26 (+3.0, −2.3); GAD3 in every CIS election with Tezanos (Izq −1.0, Der +1.0) and in 2016 (+1.0, −1.0); Sigma Dos in the six with Tezanos (+2.0, −1.0); NC Report only in es 2023-07-23 and es-an 2026-05-17 (−4.0, +3.0); Hamalgama in es 2023-07-23 only (−4.4, +4.7). Plus one `poll-errors` row of `es` on `2012-11-25` for the CIS with no event folder (must be dropped). The event 2023 `polls` rows: the five houses above with the same errors and `days` 11 (CIS), 6, 6, 7, 10, plus `Ipsos` with `days` 90 (−2.9, +2.5) that must be excluded; `results` rows PP 33.06, PSOE 31.68, VOX 12.38, SUMAR 12.33; `meta.blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'SUMAR']}`. Run `polls.json`: polls dated 2026 for CIS, GAD3, Sigma Dos and NC Report; Hamalgama only in 2025.

- [ ] **Step 1: Write the failing tests**

```python
def test_read_error_rows_keeps_only_published_events(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); expected = write_lean_bundle(fs)
    rows, inputs = data.read_error_rows(bundle.BundleReader(fs), catalogue())
    assert [r['event_date'] for r in rows if r['pollster'] == 'CIS' and r['scope'] == 'es'] == expected['cis_es_dates']
    assert '2012-11-25' not in {r['event_date'] for r in rows}
    cis23 = next(r for r in rows if r['pollster'] == 'CIS' and r['event_date'] == '2023-07-23')
    assert (cis23['label'], cis23['short'], cis23['year'], cis23['lean']) == ('Generales 2023', 'GEN 23', 2023, 5.9)
    assert inputs['es']['run_id'] and inputs['es-an']['run_id'] and inputs['es-ct'] == {'skipped': 'no analysis'}

def test_read_election_needs_its_three_parts(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_lean_bundle(fs)
    election = data.read_election(bundle.BundleReader(fs), 'es', '2023-07-23')
    assert set(election) == {'meta', 'polls', 'results'} and election['meta']['blocks']['Izquierda'] == ['PSOE', 'SUMAR']
    with pytest.raises(data.MissingInput) as info:
        data.read_election(bundle.BundleReader(fs), 'es', '2019-11-10')
    assert info.value.part == 'events/es/2019-11-10/polls'

def test_active_pollsters_without_a_national_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_lean_bundle(fs)
    rows, _ = data.read_error_rows(bundle.BundleReader(fs), catalogue())
    assert data.active_pollsters(bundle.BundleReader(fs), rows, '2026-01-01') == ({'CIS', 'GAD3', 'Sigma Dos', 'NC Report'}, 'run+events')
    fs.remove(bundle.path_manifest())
    assert data.active_pollsters(bundle.BundleReader(fs), rows, '2026-01-01') == ({'CIS', 'GAD3', 'Sigma Dos', 'NC Report'}, 'events')

def test_load_params_reads_the_repository_file():
    params = data.load_params(FileSystem(os.path.join(ROOT, 'data')))
    assert params['pollster'] == 'CIS' and params['since'] == '2018-09-01' and params['control'] == ['2008-01-01', '2017-01-01']
    assert params['min_elections'] == 10 and params['max_days'] == 60 and params['active_from'] == '2026-01-01'
    assert params['election'] == {'scope': 'es', 'date': '2023-07-23'} and params['blocks'] == ['Izquierda', 'Derecha']
```

(`catalogue()` is `webapi.checks.catalogue` with `fresh_app.data = FileSystem(ROOT/data)` set in a fixture of the test module; `fs.remove` as the project's `FileSystem` names it — check `mtpy/core/io.py`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_lean_data.py -q`
Expected: FAIL with `ModuleNotFoundError: mtpy.lib.studies.lean`.

- [ ] **Step 3: Write `data/sesgo-cis/params.json` and its README row**

Values of the Interfaces block (`labels` included). README row: `sesgo-cis/params.json` | Parámetros del estudio del sesgo del CIS (casa, periodos con y sin Tezanos, bloques, mínimo de elecciones, ventana de 2023, actividad en 2026, elección analizada) | Curación propia.

- [ ] **Step 4: Implement `write_lean_bundle` in `tests/fakes.py`**

Use `bundle.BundleWriter(fs)`: `write_json(path_analysis(scope, 'poll-errors'), 'poll-errors', {'blocks': ['Derecha', 'Izquierda'], 'rows': rows}, scope)` plus the analysis `meta` from `fixture_data('analysis-meta')`; event folders through `write_json(path_event(scope, date, 'meta'), 'event-meta', fixture_data('event-meta') | {...}, scope)`; the 2023 event parts `polls`/`results` with the rows above; `update_manifest` + a run `polls` part as `write_bundle` does (`tests/test_webapi_unit.py:25-41`), with the `polls` rows replaced by the 2026 polls. Return `{'cis_es_dates': [...], ...}` with every figure the later tests assert (write them down in the dict as you compute them by hand).

- [ ] **Step 5: Implement `mtpy/lib/studies/lean/data.py`**

`read_error_rows`: for each catalogue row (`scode` index, `name` column) with `reader.analysis_exists(code)` and a readable `poll-errors` part: `dates = set(reader.list_events(code))`, keep rows with `event_date in dates`, add the fields, `lean = round(Izquierda − Derecha, 2)` only when both are numbers. `short`: `code.upper().removeprefix('ES-')` → `'GEN'` when `code == 'es'`, plus `' ' + event_date[2:4]`. `inputs[code]` from the part envelope (`run_id`, `generated_at`) or `{'skipped': 'no analysis'}`. `read_election`: three `read_json` calls with `MissingInput(path)` on a missing file. `active_pollsters`: (a) manifest `scopes['es']['latest']` → run `polls` part, names with `date >= active_from`; (b) `rows` with `event_date >= active_from`; union; source string as in the Interfaces.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_lean_data.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mtpy/lib/studies/lean/__init__.py mtpy/lib/studies/lean/data.py data/sesgo-cis/params.json data/README.md tests/fakes.py tests/test_lean_data.py
git commit -m "Sesgo del CIS: lectura de las entradas desde las partes publicadas y parámetros del estudio"
```

---

### Task 3: Statistics and tables (`studies/lean/stats.py`)

**Files:**
- Create: `mtpy/lib/studies/lean/stats.py`
- Test: `tests/test_lean_stats.py`

**Interfaces:**
- Consumes: rows of `read_error_rows` (Task 2).
- Produces: `sign_chance(n: int, k: int) -> dict | None` (`{n, k, ways, total, p, odds, odds_exact}`, the same values as `pages.metodo.sign_chance`); `t_test(values: list[float]) -> dict | None` (`{n, mean, sd, t, p}`; `p` two-sided from `scipy.stats.t.sf`; `None` with fewer than 2 values or `sd == 0`); `summarize_period(rows: list[dict], pollster: str) -> dict` where `rows` are the rows of every house inside the period: `{n, k_left, n_right, k_right, mean_left, mean_right, n_lean, k_lean, mean_lean, exceptions, consensus: {n, k, mean_diff}, dates: {n, k}, chance: {coin, dates, consensus}, t: {all, dates, consensus}, industry: {n_events, left, right, mean}, rows: [consensus rows]}` with `rows` = one per election of the pollster `{scope, event_date, consensus: float | None, n_others: int, others_left: int}`; `house_table(rows: list[dict], pollster: str, min_elections: int, active: set[str]) -> list[dict]` (`{pollster, slug, n, k_left, share_left, n_lean, mean_lean, mean_left, active_2026, cis}`, sorted by `mean_lean` then `share_left`, deterministic ties by name); `matrix_table(rows, houses: list[dict]) -> dict` (`{'events': [{scope, event_date, label, short}], 'houses': [names], 'cells': [[x, y, Izquierda]]}`, events chronological then catalogue order, houses the CIS first then by `share_left` descending then name); `election_table(election: dict, pollster: str, max_days: int, blocks: list[str]) -> dict` (`{blocks, max_days, result, cis, consensus, houses}` of spec D3, `houses` by `dist` ascending; `slug` via `pages.slugs.slugify`).

- [ ] **Step 1: Write the failing tests**

```python
def test_sign_chance_matches_the_method_card():
    assert stats.sign_chance(41, 38) == metodo.sign_chance(41, 38)
    assert stats.sign_chance(41, 38)['ways'] == 11522 and stats.sign_chance(41, 38)['odds'] == 190000000
    assert stats.sign_chance(0, 0) is None

def test_t_test_known_values():
    out = stats.t_test([2.0, 4.0, 6.0, 8.0])
    assert (out['n'], out['mean'], out['sd']) == (4, 5.0, pytest.approx(2.582, abs=1e-3)) and out['t'] == pytest.approx(3.873, abs=1e-3)
    assert 0.02 < out['p'] < 0.04 and stats.t_test([1.0]) is None and stats.t_test([2.0, 2.0]) is None

def test_summarize_period_counts_the_synthetic_cis(fresh_app, tmp_path):
    rows, expected = lean_rows(tmp_path)   # helper of the test module: write_lean_bundle + read_error_rows
    with_rows = [r for r in rows if r['event_date'] >= '2018-09-01']
    out = stats.summarize_period(with_rows, 'CIS')
    assert (out['n'], out['k_left'], out['n_right'], out['k_right']) == (6, 5, 6, 6)
    assert out['mean_left'] == pytest.approx(4.13, abs=0.01) and out['mean_lean'] == pytest.approx(8.28, abs=0.01)
    assert out['exceptions'] == ['Andalucía 2026'] and out['dates'] == {'n': 6, 'k': 5}
    assert out['consensus']['n'] == 6 and out['consensus']['k'] == 5 and out['chance']['coin'] == stats.sign_chance(6, 5)
    assert out['industry'] == expected['industry']   # computed by hand in write_lean_bundle
    assert set(out['t']) == {'all', 'dates', 'consensus'} and out['t']['all']['n'] == 6

def test_summarize_period_skips_rows_without_the_right_block(fresh_app, tmp_path):
    rows, _ = lean_rows(tmp_path)
    rows = [dict(r, Derecha=None, lean=None) if r['event_date'] == '2019-04-28' else r for r in rows]
    out = stats.summarize_period([r for r in rows if r['event_date'] >= '2018-09-01'], 'CIS')
    assert (out['n'], out['n_right'], out['n_lean']) == (6, 5, 5)

def test_consensus_and_industry_need_other_houses(fresh_app, tmp_path):
    rows, _ = lean_rows(tmp_path)
    alone = [r for r in rows if r['event_date'] >= '2018-09-01' and not (r['scope'] == 'es-an' and r['pollster'] != 'CIS')]
    out = stats.summarize_period(alone, 'CIS')
    assert out['n'] == 6 and out['consensus']['n'] == 3 and out['industry']['n_events'] == 1   # only 2023 has >= 3 other houses

def test_house_table_filters_and_orders(fresh_app, tmp_path):
    rows, _ = lean_rows(tmp_path)
    houses = stats.house_table([r for r in rows if r['event_date'] >= '2018-09-01'], 'CIS', 3, {'CIS', 'GAD3', 'Sigma Dos', 'NC Report'})
    assert [h['pollster'] for h in houses] == ['GAD3', 'Sigma Dos', 'CIS'] and houses[-1]['cis'] is True
    assert houses[0]['slug'] == 'gad3' and houses[0]['share_left'] == 0 and houses[1]['share_left'] == 1.0

def test_matrix_and_election_tables(fresh_app, tmp_path):
    rows, expected = lean_rows(tmp_path)
    houses = stats.house_table([r for r in rows if r['event_date'] >= '2018-09-01'], 'CIS', 3, {'CIS', 'GAD3', 'Sigma Dos'})
    matrix = stats.matrix_table([r for r in rows if r['event_date'] >= '2018-09-01'], houses)
    assert matrix['houses'] == ['CIS', 'Sigma Dos', 'GAD3'] and matrix['events'][0]['short'] == 'AN 18'
    assert [c for c in matrix['cells'] if c[1] == 0 and c[0] == 2] == [[2, 0, 6.4]]
    election = stats.election_table(expected['election'], 'CIS', 60, ['Izquierda', 'Derecha'])
    assert [h['pollster'] for h in election['houses']][0] != 'Ipsos' and all(h['days'] <= 60 for h in election['houses'])
    assert election['result'] == {'Izquierda': 44.01, 'Derecha': 45.44, 'gap': -1.43}
    assert election['cis']['e_left'] == 3.1 and election['consensus']['n'] == 4 and election['cis']['dist'] == pytest.approx(4.18, abs=0.01)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_lean_stats.py -q`
Expected: FAIL with `ImportError` (`stats`).

- [ ] **Step 3: Implement `mtpy/lib/studies/lean/stats.py`**

`sign_chance` as `metodo.sign_chance` (copy, pure `math`). `t_test` imports `scipy.stats` inside. `summarize_period`: group `rows` by `(scope, event_date)`; the pollster's rows give `n`, `k_left`, means; `exceptions` = labels with `Izquierda <= 0` in `event_date` descending; `dates`: mean of the pollster's `Izquierda` per `event_date`; consensus rows: other houses' mean `Izquierda` when at least one; `industry`: elections with `>= 3` other houses, sign of their mean `Izquierda`; `chance.coin = sign_chance(n, k_left)`, `.dates = sign_chance(dates.n, dates.k)`, `.consensus = sign_chance(consensus.n, consensus.k)`; `t.all` on the pollster's `Izquierda`, `t.dates` on the date means, `t.consensus` on `cis − consensus`. Round means and `mean_diff` to 2 decimals. `house_table`/`matrix_table`/`election_table` as in the Interfaces; `dist = round(hypot(e_left, e_right), 2)`, `gap = Izquierda − Derecha` of the block shares, shares = result block share + error, result block shares = sum of `pct` of `meta.blocks[block]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_lean_stats.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/lean/stats.py tests/test_lean_stats.py
git commit -m "Sesgo del CIS: pruebas de azar, t de Student, ranking, matriz y tabla de 2023"
```

---

### Task 4: The study in one call (`studies/lean/study.py`)

**Files:**
- Create: `mtpy/lib/studies/lean/study.py`
- Test: `tests/test_lean_study.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `@dataclass LeanStudy` with `params: dict`, `periods: dict` (`{'with': {from, to, label, first_event, last_event, n}, 'without': {...}}`), `rows: list[dict]` (every house, both periods), `summary: dict` (`{'with': ..., 'without': ..., 'election': {cis_dist, consensus_dist, closer, rank, n}}`), `elections: list[dict]` (the pollster's rows of both periods with `period`, `consensus`, `n_others`, `others_left`), `houses: list[dict]`, `matrix: dict`, `election: dict` (with `scope`, `date`, `name` added), `inputs: dict` (`{'scopes': {...}, 'election': {'generated_at'}, 'active_source': str, 'run_id': str | None}`); `compute_lean(reader: BundleReader, data: FileSystem, params: dict | None = None) -> LeanStudy`; raises `MissingInput` when the national scope has no analysis or the election parts are missing.

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_lean_on_the_synthetic_bundle(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); expected = write_lean_bundle(fs)
    study = compute_lean(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'min_elections': 3})
    assert study.params['min_elections'] == 3 and study.params['pollster'] == 'CIS'
    assert study.periods['with'] == {'from': '2018-09-01', 'to': None, 'label': '2018 – 2026', 'first_event': '2018-12-02', 'last_event': '2026-05-17', 'n': 6}
    assert study.periods['without']['n'] == 4 and study.periods['without']['label'] == '2008 – 2016'
    assert study.summary['with']['k_left'] == 5 and study.summary['without']['k_left'] == 2
    assert study.summary['election'] == expected['summary_election']
    assert [e['period'] for e in study.elections].count('with') == 6 and study.elections[0]['event_date'] == '2026-05-17'
    assert [h['pollster'] for h in study.houses] == expected['houses'] and study.matrix['houses'][0] == 'CIS'
    assert study.election['date'] == '2023-07-23' and study.inputs['active_source'] == 'run+events'

def test_compute_lean_fails_without_the_national_analysis(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_lean_bundle(fs)
    fs.remove(bundle.path_analysis('es', 'meta'))
    with pytest.raises(MissingInput):
        compute_lean(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')))

def test_compute_lean_matches_the_method_card(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_lean_bundle(fs)
    fresh_app.fs = fs; fresh_app.data = FileSystem(os.path.join(ROOT, 'data'))   # `house_effects_example` reads the site and the catalogue
    study = compute_lean(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'min_elections': 3})
    card = metodo.house_effects_example()
    assert (card['n'], card['over']['Izquierda']) == (study.summary['with']['n'], study.summary['with']['k_left'])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_lean_study.py -q`
Expected: FAIL with `ImportError` (`compute_lean`).

- [ ] **Step 3: Implement `study.py`**

`compute_lean`: `params = load_params(data) | (params or {})`; `catalogue = data.read_csv('es-scopes.csv').set_index('scode')`; rows via `read_error_rows`; raise `MissingInput('analysis/es')` when `'es'` is skipped; split by period; `summary[period] = summarize_period(...)`; `elections` = the pollster's rows of both periods merged with their consensus rows, `period` added, newest first; `active, source = active_pollsters(...)`; `houses = house_table(with_rows, ...)`; `matrix = matrix_table(with_rows, houses)`; `election = election_table(read_election(...), ...) | {scope, date, name}`; `summary['election'] = {cis_dist, consensus_dist, closer (houses with dist < cis dist, the consensus not counted), rank (1-based position of the pollster among houses), n}`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_lean_study.py tests/test_lean_data.py tests/test_lean_stats.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/lean/study.py tests/test_lean_study.py
git commit -m "Sesgo del CIS: el estudio en una llamada (compute_lean)"
```

---

### Task 5: Exporters, generic `publish_study`, fixtures and docs

**Files:**
- Create: `mtpy/lib/publish/lean.py`
- Modify: `mtpy/lib/publish/studies.py:21-23` (`STUDY_TITLES`), `:553-656` (`STUDY_EXPORTERS`, `publish_study`), `mtpy/lib/publish/__init__.py:56-60`
- Create: `tests/fixtures/bundle/sesgo-cis-{meta,summary,elections,houses,matrix,election}.json`
- Modify: `docs/web/contrato.md:390-434`, `:690-710`; `deploy/README.md:975-1013`
- Test: `tests/test_publish_lean.py`, `tests/test_publish_studies.py`, `tests/test_bundle_unit.py`

**Interfaces:**
- Consumes: `LeanStudy` (Task 4), the registry (Task 1).
- Produces: in `publish/lean.py`: `export_lean_summary(study) -> (dict, {})`, `export_lean_elections(study) -> (dict, {'elections': DataFrame})`, `export_lean_houses(study) -> (dict, {'houses': DataFrame})`, `export_lean_matrix(study) -> (dict, {'matrix': DataFrame})` (long frame `pollster, scope, event_date, Izquierda`), `export_lean_election(study) -> (dict, {'election-houses': DataFrame})`, `export_lean_meta(study, study_id='sesgo-cis', registration=None, checked_at=None, prov=None, seconds=None) -> (dict, {})` (`subtitle` = `STUDY_TITLES['sesgo-cis'][1].format(k=..., n=...)`, `sources` = the four site links plus the three external URLs of the spec), `compute_for_publish(data: FileSystem, writer: BundleWriter) -> LeanStudy` (`compute_lean(bundle.BundleReader(writer.fs, prefix=bundle.PREFIX), data)`); in `publish/studies.py`: `STUDY_TITLES['sesgo-cis'] = ('El sesgo del CIS de Tezanos', 'Desde el nombramiento de José Félix Tezanos en 2018, el CIS ha dado al bloque de izquierdas más voto del que obtuvo en {k} de {n} elecciones generales y autonómicas celebradas. Es imposible que tal cosa pueda ocurrir por azar ni por un fallo general de las encuestas. Y no, en 2023 tampoco «acertó».')`, `STUDY_EXPORTERS: dict[str, dict[str, callable]]`, `STUDY_META: dict[str, callable]`, `STUDY_RUNNERS: dict[str, dict]` with `{'compute': callable(data, writer), 'registrable': bool}`; `publish_study(study_id, writer, register=False, force=False, data=None, compute=None, prov=None, today=None)` unchanged in signature, calling `compute(data, writer)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_publish_lean.py` (module fixture: synthetic bundle in a `tmp_path`, `fresh_app`, `study = compute_lean(...)` with `min_elections 3`):

```python
def test_export_lean_parts_validate(study):
    for part in bundle.STUDY_PARTS['sesgo-cis'][1:]:
        data, frames = publish.STUDY_EXPORTERS['sesgo-cis'][part](study)
        valid(bundle.STUDY_SCHEMAS['sesgo-cis'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['sesgo-cis'].get(part, ()))
    meta, frames = publish.export_lean_meta(study, prov=fake_prov())
    valid('sesgo-cis-meta', meta); assert frames == {}
    assert meta['subtitle'].startswith('Desde el nombramiento de José Félix Tezanos en 2018, el CIS ha dado al bloque de izquierdas más voto del que obtuvo en 5 de 6 elecciones')
    assert [s['url'] for s in meta['sources']][-3:] == [URL_INDEPENDIENTE_2018_07, URL_INDEPENDIENTE_2018_09, URL_UNED]

def test_publish_lean_writes_meta_last_and_updates_the_manifest(study, fresh_app, tmp_path):
    fs = RecordingFileSystem(str(tmp_path)); write_lean_bundle(fs); writer = make_writer(fs)
    out = publish.publish_study('sesgo-cis', writer, compute=lambda data, writer: study, prov=fake_prov)
    assert out == {'study': 'sesgo-cis', 'registered_at': None, 'checked': False, 'seconds': out['seconds']}
    written = [p for p in fs.written if '/studies/sesgo-cis/' in p]
    assert written[-1].endswith('/studies/sesgo-cis/meta.json') and fs.written[-1].endswith('/manifest.json')
    assert publish.read_manifest(writer)['studies']['sesgo-cis'] == {'published_at': '2026-10-08T12:00:00Z', 'registered_at': None, 'checked': False}

def test_publish_lean_refuses_to_register(study, fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_lean_bundle(fs)
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('sesgo-cis', make_writer(fs), register=True, compute=lambda data, writer: study, prov=fake_prov)

def test_publish_lean_reads_the_live_prefix_in_a_dry_run(study, fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); write_lean_bundle(fs)
    writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX)
    out = publish.publish_study('sesgo-cis', writer, prov=fake_prov)   # the default compute reads site/v1
    assert out['study'] == 'sesgo-cis' and writer.study_exists('sesgo-cis')
```

Add to `tests/test_publish_studies.py` an assertion that `publish.STUDY_RUNNERS['voto-cera']['registrable'] is True` and keep every existing test green. (`make_writer` and `fake_prov` come from `tests/test_publish_unit.py:393-401`: check whether `make_writer` takes the `fs` or builds its own, and adapt the calls; its clock is fixed at 2026-10-08T12:00Z, which the manifest assertion relies on.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_lean.py -q`
Expected: FAIL (`STUDY_EXPORTERS['sesgo-cis']` KeyError / no `export_lean_meta`).

- [ ] **Step 3: Implement `publish/lean.py` and generalise `publish_study`**

`publish/lean.py` imports `compute_lean` and `bundle` inside `compute_for_publish`; exporters round with `_rounded` of `studies.py` (import the helpers from there). `studies.py`: `STUDY_EXPORTERS = {'voto-cera': {...the seven...}, 'sesgo-cis': {'summary': export_lean_summary, ...}}`; `STUDY_META = {'voto-cera': export_study_meta, 'sesgo-cis': export_lean_meta}`; `STUDY_RUNNERS = {'voto-cera': {'compute': lambda data, writer: compute_cera(data) (imported inside a function `_cera_compute`), 'registrable': True}, 'sesgo-cis': {'compute': compute_for_publish, 'registrable': False}}`; in `publish_study`: `runner = STUDY_RUNNERS[study_id]`; `if register and not runner['registrable']: raise PublishRefused(f'study: {study_id} has nothing to register')` before any read; `compute = compute or runner['compute']`; `study = compute(data_fs, writer)`; loop `bundle.STUDY_PARTS[study_id][1:]` with `STUDY_EXPORTERS[study_id]`; `checked` only when `'check'` is a part; registration keys read from the previous meta only when `registrable`; meta via `STUDY_META[study_id](study, study_id, registration, checked_at, info, seconds)`. `__init__.py` re-exports the new names.

- [ ] **Step 4: Generate the fixtures**

Run a one-off script from the repo root (do not commit it) that builds the synthetic bundle in a temp dir, computes the study with `min_elections 3`, runs each exporter and `bundle.envelope(schema, data, None, generated_at='2026-10-08T12:00:00Z')` → `tests/fixtures/bundle/sesgo-cis-{part}.json` (pretty JSON, `ensure_ascii=False`). Check `tests/test_bundle_unit.py::test_study_schemas_validate_the_fixtures` passes for the twelve cases.

- [ ] **Step 5: Update the contract and the deploy README**

`docs/web/contrato.md` «Estudios»: «Hoy hay dos: `voto-cera` y `sesgo-cis`»; the registry is per study (`STUDY_PARTS[study]`…); a second table with the six parts of `sesgo-cis`, their schemas `sesgo-cis-{parte}@1` and required keys, and the four CSV twins; «Rutas de estudios»: `study ∈ voto-cera, sesgo-cis`. `deploy/README.md` «Estudios»: a paragraph for `{"what":["study"],"study":"sesgo-cis"}` (reads only published parts, needs the analyses published, no registration, chain `["analysis","study"]`).

- [ ] **Step 6: Run the suites**

Run: `python -m pytest tests/test_publish_lean.py tests/test_publish_studies.py tests/test_bundle_unit.py tests/test_jobs_publish.py tests/test_web_api.py -q`
Expected: PASS (including the twelve fixture cases).

- [ ] **Step 7: Commit**

```bash
git add mtpy/lib/publish/lean.py mtpy/lib/publish/studies.py mtpy/lib/publish/__init__.py tests/fixtures/bundle/sesgo-cis-*.json tests/test_publish_lean.py tests/test_publish_studies.py docs/web/contrato.md deploy/README.md
git commit -m "Sesgo del CIS: exportadores, publicación genérica por estudio, fixtures y contrato"
```

---

### Task 6: API with two studies

**Files:**
- Test: `tests/test_web_api.py:368-420`

**Interfaces:**
- Consumes: `write_study(fs, study='sesgo-cis')` (Task 1) and the fixtures (Task 5).

- [ ] **Step 1: Write the failing test**

```python
def test_studies_api_serves_both_studies(api, fresh_app, tmp_path):
    write_study(fresh_app.fs); write_study(fresh_app.fs, study='sesgo-cis'); webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/api/v1/studies')
    assert [s['study'] for s in json.loads(body)['studies']] == ['voto-cera', 'sesgo-cis']
    assert json.loads(body)['studies'][1]['registered_at'] is None
    status, headers, body = call(api, '/api/v1/studies/sesgo-cis/matrix', query='format=csv')
    assert status == 200 and headers['content-disposition'] == 'attachment; filename="sesgo-cis-matrix.csv"'
    status, _, body = call(api, '/api/v1/studies/sesgo-cis/election', query='format=csv&table=election-houses')
    assert status == 200
    for path, query, message in (('/api/v1/studies/sesgo-cis/census', '', 'unknown part'),
                                 ('/api/v1/studies/sesgo-cis/summary', 'format=csv', 'no csv for this part'),
                                 ('/api/v1/studies/voto-cera/matrix', '', 'unknown part')):
        got, _, body = call(api, path, query=query)
        assert got == 404 and json.loads(body)['message'] == message
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `python -m pytest tests/test_web_api.py -q -k studies`
Expected: PASS already if Task 1 and Task 5 were complete; if `index_action` reads `registered_at` from a meta without the key, fix `mtpy/controllers/Studies.py:30` to `data.get(key)`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_web_api.py mtpy/controllers/Studies.py
git commit -m "API de estudios: los dos estudios, sus partes y sus CSV"
```

---

### Task 7: Page view models (`pages/lean.py`) and context (`lean_context`)

**Files:**
- Create: `mtpy/lib/pages/lean.py`
- Modify: `mtpy/lib/pages/analisis.py` (`STUDIES`, `CONTEXTS`, new `lean_context`), `mtpy/lib/pages/__init__.py:16-17`
- Modify: `tests/test_pages.py:1446-1456` (package layout)
- Test: `tests/test_pages.py` (new `test_lean_context_*`)

**Interfaces:**
- Consumes: the parts via `study_part('sesgo-cis', part)`.
- Produces: in `pages/lean.py`: `LEAN = 'sesgo-cis'`, `VERDICTS = [('SÍ', 'q1'), ('NO', 'q3'), ('NO', 'q3')]`, `QUESTIONS = ('¿Se inclina el CIS hacia la izquierda desde que lo preside Tezanos?', '¿Puede ser azar, o un fallo general de las encuestas?', '¿Fue el CIS «el único que acertó» en 2023?')`, `verdict_rows() -> list[dict]`, `kpis(summary: dict, periods: dict) -> list[dict]` (three `m.kpis` rows: label/value/support of spec D6, `unit=''`), `tests(summary: dict) -> list[dict]` (the three boxes: `{key, label, value, text, highlight}` with `value` «1 entre {odds}» formatted with `fmt_int`, or «{k} de {n}»), `tech_note(summary: dict) -> dict` (the t values of both periods formatted with one decimal), `compare_boxes(summary, periods) -> list[dict]` (the two `.cmp` boxes), `island(parts: dict) -> dict` (`{'bull': {'with': rows, 'without': rows, 'blocks'}, 'matrix', 'consensus': [{x, y, label, national}], 'ranking': [{x, y, name, k, n, cis}], 'election': {'houses', 'consensus', 'blocks'}, 'distance': [{name, dist, cis, consensus}]}`); in `analisis.py`: `STUDIES['sesgo-cis'] = {'template': 'analisis_sesgo_cis.html'}`, `lean_context(state) -> dict` (common context plus `study_meta`, `summary`, `periods`, `verdicts`, `kpis`, `tests`, `tech`, `compare`, `elections` (with-Tezanos rows, newest first), `houses`, `election`, `sources`, `csv_links`, `initial`, `published_at`), `CONTEXTS[LEAN] = lean_context`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pages_package_layout():   # update the expected list
    ... == ['analisis', 'backtest', 'casas', 'cera', 'common', 'elecciones', 'escanos', 'filters', 'inicio', 'lean', 'metodo', 'promedios', 'slugs']

def test_lean_context_figures_come_from_the_parts(api, fresh_app):
    publish_study(fresh_app.fs, study='sesgo-cis')   # extend the helper with `study`
    context = pages.lean_context(pages.parse_state({}))
    summary = fixture_data('sesgo-cis-summary')
    assert [k['label'] for k in context['kpis']] == ['Elecciones con la izquierda sobreestimada', 'Cuánto sobreestima a la izquierda frente a la derecha', 'A la izquierda del consenso de las demás encuestadoras']
    assert context['kpis'][0]['value'] == '{} de {}'.format(summary['with']['k_left'], summary['with']['n'])
    assert [t['key'] for t in context['tests']] == ['coin', 'dates', 'consensus'] and context['tests'][2]['highlight'] is True
    assert set(context['initial']) == {'state', 'bull', 'matrix', 'consensus', 'ranking', 'election', 'distance'}
    assert [link['label'] for link in context['csv_links']] == list(pages.CSV_LABELS['sesgo-cis'].values())
    assert context['elections'][0]['event_date'] >= context['elections'][-1]['event_date'] and all(e['period'] == 'with' for e in context['elections'])
    assert context['verdicts'][0]['label'] == 'SÍ' and context['verdicts'][1]['tone'] == 'q3'

def test_lean_context_survives_missing_parts(api, fresh_app):
    publish_study(fresh_app.fs, study='sesgo-cis')
    fresh_app.fs.remove(bundle.path_study('sesgo-cis', 'election')); webapi.site().manifest_cache_clear()
    context = pages.lean_context(pages.parse_state({}))
    assert context['election'] is None and context['initial']['distance'] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pages.py -q -k "lean or package_layout"`
Expected: FAIL (`pages.lean_context` missing; layout list).

- [ ] **Step 3: Implement `pages/lean.py` and `lean_context`**

`pages/lean.py` imports only `filters` (`fmt_num`, `fmt_int`, `delta_text`) and `slugs`; every function returns `None`/empty for missing keys (`_get` pattern of `cera.py`). `lean_context` in `analisis.py` mirrors `cera_context`: `parts = {part: study_part(LEAN, part) or {} for part in bundle.STUDY_PARTS[LEAN]}`, `summary = parts['summary']`, `periods = _get(meta, 'periods')`, `published_at` from `site().manifest()['studies']['sesgo-cis']['published_at']` guarded (`None` without it). Re-export `lean_context` and `lean` names in `pages/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pages.py -q -k "lean or package_layout or analisis"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/lean.py mtpy/lib/pages/analisis.py mtpy/lib/pages/__init__.py tests/test_pages.py
git commit -m "Sesgo del CIS: contexto de la página y modelos de vista"
```

---

### Task 8: The template `analisis_sesgo_cis.html`

**Files:**
- Create: `web/templates/analisis_sesgo_cis.html`
- Modify: `web/dist/css/site.css` (after `:895`, the study block): `.chart.square`, `.chart.square.wide`, `.cmp`, `.cmp .box`, `.cmp .box.after`, `.tests`, `.test`, `.test.hi`, `.tech`, `.chart-scroll`, `.qcard .chart#matrix-chart { height: auto; }`, `.qcard .chart#distance-chart { height: auto; }`, mobile rules for `.cmp`/`.tests` (one column under 720 px)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `lean_context` (Task 7); chart ids `bull-without`, `bull-with`, `matrix-chart`, `consensus-chart`, `ranking-chart`, `bull-2023`, `distance-chart`; table ids `elections-table` (+ button `elections-more`), `election-table`.

- [ ] **Step 1: Write the failing tests**

```python
def test_lean_page_structure_and_wording(api, fresh_app):
    publish_study(fresh_app.fs, study='sesgo-cis')
    status, _, body = call(api, '/analisis/sesgo-cis'); page = html(body)
    assert status == 200 and '<h1 class="study-title">El sesgo del CIS de Tezanos</h1>' in page
    assert '/dist/js/pages/analisis-sesgo-cis.js' in page and 'id="initial-data"' in page
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page) == ['SÍ', 'NO', 'NO']
    assert page.count('<div class="kpi">') == 3 and 'Cuánto sobreestima a la izquierda frente a la derecha' in page
    for chart in ('bull-without', 'bull-with', 'matrix-chart', 'consensus-chart', 'ranking-chart', 'bull-2023', 'distance-chart'):
        assert re.search(r'<div class="chart[^"]*" id="%s" role="img" aria-label="[^"]+"></div>' % chart, page), chart
    assert 'class="chart square" id="bull-with"' in page and 'class="chart square wide" id="bull-2023"' in page
    assert page.count('class="pill brand" href="/api/v1/studies/sesgo-cis/') == 4
    assert 'Diferencia</td>' in page and 'Ganador' not in page and 'el CIS es la única casa prácticamente monocolor' in page
    assert '<button type="button" class="more" id="elections-more"' in page
    assert_public_wording(page, '/analisis/sesgo-cis')
    assert 'empresa' not in visible_text(page).lower() and 'max-width' not in page

def test_analisis_lists_each_study_only_when_published(api, fresh_app):
    assert call(api, '/analisis/sesgo-cis')[0] == 404
    publish_study(fresh_app.fs, study='sesgo-cis')
    page = html(call(api, '/analisis')[2]); home = html(call(api, '/')[2])
    assert page.count('class="card study-card"') == 1 and 'href="/analisis/sesgo-cis"' in page and 'href="/analisis/sesgo-cis"' in home
    publish_study(fresh_app.fs)
    page = html(call(api, '/analisis')[2])
    assert [m for m in re.findall(r'href="(/analisis/[a-z-]+)"', page)] == ['/analisis/voto-cera', '/analisis/sesgo-cis']
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pages.py -q -k "lean_page or lists_each_study"`
Expected: FAIL (template missing → `TemplateNotFound`).

- [ ] **Step 3: Write the template**

Follow `analisis_voto_cera.html` for the skeleton (blocks, `#study-head` with the pills «Análisis», «Evidencia histórica · {{ summary['with'].n|int }} elecciones desde 2018», «Actualizado el {{ published_at|date }}» when present; `nav.toc` with the five links; `m.kpis(kpis)`; `resumen` with three `answer` calls; `#p1`, `#p2`, `#p3` as `qcard`s; `#datos`). Texts: the mockup's, with the variables of the context; the lead of `#p1` with the verified facts (Tezanos's role and dates, the method change); the `.sum` of every question; the `.tech` note; «Cómo leerlo» (`m.how`) in the three question cards (3-4 sentences each). Tables: `#elections-table` (Elección, Fecha, Días antes, Izquierda, Derecha with `shade_around(0, 8, '#d62728', '#2ca02c')`, rows after the eighth `hidden`, button `elections-more` «Ver las {{ n }} elecciones»); `#election-table` with the rows Izquierda, Derecha, Diferencia (`.pos`/`.neg` on the error cells). Every chart `div.chart` carries `role="img"` and an `aria-label` with live figures; `#matrix-chart` sits inside `<div class="chart-scroll">` (horizontal scroll on phones). Guard every optional with `{% if %}` (`StrictUndefined`). Then try `assert_public_wording` on `/analisis/voto-cera` in `test_public_wording_uses_encuestadora_and_medio_and_short_names`: add the path if it passes; if it fails, leave it out and note the offending text in the task notes (spec D6).

- [ ] **Step 4: Run the tests and the strict-template compile**

Run: `python -m pytest tests/test_pages.py -q -k "lean or templates_compile or analisis or inicio"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/templates/analisis_sesgo_cis.html web/dist/css/site.css tests/test_pages.py
git commit -m "Sesgo del CIS: la página /analisis/sesgo-cis"
```

---

### Task 9: Chart modules: `matrix.js`, `ranking.js`, `distance.js`, and the bullseye/scatter options

**Files:**
- Create: `web/dist/js/charts/matrix.js`, `web/dist/js/charts/ranking.js`, `web/dist/js/charts/distance.js`
- Modify: `web/dist/js/charts/bullseye.js:141-240` (`interval`, `highlights`), `web/dist/js/charts/scatter.js` (`diagonal`, `axes`, `colorOf`)
- Test: `tests/test_js_charts.py:43-70` (imports `MATRIX_JS` as `mx`, `RANKING_JS` as `rk`, `DISTANCE_JS` as `ds`) and new tests

**Interfaces:**
- Produces: `matrixOption({events, houses, cells}, {cap = 8, positive = '#d62728', negative = '#1d4f86'} = {})` + `renderMatrix(el, data, opts)` (sets `el.style.height = houses.length * 26 + 70`); `rankingOption(rows, {pollster = 'CIS', xMax = 100, yMin = -2, yMax = 10, yInterval = 2} = {})` + `renderRanking(el, rows, opts)` with rows `{x (0–100), y, name, k, n, cis}`: series in this order: the two dashed guide lines (`x = 50`, `y = 0`), the «Sin sesgo» point (label `position: 'bottom'`), the pack scatter with `markArea` (one area, `label.show: false`, bounds `min/max ± 4 %` and `± 0.5 points`) and `labelLayout: {hideOverlap: true}`, the highlighted pollster in red with label `'{name} · {k} de {n} · {y} puntos'` at `position: 'left'`; tooltip text «Sobrestimó a la Izquierda en {k} de {n} elecciones ({x}%) · {y} puntos de media»; `distanceOption(rows)` + `renderDistance(el, rows)` with rows `{name, dist, cis, consensus}`: horizontal bars in row order, `itemStyle.color` red for `cis`, `THEME.textColor` for `consensus`, grey otherwise, value label right, no x axis labels, height `rows.length * 15 + 30`; `bullseyeOption` gains `interval` (number: axis `interval`, `limit` = the smallest multiple of `interval` strictly above the reach, min/max labels hidden) and `highlights` (array of `{row, color, symbol = 'circle', label = null}`; each one a series with `id: 'highlight-' + index` at `HIGHLIGHT_Z` with `symbol` and, with `label`, a label `position: 'right'`; the existing single `highlight` keeps working with `id: 'highlight'`); `scatterOption` gains `diagonal` (boolean: a dashed line from `[min, min]` to `[max, max]` of the fixed axes), `axes` (`{xMin, xMax, yMin, yMax, interval}`: fixed limits and `interval`, `scale: false`) and `colorOf` (`row => colour`, overriding the accent per point).

- [ ] **Step 1: Write the failing tests**

```python
def test_matrix_option_colours_by_sign_and_bolds_the_general_elections():
    option = run_js('mx.matrixOption({events: [{short: "AN 18", label: "Andalucía 2018"}, {short: "GEN 23", label: "Generales 2023"}], houses: ["CIS", "GAD3"], cells: [[0, 0, 12.6], [1, 1, -4.0]]})')
    assert option['xAxis']['position'] == 'top' and option['yAxis']['inverse'] is True and option['series'][0]['type'] == 'heatmap'
    colours = [cell['itemStyle']['color'] for cell in option['series'][0]['data']]
    assert colours[0].startswith('rgba(214,39,40') and colours[1].startswith('rgba(29,79,134')

def test_ranking_option_marks_the_centre_the_pack_and_the_pollster():
    rows = [{'x': 93, 'y': 8.0, 'name': 'CIS', 'k': 38, 'n': 41, 'cis': True}, {'x': 77, 'y': 1.6, 'name': 'Sigma Dos', 'k': 23, 'n': 30, 'cis': False}]
    option = run_js('rk.rankingOption(%s)' % json.dumps(rows))
    kinds = [s['type'] for s in option['series']]
    assert kinds == ['line', 'line', 'scatter', 'scatter', 'scatter'] and option['yAxis']['interval'] == 2 and option['xAxis']['max'] == 100
    assert option['series'][3]['markArea']['label']['show'] is False and option['series'][4]['data'][0]['value'][:2] == [93, 8.0]

def test_distance_option_colours_the_pollster_and_the_consensus():
    option = run_js('ds.distanceOption([{name: "Opinometre", dist: 0.5}, {name: "Consenso (20 casas)", dist: 3.3, consensus: true}, {name: "CIS", dist: 4.2, cis: true}])')
    colours = [d['itemStyle']['color'] for d in option['series'][0]['data']]
    assert colours[2] == '#ce3b3b' and colours[1] == '#1a1c22' and option['xAxis']['axisLabel']['show'] is False

def test_bullseye_interval_and_highlights():
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], interval: 5, highlights: [{row: rows[0], color: "#ce3b3b", label: "CIS"}, {row: rows[1], color: "#1a1c22", symbol: "diamond", label: "Consenso"}]})')
    assert option['xAxis']['interval'] == 5 and option['xAxis']['max'] % 5 == 0 and option['xAxis']['max'] > 10
    marked = [s for s in option['series'] if s.get('id', '').startswith('highlight')]
    assert len(marked) == 2 and marked[1]['symbol'] == 'diamond' and marked[0]['label']['show'] is True

def test_scatter_diagonal_and_fixed_axes():
    option = run_js('sc.scatterOption([{x: 1, y: 2, name: "a"}], {x: "x", y: "y", name: "name", diagonal: true, axes: {xMin: -10, xMax: 15, yMin: -5, yMax: 15, interval: 5}})')
    assert option['xAxis']['min'] == -10 and option['yAxis']['interval'] == 5 and option['series'][-1]['data'] == [[-10, -10], [15, 15]]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_js_charts.py -q -k "matrix or ranking or distance or interval or diagonal"`
Expected: FAIL (modules missing / options ignored).

- [ ] **Step 3: Implement the modules and the options**

Follow `heatmap.js` for `matrix.js` (no `visualMap`; per-cell `itemStyle.color` = `rgba(214,39,40,a)` positive / `rgba(29,79,134,a)` negative with `a = 0.15 + 0.75 · min(|v|, cap) / cap`, `borderColor '#fff'`, `borderWidth 2`; x labels rotated 90, `rich` bold for `short` starting with `GEN`; tooltip «{house} · {label}: error en el bloque de izquierdas {valor}» with `fmtNum(v, 1)` and a sign). `ranking.js` and `distance.js` follow `scatter.js`/`bars.js` style (`baseOption`, `tooltipStyle`, `axisLabelStyle`, `mountChart`). In `bullseye.js` keep `highlight` (single) working and add `highlights`; in `scatter.js` keep every existing test green.

- [ ] **Step 4: Run the JS tests**

Run: `python -m pytest tests/test_js_charts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/matrix.js web/dist/js/charts/ranking.js web/dist/js/charts/distance.js web/dist/js/charts/bullseye.js web/dist/js/charts/scatter.js tests/test_js_charts.py
git commit -m "Gráficos: matriz de signos, ranking, distancia al resultado; marcas y realces en la diana, diagonal en la dispersión"
```

---

### Task 10: The page script `analisis-sesgo-cis.js`

**Files:**
- Create: `web/dist/js/pages/analisis-sesgo-cis.js`
- Test: `tests/test_pages.py` (island shape, already in Task 7) and a manual browser check (Step 4)

**Interfaces:**
- Consumes: the `initial` island of Task 7 and the modules of Task 9.

- [ ] **Step 1: Write the script**

Pattern of `analisis-cera.js`: `readInitial`, `tryDraw`, `wireMore('elections-more', 'elections-table')`, `mountWhenReady(paint)`. `paint()`: `renderBullseye` on `#bull-without` and `#bull-with` with `{blocks, groupOf: row => row.scope === 'es' ? 'generales' : 'autonómicas', colors: {generales: THEME.accentColor, autonómicas: '#c0672e'}, labelOf: row => row.event_date.slice(0, 4), onTop: g => g === 'generales', interval: 5, labels: true}` (labels only on generales: `labelOf` returns `''` for the rest); `renderMatrix('#matrix-chart', data.matrix)`; `renderScatter('#consensus-chart', data.consensus, {x: 'x', y: 'y', name: 'label', labelled: [national labels], diagonal: true, axes: {xMin: -10, xMax: 15, yMin: -5, yMax: 15, interval: 5}, colorOf: row => row.national ? THEME.accentColor : '#f28e2b', xName: 'Media de las demás casas, error en la izquierda', xLabel: 'demás casas', yLabel: 'CIS'})`; `renderRanking('#ranking-chart', data.ranking)`; `renderBullseye('#bull-2023', data.election.houses, {blocks, color: THEME.accentColor, interval: 2, labelOf: row => extremes only, highlights: [{row: cis, color: '#ce3b3b', label: 'CIS'}, {row: consensus, color: THEME.textColor, symbol: 'diamond', label: 'Consenso'}]})`; `renderDistance('#distance-chart', data.distance)`. Skip a chart whose container or data is missing (`container(id, ready)` as in the CERA script).

- [ ] **Step 2: Run the page tests**

Run: `python -m pytest tests/test_pages.py -q -k lean`
Expected: PASS (the script tag is asserted in Task 8).

- [ ] **Step 3: Lint the module with node**

Run: `node --check web/dist/js/pages/analisis-sesgo-cis.js && node --input-type=module -e "await import('file://$PWD/web/dist/js/charts/matrix.js'); console.log('ok')"`
Expected: `ok`.

- [ ] **Step 4: Browser check with a local publication**

Run (local bundle, no S3): `S3_BUCKET= python job.py publish '{"what":["study"],"study":"sesgo-cis"}'` against a local `files/site/v1` that already holds the analyses (if the local bundle lacks them, sync `site/v1/analysis/` and `site/v1/events/es/2023-07-23/` from S3 first as the brainstorm did), then serve the web and open `/analisis/sesgo-cis` at 1280 px and 390 px: every chart draws, the bullseyes are square, the matrix scrolls horizontally on mobile, no console errors. Record the outcome in the task notes.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/pages/analisis-sesgo-cis.js
git commit -m "Sesgo del CIS: los gráficos de la página"
```

---

### Task 11: Note in the Método card and the memory of the section

**Files:**
- Modify: `web/templates/metodo_promedio.html:452-466` (end of the `ejemplo-cis` card), `mtpy/lib/pages/metodo.py` (`metodo_section_context`: `lean_published: bool` = `site().study_exists`-like check through `bundle_part(site().study, 'sesgo-cis', 'meta') is not None`)
- Test: `tests/test_pages_ejemplo.py`

- [ ] **Step 1: Write the failing test**

```python
def test_method_card_links_the_study_only_when_published(house_site, fresh_app):
    page = html(call(house_site, '/metodo/promedio')[2])
    assert '/analisis/sesgo-cis' not in page
    write_study(fresh_app.fs, study='sesgo-cis'); webapi.site().manifest_cache_clear()
    page = html(call(house_site, '/metodo/promedio')[2])
    assert 'El análisis completo, con el CIS de antes de 2018, las demás casas y 2023, está en <a href="/analisis/sesgo-cis">' in page
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_pages_ejemplo.py -q -k links_the_study`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `lean_published` to the `promedio` section context and the sentence at the end of the last paragraph of the card, guarded by `{% if lean_published %}`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_pages_ejemplo.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/templates/metodo_promedio.html mtpy/lib/pages/metodo.py tests/test_pages_ejemplo.py
git commit -m "Método: enlace al análisis del sesgo del CIS cuando está publicado"
```

---

### Task 12: Whole suite, spec close and hand-over notes

**Files:**
- Modify: `docs/superpowers/specs/2026-10-24-web-analisis-sesgo-cis-design.md` (state line and a «Cierre» paragraph with the commits and the test count)

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS (plus the skips the suite already has); note the count.

- [ ] **Step 2: Publish for real and walk the page (Luis's checklist, written in the spec close)**

Commands for Luis, in order: `python job.py publish '{"what":["study"],"study":"sesgo-cis"}'` against the S3 bundle (the analyses are already published there), then walk `/analisis`, `/analisis/sesgo-cis`, `/` (band) and `/metodo/promedio` (the link), and deploy with the next integration.

- [ ] **Step 3: Close the spec and commit**

```bash
git add docs/superpowers/specs/2026-10-24-web-analisis-sesgo-cis-design.md
git commit -m "Spec del sesgo del CIS: cierre con los commits y las pruebas"
```
