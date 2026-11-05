# Análisis «¿Voto oculto de quién?» Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a fourth study, `voto-oculto`, computed from the already-published parts of the bundle (the final polls of the 92 published elections against their results), republished with every `update`, and its live page `/analisis/voto-oculto` with the approved mockup's structure, verdicts, conditioned texts and three charts.

**Architecture:** A light package `mtpy/lib/studies/shy/` reads `meta`/`cycle`/`results` of every published election of every scope of the catalogue (`data/es-scopes.csv`) and `meta`/`polls` of the latest national run through a `BundleReader`, measures the signed error of the final polls (14 days) for the two blocks and four party lineages with the sum rule, and `mtpy/lib/publish/shy.py` exports four parts. The study is `live`, so the existing `studies` action of `publish` (last step of `update`) republishes it. The page follows `/analisis/rebano`: view models in `pages/shy.py`, a Jinja template with the verdicts computed in the study, a page script drawing a new `charts/errors.js` (signed bars and a strip by party).

**Tech Stack:** Python 3.11, pandas, numpy, scipy (`ttest_1samp`, `binomtest`), Jinja2, pytest; vanilla ES modules on ECharts 5 tested with `node`.

**Spec:** `docs/superpowers/specs/2026-11-05-web-analisis-voto-oculto-design.md` (approved 2026-11-06; mockup v4 `docs/superpowers/specs/2026-11-05-web-analisis-voto-oculto-mockup.html`).

## Global Constraints

- No database access anywhere in the study or the page: inputs come only from the bundle (`BundleReader(fs, prefix=bundle.PREFIX)`, also in a dry run) and from `data/` (`voto-oculto/params.json`, `voto-oculto/incumbents.json`, `es-scopes.csv`), read with the `FileSystem` that `publish_study` passes to `compute`.
- `voto-cera`, `sesgo-cis` and `rebano` keep their parts, schemas, CSV, registration and tests. `bundle.STUDIES` gains `'voto-oculto'` **at the end of the tuple as it stands** when the task runs (the `andorra` study is being built in parallel in the worktree `analisis-andorra` and appends its own slug the same way; tests assert membership and order after `rebano`, never the exact tuple).
- `mtpy.lib.publish` stays light: the study package and `scipy` are imported inside functions (`test_importing_publish_does_not_load_the_study_package` gains `mtpy.lib.studies.shy`); `pages` and `webapi` never import `publish` or the model.
- Code identifiers in English («shy» = voto oculto); Spanish only in public texts. Public wording: «sólo» with accent, «encuestadora»/«casa», never «empresa», «actualización» for the model runs, quotes «»; `assert_public_wording` passes on `/analisis/voto-oculto`.
- Every figure of the page comes from the published parts (never typed into the template); every strong sentence is conditioned as spec D6 says and has a neutral fallback. The verdicts are computed in the study (`summary.verdicts`), the page only paints them.
- Parameters (spec D4, `data/voto-oculto/params.json`): `window_days 14`, `recent_from "2018-09-01"`, `blocks ["Izquierda", "Derecha"]`, `lineages {"PSOE": ["PSOE", "PSC", "PSE-EE"], "PP": ["PP", "AP", "CD"], "VOX": ["VOX"], "UP": ["SUMAR", "FA", "UP", "IU", "PCE"]}`, `labels {"UP": "IU / Podemos / Sumar", "PP": "PP (AP, CD)", "VOX": "Vox"}`, `size_parties ["PSOE", "PP", "VOX", "UP"]`, `size_threshold 2`, `big_threshold -3`, `pp_early ["1982-01-01", "1996-01-01"]`, `extremes 3`.
- The measure (spec D4): a **final poll** has `weight > 0` and its fieldwork end (`end_date`, or `date` without it) between 0 and `window_days` days before the election, both ends included (for the current run, before `as_of`). A **group error** (block or lineage, the same rule) is, per final poll, the sum of the group's members present in `results` that the poll estimates (not null) minus the sum of the results of those same members, averaged over the polls; `None` when no member is in `results` or no poll estimates one. Negative = underestimated. `lean = Izquierda − Derecha` (`None` if either is missing).
- A **figures** block is always `{n, under, over, mean, median, p_t, p_sign, size, size_over}`: `under` = errors < 0, `over` = errors > 0, `p_t` = `scipy.stats.ttest_1samp(values, 0).pvalue` (`None` with `n < 3`), `p_sign` = `binomtest(max(under, over), n, alternative='greater').pvalue` (one-sided), `size` = mean `|error|`, `size_over` = share of `|error| > size_threshold`; `None` for an empty list.
- Rounding: errors, means, medians and sizes to 2 decimals; shares to 2 decimals; p values with three significant figures (`float(f'{p:.3g}')`, as `publish/lean.py::_significant`). Figures are computed over unrounded errors; the rows of `elections` carry the rounded ones.
- Fluid layout, charts mounted with `mountChart`, every chart module exports its `*Option` for `tests/test_js_charts.py`.
- Git: Luis commits in parallel in the same tree. Before each commit run `git status --short` and `git log -1`, add only the task's files by path, never `git commit -a`, never rewrite history. Commit messages in the repo's style («Área: qué cambia»), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run tests with `python -m pytest -q` from the repo root; JS tests need `node` on PATH (`tests/test_js_charts.py` skips without it). Never run a job against the database or S3 from a task.

## Review Focus

1. A lineage present in the results that no final poll estimates (Vox in the 2018-2019 regionals where the polls carry it as null): the cell is `None`, not `0`, and the election does not count in `vox.n` — Task 3 `test_group_error_is_none_when_no_poll_estimates_the_party`.
2. A scope of the catalogue without an events folder, or one whose `list_events` raises: skipped and annotated in `meta.inputs`, the study goes on with the others — Task 2 `test_read_elections_skips_a_scope_without_events`.
3. The election of the national run gets published with results (after the 29-N): `current` is `None`, the election enters `elections` with the incumbent of its row (`es/2026-11-29: PSOE`) and the verdicts are recomputed over it — Task 4 `test_compute_shy_moves_the_current_election_to_the_past_once_published`.
4. The run's `meta.bmaps` without `blocks` or without a `main` party in the window: `current.values` has only the keys with data and the card «Después del 29-N» shows only those columns, never a crash — Task 2 `test_read_current_without_blocks`, Task 6 `test_shy_page_after_card_shows_only_the_values_present`.
5. The verdicts flip (`left.all.mean < 0` with `p_t < 0.05`, or a party both negative and significant): P1 «SÍ», the Tezanos pill «Confirmada», P3 «DE {label}» with the backing sentences, so a future recount never shows «NO» over contradicting figures — Task 4 `test_verdicts_in_the_three_cases`, Task 6 `test_shy_page_conditions_follow_the_figures`.

---

### Task 1: Study registry, schemas, fixtures and CSV labels

**Files:**
- Modify: `mtpy/lib/bundle.py:79-93` (`STUDIES`, `STUDY_PARTS`, `STUDY_SCHEMAS`, `STUDY_CSV`, `STUDY_CSV_OF`), `:177-186` (`SCHEMAS`, after the `rebano-*` entries), module docstring tree `:33`
- Modify: `mtpy/lib/pages/analisis.py:42-67` (`CSV_LABELS`)
- Create: `tests/fixtures/bundle/voto-oculto-meta.json`, `voto-oculto-summary.json`, `voto-oculto-elections.json`, `voto-oculto-groups.json` (envelopes like `rebano-summary.json`: `schema`, `contract 1`, `scope null`, `run_id null`, `mode null`, `generated_at`, `data`)
- Modify: `tests/test_bundle_unit.py:86-105` (registry test), `tests/test_web_api.py` (next to `:109`)
- Test: `tests/test_bundle_unit.py`, `tests/test_webapi_unit.py`, `tests/test_web_api.py`

**Interfaces:**
- Produces: `'voto-oculto' in bundle.STUDIES` after `'rebano'`; `STUDY_PARTS['voto-oculto'] == ('meta', 'summary', 'elections', 'groups')`; `STUDY_SCHEMAS['voto-oculto'][part] == 'voto-oculto-' + part`; `STUDY_CSV['voto-oculto'] == ('elections', 'groups')`; `STUDY_CSV_OF['voto-oculto'] == {'elections': ('elections',), 'groups': ('groups',)}`; `SCHEMAS` entries: `'voto-oculto-meta': {'study': str, 'title': str, 'subtitle': str, 'scopes': list, 'election': (dict, type(None)), 'params': dict, 'inputs': dict, 'sources': list, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'seconds': dict}`, `'voto-oculto-summary': {'n_elections': int, 'n_general': int, 'n_regional': int, 'n_scopes': int, 'first_year': int, 'first_year_regional': (int, type(None)), 'last_year': int, 'left': dict, 'right': dict, 'lean': dict, 'vox': (dict, type(None)), 'parties': list, 'size': dict, 'psoe_general': (dict, type(None)), 'pp': dict, 'podemos': dict, 'current': (dict, type(None)), 'verdicts': dict}`, `'voto-oculto-elections': {'rows': list}`, `'voto-oculto-groups': {'rows': list}`; `pages.analisis.CSV_LABELS['voto-oculto'] == {'elections': 'Error de los sondeos finales por elección, bloque y partido', 'groups': 'Resumen por grupo de elecciones y por partido'}`.
- Fixture content (the mockup v4 figures, so the page tests read like the mockup): `summary` with `n_elections 92, n_general 16, n_regional 76, n_scopes 18, first_year 1977, first_year_regional 2009, last_year 2026`; `left {all: {n 92, under 40, over 52, mean 0.47, median 0.63, p_t 0.128, p_sign 0.126, size 2.42, size_over 0.52}, general: {16, 10, 6, -0.46, -0.48, 0.523, 0.227, 2.35, 0.5}, regional: {76, 30, 46, 0.67, 0.91, 0.052, 0.042, 2.43, 0.53}, recent: {41, 17, 24, 0.8, 0.61, 0.059, 0.174, 2.04, 0.44}}` (the nine keys in that order); `right {all: {92, 45, 47, 0.63, 0.45, 0.069, 0.459, 2.66, 0.52}, general: {16, 8, 8, 0.82, 0.24, 0.196, 0.598, 2.1, 0.44}, regional: {76, 37, 39, 0.59, 0.46, 0.141, 0.454, 2.77, 0.54}, recent: {41, 20, 21, 0.4, 0.3, 0.3, 0.5, 2.1, 0.44}}`; `lean {all: {92, 43, 49, -0.16, 0.46, 0.778, 0.301, 4.21, 0.75}, general: {16, 9, 7, -1.28, -1.78, 0.324, 0.402, 4.38, 0.81}, regional: {76, 34, 42, 0.08, 0.51, 0.894, 0.211, 4.18, 0.74}, recent: {41, 19, 22, 0.4, 0.5, 0.6, 0.4, 3.9, 0.7}}`; `vox` = `{n 39, under 22, over 17, mean -0.33, median -0.17, p_t 0.278, p_sign 0.261, size 1.37, size_over 0.18, extremes: {under: [{label 'Andalucía 2018', error -6.01}, {label 'Castilla y León 2022', error -3.92}, {label 'Canarias 2023', error -3.68}], over: [{label 'Andalucía 2022', error 3.39}, {label 'Asturias 2019', error 2.57}, {label 'La Rioja 2019', error 2.13}]}}`; `parties` the eight rows `[{key, label, figures}]` in this order: `PSOE` «PSOE» `{92, 49, 43, -0.12, -0.3, 0.679, 0.301, 2.26, 0.47}`, `PP` «PP (AP, CD)» `{91, 46, 45, 0.02, -0.14, 0.93, 0.5, 2.12, 0.44}`, `VOX` «Vox» (the `vox` figures without `extremes`), `UP` «IU / Podemos / Sumar» `{77, 26, 51, 0.8, 0.86, 0.00263, 0.00294, 1.78, 0.3}`, `Izquierda` «Izquierda (bloque)» (= `left.all`), `Derecha` «Derecha (bloque)» (= `right.all`), `incumbent_general` «Gobierno, generales» `{16, 11, 5, -1.23, -1.53, 0.0793, 0.105, 2.4, 0.56}`, `incumbent_regional` «Gobierno, autonómicas» `{76, 34, 42, 0.23, 0.35, 0.504, 0.211, 2.4, 0.5}`; `size {n_pairs 299, mean 1.98, share_over 0.38}`; `psoe_general` `{16, 10, 6, -1.29, -1.27, 0.0848, 0.227, 2.55, 0.56}` plus `big: ['Generales 1977', 'Generales 1986', 'Generales 1993', 'Generales 1996', 'Generales 2004', 'Generales 2023']`; `pp {early: {from '1982-01-01', to '1996-01-01', errors: [{label 'Generales 1982', error -5.26}, {label 'Generales 1986', error -1.16}, {label 'Generales 1989', error -3.36}, {label 'Generales 1993', error -0.48}]}, late: {10, 4, 6, 0.6, 0.65, 0.507, 0.377, 2.26, 0.5}, all: <the PP figures>}`; `podemos {before: {42, 22, 20, -0.06, -0.05, 0.866, 0.439, 1.59, 0.21}, general_recent: {3, 2, 1, -0.29, -0.57, 0.679, 0.5, 0.86, 0.0}, regional_recent: {32, 2, 30, 2.03, 1.94, 1.6e-07, 1.5e-07, 2.11, 0.44}}`; `current {as_of '2026-11-05', days_left 24, n_polls 12, from '2026-10-22', to '2026-11-05', values: {Izquierda 35.4, Derecha 52.6, PSOE 26.4, PP 31.3, VOX 19.5, FA 6.5}, names: ['PP', 'PSOE', 'VOX', 'FA']}`; `verdicts {p1 'NO', p2 'NO', p3 'DE NADIE'}`. `elections.rows`: eight rows with the fields of spec D3 (`scope, scope_name, event_date, label, short, general, n_cycle, n_final, blocks, Izquierda, Derecha, lean, PSOE, PP, VOX, UP, names, incumbent, incumbent_error`): `es 1977-06-15` («Generales 1977», `GEN 77`, n_final 7, Izq −1.82, Der 2.88, lean −4.7, PSOE −4.42, PP 0.6 (`AP`), VOX null, UP −1.04 (`PCE`), incumbent `UCD` −1.87), `es 1982-10-28` (2, 3.53, −1.47, 5.0, 0.79, −5.26 `AP`, null, 2.74 `PCE`, `UCD` 1.23), `es 2019-04-28` («Generales abr. 2019», `GEN 19A`, 14, −0.5, 3.41, −3.92, 0.47, 3.97, 0.83, −1.16, `PSOE` 0.47), `es 2019-11-10` («Generales nov. 2019», `GEN 19N`, 17, −0.01, 0.5, −0.52, −0.76, 0.14, −2.0, −0.57, `PSOE` −0.76), `es 2023-07-23` (20, −2.56, 1.87, −4.43, −3.42, 1.15, 0.71, 0.86 (`SUMAR`), `PSOE` −3.42), `es-an 2018-12-02` («Andalucía 2018», `AN 18`, 8, 8.77, −6.13, 14.89, 4.96, 1.05, −6.01, 3.87, `PSOE` 4.96), `es-md 2021-05-04` («Madrid 2021», `MD 21`, 18, 3.99, −3.7, 7.69, 5.08, −3.78, −0.04, 0.23, `PP` −3.78), `es-cl 2026-03-15` («Castilla y León 2026», `CL 26`, 10, 0.61, −1.16, 1.77, −2.39, −2.62, 1.2, 3.33 (`UP+IU`), `PP` −2.62); `blocks` of each row the two lists of the real event (`Izquierda: ['PSOE', 'SUMAR']` style, any plausible list), `n_cycle` ≥ `n_final`, and the two dicts `results` and `polls` for the six keys `Izquierda, Derecha, PSOE, PP, VOX, UP` (any plausible shares with `polls − results` equal to the row's error, `null` where the error is `null`; for `es 2023-07-23`: `results {Izquierda 44.0, Derecha 45.5, PSOE 31.7, PP 33.1, VOX 12.4, UP 12.3}`). `groups.rows`: the long rows of every figures block above (`table` ∈ `left`, `right`, `lean`, `vox`, `parties`, `podemos`, `pp`; `key` the sub-key; `label` the Spanish label of the table row). `meta`: `study 'voto-oculto'`, the title «¿Voto oculto de quién?», the mockup v4 subtitle, `scopes` the 18 codes, `election {date '2026-11-29', days_left 24, as_of '2026-11-05', run_id '20261105-103609'}`, `params` of the Global Constraints, `inputs {es: {events: 16, skipped: []}, 'es-cb': {events: 2, skipped: [{date '2011-05-22', reason 'no final polls'}, {date '2019-05-26', reason 'no final polls'}]}, run: {run_id, as_of}, incumbents: []}`, `sources` the four references of spec «Hechos del contexto» (`{label, url}`), `commit`, `dirty false`, `versions`, `seconds`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_bundle_unit.py::test_study_registry_is_per_study` (`:86`) add:

```python
assert 'voto-oculto' in bundle.STUDIES and bundle.STUDIES.index('voto-oculto') > bundle.STUDIES.index('rebano')
assert bundle.STUDY_PARTS['voto-oculto'] == ('meta', 'summary', 'elections', 'groups')
assert bundle.STUDY_SCHEMAS['voto-oculto'] == {part: 'voto-oculto-' + part for part in bundle.STUDY_PARTS['voto-oculto']}
assert bundle.STUDY_CSV['voto-oculto'] == ('elections', 'groups')
assert bundle.path_study('voto-oculto', 'groups') == 'studies/voto-oculto/groups.json'
assert bundle.path_study_csv('voto-oculto', 'elections') == 'studies/voto-oculto/csv/elections.csv'
with pytest.raises(ValueError):
    bundle.path_study('voto-oculto', 'houses')
```

The parametrised `test_study_schemas_validate_the_fixtures` (`:292`) covers the four new fixtures by itself. In `tests/test_web_api.py` add `test_study_routes_serve_voto_oculto_parts_and_csv` (pattern of `:109`): `write_study(fresh_app.fs, study='voto-oculto')`, then `GET /api/v1/studies/voto-oculto/summary` is 200 with `data.verdicts == {'p1': 'NO', 'p2': 'NO', 'p3': 'DE NADIE'}`, `GET /api/v1/studies/voto-oculto/elections?format=csv` is 200 as `voto-oculto-elections.csv`, and `GET /api/v1/studies/voto-oculto/houses` is 404.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_web_api.py -q -k "registry or fixtures or voto_oculto"`
Expected: FAIL (`'voto-oculto' not in STUDIES`; fixtures missing).

- [ ] **Step 3: Declare the study and write the fixtures**

Add the entries of the Interfaces block to `mtpy/lib/bundle.py` (constants and `SCHEMAS`; the docstring tree line becomes «(voto-cera, sesgo-cis, rebano, voto-oculto)»), the four fixture files and `CSV_LABELS['voto-oculto']`.

- [ ] **Step 4: Run the affected suites**

Run: `python -m pytest tests/test_bundle_unit.py tests/test_web_api.py tests/test_webapi_unit.py tests/test_pages.py -q`
Expected: all PASS (`/analisis/voto-oculto` is still 404: `pages.analisis.STUDIES` has no template until Task 6).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/pages/analisis.py tests/test_bundle_unit.py tests/test_web_api.py tests/fixtures/bundle/voto-oculto-meta.json tests/fixtures/bundle/voto-oculto-summary.json tests/fixtures/bundle/voto-oculto-elections.json tests/fixtures/bundle/voto-oculto-groups.json
git commit -m "Estudios: registro, esquemas y fixtures del estudio voto-oculto"
```

---

### Task 2: Parameters, the Government table, the catalogue and the readers (`studies/shy/data.py`)

**Files:**
- Create: `mtpy/lib/studies/shy/__init__.py` (docstring: the shy vote study, the final polls against the results, from the published parts), `mtpy/lib/studies/shy/data.py`
- Create: `data/voto-oculto/params.json`, `data/voto-oculto/incumbents.json`; Modify: `data/README.md` (two table rows after `rebano/params.json` `:32`: `voto-oculto/params.json` | Parámetros del estudio del voto oculto (ventana de sondeos finales, periodo reciente, bloques, linajes y etiquetas, partidos del tamaño del error, umbrales, periodo del voto oculto del PP, extremos) | Curación propia. · `voto-oculto/incumbents.json` | Partido del Gobierno (de España o de la comunidad) el día de cada elección publicada, por ámbito y fecha, con nombres alternativos | Curación propia (apéndice de la spec del estudio).)
- Create: `tests/fakes.py::write_shy_bundle` (append, after `write_herd_bundle`), `tests/test_shy_data.py`
- Test: `tests/test_shy_data.py`, `tests/test_data_files.py` (if it enumerates `data/` files, add the two new ones)

**Interfaces:**
- Produces: `PARAMS_FILE = 'voto-oculto/params.json'`, `INCUMBENTS_FILE = 'voto-oculto/incumbents.json'`, `CATALOGUE_FILE = 'es-scopes.csv'`, `NATIONAL = 'es'`; `load_params(data: FileSystem) -> dict`; `load_incumbents(data: FileSystem) -> dict` (`{scope: {date: [names]}}`); `load_catalogue(data: FileSystem) -> list[tuple[str, str]]` (`(scode, name)` of the rows whose `scode` is `es` or starts with `es-`, in file order); `MissingInput(Exception)` with `.part` (same shape as `studies/herd/data.py`); `Election` dataclass (`scope: str`, `scope_name: str`, `event_date: str`, `general: bool`, `label: str`, `short: str`, `blocks: Optional[dict]`, `parties: list[str]`, `results: dict[str, float]`, `n_cycle: int`, `polls: pd.DataFrame` with the final polls only: `date, pollster_id, pollster, start_date, end_date, sample_size, mtype, weight` plus one float column per party); `final_polls(polls: pd.DataFrame, until: str, window_days: int) -> pd.DataFrame` (rows with `weight > 0` whose `end_date`, or `date` without it, is `0..window_days` days before `until`); `read_elections(reader: BundleReader, catalogue: list, params: dict) -> tuple[list[Election], dict]` (chronological, ties by catalogue order; `inputs` keyed by scope: `{'events': n, 'skipped': [{'date', 'reason'}]}` with `reason` ∈ `'parts'`, `'no final polls'`, or `{'skipped': 'no events'}` for a scope without an events folder or whose `list_events` raises); `is_current(reader, scope, date) -> bool` (no `results` part under `events/{scope}/{date}/`); `read_current(reader: BundleReader, params: dict) -> tuple[Optional[dict], dict]` (the `latest` run of `manifest.scopes.es`: `None` and `{'skipped': reason}` with `reason` ∈ `'no manifest'`, `'no run'`, `'election published'`; else `{'run_id', 'as_of', 'event_date', 'blocks': meta.bmaps.blocks or None, 'main': meta.bmaps.main or [], 'polls': final_polls(run polls, as_of, window_days)}` and `{'run_id', 'as_of'}`); labels through `studies.lean.data._label` and `_short` with `month=True` when two published elections of the scope share the year.
- `write_shy_bundle(fs) -> dict` in `tests/fakes.py` writes through `bundle.BundleWriter(fs)` (shapes of the event parts and of the run parts copied from `write_herd_bundle`; every poll `weight 1`, `sample_size 1000`, `start_date = end_date = date`, `mtype None` unless said): events `es/1989-10-29` (`meta` name «Generales», `blocks {Izquierda: ['PSOE', 'IU'], Derecha: ['AP', 'CDS']}`; `results` AP 25.8, PSOE 39.6, IU 9.1, CDS 7.9; `cycle` polls, parties `['AP', 'PSOE', 'IU', 'CDS']`: `1989-10-20` AP 24, PSOE 42, IU 8, CDS 9 · `1989-10-25` AP 23, PSOE 44, IU 10, CDS null · `1989-09-30` AP 30, PSOE 40, IU 9, CDS 8 · `1989-10-27` with `weight 0` AP 20, PSOE 50, IU 10, CDS 5), `es/2023-07-23` (blocks `{Izquierda: ['PSOE', 'SUMAR'], Derecha: ['PP', 'VOX']}`; results PP 33.1, PSOE 31.7, VOX 12.4, SUMAR 12.3; polls `2023-07-10` PP 35, PSOE 27, VOX 14, SUMAR 13 · `2023-07-16` 34, 28, 13, 12 · `2023-07-19` 33, 29, 13, 13), `es/2016-06-26` (`meta` and `results` only, no `cycle`), `es-md/2015-05-24` (name «Madrid»; blocks `{Izquierda: ['PSOE', 'UP', 'IU'], Derecha: ['PP', 'Cs']}`; results PP 33.1, PSOE 25.4, UP 18.6, Cs 12.1, IU 4.2; polls `2015-05-15` PP 34, PSOE 24, UP 20, Cs 13, IU 4 · `2015-05-20` PP 35, PSOE 25, UP 18, Cs 12, IU null), `es-md/2019-05-26` (the three parts; its only poll ends `2019-04-30`), `es-md/2021-05-04` (blocks `{Izquierda: ['PSOE', 'MM', 'UP'], Derecha: ['PP', 'VOX']}`; results PP 44.7, MM 16.9, PSOE 16.8, VOX 9.1, UP 7.2; polls `2021-04-25` PP 43, PSOE 20, MM 14, VOX 10, UP 6 · `2021-05-02` PP 42, PSOE 20, MM 16, VOX 10, UP 6); the manifest with `scopes.es.latest = SHY_RUN = '20261105-120000'` and that run's `meta` (`event_date '2026-11-29'`, `as_of '2026-11-05'`, `bmaps {'blocks': {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP', 'VOX']}, 'main': ['PP', 'PSOE', 'VOX', 'FA']}`) and `polls` (parties `['PP', 'PSOE', 'VOX', 'FA', 'UP']`: `2026-10-20` PP 31, PSOE 27, VOX 19, FA 7, UP 3 · `2026-10-25` 32, 26, 20, 6, 3 · `2026-11-03` 30, 28, 18, 8, UP null). It also writes a data folder `shy-data/` under `fs` with `voto-oculto/params.json` (the Global Constraints values but `lineages {'PSOE': ['PSOE', 'PSC'], 'PP': ['PP', 'AP'], 'VOX': ['VOX'], 'UP': ['SUMAR', 'UP', 'IU']}`), `voto-oculto/incumbents.json` (`{'es': {'1989-10-29': ['PSOE'], '2016-06-26': ['PP'], '2023-07-23': ['PSOE'], '2026-11-29': ['PSOE']}, 'es-md': {'2015-05-24': ['PP'], '2019-05-26': ['PP'], '2021-05-04': ['PP']}}`) and `es-scopes.csv` (`scode,name` rows `es,España`, `es-md,Madrid`, `es-xx,Ámbito X`). Returns `{'dates': {'es': ['1989-10-29', '2016-06-26', '2023-07-23'], 'es-md': ['2015-05-24', '2019-05-26', '2021-05-04']}, 'run_id': SHY_RUN, 'data': FileSystem(fs.get_path('shy-data'))}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_params_and_catalogue_read_the_repository_files():
    data_fs = FileSystem(os.path.join(ROOT, 'data'))
    params = data.load_params(data_fs)
    assert (params['window_days'], params['recent_from'], params['size_threshold'], params['big_threshold']) == (14, '2018-09-01', 2, -3)
    assert params['lineages']['UP'] == ['SUMAR', 'FA', 'UP', 'IU', 'PCE'] and params['pp_early'] == ['1982-01-01', '1996-01-01']
    codes = [code for code, _ in data.load_catalogue(data_fs)]
    assert 'es' in codes and 'es-md' in codes and len(codes) >= 18 and all(code == 'es' or code.startswith('es-') for code in codes)
    assert dict(data.load_catalogue(data_fs))['es-md'] == 'Madrid'
    incumbents = data.load_incumbents(data_fs)
    assert incumbents['es']['2026-11-29'] == ['PSOE'] and incumbents['es-cb']['2023-05-28'] == ['PRC'] and len(incumbents['es']) == 17

def test_final_polls_window_and_weight():
    polls = pd.DataFrame({'date': ['2023-07-10', '2023-07-09', '2023-07-23', '2023-07-20', '2023-07-01'],
                          'end_date': ['2023-07-10', None, '2023-07-23', '2023-07-20', '2023-07-09'], 'weight': [1, 1, 1, 0, 1]})
    out = data.final_polls(polls, '2023-07-23', 14)
    assert out['date'].tolist() == ['2023-07-10', '2023-07-09', '2023-07-23', '2023-07-01']   # end 13, date 14, end 0, end 14 days before; weight 0 out

def test_read_elections_reads_the_published_elections_of_the_catalogue(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    elections, inputs = data.read_elections(bundle.BundleReader(fs), data.load_catalogue(made['data']), data.load_params(made['data']))
    assert [(e.scope, e.event_date) for e in elections] == [('es', '1989-10-29'), ('es-md', '2015-05-24'), ('es-md', '2021-05-04'), ('es', '2023-07-23')]
    assert [e.label for e in elections] == ['Generales 1989', 'Madrid 2015', 'Madrid 2021', 'Generales 2023'] and elections[1].short == 'MD 15'
    assert elections[0].polls.shape[0] == 2 and elections[0].n_cycle == 4 and elections[0].results['AP'] == 25.8 and elections[0].general
    assert elections[1].blocks == {'Izquierda': ['PSOE', 'UP', 'IU'], 'Derecha': ['PP', 'Cs']} and elections[1].scope_name == 'Madrid'
    assert inputs['es'] == {'events': 2, 'skipped': [{'date': '2016-06-26', 'reason': 'parts'}]}
    assert inputs['es-md'] == {'events': 2, 'skipped': [{'date': '2019-05-26', 'reason': 'no final polls'}]}

def test_read_elections_skips_a_scope_without_events(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    _, inputs = data.read_elections(bundle.BundleReader(fs), data.load_catalogue(made['data']), data.load_params(made['data']))
    assert inputs['es-xx'] == {'skipped': 'no events'}

def test_read_elections_adds_the_month_when_the_year_repeats(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    copy_event(fs, 'es', '2023-07-23', '2023-11-19')   # helper of the test: the same three parts under another date
    elections, _ = data.read_elections(bundle.BundleReader(fs), data.load_catalogue(made['data']), data.load_params(made['data']))
    assert [e.label for e in elections if e.event_date.startswith('2023')] == ['Generales jul. 2023', 'Generales nov. 2023']
    assert [e.short for e in elections if e.event_date.startswith('2023')] == ['GEN 23J', 'GEN 23N']

def test_read_current_reads_the_latest_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    current, inputs = data.read_current(bundle.BundleReader(fs), data.load_params(made['data']))
    assert (current['run_id'], current['as_of'], current['event_date']) == ('20261105-120000', '2026-11-05', '2026-11-29')
    assert current['polls']['date'].tolist() == ['2026-10-25', '2026-11-03'] and current['main'] == ['PP', 'PSOE', 'VOX', 'FA']
    assert current['blocks'] == {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP', 'VOX']} and inputs == {'run_id': '20261105-120000', 'as_of': '2026-11-05'}

def test_read_current_without_blocks(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    rewrite_run_meta(fs, bmaps={'main': ['PP', 'PSOE']})   # helper of the test: the run meta without `blocks`
    current, _ = data.read_current(bundle.BundleReader(fs), data.load_params(made['data']))
    assert current['blocks'] is None and current['main'] == ['PP', 'PSOE']

def test_read_current_is_none_without_a_run_or_once_the_election_is_published(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs); params = data.load_params(made['data'])
    writer = bundle.BundleWriter(fs)
    writer.write_json(bundle.path_event('es', '2026-11-29', 'results'), 'event-results', {'n_seats': 350, 'rows': []}, 'es')
    assert data.read_current(bundle.BundleReader(fs), params) == (None, {'skipped': 'election published'})
    fs.remove(bundle.path_manifest())
    assert data.read_current(bundle.BundleReader(fs), params) == (None, {'skipped': 'no manifest'})
```

(`copy_event` and `rewrite_run_meta` are small helpers of the test file, written with `BundleReader.read_json` and `BundleWriter.write_json`; `fs.remove` as `mtpy/core/io.py` names it.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_shy_data.py -q`
Expected: FAIL with `ModuleNotFoundError: mtpy.lib.studies.shy`.

- [ ] **Step 3: Write the data files, their README rows, `write_shy_bundle` and `data.py`**

`data/voto-oculto/incumbents.json` from the spec appendix (95 rows: the 92 elections, Cantabria 2011 and 2019, and `es/2026-11-29: PSOE`; the alternatives in brackets as further names; `es-ct 2017` and `2021` `['JxCat', 'PDeCAT', 'JxSí']` / `['JxCat', 'Junts']`, `es-nc 2023` `['PSOE', 'PSN', 'PSN-PSOE']`, as the recon table of the spec). Signatures of the Interfaces block; `polls` columns cast: `sample_size` float with `NaN` for null, shares float. `read_elections` reads `manifest`-independent paths: `reader.list_events(scope)` inside a `try` and `reader.exists(bundle.path_event(scope, date, part))` for the three parts.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_shy_data.py tests/test_data_files.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/shy/__init__.py mtpy/lib/studies/shy/data.py data/voto-oculto/params.json data/voto-oculto/incumbents.json data/README.md tests/fakes.py tests/test_shy_data.py
git commit -m "Estudio voto-oculto: parámetros, tabla del Gobierno, catálogo y lectura de elecciones y run actual"
```

---

### Task 3: The measure (`studies/shy/stats.py`)

**Files:**
- Create: `mtpy/lib/studies/shy/stats.py`, `tests/test_shy_stats.py`

**Interfaces:**
- Produces: `group_error(polls: pd.DataFrame, names: list[str], results: dict) -> tuple[Optional[float], Optional[str]]` (the sum rule of the Global Constraints over the given polls; the second value is the present members joined with `+` in `names` order, `None` when none is in `results`); `figures(values: list) -> Optional[dict]` (the figures block, rounded as the Global Constraints say; `None` for an empty list after dropping `None`); `extremes(rows: list[dict], key: str, k: int) -> dict` (`{'under': [{'label', 'error'}] × k` the most negative ascending, `'over': […] × k` the most positive descending}`, only rows with `row[key]` not `None`, fewer when there are fewer); `size(rows: list[dict], parties: list[str], threshold: float) -> dict` (`{'n_pairs', 'mean', 'share_over'}` over every `row[party]` not `None`); `significant(p) -> Optional[float]` (three significant figures).

- [ ] **Step 1: Write the failing tests**

```python
RES = {'PSOE': 25.4, 'UP': 18.6, 'IU': 4.2, 'PP': 33.1}
POLLS = pd.DataFrame([{'UP': 20.0, 'IU': 4.0, 'PSOE': 24.0, 'PP': 34.0, 'VOX': None}, {'UP': 18.0, 'IU': None, 'PSOE': 25.0, 'PP': 35.0, 'VOX': None}])

def test_group_error_sums_the_members_each_poll_estimates():
    assert stats.group_error(POLLS, ['SUMAR', 'UP', 'IU'], RES) == (pytest.approx(0.3), 'UP+IU')   # (24 − 22.8 + 18 − 18.6) / 2
    assert stats.group_error(POLLS, ['PSOE', 'PSC'], RES) == (pytest.approx(-0.9), 'PSOE')
    assert stats.group_error(POLLS, ['PSOE', 'UP', 'IU'], RES) == (pytest.approx(-0.6), 'PSOE+UP+IU')   # the block rule is the same

def test_group_error_is_none_when_no_poll_estimates_the_party():
    assert stats.group_error(POLLS, ['VOX'], RES | {'VOX': 9.1}) == (None, 'VOX')
    assert stats.group_error(POLLS, ['VOX'], RES) == (None, None)
    assert stats.group_error(POLLS.iloc[0:0], ['PSOE'], RES) == (None, 'PSOE')

def test_figures_counts_signs_and_tests_the_mean():
    assert stats.figures([1.0, 2.0, 3.0]) == {'n': 3, 'under': 0, 'over': 3, 'mean': 2.0, 'median': 2.0, 'p_t': 0.0742, 'p_sign': 0.125, 'size': 2.0, 'size_over': 0.33}
    assert stats.figures([-1.0, 0.0, 1.0]) == {'n': 3, 'under': 1, 'over': 1, 'mean': 0.0, 'median': 0.0, 'p_t': 1.0, 'p_sign': 0.875, 'size': 0.67, 'size_over': 0.0}
    assert stats.figures([2.5, None, -1.0])['p_t'] is None and stats.figures([2.5, None, -1.0])['n'] == 2
    assert stats.figures([]) is None and stats.figures([None]) is None

def test_extremes_and_size():
    rows = [{'label': 'A', 'VOX': -6.01}, {'label': 'B', 'VOX': 3.39}, {'label': 'C', 'VOX': None}, {'label': 'D', 'VOX': -0.5}, {'label': 'E', 'VOX': 2.1}]
    assert stats.extremes(rows, 'VOX', 3) == {'under': [{'label': 'A', 'error': -6.01}, {'label': 'D', 'error': -0.5}], 'over': [{'label': 'B', 'error': 3.39}, {'label': 'E', 'error': 2.1}]}
    rows = [{'PSOE': 3.4, 'PP': -2.3, 'VOX': None, 'UP': -0.1}, {'PSOE': -3.7, 'PP': 0.9, 'VOX': 0.9333, 'UP': 0.3667}]
    assert stats.size(rows, ['PSOE', 'PP', 'VOX', 'UP'], 2) == {'n_pairs': 7, 'mean': 1.67, 'share_over': 0.43}
    assert stats.significant(0.0742195) == 0.0742 and stats.significant(None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_shy_stats.py -q`
Expected: FAIL with `ImportError` (no `stats`).

- [ ] **Step 3: Implement `stats.py`**

Signatures of the Interfaces block; `scipy.stats.ttest_1samp` and `binomtest` imported inside `figures`. `p_sign` with `n == 0` never happens (`None` returned before).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_shy_stats.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/shy/stats.py tests/test_shy_stats.py
git commit -m "Estudio voto-oculto: error de bloques y linajes con la regla de suma, cifras, extremos y tamaño"
```

---

### Task 4: The study (`studies/shy/study.py`)

**Files:**
- Create: `mtpy/lib/studies/shy/study.py`, `tests/test_shy_study.py`

**Interfaces:**
- Consumes: Task 2 readers and Task 3 statistics.
- Produces: `ShyStudy` dataclass with `params: dict`, `scopes: list[str]` (codes read, catalogue order), `election: Optional[dict]` (`{date, days_left, as_of, run_id}` of the current run), `summary: dict` (exactly the keys of spec D3 and Task 1's schema), `elections: list[dict]` (the rows of spec D3, chronological), `groups: list[dict]` (the long rows: `{table, key, label, n, under, over, mean, median, p_t, p_sign, size, size_over}`), `inputs: dict` (`{scope: ...}` of `read_elections` plus `run` and `incumbents`: the `(scope, date)` without a row or without a present name, as `[{'scope', 'date'}]`); `election_row(election: Election, params: dict, incumbents: dict) -> dict` (one `elections` row: block errors from `election.blocks`, lineage errors and `names` from `params.lineages`, `lean`, `incumbent`/`incumbent_error` from the first name of the table row present in `results` through `group_error` with that single name; `None` and an `inputs.incumbents` note without row or name; plus `results` (`{key: the sum of the results of the present members}`) and `polls` (`{key: results[key] + error}`: the mean of the final polls where a member a poll does not estimate counts its result, so that `polls − results == error` exactly) for the six keys `Izquierda, Derecha, PSOE, PP, VOX, UP`, `None` where the error is `None`); `verdicts(left_all: Optional[dict], vox: Optional[dict], parties: list[dict], labels: dict) -> dict` (`p1` «SÍ» iff `left_all` has `mean < 0` and `p_t` not `None` and `< 0.05`, else «NO»; `p2` the same over `vox`, «SIN DATOS» when `vox` is `None`; `p3` «DE NADIE» unless some row of `parties` with `key` in `('PSOE', 'PP', 'VOX', 'UP')` has `figures.mean < 0` and `figures.p_t < 0.05`: then «DE » + the upper-cased `labels.get(key, key)` of each, joined by « Y »); `compute_shy(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> ShyStudy` (raises `MissingInput('events')` without elections). Rules: `left`/`right`/`lean` have the four sub-blocks `all`, `general`, `regional`, `recent` (`recent` = `event_date >= params.recent_from`); `vox` = figures of the `VOX` column plus `extremes` (`params.extremes`); `parties` in the order `PSOE, PP, VOX, UP, Izquierda, Derecha, incumbent_general, incumbent_regional` with `label` from `params.labels` (`'Izquierda (bloque)'`, `'Derecha (bloque)'`, `'Gobierno, generales'`, `'Gobierno, autonómicas'` fixed), rows whose figures are `None` kept with `figures None`; `size` over `params.size_parties`; `psoe_general` = figures of `PSOE` in the generals plus `big` = labels with error `< params.big_threshold`, chronological; `pp` = `{early: {from, to, errors: [{label, error}] of the generals with pp_early[0] <= date < pp_early[1]}, late: figures of the generals with date >= pp_early[1], all: figures of PP}`; `podemos` = `{before: figures of UP with date < recent_from, general_recent, regional_recent}`; `current` = `{as_of, days_left (event_date − as_of in days), n_polls, from, to (min and max end of the window polls), values, names}` with `values` = the simple mean, per block, of the per-poll sums of the block's parties the poll estimates (keys only for the blocks of `current.blocks` with some poll) and, per party of `current.main`, the mean of its non-null shares (keys only when some poll has it), rounded to 1 decimal, or `None` without a run or without window polls (`n_polls 0` → `current` kept with empty `values`); `verdicts` through `verdicts(...)`; `groups` = every figures block of `summary` flattened (`left`, `right`, `lean` × four, `vox`, `parties` × eight, `podemos` × three, `pp` × `late`/`all`).

- [ ] **Step 1: Write the failing tests**

Figures computed with the rules over the Task 2 bundle (`plan/synthetic.py` of the author's scratchpad; they are the assertions):

```python
def test_compute_shy_on_the_synthetic_bundle(fresh_app, tmp_path):
    study = compute(tmp_path)   # write_shy_bundle + compute_shy(BundleReader(fs), made['data'])
    s = study.summary
    assert (s['n_elections'], s['n_general'], s['n_regional'], s['n_scopes']) == (4, 2, 2, 2) and study.scopes == ['es', 'es-md']
    assert (s['first_year'], s['first_year_regional'], s['last_year']) == (1989, 2015, 2023)
    assert s['left']['all'] == {'n': 4, 'under': 2, 'over': 2, 'mean': -0.13, 'median': -0.25, 'p_t': 0.928, 'p_sign': 0.688, 'size': 1.83, 'size_over': 0.5}
    assert s['left']['general'] == {'n': 2, 'under': 1, 'over': 1, 'mean': -0.02, 'median': -0.02, 'p_t': None, 'p_sign': 0.75, 'size': 3.32, 'size_over': 1.0}
    assert (s['left']['regional']['mean'], s['left']['recent']['mean']) == (-0.25, -1.62)
    assert (s['right']['all']['mean'], s['right']['regional']['mean'], s['lean']['all']['mean'], s['lean']['general']['mean']) == (0.15, 0.25, -0.28, -0.06)
    assert s['vox'] == {'n': 2, 'under': 0, 'over': 2, 'mean': 0.92, 'median': 0.92, 'p_t': None, 'p_sign': 0.25, 'size': 0.92, 'size_over': 0.0,
                        'extremes': {'under': [], 'over': [{'label': 'Generales 2023', 'error': 0.93}, {'label': 'Madrid 2021', 'error': 0.9}]}}
    parties = {row['key']: row for row in s['parties']}
    assert [row['key'] for row in s['parties']] == ['PSOE', 'PP', 'VOX', 'UP', 'Izquierda', 'Derecha', 'incumbent_general', 'incumbent_regional']
    assert parties['PSOE']['figures'] == {'n': 4, 'under': 2, 'over': 2, 'mean': 0.5, 'median': 1.15, 'p_t': 0.79, 'p_sign': 0.688, 'size': 2.8, 'size_over': 0.75}
    assert (parties['PP']['figures']['mean'], parties['PP']['label'], parties['UP']['figures']['mean'], parties['UP']['label']) == (-0.55, 'PP (AP, CD)', -0.16, 'IU / Podemos / Sumar')
    assert (parties['incumbent_general']['figures']['mean'], parties['incumbent_regional']['figures']['mean']) == (-0.15, -0.4)
    assert s['size'] == {'n_pairs': 14, 'mean': 1.56, 'share_over': 0.36}
    assert s['psoe_general']['big'] == ['Generales 2023'] and s['psoe_general']['mean'] == -0.15
    assert s['pp']['early'] == {'from': '1982-01-01', 'to': '1996-01-01', 'errors': [{'label': 'Generales 1989', 'error': -2.3}]} and s['pp']['late']['n'] == 1
    assert (s['podemos']['before']['mean'], s['podemos']['general_recent']['mean'], s['podemos']['regional_recent']['mean']) == (0.1, 0.37, -1.2)
    assert s['current'] == {'as_of': '2026-11-05', 'days_left': 24, 'n_polls': 2, 'from': '2026-10-25', 'to': '2026-11-03',
                            'values': {'Izquierda': 35.5, 'Derecha': 50.0, 'PP': 31.0, 'PSOE': 27.0, 'VOX': 19.0, 'FA': 7.0}, 'names': ['PP', 'PSOE', 'VOX', 'FA']}
    assert s['verdicts'] == {'p1': 'NO', 'p2': 'NO', 'p3': 'DE NADIE'}
    assert study.election == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05', 'run_id': '20261105-120000'}

def test_compute_shy_election_rows(fresh_app, tmp_path):
    rows = {(r['scope'], r['event_date']): r for r in compute(tmp_path).elections}
    r89 = rows[('es', '1989-10-29')]
    assert (r89['label'], r89['short'], r89['general'], r89['n_cycle'], r89['n_final']) == ('Generales 1989', 'GEN 89', True, 4, 2)
    assert (r89['Izquierda'], r89['Derecha'], r89['lean'], r89['PSOE'], r89['PP'], r89['VOX'], r89['UP']) == (3.3, -1.75, 5.05, 3.4, -2.3, None, -0.1)
    assert r89['names'] == {'PSOE': 'PSOE', 'PP': 'AP', 'VOX': None, 'UP': 'IU'} and (r89['incumbent'], r89['incumbent_error']) == ('PSOE', 3.4)
    r15 = rows[('es-md', '2015-05-24')]
    assert (r15['UP'], r15['names']['UP'], r15['Izquierda'], r15['Derecha'], r15['lean']) == (0.3, 'UP+IU', -0.6, 1.8, -2.4)
    assert (r15['scope_name'], r15['incumbent'], r15['incumbent_error']) == ('Madrid', 'PP', 1.4)
    assert (r89['results']['Izquierda'], r89['polls']['Izquierda'], r15['results']['UP'], r15['polls']['UP'], r89['polls']['VOX']) == (48.7, 52.0, 22.8, 23.1, None)
    assert rows[('es', '2023-07-23')]['PSOE'] == -3.7 and rows[('es-md', '2021-05-04')]['blocks'] == {'Izquierda': ['PSOE', 'MM', 'UP'], 'Derecha': ['PP', 'VOX']}
    assert [k for k in rows] == [('es', '1989-10-29'), ('es-md', '2015-05-24'), ('es-md', '2021-05-04'), ('es', '2023-07-23')]

def test_compute_shy_groups_and_inputs(fresh_app, tmp_path):
    study = compute(tmp_path)
    tables = {(g['table'], g['key']) for g in study.groups}
    assert {('left', 'all'), ('right', 'recent'), ('lean', 'general'), ('vox', 'all'), ('parties', 'UP'), ('podemos', 'before'), ('pp', 'late')} <= tables
    assert next(g for g in study.groups if (g['table'], g['key']) == ('parties', 'PSOE'))['mean'] == 0.5
    assert study.inputs['es']['skipped'] == [{'date': '2016-06-26', 'reason': 'parts'}] and study.inputs['es-xx'] == {'skipped': 'no events'}
    assert study.inputs['run'] == {'run_id': '20261105-120000', 'as_of': '2026-11-05'} and study.inputs['incumbents'] == []

def test_compute_shy_notes_an_election_without_an_incumbent_row(fresh_app, tmp_path):
    study = compute(tmp_path, incumbents={'es': {'2023-07-23': ['PSOE']}})   # the test rewrites shy-data/voto-oculto/incumbents.json
    assert study.inputs['incumbents'] == [{'scope': 'es', 'date': '1989-10-29'}, {'scope': 'es-md', 'date': '2015-05-24'}, {'scope': 'es-md', 'date': '2021-05-04'}]
    assert study.summary['parties'][6]['figures']['n'] == 1 and study.summary['parties'][7]['figures'] is None

def test_compute_shy_without_a_run_and_without_elections(fresh_app, tmp_path):
    study = compute(tmp_path, without_manifest=True)
    assert study.summary['current'] is None and study.election is None and study.inputs['run'] == {'skipped': 'no manifest'}
    with pytest.raises(MissingInput) as info:
        compute(tmp_path, without_events=True)
    assert info.value.part == 'events'

def test_compute_shy_moves_the_current_election_to_the_past_once_published(fresh_app, tmp_path):
    # publish_current writes events/es/2026-11-29/{meta, cycle, results}: meta with the run's blocks and parties; cycle with two
    # polls, 2026-11-20 PP 32, PSOE 26, VOX 20, FA 6, UP 3 and 2026-11-25 PP 30, PSOE 28, VOX 18, FA 8, UP null; results
    # PP 31.5, PSOE 28.4, VOX 17.2, FA 7.0, UP 2.8
    study = compute(tmp_path, publish_current=True)
    assert study.summary['current'] is None and study.election is None
    last = study.elections[-1]
    assert (last['event_date'], last['label'], last['n_final'], last['incumbent'], last['incumbent_error']) == ('2026-11-29', 'Generales 2026', 2, 'PSOE', -1.4)
    assert (last['PP'], last['UP'], last['names']['UP'], last['Izquierda']) == (-0.5, 0.1, 'FA+UP', -1.3)   # FA+UP: (9 − 9.8, 8 − 7.0) / 2; Izq: (35 − 38.2, 36 − 35.4) / 2
    assert (last['results']['Izquierda'], last['polls']['Izquierda']) == (38.2, 36.9)
    assert study.summary['n_elections'] == 5 and study.summary['last_year'] == 2026 and study.summary['verdicts']['p1'] == 'NO'

def test_verdicts_in_the_three_cases():
    no = {'mean': 0.47, 'p_t': 0.128}; yes = {'mean': -1.2, 'p_t': 0.01}; few = {'mean': -1.2, 'p_t': None}
    labels = {'UP': 'IU / Podemos / Sumar', 'PP': 'PP (AP, CD)', 'VOX': 'Vox'}
    assert study_mod.verdicts(no, no, [{'key': 'PSOE', 'figures': no}], labels) == {'p1': 'NO', 'p2': 'NO', 'p3': 'DE NADIE'}
    assert study_mod.verdicts(yes, few, [{'key': 'PSOE', 'figures': yes}, {'key': 'UP', 'figures': yes}, {'key': 'Izquierda', 'figures': yes}], labels) == \
        {'p1': 'SÍ', 'p2': 'NO', 'p3': 'DE PSOE Y DE IU / PODEMOS / SUMAR'}
    assert study_mod.verdicts(no, None, [{'key': 'VOX', 'figures': None}], labels)['p2'] == 'SIN DATOS'
```

(`compute(tmp_path, **changes)` is a helper of the test file: `write_shy_bundle`, the change, then `compute_shy(bundle.BundleReader(fs), made['data'])`; `publish_current` writes `events/es/2026-11-29/{meta,cycle,results}` with the run's polls and parties.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_shy_study.py -q`
Expected: FAIL (`study` module missing).

- [ ] **Step 3: Implement `study.py`**

`compute_shy` loads params (`load_params` updated with the overrides), the catalogue, the incumbents, the elections and the current run; builds the rows with `election_row`; then the summary, the groups and the verdicts of the Interfaces block. `days_left` = `event_date − as_of` in days.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_shy_study.py tests/test_shy_stats.py tests/test_shy_data.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/shy/study.py tests/test_shy_study.py
git commit -m "Estudio voto-oculto: el estudio entero desde las partes publicadas (filas, resumen, grupos, veredictos)"
```

---

### Task 5: Exporters and the publish registry (`publish/shy.py`)

**Files:**
- Create: `mtpy/lib/publish/shy.py`, `tests/test_publish_shy.py`
- Modify: `mtpy/lib/publish/studies.py:21-35` (`STUDY_TITLES`), `:570-595` (late imports, `STUDY_EXPORTERS`, `STUDY_META`, `STUDY_RUNNERS`), `mtpy/lib/publish/__init__.py:61-63` (re-exports)
- Modify: `tests/test_publish_studies.py:186` (module list), `tests/test_publish_unit.py:1551` (package module names), `tests/test_publish_herd.py` or wherever `test_study_runners_know_which_studies_are_live` lives (the dict gains `'voto-oculto': True`), `tests/test_pipelines_publish.py` and `tests/test_jobs_update.py` (the tests that spell the live studies or their line)
- Modify: `deploy/README.md:1375-1392` («### Estudios»: a paragraph «**Voto oculto.**» after the rebaño one and the command), `docs/web/contrato.md:457` («Hoy hay cuatro: … y `voto-oculto`»), after `:543` (section `### voto-oculto` with the four parts and their required keys of Task 1), `:823` (routes row: `voto-oculto` with its parts and CSV)

**Interfaces:**
- Produces: `STUDY_TITLES['voto-oculto'] = ('¿Voto oculto de quién?', <subtitle template>)` with placeholders `{n}`, `{k_under}`, `{k_over}`, `{vox_under}`, `{vox_n}`, `{size}`, `{tail}`: «Se dice que la gente de izquierdas contesta menos a las encuestas y que los votantes de Vox esconden su voto, y que por eso los sondeos infravaloran a unos y a otros. En las {n} elecciones con sondeos de nuestra base, los sondeos finales han dado a la izquierda menos voto del que sacó en {k_under} y más en {k_over}; a Vox, menos en {vox_under} de {vox_n}. Los sondeos se equivocan con cada partido grande en {size} puntos de media{tail}.» where `tail` = «, y el sentido del error no se puede adivinar» with `verdicts.p3 == 'DE NADIE'` and `p1 == 'NO'`, else «, y el sentido del error se inclina hacia {p3 without «DE »}» (or «hacia la izquierda» with `p1 == 'SÍ'`); the Vox clause only with `vox` not `None`; `size` with one decimal and a comma (`fmt`: `2,0`). `compute_for_publish(data, writer) -> ShyStudy`; `export_shy_summary(study) -> (data, {})`; `export_shy_elections(study) -> (data, {'elections': frame})` (frame flat: `blocks` as two columns `block_left`/`block_right` with the parties joined by `+`, `names` as `name_psoe, name_pp, name_vox, name_up`, `polls` and `results` as `polls_{key}`/`results_{key}` in lower case); `export_shy_groups(study) -> (data, {'groups': frame})`; `export_shy_meta(study, study_id='voto-oculto', registration=None, checked_at=None, prov=None, seconds=None) -> (data, {})` with `sources` = `SHY_SOURCES` (the Tezanos quote in The Objective 2022-03-16, El Independiente 2019-02-28, the REIS article `reis.cis.es/index.php/reis/article/view/2549`, Pavía y Larraz REIS 798; `{label, url}`); `STUDY_RUNNERS['voto-oculto'] = {'compute': compute_shy_for_publish, 'registrable': False, 'live': True}`; `mtpy/lib/publish/__init__.py` re-exports `compute_shy_for_publish`, `export_shy_elections`, `export_shy_groups`, `export_shy_meta`, `export_shy_summary`. Any existing test that spells the line of the `studies` action («studies: sesgo-cis published in 2 s, rebano published in 2 s», in `tests/test_pipelines_publish.py` and `tests/test_jobs_update.py`) is rewritten over `[s for s in bundle.STUDIES if publish.STUDY_RUNNERS[s]['live']]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_shy_parts_validate(study):   # fixture: compute_shy on write_shy_bundle with made['data'], as test_publish_herd does
    for part in bundle.STUDY_PARTS['voto-oculto'][1:]:
        data, frames = publish.STUDY_EXPORTERS['voto-oculto'][part](study)
        valid(bundle.STUDY_SCHEMAS['voto-oculto'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['voto-oculto'].get(part, ()))
    meta, frames = publish.export_shy_meta(study, prov=fake_prov())
    valid('voto-oculto-meta', meta) and frames == {}
    assert meta['subtitle'] == ('Se dice que la gente de izquierdas contesta menos a las encuestas y que los votantes de Vox esconden su voto, '
                                'y que por eso los sondeos infravaloran a unos y a otros. En las 4 elecciones con sondeos de nuestra base, los '
                                'sondeos finales han dado a la izquierda menos voto del que sacó en 2 y más en 2; a Vox, menos en 0 de 2. Los '
                                'sondeos se equivocan con cada partido grande en 1,6 puntos de media, y el sentido del error no se puede adivinar.')
    assert meta['election'] == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05', 'run_id': '20261105-120000'}
    assert meta['scopes'] == ['es', 'es-md'] and meta['params']['window_days'] == 14 and meta['inputs']['es-xx'] == {'skipped': 'no events'}
    assert [s['url'] for s in meta['sources']][0].startswith('https://theobjective.com')

def test_export_shy_subtitle_follows_the_verdicts(study):
    tilted = dataclasses.replace(study, summary=study.summary | {'verdicts': {'p1': 'NO', 'p2': 'NO', 'p3': 'DE PSOE'}})
    assert publish.export_shy_meta(tilted, prov=fake_prov())[0]['subtitle'].endswith('y el sentido del error se inclina hacia PSOE.')
    left = dataclasses.replace(study, summary=study.summary | {'verdicts': {'p1': 'SÍ', 'p2': 'NO', 'p3': 'DE NADIE'}, 'vox': None})
    subtitle = publish.export_shy_meta(left, prov=fake_prov())[0]['subtitle']
    assert 'a Vox' not in subtitle and subtitle.endswith('se inclina hacia la izquierda.')

def test_export_shy_frames_follow_the_study(study):
    data, frames = publish.STUDY_EXPORTERS['voto-oculto']['elections'](study)
    assert list(frames['elections']['PSOE']) == [r['PSOE'] for r in data['rows']] and frames['elections']['block_left'].iloc[0] == 'PSOE+IU'
    assert list(frames['elections']['name_up']) == ['IU', 'UP+IU', 'UP', 'SUMAR']
    data, frames = publish.STUDY_EXPORTERS['voto-oculto']['groups'](study)
    assert len(frames['groups']) == len(data['rows']) and list(frames['groups'].columns)[:4] == ['table', 'key', 'label', 'n']

def test_publish_study_voto_oculto_writes_the_parts_and_the_manifest(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    out = publish.publish_study('voto-oculto', make_writer(fs), data=made['data'], prov=fake_prov)
    assert out['study'] == 'voto-oculto' and out['registered_at'] is None and out['checked'] is False
    reader = bundle.BundleReader(fs)
    assert reader.read_json(bundle.path_study('voto-oculto', 'summary'))['data']['verdicts']['p3'] == 'DE NADIE'
    assert reader.read_json(bundle.path_manifest())['data']['studies']['voto-oculto'] == {'published_at': ANY, 'registered_at': None, 'checked': False}
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('voto-oculto', make_writer(fs), data=made['data'], register=True, prov=fake_prov)

def test_voto_oculto_is_live_and_the_other_studies_are_untouched():
    assert publish.STUDY_RUNNERS['voto-oculto'] == {'compute': publish.compute_shy_for_publish, 'registrable': False, 'live': True}
    assert publish.STUDY_RUNNERS['voto-cera']['live'] is False and publish.STUDY_RUNNERS['rebano']['live'] is True
```

In `tests/test_publish_studies.py:186` add `'mtpy.lib.studies.shy'` to the modules that must not load on `import mtpy.lib.publish`; in `tests/test_publish_unit.py:1551` add `'shy'` to the expected module names; update the two existing tests named in Files.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_shy.py tests/test_publish_studies.py tests/test_publish_unit.py tests/test_pipelines_publish.py -q -k "shy or voto_oculto or runners or study_package or names or live_studies"`
Expected: FAIL (`KeyError: 'voto-oculto'`, missing module).

- [ ] **Step 3: Implement `publish/shy.py`, register it, write the docs**

Pattern of `publish/herd.py` (`_rounded`, `_clean`, the subtitle rendered from `study.summary`); the late import block of `studies.py` gains `from .shy import (compute_for_publish as compute_shy_for_publish, export_shy_elections, export_shy_groups, export_shy_meta, export_shy_summary)  # noqa: E402`. `deploy/README.md`: «**Voto oculto.** `{"what":["study"],"study":"voto-oculto"}` calcula «¿Voto oculto de quién?» (`mtpy/lib/studies/shy/`) desde los sondeos finales (`cycle`) y los resultados (`results`) de todas las elecciones publicadas de los ámbitos del catálogo y los sondeos del run `latest` de `es` (`polls`), con los linajes y la tabla del Gobierno de `data/voto-oculto/`; sin base de datos ni registro; es un estudio vivo (`studies` lo republica en cada `update`).» plus the command line `S3_BUCKET= python job.py publish '{"what":["study"],"study":"voto-oculto"}'`. `docs/web/contrato.md`: the `### voto-oculto` table and the routes row.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_publish_shy.py tests/test_publish_studies.py tests/test_publish_herd.py tests/test_publish_lean.py tests/test_publish_unit.py tests/test_pipelines_publish.py tests/test_jobs_publish.py tests/test_jobs_update.py -q`
Expected: PASS (the `studies` action now publishes three live studies in `bundle.STUDIES` order).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish/shy.py mtpy/lib/publish/studies.py mtpy/lib/publish/__init__.py tests/test_publish_shy.py tests/test_publish_studies.py tests/test_publish_unit.py tests/test_publish_herd.py tests/test_pipelines_publish.py deploy/README.md docs/web/contrato.md
git commit -m "Estudio voto-oculto: exportadores, meta con la entradilla viva, estudio vivo en el registro y documentación"
```

---

### Task 6: The page `/analisis/voto-oculto` (view models, context, template)

**Files:**
- Create: `mtpy/lib/pages/shy.py`, `web/templates/analisis_voto_oculto.html`
- Modify: `mtpy/lib/pages/analisis.py:19` (import `shy`), `:23-25` (`STUDIES`), `CONTEXTS` and `shy_context` next to `herd_context` (`:658`), `mtpy/lib/pages/__init__.py:16-17, 41` (exports), `web/dist/css/site.css` (chart heights: `#left-chart` 340 px and 280 on phones, `#vox-chart` 260 px, `#parties-chart` 340 px, next to `.qcard .chart` `:886`)
- Modify: `tests/test_pages.py:1835-1847` (package layout list gains `'shy'`), new tests next to `test_herd_page_structure_and_wording` (`:5542`)
- Modify: `mtpy/lib/pages/metodo.py:821` (flag `shy_published` next to `lean_published`), `web/templates/metodo_promedio.html:468` (one sentence with the link, only with the flag)

**Interfaces:**
- Consumes: the four parts through `study_part('voto-oculto', part)`; `pages.filters` (`fmt_num`, `fmt_int`, `delta_text`, the `num`/`int`/`delta` filters); `_csv_links('voto-oculto')`, `_published_at('voto-oculto')`; `pages.herd.names`-like list prose (copy the macro).
- Produces in `pages/shy.py`: `SHY = 'voto-oculto'`; `QUESTIONS = ('¿Subestiman las encuestas a la izquierda?', '¿Hay voto oculto de Vox?', '¿De quién es el voto oculto, entonces?')`; `TONES = {'NO': 'q3', 'SÍ': 'q1', 'DE NADIE': 'qb', 'SIN DATOS': 'q2'}` (any «DE …» other than «DE NADIE» → `'q1'`); `verdict_rows(summary: Optional[dict]) -> list` of three `{'label', 'tone', 'question'}` from `summary.verdicts` (`'–'`/`'q2'` without them); `flags(summary: Optional[dict], meta: Optional[dict]) -> dict` with `has_run` (`summary.current` not `None`), `held` (`meta.election.date` ≤ today, Madrid), `election_day`/`election_year`/`days_left` from `meta.election`, `p1_yes`, `p2_yes`, `p3_someone`, `over_more` (`left.all.over > under`), `psoe_streak` (`psoe_general.under > over` and `big` non-empty), `regional_both_over` (`left.regional.over > under` and `left.regional.mean > 0` and `right.regional.mean > 0`), `vox_chance` (`vox.p_sign >= 0.05`), `pp_late_against` (`pp.late.over > under`), `podemos_all` (`regional_recent.over == regional_recent.n`), `incumbent_vanishes` (`incumbent_regional.p_sign >= 0.05`), `far_from_zero` (number of `parties` rows with `|mean| >= 1`), `coin` («19 ó 20» for an odd `vox.n`, «20» for an even one); `kpis(summary) -> list` of three `m.kpis` rows (`label`, `value`, `unit`, `support`): «Elecciones en que los sondeos subestimaron a la izquierda» `{under} de {n}` with the support of the mockup («y la sobreestimaron en {over} · sesgo medio {mean} puntos: cuando se equivocan, más veces le dan de más que de menos» / «… de menos que de más»), «Elecciones en que subestimaron a Vox» `{under} de {n}` («sesgo medio {mean} puntos · con una moneda saldrían {coin}: la probabilidad de {under} o más por azar es del {p_sign %}»), «Error de los sondeos finales con un partido grande» `{size.mean}` unit « puntos» («en cualquier sentido · PSOE, PP, Vox e IU/Podemos/Sumar en {n_pairs} pares elección-partido · en el {share %} de los casos, más de 2 puntos»); `left_rows(summary) -> list` (the five rows of the P1 table: `{'label', 'n', 'under', 'over', 'mean', 'p_t', 'bold'}` for `left.general` «Generales {first}-{last}», `left.regional` «Autonómicas {first_regional}-{last}», `left.recent` «Desde sept. 2018», `left.all` «Todas» (bold), `lean.all` «Izquierda − Derecha, todas»); `vox_tests(summary) -> list` of the three P2 boxes (`{'key', 'label', 'value', 'text'}`: «{under} de {n}» with the coin sentence; «{mean} puntos» with «Mediana {median}: un error que no se distingue de cero» with `vox_chance` and «Mediana {median}» without it; «{worst under} y {worst over}» with the two extremes of each side, «–» for an empty side); `party_rows(summary) -> list` (`{'key', 'label', 'n', 'under', 'over', 'mean', 'p_t', 'size'}` of `parties`, rows with `figures None` shown with dashes); `big_years(summary) -> str` (the years of `psoe_general.big` in prose: «1977, 1986, 1993, 1996, 2004 y 2023»); `pct(p) -> str` (a probability as a percentage: «26 %» from 10 % up, one decimal below («4,2 %»), two significant figures below 1 % («0,04 %»), «menos del 0,01 %» below that); `after_rows(summary, elections, meta) -> dict` (`{'columns': [...], 'label': str or None, 'now': dict or None, 'polls': dict or None, 'results': dict or None, 'error': dict or None, 'period': str or None}`: with `summary.current`, `columns` = the blocks `Izquierda`, `Derecha` present in `values` followed by the parties of `current.names` present in `values`, `now` = `values`, `period` = «del 25 de octubre al 3 de noviembre (2)»; without `current` and without `meta.election`, the newest row of `elections` with `scope == 'es'` **only when its year equals `summary.last_year`** (the election just published): `columns` = the six keys with a non-null error, `label` = the row's label, `polls`/`results`/`error` = the row's dicts; otherwise everything `None` and the template writes one neutral sentence); `island(parts) -> dict` (`{'left_bars': [{short, label, value, general}] chronological from elections with Izquierda not None, 'left_mean': left.all.mean, 'vox_bars': [...] from elections with VOX not None, 'vox_mean': vox.mean, 'party_strip': {'rows': [{party: label, label: election label, value}], 'means': {label: mean}, 'order': ['Derecha', 'Izquierda', 'IU / Podemos / Sumar', 'Vox', 'PP', 'PSOE']}}`). In `pages/analisis.py`: `STUDIES['voto-oculto'] = {'template': 'analisis_voto_oculto.html'}`, `CONTEXTS['voto-oculto'] = shy_context`, `shy_context(state) -> dict` with `study_meta, summary, elections, verdicts, kpis, left_rows, vox_tests, party_rows, after, csv_links, sources, published_at, initial` and the flags. In `pages/metodo.py`: `context['shy_published'] = bundle_part(site().study, 'voto-oculto', 'meta') is not None` in the `/metodo/promedio` context; the sentence in `metodo_promedio.html` after the `lean_published` one: `{% if shy_published %} De quién es el voto oculto, elección a elección, está en <a href="/analisis/voto-oculto">¿Voto oculto de quién?</a>.{% endif %}`.
- Template `analisis_voto_oculto.html`: the structure of spec D6 with the mockup v4 texts (ids `study-head`, `resumen`, `p1`, `p2`, `p3`, `despues`, `datos`; charts `#left-chart`, `#vox-chart`, `#parties-chart` with `class="chart"`, `role="img"` and a live `aria-label`; the `.tests` boxes; the P3 table with `scroll-hint`; the Tezanos `.quote` with the pill «Refutada» (`pill bad`) or «Confirmada» (`pill ok`) by `p2_yes`; the «Después del 29-N» table with `after.columns`; `m.how` in P1, P2 and P3; `shot(slug)` macro as in `analisis_rebano.html` with the file `{site_title|lower}-voto-oculto-{slug}.png`; `id="initial-data"` island; page script `/dist/js/pages/analisis-voto-oculto.js`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_pages.py` (helpers `publish_study(fs, meta=None, parts=None, study='voto-oculto')`, `call`, `html`, `visible_text`, `fixture_data`):

```python
def test_shy_page_structure_and_wording(api, fresh_app):
    publish_study(fresh_app.fs, study='voto-oculto')
    status, _, body = call(api, '/analisis/voto-oculto'); page = html(body)
    assert status == 200 and '<h1 class="study-title">¿Voto oculto de quién?</h1>' in page
    assert '/dist/js/pages/analisis-voto-oculto.js' in page and 'id="initial-data"' in page
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page) == ['NO', 'NO', 'DE NADIE']
    assert page.count('<div class="kpi">') == 3 and '40 de 92' in page and '22 de 39' in page and '2,0<small> puntos</small>' in page
    assert 'Evidencia histórica · 92 elecciones con sondeos · 16 generales desde 1977 y 76 autonómicas desde 2009' in page
    for chart in ('left-chart', 'vox-chart', 'parties-chart'):
        assert re.search(r'<div class="chart[^"]*" id="%s" role="img" aria-label="[^"]+"></div>' % chart, page), chart
    assert page.count('class="pill brand" href="/api/v1/studies/voto-oculto/') == 2
    assert 'Refutada' in page and 'Tezanos' in page and 'con una moneda saldrían 19 ó 20' in page and 'del 26 %' in page
    assert 'más veces le dan de más que de menos' in page and '(1977, 1986, 1993, 1996, 2004 y 2023)' in page
    assert 'la cuenta que vale es Izquierda − Derecha, repartida 34 a 42' in page and 'en las 76 autonómicas la racha desaparece' in page
    assert 'Desde finales de 2018, en las 32 autonómicas' in page and 'de más en 30' in page and 'Andalucía 2018' in page
    assert 'Sondeos finales, hoy' in page and '35,4' in page and 'sondeos del 22 de octubre al 5 de noviembre (12)' in page
    assert 'Frente Amplio' not in page and 'FA' in visible_text(page)   # the column is the run's party name
    assert_public_wording(page, '/analisis/voto-oculto')
    assert 'empresa' not in visible_text(page).lower() and 'max-width' not in page

def test_shy_page_conditions_follow_the_figures(api, fresh_app):
    summary = fixture_data('voto-oculto-summary')
    summary = summary | {'verdicts': {'p1': 'SÍ', 'p2': 'SÍ', 'p3': 'DE PSOE'},
                         'left': summary['left'] | {'all': summary['left']['all'] | {'under': 60, 'over': 32, 'mean': -1.1, 'p_t': 0.004}},
                         'vox': summary['vox'] | {'under': 30, 'over': 9, 'mean': -1.4, 'p_t': 0.001, 'p_sign': 0.0004}}
    publish_study(fresh_app.fs, study='voto-oculto', parts={'summary': summary})
    page = html(call(api, '/analisis/voto-oculto')[2])
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', page) == ['SÍ', 'SÍ', 'DE PSOE']
    assert 'Confirmada' in page and 'Refutada' not in page and 'más veces de menos que de más' in page
    assert 'la creencia se sostiene' in page and 'lo que se espera de unos sondeos sin sesgo' not in page and 'del 0,04 %' in page

def test_shy_page_after_card_shows_only_the_values_present(api, fresh_app):
    summary = fixture_data('voto-oculto-summary')
    summary['current'] = summary['current'] | {'values': {'PSOE': 26.4, 'PP': 31.3}, 'names': ['PP', 'PSOE', 'VOX']}
    publish_study(fresh_app.fs, study='voto-oculto', parts={'summary': summary})
    page = html(call(api, '/analisis/voto-oculto')[2])
    table = page[page.index('id="despues"'):]
    assert '<th>PP</th><th>PSOE</th>' in table and 'Izquierda' not in table[:table.index('</thead>')] and 'VOX' not in table[:table.index('</thead>')]

def test_shy_page_once_the_election_is_published(api, fresh_app):
    summary = fixture_data('voto-oculto-summary') | {'current': None, 'n_elections': 93, 'last_year': 2026}
    elections = fixture_data('voto-oculto-elections')
    elections['rows'].append(elections['rows'][-1] | {'scope': 'es', 'scope_name': 'España', 'event_date': '2026-11-29', 'label': 'Generales 2026',
                                                      'short': 'GEN 26', 'general': True, 'Izquierda': -1.2, 'Derecha': 0.8, 'incumbent': 'PSOE',
                                                      'polls': {'Izquierda': 35.4, 'Derecha': 52.6}, 'results': {'Izquierda': 36.6, 'Derecha': 51.8}})
    publish_study(fresh_app.fs, study='voto-oculto', meta={'election': None}, parts={'summary': summary, 'elections': elections})
    page = html(call(api, '/analisis/voto-oculto')[2])
    assert 'Página viva · con la fila de 2026' in page and 'La fila de 2026 ya está en las cuentas de arriba' in page
    assert 'Sondeos finales, hoy' not in page and 'Generales 2026' in page and '36,6' in page and '−1,2' in page

def test_shy_page_without_a_run_speaks_of_the_past(api, fresh_app):
    summary = fixture_data('voto-oculto-summary') | {'current': None}
    publish_study(fresh_app.fs, study='voto-oculto', meta={'election': None}, parts={'summary': summary})
    status, _, body = call(api, '/analisis/voto-oculto'); page = html(body)
    assert status == 200 and 'Sondeos finales, hoy' not in page and 'a 24 días' not in page and 'Si la creencia tuviera razón' not in page

def test_shy_page_table_rows_and_dashes(api, fresh_app):
    summary = fixture_data('voto-oculto-summary')
    summary['parties'][7] = summary['parties'][7] | {'figures': None}
    publish_study(fresh_app.fs, study='voto-oculto', parts={'summary': summary})
    page = html(call(api, '/analisis/voto-oculto')[2])
    rows = re.findall(r'<tr[^>]*><td class="b">([^<]+)</td>', page[page.index('id="p3"'):page.index('id="despues"')])
    assert rows[:8] == ['PSOE', 'PP (AP, CD)', 'Vox', 'IU / Podemos / Sumar', 'Izquierda (bloque)', 'Derecha (bloque)', 'Gobierno, generales', 'Gobierno, autonómicas']
    assert 'Gobierno, autonómicas</td><td class="mut">–</td>' in page

def test_analisis_index_lists_voto_oculto_when_published(api, fresh_app):
    publish_study(fresh_app.fs, study='voto-oculto')
    page = html(call(api, '/analisis')[2])
    assert '/analisis/voto-oculto' in page and '¿Voto oculto de quién?' in page
    home = html(call(api, '/')[2])
    assert '/analisis/voto-oculto' in home

def test_metodo_promedio_links_to_voto_oculto_only_when_published(api, fresh_app):
    page = html(call(api, '/metodo/promedio')[2])
    assert '/analisis/voto-oculto' not in page
    publish_study(fresh_app.fs, study='voto-oculto')
    page = html(call(api, '/metodo/promedio')[2])
    assert '<a href="/analisis/voto-oculto">¿Voto oculto de quién?</a>' in page
    assert_public_wording(page, '/metodo/promedio')
```

(`fixture_data` returns the `data` of a fixture; if the file's `publish_study` cannot take `elections` through `parts`, use `tests.test_webapi_unit.write_study` directly, which does. The island (`initial`) of the first test: `json.loads` of the `#initial-data` script has `left_bars` of length 8 with `general` booleans, `vox_bars` of length 6 (the rows with `VOX`), `party_strip.order` of six labels and `party_strip.means['PSOE'] == -0.12`; add those four assertions to it.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pages.py -q -k "shy or voto_oculto"`
Expected: FAIL (404: no template registered).

- [ ] **Step 3: Write `pages/shy.py`, register the study, write the template and the Método sentence**

Texts from the mockup v4 with the live variables and the conditions of spec D6 (every strong sentence behind its flag with a neutral fallback of the same tone). In the Podemos note the Ciudadanos figure is omitted and the example «igual que al PP en abril de 2019 (+4,0)» is read from the `elections` row with `scope 'es'` and `event_date '2019-04-28'` (its `PP` value), omitted without that row. The pills of the header: «Página viva · añade la fila de {election_year} en cuanto haya escrutinio» with a run, «Página viva · con la fila de {last_year}» once published (no run, no `meta.election`), «Evidencia histórica · …» always. «Cómo leerlo» (`m.how`) in P1, P2 and P3. Add `'shy'` to `test_pages_package_layout`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pages.py -q`
Expected: PASS (`assert_public_wording` included).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/shy.py mtpy/lib/pages/analisis.py mtpy/lib/pages/__init__.py mtpy/lib/pages/metodo.py web/templates/analisis_voto_oculto.html web/templates/metodo_promedio.html web/dist/css/site.css tests/test_pages.py
git commit -m "Web: página /analisis/voto-oculto con veredictos, textos vivos condicionados, tablas y la isla de los gráficos; enlace desde /metodo/promedio"
```

---

### Task 7: The charts (`charts/errors.js`) and the page script

**Files:**
- Create: `web/dist/js/charts/errors.js`, `web/dist/js/pages/analisis-voto-oculto.js`
- Modify: `tests/test_js_charts.py:28-62` (alias `er` for `charts/errors.js` in `run_js`), new tests at the end

**Interfaces:**
- Produces in `charts/errors.js`: `errorBarsOption(rows, {mean = null, meanLabel = 'media', yMin = 8, step = 2, boldOf = (row) => Boolean(row.general), rotate = 55, dim = 0.55} = {}) -> option`: category x axis of `rows[].short` (`axisLabel.rotate`, `fontSize 10` (9 when `rotate >= 90`), `interval: () => true`, `formatter` returning `{bold|<short>}` for `boldOf(row)` rows with a `rich.bold` style of weight 700, plain `<short>` otherwise); value y axis symmetric `±yMax` with `yMax = max(yMin, ceil(max|value| / step) · step)` and `interval: step`, labels `fmtDelta(v, 0)`-style with the typographic minus (`fmtNum` of `format.js` with a sign); one `bar` series with `data[i] = {value, itemStyle: {color: value < 0 ? '#d62728' : '#1d4f86', opacity: boldOf(row) ? 1 : dim}}`, `markLine` at `mean` when not `null` (dashed `#e8912b`, label `${meanLabel} ${fmtDelta(mean, 1)}`), tooltip «{label}: {value with sign, 1 decimal}»; `renderErrorBars(el, rows, opts)`. `errorStripOption({rows, means, order}, {xMax = 8, step = 2} = {}) -> option`: category y axis of `order` (bottom to top as given), value x axis `−xMax..xMax` with `interval: step`, a `scatter` series of `rows` (`[value, party]`, `symbolSize 9`, color by sign as above with `opacity 0.5`, white border), a second `scatter` series of the means (`[means[label], label]` for the labels of `order` with a mean, `symbol 'rect'`, `symbolSize [4, 22]`, color `#e8912b`), a `markLine` at `x = 0` (black, solid), tooltip «{label} · {party}: {value}»; `renderErrorStrip(el, data, opts)`. Page script `analisis-voto-oculto.js` (pattern of `analisis-rebano.js`): `drawLeft(data)` into `#left-chart` with `{mean: data.left_mean, rotate: 90}`, `drawVox(data)` into `#vox-chart` with `{mean: data.vox_mean, yMin: 5, step: 2.5}`, `drawParties(data.party_strip)` into `#parties-chart`, `mountWhenReady(paint)`, `wireCardImages(document, showError)`.

- [ ] **Step 1: Write the failing tests**

```python
BARS = [{'short': 'GEN 77', 'label': 'Generales 1977', 'value': -1.82, 'general': True}, {'short': 'MD 21', 'label': 'Madrid 2021', 'value': 3.99, 'general': False},
        {'short': 'AN 18', 'label': 'Andalucía 2018', 'value': 8.77, 'general': False}]

def test_error_bars_option_colors_by_sign_scales_the_axis_and_marks_the_mean():
    option = run_js('er.errorBarsOption(%s, {mean: 0.47, rotate: 90})' % json.dumps(BARS))
    data = option['series'][0]['data']
    assert option['xAxis']['data'] == ['GEN 77', 'MD 21', 'AN 18'] and option['xAxis']['axisLabel']['rotate'] == 90 and option['xAxis']['axisLabel']['fontSize'] == 9
    assert [d['itemStyle']['color'] for d in data] == ['#d62728', '#1d4f86', '#1d4f86'] and [d['itemStyle']['opacity'] for d in data] == [1, 0.55, 0.55]
    assert (option['yAxis']['min'], option['yAxis']['max'], option['yAxis']['interval']) == (-10, 10, 2)
    assert option['series'][0]['markLine']['data'][0]['yAxis'] == 0.47
    labels = run_js('(() => { const o = er.errorBarsOption(%s); const f = o.xAxis.axisLabel.formatter; return [f("GEN 77", 0), f("MD 21", 1)]; })()' % json.dumps(BARS))
    assert labels == ['{bold|GEN 77}', 'MD 21']
    small = run_js('er.errorBarsOption(%s, {yMin: 5, step: 2.5})' % json.dumps(BARS[:2]))
    assert (small['yAxis']['min'], small['yAxis']['max'], small['yAxis']['interval']) == (-5, 5, 2.5) and 'markLine' not in small['series'][0]

def test_error_strip_option_lists_the_parties_with_their_means():
    data = {'rows': [{'party': 'PSOE', 'label': 'Generales 2023', 'value': -3.42}, {'party': 'PP', 'label': 'Generales 2023', 'value': 1.15}, {'party': 'PP', 'label': 'Madrid 2021', 'value': -3.78}],
            'means': {'PSOE': -0.12, 'PP': 0.02}, 'order': ['PP', 'PSOE']}
    option = run_js('er.errorStripOption(%s)' % json.dumps(data))
    assert option['yAxis']['data'] == ['PP', 'PSOE'] and (option['xAxis']['min'], option['xAxis']['max'], option['xAxis']['interval']) == (-8, 8, 2)
    points = option['series'][0]['data']
    assert [p['value'] for p in points] == [[-3.42, 'PSOE'], [1.15, 'PP'], [-3.78, 'PP']] and points[0]['itemStyle']['color'] == '#d62728' and points[1]['itemStyle']['color'] == '#1d4f86'
    assert [m['value'] for m in option['series'][1]['data']] == [[0.02, 'PP'], [-0.12, 'PSOE']] and option['series'][1]['symbol'] == 'rect'
    assert option['series'][0]['markLine']['data'][0]['xAxis'] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_js_charts.py -q -k "error_bars or error_strip"`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `errors.js` and the page script**

Imports and style from `charts/weeks.js` (`axisLabelStyle`, `baseOption`, `escapeHtml`, `mountChart`, `THEME`, `tooltipStyle`; `fmtNum`/`fmtDelta` of `format.js`). Check the page with the fixture bundle in a headless browser as the previous studies did (`check_app.py` + `check.mjs` of the scratchpad of 2026-10-26, or `node --check` plus the SSR of `tests/test_js_charts.py`); note what could not be verified in the task report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_js_charts.py tests/test_js_card_image.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/errors.js web/dist/js/pages/analisis-voto-oculto.js tests/test_js_charts.py
git commit -m "Web: gráficos de errores con signo (barras cronológicas y nube por partido) y script de /analisis/voto-oculto"
```

---

### Task 8: Closing state, memory and the full suite

**Files:**
- Modify: `docs/superpowers/specs/2026-11-05-web-analisis-voto-oculto-design.md` (a «Cierre (fecha)» section at the end)
- Test: the whole suite

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (report the new total; dev had 2146 passed with the rebaño integrated; the Andorra worktree may land in between, so the baseline is whatever `git log -1` of dev shows).

- [ ] **Step 2: Write the closing state**

Append to the spec a «Cierre (fecha)» section: what exists (files per task), the rulings taken while building (the `SIN DATOS` verdict without Vox data, the «Después» card after the publication showing the error row only, the subtitle tail by verdict, the omitted Ciudadanos figure, the PP April 2019 example read from `elections`), what was not verified (browser, S3, ECS) and Luis's list: `S3_BUCKET= python job.py publish '{"what":["study"],"study":"voto-oculto"}'` locally first, then against his S3 (or wait for the daily `update`, whose `studies` step publishes it once the image is deployed), walk `/analisis/voto-oculto` (desktop and phone: the 92 bars, the P3 table scroll, the download buttons), `/analisis`, Inicio and `/metodo/promedio`, deploy the image, and after the 29-N: `publish event` of `es/2026-11-29` then `publish '{"what":["studies"]}'` (the page adds the 2026 row by itself).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-11-05-web-analisis-voto-oculto-design.md
git commit -m "Spec voto-oculto: cierre de la construcción"
```

- [ ] **Step 4: Memory (outside the repo)**

Update `project-analisis-voto-oculto` to «construido» with the commit range and the pending steps of Luis, and the point 7 of `ideas-analisis-redes` to HECHO.
