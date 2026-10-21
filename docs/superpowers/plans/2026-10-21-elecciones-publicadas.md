# Elecciones publicadas sin entrar en el modelo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish on the web every past election with an official result (not only the `featured` ones), with its poll errors everywhere errors per election are shown, while every model estimator keeps using only `featured` elections.

**Architecture:** "Published" is derived, not stored: `get_events` gains `has_results`. The `Computer` gains a third poll selection, `featured='results'` (polls of elections with an official result), used only by the two publications that show errors (`publish analysis` → `poll-errors`, `publish events`). The event `meta` part carries `featured`, and the election page shows a one-line notice when it is false.

**Tech Stack:** Python 3, pandas, pytest, Jinja2 templates, JSON bundle (`mtpy/lib/bundle.py`).

**Spec:** `docs/superpowers/specs/2026-10-21-elecciones-publicadas-design.md`

## Global Constraints

- The default of `Computer.filter_polls` / `get_polls_metric` stays `featured=True`. `featured=False` keeps meaning "no election filter" (`party_first_polls`, `computer.py:2424`, depends on it).
- Nothing that feeds the model changes: rating, ranking, «Camino al rating», house effects (`pollsters_parties`, `house-effects` part), drift, herding, `error-data`, average, forecast, backtest. Only `poll-errors` and the event bundles change.
- No database migration: no new column, no `ALTER`.
- Code names in neutral English; public texts in Spanish, in the first person plural ("usamos"), with no fixed labels.
- The profile card «Efectos de casa por elección» stays with the analysed (`featured`) elections only. Don't touch it.
- Leave «elecciones analizadas» in `casa.html` and `metodo_promedio.html` as is: in public texts it means "elections whose polls we measured", which is still true.
- Luis commits in parallel in the same tree (`git commit -a`). Before each commit, run `git status` and stage only this task's files by name.
- Test command: `pytest -q` (baseline at `ad9eff4`: 1233 passed, 75 skipped).

## Review Focus

1. **The current cycle, with no result yet, must not get into `poll-errors`.** `featured='results'` has to leave out the polls of an election with no `votes` (the upcoming one). Pinned in Task 2.
2. **An event bundle published before this change has no `featured` in its `meta`.** The page must read it as analysed (`meta.get('featured', True)`) and not show the notice. Pinned in Task 4.
3. **A featured election must publish the same as before.** Its `cycle` and errors with `'results'` are the same polls as with `True`. Pinned in Task 2: the two selections are equal when every election with a result is featured.
4. **`event_dates_to_publish('all')` leaves out today and the future**, even if a future date somehow had `has_results`. Kept by the existing `days < today` check; Task 3's test covers it with the 2026-11-29 row.
5. **An election with a result but a `Computer` that raises or returns no errors** still gives `NothingToPublish`, not a half-written bundle. Already covered by the existing tests, which must keep passing unchanged.

---

### Task 1: `get_events` says which elections have an official result

**Files:**
- Modify: `mtpy/lib/data.py` (`get_events`, ~line 647)
- Test: `tests/test_data_unit.py`

**Interfaces:**
- Produces: `get_events(scope: str) -> pd.DataFrame` with the current columns plus `has_results: bool`. It is true when `events_data` has a row of that `scope`/`date` with `region_id = 0` and `votes IS NOT NULL` (the same rule as `update_featured`).

- [ ] **Step 1: Write the failing test** `test_get_events_flags_the_elections_with_an_official_result`. It monkeypatches `data.Events` and `data.EventsData` with `FakeTable` subclasses, in the style of `test_get_pollster_events_counts_the_rated_polls_of_each_election`. Events: `2019-05-26` (featured False), `2023-05-28` (featured True), `2027-05-23` (featured False). EventsData returns rows only for `2019-05-26` and `2023-05-28`. Assert `get_events('es-ar')['has_results'].tolist() == [True, True, False]`, with dtype bool. Also assert the EventsData query filters contain `"scope = 'es-ar'"`, `'region_id = 0'` and `'votes IS NOT NULL'`.
- [ ] **Step 2: Run** `pytest tests/test_data_unit.py -q -k official_result`. Expected: FAIL (KeyError `has_results`).
- [ ] **Step 3: Implement.** In `get_events`, read `EventsData` with those three filters and set `has_results = events['date'].isin(data['date'])`. Update the docstring. In `update_featured`, replace its own `isin(data['date'])` with `get_events(scope)['has_results']`, keeping the result the same.
- [ ] **Step 4: Run** `pytest tests/test_data_unit.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `mtpy/lib/data.py tests/test_data_unit.py`. Message: «get_events: has_results, las elecciones con resultado oficial».

### Task 2: the `'results'` selection in the `Computer`

**Files:**
- Modify: `mtpy/lib/computer.py` (`filter_polls` ~615, `get_polls_metric` ~2061)
- Test: `tests/test_computer_unit.py`

**Interfaces:**
- Produces: `Computer.filter_polls(featured: bool | Literal['results'] = True, **kwargs)` and `get_polls_metric(..., featured: bool | Literal['results'] = True, ...)`, which passes it through. Semantics:
  - `True`: the current filter (`df['featured'] & df['computed']`).
  - `'results'`: polls with `computed` true whose `event_date` has an official result, i.e. `df['event_date'].map(self.events['votes'].notna()).fillna(False)`. `self.events` is indexed by date and `votes` comes from `get_event_series`. Don't use the `computed` of election rows: `load_events` sets it to `featured`.
  - `False`: no election filter, as today.
  - The other filters (`pollster` notnull, `drop_mtypes`, `drop_ctypes`, `weight_over > 0`, `drange`, `n_last`) apply in all three cases.

- [ ] **Step 1: Write the failing tests.** Build on `single_event_computer()` (`tests/test_computer_unit.py` ~380) a computer with three elections: `2015-05-24` featured with result, `2019-05-26` not featured with result, `2023-05-28` not featured without result (`votes` NaN). Give each one poll, and set `c.events = pd.DataFrame({'featured': [...], 'votes': [1000, 900, np.nan]}, index=pd.to_datetime([...]))`.
  - `test_filter_polls_results_takes_every_election_with_an_official_result`: the event dates of `c.filter_polls('results')` == {2015-05-24, 2019-05-26}; of `c.filter_polls(True)` == {2015-05-24}; of `c.filter_polls(False)` == all three.
  - `test_filter_polls_results_equals_featured_when_every_election_with_a_result_is_featured`: with `2019-05-26` set featured, `filter_polls('results')` equals `filter_polls(True)` (`pd.testing.assert_frame_equal`).
  - `test_get_polls_metric_passes_the_results_selection`: `c.get_polls_metric('error', bmap='vs', featured='results')` has rows for the two elections with a result.
- [ ] **Step 2: Run** `pytest tests/test_computer_unit.py -q -k results`. Expected: FAIL.
- [ ] **Step 3: Implement** the branch in `filter_polls` and the type and docstring in both methods.
- [ ] **Step 4: Run** `pytest tests/test_computer_unit.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `mtpy/lib/computer.py tests/test_computer_unit.py`. Message: «Computer: selección 'results' de sondeos, las elecciones con resultado oficial».

### Task 3: the publications use the `'results'` selection

**Files:**
- Modify: `mtpy/lib/publish/analysis.py:258`, `mtpy/lib/publish/events.py` (`event_dates_to_publish` ~314, `publish_event` ~349, `export_event_meta` ~277), `mtpy/lib/bundle.py:124` (schema `event-meta`)
- Modify: `tests/fakes.py` (`synthetic_event`), `tests/fixtures/bundle/event-meta.json`
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Consumes: `has_results` (Task 1) and `featured='results'` (Task 2).
- Produces: the `event-meta` schema with `'featured': bool`, and `export_event_meta(..., featured: bool, ...)`, which writes it. `publish_event` writes in `meta.featured` the election's `featured` value from `get_events`.

- [ ] **Step 1: Adapt the fakes.** In `synthetic_event`, set `events['has_results'] = [True, True, False]`, and update its docstring. In `tests/fixtures/bundle/event-meta.json`, add `"featured": true` to `data`.
- [ ] **Step 2: Write and adjust the failing tests.**
  - In `test_publish_analysis_writes_the_parts_and_meta_last`, the expected call is now `('get_polls_metric', 'error', 'vs', {'drange': (6, 42), 'n_last': 1, 'featured': 'results'})`. `POLL_ERRORS_PARAMS` stays the same, because `params` of the analysis meta records it. Pass `featured='results'` apart.
  - In `test_publish_event_skips_what_cannot_be_published`, the 2026-11-29 and 2020-01-01 messages become `'2026-11-29 is not an election of es with official results'` and `'2020-01-01 is not an election of es with official results'`. 2019-11-10 keeps `'no official results for es 2019-11-10'`.
  - New `test_publish_event_publishes_an_election_with_results_that_is_not_featured`: `synthetic_event()` with `featured` of 2023-07-23 set to False. Publishing succeeds. `meta.data.featured is False`. The `FakeComputer` calls include `('filter_polls', 'results', {})` and a `get_polls_metric` with `featured='results'`.
  - In `test_publish_event_writes_the_parts_and_meta_last`, add `meta['data']['featured'] is True`.
  - New `test_event_dates_to_publish_takes_the_past_elections_with_results`: events 2019-11-10 (featured False, has_results True), 2023-07-23 (True, True), 2026-11-29 (False, False). `event_dates_to_publish('es', 'all', events=..., today='2026-10-11') == ['2019-11-10', '2023-07-23']`. Add a fourth row, 2027-01-01 with has_results True, which must also stay out because it is after today.
- [ ] **Step 3: Run** `pytest tests/test_publish_unit.py -q`. Expected: the new and adjusted tests FAIL.
- [ ] **Step 4: Implement.**
  - `analysis.py:258`: `comp.get_polls_metric('error', featured='results', **POLL_ERRORS_PARAMS)`.
  - `event_dates_to_publish`: `has_results` instead of `featured`. Docstring: "the past elections of the scope with an official result".
  - `publish_event`: select the row by `has_results`, with the error message above. Use `get_polls_metric('error', bmap='vs', featured='results')` and `filter_polls('results')`. Pass `featured=bool(row['featured'].iloc[0])` to `export_event_meta`. Update the docstrings that say «featured».
  - `bundle.py`: `'featured': bool` in `event-meta`.
- [ ] **Step 5: Run** `pytest tests/test_publish_unit.py tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_jobs_publish.py -q`. Expected: PASS. If a test checks the fixtures against the schema, it should pass thanks to Step 1.
- [ ] **Step 6: Commit** the eight files above by name. Message: «publish: elecciones con resultado oficial en poll-errors y en events, featured en la meta».

### Task 4: notice on the page of an election that is not analysed

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py` (`eleccion_context`, ~1085-1160), `web/templates/eleccion.html:277-279`
- Modify: `tests/test_webapi_unit.py` (`write_event`: new parameter `featured=True`, written to the meta)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `meta.featured` (Task 3).
- Produces: context key `analysed: bool = bool(meta.get('featured', True))`.

- [ ] **Step 1: Write the failing tests** in `tests/test_pages.py`, next to `test_eleccion_without_model_shows_a_notice`:
  - `test_eleccion_not_analysed_says_it_does_not_count_for_the_rating`: `write_event(fresh_app.fs, date='2019-11-10', model=False, featured=False)`. The page at `/elecciones/es/generales/2019-11` contains the notice text below and does not contain `Sin backtest para esta elección`.
  - `test_eleccion_meta_without_featured_counts_as_analysed`: write an event and drop `featured` from its `meta.json` (rewrite the JSON). The page shows `Sin backtest para esta elección` and not the new notice.
- [ ] **Step 2: Run** `pytest tests/test_pages.py -q -k "not_analysed or without_featured"`. Expected: FAIL.
- [ ] **Step 3: Implement.** Set `context['analysed']`. In the template, inside the `model` card, turn the `{% else %}` of `horizon_tables` into this branch:

```jinja
{% elif not analysed %}
<p class="notice">Esta elección tuvo menos de 10 sondeos útiles, así que no la usamos para el rating de las encuestadoras ni para el backtest.</p>
{% else %}
<p class="notice">Sin backtest para esta elección.</p>
```

- [ ] **Step 4: Run** `pytest -q`. Expected: everything passes (≥ 1233 + the new tests), with no failures.
- [ ] **Step 5: Commit** `mtpy/lib/pages/elecciones.py web/templates/eleccion.html tests/test_webapi_unit.py tests/test_pages.py`. Message: «Ficha de elección: aviso cuando no cuenta para el rating ni el backtest».

### Task 5: closing

- [ ] **Step 1:** Mark the spec as built («Estado: aprobada… Construida el … en …»), with the commit range. Commit it.
- [ ] **Step 2:** Tell Luis what he has to run, in this order: `publish analysis` for every scope, then `publish events --event-date all` for every scope, then walk Aragón 2019, the /casas history, the CIS profile and /metodo/promedio.
