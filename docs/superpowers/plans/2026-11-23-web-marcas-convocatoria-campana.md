# Key dates of an election («Convocatoria», «Campaña»…) on the time series — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The charts of the poll average and of the evolution of the updates mark the key dates of the election (its call, the start of its campaign and any other date written for it), and the table of the evolution compares the last day with the day before, the start of the campaign and the call.

**Architecture:** The dates live in a new `dates` key of each election entry of `data/params.json`, which the web already reads through a cache (`webapi.event_params`). One function of `mtpy/lib/pages/common.py` turns them into the list of marks every page puts in its initial data (`key_dates`); in the browser `seriesViews` adds them to the `marks` the average already draws, and the evolution chart and its table of changes learn to read them.

**Tech Stack:** Python 3.11 (Jinja2 pages of `mtpy/lib/pages`), plain ES modules with ECharts 5.6 (`web/dist/js`), pytest (the JS tests run the modules with `node`).

**Spec:** `docs/superpowers/specs/2026-11-23-web-marcas-convocatoria-campana-design.md`

## Global Constraints

- PEP8, lines ≤ 120, a numpydoc docstring on every Python function and a JSDoc on every JS function; identifiers and comments in English, public texts and test docstrings in Spanish.
- Public texts: «sólo» with its accent, «encuestadora» or «casa», never «empresa» (`assert_public_wording`).
- Labels of the marks: `Convocatoria` (`call`) and `Campaña` (`campaign`); a `key` date carries its own label.
- `call` is the day before the decree is published: 55 days before the election in the seed; `campaign`, 16 days before.
- The web draws only what the file says: an election without `dates` has no marks, and without `dates.call` no evolution card.
- No script is added to the repo: the seed of `params.json` runs from the session scratchpad.
- Tests never touch the network or a database. Commit only the files of the task, with `git add <files>` (the author commits in parallel in this tree: check `git status` and `git log` first, never rewrite his commits); message in Spanish in the style of the log, ending with the `Co-Authored-By` line.
- Run the tests with `python -m pytest` (pyenv 3.11.9).

## Review Focus

- A typo in `dates` (a date that is not `YYYY-MM-DD`, `key` written as a list, an empty label): the page renders without that mark, never a 500. (Task 1, `test_key_dates_skip_what_is_not_a_date`)
- An election without `dates` (the generals before 1994, a scope not called): no marks, the page renders, and the event page has no evolution card. (Task 4, `test_event_page_without_a_call_has_no_evolution_card`, `test_past_election_without_dates_has_no_key_dates`)
- A series of runs that starts after the call, or whose last day is the call or the first day of the campaign: that column is left out instead of comparing a day with itself or with the wrong run. (Task 3, `test_changes_leave_out_a_date_the_runs_do_not_reach`)
- A party missing on the day of the campaign or of the call (SALF without a figure that day): a dash in its cell, never `NaN`. (Task 3, `test_changes_compare_yesterday_the_campaign_and_the_call`)
- The downloaded image of the evolution with some parties hidden: the marks stay in the image. (Task 2, `test_evolution_image_keeps_the_marks`)

---

### Task 1: `dates` in `params.json` and the marks of an election

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`KEY_DATE_LABELS`, `key_dates`), `mtpy/lib/pages/__init__.py` (export `key_dates`)
- Modify: `data/params.json` (seed, written from the scratchpad), `data/README.md` (the `params.json` row)
- Test: `tests/test_pages_key_dates.py` (new), `tests/test_data_files.py`, `tests/test_load_unit.py`

**Interfaces:**
- Produces: `pages.key_dates(params: dict) -> list` — `[{'date': 'YYYY-MM-DD', 'label': str, 'kind': 'call' | 'campaign' | 'key'}]`, ascending by date and, on the same day, `call`, `campaign`, `key`. `params` is the entry of the election (`webapi.event_params(scope, date)`); `{}` or an entry without `dates` gives `[]`.
- Produces: `params.json` entries with `"dates": {"call": ..., "campaign": ..., "key": {date: label}}`, every key optional.

- [ ] **Step 1: Write the failing tests of `key_dates`** in `tests/test_pages_key_dates.py`

```python
def test_key_dates_give_the_call_the_campaign_and_the_key_dates_by_date():
    params = {'dates': {'campaign': '2004-02-27', 'call': '2004-01-19', 'key': {'2004-03-11': '11-M'}}}
    assert pages.key_dates(params) == [
        {'date': '2004-01-19', 'label': 'Convocatoria', 'kind': 'call'},
        {'date': '2004-02-27', 'label': 'Campaña', 'kind': 'campaign'},
        {'date': '2004-03-11', 'label': '11-M', 'kind': 'key'}]
    assert pages.key_dates({}) == [] and pages.key_dates({'bmaps': {}}) == [] and pages.key_dates({'dates': None}) == []
    # the three keys are optional, and a key date on the day of the call goes after it
    assert [entry['kind'] for entry in pages.key_dates({'dates': {'call': '2026-10-05', 'key': {'2026-10-05': 'X'}}})] == ['call', 'key']


def test_key_dates_skip_what_is_not_a_date():
    params = {'dates': {'call': '5 de octubre', 'campaign': 20261113, 'key': ['2026-11-01']}}
    assert pages.key_dates(params) == []
    params = {'dates': {'call': '2026-10-05', 'key': {'2026-11-01': '', '2026-13-40': 'X', '2026-11-02': 'Debate'}}}
    assert [entry['date'] for entry in pages.key_dates(params)] == ['2026-10-05', '2026-11-02']
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_pages_key_dates.py -q`
Expected: FAIL, `module 'mtpy.lib.pages' has no attribute 'key_dates'`

- [ ] **Step 3: Implement `KEY_DATE_LABELS = {'call': 'Convocatoria', 'campaign': 'Campaña'}` and `key_dates(params: dict) -> list`** in `mtpy/lib/pages/common.py`, and export `key_dates` from `mtpy/lib/pages/__init__.py`

A valid date is a `str` of ten characters that `date.fromisoformat` accepts; anything else is skipped silently, as is a `key` that is not a dict or a label that is not a non-empty `str`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest tests/test_pages_key_dates.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing tests of the file**

In `tests/test_data_files.py`:

```python
def test_params_dates_are_dates_in_their_order():
    """Las fechas clave de cada elección son fechas válidas: la convocatoria antes de la campaña y ésta antes de
    la elección; las próximas elecciones sin convocar no llevan ninguna."""
    params = json.load(open(os.path.join(DATA, 'params.json'), encoding='utf-8'))
    for scope, events in params.items():
        for event, entry in events.items():
            dates = entry.get('dates')
            if dates is None:
                continue
            assert set(dates) <= {'call', 'campaign', 'key'}, (scope, event)
            order = [dates[key] for key in ('call', 'campaign') if key in dates] + [event]
            assert order == sorted(order) and len(set(order)) == len(order), (scope, event)
            for day, label in list((dates.get('key') or {}).items()) + [(day, 'x') for day in order]:
                assert date.fromisoformat(day).isoformat() == day and isinstance(label, str) and label, (scope, event)
    assert params['es']['2026-11-29']['dates'] == {'call': '2026-10-05', 'campaign': '2026-11-13'}
    assert params['es']['2023-07-23']['dates'] == {'call': '2023-05-29', 'campaign': '2023-07-07'}
    assert params['es']['2019-11-10']['dates'] == {'call': '2019-09-24', 'campaign': '2019-11-01'}
    assert params['es']['2016-06-26']['dates']['call'] == '2016-05-03'
    assert params['es']['2004-03-14']['dates']['key'] == {'2004-03-11': '11-M'}
    # sin fechas: las generales anteriores a 1994 y las elecciones que aún no se han convocado
    assert 'dates' not in params['es']['1993-06-06'] and 'dates' not in params['es-md']['2027-05-23']


def test_params_file_is_written_in_its_own_style():
    """El fichero, con las fechas clave, sigue saliendo de `format_params` tal cual."""
    text = open(os.path.join(DATA, 'params.json'), encoding='utf-8').read()
    assert format_params(json.loads(text)) == text
```

In `tests/test_load_unit.py`, next to the tests of `merge_event_params`:

```python
def test_merge_event_params_keeps_the_key_dates():
    entry = {'parties': {'event': ['A'], 'polls': ['A']}, 'bmaps': {'main': ['A'], 'blocks': {}, 'vs': {}},
             'dates': {'call': '2026-10-05', 'campaign': '2026-11-13'}}
    derived = {'parties': {'event': ['B'], 'polls': ['B']}, 'bmaps': {'main': ['B'], 'blocks': {}, 'vs': {}}}
    assert merge_event_params(entry, derived)['dates'] == {'call': '2026-10-05', 'campaign': '2026-11-13'}
```

- [ ] **Step 6: Run them**

Run: `python -m pytest tests/test_data_files.py tests/test_load_unit.py -q -k "dates or own_style"`
Expected: `test_params_dates_are_dates_in_their_order` FAILS (`KeyError: 'dates'`); the other two PASS already (if `test_params_file_is_written_in_its_own_style` fails before the seed, stop: the seed would reformat the file).

- [ ] **Step 7: Seed `data/params.json`** with a one-off script kept in the session scratchpad (never in the repo)

The script loads the file with `json`, adds `dates` at the end of each entry and writes `mtpy.lib.data.format_params(data)` back, UTF-8:
- every election with `'1994-01-01' <= date < '2026-11-23'`: `call` = date − 55 days, `campaign` = date − 16 days;
- `es` `2016-06-26`: `call` `2016-05-03`; `es` `2019-11-10`: `call` `2019-09-24`, `campaign` `2019-11-01` (check both decrees with a web search before writing them; if a date differs, write the checked one and say so in the spec);
- `es` `2026-11-29`: `call` `2026-10-05`, `campaign` `2026-11-13`;
- `es` `2004-03-14`: add `key` `{"2004-03-11": "11-M"}`.

Then `git diff --stat data/params.json` must show additions only.

- [ ] **Step 8: Run the tests of the file and of the data**

Run: `python -m pytest tests/test_data_files.py tests/test_load_unit.py tests/test_data_unit.py tests/test_pages_key_dates.py -q`
Expected: PASS

- [ ] **Step 9: Document the key** in the `params.json` row of `data/README.md`: `dates`, the key dates of the election the web marks on the time series (`call`, the day of the call, the one before the decree is published; `campaign`, the first day of the campaign; `key`, other dates with their label), written by hand.

- [ ] **Step 10: Commit**

```bash
git add mtpy/lib/pages/common.py mtpy/lib/pages/__init__.py data/params.json data/README.md tests/test_pages_key_dates.py tests/test_data_files.py tests/test_load_unit.py
git commit -m "Datos: fechas clave de cada elección en params.json (dates: call, campaign, key) y pages.key_dates, que las da como marcas"
```

### Task 2: The marks on the charts

**Files:**
- Modify: `web/dist/js/charts/series.js` (`alongMark`, `marksSeries`), `web/dist/js/pages/series-wire.js` (`seriesViews`), `web/dist/js/pages/sondeo.js` (`dataOf`), `web/dist/js/charts/evolution.js` (`evolutionOption`, `evolutionImageOption`), `web/dist/js/pages/evolution-wire.js` (`dataOf`)
- Test: `tests/test_js_charts.py`, `tests/test_js_evolution.py`

**Interfaces:**
- Consumes: `initial.key_dates` — `[{date, label, kind}]` (Task 1 shape; absent on a page without it).
- Produces: `alongMark({date, label}, position = 'insideEndTop')` (the mark-line item of a date, its label written along the line inside the plot) and `marksSeries(data)` (the silent empty `line` series named `marks` that carries the mark lines), both exported by `series.js`; `seriesOption` builds its marks with them, with the same output as today.
- Produces: `seriesViews(initial, catalog)` → `partidos.marks` = the key dates as `{date, label}`; `bloques.marks` = the key dates followed by the party entries.
- Produces: `evolutionOption({..., marks = []})` — with marks, one more series, `marksSeries`, after the party lines; without them, the series of today.

- [ ] **Step 1: Write the failing tests**

In `tests/test_js_charts.py`:

```python
def test_series_views_carry_the_key_dates_in_both_views():
    """Las fechas clave de la elección van en las dos vistas; las entradas de partidos, sólo en Bloques, detrás."""
    initial = {'series': {'dates': ['2026-10-01'], 'parties': ['PP', 'PSOE'], 'mean': {'PP': [30], 'PSOE': [28]},
                          'lo': {}, 'hi': {}, 'first_polls': {'SALF': '2024-06-06'}},
               'polls': {'parties': [], 'polls': []}, 'projection': None,
               'key_dates': [{'date': '2026-10-05', 'label': 'Convocatoria', 'kind': 'call'},
                             {'date': '2026-11-13', 'label': 'Campaña', 'kind': 'campaign'}],
               'catalog': {'parties': [{'name': n, 'color': '#000000', 'fullname': n, 'block': None} for n in ('PP', 'PSOE')],
                           'bmaps': {'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}}}
    views = run_js('sw.seriesViews(%s, new c.Catalog(%s.catalog))' % (json.dumps(initial), json.dumps(initial)))
    key = [{'date': '2026-10-05', 'label': 'Convocatoria'}, {'date': '2026-11-13', 'label': 'Campaña'}]
    assert views['partidos']['marks'] == key
    assert views['bloques']['marks'] == key + [{'date': '2024-06-06', 'label': 'Entra SALF'}]


def test_series_marks_keep_their_shape_through_the_shared_builders():
    """`alongMark` y `marksSeries` dan lo mismo que `seriesOption` dibujaba: la etiqueta a lo largo de la línea."""
    mark = run_js("s.alongMark({date: '2026-10-05', label: 'Convocatoria'})")
    assert mark['label'] == {'formatter': 'Convocatoria', 'position': 'insideEndTop', 'color': '#555'}
    series = run_js("s.marksSeries([s.alongMark({date: '2026-10-05', label: 'Convocatoria'})])")
    assert series['name'] == 'marks' and series['data'] == [] and series['silent'] is True
    assert series['markLine']['lineStyle'] == {'type': 'dashed', 'color': '#555'} and len(series['markLine']['data']) == 1
```

In `tests/test_js_evolution.py` (`option(**over)` is the helper of the file):

```python
MARKS = [{'date': '2026-10-05', 'label': 'Convocatoria', 'kind': 'call'}, {'date': '2026-10-07', 'label': 'Campaña', 'kind': 'campaign'}]


def test_evolution_option_draws_the_key_dates_as_mark_lines():
    """Las fechas clave son líneas discontinuas de una serie aparte, que la leyenda no toca; sin ellas, ninguna."""
    drawn = option(marks=MARKS)
    marks = drawn['series'][-1]
    assert marks['name'] == 'marks' and 'marks' not in drawn['legend']['data']
    assert [m['label']['formatter'] for m in marks['markLine']['data']] == ['Convocatoria', 'Campaña']
    assert all(entry['name'] != 'marks' for entry in option()['series'])


def test_evolution_image_keeps_the_marks():
    """La imagen deja fuera los partidos apagados, pero no las marcas."""
    data = dict(DATA, marks=MARKS)
    image = run_js('e.evolutionImageOption({...%s, runs: e.dailyRuns(runs)})' % json.dumps(data))
    assert [entry['name'] for entry in image['series']] == ['PP', 'PSOE', 'marks']
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_js_charts.py tests/test_js_evolution.py -q -k "key_dates or builders or marks"`
Expected: the four new tests FAIL (`s.alongMark is not a function`, `marks` missing)

- [ ] **Step 3: Implement**

- `series.js`: export `alongMark` and `marksSeries`; `seriesOption` uses them for its entries and its mark series (no change in its output).
- `series-wire.js`: `seriesViews` reads `initial.key_dates || []`, keeps `{date, label}` of each and puts them in `partidos.marks` and in front of the entries in `bloques.marks`.
- `sondeo.js`: `dataOf()` passes `marks: current.marks || []`, as `promedios.js` and `elecciones.js` do.
- `evolution.js`: `evolutionOption` takes `marks`; a mark on the day of the first run takes `alongMark(entry, 'insideEndBottom')` (its label on the plot side of the line, off the axis labels; Task 5 confirms the side in the browser), the others the default. `evolutionImageOption` keeps the series named `marks`.
- `evolution-wire.js`: `dataOf` passes `marks: initial.key_dates || []`.

ECharts already leaves out a mark line whose day is outside the time axis (`containData`), so nothing filters them.

- [ ] **Step 4: Run the JS tests**

Run: `python -m pytest tests/test_js_charts.py tests/test_js_evolution.py tests/test_js_event_page.py tests/test_js_blocks.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/series.js web/dist/js/pages/series-wire.js web/dist/js/pages/sondeo.js web/dist/js/charts/evolution.js web/dist/js/pages/evolution-wire.js tests/test_js_charts.py tests/test_js_evolution.py
git commit -m "Web: las fechas clave de la elección como marcas de los gráficos del promedio y de la evolución"
```

### Task 3: The table of the evolution compares with the campaign and the call

**Files:**
- Modify: `web/dist/js/evolution-changes.js` (`evolutionChanges`, `changesTableHtml`, the header comment), `web/dist/js/pages/evolution-wire.js` (`fillChanges`)
- Test: `tests/test_js_evolution.py` (rewrite the tests of the week and the start: `test_changes_compare_yesterday_a_week_and_the_start`, `test_changes_week_is_null_without_a_day_a_week_back`, `test_changes_table_html`)

**Interfaces:**
- Consumes: `initial.key_dates` (the dates of kind `campaign` and `call`).
- Produces: `evolutionChanges(runs, names, field = 'pct', {campaign = null, call = null} = {})` → `{days: {last, previous, campaign, call}, rows: [{name, value, day, campaign, call}]}`, or null with fewer than two days. `days.campaign` and `days.call` are the day of the reference run of each date, or null.
- Produces: `changesTableHtml(changes, view, options)` with the columns `day`, `campaign`, `call` in that order, a null day leaving its column out.

Rule of a date column (`campaign`, then `call`): the reference run is the last daily run on that date or before it. The column is null when there is no date, when the last day is not after the date, when no run is on or before it, or when the day of the reference run already heads an earlier column (`previous`, or `campaign` for `call`). The change of a name is its figure shown on the last day minus the one of the reference run (`shownValue`), null when it has no figure on either.

- [ ] **Step 1: Rewrite the failing tests** over `CHANGE_DAYS` (October 5 to 15, without the 8th) and `CHANGE_VALUES` of the file; `changes(...)` stands for the helper the current tests use to call `ch.evolutionChanges` with those runs, extended with the dates:

```python
def test_changes_compare_yesterday_the_campaign_and_the_call():
    """Tres columnas: el día anterior, el inicio de la campaña y la convocatoria; un partido sin cifra ese día, raya."""
    out = changes(campaign='2026-10-12', call='2026-10-05')
    assert out['days'] == {'last': '2026-10-15', 'previous': '2026-10-14', 'campaign': '2026-10-12', 'call': '2026-10-05'}
    pp = next(row for row in out['rows'] if row['name'] == 'PP')
    assert pp['value'] == 31.9 and pp['day'] == -0.1 and pp['call'] == -0.8
    # the name without a figure on the day of the call (the fixture of the missing name): null, written as a dash
    assert next(row for row in out['rows'] if row['name'] == 'VOX')['call'] is None


def test_changes_leave_out_a_date_the_runs_do_not_reach():
    """Sin fecha, antes de que empiece, con la serie empezada después o repetida en otra columna: sin columna."""
    assert changes()['days']['campaign'] is None and changes()['days']['call'] is None
    # the campaign starts on the last day or later: two columns
    assert changes(campaign='2026-10-15', call='2026-10-05')['days']['campaign'] is None
    assert changes(campaign='2026-10-20', call='2026-10-05')['days']['call'] == '2026-10-05'
    # the series starts after the call
    assert changes(call='2026-10-01')['days']['call'] is None
    # the 8th has no run: the reference of a campaign that starts that day is the 7th
    assert changes(campaign='2026-10-08', call='2026-10-05')['days']['campaign'] == '2026-10-07'
    # the day before is the day of the campaign: one column for both
    out = changes(campaign='2026-10-14', call='2026-10-05')
    assert out['days']['previous'] == '2026-10-14' and out['days']['campaign'] is None and out['days']['call'] == '2026-10-05'


def test_changes_table_html():
    """Cabeceras con fecha («vs 14-oct»), con lo que es cada una en el título; la columna sin día no sale."""
    html = table(campaign='2026-10-12', call='2026-10-05')
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', html) == ['Partido', '15-oct', 'vs 14-oct', 'vs 12-oct', 'vs 5-oct']
    assert 'title="Frente al 12 de octubre de 2026, inicio de la campaña"' in html
    assert 'title="Frente al 5 de octubre de 2026, convocatoria"' in html
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', table(call='2026-10-05')) == ['Partido', '15-oct', 'vs 14-oct', 'vs 5-oct']
```

Take the exact figures (`31.9`, `-0.1`, `-0.8`), the missing name and the date text of the titles (`fmtDate`) from the fixtures and the current tests of the file; keep `test_changes_missing_name_gives_a_dash` and `test_changes_with_a_single_day` passing.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_js_evolution.py -q -k changes`
Expected: FAIL (`days` has `week` and `start`)

- [ ] **Step 3: Implement** `evolutionChanges` and `changesTableHtml` as the Interfaces block says (titles: `Frente al {fmtDate(day)}`, then `, inicio de la campaña` or `, convocatoria`), update the header comment of the module, and in `evolution-wire.js` pass `{campaign, call}`, the dates of those kinds in `initial.key_dates`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_js_evolution.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/evolution-changes.js web/dist/js/pages/evolution-wire.js tests/test_js_evolution.py
git commit -m "Web: la tabla de la evolución compara con el día anterior, el inicio de la campaña y la convocatoria"
```

### Task 4: The pages pass the key dates, and the evolution card needs a call

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`promedios_context`), `mtpy/lib/pages/sondeos.py` (`sondeo_context`), `mtpy/lib/pages/elecciones.py` (`eleccion_context`)
- Modify: `web/templates/promedios.html`, `web/templates/eleccion.html`, `web/templates/sondeo.html` (the «Cómo se lee» texts)
- Test: `tests/test_pages.py` (and the tests of the poll page, in the file that already covers its `initial`)

**Interfaces:**
- Consumes: `pages.key_dates(params)` (Task 1), `webapi.event_params(scope, date)`.
- Produces: `context['key_dates']` and `initial['key_dates']` on the three pages (in the event page, right after `runs`); `context['evolution']` (bool: the election of the run has a `call`) on the event page.

- [ ] **Step 1: Write the failing tests** in `tests/test_pages.py` (`api`, `call`, `initial`, `html`, `write_far_bundle` are the helpers of the file; the fixture run is `es` for `2026-11-29`, the past election of the fixture `2023-07-23`)

```python
def test_event_page_passes_the_key_dates_of_its_election(api):
    """La página del ámbito lleva la convocatoria y el inicio de la campaña del 29-N, y la tarjeta de la evolución."""
    body = call(api, '/es', query='mode=forecast')[2]
    assert initial(body)['key_dates'] == [{'date': '2026-10-05', 'label': 'Convocatoria', 'kind': 'call'},
                                          {'date': '2026-11-13', 'label': 'Campaña', 'kind': 'campaign'}]
    page = html(body)
    assert 'id="evolution-chart"' in page and 'frente al día anterior, al inicio de la campaña' in page


def test_event_page_without_a_call_has_no_evolution_card(api, fresh_app):
    """Sin convocatoria en params.json no hay tarjeta de la evolución; la sección sigue con la de la encuestadora."""
    write_far_bundle(fresh_app.fs)
    body = call(api, '/es')[2]
    page = html(body)
    assert initial(body)['key_dates'] == [] and 'id="evolution-chart"' not in page
    assert 'id="pollster-chart"' in page and '<section class="event-section" id="evolucion">' in page
    assert 'fechas clave' not in page


def test_past_election_passes_its_key_dates(api):
    """El detalle de una elección pasada lleva sus fechas clave."""
    data = initial(call(api, '/elecciones/es/generales/2023-07')[2])
    assert [(entry['kind'], entry['date']) for entry in data['key_dates']] == [('call', '2023-05-29'), ('campaign', '2023-07-07')]


def test_past_election_without_dates_has_no_key_dates(api, monkeypatch):
    """Una elección sin `dates` (las anteriores a 1994) se pinta sin marcas."""
    monkeypatch.setattr(pages.elecciones, 'event_params', lambda scope, date: {})
    body = call(api, '/elecciones/es/generales/2023-07')[2]
    assert initial(body)['key_dates'] == [] and 'fechas clave' not in html(body)
```

Add the same check of `initial['key_dates']` to the test of the poll page that already reads its `initial`, and insert `'key_dates'` after `'runs'` in the list of `test_event_page_initial_keys`. Take the address of the past election and of the poll from the tests of the file that already call them.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_pages.py -q -k "key_dates or without_a_call or initial_keys"`
Expected: FAIL (`KeyError: 'key_dates'`)

- [ ] **Step 3: Implement the contexts**

- `promedios_context`: `dates = key_dates(event_params(scope, meta.get('event_date')))`; `context['key_dates'] = dates`; `context['evolution'] = any(entry['kind'] == 'call' for entry in dates)`; `'key_dates': dates` in `initial` after `'runs'`; `evolucion` joins the `missing` sections when there is neither `context['evolution']` nor `context['pollsters']`.
- `sondeo_context` and `eleccion_context` (which already has `params`): `context['key_dates']` and `initial['key_dates']`.

- [ ] **Step 4: Change the templates**

- `promedios.html`: the evolution card inside `{% if evolution %}`; the `#evolucion` section inside `{% if evolution or pollsters %}`.
- Second paragraph of the «Cómo se lee» of the evolution, replaced by: «La tabla que se abre con «Ver los datos» da la cifra de cada partido el último día y cuánto ha cambiado frente al día anterior, al inicio de la campaña (cuando ya ha empezado) y a la convocatoria, cada columna con su fecha: flecha verde si sube y roja si baja. Recoge todos los partidos, también los que no estén marcados en la leyenda.»
- One sentence, only `{% if key_dates %}`, at the end of the first paragraph of the «Cómo se lee» of each chart with marks (the average and the evolution in `promedios.html`, the average in `eleccion.html`, the house chart in `sondeo.html`): «Las líneas verticales discontinuas marcan las fechas clave de la elección, como la convocatoria y el inicio de la campaña.»

- [ ] **Step 5: Run the page tests**

Run: `python -m pytest tests/test_pages.py tests/test_pages_key_dates.py -q` and then every other `tests/test_pages_*.py`
Expected: PASS. A test that rendered the evolution card for an election without `dates` in the real file (another scope, a far election) now fails for a good reason: give its election a call by patching `event_params` in that test, or assert the absence, whichever the test was about.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q -m "not integration"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add mtpy/lib/pages/promedios.py mtpy/lib/pages/sondeos.py mtpy/lib/pages/elecciones.py web/templates/promedios.html web/templates/eleccion.html web/templates/sondeo.html tests/
git commit -m "Web: las páginas del ámbito, del sondeo y de una elección pasada pasan las fechas clave; la tarjeta de la evolución sólo con la elección convocada"
```

### Task 5: Check in the browser and close

**Files:**
- Modify (only if the check asks for it): `web/dist/js/charts/series.js`, `web/dist/js/charts/evolution.js` and their tests
- Modify: `docs/superpowers/specs/2026-11-23-web-marcas-convocatoria-campana-design.md` (a «Cambios tras la implementación» section, if anything differs from the design)

- [ ] **Step 1: Serve the pages** with the browser harness of the earlier web work (a fixture bundle with the real published history of `es` and the real `data/`, served locally; headless Chrome driven from a script in the scratchpad) and take screenshots, desktop (1280 px) and phone (390 px), of: `/es` average with the whole cycle and at 3 months, in Partidos and in Bloques; the evolution chart and its table open; the page of a poll; the detail of the general election of 2004 (the 11-M) and of 2023.

- [ ] **Step 2: Look at each screenshot.** Every mark must be readable: «Convocatoria» and «Campaña» do not cover each other, «Estimación», «Elección», an «Entra …» label or the axis labels; in the evolution, «Convocatoria» sits on the first point with its label inside the plot; a mark outside the window is not drawn; the table reads «vs 22-nov», «vs 13-nov», «vs 5-oct» (the days of the real history). Fix what fails (the `position` of a label, a margin) with its test, and repeat.

- [ ] **Step 3: Download the image** of the average and of the evolution (with the table open) and look at them: the marks are in both.

- [ ] **Step 4: Run the whole suite again**

Run: `python -m pytest -q -m "not integration"`
Expected: PASS

- [ ] **Step 5: Write what changed** from the design in the spec, if anything, and commit the fixes and the docs.
