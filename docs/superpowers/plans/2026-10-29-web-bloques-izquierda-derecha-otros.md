# Web «Bloques» = Izquierda, Derecha y Otros — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every «Bloques» view of the site shows three blocks — Izquierda and Derecha from `bmaps.vs`, and Otros as the remainder up to 100 % / the chamber — instead of the four blocks of `bmaps.blocks`.

**Architecture:** The run publishes the three blocks (summary table `blocks`, projection group `blocks`, meta `bmaps.blocks`), with Otros as the complement of Izquierda + Derecha so its 95 % range is real. Python pages order them, paint Otros grey, group past results and the «vs {año}» deltas with `vs` (deltas through the smap lineage). The browser computes Otros of polls/series/results as 100 − the other blocks.

**Tech Stack:** Python 3 (pandas, Jinja2 controllers in `mtpy/lib/pages`), vanilla ES modules in `web/dist/js` tested from pytest through `node`.

**Spec:** `docs/superpowers/specs/2026-10-28-web-bloques-izquierda-derecha-otros-design.md`

## Global Constraints

- Block names, exactly: `'Izquierda'`, `'Derecha'`, `'Otros'` (the rest block; constant `REST_BLOCK`).
- Table order: Izquierda, Derecha, Otros; hemicycle and Escaños stacked bar: Izquierda, Otros, Derecha; unknown names (old runs with four blocks) last, in input order.
- Otros colour: `OTHERS_COLOR` `#9e9e9e`, always, even when the map lists parties under Otros.
- Otros has no P(mayoría): `p_majority` null, rendered «–».
- Otros vote = 100 − I − D; Otros seats = chamber − I − D. Only when both I and D have a value; otherwise Otros has none.
- Applies to every scope (national and regional).
- Do not touch: `params.json`, the model, `bmaps.vs`/`bmaps.main` in the meta, `_p_majority` (still `vs`), anything that already reads `vs` (Inicio, Calculadora, dianas, Sesgo bloques, Histórico, rating, `charts/diverging.js`), «Lo que dijo el modelo», party ordering (`HEMICYCLE_BLOCKS`, `catalog.order`, `BLOCK_ORDER` of `catalog.js` for parties).
- Identifiers in neutral English; public text in Spanish with «sólo» accented, «encuestadora»/«casa», never «empresa» (`assert_public_wording`).
- Luis commits in parallel in the same tree: before each commit run `git status` and `git log -1`, `git add` only the files of the task, never amend or rewrite his commits. Commit messages in Spanish, prefixed like the repo (`publish:`, `web:`), ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Test runner: `python -m pytest <file> -k <name> -q`; JS tests skip without `node`.

## Review Focus

1. A successor and its source both in today's `vs` (FA inherits SUMAR, and SUMAR is also listed in Izquierda 2026): the «vs 2023» of Izquierda must count Sumar's 2023 result once, not twice. → Task 4 test.
2. A run pinned with `?run=` that published four blocks: Promedios/Escaños render without error, the four blocks keep their colours, unknown names go last. → Tasks 2 and 5 tests.
3. A poll (or series date, or result) missing Izquierda or Derecha: Otros gets no value, never 100 − one block. → Task 5 test.
4. Projection where `hi` of I + D exceeds 100 or `lo` hits 0: Otros `lo` clipped at 0 and `hi` ≤ 100. → Task 1 test.
5. A regional scope where Otros is the largest block: the «Mayoría absoluta» KPI and the Inicio chip keep using `p_majority` of `vs` (they must not pick Otros). → Task 2 test.

---

### Task 1: The run publishes the three blocks

**Files:**
- Modify: `mtpy/lib/utils.py` (add `REST_BLOCK`, `web_blocks`)
- Modify: `mtpy/lib/publish/forecast.py` (`export_summary`, `export_projection`, the meta builder near line 562)
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Produces: `mtpy.lib.utils.REST_BLOCK = 'Otros'`; `web_blocks(vs: dict, names: list) -> Optional[dict]` returning `{'Izquierda': [...], 'Derecha': [...], 'Otros': [every name of names not in I∪D, in names order]}` (I/D keep the `vs` order of their members, filtered to `names`), or `None` when `vs` lacks `'Izquierda'` or `'Derecha'`.
- Produces: in forecast.py, `_blocks_summary_table(sim, bmap: dict) -> pd.DataFrame` (rows Izquierda, Derecha, Otros; same columns as `_summary_table`) and `_blocks_projection(sim, bmap: dict) -> pd.DataFrame` (same layout as `sim.projection`).
- Produces (published data): `summary.blocks` rows in order Izquierda, Derecha, Otros, Otros `p_majority` null; `projection.groups.blocks` with the same three names; `meta.bmaps.blocks` = the `web_blocks` map (members listed under Otros so Escaños can sum its seats per simulation). When `web_blocks` returns `None`, all three keep today's behaviour (`'blocks'` bmap of the model).

- [ ] **Step 1: Write the failing tests** in `tests/test_publish_unit.py`

```python
def test_web_blocks_puts_everything_outside_vs_in_otros():
    from mtpy.lib.utils import web_blocks
    vs = {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA']}
    assert web_blocks(vs, ['PP', 'PSOE', 'VOX', 'FA', 'PNV', 'ERC']) == {
        'Izquierda': ['PSOE', 'FA'], 'Derecha': ['PP', 'VOX'], 'Otros': ['PNV', 'ERC']}
    assert list(web_blocks(vs, ['PP', 'PSOE'])) == ['Izquierda', 'Derecha', 'Otros']
    assert web_blocks({'Derecha': ['PP']}, ['PP']) is None and web_blocks({}, ['PP']) is None


def test_export_summary_blocks_are_left_right_and_the_rest():
    sim = synthetic_simulator()
    sim.model.bmaps = {**sim.model.bmaps, 'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}
    data = valid('summary', publish.export_summary(sim)[0])
    rows = {r['name']: r for r in data['blocks']}
    assert [r['name'] for r in data['blocks']] == ['Izquierda', 'Derecha', 'Otros']
    assert rows['Otros']['p_majority'] is None
    for key in ('pct', 'pct_mean'):
        assert sum(r[key] for r in data['blocks']) == pytest.approx(100, abs=0.05)
    assert rows['Otros']['pct_lo'] < rows['Otros']['pct'] < rows['Otros']['pct_hi']   # a real range, residual included
    assert rows['Otros']['pct'] == pytest.approx(30.0, abs=0.01)   # VOX (15) + the residual '-' (15), not VOX alone
    assert sum(r['seats'] for r in data['blocks']) == 10
    assert set(data['p_majority']) == {'Derecha', 'Izquierda'}


def test_export_projection_otros_is_the_complement_of_left_and_right():
    sim = synthetic_simulator(mode='forecast')
    sim.model.bmaps = {**sim.model.bmaps, 'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}
    data = valid('projection', publish.export_projection(sim)[0], mode='forecast')
    blocks = data['groups']['blocks']
    assert blocks['names'] == ['Izquierda', 'Derecha', 'Otros']
    for i in range(len(data['dates'])):
        assert sum(blocks['mean'][n][i] for n in blocks['names']) == pytest.approx(100, abs=0.05)
        assert 0 <= blocks['lo']['Otros'][i] <= blocks['mean']['Otros'][i] <= blocks['hi']['Otros'][i] <= 100


def test_export_meta_writes_the_three_block_map():
    sim = synthetic_simulator()
    meta = publish.export_meta(sim, RUN, '2026-10-05T12:00:00Z', 50, 10, publish.DEFAULT_CORRECTORS, {}, {}, 0, None,
                               ...)  # the remaining arguments exactly as the existing call at line ~380
    assert meta['bmaps']['blocks'] == {'Izquierda': ['PSOE'], 'Derecha': ['PP', 'VOX'], 'Otros': []}
    assert meta['bmaps']['vs'] == BMAPS['vs'] and meta['bmaps']['main'] == BMAPS['main']


def test_export_summary_keeps_the_model_blocks_without_left_and_right():
    sim = synthetic_simulator()
    sim.model.bmaps = {**sim.model.bmaps, 'vs': {'Derecha': ['PP', 'VOX'], 'Nadie': ['UP']}}
    data = valid('summary', publish.export_summary(sim)[0])
    assert [r['name'] for r in data['blocks']] == list(BMAPS['blocks'])
```

In the meta test, the trailing `...` stands for the remaining positional arguments of `export_meta` (`prov`, `freeze`): copy them from the existing call at line ~380 of the same file.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_publish_unit.py -k "web_blocks or blocks_are_left or complement_of_left or three_block_map or keeps_the_model_blocks" -q`
Expected: FAIL (`ImportError: web_blocks`, wrong block names).

- [ ] **Step 3: Implement**

- `web_blocks` in `utils.py` next to `build_blocks`.
- `_blocks_summary_table(sim, bmap)`: call `_summary_table` with `{**bmap, UNION: I + D}` (a private key, e.g. `'__vs__'`), then set Otros `pct`, `pct_mean` = 100 − UNION's, `pct_lo` = 100 − UNION `pct_hi`, `pct_hi` = 100 − UNION `pct_lo` (the residual `'-'` is not in `sim.shares()` nor in `pct`, so the member sum would miss it), Otros `p_majority` = NA, drop UNION. Seats come from the member sum: with `split=True` every simulation sums the chamber, so that equals the complement.
- `_blocks_projection(sim, bmap)`: `sim.projection({'Izquierda': I, 'Derecha': D, UNION: I + D})`, then Otros `mean` = 100 − UNION `mean`, `lo` = clip(100 − UNION `hi`, 0), `hi` = 100 − UNION `lo`; drop UNION; columns in order Izquierda, Derecha, Otros. The band of I + D already carries the M7 correlation (`block_variance`).
- `export_summary` / `export_projection`: use them for the `'blocks'` group when `web_blocks(sim.model.bmaps.get('vs') or {}, sim.params['names'])` is not `None`; otherwise unchanged.
- Meta builder: `'bmaps': {**fc.bmaps, 'blocks': web}` when `web` is not `None`.

- [ ] **Step 4: Run the task's tests and the whole publish file**

Run: `python -m pytest tests/test_publish_unit.py -q`
Expected: all PASS. Existing tests that pinned four `blocks` or a `3 + 2 + 2` frame shape are updated to the new counts (`3 + 2 + 3`) — change the expected value only, never the meaning.

- [ ] **Step 5: Commit**

```bash
git status && git log -1 --oneline
git add mtpy/lib/utils.py mtpy/lib/publish/forecast.py tests/test_publish_unit.py
git commit -m "publish: los bloques de la run son Izquierda, Derecha y Otros (el resto hasta 100 y hasta la cámara)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Order, colour and «–» of the blocks in the pages

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`block_color`)
- Modify: `mtpy/lib/pages/promedios.py:18` (`BLOCK_ORDER`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `REST_BLOCK` from `mtpy.lib.utils` (Task 1).
- Produces: `promedios.BLOCK_ORDER = ('Izquierda', 'Derecha', REST_BLOCK)` (already imported by `elecciones.py` and `pages/__init__.py`); `block_color(REST_BLOCK, meta, parties) == OTHERS_COLOR` whatever the maps list.

- [ ] **Step 1: Write the failing tests**

```python
def test_block_color_paints_otros_grey_even_when_the_map_lists_parties():
    meta = {'bmaps': {'vs': {'Izquierda': ['PSOE'], 'Derecha': ['PP']},
                      'blocks': {'Izquierda': ['PSOE'], 'Derecha': ['PP'], 'Otros': ['PNV']}}}
    # build `parties` as the neighbouring block_color tests do
    assert block_color('Otros', meta, parties) == OTHERS_COLOR
    assert block_color('Izquierda', meta, parties) == parties['PSOE']['color']


def test_block_result_rows_put_left_right_and_the_rest_and_dash_the_majority_of_otros():
    summary = {'blocks': [{'name': 'Otros', 'pct': 10, 'seats': 30, 'p_majority': None},
                          {'name': 'Derecha', 'pct': 50, 'seats': 180, 'p_majority': 0.9},
                          {'name': 'Izquierda', 'pct': 40, 'seats': 140, 'p_majority': 0.1}]}
    rows = block_result_rows(summary, META, parties)          # META/parties as in the test at line ~887
    assert [r['name'] for r in rows] == ['Izquierda', 'Derecha', 'Otros']
    assert rows[2]['color'] == OTHERS_COLOR and rows[2]['p_majority'] is None


def test_block_result_rows_of_an_old_run_keep_its_four_blocks_last():
    names = [r['name'] for r in block_result_rows({'blocks': [
        {'name': n, 'pct': 1, 'seats': 1} for n in ('Derecha', 'Separatista', 'Izquierda', 'Regionalista')]}, META, parties)]
    assert names == ['Izquierda', 'Derecha', 'Separatista', 'Regionalista']
```

Plus, in the existing rendering test of Escaños (`test_escanos_table_histograms_blocks_and_pills`, line ~650) and of Promedios (`test_promedios_results_switch_between_parties_and_blocks_from_the_hemicycle_card`, line ~3012): make the fixture's summary carry the three blocks with Otros `p_majority: None` and assert the Otros row renders «–» in the P(mayoría) cell. In the Promedios KPI test (grep `Mayoría absoluta` in tests/test_pages.py), add a regional-like summary where Otros has the most seats and assert the KPI still names the `p_majority` block (a `vs` block).

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_pages.py -k "otros_grey or left_right_and_the_rest or old_run_keep" -q`
Expected: FAIL (order Derecha first; Otros coloured like PNV).

- [ ] **Step 3: Implement** — `block_color` returns `OTHERS_COLOR` first for `REST_BLOCK`; `BLOCK_ORDER = ('Izquierda', 'Derecha', REST_BLOCK)`. Update docstrings that describe the old order.

- [ ] **Step 4: Run the pages tests**

Run: `python -m pytest tests/test_pages.py -q`
Expected: all PASS; tests that pinned «Derecha first» or four-block names are updated to the new order.

- [ ] **Step 5: Commit** (same `git status`/`git log -1` check; add only `mtpy/lib/pages/common.py mtpy/lib/pages/promedios.py tests/test_pages.py`) — message `web: bloques en el orden Izquierda, Derecha, Otros y Otros siempre en gris`.

---

### Task 3: Past elections grouped with their `vs`

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py` (`party_blocks` ~650, `event_block_rows` ~686, `curve_block_colors` ~921, `run_less_colours` ~619, the detail context ~1120–1207, `last_results` ~229)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `REST_BLOCK`; `BLOCK_ORDER` (Task 2).
- Produces: `party_blocks(names: list, params: dict, event_blocks: Optional[dict]) -> dict` → `{name: 'Izquierda' | 'Derecha' | REST_BLOCK}` using `params['bmaps']['vs']` when it has both blocks, else `event_blocks` (the event meta `blocks`, which is its `vs`); every party outside them → `REST_BLOCK` (the `catalog` argument goes; update the three callers).
- Produces: `event_block_rows(rows, blocks, color)` → rows in `BLOCK_ORDER`; Izquierda/Derecha are member sums; Otros `pct` = round(100 − I − D, 2), `seats` = (sum of `seats` of all `rows`) − I − D; Otros `None` when I or D is `None`.
- Produces: the curve map passed to the browser (`initial.catalog.bmaps.blocks`) is `{'Izquierda': [...], 'Derecha': [...], 'Otros': []}` — Otros empty: the browser computes it as the rest (Task 5).

- [ ] **Step 1: Write the failing tests**

```python
def test_party_blocks_use_the_vs_of_the_election_and_send_the_rest_to_otros():
    params = {'bmaps': {'vs': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'SUMAR']}}}
    assert party_blocks(['PP', 'PSOE', 'PNV', 'UPN'], params, None) == {
        'PP': 'Derecha', 'PSOE': 'Izquierda', 'PNV': 'Otros', 'UPN': 'Otros'}
    assert party_blocks(['PP', 'ERC'], {}, {'Derecha': ['PP'], 'Izquierda': ['PSOE']}) == {'PP': 'Derecha', 'ERC': 'Otros'}


def test_event_block_rows_make_otros_the_rest_of_the_vote_and_the_chamber():
    rows = [{'party': 'PP', 'pct': 33.0, 'seats': 137}, {'party': 'PSOE', 'pct': 31.7, 'seats': 121},
            {'party': 'VOX', 'pct': 12.4, 'seats': 33}, {'party': 'PNV', 'pct': 1.1, 'seats': 5},
            {'party': 'X', 'pct': 0.3, 'seats': None}]
    blocks = {'PP': 'Derecha', 'VOX': 'Derecha', 'PSOE': 'Izquierda', 'PNV': 'Otros', 'X': 'Otros'}
    out = event_block_rows(rows, blocks, lambda b: '#000')
    assert [r['name'] for r in out] == ['Izquierda', 'Derecha', 'Otros']
    assert out[2]['pct'] == round(100 - 31.7 - 45.4, 2) and out[2]['seats'] == 296 - 121 - 170
```

Plus: update `test_last_results_sum_the_previous_blocks_with_the_catalogue` (line ~4246) to the new rule (three blocks from the election's `vs`, Otros by difference); update `test_curve_block_colors_follow_the_page_then_the_first_party` so an empty Otros list gets `OTHERS_COLOR` without indexing; update `test_run_less_colours_take_the_first_party_of_each_named_block` to the simplified contract (Izquierda/Derecha colours from the election `vs`, Otros grey); in an `events_site` detail test assert the initial data's `catalog.bmaps.blocks` is `{'Izquierda': [...], 'Derecha': [...], 'Otros': []}` and the results table by block lists the three blocks.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_pages.py -k "party_blocks_use_the_vs or rest_of_the_vote_and_the_chamber" -q`
Expected: FAIL.

- [ ] **Step 3: Implement** the interfaces above. `run_less_colours(meta, national, blocks_of)` may drop arguments it no longer needs (Otros is grey by `block_color`); update its caller.

- [ ] **Step 4: Run the pages tests**

Run: `python -m pytest tests/test_pages.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit** (status check; `mtpy/lib/pages/elecciones.py tests/test_pages.py`) — message `web: las elecciones pasadas agrupan su resultado con su vs; Otros es el resto`.

---

### Task 4: «vs {año}» of the blocks through the smap lineage

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`inherited_results` ~52; new `lineage_blocks`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `inherited_results` output (`parties[name]['inherited']`); `REST_BLOCK`; `meta['bmaps']['vs']` of the current run.
- Produces: `lineage_blocks(previous: dict, inherited: dict, vs: dict) -> dict` → `{'Izquierda': {'pct', 'seats'}, 'Derecha': {...}, 'Otros': {...}}`. `previous` = original `results['parties']`; `inherited` = the parties after `inherited_results`. Each previous party is assigned to the block of today's `vs` that lists it, or that lists the party which inherits it (`inherited[m]['inherited']`); each previous party counts **once**. Otros pct = round(100 − I − D, 2); seats = (sum of all previous seats) − I − D.
- Produces: `inherited_results(results, meta)` returns `results | {'parties': ..., 'blocks': lineage_blocks(...)}` when `meta.bmaps.vs` has both blocks; otherwise `blocks` untouched. `block_result_rows` and `seat_deltas` need no change.

- [ ] **Step 1: Write the failing tests**

```python
def test_lineage_blocks_follow_the_heirs_and_count_each_previous_party_once():
    previous = {'PP': {'pct': 33.0, 'seats': 136}, 'UPN': {'pct': 0.2, 'seats': 1}, 'VOX': {'pct': 12.4, 'seats': 33},
                'PSOE': {'pct': 31.7, 'seats': 121}, 'SUMAR': {'pct': 12.3, 'seats': 31}, 'PNV': {'pct': 1.1, 'seats': 5},
                'ERC': {'pct': 1.9, 'seats': 7}, 'BNG': {'pct': 0.6, 'seats': 1}, 'CC': {'pct': 0.5, 'seats': 1},
                'JxCat': {'pct': 1.6, 'seats': 7}, 'EHB': {'pct': 1.4, 'seats': 6}}
    inherited = dict(previous, PP={'pct': 33.2, 'seats': 137, 'inherited': ['PP', 'UPN']},
                     FA={'pct': 12.3, 'seats': 31, 'inherited': ['SUMAR']})
    vs = {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'SUMAR', 'FA', 'UP']}
    out = lineage_blocks(previous, inherited, vs)
    assert out['Derecha'] == {'pct': 45.6, 'seats': 170}          # UPN's seat is Derecha's already
    assert out['Izquierda'] == {'pct': 44.0, 'seats': 152}        # Sumar counted once, not twice through FA
    assert out['Otros']['seats'] == 349 - 170 - 152 and out['Otros']['pct'] == round(100 - 45.6 - 44.0, 2)


def test_inherited_results_replace_the_blocks_with_the_lineage_of_the_run_vs():
    results = {'date': '2023-07-23', 'year': 2023,
               'parties': {'PP': {'seats': 137, 'pct': 33.06}, 'UPN': {'seats': 1, 'pct': 0.21},
                           'SUMAR': {'seats': 31, 'pct': 12.33}, 'PNV': {'seats': 5, 'pct': 1.12}},
               'blocks': {'Derecha': {'seats': 137, 'pct': 33.06}}}
    meta = {'parties': [{'name': 'PP'}, {'name': 'FA'}],
            'smap': {'FA': [{'type': 'agg', 'names': ['SUMAR']}], 'PP': {'type': 'agg', 'names': ['UPN']}},
            'bmaps': {'vs': {'Derecha': ['PP'], 'Izquierda': ['FA']}}}
    out = pages.inherited_results(results, meta)
    assert out['blocks'] == {'Izquierda': {'pct': 12.33, 'seats': 31}, 'Derecha': {'pct': 33.27, 'seats': 138},
                             'Otros': {'pct': round(100 - 12.33 - 33.27, 2), 'seats': 174 - 31 - 138}}
    assert pages.inherited_results(results, {**meta, 'bmaps': {}})['blocks'] == results['blocks']
```

The existing `test_inherited_results_sum_the_parties_a_run_party_absorbs` (no `bmaps` in its meta) keeps asserting `out['blocks'] == results['blocks']`.

- [ ] **Step 2: Run them to see them fail** — `python -m pytest tests/test_pages.py -k "lineage_blocks or replace_the_blocks_with_the_lineage" -q` → FAIL (`NameError: lineage_blocks`).

- [ ] **Step 3: Implement** `lineage_blocks` and the `inherited_results` change.

- [ ] **Step 4: Run** `python -m pytest tests/test_pages.py -q` → all PASS (Promedios/Escaños deltas of blocks in existing fixtures change to the lineage values; update expectations, not meanings).

- [ ] **Step 5: Commit** (status check; `mtpy/lib/pages/promedios.py tests/test_pages.py`) — message `web: el «vs año» de los bloques sigue el linaje del smap, como los partidos`.

---

### Task 5: The browser computes Otros as the rest

**Files:**
- Modify: `web/dist/js/blocks.js` (`blockSeries`, `blockPolls`, `blockResult`; export `REST_BLOCK`)
- Modify: `web/dist/js/catalog.js` (`orderBlocks`, `blockColor`)
- Test: `tests/test_js_blocks.py`, `tests/test_js_charts.py` (the catalog tests)

**Interfaces:**
- Produces: `export const REST_BLOCK = 'Otros'` in `blocks.js`; `catalog.js` imports it and adds `export const BLOCK_VIEW_ORDER = ['Izquierda', REST_BLOCK, 'Derecha']`.
- Produces: in a block map that has `REST_BLOCK`, its value is 100 − the sum of the **other** blocks of the map, whatever members it lists, and only when every other block has a value (poll, series date, result); in the series its band half-width is `sqrt(Σ half_b²)` over the other blocks. A map without `REST_BLOCK` (old runs) behaves exactly as today.
- Produces: `catalog.orderBlocks` sorts by `BLOCK_VIEW_ORDER`, unknown names last in input order; `catalog.blockColor(REST_BLOCK) === OTHERS_COLOR`. `BLOCK_ORDER` and `catalog.order` (parties) unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_js_blocks.py`

```python
REST = {'Izquierda': ['PSOE'], 'Derecha': ['PP', 'VOX'], 'Otros': ['PNV']}


def test_rest_block_series_is_the_complement_with_the_band_of_the_others():
    out = run_js('b.blockSeries(series, %s)' % json.dumps(REST))
    assert out['mean']['Otros'] == [100 - 28 - 40, 100 - 28 - 31]
    half = (2 ** 2 + 2 ** 2 + 2 ** 2) ** 0.5          # PSOE, PP, VOX on day one
    assert out['lo']['Otros'][0] == pytest.approx(32 - half) and out['hi']['Otros'][0] == pytest.approx(32 + half)


def test_rest_block_needs_every_other_block():
    out = run_js('b.blockPolls(polls, %s)' % json.dumps(REST))
    assert out['polls'][0]['Otros'] == 100 - 28 - 40
    assert 'Otros' not in out['polls'][1]                # poll B has no Derecha
    res = run_js('b.blockResult({date: "d", values: {PP: 33, VOX: 12, PSOE: 32, PNV: 1}}, %s)' % json.dumps(REST))
    assert res['values'] == {'Izquierda': 32, 'Derecha': 45, 'Otros': 23}
    assert 'Otros' not in run_js('b.blockResult({date: "d", values: {PSOE: 32}}, %s)' % json.dumps(REST))['values']
```

And in the catalog tests (`tests/test_js_charts.py`, next to the other `Catalog` checks):

```python
def test_catalog_orders_the_block_view_left_rest_right_and_paints_otros_grey():
    cat = 'new c.Catalog({parties: [{name: "PNV", color: "#00ff00"}], bmaps: {blocks: {Otros: ["PNV"]}}})'
    assert run_js('%s.orderBlocks(["Derecha", "Otros", "Izquierda", "Separatista"])' % cat) == [
        'Izquierda', 'Otros', 'Derecha', 'Separatista']
    assert run_js('%s.blockColor("Otros")' % cat) == '#9e9e9e'
```

(Use the `run_js` helper that file already has for `CATALOG_JS`.)

- [ ] **Step 2: Run them to see them fail** — `python -m pytest tests/test_js_blocks.py tests/test_js_charts.py -k "rest_block or left_rest_right" -q` → FAIL.

- [ ] **Step 3: Implement** the interfaces above; update the header comments of both files.

- [ ] **Step 4: Run** `python -m pytest tests/test_js_blocks.py tests/test_js_charts.py tests/test_js_seats.py -q` → all PASS (the existing test with a block literally named `Otros` in `test_block_series_without_any_party_of_the_block_is_null` now has a rest block: rename that block in the test to `Nadie`, keeping its meaning).

- [ ] **Step 5: Commit** (status check; the two JS files and two test files) — message `web: Otros es el resto hasta 100 en sondeos, serie y resultado; hemiciclo Izquierda, Otros, Derecha`.

---

### Task 6: Public texts and the full suite

**Files:**
- Modify: `web/templates/escanos.html:59`, `web/templates/promedios.html:56,89`, `web/templates/eleccion.html:52,168` and any other hit of the grep below
- Test: `tests/test_pages.py` (rendering), the whole suite

- [ ] **Step 1: Find every public mention of the old blocks**

Run: `grep -rn "Regionalista\|Separatista\|Bloques ideológicos\|cada bloque suma sus partidos" web/templates mtpy/lib/pages`
Expected: the five template lines above plus whatever else mentions them; leave code constants for parties (`HEMICYCLE_BLOCKS`) alone.

- [ ] **Step 2: Rewrite the copy** — exact texts:
  - Escaños caption: «Izquierda y Derecha, con todos sus partidos; Otros es el resto de la cámara.»
  - Promedios / election detail, after «En la vista Bloques, cada bloque suma sus partidos»: append « y Otros es lo que falta hasta 100.»
  - Promedios/eleccion «Con la vista Bloques, cada sector es un bloque…»: keep, it stays true.
  Add a rendering assert for the Escaños caption to the existing Escaños page test.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all PASS (≈1400 tests; JS tests need `node`). `assert_public_wording` passes.

- [ ] **Step 4: Commit** (status check; templates + tests) — message `web: textos de la vista Bloques con Izquierda, Derecha y Otros`.

- [ ] **Step 5: Hand-off note for Luis** (in the final report, no code): publish today's run of `es` and of the regional scopes with forecast (`publish forecast`) to see the three blocks in Promedios and Escaños; past elections change on deploy; no event or backtest republication needed.
