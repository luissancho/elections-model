# Question 3 of the CERA study on the model's forecast for 29-N — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The card `#p3` of `/analisis/voto-cera` answers «¿Puede el voto CERA arrebatar la mayoría absoluta a la derecha el 29-N?» with figures of the model: for every scenario of exterior vote, the seats it takes from the right starting from the central allocation and the probability that the right keeps its majority in the 10,000 simulations.

**Architecture:** A job action (`load majority`) rebuilds the published run of 23-N through the same code `publish_forecast` uses, refuses unless the rebuilt run equals the published `dist` and `scenario`, applies every scenario to the simulations and to the central allocation with a paired, vectorised D'Hondt, and writes a frozen snapshot, `data/cera/majority.json`. `compute_cera` carries that file, `publish study` exports it as a new part `majority` (CSV twins and, after 29-N, `outcome`), and the page builds the whole card from that part, in HTML and CSS.

**Tech Stack:** Python 3.11 (numpy, pandas), Jinja2 pages of `mtpy/lib/pages`, one stylesheet (`web/dist/css/site.css`), plain ES modules (`web/dist/js`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-web-analisis-voto-cera-mayoria-design.md` (mockup, tenth version: `docs/superpowers/specs/2026-11-24-web-analisis-voto-cera-mayoria-mockup.html`). Its «Cifras de la prueba» are the figures every task checks against.

## Global Constraints

- All code inside `mtpy` (lib, pipelines, jobs): no loose scripts, no FastAPI. PEP8, lines ≤ 120, a numpydoc docstring on every function, identifiers and comments in English; Spanish only in public texts and in test docstrings.
- Public texts: «sólo» with its accent, «encuestadora» or «casa», never «empresa»; the page passes `assert_public_wording`. The copy of the card is the mockup's, verbatim; every figure in it comes from the part, never typed.
- Not touched: questions 1 and 2, «Lo que dice el debate», «Comprobación», the seven parts the study publishes today (JSON and CSV, byte for byte) and the registration (`registered_at`, `registered_commit`, `registered`).
- `compute_cera` and `publish study` never touch the database. Only the action of Task 6 does, and only to read.
- The snapshot is taken **before 2026-11-29**, locally, on the run `20261124-102045` of `es` (data up to 2026-11-23, 10,000 simulations, seed 42). Tasks 1 to 8 come first for that reason. Nothing is added to its axes afterwards.
- Blocks and majority are the run's (`meta.bmaps.blocks`: `Derecha` = PP, VOX, SALF; `Izquierda` = PSOE, FA, UP; `meta.majority` = 176), never the study's 2023 blocks, which are used only inside `_biased_split`.
- Tests never touch the network or a database. Run them with `python -m pytest` (pyenv 3.11.9); `pytest.ini` turns `FutureWarning` and pandas `PerformanceWarning` into errors. `compute_cera` takes 15 s: the new test modules share one session-scoped study (Task 4), they do not compute their own.
- Luis works and launches jobs in this same tree: before touching a shared file or committing, look at `git status`, `git log` and the running jobs. Commit only the files of the task with `git add <files>`, never rewrite his commits, never push. Messages in Spanish in the style of the log, ending with the `Co-Authored-By` line. Deploying and publishing are his.

## Review Focus

- The exact tie at the last seat, and a province where no party reaches the threshold: the vectorised allocation gives what `Simulator.alloc_seats` gives, row by row. (Task 3, `test_allocate_equals_the_exact_allocation_with_ties_and_threshold`)
- A scenario that takes more votes from a party than it has in a simulation (the whole-census grid removes the 2023 exterior vote first): its votes stop at zero and the threshold base moves by the net sum. (Task 3, `test_a_delta_that_takes_more_votes_than_there_are_stops_at_zero`, `test_the_threshold_base_moves_with_the_net_votes_of_the_scenario`)
- A run that is not the published one (a poll corrected or loaded late, another seed, another `n_sim`), or a scope with a threshold over the whole scope: the action refuses and writes nothing. (Task 6, `test_take_snapshot_refuses_a_run_that_differs_in_one_simulation`, `test_take_snapshot_refuses_a_scope_threshold`)
- The study published without the snapshot, or with the snapshot of another election; and the page of a bundle that has no `majority` part yet (deployed before publishing): a message that says what to do, and a page that still renders. (Task 5, `test_read_majority_refuses_a_snapshot_that_is_not_this_study_s`; Task 10, `test_export_study_majority_fails_clearly_without_the_snapshot`; Task 12, `test_cera_page_without_the_majority_part_keeps_rendering`)
- After 29-N, an official sigla that is not the model's name (a coalition, another spelling): the study refuses to publish and names the party, instead of counting the majority without it. (Task 9, `test_majority_outcome_refuses_a_block_party_missing_from_the_results`)

## File Structure

| File | Responsibility |
|---|---|
| `mtpy/lib/simulator.py` | new `Simulator.central_shares()`: the vote behind `central_result()` |
| `mtpy/lib/publish/forecast.py` | `fitted_simulator`, `run_mode`: the recipe of the run, shared with the action |
| `mtpy/lib/studies/cera/majority.py` (new) | pure arrays: `Draws`, vectorised D'Hondt, the paired measure, `seat_range` |
| `mtpy/lib/studies/cera/stress.py` | `exterior_votes` and `placement_plan`, extracted from what exists |
| `mtpy/lib/studies/cera/scenarios.py` (new) | from a `CeraStudy` to the votes of each scenario, and `build_snapshot` |
| `mtpy/lib/studies/cera/data.py`, `study.py` | `read_majority`, `check_majority`; `CeraStudy.majority` and `.outcome` |
| `mtpy/lib/studies/cera/snapshot.py` (new) | the only module that touches database and bundle: `take_snapshot` |
| `mtpy/pipelines/load/Majority.py` (new), `mtpy/jobs/Load.py`, `deploy/ecs/job.py` | the action `load majority`, local only |
| `data/cera/majority.json` (new), `data/cera/params.json` | the snapshot; `majority.aliases` |
| `mtpy/lib/studies/cera/check.py` | `majority_outcome` |
| `mtpy/lib/bundle.py`, `mtpy/lib/publish/studies.py`, `docs/web/contrato.md` | the part `majority`, its schema and CSV twins |
| `mtpy/lib/pages/cera.py`, `analisis.py`, `web/templates/analisis_voto_cera.html`, `web/dist/css/site.css`, `web/dist/js/pages/analisis-cera.js` | the card |

## The snapshot (`data/cera/majority.json`)

Every task that writes or reads it uses these keys. Counts are integers over `run.n_sim`; no rounded probability is stored.

```
run          {scope, event_date, run_id, as_of, n_sim, seed, commit, taken_on}
majority     176
blocks       {"Derecha": [...], "Izquierda": [...]}          the run's
lineage      {sigla of 2023: party of the model or "-"}      as applied
base         {with_majority, central: {"Derecha", "Izquierda"}, right_seats: {"170": n, ...}}
expected     {"convocatoria" | "sin_litigio": {table, counts, provinces}}
               table      {census: [2023, 29-N], new_electors, voters: [2023, 29-N], votes: [2023, 29-N]}
               provinces  [{region_id, name, moved, right_lost, right_gained}]   one row per province
new_electors {participations, left_fracs, electors, cells}   cells[i][j] = counts
grid         {participations, left_fracs, electors, cells}
placement    {participations, fracs, pool, cells}
```

`counts` of one scenario: `{votes, lost, gained, moved, right: {"-2": n, "-1": n, "0": n, "1": n}, left: {...}, central: {right, left}}`. `lost`/`gained`: simulations in which the right loses/gets the majority; `moved`: simulations with some seat changed; `right`/`left`: distribution of the block's change of seats; `central`: its change in the central allocation. A probability is always `round(100 * count / n_sim, decimals)`.

---

### Task 1: `Simulator.central_shares()`

**Files:**
- Modify: `mtpy/lib/simulator.py` (`central_result`, :1056)
- Test: `tests/test_simulator_unit.py` (next to the `central_result` tests, :721)

**Interfaces:**
- Produces: `Simulator.central_shares(self) -> Optional[pd.DataFrame]`: the `vpred_pct` of the deterministic pass behind `central_result()`, index `params['regions']` (the total included), columns `categories()` (the parties and `'-'`), floats. `None` when `central_result()` is `None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_central_shares_are_the_vote_behind_the_central_allocation():
    sim = synthetic_simulator()
    shares, central = sim.central_shares(), sim.central_result()
    assert list(shares.index) == REGIONS and list(shares.columns) == NAMES + ['-']
    for loc, region in enumerate(REGIONS[1:], start=1):
        valid = float(sim.prev_totals['votes'].loc[region])
        votes = np.round(shares.loc[region, NAMES].to_numpy() * valid / 100)
        seats = Simulator.alloc_seats(dict(zip(NAMES, votes.tolist())), int(sim.reg_totals['seats'].loc[region]),
                                      valid_votes=valid, threshold=sim.threshold)
        assert [seats[name] for name in NAMES] == central.iloc[loc].tolist()
    flat = synthetic_simulator()
    flat.params['split'] = False
    assert flat.central_shares() is None


def test_central_shares_leave_the_simulations_and_the_generators_alone():
    a, b = synthetic_simulator(), synthetic_simulator()
    a.params['n_sim'] = b.params['n_sim'] = 5
    a.run(reset=False)
    b.central_shares()
    b.run(reset=False)
    assert np.array_equal(a.results, b.results) and b.params['random'] is True
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_simulator_unit.py -q -k central_shares`
Expected: FAIL, `'Simulator' object has no attribute 'central_shares'`

- [ ] **Step 3: Implement `central_shares`**, taking the deterministic `build_frame` → `build_umat` (with `random` off and restored) out of `central_result` into a private `_central_pass(self) -> tuple` (frame, umat) that both use. No new cached attribute: `central_shares` runs the pass each time.

- [ ] **Step 4: Run the whole file** (the `central_result` tests at :721-771 pin what must not change)

Run: `python -m pytest tests/test_simulator_unit.py -q`
Expected: PASS

- [ ] **Step 5: Commit** (`mtpy/lib/simulator.py`, `tests/test_simulator_unit.py`)

### Task 2: the recipe of the run, shared

**Files:**
- Modify: `mtpy/lib/publish/forecast.py` (`publish_forecast`, :998-1017), `mtpy/lib/publish/__init__.py` (exports)
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Produces: `fitted_simulator(scope: str, event_date: str, drange, alpha: float, seed: int, max_fc: int, correctors: dict, verbose: int = 0, today: Optional[str] = None, limit_date: Optional[str] = None, simulator: Optional[Callable] = None, step: Optional[Callable[[str], None]] = None) -> Simulator`: builds the simulator in `nowcast` mode and fits it with exactly the arguments `publish_forecast` passes today (`limit_date` only when it is not `None`); calls `step('init')` after building and `step('fit')` after fitting; a `ValueError` of either phase becomes `NothingToPublish`.
- Produces: `run_mode(sim, mode: str, n_sim: int) -> None`: `'nowcast'` → `sim.run(split=True, random=True, n_sim=n_sim)`; `'forecast'` → `sim.mode = 'forecast'`, then `sim.run(split=True, random=True, n_sim=n_sim, horizon='deadline')`.

- [ ] **Step 1: Write the failing test**

```python
def test_fitted_simulator_and_run_mode_are_the_recipe_of_publish_forecast():
    factory, steps = FakeSimulator(), []
    sim = publish.fitted_simulator('es', '2026-11-29', drange=6, alpha=0.05, seed=42, max_fc=10,
                                   correctors=dict(publish.DEFAULT_CORRECTORS), today='2026-11-23',
                                   simulator=factory, step=steps.append)
    publish.run_mode(sim, 'forecast', 50)
    assert steps == ['init', 'fit'] and [call[0] for call in factory.calls] == ['init', 'fit_forecast', 'run']
    init = factory.calls[0][3]
    assert init['today'] == '2026-11-23' and init['mode'] == 'nowcast' and init['house_effects'] is True
    assert 'limit_date' not in init and factory.calls[1][1] == {'names': NAMES, 'max_fc': 10, 'fillna': True}
    assert factory.calls[2][1:] == ('forecast', {'split': True, 'random': True, 'n_sim': 50, 'horizon': 'deadline'})
```

- [ ] **Step 2: Run it to see it fail** (`python -m pytest tests/test_publish_unit.py -q -k recipe`; `module has no attribute 'fitted_simulator'`)

- [ ] **Step 3: Implement both and make `publish_forecast` use them**, without changing what it reports (the `init`, `fit`, `nowcast`, `forecast` laps and their seconds) or raises.

- [ ] **Step 4: Run the publish tests** (`python -m pytest tests/test_publish_unit.py -q`; PASS, with `test_publish_forecast_writes_the_whole_layout_in_order` untouched)

- [ ] **Step 5: Commit**

### Task 3: the paired measure (`majority.py`)

**Files:**
- Create: `mtpy/lib/studies/cera/majority.py` (imports numpy, pandas and `Simulator`, as `seats.py` does; nothing else of the project)
- Test: `tests/test_cera_majority.py` (new)

**Interfaces:**
- Produces: `RIGHT, LEFT, REST = 'Derecha', 'Izquierda', '-'`; `LINEAGE = {'SUMAR': 'FA', 'UPN': 'PP'}`.
- Produces: `@dataclass Draws`: `names: list` (parties, simulator order), `regions: list` (ids, the total included), `default_region: int`, `votes: np.ndarray` (simulations × regions × parties, the rounded votes the simulator allocated), `central: np.ndarray` (regions × parties), `base: np.ndarray` (regions; valid votes, the threshold base), `seats: np.ndarray` (regions), `threshold: Optional[float]`, `results: np.ndarray` (simulations × regions × parties, the stored seats), `central_seats: np.ndarray` (regions × parties).
- Produces: `allocate(votes: np.ndarray, n_seats: int, base: float, threshold: Optional[float], names: list) -> np.ndarray` (draws × parties → integer seats).
- Produces: `seats_with(draws: Draws, delta: np.ndarray) -> np.ndarray` (simulations × regions × parties) and `central_with(draws: Draws, delta: np.ndarray) -> np.ndarray` (regions × parties). `delta` is regions × (parties + 1), the last column the residual; its row of the total is ignored, and in what both return the row of the total is the sum of the others.
- Produces: `to_model(frame: pd.DataFrame, siglas: dict, names: list, regions: list, lineage: dict) -> np.ndarray` (votes by `region_id` and 2023 candidature → `delta`; a region of `regions` the frame does not have gets zeros), and `applied_lineage(siglas: dict, names: list, lineage: dict) -> dict` (sigla → party or `'-'`).
- Produces: `measure(draws: Draws, delta: np.ndarray, blocks: dict, majority: int) -> dict` (the `counts` above, plain Python ints), `province_counts(draws, delta, blocks) -> list` (`[{region_id, moved, right_lost, right_gained}]`, the total left out), `base_counts(draws, blocks, majority) -> dict` (the `base` above), `reproduction_problems(draws: Draws) -> list` (at most two texts, one for the simulations, «n of N simulations differ», and one for the central allocation; empty when a zero `delta` gives `results` and `central_seats`).
- Produces: `seat_range(dist: dict, alpha: float = 0.05) -> list`: `[lo, hi]`, the quantiles `alpha / 2` and `1 − alpha / 2` of the distribution expanded (`np.quantile`, linear: the count of `Simulator.quantile`), as floats.

Rules the tests pin (spec D1): votes with a scenario are `max(votes + delta, 0)`, `delta` unrounded; the threshold base of a province is `base + delta.sum()` over its row, residual included, before the cut at zero; a candidature whose sigla is `None` or, after `lineage`, is not a party of the model goes to the residual, which counts for the threshold and never gets seats.

- [ ] **Step 1: Write the failing tests**

```python
NAMES, BLOCKS = ['R', 'L'], {'Derecha': ['R'], 'Izquierda': ['L']}


def tiny():
    """Four simulations, two provinces (3 and 2 seats), no threshold; region 0 is the total."""
    one = np.array([[310., 100.], [250., 150.], [210., 190.], [150., 250.]])
    votes = np.stack([np.zeros((4, 2)), one, np.tile([100., 90.], (4, 1))], axis=1)
    draws = Draws(names=NAMES, regions=[0, 1, 2], default_region=0, votes=votes,
                  central=np.array([[0., 0.], [250., 150.], [100., 90.]]), base=np.array([0., 400., 190.]),
                  seats=np.array([5, 3, 2]), threshold=None, results=None, central_seats=None)
    zero = np.zeros((3, 3))
    draws.results, draws.central_seats = seats_with(draws, zero), central_with(draws, zero)
    return draws


def test_measure_counts_what_a_hand_count_gives():
    draws = tiny()
    assert draws.results[:, 0, 0].tolist() == [4, 3, 3, 2] and draws.central_seats[0].tolist() == [3, 2]
    delta = np.array([[0., 0., 0.], [0., 30., 0.], [100., 0., 0.]])
    assert measure(draws, delta, BLOCKS, majority=3) == {
        'votes': 130, 'lost': 0, 'gained': 1, 'moved': 4, 'right': {'0': 2, '1': 2}, 'left': {'-1': 2, '0': 2},
        'central': {'right': 1, 'left': -1}}
    assert province_counts(draws, delta, BLOCKS) == [
        {'region_id': 1, 'moved': 2, 'right_lost': 2, 'right_gained': 0},
        {'region_id': 2, 'moved': 4, 'right_lost': 0, 'right_gained': 4}]
    assert base_counts(draws, BLOCKS, majority=3) == {
        'with_majority': 3, 'central': {'Derecha': 3, 'Izquierda': 2}, 'right_seats': {'2': 1, '3': 2, '4': 1}}


def test_a_zero_delta_changes_nothing_and_reproduces_the_stored_seats():
    draws = tiny()
    assert measure(draws, np.zeros((3, 3)), BLOCKS, 3)['moved'] == 0 and reproduction_problems(draws) == []
    draws.results = draws.results.copy()
    draws.results[2, 1] = [3, 0]
    problems = reproduction_problems(draws)
    assert len(problems) == 1 and '1 of 4' in problems[0]
    draws.central_seats = draws.central_seats[:, ::-1]
    assert len(reproduction_problems(draws)) == 2


def test_allocate_equals_the_exact_allocation_with_ties_and_threshold():
    rng, names = np.random.default_rng(7), ['A', 'B', 'C', 'D', 'E']
    votes = np.round(rng.dirichlet([8, 6, 3, 1, 0.3], size=300) * 10000)
    votes[:4] = [[600, 300, 100, 0, 0], [500, 500, 0, 0, 0], [0, 0, 0, 0, 0], [20, 20, 20, 20, 20]]
    for n_seats, threshold in ((7, 3.0), (3, 3.0), (2, 25.0), (5, None)):
        got = allocate(votes, n_seats, 10500., threshold, names)
        for row, seats in zip(votes, got):
            exact = Simulator.alloc_seats(dict(zip(names, row.tolist())), n_seats, valid_votes=10500.,
                                          threshold=threshold)
            assert seats.tolist() == [exact[name] for name in names]


def test_a_delta_that_takes_more_votes_than_there_are_stops_at_zero():
    draws = tiny()
    seats = seats_with(draws, np.array([[0., 0., 0.], [0., -500., 50.], [0., 0., 0.]]))
    assert (seats[:, 1] == [3, 0]).all() and (seats >= 0).all()


def test_the_threshold_base_moves_with_the_net_votes_of_the_scenario():
    votes = np.array([[[0., 0., 0.], [500., 260., 240.]]])
    draws = Draws(names=['A', 'B', 'C'], regions=[0, 1], default_region=0, votes=votes, central=votes[0],
                  base=np.array([0., 1000.]), seats=np.array([4, 4]), threshold=25.0, results=None,
                  central_seats=None)
    assert seats_with(draws, np.zeros((2, 4)))[0, 1].tolist() == [3, 1, 0]  # C has 24 % of 1000: out
    less = np.array([[0., 0., 0., 0.], [0., 0., 0., -50.]])  # 50 residual votes less: C has 25.3 % of 950
    assert seats_with(draws, less)[0, 1].tolist() == [2, 1, 1] and central_with(draws, less)[1].tolist() == [2, 1, 1]


def test_to_model_follows_the_lineage_and_sends_the_rest_to_the_residual():
    frame = pd.DataFrame({'Sumar': [10., 1.], 'UPN': [5., 0.], 'Zeta': [2., 2.], 'Nadie': [1., 0.]}, index=[28, 8])
    siglas = {'Sumar': 'SUMAR', 'UPN': 'UPN', 'Zeta': 'ZZ', 'Nadie': None}
    assert to_model(frame, siglas, ['PP', 'FA', 'UP'], [0, 28, 8, 31], LINEAGE).tolist() == [
        [0., 0., 0., 0.], [5., 10., 0., 3.], [0., 1., 0., 2.], [0., 0., 0., 0.]]
    assert applied_lineage(siglas, ['PP', 'FA', 'UP'], LINEAGE) == {'SUMAR': 'FA', 'UPN': 'PP', 'ZZ': '-'}


def test_seat_range_is_the_95_per_cent_of_the_stored_distribution():
    assert seat_range({'-2': 8, '-1': 367, '0': 9429, '1': 196}) == [-1.0, 0.0]
    assert seat_range({'0': 10}) == [0.0, 0.0]
```

- [ ] **Step 2: Run them to see them fail** (`python -m pytest tests/test_cera_majority.py -q`; `ModuleNotFoundError`)

- [ ] **Step 3: Implement the module.** `allocate` builds the quotients of every draw at once and takes the `n_seats` largest; a draw whose cut is tied, or where nothing is eligible, is allocated with `Simulator.alloc_seats`, which is the reference for ties (it gives the seat to the first party in order). The version of this algorithm that reproduced the 10,000 simulations is `alloc_region` in `files/stage/cera-majority-spike/scenarios.py` (outside the repository).

- [ ] **Step 4: Run the tests to see them pass**

- [ ] **Step 5: Commit**

### Task 4: the votes of each scenario (`stress.py`, `scenarios.py`)

**Files:**
- Modify: `mtpy/lib/studies/cera/stress.py` (`exterior_grid` :135, `_greedy_seats` :318, `optimal_placement` :359), `mtpy/lib/studies/cera/study.py` (`_seats_block` :536), `tests/conftest.py`
- Create: `mtpy/lib/studies/cera/scenarios.py`, `tests/cera_doubles.py`
- Test: `tests/test_cera_scenarios.py` (new)

**Interfaces:**
- Consumes: Task 3 (`Draws`, `to_model`, `measure`, `province_counts`, `base_counts`, `applied_lineage`, `LINEAGE`).
- Produces (`stress.py`): `exterior_votes(cera_2023: pd.DataFrame, results: Results, census_by_province: pd.Series, participation: float, left_frac: float, siglas: dict, blocks: dict) -> pd.DataFrame`, the frame `exterior_grid` adds today to the peninsular results (`_biased_split(...)` times `electors * participation * VALID_SHARE`), which `exterior_grid` now calls; `placement_plan(results: Results, pool: pd.Series, participation: float, frac: float, candidature: str, step: int = 500) -> pd.Series`, the votes placed by `region_id`. `_greedy_seats` returns `(seats, placed)`; `optimal_placement` returns what it returns today.
- Produces (`study.py`): `CeraStudy.placement` gains `electors`, the pool by `region_id` (`pd.Series`); `pool` stays the integer. The exporters read `placement` key by key, so nothing published changes.
- Produces (`scenarios.py`), all taking the `CeraStudy` and returning votes by `region_id` and 2023 candidature unless said:
  - `expected_votes(study, name: str) -> pd.DataFrame`: `study.scenarios[name]['extra']`.
  - `expected_table(study, name: str) -> dict`: `{census: [2023, 29-N], new_electors, voters: [2023, 29-N], votes: [2023, 29-N]}`, the figures `publish.studies._scenario` computes (:311), with 2023's census as the scenario's minus its new electors.
  - `new_elector_votes(study, participation: float, left_frac: float) -> pd.DataFrame`: `increment_votes(study.scenarios['convocatoria']['new'], {'latam': participation, 'rest': participation}, _biased_split(study.split, left_frac, siglas, study.blocks), valid=study.params['participation']['valid'])`.
  - `exterior_change(study, participation: float, left_frac: float) -> pd.DataFrame`: `exterior_votes(...)` minus the 2023 exterior vote, `study.cera[study.params['baseline_scrutiny']]`.
  - `placement_change(study, participation: float, frac: float) -> pd.DataFrame`: `placement_plan(...)` of `study.placement['electors']` as the only column, `PLACEMENT_CANDIDATURE`.
  - `build_snapshot(draws: Draws, study, meta: dict) -> dict`: the snapshot above without `run.commit` and `run.taken_on`, plain Python types. `meta` is the published `meta` of the run (`scope`, `event_date`, `run_id`, `as_of`, `n_sim`, `seed`, `majority`, `bmaps.blocks`). Axes: `GRID_PARTICIPATIONS` × `GRID_LEFT_FRACS` for `new_electors` and `grid`, `PLACEMENT_PARTICIPATIONS` × `PLACEMENT_FRACS` for `placement`. `siglas` are those of the baseline results (`study.results[study.params['election']['baseline']].siglas`); province names, `study.province_names`.
- Produces (tests): the session fixture `cera_study` in `tests/conftest.py` (`compute_cera(FileSystem(<repo>/data))`, imported inside the fixture); `tests/cera_doubles.py` with `META` and `double_draws(study)`.

`double_draws(study)`: a `Draws` over the total (region 0) and the 52 provinces of the baseline results, parties `['PP', 'PSOE', 'VOX', 'FA', 'UP', 'SALF']`; `central` is `to_model` of the official 2023 votes without its residual column, each of 20 simulations is `central` times a seeded log-normal noise (σ = 0.05), rounded; `base` the valid votes (candidatures plus blank), `seats` the official magnitudes, `threshold` 3.0, and `results`/`central_seats` from `seats_with`/`central_with` with a zero `delta`. `META = {'scope': 'es', 'event_date': '2026-11-29', 'run_id': '20261124-102045', 'as_of': '2026-11-23', 'n_sim': 20, 'seed': 42, 'majority': 176, 'bmaps': {'blocks': {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA', 'UP']}}}`.

- [ ] **Step 1: Keep what the study publishes today.** From the session scratchpad (no file in the repo), write the JSON (`sort_keys`) and every CSV of the seven `export_study_*` of `compute_cera(FileSystem('data'))` to a folder. Task 10 compares against it.

- [ ] **Step 2: Write the failing tests**

```python
def test_the_votes_of_each_scenario_add_up_to_what_the_spec_says(cera_study):
    study = cera_study
    assert round(expected_votes(study, 'convocatoria').values.sum()) == 30482
    assert round(expected_votes(study, 'sin_litigio').values.sum()) == 22412
    table = expected_table(study, 'convocatoria')
    assert table == {'census': [2327388, 2769645], 'new_electors': 442257, 'voters': [208431, 239221],
                     'votes': [203152, 233634]}
    published = publish.export_study_prediction(study)[0]['scenarios']['sin_litigio']
    assert expected_table(study, 'sin_litigio')['voters'] == [published['voters_2023'], published['voters_projected']]
    assert [round(new_elector_votes(study, p, 0.8).values.sum()) for p in (0.09, 0.5, 0.7)] == [39405, 218917, 306484]
    cera = study.cera[study.params['baseline_scrutiny']]
    assert exterior_change(study, 0.3, 0.6).values.sum() == pytest.approx(2769645 * 0.3 * 0.99 - cera.values.sum())
    plan = placement_change(study, 1.0, 1.0)
    assert list(plan.columns) == [PLACEMENT_CANDIDATURE] and plan.values.sum() == pytest.approx(181032 * 0.99)
    assert int(study.placement['electors'].sum()) == study.placement['pool'] == 181032


def test_build_snapshot_has_every_scenario_on_its_axes(cera_study):
    snapshot = build_snapshot(double_draws(cera_study), cera_study, META)
    assert set(snapshot) == {'run', 'majority', 'blocks', 'lineage', 'base', 'expected', 'new_electors', 'grid',
                             'placement'}
    assert snapshot['run'] == {key: META[key] for key in ('scope', 'event_date', 'run_id', 'as_of', 'n_sim', 'seed')}
    assert snapshot['blocks'] == META['bmaps']['blocks'] and snapshot['majority'] == 176
    assert snapshot['new_electors']['participations'] == snapshot['grid']['participations'] == GRID_PARTICIPATIONS
    assert [len(snapshot[key]['cells']) for key in ('new_electors', 'grid', 'placement')] == [5, 5, 4]
    assert len(snapshot['placement']['cells'][0]) == 5 and snapshot['placement']['pool'] == 181032
    assert snapshot['grid']['electors'] == 2769645 and snapshot['new_electors']['electors'] == 442257
    assert set(snapshot['expected']) == {'convocatoria', 'sin_litigio'}
    provinces = snapshot['expected']['convocatoria']['provinces']
    assert len(provinces) == 52 and set(provinces[0]) == {'region_id', 'name', 'moved', 'right_lost', 'right_gained'}
    assert snapshot['lineage']['SUMAR'] == 'FA' and snapshot['lineage']['UPN'] == 'PP'
    cell = snapshot['grid']['cells'][0][0]
    assert set(cell) == {'votes', 'lost', 'gained', 'moved', 'right', 'left', 'central'}
    assert sum(cell['right'].values()) == 20 and json.loads(json.dumps(snapshot)) == snapshot
```

- [ ] **Step 3: Run them to see them fail** (`python -m pytest tests/test_cera_scenarios.py -q`)

- [ ] **Step 4: Extract `exterior_votes` and `placement_plan` in `stress.py`, add `placement['electors']` in `_seats_block`, and write `scenarios.py`.** `placement_plan` computes the costs as `optimal_placement` does and returns the second value of `_greedy_seats`.

- [ ] **Step 5: Run the new tests and the study's** (`python -m pytest tests/test_cera_scenarios.py tests/test_cera_seats.py tests/test_cera_study.py tests/test_publish_studies.py -q`; PASS without editing the three old files)

- [ ] **Step 6: Commit**

### Task 5: the study reads the snapshot

**Files:**
- Modify: `mtpy/lib/studies/cera/data.py`, `mtpy/lib/studies/cera/study.py` (`CeraStudy` :87, `compute_cera` :615)
- Test: `tests/test_cera_scenarios.py`

**Interfaces:**
- Produces: `MAJORITY = 'cera/majority.json'`; `check_majority(snapshot: dict, params: Optional[dict] = None) -> list` (texts of what is wrong); `read_majority(data: FileSystem, params: dict) -> Optional[dict]` (`None` when the file does not exist; `ValueError('cera/majority.json: ' + problems)` when it is not valid).
- Produces: `CeraStudy.majority: dict | None`, and `compute_cera(data, params=None, majority=True)`: with `majority=False` the file is not read. A study without the file is computed as today; Task 10 is where its absence stops the publication.

`check_majority` reports: a missing key of the snapshot or of `run`; with `params`, a `run.scope` or `run.event_date` other than `params['election']`'s; a `blocks` without `Derecha` or `Izquierda`; a count outside `0..n_sim`; a `right` or `left` distribution that does not add up to `n_sim`; `cells` that are not the size of their axes; the two censuses of `expected` without the same, non-empty, list of provinces. Each text names the key it is about.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture(scope='module')
def good(cera_study):
    snapshot = build_snapshot(double_draws(cera_study), cera_study, META)
    snapshot['run'].update(commit='abc1234', taken_on='2026-11-25')
    return snapshot


def test_read_majority_is_none_without_the_file_and_gives_back_a_valid_one(tmp_path, cera_study, good):
    fs = FileSystem(str(tmp_path))
    assert read_majority(fs, cera_study.params) is None and check_majority(good, cera_study.params) == []
    fs.write_bytes(json.dumps(good).encode(), MAJORITY)
    assert read_majority(fs, cera_study.params) == good


def test_read_majority_refuses_a_snapshot_that_is_not_this_study_s(tmp_path, cera_study, good):
    fs = FileSystem(str(tmp_path))
    for change, word in ((lambda s: s['run'].update(event_date='2030-01-01'), 'event_date'),
                         (lambda s: s['grid']['cells'].pop(), 'grid'),
                         (lambda s: s['expected']['convocatoria']['counts'].update(lost=21), 'lost'),
                         (lambda s: s['expected']['sin_litigio']['provinces'].pop(), 'provinces'),
                         (lambda s: s.pop('base'), 'base')):
        bad = copy.deepcopy(good)
        change(bad)
        fs.write_bytes(json.dumps(bad).encode(), MAJORITY)
        with pytest.raises(ValueError, match=word):
            read_majority(fs, cera_study.params)


def test_compute_cera_carries_the_snapshot_of_the_repository(cera_study):
    assert cera_study.majority == read_majority(FileSystem(os.path.join(ROOT, 'data')), cera_study.params)
```

- [ ] **Step 2: Run them to see them fail**
- [ ] **Step 3: Implement `check_majority`, `read_majority`, the field and the argument** (document both in the docstrings of `CeraStudy` and `compute_cera`)
- [ ] **Step 4: Run** `python -m pytest tests/test_cera_scenarios.py tests/test_cera_study.py tests/test_publish_studies.py -q` (PASS; until Task 8 the real study has `majority is None`)
- [ ] **Step 5: Commit**

### Task 6: `take_snapshot`

**Files:**
- Create: `mtpy/lib/studies/cera/snapshot.py` (never imported by the package's `__init__`: it is the one module that needs the database)
- Test: `tests/test_cera_snapshot.py` (new)

**Interfaces:**
- Consumes: `fitted_simulator`, `run_mode`, `export_dist`, `export_scenario`, `provenance`, `REPO_ROOT`, `read_manifest` (`mtpy.lib.publish`); `bundle.path_part`, `BundleReader`; `compute._phase`; Tasks 1, 3, 4, 5.
- Produces: `class SnapshotRefused(ValueError)`.
- Produces: `draws_of(sim) -> Draws`: `votes = round(units[..., vpred_pct] × prev_totals['votes'] / 100)` and `central` the same with `central_shares()`, over `params['regions']` and `params['names']`; `base = prev_totals['votes']`, `seats = reg_totals['seats']`, `threshold = sim.threshold`, `results = sim.results`, `central_seats = central_result()`.
- Produces: `rebuild_run(meta: dict, simulator: Optional[Callable] = None, verbose: int = 0) -> Simulator`: `fitted_simulator` with `meta`'s `scope`, `event_date`, `drange` (as a tuple), `alpha`, `seed`, `max_fc`, `correctors`, `limit_date` and `today = meta['as_of']` (the meta does not keep `today`), then `run_mode(sim, 'forecast', meta['n_sim'])`. The nowcast run is skipped: every `run` seeds again.
- Produces: `take_snapshot(scope: str, run_id: Optional[str] = None, save: bool = True, overwrite: bool = False, verbose: int = 0, progress: Optional[Callable] = None, report: Optional[Callable] = None, reader: Optional[BundleReader] = None, data: Optional[FileSystem] = None, rebuild: Optional[Callable] = None, compute: Optional[Callable] = None, prov: Optional[Callable] = None, today=None) -> dict`: `{'status': 'saved' | 'checked', 'action': 'majority', 'run_id', 'path', 'commit', 'dirty', 'n_sim', 'base': {'with_majority', 'right', 'left'}, 'expected': {name: {'votes', 'lost', 'gained', 'moved'}}, 'seconds': {..., 'total'}}`. Defaults: `reader = BundleReader(App.get_().fs)`, `data = App.get_().data`, `rebuild = rebuild_run`, `compute = lambda: compute_cera(data, majority=False)`, `prov = lambda: provenance(cwd=REPO_ROOT)`.

Order; each failure is a `SnapshotRefused` and nothing is written:
1. `scope` is the study's (`load_params(data)['election']['scope']`).
2. When `save`: the file exists and not `overwrite` («cera/majority.json exists: pass overwrite»); then a dirty tree («the snapshot needs a clean tree», untracked files count).
3. The run: `run_id` or the `latest` of the manifest («no published run of es»), and its `meta`, whose `event_date` is the study's.
4. The rebuilt run: `sim.threshold_scope` is `None` and `sim.params['split']` is true; `export_dist(sim)[0]` equals the published `dist` of the `forecast` mode (the reason says how many simulations differ, «1 of 200 simulations differ from the published run») and `export_scenario(sim)[0]` its `scenario`.
5. `reproduction_problems(draws_of(sim))` is empty.
6. `compute()`, `build_snapshot`, `run.commit` and `run.taken_on` (`today` or the day), and `check_majority` is empty.
7. Write, when `save`: `json.dumps(snapshot, ensure_ascii=False, indent=1) + '\n'` to `MAJORITY`.

Phases through `progress` with `_phase` (`read`, `rebuild`, `check`, `scenarios`, `save`), as `load.load_params` does (`mtpy/lib/load.py:589`). Through `report`, at level 1, one line with the base and one per census with the expected vote (votes, lost, gained, moved): the figures Task 8 compares.

- [ ] **Step 1: Write the failing tests.** Doubles, in the test module: `consistent()` is `fakes.synthetic_simulator()` after `sim.run(reset=False, n_sim=200)`, a real run, so its `results` are the allocation of its `units`; `published(tmp_path, sim, **meta)` writes with `make_writer` (`tests/test_publish_unit.py:595`) and `BundleWriter.write_json`, the way `publish_mode` does, the `meta` of the run (the fixture `meta` with `scope` `es`, `run_id` `RUN`, `event_date` `2026-11-29`, `as_of` `2026-11-23`, `n_sim` 200, `seed` 42, `majority` 6, `bmaps.blocks` of `fakes.BMAPS`, and what `**meta` overrides) and the `dist` and `scenario` of the `forecast` mode (`export_dist(sim)[0]`, `export_scenario(sim)[0]`), and returns a `BundleReader` on the same folder; `bare(cera_study)` is `dataclasses.replace(cera_study, majority=None)`.

```python
def snap(tmp_path, sim, study, scope='es', **kwargs):
    kwargs.setdefault('prov', lambda: {'commit': 'abc1234', 'dirty': False})
    kwargs.setdefault('reader', published(tmp_path, consistent()))
    kwargs.setdefault('run_id', RUN)
    return take_snapshot(scope, data=FileSystem(str(tmp_path / 'data')), rebuild=lambda meta, **_: sim,
                         compute=lambda: bare(study), today='2026-11-25', **kwargs)


def test_take_snapshot_writes_the_file_from_the_published_run(tmp_path, cera_study):
    out = snap(tmp_path, consistent(), cera_study)
    written = json.loads((tmp_path / 'data' / MAJORITY).read_text(encoding='utf-8'))
    assert out['status'] == 'saved' and written['run'] == {
        'scope': 'es', 'event_date': '2026-11-29', 'run_id': RUN, 'as_of': '2026-11-23', 'n_sim': 200, 'seed': 42,
        'commit': 'abc1234', 'taken_on': '2026-11-25'}
    assert check_majority(written, cera_study.params) == []
    assert out['base']['with_majority'] == written['base']['with_majority'] and out['n_sim'] == 200
    assert [row['region_id'] for row in written['expected']['convocatoria']['provinces']] == [28, 8]


def test_take_snapshot_refuses_a_run_that_differs_in_one_simulation(tmp_path, cera_study):
    sim = consistent()
    sim.results[7, 1] = [6, 0, 0]  # the six seats of Madrid to one party, in one simulation
    sim.results[7, 0] = sim.results[7, 1:].sum(axis=0)
    with pytest.raises(SnapshotRefused, match='1 of 200'):
        snap(tmp_path, sim, cera_study)
    assert not (tmp_path / 'data' / MAJORITY).exists()
```

and, each asserting the message and that no file was written:

| Test | Arrangement | Refusal |
|---|---|---|
| `test_take_snapshot_refuses_when_the_allocation_does_not_reproduce_the_run` | `sim.prev_totals['votes']` halved (the bundle published from the unchanged one) | step 5 |
| `test_take_snapshot_refuses_when_the_central_allocation_differs` | `published` from a simulator whose `central_table` has one seat moved | «scenario» |
| `test_take_snapshot_refuses_a_scope_threshold` | `sim.threshold_scope = 5.0` | «threshold» |
| `test_take_snapshot_refuses_another_scope_or_election` | `scope='es-md'`; a bundle `published(..., event_date='2027-05-23')` | «scope», «event_date» |
| `test_take_snapshot_does_not_overwrite_and_a_dry_run_writes_nothing` | saved; again → «exists»; `overwrite=True` → saved; in a fresh folder, `save=False` → `checked`, no file | |
| `test_take_snapshot_needs_a_clean_tree_to_write_but_not_to_check` | `prov` dirty: `save` → «clean tree»; `save=False` → `checked`, `out['dirty'] is True` | |
| `test_take_snapshot_takes_the_latest_run_and_refuses_without_one` | `run_id=None` with a manifest whose `latest` is `RUN` → that run; an empty bundle → «no published run» | |
| `test_take_snapshot_overwrites_a_file_the_study_cannot_read` | `{}` written at `MAJORITY`, `overwrite=True`, and the default `compute` replaced by a recorder (`monkeypatch` on `snapshot.compute_cera`) → saved, called with `majority=False` | |

- [ ] **Step 2: Run them to see them fail** (`python -m pytest tests/test_cera_snapshot.py -q`)
- [ ] **Step 3: Implement the module**
- [ ] **Step 4: Run the tests to see them pass**
- [ ] **Step 5: Commit**

### Task 7: the action `load majority`

**Files:**
- Create: `mtpy/pipelines/load/Majority.py`
- Modify: `mtpy/jobs/Load.py` (`sequence`, `WHAT`, `run`, docstring), `deploy/ecs/job.py` (`run_job_value`, :108-118)
- Test: `tests/test_pipelines_load.py`, `tests/test_jobs_load.py`, `tests/test_ecs_job.py`

**Interfaces:**
- Consumes: `take_snapshot`, `SnapshotRefused` (imported inside the task method, so loading the jobs does not load the study).
- Produces: `class Majority(ScopePipeline)`, on the model of `Params`: `label = 'majority'`, `pipes` with `update` and `rebulk`, both `{'Setup': ['prepare'], 'Load': ['snapshot']}`; `start` prints `load majority: 1 ámbito[ · overwrite][ · dry run]`; `snapshot_task(self, chunk=None, dry_run=False, run_id=None, snapshotter=None, **kwargs)` sets `self.result` to `self.collect(chunk, work, action='majority')`, where `work` calls `(snapshotter or take_snapshot)(chunk, run_id=run_id, save=not dry_run, overwrite=self._action == 'rebulk', verbose=self.model_verbose, progress=self.progress, report=...)` and turns a `SnapshotRefused` into `{'status': 'refused', 'action': 'majority', 'reason': str(e)}`.
- Produces: `Load.run(..., run_id=None, snapshotter=None, ...)`, both forwarded; `WHAT = ('polls', 'params', 'majority')`, `sequence = {'load': ['polls', 'params', 'majority']}`.
- Lines (`line_of`): `es: majority saved (run 20261124-102045, right 190 seats, majority in 8734 of 10000) in 95 s`; with `checked`, `…of 10000, dry run) in 95 s`; `es: majority refused (<reason>)` and `es: majority failed (<reason>)`.
- Commands (in the `Load` docstring): `python job.py load '{"what":"majority","run_id":"20261124-102045","dry_run":true}'`, the same without `dry_run`, and with `"overwrite":true`.

- [ ] **Step 1: Write the failing tests**

```python
class FakeSnapshotter:
    def __init__(self, refuse=None):
        self.calls, self.refuse = [], refuse

    def __call__(self, scope, run_id=None, save=True, overwrite=False, **kwargs):
        self.calls.append((scope, run_id, save, overwrite))
        if self.refuse:
            raise SnapshotRefused(self.refuse)
        return {'status': 'saved' if save else 'checked', 'action': 'majority', 'run_id': run_id or 'r', 'n_sim': 10000,
                'base': {'with_majority': 8734, 'right': 190, 'left': 129}, 'seconds': {'total': 95.2}}


def majority(scope, snapshotter, action='update', position=(1, 1), **kwargs):
    p = Majority()
    p.run(action, chunk=scope, position=position, snapshotter=snapshotter, **kwargs)
    return p


def test_majority_takes_the_snapshot_and_builds_its_line(fresh_app, capsys):
    fake = FakeSnapshotter()
    p = majority('es', fake, run_id='20261124-102045')
    assert fake.calls == [('es', '20261124-102045', True, False)]
    assert p.line == 'es: majority saved (run 20261124-102045, right 190 seats, majority in 8734 of 10000) in 95 s'
    assert 'load majority: 1 ámbito' in capsys.readouterr().out


def test_majority_dry_run_and_overwrite_reach_the_snapshot(fresh_app, capsys):
    fake = FakeSnapshotter()
    p = majority('es', fake, action='rebulk', dry_run=True)
    assert fake.calls == [('es', None, False, True)] and p.line.endswith('of 10000, dry run) in 95 s')
    assert 'load majority: 1 ámbito · overwrite · dry run' in capsys.readouterr().out


def test_majority_turns_a_refusal_into_a_line(fresh_app):
    p = majority('es', FakeSnapshotter(refuse='1 of 10000 simulations differ from the published run'))
    assert p.result['status'] == 'refused'
    assert p.line == 'es: majority refused (1 of 10000 simulations differ from the published run)'
```

in `tests/test_jobs_load.py`:

```python
def test_load_majority_passes_the_run_and_the_overwrite(fresh_app, scopes, events):
    fake = FakeSnapshotter()
    out = Load().run(what='majority', scopes=['es'], run_id='20261124-102045', overwrite=True, snapshotter=fake)
    assert out['es']['status'] == 'saved' and fake.calls == [('es', '20261124-102045', True, True)]
```

(and the message of `test_load_unknown_action` becomes `expected one of polls, params, majority`); in `tests/test_ecs_job.py`:

```python
def test_run_job_value_refuses_load_majority():
    for job, params in (('load', '{"what":"majority"}'), ('load', '{"what":["polls","majority"]}'),
                        ('pipeline', '{"select":"load.majority","chunks":"es"}')):
        with pytest.raises(ecs_job.UsageError, match='laptop'):
            ecs_job.run_job_value(job, params)
```

- [ ] **Step 2: Run them to see them fail**
- [ ] **Step 3: Implement the pipeline, the two arguments of `Load` and the refusal of the ECS launcher** (`'load majority runs on the laptop (it writes data/cera/majority.json)'`; the `pipeline` job answers `this pipeline runs on the laptop`, as for `load.params`)
- [ ] **Step 4: Run** `python -m pytest tests/test_pipelines_load.py tests/test_jobs_load.py tests/test_ecs_job.py -q` (PASS)
- [ ] **Step 5: Commit**

### Task 8: take the snapshot

**Files:**
- Create: `data/cera/majority.json` (written by the action)
- Modify: `data/README.md` (a row for the file: what it is, that only `load majority` writes it and that it is never retaken after 29-N)
- Test: `tests/test_cera_study.py`

This task reads the database and the published bundle through the local `.env` (the RDS and `S3_BUCKET`). The `.env` is not versioned: in a worktree, link it from the main tree and never commit it. It is the one step of the plan that cannot be redone after 29-N.

- [ ] **Step 1: Look before running.** `git status` shows nothing, not even an untracked file; `git log -3`; and no job of Luis is running (`ps aux | grep "job.py"`). If one is, wait.

- [ ] **Step 2: Dry run**

Run: `python job.py load '{"what":"majority","run_id":"20261124-102045","dry_run":true,"verbose":2}'`
Expected (a few minutes): `es: majority checked (run 20261124-102045, right 190 seats, majority in 8734 of 10000, dry run) in … s`, and above it the expected vote: convocatoria 30482 votes, 5 lost, 2 gained, 897 moved; sin_litigio 22412, 4, 2, 676.
If the line says `refused`, **stop**: do not change the action to make it pass. Report the reason to Luis; a rebuilt run that differs from the published one means the inputs of 23-N have changed.

- [ ] **Step 3: Take it** (the same command without `dry_run`). Expected: `es: majority saved (…)` and `data/cera/majority.json` in `git status`.

- [ ] **Step 4: Write the test of the real file and see it pass.** Figures from «Cifras de la prueba»:

```python
def test_the_real_snapshot_is_the_run_of_23_n_and_agrees_with_the_study(study):
    snap = study.majority
    assert {key: snap['run'][key] for key in ('scope', 'event_date', 'run_id', 'as_of', 'n_sim', 'seed')} == {
        'scope': 'es', 'event_date': '2026-11-29', 'run_id': '20261124-102045', 'as_of': '2026-11-23',
        'n_sim': 10000, 'seed': 42}
    assert snap['majority'] == 176
    assert snap['blocks'] == {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA', 'UP']}
    assert snap['base']['with_majority'] == 8734 and snap['base']['central'] == {'Derecha': 190, 'Izquierda': 129}
    assert snap['new_electors']['participations'] == snap['grid']['participations']
    expected = snap['expected']['convocatoria']
    assert expected['table'] == expected_table(study, 'convocatoria') and len(expected['provinces']) == 52
    counts = expected['counts']
    assert (counts['votes'], counts['lost'], counts['gained'], counts['moved']) == (30482, 5, 2, 897)
    assert counts['right'] == {'-2': 8, '-1': 367, '0': 9429, '1': 196} and counts['central'] == {'right': 0, 'left': 0}
    lit = snap['expected']['sin_litigio']['counts']
    assert (lit['votes'], lit['lost'], lit['gained'], lit['moved']) == (22412, 4, 2, 676)
    madrid = next(row for row in expected['provinces'] if row['region_id'] == 28)
    assert (madrid['moved'], madrid['right_lost'], madrid['right_gained']) == (172, 59, 39)
    assert sum(1 for row in expected['provinces'] if row['moved']) == 50
    cells = [snap[key]['cells'][i][j] for key, i, j in (
        ('grid', 0, 0), ('grid', 2, 1), ('grid', 2, 3), ('grid', 3, 3), ('grid', 4, 3), ('new_electors', 0, 0),
        ('new_electors', 3, 3), ('placement', 3, 4), ('placement', 0, 0))]
    assert [(c['lost'], c['gained'], c['central']['right']) for c in cells] == [
        (42, 17, 0), (335, 12, -2), (987, 0, -4), (1739, 0, -8), (2635, 0, -10), (8, 4, 0), (218, 1, -2),
        (1223, 0, -5), (83, 0, -1)]
    every = [cell for key in ('new_electors', 'grid', 'placement') for row in snap[key]['cells'] for cell in row]
    assert len(every) == 60
    assert all(seat_range(c['right'])[0] <= c['central']['right'] <= seat_range(c['right'])[1] for c in every)
    assert check_majority(snap, study.params) == []
```

Run: `python -m pytest tests/test_cera_study.py tests/test_cera_scenarios.py -q`
Expected: PASS. A figure that differs is not a test to adjust: stop and report it.

- [ ] **Step 5: Commit** `data/cera/majority.json`, `data/README.md` and the test (the message names the run, the day and «foto fija»).

### Task 9: `outcome`

**Files:**
- Modify: `mtpy/lib/studies/cera/check.py`, `mtpy/lib/studies/cera/study.py` (`_checks_block` :586, `compute_cera`), `data/cera/params.json` (`"majority": {"aliases": {}}`)
- Test: `tests/test_cera_check.py`, `tests/test_cera_study.py`

**Interfaces:**
- Produces: `majority_outcome(results: Results, cera: pd.DataFrame, blocks: dict, majority: int, aliases: Optional[dict] = None) -> dict`: `{'right', 'left', 'right_without', 'left_without', 'has_majority', 'majority_without', 'changed', 'moved', 'net': {'Derecha', 'Izquierda'}}`. `blocks` are the snapshot's (parties of the model); a party stands for the official siglas equal to its name or mapped to it by `aliases` (official sigla → party of the model). Seats with and without the exterior vote as `with_without` counts them (`allocate_results(results)` and `allocate_results(peninsular_results(results, cera))`); `moved`: seats that change hands (the positive differences by sigla, added); `changed`: `has_majority != majority_without`.
- Raises `ValueError('voto-cera: no candidature of the results stands for SALF: add its sigla to majority.aliases in data/cera/params.json')` when a party of either block has no sigla in the results.
- Produces: `CeraStudy.outcome: dict | None`: `None` unless the study has both the check of its election (`checks.get(params['election']['date'])`) and a snapshot; then `majority_outcome` with the snapshot's `blocks` and `majority` and `params['majority']['aliases']`.

- [ ] **Step 1: Write the failing tests** (with `study_inputs`, already imported in `tests/test_cera_check.py`; the invented results of 29-N are those of 2023, where the exterior vote moved one seat in Madrid from the PSOE to the PP)

```python
def test_majority_outcome_counts_the_blocks_with_and_without_the_exterior_vote(study_inputs):
    blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'FA']}
    out = majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, blocks, 176, aliases={'SUMAR': 'FA'})
    assert out == {'right': 170, 'left': 152, 'right_without': 169, 'left_without': 153, 'has_majority': False,
                   'majority_without': False, 'changed': False, 'moved': 1, 'net': {'Derecha': 1, 'Izquierda': -1}}
    low = majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, blocks, 170, aliases={'SUMAR': 'FA'})
    assert (low['has_majority'], low['majority_without'], low['changed']) == (True, False, True)


def test_majority_outcome_refuses_a_block_party_missing_from_the_results(study_inputs):
    blocks = {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA']}
    with pytest.raises(ValueError, match='SALF'):
        majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, blocks, 176, aliases={'SUMAR': 'FA'})
    with pytest.raises(ValueError, match='FA'):
        majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, {'Derecha': ['PP'], 'Izquierda': ['FA']}, 176)
```

and in `tests/test_cera_study.py`:

```python
def test_the_study_has_no_outcome_until_the_election_is_checked(study):
    assert study.outcome is None and study.params['majority'] == {'aliases': {}}
```

- [ ] **Step 2: Run them to see them fail**
- [ ] **Step 3: Implement `majority_outcome`, the field and the parameter**
- [ ] **Step 4: Run** `python -m pytest tests/test_cera_check.py tests/test_cera_study.py tests/test_cera_data.py tests/test_load_cera.py -q` (PASS)
- [ ] **Step 5: Commit**

### Task 10: the part `majority`

**Files:**
- Modify: `mtpy/lib/bundle.py` (`STUDY_PARTS` :87, `STUDY_CSV` :97, `STUDY_CSV_OF` :103, `SCHEMAS` :185), `mtpy/lib/publish/studies.py` (`STUDY_EXPORTERS` :607), `mtpy/lib/publish/__init__.py`, `mtpy/lib/pages/analisis.py` (`CSV_LABELS` :44), `docs/web/contrato.md` (`### voto-cera`, :531-561, and the route row, :986)
- Create: `tests/fixtures/bundle/study-majority.json`
- Test: `tests/test_publish_studies.py`, `tests/test_bundle_unit.py` (:86), `tests/test_pages.py` (`csv_links`, :5929)

**Interfaces:**
- Consumes: `study.majority`, `study.outcome`, `seat_range`.
- Produces: `STUDY_PARTS['voto-cera'] = ('meta', 'census', 'participation', 'vote', 'placement', 'prediction', 'sensitivity', 'majority', 'check')`; schema `'study-majority': {'run': dict, 'majority': int, 'blocks': dict, 'lineage': dict, 'base': dict, 'expected': dict, 'new_electors': dict, 'grid': dict, 'placement': dict, 'outcome': (dict, type(None))}`.
- Produces: `export_study_majority(study) -> tuple`, exported from `mtpy.lib.publish` (the study package stays lazily imported). `data` is a copy of the snapshot plus `outcome` and, in every `counts`, `range`: `seat_range(counts['right'])`. Frames, in this order (the first is the default CSV of the part): `majority-grid` and `majority-new-electors` (`participation`, `left_frac`, `votes`, `lost`, `gained`, `moved`, `with_majority`, `central_right`, `central_left`, `right_lo`, `right_hi`, `mean_right`, `mean_left`), `majority-placement` (the same with `frac`), `majority-provinces` (`census`, `region_id`, `name`, `moved`, `right_lost`, `right_gained`; both censuses). `with_majority` is `base.with_majority − lost + gained`; the means, of the distributions, with four decimals.
- Without a snapshot: `ValueError("voto-cera: data/cera/majority.json is missing: take it with python job.py load '{\"what\":\"majority\"}'")`.
- Labels of the new CSVs: «Mayoría: rejilla de todo el censo», «Mayoría: rejilla de los electores nuevos», «Mayoría: colocación a ciegas», «Mayoría: provincias»; the labels of `last-seat`, `sensitivity-grid`, `sensitivity-scenarios` and `placement-grid` gain «(sobre 2023)».

- [ ] **Step 1: Write the failing tests** (in `tests/test_publish_studies.py`, with its module fixture `study`)

```python
def test_export_study_majority_carries_the_snapshot_the_ranges_and_the_csv(study):
    data, frames = publish.export_study_majority(study)
    assert data['outcome'] is None
    assert all(data[key] == study.majority[key] for key in ('run', 'majority', 'blocks', 'lineage', 'base'))
    assert 'range' not in study.majority['grid']['cells'][0][0]  # the study's own copy is not touched
    assert data['expected']['convocatoria']['counts']['range'] == [-1.0, 0.0]
    assert data['grid']['cells'][4][3]['range'] == [-15.0, -5.0] and data['grid']['cells'][2][1]['range'] == [-5.0, 1.0]
    assert list(frames) == ['majority-grid', 'majority-new-electors', 'majority-placement', 'majority-provinces']
    grid = frames['majority-grid'].set_index(['participation', 'left_frac'])
    assert grid.loc[(0.7, 0.8), ['lost', 'with_majority', 'central_right', 'right_lo', 'right_hi']].tolist() == [
        2635, 6099, -10, -15, -5]
    assert len(frames['majority-grid']) == len(frames['majority-new-electors']) == 20
    assert len(frames['majority-placement']) == 20 and len(frames['majority-provinces']) == 104


def test_export_study_majority_fails_clearly_without_the_snapshot(study):
    with pytest.raises(ValueError, match='load'):
        publish.export_study_majority(dataclasses.replace(study, majority=None))
```

and change the pinned tuple of `tests/test_bundle_unit.py:86` and the `csv_links` count of `tests/test_pages.py` (11 → 15).

- [ ] **Step 2: Run them to see them fail**
- [ ] **Step 3: Implement the registries, the exporter and the labels**
- [ ] **Step 4: Write the fixture** `tests/fixtures/bundle/study-majority.json`: the envelope of `study-sensitivity.json` with `schema` `study-majority@1` and, as `data`, the real `export_study_majority(compute_cera(...))[0]`, written from the session scratchpad. The page tests of Tasks 11 and 12 read the spec's figures from it.
- [ ] **Step 5: Update `docs/web/contrato.md`**: «Nueve partes», the row of `majority` in the parts table (its mandatory keys) and in the CSV table, the route row, and one paragraph: the part is a frozen snapshot of the run named in `run`, written once by `load majority`; counts are over `run.n_sim`; `outcome` is `null` until the election is checked.
- [ ] **Step 6: Compare with what the study published before.** Export again the seven parts of Step 1 of Task 4 from the scratchpad and `diff -r` the two folders: no difference. Then `python -m pytest -q` (the whole suite: the registration tests of `tests/test_publish_studies.py` run with the real study).
- [ ] **Step 7: Commit**

### Task 11: the view model of the card

**Files:**
- Modify: `mtpy/lib/pages/cera.py` (new functions; nothing calls them until Task 12)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: the part `majority` (the fixture is the real one) and the study `meta` (`params.litigation.count`, `params.placement_countries`).
- Produces: `cera._increment(prediction: dict) -> dict`: `{'scenarios': [{'key', 'label', 'changed', 'unchanged'}]}`, the part of `_question_3` that «Lo que dice el debate» and «Comprobación» read (template :722, :777, :839).
- Produces: `cera._majority(part: dict, meta: dict) -> Optional[dict]`; `None` when the part is empty. Keys:
  - `as_of`, `n_sim`, `majority`; `base`: `{'right', 'needed', 'p', 'quartile', 'needed_quartile', 'p_quartile', 'p_without'}` (`needed = right − majority + 1`; `quartile`: percentile 25 of `base.right_seats`; `p_quartile`: simulations at or under it; `p_without`: simulations under the majority).
  - `expected`: `{'seats', 'right', 'p', 'net'}` (the central change of the right with the convocation census, its seats then, the probability, and `lost − gained`); `extreme`: `{'seats', 'right', 'p', 'participation', 'left_frac'}`, the last cell of `grid`.
  - `table`: two rows `{label, before, after, change}` («Censo CERA», «Votantes CERA»); `litigation`.
  - `changes`: `{'moved', 'total', 'down', 'up', 'majority'}`, per cent of the simulations with two decimals: some seat moved; the right's total changed, and how much of it down and up; `lost + gained`.
  - `provinces`: `{'rows': [{name, p, lost, gained, other, width}], 'ever', 'total'}`: the ten with most `moved`; `p`, `lost`, `gained` and `other` in per cent with two decimals (`other`: moved without a change of the right); `width` relative to the first.
  - `ruler`: `{'max': 16, 'ticks': [0, 2, …, 16], 'marks': [{'at', 'right'}], 'groups': [{'title', 'rows': [{name, detail, seats, lo, hi, p}]}]}`. Marks: `needed_quartile` (right `quartile`) and `needed` (right `base.right`). Rows from a constant `RULER`: «Electores nuevos desde 2023»: (`new_electors`, 0.09, 0.5, detail «como 2023»), (`new_electors`, 0.5, 0.8), (`placement`, 1.0, 1.0, name «Colocación a ciegas», detail «100 % voto + 100 % PSOE»); «Todo el censo exterior»: (`grid`, 0.3, 0.6), (`grid`, 0.3, 0.8), (`grid`, 0.5, 0.8), (`grid`, 0.7, 0.8). Name of a cell: «30 % voto + 60 % izq.». `seats`: what the right loses in the central allocation (`−central.right`); `lo`, `hi`: `−range`, cut at zero; `p`: one decimal. A row whose cell is not on the axes of the part is left out.
  - `grid`: `{'electors', 'cols': ['50 % izq.', …], 'rows': [{'label': '9 % voto', 'cells': [{seats, p, drop}]}], 'max_drop'}`; `seats` the signed change of the right, `drop = 100 * (lost − gained) / n_sim`, unrounded.
  - `placement`: `{'pool', 'countries', 'seats': [least, most], 'p': [highest, lowest]}` over the cells of the placement grid (`countries` as `_question_3` names them today).
  - `summary`: `{'needed', 'extreme', 'middle'}` (`extreme`: seats lost in the last cell of `grid`; `middle`: in the second row of the ruler); `outcome`: the part's, or `None`.
- Probability of a scenario: `round(100 * (base.with_majority − lost + gained) / n_sim, 1)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cera_majority_view_model_reads_everything_from_the_part():
    q3 = pages.cera._majority(fixture_data('study-majority'), fixture_data('study-meta'))
    assert (q3['as_of'], q3['n_sim'], q3['majority'], q3['litigation']) == ('2026-11-23', 10000, 176, 163020)
    assert q3['base'] == {'right': 190, 'needed': 15, 'p': 87.3, 'quartile': 181, 'needed_quartile': 6,
                          'p_quartile': 27.7, 'p_without': 12.7}
    assert q3['expected'] == {'seats': 0, 'right': 190, 'p': 87.3, 'net': 3}
    assert q3['extreme'] == {'seats': -10, 'right': 180, 'p': 61.0, 'participation': 70, 'left_frac': 80}
    assert q3['table'] == [{'label': 'Censo CERA', 'before': 2327388, 'after': 2769645, 'change': 442257},
                           {'label': 'Votantes CERA', 'before': 208431, 'after': 239221, 'change': 30790}]
    assert q3['changes'] == {'moved': 8.97, 'total': 5.71, 'down': 3.75, 'up': 1.96, 'majority': 0.07}
    assert q3['provinces']['rows'][0] == {'name': 'Madrid', 'p': 1.72, 'lost': 0.59, 'gained': 0.39, 'other': 0.74,
                                          'width': 100.0}
    assert (len(q3['provinces']['rows']), q3['provinces']['ever'], q3['provinces']['total']) == (10, 50, 52)
    rows = [row for group in q3['ruler']['groups'] for row in group['rows']]
    assert [(r['name'], r['seats'], r['lo'], r['hi'], r['p']) for r in rows] == [
        ('9 % voto + 50 % izq.', 0, 0, 1, 87.3), ('50 % voto + 80 % izq.', 2, 0, 3, 85.2),
        ('Colocación a ciegas', 5, 0, 8, 75.1), ('30 % voto + 60 % izq.', 2, 0, 5, 84.1),
        ('30 % voto + 80 % izq.', 4, 1, 8, 77.5), ('50 % voto + 80 % izq.', 8, 3, 12, 70.0),
        ('70 % voto + 80 % izq.', 10, 5, 15, 61.0)]
    assert rows[0]['detail'] == 'como 2023' and rows[2]['detail'] == '100 % voto + 100 % PSOE'
    assert q3['ruler']['marks'] == [{'at': 6, 'right': 181}, {'at': 15, 'right': 190}]
    assert q3['grid']['rows'][0]['label'] == '9 % voto' and q3['grid']['cols'] == ['50 % izq.', '60 % izq.', '70 % izq.',
                                                                                 '80 % izq.']
    assert q3['grid']['rows'][4]['cells'][3] == {'seats': -10, 'p': 61.0, 'drop': 26.35}
    assert q3['grid']['max_drop'] == 26.35 and q3['grid']['electors'] == 2769645
    assert q3['placement'] == {'pool': 181032, 'countries': [], 'seats': [1, 5], 'p': [86.5, 75.1]}
    assert q3['summary'] == {'needed': 15, 'extreme': 10, 'middle': 2} and q3['outcome'] is None


def test_cera_majority_view_model_skips_a_row_the_part_does_not_have():
    part = copy.deepcopy(fixture_data('study-majority'))
    part['new_electors']['participations'][0] = 0.1
    rows = [row['name'] for group in pages.cera._majority(part, fixture_data('study-meta'))['ruler']['groups']
            for row in group['rows']]
    assert '9 % voto + 50 % izq.' not in rows and len(rows) == 6
    assert pages.cera._majority({}, fixture_data('study-meta')) is None


def test_cera_increment_keeps_what_the_debate_and_the_check_read():
    scenarios = pages.cera._increment(fixture_data('study-prediction'))['scenarios']
    assert [s['key'] for s in scenarios] == ['convocatoria', 'sin_litigio']
    assert set(scenarios[0]) == {'key', 'label', 'changed', 'unchanged'} and scenarios[0]['unchanged'] is True
```

- [ ] **Step 2: Run them to see them fail**
- [ ] **Step 3: Implement `_majority` and `_increment`**
- [ ] **Step 4: Run** `python -m pytest tests/test_pages.py -q -k "majority or increment"` (PASS)
- [ ] **Step 5: Commit**

### Task 12: the card

**Files:**
- Modify: `mtpy/lib/pages/analisis.py` (`QUESTIONS` :33, `_island` :493, `cera_context` :607), `mtpy/lib/pages/cera.py` (what only the old card read), `web/templates/analisis_voto_cera.html` (index :84, summary card :103-112, `#p3` :392-613, the uses of `c1`/`c2` outside it), `web/dist/css/site.css` (:1009-1016, :1066), `web/dist/js/pages/analisis-cera.js` (:245-286, :301)
- Delete: `web/dist/js/charts/radar.js` (only this page used it) and its test `test_radar_option_two_series` (`tests/test_js_charts.py:909`)
- Test: `tests/test_pages.py` (`test_cera_page_questions_2_and_3` :7347, and the three tests that pin the old context: :5929, :7196, :7633)

**Interfaces:**
- Consumes: `cera._majority`, `cera._increment` (Task 11); `checked`, `study_meta`, `figures`, `m.kpis`, `m.how`, `shot`, the filter `shade_scale`.
- Produces: `QUESTIONS[2] = '¿Puede el voto CERA arrebatar la mayoría absoluta a la derecha el 29-N?'`; context `q3 = _majority(parts['majority'], meta)` and `increment = _increment(parts['prediction'])`; the island without `gap` and `radar`.

The card is the mockup, block by block, with the page's own components. From top to bottom: header and verdict; the band of `q3.outcome` (only when it is not `None`); lead and the pill «Foto fija · datos hasta el 23 de noviembre» (`q3.as_of`); the three figures; «Con el voto que se espera» in two columns (left: table, the three sentences and their note; right: the ten provinces) and the litigation note at full width; «¿Qué haría falta para que importara?» in two columns of three and two fifths (the scenario chart and its two notes; the grid and its note); the comment on the placement; the summary; «Cómo leerlo».

- The scenario chart, the province bars and the three sentences are HTML and CSS, like the `.herd-bars` of `/analisis/rebano` (`site.css:1078`): they must come out in the image of the card (`shot('pregunta-3')`, html2canvas). Positions of the chart are percentages of `ruler.max`; its two lines are labelled «si la derecha suma {{ mark.right }}».
- The page's province names are the study's («S. C. Tenerife», «Baleares», «Orense»).
- Grid cells: two lines (seats with sign, probability), background from `drop` through `shade_scale(0, q3.grid.max_drop)`.
- Signs with «−» (U+2212). The second figure says «(3 simulaciones menos de 10.000)» from `expected.net`: «más» if negative, nothing if zero. A quantile that is not a whole number is shown with one decimal.
- Index entry: «3 · ¿Quitan la mayoría?». Card of the three questions: «Con el voto que se espera no cambia ningún escaño del reparto central del modelo, y sólo en el 0,07 % de sus simulaciones cambia el sentido de la mayoría.», with its figures from `q3`.
- `c1` and `c2` come from `increment.scenarios`; the three sentences that use them (:722, :777, :839) do not change.
- The band: «Qué pasó el 29-N. La derecha obtuvo {right} escaños. El voto CERA, contado el {study_meta.checked_at}, movió {moved} escaños ({net of each block}) y {no cambió | cambió} la mayoría.», and under it «Lo que sigue es la foto del 23 de noviembre, tal como se publicó antes de votar: no se ha recalculado.»
- Without the part (`q3` is `None`): the card keeps its header and shows one note, «Esta pregunta se está actualizando con la predicción del modelo para el 29-N.»; the card of the three questions shows only the title and the verdict of the third.
- «Cómo leerlo» explains, in this order (spec D8): what a simulation is and what is compared; the two readings and why they may differ; why two references, 15 and 6; how the 2023 candidatures become the parties of 29-N (SUMAR → FA, UPN → PP, the rest without seat); what each chart moves (only the new electors, the whole exterior census, the placed pool); why the placement is blind and a limit; why it is a frozen snapshot; and that 30.482 of the 30.790 new voters are votes to candidatures. These texts, the fallback note and the band are provisional until Luis reviews them.
- At 390 px the two columns stack (`.grid2` already does) and the labels of the chart go above their bar.
- The card does not nest a `<section>`: `cera_section` in the tests cuts at the first closing tag.
- Removed: `drawGap`, `drawRadar`, `checkedScenario`, `showScenario`, `RADAR_AXES`, the `#scenario` wiring and the `renderRadar` import; the CSS of `#gap-chart` and the two `#p3:has(#scenario …)` rules; and `_question_3`, `_scenario_table`, `_grid`, `_scenario_rows`, `GAP_SHOWN` and the `_figures` keys that only the old card read. Before deleting a name, `grep` it in `web/templates/` and `mtpy/lib/pages/`: `_figures` also feeds the KPI row, the debate and the check.

- [ ] **Step 1: Rewrite the question-3 half of `test_cera_page_questions_2_and_3`** (from `# question 3`, :7384) and add the two states

```python
    p3 = cera_section(page, 'p3')
    assert page[:page.index('id="p3"')].endswith('<section class="card qcard shot q3" ')
    assert ('<div class="qn">3</div><h2 class="qt">¿Puede el voto CERA arrebatar la mayoría absoluta a la derecha el '
            '29-N?</h2>') in p3
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', p3) == ['NO']
    text = visible_text(p3)
    for piece in ('Foto fija · datos hasta el 23 de noviembre', '190 escaños', '−10 escaños', '2.769.645', '+442.257',
                  '239.221', '+30.790', '3 simulaciones menos de 10.000', '8,97 %', '5,71 %',
                  '3,75 % a la baja y 1,96 % al alza', 'Sólo en el', '0,07 %', 'cambia el sentido de la mayoría',
                  'Madrid', '1,72 %', 'ocurre alguna vez en 50 de las 52',
                  'El censo incluye las 163.020 inscripciones en litigio.', 'si la derecha suma 181',
                  'si la derecha suma 190', '9 % voto + 50 % izq.', 'como 2023', '50 % voto + 80 % izq.',
                  'Colocación a ciegas', '30 % voto + 60 % izq.', '84,1 %', '70 % voto + 80 % izq.', '61,0 %',
                  'Si los 2.769.645 electores del 29-N votaran así', 'entre 1 y 5 escaños', '86,5 %', '75,1 %',
                  'ni un escaño: la derecha sigue en 190', 'dos escaños'):
        assert piece in text, piece
    assert p3.count('class="rrow"') == 7 and len(re.findall(r'<td[^>]*class="[^"]*cell-shade', p3)) == 20
    for gone in ('gap-chart', 'radar-chart', 'id="scenario"', 'prediction-table', 'placement-table', 'la consigue en',
                 'Qué pasó el 29-N', 'None'):
        assert gone not in p3, gone
    assert '<a href="#p3" class="q3">3 · ¿Quitan la mayoría?</a>' in page
    summary = visible_text(cera_section(page, 'resumen'))
    assert '¿Puede el voto CERA arrebatar la mayoría absoluta a la derecha el 29-N?' in summary
    assert 'sólo en el 0,07 % de sus simulaciones cambia el sentido de la mayoría' in summary


def test_cera_page_shows_what_happened_once_the_part_has_an_outcome(api, fresh_app):
    outcome = {'right': 181, 'left': 140, 'right_without': 182, 'left_without': 139, 'has_majority': True,
               'majority_without': True, 'changed': False, 'moved': 1, 'net': {'Derecha': -1, 'Izquierda': 1}}
    publish_study(fresh_app.fs, meta={'checked_at': '2026-12-04'}, parts={'majority': {'outcome': outcome}})
    text = visible_text(cera_section(cera_page(api), 'p3'))
    for piece in ('Qué pasó el 29-N', 'La derecha obtuvo 181 escaños', 'movió 1 escaño', 'no cambió la mayoría',
                  'Lo que sigue es la foto del 23 de noviembre', '190 escaños', '0,07 %'):
        assert piece in text, piece


def test_cera_page_without_the_majority_part_keeps_rendering(api, fresh_app):
    publish_study(fresh_app.fs)  # then the bundle of a study published before this change: no such part
    bundle.BundleWriter(fresh_app.fs).remove(bundle.path_study('voto-cera', 'majority'))
    webapi.site().manifest_cache_clear()
    page = cera_page(api)
    p3 = visible_text(cera_section(page, 'p3'))
    assert 'Esta pregunta se está actualizando' in p3 and 'None' not in p3
    assert 'id="p1"' in page and 'id="debate"' in page and 'id="comprobacion"' in page
```

The class names (`rrow`, `cell-shade`) are those the template ends up using: fix the two counts to them. In the three tests that pin the old context, the assertions on `initial.gap` and `initial.radar` go, and those on `figures` stay for the keys that remain.

- [ ] **Step 2: Run them to see them fail**
- [ ] **Step 3: Wire the context, write the card, its styles and the JS clean-up.** Copy the texts from the mockup; take its CSS as the starting point and name the classes after the page's conventions.
- [ ] **Step 4: Run** `python -m pytest tests/test_pages.py tests/test_js_charts.py tests/test_pages_titles.py tests/test_pages_structured.py tests/test_pages_descriptions.py -q` (PASS; `assert_public_wording` runs over `/analisis/voto-cera`)
- [ ] **Step 5: Commit**

### Task 13: the page in a browser, and the loose ends

**Files:**
- Modify: `docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-design.md` (one line under the title: question 3 was redone on 2026-11-24, with the path of the new spec)
- No file of the repository for the checks: the scripts live in the session scratchpad.

- [ ] **Step 1: Serve the page locally without publishing.** A scratchpad app, as in the earlier browser checks (`docs/superpowers/plans/2026-11-09-web-detalle-sondeo.md:736-756`: `uvicorn` on the `api` of `api.py`, `/dist/` from `web/dist`, headless Chrome), over a bundle built in a temporary folder with the test helpers (`tests.test_webapi_unit.write_bundle` and `write_study`, as `fresh_app` does): the `majority` part of the fixture is the real one. Never run `publish study` against the bucket: that is Luis's step.
- [ ] **Step 2: Look at `/analisis/voto-cera` at 1280 and at 390 px** (screenshots read back): the tables, the province bars and the scenario chart are not cut, the chart has its axis and its two lines, the two charts of «¿Qué haría falta?» sit side by side at 1280 (the scenario one wider) and stacked at 390, the litigation note takes the whole width, no error in the console, and the image of the card downloads with the chart in it. Compare with the mockup; fix what differs and rerun the page tests.
- [ ] **Step 3: Sweep the texts.** `grep -rn "cero escaños\|Mueven escaños\|alterar el reparto" web/templates mtpy/lib docs/web`: what speaks of question 3 says the new thing; what speaks of the registered prediction («Lo que dice el debate», «Comprobación») stays.
- [ ] **Step 4: Run the whole suite** (`python -m pytest -q`): the same result as before the first task plus the new tests.
- [ ] **Step 5: Add the pointer line to the 2026-10-19 spec and commit.**
- [ ] **Step 6: Hand over to Luis** what is his: review the snapshot (its summary line and `data/cera/majority.json`) and the texts («Cómo leerlo», the fallback note, the band), deploy, and publish the study (`python job.py publish '{"what":["study"],"study":"voto-cera"}'`, without `register`). After the exterior count of 29-N: load results and scrutiny as planned, add to `majority.aliases` the aliases the official siglas need, and publish the study again. Still open and his: the study's 2023 blocks (`data/cera/params.json`) leave FA and SALF in «Otros» in the check of the registered prediction.
