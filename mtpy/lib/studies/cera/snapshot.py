"""
The snapshot of the CERA study on the published forecast (`cera/majority.json`).

`take_snapshot` rebuilds a published run of the model with the recipe of `publish_forecast`, refuses unless the
rebuilt run is the published one simulation by simulation, measures every scenario of exterior vote on its draws
(`scenarios.build_snapshot`) and writes the file. The file is taken before an election and cannot be retaken after
it, so every doubt is a refusal (`SnapshotRefused`) and nothing is written.

It is the one module of the study that needs the database (through the simulator): the package `__init__` never
imports it.
"""
import json
import time
from typing import Callable, Optional

import numpy as np

from ... import bundle
from ...bundle import BundleReader
from ...compute import _phase
from ...publish import (
    REPO_ROOT,
    export_dist,
    export_scenario,
    fitted_simulator,
    provenance,
    read_manifest,
    run_mode,
)
from ...publish.common import _today
from ....core.app import App
from ....core.io import FileSystem
from .data import MAJORITY, check_majority
from .majority import Draws, reproduction_problems
from .scenarios import build_snapshot
from .study import compute_cera

MODE = 'forecast'


class SnapshotRefused(ValueError):
    """The snapshot cannot be taken: the reason is the message; nothing was written."""


def draws_of(sim) -> Draws:
    """
    The simulations of a run as the draws of the study: the very votes the simulator allocated.

    Each province gets `round(vpred_pct × valid / 100)` with the valid votes of the previous election
    (`prev_totals['votes']`), as `Simulator.simulate` does (a party without a share, NaN, gets no votes); the central
    allocation the same with `central_shares()`. The votes are exact, not an approximation: `build_umat` hands
    `simulate` shares already rounded to 2 decimals, and those are the ones `units` keeps (the `round(2)` of `run`
    changes nothing) and `central_shares()` returns.

    Parameters
    ----------
    sim : Simulator
        A simulator already run, with provinces (`split`).

    Returns
    -------
    Draws
        Over `params['regions']` and `params['names']`, with the stored `results` and `central_result()`.
    """
    names, regions = list(sim.params['names']), list(sim.params['regions'])
    cats = sim.categories()
    cols = [cats.index(name) for name in names]
    valid = sim.prev_totals['votes'].loc[regions].to_numpy(dtype=float)
    shares = np.nan_to_num(sim.units[:, :, cols, sim.cols_unit.index('vpred_pct')])
    central = np.nan_to_num(sim.central_shares()[names].to_numpy(dtype=float))

    return Draws(names=names, regions=regions, default_region=sim.default_region,
                 votes=np.round(shares * valid[None, :, None] / 100), central=np.round(central * valid[:, None] / 100),
                 base=valid, seats=sim.reg_totals['seats'].loc[regions].to_numpy(),
                 threshold=sim.threshold, results=sim.results, central_seats=sim.central_result().to_numpy())


def rebuild_run(meta: dict, simulator: Optional[Callable] = None, verbose: int = 0):
    """
    Rebuild a published run: the simulator of `publish_forecast` with the parameters of its `meta`, run in the
    `forecast` mode.

    The nowcast run of the publication is skipped: every `run` seeds the generators again, so the forecast does not
    depend on it. The anchor is capped at the run's `as_of` (the `meta` does not keep the `today` of the run).

    Parameters
    ----------
    meta : dict
        The published `meta` of the run.
    simulator : callable, optional
        Injectable replacement of `Simulator`.
    verbose : int, default 0
        Simulator verbosity.

    Returns
    -------
    Simulator
        The simulator, run in the `forecast` mode.
    """
    sim = fitted_simulator(meta['scope'], meta['event_date'], tuple(meta['drange']), meta['alpha'], meta['seed'],
                           meta['max_fc'], meta['correctors'], verbose=verbose, today=meta['as_of'],
                           limit_date=meta['limit_date'], simulator=simulator)
    run_mode(sim, MODE, meta['n_sim'])

    return sim


def _read_run(reader: BundleReader, scope: str, run_id: Optional[str], study) -> tuple:
    """The run to rebuild: its id, its `meta` and its published `dist` and `scenario` of the `forecast` mode."""
    own = study.params['election']
    if scope != own['scope']:
        raise SnapshotRefused(f"scope {scope} is not the study's {own['scope']}")
    if run_id is None:
        run_id = (read_manifest(reader)['scopes'].get(scope) or {}).get('latest')
        if run_id is None:
            raise SnapshotRefused(f'no published run of {scope}')
    try:
        meta = reader.read_json(bundle.path_part(scope, run_id, 'meta'))['data']
        dist, scenario = (reader.read_json(bundle.path_part(scope, run_id, part, MODE))['data']
                          for part in ('dist', 'scenario'))
    except FileNotFoundError as e:
        raise SnapshotRefused(f'the run {run_id} of {scope} is not published whole: {e}') from e
    if meta['event_date'] != own['date']:
        raise SnapshotRefused(f"event_date {meta['event_date']} of the run {run_id} is not the study's {own['date']}")

    return run_id, meta, dist, scenario


def _check_run(sim, dist: dict, scenario: dict) -> Draws:
    """Refuse unless the rebuilt run is the published one and its draws reproduce it; return the draws."""
    if sim.threshold_scope is not None:
        raise SnapshotRefused(f'the run has a threshold over the whole scope ({sim.threshold_scope}): '
                              'the study allocates with the threshold of the province only')
    if not sim.params['split']:
        raise SnapshotRefused('the run does not allocate the seats by province (split)')

    rebuilt = bundle.jsonable(export_dist(sim)[0])
    if rebuilt != dist:
        if rebuilt['parties'] != dist['parties'] or len(rebuilt['seats']) != len(dist['seats']):
            raise SnapshotRefused('the rebuilt run has {} simulations of {} and the published one {} of {}'.format(
                len(rebuilt['seats']), rebuilt['parties'], len(dist['seats']), dist['parties']))
        differ = sum(mine != theirs for mine, theirs in zip(rebuilt['seats'], dist['seats']))
        raise SnapshotRefused('{} of {} simulations differ from the published run'.format(differ, len(dist['seats'])))
    if bundle.jsonable(export_scenario(sim)[0]) != scenario:
        raise SnapshotRefused('the central allocation (scenario) differs from the published run')

    draws = draws_of(sim)
    problems = reproduction_problems(draws)
    if problems:
        raise SnapshotRefused('the allocation of the votes of the simulator does not reproduce the run: '
                              + '; '.join(problems))

    return draws


def take_snapshot(
    scope: str,
    run_id: Optional[str] = None,
    save: bool = True,
    overwrite: bool = False,
    verbose: int = 0,
    progress: Optional[Callable] = None,
    report: Optional[Callable] = None,
    reader: Optional[BundleReader] = None,
    data: Optional[FileSystem] = None,
    rebuild: Optional[Callable] = None,
    compute: Optional[Callable] = None,
    prov: Optional[Callable] = None,
    today=None
) -> dict:
    """
    Take the snapshot of the study on a published run and write `cera/majority.json`.

    In order, each failure a `SnapshotRefused` with nothing written: with `save`, the file must not exist (unless
    `overwrite`) and the tree must be clean; the study is computed and `scope` must be its own; the run (`run_id`, or
    the `latest` of the manifest) must be of the study's election; the rebuilt run must have no threshold over the
    whole scope, allocate by province and give the published `dist` and `scenario` of the `forecast` mode; its draws
    must reproduce its seats (`reproduction_problems`); and the snapshot must pass `check_majority`.

    Parameters
    ----------
    scope : str
        Scope of the run; it must be the study's.
    run_id : str, optional
        The published run; the `latest` of the manifest by default.
    save : bool, default True
        Write the file; `False` checks everything and writes nothing (a dirty tree is then allowed).
    overwrite : bool, default False
        Replace an existing file.
    verbose : int, default 0
        Simulator verbosity.
    progress : callable, optional
        `progress(step, seconds)` after each phase: `read`, `rebuild`, `check`, `scenarios` and `save`.
    report : callable, optional
        `report(text, level)`: at level 1, one line with the base and one per census with its expected vote.
    reader : BundleReader, optional
        The bundle; `BundleReader(App.get_().fs)` by default.
    data : FileSystem, optional
        The file system rooted at `data/`, where the file is written; `App.get_().data` by default.
    rebuild : callable, optional
        `rebuild(meta, verbose=...)` returning the simulator run; `rebuild_run` by default.
    compute : callable, optional
        Returns the study; `compute_cera(data, majority=False)` by default (the file is not opened).
    prov : callable, optional
        Returns `{'commit', 'dirty', ...}`; `provenance(cwd=REPO_ROOT)` by default.
    today : str, optional
        The day the snapshot is taken (`YYYY-MM-DD`); today in Europe/Madrid by default.

    Returns
    -------
    dict
        `{'status': 'saved' | 'checked', 'action': 'majority', 'run_id', 'path', 'commit', 'dirty', 'n_sim',
        'base': {'with_majority', 'right', 'left'}, 'expected': {census: {'votes', 'lost', 'gained', 'moved'}},
        'seconds': {phase: s, 'total': s}}`.

    Raises
    ------
    SnapshotRefused
        The snapshot cannot be taken; the message says why.
    """
    reader = reader or BundleReader(App.get_().fs)
    data = data or App.get_().data
    rebuild = rebuild or rebuild_run
    compute = compute or (lambda: compute_cera(data, majority=False))
    prov = prov or (lambda: provenance(cwd=REPO_ROOT))
    start = time.perf_counter()
    seconds = {}

    tree = prov()
    if save:
        if data.exists(MAJORITY) and not overwrite:
            raise SnapshotRefused(f'{MAJORITY} exists: pass overwrite')
        if tree['dirty']:
            raise SnapshotRefused('the snapshot needs a clean tree (untracked files count): commit first')

    def read():
        study = compute()
        return (study, *_read_run(reader, scope, run_id, study))

    study, run_id, meta, dist, scenario = _phase('read', read, seconds, progress)
    sim = _phase('rebuild', lambda: rebuild(meta, verbose=verbose), seconds, progress)
    draws = _phase('check', lambda: _check_run(sim, dist, scenario), seconds, progress)

    def measure():
        out = build_snapshot(draws, study, meta)
        out['run'].update(commit=tree['commit'], taken_on=_today(today).isoformat())
        problems = check_majority(out, study.params)
        if problems:
            raise SnapshotRefused('the snapshot is not valid: ' + '; '.join(problems))
        return out

    snapshot = _phase('scenarios', measure, seconds, progress)
    if save:
        _phase('save', lambda: data.write_bytes(
            (json.dumps(snapshot, ensure_ascii=False, indent=1) + '\n').encode('utf-8'), MAJORITY), seconds, progress)
    seconds['total'] = time.perf_counter() - start

    base = snapshot['base']
    expected = {name: {key: census['counts'][key] for key in ('votes', 'lost', 'gained', 'moved')}
                for name, census in snapshot['expected'].items()}
    n_sim = snapshot['run']['n_sim']
    if report is not None:
        report('{} {}: base with_majority {} of {}, central Derecha {}, Izquierda {}'.format(
            scope, run_id, base['with_majority'], n_sim, base['central']['Derecha'], base['central']['Izquierda']), 1)
        for name, counts in expected.items():
            report('{} {}: {} votes {}, lost {}, gained {}, moved {}'.format(
                scope, run_id, name, counts['votes'], counts['lost'], counts['gained'], counts['moved']), 1)

    return {'status': 'saved' if save else 'checked', 'action': 'majority', 'run_id': run_id,
            'path': data.get_path(MAJORITY), 'commit': tree['commit'], 'dirty': bool(tree['dirty']), 'n_sim': n_sim,
            'base': {'with_majority': base['with_majority'], 'right': base['central']['Derecha'],
                     'left': base['central']['Izquierda']},
            'expected': expected, 'seconds': seconds}
