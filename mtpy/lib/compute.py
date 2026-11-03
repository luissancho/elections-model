"""
Computation of the poll series of a scope and of the pollster ratings: the sequence of
`notebooks/data-load/PollsCompute.ipynb`, shared by the `compute` job (`mtpy/jobs/Compute.py`) and the batch
loader of the regional scopes (`load/run_load.py`).

`compute_scope` runs the series of one scope in one of two modes: `full` (weights, featured events, errors,
deviations, drift, house effects and herding) or `light` (weights and herding, what the polls of the current cycle
can change); `average_errors_scope` the errors of the poll average in the past elections (the table of the
uncertainty options of method 15, an action of its own); `rate_scope` the cross-scope pollster ratings of its
events; `replicate_catalogues` the versioned catalogues (scopes, districts and the prior quality of the pollsters),
once for every scope. The first three report each phase through an optional `progress(step, seconds)` callback; all
return the figures of the run, and a failure propagates to the caller, which decides how to isolate it.
`resolve_mode` is the rule that picks the mode of a scope from what is pending (`auto`), pure, for the pipeline.
`pending_events` and `result_dates` are imported only to be re-exported for `mtpy/pipelines/compute`.
"""
import time
from typing import Callable, Optional

from .computer import Computer
from .data import db_now, mark_computed, pending_events, result_dates, save_catalogues, update_featured  # noqa: F401
from .quality import load_criteria, read_quality

# Phases of the `full` mode of `compute_scope`, in order; the weights run twice (see `compute_scope`)
COMPUTE_STEPS = ('weights', 'featured', 'errors', 'deviations', 'drift', 'house_effects', 'herding')
# Phases of each mode of `compute_scope`
MODES = {
    'light': ('weights', 'herding'),
    'full': COMPUTE_STEPS
}
# Mode resolved by `resolve_mode` from what is pending (the default of the pipeline and the jobs)
AUTO = 'auto'
# Phase of `average_errors_scope`
AVERAGE_ERRORS_STEP = 'average_errors'
# Phase of `rate_scope`
RATINGS_STEP = 'ratings'


def _phase(name: str, action: Callable, seconds: dict, progress: Optional[Callable]):
    """
    Run one phase, add its seconds to `seconds[name]` and report it.

    Parameters
    ----------
    name : str
        Phase key (one of `COMPUTE_STEPS` or `RATINGS_STEP`).
    action : callable
        Function without arguments that runs the phase; its result is returned.
    seconds : dict
        Seconds of each phase, updated in place (a phase that runs twice accumulates).
    progress : callable, optional
        ``progress(step, seconds)`` called with the seconds of this run of the phase.

    Returns
    -------
    object
        Whatever `action` returns.
    """
    start = time.perf_counter()
    result = action()
    elapsed = time.perf_counter() - start
    seconds[name] = seconds.get(name, 0.) + elapsed
    if progress is not None:
        progress(name, elapsed)

    return result


def compute_scope(
    scope: str,
    save: bool = True,
    verbose: int = 0,
    progress: Optional[Callable] = None,
    computer: Optional[Callable] = None,
    featured: Optional[Callable] = None,
    mark: Optional[Callable] = None,
    clock: Optional[Callable] = None,
    mode: str = 'full'
) -> dict:
    """
    Compute the poll series of a scope in one of two modes.

    `full` is the sequence of `notebooks/data-load/PollsCompute.ipynb` without the ratings: weights, featured
    events, errors, deviations, drift, house effects and herding. The weights are computed twice because
    `polls.featured` copies the flag of the event, which in turn needs the weights; drift, house effects and
    herding read the errors stored by the previous phases, so the three run on a fresh series and only when `save`
    is set (a dry run stops after the deviations).

    `light` recomputes only what the polls of the current cycle can change: one pass of the weights over the whole
    scope (the overlap weight of a poll depends on its neighbours of the cycle, and the pass copies to every new poll
    the featured flag of its election) and, with `save` and a fresh series that reads those weights, the herding of
    every cycle. No featured events: `update_featured` needs an official result, and the pipeline only picks `light`
    when no pending election has one (`resolve_mode`). Errors, deviations, drift and house effects only change with a
    new result, data of past cycles, the catalogues or the method: they are left as stored. A dry run only weighs.

    With `save`, once every phase has finished the events of the scope are marked as computed (`computed_at`) with
    the instant the compute started, read from the database before the series is built: what is loaded while the
    compute runs did not enter the series and must stay pending. A failure in any phase propagates before the mark,
    so the scope stays pending. The mark is not a phase: it is not reported to `progress` nor added to `seconds`.
    The errors of the poll average are not a phase of either mode: see `average_errors_scope`.

    Parameters
    ----------
    scope : str
        Scope to compute.
    save : bool, default True
        Whether to write the results into the database.
    verbose : int, default 0
        Verbosity of the `Computer`.
    progress : callable, optional
        ``progress(step, seconds)`` reported after each phase (keys of `MODES[mode]`; `weights` twice in `full`).
    computer : callable, optional
        Factory with the signature ``Computer(scope=..., verbose=...)``; the real `Computer` by default.
    featured : callable, optional
        ``update_featured(scope) -> int``; the real one by default. Not called in `light`.
    mark : callable, optional
        ``mark_computed(scope, started_at)``; the real one by default. Not called without `save`.
    clock : callable, optional
        ``db_now() -> datetime`` that gives the instant the compute starts; the real one by default. Not called
        without `save`.
    mode : {'light', 'full'}, default 'full'
        Phases to run (see above).

    Returns
    -------
    dict
        ``n_polls`` (polls of the series), ``n_events`` (elections of the scope), ``featured`` (events marked as
        featured; ``None`` in `light` or without `save`), ``mode`` and ``seconds`` (per phase run and ``total``).

    Raises
    ------
    ValueError
        If `mode` is not ``light`` or ``full``, before anything is built.
    """
    if mode not in MODES:
        raise ValueError('compute: unknown mode "{}" (expected light or full)'.format(mode))

    computer = Computer if computer is None else computer
    featured = update_featured if featured is None else featured
    mark = mark_computed if mark is None else mark
    clock = db_now if clock is None else clock
    seconds = {}
    start = time.perf_counter()
    started_at = clock() if save else None

    def fresh():
        return computer(scope=scope, verbose=verbose).build_series()

    comp = fresh()
    _phase('weights', lambda: comp.compute_weights(save=save, overwrite=True), seconds, progress)
    n_featured = None

    if mode == 'full':
        if save:
            n_featured = _phase('featured', lambda: featured(scope), seconds, progress)

        comp = fresh()
        _phase('weights', lambda: comp.compute_weights(save=save, overwrite=True), seconds, progress)
        _phase('errors', lambda: comp.compute_errors(save=save), seconds, progress)
        _phase('deviations', lambda: comp.compute_deviations(save=save), seconds, progress)

    if save:
        comp = fresh()
        if mode == 'full':
            _phase('drift', lambda: comp.compute_drift(save=save), seconds, progress)
            _phase('house_effects', lambda: comp.compute_house_effects(save=save), seconds, progress)
        _phase('herding', lambda: comp.compute_herding(save=save), seconds, progress)
        mark(scope, started_at)

    seconds['total'] = time.perf_counter() - start

    return {
        'n_polls': int(comp.series['pollster'].notnull().sum()),
        'n_events': len(comp.event_dates),
        'featured': n_featured,
        'mode': mode,
        'seconds': seconds
    }


def average_errors_scope(
    scope: str,
    save: bool = True,
    verbose: int = 0,
    progress: Optional[Callable] = None,
    computer: Optional[Callable] = None
) -> dict:
    """
    Compute the errors of the poll average of the past elections of a scope (`Computer.compute_average_errors`:
    one fit of the average per featured election with a result and per horizon, the slowest phase of all), the
    table that only the uncertainty options of method 15 read (`error_model='average'`, `drift_scale='auto'`).

    An action of its own, not a phase of `compute_scope`: its Forecasters take the prior of the house effects from
    the table stored in the database, so run it after a `full` compute of the scope. It never marks `computed_at`
    (it computes nothing pending). Without `save` it computes without saving, as `rate_scope`.

    Parameters
    ----------
    scope : str
        Scope to compute.
    save : bool, default True
        Whether to write the table into the database.
    verbose : int, default 0
        Verbosity of the `Computer`.
    progress : callable, optional
        ``progress(step, seconds)`` reported once, with `AVERAGE_ERRORS_STEP`.
    computer : callable, optional
        Factory with the signature ``Computer(scope=..., verbose=...)``; the real `Computer` by default.

    Returns
    -------
    dict
        ``n_rows`` (rows of the table: election, party and horizon), ``n_events`` (elections in it) and
        ``seconds`` (``average_errors`` and ``total``).
    """
    computer = Computer if computer is None else computer
    seconds = {}
    start = time.perf_counter()

    comp = computer(scope=scope, verbose=verbose).build_series()
    df = _phase(AVERAGE_ERRORS_STEP, lambda: comp.compute_average_errors(save=save), seconds, progress)
    seconds['total'] = time.perf_counter() - start

    return {
        'n_rows': int(df.shape[0]),
        'n_events': int(df['event_date'].nunique()) if df.shape[0] > 0 else 0,
        'seconds': seconds
    }


def check_mode(mode: str) -> str:
    """
    Validate the mode asked to the pipeline or a job: ``auto`` (`resolve_mode` picks), ``light`` or ``full``.

    Parameters
    ----------
    mode : str
        Mode to check.

    Returns
    -------
    str
        The same mode.

    Raises
    ------
    ValueError
        If it is none of the three.
    """
    if mode not in (AUTO, *MODES):
        raise ValueError('compute: unknown mode "{}" (expected auto, light or full)'.format(mode))

    return mode


def resolve_mode(
    mode: str,
    overwrite: bool,
    pending: Optional[list[str]],
    results: Optional[list[str]]
) -> tuple[str, str]:
    """
    Pick the mode of the compute of a scope and the reason, the rule of the pipeline.

    A requested mode (``light`` or ``full``) stands. With ``auto``: `overwrite` is a full compute (everything is
    redone); otherwise ``full`` when a pending election has an official result (its errors, featured flag, drift
    and house effects have to be computed) and ``light`` when the pending elections are only cycles without a
    result (new polls). The rule never avoids the skip of a scope with nothing pending, which is up to `overwrite`
    only: with an empty `pending` and no `overwrite` it still answers ``light``, a case the pipeline never reaches.

    Parameters
    ----------
    mode : str
        ``auto``, ``light`` or ``full`` (see `check_mode`).
    overwrite : bool
        Whether the compute was asked for every scope, pending or not.
    pending : list of str, optional
        Dates of the pending elections of the scope (`pending_events`); not read with `overwrite` or a requested
        mode.
    results : list of str, optional
        Dates of the elections of the scope with an official result (`result_dates`); read only with ``auto``,
        without `overwrite`.

    Returns
    -------
    tuple of str
        ``(mode, reason)`` with the reason among ``requested``, ``overwrite``, ``results pending`` and
        ``polls pending``.

    Raises
    ------
    ValueError
        If `mode` is none of the three.
    """
    mode = check_mode(mode)
    if mode != AUTO:
        return mode, 'requested'
    if overwrite:
        return 'full', 'overwrite'
    if set(pending or ()) & set(results or ()):
        return 'full', 'results pending'

    return 'light', 'polls pending'


def rate_scope(
    scope: str,
    save: bool = True,
    verbose: int = 0,
    progress: Optional[Callable] = None,
    computer: Optional[Callable] = None
) -> dict:
    """
    Compute the pollster ratings of the events of a scope with the polls of every scope (see
    `Computer.compute_ratings` with ``scopes='all'``).

    Parameters
    ----------
    scope : str
        Scope whose events are rated.
    save : bool, default True
        Whether to write the ratings into the database.
    verbose : int, default 0
        Verbosity of the `Computer`.
    progress : callable, optional
        ``progress(step, seconds)`` reported once, with `RATINGS_STEP`.
    computer : callable, optional
        Factory with the signature ``Computer(scope=..., verbose=...)``; the real `Computer` by default.

    Returns
    -------
    dict
        ``n_pollsters`` and ``n_events`` of the ratings table (``comp.ratings``) and ``seconds``
        (``ratings`` and ``total``).
    """
    computer = Computer if computer is None else computer
    seconds = {}
    start = time.perf_counter()

    comp = computer(scope=scope, verbose=verbose).build_series()
    # `compute_ratings` returns the polls with their rating columns; the ratings table (indexed by
    # `event_date` and `pollster_id`) is left in `comp.ratings`
    _phase(RATINGS_STEP, lambda: comp.compute_ratings(save=save, scopes='all'), seconds, progress)
    seconds['total'] = time.perf_counter() - start

    return {
        'n_pollsters': int(comp.ratings.index.get_level_values('pollster_id').nunique()),
        'n_events': int(comp.ratings.index.get_level_values('event_date').nunique()),
        'seconds': seconds
    }


def replicate_catalogues(save: bool = True) -> dict:
    """
    Replicate the versioned catalogues into the database (`data.save_catalogues`): scopes, districts and the
    evaluations of the prior quality, from which `pollsters.quality` is derived. Run it after editing
    `data/pollsters-quality.csv` or `data/pollsters-quality-criteria.json`, before the ratings.

    Parameters
    ----------
    save : bool, default True
        Whether to write into the database; without it, only the rubric and the evaluations are read and
        validated (`quality.read_quality`), so a malformed file fails before anything is written.

    Returns
    -------
    dict
        The figures of `save_catalogues` (``scopes``, ``districts``, ``quality`` and ``pollsters``) or, without
        ``save``, only ``quality`` (the evaluations read); and ``seconds`` (``total``).
    """
    start = time.perf_counter()
    if save:
        figures = save_catalogues()
    else:
        figures = {'quality': int(read_quality(load_criteria()).shape[0])}

    return {**figures, 'seconds': {'total': time.perf_counter() - start}}
