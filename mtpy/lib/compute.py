"""
Computation of the poll series of a scope and of the pollster ratings: the sequence of
`notebooks/data-load/PollsCompute.ipynb`, shared by the `compute` job (`mtpy/jobs/Compute.py`) and the batch
loader of the regional scopes (`load/run_load.py`).

`compute_scope` runs weights, featured events, errors, deviations, drift, house effects and herding of one scope;
`rate_scope` the cross-scope pollster ratings of its events; `replicate_catalogues` the versioned catalogues
(scopes, districts and the prior quality of the pollsters), once for every scope. The first two report each phase
through an optional `progress(step, seconds)` callback; the three return the figures of the run, and a failure
propagates to the caller, which decides how to isolate it.
"""
import time
from typing import Callable, Optional

from .computer import Computer
from .data import save_catalogues, update_featured
from .quality import load_criteria, read_quality

# Phases of `compute_scope`, in order; the weights run twice (see `compute_scope`)
COMPUTE_STEPS = ('weights', 'featured', 'errors', 'deviations', 'drift', 'house_effects', 'herding')
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
    featured: Optional[Callable] = None
) -> dict:
    """
    Compute the poll series of a scope: weights, featured events, errors, deviations, drift, house effects and
    herding (the sequence of `notebooks/data-load/PollsCompute.ipynb` without the ratings).

    The weights are computed twice because `polls.featured` copies the flag of the event, which in turn needs
    the weights; drift, house effects and herding read the errors stored by the previous phases, so they run on a
    fresh series and only when `save` is set (a dry run stops after the deviations).

    Parameters
    ----------
    scope : str
        Scope to compute.
    save : bool, default True
        Whether to write the results into the database.
    verbose : int, default 0
        Verbosity of the `Computer`.
    progress : callable, optional
        ``progress(step, seconds)`` reported after each phase (keys of `COMPUTE_STEPS`; `weights` twice).
    computer : callable, optional
        Factory with the signature ``Computer(scope=..., verbose=...)``; the real `Computer` by default.
    featured : callable, optional
        ``update_featured(scope) -> int``; the real one by default.

    Returns
    -------
    dict
        ``n_polls`` (polls of the series), ``n_events`` (elections of the scope), ``featured`` (events marked as
        featured, ``None`` without `save`) and ``seconds`` (per phase and ``total``).
    """
    computer = Computer if computer is None else computer
    featured = update_featured if featured is None else featured
    seconds = {}
    start = time.perf_counter()

    def fresh():
        return computer(scope=scope, verbose=verbose).build_series()

    comp = fresh()
    _phase('weights', lambda: comp.compute_weights(save=save, overwrite=True), seconds, progress)
    n_featured = None
    if save:
        n_featured = _phase('featured', lambda: featured(scope), seconds, progress)

    comp = fresh()
    _phase('weights', lambda: comp.compute_weights(save=save, overwrite=True), seconds, progress)
    _phase('errors', lambda: comp.compute_errors(save=save), seconds, progress)
    _phase('deviations', lambda: comp.compute_deviations(save=save), seconds, progress)

    if save:
        comp = fresh()
        _phase('drift', lambda: comp.compute_drift(save=save), seconds, progress)
        _phase('house_effects', lambda: comp.compute_house_effects(save=save), seconds, progress)
        _phase('herding', lambda: comp.compute_herding(save=save), seconds, progress)

    seconds['total'] = time.perf_counter() - start

    return {
        'n_polls': int(comp.series['pollster'].notnull().sum()),
        'n_events': len(comp.event_dates),
        'featured': n_featured,
        'seconds': seconds
    }


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
