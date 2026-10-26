"""
Loading of the polls of an election from its Wikipedia article: the sequence of
`notebooks/data-load/PollsLoadWikipedia.ipynb`, shared by the `load` job (`mtpy/jobs/Load.py`).

`load_polls` reads the article, checks that every party, pollster and sponsor is mapped (with `force`, it adds the
unknown pollsters and sponsors to the catalogues), builds and selects the series, saves polls and results and
recomputes the weights of the event. Each phase is reported through an optional `progress(step, seconds)`
callback; known failures come back as a ``failed`` result, unexpected exceptions propagate to the caller.
"""
import json
import time
import warnings
from typing import Callable, Optional

import pandas as pd

from ..core.app import App
from ..models.elections import Pollsters, Sponsors
from .compute import _phase
from .computer import Computer
from .loader import WikipediaLoader
from .quality import load_criteria, quality_from_points

# Phases of `load_polls`, in order
LOAD_STEPS = ('read', 'names', 'series', 'save', 'weights')
# Classes of names that must be mapped, in the order of the failure reason
NAME_CLASSES = ('parties', 'pollsters', 'sponsors')
# Failure reason of an event without a polls article in `wp-urls.json` (formatted with the event date)
NO_POLLS_ARTICLE = 'no polls article for {}'
# Refusal of `years` with `overwrite`: the overwrite deletes the whole cycle and reinserts only the years read
YEARS_WITH_OVERWRITE = ('load: "years" cannot be combined with "overwrite" '
                        '(the other years of the cycle would be deleted)')


def has_polls_article(scope: str, event_date: str, urls: Optional[dict] = None) -> bool:
    """
    Whether `wp-urls.json` lists a polls article for the event.

    Parameters
    ----------
    scope : str
        Scope of the event.
    event_date : str
        Date of the event (`YYYY-MM-DD`).
    urls : dict, optional
        Content of `wikipedia/wp-urls.json`; read through the app data store by default.

    Returns
    -------
    bool
        True when ``urls[scope][event_date]`` exists and has a ``polls`` entry.
    """
    if urls is None:
        urls = json.loads(App.get_().data.read('wikipedia/wp-urls.json'))

    return 'polls' in urls.get(scope, {}).get(event_date, {})


def unevaluated_quality() -> float:
    """Prior quality of a pollster without evaluation (the floor of the quality scale)."""
    return quality_from_points(0, load_criteria()['scale'])


def add_catalogue_entries(pollsters: list[str], sponsors: list[str]) -> dict[str, list[str]]:
    """
    Insert new pollsters (without mtype, with the unevaluated quality) and sponsors into the database, keeping the
    existing rows.

    Parameters
    ----------
    pollsters, sponsors : list of str
        Names to add.

    Returns
    -------
    dict
        ``pollsters`` and ``sponsors``: the names inserted (empty lists when there was nothing to add).
    """
    if len(pollsters) > 0:
        quality = unevaluated_quality()
        Pollsters().insert(pd.DataFrame({
            'id': [None] * len(pollsters),
            'name': pollsters,
            'mtype': [None] * len(pollsters),
            'quality': [quality] * len(pollsters),
            'rating': [None] * len(pollsters)
        }), truncate=False)
    if len(sponsors) > 0:
        Sponsors().insert(pd.DataFrame({'id': [None] * len(sponsors), 'name': sponsors}), truncate=False)

    return {'pollsters': list(pollsters), 'sponsors': list(sponsors)}


def unmapped_reason(missing: dict[str, list[str]]) -> str:
    """
    Failure reason of the names without id.

    Parameters
    ----------
    missing : dict
        Names by class (`parties`, `pollsters`, `sponsors`); absent or empty classes are left out.

    Returns
    -------
    str
        E.g. ``'unmapped parties: X; unmapped pollsters: Y, Z'``, with the names sorted.
    """
    return '; '.join(
        'unmapped {}: {}'.format(cls, ', '.join(sorted(missing[cls]))) for cls in NAME_CLASSES if missing.get(cls)
    )


def _names(loader, kind: str) -> dict[str, list[str]]:
    """Sorted, de-duplicated names by class that the loader read as `missing` or `skipped`."""
    return {cls: sorted(set(getattr(loader, '{}_{}'.format(cls, kind)))) for cls in NAME_CLASSES}


def load_polls(
    scope: str,
    event_date: str,
    overwrite: bool = False,
    force: bool = False,
    years: Optional[list[str]] = None,
    save: bool = True,
    verbose: int = 0,
    progress: Optional[Callable[[str, float], None]] = None,
    report: Optional[Callable[[str, int], None]] = None,
    loader: Optional[Callable] = None,
    computer: Optional[Callable] = None,
    catalogues: Optional[Callable] = None,
    urls: Optional[dict] = None
) -> dict:
    """
    Load the polls and results of an election from its Wikipedia article.

    Parameters
    ----------
    scope : str
        Scope of the election.
    event_date : str
        Date of the election (`YYYY-MM-DD`); it must have a polls article in `wp-urls.json`.
    overwrite : bool, default False
        Replace the polls of the cycle instead of adding only the new ones.
    force : bool, default False
        Add the unknown pollsters and sponsors to the catalogues (a missing party always fails).
    years : list of str, optional
        Restrict the polls to these years (as ``'2026'``); not allowed together with ``overwrite``.
    save : bool, default True
        Write polls, results and weights into the database; without it, nothing is written (a dry run).
    verbose : int, default 0
        Verbosity of the loader and the `Computer`.
    progress : callable, optional
        ``progress(step, seconds)`` reported after each phase (keys of `LOAD_STEPS`).
    report : callable, optional
        ``report(text, level)`` receives the lines of the names block (level 1: counts and missing names; level 2:
        skipped names and the parties checks). Nothing is printed without it.
    loader : callable, optional
        Factory with the signature of `WikipediaLoader`; the real one by default.
    computer : callable, optional
        Factory with the signature ``Computer(scope=..., event_dates=..., verbose=..., path=...)``.
    catalogues : callable, optional
        ``catalogues(pollsters, sponsors)`` adding the new names (`add_catalogue_entries` by default).
    urls : dict, optional
        Content of `wp-urls.json`; read through the app data store by default.

    Returns
    -------
    dict
        ``status`` (`loaded`, `checked` for a dry run or `failed`), ``action`` (`polls`), ``event_date``,
        ``n_read`` (polls in the article), ``n_saved`` (polls selected), ``overwrite``, ``missing`` and ``skipped``
        (names by class), ``added`` (pollsters and sponsors added, or that `force` would add in a dry run),
        ``seconds`` (per phase and ``total``) and, when failed, ``reason``.

    Raises
    ------
    ValueError
        If ``years`` and ``overwrite`` are both given: the overwrite would delete the other years of the cycle.
    """
    if years and overwrite:
        raise ValueError(YEARS_WITH_OVERWRITE)

    loader = WikipediaLoader if loader is None else loader
    computer = Computer if computer is None else computer
    catalogues = add_catalogue_entries if catalogues is None else catalogues
    seconds, start = {}, time.perf_counter()
    result = {
        'status': 'loaded', 'action': 'polls', 'event_date': event_date, 'n_read': 0, 'n_saved': 0,
        'overwrite': overwrite, 'missing': {cls: [] for cls in NAME_CLASSES},
        'skipped': {cls: [] for cls in NAME_CLASSES}, 'added': {'pollsters': [], 'sponsors': []}, 'seconds': seconds
    }

    def finish(status: str, reason: Optional[str] = None) -> dict:
        result['status'] = status
        if reason is not None:
            result['reason'] = reason
        seconds['total'] = time.perf_counter() - start
        return result

    def say(text: str, level: int):
        if report is not None:
            report(text, level)

    if not has_polls_article(scope, event_date, urls):
        return finish('failed', NO_POLLS_ARTICLE.format(event_date))

    # The loader reads the article and the catalogues of names on construction and in `read_data`
    def read():
        wiki = loader(scope, event_date, years=years, verbose=verbose)
        return wiki.read_data()

    wiki = _phase('read', read, seconds, progress)

    def check_names():
        nonlocal wiki
        result['n_read'] = len(wiki.data)
        result['missing'], result['skipped'] = _names(wiki, 'missing'), _names(wiki, 'skipped')
        missing, skipped = result['missing'], result['skipped']
        say('# Polls: {}'.format(result['n_read']), 1)
        say('', 1)
        for cls in NAME_CLASSES:
            say('Missing {}: {}'.format(cls.capitalize(), set(missing[cls])), 1)
        for cls in NAME_CLASSES:
            if skipped[cls]:
                say('Skipped {}: {}'.format(cls, set(skipped[cls])), 2)

        if missing['parties'] or (not force and (missing['pollsters'] or missing['sponsors'])):
            return unmapped_reason(missing)
        if force and (missing['pollsters'] or missing['sponsors']):
            if save:
                result['added'] = catalogues(missing['pollsters'], missing['sponsors'])
                for name in result['added']['pollsters']:
                    warnings.warn('new pollster «{}» without mtype: check it'.format(name))
                wiki.reload_catalogues()
                wiki.read_data(pages=wiki.pages)
                result['n_read'] = len(wiki.data)
                result['missing'] = _names(wiki, 'missing')
                if any(result['missing'].values()):
                    return 'unmapped after adding: ' + unmapped_reason(result['missing'])
            else:
                result['added'] = {'pollsters': missing['pollsters'], 'sponsors': missing['sponsors']}

        return None

    reason = _phase('names', check_names, seconds, progress)
    if reason is not None:
        return finish('failed', reason)
    if result['n_read'] == 0:
        return finish('loaded' if save else 'checked')

    def build():
        wiki.build_series()
        if wiki.fixes_missing:
            warnings.warn('Poll fixes without a poll read: {}'.format(', '.join(wiki.fixes_missing)))

    _phase('series', build, seconds, progress)

    def select_and_save():
        wiki.select_series(overwrite)
        result['n_saved'] = len(wiki.polls)
        if save:
            wiki.save_polls(overwrite)
            wiki.save_results(overwrite)
            # The checks query the database: a failure there must not fail a save already done nor skip the weights
            try:
                checks = wiki.parties_checks.to_string()
            except Exception as e:
                warnings.warn('parties checks failed: {}: {}'.format(type(e).__name__, e))
            else:
                say(checks, 2)

    _phase('save', select_and_save, seconds, progress)

    if save and (result['n_saved'] > 0 or overwrite):
        def weights():
            computer(scope=scope, event_dates=[event_date], verbose=verbose, path='.').build_series() \
                .compute_weights(save=True, overwrite=True)

        _phase('weights', weights, seconds, progress)

    return finish('loaded' if save else 'checked')
