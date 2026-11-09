"""
The herd study in one call: `compute_herd` reads the published elections and the one in progress through the bundle
reader (`data`) and chains the measures of `stats`: for every election, the dispersion of the same pairs of houses far
from the election, in the middle of the cycle and in its last weeks, and how it changes between them. `HerdStudy`
carries the results; this module only reads and wires, the figures come from `stats`.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ...bundle import BundleReader
from ....core.io import FileSystem
from .data import Election, MissingInput, load_params, read_current, read_elections
from .stats import (_sig3, bootstrap_median, bootstrap_pooled, by_buckets, comparisons, compare, count, histogram,
                    null_distribution, permutation_p, pooled, pooled_p, pooled_terms)

# Each comparison: the period taken as reference and the one expected to be closer
COMPARISONS = {'far_final': ('far', 'final'), 'far_mid': ('far', 'mid'), 'mid_final': ('mid', 'final')}
KINDS = ('generales', 'autonómicas')


@dataclass
class HerdStudy:
    """
    Results of the herd study.

    Attributes
    ----------
    params : dict
        Parameters of the study, ``data/rebano/params.json`` updated with the overrides.
    election : dict or None
        ``{scope, date, days_left, as_of}`` of the election in progress; ``None`` without one.
    summary : dict
        ``n_elections``, ``n_scopes``, ``since_year`` (the first year of the elections that enter any comparison),
        ``comparisons`` (the ``count`` of each comparison with its own ``since_year``, its ``pooled`` result over all
        the elections, ``{elections, pairs, rows_a, rows_b, r_a, r_b, change, p}``, and the bootstrap interval ``ci``
        ``{lo, hi}`` of the median change, ``None`` with fewer than 3 elections), ``kinds`` (the ``count`` of
        ``far_final`` by kind of election) and ``robustness`` (``pair_days`` and ``min_rows``, the values tried, and
        ``rows``: one per ``pair_days``, ``min_rows`` and comparison, in that order, with ``default``, ``n``, ``k``,
        ``median_change``, ``change`` and ``p``; a cell without elections has ``n`` and ``k`` 0 and ``None`` in the
        rest).
    elections : list of dict
        One row per past election and comparison with a result (see ``_row``), in the order of the elections.
    weeks : dict
        ``buckets`` (``{lo, hi, label}``, farthest from the election first), ``rows`` (one per election and bucket,
        the election in progress last: ``event_date``, ``label``, ``current``, ``lo``, ``hi``, ``pairs``,
        ``comparisons`` and ``ratio`` at two decimals) and ``mean`` (per bucket: ``lo``, ``hi``, ``k`` past
        elections with a ratio and their mean ``ratio``).
    chance : dict or None
        The null-hypothesis test of the last weeks against the rest of the cycle, over the past elections with
        pairs counted in the ``chance`` periods (``final`` and ``rest``, replacing the study's): ``periods``,
        ``permutations``, ``elections``, ``falls`` (elections where ``final`` is closer), ``pairs``, ``rows_a`` and
        ``rows_b`` (``rest`` and ``final``), ``r_a``, ``r_b`` (two decimals), ``change`` and ``ci`` (``{lo, hi}`` of
        the bootstrap, three decimals; ``None`` with fewer than 3 elections), ``count_one`` and ``count_two`` (shuffles
        as extreme as the observed one, one and two sided), ``p_one`` and ``p_two`` (three significant figures) and
        ``null`` (``mean``, ``sd``, ``min`` and ``max`` of the shuffled changes, four decimals, and the ``bins`` of
        their ``histogram`` of ``bin_width``); ``None`` when no election has pairs counted.
    current : dict or None
        The election in progress: ``scope``, ``event_date``, ``label``, ``days_left``, ``as_of``, ``windows`` and
        ``far_mid`` and ``mid_final`` (a row without ``kind`` and ``comparison``, or ``None``).
    inputs : dict
        ``{'elections': ..., 'current': ...}`` as returned by ``read_elections`` and ``read_current``.
    """
    params: dict
    election: Optional[dict]
    summary: dict
    elections: list
    weeks: dict
    chance: Optional[dict]
    current: Optional[dict]
    inputs: dict


def _results(election: Election, params: dict, rows: Optional[pd.DataFrame] = None) -> dict:
    """
    The results of an election in each comparison.

    Parameters
    ----------
    election : Election
        The election.
    params : dict
        Parameters of the study.
    rows : pandas.DataFrame, optional
        The ``comparisons`` of the election, when already computed.

    Returns
    -------
    dict
        By key of ``COMPARISONS``, the unrounded ``compare`` result with its ``p`` (``permutation_p``, three
        significant figures); the comparisons without result are left out.
    """
    if rows is None:
        rows = comparisons(election.polls, election.names, params)

    results = {}

    for key, (a, b) in COMPARISONS.items():
        result = compare(rows, a, b, params['min_rows'])

        if result is None:
            continue

        p = permutation_p(rows, a, b, params['min_rows'], params['permutations'], params['seed'])
        results[key] = result | {'p': _sig3(p)}

    return results


def _row(election: Election, result: dict) -> dict:
    """The published figures of a result: ratios at two decimals, change at three (the page writes a change below
    half a point with one decimal), ``falls`` and ``p`` as computed."""
    return {'scope': election.scope, 'event_date': election.event_date, 'label': election.label,
            'pairs': result['pairs'], 'rows_a': result['rows_a'], 'rows_b': result['rows_b'],
            'r_a': round(result['r_a'], 2), 'r_b': round(result['r_b'], 2), 'change': round(result['change'], 3),
            'falls': result['falls'], 'p': result['p']}


def _count(results: list) -> dict:
    """``count`` of unrounded results with the median change at two decimals."""
    tally = count(results)

    if tally['median_change'] is not None:
        tally['median_change'] = round(tally['median_change'], 2)

    return tally


def _pooled_block(rows_by_election: list, key: str, params: dict, changes: list) -> dict:
    """
    The pooled result and the interval of the median change of a comparison.

    Parameters
    ----------
    rows_by_election : list of pandas.DataFrame
        The ``comparisons`` of the elections with a result in the comparison ``key``.
    key : str
        Key of ``COMPARISONS``.
    params : dict
        Parameters of the study; uses ``min_rows``, ``permutations``, ``resamples`` and ``seed``.
    changes : list of float
        The unrounded change of each of those elections.

    Returns
    -------
    dict
        ``pooled`` (ratios at two decimals, change at three, ``p`` at three significant figures; ``None`` without
        pairs) and ``ci`` (``{lo, hi}`` at three decimals; ``None`` with fewer than 3 changes).
    """
    a, b = COMPARISONS[key]
    result = pooled(rows_by_election, a, b, params['min_rows'])
    interval = bootstrap_median(changes, params['resamples'], params['seed'])

    if result is not None:
        p = pooled_p(rows_by_election, a, b, params['min_rows'], params['permutations'], params['seed'])
        result = {'elections': result['elections'], 'pairs': result['pairs'], 'rows_a': result['rows_a'],
                  'rows_b': result['rows_b'], 'r_a': round(result['r_a'], 2), 'r_b': round(result['r_b'], 2),
                  'change': round(result['change'], 3), 'p': _sig3(p)}

    return {'pooled': result, 'ci': None if interval is None else {k: round(v, 3) for k, v in interval.items()}}


def _robustness(elections: list, params: dict) -> dict:
    """
    The main results over a grid of ``pair_days`` and ``min_rows``.

    Parameters
    ----------
    elections : list of Election
        The past elections.
    params : dict
        Parameters of the study; ``robustness`` holds the ``pair_days`` and ``min_rows`` values of the grid.

    Returns
    -------
    dict
        ``pair_days`` and ``min_rows`` (the grid values) and ``rows``: for each ``pair_days`` (outer), ``min_rows``
        (inner) and comparison, ``{pair_days, min_rows, default, comparison, n, k, median_change, change, p}`` with
        the median change and the pooled change at three decimals and ``p`` the pooled one (three significant
        figures); a cell without elections has ``n`` and ``k`` 0 and ``None`` in the rest.
    """
    grid = params['robustness']
    rows = []

    for pair_days in grid['pair_days']:
        cell_rows = [comparisons(election.polls, election.names, params | {'pair_days': pair_days})
                     for election in elections]

        for min_rows in grid['min_rows']:
            for key, (a, b) in COMPARISONS.items():
                compared = [(r, compare(r, a, b, min_rows)) for r in cell_rows]
                used = [r for r, result in compared if result is not None]
                # ``count`` gets ``p`` 1.0 so that ``clear`` is 0: the grid does not run per-election permutations
                tally = count([result | {'p': 1.0} for _, result in compared if result is not None])
                result = pooled(used, a, b, min_rows)
                p = None if result is None else _sig3(
                    pooled_p(used, a, b, min_rows, params['permutations'], params['seed']))
                median = tally['median_change']
                rows.append({'pair_days': pair_days, 'min_rows': min_rows,
                             'default': pair_days == params['pair_days'] and min_rows == params['min_rows'],
                             'comparison': key, 'n': tally['n'], 'k': tally['k'],
                             'median_change': None if median is None else round(median, 3),
                             'change': None if result is None else round(result['change'], 3), 'p': p})

    return {'pair_days': list(grid['pair_days']), 'min_rows': list(grid['min_rows']), 'rows': rows}


def _chance(elections: list, params: dict) -> Optional[dict]:
    """
    The null-hypothesis test of the last weeks against the rest of the cycle.

    Parameters
    ----------
    elections : list of Election
        The past elections.
    params : dict
        Parameters of the study; ``chance`` holds ``periods`` (``final`` and ``rest``, replacing the study's),
        ``permutations``, ``bin_width`` and ``resamples``; also uses ``min_rows`` and ``seed``.

    Returns
    -------
    dict or None
        As described in ``HerdStudy.chance``; ``None`` when no election has pairs counted.
    """
    config = params['chance']
    chance_params = params | {'periods': config['periods']}
    min_rows = params['min_rows']
    used = []
    falls = 0

    for election in elections:
        rows = comparisons(election.polls, election.names, chance_params)
        result = compare(rows, 'rest', 'final', min_rows)

        if result is not None:
            used.append(rows)
            falls += result['falls']

    result = pooled(used, 'rest', 'final', min_rows)

    if result is None:
        return None

    interval = bootstrap_pooled(pooled_terms(used, 'rest', 'final', min_rows), config['resamples'], params['seed'])
    test = null_distribution(used, 'rest', 'final', min_rows, config['permutations'], params['seed'])
    permuted = test['permuted']
    finite = permuted[np.isfinite(permuted)]

    return {'periods': {name: list(bounds) for name, bounds in config['periods'].items()},
            'permutations': config['permutations'], 'elections': result['elections'], 'falls': falls,
            'pairs': result['pairs'], 'rows_a': result['rows_a'], 'rows_b': result['rows_b'],
            'r_a': round(result['r_a'], 2), 'r_b': round(result['r_b'], 2), 'change': round(result['change'], 3),
            'ci': None if interval is None else {k: round(v, 3) for k, v in interval.items()},
            'count_one': int(test['count_one']), 'count_two': int(test['count_two']),
            'p_one': _sig3(test['p_one']), 'p_two': _sig3(test['p_two']),
            'null': {'mean': round(float(finite.mean()), 4), 'sd': round(float(finite.std()), 4),
                     'min': round(float(finite.min()), 4), 'max': round(float(finite.max()), 4),
                     'bin_width': config['bin_width'], 'bins': histogram(permuted, config['bin_width'])}}


def _since_year(dates: list) -> Optional[str]:
    """The year of the first of ``dates`` (ISO), ``None`` without dates."""
    return min(dates)[:4] if dates else None


def compute_herd(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> HerdStudy:
    """
    The herd study over the published bundle.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    data : FileSystem
        The ``data/`` directory of the repository (parameters and the scopes catalogue ``es-scopes.csv``, for the
        names of the regional scopes in the labels).
    params : dict, optional
        Overrides of the parameters of ``data/rebano/params.json``.

    Returns
    -------
    HerdStudy
        The results.

    Raises
    ------
    MissingInput
        ``events``: no past election has the ``cycle`` and ``results`` parts.
    """
    params = load_params(data) | (params or {})
    names = data.read_csv('es-scopes.csv').set_index('scode')['name'].to_dict()
    elections, inputs_elections = read_elections(reader, params, names)

    if not elections:
        raise MissingInput('events')

    current, inputs_current = read_current(reader, params, names)
    rows = []
    raw = {key: [] for key in COMPARISONS}
    dates = {key: [] for key in COMPARISONS}
    raw_kinds = {kind: [] for kind in KINDS}
    counted = []
    all_rows = {}
    pooled_rows = {key: [] for key in COMPARISONS}

    for election in elections:
        all_rows[election.event_date, election.scope] = comparisons(election.polls, election.names, params)
        results = _results(election, params, all_rows[election.event_date, election.scope])

        if results:
            counted.append(election)

        for key, result in results.items():
            rows.append(_row(election, result) | {'kind': election.kind, 'comparison': key})
            raw[key].append(result)
            dates[key].append(election.event_date)
            pooled_rows[key].append(all_rows[election.event_date, election.scope])

            if key == 'far_final':
                raw_kinds[election.kind].append(result)

    summary = {'n_elections': len(counted), 'n_scopes': len({election.scope for election in counted}),
               'since_year': _since_year([election.event_date for election in counted]),
               'comparisons': {key: _count(results) | {'since_year': _since_year(dates[key])}
                               | _pooled_block(pooled_rows[key], key, params, [r['change'] for r in results])
                               for key, results in raw.items()},
               'kinds': {kind: _count(results) for kind, results in raw_kinds.items()},
               'robustness': _robustness(elections, params)}

    current_rows = None if current is None else comparisons(current.polls, current.names, params)
    weeks = _weeks(elections, all_rows, current, current_rows, params)

    return HerdStudy(params=params, election=None if current is None else _election(current), summary=summary,
                     elections=rows, weeks=weeks, chance=_chance(elections, params),
                     current=None if current is None else _current(current, params, current_rows),
                     inputs={'elections': inputs_elections, 'current': inputs_current})


def _weeks(elections: list, all_rows: dict, current: Optional[Election], current_rows: Optional[pd.DataFrame],
           params: dict) -> dict:
    """
    The dispersion by bucket of weeks to the election, per election and on average.

    Parameters
    ----------
    elections : list of Election
        The past elections.
    all_rows : dict
        ``comparisons`` of each past election, by ``(event_date, scope)``.
    current : Election or None
        The election in progress.
    current_rows : pandas.DataFrame or None
        The ``comparisons`` of the election in progress.
    params : dict
        Parameters of the study; uses ``weeks`` (``scope``, ``since``, ``min_pairs`` and ``buckets``).

    Returns
    -------
    dict
        ``buckets``, ``rows`` and ``mean`` as described in ``HerdStudy.weeks``; the elections of another scope or
        before ``since`` are left out.
    """
    config = params['weeks']
    buckets = [[lo, hi] for lo, hi in reversed(config['buckets'])]
    chosen = [(election, all_rows[election.event_date, election.scope]) for election in elections
              if election.scope == config['scope']
              and (config['since'] is None or election.event_date >= config['since'])]

    if current is not None and current.scope == config['scope']:
        chosen.append((current, current_rows))

    rows = []
    ratios = {(lo, hi): [] for lo, hi in buckets}

    for election, election_rows in chosen:
        for entry in by_buckets(election_rows, buckets, config['min_pairs']):
            ratio = entry['ratio']
            rows.append({'event_date': election.event_date, 'label': election.label, 'current': election.current,
                         'lo': entry['lo'], 'hi': entry['hi'], 'pairs': entry['pairs'],
                         'comparisons': entry['comparisons'], 'ratio': None if ratio is None else round(ratio, 2)})

            if ratio is not None and not election.current:
                ratios[entry['lo'], entry['hi']].append(ratio)

    mean = [{'lo': lo, 'hi': hi, 'k': len(values),
             'ratio': round(sum(values) / len(values), 2) if values else None}
            for (lo, hi), values in ratios.items()]

    return {'buckets': [{'lo': lo, 'hi': hi, 'label': f'{lo}-{hi}'} for lo, hi in buckets], 'rows': rows, 'mean': mean}


def _days_left(current: Election) -> Optional[int]:
    """Days from the ``as_of`` of the run to the election; ``None`` without ``as_of``."""
    if current.as_of is None:
        return None

    return (date.fromisoformat(current.event_date) - date.fromisoformat(current.as_of)).days


def _election(current: Election) -> dict:
    """``{scope, date, days_left, as_of}`` of the election in progress."""
    return {'scope': current.scope, 'date': current.event_date, 'days_left': _days_left(current),
            'as_of': current.as_of}


def _current(current: Election, params: dict, rows: Optional[pd.DataFrame] = None) -> dict:
    """
    The election in progress with what the study can say of it so far.

    Parameters
    ----------
    current : Election
        The election in progress.
    params : dict
        Parameters of the study.
    rows : pandas.DataFrame, optional
        The ``comparisons`` of the election, when already computed.

    Returns
    -------
    dict
        ``scope``, ``event_date``, ``label``, ``days_left``, ``as_of``, ``windows`` (the first day of the ``mid``
        and ``final`` periods, ISO) and ``far_mid`` and ``mid_final`` (a row, or ``None``).
    """
    event = pd.Timestamp(current.event_date)
    windows = {name: (event - pd.Timedelta(days=params['periods'][name][1] - 1)).strftime('%Y-%m-%d')
               for name in ('mid', 'final')}
    results = _results(current, params, rows)
    out = {'scope': current.scope, 'event_date': current.event_date, 'label': current.label,
           'days_left': _days_left(current), 'as_of': current.as_of, 'windows': windows}

    for key in ('far_mid', 'mid_final'):
        out[key] = _row(current, results[key]) if key in results else None

    return out
