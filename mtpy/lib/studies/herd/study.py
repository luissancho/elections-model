"""
The herd study in one call: `compute_herd` reads the published elections and the one in progress through the bundle
reader (`data`) and chains the measures of `stats`: for every election, the dispersion of the same pairs of houses far
from the election, in the middle of the cycle and in its last weeks, and how it changes between them. `HerdStudy`
carries the results; this module only reads and wires, the figures come from `stats`.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from ...bundle import BundleReader
from ....core.io import FileSystem
from .data import Election, MissingInput, load_params, read_current, read_elections
from .stats import _sig3, by_buckets, comparisons, compare, count, permutation_p

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
        ``comparisons`` (the ``count`` of each comparison with its own ``since_year``) and ``kinds`` (the ``count`` of
        ``far_final`` by kind of election).
    elections : list of dict
        One row per past election and comparison with a result (see ``_row``), in the order of the elections.
    weeks : dict
        ``buckets`` (``{lo, hi, label}``, farthest from the election first), ``rows`` (one per election and bucket,
        the election in progress last: ``event_date``, ``label``, ``current``, ``lo``, ``hi``, ``pairs``,
        ``comparisons`` and ``ratio`` at two decimals) and ``mean`` (per bucket: ``lo``, ``hi``, ``k`` past
        elections with a ratio and their mean ``ratio``).
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

    for election in elections:
        all_rows[election.event_date, election.scope] = comparisons(election.polls, election.names, params)
        results = _results(election, params, all_rows[election.event_date, election.scope])

        if results:
            counted.append(election)

        for key, result in results.items():
            rows.append(_row(election, result) | {'kind': election.kind, 'comparison': key})
            raw[key].append(result)
            dates[key].append(election.event_date)

            if key == 'far_final':
                raw_kinds[election.kind].append(result)

    summary = {'n_elections': len(counted), 'n_scopes': len({election.scope for election in counted}),
               'since_year': _since_year([election.event_date for election in counted]),
               'comparisons': {key: _count(results) | {'since_year': _since_year(dates[key])}
                               for key, results in raw.items()},
               'kinds': {kind: _count(results) for kind, results in raw_kinds.items()}}

    current_rows = None if current is None else comparisons(current.polls, current.names, params)
    weeks = _weeks(elections, all_rows, current, current_rows, params)

    return HerdStudy(params=params, election=None if current is None else _election(current), summary=summary,
                     elections=rows, weeks=weeks,
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
