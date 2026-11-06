"""
The herd study in one call: `compute_herd` reads the published parts through the bundle reader (`data`) and chains
the measures of `stats` for every general-election cycle since ``params['since']`` and for the cycle in progress: the
dispersion of the polls around the average in the rest of the cycle, the final window and the last weeks, by weeks to
the election, the herding of each house in the final window, and the summary over the past cycles. `HerdStudy` carries
the results; this module only reads and wires, every figure comes from `data` and `stats`.

Ratios enter the rows rounded to 2 decimals; the means over cycles use the unrounded ratios and round last, while
the verdicts (``narrowed``, ``below_one``, ``rank_before``) compare the rounded ratios, the ones the page prints, so a
printed 1,00 is never «below 1». The houses of ``params['exclude']`` (the CIS) stay out of every pooled figure and are
measured apart.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ...bundle import BundleReader
from ....core.io import FileSystem
from .data import Cycle, MissingInput, load_params, read_current, read_cycles, read_herding
from .stats import accuracy, dispersion, house_ratio, in_days, main_series, residual_rows, sign_test


@dataclass
class HerdStudy:
    """
    Results of the herd study, the blocks of the published parts.

    Attributes
    ----------
    params : dict
        Parameters of the study, ``data/rebano/params.json`` updated with the overrides.
    election : dict or None
        ``{date, days_left, as_of}`` of the cycle in progress; ``None`` without it.
    summary : dict
        ``n_cycles``, ``k_narrowed``, ``narrowed``, ``mean_before``, ``mean_last``, ``mean_final``, ``k_below_one``,
        ``below_one``, ``sign``, ``current``, ``houses_now``, ``cis`` and ``accuracy`` (``None`` the four last but
        ``accuracy`` without a cycle in progress; ``accuracy`` ``None`` when a group is empty).
    cycles : list of dict
        One row per cycle, past ascending then the current one: ``event_date``, ``label``, ``current``, ``names``,
        ``consensus``, ``polls``, ``with_sample``, ``houses``, ``before``, ``last``, ``final``, ``change`` and ``cis``.
        ``polls`` counts every poll ending 0 to ``max_days`` days before the election; ``with_sample`` those of them
        with sample and weight above zero outside ``exclude``, and ``houses`` their distinct houses.
    weeks : dict
        ``buckets`` (``[{lo, hi, label}]``, farthest first), ``rows`` (``[{event_date, label, current, lo, hi, k,
        ratio}]`` per cycle and bucket) and ``mean`` (``[{lo, hi, k, ratio}]`` over the past cycles).
    houses : list of dict
        One row per cycle and house with ``min_polls`` polls or more in the final window: ``event_date``, ``label``,
        ``current``, ``pollster``, ``k``, ``h``, ``p``, ``cis``, ``h_cycle``, ``last_error_avg`` and
        ``last_error_blocks``; by cycle, then ``h`` ascending with ``None`` last.
    inputs : dict
        ``cycles`` (inputs of ``read_cycles`` by date), ``current`` (of ``read_current``) and ``herding`` (of
        ``read_herding``).
    """

    params: dict
    election: Optional[dict]
    summary: dict
    cycles: list
    weeks: dict
    houses: list
    inputs: dict


def _round(value: Optional[float]) -> Optional[float]:
    """``value`` rounded to 2 decimals, ``None`` kept."""
    return None if value is None else round(float(value), 2)


def _rounded(measure: dict) -> dict:
    """A ``{k, ratio}`` of ``dispersion(..., round_to=None)`` with its ratio rounded to 2 decimals."""
    return {'k': measure['k'], 'ratio': _round(measure['ratio'])}


def _mean(values: list) -> Optional[float]:
    """Plain mean of the values that are not ``None``, unrounded; ``None`` without any."""
    values = [value for value in values if value is not None]

    return float(np.mean(values)) if values else None


def _windows(params: dict) -> dict:
    """The day windows of the study: ``before`` (``window_days`` to ``max_days``), ``last`` (0 to
    ``window_days − 1``), ``final`` (0 to ``final_days − 1``) and ``cycle`` (0 to ``max_days``)."""
    return {'before': (params['window_days'], params['max_days']), 'last': (0, params['window_days'] - 1),
            'final': (0, params['final_days'] - 1), 'cycle': (0, params['max_days'])}


def _buckets(params: dict) -> list:
    """The week buckets of the parameters, farthest first, each ``{lo, hi, label}``."""
    return [{'lo': lo, 'hi': hi, 'label': f'{lo}-{hi} sem.'}
            for lo, hi in sorted(params['buckets'], key=lambda bucket: bucket[0], reverse=True)]


def _measure(cycle: Cycle, params: dict, herding: dict) -> dict:
    """
    Every measure of one cycle.

    Returns a dict with ``row`` (the row of ``cycles``), ``raw`` (unrounded ratios of ``before``, ``last``, ``final``
    and ``cycle``), ``cycle`` (the rounded ``{k, ratio}`` of the whole cycle), ``weeks`` (unrounded ``{k, ratio}``
    per bucket of ``_buckets``, the ratio ``None`` below ``min_polls`` polls), ``houses`` (its rows of ``houses``)
    and ``pending`` (``{pollster, k}`` of the houses not excluded with fewer than ``min_polls`` polls in the final
    window).
    """
    exclude, windows = params['exclude'], _windows(params)
    election = pd.Timestamp(cycle.event_date)
    polls = cycle.polls
    days = (election - pd.to_datetime(polls['end_date'].fillna(polls['date']))).dt.days
    polls = polls[(days >= 0) & (days <= params['max_days'])]
    sampled = polls[(polls['sample_size'] > 0) & (polls['weight'] > 0) & ~polls['pollster'].isin(exclude)]

    names = main_series(polls, cycle.parties, params['n_series'])
    rows = in_days(residual_rows(cycle, names, election, params), *windows['cycle'])
    apart = rows['pollster'].isin(exclude)
    pooled, cis = rows[~apart], rows[apart]

    raw = {window: dispersion(in_days(pooled, lo, hi), round_to=None) for window, (lo, hi) in windows.items()}
    ratios = {window: measure['ratio'] for window, measure in raw.items()}
    change = None
    if ratios['last'] is not None and ratios['before'] is not None:
        change = _round(ratios['last'] - ratios['before'])

    last_cis = in_days(cis, *windows['last'])
    row = {'event_date': cycle.event_date, 'label': cycle.label, 'current': cycle.current, 'names': names,
           'consensus': 'curve' if cycle.curve is not None else 'local', 'polls': int(len(polls)),
           'with_sample': int(len(sampled)), 'houses': int(sampled['pollster'].nunique()),
           'before': _rounded(raw['before']), 'last': _rounded(raw['last']), 'final': _rounded(raw['final']),
           'change': change,
           'cis': {'last': dispersion(last_cis), 'cycle': dispersion(cis), 'own': house_ratio(last_cis)}}

    houses, pending = [], []
    for pollster, own in in_days(rows, *windows['last']).groupby('pollster', sort=True):
        measure = house_ratio(own)
        excluded = pollster in exclude

        if measure['k'] >= params['min_polls']:
            errors = cycle.errors.get(pollster, {})
            houses.append({'event_date': cycle.event_date, 'label': cycle.label, 'current': cycle.current,
                           'pollster': pollster, 'k': measure['k'], 'h': measure['h'], 'p': measure['p'],
                           'cis': excluded, 'h_cycle': herding.get((cycle.event_date, pollster), {}).get('ratio'),
                           'last_error_avg': errors.get('last_error_avg'),
                           'last_error_blocks': errors.get('last_error_blocks')})
        elif not excluded and measure['k'] >= 1:
            pending.append({'pollster': pollster, 'k': measure['k']})

    # a bucket with fewer than ``min_polls`` polls has no ratio: one poll alone is not a dispersion between polls
    weeks = []
    for bucket in _buckets(params):
        week = dispersion(in_days(pooled, bucket['lo'] * 7, bucket['hi'] * 7 + 6), round_to=None)
        weeks.append(week if week['k'] >= params['min_polls'] else {'k': week['k'], 'ratio': None})

    return {'row': row, 'raw': ratios, 'cycle': _rounded(raw['cycle']), 'weeks': weeks, 'houses': houses,
            'pending': sorted(pending, key=lambda house: (-house['k'], house['pollster']))}


def _summary_current(now: dict, past: list, mean_last: Optional[float], days_left: int) -> dict:
    """The ``current`` block of the summary: the cycle in progress against the past ones."""
    row = now['row']
    before = row['before']['ratio']
    rank = None
    if before is not None:
        rank = 1 + sum(1 for cycle in past if cycle['row']['before']['ratio'] is not None
                       and cycle['row']['before']['ratio'] < before)
    whole = now['raw']['cycle']

    return {'label': row['label'], 'before': row['before'], 'last': row['last'], 'final': row['final'],
            'cycle': now['cycle'], 'days_left': days_left, 'rank_before': rank,
            'like_final': bool(whole is not None and mean_last is not None and whole <= mean_last)}


def _houses_now(now: dict) -> dict:
    """The ``houses_now`` block: measurable houses of the window in progress, those below 1 and those pending."""
    measurable = [house for house in now['houses'] if not house['cis']]
    herd = sorted((house for house in measurable if house['h'] is not None and house['h'] < 1),
                  key=lambda house: (house['h'], house['pollster']))

    return {'n': len(measurable), 'k': len(herd), 'herd': [{'pollster': h['pollster'], 'h': h['h']} for h in herd],
            'pending': now['pending']}


def _summary(past: list, now: Optional[dict], houses: list, days_left: Optional[int]) -> dict:
    """The summary of spec D3 from the measures of the past cycles and the one in progress."""
    comparable = [cycle for cycle in past if cycle['raw']['last'] is not None and cycle['raw']['before'] is not None]
    # the verdicts compare the printed (rounded) ratios; the means below use the unrounded ones
    narrowed = [cycle['row']['label'] for cycle in comparable
                if cycle['row']['last']['ratio'] < cycle['row']['before']['ratio']]
    below_one = [cycle['row']['label'] for cycle in past
                 if cycle['row']['final']['ratio'] is not None and cycle['row']['final']['ratio'] < 1]
    means = {window: _mean([cycle['raw'][window] for cycle in past]) for window in ('before', 'last', 'final')}

    summary = {'n_cycles': len(past), 'k_narrowed': len(narrowed), 'narrowed': narrowed,
               'mean_before': _round(means['before']), 'mean_last': _round(means['last']),
               'mean_final': _round(means['final']), 'k_below_one': len(below_one), 'below_one': below_one,
               'sign': sign_test(len(comparable), len(narrowed)), 'current': None, 'houses_now': None, 'cis': None,
               'accuracy': accuracy([house for house in houses if not house['current'] and not house['cis']])}

    if now is not None:
        cis = now['row']['cis']
        summary |= {'current': _summary_current(now, past, means['last'], days_left), 'houses_now': _houses_now(now),
                    'cis': {'current': cis['last'], 'cycle': cis['cycle'], 'own': cis['own']}}

    return summary


def _weeks(measures: list, params: dict) -> dict:
    """The ``weeks`` block: the buckets, a row per cycle and bucket and the mean over the past cycles."""
    buckets = _buckets(params)
    rows = [{'event_date': m['row']['event_date'], 'label': m['row']['label'], 'current': m['row']['current'],
             'lo': bucket['lo'], 'hi': bucket['hi']} | _rounded(week)
            for m in measures for bucket, week in zip(buckets, m['weeks'])]
    mean = []
    for position, bucket in enumerate(buckets):
        ratios = [m['weeks'][position]['ratio'] for m in measures
                  if not m['row']['current'] and m['weeks'][position]['ratio'] is not None]
        mean.append({'lo': bucket['lo'], 'hi': bucket['hi'], 'k': len(ratios), 'ratio': _round(_mean(ratios))})

    return {'buckets': buckets, 'rows': rows, 'mean': mean}


def compute_herd(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> HerdStudy:
    """
    Compute the herd study from the published bundle.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    data : FileSystem
        The ``data/`` directory of the repository (parameters).
    params : dict, optional
        Overrides of the parameters file.

    Returns
    -------
    HerdStudy
        The results.

    Raises
    ------
    MissingInput
        ``'events/es'`` (``events/{scope}``) when no past cycle has a published ``cycle`` part.
    """
    params = load_params(data) | (params or {})
    cycles, cycle_inputs = read_cycles(reader, params)

    if not cycles:
        raise MissingInput(f"events/{params['scope']}")

    current, current_inputs = read_current(reader, params)
    herding, herding_inputs = read_herding(reader, params['scope'])

    past = [_measure(cycle, params, herding) for cycle in cycles]
    now = _measure(current, params, herding) if current is not None else None
    measures = past + ([now] if now is not None else [])

    houses = sorted((house for m in measures for house in m['houses']),
                    key=lambda house: (house['event_date'], house['h'] is None, house['h'] or 0, house['pollster']))

    election = days_left = None
    if current is not None:
        as_of = current_inputs['as_of']
        days_left = int((pd.Timestamp(current.event_date) - pd.Timestamp(as_of)).days)
        election = {'date': current.event_date, 'days_left': days_left, 'as_of': as_of}

    return HerdStudy(params=params, election=election, summary=_summary(past, now, houses, days_left),
                     cycles=[m['row'] for m in measures], weeks=_weeks(measures, params), houses=houses,
                     inputs={'cycles': cycle_inputs, 'current': current_inputs, 'herding': herding_inputs})
