"""
Pure statistics of the herd study: how the polls of a cycle disperse around the published average, relative to the
sampling error of their samples, and the herding of each house.

A poll's residual in a series is its share minus the level of the average at the end of its fieldwork (the published
curve, or the local consensus of the other houses when the cycle has no curve). Its expected variance, were it a
simple random sample around that level, is ``1e4 · p(1 − p) / n`` in squared points. The dispersion of a set of rows
is the root of the mean squared residual over the mean expected variance: about 1 for polls that only differ by
sampling error, below 1 for polls closer to each other than their samples allow.
"""
import math
from typing import Optional

import numpy as np
import pandas as pd

from .data import Cycle

# Bounds of the level, as a proportion, in the expected variance of a share
LEVEL_BOUNDS = (0.005, 0.995)
ROW_COLUMNS = ['poll', 'pollster', 'days', 'name', 'resid', 'var']


def _sig3(value: float) -> float:
    """``value`` with three significant figures (rounded to decimals, a tiny p would become zero)."""
    return float(f'{float(value):.3g}')


def _is_number(value) -> bool:
    """Whether ``value`` is a usable number (not ``None`` nor NaN)."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def main_series(polls: pd.DataFrame, parties: list[str], n_series: int) -> list[str]:
    """
    The series the study measures: the parties with the highest mean share over the polls.

    Parameters
    ----------
    polls : pandas.DataFrame
        The polls of a cycle (``Cycle.polls``), one float column per party.
    parties : list of str
        The candidate parties, in the order that breaks ties.
    n_series : int
        How many series to keep.

    Returns
    -------
    list of str
        The ``n_series`` parties with the highest mean share, descending, ties in the order of ``parties``.
    """
    means = [(-polls[name].mean(), position, name) for position, name in enumerate(parties)
             if name in polls and _is_number(polls[name].mean())]

    return [name for _, _, name in sorted(means)[:n_series]]


def curve_level(curve: pd.DataFrame, name: str, when: pd.Timestamp) -> Optional[float]:
    """
    Level of the average of one series on a day.

    Parameters
    ----------
    curve : pandas.DataFrame
        The average (``Cycle.curve``), indexed by ``Timestamp`` with one column per party.
    name : str
        The series.
    when : pandas.Timestamp
        The day.

    Returns
    -------
    float or None
        The value on the last date of the curve not after ``when`` (the last date when ``when`` is after the end of
        the curve); ``None`` when ``when`` is before its first date, without the series or with a missing value.
    """
    if name not in curve or when < curve.index.min():
        return None

    value = float(curve[name].loc[:when].iloc[-1])

    return value if _is_number(value) else None


def local_level(polls: pd.DataFrame, index, name: str, window: int, exclude: list[str]) -> Optional[float]:
    """
    Local consensus of one series around one poll, for the cycles without a published curve: the mean share of the
    polls of the other houses.

    Parameters
    ----------
    polls : pandas.DataFrame
        The polls of the cycle with a ``pollster`` column, an ``end`` column (end of fieldwork as ``Timestamp``) and
        one float column per party; every published poll counts, with or without sample.
    index
        Label of the poll in ``polls``.
    name : str
        The series.
    window : int
        Days either side of the poll's end within which another poll counts.
    exclude : list of str
        Houses that never enter the consensus (the CIS).

    Returns
    -------
    float or None
        The plain mean of the shares of the polls of houses other than the poll's and not in ``exclude``, ending
        within ``± window`` days of it with a finite share; ``None`` with fewer than two such polls.
    """
    own = polls.loc[index]
    near = (polls['end'] - own['end']).abs() <= pd.Timedelta(days=window)
    others = polls[near & (polls['pollster'] != own['pollster']) & ~polls['pollster'].isin(exclude)][name].dropna()

    return float(others.mean()) if len(others) >= 2 else None


def residual_rows(cycle: Cycle, names: list[str], election: pd.Timestamp, params: dict) -> pd.DataFrame:
    """
    Residuals of the polls of a cycle around the level of the average, one row per poll and series.

    Parameters
    ----------
    cycle : Cycle
        The cycle; its ``curve`` gives the level, or the local consensus (``local_level``) when it is ``None``.
    names : list of str
        The series measured (``main_series``).
    election : pandas.Timestamp
        Election date.
    params : dict
        Parameters of the study; uses ``local_window`` and ``exclude`` (for the local consensus only).

    Returns
    -------
    pandas.DataFrame
        Columns ``poll`` (label of the poll in ``cycle.polls``), ``pollster``, ``days`` (int, days from the end of
        fieldwork, ``end_date`` or ``date`` without it, to the election), ``name``, ``resid`` (share minus level) and
        ``var`` (``1e4 · p(1 − p) / n``, ``p`` the level as a proportion within ``LEVEL_BOUNDS`` and ``n`` the
        sample). Only the polls with ``sample_size`` and ``weight`` above zero enter, and of them the series with a
        finite share and level. The excluded houses keep their rows: the caller splits them.
    """
    polls = cycle.polls.assign(end=pd.to_datetime(cycle.polls['end_date'].fillna(cycle.polls['date'])))
    entered = polls[(polls['sample_size'] > 0) & (polls['weight'] > 0)]
    rows = []

    for index, poll in entered.iterrows():
        days = int((election - poll['end']).days)

        for name in names:
            share = poll[name]

            if not _is_number(share):
                continue

            if cycle.curve is not None:
                level = curve_level(cycle.curve, name, poll['end'])
            else:
                level = local_level(polls, index, name, params['local_window'], params['exclude'])

            if level is None:
                continue

            p = min(max(level / 100, LEVEL_BOUNDS[0]), LEVEL_BOUNDS[1])
            rows.append({'poll': index, 'pollster': poll['pollster'], 'days': days, 'name': name,
                         'resid': float(share - level), 'var': 1e4 * p * (1 - p) / poll['sample_size']})

    return pd.DataFrame(rows, columns=ROW_COLUMNS).astype({'days': int, 'resid': float, 'var': float})


def in_days(rows: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """
    The rows whose poll ended between ``lo`` and ``hi`` days before the election, both included.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``residual_rows``.
    lo, hi : int
        Bounds of ``days``.

    Returns
    -------
    pandas.DataFrame
        The filtered rows.
    """
    return rows[(rows['days'] >= lo) & (rows['days'] <= hi)]


def dispersion(rows: pd.DataFrame) -> dict:
    """
    Dispersion of a set of residuals relative to the sampling error.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``residual_rows``.

    Returns
    -------
    dict
        ``k`` (distinct polls) and ``ratio`` (``sqrt(mean(resid²) / mean(var))`` rounded to 2 decimals, ``None``
        without polls).
    """
    k = int(rows['poll'].nunique())

    if k == 0:
        return {'k': 0, 'ratio': None}

    ratio = math.sqrt(float((rows['resid'] ** 2).mean()) / float(rows['var'].mean()))

    return {'k': k, 'ratio': round(ratio, 2)}


def house_ratio(rows: pd.DataFrame) -> dict:
    """
    Herding of one house: the spread of its polls around their own mean, relative to what their samples allow.

    In each series, ``ss_obs = Σ(r − r̄)²`` over the residuals of the house and ``ss_exp = (1 − 1/k_s) · Σ var``,
    its expectation for independent samples (``k_s`` the rows of the series); both are summed over the series and
    ``h = sqrt(ss_obs / ss_exp)``. Under independent sampling ``h² · (k − 1)`` is about a chi-squared of ``k − 1``
    degrees of freedom, whose lower tail is ``p``: a small ``p`` says the polls are closer to each other than chance.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``residual_rows`` of one house (the caller filters the window).

    Returns
    -------
    dict
        ``k`` (distinct polls), ``h`` rounded to 2 decimals and ``p`` (``chi2.cdf(h² · (k − 1), k − 1)``) with three
        significant figures; ``h`` and ``p`` ``None`` with fewer than two polls.
    """
    from scipy.stats import chi2

    k = int(rows['poll'].nunique())

    if k < 2:
        return {'k': k, 'h': None, 'p': None}

    ss_obs = ss_exp = 0.0

    for _, series in rows.groupby('name'):
        resid = series['resid']
        ss_obs += float(((resid - resid.mean()) ** 2).sum())
        ss_exp += (1 - 1 / len(series)) * float(series['var'].sum())

    if ss_exp <= 0:
        return {'k': k, 'h': None, 'p': None}

    h = math.sqrt(ss_obs / ss_exp)
    p = chi2.cdf(h ** 2 * (k - 1), k - 1)

    return {'k': k, 'h': round(h, 2), 'p': _sig3(p)}


def sign_test(n: int, k: int) -> Optional[dict]:
    """
    How likely ``k`` or more of ``n`` cases fall to one side by chance: the upper tail of a binomial of a fair coin.

    Parameters
    ----------
    n : int
        Cases.
    k : int
        Cases on the side counted.

    Returns
    -------
    dict or None
        ``n``, ``k`` and ``p`` (``binom.sf(k − 1, n, 0.5)``) with three significant figures; ``None`` without cases.
    """
    from scipy.stats import binom

    if n <= 0:
        return None

    return {'n': n, 'k': k, 'p': _sig3(binom.sf(k - 1, n, 0.5))}


def _error_group(rows: list) -> dict:
    """``k``, mean ``last_error_avg`` and mean ``abs(last_error_blocks)`` of a group of rows, rounded to 2."""
    blocks = [abs(row['last_error_blocks']) for row in rows if _is_number(row.get('last_error_blocks'))]

    return {'k': len(rows), 'error_avg': round(float(np.mean([row['last_error_avg'] for row in rows])), 2),
            'error_blocks': round(float(np.mean(blocks)), 2) if blocks else None}


def accuracy(rows: list[dict]) -> Optional[dict]:
    """
    Whether the houses that herd (``h < 1``) erred less than the rest with their last poll.

    Parameters
    ----------
    rows : list of dict
        Rows of the houses with ``h``, ``last_error_avg`` and ``last_error_blocks``; only those with an
        ``last_error_avg`` and an ``h`` enter.

    Returns
    -------
    dict or None
        ``n`` (houses entered), ``herd`` and ``rest`` (each ``k``, ``error_avg`` the mean of ``last_error_avg`` and
        ``error_blocks`` the mean of ``abs(last_error_blocks)``, rounded to 2); ``None`` when a group is empty.
    """
    entered = [row for row in rows if _is_number(row.get('last_error_avg')) and _is_number(row.get('h'))]
    herd = [row for row in entered if row['h'] < 1]
    rest = [row for row in entered if row['h'] >= 1]

    if not herd or not rest:
        return None

    return {'n': len(entered), 'herd': _error_group(herd), 'rest': _error_group(rest)}
