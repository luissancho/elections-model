"""
Pure statistics of the herd study: how far apart the polls of two different houses are, relative to what their
samples allow, and whether that distance shrinks as the election nears.

Two polls of different houses, fielded within a few days of each other, measure the same opinion: their difference in a
series, squared (``d2``), has an expectation of ``v = 1e4 · p(1 − p) · (1/n_i + 1/n_j)`` squared points were both
simple random samples. The dispersion of a set of comparisons is ``sqrt(Σ d2 / Σ v)``: about 1 for polls that only
differ by sampling error, below 1 for polls closer to each other than their samples allow. Herding shows as the
dispersion of the same pair of houses falling from far away from the election to its last weeks.
"""
import math
from typing import Optional

import numpy as np
import pandas as pd

# Bounds of the level, as a proportion, in the expected variance of a difference
LEVEL_BOUNDS = (0.005, 0.995)
COMPARISON_COLUMNS = ['pair', 'poll_i', 'poll_j', 'days', 'period', 'name', 'd2', 'v']
# Relative tolerance when comparing a permuted statistic with the observed one (floating-point noise)
PERMUTATION_TOLERANCE = 1e-9


def _sig3(value: float) -> float:
    """``value`` with three significant figures (rounded to decimals, a tiny p would become zero)."""
    return float(f'{float(value):.3g}')


def _is_number(value) -> bool:
    """Whether ``value`` is a usable number (not ``None`` nor NaN)."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def period_of(days: float, periods: dict) -> Optional[str]:
    """
    Period of the cycle a number of days before the election falls in.

    Parameters
    ----------
    days : float
        Days to the election.
    periods : dict
        Name of each period mapped to its ``[lo, hi)`` bounds in days (half-open: ``lo`` included, ``hi`` not),
        except the period with the largest ``hi``, which also includes it (the last day of the cycle counts).

    Returns
    -------
    str or None
        The name of the period containing ``days``; ``None`` outside all of them.
    """
    last = max(hi for _, hi in periods.values())

    for name, (lo, hi) in periods.items():
        if lo <= days < hi or days == hi == last:
            return name

    return None


def comparisons(polls: pd.DataFrame, names: list[str], params: dict) -> pd.DataFrame:
    """
    Comparisons between the polls of different houses fielded close in time, one row per pair of polls and series.

    Parameters
    ----------
    polls : pandas.DataFrame
        Columns ``poll`` (label), ``pollster``, ``days`` (int, days to the election), ``n`` (float, sample) and one
        float column per series in ``names``. ``poll_i`` of a pair is the earlier row in this order, ``poll_j`` the
        later one.
    names : list of str
        The series compared.
    params : dict
        Uses ``pair_days`` (maximum ``|days_i − days_j|``) and ``periods`` (see ``period_of``).

    Returns
    -------
    pandas.DataFrame
        Columns ``COMPARISON_COLUMNS``: ``pair`` (both houses sorted and joined by ``|``), ``poll_i``, ``poll_j``,
        ``days`` (mean of the two), ``period``, ``name``, ``d2`` (squared difference of the shares) and ``v``
        (``1e4 · p(1 − p) · (1/n_i + 1/n_j)``, ``p`` the mean share as a proportion within ``LEVEL_BOUNDS``). Only
        the series with both shares finite enter.
    """
    rows = []
    frame = polls.reset_index(drop=True)

    for i in range(len(frame)):
        for j in range(i + 1, len(frame)):
            first, second = frame.iloc[i], frame.iloc[j]

            if first['pollster'] == second['pollster'] or abs(first['days'] - second['days']) > params['pair_days']:
                continue

            days = (first['days'] + second['days']) / 2
            pair = '|'.join(sorted([first['pollster'], second['pollster']]))

            for name in names:
                x_i, x_j = first[name], second[name]

                if not (np.isfinite(x_i) and np.isfinite(x_j)):
                    continue

                p = min(max((x_i + x_j) / 200, LEVEL_BOUNDS[0]), LEVEL_BOUNDS[1])
                rows.append({'pair': pair, 'poll_i': first['poll'], 'poll_j': second['poll'], 'days': days,
                             'period': period_of(days, params['periods']), 'name': name,
                             'd2': float((x_i - x_j) ** 2),
                             'v': float(1e4 * p * (1 - p) * (1 / first['n'] + 1 / second['n']))})

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def dispersion(rows: pd.DataFrame) -> Optional[float]:
    """
    Dispersion of a set of comparisons relative to the sampling error.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``comparisons`` (``d2`` and ``v`` columns).

    Returns
    -------
    float or None
        ``sqrt(Σ d2 / Σ v)``, unrounded; ``None`` without rows.
    """
    if len(rows) == 0:
        return None

    return math.sqrt(float(rows['d2'].sum()) / float(rows['v'].sum()))


def _shared_pairs(rows: pd.DataFrame, a: str, b: str, min_rows: int) -> pd.DataFrame:
    """The rows of periods ``a`` and ``b`` of the pairs of houses with at least ``min_rows`` rows in each."""
    in_a = rows[rows['period'] == a].groupby('pair').size()
    in_b = rows[rows['period'] == b].groupby('pair').size()
    pairs = in_a[in_a >= min_rows].index.intersection(in_b[in_b >= min_rows].index)

    return rows[rows['pair'].isin(pairs) & rows['period'].isin([a, b])]


def compare(rows: pd.DataFrame, a: str, b: str, min_rows: int) -> Optional[dict]:
    """
    Dispersion of the same pairs of houses in two periods.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``comparisons``.
    a, b : str
        The periods compared (``a`` the reference, ``b`` the one expected to be closer).
    min_rows : int
        Minimum rows a pair of houses needs in each of the two periods to count.

    Returns
    -------
    dict or None
        ``pairs`` (pairs of houses counted), ``rows_a``, ``rows_b``, ``r_a``, ``r_b`` (dispersions), ``change``
        (``r_b / r_a − 1``) and ``falls`` (``r_b < r_a``), all unrounded; ``None`` without pairs or when ``r_a`` is 0.
    """
    shared = _shared_pairs(rows, a, b, min_rows)

    if len(shared) == 0:
        return None

    in_a, in_b = shared[shared['period'] == a], shared[shared['period'] == b]
    r_a, r_b = dispersion(in_a), dispersion(in_b)

    if r_a == 0:
        return None

    return {'pairs': int(shared['pair'].nunique()), 'rows_a': len(in_a), 'rows_b': len(in_b), 'r_a': r_a, 'r_b': r_b,
            'change': r_b / r_a - 1, 'falls': bool(r_b < r_a)}


def permutation_p(rows: pd.DataFrame, a: str, b: str, min_rows: int, permutations: int,
                  seed: int) -> Optional[float]:
    """
    Permutation test of the fall in dispersion between two periods.

    Within each pair of houses counted by ``compare``, the labels ``a``/``b`` of its rows are shuffled; the statistic is
    ``(Σ d2_b / Σ v_b) / (Σ d2_a / Σ v_a)`` over all the pairs.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``comparisons``.
    a, b : str
        The periods compared.
    min_rows : int
        As in ``compare``.
    permutations : int
        Number of shuffles.
    seed : int
        Seed of the generator.

    Returns
    -------
    float or None
        Share of the permuted statistics not above the observed one (the p-value of a fall); ``None`` when
        ``compare`` is ``None``.
    """
    if compare(rows, a, b, min_rows) is None:
        return None

    shared = _shared_pairs(rows, a, b, min_rows)
    generator = np.random.default_rng(seed)
    sum_d2_a = np.zeros(permutations)
    sum_v_a = np.zeros(permutations)
    total_d2 = total_v = 0.0
    obs_d2_b = obs_v_b = 0.0

    for _, group in shared.groupby('pair'):
        d2, v = group['d2'].to_numpy(float), group['v'].to_numpy(float)
        is_a = (group['period'] == a).to_numpy()
        size = len(group)
        # A random arrangement per permutation: the rows with the smallest keys take the label a
        order = np.argsort(generator.random((permutations, size)), axis=1)
        label_a = np.zeros((permutations, size), dtype=bool)
        np.put_along_axis(label_a, order[:, :int(is_a.sum())], True, axis=1)
        sum_d2_a += label_a @ d2
        sum_v_a += label_a @ v
        total_d2 += d2.sum()
        total_v += v.sum()
        obs_d2_b += d2[~is_a].sum()
        obs_v_b += v[~is_a].sum()

    observed = (obs_d2_b / obs_v_b) / ((total_d2 - obs_d2_b) / (total_v - obs_v_b))

    with np.errstate(divide='ignore', invalid='ignore'):
        permuted = ((total_d2 - sum_d2_a) / (total_v - sum_v_a)) / (sum_d2_a / sum_v_a)

    return float(np.mean(permuted <= observed * (1 + PERMUTATION_TOLERANCE)))


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


def count(results: list[dict], alpha: float = 0.05) -> dict:
    """
    Tally of the results of ``compare`` (with their ``p`` from ``permutation_p``) over several elections.

    Parameters
    ----------
    results : list of dict
        Dicts with ``falls``, ``p`` and ``change``.
    alpha : float, optional
        Significance level of a clear fall.

    Returns
    -------
    dict
        ``n`` (results), ``k`` (those that fall), ``sign_p`` (``sign_test(n, k)['p']``), ``clear`` (those with
        ``p < alpha``) and ``median_change``; without results ``n``, ``k`` and ``clear`` are 0 and the rest ``None``.
    """
    if not results:
        return {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}

    k = sum(1 for result in results if result['falls'])

    return {'n': len(results), 'k': k, 'sign_p': sign_test(len(results), k)['p'],
            'clear': sum(1 for result in results if result['p'] < alpha),
            'median_change': float(np.median([result['change'] for result in results]))}
