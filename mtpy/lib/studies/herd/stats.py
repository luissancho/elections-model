"""
Pure statistics of the herd study: how far apart the polls of two different houses are, relative to what their
samples allow, and whether that distance shrinks as the election nears.

Two polls of different houses, fielded within a few days of each other, measure the same opinion: their difference in a
series, squared (``d2``), has an expectation of ``v = 1e4 · p(1 − p) · (1/n_i + 1/n_j)`` squared points were both
simple random samples. The dispersion of a set of comparisons is ``sqrt(Σ d2 / Σ v)``: about 1 for polls that only
differ by sampling error, below 1 for polls closer to each other than their samples allow. Herding shows as the
dispersion of the same pairs of houses falling from far away from the election to its last weeks.

Two periods are compared pair of houses by pair of houses: each pair ``g`` has its own ``r²_g = Σ d2_g / Σ v_g`` in
each period, and the pairs are combined with one weight per pair, the same in both periods,
``w_g = 1 / (1/Σ v_a,g + 1/Σ v_b,g)``. Pooling all the rows instead would weigh each pair by its rows, which differ
between the periods: a change in the mix of pairs would read as a change in the dispersion.

The rows of one pair of polls (one per series) are not independent, so the permutation tests shuffle the period labels
of whole pairs of polls (blocks), never of single rows.

The null test of the pooled change shuffles the same blocks and keeps every permuted change, whose distribution is the
one expected if the period labels carried no information; the bootstrap over elections gives an interval for the
observed change.
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


def bucket_of(days: float, buckets: list) -> Optional[tuple[int, int]]:
    """
    Bucket of weeks a number of days before the election falls in.

    Parameters
    ----------
    days : float
        Days to the election.
    buckets : list
        ``[lo, hi]`` pairs in weeks; a bucket covers the days ``lo · 7 <= days < (hi + 1) · 7``.

    Returns
    -------
    tuple of int or None
        The ``(lo, hi)`` of the bucket containing ``days``; ``None`` outside all of them.
    """
    for lo, hi in buckets:
        if lo * 7 <= days < (hi + 1) * 7:
            return lo, hi

    return None


def comparisons(polls: pd.DataFrame, names: list[str], params: dict) -> pd.DataFrame:
    """
    Comparisons between the polls of different houses fielded close in time, one row per pair of polls and series.

    Parameters
    ----------
    polls : pandas.DataFrame
        Columns ``poll`` (label), ``pollster``, ``days`` (int, days to the election), ``n`` (float, sample) and one
        float column per series in ``names``. ``poll_i`` of a pair is the earlier row in this order, ``poll_j`` the
        later one. The polls without ``pollster`` are left out.
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
    frame = polls[polls['pollster'].notna()].reset_index(drop=True)

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


def by_buckets(rows: pd.DataFrame, buckets: list, min_pairs: int) -> list[dict]:
    """
    Dispersion of the comparisons in each bucket of weeks to the election.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of ``comparisons`` (``poll_i``, ``poll_j``, ``days``, ``d2`` and ``v`` columns).
    buckets : list
        ``[lo, hi]`` pairs in weeks (see ``bucket_of``).
    min_pairs : int
        Minimum number of distinct pairs of polls for a bucket to have a ratio.

    Returns
    -------
    list of dict
        One entry per bucket, in the order of ``buckets``: ``lo``, ``hi``, ``pairs`` (distinct pairs of polls),
        ``comparisons`` (rows) and ``ratio`` (``dispersion``, unrounded; ``None`` below ``min_pairs``).
    """
    keys = [bucket_of(days, buckets) for days in rows['days']]
    out = []

    for lo, hi in buckets:
        inside = rows[np.array([key == (lo, hi) for key in keys], dtype=bool)]
        pairs = len(inside[['poll_i', 'poll_j']].drop_duplicates())
        out.append({'lo': lo, 'hi': hi, 'pairs': pairs, 'comparisons': len(inside),
                    'ratio': dispersion(inside) if pairs >= min_pairs else None})

    return out


def _shared_pairs(rows: pd.DataFrame, a: str, b: str, min_rows: int) -> pd.DataFrame:
    """The rows of periods ``a`` and ``b`` of the pairs of houses with at least ``min_rows`` rows in each."""
    in_a = rows[rows['period'] == a].groupby('pair').size()
    in_b = rows[rows['period'] == b].groupby('pair').size()
    pairs = in_a[in_a >= min_rows].index.intersection(in_b[in_b >= min_rows].index)

    return rows[rows['pair'].isin(pairs) & rows['period'].isin([a, b])]


def _weighted_sums(d2_a, v_a, d2_b, v_b) -> tuple:
    """
    The terms of one pair of houses in the weighted squared dispersions of both periods.

    Parameters
    ----------
    d2_a, v_a, d2_b, v_b : float or numpy.ndarray
        ``Σ d2`` and ``Σ v`` of the pair in period ``a`` and in period ``b`` (arrays: one value per permutation).

    Returns
    -------
    tuple
        ``(w, w · r²_a, w · r²_b)`` with ``w = 1 / (1/v_a + 1/v_b) = v_a · v_b / (v_a + v_b)``, so that
        ``w · r²_a = d2_a · v_b / (v_a + v_b)`` and ``w · r²_b = d2_b · v_a / (v_a + v_b)``.
    """
    total = v_a + v_b

    return v_a * v_b / total, d2_a * v_b / total, d2_b * v_a / total


def compare(rows: pd.DataFrame, a: str, b: str, min_rows: int) -> Optional[dict]:
    """
    Dispersion of the same pairs of houses in two periods, each pair with the same weight in both.

    For each pair of houses ``g`` counted, ``r²_a,g = Σ d2_a,g / Σ v_a,g`` and ``r²_b,g`` likewise, and its weight
    ``w_g = 1 / (1/Σ v_a,g + 1/Σ v_b,g)``; ``r_a = sqrt(Σ w_g r²_a,g / Σ w_g)`` and ``r_b`` likewise. With one pair
    this is the dispersion of its rows in each period.

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

    sums = shared.groupby(['pair', 'period'])[['d2', 'v']].sum().unstack('period')
    weight, term_a, term_b = _weighted_sums(sums[('d2', a)].to_numpy(float), sums[('v', a)].to_numpy(float),
                                            sums[('d2', b)].to_numpy(float), sums[('v', b)].to_numpy(float))
    r_a = math.sqrt(term_a.sum() / weight.sum())
    r_b = math.sqrt(term_b.sum() / weight.sum())

    if r_a == 0:
        return None

    return {'pairs': len(sums), 'rows_a': int((shared['period'] == a).sum()),
            'rows_b': int((shared['period'] == b).sum()), 'r_a': r_a, 'r_b': r_b, 'change': r_b / r_a - 1,
            'falls': bool(r_b < r_a)}


def _blocks(group: pd.DataFrame, a: str) -> tuple:
    """
    The pairs of polls (blocks) of one pair of houses, with their series added up.

    Parameters
    ----------
    group : pandas.DataFrame
        Rows of ``comparisons`` of one pair of houses (``poll_i``, ``poll_j``, ``period``, ``d2`` and ``v``); all the
        rows of a pair of polls share their period.
    a : str
        The reference period.

    Returns
    -------
    tuple of numpy.ndarray
        ``(d2, v, is_a)``: ``Σ d2`` and ``Σ v`` of each pair of polls (in order of appearance) and whether its period
        is ``a``.
    """
    blocks = group.groupby(['poll_i', 'poll_j'], sort=False).agg(d2=('d2', 'sum'), v=('v', 'sum'),
                                                                  period=('period', 'first'))

    return blocks['d2'].to_numpy(float), blocks['v'].to_numpy(float), (blocks['period'] == a).to_numpy()


def _permuted_terms(shared: pd.DataFrame, a: str, permutations: int, generator: np.random.Generator) -> tuple:
    """
    Σ w · r² of each period, permuted and observed, over the pairs of houses of ``shared``.

    Within each pair of houses the period labels of its blocks (pairs of polls) are shuffled; each pair keeps its
    number of blocks in ``a``.

    Parameters
    ----------
    shared : pandas.DataFrame
        Rows of ``_shared_pairs``.
    a : str
        The reference period.
    permutations : int
        Number of shuffles.
    generator : numpy.random.Generator
        Source of the shuffles (advanced).

    Returns
    -------
    tuple
        ``(perm_a, perm_b, obs_a, obs_b)``: ``Σ w · r²_a`` and ``Σ w · r²_b`` per permutation (arrays) and observed
        (floats); ``Σ w`` cancels out in the ratio.
    """
    perm_a, perm_b = np.zeros(permutations), np.zeros(permutations)
    obs_a = obs_b = 0.0

    for _, group in shared.groupby('pair'):
        d2, v, is_a = _blocks(group, a)
        size = len(d2)
        # A random arrangement per permutation: the blocks with the smallest keys take the label a
        order = np.argsort(generator.random((permutations, size)), axis=1)
        label_a = np.zeros((permutations, size), dtype=bool)
        np.put_along_axis(label_a, order[:, :int(is_a.sum())], True, axis=1)
        d2_a, v_a = label_a @ d2, label_a @ v
        _, term_a, term_b = _weighted_sums(d2_a, v_a, d2.sum() - d2_a, v.sum() - v_a)
        perm_a += term_a
        perm_b += term_b
        _, term_a, term_b = _weighted_sums(d2[is_a].sum(), v[is_a].sum(), d2[~is_a].sum(), v[~is_a].sum())
        obs_a += term_a
        obs_b += term_b

    return perm_a, perm_b, obs_a, obs_b


def _share_not_above(perm_a: np.ndarray, perm_b: np.ndarray, obs_a: float, obs_b: float) -> float:
    """Share of the permuted ``r_b² / r_a²`` not above the observed one (with ``PERMUTATION_TOLERANCE``)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        permuted = perm_b / perm_a

    return float(np.mean(permuted <= obs_b / obs_a * (1 + PERMUTATION_TOLERANCE)))


def permutation_p(rows: pd.DataFrame, a: str, b: str, min_rows: int, permutations: int,
                  seed: int) -> Optional[float]:
    """
    Permutation test of the fall in dispersion between two periods.

    Within each pair of houses counted by ``compare``, the period labels of its pairs of polls are shuffled, each with
    its series together (each pair of houses keeps its number of pairs of polls in each period); the statistic is the
    one of ``compare``, ``r_b² / r_a²``, recomputed for each permutation with the weights of that permutation.

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
    terms = _permuted_terms(shared, a, permutations, np.random.default_rng(seed))

    return _share_not_above(*terms)


def _pair_terms(shared: pd.DataFrame, a: str, b: str) -> tuple:
    """``_weighted_sums`` of every pair of houses of ``shared`` (rows of ``_shared_pairs``), as arrays."""
    sums = shared.groupby(['pair', 'period'])[['d2', 'v']].sum().unstack('period')

    return _weighted_sums(sums[('d2', a)].to_numpy(float), sums[('v', a)].to_numpy(float),
                          sums[('d2', b)].to_numpy(float), sums[('v', b)].to_numpy(float))


def pooled(rows_by_election: list, a: str, b: str, min_rows: int) -> Optional[dict]:
    """
    Dispersion of the same pairs of houses in two periods over several elections, each pair with one weight.

    Like ``compare`` but the pairs of houses of every election are accumulated (``Σ w``, ``Σ w · r²_a`` and
    ``Σ w · r²_b`` over all of them).

    Parameters
    ----------
    rows_by_election : list of pandas.DataFrame
        Rows of ``comparisons`` of each election.
    a, b : str
        The periods compared.
    min_rows : int
        As in ``compare``.

    Returns
    -------
    dict or None
        ``elections`` (those with pairs counted), ``pairs`` (pairs of houses, summed), ``rows_a``, ``rows_b``,
        ``r_a``, ``r_b``, ``change`` and ``falls`` (as in ``compare``), unrounded; ``None`` without pairs or when
        ``r_a`` is 0.
    """
    total_w = total_a = total_b = 0.0
    elections = pairs = rows_a = rows_b = 0

    for rows in rows_by_election:
        shared = _shared_pairs(rows, a, b, min_rows)

        if len(shared) == 0:
            continue

        weight, term_a, term_b = _pair_terms(shared, a, b)
        total_w += weight.sum()
        total_a += term_a.sum()
        total_b += term_b.sum()
        elections += 1
        pairs += len(weight)
        rows_a += int((shared['period'] == a).sum())
        rows_b += int((shared['period'] == b).sum())

    if elections == 0:
        return None

    r_a, r_b = math.sqrt(total_a / total_w), math.sqrt(total_b / total_w)

    if r_a == 0:
        return None

    return {'elections': elections, 'pairs': pairs, 'rows_a': rows_a, 'rows_b': rows_b, 'r_a': r_a, 'r_b': r_b,
            'change': r_b / r_a - 1, 'falls': bool(r_b < r_a)}


def pooled_p(rows_by_election: list, a: str, b: str, min_rows: int, permutations: int,
             seed: int) -> Optional[float]:
    """
    Permutation test of the fall in the pooled dispersion over several elections.

    The labels of the pairs of polls are shuffled within each pair of houses of each election (as in
    ``permutation_p``) and the statistic ``R_b² / R_a²`` of ``pooled`` is recomputed for each permutation.

    Parameters
    ----------
    rows_by_election : list of pandas.DataFrame
        Rows of ``comparisons`` of each election.
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
        Share of the permuted statistics not above the observed one; ``None`` when ``pooled`` is ``None``.
    """
    if pooled(rows_by_election, a, b, min_rows) is None:
        return None

    generator = np.random.default_rng(seed)
    perm_a, perm_b = np.zeros(permutations), np.zeros(permutations)
    obs_a = obs_b = 0.0

    for rows in rows_by_election:
        shared = _shared_pairs(rows, a, b, min_rows)

        if len(shared) == 0:
            continue

        terms = _permuted_terms(shared, a, permutations, generator)
        perm_a += terms[0]
        perm_b += terms[1]
        obs_a += terms[2]
        obs_b += terms[3]

    return _share_not_above(perm_a, perm_b, obs_a, obs_b)


def pooled_terms(rows_by_election: list, a: str, b: str, min_rows: int) -> list[tuple[float, float]]:
    """
    The terms of each election in the pooled dispersion: ``(Σ w · r²_a, Σ w · r²_b)`` over its pairs of houses.

    Parameters
    ----------
    rows_by_election : list of pandas.DataFrame
        Rows of ``comparisons`` of each election.
    a, b : str
        The periods compared.
    min_rows : int
        As in ``compare``.

    Returns
    -------
    list of tuple
        ``(A_e, B_e)`` of each election with pairs of houses counted, in order; the pooled change is
        ``sqrt(Σ B_e / Σ A_e) − 1``.
    """
    terms = []

    for rows in rows_by_election:
        shared = _shared_pairs(rows, a, b, min_rows)

        if len(shared) == 0:
            continue

        _, term_a, term_b = _pair_terms(shared, a, b)
        terms.append((float(term_a.sum()), float(term_b.sum())))

    return terms


def bootstrap_pooled(terms: list, resamples: int, seed: int) -> Optional[dict]:
    """
    Percentile bootstrap interval (95 %) of the pooled change, resampling elections with replacement.

    Parameters
    ----------
    terms : list of tuple
        ``(A_e, B_e)`` per election (see ``pooled_terms``).
    resamples : int
        Number of resamples.
    seed : int
        Seed of the generator.

    Returns
    -------
    dict or None
        ``lo`` and ``hi`` (percentiles 2.5 and 97.5 of ``sqrt(Σ B / Σ A) − 1`` over the resamples), unrounded;
        resamples with ``Σ A`` 0 are discarded; ``None`` with fewer than 3 elections or 3 valid resamples.
    """
    if len(terms) < 3:
        return None

    values = np.asarray(terms, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    total_a, total_b = values[picks, 0].sum(axis=1), values[picks, 1].sum(axis=1)
    valid = total_a > 0

    if valid.sum() < 3:
        return None

    changes = np.sqrt(total_b[valid] / total_a[valid]) - 1

    return {'lo': float(np.percentile(changes, 2.5)), 'hi': float(np.percentile(changes, 97.5))}


def null_distribution(rows_by_election: list, a: str, b: str, min_rows: int, permutations: int,
                      seed: int) -> Optional[dict]:
    """
    Distribution of the pooled change when the period labels of the pairs of polls are shuffled.

    The shuffles are those of ``pooled_p``; each gives a change ``sqrt(R_b² / R_a²) − 1``.

    Parameters
    ----------
    rows_by_election : list of pandas.DataFrame
        Rows of ``comparisons`` of each election.
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
    dict or None
        ``observed`` (the pooled change), ``permuted`` (array, one change per shuffle), ``count_one`` (shuffles with
        a change not above the observed one), ``count_two`` (shuffles with a change at least as far from 0),
        ``p_one`` and ``p_two`` (``(count + 1) / (permutations + 1)``); ``None`` when ``pooled`` is ``None``.
    """
    if pooled(rows_by_election, a, b, min_rows) is None:
        return None

    generator = np.random.default_rng(seed)
    perm_a, perm_b = np.zeros(permutations), np.zeros(permutations)
    obs_a = obs_b = 0.0

    for rows in rows_by_election:
        shared = _shared_pairs(rows, a, b, min_rows)

        if len(shared) == 0:
            continue

        terms = _permuted_terms(shared, a, permutations, generator)
        perm_a += terms[0]
        perm_b += terms[1]
        obs_a += terms[2]
        obs_b += terms[3]

    with np.errstate(divide='ignore', invalid='ignore'):
        permuted = np.sqrt(perm_b / perm_a) - 1

    observed = math.sqrt(obs_b / obs_a) - 1
    count_one = int(np.sum(permuted <= observed + PERMUTATION_TOLERANCE))
    count_two = int(np.sum(np.abs(permuted) >= abs(observed) - PERMUTATION_TOLERANCE))

    return {'observed': observed, 'permuted': permuted, 'count_one': count_one, 'count_two': count_two,
            'p_one': (count_one + 1) / (permutations + 1), 'p_two': (count_two + 1) / (permutations + 1)}


def histogram(values, bin_width: float) -> list[dict]:
    """
    Counts of ``values`` in bins of a fixed width whose edges are multiples of it.

    Parameters
    ----------
    values : array-like of float
        The values.
    bin_width : float
        Width of each bin.

    Returns
    -------
    list of dict
        ``lo``, ``hi`` (edges rounded to the decimals of ``bin_width``) and ``count`` of each bin, from the one with
        the smallest value to the one with the largest (the last bin includes its upper edge); one bin when all the
        values coincide; empty without values.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return []

    decimals = max(0, int(round(-math.log10(bin_width))))
    first = math.floor(round(values.min() / bin_width, 9))
    last = max(math.ceil(round(values.max() / bin_width, 9)), first + 1)
    edges = np.round(np.arange(first, last + 1) * bin_width, decimals)
    counts, _ = np.histogram(values, bins=edges)

    return [{'lo': float(edges[k]), 'hi': float(edges[k + 1]), 'count': int(counts[k])} for k in range(len(counts))]


def bootstrap_median(changes: list, resamples: int, seed: int) -> Optional[dict]:
    """
    Percentile bootstrap interval (95 %) of the median of a set of changes.

    Parameters
    ----------
    changes : list of float
        The changes (one per election).
    resamples : int
        Number of resamples with replacement.
    seed : int
        Seed of the generator.

    Returns
    -------
    dict or None
        ``lo`` and ``hi`` (percentiles 2.5 and 97.5 of the resampled medians), unrounded; ``None`` with fewer than
        3 changes.
    """
    if len(changes) < 3:
        return None

    values = np.asarray(changes, dtype=float)
    generator = np.random.default_rng(seed)
    medians = np.median(values[generator.integers(0, len(values), size=(resamples, len(values)))], axis=1)

    return {'lo': float(np.percentile(medians, 2.5)), 'hi': float(np.percentile(medians, 97.5))}


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
