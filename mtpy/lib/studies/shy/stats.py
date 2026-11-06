"""
Pure statistics of the shy-vote study: how far the final polls of an election fall from the result, for a party, a
block or a lineage.

The error of a group in a poll is the sum of the shares the poll gives to the members of the group that appear in the
result, minus the sum of their results; a poll that does not estimate a member leaves it out of both sums. Negative
means the group was underestimated. The error of the group in an election is the mean over its polls.
"""
import math
from typing import Optional

import pandas as pd


def _is_number(value) -> bool:
    """Whether ``value`` is a usable number (not ``None`` nor NaN)."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def significant(p: Optional[float]) -> Optional[float]:
    """
    A p value with three significant figures (rounding to decimals would turn a tiny p into zero).

    Parameters
    ----------
    p : float or None
        The p value.

    Returns
    -------
    float or None
        ``p`` with three significant figures, ``None`` when ``p`` is ``None``.
    """
    return None if p is None else float(f'{float(p):.3g}')


def group_error(polls: pd.DataFrame, names: list[str], results: dict) -> tuple[Optional[float], Optional[str]]:
    """
    Mean error of a group (block or lineage) over the polls of an election.

    Parameters
    ----------
    polls : pandas.DataFrame
        The polls, one float column per party, ``NaN`` (or ``None``) for a share the poll does not estimate.
    names : list of str
        The members of the group.
    results : dict
        Result of the election as ``{party: pct}``.

    Returns
    -------
    tuple
        The mean error in points (negative: underestimated), ``None`` when no poll estimates a member present in
        ``results``; and the present members joined with ``+`` in the order of ``names``, ``None`` when none is
        present.
    """
    present = [name for name in names if name in results]
    if not present:
        return None, None

    label = '+'.join(present)
    errors = []
    for _, poll in polls.iterrows():
        estimated = [name for name in present if name in poll and _is_number(poll[name]) and not pd.isna(poll[name])]
        if estimated:
            errors.append(sum(float(poll[name]) for name in estimated) - sum(results[name] for name in estimated))

    return (sum(errors) / len(errors) if errors else None), label


def figures(values: list, threshold: float = 2) -> Optional[dict]:
    """
    The figures block of a list of errors.

    Parameters
    ----------
    values : list
        Errors in points; ``None`` entries are dropped.
    threshold : float, default 2
        Absolute error, in points, above which an error counts as large.

    Returns
    -------
    dict or None
        ``n``, ``under`` (errors below zero), ``over`` (above zero), ``mean``, ``median``, ``p_t`` (two-sided t test
        of a zero mean, ``None`` with fewer than 3 values), ``p_sign`` (one-sided sign test on the majority sign),
        ``size`` (mean absolute error) and ``size_over`` (share of absolute errors above ``threshold``); ``None``
        when no value remains.
    """
    from scipy import stats as scipy_stats

    values = [float(value) for value in values if _is_number(value)]
    if not values:
        return None

    n = len(values)
    under = sum(1 for value in values if value < 0)
    over = sum(1 for value in values if value > 0)
    ordered = sorted(values)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    p_t = float(scipy_stats.ttest_1samp(values, 0).pvalue) if n >= 3 else None
    p_sign = float(scipy_stats.binomtest(max(under, over), n, alternative='greater').pvalue)

    return {
        'n': n,
        'under': under,
        'over': over,
        'mean': round(sum(values) / n, 2),
        'median': round(median, 2),
        'p_t': significant(p_t),
        'p_sign': significant(p_sign),
        'size': round(sum(abs(value) for value in values) / n, 2),
        'size_over': round(sum(1 for value in values if abs(value) > threshold) / n, 2)
    }


def extremes(rows: list[dict], key: str, k: int) -> dict:
    """
    The elections where a party or group was most under- and overestimated.

    Parameters
    ----------
    rows : list of dict
        One row per election, with ``label`` and the error under ``key`` (``None`` when absent).
    key : str
        The key of the error.
    k : int
        How many rows to keep on each side.

    Returns
    -------
    dict
        ``under``: up to ``k`` ``{'label', 'error'}`` with the most negative errors (below zero), ascending;
        ``over``: up to ``k`` with the most positive (above zero), descending. Errors with two decimals.
    """
    kept = [{'label': row['label'], 'error': round(float(row[key]), 2)} for row in rows if _is_number(row.get(key))]
    under = sorted((row for row in kept if row['error'] < 0), key=lambda row: row['error'])[:k]
    over = sorted((row for row in kept if row['error'] > 0), key=lambda row: -row['error'])[:k]

    return {'under': under, 'over': over}


def size(rows: list[dict], parties: list[str], threshold: float) -> dict:
    """
    Size of the errors over every (election, party) pair.

    Parameters
    ----------
    rows : list of dict
        One row per election, with the error of each party under its name (``None`` when absent).
    parties : list of str
        The parties to count.
    threshold : float
        Absolute error, in points, above which an error counts as large.

    Returns
    -------
    dict
        ``n_pairs``, ``mean`` (mean absolute error) and ``share_over`` (share above ``threshold``), the last two with
        two decimals (``None`` without pairs).
    """
    errors = [abs(float(row[party])) for row in rows for party in parties if _is_number(row.get(party))]
    if not errors:
        return {'n_pairs': 0, 'mean': None, 'share_over': None}

    return {
        'n_pairs': len(errors),
        'mean': round(sum(errors) / len(errors), 2),
        'share_over': round(sum(1 for error in errors if error > threshold) / len(errors), 2)
    }
