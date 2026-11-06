"""Tests of the pure statistics of the herd study (comparisons between pairs of houses)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.studies.herd.stats import compare, comparisons, count, dispersion, period_of, permutation_p

PARAMS = {'pair_days': 7, 'periods': {'final': [0, 21], 'mid': [21, 56], 'far': [56, 365]}}


def test_period_of_half_open():
    p = PARAMS['periods']
    assert [period_of(d, p) for d in (0, 20.5, 21, 55.5, 56, 365, 365.5)] == [
        'final', 'final', 'mid', 'mid', 'far', 'far', None]


def test_comparisons_pairs_other_houses_within_a_week():
    polls = pd.DataFrame({'poll': ['a', 'b', 'c', 'd'], 'pollster': ['A', 'B', 'A', 'C'], 'days': [30, 25, 27, 40],
                          'n': [1000.0] * 4, 'PP': [30.0, 34.0, 31.0, 30.0], 'PSOE': [25.0, np.nan, 25.0, 25.0]})
    rows = comparisons(polls, ['PP', 'PSOE'], PARAMS)
    # a-b and b-c (A|B); a-c is the same house; d is 10+ days away from b and 13 from c, 10 from a
    assert sorted(rows[['poll_i', 'poll_j']].apply(tuple, axis=1)) == [('a', 'b'), ('b', 'c')]
    ab = rows[(rows.poll_i == 'a') & (rows.poll_j == 'b')].iloc[0]
    assert (ab.pair, ab['name'], ab.days, ab.period, ab.d2) == ('A|B', 'PP', 27.5, 'mid', 16.0)
    assert ab.v == pytest.approx(4.352)          # 1e4 · 0,32 · 0,68 · 2/1000
    assert set(rows.name) == {'PP'}             # PSOE missing in b: no PSOE comparison


def test_comparisons_empty_has_columns():
    polls = pd.DataFrame({'poll': ['a'], 'pollster': ['A'], 'days': [30], 'n': [1000.0], 'PP': [30.0]})
    assert list(comparisons(polls, ['PP'], PARAMS).columns) == [
        'pair', 'poll_i', 'poll_j', 'days', 'period', 'name', 'd2', 'v']


def test_compare_and_dispersion_on_the_same_house_pairs():
    rows = pd.DataFrame({'pair': ['A|B'] * 6 + ['A|C'] * 2, 'period': ['far'] * 3 + ['final'] * 3 + ['final'] * 2,
                         'd2': [4.0] * 3 + [1.0] * 3 + [9.0] * 2, 'v': [2.0] * 8})
    out = compare(rows, 'far', 'final', 3)
    assert out['pairs'] == 1 and (out['rows_a'], out['rows_b']) == (3, 3)
    assert out['r_a'] == pytest.approx(2 ** 0.5) and out['r_b'] == pytest.approx(0.5 ** 0.5)
    assert out['change'] == pytest.approx(-0.5) and out['falls'] is True
    assert dispersion(rows[rows.period == 'far']) == pytest.approx(2 ** 0.5) and dispersion(rows.iloc[:0]) is None
    assert compare(rows, 'far', 'final', 4) is None


def test_compare_without_differences_in_a_is_none():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'period': ['far'] * 3 + ['final'] * 3,
                         'd2': [0.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    assert compare(rows, 'far', 'final', 3) is None and permutation_p(rows, 'far', 'final', 3, 200, 0) is None


def test_permutation_p_small_exact_case():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'period': ['far'] * 3 + ['final'] * 3,
                         'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    p = permutation_p(rows, 'far', 'final', 3, 2000, 0)
    assert abs(p - 0.05) < 0.015                 # exact: 1 of the C(6,3) = 20 arrangements
    assert permutation_p(rows, 'far', 'final', 3, 2000, 0) == p


def test_count_and_sign():
    results = [{'falls': True, 'p': 0.01, 'change': -0.5}, {'falls': True, 'p': 0.3, 'change': -0.1},
               {'falls': True, 'p': 0.2, 'change': -0.2}, {'falls': False, 'p': 0.9, 'change': 0.4}]
    assert count(results) == {'n': 4, 'k': 3, 'sign_p': 0.312, 'clear': 1, 'median_change': pytest.approx(-0.15)}
    assert count([]) == {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}
