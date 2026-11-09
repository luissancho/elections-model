"""Tests of the pure statistics of the herd study (comparisons between pairs of houses)."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mtpy.lib.studies.herd.stats import (COMPARISON_COLUMNS, _blocks, bootstrap_median, bootstrap_pooled, bucket_of,
                                         by_buckets, compare, comparisons, count, dispersion, histogram,
                                         null_distribution, period_of, permutation_p, pooled, pooled_p, pooled_terms)

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


def test_comparisons_skip_polls_without_pollster():
    polls = pd.DataFrame({'poll': ['a', 'b', 'c'], 'pollster': ['A', None, 'B'], 'days': [30, 29, 28],
                          'n': [1000.0] * 3, 'PP': [30.0, 32.0, 31.0]})
    rows = comparisons(polls, ['PP'], PARAMS)
    assert list(rows[['poll_i', 'poll_j']].apply(tuple, axis=1)) == [('a', 'c')]


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


def two_pairs():
    """Two pairs of houses whose number of rows changes between the periods: A|B has 9 far rows and 3 final rows,
    A|C 3 far rows and 9 final rows (all ``v`` = 2)."""
    return pd.DataFrame({'pair': ['A|B'] * 12 + ['A|C'] * 12,
                         'poll_i': ['p%d' % i for i in range(24)], 'poll_j': ['q%d' % i for i in range(24)],
                         'period': ['far'] * 9 + ['final'] * 3 + ['far'] * 3 + ['final'] * 9,
                         'd2': [4.0] * 9 + [5.0] * 3 + [1.0] * 3 + [1.2] * 9, 'v': [2.0] * 24})


def test_compare_weighs_each_pair_of_houses_the_same_in_both_periods():
    # A|B: far Σd2 36, Σv 18, r² 2; final Σd2 15, Σv 6, r² 2,5.  A|C: far Σd2 3, Σv 6, r² 0,5; final Σd2 10,8,
    # Σv 18, r² 0,6. Both pairs get further apart. Weights w = 1 / (1/Σv_a + 1/Σv_b): A|B 1/(1/18 + 1/6) = 4,5,
    # A|C 1/(1/6 + 1/18) = 4,5. R_a² = (4,5·2 + 4,5·0,5) / 9 = 1,25 and R_b² = (4,5·2,5 + 4,5·0,6) / 9 = 1,55:
    # change = sqrt(1,55 / 1,25) − 1 = sqrt(1,24) − 1 = 0,11355, no fall. Pooled over all rows it would fall:
    # R_a² = 39/24 = 1,625, R_b² = 25,8/24 = 1,075, change −0,187, only because A|B weighs more far away.
    out = compare(two_pairs(), 'far', 'final', 3)
    assert out['pairs'] == 2 and (out['rows_a'], out['rows_b']) == (12, 12)
    assert out['r_a'] == pytest.approx(1.25 ** 0.5) and out['r_b'] == pytest.approx(1.55 ** 0.5)
    assert out['change'] == pytest.approx(1.24 ** 0.5 - 1) and out['falls'] is False


def statistic(rows, is_a):
    """R_b² / R_a² of ``rows`` with the labels ``is_a`` (reference for the permutation test)."""
    num_a = num_b = 0.0

    for pair in rows['pair'].unique():
        mask = (rows['pair'] == pair).to_numpy()
        in_a, in_b = mask & is_a, mask & ~is_a
        d2_a, v_a, d2_b, v_b = (rows['d2'][in_a].sum(), rows['v'][in_a].sum(), rows['d2'][in_b].sum(),
                                rows['v'][in_b].sum())
        weight = 1 / (1 / v_a + 1 / v_b)
        num_a += weight * d2_a / v_a
        num_b += weight * d2_b / v_b

    return num_b / num_a


def test_permutation_p_shuffles_within_each_pair_with_its_weights():
    # Exact p: every arrangement of the labels within each pair (C(12,9) · C(12,3) = 220 · 220), with d2 that differ
    # inside the pairs. Exact p ≈ 0,393; the pooled statistic, shuffled the same way, would give ≈ 0,466
    rows = two_pairs()
    rows['d2'] = [5.5, 3.0, 6.5, 6.0, 2.0, 2.0, 3.5, 3.0, 5.0, 2.0, 5.5, 2.0,
                  1.0, 0.5, 1.0, 1.5, 2.0, 0.5, 2.0, 1.5, 1.5, 1.5, 1.5, 1.0]
    observed = statistic(rows, (rows['period'] == 'far').to_numpy())
    options = []

    for start in (0, 12):
        n_a = int((rows['period'][start:start + 12] == 'far').sum())
        options.append([np.isin(np.arange(12), chosen) for chosen in combinations(range(12), n_a)])

    below = [statistic(rows, np.concatenate([x, y])) <= observed * (1 + 1e-9) for x in options[0] for y in options[1]]
    exact = float(np.mean(below))
    p = permutation_p(rows, 'far', 'final', 3, 4000, 0)
    assert exact == pytest.approx(0.3926, abs=1e-3) and abs(p - exact) < 0.025


def test_compare_without_differences_in_a_is_none():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'poll_i': ['p%d' % i for i in range(6)],
                         'poll_j': ['q%d' % i for i in range(6)], 'period': ['far'] * 3 + ['final'] * 3,
                         'd2': [0.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    assert compare(rows, 'far', 'final', 3) is None and permutation_p(rows, 'far', 'final', 3, 200, 0) is None


def test_permutation_p_small_exact_case():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'poll_i': ['p%d' % i for i in range(6)],
                         'poll_j': ['q%d' % i for i in range(6)], 'period': ['far'] * 3 + ['final'] * 3,
                         'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    p = permutation_p(rows, 'far', 'final', 3, 2000, 0)
    assert abs(p - 0.05) < 0.015                 # exact: 1 of the C(6,3) = 20 arrangements
    assert permutation_p(rows, 'far', 'final', 3, 2000, 0) == p


def test_blocks_sum_the_rows_of_each_pair_of_polls():
    group = pd.DataFrame({'pair': ['A|B'] * 4, 'poll_i': ['a1', 'a1', 'a2', 'a2'], 'poll_j': ['b1', 'b1', 'b2', 'b2'],
                          'period': ['far', 'far', 'final', 'final'], 'd2': [1.0, 3.0, 0.5, 0.5],
                          'v': [2.0, 2.0, 1.0, 1.0]})
    d2, v, is_a = _blocks(group, 'far')
    assert d2.tolist() == [4.0, 1.0] and v.tolist() == [4.0, 2.0] and is_a.tolist() == [True, False]


def test_permutation_p_keeps_the_series_of_a_pair_of_polls_together():
    # 2 pairs of polls far (d2 4 per series) and 2 final (d2 1), two series each: C(4,2) = 6 arrangements of the
    # blocks, the observed fall is the extreme one: p = 1/6 (by rows it would be 1/C(8,4) = 1/70)
    rows = pd.DataFrame({'pair': ['A|B'] * 8, 'poll_i': ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4'],
                         'poll_j': ['b1'] * 2 + ['b2'] * 2 + ['b3'] * 2 + ['b4'] * 2,
                         'period': ['far'] * 4 + ['final'] * 4, 'd2': [4.0] * 4 + [1.0] * 4, 'v': [2.0] * 8})
    p = permutation_p(rows, 'far', 'final', 3, 3000, 0)
    assert abs(p - 1 / 6) < 0.025


def test_permutation_p_one_block_per_period():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'poll_i': ['a1'] * 3 + ['a2'] * 3, 'poll_j': ['b1'] * 3 + ['b2'] * 3,
                         'period': ['far'] * 3 + ['final'] * 3, 'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    p = permutation_p(rows, 'far', 'final', 3, 2000, 0)
    assert abs(p - 0.5) < 0.03                   # two arrangements: the observed one and the swap


def two_elections():
    """Two elections with one pair of houses each; by hand: R_a = √2, R_b² = 22,5/33 (see the plan)."""
    e1 = pd.DataFrame({'pair': ['A|B'] * 6, 'poll_i': ['a%d' % i for i in range(6)],
                       'poll_j': ['b%d' % i for i in range(6)], 'period': ['far'] * 3 + ['final'] * 3,
                       'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    e2 = pd.DataFrame({'pair': ['C|D'] * 7, 'poll_i': ['c%d' % i for i in range(7)],
                       'poll_j': ['d%d' % i for i in range(7)], 'period': ['far'] * 4 + ['final'] * 3,
                       'd2': [2.0] * 4 + [1.0] * 3, 'v': [1.0] * 7})
    return [e1, e2]


def test_pooled_weighs_the_pairs_of_every_election():
    e1, e2 = two_elections()
    one = pooled([e1], 'far', 'final', 3)
    same = compare(e1, 'far', 'final', 3)
    assert (one['elections'], one['pairs'], one['rows_a'], one['rows_b']) == (1, 1, 3, 3)
    assert one['r_a'] == pytest.approx(same['r_a']) and one['r_b'] == pytest.approx(same['r_b'])
    both = pooled([e1, e2], 'far', 'final', 3)
    # w1 = 3 (r²: 2 -> 0,5), w2 = 12/7 (r²: 2 -> 1): R_a² = (6 + 24/7)/(33/7) = 2,
    # R_b² = (1,5 + 12/7)/(33/7) = 22,5/33
    assert (both['elections'], both['pairs'], both['rows_a'], both['rows_b']) == (2, 2, 7, 6)
    assert both['r_a'] == pytest.approx(2 ** 0.5) and both['r_b'] == pytest.approx((22.5 / 33) ** 0.5)
    assert both['change'] == pytest.approx(-0.4161, abs=1e-4) and both['falls'] is True
    assert pooled([e1.iloc[:0]], 'far', 'final', 3) is None and pooled([], 'far', 'final', 3) is None


def block_frame(prefix, far_d2, final_d2):
    """One pair of houses whose pairs of polls have two series each: ``far_d2``/``final_d2`` per block."""
    blocks = [('far', d2) for d2 in far_d2] + [('final', d2) for d2 in final_d2]
    rows = [{'pair': prefix, 'poll_i': '%s_i%d' % (prefix, k), 'poll_j': '%s_j%d' % (prefix, k), 'period': period,
             'd2': d2 / 2, 'v': 1.0} for k, (period, d2) in enumerate(blocks) for _ in range(2)]

    return pd.DataFrame(rows)


def test_pooled_p_matches_the_exact_enumeration():
    # two elections with 4 blocks each (2 far + 2 final): 6 · 6 = 36 arrangements enumerated with itertools over
    # the statistic Σ w·r²_b / Σ w·r²_a of the blocks
    frames = [block_frame('A|B', [3.0, 5.0], [2.0, 4.0]), block_frame('C|D', [6.0, 2.0], [1.0, 7.0])]
    blocks = [(np.array([3.0, 5.0, 2.0, 4.0]), np.array([2.0] * 4)),
              (np.array([6.0, 2.0, 1.0, 7.0]), np.array([2.0] * 4))]

    def terms(d2, v, in_a):
        d2_a, v_a, d2_b, v_b = d2[in_a].sum(), v[in_a].sum(), d2[~in_a].sum(), v[~in_a].sum()
        return v_a * v_b / (v_a + v_b) * np.array([d2_a / v_a, d2_b / v_b])

    def statistic_of(choices):
        total = sum(terms(d2, v, np.isin(np.arange(4), chosen)) for (d2, v), chosen in zip(blocks, choices))
        return total[1] / total[0]

    observed = statistic_of([(0, 1), (0, 1)])
    options = list(combinations(range(4), 2))
    exact = float(np.mean([statistic_of([x, y]) <= observed * (1 + 1e-9) for x in options for y in options]))
    assert 0 < exact < 1
    assert abs(pooled_p(frames, 'far', 'final', 3, 4000, 0) - exact) < 0.03


def test_pooled_terms_are_the_weighted_sums_of_each_election():
    e1, e2 = two_elections()
    terms = pooled_terms([e1, e2], 'far', 'final', 3)
    assert terms[0] == pytest.approx((6.0, 1.5)) and terms[1] == pytest.approx((24 / 7, 12 / 7))
    assert pooled_terms([e1.iloc[:0]], 'far', 'final', 3) == []
    # the pooled change follows from the terms: sqrt(ΣB/ΣA) − 1
    assert (sum(b for _, b in terms) / sum(a for a, _ in terms)) ** 0.5 - 1 == pytest.approx(-0.4161, abs=1e-4)


def test_bootstrap_pooled_interval_is_deterministic_and_brackets_the_change():
    terms = [(6.0, 1.5), (24 / 7, 12 / 7), (4.0, 3.0), (2.0, 2.5)]
    ci = bootstrap_pooled(terms, 2000, 0)
    change = (sum(b for _, b in terms) / sum(a for a, _ in terms)) ** 0.5 - 1
    assert ci['lo'] <= change <= ci['hi'] and bootstrap_pooled(terms, 2000, 0) == ci
    assert bootstrap_pooled(terms[:2], 2000, 0) is None
    assert bootstrap_pooled([(0.0, 0.0)] * 4, 50, 0) is None          # every resample divides by zero


def test_null_distribution_matches_the_exact_enumeration():
    # the two-election, four-block case of test_pooled_p_matches_the_exact_enumeration: enumerate the 36 arrangements
    # (statistic Σw·r²_b / Σw·r²_a), p_one = share <= observed, p_two = share with |√ratio − 1| >= |observed change|
    rows = [block_frame('A|B', [3.0, 5.0], [2.0, 4.0]), block_frame('C|D', [6.0, 2.0], [1.0, 7.0])]
    blocks = [(np.array([3.0, 5.0, 2.0, 4.0]), np.array([2.0] * 4)),
              (np.array([6.0, 2.0, 1.0, 7.0]), np.array([2.0] * 4))]

    def terms(d2, v, in_a):
        d2_a, v_a, d2_b, v_b = d2[in_a].sum(), v[in_a].sum(), d2[~in_a].sum(), v[~in_a].sum()
        return v_a * v_b / (v_a + v_b) * np.array([d2_a / v_a, d2_b / v_b])

    def change_of(choices):
        total = sum(terms(d2, v, np.isin(np.arange(4), chosen)) for (d2, v), chosen in zip(blocks, choices))
        return (total[1] / total[0]) ** 0.5 - 1

    observed = change_of([(0, 1), (0, 1)])
    options = list(combinations(range(4), 2))
    changes = np.array([change_of([x, y]) for x in options for y in options])
    exact_one = float(np.mean(changes <= observed + 1e-9))
    exact_two = float(np.mean(np.abs(changes) >= abs(observed) - 1e-9))
    assert 0 < exact_one < 1 and 0 < exact_two < 1
    out = null_distribution(rows, 'far', 'final', 3, 4000, 0)
    assert out['observed'] == pytest.approx(observed)
    assert out['permuted'].shape == (4000,) and isinstance(out['count_one'], int)
    assert abs(out['p_one'] - exact_one) < 0.03 and abs(out['p_two'] - exact_two) < 0.03
    assert out['p_one'] == pytest.approx((out['count_one'] + 1) / 4001)
    assert out['p_two'] == pytest.approx((out['count_two'] + 1) / 4001)
    assert null_distribution([], 'far', 'final', 3, 100, 0) is None


def test_null_distribution_with_a_zero_change_has_p_two_near_one():
    # identical far and final rows: observed change 0, every arrangement gives |change| >= 0
    rows = pd.DataFrame({'pair': ['A|B'] * 8, 'poll_i': ['a%d' % i for i in range(8)],
                         'poll_j': ['b%d' % i for i in range(8)], 'period': ['far'] * 4 + ['final'] * 4,
                         'd2': [2.0] * 8, 'v': [1.0] * 8})
    out = null_distribution([rows], 'far', 'final', 3, 500, 0)
    assert out['observed'] == pytest.approx(0.0) and out['p_two'] == pytest.approx(1.0)


def test_histogram_bins_cover_the_values_with_fixed_width():
    assert histogram([-0.012, 0.004, 0.031], 0.01) == [
        {'lo': -0.02, 'hi': -0.01, 'count': 1}, {'lo': -0.01, 'hi': 0.0, 'count': 0},
        {'lo': 0.0, 'hi': 0.01, 'count': 1}, {'lo': 0.01, 'hi': 0.02, 'count': 0},
        {'lo': 0.02, 'hi': 0.03, 'count': 0}, {'lo': 0.03, 'hi': 0.04, 'count': 1}]
    assert histogram([0.05, 0.05], 0.01) == [{'lo': 0.05, 'hi': 0.06, 'count': 2}]
    assert histogram([], 0.01) == []


def test_pooled_p_small_with_a_strong_fall_and_none_without_pairs():
    # each election: 4 far blocks with d2 10 and 4 final blocks with d2 1: the observed is the extreme of
    # 70 · 70 arrangements
    frames = [block_frame('A|B', [10.0] * 4, [1.0] * 4), block_frame('C|D', [10.0] * 4, [1.0] * 4)]
    assert pooled_p(frames, 'far', 'final', 3, 2000, 0) < 0.01
    assert pooled_p([], 'far', 'final', 3, 200, 0) is None


def test_bootstrap_median_interval_contains_the_median_and_is_deterministic():
    changes = [-0.5, -0.1, -0.2, 0.4]
    ci = bootstrap_median(changes, 2000, 0)
    assert ci['lo'] <= -0.15 <= ci['hi'] and min(changes) <= ci['lo'] and ci['hi'] <= max(changes)
    assert bootstrap_median(changes, 2000, 0) == ci and bootstrap_median([-0.1, 0.2], 2000, 0) is None


def test_count_and_sign():
    results = [{'falls': True, 'p': 0.01, 'change': -0.5}, {'falls': True, 'p': 0.3, 'change': -0.1},
               {'falls': True, 'p': 0.2, 'change': -0.2}, {'falls': False, 'p': 0.9, 'change': 0.4}]
    assert count(results) == {'n': 4, 'k': 3, 'sign_p': 0.312, 'clear': 1, 'median_change': pytest.approx(-0.15)}
    assert count([]) == {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}


BUCKETS = [[0, 2], [3, 4], [5, 7], [8, 11], [12, 17], [18, 26], [27, 52]]


def test_bucket_of_edges():
    days = (0, 20, 20.5, 21, 34.5, 35, 55, 56, 83, 84, 125, 126, 188, 189, 365, 400)
    assert [bucket_of(d, BUCKETS) for d in days] == [
        (0, 2), (0, 2), (0, 2), (3, 4), (3, 4), (5, 7), (5, 7), (8, 11), (8, 11), (12, 17), (12, 17), (18, 26),
        (18, 26), (27, 52), (27, 52), None]


def test_by_buckets_counts_distinct_poll_pairs():
    rows = pd.DataFrame({'poll_i': ['a', 'a', 'a', 'a', 'c'], 'poll_j': ['b', 'b', 'c', 'c', 'd'],
                         'days': [10.0, 10.0, 12.0, 12.0, 30.0], 'd2': [4.0, 0.0, 4.0, 0.0, 9.0], 'v': [2.0] * 5})
    out = by_buckets(rows, BUCKETS, 2)
    assert [(b['lo'], b['hi'], b['pairs'], b['comparisons']) for b in out[:3]] == [
        (0, 2, 2, 4), (3, 4, 1, 1), (5, 7, 0, 0)]
    assert out[0]['ratio'] == pytest.approx(1.0) and out[1]['ratio'] is None and out[2]['ratio'] is None


def test_by_buckets_without_rows():
    empty = pd.DataFrame(columns=COMPARISON_COLUMNS)
    out = by_buckets(empty, BUCKETS, 2)
    assert len(out) == 7 and all((b['pairs'], b['comparisons'], b['ratio']) == (0, 0, None) for b in out)
