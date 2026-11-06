"""Tests de `mtpy/lib/studies/shy/stats.py`: el error de un grupo con la regla de suma, las cifras, los extremos y el
tamaño, con cifras calculadas a mano."""
import pandas as pd
import pytest

from mtpy.lib.studies.shy import stats

RES = {'PSOE': 25.4, 'UP': 18.6, 'IU': 4.2, 'PP': 33.1}
POLLS = pd.DataFrame([{'UP': 20.0, 'IU': 4.0, 'PSOE': 24.0, 'PP': 34.0, 'VOX': None}, {'UP': 18.0, 'IU': None, 'PSOE': 25.0, 'PP': 35.0, 'VOX': None}])


def test_group_error_sums_the_members_each_poll_estimates():
    assert stats.group_error(POLLS, ['SUMAR', 'UP', 'IU'], RES) == (pytest.approx(0.3), 'UP+IU')   # (24 − 22.8 + 18 − 18.6) / 2
    assert stats.group_error(POLLS, ['PSOE', 'PSC'], RES) == (pytest.approx(-0.9), 'PSOE')
    assert stats.group_error(POLLS, ['PSOE', 'UP', 'IU'], RES) == (pytest.approx(-0.6), 'PSOE+UP+IU')   # the block rule is the same


def test_group_error_is_none_when_no_poll_estimates_the_party():
    assert stats.group_error(POLLS, ['VOX'], RES | {'VOX': 9.1}) == (None, 'VOX')
    assert stats.group_error(POLLS, ['VOX'], RES) == (None, None)
    assert stats.group_error(POLLS.iloc[0:0], ['PSOE'], RES) == (None, 'PSOE')


def test_figures_counts_signs_and_tests_the_mean():
    assert stats.figures([1.0, 2.0, 3.0]) == {'n': 3, 'under': 0, 'over': 3, 'mean': 2.0, 'median': 2.0, 'p_t': 0.0742, 'p_sign': 0.125, 'size': 2.0, 'size_over': 0.33}
    assert stats.figures([-1.0, 0.0, 1.0]) == {'n': 3, 'under': 1, 'over': 1, 'mean': 0.0, 'median': 0.0, 'p_t': 1.0, 'p_sign': 0.875, 'size': 0.67, 'size_over': 0.0}
    assert stats.figures([2.5, None, -1.0])['p_t'] is None and stats.figures([2.5, None, -1.0])['n'] == 2
    assert stats.figures([]) is None and stats.figures([None]) is None


def test_extremes_and_size():
    rows = [{'label': 'A', 'VOX': -6.01}, {'label': 'B', 'VOX': 3.39}, {'label': 'C', 'VOX': None}, {'label': 'D', 'VOX': -0.5}, {'label': 'E', 'VOX': 2.1}]
    assert stats.extremes(rows, 'VOX', 3) == {'under': [{'label': 'A', 'error': -6.01}, {'label': 'D', 'error': -0.5}], 'over': [{'label': 'B', 'error': 3.39}, {'label': 'E', 'error': 2.1}]}
    rows = [{'PSOE': 3.4, 'PP': -2.3, 'VOX': None, 'UP': -0.1}, {'PSOE': -3.7, 'PP': 0.9, 'VOX': 0.9333, 'UP': 0.3667}]
    assert stats.size(rows, ['PSOE', 'PP', 'VOX', 'UP'], 2) == {'n_pairs': 7, 'mean': 1.67, 'share_over': 0.43}
    assert stats.significant(0.0742195) == 0.0742 and stats.significant(None) is None
