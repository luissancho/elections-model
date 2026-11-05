"""Tests de `mtpy/lib/studies/herd/stats.py`: la dispersión de los sondeos frente a la curva, el rebaño de cada casa y
las pruebas, con cifras calculadas a mano sobre el paquete sintético de `write_herd_bundle`."""
import os

import numpy as np
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.herd import data, stats
from tests.fakes import write_herd_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARAMS = data.load_params(FileSystem(os.path.join(ROOT, 'data')))
ELECTION = pd.Timestamp('2023-07-23')
NAMES = ['PP', 'PSOE', 'VOX', 'SUMAR']


@pytest.fixture(scope='module')
def cycles(tmp_path_factory):
    """Los dos ciclos pasados del paquete sintético, por fecha de la elección."""
    fs = FileSystem(str(tmp_path_factory.mktemp('herd')))
    write_herd_bundle(fs)
    past, _ = data.read_cycles(bundle.BundleReader(fs), PARAMS)
    return {cycle.event_date: cycle for cycle in past}


def _rows(cycle):
    """`residual_rows` de un ciclo con las cuatro series y los parámetros del repositorio."""
    return stats.residual_rows(cycle, NAMES, pd.Timestamp(cycle.event_date), PARAMS)


@pytest.fixture(scope='module')
def herd_2023(cycles):
    rows = _rows(cycles['2023-07-23'])
    return rows[rows['pollster'] != 'CIS']


@pytest.fixture(scope='module')
def herd_2023_cis(cycles):
    rows = _rows(cycles['2023-07-23'])
    return rows[rows['pollster'] == 'CIS']


@pytest.fixture(scope='module')
def herd_2019(cycles):
    rows = _rows(cycles['2019-11-10'])
    return rows[rows['pollster'] != 'CIS']


@pytest.fixture(scope='module')
def polls_2023(cycles):
    return cycles['2023-07-23'].polls


@pytest.fixture
def cycle_with_a_null_share(cycles):
    cycle = cycles['2023-07-23']
    polls = cycle.polls.copy()
    polls.loc[polls.index[0], 'VOX'] = np.nan
    return data.Cycle(event_date=cycle.event_date, label=cycle.label, current=cycle.current, polls=polls,
                      parties=list(cycle.parties), curve=cycle.curve, errors=dict(cycle.errors),
                      inputs=dict(cycle.inputs))


def test_dispersion_of_the_2023_windows(herd_2023):
    rows = herd_2023
    assert stats.dispersion(stats.in_days(rows, 56, 365)) == {'k': 3, 'ratio': 1.57}
    assert stats.dispersion(stats.in_days(rows, 0, 55)) == {'k': 7, 'ratio': 1.31}
    assert stats.dispersion(stats.in_days(rows, 0, 20)) == {'k': 4, 'ratio': 1.14}
    assert stats.dispersion(stats.in_days(rows, 189, 365)) == {'k': 0, 'ratio': None}


def test_polls_without_sample_or_weight_are_out(herd_2023):
    # Panel X y Celeste Tel nunca entran; el CIS lo separa quien llama
    assert set(herd_2023['pollster']) == {'GAD3', 'NC Report', 'Sigma Dos'}


def test_residual_rows_columns_and_variance(herd_2023, herd_2023_cis):
    assert list(herd_2023.columns) == ['poll', 'pollster', 'days', 'name', 'resid', 'var']
    first = herd_2023[herd_2023['poll'] == 0]
    assert first['days'].tolist() == [144] * 4 and first['resid'].tolist() == [2.0, -2.0, 2.0, -2.0]
    assert first['var'].round(3).tolist() == [2.211, 2.1, 1.131, 1.056]
    assert herd_2023_cis['var'].mean() == pytest.approx(0.406125)


def test_house_ratio_of_gad3_and_nc_report_in_the_final_window(herd_2023):
    last = stats.in_days(herd_2023, 0, 55)
    assert stats.house_ratio(last[last['pollster'] == 'GAD3']) == {'k': 3, 'h': 0.32, 'p': 0.0975}
    assert stats.house_ratio(last[last['pollster'] == 'NC Report']) == {'k': 3, 'h': 1.97, 'p': 0.98}
    assert stats.house_ratio(last[last['pollster'] == 'Sigma Dos']) == {'k': 1, 'h': None, 'p': None}


def test_cis_apart_against_the_curve(herd_2023_cis):
    assert stats.dispersion(stats.in_days(herd_2023_cis, 0, 55)) == {'k': 1, 'ratio': 4.96}
    assert stats.dispersion(herd_2023_cis) == {'k': 2, 'ratio': 4.3}


def test_local_consensus_excludes_the_house_and_the_cis(herd_2019):
    # el ciclo de 2019 no tiene curva
    assert stats.dispersion(stats.in_days(herd_2019, 0, 55)) == {'k': 3, 'ratio': 0.96}
    assert stats.dispersion(stats.in_days(herd_2019, 56, 365)) == {'k': 3, 'ratio': 2.88}


def test_local_level_needs_two_other_polls(cycles):
    polls = cycles['2019-11-10'].polls.assign(end=lambda f: pd.to_datetime(f['end_date']))
    assert stats.local_level(polls, 0, 'PP', 10, ['CIS']) == 31.5
    assert stats.local_level(polls, 0, 'PP', 10, ['CIS', 'Sigma Dos']) is None


def test_curve_level_falls_back_to_the_last_day_before():
    curve = pd.DataFrame({'PP': [30.0, 31.0]}, index=pd.to_datetime(['2023-07-01', '2023-07-02']))
    assert stats.curve_level(curve, 'PP', pd.Timestamp('2023-07-09')) == 31.0
    assert stats.curve_level(curve, 'PP', pd.Timestamp('2023-06-30')) is None


def test_residuals_skip_only_the_missing_series(cycle_with_a_null_share):
    # un sondeo con VOX nulo
    rows = stats.residual_rows(cycle_with_a_null_share, ['PP', 'PSOE', 'VOX', 'SUMAR'], ELECTION, PARAMS)
    assert rows.loc[rows['poll'] == 0, 'name'].tolist() == ['PP', 'PSOE', 'SUMAR']


def test_main_series_picks_the_four_largest(polls_2023):
    assert stats.main_series(polls_2023, ['PP', 'PSOE', 'VOX', 'SUMAR', 'Cs'], 4) == ['PP', 'PSOE', 'VOX', 'SUMAR']


def test_sign_test_and_accuracy():
    assert stats.sign_test(7, 7) == {'n': 7, 'k': 7, 'p': 0.00781}
    assert stats.sign_test(2, 2)['p'] == 0.25 and stats.sign_test(0, 0) is None
    rows = [{'h': 0.32, 'last_error_avg': 1.5, 'last_error_blocks': -2.0},
            {'h': 1.97, 'last_error_avg': 2.5, 'last_error_blocks': 1.0},
            {'h': 0.5, 'last_error_avg': None, 'last_error_blocks': None}]
    assert stats.accuracy(rows) == {'n': 2, 'herd': {'k': 1, 'error_avg': 1.5, 'error_blocks': 2.0},
                                    'rest': {'k': 1, 'error_avg': 2.5, 'error_blocks': 1.0}}
    assert stats.accuracy(rows[:1]) is None
