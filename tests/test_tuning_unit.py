"""Unit tests of the tuning harness of the average on future polls (no database)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.forecaster import Forecaster
from mtpy.lib.tuning import predict_polls


def _fake_fitted_forecaster(effects=None):
    """Fitted Forecaster without database: PP 30 flat and PSOE 20 flat up to 2026-03-10 plus 10 days."""
    fc = Forecaster.__new__(Forecaster)
    dates = pd.date_range('2026-01-01', '2026-03-20', freq='D', name='date')
    fc.forecast = pd.DataFrame({'PP': 30., 'PSOE': 20., '-': 50.}, index=dates)
    fc.date_fit_last = pd.Timestamp('2026-03-10')
    fc.max_fc = 10
    fc.names = ['PP', 'PSOE']
    if effects is None:
        fc.house_effects = None
    else:
        index = pd.MultiIndex.from_tuples(list(effects), names=['pollster_id', 'name'])
        fc.house_effects = pd.DataFrame({'effect': list(effects.values())}, index=index)
    return fc


def test_predict_polls_reads_the_average_at_the_poll_date_and_adds_the_house_effect():
    fc = _fake_fitted_forecaster(effects={(1, 'PP'): 2.0})
    polls = pd.DataFrame({'date': [pd.Timestamp('2026-03-12')], 'pollster_id': [1], 'sponsor_id': [0], 'pollster': ['A'],
                          'sample_size': [1000], 'PP': [33.], 'PSOE': [19.]})
    out = predict_polls(fc, polls, ['PP', 'PSOE'])
    assert out.columns.tolist() == ['date', 'pollster_id', 'pollster', 'sample_size', 'party', 'published',
                                    'predicted', 'residual']
    assert out.loc[out.party == 'PP', 'predicted'].iloc[0] == pytest.approx(32.)
    assert out.loc[out.party == 'PP', 'residual'].iloc[0] == pytest.approx(1.)
    assert out.loc[out.party == 'PSOE', 'predicted'].iloc[0] == pytest.approx(20.)


def test_predict_polls_holds_the_last_value_beyond_the_prolongation_and_skips_missing_parties():
    fc = _fake_fitted_forecaster(effects={})
    polls = pd.DataFrame({'date': [pd.Timestamp('2026-04-30')], 'pollster_id': [2], 'sponsor_id': [0], 'pollster': ['B'],
                          'sample_size': [np.nan], 'PP': [31.], 'PSOE': [np.nan]})
    out = predict_polls(fc, polls, ['PP', 'PSOE'])
    assert out['party'].tolist() == ['PP']
    assert out['predicted'].iloc[0] == pytest.approx(30.)


def test_house_effect_of_is_zero_without_fitted_effects_or_for_an_unknown_house():
    assert _fake_fitted_forecaster(effects=None).house_effect_of(1, 'PP') == 0.0
    assert _fake_fitted_forecaster(effects={}).house_effect_of(1, 'PP') == 0.0
    fc = _fake_fitted_forecaster(effects={(1, 'PP'): 2.0})
    assert fc.house_effect_of(1, 'PP') == 2.0
    assert fc.house_effect_of(1, 'PSOE') == 0.0
    assert fc.house_effect_of(9, 'PP') == 0.0
