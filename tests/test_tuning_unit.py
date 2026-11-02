"""Unit tests of the tuning harness of the average on future polls (no database)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.forecaster import Forecaster
from mtpy.lib.tuning import autocorrelation, predict_polls, roughness, score_polls


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


def test_predict_polls_handles_party_names_that_are_not_python_identifiers():
    fc = _fake_fitted_forecaster(effects={})
    fc.forecast['EH Bildu'] = 5.
    fc.forecast['UPN-PP'] = 2.
    polls = pd.DataFrame({'date': [pd.Timestamp('2026-03-12')], 'pollster_id': [1], 'sponsor_id': [0], 'pollster': ['A'],
                          'sample_size': [1000], 'EH Bildu': [6.], 'UPN-PP': [3.]})
    out = predict_polls(fc, polls, ['EH Bildu', 'UPN-PP'])
    assert out['party'].tolist() == ['EH Bildu', 'UPN-PP']
    assert out['published'].tolist() == [6., 3.]
    assert out['residual'].tolist() == pytest.approx([1., 1.])


def test_score_polls_rmse_mae_skill_and_by_party():
    pred = pd.DataFrame({'party': ['PP', 'PP', 'PSOE'], 'published': [32., 28., 20.], 'predicted': [30., 30., 20.],
                         'residual': [2., -2., 0.], 'sample_size': [1000., 1000., np.nan], 'date': pd.to_datetime(['2026-01-02'] * 3)})
    s = score_polls(pred, k=7)
    assert s['k'] == 7 and s['n_cells'] == 3 and s['n_polls'] == 1
    assert s['rmse'] == pytest.approx(np.sqrt(8 / 3)) and s['mae'] == pytest.approx(4 / 3)
    assert s['rmse_by_party']['PP'] == pytest.approx(2.) and s['rmse_by_party']['PSOE'] == 0.
    assert s['skill'] < 1


def test_score_polls_without_cells_is_nan_not_an_error():
    s = score_polls(pd.DataFrame(columns=['party', 'published', 'predicted', 'residual', 'sample_size', 'date']), k=28)
    assert s['n_cells'] == 0 and np.isnan(s['rmse']) and np.isnan(s['skill'])


def test_score_polls_skill_is_nan_without_any_sample_size():
    pred = pd.DataFrame({'party': ['PP'], 'published': [30.], 'predicted': [29.], 'residual': [1.],
                         'sample_size': [np.nan], 'date': pd.to_datetime(['2026-01-02'])})
    s = score_polls(pred, k=7)
    assert s['n_cells'] == 1 and s['rmse'] == pytest.approx(1.) and np.isnan(s['skill'])


def test_autocorrelation_and_roughness():
    pred = pd.DataFrame({'party': ['PP'] * 4, 'date': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04']),
                         'residual': [1., 1., -1., -1.]})
    assert autocorrelation(pred) > 0
    days = pd.date_range('2026-01-01', periods=5, freq='D')
    assert roughness(pd.DataFrame({'PP': [30., 30., 30., 30., 30.]}, index=days), ['PP']) == 0.
    assert roughness(pd.DataFrame({'PP': [30., 31., 30., 31., 30.]}, index=days), ['PP']) == pytest.approx(1.)
    assert roughness(pd.DataFrame({'PP': [np.nan, 30., 30.5, np.nan, np.nan]}, index=days), ['PP']) == pytest.approx(0.5)  # sin el relleno


def test_autocorrelation_stays_within_each_party_and_is_nan_with_few_pairs():
    dates = pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03'])
    pred = pd.DataFrame({'party': ['PP', 'PP', 'PSOE'], 'date': dates, 'residual': [1., 2., 5.]})
    assert np.isnan(autocorrelation(pred))
    assert np.isnan(autocorrelation(pred.iloc[:0]))
