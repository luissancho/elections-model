"""Unit tests of the tuning harness of the average on future polls (no database)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib import tuning
from mtpy.lib.forecaster import Forecaster
from mtpy.lib.tuning import (autocorrelation, cycle_cuts, default_config, future_polls, predict_polls, roughness,
                             run_cut, score_polls)


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


def test_cycle_cuts_weekly_window_and_min_polls():
    dates = pd.Series(pd.to_datetime(['2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-06-15',
                                       '2023-07-01', '2023-07-10', '2023-07-15']))
    cuts = cycle_cuts('2023-07-23', dates, cut_days=7, window=365, min_polls=2, lookback=90)
    assert cuts[0] >= pd.Timestamp('2022-07-23') and cuts[-1] <= pd.Timestamp('2023-07-22')
    assert all((b - a).days == 7 for a, b in zip(cuts, cuts[1:]))
    assert pd.Timestamp('2022-10-01') not in cuts            # sin sondeos en los 90 días anteriores
    assert cuts[-1] in {pd.Timestamp('2023-07-22'), pd.Timestamp('2023-07-21'), pd.Timestamp('2023-07-20'),
                        pd.Timestamp('2023-07-19'), pd.Timestamp('2023-07-18'), pd.Timestamp('2023-07-17'),
                        pd.Timestamp('2023-07-16')}


def test_cycle_cuts_of_a_running_cycle_stop_today():
    dates = pd.Series(pd.date_range('2026-01-01', '2026-10-30', freq='7D'))
    cuts = cycle_cuts('2026-11-29', dates, today='2026-11-01')
    assert cuts[-1] <= pd.Timestamp('2026-11-01')


def test_default_config_is_the_live_model_and_a_fresh_dict():
    config = default_config()
    assert config == {'reg_params': {}, 'house_effects': True, 'he_params': {}, 'dispersion': True, 'herding': True,
                      'disp_params': {}, 'weights': None, 'max_fc': 10, 'label': 'default'}
    config['reg_params']['bw'] = 45
    assert default_config()['reg_params'] == {}


EVENT = '2023-07-23'
CUT = pd.Timestamp('2023-06-23')  # 30 días antes de la elección


def _poll_rows(dates, **columns):
    """Polls as `get_poll_series` returns them: one row per date, pollster ids 1, 2, ..."""
    n = len(dates)
    frame = pd.DataFrame({
        'date': pd.to_datetime(dates), 'pollster_id': range(1, n + 1), 'sponsor_id': [0] * n,
        'pollster': ['P{}'.format(i) for i in range(1, n + 1)], 'sample_size': [1000.] * n, 'mtype': ['Pública'] * n,
        'ctype': [None] * n, 'start_date': pd.to_datetime(dates), 'end_date': pd.to_datetime(dates),
        'event_date': pd.Timestamp(EVENT), 'computed': [True] * n, 'weight_over': [1.] * n
    })
    frame['days'] = (pd.Timestamp(EVENT) - frame['date']).dt.days
    for name, values in columns.items():
        frame[name] = values
    return frame


class FakeForecaster(Forecaster):
    """Forecaster without database: flat average (PP 30, PSOE 20) up to the cut, `n_series` polls in its series."""
    instances = []
    params = None

    def __init__(self, scope, event_date, **kwargs):
        self.scope, self.event_date, self.kwargs = scope, event_date, kwargs
        self.event_params, self.bmap, self.house_effects = FakeForecaster.params, kwargs.get('bmap'), None
        FakeForecaster.instances.append(self)

    def build_series(self):
        dates = pd.to_datetime(['2023-05-01', '2023-06-01', '2023-06-20'])
        index = pd.MultiIndex.from_arrays([dates, [1., 2., 3.], [0., 0., 0.]],
                                          names=['date', 'pollster_id', 'sponsor_id'])
        self.series_raw = pd.DataFrame({'pollster': ['P1', 'P2', 'P3']}, index=index)
        return self

    def fit_forecast(self, names=None, max_fc=0, fillna=False):
        self.fit_names, self.max_fc = names, max_fc
        days = pd.date_range('2023-05-01', '2023-07-23', freq='D', name='date')
        self.forecast = pd.DataFrame({'PP': 30., 'PSOE': 20.}, index=days)
        self.date_fit_last = CUT
        return self.forecast


def _patch_model(monkeypatch, future, cut_polls=None, smap=None):
    """Patch the database reads and the Forecaster of `tuning`; returns the list of `get_poll_series` calls."""
    calls = []

    def fake_poll_series(scope, event_dates, metric='pct', drange=None, drop_mtypes=None, drop_contexts=None):
        calls.append({'scope': scope, 'event_dates': event_dates, 'drange': drange, 'drop_mtypes': drop_mtypes})
        return (cut_polls if drange is not None and drange[1] is None else future).copy()

    params = {'parties': {'event': ['PP', 'PSOE'], 'polls': ['PP', 'PSOE', 'Cs']},
              'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {}, 'vs': {}}, 'smap': smap or {}}
    monkeypatch.setattr(tuning, 'get_poll_series', fake_poll_series)
    monkeypatch.setattr(tuning, 'Forecaster', FakeForecaster)
    FakeForecaster.instances, FakeForecaster.params = [], params
    return calls


def test_future_polls_keeps_the_window_and_the_polls_that_enter_the_model(monkeypatch):
    polls = _poll_rows(['2023-06-23', '2023-06-24', '2023-06-26', '2023-06-27', '2023-06-30', '2023-07-01'],
                       PP=[30., 31., 32., 33., 34., 35.])
    polls.loc[2, 'computed'] = False
    polls.loc[3, 'weight_over'] = 0.
    calls = _patch_model(monkeypatch, future=polls)
    out = future_polls('es', EVENT, CUT, 7, drop_mtypes=['Agregador'])
    assert calls[0]['drange'] == (23, 29) and calls[0]['drop_mtypes'] == ['Agregador']
    # (corte, corte + 7], computed y weight_over > 0
    assert out['date'].dt.strftime('%m-%d').tolist() == ['06-24', '06-30']
    assert out.index.tolist() == [0, 1]


def test_run_cut_scores_cumulative_horizons_with_the_live_flags(monkeypatch):
    future = _poll_rows(['2023-06-26', '2023-07-10'], PP=[31., 33.], PSOE=[20., 18.])
    calls = _patch_model(monkeypatch, future=future)
    out = run_cut('es', EVENT, CUT, config=default_config(), horizons=(7, 28))

    fc = FakeForecaster.instances[0]
    assert fc.kwargs['drange'] == 30 and fc.kwargs['house_effects'] is True and fc.kwargs['dispersion'] is True
    assert fc.kwargs['disp_params'] == {'herding': True} and fc.kwargs['weight_overrides'] is None
    assert fc.max_fc == 10 and fc.fit_names == ['PP', 'PSOE']
    assert calls[-1]['drange'] == (2, 29)  # una sola lectura de los sondeos futuros, con el mayor k

    polls = out['polls']
    assert polls['days_ahead'].tolist() == [3, 3, 17, 17] and polls['k'].tolist() == [7, 7, 28, 28]
    assert [s['k'] for s in out['scores']] == [7, 28]
    assert out['scores'][0]['n_polls'] == 1 and out['scores'][0]['rmse'] == pytest.approx(np.sqrt(0.5))
    assert out['scores'][1]['n_polls'] == 2 and out['scores'][1]['n_cells'] == 4
    assert out['roughness'] == 0. and np.isnan(out['autocorr'])
    meta = out['meta']
    assert meta['scope'] == 'es' and meta['event_date'] == EVENT and meta['cut'] == CUT and meta['drange'] == 30
    assert meta['n_polls'] == 3 and meta['n_overrides'] == 0 and meta['label'] == 'default'
    assert meta['max_fc'] == 10 and meta['names'] == ['PP', 'PSOE'] and out['seconds'] >= 0


def test_run_cut_maps_the_config_without_mutating_it_and_max_fc_overrides(monkeypatch):
    _patch_model(monkeypatch, future=_poll_rows(['2023-06-26'], PP=[31.], PSOE=[20.]))
    config = default_config() | {'herding': False, 'dispersion': False, 'house_effects': False,
                                 'disp_params': {'herd_prior': 10.}, 'reg_params': {'bw': 45},
                                 'he_params': {'tau': .16}}
    run_cut('es', EVENT, CUT, config=config, max_fc=3)
    fc = FakeForecaster.instances[0]
    assert fc.kwargs['disp_params'] == {'herd_prior': 10., 'herding': False}
    assert config['disp_params'] == {'herd_prior': 10.}
    assert fc.kwargs['house_effects'] is False and fc.kwargs['dispersion'] is False
    assert fc.kwargs['reg_params'] == {'bw': 45} and fc.kwargs['he_params'] == {'tau': .16}
    assert fc.max_fc == 3


def test_run_cut_without_future_polls_returns_a_complete_result(monkeypatch):
    _patch_model(monkeypatch, future=_poll_rows([]))
    out = run_cut('es', EVENT, CUT, config=default_config(), horizons=(7, 28))
    assert out['polls'].empty and {'days_ahead', 'k', 'party', 'residual'} <= set(out['polls'].columns)
    assert [s['n_polls'] for s in out['scores']] == [0, 0] and all(np.isnan(s['rmse']) for s in out['scores'])
    assert np.isnan(out['autocorr']) and out['meta']['n_polls'] == 3


def test_run_cut_applies_the_succession_of_parties_to_the_average_and_the_future_polls(monkeypatch):
    future = _poll_rows(['2023-06-26'], PP=[30.], Cs=[2.], PSOE=[20.])
    _patch_model(monkeypatch, future=future, smap={'PP': {'type': 'agg', 'names': ['Cs']}})
    out = run_cut('es', EVENT, CUT, config=default_config(), horizons=(7,))
    assert FakeForecaster.instances[0].bmap == {'PP': ['PP', 'Cs'], 'PSOE': ['PSOE']}
    pp = out['polls'].loc[out['polls']['party'] == 'PP'].iloc[0]
    assert pp['published'] == 32. and pp['residual'] == pytest.approx(2.)


def test_run_cut_with_weights_passes_the_reweighted_polls_as_overrides(monkeypatch):
    cut_polls = _poll_rows(['2023-05-01', '2023-06-01', '2023-06-20'], PP=[30.] * 3, PSOE=[20.] * 3)
    calls = _patch_model(monkeypatch, future=_poll_rows(['2023-06-26'], PP=[31.], PSOE=[20.]), cut_polls=cut_polls)
    seen = {}

    def fake_reweight(polls, wspan, ol_max=5000, ol_dev=3, drop_mtypes=None):
        seen.update(wspan=wspan, ol_max=ol_max)
        return pd.DataFrame({'proc_sample': 1000, 'weight_sample': 1.5, 'weight_over': 0.5}, index=polls.index)

    monkeypatch.setattr(tuning.Computer, 'reweight', staticmethod(fake_reweight))
    out = run_cut('es', EVENT, CUT, config=default_config() | {'weights': {'wspan': 7}})
    assert calls[0]['drange'] == (30, None) and seen == {'wspan': 7, 'ol_max': 5000}
    overrides = FakeForecaster.instances[0].kwargs['weight_overrides']
    assert overrides.index.names == ['date', 'pollster_id', 'sponsor_id']
    assert isinstance(overrides.index[0][0], pd.Timestamp)
    assert overrides.columns.tolist() == ['weight_over', 'weight_sample']  # nunca proc_sample
    assert (overrides['weight_over'] == 0.5).all() and (overrides['weight_sample'] == 1.5).all()
    assert out['meta']['n_overrides'] == 3

    run_cut('es', EVENT, CUT, config=default_config() | {'weights': {'wspan': 7, 'ol_max': 3000}})
    assert seen['ol_max'] == 3000
    assert FakeForecaster.instances[-1].kwargs['weight_overrides'].columns.tolist() == ['weight_over', 'weight_sample']


def test_run_cut_leaves_out_and_counts_the_cells_of_a_party_without_average(monkeypatch):
    class NoPsoeYet(FakeForecaster):
        def fit_forecast(self, names=None, max_fc=0, fillna=False):
            super().fit_forecast(names=names, max_fc=max_fc, fillna=fillna)
            self.forecast['PSOE'] = np.nan
            return self.forecast

    _patch_model(monkeypatch, future=_poll_rows(['2023-06-26'], PP=[31.], PSOE=[20.]))
    monkeypatch.setattr(tuning, 'Forecaster', NoPsoeYet)
    out = run_cut('es', EVENT, CUT, config=default_config(), horizons=(7,))
    assert out['polls']['party'].tolist() == ['PP'] and out['meta']['n_unpredicted'] == 1
    assert out['scores'][0]['rmse'] == pytest.approx(1.)
