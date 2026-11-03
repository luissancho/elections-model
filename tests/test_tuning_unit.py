"""Unit tests of the tuning harness of the average on future polls (no database)."""
import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from mtpy.lib import tuning
from mtpy.lib.forecaster import Forecaster
from mtpy.lib.tuning import (DEFAULTS, GRID, ORDER_SHRINKAGE, ORDER_SMOOTHING, TUNING_FILES, TUNING_PREFIX,
                             NothingToTune, TuningFailed, apply_setting, autocorrelation, coordinate_descent,
                             copy_tuning, cycle_cuts, default_config, future_polls, normalize_config, predict_polls,
                             roughness, run_cut, run_search, run_tuning, score_polls, search_cycles, tuning_fs,
                             tuning_scope)


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
    assert fc.kwargs['event_params'] is None   # the Forecaster derives the params unless they are given
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


def test_run_cut_passes_the_params_read_by_the_caller_to_the_forecaster(monkeypatch):
    _patch_model(monkeypatch, future=_poll_rows(['2023-06-26'], PP=[31.], PSOE=[20.]))
    params = FakeForecaster.params
    run_cut('es', EVENT, CUT, config=default_config(), event_params=params)
    assert FakeForecaster.instances[0].kwargs['event_params'] is params


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


def test_apply_setting_puts_each_parameter_where_it_lives():
    c = apply_setting(default_config(), 'bw', ('fixed', 45))
    assert c['reg_params'] == {'bw_type': 'fixed', 'bw': 45} and c['label'] == "bw=('fixed', 45)"
    c = apply_setting(default_config(), 'bw', ('adaptive', 2))
    assert c['reg_params']['bw_type'] == 'adaptive' and c['reg_params']['bw_kwargs']['n_iter'] == 2
    assert apply_setting(default_config(), 'wspan', 7)['weights'] == {'wspan': 7}
    assert apply_setting(default_config(), 'tau', 0.16)['he_params'] == {'tau': 0.16}
    assert apply_setting(default_config(), 'he_n_iter', 1)['he_params'] == {'n_iter': 1}
    assert apply_setting(default_config(), 'herd_prior', 10.)['disp_params'] == {'herd_prior': 10.}
    with pytest.raises(ValueError):
        apply_setting(default_config(), 'bw_cap', 0.5)
    with pytest.raises(ValueError):
        apply_setting(default_config(), 'wspan', 0)


def test_coordinate_descent_picks_the_best_allowed_value_per_key_and_records_every_trial():
    def evaluate(c):
        a, b = c.get('a', 0), c.get('b', 0)
        return {'rmse7': (a - 2) ** 2 + (b - 20) ** 2 / 100, 'rmse28': 1.0 if b != 30 else 2.0,
                'roughness': 0.02 if a != 3 else 0.05}

    def setter(c, k, v):
        return {**c, k: v}

    best, trials = coordinate_descent(evaluate, {'a': 0, 'b': 0}, {'a': [0, 1, 2, 3], 'b': [0, 20, 30]}, ['a', 'b'],
                                      rounds=2, setter=setter)
    assert best['a'] == 2 and best['b'] == 20
    assert set(trials.columns) >= {'round', 'key', 'value', 'rmse7', 'rmse28', 'roughness', 'chosen'}
    assert trials.loc[(trials.key == 'a') & (trials.value == 3), 'chosen'].eq(False).all()   # roughness 2.5x start
    assert trials.loc[(trials.key == 'b') & (trials.value == 30), 'chosen'].eq(False).all()  # k = 28 worse than 2 %


def test_grid_and_orders_are_the_fixed_search_space():
    assert GRID == {
        'bw': [('fixed', 'isj'), ('fixed', 30), ('fixed', 45), ('fixed', 60), ('fixed', 90), ('adaptive', 1),
               ('adaptive', 2), ('adaptive', 3)],
        'bw_cap': [None, 1.0, 1.5, 2.0], 'poly_deg': [0, 1], 'max_fc': [0, 3, 10], 'wspan': [2, 4, 7, 14],
        'tau': [0.04, 0.08, 0.16], 'he_n_iter': [1, 3], 'herd_prior': [2., 5., 10.]
    }
    assert ORDER_SMOOTHING == ['bw', 'bw_cap', 'poly_deg', 'max_fc', 'wspan']
    assert ORDER_SHRINKAGE == ['tau', 'he_n_iter', 'herd_prior']


def test_apply_setting_maps_the_remaining_keys_and_never_mutates_its_input():
    start = apply_setting(default_config(), 'bw', ('fixed', 45))
    snapshot = {k: (dict(v) if isinstance(v, dict) else v) for k, v in start.items()}
    c = apply_setting(start, 'bw', ('adaptive', 3))
    assert c['reg_params'] == {'bw_type': 'adaptive', 'bw_kwargs': {'n_iter': 3, 'min_delta': None}}
    assert apply_setting(default_config(), 'bw', ('fixed', 'isj'))['reg_params'] == {'bw_type': 'fixed', 'bw': 'isj'}
    c = apply_setting(start, 'bw_cap', 1.5)
    assert c['reg_params'] == {'bw_type': 'fixed', 'bw': 45, 'bw_cap': 1.5} and c['label'] == 'bw_cap=1.5'
    assert apply_setting(start, 'bw_cap', None)['reg_params']['bw_cap'] is None
    assert apply_setting(start, 'poly_deg', 0)['reg_params']['poly_deg'] == 0
    assert apply_setting(default_config(), 'max_fc', 3)['max_fc'] == 3
    assert start == snapshot and start['reg_params'] == {'bw_type': 'fixed', 'bw': 45}
    assert apply_setting(start, 'bw_cap', 1.0)['reg_params']['bw_cap'] == 1.0   # x1 of the pilot is in the grid
    with pytest.raises(ValueError):
        apply_setting(default_config(), 'tau', 0.5)
    with pytest.raises(ValueError):
        apply_setting(default_config(), 'ol_max', 3)


def test_coordinate_descent_evaluates_each_distinct_configuration_once():
    calls = []

    def evaluate(c):
        calls.append(dict(c))
        return {'rmse7': abs(c['a'] - 1) + abs(c['b'] - 1), 'rmse28': 1.0, 'roughness': 0.02}

    def setter(c, k, v):
        return {**c, k: v, 'label': f'{k}={v}'}

    best, trials = coordinate_descent(evaluate, {'a': 0, 'b': 0, 'label': 'start'}, {'a': [0, 1], 'b': [0, 1]},
                                      ['a', 'b'], rounds=2, setter=setter)
    assert (best['a'], best['b']) == (1, 1)
    distinct = {(c['a'], c['b']) for c in calls}
    assert len(calls) == len(distinct) == 4   # (0, 0), (1, 0), (1, 1), (0, 1): the label never makes a new one
    assert len(trials) == 1 + 2 * 2 * 2       # the start row plus every candidate of every key and round
    start_row = trials.loc[trials['round'] == 0]
    assert len(start_row) == 1 and start_row['chosen'].all()
    memoised = trials.loc[(trials['round'] == 2) & (trials.key == 'a') & (trials.value == 1)]
    assert memoised['rmse7'].iloc[0] == pytest.approx(0.) and memoised['chosen'].iloc[0]


def test_coordinate_descent_keeps_the_current_value_on_ties_and_when_no_candidate_passes():
    def evaluate(c):
        rmse7 = {0: 1.0, 1: 1.0, 2: 0.5}[c['a']]
        rmse28 = 5.0 if c['b'] == 1 else 1.0
        return {'rmse7': rmse7 if c['b'] == 0 else 0.1, 'rmse28': rmse28, 'roughness': 0.02}

    def setter(c, k, v):
        return {**c, k: v}

    best, trials = coordinate_descent(evaluate, {'a': 1, 'b': 0}, {'a': [0, 1], 'b': [1]}, ['a', 'b'], rounds=1,
                                      setter=setter)
    assert best == {'a': 1, 'b': 0}   # a: 0 ties with the current 1; b: the only candidate fails the k = 28 guard
    assert trials.loc[(trials.key == 'a') & (trials.value == 1), 'chosen'].iloc[0]
    assert not trials.loc[(trials.key == 'a') & (trials.value == 0), 'chosen'].iloc[0]
    assert not trials.loc[trials.key == 'b', 'chosen'].any()
    assert trials['value'].dtype == object


def test_normalize_config_fills_the_implicit_defaults_and_keeps_explicit_values_and_label():
    start = default_config()
    out = normalize_config(start)
    assert start == default_config()   # input untouched
    assert out['reg_params'] == {'bw_type': 'adaptive', 'bw_kwargs': {'n_iter': 3, 'min_delta': None},
                                 'bw_cap': None, 'poly_deg': 1}
    assert out['he_params'] == {'tau': 0.08, 'n_iter': 3} and out['disp_params'] == {'herd_prior': 5.}
    assert out['weights'] == {'wspan': 4} and out['max_fc'] == 10 and out['label'] == 'default'
    assert all(DEFAULTS[key] in GRID[key] for key in GRID) and set(DEFAULTS) == set(GRID)
    explicit = apply_setting(apply_setting(default_config(), 'bw', ('fixed', 45)), 'tau', 0.16)
    explicit['disp_params'] = {'n_prior': 7.}
    out = normalize_config(explicit)
    assert out['reg_params']['bw_type'] == 'fixed' and out['reg_params']['bw'] == 45
    assert out['he_params'] == {'tau': 0.16, 'n_iter': 3}
    assert out['disp_params'] == {'n_prior': 7.}   # herd_prior follows the explicit n_prior: left implicit
    assert out['label'] == 'tau=0.16'


def test_coordinate_descent_from_the_normalised_defaults_evaluates_the_live_model_once():
    calls = []

    def evaluate(c):
        calls.append(c)
        return {'rmse7': 1.0, 'rmse28': 1.0, 'roughness': 0.02}

    grid = {key: [DEFAULTS[key]] for key in GRID}
    order = ORDER_SMOOTHING + ORDER_SHRINKAGE
    for start, normalize in ((normalize_config(default_config()), None), (default_config(), normalize_config)):
        calls.clear()
        best, trials = coordinate_descent(evaluate, start, grid, order, rounds=1, normalize=normalize)
        assert len(calls) == 1
        assert trials['chosen'].all() and len(trials) == 1 + len(order)
        assert normalize_config(best) == {**normalize_config(default_config()), 'label': best['label']}


def test_coordinate_descent_rejects_the_best_rmse7_candidate_when_only_its_roughness_is_too_high():
    def evaluate(c):
        figures = {0: (3.0, 0.02), 1: (1.0, 0.04), 2: (2.0, 0.025)}[c['a']]   # (rmse7, roughness)
        return {'rmse7': figures[0], 'rmse28': 1.0, 'roughness': figures[1]}

    def setter(c, k, v):
        return {**c, k: v}

    best, trials = coordinate_descent(evaluate, {'a': 0}, {'a': [0, 1, 2]}, ['a'], rounds=1, setter=setter)
    assert best['a'] == 2   # a = 1 has the best rmse7 but twice the roughness of the start
    assert not trials.loc[(trials.key == 'a') & (trials.value == 1), 'chosen'].iloc[0]
    best, _ = coordinate_descent(evaluate, {'a': 0}, {'a': [0, 1, 2]}, ['a'], rounds=1, roughness_ratio=1e9,
                                 setter=setter)
    assert best['a'] == 1   # without the roughness guard it would win


def test_coordinate_descent_guards_k28_against_the_current_and_roughness_against_the_start():
    figures = {   # (a, b): (rmse7, rmse28, roughness)
        (0, 0): (3.0, 1.0, 1.0), (1, 0): (2.0, 1.015, 1.4),
        (1, 1): (1.0, 1.03, 1.2),   # k = 28 within 2 % of the current (1.015) but not of the start (1.0)
        (1, 2): (0.5, 1.0, 1.6)     # roughness within 1.5 x the current (1.4) but not the start (1.0)
    }

    def evaluate(c):
        rmse7, rmse28, rough = figures[(c['a'], c['b'])]
        return {'rmse7': rmse7, 'rmse28': rmse28, 'roughness': rough}

    def setter(c, k, v):
        return {**c, k: v}

    best, trials = coordinate_descent(evaluate, {'a': 0, 'b': 0}, {'a': [0, 1], 'b': [0, 1, 2]}, ['a', 'b'],
                                      rounds=1, setter=setter)
    # a k = 28 guard against the start would keep b = 0; a roughness guard against the current would take b = 2
    assert (best['a'], best['b']) == (1, 1)
    assert trials.loc[(trials.key == 'b') & (trials.value == 1), 'chosen'].iloc[0]


# --- Orchestration per scope (fake runner, no database) ------------------------------------------------------------

def _polls_rows(label, cut):
    """Predictions of a fake cut: one future poll a week after the cut, PP and PSOE."""
    date = pd.Timestamp(cut) + pd.Timedelta(days=7)
    return pd.DataFrame({'date': [date, date], 'pollster_id': [1, 1], 'pollster': ['P1', 'P1'],
                         'sample_size': [1000, 1000], 'party': ['PP', 'PSOE'], 'published': [31., 20.],
                         'predicted': [30., 20.], 'residual': [1., 0.], 'days_ahead': [7, 7], 'k': [7, 7]})


class FakeCut:
    """`run_cut` sin modelo: RMSE que depende de la configuración, para probar la agregación y la búsqueda."""
    def __init__(self):
        self.calls = []
        self.configs = []

    def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
        self.calls.append((event_date, cut, config['label']))
        self.configs.append((event_date, cut, config))
        base = 2.0 if config['reg_params'].get('bw') == 45 else 2.5
        return {'meta': {'n_polls': 50, 'drange': 30}, 'polls': _polls_rows(config['label'], cut),
                'scores': [{'k': k, 'n_polls': 4, 'n_cells': 12, 'rmse': base + k / 100, 'mae': base, 'skill': 0.3,
                            'rmse_by_party': {}} for k in horizons],
                'autocorr': 0.1, 'roughness': 0.4, 'seconds': 0.0}


def _patch_cycle(monkeypatch, cuts, cycles=None):
    """Patch the cuts of every cycle, the poll dates and, with `cycles`, the cycles of the scope."""
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp(cut) for cut in cuts])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2019-09-01'])))
    if cycles is not None:
        monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: cycles)


def test_run_tuning_aggregates_scores_by_partition_cycle_and_k(monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23'), pd.Timestamp('2023-06-30')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    cycles = {'search': ['2019-11-10'], 'check': ['2023-07-23']}
    out = run_tuning('es', [default_config()], cycles=cycles, runner=FakeCut())
    assert set(out['scores']['partition']) == {'search', 'check'} and set(out['scores']['k']) == {7, 28}
    assert out['cuts'].shape[0] == 4 and (out['cuts']['config'] == 'default').all()
    row = out['scores'].query("partition == 'check' and k == 7").iloc[0]
    assert row['rmse'] == pytest.approx(2.57) and row['n_cells'] == 24


def test_run_tuning_weights_the_cuts_by_their_cells_and_keeps_one_row_per_cut_and_prediction(monkeypatch):
    class Uneven(FakeCut):
        def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
            out = super().__call__(scope, event_date, cut, config, horizons=horizons)
            first = pd.Timestamp(cut) == pd.Timestamp('2023-06-23')
            error = 1. if first else 3.
            out['scores'] = [s | {'n_cells': 30 if first else 10, 'rmse': error, 'mae': error} for s in out['scores']]
            out['roughness'] = 0.2 if first else 0.6
            out['meta'] = out['meta'] | {'n_overrides': 5}
            return out

    _patch_cycle(monkeypatch, ['2023-06-23', '2023-06-30'])
    out = run_tuning('es', [default_config()], cycles={'search': [], 'check': ['2023-07-23']}, runner=Uneven())
    row = out['scores'].query('k == 7').iloc[0]
    assert row['n_cells'] == 40 and row['n_cuts'] == 2 and row['event_date'] == '2023-07-23'
    assert row['mae'] == pytest.approx((30 * 1 + 10 * 3) / 40)
    assert row['rmse'] == pytest.approx(np.sqrt((30 * 1 + 10 * 9) / 40))   # every cell weighs the same
    assert row['roughness'] == pytest.approx((0.2 + 0.6) / 2)   # a property of each cut: plain mean, as the search
    cuts = out['cuts']
    assert cuts[['rmse7', 'rmse28']].iloc[0].tolist() == [1., 1.] and cuts['n_overrides'].tolist() == [5, 5]
    assert {'config', 'partition', 'event_date', 'cut', 'n_polls', 'seconds'} <= set(cuts.columns)
    polls = out['polls']
    assert polls.shape[0] == 4 and {'config', 'event_date', 'cut', 'party', 'residual'} <= set(polls.columns)


def test_run_tuning_reads_the_params_of_each_cycle_once_for_every_cut_and_configuration(monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23', '2023-06-30'])
    reads, seen = [], []
    monkeypatch.setattr(tuning, 'get_event_params', lambda scope, event_date, path=None: reads.append(
        (scope, event_date, path)) or {'event': event_date})

    def fake_run_cut(scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0, event_params=None):
        seen.append(event_params)
        return FakeCut()(scope, event_date, cut, config, horizons=horizons)

    monkeypatch.setattr(tuning, 'run_cut', fake_run_cut)
    configs = [default_config(), apply_setting(default_config(), 'bw', ('fixed', 45))]
    run_tuning('es', configs, cycles={'search': ['2019-11-10'], 'check': ['2023-07-23']})
    assert reads == [('es', '2019-11-10', '.'), ('es', '2023-07-23', '.')]
    assert len(seen) == 8 and seen.count({'event': '2019-11-10'}) == 4


def test_run_tuning_refuses_two_configurations_with_the_same_label(monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'])
    with pytest.raises(ValueError):
        run_tuning('es', [default_config(), default_config()], cycles={'search': ['2019-11-10']}, runner=FakeCut())


def test_run_search_smoothing_prefers_the_fixed_45_day_bandwidth_with_effects_off(monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2019-10-11')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2019-09-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    runner = FakeCut()
    out = run_search('es', phase='smoothing', grid={'bw': [('fixed', 'isj'), ('fixed', 45)]}, runner=runner)
    assert out['best']['reg_params']['bw'] == 45
    # fase 1: todo apagado
    assert all(not c['house_effects'] and not c['dispersion'] and not c['herding'] for _, _, c in runner.configs)
    assert out['best']['house_effects'] is False


def test_run_search_starts_from_the_normalised_default_and_scores_only_the_search_cycles(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20', '2019-11-10'], 'check': ['2023-07-23']})
    runner = FakeCut()
    out = run_search('es', phase='smoothing', grid={'bw': [('adaptive', 3), ('fixed', 45)]}, runner=runner)
    start = runner.configs[0][2]
    assert start['weights'] == {'wspan': 4} and start['label'] == 'default'   # never the stored weights
    assert {event for event, _, _ in runner.configs} == {'2015-12-20', '2019-11-10'}
    assert set(out['scores']['partition']) == {'search'}
    # the start and ('adaptive', 3) are the same configuration: evaluated once per round, no more
    assert len(runner.calls) == 2 * 2
    assert out['trials']['evaluation'].tolist()[:3] == [0, 0, 1]


def test_run_search_weighs_every_search_cycle_the_same(monkeypatch):
    class Lopsided(FakeCut):
        def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
            out = super().__call__(scope, event_date, cut, config, horizons=horizons)
            big = event_date == '2019-11-10'
            fixed = config['reg_params'].get('bw') == 45
            # 45 days is better in the big cycle and worse in the small one, by more: equal weights reject it
            rmse = (1.0 if fixed else 1.5) if big else (3.0 if fixed else 2.0)
            out['scores'] = [s | {'n_cells': 100 if big else 10, 'rmse': rmse} for s in out['scores']]
            return out

    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20', '2019-11-10'], 'check': []})
    out = run_search('es', phase='smoothing', grid={'bw': [('fixed', 45)]}, runner=Lopsided())
    assert out['best']['reg_params'].get('bw_type') == 'adaptive'
    trial = out['trials'].query("key == 'bw'").iloc[0]
    assert trial['rmse7'] == pytest.approx((1.0 + 3.0) / 2)


def test_run_search_never_runs_the_bandwidth_cap_under_a_fixed_bandwidth(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2019-11-10'], 'check': []})
    runner = FakeCut()
    start = apply_setting(default_config(), 'bw', ('fixed', 45))
    out = run_search('es', phase='smoothing', start=start, grid={'bw_cap': [None, 1.0, 1.5]}, rounds=1,
                     runner=runner)
    assert len(runner.calls) == 1   # the caps are no-ops under a fixed bandwidth
    caps = out['trials'].query("key == 'bw_cap'")
    assert caps['chosen'].tolist() == [True, False, False] and caps['evaluation'].tolist() == [0, 0, 0]


def test_run_search_check_evaluates_start_and_default_with_everything_on(monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    runner = FakeCut()
    start = apply_setting(default_config(), 'bw', ('fixed', 45))
    out = run_search('es', phase='check', start=start, runner=runner)
    assert set(out['scores']['partition']) == {'check'}
    assert set(out['scores']['config']) == {'default', "bw=('fixed', 45)"}
    assert all(c['house_effects'] and c['dispersion'] and c['herding'] for _, _, c in runner.configs)


def test_run_search_check_compares_with_the_normalised_default_not_the_stored_weights(monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'], {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    runner = FakeCut()
    start = apply_setting(default_config(), 'wspan', 7) | {'house_effects': False}
    out = run_search('es', phase='check', start=start, runner=runner)
    default = next(c for _, _, c in runner.configs if c['label'] == 'default')
    assert default == normalize_config(default_config()) and default['weights'] == {'wspan': 4}
    assert out['best']['house_effects'] is True and out['trials'].empty


def test_search_cycles_adds_the_running_cycle_and_splits_at_2020(monkeypatch):
    calls = {}

    def fake_dates(scope, date_from=None, date_to=None, skip=0, min_polls=0, featured=False):
        calls.update(date_from=date_from, min_polls=min_polls, featured=featured)
        return ['2008-03-09', '2011-11-20', '2019-11-10', '2023-07-23']

    asked = []

    def fake_next(scope, date_from=None):
        asked.append(date_from)
        return '2026-11-29' if date_from < '2026-11-29' else None

    monkeypatch.setattr(tuning, 'get_event_dates', fake_dates)
    monkeypatch.setattr(tuning, 'get_next_event_date', fake_next)
    cycles = search_cycles('es', today='2026-11-02')
    assert calls == {'date_from': '2008-01-01', 'min_polls': 10, 'featured': True}
    assert cycles == {'search': ['2008-03-09', '2011-11-20', '2019-11-10'], 'check': ['2023-07-23', '2026-11-29']}
    # asked from today, not from the last featured election: a held election that is not featured cannot hide it
    assert asked == ['2026-11-02']
    assert search_cycles('es', today='2026-12-01')['check'] == ['2023-07-23']


def test_tuning_scope_writes_the_tables_and_run_json(fresh_app, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    figures = tuning_scope('es-md', out=str(tmp_path), mode='evaluate', configs=[default_config()], runner=FakeCut(),
                           stats=lambda scope: {'db_polls': 7, 'db_last_poll': '2026-10-03'},
                           prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    assert figures['n_configs'] == 1 and figures['n_cuts'] == 2 and figures['folder'] == str(tmp_path / 'es-md')
    for name in ('cuts.csv', 'polls.csv', 'scores.csv', 'run.json'):
        assert (tmp_path / 'es-md' / name).exists()
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['scope'] == 'es-md' and run['mode'] == 'evaluate' and run['commit'] == 'abc1234'
    assert run['configs'] == ['default']


def test_tuning_scope_run_json_records_the_run(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'], {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    tuning_scope('es', out=str(tmp_path), mode='evaluate', configs=[default_config()], runner=FakeCut(),
                 stats=lambda scope: {'db_polls': 7, 'db_last_poll': '2026-10-03'},
                 prov=lambda: {'commit': 'abc1234', 'dirty': True, 'versions': {}})
    assert TUNING_FILES == ('cuts.csv', 'polls.csv', 'scores.csv', 'trials.csv', 'run.json')
    assert all((tmp_path / name).exists() for name in TUNING_FILES)   # es at the root
    run = json.load(open(tmp_path / 'run.json', encoding='utf-8'))
    assert set(run) >= {'scope', 'mode', 'configs', 'cycles', 'cut_days', 'horizons', 'rounds', 'commit', 'dirty',
                        'run_at', 'db_polls', 'db_last_poll', 'seconds'}
    assert run['cycles'] == {'search': ['2019-11-10'], 'check': ['2023-07-23']} and run['dirty'] is True
    assert run['cut_days'] == 7 and run['horizons'] == [7, 28] and run['db_polls'] == 7


def test_tuning_scope_smoothing_saves_the_best_with_everything_on_and_its_trials(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    figures = tuning_scope('es-md', out=str(tmp_path), mode='smoothing', runner=FakeCut(),
                           grid={'bw': [('fixed', 'isj'), ('fixed', 45)]}, stats=lambda scope: {},
                           prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    best = json.load(open(tmp_path / 'es-md' / 'best.json', encoding='utf-8'))
    assert best['reg_params']['bw'] == 45 and figures['best'] == best
    assert best['house_effects'] is True and best['dispersion'] is True and best['herding'] is True
    trials = pd.read_csv(tmp_path / 'es-md' / 'trials-smoothing.csv')
    assert trials.shape[0] == 1 + 2 * 2 and (tmp_path / 'es-md' / 'trials.csv').exists()
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['mode'] == 'smoothing' and run['rounds'] == 2 and run['cut_days'] == 7
    assert run['cycles'] == {'search': ['2019-11-10']}


def test_tuning_scope_shrinkage_reads_the_best_of_the_smoothing_phase(fresh_app, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'cycle_cuts', lambda *a, **k: [pd.Timestamp('2023-06-23')])
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(pd.to_datetime(['2023-06-01'])))
    monkeypatch.setattr(tuning, 'search_cycles',
                        lambda scope, **k: {'search': ['2019-11-10', '2023-07-23'], 'check': []})
    (tmp_path / 'es-md').mkdir()
    json.dump(apply_setting(default_config(), 'bw', ('fixed', 45)), open(tmp_path / 'es-md' / 'best.json', 'w'))
    runner = FakeCut()
    tuning_scope('es-md', out=str(tmp_path), mode='shrinkage', runner=runner, grid={'herd_prior': [2., 5.]},
                 stats=lambda scope: {}, prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    assert all(c['reg_params']['bw'] == 45 and c['house_effects'] for _, _, c in runner.configs)  # del best, todo on
    assert all(cut.year >= 2015 for _, cut, _ in runner.configs)                                    # ciclos desde 2015


def test_run_search_shrinkage_keeps_the_search_cycles_since_2015_with_cuts_every_14_days(monkeypatch):
    seen = {}

    def fake_cuts(event_date, poll_dates, cut_days=7, **kwargs):
        seen.setdefault('cut_days', set()).add(cut_days)
        return [pd.Timestamp(event_date) - pd.Timedelta(days=30)]

    monkeypatch.setattr(tuning, 'cycle_cuts', fake_cuts)
    monkeypatch.setattr(tuning, 'poll_dates', lambda scope, event_date: pd.Series(dtype='datetime64[ns]'))
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': ['2011-11-20', '2015-12-20'],
                                                                     'check': ['2023-07-23']})
    runner = FakeCut()
    out = run_search('es', phase='shrinkage', start=default_config(), grid={'herd_prior': [2., 5.]}, runner=runner)
    assert {event for event, _, _ in runner.configs} == {'2015-12-20'} and seen['cut_days'] == {14}
    assert out['trials']['round'].max() == 1


def test_tuning_scope_shrinkage_without_the_best_of_the_smoothing_phase_refuses(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'], {'search': ['2019-11-10'], 'check': []})
    with pytest.raises(FileNotFoundError):
        tuning_scope('es-md', out=str(tmp_path), mode='shrinkage', runner=FakeCut(), stats=lambda scope: {},
                     prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})


def test_tuning_scope_without_cycles_raises_nothing_to_tune(fresh_app, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'search_cycles', lambda scope, **k: {'search': [], 'check': []})
    with pytest.raises(NothingToTune):
        tuning_scope('es-xx', out=str(tmp_path), mode='evaluate', configs=[default_config()], runner=FakeCut())
    assert not any(tmp_path.iterdir())


def test_tuning_fs_mirrors_the_backtest_layout_under_its_own_prefix(fresh_app, tmp_path):
    fs, folder = tuning_fs(str(tmp_path), 'es')
    assert folder == '' and tuning_fs(str(tmp_path), 'es-md')[1] == 'es-md'
    if fresh_app.fs is not None:
        assert tuning_fs(None, 'es')[1] == TUNING_PREFIX == 'backtest/tuning'
        assert tuning_fs(None, 'es-md')[1] == 'backtest/tuning/es-md'


class FailingCut(FakeCut):
    """`FakeCut` that raises for some configurations and cycles (a fixed small bandwidth on a sparse cycle)."""
    def __init__(self, fails, error=RuntimeError):
        super().__init__()
        self.fails, self.error = fails, error

    def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
        self.calls.append((event_date, cut, config['label']))
        if self.fails(event_date, config):
            raise self.error('no polls in the window')
        return super().__call__(scope, event_date, cut, config, horizons=horizons)


def test_run_search_never_keeps_a_configuration_with_a_failed_cut(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20', '2019-11-10'], 'check': []})
    # 45 days would win on the cycle it survives; it fails on the other one
    runner = FailingCut(lambda event_date, config: config['reg_params'].get('bw') == 45 and event_date == '2015-12-20')
    with pytest.warns(UserWarning):
        out = run_search('es', phase='smoothing', grid={'bw': [('fixed', 45)]}, rounds=1, runner=runner)
    assert out['best']['reg_params'].get('bw_type') == 'adaptive'
    trial = out['trials'].query("key == 'bw'").iloc[0]
    assert not trial['chosen'] and trial[['rmse7', 'rmse28', 'roughness']].astype(float).isna().all()
    assert out['cuts']['error'].notna().sum() == 1


def test_run_search_stops_when_its_start_has_a_failed_cut(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20', '2019-11-10'], 'check': []})
    runner = FailingCut(lambda event_date, config: config['label'] == 'default' and event_date == '2019-11-10')
    with pytest.warns(UserWarning), pytest.raises(TuningFailed):
        run_search('es', phase='smoothing', grid={'bw': [('fixed', 45)]}, runner=runner)


def test_run_search_check_stops_when_a_configuration_has_a_failed_cut(monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'], {'search': [], 'check': ['2023-07-23']})
    runner = FailingCut(lambda event_date, config: config['label'] == 'default')
    with pytest.warns(UserWarning), pytest.raises(TuningFailed):
        run_search('es', phase='check', start=apply_setting(default_config(), 'bw', ('fixed', 45)), runner=runner)


def test_tuning_scope_raises_tuning_failed_when_every_cut_fails(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'], {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    with pytest.warns(UserWarning), pytest.raises(TuningFailed):
        tuning_scope('es-md', out=str(tmp_path), mode='evaluate', configs=[default_config()],
                     runner=FailingCut(lambda event_date, config: True), stats=lambda scope: {},
                     prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    assert not any(tmp_path.iterdir())


def test_tuning_scope_without_cuts_still_raises_nothing_to_tune(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, [], {'search': ['2019-11-10'], 'check': []})
    with pytest.raises(NothingToTune):
        tuning_scope('es-md', out=str(tmp_path), mode='evaluate', configs=[default_config()], runner=FakeCut())


def test_run_tuning_retries_a_cut_after_a_connection_error_only(monkeypatch):
    from psycopg2 import OperationalError, ProgrammingError

    _patch_cycle(monkeypatch, ['2023-06-23'])
    monkeypatch.setattr(tuning, 'RETRY_SECONDS', 0)

    class Flaky(FakeCut):
        def __init__(self, failures, error):
            super().__init__()
            self.failures, self.error = failures, error

        def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
            if self.failures > 0:
                self.failures -= 1
                self.calls.append(('failed', cut, config['label']))
                raise self.error('server closed the connection unexpectedly')
            return super().__call__(scope, event_date, cut, config, horizons=horizons)

    cycles = {'search': [], 'check': ['2023-07-23']}
    runner = Flaky(1, OperationalError)
    with pytest.warns(UserWarning, match='retry 1 of 3'):
        out = run_tuning('es', [default_config()], cycles=cycles, runner=runner)
    assert len(runner.calls) == 2 and out['cuts']['error'].isna().all()

    runner = Flaky(10, OperationalError)
    with pytest.warns(UserWarning):
        out = run_tuning('es', [default_config()], cycles=cycles, runner=runner)
    assert len(runner.calls) == 1 + tuning.CUT_RETRIES and out['cuts']['error'].notna().all()

    runner = Flaky(1, ProgrammingError)   # not a connection error: no retry
    with pytest.warns(UserWarning, match='skipped'):
        out = run_tuning('es', [default_config()], cycles=cycles, runner=runner)
    assert len(runner.calls) == 1 and out['cuts']['error'].notna().all()


def test_run_search_labels_an_unlabelled_start_so_its_tables_are_kept(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2019-11-10'], 'check': []})
    start = apply_setting(default_config(), 'bw', ('fixed', 45))
    del start['label']
    out = run_search('es', phase='smoothing', start=start, grid={'poly_deg': [1]}, rounds=1, runner=FakeCut())
    assert out['configs'] == ['start'] and out['trials']['label'].iloc[0] == 'start'
    assert set(out['cuts']['config']) == {'start'} and not out['scores'].empty


def test_run_search_refuses_a_grid_without_the_keys_of_its_phase(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2019-11-10'], 'check': []})
    with pytest.raises(ValueError):
        run_search('es', phase='shrinkage', grid={'bw': [('fixed', 45)]}, runner=FakeCut())


# --- Pool of processes, search tables per phase, parts and merge, push and pull ---------------------------------------

def pool_fake_cut(scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
    """`run_cut` without model nor database, importable by a process of the pool: the first cut takes longer, so the
    cuts end in another order than they were sent; the cut of 2019-10-25 warns and the one of 2019-11-01 fails."""
    cut = pd.Timestamp(cut)
    if cut == pd.Timestamp('2019-10-11'):
        time.sleep(0.5)
    if cut == pd.Timestamp('2019-11-01'):
        raise ValueError('no future polls')
    if cut == pd.Timestamp('2019-10-25'):
        warnings.warn('cut {:%Y-%m-%d} warned'.format(cut))
    out = FakeCut()(scope, event_date, cut, config, horizons=horizons)
    shift = (cut - pd.Timestamp('2019-10-01')).days / 100
    out['scores'] = [score | {'rmse': score['rmse'] + shift} for score in out['scores']]
    out['seconds'] = 1.5

    return out


def test_run_tuning_with_a_pool_returns_the_tables_and_lines_of_the_sequential_run(monkeypatch, capsys):
    _patch_cycle(monkeypatch, ['2019-10-11', '2019-10-18', '2019-10-25', '2019-11-01'])
    cycles = {'search': ['2015-12-20', '2019-11-10'], 'check': []}
    configs = [default_config(), apply_setting(default_config(), 'bw', ('fixed', 45))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sequential = run_tuning('es', configs, cycles=cycles, runner=pool_fake_cut, verbose=1)
    sequential_out = capsys.readouterr().out
    sequential_warnings = sorted(str(w.message) for w in caught)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pooled = run_tuning('es', configs, cycles=cycles, runner=pool_fake_cut, verbose=1, workers=2,
                            initializer=None)
    assert capsys.readouterr().out == sequential_out and sequential_out.count('\n') == 2 * 2 * 3
    assert sorted(str(w.message) for w in caught) == sequential_warnings
    assert any('warned' in message for message in sequential_warnings)
    for name in ('cuts', 'polls', 'scores'):
        pd.testing.assert_frame_equal(pooled[name], sequential[name])
    cuts = pooled['cuts']
    assert cuts.shape[0] == 2 * 2 * 4 and cuts['error'].notna().sum() == 2 * 2
    assert cuts['cut'].tolist() == [pd.Timestamp(c) for c in ['2019-10-11', '2019-10-18', '2019-10-25',
                                                              '2019-11-01']] * 4


def test_run_tuning_refuses_fewer_than_one_worker(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'])
    with pytest.raises(ValueError, match='workers'):
        run_tuning('es', [default_config()], cycles={'search': ['2019-11-10']}, runner=FakeCut(), workers=0)


def test_tuning_scope_keeps_the_scores_of_each_search_phase_beside_its_trials(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2019-11-10'], 'check': ['2023-07-23']})
    prov = lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}   # noqa: E731
    tuning_scope('es-md', out=str(tmp_path), mode='smoothing', runner=FakeCut(), grid={'bw': [('fixed', 45)]},
                 stats=lambda scope: {}, prov=prov)
    folder = tmp_path / 'es-md'
    smoothing = pd.read_csv(folder / 'scores-smoothing.csv')
    pd.testing.assert_frame_equal(smoothing, pd.read_csv(folder / 'scores.csv'))
    tuning_scope('es-md', out=str(tmp_path), mode='check', runner=FakeCut(), stats=lambda scope: {}, prov=prov)
    pd.testing.assert_frame_equal(pd.read_csv(folder / 'scores-smoothing.csv'), smoothing)   # survives the check
    assert set(pd.read_csv(folder / 'scores-check.csv')['partition']) == {'check'}


class ShrinkCut(FakeCut):
    """`FakeCut` whose RMSE depends on the cycle, the herding prior, tau and the backfitting iterations: each cycle
    prefers other values (2015 tau 0.04, 2019 tau 0.08, as the real cycles diverge after the first key)."""
    BASE = {'2015-12-20': {2.: 1.0, 5.: 2.0, 10.: 3.0}, '2019-11-10': {2.: 3.0, 5.: 2.5, 10.: 1.5}}

    def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
        out = super().__call__(scope, event_date, cut, config, horizons=horizons)
        base = self.BASE[event_date][config['disp_params']['herd_prior']]
        if config['he_params']['tau'] == 0.04:
            base += -0.5 if event_date == '2015-12-20' else 0.6
        if config['he_params']['n_iter'] == 1:
            base += -0.1 if event_date == '2015-12-20' else 0.05
        out['scores'] = [s | {'rmse': base + s['k'] / 100} for s in out['scores']]
        return out


def _shrinkage_parts(tmp_path, monkeypatch, grid):
    """Smoothing best in `es-md`, one shrinkage part per cycle in `parts/` and the whole run in `whole/`."""
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20', '2019-11-10'], 'check': []})
    prov = lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}   # noqa: E731
    best = {**apply_setting(default_config(), 'bw', ('fixed', 45)), 'house_effects': True, 'dispersion': True,
            'herding': True}
    for name in ('parts', 'whole'):
        (tmp_path / name / 'es-md').mkdir(parents=True)
        json.dump(best, open(tmp_path / name / 'es-md' / 'best.json', 'w'))
    for date in ('2015-12-20', '2019-11-10'):
        tuning_scope('es-md', out=str(tmp_path / 'parts'), mode='shrinkage', runner=ShrinkCut(), grid=grid,
                     cycles={'search': [date], 'check': []}, part=date, stats=lambda scope: {}, prov=prov)
    tuning_scope('es-md', out=str(tmp_path / 'whole'), mode='shrinkage', runner=ShrinkCut(), grid=grid,
                 stats=lambda scope: {}, prov=prov)
    return best, prov


def test_tuning_scope_shrinkage_part_writes_its_own_tables_and_leaves_the_best_alone(fresh_app, tmp_path,
                                                                                   monkeypatch):
    best, _ = _shrinkage_parts(tmp_path, monkeypatch, {'herd_prior': [2., 5., 10.]})
    folder = tmp_path / 'parts' / 'es-md'
    names = sorted(path.name for path in folder.iterdir())
    assert names == ['best.json'] + sorted('{}-shrinkage-{}.{}'.format(table, date, ext)
                                           for date in ('2015-12-20', '2019-11-10')
                                           for table, ext in (('cuts', 'csv'), ('run', 'json'), ('scores', 'csv'),
                                                              ('trials', 'csv')))
    assert json.load(open(folder / 'best.json')) == best   # a part never decides
    trials = pd.read_csv(folder / 'trials-shrinkage-2019-11-10.csv')
    assert 'config_key' in trials.columns and trials['config_key'].notna().all()
    run = json.load(open(folder / 'run-shrinkage-2019-11-10.json'))
    assert run['mode'] == 'shrinkage' and run['part'] == '2019-11-10' and run['cycles'] == {'search': ['2019-11-10']}


def test_tuning_scope_merge_repeats_the_descent_on_the_joined_parts_as_the_whole_run(fresh_app, tmp_path,
                                                                                   monkeypatch):
    _, prov = _shrinkage_parts(tmp_path, monkeypatch, {'herd_prior': [2., 5., 10.]})
    figures = tuning_scope('es-md', out=str(tmp_path / 'parts'), mode='merge', grid={'herd_prior': [2., 5., 10.]},
                           stats=lambda scope: {}, prov=prov)
    parts, whole = tmp_path / 'parts' / 'es-md', tmp_path / 'whole' / 'es-md'
    merged = json.load(open(parts / 'best.json'))
    assert merged == json.load(open(whole / 'best.json')) and merged['disp_params']['herd_prior'] == 2.
    assert figures['best'] == merged and figures['n_cuts'] == 2 and figures['n_configs'] == 3
    columns = ['round', 'key', 'value', 'label', 'rmse7', 'rmse28', 'roughness', 'chosen', 'config_key']
    pd.testing.assert_frame_equal(pd.read_csv(parts / 'trials-shrinkage.csv')[columns],
                                  pd.read_csv(whole / 'trials-shrinkage.csv')[columns])
    scores = pd.read_csv(parts / 'scores-shrinkage.csv')
    assert set(scores['part']) == {'2015-12-20', '2019-11-10'}
    run = json.load(open(parts / 'run.json'))
    assert run['mode'] == 'merge' and run['parts'] == ['2015-12-20', '2019-11-10']
    assert run['cycles'] == {'search': ['2015-12-20', '2019-11-10']}


def test_tuning_scope_merge_of_product_parts_is_the_whole_run_with_the_whole_shrinkage_grid(fresh_app, tmp_path,
                                                                                        monkeypatch):
    # each cycle on its own would descend elsewhere (2015 keeps tau 0.04, 2019 tau 0.08); the joined descent goes
    # on to he_n_iter and herd_prior from tau 0.08, configurations a part only has because it ran the whole product
    _, prov = _shrinkage_parts(tmp_path, monkeypatch, None)
    figures = tuning_scope('es-md', out=str(tmp_path / 'parts'), mode='merge', stats=lambda scope: {}, prov=prov)
    parts, whole = tmp_path / 'parts' / 'es-md', tmp_path / 'whole' / 'es-md'
    merged = json.load(open(parts / 'best.json'))
    assert merged == json.load(open(whole / 'best.json')) == figures['best']
    assert merged['label'] == "bw=('fixed', 45); he_n_iter=1; herd_prior=2.0"
    assert merged['he_params'] == {'tau': 0.08, 'n_iter': 1} and merged['disp_params']['herd_prior'] == 2.
    columns = ['round', 'key', 'value', 'label', 'rmse7', 'rmse28', 'roughness', 'chosen', 'config_key']
    pd.testing.assert_frame_equal(pd.read_csv(parts / 'trials-shrinkage.csv')[columns],
                                  pd.read_csv(whole / 'trials-shrinkage.csv')[columns])
    for name in ('2015-12-20', '2019-11-10'):
        trials = pd.read_csv(parts / 'trials-shrinkage-{}.csv'.format(name))
        assert len(trials) == 3 * 2 * 3 and set(trials['key']) == {'product'}
        assert trials['chosen'].tolist() == [True] + [False] * 17   # the start, first; nothing is chosen in a part
        run = json.load(open(parts / 'run-shrinkage-{}.json'.format(name)))
        assert run['search'] == 'product' and run['rounds'] is None and run['start']['reg_params']['bw'] == 45
    assert json.load(open(whole / 'run.json'))['search'] == 'descent'


def test_tuning_scope_merge_never_keeps_a_configuration_missing_from_a_part(fresh_app, tmp_path, monkeypatch):
    _, prov = _shrinkage_parts(tmp_path, monkeypatch, {'tau': [0.04, 0.08]})   # parts without herd_prior
    with pytest.warns(UserWarning, match='missing from some part'):
        tuning_scope('es-md', out=str(tmp_path / 'parts'), mode='merge', grid={'tau': [0.04, 0.08],
                                                                               'herd_prior': [2., 10.]},
                     stats=lambda scope: {}, prov=prov)
    trials = pd.read_csv(tmp_path / 'parts' / 'es-md' / 'trials-shrinkage.csv')
    herd = trials.loc[trials['key'] == 'herd_prior']
    assert herd['rmse7'].isna().all() and not herd['chosen'].any()


def test_tuning_scope_merge_writes_its_own_tables_and_a_second_merge_repeats_the_first(fresh_app, tmp_path,
                                                                                      monkeypatch):
    _, prov = _shrinkage_parts(tmp_path, monkeypatch, None)
    folder = tmp_path / 'parts' / 'es-md'
    (folder / 'polls.csv').write_text('stale')
    tuning_scope('es-md', out=str(tmp_path / 'parts'), mode='merge', stats=lambda scope: {}, prov=prov)
    first = json.load(open(folder / 'best.json'))
    pd.testing.assert_frame_equal(pd.read_csv(folder / 'trials.csv'), pd.read_csv(folder / 'trials-shrinkage.csv'))
    scores = pd.read_csv(folder / 'scores.csv')
    assert set(scores['part']) == {'2015-12-20', '2019-11-10'} and scores['config'].nunique() == 18
    cuts = pd.read_csv(folder / 'cuts.csv')
    assert len(cuts) == 2 * 18 and set(cuts['part']) == {'2015-12-20', '2019-11-10'}
    assert not (folder / 'polls.csv').exists()
    tuning_scope('es-md', out=str(tmp_path / 'parts'), mode='merge', stats=lambda scope: {}, prov=prov)
    assert json.load(open(folder / 'best.json')) == first   # from the start the parts recorded, not this best.json


def test_run_search_refuses_a_product_in_the_check(monkeypatch):
    _patch_cycle(monkeypatch, ['2023-06-23'], {'search': [], 'check': ['2023-07-23']})
    with pytest.raises(ValueError, match='product'):
        run_search('es', phase='check', product=True, runner=FakeCut())


def test_run_tuning_stops_the_pool_when_recording_a_cut_raises(monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11', '2019-10-18'])
    closed, alive = [], []

    def generator(tasks):
        try:
            for _ in tasks:
                yield {'result': {}, 'error': None, 'warnings': []}   # no 'meta': recording it raises
        finally:
            closed.append(True)

    def outcomes(tasks, workers, initializer):
        alive.append(generator(tasks))   # a reference held elsewhere: only an explicit close ends it
        return alive[-1]

    monkeypatch.setattr(tuning, 'pooled_outcomes', outcomes)
    with pytest.raises(KeyError):
        run_tuning('es', [default_config()], cycles={'search': ['2019-11-10']}, runner=FakeCut(), workers=2)
    assert closed == [True]


def test_tuning_scope_merge_needs_parts_and_the_best_of_the_smoothing_phase(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20'], 'check': []})
    with pytest.raises(FileNotFoundError):
        tuning_scope('es-md', out=str(tmp_path), mode='merge', stats=lambda scope: {})
    (tmp_path / 'es-md').mkdir()
    json.dump(default_config(), open(tmp_path / 'es-md' / 'best.json', 'w'))
    with pytest.raises(NothingToTune, match='no shrinkage part'):
        tuning_scope('es-md', out=str(tmp_path), mode='merge', stats=lambda scope: {})


def test_tuning_scope_refuses_a_part_outside_the_shrinkage_mode(fresh_app, tmp_path, monkeypatch):
    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2015-12-20'], 'check': []})
    with pytest.raises(ValueError, match='part'):
        tuning_scope('es-md', out=str(tmp_path), mode='smoothing', part='2015-12-20', runner=FakeCut())


def test_copy_tuning_mirrors_the_tuning_files_of_the_scope_only(tmp_path):
    src = tmp_path / 'src'
    (src / 'es-md').mkdir(parents=True)
    names = ['cuts.csv', 'polls.csv', 'scores.csv', 'trials.csv', 'run.json', 'best.json', 'trials-smoothing.csv',
             'scores-smoothing.csv', 'trials-shrinkage-2019-11-10.csv', 'run-shrinkage-2019-11-10.json']
    for name in names + ['notes.txt']:
        (src / name).write_text(name)
    (src / 'es-md' / 'cuts.csv').write_text('regional')
    dst = tmp_path / 'dst'
    dst.mkdir()
    (dst / 'trials-check.csv').write_text('stale')
    (dst / 'keep.txt').write_text('not a tuning file')
    done = copy_tuning('es', tuning_fs(str(src), 'es'), tuning_fs(str(dst), 'es'))
    assert sorted(done['copied']) == sorted(names) and done['removed'] == ['trials-check.csv']
    assert sorted(path.name for path in dst.iterdir()) == sorted(names + ['keep.txt'])
    assert (dst / 'best.json').read_text() == 'best.json'
    with pytest.raises(NothingToTune):
        copy_tuning('es-xx', tuning_fs(str(src), 'es-xx'), tuning_fs(str(dst), 'es-xx'))


# --- Watchdog of a cut in a process of the pool ----------------------------------------------------------------------

class SleepyCut(FakeCut):
    """`FakeCut` that hangs (sleeps 2 s) on its first `hangs` calls, as a cut on a dead database link."""
    def __init__(self, hangs):
        super().__init__()
        self.hangs = hangs

    def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
        if len(self.calls) < self.hangs:
            self.calls.append((event_date, cut, config['label']))
            time.sleep(2)
        return super().__call__(scope, event_date, cut, config, horizons=horizons)


def _watchdog(monkeypatch):
    """A watchdog of 0.2 s, no wait between retries and the resets of the connection recorded."""
    resets = []
    monkeypatch.setattr(tuning, 'CUT_TIMEOUT', 0.2)
    monkeypatch.setattr(tuning, 'RETRY_SECONDS', 0.)
    monkeypatch.setattr(tuning, 'reset_connection', lambda: resets.append(True))
    return resets


def test_pool_cut_times_out_a_hung_cut_resets_the_connection_and_retries(monkeypatch):
    resets = _watchdog(monkeypatch)
    runner = SleepyCut(hangs=1)
    began = time.perf_counter()
    outcome = tuning.pool_cut((runner, 'es', '2019-11-10', pd.Timestamp('2019-10-11'), default_config(), (7, 28),
                               None, 0, None))
    assert time.perf_counter() - began < 1.5
    assert outcome['error'] is None and outcome['result']['scores'][0]['k'] == 7
    assert len(resets) == 1 and any('retry 1 of' in message for message, _ in outcome['warnings'])
    assert signal_is_quiet()


def test_pool_cut_records_a_cut_that_hangs_on_every_attempt_as_an_error(monkeypatch):
    resets = _watchdog(monkeypatch)
    outcome = tuning.pool_cut((SleepyCut(hangs=99), 'es', '2019-11-10', pd.Timestamp('2019-10-11'),
                               default_config(), (7, 28), None, 0, None))
    assert outcome['result'] is None and 'did not finish in 0.2 s' in outcome['error']
    assert len(resets) == 1 + tuning.CUT_RETRIES and signal_is_quiet()


def signal_is_quiet():
    """No timer left armed and the default handler of SIGALRM back."""
    import signal
    return signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0) and signal.getsignal(signal.SIGALRM) == signal.SIG_DFL


def test_reset_connection_forgets_the_engine_without_touching_its_sockets(fresh_app):
    seen = []

    class Engine:
        def dispose(self, close=True):
            seen.append(('engine', close))

    class Adapter:
        _client = Engine()

        def dispose(self):
            seen.append(('adapter',))

    fresh_app.set('db', Adapter())
    tuning.reset_connection()
    assert seen == [('engine', False), ('adapter',)]


def test_boot_worker_boots_the_app_and_makes_the_queries_interruptible(monkeypatch):
    import psycopg2.extensions
    import psycopg2.extras
    from mtpy import mtpy as app_module
    booted, callbacks = [], []
    monkeypatch.setattr(app_module, 'run', lambda: booted.append(True))
    monkeypatch.setattr(psycopg2.extensions, 'set_wait_callback', callbacks.append)
    tuning.boot_worker()
    assert booted == [True] and callbacks == [psycopg2.extras.wait_select]


# --- Label of the configuration kept ---------------------------------------------------------------------------------

def test_best_label_lists_every_setting_that_differs_from_the_defaults():
    config = apply_setting(apply_setting(default_config(), 'max_fc', 0), 'bw_cap', 1.0)
    assert tuning.best_label(config) == 'bw_cap=1.0; max_fc=0'
    assert tuning.best_label(normalize_config(default_config())) == 'default'
    fixed = apply_setting(apply_setting(default_config(), 'bw_cap', 1.5), 'bw', ('fixed', 45))
    assert tuning.best_label(fixed) == "bw=('fixed', 45)"   # the cap is a no-op under a fixed bandwidth
    shrunk = apply_setting(apply_setting(default_config(), 'tau', 0.04), 'he_n_iter', 1)
    assert tuning.best_label(shrunk) == 'tau=0.04; he_n_iter=1'


def test_tuning_scope_saves_the_best_with_the_label_of_all_its_settings(fresh_app, tmp_path, monkeypatch):
    class Prolonged(FakeCut):
        def __call__(self, scope, event_date, cut, config, horizons=(7, 28), max_fc=None, verbose=0):
            out = super().__call__(scope, event_date, cut, config, horizons=horizons)
            if config['max_fc'] == 0:
                out['scores'] = [s | {'rmse': s['rmse'] - 0.2} for s in out['scores']]
            return out

    _patch_cycle(monkeypatch, ['2019-10-11'], {'search': ['2019-11-10'], 'check': []})
    figures = tuning_scope('es-md', out=str(tmp_path), mode='smoothing', runner=Prolonged(), rounds=1,
                           grid={'bw': [('fixed', 45)], 'max_fc': [0, 10]}, stats=lambda scope: {},
                           prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    best = json.load(open(tmp_path / 'es-md' / 'best.json', encoding='utf-8'))
    assert best['label'] == "bw=('fixed', 45); max_fc=0" and figures['best']['label'] == best['label']


# --- Final fix wave: product from a hand-written start, late watchdog, unusable start ------------------------------

def _hand_written_start():
    """A start written by hand: `herd_prior` as an int and `bw_kwargs` without `min_delta` (same model, other form)."""
    hand = {**default_config(), 'reg_params': {'bw_type': 'adaptive', 'bw_kwargs': {'n_iter': 3}},
            'disp_params': {'herd_prior': 5}, 'label': 'start'}
    return normalize_config(hand) | {'house_effects': True, 'dispersion': True, 'herding': True}


def test_product_configs_writes_every_grid_value_as_the_descent_does_from_a_hand_written_start():
    start = _hand_written_start()
    order = ['bw', 'herd_prior']
    products = {tuning.config_key(tuning.effective_config(config))
                for config in tuning.product_configs(start, GRID, order)}
    visited = []

    def evaluate(config):
        visited.append(tuning.config_key(tuning.effective_config(config)))
        return {'rmse7': 1., 'rmse28': 1., 'roughness': 1.}

    coordinate_descent(evaluate, start, GRID, order, rounds=2, normalize=normalize_config)
    assert set(visited) <= products


def test_merge_parts_finds_every_candidate_of_a_product_part_from_a_hand_written_start():
    start = _hand_written_start()
    configs = tuning.product_configs(start, GRID, ORDER_SHRINKAGE)
    keys = [tuning.config_key(tuning.effective_config(config)) for config in configs]
    trials = pd.DataFrame({'evaluation': range(len(keys)), 'config_key': keys, 'rmse7': 1., 'rmse28': 1.,
                           'roughness': 1.})
    scores = pd.DataFrame([{'evaluation': number, 'event_date': '2019-11-10', 'k': k, 'rmse': 1. + k / 100,
                            'roughness': 1.} for number in range(len(keys)) for k in (7, 28)])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = tuning.merge_parts({'2019-11-10': (trials, scores)}, start)
    assert out['missing'] == [] and out['n_configs'] == 7   # start, tau 2, he_n_iter 1, herd_prior 3 (5.0 too)


def test_timed_call_returns_the_result_when_the_watchdog_fires_after_the_cut_finished(monkeypatch):
    import signal
    real = signal.setitimer

    def late_alarm(which, seconds, *args):
        if seconds == 0:   # the alarm lands between `run` returning and the timer being stopped
            signal.raise_signal(signal.SIGALRM)
        return real(which, seconds, *args)

    monkeypatch.setattr(signal, 'setitimer', late_alarm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert tuning.timed_call(lambda value: {'value': value}, 3) == {'value': 3}
    assert any('after the cut finished' in str(w.message) for w in caught)


def test_timed_call_still_stops_a_cut_that_runs_past_the_watchdog(monkeypatch):
    monkeypatch.setattr(tuning, 'CUT_TIMEOUT', 0.05)

    def hang():
        time.sleep(2)
        return 'never'

    with pytest.raises(tuning.CutTimeout):
        tuning.timed_call(hang)


@pytest.mark.parametrize('missing', ['rmse7', 'rmse28', 'roughness'])
def test_coordinate_descent_refuses_a_start_without_figures(missing):
    def evaluate(config):
        figures = {'rmse7': 1., 'rmse28': 1., 'roughness': 1.}
        return figures | ({missing: np.nan} if config.get('label') == 'start' else {})

    start = {'a': 1, 'label': 'start'}
    with pytest.raises(ValueError, match='start'):
        coordinate_descent(evaluate, start, {'a': [1, 2]}, ['a'], rounds=1,
                           setter=lambda config, key, value: {**config, key: value, 'label': f'{key}={value}'})
