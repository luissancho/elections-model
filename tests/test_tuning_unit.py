"""Unit tests of the tuning harness of the average on future polls (no database)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib import tuning
from mtpy.lib.forecaster import Forecaster
from mtpy.lib.tuning import (GRID, ORDER_SHRINKAGE, ORDER_SMOOTHING, apply_setting, autocorrelation,
                             coordinate_descent, cycle_cuts, default_config, future_polls, predict_polls, roughness,
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
