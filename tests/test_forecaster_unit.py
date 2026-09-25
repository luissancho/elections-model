"""Tests del Forecaster que no necesitan base de datos (M6: efectos de casa)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.forecaster import Forecaster


def _polls(offsets, n=6, start='2026-01-01', truth=None):
    """Encuestas de varias casas con desplazamiento fijo sobre una verdad plana."""
    truth = truth or {'PP': 30., 'PSOE': 20.}
    rows = []
    for pid, (off, n_polls) in enumerate(offsets, start=1):
        for i in range(n_polls):
            rows.append({'date': pd.Timestamp(start) + pd.Timedelta(days=7 * i + pid), 'pollster_id': pid,
                         'PP': truth['PP'] + off, 'PSOE': truth['PSOE'] - off / 2, 'weight': 1.})
    return pd.DataFrame(rows).set_index('date').sort_index()


def _fitted(index, truth=None):
    truth = truth or {'PP': 30., 'PSOE': 20.}
    days = pd.date_range(index.min() - pd.Timedelta(days=3), index.max() + pd.Timedelta(days=3), freq='D')
    return pd.DataFrame({k: v for k, v in truth.items()}, index=days)


def test_house_deviations_recover_known_offsets():
    polls = _polls([(2., 6), (-1., 6), (0., 6)])
    dev = Forecaster.house_deviations(polls, _fitted(polls.index), ['PP', 'PSOE'], polls['weight'], min_polls=5)
    assert list(dev.index.names) == ['pollster_id', 'name']
    assert dev.loc[(1, 'PP'), 'dev'] == pytest.approx(2.) and dev.loc[(1, 'PSOE'), 'dev'] == pytest.approx(-1.)
    assert dev.loc[(2, 'PP'), 'dev'] == pytest.approx(-1.) and dev.loc[(3, 'PP'), 'dev'] == pytest.approx(0.)
    assert (dev['n'] == 6).all()
    assert (dev['dev_err'] == 0).all()   # sin ruido, error típico nulo


def test_house_deviations_short_houses_and_missing_fit():
    polls = _polls([(2., 6), (1., 2)])
    fitted = _fitted(polls.index)
    fitted.loc[fitted.index[:2]] = np.nan          # días sin promedio: residuo descartado
    dev = Forecaster.house_deviations(polls, fitted, ['PP', 'PSOE'], polls['weight'], min_polls=5)
    assert dev.loc[(2, 'PP'), 'n'] == 2
    assert np.isinf(dev.loc[(2, 'PP'), 'dev_err'])   # menos de min_polls: sin información
    assert dev.loc[(1, 'PP'), 'n'] <= 6
    # Ruido: el error típico es positivo y baja con más encuestas
    rng = np.random.default_rng(0)
    noisy = _polls([(2., 6), (2., 24)])
    noisy['PP'] += rng.normal(0, 1, noisy.shape[0])
    dev = Forecaster.house_deviations(noisy, _fitted(noisy.index), ['PP'], noisy['weight'], min_polls=5)
    assert dev.loc[(2, 'PP'), 'dev_err'] < dev.loc[(1, 'PP'), 'dev_err'] > 0


def test_apply_house_effects_only_touches_poll_rows():
    series = pd.DataFrame({
        'pollster': [None, 'A', 'A', 'B'], 'pollster_id': [np.nan, 1, 1, 2],
        'PP': [33., 32., 34., 31.], 'PSOE': [28., 27., 27., 26.], 'other': [1., 2., 3., 4.]
    }, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2023-07-23'), np.nan, np.nan), (pd.Timestamp('2026-01-05'), 1, 0),
         (pd.Timestamp('2026-02-05'), 1, 0), (pd.Timestamp('2026-02-06'), 2, 0)], names=['date', 'pollster_id', 'sponsor_id']))
    effects = pd.DataFrame({'effect': [2., -1.]}, index=pd.MultiIndex.from_tuples([(1, 'PP'), (2, 'PSOE')], names=['pollster_id', 'name']))
    out = Forecaster.apply_house_effects(series, effects, ['PP', 'PSOE'])
    assert out['PP'].tolist() == [33., 30., 32., 31.]
    assert out['PSOE'].tolist() == [28., 27., 27., 27.]
    assert out['other'].tolist() == series['other'].tolist()
    assert series['PP'].tolist() == [33., 32., 34., 31.]   # sin efectos laterales


def test_house_prior_from_history_is_relative_shrunk_and_decayed():
    history = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-04-28', '2023-07-23']), 'dev_result_c': [2., 3.], 'level': [20., 30.], 'n_result': [10, 10]
    })
    ref = pd.Timestamp('2027-08-22')
    prior, err = Forecaster.house_prior(history, level=25., ref_date=ref, year_decay=0.9, he_tau=0.08, he_prior_events=1)
    w = [0.9 ** ((ref - d).days / 365.25) * 10 for d in history['event_date']]
    expected = (0.1 * w[0] + 0.1 * w[1]) / (w[0] + w[1] + 10) * 25.
    assert prior == pytest.approx(expected)
    assert 0 < prior < 2.5                      # encogido respecto al 10 % relativo
    assert err == pytest.approx(0.08 * 25.)
    # Sin historia: prior 0 con la misma incertidumbre; nivel con suelo de 2 puntos
    assert Forecaster.house_prior(history.iloc[:0], 25., ref) == (0., pytest.approx(2.))
    assert Forecaster.house_prior(history.iloc[:0], 0.5, ref)[1] == pytest.approx(0.08 * 2.)
    # Desviación relativa acotada: 5 puntos sobre un nivel de 2 no pasan de 0,5
    big = pd.DataFrame({'event_date': [pd.Timestamp('2023-07-23')], 'dev_result_c': [5.], 'level': [1.], 'n_result': [10]})
    w1 = 0.9 ** ((ref - pd.Timestamp('2023-07-23')).days / 365.25) * 10
    assert Forecaster.house_prior(big, 10., ref)[0] == pytest.approx(0.5 * w1 / (w1 + 10) * 10.)


def test_center_effects_weighted_mean_is_zero_per_name():
    effects = pd.DataFrame({'effect': [2., -1., 0., 1.]}, index=pd.MultiIndex.from_tuples(
        [(1, 'PP'), (2, 'PP'), (3, 'PP'), (1, 'PSOE')], names=['pollster_id', 'name']))
    totals = pd.Series({1: 1., 2: 1., 3: 2.})
    out = Forecaster.center_effects(effects, totals)
    assert out.loc[(1, 'PP'), 'effect'] == pytest.approx(1.75)
    assert out.loc[(2, 'PP'), 'effect'] == pytest.approx(-1.25)
    assert out.loc[(3, 'PP'), 'effect'] == pytest.approx(-0.25)
    assert out.loc[(1, 'PSOE'), 'effect'] == pytest.approx(0.)     # una sola casa: se centra a 0
    assert 'center' in out.columns


# --- M6: backfitting completo sin base de datos ---

def _synthetic_forecaster(seed=0):
    """Forecaster sin base de datos: serie sintética de 300 días, dos nombres, cuatro casas con desplazamiento."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-10-27')
    days = pd.date_range(start, end, freq='D')
    t = np.arange(len(days))
    truth = pd.DataFrame({'PP': 30 + 3 * np.sin(t / 60), 'PSOE': 22 - 2 * np.sin(t / 80)}, index=days)
    # Casa 1 (+2 al PP) sólo en la primera mitad, casa 2 (-1) sólo en la segunda: el promedio bruto tiene un escalón
    houses = {1: (2., -1., days[:150]), 2: (-1., 1., days[150:]), 3: (0., 0., days), 4: (0.5, 0., days)}
    rows = []
    for pid, (off_pp, off_psoe, span) in houses.items():
        for d in span[pid::10]:
            rows.append({'date': d, 'pollster_id': pid, 'sponsor_id': 0, 'pollster': 'H{}'.format(pid), 'computed': True,
                         'weight_over': 1., 'weight_sample': 1., 'weight_rating': 1., 'weight': 1.,
                         'PP': truth.loc[d, 'PP'] + off_pp + rng.normal(0, 0.8), 'PSOE': truth.loc[d, 'PSOE'] + off_psoe + rng.normal(0, 0.8)})
    series = pd.DataFrame(rows).set_index(['date', 'pollster_id', 'sponsor_id']).sort_index()
    series['-'] = 100. - series[['PP', 'PSOE']].sum(axis=1)

    fc = Forecaster.__new__(Forecaster)
    fc.scope, fc.event_date = 'es', '2027-08-22'
    fc.names = ['PP', 'PSOE']
    fc.date_start, fc.date_end = start - pd.Timedelta(days=30), pd.Timestamp('2027-08-22')
    fc.date_first, fc.date_last = series.index.get_level_values('date').min(), series.index.get_level_values('date').max()
    fc.alpha, fc.verbose = 0.05, 0
    fc.reg_params = Forecaster.set_reg_params(fc, None)
    fc.he_params = Forecaster.set_he_params(fc, {'prior': None})
    fc.he_enabled, fc.house_effects, fc.date_fit_last = True, None, None
    fc.series_raw = None
    fc._set_series(series)
    return fc, truth, houses


def test_fit_house_effects_recovers_centered_offsets_and_removes_house_step():
    fc, truth, houses = _synthetic_forecaster()
    raw = fc.series_raw.copy()
    raw_fit = fc.fit('PP')                     # promedio bruto, antes de corregir

    effects = fc.fit_house_effects(prior=None, n_iter=3)
    assert list(effects.index.names) == ['pollster_id', 'name']
    assert {'n', 'dev', 'dev_err', 'prior', 'prior_err', 'effect', 'effect_err'} <= set(effects.columns)

    # Los efectos se recentran con los pesos de precisión del algoritmo (`w`, con decaimiento por antigüedad):
    # media ponderada 0 por nombre; y recuperan los desplazamientos relativos
    pp = effects.xs('PP', level='name')
    w = pp['w']
    assert abs((pp['effect'] * w).sum() / w.sum()) < 1e-6
    offs = pd.Series({pid: h[0] for pid, h in houses.items()})
    centered = offs - (offs * w).sum() / w.sum()
    assert (pp['effect'] - centered).abs().max() < 0.5, (pp['effect'], centered)

    # La serie corregida es la bruta menos el efecto de su casa (sólo encuestas); series_raw no cambia
    polls = fc.series['pollster'].notnull()
    pids = fc.series.index.get_level_values('pollster_id')
    expected = raw.loc[polls, 'PP'].to_numpy() - pp['effect'].reindex(pids[polls]).to_numpy()
    assert np.allclose(fc.series.loc[polls, 'PP'].to_numpy(), expected)
    assert fc.series_raw['PP'].equals(raw['PP'])
    assert np.allclose(fc.series['-'], 100 - fc.series[['PP', 'PSOE']].sum(axis=1))

    # Tras corregir, las casas ya no se desvían del promedio, y el escalón artificial desaparece
    corr_fit = fc.fit('PP')
    dev = Forecaster.house_deviations(fc.fc_series, corr_fit.to_frame('PP'), ['PP'], fc.fc_series['weight'])
    assert dev['dev'].abs().max() < 0.5
    ix = truth.index
    shift = float((offs * w).sum() / w.sum())
    rmse_raw = np.sqrt(np.nanmean((raw_fit.reindex(ix) - truth['PP'] - shift) ** 2))
    rmse_corr = np.sqrt(np.nanmean((corr_fit.reindex(ix) - truth['PP'] - shift) ** 2))
    assert rmse_corr < rmse_raw


def test_fit_forecast_applies_house_effects_lazily_and_can_be_disabled():
    fc, _, _ = _synthetic_forecaster()
    fc.fit_forecast(names=['PP'], max_fc=0)
    assert fc.house_effects is not None                       # se estimaron al ajustar
    assert not fc.series['PP'].equals(fc.series_raw['PP'])

    off, _, _ = _synthetic_forecaster()
    off.he_enabled = False
    off.fit_forecast(names=['PP'], max_fc=0)
    assert off.house_effects is None
    assert off.series['PP'].equals(off.series_raw['PP'])


def test_industry_bias_from_history_is_relative_decayed_and_uncertain():
    history = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-04-28'] * 2 + ['2023-07-23'] * 2), 'pollster_id': [1, 2, 1, 2],
        'party': ['PSOE'] * 4, 'industry': [-1., -1., -1.5, -1.5], 'level': [28., 28., 31.7, 31.7]
    })
    ref = pd.Timestamp('2027-08-22')
    out = Forecaster.industry_bias(history, pd.Series({'PSOE': 27., 'PP': 32.}), ref, year_decay=0.9, prior_events=1.)
    assert list(out.columns) == ['n_events', 'rel', 'rel_err', 'bias', 'bias_err']
    w = [0.9 ** ((ref - d).days / 365.25) for d in pd.to_datetime(['2019-04-28', '2023-07-23'])]
    r = [-1. / 28., -1.5 / 31.7]
    rel = (w[0] * r[0] + w[1] * r[1]) / (w[0] + w[1] + 1.)
    assert out.loc['PSOE', 'n_events'] == 2
    assert out.loc['PSOE', 'rel'] == pytest.approx(rel)
    assert out.loc['PSOE', 'bias'] == pytest.approx(rel * 27.)
    assert out.loc['PSOE', 'bias_err'] > 0
    # Un partido sin historia: sesgo 0 sin incertidumbre añadida
    assert out.loc['PP', 'n_events'] == 0 and out.loc['PP', 'bias'] == 0 and out.loc['PP', 'bias_err'] == 0
    # Una sola elección: la propia magnitud es la incertidumbre (no hay evidencia de estabilidad)
    one = Forecaster.industry_bias(history.iloc[2:], pd.Series({'PSOE': 27.}), ref)
    assert one.loc['PSOE', 'bias_err'] == pytest.approx(abs(one.loc['PSOE', 'bias']))
