"""Tests del Forecaster que no necesitan base de datos (M6: efectos de casa)."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.forecaster import Forecaster
from mtpy.lib.utils import build_blocks

matplotlib.use('Agg')


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

def _synthetic_forecaster(seed=0, noise=None):
    """
    Forecaster sin base de datos: serie sintética de 300 días, dos nombres, cuatro casas con desplazamiento.
    `noise` fija la desviación típica del ruido de una casa (0,8 por defecto; muestras de 1.000).
    """
    rng = np.random.default_rng(seed)
    noise = noise or {}
    start, end = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-10-27')
    days = pd.date_range(start, end, freq='D')
    t = np.arange(len(days))
    truth = pd.DataFrame({'PP': 30 + 3 * np.sin(t / 60), 'PSOE': 22 - 2 * np.sin(t / 80)}, index=days)
    # Casa 1 (+2 al PP) sólo en la primera mitad, casa 2 (-1) sólo en la segunda: el promedio bruto tiene un escalón
    houses = {1: (2., -1., days[:150]), 2: (-1., 1., days[150:]), 3: (0., 0., days), 4: (0.5, 0., days)}
    rows = []
    for pid, (off_pp, off_psoe, span) in houses.items():
        for d in span[pid::10]:
            sd = noise.get(pid, 0.8)
            rows.append({'date': d, 'pollster_id': pid, 'sponsor_id': 0, 'pollster': 'H{}'.format(pid), 'computed': True,
                         'proc_sample': 1000, 'sample_size': 1000, 'weight_over': 1., 'weight_sample': 1., 'weight_rating': 1., 'weight': 1.,
                         'PP': truth.loc[d, 'PP'] + off_pp + rng.normal(0, sd), 'PSOE': truth.loc[d, 'PSOE'] + off_psoe + rng.normal(0, sd)})
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
    fc.max_fc = 0
    fc.disp_enabled, fc.disp_params, fc.dispersion = False, Forecaster.set_disp_params(fc, None), None
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

    # Los efectos se recentran con los pesos de las encuestas en el promedio (`w`, con decaimiento por antigüedad):
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


def test_house_effects_are_centred_with_the_weights_of_the_average():
    # Una casa muy desviada y con mal rating (el CIS con el PP) pesa poco en el promedio: debe pesar igual de
    # poco en el consenso respecto al que se miden los efectos, o corregirlos desplaza el nivel del promedio
    fc, _, _ = _synthetic_forecaster()
    series = fc.series_raw.copy()
    low = series.index.get_level_values('pollster_id') == 3
    series.loc[low, 'PP'] -= 5.
    series.loc[low, ['weight_rating', 'weight']] = 0.2
    fc.series_raw = None
    fc._set_series(series)
    raw_fit = fc.fit('PP')

    pp = fc.fit_house_effects(prior=None).xs('PP', level='name')
    corr_fit = fc.fit('PP')
    assert abs(float((corr_fit - raw_fit).mean())) < 0.2            # el nivel medio del ciclo no cambia
    assert pp.loc[3, 'effect'] - pp.loc[4, 'effect'] == pytest.approx(-5.5, abs=0.5)   # y la casa se corrige entera


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


def test_fc_series_raw_keeps_the_published_polls_after_house_effects():
    fc, _, _ = _synthetic_forecaster()
    published = fc.fc_series['PP'].copy()
    fc.fit_forecast(names=['PP'], max_fc=0)
    assert fc.fc_series_raw['PP'].equals(published)       # lo publicado no cambia
    assert not fc.fc_series['PP'].equals(published)       # lo que se promedia sí


def test_plot_forecast_series_draws_the_published_polls_not_the_corrected_ones():
    fc, _, _ = _synthetic_forecaster()
    fc.blocks = build_blocks(['PP', 'PSOE'], {'PP': '#1d84ce', 'PSOE': '#ef1c27'})
    fc.fit_forecast(max_fc=0)
    raw = fc.series_raw.reset_index(['pollster_id', 'sponsor_id'])

    # Los puntos de la casa destacada y los del resto son los sondeos tal como se publicaron
    _, ax = matplotlib.pyplot.subplots()
    fc.plot_forecast_series(
        names=['PP'], pollster=1, dt_max='2026-10-27', show_forecast=False, show_events=False, show=False, ax=ax
    )
    house, others = (np.sort(c.get_offsets()[:, 1]) for c in ax.collections)
    assert np.allclose(house, np.sort(raw.loc[raw.pollster_id == 1, 'PP']))
    assert np.allclose(others, np.sort(raw.loc[raw.pollster_id != 1, 'PP']))
    matplotlib.pyplot.close('all')


# --- Muestra efectiva: dispersión de cada casa frente a su error muestral ---

def test_house_dispersion_measures_the_excess_noise_of_a_house():
    # Casa 1: ruido de muestreo puro (n = 1000, sd ≈ 1,45 puntos); casa 2: el triple
    rng = np.random.default_rng(1)
    truth = {'PP': 30., 'PSOE': 20.}
    rows = []
    for pid, sd_mult in [(1, 1.), (2, 3.)]:
        for i in range(200):
            rows.append({'date': pd.Timestamp('2026-01-01') + pd.Timedelta(days=i), 'pollster_id': pid, 'proc_sample': 1000,
                         'PP': 30. + rng.normal(0, sd_mult * 100 * np.sqrt(.3 * .7 / 1000)),
                         'PSOE': 20. + rng.normal(0, sd_mult * 100 * np.sqrt(.2 * .8 / 1000))})
    polls = pd.DataFrame(rows).set_index('date').sort_index()
    disp = Forecaster.house_dispersion(polls, _fitted(polls.index, truth), ['PP', 'PSOE'], min_polls=5, n_prior=0)
    assert disp.loc[1, 'ratio'] == pytest.approx(1., abs=0.15)
    assert disp.loc[2, 'ratio'] == pytest.approx(3., abs=0.4)
    assert disp.loc[1, 'factor'] == 1.                        # nunca se premia la poca dispersión
    assert disp.loc[2, 'factor'] == pytest.approx(1. / disp.loc[2, 'ratio'])
    # Con pocas encuestas no hay medida: ratio 1; y el encogimiento acerca la ratio a 1
    few = Forecaster.house_dispersion(polls.iloc[:3], _fitted(polls.index, truth), ['PP'], min_polls=5)
    assert (few['ratio'] == 1.).all()
    shrunk = Forecaster.house_dispersion(polls, _fitted(polls.index, truth), ['PP', 'PSOE'], min_polls=5, n_prior=200)
    assert 1. < shrunk.loc[2, 'ratio'] < disp.loc[2, 'ratio']


def test_fit_dispersion_rescales_the_sample_weight_of_the_noisy_house_only():
    fc, _, _ = _synthetic_forecaster(noise={3: 4.5})
    fc.disp_enabled = True
    fc.disp_params['herding'] = False  # Solo M12: las demás casas oscilan menos que su muestra (0,8 frente a 1,45)
    before = fc.fc_series.groupby('pollster_id')[['weight_sample', 'weight']].mean()

    disp = fc.fit_dispersion()
    after = fc.fc_series.groupby('pollster_id')[['weight_sample', 'weight']].mean()
    assert disp.loc[3, 'ratio'] > 2.
    assert after.loc[3, 'weight_sample'] == pytest.approx(before.loc[3, 'weight_sample'] * disp.loc[3, 'factor'])
    assert after.loc[3, 'weight'] == pytest.approx(before.loc[3, 'weight'] * disp.loc[3, 'factor'])
    for pid in (1, 2, 4):
        assert after.loc[pid, 'weight'] == pytest.approx(before.loc[pid, 'weight'])
    # Los pesos cambian también en la serie publicada, para que los efectos de casa los usen
    assert fc.fc_series_raw.groupby('pollster_id')['weight'].mean().loc[3] == pytest.approx(after.loc[3, 'weight'])
    # fit_forecast lo aplica una sola vez
    fc.fit_forecast(names=['PP'], max_fc=0)
    assert fc.fc_series.groupby('pollster_id')['weight'].mean().loc[3] == pytest.approx(after.loc[3, 'weight'])


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


# --- M10: error estándar del promedio con cluster por casa ---

def test_reg_params_default_is_cluster_by_pollster():
    fc, _, _ = _synthetic_forecaster()
    assert fc.reg_params['cov_type'] == 'cluster'
    assert 'hac_lags' not in fc.reg_params['cov_kwargs']
    hac = Forecaster.set_reg_params(fc, {'cov_type': 'hac'})
    assert hac['cov_kwargs'] == {'hac_lags': 1, 'kernel': 'bartlett'}


def test_fit_cluster_error_exceeds_hac_when_houses_disagree():
    """Sobre la serie bruta (casas desplazadas +2 / -1 / 0 / +0.5) el error con cluster por casa es mayor que el
    HAC en orden de filas, que no ve la dependencia por casa."""
    fc, _, _ = _synthetic_forecaster()
    _, stat_cluster = fc.fit('PP', ret_stat=True)
    fc.reg_params = Forecaster.set_reg_params(fc, {'cov_type': 'hac'})
    _, stat_hac = fc.fit('PP', ret_stat=True)
    err_cluster = stat_cluster.dropna().apply(lambda d: d['err'])
    err_hac = stat_hac.dropna().apply(lambda d: d['err'])
    ratio = (err_cluster / err_hac).dropna()
    assert ratio.median() > 1.2, ratio.describe()



def test_usable_history_needs_three_events():
    """M11: con menos de 3 elecciones previas el prior de los efectos de casa se anula."""
    dates = pd.to_datetime(['2015-05-24', '2019-05-26', '2021-05-04'])
    history = pd.DataFrame({'event_date': dates.repeat(2), 'pollster_id': [1, 2] * 3, 'dev_result_c': 1.})
    assert Forecaster.usable_history(history).shape[0] == 6
    short = Forecaster.usable_history(history.iloc[:4])
    assert short.shape[0] == 0 and list(short.columns) == list(history.columns)


def test_require_polls_names_the_event_without_polls():
    """M11: un evento sin sondeos (una comunidad que acaba de votar) falla con un mensaje claro."""
    polls = pd.DataFrame({'PP': [30.]}, index=pd.to_datetime(['2026-01-01']))
    assert Forecaster.require_polls(polls, 'es-md', '2027-05-23') is polls
    with pytest.raises(ValueError, match='No polls for es-an 2030-06-16'):
        Forecaster.require_polls(polls.iloc[:0], 'es-an', '2030-06-16')


# --- M11b: prior de efectos de casa global (todas las elecciones, con peso por ámbito y partido raíz) ---

def test_house_prior_weighs_regional_elections_by_their_scope_weight():
    """Dos elecciones a la misma distancia y con los mismos sondeos: la nacional pesa 1 y la autonómica 0,5."""
    ref = pd.Timestamp('2027-05-23')
    dates = pd.to_datetime(['2023-07-23', '2023-05-28'])
    history = pd.DataFrame({
        'event_date': dates, 'event_scope': ['es', 'es-md'], 'dev_result_c': [2., -2.], 'level': [20., 20.],
        'n_result': [10, 10], 'w_scope': [1., .5]
    })
    years = ((ref - dates).days / 365.25).to_numpy()
    w = 0.9 ** years * 10 * np.array([1., .5])
    expected = float((w * np.array([.1, -.1])).sum() / (w.sum() + 10 * 1.)) * 30.
    mean, err = Forecaster.house_prior(history, 30., ref, year_decay=0.9, he_tau=0.08, he_prior_events=1., n_cap=10)
    assert mean == pytest.approx(expected) and mean > 0 and err == pytest.approx(0.08 * 30.)
    # Sin la columna, todas las elecciones pesan igual (comportamiento anterior)
    same, _ = Forecaster.house_prior(history.drop(columns='w_scope'), 30., ref, year_decay=0.9)
    assert same < mean


def test_industry_bias_weighs_elections_by_their_scope_weight():
    history = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2023-05-28']), 'event_scope': ['es', 'es-md'], 'party': ['PP', 'PP'],
        'industry': [2., -2.], 'level': [20., 20.], 'w_scope': [1., .5]
    })
    weighted = Forecaster.industry_bias(history, pd.Series({'PP': 30.}), pd.Timestamp('2027-05-23'))
    flat = Forecaster.industry_bias(history.drop(columns='w_scope'), pd.Series({'PP': 30.}), pd.Timestamp('2027-05-23'))
    assert weighted.loc['PP', 'bias'] > 0 and weighted.loc['PP', 'bias'] > flat.loc['PP', 'bias']


def test_usable_history_counts_scope_and_date_pairs():
    """Tres autonómicas del mismo día son tres elecciones."""
    history = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-05-28'] * 3), 'event_scope': ['es-md', 'es-cl', 'es-vc'],
        'pollster_id': [1, 1, 1], 'dev_result_c': 1.
    })
    assert Forecaster.usable_history(history).shape[0] == 3
    assert Forecaster.usable_history(history.iloc[:2]).shape[0] == 0


# --- M13: herding ---

def test_pool_herding_ratio_and_lower_tail():
    # Dos series de una casa con 10 encuestas: la mitad de la varianza esperada -> razón sqrt(0,5)
    rows = pd.DataFrame({'pollster_id': [7, 7], 'name': ['PP', 'PSOE'], 'k': [10, 10], 'ss_obs': [4., 5.], 'ss_exp': [9., 9.]})
    out = Forecaster.pool_herding(rows)
    assert out.loc[7, 'n'] == 10 and out.loc[7, 'n_series'] == 2 and out.loc[7, 'dof'] == 9
    assert out.loc[7, 'ratio'] == pytest.approx(np.sqrt(0.5))
    assert 0 < out.loc[7, 'p_value'] < 0.5  # por debajo de lo esperado: cola inferior
    # Sin herding la razón es 1 y el p-valor ronda 0,5
    same = Forecaster.pool_herding(rows.assign(ss_obs=[9., 9.]))
    assert same.loc[7, 'ratio'] == pytest.approx(1.) and 0.4 < same.loc[7, 'p_value'] < 0.7
    assert Forecaster.pool_herding(rows.iloc[0:0]).shape[0] == 0


def test_shrink_ratio_toward_one():
    out = Forecaster.shrink_ratio(pd.Series([0.5, 2.]), pd.Series([5, 45]), n_prior=5.)
    assert out.tolist() == pytest.approx([0.75, 1.9])


def test_dispersion_factor_is_symmetric():
    ratio = pd.Series([2., 1., 1., 1.], index=list('ABCD'))  # M12: razón recortada por debajo en 1
    herd = pd.Series([0.4, 0.6, 1.3, np.nan], index=list('ABCD'))
    out = Forecaster.dispersion_factor(ratio, herd, max_ratio=10.)
    # A: sobredispersa (manda M12); B: rebaño; C: ni una cosa ni otra; D: sin medida
    assert out.tolist() == pytest.approx([0.5, 0.6, 1., 1.])
    # El suelo es 1 / max_ratio
    assert Forecaster.dispersion_factor(pd.Series([1.]), pd.Series([0.01]), max_ratio=10.).iloc[0] == pytest.approx(0.1)


def test_fit_dispersion_penalizes_the_herder_by_default():
    # Ruido de muestreo puro (1,45 puntos con 1.000 entrevistas al 30 %) salvo la casa 4, pegada a la verdad
    fc, _, _ = _synthetic_forecaster(noise={1: 1.45, 2: 1.45, 3: 1.45, 4: 0.3})
    fc.disp_enabled = True
    assert fc.disp_params['herding'] is True
    disp = fc.fit_dispersion()
    # El promedio sin la casa también tiene error: el residuo no baja hasta el 0,3 / 1,45 del ruido
    assert disp.loc[4, 'herd_ratio'] < 0.75 and disp.loc[4, 'factor'] == pytest.approx(disp.loc[4, 'herd_ratio'])
    others = disp.loc[[1, 2, 3], 'factor']
    assert (others > 0.9).all() and (others > disp.loc[4, 'factor']).all()
    herd = fc.measure_herding(names=['PP', 'PSOE'])
    assert herd.loc[4, 'p_value'] < 0.01 and (herd.loc[[1, 2, 3], 'p_value'] > 0.01).all()


# --- Fase 9: los pesos de los sondeos en un día (`weights_at`) ---
def test_weights_at_matches_the_fitted_average_and_its_band_on_the_same_grid():
    fc, _, _ = _synthetic_forecaster()
    fc.fit_forecast(names=['PP'], max_fc=10)
    when = fc.date_fit_last
    out = fc.weights_at('PP')                       # date=None → date_fit_last
    assert fc.max_fc == 10 and out['stats']['date'] == when
    cell = fc.fc_stat.loc[when, 'PP']
    assert out['stats']['mean'] == pytest.approx(fc.forecast.loc[when, 'PP'])
    assert out['stats']['cmin'] == pytest.approx(cell['cmin']) and out['stats']['cmax'] == pytest.approx(cell['cmax'])
    assert out['stats']['err'] == pytest.approx(cell['err']) and out['stats']['nobs'] == cell['nobs']
    assert out['stats']['neff'] == pytest.approx(cell['neff']) and out['stats']['bandwidth'] > 0
    polls = out['polls']
    assert len(polls) == cell['nobs'] and polls.index.is_monotonic_increasing
    assert list(polls.columns) == ['pollster_id', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight',
                                   'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating']
    assert np.allclose(polls['weight'], polls['weight_kernel'] * polls['weight_over'] * polls['weight_sample'] * polls['weight_rating'])
    assert np.allclose(polls['pct'], fc.fc_series_raw.loc[polls.index, 'PP'])     # lo publicado
    assert np.allclose(polls['pct_adj'], fc.fc_series.loc[polls.index, 'PP'])     # lo que ajusta la recta
    assert not np.allclose(polls['pct'], polls['pct_adj']) and polls['sponsor'].isnull().all()
    line, kernel = out['line'], out['kernel']
    assert line.index[-1] == when and (line.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert line.loc[when, 'mean'] == pytest.approx(out['stats']['mean'])
    assert line.index[0] <= polls.index[0]                       # la ventana empieza en el primer sondeo con kernel apreciable
    assert kernel.index.equals(line.index) and kernel.loc[when] == pytest.approx(1.0) and kernel.max() <= 1.0
    assert kernel.iloc[0] < kernel.loc[when]


def test_weights_at_reads_a_party_without_recent_polls_at_the_end_of_its_grid():
    fc, _, _ = _synthetic_forecaster()
    dates = fc.series.index.get_level_values('date')
    cut = fc.date_last - pd.Timedelta(days=60)
    fc.series.loc[dates > cut, 'PSOE'] = np.nan
    fc.series_raw.loc[dates > cut, 'PSOE'] = np.nan
    fc.fit_forecast(names=['PP', 'PSOE'], max_fc=10, fillna=True)
    last = fc.series.loc[fc.series['PSOE'].notnull()].index.get_level_values('date').max()
    end = last + pd.Timedelta(days=10)
    out = fc.weights_at('PSOE')
    assert end < fc.date_fit_last and out['stats']['date'] == end
    assert out['stats']['mean'] == pytest.approx(fc.forecast.loc[end, 'PSOE'])
    assert out['line'].index[-1] == end and out['polls'].index.max() <= last


def test_weights_at_returns_none_without_three_polls_or_before_the_first():
    fc, _, _ = _synthetic_forecaster()
    assert fc.weights_at('PP', date='2025-12-01') is None
    dates = fc.series.index.get_level_values('date')
    fc.series.loc[dates > dates.min() + pd.Timedelta(days=2), 'PSOE'] = np.nan   # quedan dos sondeos (días 1 y 3)
    with pytest.warns(UserWarning):
        assert fc.weights_at('PSOE') is None


def test_print_weights_keeps_its_columns_and_plot_weights_draws_the_daily_kernel(monkeypatch):
    monkeypatch.setattr('mtpy.lib.forecaster.plot_figure', lambda **kwargs: None)   # `plot_figure` vacía la figura
    fc, _, _ = _synthetic_forecaster()
    out = fc.weights_at('PP', date='2026-06-01')
    table = fc.print_weights('PP', date='2026-06-01', show=False)
    assert list(table.columns) == ['date', 'pollster', 'weight', 'tfs', 'tte', 'sample_size', 'weight_kernel',
                                   'weight_over', 'weight_sample', 'weight_rating']
    assert len(table) == len(out['polls']) and table['tfs'].min() == 0 and table['tte'].min() == 0
    assert table['date'].iloc[0] == out['polls'].index[0].strftime('%d-%b')
    assert np.allclose(table['weight'], out['polls']['weight'])
    _, ax = matplotlib.pyplot.subplots()
    fc.plot_weights('PP', date='2026-06-01', show=False, ax=ax)
    twin = [a for a in ax.figure.axes if a is not ax][0]
    assert len(ax.collections) == 1 and len(ax.collections[0].get_offsets()) == len(out['polls'])
    assert len(twin.lines) == 1 and len(twin.lines[0].get_xdata()) == len(out['kernel'])
    matplotlib.pyplot.close('all')


def test_weights_at_window_covers_every_poll_of_the_regression():
    """Un sondeo lejano con mucho peso entra en la regresión aunque su kernel baje de 0,01: la ventana de la recta,
    la campana y el eje debe cubrirlo (revisión final de la fase 9)."""
    fc, _, _ = _synthetic_forecaster()
    fc.reg_params = Forecaster.set_reg_params(fc, {'bw_type': 'fixed', 'bw': 30})
    first = fc.series.index[0]
    fc.series.loc[first, 'weight'] = 50.
    date = first[0] + pd.Timedelta(days=96)          # kernel exp(-(96/30)²/2) ≈ 0,006 < 0,01
    out = fc.weights_at('PP', date=date)
    polls, line = out['polls'], out['line']
    assert polls.index.min() == first[0] and polls['weight_kernel'].min() < 1e-2
    assert line.index[0] <= polls.index.min() and line.index[-1] >= max(polls.index.max(), date)
    assert out['kernel'].index.equals(line.index) and out['stats']['bandwidth'] == 30.


# --- Backtest: ratings recalculados en memoria ---

def test_weight_ratings_override_replaces_the_stored_weight(monkeypatch):
    """`weight_ratings` sustituye al `weight_rating` guardado en los sondeos de las casas del índice (el de su elección);
    las ausentes quedan con el relleno por calidad y los resultados oficiales no se tocan."""
    from mtpy.lib import forecaster as fc_module
    event_date = pd.Timestamp('2023-07-23')
    polls = pd.DataFrame({
        'event_date': event_date, 'date': pd.to_datetime(['2023-07-10', '2023-07-12', '2023-07-14']),
        'pollster_id': [1, 2, 1], 'sponsor_id': 0, 'pollster': ['A', 'B', 'A'], 'computed': True,
        'weight_over': [1.0, 1.0, 2.0], 'weight_sample': [0.5, 0.5, 1.5], 'weight_rating': [9.0, 9.0, 9.0],
        'PP': [33.0, 32.0, 34.0], 'PSOE': [30.0, 31.0, 29.0],
    }).set_index(['date', 'pollster_id', 'sponsor_id'])
    events = pd.DataFrame({
        'date': pd.to_datetime(['2019-11-10', '2023-07-23']), 'pollster_id': np.nan, 'sponsor_id': np.nan,
        'event_date': pd.to_datetime(['2019-11-10', '2023-07-23']), 'weight_over': 1.0, 'weight_sample': 1.0,
        'PP': [20.0, 33.0], 'PSOE': [28.0, 30.0], 'votes': [1000, 1000], 'computed': True,
    }).set_index('date')
    extra = ['start_date', 'end_date', 'sponsor', 'sample_size', 'parties', 'days', 'mtype', 'proc_sample', 'rating',
             'error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias',
             'bias_dev_adj', 'bias_dev_err']
    polls = polls.assign(**{col: np.nan for col in extra})
    parties = pd.DataFrame({'name': ['PP', 'PSOE'], 'color': ['#00f', '#f00']})
    pollsters = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'quality': [80., 40.]})
    monkeypatch.setattr(fc_module, 'get_parties', lambda: parties)
    monkeypatch.setattr(fc_module, 'get_pollsters', lambda: pollsters)

    def build(weight_ratings):
        fc = Forecaster.__new__(Forecaster)
        fc.scope, fc.event_date, fc.verbose, fc.weight_ratings = 'es', '2023-07-23', 0, weight_ratings
        fc.date_start, fc.date_end = pd.Timestamp('2019-11-10'), event_date
        fc.bmap = None
        fc.load_events = lambda: events
        fc.load_polls = lambda: polls
        return fc.build_series()

    index = pd.MultiIndex.from_tuples([(event_date, 1)], names=['event_date', 'pollster_id'])
    series = build(pd.Series([2.0], index=index)).series_raw
    house_a = series.loc[series.pollster == 'A']
    house_b = series.loc[series.pollster == 'B']
    assert (house_a.weight_rating == 2.0).all()
    assert np.allclose(house_a.weight, house_a.weight_over * house_a.weight_sample * 2.0)
    quality = pollsters.set_index('id').quality / 100
    fill = np.log1p(quality[2]) / np.log1p(quality.mean())
    assert house_b.weight_rating.iloc[0] == pytest.approx(fill)
    assert series.loc[series.pollster.isnull(), 'weight_rating'].isnull().all()
    # Sin la serie, el valor guardado manda
    stored = build(None).series_raw
    assert (stored.loc[stored.pollster.notnull(), 'weight_rating'] == 9.0).all()
