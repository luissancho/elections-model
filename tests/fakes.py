"""Dobles de prueba de la publicación: un Simulator sintético que ejercita sus métodos reales de salida sin base
y un Computer sintético con los resúmenes de casas que publica `publish analysis`."""
import json
import os
import types
import warnings

import numpy as np
import pandas as pd

from mtpy.core.io import FileSystem
from mtpy.lib import publish
from mtpy.lib.simulator import Simulator

NAMES = ['PP', 'PSOE', 'VOX']
REGIONS = [0, 28, 8]  # total del ámbito, Madrid, Barcelona
SEATS = {0: 10, 28: 6, 8: 4}
BMAPS = {'main': NAMES, 'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']},
         'vs': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}}
COLORS = {'PP': '#1d84ce', 'PSOE': '#ef1c27', 'VOX': '#63be21'}


def synthetic_forecaster(date_last='2026-10-01', date_fit_last='2026-10-05'):
    """Salidas de un Forecaster ajustado: promedio y estadísticos diarios, sondeos, resultado anterior,
    efectos de casa y dispersión, con NaN donde el modelo real los deja."""
    dates = pd.date_range('2026-08-01', '2026-10-20', freq='D', name='date')
    forecast = pd.DataFrame({'PP': 40., 'PSOE': 30., 'VOX': 15.}, index=dates)
    forecast.loc[dates < '2026-09-01'] = np.nan  # antes del primer sondeo no hay promedio
    forecast['-'] = 100. - forecast[NAMES].sum(axis=1, min_count=1)
    fitted = (dates >= '2026-09-01') & (dates <= date_fit_last)
    fc_stat = pd.DataFrame({
        n: [{'mean': forecast.loc[d, n], 'cmin': forecast.loc[d, n] - 2., 'cmax': forecast.loc[d, n] + 2.,
             'err': 1., 'nobs': 40., 'neff': 30.} if ok else None for d, ok in zip(dates, fitted)]
        for n in NAMES
    }, index=dates)
    polls = pd.DataFrame({
        'date': pd.to_datetime(['2026-09-05', '2026-09-12', '2026-09-19', '2026-09-26', '2026-09-30', date_last]),
        'pollster_id': [1, 2, 1, 3, 2, 1], 'sponsor_id': [0, 0, 0, 5, 0, 0],
        'pollster': ['CIS', 'GAD3', 'CIS', '40dB', 'GAD3', 'CIS'], 'sponsor': [None, None, None, 'El País', None, None],
        'start_date': pd.to_datetime(['2026-09-01', '2026-09-08', '2026-09-15', '2026-09-22', '2026-09-26', '2026-09-28']),
        'end_date': pd.to_datetime(['2026-09-04', '2026-09-11', '2026-09-18', '2026-09-25', '2026-09-29', date_last]),
        'sample_size': [4000, 1000, 4000, 2000, 1000, 4000], 'mtype': ['cati', 'cati', 'cati', 'online', 'cati', 'cati'],
        'rating': [0.8, 0.9, 0.8, 0.7, 0.9, 0.8], 'weight': [1.2, 0.9, 1.2, 0.8, 0.9, 1.2],
        'PP': [41., 39.5, 40.2, 40., 39.8, 40.5], 'PSOE': [29., 31., 30.1, 30., 30.5, 29.5], 'VOX': [15., 14.5, np.nan, 15.5, 15., 14.8],
    }).set_index('date')
    polls['-'] = 100. - polls[NAMES].sum(axis=1, min_count=1)
    results = pd.DataFrame({'date': [pd.Timestamp('2023-07-23')], 'pollster': [None], 'PP': [33.1], 'PSOE': [31.7],
                            'VOX': [12.4], '-': [22.8]}).set_index('date')
    house_effects = pd.DataFrame({
        'pollster': ['CIS', 'CIS', 'GAD3', 'GAD3'], 'n': [3, 3, 2, 2], 'w': [1.2, 1.2, 0.9, 0.9], 'level': [40., 30., 40., 30.],
        'dev': [0.5, -0.4, -0.3, 0.6], 'dev_err': [0.3, 0.3, 0.4, 0.4], 'prior': [0.4, np.nan, -0.2, 0.5],
        'prior_err': [0.5, np.nan, 0.5, 0.5], 'effect': [0.45, -0.4, -0.25, 0.55], 'effect_err': [0.25, 0.3, 0.3, 0.3],
        'center': [0.1, 0.1, 0.1, 0.1],
    }, index=pd.MultiIndex.from_tuples([(1, 'PP'), (1, 'PSOE'), (2, 'PP'), (2, 'PSOE')], names=['pollster_id', 'name']))
    dispersion = pd.DataFrame({
        'pollster': ['CIS', 'GAD3', '40dB'], 'n': [3, 2, 1], 'ss_obs': [1.1, 0.8, np.nan], 'ss_exp': [1.0, 1.0, np.nan],
        'ratio_raw': [1.1, 0.8, np.nan], 'ratio': [1.05, 0.9, 1.], 'factor': [0.95, 1., 1.], 'herd_ratio': [np.nan, 0.85, np.nan],
    }, index=pd.Index([1, 2, 3], name='pollster_id'))
    def weights_at(name, date=None):
        """Pesos sintéticos: kernel creciente con la fecha, recta plana en el promedio y campana de 30 días."""
        if name not in NAMES:
            return None
        when = pd.Timestamp(date) if date is not None else pd.Timestamp(date_fit_last)
        frame = polls[polls[name].notnull()]
        out = pd.DataFrame({
            'pollster_id': frame['pollster_id'], 'pollster': frame['pollster'], 'sponsor': frame['sponsor'],
            'sample_size': frame['sample_size'], 'pct': frame[name], 'pct_adj': frame[name] - 0.3,
            'weight_kernel': np.linspace(0.4, 1., len(frame)), 'weight_over': 1., 'weight_sample': frame['weight'],
            'weight_rating': 1.,
        }, index=frame.index)
        out['weight'] = out['weight_kernel'] * out['weight_over'] * out['weight_sample'] * out['weight_rating']
        out = out[['pollster_id', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight', 'weight_kernel',
                   'weight_over', 'weight_sample', 'weight_rating']]
        days = pd.date_range(frame.index.min(), max(frame.index.max(), when), freq='D')
        mean = float(forecast[name].dropna().iloc[0])
        line = pd.DataFrame({'mean': mean, 'cmin': mean - 2., 'cmax': mean + 2., 'err': 1.}, index=days)
        kernel = pd.Series(np.exp(-np.square((days - when).days / 30.) / 2), index=days)
        stats = {'date': when, 'mean': mean, 'cmin': mean - 2., 'cmax': mean + 2., 'err': 1., 'nobs': len(frame),
                 'neff': 0.8 * len(frame), 'bandwidth': 30.}
        return {'polls': out, 'line': line, 'kernel': kernel, 'stats': stats}

    calls = []

    def fit_groups(groups, max_fc=0):
        """Bloques sintéticos: la suma constante de sus partidos en los días ajustados, con banda de ±1."""
        calls.append(('fit_groups', max_fc))
        frame = pd.DataFrame({g: forecast[members].sum(axis=1).where(fitted) for g, members in groups.items()},
                             index=dates)
        stat = pd.DataFrame({g: [{'mean': v, 'cmin': v - 1., 'cmax': v + 1., 'err': .5} if ok else None
                                 for v, ok in zip(frame[g], fitted)] for g in groups}, index=dates)
        return frame, stat

    def cycle_births(groups, min_days=90):
        """VOX entra a mitad de ciclo si algún grupo lo lleva."""
        return {'VOX': '2026-09-19'} if any('VOX' in members for members in groups.values()) else {}

    return types.SimpleNamespace(
        forecast=forecast, fc_stat=fc_stat, fc_series_raw=polls, nfc_series=results, house_effects=house_effects,
        dispersion=dispersion, date_last=pd.Timestamp(date_last), date_fit_last=pd.Timestamp(date_fit_last),
        bmaps=BMAPS, colors=COLORS, names=NAMES, weights_at=weights_at, max_fc=10, calls=calls,
        fit_groups=fit_groups, cycle_births=cycle_births,
    )


def synthetic_simulator(mode='nowcast', n_sim=200, seed=0):
    """Simulator reconstruido con `__new__` y arrays sintéticos (patrón `test_simulator_unit.py:438`):
    `summary`, `dist`, `unit_summary`, `vote_forecast`, `projection`... son los métodos reales."""
    rng = np.random.default_rng(seed)
    sim = Simulator.__new__(Simulator)
    sim.scope, sim.event_date, sim.mode, sim.seed, sim.verbose = 'es', '2026-11-29', mode, 42, 0
    sim.drange, sim.alpha = (6, None), 0.05
    sim.names, sim.default_region, sim.regions = NAMES, 0, REGIONS
    sim.region_names = {0: 'es', 28: 'Madrid', 8: 'Barcelona'}
    # Como en producción: los totales de la elección futura sólo traen escaños (el cargador no rellena `votes`)
    # y los votos válidos de la elección anterior están en `prev_totals`
    sim.reg_totals = pd.DataFrame({'votes': [np.nan, np.nan, np.nan], 'seats': [10, 6, 4]},
                                  index=pd.Index(REGIONS, name='region_id'))
    sim.prev_totals = pd.DataFrame({'votes': [1000., 600., 400.], 'blank_pct': [1.0, 1.2, 0.8]},
                                   index=pd.Index(REGIONS, name='region_id'))
    sim.params = {'n_sim': n_sim, 'split': True, 'random': True, 'names': NAMES, 'regions': REGIONS,
                  'horizon': None, 'date_prior': 'historical'}
    sim.cols_forecast = ['mean', 'regional', 'err', 'nobs', 'error']
    sim.cols_frame = sim.cols_forecast + ['pct', 'pct_err', 'drift', 'std_err', 'rand', 'vpred', 'dof']
    sim.cols_unit = ['prev_pct', 'vpred_pct']
    sim.forecast = pd.DataFrame({'mean': [40., 30., 15., 15.], 'regional': [0, 0, 0, 0], 'err': [1.5, 1.2, 1.0, np.nan],
                                 'nobs': [40., 40., 30., np.nan], 'error': [2.0, 1.8, 1.5, np.nan]}, index=NAMES + ['-'])
    sim.as_of, sim.deadline, sim.horizon_max = pd.Timestamp('2026-10-05'), pd.Timestamp('2026-11-29'), 55
    sim.ages = pd.Series({'PP': 40., 'PSOE': 40., 'VOX': 12.}, name='age')
    sim.v2err = sim.v2drift = sim.v2seats = None
    sim.computer, sim.est_scope, sim.n_last = None, 'es', 1  # no past elections: no error estimator
    sim.error_model = sim.error_model_used = 'polls'  # error terminal de un sondeo suelto (por defecto)
    sim.drift_scale = None  # deriva sin escalar (por defecto)
    sim.tail_dof = None  # grados de libertad de la t por partido, `nobs - 2` (por defecto)
    sim.composition_ratio, sim.composition_rho = 1.0, 0.
    sim.house_effects = sim.dispersion = sim.regional_noise = True
    sim.industry_bias, sim.composition, sim.smap = False, None, {'UP': [{'agg': ['UP', 'MP']}]}
    sim.threshold, sim.threshold_scope = 3.0, None
    sim.v2swing, sim.rng, sim.rng_swing, sim.region_groups, sim.central_table = None, np.random.default_rng(42), None, {}, None
    # Resultados previos por región (la fila 0 es la media ponderada por `prev_totals.votes`): la base del swing
    prev = pd.DataFrame({'PP': [38., 42., 32.], 'PSOE': [32., 28., 38.], 'VOX': [14., 15., 12.]},
                        index=pd.Index(REGIONS, name='region_id'))
    sim.prev_results = pd.concat({'votes': prev * 10, 'pct': prev, 'seats': prev * 0}, axis=1)
    sim.parties = pd.DataFrame({'id': [1, 2, 3], 'name': NAMES, 'fullname': ['Partido Popular', 'PSOE', 'Vox'],
                                'color': [COLORS[n] for n in NAMES], 'block': ['Derecha', 'Izquierda', 'Derecha'],
                                'regional': [0, 0, 0]})
    sim.model = synthetic_forecaster()

    # Simulaciones: cuotas nacionales, cuotas y escaños por circunscripción; el total (región 0) es la suma
    vpred = sim.cols_frame.index('vpred')
    shares = rng.normal([40., 30., 15.], [2., 2., 1.5], size=(n_sim, 3)).clip(1.)
    others = (100. - shares.sum(axis=1)).clip(0.)
    frames = np.zeros((n_sim, 4, len(sim.cols_frame)))
    frames[:, :3, vpred], frames[:, 3, vpred] = shares, others
    units = np.zeros((n_sim, 3, 4, 2))
    results = np.zeros((n_sim, 3, 3), dtype=int)
    units[:, 0, :3, 1], units[:, 0, 3, 1] = shares, others
    for loc, region in enumerate(REGIONS[1:], start=1):
        local = (shares + rng.normal(0., 1., size=(n_sim, 3))).clip(1.)
        units[:, loc, :3, 1], units[:, loc, 3, 1] = local, (100. - local.sum(axis=1)).clip(0.)
        results[:, loc] = np.stack([rng.multinomial(SEATS[region], p / p.sum()) for p in local])
    results[:, 0] = results[:, 1:].sum(axis=1)
    sim.frames, sim.units, sim.results = frames.round(2), units.round(2), results
    sim.horizons = np.zeros(n_sim, dtype=int)
    return sim


class FakeSimulator:
    """Fábrica con la firma `Simulator(scope, event_date, **kwargs)`: devuelve el simulador sintético con
    `fit_forecast` y `run` inertes y anota las llamadas; con `fail`, lanza esa excepción al construir; con `warn`,
    `fit_forecast` emite esos `UserWarning` (como los avisos por partido del `Forecaster`)."""

    def __init__(self, fail=None, warn=()):
        self.calls, self.fail, self.warn = [], fail, list(warn)

    def __call__(self, scope, event_date, **kwargs):
        if self.fail is not None:
            raise self.fail
        sim = synthetic_simulator(mode=kwargs.get('mode', 'nowcast'))
        sim.scope, sim.event_date = scope, event_date
        self.calls.append(('init', scope, event_date, kwargs))

        def fit_forecast(**kw):
            self.calls.append(('fit_forecast', kw))
            for message in self.warn:
                warnings.warn(message)
            return sim.forecast

        def run(**kw):
            self.calls.append(('run', sim.mode, kw))
            return sim

        sim.fit_forecast, sim.run = fit_forecast, run
        return sim


HOUSES = ['40dB', 'CIS', 'GAD3']
ANALYSIS_EVENTS = ['2019-11-10', '2023-07-23', '2026-11-29']  # dos elecciones pasadas y el ciclo en curso


def synthetic_analysis():
    """Resúmenes de casas con la forma de los métodos reales del `Computer`, para 3 casas, 2 elecciones pasadas
    y el ciclo en curso: `house_effects` (`print_house_effects`), `herding_summary`, `herding` (filas de
    `get_herding`), `poll_errors` (`get_polls_metric('error', bmap='vs', ...)`, con `event_date` ISO) y
    `error_data` (`get_error_estimator_data`). Lleva NaN donde el modelo real los deja y decimales de sobra."""
    names = NAMES + list(BMAPS['vs'])
    index = pd.MultiIndex.from_product([HOUSES, names], names=['pollster', 'name'])
    house_effects = pd.DataFrame({
        # 40dB sin historia (sólo ciclo); CIS con dos elecciones; GAD3 con una y sin sondeos en el ciclo
        'n_events': [0.] * 5 + [2.] * 5 + [1.] * 5,
        'hist': [np.nan] * 5 + [1.234, -0.567, 0.891, 2.125, -0.567] + [-0.4321, 0.9876, 0.1111, -0.321, 0.9876],
        'trend': [np.nan] * 5 + [0.3456, -0.1234, 0.0789, 0.4245, -0.1234] + [np.nan] * 5,
        'cycle': [0.5555, -0.4444, 0.3333, 0.8888, -0.4444] + [1.0101, -0.2020, 0.3030, 1.3131, -0.2020] + [np.nan] * 5,
        'n_cycle': [4.] * 5 + [6.] * 5 + [np.nan] * 5,
    }, index=index)
    herding_summary = pd.DataFrame({
        'herding': [0.81234, 1.04567, 0.65432], 'herding_last': [0.81234, 1.10101, 0.70707],
        'herding_p': [0.04567, 0.61234, 0.00123], 'herding_n': [4., 25., 18.], 'herding_events': [1., 2., 2.],
    }, index=pd.Index(HOUSES, name='pollster'))
    herding = pd.DataFrame({  # desordenado, como llega de la base
        'event_date': pd.to_datetime(['2026-11-29'] * 3 + ['2023-07-23'] * 2),
        'event_scope': 'es', 'pollster_id': [3, 1, 2, 1, 2], 'pollster': ['40dB', 'CIS', 'GAD3', 'CIS', 'GAD3'],
        'n': [4, 10, 8, 15, 10], 'n_series': 4, 'ss_obs': [1.1, 2.2, 0.9, 3.3, 1.4], 'ss_exp': [1.6, 2.0, 1.9, 3.0, 2.1],
        'dof': [12, 30, 24, 45, 30], 'ratio': [0.81234, 1.10101, 0.70707, 0.99876, 0.61234],
        'p_value': [0.04567, 0.61234, 0.00123, 0.45678, 0.00045],
    })
    poll_errors = pd.DataFrame({
        'event_date': '2023-07-23', 'sample_size': [1200, 0, 3000], 'proc_sample': [1200, 1000, 3000],
        'weight': [0.81234, 0.56789, 1.23456], 'error_avg': [1.2345, 2.3456, 0.9876], 'bias_avg': [0.1234, -0.5678, 0.2468],
        'error_blocks': [2.3456, 4.5678, 1.3579], 'bias_blocks': [0.2345, -1.2345, 0.5432],
        'error_within': [0.4321, 0.8765, 0.2222], 'bias_within': [0.1111, -0.2222, 0.3333], 'bias': [0.3456, -0.6789, 0.1234],
        'Derecha': [1.2345, -2.3456, 0.6789], 'Izquierda': [-1.1111, 2.2222, -0.6789],
    }, index=pd.MultiIndex.from_tuples([('2023-23J', '40dB', 8), ('2023-23J', 'CIS', 13), ('2023-23J', 'GAD3', 7)],
                                       names=['event', 'pollster', 'days']))
    error_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-07-15', '2023-07-10', '2023-07-16', '2023-07-15']),
        'pollster': ['40dB', 'CIS', 'GAD3', '40dB'], 'party': ['VOX', 'PSOE', 'PP', 'PP'], 'regional': [0, 0, 0, 0],
        'weeks': [1, 2, 1, 1], 'weight': [0.81, 0.57, 1.23, 0.81], 'color': [COLORS[n] for n in ('VOX', 'PSOE', 'PP', 'PP')],
        'pct': [12.3456, 30.9876, 34.5678, 33.3333], 'error': [0.1234, 0.8765, 1.4321, 0.2345],
    })
    return types.SimpleNamespace(house_effects=house_effects, herding_summary=herding_summary, herding=herding,
                                 poll_errors=poll_errors, error_data=error_data)


BACKTEST_EVENTS = ['2019-11-10', '2023-07-23']
BACKTEST_HORIZONS = [6, 30]


def synthetic_backtest():
    """Tablas de `backtest/results` con la forma de las reales: 2 elecciones x 2 horizontes (4 casos; los escaños no
    son válidos en `2019-11-10 x 30`), los partidos `NAMES`, los bloques `vs`, decimales de sobra y algún NaN.
    Devuelve `SimpleNamespace(run, metrics, by_horizon, blocks, shares, seats, calibration)`; `run` es el `run.json`
    del repositorio con `events` y `horizons` sintéticos."""
    run = {'run_at': '2026-10-07T13:35:53', 'commit': 'a59059f', 'scope': 'es', 'n_sim': 500, 'seed': 42, 'max_fc': 10,
           'nowcast_only': False, 'house_effects': True, 'industry_bias': False, 'dispersion': True, 'herding': True,
           'composition': None, 'regional_noise': True, 'min_events': None, 'events': list(BACKTEST_EVENTS),
           'horizons': list(BACKTEST_HORIZONS), 'db_polls': 3836, 'db_last_poll': '2026-10-03',
           'note': 'Run per event in parallel (one process per election, identical parameters) and merged.'}
    cases = [(e, h) for e in BACKTEST_EVENTS for h in BACKTEST_HORIZONS]
    valid = [not (e == '2019-11-10' and h == 30) for e, h in cases]
    metrics = pd.DataFrame({
        'event_date': [e for e, _ in cases], 'horizon': [h for _, h in cases],
        'mae_shares': [1.97049, 2.50981, 1.40877, 1.50234], 'rmse_shares': [2.52987, 3.23107, 1.9, 2.1],
        'mae_last_poll': [1.7368, 2.3012, 1.6125, 2.0143], 'mae_mean_4w': [1.8576, 2.2754, 1.5871, 2.1096],
        'mae_prev_result': [5.7712, 5.9031, 5.1204, 5.6388],
        'cov_shares95': [0.9333, 1., 0.9, 1.], 'cov_seats95': [np.nan, np.nan, 1., 0.9333],
        'mae_seats': [np.nan, np.nan, 4.5, 5.25], 'brier_vs': [0.03661, 0.04123, 0.03, 0.0501],
        'log_score_vs': [0.15605, 0.17001, 0.1, np.nan], 'n_prov_cells': [277.5, 276.75, 278., 277.],
        'seats_valid': valid,
    })
    by_horizon = pd.DataFrame({
        'horizon': BACKTEST_HORIZONS, 'n_cases': [2, 2], 'n_seats_cases': [2, 1],
        'mae_shares': [1.6896342173571752, 2.0063], 'mae_last_poll': [1.7368, 2.2471],
        'mae_mean_4w': [1.8576, 2.2322], 'mae_prev_result': [5.7712, 5.7712], 'cov_shares95': [0.9566666666, 1.],
        'cov_seats95': [1., 0.9333], 'mae_seats': [4.875, 5.25], 'brier_vs': [0.036605, 0.0457], 'log_score_vs': [0.1275, np.nan],
    })
    blocks = pd.DataFrame([
        {'event_date': e, 'horizon': h, 'block': block, 'parties': '+'.join(parties), 'official_seats': official,
         'model_seats': model, 'p_majority': p, 'majority': official >= 176, 'brier': brier, 'log_score': log,
         'seats_valid': ok, 'lo95': model - 11.0, 'hi95': model + 11.0, 'q05': model - 9.0, 'q50': model,
         'q95': model + 9.0, 'p_majority_h': p, 'q50_h': model}
        for (e, h), ok in zip(cases, valid)
        for block, parties, official, model, p, brier, log in (
            ('Derecha', ['PP', 'VOX'], 169, 171.0, 0.38412, 0.38412 ** 2, 0.4813),
            ('Izquierda', ['PSOE'], 121, 118.0, np.nan if h == 30 else 0.00098, 0.0, 0.0010005))
    ])
    shares = pd.DataFrame([
        {'event_date': e, 'horizon': h, 'party': party, 'main': True, 'official': official, 'model': model,
         'fc_lo': model - 1.12345, 'fc_hi': model + 1.12345, 'last_poll': model + 0.2468, 'lo95': model - 3.14159,
         'hi95': model + 3.14159}
        for e, h in cases
        for party, official, model in (('PP', 33.05, 32.41234), ('PSOE', 31.7, 30.98765), ('VOX', 12.39, 13.00501))
    ])
    rows = []
    for (e, h), ok in zip(cases, valid):
        for party, official, model, mean in (('PSOE', 120, 118, 118.432), ('PP', 137, 140, 139.5678), ('VOX', 33, 30, 30.0049)):
            rows.append({'event_date': e, 'horizon': h, 'party': party, 'main': True, 'official': official,
                         'model': model, 'mean': mean, 'crps': 2.34567, 'seats_valid': ok, 'lo50': mean - 3.1,
                         'hi50': mean + 3.1, 'lo95': mean - 9.5, 'hi95': mean + 9.5})
    seats = pd.DataFrame(rows)
    seats.loc[seats.index[-1], 'lo50'] = np.nan
    # the cells of the PIT (`calibration.csv`): national shares in both modes, seats only with valid seats, the
    # provincial shares of the nowcast; a PIT of more decimals and a cell without result (NaN)
    cells = []
    for (e, h), ok in zip(cases, valid):
        for party, official, model, pit in (('PP', 33.05, 32.41234, 0.61234), ('PSOE', 31.7, 30.98765, 0.27891)):
            for mode in ('now', 'h'):
                cells.append({'scope': 'es', 'event_date': e, 'horizon': h, 'kind': 'shares', 'mode': mode,
                              'party': party, 'region_id': np.nan, 'main': True, 'official': official,
                              'model': model, 'pit': pit})
                if ok:
                    cells.append({'scope': 'es', 'event_date': e, 'horizon': h, 'kind': 'seats', 'mode': mode,
                                  'party': party, 'region_id': np.nan, 'main': True, 'official': 120.0,
                                  'model': 118.0, 'pit': 0.45})
        cells.append({'scope': 'es', 'event_date': e, 'horizon': h, 'kind': 'prov_shares', 'mode': 'now',
                      'party': 'PP', 'region_id': 28.0, 'main': True, 'official': 30.1, 'model': 29.87654,
                      'pit': np.nan if h == 30 else 0.05})
    calibration = pd.DataFrame(cells)
    return types.SimpleNamespace(run=run, metrics=metrics, by_horizon=by_horizon, blocks=blocks, shares=shares,
                                 seats=seats, calibration=calibration)


def write_backtest_source(root, scope='es', ban=False):
    """Escribe en `backtest.results_dir(root, scope)` los seis CSV de `synthetic_backtest` (`by_horizon.csv` y
    `calibration.csv` incluidos) y su `run.json`; devuelve esa carpeta. Con `ban`, también `ban.csv`: las filas `shares` de `2023-07-23` con horizonte 0."""
    from mtpy.lib import backtest
    folder = backtest.results_dir(root, scope)
    os.makedirs(folder, exist_ok=True)
    tables = synthetic_backtest()
    for part, name in publish.BACKTEST_FILES.items():
        getattr(tables, part.replace('-', '_')).to_csv(os.path.join(folder, name), index=False)
    if ban:
        shares = tables.shares
        shares.loc[shares['event_date'].astype(str) == '2023-07-23'].assign(horizon=0).to_csv(
            os.path.join(folder, publish.EVENT_BAN_FILE), index=False)
    with open(os.path.join(folder, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump({**tables.run, 'scope': scope}, fh)
    return folder


def synthetic_event():
    """Tablas que lee `publish_event` de la base: `events` (dos elecciones destacadas pasadas con resultado
    oficial según `has_results` y el ciclo en curso, no destacado y sin resultado), `results` (sólo `2023-07-23`, región 0: una fila por `NAMES` y `Cs` sin escaños, con `pct` de
    más decimales y 350 escaños) y `pollsters` (`id`/`name` de `HOUSES`: `CIS` 1, `GAD3` 2, `40dB` 3)."""
    events = pd.DataFrame({
        'date': ['2019-11-10', '2023-07-23', '2026-11-29'], 'scope': 'es',
        'name': ['Generales 2019 (noviembre)', 'Generales 2023', None], 'featured': [True, True, False],
        'has_results': [True, True, False],
    })
    results = pd.DataFrame({
        'date': '2023-07-23', 'scope': 'es', 'region_id': 0, 'region': 'España',
        'party_id': [1, 2, 3, 4], 'party': ['PP', 'PSOE', 'VOX', 'Cs'],
        'votes': [8160837, 7760970, 3033744, 18054], 'pct': [33.05612, 31.43912, 12.28781, 0.13], 'seats': [160, 150, 40, 0],
    })
    pollsters = pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['CIS', 'GAD3', '40dB', 'ElectoPanel']})
    return types.SimpleNamespace(events=events, results=results, pollsters=pollsters)


def synthetic_cycle():
    """Sondeos del ciclo de `2023-07-23` con la forma de `Computer.filter_polls(featured=True)`: los tres de
    `poll_errors` (40dB a 8 días, CIS a 13 sin muestra conocida, GAD3 a 7), indexados por las claves de la serie,
    con `pollster` categórico como en la real y los pesos por separado (`weight_over`, `weight_sample`)."""
    frame = pd.DataFrame({
        'event_date': pd.Timestamp('2023-07-23'), 'date': pd.to_datetime(['2023-07-15', '2023-07-10', '2023-07-16']),
        'pollster_id': [3, 1, 2], 'sponsor_id': [5, 0, 0],
        'start_date': pd.to_datetime(['2023-07-10', '2023-07-03', '2023-07-12']),
        'end_date': pd.to_datetime(['2023-07-14', '2023-07-08', '2023-07-15']),
        'pollster': pd.Categorical(['40dB', 'CIS', 'GAD3']), 'sponsor': ['El País', None, None],
        'mtype': ['online', 'cati', 'cati'], 'ctype': [None, None, None], 'computed': True, 'featured': True,
        'sample_size': [1200, 0, 3000], 'parties': 3, 'days': [8, 13, 7], 'proc_sample': [1200, 1000, 3000],
        'rating': [0.7, 0.8, 0.9], 'weight_sample': [0.9, 0.7, 1.1], 'weight_over': [0.90260, 0.81127, 1.12233],
        'PP': [33.5, 31.2, 34.1], 'PSOE': [29.9, 32.4, 30.3], 'VOX': [13.1, 14.0, np.nan],
    })
    return frame.set_index(['event_date', 'date', 'pollster_id', 'sponsor_id']).sort_index()


def _ban_poll(date, pollster_id, sponsor_id, pollster, sponsor, mtype, days, sample_size, shares, **errors):
    """One row with the index and columns of `synthetic_cycle()` and the stored errors."""
    row = {'event_date': pd.Timestamp('2023-07-23'), 'date': pd.Timestamp(date), 'pollster_id': pollster_id,
           'sponsor_id': sponsor_id, 'start_date': pd.Timestamp(date) - pd.Timedelta(days=3), 'end_date': pd.Timestamp(date),
           'pollster': pollster, 'sponsor': sponsor, 'mtype': mtype, 'ctype': 'wban', 'computed': True, 'featured': True,
           'sample_size': sample_size, 'parties': 3, 'days': days, 'proc_sample': sample_size, 'rating': 0.8,
           'weight_sample': 1.0, 'weight_over': 1.0, **dict(zip(NAMES, shares)), **errors}
    return row


def synthetic_ban():
    """Sondeos de la semana de veto de `2023-07-23` (`ctype 'wban'`) con los errores guardados: GAD3 a 1 y 2 días y
    ElectoPanel (id 4, online, patrocinador Electomanía) a 2, indexados como `synthetic_cycle()`."""
    frame = pd.DataFrame([
        _ban_poll('2023-07-22', 2, 0, 'GAD3', None, 'cati', 1, 10004, [35.5, 28.6, 12.7],
                  error_avg=2.1, error_blocks=6.96, bias_blocks=15.16),
        _ban_poll('2023-07-21', 2, 0, 'GAD3', None, 'cati', 2, 3000, [34.8, 29.0, 12.8],
                  error_avg=2.3, error_blocks=6.2, bias_blocks=12.0, weight_over=0.0),
        _ban_poll('2023-07-21', 4, 30, 'ElectoPanel', 'Electomanía', 'online', 2, 1250, [34.0, 29.5, 13.0],
                  error_avg=1.6, error_blocks=3.1, bias_blocks=-2.0),
    ])
    return frame.set_index(['event_date', 'date', 'pollster_id', 'sponsor_id']).sort_index()


def synthetic_legal():
    """Sondeos legales de `2023-07-23` con paneles online incluidos: `synthetic_cycle()` y un ElectoPanel a 9 días."""
    panel = pd.DataFrame([_ban_poll('2023-07-14', 4, 30, 'ElectoPanel', 'Electomanía', 'online', 9, 1250,
                                    [33.0, 30.0, 13.5], error_avg=1.4, error_blocks=3.0, bias_blocks=-1.0)])
    panel['ctype'] = None
    panel = panel.set_index(['event_date', 'date', 'pollster_id', 'sponsor_id'])
    cycle = synthetic_cycle()
    cycle['pollster'] = cycle['pollster'].astype(object)
    return pd.concat([cycle, panel]).sort_index()


RATING_HOUSES = {'CIS': (1, .8), 'GAD3': (2, .6), '40dB': (3, .4)}  # casa -> (pollster_id, rating)
RATING_ELECTIONS = [('es', '2019-11-10', 'generales'), ('es-md', '2021-05-04', 'autonomicas')]


def synthetic_rating_check():
    """Lo que devuelve `Computer.rating_check_data`: pares casa-elección y sondeos de la última campaña de tres casas
    en dos elecciones (una general y una autonómica), con los diferenciales en LOR y los ratings en 0-1; el
    diferencial crece al bajar el rating y `40dB` no tiene rating previo (`d_prev` NaN)."""
    pairs, polls = [], []
    for e, (scope, date, population) in enumerate(RATING_ELECTIONS):
        for name, (pid, rating) in RATING_HOUSES.items():
            pairs.append({
                'event_scope': scope, 'event_date': date, 'population': population, 'pollster_id': pid,
                'pollster': name, 'd_obs': (pid - 2) * .01 + e * .001, 'error_avg': 1. + pid + e, 'n_polls': 4,
                'n_event_polls': 12, 'rating': rating, 'rating_adj': rating - .01,
                'd_prev': np.nan if name == '40dB' else (pid - 2) * .012, 'num_events_w': 2. + e,
                'quality': rating - .05, 'weight_rating': rating,
            })
            for party, error in (('X', pid - 2. + e), ('Y', 2. - pid)):
                polls.append({'event_scope': scope, 'event_date': date, 'population': population, 'pollster_id': pid,
                              'pollster': name, 'party': party, 'error': error, 'weight_over': 1.,
                              'weight_sample': float(pid), 'weight_rating': rating})
    return {'pairs': pd.DataFrame(pairs), 'polls': pd.DataFrame(polls), 'n_scopes': 2, 'seconds': 0.1}


CYCLE_SEATS = {('2023-07-16', 2, 0): {'PP': [135, 140], 'PSOE': [120, 120]}}


class FakeComputer:
    """Fábrica con la firma `Computer(scope=..., path=...)`: devuelve un Computer sintético con `scope`,
    `event_dates`, `event_params` (bloques `vs` y partidos `main` del ciclo, como los lee el real) y `names`,
    `build_series` (devuelve el propio objeto), `filter_polls` (`synthetic_cycle`) y los cuatro resúmenes de
    `synthetic_analysis`, `rating_check_data` (`synthetic_rating_check`) con `bias_dev_tau` y `bias_dev_sigma`; anota
    las llamadas y, con `fail`, lanza esa excepción al construir; con `fail_effects`, la lanza `print_house_effects`
    (como el real cuando el evento en curso no tiene sondeos); con `fail_rating_check`, la lanza `rating_check_data`
    (como un corte de la conexión con la base en el paso más lento)."""

    def __init__(self, fail=None, fail_effects=None, fail_rating_check=None):
        self.calls, self.fail, self.fail_effects, self.fail_rating_check = [], fail, fail_effects, fail_rating_check

    def __call__(self, scope, path=None, **kwargs):
        if self.fail is not None:
            raise self.fail
        self.calls.append(('init', scope, path, kwargs))
        frames = synthetic_analysis()
        bmaps = {'main': list(BMAPS['main']), 'blocks': dict(BMAPS['blocks']), 'vs': dict(BMAPS['vs'])}
        comp = types.SimpleNamespace(scope=scope, path=path, event_dates=list(ANALYSIS_EVENTS), herding=None,
                                     event_params={dt: {'bmaps': bmaps} for dt in ANALYSIS_EVENTS}, names=list(NAMES),
                                     series=pd.concat([synthetic_legal(), synthetic_ban()]))

        def load_herding():
            comp.herding = frames.herding.copy()
            return comp.herding

        def herding_summary(**kw):
            if comp.herding is None:
                load_herding()
            return frames.herding_summary.copy()

        def get_polls_metric(metric='pct', bmap=None, **kw):
            self.calls.append(('get_polls_metric', metric, bmap, kw))
            errors = frames.poll_errors.copy()  # `event` and `pollster` are categorical in the real frame
            errors.index = errors.index.set_levels(
                [pd.CategoricalIndex(lvl, categories=list(lvl) + (['Sin sondeos'] if lvl.name == 'pollster' else []), name=lvl.name)
                 if lvl.name in ('event', 'pollster') else lvl for lvl in errors.index.levels])
            return errors

        def filter_polls(featured=True, **kw):
            self.calls.append(('filter_polls', featured, kw))
            return synthetic_legal() if kw.get('drop_mtypes') == [] else synthetic_cycle()

        comp.build_series = lambda: comp
        comp.load_herding, comp.herding_summary, comp.get_polls_metric = load_herding, herding_summary, get_polls_metric
        comp.filter_polls = filter_polls
        def print_house_effects(**kw):
            if self.fail_effects is not None:
                raise self.fail_effects
            return frames.house_effects.copy()

        def rating_check_data(scopes='all', **kw):
            self.calls.append(('rating_check_data', scopes, kw))
            if self.fail_rating_check is not None:
                raise self.fail_rating_check
            return synthetic_rating_check()

        comp.print_house_effects = print_house_effects
        comp.get_error_estimator_data = lambda **kw: frames.error_data.copy()
        comp.rating_check_data, comp.bias_dev_tau, comp.bias_dev_sigma = rating_check_data, .01, .014
        return comp


class RecordingFileSystem(FileSystem):
    """FileSystem local que anota el orden de las escrituras."""

    def __init__(self, path):
        super().__init__(path)
        self.written = []

    def write_bytes(self, content, name):
        self.written.append(name)
        return super().write_bytes(content, name)


LEAN_RUN = '20261024-120000'
LEAN_BLOCKS = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'SUMAR']}
LEAN_ERRORS = {  # (ámbito, fecha) -> {casa: (error Izquierda, error Derecha)}
    ('es', '2008-03-09'): {'CIS': (-1.6, -1.2)},
    ('es', '2011-11-20'): {'CIS': (0.4, 2.0)},
    ('es', '2015-12-20'): {'CIS': (-6.3, 5.0)},
    ('es', '2016-06-26'): {'CIS': (3.0, -2.3), 'GAD3': (1.0, -1.0)},
    ('es', '2019-04-28'): {'CIS': (0.1, -0.1), 'GAD3': (-1.0, 1.0), 'Sigma Dos': (2.0, -1.0)},
    ('es', '2019-11-10'): {'CIS': (6.4, -6.1), 'GAD3': (-1.0, 1.0), 'Sigma Dos': (2.0, -1.0)},
    ('es', '2023-07-23'): {'CIS': (3.1, -2.8), 'GAD3': (-1.0, 1.0), 'Sigma Dos': (2.0, -1.0),
                           'NC Report': (-4.0, 3.0), 'Hamalgama': (-4.4, 4.7)},
    ('es-an', '2018-12-02'): {'CIS': (12.6, -9.5), 'GAD3': (-1.0, 1.0), 'Sigma Dos': (2.0, -1.0)},
    ('es-an', '2022-06-19'): {'CIS': (3.1, -4.9), 'GAD3': (-1.0, 1.0), 'Sigma Dos': (2.0, -1.0)},
    ('es-an', '2026-05-17'): {'CIS': (-0.5, -1.5), 'GAD3': (-1.0, 1.0), 'Sigma Dos': (2.0, -1.0),
                              'NC Report': (-4.0, 3.0)},
}


def bundle_fixture(name):
    """`data` de un fixture del paquete (`tests/fixtures/bundle/{name}.json`)."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    with open(os.path.join(root, 'tests', 'fixtures', 'bundle', name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)['data']


def write_lean_bundle(fs):
    """
    Paquete sintético con las entradas del estudio del sesgo del CIS (sin base de datos).

    Escribe, para `es` y `es-an`, la `meta` y el `poll-errors` del análisis (errores de `LEAN_ERRORS`, `days` 10,
    muestra 1000 y peso 1, más una fila del CIS de `es` en `2012-11-25` sin carpeta de evento) y una carpeta de
    evento (`meta.json`) por fecha; el evento `es/2023-07-23` completo (`meta` con bloques, `polls` con las cinco
    casas a 11, 6, 6, 7 y 10 días más Ipsos a 90, y `results`); y el manifest con el run nacional, cuyo `polls` tiene
    sondeos de 2026 del CIS, GAD3, Sigma Dos y NC Report y de Hamalgama sólo en 2025.

    Parameters
    ----------
    fs : FileSystem
        Raíz del paquete de prueba.

    Returns
    -------
    dict
        Las cifras que comprueban los tests: `cis_es_dates`, `industry`, `election` (lo que devuelve
        `read_election` para `es`/`2023-07-23`), `summary_election` (distancias a 2 decimales) y `houses`.
    """
    from mtpy.lib import bundle

    writer = bundle.BundleWriter(fs)
    days = {'CIS': 11, 'GAD3': 6, 'Sigma Dos': 6, 'NC Report': 7, 'Hamalgama': 10}

    def error_row(date, house, errors, **extra):
        left, right = errors
        return {'event_date': date, 'event': date, 'pollster': house, 'days': 10, 'sample_size': 1000, 'weight': 1,
                'Derecha': right, 'Izquierda': left} | extra

    for scope in ('es', 'es-an'):
        dates = sorted(date for code, date in LEAN_ERRORS if code == scope)
        rows = [error_row(date, house, errors) for date in dates for house, errors in LEAN_ERRORS[scope, date].items()]
        if scope == 'es':
            rows.append(error_row('2012-11-25', 'CIS', (1.0, -1.0)))
        writer.write_json(bundle.path_analysis(scope, 'poll-errors'), 'poll-errors',
                          {'blocks': ['Derecha', 'Izquierda'], 'rows': rows}, scope)
        writer.write_json(bundle.path_analysis(scope, 'meta'), 'analysis-meta',
                          bundle_fixture('analysis-meta') | {'scope': scope, 'events': dates}, scope)
        for date in dates:
            meta = bundle_fixture('event-meta') | {'scope': scope, 'event_date': date, 'name': f'{scope} {date}'}
            writer.write_json(bundle.path_event(scope, date, 'meta'), 'event-meta', meta, scope)

    meta = bundle_fixture('event-meta') | {'scope': 'es', 'event_date': '2023-07-23', 'name': 'Generales 2023',
                                           'blocks': LEAN_BLOCKS}
    polls = {'blocks': ['Derecha', 'Izquierda'], 'rows': [
        error_row('2023-07-23', house, LEAN_ERRORS['es', '2023-07-23'][house], days=days[house]) for house in days
    ] + [error_row('2023-07-23', 'Ipsos', (-2.9, 2.5), days=90)]}
    results = {'n_seats': 350, 'rows': [
        {'party_id': index, 'party': party, 'votes': 1000 * (index + 1), 'pct': pct, 'seats': 100}
        for index, (party, pct) in enumerate([('PP', 33.06), ('PSOE', 31.68), ('VOX', 12.38), ('SUMAR', 12.33)])
    ]}
    for part, schema, content in (('meta', 'event-meta', meta), ('polls', 'event-polls', polls),
                                  ('results', 'event-results', results)):
        writer.write_json(bundle.path_event('es', '2023-07-23', part), schema, content, 'es')

    run_polls = [{'date': date, 'pollster_id': index, 'pollster': house, 'sample': 1000, 'PP': 33.0}
                 for index, (house, date) in enumerate([
                     ('CIS', '2026-10-05'), ('GAD3', '2026-09-20'), ('Sigma Dos', '2026-10-01'),
                     ('NC Report', '2026-03-15'), ('Hamalgama', '2025-11-10'), ('GAD3', '2025-06-01')])]
    writer.write_json(bundle.path_part('es', LEAN_RUN, 'polls'), 'polls',
                      bundle_fixture('polls') | {'polls': run_polls}, 'es', run_id=LEAN_RUN)
    publish.update_manifest(writer, {'es': publish.latest_entry(bundle_fixture('headline') | {'run_id': LEAN_RUN})})

    return {
        'cis_es_dates': ['2008-03-09', '2011-11-20', '2015-12-20', '2016-06-26', '2019-04-28', '2019-11-10',
                         '2023-07-23'],
        'industry': {'n_events': 2, 'left': 0, 'right': 2, 'mean': -1.43},
        # el `mean` es el round(-1.425, 2) de Python (-1.43): es 2023 -1.85 y es-an 2026 -1.00
        'election': {'meta': meta, 'polls': polls, 'results': results},
        'summary_election': {'cis_dist': 4.18, 'consensus_dist': 2.67, 'closer': 2, 'rank': 3, 'n': 5},
        'houses': ['GAD3', 'Sigma Dos', 'CIS'],
    }


HERD_RUN = '20261105-120000'
HERD_POLLS = {  # (ámbito, elección) -> [(casa, días antes de la elección, PP, PSOE, mtype)]
    ('es', '2019-11-10'): [
        ('A', 200, 30, 25), ('B', 198, 34, 25), ('A', 100, 30, 25), ('B', 97, 34, 25),
        ('A', 50, 30, 25), ('B', 48, 33, 25), ('A', 40, 30, 25), ('B', 38, 33, 25),
        ('A', 15, 31, 25), ('B', 13, 32, 25), ('A', 4, 31, 25), ('B', 2, 32, 25),
        ('CIS', 14, 20, 40), ('Agg', 12, 50, 10, 'aggr'),
    ],
    ('es', '2023-07-23'): [
        ('A', 150, 30, 25), ('B', 148, 31, 25), ('A', 120, 30, 25), ('B', 118, 31, 25),
        ('A', 10, 30, 25), ('B', 8, 33, 25), ('A', 3, 30, 25), ('B', 1, 33, 25),
    ],
    ('es-md', '2021-05-04'): [('A', 16, 40, 20), ('D', 14, 41, 20), ('A', 6, 40, 20), ('D', 4, 41, 20)],
    ('es-md', '2023-05-28'): [('A', 10, 40, 20)],
    ('es', '2026-11-29'): [
        ('A', 100, 30, 25), ('B', 98, 34, 25), ('A', 90, 30, 25), ('B', 88, 34, 25),
        ('A', 40, 30, 25), ('B', 38, 32, 25), ('A', 30, 30, 25), ('B', 28, 32, 25),
    ],
}
HERD_IDS = {'A': 1, 'B': 2, 'D': 3, 'CIS': 4, 'Agg': 5}


def write_herd_bundle(fs):
    """
    Paquete sintético con las entradas del estudio del rebaño (sin base de datos).

    Escribe, para cada elección pasada de `HERD_POLLS` (`es/2019-11-10`, `es/2023-07-23`, `es-md/2021-05-04` y
    `es-md/2023-05-28`), su `meta` y su `cycle`, y los `results` de todas salvo `es-md/2023-05-28` (un ciclo sin
    escrutinio). Escribe también el run en curso `HERD_RUN` de `es` (elección `2026-11-29`, `as_of` 2026-11-05, con
    `meta` y `polls`) y el manifiesto con `es` y `es-md`. Todos los sondeos llevan `sample_size` 1000, `weight` 1,
    `start_date = end_date = date` (la elección menos los días de la tupla), `mtype` nulo salvo que la tupla lo diga
    y los partidos `PP` y `PSOE`.

    Parameters
    ----------
    fs : FileSystem
        Raíz del paquete de prueba.

    Returns
    -------
    dict
        `events` (las elecciones pasadas escritas, `(ámbito, fecha)`) y `run_id` (el run en curso).
    """
    from mtpy.lib import bundle
    from mtpy.lib.publish import POLL_COLUMNS

    writer = bundle.BundleWriter(fs)
    parties = ['PP', 'PSOE']

    def poll(event, house, days, pp, psoe, mtype=None):
        end = (pd.Timestamp(event) - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

        return {'date': end, 'pollster_id': HERD_IDS[house], 'pollster': house, 'sponsor': None, 'start_date': end,
                'end_date': end, 'sample_size': 1000, 'mtype': mtype, 'rating': None, 'weight': 1,
                'PP': float(pp), 'PSOE': float(psoe)}

    events = []
    for (scope, date), rows in HERD_POLLS.items():
        if date == '2026-11-29':
            continue
        events.append((scope, date))
        meta = bundle_fixture('event-meta') | {'scope': scope, 'event_date': date, 'name': f'{scope} {date}'}
        cycle = {'parties': parties, 'columns': list(POLL_COLUMNS), 'polls': [poll(date, *row) for row in rows]}
        parts = [('meta', 'event-meta', meta), ('cycle', 'event-cycle', cycle)]
        if (scope, date) != ('es-md', '2023-05-28'):
            parts.append(('results', 'event-results', {'n_seats': 350, 'rows': []}))
        for part, schema, content in parts:
            writer.write_json(bundle.path_event(scope, date, part), schema, content, scope)

    run_meta = bundle_fixture('meta') | {'scope': 'es', 'event_date': '2026-11-29', 'as_of': '2026-11-05',
                                         'run_id': HERD_RUN}
    run_polls = {'parties': parties, 'columns': list(POLL_COLUMNS), 'results': [],
                 'polls': [poll('2026-11-29', *row) for row in HERD_POLLS[('es', '2026-11-29')]]}
    for part, content in (('meta', run_meta), ('polls', run_polls)):
        writer.write_json(bundle.path_part('es', HERD_RUN, part), part, content, 'es', run_id=HERD_RUN)
    headline = bundle_fixture('headline') | {'run_id': HERD_RUN, 'event_date': '2026-11-29', 'as_of': '2026-11-05'}
    publish.update_manifest(writer, {'es': publish.latest_entry(headline),
                                     'es-md': publish.latest_entry(headline | {'run_id': 'unused'})})

    return {'events': events, 'run_id': HERD_RUN}


SHY_RUN = '20261105-120000'
SHY_BLOCKS = {
    ('es', '1989-10-29'): {'Izquierda': ['PSOE', 'IU'], 'Derecha': ['AP', 'CDS']},
    ('es', '2023-07-23'): {'Izquierda': ['PSOE', 'SUMAR'], 'Derecha': ['PP', 'VOX']},
    ('es', '2016-06-26'): {'Izquierda': ['PSOE'], 'Derecha': ['PP']},
    ('es-md', '2015-05-24'): {'Izquierda': ['PSOE', 'UP', 'IU'], 'Derecha': ['PP', 'Cs']},
    ('es-md', '2019-05-26'): {'Izquierda': ['PSOE', 'UP'], 'Derecha': ['PP', 'Cs']},
    ('es-md', '2021-05-04'): {'Izquierda': ['PSOE', 'MM', 'UP'], 'Derecha': ['PP', 'VOX']},
}
SHY_RESULTS = {
    ('es', '1989-10-29'): {'AP': 25.8, 'PSOE': 39.6, 'IU': 9.1, 'CDS': 7.9},
    ('es', '2023-07-23'): {'PP': 33.1, 'PSOE': 31.7, 'VOX': 12.4, 'SUMAR': 12.3},
    ('es', '2016-06-26'): {'PP': 33.0, 'PSOE': 22.6},
    ('es-md', '2015-05-24'): {'PP': 33.1, 'PSOE': 25.4, 'UP': 18.6, 'Cs': 12.1, 'IU': 4.2},
    ('es-md', '2019-05-26'): {'PP': 22.2, 'PSOE': 27.3, 'UP': 14.0, 'Cs': 19.4},
    ('es-md', '2021-05-04'): {'PP': 44.7, 'MM': 16.9, 'PSOE': 16.8, 'VOX': 9.1, 'UP': 7.2},
}
# (fecha, cifras por partido, peso) por elección
SHY_POLLS = {
    ('es', '1989-10-29'): (['AP', 'PSOE', 'IU', 'CDS'], [
        ('1989-10-20', (24, 42, 8, 9), 1), ('1989-10-25', (23, 44, 10, None), 1),
        ('1989-09-30', (30, 40, 9, 8), 1), ('1989-10-27', (20, 50, 10, 5), 0)]),
    ('es', '2023-07-23'): (['PP', 'PSOE', 'VOX', 'SUMAR'], [
        ('2023-07-10', (35, 27, 14, 13), 1), ('2023-07-16', (34, 28, 13, 12), 1),
        ('2023-07-19', (33, 29, 13, 13), 1)]),
    ('es-md', '2015-05-24'): (['PP', 'PSOE', 'UP', 'Cs', 'IU'], [
        ('2015-05-15', (34, 24, 20, 13, 4), 1), ('2015-05-20', (35, 25, 18, 12, None), 1)]),
    ('es-md', '2019-05-26'): (['PP', 'PSOE', 'UP', 'Cs'], [('2019-04-30', (25, 26, 14, 18), 1)]),
    ('es-md', '2021-05-04'): (['PP', 'PSOE', 'MM', 'VOX', 'UP'], [
        ('2021-04-25', (43, 20, 14, 10, 6), 1), ('2021-05-02', (42, 20, 16, 10, 6), 1)]),
}
SHY_CURRENT = (['PP', 'PSOE', 'VOX', 'FA', 'UP'], [
    ('2026-10-20', (31, 27, 19, 7, 3), 1), ('2026-10-25', (32, 26, 20, 6, 3), 1),
    ('2026-11-03', (30, 28, 18, 8, None), 1)])


def write_shy_bundle(fs):
    """
    Paquete sintético con las entradas del estudio del voto oculto (sin base de datos).

    Escribe, bajo `fs`, los eventos de `SHY_RESULTS` (`meta` y `results`; el `cycle` de los de `SHY_POLLS`, con todos
    los sondeos con `sample_size` 1000 y `start_date = end_date = date`; `es/2016-06-26` sin `cycle`), el manifest con
    `SHY_RUN` como `latest` de `es` y su `meta` (elección del 2026-11-29, `as_of` 2026-11-05) y `polls`; y una
    carpeta de datos `shy-data/` con los parámetros (linajes cortos), la tabla del Gobierno y el catálogo de ámbitos.

    Parameters
    ----------
    fs : FileSystem
        Raíz del paquete de prueba.

    Returns
    -------
    dict
        `dates` (las elecciones escritas por ámbito), `run_id` y `data` (el `FileSystem` de la carpeta de datos).
    """
    from mtpy.lib import bundle
    from mtpy.lib.publish import POLL_COLUMNS

    writer = bundle.BundleWriter(fs)

    def records(names, rows):
        out = []
        for date, figures, weight in rows:
            record = {'date': date, 'pollster_id': 1, 'pollster': 'GAD3', 'sponsor': None, 'start_date': date,
                      'end_date': date, 'sample_size': 1000, 'mtype': None, 'rating': None, 'weight': weight}
            out.append(record | {name: (None if figure is None else float(figure))
                                 for name, figure in zip(names, figures)})
        return out

    dates = {}
    for (scope, date), blocks in SHY_BLOCKS.items():
        dates.setdefault(scope, []).append(date)
        name = 'Generales' if scope == 'es' else 'Madrid'
        meta = bundle_fixture('event-meta') | {'scope': scope, 'event_date': date, 'name': name, 'blocks': blocks}
        rows = [{'party_id': i, 'party': party, 'votes': 1000, 'pct': pct, 'seats': 1}
                for i, (party, pct) in enumerate(SHY_RESULTS[(scope, date)].items(), 1)]
        parts = [('meta', 'event-meta', meta), ('results', 'event-results', {'n_seats': 350, 'rows': rows})]
        if (scope, date) in SHY_POLLS:
            names, polls = SHY_POLLS[(scope, date)]
            cycle = {'parties': names, 'columns': list(POLL_COLUMNS), 'polls': records(names, polls)}
            parts.append(('cycle', 'event-cycle', cycle))
        for part, schema, content in parts:
            writer.write_json(bundle.path_event(scope, date, part), schema, content, scope)

    names, polls = SHY_CURRENT
    bmaps = {'blocks': {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP', 'VOX']},
             'main': ['PP', 'PSOE', 'VOX', 'FA']}
    run_meta = bundle_fixture('meta') | {'scope': 'es', 'event_date': '2026-11-29', 'as_of': '2026-11-05',
                                         'run_id': SHY_RUN, 'bmaps': bmaps}
    run_polls = {'parties': names, 'columns': list(POLL_COLUMNS), 'polls': records(names, polls), 'results': []}
    for part, content in (('meta', run_meta), ('polls', run_polls)):
        writer.write_json(bundle.path_part('es', SHY_RUN, part), part, content, 'es', run_id=SHY_RUN)
    headline = bundle_fixture('headline') | {'run_id': SHY_RUN, 'event_date': '2026-11-29', 'as_of': '2026-11-05'}
    publish.update_manifest(writer, {'es': publish.latest_entry(headline)})

    data = FileSystem(fs.get_path('shy-data'))
    params = {'window_days': 14, 'recent_from': '2018-09-01', 'blocks': ['Izquierda', 'Derecha'],
              'lineages': {'PSOE': ['PSOE', 'PSC'], 'PP': ['PP', 'AP'], 'VOX': ['VOX'], 'UP': ['SUMAR', 'UP', 'IU']},
              'labels': {'UP': 'IU / Podemos / Sumar', 'PP': 'PP (AP, CD)', 'VOX': 'Vox'},
              'size_parties': ['PSOE', 'PP', 'VOX', 'UP'], 'size_threshold': 2, 'big_threshold': -3,
              'pp_early': ['1982-01-01', '1996-01-01'], 'extremes': 3}
    incumbents = {'es': {'1989-10-29': ['PSOE'], '2016-06-26': ['PP'], '2023-07-23': ['PSOE'],
                         '2026-11-29': ['PSOE']},
                  'es-md': {'2015-05-24': ['PP'], '2019-05-26': ['PP'], '2021-05-04': ['PP']}}
    os.makedirs(data.get_path('voto-oculto'), exist_ok=True)
    for name, content in (('params', params), ('incumbents', incumbents)):
        with open(data.get_path(f'voto-oculto/{name}.json'), 'w', encoding='utf-8') as fh:
            json.dump(content, fh, ensure_ascii=False)
    data.write_csv(pd.DataFrame({'scode': ['es', 'es-md', 'es-xx'], 'name': ['España', 'Madrid', 'Ámbito X']}),
                   'es-scopes.csv')

    return {'dates': {scope: sorted(found) for scope, found in dates.items()}, 'run_id': SHY_RUN, 'data': data}


BAN_RUN = '20261106-120000'
BAN_IDS = {'CIS': 1, 'GAD3': 2, '40dB': 3, 'ElectoPanel': 4, 'Sigma Dos': 5, 'YouGov': 9}
BAN_MTYPES = {'ElectoPanel': 'online', 'YouGov': 'aggr'}  # el resto, `cati`
BAN_EVENTS = {  # (ámbito, fecha) -> lo que escribe `write_ban_bundle`
    ('es', '2019-11-10'): {
        'name': 'Generales nov. 2019', 'main': ['PSOE', 'PP', 'VOX'], 'official': [28.0, 20.8, 15.1],
        'frozen': [27.4, 20.3, 13.9], 'ban_rows': [27.3, 19.9, 14.8],
        'polls': [  # (casa, fin de campo, días, muestra, error_avg, cifras)
            ('GAD3', '2019-11-09', 1, 13000, 0.86, [27.0, 20.1, 16.3]),
            ('GAD3', '2019-11-10', 0, 13000, 0.80, [27.3, 20.1, 16.0]),
            ('ElectoPanel', '2019-11-08', 2, 1250, 0.67, [27.1, 20.4, 15.5]),
        ],
        'legal': [('GAD3', 10, 3000, 0.77), ('ElectoPanel', 6, 1250, 0.61)],  # (casa, días, muestra, error_avg)
        'pollsters': [('GAD3', 10, 0.77), ('40dB', 12, 1.11)],  # (casa, last_days, last_error_avg)
    },
    ('es', '2023-07-23'): {
        'name': 'Generales 2023', 'main': ['PP', 'PSOE', 'VOX', 'SUMAR'], 'official': [33.1, 31.7, 12.4, 12.3],
        'frozen': [34.2, 28.3, 13.1, 13.1], 'ban_rows': None,
        'polls': [
            ('YouGov', '2023-07-18', 5, 1000, 1.5, [33.0, 29.0, 13.0, 13.0]),
            ('40dB', '2023-07-21', 2, 2000, 1.71, [31.7, 29.4, 13.8, 13.8]),
            ('GAD3', '2023-07-22', 1, 10004, 2.10, [35.5, 28.6, 12.7, 11.9]),
        ],
        'legal': [('40dB', 7, 2000, 1.45), ('GAD3', 6, 3000, 2.73), ('YouGov', 8, 1000, 1.6)],
        'pollsters': [('40dB', 7, 1.45), ('GAD3', 6, 2.73), ('CIS', 13, 1.9)],
    },
    ('es-md', '2021-05-04'): {
        'name': 'Elecciones a la Asamblea de Madrid 2021', 'main': ['PP', 'MM', 'PSOE'], 'official': [44.7, 17.0, 16.9], 'frozen': None,
        'ban_rows': None,
        'polls': [
            ('GAD3', '2021-05-03', 1, 8500, 0.97, [44.0, 17.5, 17.0]),
            ('Sigma Dos', '2021-05-03', 1, 1000, 1.20, [43.0, 18.0, 17.5]),
        ],
        'legal': [('GAD3', 60, 8500, 1.72), ('Sigma Dos', 8, 1000, 1.50)],
        'pollsters': [('GAD3', 60, 1.72), ('Sigma Dos', 8, 1.50)],
    },
}
BAN_CURRENT = '2026-11-29'  # la elección en curso: `meta` y `ban` sin `results` (la guarda legal)
BAN_CURRENT_SHARE = 31.4  # la cifra del PP del sondeo de GAD3 de la elección en curso, que el estudio no debe leer


def _ban_record(house, end, days, sample_size, error_avg, names, shares):
    """One poll of an `event-ban` part as `export_event_ban` writes it."""
    start = (pd.Timestamp(end) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    return {'date': end, 'pollster_id': BAN_IDS[house], 'pollster': house, 'sponsor': f'Medio de {house}',
            'start_date': start, 'end_date': end, 'sample_size': sample_size, 'mtype': BAN_MTYPES.get(house, 'cati'),
            'rating': 0.8, 'weight': 1.0, 'days': days, 'error_avg': error_avg, 'error_blocks': 2.0,
            'bias_blocks': 0.5} | dict(zip(names, shares))


def write_ban_bundle(fs):
    """
    Paquete sintético con las entradas del estudio de los sondeos «clandestinos» (sin base de datos).

    Escribe el manifest con `es` (run `BAN_RUN`, cuya `meta` lleva la elección del 29-11-2026) y `es-md`; los eventos
    de `BAN_EVENTS` (`meta`, `ban`, `pollsters`, `results` y, en los nacionales, `model` con las filas del horizonte
    6, otras del 30 y una de `Cs` fuera de los principales; en 2019, también las filas `ban`); y la elección en curso
    `es/2026-11-29` con `meta` y un `ban` de un sondeo de GAD3 pero sin `results`. Las cifras legales de cada casa
    son las del clandestino menos un punto.

    Parameters
    ----------
    fs : FileSystem
        Raíz del paquete de prueba.

    Returns
    -------
    dict
        Las cifras calculadas a mano que comprueban los tests: `pairs` (`(ámbito, fecha, casa) -> d`), `summary`,
        `consensus`, `cells` (`(fecha, partido) -> (closer, direction, crossed, half)`), `errors` (por fecha:
        `legal`, `frozen`, `consensus`, `with`) y `next`.
    """
    from mtpy.lib import bundle

    writer = bundle.BundleWriter(fs)
    columns = list(bundle_fixture('event-ban')['columns'])

    for (scope, date), event in BAN_EVENTS.items():
        names = event['main']
        meta = bundle_fixture('event-meta') | {'scope': scope, 'event_date': date, 'name': event['name'],
                                               'main': names, 'blocks': {'Todos': names}, 'featured': True}
        polls = [_ban_record(house, end, days, sample, error, names, shares)
                 for house, end, days, sample, error, shares in event['polls']]
        legal = [_ban_record(house, (pd.Timestamp(date) - pd.Timedelta(days=days)).strftime('%Y-%m-%d'), days,
                             sample, error, names, [share - 1 for share in event['polls'][0][5]])
                 for house, days, sample, error in event['legal']]
        pollsters = {'blocks': ['Todos'], 'rows': [
            {'pollster_id': BAN_IDS[house], 'pollster': house, 'n_polls': 5, 'sample_size': 1000, 'error_avg': 1.0,
             'error_blocks': 2.0, 'bias_blocks': 0.5, 'last_days': last_days, 'last_error_avg': last_error,
             'last_error_blocks': 2.0}
            for house, last_days, last_error in event['pollsters']]}
        results = {'n_seats': 350, 'rows': [
            {'party_id': index, 'party': party, 'votes': 1000 * (10 - index), 'pct': pct, 'seats': 10}
            for index, (party, pct) in enumerate(list(zip(names, event['official'])) + [('Cs', 4.2)])]}
        parts = [('meta', 'event-meta', meta),
                 ('ban', 'event-ban', {'parties': names, 'columns': columns, 'polls': polls, 'legal': legal}),
                 ('pollsters', 'event-pollsters', pollsters), ('results', 'event-results', results)]

        if event['frozen'] is not None:
            def rows(horizon, figures):
                return [{'event_date': date, 'horizon': horizon, 'party': party, 'main': True, 'official': official,
                         'model': model}
                        for party, official, model in zip(names, event['official'], figures)] + [
                    {'event_date': date, 'horizon': horizon, 'party': 'Cs', 'main': False, 'official': 4.2,
                     'model': 9.9}]
            model = {'horizons': [6, 30], 'seats': [], 'blocks': [],
                     'shares': rows(6, event['frozen']) + rows(30, [share + 5 for share in event['frozen']])}
            if event['ban_rows'] is not None:
                model['ban'] = rows(0, event['ban_rows'])
            parts.append(('model', 'event-model', model))

        for part, schema, content in parts:
            writer.write_json(bundle.path_event(scope, date, part), schema, content, scope)

    names = ['PP', 'PSOE', 'VOX', 'FA']
    meta = bundle_fixture('event-meta') | {'scope': 'es', 'event_date': BAN_CURRENT, 'name': 'Generales 2026',
                                           'main': names, 'blocks': {'Todos': names}, 'featured': True}
    ban = {'parties': names, 'columns': columns, 'legal': [],
           'polls': [_ban_record('GAD3', '2026-11-27', 2, 9000, None, names, [BAN_CURRENT_SHARE, 28.0, 15.0, 9.0])]}
    writer.write_json(bundle.path_event('es', BAN_CURRENT, 'meta'), 'event-meta', meta, 'es')
    writer.write_json(bundle.path_event('es', BAN_CURRENT, 'ban'), 'event-ban', ban, 'es')

    run_meta = bundle_fixture('meta') | {'scope': 'es', 'event_date': BAN_CURRENT, 'as_of': '2026-11-06',
                                         'run_id': BAN_RUN}
    writer.write_json(bundle.path_part('es', BAN_RUN, 'meta'), 'meta', run_meta, 'es', run_id=BAN_RUN)
    headline = bundle_fixture('headline') | {'run_id': BAN_RUN, 'event_date': BAN_CURRENT, 'as_of': '2026-11-06'}
    regional = bundle_fixture('headline') | {'run_id': '20261106-130000', 'event_date': '2027-05-23'}
    publish.update_manifest(writer, {'es-md': publish.latest_entry(regional), 'es': publish.latest_entry(headline)})

    return {
        'pairs': {('es', '2019-11-10', 'GAD3'): 0.03, ('es', '2019-11-10', 'ElectoPanel'): 0.06,
                  ('es', '2023-07-23', '40dB'): 0.26, ('es', '2023-07-23', 'GAD3'): -0.63,
                  ('es-md', '2021-05-04', 'Sigma Dos'): -0.30},
        'summary': {'pairs': {'n': 5, 'k': 2, 'ties': 0, 'share': 0.4, 'mean_d': -0.12, 'p': 0.812,
                              'generals': {'n': 4, 'k': 1, 'mean_d': -0.07, 'n_events': 2},
                              'regional': {'n': 1, 'k': 1, 'mean_d': -0.3, 'n_events': 1}},
                    'elections': {'n': 2, 'k_beaten': 1, 'mean_legal': 1.48, 'mean_frozen': 1.13,
                                  'mean_consensus': 0.96, 'mean_with': 0.63, 'k_with': 1},
                    'cells': {'n': 7, 'k_closer': 4, 'k_direction': 4, 'k_crossed': 1, 'k_half': 1},
                    'big': {'threshold': 5000, 'n': 2, 'k': 1, 'mean_d': -0.3}},
        'consensus': {'2019-11-10': {'PSOE': 27.3, 'PP': 20.1, 'VOX': 16.0},
                      '2023-07-23': {'PP': 33.6, 'PSOE': 29.0, 'VOX': 13.25, 'SUMAR': 12.85}},
        'cells': {  # (closer, direction, crossed, half)
            ('2019-11-10', 'PSOE'): (False, False, False, False), ('2019-11-10', 'PP'): (False, False, False, False),
            ('2019-11-10', 'VOX'): (True, True, True, False), ('2023-07-23', 'PP'): (True, True, False, True),
            ('2023-07-23', 'PSOE'): (True, True, False, False), ('2023-07-23', 'VOX'): (False, False, False, False),
            ('2023-07-23', 'SUMAR'): (True, True, False, False)},
        'errors': {'2019-11-10': {'legal': 0.94, 'frozen': 0.77, 'consensus': 0.77, 'with': 0.63},
                   '2023-07-23': {'legal': 2.03, 'frozen': 1.5, 'consensus': 1.15, 'with': None}},
        'next': {'event_date': BAN_CURRENT, 'open': False},
    }
