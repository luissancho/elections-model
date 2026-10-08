"""Dobles de prueba de la publicación: un Simulator sintético que ejercita sus métodos reales de salida sin base."""
import types

import numpy as np
import pandas as pd

from mtpy.core.io import FileSystem
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
    return types.SimpleNamespace(
        forecast=forecast, fc_stat=fc_stat, fc_series_raw=polls, nfc_series=results, house_effects=house_effects,
        dispersion=dispersion, date_last=pd.Timestamp(date_last), date_fit_last=pd.Timestamp(date_fit_last),
        bmaps=BMAPS, colors=COLORS, names=NAMES,
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
    sim.reg_totals = pd.DataFrame({'votes': [1000, 600, 400], 'seats': [10, 6, 4]}, index=pd.Index(REGIONS, name='region_id'))
    sim.params = {'n_sim': n_sim, 'split': True, 'random': True, 'names': NAMES, 'regions': REGIONS,
                  'horizon': None, 'date_prior': 'historical'}
    sim.cols_forecast = ['mean', 'regional', 'err', 'nobs', 'error']
    sim.cols_frame = sim.cols_forecast + ['pct', 'pct_err', 'drift', 'std_err', 'rand', 'vpred']
    sim.cols_unit = ['prev_pct', 'vpred_pct']
    sim.forecast = pd.DataFrame({'mean': [40., 30., 15., 15.], 'regional': [0, 0, 0, 0], 'err': [1.5, 1.2, 1.0, np.nan],
                                 'nobs': [40., 40., 30., np.nan], 'error': [2.0, 1.8, 1.5, np.nan]}, index=NAMES + ['-'])
    sim.as_of, sim.deadline, sim.horizon_max = pd.Timestamp('2026-10-05'), pd.Timestamp('2026-11-29'), 55
    sim.ages = pd.Series({'PP': 40., 'PSOE': 40., 'VOX': 12.}, name='age')
    sim.v2err = sim.v2drift = sim.v2seats = None
    sim.composition_ratio, sim.composition_rho = 1.0, 0.
    sim.house_effects = sim.dispersion = sim.regional_noise = True
    sim.industry_bias, sim.composition, sim.smap = False, None, {'UP': [{'agg': ['UP', 'MP']}]}
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
    `fit_forecast` y `run` inertes y anota las llamadas; con `fail`, lanza esa excepción al construir."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, scope, event_date, **kwargs):
        if self.fail is not None:
            raise self.fail
        sim = synthetic_simulator(mode=kwargs.get('mode', 'nowcast'))
        sim.scope, sim.event_date = scope, event_date
        self.calls.append(('init', scope, event_date, kwargs))

        def fit_forecast(**kw):
            self.calls.append(('fit_forecast', kw))
            return sim.forecast

        def run(**kw):
            self.calls.append(('run', sim.mode, kw))
            return sim

        sim.fit_forecast, sim.run = fit_forecast, run
        return sim


class RecordingFileSystem(FileSystem):
    """FileSystem local que anota el orden de las escrituras."""

    def __init__(self, path):
        super().__init__(path)
        self.written = []

    def write_bytes(self, content, name):
        self.written.append(name)
        return super().write_bytes(content, name)
