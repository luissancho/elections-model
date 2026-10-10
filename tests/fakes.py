"""Dobles de prueba de la publicación: un Simulator sintético que ejercita sus métodos reales de salida sin base
y un Computer sintético con los resúmenes de casas que publica `publish analysis`."""
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
        # 40dB sin historia (solo ciclo); CIS con dos elecciones; GAD3 con una y sin sondeos en el ciclo
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


class FakeComputer:
    """Fábrica con la firma `Computer(scope=..., path=...)`: devuelve un Computer sintético con `scope`,
    `event_dates` y `event_params` (bloques `vs` y partidos `main` del ciclo, como los lee el real),
    `build_series` (devuelve el propio objeto) y los cuatro resúmenes de `synthetic_analysis`; anota las
    llamadas y, con `fail`, lanza esa excepción al construir."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, scope, path=None, **kwargs):
        if self.fail is not None:
            raise self.fail
        self.calls.append(('init', scope, path, kwargs))
        frames = synthetic_analysis()
        bmaps = {'main': list(BMAPS['main']), 'blocks': dict(BMAPS['blocks']), 'vs': dict(BMAPS['vs'])}
        comp = types.SimpleNamespace(scope=scope, path=path, event_dates=list(ANALYSIS_EVENTS), herding=None,
                                     event_params={dt: {'bmaps': bmaps} for dt in ANALYSIS_EVENTS})

        def load_herding():
            comp.herding = frames.herding.copy()
            return comp.herding

        def herding_summary(**kw):
            if comp.herding is None:
                load_herding()
            return frames.herding_summary.copy()

        def get_polls_metric(metric='pct', bmap=None, **kw):
            self.calls.append(('get_polls_metric', metric, bmap, kw))
            return frames.poll_errors.copy()

        comp.build_series = lambda: comp
        comp.load_herding, comp.herding_summary, comp.get_polls_metric = load_herding, herding_summary, get_polls_metric
        comp.print_house_effects = lambda **kw: frames.house_effects.copy()
        comp.get_error_estimator_data = lambda **kw: frames.error_data.copy()
        return comp


class RecordingFileSystem(FileSystem):
    """FileSystem local que anota el orden de las escrituras."""

    def __init__(self, path):
        super().__init__(path)
        self.written = []

    def write_bytes(self, content, name):
        self.written.append(name)
        return super().write_bytes(content, name)
