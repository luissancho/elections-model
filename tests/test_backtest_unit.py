"""Tests de las métricas del backtest (sin base de datos)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.backtest import (_quantile_cols, brier, covered, crps_from_samples, log_score, majority_reliability, pit,
                               summarize_calibration, summarize_case)


def test_crps_point_mass_equals_absolute_error():
    assert crps_from_samples(np.array([5., 5., 5.]), 7.) == pytest.approx(2.)


def test_crps_two_point_distribution():
    # X ∈ {0, 2} equiprobable, y = 0: E|X-y| = 1, E|X-X'| = 1 -> CRPS = 0.5
    assert crps_from_samples(np.array([0., 2.]), 0.) == pytest.approx(0.5)


def test_crps_matches_brute_force(rng=np.random.default_rng(0)):
    x = rng.normal(size=200)
    y = 0.3
    brute = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert crps_from_samples(x, y) == pytest.approx(brute)


def test_brier_log_score_and_coverage():
    assert brier(0.8, True) == pytest.approx(0.04)
    assert brier(0.8, False) == pytest.approx(0.64)
    assert log_score(0.5, True) == pytest.approx(np.log(2))
    assert log_score(1.0, False) == pytest.approx(-np.log(1e-3))
    assert covered(1, 3, 2) and not covered(1, 3, 4) and not covered(np.nan, 3, 2)


def summary_frames():
    """Las tablas `shares`, `seats` y `blocks` de un caso (sin la ejecución a horizonte) con todas las columnas que
    `summarize_case` lee."""
    shares = pd.DataFrame({
        'party': ['A', 'B', 'C'], 'main': [True, True, False], 'official': [30., 20., 5.],
        'model': [31., 18., 6.], 'last_poll': [33., 17., 5.], 'mean_4w': [32., 19., 5.], 'prev_result': [28., 25., 4.],
        'fc_lo': [30.5, 17., 5.], 'fc_hi': [31.5, 19., 7.],
        'lo50': [30., 17.5, 5.5], 'hi50': [32., 18.5, 6.5], 'lo80': [29., 17., 5.], 'hi80': [33., 19., 7.],
        'lo95': [28., 16., 4.], 'hi95': [34., 20., 8.],
    })
    seats = pd.DataFrame({
        'party': ['A', 'B', 'C'], 'main': [True, True, False], 'official': [120., 80., 10.], 'model': [125., 75., 12.],
        'mean': [125., 75., 12.], 'crps': [3., 4., 1.], 'seats_valid': [True] * 3,
        'lo50': [122., 73., 11.], 'hi50': [128., 77., 13.], 'lo80': [118., 70., 9.], 'hi80': [132., 80., 15.],
        'lo95': [115., 68., 8.], 'hi95': [135., 82., 16.],
    })
    blocks = pd.DataFrame({'block': ['D', 'I'], 'brier': [0.04, 0.01], 'log_score': [0.2, 0.1], 'seats_valid': [True, True]})
    return shares, seats, blocks


def test_summarize_case_on_synthetic_frames():
    shares, seats, blocks = summary_frames()

    m = summarize_case(shares, seats, blocks)
    assert m['mae_shares'] == pytest.approx(1.5)          # sólo A y B (main)
    assert m['mae_last_poll'] == pytest.approx(3.0)
    assert m['cov_shares95'] == pytest.approx(1.0)
    assert m['cov_shares50'] == pytest.approx(0.5)        # 30 ∈ [30, 32]; 20 ∉ [17.5, 18.5]
    assert m['mae_seats'] == pytest.approx(5.0)
    assert m['cov_seats80'] == pytest.approx(1.0)
    assert m['crps_seats'] == pytest.approx(3.5)
    assert m['brier_vs'] == pytest.approx(0.025)
    assert m['seats_valid'] is True

    # Sin columnas `_h` (ejecución sólo nowcast) las métricas del horizonte quedan a NaN
    assert np.isnan(m['cov_shares95_h']) and np.isnan(m['crps_seats_h']) and np.isnan(m['brier_vs_h'])

    seats_invalid = seats.assign(seats_valid=False)
    m = summarize_case(shares, seats_invalid, blocks)
    assert np.isnan(m['mae_seats']) and np.isnan(m['brier_vs'])


def test_summarize_case_horizon_columns():
    """M5: la segunda ejecución (a horizonte, sufijo `_h`) tiene sus propias coberturas, CRPS y Brier."""
    shares = pd.DataFrame({
        'party': ['A', 'B'], 'main': [True, True], 'official': [30., 20.], 'model': [31., 18.],
        'last_poll': [33., 17.], 'mean_4w': [32., 19.], 'prev_result': [28., 25.],
        'fc_lo': [30.5, 17.], 'fc_hi': [31.5, 19.],
        'lo50': [30.5, 17.5], 'hi50': [31.5, 18.5], 'lo80': [30.2, 17.], 'hi80': [32., 19.], 'lo95': [29.9, 16.], 'hi95': [34., 19.5],
        # Intervalos a horizonte más anchos: cubren los dos resultados al 95 % y al 50 % sólo A
        'lo50_h': [29.5, 16.5], 'hi50_h': [32.5, 19.5], 'lo80_h': [29., 16.], 'hi80_h': [33., 20.5],
        'lo95_h': [28., 15.], 'hi95_h': [34., 21.],
    })
    seats = pd.DataFrame({
        'party': ['A', 'B'], 'main': [True, True], 'official': [120., 80.], 'model': [125., 75.], 'mean': [125., 75.],
        'crps': [3., 4.], 'seats_valid': [True, True],
        'lo50': [122., 73.], 'hi50': [128., 77.], 'lo80': [118., 70.], 'hi80': [132., 80.], 'lo95': [115., 68.], 'hi95': [135., 82.],
        'model_h': [123., 77.], 'mean_h': [123., 77.], 'crps_h': [4., 5.],
        'lo50_h': [119., 72.], 'hi50_h': [127., 82.], 'lo80_h': [115., 68.], 'hi80_h': [131., 86.], 'lo95_h': [110., 65.], 'hi95_h': [136., 90.],
    })
    blocks = pd.DataFrame({
        'block': ['D', 'I'], 'brier': [0.04, 0.01], 'log_score': [0.2, 0.1], 'seats_valid': [True, True],
        'brier_h': [0.09, 0.04], 'log_score_h': [0.3, 0.2]
    })

    m = summarize_case(shares, seats, blocks)
    assert m['cov_shares95'] == pytest.approx(0.5)      # 20 ∉ [16, 19.5]
    assert m['cov_shares95_h'] == pytest.approx(1.0)
    assert m['cov_shares50_h'] == pytest.approx(0.5)    # 20 ∉ [16.5, 19.5]
    assert m['cov_seats50'] == pytest.approx(0.0)
    assert m['cov_seats50_h'] == pytest.approx(1.0)
    assert m['mae_seats_h'] == pytest.approx(3.0)
    assert m['crps_seats_h'] == pytest.approx(4.5)
    assert m['brier_vs_h'] == pytest.approx(0.065)
    assert m['log_score_vs_h'] == pytest.approx(0.25)


def test_block_quantiles_are_those_of_the_summed_simulations():
    """La horquilla de escaños de un bloque es la de la suma de sus partidos simulación a simulación (no la suma
    de las horquillas de cada partido, que ignora la correlación); los partidos sin simular no cuentan."""
    from mtpy.lib.backtest import _block_quantiles

    dist = pd.DataFrame({'A': [100, 120, 140, 160], 'B': [60, 40, 20, 0], 'C': [10, 10, 10, 10]})
    out = _block_quantiles(dist, ['A', 'B'])
    grid = {'q{:02d}'.format(5 * i) for i in range(1, 20)}   # más los cuantiles de 5 en 5 (calibración)
    assert set(out) == {'lo50', 'hi50', 'lo80', 'hi80', 'lo95', 'hi95'} | grid
    assert out['lo95'] == out['hi95'] == 160  # A y B se compensan: el bloque suma 160 en toda simulación
    out = _block_quantiles(dist, ['A', 'C', 'Z'], suffix='_h')
    assert set(out) == {'lo50_h', 'hi50_h', 'lo80_h', 'hi80_h', 'lo95_h', 'hi95_h'} | {q + '_h' for q in grid}
    assert out['lo95_h'] == pytest.approx(111.5) and out['hi95_h'] == pytest.approx(168.5)  # cuantiles de 110..170
    assert _block_quantiles(dist, ['Z']) == {}


def test_provinces_report_the_central_allocation_and_keep_the_median(monkeypatch):
    from mtpy.lib import backtest
    from tests.fakes import NAMES, synthetic_simulator
    sim = synthetic_simulator()
    empty = pd.DataFrame(index=pd.Index([], name='region_id'))
    monkeypatch.setattr(backtest, 'official_provinces', lambda scope, event_date: (empty, empty))
    rows = backtest._provinces(sim, 'es', '2026-11-29', 30, {'PP'}).set_index(['region', 'party'])
    assert rows.loc[(28, 'PP'), 'model_seats'] == 3 and rows.loc[(8, 'VOX'), 'model_seats'] == 0
    medians = pd.DataFrame(sim.results[:, 1, :], columns=NAMES).median()
    assert rows.loc[(28, 'PSOE'), 'median_seats'] == medians['PSOE']


def test_summarize_case_provincial_metrics_skip_cells_without_projection():
    """M9 (revisión): las cuotas provinciales se evalúan sólo donde el modelo proyecta (mediana finita), con el
    mismo denominador en cobertura y error; un caso huérfano (escaños no válidos) no aporta métricas provinciales."""
    shares = pd.DataFrame({
        'party': ['A'], 'main': [True], 'official': [30.], 'model': [31.], 'last_poll': [33.], 'mean_4w': [32.],
        'prev_result': [28.], 'fc_lo': [30.5], 'fc_hi': [31.5], 'lo50': [30.], 'hi50': [32.], 'lo80': [29.], 'hi80': [33.],
        'lo95': [28.], 'hi95': [34.],
    })
    seats = pd.DataFrame({
        'party': ['A'], 'main': [True], 'official': [120.], 'model': [125.], 'mean': [125.], 'crps': [3.], 'seats_valid': [True],
        'lo50': [122.], 'hi50': [128.], 'lo80': [118.], 'hi80': [132.], 'lo95': [115.], 'hi95': [135.],
    })
    blocks = pd.DataFrame({'block': ['D'], 'brier': [0.04], 'log_score': [0.2], 'seats_valid': [True]})
    provinces = pd.DataFrame({
        'region': [1, 1, 2, 2], 'party': ['A', 'B', 'A', 'B'], 'official_pct': [30., 10., 20., 5.],
        'model_pct': [31., 12., 22., np.nan], 'official_seats': [3., 1., 2., 1.], 'model_seats': [3., 1., 2., 0.],
        'crps_seats': [0.1, 0.2, 0.3, 1.0],
        'lo50_pct': [30., 11., 21., np.nan], 'hi50_pct': [32., 13., 23., np.nan],
        'lo80_pct': [29., 10., 20., np.nan], 'hi80_pct': [33., 14., 24., np.nan],
        'lo95_pct': [28., 9., 19., np.nan], 'hi95_pct': [34., 15., 25., np.nan],
        'lo95_seats': [2., 0., 1., 0.], 'hi95_seats': [4., 2., 3., 0.],
    })

    m = summarize_case(shares, seats, blocks, provinces)
    assert m['n_prov_cells'] == 3 and m['n_prov_missing'] == 1
    assert m['cov_prov_shares50'] == pytest.approx(1 / 3)          # 30 ∈ [30, 32]; 10 ∉ [11, 13]; 20 ∉ [21, 23]
    assert m['cov_prov_shares95'] == pytest.approx(1.0)
    assert m['mae_prov_shares'] == pytest.approx(5 / 3)             # |1| + |2| + |2| sobre las tres celdas proyectadas
    assert m['cov_prov_seats95'] == pytest.approx(0.75)             # la celda sin proyección cuenta como 0 escaños
    assert m['crps_prov_seats'] == pytest.approx(0.4)

    m = summarize_case(shares, seats.assign(seats_valid=False), blocks.assign(seats_valid=False), provinces)
    assert np.isnan(m['cov_prov_shares50']) and np.isnan(m['mae_prov_shares']) and np.isnan(m['n_prov_cells'])


def test_case_series_is_the_fitted_series_up_to_the_anchor():
    """La serie de un caso es la del Forecaster del simulador hasta `as_of` incluido, con la elección y el horizonte delante."""
    from mtpy.lib import backtest
    from tests.fakes import NAMES, synthetic_simulator
    sim = synthetic_simulator()
    sim.as_of = pd.Timestamp('2026-10-03')
    frame = backtest.case_series(sim, '2026-11-29', 57)
    assert list(frame.columns) == ['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi']
    assert frame.shape[0] == 33 * 3 and frame['date'].max() == '2026-10-03' and set(frame['name']) == set(NAMES)
    assert (frame['event_date'] == '2026-11-29').all() and (frame['horizon'] == 57).all()


def fake_results():
    """Lo que devuelve `run_backtest`: las tablas sintéticas (2 elecciones x 2 horizontes) y `provinces`, `series` y `meta`
    vacías."""
    import pandas as pd
    from tests.fakes import synthetic_backtest
    tables = synthetic_backtest()
    return {'shares': tables.shares, 'seats': tables.seats, 'blocks': tables.blocks, 'provinces': pd.DataFrame(),
            'series': pd.DataFrame(columns=['event_date', 'horizon', 'date', 'name', 'mean', 'lo', 'hi']),
            'meta': pd.DataFrame(), 'metrics': tables.metrics, 'by_horizon': tables.by_horizon}


def test_results_fs_keeps_es_at_the_root_locally_and_under_the_prefix_in_the_app_fs(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    from mtpy.lib import backtest, publish
    fs, folder = backtest.results_fs(str(tmp_path), 'es')
    assert isinstance(fs, FileSystem) and fs.path == str(tmp_path) and folder == ''
    assert backtest.results_fs(str(tmp_path), 'es-md')[1] == 'es-md'
    assert backtest.results_path('', 'metrics.csv') == 'metrics.csv' and backtest.results_path('es-md', 'run.json') == 'es-md/run.json'
    with pytest.raises(RuntimeError, match='no file system configured'):
        backtest.results_fs(None, 'es')
    fresh_app.set('fs', FileSystem(str(tmp_path / 'files')))
    fs, folder = backtest.results_fs(None, 'es-md')
    assert fs is fresh_app.fs and folder == 'backtest/results/es-md' and backtest.results_fs(None, 'es')[1] == 'backtest/results'
    assert publish.backtest_source('es-md', 'x')[1] == 'es-md' and publish.backtest_source('es')[1] == 'backtest/results'
    assert backtest.describe_source(fs, folder) == str(tmp_path / 'files' / 'backtest' / 'results' / 'es-md')
    assert backtest.describe_source(FileSystem(backtest.RESULTS_ROOT), '') == 'backtest/results'


def test_describe_source_names_the_bucket_of_an_s3_file_system():
    from mtpy.core.services.s3 import S3
    from mtpy.lib import backtest
    fs = S3.__new__(S3)
    fs.bucket = 'elections-s3-bucket'
    assert backtest.describe_source(fs, 'backtest/results/es-md') == 's3://elections-s3-bucket/backtest/results/es-md'


def test_public_source_drops_the_bucket_of_s3_and_describes_a_local_folder(tmp_path):
    """La fuente pública del meta del backtest: el prefijo sin el bucket en S3, la descripción local si no."""
    from mtpy.core.io import FileSystem
    from mtpy.core.services.s3 import S3
    from mtpy.lib import backtest
    fs = S3.__new__(S3)
    fs.bucket = 'elections-s3-bucket'
    assert backtest.public_source(fs, 'backtest/results/es-md') == 'backtest/results/es-md'
    assert backtest.public_source(FileSystem(backtest.RESULTS_ROOT), 'es-md') == 'backtest/results/es-md'
    local = FileSystem(str(tmp_path))
    assert backtest.public_source(local, 'es-md') == backtest.describe_source(local, 'es-md') == str(tmp_path / 'es-md')


def test_backtest_scope_writes_into_the_app_fs_without_out(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    from mtpy.lib import backtest
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    figures = backtest.backtest_scope('es-md', events=['2023-05-28'], runner=lambda **kwargs: fake_results(),
                                      stats=lambda scope: {'db_polls': 1, 'db_last_poll': None},
                                      prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    folder = tmp_path / 'backtest' / 'results' / 'es-md'
    assert (folder / 'metrics.csv').exists() and (folder / 'run.json').exists() and figures['folder'] == str(folder)


def test_backtest_scope_writes_the_tables_and_the_run_record(tmp_path):
    """`backtest_scope` ejecuta el backtest de un ámbito, escribe sus ocho tablas y `run.json` (con los parámetros,
    las elecciones evaluadas, los horizontes ordenados, el commit y el estado de la base) y devuelve las cifras."""
    import json
    import pandas as pd
    from mtpy.lib import backtest
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return fake_results()

    figures = backtest.backtest_scope(
        'es-md', out=str(tmp_path), events=['2019-11-10', '2023-07-23'], horizons=[30, 6], n_sim=20, composition=0.25,
        min_events=1, verbose=1, runner=runner, stats=lambda scope: {'db_polls': 12, 'db_last_poll': '2026-10-03'},
        prov=lambda: {'commit': 'abc1234', 'dirty': True, 'versions': {}})
    folder = tmp_path / 'es-md'
    assert sorted(p.name for p in folder.iterdir()) == ['blocks.csv', 'by_horizon.csv', 'meta.csv', 'metrics.csv',
                                                        'provinces.csv', 'run.json', 'seats.csv', 'series.csv', 'shares.csv']
    assert pd.read_csv(folder / 'metrics.csv').shape[0] == 4 and pd.read_csv(folder / 'by_horizon.csv').shape[0] == 2
    run = json.load(open(folder / 'run.json', encoding='utf-8'))
    assert set(run) == {'run_at', 'commit', 'scope', 'n_sim', 'seed', 'max_fc', 'nowcast_only', 'house_effects',
                        'industry_bias', 'dispersion', 'herding', 'composition', 'regional_noise', 'min_events',
                        'recompute_ratings', 'events', 'horizons', 'db_polls', 'db_last_poll'}
    assert run['scope'] == 'es-md' and run['commit'] == 'abc1234' and run['run_at'].startswith('20')
    assert run['n_sim'] == 20 and run['seed'] == 42 and run['composition'] == 0.25 and run['min_events'] == 1
    assert run['events'] == ['2019-11-10', '2023-07-23'] and run['horizons'] == [6, 30]
    assert run['db_polls'] == 12 and run['db_last_poll'] == '2026-10-03'
    assert seen['scope'] == 'es-md' and seen['events'] == ['2019-11-10', '2023-07-23'] and seen['horizons'] == [30, 6]
    assert seen['n_sim'] == 20 and seen['verbose'] == 1 and seen['composition'] == 0.25 and seen['min_events'] == 1
    assert figures['n_cases'] == 4 and figures['n_events'] == 2 and figures['n_horizons'] == 2
    assert figures['folder'] == str(folder) and figures['by_horizon'].shape[0] == 2 and figures['seconds']['total'] >= 0


def test_backtest_scope_recomputes_the_ratings_once_and_passes_them_to_every_case(tmp_path, monkeypatch):
    """`--recompute-ratings`: los ratings se recalculan una vez por ámbito (`rating_overrides`), el ejecutor los recibe
    como `weight_ratings` (los pasa a cada caso) y `run.json` lo anota; sin la opción no se calcula nada."""
    import json
    import pandas as pd
    from mtpy.lib import backtest
    fixed = pd.Series([2.0], index=pd.MultiIndex.from_tuples([('2023-07-23', 1)], names=['event_date', 'pollster_id']))
    calls, seen = [], []
    monkeypatch.setattr(backtest, 'rating_overrides', lambda scope: calls.append(scope) or fixed)

    def runner(**kwargs):
        seen.append(kwargs)
        return fake_results()

    options = dict(runner=runner, stats=lambda scope: {'db_polls': 1, 'db_last_poll': None},
                   prov=lambda: {'commit': None, 'dirty': False, 'versions': {}})
    backtest.backtest_scope('es', out=str(tmp_path / 'on'), recompute_ratings=True, **options)
    assert calls == ['es'] and seen[0]['weight_ratings'] is fixed
    assert json.load(open(tmp_path / 'on' / 'run.json', encoding='utf-8'))['recompute_ratings'] is True
    assert 'recompute_ratings' not in seen[0]

    backtest.backtest_scope('es', out=str(tmp_path / 'off'), **options)
    assert calls == ['es'] and seen[1]['weight_ratings'] is None
    assert json.load(open(tmp_path / 'off' / 'run.json', encoding='utf-8'))['recompute_ratings'] is False


def test_run_backtest_passes_the_ratings_to_every_case(monkeypatch):
    """`run_backtest` traslada `weight_ratings` a cada caso."""
    import warnings
    from mtpy.lib import backtest
    seen = []

    def failing_case(*args, **kwargs):
        seen.append(kwargs.get('weight_ratings'))
        raise ValueError('no usable polls')

    monkeypatch.setattr(backtest, 'run_case', failing_case)
    marker = pd.Series([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        backtest.run_backtest(scope='es', events=['2023-07-23'], horizons=[6, 30], weight_ratings=marker)
    assert seen == [marker, marker]


def test_backtest_scope_evaluates_the_default_elections_of_es_at_the_root(tmp_path):
    """Sin `events`, `es` evalúa `DEFAULT_EVENTS` (sin base) y escribe en la raíz de `out`."""
    from mtpy.lib import backtest
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return fake_results()

    backtest.backtest_scope('es', out=str(tmp_path), runner=runner, stats=lambda scope: {'db_polls': 1, 'db_last_poll': None},
                            prov=lambda: {'commit': None, 'dirty': False, 'versions': {}})
    assert seen['events'] == backtest.DEFAULT_EVENTS and seen['horizons'] is None and seen['n_sim'] == 1000
    assert (tmp_path / 'run.json').exists() and (tmp_path / 'metrics.csv').exists()


def test_backtest_scope_skips_a_scope_without_elections_or_cases(tmp_path, monkeypatch):
    """Un ámbito sin elecciones evaluables o sin ningún caso evaluado lanza `NothingToEvaluate` y no escribe nada."""
    import pandas as pd
    import pytest
    from mtpy.lib import backtest
    monkeypatch.setattr(backtest, 'default_events', lambda scope, since='2019-01-01': [])
    with pytest.raises(backtest.NothingToEvaluate, match='no elections to evaluate'):
        backtest.backtest_scope('es-xx', out=str(tmp_path), runner=lambda **kwargs: fake_results())
    empty = {**fake_results(), 'metrics': pd.DataFrame(), 'by_horizon': pd.DataFrame()}
    with pytest.raises(backtest.NothingToEvaluate, match='no case could be evaluated'):
        backtest.backtest_scope('es-xx', out=str(tmp_path), events=['2023-05-28'], runner=lambda **kwargs: empty)
    assert not (tmp_path / 'es-xx').exists()


def test_poll_stats_counts_the_polls_of_the_scope(monkeypatch):
    """`poll_stats` registra cuántos sondeos del ámbito hay en la base y la fecha del último (`None` sin sondeos)."""
    import pandas as pd
    from mtpy.lib import backtest
    from mtpy.models import elections

    class FakePolls:
        def get_results(self, query=None, formatted=False):
            assert formatted and len(query['filters']) == 1
            if "event_scope = 'es-xx'" in query['filters'][0]:
                return pd.DataFrame({'date': pd.to_datetime([])})
            return pd.DataFrame({'date': pd.to_datetime(['2026-10-01', '2026-10-03'])})

    monkeypatch.setattr(elections, 'Polls', FakePolls)
    assert backtest.poll_stats('es') == {'db_polls': 2, 'db_last_poll': '2026-10-03'}
    assert backtest.poll_stats('es-xx') == {'db_polls': 0, 'db_last_poll': None}


def test_run_backtest_survives_when_every_case_is_skipped(monkeypatch):
    """M11: un ámbito cuyos casos fallan todos devuelve tablas vacías, no una excepción."""
    import warnings
    from mtpy.lib import backtest

    def failing_case(*args, **kwargs):
        raise ValueError('no usable polls')

    monkeypatch.setattr(backtest, 'run_case', failing_case)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = backtest.run_backtest(scope='es-xx', events=['2023-05-28'], horizons=[6, 30])
    assert out['metrics'].shape[0] == 0 and out['by_horizon'].shape[0] == 0
    assert out['meta'].shape[0] == 2 and out['meta']['error'].eq('no usable polls').all()


def test_run_backtest_passes_min_events_to_every_case(monkeypatch):
    """Validación del mínimo (M11): `run_backtest` traslada `min_events` a cada caso."""
    import warnings
    from mtpy.lib import backtest

    seen = []

    def failing_case(*args, **kwargs):
        seen.append(kwargs.get('min_events'))
        raise ValueError('no usable polls')

    monkeypatch.setattr(backtest, 'run_case', failing_case)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        backtest.run_backtest(scope='es-xx', events=['2023-05-28'], horizons=[6, 30], min_events=99)
    assert seen == [99, 99]


def test_run_backtest_keeps_the_series_of_the_shortest_horizon_of_each_election(monkeypatch):
    """De las series de los casos sólo queda la del horizonte más corto que corrió en cada elección (el 6, o el 30
    si el caso a 6 días se omitió); las demás tablas siguen igual."""
    import warnings
    from mtpy.lib import backtest
    from tests.fakes import BACKTEST_EVENTS, BACKTEST_HORIZONS

    def case(scope, event_date, horizon, **kwargs):
        if (event_date, horizon) == ('2019-11-10', 6):
            raise ValueError('no usable polls')
        shares, seats, blocks = (t.assign(event_date=event_date, horizon=horizon) for t in summary_frames())
        series = pd.DataFrame({'event_date': [event_date] * 2, 'horizon': [horizon] * 2, 'date': ['2023-07-01', '2023-07-02'],
                               'name': ['PP', 'PP'], 'mean': [30.0, 30.5], 'lo': [29.0, 29.5], 'hi': [31.0, 31.5]})
        calibration = pd.DataFrame({'event_date': [event_date], 'horizon': [horizon], 'kind': ['shares'], 'mode': ['now'],
                                    'party': ['A'], 'pit': [0.5]})
        return {'shares': shares, 'seats': seats, 'blocks': blocks, 'provinces': pd.DataFrame(),
                'meta': pd.DataFrame([{'event_date': event_date, 'horizon': horizon}]), 'series': series,
                'calibration': calibration}

    monkeypatch.setattr(backtest, 'run_case', case)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = backtest.run_backtest(scope='es', events=BACKTEST_EVENTS, horizons=BACKTEST_HORIZONS)
    kept = out['series'][['event_date', 'horizon']].drop_duplicates()
    assert sorted(map(tuple, kept.itertuples(index=False))) == [('2019-11-10', 30), ('2023-07-23', 6)]
    assert out['series'].shape[0] == 4 and out['metrics'].shape[0] == 3
    assert out['calibration'].shape[0] == 3   # la calibración de cada caso que corrió, concatenada
    assert list(backtest.shortest_horizon(pd.DataFrame(columns=out['series'].columns)).columns) == list(out['series'].columns)


def test_run_backtest_script_delegates_to_the_job(fresh_app, monkeypatch, capsys):
    """`backtest/run_backtest.py` es un envoltorio: traduce sus opciones a `Backtest.run` (la composición la normaliza
    el job) y `--scopes` sin `all` pasa tal cual."""
    import importlib.util
    import os
    import sys
    from mtpy import mtpy as app
    from mtpy.jobs import Backtest as job_module
    seen = {}
    monkeypatch.setattr(app, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(job_module.Backtest, 'run', lambda self, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(sys, 'argv', ['run_backtest.py', '--scopes', 'es-md', 'es-cl', '--horizons', '6', '30',
                                      '--n-sim', '20', '--composition', 'auto', '--no-herding', '--out', 'x'])
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    spec = importlib.util.spec_from_file_location('run_backtest', os.path.join(root, 'backtest', 'run_backtest.py'))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main()
    assert seen == {'scopes': ['es-md', 'es-cl'], 'events': None, 'horizons': [6, 30], 'n_sim': 20, 'seed': 42, 'max_fc': 10,
                    'nowcast_only': False, 'house_effects': True, 'industry_bias': False, 'dispersion': True,
                    'herding': False, 'composition': 'auto', 'regional_noise': True, 'min_events': None, 'recompute_ratings': False, 'out': 'x'}
    hint = capsys.readouterr().out
    assert hint.startswith('Tables written to x. publish backtest/event read the app file system: run python job.py backtest ')
    assert '\'{"action":"push","scopes":["es-md","es-cl"],"out":"x"}\' (with S3_BUCKET set)' in hint
    default = os.path.join(root, 'backtest', 'results')
    assert '\'{"action":"push","scopes":["es"]}\'' in script.push_hint(default, ['es'], default)


def test_backtest_simulates_a_thousand_times_by_default_like_the_forecast():
    """El backtest simula 1.000 veces por caso por defecto, como `publish forecast`, en la librería y en el script."""
    import inspect
    import os
    from mtpy.lib import backtest as lib
    assert inspect.signature(lib.run_case).parameters['n_sim'].default == 1000
    assert inspect.signature(lib.run_backtest).parameters['n_sim'].default == 1000
    script = open(os.path.join(os.path.dirname(__file__), '..', 'backtest', 'run_backtest.py'), encoding='utf-8').read()
    assert "add_argument('--n-sim', type=int, default=1000)" in script and '[--n-sim 1000]' in script


def test_pit_counts_the_mass_below_the_outcome_with_half_the_ties():
    assert pit(np.array([1., 2., 3., 4.]), 2.5) == pytest.approx(0.5)
    assert pit(np.array([1., 2., 2., 4.]), 2.) == pytest.approx(0.5)   # 1 por debajo + 2 empates / 2 = 2 de 4
    assert np.isnan(pit(np.array([1., 2.]), np.nan))


def test_pit_ignores_missing_samples():
    """Las simulaciones sin valor (una celda provincial sin proyección) no cuentan; sin ninguna, el PIT es NaN."""
    assert pit(np.array([1., np.nan, 3.]), 2.) == pytest.approx(0.5)
    assert np.isnan(pit(np.array([np.nan, np.nan]), 2.))


def test_quantile_cols_carry_the_five_percent_grid():
    cols = _quantile_cols(np.arange(1., 101.))
    assert cols['lo95'] < cols['q05'] < cols['q50'] < cols['q95'] < cols['hi95'] and 'q50_h' in _quantile_cols(np.arange(3.), suffix='_h')


def test_quantile_cols_keep_the_intervals_and_add_nineteen_quantiles():
    """Las columnas de hoy (`lo`/`hi` de `LEVELS`) siguen igual y se añaden `q05`…`q95`, también con sufijo."""
    from mtpy.lib.backtest import QUANTILES
    x = np.arange(1., 101.)
    cols = _quantile_cols(x)
    assert QUANTILES[0] == pytest.approx(0.05) and QUANTILES[-1] == pytest.approx(0.95) and len(QUANTILES) == 19
    assert [k for k in cols if k.startswith('q')] == ['q{:02d}'.format(5 * i) for i in range(1, 20)]
    assert cols['lo95'] == pytest.approx(np.quantile(x, 0.025)) and cols['hi50'] == pytest.approx(np.quantile(x, 0.75))
    assert cols['q10'] == pytest.approx(np.quantile(x, 0.10))
    assert set(_quantile_cols(x, suffix='_h')) == {k + '_h' for k in cols}


def test_summarize_calibration_weights_each_election_equally_and_bins_by_horizon_group():
    cal = pd.DataFrame({
        'population': ['generales'] * 6, 'kind': ['shares'] * 6, 'mode': ['now'] * 6,
        'event_date': ['2023-07-23'] * 4 + ['2019-11-10'] * 2, 'horizon': [6, 6, 6, 6, 14, 14],
        'pit': [0.05, 0.05, 0.05, 0.05, 0.95, 0.95], 'official': 1., 'model': 1., 'main': True, 'party': list('ABCDEF')})
    out = summarize_calibration(cal, bins=10)
    row = out.query("group == '6-14'")
    assert row.loc[row.bin == 0, 'observed'].iloc[0] == pytest.approx(0.5)   # 2023 pesa 1/2 aunque tenga 4 celdas
    assert row.loc[row.bin == 9, 'observed'].iloc[0] == pytest.approx(0.5)
    assert row['n_events'].iloc[0] == 2 and row['n_cells'].iloc[0] == 6 and out['expected'].eq(0.1).all()


def test_summarize_calibration_drops_nan_pit_and_keeps_only_now_without_h():
    cal = pd.DataFrame({'population': ['generales'] * 2, 'kind': ['shares'] * 2, 'mode': ['now'] * 2, 'event_date': ['2023-07-23'] * 2,
                        'horizon': [6, 6], 'pit': [0.5, np.nan], 'official': [1., np.nan], 'model': 1., 'main': True, 'party': ['A', 'B']})
    out = summarize_calibration(cal)
    assert out['n_cells'].iloc[0] == 1 and set(out['mode']) == {'now'}


def test_summarize_calibration_lists_every_bin_and_sends_provincial_cells_to_their_population():
    """Cada grupo trae sus `bins` cajas (vacías a 0) que suman 1; las filas `prov_*` van a `provincias` sea cual sea
    la población que traigan; un PIT de 1 cae en la última caja y los horizontes fuera de los grupos no cuentan."""
    cal = pd.DataFrame({
        'population': ['autonomicas'] * 4, 'kind': ['shares', 'shares', 'prov_shares', 'shares'], 'mode': ['h', 'h', 'now', 'h'],
        'event_date': ['2023-05-28'] * 4, 'horizon': [30, 60, 30, 20], 'pit': [1.0, 0.31, 0.5, 0.5], 'party': list('ABCD')})
    out = summarize_calibration(cal, bins=5)
    assert list(out.columns) == ['population', 'kind', 'mode', 'group', 'bin', 'expected', 'observed', 'n_cells', 'n_events']
    auto = out.query("population == 'autonomicas'")
    assert auto['group'].unique().tolist() == ['30-60'] and auto['bin'].tolist() == [0, 1, 2, 3, 4]
    assert auto['observed'].tolist() == pytest.approx([0., 0.5, 0., 0., 0.5]) and auto['n_cells'].eq(2).all()
    prov = out.query("population == 'provincias'")
    assert set(prov['kind']) == {'prov_shares'} and prov.loc[prov.bin == 2, 'observed'].iloc[0] == pytest.approx(1.)
    assert summarize_calibration(cal.iloc[:0]).empty


def test_majority_reliability_bins_probabilities():
    blocks = pd.DataFrame({'p_majority': [0.05, 0.95, 0.6], 'majority': [False, True, False], 'seats_valid': True})
    out = majority_reliability(blocks)
    assert out.loc[out.n > 0, 'observed'].tolist() == [0., 0., 1.] and out['mode'].eq('now').all()


def test_majority_reliability_adds_the_horizon_run_and_skips_invalid_seats():
    """Con `p_majority_h` sale también el modo `h`; los casos con escaños no válidos (huérfanos) no cuentan."""
    blocks = pd.DataFrame({'p_majority': [0.05, 0.95, 0.6], 'p_majority_h': [0.2, 0.8, 0.0], 'majority': [False, True, True],
                           'seats_valid': [True, True, False]})
    out = majority_reliability(blocks)
    assert out['mode'].unique().tolist() == ['now', 'h'] and out.groupby('mode')['bin'].count().eq(6).all()
    h = out.query("mode == 'h' and n > 0")
    assert h['p_mean'].tolist() == pytest.approx([0.2, 0.8]) and h['observed'].tolist() == [0., 1.] and h['n'].sum() == 2


def test_provinces_carry_the_pit_of_shares_and_seats(monkeypatch):
    """Cada celda provincial lleva el PIT de su cuota y de sus escaños; sin cuota oficial el PIT de la cuota es NaN
    (aunque la tabla la anote como 0) y sin escaños oficiales cuentan 0."""
    from mtpy.lib import backtest
    from tests.fakes import synthetic_simulator
    sim = synthetic_simulator()
    pct = pd.DataFrame({'PP': [40., 41.], 'PSOE': [30., 31.]}, index=pd.Index([28, 8], name='region_id'))
    seats = pd.DataFrame({'PP': [4., 2.], 'PSOE': [3., 2.]}, index=pd.Index([28, 8], name='region_id'))
    monkeypatch.setattr(backtest, 'official_provinces', lambda scope, event_date: (pct, seats))
    rows = backtest._provinces(sim, 'es', '2026-11-29', 30, {'PP'}).set_index(['region', 'party'])
    samples = sim.units[:, 1, 0, sim.cols_unit.index('vpred_pct')].astype(float)
    assert rows.loc[(28, 'PP'), 'pit_pct'] == pytest.approx(pit(samples, 40.))
    assert rows.loc[(28, 'PP'), 'pit_seats'] == pytest.approx(pit(sim.results[:, 1, 0].astype(float), 4.))
    assert rows.loc[(28, 'VOX'), 'official_pct'] == 0 and np.isnan(rows.loc[(28, 'VOX'), 'pit_pct'])
    assert rows.loc[(28, 'VOX'), 'pit_seats'] == pytest.approx(pit(sim.results[:, 1, 2].astype(float), 0.))
    assert {'q05_pct', 'q95_pct', 'q50_seats'} <= set(rows.columns)


def test_case_calibration_has_one_row_per_cell_kind_and_mode():
    """`_calibration`: cuotas y escaños nacionales en los dos modos (`h` sólo con la ejecución a horizonte), las
    celdas provinciales sólo en `now` y los escaños (nacionales y provinciales) sólo con escaños válidos, como en
    `summarize_case`; un partido sin resultado oficial lleva PIT NaN."""
    from mtpy.lib.backtest import _calibration
    names = ['A', 'B']
    run = {'shares': pd.DataFrame({'A': [29., 31., 33.], 'B': [9., 10., 11.]}),
           'dist': pd.DataFrame({'A': [100., 110., 120.], 'B': [5., 6., 7.]}),
           'summary': pd.DataFrame({'pct': [31., 10.], 'seats': [110., 6.]}, index=names)}
    official = pd.DataFrame({'pct': [32., np.nan], 'seats': [110., np.nan]}, index=names)
    provinces = pd.DataFrame({'region': [28], 'party': ['A'], 'main': [True], 'official_pct': [30.], 'model_pct': [29.],
                              'pit_pct': [0.25], 'official_seats': [12.], 'model_seats': [11.], 'pit_seats': [0.75]})

    cal = _calibration('2023-07-23', 6, names, {'A'}, official, run, run, provinces, seats_valid=True)
    assert list(cal.columns) == ['event_date', 'horizon', 'kind', 'mode', 'party', 'region_id', 'main', 'official', 'model', 'pit']
    assert sorted(set(zip(cal['kind'], cal['mode']))) == [('prov_seats', 'now'), ('prov_shares', 'now'), ('seats', 'h'),
                                                          ('seats', 'now'), ('shares', 'h'), ('shares', 'now')]
    a = cal.query("kind == 'shares' and mode == 'now' and party == 'A'").iloc[0]
    assert a['pit'] == pytest.approx(2 / 3) and a['model'] == 31. and a['official'] == 32. and bool(a['main'])
    assert cal.query("kind == 'seats' and party == 'A'")['pit'].tolist() == pytest.approx([0.5, 0.5])
    assert cal.query("party == 'B'")['pit'].isna().all() and pd.isna(a['region_id'])
    prov = cal.query("kind == 'prov_seats'").iloc[0]
    assert (prov['region_id'], prov['official'], prov['model'], prov['pit']) == (28, 12., 11., 0.75)

    only = _calibration('2023-07-23', 6, names, {'A'}, official, run, None, provinces, seats_valid=False)
    assert set(only['mode']) == {'now'} and set(only['kind']) == {'shares', 'prov_shares'}
    assert 'seats' not in set(_calibration('2023-07-23', 6, names, {'A'}, official, run, run, provinces, False)['kind'])
