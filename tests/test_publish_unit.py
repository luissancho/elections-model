"""Tests de `mtpy/lib/publish/` sin base de datos: exportadores sobre el Simulator sintético de `tests/fakes.py`."""
import math
import os
import zoneinfo
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, data, publish
from tests.fakes import (BMAPS, NAMES, REGIONS, FakeComputer, FakeSimulator, RecordingFileSystem, synthetic_analysis,
                         synthetic_backtest, synthetic_cycle, synthetic_event, synthetic_simulator, write_backtest_source, CYCLE_SEATS)


def valid(name, data, mode='nowcast'):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` y vuelve sin NaN."""
    env = bundle.jsonable(bundle.envelope(name, data, 'es', run_id='20261008-120000', mode=mode,
                                          generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(name, env)['data']


def test_export_vote_rows_are_rounded_and_ordered():
    data, frame = publish.export_vote(synthetic_simulator())
    data = valid('vote', data)
    assert data['horizon'] == 0 and data['when'] == '2026-10-05'
    assert [r['name'] for r in data['rows']] == NAMES
    row = data['rows'][0]
    assert row['pct'] == 40.0 and row['lo'] <= row['pct'] <= row['hi']
    assert all(round(r['pct'], 2) == r['pct'] for r in data['rows'])
    assert list(frame.columns) == ['name', 'pct', 'sd', 'lo', 'hi']


def test_export_summary_groups_and_probabilities():
    sim = synthetic_simulator()
    data, frame = publish.export_summary(sim)
    data = valid('summary', data)
    assert data['n_seats'] == 10 and data['majority'] == 6
    assert [r['name'] for r in data['parties']] == NAMES
    assert sum(data['totals'].values()) == 10
    assert set(data['p_majority']) == {'Derecha', 'Izquierda'}
    assert all(0 <= p <= 1 for p in data['p_majority'].values())
    assert set(data['parties'][0]) == {'name', 'pct', 'pct_mean', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_median',
                                       'seats_lo', 'seats_hi', 'seats_min', 'seats_max', 'p_seats', 'p_majority', 'p_first'}
    assert all(round(r['p_first'], 3) == r['p_first'] for r in data['parties'])
    assert all(isinstance(r['seats'], int) for r in data['parties'])
    assert sorted(frame['group'].unique()) == ['blocks', 'parties', 'vs']


def test_export_summary_writes_null_for_a_block_without_parties():
    sim = synthetic_simulator()
    sim.model.bmaps = {**sim.model.bmaps, 'vs': {'Derecha': ['PP', 'VOX'], 'Nadie': ['UP']}}
    data, _ = publish.export_summary(sim)
    data = valid('summary', data)
    nobody = [r for r in data['vs'] if r['name'] == 'Nadie'][0]
    assert nobody['p_first'] is None and nobody['seats_mean'] is None
    assert nobody['seats'] is None
    assert data['p_majority']['Nadie'] is None


def test_export_dist_rows_sum_the_chamber():
    data, frame = publish.export_dist(synthetic_simulator(n_sim=50))
    data = valid('dist', data)
    assert data['parties'] == NAMES and len(data['seats']) == 50
    assert all(sum(row) == 10 for row in data['seats'])
    assert frame.shape == (50, 3)


def test_export_districts_excludes_the_total_region():
    data, frame = publish.export_districts(synthetic_simulator())
    data = valid('districts', data)
    assert [r['id'] for r in data['regions']] == [28, 8]
    assert [r['seats'] for r in data['regions']][0] == 6
    assert len(data['rows']) == 2 * len(NAMES)
    assert set(data['rows'][0]) == {'region_id', 'region', 'name', 'pct', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_lo',
                                    'seats_hi', 'p_seats'}
    assert frame.shape[0] == 6
    assert all(isinstance(r['seats'], float) and r['seats'] == round(r['seats'], 1) for r in data['rows'])


def test_export_districts_carries_the_community_and_the_votes():
    communities = {28: {'code': 13, 'name': 'Madrid'}, 8: {'code': 9, 'name': 'Cataluña'}}
    result = valid('districts', publish.export_districts(synthetic_simulator(), communities=communities)[0])
    assert result['regions'][0] == {'id': 28, 'name': 'Madrid', 'seats': 6, 'community': {'code': 13, 'name': 'Madrid'}, 'votes': 600}
    assert result['regions'][1]['community'] == {'code': 9, 'name': 'Cataluña'} and result['regions'][1]['votes'] == 400
    bare = valid('districts', publish.export_districts(synthetic_simulator())[0])
    assert bare['regions'][0]['community'] is None and bare['regions'][0]['votes'] == 600


def test_export_districts_takes_the_votes_from_the_previous_election():
    # The totals of the upcoming election carry no votes (the loader only fills its seats): the votes of
    # `districts.regions` are the valid votes of the previous election, `sim.prev_totals`, as integers
    sim = synthetic_simulator()
    assert sim.reg_totals['votes'].isnull().all()
    sim.prev_totals.loc[28, 'votes'] = 600.4
    result = valid('districts', publish.export_districts(sim)[0])
    assert [r['votes'] for r in result['regions']] == [600, 400]
    assert all(isinstance(r['votes'], int) for r in result['regions'])


def test_district_communities_come_from_the_catalogue_and_are_null_in_their_own_scope(fresh_app):
    fresh_app.set('data', FileSystem(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')))
    national = data.get_district_communities('es')
    assert len(national) == 52 and national[1] == {'code': 16, 'name': 'País Vasco'} and national[28] == {'code': 13, 'name': 'Madrid'}
    assert data.get_district_communities('es-pv') == {1: None, 20: None, 48: None}


def test_without_the_catalogue_the_districts_carry_no_community(fresh_app):
    assert publish.forecast.district_communities('es') == {}
    communities = publish.forecast.district_communities('es')
    regions = valid('districts', publish.export_districts(synthetic_simulator(), communities)[0])['regions']
    assert [r['community'] for r in regions] == [None, None] and [r['votes'] for r in regions] == [600, 400]


def test_a_missing_catalogue_raises_instead_of_falling_back(fresh_app, tmp_path):
    fresh_app.set('data', FileSystem(str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        publish.forecast.district_communities('es')


def test_export_scenario_is_a_real_simulation():
    sim = synthetic_simulator()
    data, frame = publish.export_scenario(sim)
    data = valid('scenario', data)
    assert data['simulation'] == sim.scenario()
    assert [r['region_id'] for r in data['rows']] == REGIONS and data['rows'][0]['region'] == 'es'
    assert data['rows'][0]['seats'] == [a + b for a, b in zip(data['rows'][1]['seats'], data['rows'][2]['seats'])]
    assert list(frame.columns) == ['region_id', 'region'] + NAMES


def test_export_projection_follows_the_mode():
    data, frame = publish.export_projection(synthetic_simulator())
    data = valid('projection', data)
    assert data['dates'] == ['2026-10-05'] and set(data['groups']) == {'parties', 'vs', 'blocks'}
    assert data['groups']['parties']['names'] == NAMES and len(data['groups']['parties']['mean']['PP']) == 1
    assert data['groups']['vs']['mean']['Derecha'][0] == pytest.approx(55.0, abs=0.01)
    data, frame = publish.export_projection(synthetic_simulator(mode='forecast'))
    data = valid('projection', data, mode='forecast')
    assert len(data['dates']) == 56 and data['dates'][-1] == '2026-11-29'
    assert list(frame.columns) == ['date', 'group', 'name', 'mean', 'lo', 'hi'] and frame.shape[0] == 56 * (3 + 2 + 2)


def test_headline_mode_combines_vote_and_seats():
    sim = synthetic_simulator(mode='forecast')
    head = bundle.jsonable(publish.headline_mode(sim))
    assert [p['name'] for p in head['parties']] == NAMES
    assert set(head['parties'][0]) == {'name', 'pct', 'lo', 'hi', 'seats', 'seats_lo', 'seats_hi', 'p_first'}
    assert sum(p['seats'] for p in head['parties']) == 10
    assert head['parties'][0]['hi'] > publish.export_vote(synthetic_simulator())[0]['rows'][0]['hi'] - 1e-9
    assert set(head['p_majority']) == {'Derecha', 'Izquierda'}


def test_export_series_stops_at_the_last_fitted_day():
    sim = synthetic_simulator()
    data, frame = publish.export_series(sim.model, NAMES)
    data = valid('series', data, mode=None)
    assert data['dates'][0] == '2026-09-01' and data['dates'][-1] == '2026-10-05'
    assert data['parties'] == NAMES and len(data['mean']['PP']) == len(data['dates'])
    assert data['lo']['PP'][0] == 38.0 and data['hi']['PP'][0] == 42.0
    assert list(frame.columns) == ['date', 'name', 'mean', 'lo', 'hi'] and frame.shape[0] == 35 * 3


def test_export_series_without_statistics_gives_null_bounds():
    sim = synthetic_simulator()
    sim.model.fc_stat['VOX'] = None
    data, _ = publish.export_series(sim.model, NAMES)
    assert valid('series', data, mode=None)['lo']['VOX'] == [None] * 35


def test_export_polls_keeps_the_published_figures_and_the_previous_result():
    sim = synthetic_simulator()
    data, frame = publish.export_polls(sim.model, NAMES)
    data = valid('polls', data, mode=None)
    assert data['columns'] == publish.POLL_COLUMNS and len(data['polls']) == 6
    first = data['polls'][0]
    assert first['date'] == '2026-09-05' and first['pollster'] == 'CIS' and first['sponsor'] is None and first['PP'] == 41.0
    assert data['polls'][2]['VOX'] is None
    assert data['results'] == [{'date': '2023-07-23', 'PP': 33.1, 'PSOE': 31.7, 'VOX': 12.4}]
    seat_columns = [c + n for n in NAMES for c in ('seats_min_', 'seats_max_')]
    assert sorted(frame.columns) == sorted(publish.POLL_COLUMNS + NAMES + seat_columns)
    assert all('seats' not in poll for poll in data['polls']) and frame['seats_min_PP'].isna().all()


def test_export_polls_carries_the_pollsters_seat_projections():
    sim = synthetic_simulator()
    seats = {('2026-09-05', 1, 0): {'PP': [140, 142], 'PSOE': [100, 100], 'OTRO': [1, 1]}, ('2026-09-12', 2, 0): {'VOX': [30, 30]}}
    data, frame = publish.export_polls(sim.model, NAMES, seats=seats)
    data = valid('polls', data, mode=None)
    assert data['polls'][0]['seats'] == {'PP': [140, 142], 'PSOE': [100, 100]}  # solo los partidos exportados
    assert data['polls'][1]['seats'] == {'VOX': [30, 30]} and 'seats' not in data['polls'][2]
    assert frame.loc[0, 'seats_min_PP'] == 140 and frame.loc[0, 'seats_max_PP'] == 142 and pd.isna(frame.loc[0, 'seats_min_VOX'])
    assert frame.loc[1, 'seats_min_VOX'] == 30 and str(frame['seats_min_PP'].dtype) == 'Int64'
    # `poll_seats`: el doble sin `load_poll_seats` no publica nada; con él, lo que devuelva
    assert publish.poll_seats(sim.model) == {}
    sim.model.load_poll_seats = lambda: seats
    assert publish.poll_seats(sim.model) is seats


def test_forecaster_load_poll_seats_pairs_min_and_max_by_poll(monkeypatch):
    import types
    from mtpy.lib import forecaster as fc_module
    frames = {
        'seats_min': pd.DataFrame({'date': pd.to_datetime(['2026-09-05', '2026-09-12']), 'pollster_id': [1, 2], 'sponsor_id': [0, 5],
                                   'pollster': ['CIS', 'GAD3'], 'PP': [140, np.nan], 'PSOE': [100, np.nan]}),
        'seats_max': pd.DataFrame({'date': pd.to_datetime(['2026-09-05', '2026-09-12']), 'pollster_id': [1, 2], 'sponsor_id': [0, 5],
                                   'pollster': ['CIS', 'GAD3'], 'PP': [142, np.nan], 'PSOE': [np.nan, np.nan]}),
    }
    calls = []

    def fake_series(**kwargs):
        calls.append(kwargs)
        return frames[kwargs['metric']].copy()

    monkeypatch.setattr(fc_module, 'get_poll_series', fake_series)
    fake = types.SimpleNamespace(scope='es', date_end=pd.Timestamp('2026-11-29'), drange=(6, None), drop_mtypes=['exit'],
                                 parties=pd.DataFrame({'name': ['PP', 'PSOE', 'VOX']}))
    out = fc_module.Forecaster.load_poll_seats(fake)
    assert out == {('2026-09-05', 1, 0): {'PP': [140, 142], 'PSOE': [100, 100]}}  # sin máximo, el mínimo repetido
    assert [c['metric'] for c in calls] == ['seats_min', 'seats_max']
    assert calls[0]['scope'] == 'es' and calls[0]['event_dates'] == ['2026-11-29'] and calls[0]['drange'] == (6, None)
    assert calls[0]['drop_mtypes'] == ['exit']


def test_export_fan_house_effects_and_dispersion():
    sim = synthetic_simulator()
    fan, _ = publish.export_fan(sim)
    fan = valid('fan', fan, mode=None)
    assert fan['horizons'] == [0, 7, 14, 30, 55] and len(fan['rows']) == 5 * 3
    assert set(fan['rows'][0]) == {'name', 'horizon', 'mean', 'sd', 'lo', 'hi'}
    assert list(publish.export_fan(sim)[1].columns) == ['name', 'horizon', 'mean', 'sd', 'lo', 'hi']
    he, frame = publish.export_house_effects(sim.model)
    he = valid('house-effects', he, mode=None)
    assert len(he['rows']) == 4 and he['rows'][1]['prior'] is None and he['rows'][0]['name'] == 'PP'
    assert list(frame.columns) == publish.HE_COLUMNS
    disp, frame = publish.export_dispersion(sim.model)
    disp = valid('dispersion', disp, mode=None)
    assert [r['pollster_id'] for r in disp['rows']] == [1, 2, 3] and disp['rows'][0]['herd_ratio'] is None
    sim.model.house_effects = sim.model.dispersion = None
    assert publish.export_house_effects(sim.model)[0] == {'rows': []}
    assert list(publish.export_dispersion(sim.model)[1].columns) == publish.DISP_COLUMNS


def test_export_weights_reads_the_as_of_day_and_writes_the_long_csv():
    sim = synthetic_simulator()
    data, frame = publish.export_weights(sim.model, NAMES, sim.as_of)
    data = valid('weights', data, mode=None)
    assert data['date'] == '2026-10-05' and data['parties'] == NAMES
    assert list(frame.columns) == publish.WEIGHTS_COLUMNS and len(frame) == 6 + 6 + 5
    row = data['rows'][0]
    assert row['name'] == 'PP' and row['date'] == '2026-09-05' and row['sponsor'] is None and row['pct'] == 41.0
    assert row['pct_adj'] == 40.7 and row['weight'] == round(0.4 * 1.2, 3) and row['weight_kernel'] == 0.4
    assert data['rows'][6]['name'] == 'PSOE' and {r['name'] for r in data['rows']} == set(NAMES)
    line = data['line']['PP']
    assert line['dates'][0] == '2026-09-05' and line['dates'][-1] == '2026-10-05' and len(line['dates']) == 31
    assert line['mean'][0] == 40.0 and line['lo'][0] == 38.0 and line['hi'][0] == 42.0
    kernel = data['kernel']['PP']
    assert kernel['dates'] == line['dates'] and kernel['values'][-1] == 1.0 and kernel['values'][0] < 1.0
    assert data['stats']['PP'] == {'date': '2026-10-05', 'mean': 40.0, 'lo': 38.0, 'hi': 42.0, 'err': 1.0, 'nobs': 6,
                                   'neff': 4.8, 'bandwidth': 30.0}
    assert data['stats']['VOX']['nobs'] == 5


def test_export_weights_skips_a_party_without_fit():
    sim = synthetic_simulator()
    original = sim.model.weights_at
    sim.model.weights_at = lambda name, date=None: None if name == 'VOX' else original(name, date)
    data, frame = publish.export_weights(sim.model, NAMES, sim.as_of)
    assert data['parties'] == ['PP', 'PSOE'] and set(data['stats']) == {'PP', 'PSOE'} and len(frame) == 12


def test_provenance_reads_git_or_the_image_env(monkeypatch):
    outputs = iter([b'abc1234\n', b' M mtpy/lib/publish.py\n'])
    monkeypatch.setattr(publish.common.subprocess, 'check_output', lambda *a, **k: next(outputs))
    prov = publish.provenance()
    assert prov['commit'] == 'abc1234' and prov['dirty'] is True
    assert set(prov['versions']) == {'python', 'numpy', 'pandas', 'scipy', 'statsmodels'}

    def boom(*a, **k):
        raise OSError('no git')
    monkeypatch.setattr(publish.common.subprocess, 'check_output', boom)
    monkeypatch.setenv('GIT_COMMIT', 'deadbee')
    prov = publish.provenance()
    assert prov['commit'] == 'deadbee' and prov['dirty'] is False


def test_provenance_from_ecs_takes_the_commit_of_the_task_env(monkeypatch):
    """En la imagen de ECS no hay `.git`: el commit sale de `GIT_COMMIT` de la tarea y `dirty` queda `False`."""
    def no_git(*a, **k):
        raise FileNotFoundError('git')
    monkeypatch.setattr(publish.common.subprocess, 'check_output', no_git)
    monkeypatch.setenv('GIT_COMMIT', 'abc1234')
    prov = publish.provenance()
    assert prov['commit'] == 'abc1234' and prov['dirty'] is False
    monkeypatch.delenv('GIT_COMMIT')
    assert publish.provenance()['commit'] is None


def test_next_event_date_forwards_date_from(monkeypatch):
    """`next_event_date` pasa `date_from` a `get_next_event_date` (por defecto `None`: desde ahora)."""
    seen = []
    monkeypatch.setattr(publish.common, 'get_next_event_date',
                        lambda scope, date_from=None: seen.append((scope, date_from)))
    publish.next_event_date('es')
    publish.next_event_date('es', date_from='2026-11-28')
    assert seen == [('es', None), ('es', '2026-11-28')]


def test_export_meta_and_headline():
    sim = synthetic_simulator()
    prov = {'commit': 'abc1234', 'dirty': False, 'versions': {'python': '3.11.9'}}
    meta = publish.export_meta(sim, '20261008-120000', '2026-10-08T12:00:00Z', n_sim=200, max_fc=10,
                               correctors=publish.DEFAULT_CORRECTORS, seconds={'init': 1.0, 'fit': 2.0, 'nowcast': 3.0,
                               'forecast': 3.0, 'export': 1.0, 'total': 10.0}, clip_rate={'nowcast': 0.0, 'forecast': 0.01},
                               db_polls=7, db_last_poll='2026-10-01', prov=prov, freeze=False)
    meta = valid('meta', meta, mode=None)
    assert meta['run_id'] == '20261008-120000' and meta['commit'] == 'abc1234' and meta['dirty'] is False
    assert meta['as_of'] == '2026-10-05' and meta['date_last'] == '2026-10-01' and meta['horizon_max'] == 55
    assert meta['drange'] == [6, None] and meta['n_polls'] == 6 and meta['n_pollsters'] == 3
    assert meta['n_seats'] == 10 and meta['majority'] == 6
    assert meta['parties'][0] == {'name': 'PP', 'id': 1, 'fullname': 'Partido Popular', 'color': '#1d84ce', 'block': 'Derecha', 'regional': 0}
    assert meta['regions'][1] == {'id': 28, 'name': 'Madrid', 'seats': 6}
    assert meta['diagnostics'] == {'drift_k': None, 'multiplier': None, 'ages': {'PP': 40.0, 'PSOE': 40.0, 'VOX': 12.0},
                                   'composition': 1.0, 'clip_rate': {'nowcast': 0.0, 'forecast': 0.01}}
    assert meta['smap'] == {'UP': [{'agg': ['UP', 'MP']}]} and 'vs' in meta['bmaps']

    head = publish.export_headline(sim, '20261008-120000', '2026-10-08T12:00:00Z', {'parties': [], 'p_majority': {}},
                                   {'parties': [], 'p_majority': {}})
    head = valid('headline', head, mode=None)
    assert head['as_of'] == '2026-10-05' and head['n_polls'] == 6 and head['event_date'] == '2026-11-29'


def test_export_meta_and_headline_carry_the_backfill_keys():
    sim = synthetic_simulator()
    meta = publish.export_meta(sim, RUN, '2026-10-05T12:00:00Z', 50, 10, publish.DEFAULT_CORRECTORS, {}, {}, 0, None,
                               {'commit': None, 'dirty': False, 'versions': {}}, False, limit_date='2026-10-05')
    assert meta['limit_date'] == '2026-10-05' and meta['run_at'] == '2026-10-05T12:00:00Z'
    head = publish.export_headline(sim, RUN, '2026-10-05T12:00:00Z', {}, {}, backfill=True)
    assert head['backfill'] is True
    assert publish.export_headline(sim, RUN, '2026-10-05T12:00:00Z', {}, {})['backfill'] is False
    assert bundle.validate('headline', bundle.envelope('headline', head, 'es', run_id=RUN))  # claves nuevas en el esquema


RUN = '20261008-120000'


def fixed_clock():
    return datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def make_writer(tmp_path, fs=None):
    return bundle.BundleWriter(fs or FileSystem(str(tmp_path)), clock=fixed_clock)


def fake_stats(scope, event_date):
    return 7, '2026-10-01'


def fake_prov():
    return {'commit': 'abc1234', 'dirty': False, 'versions': {'python': '3.11.9'}}


def publish_es(tmp_path, run_id=RUN, fs=None, factory=None):
    writer = make_writer(tmp_path, fs)
    factory = factory or FakeSimulator()
    result = publish.publish_forecast('es', writer, run_id, '2026-11-29', n_sim=50, simulator=factory, stats=fake_stats, prov=fake_prov)
    return writer, factory, result


def test_publish_forecast_writes_the_whole_layout_in_order(tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    writer, factory, result = publish_es(tmp_path, fs=fs)
    root = 'site/v1/runs/es/' + RUN + '/'
    expected = {root + p + '.json' for p in bundle.RUN_PARTS} | {root + 'csv/' + p + '.csv' for p in bundle.RUN_PARTS[2:]}
    for mode in bundle.MODES:
        expected |= {root + mode + '/' + p + '.json' for p in bundle.MODE_PARTS}
        expected |= {root + 'csv/' + mode + '-' + p + '.csv' for p in bundle.MODE_PARTS}
    assert set(fs.written) == expected
    assert fs.written[-1] == root + 'headline.json' and fs.written[-2] == root + 'meta.json'
    assert [c[0] for c in factory.calls] == ['init', 'fit_forecast', 'run', 'run']
    assert factory.calls[0][3]['mode'] == 'nowcast' and factory.calls[0][3]['house_effects'] is True
    assert factory.calls[1][1] == {'names': NAMES, 'max_fc': 10, 'fillna': True}
    assert factory.calls[2][1:] == ('nowcast', {'split': True, 'random': True, 'n_sim': 50})
    assert factory.calls[3][1:] == ('forecast', {'split': True, 'random': True, 'n_sim': 50, 'horizon': 'deadline'})
    assert result['entry'] == {'latest': RUN, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
                               'date_last': '2026-10-01', 'n_polls': 6}
    assert set(result['seconds']) == {'init', 'fit', 'nowcast', 'forecast', 'export', 'total'}
    assert all(v >= 0 and round(v, 1) == v for v in result['seconds'].values())
    meta = writer.read_json(bundle.path_part('es', RUN, 'meta'))['data']
    assert meta['n_sim'] == 50 and meta['db_polls'] == 7 and meta['commit'] == 'abc1234' and meta['freeze'] is False
    assert set(meta['diagnostics']['clip_rate']) == {'nowcast', 'forecast'}
    for name in fs.written:
        if name.endswith('.json'):
            env = bundle.loads(fs.read_bytes(name))
            bundle.validate(env['schema'].split('@')[0], env)
    assert not writer.exists(bundle.path_history('es')) and not writer.exists(bundle.path_manifest())


def test_publish_forecast_refuses_to_rewrite_a_run(tmp_path):
    publish_es(tmp_path)
    factory = FakeSimulator()
    with pytest.raises(FileExistsError):
        publish_es(tmp_path, factory=factory)
    assert factory.calls == []


def test_publish_forecast_backfill_freezes_the_polls_and_stamps_the_day(tmp_path):
    writer, factory = make_writer(tmp_path), FakeSimulator()
    out = publish.publish_forecast('es', writer, '20261005-120000', '2026-11-29', n_sim=50, simulator=factory,
                                   stats=fake_stats, prov=fake_prov, limit_date='2026-10-05', run_at='2026-10-05T12:00:00Z')
    assert factory.calls[0][3]['limit_date'] == '2026-10-05'
    assert factory.calls[0][3]['drange'] == (55, None)  # 2026-10-05 a 2026-11-29: solo sondeos hasta ese día
    meta = writer.read_json(bundle.path_part('es', '20261005-120000', 'meta'))['data']
    head = writer.read_json(bundle.path_part('es', '20261005-120000', 'headline'))['data']
    assert meta['limit_date'] == '2026-10-05' and meta['run_at'] == '2026-10-05T12:00:00Z'
    weights = writer.read_json(bundle.path_part('es', '20261005-120000', 'weights'))['data']
    assert weights['date'] == head['as_of'] == '2026-10-05'
    assert head['backfill'] is True and out['entry']['run_at'] == '2026-10-05T12:00:00Z'
    normal = publish.publish_forecast('es', writer, RUN, '2026-11-29', n_sim=50, simulator=factory, stats=fake_stats, prov=fake_prov)
    assert 'limit_date' not in [c for c in factory.calls if c[0] == 'init'][-1][3]
    assert [c for c in factory.calls if c[0] == 'init'][-1][3]['drange'] == 6
    assert writer.read_json(bundle.path_part('es', RUN, 'headline'))['data']['backfill'] is False
    assert writer.read_json(bundle.path_part('es', RUN, 'meta'))['data']['limit_date'] is None
    assert normal['entry']['run_at'] == '2026-10-08T12:00:00Z'


def test_history_is_rebuilt_from_complete_runs_and_is_idempotent(tmp_path):
    writer, _, _ = publish_es(tmp_path)
    publish_es(tmp_path, run_id='20261009-120000')
    writer.write_json(bundle.path_part('es', '20261010-120000', 'meta'), 'meta',
                      writer.read_json(bundle.path_part('es', RUN, 'meta'))['data'], 'es', run_id='20261010-120000')
    history = publish.rebuild_history(writer, 'es')
    assert [r['run_id'] for r in history['runs']] == [RUN, '20261009-120000']
    assert set(history['runs'][0]) >= {'nowcast', 'forecast', 'as_of', 'n_polls'}
    first = (tmp_path / 'site' / 'v1' / 'runs' / 'es' / 'history.json').read_bytes()
    publish.rebuild_history(writer, 'es')
    assert (tmp_path / 'site' / 'v1' / 'runs' / 'es' / 'history.json').read_bytes() == first


def test_manifest_merges_scopes_and_freeze(tmp_path):
    writer = make_writer(tmp_path)
    assert publish.read_manifest(writer)['scopes'] == {} and publish.read_manifest(writer)['freeze'] == {'active': False, 'message': None, 'since': None}
    publish.update_manifest(writer, {'es': {'latest': RUN}, 'es-md': {'latest': RUN}})
    data = publish.update_manifest(writer, {'es': {'latest': '20261009-120000'}})
    assert data['scopes'] == {'es': {'latest': '20261009-120000'}, 'es-md': {'latest': RUN}}
    assert data['updated_at'] == '2026-10-08T12:00:00Z' and data['attribution'] == publish.ATTRIBUTION
    data = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda electoral'}, today='2026-11-23')
    assert data['freeze'] == {'active': True, 'message': 'Veda electoral', 'since': '2026-11-23'} and len(data['scopes']) == 2
    assert publish.update_manifest(writer, freeze={'active': False})['freeze'] == {'active': False, 'message': None, 'since': None}
    with pytest.raises(ValueError, match='active'):
        publish.update_manifest(writer, freeze={'message': 'x'})
    bundle.validate('manifest', writer.read_json(bundle.path_manifest()))


def test_loreg_guard_covers_the_national_window_for_every_scope():
    with pytest.raises(publish.PublishRefused, match=r'es-md 2027-05-23: inside the LOREG window \(art\. 69\.7\) of es 2026-11-29'):
        publish.loreg_guard('es-md', '2027-05-23', today='2026-11-25', national='2026-11-29')
    assert publish.loreg_guard('es-md', '2027-05-23', today='2026-11-25', national='2026-11-29', force=True) is True
    assert publish.loreg_guard('es-md', '2027-05-23', today='2026-11-20', national='2026-11-29') is False
    with pytest.raises(publish.PublishRefused, match='of es-md 2026-11-27'):
        publish.loreg_guard('es-md', '2026-11-27', today='2026-11-25')
    with pytest.raises(publish.PublishRefused):
        publish.loreg_guard('es', '2026-11-29', today='2026-11-29')


def test_update_manifest_stamps_since_on_activation(tmp_path):
    writer = make_writer(tmp_path)
    assert publish.read_manifest(writer)['freeze'] == {'active': False, 'message': None, 'since': None}
    on = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')['freeze']
    assert on == {'active': True, 'message': 'Veda', 'since': '2026-11-23'}
    again = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda (2)'}, today='2026-11-24')['freeze']
    assert again['since'] == '2026-11-23' and again['message'] == 'Veda (2)'
    assert publish.update_manifest(writer, freeze={'active': False})['freeze'] == {'active': False, 'message': None, 'since': None}


def test_read_manifest_adds_since_to_an_old_manifest(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_json(bundle.path_manifest(), 'manifest', {
        'contract': bundle.CONTRACT, 'updated_at': None, 'scopes': {}, 'freeze': {'active': True, 'message': 'x'},
        'attribution': dict(publish.ATTRIBUTION)}, scope=None)
    assert publish.read_manifest(writer)['freeze'] == {'active': True, 'message': 'x', 'since': None}


def test_point_and_unpublish(tmp_path):
    writer, _, first = publish_es(tmp_path)
    _, _, second = publish_es(tmp_path, run_id='20261009-120000')
    publish.rebuild_history(writer, 'es')
    publish.update_manifest(writer, {'es': second['entry']})
    with pytest.raises(ValueError, match='not found'):
        publish.point(writer, 'es', '20261010-120000')
    assert publish.point(writer, 'es', RUN)['scopes']['es'] == first['entry']
    publish.update_manifest(writer, {'es': second['entry'], 'es-md': {'latest': RUN}})
    out = publish.unpublish(writer, 'es', '20261009-120000')
    assert out == {'removed': '20261009-120000', 'latest': RUN}
    assert not writer.exists(bundle.path_run('es', '20261009-120000'))
    manifest = publish.read_manifest(writer)
    assert manifest['scopes']['es'] == first['entry'] and manifest['scopes']['es-md'] == {'latest': RUN}
    assert [r['run_id'] for r in writer.read_json(bundle.path_history('es'))['data']['runs']] == [RUN]
    assert publish.unpublish(writer, 'es', RUN) == {'removed': RUN, 'latest': None}
    assert 'es' not in publish.read_manifest(writer)['scopes']
    assert writer.read_json(bundle.path_history('es'))['data']['runs'] == []
    with pytest.raises(ValueError, match='not found'):
        publish.unpublish(writer, 'es', RUN)


def test_loreg_guard_blocks_an_election_in_the_five_days_before_it():
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-23') is False
    assert publish.loreg_guard('es-md', '2026-11-29', today='2026-11-18') is False
    with pytest.raises(publish.PublishRefused, match='LOREG'):
        publish.loreg_guard('es', '2026-11-29', today=date(2026, 11, 24))
    with pytest.raises(publish.PublishRefused):
        publish.loreg_guard('es', '2026-11-29', today='2026-11-29')
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-28', force=True) is True
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-30') is False


@pytest.mark.parametrize('spec, days', [
    ({'from': '2026-10-05', 'to': '2026-10-07'}, ['2026-10-05', '2026-10-06', '2026-10-07']),
    ({'from': '2026-10-05'}, ['2026-10-05']),
    ({'from': '2026-10-05', 'to': '2026-10-08'}, ['2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08']),
])
def test_backfill_days_expands_the_range(spec, days):
    assert publish.backfill_days(spec, today='2026-10-09') == days
    assert publish.backfill_run_id('2026-10-05') == '20261005-120000'
    assert publish.backfill_run_at('2026-10-05') == '2026-10-05T12:00:00Z'


@pytest.mark.parametrize('spec', [
    {'from': '2026-10-07', 'to': '2026-10-05'}, {'from': '05/10/2026'}, {'to': '2026-10-05'}, 'ayer',
    {'from': '2026-10-05', 'to': '2026-10-10'}, {'from': '2026-10-05', 'to': '2026-10-09'}, {'from': '2026-10-09'},
])
def test_backfill_days_rejects_bad_ranges(spec):
    with pytest.raises(ValueError, match='publish: backfill'):
        publish.backfill_days(spec, today='2026-10-09')


def test_resolve_scopes_uses_the_catalogue():
    catalogue = pd.DataFrame({'parent': [None, 'es', 'es']}, index=pd.Index(['es', 'es-an', 'es-md'], name='scode'))
    assert publish.resolve_scopes('all', catalogue) == ['es', 'es-an', 'es-md']
    assert publish.resolve_scopes('es-md', catalogue) == ['es-md']
    assert publish.resolve_scopes(['es-md', 'es'], catalogue) == ['es-md', 'es']
    with pytest.raises(ValueError, match='es-xx'):
        publish.resolve_scopes(['es-xx'], catalogue)


def test_publish_forecast_turns_a_scope_without_polls_into_nothing_to_publish(tmp_path):
    """Un `ValueError` al construir o ajustar el promedio es `NothingToPublish` (ámbito omitido)."""
    factory = FakeSimulator(fail=ValueError('No polls for es-cb 2027-05-23: nothing to forecast'))
    with pytest.raises(publish.NothingToPublish, match='No polls for es-cb 2027-05-23: nothing to forecast'):
        publish_es(tmp_path, factory=factory)


def test_publish_forecast_lets_a_later_value_error_through(tmp_path, monkeypatch):
    """Un `ValueError` posterior al ajuste (validación del paquete) no se disfraza de `NothingToPublish`."""
    def broken(sim):
        raise ValueError('fan: missing keys [\'rows\']')

    monkeypatch.setattr(publish.forecast, 'export_fan', broken)
    with pytest.raises(ValueError, match='fan: missing keys') as info:
        publish_es(tmp_path)
    assert not isinstance(info.value, publish.NothingToPublish)


@pytest.mark.parametrize('bad', ['/', '..', '.', '2026-10-08'])
def test_point_and_unpublish_reject_an_invalid_run_id(tmp_path, bad):
    """Un `run_id` que no es `YYYYMMDD-HHMMSS` no llega a formar una ruta: `unpublish('/')` borraba el ámbito."""
    writer, _, _ = publish_es(tmp_path)
    publish_es(tmp_path, run_id='20261009-120000')
    for action in (publish.unpublish, publish.point):
        with pytest.raises(ValueError, match='invalid run id'):
            action(writer, 'es', bad)
    assert writer.exists(bundle.path_run('es', RUN)) and writer.exists(bundle.path_run('es', '20261009-120000'))
    assert writer.list_runs('es') == [RUN, '20261009-120000']


def test_point_and_unpublish_reject_an_invalid_scope(tmp_path):
    writer, _, _ = publish_es(tmp_path)
    for action in (publish.point, publish.unpublish):
        with pytest.raises(ValueError, match='invalid scope'):
            action(writer, 'es/..', RUN)
    assert writer.exists(bundle.path_run('es', RUN))


class UnreadableFileSystem(FileSystem):
    """FileSystem cuyo `exists` se traga el error (como fsspec) y cuya lectura falla con permisos."""

    def exists(self, name):
        return False

    def read_bytes(self, name):
        raise PermissionError(name)


def test_read_manifest_only_defaults_when_the_manifest_is_missing(tmp_path):
    """Un error de lectura se propaga: `update_manifest` no debe pisar el manifest real con el de defecto."""
    assert publish.read_manifest(make_writer(tmp_path))['scopes'] == {}
    with pytest.raises(PermissionError):
        publish.read_manifest(bundle.BundleReader(UnreadableFileSystem(str(tmp_path))))


def test_manifest_refreshes_the_attribution(tmp_path, monkeypatch):
    writer = make_writer(tmp_path)
    publish.update_manifest(writer, {'es': {'latest': RUN}})
    monkeypatch.setitem(publish.ATTRIBUTION, 'model', 'x')
    publish.update_manifest(writer, {'es-md': {'latest': RUN}})
    assert writer.read_json(bundle.path_manifest())['data']['attribution']['model'] == 'x'


def test_loreg_guard_normalises_a_datetime():
    with pytest.raises(publish.PublishRefused):
        publish.loreg_guard('es', '2026-11-29', today=datetime(2026, 11, 24, 23, 30, tzinfo=timezone.utc))


def test_madrid_time_zone_and_its_fallback(monkeypatch):
    """La guarda usa la fecha de Madrid; sin tzdata cae a UTC+1 fijo con un aviso."""
    assert publish.common._madrid_tz().utcoffset(datetime(2026, 11, 24, 12, 0)) == timedelta(hours=1)

    def missing(key):
        raise zoneinfo.ZoneInfoNotFoundError(key)

    monkeypatch.setattr(publish.common, 'ZoneInfo', missing)
    with pytest.warns(UserWarning, match='Europe/Madrid'):
        tz = publish.common._madrid_tz()
    assert tz.utcoffset(datetime(2026, 7, 1, 12, 0)) == timedelta(hours=1)


def test_check_run_rejects_a_trailing_newline(tmp_path):
    """Un `run_id` o un `scope` con salto de línea o espacio final se rechaza."""
    writer, _, _ = publish_es(tmp_path)
    for bad in ('20261008-120000\n', '20261008-120000 '):
        with pytest.raises(ValueError, match='invalid run id'):
            publish.point(writer, 'es', bad)
    with pytest.raises(ValueError, match='invalid scope'):
        publish.unpublish(writer, 'es\n', RUN)
    assert writer.exists(bundle.path_run('es', RUN))


def test_fit_forecast_value_error_is_nothing_to_publish(tmp_path):
    """El `ValueError` de `fit_forecast` se convierte en `NothingToPublish`."""
    factory = FakeSimulator()

    def failing(scope, event_date, **kwargs):
        sim = factory(scope, event_date, **kwargs)

        def fit_forecast(**kw):
            raise ValueError('Not enough polls to fit the average of es 2026-11-29')
        sim.fit_forecast = fit_forecast
        return sim
    with pytest.raises(publish.NothingToPublish, match='Not enough polls'):
        publish.publish_forecast('es', make_writer(tmp_path), RUN, '2026-11-29', n_sim=5, simulator=failing,
                                 stats=fake_stats, prov=fake_prov)


def test_analysis_exporters_round_convert_and_keep_the_nulls():
    """Los exportadores del análisis redondean (2 decimales en puntos, 3 en ratios y probabilidades), pasan
    fechas a ISO y enteros a int, y dejan null donde el modelo no tiene valor."""
    frames = synthetic_analysis()
    data, frame = publish.export_analysis_house_effects(frames.house_effects, NAMES + ['UP'], ['Derecha', 'Izquierda'])
    data = valid('analysis-house-effects', data, mode=None)
    assert data['parties'] == NAMES and data['blocks'] == ['Derecha', 'Izquierda'] and len(data['rows']) == 15
    rows = {(r['pollster'], r['name']): r for r in data['rows']}
    assert rows[('CIS', 'PP')] == {'pollster': 'CIS', 'name': 'PP', 'n_events': 2, 'hist': 1.23, 'trend': 0.35,
                                   'cycle': 1.01, 'n_cycle': 6}
    assert rows[('GAD3', 'PP')]['trend'] is None and rows[('GAD3', 'PP')]['n_cycle'] is None
    assert rows[('40dB', 'Derecha')]['n_events'] == 0 and rows[('40dB', 'Derecha')]['hist'] is None
    assert list(frame.columns) == ['pollster', 'name', 'n_events', 'hist', 'trend', 'cycle', 'n_cycle']

    data, summary, events = publish.export_analysis_herding(frames.herding_summary, frames.herding)
    data = valid('analysis-herding', data, mode=None)
    assert data['rows'][0] == {'pollster': '40dB', 'herding': 0.812, 'herding_last': 0.812, 'herding_p': 0.046,
                               'herding_n': 4, 'herding_events': 1}
    assert [(e['event_date'], e['pollster']) for e in data['events']] == [
        ('2023-07-23', 'CIS'), ('2023-07-23', 'GAD3'), ('2026-11-29', '40dB'), ('2026-11-29', 'CIS'), ('2026-11-29', 'GAD3')]
    assert data['events'][0] == {'event_date': '2023-07-23', 'pollster': 'CIS', 'n': 15, 'ratio': 0.999, 'p_value': 0.457}
    assert summary.shape[0] == 3 and list(events.columns) == ['event_date', 'pollster', 'n', 'ratio', 'p_value']

    data, frame = publish.export_poll_errors(frames.poll_errors, ['Derecha', 'Izquierda', 'Nadie'])
    data = valid('poll-errors', data, mode=None)
    assert data['blocks'] == ['Derecha', 'Izquierda']
    assert data['rows'][1] == {'event_date': '2023-07-23', 'event': '2023-23J', 'pollster': 'CIS', 'days': 13,
                               'sample_size': None, 'weight': 0.568, 'error_avg': 2.35, 'bias_avg': -0.57,
                               'error_blocks': 4.57, 'bias_blocks': -1.23, 'bias': -0.68, 'Derecha': -2.35, 'Izquierda': 2.22}
    assert data['rows'][0]['sample_size'] == 1200 and list(frame.columns) == list(data['rows'][0])

    data, frame = publish.export_error_data(frames.error_data)
    data = valid('error-data', data, mode=None)
    assert data['rows'][0] == {'date': '2023-07-15', 'pollster': '40dB', 'party': 'VOX', 'regional': 0, 'weeks': 1,
                               'weight': 0.81, 'pct': 12.35, 'error': 0.12}
    assert 'color' not in frame.columns

    meta = publish.export_analysis_meta('es', '2026-11-29', ['2023-07-23'], 3, {'bmap': 'vs'}, fake_prov(), {'total': 1.0})
    meta = valid('analysis-meta', meta, mode=None)
    assert meta['commit'] == 'abc1234' and meta['versions'] == {'python': '3.11.9'} and meta['n_pollsters'] == 3


def test_publish_analysis_writes_the_parts_and_meta_last(tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    out = publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    assert out['scope'] == 'es' and out['n_pollsters'] == 3
    written = [p for p in fs.written if p.startswith('site/v1/analysis/es/')]
    assert written[-1].endswith('/meta.json') and len([p for p in written if p.endswith('.json')]) == 5
    assert len([p for p in written if p.endswith('.csv')]) == 5
    meta = writer.read_json(bundle.path_analysis('es', 'meta'))
    assert meta['schema'] == 'analysis-meta@1' and meta['run_id'] is None and meta['data']['event_date'] == '2026-11-29'
    he = writer.read_json(bundle.path_analysis('es', 'house-effects'))['data']
    assert he['blocks'] == ['Derecha', 'Izquierda'] and {'pollster', 'name', 'hist', 'trend', 'cycle', 'n_events', 'n_cycle'} <= set(he['rows'][0])
    herding = writer.read_json(bundle.path_analysis('es', 'herding'))['data']
    assert [r['pollster'] for r in herding['rows']] and herding['events'][0]['event_date'] == '2023-07-23'
    errors = writer.read_json(bundle.path_analysis('es', 'poll-errors'))['data']
    assert errors['rows'][0]['event_date'] == '2023-07-23' and 'Derecha' in errors['rows'][0]
    assert bundle.BundleReader(fs).analysis_exists('es')


def test_publish_analysis_records_its_parameters_and_the_database(tmp_path):
    """`meta` guarda los parámetros de los errores, las elecciones del Computer y el estado de la base; los
    errores se piden con `bmap='vs'`, `drange=(6, 42)` y `n_last=1`."""
    factory = FakeComputer()
    writer = make_writer(tmp_path)
    out = publish.publish_analysis('es', writer, computer=factory, stats=fake_stats, prov=fake_prov)
    assert factory.calls[0][:3] == ('init', 'es', '.')
    assert factory.calls[1] == ('get_polls_metric', 'error', 'vs', {'drange': (6, 42), 'n_last': 1})
    meta = writer.read_json(bundle.path_analysis('es', 'meta'))['data']
    assert meta['events'] == ['2019-11-10', '2023-07-23', '2026-11-29'] and meta['scope'] == 'es'
    assert meta['params'] == {'bmap': 'vs', 'drange': [6, 42], 'n_last': 1, 'main': NAMES,
                              'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}, 'db_polls': 7,
                              'db_last_poll': '2026-10-01'}
    assert set(meta['seconds']) == {'build', 'house_effects', 'herding', 'poll_errors', 'error_data', 'total'}
    assert out['seconds'] == meta['seconds'] and out['event_date'] == '2026-11-29'
    assert (tmp_path / 'site' / 'v1' / 'analysis' / 'es' / 'csv' / 'herding-events.csv').exists()


def test_publish_analysis_overwrites_and_keeps_the_old_meta_on_failure(tmp_path, monkeypatch):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    first = writer.read_json(bundle.path_analysis('es', 'meta'))['generated_at']
    writer.clock = lambda: datetime(2030, 1, 1, tzinfo=timezone.utc)  # una reescritura cambiaría `generated_at`
    monkeypatch.setattr(publish.analysis, 'export_error_data', lambda frame: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError):
        publish.publish_analysis('es', writer, computer=FakeComputer(), stats=fake_stats, prov=fake_prov)
    assert writer.read_json(bundle.path_analysis('es', 'meta'))['generated_at'] == first
    assert writer.read_json(bundle.path_analysis('es', 'house-effects'))['generated_at'] == '2030-01-01T00:00:00Z'


def test_publish_analysis_turns_a_computer_value_error_into_nothing_to_publish(tmp_path):
    with pytest.raises(publish.NothingToPublish, match='no events'):
        publish.publish_analysis('es-xx', make_writer(tmp_path), computer=FakeComputer(fail=ValueError('no events')),
                                 stats=fake_stats, prov=fake_prov)
    assert not (tmp_path / 'site').exists()


def test_publish_analysis_without_polls_in_the_current_event_is_nothing_to_publish(tmp_path):
    """El `ValueError` de `print_house_effects` (evento en curso sin sondeos) se convierte en `NothingToPublish`
    y no se escribe nada."""
    factory = FakeComputer(fail_effects=ValueError('No polls for es-cb 2027-05-23: nothing to forecast'))
    with pytest.raises(publish.NothingToPublish, match='No polls for es-cb 2027-05-23'):
        publish.publish_analysis('es-cb', make_writer(tmp_path), computer=factory, stats=fake_stats, prov=fake_prov)
    assert not (tmp_path / 'site').exists()


def test_publish_backtest_writes_the_parts_and_meta_last(tmp_path):
    write_backtest_source(str(tmp_path / 'bt'))
    fs = RecordingFileSystem(str(tmp_path / 'site'))
    writer = bundle.BundleWriter(fs)
    out = publish.publish_backtest('es', writer, source=str(tmp_path / 'bt'), prov=fake_prov)
    assert out['scope'] == 'es' and out['n_cases'] == 4 and out['n_events'] == 2
    written = [p for p in fs.written if p.startswith('site/v1/backtest/es/')]
    assert written[-1].endswith('/meta.json')
    assert sum(p.endswith('.json') for p in written) == 6 and sum(p.endswith('.csv') for p in written) == 5
    meta = writer.read_json(bundle.path_backtest('es', 'meta'))
    assert meta['schema'] == 'backtest-meta@1' and meta['run_id'] is None and meta['mode'] is None
    assert meta['data']['events'] == ['2019-11-10', '2023-07-23'] and meta['data']['horizons'] == [6, 30]
    assert meta['data']['params']['n_sim'] == 500 and 'note' not in meta['data']['params']
    assert meta['data']['n_seats_cases'] == 3 and meta['data']['published'] == fake_prov()
    hz = writer.read_json(bundle.path_backtest('es', 'by-horizon'))['data']
    assert hz['columns'][:3] == ['horizon', 'n_cases', 'n_seats_cases']
    assert hz['rows'][0] == {**hz['rows'][0], 'horizon': 6, 'mae_shares': 1.69, 'cov_shares95': 0.957, 'brier_vs': 0.037}
    seats = writer.read_json(bundle.path_backtest('es', 'seats'))['data']['rows']
    assert seats[0]['official'] == 120 and seats[0]['mean'] == 118.43 and seats[0]['seats_valid'] is True
    assert isinstance(seats[0]['official'], int) and isinstance(seats[0]['model'], int)
    assert '120.0' not in (tmp_path / 'site' / 'site' / 'v1' / 'backtest' / 'es' / 'csv' / 'seats.csv').read_text()
    assert seats[-1]['lo50'] is None
    assert bundle.BundleReader(fs).backtest_exists('es')


def test_publish_backtest_without_results_is_nothing_to_publish(tmp_path):
    with pytest.raises(publish.NothingToPublish, match='no backtest results for es-cb'):
        publish.publish_backtest('es-cb', make_writer(tmp_path), source=str(tmp_path / 'empty'))


def test_backtest_source_follows_the_backtest_script():
    assert publish.backtest_source('es', '/x') == '/x'
    assert publish.backtest_source('es-cb', '/x') == os.path.join('/x', 'es-cb')
    assert publish.backtest_source('es') == os.path.join(publish.REPO_ROOT, 'backtest', 'results')


def test_export_backtest_table_keeps_a_missing_boolean_as_null():
    frame = pd.DataFrame({'event_date': ['a', 'b', 'c'], 'seats_valid': [True, np.nan, False],
                          'majority': [False, True, np.nan]})
    data, table = publish.export_backtest_table(frame)
    assert [row['seats_valid'] for row in data['rows']] == [True, None, False]
    assert [row['majority'] for row in data['rows']] == [False, True, None]
    assert all(type(row['seats_valid']) in (bool, type(None)) for row in data['rows'])


def test_prob_columns_stay_float_when_integral():
    frame = pd.DataFrame({'event_date': ['a', 'b'], 'cov_seats95': [1.0, 1.0], 'n_cases': [2.0, 3.0]})
    data, table = publish.export_backtest_table(frame)
    assert [row['cov_seats95'] for row in data['rows']] == [1.0, 1.0]
    assert all(type(row['cov_seats95']) is float for row in data['rows'])
    assert [row['n_cases'] for row in data['rows']] == [2, 3] and table['cov_seats95'].tolist() == [1.0, 1.0]
    assert table.to_csv(index=False).splitlines()[1] == 'a,1.0,2'


def test_export_backtest_table_rounds_types_and_nulls():
    tables = synthetic_backtest()
    data, frame = publish.export_backtest_table(tables.metrics)
    assert data['columns'] == list(tables.metrics.columns) == list(frame.columns)
    first = data['rows'][0]
    assert first['mae_shares'] == 1.97 and first['cov_shares95'] == 0.933 and first['brier_vs'] == 0.037
    assert first['horizon'] == 6 and type(first['horizon']) is int and first['seats_valid'] is True
    assert pd.isna(first['mae_seats']) and frame['mae_shares'].iloc[0] == 1.97
    blocks, _ = publish.export_backtest_table(tables.blocks)
    assert blocks['rows'][0]['majority'] is False and blocks['rows'][0]['p_majority'] == 0.384
    assert blocks['rows'][0]['official_seats'] == 169
    valid('backtest-metrics', data, mode=None)
    assert valid('backtest-metrics', data, mode=None)['rows'][0]['mae_seats'] is None


def test_export_backtest_meta_without_run_json_takes_events_from_the_metrics():
    tables = synthetic_backtest()
    meta = publish.export_backtest_meta('es-cb', {}, tables, '/tmp/x', fake_prov(), {'total': 1.0})
    assert meta['events'] == ['2019-11-10', '2023-07-23'] and meta['horizons'] == [6, 30]
    assert meta['params'] == {} and meta['run_at'] is None and meta['commit'] is None
    assert meta['n_cases'] == 4 and meta['n_seats_cases'] == 3 and meta['source'] == '/tmp/x'
    valid('backtest-meta', meta, mode=None)


def test_publish_backtest_overwrites_and_keeps_the_old_meta_on_failure(tmp_path, monkeypatch):
    write_backtest_source(str(tmp_path / 'bt'))
    writer = bundle.BundleWriter(FileSystem(str(tmp_path / 'site')))
    publish.publish_backtest('es', writer, source=str(tmp_path / 'bt'), prov=fake_prov)
    first = writer.read_json(bundle.path_backtest('es', 'meta'))['generated_at']
    writer.clock = lambda: datetime(2030, 1, 1, tzinfo=timezone.utc)
    real, calls = publish.export_backtest_table, []

    def broken(frame):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError('boom')
        return real(frame)

    monkeypatch.setattr(publish.backtest, 'export_backtest_table', broken)
    with pytest.raises(RuntimeError):
        publish.publish_backtest('es', writer, source=str(tmp_path / 'bt'), prov=fake_prov)
    assert writer.read_json(bundle.path_backtest('es', 'meta'))['generated_at'] == first
    assert writer.read_json(bundle.path_backtest('es', 'metrics'))['generated_at'] == '2030-01-01T00:00:00Z'
    assert writer.read_json(bundle.path_backtest('es', 'seats'))['generated_at'] == first


def test_publish_backtest_converts_the_repository_results(tmp_path):
    """La conversión real de `backtest/results` (datos del repositorio, sin base): seis JSON válidos."""
    root = os.path.join(publish.REPO_ROOT, 'backtest', 'results')
    if not os.path.isfile(os.path.join(root, 'metrics.csv')):
        pytest.skip('backtest/results no está en el repositorio')
    writer = make_writer(tmp_path)
    out = publish.publish_backtest('es', writer, prov=fake_prov)
    meta = writer.read_json(bundle.path_backtest('es', 'meta'))['data']
    assert out['n_events'] == len(meta['events']) == 5 and meta['source'] == 'backtest/results'
    assert sorted(meta['horizons']) == [6, 14, 30, 60, 90, 180]
    official = [row['official'] for row in writer.read_json(bundle.path_backtest('es', 'seats'))['data']['rows']]
    assert official and all(value is None or type(value) is int for value in official)


def event_readers(ev):
    return dict(events=lambda scope: ev.events, results=lambda scope, date, region: ev.results.loc[ev.results.date == date],
                pollsters=lambda: ev.pollsters, seats=lambda scope, date, names: dict(CYCLE_SEATS))


def test_publish_event_writes_the_parts_and_meta_last(tmp_path):
    ev = synthetic_event()
    write_backtest_source(str(tmp_path / 'bt'))
    fs = RecordingFileSystem(str(tmp_path / 'site'))
    writer = bundle.BundleWriter(fs)
    out = publish.publish_event('es', '2023-07-23', writer, computer=FakeComputer(), source=str(tmp_path / 'bt'),
                                prov=fake_prov, **event_readers(ev))
    assert out['n_polls'] == 3 and out['n_pollsters'] == 3 and out['has_model'] is True
    written = [p for p in fs.written if p.startswith('site/v1/events/es/2023-07-23/')]
    assert written[-1].endswith('/meta.json') and sum(p.endswith('.json') for p in written) == 6 and sum(p.endswith('.csv') for p in written) == 7
    meta = writer.read_json(bundle.path_event('es', '2023-07-23', 'meta'))
    assert meta['schema'] == 'event-meta@1' and meta['data']['name'] == 'Generales 2023' and meta['data']['horizons'] == [6, 30]
    assert set(meta['data']['summary']) == {'n_polls', 'n_pollsters', 'error_avg', 'error_blocks', 'bias_blocks', 'bias'}
    assert list(meta['data']['blocks']) == ['Derecha', 'Izquierda'] and meta['data']['main'] == list(BMAPS['main'])
    houses = writer.read_json(bundle.path_event('es', '2023-07-23', 'pollsters'))['data']['rows']
    assert [r['pollster'] for r in houses] == ['GAD3', '40dB', 'CIS']
    assert houses[0] == {**houses[0], 'pollster_id': 2, 'n_polls': 1, 'last_days': 7, 'last_Derecha': 0.68}
    polls = writer.read_json(bundle.path_event('es', '2023-07-23', 'polls'))['data']['rows']
    assert [r['days'] for r in polls] == [7, 8, 13] and polls[1]['pollster_id'] == 3
    cycle = writer.read_json(bundle.path_event('es', '2023-07-23', 'cycle'))
    assert cycle['schema'] == 'event-cycle@1' and cycle['data']['parties'] == NAMES and cycle['data']['columns'] == publish.POLL_COLUMNS
    rows = cycle['data']['polls']
    assert [(r['date'], r['pollster']) for r in rows] == [('2023-07-10', 'CIS'), ('2023-07-15', '40dB'), ('2023-07-16', 'GAD3')]
    assert rows[0]['sample_size'] is None and rows[1]['sponsor'] == 'El País' and rows[1]['sample_size'] == 1200
    assert rows[2]['seats'] == {'PP': [135, 140], 'PSOE': [120, 120]} and 'seats' not in rows[0] and rows[2]['VOX'] is None
    assert rows[0]['weight'] == round(0.81127 * 0.7, 3) and rows[0]['PP'] == 31.2
    twin = pd.read_csv(tmp_path / 'site' / 'site' / 'v1' / 'events' / 'es' / '2023-07-23' / 'csv' / 'cycle.csv')
    assert list(twin.columns)[:len(publish.POLL_COLUMNS)] == publish.POLL_COLUMNS and 'seats_min_PP' in twin.columns
    assert twin['seats_max_PP'].tolist()[-1] == 140 and pd.isna(twin['sample_size'].iloc[0])
    results = writer.read_json(bundle.path_event('es', '2023-07-23', 'results'))['data']
    assert results['n_seats'] == 350 and results['rows'][-1]['party'] == 'Cs' and results['rows'][0]['votes'] >= results['rows'][1]['votes']
    model = writer.read_json(bundle.path_event('es', '2023-07-23', 'model'))['data']
    assert {r['event_date'] for r in model['shares']} == {'2023-07-23'} and model['horizons'] == [6, 30]
    assert bundle.BundleReader(fs).list_events('es') == ['2023-07-23']


def test_publish_event_without_backtest_has_an_empty_model(tmp_path):
    ev = synthetic_event()
    writer = bundle.BundleWriter(FileSystem(str(tmp_path / 'site')))
    write_backtest_source(str(tmp_path / 'bt'))
    publish.publish_event('es', '2023-07-23', writer, computer=FakeComputer(), source=str(tmp_path / 'bt'),
                          prov=fake_prov, **event_readers(ev))
    assert writer.exists(bundle.path_event_csv('es', '2023-07-23', 'model-shares'))
    out = publish.publish_event('es', '2023-07-23', writer, computer=FakeComputer(), source=str(tmp_path / 'none'),
                                prov=fake_prov, **event_readers(ev))
    assert out['has_model'] is False
    assert writer.read_json(bundle.path_event('es', '2023-07-23', 'model'))['data'] == {
        'horizons': [], 'shares': [], 'seats': [], 'blocks': []}
    meta = writer.read_json(bundle.path_event('es', '2023-07-23', 'meta'))['data']
    assert meta['has_model'] is False and meta['horizons'] == []
    for name in ('model-shares', 'model-seats', 'model-blocks'):
        assert not writer.exists(bundle.path_event_csv('es', '2023-07-23', name))


@pytest.mark.parametrize('date, reason', [('2019-11-10', 'no official results for es 2019-11-10'),
                                          ('2026-11-29', '2026-11-29 is not a featured election of es'),
                                          ('2020-01-01', '2020-01-01 is not a featured election of es')])
def test_publish_event_skips_what_cannot_be_published(tmp_path, date, reason):
    with pytest.raises(publish.NothingToPublish, match=reason):
        publish.publish_event('es', date, make_writer(tmp_path), computer=FakeComputer(), prov=fake_prov,
                              **event_readers(synthetic_event()))
    assert not (tmp_path / 'site').exists()


def test_publish_event_turns_a_computer_value_error_into_nothing_to_publish(tmp_path):
    with pytest.raises(publish.NothingToPublish, match='no events'):
        publish.publish_event('es', '2023-07-23', make_writer(tmp_path), computer=FakeComputer(fail=ValueError('no events')),
                              prov=fake_prov, **event_readers(synthetic_event()))
    assert not (tmp_path / 'site').exists()


def test_publish_event_overwrites_and_keeps_the_old_meta_on_failure(tmp_path, monkeypatch):
    ev = synthetic_event()
    write_backtest_source(str(tmp_path / 'bt'))
    writer = bundle.BundleWriter(FileSystem(str(tmp_path / 'site')))
    args = dict(computer=FakeComputer(), source=str(tmp_path / 'bt'), prov=fake_prov, **event_readers(ev))
    publish.publish_event('es', '2023-07-23', writer, **args)
    first = writer.read_json(bundle.path_event('es', '2023-07-23', 'meta'))['generated_at']
    writer.clock = lambda: datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(publish.events, 'export_event_model', lambda source, event_date: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError):
        publish.publish_event('es', '2023-07-23', writer, **args)
    assert writer.read_json(bundle.path_event('es', '2023-07-23', 'meta'))['generated_at'] == first
    assert writer.read_json(bundle.path_event('es', '2023-07-23', 'results'))['generated_at'] == '2030-01-01T00:00:00Z'
    assert writer.read_json(bundle.path_event('es', '2023-07-23', 'model'))['generated_at'] == first


def test_event_dates_to_publish():
    ev = synthetic_event()
    assert publish.event_dates_to_publish('es', 'all', events=lambda scope: ev.events, today='2026-10-11') == ['2019-11-10', '2023-07-23']
    assert publish.event_dates_to_publish('es', '2023-07-23') == ['2023-07-23']
    assert publish.event_dates_to_publish('es', ['2023-07-23', '2019-11-10']) == ['2023-07-23', '2019-11-10']
    with pytest.raises(ValueError):
        publish.event_dates_to_publish('es', '2023-13-01')


def test_event_summary_weights_by_poll_weight():
    frame = synthetic_analysis().poll_errors
    summary = publish.event_summary(frame, ['Derecha', 'Izquierda'])
    weights = frame['weight'].to_numpy()
    assert summary['n_polls'] == 3 and summary['n_pollsters'] == 3
    assert summary['error_blocks'] == round(float(np.average(frame['error_blocks'], weights=weights)), 2)
    assert summary['bias']['Derecha'] == round(float(np.average(frame['Derecha'], weights=weights)), 2)
    assert set(summary['bias']) == {'Derecha', 'Izquierda'}


def test_export_event_cycle_keeps_the_shape_of_the_run_polls_and_skips_unknown_parties():
    data, table = publish.export_event_cycle(synthetic_cycle(), ['PP', 'VOX', 'Nadie'], {('2023-07-10', 1, 0): {'PP': [130, 130]}})
    assert data['parties'] == ['PP', 'VOX'] and set(data['polls'][0]) >= set(publish.POLL_COLUMNS) | {'PP', 'VOX', 'seats'}
    assert data['polls'][0]['seats'] == {'PP': [130, 130]} and all('seats' not in row for row in data['polls'][1:])
    assert [row['pollster_id'] for row in data['polls']] == [1, 3, 2] and data['polls'][0]['sponsor'] is None
    assert 'PSOE' not in data['polls'][0] and table['seats_min_PP'].tolist()[0] == 130 and 'seats_min_PSOE' not in table.columns
    empty, _ = publish.export_event_cycle(synthetic_cycle().iloc[:0], ['PP'])
    assert empty == {'parties': ['PP'], 'columns': publish.POLL_COLUMNS, 'polls': []}


def test_export_event_pollsters_orders_by_error_with_nulls_last():
    frame = synthetic_analysis().poll_errors.copy()
    extra = frame.iloc[[0]].rename(index={'40dB': 'IPSOS'}, level='pollster')
    extra['error_blocks'] = np.nan
    both = pd.concat([frame, extra])
    data, table = publish.export_event_pollsters(both, ['Derecha', 'Izquierda'], {'GAD3': 2})
    assert [r['pollster'] for r in data['rows']] == ['GAD3', '40dB', 'CIS', 'IPSOS']
    assert data['rows'][0]['pollster_id'] == 2 and data['rows'][1]['pollster_id'] is None
    assert list(table.columns) == publish.EVENT_POLLSTER_COLUMNS + ['last_Derecha', 'last_Izquierda']


def test_publish_package_layout():
    """Un submódulo por papel y ninguno tapado por un nombre exportado por el paquete."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(publish.__path__))
    assert names == ['analysis', 'backtest', 'common', 'events', 'forecast', 'pointers']
    for name in names:
        assert inspect.ismodule(getattr(publish, name)), name
    assert publish.loreg_guard.__module__ == 'mtpy.lib.publish.common'
    assert publish.publish_forecast.__module__ == 'mtpy.lib.publish.forecast'
    assert publish.publish_analysis.__module__ == 'mtpy.lib.publish.analysis'
    assert publish.publish_backtest.__module__ == 'mtpy.lib.publish.backtest'
    assert publish.publish_event.__module__ == 'mtpy.lib.publish.events'
    assert publish.update_manifest.__module__ == 'mtpy.lib.publish.pointers'


def test_publish_forecast_reports_each_phase_as_it_ends(tmp_path):
    """`progress(step, seconds)` tras cada fase, en orden; `meta` llega con el titular ya escrito."""
    writer, steps = make_writer(tmp_path), []

    def progress(step, seconds):
        steps.append((step, seconds, writer.exists(bundle.path_part('es', RUN, 'headline'))))

    result = publish.publish_forecast('es', writer, RUN, '2026-11-29', n_sim=50, simulator=FakeSimulator(),
                                      stats=fake_stats, prov=fake_prov, progress=progress)
    assert [s[0] for s in steps] == ['init', 'fit', 'nowcast', 'export-nowcast', 'forecast', 'export', 'meta']
    assert all(s[1] >= 0 for s in steps) and [s[2] for s in steps] == [False] * 6 + [True]
    assert set(result['seconds']) == {'init', 'fit', 'nowcast', 'forecast', 'export', 'total'}
    exports = dict((s[0], s[1]) for s in steps)
    assert result['seconds']['export'] == round(exports['export-nowcast'] + exports['export'], 1)
