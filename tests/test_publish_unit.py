"""Tests de `mtpy/lib/publish.py` sin base de datos: exportadores sobre el Simulator sintético de `tests/fakes.py`."""
import math
import zoneinfo
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from tests.fakes import NAMES, REGIONS, FakeSimulator, RecordingFileSystem, synthetic_simulator


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
    assert data['regions'][0] == {'id': 28, 'name': 'Madrid', 'seats': 6}
    assert len(data['rows']) == 2 * len(NAMES)
    assert set(data['rows'][0]) == {'region_id', 'region', 'name', 'pct', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_lo',
                                    'seats_hi', 'p_seats'}
    assert frame.shape[0] == 6
    assert all(isinstance(r['seats'], float) and r['seats'] == round(r['seats'], 1) for r in data['rows'])


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
    assert list(frame.columns) == publish.POLL_COLUMNS + NAMES


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


def test_provenance_reads_git_or_the_image_env(monkeypatch):
    outputs = iter([b'abc1234\n', b' M mtpy/lib/publish.py\n'])
    monkeypatch.setattr(publish.subprocess, 'check_output', lambda *a, **k: next(outputs))
    prov = publish.provenance()
    assert prov['commit'] == 'abc1234' and prov['dirty'] is True
    assert set(prov['versions']) == {'python', 'numpy', 'pandas', 'scipy', 'statsmodels'}

    def boom(*a, **k):
        raise OSError('no git')
    monkeypatch.setattr(publish.subprocess, 'check_output', boom)
    monkeypatch.setenv('GIT_COMMIT', 'deadbee')
    prov = publish.provenance()
    assert prov['commit'] == 'deadbee' and prov['dirty'] is False


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
    assert publish.read_manifest(writer)['scopes'] == {} and publish.read_manifest(writer)['freeze'] == {'active': False, 'message': None}
    publish.update_manifest(writer, {'es': {'latest': RUN}, 'es-md': {'latest': RUN}})
    data = publish.update_manifest(writer, {'es': {'latest': '20261009-120000'}})
    assert data['scopes'] == {'es': {'latest': '20261009-120000'}, 'es-md': {'latest': RUN}}
    assert data['updated_at'] == '2026-10-08T12:00:00Z' and data['attribution'] == publish.ATTRIBUTION
    data = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda electoral'})
    assert data['freeze'] == {'active': True, 'message': 'Veda electoral'} and len(data['scopes']) == 2
    assert publish.update_manifest(writer, freeze={'active': False})['freeze'] == {'active': False, 'message': None}
    with pytest.raises(ValueError, match='active'):
        publish.update_manifest(writer, freeze={'message': 'x'})
    bundle.validate('manifest', writer.read_json(bundle.path_manifest()))


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


def test_loreg_guard_only_blocks_es_in_the_five_days_before_the_election():
    assert publish.loreg_guard('es', '2026-11-29', today='2026-11-23') is False
    assert publish.loreg_guard('es-md', '2026-11-29', today='2026-11-27') is False
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

    monkeypatch.setattr(publish, 'export_fan', broken)
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
    assert publish._madrid_tz().utcoffset(datetime(2026, 11, 24, 12, 0)) == timedelta(hours=1)

    def missing(key):
        raise zoneinfo.ZoneInfoNotFoundError(key)

    monkeypatch.setattr(publish, 'ZoneInfo', missing)
    with pytest.warns(UserWarning, match='Europe/Madrid'):
        tz = publish._madrid_tz()
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
