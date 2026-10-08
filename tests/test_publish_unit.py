"""Tests de `mtpy/lib/publish.py` sin base de datos: exportadores sobre el Simulator sintético de `tests/fakes.py`."""
import math

import numpy as np
import pandas as pd
import pytest

from mtpy.lib import bundle, publish
from tests.fakes import NAMES, REGIONS, synthetic_simulator


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
