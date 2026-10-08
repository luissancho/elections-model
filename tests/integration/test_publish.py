"""Tests de integración de la publicación: `publish_forecast('es')` con `n_sim=20` contra la base."""
import json
import os

import numpy as np
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish

pytestmark = pytest.mark.integration

EVENT = '2026-11-29'
RUNS = ('20261008-120000', '20261008-120001')


@pytest.fixture(scope='module')
def published(app, tmp_path_factory):
    """Dos publicaciones de `es` con la misma semilla en un paquete temporal; devuelve el escritor."""
    writer = bundle.BundleWriter(FileSystem(str(tmp_path_factory.mktemp('bundle'))))
    for run_id in RUNS:
        publish.publish_forecast('es', writer, run_id, EVENT, n_sim=20, seed=42)
    publish.rebuild_history(writer, 'es')
    return writer


def read(writer, part, mode=None, run_id=RUNS[0]):
    return writer.read_json(bundle.path_part('es', run_id, part, mode))['data']


def test_every_file_validates(published):
    root = os.path.join(published.fs.path, 'site', 'v1')
    count = 0
    for folder, _, files in os.walk(root):
        for name in files:
            if name.endswith('.json'):
                with open(os.path.join(folder, name), 'rb') as fh:
                    env = bundle.loads(fh.read())
                bundle.validate(env['schema'].split('@')[0], env)
                count += 1
    assert count == 2 * (len(bundle.RUN_PARTS) + 2 * len(bundle.MODE_PARTS)) + 1


def test_seats_sum_350_and_probabilities_are_bounded(published):
    for mode in bundle.MODES:
        dist = read(published, 'dist', mode)
        assert dist['n_seats'] == 350 and all(sum(row) == 350 for row in dist['seats']) and len(dist['seats']) == 20
        summary = read(published, 'summary', mode)
        assert sum(summary['totals'].values()) == 350
        assert all(0 <= r[k] <= 1 for r in summary['parties'] for k in ('p_seats', 'p_majority', 'p_first') if r[k] is not None)


def test_parties_belong_to_the_event(published, app):
    with open(os.path.join(app.datapath, 'params.json'), encoding='utf-8') as fh:
        event_parties = set(json.load(fh)['es'][EVENT]['parties']['event'])
    meta = read(published, 'meta')
    assert {p['name'] for p in meta['parties']} <= event_parties
    assert meta['n_seats'] == 350 and meta['event_date'] == EVENT and meta['n_polls'] > 0


def test_forecast_vote_matches_a_forecast_mode_simulator(published):
    from mtpy.lib.simulator import Simulator

    sim = Simulator(scope='es', event_date=EVENT, drange=6, alpha=0.05, seed=42, mode='forecast', verbose=0, path='.')
    sim.fit_forecast(names=sim.params['names'], max_fc=10, fillna=True)
    expected = sim.vote_forecast().round(2)
    rows = {r['name']: r for r in read(published, 'vote', 'forecast')['rows']}
    assert set(rows) == set(expected.index)
    for name, row in expected.iterrows():
        assert rows[name]['pct'] == pytest.approx(row['pct']) and rows[name]['hi'] == pytest.approx(row['hi'])
    assert read(published, 'vote', 'forecast')['horizon'] == sim.horizon_max


def test_same_seed_gives_the_same_dist_and_two_history_entries(published):
    first = read(published, 'dist', 'nowcast', RUNS[0])['seats']
    second = read(published, 'dist', 'nowcast', RUNS[1])['seats']
    assert np.array_equal(first, second)
    history = published.read_json(bundle.path_history('es'))['data']
    assert [r['run_id'] for r in history['runs']] == list(RUNS)
