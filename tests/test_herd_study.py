"""Tests de `mtpy/lib/studies/herd/study.py`: el estudio del rebaño entero sobre el paquete sintético de
`write_herd_bundle`, con las cifras calculadas a mano."""
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.herd.data import MissingInput
from mtpy.lib.studies.herd.study import COMPARISONS, _since_year, compute_herd
from tests.fakes import write_herd_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def compute(tmp_path, build=write_herd_bundle, params=None):
    """Study over a synthetic bundle in ``tmp_path`` (``build`` writes it), with the repository ``data/``."""
    fs = FileSystem(str(tmp_path))
    build(fs)
    data = FileSystem(os.path.join(ROOT, 'data'))

    return compute_herd(bundle.BundleReader(fs), data, params)


def test_compute_herd_weeks(fresh_app, tmp_path):
    weeks = compute(tmp_path).weeks
    assert [b['label'] for b in weeks['buckets']] == ['27-52', '18-26', '12-17', '8-11', '5-7', '3-4', '0-2']
    rows = {(r['label'], r['lo']): r for r in weeks['rows']}
    assert {r['label'] for r in weeks['rows']} == {'Generales 2019', 'Generales 2023', 'Generales 2026'}
    assert (rows[('Generales 2019', 5)]['pairs'], rows[('Generales 2019', 5)]['ratio']) == (2, 1.06)
    assert rows[('Generales 2019', 0)]['ratio'] == 0.35
    assert rows[('Generales 2019', 27)]['pairs'] == 1 and rows[('Generales 2019', 27)]['ratio'] is None
    assert (rows[('Generales 2023', 0)]['pairs'], rows[('Generales 2023', 0)]['ratio']) == (3, 1.06)
    assert rows[('Generales 2026', 12)]['current'] is True and rows[('Generales 2026', 12)]['ratio'] == 1.41
    assert rows[('Generales 2026', 3)]['ratio'] is None
    mean = {m['lo']: (m['k'], m['ratio']) for m in weeks['mean']}
    assert mean[0] == (2, 0.7) and mean[5] == (1, 1.06) and mean[27] == (0, None)
    assert len(weeks['rows']) == 3 * 7


def test_compute_herd_weeks_without_elections_of_the_scope(fresh_app, tmp_path):
    weeks = compute(tmp_path, params={'weeks': {'scope': 'es', 'since': '2030-01-01', 'min_pairs': 2,
                                                'buckets': [[0, 2], [27, 52]]}}).weeks
    assert [r['label'] for r in weeks['rows']] == ['Generales 2026'] * 2 and weeks['mean'] == [
        {'lo': 27, 'hi': 52, 'k': 0, 'ratio': None}, {'lo': 0, 'hi': 2, 'k': 0, 'ratio': None}]


def test_compute_herd_weeks_skips_a_current_election_of_another_scope(fresh_app, tmp_path):
    weeks = compute(tmp_path, params={'weeks': {'scope': 'es-md', 'since': '2000-01-01', 'min_pairs': 2,
                                                'buckets': [[0, 2]]}}).weeks
    assert [r['label'] for r in weeks['rows']] == ['Madrid 2021'] and not any(r['current'] for r in weeks['rows'])


def test_comparisons_constant():
    assert COMPARISONS == {'far_final': ('far', 'final'), 'far_mid': ('far', 'mid'), 'mid_final': ('mid', 'final')}


def test_compute_herd_rows(fresh_app, tmp_path):
    rows = {(r['label'], r['comparison']): r for r in compute(tmp_path).elections}
    assert set(rows) == {('Generales 2019', 'far_final'), ('Generales 2019', 'far_mid'),
                         ('Generales 2019', 'mid_final'), ('Generales 2023', 'far_final')}
    g19 = rows[('Generales 2019', 'far_final')]
    assert (g19['pairs'], g19['rows_a'], g19['rows_b'], g19['r_a'], g19['r_b'], g19['change'], g19['falls']) == (
        1, 4, 4, 1.41, 0.35, -0.749, True)
    assert abs(g19['p'] - 0.214) < 0.03           # exact 15/70
    assert rows[('Generales 2019', 'far_mid')]['change'] == -0.248
    assert rows[('Generales 2019', 'mid_final')]['change'] == -0.667
    g23 = rows[('Generales 2023', 'far_final')]
    assert (g23['r_a'], g23['r_b'], g23['change'], g23['falls'], g23['kind']) == (0.35, 1.06, 1.986, False, 'generales')
    assert set(g19) == {'scope', 'event_date', 'label', 'kind', 'comparison', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b',
                        'change', 'falls', 'p'}


def test_compute_herd_summary(fresh_app, tmp_path):
    s = compute(tmp_path).summary
    assert (s['n_elections'], s['n_scopes'], s['since_year']) == (2, 1, '2019')
    assert s['comparisons']['far_final'] == {'n': 2, 'k': 1, 'sign_p': 0.75, 'clear': 0, 'median_change': 0.62,
                                             'since_year': '2019'}
    assert s['comparisons']['far_mid'] == {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.25,
                                           'since_year': '2019'}
    assert s['comparisons']['mid_final'] == {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.67,
                                             'since_year': '2019'}
    # exact permutation p: far_mid 23/70, mid_final 15/70: neither is clear
    assert s['kinds']['autonómicas'] == {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}
    assert s['kinds']['generales']['n'] == 2


def test_compute_herd_current(fresh_app, tmp_path):
    study = compute(tmp_path)
    c = study.current
    assert (c['label'], c['days_left'], c['windows']) == ('Generales 2026', 24,
                                                           {'mid': '2026-10-05', 'final': '2026-11-09'})
    assert c['far_mid']['change'] == -0.498 and c['mid_final'] is None
    assert study.election == {'scope': 'es', 'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05'}
    assert study.inputs['current']['run_id'] and 'es-md' in study.inputs['elections']


def test_compute_herd_without_current(fresh_app, tmp_path):
    def build(fs):
        write_herd_bundle(fs)
        writer = bundle.BundleWriter(fs)
        cycle = {'parties': ['PP', 'PSOE'], 'columns': [], 'polls': []}
        writer.write_json(bundle.path_event('es', '2026-11-29', 'cycle'), 'event-cycle', cycle, 'es')
        writer.write_json(bundle.path_event('es', '2026-11-29', 'results'), 'event-results',
                          {'n_seats': 350, 'rows': []}, 'es')

    study = compute(tmp_path, build)
    assert study.current is None and study.election is None


def test_compute_herd_without_elections_raises(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    data = FileSystem(os.path.join(ROOT, 'data'))
    with pytest.raises(MissingInput) as error:
        compute_herd(bundle.BundleReader(fs), data)
    assert error.value.part == 'events'


def test_since_year_is_the_first_year_of_the_dates():
    assert _since_year(['2023-07-23', '1986-06-22', '2019-04-28']) == '1986' and _since_year([]) is None
