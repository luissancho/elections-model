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


def test_compute_herd_weeks_without_comparisons(fresh_app, tmp_path):
    weeks = compute(tmp_path, params={'pair_days': 0}).weeks
    assert len(weeks['rows']) == 3 * 7 and all(r['ratio'] is None and r['pairs'] == 0 for r in weeks['rows'])
    assert all(m['k'] == 0 and m['ratio'] is None for m in weeks['mean'])


def test_comparisons_constant():
    assert COMPARISONS == {'far_final': ('far', 'final'), 'far_mid': ('far', 'mid'), 'mid_final': ('mid', 'final')}


def test_compute_herd_rows(fresh_app, tmp_path):
    rows = {(r['label'], r['comparison']): r for r in compute(tmp_path).elections}
    assert set(rows) == {('Generales 2019', 'far_final'), ('Generales 2019', 'far_mid'),
                         ('Generales 2019', 'mid_final'), ('Generales 2023', 'far_final')}
    g19 = rows[('Generales 2019', 'far_final')]
    assert (g19['pairs'], g19['rows_a'], g19['rows_b'], g19['r_a'], g19['r_b'], g19['change'], g19['falls']) == (
        1, 4, 4, 1.41, 0.35, -0.749, True)
    assert abs(g19['p'] - 1 / 6) < 0.03           # exact 1/6: 4 blocks (2 pairs of polls per period)
    assert rows[('Generales 2019', 'far_mid')]['change'] == -0.248
    assert rows[('Generales 2019', 'mid_final')]['change'] == -0.667
    g23 = rows[('Generales 2023', 'far_final')]
    assert (g23['r_a'], g23['r_b'], g23['change'], g23['falls'], g23['kind']) == (0.35, 1.06, 1.986, False, 'generales')
    assert abs(g23['p'] - 1.0) < 1e-9              # the observed rise is the largest of the 10 arrangements
    assert set(g19) == {'scope', 'event_date', 'label', 'kind', 'comparison', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b',
                        'change', 'falls', 'p'}


def test_compute_herd_summary(fresh_app, tmp_path):
    s = compute(tmp_path).summary
    assert (s['n_elections'], s['n_scopes'], s['since_year']) == (2, 1, '2019')
    tally = ('n', 'k', 'sign_p', 'clear', 'median_change', 'since_year')
    assert {k: s['comparisons']['far_final'][k] for k in tally} == {
        'n': 2, 'k': 1, 'sign_p': 0.75, 'clear': 0, 'median_change': 0.62, 'since_year': '2019'}
    assert {k: s['comparisons']['far_mid'][k] for k in tally} == {
        'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.25, 'since_year': '2019'}
    assert {k: s['comparisons']['mid_final'][k] for k in tally} == {
        'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.67, 'since_year': '2019'}
    ff = s['comparisons']['far_final']
    pooled = ff['pooled']
    assert {k: pooled[k] for k in ('elections', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b', 'change')} == {
        'elections': 2, 'pairs': 2, 'rows_a': 8, 'rows_b': 10, 'r_a': 0.98, 'r_b': 0.81, 'change': -0.173}
    assert abs(pooled['p'] - 22 / 60) < 0.03 and ff['ci'] is None       # 2 elections: no interval
    fm = s['comparisons']['far_mid']
    assert fm['pooled']['change'] == -0.248 and abs(fm['pooled']['p'] - 1 / 6) < 0.03 and fm['ci'] is None
    assert s['comparisons']['mid_final']['pooled']['change'] == -0.667
    assert s['kinds']['autonómicas'] == {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}
    assert s['kinds']['generales']['n'] == 2


def test_compute_herd_robustness(fresh_app, tmp_path):
    study = compute(tmp_path)
    r = study.summary['robustness']
    assert r['pair_days'] == [3, 5, 7, 10] and r['min_rows'] == [3, 6] and len(r['rows']) == 24
    assert [(row['pair_days'], row['min_rows'], row['comparison']) for row in r['rows'][:4]] == [
        (3, 3, 'far_final'), (3, 3, 'far_mid'), (3, 3, 'mid_final'), (3, 6, 'far_final')]
    default = {row['comparison']: row for row in r['rows'] if row['default']}
    s = study.summary['comparisons']
    assert set(default) == set(COMPARISONS)
    assert all(row['pair_days'] == 7 and row['min_rows'] == 3 for row in default.values())
    for key, row in default.items():
        # the summary publishes the median at two decimals, the grid at three
        assert (row['n'], row['k'], round(row['median_change'], 2), row['change'], row['p']) == (
            s[key]['n'], s[key]['k'], s[key]['median_change'], s[key]['pooled']['change'], s[key]['pooled']['p'])
    d3 = next(row for row in r['rows'] if (row['pair_days'], row['min_rows'], row['comparison']) == (3, 3, 'far_final'))
    assert (d3['n'], d3['k'], d3['median_change'], d3['change']) == (2, 1, 0.618, -0.234)
    empty = next(row for row in r['rows']
                 if (row['pair_days'], row['min_rows'], row['comparison']) == (3, 6, 'far_final'))
    assert empty == {'pair_days': 3, 'min_rows': 6, 'default': False, 'comparison': 'far_final', 'n': 0, 'k': 0,
                     'median_change': None, 'change': None, 'p': None}
    ten = next(row for row in r['rows']
               if (row['pair_days'], row['min_rows'], row['comparison']) == (10, 6, 'mid_final'))
    assert (ten['n'], ten['k'], ten['change']) == (1, 1, -0.667)


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


def test_compute_herd_chance(fresh_app, tmp_path):
    c = compute(tmp_path).chance
    assert c['periods'] == {'final': [0, 21], 'rest': [21, 365]} and c['permutations'] == 20000
    assert (c['elections'], c['falls'], c['pairs'], c['rows_a'], c['rows_b'], c['r_a'], c['r_b']) == (
        2, 1, 2, 12, 10, 0.94, 0.77)
    assert c['change'] == pytest.approx(-0.1775, abs=6e-4) and c['ci'] is None
    assert abs(c['p_one'] - 22 / 150) < 0.01 and abs(c['p_two'] - 65 / 150) < 0.015      # exact: 150 arrangements
    null = c['null']
    assert abs(null['min'] - (-0.542)) < 0.01 and abs(null['max'] - 1.015) < 0.01 and null['bin_width'] == 0.01
    assert sum(b['count'] for b in null['bins']) == 20000
    assert null['bins'][0]['lo'] <= null['min'] < null['bins'][0]['hi']
    assert set(c) == {'periods', 'permutations', 'elections', 'falls', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b',
                      'change', 'ci', 'count_one', 'count_two', 'p_one', 'p_two', 'null'}
    assert isinstance(c['count_one'], int) and isinstance(c['count_two'], int)
    assert c['count_one'] >= 0 and c['count_two'] >= 0
    assert c['p_one'] == pytest.approx((c['count_one'] + 1) / 20001, rel=1e-2)
    assert c['p_two'] == pytest.approx((c['count_two'] + 1) / 20001, rel=1e-2)
    # The comparisons of the study relabelled with the periods of the test: the figures are those of recomputing them
    assert (c['count_one'], c['count_two'], c['p_one'], c['p_two']) == (2856, 8696, 0.143, 0.435)
    assert (c['null']['mean'], c['null']['sd'], c['null']['min'], c['null']['max']) == (0.0257, 0.2655, -0.542, 1.015)


def test_compute_herd_chance_is_none_without_shared_pairs(fresh_app, tmp_path):
    params = {'chance': {'periods': {'final': [0, 1], 'rest': [1, 2]}, 'permutations': 100, 'bin_width': 0.01,
                         'resamples': 100}}
    assert compute(tmp_path, params=params).chance is None
