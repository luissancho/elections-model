"""Tests de `mtpy/lib/studies/herd/study.py`: el estudio del rebaño entero sobre el paquete sintético de
`write_herd_bundle`, con las cifras calculadas a mano en `tests/test_herd_stats.py`."""
import math
import os
import tempfile

import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.herd.data import MissingInput
from mtpy.lib.studies.herd.study import compute_herd
from tests.fakes import HERD_ERRORS, HERD_IDS, HERD_LEVELS, HERD_RUN, bundle_fixture, write_herd_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ELECTION = '2026-11-29'


def _drop_window(fs):
    """Rewrite the run ``polls`` part without the polls of the final window (ended 55 days or less before the
    election)."""
    reader, writer = bundle.BundleReader(fs), bundle.BundleWriter(fs)
    name = bundle.path_part('es', HERD_RUN, 'polls')
    polls = reader.read_json(name)['data']
    early = [poll for poll in polls['polls'] if (pd.Timestamp(ELECTION) - pd.Timestamp(poll['end_date'])).days > 55]
    writer.write_json(name, 'polls', polls | {'polls': early}, 'es', run_id=HERD_RUN)


def _publish_current(fs):
    """Publish the election of the run with results: ``meta``, ``cycle`` (the run polls), ``results`` and
    ``pollsters`` (a SocioMétrica row)."""
    reader, writer = bundle.BundleReader(fs), bundle.BundleWriter(fs)
    polls = reader.read_json(bundle.path_part('es', HERD_RUN, 'polls'))['data']
    meta = bundle_fixture('event-meta') | {'scope': 'es', 'event_date': ELECTION, 'name': 'Generales 2026'}
    cycle = {'parties': polls['parties'], 'columns': polls['columns'], 'polls': polls['polls']}
    pollsters = {'blocks': ['Derecha', 'Izquierda'], 'rows': [
        {'pollster_id': HERD_IDS['SocioMétrica'], 'pollster': 'SocioMétrica', 'last_error_avg': 2.0,
         'last_error_blocks': 1.0}]}
    for part, schema, content in (('meta', 'event-meta', meta), ('cycle', 'event-cycle', cycle),
                                  ('results', 'event-results', {'n_seats': 350, 'rows': []}),
                                  ('pollsters', 'event-pollsters', pollsters)):
        writer.write_json(bundle.path_event('es', ELECTION, part), schema, content, 'es')


# Two more CIS polls in the final window of 2023 (with the one of 2023-07-13, three: the CIS reaches ``min_polls``)
EXTRA_CIS = [('2023-06-20', (2, -2, 1, -1)), ('2023-07-01', (5, -5, 3, -3))]


def _add_cis_2023(fs):
    """Rewrite the ``cycle`` part of 2023 with the polls of ``EXTRA_CIS``: copies of its CIS poll of 2023-07-13
    (``sample_size`` 4000) with their end date and the curve levels plus their residuals."""
    reader, writer = bundle.BundleReader(fs), bundle.BundleWriter(fs)
    name = bundle.path_event('es', '2023-07-23', 'cycle')
    cycle = reader.read_json(name)['data']
    model = next(poll for poll in cycle['polls'] if poll['pollster'] == 'CIS' and poll['end_date'] == '2023-07-13')
    extra = [model | {'date': end, 'start_date': end, 'end_date': end} |
             {party: float(level + residual)
              for party, level, residual in zip(cycle['parties'], HERD_LEVELS, residuals)}
             for end, residuals in EXTRA_CIS]
    writer.write_json(name, 'event-cycle', cycle | {'polls': cycle['polls'] + extra}, 'es')


def compute(tmp_path, drop_window=False, without_manifest=False, without_events=False, publish_current=False,
            add_cis=False):
    """Write the synthetic bundle in a fresh folder under ``tmp_path``, apply the changes asked and compute the
    study with the repository parameters."""
    fs = FileSystem(tempfile.mkdtemp(dir=str(tmp_path)))
    written = write_herd_bundle(fs)
    writer = bundle.BundleWriter(fs)

    if drop_window:
        _drop_window(fs)
    if without_manifest:
        writer.remove(bundle.path_manifest())
    if without_events:
        for date in written['dates']:
            writer.remove(bundle.path_event('es', date, 'cycle'))
    if publish_current:
        _publish_current(fs)
    if add_cis:
        _add_cis_2023(fs)

    return compute_herd(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')))


def test_compute_herd_on_the_synthetic_bundle(fresh_app, tmp_path):
    study = compute(tmp_path)
    s = study.summary
    assert (s['n_cycles'], s['k_narrowed'], s['narrowed']) == (2, 2, ['2019', '2023'])
    assert (s['mean_before'], s['mean_last'], s['mean_final']) == (2.23, 1.14, 1.05)
    assert (s['k_below_one'], s['below_one'], s['sign']) == (1, ['2019'], {'n': 2, 'k': 2, 'p': 0.25})
    assert s['current'] == {'label': '2026', 'before': {'k': 2, 'ratio': 0.78}, 'last': {'k': 4, 'ratio': 0.78},
                            'final': {'k': 0, 'ratio': None}, 'cycle': {'k': 6, 'ratio': 0.78}, 'days_left': 24,
                            'rank_before': 1, 'like_final': True}
    assert s['houses_now'] == {'n': 1, 'k': 1, 'herd': [{'pollster': 'SocioMétrica', 'h': 0.0}],
                               'pending': [{'pollster': 'Sigma Dos', 'k': 1}]}
    assert s['cis'] == {'current': {'k': 1, 'ratio': 5.66}, 'cycle': {'k': 1, 'ratio': 5.66},
                        'own': {'k': 1, 'h': None, 'p': None}}
    assert s['accuracy'] == {'n': 2, 'herd': {'k': 1, 'error_avg': 1.5, 'error_blocks': 2.0},
                             'rest': {'k': 1, 'error_avg': 2.5, 'error_blocks': 1.0}}
    assert study.election == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05'}
    assert study.params['window_days'] == 56 and set(study.inputs) == {'cycles', 'current', 'herding'}
    assert study.inputs['current']['run_id'] == HERD_RUN and study.inputs['herding']['generated_at']
    assert set(study.inputs['cycles']) == {'2019-11-10', '2023-07-23'}


def test_compute_herd_cycle_rows(fresh_app, tmp_path):
    rows = {row['event_date']: row for row in compute(tmp_path).cycles}
    r23 = rows['2023-07-23']
    assert (r23['label'], r23['current'], r23['names'], r23['consensus']) == ('2023', False,
                                                                              ['PP', 'PSOE', 'VOX', 'SUMAR'], 'curve')
    assert (r23['polls'], r23['with_sample'], r23['houses']) == (14, 10, 3)
    assert (r23['before'], r23['last'], r23['final'], r23['change']) == (
        {'k': 3, 'ratio': 1.57}, {'k': 7, 'ratio': 1.31}, {'k': 4, 'ratio': 1.14}, -0.26)
    assert r23['cis'] == {'last': {'k': 1, 'ratio': 4.96}, 'cycle': {'k': 2, 'ratio': 4.3},
                          'own': {'k': 1, 'h': None, 'p': None}}
    assert rows['2019-11-10']['consensus'] == 'local' and rows['2019-11-10']['last'] == {'k': 3, 'ratio': 0.96}
    assert rows['2026-11-29']['current'] and rows['2026-11-29']['names'] == ['PP', 'PSOE', 'VOX', 'FA']
    assert list(rows) == ['2019-11-10', '2023-07-23', '2026-11-29']


def test_compute_herd_weeks_and_houses(fresh_app, tmp_path):
    study = compute(tmp_path)
    assert [b['label'] for b in study.weeks['buckets']] == ['27-52 sem.', '18-26 sem.', '12-17 sem.', '8-11 sem.',
                                                            '6-7 sem.', '4-5 sem.', '2-3 sem.', '0-1 sem.']
    w23 = {(r['lo'], r['hi']): (r['k'], r['ratio']) for r in study.weeks['rows'] if r['event_date'] == '2023-07-23'}
    # a bucket below ``min_polls`` (3) polls keeps its count without a ratio (it drew 1,75 with 2 and 0,78 with 1)
    assert w23[(0, 1)] == (3, 1.11) and w23[(4, 5)] == (2, None) and w23[(6, 7)] == (1, None)
    assert w23[(27, 52)] == (0, None)
    assert len(study.weeks['rows']) == 3 * 8
    mean = {(r['lo'], r['hi']): (r['k'], r['ratio']) for r in study.weeks['mean']}
    assert mean[(0, 1)] == (2, 1.04) and mean[(12, 17)] == (1, 2.88) and mean[(27, 52)] == (0, None)
    houses = [(h['event_date'], h['pollster'], h['k'], h['h'], h['p'], h['h_cycle'], h['last_error_avg'])
              for h in study.houses]
    assert houses == [('2023-07-23', 'GAD3', 3, 0.32, 0.0975, 0.61, 1.5),
                      ('2023-07-23', 'NC Report', 3, 1.97, 0.98, None, 2.5),
                      ('2026-11-29', 'SocioMétrica', 3, 0.0, 0.0, 0.78, None)]
    assert 'slug' not in study.houses[0] and study.houses[0]['cis'] is False
    assert study.houses[0]['last_error_blocks'] == HERD_ERRORS['2023-07-23']['GAD3'][1]


def test_compute_herd_measures_the_cis_apart_once_it_has_enough_polls(fresh_app, tmp_path):
    """The path of the real 2023: the CIS with ``min_polls`` polls in the final window gets its row in ``houses``
    (``cis: True``) and its own oscillation in ``cycles``, but stays out of ``houses_now`` (the 2026 window), of
    ``accuracy`` and of every pooled figure."""
    study = compute(tmp_path, add_cis=True)
    cis = [house for house in study.houses if house['pollster'] == 'CIS']
    assert [(house['event_date'], house['cis'], house['k']) for house in cis] == [('2023-07-23', True, 3)]
    assert math.isfinite(cis[0]['h']) and math.isfinite(cis[0]['p'])
    s = study.summary
    assert 'CIS' not in [house['pollster'] for house in s['houses_now']['herd']] + \
        [house['pollster'] for house in s['houses_now']['pending']]
    assert s['houses_now']['n'] == 1
    # accuracy as the base pin: the CIS (it has errors in 2023) counts in neither group
    assert s['accuracy'] == {'n': 2, 'herd': {'k': 1, 'error_avg': 1.5, 'error_blocks': 2.0},
                             'rest': {'k': 1, 'error_avg': 2.5, 'error_blocks': 1.0}}
    r23 = {row['event_date']: row for row in study.cycles}['2023-07-23']
    own = r23['cis']['own']
    assert own['k'] == 3 and math.isfinite(own['h']) and math.isfinite(own['p'])
    assert (own['h'], own['p']) == (cis[0]['h'], cis[0]['p'])
    assert (r23['before']['ratio'], r23['last']['ratio'], r23['final']['ratio']) == (1.57, 1.31, 1.14)


def test_compute_herd_with_an_empty_final_window(fresh_app, tmp_path):
    # the run polls of the window removed: the current cycle has only its two early polls
    study = compute(tmp_path, drop_window=True)
    assert study.summary['current']['last'] == {'k': 0, 'ratio': None}
    assert study.summary['houses_now'] == {'n': 0, 'k': 0, 'herd': [], 'pending': []}
    assert study.summary['cis']['current'] == {'k': 0, 'ratio': None}


def test_compute_herd_without_a_run_and_without_cycles(fresh_app, tmp_path):
    study = compute(tmp_path, without_manifest=True)
    assert study.summary['current'] is None and study.summary['houses_now'] is None and study.election is None
    assert study.inputs['current'] == {'skipped': 'no manifest'} and study.summary['n_cycles'] == 2
    assert study.summary['cis'] is None and [r['event_date'] for r in study.cycles] == ['2019-11-10', '2023-07-23']
    with pytest.raises(MissingInput) as info:
        compute(tmp_path, without_events=True)
    assert info.value.part == 'events/es'


def test_compute_herd_moves_the_current_cycle_to_the_past_once_published(fresh_app, tmp_path):
    # the event 2026-11-29 published with results and a cycle part copied from the run polls
    study = compute(tmp_path, publish_current=True)
    assert study.summary['current'] is None and [r['event_date'] for r in study.cycles][-1] == '2026-11-29'
    assert study.summary['n_cycles'] == 3 and study.cycles[-1]['current'] is False
    assert study.inputs['current'] == {'skipped': 'election published'} and study.election is None
