"""Tests de `mtpy/lib/publish/studies.py` sobre el estudio CERA real de `data/`, calculado una vez por módulo."""
import dataclasses
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from tests.fakes import RecordingFileSystem
from tests.test_publish_unit import fake_prov, make_writer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STUDY = 'voto-cera'
DAY = '2026-10-08'  # la fecha del reloj fijo de `make_writer`


def valid(schema, data):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` con el sobre de un estudio."""
    env = bundle.jsonable(bundle.envelope(schema, data, None, generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(schema, env)['data']


def dirty_prov():
    return {**fake_prov(), 'dirty': True}


@pytest.fixture(scope='module')
def study():
    from mtpy.lib.studies.cera import compute_cera
    return compute_cera(FileSystem(os.path.join(ROOT, 'data')))


def run(study, writer, **kwargs):
    """`publish_study` con el estudio ya calculado y la procedencia fija (salvo que se pase otra)."""
    kwargs.setdefault('prov', fake_prov)
    return publish.publish_study(STUDY, writer, compute=lambda data, params=None: study, **kwargs)


def meta_of(writer):
    return writer.read_json(bundle.path_study(STUDY, 'meta'))['data']


def test_export_study_parts_validate(study):
    for part in bundle.STUDY_PARTS:
        data, frames = getattr(publish, 'export_study_' + part.replace('-', '_'))(study)
        valid(bundle.STUDY_SCHEMAS[part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF.get(part, ()))


def test_export_study_figures(study):
    census = valid('study-census', publish.export_study_census(study)[0])
    assert census['series']['values'][-1] == 2762829 and census['series']['dates'][-1] == '2026-09-01'
    assert census['periods']['article'] == [['2019-03-01', '2022-10-01'], ['2022-10-01', '2026-05-01']]
    assert sum(row['g2'] for row in census['provinces']) == 422253
    assert {'g1', 'g2', 'ratio', 'g1_now', 'g2_now', 'ratio_now'} <= set(census['provinces'][0])

    prediction = valid('study-prediction', publish.export_study_prediction(study)[0])
    left = [row for row in prediction['scenarios']['convocatoria']['by_block'] if row['block'] == 'Izquierda'][0]
    assert (left['before'], left['after'], left['change']) == (152, 152, 0)
    convocatoria = prediction['scenarios']['convocatoria']
    assert convocatoria['census'] == 2769645 and convocatoria['new_electors'] == 442257
    assert convocatoria['voters_2023'] == 208431 and convocatoria['votes_2023'] == 203152
    assert len(prediction['last_seat']) == 52 and 'net_b' in prediction['last_seat'][0]

    check = valid('study-check', publish.export_study_check(study)[0])
    assert [e['date'] for e in check['elections']] == ['2019-11-10', '2023-07-23'] and check['verdict'] is None
    last = check['elections'][-1]
    assert (last['net_psoe'], last['net_left'], last['n_over_gap']) == (-1, -1, 9)
    assert [d['region_id'] for d in last['decided']] == [28]
    assert [row['gap'] for row in check['rows']] == sorted(row['gap'] for row in check['rows'])

    meta = valid('study-meta', publish.export_study_meta(study)[0])
    assert meta['election']['census_total'] == 2769645 and meta['series_last'] == '2026-09-01'
    assert meta['regression']['p_slope'] < 0.001 < meta['regression']['p_level']

    sensitivity = valid('study-sensitivity', publish.export_study_sensitivity(study)[0])
    assert len(sensitivity['grid']['values']) == 5 and {len(row) for row in sensitivity['grid']['values']} == {4}
    assert sensitivity['placement']['contrast'] == 0 and sensitivity['placement']['pool'] == 181032

    vote = valid('study-vote', publish.export_study_vote(study)[0])
    assert vote['ecological']['n'] == 50 and vote['ecological']['r'] == pytest.approx(-0.743, abs=0.001)

    placement = valid('study-placement', publish.export_study_placement(study)[0])
    assert placement['sensitive']['article']['all'] == {'stock_share': 24.1, 'growth_share': 17.39}


def test_publish_study_writes_the_parts_in_order_and_meta_last(study, tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    writer = make_writer(tmp_path, fs)
    out = run(study, writer)
    root = 'site/v1/studies/' + STUDY + '/'
    jsons = [name for name in fs.written if name.endswith('.json') and name.startswith(root)]
    assert jsons == [root + part + '.json' for part in (*bundle.STUDY_PARTS[1:], 'meta')]
    csvs = {name for name in fs.written if name.endswith('.csv')}
    assert csvs == {root + 'csv/' + name + '.csv' for name in bundle.STUDY_CSV}
    assert fs.written[-1] == 'site/v1/manifest.json'
    assert writer.study_exists(STUDY)
    assert out['study'] == STUDY and out['registered_at'] is None and out['checked'] is False
    assert 'compute' in out['seconds'] and 'total' in out['seconds']


def test_publish_study_does_not_register_by_default(study, tmp_path):
    writer = make_writer(tmp_path)
    run(study, writer)
    meta = meta_of(writer)
    assert meta['registered_at'] is None and meta['registered'] is None and meta['registered_commit'] is None
    entry = publish.read_manifest(writer)['studies'][STUDY]
    assert entry == {'published_at': '2026-10-08T12:00:00Z', 'registered_at': None, 'checked': False}


def test_publish_study_registers_only_with_a_clean_tree(study, tmp_path):
    writer = make_writer(tmp_path)
    with pytest.raises(publish.PublishRefused, match='clean tree'):
        run(study, writer, register=True, prov=dirty_prov)
    assert not (tmp_path / 'site').exists()
    out = run(study, writer, register=True)
    meta = meta_of(writer)
    assert out['registered_at'] == meta['registered_at'] == DAY and meta['registered_commit'] == 'abc1234'
    assert meta['registered']['max_gain'] == 1 and 'más de 1 escaño' in meta['registered']['claim']
    assert set(meta['registered']['seats']) == {'convocatoria', 'sin_litigio'}
    assert publish.read_manifest(writer)['studies'][STUDY]['registered_at'] == DAY


def test_publish_study_keeps_the_registration_and_refuses_to_register_twice(study, tmp_path):
    writer = make_writer(tmp_path)
    run(study, writer, register=True)
    registered = meta_of(writer)['registered']
    run(study, writer, prov=dirty_prov)
    meta = meta_of(writer)
    assert meta['registered_at'] == DAY and meta['registered'] == registered and meta['registered_changed_at'] is None
    with pytest.raises(publish.PublishRefused, match='already registered on ' + DAY):
        run(study, writer, register=True)
    run(study, writer, register=True, force=True, today='2026-10-20')
    meta = meta_of(writer)
    assert meta['registered_at'] == DAY and meta['registered_changed_at'] == '2026-10-20'


def test_publish_study_sets_checked_at_when_the_election_has_a_scrutiny(study, tmp_path):
    writer = make_writer(tmp_path)
    run(study, writer)
    assert meta_of(writer)['checked_at'] is None
    checked = dataclasses.replace(study, verdict_input={'net_psoe': 0, 'net_left': 0})
    out = run(checked, writer)
    assert meta_of(writer)['checked_at'] == DAY and out['checked'] is True
    assert writer.read_json(bundle.path_study(STUDY, 'check'))['data']['verdict'] == 'acierto'
    assert publish.read_manifest(writer)['studies'][STUDY]['checked'] is True
    run(checked, writer, today='2026-12-01')
    assert meta_of(writer)['checked_at'] == DAY


def test_publish_study_rejects_an_unknown_study(study, tmp_path):
    with pytest.raises(ValueError, match='nope'):
        publish.publish_study('nope', make_writer(tmp_path), compute=lambda data, params=None: study, prov=fake_prov)


def test_importing_publish_does_not_load_the_study_package():
    """La web importa `publish` (sus constantes): el estudio CERA y su regresión se cargan solo al publicarlo."""
    import subprocess
    import sys
    code = ("import sys, mtpy.lib.publish; "
            "print(sorted(m for m in ('mtpy.lib.studies.cera', 'statsmodels.api') if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == '[]'
