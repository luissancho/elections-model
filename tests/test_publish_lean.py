"""Tests de `mtpy/lib/publish/lean.py`: los exportadores del sesgo del CIS y su publicación genérica por estudio."""
import os
from types import SimpleNamespace

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.publish.lean import URL_INDEPENDIENTE_2018_07, URL_INDEPENDIENTE_2018_09, URL_UNED
from mtpy.lib.studies.lean.study import compute_lean
from tests.fakes import RecordingFileSystem, write_lean_bundle
from tests.test_publish_unit import fake_prov, make_writer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def valid(schema, data):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` con el sobre de un estudio."""
    env = bundle.jsonable(bundle.envelope(schema, data, None, generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(schema, env)['data']


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    """El estudio calculado una vez sobre el paquete sintético (mínimo de 3 elecciones)."""
    fs = FileSystem(str(tmp_path_factory.mktemp('lean')))
    write_lean_bundle(fs)
    return compute_lean(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'min_elections': 3})


def test_export_lean_parts_validate(study):
    for part in bundle.STUDY_PARTS['sesgo-cis'][1:]:
        data, frames = publish.STUDY_EXPORTERS['sesgo-cis'][part](study)
        valid(bundle.STUDY_SCHEMAS['sesgo-cis'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['sesgo-cis'].get(part, ()))
    meta, frames = publish.export_lean_meta(study, prov=fake_prov())
    valid('sesgo-cis-meta', meta)
    assert frames == {}
    assert meta['subtitle'].startswith(
        'Desde el nombramiento de José Félix Tezanos en 2018, el CIS ha dado al bloque '
        'de izquierdas más voto del que obtuvo en 5 de 6 elecciones')
    assert [s['url'] for s in meta['sources']][-3:] == [
        URL_INDEPENDIENTE_2018_07, URL_INDEPENDIENTE_2018_09, URL_UNED]


def test_export_lean_frames_follow_the_study(study):
    _, frames = publish.STUDY_EXPORTERS['sesgo-cis']['elections'](study)
    assert len(frames['elections']) == len(study.elections)
    _, frames = publish.STUDY_EXPORTERS['sesgo-cis']['houses'](study)
    assert list(frames['houses']['pollster']) == [h['pollster'] for h in study.houses]
    data, frames = publish.STUDY_EXPORTERS['sesgo-cis']['matrix'](study)
    assert list(frames['matrix'].columns) == ['pollster', 'scope', 'event_date', 'Izquierda']
    assert len(frames['matrix']) == len(data['cells'])
    data, frames = publish.STUDY_EXPORTERS['sesgo-cis']['election'](study)
    assert len(frames['election-houses']) == len(data['houses'])
    summary, _ = publish.STUDY_EXPORTERS['sesgo-cis']['summary'](study)
    assert set(summary) == {'with', 'without', 'election'}


def test_export_lean_summary_keeps_three_significant_figures_in_p(study):
    """Los valores p van con tres cifras significativas, fuera del redondeo a 2 decimales del resto (revisión
    final: un p de 5e-06 se publicaba como 0.0 y la nota técnica afirmaba una probabilidad falsa)."""
    summary = {key: dict(study.summary[key]) for key in ('with', 'without')}
    summary['with']['t'] = {'all': {'n': 6, 'mean': 4.1349, 't': 9.87654, 'p': 4.98765e-06},
                            'dates': {'n': 6, 'mean': 4.1, 't': 7.0, 'p': 0.0538123},
                            'consensus': None}
    summary['with']['chance'] = dict(summary['with']['chance'],
                                     coin={'n': 41, 'k': 38, 'p': 1.23456e-08, 'odds': 81000})
    data, _ = publish.export_lean_summary(SimpleNamespace(summary={**summary, 'election': study.summary['election']}))
    assert data['with']['t']['all'] == {'n': 6, 'mean': 4.13, 't': 9.88, 'p': 4.99e-06}
    assert data['with']['t']['dates']['p'] == 0.0538 and data['with']['t']['consensus'] is None
    assert data['with']['chance']['coin']['p'] == 1.23e-08
    assert data['with']['chance']['dates']['p'] == float('{:.3g}'.format(study.summary['with']['chance']['dates']['p']))
    assert data['with']['mean_left'] == round(study.summary['with']['mean_left'], 2)
    valid('sesgo-cis-summary', data)


def test_publish_lean_writes_meta_last_and_updates_the_manifest(study, fresh_app, tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    write_lean_bundle(fs)
    writer = make_writer(tmp_path, fs)
    out = publish.publish_study('sesgo-cis', writer, compute=lambda data, writer: study, prov=fake_prov)
    assert out == {'study': 'sesgo-cis', 'registered_at': None, 'checked': False, 'seconds': out['seconds']}
    written = [p for p in fs.written if '/studies/sesgo-cis/' in p]
    assert written[-1].endswith('/studies/sesgo-cis/meta.json') and fs.written[-1].endswith('/manifest.json')
    assert publish.read_manifest(writer)['studies']['sesgo-cis'] == {'published_at': '2026-10-08T12:00:00Z',
                                                                     'registered_at': None, 'checked': False}


def test_publish_lean_refuses_to_register(study, fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('sesgo-cis', make_writer(tmp_path, fs), register=True,
                              compute=lambda data, writer: study, prov=fake_prov)


def test_publish_lean_reads_the_live_prefix_in_a_dry_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX)
    out = publish.publish_study('sesgo-cis', writer, prov=fake_prov)
    assert out['study'] == 'sesgo-cis' and writer.study_exists('sesgo-cis')
