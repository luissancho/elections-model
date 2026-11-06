"""Tests de `mtpy/lib/publish/herd.py`: los exportadores del estudio del rebano y su publicación genérica."""
import dataclasses
import os
from unittest.mock import ANY

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.studies.herd.study import compute_herd
from tests.fakes import write_herd_bundle
from tests.test_publish_unit import fake_prov, make_writer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def valid(schema, data):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` con el sobre de un estudio."""
    env = bundle.jsonable(bundle.envelope(schema, data, None, generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(schema, env)['data']


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    """El estudio calculado una vez sobre el paquete sintético."""
    fs = FileSystem(str(tmp_path_factory.mktemp('herd')))
    write_herd_bundle(fs)
    return compute_herd(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')))


def test_export_herd_parts_validate(study):
    for part in bundle.STUDY_PARTS['rebano'][1:]:
        data, frames = publish.STUDY_EXPORTERS['rebano'][part](study)
        valid(bundle.STUDY_SCHEMAS['rebano'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['rebano'].get(part, ()))
    meta, frames = publish.export_herd_meta(study, prov=fake_prov())
    valid('rebano-meta', meta)
    assert frames == {}
    assert meta['subtitle'].startswith('En las 2 generales desde 2008, los sondeos se han parecido más entre sí')
    assert 'misma urna' in meta['subtitle']
    assert meta['subtitle'].endswith(' En 2026 ya van así desde el principio, y la única casa con sondeos suficientes '
                                     'publica cifras que oscilan menos de lo que permite el azar de sus muestras.')
    assert meta['election'] == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05'}
    assert meta['params']['window_days'] == 56
    assert [s['url'] for s in meta['sources']][:2] == ['/elecciones', '/casas']


def test_export_herd_meta_drops_the_conditioned_clauses(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    bundle.BundleWriter(fs).remove(bundle.path_manifest())
    study = compute_herd(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'final_days': 1})
    meta, _ = publish.export_herd_meta(study, prov=fake_prov())
    assert 'misma urna' not in meta['subtitle'] and 'ya van así' not in meta['subtitle']
    assert meta['election'] is None


def test_export_herd_meta_counts_the_houses_of_the_current_clause(study):
    """La frase de 2026 del subtítulo: «la única casa» con una sola, «las n casas» con todas, «k de las n» si no, y
    ninguna frase cuando ninguna casa baja de 1."""
    def subtitle(houses_now):
        changed = dataclasses.replace(study, summary=study.summary | {'houses_now': houses_now})
        return publish.export_herd_meta(changed, prov=fake_prov())[0]['subtitle']

    two_of_three = subtitle({'n': 3, 'k': 2, 'herd': [], 'pending': []})
    assert 'y 2 de las 3 casas con sondeos suficientes publican cifras que oscilan' in two_of_three
    assert 'y las 2 casas con sondeos suficientes publican cifras' in subtitle({'n': 2, 'k': 2, 'herd': [],
                                                                                'pending': []})
    none = subtitle({'n': 1, 'k': 0, 'herd': [], 'pending': []})
    assert 'ya van así' not in none and none.endswith('misma urna.')


def test_export_herd_frames_follow_the_study(study):
    data, frames = publish.STUDY_EXPORTERS['rebano']['cycles'](study)
    assert list(frames['cycles']['before_ratio']) == [r['before']['ratio'] for r in data['rows']]
    data, frames = publish.STUDY_EXPORTERS['rebano']['weeks'](study)
    assert len(frames['weeks']) == len(data['rows']) + len(data['mean'])
    data, frames = publish.STUDY_EXPORTERS['rebano']['houses'](study)
    assert list(frames['houses']['pollster']) == [h['pollster'] for h in data['rows']]


def test_publish_study_rebano_writes_the_parts_and_the_manifest(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    out = publish.publish_study('rebano', make_writer(tmp_path, fs), prov=fake_prov)
    assert out['study'] == 'rebano' and out['registered_at'] is None and out['checked'] is False
    reader = bundle.BundleReader(fs)
    assert reader.read_json(bundle.path_study('rebano', 'summary'))['data']['k_narrowed'] == 2
    assert reader.read_json(bundle.path_manifest())['data']['studies']['rebano'] == {
        'published_at': ANY, 'registered_at': None, 'checked': False}
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('rebano', make_writer(tmp_path, fs), register=True, prov=fake_prov)


def test_study_runners_know_which_studies_are_live():
    assert {s: r['live'] for s, r in publish.STUDY_RUNNERS.items()} == {
        'voto-cera': False, 'sesgo-cis': True, 'rebano': True, 'voto-oculto': True, 'andorra': True}
