"""Tests de `mtpy/lib/publish/herd.py`: los exportadores del estudio del rebano y su publicación genérica."""
import dataclasses
import os
from unittest.mock import ANY

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.publish.herd import _subtitle
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


SIG = {'far_final': {'n': 23, 'k': 19, 'sign_p': 0.0013, 'clear': 3, 'median_change': -0.15},
       'far_mid': {'n': 17, 'k': 9, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.1},
       'mid_final': {'n': 29, 'k': 23, 'sign_p': 0.0012, 'clear': 2, 'median_change': -0.15}}


def test_subtitle_when_the_final_weeks_are_significant_and_the_middle_is_not():
    s = {'n_elections': 33, 'since_year': '1977', 'comparisons': SIG}
    assert _subtitle(s) == ('En 19 de las 23 elecciones con datos desde 1977, las mismas encuestadoras se parecen más '
                            'entre sí en las tres últimas semanas que lejos de la votación: por azar pasaría una vez '
                            'de cada 769. Hasta tres semanas antes, no: 9 de 17, como una moneda.')


def test_subtitle_without_significance():
    s = {'n_elections': 2, 'since_year': '2019', 'comparisons': {
        'far_final': {'n': 2, 'k': 1, 'sign_p': 0.75, 'clear': 0, 'median_change': 0.62},
        'far_mid': {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.25},
        'mid_final': {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.67}}}
    assert _subtitle(s) == ('En 1 de las 2 elecciones con datos desde 2019, las mismas encuestadoras se parecen más '
                            'entre sí en las tres últimas semanas que lejos de la votación, una proporción que podría '
                            'deberse al azar.')


def test_subtitle_other_branches():
    far_mid = {'n': 17, 'k': 15, 'sign_p': 0.001, 'clear': 2, 'median_change': -0.2}
    final = dict(SIG['far_final'], sign_p=0.0002)
    text = _subtitle({'since_year': '1977', 'comparisons': {**SIG, 'far_mid': far_mid, 'far_final': final}})
    assert text.endswith('por azar pasaría una vez de cada 5.000. Y también hasta tres semanas antes: 15 de 17.')
    no_mid = _subtitle({'since_year': '1977', 'comparisons': {**SIG, 'far_mid': dict(far_mid, n=0, k=0)}})
    assert no_mid.endswith('una vez de cada 769.')
    assert 'copi' not in text.lower()


def test_export_herd_parts_validate(study):
    summary, frames = publish.export_herd_summary(study)
    assert valid('rebano-summary', summary) and frames == {}
    elections, frames = publish.export_herd_elections(study)
    valid('rebano-elections', elections)
    frame = frames['elections']
    assert len(frame) == 4 and set(frame.columns) == set(study.elections[0])
    assert len(list(study.elections[0])) == 13
    current, frames = publish.export_herd_current(study)
    valid('rebano-current', current)
    assert current['current']['days_left'] == 24 and current['current']['mid_final'] is None and frames == {}
    meta, frames = publish.export_herd_meta(study, prov=fake_prov())
    valid('rebano-meta', meta)
    assert frames == {}
    assert meta['title'] == '¿Se comportan las encuestadoras como un rebaño?'
    assert meta['election'] == study.election and meta['scope'] == 'es'
    assert 'copi' not in meta['subtitle'].lower()
    assert [s['url'] for s in meta['sources']] == ['/elecciones', '/metodo/promedio', '/analisis']


def test_export_herd_current_is_none_without_an_election_in_progress(study):
    idle = dataclasses.replace(study, current=None, election=None)
    assert publish.export_herd_current(idle)[0] == {'current': None}
    valid('rebano-current', {'current': None})
    meta = publish.export_herd_meta(idle, prov=fake_prov())[0]
    assert meta['election'] is None and valid('rebano-meta', meta)


def test_publish_study_rebano_writes_the_four_parts(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    out = publish.publish_study('rebano', make_writer(tmp_path, fs), prov=fake_prov)
    assert out['study'] == 'rebano' and out['registered_at'] is None and out['checked'] is False
    reader = bundle.BundleReader(fs)
    for part in bundle.STUDY_PARTS['rebano']:
        assert reader.read_json(bundle.path_study('rebano', part))['data'] is not None
    assert reader.read_json(bundle.path_study('rebano', 'summary'))['data']['n_elections'] == 2
    assert fs.exists(f"{bundle.PREFIX}/{bundle.path_study_csv('rebano', 'elections')}")
    assert reader.read_json(bundle.path_manifest())['data']['studies']['rebano'] == {
        'published_at': ANY, 'registered_at': None, 'checked': False}
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('rebano', make_writer(tmp_path, fs), register=True, prov=fake_prov)


def test_study_runners_know_which_studies_are_live():
    assert {s: r['live'] for s, r in publish.STUDY_RUNNERS.items()} == {
        'voto-cera': False, 'sesgo-cis': True, 'rebano': True, 'voto-oculto': True, 'andorra': True}
