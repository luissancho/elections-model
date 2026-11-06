"""Tests de `mtpy/lib/publish/shy.py`: los exportadores del estudio «¿Voto oculto de quién?» y su publicación."""
import dataclasses
from unittest.mock import ANY

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.studies.shy.study import compute_shy
from tests.fakes import write_shy_bundle
from tests.test_publish_unit import fake_prov, make_writer

SUBTITLE = ('Se dice que la gente de izquierdas contesta menos a las encuestas y que los votantes de Vox esconden su '
            'voto, y que por eso los sondeos infravaloran a unos y a otros. En las 4 elecciones con sondeos de '
            'nuestra base, los sondeos finales han dado a la izquierda menos voto del que sacó en 2 y más en 2; a '
            'Vox, menos en 0 de 2. Los sondeos se equivocan con cada partido grande en 1,6 puntos de media, y el '
            'sentido del error no se puede adivinar.')


def valid(schema, data):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` con el sobre de un estudio."""
    env = bundle.jsonable(bundle.envelope(schema, data, None, generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(schema, env)['data']


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    """El estudio calculado una vez sobre el paquete sintético."""
    fs = FileSystem(str(tmp_path_factory.mktemp('shy')))
    made = write_shy_bundle(fs)
    return compute_shy(bundle.BundleReader(fs), made['data'])


def subtitle_of(study, **summary):
    """El subtítulo de la meta con `summary` actualizado."""
    changed = dataclasses.replace(study, summary=study.summary | summary)
    return publish.export_shy_meta(changed, prov=fake_prov())[0]['subtitle']


def test_export_shy_parts_validate(study):
    for part in bundle.STUDY_PARTS['voto-oculto'][1:]:
        data, frames = publish.STUDY_EXPORTERS['voto-oculto'][part](study)
        valid(bundle.STUDY_SCHEMAS['voto-oculto'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['voto-oculto'].get(part, ()))
    meta, frames = publish.export_shy_meta(study, prov=fake_prov())
    valid('voto-oculto-meta', meta)
    assert frames == {}
    assert meta['subtitle'] == SUBTITLE
    assert meta['election'] == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05',
                                'run_id': '20261105-120000'}
    assert meta['scopes'] == ['es', 'es-md'] and meta['params']['window_days'] == 14
    assert meta['inputs']['es-xx'] == {'skipped': 'no events'}
    assert [s['url'] for s in meta['sources']][0].startswith('https://theobjective.com')


def test_export_shy_subtitle_follows_the_verdicts(study):
    tilted = subtitle_of(study, verdicts={'p1': 'NO', 'p2': 'NO', 'p3': 'DE PSOE'})
    assert tilted.endswith('y el sentido del error se inclina hacia PSOE.')
    many = subtitle_of(study, verdicts={'p1': 'NO', 'p2': 'NO', 'p3': 'DE PSOE Y DE IU / PODEMOS / SUMAR'})
    assert many.endswith('se inclina hacia PSOE y IU / Podemos / Sumar.')
    left = subtitle_of(study, verdicts={'p1': 'SÍ', 'p2': 'NO', 'p3': 'DE NADIE'}, vox=None)
    assert 'a Vox' not in left and left.endswith('se inclina hacia la izquierda.')
    both = subtitle_of(study, verdicts={'p1': 'SÍ', 'p2': 'NO', 'p3': 'DE PSOE'})
    assert both.endswith('se inclina hacia la izquierda.') and 'a Vox, menos en 0 de 2' in both


def test_export_shy_frames_follow_the_study(study):
    data, frames = publish.STUDY_EXPORTERS['voto-oculto']['elections'](study)
    assert list(frames['elections']['PSOE']) == [r['PSOE'] for r in data['rows']]
    assert frames['elections']['block_left'].iloc[0] == 'PSOE+IU'
    assert list(frames['elections']['name_up']) == ['IU', 'UP+IU', 'UP', 'SUMAR']
    assert {'polls_psoe', 'results_vox', 'name_psoe'} <= set(frames['elections'].columns)
    data, frames = publish.STUDY_EXPORTERS['voto-oculto']['groups'](study)
    assert len(frames['groups']) == len(data['rows'])
    assert list(frames['groups'].columns)[:4] == ['table', 'key', 'label', 'n']


def test_publish_study_voto_oculto_writes_the_parts_and_the_manifest(tmp_path):
    fs = FileSystem(str(tmp_path))
    made = write_shy_bundle(fs)
    out = publish.publish_study('voto-oculto', make_writer(tmp_path, fs), data=made['data'], prov=fake_prov)
    assert out['study'] == 'voto-oculto' and out['registered_at'] is None and out['checked'] is False
    reader = bundle.BundleReader(fs)
    assert reader.read_json(bundle.path_study('voto-oculto', 'summary'))['data']['verdicts']['p3'] == 'DE NADIE'
    assert reader.read_json(bundle.path_manifest())['data']['studies']['voto-oculto'] == {
        'published_at': ANY, 'registered_at': None, 'checked': False}
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('voto-oculto', make_writer(tmp_path, fs), data=made['data'], register=True,
                              prov=fake_prov)


def test_voto_oculto_is_live_and_the_other_studies_are_untouched():
    assert publish.STUDY_RUNNERS['voto-oculto'] == {'compute': publish.compute_shy_for_publish,
                                                    'registrable': False, 'live': True}
    assert publish.STUDY_RUNNERS['voto-cera']['live'] is False and publish.STUDY_RUNNERS['rebano']['live'] is True
