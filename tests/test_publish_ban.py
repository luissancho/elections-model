"""Tests de `mtpy/lib/publish/ban.py`: los exportadores del estudio `andorra` y su publicación genérica."""
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.studies.ban.study import compute_ban
from tests.fakes import BAN_EVENTS, RecordingFileSystem, write_ban_bundle
from tests.test_publish_unit import fake_prov, make_writer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def valid(schema, data):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` con el sobre de un estudio."""
    env = bundle.jsonable(bundle.envelope(schema, data, None, generated_at='2026-11-05T12:00:00Z'))
    return bundle.validate(schema, env)['data']


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    """El estudio calculado una vez sobre el paquete sintético."""
    fs = FileSystem(str(tmp_path_factory.mktemp('ban')))
    write_ban_bundle(fs)
    return compute_ban(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')))


def test_export_ban_parts_validate(study):
    for part in bundle.STUDY_PARTS['andorra'][1:]:
        data, frames = publish.STUDY_EXPORTERS['andorra'][part](study)
        valid(bundle.STUDY_SCHEMAS['andorra'][part], data)
        assert set(frames) == set(bundle.STUDY_CSV_OF['andorra'].get(part, ()))
    meta, frames = publish.export_ban_meta(study, prov=fake_prov())
    valid('andorra-meta', meta)
    assert frames == {}
    assert meta['next'] == {'event_date': '2026-11-29', 'open': False}
    assert [s['url'] for s in meta['sources']][-3:] == ['/elecciones', '/metodo', '/analisis']


def test_export_ban_rows_carry_the_house_id_and_no_slug(study):
    pairs, _ = publish.export_ban_pairs(study)
    houses, _ = publish.export_ban_houses(study)
    latest, _ = publish.export_ban_latest(study)
    rows = pairs['rows'] + houses['rows']
    assert rows and all('pollster_id' in r and 'pollster' in r and 'slug' not in r for r in rows)
    assert all('slug' not in poll for poll in latest['polls'])


def test_export_ban_summary_keeps_three_significant_figures_in_p(study):
    data, _ = publish.export_ban_summary(study)
    assert data['pairs']['p'] == float(f"{study.summary['pairs']['p']:.3g}")
    assert 0 < data['pairs']['p'] < 1


def test_export_ban_elections_flattens_the_model(study):
    data, frames = publish.export_ban_elections(study)
    frame = frames['elections']
    assert len(frame) == len(data['rows'])
    assert {'model_error_legal', 'model_error_frozen', 'model_error_consensus', 'model_error_with', 'model_beaten',
            'model_beaten_with'} <= set(frame.columns) and 'model' not in frame.columns
    for row, flat in zip(data['rows'], frame.to_dict('records')):
        if row['model'] is None:
            assert flat['model_error_legal'] is None or flat['model_error_legal'] != flat['model_error_legal']
        else:
            assert flat['model_error_frozen'] == row['model']['error_frozen']


def test_export_ban_latest_frame_has_a_row_per_poll_with_the_shares(study):
    data, frames = publish.export_ban_latest(study)
    frame = frames['latest-polls']
    assert len(frame) == len(data['polls'])
    assert set(data['parties']) <= set(frame.columns)
    assert all(round(v, 1) == v for poll in data['polls'] for v in poll['pct'].values())


def test_export_ban_meta_formats_the_subtitle_with_the_live_figures(study):
    meta, _ = publish.export_ban_meta(study, prov=fake_prov())
    s = study.summary
    assert meta['title'] == publish.STUDY_TITLES['andorra'][0]
    assert f"En las {s['elections']['n']} generales desde 2015" in meta['subtitle']
    assert f"que el promedio congelado en {s['elections']['k_beaten']}, y quedó más cerca del resultado en " \
           f"{s['cells']['k_closer']} de las {s['cells']['n']} cifras de partido." in meta['subtitle']
    assert '{' not in meta['subtitle']


def test_export_ban_meta_without_a_model_uses_the_pairs_sentence(study):
    import dataclasses
    bare = dataclasses.replace(study, summary={**study.summary, 'elections': None})
    meta, _ = publish.export_ban_meta(bare, prov=fake_prov())
    pairs = study.summary['pairs']
    assert meta['subtitle'].endswith(f" En {pairs['n']} pares casa a casa desde 1996, el sondeo clandestino estuvo "
                                     f"más cerca del resultado que el último legal en {pairs['k']}.")
    assert meta['subtitle'].startswith('Los cinco días anteriores a la votación')


def test_publish_study_andorra_writes_the_seven_parts_and_five_csv(study, fresh_app, tmp_path):
    fs = RecordingFileSystem(str(tmp_path))
    write_ban_bundle(fs)
    writer = make_writer(tmp_path, fs)
    out = publish.publish_study('andorra', writer, compute=lambda data, writer: study, prov=fake_prov)
    assert out == {'study': 'andorra', 'registered_at': None, 'checked': False, 'seconds': out['seconds']}
    written = [p for p in fs.written if '/studies/andorra/' in p]
    assert written[-1].endswith('/studies/andorra/meta.json') and fs.written[-1].endswith('/manifest.json')
    assert [os.path.basename(p) for p in written if p.endswith('.json')] == \
        [f'{part}.json' for part in bundle.STUDY_PARTS['andorra'][1:] + ('meta',)]
    assert sorted(os.path.basename(p)[:-4] for p in written if p.endswith('.csv')) == \
        sorted(bundle.STUDY_CSV['andorra'])
    assert publish.read_manifest(writer)['studies']['andorra'] == {'published_at': '2026-10-08T12:00:00Z',
                                                                   'registered_at': None, 'checked': False}
    reader = bundle.BundleReader(fs)
    assert reader.read_json(bundle.path_study('andorra', 'meta'))['data']['next'] == {'event_date': '2026-11-29', 'open': False}
    assert reader.read_json(bundle.path_study('andorra', 'summary'))['data']['pairs']['p'] == 0.812


def test_publish_study_andorra_refuses_register(study, fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('andorra', make_writer(tmp_path, fs), register=True,
                              compute=lambda data, writer: study, prov=fake_prov)


def test_andorra_is_a_live_study():
    assert publish.STUDY_RUNNERS['andorra'] == {'compute': publish.STUDY_RUNNERS['andorra']['compute'],
                                                'registrable': False, 'live': True}


def test_publish_study_andorra_writes_nothing_without_clandestine_polls(fresh_app, tmp_path):
    """Sin sondeos clandestinos en ninguna elección nacional el cálculo para antes de escribir: no queda ningún
    fichero de `studies/andorra/` (partes nuevas junto a un `meta` viejo)."""
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    writer = make_writer(tmp_path, fs)

    for (scope, date) in BAN_EVENTS:
        if scope == 'es':
            path = bundle.path_event(scope, date, 'ban')
            part = writer.read_json(path)
            writer.write_json(path, 'event-ban', {**part['data'], 'polls': [], 'legal': []}, scope)

    with pytest.raises(publish.NothingToPublish):
        publish.publish_study('andorra', writer, data=FileSystem(os.path.join(ROOT, 'data')), prov=fake_prov)

    assert not writer.study_exists('andorra')
    assert not [p for p in tmp_path.rglob('*') if p.is_file() and '/studies/andorra/' in str(p)]
