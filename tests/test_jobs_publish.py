"""Tests del job `publish` (`mtpy/jobs/Publish.py`) con la publicación y la base sustituidas."""
import types

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from tests.fakes import FakeSimulator


def entry(scope, run_id):
    return {'latest': run_id, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
            'date_last': '2026-10-01', 'n_polls': 6}


@pytest.fixture
def patched(monkeypatch):
    """`resolve_scopes`, `next_event_date` y `publish_forecast` sin base: `es` publica de verdad con
    `FakeSimulator`, `db_stats` y `provenance` sustituidos, `es-md` no tiene evento, `es-cb` no tiene
    sondeos, `es-ar` rompe y `es-ri` falla con un `ValueError` del paquete."""
    calls = []
    real_forecast = publish.publish_forecast

    def fake_forecast(scope, writer, run_id, event_date, **kwargs):
        calls.append((scope, run_id, event_date, kwargs))
        if scope == 'es-cb':
            raise publish.NothingToPublish('No polls for es-cb 2027-05-23: nothing to forecast')
        if scope == 'es-ri':
            raise ValueError("vote: missing keys ['rows']")
        if scope == 'es-ar':
            raise KeyError('x')
        return real_forecast(scope, writer, run_id, event_date, simulator=FakeSimulator(),
                             stats=lambda scope, event_date: (7, '2026-10-01'),
                             prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    monkeypatch.setattr(publish, 'resolve_scopes', lambda scopes, catalogue=None: ['es', 'es-md', 'es-cb', 'es-ar'] if scopes == 'all' else list(scopes))
    monkeypatch.setattr(publish, 'next_event_date', lambda scope: {'es': '2026-11-29', 'es-cb': '2027-05-23', 'es-ar': '2026-02-08',
                                                              'es-ri': '2027-05-23'}.get(scope))
    monkeypatch.setattr(publish, 'publish_forecast', fake_forecast)
    return calls


def test_publish_isolates_scopes_and_keeps_previous_manifest_entries(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    publish.update_manifest(bundle.BundleWriter(fs), {'es-ar': entry('es-ar', '20261001-000000')})
    errors = []
    fresh_app.set('logger', types.SimpleNamespace(error=errors.append, info=lambda m: None))
    with pytest.raises(RuntimeError, match='es-ar'):
        Publish().run(what='forecast', scopes='all', fs=fs, n_sim=20)
    out = capsys.readouterr().out
    assert 'es: published' in out and 'es-md: skipped (no upcoming event)' in out
    assert 'es-cb: skipped (No polls' in out and "es-ar: failed (KeyError: 'x')" in out
    assert 'manifest: 2 scopes, freeze off' in out
    assert len(errors) == 1 and 'KeyError' in errors[0]
    assert [c[0] for c in patched] == ['es', 'es-cb', 'es-ar'] and patched[0][3]['n_sim'] == 20
    manifest = publish.read_manifest(bundle.BundleReader(fs))
    assert manifest['scopes']['es']['latest'] == patched[0][1] and manifest['scopes']['es-ar']['latest'] == '20261001-000000'


def test_dry_run_writes_apart_and_leaves_the_pointers_alone(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    result = Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)), dry_run=True)
    assert result['es']['status'] == 'published'
    assert (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es').exists()
    assert not (tmp_path / 'site').exists()
    assert not (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es' / 'history.json').exists()


def test_loreg_window_refuses_es_unless_forced(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, today='2026-11-26')
    assert result['es']['status'] == 'refused' and 'LOREG' in result['es']['reason'] and patched == []
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, today='2026-11-26', force=True)
    assert result['es']['status'] == 'published' and patched[0][3]['freeze'] is True


def test_manifest_point_and_unpublish_actions(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    run_id = patched[0][1]
    Publish().run(what=['manifest'], fs=fs, freeze={'active': True, 'message': 'Veda'})
    assert publish.read_manifest(bundle.BundleReader(fs))['freeze']['active'] is True
    with pytest.raises(ValueError, match='run'):
        Publish().run(what=['point'], scopes=['es'], fs=fs)
    assert Publish().run(what=['point'], scopes=['es'], fs=fs, run=run_id)['es']['status'] == 'published'
    with pytest.raises(RuntimeError):
        Publish().run(what=['point'], scopes=['es'], fs=fs, run='20200101-000000')
    assert Publish().run(what=['unpublish'], scopes=['es'], fs=fs, run=run_id)['es'] == {'status': 'published', 'removed': run_id, 'latest': None}
    assert 'es' not in publish.read_manifest(bundle.BundleReader(fs))['scopes']


def test_deferred_and_unknown_what_are_rejected(fresh_app, tmp_path):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='phase 4'):
        Publish().run(what=['analysis'], fs=FileSystem(str(tmp_path)))
    with pytest.raises(ValueError, match='unknown'):
        Publish().run(what='nope', fs=FileSystem(str(tmp_path)))
    with pytest.raises(RuntimeError, match='file system'):
        Publish().run(what=['manifest'])


def test_failed_forecast_is_not_hidden_by_a_later_successful_action(fresh_app, tmp_path, patched, monkeypatch):
    """Un `forecast` fallido sigue contando aunque el `point` posterior del mismo ámbito funcione."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    run_id = patched[0][1]

    def broken(*args, **kwargs):
        raise KeyError('x')

    monkeypatch.setattr(publish, 'publish_forecast', broken)
    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['forecast', 'point'], scopes=['es'], fs=fs, run=run_id)


def test_error_resolving_the_event_date_only_fails_that_scope(fresh_app, tmp_path, patched, monkeypatch):
    """Un error de base al resolver la fecha de un ámbito no detiene a los demás."""
    from mtpy.jobs.Publish import Publish

    def next_date(scope):
        if scope == 'es-md':
            raise RuntimeError('db down')
        return '2026-11-29'

    monkeypatch.setattr(publish, 'next_event_date', next_date)
    with pytest.raises(RuntimeError, match='es-md'):
        Publish().run(what=['forecast'], scopes=['es-md', 'es'], fs=FileSystem(str(tmp_path)))
    assert [c[0] for c in patched] == ['es']


def test_forecast_rebuilds_the_history_of_each_published_scope(fresh_app, tmp_path, patched):
    """Tras publicar, `history.json` del ámbito contiene exactamente la ejecución publicada."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    assert (tmp_path / 'site' / 'v1' / 'runs' / 'es' / 'history.json').exists()
    runs = bundle.BundleReader(fs).read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in runs] == [patched[0][1]]


def test_a_value_error_other_than_nothing_to_publish_fails_the_scope(fresh_app, tmp_path, patched, capsys):
    """Solo `NothingToPublish` omite el ámbito; cualquier otro `ValueError` (validación del paquete) falla."""
    from mtpy.jobs.Publish import Publish

    with pytest.raises(RuntimeError, match='es-ri'):
        Publish().run(what=['forecast'], scopes=['es-ri', 'es-cb', 'es'], fs=FileSystem(str(tmp_path)))
    out = capsys.readouterr().out
    assert "es-ri: failed (ValueError: vote: missing keys ['rows'])" in out
    assert 'es-cb: skipped (No polls' in out and 'es: published' in out


def test_event_date_is_normalised_or_fails_the_scope(fresh_app, tmp_path, patched, capsys):
    """`event_date` llega en ISO a `meta`, `headline` y al SQL de `db_stats`; si no es una fecha, el ámbito falla."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['forecast'], scopes=['es'], fs=fs, event_date='29/11/2026')
    assert 'es: failed (ValueError: Invalid isoformat' in capsys.readouterr().out and patched == []
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, event_date='20261129')
    assert result['es']['status'] == 'published' and patched[0][2] == '2026-11-29'


def test_manifest_write_failure_keeps_the_summary(fresh_app, tmp_path, patched, capsys, monkeypatch):
    """Si falla la escritura del manifest, el resumen se imprime y el job termina con error."""
    from mtpy.jobs.Publish import Publish

    def boom(writer, scopes=None, freeze=None):
        raise OSError('s3 down')
    monkeypatch.setattr(publish, 'update_manifest', boom)
    with pytest.raises(RuntimeError, match='manifest'):
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)))
    out = capsys.readouterr().out
    assert 'es: published' in out and 'manifest: failed (OSError: s3 down)' in out


def test_backfill_publishes_one_run_per_day_and_never_moves_latest_back(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    today_run = publish.read_manifest(writer)['scopes']['es']['latest']
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert [r['status'] for r in result['es']] == ['published', 'published']
    assert writer.list_runs('es') == ['20261005-120000', '20261006-120000', today_run]
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000', '20261006-120000', today_run]
    assert [r['backfill'] for r in history] == [True, True, False]
    assert history[0]['run_at'] == '2026-10-05T12:00:00Z'
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == today_run
    out = capsys.readouterr().out
    assert 'es: published 20261005-120000' in out and 'backfill 2026-10-05' in out
    again = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert [r['status'] for r in again['es']] == ['skipped', 'skipped'] and 'run exists' in capsys.readouterr().out
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == today_run


def test_backfill_alone_points_latest_to_its_newest_day(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert publish.read_manifest(bundle.BundleWriter(fs))['scopes']['es']['latest'] == '20261006-120000'


def test_backfill_range_errors_fail_before_touching_the_bundle(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    with pytest.raises(ValueError, match='publish: backfill'):
        Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-07', 'to': '2026-10-05'})
    assert patched == [] and not (tmp_path / 'site').exists()


def test_backfill_respects_a_pointed_run(fresh_app, tmp_path, patched, monkeypatch):
    """Un relleno no deshace un `point` hacia atrás; una publicación normal posterior sí mueve `latest`."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    run_a, run_b, run_c = '20261008-100000', '20261008-110000', '20261008-120000'
    for rid in (run_a, run_b):
        monkeypatch.setattr(bundle, 'run_id', lambda now=None, rid=rid: rid)
        Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == run_b
    Publish().run(what=['point'], scopes=['es'], fs=fs, run=run_a)
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert [r['status'] for r in result['es']] == ['published', 'published']
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == run_a
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000', '20261006-120000', run_a, run_b]
    monkeypatch.setattr(bundle, 'run_id', lambda now=None: run_c)
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == run_c


def test_backfill_needs_forecast_in_what(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='publish: "backfill" needs "forecast" in what'):
        Publish().run(what=['manifest'], fs=FileSystem(str(tmp_path)), backfill={'from': '2026-10-05'}, today='2026-10-09')
    assert patched == [] and not (tmp_path / 'site').exists()


def test_backfill_failed_day_is_reported_until_unpublished(fresh_app, tmp_path, patched, monkeypatch, capsys):
    """Un día que falla a medias deja su carpeta: el job falla, los demás días se escriben y un reintento
    lo da por fallido (incompleto) en vez de omitirlo, hasta que se borra con `unpublish`."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    real_meta, broken = publish.export_meta, []

    def export_meta(sim, run_id, *args, **kwargs):
        if run_id == '20261006-120000' and not broken:
            broken.append(run_id)
            raise OSError('s3 down')
        return real_meta(sim, run_id, *args, **kwargs)

    monkeypatch.setattr(publish, 'export_meta', export_meta)
    span = {'from': '2026-10-05', 'to': '2026-10-07'}
    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill=span, today='2026-10-09')
    assert 'es: failed 2026-10-06 (OSError: s3 down)' in capsys.readouterr().out
    assert writer.exists(bundle.path_run('es', '20261006-120000'))
    assert writer.list_runs('es') == ['20261005-120000', '20261007-120000']
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == '20261007-120000'
    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill=span, today='2026-10-09')
    out = capsys.readouterr().out
    assert 'es: skipped 2026-10-05 (run exists)' in out and 'es: skipped 2026-10-07 (run exists)' in out
    assert 'es: failed 2026-10-06 (incomplete run exists; unpublish it first)' in out
    Publish().run(what=['unpublish'], scopes=['es'], fs=fs, run='20261006-120000')
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill=span, today='2026-10-09')
    assert [r['status'] for r in result['es']] == ['skipped', 'published', 'skipped']
    assert writer.list_runs('es') == ['20261005-120000', '20261006-120000', '20261007-120000']


def test_backfill_retry_repairs_the_history_and_the_manifest(fresh_app, tmp_path, patched):
    """Si tras un relleno se pierden `history.json` y la entrada del manifest, repetirlo los reconstruye."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    span = {'from': '2026-10-05', 'to': '2026-10-06'}
    Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill=span, today='2026-10-09')
    writer.remove(bundle.path_history('es'))
    writer.remove(bundle.path_manifest())
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill=span, today='2026-10-09')
    assert [r['status'] for r in result['es']] == ['skipped', 'skipped']
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000', '20261006-120000']
    assert all(r['backfill'] is True for r in history)
    entry = publish.read_manifest(writer)['scopes']['es']
    assert entry['latest'] == '20261006-120000' and entry['run_at'] == '2026-10-06T12:00:00Z'
