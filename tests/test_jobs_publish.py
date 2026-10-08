"""Tests del job `publish` (`mtpy/jobs/Publish.py`) con la publicación y la base sustituidas."""
import types

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish


def entry(scope, run_id):
    return {'latest': run_id, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
            'date_last': '2026-10-01', 'n_polls': 6}


@pytest.fixture
def patched(monkeypatch):
    """`resolve_scopes`, `next_event_date` y `publish_forecast` sin base: `es` publica, `es-md` no tiene
    evento, `es-cb` no tiene sondeos y `es-ar` rompe."""
    calls = []

    def fake_forecast(scope, writer, run_id, event_date, **kwargs):
        calls.append((scope, run_id, event_date, kwargs))
        if scope == 'es-cb':
            raise ValueError('No polls for es-cb 2027-05-23: nothing to forecast')
        if scope == 'es-ar':
            raise KeyError('x')
        writer.write_json(bundle.path_part(scope, run_id, 'headline'), 'headline',
                          {**entry(scope, run_id), 'run_id': run_id, 'nowcast': {}, 'forecast': {}}, scope, run_id=run_id)
        return {'scope': scope, 'entry': entry(scope, run_id), 'seconds': {'total': 12.3}}

    monkeypatch.setattr(publish, 'resolve_scopes', lambda scopes, catalogue=None: ['es', 'es-md', 'es-cb', 'es-ar'] if scopes == 'all' else list(scopes))
    monkeypatch.setattr(publish, 'next_event_date', lambda scope: {'es': '2026-11-29', 'es-cb': '2027-05-23', 'es-ar': '2026-02-08'}.get(scope))
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
