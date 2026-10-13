"""Tests del job `publish` (`mtpy/jobs/Publish.py`) con la publicación y la base sustituidas."""
import types

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from tests.fakes import FakeComputer, FakeSimulator, synthetic_event, write_backtest_source


def entry(scope, run_id):
    return {'latest': run_id, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
            'date_last': '2026-10-01', 'n_polls': 6}


def fixed_prov(monkeypatch):
    """Fija `publish.provenance` (sin `git` real) para los jobs de `backtest`."""
    monkeypatch.setattr(publish.backtest, 'provenance', lambda cwd=None: {'commit': 'abc1234', 'dirty': False, 'versions': {}})


@pytest.fixture
def patched(monkeypatch):
    """`resolve_scopes`, `next_event_date`, `publish_forecast` y `publish_analysis` sin base: `es` publica de
    verdad con `FakeSimulator` (o `FakeComputer`), `db_stats` y la procedencia sustituidos, `es-md` no tiene
    evento, `es-cb` no tiene sondeos, `es-ar` rompe y `es-ri` falla con un `ValueError` del paquete."""
    calls = []
    real_forecast = publish.publish_forecast
    real_analysis = publish.publish_analysis
    real_event = publish.publish_event

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

    def fake_analysis(scope, writer, **kwargs):
        calls.append((scope, 'analysis', kwargs))
        if scope == 'es-ar':
            raise KeyError('x')
        no_polls = ValueError('No polls for es-cb 2027-05-23: nothing to forecast') if scope == 'es-cb' else None
        return real_analysis(scope, writer, computer=FakeComputer(fail_effects=no_polls),
                             stats=lambda scope, event_date: (7, '2026-10-01'),
                             prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    def fake_event(scope, event_date, writer, **kwargs):
        ev = synthetic_event()
        return real_event(scope, event_date, writer, computer=FakeComputer(), events=lambda s: ev.events,
                          results=lambda s, d, r: ev.results.loc[ev.results.date == d], pollsters=lambda: ev.pollsters,
                          prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    monkeypatch.setattr(publish, 'publish_event', fake_event)
    monkeypatch.setattr(publish, 'publish_analysis', fake_analysis)
    monkeypatch.setattr(publish, 'resolve_scopes', lambda scopes, catalogue=None: ['es', 'es-md', 'es-cb', 'es-ar'] if scopes == 'all' else list(scopes))
    next_dates = {'es': '2026-11-29', 'es-cb': '2027-05-23', 'es-ar': '2026-02-08', 'es-ri': '2027-05-23'}
    monkeypatch.setattr(publish, 'next_event_date', lambda scope, date_from=None: next_dates.get(scope))
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


def test_national_window_refuses_regional_forecast_and_analysis_but_not_events(fresh_app, tmp_path, patched, capsys,
                                                                               monkeypatch):
    from mtpy.jobs.Publish import Publish
    fixed_prov(monkeypatch)
    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['forecast'], scopes=['es-cb'], fs=fs, today='2026-11-25')
    assert result['es-cb']['status'] == 'refused' and 'of es 2026-11-29' in result['es-cb']['reason'] and patched == []
    result = Publish().run(what=['analysis'], scopes=['es-cb'], fs=fs, today='2026-11-25')
    assert result['es-cb']['status'] == 'refused' and 'es-cb: analysis refused (' in capsys.readouterr().out
    result = Publish().run(what=['forecast'], scopes=['es-cb'], fs=fs, today='2026-11-25', force=True)
    # el simulador falso de `es-cb` no tiene sondeos: llega a publicar (no `refused`) y la ejecución queda marcada
    assert result['es-cb']['status'] == 'skipped' and patched[-1][3]['freeze'] is True
    write_backtest_source(str(tmp_path / 'bt'))
    assert Publish().run(what=['backtest'], scopes=['es'], fs=fs, today='2026-11-25', source=str(tmp_path / 'bt'))['es']['status'] == 'published'


def test_national_window_covers_the_election_day(fresh_app, tmp_path, patched, monkeypatch):
    """El día de la elección general la ventana nacional sigue abierta: el job pide la próxima de `es` desde la
    víspera (desde hoy, estricto, no habría elección próxima) y un autonómico queda `refused`; al día siguiente,
    no."""
    from mtpy.jobs.Publish import Publish

    def next_date(scope, date_from=None):
        if scope == 'es':
            return '2026-11-29' if date_from is not None and date_from <= '2026-11-28' else None
        return {'es-cb': '2027-05-23'}.get(scope)

    monkeypatch.setattr(publish, 'next_event_date', next_date)
    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['forecast'], scopes=['es-cb'], fs=fs, today='2026-11-29')
    assert result['es-cb']['status'] == 'refused' and 'of es 2026-11-29' in result['es-cb']['reason'] and patched == []
    result = Publish().run(what=['forecast'], scopes=['es-cb'], fs=fs, today='2026-11-30')
    # fuera de la ventana el doble de `es-cb` llega a publicar y queda `skipped` (no tiene sondeos)
    assert result['es-cb']['status'] == 'skipped' and [c[0] for c in patched] == ['es-cb']


def test_analysis_of_a_scope_without_upcoming_event_is_published_unless_in_the_national_window(fresh_app, tmp_path,
                                                                                                patched, monkeypatch):
    """Un ámbito sin elección próxima (`es-md`) publica su análisis desde su última elección: solo lo guarda la
    ventana nacional, nunca queda `skipped` por no tener evento."""
    from mtpy.jobs.Publish import Publish

    calls = []

    def fake_analysis(scope, writer, **kwargs):
        calls.append(scope)
        return {'event_date': '2023-05-28', 'n_pollsters': 4, 'seconds': {'total': 0.1}}

    monkeypatch.setattr(publish, 'publish_analysis', fake_analysis)
    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis'], scopes=['es-md'], fs=fs, today='2026-10-13')
    assert result['es-md'] == {'status': 'published', 'event_date': '2023-05-28', 'n_pollsters': 4,
                               'seconds': {'total': 0.1}}
    assert calls == ['es-md']
    result = Publish().run(what=['analysis'], scopes=['es-md'], fs=fs, today='2026-11-25')
    assert result['es-md']['status'] == 'refused' and 'of es 2026-11-29' in result['es-md']['reason']
    assert calls == ['es-md']


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


def test_unknown_what_is_rejected(fresh_app, tmp_path):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='unknown'):
        Publish().run(what='nope', fs=FileSystem(str(tmp_path)))
    with pytest.raises(RuntimeError, match='file system'):
        Publish().run(what=['manifest'])


def test_analysis_action_publishes_each_scope(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis'], scopes=['es'], fs=fs)
    assert result['es']['status'] == 'published'
    assert bundle.BundleReader(fs).analysis_exists('es')
    assert 'es: analysis published (event 2026-11-29, 3 pollsters)' in capsys.readouterr().out


def test_analysis_isolates_scopes_and_dry_run_writes_apart(fresh_app, tmp_path, patched, capsys):
    """Con `dry_run`, el análisis va a `site-dry/v1/analysis/`; un ámbito sin datos se omite, uno que rompe
    falla sin detener a los demás, y el manifest no se toca."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    with pytest.raises(RuntimeError, match=r'es-ar \(analysis\)'):
        Publish().run(what=['analysis'], scopes=['es-cb', 'es-ar', 'es'], fs=fs, dry_run=True)
    out = capsys.readouterr().out
    assert 'es-cb: analysis skipped (No polls for es-cb 2027-05-23: nothing to forecast)' in out and "es-ar: analysis failed (KeyError: 'x')" in out
    assert 'es: analysis published (event 2026-11-29, 3 pollsters) in ' in out and 'manifest' not in out
    assert bundle.BundleReader(fs, prefix=bundle.DRY_PREFIX).analysis_exists('es')
    assert not (tmp_path / 'site').exists()
    assert [c[2] for c in patched] == [{'verbose': 0}] * 3


def test_analysis_of_a_scope_without_polls_in_its_current_event_is_skipped(fresh_app, tmp_path, patched, capsys):
    """Un ámbito cuyo evento en curso aún no tiene sondeos se omite (como en `forecast`) y el job no falla."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis'], scopes=['es-cb', 'es'], fs=fs)
    assert result['es-cb'] == {'status': 'skipped', 'reason': 'No polls for es-cb 2027-05-23: nothing to forecast'}
    assert result['es']['status'] == 'published'
    out = capsys.readouterr().out
    assert 'es-cb: analysis skipped (No polls for es-cb 2027-05-23: nothing to forecast)' in out
    assert 'es: analysis published (event 2026-11-29, 3 pollsters)' in out
    reader = bundle.BundleReader(fs)
    assert reader.analysis_exists('es') and not reader.exists('analysis/es-cb')


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

    def next_date(scope, date_from=None):
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

    monkeypatch.setattr(publish.forecast, 'export_meta', export_meta)
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


def test_backtest_action_publishes_each_scope(fresh_app, tmp_path, patched, capsys, monkeypatch):
    from mtpy.jobs.Publish import Publish
    fixed_prov(monkeypatch)
    write_backtest_source(str(tmp_path / 'bt'))
    fs = FileSystem(str(tmp_path / 'site'))
    result = Publish().run(what=['backtest'], scopes=['es', 'es-cb'], fs=fs, source=str(tmp_path / 'bt'))
    assert result['es']['status'] == 'published' and result['es-cb'] == {'status': 'skipped', 'reason': 'no backtest results for es-cb'}
    out = capsys.readouterr().out
    assert 'es: backtest published (4 cases, 2 elections) in ' in out and 'es-cb: backtest skipped (no backtest results for es-cb)' in out
    assert bundle.BundleReader(fs).backtest_exists('es') and 'manifest' not in out


def test_backtest_dry_run_writes_apart_and_a_failure_fails_the_job(fresh_app, tmp_path, patched, capsys, monkeypatch):
    from mtpy.jobs.Publish import Publish
    fixed_prov(monkeypatch)
    write_backtest_source(str(tmp_path / 'bt'))
    fs = FileSystem(str(tmp_path / 'site'))
    monkeypatch.setattr(publish, 'resolve_scopes', lambda scopes, catalogue=None: list(scopes))
    Publish().run(what=['backtest'], scopes=['es'], fs=fs, source=str(tmp_path / 'bt'), dry_run=True)
    assert bundle.BundleReader(fs, prefix=bundle.DRY_PREFIX).backtest_exists('es')
    assert not bundle.BundleReader(fs).backtest_exists('es')
    monkeypatch.setattr(publish, 'publish_backtest', lambda *a, **k: (_ for _ in ()).throw(KeyError('x')))
    with pytest.raises(RuntimeError, match=r'es \(backtest\)'):
        Publish().run(what=['backtest'], scopes=['es'], fs=fs)
    assert "es: backtest failed (KeyError: 'x')" in capsys.readouterr().out


def test_event_needs_its_date(fresh_app, tmp_path):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='"event_date" is required for event'):
        Publish().run(what=['event'], fs=FileSystem(str(tmp_path)))


def test_event_action_publishes_each_date(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path / 'site'))
    result = Publish().run(what=['event'], scopes=['es'], event_date=['2023-07-23', '2019-11-10'], fs=fs)
    assert [r['status'] for r in result['es']] == ['published', 'skipped']
    out = capsys.readouterr().out
    assert 'es: event 2023-07-23 published (3 polls, 3 pollsters) in ' in out
    assert 'es: event 2019-11-10 skipped (no official results for es 2019-11-10)' in out
    assert bundle.BundleReader(fs).event_exists('es', '2023-07-23') and 'manifest' not in out
    assert result['es'][0]['event_date'] == '2023-07-23'


def test_event_action_fails_a_date_without_stopping_the_others(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['event'], scopes=['es'], event_date='2023-13-01', fs=FileSystem(str(tmp_path / 'site')))
    assert 'es: event 2023-13-01 failed (ValueError' in capsys.readouterr().out


def test_event_all_and_invalid_list_through_the_job(fresh_app, tmp_path, patched, capsys, monkeypatch):
    from mtpy.jobs.Publish import Publish

    monkeypatch.setattr(publish.events, 'get_events', lambda scope: synthetic_event().events)
    fs = FileSystem(str(tmp_path / 'site'))
    result = Publish().run(what=['event'], scopes=['es'], event_date='all', today='2026-10-13', fs=fs)
    assert [(r['event_date'], r['status']) for r in result['es']] == [('2019-11-10', 'skipped'),
                                                                      ('2023-07-23', 'published')]
    capsys.readouterr()
    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['event'], scopes=['es'], event_date=['2023-07-23', '2023-13-01'], fs=fs)
    out = capsys.readouterr().out
    assert out.count('failed (') == 1 and 'ValueError: month must be in 1..12' in out and 'published' not in out


def test_event_dry_run_writes_apart(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    Publish().run(what=['event'], scopes=['es'], event_date='2023-07-23', fs=FileSystem(str(tmp_path)), dry_run=True)
    assert (tmp_path / 'site-dry' / 'v1' / 'events' / 'es' / '2023-07-23' / 'meta.json').exists()
    assert not (tmp_path / 'site').exists()


def test_event_requires_a_date_and_excludes_forecast(fresh_app, tmp_path):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='event_date'):
        Publish().run(what=['event'], fs=FileSystem(str(tmp_path)))
    with pytest.raises(ValueError, match='cannot be combined'):
        Publish().run(what=['event', 'forecast'], event_date='2023-07-23', fs=FileSystem(str(tmp_path)))


NO_POLLS = 'No polls available for `X`: skipped'


def warned_forecast(monkeypatch, messages):
    """`publish_forecast` real con un `FakeSimulator` que avisa `messages` al ajustar el promedio."""
    real = publish.forecast.publish_forecast

    def fake(scope, writer, run_id, event_date, **kwargs):
        return real(scope, writer, run_id, event_date, simulator=FakeSimulator(warn=messages),
                    stats=lambda scope, event_date: (7, '2026-10-01'),
                    prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    monkeypatch.setattr(publish, 'publish_forecast', fake)


def test_forecast_collects_the_model_warnings_into_the_result(fresh_app, tmp_path, patched, monkeypatch, capsys,
                                                              recwarn):
    """Los avisos del modelo no salen sueltos: van al resultado (únicos, en orden) y al recuento de la línea."""
    from mtpy.jobs.Publish import Publish

    warned_forecast(monkeypatch, [NO_POLLS, NO_POLLS])
    result = Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)))
    assert result['es']['status'] == 'published' and result['es']['warnings'] == [NO_POLLS]
    summary = [line for line in capsys.readouterr().out.splitlines() if line.startswith('es: published')]
    assert len(summary) == 1 and summary[0].endswith(' · 1 aviso')
    assert NO_POLLS not in summary[0] and len(recwarn) == 0


def test_forecast_prints_the_model_warnings_live_only_when_verbose(fresh_app, tmp_path, patched, monkeypatch, capsys,
                                                                   recwarn):
    from mtpy.jobs.Publish import Publish

    warned_forecast(monkeypatch, [NO_POLLS])
    Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path / 'quiet')))
    assert NO_POLLS not in capsys.readouterr().out
    Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path / 'verbose')), verbose=1)
    out = capsys.readouterr().out
    live = out.index('es · aviso: ' + NO_POLLS)
    assert out.count(NO_POLLS) == 1 and live < out.index('\nes: published') and len(recwarn) == 0


def test_forecast_keeps_the_warnings_of_a_skipped_or_failed_scope(fresh_app, tmp_path, patched, monkeypatch, capsys,
                                                                  recwarn):
    """Un ámbito que avisa y luego se salta o rompe lleva sus avisos (sin repetir) y nada escapa del job."""
    import warnings

    from mtpy.jobs.Publish import Publish

    def fake(scope, writer, run_id, event_date, **kwargs):
        warnings.warn('Fewer than three usable polls for `A`: not fitted')
        warnings.warn('Fewer than three usable polls for `B`: not fitted')
        warnings.warn('Fewer than three usable polls for `A`: not fitted')
        if scope == 'es-cb':
            raise publish.NothingToPublish('Not enough polls to fit the average')
        raise KeyError('x')

    monkeypatch.setattr(publish, 'publish_forecast', fake)
    with pytest.raises(RuntimeError, match='es-ar'):
        Publish().run(what=['forecast'], scopes=['es-cb', 'es-ar'], fs=FileSystem(str(tmp_path)))
    out = capsys.readouterr().out
    assert 'es-cb: skipped (Not enough polls to fit the average) · 2 avisos' in out
    assert "es-ar: failed (KeyError: 'x') · 2 avisos" in out
    assert 'not fitted' not in out and len(recwarn) == 0


def test_forecast_without_model_warnings_has_none(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    result = Publish().run(what=['forecast'], scopes=['es', 'es-md'], fs=FileSystem(str(tmp_path)))
    assert result['es']['warnings'] == [] and result['es-md']['warnings'] == []


def test_forecast_prints_its_progress_live_before_the_summary(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    infos = []
    fresh_app.set('logger', types.SimpleNamespace(error=lambda m: None, info=infos.append))
    Publish().run(what=['forecast'], scopes=['es', 'es-cb'], fs=FileSystem(str(tmp_path)))
    out = capsys.readouterr().out
    lines = out.splitlines()
    summary = next(i for i, line in enumerate(lines) if line.startswith('es: published'))
    expected = ['publish forecast: 2 ámbitos · run ' + patched[0][1], '[1/2] es: empezando', '[1/2] es: simulador listo (',
                '[1/2] es: promedio ajustado (', '[1/2] es: simulación de hoy (', '[1/2] es: exportación de hoy (',
                '[1/2] es: simulación de la elección (', '[1/2] es: exportación (', '[1/2] es: meta y titular (',
                '[1/2] es: published ', '[2/2] es-cb: empezando', '[2/2] es-cb: skipped (No polls']
    found = [next(i for i, line in enumerate(lines) if text in line) for text in expected]
    assert found == sorted(found) and found[-1] < summary
    assert all(lines[i].startswith('[') and lines[i][9:11] == '] ' for i in found)  # [hh:mm:ss] delante
    assert any('[1/2] es: promedio ajustado (' in m for m in infos)
    assert any(m.startswith('es: published') for m in infos)  # el alert final con el resumen sigue llegando


def test_backfill_progress_names_the_day(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)),
                  backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    out = capsys.readouterr().out
    assert 'publish forecast: 1 ámbito · backfill 2026-10-05..2026-10-06 (2 días)' in out
    assert out.index('[1/1] es 2026-10-05: empezando') < out.index('[1/1] es 2026-10-06: promedio ajustado (')


class ListHandler:
    """Recoge los mensajes que recibe un `logging.Logger` (sin escribir en consola)."""

    def __new__(cls):
        import logging

        class _Handler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.messages = []

            def emit(self, record):
                self.messages.append(record.getMessage())

        return _Handler()


def app_log(name, *handlers):
    """Logger con la forma del `Log` de la app (`loggers['info']`, `info`, `error`) y estos manejadores."""
    import logging

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        logger.addHandler(handler)
    return types.SimpleNamespace(loggers={'info': logger, 'error': logger}, info=logger.info, error=logger.error)


def drop_handlers(log):
    for handler in list(log.loggers['info'].handlers):
        log.loggers['info'].removeHandler(handler)
        handler.close()


PROGRESS = ['[1/1] es: empezando', '[1/1] es: promedio ajustado (', '[1/1] es: meta y titular (']


def test_progress_is_not_echoed_twice_by_a_console_logger(fresh_app, tmp_path, patched, capsys):
    """Con un logger que ya escribe en consola (`App` con `verbose`), cada línea de progreso sale una vez, con la
    marca de tiempo del logger y no la propia."""
    import logging
    import sys

    from mtpy.jobs.Publish import Publish

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    log = app_log('test-publish-console', console)
    fresh_app.set('logger', log)
    try:
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)))
    finally:
        drop_handlers(log)
    lines = capsys.readouterr().out.splitlines()
    for text in PROGRESS:
        found = [line for line in lines if text in line]
        assert len(found) == 1, text
        assert found[0].index(text) == len('[2026-10-14 01:19:24,279] ')  # solo el prefijo del logger


def test_progress_without_logger_prints_once_with_the_time(fresh_app, tmp_path, patched, capsys):
    import re

    from mtpy.jobs.Publish import Publish

    Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)))
    lines = capsys.readouterr().out.splitlines()
    for text in PROGRESS:
        found = [line for line in lines if text in line]
        assert len(found) == 1 and re.fullmatch(r'\[\d\d:\d\d:\d\d\] ' + re.escape(text) + r'.*', found[0]), text


def test_progress_with_a_file_logger_prints_and_logs_the_bare_text(fresh_app, tmp_path, patched, capsys):
    import logging

    from mtpy.jobs.Publish import Publish

    listed = ListHandler()
    log = app_log('test-publish-file', logging.NullHandler(), logging.FileHandler(str(tmp_path / 'info.log')), listed)
    fresh_app.set('logger', log)
    try:
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path / 'site')))
    finally:
        drop_handlers(log)
    out = capsys.readouterr().out
    assert all(out.count(text) == 1 for text in PROGRESS)
    assert '[1/1] es: empezando' in listed.messages  # el texto desnudo, sin `[hh:mm:ss]`
    assert '[1/1] es: empezando' in (tmp_path / 'info.log').read_text()


def test_logger_echoes_console_only_with_a_console_stream_handler(tmp_path):
    import logging
    import sys

    from mtpy.jobs.Publish import Publish

    echoes = Publish._logger_echoes_console
    assert echoes(types.SimpleNamespace(info=lambda m: None)) is False
    for stream in (sys.stdout, sys.stderr):
        log = app_log('test-echo-stream', logging.StreamHandler(stream))
        try:
            assert echoes(log) is True and echoes(log.loggers['info']) is True
        finally:
            drop_handlers(log)
    log = app_log('test-echo-file', logging.FileHandler(str(tmp_path / 'x.log')), logging.NullHandler())
    try:
        assert echoes(log) is False
    finally:
        drop_handlers(log)
    parent = app_log('test-echo-parent', logging.StreamHandler(sys.stdout))
    child = logging.getLogger('test-echo-parent.child')
    try:
        assert echoes(child) is True  # hereda la consola del padre por propagación
        parent.loggers['info'].setLevel(logging.WARNING)
        assert echoes(parent) is False  # un `info` que no se emite no se repite
    finally:
        drop_handlers(parent)
