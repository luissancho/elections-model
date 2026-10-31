"""Tests del job `publish` (`mtpy/jobs/Publish.py`) con la publicación y la base sustituidas: los de extremo a extremo,
que cruzan pipelines o leen el manifest tras el job (los de cada pipeline están en `test_pipelines_publish.py`)."""
import types

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from tests.fakes import synthetic_event, write_backtest_source
from tests.publish_fixtures import NO_POLLS, entry, fixed_prov, patched, warned_forecast  # noqa: F401


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


@pytest.mark.skip(reason='Tarea 9')
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


@pytest.mark.skip(reason='Tarea 9')
def test_analysis_of_a_scope_without_upcoming_event_is_published_unless_in_the_national_window(fresh_app, tmp_path,
                                                                                                patched, monkeypatch):
    """Un ámbito sin elección próxima (`es-md`) publica su análisis desde su última elección: sólo lo guarda la
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
                               'seconds': {'total': 0.1}, 'house_effects': True}
    assert calls == ['es-md']
    result = Publish().run(what=['analysis'], scopes=['es-md'], fs=fs, today='2026-11-25')
    assert result['es-md']['status'] == 'refused' and 'of es 2026-11-29' in result['es-md']['reason']
    assert calls == ['es-md']


@pytest.mark.skip(reason='Tarea 9')
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


@pytest.mark.skip(reason='Tarea 9')
def test_analysis_action_publishes_each_scope(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis'], scopes=['es'], fs=fs)
    assert result['es']['status'] == 'published'
    assert bundle.BundleReader(fs).analysis_exists('es')
    assert 'es: analysis published (event 2026-11-29, 3 pollsters)' in capsys.readouterr().out


@pytest.mark.skip(reason='Tarea 9')
def test_analysis_isolates_scopes_and_dry_run_writes_apart(fresh_app, tmp_path, patched, capsys):
    """Con `dry_run`, el análisis va a `site-dry/v1/analysis/`; un ámbito sin sondeos en el ciclo actual se publica sin
    efectos de casa, uno que rompe falla sin detener a los demás, y el manifest no se toca."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    with pytest.raises(RuntimeError, match=r'es-ar \(analysis\)'):
        Publish().run(what=['analysis'], scopes=['es-cb', 'es-ar', 'es'], fs=fs, dry_run=True)
    out = capsys.readouterr().out
    assert 'es-cb: analysis published (event 2026-11-29, 3 pollsters)' in out and "es-ar: analysis failed (KeyError: 'x')" in out
    assert 'es: analysis published (event 2026-11-29, 3 pollsters) in ' in out and 'manifest' not in out
    assert bundle.BundleReader(fs, prefix=bundle.DRY_PREFIX).analysis_exists('es')
    assert not (tmp_path / 'site').exists()
    assert [c[2] for c in patched] == [{'verbose': 0}] * 3


@pytest.mark.skip(reason='Tarea 9')
def test_analysis_of_a_scope_without_polls_in_its_current_event_is_published_without_effects(fresh_app, tmp_path, patched, capsys):
    """Un ámbito cuyo evento en curso aún no tiene sondeos publica el análisis de las elecciones pasadas, sin
    efectos de casa."""
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['analysis'], scopes=['es-cb', 'es'], fs=fs)
    assert result['es-cb']['status'] == 'published' and result['es']['status'] == 'published'
    out = capsys.readouterr().out
    assert 'es-cb: analysis published (event 2026-11-29, 3 pollsters)' in out
    assert 'es-cb: analysis published (event 2026-11-29, 3 pollsters) in 0 s; sin efectos de casa' in out
    assert 'es: analysis published (event 2026-11-29, 3 pollsters) in 0 s\n' in out
    reader = bundle.BundleReader(fs)
    assert reader.analysis_exists('es') and reader.analysis_exists('es-cb')


@pytest.mark.skip(reason='Tarea 9')
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


def test_backfill_needs_forecast_in_what(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='publish: "backfill" needs "forecast" in what'):
        Publish().run(what=['manifest'], fs=FileSystem(str(tmp_path)), backfill={'from': '2026-10-05'}, today='2026-10-09')
    assert patched == [] and not (tmp_path / 'site').exists()


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


def test_backfill_replace_must_be_a_boolean(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(ValueError, match='publish: backfill "replace" must be true or false'):
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)),
                      backfill={'from': '2026-10-05', 'replace': 'yes'}, today='2026-10-09')
    assert patched == [] and not (tmp_path / 'site').exists()


@pytest.mark.skip(reason='Tarea 9')
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


@pytest.mark.skip(reason='Tarea 9')
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


@pytest.mark.skip(reason='Tarea 9')
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


@pytest.mark.skip(reason='Tarea 9')
def test_event_action_fails_a_date_without_stopping_the_others(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['event'], scopes=['es'], event_date='2023-13-01', fs=FileSystem(str(tmp_path / 'site')))
    assert 'es: event 2023-13-01 failed (ValueError' in capsys.readouterr().out


@pytest.mark.skip(reason='Tarea 9')
def test_event_all_and_invalid_list_through_the_job(fresh_app, tmp_path, patched, capsys, monkeypatch):
    from mtpy.jobs.Publish import Publish

    monkeypatch.setattr(publish.events, 'get_events', lambda scope: synthetic_event().events)
    fs = FileSystem(str(tmp_path / 'site'))
    result = Publish().run(what=['event'], scopes=['es'], event_date='all', today='2026-10-13', fs=fs)
    assert [(r['event_date'], r['status']) for r in result['es']] == [('2019-11-10', 'skipped'),
                                                                      ('2023-07-23', 'published')]
    capsys.readouterr()
    with pytest.raises(RuntimeError, match='es'):
        Publish().run(what=['event'], scopes=['es'], event_date=['2023-07-23', '2023-13-01'], fs=fs, verbose=0)
    out = capsys.readouterr().out
    assert out.count('failed (') == 1 and 'ValueError: month must be in 1..12' in out and 'published' not in out


@pytest.mark.skip(reason='Tarea 9')
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


def test_verbose_0_prints_only_the_header_and_the_summary(fresh_app, tmp_path, patched, monkeypatch, capsys, recwarn):
    from mtpy.jobs.Publish import Publish

    warned_forecast(monkeypatch, [NO_POLLS])
    Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)), verbose=0)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4 and lines[0].startswith('[') and 'publish forecast: 1 ámbito · run ' in lines[0]
    assert lines[1].startswith('es: published') and lines[1].endswith(' · 1 aviso') and lines[2].startswith('manifest')
    assert lines[3] == 'parties: 5 partidos'  # el catálogo de partidos se reescribe con cada forecast
    assert not any(text in '\n'.join(lines) for text in ('empezando', 'simulador listo', NO_POLLS))
    assert len(recwarn) == 0


@pytest.mark.skip(reason='Tarea 9')
def test_the_model_gets_verbose_minus_two(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish

    for level, expected in ((0, 0), (1, 0), (2, 0), (3, 1)):
        patched.clear()
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path / str(level))), verbose=level)
        assert patched[0][3]['verbose'] == expected
    patched.clear()
    Publish().run(what=['analysis'], scopes=['es'], fs=FileSystem(str(tmp_path / 'analysis')), verbose=3)
    assert patched[0][2] == {'verbose': 1}


@pytest.mark.skip(reason='Tarea 9')
def test_analysis_prints_its_scope_lines_live_at_verbose_1(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    Publish().run(what=['analysis'], scopes=['es'], fs=FileSystem(str(tmp_path / 'one')))
    out = capsys.readouterr().out
    assert out.index('[1/1] es: empezando') < out.index('[1/1] es: analysis published') < out.index('\nes: analysis published')
    Publish().run(what=['analysis'], scopes=['es'], fs=FileSystem(str(tmp_path / 'zero')), verbose=0)
    assert 'empezando' not in capsys.readouterr().out


def test_forecast_prints_its_progress_live_before_the_summary(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    infos = []
    fresh_app.set('logger', types.SimpleNamespace(error=lambda m: None, info=infos.append))
    Publish().run(what=['forecast'], scopes=['es', 'es-cb'], fs=FileSystem(str(tmp_path)), verbose=2)
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
    assert any('es: published' in m for m in infos)  # el alert final con el resumen (tras su cabecera) sigue llegando


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
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)), verbose=2)
    finally:
        drop_handlers(log)
    lines = capsys.readouterr().out.splitlines()
    for text in PROGRESS:
        found = [line for line in lines if text in line]
        assert len(found) == 1, text
        assert found[0].index(text) == len('[2026-10-14 01:19:24,279] ')  # sólo el prefijo del logger


def test_progress_without_logger_prints_once_with_the_time(fresh_app, tmp_path, patched, capsys):
    import re

    from mtpy.jobs.Publish import Publish

    Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)), verbose=2)
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
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path / 'site')), verbose=2)
    finally:
        drop_handlers(log)
    out = capsys.readouterr().out
    assert all(out.count(text) == 1 for text in PROGRESS)
    assert '[1/1] es: empezando' in listed.messages  # el texto desnudo, sin `[hh:mm:ss]`
    assert '[1/1] es: empezando' in (tmp_path / 'info.log').read_text()


def test_logger_echoes_console_only_with_a_console_stream_handler(tmp_path):
    import logging
    import sys

    from mtpy.pipelines.publish.Forecast import Forecast

    echoes = Forecast._logger_echoes_console
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


def test_publish_class_docstring_survives_and_mentions_the_levels():
    from mtpy.jobs.Publish import Publish

    assert Publish.__doc__ is not None and 'four levels' in Publish.__doc__


def test_parties_action_writes_the_catalogue_next_to_the_manifest(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['parties'], fs=fs)
    assert result == {} and (tmp_path / 'site' / 'v1' / 'parties.json').exists()
    assert not (tmp_path / 'site' / 'v1' / 'manifest.json').exists()
    assert 'parties: 5 partidos' in capsys.readouterr().out
    env = bundle.BundleReader(fs).read_json(bundle.path_parties())
    assert env['schema'] == 'parties@1' and [row['name'] for row in env['data']['rows']][:2] == ['PP', 'PSOE']


def test_forecast_and_analysis_refresh_the_catalogue_but_the_pointers_do_not(fresh_app, tmp_path, patched, capsys):
    """`forecast` reescribe el catálogo una vez por job, también junto a `parties`; `manifest` solo no lo toca (la
    parte de `analysis` y `point` está en el test siguiente, de la Tarea 9)."""
    from mtpy.jobs.Publish import Publish

    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    fs = FileSystem(str(tmp_path / 'a'))
    path = tmp_path / 'a' / 'site' / 'v1' / 'parties.json'
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    out = capsys.readouterr().out
    assert path.exists() and out.count('parties: 5 partidos') == 1 and out.index('es: published') < out.index('parties: 5')
    path.unlink()
    # las dos acciones juntas (en otro paquete: el run id es el segundo actual) escriben el catálogo una sola vez
    Publish().run(what=['forecast', 'parties'], scopes=['es'], fs=FileSystem(str(tmp_path / 'b')))
    assert (tmp_path / 'b' / 'site' / 'v1' / 'parties.json').exists() and capsys.readouterr().out.count('parties: 5 partidos') == 1
    Publish().run(what=['manifest'], fs=fs)
    assert not path.exists() and 'parties' not in capsys.readouterr().out


@pytest.mark.skip(reason='Tarea 9')
def test_analysis_refreshes_the_catalogue_but_point_does_not(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish

    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    fs = FileSystem(str(tmp_path / 'a'))
    path = tmp_path / 'a' / 'site' / 'v1' / 'parties.json'
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    path.unlink()
    capsys.readouterr()
    Publish().run(what=['analysis'], scopes=['es'], fs=fs)
    assert path.exists() and 'parties: 5 partidos' in capsys.readouterr().out
    path.unlink()
    # las tres acciones juntas (en otro paquete: el run id es el segundo actual) escriben el catálogo una sola vez
    Publish().run(what=['forecast', 'analysis', 'parties'], scopes=['es'], fs=FileSystem(str(tmp_path / 'b')))
    assert (tmp_path / 'b' / 'site' / 'v1' / 'parties.json').exists() and capsys.readouterr().out.count('parties: 5 partidos') == 1
    Publish().run(what=['point'], scopes=['es'], run=publish.read_manifest(bundle.BundleReader(fs))['scopes']['es']['latest'], fs=fs)
    assert not path.exists() and 'parties' not in capsys.readouterr().out


def test_parties_dry_run_writes_apart_and_a_failure_fails_the_job(fresh_app, tmp_path, patched, capsys, monkeypatch):
    from mtpy.jobs.Publish import Publish

    fs = FileSystem(str(tmp_path))
    Publish().run(what=['parties'], fs=fs, dry_run=True)
    assert (tmp_path / 'site-dry' / 'v1' / 'parties.json').exists() and not (tmp_path / 'site').exists()

    def broken(writer, **kwargs):
        raise KeyError('color')

    monkeypatch.setattr(publish, 'publish_parties', broken)
    errors = []
    fresh_app.set('logger', types.SimpleNamespace(error=errors.append, info=lambda m: None))
    with pytest.raises(RuntimeError, match='parties'):
        Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    out = capsys.readouterr().out
    assert 'es: published' in out and "parties: failed (KeyError: 'color')" in out and len(errors) == 1


@pytest.mark.skip(reason='Tarea 9')
def test_publish_study_runs_the_study_action(fresh_app, tmp_path, monkeypatch, capsys):
    """`study` publica el estudio por su slug (con `register` y `force` del job), sin guardia LOREG; un registro
    rechazado queda `refused` y cualquier otro error, `failed`, que hace fallar el job."""
    from mtpy.jobs.Publish import Publish

    calls = []

    def fake_study(study, writer, register=False, force=False, **kwargs):
        calls.append((study, register, force))
        if register and not force:
            raise publish.PublishRefused('study: already registered on 2026-10-08')
        return {'study': study, 'registered_at': '2026-10-08' if register else None, 'checked': False,
                'seconds': {'total': 12.3}}

    monkeypatch.setattr(publish, 'publish_study', fake_study)
    fs = FileSystem(str(tmp_path))
    with pytest.raises(ValueError, match='"study" is required for study'):
        Publish().run(what='study', fs=fs)
    result = Publish().run(what='study', study='voto-cera', fs=fs, today='2026-11-27')
    assert result['voto-cera'] == {'status': 'published', 'registered_at': None, 'checked': False,
                                   'seconds': {'total': 12.3}}
    assert 'voto-cera: study published (sin registro, sin comprobar) in 12 s' in capsys.readouterr().out
    result = Publish().run(what=['study'], study='voto-cera', register=True, fs=fs)
    assert result['voto-cera'] == {'status': 'refused', 'reason': 'study: already registered on 2026-10-08'}
    result = Publish().run(what=['study'], study='voto-cera', register=True, force=True, fs=fs)
    assert result['voto-cera']['registered_at'] == '2026-10-08'
    assert 'voto-cera: study published (registrado el 2026-10-08, sin comprobar)' in capsys.readouterr().out
    assert calls == [('voto-cera', False, False), ('voto-cera', True, False), ('voto-cera', True, True)]
    monkeypatch.setattr(publish, 'publish_study', lambda study, writer, **kwargs: {}['x'])
    with pytest.raises(RuntimeError, match='voto-cera'):
        Publish().run(what=['study'], study='voto-cera', fs=fs)
    assert 'voto-cera: study failed' in capsys.readouterr().out


def test_publish_stop_on_error_stops_at_the_first_failed_scope_and_still_writes_the_manifest(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish
    fs = FileSystem(str(tmp_path))
    with pytest.raises(RuntimeError, match=r'publish: failed scopes: es-ar \(forecast\)'):
        Publish().run(what='forecast', scopes=['es', 'es-ar', 'es-cb'], fs=fs, stop_on_error=True)
    assert [c[0] for c in patched] == ['es', 'es-ar']
    assert publish.read_manifest(bundle.BundleReader(fs))['scopes']['es']['latest'] == patched[0][1]


def test_publish_refuses_an_empty_scope_list(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish
    with pytest.raises(ValueError, match='publish: no scopes to run'):
        Publish().run(what='forecast', scopes=[], fs=FileSystem(str(tmp_path)))


def test_publish_alert_has_the_header_and_the_failures_first(fresh_app, tmp_path, patched):
    from mtpy.jobs.Publish import Publish
    sent = []
    fresh_app.set('ps', types.SimpleNamespace(send=lambda msg, **kw: sent.append(msg)))
    with pytest.raises(RuntimeError):
        Publish().run(what='forecast', scopes='all', fs=FileSystem(str(tmp_path)))
    lines = sent[-1].splitlines()
    assert lines[0].startswith('Publish job executed in ') and lines[0].endswith(' with 1 failure')
    assert lines[1] == "es-ar: failed (KeyError: 'x')" and lines[2].startswith('es: published')
    assert 'manifest: 1 scopes, freeze off' in lines and 'parties: 5 partidos' in lines
