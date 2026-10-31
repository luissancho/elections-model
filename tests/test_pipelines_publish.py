"""Tests de los pipelines del grupo `publish` ejecutados sueltos, sin base."""
import types

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.pipelines.publish.Forecast import Forecast
from mtpy.pipelines.publish.Manifest import Manifest
from mtpy.pipelines.publish.Parties import Parties
from tests.publish_fixtures import NO_POLLS, patched, warned_forecast  # noqa: F401


def forecast(fs, scope, position=(1, 1), **kwargs):
    p = Forecast()
    p.run('publish', chunk=scope, position=position, fs=fs, verbose=0, **kwargs)
    return p


def forecast_scopes(fs, scopes, **kwargs):
    """Un mismo `Forecast` sobre varios ámbitos, como lo ejecuta el job: el pipeline y (resultado, línea) de cada
    ámbito."""
    p = Forecast()
    outcomes = []
    for index, scope in enumerate(scopes, 1):
        p.run('publish', chunk=scope, position=(index, len(scopes)), fs=fs, **{'verbose': 0, **kwargs})
        outcomes.append((p.result, p.line))
    return p, outcomes


def backfill(fs, span, today='2026-10-09', **kwargs):
    """`Forecast` de relleno de `span` (con su `finish`, que escribe el manifest como el job)."""
    p = forecast(fs, 'es', days=publish.backfill_days(span, today), today=today, **kwargs)
    p.finish(fs=fs, **kwargs)
    return p


def test_forecast_publishes_a_scope_and_keeps_its_manifest_entry_for_finish(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es', position=(1, 3))
    assert p.result['status'] == 'published' and p.result['run_id'] == p.run_id and 'entry' in p.result
    assert p.line.startswith('es: published {} (as_of '.format(p.run_id)) and p.entries == {'es': p.result['entry']}
    assert not (tmp_path / 'site' / 'v1' / 'manifest.json').exists()
    assert p.finish(fs=fs) is None and p.line == 'manifest: 1 scopes, freeze off'
    assert publish.read_manifest(bundle.BundleReader(fs))['scopes']['es']['latest'] == p.run_id


def test_forecast_statuses_skipped_failed_refused(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    assert forecast(fs, 'es-md').result['status'] == 'skipped' and forecast(fs, 'es-md').line == 'es-md: skipped (no upcoming event)'
    assert forecast(fs, 'es-cb').line.startswith('es-cb: skipped (No polls')
    assert forecast(fs, 'es-ar').line == "es-ar: failed (KeyError: 'x')" and forecast(fs, 'es-ar').failure_name('es-ar') == 'es-ar (forecast)'
    p = forecast(fs, 'es', today='2026-11-26')
    assert p.result['status'] == 'refused' and 'LOREG' in p.result['reason'] and p.entries == {}
    p.line = None
    assert p.finish(fs=fs) is None and p.line is None  # sin entradas no hay línea de manifest


def test_forecast_dry_run_writes_apart_and_finish_says_so(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es', dry_run=True)
    assert (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es').exists() and p.finish(fs=fs, dry_run=True) is None
    assert p.line == 'manifest: not written (dry run)'


def test_forecast_finish_failure_is_its_own_failure(fresh_app, tmp_path, patched, monkeypatch):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es')
    monkeypatch.setattr(publish, 'update_manifest', lambda *a, **k: (_ for _ in ()).throw(OSError('s3 down')))
    assert p.finish(fs=fs) == 'manifest' and p.line == 'manifest: failed (OSError: s3 down)'


def test_forecast_backfill_gives_one_result_and_one_line_per_day(fresh_app, tmp_path, patched, capsys):
    fs = FileSystem(str(tmp_path))
    days = publish.backfill_days({'from': '2026-10-05', 'to': '2026-10-06'}, '2026-10-09')
    p = forecast(fs, 'es', days=days, today='2026-10-09')
    assert [r['status'] for r in p.result] == ['published', 'published'] and [r['day'] for r in p.result] == days
    assert len(p.line) == 2 and p.line[0].startswith('es: published 20261005-120000') and 'backfill 2026-10-05' in p.line[0]
    assert 'publish forecast: 1 ámbito · backfill 2026-10-05..2026-10-06 (2 días)' in capsys.readouterr().out


def test_parties_and_manifest_run_once(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = Parties()
    p.run('publish', chunk=None, position=(1, 1), fs=fs)
    assert p.result == {'status': 'published', 'n_parties': 5} and p.line == 'parties: 5 partidos' and p.chunked is False
    assert p.failure_name(None) == 'parties'
    m = Manifest()
    m.run('publish', chunk=None, position=(1, 1), fs=fs, freeze={'active': True, 'message': 'Veda'})
    assert m.result['status'] == 'published' and m.line == 'manifest: 0 scopes, freeze on'
    m.run('publish', chunk=None, position=(1, 1), fs=fs, dry_run=True)
    assert m.line == 'manifest: not written (dry run)'


def test_forecast_passes_today_to_the_publication(fresh_app, tmp_path, patched):
    """El día tomado como hoy llega a `publish_forecast`, que lo da al Simulator como tope del ancla (2026-10-26)."""
    p = forecast(FileSystem(str(tmp_path)), 'es', today='2026-10-20')
    assert p.result['status'] == 'published' and patched[0][3]['today'] == '2026-10-20'


def test_loreg_window_refuses_es_unless_forced(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es', today='2026-11-26')
    assert p.result['status'] == 'refused' and 'LOREG' in p.result['reason'] and patched == []
    p = forecast(fs, 'es', today='2026-11-26', force=True)
    assert p.result['status'] == 'published' and patched[0][3]['freeze'] is True


def test_national_window_covers_the_election_day(fresh_app, tmp_path, patched, monkeypatch):
    """El día de la elección general la ventana nacional sigue abierta: el pipeline pide la próxima de `es` desde la
    víspera (desde hoy, estricto, no habría elección próxima) y un autonómico queda `refused`; al día siguiente,
    no."""
    def next_date(scope, date_from=None):
        if scope == 'es':
            return '2026-11-29' if date_from is not None and date_from <= '2026-11-28' else None
        return {'es-cb': '2027-05-23'}.get(scope)

    monkeypatch.setattr(publish, 'next_event_date', next_date)
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es-cb', today='2026-11-29')
    assert p.result['status'] == 'refused' and 'of es 2026-11-29' in p.result['reason'] and patched == []
    p = forecast(fs, 'es-cb', today='2026-11-30')
    # fuera de la ventana el doble de `es-cb` llega a publicar y queda `skipped` (no tiene sondeos)
    assert p.result['status'] == 'skipped' and [c[0] for c in patched] == ['es-cb']


def test_forecast_rebuilds_the_history_of_each_published_scope(fresh_app, tmp_path, patched):
    """Tras publicar, `history.json` del ámbito contiene exactamente la ejecución publicada."""
    fs = FileSystem(str(tmp_path))
    forecast(fs, 'es')
    assert (tmp_path / 'site' / 'v1' / 'runs' / 'es' / 'history.json').exists()
    runs = bundle.BundleReader(fs).read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in runs] == [patched[0][1]]


def test_a_value_error_other_than_nothing_to_publish_fails_the_scope(fresh_app, tmp_path, patched):
    """Sólo `NothingToPublish` omite el ámbito; cualquier otro `ValueError` (validación del paquete) falla."""
    p, outcomes = forecast_scopes(FileSystem(str(tmp_path)), ['es-ri', 'es-cb', 'es'])
    assert [result['status'] for result, _ in outcomes] == ['failed', 'skipped', 'published']
    assert outcomes[0][1] == "es-ri: failed (ValueError: vote: missing keys ['rows'])"
    assert outcomes[1][1].startswith('es-cb: skipped (No polls') and outcomes[2][1].startswith('es: published')
    assert list(p.entries) == ['es']


def test_event_date_is_normalised_or_fails_the_scope(fresh_app, tmp_path, patched):
    """`event_date` llega en ISO a `meta`, `headline` y al SQL de `db_stats`; si no es una fecha, el ámbito falla."""
    fs = FileSystem(str(tmp_path))
    p = forecast(fs, 'es', event_date='29/11/2026')
    assert p.line.startswith('es: failed (ValueError: Invalid isoformat') and patched == []
    p = forecast(fs, 'es', event_date='20261129')
    assert p.result['status'] == 'published' and patched[0][2] == '2026-11-29'


def test_backfill_respects_a_pointed_run(fresh_app, tmp_path, patched, monkeypatch):
    """Un relleno no deshace un `point` hacia atrás; una publicación normal posterior sí mueve `latest`."""
    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    run_a, run_b, run_c = '20261008-100000', '20261008-110000', '20261008-120000'
    for rid in (run_a, run_b):
        monkeypatch.setattr(bundle, 'run_id', lambda now=None, rid=rid: rid)
        forecast(fs, 'es').finish(fs=fs)
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == run_b
    publish.point(writer, 'es', run_a)
    p = backfill(fs, {'from': '2026-10-05', 'to': '2026-10-06'})
    assert [r['status'] for r in p.result] == ['published', 'published'] and p.entries == {}
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == run_a
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000', '20261006-120000', run_a, run_b]
    monkeypatch.setattr(bundle, 'run_id', lambda now=None: run_c)
    forecast(fs, 'es').finish(fs=fs)
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == run_c


def test_backfill_failed_day_is_reported_until_unpublished(fresh_app, tmp_path, patched, monkeypatch):
    """Un día que falla a medias deja su carpeta: los demás días se escriben y un reintento lo da por fallido
    (incompleto) en vez de omitirlo, hasta que se borra con `unpublish`."""
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
    p = backfill(fs, span)
    assert [r['status'] for r in p.result] == ['published', 'failed', 'published']
    assert p.line == 'manifest: 1 scopes, freeze off'
    assert 'es: failed 2026-10-06 (OSError: s3 down)' in p.summary('es')
    assert writer.exists(bundle.path_run('es', '20261006-120000'))
    assert writer.list_runs('es') == ['20261005-120000', '20261007-120000']
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == '20261007-120000'
    p = backfill(fs, span)
    assert [r['status'] for r in p.result] == ['skipped', 'failed', 'skipped']
    lines = p.summary('es')
    assert 'es: skipped 2026-10-05 (run exists)' in lines and 'es: skipped 2026-10-07 (run exists)' in lines
    assert 'es: failed 2026-10-06 (incomplete run exists; unpublish it first)' in lines
    publish.unpublish(writer, 'es', '20261006-120000')
    p = backfill(fs, span)
    assert [r['status'] for r in p.result] == ['skipped', 'published', 'skipped']
    assert writer.list_runs('es') == ['20261005-120000', '20261006-120000', '20261007-120000']


def test_backfill_replace_recomputes_the_existing_days(fresh_app, tmp_path, patched):
    """Con `replace`, un día cuyo run ya existe (completo o a medias) se borra y se recalcula en vez de saltarse;
    un día nuevo se publica como siempre."""
    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    backfill(fs, {'from': '2026-10-05', 'to': '2026-10-06'})
    stale = tmp_path / bundle.PREFIX / bundle.path_run('es', '20261005-120000') / 'stale.txt'
    stale.write_text('old')
    writer.remove(bundle.path_part('es', '20261006-120000', 'headline'))
    p = backfill(fs, {'from': '2026-10-05', 'to': '2026-10-07'}, replace=True)
    assert [r['status'] for r in p.result] == ['published', 'published', 'published']
    assert [r['replaced'] for r in p.result] == [True, True, False]
    assert not stale.exists()
    assert writer.list_runs('es') == ['20261005-120000', '20261006-120000', '20261007-120000']
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000', '20261006-120000', '20261007-120000']
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == '20261007-120000'
    lines = p.summary('es')
    assert lines[0].startswith('es: published 20261005-120000 (as_of 2026-10-05, 6 polls, backfill 2026-10-05, replaced)')
    assert 'backfill 2026-10-07)' in lines[2]


def test_backfill_replace_keeps_the_pointers_consistent_when_the_new_run_fails(fresh_app, tmp_path, patched, monkeypatch):
    """Si el run que sustituye a un día falla, el borrado ya no figura en `history.json` ni como `latest`,
    y el reintento con `replace` recalcula el run a medias que dejó."""
    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    backfill(fs, {'from': '2026-10-05', 'to': '2026-10-06'})
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == '20261006-120000'
    real_meta = publish.export_meta

    def export_meta(sim, run_id, *args, **kwargs):
        if run_id == '20261006-120000':
            raise OSError('s3 down')
        return real_meta(sim, run_id, *args, **kwargs)

    monkeypatch.setattr(publish.forecast, 'export_meta', export_meta)
    span = {'from': '2026-10-06'}
    p = backfill(fs, span, replace=True)
    assert p.summary('es') == ['es: failed 2026-10-06 (OSError: s3 down)']
    assert writer.exists(bundle.path_run('es', '20261006-120000'))
    assert writer.list_runs('es') == ['20261005-120000']
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000']
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == '20261005-120000'
    monkeypatch.setattr(publish.forecast, 'export_meta', real_meta)
    p = backfill(fs, span, replace=True)
    assert [(r['status'], r['replaced']) for r in p.result] == [('published', True)]
    assert writer.list_runs('es') == ['20261005-120000', '20261006-120000']
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == '20261006-120000'


def test_backfill_replace_leaves_a_refused_day_in_place(fresh_app, tmp_path, patched):
    """En la ventana LOREG el relleno se rechaza antes de borrar nada, también con `replace`."""
    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    backfill(fs, {'from': '2026-10-05'})
    head = writer.read_json(bundle.path_part('es', '20261005-120000', 'headline'))
    p = backfill(fs, {'from': '2026-10-05'}, today='2026-11-27', replace=True)
    assert p.result[0]['status'] == 'refused'
    assert writer.list_runs('es') == ['20261005-120000']
    assert writer.read_json(bundle.path_part('es', '20261005-120000', 'headline')) == head


def test_backfill_replace_in_a_dry_run_rewrites_the_dry_run_without_pointers(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    dry = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX)
    backfill(fs, {'from': '2026-10-05'}, dry_run=True)
    p = backfill(fs, {'from': '2026-10-05'}, replace=True, dry_run=True)
    assert [(r['status'], r['replaced']) for r in p.result] == [('published', True)]
    assert p.line == 'manifest: not written (dry run)'
    assert dry.list_runs('es') == ['20261005-120000']
    assert not dry.exists(bundle.path_history('es')) and not dry.exists(bundle.path_manifest())
    assert not (tmp_path / 'site').exists()


def test_forecast_collects_the_model_warnings_into_the_result(fresh_app, tmp_path, patched, monkeypatch, recwarn):
    """Los avisos del modelo no salen sueltos: van al resultado (únicos, en orden) y al recuento de la línea."""
    warned_forecast(monkeypatch, [NO_POLLS, NO_POLLS])
    p = forecast(FileSystem(str(tmp_path)), 'es')
    assert p.result['status'] == 'published' and p.result['warnings'] == [NO_POLLS]
    assert p.line.startswith('es: published') and p.line.endswith(' · 1 aviso')
    assert NO_POLLS not in p.line and len(recwarn) == 0


def test_forecast_keeps_the_warnings_of_a_skipped_or_failed_scope(fresh_app, tmp_path, patched, monkeypatch, capsys,
                                                                  recwarn):
    """Un ámbito que avisa y luego se salta o rompe lleva sus avisos (sin repetir) y nada escapa del pipeline."""
    import warnings

    def fake(scope, writer, run_id, event_date, **kwargs):
        warnings.warn('Fewer than three usable polls for `A`: not fitted')
        warnings.warn('Fewer than three usable polls for `B`: not fitted')
        warnings.warn('Fewer than three usable polls for `A`: not fitted')
        if scope == 'es-cb':
            raise publish.NothingToPublish('Not enough polls to fit the average')
        raise KeyError('x')

    monkeypatch.setattr(publish, 'publish_forecast', fake)
    _, outcomes = forecast_scopes(FileSystem(str(tmp_path)), ['es-cb', 'es-ar'])
    assert [line for _, line in outcomes] == ['es-cb: skipped (Not enough polls to fit the average) · 2 avisos',
                                              "es-ar: failed (KeyError: 'x') · 2 avisos"]
    assert 'not fitted' not in capsys.readouterr().out and len(recwarn) == 0


def test_forecast_without_model_warnings_has_none(fresh_app, tmp_path, patched):
    _, outcomes = forecast_scopes(FileSystem(str(tmp_path)), ['es', 'es-md'])
    assert [result['warnings'] for result, _ in outcomes] == [[], []]


def test_verbose_0_prints_only_the_header(fresh_app, tmp_path, patched, monkeypatch, capsys, recwarn):
    warned_forecast(monkeypatch, [NO_POLLS])
    p = forecast(FileSystem(str(tmp_path)), 'es')
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith('[') and 'publish forecast: 1 ámbito · run ' in lines[0]
    assert p.line.endswith(' · 1 aviso') and len(recwarn) == 0


def test_verbose_1_is_the_default_and_prints_the_scope_lines_and_warnings_live(fresh_app, tmp_path, patched,
                                                                             monkeypatch, capsys, recwarn):
    warned_forecast(monkeypatch, [NO_POLLS])
    p = Forecast()
    p.run('publish', chunk='es', position=(1, 1), fs=FileSystem(str(tmp_path)))
    out = capsys.readouterr().out
    assert out.index('[1/1] es: empezando') < out.index('es · aviso: ' + NO_POLLS) < out.index('[1/1] es: published')
    assert out.count(NO_POLLS) == 1 and 'simulador listo' not in out and len(recwarn) == 0


def test_verbose_2_adds_the_phase_lines(fresh_app, tmp_path, patched, capsys):
    forecast_scopes(FileSystem(str(tmp_path)), ['es'], verbose=2)
    out = capsys.readouterr().out
    assert out.index('[1/1] es: empezando') < out.index('[1/1] es: simulador listo (') < out.index('[1/1] es: meta y titular (')


def test_the_model_gets_verbose_minus_two(fresh_app, tmp_path, patched):
    for level, expected in ((0, 0), (1, 0), (2, 0), (3, 1)):
        patched.clear()
        forecast_scopes(FileSystem(str(tmp_path / str(level))), ['es'], verbose=level)
        assert patched[0][3]['verbose'] == expected


def test_forecast_prints_its_progress_live(fresh_app, tmp_path, patched, capsys):
    infos = []
    fresh_app.set('logger', types.SimpleNamespace(error=lambda m: None, info=infos.append))
    p, _ = forecast_scopes(FileSystem(str(tmp_path)), ['es', 'es-cb'], verbose=2)
    lines = capsys.readouterr().out.splitlines()
    expected = ['publish forecast: 2 ámbitos · run ' + p.run_id, '[1/2] es: empezando', '[1/2] es: simulador listo (',
                '[1/2] es: promedio ajustado (', '[1/2] es: simulación de hoy (', '[1/2] es: exportación de hoy (',
                '[1/2] es: simulación de la elección (', '[1/2] es: exportación (', '[1/2] es: meta y titular (',
                '[1/2] es: published ', '[2/2] es-cb: empezando', '[2/2] es-cb: skipped (No polls']
    found = [next(i for i, line in enumerate(lines) if text in line) for text in expected]
    assert found == sorted(found)
    assert all(lines[i].startswith('[') and lines[i][9:11] == '] ' for i in found)  # [hh:mm:ss] delante
    assert any('[1/2] es: promedio ajustado (' in m for m in infos)


def test_backfill_progress_names_the_day(fresh_app, tmp_path, patched, capsys):
    days = publish.backfill_days({'from': '2026-10-05', 'to': '2026-10-06'}, '2026-10-09')
    forecast_scopes(FileSystem(str(tmp_path)), ['es'], days=days, today='2026-10-09', verbose=2)
    out = capsys.readouterr().out
    assert 'publish forecast: 1 ámbito · backfill 2026-10-05..2026-10-06 (2 días)' in out
    assert out.index('[1/1] es 2026-10-05: empezando') < out.index('[1/1] es 2026-10-06: promedio ajustado (')
