"""Tests del job `update` (`mtpy/jobs/Update.py`): la cadena load → compute → publish con las fábricas falsas de los
tres grupos y la base sustituida."""
import importlib

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import compute as compute_lib
from mtpy.lib import publish
from tests.publish_fixtures import patched  # noqa: F401
from tests.test_compute_unit import FakeComputer
from tests.test_load_unit import FakeCatalogues, FakeLoader

module = importlib.import_module('mtpy.jobs.Update')
Update = module.Update

EVENTS = {'es': '2026-11-29', 'es-md': '2027-05-23'}


@pytest.fixture(autouse=True)
def marks(monkeypatch):
    """`compute_scope` sin base: el reloj y la marca de `computed_at` sustituidos (anota las marcas)."""
    seen = []
    monkeypatch.setattr(compute_lib, 'db_now', lambda: 'T0')
    monkeypatch.setattr(compute_lib, 'mark_computed', lambda scope, at: seen.append((scope, at)))
    return seen


@pytest.fixture(autouse=True)
def studies(monkeypatch):
    """`publish_study` sin S3 ni bundle: devuelve un resultado publicado (anota los estudios)."""
    seen = []
    monkeypatch.setattr(publish, 'publish_study', lambda study, writer, **kw: (seen.append(study) or {
        'study': study, 'registered_at': None, 'checked': False, 'seconds': {'total': 1.0}}))
    return seen


@pytest.fixture
def scopes(monkeypatch):
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: EVENTS.get(scope))
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: event_date != '2027-05-23')


@pytest.fixture
def alerts(monkeypatch):
    seen = []
    monkeypatch.setattr(Update, 'alert', lambda self, msg: seen.append(msg))
    return seen


def run(tmp_path, loader=None, computer=None, pending=None, results=None, **kwargs):
    """Por defecto, la rutina diaria: una elección pendiente sin resultado (compute `light`)."""
    computer = computer or FakeComputer()
    outcome = Update().run(fs=FileSystem(str(tmp_path)), loader=loader or FakeLoader(), computer=computer,
                           catalogues=FakeCatalogues(), featured=lambda scope: 7,
                           pending=pending or (lambda scope: ['2026-11-29']),
                           results=results or (lambda scope: ['2023-07-23']), **{'verbose': 0, 'n_sim': 20, **kwargs})
    return outcome, computer


def test_update_runs_the_three_groups_on_es_with_one_alert(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'])
    out = capsys.readouterr().out
    assert 'load polls: 1 ámbito' in out and 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s' in out
    assert 'compute series: 1 ámbito' in out
    assert 'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s' in out
    assert 'publish forecast: 1 ámbito' in out and 'es: published' in out and 'es: analysis published' in out
    assert 'parties: ' in out and 'manifest: 1 scopes, freeze off' in out
    assert 'studies: ' in out and out.index('parties: ') < out.index('studies: ')
    # un Computer de los pesos de la carga (2 sondeos guardados) y dos del compute light (pesos y rebaño)
    assert [scope for scope, _ in computer.inits] == ['es'] * 3
    assert set(results) == {'es', 'parties', 'studies'} and set(results['es']) == {'polls', 'compute', 'forecast', 'analysis'}
    assert results['es']['forecast']['status'] == 'published' and results['parties']['status'] == 'published'
    assert len(alerts) == 1 and alerts[0].startswith('Update job executed in')
    assert patched[0][3]['n_sim'] == 20
    assert (tmp_path / 'site' / 'v1' / 'manifest.json').exists()


def test_update_publishes_the_catalogue_even_when_a_study_fails(fresh_app, tmp_path, scopes, events, patched, alerts,
                                                                monkeypatch):
    def boom(study, writer, **kw):
        raise KeyError(study)

    monkeypatch.setattr(publish, 'publish_study', boom)
    with pytest.raises(RuntimeError, match='studies'):
        Update().run(fs=FileSystem(str(tmp_path)), loader=FakeLoader(), computer=FakeComputer(),
                     catalogues=FakeCatalogues(), featured=lambda scope: 7, pending=lambda scope: ['2026-11-29'],
                     results=lambda scope: ['2023-07-23'], verbose=0, n_sim=20, scopes=['es'])
    assert (tmp_path / 'site' / 'v1' / 'parties.json').exists()


def test_update_skips_the_compute_of_a_scope_with_nothing_pending(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'], pending=lambda scope: [])
    assert computer.inits == [('es', 0)] and results['es']['compute']['status'] == 'skipped'  # sólo los pesos de la carga
    assert 'es: series skipped (nothing loaded since the last compute)' in capsys.readouterr().out
    assert results['es']['forecast']['status'] == 'published'


def test_update_overwrite_reloads_the_cycle_and_computes_anyway(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    loader = FakeLoader()
    results, computer = run(tmp_path, loader=loader, scopes=['es'], overwrite=True, pending=lambda scope: [])
    out = capsys.readouterr().out
    assert 'load polls: 1 ámbito · overwrite' in out and ('select_series', True) in loader.calls
    assert 'compute series: 1 ámbito · overwrite' in out and [scope for scope, _ in computer.inits] == ['es'] * 4


def test_update_stops_at_the_first_failure_and_publishes_nothing(fresh_app, tmp_path, scopes, events, patched, alerts):
    with pytest.raises(RuntimeError, match=r'update: failed scopes: es \(polls\)'):
        run(tmp_path, loader=FakeLoader(missing={'pollsters': ['Casa Nueva']}), scopes=['es'])
    assert not (tmp_path / 'site').exists() and len(alerts) == 1 and 'with 1 failure' in alerts[0]
    with pytest.raises(RuntimeError, match=r'update: failed scopes: es \(series\)'):
        run(tmp_path, computer=FakeComputer(fail_at='herding'), scopes=['es'])
    assert not (tmp_path / 'site').exists()


def test_update_without_stop_on_error_publishes_what_it_has(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    with pytest.raises(RuntimeError, match=r'update: failed scopes: es \(polls\)'):
        run(tmp_path, loader=FakeLoader(missing={'pollsters': ['Casa Nueva']}), scopes=['es'], stop_on_error=False,
            pending=lambda scope: [])
    out = capsys.readouterr().out
    assert 'es: polls failed (' in out and 'es: series skipped (' in out and 'es: published' in out
    assert (tmp_path / 'site' / 'v1' / 'manifest.json').exists()


def test_update_all_skips_a_scope_without_article_and_goes_on(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, _ = run(tmp_path, scopes='all')
    out = capsys.readouterr().out
    assert 'es-md: polls skipped (no polls article for 2027-05-23)' in out
    assert results['es-md']['polls']['status'] == 'skipped' and results['es']['forecast']['status'] == 'published'
    assert results['es-md']['forecast']['status'] == 'skipped'  # sin evento próximo en `patched`


def test_update_dry_run_touches_nothing(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    loader = FakeLoader()
    results, computer = run(tmp_path, loader=loader, scopes=['es'], dry_run=True, pending=lambda scope: [])
    out = capsys.readouterr().out
    assert 'es: polls checked (5 read, 2 new, 2026-11-29, dry run) in 0 s' in out
    assert computer.inits == [] and 'es: series skipped (' in out
    assert (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es').exists() and not (tmp_path / 'site').exists()


def test_update_rejects_event_date_and_years_with_overwrite(fresh_app, tmp_path, scopes, events):
    with pytest.raises(ValueError, match='update: "event_date" is not accepted'):
        run(tmp_path, scopes=['es'], event_date='2023-07-23')
    with pytest.raises(ValueError, match='update: "what" is not accepted'):
        run(tmp_path, scopes=['es'], what=['forecast'])
    with pytest.raises(ValueError, match='"years" cannot be combined with "overwrite"'):
        run(tmp_path, scopes=['es'], years=[2026], overwrite=True)


def test_update_fails_early_without_a_file_system_or_scopes(fresh_app, scopes, events):
    fresh_app.set('fs', None)
    with pytest.raises(RuntimeError, match='update: no file system configured'):
        Update().run(scopes=['es'])
    with pytest.raises(ValueError, match='update: no scopes to run'):
        Update().run(scopes=[], fs=object())


def test_update_computes_full_when_a_pending_election_has_results(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'], pending=lambda scope: ['2023-07-23', '2026-11-29'])
    assert [scope for scope, _ in computer.inits] == ['es'] * 4 and results['es']['compute']['mode'] == 'full'
    assert 'es: series computed (full, results pending: 2 polls, 2 events, 7 featured) in 0 s' in capsys.readouterr().out


def test_update_passes_a_forced_mode_to_the_compute(fresh_app, tmp_path, scopes, events, patched, alerts, capsys):
    results, computer = run(tmp_path, scopes=['es'], mode='full')
    out = capsys.readouterr().out
    assert 'compute series: 1 ámbito · full' in out and results['es']['compute']['mode_reason'] == 'requested'


def test_update_rejects_an_unknown_mode_before_loading(fresh_app, tmp_path, scopes, events):
    loader = FakeLoader()
    with pytest.raises(ValueError, match='compute: unknown mode "x" \\(expected auto, light or full\\)'):
        run(tmp_path, loader=loader, scopes=['es'], mode='x')
    assert loader.calls == []


def test_update_lists_the_new_polls_in_the_alert(fresh_app, tmp_path, scopes, events, patched, alerts):
    run(tmp_path, scopes=['es'])
    body = alerts[0].splitlines()
    loaded = next(i for i, line in enumerate(body) if line.startswith('es: polls loaded'))
    assert body[loaded + 1:loaded + 3] == ['es: + Casa 1, 2026-11-09 → 2026-11-11',
                                           'es: + Casa 0 / Medio, 2026-11-10 → 2026-11-12']
