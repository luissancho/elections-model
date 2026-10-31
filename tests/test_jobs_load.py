"""Tests del job `load` (`mtpy/jobs/Load.py`) con el loader, el `Computer` y los catálogos sustituidos."""
import importlib
import types

import pytest

from mtpy.lib.load import NO_EVENTS
from tests.test_load_unit import FakeCatalogues, FakeComputer, FakeLoader
from tests.test_pipelines_load import FakeParamsLoader

module = importlib.import_module('mtpy.jobs.Load')
Load = module.Load

EVENTS = {'es': '2026-11-29', 'es-md': '2027-05-23'}


@pytest.fixture
def scopes(monkeypatch):
    """`resolve_scopes` sin base: `'all'` son `es` y `es-md`; una lista se devuelve tal cual."""
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))


@pytest.fixture
def events(monkeypatch):
    """Próxima elección sin base y artículo de sondeos para todas las fechas salvo la de `es-md`."""
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: EVENTS.get(scope))
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: event_date != '2027-05-23')


def run(loader, **kwargs):
    return Load().run(loader=loader, computer=FakeComputer(), catalogues=FakeCatalogues(), **kwargs)


def test_load_runs_every_scope_and_prints_the_summary(fresh_app, scopes, events, capsys):
    # With "all", a scope whose event has no polls article is skipped, not failed
    results = run(FakeLoader(), what='polls', scopes='all', verbose=0)
    out = capsys.readouterr().out
    assert 'load polls: 2 ámbitos' in out
    assert 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s' in out
    assert 'es-md: polls skipped (no polls article for 2027-05-23)' in out
    assert 'empezando' not in out
    assert results['es-md']['status'] == 'skipped' and results['es']['status'] == 'loaded'


def test_load_scope_named_explicitly_without_article_fails(fresh_app, scopes, events, capsys):
    with pytest.raises(RuntimeError, match=r'load: failed scopes: es-md \(polls\)'):
        run(FakeLoader(), scopes=['es-md'], verbose=0)
    assert 'es-md: polls failed (no polls article for 2027-05-23)' in capsys.readouterr().out


def test_load_returns_the_result_of_each_scope(fresh_app, scopes, events):
    results = run(FakeLoader(), scopes=['es'], verbose=0)
    assert results['es']['status'] == 'loaded' and results['es']['warnings'] == []


def test_load_header_flags_and_overwrite_line(fresh_app, scopes, events, capsys):
    results = run(FakeLoader(), scopes=['es'], overwrite=True, force=True, dry_run=True, verbose=0)
    out = capsys.readouterr().out
    assert 'load polls: 1 ámbito · overwrite · force · dry run' in out
    assert 'es: polls checked (5 read, 5 new, 2026-11-29, overwrite, dry run) in 0 s' in out
    assert results['es']['status'] == 'checked'
    results = run(FakeLoader(), scopes=['es'], overwrite=True, verbose=0)
    assert 'es: polls loaded (5 read, 5 saved, 2026-11-29, overwrite) in 0 s' in capsys.readouterr().out
    assert results['es']['status'] == 'loaded'


def test_load_event_date_parameter_and_scope_without_upcoming_event(fresh_app, scopes, events, capsys, monkeypatch):
    loader = FakeLoader()
    run(loader, scopes=['es'], event_date='2023-07-23', verbose=0)
    assert loader.inits[0][:2] == ('es', '2023-07-23')
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: None)
    loader = FakeLoader()
    with pytest.raises(RuntimeError, match=r'load: failed scopes: es \(polls\)'):
        run(loader, scopes=['es'], verbose=0)
    assert loader.inits == []
    assert 'es: polls failed (no upcoming event)' in capsys.readouterr().out
    results = run(FakeLoader(), scopes='all', verbose=0)
    assert 'es: polls skipped (no upcoming event)' in capsys.readouterr().out
    assert results['es']['status'] == results['es-md']['status'] == 'skipped'


def test_load_years_are_passed_as_strings(fresh_app, scopes, events):
    loader = FakeLoader()
    run(loader, scopes=['es'], years=[2025, '2026'], verbose=0)
    assert loader.inits[0][2] == ['2025', '2026']


def test_load_refuses_years_with_overwrite(fresh_app, scopes, events):
    loader = FakeLoader()
    with pytest.raises(ValueError, match='cannot be combined'):
        run(loader, scopes=['es'], years=['2026'], overwrite=True, verbose=0)
    assert loader.inits == []


def test_load_isolates_a_failing_scope_and_reports_the_exception(fresh_app, scopes, events, capsys, monkeypatch):
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: True)

    class Exploding(FakeLoader):
        def __call__(self, scope, *args, **kwargs):
            if scope == 'es-md':
                raise RuntimeError('boom')
            return super().__call__(scope, *args, **kwargs)

    with pytest.raises(RuntimeError, match=r'load: failed scopes: es-md \(polls\)'):
        run(Exploding(), scopes='all', verbose=0)
    out = capsys.readouterr().out
    assert 'es-md: polls failed (RuntimeError: boom)' in out
    assert 'es: polls loaded' in out


def test_load_unknown_action(fresh_app, scopes, events):
    with pytest.raises(ValueError, match=r'load: unknown what "x" \(expected one of polls, params\)'):
        run(FakeLoader(), what=['x'], scopes=['es'])


# --- acción `params` ---

def test_load_params_runs_every_scope_and_prints_the_summary(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader()
    out = Load().run(what='params', scopes='all', params_loader=loader, verbose=1)
    text = capsys.readouterr().out
    assert [call[0] for call in loader.calls] == ['es', 'es-md'] and all(call[1] for call in loader.calls)
    assert 'load params: 2 ámbitos' in text
    assert 'es: params loaded (3 elections, 2 changed, 1 skipped) in 0 s' in text
    assert 'es 2023-07-23: bmaps.vs null -> {}' in text
    assert out['es']['status'] == 'loaded'


def test_load_params_scope_without_elections_is_skipped_with_all_and_fails_when_named(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader(fail={'es-md': ValueError(NO_EVENTS.format('es-md'))})
    out = Load().run(what='params', scopes='all', params_loader=loader)
    assert out['es-md']['status'] == 'skipped' and 'es-md: params skipped (no elections for es-md)' in capsys.readouterr().out
    with pytest.raises(RuntimeError, match=r'es-md \(params\)'):
        Load().run(what='params', scopes=['es-md'], params_loader=loader)


def test_load_params_isolates_an_exception(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader(fail={'es': KeyError('boom')})
    with pytest.raises(RuntimeError, match=r'es \(params\)'):
        Load().run(what='params', scopes='all', params_loader=loader)
    text = capsys.readouterr().out
    assert "es: params failed (KeyError: 'boom')" in text and 'es-md: params loaded' in text


def test_load_both_actions_run_polls_first(fresh_app, scopes, events, capsys):
    loader = FakeParamsLoader()
    run(FakeLoader(), what=['polls', 'params'], scopes=['es'], params_loader=loader)
    text = capsys.readouterr().out
    assert text.index('load polls: 1 ámbito') < text.index('load params: 1 ámbito') and loader.calls == [('es', True, 0)]


def test_load_translates_what_and_scopes_and_returns_by_action(fresh_app, scopes, events):
    loader = FakeParamsLoader()
    out = run(FakeLoader(), what=['polls', 'params'], scopes=['es'], params_loader=loader)
    assert set(out['es']) == {'polls', 'params'} and out['es']['polls']['status'] == 'loaded'


def test_load_stop_on_error_stops_at_the_first_failed_scope(fresh_app, scopes, events):
    loader = FakeParamsLoader(fail={'es': KeyError('boom')})
    with pytest.raises(RuntimeError, match=r'load: failed scopes: es \(params\)'):
        Load().run(what='params', scopes='all', params_loader=loader, stop_on_error=True)
    assert [call[0] for call in loader.calls] == ['es']


def test_load_refuses_an_empty_scope_list(fresh_app, scopes, events):
    with pytest.raises(ValueError, match='load: no scopes to run'):
        Load().run(scopes=[])


def test_load_alert_puts_the_failures_first(fresh_app, scopes, events):
    sent = []
    fresh_app.set('ps', types.SimpleNamespace(send=lambda msg, **kw: sent.append(msg)))
    with pytest.raises(RuntimeError):
        Load().run(what='params', scopes='all', params_loader=FakeParamsLoader(fail={'es-md': KeyError('boom')}))
    lines = sent[-1].splitlines()
    assert lines[0].endswith('with 1 failure') and lines[1].startswith('es-md: params failed') and lines[2].startswith('es: params loaded')
