"""Tests del job `load` (`mtpy/jobs/Load.py`) con el loader, el `Computer` y los catálogos sustituidos."""
import importlib

import pytest

from tests.test_load_unit import FakeCatalogues, FakeComputer, FakeLoader

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
    monkeypatch.setattr(module, 'next_event_date', lambda scope: EVENTS.get(scope))
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: event_date != '2027-05-23')


def run(loader, **kwargs):
    return Load().run(loader=loader, computer=FakeComputer(), catalogues=FakeCatalogues(), **kwargs)


def test_load_runs_every_scope_and_prints_the_summary(fresh_app, scopes, events, capsys):
    with pytest.raises(RuntimeError, match=r'load: failed scopes: es-md \(polls\)'):
        run(FakeLoader(), what='polls', scopes='all', verbose=0)
    out = capsys.readouterr().out
    assert 'load polls: 2 ámbitos' in out
    assert 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s' in out
    assert 'es-md: polls failed (no polls article for 2027-05-23)' in out
    assert 'empezando' not in out


def test_load_returns_the_result_of_each_scope(fresh_app, scopes, events):
    results = run(FakeLoader(), scopes=['es'], verbose=0)
    assert results['es']['status'] == 'loaded' and results['es']['warnings'] == []


def test_load_header_flags_and_overwrite_line(fresh_app, scopes, events, capsys):
    results = run(FakeLoader(), scopes=['es'], overwrite=True, force=True, dry_run=True, verbose=0)
    out = capsys.readouterr().out
    assert 'load polls: 1 ámbito · overwrite · force · dry run' in out
    assert 'es: polls checked (5 read, 5 new, 2026-11-29, dry run) in 0 s' in out
    assert results['es']['status'] == 'checked'
    results = run(FakeLoader(), scopes=['es'], overwrite=True, verbose=0)
    assert 'es: polls loaded (5 read, 5 saved, 2026-11-29, overwrite) in 0 s' in capsys.readouterr().out
    assert results['es']['status'] == 'loaded'


def test_load_event_date_parameter_and_scope_without_upcoming_event(fresh_app, scopes, events, capsys, monkeypatch):
    loader = FakeLoader()
    run(loader, scopes=['es'], event_date='2023-07-23', verbose=0)
    assert loader.inits[0][:2] == ('es', '2023-07-23')
    monkeypatch.setattr(module, 'next_event_date', lambda scope: None)
    loader = FakeLoader()
    with pytest.raises(RuntimeError, match=r'load: failed scopes: es \(polls\)'):
        run(loader, scopes=['es'], verbose=0)
    assert loader.inits == []
    assert 'es: polls failed (no upcoming event)' in capsys.readouterr().out


def test_load_unmapped_names_fail_the_scope_and_force_adds_them(fresh_app, scopes, events, capsys):
    with pytest.raises(RuntimeError):
        run(FakeLoader(missing={'pollsters': ['Sumar']}), scopes=['es'], verbose=1)
    out = capsys.readouterr().out
    assert 'es: polls failed (unmapped pollsters: Sumar)' in out
    assert "Missing Pollsters: {'Sumar'}" in out
    results = run(FakeLoader(missing={'pollsters': ['Sumar']}), scopes=['es'], force=True, verbose=0)
    out = capsys.readouterr().out
    assert 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s · 1 pollster added · 1 aviso' in out
    assert results['es']['added']['pollsters'] == ['Sumar'] and len(results['es']['warnings']) == 1


def test_load_line_counts_added_pollsters_and_sponsors_to_add(fresh_app, scopes, events, capsys):
    missing = {'pollsters': ['Sumar', 'UJA'], 'sponsors': ['Artículo 14']}
    run(FakeLoader(missing=missing), scopes=['es'], force=True, verbose=0)
    assert ' · 2 pollsters added, 1 sponsor added' in capsys.readouterr().out
    run(FakeLoader(missing=missing), scopes=['es'], force=True, dry_run=True, verbose=0)
    out = capsys.readouterr().out
    assert 'dry run) in 0 s · 2 pollsters to add, 1 sponsor to add' in out


def test_load_verbosity_levels(fresh_app, scopes, events, capsys):
    run(FakeLoader(), scopes=['es'], verbose=0)
    out = capsys.readouterr().out
    assert 'empezando' not in out and 'Missing' not in out
    run(FakeLoader(), scopes=['es'], verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '# Polls: 5' in out and 'Missing Parties: set()' in out
    assert 'lectura (' not in out
    loader = FakeLoader(skipped={'pollsters': ['CIS (x)']})
    run(loader, scopes=['es'], verbose=2)
    out = capsys.readouterr().out
    for label in ('lectura', 'nombres', 'series', 'guardado', 'pesos'):
        assert '[1/1] es: {} ('.format(label) in out, label
    assert 'Skipped pollsters' in out
    loader = FakeLoader()
    run(loader, scopes=['es'], verbose=3)
    assert loader.inits[0][3] == 1  # el loader recibe verbose - 2


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
    with pytest.raises(ValueError, match=r'load: unknown what "x" \(expected one of polls\)'):
        run(FakeLoader(), what=['x'], scopes=['es'])
