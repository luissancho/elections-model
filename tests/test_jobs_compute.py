"""Tests del job `compute` (`mtpy/jobs/Compute.py`) con el `Computer` y el catálogo de ámbitos sustituidos."""
import importlib

import pytest

from tests.test_compute_unit import FakeComputer

module = importlib.import_module('mtpy.jobs.Compute')
Compute = module.Compute


@pytest.fixture
def scopes(monkeypatch):
    """`resolve_scopes` sin base: `'all'` son `es` y `es-md`; una lista se devuelve tal cual."""
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))


def run(factory, featured=None, **kwargs):
    seen = []
    featured = featured if featured is not None else (lambda scope: seen.append(scope) or 7)
    return Compute().run(computer=factory, featured=featured, **kwargs), seen


def test_compute_runs_the_series_of_every_scope_and_then_the_ratings(fresh_app, scopes, capsys):
    factory = FakeComputer()
    results, featured = run(factory, what=['compute', 'ratings'], scopes='all', verbose=0)
    # la serie de todos los ámbitos primero (tres Computer por ámbito) y los ratings después (uno por ámbito)
    assert [scope for scope, _ in factory.inits] == ['es'] * 3 + ['es-md'] * 3 + ['es', 'es-md']
    assert featured == ['es', 'es-md']
    assert results['es'] == {'status': 'computed', 'action': 'ratings', 'n_pollsters': 2, 'n_events': 1,
                             'seconds': results['es']['seconds'], 'warnings': []}
    out = capsys.readouterr().out
    assert 'compute series: 2 ámbitos' in out and 'compute ratings: 2 ámbitos' in out
    assert 'es: series computed (2 polls, 2 events, 7 featured) in 0 s' in out
    assert 'es-md: ratings computed (2 pollsters, 1 events) in 0 s' in out
    assert 'empezando' not in out and 'pesos' not in out  # nivel 0: cabecera y resumen


def test_compute_isolates_a_failing_scope_and_fails_at_the_end(fresh_app, scopes, capsys):
    factory = FakeComputer(fail_at='errors', fail_scope='es-ar')
    with pytest.raises(RuntimeError, match=r'compute: failed scopes: es-ar \(series\)'):
        run(factory, what=['compute', 'ratings'], scopes=['es', 'es-ar'], verbose=0)
    out = capsys.readouterr().out
    assert 'es: series computed (2 polls, 2 events, 7 featured) in 0 s' in out
    assert 'es-ar: series failed (RuntimeError: boom errors)' in out
    assert 'es-ar: ratings computed (2 pollsters, 1 events) in 0 s' in out  # las acciones siguientes siguen


def test_compute_dry_run_saves_nothing_and_says_so(fresh_app, scopes, capsys):
    factory = FakeComputer()
    results, featured = run(factory, what='compute', scopes=['es'], dry_run=True, verbose=0)
    assert all(call[1] is False for call in factory.calls if call[0] != 'build_series') and featured == []
    assert results['es']['status'] == 'computed' and results['es']['featured'] is None
    out = capsys.readouterr().out
    assert 'compute series: 1 ámbito · dry run' in out and 'es: series computed (2 polls, 2 events, dry run) in 0 s' in out


def test_compute_verbosity_levels(fresh_app, scopes, capsys):
    factory = FakeComputer()
    run(factory, what='compute', scopes=['es'], verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '[1/1] es: series computed' in out and 'pesos (' not in out
    assert factory.inits[0] == ('es', 0)
    factory = FakeComputer()
    run(factory, what=['compute', 'ratings'], scopes=['es'], verbose=2)
    out = capsys.readouterr().out
    for label in ('pesos', 'eventos destacados', 'errores', 'desviaciones', 'deriva', 'efectos de casa', 'rebaño', 'ratings'):
        assert '[1/1] es: {} ('.format(label) in out, label
    factory = FakeComputer()
    run(factory, what='ratings', scopes=['es'], verbose=3)
    assert factory.inits == [('es', 1)]  # el Computer recibe verbose - 2


def test_compute_ratings_only_and_unknown_actions(fresh_app, scopes):
    factory = FakeComputer()
    results, featured = run(factory, what='ratings', scopes=['es-md'])
    assert [scope for scope, _ in factory.inits] == ['es-md'] and featured == []
    assert results['es-md']['action'] == 'ratings' and results['es-md']['status'] == 'computed'
    with pytest.raises(ValueError, match='compute: unknown what "x"'):
        run(FakeComputer(), what=['x'], scopes=['es'])


def test_compute_reports_the_model_warnings_as_notes(fresh_app, scopes, capsys):
    """Nivel 1: los avisos del modelo salen como `· aviso:` del ámbito, no crudos por stderr, y se cuentan en la
    línea de resumen y en el resultado, como en `publish`."""
    import warnings

    class WarningComputer(FakeComputer):
        def __call__(self, scope, verbose=0, **kwargs):
            comp = super().__call__(scope, verbose, **kwargs)
            errors = comp.compute_errors

            def compute_errors(save=False):
                warnings.warn('serie corta', RuntimeWarning)
                return errors(save=save)
            comp.compute_errors = compute_errors
            return comp

    results, _ = run(WarningComputer(), what='compute', scopes=['es'], verbose=1)
    out, err = capsys.readouterr()
    assert '[1/1] es · aviso: serie corta' in out and 'serie corta' not in err
    assert results['es']['warnings'] == ['serie corta']
    assert 'es: series computed (2 polls, 2 events, 7 featured) in 0 s · 1 aviso' in out


def test_compute_catalogues_runs_first_once_and_dry_run_only_checks(fresh_app, scopes, capsys):
    """`catalogues` es global (no por ámbito) y va antes que las series y los ratings, que leen los ámbitos y la
    calidad que escribe; en seco sólo valida."""
    calls = []

    def catalogues(save):
        calls.append(('catalogues', save))
        figures = {'scopes': 18, 'districts': 118, 'quality': 72, 'pollsters': 135} if save else {'quality': 72}
        return dict(figures, seconds={'total': 0.0})

    factory = FakeComputer()
    results, _ = run(factory, catalogues=catalogues, what=['ratings', 'catalogues'], scopes=['es'], verbose=0)
    assert calls == [('catalogues', True)] and [scope for scope, _ in factory.inits] == ['es']
    assert results['catalogues']['status'] == 'computed' and results['es']['action'] == 'ratings'
    out = capsys.readouterr().out
    assert out.index('compute catálogos') < out.index('compute ratings: 1 ámbito')
    assert 'catalogues: saved (18 scopes, 118 districts, 72 quality evaluations, 135 pollsters) in 0 s' in out
    run(FakeComputer(), catalogues=catalogues, what='catalogues', scopes=['es'], dry_run=True, verbose=0)
    assert calls[-1] == ('catalogues', False)
    assert 'catalogues: checked (72 quality evaluations, dry run) in 0 s' in capsys.readouterr().out


def test_compute_catalogues_failure_fails_the_job_at_the_end(fresh_app, scopes, capsys):
    def catalogues(save):
        raise ValueError('pollsters-quality.csv: standards must be between 0 and 1')

    factory = FakeComputer()
    with pytest.raises(RuntimeError, match=r'catalogues \(catálogos\)'):
        run(factory, catalogues=catalogues, what=['catalogues', 'ratings'], scopes=['es'], verbose=0)
    out = capsys.readouterr().out
    assert 'catalogues: catálogos failed (ValueError: pollsters-quality.csv: standards' in out
    assert 'es: ratings computed' in out  # las acciones siguientes siguen


def test_compute_stop_on_error_stops_before_the_ratings(fresh_app, scopes):
    factory = FakeComputer(fail_at='errors', fail_scope='es')
    with pytest.raises(RuntimeError, match=r'compute: failed scopes: es \(series\)'):
        run(factory, what=['compute', 'ratings'], scopes=['es', 'es-md'], stop_on_error=True)
    assert [scope for scope, _ in factory.inits] == ['es'] * 2  # la serie de es se corta en los errores; ni es-md ni los ratings


def test_compute_refuses_an_empty_scope_list(fresh_app, scopes):
    with pytest.raises(ValueError, match='compute: no scopes to run'):
        run(FakeComputer(), scopes=[])
