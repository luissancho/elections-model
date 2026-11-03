"""Tests de los pipelines `compute.Compute`, `compute.Ratings`, `compute.AverageErrors` y `compute.Catalogues`
ejecutados sueltos, sin base."""
import warnings

import pytest

from mtpy.lib import compute as compute_lib
from mtpy.pipelines.compute.AverageErrors import AverageErrors
from mtpy.pipelines.compute.Catalogues import Catalogues
from mtpy.pipelines.compute.Compute import Compute
from mtpy.pipelines.compute.Ratings import Ratings
from tests.test_compute_unit import FakeComputer


@pytest.fixture(autouse=True)
def marks(monkeypatch):
    """`compute_scope` sin base: el reloj y la marca de `computed_at` sustituidos (anota las marcas)."""
    seen = []
    monkeypatch.setattr(compute_lib, 'db_now', lambda: 'T0')
    monkeypatch.setattr(compute_lib, 'mark_computed', lambda scope, at: seen.append((scope, at)))
    return seen


def series(factory, scope='es', pipeline=Compute, position=(1, 2), **kwargs):
    """Por defecto, la rutina diaria: una elección pendiente sin resultado (`light`)."""
    p = pipeline()
    p.run('update', chunk=scope, position=position, computer=factory, featured=lambda scope: 7,
          **{'verbose': 0, 'pending': lambda scope: ['2026-11-29'], 'results': lambda scope: ['2023-07-23'], **kwargs})
    return p


def test_compute_skips_a_scope_with_nothing_pending(fresh_app):
    factory = FakeComputer()
    p = series(factory, pending=lambda scope: [])
    assert p.result == {'status': 'skipped', 'action': 'series', 'reason': 'nothing loaded since the last compute',
                        'warnings': []}
    assert p.line == 'es: series skipped (nothing loaded since the last compute)' and factory.inits == []


def test_compute_overwrite_computes_anyway_and_says_so_in_the_header(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, pending=lambda scope: [], overwrite=True)
    assert p.result['status'] == 'computed' and factory.inits == [('es', 0)] * 3
    assert p.line == 'es: series computed (full, overwrite: 2 polls, 2 events, 7 featured) in 0 s'
    assert 'compute series: 2 ámbitos · overwrite' in capsys.readouterr().out


def test_ratings_never_skip(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, 'es-md', Ratings, pending=lambda scope: [], overwrite=True)
    assert p.result['status'] == 'computed' and p.result['action'] == 'ratings'
    header = capsys.readouterr().out
    assert 'compute ratings: 2 ámbitos' in header and 'overwrite' not in header


def test_a_failing_pending_events_is_a_failed_result(fresh_app):
    def broken(scope):
        raise RuntimeError('db down')

    factory = FakeComputer()
    p = series(factory, pending=broken)
    assert p.result == {'status': 'failed', 'reason': 'RuntimeError: db down', 'action': 'series', 'warnings': []}
    assert p.line == 'es: series failed (RuntimeError: db down)' and factory.inits == []


def test_the_real_pending_events_is_the_default(fresh_app, monkeypatch):
    asked = []
    monkeypatch.setattr(compute_lib, 'pending_events', lambda scope: asked.append(scope) or [])
    p = Compute()
    p.run('update', chunk='es-cl', position=(1, 1), computer=FakeComputer(), featured=lambda scope: 0, verbose=0)
    assert asked == ['es-cl'] and p.result['status'] == 'skipped'


def test_compute_results_pending_makes_it_full(fresh_app):
    factory = FakeComputer()
    p = series(factory, pending=lambda scope: ['2023-07-23', '2026-11-29'])  # todo pendiente: tras el ALTER
    assert p.result['mode'] == 'full' and p.result['mode_reason'] == 'results pending'
    assert factory.inits == [('es', 0)] * 3
    assert p.line == 'es: series computed (full, results pending: 2 polls, 2 events, 7 featured) in 0 s'


def test_compute_consults_the_results_only_when_the_mode_is_auto(fresh_app, capsys):
    asked = []

    def results(scope):
        asked.append(scope)
        return []

    series(FakeComputer(), results=results)
    series(FakeComputer(), results=results, overwrite=True)
    series(FakeComputer(), results=results, mode='full')
    series(FakeComputer(), results=results, pending=lambda scope: [])
    assert asked == ['es']  # sólo la primera: auto, sin overwrite y con pendientes


def test_a_forced_mode_sets_the_header_and_the_reason(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, mode='full', position=(1, 1))
    assert p.result['mode_reason'] == 'requested' and factory.inits == [('es', 0)] * 3
    assert p.line == 'es: series computed (full, requested: 2 polls, 2 events, 7 featured) in 0 s'
    assert 'compute series: 1 ámbito · full' in capsys.readouterr().out
    p = series(FakeComputer(), mode='light', position=(1, 1), pending=lambda scope: ['2023-07-23'], dry_run=True)
    assert p.line == 'es: series computed (light, requested: 2 polls, 2 events, dry run) in 0 s'
    assert 'compute series: 1 ámbito · light · dry run' in capsys.readouterr().out


def test_a_forced_mode_does_not_avoid_the_skip(fresh_app):
    factory = FakeComputer()
    p = series(factory, mode='full', pending=lambda scope: [])
    assert p.result['status'] == 'skipped' and factory.inits == []


def test_an_unknown_mode_is_a_failed_result(fresh_app):
    p = series(FakeComputer(), mode='x')
    assert p.result['status'] == 'failed' and p.result['reason'] == 'ValueError: compute: unknown mode "x" (expected auto, light or full)'


def test_a_failing_result_dates_is_a_failed_result(fresh_app):
    def broken(scope):
        raise RuntimeError('db down')

    factory = FakeComputer()
    p = series(factory, results=broken)
    assert p.result == {'status': 'failed', 'reason': 'RuntimeError: db down', 'action': 'series', 'warnings': []}
    assert factory.inits == []


def test_the_real_result_dates_is_the_default(fresh_app, monkeypatch):
    asked = []
    monkeypatch.setattr(compute_lib, 'result_dates', lambda scope: asked.append(scope) or ['2023-07-23'])
    p = Compute()
    p.run('update', chunk='es-cl', position=(1, 1), computer=FakeComputer(), featured=lambda scope: 0, verbose=0,
          pending=lambda scope: ['2027-05-23'])
    assert asked == ['es-cl'] and p.result['mode'] == 'light'


def figures(save):
    full = {'scopes': 18, 'districts': 118, 'quality': 72, 'pollsters': 135}
    return dict(full if save else {'quality': 72}, seconds={'total': 0.0})


def test_compute_series_line_and_result(fresh_app, capsys):
    p = series(FakeComputer())
    assert p.result['status'] == 'computed' and p.result['action'] == 'series' and p.result['warnings'] == []
    assert p.result['mode'] == 'light' and p.result['mode_reason'] == 'polls pending' and p.result['featured'] is None
    assert p.line == 'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s'
    assert 'compute series: 2 ámbitos' in capsys.readouterr().out


def test_ratings_line_and_result(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, 'es-md', Ratings)
    assert p.result['action'] == 'ratings' and p.result['n_pollsters'] == 2
    assert p.line == 'es-md: ratings computed (2 pollsters, 1 events) in 0 s'
    assert factory.inits == [('es-md', 0)] and 'compute ratings: 2 ámbitos' in capsys.readouterr().out


def test_average_errors_line_result_and_header(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, pipeline=AverageErrors, pending=lambda scope: [])  # nunca salta
    assert p.result['status'] == 'computed' and p.result['action'] == 'errores del promedio'
    assert p.result['n_rows'] == 2 and p.result['n_events'] == 1 and 'mode' not in p.result
    assert p.line == 'es: average errors computed (2 rows, 1 events) in 0 s'
    assert factory.calls == [('build_series',), ('average_errors', True)]
    header = capsys.readouterr().out
    assert 'compute errores del promedio: 2 ámbitos' in header and 'overwrite' not in header
    p = series(FakeComputer(), pipeline=AverageErrors, position=(1, 1), dry_run=True, overwrite=True)
    assert p.line == 'es: average errors computed (2 rows, 1 events, dry run) in 0 s'
    assert 'compute errores del promedio: 1 ámbito · dry run' in capsys.readouterr().out


def test_average_errors_failure_names_the_action(fresh_app):
    p = series(FakeComputer(fail_at='average_errors'), pipeline=AverageErrors)
    assert p.line == 'es: errores del promedio failed (RuntimeError: boom average_errors)'


def test_compute_failure_is_a_failed_result_with_its_reason(fresh_app):
    p = series(FakeComputer(fail_at='herding', fail_scope='es'))
    assert p.result == {'status': 'failed', 'action': 'series', 'reason': 'RuntimeError: boom herding',
                        'warnings': []}
    assert p.line == 'es: series failed (RuntimeError: boom herding)'


def test_compute_dry_run_saves_nothing_and_says_so(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, position=(1, 1), dry_run=True)
    assert all(call[1] is False for call in factory.calls if call[0] != 'build_series')
    assert p.result['status'] == 'computed' and p.result['featured'] is None
    assert p.line == 'es: series computed (light, polls pending: 2 polls, 2 events, dry run) in 0 s'
    assert 'compute series: 1 ámbito · dry run' in capsys.readouterr().out


def test_compute_verbosity_levels(fresh_app, capsys):
    series(FakeComputer(), position=(1, 1), verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '[1/1] es: series computed' in out and 'pesos (' not in out
    series(FakeComputer(), position=(1, 1), verbose=2)
    out = capsys.readouterr().out
    assert '[1/1] es: pesos (' in out and '[1/1] es: rebaño (' in out and 'errores (' not in out  # light
    series(FakeComputer(), position=(1, 1), verbose=2, overwrite=True)
    out = capsys.readouterr().out
    for label in ('pesos', 'eventos destacados', 'errores', 'desviaciones', 'deriva', 'efectos de casa', 'rebaño'):
        assert '[1/1] es: {} ('.format(label) in out, label
    assert 'errores del promedio' not in out
    series(FakeComputer(), pipeline=Ratings, position=(1, 1), verbose=2)
    assert '[1/1] es: ratings (' in capsys.readouterr().out
    series(FakeComputer(), pipeline=AverageErrors, position=(1, 1), verbose=2)
    assert '[1/1] es: errores del promedio (' in capsys.readouterr().out
    factory = FakeComputer()
    series(factory, pipeline=Ratings, position=(1, 1), verbose=3)
    assert factory.inits == [('es', 1)]  # el Computer recibe verbose - 2


def test_compute_reports_the_model_warnings_as_notes(fresh_app, capsys):
    class WarningComputer(FakeComputer):
        def __call__(self, scope, verbose=0, **kwargs):
            comp = super().__call__(scope, verbose, **kwargs)
            weights = comp.compute_weights

            def compute_weights(save=False, overwrite=False):
                warnings.warn('serie corta', RuntimeWarning)
                return weights(save=save, overwrite=overwrite)
            comp.compute_weights = compute_weights
            return comp

    p = series(WarningComputer(), position=(1, 1), verbose=1)
    out, err = capsys.readouterr()
    assert '[1/1] es · aviso: serie corta' in out and 'serie corta' not in err
    assert p.result['warnings'] == ['serie corta']
    assert p.line == 'es: series computed (light, polls pending: 2 polls, 2 events) in 0 s · 1 aviso'


def test_catalogues_runs_once_and_dry_run_only_checks(fresh_app, capsys):
    calls = []

    def catalogues(save):
        calls.append(save)
        return figures(save)

    p = Catalogues()
    p.run('update', chunk=None, position=(1, 1), catalogues=catalogues, verbose=0)
    assert p.result['status'] == 'computed' and p.result['warnings'] == [] and calls == [True]
    assert p.line == 'catalogues: saved (18 scopes, 118 districts, 72 quality evaluations, 135 pollsters) in 0 s'
    assert p.failure_name(None) == 'catalogues (catálogos)' and Catalogues.chunked is False
    assert 'compute catálogos\n' in capsys.readouterr().out
    p = Catalogues()
    p.run('update', chunk=None, position=(1, 1), catalogues=catalogues, dry_run=True, verbose=0)
    assert calls == [True, False]
    assert p.line == 'catalogues: checked (72 quality evaluations, dry run) in 0 s'
    assert 'compute catálogos · dry run' in capsys.readouterr().out


def test_catalogues_failure_result(fresh_app):
    def catalogues(save):
        raise ValueError('standards must be between 0 and 1')

    p = Catalogues()
    p.run('update', chunk=None, position=(1, 1), catalogues=catalogues, verbose=0)
    assert p.result == {'status': 'failed', 'action': 'catálogos', 'warnings': [],
                        'reason': 'ValueError: standards must be between 0 and 1'}
    assert p.line == 'catalogues: catálogos failed (ValueError: standards must be between 0 and 1)'
