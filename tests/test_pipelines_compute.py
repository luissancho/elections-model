"""Tests de los pipelines `compute.Compute`, `compute.Ratings` y `compute.Catalogues` ejecutados sueltos, sin base."""
import warnings

from mtpy.pipelines.compute.Catalogues import Catalogues
from mtpy.pipelines.compute.Compute import Compute
from mtpy.pipelines.compute.Ratings import Ratings
from tests.test_compute_unit import FakeComputer


def series(factory, scope='es', pipeline=Compute, position=(1, 2), **kwargs):
    p = pipeline()
    p.run('update', chunk=scope, position=position, computer=factory, featured=lambda scope: 7,
          **{'verbose': 0, **kwargs})
    return p


def figures(save):
    full = {'scopes': 18, 'districts': 118, 'quality': 72, 'pollsters': 135}
    return dict(full if save else {'quality': 72}, seconds={'total': 0.0})


def test_compute_series_line_and_result(fresh_app, capsys):
    p = series(FakeComputer())
    assert p.result['status'] == 'computed' and p.result['action'] == 'series' and p.result['warnings'] == []
    assert p.line == 'es: series computed (2 polls, 2 events, 7 featured) in 0 s'
    assert 'compute series: 2 ámbitos' in capsys.readouterr().out


def test_ratings_line_and_result(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, 'es-md', Ratings)
    assert p.result['action'] == 'ratings' and p.result['n_pollsters'] == 2
    assert p.line == 'es-md: ratings computed (2 pollsters, 1 events) in 0 s'
    assert factory.inits == [('es-md', 0)] and 'compute ratings: 2 ámbitos' in capsys.readouterr().out


def test_compute_failure_is_a_failed_result_with_its_reason(fresh_app):
    p = series(FakeComputer(fail_at='errors', fail_scope='es'))
    assert p.result == {'status': 'failed', 'action': 'series', 'reason': 'RuntimeError: boom errors',
                        'warnings': []}
    assert p.line == 'es: series failed (RuntimeError: boom errors)'


def test_compute_dry_run_saves_nothing_and_says_so(fresh_app, capsys):
    factory = FakeComputer()
    p = series(factory, position=(1, 1), dry_run=True)
    assert all(call[1] is False for call in factory.calls if call[0] != 'build_series')
    assert p.result['status'] == 'computed' and p.result['featured'] is None
    assert p.line == 'es: series computed (2 polls, 2 events, dry run) in 0 s'
    assert 'compute series: 1 ámbito · dry run' in capsys.readouterr().out


def test_compute_verbosity_levels(fresh_app, capsys):
    series(FakeComputer(), position=(1, 1), verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '[1/1] es: series computed' in out and 'pesos (' not in out
    factory = FakeComputer()
    series(factory, position=(1, 1), verbose=2)
    out = capsys.readouterr().out
    for label in ('pesos', 'eventos destacados', 'errores', 'desviaciones', 'deriva', 'efectos de casa', 'rebaño'):
        assert '[1/1] es: {} ('.format(label) in out, label
    series(FakeComputer(), pipeline=Ratings, position=(1, 1), verbose=2)
    assert '[1/1] es: ratings (' in capsys.readouterr().out
    factory = FakeComputer()
    series(factory, pipeline=Ratings, position=(1, 1), verbose=3)
    assert factory.inits == [('es', 1)]  # el Computer recibe verbose - 2


def test_compute_reports_the_model_warnings_as_notes(fresh_app, capsys):
    class WarningComputer(FakeComputer):
        def __call__(self, scope, verbose=0, **kwargs):
            comp = super().__call__(scope, verbose, **kwargs)
            errors = comp.compute_errors

            def compute_errors(save=False):
                warnings.warn('serie corta', RuntimeWarning)
                return errors(save=save)
            comp.compute_errors = compute_errors
            return comp

    p = series(WarningComputer(), position=(1, 1), verbose=1)
    out, err = capsys.readouterr()
    assert '[1/1] es · aviso: serie corta' in out and 'serie corta' not in err
    assert p.result['warnings'] == ['serie corta']
    assert p.line == 'es: series computed (2 polls, 2 events, 7 featured) in 0 s · 1 aviso'


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
