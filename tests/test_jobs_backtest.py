"""Tests del job `backtest` (`mtpy/jobs/Backtest.py`) con el backtest, la base y el catálogo de ámbitos sustituidos."""
import importlib
import json

import pytest

from tests.test_pipelines_backtest import FakeRunner, patched as pipeline_patched, stats  # noqa: F401

module = importlib.import_module('mtpy.jobs.Backtest')
Backtest = module.Backtest


@pytest.fixture
def patched(monkeypatch, pipeline_patched):
    """`resolve_scopes` sin base (`'all'` son `es` y `es-md`) además de `default_events` y la procedencia."""
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))


def test_backtest_runs_every_scope_and_writes_its_folder(fresh_app, patched, tmp_path, capsys):
    runner = FakeRunner()
    results = Backtest().run(scopes='all', out=str(tmp_path), n_sim=20, verbose=0, runner=runner, stats=stats)
    assert [call['scope'] for call in runner.calls] == ['es', 'es-md']
    assert all(call['n_sim'] == 20 and call['verbose'] == 0 for call in runner.calls)
    assert results['es'] == {'status': 'completed', 'n_cases': 4, 'n_events': 2, 'n_horizons': 2,
                             'events': ['2019-11-10', '2023-07-23'], 'horizons': [6, 30], 'folder': str(tmp_path),
                             'seconds': results['es']['seconds'], 'warnings': []}
    assert results['es-md']['status'] == 'completed' and results['es-md']['folder'] == str(tmp_path / 'es-md')
    assert (tmp_path / 'metrics.csv').exists() and (tmp_path / 'es-md' / 'run.json').exists()
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['scope'] == 'es-md' and run['n_sim'] == 20 and run['commit'] == 'abc1234' and run['db_polls'] == 7
    out = capsys.readouterr().out
    assert 'backtest: 2 ámbitos · 20 simulaciones · {}'.format(tmp_path) in out
    assert 'es: backtest completed (4 cases, 2 events, 2 horizons) in 0 s' in out
    assert 'es-md: backtest completed (4 cases, 2 events, 2 horizons) in 0 s' in out
    assert 'empezando' not in out and 'mae_shares' not in out  # nivel 0: cabecera y resumen, sin la tabla


def test_backtest_skips_and_isolates_scopes_and_fails_at_the_end(fresh_app, patched, tmp_path, capsys):
    runner = FakeRunner(fail_scope='es-ar')
    with pytest.raises(RuntimeError, match='backtest: failed scopes: es-ar'):
        Backtest().run(scopes=['es-xx', 'es-ar', 'es'], out=str(tmp_path), verbose=0, runner=runner, stats=stats)
    assert [call['scope'] for call in runner.calls] == ['es-ar', 'es']  # `es-xx` no llega al modelo
    out = capsys.readouterr().out
    assert 'es-xx: skipped (no elections to evaluate)' in out and "es-ar: failed (KeyError: 'x')" in out
    assert 'es: backtest completed (4 cases, 2 events, 2 horizons)' in out
    assert (tmp_path / 'run.json').exists() and not (tmp_path / 'es-xx').exists() and not (tmp_path / 'es-ar').exists()


@pytest.mark.parametrize('given, expected', [('auto', 'auto'), ('0.25', 0.25), (0.5, 0.5), (1, 1.0), (None, None)])
def test_backtest_normalises_composition(fresh_app, patched, tmp_path, given, expected):
    runner = FakeRunner()
    results = Backtest().run(scopes=['es'], out=str(tmp_path), verbose=0, composition=given, runner=runner, stats=stats)
    assert runner.calls[0]['composition'] == expected and results['es']['status'] == 'completed'
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['composition'] == expected


@pytest.mark.parametrize('bad', ['x', 2, 0, -0.5, True, [0.5]])
def test_backtest_rejects_a_bad_composition_before_running(fresh_app, patched, tmp_path, bad):
    runner = FakeRunner()
    with pytest.raises(ValueError, match='backtest: composition must be "auto", a ratio in \\(0, 1\\] or null'):
        Backtest().run(scopes=['es'], out=str(tmp_path), composition=bad, runner=runner, stats=stats)
    assert runner.calls == [] and not (tmp_path / 'run.json').exists()


def test_backtest_stop_on_error_stops_at_the_first_failure(fresh_app, patched, tmp_path):
    runner = FakeRunner(fail_scope='es-ar')
    with pytest.raises(RuntimeError, match='backtest: failed scopes: es-ar'):
        Backtest().run(scopes=['es-ar', 'es'], out=str(tmp_path), verbose=0, runner=runner, stats=stats,
                       stop_on_error=True)
    assert [call['scope'] for call in runner.calls] == ['es-ar']


def test_backtest_refuses_an_empty_scope_list(fresh_app, patched, tmp_path):
    with pytest.raises(ValueError, match='backtest: no scopes to run'):
        Backtest().run(scopes=[], out=str(tmp_path), verbose=0, runner=FakeRunner(), stats=stats)
