"""Tests del job `backtest` (`mtpy/jobs/Backtest.py`) con el backtest, la base y el catálogo de ámbitos sustituidos."""
import importlib
import json
import warnings

import pandas as pd
import pytest

from mtpy.lib import backtest as lib
from mtpy.lib.publish import common
from tests.test_backtest_unit import fake_results

module = importlib.import_module('mtpy.jobs.Backtest')
Backtest = module.Backtest


class FakeRunner:
    """`run_backtest` sin modelo: registra sus argumentos, rompe en `fail_scope` y avisa de un caso omitido con `warn`."""

    def __init__(self, fail_scope=None, warn=False):
        self.calls = []
        self.fail_scope = fail_scope
        self.warn = warn

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs['scope'] == self.fail_scope:
            raise KeyError('x')
        if self.warn:
            warnings.warn('Backtest 2023-05-28 at 180 days skipped: no usable polls')
        return fake_results()


@pytest.fixture
def patched(monkeypatch):
    """`resolve_scopes` sin base (`'all'` son `es` y `es-md`), `default_events` sin base (`es-xx` no tiene elecciones)
    y la procedencia sin `git`."""
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))
    monkeypatch.setattr(lib, 'default_events', lambda scope, since='2019-01-01': [] if scope == 'es-xx' else ['2023-05-28'])
    monkeypatch.setattr(common, 'provenance', lambda cwd=None: {'commit': 'abc1234', 'dirty': False, 'versions': {}})


def stats(scope):
    return {'db_polls': 7, 'db_last_poll': '2026-10-03'}


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


def test_backtest_skips_a_scope_whose_cases_all_fail(fresh_app, patched, tmp_path, capsys):
    empty = {**fake_results(), 'metrics': pd.DataFrame(), 'by_horizon': pd.DataFrame()}
    results = Backtest().run(scopes=['es-md'], out=str(tmp_path), verbose=0, runner=lambda **kwargs: empty, stats=stats)
    assert results['es-md'] == {'status': 'skipped', 'reason': 'no case could be evaluated', 'warnings': []}
    assert 'es-md: skipped (no case could be evaluated)' in capsys.readouterr().out and not (tmp_path / 'es-md').exists()


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


def test_backtest_verbosity_levels_and_the_horizon_table(fresh_app, patched, tmp_path, capsys):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), verbose=1, runner=runner, stats=stats)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '[1/1] es: backtest completed' in out
    assert 'horizon  n_cases' in out and 'mae_shares' in out  # la tabla `by_horizon` del ámbito
    assert runner.calls[-1]['verbose'] == 0
    Backtest().run(scopes=['es'], out=str(tmp_path), verbose=2, runner=runner, stats=stats)
    assert runner.calls[-1]['verbose'] == 1  # el backtest escribe la línea de cada caso


def test_backtest_reports_the_skipped_cases_as_notes(fresh_app, patched, tmp_path, capsys):
    """Nivel 1: los casos que el backtest omite (sin sondeos útiles) salen como `· aviso:` del ámbito, no crudos
    por stderr, y se cuentan en la línea de resumen y en el resultado, como en `publish` y `compute`."""
    results = Backtest().run(scopes=['es'], out=str(tmp_path), verbose=1, runner=FakeRunner(warn=True), stats=stats)
    out, err = capsys.readouterr()
    assert '[1/1] es · aviso: Backtest 2023-05-28 at 180 days skipped: no usable polls' in out and 'skipped' not in err
    assert results['es']['warnings'] == ['Backtest 2023-05-28 at 180 days skipped: no usable polls']
    assert 'es: backtest completed (4 cases, 2 events, 2 horizons) in 0 s · 1 aviso' in out


def test_backtest_passes_every_parameter_through_and_records_it(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es-md'], out=str(tmp_path), events=['2023-05-28'], horizons=[6, 30], n_sim=50, seed=7,
                   max_fc=5, nowcast_only=True, house_effects=False, industry_bias=True, dispersion=False, herding=False,
                   regional_noise=False, min_events=1, verbose=0, runner=runner, stats=stats)
    call = runner.calls[0]
    assert call == {'scope': 'es-md', 'events': ['2023-05-28'], 'horizons': [6, 30], 'verbose': 0, 'n_sim': 50, 'seed': 7,
                    'max_fc': 5, 'nowcast_only': True, 'house_effects': False, 'industry_bias': True, 'dispersion': False,
                    'herding': False, 'composition': None, 'regional_noise': False, 'min_events': 1}
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['seed'] == 7 and run['nowcast_only'] is True and run['house_effects'] is False and run['min_events'] == 1
    assert run['events'] == ['2019-11-10', '2023-07-23'] and run['horizons'] == [6, 30]  # lo evaluado, no lo pedido
