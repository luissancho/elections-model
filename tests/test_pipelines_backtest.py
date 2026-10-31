"""Tests del pipeline `backtest.Backtest` ejecutado suelto, con el modelo y la base sustituidos."""
import json
import warnings

import pandas as pd
import pytest

from mtpy.lib import backtest as lib
from mtpy.lib.publish import common
from mtpy.pipelines.backtest.Backtest import Backtest
from tests.test_backtest_unit import fake_results


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
    """`default_events` sin base (`es-xx` no tiene elecciones) y la procedencia sin `git`."""
    monkeypatch.setattr(lib, 'default_events', lambda scope, since='2019-01-01': [] if scope == 'es-xx' else ['2023-05-28'])
    monkeypatch.setattr(common, 'provenance', lambda cwd=None: {'commit': 'abc1234', 'dirty': False, 'versions': {}})


def stats(scope):
    return {'db_polls': 7, 'db_last_poll': '2026-10-03'}


def evaluate(runner=None, scope='es-md', tmp_path=None, position=(2, 2), **kwargs):
    p = Backtest()
    p.run('run', chunk=scope, position=position, out=str(tmp_path), runner=runner or FakeRunner(), stats=stats,
          **{'n_sim': 20, 'verbose': 0, **kwargs})
    return p


def test_backtest_run_writes_the_folder_and_builds_its_line(fresh_app, patched, tmp_path, capsys):
    p = evaluate(tmp_path=tmp_path)
    assert p.result == {'status': 'completed', 'n_cases': 4, 'n_events': 2, 'n_horizons': 2,
                        'events': ['2019-11-10', '2023-07-23'], 'horizons': [6, 30],
                        'folder': str(tmp_path / 'es-md'), 'seconds': p.result['seconds'], 'warnings': []}
    assert p.line == 'es-md: backtest completed (4 cases, 2 events, 2 horizons) in 0 s'
    assert (tmp_path / 'es-md' / 'run.json').exists()
    out = capsys.readouterr().out
    assert 'backtest: 2 ámbitos · 20 simulaciones · {}'.format(tmp_path) in out
    assert 'empezando' not in out and 'mae_shares' not in out


def test_backtest_skips_a_scope_without_elections_or_cases(fresh_app, patched, tmp_path):
    p = evaluate(scope='es-xx', tmp_path=tmp_path)
    assert p.result == {'status': 'skipped', 'reason': 'no elections to evaluate', 'warnings': []}
    assert p.line == 'es-xx: skipped (no elections to evaluate)'
    empty = {**fake_results(), 'metrics': pd.DataFrame(), 'by_horizon': pd.DataFrame()}
    p = evaluate(lambda **kwargs: empty, tmp_path=tmp_path)
    assert p.result == {'status': 'skipped', 'reason': 'no case could be evaluated', 'warnings': []}
    assert not (tmp_path / 'es-md').exists()


def test_backtest_failure_result_and_line(fresh_app, patched, tmp_path):
    p = evaluate(FakeRunner(fail_scope='es-ar'), scope='es-ar', tmp_path=tmp_path)
    assert p.result == {'status': 'failed', 'reason': "KeyError: 'x'", 'warnings': []}
    assert p.line == "es-ar: failed (KeyError: 'x')"
    assert p.failure_name('es-ar') == 'es-ar' and not (tmp_path / 'es-ar').exists()


def test_backtest_verbosity_levels_and_the_horizon_table(fresh_app, patched, tmp_path, capsys):
    runner = FakeRunner()
    evaluate(runner, scope='es', tmp_path=tmp_path, position=(1, 1), verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '[1/1] es: backtest completed' in out
    assert out.index('backtest completed') < out.index('horizon  n_cases') and 'mae_shares' in out
    assert runner.calls[-1]['verbose'] == 0
    evaluate(runner, scope='es', tmp_path=tmp_path, position=(1, 1), verbose=2)
    assert runner.calls[-1]['verbose'] == 1


def test_backtest_reports_the_skipped_cases_as_notes(fresh_app, patched, tmp_path, capsys):
    p = evaluate(FakeRunner(warn=True), scope='es', tmp_path=tmp_path, position=(1, 1), verbose=1)
    out, err = capsys.readouterr()
    assert '[1/1] es · aviso: Backtest 2023-05-28 at 180 days skipped: no usable polls' in out and 'skipped' not in err
    assert p.result['warnings'] == ['Backtest 2023-05-28 at 180 days skipped: no usable polls']
    assert p.line == 'es: backtest completed (4 cases, 2 events, 2 horizons) in 0 s · 1 aviso'


def test_backtest_passes_every_parameter_through_and_records_it(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    evaluate(runner, tmp_path=tmp_path, events=['2023-05-28'], horizons=[6, 30], n_sim=50, seed=7, max_fc=5,
             nowcast_only=True, house_effects=False, industry_bias=True, dispersion=False, herding=False,
             regional_noise=False, min_events=1)
    assert runner.calls[0] == {'scope': 'es-md', 'events': ['2023-05-28'], 'horizons': [6, 30], 'verbose': 0,
                               'n_sim': 50, 'seed': 7, 'max_fc': 5, 'nowcast_only': True, 'house_effects': False,
                               'industry_bias': True, 'dispersion': False, 'herding': False, 'composition': None,
                               'regional_noise': False, 'min_events': 1, 'weight_ratings': None}
    run = json.load(open(tmp_path / 'es-md' / 'run.json', encoding='utf-8'))
    assert run['seed'] == 7 and run['nowcast_only'] is True and run['house_effects'] is False and run['min_events'] == 1
    assert run['recompute_ratings'] is False
    assert run['events'] == ['2019-11-10', '2023-07-23'] and run['horizons'] == [6, 30]


def seeded(tmp_path, scope='es-md'):
    """Una carpeta local con los nueve ficheros de `scope` (por `backtest_scope` con el runner falso)."""
    from mtpy.lib import backtest
    from tests.test_backtest_unit import fake_results
    local = tmp_path / 'local'
    backtest.backtest_scope(scope, out=str(local), events=['2023-05-28'], runner=lambda **kwargs: fake_results(),
                            stats=stats, prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}})
    return local


def test_push_copies_the_nine_files_and_removes_the_stale_ones(fresh_app, tmp_path, capsys):
    from mtpy.core.io import FileSystem
    remote = tmp_path / 'remote'
    fresh_app.set('fs', FileSystem(str(remote)))
    local = seeded(tmp_path)
    (local / 'es-md' / 'series.csv').unlink()
    stale = remote / 'backtest' / 'results' / 'es-md'
    stale.mkdir(parents=True)
    (stale / 'series.csv').write_text('old')
    p = Backtest()
    p.run('push', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    assert p.result['status'] == 'pushed' and p.result['n_files'] == 8 and p.result['n_removed'] == 1
    assert p.line == 'es-md: backtest pushed (8 files, 1 removed) to {}'.format(stale)
    assert sorted(f.name for f in stale.iterdir()) == ['blocks.csv', 'by_horizon.csv', 'meta.csv', 'metrics.csv', 'provinces.csv', 'run.json', 'seats.csv', 'shares.csv']
    assert 'backtest push: 1 ámbito · {} → {}'.format(local, remote / 'backtest' / 'results') in capsys.readouterr().out


def test_pull_brings_the_files_back_to_the_local_copy(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    remote = tmp_path / 'remote'
    fresh_app.set('fs', FileSystem(str(remote)))
    local = seeded(tmp_path)
    Backtest().run('push', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    (local / 'es-md' / 'metrics.csv').write_text('changed')
    p = Backtest()
    p.run('pull', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    assert p.result['status'] == 'pulled' and p.result['n_files'] == 9 and p.result['n_removed'] == 0
    assert p.line == 'es-md: backtest pulled (9 files) to {}'.format(local / 'es-md')
    assert (local / 'es-md' / 'metrics.csv').read_text() != 'changed'


def test_push_without_local_results_is_skipped_and_touches_nothing(fresh_app, tmp_path):
    from mtpy.core.io import FileSystem
    fresh_app.set('fs', FileSystem(str(tmp_path / 'remote')))
    p = Backtest()
    p.run('push', chunk='es-xx', position=(1, 1), out=str(tmp_path / 'local'), verbose=0)
    assert p.result['status'] == 'skipped' and p.line == 'es-xx: skipped (no backtest results for es-xx in {})'.format(tmp_path / 'local' / 'es-xx')
    assert not (tmp_path / 'remote').exists()
