"""Tests del pipeline `tune.Tune` ejecutado suelto, con `tuning.tuning_scope` y la base sustituidos."""
import warnings

import pytest

from mtpy.lib import tuning
from mtpy.pipelines.tune.Tune import Tune


class FakeScope:
    """
    `tuning.tuning_scope` sin modelo ni base: registra sus argumentos y devuelve las cifras.

    `es-xx` no tiene ciclos (`NothingToTune`), `missing` es el ámbito sin `best.json` (`FileNotFoundError`),
    `fail_scope` no puede puntuarse (`TuningFailed`) y `warn` avisa de un corte omitido.
    """

    def __init__(self, fail_scope=None, missing=None, warn=False):
        self.calls = []
        self.fail_scope = fail_scope
        self.missing = missing
        self.warn = warn

    def __call__(self, scope, **kwargs):
        self.calls.append({'scope': scope, **kwargs})
        if scope == 'es-xx':
            raise tuning.NothingToTune('no cycles to tune for es-xx')
        if scope == self.missing:
            raise FileNotFoundError('no best.json in {}: run the previous phase or pass start'.format(scope))
        if scope == self.fail_scope:
            raise tuning.TuningFailed('start start of the smoothing search on {}: 1 of 2 cuts failed'.format(scope))
        if self.warn:
            warnings.warn('cut 2019-10-01 of 2019-11-10 skipped: no future polls')
        figures = {'n_configs': len(kwargs.get('configs') or [None]), 'n_cuts': 2,
                   'cycles': {'search': ['2019-11-10'], 'check': []},
                   'folder': '{}/{}'.format(kwargs.get('out'), scope), 'seconds': {'run': 0., 'write': 0., 'total': 0.}}
        if kwargs.get('mode') != 'evaluate':
            figures['best'] = {**tuning.default_config(), 'label': 'bw=(\'fixed\', 45)'}
        return figures


@pytest.fixture
def patched(monkeypatch):
    """`search_cycles` sin base: `es-md` tiene dos ciclos de búsqueda y uno de comprobación."""
    monkeypatch.setattr(tuning, 'search_cycles',
                        lambda scope: {'search': ['2011-05-22', '2015-05-24'], 'check': ['2023-05-28']})


def stats(scope):
    return {'db_polls': 7, 'db_last_poll': '2026-10-03'}


def evaluate(runner=None, scope='es-md', tmp_path=None, position=(2, 2), **kwargs):
    p = Tune()
    p.run('run', chunk=scope, position=position, out=str(tmp_path), runner=runner or FakeScope(), stats=stats,
          **{'mode': 'evaluate', 'configs': [tuning.default_config()], 'verbose': 0, **kwargs})
    return p


def test_tune_run_writes_the_folder_and_builds_its_line(fresh_app, patched, tmp_path, capsys):
    p = evaluate(tmp_path=tmp_path)
    assert p.result['status'] == 'completed' and p.result['n_configs'] == 1
    assert p.result['folder'] == '{}/es-md'.format(tmp_path) and p.result['warnings'] == []
    assert p.line == 'es-md: tune completed (1 configs, 2 cuts) in 0 s'
    out = capsys.readouterr().out
    assert 'tune: 2 ámbitos · evaluate · {}'.format(tmp_path) in out
    assert 'empezando' not in out


def test_tune_skips_a_scope_without_cycles(fresh_app, patched, tmp_path):
    p = evaluate(tmp_path=tmp_path, scope='es-xx')
    assert p.result['status'] == 'skipped' and 'no cycles' in p.result['reason']
    assert p.line == 'es-xx: skipped (no cycles to tune for es-xx)'


def test_tune_skips_a_scope_without_best_json(fresh_app, patched, tmp_path):
    p = evaluate(FakeScope(missing='es-md'), tmp_path=tmp_path, mode='check', configs=None)
    assert p.result['status'] == 'skipped' and 'no best.json' in p.result['reason']


def test_tune_reports_a_failed_tuning_as_failed_not_skipped(fresh_app, patched, tmp_path):
    p = evaluate(FakeScope(fail_scope='es-md'), tmp_path=tmp_path, mode='smoothing', configs=None)
    assert p.result['status'] == 'failed' and p.result['reason'].startswith('TuningFailed: start start')
    assert p.line.startswith('es-md: failed (TuningFailed: ')


def test_tune_passes_every_parameter_through(fresh_app, patched, tmp_path):
    runner = FakeScope()
    grid = {'bw': [('fixed', 45), ('fixed', 60)]}
    evaluate(runner, tmp_path=tmp_path, mode='smoothing', configs=None, grid=grid, cut_days=14, horizons=(7, 28, 56),
             rounds=1, verbose=2)
    call = runner.calls[0]
    assert call == {'scope': 'es-md', 'out': str(tmp_path), 'mode': 'smoothing', 'configs': None, 'grid': grid,
                    'cycles': None, 'cut_days': 14, 'horizons': (7, 28, 56), 'rounds': 1, 'verbose': 1,
                    'stats': stats, 'workers': 1, 'part': None}


def test_tune_passes_the_workers_and_names_a_shrinkage_part_after_its_cycles(fresh_app, patched, tmp_path):
    runner = FakeScope()
    evaluate(runner, tmp_path=tmp_path, mode='shrinkage', configs=None, cycles=['2015-05-24', '2011-05-22'],
             workers=8)
    evaluate(runner, tmp_path=tmp_path, mode='smoothing', configs=None, cycles=['2015-05-24'])
    evaluate(runner, tmp_path=tmp_path, mode='shrinkage', configs=None)
    assert [(call['workers'], call['part']) for call in runner.calls] == [
        (8, '2011-05-22_2015-05-24'), (1, None), (1, None)]


def seeded(tmp_path, scope):
    """A repository copy of the tuning tables of `scope` under `tmp_path/local`."""
    local = tmp_path / 'local'
    folder = local if scope == 'es' else local / scope
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('scores.csv', 'trials-smoothing.csv', 'best.json'):
        (folder / name).write_text(name)
    return local


def test_tune_push_and_pull_copy_the_tables_between_the_repository_and_the_app(fresh_app, patched, tmp_path,
                                                                             capsys):
    from mtpy.core.io import FileSystem
    fresh_app.set('fs', FileSystem(str(tmp_path / 'remote')))
    local = seeded(tmp_path, 'es-md')
    p = Tune()
    p.run('push', chunk='es-md', position=(1, 1), out=str(local), verbose=0)
    assert p.result['status'] == 'pushed' and p.result['n_files'] == 3 and p.result['n_removed'] == 0
    assert (tmp_path / 'remote' / 'backtest' / 'tuning' / 'es-md' / 'best.json').exists()
    assert p.line.startswith('es-md: tune pushed (3 files) to ')
    assert 'tune push: 1 ámbito · {} → '.format(local) in capsys.readouterr().out
    p = Tune()
    p.run('pull', chunk='es-md', position=(1, 1), out=str(tmp_path / 'copy'), verbose=0)
    assert p.result['status'] == 'pulled' and (tmp_path / 'copy' / 'es-md' / 'trials-smoothing.csv').exists()
    p = Tune()
    p.run('pull', chunk='es-ar', position=(1, 1), out=str(tmp_path / 'copy'), verbose=0)
    assert p.result['status'] == 'skipped' and 'no tuning tables' in p.result['reason']


def test_tune_restricts_the_cycles_of_the_scope_and_warns_of_the_unknown_ones(fresh_app, patched, tmp_path):
    runner = FakeScope()
    p = evaluate(runner, tmp_path=tmp_path, cycles=['2015-05-24', '2023-05-28', '2019-05-26'])
    assert runner.calls[0]['cycles'] == {'search': ['2015-05-24'], 'check': ['2023-05-28']}
    assert p.result['warnings'] == ['2019-05-26 is not a cycle of es-md (ignored)']
    assert p.line == 'es-md: tune completed (1 configs, 2 cuts) in 0 s · 1 aviso'


def test_tune_reports_the_warnings_of_the_scope(fresh_app, patched, tmp_path, capsys):
    p = evaluate(FakeScope(warn=True), scope='es', tmp_path=tmp_path, position=(1, 1), verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out
    assert '[1/1] es · aviso: cut 2019-10-01 of 2019-11-10 skipped: no future polls' in out
    assert p.line == 'es: tune completed (1 configs, 2 cuts) in 0 s · 1 aviso'
