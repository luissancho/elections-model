"""Tests del job `tune` (`mtpy/jobs/Tune.py`) y de `backtest/run_tuning.py`, con `tuning.tuning_scope`, la base y
el catálogo de ámbitos sustituidos."""
import importlib
import importlib.util
import os

import pandas as pd
import pytest

from mtpy.lib import tuning
from tests.test_pipelines_tune import FakeScope, patched as pipeline_patched, stats  # noqa: F401

module = importlib.import_module('mtpy.jobs.Tune')
Tune = module.Tune

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'backtest', 'run_tuning.py')


def load_script():
    """El módulo de `backtest/run_tuning.py`, sin ejecutar `main`."""
    spec = importlib.util.spec_from_file_location('run_tuning', SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def patched(monkeypatch, pipeline_patched):
    """`resolve_scopes` sin base (`'all'` son `es` y `es-md`) además de `search_cycles`."""
    monkeypatch.setattr(module, 'resolve_scopes', lambda scopes: ['es', 'es-md'] if scopes == 'all' else list(scopes))


def test_tune_job_runs_every_scope_and_parses_configs(fresh_app, patched, tmp_path, capsys):
    runner = FakeScope()
    results = Tune().run(scopes='all', out=str(tmp_path), mode='evaluate', configs=[{'bw': ['fixed', 45]}, 'default'],
                         verbose=0, runner=runner)
    assert [c['scope'] for c in runner.calls] == ['es', 'es-md']
    assert [c['label'] for c in runner.calls[0]['configs']] == ["bw=('fixed', 45)", 'default']
    assert runner.calls[0]['configs'][0] == tuning.apply_setting(tuning.default_config(), 'bw', ('fixed', 45))
    assert runner.calls[0]['configs'][1] == tuning.default_config()
    assert results['es']['status'] == 'completed' and results['es-md']['n_configs'] == 2
    out = capsys.readouterr().out
    assert 'tune: 2 ámbitos · evaluate · {}'.format(tmp_path) in out
    assert 'es: tune completed (2 configs, 2 cuts) in 0 s' in out and 'es-md: tune completed' in out


def test_tune_job_applies_the_settings_of_a_config_in_order(fresh_app, patched, tmp_path):
    runner = FakeScope()
    Tune().run(scopes=['es'], out=str(tmp_path), mode='evaluate', verbose=0, runner=runner,
               configs=[{'bw': ['adaptive', 2], 'bw_cap': 1.5, 'wspan': 7},
                        {'house_effects': False, 'dispersion': False, 'herding': False}])
    first, second = runner.calls[0]['configs']
    expected = tuning.apply_setting(tuning.apply_setting(tuning.apply_setting(
        tuning.default_config(), 'bw', ('adaptive', 2)), 'bw_cap', 1.5), 'wspan', 7)
    assert first == {**expected, 'label': "bw=('adaptive', 2), bw_cap=1.5, wspan=7"}
    assert second == {**tuning.default_config(), 'house_effects': False, 'dispersion': False, 'herding': False,
                      'label': 'house_effects=False, dispersion=False, herding=False'}


def test_tune_job_normalises_horizons_cycles_and_grid(fresh_app, patched, tmp_path):
    runner = FakeScope()
    Tune().run(scopes=['es-md'], out=str(tmp_path), mode='smoothing', horizons=[7, 28, 56], cycles=['2015-05-24'],
               grid={'bw': [['fixed', 45], ['adaptive', 3]], 'wspan': [4, 7]}, cut_days=14, rounds=1, verbose=0,
               runner=runner)
    call = runner.calls[0]
    assert call['horizons'] == (7, 28, 56) and call['cycles'] == {'search': ['2015-05-24'], 'check': []}
    assert call['grid'] == {'bw': [('fixed', 45), ('adaptive', 3)], 'wspan': [4, 7]}
    assert call['cut_days'] == 14 and call['rounds'] == 1 and call['configs'] is None and call['mode'] == 'smoothing'


@pytest.mark.parametrize('kwargs, message', [
    ({'mode': 'search'}, 'tune: mode must be smoothing, shrinkage, check, evaluate or merge'),
    ({'mode': 'evaluate'}, 'tune: the evaluate mode needs configs'),
    ({'mode': 'smoothing', 'configs': ['default']}, 'tune: configs only apply to the evaluate mode'),
    ({'mode': 'evaluate', 'configs': ['best']}, 'tune: a config must be "default" or'),
    ({'mode': 'evaluate', 'configs': [{}]}, 'tune: a config must be "default" or'),
    ({'mode': 'evaluate', 'configs': [{'bw': ['fixed', 50]}]}, 'tune: bad config'),
    ({'mode': 'evaluate', 'configs': [{'house_effects': 'no'}]}, 'tune: bad config'),
    ({'mode': 'evaluate', 'configs': ['default'], 'grid': {'bw': [['fixed', 45]]}}, 'tune: grid only applies'),
    ({'grid': {'bw': [['fixed', 50]]}}, 'tune: grid'),
    ({'grid': {'house_effects': [True]}}, 'tune: grid'),
    ({'horizons': [7, 'x']}, 'tune: horizons'),
    ({'horizons': []}, 'tune: horizons'),
    ({'cycles': '2019-11-10'}, 'tune: cycles'),
    ({'cycles': ['2019-13-10']}, 'tune: cycles'),
    ({'cut_days': 0}, 'tune: cut_days'),
    ({'cut_days': True}, 'tune: cut_days'),
    ({'rounds': 1.5}, 'tune: rounds'),
    ({'action': 'sync'}, 'tune: unknown action "sync"'),
    ({'workers': 0}, 'tune: workers'),
    ({'workers': True}, 'tune: workers'),
    ({'workers': 2.5}, 'tune: workers'),
    ({'workers': None}, 'tune: workers'),
    ({'mode': 'merge', 'configs': ['default']}, 'tune: configs only apply to the evaluate mode'),
    ({'mode': 'check', 'grid': {'tau': [0.04]}}, 'tune: grid only applies'),
    ({'scopes': []}, 'tune: no scopes to run'),
])
def test_tune_job_rejects_bad_arguments_before_running(fresh_app, patched, tmp_path, kwargs, message):
    runner = FakeScope()
    with pytest.raises(ValueError, match='^' + message.replace('(', '\\(').replace(')', '\\)')):
        Tune().run(**{'scopes': ['es'], 'out': str(tmp_path), 'verbose': 0, 'runner': runner, **kwargs})
    assert runner.calls == []


def test_tune_job_skips_and_isolates_scopes_and_fails_at_the_end(fresh_app, patched, tmp_path, capsys):
    runner = FakeScope(fail_scope='es-ar', missing='es-cl')
    with pytest.raises(RuntimeError, match='tune: failed scopes: es-ar'):
        Tune().run(scopes=['es-xx', 'es-cl', 'es-ar', 'es'], out=str(tmp_path), mode='shrinkage', verbose=0,
                   runner=runner)
    assert [c['scope'] for c in runner.calls] == ['es-xx', 'es-cl', 'es-ar', 'es']
    out = capsys.readouterr().out
    assert 'es-xx: skipped (no cycles to tune for es-xx)' in out and 'es-cl: skipped (no best.json in es-cl' in out
    assert 'es-ar: failed (TuningFailed: ' in out and 'es: tune completed (1 configs, 2 cuts) in 0 s' in out


def test_tune_job_stop_on_error_stops_at_the_first_failure(fresh_app, patched, tmp_path):
    runner = FakeScope(fail_scope='es-ar')
    with pytest.raises(RuntimeError, match='tune: failed scopes: es-ar'):
        Tune().run(scopes=['es-ar', 'es'], out=str(tmp_path), verbose=0, runner=runner, stop_on_error=True)
    assert [c['scope'] for c in runner.calls] == ['es-ar']


def test_tune_job_passes_the_workers_and_runs_the_merge(fresh_app, patched, tmp_path):
    runner = FakeScope()
    Tune().run(scopes=['es'], out=str(tmp_path), mode='merge', grid={'tau': [0.04, 0.08]}, workers=8, verbose=0,
               runner=runner)
    call = runner.calls[0]
    assert call['mode'] == 'merge' and call['workers'] == 8 and call['grid'] == {'tau': [0.04, 0.08]}


def test_tune_job_pushes_and_pulls_the_tables(fresh_app, patched, tmp_path):
    from mtpy.core.io import FileSystem
    from tests.test_pipelines_tune import seeded
    fresh_app.set('fs', FileSystem(str(tmp_path / 'remote')))
    local = seeded(tmp_path, 'es')
    results = Tune().run(scopes=['es'], action='push', out=str(local), verbose=0)
    assert results['es']['status'] == 'pushed' and (tmp_path / 'remote' / 'backtest' / 'tuning' / 'best.json').exists()
    results = Tune().run(scopes=['es'], action='pull', out=str(tmp_path / 'copy'), verbose=0)
    assert results['es']['status'] == 'pulled' and (tmp_path / 'copy' / 'scores.csv').exists()


@pytest.mark.parametrize('text, expected', [
    ('fixed,45', ('fixed', 45)), ('adaptive,3', ('adaptive', 3)), ('fixed,isj', ('fixed', 'isj')), ('1.5', 1.5),
    ('7', 7), ('0', 0), ('null', None), ('None', None), ('false', False), ('true', True)
])
def test_run_tuning_parses_a_value(text, expected):
    value = load_script().parse_value(text)
    assert value == expected and type(value) is type(expected)


def test_run_tuning_parses_the_configs():
    script = load_script()
    assert script.parse_configs(None) is None
    assert script.parse_configs([['default'], ['bw=fixed,45', 'bw_cap=1.5'], ['max_fc=0']]) == [
        'default', {'bw': ('fixed', 45), 'bw_cap': 1.5}, {'max_fc': 0}]
    with pytest.raises(ValueError, match='--config expects key=value'):
        script.parse_configs([['bw']])
    with pytest.raises(ValueError, match='"default" goes alone'):
        script.parse_configs([['default', 'wspan=7']])


def test_run_tuning_turns_its_options_into_the_job_arguments():
    script = load_script()
    args = script.build_parser().parse_args(['--scopes', 'es', 'es-md', '--config', 'default', '--config',
                                             'bw=fixed,45', '--cycles', '2019-11-10', '--cut-days', '28',
                                             '--horizons', '7', '28', '--rounds', '1'])
    assert script.job_arguments(args) == {
        'scopes': ['es', 'es-md'], 'mode': 'evaluate', 'configs': ['default', {'bw': ('fixed', 45)}],
        'cycles': ['2019-11-10'], 'cut_days': 28, 'horizons': (7, 28), 'rounds': 1, 'out': tuning.TUNING_ROOT,
        'verbose': 1, 'workers': 1}
    args = script.build_parser().parse_args([])
    assert script.job_arguments(args) == {
        'scopes': ['es'], 'mode': 'smoothing', 'configs': None, 'cycles': None, 'cut_days': None,
        'horizons': (7, 28), 'rounds': None, 'out': tuning.TUNING_ROOT, 'verbose': 1, 'workers': 1}
    args = script.build_parser().parse_args(['--mode', 'merge', '--workers', '8'])
    assert script.job_arguments(args)['mode'] == 'merge' and script.job_arguments(args)['workers'] == 8


def test_run_tuning_reports_the_completed_scopes_when_another_one_fails(fresh_app, tmp_path, capsys):
    script = load_script()

    class FailingJob:
        """The `tune` job after a run where `es-md` completed and `es-ar` failed."""
        def run(self, **kwargs):
            self.results = {('tune', 'es-md'): {'status': 'completed', 'folder': str(tmp_path / 'es-md')},
                            ('tune', 'es-ar'): {'status': 'failed', 'reason': 'TuningFailed: start'}}
            raise RuntimeError('tune: failed scopes: es-ar')

    with pytest.raises(RuntimeError, match='es-ar'):
        script.run_job(FailingJob(), {'scopes': ['es-md', 'es-ar']}, str(tmp_path))
    assert 'es-md: tables written to {}'.format(tmp_path / 'es-md') in capsys.readouterr().out


def test_run_tuning_reports_the_best_config_and_the_check_scores(fresh_app, tmp_path, capsys):
    script = load_script()
    rows = [{'config': config, 'partition': 'check', 'event_date': event, 'k': k, 'rmse': rmse, 'roughness': 0.1}
            for config, base in (('default', 2.), ('best', 1.)) for event in ('2023-07-23', '2026-11-29')
            for k, rmse in ((7, base), (28, base + 1))]
    os.makedirs(tmp_path / 'es-md')
    pd.DataFrame(rows).to_csv(tmp_path / 'es-md' / 'scores.csv', index=False)
    best = {**tuning.apply_setting(tuning.default_config(), 'bw', ('fixed', 45)), 'label': 'best'}
    script.report({'es-md': {'status': 'completed', 'folder': str(tmp_path / 'es-md'), 'best': best},
                   'es-xx': {'status': 'skipped', 'reason': 'no cycle to tune'}}, str(tmp_path))
    out = capsys.readouterr().out
    assert 'es-md: tables written to {}'.format(tmp_path / 'es-md') in out
    assert 'es-md: best = best' in out and '"bw": 45' in out
    lines = [line.split() for line in out.splitlines() if line.strip().startswith(('default', 'best '))]
    assert lines == [['default', '2', '2.0', '3.0', '0.1'], ['best', '2', '1.0', '2.0', '0.1']]
    assert 'es-xx' not in out


def test_check_scores_lists_the_default_and_the_tuned_configuration_under_its_settings_label():
    script = load_script()
    label = 'bw_cap=1.0; max_fc=0; he_n_iter=1; herd_prior=2.0'
    rows = [{'config': config, 'partition': 'check', 'event_date': event, 'k': k, 'rmse': base + k / 100,
             'roughness': 0.01}
            for config, base in ((label, 1.0), ('default', 1.2)) for event in ('2023-07-23', '2026-11-29')
            for k in (7, 28)]
    rows.append({'config': 'default', 'partition': 'search', 'event_date': '2019-11-10', 'k': 7, 'rmse': 9.,
                 'roughness': 0.01})
    table = script.check_scores(pd.DataFrame(rows))
    assert table['config'].tolist() == ['default', label]
    assert table['n_cycles'].tolist() == [2, 2] and table['rmse7'].round(2).tolist() == [1.27, 1.07]
