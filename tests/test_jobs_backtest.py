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


def test_backtest_job_passes_error_model_to_the_runner(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), error_model='average', verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['error_model'] == 'average'
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['error_model'] == 'average'


def test_backtest_job_defaults_to_the_polls_error_model(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['error_model'] == 'polls'
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['error_model'] == 'polls'


@pytest.mark.parametrize('bad', ['house', None, 1, ['average']])
def test_backtest_rejects_a_bad_error_model_before_running(fresh_app, patched, tmp_path, bad):
    runner = FakeRunner()
    with pytest.raises(ValueError, match='backtest: error_model must be "polls" or "average"'):
        Backtest().run(scopes=['es'], out=str(tmp_path), error_model=bad, runner=runner, stats=stats)
    assert runner.calls == [] and not (tmp_path / 'run.json').exists()


def test_backtest_job_passes_drift_scale_to_the_runner(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), drift_scale='auto', verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['drift_scale'] == 'auto'
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['drift_scale'] == 'auto'


@pytest.mark.parametrize('value, expected', [(None, None), ('auto', 'auto'), (1.5, 1.5), ('2', 2.), (3, 3.)])
def test_backtest_job_normalises_the_drift_scale(fresh_app, patched, tmp_path, value, expected):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), drift_scale=value, verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['drift_scale'] == expected
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['drift_scale'] == expected


@pytest.mark.parametrize('bad', ['house', 0, -1., True, float('nan'), ['auto']])
def test_backtest_rejects_a_bad_drift_scale_before_running(fresh_app, patched, tmp_path, bad):
    runner = FakeRunner()
    with pytest.raises(ValueError, match='backtest: drift_scale must be "auto", a positive number or null'):
        Backtest().run(scopes=['es'], out=str(tmp_path), drift_scale=bad, runner=runner, stats=stats)
    assert runner.calls == [] and not (tmp_path / 'run.json').exists()


@pytest.mark.parametrize('value, expected', [(None, None), (5, 5), ('8', 8), (2, 2)])
def test_backtest_job_passes_the_tail_dof_to_the_runner(fresh_app, patched, tmp_path, value, expected):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), tail_dof=value, verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['tail_dof'] == expected
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['tail_dof'] == expected


@pytest.mark.parametrize('bad', [0, 1, -3, 2.5, True, 'x', [5]])
def test_backtest_rejects_a_bad_tail_dof_before_running(fresh_app, patched, tmp_path, bad):
    runner = FakeRunner()
    with pytest.raises(ValueError, match='backtest: tail_dof must be an integer of at least 2 or null'):
        Backtest().run(scopes=['es'], out=str(tmp_path), tail_dof=bad, runner=runner, stats=stats)
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


def test_backtest_action_must_be_run_push_or_pull(fresh_app, patched, tmp_path):
    with pytest.raises(ValueError, match='backtest: unknown action "sync"'):
        Backtest().run(scopes=['es'], action='sync', out=str(tmp_path))


def test_backtest_pull_through_the_job(fresh_app, patched, tmp_path):
    from mtpy.core.io import FileSystem
    from tests.test_pipelines_backtest import seeded
    fresh_app.set('fs', FileSystem(str(tmp_path / 'remote')))
    local = seeded(tmp_path, 'es')
    Backtest().run(scopes=['es'], action='push', out=str(local), verbose=0)
    results = Backtest().run(scopes=['es'], action='pull', out=str(tmp_path / 'copy'), verbose=0)
    assert results['es']['status'] == 'pulled' and (tmp_path / 'copy' / 'run.json').exists()


def tuned_config(**changes):
    """Una configuración afinada como la que guarda la búsqueda en `best.json` (`default_config` con ajustes)."""
    config = {'reg_params': {'bw_type': 'fixed', 'bw': 45, 'poly_deg': 1}, 'house_effects': True,
              'he_params': {'tau': 0.16}, 'dispersion': True, 'herding': True, 'disp_params': {'herd_prior': 10.},
              'weights': {'wspan': 7}, 'max_fc': 3, 'label': 'best'}
    return {**config, **changes}


@pytest.mark.parametrize('as_path', [True, False])
def test_backtest_job_passes_the_config_to_the_runner(fresh_app, patched, tmp_path, as_path):
    """El job pasa `config` (un dict o la ruta de un JSON) al ejecutor, con los interruptores y `max_fc` de la
    configuración, y `run.json` la registra (etiqueta y ajustes)."""
    config = tuned_config(herding=False, dispersion=False)
    path = tmp_path / 'best.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    runner = FakeRunner()
    out = tmp_path / 'out'
    Backtest().run(scopes=['es'], out=str(out), config=str(path) if as_path else config, verbose=0, runner=runner,
                   stats=stats)
    call = runner.calls[0]
    assert call['config'] == config
    assert (call['house_effects'], call['dispersion'], call['herding'], call['max_fc']) == (True, False, False, 3)
    run = json.load(open(out / 'run.json', encoding='utf-8'))
    assert run['config'] == config and run['config']['label'] == 'best'
    assert run['max_fc'] == 3 and run['herding'] is False


def test_backtest_job_runs_without_a_config_by_default(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['config'] is None and runner.calls[0]['max_fc'] == 10
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['config'] is None


def test_backtest_job_accepts_flags_that_agree_with_the_config(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), config=tuned_config(herding=False), herding=False, max_fc=3,
                   verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['herding'] is False and runner.calls[0]['max_fc'] == 3


@pytest.mark.parametrize('flag, value', [('house_effects', False), ('dispersion', False), ('herding', False),
                                         ('max_fc', 0)])
def test_backtest_rejects_flags_that_contradict_the_config(fresh_app, patched, tmp_path, flag, value):
    runner = FakeRunner()
    with pytest.raises(ValueError, match='backtest: {} contradicts the config'.format(flag)):
        Backtest().run(scopes=['es'], out=str(tmp_path), config=tuned_config(), runner=runner, stats=stats,
                       **{flag: value})
    assert runner.calls == [] and not (tmp_path / 'run.json').exists()


@pytest.mark.parametrize('bad', [
    'missing.json', 5, ['best'], {k: v for k, v in tuned_config().items() if k != 'label'},
    tuned_config(extra=1), tuned_config(reg_params=None), tuned_config(he_params=[]), tuned_config(disp_params='x'),
    tuned_config(house_effects=1), tuned_config(dispersion='yes'), tuned_config(herding=None),
    tuned_config(max_fc=-1), tuned_config(max_fc=True), tuned_config(max_fc=2.5), tuned_config(label=3),
    tuned_config(weights={}), tuned_config(weights={'wspan': 0}), tuned_config(weights={'wspan': '7'}),
    tuned_config(weights={'wspan': 7, 'ol_max': 0}), tuned_config(weights={'wspan': 7, 'other': 1}),
    tuned_config(weights=7),
    # claves internas mal escritas: correrían el valor vivo con la etiqueta afinada
    tuned_config(reg_params={'bw_typ': 'fixed', 'bw': 45}),
    tuned_config(reg_params={'bw_type': 'adaptive', 'bw_kwargs': {'n_iters': 2}}),
    tuned_config(reg_params={'bw_type': 'adaptive', 'bw_kwargs': 2}),
    tuned_config(he_params={'tauu': 0.16}),
    tuned_config(disp_params={'herd_priors': 10.}),
    tuned_config(disp_params={'herding': False})   # el interruptor va arriba, no en disp_params
])
def test_backtest_rejects_a_bad_config_before_running(fresh_app, patched, tmp_path, bad):
    runner = FakeRunner()
    bad = str(tmp_path / bad) if bad == 'missing.json' else bad
    with pytest.raises(ValueError, match='backtest: config'):
        Backtest().run(scopes=['es'], out=str(tmp_path), config=bad, runner=runner, stats=stats)
    assert runner.calls == [] and not (tmp_path / 'run.json').exists()


def test_backtest_rejects_a_config_file_that_is_not_json(fresh_app, patched, tmp_path):
    path = tmp_path / 'best.json'
    path.write_text('{not json', encoding='utf-8')
    with pytest.raises(ValueError, match='backtest: config'):
        Backtest().run(scopes=['es'], out=str(tmp_path), config=str(path), runner=FakeRunner(), stats=stats)


def test_backtest_config_inner_keys_are_those_of_the_forecaster_setters():
    """Las claves admitidas en `reg_params`, `he_params` y `disp_params` son las que leen los `set_*_params`."""
    from mtpy.lib.forecaster import Forecaster
    fc = Forecaster.__new__(Forecaster)
    assert set(module.CONFIG_PARAMS['reg_params']) == set(fc.set_reg_params())
    assert set(module.CONFIG_PARAMS['he_params']) == set(fc.set_he_params())
    assert set(module.CONFIG_PARAMS['disp_params']) == set(fc.set_disp_params()) - {'herding'}
    assert set(module.BW_KWARGS) == set(fc.set_reg_params()['bw_kwargs'])


def test_backtest_config_accepts_the_tuned_best_and_the_default():
    """`backtest/tuning/best.json` y `tuning.default_config()` pasan la validación sin cambios."""
    import os
    from mtpy.lib.tuning import default_config
    path = os.path.join(os.path.dirname(__file__), '..', 'backtest', 'tuning', 'best.json')
    assert module.Backtest._config(path) == json.load(open(path, encoding='utf-8'))
    assert module.Backtest._config(default_config()) == default_config()


def test_job_backtest_rejects_flat_with_recompute_ratings(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    with pytest.raises(ValueError, match='backtest: flat_ratings and recompute_ratings are exclusive'):
        Backtest().run(scopes=['es'], out=str(tmp_path), flat_ratings=True, recompute_ratings=True, runner=runner,
                       stats=stats)
    assert runner.calls == [] and not (tmp_path / 'run.json').exists()


def test_job_backtest_passes_flat_ratings(fresh_app, patched, tmp_path):
    runner = FakeRunner()
    Backtest().run(scopes=['es'], out=str(tmp_path), flat_ratings=True, verbose=0, runner=runner, stats=stats)
    assert runner.calls[0]['weight_ratings'] == 'flat'
    assert json.load(open(tmp_path / 'run.json', encoding='utf-8'))['flat_ratings'] is True
