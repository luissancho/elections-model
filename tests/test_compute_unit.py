"""Tests de `mtpy.lib.compute`: la secuencia de cómputo de un ámbito y sus ratings, con un `Computer` falso."""
import pandas as pd
import pytest

from mtpy.lib import compute


class FakeComputer:
    """Fábrica con la firma `Computer(scope=..., verbose=...)`: anota cada construcción y cada fase con su `save`
    (y sus argumentos), y lanza `RuntimeError` en la fase `fail_at` (de todos los ámbitos, o sólo de `fail_scope`)."""

    def __init__(self, fail_at=None, fail_scope=None):
        self.inits, self.calls, self.fail_at, self.fail_scope = [], [], fail_at, fail_scope

    def __call__(self, scope, verbose=0, **kwargs):
        self.inits.append((scope, verbose))
        factory = self

        class Comp:
            series = pd.DataFrame({'pollster': ['A', 'B', None]})
            event_dates = ['2019-11-10', '2023-07-23']

            def build_series(self):
                factory.calls.append(('build_series',))
                return self

            def _phase(self, name, save, **kwargs):
                if factory.fail_at == name and factory.fail_scope in (None, scope):
                    raise RuntimeError('boom ' + name)
                factory.calls.append((name, save) + tuple(sorted(kwargs.items())))
                return pd.DataFrame({'event_date': ['2023-07-23'] * 2, 'pollster_id': [1, 2]})

            def compute_weights(self, save=False, overwrite=False):
                return self._phase('weights', save, overwrite=overwrite)

            def compute_errors(self, save=False):
                return self._phase('errors', save)

            def compute_deviations(self, save=False):
                return self._phase('deviations', save)

            def compute_drift(self, save=False):
                return self._phase('drift', save)

            def compute_house_effects(self, save=False):
                return self._phase('house_effects', save)

            def compute_herding(self, save=False):
                return self._phase('herding', save)

            def compute_average_errors(self, save=False):
                return self._phase('average_errors', save)

            def compute_ratings(self, save=False, scopes=None):
                # Como el real: devuelve la tabla de sondeos con sus columnas de rating (índice de `keys`) y deja
                # la tabla de ratings en `self.ratings` (índice `event_date`/`pollster_id`), donde entra una casa
                # (2) rateada sólo con sondeos de otros ámbitos
                self._phase('ratings', save, scopes=scopes)
                self.ratings = pd.DataFrame({'rating': [50., 60.]}, index=pd.MultiIndex.from_tuples(
                    [('2023-07-23', 1), ('2023-07-23', 2)], names=['event_date', 'pollster_id']))
                return pd.DataFrame({'rating': [50., 50.], 'weight_rating': [1., 1.]}, index=pd.MultiIndex.from_tuples(
                    [('2023-07-23', '2023-07-01', 1, 0), ('2023-07-23', '2023-07-10', 1, 0)],
                    names=['event_date', 'date', 'pollster_id', 'sponsor_id']))

        return Comp()


def phases(factory):
    return [call[0] for call in factory.calls if call[0] != 'build_series']


def test_compute_scope_runs_the_sequence_of_the_loader_and_reports_each_phase():
    factory, featured, steps = FakeComputer(), [], []
    result = compute.compute_scope('es-md', save=True, verbose=2, progress=lambda step, seconds: steps.append(step),
                                   computer=factory, featured=lambda scope: featured.append(scope) or 7,
                                   mark=lambda scope, at: None, clock=lambda: 'T0')
    # pesos, destacados, pesos otra vez (el `featured` de los sondeos copia el del evento), errores, desviaciones;
    # y con una serie fresca (leen los errores guardados) deriva, efectos de casa y rebaño; los errores del promedio
    # son una acción aparte (`average_errors_scope`)
    assert phases(factory) == ['weights', 'weights', 'errors', 'deviations', 'drift', 'house_effects', 'herding']
    assert steps == ['weights', 'featured', 'weights', 'errors', 'deviations', 'drift', 'house_effects', 'herding']
    assert tuple(dict.fromkeys(steps)) == compute.COMPUTE_STEPS == compute.MODES['full']  # el orden declarado es el que corre
    assert featured == ['es-md'] and factory.inits == [('es-md', 2)] * 3
    assert all(call[1] is True for call in factory.calls if call[0] != 'build_series')
    assert [call for call in factory.calls if call[0] == 'weights'] == [('weights', True, ('overwrite', True))] * 2
    assert result['featured'] == 7 and result['n_polls'] == 2 and result['n_events'] == 2 and result['mode'] == 'full'
    assert set(result['seconds']) == {'weights', 'featured', 'errors', 'deviations', 'drift', 'house_effects',
                                      'herding', 'total'}
    assert result['seconds']['total'] >= 0


def test_compute_scope_dry_run_stops_after_the_deviations_and_saves_nothing():
    factory, featured, steps = FakeComputer(), [], []
    result = compute.compute_scope('es', save=False, progress=lambda step, seconds: steps.append(step),
                                   computer=factory, featured=lambda scope: featured.append(scope) or 1)
    assert phases(factory) == ['weights', 'weights', 'errors', 'deviations'] and featured == []
    assert steps == ['weights', 'weights', 'errors', 'deviations']
    assert all(call[1] is False for call in factory.calls if call[0] != 'build_series')
    assert result['featured'] is None and 'drift' not in result['seconds'] and result['mode'] == 'full'


def test_compute_scope_lets_a_failure_propagate():
    with pytest.raises(RuntimeError, match='boom errors'):
        compute.compute_scope('es', save=True, computer=FakeComputer(fail_at='errors'), featured=lambda scope: 0,
                              mark=lambda scope, at: None, clock=lambda: 'T0')


def test_compute_scope_marks_the_scope_with_the_instant_it_started():
    marks, clocks = [], []
    compute.compute_scope('es-md', save=True, computer=FakeComputer(), featured=lambda scope: 0,
                          mark=lambda scope, at: marks.append((scope, at)), clock=lambda: clocks.append('T0') or 'T0')
    assert clocks == ['T0'] and marks == [('es-md', 'T0')]


def test_compute_scope_dry_run_and_failure_do_not_mark():
    marks = []
    compute.compute_scope('es', save=False, computer=FakeComputer(), featured=lambda scope: 0,
                          mark=lambda scope, at: marks.append(scope), clock=lambda: marks.append('clock'))
    with pytest.raises(RuntimeError, match='boom herding'):
        compute.compute_scope('es', save=True, computer=FakeComputer(fail_at='herding'), featured=lambda scope: 0,
                              mark=lambda scope, at: marks.append(scope), clock=lambda: 'T0')
    assert marks == []


def test_rate_scope_rates_with_every_scope_and_counts_the_table():
    factory, steps = FakeComputer(), []
    result = compute.rate_scope('es', save=True, verbose=1, progress=lambda step, seconds: steps.append(step),
                                computer=factory)
    assert factory.calls == [('build_series',), ('ratings', True, ('scopes', 'all'))] and factory.inits == [('es', 1)]
    assert steps == ['ratings'] and set(result['seconds']) == {'ratings', 'total'}
    assert result['n_pollsters'] == 2 and result['n_events'] == 1


def test_the_real_computer_and_featured_updater_are_the_defaults(monkeypatch):
    factory, featured, marks = FakeComputer(), [], []
    monkeypatch.setattr(compute, 'Computer', factory)
    monkeypatch.setattr(compute, 'update_featured', lambda scope: featured.append(scope) or 3)
    monkeypatch.setattr(compute, 'db_now', lambda: 'T0')
    monkeypatch.setattr(compute, 'mark_computed', lambda scope, at: marks.append((scope, at)))
    assert compute.compute_scope('es-cl', save=True)['featured'] == 3 and featured == ['es-cl']
    assert marks == [('es-cl', 'T0')]
    assert compute.rate_scope('es-cl', save=False)['n_events'] == 1
    assert marks == [('es-cl', 'T0')]
    assert factory.inits[0] == ('es-cl', 0)


def test_replicate_catalogues_saves_or_only_checks_the_rubric(monkeypatch):
    """`catalogues`: con escritura, `save_catalogues()` y sus cifras; en seco, sólo valida la rúbrica y el CSV de
    la calidad previa (`read_quality`) y cuenta las evaluaciones, sin tocar la base."""
    saved = []
    monkeypatch.setattr(compute, 'save_catalogues', lambda: saved.append(1) or {'scopes': 18, 'districts': 118,
                                                                                'quality': 72, 'pollsters': 135})
    monkeypatch.setattr(compute, 'load_criteria', lambda: {'scale': {}})
    monkeypatch.setattr(compute, 'read_quality', lambda criteria: pd.DataFrame({'pollster_id': [1, 2, 3]}))
    out = compute.replicate_catalogues(save=True)
    assert saved == [1] and out['scopes'] == 18 and out['quality'] == 72 and out['pollsters'] == 135
    assert set(out['seconds']) == {'total'}
    out = compute.replicate_catalogues(save=False)
    assert saved == [1] and out['quality'] == 3 and 'scopes' not in out


def test_compute_scope_light_runs_weights_and_herding_only():
    factory, featured, steps, marks = FakeComputer(), [], [], []
    result = compute.compute_scope('es', save=True, verbose=1, mode='light',
                                   progress=lambda step, seconds: steps.append(step), computer=factory,
                                   featured=lambda scope: featured.append(scope) or 7,
                                   mark=lambda scope, at: marks.append((scope, at)), clock=lambda: 'T0')
    # una pasada de pesos sobre el ámbito entero (copia a cada sondeo el destacado de su elección) y, con una serie
    # fresca que lee esos pesos, el rebaño; sin destacados: exigen resultado oficial y en auto el light sólo sale sin él
    assert phases(factory) == ['weights', 'herding'] and steps == ['weights', 'herding'] and featured == []
    assert [call for call in factory.calls if call[0] == 'weights'] == [('weights', True, ('overwrite', True))]
    assert factory.inits == [('es', 1)] * 2 and marks == [('es', 'T0')]
    assert result['mode'] == 'light' and result['featured'] is None and result['n_polls'] == 2
    assert set(result['seconds']) == {'weights', 'herding', 'total'}
    assert tuple(dict.fromkeys(steps)) == compute.MODES['light']


def test_compute_scope_light_dry_run_only_weighs_and_does_not_mark():
    factory, steps, marks = FakeComputer(), [], []
    result = compute.compute_scope('es-md', save=False, mode='light', progress=lambda step, seconds: steps.append(step),
                                   computer=factory, featured=lambda scope: 0,
                                   mark=lambda scope, at: marks.append(scope), clock=lambda: marks.append('clock'))
    assert phases(factory) == ['weights'] and steps == ['weights'] and marks == []
    assert factory.calls[1] == ('weights', False, ('overwrite', True)) and result['mode'] == 'light'


def test_compute_scope_rejects_an_unknown_mode_before_building_anything():
    factory = FakeComputer()
    with pytest.raises(ValueError, match='compute: unknown mode "auto" \\(expected light or full\\)'):
        compute.compute_scope('es', save=True, mode='auto', computer=factory, featured=lambda scope: 0,
                              mark=lambda scope, at: None, clock=lambda: 'T0')
    assert factory.inits == []


def test_average_errors_scope_fits_the_table_once_and_never_marks(monkeypatch):
    marks = []
    monkeypatch.setattr(compute, 'mark_computed', lambda scope, at: marks.append(scope))
    monkeypatch.setattr(compute, 'db_now', lambda: marks.append('clock'))
    factory, steps = FakeComputer(), []
    result = compute.average_errors_scope('es', save=True, verbose=2, progress=lambda step, seconds: steps.append(step),
                                          computer=factory)
    assert factory.calls == [('build_series',), ('average_errors', True)] and factory.inits == [('es', 2)]
    assert steps == ['average_errors'] and set(result['seconds']) == {'average_errors', 'total'}
    assert result['n_rows'] == 2 and result['n_events'] == 1 and marks == []
    result = compute.average_errors_scope('es', save=False, computer=factory)
    assert factory.calls[-1] == ('average_errors', False) and result['n_rows'] == 2  # en seco calcula sin guardar


def test_resolve_mode_applies_the_auto_rule():
    assert compute.resolve_mode('light', True, None, None) == ('light', 'requested')
    assert compute.resolve_mode('full', False, [], []) == ('full', 'requested')
    assert compute.resolve_mode('auto', True, None, None) == ('full', 'overwrite')
    assert compute.resolve_mode('auto', False, ['2023-07-23', '2026-11-29'], ['2019-11-10', '2023-07-23']) == ('full', 'results pending')
    assert compute.resolve_mode('auto', False, ['2026-11-29'], ['2019-11-10', '2023-07-23']) == ('light', 'polls pending')
    assert compute.resolve_mode('auto', False, [], ['2023-07-23']) == ('light', 'polls pending')
    with pytest.raises(ValueError, match='compute: unknown mode "x" \\(expected auto, light or full\\)'):
        compute.resolve_mode('x', False, [], [])


def test_check_mode_accepts_the_three_modes_and_nothing_else():
    assert [compute.check_mode(m) for m in ('auto', 'light', 'full')] == ['auto', 'light', 'full']
    with pytest.raises(ValueError, match='compute: unknown mode "Light" \\(expected auto, light or full\\)'):
        compute.check_mode('Light')
