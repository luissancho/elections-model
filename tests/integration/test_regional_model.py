"""Tests de integración del Simulator en ámbitos autonómicos (M11)."""
import pytest

pytestmark = pytest.mark.integration


def regional_sim(scope, event_date, **kwargs):
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope=scope, event_date=event_date, drange=6, seed=42, verbose=0, path='.', **kwargs)
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    return sim


@pytest.mark.parametrize('scope, event_date', [('es-md', '2023-05-28'), ('es-cl', '2022-02-13')])
def test_deterministic_seats_are_close_to_the_official_ones(app, scope, event_date):
    from mtpy.lib.backtest import official_results
    sim = regional_sim(scope, event_date)
    sim.run(split=True, random=False)
    model = sim.result().loc[scope]
    official = official_results(scope, event_date)['seats']
    main = sim.event_params['bmaps']['main']
    assert int(model.sum()) == sim.n_seats
    assert (model[main] - official.reindex(main).fillna(0)).abs().mean() <= 1.5


def test_scope_settings_are_resolved(app):
    md = regional_sim('es-md', '2023-05-28')
    assert (md.threshold, md.threshold_scope) == (5.0, None)
    assert md.regions == [0, 28] and md.region_names[28] == 'Madrid'
    md.run(split=True, random=False)
    assert (md.frame()['regional'] == 0).all()
    cn = regional_sim('es-cn', '2023-05-28')
    assert (cn.threshold, cn.threshold_scope) == (15.0, 4.0) and 100 in cn.regions


def test_random_run_in_a_single_district_scope(app):
    md = regional_sim('es-md', '2023-05-28')
    md.run(split=True, random=True, n_sim=20)
    assert (md.dist().sum(axis=1) == 135).all()
    md.run(split=False, random=False)
    assert int(md.totals().sum()) == 135


def test_previous_event_without_polls_is_a_valid_base(app):
    # Foco 1: una elección previa sin sondeos cargados sirve de base del swing
    sim = regional_sim('es-as', '2023-05-28')
    assert sim.prev_date == '2019-05-26'
    sim.run(split=True, random=True, n_sim=10)
    assert (sim.dist().sum(axis=1) == 45).all()
