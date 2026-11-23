"""Tests del reescalado de las circunscripciones por el censo (sin base de datos)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.utils import census_factors, census_shares, census_totals


def _census():
    return pd.DataFrame({
        'date': ['2023-07-23'] * 2 + ['2026-11-29'] * 2,
        'region_id': [1, 2, 1, 2],
        'residents': [100, 200, 110, 190],
        'abroad': [10, 10, 50, 50]
    })


def test_census_factors_is_the_change_of_the_residents():
    factors = census_factors(_census(), '2026-11-29', '2023-07-23')
    assert factors.to_dict() == {1: 1.1, 2: 0.95}


def test_census_factors_without_census_or_election_is_none():
    assert census_factors(None, '2026-11-29', '2023-07-23') is None
    assert census_factors(_census(), '2026-11-29', '2019-11-10') is None
    assert census_factors(_census(), '2027-01-01', '2023-07-23') is None


def test_census_totals_scales_the_districts_and_the_total():
    totals = pd.DataFrame({'votes': [300., 100., 200.], 'blank_pct': [1.33, 2., 1.]}, index=[0, 1, 2])
    out = census_totals(totals, pd.Series({1: 1.1, 2: 0.95}), [1, 2], 0)
    assert out.loc[1, 'votes'] == pytest.approx(110.)
    assert out.loc[2, 'votes'] == pytest.approx(190.)
    assert out.loc[0, 'votes'] == pytest.approx(300.)
    assert out.loc[0, 'blank_pct'] == pytest.approx(round((2. * 110 + 1. * 190) / 300, 2))
    assert totals.loc[1, 'votes'] == 100.  # La tabla de entrada no se toca


def test_census_totals_needs_every_district():
    totals = pd.DataFrame({'votes': [300., 100., 200.], 'blank_pct': [1., 1., 1.]}, index=[0, 1, 2])
    with pytest.raises(ValueError):
        census_totals(totals, pd.Series({1: 1.1}), [1, 2], 0)


def test_census_shares_recomputes_the_total_from_the_districts():
    prev = pd.DataFrame({'A': [40., 60., 30.], 'B': [20., 0., 30.]}, index=[0, 1, 2])
    out = census_shares(prev, pd.Series({0: 300., 1: 110., 2: 190.}), [1, 2], 0)
    assert out.loc[0, 'A'] == pytest.approx((60 * 110 + 30 * 190) / 300)
    assert out.loc[0, 'B'] == pytest.approx(30 * 190 / 300)
    assert np.allclose(out.loc[[1, 2]], prev.loc[[1, 2]])
    assert prev.loc[0, 'A'] == 40.


def test_census_shares_with_the_same_votes_gives_back_the_total():
    prev = pd.DataFrame({'A': [40., 60., 30.]}, index=[0, 1, 2])
    out = census_shares(prev, pd.Series({0: 300., 1: 100., 2: 200.}), [1, 2], 0)
    assert out.loc[0, 'A'] == pytest.approx(40.)


# --- En el simulador (`Simulator.apply_census`, `build_umat`) -------------------------------------------------

def _simulator(monkeypatch, census):
    """Simulator mínimo (patrón `__new__`): dos provincias iguales, un partido al 20 % y al 50 % (35 % nacional)."""
    from mtpy.lib import simulator
    from mtpy.lib.simulator import Simulator

    monkeypatch.setattr(simulator, 'get_census', lambda scope: census)
    sim = Simulator.__new__(Simulator)
    sim.scope, sim.event_date, sim.prev_date = 'es', '2026-11-29', '2023-07-23'
    sim.swing, sim.default_region, sim.verbose, sim.smap = 'logistic', 0, 0, {}
    sim.regional_noise, sim.v2swing, sim.rng_swing, sim.rng_geo = False, None, None, None
    sim.geo_cache, sim.geo_summary = {}, {}
    sim.cols_unit = ['prev_pct', 'vpred_pct']
    regions = pd.Index([0, 1, 2], name='region_id')
    sim.regions = [0, 1, 2]
    sim.params = {'names': ['A'], 'regions': [0, 1, 2], 'random': False}
    sim.prev_results = pd.concat({'pct': pd.DataFrame({'A': [35., 20., 50.]}, index=regions)}, axis=1)
    sim.prev_totals = pd.DataFrame({'votes': [200., 100., 100.], 'blank_pct': [0., 0., 0.]}, index=regions)
    sim.census_applied, sim.census_pcts = False, None
    return sim


def test_apply_census_reweighs_the_districts_and_keeps_the_national_forecast(monkeypatch):
    """El censo de la provincia 2 se duplica: pesa 200 frente a 100 y el resultado anterior «con el censo de hoy»
    es el 40 %, no el 35 %. Con el partido al 50 % nacional, las provincias ponderadas siguen dando el 50 %."""
    census = pd.DataFrame({'date': ['2023-07-23'] * 2 + ['2026-11-29'] * 2, 'region_id': [1, 2, 1, 2],
                           'residents': [100, 100, 100, 200], 'abroad': [0, 0, 0, 0]})
    sim = _simulator(monkeypatch, census)
    assert sim.apply_census() is True and sim.census_applied is True
    assert sim.prev_totals['votes'].tolist() == pytest.approx([300., 100., 200.])
    assert sim.census_pcts.loc[0, 'A'] == pytest.approx(40.)
    assert sim.prev_results['pct'].loc[0, 'A'] == 35.  # El resultado oficial no se toca

    umat = sim.build_umat(pd.DataFrame({'vpred': [50., 50.]}, index=['A', '-']))['vpred_pct']
    assert (umat.loc[1, 'A'] * 100 + umat.loc[2, 'A'] * 200) / 300 == pytest.approx(50., abs=0.01)
    assert umat.loc[1, 'A'] < umat.loc[2, 'A']


def test_apply_census_without_the_census_of_the_scope_changes_nothing(monkeypatch):
    sim = _simulator(monkeypatch, None)
    assert sim.apply_census() is False and sim.census_applied is False
    assert sim.prev_totals['votes'].tolist() == [200., 100., 100.] and sim.census_pcts is None


def test_census_file_of_es_covers_every_province_of_the_elections_it_has():
    """`data/census/es.csv`: 52 provincias por elección, de 1986 a la del 29-N, con el total del INE para esta."""
    import os

    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'census', 'es.csv')
    census = pd.read_csv(path)
    assert (census.groupby('date')['region_id'].nunique() == 52).all()
    assert set(census['region_id']) == set(range(1, 53))
    last = census.loc[census['date'] == '2026-11-29']
    assert int(last['residents'].sum()) == 35958711 and int(last['abroad'].sum()) == 2769645
    prev = census.loc[census['date'] == '2023-07-23']
    assert int(prev['residents'].sum() + prev['abroad'].sum()) == 37469458
