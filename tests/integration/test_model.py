"""Tests de integración: necesitan la base de datos local y files/params.json."""
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def sim27(app):
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2027-08-22', drange=6, seed=42, verbose=0, path='.')
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    return sim


def test_run_with_names_subset_does_not_raise(sim27):
    """B3: `names` con un subconjunto de partidos era inutilizable en eventos con smap."""
    sim27.run(split=True, random=False, names=['PP', 'PSOE', 'VOX'])
    assert int(sim27.totals().sum()) == 350


def test_mt_simulations_sum_to_350(sim27):
    sim27.run(split=True, random=True, n_sim=20)
    assert (sim27.dist().sum(axis=1) == 350).all()


def test_mt_is_reproducible_with_seed(sim27):
    sim27.run(split=True, random=True, n_sim=5)
    first = sim27.results.copy()
    sim27.run(split=True, random=True, n_sim=5)
    assert np.array_equal(first, sim27.results)


def test_result_is_indexed_by_region_name(sim27):
    sim27.run(split=True, random=False)
    res = sim27.result()
    assert 'es' in res.index and 'Madrid' in res.index
    assert int(res.loc['es'].sum()) == 350
    assert sim27.unit(0, region='Madrid').shape[0] == len(sim27.params['names'])


def test_smap_regions_rule_2019_04(app):
    """B4/B5: las reglas `regions` del smap se comparan por region_id y JxCat hereda de CDC."""
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2019-04-28', drange=6, seed=42, verbose=0, path='.')
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    sim.run(split=True, random=False)

    res = sim.result()
    assert res.loc['es', 'COMPROMIS'] > 0
    assert res.loc['es', 'NA+'] > 0
    assert res.loc['es', 'JxCat'] > 0

    # COMPROMIS sólo puede obtener escaños en Alicante, Castellón y Valencia; NA+ sólo en Navarra
    valencia = {'Alicante', 'Castellón', 'Valencia'}
    with_seats = set(res.index[res['COMPROMIS'] > 0]) - {'es'}
    assert with_seats <= valencia, with_seats
    with_seats = set(res.index[res['NA+'] > 0]) - {'es'}
    assert with_seats == {'Navarra'}, with_seats


def test_forecaster_ffill_without_future_warnings(app):
    """B26: fit_forecast(fillna=True) no debe usar APIs de pandas eliminadas en la versión 3."""
    from mtpy.lib.forecaster import Forecaster
    fc = Forecaster(scope='es', event_date='2027-08-22', bmap='max', path='.').build_series()
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        fc.fit_forecast(names=['PP'], max_fc=3, fillna=True)
    assert np.isfinite(fc.forecast['PP'].iloc[-1])
