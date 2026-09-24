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
    madrid = sim27.unit(0, region='Madrid')
    assert madrid.shape[0] == len(sim27.params['names']) + 1
    assert '-' in madrid.index


def test_others_and_provincial_shares_are_consistent(sim27):
    """M1: '-' es el resto del promedio; las cuotas provinciales suman 100 y reproducen el pronóstico nacional."""
    sim27.run(split=True, random=False)
    names = sim27.params['names']
    frame = sim27.frame()

    assert '-' in frame.index
    assert frame.loc['-', 'vpred'] == pytest.approx(100 - frame.loc[names, 'vpred'].sum(), abs=0.02)

    for region in sim27.params['regions']:
        unit = sim27.unit(0, region)
        assert unit['vpred_pct'].sum() == pytest.approx(100, abs=0.15), region
        assert unit['prev_pct'].sum() == pytest.approx(100, abs=0.15), region

    # El voto a otras candidaturas se conserva donde existe (Soria) y es pequeño donde no (Madrid)
    assert sim27.unit(0, 'Soria').loc['-', 'prev_pct'] > 15
    assert sim27.unit(0, 'Madrid').loc['-', 'prev_pct'] < 3

    # La media provincial ponderada por votos válidos reproduce la cuota nacional simulada
    provinces = [r for r in sim27.params['regions'] if r != sim27.default_region]
    weights = sim27.prev_totals.loc[provinces, 'votes']
    for party in names:
        shares = [sim27.unit(0, r).loc[party, 'vpred_pct'] for r in provinces]
        weighted = float((weights.values * np.nan_to_num(shares)).sum() / weights.sum())
        assert weighted == pytest.approx(frame.loc[party, 'vpred'], abs=0.5), party

    assert int(sim27.result().loc['es'].sum()) == 350


def test_threshold_only_changes_provinces_with_parties_near_3pct(sim27):
    """M1: sin barrera el reparto solo cambia donde algún partido está entre el 2 y el 3 % (invariancia de escala)."""
    sim27.run(split=True, random=False)
    with_threshold = sim27.result().copy()

    sim27.threshold = None
    try:
        sim27.run(split=True, random=False)
        without = sim27.result().copy()
    finally:
        sim27.threshold = 3.0

    provinces = [r for r in sim27.params['regions'] if r != sim27.default_region]
    for region in provinces:
        name = sim27.region_names[region]
        shares = sim27.unit(0, region)['vpred_pct'].drop('-')
        near = bool(((shares >= 2) & (shares < 3)).any())
        if not near:
            assert with_threshold.loc[name].equals(without.loc[name]), name


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


# --- M2: resúmenes de la simulación ---

SUMMARY_COLUMNS = [
    'pct', 'pct_mean', 'pct_lo', 'pct_hi',
    'seats', 'seats_mean', 'seats_median', 'seats_lo', 'seats_hi', 'seats_min', 'seats_max',
    'p_seats', 'p_majority', 'p_first'
]


def test_totals_sum_to_n_seats_both_methods(sim27):
    sim27.run(split=True, random=True, n_sim=30)
    assert int(sim27.totals().sum()) == 350
    assert int(sim27.totals(method='mean').sum()) == 350
    assert set(sim27.totals().index) == set(sim27.params['names'])
    sorted_ = sim27.totals(sort=True)
    assert (sorted_.diff().dropna() <= 0).all()


def test_scenario_is_a_real_simulation(sim27):
    sim27.run(split=True, random=True, n_sim=30)
    idx = sim27.scenario()
    assert 0 <= idx < 30
    res = sim27.result(idx)
    assert int(res.loc['es'].sum()) == 350
    assert res.loc['es'].astype(int).equals(sim27.dist().iloc[idx].astype(int))
    dist = sim27.dist()
    l1 = dist.sub(sim27.central(dist)).abs().sum(axis=1)
    assert l1.iloc[idx] == l1.min()


def test_summary_columns_and_consistency(sim27):
    sim27.run(split=True, random=True, n_sim=30)
    s = sim27.summary()
    assert list(s.columns) == SUMMARY_COLUMNS
    assert list(s.index) == sim27.params['names']
    assert ((s['seats_lo'] <= s['seats_median']) & (s['seats_median'] <= s['seats_hi'])).all()
    for col in ['p_seats', 'p_majority', 'p_first']:
        assert ((s[col] >= 0) & (s[col] <= 1)).all()
    assert int(s['seats'].sum()) == 350
    assert s['p_first'].sum() == pytest.approx(1)
    assert s.loc['PP', 'pct_lo'] <= s.loc['PP', 'pct'] <= s.loc['PP', 'pct_hi']


def test_summary_blocks_and_probabilities(sim27):
    sim27.run(split=True, random=True, n_sim=30)
    blocks = sim27.summary('blocks')
    assert blocks['seats_mean'].sum() == pytest.approx(350)
    assert int(blocks['seats'].sum()) == 350
    vs = sim27.summary('vs')
    assert list(vs.index) == ['Derecha', 'Izquierda']
    probs = sim27.probabilities('vs')
    assert probs.equals(vs['p_majority'].rename('p_majority'))
    coalitions = sim27.probabilities({
        'PP+VOX': ['PP', 'VOX'], 'PP+VOX+UPN+CC': ['PP', 'VOX', 'UPN', 'CC'], 'PSOE+SUMAR+UP': ['PSOE', 'SUMAR', 'UP']
    })
    assert coalitions.notnull().all()  # las coaliciones pueden solaparse
    assert ((coalitions >= 0) & (coalitions <= 1)).all()
    assert coalitions['PP+VOX+UPN+CC'] >= coalitions['PP+VOX']


def test_ls_totals_respect_zero_parties(sim27):
    """B2: en el modo de regresión, los partidos sin escaños en ninguna simulación se quedan en 0."""
    sim27.run(split=False, random=False)
    totals = sim27.totals()
    assert int(totals.sum()) == 350
    assert (totals[sim27.dist().max() == 0] == 0).all()
