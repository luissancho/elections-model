"""Tests de integración: necesitan la base de datos local y files/params.json."""
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def sim26(app):
    """Simulador de referencia (sin efectos de casa ni ruido conjunto): la base de las regresiones numéricas."""
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2026-11-29', drange=6, seed=42, verbose=0, path='.', house_effects=False, composition=1.0, regional_noise=False)
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    return sim


@pytest.fixture(scope='module')
def sim26_he(app):
    """Simulador con efectos de casa (M6, por defecto) y ruido conjunto (M7, opcional)."""
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2026-11-29', drange=6, seed=42, verbose=0, path='.', composition='auto')
    assert sim.house_effects is True
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    return sim


def test_run_with_names_subset_does_not_raise(sim26):
    """B3: `names` con un subconjunto de partidos era inutilizable en eventos con smap."""
    sim26.run(split=True, random=False, names=['PP', 'PSOE', 'VOX'])
    assert int(sim26.totals().sum()) == 350


def test_mt_simulations_sum_to_350(sim26):
    sim26.run(split=True, random=True, n_sim=20)
    assert (sim26.dist().sum(axis=1) == 350).all()


def test_mt_is_reproducible_with_seed(sim26):
    sim26.run(split=True, random=True, n_sim=5)
    first = sim26.results.copy()
    sim26.run(split=True, random=True, n_sim=5)
    assert np.array_equal(first, sim26.results)


def test_result_is_indexed_by_region_name(sim26):
    sim26.run(split=True, random=False)
    res = sim26.result()
    assert 'es' in res.index and 'Madrid' in res.index
    assert int(res.loc['es'].sum()) == 350
    madrid = sim26.unit(0, region='Madrid')
    assert madrid.shape[0] == len(sim26.params['names']) + 1
    assert '-' in madrid.index


def test_others_and_provincial_shares_are_consistent(sim26):
    """M1: '-' es el resto del promedio; las cuotas provinciales suman 100 y reproducen el pronóstico nacional."""
    sim26.run(split=True, random=False)
    names = sim26.params['names']
    frame = sim26.frame()

    assert '-' in frame.index
    assert frame.loc['-', 'vpred'] == pytest.approx(100 - frame.loc[names, 'vpred'].sum(), abs=0.02)

    for region in sim26.params['regions']:
        unit = sim26.unit(0, region)
        assert unit['vpred_pct'].sum() == pytest.approx(100, abs=0.15), region
        assert unit['prev_pct'].sum() == pytest.approx(100, abs=0.15), region

    # El voto a otras candidaturas se conserva donde existe (Soria) y es pequeño donde no (Madrid)
    assert sim26.unit(0, 'Soria').loc['-', 'prev_pct'] > 15
    assert sim26.unit(0, 'Madrid').loc['-', 'prev_pct'] < 3

    # La media provincial ponderada por votos válidos reproduce la cuota nacional simulada
    provinces = [r for r in sim26.params['regions'] if r != sim26.default_region]
    weights = sim26.prev_totals.loc[provinces, 'votes']
    for party in names:
        shares = [sim26.unit(0, r).loc[party, 'vpred_pct'] for r in provinces]
        weighted = float((weights.values * np.nan_to_num(shares)).sum() / weights.sum())
        assert weighted == pytest.approx(frame.loc[party, 'vpred'], abs=0.5), party

    assert int(sim26.result().loc['es'].sum()) == 350


def test_threshold_only_changes_provinces_with_parties_near_3pct(sim26):
    """M1: sin barrera el reparto solo cambia donde algún partido está entre el 2 y el 3 % (invariancia de escala)."""
    sim26.run(split=True, random=False)
    with_threshold = sim26.result().copy()

    sim26.threshold = None
    try:
        sim26.run(split=True, random=False)
        without = sim26.result().copy()
    finally:
        sim26.threshold = 3.0

    provinces = [r for r in sim26.params['regions'] if r != sim26.default_region]
    for region in provinces:
        name = sim26.region_names[region]
        shares = sim26.unit(0, region)['vpred_pct'].drop('-')
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
    fc = Forecaster(scope='es', event_date='2026-11-29', bmap='max', path='.').build_series()
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


def test_totals_sum_to_n_seats_both_methods(sim26):
    sim26.run(split=True, random=True, n_sim=30)
    assert int(sim26.totals().sum()) == 350
    assert int(sim26.totals(method='mean').sum()) == 350
    assert set(sim26.totals().index) == set(sim26.params['names'])
    sorted_ = sim26.totals(sort=True)
    assert (sorted_.diff().dropna() <= 0).all()


def test_scenario_is_a_real_simulation(sim26):
    sim26.run(split=True, random=True, n_sim=30)
    idx = sim26.scenario()
    assert 0 <= idx < 30
    res = sim26.result(idx)
    assert int(res.loc['es'].sum()) == 350
    assert res.loc['es'].astype(int).equals(sim26.dist().iloc[idx].astype(int))
    dist = sim26.dist()
    l1 = dist.sub(sim26.central(dist)).abs().sum(axis=1)
    assert l1.iloc[idx] == l1.min()


def test_summary_columns_and_consistency(sim26):
    sim26.run(split=True, random=True, n_sim=30)
    s = sim26.summary()
    assert list(s.columns) == SUMMARY_COLUMNS
    assert list(s.index) == sim26.params['names']
    assert ((s['seats_lo'] <= s['seats_median']) & (s['seats_median'] <= s['seats_hi'])).all()
    for col in ['p_seats', 'p_majority', 'p_first']:
        assert ((s[col] >= 0) & (s[col] <= 1)).all()
    assert int(s['seats'].sum()) == 350
    assert s['p_first'].sum() == pytest.approx(1)
    assert s.loc['PP', 'pct_lo'] <= s.loc['PP', 'pct'] <= s.loc['PP', 'pct_hi']


def test_summary_blocks_and_probabilities(sim26):
    sim26.run(split=True, random=True, n_sim=30)
    blocks = sim26.summary('blocks')
    assert blocks['seats_mean'].sum() == pytest.approx(350)
    assert int(blocks['seats'].sum()) == 350
    vs = sim26.summary('vs')
    assert list(vs.index) == ['Derecha', 'Izquierda']
    probs = sim26.probabilities('vs')
    assert probs.equals(vs['p_majority'].rename('p_majority'))
    coalitions = sim26.probabilities({
        'PP+VOX': ['PP', 'VOX'], 'PP+VOX+UPN+CC': ['PP', 'VOX', 'UPN', 'CC'], 'PSOE+SUMAR+UP': ['PSOE', 'SUMAR', 'UP']
    })
    assert coalitions.notnull().all()  # las coaliciones pueden solaparse
    assert ((coalitions >= 0) & (coalitions <= 1)).all()
    assert coalitions['PP+VOX+UPN+CC'] >= coalitions['PP+VOX']


def test_ls_totals_respect_zero_parties(sim26):
    """B2: en el modo de regresión, los partidos sin escaños en ninguna simulación se quedan en 0."""
    sim26.run(split=False, random=False)
    totals = sim26.totals()
    assert int(totals.sum()) == 350
    assert (totals[sim26.dist().max() == 0] == 0).all()


# --- M4: sucesión de partidos en las encuestas ---

def test_predecessor_polls_count_for_successor_2023(app):
    """M4: las encuestas de UP y MP anteriores a SUMAR entran en el promedio de SUMAR (y las de Cs en el del PP)."""
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2023-07-23', drange=90, seed=42, verbose=0)
    blocks = sim.model.blocks
    assert set(blocks.loc['SUMAR', 'parties']) >= {'SUMAR', 'UP', 'MP'}
    assert set(blocks.loc['PP', 'parties']) >= {'PP', 'Cs'}
    assert blocks.loc['ERC', 'parties'] == ['ERC']

    sim.fit_forecast(names=sim.params['names'], max_fc=10, fillna=True)
    fc = sim.forecast['mean']
    assert fc['SUMAR'] > 8          # antes del cambio quedaba muy por debajo (los sondeos listaban UP y MP)
    assert fc['-'] < 8              # el residuo ya no absorbe a los predecesores


# --- M5: nowcast, horizonte y fecha incierta ---

def test_anchor_is_last_fitted_day_not_deadline(sim26):
    """El pronóstico se lee en `as_of` (última encuesta + max_fc), no en `deadline - drange`."""
    import pandas as pd
    date_last = pd.Timestamp(sim26.model.date_last)
    assert sim26.as_of == pd.Timestamp(sim26.model.date_fit_last)
    assert date_last <= sim26.as_of <= date_last + pd.Timedelta(days=3)  # max_fc=3 en la fixture
    assert sim26.as_of < pd.Timestamp(sim26.limit_date)
    assert sim26.horizon_max == (sim26.deadline - sim26.as_of).days
    assert sim26.horizon_max > 0
    assert sim26.forecast.loc['PP', 'mean'] == pytest.approx(sim26.model.forecast.loc[sim26.as_of, 'PP'])
    assert len(sim26.durations) >= 12 and max(sim26.durations) <= 1491
    assert sim26.v2drift is not None and sim26.v2drift.k > 0


# Regresión numérica de referencia (MT n=200, semilla 42, sin efectos de casa). Era 137 / 112 / 62 / 8 hasta M6b:
# el nuevo `bias` del rating cambió `weight_rating` y con él el promedio (ver `metodo-6-house-effects.md`);
# 139 / 111 / 62 / 8 hasta los cuatro sondeos cargados a finales de septiembre de 2026 (140 / 109 / 61 / 8); y
# desde el método 11 los ratings son globales (todas las elecciones, autonómicas al 0,5), lo que vuelve a
# mover `weight_rating` (ver `metodo-11-ambitos-autonomicos.md`). Depende de los sondeos y ratings guardados.
# Desde el 06-10-2026 el evento es 2026-11-29 (elecciones convocadas) con el reparto de escaños del real decreto,
# Madrid 38 y Cádiz 8: un escaño pasa del PP (Cádiz) a VOX (Madrid), 141 / 107 / 62 / 8 -> 140 / 107 / 63 / 8.
REGRESSION_TOTALS = [140, 107, 63, 8]


def test_nowcast_regression_seed_42(sim26):
    """La secuencia aleatoria del nowcast no cambia con M5: reproduce la regresión de referencia."""
    sim26.run(split=True, random=True, n_sim=200)
    assert (sim26.horizons == 0).all()
    assert sim26.totals()[['PP', 'PSOE', 'VOX', 'SUMAR']].tolist() == REGRESSION_TOTALS
    assert (sim26.frame(0)['drift'].drop('-') == 0).all()


def test_deadline_horizon_widens_intervals(sim26):
    sim26.run(split=True, random=True, n_sim=100)
    now = sim26.summary()
    sim26.run(split=True, random=True, n_sim=100, horizon='deadline')
    fwd = sim26.summary()
    assert (sim26.horizons == sim26.horizon_max).all()
    assert (sim26.dist().sum(axis=1) == 350).all()
    for party in ['PP', 'PSOE', 'VOX']:
        assert fwd.loc[party, 'pct_hi'] - fwd.loc[party, 'pct_lo'] > now.loc[party, 'pct_hi'] - now.loc[party, 'pct_lo'], party
    frame = sim26.frame(0)
    assert frame.loc['PP', 'drift'] > 0
    expected = sim26.v2drift.sigma(sim26.horizon_max, sim26.forecast.loc['PP', 'mean'])
    assert frame.loc['PP', 'drift'] == pytest.approx(expected, abs=0.01)
    # La deriva es proporcional al nivel: una regional al 1 % se mueve mucho menos que el PP
    assert frame.loc['PNV', 'drift'] < 0.2 * frame.loc['PP', 'drift']


def test_random_horizon_uses_the_date_prior(sim26):
    import pandas as pd
    from mtpy.lib.simulator import Simulator
    sim26.run(split=True, random=True, n_sim=50, horizon='random')
    h = sim26.horizons
    assert h.shape == (50,) and h.min() >= 0 and h.max() <= sim26.horizon_max
    elapsed = (sim26.as_of - pd.Timestamp(sim26.prev_date)).days
    cands = Simulator.horizon_candidates('historical', sim26.horizon_max, elapsed, sim26.durations)
    assert set(h.tolist()) <= set(cands.tolist())
    # Con la elección convocada (54 días), toda legislatura histórica supera el límite y los sorteos son el propio
    # límite; el prior sólo reparte fechas cuando el límite queda lejos
    assert len(set(h.tolist())) > 1 or (h == sim26.horizon_max).all()
    assert (sim26.dist().sum(axis=1) == 350).all()

    first, results = h.copy(), sim26.results.copy()
    sim26.run(split=True, random=True, n_sim=50, horizon='random')
    assert np.array_equal(first, sim26.horizons) and np.array_equal(results, sim26.results)

    sim26.run(split=True, random=True, n_sim=20, horizon='random', date_prior='uniform')
    assert 0 <= sim26.horizons.min() and sim26.horizons.max() <= sim26.horizon_max
    assert len(set(sim26.horizons.tolist())) > 1
    with pytest.raises(ValueError):
        sim26.run(split=True, random=False, horizon='random')
    with pytest.raises(ValueError):
        sim26.run(split=True, random=True, n_sim=2, horizon='tomorrow')


def test_fan_widths_grow_with_horizon(sim26):
    fan = sim26.fan()
    assert list(fan.columns) == ['party', 'horizon', 'mean', 'sd', 'lo', 'hi']
    assert {h for h in (0, 30, 90, 180) if h <= sim26.horizon_max} | {sim26.horizon_max} <= set(fan['horizon'])
    for party, g in fan.groupby('party'):
        g = g.sort_values('horizon')
        assert (np.diff(g['hi'] - g['lo']) >= -1e-9).all(), party
    wide = sim26.fan(wide=True)
    assert wide.shape[0] == fan['party'].nunique()
    assert ('lo', 0) in wide.columns and ('hi', sim26.horizon_max) in wide.columns
    # A horizonte 0 la desviación es la del nowcast: sqrt(err² + error²) del pronóstico
    pp = fan[(fan['party'] == 'PP') & (fan['horizon'] == 0)].iloc[0]
    fc = sim26.forecast.loc['PP']
    assert pp['mean'] == pytest.approx(fc['mean'])
    assert pp['sd'] == pytest.approx(np.sqrt(fc['err'] ** 2 + fc['error'] ** 2), abs=1e-6)


# --- M6: efectos de casa ---

def test_house_effects_are_fitted_and_centered(sim26, sim26_he):
    import pandas as pd
    he = sim26_he.model.house_effects
    assert list(he.index.names) == ['pollster_id', 'name']
    cis = he.loc[he['pollster'] == 'CIS'].reset_index().set_index('name')
    assert cis.loc['PP', 'effect'] < -3 and cis.loc['PSOE', 'effect'] > 2
    assert cis.loc['PP', 'n'] >= 10
    # Recentrado por nombre: el nivel del promedio no depende de la corrección
    for name, g in he.groupby(level='name'):
        assert abs((g['effect'] * g['w']).sum() / g['w'].sum()) < 1e-6, name
    # La serie corregida sólo cambia en las filas de encuestas; el promedio corregido del PP queda cerca del bruto
    # (con los ratings globales del método 11 la diferencia es de 1,0 puntos: las casas con efecto positivo en
    # el PP pesan más en la ventana que en el centrado)
    raw, corr = sim26_he.model.series_raw, sim26_he.model.series
    events = raw['pollster'].isnull()
    assert raw.loc[events, sim26_he.model.names].equals(corr.loc[events, sim26_he.model.names])
    assert not raw.loc[~events, 'PP'].equals(corr.loc[~events, 'PP'])
    assert abs(sim26_he.forecast.loc['PP', 'mean'] - sim26.forecast.loc['PP', 'mean']) < 1.5
    # El prior histórico se cargó (hay casas con prior distinto de 0)
    assert (he['prior'].abs() > 0).any()


def test_simulator_without_house_effects_keeps_regression(sim26):
    assert sim26.house_effects is False and sim26.model.house_effects is None
    assert sim26.model.series['PP'].equals(sim26.model.series_raw['PP'])
    sim26.run(split=True, random=True, n_sim=200)
    assert sim26.totals()[['PP', 'PSOE', 'VOX', 'SUMAR']].tolist() == REGRESSION_TOTALS


def test_industry_bias_shifts_the_forecast(app, sim26_he):
    """El sesgo del sector (opcional) desplaza el consenso y ensancha su error; no toca los efectos de casa."""
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2026-11-29', drange=6, seed=42, verbose=0, path='.', industry_bias=True)
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    bias = sim.industry_bias_table
    assert {'bias', 'bias_err'} <= set(bias.columns)
    # En las generales las encuestas subestimaban al PSOE (sesgo −1 punto); con la historia global del método 11
    # (autonómicas al 0,5) el sesgo del sector con el PSOE queda cerca de 0: sólo se comprueba la mecánica
    assert bias.loc['PSOE', 'bias'] != 0 and abs(bias.loc['PSOE', 'bias']) < 1
    for name in ['PP', 'PSOE']:
        assert sim.forecast.loc[name, 'mean'] == pytest.approx(sim26_he.forecast.loc[name, 'mean'] - bias.loc[name, 'bias'], abs=1e-6)
        assert sim.forecast.loc[name, 'err'] >= sim26_he.forecast.loc[name, 'err']
    assert sim26_he.industry_bias is False and sim26_he.industry_bias_table is None


# --- M7: ruido conjunto composicional ---

def test_composition_ratio_reduces_block_variance_not_marginals(sim26, sim26_he):
    """M7: los intervalos por partido no cambian; la suma de los bloques y el recorte del residuo sí."""
    assert sim26.composition_ratio == 1.0
    assert 0.1 < sim26_he.composition_ratio < 0.6
    sim26.run(split=True, random=True, n_sim=400)
    ref = sim26.shares(); ref_clip = sim26.clip_rate()
    sim26_he.run(split=True, random=True, n_sim=400)
    new = sim26_he.shares()
    assert sim26_he.composition_rho < 0
    assert (sim26_he.dist().sum(axis=1) == 350).all()
    # Marginal del PP: misma escala (el error de encuesta difiere algo con los efectos de casa)
    assert new['PP'].std() == pytest.approx(ref['PP'].std(), rel=0.15)
    # La suma de los nacionales se contrae hacia sqrt(r) veces la independiente; un par (PP + VOX) sólo algo
    national = [n for n in sim26_he.params['names'] if sim26_he.forecast.loc[n, 'regional'] == 0]
    total = new[national].sum(axis=1).std(); indep_total = np.sqrt(new[national].var().sum())
    assert total < 0.7 * indep_total, (total, indep_total)
    assert ref[national].sum(axis=1).std() > 0.85 * np.sqrt(ref[national].var().sum())
    assert sim26_he.composition_rho < -0.1
    assert (new['PP'] + new['VOX']).std() < 0.97 * np.sqrt(new['PP'].var() + new['VOX'].var())
    # El residuo se recorta menos
    assert sim26_he.clip_rate() <= ref_clip


# --- M8: multiplicador de deriva por edad del partido ---

def test_party_ages_and_drift_multiplier(sim26_he):
    ages = sim26_he.ages
    assert ages['SALF'] < 4 < ages['VOX'] and ages['PP'] > 30
    assert 2 <= sim26_he.v2drift.multiplier <= 5 and sim26_he.v2drift.n_young >= 6
    sim26_he.run(split=True, random=True, n_sim=50, horizon='deadline')
    frame = sim26_he.frame(0)
    rel = frame['drift'] / frame['pct']
    # Los jóvenes (SUMAR, SALF) derivan `multiplier` veces más que los establecidos, en proporción a su cuota
    assert rel['SUMAR'] == pytest.approx(sim26_he.v2drift.multiplier * rel['PSOE'], rel=0.05)
    assert rel['VOX'] == pytest.approx(rel['PSOE'], rel=0.05)
    # En el abanico, la varianza añadida entre 0 y el límite es la de la deriva: `multiplier²` veces mayor en los jóvenes
    h = sim26_he.horizon_max
    fan = sim26_he.fan(horizons=[0, h]).pivot(index='party', columns='horizon', values='sd')
    added = (fan[h] ** 2 - fan[0] ** 2) / (sim26_he.forecast.loc[fan.index, 'mean'] ** 2 * sim26_he.v2drift.k * h)
    assert added['PP'] == pytest.approx(1., rel=0.02) and added['VOX'] == pytest.approx(1., rel=0.02)
    assert added['SALF'] == pytest.approx(sim26_he.v2drift.multiplier ** 2, rel=0.02)
    assert added['SUMAR'] == pytest.approx(sim26_he.v2drift.multiplier ** 2, rel=0.02)


def test_party_age_uses_the_raw_party_not_its_block(app):
    """M8 (revisión): SUMAR nació en 2023 aunque su bloque herede las encuestas de UP y MP desde 2019."""
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope='es', event_date='2023-07-23', drange=30, seed=42, verbose=0, path='.')
    sim.fit_forecast(names=sim.params['names'], max_fc=10, fillna=True)
    assert sim.ages['SUMAR'] < 1.0
    assert sim.ages['PP'] > 30 and sim.ages['VOX'] < 10
    assert 'SUMAR' in sim.model.party_first_polls.index


# --- M9: oscilaciones autonómicas y provinciales ---

def test_regional_noise_widens_provinces_but_not_the_national_draw(sim26_he):
    import pandas as pd
    assert sim26_he.regional_noise is True and sim26_he.v2swing is not None
    assert sim26_he.v2swing.b_r > 0 and sim26_he.v2swing.sigma_region(30.) > sim26_he.v2swing.sigma_province(30.)
    # Sin oscilaciones primero, para que la fixture quede en su estado por defecto (con ellas) al terminar
    sim26_he.regional_noise = False
    try:
        sim26_he.run(split=True, random=True, n_sim=100)
        shares_off = sim26_he.shares().copy()
        madrid_off = np.array([sim26_he.unit(i, 'Madrid').loc['PP', 'vpred_pct'] for i in range(100)])
        provinces = [r for r in sim26_he.params['regions'] if r != sim26_he.default_region]
        pp_off = {i: np.array([sim26_he.unit(i, r).loc['PP', 'vpred_pct'] for r in provinces]) for i in (0, 7)}
    finally:
        sim26_he.regional_noise = True
    sim26_he.run(split=True, random=True, n_sim=100)
    shares_on = sim26_he.shares().copy()
    madrid_on = np.array([sim26_he.unit(i, 'Madrid').loc['PP', 'vpred_pct'] for i in range(100)])
    assert np.allclose(shares_on.to_numpy(), shares_off.to_numpy(), equal_nan=True)   # mismo sorteo nacional
    assert madrid_on.std() > 1.2 * madrid_off.std()                                   # más dispersión provincial
    assert (sim26_he.dist().sum(axis=1) == 350).all()
    # Las oscilaciones conservan la media nacional ponderada de las cuotas provinciales de cada simulación
    # (la de la proyección determinista, que a su vez queda a una o dos décimas del sorteo nacional)
    w = sim26_he.prev_totals.loc[provinces, 'votes'].to_numpy()
    for i in (0, 7):
        u = np.array([sim26_he.unit(i, r).loc['PP', 'vpred_pct'] for r in provinces])
        assert (u * w).sum() / w.sum() == pytest.approx((pp_off[i] * w).sum() / w.sum(), abs=0.1)   # dos pasadas: ~0,05, hasta 0,06
        assert (u * w).sum() / w.sum() == pytest.approx(sim26_he.frame(i).loc['PP', 'vpred'], abs=0.3)
    summary = sim26_he.unit_summary('Barcelona')
    assert {'pct', 'pct_lo', 'pct_hi', 'seats', 'seats_lo', 'seats_hi', 'p_seats'} <= set(summary.columns)
    assert summary.loc['ERC', 'pct_hi'] > summary.loc['ERC', 'pct_lo']
    assert summary.loc['PP', 'pct_hi'] - summary.loc['PP', 'pct_lo'] > 6   # IC 95 % del PP en Barcelona, con oscilaciones


# --- M10: error estándar del promedio con cluster por casa ---

def test_average_error_is_cluster_robust_by_pollster(sim26_he):
    model = sim26_he.model
    assert model.reg_params['cov_type'] == 'cluster'
    err = model.fc_stat['PP'].loc[sim26_he.as_of]['err']
    assert np.isfinite(err) and 0.1 < err < 1.0
    # Mismo ajuste (misma rejilla, `max_fc` de la fixture) con los tres estimadores: el cluster por casa es
    # mayor que el Newey-West en orden de filas, que no ve la dependencia por casa (medido: 0,25 frente a 0,20)
    params = model.reg_params
    errs = {}
    try:
        for cov_type in ('cluster', 'hac', 'hc1'):
            model.reg_params = model.set_reg_params({'cov_type': cov_type})
            _, stat = model.fit('PP', max_fc=3, ret_stat=True)
            errs[cov_type] = stat.loc[sim26_he.as_of]['err']
    finally:
        model.reg_params = params
    assert errs['cluster'] == pytest.approx(err, rel=1e-6)
    assert errs['cluster'] > errs['hac'] and errs['cluster'] > errs['hc1']
