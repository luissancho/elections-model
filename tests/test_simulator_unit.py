"""Tests del Simulator que no necesitan base de datos."""
from mtpy.lib.simulator import Simulator


def test_dhondt_known_allocation():
    # Ejemplo clásico: 100.000 / 80.000 / 30.000 / 20.000 votos y 8 escaños -> 4 / 3 / 1 / 0
    seats = Simulator.alloc_dhondt({'A': 100000, 'B': 80000, 'C': 30000, 'D': 20000}, 8)
    assert seats == {'A': 4, 'B': 3, 'C': 1, 'D': 0}


def test_dhondt_single_seat_goes_to_plurality():
    assert Simulator.alloc_dhondt({'A': 51, 'B': 49}, 1) == {'A': 1, 'B': 0}


def test_threshold_excludes_party_below_3pct():
    """M1: la barrera del 3 % (art. 163.1.a LOREG) deja fuera del reparto a quien no la alcanza."""
    seats = Simulator.alloc_seats({'A': 60, 'B': 37, 'C': 2.9}, 37, valid_votes=100, threshold=3.0)
    assert seats['C'] == 0
    assert sum(seats.values()) == 37


def test_without_threshold_c_gets_a_seat():
    seats = Simulator.alloc_seats({'A': 60, 'B': 37, 'C': 2.9}, 37, valid_votes=100, threshold=None)
    assert seats['C'] == 1
    assert sum(seats.values()) == 37


def test_threshold_base_is_valid_votes_not_the_sum():
    # B tiene el 3,0 % de 100 votos válidos (elegible) pero el 2,97 % de 101 (excluido)
    assert Simulator.alloc_seats({'A': 60, 'B': 3}, 21, valid_votes=100, threshold=3.0)['B'] == 1
    assert Simulator.alloc_seats({'A': 60, 'B': 3}, 21, valid_votes=101, threshold=3.0)['B'] == 0


def test_no_eligible_party_allocates_nothing():
    seats = Simulator.alloc_seats({'A': 1, 'B': 1}, 1, valid_votes=100, threshold=3.0)
    assert seats == {'A': 0, 'B': 0}


# --- M2: resúmenes de la simulación (funciones puras, sin base de datos) ---

import numpy as np
import pandas as pd
import pytest

from mtpy.lib.utils import build_blocks


@pytest.fixture
def dist6():
    # Seis simulaciones de 10 escaños; mediana y media [6, 2, 2, 0]; mayoría absoluta = 6
    return pd.DataFrame({'A': [5, 6, 6, 7, 8, 4], 'B': [3, 2, 2, 1, 0, 4], 'C': [2] * 6, 'D': [0] * 6})


def test_round_proportional_exact_sums_and_zeros():
    rp = Simulator.round_proportional
    assert rp(pd.Series([1, 1, 1, 0]), 4).tolist() == [2, 1, 1, 0]
    assert rp(pd.Series([4.5, 3.5, 2.0]), 10).tolist() == [5, 3, 2]
    assert rp(pd.Series([50.6, 30.3, 19.1]), 100).tolist() == [51, 30, 19]
    assert rp(pd.Series([10, 10, 5]), 20).tolist() == [8, 8, 4]  # base mayor que el total
    assert rp(pd.Series([3, 3, 3]), 10).tolist() == [4, 3, 3]  # empate: posición
    assert rp(pd.Series([0, 0, 0]), 10).tolist() == [0, 0, 0]
    assert rp(pd.Series([2, np.nan, 2]), 4).tolist() == [2, 0, 2]  # NaN cuenta como 0


def test_round_proportional_ls_like_vector_keeps_zeros():
    """B2: la expansión proporcional nunca da escaños a partidos con 0 (antes los 'ciclos' daban +1 a todos)."""
    raw = pd.Series([126, 105, 61, 10, 9, 5, 4, 3, 2, 1, 0, 0, 0], index=list('ABCDEFGHIJKLM'))
    out = Simulator.round_proportional(raw, 350)
    assert int(out.sum()) == 350
    assert (out[raw == 0] == 0).all()
    scaled = raw * 350 / raw.sum()
    assert ((out >= np.floor(scaled)) & (out <= np.floor(scaled) + 1)).all()
    assert out.index.equals(raw.index) and out.dtype.kind == 'i'


def test_central_median_mean_and_error(dist6):
    assert Simulator.central(dist6).tolist() == [6, 2, 2, 0]
    assert Simulator.central(dist6, 'mean').tolist() == [6, 2, 2, 0]
    with pytest.raises(ValueError):
        Simulator.central(dist6, 'mode')


def test_describe_dist_columns_and_nan(dist6):
    desc = Simulator.describe_dist(dist6, alpha=0.05)
    assert list(desc.columns) == ['mean', 'median', 'lo', 'hi', 'min', 'max']
    assert desc.loc['A', ['lo', 'hi']].tolist() == [4, 8]  # con 6 filas el 95 % es mín-máx
    assert desc.loc['B', 'median'] == 2
    nan_col = dist6.assign(E=np.nan)
    assert Simulator.describe_dist(nan_col).loc['E'].isnull().all()


def test_summarize_seats_values(dist6):
    s = Simulator.summarize_seats(dist6, 10)
    assert list(s.columns) == [
        'seats', 'seats_mean', 'seats_median', 'seats_lo', 'seats_hi', 'seats_min', 'seats_max',
        'p_seats', 'p_majority', 'p_first'
    ]
    assert s['seats'].tolist() == [6, 2, 2, 0]
    assert s['p_seats'].tolist() == pytest.approx([1, 5 / 6, 1, 0])
    assert s['p_majority'].tolist() == pytest.approx([4 / 6, 0, 0, 0])
    assert s['p_first'].tolist() == pytest.approx([1, 0, 0, 0])
    assert ((s['seats_lo'] <= s['seats_median']) & (s['seats_median'] <= s['seats_hi'])).all()

    given = Simulator.summarize_seats(dist6, 10, headline=pd.Series({'A': 7, 'B': 1, 'C': 2, 'D': 0}))
    assert given['seats'].tolist() == [7, 1, 2, 0]

    blocks = dist6.assign(E=np.nan)  # un bloque sin partidos presentes
    s = Simulator.summarize_seats(blocks, 10)
    assert s.loc['E', ['p_seats', 'p_majority', 'p_first']].isnull().all()


def test_majority_probs_groups(dist6):
    p = Simulator.majority_probs(dist6, 8, groups={'AC': ['A', 'C'], 'BD': ['B', 'D']})
    assert p['AC'] == pytest.approx(4 / 6)
    assert p['BD'] == 0
    blocks = build_blocks({'AC': ['A', 'C'], 'BD': ['B', 'D']}, ['A', 'B', 'C', 'D'])
    assert Simulator.majority_probs(dist6, 8, groups=blocks).equals(p)
    assert np.isnan(Simulator.majority_probs(dist6, 8, groups={'X': ['Z']})['X'])
    overlap = Simulator.majority_probs(dist6, 8, groups={'AC': ['A', 'C'], 'ABC': ['A', 'B', 'C']})
    assert overlap['AC'] == pytest.approx(4 / 6) and overlap['ABC'] == 1  # los grupos pueden solaparse


def test_closest_simulation_ties(dist6):
    assert Simulator.closest_simulation(dist6, pd.Series([6, 2, 2, 0], index=list('ABCD'))) == 1
    two = pd.DataFrame({'A': [7, 8], 'B': [1, 2], 'C': [2, 2], 'D': [0, 0]})  # L1 = 2 y 2; L2 = 2 y 4
    assert Simulator.closest_simulation(two, pd.Series([6, 2, 2, 0], index=list('ABCD'))) == 0


# --- M4: sucesión de partidos en las encuestas ---

def test_poll_blocks_from_smap_agg_rules():
    names = ['PP', 'PSOE', 'SUMAR', 'ERC', 'NA+']
    smap = {
        'PP': [{'type': 'agg', 'names': ['Cs']}],
        'SUMAR': [{'type': 'agg', 'names': ['UP', 'MP', 'MES']}],
        'NA+': [{'type': 'agg', 'names': ['PP', 'Cs'], 'regions': ['31']}],   # regional: no aplica a encuestas
        'UP': [{'type': 'sub', 'names': ['SUMAR']}],                           # sub: no aplica
        'PRC': [{'type': 'split', 'names': ['PP', 'PSOE']}],                   # partido no simulado: se ignora
    }
    blocks = Simulator.poll_blocks(names, smap)
    assert list(blocks) == names
    assert blocks['PP'] == ['PP', 'Cs']
    assert blocks['SUMAR'] == ['SUMAR', 'UP', 'MP', 'MES']
    assert blocks['NA+'] == ['NA+'] and blocks['ERC'] == ['ERC']


def test_poll_blocks_never_absorbs_a_simulated_party():
    blocks = Simulator.poll_blocks(['A', 'B'], {'A': {'type': 'agg', 'names': ['B', 'C']}})
    assert blocks == {'A': ['A', 'C'], 'B': ['B']}


# --- M5: deriva de la opinión con el horizonte y fecha incierta (sin base de datos) ---

from mtpy.lib.computer import DriftEstimator


def _drift_table(k=1e-4, event_date='2023-07-23'):
    # Paseo aleatorio relativo: rms = level · sqrt(k · d) a partir de 60 días; por debajo la curva empírica
    # crece como d (artefacto de suavizado). Dos partidos de nivel muy distinto con la misma deriva relativa.
    horizons = [7, 14, 30, 60, 90, 180, 365]
    rows = []
    for party, level in [('A', 30.), ('B', 1.)]:
        for d in horizons:
            rms = level * (np.sqrt(k * d) if d >= 60 else 0.0035 * d)
            rows.append({'event_date': event_date, 'party': party, 'horizon': d, 'level': level, 'n': 100, 'rms': rms, 'bias': 0.})
    return pd.DataFrame(rows)


def test_drift_estimator_recovers_relative_random_walk_constant():
    est = DriftEstimator.fit(_drift_table(), d_min=60)
    assert est.k == pytest.approx(1e-4)
    assert est.sigma(0, 30.) == 0
    assert est.sigma(100, 30.) == pytest.approx(30 * np.sqrt(1e-2))   # 3 puntos para un partido al 30 %
    assert est.sigma(100, 1.) == pytest.approx(np.sqrt(1e-2))         # una décima para uno al 1 %
    sig = est.sigma(np.array([0, 7, 30, 90, 365]), 20.)
    assert (np.diff(sig) >= 0).all()
    vec = est.sigma(90, np.array([30., 1.]))
    assert vec.tolist() == pytest.approx([30 * np.sqrt(9e-3), np.sqrt(9e-3)])
    assert list(est.curve.index) == [7, 14, 30, 60, 90, 180, 365]
    assert est.curve.loc[90] == pytest.approx(np.sqrt(1e-4 * 90))       # deriva relativa a 90 días


def test_drift_estimator_ignores_short_horizons_and_is_geometric():
    table = _drift_table()
    table.loc[table['horizon'] < 60, 'rms'] *= 10  # los horizontes cortos no entran en el ajuste de k
    assert DriftEstimator.fit(table, d_min=60).k == pytest.approx(1e-4)
    # Con pesos distintos por fila el ajuste sigue siendo exacto porque los datos cumplen la ley
    table.loc[table['party'] == 'B', 'n'] = 5
    assert DriftEstimator.fit(table, d_min=60).k == pytest.approx(1e-4)
    # Media geométrica: un partido con deriva relativa 100 veces mayor (nacido en el ciclo) sólo la multiplica por 10;
    # la cuadrática (varianza media) la multiplica por 50,5
    table = _drift_table()
    table.loc[table['party'] == 'B', 'rms'] *= 10
    assert DriftEstimator.fit(table, d_min=60).k == pytest.approx(1e-4 * 10)
    assert DriftEstimator.fit(table, d_min=60, agg='quadratic').k == pytest.approx(1e-4 * 50.5)
    assert DriftEstimator.fit(_drift_table(), d_min=60, agg='quadratic').k == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        DriftEstimator.fit(table, agg='harmonic')


def test_drift_estimator_year_decay_favours_recent_cycles():
    old = _drift_table(k=0.01, event_date='2011-11-20')
    new = _drift_table(k=0.09, event_date='2023-07-23')
    table = pd.concat([old, new], ignore_index=True)
    flat = DriftEstimator.fit(table, d_min=60).k
    decayed = DriftEstimator.fit(table, d_min=60, decay=0.9).k
    assert flat == pytest.approx(np.sqrt(0.01 * 0.09))        # mismo peso: media geométrica de los dos ciclos
    assert flat < decayed < 0.09                               # con decaimiento pesa más el ciclo reciente
    w_old = 0.9 ** ((pd.Timestamp('2023-07-23') - pd.Timestamp('2011-11-20')).days / 365.25)
    assert decayed == pytest.approx(np.exp((np.log(0.01) * w_old + np.log(0.09)) / (1 + w_old)))
    assert DriftEstimator.fit(new, d_min=60, decay=0.9).k == pytest.approx(0.09)  # un solo ciclo: sin efecto


def test_drift_estimator_without_long_horizons_is_zero():
    table = _drift_table()
    table = table[table['horizon'] < 60]
    with pytest.warns(UserWarning):
        est = DriftEstimator.fit(table, d_min=60)
    assert np.isnan(est.k)
    assert est.sigma(100, 30.) == 0
    assert est.var(np.array([10, 20]), 30.).tolist() == [0., 0.]


def test_horizon_candidates_historical_prior():
    cands = Simulator.horizon_candidates('historical', 450, elapsed_days=1000, durations=[600, 1200, 1400, 1491])
    assert cands.tolist() == [200, 400, 450]  # 600 descartada (ya superada); 491 acotada por el límite


def test_horizon_candidates_deadline_uniform_and_array():
    assert Simulator.horizon_candidates('deadline', 332).tolist() == [332]
    uniform = Simulator.horizon_candidates('uniform', 5)
    assert uniform.tolist() == [0, 1, 2, 3, 4, 5]
    assert Simulator.horizon_candidates([-3, 10, 999], 100).tolist() == [0, 10, 100]
    with pytest.raises(ValueError):
        Simulator.horizon_candidates('tomorrow', 100)


def test_horizon_candidates_falls_back_to_uniform():
    with pytest.warns(UserWarning):
        cands = Simulator.horizon_candidates('historical', 300, elapsed_days=5000, durations=[600, 1491])
    assert cands.tolist() == list(range(301))


def test_horizon_sampler_stays_within_bounds_and_is_reproducible():
    sampler = Simulator.horizon_sampler('uniform', 50)
    draws = sampler(np.random.default_rng(0), 500)
    assert draws.shape == (500,)
    assert draws.min() >= 0 and draws.max() <= 50
    assert np.array_equal(draws, sampler(np.random.default_rng(0), 500))
    hist = Simulator.horizon_sampler('historical', 450, 1000, [600, 1200, 1400, 1491])(np.random.default_rng(1), 300)
    assert set(hist.tolist()) <= {200, 400, 450}
