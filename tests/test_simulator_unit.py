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
