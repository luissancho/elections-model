"""Tests de las utilidades estadísticas que no necesitan base de datos (`mtpy/core/utils/stat.py`)."""
import numpy as np
import pytest

from mtpy.core.utils.stat import Kernel


# --- Sondeos futuros (parte A): tope del ancho adaptativo en la frontera (`bw_cap`) ---

def test_adaptive_bandwidth_cap_limits_the_boundary():
    """El ancho adaptativo se mide sobre una rejilla diaria, como en el promedio: la media geométrica de la densidad
    sale de los puntos de predicción, y con sólo dos (50 y 102) la frontera no llega a ensancharse por encima del
    tope (12,7 frente a 15). El tope se aplica una vez, tras la última iteración: el interior no cambia."""
    x = np.concatenate([np.linspace(0, 100, 60), [101, 102]])   # densidad que cae al final
    p = np.arange(0., 103.)                                      # rejilla diaria; 50 es el interior, 102 la frontera
    free = Kernel(x, bw=10., bw_type='adaptive', n_iter=3).get_bw_adaptive(p=p)
    capped = Kernel(x, bw=10., bw_type='adaptive', n_iter=3, bw_cap=1.5).get_bw_adaptive(p=p)
    assert free[102] > 15.                                   # la frontera se ensancha sin tope
    assert capped[102] == pytest.approx(15.)                  # con tope: 1,5 × piloto
    assert capped.max() == pytest.approx(15.)                 # ningún punto lo pasa
    assert capped[50] == pytest.approx(free[50])              # el interior no cambia
    assert np.array_equal(capped[free <= 15.], free[free <= 15.])  # sólo cambian los puntos que pasaban el tope
    # Las iteraciones que devuelve `ret_iter` son las del ancho libre; el tope sólo recorta el ancho final
    final, iterations = Kernel(x, bw=10., n_iter=3, bw_cap=1.5).get_bw_adaptive(p=p, ret_iter=True)
    assert np.array_equal(final, capped) and np.array_equal(iterations[-1], free)
    # Sin tope, el mismo resultado que antes de `bw_cap`
    assert np.array_equal(Kernel(x, bw=10., n_iter=3, bw_cap=None).get_bw_adaptive(p=p), free)
