"""Tests de las utilidades estadísticas que no necesitan base de datos (`mtpy/core/utils/stat.py`)."""
import numpy as np
import pytest

from mtpy.core.utils.stat import Kernel


# --- Sondeos futuros (parte A): tope del ancho adaptativo en la frontera (`bw_cap`) ---

def test_adaptive_bandwidth_cap_limits_the_boundary():
    """El ancho adaptativo se mide sobre una rejilla diaria, como en el promedio: la media geométrica de la densidad
    sale de todos los puntos de la rejilla, así que el tope de unos pocos puntos de la frontera apenas mueve el
    interior (en una rejilla de dos puntos lo movería: los anchos se reescalan juntos en cada iteración)."""
    x = np.concatenate([np.linspace(0, 100, 60), [101, 102]])   # densidad que cae al final
    p = np.arange(0., 103.)                                      # rejilla diaria; 50 es el interior, 102 la frontera
    free = Kernel(x, bw=10., bw_type='adaptive', n_iter=3).get_bw_adaptive(p=p)
    capped = Kernel(x, bw=10., bw_type='adaptive', n_iter=3, bw_cap=1.5).get_bw_adaptive(p=p)
    assert free[102] > 15.                                   # la frontera se ensancha sin tope
    assert capped[102] == pytest.approx(15.)                  # con tope: 1,5 × piloto
    assert capped.max() == pytest.approx(15.)                 # ningún punto lo pasa
    assert capped[50] == pytest.approx(free[50], rel=1e-3)    # el interior no cambia
    # Sin tope, el mismo resultado que antes de `bw_cap`
    assert np.array_equal(Kernel(x, bw=10., n_iter=3, bw_cap=None).get_bw_adaptive(p=p), free)
