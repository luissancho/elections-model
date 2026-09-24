"""Tests del Simulator que no necesitan base de datos."""
from mtpy.lib.simulator import Simulator


def test_dhondt_known_allocation():
    # Ejemplo clásico: 100.000 / 80.000 / 30.000 / 20.000 votos y 8 escaños -> 4 / 3 / 1 / 0
    seats = Simulator.alloc_dhondt({'A': 100000, 'B': 80000, 'C': 30000, 'D': 20000}, 8)
    assert seats == {'A': 4, 'B': 3, 'C': 1, 'D': 0}


def test_dhondt_single_seat_goes_to_plurality():
    assert Simulator.alloc_dhondt({'A': 51, 'B': 49}, 1) == {'A': 1, 'B': 0}
