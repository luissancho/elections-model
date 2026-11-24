"""La medida pareada del voto CERA: los mismos sorteos repartidos con y sin un escenario de voto exterior."""
import numpy as np
import pandas as pd

from mtpy.lib.simulator import Simulator
from mtpy.lib.studies.cera.majority import (
    LINEAGE,
    Draws,
    allocate,
    applied_lineage,
    base_counts,
    central_with,
    measure,
    province_counts,
    reproduction_problems,
    seat_range,
    seats_with,
    to_model,
)

NAMES, BLOCKS = ['R', 'L'], {'Derecha': ['R'], 'Izquierda': ['L']}


def tiny():
    """Four simulations, two provinces (3 and 2 seats), no threshold; region 0 is the total."""
    one = np.array([[310., 100.], [250., 150.], [210., 190.], [150., 250.]])
    votes = np.stack([np.zeros((4, 2)), one, np.tile([100., 90.], (4, 1))], axis=1)
    draws = Draws(names=NAMES, regions=[0, 1, 2], default_region=0, votes=votes,
                  central=np.array([[0., 0.], [250., 150.], [100., 90.]]), base=np.array([0., 400., 190.]),
                  seats=np.array([5, 3, 2]), threshold=None, results=None, central_seats=None)
    zero = np.zeros((3, 3))
    draws.results, draws.central_seats = seats_with(draws, zero), central_with(draws, zero)
    return draws


def test_measure_counts_what_a_hand_count_gives():
    """La medida de un escenario coincide con el recuento a mano de cuatro simulaciones."""
    draws = tiny()
    assert draws.results[:, 0, 0].tolist() == [4, 3, 3, 2] and draws.central_seats[0].tolist() == [3, 2]
    delta = np.array([[0., 0., 0.], [0., 30., 0.], [100., 0., 0.]])
    assert measure(draws, delta, BLOCKS, majority=3) == {
        'votes': 130, 'lost': 0, 'gained': 1, 'moved': 4, 'right': {'0': 2, '1': 2}, 'left': {'-1': 2, '0': 2},
        'central': {'right': 1, 'left': -1}}
    assert province_counts(draws, delta, BLOCKS) == [
        {'region_id': 1, 'moved': 2, 'right_lost': 2, 'right_gained': 0},
        {'region_id': 2, 'moved': 4, 'right_lost': 0, 'right_gained': 4}]
    assert base_counts(draws, BLOCKS, majority=3) == {
        'with_majority': 3, 'central': {'Derecha': 3, 'Izquierda': 2}, 'right_seats': {'2': 1, '3': 2, '4': 1}}


def test_a_zero_delta_changes_nothing_and_reproduces_the_stored_seats():
    """Un delta nulo no mueve nada y reproduce los escaños guardados; si no, avisa de lo que difiere."""
    draws = tiny()
    assert measure(draws, np.zeros((3, 3)), BLOCKS, 3)['moved'] == 0 and reproduction_problems(draws) == []
    draws.results = draws.results.copy()
    draws.results[2, 1] = [3, 0]
    problems = reproduction_problems(draws)
    assert len(problems) == 1 and '1 of 4' in problems[0]
    draws.central_seats = draws.central_seats[:, ::-1]
    assert len(reproduction_problems(draws)) == 2


def test_allocate_equals_the_exact_allocation_with_ties_and_threshold():
    """El reparto vectorizado coincide con el exacto del simulador, con empates, umbral y sorteos sin elegibles."""
    rng, names = np.random.default_rng(7), ['A', 'B', 'C', 'D', 'E']
    votes = np.round(rng.dirichlet([8, 6, 3, 1, 0.3], size=300) * 10000)
    votes[:4] = [[600, 300, 100, 0, 0], [500, 500, 0, 0, 0], [0, 0, 0, 0, 0], [20, 20, 20, 20, 20]]
    for n_seats, threshold in ((7, 3.0), (3, 3.0), (2, 25.0), (5, None)):
        got = allocate(votes, n_seats, 10500., threshold, names)
        for row, seats in zip(votes, got):
            exact = Simulator.alloc_seats(dict(zip(names, row.tolist())), n_seats, valid_votes=10500.,
                                          threshold=threshold)
            assert seats.tolist() == [exact[name] for name in names]


def test_a_delta_that_takes_more_votes_than_there_are_stops_at_zero():
    """Un delta que quita más votos de los que hay deja el partido en cero, sin votos negativos."""
    draws = tiny()
    seats = seats_with(draws, np.array([[0., 0., 0.], [0., -500., 50.], [0., 0., 0.]]))
    assert (seats[:, 1] == [3, 0]).all() and (seats >= 0).all()


def test_the_threshold_base_moves_with_the_net_votes_of_the_scenario():
    """La base del umbral de una provincia se mueve con los votos netos del escenario, residuo incluido."""
    votes = np.array([[[0., 0., 0.], [500., 260., 240.]]])
    draws = Draws(names=['A', 'B', 'C'], regions=[0, 1], default_region=0, votes=votes, central=votes[0],
                  base=np.array([0., 1000.]), seats=np.array([4, 4]), threshold=25.0, results=None,
                  central_seats=None)
    assert seats_with(draws, np.zeros((2, 4)))[0, 1].tolist() == [3, 1, 0]  # C has 24 % of 1000: out
    less = np.array([[0., 0., 0., 0.], [0., 0., 0., -50.]])  # 50 residual votes less: C has 25.3 % of 950
    assert seats_with(draws, less)[0, 1].tolist() == [2, 1, 1] and central_with(draws, less)[1].tolist() == [2, 1, 1]


def test_to_model_follows_the_lineage_and_sends_the_rest_to_the_residual():
    """to_model sigue el linaje, manda el resto al residuo e ignora las provincias que el modelo no tiene."""
    frame = pd.DataFrame({'Sumar': [10., 1.], 'UPN': [5., 0.], 'Zeta': [2., 2.], 'Nadie': [1., 0.]}, index=[28, 8])
    siglas = {'Sumar': 'SUMAR', 'UPN': 'UPN', 'Zeta': 'ZZ', 'Nadie': None}
    assert to_model(frame, siglas, ['PP', 'FA', 'UP'], [0, 28, 8, 31], LINEAGE).tolist() == [
        [0., 0., 0., 0.], [5., 10., 0., 3.], [0., 1., 0., 2.], [0., 0., 0., 0.]]
    assert to_model(frame, siglas, ['PP', 'FA', 'UP'], [0, 28], LINEAGE).tolist() == [
        [0., 0., 0., 0.], [5., 10., 0., 3.]]
    assert applied_lineage(siglas, ['PP', 'FA', 'UP'], LINEAGE) == {'SUMAR': 'FA', 'UPN': 'PP', 'ZZ': '-'}


def test_seat_range_is_the_95_per_cent_of_the_stored_distribution():
    """El rango de escaños es el 95 % de la distribución guardada, expandida con cuantiles lineales."""
    assert seat_range({'-2': 8, '-1': 367, '0': 9429, '1': 196}) == [-1.0, 0.0]
    assert seat_range({'0': 10}) == [0.0, 0.0]
