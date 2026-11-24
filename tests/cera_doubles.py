"""Dobles de prueba del estudio CERA: sorteos sintéticos del simulador sobre los resultados oficiales de 2023."""
import numpy as np

from mtpy.lib.studies.cera.majority import LINEAGE, Draws, central_with, seats_with, to_model

META = {'scope': 'es', 'event_date': '2026-11-29', 'run_id': '20261124-102045', 'as_of': '2026-11-23', 'n_sim': 20,
        'seed': 42, 'majority': 176,
        'bmaps': {'blocks': {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA', 'UP']}}}
PARTIES = ['PP', 'PSOE', 'VOX', 'FA', 'UP', 'SALF']


def double_draws(study, n_sim: int = 20, seed: int = 42):
    """
    Sorteos sintéticos: el total (región 0) y las 52 provincias de 2023, cada simulación el voto oficial con ruido.

    Parameters
    ----------
    study : CeraStudy
        El estudio, del que se toman los resultados de la elección base.
    n_sim : int, default 20
        Simulaciones.
    seed : int, default 42
        Semilla del ruido log-normal (sigma 0,05).

    Returns
    -------
    Draws
        Con `results` y `central_seats` repartidos con un `delta` nulo.
    """
    results = study.results[study.params['election']['baseline']]
    regions = [0] + [int(region) for region in results.votes.index]
    central = to_model(results.votes, results.siglas, PARTIES, regions, LINEAGE)[:, :-1]
    noise = np.random.default_rng(seed).lognormal(0., 0.05, size=(n_sim,) + central.shape)
    base = np.zeros(len(regions))
    base[1:] = (results.totals['candidates'] + results.totals['blank']).reindex(results.votes.index).values
    seats = np.zeros(len(regions), dtype=int)
    seats[1:] = results.totals['n_seats'].reindex(results.votes.index).values
    seats[0] = seats.sum()
    draws = Draws(names=list(PARTIES), regions=regions, default_region=0, votes=np.round(central * noise),
                  central=central, base=base, seats=seats, threshold=3.0, results=None, central_seats=None)
    zero = np.zeros((len(regions), len(PARTIES) + 1))
    draws.results, draws.central_seats = seats_with(draws, zero), central_with(draws, zero)

    return draws
