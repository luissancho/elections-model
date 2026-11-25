"""
The paired measure of the CERA study: what a scenario of extra exterior votes changes in the simulations of the model.

The model simulates an election many times; each simulation has the votes of every province and party and the seats
the D'Hondt rule gave them. Here the very same simulation is allocated again with the votes of a scenario added
(`delta`) and compared with the stored seats draw by draw, so the difference is the effect of the scenario and not
the noise of the simulation. The same is done with the central allocation of the model.

Everything is pure arrays (no database, no files, no study objects). What `measure`, `province_counts` and
`base_counts` return is plain Python, ready for `json.dumps`; the keys of the distributions are strings of integers in
ascending numeric order.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ...simulator import Simulator

RIGHT, LEFT, REST = 'Derecha', 'Izquierda', '-'
LINEAGE = {'SUMAR': 'FA', 'UPN': 'PP'}


@dataclass
class Draws:
    """
    The simulations of a published run and the central allocation, as arrays.

    Attributes
    ----------
    names : list
        Parties, in the order of the simulator.
    regions : list
        Ids of the regions, the total included.
    default_region : int
        Id of the total (the whole scope), whose seats are the sum of the others.
    votes : np.ndarray
        Simulations × regions × parties: the rounded votes the simulator allocated.
    central : np.ndarray
        Regions × parties: the votes of the central allocation.
    base : np.ndarray
        Regions: the valid votes, base of the threshold.
    seats : np.ndarray
        Regions: seats of each region.
    threshold : float, optional
        Percentage of the valid votes of the region needed to take part in the allocation.
    results : np.ndarray
        Simulations × regions × parties: the stored seats.
    central_seats : np.ndarray
        Regions × parties: the stored seats of the central allocation.
    """
    names: list
    regions: list
    default_region: int
    votes: np.ndarray
    central: np.ndarray
    base: np.ndarray
    seats: np.ndarray
    threshold: Optional[float]
    results: np.ndarray
    central_seats: np.ndarray


def allocate(votes: np.ndarray, n_seats: int, base: float, threshold: Optional[float], names: list) -> np.ndarray:
    """
    D'Hondt allocation with the legal threshold of many draws of one region at once.

    The quotients of every draw are built together and the `n_seats` largest are taken. A draw whose cut is tied, or
    where nothing is eligible, is allocated with `Simulator.alloc_seats`, the reference for those cases (a tie goes to
    the first party in order).

    Parameters
    ----------
    votes : np.ndarray
        Draws × parties.
    n_seats : int
        Seats of the region.
    base : float
        Valid votes, base of the threshold (the votes of each draw if it is zero).
    threshold : float, optional
        Percentage of `base` needed to take part in the allocation.
    names : list
        Parties, in the order of the columns.

    Returns
    -------
    np.ndarray
        Draws × parties, integer seats.
    """
    votes = np.asarray(votes, dtype=float)
    n_draws, n_parties = votes.shape
    if n_seats <= 0 or n_draws == 0:
        return np.zeros(votes.shape, dtype=np.int16)

    eligible = votes > 0
    if threshold is not None:
        row_base = np.full(n_draws, float(base)) if base else votes.sum(axis=1)
        # Without a positive base the simulator lets every candidature in
        positive = (row_base > 0)[:, None]
        share = 100. * votes / np.where(positive, row_base[:, None], 1.)
        eligible = eligible & (~positive | (share >= threshold))

    quotients = np.where(eligible, votes, 0.)[:, :, None] / np.arange(1, n_seats + 1)[None, None, :]
    flat = quotients.reshape(n_draws, -1)
    kth = np.partition(flat, flat.shape[1] - n_seats, axis=1)[:, flat.shape[1] - n_seats]
    seats = (quotients >= kth[:, None, None]).sum(axis=2).astype(np.int16)

    for i in np.where((seats.sum(axis=1) != n_seats) | (kth <= 0))[0]:
        exact = Simulator.alloc_seats(dict(zip(names, votes[i].tolist())), n_seats,
                                      valid_votes=float(base), threshold=threshold)
        seats[i] = [exact[name] for name in names]

    return seats


def _scenario_votes(votes: np.ndarray, delta_row: np.ndarray) -> np.ndarray:
    """Votes with a scenario: `max(votes + delta, 0)`, the residual column of `delta_row` left out."""
    return np.clip(votes + delta_row[:len(delta_row) - 1], 0., None)


def seats_with(draws: Draws, delta: np.ndarray) -> np.ndarray:
    """
    Seats of every simulation with the votes of a scenario added.

    The threshold base of a province is its valid votes plus the whole row of `delta` (the residual included),
    before the votes are cut at zero. The total gets the sum of the provinces.

    Parameters
    ----------
    draws : Draws
        The simulations.
    delta : np.ndarray
        Regions × (parties + 1): votes of the scenario, the last column the residual. The row of the total is ignored.

    Returns
    -------
    np.ndarray
        Simulations × regions × parties, integer seats.
    """
    out = np.zeros(draws.votes.shape, dtype=np.int16)
    default = draws.regions.index(draws.default_region)
    for r in range(len(draws.regions)):
        if r == default:
            continue
        out[:, r, :] = allocate(_scenario_votes(draws.votes[:, r, :], delta[r]), int(draws.seats[r]),
                                float(draws.base[r] + delta[r].sum()), draws.threshold, draws.names)
    out[:, default, :] = out.sum(axis=1)

    return out


def central_with(draws: Draws, delta: np.ndarray) -> np.ndarray:
    """
    Seats of the central allocation of the model with the votes of a scenario added.

    Parameters
    ----------
    draws : Draws
        The simulations.
    delta : np.ndarray
        Regions × (parties + 1), as in `seats_with`.

    Returns
    -------
    np.ndarray
        Regions × parties, integer seats; the total is the sum of the provinces.
    """
    out = np.zeros(draws.central.shape, dtype=np.int16)
    default = draws.regions.index(draws.default_region)
    for r in range(len(draws.regions)):
        if r == default:
            continue
        out[r] = allocate(_scenario_votes(draws.central[r][None, :], delta[r]), int(draws.seats[r]),
                          float(draws.base[r] + delta[r].sum()), draws.threshold, draws.names)[0]
    out[default] = out.sum(axis=0)

    return out


def to_model(frame: pd.DataFrame, siglas: dict, names: list, regions: list, lineage: dict) -> np.ndarray:
    """
    Votes by province and candidature of 2023 as the `delta` of the model.

    A candidature whose sigla is `None` or, after `lineage`, is not a party of the model goes to the residual
    column. A region of `regions` the frame does not have gets zeros, and a row of the frame whose region is not in
    `regions` is ignored.

    Parameters
    ----------
    frame : pd.DataFrame
        Votes, indexed by `region_id`, one column per candidature.
    siglas : dict
        Sigla of 2023 of each candidature (column of `frame`); `None` if it has none.
    names : list
        Parties of the model.
    regions : list
        Ids of the regions, the total included.
    lineage : dict
        Party of the model that continues a sigla of 2023.

    Returns
    -------
    np.ndarray
        Regions × (parties + 1), the last column the residual.
    """
    out = np.zeros((len(regions), len(names) + 1))
    by_region = frame.astype(float).groupby(level=0).sum()
    for column in by_region.columns:
        sigla = siglas.get(column)
        party = lineage.get(sigla, sigla)
        j = names.index(party) if party in names else len(names)
        for r, region in enumerate(regions):
            if region in by_region.index:
                out[r, j] += by_region.at[region, column]

    return out


def applied_lineage(siglas: dict, names: list, lineage: dict) -> dict:
    """
    The lineage as `to_model` applies it.

    Parameters
    ----------
    siglas : dict
        Sigla of 2023 of each candidature.
    names : list
        Parties of the model.
    lineage : dict
        Party of the model that continues a sigla of 2023.

    Returns
    -------
    dict
        Sigla → party of the model, or `'-'` (the residual) if it has none; the candidatures without sigla left out.
    """
    out = {}
    for sigla in siglas.values():
        if sigla is None:
            continue
        party = lineage.get(sigla, sigla)
        out[sigla] = party if party in names else REST

    return out


def _block_seats(seats: np.ndarray, names: list, default: int, parties: list) -> np.ndarray:
    """Seats of a block in the total: one per simulation for a 3-D `seats`, a scalar for a 2-D one."""
    cols = [names.index(p) for p in parties if p in names]
    return seats[..., default, cols].sum(axis=-1) if seats.ndim == 3 else seats[default, cols].sum()


def _distribution(change: np.ndarray) -> dict:
    """Count of each value of `change`, with the values as strings of integers in ascending numeric order."""
    values, counts = np.unique(np.asarray(change, dtype=int), return_counts=True)

    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def measure(draws: Draws, delta: np.ndarray, blocks: dict, majority: int) -> dict:
    """
    What a scenario changes, simulation by simulation.

    Parameters
    ----------
    draws : Draws
        The simulations.
    delta : np.ndarray
        Regions × (parties + 1), votes of the scenario.
    blocks : dict
        Parties of each block, under `RIGHT` and `LEFT`.
    majority : int
        Seats of the absolute majority.

    Returns
    -------
    dict
        `votes` (net votes of the scenario), `lost` / `gained` (simulations in which the right loses / gets the
        majority), `moved` (simulations with some seat changed), `right` / `left` (distribution of the change of
        seats of each block) and `central` (the change in the central allocation).
    """
    default = draws.regions.index(draws.default_region)
    res, central = seats_with(draws, delta), central_with(draws, delta)
    right0 = _block_seats(draws.results, draws.names, default, blocks[RIGHT])
    left0 = _block_seats(draws.results, draws.names, default, blocks[LEFT])
    right = _block_seats(res, draws.names, default, blocks[RIGHT])
    left = _block_seats(res, draws.names, default, blocks[LEFT])
    rows = [r for r in range(len(draws.regions)) if r != default]

    return {
        'votes': int(round(float(delta[rows].sum()))),
        'lost': int(((right0 >= majority) & (right < majority)).sum()),
        'gained': int(((right0 < majority) & (right >= majority)).sum()),
        'moved': int((res != draws.results).any(axis=(1, 2)).sum()),
        'right': _distribution(right - right0),
        'left': _distribution(left - left0),
        'central': {
            'right': int(_block_seats(central, draws.names, default, blocks[RIGHT])
                         - _block_seats(draws.central_seats, draws.names, default, blocks[RIGHT])),
            'left': int(_block_seats(central, draws.names, default, blocks[LEFT])
                        - _block_seats(draws.central_seats, draws.names, default, blocks[LEFT])),
        }}


def province_counts(draws: Draws, delta: np.ndarray, blocks: dict) -> list:
    """
    What a scenario changes in each province, in simulations.

    Parameters
    ----------
    draws : Draws
        The simulations.
    delta : np.ndarray
        Regions × (parties + 1), votes of the scenario.
    blocks : dict
        Parties of each block, under `RIGHT`.

    Returns
    -------
    list
        `[{region_id, moved, right_lost, right_gained}]`, the total left out: simulations in which some seat of the
        province changes and in which the right block gets fewer / more seats in it.
    """
    res = seats_with(draws, delta)
    cols = [draws.names.index(p) for p in blocks[RIGHT] if p in draws.names]
    default = draws.regions.index(draws.default_region)
    out = []
    for r, region in enumerate(draws.regions):
        if r == default:
            continue
        change = res[:, r, cols].sum(axis=1) - draws.results[:, r, cols].sum(axis=1)
        out.append({'region_id': int(region), 'moved': int((res[:, r, :] != draws.results[:, r, :]).any(axis=1).sum()),
                    'right_lost': int((change < 0).sum()), 'right_gained': int((change > 0).sum())})

    return out


def base_counts(draws: Draws, blocks: dict, majority: int) -> dict:
    """
    The simulations as they are, without a scenario.

    Parameters
    ----------
    draws : Draws
        The simulations.
    blocks : dict
        Parties of each block, under `RIGHT` and `LEFT`.
    majority : int
        Seats of the absolute majority.

    Returns
    -------
    dict
        `with_majority` (simulations in which the right has the majority), `central` (seats of each block in the
        central allocation) and `right_seats` (distribution of the seats of the right).
    """
    default = draws.regions.index(draws.default_region)
    right = _block_seats(draws.results, draws.names, default, blocks[RIGHT])

    return {
        'with_majority': int((right >= majority).sum()),
        'central': {RIGHT: int(_block_seats(draws.central_seats, draws.names, default, blocks[RIGHT])),
                    LEFT: int(_block_seats(draws.central_seats, draws.names, default, blocks[LEFT]))},
        'right_seats': _distribution(right)}


def reproduction_problems(draws: Draws) -> list:
    """
    Whether a zero scenario gives back the stored seats.

    Parameters
    ----------
    draws : Draws
        The simulations.

    Returns
    -------
    list
        At most two texts: one with the number of simulations that differ and one for the central allocation; empty
        when `results` and `central_seats` are reproduced.
    """
    zero = np.zeros((len(draws.regions), len(draws.names) + 1))
    out = []
    differ = int((seats_with(draws, zero) != draws.results).any(axis=(1, 2)).sum())
    if differ:
        out.append('{} of {} simulations differ from the stored seats'.format(differ, len(draws.results)))
    if not np.array_equal(central_with(draws, zero), draws.central_seats):
        out.append('the central allocation differs from the stored seats')

    return out


def seat_range(dist: dict, alpha: float = 0.05) -> list:
    """
    Interval of a stored distribution of changes of seats.

    Parameters
    ----------
    dist : dict
        Count of each change, as `{'-1': n, '0': m}`.
    alpha : float, default 0.05
        The interval leaves `alpha / 2` of the simulations on each side.

    Returns
    -------
    list
        `[lo, hi]`, the quantiles `alpha / 2` and `1 - alpha / 2` of the distribution expanded (`np.quantile`, linear:
        the count of `Simulator.quantile`), as floats.
    """
    values = np.repeat([int(k) for k in dist], [int(n) for n in dist.values()])

    return [float(q) for q in np.quantile(values, [alpha / 2, 1 - alpha / 2])]
