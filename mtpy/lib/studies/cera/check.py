"""
The check with and without the CERA vote: which seats the vote of the Spaniards resident abroad decided.

The official allocation (with the CERA votes) is compared with the allocation of the peninsular results (without
them) province by province, and the CERA voters of each province with the gap of its last seat. The verdict rule
then judges a prediction of the seats the new CERA electors would move against the largest gain measured.
"""
import pandas as pd

from .data import Results
from .seats import OTHER_BLOCK, _sigla_seats, allocate_results, block_seats, last_seat, peninsular_results

ROW_COLUMNS = [
    'region_id', 'name', 'magnitude', 'winner_with', 'winner_without', 'gap', 'voters_cera', 'over_gap', 'decided',
    'crosses_block'
]
HIT = 'acierto'
MISS = 'fallo'


def _block_of(candidature: str | None, siglas: dict, blocks: dict) -> str:
    """
    The block of a candidature.

    Parameters
    ----------
    candidature : str or None
        The candidature full name.
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    str
        The block whose siglas hold the candidature's sigla, `'Otros'` when it has no sigla or the sigla is in no
        block.
    """
    sigla = siglas.get(candidature)
    for block, members in blocks.items():
        if sigla in members:
            return block

    return OTHER_BLOCK


def with_without(results: Results, cera: pd.DataFrame, siglas: dict, blocks: dict,
                 voters: pd.Series | None = None) -> dict:
    """
    Compare the seats of an election with and without its CERA votes.

    The seats with the CERA vote are the allocation of `results` (the official seats); those without it are the
    allocation of `peninsular_results(results, cera)`. A province is decided by the CERA vote when any of its seats
    differs between the two, and its CERA voters are over the gap when they outnumber the votes the runner-up of
    its official last seat lacked (`last_seat`).

    Parameters
    ----------
    results : Results
        The results of the election, with the CERA votes.
    cera : pd.DataFrame
        The CERA votes by `region_id` and candidature full name.
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.
    voters : pd.Series, optional
        The CERA voters by `region_id` (the `voters` of the scrutiny, blank and invalid votes included); the row
        sum of `cera` if `None`. Provinces missing from it count with 0.

    Returns
    -------
    dict
        `rows` (one row per province: `region_id`, `name`, `magnitude`, `winner_with` and `winner_without` (the
        last-seat winners of the official and of the peninsular allocations), `gap` (official), `voters_cera`,
        `over_gap` (`voters_cera > gap`), `decided` (any seat differs) and `crosses_block` (decided and the two
        winners are in different blocks, `'Otros'` for the candidatures in none)), `net_by_sigla` and
        `net_by_block` (`{name: seats with − seats without}`, the candidatures without a sigla under `'Otros'`, the
        blocks with `'Otros'`), `n_over_gap`, `n_decided` and `decided` (the sorted `region_id`s decided).
    """
    seats_with = allocate_results(results)
    without = peninsular_results(results, cera)
    seats_without = allocate_results(without)
    last = last_seat(results)
    last_without = last_seat(without).set_index('region_id')

    if voters is None:
        voters = cera.sum(axis=1)
    changed = seats_with.ne(seats_without.reindex_like(seats_with)).any(axis=1)

    rows = last[['region_id', 'name', 'magnitude']].copy()
    rows['winner_with'] = last['winner']
    rows['winner_without'] = rows['region_id'].map(last_without['winner'])
    rows['gap'] = last['gap']
    rows['voters_cera'] = rows['region_id'].map(voters).fillna(0).astype('int64')
    rows['over_gap'] = rows['voters_cera'] > pd.to_numeric(rows['gap'], errors='coerce')
    rows['decided'] = changed.reindex(rows['region_id'], fill_value=False).to_numpy(dtype=bool)
    block_with = rows['winner_with'].map(lambda candidature: _block_of(candidature, siglas, blocks))
    block_without = rows['winner_without'].map(lambda candidature: _block_of(candidature, siglas, blocks))
    rows['crosses_block'] = rows['decided'] & block_with.ne(block_without)

    sigla_with, sigla_without = _sigla_seats(seats_with, siglas), _sigla_seats(seats_without, siglas)
    block_with_seats = block_seats(seats_with, siglas, blocks)
    block_without_seats = block_seats(seats_without, siglas, blocks)

    return {
        'rows': rows[ROW_COLUMNS],
        'net_by_sigla': {sigla: sigla_with[sigla] - sigla_without[sigla] for sigla in sigla_with},
        'net_by_block': {block: block_with_seats[block] - block_without_seats[block] for block in block_with_seats},
        'n_over_gap': int(rows['over_gap'].sum()),
        'n_decided': int(rows['decided'].sum()),
        'decided': sorted(int(region) for region in rows.loc[rows['decided'], 'region_id'])
    }


def verdict(net_psoe: int, net_left: int, max_gain: int) -> str:
    """
    Judge a prediction of the seats the CERA vote moves against the largest gain measured.

    Parameters
    ----------
    net_psoe : int
        The seats the PSOE is predicted to gain.
    net_left : int
        The seats the left block is predicted to gain.
    max_gain : int
        The largest gain the measurement allows.

    Returns
    -------
    str
        `'acierto'` when neither prediction exceeds `max_gain`, `'fallo'` otherwise.
    """
    return HIT if max(net_psoe, net_left) <= max_gain else MISS
