"""
The check with and without the CERA vote: which seats the vote of the Spaniards resident abroad decided.

The official allocation (with the CERA votes) is compared with the allocation of the peninsular results (without
them) province by province, and the CERA voters of each province with the gap of its last seat. The verdict rule
then judges a prediction of the seats the new CERA electors would move against the largest gain measured.
"""
from typing import Optional

import pandas as pd

from .data import Results
from .majority import LEFT, RIGHT
from .seats import OTHER_BLOCK, _sigla_seats, allocate_results, block_seats, last_seat
from .stress import peninsular_results

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


def majority_outcome(results: Results, cera: pd.DataFrame, blocks: dict, majority: int,
                     aliases: Optional[dict] = None) -> dict:
    """
    The seats of the two blocks of the majority, with and without the exterior vote, from the official results.

    A party of the model stands for the official siglas equal to its name or mapped to it by `aliases`; their seats
    add up. Only the blocks `RIGHT` and `LEFT` are read: the parties of any other block are never looked for.

    Parameters
    ----------
    results : Results
        The official results of the election, with the CERA votes.
    cera : pd.DataFrame
        The CERA votes by `region_id` and candidature full name.
    blocks : dict
        Block name to the list of its parties of the model (the snapshot's `blocks`).
    majority : int
        The seats of the absolute majority.
    aliases : dict, optional
        Official sigla to the party of the model it stands for; empty if `None`.

    Returns
    -------
    dict
        `right`, `left` (official seats of the blocks), `right_without`, `left_without` (seats of the peninsular
        allocation), `has_majority` (`right >= majority`) and `majority_without` (the same without the exterior
        vote), `changed` (they differ), `moved` (seats that change hands: the positive differences by sigla over
        all the siglas of the results, added) and `net` (`{RIGHT: right - right_without, LEFT: left -
        left_without}`).

    Raises
    ------
    ValueError
        If a party of the right or of the left block has no sigla in the results.
    """
    aliases = aliases or {}
    seats = _sigla_seats(allocate_results(results), results.siglas)
    seats_without = _sigla_seats(allocate_results(peninsular_results(results, cera)), results.siglas)

    for name in (RIGHT, LEFT):
        for party in blocks[name]:
            if not any(aliases.get(sigla, sigla) == party for sigla in seats):
                raise ValueError(
                    f'voto-cera: no candidature of the results stands for {party}: add its sigla to '
                    f'majority.aliases in data/cera/params.json'
                )

    def total(sigla_seats: dict, name: str) -> int:
        """Add up the seats of the siglas that stand for the parties of a block."""
        return sum(count for sigla, count in sigla_seats.items() if aliases.get(sigla, sigla) in blocks[name])

    right, left = total(seats, RIGHT), total(seats, LEFT)
    right_without, left_without = total(seats_without, RIGHT), total(seats_without, LEFT)
    has_majority, majority_without = right >= majority, right_without >= majority

    return {
        'right': right,
        'left': left,
        'right_without': right_without,
        'left_without': left_without,
        'has_majority': has_majority,
        'majority_without': majority_without,
        'changed': has_majority != majority_without,
        'moved': sum(max(count - seats_without.get(sigla, 0), 0) for sigla, count in seats.items()),
        'net': {RIGHT: right - right_without, LEFT: left - left_without}
    }
