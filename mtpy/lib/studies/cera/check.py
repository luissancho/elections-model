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
# the target of `majority.aliases` for a candidature the model does not simulate: its seats go to the rest
REST_ALIAS = '-'


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


def _moved_seats(seats_with: pd.DataFrame, seats_without: pd.DataFrame) -> int:
    """
    The seats that change hands between two allocations, province by province.

    Parameters
    ----------
    seats_with : pd.DataFrame
        Seats with the CERA vote (`allocate_results`), `region_id` × candidature.
    seats_without : pd.DataFrame
        Seats without it, on the same provinces and candidatures.

    Returns
    -------
    int
        The positive differences by province and candidature, added: two seats that offset in the national total
        (one won in a province, one lost in another) both count, as in the provinces the check lists as decided.
    """
    return int((seats_with - seats_without.reindex_like(seats_with)).clip(lower=0).sum().sum())


def _outcome_errors(seats: dict, seats_without: dict, blocks: dict, aliases: dict) -> list[str]:
    """
    What keeps the official siglas from being counted by the parties of the model.

    Parameters
    ----------
    seats, seats_without : dict
        Seats by official sigla with and without the exterior vote (`_sigla_seats`).
    blocks : dict
        Block name to the list of its parties of the model; every block names parties of the model.
    aliases : dict
        Official sigla to the party of the model it stands for, or `REST_ALIAS`.

    Returns
    -------
    list[str]
        One sentence per kind of error, empty when every seat can be counted: the aliases to no party of the model,
        the seat-winning siglas that stand for no party of the model (with their seats) and the parties of `RIGHT`
        and `LEFT`, in that order, for which no sigla of the results stands.
    """
    parties = {party for members in blocks.values() for party in members}
    errors = []

    wrong = [f'{sigla} → {target}' for sigla, target in aliases.items()
             if target not in parties and target != REST_ALIAS]
    if wrong:
        errors.append(f'the aliases {", ".join(wrong)} point to no party of the model nor to "{REST_ALIAS}"')

    unmapped = [
        f'{sigla} ({seats.get(sigla, 0)})' if seats.get(sigla, 0) == seats_without.get(sigla, 0)
        else f'{sigla} ({seats.get(sigla, 0)}; {seats_without.get(sigla, 0)} without the exterior vote)'
        for sigla in sorted(set(seats) | set(seats_without), key=lambda s: (-seats.get(s, 0), s))
        if (seats.get(sigla, 0) or seats_without.get(sigla, 0))
        and aliases.get(sigla, sigla) not in parties and aliases.get(sigla, sigla) != REST_ALIAS
    ]
    if unmapped:
        errors.append(
            f'the siglas {", ".join(unmapped)} win seats and stand for no party of the model: map each to a party of '
            f'the model or to "{REST_ALIAS}" in majority.aliases of data/cera/params.json'
        )

    stands = {aliases.get(sigla, sigla) for sigla in seats}
    missing = [party for name in (RIGHT, LEFT) for party in blocks[name] if party not in stands]
    if missing:
        errors.append(
            f'no candidature of the results stands for {", ".join(missing)}: add its sigla to majority.aliases in '
            f'data/cera/params.json'
        )

    return errors


def majority_outcome(results: Results, cera: pd.DataFrame, blocks: dict, majority: int,
                     aliases: Optional[dict] = None) -> dict:
    """
    The seats of the two blocks of the majority, with and without the exterior vote, from the official results.

    The parties of the model are those of every block of `blocks`. An official sigla stands for `aliases.get(sigla,
    sigla)`: a party of the model or `REST_ALIAS` (`"-"`, a candidature the model does not simulate); the seats of
    several siglas that stand for one party add up. Every seat must be accounted for: an alias to anything else, a
    seat-winning sigla (with or without the exterior vote) that stands for nothing of the two, or a party of `RIGHT`
    or `LEFT` for which no sigla stands is refused, all of them in one message. Only the seats of `RIGHT` and `LEFT`
    are counted, and a party of another block is never required in the results.

    Parameters
    ----------
    results : Results
        The official results of the election, with the CERA votes.
    cera : pd.DataFrame
        The CERA votes by `region_id` and candidature full name.
    blocks : dict
        Block name to the list of its parties of the model (the snapshot's `blocks`, every key).
    majority : int
        The seats of the absolute majority.
    aliases : dict, optional
        Official sigla to the party of the model it stands for, or `"-"`; empty if `None`.

    Returns
    -------
    dict
        `right`, `left` (official seats of the blocks), `right_without`, `left_without` (seats of the peninsular
        allocation), `has_majority` (`right >= majority`) and `majority_without` (the same without the exterior
        vote), `changed` (they differ), `moved` (seats the exterior vote decided: the positive differences by
        province and candidature between the two allocations, added, `_moved_seats`) and `net` (`{RIGHT: right -
        right_without, LEFT: left - left_without}`).

    Raises
    ------
    ValueError
        If an official sigla cannot be counted (`_outcome_errors`), or if the right, the left and the rest do not add
        up to the seats of the chamber, with or without the exterior vote.
    """
    aliases = aliases or {}
    allocation = allocate_results(results)
    allocation_without = allocate_results(peninsular_results(results, cera))
    seats = _sigla_seats(allocation, results.siglas)
    seats_without = _sigla_seats(allocation_without, results.siglas)

    errors = _outcome_errors(seats, seats_without, blocks, aliases)
    if errors:
        raise ValueError('voto-cera: the seats of the results cannot be counted by the parties of the model: '
                         + '; '.join(errors))

    def total(sigla_seats: dict, members: list) -> int:
        """Add up the seats of the siglas that stand for some of `members`."""
        return sum(count for sigla, count in sigla_seats.items() if aliases.get(sigla, sigla) in members)

    in_blocks = set(blocks[RIGHT]) | set(blocks[LEFT])
    rest_of = [party for members in blocks.values() for party in members if party not in in_blocks] + [REST_ALIAS]
    chamber = int(results.totals['n_seats'].sum())
    for sigla_seats in (seats, seats_without):
        counted = total(sigla_seats, blocks[RIGHT]) + total(sigla_seats, blocks[LEFT]) + total(sigla_seats, rest_of)
        if counted != chamber:
            raise ValueError(f'voto-cera: the right, the left and the rest add up to {counted} seats and do not add '
                             f'up to the {chamber} seats of the chamber: never publish this count')

    right, left = total(seats, blocks[RIGHT]), total(seats, blocks[LEFT])
    right_without, left_without = total(seats_without, blocks[RIGHT]), total(seats_without, blocks[LEFT])
    has_majority, majority_without = right >= majority, right_without >= majority

    return {
        'right': right,
        'left': left,
        'right_without': right_without,
        'left_without': left_without,
        'has_majority': has_majority,
        'majority_without': majority_without,
        'changed': has_majority != majority_without,
        'moved': _moved_seats(allocation, allocation_without),
        'net': {RIGHT: right - right_without, LEFT: left - left_without}
    }
