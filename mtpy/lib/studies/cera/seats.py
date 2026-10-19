"""
The Congress seats of the CERA study: the D'Hondt allocation with every candidature, its validation against the
official seats, the last seat of each province and the CERA participation history.

Every function takes the `data.Results` of an election (and, where needed, other votes in its shape) and returns
plain pandas objects or dictionaries. The allocation is the one of the LOREG (art. 163): D'Hondt over the
candidatures with at least `THRESHOLD` percent of the valid votes of the province (votes to candidatures plus
blank votes).
"""
import math

import pandas as pd

from ...simulator import Simulator
from .data import Results

THRESHOLD = 3.0
LAST_SEAT_COLUMNS = ['region_id', 'name', 'magnitude', 'winner', 'runner_up', 'gap']
OTHER_BLOCK = 'Otros'


def _valid_votes(results: Results, votes: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Votes by candidature and the base of the threshold of each province.

    Parameters
    ----------
    results : Results
        The results of the election.
    votes : pd.DataFrame or None
        Votes replacing `results.votes` (`region_id` × candidature), or `None` for the official ones.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The votes and the valid votes by `region_id`: `candidates + blank` of `results.totals` for the official
        votes, the row sum of `votes` plus the province's `blank` otherwise.
    """
    if votes is None:
        return results.votes, results.totals['candidates'] + results.totals['blank']

    return votes, votes.sum(axis=1) + results.totals['blank'].reindex(votes.index)


def allocate_results(results: Results, votes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Allocate the seats of every province by D'Hondt with the legal threshold.

    Parameters
    ----------
    results : Results
        The results of the election; `totals['n_seats']` gives the magnitude of each province.
    votes : pd.DataFrame, optional
        Votes replacing `results.votes` (`region_id` × candidature). The threshold base is then recomputed as
        their row sum plus the province's blank votes.

    Returns
    -------
    pd.DataFrame
        Seats with `region_id` as index and the columns of the votes (0 for the candidatures without seat).
    """
    votes, valid = _valid_votes(results, votes)
    rows = {
        region: Simulator.alloc_seats(
            row.to_dict(), int(results.totals.at[region, 'n_seats']), valid_votes=float(valid.at[region]),
            threshold=THRESHOLD
        )
        for region, row in votes.iterrows()
    }

    seats = pd.DataFrame.from_dict(rows, orient='index')[list(votes.columns)]

    return seats.rename_axis(votes.index.name).astype('int64')


def check_official(results: Results) -> list[str]:
    """
    Compare the computed allocation with the official seats.

    Parameters
    ----------
    results : Results
        The results of the election, with the official `seats`.

    Returns
    -------
    list[str]
        One `'{province}: {candidature} {computed} != {official}'` line per mismatching cell; empty when the
        allocation reproduces every official seat.
    """
    computed = allocate_results(results)
    official = results.seats.reindex(index=computed.index, columns=computed.columns, fill_value=0)
    diff = computed.ne(official)

    return [
        f'{results.names.at[region]}: {candidature} {computed.at[region, candidature]} != '
        f'{official.at[region, candidature]}'
        for region, candidature in diff.stack().loc[lambda s: s].index
    ]


def last_seat(results: Results, votes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Find the last seat of every province and the votes its closest contender lacks to take it.

    The winner holds the smallest allocated D'Hondt quotient (`votes / seats`), `q_min`. The runner-up is the
    candidature, other than the winner, with the largest next quotient (`votes / (seats + 1)`) among those that
    pass the threshold. The gap is the fewest votes it has to add to strictly exceed `q_min` with its next
    quotient: `floor(q_min · (s_r + 1) − v_r) + 1`, at least 1, with `s_r` the runner-up's seats and `v_r` its
    votes. Reaching `q_min` exactly is not always enough (the LOREG gives that tie to the larger vote total and
    an equal total to a draw); the strict excess takes the seat whatever the totals. The article's table 10
    rounds `q_min · (s_r + 1) − v_r` instead (the votes to equal the quotient), so its figures are one less
    wherever that deficit has a fraction of one half or less.

    Parameters
    ----------
    results : Results
        The results of the election.
    votes : pd.DataFrame, optional
        Votes replacing `results.votes`, allocated as in `allocate_results`.

    Returns
    -------
    pd.DataFrame
        Columns `region_id`, `name`, `magnitude`, `winner`, `runner_up` and `gap`, one row per province. The
        runner-up and the gap are `None` where no other candidature passes the threshold.
    """
    votes, valid = _valid_votes(results, votes)
    seats = allocate_results(results, votes)
    rows = []

    for region, row in votes.iterrows():
        won = seats.loc[region]
        holders = won[won > 0].index
        quotients = row[holders] / won[holders]
        winner = quotients.idxmin()

        passing = row[(100. * row / valid.at[region] >= THRESHOLD) & (row > 0)].drop(winner, errors='ignore')
        runner_up, gap = None, None
        if len(passing) > 0:
            runner_up = (passing / (won[passing.index] + 1)).idxmax()
            needed = math.floor(row[winner] * (won[runner_up] + 1) / won[winner] - row[runner_up]) + 1
            gap = int(max(needed, 1))

        rows.append([
            region, results.names.at[region], int(results.totals.at[region, 'n_seats']), winner, runner_up, gap
        ])

    return pd.DataFrame(rows, columns=LAST_SEAT_COLUMNS)


def block_seats(seats: pd.DataFrame, siglas: dict, blocks: dict) -> dict[str, int]:
    """
    Add up the seats of each block.

    Parameters
    ----------
    seats : pd.DataFrame
        Seats with one column per candidature full name.
    siglas : dict
        Candidature full name to its sigla (`None` for the others).
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    dict[str, int]
        Seats of every block in `blocks` and of `'Otros'`, which takes the candidatures without a sigla or
        whose sigla is in no block.
    """
    block_of = {sigla: block for block, members in blocks.items() for sigla in members}
    totals = {block: 0 for block in [*blocks, OTHER_BLOCK]}

    for candidature, count in seats.sum().items():
        totals[block_of.get(siglas.get(candidature), OTHER_BLOCK)] += int(count)

    return totals


def participation_history(results: dict[str, Results]) -> pd.DataFrame:
    """
    Summarise the CERA participation of each province over several elections.

    Parameters
    ----------
    results : dict[str, Results]
        Election date to its results. Elections whose `totals` lack `registered_cera` or `counted_cera` are
        skipped.

    Returns
    -------
    pd.DataFrame
        Columns `region_id`, `mean` and `max` of `100 · counted_cera / registered_cera` (percent).
    """
    rates = [
        100. * r.totals['counted_cera'] / r.totals['registered_cera']
        for r in results.values() if {'registered_cera', 'counted_cera'} <= set(r.totals.columns)
    ]
    table = pd.concat(rates, axis=1)

    return pd.DataFrame({'mean': table.mean(axis=1), 'max': table.max(axis=1)}).rename_axis('region_id').reset_index()
