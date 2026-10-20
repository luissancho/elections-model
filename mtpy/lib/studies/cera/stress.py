"""
The stress tests of the CERA study, built on the allocation and the increment model of `seats`.

The results without the CERA votes (`peninsular_results`), the whole exterior vote at several participations and
left shares (`exterior_grid`), the increment model over census, participation and bias (`scenario_table`), the
optimal placement of a pool of new electors (`optimal_placement`, an upper bound) and the replication of the report
of the «23 seats» (`report_replication`).

Every function takes the `data.Results` of an election (and, where needed, other votes in its shape) and returns
plain pandas objects or dictionaries.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .data import Results
from .seats import (
    LATAM,
    REST,
    THRESHOLD,
    _allocate_row,
    allocate_results,
    block_seats,
    empirical_split,
    increment_model,
    increment_votes,
    last_seat,
    new_electors,
    projected_census
)

LEFT = 'Izquierda'
RIGHT = 'Derecha'
EMPIRICAL = 'empirico'
VALID_SHARE = 0.99
CONTRAST_RATE = 0.05


def peninsular_results(results: Results, cera: pd.DataFrame) -> Results:
    """
    The results without the CERA votes: what the electors resident in Spain voted.

    Parameters
    ----------
    results : Results
        The results of the election.
    cera : pd.DataFrame
        CERA votes by `region_id` and candidature full name; missing provinces or candidatures count as 0 and
        candidatures absent from `results.votes` are ignored.

    Returns
    -------
    Results
        A copy whose `votes` are `results.votes − cera`, floored at 0 cell by cell, and whose `totals` have
        `candidates` reduced by the votes actually removed in each province and `valid` recomputed as
        `candidates + blank`. `seats`, `names`, `siglas` and `date` are those of `results`.
    """
    cera = cera.reindex(index=results.votes.index, columns=results.votes.columns, fill_value=0)
    votes = (results.votes - cera).clip(lower=0)
    totals = results.totals.copy()
    totals['candidates'] = totals['candidates'] - (results.votes - votes).sum(axis=1)
    totals['valid'] = totals['candidates'] + totals['blank']

    return replace(results, votes=votes, totals=totals)


def _block_columns(columns: pd.Index, siglas: dict, blocks: dict, block: str) -> list[str]:
    """
    The candidatures of a block.

    Parameters
    ----------
    columns : pd.Index
        Candidature full names.
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.
    block : str
        The block.

    Returns
    -------
    list[str]
        The candidatures of `columns` whose sigla is in `blocks[block]`, in their order.
    """
    members = set(blocks.get(block, []))

    return [candidature for candidature in columns if siglas.get(candidature) in members]


def _biased_split(weights: pd.DataFrame, left_frac: float, siglas: dict, blocks: dict) -> pd.DataFrame:
    """
    Re-split the vote shares of each province between the left and the right blocks.

    The candidatures outside `Izquierda` and `Derecha` keep their share of the province (`weights` normalised by
    row). The combined share of the two blocks goes `left_frac` to `Izquierda` and `1 − left_frac` to `Derecha`;
    within each block, in proportion to the `weights` of its candidatures in the province, or, where the block
    has no weight in the province, to the block's weights added up over every province.

    Parameters
    ----------
    weights : pd.DataFrame
        Votes or shares by `region_id` and candidature.
    left_frac : float
        The fraction (0-1) of the two blocks' share that goes to `Izquierda`.
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    pd.DataFrame
        Shares with the index and columns of `weights`, each row adding up to 1 (0 where the row of `weights`
        has no weight).
    """
    shares = weights.div(weights.sum(axis=1), axis=0).fillna(0.)
    left = _block_columns(weights.columns, siglas, blocks, LEFT)
    right = _block_columns(weights.columns, siglas, blocks, RIGHT)
    both = shares[left + right].sum(axis=1)
    biased = shares.copy()

    for columns, frac in ((left, left_frac), (right, 1. - left_frac)):
        block = weights[columns]
        within = block.div(block.sum(axis=1), axis=0)
        national = block.sum() / block.values.sum()
        within.loc[block.sum(axis=1) <= 0] = national.values
        biased[columns] = within.mul(both * frac, axis=0)

    return biased


def exterior_grid(results: Results, cera_2023: pd.DataFrame, census_by_province: pd.Series,
                  participations: list[float], left_fracs: list[float], siglas, blocks) -> pd.DataFrame:
    """
    The change of the left block's seats with the whole exterior vote at several participations and left shares.

    The article's table 11. The CERA votes of 2023 are taken out of the results (`peninsular_results`) and the
    whole exterior vote is added back: `census_p · participation · 0.99` votes in each province, split as
    `_biased_split` does with the 2023 CERA votes of the province as weights (separatists, regionalists and the
    others keep their 2023 CERA share; the rest goes `left_frac` to `Izquierda` and `1 − left_frac` to
    `Derecha`).

    Parameters
    ----------
    results : Results
        The results of the election (2023).
    cera_2023 : pd.DataFrame
        The CERA votes of the election by `region_id` and candidature.
    census_by_province : pd.Series
        The CERA electors by `region_id`.
    participations : list[float]
        Participation rates (0-1), the rows.
    left_fracs : list[float]
        Fractions (0-1) of the left + right vote that go to the left, the columns.
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    pd.DataFrame
        Seats of `Izquierda` minus its official seats, with the participations as index and the fractions as
        columns (integers).
    """
    base = peninsular_results(results, cera_2023)
    weights = cera_2023.reindex(index=results.votes.index, columns=results.votes.columns, fill_value=0)
    electors = census_by_province.reindex(results.votes.index, fill_value=0)
    official = block_seats(results.seats, siglas, blocks)[LEFT]
    grid = pd.DataFrame(0, index=pd.Index(participations, name='participation'),
                        columns=pd.Index(left_fracs, name='left_frac'), dtype='int64')

    for frac in left_fracs:
        shares = _biased_split(weights, frac, siglas, blocks)
        for participation in participations:
            votes = base.votes + shares.mul(electors * participation * VALID_SHARE, axis=0)
            grid.loc[participation, frac] = block_seats(allocate_results(base, votes), siglas, blocks)[LEFT] - official

    return grid


def scenario_table(results: Results, baseline: pd.DataFrame, census: pd.DataFrame, countries: pd.DataFrame,
                   participation: dict, split: pd.DataFrame, siglas: dict, blocks: dict,
                   scenarios: list[dict]) -> pd.DataFrame:
    """
    The increment model over several scenarios of census, participation and bias (the article's table 9).

    Parameters
    ----------
    results : Results
        The results the new votes are added to (2023).
    baseline : pd.DataFrame
        The census of the election, with `region_id`, `country_code` and `electors`.
    census : pd.DataFrame
        The real new census, same columns.
    countries : pd.DataFrame
        The countries table, with `code` and `region`.
    participation : dict
        Participation rate (0-1) by origin, `'latam'` and `'rest'`, and optionally the `valid` fraction.
    split : pd.DataFrame
        The empirical split of the new votes by `region_id` and candidature (rows adding up to 1).
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.
    scenarios : list[dict]
        One dict per row with `census_x` (`None` for the real `census`, or a factor that multiplies every cell
        of `baseline`), `part_x` (a factor of both participation rates) and `bias` (`'empirico'` for `split`,
        or the fraction of the left + right share of each province that goes to the left, split within each
        block as `split`; see `_biased_split`).

    Returns
    -------
    pd.DataFrame
        Columns `census_x` (object, `None` for the real census), `part_x`, `bias`, `d_left` and `d_right` (the
        change of the seats of `Izquierda` and of `Derecha`), one row per scenario.
    """
    rows = []

    for scenario in scenarios:
        factor = scenario['census_x']
        new_census = census if factor is None else projected_census(baseline, countries, factor, factor)
        new = new_electors(new_census, baseline, countries)
        rates = {origin: participation[origin] * scenario['part_x'] for origin in (LATAM, REST)}
        bias = scenario['bias']
        shares = split if bias == EMPIRICAL else _biased_split(split, bias, siglas, blocks)
        extra = increment_votes(new, rates, shares, valid=participation.get('valid', VALID_SHARE))
        by_block = increment_model(results, extra, siglas, blocks)['by_block']
        rows.append([factor, scenario['part_x'], bias, by_block[LEFT]['change'], by_block[RIGHT]['change']])

    table = pd.DataFrame(rows, columns=['census_x', 'part_x', 'bias', 'd_left', 'd_right'])
    table['census_x'] = pd.Series([row[0] for row in rows], index=table.index, dtype='object')

    return table


def placement_pool(long: pd.DataFrame, census: pd.DataFrame, countries: pd.DataFrame, codes: list[str],
                   since: str) -> pd.Series:
    """
    The electors some countries gained since a date, by province: the pool of the optimal placement.

    Parameters
    ----------
    long : pd.DataFrame
        The monthly census by province and country, with `date`, `region_id`, `country_code` and `electors`.
    census : pd.DataFrame
        The convocation census, with `region_id`, `country_code` and `electors`.
    countries : pd.DataFrame
        The countries table, with `code` and `name`.
    codes : list[str]
        Country names as in `countries` (`ARGENTINA`, `MÉXICO`, `CUBA`, `VENEZUELA`).
    since : str
        The start date; the census of `long` at the last month on or before it is the reference.

    Returns
    -------
    pd.Series
        Electors of those countries in `census` minus those in `long` at the reference month, by `region_id`,
        clipped at 0.

    Raises
    ------
    ValueError
        If a name is not in `countries`.
    """
    code_of = countries.set_index('name')['code']
    unknown = [name for name in codes if name not in code_of.index]
    if unknown:
        raise ValueError(f'Unknown countries: {", ".join(unknown)}')

    wanted = code_of[codes].tolist()
    start = long.loc[long['date'] <= pd.Timestamp(since), 'date'].max()
    before = long[long['date'].eq(start) & long['country_code'].isin(wanted)].groupby('region_id')['electors'].sum()
    now = census[census['country_code'].isin(wanted)].groupby('region_id')['electors'].sum()

    return now.sub(before, fill_value=0).clip(lower=0).rename('electors')


def _seat_cost(row: pd.Series, won: pd.Series, blank: float, candidature: str) -> float:
    """
    The votes a candidature needs to win one more seat in a province.

    Its next quotient (`(votes + x) / (seats + 1)`) has to exceed the smallest allocated quotient held by another
    candidature, and it has to reach `THRESHOLD` percent of the valid votes, which grow with `x`.

    Parameters
    ----------
    row : pd.Series
        Votes of each candidature of the province.
    won : pd.Series
        Seats of each candidature of the province, as allocated with `row`.
    blank : float
        Blank votes of the province.
    candidature : str
        The candidature.

    Returns
    -------
    float
        The votes, at least 1; infinite when no other candidature holds a seat.
    """
    others = won.drop(candidature)
    holders = others[others > 0].index
    if len(holders) == 0:
        return np.inf

    lowest = (row[holders] / others[holders]).min()
    votes = row[candidature]
    quotient = np.floor(lowest * (won[candidature] + 1) - votes) + 1
    threshold = np.ceil((THRESHOLD / 100. * (row.sum() + blank) - votes) / (1. - THRESHOLD / 100.))

    return float(max(quotient, threshold, 1.))


def _greedy_seats(results: Results, seats: pd.DataFrame, costs: pd.Series, candidature: str, votes_left: float,
                  step: int) -> int:
    """
    Place votes of one candidature, `step` at a time, where its next seat is cheapest, and count its seats.

    Parameters
    ----------
    results : Results
        The results the votes are added to.
    seats : pd.DataFrame
        Their allocation, as `allocate_results` returns.
    costs : pd.Series
        The `_seat_cost` of the candidature in each province with those results.
    candidature : str
        The candidature the votes go to.
    votes_left : float
        The votes to place.
    step : int
        The votes placed at a time.

    Returns
    -------
    int
        The candidature's seats once every vote is placed.
    """
    votes, seats, costs = results.votes.astype('float64'), seats.copy(), costs.copy()
    blank, magnitude = results.totals['blank'], results.totals['n_seats']

    while votes_left > 0:
        region = costs.idxmin()
        placed = min(step, votes_left)
        votes.at[region, candidature] += placed
        votes_left -= placed
        row = votes.loc[region]
        won = pd.Series(_allocate_row(row, int(magnitude.at[region]), float(row.sum() + blank.at[region])))
        seats.loc[region] = won[seats.columns].values
        costs.at[region] = _seat_cost(row, seats.loc[region], float(blank.at[region]), candidature)

    return int(seats[candidature].sum())


def optimal_placement(results: Results, pool: pd.Series, participations: list[float], fracs: list[float],
                      candidature: str, step: int = 500, split: pd.DataFrame | None = None) -> dict:
    """
    The seats a candidature would win placing a pool of new electors where its seats are cheapest (an upper bound).

    The article's table 13 (14 of the spec). For each participation `p` and fraction `f`, the useful votes
    `pool.sum() · p · f · 0.99` go to `candidature`, `step` at a time, each time to the province where its next
    seat costs the fewest votes (`_seat_cost`, recomputed after every step), over the official results. The
    contrast places the pool in its real provinces instead, at the observed Latin American participation
    (`CONTRAST_RATE`) and with the empirical split, through `increment_votes` and `increment_model`.

    Parameters
    ----------
    results : Results
        The results of the election (2023).
    pool : pd.Series
        Electors to place by `region_id`, as `placement_pool` returns.
    participations : list[float]
        Participation rates (0-1), the rows of the grid.
    fracs : list[float]
        Fractions (0-1) of the votes that go to `candidature`, the columns.
    candidature : str
        The candidature full name.
    step : int, default 500
        The votes placed at a time.
    split : pd.DataFrame, optional
        The empirical split of the contrast by `region_id` and candidature (rows adding up to 1), the 2023 CERA
        one in the study; the province shares of `results.votes` if `None`.

    Returns
    -------
    dict
        `grid` (the seats `candidature` gains over the official allocation, participations × fractions,
        integers) and `contrast` (its change of seats without the placement, which should be 0).
    """
    seats = allocate_results(results)
    blank = results.totals['blank']
    costs = pd.Series({
        region: _seat_cost(row, seats.loc[region], float(blank.at[region]), candidature)
        for region, row in results.votes.iterrows()
    })
    official = int(seats[candidature].sum())
    grid = pd.DataFrame(0, index=pd.Index(participations, name='participation'),
                        columns=pd.Index(fracs, name='frac'), dtype='int64')

    for participation in participations:
        for frac in fracs:
            useful = float(pool.sum()) * participation * frac * VALID_SHARE
            grid.loc[participation, frac] = _greedy_seats(results, seats, costs, candidature, useful, step) - official

    if split is None:
        split = empirical_split(results.votes)
    new = pd.DataFrame({'region_id': pool.index, 'origin': LATAM, 'electors': pool.values})
    extra = increment_votes(new, {LATAM: CONTRAST_RATE, REST: CONTRAST_RATE}, split)
    after = increment_model(results, extra, results.siglas, {})['seats_after']

    return {'grid': grid, 'contrast': int(after[candidature].sum()) - official}


def report_replication(results: Results, census_by_province: pd.Series, hist: pd.DataFrame) -> dict:
    """
    Replicate the report of the «23 seats»: the CERA voters at the historical participation against the gaps.

    Parameters
    ----------
    results : Results
        The results whose last-seat gaps are compared (2023).
    census_by_province : pd.Series
        The CERA electors by `region_id`.
    hist : pd.DataFrame
        The historical CERA participation (percent), with `region_id`, `mean` and `max`, as
        `participation_history` returns; provinces missing from it count with 0 votes.

    Returns
    -------
    dict
        `votes_mean` and `votes_max` (`Σ census_p · rate_p / 100` with the mean and the maximum rates),
        `n_mean` and `n_max` (the provinces where those votes exceed the gap of `last_seat`) and
        `participation_mean` and `participation_max` (the participation those votes give over the whole census,
        `100 · votes / Σ census_p`).
    """
    last = last_seat(results).set_index('region_id')
    electors = census_by_province.reindex(last.index, fill_value=0)
    rates = hist.set_index('region_id')[['mean', 'max']].reindex(last.index, fill_value=0.)
    gap = pd.to_numeric(last['gap'], errors='coerce')
    out = {}

    for column in ('mean', 'max'):
        votes = electors * rates[column] / 100.
        out[f'votes_{column}'] = float(votes.sum())
        out[f'n_{column}'] = int((votes > gap).sum())
        out[f'participation_{column}'] = float(100. * votes.sum() / electors.sum())

    return out
