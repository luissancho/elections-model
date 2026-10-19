"""
The Congress seats of the CERA study: the D'Hondt allocation with every candidature, its validation against the
official seats, the last seat of each province and the CERA participation history.

It also holds the increment model: the electors the CERA census gains over the 2023 one (with the article's
projection or with a real census, and the scenario without the inscriptions in litigation), the votes they would
add with a participation by origin, and the seats of the 2023 results plus those votes.

The stress tests built on them are in `stress`.

Every function takes the `data.Results` of an election (and, where needed, other votes in its shape) and returns
plain pandas objects or dictionaries. The allocation is the one of the LOREG (art. 163): D'Hondt over the
candidatures with at least `THRESHOLD` percent of the valid votes of the province (votes to candidatures plus
blank votes).
"""
import numpy as np
import pandas as pd

from ...simulator import Simulator
from .data import Results

THRESHOLD = 3.0
LAST_SEAT_COLUMNS = ['region_id', 'name', 'magnitude', 'winner', 'runner_up', 'gap']
OTHER_BLOCK = 'Otros'
LATAM = 'latam'
REST = 'rest'


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


def _allocate_row(row: pd.Series, n_seats: int, valid: float) -> dict[str, int]:
    """
    Allocate the seats of one province by D'Hondt with the legal threshold, ties to the larger vote total.

    Parameters
    ----------
    row : pd.Series
        Votes of each candidature of the province.
    n_seats : int
        Seats of the province.
    valid : float
        Base of the threshold (votes to candidatures plus blank votes).

    Returns
    -------
    dict[str, int]
        Seats of each candidature, 0 for those without seat.
    """
    ordered = row.sort_values(ascending=False, kind='stable').to_dict()

    return Simulator.alloc_seats(ordered, n_seats, valid_votes=valid, threshold=THRESHOLD)


def allocate_results(results: Results, votes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Allocate the seats of every province by D'Hondt with the legal threshold.

    The candidatures go to `Simulator.alloc_seats` ordered by total votes, largest first, so an exact tie of
    quotients goes to the larger vote total, as the LOREG rules (art. 163.1.c).

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
        region: _allocate_row(row, int(results.totals.at[region, 'n_seats']), float(valid.at[region]))
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
    Find the last seat of every province and the votes its closest contender lacks to equal it.

    The winner holds the smallest allocated D'Hondt quotient (`votes / seats`), `q_min`. The runner-up is the
    candidature, other than the winner, with the largest next quotient (`votes / (seats + 1)`) among those that
    pass the threshold. The gap follows the article's table 10: the votes the runner-up needs for its next
    quotient to equal `q_min`, `round(q_min · (s_r + 1) − v_r)` (Python's rounding, half to even), at least 1,
    with `s_r` the runner-up's seats and `v_r` its votes. Equalling is enough in practice: the LOREG gives a tie
    of quotients to the larger vote total (art. 163.1.c), so equalling takes the seat whenever the runner-up's
    total is the larger one, and strictly exceeding the quotient never takes more than one vote more.

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
            needed = round(row[winner] * (won[runner_up] + 1) / won[winner] - row[runner_up])
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


def _origin(country_codes: pd.Series, countries: pd.DataFrame) -> pd.Series:
    """
    The origin of each country code: `'latam'` for the Latin American countries, `'rest'` for every other.

    Parameters
    ----------
    country_codes : pd.Series
        INE country codes.
    countries : pd.DataFrame
        The countries table, with `code` and `region`.

    Returns
    -------
    pd.Series
        `'latam'` or `'rest'`, aligned with `country_codes`.
    """
    region = country_codes.map(countries.set_index('code')['region'])

    return region.eq(LATAM).map({True: LATAM, False: REST})


def new_electors(census: pd.DataFrame, baseline: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """
    The electors a census gains over the baseline, by province and origin.

    Both censuses are joined on `region_id` and `country_code` (a cell present in only one of them counts with 0
    electors in the other) and their difference is added up by province and origin. Negative differences, the
    cells that lost electors, are kept.

    Parameters
    ----------
    census : pd.DataFrame
        The new census, with `region_id`, `country_code` and `electors`.
    baseline : pd.DataFrame
        The census it is compared with, same columns.
    countries : pd.DataFrame
        The countries table, with `code` and `region`.

    Returns
    -------
    pd.DataFrame
        Columns `region_id`, `origin` (`'latam'` or `'rest'`) and `electors` (the difference; integer when both
        censuses are, float otherwise, as with a `projected_census`).
    """
    keys = ['region_id', 'country_code']
    both = census[[*keys, 'electors']].merge(
        baseline[[*keys, 'electors']], on=keys, how='outer', suffixes=('_new', '_base')
    ).fillna({'electors_new': 0, 'electors_base': 0})
    both['origin'] = _origin(both['country_code'], countries)
    both['electors'] = both['electors_new'] - both['electors_base']
    if all(pd.api.types.is_integer_dtype(frame['electors']) for frame in (census, baseline)):
        both['electors'] = both['electors'].astype('int64')

    return both.groupby(['region_id', 'origin'], as_index=False)['electors'].sum()


def projected_census(baseline: pd.DataFrame, countries: pd.DataFrame, factor_latam: float,
                     factor_rest: float) -> pd.DataFrame:
    """
    Project a census by origin, as the article does (×1.264 the Latin American electors, ×1.194 the rest).

    Parameters
    ----------
    baseline : pd.DataFrame
        The census to project, with `region_id`, `country_code` and `electors`.
    countries : pd.DataFrame
        The countries table, with `code` and `region`.
    factor_latam : float
        Factor of the electors of the Latin American countries.
    factor_rest : float
        Factor of the electors of every other country.

    Returns
    -------
    pd.DataFrame
        A copy of `baseline` whose `electors` (float) are multiplied by the factor of their origin.
    """
    projected = baseline.copy()
    factor = _origin(projected['country_code'], countries).map({LATAM: factor_latam, REST: factor_rest})
    projected['electors'] = projected['electors'] * factor

    return projected


def _capped_shares(total: int, weights: pd.Series, caps: pd.Series) -> pd.Series:
    """
    Split an integer total in proportion to some weights without exceeding each item's cap.

    The shares are proportional to `weights`; an item whose share exceeds its cap takes its cap and the remainder
    is split again over the other items, until no share exceeds its cap. The shares are then floored and the
    units left go, one each, to the items with the largest fractional parts (the first ones by position on a tie).

    Parameters
    ----------
    total : int
        The integer to split.
    weights : pd.Series
        Non-negative weights; items with weight 0 take nothing.
    caps : pd.Series
        The most each item can take, aligned with `weights`.

    Returns
    -------
    pd.Series
        Integer shares, aligned with `weights`, adding up to exactly `total`.

    Raises
    ------
    ValueError
        If the caps of the items with a positive weight add up to less than `total`.
    """
    caps = caps.astype('float64')
    active = (weights > 0) & (caps > 0)
    if total > caps[active].sum():
        raise ValueError(f'cannot split {total}: the items with weight hold only {caps[active].sum():.0f}')

    shares = pd.Series(0., index=weights.index)
    remaining = float(total)
    while active.any():
        share = remaining * weights[active] / weights[active].sum()
        over = share[share >= caps[active]].index
        if len(over) == 0:
            shares[active] = share
            break
        shares[over] = caps[over]
        remaining -= caps[over].sum()
        active[over] = False

    floors = np.floor(shares)
    left = int(total - floors.sum())
    fraction = (shares - floors).sort_values(ascending=False, kind='stable')
    floors[fraction.index[:left]] += 1

    return floors.astype('int64')


def litigation_census(census: pd.DataFrame, long: pd.DataFrame, countries: pd.DataFrame, count: int,
                      since: str = '2022-10-01') -> pd.DataFrame:
    """
    The census without the inscriptions in litigation: `count` electors fewer of Latin American origin.

    Nobody publishes where the suspended inscriptions are, so they are taken out of the Latin American rows of
    `census` in two steps. First by province, in proportion to each province's Latin American growth in `long`
    between `since` (the last month on or before it) and the latest month of `long` (provinces that did not grow
    take nothing). Then, within each province, over its Latin American countries in proportion to their electors.
    No cell goes below 0: a share larger than what a province or a cell holds takes what it holds and the rest is
    split again over the others (`_capped_shares`), so exactly `count` electors are removed.

    Parameters
    ----------
    census : pd.DataFrame
        The convocation census, with `region_id`, `country_code` and `electors` (other columns are kept).
    long : pd.DataFrame
        The monthly census by province and country, with `date`, `region_id`, `country_code` and `electors`.
    countries : pd.DataFrame
        The countries table, with `code` and `region`.
    count : int
        The electors to remove; 0 returns an identical copy of `census`.
    since : str, default '2022-10-01'
        The start of the growth that weighs the provinces (the law's entry into force).

    Returns
    -------
    pd.DataFrame
        A copy of `census` with fewer `electors` in its Latin American rows.
    """
    reduced = census.copy()
    if count == 0:
        return reduced

    latam_long = long[_origin(long['country_code'], countries).eq(LATAM)]
    start = latam_long.loc[latam_long['date'] <= pd.Timestamp(since), 'date'].max()
    end = latam_long['date'].max()
    by_date = latam_long[latam_long['date'].isin([start, end])].pivot_table(
        index='region_id', columns='date', values='electors', aggfunc='sum', fill_value=0
    )
    growth = (by_date[end] - by_date[start]).clip(lower=0)

    latam = _origin(reduced['country_code'], countries).eq(LATAM)
    held = reduced[latam].groupby('region_id')['electors'].sum()
    by_province = _capped_shares(count, growth.reindex(held.index, fill_value=0), held)

    for region, removed in by_province[by_province > 0].items():
        rows = reduced.index[latam & reduced['region_id'].eq(region)]
        cells = reduced.loc[rows, 'electors']
        reduced.loc[rows, 'electors'] = cells - _capped_shares(int(removed), cells, cells)

    return reduced


def empirical_split(votes: pd.DataFrame) -> pd.DataFrame:
    """
    The share of each candidature in the votes of each province: the empirical split of new votes.

    Parameters
    ----------
    votes : pd.DataFrame
        Votes by `region_id` and candidature full name (the CERA votes of an election, in the study).

    Returns
    -------
    pd.DataFrame
        `votes` divided by its row sums (rows adding up to 1), same index and columns.
    """
    return votes.div(votes.sum(axis=1), axis=0)


def increment_votes(new: pd.DataFrame, participation: dict, split: pd.DataFrame, valid: float = 0.99) -> pd.DataFrame:
    """
    The votes the new electors would add to each candidature of each province.

    The votes of a province are `electors · participation[origin] · valid` added up over its origins; they go to
    the candidatures with the shares of `split`. Provinces absent from `split` get no votes.

    Parameters
    ----------
    new : pd.DataFrame
        New electors, with `region_id`, `origin` and `electors` (as `new_electors` returns).
    participation : dict
        Participation rate (0-1) by origin, `'latam'` and `'rest'`; other keys are ignored.
    split : pd.DataFrame
        Share of each candidature (columns) in each province (`region_id` index), rows adding up to 1: the CERA
        votes of 2023 to each candidature over the province's total.
    valid : float, default 0.99
        Fraction of the votes that are valid.

    Returns
    -------
    pd.DataFrame
        Float votes with the index and columns of `split`.
    """
    rate = new['origin'].map({LATAM: participation[LATAM], REST: participation[REST]})
    votes = (new['electors'] * rate * valid).groupby(new['region_id']).sum()

    return split.mul(votes.reindex(split.index, fill_value=0.), axis=0)


def _changes(before: dict[str, int], after: dict[str, int]) -> dict[str, dict[str, int]]:
    """
    Pair two seat tallies by key.

    Parameters
    ----------
    before : dict[str, int]
        Seats by key before.
    after : dict[str, int]
        Seats by key after, same keys.

    Returns
    -------
    dict[str, dict[str, int]]
        `{key: {'before', 'after', 'change'}}`.
    """
    return {key: {'before': before[key], 'after': after[key], 'change': after[key] - before[key]} for key in before}


def _sigla_seats(seats: pd.DataFrame, siglas: dict) -> dict[str, int]:
    """
    Add up the seats of each sigla.

    Parameters
    ----------
    seats : pd.DataFrame
        Seats with one column per candidature full name.
    siglas : dict
        Candidature full name to its sigla; candidatures without one go to `'Otros'`.

    Returns
    -------
    dict[str, int]
        Seats by sigla.
    """
    labels = [siglas.get(candidature) or OTHER_BLOCK for candidature in seats.columns]

    return {sigla: int(count) for sigla, count in seats.sum().groupby(labels).sum().items()}


def increment_model(results: Results, extra: pd.DataFrame, siglas: dict, blocks: dict) -> dict:
    """
    Allocate the seats of the results plus some extra votes and compare them with the seats of the results.

    Parameters
    ----------
    results : Results
        The results of the election the votes are added to.
    extra : pd.DataFrame
        Extra votes by `region_id` and candidature (as `increment_votes` returns); missing cells count as 0.
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    dict
        `seats_before` and `seats_after` (the allocations, `region_id` × candidature), `by_block` and `by_sigla`
        (`{name: {'before', 'after', 'change'}}`, the blocks with `'Otros'`), `changed` (the sorted `region_id`s
        whose seats differ) and `votes_extra` (the extra votes added up).
    """
    votes = results.votes.add(extra, fill_value=0).reindex(index=results.votes.index, columns=results.votes.columns)
    before = allocate_results(results)
    after = allocate_results(results, votes)

    return {
        'seats_before': before,
        'seats_after': after,
        'by_block': _changes(block_seats(before, siglas, blocks), block_seats(after, siglas, blocks)),
        'by_sigla': _changes(_sigla_seats(before, siglas), _sigla_seats(after, siglas)),
        'changed': sorted(int(region) for region in before.index[after.ne(before).any(axis=1)]),
        'votes_extra': float(extra.values.sum())
    }


def net_effect(last: pd.DataFrame, extra: pd.DataFrame, after: pd.DataFrame, before: pd.DataFrame) -> pd.DataFrame:
    """
    The net effect of the extra votes on the last seat of each province (the article's table 10).

    Parameters
    ----------
    last : pd.DataFrame
        The last seat of each province, as `last_seat` returns.
    extra : pd.DataFrame
        Extra votes by `region_id` and candidature.
    after : pd.DataFrame
        Seats with the extra votes (`region_id` × candidature).
    before : pd.DataFrame
        Seats without them, same shape.

    Returns
    -------
    pd.DataFrame
        `last` plus `d_runner` and `d_winner` (the extra votes of the runner-up and of the winner), `net`
        (`d_runner − d_winner`), `flipped` (whether the province's seats changed) and `participation_needed` (the
        factor the participation would have to be multiplied by for `net` to reach `gap`, `gap / net`; `None`
        when `net ≤ 0`). Every derived column but `flipped` is `None` where there is no runner-up, and is float
        when it has no `None`, object otherwise.
    """
    flipped = after.ne(before.reindex_like(after)).any(axis=1)
    rows = []

    for _, row in last.iterrows():
        region = row['region_id']
        d_runner, d_winner, net, needed = None, None, None, None
        if pd.notna(row['runner_up']):
            votes = extra.loc[region] if region in extra.index else pd.Series(dtype='float64')
            d_runner = float(votes.get(row['runner_up'], 0.))
            d_winner = float(votes.get(row['winner'], 0.))
            net = d_runner - d_winner
            needed = row['gap'] / net if net > 0 else None
        rows.append([d_runner, d_winner, net, bool(flipped.get(region, False)), needed])

    derived = pd.DataFrame(rows, columns=['d_runner', 'd_winner', 'net', 'flipped', 'participation_needed'],
                           index=last.index, dtype='object')
    derived['flipped'] = derived['flipped'].astype(bool)
    for column in ['d_runner', 'd_winner', 'net', 'participation_needed']:
        if derived[column].notna().all():
            derived[column] = derived[column].astype('float64')

    return pd.concat([last, derived], axis=1)
