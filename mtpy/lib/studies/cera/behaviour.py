"""
Who votes from abroad, whom they vote for and where the new CERA electors register.

Participation: by origin in the regional processes with figures by consulate (`participation_by_origin`), in every
general election with CERA columns in Infoelectoral (`participation_generales`) and in the regional processes of
`autonomicas.csv` (`autonomic_processes`). Vote: the left share over left + right of the CERA vote, its margin and the
exterior premium PSOE − PP (`generales_processes`), the representation ratios of 2023 (`representation_ratios`) and
the ecological inference between the Latin American weight of each province's CERA census and its left share
(`ecological`). Placement: the share of the sensitive provinces in the census and in the new electors
(`sensitive_share`), the acceleration of the additions by province (`acceleration`) and the «Madrid magnet»
(`madrid_magnet`).

Countries are always joined on their INE code, except in the consulate files, which only carry the name written
as in `countries.csv`. Only `participation_processes` and `available_results` read files.
"""
import json
import re

import numpy as np
import pandas as pd

from ....core.io import FileSystem
from .data import Results, read_consulates
from .seats import OTHER_BLOCK
from .stress import LEFT, RIGHT

ORIGINS = ('europe', 'usa_canada', 'latam', 'other')
ORIGIN_COLUMNS = ['origin', 'electors', 'voters', 'rate']
GENERALES_COLUMNS = ['date', 'electors', 'voters', 'rate', 'period']
RATIO_COLUMNS = ['sigla', 'share_cera', 'share_total', 'ratio']
ROGADO_START = '2011-03-28'
ROGADO_END = '2022-10-01'
BEFORE, ROGADO, NO_ROGADO = 'antes', 'rogado', 'sin_rogado'
RESULTS_FILE = re.compile(r'PROV_02_(\d{4})(\d{2})_1\.xlsx$')
ELECTION_DATES = 'wikipedia/wp-urls.json'
SENSITIVE_MAGNITUDE = 5
MADRID = 28
MADRID_ALL = 'Conjunto'
MADRID_SINGLES = ('MÉXICO', 'VENEZUELA')
PSOE, PP = 'PSOE', 'PP'


def _codes(names: list[str], countries: pd.DataFrame) -> list[int]:
    """
    The INE codes of some countries named as in `countries.csv`.

    Parameters
    ----------
    names : list[str]
        Country names, as in the column `name` of `countries`.
    countries : pd.DataFrame
        The table of `data.read_countries`.

    Returns
    -------
    list[int]
        The code of each name, in the same order.

    Raises
    ------
    ValueError
        Listing the names that are not in `countries`.
    """
    code_of = countries.set_index('name')['code']
    missing = [name for name in names if name not in code_of.index]
    if missing:
        raise ValueError(f'cera: countries not in countries.csv: {missing}')

    return [int(code_of[name]) for name in names]


def participation_by_origin(consulates: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """
    The CERA participation of a regional process by origin of the electors.

    The consulate files carry only the country name, written as in `countries.csv`, so this is the one join on
    names of the study.

    Parameters
    ----------
    consulates : pd.DataFrame
        The table of `data.read_consulates`: `country`, `electors` and `voters`.
    countries : pd.DataFrame
        The table of `data.read_countries`, with `name` and `region`.

    Returns
    -------
    pd.DataFrame
        Columns `ORIGIN_COLUMNS`, one row per origin present, in the order of `ORIGINS`; `rate` is
        `100 · voters / electors`.

    Raises
    ------
    ValueError
        Listing the countries of `consulates` that are not in `countries`.
    """
    region_of = countries.set_index('name')['region']
    missing = sorted(set(consulates['country']) - set(region_of.index))
    if missing:
        raise ValueError(f'cera: consulate countries not in countries.csv: {missing}')

    table = consulates.assign(origin=consulates['country'].map(region_of))
    table = table.groupby('origin')[['electors', 'voters']].sum()
    table = table.reindex([origin for origin in ORIGINS if origin in table.index]).reset_index()
    table['rate'] = 100. * table['voters'] / table['electors']

    return table[ORIGIN_COLUMNS]


def participation_processes(data: FileSystem, countries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    The participation by origin of every regional process with figures by consulate (`cera/consulados/`).

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    countries : pd.DataFrame
        The table of `data.read_countries`.

    Returns
    -------
    dict[str, pd.DataFrame]
        File stem (`cyl-2026`, `aragon-2026`, ...) to its `participation_by_origin`, in the order of the names.
    """
    processes = {}
    for name in data.glob('cera/consulados/*.csv'):
        stem = name.rsplit('/', 1)[-1][:-len('.csv')]
        processes[stem] = participation_by_origin(read_consulates(data.get_path(name)), countries)

    return processes


def available_results(data: FileSystem) -> list[str]:
    """
    The dates of the general elections with an Infoelectoral results file (`infoelectoral/es/PROV_02_*_1.xlsx`).

    The file name only carries the year and month; the day comes from the general elections listed under `es` in
    `wikipedia/wp-urls.json`, and is the first of the month when that month is not listed there.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.

    Returns
    -------
    list[str]
        `YYYY-MM-DD` dates, sorted. Every one can be passed to `data.read_results`, which fails on the oldest files,
        those without the CERA columns.
    """
    known = json.loads(data.read(ELECTION_DATES)).get('es', {}) if data.exists(ELECTION_DATES) else {}
    day_of = {date[:7]: date for date in known}
    dates = []
    for name in data.glob('infoelectoral/es/PROV_02_*_1.xlsx'):
        match = RESULTS_FILE.search(name)
        if match:
            month = f'{match.group(1)}-{match.group(2)}'
            dates.append(day_of.get(month, f'{month}-01'))

    return sorted(dates)


def _period(date: str) -> str:
    """
    The period of a general election with respect to the voto rogado.

    Parameters
    ----------
    date : str
        The date of the election (`YYYY-MM-DD`).

    Returns
    -------
    str
        `'antes'` before `ROGADO_START`, `'rogado'` from it until `ROGADO_END`, `'sin_rogado'` from then on.
    """
    if date < ROGADO_START:
        return BEFORE

    return ROGADO if date < ROGADO_END else NO_ROGADO


def participation_generales(results: dict[str, Results]) -> pd.DataFrame:
    """
    The CERA participation of each general election: CERA voters over the CERA census of Infoelectoral.

    Parameters
    ----------
    results : dict[str, Results]
        Election date to its results. Elections whose `totals` lack `registered_cera` or `counted_cera` are
        skipped.

    Returns
    -------
    pd.DataFrame
        Columns `GENERALES_COLUMNS`, one row per election sorted by date: `electors` and `voters` are the sums of
        `registered_cera` and `counted_cera` over the provinces, `rate` is `100 · voters / electors` and `period`
        is the one of `_period`.
    """
    rows = []
    for date, result in sorted(results.items()):
        if not {'registered_cera', 'counted_cera'} <= set(result.totals.columns):
            continue
        electors, voters = int(result.totals['registered_cera'].sum()), int(result.totals['counted_cera'].sum())
        rows.append([date, electors, voters, 100. * voters / electors, _period(date)])

    return pd.DataFrame(rows, columns=GENERALES_COLUMNS)


def _sigla_votes(votes: pd.DataFrame, siglas: dict) -> pd.DataFrame:
    """
    Add up the votes of the candidatures of each sigla, province by province.

    Parameters
    ----------
    votes : pd.DataFrame
        Votes with `region_id` as index and one column per candidature full name.
    siglas : dict
        Candidature full name to its sigla; candidatures without one go to `'Otros'`.

    Returns
    -------
    pd.DataFrame
        Votes with `region_id` as index and one column per sigla.
    """
    labels = [siglas.get(candidature) or OTHER_BLOCK for candidature in votes.columns]

    return votes.T.groupby(labels).sum().T


def shares_by_block(votes: pd.DataFrame, siglas: dict, blocks: dict) -> pd.DataFrame:
    """
    Add up the votes of each block, province by province.

    Parameters
    ----------
    votes : pd.DataFrame
        Votes with `region_id` as index and one column per candidature full name.
    siglas : dict
        Candidature full name to its sigla (`None` for the others).
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    pd.DataFrame
        Votes with `region_id` as index and one column per block of `blocks` plus `'Otros'`, which takes the
        candidatures without a sigla or whose sigla is in no block. The shares come out of it with `left_share`.
    """
    block_of = {sigla: block for block, members in blocks.items() for sigla in members}
    labels = [block_of.get(siglas.get(candidature), OTHER_BLOCK) for candidature in votes.columns]
    table = votes.T.groupby(labels).sum().T

    return table.reindex(columns=[*blocks, OTHER_BLOCK], fill_value=0).rename_axis(index='region_id', columns=None)


def left_share(block_votes: pd.DataFrame) -> float:
    """
    The share of the left over left + right of some block votes, added up over every province.

    Parameters
    ----------
    block_votes : pd.DataFrame
        The table of `shares_by_block`.

    Returns
    -------
    float
        `100 · left / (left + right)`, in percent.
    """
    left, right = block_votes[LEFT].sum(), block_votes[RIGHT].sum()

    return 100. * left / (left + right)


def _psoe_pp(votes: pd.DataFrame, siglas: dict) -> float:
    """
    The margin PSOE − PP of some votes, in points over the votes to candidatures of every province.

    Parameters
    ----------
    votes : pd.DataFrame
        Votes with `region_id` as index and one column per candidature full name.
    siglas : dict
        Candidature full name to its sigla.

    Returns
    -------
    float
        `100 · (PSOE − PP) / candidature votes`.
    """
    by_sigla = _sigla_votes(votes, siglas).sum()

    return 100. * (by_sigla.get(PSOE, 0) - by_sigla.get(PP, 0)) / by_sigla.sum()


def generales_processes(cera: dict[str, pd.DataFrame], results: dict[str, Results], blocks: dict) -> list[dict]:
    """
    The block vote and the exterior premium of the CERA vote of each general election.

    Parameters
    ----------
    cera : dict[str, pd.DataFrame]
        Election date to its CERA votes by `region_id` and candidature full name.
    results : dict[str, Results]
        Election date to its results; it must hold every date of `cera`, whose siglas it gives.
    blocks : dict
        Block name to the list of its siglas.

    Returns
    -------
    list[dict]
        One dictionary per election of `cera`, sorted by date: `date`, `left_share` (`left_share` of the CERA vote),
        `margin` (left − right in points over left + right), `psoe_pp_cera` and `psoe_pp_total` (PSOE − PP in
        points over the votes to candidatures of the CERA vote and of the whole election) and `premium`
        (`psoe_pp_cera − psoe_pp_total`).
    """
    rows = []
    for date in sorted(cera):
        siglas = results[date].siglas
        by_block = shares_by_block(cera[date], siglas, blocks)
        left, right = by_block[LEFT].sum(), by_block[RIGHT].sum()
        psoe_pp_cera, psoe_pp_total = _psoe_pp(cera[date], siglas), _psoe_pp(results[date].votes, siglas)
        rows.append({
            'date': date,
            'left_share': left_share(by_block),
            'margin': 100. * (left - right) / (left + right),
            'psoe_pp_cera': psoe_pp_cera,
            'psoe_pp_total': psoe_pp_total,
            'premium': psoe_pp_cera - psoe_pp_total
        })

    return rows


def representation_ratios(cera_votes: pd.DataFrame, results: Results, siglas: dict, main: list[str]) -> pd.DataFrame:
    """
    The representation of each party in the CERA vote with respect to its share of the whole election.

    Parameters
    ----------
    cera_votes : pd.DataFrame
        CERA votes by `region_id` and candidature full name.
    results : Results
        The results of the election (CERA votes included).
    siglas : dict
        Candidature full name to its sigla.
    main : list[str]
        The siglas to report.

    Returns
    -------
    pd.DataFrame
        Columns `RATIO_COLUMNS`, one row per sigla of `main` in its order: `share_cera` and `share_total` are the
        percent of the votes to candidatures (CERA and whole election) and `ratio` is `100 · share_cera /
        share_total`.
    """
    cera = _sigla_votes(cera_votes, siglas).sum()
    total = _sigla_votes(results.votes, siglas).sum()
    share_cera = 100. * cera.reindex(main, fill_value=0) / cera.sum()
    share_total = 100. * total.reindex(main, fill_value=0) / total.sum()
    table = pd.DataFrame({
        'sigla': main, 'share_cera': share_cera.values, 'share_total': share_total.values
    })
    table['ratio'] = 100. * table['share_cera'] / table['share_total']

    return table[RATIO_COLUMNS]


def _autonomic_row(row: pd.Series) -> dict:
    """
    The participation and the vote of one regional process of `autonomicas.csv`.

    Parameters
    ----------
    row : pd.Series
        A row of `data.read_autonomicas`.

    Returns
    -------
    dict
        `date`, `electors`, `voters`, `rate` (`100 · voters / electors`), `pp_pct`, `psoe_pct`, `vox_pct` and
        `other_left_pct` (percent of the voters, as the article's table 5), `left_share` (`100 · left / (left +
        right)`, its table 4) and `margin` (`100 · (psoe + other_left − pp − vox) / voters`, its figure 6).
    """
    voters = int(row['voters'])
    pct = {party: 100. * row[party] / voters for party in ('pp', 'psoe', 'vox', 'other_left')}

    return {
        'date': row['date'],
        'electors': int(row['electors']),
        'voters': voters,
        'rate': 100. * voters / row['electors'],
        **{f'{party}_pct': value for party, value in pct.items()},
        'left_share': 100. * row['left'] / (row['left'] + row['right']),
        'margin': pct['psoe'] + pct['other_left'] - pct['pp'] - pct['vox']
    }


def autonomic_processes(autonomicas: pd.DataFrame) -> list[dict]:
    """
    The previous and the last regional process of each community, with the changes of the left vote.

    Parameters
    ----------
    autonomicas : pd.DataFrame
        The table of `data.read_autonomicas`; the last process of a community is the row with a `previous` date,
        which names the row of the previous one.

    Returns
    -------
    list[dict]
        One dictionary per community, in the order of the table: `community`, `previous` and `last` (as
        `_autonomic_row`), `d_psoe` and `d_other_left` (change of their percent of the voters, in points) and
        `d_block` (their sum, the change of the left block over the voters).

    Raises
    ------
    ValueError
        If a community has no row with a `previous` date or that date has no row.
    """
    rows = []
    for community, group in autonomicas.groupby('community', sort=False):
        lasts = group[group['previous'].notna()]
        if lasts.empty:
            raise ValueError(f'cera: {community} has no process with a previous one')
        last = lasts.sort_values('date').iloc[-1]
        previous = group[group['date'] == last['previous']]
        if previous.empty:
            raise ValueError(f'cera: {community} has no process on {last["previous"]}')
        before, after = _autonomic_row(previous.iloc[0]), _autonomic_row(last)
        d_psoe = after['psoe_pct'] - before['psoe_pct']
        d_other_left = after['other_left_pct'] - before['other_left_pct']
        rows.append({
            'community': community, 'previous': before, 'last': after,
            'd_psoe': d_psoe, 'd_other_left': d_other_left, 'd_block': d_psoe + d_other_left
        })

    return rows


def _latam_share(long: pd.DataFrame, at: pd.Timestamp, countries: pd.DataFrame) -> pd.DataFrame:
    """
    The CERA census of each province on a date and the share of it living in Latin America.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    at : pd.Timestamp
        A first-of-month date of `long`.
    countries : pd.DataFrame
        The table of `data.read_countries`, with `code` and `region`.

    Returns
    -------
    pd.DataFrame
        Indexed by `region_id`: `census` (electors) and `latam_share` (percent of them in `latam` countries).

    Raises
    ------
    KeyError
        If `long` has no row on `at`.
    """
    month = long[long['date'] == at]
    if month.empty:
        raise KeyError(f'cera: no monthly census on {at:%Y-%m-%d}')
    latam = month['country_code'].map(countries.set_index('code')['region']).eq('latam')
    census = month.groupby('region_id')['electors'].sum()
    in_latam = month[latam].groupby('region_id')['electors'].sum().reindex(census.index, fill_value=0)

    return pd.DataFrame({'census': census, 'latam_share': 100. * in_latam / census})


def ecological(cera_votes: pd.DataFrame, voters: pd.Series, siglas: dict, blocks: dict, long: pd.DataFrame,
               at: pd.Timestamp, countries: pd.DataFrame, names: pd.Series, min_voters: int,
               only: list[int] | None = None) -> dict:
    """
    The ecological inference between the Latin American weight of each province's CERA census and its left share.

    Parameters
    ----------
    cera_votes : pd.DataFrame
        CERA votes by `region_id` and candidature full name.
    voters : pd.Series
        CERA voters by `region_id` (the scrutiny's `voters`).
    siglas : dict
        Candidature full name to its sigla.
    blocks : dict
        Block name to the list of its siglas.
    long : pd.DataFrame
        The table of `data.read_monthly`.
    at : pd.Timestamp
        The monthly close whose census gives the Latin American weight.
    countries : pd.DataFrame
        The table of `data.read_countries`.
    names : pd.Series
        Province names by `region_id`.
    min_voters : int
        Provinces with fewer CERA voters are left out (Ceuta and Melilla with `params['ecological_min_voters']`).
    only : list[int], optional
        The `region_id` to keep besides; the article's 43 provinces reproduce its regression.

    Returns
    -------
    dict
        `rows` (`[{region_id, name, latam_share, left_share, census}]`, by `region_id`), `r` (Pearson's
        correlation), `slope` (least squares, points of left share per point of Latin American share),
        `quartiles` (mean left share of each quartile of `latam_share`, lowest first) and `n`.
    """
    by_block = shares_by_block(cera_votes, siglas, blocks)
    table = _latam_share(long, at, countries).join(
        (100. * by_block[LEFT] / (by_block[LEFT] + by_block[RIGHT])).rename('left_share'), how='inner'
    )
    keep = voters.reindex(table.index, fill_value=0) >= min_voters
    if only is not None:
        keep &= table.index.isin(only)
    table = table[keep].sort_index()
    table.insert(0, 'name', table.index.map(names))

    x, y = table['latam_share'].to_numpy(), table['left_share'].to_numpy()
    quartiles = table.groupby(pd.qcut(table['latam_share'], 4, labels=False))['left_share'].mean()
    rows = table.reset_index()[['region_id', 'name', 'latam_share', 'left_share', 'census']]

    return {
        'rows': [
            {**row, 'region_id': int(row['region_id']), 'census': int(row['census'])}
            for row in rows.to_dict('records')
        ],
        'r': float(np.corrcoef(x, y)[0, 1]),
        'slope': float(np.polyfit(x, y, 1)[0]),
        'quartiles': [float(value) for value in quartiles],
        'n': len(table)
    }


def _electors_by_province(long: pd.DataFrame, codes: list[int], date: pd.Timestamp) -> pd.Series:
    """
    The electors of some countries on a monthly close, by province.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    codes : list[int]
        INE country codes.
    date : pd.Timestamp
        A first-of-month date of `long`.

    Returns
    -------
    pd.Series
        Electors by `region_id` (only the provinces with a row).

    Raises
    ------
    KeyError
        If `long` has no row on `date`.
    """
    if not (long['date'] == date).any():
        raise KeyError(f'cera: no monthly census on {date:%Y-%m-%d}')
    rows = long[(long['date'] == date) & long['country_code'].isin(codes)]

    return rows.groupby('region_id')['electors'].sum()


def _stock_and_growth(long: pd.DataFrame, codes: list[int], start: pd.Timestamp,
                      end: pd.Timestamp) -> pd.DataFrame:
    """
    The electors of some countries at the start of a period and their growth in it, by province.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    codes : list[int]
        INE country codes.
    start, end : pd.Timestamp
        First-of-month dates of `long`.

    Returns
    -------
    pd.DataFrame
        Indexed by `region_id`: `stock` (electors on `start`) and `growth` (on `end` minus on `start`).
    """
    first, last = _electors_by_province(long, codes, start), _electors_by_province(long, codes, end)
    both = pd.concat({'stock': first, 'end': last}, axis=1).fillna(0)

    return pd.DataFrame({'stock': both['stock'], 'growth': both['end'] - both['stock']})


def sensitive_share(long: pd.DataFrame, countries: pd.DataFrame, codes: list[str], magnitudes: pd.Series,
                    start: str, end: pd.Timestamp, max_magnitude: int = SENSITIVE_MAGNITUDE) -> dict:
    """
    The share of the sensitive provinces in the census of some countries and in its growth.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    countries : pd.DataFrame
        The table of `data.read_countries`.
    codes : list[str]
        The countries, named as in `countries.csv` (joined to their INE code).
    magnitudes : pd.Series
        Seats by `region_id`.
    start : str
        The first-of-month date of the census before the law (`2022-10-01`).
    end : pd.Timestamp
        The first-of-month date the growth runs to.
    max_magnitude : int, default 5
        The provinces with at most these seats are the sensitive ones.

    Returns
    -------
    dict
        `stock_share` (percent of the electors on `start` registered in sensitive provinces) and `growth_share`
        (percent of the growth from `start` to `end` that went to them).
    """
    table = _stock_and_growth(long, _codes(codes, countries), pd.Timestamp(start), pd.Timestamp(end))
    sensitive = magnitudes.reindex(table.index) <= max_magnitude

    return {
        'stock_share': 100. * table.loc[sensitive, 'stock'].sum() / table['stock'].sum(),
        'growth_share': 100. * table.loc[sensitive, 'growth'].sum() / table['growth'].sum()
    }


def acceleration(two_periods_by_province: pd.DataFrame, magnitudes: pd.Series, gaps: pd.Series) -> dict:
    """
    The acceleration of the CERA additions by province against the size of the province and its last-seat gap.

    Parameters
    ----------
    two_periods_by_province : pd.DataFrame
        The table of `census.two_periods` with the key `region_id`.
    magnitudes : pd.Series
        Seats by `region_id`.
    gaps : pd.Series
        The last-seat gap by `region_id` (`seats.last_seat`).

    Returns
    -------
    dict
        `rows` (`[{region_id, name, g1, g2, ratio, magnitude, gap}]`, `ratio = g2 / g1`), the Pearson correlations
        `r_size_growth` (magnitude and `g2`), `r_size_ratio` (magnitude and `ratio`) and `r_gap_growth` (gap and
        `g2`), over the provinces with a finite ratio and a gap, and `share_sensitive_growth` and
        `share_sensitive_stock`: percent of the additions of the second period, and of the electors at its start,
        in the provinces with at most `SENSITIVE_MAGNITUDE` seats.
    """
    table = two_periods_by_province.set_index('region_id')
    table = table.assign(magnitude=magnitudes.reindex(table.index), gap=gaps.reindex(table.index))
    valid = table[np.isfinite(table['ratio']) & table['gap'].notna()]
    sensitive = table['magnitude'] <= SENSITIVE_MAGNITUDE
    rows = table.reset_index()[['region_id', 'name', 'g1', 'g2', 'ratio', 'magnitude', 'gap']]

    return {
        'rows': [
            {**row, 'region_id': int(row['region_id']), 'g1': int(row['g1']), 'g2': int(row['g2']),
             'magnitude': int(row['magnitude'])}
            for row in rows.to_dict('records')
        ],
        'r_size_growth': float(valid['magnitude'].corr(valid['g2'])),
        'r_size_ratio': float(valid['magnitude'].corr(valid['ratio'])),
        'r_gap_growth': float(valid['gap'].corr(valid['g2'])),
        'share_sensitive_growth': 100. * table.loc[sensitive, 'g2'].sum() / table['g2'].sum(),
        'share_sensitive_stock': 100. * table.loc[sensitive, 'c2_start'].sum() / table['c2_start'].sum()
    }


def madrid_magnet(long: pd.DataFrame, countries: pd.DataFrame, codes: list[str], periods: list[tuple]) -> list[dict]:
    """
    The share of Madrid in the census of some countries before the law and in their additions of each period.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    countries : pd.DataFrame
        The table of `data.read_countries`.
    codes : list[str]
        The countries, named as in `countries.csv` (`params['madrid_countries']`).
    periods : list[tuple]
        Two `(start, end)` pairs of first-of-month dates; the stock is the census at the start of the second one.

    Returns
    -------
    list[dict]
        `{'label', 'stock_share', 'share_p1', 'share_p2'}` (percent in Madrid) for the countries together
        (`'Conjunto'`) and for each of `MADRID_SINGLES` among them. The share of a period whose total growth is
        not positive is `None` (Venezuela lost electors before the law, so Madrid's share of its «additions»
        would be meaningless).
    """
    groups = [(MADRID_ALL, codes), *[(name, [name]) for name in MADRID_SINGLES if name in codes]]
    rows = []
    for label, names in groups:
        country_codes = _codes(names, countries)
        shares = []
        for start, end in periods:
            growth = _stock_and_growth(long, country_codes, pd.Timestamp(start), pd.Timestamp(end))['growth']
            total = growth.sum()
            shares.append(100. * growth.get(MADRID, 0) / total if total > 0 else None)
        stock = _electors_by_province(long, country_codes, pd.Timestamp(periods[1][0]))
        rows.append({
            'label': label, 'stock_share': 100. * stock.get(MADRID, 0) / stock.sum(),
            'share_p1': shares[0], 'share_p2': shares[1]
        })

    return rows
