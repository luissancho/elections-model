"""
The CERA study in one call: `compute_cera` reads every input through the data file system and chains the
computations of the other modules in the order of the spec, and `CeraStudy` carries their results.

The order is the D'Hondt validation of the elections it uses (it fails before anything else if the official seats
do not reproduce) and the last seat of the baseline election, then the series and the growth (question 1), the
participation, the vote and the placement (question 2), the increment model in both census scenarios, the
sensitivity grids and the replication of the «23 seats» report (question 3), and the check with and without the
CERA vote of every election with a CERA scrutiny. This module only reads and wires: every figure comes from
`census`, `behaviour`, `seats`, `stress` and `check`.
"""
import re

from dataclasses import dataclass

import pandas as pd

from ....core.io import FileSystem
from .behaviour import (
    acceleration,
    autonomic_processes,
    available_results,
    ecological,
    generales_processes,
    madrid_magnet,
    participation_generales,
    participation_processes,
    representation_ratios,
    sensitive_share
)
from .census import MARKS, baseline_date, election_growth, trend_break, two_periods, update_periods
from .check import with_without
from .data import (
    MissingCeraColumns,
    Results,
    load_params,
    national_series,
    read_autonomicas,
    read_candidatures_map,
    read_countries,
    read_election_census,
    read_monthly,
    read_provinces,
    read_results,
    read_scrutiny,
    scrutiny_votes
)
from .seats import (
    check_official,
    empirical_split,
    increment_model,
    increment_votes,
    last_seat,
    litigation_census,
    net_effect,
    new_electors,
    participation_history
)
from .stress import EMPIRICAL, exterior_grid, optimal_placement, placement_pool, report_replication, scenario_table

PROVINCE_NAMES = 'es-provinces.csv'
COUNTRIES = 'cera/countries.csv'
AUTONOMICAS = 'cera/autonomicas.csv'
CENSUS_TABLE = re.compile(r'_tab8\.xlsx$')
SCRUTINY = re.compile(r'cera/escrutinio/es-(\d{4}-\d{2}-\d{2})\.csv$')
GROWTH_KEYS = {'provinces': 'region_id', 'countries': 'country_code', 'origins': 'region', 'exposure': 'exposure'}
ELECTION_KEYS = {'provinces': 'region_id', 'countries': 'country_code'}
RATIO_SIGLAS = ['PP', 'PSOE', 'SUMAR', 'VOX']
SENSITIVE_COUNTRIES = ['ARGENTINA', 'CUBA', 'VENEZUELA']
CONVOCATION, NO_LITIGATION = 'convocatoria', 'sin_litigio'
GRID_PARTICIPATIONS = [0.09, 0.2, 0.3, 0.5, 0.7]
GRID_LEFT_FRACS = [0.5, 0.6, 0.7, 0.8]
SCENARIO_ROWS = [
    {'census_x': None, 'part_x': 1, 'bias': EMPIRICAL},
    {'census_x': None, 'part_x': 2, 'bias': EMPIRICAL},
    {'census_x': None, 'part_x': 1, 'bias': 0.9},
    {'census_x': 1.23, 'part_x': 3, 'bias': 0.9},
    {'census_x': 1.5, 'part_x': 2, 'bias': 0.9},
    {'census_x': 2.0, 'part_x': 3, 'bias': 0.9}
]
PLACEMENT_PARTICIPATIONS = [0.09, 0.2, 0.5, 1.0]
PLACEMENT_FRACS = [0.5, 0.6, 0.7, 0.8, 1.0]
PLACEMENT_CANDIDATURE = 'PARTIDO SOCIALISTA OBRERO ESPAÑOL'
VERDICT_SIGLA, VERDICT_BLOCK = 'PSOE', 'Izquierda'


@dataclass
class CeraStudy:
    """
    Every result of the CERA study, as `compute_cera` computes it.

    Frames keep the columns of the function that builds them (see its docstring); dates are `YYYY-MM-DD` strings
    as keys and `pd.Timestamp` inside the census frames and periods.

    Attributes
    ----------
    params : dict
        The parameters of the study (`cera/params.json`).
    countries : pd.DataFrame
        The countries of residence (`cera/countries.csv`): `code`, `name`, `region` and `exposure`.
    province_names : pd.Series
        Province names by `region_id` (`es-provinces.csv`).
    long : pd.DataFrame
        The monthly CERA census by province and country (`data.read_monthly`).
    series : pd.Series
        National CERA electors by month (`data.national_series`).
    series_last : pd.Timestamp
        The last monthly close.
    marks : list[tuple[str, str]]
        The dated marks of the series chart (`census.MARKS`).
    baseline_date : pd.Timestamp
        The monthly close closest to the CERA census of the baseline election in Infoelectoral (2023-06-01).
    baseline : pd.DataFrame
        The census of the baseline election: `region_id`, `country_code` and `electors` of `long` at
        `baseline_date`.
    census : pd.DataFrame
        The convocation census by province and country (`tab8`, `data.read_election_census`).
    census_by_province : pd.Series
        Its electors by `region_id`.
    periods_article : list
        The article's two periods (`params['periods']['article']`).
    periods_now : list[tuple[pd.Timestamp, pd.Timestamp]]
        The same periods ending at `series_last` (`census.update_periods`).
    growth : dict[str, dict[str, pd.DataFrame]]
        `census.two_periods` by group and version: `growth[group][version]`, with `group` one of `provinces`
        (`region_id`), `countries` (`country_code`), `origins` (`region`) and `exposure`, and `version` `article`
        (`periods_article`) or `now` (`periods_now`).
    election_growth : dict[str, pd.DataFrame]
        `census.election_growth` from `baseline_date` to `census`, by `provinces` and by `countries`.
    regression : dict
        `census.trend_break` over the whole `series`.
    participation : dict
        `origin`: `behaviour.participation_processes` (file stem to the participation by origin);
        `generales`: `behaviour.participation_generales` over `results`.
    generales : list[dict]
        `behaviour.generales_processes`: the block vote and the exterior premium of each election of `cera`.
    autonomics : list[dict]
        `behaviour.autonomic_processes` of `cera/autonomicas.csv` (participation and vote).
    ratios : pd.DataFrame
        `behaviour.representation_ratios` of the baseline scrutiny for `RATIO_SIGLAS`.
    ecological : dict
        `behaviour.ecological` of the baseline scrutiny at `baseline_date`, leaving out the provinces with fewer
        CERA voters than `params['ecological_min_voters']`.
    sensitive : dict[str, dict]
        By version (`article`: from the law to the end of the article's second period; `now`: to the census
        closed for the election, `params['election']['census_closed']`): `start`, `end`, and the
        `behaviour.sensitive_share` of `SENSITIVE_COUNTRIES` (`countries`) and of every country (`all`).
    acceleration : dict[str, dict]
        `behaviour.acceleration` by version (`article`, `now`), over `growth['provinces'][version]`.
    madrid : dict[str, list[dict]]
        `behaviour.madrid_magnet` of `params['madrid_countries']` by version (`article`, `now`).
    results : dict[str, Results]
        Election date to its Infoelectoral results, for every general election with CERA columns (1986–2023).
    scrutinies : dict[str, pd.DataFrame]
        Election date to its CERA scrutiny (`data.read_scrutiny`), for the elections with a scrutiny file and
        Infoelectoral results.
    skipped_scrutinies : list[str]
        The dates of the scrutiny files left out because their Infoelectoral results (`PROV_02` XLSX) are not
        loaded yet, ascending (`_read_scrutinies`).
    cera : dict[str, pd.DataFrame]
        Election date to its CERA votes by `region_id` and candidature full name (`data.scrutiny_votes`), same keys.
    voters : dict[str, pd.Series]
        Election date to its CERA voters by `region_id` (the scrutiny's `voters`, Infoelectoral's figures).
    blocks : dict
        Block name to the list of its siglas (`params['blocks']`).
    split : pd.DataFrame
        The empirical split of the new votes: `seats.empirical_split` of the CERA votes of `params['split_source']`.
    last : pd.DataFrame
        `seats.last_seat` of the baseline election (52 rows).
    scenarios : dict[str, dict]
        `convocatoria` (the convocation `census`) and `sin_litigio` (`seats.litigation_census` without
        `params['litigation']['count']`), each with `census` (province × country), `new` (`seats.new_electors`),
        `extra` (`seats.increment_votes`), `model` (`seats.increment_model`) and `net` (`seats.net_effect`).
    grid : pd.DataFrame
        `stress.exterior_grid`: participations `GRID_PARTICIPATIONS` × left fractions `GRID_LEFT_FRACS`.
    scenario_rows : pd.DataFrame
        `stress.scenario_table` of `SCENARIO_ROWS`.
    placement : dict
        `stress.optimal_placement` of the pool (`grid`, `contrast`) with `PLACEMENT_PARTICIPATIONS` ×
        `PLACEMENT_FRACS` for `PLACEMENT_CANDIDATURE`, and `pool`, the electors of the pool
        (`stress.placement_pool` of `params['placement_countries']` since the law).
    hist : pd.DataFrame
        `seats.participation_history` over `results`: mean and maximum CERA participation by province since 1986.
    report : dict
        `stress.report_replication` of the baseline election with `census_by_province` and `hist` (votes, provinces
        over the gap and the participation those rates give over the whole convocation census).
    checks : dict[str, dict]
        Election date to `check.with_without` with the scrutiny's `voters`, for every election of `cera`.
    verdict_input : dict or None
        `{'net_psoe', 'net_left'}` of the check of the study's election (`params['election']['date']`), the
        arguments of `check.verdict`; `None` while that election has no check.
    """
    params: dict
    countries: pd.DataFrame
    province_names: pd.Series
    long: pd.DataFrame
    series: pd.Series
    series_last: pd.Timestamp
    marks: list
    baseline_date: pd.Timestamp
    baseline: pd.DataFrame
    census: pd.DataFrame
    census_by_province: pd.Series
    periods_article: list
    periods_now: list
    growth: dict
    election_growth: dict
    regression: dict
    participation: dict
    generales: list
    autonomics: list
    ratios: pd.DataFrame
    ecological: dict
    sensitive: dict
    acceleration: dict
    madrid: dict
    results: dict
    scrutinies: dict
    skipped_scrutinies: list
    cera: dict
    voters: dict
    blocks: dict
    split: pd.DataFrame
    last: pd.DataFrame
    scenarios: dict
    grid: pd.DataFrame
    scenario_rows: pd.DataFrame
    placement: dict
    hist: pd.DataFrame
    report: dict
    checks: dict
    verdict_input: dict | None


def _read_all_results(data: FileSystem) -> dict[str, Results]:
    """
    Read the Infoelectoral results of every general election that has the CERA columns.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.

    Returns
    -------
    dict[str, Results]
        Election date to its results, sorted by date; the oldest files (1977–1982), without the CERA columns, are
        left out.

    Raises
    ------
    ValueError
        As `data.read_results` does, for any reason other than missing CERA columns (`data.MissingCeraColumns`).
    """
    results = {}
    for date in available_results(data):
        try:
            results[date] = read_results(data, date)
        except MissingCeraColumns:
            pass

    return results


def _read_scrutinies(data: FileSystem, results: dict, candidatures: dict) -> tuple[dict, list]:
    """
    Read the CERA scrutinies of `cera/escrutinio/` whose election has Infoelectoral results.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    results : dict
        Election date to its results (`_read_all_results`).
    candidatures : dict
        The candidatures map (`data.read_candidatures_map`).

    Returns
    -------
    tuple
        `(scrutinies, skipped)`: election date to its scrutiny (`data.read_scrutiny`), sorted by date, and the
        dates of the scrutiny files left out because their election has no results loaded, ascending.

    Raises
    ------
    ValueError
        If a scrutiny with results has no entry in `cera/escrutinio/candidatures.json`.
    """
    scrutinies, skipped = {}, []
    for name in data.glob('cera/escrutinio/es-*.csv'):
        match = SCRUTINY.search(name)
        if not match:
            continue
        date = match.group(1)
        if date not in results:
            skipped.append(date)
            continue
        if date not in candidatures:
            raise ValueError(f'cera: no candidatures map for {date} in cera/escrutinio/candidatures.json')
        scrutinies[date] = read_scrutiny(data.get_path(name))

    return dict(sorted(scrutinies.items())), sorted(skipped)


def _read_inputs(data: FileSystem, params: dict) -> dict:
    """
    Read the census, the results and the CERA scrutinies of the study.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    params : dict
        The parameters of the study.

    Returns
    -------
    dict
        The fields `countries`, `province_names`, `long`, `series`, `series_last`, `census`, `census_by_province`,
        `results`, `scrutinies`, `skipped_scrutinies`, `cera`, `voters` and `blocks` of `CeraStudy`.

    Raises
    ------
    ValueError
        If a scrutiny with results has no entry in the candidatures map (`_read_scrutinies`).
    """
    provinces = read_provinces(data)
    long = read_monthly(data, provinces)
    series = national_series(long)
    table = next(name for name in params['election']['census_tables'] if CENSUS_TABLE.search(name))
    census = read_election_census(data.get_path(f'cera/{table}'), provinces)
    results = _read_all_results(data)

    candidatures = read_candidatures_map(data)
    scrutinies, skipped = _read_scrutinies(data, results, candidatures)

    return {
        'countries': read_countries(data.get_path(COUNTRIES)),
        'province_names': pd.read_csv(data.get_path(PROVINCE_NAMES)).set_index('code')['province'],
        'long': long,
        'series': series,
        'series_last': series.index[-1],
        'census': census,
        'census_by_province': census.groupby('region_id')['electors'].sum(),
        'results': results,
        'scrutinies': scrutinies,
        'skipped_scrutinies': skipped,
        'cera': {date: scrutiny_votes(scrutiny, candidatures[date]) for date, scrutiny in scrutinies.items()},
        'voters': {date: scrutiny['voters'] for date, scrutiny in scrutinies.items()},
        'blocks': params['blocks']
    }


def _dhondt_block(params: dict, inputs: dict) -> dict:
    """
    Validate the D'Hondt allocation of the elections the study uses and find the last seat of the baseline one.

    Parameters
    ----------
    params : dict
        The parameters of the study.
    inputs : dict
        The output of `_read_inputs`.

    Returns
    -------
    dict
        The field `last` of `CeraStudy`.

    Raises
    ------
    ValueError
        If the allocation of the baseline election or of an election with a scrutiny differs from its official
        seats.
    """
    results = inputs['results']
    for date in sorted({params['election']['baseline'], *inputs['scrutinies']}):
        errors = check_official(results[date])
        if errors:
            raise ValueError(f"cera: D'Hondt does not reproduce the official seats of {date}: {'; '.join(errors)}")

    return {'last': last_seat(results[params['election']['baseline']])}


def _census_block(params: dict, inputs: dict) -> dict:
    """
    The series, the growth in two periods, the growth to the convocation census and the trend break (question 1).

    Parameters
    ----------
    params : dict
        The parameters of the study.
    inputs : dict
        The output of `_read_inputs`.

    Returns
    -------
    dict
        The fields `marks`, `baseline_date`, `baseline`, `periods_article`, `periods_now`, `growth`,
        `election_growth` and `regression` of `CeraStudy`.
    """
    long, countries, names = inputs['long'], inputs['countries'], inputs['province_names']
    baseline_census = int(inputs['results'][params['election']['baseline']].totals['registered_cera'].sum())
    at = baseline_date(inputs['series'], baseline_census)
    periods = {'article': params['periods']['article']}
    periods['now'] = update_periods(periods['article'], inputs['series_last'])

    return {
        'marks': MARKS,
        'baseline_date': at,
        'baseline': long.loc[long['date'] == at, ['region_id', 'country_code', 'electors']].reset_index(drop=True),
        'periods_article': periods['article'],
        'periods_now': periods['now'],
        'growth': {
            group: {
                version: two_periods(long, version_periods, key, countries, names)
                for version, version_periods in periods.items()
            }
            for group, key in GROWTH_KEYS.items()
        },
        'election_growth': {
            group: election_growth(long, inputs['census'], at, key, countries, names)
            for group, key in ELECTION_KEYS.items()
        },
        'regression': trend_break(inputs['series'])
    }


def _behaviour_block(data: FileSystem, params: dict, inputs: dict, census: dict, last: pd.DataFrame) -> dict:
    """
    Who votes from abroad, whom they vote for and where the new electors register (question 2).

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/` (consulate files and `cera/autonomicas.csv`).
    params : dict
        The parameters of the study.
    inputs : dict
        The output of `_read_inputs`.
    census : dict
        The output of `_census_block`.
    last : pd.DataFrame
        The last seat of the baseline election.

    Returns
    -------
    dict
        The fields `participation`, `generales`, `autonomics`, `ratios`, `ecological`, `sensitive`,
        `acceleration` and `madrid` of `CeraStudy`.
    """
    long, countries, results = inputs['long'], inputs['countries'], inputs['results']
    scrutiny = params['baseline_scrutiny']
    base = results[params['election']['baseline']]
    magnitudes = base.totals['n_seats']
    law = census['periods_article'][1][0]
    ends = {'article': census['periods_article'][1][1], 'now': params['election']['census_closed']}
    periods = {'article': census['periods_article'], 'now': census['periods_now']}
    every_country = list(countries['name'])

    return {
        'participation': {
            'origin': participation_processes(data, countries),
            'generales': participation_generales(results)
        },
        'generales': generales_processes(inputs['cera'], results, inputs['blocks']),
        'autonomics': autonomic_processes(read_autonomicas(data.get_path(AUTONOMICAS))),
        'ratios': representation_ratios(inputs['cera'][scrutiny], base, base.siglas, RATIO_SIGLAS),
        'ecological': ecological(
            inputs['cera'][scrutiny], inputs['voters'][scrutiny], base.siglas, inputs['blocks'], long,
            census['baseline_date'], countries, base.names, params['ecological_min_voters']
        ),
        'sensitive': {
            version: {
                'start': law,
                'end': end,
                'countries': sensitive_share(long, countries, SENSITIVE_COUNTRIES, magnitudes, law, end,
                                             params['sensitive_magnitude']),
                'all': sensitive_share(long, countries, every_country, magnitudes, law, end,
                                       params['sensitive_magnitude'])
            }
            for version, end in ends.items()
        },
        'acceleration': {
            version: acceleration(census['growth']['provinces'][version], magnitudes,
                                  last.set_index('region_id')['gap'])
            for version in periods
        },
        'madrid': {
            version: madrid_magnet(long, countries, params['madrid_countries'], version_periods)
            for version, version_periods in periods.items()
        }
    }


def _scenario(census: pd.DataFrame, baseline: pd.DataFrame, countries: pd.DataFrame, params: dict,
              split: pd.DataFrame, results: Results, blocks: dict, last: pd.DataFrame) -> dict:
    """
    The increment model of one census scenario over the baseline election.

    Parameters
    ----------
    census : pd.DataFrame
        The census of the scenario by province and country.
    baseline : pd.DataFrame
        The census of the baseline election by province and country.
    countries : pd.DataFrame
        The countries table.
    params : dict
        The parameters of the study (`participation`).
    split : pd.DataFrame
        The empirical split of the new votes.
    results : Results
        The results of the baseline election.
    blocks : dict
        Block name to the list of its siglas.
    last : pd.DataFrame
        The last seat of the baseline election.

    Returns
    -------
    dict
        `census`, `new`, `extra`, `model` and `net`.
    """
    participation = params['participation']
    new = new_electors(census, baseline, countries)
    extra = increment_votes(new, participation, split, valid=participation['valid'])
    model = increment_model(results, extra, results.siglas, blocks)

    return {
        'census': census, 'new': new, 'extra': extra, 'model': model,
        'net': net_effect(last, extra, model['seats_after'], model['seats_before'])
    }


def _seats_block(params: dict, inputs: dict, census: dict, last: pd.DataFrame) -> dict:
    """
    The increment model in both census scenarios, the sensitivity grids and the «23 seats» replication (question 3).

    Parameters
    ----------
    params : dict
        The parameters of the study.
    inputs : dict
        The output of `_read_inputs`.
    census : dict
        The output of `_census_block`.
    last : pd.DataFrame
        The last seat of the baseline election.

    Returns
    -------
    dict
        The fields `split`, `scenarios`, `grid`, `scenario_rows`, `placement`, `hist` and `report` of `CeraStudy`.
    """
    long, countries, blocks = inputs['long'], inputs['countries'], inputs['blocks']
    convocation, baseline = inputs['census'], census['baseline']
    results = inputs['results'][params['election']['baseline']]
    cera = inputs['cera'][params['split_source']]
    split = empirical_split(cera)
    law = census['periods_article'][1][0]
    no_litigation = litigation_census(convocation, long, countries, params['litigation']['count'], since=law)
    scenario = {CONVOCATION: convocation, NO_LITIGATION: no_litigation}

    pool = placement_pool(long, convocation, countries, params['placement_countries'], law)
    placement = optimal_placement(results, pool, PLACEMENT_PARTICIPATIONS, PLACEMENT_FRACS, PLACEMENT_CANDIDATURE,
                                  split=split)
    hist = participation_history(inputs['results'])

    return {
        'split': split,
        'scenarios': {
            name: _scenario(frame, baseline, countries, params, split, results, blocks, last)
            for name, frame in scenario.items()
        },
        'grid': exterior_grid(results, inputs['cera'][params['baseline_scrutiny']], inputs['census_by_province'],
                              GRID_PARTICIPATIONS, GRID_LEFT_FRACS, results.siglas, blocks),
        'scenario_rows': scenario_table(results, baseline, convocation, countries, params['participation'], split,
                                        results.siglas, blocks, SCENARIO_ROWS),
        'placement': {**placement, 'pool': int(pool.sum())},
        'hist': hist,
        'report': report_replication(results, inputs['census_by_province'], hist)
    }


def _checks_block(params: dict, inputs: dict) -> dict:
    """
    The check with and without the CERA vote of every election with a scrutiny, and the input of the verdict.

    Parameters
    ----------
    params : dict
        The parameters of the study.
    inputs : dict
        The output of `_read_inputs`.

    Returns
    -------
    dict
        The fields `checks` and `verdict_input` of `CeraStudy`.
    """
    results, blocks = inputs['results'], inputs['blocks']
    checks = {
        date: with_without(results[date], cera, results[date].siglas, blocks, voters=inputs['voters'][date])
        for date, cera in inputs['cera'].items()
    }
    check = checks.get(params['election']['date'])
    verdict_input = None if check is None else {
        'net_psoe': check['net_by_sigla'][VERDICT_SIGLA], 'net_left': check['net_by_block'][VERDICT_BLOCK]
    }

    return {'checks': checks, 'verdict_input': verdict_input}


def compute_cera(data: FileSystem, params: dict | None = None) -> CeraStudy:
    """
    Compute the whole CERA study from its inputs in `data/`.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    params : dict, optional
        The parameters of the study; `cera/params.json` (`data.load_params`) if `None`.

    Returns
    -------
    CeraStudy
        Every result of the study.

    Raises
    ------
    ValueError
        If D'Hondt does not reproduce the official seats of the baseline election or of an election with a CERA
        scrutiny.
    """
    params = load_params(data) if params is None else params
    inputs = _read_inputs(data, params)
    dhondt = _dhondt_block(params, inputs)
    census = _census_block(params, inputs)
    behaviour = _behaviour_block(data, params, inputs, census, dhondt['last'])
    seats = _seats_block(params, inputs, census, dhondt['last'])
    checks = _checks_block(params, inputs)

    return CeraStudy(params=params, **inputs, **dhondt, **census, **behaviour, **seats, **checks)
