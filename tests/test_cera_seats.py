"""El reparto D'Hondt del estudio CERA: validación oficial, último escaño, bloques y participación histórica."""
import os

from types import SimpleNamespace

import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera.census import baseline_date
from mtpy.lib.studies.cera.data import (
    Results,
    load_params,
    national_series,
    read_candidatures_map,
    read_countries,
    read_election_census,
    read_monthly,
    read_provinces,
    read_results,
    read_scrutiny,
    scrutiny_votes
)
from mtpy.lib.studies.cera.seats import (
    THRESHOLD,
    allocate_results,
    block_seats,
    check_official,
    increment_model,
    increment_votes,
    last_seat,
    litigation_census,
    net_effect,
    new_electors,
    participation_history,
    projected_census
)
from mtpy.lib.studies.cera.stress import (
    exterior_grid,
    optimal_placement,
    peninsular_results,
    placement_pool,
    report_replication,
    scenario_table
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CERA_CENSUS_2023 = 2328260
PSOE = 'PARTIDO SOCIALISTA OBRERO ESPAÑOL'


def _cera_votes(data_fs, date):
    """CERA votes by province and candidature full name of one election."""
    scrutiny = read_scrutiny(data_fs.get_path(f'cera/escrutinio/es-{date}.csv'))
    return scrutiny_votes(scrutiny, read_candidatures_map(data_fs)[date])


@pytest.fixture(scope='module')
def study_inputs():
    """Every input of the seat and behaviour tests of the study, read once per module."""
    data_fs = FileSystem(os.path.join(ROOT, 'data'))
    params = load_params(data_fs)
    provinces = read_provinces(data_fs)
    long = read_monthly(data_fs, provinces)
    series = national_series(long)
    at = baseline_date(series, CERA_CENSUS_2023)
    baseline = long.loc[long['date'] == at, ['region_id', 'country_code', 'electors']].reset_index(drop=True)
    census = read_election_census(data_fs.get_path('cera/ine/elecgen2026_tab8.xlsx'), provinces)
    results_2023 = read_results(data_fs, '2023-07-23')
    results_2019 = read_results(data_fs, '2019-11-10')
    cera_2023 = _cera_votes(data_fs, '2023-07-23')

    return SimpleNamespace(
        data_fs=data_fs,
        params=params,
        countries=read_countries(data_fs.get_path('cera/countries.csv')),
        provinces=provinces,
        long=long,
        series=series,
        baseline_date=at,
        baseline=baseline,
        census=census,
        census_by_province=census.groupby('region_id')['electors'].sum(),
        results_2023=results_2023,
        results_2019=results_2019,
        cera_2023=cera_2023,
        cera_2019=_cera_votes(data_fs, '2019-11-10'),
        siglas=results_2023.siglas,
        siglas_2019=results_2019.siglas,
        blocks=params['blocks'],
        split=cera_2023.div(cera_2023.sum(axis=1), axis=0),
        magnitudes=results_2023.totals['n_seats'],
        hist=participation_history({'2019-11-10': results_2019, '2023-07-23': results_2023})
    )


@pytest.fixture(scope='module')
def params(study_inputs):
    return study_inputs.params


@pytest.fixture(scope='module')
def results_2023(study_inputs):
    return study_inputs.results_2023


@pytest.fixture(scope='module')
def results_2019(study_inputs):
    return study_inputs.results_2019


@pytest.fixture(scope='module')
def cera_2023(study_inputs):
    return study_inputs.cera_2023


@pytest.fixture(scope='module')
def cera_2019(study_inputs):
    return study_inputs.cera_2019


@pytest.fixture(scope='module')
def siglas(study_inputs):
    return study_inputs.siglas


@pytest.fixture(scope='module')
def siglas_2019(study_inputs):
    return study_inputs.siglas_2019


@pytest.fixture(scope='module')
def blocks(study_inputs):
    return study_inputs.blocks


def test_study_inputs_are_consistent(study_inputs):
    s = study_inputs
    assert s.baseline_date.strftime('%Y-%m-%d') == '2023-06-01' and list(s.baseline.columns) == [
        'region_id', 'country_code', 'electors'
    ]
    assert s.baseline['electors'].sum() == s.series[s.baseline_date] and s.census_by_province.sum() == 2769645
    assert s.cera_2023.values.sum() == 203152 and (s.split.sum(axis=1).round(9) == 1).all()
    assert s.magnitudes.sum() == 350 and len(s.hist) == 52 and (s.hist['max'] >= s.hist['mean']).all()


def test_dhondt_reproduces_the_official_350_seats(results_2023, results_2019):
    assert check_official(results_2023) == [] and check_official(results_2019) == []


def test_allocate_results_recomputes_the_threshold_with_other_votes(results_2023):
    seats = allocate_results(results_2023)
    assert THRESHOLD == 3.0 and seats.values.sum() == 350 and list(seats.columns) == list(results_2023.votes.columns)
    votes = results_2023.votes.copy()
    votes.loc[28, 'PARTIDO POPULAR'] += 10 ** 8
    assert allocate_results(results_2023, votes).loc[28, 'PARTIDO POPULAR'] == results_2023.totals.loc[28, 'n_seats']


def test_allocate_results_breaks_a_quotient_tie_by_the_larger_total():
    # 2 seats: B takes the first (200); A's 100 ties B's second quotient (200 / 2) and the larger total wins.
    votes = pd.DataFrame({'A': [100], 'B': [200]}, index=pd.Index([1], name='region_id'))
    totals = pd.DataFrame({'candidates': [300], 'blank': [0], 'n_seats': [2]}, index=votes.index)
    results = Results(
        date='2000-01-01', votes=votes, seats=votes * 0, totals=totals, names=pd.Series({1: 'X'}), siglas={}
    )
    assert allocate_results(results).loc[1].to_dict() == {'A': 0, 'B': 2}


def test_last_seat_gaps_match_table_10(results_2023):
    last = last_seat(results_2023).set_index('name')
    assert last.loc['Girona', 'gap'] == 284 and last.loc['Cantabria', 'gap'] == 403
    assert last.loc['Tarragona', 'gap'] == 1247 and last.loc['Santa Cruz de Tenerife', 'gap'] == 1565
    assert last.loc['Madrid', 'gap'] == 1339 and last.loc['Ceuta', 'gap'] == 1586 and last.loc['Málaga', 'gap'] == 2428
    assert list(last.sort_values('gap').index[:10]) == [
        'Girona', 'Cantabria', 'Tarragona', 'Madrid', 'Albacete', 'Salamanca', 'Santa Cruz de Tenerife', 'Ceuta',
        'Teruel', 'Málaga'
    ]
    assert last.loc['Cantabria', 'winner'].startswith('VOX')
    assert last.loc['Cantabria', 'runner_up'].startswith('PARTIDO POPULAR')
    assert len(last) == 52 and (last['magnitude'] >= 1).all()
    assert list(last.reset_index().columns) == ['name', 'region_id', 'magnitude', 'winner', 'runner_up', 'gap']


def test_block_seats_2023(results_2023, params):
    blocks = block_seats(results_2023.seats, results_2023.siglas, params['blocks'])
    assert blocks['Izquierda'] == 152 and blocks['Derecha'] == 170
    assert sum(blocks.values()) == 350 and blocks['Regionalista'] == 8


def test_participation_history_skips_elections_without_cera_columns(results_2019, results_2023):
    hist = participation_history({'2019-11-10': results_2019, '2023-07-23': results_2023}).set_index('region_id')
    rates = [
        100 * r.totals.loc[28, 'counted_cera'] / r.totals.loc[28, 'registered_cera']
        for r in (results_2019, results_2023)
    ]
    assert hist.loc[28, 'mean'] == pytest.approx(sum(rates) / 2) and hist.loc[28, 'max'] == pytest.approx(max(rates))
    bare = SimpleNamespace(totals=results_2019.totals.drop(columns='counted_cera'))
    only_2023 = participation_history({'2019-11-10': bare, '2023-07-23': results_2023}).set_index('region_id')
    assert only_2023.loc[28, 'mean'] == pytest.approx(rates[1]) and only_2023.loc[28, 'max'] == pytest.approx(rates[1])


def test_article_projection_adds_540k_electors_and_37k_votes_and_moves_nothing(study_inputs):
    s = study_inputs
    projected = projected_census(s.baseline, s.countries, 1.264, 1.194)
    new = new_electors(projected, s.baseline, s.countries)
    assert list(new.columns) == ['region_id', 'origin', 'electors'] and set(new['origin']) == {'latam', 'rest'}
    assert 530_000 <= new['electors'].sum() <= 550_000
    extra = increment_votes(new, {'latam': 0.05, 'rest': 0.10}, s.split)
    assert 35_000 <= extra.values.sum() <= 40_000
    out = increment_model(s.results_2023, extra, s.results_2023.siglas, s.params['blocks'])
    assert out['changed'] == [] and out['by_block']['Izquierda']['change'] == 0
    assert out['by_block']['Izquierda']['before'] == 152 and out['by_sigla']['PP'] == {
        'before': 137, 'after': 137, 'change': 0
    }
    assert out['votes_extra'] == pytest.approx(extra.values.sum())


def test_real_census_moves_nothing_in_both_scenarios(study_inputs):
    s = study_inputs
    new = new_electors(s.census, s.baseline, s.countries)
    assert new['electors'].sum() == s.census['electors'].sum() - s.baseline['electors'].sum()
    assert 430_000 <= new['electors'].sum() <= 450_000
    extra = increment_votes(new, s.params['participation'], s.split)
    assert increment_model(s.results_2023, extra, s.siglas, s.blocks)['changed'] == []
    reduced = litigation_census(s.census, s.long, s.countries, 163020)
    assert s.census['electors'].sum() - reduced['electors'].sum() == 163020 and (reduced['electors'] >= 0).all()
    assert list(reduced.columns) == list(s.census.columns)
    extra_b = increment_votes(new_electors(reduced, s.baseline, s.countries), s.params['participation'], s.split)
    out_b = increment_model(s.results_2023, extra_b, s.siglas, s.blocks)
    assert out_b['changed'] == [] and extra_b.values.sum() < extra.values.sum()
    assert litigation_census(s.census, s.long, s.countries, 0).equals(s.census)


def test_litigation_census_caps_the_cells_and_removes_exactly_the_count():
    countries = pd.DataFrame({'code': [1, 2, 3], 'region': ['latam', 'latam', 'europe']})
    census = pd.DataFrame({
        'region_id': [10, 10, 10, 20], 'country_code': [1, 2, 3, 1], 'electors': [30, 10, 500, 1000]
    })
    long = pd.DataFrame({
        'date': pd.to_datetime(['2022-10-01'] * 3 + ['2026-09-01'] * 3),
        'region_id': [10, 20, 10] * 2, 'country_code': [1, 1, 3] * 2, 'electors': [0, 0, 0, 900, 100, 7]
    })
    # Province 10 would take 90 % of 100 by growth but holds only 40 latam electors: the rest goes to 20.
    reduced = litigation_census(census, long, countries, 100).set_index(['region_id', 'country_code'])['electors']
    assert reduced.to_dict() == {(10, 1): 0, (10, 2): 0, (10, 3): 500, (20, 1): 940}


def test_net_effect_signs(study_inputs):
    s = study_inputs
    extra = increment_votes(new_electors(s.census, s.baseline, s.countries), s.params['participation'], s.split)
    out = increment_model(s.results_2023, extra, s.siglas, s.blocks)
    net = net_effect(last_seat(s.results_2023), extra, out['seats_after'], out['seats_before']).set_index('name')
    assert net.loc['Cantabria', 'net'] > 0 and net.loc['Madrid', 'net'] < 0 and not net['flipped'].any()
    cantabria = net.loc['Cantabria']
    assert cantabria['net'] == pytest.approx(cantabria['d_runner'] - cantabria['d_winner'])
    assert cantabria['participation_needed'] == pytest.approx(cantabria['gap'] / cantabria['net'])
    assert net.loc['Madrid', 'participation_needed'] is None


def test_peninsular_results_take_the_cera_votes_out(study_inputs):
    s = study_inputs
    peninsular = peninsular_results(s.results_2023, s.cera_2023)
    removed = s.results_2023.votes.values.sum() - peninsular.votes.values.sum()
    # The 70 Asturias votes of ESPAÑA VACIADA belong to ASTURIAS EXISTE-ESPAÑA VACIADA: that cell floors at 0.
    assert removed == s.cera_2023.values.sum() - 70 and (peninsular.votes >= 0).all().all()
    totals = peninsular.totals
    assert (totals['valid'] == totals['candidates'] + totals['blank']).all()
    assert (totals['candidates'] == peninsular.votes.sum(axis=1)).all()
    assert peninsular.seats is s.results_2023.seats and list(peninsular.votes.columns) == list(
        s.results_2023.votes.columns
    )


def test_exterior_grid_is_monotone_and_zero_at_the_observed_cell_with_real_split(study_inputs):
    s = study_inputs
    grid = exterior_grid(
        s.results_2023, s.cera_2023, s.census_by_province, [0.09, 0.2, 0.3, 0.5, 0.7], [0.5, 0.6, 0.7, 0.8],
        s.siglas, s.blocks
    )
    assert grid.shape == (5, 4) and (grid.diff().fillna(0) >= 0).all().all()
    assert (grid.T.diff().fillna(0) >= 0).all().all()
    assert grid.loc[0.09, 0.5] <= 1 and grid.loc[0.7, 0.8] >= 10
    assert grid.values.tolist() == [[1, 1, 1, 1], [1, 1, 5, 6], [1, 4, 6, 7], [3, 7, 7, 11], [4, 7, 8, 11]]


def test_scenario_table_rows(study_inputs):
    s = study_inputs
    table = scenario_table(
        s.results_2023, s.baseline, s.census, s.countries, s.params['participation'], s.split, s.siglas, s.blocks,
        scenarios=[{'census_x': None, 'part_x': 1, 'bias': 'empirico'}, {'census_x': 2.0, 'part_x': 3, 'bias': 0.9}]
    )
    assert list(table.columns) == ['census_x', 'part_x', 'bias', 'd_left', 'd_right']
    assert table.iloc[0]['census_x'] is None and table.iloc[1]['census_x'] == 2.0
    assert (table.iloc[1]['d_left'], table.iloc[1]['d_right']) == (6, -6)
    assert table.iloc[0]['d_left'] == 0 and table.iloc[0]['d_right'] == 0 and table.iloc[1]['d_left'] >= 3


def test_optimal_placement_grows_with_the_levers_and_is_zero_without_placement(study_inputs):
    s = study_inputs
    pool = placement_pool(s.long, s.census, s.countries, s.params['placement_countries'], '2022-10-01')
    assert 150_000 <= pool.sum() <= 200_000
    out = optimal_placement(s.results_2023, pool, [0.09, 0.2, 0.5, 1.0], [0.5, 0.6, 0.7, 0.8, 1.0], PSOE)
    grid = out['grid']
    assert grid.shape == (4, 5) and (grid.diff().fillna(0) >= 0).all().all()
    assert (grid.T.diff().fillna(0) >= 0).all().all()
    # ~181.000 electors, three times the article's 62.800: 8.065 useful votes at 9 % / 50 % buy the five cheapest
    # PSOE seats (Tarragona, Madrid, Girona, Salamanca, Ceuta), where the article's pool bought two.
    assert grid.loc[0.09, 0.5] == 5 and grid.loc[1.0, 1.0] == 24
    assert grid.loc[1.0, 1.0] >= grid.loc[0.5, 0.8] >= grid.loc[0.09, 0.5] and out['contrast'] == 0
    article = optimal_placement(s.results_2023, pool * 62_800 / pool.sum(), [0.09], [0.5], PSOE, step=100,
                                split=s.split)
    assert article['grid'].loc[0.09, 0.5] == 2 and article['contrast'] == 0


def test_report_replication_counts_provinces(study_inputs):
    out = report_replication(study_inputs.results_2023, study_inputs.census_by_province, study_inputs.hist)
    assert out['n_max'] >= out['n_mean'] >= 9 and out['votes_max'] > out['votes_mean']
    # With the 2019 and 2023 rates only; the report's mean since 1986 gives ~523.000 votes and 20 provinces.
    assert (out['n_mean'], out['n_max']) == (10, 11) and 220_000 <= out['votes_mean'] <= 230_000
