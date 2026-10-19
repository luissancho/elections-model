"""El reparto D'Hondt del estudio CERA: validación oficial, último escaño, bloques y participación histórica."""
import os

from types import SimpleNamespace

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera.census import baseline_date
from mtpy.lib.studies.cera.data import (
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
    last_seat,
    participation_history
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CERA_CENSUS_2023 = 2328260


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


def test_last_seat_gaps_match_table_10(results_2023):
    # The article rounds the deficit to equal the last quotient (Girona 284, Cantabria 403, Tarragona 1247,
    # Madrid 1339, S. C. de Tenerife 1565, Ceuta 1586, Málaga 2428); the gap is the votes to strictly exceed it.
    last = last_seat(results_2023).set_index('name')
    assert last.loc['Girona', 'gap'] == 285 and last.loc['Cantabria', 'gap'] == 404
    assert last.loc['Tarragona', 'gap'] == 1248 and last.loc['Santa Cruz de Tenerife', 'gap'] == 1565
    assert last.loc['Madrid', 'gap'] == 1340 and last.loc['Ceuta', 'gap'] == 1587 and last.loc['Málaga', 'gap'] == 2429
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
        100 * r.totals.loc[28, 'counted_cera'] / r.totals.loc[28, 'registered_cera'] for r in (results_2019, results_2023)
    ]
    assert hist.loc[28, 'mean'] == pytest.approx(sum(rates) / 2) and hist.loc[28, 'max'] == pytest.approx(max(rates))
    bare = SimpleNamespace(totals=results_2019.totals.drop(columns='counted_cera'))
    only_2023 = participation_history({'2019-11-10': bare, '2023-07-23': results_2023}).set_index('region_id')
    assert only_2023.loc[28, 'mean'] == pytest.approx(rates[1]) and only_2023.loc[28, 'max'] == pytest.approx(rates[1])
