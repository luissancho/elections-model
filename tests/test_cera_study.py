"""El estudio CERA de una vez: `compute_cera` lee todas las entradas y encadena los cálculos en el orden de la spec."""
import os

import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera import CeraStudy, compute_cera
from mtpy.lib.studies.cera import study as study_module

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope='module')
def data_fs():
    return FileSystem(os.path.join(ROOT, 'data'))


@pytest.fixture(scope='module')
def study(data_fs):
    return compute_cera(data_fs)


def test_compute_cera_runs_end_to_end(study):
    assert isinstance(study, CeraStudy) and study.params['election']['date'] == '2026-11-29'
    assert study.series_last == study.long['date'].max() and study.series_last >= pd.Timestamp('2026-09-01')
    assert set(study.scenarios) == {'convocatoria', 'sin_litigio'}
    assert study.scenarios['convocatoria']['model']['changed'] == []
    assert sorted(study.checks) == ['2019-11-10', '2023-07-23']
    assert study.checks['2023-07-23']['n_over_gap'] == 9
    assert study.verdict_input is None


def test_compute_cera_census_block(study):
    assert study.baseline_date == pd.Timestamp('2023-06-01') and study.series.index[-1] == study.series_last
    assert len(study.province_names) == 52 and study.province_names[28] == 'Madrid'
    assert [date for date, _ in study.marks] == ['2022-10-21', '2023-07-23', '2025-10-22', '2026-08-01']
    (start_1, end_1), (start_2, end_2) = study.periods_now   # el segundo periodo acaba en el último cierre
    assert end_1 == start_2 == pd.Timestamp('2022-10-01') and end_2 == study.series_last
    assert start_1 + pd.DateOffset(months=(end_2.year - start_2.year) * 12 + end_2.month - start_2.month) == end_1
    assert set(study.growth) == {'provinces', 'countries', 'origins', 'exposure'}
    assert all(set(by_period) == {'article', 'now'} for by_period in study.growth.values())
    assert study.growth['provinces']['article']['g2'].sum() == 422253 and len(study.growth['provinces']['now']) == 52
    assert study.growth['exposure']['article']['g2'].sum() == 422253
    assert set(study.election_growth) == {'provinces', 'countries'}
    assert study.election_growth['provinces']['election'].sum() == 2769645 == study.census['electors'].sum()
    assert study.census_by_province.sum() == 2769645 and study.regression['p_slope'] < 0.05


def test_compute_cera_behaviour_block(study):
    assert sorted(study.participation['origin']) == ['aragon-2026', 'cyl-2026', 'extremadura-2023', 'extremadura-2025']
    assert len(study.participation['generales']) == 13 and len(study.results) == 13
    assert [row['date'] for row in study.generales] == ['2019-11-10', '2023-07-23']
    assert len(study.autonomics) == 3 and list(study.ratios['sigla']) == ['PP', 'PSOE', 'SUMAR', 'VOX']
    assert study.ecological['n'] == 50 and round(study.ecological['r'], 3) == -0.743
    for version in ('article', 'now'):
        assert set(study.sensitive[version]) == {'countries', 'all', 'start', 'end'}
        assert len(study.acceleration[version]['rows']) == 52 and len(study.madrid[version]) == 3
    assert round(study.sensitive['article']['all']['stock_share'], 1) == 24.1
    assert round(study.sensitive['article']['countries']['growth_share'], 1) == 26.7


def test_compute_cera_seats_block(study):
    assert len(study.last) == 52 and set(study.cera) == set(study.checks) == set(study.voters)
    for name, scenario in study.scenarios.items():
        assert set(scenario) == {'census', 'new', 'extra', 'model', 'net'} and scenario['model']['changed'] == []
        assert len(scenario['net']) == 52
    convocatoria, sin_litigio = study.scenarios['convocatoria'], study.scenarios['sin_litigio']
    assert convocatoria['new']['electors'].sum() == 442257 and sin_litigio['new']['electors'].sum() == 279237
    assert study.grid.shape == (5, 4) and study.grid.loc[0.09, 0.5] <= 1
    assert len(study.scenario_rows) == 6 and (study.scenario_rows.iloc[0][['d_left', 'd_right']] == 0).all()
    assert study.placement['grid'].shape == (4, 5) and study.placement['contrast'] == 0
    assert study.placement['grid'].loc[0.09, 0.5] == 5 and 150_000 <= study.placement['pool'] <= 200_000
    report = study.report
    assert report['n_max'] >= report['n_mean'] and (report['n_mean'], report['n_max']) == (20, 27)
    assert 520_000 <= report['votes_mean'] <= 526_000 and 1_110_000 <= report['votes_max'] <= 1_117_000
    assert 0 < report['participation_mean'] < report['participation_max'] < 100 and len(study.hist) == 52


def test_compute_cera_checks_block(study):
    check_2023, check_2019 = study.checks['2023-07-23'], study.checks['2019-11-10']
    assert check_2023['decided'] == [28] and check_2019['decided'] == [48]
    assert check_2023['net_by_sigla']['PSOE'] == -1 and check_2023['net_by_block']['Izquierda'] == -1
    assert check_2023['rows'].set_index('region_id').loc[28, 'voters_cera'] == 44807


def test_compute_cera_fails_when_dhondt_does_not_reproduce_the_official_seats(study, monkeypatch):
    monkeypatch.setattr(study_module, 'check_official', lambda results: [f'Madrid: X 1 != 2 ({results.date})'])
    inputs = {'results': study.results, 'scrutinies': study.scrutinies}
    with pytest.raises(ValueError, match="cera: D'Hondt does not reproduce the official seats of 2019-11-10: Madrid"):
        study_module._dhondt_block(study.params, inputs)


def test_read_all_results_skips_only_the_files_without_cera_columns(data_fs, monkeypatch):
    def broken(data, date):
        raise ValueError('PROV_02_202307_1.xlsx: the candidature columns are not «Votos» / «Diputados» pairs')

    monkeypatch.setattr(study_module, 'read_results', broken)
    with pytest.raises(ValueError, match='Diputados'):
        study_module._read_all_results(data_fs)
