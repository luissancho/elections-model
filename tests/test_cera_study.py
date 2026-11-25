"""El estudio CERA de una vez: `compute_cera` lee todas las entradas y encadena los cálculos en el orden de la spec."""
import os

import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera import CeraStudy, compute_cera
from mtpy.lib.studies.cera import study as study_module
from mtpy.lib.studies.cera.check import with_without
from mtpy.lib.studies.cera.data import check_majority
from mtpy.lib.studies.cera.majority import seat_range
from mtpy.lib.studies.cera.scenarios import expected_table
from mtpy.lib.studies.cera.seats import allocate_results
from mtpy.lib.studies.cera.stress import peninsular_results
from tests.test_cera_check import BLOCKS_2023
from tests.test_cera_seats import study_inputs  # noqa: F401 (the module-scoped fixture shared with the seat tests)

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


def test_read_scrutinies_skips_the_elections_without_results_and_needs_their_candidatures(data_fs):
    candidatures = study_module.read_candidatures_map(data_fs)
    results = {'2019-11-10': object(), '2023-07-23': object()}
    scrutinies, skipped = study_module._read_scrutinies(data_fs, results, candidatures)
    assert sorted(scrutinies) == ['2019-11-10', '2023-07-23'] and skipped == []
    # the 29-N scrutiny lands before its PROV_02 XLSX: left out, and said so
    scrutinies, skipped = study_module._read_scrutinies(data_fs, {'2023-07-23': object()}, candidatures)
    assert sorted(scrutinies) == ['2023-07-23'] and skipped == ['2019-11-10']
    # a scrutiny with results and no entry in candidatures.json is an error, not a silent KeyError
    without = {date: mapping for date, mapping in candidatures.items() if date != '2023-07-23'}
    with pytest.raises(ValueError, match=r'cera: no candidatures map for 2023-07-23 in '
                                         r'cera/escrutinio/candidatures\.json'):
        study_module._read_scrutinies(data_fs, results, without)


def test_compute_cera_records_the_skipped_scrutinies(study):
    assert study.skipped_scrutinies == []


def test_the_real_snapshot_is_the_run_of_23_n_and_agrees_with_the_study(study):
    """La foto fija `cera/majority.json` es la del run 20261124-135758 (as_of 23-N) y cuadra con el estudio."""
    snap = study.majority
    assert {key: snap['run'][key] for key in ('scope', 'event_date', 'run_id', 'as_of', 'n_sim', 'seed')} == {
        'scope': 'es', 'event_date': '2026-11-29', 'run_id': '20261124-135758', 'as_of': '2026-11-23',
        'n_sim': 10000, 'seed': 42}
    assert snap['majority'] == 176
    assert snap['blocks'] == {
        'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA', 'UP'],
        'Otros': ['ERC', 'JxCat', 'EHB', 'PNV', 'BNG', 'CC', 'AA']}
    assert snap['base']['with_majority'] == 8807 and snap['base']['central'] == {'Derecha': 190, 'Izquierda': 129}
    assert snap['new_electors']['participations'] == snap['grid']['participations']
    expected = snap['expected']['convocatoria']
    if study.params['election']['date'] not in study.checks:
        # the frozen table is the live study's only while there are no 29-N results (spec D3): the scrutiny of the
        # election would change the live inputs, never the snapshot
        assert expected['table'] == expected_table(study, 'convocatoria')
    assert len(expected['provinces']) == 52
    counts = expected['counts']
    assert (counts['votes'], counts['lost'], counts['gained'], counts['moved']) == (30482, 8, 3, 817)
    assert counts['right'] == {'-2': 7, '-1': 345, '0': 9482, '1': 164, '2': 2}
    assert counts['central'] == {'right': 0, 'left': 0}
    lit = snap['expected']['sin_litigio']['counts']
    assert (lit['votes'], lit['lost'], lit['gained'], lit['moved']) == (22412, 6, 3, 625)
    madrid = next(row for row in expected['provinces'] if row['region_id'] == 28)
    assert (madrid['moved'], madrid['right_lost'], madrid['right_gained']) == (189, 77, 34)
    assert sum(1 for row in expected['provinces'] if row['moved']) == 49
    cells = [snap[key]['cells'][i][j] for key, i, j in (
        ('grid', 0, 0), ('grid', 2, 1), ('grid', 2, 3), ('grid', 3, 3), ('grid', 4, 3), ('new_electors', 0, 0),
        ('new_electors', 3, 3), ('placement', 3, 4), ('placement', 0, 0))]
    assert [(c['lost'], c['gained'], c['central']['right']) for c in cells] == [
        (32, 26, 0), (345, 11, -2), (1005, 0, -4), (1815, 0, -8), (2734, 0, -10), (13, 7, 0), (228, 0, -1),
        (1234, 0, -5), (84, 0, -1)]
    every = [cell for key in ('new_electors', 'grid', 'placement') for row in snap[key]['cells'] for cell in row]
    assert len(every) == 60
    assert all(seat_range(c['right'])[0] <= c['central']['right'] <= seat_range(c['right'])[1] for c in every)
    assert check_majority(snap, study.params) == []


def assert_outcome_agrees(outcome, results, cera, check, majority):
    """El resultado de una elección cuadra con su comprobación: los escaños de los dos bloques caben en la cámara,
    la mayoría es la de la foto y `moved` son los escaños que cambian provincia a provincia entre el reparto oficial y
    el reparto sin el voto exterior, en las provincias que la comprobación da por decididas."""
    chamber = int(results.totals['n_seats'].sum())
    assert outcome['right'] + outcome['left'] <= chamber
    assert outcome['right_without'] + outcome['left_without'] <= chamber
    assert outcome['has_majority'] == (outcome['right'] >= majority)
    assert outcome['majority_without'] == (outcome['right_without'] >= majority)
    differ = allocate_results(results) - allocate_results(peninsular_results(results, cera))
    assert outcome['moved'] == int(differ.clip(lower=0).sum().sum())
    assert sorted(int(region) for region in differ.index[differ.ne(0).any(axis=1)]) == check['decided']
    assert outcome['moved'] >= check['n_decided'] and (outcome['moved'] == 0) == (check['n_decided'] == 0)


def test_the_study_has_an_outcome_exactly_when_its_election_is_checked(study):
    """El resultado existe sólo cuando el estudio tiene la comprobación de su elección y entonces cuadra con ella;
    los alias de siglas son un diccionario de cadenas (vacío hoy, con las siglas del 29-N después). Vale antes y
    después de cargar el escrutinio: la rama con resultados la ejercita, con 2023 en su lugar, la prueba de
    `_outcome_of`."""
    date = study.params['election']['date']
    aliases = study.params['majority']['aliases']
    assert isinstance(aliases, dict) and all(isinstance(key, str) and isinstance(value, str)
                                             for key, value in aliases.items())
    if date not in study.checks:
        assert study.outcome is None
    else:
        assert_outcome_agrees(study.outcome, study.results[date], study.cera[date], study.checks[date],
                              study.majority['majority'])


def test_outcome_of_needs_the_check_of_the_election_and_a_snapshot(study_inputs):
    """La rama que correrá tras el 29-N, con 2023 en el lugar del 29-N y los bloques de la foto: con escrutinio y foto
    da el resultado (UPN suma al PP: 171) y cuadra con la comprobación; sin una de las dos, None."""
    s = study_inputs
    date = '2026-11-29'
    params = {'election': {'date': date}, 'majority': {'aliases': {'SUMAR': 'FA', 'UPN': 'PP'}}}
    inputs = {'results': {date: s.results_2023}, 'cera': {date: s.cera_2023}}
    checks = {date: with_without(s.results_2023, s.cera_2023, s.siglas, s.blocks)}
    snapshot = {'blocks': BLOCKS_2023, 'majority': 176}
    out = study_module._outcome_of(params, inputs, checks, snapshot)
    assert out == {'right': 171, 'left': 152, 'right_without': 170, 'left_without': 153, 'has_majority': False,
                   'majority_without': False, 'changed': False, 'moved': 1, 'net': {'Derecha': 1, 'Izquierda': -1}}
    assert_outcome_agrees(out, s.results_2023, s.cera_2023, checks[date], 176)
    assert study_module._outcome_of(params, inputs, checks, None) is None
    assert study_module._outcome_of(params, inputs, {}, snapshot) is None


def test_outcome_of_reads_params_without_aliases_as_no_aliases(study_inputs):
    """Unos parámetros sin `majority.aliases` valen como sin alias: el estudio se niega por las siglas sin mapear, no
    con un `KeyError`."""
    s = study_inputs
    date = '2026-11-29'
    inputs = {'results': {date: s.results_2023}, 'cera': {date: s.cera_2023}}
    snapshot = {'blocks': BLOCKS_2023, 'majority': 176}
    for params in ({'election': {'date': date}}, {'election': {'date': date}, 'majority': {}}):
        with pytest.raises(ValueError, match=r'SUMAR \(31\), UPN \(1\)'):
            study_module._outcome_of(params, inputs, {date: {}}, snapshot)
