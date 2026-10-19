"""Quién vota desde el exterior, a quién vota y dónde se inscriben los nuevos electores del CERA."""
import pandas as pd
import pytest

from mtpy.lib.studies.cera.behaviour import (
    ORIGINS,
    acceleration,
    autonomic_processes,
    available_results,
    ecological,
    generales_processes,
    left_share,
    madrid_magnet,
    participation_by_origin,
    participation_generales,
    participation_processes,
    representation_ratios,
    sensitive_share,
    shares_by_block
)
from mtpy.lib.studies.cera.census import two_periods
from mtpy.lib.studies.cera.data import read_autonomicas, read_consulates, read_results, read_scrutiny
from mtpy.lib.studies.cera.seats import last_seat
from tests.test_cera_seats import study_inputs  # noqa: F401 (the module-scoped fixture shared with the seat tests)

ARTICLE_EXCLUDED = (51, 52, 26, 31, 7, 30, 35, 15, 28)
ARTICLE_43 = [region for region in range(1, 53) if region not in ARTICLE_EXCLUDED]
ARTICLE_PERIODS = [('2019-03-01', '2022-10-01'), ('2022-10-01', '2026-05-01')]


def _results_all(data_fs):
    """The results of every general election whose Infoelectoral file has the CERA columns."""
    results = {}
    for date in available_results(data_fs):
        try:
            results[date] = read_results(data_fs, date)
        except ValueError:
            pass
    return results


def _rates(process):
    """Participation by origin of a process, rounded to one decimal."""
    return process.set_index('origin')['rate'].round(1).to_dict()


def test_participation_by_origin_cyl_2026(study_inputs):
    s = study_inputs
    consulates = read_consulates(s.data_fs.get_path('cera/consulados/cyl-2026.csv'))
    table = participation_by_origin(consulates, s.countries)
    rates = _rates(table)
    assert rates['europe'] > 2 * rates['latam']
    assert rates['europe'] == 14.4 and rates['usa_canada'] == 5.0 and rates['latam'] == 4.8
    assert list(table.columns) == ['origin', 'electors', 'voters', 'rate']
    assert table['electors'].sum() == 180234 and table['voters'].sum() == 14295
    assert list(table['origin']) == [origin for origin in ORIGINS if origin in set(table['origin'])]


def test_participation_processes_reads_every_consulate_file(study_inputs):
    s = study_inputs
    processes = participation_processes(s.data_fs, s.countries)
    assert sorted(processes) == ['aragon-2026', 'cyl-2026', 'extremadura-2023', 'extremadura-2025']
    aragon, extremadura = _rates(processes['aragon-2026']), _rates(processes['extremadura-2025'])
    assert aragon['europe'] == 15.8 and aragon['usa_canada'] == 5.1 and aragon['latam'] == 4.0
    assert aragon['europe'] > 2 * aragon['latam'] and aragon['usa_canada'] > aragon['latam']
    assert extremadura['europe'] > 2 * extremadura['latam']
    assert extremadura['europe'] == 9.2 and extremadura['latam'] == 3.2
    assert _rates(processes['extremadura-2023'])['latam'] == 4.3
    with pytest.raises(ValueError, match='ATLANTIS'):
        participation_by_origin(pd.DataFrame({'country': ['ATLANTIS'], 'electors': [1], 'voters': [0]}), s.countries)


def test_available_results_lists_every_general_election(study_inputs):
    dates = available_results(study_inputs.data_fs)
    assert len(dates) == 16 and dates[0] == '1977-06-15' and dates[-1] == '2023-07-23'
    assert '2019-11-10' in dates and '2019-04-28' in dates and dates == sorted(dates)


def test_participation_generales_periods(study_inputs):
    table = participation_generales(_results_all(study_inputs.data_fs)).set_index('date')
    assert list(table.columns) == ['electors', 'voters', 'rate', 'period']
    assert round(table.loc['2023-07-23', 'rate'], 1) == 9.0
    assert table.loc['2023-07-23', 'electors'] == 2328260 and table.loc['2023-07-23', 'voters'] == 208431
    assert table.loc['2019-11-10', 'period'] == 'rogado' and table.loc['2023-07-23', 'period'] == 'sin_rogado'
    assert table.index[0] == '1986-06-22' and table.loc['1986-06-22', 'period'] == 'antes'
    assert table.loc['2011-11-20', 'period'] == 'rogado' and table.loc['2008-03-09', 'period'] == 'antes'
    assert len(table) == 13


def test_participation_generales_skips_results_without_cera_columns(study_inputs):
    s = study_inputs
    bare = read_results(s.data_fs, '2019-11-10')
    bare.totals = bare.totals.drop(columns=['registered_cera', 'counted_cera'])
    table = participation_generales({'2019-11-10': bare, '2023-07-23': s.results_2023})
    assert list(table['date']) == ['2023-07-23']


def test_shares_by_block_and_left_share(study_inputs):
    s = study_inputs
    by_block = shares_by_block(s.cera_2023, s.siglas, s.blocks)
    assert list(by_block.columns) == [*s.blocks, 'Otros'] and by_block.index.name == 'region_id'
    assert by_block.values.sum() == s.cera_2023.values.sum()
    assert round(left_share(by_block), 1) == 49.7


def test_representation_ratios_2023(study_inputs):
    s = study_inputs
    table = representation_ratios(s.cera_2023, s.results_2023, s.siglas, ['PP', 'PSOE', 'SUMAR', 'VOX'])
    assert list(table.columns) == ['sigla', 'share_cera', 'share_total', 'ratio']
    ratios = table.set_index('sigla')['ratio'].round(1)
    assert ratios['SUMAR'] == 119.7 and ratios['PP'] == 99.9 and ratios['PSOE'] == 91.6 and ratios['VOX'] == 90.5
    assert round(table.set_index('sigla').loc['PP', 'share_cera'], 1) == 33.3


def test_generales_left_share(study_inputs):
    s = study_inputs
    cera = {'2019-11-10': s.cera_2019, '2023-07-23': s.cera_2023}
    results = {'2019-11-10': s.results_2019, '2023-07-23': s.results_2023}
    rows = {row['date']: row for row in generales_processes(cera, results, s.blocks)}
    assert round(rows['2019-11-10']['left_share'], 1) == 55.2 and round(rows['2023-07-23']['left_share'], 1) == 49.7
    row = rows['2023-07-23']
    assert row['margin'] == pytest.approx(2 * row['left_share'] - 100)
    assert row['premium'] == pytest.approx(row['psoe_pp_cera'] - row['psoe_pp_total'])
    assert round(row['psoe_pp_cera'], 1) == -4.0 and round(row['psoe_pp_total'], 1) == -1.4
    assert round(row['margin'], 1) == -0.5 and round(rows['2019-11-10']['premium'], 1) == 2.8


def test_autonomic_processes_cyl(study_inputs):
    rows = autonomic_processes(read_autonomicas(study_inputs.data_fs.get_path('cera/autonomicas.csv')))
    assert [row['community'] for row in rows] == ['Castilla y León', 'Andalucía', 'Aragón']
    cyl = [row for row in rows if row['community'] == 'Castilla y León'][0]
    assert round(cyl['last']['rate'], 1) == 6.7 and round(cyl['previous']['rate'], 1) == 1.9
    assert round(cyl['last']['left_share'], 1) == 48.2 and round(cyl['previous']['left_share'], 1) == 51.7
    assert cyl['last']['date'] == '2026-03-15' and cyl['previous']['date'] == '2022-02-13'
    assert round(cyl['previous']['margin'], 1) == 8.9 and round(cyl['last']['margin'], 1) == -0.8
    assert round(cyl['d_psoe'], 1) == 6.2 and round(cyl['d_other_left'], 1) == -8.3 and round(cyl['d_block'], 1) == -2.1
    assert round(cyl['last']['pp_pct'], 1) == 27.2 and round(cyl['last']['vox_pct'], 1) == 16.3


def test_ecological_is_negative_over_all_provinces_but_ceuta_and_melilla(study_inputs):
    s = study_inputs
    voters = read_scrutiny(s.data_fs.get_path('cera/escrutinio/es-2023-07-23.csv'))['voters']
    args = (s.cera_2023, voters, s.siglas, s.blocks, s.long, s.baseline_date, s.countries, s.results_2023.names,
            s.params['ecological_min_voters'])
    out = ecological(*args)
    assert out['n'] == 50 and out['r'] < -0.5 and out['slope'] < 0
    assert round(out['r'], 3) == -0.743 and round(out['slope'], 3) == -0.332
    assert [round(q, 1) for q in out['quartiles']] == [59.9, 58.2, 53.3, 44.7]
    assert {row['region_id'] for row in out['rows']} == set(range(1, 51))
    assert list(out['rows'][0]) == ['region_id', 'name', 'latam_share', 'left_share', 'census']
    assert len(out['quartiles']) == 4 and out['quartiles'][0] > out['quartiles'][-1]
    article = ecological(*args, only=ARTICLE_43)
    assert article['n'] == 43 and article['r'] == pytest.approx(-0.77, abs=0.02)
    assert round(article['r'], 3) == -0.767 and round(article['slope'], 3) == -0.344
    assert round(article['quartiles'][0], 1) == 59.8 and round(article['quartiles'][-1], 1) == 44.3
    ourense = [row for row in out['rows'] if row['region_id'] == 32][0]
    assert round(ourense['left_share'], 1) == 29.7 and round(ourense['latam_share'], 1) == 74.0


def test_sensitive_share_article_periods(study_inputs):
    s = study_inputs
    out = sensitive_share(s.long, s.countries, ['ARGENTINA', 'CUBA', 'VENEZUELA'], s.magnitudes, '2022-10-01',
                          pd.Timestamp('2026-05-01'))
    assert round(out['stock_share'], 1) == 30.0 and round(out['growth_share'], 1) == 26.7
    every = sensitive_share(s.long, s.countries, list(s.countries['name']), s.magnitudes, '2022-10-01',
                            pd.Timestamp('2026-05-01'))
    assert round(every['stock_share'], 1) == 24.1 and round(every['growth_share'], 1) == 17.4


def test_acceleration_by_province(study_inputs):
    s = study_inputs
    names = pd.read_csv(s.data_fs.get_path('es-provinces.csv')).set_index('code')['province']
    by_province = two_periods(s.long, ARTICLE_PERIODS, 'region_id', s.countries, names)
    gaps = last_seat(s.results_2023).set_index('region_id')['gap']
    out = acceleration(by_province, s.magnitudes, gaps)
    rows = pd.DataFrame(out['rows']).set_index('region_id')
    assert len(rows) == 52 and {'ratio', 'magnitude', 'g2', 'gap'} <= set(rows.columns)
    assert round(rows.loc[34, 'ratio'], 1) == 6.3 and rows.loc[34, 'g1'] == 175 and rows.loc[34, 'g2'] == 1094
    assert round(rows['ratio'].min(), 2) == 1.25 and round(rows['ratio'].median(), 2) == 2.28
    assert round(out['r_size_growth'], 2) == 0.88 and round(out['r_size_ratio'], 2) == -0.23
    assert round(out['r_gap_growth'], 2) == -0.02
    assert round(out['share_sensitive_stock'], 1) == 24.1 and round(out['share_sensitive_growth'], 1) == 17.4


def test_madrid_magnet(study_inputs):
    s = study_inputs
    rows = madrid_magnet(s.long, s.countries, s.params['madrid_countries'], ARTICLE_PERIODS)
    assert [row['label'] for row in rows] == ['Conjunto', 'MÉXICO', 'VENEZUELA']
    assert list(rows[0]) == ['label', 'stock_share', 'share_p1', 'share_p2']
    together, mexico, venezuela = rows
    assert [round(together[key], 1) for key in ('stock_share', 'share_p1', 'share_p2')] == [9.7, 36.8, 28.2]
    assert [round(mexico[key], 1) for key in ('stock_share', 'share_p1', 'share_p2')] == [27.0, 60.2, 57.6]
    assert round(venezuela['stock_share'], 1) == 11.8 and venezuela['share_p1'] is None
    assert round(venezuela['share_p2'], 1) == 44.5
