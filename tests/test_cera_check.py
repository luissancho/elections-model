"""La comprobación con y sin CERA: qué escaños decidió el voto exterior y la regla del veredicto."""
from mtpy.lib.studies.cera.check import verdict, with_without
from mtpy.lib.studies.cera.data import read_scrutiny
from tests.test_cera_seats import study_inputs  # noqa: F401 (the module-scoped fixture shared with the seat tests)

OVER_GAP_2023 = ['Asturias', 'Cantabria', 'Gipuzkoa', 'Girona', 'Madrid', 'Málaga', 'Salamanca',
                 'Santa Cruz de Tenerife', 'Tarragona']


def test_with_without_cera_2023_madrid_and_nine_provinces(study_inputs):
    s = study_inputs
    out = with_without(s.results_2023, s.cera_2023, s.siglas, s.blocks)
    rows = out['rows'].set_index('name')
    assert out['n_over_gap'] == 9 and out['decided'] == [28] and out['n_decided'] == 1
    assert sorted(rows.index[rows['over_gap']]) == OVER_GAP_2023
    assert rows.loc['Madrid', 'winner_with'].startswith('PARTIDO POPULAR')
    assert rows.loc['Madrid', 'winner_without'].startswith('PARTIDO SOCIALISTA')
    assert out['net_by_sigla']['PSOE'] == -1 and out['net_by_sigla']['PP'] == 1
    assert out['net_by_block']['Izquierda'] == -1
    assert out['net_by_block']['Derecha'] == 1 and sum(out['net_by_block'].values()) == 0
    assert rows.loc['Madrid', 'crosses_block'] and rows.loc['Girona', 'over_gap'] and not rows.loc['Girona', 'decided']
    assert rows['crosses_block'].sum() == 1 and rows.loc['Madrid', 'gap'] == 1339
    assert rows.loc['Madrid', 'voters_cera'] == s.cera_2023.loc[28].sum() and len(rows) == 52
    assert list(out['rows'].columns) == [
        'region_id', 'name', 'magnitude', 'winner_with', 'winner_without', 'gap', 'voters_cera', 'over_gap',
        'decided', 'crosses_block'
    ]


def test_with_without_cera_2019_bizkaia(study_inputs):
    s = study_inputs
    out = with_without(s.results_2019, s.cera_2019, s.siglas_2019, s.blocks)
    assert out['decided'] == [48] and out['net_by_sigla']['PP'] == 1 and out['net_by_sigla']['PNV'] == -1
    assert out['n_over_gap'] == 10 and out['net_by_block']['Derecha'] == 1 and out['net_by_block']['Regionalista'] == -1
    assert out['rows'].set_index('region_id').loc[48, 'crosses_block']


def test_with_without_counts_the_scrutiny_voters_when_given(study_inputs):
    s = study_inputs
    voters = read_scrutiny(s.data_fs.get_path('cera/escrutinio/es-2023-07-23.csv'))['voters']
    out = with_without(s.results_2023, s.cera_2023, s.siglas, s.blocks, voters=voters)
    rows = out['rows'].set_index('region_id')
    assert rows.loc[28, 'voters_cera'] == 44807 and out['n_over_gap'] == 9 and out['decided'] == [28]


def test_verdict_threshold():
    assert verdict(-1, -1, 1) == verdict(0, 0, 1) == verdict(1, 1, 1) == 'acierto'
    assert verdict(2, 0, 1) == verdict(0, 2, 1) == 'fallo'
