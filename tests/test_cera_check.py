"""La comprobación con y sin CERA: qué escaños decidió el voto exterior y la regla del veredicto."""
import pytest

from mtpy.lib.studies.cera.check import majority_outcome, verdict, with_without
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


def test_majority_outcome_counts_the_blocks_with_and_without_the_exterior_vote(study_inputs):
    """Los escaños de los dos bloques con y sin el voto exterior (2023: Madrid pasa un escaño del PSOE al PP)."""
    blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'FA']}
    out = majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, blocks, 176, aliases={'SUMAR': 'FA'})
    assert out == {'right': 170, 'left': 152, 'right_without': 169, 'left_without': 153, 'has_majority': False,
                   'majority_without': False, 'changed': False, 'moved': 1, 'net': {'Derecha': 1, 'Izquierda': -1}}
    low = majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, blocks, 170, aliases={'SUMAR': 'FA'})
    assert (low['has_majority'], low['majority_without'], low['changed']) == (True, False, True)


def test_majority_outcome_refuses_a_block_party_missing_from_the_results(study_inputs):
    """Un partido de un bloque sin sigla en los resultados se rechaza con su nombre."""
    blocks = {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE', 'FA']}
    with pytest.raises(ValueError, match='SALF'):
        majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, blocks, 176, aliases={'SUMAR': 'FA'})
    with pytest.raises(ValueError, match='FA'):
        majority_outcome(study_inputs.results_2023, study_inputs.cera_2023, {'Derecha': ['PP'], 'Izquierda': ['FA']},
                         176)


def test_majority_outcome_reads_only_the_two_blocks_of_the_majority(study_inputs):
    """Un tercer bloque (Otros) no se busca en los resultados ni hace que el estudio se niegue."""
    blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'FA']}
    more = dict(blocks, Otros=['ERC', 'NOPE'])
    args = (study_inputs.results_2023, study_inputs.cera_2023)
    assert majority_outcome(*args, more, 176, aliases={'SUMAR': 'FA'}) == \
        majority_outcome(*args, blocks, 176, aliases={'SUMAR': 'FA'})
