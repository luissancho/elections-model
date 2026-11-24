"""La comprobación con y sin CERA: qué escaños decidió el voto exterior y la regla del veredicto."""
import pandas as pd
import pytest

from mtpy.lib.studies.cera.check import majority_outcome, verdict, with_without
from mtpy.lib.studies.cera.data import Results, read_scrutiny
from tests.test_cera_seats import study_inputs  # noqa: F401 (the module-scoped fixture shared with the seat tests)

OVER_GAP_2023 = ['Asturias', 'Cantabria', 'Gipuzkoa', 'Girona', 'Madrid', 'Málaga', 'Salamanca',
                 'Santa Cruz de Tenerife', 'Tarragona']
# the blocks of the real snapshot (cera/majority.json) and the same without UP and SALF, which did not stand in 2023
SNAPSHOT_BLOCKS = {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP', 'VOX', 'SALF'],
                   'Otros': ['ERC', 'JxCat', 'EHB', 'PNV', 'BNG', 'CC', 'AA']}
BLOCKS_2023 = SNAPSHOT_BLOCKS | {'Izquierda': ['PSOE', 'FA'], 'Derecha': ['PP', 'VOX']}


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


def test_majority_outcome_counts_every_seat_of_the_blocks(study_inputs):
    """Los escaños de los dos bloques con y sin el voto exterior, con los bloques de la foto (2023: Madrid pasa un
    escaño del PSOE al PP). Dos siglas suman a un partido: el escaño de UPN va al PP (PP 137 + UPN 1 + VOX 33 = 171) y
    los 31 de SUMAR a FA (PSOE 121 + 31 = 152); los otros 27 son de Otros, cuyo AA no se presentó y no hace falta."""
    s = study_inputs
    out = majority_outcome(s.results_2023, s.cera_2023, BLOCKS_2023, 176, aliases={'SUMAR': 'FA', 'UPN': 'PP'})
    assert out == {'right': 171, 'left': 152, 'right_without': 170, 'left_without': 153, 'has_majority': False,
                   'majority_without': False, 'changed': False, 'moved': 1, 'net': {'Derecha': 1, 'Izquierda': -1}}
    low = majority_outcome(s.results_2023, s.cera_2023, BLOCKS_2023, 171, aliases={'SUMAR': 'FA', 'UPN': 'PP'})
    assert (low['has_majority'], low['majority_without'], low['changed']) == (True, False, True)


def test_majority_outcome_sends_a_sigla_aliased_to_a_dash_to_the_rest(study_inputs):
    """`"-"` es una candidatura que el modelo no simula: el escaño de UPN cuenta en el resto, no en la derecha."""
    s = study_inputs
    out = majority_outcome(s.results_2023, s.cera_2023, BLOCKS_2023, 176, aliases={'SUMAR': 'FA', 'UPN': '-'})
    assert (out['right'], out['left'], out['right_without'], out['left_without']) == (170, 152, 169, 153)


def test_majority_outcome_refuses_a_seat_winning_sigla_that_stands_for_no_party(study_inputs):
    """Sin alias, SUMAR (31) y UPN (1) ganan escaños sin ser partidos del modelo: se rechazan con sus escaños en un
    solo mensaje, que dice dónde mapearlos; `aliases=None` vale como `{}`."""
    s = study_inputs
    for aliases in ({}, None):
        with pytest.raises(ValueError) as refused:
            majority_outcome(s.results_2023, s.cera_2023, BLOCKS_2023, 176, aliases=aliases)
        message = str(refused.value)
        assert 'SUMAR (31), UPN (1)' in message and 'majority.aliases' in message and 'data/cera/params.json' in message
        assert '"-"' in message and 'FA' in message  # FA, a party of the left, has no sigla either


def test_majority_outcome_refuses_an_alias_to_no_party(study_inputs):
    """Un alias con el destino mal escrito no se descarta en silencio: el mensaje nombra la sigla y el destino."""
    s = study_inputs
    with pytest.raises(ValueError, match=r'UPN → Pp'):
        majority_outcome(s.results_2023, s.cera_2023, BLOCKS_2023, 176, aliases={'SUMAR': 'FA', 'UPN': 'Pp'})


def test_majority_outcome_refuses_the_block_parties_missing_from_the_results(study_inputs):
    """Con los bloques reales de la foto faltan SALF y UP, que no se presentaron en 2023: se nombran los dos, en el
    orden de los bloques (derecha, izquierda)."""
    s = study_inputs
    with pytest.raises(ValueError, match=r'no candidature of the results stands for SALF, UP'):
        majority_outcome(s.results_2023, s.cera_2023, SNAPSHOT_BLOCKS, 176, aliases={'SUMAR': 'FA', 'UPN': 'PP'})


def test_majority_outcome_refuses_seats_that_do_not_add_up_to_the_chamber(study_inputs):
    """Un partido en dos bloques contaría dos veces sus escaños: derecha + izquierda + resto ya no da la cámara."""
    s = study_inputs
    blocks = dict(BLOCKS_2023, Izquierda=['PSOE', 'FA', 'VOX'])
    with pytest.raises(ValueError, match=r'do not add up to the 350 seats'):
        majority_outcome(s.results_2023, s.cera_2023, blocks, 176, aliases={'SUMAR': 'FA', 'UPN': 'PP'})


def offsetting_results():
    """Dos provincias de un escaño en las que el voto exterior decide uno en cada sentido: A (PSOE) gana la 1 sólo con
    él y B (PP) la 2."""
    index = pd.Index([1, 2], name='region_id')
    votes = pd.DataFrame({'A': [100, 99], 'B': [99, 100]}, index=index)
    totals = pd.DataFrame({'candidates': [199, 199], 'blank': [0, 0], 'n_seats': [1, 1]}, index=index)
    results = Results(date='2000-01-01', votes=votes, seats=votes * 0, totals=totals,
                      names=pd.Series({1: 'Uno', 2: 'Dos'}), siglas={'A': 'PSOE', 'B': 'PP'})
    cera = pd.DataFrame({'A': [2, 0], 'B': [0, 2]}, index=index)

    return results, cera


def test_majority_outcome_moves_count_the_seats_province_by_province():
    """Dos provincias que se compensan: el saldo nacional es 0, pero el voto exterior decidió 2 escaños, los mismos
    que cuenta la comprobación provincia a provincia."""
    results, cera = offsetting_results()
    out = majority_outcome(results, cera, {'Derecha': ['PP'], 'Izquierda': ['PSOE']}, 2)
    assert out['moved'] == 2 and out['net'] == {'Derecha': 0, 'Izquierda': 0}
    assert (out['right'], out['left'], out['right_without'], out['left_without']) == (1, 1, 1, 1)
    assert with_without(results, cera, results.siglas, {'Derecha': ['PP'], 'Izquierda': ['PSOE']})['decided'] == [1, 2]


def test_majority_outcome_moves_one_seat_in_2019_and_2023(study_inputs):
    """En 2019 (Bizkaia, del PNV al PP) y en 2023 (Madrid, del PSOE al PP) el voto exterior decidió un escaño."""
    s = study_inputs
    aliases_2019 = {'MP': 'FA', 'NA+': 'PP', 'Cs': '-', 'CUP': '-', 'TE': '-', 'PRC': '-'}
    out_2019 = majority_outcome(s.results_2019, s.cera_2019, SNAPSHOT_BLOCKS | {'Derecha': ['PP', 'VOX']}, 176,
                                aliases=aliases_2019)
    assert (out_2019['right'], out_2019['left'], out_2019['right_without'], out_2019['left_without']) == \
        (143, 158, 142, 158)
    assert out_2019['moved'] == 1 and out_2019['net'] == {'Derecha': 1, 'Izquierda': 0}
    out_2023 = majority_outcome(s.results_2023, s.cera_2023, BLOCKS_2023, 176, aliases={'SUMAR': 'FA', 'UPN': 'PP'})
    assert out_2023['moved'] == 1
