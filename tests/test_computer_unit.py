"""Tests del Computer que no necesitan base de datos (M6: descomposición del error y resumen de efectos de casa)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.computer import Computer


def test_block_error_decomposition_is_additive():
    # Errores por partido (puntos) de dos encuestas; bloques Derecha = PP+VOX, Izquierda = PSOE+SUMAR
    errors = pd.DataFrame({'PP': [2., 1.], 'VOX': [-2., 1.], 'PSOE': [1., -3.], 'SUMAR': [-1., 0.], 'ERC': [0.5, 0.5]})
    blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'SUMAR']}
    out = Computer.block_error_decomposition(errors, blocks)
    assert list(out.columns) == ['error_main', 'error_between', 'error_within']
    # Encuesta 1: los errores se compensan dentro de cada bloque -> todo es "dentro"
    assert out.loc[0].tolist() == pytest.approx([6., 0., 6.])
    # Encuesta 2: |1+1| + |-3+0| = 5 entre bloques; 5 en total -> nada dentro
    assert out.loc[1].tolist() == pytest.approx([5., 5., 0.])
    # ERC no está en ningún bloque: no cuenta; siempre main = between + within
    assert np.allclose(out['error_main'], out['error_between'] + out['error_within'])


def test_within_block_bias_measures_only_the_split_between_allies():
    polls = pd.DataFrame({'PP': [30., 33.], 'VOX': [15., 12.], 'PSOE': [28., 28.], 'SUMAR': [12., 12.]})
    events = pd.DataFrame({'PP': [33., 33.], 'VOX': [12., 12.], 'PSOE': [28., 28.], 'SUMAR': [12., 12.]})
    blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'SUMAR']}
    within = Computer.within_block_bias(polls, events, blocks)
    # Encuesta 2 clava el resultado: sesgo 0; encuesta 1 reparte mal la derecha (misma suma): sesgo > 0
    assert within.iloc[1] == pytest.approx(0.)
    assert within.iloc[0] > 0
    # El sesgo dentro del bloque no cambia si el bloque entero sube o baja proporcionalmente
    scaled = polls.copy()
    scaled[['PP', 'VOX']] *= 1.1
    assert Computer.within_block_bias(scaled, events, blocks).iloc[0] == pytest.approx(within.iloc[0])


def test_house_effects_summary_decays_and_sums_blocks():
    df = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-04-28'] * 2 + ['2023-07-23'] * 2 + ['2027-08-22'] * 2),
        'pollster_id': [1] * 6, 'pollster': ['CIS'] * 6, 'party': ['PP', 'PSOE'] * 3,
        'dev_result_c': [-1., 1., -3., 3., np.nan, np.nan], 'n_result': [10, 10, 10, 10, np.nan, np.nan],
        'dev_cycle': [-1.5, 1.2, -3., 4.5, -6., 4.5], 'n_cycle': [20, 20, 30, 30, 35, 35], 'level': [20., 28., 33., 31.7, np.nan, np.nan]
    })
    blocks = {'Derecha': ['PP'], 'Izquierda': ['PSOE']}
    out = Computer.house_effects_summary(df, blocks, current='2027-08-22', year_decay=0.9)
    cis = out.loc['CIS']
    assert set(out.columns) >= {'n_events', 'hist', 'trend', 'cycle'}
    # Histórico: media decaída de -1 (2019) y -3 (2023) para el PP: entre ambas y más cerca de 2023
    assert -3 < cis.loc['PP', 'hist'] < -2
    assert cis.loc['PP', 'trend'] < 0                     # el sesgo se acentúa con el tiempo
    assert cis.loc['PP', 'cycle'] == pytest.approx(-6.)   # desviación del ciclo actual
    assert cis.loc['PP', 'n_events'] == 2
    # Bloques: suma de los partidos del bloque
    assert cis.loc['Derecha', 'hist'] == pytest.approx(cis.loc['PP', 'hist'])
    assert cis.loc['Izquierda', 'cycle'] == pytest.approx(4.5)


def test_within_block_bias_ignores_parties_the_poll_does_not_report():
    """Una encuesta que no publica a un partido del bloque no debe ser penalizada por ello (revisión M6)."""
    polls = pd.DataFrame({'PP': [33.], 'VOX': [12.], 'PSOE': [28.], 'UP': [12.], 'MP': [np.nan]})
    events = pd.DataFrame({'PP': [33.], 'VOX': [12.], 'PSOE': [28.], 'UP': [12.], 'MP': [2.4]})
    blocks = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'UP', 'MP']}
    # El reparto entre los partidos que sí publica es exacto: sesgo 0
    assert Computer.within_block_bias(polls, events, blocks).iloc[0] == pytest.approx(0.)


# --- M7: razón de varianza de la suma (composición) ---

def test_composition_ratio_from_industry_errors():
    industry = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-04-28'] * 3 + ['2023-07-23'] * 3), 'party': ['A', 'B', 'C'] * 2,
        'industry': [2., -1., -1., 1., 1., -3.]
    })
    ref = pd.Timestamp('2027-08-22')
    # Sin decaimiento: S = (0, -1), Q = (6, 11) -> r = 1 / 17
    assert Computer.composition_ratio(industry, ref, year_decay=1.0, min_events=2) == pytest.approx(1 / 17)
    w = [0.9 ** ((ref - d).days / 365.25) for d in pd.to_datetime(['2019-04-28', '2023-07-23'])]
    assert Computer.composition_ratio(industry, ref, year_decay=0.9, min_events=2) == pytest.approx(w[1] * 1 / (w[0] * 6 + w[1] * 11))
    # Acotada a [0,05, 1]; pocas elecciones -> 1 con aviso
    same = industry.assign(industry=[2., 2., 2.] * 2)
    assert Computer.composition_ratio(same, ref, min_events=2) == 1.0
    with pytest.warns(UserWarning):
        assert Computer.composition_ratio(industry, ref, min_events=3) == 1.0
    # Errores exactamente nulos: sin información, 1 (no NaN)
    zeros = industry.assign(industry=0.)
    assert Computer.composition_ratio(zeros, ref, min_events=2) == 1.0
