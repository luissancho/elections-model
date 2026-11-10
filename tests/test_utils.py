"""Tests de `mtpy/lib/utils.py` (bloques de partidos)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.utils import build_blocks, group_results


def test_party_entries_marks_only_the_parties_born_mid_cycle_with_their_inherited_labels():
    """UP y SALF entran a mitad de ciclo; FA hereda la fecha de SUMAR y PSOE entra a las dos semanas: no se marcan."""
    from mtpy.lib.utils import party_entries
    first = pd.to_datetime(pd.Series({'PP': '2023-08-07', 'VOX': '2023-08-07', 'SUMAR': '2023-08-07',
                                      'FA': '2025-05-01', 'UP': '2023-12-15', 'SALF': '2024-06-06',
                                      'PSOE': '2023-08-20'}))
    labels = {'FA': ['FA', 'SUMAR'], 'PP': ['PP', 'UPN']}
    out = party_entries(first, ['PSOE', 'FA', 'UP', 'PP', 'VOX', 'SALF'], labels)
    assert out == {'UP': '2023-12-15', 'SALF': '2024-06-06'} and list(out) == ['UP', 'SALF']
    assert party_entries(first, ['UP'], {}, min_days=200) == {}
    assert party_entries(pd.Series(dtype='datetime64[ns]'), ['UP'], {}) == {}
    # a party with a result in the previous election is not new, however late the cycle's polls list it
    assert party_entries(first, ['UP', 'SALF'], {}, known={'UP'}) == {'SALF': '2024-06-06'}
    assert party_entries(first, ['FA'], labels, known={'SUMAR'}, min_days=0) == {}


def test_first_poll_dates_reads_the_first_non_null_of_each_label():
    """La primera fecha con valor de cada etiqueta; la que nunca aparece no sale."""
    from mtpy.lib.utils import first_poll_dates
    polls = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-01', '2024-03-01']),
                          'UP': [None, None, 2.], 'PP': [30., 31., 32.]}).set_index('date')
    out = first_poll_dates(polls, ['UP', 'PP', 'CUP'])
    assert out.to_dict() == {'UP': pd.Timestamp('2024-03-01'), 'PP': pd.Timestamp('2024-01-01')}
    flat = polls.reset_index()
    assert first_poll_dates(flat, ['UP']).to_dict() == {'UP': pd.Timestamp('2024-03-01')}


def test_group_results_sums_blocks_and_ignores_non_party_columns():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'pollster': ['A', 'B'],
        'PP': [30., 31.],
        'VOX': [10., np.nan],
        'PSOE': [28., 27.],
    })
    blocks = build_blocks({'Derecha': ['PP', 'VOX'], 'PSOE': 'PSOE', 'Otro': ['XX']}, ['PP', 'VOX', 'PSOE'])
    out = group_results(df, blocks=blocks)

    assert list(out.columns) == ['Derecha', 'PSOE', 'Otro']
    assert out['Derecha'].tolist() == [40., 31.]
    assert out['PSOE'].tolist() == [28., 27.]
    assert out['Otro'].isnull().all()
    assert out.index.equals(df.index)


def test_build_blocks_missing_first_party_does_not_raise():
    """B24: un bloque cuyo primer partido no está en los sondeos no debe lanzar KeyError."""
    blocks = build_blocks({'PP': 'PP', 'X': ['CUP']}, {'PP': '#0000ff'})
    assert 'X' in blocks.index
    assert isinstance(blocks.loc['X', 'color'], str)


# --- M6: actualización normal-normal (prior + dato) ---

def test_normal_update_combines_prior_and_data_by_precision():
    from mtpy.lib.utils import normal_update
    mean, err = normal_update(np.array([2.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]))
    assert mean.tolist() == pytest.approx([1.0])            # misma precisión: la media
    assert err.tolist() == pytest.approx([np.sqrt(0.5)])
    mean, err = normal_update(np.array([2.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]))
    assert mean[0] == pytest.approx(2.0 * 4 / 5)             # el dato pesa 4 veces más


def test_normal_update_degenerate_cases_and_nan():
    from mtpy.lib.utils import normal_update
    # Sin dato (se = inf o NaN, o n insuficiente) -> el prior; sin prior (sd = inf) -> el dato
    mean, err = normal_update(np.array([2.0, 2.0, np.nan]), np.array([np.inf, np.nan, 0.5]), np.array([0.3, 0.3, 0.3]), np.array([1.0, 1.0, 1.0]))
    assert mean.tolist() == pytest.approx([0.3, 0.3, 0.3])
    assert err.tolist() == pytest.approx([1.0, 1.0, 1.0])
    mean, err = normal_update(np.array([2.0]), np.array([0.5]), np.array([0.0]), np.array([np.inf]))
    assert mean.tolist() == pytest.approx([2.0]) and err.tolist() == pytest.approx([0.5])
    # Un error típico 0 no produce NaN: se trata como dato sin incertidumbre
    mean, err = normal_update(np.array([2.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert mean.tolist() == pytest.approx([2.0]) and err.tolist() == pytest.approx([0.0])
    # Escalares
    assert normal_update(2.0, 1.0, 0.0, 1.0)[0] == pytest.approx(1.0)


def test_partial_parties_run_where_less_than_half_of_the_votes_are():
    """M11: dentro de una comunidad, un partido es "regional" si sólo concurre en circunscripciones que suman
    menos de la mitad del voto (UPL en León, Zamora y Salamanca; Por Ávila; los partidos de una isla)."""
    import numpy as np
    import pandas as pd
    from mtpy.lib.utils import partial_parties
    pct = pd.DataFrame({
        'PSOE': [30., 25., 28., np.nan], 'UPL': [20., np.nan, np.nan, np.nan], 'EV': [5., 4., np.nan, 6.],
        'NEW': [np.nan] * 4
    }, index=[1, 2, 3, 4])
    votes = pd.Series([40., 30., 20., 10.], index=[1, 2, 3, 4])
    # UPL: 40 % del voto; PSOE: 90 %; EV: 80 %; NEW no concurrió: no se marca
    assert partial_parties(pct, votes) == ['UPL']
    assert partial_parties(pct.iloc[:1], votes.iloc[:1]) == []      # distrito único


def test_build_blocks_skips_blocks_without_parties():
    """M11: en una comunidad sin partidos regionalistas ni separatistas esos bloques quedan vacíos."""
    blocks = build_blocks({'Derecha': ['PP', 'VOX'], 'Regionalista': [], 'Separatista': []}, {'PP': '#0000ff', 'VOX': '#00ff00'})
    assert blocks.index.tolist() == ['Derecha']


def test_party_roots_follow_parent_id():
    """M11b: la historia de efectos de casa de una marca regional es la de su raíz (PSC → PSOE, PDeCAT → CiU)."""
    import pandas as pd
    from mtpy.lib.utils import party_roots
    parties = pd.DataFrame({
        'id': [1, 25, 9, 20, 515, 999], 'name': ['PSOE', 'CiU', 'JxCat', 'PDeCAT', 'PSC', 'LOOP'],
        'parent_id': [None, None, 25, 25, 1, 999]
    })
    roots = party_roots(parties)
    assert roots.to_dict() == {'PSOE': 'PSOE', 'CiU': 'CiU', 'JxCat': 'CiU', 'PDeCAT': 'CiU', 'PSC': 'PSOE', 'LOOP': 'LOOP'}


# --- Swing logístico: un factor común sobre las odds de cada partido (`logistic_swing`) ---------------------

def test_logistic_swing_multiplies_the_odds_by_a_common_factor():
    """Dos provincias iguales al 20 % y al 50 % (odds 0,25 y 1) y el partido sube del 35 % al 50 %: el factor 2
    sobre las odds da 33,33 % y 66,67 % (media 50); el swing proporcional daría 28,57 % y 71,43 %."""
    from mtpy.lib.utils import logistic_swing
    prev = pd.DataFrame({'A': [35., 20., 50.]}, index=[0, 1, 2])
    out = logistic_swing(prev, pd.Series({'A': 50. / 35.}), pd.Series([2., 1., 1.], index=[0, 1, 2]), [1, 2])
    assert out['A'].tolist() == pytest.approx([50., 33.333, 66.667], abs=0.001)


def test_logistic_swing_weighs_the_districts_by_their_votes():
    """10 % con peso 3 y 40 % con peso 1 (media 17,5): con el factor 3 sobre las odds quedan en 25 % y 66,67 %,
    media ponderada 35,4167. La solución sin pesos sería otra."""
    from mtpy.lib.utils import logistic_swing
    prev = pd.DataFrame({'A': [17.5, 10., 40.]}, index=[0, 1, 2])
    out = logistic_swing(prev, pd.Series({'A': 35.416667 / 17.5}), pd.Series([4., 3., 1.], index=[0, 1, 2]), [1, 2])
    assert out.loc[[1, 2], 'A'].tolist() == pytest.approx([25., 66.667], abs=0.001)


@pytest.mark.parametrize('empty', [np.nan, 0.])
def test_logistic_swing_leaves_out_the_districts_without_base(empty):
    """Donde el partido no tiene base sigue sin tenerla, y esa provincia (aquí la de más peso) no entra en el
    total que deben sumar las demás."""
    from mtpy.lib.utils import logistic_swing
    prev = pd.DataFrame({'A': [35., 20., empty, 50.]}, index=[0, 1, 2, 3])
    out = logistic_swing(prev, pd.Series({'A': 50. / 35.}), pd.Series([7., 1., 5., 1.], index=[0, 1, 2, 3]), [1, 2, 3])
    assert out.loc[[1, 3], 'A'].tolist() == pytest.approx([33.333, 66.667], abs=0.001)
    assert (np.isnan(out.loc[2, 'A']) if np.isnan(empty) else out.loc[2, 'A'] == 0.)


def test_logistic_swing_solves_each_party_with_its_own_ratio():
    """Cada columna con su razón: B baja del 50 % al 35 % (el camino inverso del primer test), C no tiene base
    nacional (razón NaN: sin proyección, como en el proporcional) y D se queda sin votos."""
    from mtpy.lib.utils import logistic_swing
    prev = pd.DataFrame({'A': [35., 20., 50.], 'B': [50., 33.3333, 66.6667], 'C': [np.nan, 5., 6.], 'D': [4., 3., 5.]},
                        index=[0, 1, 2])
    ratios = pd.Series({'A': 50. / 35., 'B': 0.7, 'C': np.nan, 'D': 0.})
    out = logistic_swing(prev, ratios, pd.Series([2., 1., 1.], index=[0, 1, 2]), [1, 2])
    assert out['A'].tolist() == pytest.approx([50., 33.333, 66.667], abs=0.001)
    assert out['B'].tolist() == pytest.approx([35., 20., 50.], abs=0.001)
    assert out['C'].isna().all()
    assert out['D'].tolist() == [0., 0., 0.]


def test_logistic_swing_never_passes_100():
    """Feudos al 60 % y al 90 % y el partido sube un 30 %: el proporcional daría 78 % y 117 %; el logístico
    reparte el mismo total (media 97,5) sin pasar de 100 en ninguna provincia."""
    from mtpy.lib.utils import logistic_swing
    prev = pd.DataFrame({'A': [75., 60., 90.]}, index=[0, 1, 2])
    out = logistic_swing(prev, pd.Series({'A': 1.3}), pd.Series([2., 1., 1.], index=[0, 1, 2]), [1, 2])
    assert out.loc[[1, 2], 'A'].max() < 100.
    assert out.loc[[1, 2], 'A'].mean() == pytest.approx(97.5, abs=0.001)
    assert out.loc[1, 'A'] < out.loc[2, 'A']


def test_logistic_swing_keeps_the_rows_that_are_not_districts_proportional():
    """El total del ámbito (0) y la lista autonómica canaria (100) no son provincias: toman base · razón."""
    from mtpy.lib.utils import logistic_swing
    prev = pd.DataFrame({'A': [35., 36., 20., 50.]}, index=[0, 100, 1, 2])
    out = logistic_swing(prev, pd.Series({'A': 50. / 35.}), pd.Series([2., 2., 1., 1.], index=[0, 100, 1, 2]), [1, 2])
    assert out['A'].tolist() == pytest.approx([50., 51.429, 33.333, 66.667], abs=0.001)
