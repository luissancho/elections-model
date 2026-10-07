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


# --- M9: oscilaciones autonómicas y provinciales del swing ---

def test_swing_noise_fit_recovers_level_curves():
    from mtpy.lib.computer import SwingNoise
    rng = np.random.default_rng(0)
    a_r, b_r, a_p, b_p = 0.006, 0.16, 0.002, 0.012
    rows = []
    for L in (1.5, 2.5, 4., 7., 15., 40.):
        for _ in range(600):
            rows.append({'pair': rng.choice(['a', 'b', 'c', 'd']), 'level': L,
                         'reg_mean': rng.normal(0, np.sqrt(a_r + b_r / L)), 'within': rng.normal(0, np.sqrt(a_p + b_p / L))})
    res = pd.DataFrame(rows)
    noise = SwingNoise.fit(res)
    assert noise.a_r == pytest.approx(a_r, rel=0.25) and noise.b_r == pytest.approx(b_r, rel=0.15)
    assert noise.a_p == pytest.approx(a_p, rel=0.35) and noise.b_p == pytest.approx(b_p, rel=0.35)
    assert noise.n == res.shape[0]
    # Curvas: decrecientes con el nivel, nivel acotado por abajo, siempre finitas
    assert noise.sigma_region(30.) < noise.sigma_region(10.) < noise.sigma_region(3.)
    assert noise.sigma_region(0.1) == noise.sigma_region(1.0)
    assert np.isfinite(noise.sigma_province(np.array([1., 5., 50.]))).all()
    assert noise.sigma_region(30.) == pytest.approx(np.sqrt(noise.a_r + noise.b_r / 30.))


def test_swing_noise_without_enough_pairs_is_zero():
    from mtpy.lib.computer import SwingNoise
    res = pd.DataFrame({'pair': ['a'] * 10, 'level': np.linspace(2, 30, 10), 'reg_mean': 0.1, 'within': 0.05})
    with pytest.warns(UserWarning):
        noise = SwingNoise.fit(res, min_pairs=3)
    assert noise.sigma_region(10.) == 0 and noise.sigma_province(10.) == 0


def test_swing_noise_decompose_and_fit_are_unbiased_with_small_communities():
    """M9 (revisión): la desviación dentro de una comunidad de k provincias tiene varianza σ_w²(1 − 1/k) y la media
    de la comunidad arrastra σ_w²/k; el ajuste corrige ambas cosas y excluye de la parte autonómica a los
    partidos presentes en una sola comunidad (su media autonómica es nula por construcción)."""
    from mtpy.lib.computer import SwingNoise
    rng = np.random.default_rng(1)
    a_r, b_r, a_p, b_p = 0.006, 0.16, 0.003, 0.02
    sizes = {'g1': 1, 'g2': 1, 'g3': 1, 'g4': 2, 'g5': 2, 'g6': 3, 'g7': 5}
    rows = []
    for pair in range(40):
        for party, L in (('P1', 30.), ('P2', 12.), ('P3', 7.), ('P4', 4.), ('P5', 2.5), ('P6', 1.8)):
            for g, k in sizes.items():
                eps = rng.normal(0, np.sqrt(a_r + b_r / L))
                for r in range(k):
                    rows.append({'pair': pair, 'party': party, 'group': g, 'region': '{}{}'.format(g, r), 'level': L,
                                 'logres': eps + rng.normal(0, np.sqrt(a_p + b_p / L))})
        # Partido de una sola comunidad: su residuo autonómico es nulo (la razón nacional es la suya)
        for r in range(3):
            rows.append({'pair': pair, 'party': 'R1', 'group': 'g6', 'region': 'g6{}'.format(r), 'level': 20.,
                         'logres': rng.normal(0, np.sqrt(a_p + b_p / 20.))})
    df = SwingNoise.decompose(pd.DataFrame(rows))
    assert {'reg_mean', 'within', 'k', 'n_groups'} <= set(df.columns)
    assert (df.loc[df['group'] == 'g1', 'within'] == 0).all() and (df.loc[df['group'] == 'g7', 'k'] == 5).all()
    assert (df.loc[df['party'] == 'R1', 'n_groups'] == 1).all() and (df.loc[df['party'] == 'P1', 'n_groups'] == 7).all()

    noise = SwingNoise.fit(df)
    assert noise.a_r == pytest.approx(a_r, abs=0.003) and noise.b_r == pytest.approx(b_r, rel=0.2)
    assert noise.a_p == pytest.approx(a_p, abs=0.0015) and noise.b_p == pytest.approx(b_p, rel=0.3)
    # Sin corrección, la varianza provincial saldría a menos de dos tercios (uniprovinciales y comunidades de 2)
    assert noise.var_province(4.) > 0.85 * (a_p + b_p / 4.)


def test_swing_noise_wls_keeps_the_fit_when_a_coefficient_is_clipped():
    """M9 (revisión): con un coeficiente negativo se reajusta el otro con la restricción, no se recorta a posteriori."""
    from mtpy.lib.computer import SwingNoise
    L = np.array([2., 5., 20.])
    n = np.array([100., 100., 100.])
    # Varianza creciente con el nivel: pendiente negativa → b = 0 y a = media ponderada
    a, b = SwingNoise.wls_nonneg(L, np.array([0.01, 0.02, 0.03]), n)
    assert b == 0 and a == pytest.approx(0.02)
    # Ordenada negativa: a = 0 y b ajustado por el origen sobre 1/L (no el b sin restricción)
    L = np.array([1.5, 5., 50.])
    y = np.array([0.1, 0.02, 0.])
    a, b = SwingNoise.wls_nonneg(L, y, n)
    x = 1. / L
    assert a == 0 and b == pytest.approx((n * y * x).sum() / (n * x * x).sum())
    # Sin restricción activa, el resultado es el de mínimos cuadrados ponderados ordinario
    a, b = SwingNoise.wls_nonneg(L, 0.01 + 0.2 * x, n)
    assert a == pytest.approx(0.01) and b == pytest.approx(0.2)


# --- M11: ratings transversales (sondeos de varios ámbitos con peso por ámbito) ---

def bare_computer():
    """Computer sin base de datos: sólo los atributos que usa el algoritmo de ratings."""
    c = Computer.__new__(Computer)
    c.scope = 'es'
    c.keys = ['event_date', 'date', 'pollster_id', 'sponsor_id']
    c.pos_decay, c.week_decay, c.year_decay = .5, .7, .9
    c.bias_dev_tau, c.min_polls, c.verbose = .01, 3, 0
    c.pollsters = pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['A', 'B', 'C', 'D'], 'quality': [60.] * 4})
    return c


def pool(rows):
    """Un sondeo por tupla (ámbito, fecha del evento, casa), 10 días antes del evento."""
    ids = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    data = []
    for scope, event_date, pollster in rows:
        event_date = pd.Timestamp(event_date)
        dev = .01 if pollster in ('A', 'C') else -.01
        data.append({
            'event_scope': scope, 'event_date': event_date, 'date': event_date - pd.Timedelta(days=10),
            'pollster_id': ids[pollster], 'sponsor_id': 0, 'pollster': pollster, 'days': 10,
            'weight_over': 1., 'weight_sample': 1.,
            'error_avg': .03, 'error_blocks': .01, 'error_within': .02,
            'bias_avg': .2, 'bias_blocks': .1, 'bias_within': .1, 'bias': .15,
            'bias_dev_adj': dev, 'bias_dev_err': .01
        })
    return pd.DataFrame(data).set_index(['event_scope', 'event_date', 'date', 'pollster_id', 'sponsor_id']).sort_index()


ES19 = [('es', '2019-11-10', 'A'), ('es', '2019-11-10', 'B')]
MD21 = [('es-md', '2021-05-04', 'A'), ('es-md', '2021-05-04', 'C')]
REG23 = [('es-md', '2023-05-28', 'C'), ('es-cl', '2023-05-28', 'D')]
JULY23, MAY23 = pd.Timestamp('2023-07-23'), pd.Timestamp('2023-05-28')


def test_scope_weight_zero_reproduces_single_scope_ratings():
    c = bare_computer()
    mixed = c.rate_events(pool(ES19 + MD21), [JULY23], {'es': 1., 'es-md': 0.}, c.pollsters)
    alone = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters)
    pd.testing.assert_frame_equal(mixed, alone)


def test_scope_weight_one_equals_merging_the_events():
    c = bare_computer()
    mixed = c.rate_events(pool(ES19 + MD21), [JULY23], {'es': 1., 'es-md': 1.}, c.pollsters)
    merged = c.rate_events(pool(ES19 + [('es', d, p) for _, d, p in MD21]), [JULY23], {'es': 1.}, c.pollsters)
    pd.testing.assert_frame_equal(mixed, merged)


def test_same_day_events_do_not_see_each_other():
    c = bare_computer()
    weights = {'es': 1., 'es-md': .5, 'es-cl': .5}
    assert c.rate_events(pool(REG23), [MAY23], weights, c.pollsters).shape[0] == 0
    july = c.rate_events(pool(REG23), [JULY23], weights, c.pollsters).set_index('pollster_id')
    assert (july.loc[[3, 4], 'num_polls'] > 0).all()


def test_regional_only_pollster_is_rated_and_events_are_scope_date_pairs():
    # Foco 5: C sólo publica en Madrid y aun así tiene rating en un evento de `es`
    c = bare_computer()
    out = c.rate_events(pool(ES19 + MD21 + REG23), [JULY23], {'es': 1., 'es-md': .5, 'es-cl': .5}, c.pollsters)
    out = out.set_index('pollster_id')
    assert out.loc[3, 'num_events'] == 2 and out.loc[3, 'num_polls'] == 2      # C: Madrid 2021 y 2023
    assert out.loc[1, 'num_events'] == 2                                       # A: es 2019 y Madrid 2021
    assert out.loc[3, 'rating'] != out.loc[3, 'quality']


def test_centre_house_results_handles_scopes_without_enough_polls():
    """M11: en un ámbito con pocas encuestas por casa ninguna supera el mínimo; el centrado no debe fallar."""
    cols = ['event_date', 'pollster_id', 'party', 'n_result', 'dev_result', 'dev_result_err']
    result = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-05-28'] * 2), 'pollster_id': [1, 2], 'party': ['PP', 'PP'],
        'n_result': [10, 10], 'dev_result': [1., 3.], 'dev_result_err': [.1, .1]
    })
    out = Computer.centre_house_results(result)
    assert out['industry'].tolist() == [2., 2.] and out['dev_result_c'].tolist() == [-1., 1.]
    empty = Computer.centre_house_results(result.iloc[:0])
    assert empty.shape[0] == 0 and list(empty.columns) == cols + ['industry', 'dev_result_c']


# --- M13: herding ---

def test_herding_summary_pools_cycles_with_year_decay():
    c = bare_computer()
    data = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-11-10', '2023-07-23', '2023-07-23', '2026-11-29']),
        'pollster': ['A', 'A', 'B', 'A'],
        'n': [10, 20, 8, 15],
        'ss_obs': [4., 8., 18., 2.],
        'ss_exp': [10., 10., 9., 10.],
        'ratio': [np.sqrt(.4), np.sqrt(.8), np.sqrt(2.), np.sqrt(.2)],
        'p_value': [.1, .3, .99, .01]
    })
    out = c.herding_summary(data=data)
    d = np.power(.9, [7, 3, 0])  # años hasta 2026
    expected = np.sqrt((d * [4., 8., 2.]).sum() / (d * [10., 10., 10.]).sum())
    assert out.loc['A', 'herding'] == pytest.approx(expected)
    assert out.loc['A', 'herding_last'] == pytest.approx(np.sqrt(.2)) and out.loc['A', 'herding_p'] == pytest.approx(.01)
    assert out.loc['A', 'herding_n'] == 45 and out.loc['A', 'herding_events'] == 3
    # Hasta 2023 no cuenta el ciclo actual
    upto = c.herding_summary(event_date='2023-07-23', data=data)
    assert upto.loc['A', 'herding_events'] == 2 and upto.loc['B', 'herding'] == pytest.approx(np.sqrt(2.))
    assert c.herding_summary(data=data.iloc[0:0]).shape[0] == 0
