"""Tests del Computer que no necesitan base de datos (M6: descomposición del error y resumen de efectos de casa)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

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


def test_house_effects_summary_accepts_a_cycle_without_history():
    """Un ámbito sin filas en `pollsters_parties` (autonómicos) llega sólo con el ciclo: sin `dev_result_c` ni
    `n_result`. La tabla debe salir con `hist`/`trend` nulos y el `cycle` de cada casa (`publish analysis`)."""
    cycle = pd.DataFrame({
        'event_date': pd.to_datetime(['2027-05-23'] * 2), 'pollster': ['GAD3'] * 2, 'party': ['PP', 'PSOE'],
        'dev_cycle': [-1.5, 2.], 'n_cycle': [12, 12]
    })
    blocks = {'Derecha': ['PP'], 'Izquierda': ['PSOE']}
    out = Computer.house_effects_summary(cycle, blocks, current='2027-05-23', year_decay=0.9)
    gad3 = out.loc['GAD3']
    assert gad3.loc['PP', 'cycle'] == pytest.approx(-1.5) and gad3.loc['Izquierda', 'cycle'] == pytest.approx(2.)
    assert gad3.loc['PP', 'n_events'] == 0 and np.isnan(gad3.loc['PP', 'hist']) and np.isnan(gad3.loc['PP', 'trend'])


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
    c.pos_decay, c.year_decay = .5, .9
    c.bias_dev_tau, c.bias_dev_sigma, c.verbose = .01, .014, 0
    c.drop_mtypes = []
    c.pollsters = pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['A', 'B', 'C', 'D'], 'quality': [60.] * 4})
    return c


def pool(rows, days=10):
    """Un sondeo por tupla (ámbito, fecha del evento, casa), `days` días antes del evento (un entero para todos
    o un dict por casa). Un cuarto elemento opcional en la tupla son los días antes del evento de ese sondeo, y
    manda sobre `days`."""
    ids = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    data = []
    for row in rows:
        scope, event_date, pollster = row[:3]
        event_date = pd.Timestamp(event_date)
        if len(row) > 3:
            poll_days = row[3]
        else:
            poll_days = days[pollster] if isinstance(days, dict) else days
        dev = .01 if pollster in ('A', 'C') else -.01
        data.append({
            'event_scope': scope, 'event_date': event_date, 'date': event_date - pd.Timedelta(days=poll_days),
            'pollster_id': ids[pollster], 'sponsor_id': 0, 'pollster': pollster, 'days': poll_days,
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


def test_rating_weights_have_no_week_factor():
    c = bare_computer()
    out = c.poll_rating_weights(pool(ES19))
    assert list(out.columns) == ['weight_pos', 'weight_year', 'weight_scope']
    assert not hasattr(c, 'week_decay')


def test_a_distant_poll_counts_as_much_as_a_close_one():
    """D3: la desviación se mide contra el consenso de la misma fecha, así que la distancia a la elección no
    descuenta evidencia: mismo `num_polls_w`, misma desviación y mismo rating."""
    c = bare_computer()
    both_close = c.rate_events(pool(ES19, days=10), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    # B publishes 80 days before the event while A publishes 10 days before it: with a week decay B would weigh
    # 0.7 ** 10 relative to A
    mixed = c.rate_events(pool(ES19, days={'A': 10, 'B': 80}), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    assert mixed.loc[2, 'num_polls_w'] == pytest.approx(1.0)
    assert mixed.loc[1, 'num_polls_w'] == pytest.approx(mixed.loc[2, 'num_polls_w'])
    for col in ('num_polls_w', 'bias_dev_adj', 'rating_adj', 'rating'):
        assert mixed[col].tolist() == pytest.approx(both_close[col].tolist(), nan_ok=True), col


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


# --- D2 (2026-10-20): la evidencia del rating se cuenta por elecciones, no por sondeos ---

def test_polls_of_one_election_count_as_one_election():
    """D2: A publica tres sondeos en una sola elección; su evidencia es esa elección, no tres sondeos."""
    c = bare_computer()
    rows = [('es', '2019-11-10', 'A', 10), ('es', '2019-11-10', 'A', 40), ('es', '2019-11-10', 'A', 70),
            ('es', '2019-11-10', 'B', 10)]
    out = c.rate_events(pool(rows), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    assert out.loc[1, 'num_events_w'] == pytest.approx(1.0)
    assert out.loc[1, 'num_polls'] == 3


def test_rating_follows_the_grouped_formula():
    c = bare_computer()
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    tau, sigma = .01, .014
    se = sigma / 1.0                                  # one election: n_eff = 1
    d_post = .01 * tau ** 2 / (tau ** 2 + se ** 2)    # A deviates +0.01 (worse than the consensus)
    err_post = (1 / tau ** 2 + 1 / se ** 2) ** -.5    # `bayes_adjust` stores the posterior standard deviation
    r_adj = norm.cdf(-d_post / tau)
    k = (sigma / tau) ** 2
    assert out.loc[1, 'bias_dev_err'] == pytest.approx(err_post)
    assert out.loc[1, 'bias_dev_adj'] == pytest.approx(d_post)
    assert out.loc[1, 'rating'] == pytest.approx((1.0 * r_adj + k * .6) / (1.0 + k))


def test_more_elections_shrink_the_standard_error():
    """Tres elecciones del mismo peso: n_eff = 3 y se = sigma / sqrt(3)."""
    c = bare_computer()
    rows = [('es-md', '2021-05-04', 'A'), ('es-cl', '2021-05-04', 'A'), ('es-ar', '2021-05-04', 'A')]
    weights = {'es': 1., 'es-md': 1., 'es-cl': 1., 'es-ar': 1.}
    out = c.rate_events(pool(rows), [JULY23], weights, c.pollsters).set_index('pollster_id')
    assert out.loc[1, 'num_events_w'] == pytest.approx(3.0)
    assert out.loc[1, 'bias_dev_err'] == pytest.approx((1 / .01 ** 2 + 3 / .014 ** 2) ** -.5)


def test_measure_event_sigma_matches_the_weighted_spread():
    """sigma: desviación típica ponderada por `W_e` de `d_e` alrededor de la media de su casa, sólo con las casas
    de al menos 3 elecciones (B, con 2, queda fuera)."""
    c = bare_computer()
    rows = [('es-md', '2021-05-04', 'A'), ('es-cl', '2021-05-04', 'A'), ('es-ar', '2021-05-04', 'A'),
            ('es-md', '2021-05-04', 'B'), ('es-cl', '2021-05-04', 'B')]
    data = pool(rows)
    scopes = data.index.get_level_values('event_scope')
    pollsters = data['pollster'].to_numpy()
    data.loc[(pollsters == 'A') & (scopes == 'es-md'), 'bias_dev_adj'] = .00
    data.loc[(pollsters == 'A') & (scopes == 'es-cl'), 'bias_dev_adj'] = .01
    data.loc[(pollsters == 'A') & (scopes == 'es-ar'), 'bias_dev_adj'] = .02
    data.loc[(pollsters == 'B') & (scopes == 'es-md'), 'bias_dev_adj'] = -.05
    data.loc[(pollsters == 'B') & (scopes == 'es-cl'), 'bias_dev_adj'] = .05
    sigma = c.measure_event_sigma(data, {'es-md': 1., 'es-cl': 1., 'es-ar': 1.})
    assert sigma == pytest.approx(np.sqrt(np.mean([.01 ** 2, 0., .01 ** 2])))


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


# --- get_polls_metric: una métrica nula en todos los sondeos del evento no puede desaparecer ---

def single_event_computer():
    """Computer sin base de datos con un solo evento de bloques `vs` de un partido (`bias_within` nulo)."""
    c = Computer.__new__(Computer)
    c.scope, c.verbose = 'es-cb', 0
    c.keys = ['event_date', 'date', 'pollster_id', 'sponsor_id']
    c.names = ['PP', 'PSOE']
    c.event_params = {'2011-05-22': {'bmaps': {'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}}}
    c.drange, c.n_last, c.drop_mtypes, c.drop_ctypes = (0, 60), None, None, None
    event = pd.Timestamp('2011-05-22')
    index = pd.MultiIndex.from_tuples(
        [(event, pd.Timestamp('2011-05-12'), 1, 0), (event, pd.Timestamp('2011-05-15'), 2, 0)], names=c.keys
    )
    c.series = pd.DataFrame({
        'pollster': ['A', 'B'], 'mtype': ['public', 'private'], 'ctype': ['phone', 'phone'],
        'featured': [True, True], 'computed': [True, True], 'days': [10, 7], 'sample_size': [800, 1000],
        'proc_sample': [800, 1000], 'weight_over': [1., 1.], 'weight_sample': [.8, 1.],
        'error_avg': [1.5, 2.], 'bias_avg': [1., 1.2], 'error_blocks': [2., 3.], 'bias_blocks': [1.1, 1.4],
        'error_within': [0., 0.], 'bias_within': [np.nan, np.nan], 'bias': [1.1, 1.4],
        'PP': [45., 44.], 'PSOE': [40., 42.],
    }, index=index)
    c.errors = pd.DataFrame({'PP': [1.5, 0.5], 'PSOE': [-1.5, 0.5]}, index=index)
    return c


def test_get_polls_metric_keeps_a_metric_that_is_null_for_every_poll():
    """Con bloques `vs` de un solo partido `bias_within` es nulo en todos los sondeos: la columna debe seguir ahí
    (nula), no desaparecer con los partidos sin datos y romper la selección final (fallo real de `publish event`
    en es-cb 2011-05-22)."""
    df = single_event_computer().get_polls_metric('error', bmap='vs')
    assert list(df.index.names) == ['event', 'pollster', 'days'] and len(df) == 2
    assert 'bias_within' in df.columns and df['bias_within'].isna().all()
    assert df['error_within'].tolist() == [0., 0.] and {'Derecha', 'Izquierda'} <= set(df.columns)


# --- Selección 'results': los sondeos de las elecciones con resultado oficial, sean o no `featured` ---

THREE_EVENTS = ['2015-05-24', '2019-05-26', '2023-05-28']


def three_event_computer(featured=(True, False, False)):
    """Computer sin base de datos con un sondeo en cada una de tres elecciones y la fila de su resultado: 2015 y
    2019 con votos, 2023 sin ellos (el ciclo en curso); `featured` marca cada elección. Como en `load_events`, el
    `computed` de una fila de resultado es su `featured` y su `sample_size`, los votos."""
    c = Computer.__new__(Computer)
    c.scope, c.verbose = 'es-ar', 0
    c.keys = ['event_date', 'date', 'pollster_id', 'sponsor_id']
    c.names = ['PP', 'PSOE']
    c.event_params = {dt: {'bmaps': {'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE']}}} for dt in THREE_EVENTS}
    c.drange, c.n_last, c.drop_mtypes, c.drop_ctypes = (0, 60), None, None, None
    events = pd.to_datetime(THREE_EVENTS)
    polls = [(ev, ev - pd.Timedelta(days=10), i + 1, 0) for i, ev in enumerate(events)]
    results = [(ev, ev, np.nan, np.nan) for ev in events]
    index = pd.MultiIndex.from_tuples(polls + results, names=c.keys)
    c.series = pd.DataFrame({
        'pollster': ['A', 'B', 'C', None, None, None], 'mtype': None, 'ctype': None,
        'featured': list(featured) * 2, 'computed': [True, True, True] + list(featured),
        'days': [10, 10, 10, 0, 0, 0], 'sample_size': [800, 800, 800, 1000, 900, np.nan],
        'proc_sample': 800, 'weight_over': [1., 1., 1., np.nan, np.nan, np.nan], 'weight_sample': 1.,
        'error_avg': 1., 'bias_avg': 1., 'error_blocks': 2., 'bias_blocks': 1., 'error_within': 0.,
        'bias_within': np.nan, 'bias': 1., 'PP': [45., 44., 43., 44., 43., np.nan],
        'PSOE': [40., 41., 42., 41., 42., np.nan],
    }, index=index)
    c.errors = pd.DataFrame({'PP': [1., 1., np.nan], 'PSOE': [-1., -1., np.nan]}, index=index[:3])
    return c


def event_dates(df):
    """Fechas ISO de las elecciones de una selección de sondeos."""
    return sorted(df.index.get_level_values('event_date').strftime('%Y-%m-%d').unique())


def test_filter_polls_results_takes_every_election_with_an_official_result():
    c = three_event_computer()
    assert event_dates(c.filter_polls('results')) == ['2015-05-24', '2019-05-26']
    assert event_dates(c.filter_polls(True)) == ['2015-05-24']  # los estimadores no cambian
    assert event_dates(c.filter_polls(False)) == THREE_EVENTS  # sin filtro de elecciones, como hasta ahora


def test_filter_polls_results_equals_featured_when_every_election_with_a_result_is_featured():
    c = three_event_computer(featured=(True, True, False))
    pd.testing.assert_frame_equal(c.filter_polls('results'), c.filter_polls(True))


def test_filter_polls_results_leaves_out_an_election_whose_errors_are_not_computed():
    """Recién cargado el resultado y antes de `compute`, los sondeos de la elección no tienen error: la elección no
    entra en la selección (no se publica con errores vacíos ni con el aviso de no analizada)."""
    c = three_event_computer()
    c.series.loc[c.series.index.get_level_values('event_date') == pd.Timestamp('2019-05-26'), 'error_avg'] = np.nan
    assert event_dates(c.filter_polls('results')) == ['2015-05-24']


def test_get_polls_metric_passes_the_results_selection():
    df = three_event_computer().get_polls_metric('error', bmap='vs', featured='results')
    assert sorted(df['event_date'].unique()) == ['2015-05-24', '2019-05-26']


# --- Sesgo margen: el signo del error del margen entre bloques ---

def test_gap_error_is_positive_when_the_poll_favours_the_first_block():
    """`error_blocks` es el margen del primer bloque (Derecha) sobre el segundo, sondeo menos resultado: positivo
    cuando el sondeo dio al primer bloque más margen del que hubo (GAD3, 13-7-2023: 9,3 en el sondeo, 1,4 real)."""
    c = bare_computer()
    c.parties = pd.DataFrame({'name': ['PP', 'VOX', 'PSOE', 'SUMAR']})
    bmap = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE', 'SUMAR']}
    index = pd.MultiIndex.from_tuples([(pd.Timestamp('2023-07-23'), pd.Timestamp('2023-07-13'), 2, 0)],
                                      names=['event_date', 'date', 'pollster_id', 'sponsor_id'])
    polls = pd.DataFrame({'PP': [36.8], 'VOX': [12.3], 'PSOE': [28.5], 'SUMAR': [11.3]}, index=index)
    events = pd.DataFrame({'PP': [33.06], 'VOX': [12.38], 'PSOE': [31.68], 'SUMAR': [12.33]},
                          index=pd.Index([pd.Timestamp('2023-07-23')], name='event_date'))
    out = c.poll_errors(polls, events, error_type='gap', bmap=bmap)
    assert out[('poll', 'gap')].iloc[0] == pytest.approx(0.093) and out[('event', 'gap')].iloc[0] == pytest.approx(0.0143)
    assert out[('error', 'gap')].iloc[0] == pytest.approx(0.0787)


# --- 2026-10-19: el rating por sondeo lo copia el cómputo de pesos, no sólo la acción `ratings` ---

def test_compute_weights_copies_the_rating_of_the_event_to_each_poll():
    """Un sondeo cargado después del último `ratings` (la carga llama a `compute_weights`) sale con el rating de su
    casa para el evento, junto a `weight_rating`; sin fila de ratings, vacío. La serie se actualiza igual."""
    c = bare_computer()
    c.drop_mtypes, c.ol_max, c.ol_dev, c.wspan = [], 10000, 3, None
    c.pollsters = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'mtype': ['cati', 'online'], 'quality': [60., 60.]})
    event = pd.Timestamp('2026-11-29')
    c.ratings = pd.DataFrame({'rating': [62.76], 'weight_rating': [1.94]},
                             index=pd.MultiIndex.from_tuples([(event, 1)], names=['event_date', 'pollster_id']))
    # the series: the event row (no pollster) and three polls, the first one already computed
    c.series = pd.DataFrame({
        'event_date': event, 'date': pd.to_datetime(['2026-11-29', '2026-10-08', '2026-10-14', '2026-10-10']),
        'pollster_id': [0, 1, 1, 2], 'sponsor_id': 0, 'pollster': [None, 'A', 'A', 'B'], 'sponsor': None,
        'start_date': pd.to_datetime([None, '2026-10-01', '2026-10-09', '2026-10-05']),
        'end_date': pd.to_datetime([None, '2026-10-08', '2026-10-14', '2026-10-10']),
        'ctype': None, 'sample_size': [np.nan, 2000., 3000., 1000.], 'computed': [True, True, False, False],
        'featured': False, 'mtype': [None, 'cati', None, None], 'proc_sample': [np.nan, 2000., np.nan, np.nan],
        'weight_sample': [np.nan, 1., np.nan, np.nan], 'weight_over': [np.nan, 1., np.nan, np.nan],
        'weight_rating': [np.nan, 1.94, np.nan, np.nan], 'rating': [np.nan, 62.76, np.nan, np.nan],
    }).set_index(c.keys)
    out = c.compute_weights(save=False, overwrite=True).reset_index().set_index('date')
    new, other = pd.Timestamp('2026-10-14'), pd.Timestamp('2026-10-10')
    assert out.loc[new, 'rating'] == 62.76 and out.loc[new, 'weight_rating'] == 1.94 and out.loc[new, 'computed']
    assert np.isnan(out.loc[other, 'rating']) and np.isnan(out.loc[other, 'weight_rating'])  # B has no ratings row
    series = c.series.reset_index().set_index('date')['rating']
    assert series[[pd.Timestamp('2026-10-08'), new, other]].tolist() == pytest.approx([62.76, 62.76, np.nan], nan_ok=True)


# --- Calidad previa a la fecha de cada elección ---

def test_prior_quality_is_taken_at_the_date_of_each_target():
    """D3: la calidad de una elección puntuada sólo cuenta las generales anteriores a ella."""
    c = bare_computer()

    def prior(target):
        later = pd.isnull(target) or pd.Timestamp(target) > pd.Timestamp('2023-07-24')
        return pd.DataFrame({
            'quality': [56. if later else 40.] * 4, 'num_national_events': [1 if later else 0] * 4
        }, index=pd.Index([1, 2, 3, 4], name='pollster_id'))

    out = c.rate_events(
        pool(ES19), [JULY23, pd.Timestamp('2026-11-29')], {'es': 1.}, c.pollsters, prior=prior
    ).set_index(['event_date', 'pollster_id'])
    before, after = out.loc[JULY23], out.loc[pd.Timestamp('2026-11-29')]
    assert before.loc[3, 'quality'] == pytest.approx(.40) and before.loc[3, 'num_national_events'] == 0
    assert after.loc[3, 'quality'] == pytest.approx(.56) and after.loc[3, 'num_national_events'] == 1
    # C has no evidence: its rating is its prior quality on each date
    assert before.loc[3, 'rating'] == pytest.approx(before.loc[3, 'quality'])
    assert after.loc[3, 'rating'] == pytest.approx(after.loc[3, 'quality'])


def test_without_prior_the_pollsters_quality_is_used_and_num_national_events_is_null():
    c = bare_computer()
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters)
    assert out['quality'].eq(.6).all() and out['num_national_events'].isnull().all()


def test_compute_ratings_builds_the_prior_from_the_national_events(monkeypatch):
    from mtpy.lib import computer

    c = bare_computer()
    c.rating_pool = lambda scopes: pool(ES19)
    monkeypatch.setattr(computer, 'get_next_event_date', lambda scope, date_from=None: None)
    monkeypatch.setattr(computer, 'load_criteria', lambda: {'scale': {'per_point': 8, 'floor': 5}})
    monkeypatch.setattr(computer, 'read_quality', lambda criteria: pd.DataFrame({
        'pollster_id': [1], 'fieldwork': [2], 'transparency': [2], 'independence': [1], 'identity': [1],
        'standards': [1]}))
    monkeypatch.setattr(computer, 'get_national_events', lambda: pd.DataFrame({
        'pollster_id': [1, 1], 'event_date': [pd.Timestamp('2023-07-23'), pd.Timestamp('2019-11-10')]}))
    seen = {}

    def fake_rate_events(pool_, targets, weights, pollsters, prior=None):
        seen['prior'] = prior
        raise RuntimeError('stop')

    c.rate_events = fake_rate_events
    with pytest.raises(RuntimeError):
        c.compute_ratings()
    prior = seen['prior']
    assert prior(pd.Timestamp('2023-07-23')).loc[1, 'num_national_events'] == 1
    assert prior(pd.NaT).loc[1, 'num_national_events'] == 2


def overweights(rows, wspan=4):
    """`poll_overweights` sobre (inicio, fin) en días relativos a la elección."""
    c = bare_computer()
    c.wspan = wspan
    return c.poll_overweights(np.array(rows, dtype=float))


def test_overlapping_polls_keep_only_the_last_delivery():
    # A cumulative tracking: every delivery contains the previous ones
    assert overweights([(-20, -18), (-20, -17), (-20, -16)]).tolist() == [0, 0, 1]


def test_a_burst_of_polls_shares_the_weight_of_one_poll():
    # Days 0 and 3 fall in the group opened by day 0; day 6 opens a new group
    assert overweights([(-31, -30), (-28, -27), (-25, -24)]).tolist() == pytest.approx([.5, .5, 1])


def test_every_group_of_a_daily_series_sums_one():
    w = overweights([(d, d) for d in range(-40, -10)])
    assert w.sum() == pytest.approx(6)
    assert set(np.round(w, 6)) == {round(1 / 5, 6)}


def test_unsorted_input_keeps_its_order():
    assert overweights([(-25, -24), (-31, -30), (-28, -27)]).tolist() == pytest.approx([1, .5, .5])


def test_without_wspan_only_the_overlap_rule_applies():
    assert overweights([(-31, -30), (-28, -27), (-25, -24)], wspan=None).tolist() == [1, 1, 1]


# --- 2026-10-21: agregadores y paneles online (`DROP_MTYPES`): sus medias se miden, sin rating ---

def excluded_computer():
    """Computer de ratings en el que B es un panel online: no entra en el promedio ni tiene rating."""
    c = bare_computer()
    c.drop_mtypes = ['aggr', 'online']
    c.pollsters = c.pollsters.assign(mtype=[None, 'online', None, None])
    return c


def test_an_excluded_pollster_keeps_its_means_without_a_rating():
    """B (online) se mide como cualquier casa: sondeos, elecciones, errores y diferencial; pero sin rating ni peso."""
    c = excluded_computer()
    out = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    b = out.loc[2]
    assert b['num_polls'] == 1 and b['num_events'] == 1
    assert b['error_avg'] == pytest.approx(.03) and b['bias'] == pytest.approx(.15)
    assert b['bias_dev_adj'] < 0  # B se desvía -0.01: menos sesgo que el consenso
    assert np.isnan(b['rating_adj']) and np.isnan(b['rating']) and np.isnan(b['weight_rating'])
    assert not np.isnan(out.loc[1, 'rating']) and not np.isnan(out.loc[1, 'weight_rating'])  # A sí lo tiene


def test_an_excluded_pollster_does_not_move_the_weights_of_the_others():
    """La media con la que se normaliza `weight_rating` es la de las casas con rating: sin B, lo mismo."""
    c = excluded_computer()
    with_b = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    plain = bare_computer()
    others = plain.pollsters.loc[plain.pollsters['name'] != 'B']
    without_b = plain.rate_events(pool(ES19[:1]), [JULY23], {'es': 1.}, others).set_index('pollster_id')
    for col in ('rating', 'weight_rating'):
        assert with_b.loc[[1, 3, 4], col].tolist() == pytest.approx(without_b.loc[[1, 3, 4], col].tolist()), col


def test_the_rating_pool_keeps_the_polls_of_the_excluded_pollsters():
    """El pool del rating lleva también los sondeos de las casas excluidas, para medir sus medias."""
    c = excluded_computer()
    c.drange, c.n_last, c.drop_ctypes, c.names = None, None, ['wban', 'exit'], ['PP']
    c.series = pool(ES19).droplevel('event_scope').assign(
        mtype=[None, 'online'], ctype=None, featured=True, computed=True, PP=[30., 31.], rating=np.nan,
        weight_rating=np.nan)
    out = c.rating_pool(['es'])
    assert out.index.get_level_values('pollster_id').tolist() == [1, 2]
    assert out.index.get_level_values('event_scope').unique().tolist() == ['es']


def test_the_sample_median_of_an_event_ignores_the_excluded_pollsters():
    """El peso por muestra se mide contra la mediana de las casas que cuentan: un panel enorme no lo mueve."""
    c = bare_computer()
    c.drop_mtypes, c.ol_max, c.ol_dev, c.wspan = ['aggr', 'online'], 100000, 3, None
    c.pollsters = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'mtype': ['cati', 'online'], 'quality': [60., 60.]})
    event = pd.Timestamp('2026-11-29')
    c.ratings = pd.DataFrame({'rating': [], 'weight_rating': []},
                             index=pd.MultiIndex.from_tuples([], names=['event_date', 'pollster_id']))
    c.series = pd.DataFrame({
        'event_date': event, 'date': pd.to_datetime(['2026-11-29', '2026-10-01', '2026-10-08', '2026-10-10']),
        'pollster_id': [0, 1, 1, 2], 'sponsor_id': 0, 'pollster': [None, 'A', 'A', 'B'], 'sponsor': None,
        'start_date': pd.to_datetime([None, '2026-09-28', '2026-10-05', '2026-10-07']),
        'end_date': pd.to_datetime([None, '2026-10-01', '2026-10-08', '2026-10-10']),
        'ctype': None, 'sample_size': [np.nan, 1000., 2000., 9000.], 'computed': False,
        'featured': False, 'mtype': None, 'proc_sample': np.nan,
        'weight_sample': np.nan, 'weight_over': np.nan, 'weight_rating': np.nan, 'rating': np.nan,
    }).set_index(c.keys)
    out = c.compute_weights(save=False, overwrite=True).reset_index().set_index('date')
    # mediana de A: 1500, no 2000 (la de los tres sondeos)
    assert out.loc[pd.Timestamp('2026-10-01'), 'weight_sample'] == pytest.approx(round(np.sqrt(1000 / 1500), 2))
    assert out.loc[pd.Timestamp('2026-10-08'), 'weight_sample'] == pytest.approx(round(np.sqrt(2000 / 1500), 2))


# --- Calibración (parte B): error del promedio nowcast por elección, partido y horizonte (`average_errors`) ---

def _computer_without_db(monkeypatch, events, main, smap=None, extra=None):
    """Computer sin base de datos para `get_average_error_data`: `events` son sus elecciones `featured` (un sondeo
    cada una, lo único que lee de `filter_polls`), `main` los partidos principales de cada una (también los
    partidos del evento, salvo los de `extra`) y `smap` su mapa de fuentes; el catálogo de partidos es falso."""
    from mtpy.lib import computer as computer_module

    c = Computer.__new__(Computer)
    c.scope, c.verbose, c.path, c.alpha = 'es', 0, '.', .05
    c.drop_mtypes = ['aggr']
    c.event_dates = list(events)
    c.event_params = {
        d: {'parties': {'event': list(main[d]) + list((extra or {}).get(d, []))}, 'bmaps': {'main': list(main[d])},
            'smap': (smap or {}).get(d, {})}
        for d in events
    }
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), pd.Timestamp(d) - pd.Timedelta(days=10), 1, 0) for d in events],
        names=['event_date', 'date', 'pollster_id', 'sponsor_id']
    )
    c.filter_polls = lambda featured=True, **kwargs: pd.DataFrame({'pollster': ['A'] * len(events)}, index=index)
    monkeypatch.setattr(computer_module, 'get_parties', lambda: pd.DataFrame({
        'id': [1, 2, 3, 4], 'name': ['PP', 'PSOE', 'VOX', 'ERC'], 'regional': [False, False, False, True]}))
    return c


def _fake_forecaster(level, err, date_fit_last, event_date='2023-07-23', n_polls=8, calls=None):
    """Forecaster ajustado falso: promedio `level` y error `err` constantes por partido sobre todo el ciclo,
    `n_polls` sondeos semanales que acaban en `date_fit_last` (`fc_series`) y ese último día ajustado; anota en
    `calls` los argumentos de `fit_forecast`."""
    import types

    dates = pd.date_range(pd.Timestamp(event_date) - pd.Timedelta(days=400), event_date, freq='D', name='date')
    names = list(level)
    forecast = pd.DataFrame({n: float(v) for n, v in level.items()}, index=dates)
    forecast['-'] = 100. - forecast[names].sum(axis=1)
    fc_stat = pd.DataFrame({n: [{'mean': level[n], 'err': err[n]}] * len(dates) for n in names}, index=dates)
    poll_dates = pd.date_range(end=date_fit_last, periods=n_polls, freq='7D', name='date')
    fc = types.SimpleNamespace(names=names, forecast=forecast, fc_stat=fc_stat, date_fit_last=date_fit_last,
                               fc_series=pd.DataFrame({'pollster': 'A', 'weight': 1.}, index=poll_dates))
    fc.build_series = lambda: fc

    def fit_forecast(names=None, max_fc=0, fillna=False):
        if calls is not None:
            calls.append({'names': names, 'max_fc': max_fc, 'fillna': fillna})
        return fc.forecast

    fc.fit_forecast = fit_forecast
    return fc


OFFICIAL_2023 = pd.DataFrame({'pct': [33.06, 31.68]}, index=['PP', 'PSOE'])


def test_get_average_error_data_builds_one_row_per_main_party_and_horizon(monkeypatch):
    c = _computer_without_db(monkeypatch, events=['2023-07-23'], main={'2023-07-23': ['PP', 'PSOE']})
    monkeypatch.setattr('mtpy.lib.forecaster.Forecaster', lambda **kw: _fake_forecaster(
        level={'PP': 33.5, 'PSOE': 28.2}, err={'PP': 0.4, 'PSOE': 0.5},
        date_fit_last=pd.Timestamp('2023-07-23') - pd.Timedelta(days=kw['drange'])))
    monkeypatch.setattr('mtpy.lib.backtest.official_results', lambda scope, d: OFFICIAL_2023)
    df = c.get_average_error_data(horizons=(6, 30))
    assert df.shape[0] == 4 and set(df.columns) >= {'event_date', 'event_scope', 'party_id', 'party', 'horizon', 'main',
                                                    'level', 'official', 'error', 'err', 'n_polls', 'as_of'}
    row = df.query("party == 'PSOE' and horizon == 6").iloc[0]
    assert row['error'] == pytest.approx(28.2 - 31.68) and row['err'] == pytest.approx(0.5)
    assert row['as_of'] == pd.Timestamp('2023-07-17')
    assert row['event_date'] == pd.Timestamp('2023-07-23') and row['event_scope'] == 'es' and row['party_id'] == 2
    assert row['level'] == pytest.approx(28.2) and row['official'] == pytest.approx(31.68) and row['n_polls'] == 8
    assert bool(row['main']) and not bool(row['regional'])
    assert df.query("party == 'PP' and horizon == 30").iloc[0]['as_of'] == pd.Timestamp('2023-06-23')


def test_get_average_error_data_fits_the_simulator_average(monkeypatch):
    """El promedio es el del Simulator (y del backtest): las series de los partidos del evento con los sondeos de
    sus predecesores (`poll_blocks` del `smap`), efectos de casa, muestra efectiva y rebaño, y sólo se leen los
    principales; el ancla es el último día ajustado cuando cae antes de `event − horizonte`."""
    calls = []
    c = _computer_without_db(monkeypatch, events=['2023-07-23'], main={'2023-07-23': ['PP', 'PSOE']},
                             smap={'2023-07-23': {'PP': {'type': 'agg', 'names': ['Cs']}}},
                             extra={'2023-07-23': ['ERC']})
    monkeypatch.setattr('mtpy.lib.forecaster.Forecaster', lambda **kw: calls.append(kw) or _fake_forecaster(
        level={'PP': 33.5, 'PSOE': 28.2}, err={'PP': 0.4, 'PSOE': 0.5}, date_fit_last=pd.Timestamp('2023-07-10'),
        calls=calls))
    monkeypatch.setattr('mtpy.lib.backtest.official_results', lambda scope, d: OFFICIAL_2023)
    df = c.get_average_error_data(horizons=(6,), max_fc=7, house_effects=False, dispersion=True, herding=False)
    init, fit = calls
    assert init['scope'] == 'es' and init['event_date'] == '2023-07-23' and init['drange'] == 6
    assert init['bmap'] == {'PP': ['PP', 'Cs'], 'PSOE': ['PSOE'], 'ERC': ['ERC']}
    assert init['house_effects'] is False and init['dispersion'] is True and init['disp_params'] == {'herding': False}
    assert init['drop_mtypes'] == ['aggr'] and init['alpha'] == .05
    assert fit == {'names': ['PP', 'PSOE'], 'max_fc': 7, 'fillna': True}
    assert (df['as_of'] == pd.Timestamp('2023-07-10')).all()


def test_get_average_error_data_skips_what_cannot_be_measured(monkeypatch):
    """Sin resultado oficial la elección no entra; un partido sin resultado o sin promedio, tampoco; un caso con
    menos de `min_polls` sondeos o sin sondeos (el Forecaster lo rechaza) se salta con aviso."""
    events = ['2019-11-10', '2023-07-23', '2026-11-29']
    main = {'2019-11-10': ['PP', 'PSOE'], '2023-07-23': ['PP', 'PSOE', 'VOX'], '2026-11-29': ['PP', 'PSOE']}
    c = _computer_without_db(monkeypatch, events=events, main=main)

    def forecaster(**kw):
        if kw['drange'] == 180:
            raise ValueError('No polls for es {}: nothing to forecast'.format(kw['event_date']))
        n_polls = 3 if kw['event_date'] == '2019-11-10' else 8
        fit_last = pd.Timestamp(kw['event_date']) - pd.Timedelta(days=kw['drange'])
        return _fake_forecaster(level={'PP': 33.5, 'VOX': 13.}, err={'PP': 0.4, 'VOX': 0.6}, n_polls=n_polls,
                                event_date=kw['event_date'], date_fit_last=fit_last)

    official = {'2019-11-10': pd.DataFrame({'pct': [20.8, 28.0]}, index=['PP', 'PSOE']), '2023-07-23': OFFICIAL_2023,
                '2026-11-29': pd.DataFrame({'pct': []})}
    monkeypatch.setattr('mtpy.lib.forecaster.Forecaster', forecaster)
    monkeypatch.setattr('mtpy.lib.backtest.official_results', lambda scope, d: official[d])
    with pytest.warns(UserWarning) as record:
        df = c.get_average_error_data(horizons=(6, 180))
    # 2019: 3 sondeos (< 5); 2023 a 180 días: sin sondeos; 2026: sin resultado (ni se intenta)
    assert df[['event_date', 'party', 'horizon']].values.tolist() == [[pd.Timestamp('2023-07-23'), 'PP', 6]]
    messages = ' | '.join(str(w.message) for w in record)
    assert '2019-11-10' in messages and '180' in messages and '2026-11-29' not in messages


def test_get_average_error_data_takes_the_given_events_and_keeps_the_columns_when_empty(monkeypatch):
    c = _computer_without_db(monkeypatch, events=['2019-11-10', '2023-07-23'],
                             main={'2019-11-10': ['PP'], '2023-07-23': ['PP', 'PSOE']})
    seen = []

    def forecaster(**kw):
        seen.append(kw['event_date'])
        return _fake_forecaster(level={'PP': 33.5}, err={'PP': 0.4}, date_fit_last=pd.Timestamp('2023-07-17'))

    monkeypatch.setattr('mtpy.lib.forecaster.Forecaster', forecaster)
    monkeypatch.setattr('mtpy.lib.backtest.official_results', lambda scope, d: OFFICIAL_2023)
    df = c.get_average_error_data(horizons=(6,), events=['2023-07-23'])
    assert seen == ['2023-07-23'] and df['party'].tolist() == ['PP']
    empty = c.get_average_error_data(horizons=(6,), events=[])
    assert empty.shape[0] == 0
    assert {'level', 'official', 'error', 'err', 'n_polls', 'as_of', 'regional'} <= set(empty.columns)


def test_compute_average_errors_saves_only_when_asked_and_load_reads_the_current_events(monkeypatch):
    from mtpy.lib import computer as computer_module

    c = _computer_without_db(monkeypatch, events=['2023-07-23'], main={'2023-07-23': ['PP']})
    frame = pd.DataFrame({'event_date': [pd.Timestamp('2023-07-23')], 'party': ['PP']})
    seen = {}
    c.get_average_error_data = lambda **kwargs: seen.setdefault('kwargs', kwargs) and frame
    monkeypatch.setattr(computer_module, 'save_average_errors_data', lambda df: seen.setdefault('saved', len(df)))
    monkeypatch.setattr(computer_module, 'get_average_errors', lambda scope, event_dates: (
        seen.setdefault('loaded', (scope, event_dates)) and frame))
    assert c.compute_average_errors(horizons=(6,)) is frame and 'saved' not in seen
    assert seen['kwargs'] == {'horizons': (6,)} and c.average_errors is frame
    c.compute_average_errors(save=True)
    assert seen['saved'] == 1
    assert c.load_average_errors() is frame and seen['loaded'] == ('es', ['2023-07-23'])


# --- Calibración (parte B): estimador del error del propio promedio (`AverageErrorEstimator`) ---

def _average_error_rows(**overrides):
    """Doce filas de `average_errors` a 6 días: dos elecciones con los mismos seis partidos (dos regionales)."""
    rows = pd.DataFrame({'event_date': ['2019-04-28'] * 6 + ['2019-11-10'] * 6, 'horizon': 6,
                         'party': list('ABCDEF') * 2, 'level': [30, 20, 10, 5, 3, 2] * 2, 'regional': [0, 0, 0, 0, 1, 1] * 2,
                         'error': [2.5, 2.0, 1.0, 0.6, 0.4, 0.3] * 2, 'err': [0.5] * 12, 'main': True})
    return rows.assign(**overrides)


def test_average_error_estimator_fits_the_net_error_on_level_and_regional(monkeypatch):
    c = Computer.__new__(Computer)
    rows = _average_error_rows()
    monkeypatch.setattr(c, 'load_average_errors', lambda: rows)
    est = c.get_average_error_estimator(horizon=6)
    assert est.n_rows == 12 and est.horizon == 6 and est.net
    assert est.events == ['2019-04-28', '2019-11-10']
    assert est.predict(np.array([[30., 0., 1.]]))[0] == pytest.approx(np.sqrt(2.5 ** 2 - 0.5 ** 2), abs=0.3)
    assert est.predict(np.array([[30., 0., 1.]]))[0] > est.predict(np.array([[5., 0., 1.]]))[0]
    # Drop-in de `LeastSquaresEstimator.predict` en el Simulator: un array de una predicción por fila
    out = est.predict(np.array([[30., 0., 1.], [5., 1., 1.]]))
    assert isinstance(out, np.ndarray) and out.shape == (2,)


def test_average_error_estimator_without_net_fits_the_absolute_error(monkeypatch):
    c = Computer.__new__(Computer)
    rows = _average_error_rows(error=[-2.5, 2.0, -1.0, 0.6, -0.4, 0.3] * 2, err=1.)
    monkeypatch.setattr(c, 'load_average_errors', lambda: rows)
    net, gross = c.get_average_error_estimator(horizon=6), c.get_average_error_estimator(horizon=6, net=False)
    assert gross.net is False
    p = np.array([[30., 0., 1.]])
    assert gross.predict(p)[0] > net.predict(p)[0]  # `|error|` sin restar `s`
    assert gross.predict(p)[0] == pytest.approx(2.5, abs=0.3)


def test_average_error_estimator_takes_only_the_main_rows_of_the_horizon(monkeypatch):
    c = Computer.__new__(Computer)
    rows = pd.concat([_average_error_rows(), _average_error_rows(horizon=30, error=9.),
                      _average_error_rows(main=False)], ignore_index=True)
    monkeypatch.setattr(c, 'load_average_errors', lambda: rows)
    est = c.get_average_error_estimator(horizon=6)
    assert est.n_rows == 12 and est.predict(np.array([[30., 0., 1.]]))[0] < 3.


def test_average_error_estimator_needs_twelve_rows(monkeypatch):
    c = Computer.__new__(Computer)
    monkeypatch.setattr(c, 'load_average_errors', lambda: pd.DataFrame(
        columns=['event_date', 'horizon', 'party', 'level', 'regional', 'error', 'err', 'main']))
    with pytest.warns(UserWarning, match='average_errors'):
        assert c.get_average_error_estimator() is None
    monkeypatch.setattr(c, 'load_average_errors', lambda: _average_error_rows().iloc[:11])
    with pytest.warns(UserWarning, match='average_errors'):
        assert c.get_average_error_estimator(horizon=6) is None
