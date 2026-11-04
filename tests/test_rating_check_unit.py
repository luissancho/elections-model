"""Tests de las medidas puras de la validación del rating (frames construidos a mano, sin base de datos)."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mtpy.lib.rating_check import (
    POPULATIONS, bootstrap_ci, discrimination, fit_line, population_of, quarters,
    rating_check_summary, scatter_rows, simple_average_errors, skill, slope, summarize_simple_average,
)

HOUSES = {'A': (1, .8), 'B': (2, .6), 'C': (3, .4), 'D': (4, .2)}
D_OBS = {'A': -.02, 'B': -.01, 'C': .01, 'D': .02}      # crece al bajar el rating: correlación -1
D_PREV = {'A': -.015, 'B': np.nan, 'C': .012, 'D': .025}
ELECTIONS = [('es', '2019-11-10'), ('es-md', '2021-05-04'), ('es-cl', '2021-05-04')]


def build_pairs():
    rows = []
    for e, (scope, date) in enumerate(ELECTIONS):
        for name, (pid, rating) in HOUSES.items():
            rows.append({
                'event_scope': scope, 'event_date': date, 'population': population_of(scope),
                'pollster_id': pid, 'pollster': name, 'd_obs': D_OBS[name],
                'error_avg': 1. + pid + e, 'n_polls': 5, 'n_event_polls': 12 if e < 2 else 6,
                'rating': rating, 'rating_adj': rating, 'd_prev': D_PREV[name],
                'num_events_w': 0. if name == 'B' else (3. if name in 'AC' else 1.),
                'quality': rating if name == 'B' else rating - .05, 'weight_rating': rating,
            })
    return pd.DataFrame(rows)


def build_polls():
    rows = []

    def add(scope, date, name, pid, errs, w_over, w_sample, w_rating):
        for party, err in zip(['X', 'Y'], errs):
            rows.append({'event_scope': scope, 'event_date': date, 'population': population_of(scope),
                         'pollster_id': pid, 'pollster': name, 'party': party, 'error': err,
                         'weight_over': w_over, 'weight_sample': w_sample, 'weight_rating': w_rating})

    # es 2019: tres casas; es-md 2021: tres casas; es-cl 2021: sólo dos (se descarta)
    add('es', '2019-11-10', 'A', 1, (1., -2.), 1., 2., .5)
    add('es', '2019-11-10', 'B', 2, (3., 1.), 1., 1., 1.)
    add('es', '2019-11-10', 'C', 3, (-1., 2.), 2., 1., 2.)
    add('es-md', '2021-05-04', 'A', 1, (1., 1.), 1., 1., 1.)
    add('es-md', '2021-05-04', 'B', 2, (2., 0.), 1., 1., 2.)
    add('es-md', '2021-05-04', 'C', 3, (-2., 2.), 1., 1., 1.)
    add('es-cl', '2021-05-04', 'A', 1, (1., 1.), 1., 1., 1.)
    add('es-cl', '2021-05-04', 'B', 2, (2., 0.), 1., 1., 1.)
    return pd.DataFrame(rows)


PAIRS = build_pairs()
POLLS = build_polls()

# es 2019-11-10 a mano: w_flat = over * sample = (2, 1, 2); w_rated = w_flat * rating = (1, 1, 4)
_X, _Y = np.array([1., 3., -1.]), np.array([-2., 1., 2.])
_WF, _WR = np.array([2., 1., 2.]), np.array([1., 1., 4.])
EXPECTED_FLAT = (abs((_WF * _X).sum() / _WF.sum()) + abs((_WF * _Y).sum() / _WF.sum())) / 2
EXPECTED_RATED = (abs((_WR * _X).sum() / _WR.sum()) + abs((_WR * _Y).sum() / _WR.sum())) / 2


def test_population_of_separates_generales_from_autonomicas():
    assert population_of('es') == 'generales' and population_of('es-md') == 'autonomicas'


def test_bootstrap_ci_resamples_houses_keeping_all_their_pairs():
    sizes = []

    def statistic(frame):
        sizes.append(len(frame))
        return float(frame['d_obs'].mean())

    lo, hi = bootstrap_ci(PAIRS, statistic, replicas=30, seed=3)
    assert lo <= hi and set(sizes) == {12}              # cada casa trae sus 3 pares y hay 4 casas
    assert bootstrap_ci(PAIRS, statistic, replicas=30, seed=3) == (lo, hi)


def test_discrimination_perfect_inverse_order_gives_minus_one():
    out = discrimination(PAIRS, tau=.01, replicas=50)
    assert out['all']['spearman']['value'] == pytest.approx(-1.0)
    assert out['all']['n_pairs'] == 12 and out['all']['n_events'] == 3 and out['all']['n_pollsters'] == 4
    assert out['generales']['n_events'] == 1 and out['autonomicas']['n_events'] == 2
    assert out['dates'] == {'first': '2019-11-10', 'last': '2021-05-04'}


def test_discrimination_slope_skips_prior_only_pairs():
    out = discrimination(PAIRS, tau=.01, replicas=50)
    assert out['all']['slope']['n'] == 9          # B (d_prev NaN) queda fuera
    assert out['all']['skill']['n'] == 12          # pero entra en el skill con rating == quality
    assert out['all']['slope_polled']['n'] == 6    # sólo las dos elecciones con 12 sondeos (sin B)
    assert out['all']['spearman_evidence']['high']['n'] == 6 and out['all']['spearman_evidence']['low']['n'] == 6


def test_slope_matches_a_hand_fit_and_needs_three_pairs():
    sub = PAIRS[PAIRS['event_scope'] == 'es']
    x, y = sub['d_prev'].dropna(), sub.loc[sub['d_prev'].notna(), 'd_obs']
    assert slope(sub, replicas=20)['value'] == pytest.approx(np.polyfit(x, y, 1)[0])
    few = slope(PAIRS.iloc[:2])
    assert np.isnan(few['value']) and few['n'] == 1 and np.isnan(few['lo'])


def test_empty_population_returns_nan_values_and_zero_counts():
    only_es = PAIRS[PAIRS['event_scope'] == 'es']
    out = discrimination(only_es, tau=.01, replicas=20)['autonomicas']
    assert out['n_pairs'] == 0 and np.isnan(out['spearman']['value']) and out['spearman']['n'] == 0
    assert out['slope']['n'] == 0 and out['skill']['n'] == 0 and np.isnan(out['skill']['skill_rating'])


def test_skill_of_a_perfect_predictor_is_one():
    pairs = PAIRS.assign(d_obs=lambda df: -.01 * norm.ppf(df['rating']))
    assert skill(pairs, tau=.01)['skill_rating'] == pytest.approx(1.0)


def test_skill_uses_minus_tau_times_inverse_normal_of_quality_and_rating():
    out = skill(PAIRS, tau=.01)
    d = PAIRS['d_obs'].to_numpy()
    assert out['mse_none'] == pytest.approx(np.mean(d ** 2))
    assert out['mse_prior'] == pytest.approx(np.mean((d + .01 * norm.ppf(PAIRS['quality'])) ** 2))
    assert out['skill_rating'] == pytest.approx(1 - out['mse_rating'] / out['mse_none'])


def test_quarters_rank_inside_each_election():
    rows = quarters(PAIRS)
    assert [r['quarter'] for r in rows if r['population'] == 'all'] == [1, 2, 3, 4]
    assert [r['population'] for r in rows] == ['all'] * 4 + ['generales'] * 4 + ['autonomicas'] * 4
    assert rows[0]['n'] == 3                        # la casa A es el cuarto 1 en las tres elecciones
    assert rows[0]['error_avg'] == pytest.approx(np.mean([2., 3., 4.]))


def test_simple_average_errors_drops_small_elections_and_pairs_the_two_weights():
    out = simple_average_errors(POLLS)
    assert set(out['event_date']) == {'2019-11-10', '2021-05-04'} and (out['n_pollsters'] >= 3).all()
    assert len(out) == 2 and 'es-cl' not in set(out['event_scope'])
    row = out.set_index(['event_scope', 'event_date']).loc[('es', '2019-11-10')]
    assert row['mae_flat'] == pytest.approx(EXPECTED_FLAT) and row['mae_rated'] == pytest.approx(EXPECTED_RATED)
    assert row['diff'] == pytest.approx(EXPECTED_RATED - EXPECTED_FLAT) and row['n_polls'] == 3


def test_summarize_simple_average_counts_better_and_worse():
    out = summarize_simple_average(simple_average_errors(POLLS), replicas=50)
    assert out['all']['better'] + out['all']['worse'] <= out['all']['n'] and out['all']['lo'] <= out['all']['diff'] <= out['all']['hi']
    assert out['generales']['n'] == 1 and out['autonomicas']['n'] == 1


def test_scatter_rows_scale_by_100_and_fit_line_spans_d_prev():
    rows = scatter_rows(PAIRS)
    assert len(rows) == 9 and rows[0] == {'scope': 'es', 'event_date': '2019-11-10', 'pollster': 'A', 'd_prev': -1.5, 'd_obs': -2.0}
    (x0, _), (x1, _) = fit_line(PAIRS)
    assert x0 == pytest.approx(-1.5) and x1 == pytest.approx(2.5)
    assert fit_line(PAIRS.iloc[:2]) is None


def test_rating_check_summary_has_the_part_shape_and_scaled_scatter():
    data = rating_check_summary(PAIRS, POLLS, tau=.01, sigma=.014, params={})
    assert set(data) == {'summary', 'quarters', 'scatter', 'fit', 'dates', 'params'}
    assert data['params']['k'] == pytest.approx(1.96) and data['scatter'][0]['d_prev'] == pytest.approx(PAIRS['d_prev'].dropna().iloc[0] * 100, abs=.01)
    assert all('average' in data['summary'][p] for p in POPULATIONS)
    assert data['params']['replicas'] == 2000 and data['params']['quarters'] == 4


def test_rating_check_summary_accepts_the_precomputed_simple_average():
    """Con `elections` (la salida de `simple_average_errors`) el resumen es idéntico al que la calcula él."""
    data = rating_check_summary(PAIRS, POLLS, tau=.01, sigma=.014, params={})
    same = rating_check_summary(PAIRS, POLLS, tau=.01, sigma=.014, params={}, elections=simple_average_errors(POLLS))
    assert json.dumps(same, sort_keys=True) == json.dumps(data, sort_keys=True)  # NaN != NaN en un `==`


def test_rating_check_summary_drops_the_mses_but_skill_keeps_them():
    data = rating_check_summary(PAIRS, POLLS, tau=.01, sigma=.014, params={})
    assert all(set(data['summary'][p]['skill']) == {'skill_prior', 'skill_rating', 'n'} for p in POPULATIONS)
    assert {'mse_none', 'mse_prior', 'mse_rating'} <= set(skill(PAIRS, tau=.01))


def _two_party_polls(index=None, error=None):
    """Dos casas y un partido, pesos 1 y 3 (promedio de 1 y 2 -> (1 + 6) / 4 = 1.75)."""
    frame = pd.DataFrame({'event_scope': 'es', 'event_date': '2019-11-10', 'population': 'generales',
                          'pollster_id': [1, 2, 1, 2], 'pollster': list('ABAB'), 'party': ['X', 'X', 'Y', 'Y'],
                          'error': [1., 2., 3., 3.], 'weight_over': 1., 'weight_sample': [1., 3., 1., 3.],
                          'weight_rating': 1.})
    if index is not None:
        frame.index = index
    if error is not None:
        frame['error'] = error
    return frame


def test_simple_average_errors_is_correct_with_a_duplicated_index():
    out = simple_average_errors(_two_party_polls(index=[0, 1, 0, 1]), min_pollsters=2)
    # X: (1 + 6) / 4 = 1.75; Y: 3; media de 1.75 y 3
    assert out['mae_flat'].iloc[0] == pytest.approx((1.75 + 3.) / 2)
    same = simple_average_errors(_two_party_polls(), min_pollsters=2)
    assert out['mae_flat'].iloc[0] == pytest.approx(same['mae_flat'].iloc[0])


def test_simple_average_errors_ignores_rows_with_nan_error_or_zero_weight_parties():
    out = simple_average_errors(_two_party_polls(error=[1., 2., np.nan, 3.]), min_pollsters=2)
    assert out['mae_flat'].iloc[0] == pytest.approx((1.75 + 3.) / 2)     # la fila NaN no cuenta ni en sum(w)
    zero = _two_party_polls().assign(weight_rating=lambda df: np.where(df['party'] == 'Y', 0., 1.))
    row = simple_average_errors(zero, min_pollsters=2).iloc[0]
    assert row['mae_flat'] == pytest.approx(1.75) and row['mae_rated'] == pytest.approx(1.75)   # Y fuera de los dos


def test_slope_and_fit_line_ignore_infinite_d_prev_and_share_one_fit():
    pairs = PAIRS.copy()
    pairs.loc[0, 'd_prev'] = np.inf
    assert slope(pairs, replicas=10)['n'] == 8 and len(scatter_rows(pairs)) == 8
    (x0, y0), (x1, y1) = fit_line(pairs)
    assert (y1 - y0) / (x1 - x0) == pytest.approx(slope(pairs, replicas=10)['value'])
