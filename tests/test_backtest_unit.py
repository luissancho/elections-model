"""Tests de las métricas del backtest (sin base de datos)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.backtest import brier, covered, crps_from_samples, log_score, summarize_case


def test_crps_point_mass_equals_absolute_error():
    assert crps_from_samples(np.array([5., 5., 5.]), 7.) == pytest.approx(2.)


def test_crps_two_point_distribution():
    # X ∈ {0, 2} equiprobable, y = 0: E|X-y| = 1, E|X-X'| = 1 -> CRPS = 0.5
    assert crps_from_samples(np.array([0., 2.]), 0.) == pytest.approx(0.5)


def test_crps_matches_brute_force(rng=np.random.default_rng(0)):
    x = rng.normal(size=200)
    y = 0.3
    brute = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert crps_from_samples(x, y) == pytest.approx(brute)


def test_brier_log_score_and_coverage():
    assert brier(0.8, True) == pytest.approx(0.04)
    assert brier(0.8, False) == pytest.approx(0.64)
    assert log_score(0.5, True) == pytest.approx(np.log(2))
    assert log_score(1.0, False) == pytest.approx(-np.log(1e-3))
    assert covered(1, 3, 2) and not covered(1, 3, 4) and not covered(np.nan, 3, 2)


def test_summarize_case_on_synthetic_frames():
    shares = pd.DataFrame({
        'party': ['A', 'B', 'C'], 'main': [True, True, False], 'official': [30., 20., 5.],
        'model': [31., 18., 6.], 'last_poll': [33., 17., 5.], 'mean_4w': [32., 19., 5.], 'prev_result': [28., 25., 4.],
        'fc_lo': [30.5, 17., 5.], 'fc_hi': [31.5, 19., 7.],
        'lo50': [30., 17.5, 5.5], 'hi50': [32., 18.5, 6.5], 'lo80': [29., 17., 5.], 'hi80': [33., 19., 7.],
        'lo95': [28., 16., 4.], 'hi95': [34., 20., 8.],
    })
    seats = pd.DataFrame({
        'party': ['A', 'B', 'C'], 'main': [True, True, False], 'official': [120., 80., 10.], 'model': [125., 75., 12.],
        'mean': [125., 75., 12.], 'crps': [3., 4., 1.], 'seats_valid': [True] * 3,
        'lo50': [122., 73., 11.], 'hi50': [128., 77., 13.], 'lo80': [118., 70., 9.], 'hi80': [132., 80., 15.],
        'lo95': [115., 68., 8.], 'hi95': [135., 82., 16.],
    })
    blocks = pd.DataFrame({'block': ['D', 'I'], 'brier': [0.04, 0.01], 'log_score': [0.2, 0.1], 'seats_valid': [True, True]})

    m = summarize_case(shares, seats, blocks)
    assert m['mae_shares'] == pytest.approx(1.5)          # sólo A y B (main)
    assert m['mae_last_poll'] == pytest.approx(3.0)
    assert m['cov_shares95'] == pytest.approx(1.0)
    assert m['cov_shares50'] == pytest.approx(0.5)        # 30 ∈ [30, 32]; 20 ∉ [17.5, 18.5]
    assert m['mae_seats'] == pytest.approx(5.0)
    assert m['cov_seats80'] == pytest.approx(1.0)
    assert m['crps_seats'] == pytest.approx(3.5)
    assert m['brier_vs'] == pytest.approx(0.025)
    assert m['seats_valid'] is True

    # Sin columnas `_h` (ejecución solo nowcast) las métricas del horizonte quedan a NaN
    assert np.isnan(m['cov_shares95_h']) and np.isnan(m['crps_seats_h']) and np.isnan(m['brier_vs_h'])

    seats_invalid = seats.assign(seats_valid=False)
    m = summarize_case(shares, seats_invalid, blocks)
    assert np.isnan(m['mae_seats']) and np.isnan(m['brier_vs'])


def test_summarize_case_horizon_columns():
    """M5: la segunda ejecución (a horizonte, sufijo `_h`) tiene sus propias coberturas, CRPS y Brier."""
    shares = pd.DataFrame({
        'party': ['A', 'B'], 'main': [True, True], 'official': [30., 20.], 'model': [31., 18.],
        'last_poll': [33., 17.], 'mean_4w': [32., 19.], 'prev_result': [28., 25.],
        'fc_lo': [30.5, 17.], 'fc_hi': [31.5, 19.],
        'lo50': [30.5, 17.5], 'hi50': [31.5, 18.5], 'lo80': [30.2, 17.], 'hi80': [32., 19.], 'lo95': [29.9, 16.], 'hi95': [34., 19.5],
        # Intervalos a horizonte más anchos: cubren los dos resultados al 95 % y al 50 % sólo A
        'lo50_h': [29.5, 16.5], 'hi50_h': [32.5, 19.5], 'lo80_h': [29., 16.], 'hi80_h': [33., 20.5],
        'lo95_h': [28., 15.], 'hi95_h': [34., 21.],
    })
    seats = pd.DataFrame({
        'party': ['A', 'B'], 'main': [True, True], 'official': [120., 80.], 'model': [125., 75.], 'mean': [125., 75.],
        'crps': [3., 4.], 'seats_valid': [True, True],
        'lo50': [122., 73.], 'hi50': [128., 77.], 'lo80': [118., 70.], 'hi80': [132., 80.], 'lo95': [115., 68.], 'hi95': [135., 82.],
        'model_h': [123., 77.], 'mean_h': [123., 77.], 'crps_h': [4., 5.],
        'lo50_h': [119., 72.], 'hi50_h': [127., 82.], 'lo80_h': [115., 68.], 'hi80_h': [131., 86.], 'lo95_h': [110., 65.], 'hi95_h': [136., 90.],
    })
    blocks = pd.DataFrame({
        'block': ['D', 'I'], 'brier': [0.04, 0.01], 'log_score': [0.2, 0.1], 'seats_valid': [True, True],
        'brier_h': [0.09, 0.04], 'log_score_h': [0.3, 0.2]
    })

    m = summarize_case(shares, seats, blocks)
    assert m['cov_shares95'] == pytest.approx(0.5)      # 20 ∉ [16, 19.5]
    assert m['cov_shares95_h'] == pytest.approx(1.0)
    assert m['cov_shares50_h'] == pytest.approx(0.5)    # 20 ∉ [16.5, 19.5]
    assert m['cov_seats50'] == pytest.approx(0.0)
    assert m['cov_seats50_h'] == pytest.approx(1.0)
    assert m['mae_seats_h'] == pytest.approx(3.0)
    assert m['crps_seats_h'] == pytest.approx(4.5)
    assert m['brier_vs_h'] == pytest.approx(0.065)
    assert m['log_score_vs_h'] == pytest.approx(0.25)
