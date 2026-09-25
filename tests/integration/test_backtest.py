"""Backtest de un caso pequeño contra la base de datos."""
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_run_case_2023_30_days(app):
    from mtpy.lib.backtest import run_case, summarize_case

    case = run_case('es', '2023-07-23', 30, n_sim=20, seed=42)
    shares, seats, blocks, meta = case['shares'], case['seats'], case['blocks'], case['meta']

    assert meta.loc[0, 'limit_date'] == '2023-06-23'
    assert meta.loc[0, 'prev_date'] == '2019-11-10'
    assert set(shares['party']) == set(seats['party'])
    assert shares.loc[shares['party'] == 'PP', 'official'].iloc[0] == pytest.approx(33.05, abs=0.1)
    assert seats.loc[seats['party'] == 'PP', 'official'].iloc[0] == 137
    assert (seats['lo95'] <= seats['hi95']).all()
    assert seats['seats_valid'].all()
    assert list(blocks['block']) == ['Derecha', 'Izquierda']
    assert ((blocks['p_majority'] >= 0) & (blocks['p_majority'] <= 1)).all()

    m = summarize_case(shares, seats, blocks)
    assert np.isfinite(m['mae_shares']) and np.isfinite(m['mae_seats']) and np.isfinite(m['brier_vs'])
    assert 0 <= m['cov_seats95'] <= 1


def test_run_case_horizon_run_2023_90_days(app):
    """M5: cada caso se simula también a horizonte (`deadline` = la elección), con columnas `_h`."""
    from mtpy.lib.backtest import run_case, summarize_case

    case = run_case('es', '2023-07-23', 90, n_sim=20, seed=42)
    shares, seats, blocks, meta = case['shares'], case['seats'], case['blocks'], case['meta']

    assert meta.loc[0, 'as_of'] <= meta.loc[0, 'limit_date']
    assert meta.loc[0, 'horizon_max'] >= 90
    assert np.isfinite(meta.loc[0, 'drift_k']) and meta.loc[0, 'drift_k'] > 0
    pp = shares.loc[shares['party'] == 'PP'].iloc[0]
    assert pp['lo95_h'] <= pp['lo95'] and pp['hi95_h'] >= pp['hi95']
    assert {'model_h', 'mean_h', 'crps_h', 'lo95_h', 'hi95_h'} <= set(seats.columns)
    assert {'model_seats_h', 'p_majority_h', 'brier_h', 'log_score_h'} <= set(blocks.columns)

    m = summarize_case(shares, seats, blocks)
    assert 0 <= m['cov_shares95_h'] <= 1
    assert np.isfinite(m['crps_seats_h']) and np.isfinite(m['brier_vs_h'])

    # Sólo nowcast: sin columnas `_h` y con el mismo nowcast (misma semilla)
    only = run_case('es', '2023-07-23', 90, n_sim=20, seed=42, nowcast_only=True)
    assert 'lo95_h' not in only['shares'].columns
    assert only['shares']['lo95'].tolist() == pytest.approx(shares['lo95'].tolist())


def test_run_case_house_effects_flags_and_raw_baselines(app):
    """M6: el caso registra los efectos de casa aplicados y las líneas base se calculan con la serie bruta."""
    from mtpy.lib.backtest import run_case

    case = run_case('es', '2023-07-23', 30, n_sim=10, seed=42, nowcast_only=True)
    meta = case['meta'].iloc[0]
    assert bool(meta['house_effects']) is True
    assert meta['he_pollsters'] > 0 and np.isfinite(meta['he_mean_abs']) and meta['he_max_abs'] >= meta['he_mean_abs']

    off = run_case('es', '2023-07-23', 30, n_sim=10, seed=42, nowcast_only=True, house_effects=False)
    assert bool(off['meta'].iloc[0]['house_effects']) is False and off['meta'].iloc[0]['he_pollsters'] == 0
    # Las líneas base (última encuesta, media de 4 semanas) no dependen de la corrección
    a, b = case['shares'].set_index('party'), off['shares'].set_index('party')
    assert a['last_poll'].tolist() == pytest.approx(b['last_poll'].tolist(), nan_ok=True)
    assert a['mean_4w'].tolist() == pytest.approx(b['mean_4w'].tolist(), nan_ok=True)
    # El promedio sí cambia con la corrección
    assert not np.allclose(a.loc[['PP', 'PSOE'], 'model'], b.loc[['PP', 'PSOE'], 'model'])
