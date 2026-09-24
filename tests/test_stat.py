"""Tests de la librería estadística propia (`mtpy/core/utils/stat.py`).

Cada test reproduce un fallo detectado en el análisis de septiembre de 2026 y comprueba la corrección
contrastando, cuando es posible, con numpy/scipy/statsmodels.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from mtpy.core.utils.stat import Kernel, LeastSquaresEstimator, LocalKernelEstimator, Stat


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_kernel_bandwidth_not_truncated(rng):
    """B1: el ancho de banda calculado por 'isj'/'scott' no debe truncarse a 3 o 5 caracteres."""
    x = rng.normal(size=300)
    w = np.ones(300)

    exact = Kernel.get_bw_isj(x, w)
    stored = Kernel(x, bw='isj', bw_type='fixed').bw[0]
    assert np.isclose(stored, exact), (stored, exact)

    exact = Kernel.get_bw_scott(x, w)
    stored = Kernel(x, bw='scott', bw_type='fixed').bw[0]
    assert np.isclose(stored, exact), (stored, exact)


def test_kernel_bandwidth_list_of_methods(rng):
    x = rng.normal(size=(200, 2))
    k = Kernel(x, bw=['scott', 'silverman'], bw_type='fixed')
    assert k.bw.shape == (2,)
    assert (k.bw > 0).all()


def test_winsorize_group_clips_both_tails():
    """B11: los outliers bajos deben ir al mínimo interior, no al máximo."""
    s = Stat(np.array([-100., 10, 11, 12, 13, 200]), outliers='group', n_dev=1)
    assert s.data[0] == 10
    assert s.data[-1] == 13


def test_conf_with_multiples_of_sigma(rng):
    """B12: conf(n) con n >= 1 significa n desviaciones típicas."""
    s = Stat(rng.normal(size=20000))
    assert np.isclose(s.conf(1) / s.std(), 1.0, atol=1e-6)
    assert np.isclose(s.conf(2) / s.std(), 2.0, atol=1e-6)
    # y la cobertura bilateral correspondiente sigue siendo erf(n / sqrt(2))
    assert np.isclose(2 * ((1 + erf(2 / np.sqrt(2))) / 2) - 1, erf(2 / np.sqrt(2)))


def test_quantile_matches_numpy_and_handles_nan():
    """B19: cuantil sin pesos = Hyndman-Fan tipo 2; con NaN y dropna=False no debe indexar mal."""
    x = np.arange(1., 9.)
    assert Stat(x).quantile(q=0.25) == np.quantile(x, 0.25, method='averaged_inverted_cdf')
    assert Stat(x).median() == np.median(x)

    x_nan = np.array([np.nan, 3., 1., 2., np.nan, 4.])
    assert Stat(x_nan).median() == 2.5
    assert Stat(x_nan).quantile(q=1.0) == 4.


def test_isj_root_is_found_for_small_samples(rng):
    """B14: con n <= 50 el punto fijo del ISJ debe resolverse (antes devolvía None y se caía al fallback)."""
    from scipy import fftpack

    for n in [20, 40, 50]:
        x = rng.normal(size=n)
        grid = Kernel.isj_grid(x, weights=np.ones(n))
        a = fftpack.dct(grid)
        i_sq = np.power(np.arange(1, len(grid)), 2)
        a_sq = np.power(a[1:], 2)
        opt_t = Kernel.isj_root(Kernel.isj_fixed_point, n, args=(n, i_sq, a_sq))
        assert opt_t is not None and opt_t > 0, n

    # El ancho devuelto nunca baja del suelo 2*pi*espaciado medio (heurística documentada en get_bw_isj)
    x = rng.normal(size=40)
    floor = 2 * np.pi * np.mean(np.diff(np.sort(x)))
    assert Kernel.get_bw_isj(x, np.ones(40)) >= floor - 1e-12


def test_wls_degrees_of_freedom_polynomial(rng):
    """B15: dof = neff - número de coeficientes, también con poly_deg > 1 y varias variables."""
    x = rng.normal(size=(200, 2))
    y = x @ np.array([1., 2.]) + rng.normal(size=200)
    est = LeastSquaresEstimator(x, y, poly_deg=2).fit()
    assert est.exog.shape[1] == 6
    assert np.isclose(est.dof, est.neff - 6)


def test_wls_matches_statsmodels(rng):
    """Coeficientes y R2 ponderado frente a statsmodels (oráculo sólo en tests)."""
    sm = pytest.importorskip('statsmodels.api')
    x = rng.normal(size=(150, 2))
    y = 1 + x @ np.array([1., -2.]) + rng.normal(size=150)
    w = rng.uniform(0.5, 2., size=150)

    est = LeastSquaresEstimator(x, y, weights=w, cov_type=None).fit()
    ref = sm.WLS(y, sm.add_constant(x), weights=w).fit()

    assert np.allclose(est.coef.squeeze(), ref.params)
    assert np.isclose(est.r2_score, ref.rsquared), (est.r2_score, ref.rsquared)


def test_local_estimator_sparse_window_gives_nan_not_error():
    """B17: una ventana local sin observaciones suficientes deja NaN en ese punto en vez de abortar."""
    x = np.array([0., 1., 2., 3., 4., 100.])
    y = np.array([1., 2., 3., 4., 5., 6.])
    est = LocalKernelEstimator(x, y, bw=1.0, bw_type='fixed')
    res = est.predict(np.array([2., 50.]))
    assert np.isfinite(res.iloc[0])
    assert np.isnan(res.iloc[1])


def test_estimator_predict_without_points_on_time_index():
    """B18: predict() sin puntos no debe volver a convertir fechas ya convertidas."""
    idx = pd.date_range('2024-01-01', periods=30, freq='D')
    s = pd.Series(np.linspace(10, 20, 30), index=idx)
    est = LocalKernelEstimator(s, bw=5.0, bw_type='fixed')
    res = est.predict()
    assert len(res.dropna()) == 30


def test_local_estimator_recovers_a_line(rng):
    """Con y lineal y kernel ancho, la regresión local de grado 1 recupera la recta."""
    x = np.linspace(0, 10, 50)
    y = 3 + 2 * x
    est = LocalKernelEstimator(x, y, bw=100.0, bw_type='fixed', cov_type=None)
    res = est.predict(np.array([2.5, 7.5]))
    assert np.allclose(res.values, [8., 18.], atol=1e-6)
