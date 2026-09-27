"""Tests de la librería estadística propia (`mtpy/core/utils/stat.py`).

Cada test reproduce un fallo detectado en el análisis de septiembre de 2026 y comprueba la corrección
contrastando, cuando es posible, con numpy/scipy/statsmodels.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from mtpy.core.utils.stat import Kernel, KernelDensityEstimator, LeastSquaresEstimator, LocalKernelEstimator, Stat


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


def test_stat_does_not_mutate_the_callers_weights():
    """M6 (revisión): `Stat` reescalaba los pesos recibidos in situ; reutilizarlos fila a fila los corrompía en cuanto un dato era NaN."""
    import pandas as pd
    from mtpy.core.utils.stat import Stat
    w = pd.Series([0.6, 0.4], index=['a', 'b'])
    assert Stat(pd.Series([1., np.nan], index=['a', 'b']), weights=w).mean() == pytest.approx(1.)
    assert w.tolist() == [0.6, 0.4]
    assert Stat(pd.Series([1., 3.], index=['a', 'b']), weights=w).mean() == pytest.approx(1.8)
    arr = np.array([0.6, 0.4])
    Stat(np.array([np.nan, 2.]), weights=arr).mean()
    assert arr.tolist() == [0.6, 0.4]


# --- M10: error estándar robusto por grupos (casas) ---

def _grouped_sample(rng, n_groups=10, per_group=15, sd_group=1., sd_noise=0.5):
    """Regresión lineal con un efecto aleatorio por grupo: los residuos de un grupo están correlados."""
    g = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=(g.size, 2))
    y = 1 + x @ np.array([1., -2.]) + rng.normal(0, sd_group, size=n_groups)[g] + rng.normal(0, sd_noise, size=g.size)
    return x, y, g


def test_wls_cluster_matches_statsmodels(rng):
    """M10: con pesos iguales, la covarianza `cluster` (CR1) coincide con statsmodels `cov_type='cluster'`."""
    sm = pytest.importorskip('statsmodels.api')
    x, y, g = _grouped_sample(rng)
    est = LeastSquaresEstimator(x, y, groups=g, cov_type='cluster').fit()
    ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type='cluster', cov_kwds={'groups': g})
    assert np.allclose(est.coef.squeeze(), ref.params)
    assert np.allclose(np.sqrt(np.diag(est.mcov)), ref.bse, rtol=1e-6), (np.sqrt(np.diag(est.mcov)), ref.bse)


def test_wls_hc1_matches_statsmodels(rng):
    """M10: `hc1` (sándwich con la corrección `neff / dof`) es el HC1 de statsmodels cuando los pesos son iguales."""
    sm = pytest.importorskip('statsmodels.api')
    x, y, g = _grouped_sample(rng)
    est = LeastSquaresEstimator(x, y, cov_type='hc1').fit()
    ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type='HC1')
    assert np.allclose(np.sqrt(np.diag(est.mcov)), ref.bse, rtol=1e-6)


def test_cluster_error_captures_group_effects_and_hc1_does_not(rng):
    """M10 (Monte Carlo): con efectos por grupo, la sd real de la media ajustada la da el error `cluster`,
    no el `hc1`, también con pesos desiguales."""
    means, err_cluster, err_hc1 = [], [], []
    for _ in range(300):
        x, y, g = _grouped_sample(rng, n_groups=12, per_group=8)
        w = rng.uniform(0.5, 2., size=y.size)
        p = np.zeros((1, 2))
        est = LeastSquaresEstimator(x, y, weights=w, groups=g, cov_type='cluster').fit(p, alpha=0.05)
        means.append(float(est['mean'].iloc[0]))
        err_cluster.append(float(est['err'].iloc[0]))
        err_hc1.append(float(LeastSquaresEstimator(x, y, weights=w, cov_type='hc1').fit(p, alpha=0.05)['err'].iloc[0]))
    sd = float(np.std(means))
    assert np.mean(err_cluster) == pytest.approx(sd, rel=0.15), (np.mean(err_cluster), sd)
    assert np.mean(err_hc1) < 0.7 * sd


def test_groups_follow_the_rows_dropped_for_missing_values(rng):
    """M10: `groups` se filtra con las mismas filas que `y` (NaN) y sobrevive con el tamaño de la muestra."""
    x, y, g = _grouped_sample(rng)
    y[[3, 17, 40]] = np.nan
    est = LeastSquaresEstimator(x, y, groups=g, cov_type='cluster').fit()
    assert est.groups.shape[0] == est.nobs == y.size - 3
    assert np.array_equal(est.groups, g[np.isfinite(y)])


def test_local_estimator_passes_the_groups_of_the_window(rng):
    """M10: el estimador local recibe los grupos de las observaciones de su ventana; sin grupos (o con uno
    solo) cae al `hc1` con aviso, sin romper el ajuste."""
    x = np.arange(100, dtype=float)
    g = x // 10
    y = 5 + rng.normal(0, 1, size=10)[g.astype(int)] + rng.normal(0, 0.3, size=100)
    est = LocalKernelEstimator(x, y, groups=g, bw=4.0, bw_type='fixed', cov_type='cluster')
    kws = est.build_pred(np.array([50.])).get_kernel().get_weights(est.pred)
    loc = est.get_local_estimator(kws, 0)
    vind = np.abs(kws[0] * est.weights.squeeze()) >= 1e-2
    assert np.array_equal(loc.groups, g[vind]) and len(np.unique(loc.groups)) >= 2
    res = est.predict(np.array([50.]), alpha=0.05)
    hc1 = LocalKernelEstimator(x, y, bw=4.0, bw_type='fixed', cov_type='hc1').predict(np.array([50.]), alpha=0.05)
    assert np.isfinite(res['err'].iloc[0]) and res['err'].iloc[0] != pytest.approx(hc1['err'].iloc[0])
    with pytest.warns(UserWarning):
        none = LocalKernelEstimator(x, y, bw=4.0, bw_type='fixed', cov_type='cluster').predict(np.array([50.]), alpha=0.05)
    assert none['err'].iloc[0] == pytest.approx(hc1['err'].iloc[0])
    with pytest.warns(UserWarning):
        one = LocalKernelEstimator(x, y, groups=np.zeros(100), bw=4.0, bw_type='fixed', cov_type='cluster').predict(np.array([50.]), alpha=0.05)
    assert one['err'].iloc[0] == pytest.approx(hc1['err'].iloc[0])


def test_cluster_band_uses_g_minus_one_degrees_of_freedom(rng):
    """M10 (revisión): con covarianza `cluster` la banda usa la t con `G − 1` grados de libertad (convención
    CR1), nunca más que `dof`."""
    from scipy.stats import t
    x, y, g = _grouped_sample(rng, n_groups=6, per_group=20)
    p = np.zeros((1, 2))
    est = LeastSquaresEstimator(x, y, groups=g, cov_type='cluster')
    res = est.fit(p, alpha=0.05)
    assert est.dof_t == 5
    assert res['cmax'].iloc[0] - res['mean'].iloc[0] == pytest.approx(t.ppf(0.975, 5) * est.perr[0])
    plain = LeastSquaresEstimator(x, y, cov_type='hc1')
    plain.fit(p, alpha=0.05)
    assert plain.dof_t == plain.dof


def test_kernel_density_estimator_still_constructs(rng):
    """M10 (revisión): la firma nueva de `build_input` (`groups`) no rompe al estimador de densidad."""
    est = KernelDensityEstimator(rng.normal(size=100))
    assert est.data.shape == (100, 1)


def test_local_estimator_keeps_the_category_of_warnings_raised_in_the_loop(rng, monkeypatch):
    """M10 (revisión): los avisos de los ajustes locales se emiten una vez, con su categoría original."""
    x = np.arange(60, dtype=float)
    y = 5 + rng.normal(0, 0.3, size=60)
    original = LeastSquaresEstimator.fit

    def noisy_fit(self, p=None, alpha=None):
        import warnings
        warnings.warn('numerical trouble', RuntimeWarning)
        return original(self, p, alpha)

    monkeypatch.setattr(LeastSquaresEstimator, 'fit', noisy_fit)
    est = LocalKernelEstimator(x, y, bw=4.0, bw_type='fixed', cov_type='hc1')
    with pytest.warns(RuntimeWarning, match='numerical trouble') as record:
        est.predict(np.array([10., 20., 30.]), alpha=0.05)
    assert len([w for w in record if w.category is RuntimeWarning]) == 1

