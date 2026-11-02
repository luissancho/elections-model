"""Un corte del afinado del promedio con los sondeos futuros, contra la base de datos (sólo lectura)."""
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def test_run_cut_2023_30_days_predicts_the_next_week_within_poll_noise(app):
    from mtpy.lib.tuning import default_config, run_cut

    out = run_cut('es', '2023-07-23', pd.Timestamp('2023-06-23'), config=default_config(), horizons=(7, 28))
    s7 = next(s for s in out['scores'] if s['k'] == 7)
    assert s7['n_polls'] >= 5 and 1.0 < s7['rmse'] < 4.0
    assert out['meta']['n_polls'] > 100 and out['meta']['drange'] == 30
