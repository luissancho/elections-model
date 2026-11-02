"""Errores del promedio nowcast (`average_errors`) de una elección contra la base de datos, sin escribir en ella."""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def test_compute_average_errors_2023_matches_the_backtest_shares(app):
    """El promedio a 6 días de 2023 es el del backtest guardado (misma configuración: efectos de casa, muestra
    efectiva, rebaño, `max_fc` 10), y su error es el promedio menos el resultado oficial."""
    from mtpy.lib.computer import Computer

    df = Computer(scope='es').build_series().compute_average_errors(save=False, events=['2023-07-23'], horizons=(6,))
    ref = pd.read_csv('backtest/results/shares.csv').query("event_date == '2023-07-23' and horizon == 6 and main")
    merged = df.merge(ref, on='party', suffixes=('', '_ref'))

    assert set(df['party']) == {'PP', 'PSOE', 'VOX', 'SUMAR'} and len(merged) == 4
    assert (df['as_of'] == pd.Timestamp('2023-07-17')).all() and (df['n_polls'] >= 5).all()
    assert np.allclose(merged['level'], merged['model'], atol=0.05)   # mismo promedio que el backtest
    assert np.allclose(merged['official'], merged['official_ref'], atol=1e-6)
    assert np.allclose(df['error'], df['level'] - df['official']) and (df['err'] > 0).all()
