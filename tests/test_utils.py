"""Tests de `mtpy/lib/utils.py` (bloques de partidos)."""
import numpy as np
import pandas as pd
import pytest

from mtpy.lib.utils import build_blocks, group_results


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
