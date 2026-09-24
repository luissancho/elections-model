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
