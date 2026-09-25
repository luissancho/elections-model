"""Efectos de casa (M6) contra la base de datos local."""
import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def he23(app):
    """Efectos de casa medidos en la elección de 2023 (un solo evento: ~1 min)."""
    from mtpy.lib.computer import Computer
    comp = Computer(scope='es', event_dates=['2023-07-23'], verbose=0, path='.').build_series()
    return comp.compute_house_effects(save=False)


def test_house_effects_table_columns_and_centering(he23):
    df = he23
    expected = {'event_date', 'event_scope', 'pollster_id', 'party_id', 'pollster', 'party', 'level', 'n_result',
                'dev_result', 'dev_result_err', 'dev_result_c', 'industry', 'n_cycle', 'dev_cycle', 'dev_cycle_err'}
    assert expected <= set(df.columns)
    assert (pd.to_datetime(df['event_date']) == pd.Timestamp('2023-07-23')).all()
    assert df['party'].nunique() >= 4 and df['pollster'].nunique() >= 5
    # Centrado por partido con los pesos de las encuestas: la media de `dev_result_c` es 0 y `industry` es lo restado
    with_result = df.dropna(subset=['dev_result_c'])
    for party, g in with_result.groupby('party', observed=True):
        assert abs((g['dev_result_c'] * g['n_result']).sum() / g['n_result'].sum()) < 1e-6, party
    assert np.allclose(with_result['industry'], with_result['dev_result'] - with_result['dev_result_c'])


def test_house_effects_2023_cis_below_consensus_for_pp(he23):
    """En el ciclo 2019-2023 el CIS dio al PP menos que el consenso y al PSOE más."""
    cis = he23.loc[he23['pollster'] == 'CIS'].set_index('party')
    assert cis.loc['PP', 'dev_cycle'] < -1 and cis.loc['PSOE', 'dev_cycle'] > 1
    assert cis.loc['PP', 'n_cycle'] >= 10 and np.isfinite(cis.loc['PP', 'dev_cycle_err'])


@pytest.mark.skipif(not os.environ.get('MTPY_TEST_DB_WRITE'), reason='escribe en pollsters_parties: MTPY_TEST_DB_WRITE=1 para ejecutarlo')
def test_house_effects_roundtrip_in_database(app, he23):
    """Guardar reemplaza las filas del evento (la tabla se crea con sus columnas si aún no las tiene)."""
    from mtpy.lib.data import save_house_effects_data, get_house_effects
    n = save_house_effects_data(he23)
    assert n == he23.shape[0]
    back = get_house_effects('es', ['2023-07-23'])
    assert back.shape[0] == he23.shape[0]
    merged = he23.merge(back, on=['pollster_id', 'party_id'], suffixes=('', '_db'))
    assert np.allclose(merged['dev_cycle'].astype(float), merged['dev_cycle_db'].astype(float), atol=1e-3, equal_nan=True)
    assert get_house_effects('es', []).shape[0] == 0


def test_compute_errors_adds_within_block_bias_and_new_mix(app):
    """M6b: `bias_within` mide el reparto dentro del bloque y `bias` es la mezcla entre/dentro (sin guardar)."""
    from mtpy.lib.computer import Computer
    # Dos elecciones: los bloques `vs` se fusionan (PSOE+UP+MP+SUMAR) y hay encuestas sin `bias_within`
    comp = Computer(scope='es', event_dates=['2019-11-10', '2023-07-23'], verbose=0, path='.').build_series()
    assert comp.error_weights == {'blocks': .7, 'within': .3}
    df = comp.compute_errors(save=False)
    assert {'bias_within', 'error_within'} <= set(df.columns) and {'bias_within', 'error_within'} <= set(comp.series.columns)
    within = df.dropna(subset=['error_within'])
    assert within.shape[0] > 100 and (within['error_within'] >= -1e-9).all()   # puntos, nunca negativo
    # La mezcla nunca pierde filas: donde falta `bias_within`, `bias` es `bias_blocks` (revisión M6: `Stat` corrompía los pesos)
    assert df['bias'].notnull().sum() == df['bias_blocks'].notnull().sum()
    only = df['bias_blocks'].notnull() & df['bias_within'].isnull()
    if only.any():
        assert np.allclose(df.loc[only, 'bias'], df.loc[only, 'bias_blocks'], atol=0.02)
    ok = df.dropna(subset=['bias_within', 'bias_blocks', 'bias'])
    assert ok.shape[0] > 100
    assert (ok['bias_within'] >= 0).all()
    # La mezcla se hace en log-odds y se guarda en "% de odds": deshacer la transformación y comprobar los pesos
    lor = ok[['bias_blocks', 'bias_within', 'bias']].apply(comp.bias_to_lor)
    assert np.allclose(lor['bias'], 0.7 * lor['bias_blocks'] + 0.3 * lor['bias_within'], atol=0.02)
    # Las encuestas que aciertan el bloque pero reparten mal tienen `bias_within` alto y `bias_blocks` bajo
    assert ok['bias_within'].corr(ok['bias_blocks']) < 0.9
