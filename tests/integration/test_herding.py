"""Herding (M13): casas cuyas cifras se mueven menos de lo que permite su muestra. Necesita la base de datos."""
import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def fc26(app):
    from mtpy.lib.forecaster import Forecaster
    return Forecaster(scope='es', event_date='2026-11-29', drange=6, house_effects=False, dispersion=False, verbose=0, path='.').build_series()


def test_measure_herding_separates_herders_from_the_cis(fc26):
    herd = fc26.measure_herding()
    names = herd.set_index('pollster')
    assert names.loc['Hamalgama', 'ratio'] < 0.7 and names.loc['Hamalgama', 'p_value'] < 0.05
    assert names.loc['CIS', 'ratio'] > 2 and names.loc['CIS', 'p_value'] > 0.95
    assert (names['n'] >= 5).all() and (names['n_series'] == 4).all()
    # La medida no altera los pesos del promedio
    assert fc26.series['weight'].notnull().any()


def test_symmetric_dispersion_penalizes_herders_by_default(app):
    from mtpy.lib.forecaster import Forecaster
    kw = dict(scope='es', event_date='2026-11-29', drange=6, house_effects=False, verbose=0, path='.')
    base = Forecaster(**kw, disp_params={'herding': False}).build_series()
    base.fit_dispersion()
    herd = Forecaster(**kw).build_series()
    herd.fit_dispersion()
    b, h = base.dispersion.set_index('pollster'), herd.dispersion.set_index('pollster')
    assert 'herd_ratio' not in b.columns
    assert b.loc['Hamalgama', 'factor'] == pytest.approx(1.) and h.loc['Hamalgama', 'factor'] < 0.8
    # Las sobredispersas no cambian: manda M12
    assert h.loc['CIS', 'factor'] == pytest.approx(b.loc['CIS', 'factor'])
    over = h['ratio'] > 1
    assert np.allclose(h.loc[over, 'factor'], 1. / h.loc[over, 'ratio'])
    assert (h.loc[~over, 'factor'] <= 1.).all()


def test_computer_herding_table_and_summary(app):
    from mtpy.lib.computer import Computer
    comp = Computer(scope='es', event_dates=['2023-07-23'], verbose=0, path='.').build_series()
    df = comp.compute_herding(save=False)
    assert list(df.columns) == ['event_date', 'event_scope', 'pollster_id', 'pollster', 'n', 'n_series', 'ss_obs', 'ss_exp', 'dof', 'ratio', 'p_value']
    assert set(df['event_date'].dt.strftime('%Y-%m-%d')) == {'2023-07-23'}
    by = df.set_index('pollster')
    assert by.loc['Hamalgama', 'ratio'] < 0.8 and by.loc['CIS', 'ratio'] > 1.5
    summary = comp.herding_summary()
    assert summary.loc['Hamalgama', 'herding'] == pytest.approx(by.loc['Hamalgama', 'ratio'])
