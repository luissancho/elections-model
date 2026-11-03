"""Los pesos que recalcula `Computer.compute_weights` (con `reweight` y `poll_overweights` tras el refactor del
afinado) son los guardados en la base por el último `compute`. Necesita la base de datos (sólo lectura)."""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

COLUMNS = ['mtype', 'proc_sample', 'weight_sample', 'weight_over', 'weight_rating', 'rating']


# `Computer.reweight` agrupa por `pollster`, categórica en la serie de la base, sin `observed`: pandas avisa con un
# FutureWarning (que `pytest.ini` vuelve error) y el resultado no cambia, porque las medianas vacías se descartan
@pytest.mark.filterwarnings('ignore:The default of observed=False is deprecated:FutureWarning')
def test_compute_weights_reproduces_the_stored_weights_of_the_2023_cycle(app):
    from mtpy.lib.computer import Computer

    # Toda la serie de `es`, como la calcula el job `compute`: las medianas de muestra de cada casa cruzan elecciones
    comp = Computer(scope='es', verbose=0).build_series()
    stored = comp.polls[COLUMNS].copy()
    new = comp.compute_weights(save=False, overwrite=True)
    cycle = new.index.get_level_values('event_date') == pd.Timestamp('2023-07-23')
    assert cycle.sum() > 100
    stored, new = stored.reindex(new.index).loc[cycle], new.loc[cycle, COLUMNS]
    a, b = stored['mtype'].astype(object), new['mtype'].astype(object)
    assert ((a == b) | (a.isnull() & b.isnull())).all()
    for column in COLUMNS[1:]:
        a, b = stored[column].astype(float), new[column].astype(float)
        same = np.isclose(a, b, rtol=0, atol=1e-9) | (a.isnull() & b.isnull())
        assert same.all(), '{}: {} of {} polls differ'.format(column, int((~same).sum()), len(same))
