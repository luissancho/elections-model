"""Tests de integración de la carga de sondeos y resultados autonómicos (M11)."""
import pytest

pytestmark = pytest.mark.integration

# Sondeos utilizables por comunidad en la Wikipedia inglesa (01-10-2026): filas de la tabla de intención de
# voto con ventaja numérica, sin las filas de resultado ni los sondeos internos de partido (`skip`). El
# inventario de la spec (3.479) contaba todas las filas, también resultados y sondeos sólo de escaños.
INVENTORY = {
    'es-an': 350, 'es-ar': 136, 'es-as': 99, 'es-ib': 98, 'es-cn': 110, 'es-cb': 76, 'es-cl': 148, 'es-cm': 128,
    'es-ct': 509, 'es-vc': 215, 'es-ex': 118, 'es-ga': 261, 'es-md': 276, 'es-mc': 129, 'es-nc': 112, 'es-pv': 224,
    'es-ri': 80
}


def test_regional_polls_cover_the_inventory(app):
    from mtpy.models.elections import Polls
    polls = Polls().get_results(formatted=True)
    counts = polls.loc[polls['event_scope'] != 'es'].groupby('event_scope', observed=True).size()
    assert counts.shape[0] == 17
    assert counts.sum() >= 0.95 * sum(INVENTORY.values())
    # Por ámbito el listón es el 90 %: las casas con uno o dos sondeos no se dan de alta
    for scope, n in INVENTORY.items():
        assert counts[scope] >= 0.90 * n, scope


def test_regional_results_are_consistent(app):
    from mtpy.models.elections import EventsData, EventsResults
    data = EventsData().get_results(query=dict(filters=["scope <> 'es'"]), formatted=True)
    res = EventsResults().get_results(query=dict(filters=["scope <> 'es'"]), formatted=True)
    past = data.loc[(data['region_id'] == 0) & data['votes'].notnull()]
    assert past['scope'].nunique() == 17
    for _, row in past.iterrows():
        mine = res.loc[(res['scope'] == row['scope']) & (res['date'] == row['date'])]
        votes = mine.loc[mine['region_id'] == 0, 'votes'].sum()
        assert abs(votes - (row['votes'] - row['blank'])) <= 0.001 * row['votes'], (row['scope'], row['date'])
        assert mine.loc[mine['region_id'] > 0, 'seats'].sum() == row['seats'], (row['scope'], row['date'])
    upcoming = data.loc[(data['region_id'] == 0) & data['votes'].isnull()]
    assert upcoming['scope'].nunique() == 17 and (upcoming['seats'] > 0).all()


def test_event_params_work_for_an_event_without_polls(app):
    # Foco 1: una elección sin sondeos cargados (la más antigua de cada ámbito puede serlo) sigue siendo utilizable
    from mtpy.lib.data import get_event_params
    params = get_event_params('es-as', '2023-05-28', path='.')
    assert 'PSOE' in params['parties']['event'] and 'PP' in params['bmaps']['main']


def test_featured_events_have_results_and_enough_polls(app):
    from mtpy.models.elections import Events, EventsData, Polls
    events = Events().get_results(query=dict(filters=["scope <> 'es'"]), formatted=True)
    data = EventsData().get_results(query=dict(filters=["scope <> 'es'", 'region_id = 0']), formatted=True)
    polls = Polls().get_results(query=dict(filters=["event_scope <> 'es'"]), formatted=True)
    useful = polls.loc[polls['computed'] & (polls['weight_over'] > 0)].groupby(['event_scope', 'event_date'], observed=True).size()
    featured = events.loc[events['featured']]
    assert featured['scope'].nunique() == 17
    for _, row in featured.iterrows():
        assert useful.get((row['scope'], row['date']), 0) >= 10, (row['scope'], row['date'])
        assert data.loc[(data['scope'] == row['scope']) & (data['date'] == row['date']), 'votes'].notnull().all()
    assert not events.loc[events['date'] > '2026-10-01', 'featured'].any()
