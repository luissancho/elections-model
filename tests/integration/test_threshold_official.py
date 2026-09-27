"""M1: barrera del 3 % y D'Hondt sobre los votos oficiales reproducen los escaños oficiales.

Cinco elecciones (2015-2023) x 52 provincias = 260 repartos. Base de la barrera: `events_data.votes`
(votos válidos = candidaturas + blanco), en int32 en la base de datos.
"""
import pytest

pytestmark = pytest.mark.integration

DATES = ['2015-12-20', '2016-06-26', '2019-04-28', '2019-11-10', '2023-07-23']


def _cases(app):
    from mtpy.lib.data import get_event_data, get_event_results

    data = get_event_data('es', DATES)
    res = get_event_results('es', DATES)
    data['region_id'] = data['region_id'].astype(int)
    res['region_id'] = res['region_id'].astype(int)

    for dt in DATES:
        d = data.loc[data['date'].astype(str) == dt].set_index('region_id')
        r = res.loc[(res['date'].astype(str) == dt) & (res['party_id'] > 0)]
        for reg, g in r.groupby('region_id'):
            if reg == 0:
                continue
            votes = g.set_index('party')['votes'].astype('int64').to_dict()
            official = g.set_index('party')['seats'].astype(int).to_dict()
            yield dt, int(reg), votes, official, int(d.loc[reg, 'votes']), int(d.loc[reg, 'seats'])


def test_threshold_dhondt_reproduces_official_seats(app):
    from mtpy.lib.simulator import Simulator

    cases = list(_cases(app))
    assert len(cases) == 5 * 52

    bad = [
        (dt, reg) for dt, reg, votes, official, valid, n_seats in cases
        if Simulator.alloc_seats(votes, n_seats, valid_votes=valid, threshold=3.0) != official
    ]
    assert bad == []


def test_without_threshold_only_barcelona_2019_04_differs(app):
    from mtpy.lib.simulator import Simulator

    bad = [
        (dt, reg) for dt, reg, votes, official, valid, n_seats in _cases(app)
        if Simulator.alloc_seats(votes, n_seats, valid_votes=valid, threshold=None) != official
    ]
    # Front Republicà (2,72 % de los válidos) habría obtenido el escaño que fue a Unidas Podemos
    assert bad == [('2019-04-28', 8)]
