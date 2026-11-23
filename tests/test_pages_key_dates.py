"""Tests de las fechas clave de una elección (`dates` de `data/params.json`) como marcas de los gráficos
(`pages.key_dates`)."""
from mtpy.lib import pages


def test_key_dates_give_the_call_the_campaign_and_the_key_dates_by_date():
    """La convocatoria, el inicio de la campaña y cada fecha clave, por fecha, con su etiqueta y su tipo."""
    params = {'dates': {'campaign': '2004-02-27', 'call': '2004-01-19', 'key': {'2004-03-11': '11-M'}}}
    assert pages.key_dates(params) == [
        {'date': '2004-01-19', 'label': 'Convocatoria', 'kind': 'call'},
        {'date': '2004-02-27', 'label': 'Campaña', 'kind': 'campaign'},
        {'date': '2004-03-11', 'label': '11-M', 'kind': 'key'}]
    assert pages.key_dates({}) == [] and pages.key_dates({'bmaps': {}}) == [] and pages.key_dates({'dates': None}) == []
    # the three keys are optional, and a key date on the day of the call goes after it
    same_day = pages.key_dates({'dates': {'call': '2026-10-05', 'key': {'2026-10-05': 'X'}}})
    assert [entry['kind'] for entry in same_day] == ['call', 'key']


def test_key_dates_skip_what_is_not_a_date():
    """Una errata en el fichero deja esa marca fuera; la página no falla por ella."""
    params = {'dates': {'call': '5 de octubre', 'campaign': 20261113, 'key': ['2026-11-01']}}
    assert pages.key_dates(params) == []
    params = {'dates': {'call': '2026-10-05', 'key': {'2026-11-01': '', '2026-13-40': 'X', '2026-11-02': 'Debate'}}}
    assert [entry['date'] for entry in pages.key_dates(params)] == ['2026-10-05', '2026-11-02']
    # `dates` itself written as something else
    assert pages.key_dates({'dates': ['2026-10-05']}) == [] and pages.key_dates({'dates': '2026-10-05'}) == []
