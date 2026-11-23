"""Tests de las fechas clave de una elección (`dates` de `data/params.json`) como marcas de los gráficos
(`pages.key_dates`, y `pages.run_key_dates` para la elección de un run)."""
from mtpy.lib import pages
from tests.test_pages import api  # noqa: F401  (fixture de las páginas, con el `data/` real)


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


def test_run_key_dates_are_those_of_the_election_of_the_run(api):  # noqa: F811
    """Las fechas clave de un run son las de su elección en `data/params.json`, venga su fecha con hora o sin ella."""
    call_29n = {'date': '2026-10-05', 'label': 'Convocatoria', 'kind': 'call'}
    assert pages.run_key_dates('es', {'event_date': '2026-11-29'})[0] == call_29n
    assert pages.run_key_dates('es', {'event_date': '2026-11-29T00:00:00'})[0] == call_29n
    # an election without an entry in the file, a run without the date and a scope without elections: no marks
    assert pages.run_key_dates('es', {'event_date': '2027-08-22'}) == [] and pages.run_key_dates('es', {}) == []
    assert pages.run_key_dates('es', {'event_date': None}) == []
    assert pages.run_key_dates('es-zz', {'event_date': '2026-11-29'}) == []
