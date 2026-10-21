"""Agregadores y paneles online (`DROP_MTYPES`, 2026-10-21): fuera del promedio, del rating y de los listados."""
import inspect
import re

from mtpy.lib.mtypes import DROP_MTYPES


def test_every_default_drops_the_same_methodology_types():
    """Una sola lista: el promedio, el rating, la simulación y las elecciones destacadas excluyen lo mismo."""
    from mtpy.lib import data
    from mtpy.lib.computer import Computer
    from mtpy.lib.forecaster import Forecaster
    from mtpy.lib.simulator import Simulator

    assert DROP_MTYPES == ('aggr', 'online')
    for func in (Computer.__init__, Forecaster.__init__, Simulator.__init__, data.update_featured):
        assert list(inspect.signature(func).parameters['drop_mtypes'].default) == list(DROP_MTYPES), func


def test_save_ratings_leaves_the_excluded_pollsters_without_a_rating_in_the_catalogue(monkeypatch):
    """`pollsters.rating`: el último rating de cada casa; vacío (no 0) para agregadores y paneles online."""
    import pandas as pd
    from mtpy.lib import data

    saved = {}

    class Ratings:
        table = 'elections.pollsters_ratings'

        def format_data(self, df, **kwargs):
            return df

        def execute(self, query):
            pass

        def upsert(self, df):
            return len(df)

    class Pollsters:
        def format_data(self, df, **kwargs):
            return df

        def upsert(self, df):
            saved['pollsters'] = df.copy()

    monkeypatch.setattr(data, 'PollstersRatings', Ratings)
    monkeypatch.setattr(data, 'Pollsters', Pollsters)
    monkeypatch.setattr(data, 'get_pollsters', lambda: pd.DataFrame({
        'id': [1, 2, 3], 'name': ['A', 'B', 'C'], 'mtype': [None, 'online', None]}))
    ratings = pd.DataFrame({
        'event_scope': 'es', 'event_date': pd.to_datetime(['2026-11-29'] * 3), 'pollster_id': [1, 2, 3],
        'rating': [61.4, float('nan'), float('nan')]})
    data.save_ratings_data(ratings.set_index(['event_scope', 'event_date', 'pollster_id']), update_pollsters=True)
    rating = saved['pollsters'].set_index('id')['rating']
    assert rating[1] == 61 and rating[3] == 0  # C, sin evidencia ni fila con rating: 0, como hasta ahora
    assert pd.isna(rating[2])


def test_get_ratings_carries_the_type_of_each_pollster(monkeypatch):
    import pandas as pd
    from mtpy.lib import data

    class Pollsters:
        def get_results(self, query=None, formatted=False):
            return pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'mtype': [None, 'online']})

    class Ratings:
        def get_results(self, query=None, formatted=False):
            return pd.DataFrame({'event_scope': 'es', 'event_date': pd.to_datetime(['2026-11-29'] * 2),
                                 'pollster_id': [1, 2], 'rating': [61.4, float('nan')]})

    monkeypatch.setattr(data, 'Pollsters', Pollsters)
    monkeypatch.setattr(data, 'PollstersRatings', Ratings)
    monkeypatch.setattr(data, 'get_next_event_date', lambda scope, date_from=None: None)
    ratings = data.get_ratings('es', ['2026-11-29'])
    assert ratings.set_index('pollster')['mtype'].tolist() == [None, 'online']


def test_the_ratings_of_the_web_leave_out_the_excluded_pollsters(fresh_app, monkeypatch):
    """`query_ratings` (ranking de /casas y CSV de ratings): sin agregadores ni paneles online."""
    import pandas as pd
    from mtpy.lib import webapi
    from tests.test_webapi_unit import use_database

    use_database(fresh_app)
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2026-11-29'] * 3), 'pollster_id': [2, 28, 8],
        'pollster': ['GAD3', 'ElectoPanel', 'KeyData'], 'mtype': [None, 'online', 'aggr'],
        'event_scope': 'es', 'rating': [61.4, float('nan'), float('nan')], 'num_polls': [12, 40, 9]}))
    assert webapi.query_ratings('es')['pollster'].tolist() == ['GAD3']


def test_the_polls_of_the_web_leave_out_the_excluded_pollsters(fresh_app, monkeypatch):
    """`/api/v1/polls` pide la serie sin los tipos excluidos, como el promedio."""
    import pandas as pd
    from mtpy.lib import webapi
    from tests.test_webapi_unit import use_database

    use_database(fresh_app)
    seen = {}

    def fake_series(scope, event_dates, drop_mtypes=None, drop_contexts=None):
        seen.update(drop_mtypes=drop_mtypes, drop_contexts=drop_contexts)
        return pd.DataFrame({'date': pd.to_datetime(['2026-10-01']), 'pollster_id': [2], 'PP': [33.0]})

    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: '2026-11-29')
    monkeypatch.setattr(webapi.queries, 'get_poll_series', fake_series)
    webapi.query_polls('es')
    assert list(seen['drop_mtypes']) == list(DROP_MTYPES) and list(seen['drop_contexts']) == ['wban', 'exit']


# --- El perfil público de una casa excluida: sin rating, con sus errores por elección ---

from tests.test_pages import api, call, houses, houses_without_analysis, html, initial  # noqa: E402,F401


def cero_events(monkeypatch):
    """Cero (panel online del fixture de casas) sondeó las generales de 2019 y 2023: sus medias, sin rating."""
    import pandas as pd
    from mtpy.lib import webapi

    events = pd.DataFrame({
        'event_scope': ['es', 'es'], 'event_date': pd.to_datetime(['2019-11-10', '2023-07-23']), 'pollster_id': [5, 5],
        'n_polls': [9, 30], 'error_avg': [2.4, 1.6], 'error_blocks': [1.1, -0.7], 'bias_dev_adj': [0.8, -0.3]})
    ratings = pd.DataFrame({
        'event_scope': ['es', 'es'], 'event_date': pd.to_datetime(['2023-07-23', '2026-11-29']), 'pollster_id': [5, 5],
        'rating': [float('nan')] * 2, 'rating_adj': [float('nan')] * 2})
    monkeypatch.setattr(webapi.queries, 'get_pollster_events', lambda pollster_id: events)
    monkeypatch.setattr(webapi.queries, 'get_pollster_ratings', lambda pollster_id: ratings)
    webapi.queries._QUERIES.clear()


def test_the_profile_of_an_excluded_house_has_no_rating_kpi(houses, monkeypatch):
    from mtpy.lib import pages

    cero_events(monkeypatch)
    context = pages.casa_context(pages.parse_state({}), 5)
    assert [kpi['label'] for kpi in context['kpis']] == ['Confianza', 'Tipo']
    gad3 = pages.casa_context(pages.parse_state({}), 2)
    assert [kpi['label'] for kpi in gad3['kpis']] == ['Rating', 'Confianza', 'Tipo']


def test_the_profile_of_an_excluded_house_lists_its_errors_without_a_rating_path(houses, monkeypatch):
    """Sin gráfico del rating ni columnas de rating y cambio: la tabla de sus errores en cada elección sondeada."""
    cero_events(monkeypatch)
    status, _, body = call(houses, '/casas/cero')
    page = html(body)
    card = page[page.index('id="path"'):page.index('id="bullseye"')]
    assert status == 200 and 'entra así en el promedio' not in page
    assert 'Errores por elección' in card and 'Camino al rating' not in card
    assert 'id="rating-path"' not in card and 'id="path-table"' in card
    headers = re.findall(r'<th scope="col"[^>]*>([^<]*)', card)
    assert 'Rating' not in headers and 'Cambio' not in headers and 'Error medio' in headers
    tooltips = ' '.join(re.findall(r'<th scope="col"[^>]*title="([^"]*)"', card))
    assert 'rating' not in tooltips and 'entran en el modelo' not in tooltips  # sus textos no hablan de un rating
    rows = re.findall(r'<tr>\s*<td>([^<]+)</td>\s*<td>(\d+)</td>', card)
    assert [count for _, count in rows] == ['30', '9']  # de la más nueva a la más antigua
    assert initial(body)['path'] == []


def test_the_profile_of_a_rated_house_keeps_its_rating_path(houses):
    page = html(call(houses, '/casas/gad3')[2])
    card = page[page.index('id="path"'):page.index('id="bullseye"')]
    assert 'Camino al rating' in card and 'id="rating-path"' in card and 'Errores por elección' not in card
