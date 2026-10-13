"""Tests de las páginas servidas por los controladores (`Page`, `Index`, `Promedios`, `Escanos`, `Casas`...) con el
arnés ASGI."""
import json
import os
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from mtpy import mtpy
from mtpy.core.api import HttpError
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, pages, publish, webapi
from tests.test_api_asgi import call
from tests.test_webapi_unit import (  # noqa: F401
    ROOT, RUN, clear_queries, fixture_data, use_database, write_analysis, write_backtest, write_bundle, write_event
)


def site_api(app):
    """API completa (páginas + /api/v1 + not_found) registrada en `app`."""
    return mtpy.api(routes=pages.ROUTES + webapi.ROUTES, not_found=pages.NOT_FOUND)


@pytest.fixture
def api(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return site_api(fresh_app)


def html(body):
    return body.decode('utf-8')


def initial(body):
    match = re.search(r'<script type="application/json" id="initial-data">(.*?)</script>', html(body), re.S)
    assert match, 'sin bloque initial-data'
    return json.loads(match.group(1))


def test_promedios_is_rendered_html_with_cache_headers(api):
    status, headers, body = call(api, '/es', query='mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=60' and headers['etag'].startswith('"')
    assert '<title>Pronóstico electoral · Promedios</title>' in page
    assert 'Pronóstico para el' in page and 'href="/es?mode=forecast" aria-current="page"' in page
    assert 'name="mode" value="forecast" checked' in page.replace('\n', ' ')
    assert 'id="headline-table"' in page and '<th scope="col">Partido</th>' in page
    assert fixture_data('headline')['forecast']['parties'][0]['name'] in page
    assert '/dist/css/site.css' in page and '/dist/vendor/echarts-5.6.0.min.js' in page and '/dist/js/pages/promedios.js' in page
    assert 'onclick=' not in page and '<script>' not in page
    assert page.count('<script') == 3  # ECharts (defer), el módulo de la página y el bloque JSON


def test_promedios_initial_data_has_the_page_parts_with_one_run(api):
    data = initial(call(api, '/es')[2])
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs'}
    assert data['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    assert list(data['state']) == ['scope', 'mode', 'run', 'pinned']
    assert data['meta']['run_id'] == RUN and data['headline']['run_id'] == RUN
    assert data['runs']['runs'][0]['run_id'] == RUN


def test_nowcast_and_pinned_run_are_reflected(api):
    status, headers, body = call(api, '/es', query='mode=nowcast&run=' + RUN)
    page = html(body)
    assert status == 200 and 'Estimación a' in page
    assert 'name="mode" value="nowcast" checked' in page.replace('\n', ' ')
    assert 'type="hidden" name="run" value="{}"'.format(RUN) in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': True}
    assert 'href="/es/escanos?mode=nowcast&amp;run={}"'.format(RUN) in page


def test_freeze_banner_and_attribution(api, fresh_app):
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda electoral'})
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/es')[2])
    assert 'id="freeze-banner"' in page and 'Veda electoral' in page
    assert 'Sondeos: Wikipedia' in page and 'href="/api/v1/manifest"' in page


def test_frozen_manifest_written_before_phase_6_still_renders(api, fresh_app, tmp_path):
    """Un manifest congelado por un job anterior a la fase 6 (`freeze` sin `since`) no rompe las páginas: 200, banner
    y sin "congelado desde"."""
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda'})
    path = tmp_path / bundle.PREFIX / bundle.path_manifest()
    envelope = json.loads(path.read_text())
    envelope['data']['freeze'] = {'active': True, 'message': 'Veda'}
    path.write_text(json.dumps(envelope))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es')
    page = html(body)
    assert status == 200 and 'id="freeze-banner"' in page and 'Veda' in page
    assert 'congelado desde' not in page


@pytest.mark.parametrize('path, query, status, text', [
    ('/', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
    ('/', 'scope=es-xx', 400, 'Ámbito no válido'),
    ('/es', 'run=2026-10-08', 400, 'Run no válido'),
    ('/es', 'run=20200101-000000', 404, 'Ese run no existe'),
    ('/es', 'mode=tomorrow', 400, 'Modo no válido'),
    ('/elecciones', 'scope=es-xx', 400, 'Ámbito no válido'),
    ('/nope', '', 404, 'Ámbito desconocido'),  # un segmento suelto casa con `/{scope}`
    ('/nope/deeper', '', 404, 'Página no encontrada'),
    ('/promedio', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
    ('/escanos', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
    ('/casas', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
])
def test_page_errors_are_html_and_not_cached(api, fresh_app, monkeypatch, path, query, status, text):
    use_database(fresh_app)
    fake_houses(monkeypatch)
    got, headers, body = call(api, path, query=query)
    assert got == status and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'no-store' and 'etag' not in headers
    assert text in html(body) and 'Pronóstico electoral' in html(body)


def test_api_not_found_stays_json(api):
    status, headers, body = call(api, '/api/v1/nope')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': '404 Not Found'}


def test_without_bundle_the_home_is_a_503_page(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    status, headers, body = call(site_api(fresh_app), '/')
    assert status == 503 and 'Todavía no hay ningún pronóstico publicado' in html(body)
    assert headers['cache-control'] == 'no-store'


def test_default_scope_is_the_first_published(api, fresh_app):
    """Con `es` publicado, `?scope=` ausente lleva a `es`; un ámbito publicado adicional se lista en el filtro."""
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/elecciones')[2])
    assert '<option value="es" selected>' in page and '<option value="es-md">Madrid</option>' in page


def test_event_pages_live_under_the_scope_and_default_to_nowcast(api):
    status, headers, body = call(api, '/es')
    page = html(body)
    assert status == 200 and headers['cache-control'] == 'public, max-age=60'
    assert '<title>Pronóstico electoral · Promedios</title>' in page
    assert 'Estimación a' in page and 'id="event-bar"' in page and 'Elecciones generales' in page
    assert 'name="mode" value="nowcast" checked' in page.replace('\n', ' ')
    assert 'href="/es/escanos?mode=nowcast"' in page and 'id="controls"' not in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    status, _, body = call(api, '/es/escanos', query='mode=forecast&run=' + RUN)
    assert status == 200 and '· pronóstico para el' in html(body) and 'name="run" value="' + RUN + '"' in html(body)


def test_frozen_site_defaults_to_the_forecast_and_an_older_pinned_run_is_not_definitive(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    write_bundle(fresh_app.fs, run_id='20261001-120000', manifest=False)
    publish.update_manifest(writer, {'es': publish.latest_entry(fixture_data('headline') | {'run_id': RUN})},
                            freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    assert 'Pronóstico definitivo para el' in html(call(api, '/es')[2])
    page = html(call(api, '/es', query='run=20261001-120000&mode=forecast')[2])
    assert 'Pronóstico publicado el' in page and 'definitivo' not in page


@pytest.mark.parametrize('path, query, location', [
    ('/', 'scope=es&mode=nowcast', '/es?mode=nowcast'),
    ('/promedio', 'scope=es&mode=forecast&run=' + RUN, '/es?mode=forecast&run=' + RUN),
    ('/promedio', '', '/es'),
    ('/escanos', 'scope=es&region=28', '/es/escanos?region=28'),
    ('/modelo', 'scope=es', '/metodo/backtest?scope=es'),
    ('/modelo', '', '/metodo/backtest'),
])
def test_old_urls_redirect_permanently(api, path, query, location):
    status, headers, body = call(api, path, query=query)
    assert status == 301 and headers['location'] == location and headers['cache-control'] == 'no-store'
    assert body == b''


@pytest.mark.parametrize('path', ['/es-zz', '/ES', '/es/', '/es//escanos', '/healthz', '/metodo/nope'])
def test_unknown_scopes_and_sections_are_html_404s(api, path):
    status, headers, body = call(api, path)
    assert status == 404 and headers['content-type'].startswith('text/html')


def test_unpublished_catalogue_scope_is_a_404(api):
    status, _, body = call(api, '/es-md')
    assert status == 404 and 'no tiene pronóstico publicado' in html(body)


def test_global_pages_carry_the_scope_filter_and_no_event_bar(api):
    for path in ('/elecciones', '/metodo/backtest'):
        page = html(call(api, path)[2])
        assert 'id="scope-filter"' in page and 'id="event-bar"' not in page and 'name="mode"' not in page, path
    landing = html(call(api, '/metodo')[2])
    assert 'id="scope-filter"' not in landing and 'id="event-bar"' not in landing
    assert '<option value="es" selected>' in html(call(api, '/elecciones', query='scope=es')[2])


def test_menu_has_the_four_global_entries(api):
    page = html(call(api, '/es')[2])
    for label in ('Inicio', 'Histórico de errores', 'Rating de encuestadoras', 'Método'):
        assert '>' + label + '</a>' in page
    # ninguna entrada del menú es la actual: la pestaña activa de la barra de evento sí lleva `aria-current`
    menu = page[page.index('<nav aria-label="Secciones">'):]
    menu = menu[:menu.index('</nav>')]
    assert 'Portada' not in page and 'aria-current="page"' not in menu


def test_event_bar_days_text():
    from datetime import date
    assert pages.days_left('2026-11-29', today=date(2026, 10, 13)) == 47
    meta = {'event_date': '2026-11-29', 'as_of': '2026-10-13', 'run_id': RUN, 'run_at': '2026-10-13T12:00:00Z'}
    state = {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    scopes = [{'code': 'es', 'name': 'España'}, {'code': 'es-md', 'name': 'Madrid'}]
    info = pages.event_info('es', meta, '', state, today=date(2026, 10, 13), scopes=scopes)
    assert info['title'] == 'Elecciones generales' and info['days_text'] == 'Faltan 47 días'
    assert info['tabs'] == [{'href': '/es?mode=nowcast', 'label': 'Promedios', 'active': True},
                            {'href': '/es/escanos?mode=nowcast', 'label': 'Escaños', 'active': False}]
    assert pages.event_info('es', meta, '', state, today=date(2026, 11, 29), scopes=scopes)['days_text'] == 'Jornada electoral'
    assert pages.event_info('es', meta, '', state, today=date(2026, 12, 2), scopes=scopes)['days_text'] == 'Celebradas hace 3 días'
    assert pages.event_title('es-md', scopes) == 'Elecciones autonómicas · Madrid'


def test_event_info_counts_one_day_and_unknown_dates_and_points_the_form_at_the_tab():
    from datetime import date
    state = {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': True}
    scopes = [{'code': 'es', 'name': 'España'}]
    info = pages.event_info('es', {'event_date': '2026-11-29'}, '/escanos', state, today=date(2026, 11, 28),
                            scopes=scopes)
    assert info['days_text'] == 'Falta 1 día' and info['form_action'] == '/es/escanos' and info['href'] == '/es'
    assert [tab['active'] for tab in info['tabs']] == [False, True]
    assert info['tabs'][1]['href'] == '/es/escanos?mode=forecast&run=' + RUN
    unknown = pages.event_info('es', {}, '', state, today=date(2026, 11, 28), scopes=scopes)
    assert unknown['days_left'] is None and unknown['days_text'] is None
    assert pages.event_query({'mode': 'nowcast', 'run': RUN, 'pinned': False}) == '?mode=nowcast'
    assert pages.scope_query('es-md') == '?scope=es-md'
    assert pages.default_mode(True) == 'forecast' and pages.default_mode(False) == 'nowcast'


def test_parse_state_takes_the_scope_of_the_route_and_leaves_the_mode_unresolved(api):
    assert pages.parse_state({}) == {'scope': None, 'mode': None, 'run': None}
    route = pages.parse_state({'scope': 'es-xx', 'mode': 'forecast'}, 'es')
    assert route == {'scope': 'es', 'mode': 'forecast', 'run': None}
    with pytest.raises(HttpError) as invalid:
        pages.parse_state({'scope': 'es-xx'})
    assert invalid.value.status == 400


def test_title_of_rules():
    meta = {'as_of': '2026-10-13', 'event_date': '2026-11-29', 'run_at': '2026-10-13T12:00:00Z'}
    assert pages.title_of('nowcast', meta, True, True, True) == 'Estimación a 13 oct 2026'
    assert pages.title_of('forecast', meta, True, False, True) == 'Pronóstico definitivo para el 29 nov 2026'
    assert pages.title_of('forecast', meta, True, True, False) == 'Pronóstico publicado el 13 oct 2026'
    assert pages.title_of('forecast', meta, False, True, False) == 'Pronóstico publicado el 13 oct 2026'
    assert pages.title_of('forecast', meta, False, False, True) == 'Pronóstico para el 29 nov 2026'


def test_mode_text_explains_the_active_option(api):
    page = re.sub(r'\s+', ' ', html(call(api, '/es')[2]))
    assert ('Si las elecciones fueran hoy: el promedio de sondeos leído el 8 oct 2026, con los sondeos publicados hasta '
            'el 5 oct 2026.') in page
    page = re.sub(r'\s+', ' ', html(call(api, '/es', query='mode=forecast')[2]))
    assert 'Lo que puede pasar el 29 nov 2026: añade la deriva de la opinión' in page
    assert '¿Qué cambia entre Hoy y Elección?' in page


def test_method_subpages_and_links():
    assert list(pages.METODO_SECTIONS) == ['promedio', 'escanos', 'rating', 'backtest']
    subpages = pages.subpages('backtest')
    assert [page['id'] for page in subpages] == ['promedio', 'escanos', 'rating', 'backtest']
    assert subpages[3] == {'id': 'backtest', 'title': '¿Cuánto acierta el modelo?', 'href': '/metodo/backtest',
                           'active': True, 'available': True}
    assert not subpages[0]['active'] and subpages[0]['available'] and not subpages[1]['available']
    assert pages.method_href('backtest') == '/metodo/backtest' and pages.method_href('escanos') == '/metodo'
    assert pages.method_href('promedio') == '/metodo/promedio'


def test_templates_compile_with_strict_undefined():
    env = pages.environment()
    names = sorted(f for f in os.listdir(pages.TEMPLATES_DIR) if f.endswith('.html'))
    assert {'base.html', '_header.html', '_footer.html', '_event_bar.html', '_scope_filter.html', 'error.html',
            'promedios.html', 'escanos.html', 'casas.html', 'casa.html',
            'metodo_backtest.html'} <= set(names)
    for name in names:
        env.get_template(name)


def test_filters_follow_the_js_rules():
    assert pages.fmt_pct(32.66) == '32,7 %' and pages.fmt_pct(None) == '–'
    assert pages.fmt_num(1234567) == '1.234.567' and pages.fmt_num(3.14159, 1) == '3,1'
    assert pages.fmt_num(-1234.5, 1) == '-1.234,5' and pages.fmt_num(-1.25, 1) == '-1,3'
    assert pages.fmt_prob(0.995) == '> 99 %' and pages.fmt_prob(0.003) == '< 1 %' and pages.fmt_prob(0) == '0 %'
    assert pages.fmt_prob(1) == '100 %' and pages.fmt_prob(0.968) == '97 %' and pages.fmt_prob(None) == '–'
    assert pages.fmt_range(27.31, 38.02) == '27,3–38,0' and pages.fmt_range(118.0, 159.5, 0) == '118–160'
    assert pages.fmt_date('2026-10-13') == '13 oct 2026' and pages.fmt_date('2026-09-05') == '5 sept 2026'
    assert pages.fmt_date('garbage') == '–' and pages.fmt_date(None) == '–'
    assert pages.fmt_datetime('2026-10-08T18:11:02Z') == '8 oct 2026, 20:11'
    assert pages.api_url('es', 'polls', run=RUN, fmt='csv') == '/api/v1/forecast/es/polls?run={}&format=csv'.format(RUN)
    assert pages.api_url('es', 'vote', mode='forecast') == '/api/v1/forecast/es/forecast/vote'


def test_promedios_has_the_series_chart_the_polls_table_and_csv_link(api):
    body = call(api, '/es', query='mode=forecast')[2]
    page = html(body)
    assert 'id="series-chart"' in page and 'Sondeos del ciclo' in page
    assert 'href="/api/v1/forecast/es/polls?run={}&amp;format=csv"'.format(RUN) in page
    assert '<th scope="col">Fecha</th>' in page
    data = initial(body)
    assert data['projection']['dates'] and data['vote']['when']


def test_table_rows_are_newest_first_and_formatted():
    polls = [
        {'date': '2026-09-05', 'pollster': 'CIS', 'sponsor': None, 'sample_size': 4000, 'PP': 41.0, 'PSOE': 29.0},
        {'date': '2026-10-01', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': None, 'PP': None, 'PSOE': 30.5},
        {'date': '2026-10-01', 'pollster': 'CIS', 'sponsor': None, 'sample_size': 4000, 'PP': 40.5, 'PSOE': 29.5},
    ]
    rows = pages.table_rows(polls, ['PP', 'PSOE'], n=2)
    assert rows == [['1 oct 2026', 'CIS', '–', '4.000', '40,5', '29,5'], ['1 oct 2026', 'GAD3', 'ABC', '–', '–', '30,5']]
    assert len(pages.table_rows(polls, ['PP'])) == 3


def test_unexpected_error_is_a_generic_500_page_and_is_logged(api, fresh_app, monkeypatch):
    """Una excepción inesperada da un 500 HTML genérico, sin traza ni detalle, y se registra una vez."""
    def boom(state, **kwargs):
        raise RuntimeError('boom')

    errors = []
    monkeypatch.setattr(pages, 'promedios_context', boom)
    fresh_app.logger = SimpleNamespace(error=errors.append)
    status, headers, body = call(api, '/es')
    page = html(body)
    assert status == 500 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'no-store' and 'etag' not in headers
    assert 'Error interno' in page and 'Traceback' not in page and 'boom' not in page
    assert len(errors) == 1 and 'boom' in errors[0] and 'Traceback' in errors[0]


HOSTILE = '</script><script>alert(1)</script>"\'<b>&'


def test_bundle_strings_are_escaped_and_the_json_island_stays_safe(api, fresh_app, monkeypatch):
    """Textos hostiles del paquete y de la base se escapan en el HTML y en el bloque JSON; un color no válido cae al gris."""
    use_database(fresh_app)
    fake_houses(monkeypatch, name=HOSTILE)
    writer = bundle.BundleWriter(fresh_app.fs)
    meta = fixture_data('meta')
    meta['parties'][0].update(fullname=HOSTILE, color='red;background:url(http://x)')
    writer.write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    polls = fixture_data('polls')
    polls['polls'][0]['pollster'] = '<b>Hostil</b>'
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', polls, 'es', run_id=RUN)
    publish.update_manifest(writer, freeze={'active': True, 'message': HOSTILE})
    webapi.site().manifest_cache_clear()
    block = re.compile(r'<script type="application/json" id="initial-data">.*?</script>', re.S)
    for path in ('/es', '/es/escanos', '/casas', '/casas?id=1'):
        path, _, query = path.partition('?')
        body = call(api, path, query=query)[2]
        page = html(body)
        rest = block.sub('', page)
        assert '<script>alert' not in page and '&lt;script&gt;alert' in rest
        assert '<b>Hostil' not in page and 'url(' not in rest
        data = initial(body)
        assert data['meta']['parties'][0]['fullname'] == HOSTILE
    casas = initial(call(api, '/casas')[2])
    assert HOSTILE in [row['pollster'] for row in casas['ranking']]
    home = html(call(api, '/es')[2])
    assert 'background-color: {}'.format(pages.OTHERS_COLOR) in home
    assert '&lt;b&gt;Hostil&lt;/b&gt;' in home


def test_promedios_pinned_run_keeps_the_run_in_the_csv_link(api):
    page = html(call(api, '/es', query='run=' + RUN)[2])
    assert 'run={}&amp;format=csv'.format(RUN) in page
    assert '<input type="hidden" name="run" value="{}">'.format(RUN) in page


def test_promedios_initial_polls_carry_only_the_chart_columns(api):
    data = initial(call(api, '/es')[2])
    parties = set(data['polls']['parties'])
    assert set(data['polls']) == {'parties', 'polls'}
    for poll in data['polls']['polls']:
        assert set(poll) == {'date', 'pollster_id', 'pollster'} | parties


def test_without_a_published_catalogue_scope_the_home_is_a_503_page(fresh_app, tmp_path):
    """Un manifest cuyo único ámbito no está en el catálogo no se puede pedir por URL: 503, no un `<select>` vacío."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs, scope='zz')
    status, headers, body = call(site_api(fresh_app), '/')
    assert status == 503 and 'Todavía no hay ningún pronóstico publicado' in html(body)


def test_table_rows_tolerate_a_poll_without_date():
    polls = [{'date': '2026-10-01', 'pollster': 'CIS', 'PP': 40.0}, {'pollster': 'GAD3', 'PP': 41.0}]
    rows = pages.table_rows(polls, ['PP'])
    assert rows[0][0] == '1 oct 2026' and rows[1][0] == '–' and rows[1][1] == 'GAD3'


def test_promedios_title_mentions_the_projection_only_in_forecast(api):
    assert 'Promedio de sondeos y proyección' in html(call(api, '/es', query='mode=forecast')[2])
    page = html(call(api, '/es', query='mode=nowcast')[2])
    assert 'Promedio de sondeos' in page and 'y proyección' not in page
    assert 'y la proyección' not in page  # aria-label de los gráficos
    assert 'banda de incertidumbre y la proyección"' in html(call(api, '/es', query='mode=forecast')[2])


def test_promedio_caption_counts_the_polls(api, fresh_app):
    """The fixture has 1 poll of 120 used; with n_polls == shown polls the caption changes."""
    assert 'Últimos 1 de 120 sondeos' in html(call(api, '/es')[2])
    meta = fixture_data('meta') | {'n_polls': 1}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().files.clear()  # the first request cached the run files
    assert 'Todos los sondeos del ciclo (1)' in html(call(api, '/es')[2])


def test_escanos_is_rendered_with_tables_links_and_initial(api):
    status, headers, body = call(api, '/es/escanos', query='mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Pronóstico electoral · Escaños</title>' in page
    assert 'href="/es/escanos?mode=forecast" aria-current="page"' in page.replace('\n', ' ')
    assert 'id="parties-table"' in page and '<th scope="col">P(mayoría)</th>' in page
    assert 'id="blocks-table"' in page and 'id="vs-table"' in page and 'id="blocks-chart"' in page
    assert 'name="coalition" value="PP" checked' in page and 'id="coalition-result"' in page
    assert 'id="fan-chart"' in page and 'id="seats-evolution"' in page
    assert 'href="/api/v1/forecast/es/forecast/summary?run={}&amp;format=csv"'.format(RUN) in page
    assert 'href="/api/v1/forecast/es/fan?run={}&amp;format=csv"'.format(RUN) in page
    assert page.count('format=csv') == 5 and 'id="districts"' in page
    assert '/dist/js/pages/escanos.js' in page
    data = initial(body)
    assert list(data) == ['state', 'meta', 'summary', 'dist', 'fan', 'runs']
    assert data['state'] == {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    assert data['dist']['parties'] == ['PP'] and data['summary']['majority'] == 176


def test_escanos_explains_the_fan_and_the_evolution_and_closes_with_the_method(api):
    page = html(call(api, '/es/escanos')[2])
    assert 'bloque con más probabilidad de mayoría absoluta' in page
    fan = page[page.index('id="fan"'):page.index('id="evolution"')]
    assert 'deriva' in fan and 'horizonte 0' in fan and 'no es una predicción de trayectoria' in fan
    evolution = page[page.index('id="evolution"'):page.index('id="downloads"')]
    assert 'run' in evolution and 'huecos' in evolution and 'no es una serie de sondeos' in evolution
    assert '¿Cómo simulamos los escaños?' in page and 'href="/elecciones?scope=es"' in page
    assert 'href="/casas?scope=es"' in page


def test_escanos_headline_text_names_the_likeliest_block():
    summary = {'majority': 176, 'n_seats': 350,
               'blocks': [{'name': 'Izquierda', 'p_majority': 0.08}, {'name': 'Derecha', 'p_majority': 0.999}]}
    assert pages.escanos_headline_text(summary, {}) == (
        'Derecha es el bloque con más probabilidad de mayoría absoluta (> 99 %); la mayoría está en 176 escaños '
        'de 350.')
    tie = {'majority': 176, 'n_seats': 350, 'blocks': [{'name': 'A', 'p_majority': 0.5}, {'name': 'B', 'p_majority': 0.5}]}
    assert pages.escanos_headline_text(tie, {}).startswith('A es el bloque')
    assert pages.escanos_headline_text({'majority': 176, 'n_seats': 350, 'blocks': []}, {}) == (
        'La mayoría absoluta está en 176 escaños de 350.')
    assert pages.escanos_headline_text({'blocks': []}, {'majority': 130, 'n_seats': 250}) == (
        'La mayoría absoluta está en 130 escaños de 250.')


def test_event_bar_form_keeps_the_selected_region(api, fresh_app):
    write_districts(fresh_app.fs)

    def bar_form(query):
        page = html(call(api, '/es/escanos', query=query)[2])
        start = page.index('id="event-controls"')
        return page[start:page.index('</form>', start)]

    assert '<input type="hidden" name="region" value="8">' in bar_form('mode=forecast&region=8')
    # the largest district is the default selection
    assert '<input type="hidden" name="region" value="28">' in bar_form('mode=forecast')
    # pages without a region keep the form as it was
    page = html(call(api, '/es')[2])
    assert 'name="region"' not in page[page.index('id="event-controls"'):page.index('</form>')]


def test_escanos_nowcast_and_pinned_run(api):
    page = html(call(api, '/es/escanos', query='mode=nowcast&run=' + RUN)[2])
    assert 'Estimación a' in page and 'type="hidden" name="run" value="{}"'.format(RUN) in page
    assert page.count('run={}&amp;format=csv'.format(RUN)) == 5


def test_seat_rows_sort_by_seats_desc_with_nulls_last():
    rows = [{'name': 'A', 'seats': 5}, {'name': 'B', 'seats': None}, {'name': 'C', 'seats': 9}, {'name': 'D', 'seats': 5}]
    assert [r['name'] for r in pages.seat_rows(rows)] == ['C', 'A', 'D', 'B']


def test_default_coalition_is_the_first_vs_block_present_in_dist():
    meta = {'bmaps': {'vs': {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE']}}}
    summary = {'vs': [{'name': 'Derecha'}, {'name': 'Izquierda'}], 'blocks': []}
    assert pages.default_coalition(meta, summary, ['PSOE', 'VOX', 'PP']) == ['VOX', 'PP']
    assert pages.default_coalition({'bmaps': {}}, {'vs': [], 'blocks': []}, ['PP']) == []


def test_csv_links_cover_the_five_parts_in_order():
    links = pages.csv_links('es', 'nowcast', RUN)
    assert [l['label'] for l in links] == ['Resumen por partido y bloque', 'Escaños por simulación', 'Circunscripciones', 'Escenario central', 'Abanico por horizonte']
    assert links[0]['href'] == '/api/v1/forecast/es/nowcast/summary?run={}&format=csv'.format(RUN)
    assert links[4]['href'] == '/api/v1/forecast/es/fan?run={}&format=csv'.format(RUN)


DISTRICTS = {
    'parties': ['PP', 'PSOE'],
    'regions': [{'id': 8, 'name': 'Barcelona', 'seats': 32}, {'id': 28, 'name': 'Madrid', 'seats': 37}],
    'rows': [
        {'region_id': 8, 'region': 'Barcelona', 'name': 'PP', 'pct': 20.0, 'pct_lo': 17.0, 'pct_hi': 23.0, 'seats': 7.0, 'seats_mean': 7.1, 'seats_lo': 6.0, 'seats_hi': 8.0, 'p_seats': 1.0},
        {'region_id': 8, 'region': 'Barcelona', 'name': 'PSOE', 'pct': 25.0, 'pct_lo': 22.0, 'pct_hi': 28.0, 'seats': 9.0, 'seats_mean': 9.2, 'seats_lo': 8.0, 'seats_hi': 10.0, 'p_seats': 1.0},
        {'region_id': 28, 'region': 'Madrid', 'name': 'PP', 'pct': 39.0, 'pct_lo': 34.0, 'pct_hi': 44.0, 'seats': 16.0, 'seats_mean': 15.8, 'seats_lo': 14.0, 'seats_hi': 18.0, 'p_seats': 1.0},
        {'region_id': 28, 'region': 'Madrid', 'name': 'PSOE', 'pct': 24.0, 'pct_lo': 20.0, 'pct_hi': 28.0, 'seats': 9.0, 'seats_mean': 9.1, 'seats_lo': 8.0, 'seats_hi': 11.0, 'p_seats': 1.0},
    ],
}
SCENARIO = {'simulation': 3, 'parties': ['PP', 'PSOE'], 'rows': [
    {'region_id': 0, 'region': 'es', 'seats': [140, 106]},
    {'region_id': 8, 'region': 'Barcelona', 'seats': [7, 9]},
    {'region_id': 28, 'region': 'Madrid', 'seats': [16, 9]},
]}


def write_districts(fs, districts=DISTRICTS, scenario=SCENARIO, scope='es', run_id=RUN):
    writer = bundle.BundleWriter(fs)
    for mode in bundle.MODES:
        writer.write_json(bundle.path_part(scope, run_id, 'districts', mode), 'districts', districts, scope, run_id=run_id, mode=mode)
        writer.write_json(bundle.path_part(scope, run_id, 'scenario', mode), 'scenario', scenario, scope, run_id=run_id, mode=mode)


def test_escanos_districts_default_to_the_largest_and_link_the_rows(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/es/escanos', query='mode=forecast')[2])
    assert 'id="districts"' in page and '<option value="28" selected>Madrid</option>' in page
    assert 'Madrid · 37 escaños' in page  # caption del detalle
    assert 'href="/es/escanos?mode=forecast&amp;region=8"' in page
    assert '<td title="6–8">7</td>' in page and '<td title="14–18">16</td>' in page
    assert '<td>Total</td>' in page and '140' in page
    page = html(call(api, '/es/escanos', query='mode=forecast&region=8')[2])
    assert '<option value="8" selected>Barcelona</option>' in page and 'Barcelona · 32 escaños' in page
    assert '<form id="district-form" method="get" action="/es/escanos#districts">' in page
    assert 'name="scope"' not in page  # el ámbito va en la ruta


def test_escanos_region_detail_skips_parties_that_do_not_stand(api, fresh_app):
    absent = {'region_id': 28, 'region': 'Madrid', 'name': 'VOX', 'pct': None, 'pct_lo': None, 'pct_hi': None,
              'seats': 0.0, 'seats_mean': 0.0, 'seats_lo': 0.0, 'seats_hi': 0.0, 'p_seats': 0.0}
    write_districts(fresh_app.fs, districts=DISTRICTS | {'rows': DISTRICTS['rows'] + [absent]})
    page = html(call(api, '/es/escanos', query='mode=forecast&region=28')[2])
    detail = page[page.index('id="region-table"'):]
    detail = detail[:detail.index('</table>')]
    assert '– %' not in detail and detail.count('<tr>') == 3  # cabecera, PP y PSOE
    assert '34,0–44,0 %' in detail


def test_escanos_district_form_keeps_the_pinned_run(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/es/escanos', query='mode=forecast&run=' + RUN)[2])
    assert page.count('type="hidden" name="run" value="{}"'.format(RUN)) == 2  # barra de evento y formulario de circunscripción
    assert 'href="/es/escanos?mode=forecast&amp;run={}&amp;region=8"'.format(RUN) in page


@pytest.mark.parametrize('query, status, text', [
    ('region=abc', 400, 'Circunscripción no válida'),
    ('region=99', 404, 'Circunscripción no encontrada'),
])
def test_escanos_region_errors(api, fresh_app, query, status, text):
    write_districts(fresh_app.fs)
    got, headers, body = call(api, '/es/escanos', query=query)
    assert got == status and headers['cache-control'] == 'no-store' and text in html(body)


def test_escanos_without_districts_hides_the_section(api, fresh_app):
    write_districts(fresh_app.fs, districts={'parties': ['PP'], 'regions': [], 'rows': []},
                    scenario={'simulation': 0, 'parties': ['PP'], 'rows': [{'region_id': 0, 'region': 'es-md', 'seats': [70]}]})
    status, headers, body = call(api, '/es/escanos')
    assert status == 200 and 'id="districts"' not in html(body)
    assert call(api, '/es/escanos', query='region=28')[0] == 404


def test_regional_scope_renders_every_page(api, fresh_app):
    """Ámbito autonómico sintético: `bmaps` sin `max`, sin circunscripciones."""
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    meta = fixture_data('meta') | {'scope': 'es-md', 'n_seats': 135, 'majority': 68}
    meta['bmaps'] = {k: v for k, v in meta['bmaps'].items() if k != 'max'}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es-md', '20261009-120000', 'meta'), 'meta', meta, 'es-md', run_id='20261009-120000')
    write_districts(fresh_app.fs, districts={'parties': ['PP'], 'regions': [], 'rows': []},
                    scenario={'simulation': 0, 'parties': ['PP'], 'rows': [{'region_id': 0, 'region': 'es-md', 'seats': [70]}]},
                    scope='es-md', run_id='20261009-120000')
    webapi.site().manifest_cache_clear()
    for path in ('/es-md', '/es-md/escanos'):
        status, headers, body = call(api, path)
        assert status == 200 and 'Elecciones autonómicas · Madrid' in html(body), path
    assert 'name="coalition" value="PP" checked' in html(call(api, '/es-md/escanos')[2])


def test_district_table_and_region_rows():
    table = pages.district_table(DISTRICTS, SCENARIO)
    assert table['parties'] == ['PP', 'PSOE'] and [r['name'] for r in table['rows']] == ['Barcelona', 'Madrid']
    assert table['rows'][1]['cells'] == [{'seats': 16, 'range': '14–18'}, {'seats': 9, 'range': '8–11'}]
    assert table['total'] == {'seats': 246, 'cells': [{'seats': 140, 'range': '–'}, {'seats': 106, 'range': '–'}]}
    assert pages.district_table({'parties': [], 'regions': [], 'rows': []}, SCENARIO) is None
    assert [r['name'] for r in pages.region_rows(DISTRICTS, 28)] == ['PP', 'PSOE']
    absent = DISTRICTS['rows'][2] | {'name': 'VOX', 'pct': None, 'pct_lo': None, 'pct_hi': None, 'seats': 0.0}
    assert [r['name'] for r in pages.region_rows(DISTRICTS | {'rows': DISTRICTS['rows'] + [absent]}, 28)] == ['PP', 'PSOE']
    assert pages.resolve_region(None, DISTRICTS['regions'])['id'] == 28


def test_evolution_cards_explain_the_retrospective_points(api):
    for path in ('/es', '/es/escanos'):
        page = html(call(api, path)[2])
        assert 'Evolución de las publicaciones' in page and 'estimaciones retrospectivas' in page


def test_promedios_orders_the_story_and_carries_the_controls(api):
    page = html(call(api, '/es')[2])
    order = [page.index(marker) for marker in ('id="series"', 'id="headline"', 'id="vote-chart"', 'id="hemicycle"',
                                              'id="evolution"', 'id="pollster"', 'id="polls"', 'id="closing"')]
    assert order == sorted(order)
    assert 'name="window" value="all" checked' in page.replace('\n', ' ') and 'value="3m"' in page and 'value="1a"' in page
    assert 'name="view" value="partidos"' in page and 'name="view" value="bloques"' in page
    assert 'id="pollster-select"' in page and 'href="/metodo' in page
    assert re.search(r'<div class="series-controls"[^>]*\bhidden[^>]*>\s*<label>Casa', page)
    data = initial(call(api, '/es')[2])
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs'}
    assert all(abs(v * 10 - round(v * 10)) < 1e-9 for v in data['series']['mean']['PP'] if v is not None)
    assert 'pollster_id' in data['polls']['polls'][0]


def test_promedios_controls_are_hidden_without_js_and_check_the_main_parties(api):
    """Sin JS los controles no hacen nada y no se muestran; las casillas marcan los partidos `main` del run."""
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    for control in ('series-window', 'series-view', 'series-parties', 'series-blocks'):
        assert re.search(r'<fieldset id="{}" class="[^"]*" hidden>'.format(control), page), control
    assert 'name="party" value="PP" checked' in page and 'name="block" value="Derecha" checked' in page
    assert 'data-all' in page and 'data-none' in page
    assert '<option value="Sigma Dos" selected>Sigma Dos (1)</option>' in page
    assert 'href="/elecciones?scope=es"' in page and 'href="/casas?scope=es"' in page
    assert '¿Cómo promediamos las encuestas?' in page and '/dist/js/pages/promedios.js' in page


def test_promedios_hides_the_blocks_view_without_blocks(api, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    meta = fixture_data('meta') | {'scope': 'es-md'}
    meta['bmaps'] = {k: v for k, v in meta['bmaps'].items() if k not in ('max', 'blocks')}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es-md', '20261009-120000', 'meta'), 'meta', meta,
                                                 'es-md', run_id='20261009-120000')
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/es-md')[2])
    assert 'name="view"' not in page and 'id="series-parties"' in page and 'id="series-blocks"' not in page


def test_promedios_headline_sentence_uses_the_mode(api):
    assert 'Si hoy hubiera elecciones' in html(call(api, '/es')[2])
    assert 'El 29 nov 2026' in html(call(api, '/es', query='mode=forecast')[2])


def test_headline_text_names_the_leader_and_the_likeliest_majority():
    current = {'parties': [{'name': 'PSOE', 'pct': 28.04, 'seats': 110}, {'name': 'PP', 'pct': 33.06, 'seats': 135}],
               'p_majority': {'Izquierda': 0.08, 'Derecha': 0.41}}
    meta = {'event_date': '2026-11-29'}
    names = pages.Catalog({'PP': {'fullname': 'Partido Popular', 'color': '#1d84ce', 'block': 'Derecha'}})
    assert pages.headline_text(current, 'nowcast', meta, names, definitive=False) == (
        'Si hoy hubiera elecciones, Partido Popular ganaría con el 33,1 % del voto y 135 escaños; probabilidad de '
        'mayoría absoluta de Derecha: 41 %.')
    assert pages.headline_text(current, 'forecast', meta, names, definitive=False).startswith(
        'El 29 nov 2026, Partido Popular ganaría con el 33,1 %')
    assert pages.headline_text(current, 'forecast', meta, names, definitive=True).startswith(
        'Pronóstico definitivo: el 29 nov 2026, Partido Popular ganaría')
    # sin probabilidades de mayoría la frase acaba en los escaños; sin partidos no hay titular
    assert pages.headline_text(current | {'p_majority': {}}, 'nowcast', meta, names, definitive=False).endswith(
        'y 135 escaños.')
    assert pages.headline_text({'parties': []}, 'nowcast', meta, names, definitive=False) is None


def test_promedios_frozen_latest_forecast_is_definitive(api, fresh_app):
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda'})
    webapi.site().manifest_cache_clear()
    assert 'Pronóstico definitivo: el 29 nov 2026' in html(call(api, '/es')[2])
    assert 'Pronóstico definitivo:' not in html(call(api, '/es', query='mode=nowcast')[2])


def test_pollster_options_and_round_series():
    polls = [{'pollster': 'GAD3'}, {'pollster': 'CIS'}, {'pollster': 'GAD3'}, {'pollster': 'Sigma Dos'},
             {'pollster': 'CIS'}, {'pollster': 'Arce'}, {'pollster': None}]
    assert pages.pollster_options(polls) == [{'name': 'CIS', 'n': 2}, {'name': 'GAD3', 'n': 2},
                                             {'name': 'Arce', 'n': 1}, {'name': 'Sigma Dos', 'n': 1}]
    assert pages.pollster_options([]) == []
    series = {'dates': ['2026-01-01', '2026-01-02'], 'parties': ['PP'], 'mean': {'PP': [33.149, None]},
              'lo': {'PP': [31.04, None]}, 'hi': {'PP': [35.26, 36.0]}}
    out = pages.round_series(series)
    assert out == {'dates': series['dates'], 'parties': ['PP'], 'mean': {'PP': [33.1, None]},
                   'lo': {'PP': [31.0, None]}, 'hi': {'PP': [35.3, 36.0]}}
    assert series['mean']['PP'][0] == 33.149  # no muta la parte


# Casas: tres casas con ratings en `es` (CIS id 1, GAD3 id 2 y una tercera con `rating` nulo que no entra
# en el ranking) en dos elecciones pasadas y la próxima, más una cuarta casa del catálogo sin ratings en el
# ámbito. GAD3 tiene el rating más alto en la próxima elección (2026-11-29); su primer rating es de 2023.
HOUSE_EVENTS = ['2019-11-10', '2023-07-23', '2026-11-29']


def fake_houses(monkeypatch, name='CIS'):
    """Lectores de casas parcheados en `webapi`; `name` es el nombre de la casa 1 (CIS por defecto)."""
    pollsters = pd.DataFrame({
        'id': [1, 2, 3, 4], 'name': [name, 'GAD3', 'Hamalgama', 'Nueva'],
        'mtype': ['public', 'private', 'private', 'private'], 'quality': [60.0, 80.0, 40.0, 50.0],
        'rating': [55, 70, None, None],  # rating nacional truncado: distinto del camino del ámbito
    })
    ratings = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-11-10', '2023-07-23', '2023-07-23', '2026-11-29', '2026-11-29', '2026-11-29']),
        'pollster_id': [1, 1, 2, 1, 2, 3], 'pollster': [name, name, 'GAD3', name, 'GAD3', 'Hamalgama'],
        'quality': [60.0, 60.0, 80.0, 60.0, 80.0, 40.0], 'num_events': [1, 2, 1, 3, 2, 0],
        'num_polls': [10, 12, 14, 15, 17, 2], 'num_polls_w': [8.1, 9.5, 11.2, 12.0, 13.3, 1.1],
        'error_avg': [1.9, 1.8, 1.32, 1.85, 1.3, None], 'error_blocks': [4.2, 4.0, 3.05, 4.1, 3.0, None],
        'bias_blocks': [2.5, 2.4, 1.52, 2.45, 1.5, None], 'rating_adj': [50.0, 52.0, 80.0, 53.0, 83.1, None],
        'rating': [51.0, 53.0, 80.0, 55.0, 82.37, float('nan')],
    })
    house_effects = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2023-07-23', '2023-07-23', '2026-11-29']),
        'pollster_id': [2, 2, 1, 2], 'pollster': ['GAD3', 'GAD3', name, 'GAD3'], 'party': ['PP', 'PSOE', 'PP', 'PP'],
        'dev_result_c': [0.9, -0.4, -2.1, float('nan')], 'dev_cycle': [0.6, -0.2, -2.8, 0.7],
    })
    herding = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2026-11-29']), 'pollster_id': [2, 2], 'pollster': ['GAD3', 'GAD3'],
        'n': [14, 17], 'ratio': [0.781, 0.705], 'p_value': [0.034, 0.012],
    })
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: pollsters)
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings)
    monkeypatch.setattr(webapi.queries, 'get_house_effects', lambda scope: house_effects)
    monkeypatch.setattr(webapi.queries, 'get_herding', lambda scope: herding)
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: HOUSE_EVENTS[-1])


@pytest.fixture
def houses_without_analysis(api, fresh_app, monkeypatch):
    """Páginas con base simulada y los lectores de casas parcheados, sin análisis publicado."""
    use_database(fresh_app)
    fake_houses(monkeypatch)
    return api


@pytest.fixture
def houses(houses_without_analysis, fresh_app):
    """Como `houses_without_analysis`, con el análisis de casas de `es` publicado (fixtures)."""
    write_analysis(fresh_app.fs)
    return houses_without_analysis


def test_casas_lists_the_ranking_effects_and_herding(houses):
    status, _, body = call(houses, '/casas', query='scope=es')
    page = html(body)
    assert status == 200 and 'aria-current="page"' in page and 'id="ranking-table"' in page
    assert page.index('GAD3') < page.index('CIS')  # orden por rating desc en el fixture
    assert 'href="/casas?scope=es&amp;id=2"' in page and 'id="scope-filter"' in page
    assert 'id="effects-heatmap"' in page and 'id="herding-table"' in page
    assert 'Cada fila es una casa con rating para la próxima elección' in page
    assert 'href="/api/v1/analysis/es/herding?format=csv"' in page and 'href="/api/v1/ratings?scope=es&amp;format=csv"' in page
    data = initial(body)
    assert list(data) == ['state', 'meta', 'ranking', 'grid'] and data['ranking'][0]['pollster'] == 'GAD3'


def test_casas_explains_the_relative_bias(houses):
    page = html(call(houses, '/casas', query='scope=es')[2]).replace('\n', ' ')
    page = re.sub(r'\s+', ' ', page)
    assert 'comparado con el de las demás casas' in page and 'no penaliza a una casa' in page


def test_casas_without_analysis_keeps_the_ranking(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas')[2])
    assert 'id="ranking-table"' in page and 'Análisis no publicado todavía' in page
    assert 'id="effects-heatmap"' not in page and 'id="herding-table"' not in page


def test_casas_without_database_is_a_503_page(api):
    status, headers, body = call(api, '/casas')
    assert status == 503 and 'La base de datos no está disponible' in html(body) and headers['cache-control'] == 'no-store'


def test_casa_profile_shows_the_path_bullseye_and_tables(houses):
    page = html(call(houses, '/casas', query='scope=es&id=2')[2])
    assert '<h2>GAD3' in page and 'id="rating-path"' in page and 'id="bullseye-chart"' in page
    assert 'id="profile-effects"' in page and 'id="profile-herding"' in page and 'Todas las casas' in page
    data = initial(call(houses, '/casas', query='scope=es&id=2')[2])
    assert list(data) == ['state', 'meta', 'path', 'bullseye', 'blocks'] and data['path'][0]['event_date'] == '2023-07-23'


@pytest.mark.parametrize('query, status, text', [('id=abc', 400, 'Casa no válida'), ('id=999', 404, 'Casa no encontrada')])
def test_casa_profile_errors(houses, query, status, text):
    got, headers, body = call(houses, '/casas', query=query)
    assert got == status and text in html(body) and headers['cache-control'] == 'no-store'


def test_casas_ranking_excludes_null_ratings_and_links_every_house(houses):
    page = html(call(houses, '/casas')[2])
    ranking = page[page.index('id="ranking-table"'):]
    ranking = ranking[:ranking.index('</table>')]
    assert 'Hamalgama' not in ranking and ranking.count('<tr>') == 3  # cabecera, GAD3 y CIS
    assert '82,4' in ranking and '0,74' in ranking  # rating (escala 0-100) y herding (análisis) de GAD3
    assert 'id="rating-chart"' in page and 'id="effects-table"' in page
    assert 'href="/api/v1/analysis/es/herding?format=csv&amp;table=events"' in page
    assert 'href="/api/v1/analysis/es/house-effects?format=csv"' in page and 'href="/api/v1/analysis/es/poll-errors?format=csv"' in page
    data = initial(call(houses, '/casas')[2])
    assert [row['pollster_id'] for row in data['ranking']] == [2, 1]
    assert data['grid']['columns'] == ['PP', 'PSOE', 'VOX', 'Derecha', 'Izquierda']
    assert [row['pollster'] for row in data['grid']['rows']] == ['CIS']  # GAD3 no tiene `cycle`


def test_casas_without_analysis_has_no_analysis_downloads(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas')[2])
    assert '/api/v1/analysis/' not in page and 'href="/api/v1/ratings?scope=es&amp;format=csv"' in page
    assert initial(call(houses_without_analysis, '/casas')[2])['grid'] is None


def test_casa_profile_tables_and_bullseye(houses):
    status, _, body = call(houses, '/casas', query='id=2&mode=nowcast')
    page = html(body)
    assert status == 200 and 'href="/casas?scope=es"' in page  # "Todas las casas"
    assert 'id="scope-filter"' in page and 'action="/casas"' in page
    assert 'id="path-table"' in page and '2023-23J' in page
    effects = page[page.index('id="profile-effects"'):]
    effects = effects[:effects.index('</table>')]
    assert effects.count('<tr>') == 2 and '23 jul 2023' in effects  # la próxima elección no tiene resultado
    data = initial(body)
    assert [row['event_date'] for row in data['path']] == ['2023-07-23', '2026-11-29']
    assert list(data['path'][0]) == ['event_date', 'rating', 'rating_adj', 'error_blocks', 'bias_blocks', 'num_polls']
    assert data['bullseye'] == [{'event': '2023-23J', 'event_date': '2023-07-23', 'days': 7, 'Derecha': 2.81, 'Izquierda': -0.24}]
    assert data['blocks'] == ['Derecha', 'Izquierda']


def test_casa_profile_of_a_house_without_ratings(houses):
    """Una casa del catálogo sin ratings en el ámbito: perfil con tablas vacías, sin camino ni diana."""
    status, _, body = call(houses, '/casas', query='id=4')
    page = html(body)
    assert status == 200 and '<h2>Nueva' in page and 'Calidad: – · Rating: –' in page
    assert 'Nueva: sin rating en este ámbito.' in page
    assert 'id="rating-path"' not in page and 'id="bullseye-chart"' not in page
    assert 'id="path-table"' in page and 'id="profile-effects"' in page and 'id="profile-herding"' in page
    data = initial(body)
    assert data['path'] == [] and data['bullseye'] == []


def test_casa_profile_without_analysis_has_no_bullseye(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas', query='id=2')[2])
    assert 'id="rating-path"' in page and 'id="bullseye-chart"' not in page and 'Análisis no publicado todavía' in page
    assert initial(call(houses_without_analysis, '/casas', query='id=2')[2])['blocks'] == []


def test_ranking_rows_and_herding_rows():
    frame = pd.DataFrame({'pollster_id': [1, 2, 3], 'pollster': ['CIS', 'GAD3', 'X'], 'rating': [55.0, 82.37, float('nan')],
                          'quality': [60.0, 80.0, 40.0], 'num_polls': [15, 17, 2], 'extra': [1, 2, 3]})
    rows = pages.ranking_rows(frame, {'rows': [{'pollster': 'GAD3', 'herding': 0.742}]})
    assert [row['pollster'] for row in rows] == ['GAD3', 'CIS'] and rows[0]['herding'] == 0.742 and rows[1]['herding'] is None
    assert list(rows[0]) == ['pollster_id', 'pollster', 'rating', 'quality', 'num_polls', 'num_events', 'error_avg',
                             'error_blocks', 'bias_blocks', 'herding']
    assert rows[0]['num_events'] is None and pages.ranking_rows(pd.DataFrame(), None) == []
    part = {'rows': [{'pollster': 'A', 'herding': 0.9}, {'pollster': 'B', 'herding': None}, {'pollster': 'C', 'herding': 0.7}]}
    assert [row['pollster'] for row in pages.herding_rows(part)] == ['C', 'A', 'B']


def test_house_effects_grid_orders_by_the_first_block_with_nulls_last():
    def row(pollster, name, cycle):
        return {'pollster': pollster, 'name': name, 'n_events': 1, 'hist': 0.5, 'trend': None, 'cycle': cycle, 'n_cycle': 3}
    part = {'parties': ['PP'], 'blocks': ['Derecha'], 'rows': [
        row('A', 'PP', 1.0), row('A', 'Derecha', None), row('B', 'PP', None), row('B', 'Derecha', 2.0),
        row('C', 'PP', None), row('D', 'Derecha', -1.0),
    ]}
    grid = pages.house_effects_grid(part)
    assert grid['columns'] == ['PP', 'Derecha'] and [r['pollster'] for r in grid['rows']] == ['B', 'D', 'A']
    assert grid['rows'][1]['cells'] == [{'cycle': None, 'hist': None, 'trend': None}, {'cycle': -1.0, 'hist': 0.5, 'trend': None}]


def test_api_url_covers_analysis_and_data_tables():
    assert pages.api_url('es', 'herding', kind='analysis', fmt='csv') == '/api/v1/analysis/es/herding?format=csv'
    assert pages.api_url('es', 'herding', kind='analysis', fmt='csv', table='events') == '/api/v1/analysis/es/herding?format=csv&table=events'
    assert pages.api_url('es', 'ratings', kind='data', fmt='csv') == '/api/v1/ratings?scope=es&format=csv'
    assert pages.api_url('es', 'fan', run=RUN, fmt='csv') == '/api/v1/forecast/es/fan?run={}&format=csv'.format(RUN)


def test_analysis_part_is_none_when_not_published(api, fresh_app):
    assert pages.analysis_part('es', 'herding') is None
    write_analysis(fresh_app.fs)
    webapi.site().manifest_cache_clear()
    assert pages.analysis_part('es', 'herding')['rows'][0]['pollster'] == 'GAD3'


def test_casa_profile_header_uses_the_scope_rating(houses):
    """La cabecera toma rating y calidad de la última fila del camino del ámbito, no del catálogo (nacional, truncado)."""
    page = html(call(houses, '/casas', query='id=2')[2])
    assert 'Tipo: privada · Calidad: 80,0 · Rating: 82,4' in page and 'Rating: 70' not in page
    assert 'Tipo: pública' in html(call(houses, '/casas', query='id=1')[2])  # `mtype` con su etiqueta en castellano


def test_pages_package_layout():
    """Un módulo por página, `common` y `filters`; ninguno tapado por un nombre exportado (D1)."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(pages.__path__))
    assert names == ['backtest', 'casas', 'common', 'elecciones', 'escanos', 'filters', 'inicio', 'metodo', 'promedios']
    for name in names:
        assert inspect.ismodule(getattr(pages, name)), name
    assert pages.fmt_pct.__module__ == 'mtpy.lib.pages.filters'
    assert pages.common_context.__module__ == 'mtpy.lib.pages.common'
    assert pages.casas_context.__module__ == 'mtpy.lib.pages.casas'
    assert pages.TEMPLATES_DIR.endswith(os.path.join('web', 'templates')) and os.path.isdir(pages.TEMPLATES_DIR)


def test_api_url_covers_events_and_backtest():
    assert pages.api_url('es', 'pollsters', kind='event', date='2023-07-23', fmt='csv') == '/api/v1/events/es/2023-07-23/pollsters?format=csv'
    assert pages.api_url('es', 'model', kind='event', date='2023-07-23', fmt='csv', table='seats') == '/api/v1/events/es/2023-07-23/model?format=csv&table=seats'
    assert pages.api_url('es', 'metrics', kind='backtest', fmt='csv') == '/api/v1/backtest/es/metrics?format=csv'
    assert pages.api_url('es', 'events', kind='events') == '/api/v1/events/es'


@pytest.fixture
def events_site(api, fresh_app):
    """Páginas con dos elecciones publicadas: 2023 con backtest y 2019 sin él."""
    write_event(fresh_app.fs)
    write_event(fresh_app.fs, date='2019-11-10', model=False)
    return api


def test_elecciones_lists_the_published_events_newest_first(events_site):
    status, _, body = call(events_site, '/elecciones', query='scope=es')
    page = html(body)
    assert status == 200 and page.index('date=2023-07-23') < page.index('date=2019-11-10')
    assert '/elecciones?scope=es&amp;date=2023-07-23' in page and 'action="/elecciones"' in page
    data = initial(body)
    assert [row['date'] for row in data['sector']] == ['2019-11-10', '2023-07-23']
    assert data['blocks'] == ['Derecha', 'Izquierda']
    assert 'id="sector-chart"' in page and 'aria-label=' in page
    table = re.search(r'<table id="events-table">(.*?)</table>', page, re.S).group(1)
    assert [re.sub(r'<.*?>', '', th) for th in re.findall(r'<th scope="col">(.*?)</th>', table)] == [
        'Elección', 'Sondeos', 'Casas', 'Error medio', 'Error de bloques', 'Sesgo del margen', 'Modelo']



@pytest.fixture
def history_site(events_site, fresh_app):
    write_analysis(fresh_app.fs)
    return events_site


def test_elecciones_only_counts_the_published_elections(history_site, fresh_app):
    errors = fixture_data('poll-errors')
    errors['rows'].append(dict(errors['rows'][0], event_date='2011-11-20', event='2011-20N'))
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es')
    data = initial(call(history_site, '/elecciones')[2])
    assert {row['event_date'] for row in data['errors']} <= {'2023-07-23', '2019-11-10'}
    assert [e['date'] for e in data['events']] == ['2023-07-23', '2019-11-10'] and data['error_blocks'] == ['Derecha', 'Izquierda']
    page = html(call(history_site, '/elecciones')[2])
    assert 'En las 2 elecciones analizadas' in page and 'id="errors-filter"' in page and 'id="errors-histogram"' in page
    assert 'id="errors-bullseye"' in page and 'id="weeks-table"' in page and 'Sesgo de bloques' not in page and 'Desviación hacia' not in page
    assert 'id="glosario"' in page and '¿Cómo calculamos el rating de las encuestadoras?' in page


def test_elecciones_without_analysis_keeps_the_list(events_site):
    page = html(call(events_site, '/elecciones')[2])
    assert 'Análisis no publicado todavía' in page and 'id="events-table"' in page and '2 elecciones analizadas' in page


def test_eleccion_detail_has_the_error_histogram_and_the_glossary(events_site):
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    page = html(body)
    assert 'id="poll-histogram"' in page and 'id="glosario"' in page and 'Sesgo de bloques' not in page
    assert initial(body)['poll_errors'][0].keys() >= {'error_avg', 'error_blocks_abs'}


def test_elecciones_history_context_filters_and_headline(history_site):
    context = pages.elecciones_context(pages.parse_state({'scope': 'es'}))
    assert not context['analysis_missing'] and len(context['errors']) == 2
    assert context['filter_events'] == [{'date': '2023-07-23', 'name': 'Generales 2023'},
                                        {'date': '2019-11-10', 'name': 'Generales 2023'}]
    assert context['filter_pollsters'] == ['GAD3', 'Sigma Dos']
    assert context['headline_text'] == ('En las 2 elecciones analizadas, el último sondeo de cada casa se desvió de '
                                        'media 3,6 puntos en el margen entre bloques.')
    assert all(row['error_blocks_abs'] == abs(row['error_blocks']) for row in context['errors'])
    assert set(context['initial']) >= {'state', 'sector', 'blocks', 'errors', 'error_blocks', 'events'}
    assert pages.error_rows('es', ['2019-11-10']) == [] and pages.error_rows('es-md', ['2023-07-23']) is None
    page = html(call(history_site, '/elecciones')[2])
    assert page.index('id="filters"') < page.index('id="histogram"') < page.index('id="bullseye"') \
        < page.index('id="weeks"') < page.index('id="sector"') < page.index('id="events"') < page.index('id="glosario"') \
        < page.index('id="closing"')
    assert '<option value="2019-11-10">' in page and '<option value="Sigma Dos">' in page
    assert 'positivo, el sondeo daba más margen a Derecha del que hubo' in page
    assert 'Herding' not in page and 'La tabla necesita JavaScript' in page
    assert page.count('Análisis no publicado todavía') == 0

def test_elecciones_without_events_shows_a_notice(api):
    status, _, body = call(api, '/elecciones', query='scope=es')
    assert status == 200 and 'Todavía no hay elecciones publicadas' in html(body) and 'initial-data' not in html(body)
    assert '/dist/js/pages/elecciones.js' in html(body)  # el módulo se carga y debe tolerar la falta del bloque


def test_eleccion_detail_renders_results_pollsters_bullseye_and_model(events_site):
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    page = html(body)
    assert status == 200 and 'Generales 2023' in page
    assert page.count('/api/v1/events/es/2023-07-23/') == 6
    assert page.index('GAD3') < page.index('40dB') < page.index('CIS')
    assert '/casas?scope=es&amp;id=2' in page and 'href="/elecciones?scope=es"' in page
    assert 'Cs' in page and pages.OTHERS_COLOR in page  # partido fuera del catálogo del run
    data = initial(body)
    assert set(data) == {'state', 'meta', 'pollsters', 'bullseye', 'blocks', 'poll_errors'}
    assert data['bullseye'][0]['Derecha'] == 0.68 and data['pollsters'][0]['pollster'] == 'GAD3'
    assert 'id="error-chart"' in page and 'id="bullseye-chart"' in page


def test_eleccion_without_model_shows_a_notice(events_site):
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2019-11-10')
    page = html(body)
    assert status == 200 and 'Sin backtest para esta elección' in page
    assert page.count('/api/v1/events/es/2019-11-10/') == 3


@pytest.mark.parametrize('query, status, text', [('scope=es&date=2023-13-01', 400, 'Fecha no válida.'),
                                                 ('scope=es&date=2020-01-01', 404, 'Elección no publicada.')])
def test_eleccion_errors(events_site, query, status, text):
    st, headers, body = call(events_site, '/elecciones', query=query)
    assert st == status and text in html(body) and headers['cache-control'] == 'no-store'


def test_published_events_and_horizon_tables(events_site):
    events = pages.published_events('es')
    assert [event['date'] for event in events] == ['2023-07-23', '2019-11-10']
    assert events[0]['name'] == 'Generales 2023' and events[0]['has_model'] and not events[1]['has_model']
    assert set(events[0]) == {'date', 'name', 'n_polls', 'n_pollsters', 'error_avg', 'error_blocks', 'error_blocks_abs',
                              'bias_blocks', 'bias', 'has_model'}
    context = pages.eleccion_context(pages.parse_state({'scope': 'es'}), '2023-07-23')
    tables = context['horizon_tables']
    assert [table['horizon'] for table in tables] == [6, 30]
    pp = tables[0]['shares'][0]
    assert pp['name'] == 'PP' and pp['error'] == round(32.41 - 33.05, 2)
    assert set(tables[0]['blocks'][0]) == {'block', 'official_seats', 'model_seats', 'p_majority', 'majority'}
    assert pages.event_part('es', '2020-01-01', 'meta') is None
    with pytest.raises(HttpError) as missing:
        pages.resolve_event('es', '2020-01-01')
    assert missing.value.status == 404 and str(missing.value.message) == 'event not published'
    with pytest.raises(HttpError) as invalid:
        pages.resolve_event('es', '2023-13-01')
    assert invalid.value.status == 400 and str(invalid.value.message) == 'invalid date'


def test_horizon_tables_flag_the_horizons_whose_seats_are_not_valid():
    model = {'horizons': [6, 30],
             'shares': [{'horizon': h, 'party': 'PP', 'main': True, 'official': 33.0, 'model': 32.0} for h in (6, 30)],
             'seats': [{'horizon': h, 'party': 'PP', 'main': True, 'official': 137, 'model': 140, 'seats_valid': h == 6}
                       for h in (6, 30)],
             'blocks': [{'horizon': h, 'block': 'Derecha', 'official_seats': 169, 'model_seats': 171,
                         'p_majority': 0.4, 'majority': False, 'seats_valid': h == 6} for h in (6, 30)]}
    tables = pages.horizon_tables(model)
    assert [(table['horizon'], table['seats_valid']) for table in tables] == [(6, True), (30, False)]
    assert tables[1]['shares'][0]['error'] == -1.0  # el voto sigue ahí aunque los escaños no valgan


def test_eleccion_detail_replaces_the_seats_of_an_invalid_horizon_by_a_notice(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'model':
            data = {**data, 'seats': [dict(row, seats_valid=row['horizon'] != 30) for row in data['seats']]}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    page = html(body)
    assert status == 200 and page.count('Escaños no proyectables en esta elección') == 1
    assert 'id="model-shares-30"' in page and 'id="model-shares-6"' in page
    assert 'id="model-seats-6"' in page and 'id="model-blocks-6"' in page
    assert 'id="model-seats-30"' not in page and 'id="model-blocks-30"' not in page


def test_eleccion_detail_orders_houses_by_the_absolute_block_error(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'pollsters':
            rows = [dict(row, error_blocks=error) for row, error in zip(data['rows'], (-3.0, 1.0, 2.0))]
            data = {**data, 'rows': rows}
        if name == 'meta':
            data = {**data, 'summary': {**data['summary'], 'error_blocks': -2.5}}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    names = [row['pollster'] for row in real('es', '2023-07-23', 'pollsters')['rows']]
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    page = html(body)
    table = re.search(r'<table id="pollsters-table">(.*?)</table>', page, re.S).group(1)
    assert status == 200 and 'Sesgo del margen' in table and '-3,0' in table
    assert 'Error de bloques: 2,5 · Sesgo del margen: -2,5' in page
    shown = [re.sub(r'<.*?>', '', cell) for cell in re.findall(r'<tr>\s*<td>(.*?)</td>', table, re.S)]
    assert shown == [names[1], names[2], names[0]]
    data = initial(body)
    assert [row['error_blocks_abs'] for row in data['pollsters']] == [1.0, 2.0, 3.0]
    assert [row['error_blocks'] for row in data['pollsters']] == [1.0, 2.0, -3.0]


def test_pollster_rows_without_error_go_last_and_have_no_absolute_value(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'pollsters':
            errors = (None, -2.0, 1.0)
            data = {**data, 'rows': [dict(row, error_blocks=error) for row, error in zip(data['rows'], errors)]}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    context = pages.eleccion_context(pages.parse_state({'scope': 'es'}), '2023-07-23')
    assert [row['error_blocks_abs'] for row in context['pollster_rows']] == [1.0, 2.0, None]
    assert all('error_blocks_abs' in row for row in context['poll_rows'])


def test_pollster_row_without_id_is_plain_text(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'pollsters':
            data = {**data, 'rows': [dict(row, pollster_id=None) for row in data['rows']]}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    status, _, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    table = re.search(r'<table id="pollsters-table">(.*?)</table>', html(body), re.S).group(1)
    assert status == 200 and 'GAD3' in table and '<a ' not in table


def test_eleccion_without_meta_is_a_404(events_site, fresh_app, monkeypatch):
    real = pages.elecciones.event_part
    monkeypatch.setattr(pages.elecciones, 'event_part',
                        lambda scope, date, name: None if name == 'meta' and date == '2023-07-23' else real(scope, date, name))
    st, headers, body = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    assert st == 404 and 'Elección no publicada.' in html(body) and headers['cache-control'] == 'no-store'


@pytest.fixture
def backtest_site(api, fresh_app):
    """Páginas con el backtest de `es` publicado."""
    write_backtest(fresh_app.fs)
    return api


def test_backtest_renders_the_backtest_tables_and_initial(backtest_site):
    status, _, body = call(backtest_site, '/metodo/backtest', query='scope=es')
    page = html(body)
    assert status == 200 and '1,69' in page and page.count('/api/v1/backtest/es/') == 5
    assert 'id="mae-chart"' in page and 'id="coverage-chart"' in page and 'aria-label=' in page
    data = initial(body)
    assert data['horizons'] == [6, 30] and data['mae']['model'][0] == 1.69 and data['coverage']['95'][0] == 0.957
    assert data['mae']['last_poll'][0] == 1.74 and '5,77' in page
    assert data['seats'] == {'mae': [4.88, 5.25], 'mae_h': [None, None]}  # sin ejecución a horizonte en el fixture
    blocks = re.search(r'<table id="blocks-table">(.*?)</table>', page, re.S).group(1)
    rows = [re.findall(r'<td>(.*?)</td>', row) for row in re.findall(r'<tr>(.*?)</tr>', blocks, re.S) if '<td>' in row]
    invalid = [row for row in rows if row[0].startswith('10 ') and row[1] == '30']  # 2019-11-10 x 30
    assert len(invalid) == 2 and len(rows) == 8
    assert all(row[4:7] == ['—', '—', '—'] and row[7] == 'no' for row in invalid)
    assert all(row[7] == 'sí' for row in rows if row not in invalid)


def test_backtest_tolerates_a_backtest_without_run_json(api, fresh_app):
    write_backtest(fresh_app.fs, meta={'params': {}, 'commit': None, 'run_at': None})
    status, _, body = call(api, '/metodo/backtest', query='scope=es')
    page = html(body)
    assert status == 200 and 'id="params"' in page and 'Simulaciones</dt><dd>—</dd>' in page
    assert 'Versión</dt><dd>—</dd>' in page and 'Ejecutado</dt><dd>—</dd>' in page


def test_backtest_without_backtest_shows_a_notice(api):
    status, _, body = call(api, '/metodo/backtest', query='scope=es')
    page = html(body)
    assert status == 200 and 'Backtest no publicado todavía' in page and 'initial-data' not in page
    assert '/dist/js/pages/metodo.js' in page and '<title>Pronóstico electoral · Cuánto acierta el modelo</title>' in page
    assert 'action="/metodo/backtest"' in page and 'aria-current="page">Método</a>' in page


def test_backtest_rows_are_ordered_and_projected():
    rows = pages.project(fixture_data('backtest-metrics')['rows'], pages.backtest.CASE_COLUMNS)
    ordered = pages.backtest.case_rows(rows)
    assert [(r['event_date'], r['horizon']) for r in ordered] == [
        ('2023-07-23', 6), ('2023-07-23', 30), ('2019-11-10', 6), ('2019-11-10', 30)]
    assert set(ordered[0]) == set(pages.backtest.CASE_COLUMNS)


def test_metodo_renders_the_sections_without_data(api):
    status, _, body = call(api, '/metodo', query='scope=es')
    page = html(body)
    assert status == 200 and 'initial-data' not in page
    assert re.findall(r'<script type="module" src="([^"]+)"', page) == ['/dist/js/pages/metodo.js']
    assert re.findall(r'/dist/js/pages/[a-z]+\.js', page) == ['/dist/js/pages/metodo.js']
    sections = re.findall(r'<section id="([a-z-]+)" class="card">(.*?)</section>', page, re.S)
    assert [section_id for section_id, _ in sections] == ['intro', 'datos', 'como-funciona', 'publicacion',
                                                           'limitaciones']
    assert all('<p>' in body for _, body in sections)  # ninguna sección sin texto en la plantilla
    for section_id, title in pages.SECTIONS + pages.LEGACY_SECTIONS:
        assert 'id="{}"'.format(section_id) in page and title in page
    legacy = dict(sections)['como-funciona']
    assert re.findall(r'<h3 id="([a-z]+)">', legacy) == [section_id for section_id, _ in pages.LEGACY_SECTIONS]
    assert all(len(text.strip()) > 100 for text in re.split(r'<h3 id="[a-z]+">', legacy)[1:])
    assert '10.000 elecciones' in page and 'contiene el' not in page  # sin backtest, sin frase de cobertura
    for href in ('/metodo/backtest?scope=es', '/casas?scope=es', '/elecciones?scope=es', '/es/escanos'):
        assert 'href="{}'.format(href) in page


def test_metodo_landing_quotes_the_coverage_of_the_general_backtest(api, fresh_app):
    write_backtest(fresh_app.fs)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'contiene el resultado el 96 % de las veces a 6 días de la votación y el 100 % a 30 días' in page
    assert 'En las generales: 2 elecciones, a 6 y 30 días de la votación.' in page


def test_metodo_landing_reads_the_general_barrier_from_the_catalogue(api, monkeypatch):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'en las generales, el 3 % de los votos válidos' in page
    changed = webapi.catalogue().copy()
    changed.loc['es', 'threshold'] = 4.5
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: changed)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'en las generales, el 4,5 % de los votos válidos' in page and 'el 3 % de los votos' not in page
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: changed.drop(index='es'))
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'en las generales, la que fija la ley electoral, de los votos válidos' in page


def test_metodo_landing_names_a_single_backtest_election_in_singular(api, fresh_app):
    write_backtest(fresh_app.fs, meta={'events': ['2023-07-23'], 'horizons': [6]})
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'En las generales: 1 elección, a 6 días de la votación.' in page


def test_metodo_landing_dates_the_freeze_only_while_frozen(api, fresh_app):
    page = html(call(api, '/metodo')[2])
    assert 'con el pronóstico definitivo;' in page and 'desde el' not in page
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda'},
                            today='2026-11-23')
    webapi.site().manifest_cache_clear()
    assert 'con el pronóstico definitivo: desde el 23 nov 2026;' in html(call(api, '/metodo')[2])


def test_metodo_subpage_404_and_section_context(api, fresh_app):
    for section in ('nope', 'escanos'):
        with pytest.raises(HttpError) as error:
            pages.metodo_section_context({'scope': None, 'mode': None, 'run': None}, section)
        assert error.value.status == 404 and error.value.message == 'section not found', section
    write_backtest(fresh_app.fs)
    context = pages.metodo_section_context({'scope': 'es', 'mode': None, 'run': None}, 'backtest')
    assert context['section'] == 'backtest' and context['published'] and context['subpages'][3]['active']
    assert context['figures']['n_sim'] == 10000 and len(context['thresholds']) == len(webapi.catalogue())


def test_metodo_landing_has_the_index_and_only_links_available_sections(api):
    page = html(call(api, '/metodo')[2])
    assert '¿Cómo promediamos las encuestas?' in page and 'href="/metodo/backtest"' in page
    assert 'href="/metodo/promedio"' in page and 'href="/metodo/escanos"' not in page  # escanos, sin texto todavía
    assert 'id="datos"' in page and 'id="publicacion"' in page and 'id="limitaciones"' in page
    assert 'se congela' in page and 'fuera de España' in page
    assert '1.000 simulaciones' not in page and 'id="scope-filter"' not in page


def test_metodo_promedio_text(api):
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and '<title>Pronóstico electoral · ¿Cómo promediamos las encuestas?</title>' in page
    for anchor in ('que-entra', 'peso', 'efectos-de-casa', 'regresion', 'banda', 'voto-estimado'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'id="metodo-nav"' in page and '<math' in page and 'En resumen' in page
    assert 'href="/metodo/promedio"' in html(call(api, '/metodo')[2])


def test_thresholds_and_figures_come_from_the_catalogue_and_the_bundle(api, fresh_app):
    write_backtest(fresh_app.fs)
    rows = {row['code']: row for row in pages.thresholds_rows()}
    assert rows['es']['threshold'] == 3 and rows['es-cn']['threshold_scope'] == 4 and rows['es-md']['rating_weight'] == 0.5
    figures = pages.national_figures()
    assert figures['n_sim'] == fixture_data('meta')['n_sim'] and figures['coverage'][0]['horizon'] == min(fixture_data('backtest-meta')['horizons'])


def test_backtest_subpage_has_the_seats_chart_and_the_subnav(api, fresh_app):
    write_backtest(fresh_app.fs)
    status, _, body = call(api, '/metodo/backtest', query='scope=es')
    page = html(body)
    assert status == 200 and 'id="seats-chart"' in page and 'id="metodo-nav"' in page and 'id="scope-filter"' in page
    assert initial(body)['seats'].keys() == {'mae', 'mae_h'}


def test_metodo_is_in_the_menu_and_the_footer_of_every_page(api):
    page = html(call(api, '/es')[2])
    assert page.count('>Método</a>') == 2 and '>Histórico de errores</a>' in page and '>Modelo</a>' not in page


def test_frozen_manifest_marks_every_page(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda electoral'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es')
    page = html(body)
    assert status == 200 and 'Pronóstico definitivo para el' in page and 'congelado desde 23 nov 2026' in page
    assert 'freeze-banner' in page and 'href="/metodo#limitaciones"' in page
    status, _, body = call(api, '/es', query='mode=nowcast')
    assert 'Estimación a' in html(body) and 'Pronóstico definitivo' not in html(body)
    status, _, body = call(api, '/metodo', query='scope=es')
    assert 'freeze-banner' in html(body)
    publish.update_manifest(writer, freeze={'active': False})
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es', query='mode=forecast')
    assert 'freeze-banner' not in html(body) and 'Pronóstico para el' in html(body) and 'congelado desde' not in html(body)


def test_home_lists_the_published_events_with_their_headline(api, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    status, headers, body = call(api, '/')
    page = html(body)
    assert status == 200 and '<title>Pronóstico electoral · Inicio</title>' in page
    assert page.index('Elecciones generales') < page.index('Elecciones autonómicas · Madrid')
    assert 'href="/es"' in page and 'href="/es/escanos"' in page and 'href="/es-md"' in page
    leader = max(fixture_data('headline')['nowcast']['parties'], key=lambda row: row['pct'])
    assert leader['name'] in page and 'de probabilidad de mayoría absoluta' in page
    assert 'id="event-bar"' not in page and 'id="scope-filter"' not in page and 'initial-data' not in page


def test_home_tolerates_a_scope_whose_run_has_no_headline(api, fresh_app, tmp_path):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es-md', '20261009-120000', 'headline')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/')
    assert status == 200 and 'Elecciones autonómicas · Madrid' in html(body)


def test_home_lists_the_past_elections_with_a_detail_link(events_site):
    page = html(call(events_site, '/')[2])
    assert page.index('2023') < page.index('2019') and 'href="/elecciones?scope=es&amp;date=2023-07-23"' in page


def test_every_page_uses_the_card_pattern_and_the_fonts(events_site):
    for path in ('/', '/es', '/es/escanos', '/elecciones', '/elecciones?scope=es&date=2023-07-23', '/metodo'):
        page = html(call(events_site, path.split('?')[0], query=path.partition('?')[2])[2])
        assert 'Qué es esto' not in page and '<meta name="description"' in page, path
        if path not in ('/', '/metodo'):
            assert 'class="how"' in page and 'class="lead"' in page, path


def test_houses_pages_use_the_card_pattern(houses):
    for query in ('scope=es', 'scope=es&id=2'):
        page = html(call(houses, '/casas', query=query)[2])
        assert 'Qué es esto' not in page and '<meta name="description"' in page, query
        assert 'class="how"' in page and 'class="lead"' in page, query


def test_backtest_page_uses_the_card_pattern(api, fresh_app):
    write_backtest(fresh_app.fs)
    page = html(call(api, '/metodo/backtest', query='scope=es')[2])
    assert 'Qué es esto' not in page and '<meta name="description"' in page
    assert 'class="how"' in page and 'class="lead"' in page


def test_first_card_opens_its_how_to_read_and_the_rest_stay_folded(api):
    page = html(call(api, '/es/escanos')[2])
    main = page[page.index('<main'):]
    hows = re.findall(r'<details class="how"( open)?>', main)
    assert len(hows) > 1 and hows[0] == ' open' and set(hows[1:]) == {''}


def test_event_cards_of_the_home_are_h3_under_the_section_h2(api):
    page = html(call(api, '/')[2])
    events = page[page.index('id="events"'):page.index('</section>', page.index('id="events"'))]
    assert '<h2>Próximas elecciones</h2>' in events and '<h3><a href="/es">' in events
    assert events.count('<h2') == 1 and page.count('<h1') == 1


def test_fonts_are_vendored_and_declared():
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert "url('/dist/vendor/fonts/" in css and 'font-display: swap' in css
    assert os.path.isfile(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'vendor', 'fonts', 'LICENSE'))


def test_casas_uses_the_glossary_and_drops_the_block_bias(houses):
    page = html(call(houses, '/casas')[2])
    assert 'casas con rating' in page and 'id="glosario"' in page and 'Sesgo de bloques' not in page
    assert '¿Cómo calculamos el rating de las encuestadoras?' in page and 'id="scope-filter"' in page
    ranking = page[page.index('id="ranking-table"'):]
    heads = re.findall(r'<th scope="col">([^<]*)</th>', ranking[:ranking.index('</thead>')])
    assert heads == ['Casa', 'Rating', 'Calidad', 'Sondeos', 'Elecciones', 'Error medio', 'Sesgo del margen', 'Herding']


def test_casa_profile_shows_the_magnitude_and_the_signed_margin(houses):
    page = html(call(houses, '/casas', query='id=1')[2])
    assert '<th scope="col">Sesgo del margen</th>' in page and 'Error de bloques' not in page
    assert 'Sesgo de bloques' not in page and 'miden lo mismo' not in page and 'id="glosario"' in page
    assert page.count('<td') == page.count('</td>')
    row = page[page.index('id="path-table"'):]
    row = row[row.index('<tbody>'):]
    assert row[:row.index('</tr>')].count('<td') == 5
