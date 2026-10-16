"""Tests de las páginas servidas por los controladores (`Page`, `Index`, `Promedios`, `Escanos`, `Casas`...) con el
arnés ASGI."""
import json
import os
import re
from datetime import date
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


def data_href(body, key):
    """`href` del evento `key` (`ámbito:fecha`) en el JSON inicial del histórico."""
    return {event['key']: event['href'] for event in initial(body)['events']}[key]


def site_api(app):
    """API completa (páginas + /api/v1 + not_found) registrada en `app`."""
    return mtpy.api(routes=pages.ROUTES + webapi.ROUTES, not_found=pages.NOT_FOUND)


@pytest.fixture
def api(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return site_api(fresh_app)


FIXTURE_POLLS = 1  # polls of the `polls` part of tests/fixtures/bundle


def html(body):
    return body.decode('utf-8')

def header_tooltips(table):
    """{etiqueta: tooltip} de las cabeceras de una tabla, con los espacios normalizados."""
    import html as html_mod
    return {label: re.sub(r'\s+', ' ', html_mod.unescape(title)).strip()
            for title, label in re.findall(r'<th scope="col"[^>]*title="([^"]*)"[^>]*>([^<]*)</th>', table)}


def glossary_entries(glossary):
    """{término: definición} de un glosario `<dl>`, con los espacios normalizados."""
    import html as html_mod
    return {dt: re.sub(r'\s+', ' ', html_mod.unescape(re.sub(r'<[^>]+>', '', dd))).strip()
            for dt, dd in re.findall(r'<dt>([^<]*)</dt>\s*<dd>(.*?)</dd>', glossary, re.S)}



def initial(body):
    match = re.search(r'<script type="application/json" id="initial-data">(.*?)</script>', html(body), re.S)
    assert match, 'sin bloque initial-data'
    return json.loads(match.group(1))


def test_promedios_is_rendered_html_with_cache_headers(api):
    status, headers, body = call(api, '/es', query='mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=60' and headers['etag'].startswith('"')
    assert '<title>Demoscopio · Promedios</title>' in page
    assert 'pronóstico para el' in page and 'href="/es?mode=forecast" aria-current="page"' in page
    assert 'name="mode"' not in page  # `?mode=` switches the page, but no control is rendered
    assert 'id="parties-table"' in page and '<th scope="col">Partido</th>' in page
    assert fixture_data('headline')['forecast']['parties'][0]['name'] in page
    assert '/dist/css/site.css' in page and '/dist/vendor/echarts-5.6.0.min.js' in page and '/dist/js/pages/promedios.js' in page
    assert 'onclick=' not in page and '<script>' not in page
    assert page.count('<script') == 3  # ECharts (defer), el módulo de la página y el bloque JSON


def test_promedios_initial_data_has_the_page_parts_with_one_run(api):
    data = initial(call(api, '/es')[2])
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs', 'cards', 'inks'}
    assert data['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    assert list(data['state']) == ['scope', 'mode', 'run', 'pinned']
    assert data['meta']['run_id'] == RUN and data['headline']['run_id'] == RUN
    assert data['runs']['runs'][0]['run_id'] == RUN


def test_nowcast_and_pinned_run_are_reflected(api):
    status, headers, body = call(api, '/es', query='mode=nowcast&run=' + RUN)
    page = html(body)
    assert status == 200 and 'estimación a' in page
    assert 'name="mode"' not in page and '<span class="pill event-run">Run del 8 oct 2026</span>' in page
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
    assert text in html(body) and 'alt="Demoscopio"' in html(body)


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
    page = html(call(api, '/metodo/backtest')[2])
    assert '<option value="es" selected>' in page and '<option value="es-md">Madrid</option>' in page


def test_event_pages_live_under_the_scope_and_default_to_nowcast(api):
    status, headers, body = call(api, '/es')
    page = html(body)
    assert status == 200 and headers['cache-control'] == 'public, max-age=60'
    assert '<title>Demoscopio · Promedios</title>' in page
    assert 'estimación a' in page and 'id="event-bar"' in page and 'Elecciones generales' in page
    assert 'name="mode"' not in page
    assert 'href="/es/escanos?mode=nowcast"' in page and 'id="controls"' not in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': False}
    status, _, body = call(api, '/es/escanos', query='mode=forecast&run=' + RUN)
    assert status == 200 and 'pronóstico para el' in html(body) and 'Run del 8 oct 2026' in html(body)


def test_frozen_site_defaults_to_the_forecast_and_an_older_pinned_run_is_not_definitive(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    write_bundle(fresh_app.fs, run_id='20261001-120000', manifest=False)
    publish.update_manifest(writer, {'es': publish.latest_entry(fixture_data('headline') | {'run_id': RUN})},
                            freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    assert 'pronóstico definitivo para el' in html(call(api, '/es')[2])
    page = html(call(api, '/es', query='run=20261001-120000&mode=forecast')[2])
    assert 'pronóstico publicado el' in page and 'definitivo' not in page


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
    assert 'id="scope-filter"' not in html(call(api, '/elecciones')[2])  # el histórico lleva su propio filtro de ámbito
    for path in ('/metodo/backtest',):
        page = html(call(api, path)[2])
        assert 'id="scope-filter"' in page and 'id="event-bar"' not in page and 'name="mode"' not in page, path
    landing = html(call(api, '/metodo')[2])
    assert 'id="scope-filter"' not in landing and 'id="event-bar"' not in landing
    assert '<option value="es" selected>' in html(call(api, '/metodo/backtest', query='scope=es')[2])


def test_menu_has_the_four_global_entries(api):
    page = html(call(api, '/es')[2])
    for label in ('Inicio', 'Histórico Errores', 'Ranking Encuestadoras', 'Método'):
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


def test_event_info_counts_one_day_and_unknown_dates_and_marks_the_tab():
    from datetime import date
    state = {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': True}
    scopes = [{'code': 'es', 'name': 'España'}]
    info = pages.event_info('es', {'event_date': '2026-11-29'}, '/escanos', state, today=date(2026, 11, 28),
                            scopes=scopes)
    assert info['days_text'] == 'Falta 1 día' and 'form_action' not in info and info['href'] == '/es'
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
    from datetime import date
    assert pages.title_of('nowcast', meta, True, True, True, today=date(2026, 10, 20)) == 'Estimación del 13 oct 2026'
    assert pages.title_of('nowcast', meta, False, False, True, today=date(2026, 10, 20)) == 'Estimación a 20 oct 2026'
    assert pages.title_of('forecast', meta, True, False, True) == 'Pronóstico definitivo para el 29 nov 2026'
    assert pages.title_of('forecast', meta, True, True, False) == 'Pronóstico publicado el 13 oct 2026'
    assert pages.title_of('forecast', meta, False, True, False) == 'Pronóstico publicado el 13 oct 2026'
    assert pages.title_of('forecast', meta, False, False, True) == 'Pronóstico para el 29 nov 2026'

def test_footer_names_the_publication_and_the_last_poll(api):
    page = html(call(api, '/es')[2])
    footer = page[page.index('<footer'):page.index('</footer>')]
    assert 'Actualizado: {} ('.format(RUN) in footer and '· Último sondeo: ' in footer
    assert 'commit' not in footer and 'promedio leído' not in footer and '(sucio)' not in footer


def test_event_bar_and_promedios_have_no_mode_sentence_nor_headline(api):
    page = html(call(api, '/es')[2])
    assert 'mode-text' not in page and 'Si las elecciones fueran hoy' not in page
    assert 'class="headline"' not in page and 'Si hoy hubiera elecciones, ' not in page
    assert 'class="headline"' not in html(call(api, '/es/escanos')[2])  # tampoco en Escaños (2026-10-14)


def test_method_subpages_and_links():
    assert list(pages.METODO_SECTIONS) == ['promedio', 'escanos', 'rating', 'backtest']
    subpages = pages.subpages('backtest')
    assert [page['id'] for page in subpages] == ['promedio', 'escanos', 'rating', 'backtest']
    assert subpages[3] == {'id': 'backtest', 'title': '¿Cuánto acierta el modelo?', 'href': '/metodo/backtest',
                           'active': True, 'available': True}
    assert not subpages[0]['active'] and all(page['available'] for page in subpages)
    assert pages.method_href('backtest') == '/metodo/backtest' and pages.method_href('rating') == '/metodo/rating'
    assert pages.method_href('nope') == '/metodo'
    assert pages.method_href('promedio') == '/metodo/promedio' and pages.method_href('escanos') == '/metodo/escanos'


def test_header_shows_the_wordmark_the_tagline_and_the_icons(api):
    """La cabecera lleva el logotipo (SVG con el nombre como alt), la línea bajo el nombre, los iconos y las
    metaetiquetas Open Graph con el mismo título y descripción que la página."""
    status, headers, body = call(api, '/es', query='mode=forecast')
    page = html(body)
    assert status == 200
    assert '<h1><a href="/"><img src="/dist/img/logo.svg" alt="Demoscopio" width="574" height="100"></a></h1>' in page
    assert 'brand-tagline' not in page  # the tagline only lives in the description since phase 8
    assert '<link rel="icon" href="/dist/img/favicon.svg" type="image/svg+xml">' in page
    assert '<link rel="icon" href="/dist/img/favicon.ico" sizes="16x16 32x32 48x48">' in page
    assert '<link rel="apple-touch-icon" href="/dist/img/apple-touch-icon.png">' in page
    assert '<meta property="og:site_name" content="Demoscopio">' in page
    assert '<meta property="og:title" content="Demoscopio · Promedios">' in page
    assert '<link rel="canonical" href="https://demoscopio.es/es">' in page
    assert '<meta property="og:url" content="https://demoscopio.es/es">' in page
    description = re.search(r'<meta name="description" content="([^"]+)">', page).group(1)
    assert description and description == re.search(r'<meta property="og:description" content="([^"]+)">', page).group(1)
    assert 'Pronóstico electoral' not in page


def test_canonical_drops_the_view_parameters_and_keeps_the_content_ones(api, fresh_app):
    """La canonical es la URL pública de la página sin `mode` ni `run` (vistas de la misma página), con `scope` y los
    demás parámetros que eligen el contenido, y el origen sale de `web.site_url` (`WEB_SITE_URL`)."""
    status, headers, body = call(api, '/es', query='mode=nowcast&run=' + RUN)
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.es/es">' in html(body)
    status, headers, body = call(api, '/es/escanos', query='region=28&mode=forecast')
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.es/es/escanos?region=28">' in html(body)
    status, headers, body = call(api, '/metodo', query='scope=es')
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.es/metodo?scope=es">' in html(body)
    status, headers, body = call(api, '/')
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.es/">' in html(body)
    from mtpy.core.app import Config
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/'}}))
    status, headers, body = call(api, '/es')
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.dev/es">' in html(body)
    assert '<meta property="og:url" content="https://demoscopio.dev/es">' in html(body)


def test_error_page_carries_the_brand_and_the_default_description(api):
    """El error 404 HTML lleva la misma cabecera e iconos, y describe el sitio con el nombre y la línea bajo él (la
    descripción por defecto de `base.html`, que todas las páginas sobrescriben)."""
    status, headers, body = call(api, '/nope')
    page = html(body)
    assert status == 404 and '<title>Demoscopio · Error 404</title>' in page
    assert ('<meta name="description" content="Demoscopio: promedio de encuestas, previsión de escaños y ranking de '
            'encuestadoras en España.">') in page
    assert '<img src="/dist/img/logo.svg" alt="Demoscopio" width="574" height="100">' in page
    assert '<link rel="icon" href="/dist/img/favicon.svg" type="image/svg+xml">' in page
    assert 'brand-tagline' not in page  # the tagline only lives in the description since phase 8


def test_templates_compile_with_strict_undefined():
    env = pages.environment()
    names = sorted(f for f in os.listdir(pages.TEMPLATES_DIR) if f.endswith('.html'))
    assert {'base.html', '_header.html', '_brand.html', '_icons.html', '_footer.html', '_event_bar.html',
            '_scope_filter.html', 'error.html', 'promedios.html', 'escanos.html', 'casas.html', 'casa.html',
            'metodo_backtest.html'} <= set(names)
    for name in names:
        env.get_template(name)


def test_filters_follow_the_js_rules():
    assert pages.fmt_pct(32.66) == '32,7%' and pages.fmt_pct(None) == '–' and pages.fmt_pct(40, 0) == '40%'
    assert pages.fmt_num(1234567) == '1.234.567' and pages.fmt_num(3.14159, 1) == '3,1'
    assert pages.fmt_num(-1234.5, 1) == '-1.234,5' and pages.fmt_num(-1.25, 1) == '-1,3'
    assert pages.fmt_prob(0.995, 0) == '> 99%' and pages.fmt_prob(0.003, 0) == '< 1%' and pages.fmt_prob(0) == '0,0%'
    assert pages.fmt_prob(1) == '100,0%' and pages.fmt_prob(0.968) == '96,8%' and pages.fmt_prob(0.968, 0) == '97%' and pages.fmt_prob(None) == '–'
    # Con un decimal (el filtro `prob` de las tablas): el tope y el suelo se mueven un orden de magnitud
    assert pages.fmt_prob(0.968, 1) == '96,8%' and pages.fmt_prob(0.995) == '99,5%' and pages.fmt_prob(0.9995) == '> 99,9%'
    assert pages.fmt_prob(0.003) == '0,3%' and pages.fmt_prob(0.0004) == '< 0,1%' and pages.fmt_prob(0, 0) == '0%' and pages.fmt_prob(1, 0) == '100%'
    assert pages.fmt_range(27.31, 38.02) == '27,3–38,0' and pages.fmt_range(118.0, 159.5, 0) == '118–160'
    assert pages.fmt_date('2026-10-13') == '13 oct 2026' and pages.fmt_date('2026-09-05') == '5 sept 2026'
    assert pages.fmt_date('garbage') == '–' and pages.fmt_date(None) == '–'
    assert pages.fmt_datetime('2026-10-08T18:11:02Z') == '8 oct 2026, 20:11'
    assert pages.api_url('es', 'polls', run=RUN, fmt='csv') == '/api/v1/forecast/es/polls?run={}&format=csv'.format(RUN)
    assert pages.api_url('es', 'vote', mode='forecast') == '/api/v1/forecast/es/forecast/vote'


def test_promedios_has_the_series_chart_the_polls_table_and_csv_link(api):
    body = call(api, '/es', query='mode=forecast')[2]
    page = html(body)
    assert 'id="series-chart"' in page and 'Todas las encuestas' in page
    assert 'href="/api/v1/forecast/es/polls?run={}&amp;format=csv"'.format(RUN) in page
    assert '<th scope="col" data-key="date" aria-sort="descending"><button type="button" class="sort">Fecha</button></th>' in page
    data = initial(body)
    assert data['projection']['dates'] and data['vote']['when']


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
    for path in ('/es', '/es/escanos', '/casas', pages.pollster_href(HOSTILE, 1)):
        path, _, query = path.partition('?')
        body = call(api, path, query=query)[2]
        page = html(body)
        rest = block.sub('', page)
        assert '<script>alert' not in page and '&lt;script&gt;alert' in rest
        assert '<b>Hostil' not in page and 'url(' not in rest
        data = initial(body)
        assert data['meta']['parties'][0]['fullname'] == HOSTILE
    assert HOSTILE in [row['pollster'] for row in pages.casas_context(pages.parse_state({'scope': 'es'}))['ranking']]
    home = html(call(api, '/es')[2])
    assert 'background-color: {}'.format(pages.OTHERS_COLOR) in home
    assert '&lt;b&gt;Hostil&lt;/b&gt;' in home


def test_promedios_pinned_run_keeps_the_run_in_the_csv_link(api):
    page = html(call(api, '/es', query='run=' + RUN)[2])
    assert 'run={}&amp;format=csv'.format(RUN) in page
    assert 'name="run"' not in page and 'Run del 8 oct 2026' in page  # the bar shows the run, without a form


def test_promedios_initial_polls_carry_the_chart_and_table_columns(api):
    data = initial(call(api, '/es')[2])
    parties = set(data['polls']['parties'])
    assert set(data['polls']) == {'parties', 'polls'}
    for poll in data['polls']['polls']:
        assert {'date', 'pollster_id', 'pollster', 'name', 'sample_size'} | parties <= set(poll)
        extra = {'sponsor', 'start_date', 'end_date', 'seats'}
        assert set(poll) <= {'date', 'pollster_id', 'pollster', 'name', 'sample_size'} | extra | parties


def test_without_a_published_catalogue_scope_the_home_is_a_503_page(fresh_app, tmp_path):
    """Un manifest cuyo único ámbito no está en el catálogo no se puede pedir por URL: 503, no un `<select>` vacío."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs, scope='zz')
    status, headers, body = call(site_api(fresh_app), '/')
    assert status == 503 and 'Todavía no hay ningún pronóstico publicado' in html(body)


def test_polls_table_marks_the_leader_formats_the_values_and_averages_all_the_polls():
    polls = [
        {'date': '2026-10-01', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1010,
         'start_date': '2026-09-28', 'end_date': '2026-09-30', 'PP': 31.1, 'PSOE': 26.1},
        {'date': '2026-09-04', 'pollster': 'CIS', 'sponsor': None, 'sample_size': None, 'PP': 25.5, 'PSOE': 31.0},
        {'date': '2026-09-02', 'pollster': '40dB', 'sponsor': 'Prisa', 'sample_size': 2000, 'PP': None, 'PSOE': None},
    ]
    colors = {'PP': '#1d84ce', 'PSOE': '#ef1c27'}
    table = pages.polls_table(polls, ['PP', 'PSOE'], colors, n=2)
    assert table['total'] == 3 and [r['pollster'] for r in table['rows']] == ['GAD3', 'CIS']
    first = table['rows'][0]
    assert first['date'] == '01/10/26' and first['sponsor'] == 'ABC' and first['sample'] == '1.010'
    assert first['fieldwork'] == 'Trabajo de campo: 28 sept 2026 – 30 sept 2026'
    assert first['cells'] == [{'value': '31,1', 'leader': True, 'color': '#1d84ce'},
                              {'value': '26,1', 'leader': False, 'color': '#ef1c27'}]
    second = table['rows'][1]
    assert second['sample'] == '–' and second['fieldwork'] is None and second['cells'][1]['leader']
    # the mean covers every poll, not only the rows rendered
    assert table['mean'] == {'n': 3, 'cells': [{'value': '28,3', 'leader': False, 'color': '#1d84ce'},
                                               {'value': '28,6', 'leader': True, 'color': '#ef1c27'}]}
    empty = pages.polls_table(polls[2:], ['PP', 'PSOE'], {'PP': '#1', 'PSOE': '#2'})
    assert empty['rows'][0]['cells'][0] == {'value': '–', 'leader': False, 'color': '#1'}
    assert empty['mean']['cells'][0]['value'] == '–'
    assert pages.poll_months(polls) == [{'value': '2026-09', 'label': 'Sep 2026'},
                                        {'value': '2026-10', 'label': 'Oct 2026'}]


def test_polls_table_orders_by_date_keeps_ties_and_tolerates_a_poll_without_date_or_house():
    polls = [
        {'date': '2026-09-05', 'pollster': 'CIS', 'PP': 41.0},
        {'date': '2026-10-01', 'pollster': 'GAD3', 'PP': 30.5},
        {'date': '2026-10-01', 'pollster': 'CIS', 'PP': 40.5, 'PSOE': 40.5},
        {'pollster': None, 'PP': 29.0},
    ]
    table = pages.polls_table(polls, ['PP', 'PSOE'], {})
    assert [(r['pollster'], r['date']) for r in table['rows']] == [
        ('CIS', '01/10/26'), ('GAD3', '01/10/26'), ('CIS', '05/09/26'), ('–', '–')]
    assert table['rows'][0]['cells'][0]['leader'] and table['rows'][0]['cells'][1]['leader']  # tie: both marked
    assert table['rows'][3]['sponsor'] is None and table['rows'][3]['cells'][1]['color'] == pages.OTHERS_COLOR
    assert len(pages.polls_table(polls * 20, ['PP'], {}, n=None)['rows']) == 80
    assert [m['value'] for m in pages.poll_months(polls)] == ['2026-09', '2026-10']
    assert pages.poll_months([{'pollster': 'CIS'}]) == []
    assert pages.polls_table(polls[:1], ['PP'], {})['mean']['n'] == 1


def test_promedios_title_mentions_the_projection_only_in_forecast(api):
    assert 'Promedio de sondeos y proyección' in html(call(api, '/es', query='mode=forecast')[2])
    page = html(call(api, '/es', query='mode=nowcast')[2])
    assert 'Promedio de sondeos' in page and 'y proyección' not in page
    assert 'y la proyección' not in page  # aria-label de los gráficos
    assert 'banda de incertidumbre y la proyección"' in html(call(api, '/es', query='mode=forecast')[2])


def polls_tbody(page):
    start = page.index('<tbody>', page.index('id="polls-table"'))
    return page[start:page.index('</tbody>', start)]


def test_promedios_polls_table_is_server_rendered_with_fifteen_rows_and_hidden_controls(api):
    body = call(api, '/es')[2]
    page = html(body).replace('\n', ' ')
    assert 'Todas las encuestas' in page and 'id="cycle"' not in page and 'id="polls-more"' not in page
    assert re.search(r'<div class="ctrl-bar" id="polls-controls" hidden>', page)
    assert 'id="polls-from"' in page and 'id="polls-to"' in page
    assert 'Seleccionar todos' in page and 'Seleccionar todas' in page
    assert polls_tbody(page).count('<tr>') == min(15, FIXTURE_POLLS)
    assert 'Media simple' in page and 'class="p mx"' in page
    assert 'Descargar CSV' in page and 'Desliza para ver más' in page
    data = initial(body)
    assert 'table' not in data and {'date', 'pollster'} <= set(data['polls']['polls'][0])  # no sponsor in the fixture


def test_promedios_polls_table_carries_all_parties_counters_and_hides_the_pagination_up_to_fifteen(api):
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    assert '<summary>Partidos (<span data-count>1</span>/1)</summary>' in page
    assert '<summary>Encuestadoras (<span data-count>1</span>/1)</summary>' in page
    assert re.search(r'<input type="checkbox" name="party" value="PP" checked>', page)
    assert re.search(r'<input type="checkbox" name="pollster" value="Sigma Dos" checked>', page)
    assert '<option value="2026-10" selected>Oct 2026</option>' in page and 'id="polls-count">1 sondeo<' in page
    assert re.search(r'<div class="pag" id="polls-pag" hidden>', page)
    assert 'data-key="PP"' in page and 'title="Partido Popular"' in page
    assert '<th scope="col" data-key="date" aria-sort="descending"><button type="button" class="sort">Fecha</button></th>' in page
    assert ('<th scope="col" class="f l" data-key="pollster"><button type="button" class="sort">Encuestadora</button></th>'
            in page)
    assert 'th data-key' not in page and 'data-key="sample_size"' in page and '<th scope="col" data-key="sample_size"' in page
    assert '<b>Media simple</b><small>1 sondeo</small>' in page
    assert 'Cada fila es un sondeo' in page and 'no es el promedio del modelo' in page


def test_promedios_polls_table_renders_fifteen_of_many_polls_and_the_island_carries_them_all(api, fresh_app):
    polls = fixture_data('polls')
    base = polls['polls'][0]
    polls['polls'] = [base | {'date': '2026-08-{:02d}'.format(i % 28 + 1), 'sponsor': 'M{}'.format(i),
                              'start_date': '2026-08-01', 'end_date': '2026-08-02'} for i in range(20)]
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', polls, 'es', run_id=RUN)
    webapi.site().files.clear()
    body = call(api, '/es')[2]
    page = html(body).replace('\n', ' ')
    tbody = polls_tbody(page)
    assert tbody.count('<tr>') == 15 and '<small>M' in tbody
    assert 'title="Trabajo de campo: 1 ago 2026 – 2 ago 2026"' in tbody
    assert 'id="polls-count">20 sondeos<' in page and '<small>20 sondeos</small>' in page  # the mean covers all
    assert re.search(r'<div class="pag" id="polls-pag" hidden>', page)  # the JS shows it: there are more pages
    data = initial(body)['polls']['polls']
    assert len(data) == 20 and {'sponsor', 'start_date', 'end_date'} <= set(data[0])


def test_escanos_table_histograms_blocks_and_pills(api):
    page = html(call(api, '/es/escanos')[2])
    assert '<th scope="col">Rango 95%</th>' in page and 'Mín–máx' not in page and '<th scope="col">Media</th>' not in page
    # Una sola tarjeta: vista Partidos/Bloques, la barra, la tabla de la vista y un solo gráfico de crestas
    assert 'id="ridge-chart"' in page and 'data-party=' not in page and 'id="histograms"' not in page
    assert re.search(r'cada una con su\s+propio eje', page) and 'grid-small' not in page
    assert 'id="vs-table"' not in page and 'id="blocks-table"' in page and 'id="seats-bar"' in page
    flat = page.replace('\n', ' ')
    assert re.search(r'<fieldset id="seats-view" class="seg" hidden>', flat)
    assert 'name="sview" value="partidos" checked' in flat and 'name="sview" value="bloques"' in flat
    assert '<div id="seats-blocks" hidden>' in page and '<div id="seats-parties">' in page
    assert 'id="blocks-chart"' not in page and 'id="blocks"' not in page and 'id="seats"' in page
    assert '<fieldset class="pills lg" id="coalition-parties">' in page
    assert 'class="pill"' in page and 'name="coalition"' in page


def test_escanos_is_rendered_with_tables_links_and_initial(api):
    status, headers, body = call(api, '/es/escanos', query='mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Demoscopio · Escaños</title>' in page
    assert 'href="/es/escanos?mode=forecast" aria-current="page"' in page.replace('\n', ' ')
    assert 'id="parties-table"' in page and '<th scope="col">P(mayoría)</th>' in page
    assert 'id="blocks-table"' in page and 'id="vs-table"' not in page and 'id="seats-bar"' in page
    assert 'name="coalition" value="PP" checked' in page and 'id="coalition-result"' in page
    assert 'id="fan-chart"' in page and 'id="seats-evolution"' in page
    assert 'href="/api/v1/forecast/es/forecast/summary?run={}&amp;format=csv"'.format(RUN) in page
    assert 'href="/api/v1/forecast/es/fan?run={}&amp;format=csv"'.format(RUN) in page
    assert page.count('format=csv') == 5 and 'id="districts"' in page
    assert '/dist/js/pages/escanos.js' in page
    data = initial(body)
    assert list(data) == ['state', 'meta', 'summary', 'dist', 'fan', 'districts', 'scenario', 'runs']
    assert data['state'] == {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    assert data['dist']['parties'] == ['PP'] and data['summary']['majority'] == 176


def test_escanos_explains_the_fan_and_the_evolution_and_closes_with_the_method(api):
    page = html(call(api, '/es/escanos')[2])
    assert 'bloque con más probabilidad de mayoría absoluta' not in page  # sin titular desde 2026-10-14
    fan = page[page.index('id="fan"'):page.index('id="evolution"')]
    assert 'deriva' in fan and 'horizonte 0' in fan and 'no es una predicción de trayectoria' in fan
    evolution = page[page.index('id="evolution"'):page.index('id="downloads"')]
    assert 'run' in evolution and 'huecos' in evolution and 'no es una serie de sondeos' in evolution
    assert '¿Cómo simulamos los escaños?' in page and 'href="/elecciones?scope=es"' in page
    assert 'href="/casas?scope=es"' in page
    coalition = re.sub(r'\s+', ' ', page[page.index('id="coalition-result"'):page.index('id="districts"')])
    assert 'suben y bajan a la vez' not in coalition  # los partidos se sortean por separado
    assert ('La suma se hace en cada simulación: en cada una los escaños suman el total de la cámara, así que lo '
            'que gana un partido lo pierde otro') in coalition


def test_escanos_region_lives_in_the_district_form_not_in_the_event_bar(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/es/escanos', query='mode=forecast&region=8')[2])
    bar = page[page.index('id="event-bar"'):page.index('<main')]
    assert 'name="region"' not in bar and '<option value="8" selected>' in page
    form = page[page.index('id="district-form"'):]
    form = form[:form.index('</form>')]
    assert '<input type="hidden" name="mode" value="forecast">' in form
    # the largest district is the default selection
    assert '<option value="28" selected>' in html(call(api, '/es/escanos', query='mode=forecast')[2])


def test_escanos_nowcast_and_pinned_run(api):
    page = html(call(api, '/es/escanos', query='mode=nowcast&run=' + RUN)[2])
    assert '<span class="pill">estimación a ' in page and 'type="hidden" name="run" value="{}"'.format(RUN) in page
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
    assert '<td>TOTAL</td>' in page and '140' in page and '<th scope="col" data-key="total">Total</th>' in page
    page = html(call(api, '/es/escanos', query='mode=forecast&region=8')[2])
    assert '<option value="8" selected>Barcelona</option>' in page and 'Barcelona · 32 escaños' in page
    assert '<form id="district-form" method="get" action="/es/escanos#districts">' in page
    assert 'name="scope"' not in page  # el ámbito va en la ruta


def test_escanos_territory_table_has_hidden_controls_and_the_island_carries_the_districts(api, fresh_app):
    write_districts(fresh_app.fs)
    body = call(api, '/es/escanos', query='mode=forecast')[2]
    page = html(body).replace('\n', ' ')
    assert 'Resultados por territorio' in page and 'Circunscripciones</h2>' not in page
    assert re.search(r'<div class="series-controls" id="territory-controls" hidden>', page)
    assert 'name="tmode" value="esc" checked' in page and 'name="level"' not in page  # el fixture no trae comunidades
    assert '<table id="district-table" class="geo">' in page and 'data-key="total"' in page and 'TOTAL' in page
    assert 'Desliza para ver más' in page and 'id="district-form"' in page and 'id="region-table"' in page
    assert 'la media ponderada por los votos de la elección anterior' in re.sub(r'\s+', ' ', page)
    data = initial(body)
    assert {'districts', 'scenario'} <= set(data) and data['districts']['regions'][0]['id'] == 8
    with_communities = {**DISTRICTS, 'regions': [dict(r, community={'code': 13 if r['id'] == 28 else 9, 'name': 'X'}, votes=100) for r in DISTRICTS['regions']]}
    write_districts(fresh_app.fs, districts=with_communities)
    webapi.site().files.clear()
    page = html(call(api, '/es/escanos', query='mode=forecast')[2]).replace('\n', ' ')
    assert 'name="level" value="ccaa" checked' in page and 'name="level" value="prov"' in page


def test_escanos_region_detail_skips_parties_that_do_not_stand(api, fresh_app):
    absent = {'region_id': 28, 'region': 'Madrid', 'name': 'VOX', 'pct': None, 'pct_lo': None, 'pct_hi': None,
              'seats': 0.0, 'seats_mean': 0.0, 'seats_lo': 0.0, 'seats_hi': 0.0, 'p_seats': 0.0}
    write_districts(fresh_app.fs, districts=DISTRICTS | {'rows': DISTRICTS['rows'] + [absent]})
    page = html(call(api, '/es/escanos', query='mode=forecast&region=28')[2])
    detail = page[page.index('id="region-table"'):]
    detail = detail[:detail.index('</table>')]
    assert '– %' not in detail and detail.count('<tr>') == 3  # cabecera, PP y PSOE
    assert '34,0–44,0%' in detail


def test_escanos_district_form_keeps_the_pinned_run(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/es/escanos', query='mode=forecast&run=' + RUN)[2])
    assert page.count('type="hidden" name="run" value="{}"'.format(RUN)) == 1  # solo el formulario de circunscripción
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


def test_district_table_rows_are_alphabetical_with_the_accents_folded():
    regions = [{'id': i, 'name': name, 'seats': 1} for i, name in enumerate(['Zamora', 'Álava', 'Ávila', 'A Coruña'], 1)]
    table = pages.district_table({'parties': ['PP'], 'regions': regions, 'rows': []}, {'parties': ['PP'], 'rows': []})
    assert [r['name'] for r in table['rows']] == ['A Coruña', 'Álava', 'Ávila', 'Zamora']


def test_district_table_keeps_scenario_parties_ordered_by_total_and_has_communities_needs_votes():
    districts = DISTRICTS | {'parties': ['PP', 'PSOE', 'VOX']}
    scenario = {'parties': ['PP', 'PSOE', 'VOX'], 'rows': [
        {'region_id': 0, 'region': 'es', 'seats': [10, 30, 0]},
        {'region_id': 8, 'region': 'Barcelona', 'seats': [10, 0, 0]}, {'region_id': 28, 'region': 'Madrid', 'seats': [0, 30, 0]}]}
    table = pages.district_table(districts, scenario)
    assert table['parties'] == ['PSOE', 'PP']  # VOX sin escaños fuera, PSOE antes por su total
    assert [c['seats'] for c in table['rows'][0]['cells']] == [0, 10]
    regions = [dict(r, community={'code': r['id'], 'name': 'X'}, votes=10) for r in DISTRICTS['regions']]
    assert pages.has_communities(regions) is True
    assert pages.has_communities([dict(regions[0], votes=None), regions[1]]) is False
    assert pages.has_communities([{k: v for k, v in r.items() if k != 'votes'} for r in regions]) is False
    assert pages.has_communities([dict(r, community={'code': 1, 'name': 'X'}) for r in regions]) is False
    assert pages.has_communities([]) is False


def test_evolution_cards_explain_the_retrospective_points(api):
    for path in ('/es', '/es/escanos'):
        page = html(call(api, path)[2])
        assert 'Evolución de las publicaciones' in page and 'estimaciones retrospectivas' in page


def test_result_rows_and_block_result_rows_follow_the_summary_and_the_block_order():
    summary = {'parties': [{'name': 'PP', 'pct': 32.66, 'seats': 139, 'p_first': 0.95, 'p_majority': 0.0}],
               'blocks': [{'name': 'Izquierda', 'pct': 35.4, 'seats': 117, 'p_majority': 0.0},
                          {'name': 'Derecha', 'pct': 52.5, 'seats': 203, 'p_majority': 0.995},
                          {'name': 'Separatista', 'pct': None, 'seats': None, 'p_majority': None}]}
    assert pages.result_rows(summary) == [{'name': 'PP', 'pct': 32.66, 'seats': 139, 'seats_lo': None, 'seats_hi': None, 'p_first': 0.95}]
    rows = pages.block_result_rows(summary, fixture_data('meta'), pages.Catalog(fixture_data('meta')))
    assert [r['name'] for r in rows] == ['Derecha', 'Izquierda'] and rows[0]['p_majority'] == 0.995 and rows[0]['color'].startswith('#')
    assert pages.result_rows({}) == [] and pages.block_result_rows({}, fixture_data('meta'), pages.Catalog(fixture_data('meta'))) == []


def test_promedios_central_block_is_three_cards_with_ranges(api):
    page = html(call(api, '/es')[2])
    order = [page.index(m) for m in ('id="series"', 'id="hemicycle-card"', 'id="parties-card"', 'id="blocks-card"', 'id="evolution-card"')]
    assert order == sorted(order)
    heads = re.findall(r'<th scope="col">([^<]*)</th>', page[page.index('id="parties-table"'):page.index('id="blocks-result-table"')])
    assert heads == ['Partido', '% Voto', 'Escaños', 'Rango 95%', 'P(primero)']  # the bar column has a hidden head
    assert re.findall(r'<th scope="col">([^<]*)</th>', page[page.index('id="blocks-result-table"'):page.index('id="evolution-card"')]) == ['Bloque', '% Voto', 'Escaños', 'Rango 95%', 'P(mayoría)']
    for gone in ('id="vote-chart"', 'id="seats-chart"', 'id="majority"', 'id="headline-table"'):
        assert gone not in page


def test_promedios_orders_the_story_and_carries_the_controls(api):
    page = html(call(api, '/es')[2])
    order = [page.index(marker) for marker in ('id="series"', 'id="hemicycle-card"', 'id="parties-card"', 'id="blocks-card"',
                                              'id="evolution"', 'id="pollster"', 'id="polls"', 'id="closing"')]
    assert order == sorted(order)
    assert 'name="window" value="all" checked' in page.replace('\n', ' ') and 'value="3m"' in page and 'value="1a"' in page
    assert 'name="view" value="partidos"' in page and 'name="view" value="bloques"' in page
    assert 'id="pollster-select"' in page and 'href="/metodo' in page
    assert re.search(r'<div class="series-controls" id="pollster-controls" hidden>.*?<label>Encuestadora', page, re.S)
    data = initial(call(api, '/es')[2])
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs', 'cards', 'inks'}
    assert all(abs(v * 10 - round(v * 10)) < 1e-9 for v in data['series']['mean']['PP'] if v is not None)
    assert 'pollster_id' in data['polls']['polls'][0]


def test_promedios_controls_are_hidden_without_js(api):
    """Sin JS los controles no hacen nada y no se muestran."""
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    for control in ('series-window', 'series-view'):
        assert re.search(r'<fieldset id="{}" class="[^"]*" hidden>'.format(control), page), control
    before = page[:page.index('id="polls"')]
    assert 'data-all' not in before and 'data-none' not in before
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
    assert 'name="view"' not in page and 'name="pview"' not in page and 'id="series-window"' in page

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


# Casas: cuatro casas con ratings en `es` (CIS id 1, GAD3 id 2, una tercera con `rating` nulo y una quinta, Cero,
# con rating pero sin sondeos: ninguna de las dos entra en el ranking) en dos elecciones pasadas y la próxima,
# más una cuarta casa del catálogo (Nueva) sin ratings en el ámbito y un agregador evaluado (Agregador) sin
# ratings. GAD3 tiene el rating más alto en la próxima
# elección (2026-11-29); su primer rating es de 2023.
HOUSE_EVENTS = ['2019-11-10', '2023-07-23', '2026-11-29']


def fake_houses(monkeypatch, name='CIS'):
    """Lectores de casas parcheados en `webapi`; `name` es el nombre de la casa 1 (CIS por defecto)."""
    pollsters = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6], 'name': [name, 'GAD3', 'Hamalgama', 'Nueva', 'Cero', 'Agregador'],
        'mtype': ['public', 'private', 'private', 'private', 'online', 'aggr'],
        'quality': [56.0, 80.0, 40.0, 5.0, 5.0, 16.0], 'rating': [55, 70, None, None, 50, None],  # rating nacional truncado: distinto del camino del ámbito
    })
    ratings = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-11-10', '2023-07-23', '2023-07-23', '2026-11-29', '2026-11-29', '2026-11-29',
                                      '2026-11-29']),
        'pollster_id': [1, 1, 2, 1, 2, 3, 5], 'pollster': [name, name, 'GAD3', name, 'GAD3', 'Hamalgama', 'Cero'],
        'quality': [60.0, 60.0, 80.0, 60.0, 80.0, 40.0, 50.0], 'num_events': [1, 2, 1, 3, 2, 0, 0],
        'num_polls': [10, 12, 14, 15, 17, 2, 0], 'num_polls_w': [8.1, 9.5, 11.2, 12.0, 13.3, 1.1, 0.0],
        'error_avg': [1.9, 1.8, 1.32, 1.85, 1.3, None, None], 'error_blocks': [4.2, 4.0, 3.05, 4.1, -3.0, None, None],
        'bias_blocks': [2.5, 2.4, 1.52, 2.45, 1.5, None, None], 'rating_adj': [50.0, 52.0, 80.0, 53.0, 83.1, None, None],
        'bias_dev_adj': [0.5, 0.3, -0.9, 0.2, -1.0, None, None],
        'rating': [51.0, 53.0, 80.0, 55.0, 82.37, float('nan'), 50.0],
    })
    # los lectores de efectos y herding devuelven las filas de todos los ámbitos (`scope=None`, perfil global):
    # GAD3 tiene además una elección de Madrid, con un partido fuera del catálogo del run (MM)
    house_effects = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2023-07-23', '2023-07-23', '2026-11-29', '2021-05-04']),
        'event_scope': ['es', 'es', 'es', 'es', 'es-md'],
        'pollster_id': [2, 2, 1, 2, 2], 'pollster': ['GAD3', 'GAD3', name, 'GAD3', 'GAD3'], 'party': ['PP', 'PSOE', 'PP', 'PP', 'MM'],
        'dev_result_c': [0.9, -0.4, -2.1, float('nan'), -0.9], 'dev_cycle': [0.6, -0.2, -2.8, 0.7, -0.5],
    })
    herding = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2026-11-29', '2021-05-04']), 'event_scope': ['es', 'es', 'es-md'],
        'pollster_id': [2, 2, 2], 'pollster': ['GAD3', 'GAD3', 'GAD3'],
        'n': [14, 17, 9], 'ratio': [0.781, 0.705, 0.912], 'p_value': [0.034, 0.012, 0.21],
    })
    # el camino al rating: los ratings de la casa en todos los ámbitos (las próximas elecciones incluidas) y las
    # elecciones que sondeó; GAD3 sondeó Madrid 2021, Madrid y Aragón el 28-M de 2023 y las generales de 2023
    path_ratings = pd.DataFrame({
        'event_scope': ['es', 'es', 'es-md', 'es-md', 'es-ar', 'es-md', 'es', 'es'],
        'event_date': pd.to_datetime(['2023-07-23', '2026-11-29', '2021-05-04', '2023-05-28', '2023-05-28', '2027-05-23',
                                      '2019-11-10', '2023-07-23']),
        'pollster_id': [2, 2, 2, 2, 2, 2, 1, 1],
        'rating': [80.0, 82.37, 71.0, 78.5, 78.5, 82.37, 51.0, 53.0],
        'rating_adj': [80.0, 83.1, 72.3, 77.9, 77.9, 83.1, 50.0, 52.0],
    })
    path_events = pd.DataFrame({
        'event_scope': ['es', 'es-md', 'es-md', 'es-ar', 'es', 'es'],
        'event_date': pd.to_datetime(['2023-07-23', '2021-05-04', '2023-05-28', '2023-05-28', '2019-11-10', '2023-07-23']),
        'pollster_id': [2, 2, 2, 2, 1, 1],
        'n_polls': [14, 7, 3, 2, 10, 12],
        'error_avg': [1.3, 2.1, 1.8, 2.8, 1.9, 1.8], 'error_blocks': [1.5, -0.8, 0.6, -0.4, 2.5, 2.4],
        'bias_dev_adj': [-0.2, -1.5, -2.0, 0.5, 2.5, 3.6],
    })
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: pollsters)
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings)
    monkeypatch.setattr(webapi.queries, 'get_pollster_ratings',
                        lambda pollster_id: path_ratings[path_ratings.pollster_id == pollster_id].reset_index(drop=True))
    monkeypatch.setattr(webapi.queries, 'get_pollster_events',
                        lambda pollster_id: path_events[path_events.pollster_id == pollster_id].reset_index(drop=True))
    monkeypatch.setattr(webapi.queries, 'get_house_effects', lambda scope: house_effects)
    monkeypatch.setattr(webapi.queries, 'get_herding', lambda scope: herding)
    quality = pd.DataFrame({
        'pollster_id': [1, 3, 6], 'pollster': [name, 'Hamalgama', 'Agregador'],
        'fieldwork': [3, 2, 0],
        'fieldwork_source': ['https://www.cis.es/es/metodologia',
                             'https://hamalgama.example/ficha (ficha de 2026-10-01); https://hamalgama.example/metodologia', ''],
        'transparency': [3, 1, 1],
        'transparency_source': ['https://www.cis.es/es/microdatos', 'Ficha técnica del sondeo de 2026-10-01',
                                'https://agregador.example/metodo'],
        'independence': [0, 1, 0], 'independence_source': ['', 'base de datos del proyecto, consulta de patrocinadores a 2026-10-16', ''],
        'identity': [1, 1, 1], 'identity_source': ['https://www.cis.es', 'https://hamalgama.example', 'https://agregador.example'],
        'standards': [0, 0, 0], 'standards_source': ['', '', ''],
        'notes': ['Organismo público adscrito al Ministerio de la Presidencia', '', ''],
        'status': ['revisada', 'borrador', 'revisada'], 'checked_at': pd.to_datetime(['2026-10-16', '2026-10-16', '2026-10-01']),
        'points': [7, 5, 2], 'quality': [56.0, 40.0, 16.0],
    })
    monkeypatch.setattr(webapi.queries, 'get_pollsters_quality', lambda: quality)
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: HOUSE_EVENTS[-1])


@pytest.fixture
def houses_without_analysis(api, fresh_app, monkeypatch):
    """Páginas con base simulada y los lectores de casas parcheados, sin análisis publicado."""
    use_database(fresh_app)
    fake_houses(monkeypatch)
    return api


@pytest.fixture
def houses(houses_without_analysis, fresh_app):
    """Como `houses_without_analysis`, con el análisis de casas de `es` y su elección de 2023 publicados
    (fixtures): la diana del perfil solo cuenta las elecciones publicadas, como el histórico."""
    write_analysis(fresh_app.fs)
    write_event(fresh_app.fs)
    return houses_without_analysis


def test_casas_lists_the_ranking_effects_and_herding(houses):
    status, _, body = call(houses, '/casas', query='scope=es')
    page = html(body)
    assert status == 200 and 'aria-current="page"' in page and 'id="ranking-table"' in page
    assert page.index('GAD3') < page.index('CIS')  # orden por rating desc en el fixture
    assert 'href="/casas/gad3"' in page and 'id="scope-filter"' in page
    # el selector de ámbito vive en la tarjeta de efectos (el ranking es global), no sobre las tarjetas
    assert page.index('id="ranking"') < page.index('id="house-effects"') < page.index('id="scope-filter"') < page.index('id="herding"')
    assert 'id="effects-heatmap"' in page and 'id="herding-table"' in page
    # la tabla, al estilo de "Todas las encuestas", sin las barras del rating (redundantes desde el 2026-10-15)
    ranking = page[page.index('id="ranking"'):page.index('id="house-effects"')]
    assert '<table class="et et-rank" id="ranking-table">' in ranking and 'id="rating-chart"' not in page
    assert 'Cada fila es una encuestadora con rating para la próxima elección' in page
    assert 'class="headline"' not in page  # sin titular desde 2026-10-15
    assert not re.search(r'\ba el\b', page) and 'más margen a Derecha del que hubo' in page
    assert 'href="/api/v1/analysis/es/herding?format=csv"' in page and 'href="/api/v1/ratings?scope=es&amp;format=csv"' in page
    data = initial(body)
    assert list(data) == ['state', 'meta', 'grid']  # sin `ranking`: la tabla va pintada y ya no hay barras


def test_casas_ranking_drops_pollsters_without_polls_and_pages_by_fifty(houses, monkeypatch):
    """Cero tiene rating pero cero sondeos: no sale en el ranking; con más de 50 encuestadoras, la tabla esconde las
    que sobran tras "Cargar más" (el servidor las pinta ocultas)."""
    page = html(call(houses, '/casas')[2])
    ranking = re.search(r'<table[^>]*id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert 'Cero' not in ranking and ranking.count('<tr>') == 3 and 'id="ranking-more"' not in page
    assert '2 encuestadoras con rating y algún sondeo' in ranking and 'id="rating-chart"' not in page  # sin barras
    assert 'La tabla enseña las 50 primeras' in page
    many = pd.DataFrame({
        'event_date': pd.to_datetime(['2026-11-29'] * 60), 'pollster_id': list(range(1, 61)),
        'pollster': [f'P{i:02d}' for i in range(1, 61)], 'quality': [50.0] * 60, 'num_events': [1] * 60,
        'num_polls': [1] * 60, 'error_avg': [1.0] * 60, 'error_blocks': [0.5] * 60, 'bias_blocks': [0.5] * 60,
        'rating': [90.0 - i for i in range(60)],
    })
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: many)
    webapi.queries._QUERIES.clear()
    page = html(call(houses, '/casas')[2])
    ranking = re.search(r'<table[^>]*id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert ranking.count('<tr>') == 51 and ranking.count('<tr hidden>') == 10  # cabecera + 50 visibles, 10 ocultas
    assert ranking.index('P50') < ranking.index('<tr hidden>') < ranking.index('P51')
    assert '<button type="button" class="more" id="ranking-more" data-batch="50" hidden>Cargar más</button>' in page
    context = pages.casas_context(pages.parse_state({'scope': 'es'}))
    assert context['ranking_shown'] == 50 and context['ranking_hidden'] == 10 and pages.RANKING_SHOWN == 50
    assert [row['pollster'] for row in context['ranking']][:2] == ['P01', 'P02'] and 'ranking' not in context['initial']


def test_casas_ranking_is_the_national_one_whatever_the_scope(houses, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    write_analysis(fresh_app.fs, scope='es-md')
    webapi.site().manifest_cache_clear()
    national = html(call(houses, '/casas', query='scope=es')[2])
    madrid = html(call(houses, '/casas', query='scope=es-md')[2])
    table = lambda page: re.search(r'<table[^>]*id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert table(national) == table(madrid) and '82,4' in table(madrid)  # la columna Herding también es la nacional
    # la píldora del ranking lleva la fecha de la última actualización (la del run, como el pie), no la de la próxima
    # elección general ni la de hoy (Luis, 2026-10-15)
    assert 'no depende de ningún ámbito' in madrid and 'elección general del' not in madrid
    pill = re.search(r'<span class="pill">actualizado el (\d{1,2} [a-z]+ \d{4})</span>', madrid)
    assert pill and 'Actualizado: {} ({})'.format(RUN, pill.group(1)) in madrid  # la misma fecha que el pie
    assert 'href="/api/v1/ratings?scope=es&amp;format=csv"' in madrid and 'Ratings de la próxima elección general' in madrid
    ranking = table(madrid)
    assert 'href="/casas/gad3"' in ranking and 'scope=es-md&amp;id=' not in madrid  # el perfil es global: sin ámbito
    effects = madrid[madrid.index('id="house-effects"'):madrid.index('id="herding"')]
    assert 'href="/casas/' in effects and '<option value="es-md" selected>' in effects  # los efectos, del ámbito elegido


def test_casas_explains_the_relative_bias(houses):
    page = html(call(houses, '/casas', query='scope=es')[2]).replace('\n', ' ')
    page = re.sub(r'\s+', ' ', page)
    assert 'comparado con el de las demás encuestadoras' in page and 'no penaliza a una encuestadora' in page


def test_casas_without_analysis_keeps_the_ranking(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas')[2])
    assert 'id="ranking-table"' in page and 'Análisis no publicado todavía' in page
    assert 'id="effects-heatmap"' not in page and 'id="herding-table"' not in page


def test_casas_without_database_is_a_503_page(api):
    status, headers, body = call(api, '/casas')
    assert status == 503 and 'La base de datos no está disponible' in html(body) and headers['cache-control'] == 'no-store'


def test_casa_profile_shows_the_path_bullseye_and_tables(houses):
    page = html(call(houses, '/casas/gad3', query='scope=es')[2])
    assert '<h2>GAD3' in page and 'id="rating-path"' in page and 'class="chart big" id="bullseye-chart"' in page
    assert 'class="headline"' not in page  # sin titular desde 2026-10-15
    assert 'id="profile-effects"' in page and 'id="profile-herding"' in page and 'Todas las encuestadoras' in page
    data = initial(call(houses, '/casas/gad3', query='scope=es')[2])
    assert list(data) == ['state', 'meta', 'path', 'today', 'bullseye', 'blocks', 'scopes'] and data['path'][0]['event_date'] == '2021-05-04'


@pytest.mark.parametrize('query, status, text', [('id=abc', 400, 'Encuestadora no válida'),
                                                  ('id=999', 404, 'Encuestadora no encontrada')])
def test_casa_profile_errors(houses, query, status, text):
    got, headers, body = call(houses, '/casas', query=query)
    assert got == status and text in html(body) and headers['cache-control'] == 'no-store'


def test_casas_ranking_excludes_null_ratings_and_links_every_house(houses):
    page = html(call(houses, '/casas')[2])
    ranking = page[page.index('id="ranking-table"'):]
    ranking = ranking[:ranking.index('</table>')]
    assert 'Hamalgama' not in ranking and ranking.count('<tr>') == 3  # cabecera, GAD3 y CIS
    assert '82,4' in ranking and '0,74' in ranking  # rating (escala 0-100) y herding (análisis) de GAD3
    # los formatos (2026-10-17): barra verde del rating de 0 a 100, barra divergente del sesgo de bloques en ±8 en el color
    # del bloque al que favorece (la fixture solo trae el primero: el segundo, gris), rojos del error medio de 0 a 8 y
    # del rebaño (más rojo cuanto más por debajo de 1)
    from mtpy.lib.pages.filters import bar_sides, shade
    assert '<td class="f l"><a href="/casas/gad3"><b>GAD3</b></a></td>' in ranking
    assert '<td class="cell-bar"><i class="fill pos" style="left: 0; width: 82.4%"></i><span>82,4</span></td>' in ranking
    bias = bar_sides(-3.0, 8)
    assert '<td class="cell-bar cell-diverging"><i class="fill block" style="left: {}%; width: {}%; background: #9e9e9e"></i><span>-3,0</span></td>'.format(bias['left'], bias['width']) in ranking
    assert 'style="background-color: {}">1,9</td>'.format(shade(1.85, 0, 8, '#d62728')) in ranking  # CIS, error medio 1,85 de 8
    assert 'style="background-color: {}">1,3</td>'.format(shade(1.3, 0, 8, '#d62728')) in ranking  # GAD3
    assert 'style="background-color: rgba(214, 39, 40, 0.142)">0,74</td>' in ranking  # rebaño de GAD3
    assert 'style="background-color: transparent">–</td>' in ranking  # CIS no tiene rebaño en el análisis
    assert 'id="rating-chart"' not in page and 'id="effects-table"' in page
    assert 'href="/api/v1/analysis/es/herding?format=csv&amp;table=events"' in page
    assert 'href="/api/v1/analysis/es/house-effects?format=csv"' in page and 'href="/api/v1/analysis/es/poll-errors?format=csv"' in page
    data = initial(call(houses, '/casas')[2])
    assert [row['pollster_id'] for row in pages.casas_context(pages.parse_state({'scope': 'es'}))['ranking']] == [2, 1]
    assert data['grid']['columns'] == ['PP', 'PSOE', 'VOX', 'Derecha', 'Izquierda']
    assert [row['pollster'] for row in data['grid']['rows']] == ['CIS']  # GAD3 no tiene `cycle`


def test_casas_without_analysis_has_no_analysis_downloads(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas')[2])
    assert '/api/v1/analysis/' not in page and 'href="/api/v1/ratings?scope=es&amp;format=csv"' in page
    assert initial(call(houses_without_analysis, '/casas')[2])['grid'] is None


def test_casa_profile_tables_and_bullseye(houses):
    status, _, body = call(houses, '/casas/gad3', query='mode=nowcast')
    page = html(body)
    assert status == 200 and 'href="/casas">← Todas las encuestadoras' in page
    assert 'id="scope-filter"' not in page  # el perfil es global
    assert 'id="path-table"' in page
    bullseye = re.search(r'<table id="bullseye-table">(.*?)</table>', page, re.S).group(1)
    assert 'Generales 2023' in bullseye and '2023-23J' not in bullseye  # la elección, por su nombre
    effects = page[page.index('id="profile-effects"'):]
    effects = effects[:effects.index('</table>')]
    # la próxima elección no tiene resultado; Madrid 2021 no está publicada y va con fecha y ámbito
    assert effects.count('<tr>') == 3 and effects.index('Generales 2023') < effects.index('4 may 2021 · Madrid')  # de nueva a antigua
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', effects) == ['Elección', 'PP', 'PSOE', 'MM']  # el catálogo primero
    herding = page[page.index('id="profile-herding"'):]
    assert herding[:herding.index('</table>')].count('<tr>') == 4 and '29 nov 2026 · España' in herding
    data = initial(body)
    assert [row['event_date'] for row in data['path']] == ['2021-05-04', '2023-05-28', '2023-07-23']  # el gráfico, en orden
    assert list(data['path'][0]) == ['event_date', 'scopes', 'label', 'kind', 'n_polls', 'error_avg', 'error_blocks',
                                     'bias_dev_adj', 'rating', 'change']
    assert data['bullseye'] == [{'event': '2023-23J', 'event_date': '2023-07-23', 'year': 2023, 'days': 7, 'scope': 'es',
                                 'name': 'Generales 2023', 'Derecha': 2.81, 'Izquierda': -0.24}]
    assert data['blocks'] == ['Derecha', 'Izquierda'] and data['scopes'] == [{'code': 'es', 'name': 'España'}]
    # los filtros de la diana: ámbito (solo los que tienen sondeos suyos) y años, ocultos hasta que el JS los cablea
    form = re.search(r'<form id="bullseye-filter"[^>]*hidden>(.*?)</form>', page, re.S).group(1)
    assert re.findall(r'<option value="([^"]*)"', form) == ['', 'es', '2023', '2023'] and 'id="bullseye-count"' in form
    assert page.index('id="bullseye-filter"') < page.index('id="bullseye-chart"') < page.index('id="bullseye-table"')


def test_casas_ranking_and_profile_header_are_national_whatever_the_panel_scope(houses, fresh_app, monkeypatch):
    """Con el ámbito de los paneles en Madrid, el ranking (y su columna Herding) y el rating de la cabecera del
    perfil siguen siendo los nacionales; los efectos, el herding y el camino son los del ámbito."""
    write_bundle(fresh_app.fs, scope='es-md')
    write_analysis(fresh_app.fs, scope='es-md')
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    real_part = pages.casas.analysis_part

    def part(scope, name):
        data = real_part(scope, name)
        if scope == 'es-md' and name == 'herding' and data is not None:
            data = {**data, 'rows': [dict(row, herding=0.5) for row in data['rows']]}
        return data

    monkeypatch.setattr(pages.casas, 'analysis_part', part)
    page = html(call(houses, '/casas', query='scope=es-md')[2])
    ranking = re.search(r'<table[^>]*id="ranking-table">(.*?)</table>', page, re.S).group(1)
    herding = re.search(r'<table id="herding-table">(.*?)</table>', page, re.S).group(1)
    national = real_part('es', 'herding')['rows'][0]['herding']
    assert '0,50' not in ranking and pages.fmt_num(national, 2) in ranking and '0,50' in herding
    assert '<option value="es-md" selected>' in page and 'href="/casas/gad3"' in page
    page = html(call(houses, '/casas/gad3', query='scope=es-md')[2])
    assert '<div class="v">82,4<small></small></div>' in page.replace('\n', '') and 'id="scope-filter"' not in page and 'Generales 2023' in page  # perfil global


def test_casa_profile_of_a_house_without_ratings(houses):
    """Una casa del catálogo sin ratings en el ámbito: perfil con tablas vacías, sin camino ni diana."""
    status, _, body = call(houses, '/casas/nueva')
    page = html(body)
    assert status == 200 and '<h2>Nueva' in page and page.replace('\n', '').count('<div class="v">–<small></small></div>') == 1  # solo el rating sin valor
    assert '<div class="v">5,0<small></small></div>' in page.replace('\n', '') and 'sin evaluar' in page
    assert 'class="headline"' not in page  # el perfil no lleva titular (2026-10-15)
    assert 'id="rating-path"' not in page and 'id="bullseye-chart"' not in page and 'id="bullseye-filter"' not in page
    assert 'id="path-table"' in page and 'id="profile-effects"' in page and 'id="profile-herding"' in page
    data = initial(body)
    assert data['path'] == [] and data['today'] is None and data['bullseye'] == []


def test_casa_profile_without_analysis_has_no_bullseye(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas/gad3')[2])
    assert 'id="rating-path"' in page and 'id="bullseye-chart"' not in page and 'Análisis no publicado todavía' in page
    assert initial(call(houses_without_analysis, '/casas/gad3')[2])['blocks'] == []


def test_ranking_rows_and_herding_rows():
    frame = pd.DataFrame({'pollster_id': [1, 2, 3], 'pollster': ['CIS', 'GAD3', 'X'], 'rating': [55.0, 82.37, float('nan')],
                          'quality': [60.0, 80.0, 40.0], 'num_polls': [15, 17, 2], 'extra': [1, 2, 3]})
    rows = pages.ranking_rows(frame, {'rows': [{'pollster': 'GAD3', 'herding': 0.742}]})
    assert [row['pollster'] for row in rows] == ['GAD3', 'CIS'] and rows[0]['herding'] == 0.742 and rows[1]['herding'] is None
    assert list(rows[0]) == ['pollster_id', 'pollster', 'rating', 'quality', 'num_polls', 'num_events', 'error_avg',
                             'error_blocks', 'bias_blocks', 'bias_dev_adj', 'herding']
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
    page = html(call(houses, '/casas/gad3')[2])
    flat = page.replace('\n', '')
    assert '<div class="v">82,4<small></small></div>' in flat and '<div class="v">80,0<small></small></div>' in flat
    assert '<div class="l">Tipo</div><div class="v">privada<small></small></div>' in flat and '>70,' not in flat
    assert '<div class="v">pública<small></small></div>' in html(call(houses, '/casas/cis')[2]).replace('\n', '')  # `mtype` con su etiqueta en castellano


def test_casa_profile_quality_card_lists_the_rubric_with_sources(houses):
    page = html(call(houses, '/casas/cis')[2])
    kpis = page[page.index('id="profile"'):page.index('id="quality"')]
    assert '<div class="v">56,0<small></small></div>' in kpis.replace('\n', '') and '7 de 10 puntos' in kpis
    card = page[page.index('id="quality"'):page.index('id="path"')]
    rows = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>(\d) / (\d)</td>', card)
    assert [r[0] for r in rows] == ['Trabajo de campo', 'Transparencia', 'Independencia', 'Identidad', 'Estándares']
    assert [(r[1], r[2]) for r in rows] == [('3', '3'), ('3', '3'), ('0', '2'), ('1', '1'), ('0', '1')]
    assert 'Entrevistas propias con marco probabilístico' in card and 'href="https://www.cis.es/es/microdatos"' in card
    assert 'Comprobado el 16 oct 2026.' in card and 'Organismo público' in card and 'provisional' not in card
    assert 'cada punto vale 8' in card and 'mínimo de 5' in card  # el «Cómo leerlo» lee la escala del catálogo
    how = card[card.index('<details class="how"'):]
    assert '<a href="/metodo/rating#calidad-previa">La rúbrica</a> puntúa' in how  # enlace al método también si está evaluada
    assert '<dt>Confianza</dt>' not in page  # el glosario del perfil es el del camino al rating (2026-10-17)


def test_table_columns_describe_the_two_tables_with_the_scale_and_the_blocks():
    """Las columnas del ranking y del camino al rating, con su etiqueta y su descripción: la regla de puntos de
    «Confianza» sale de la escala del catálogo y se omite sin ella; el signo de «Sesgo bloques» nombra los bloques."""
    plain = {column['key']: column['text'] for column in pages.table_columns('ranking')}
    assert 'Cada punto vale' not in plain['quality'] and 'al primer bloque del que hubo; negativo, más margen al segundo' in plain['error_blocks']
    scaled = {column['key']: column['text']
              for column in pages.table_columns('ranking', ['Derecha', 'Izquierda'], {'per_point': 9, 'floor': 4})}
    assert scaled['quality'].endswith('Cada punto vale 9 de confianza, con un mínimo de 4.')
    assert 'más margen a Derecha del que hubo; negativo, más margen a Izquierda.' in scaled['error_blocks']
    assert [column['label'] for column in pages.table_columns('ranking')] == [
        'Encuestadora', 'Rating', 'Confianza', 'Elecciones', 'Sondeos', 'Diferencial', 'Error medio', 'Sesgo bloques', 'Rebaño']
    assert [column['label'] for column in pages.table_columns('path')] == [
        'Evento', 'Sondeos', 'Diferencial', 'Error medio', 'Sesgo bloques', 'Rating', 'Cambio']
    path = {column['key']: column['text'] for column in pages.table_columns('path', ['Derecha', 'Izquierda'])}
    assert 'de esa elección' in path['bias_dev_adj'] and 'de esa elección' in path['error_avg'] and 'quedó tras evaluar esa elección' in path['rating']


def test_casa_profile_quality_card_marks_drafts_and_plain_sources(houses):
    card = html(call(houses, '/casas/hamalgama')[2])
    card = card[card.index('id="quality"'):card.index('id="path"')]
    assert 'Evaluación provisional, pendiente de revisión' in card
    assert 'Ficha técnica del sondeo de 2026-10-01' in card and 'href="Ficha' not in card
    assert 'consulta de patrocinadores a 2026-10-16' in card


def test_source_parts_trims_the_punctuation_after_each_url():
    """Las URL de una fuente pierden el `)`, `;`, `,`, `.` o `:` que cierra la frase (vuelve al texto siguiente)."""
    from mtpy.lib.pages.casas import source_parts
    assert source_parts('') == [] and source_parts('base de datos') == [{'text': 'base de datos', 'url': None}]
    assert source_parts('(PDF https://x.es/a.pdf): CATI; https://y.es/b/.') == [
        {'text': '(PDF ', 'url': None}, {'text': 'https://x.es/a.pdf', 'url': 'https://x.es/a.pdf'},
        {'text': '): CATI; ', 'url': None}, {'text': 'https://y.es/b/', 'url': 'https://y.es/b/'}, {'text': '.', 'url': None}]
    assert source_parts('https://w.org/A_(b)') == [{'text': 'https://w.org/A_(b)', 'url': 'https://w.org/A_(b)'}]


def test_casa_profile_quality_card_splits_compound_sources_into_links(houses):
    """Una fuente con varias URL y texto entre ellas: cada URL es un enlace limpio y el texto queda como texto."""
    card = html(call(houses, '/casas/hamalgama')[2])
    card = card[card.index('id="quality"'):card.index('id="path"')]
    assert ('<a href="https://hamalgama.example/ficha" rel="noopener">https://hamalgama.example/ficha</a>'
            ' (ficha de 2026-10-01); '
            '<a href="https://hamalgama.example/metodologia" rel="noopener">https://hamalgama.example/metodologia</a>') in card
    assert 'href="https://hamalgama.example/ficha (' not in card and '<td>–</td>' in card  # Estándares, sin fuente


def test_casa_profile_quality_card_of_an_unrated_house(houses):
    page = html(call(houses, '/casas/nueva')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'Sin evaluar' in card and 'confianza mínima, 5' in card and 'href="/metodo/rating#calidad-previa"' in card
    assert '<th scope="row">' not in card and 'sin evaluar' in page[page.index('id="profile"'):page.index('id="quality"')]


def test_casa_profile_quality_card_of_an_informative_house(houses):
    page = html(call(houses, '/casas/cero')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'no entran en el promedio' in card and 'Sin evaluar' in card


def test_casa_profile_quality_card_of_an_evaluated_informative_house(houses):
    """Un agregador evaluado: la tabla de la rúbrica y la frase de que su calidad es informativa, juntas."""
    page = html(call(houses, '/casas/agregador')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    rows = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>(\d) / (\d)</td>', card)
    assert [(r[0], r[1]) for r in rows] == [('Trabajo de campo', '0'), ('Transparencia', '1'), ('Independencia', '0'),
                                            ('Identidad', '1'), ('Estándares', '0')]
    assert 'href="https://agregador.example/metodo"' in card and '2 de 10 puntos' in card
    assert 'la confianza es informativa' in card and 'Sin evaluar' not in card


def test_pages_package_layout():
    """Un módulo por página, `common` y `filters`; ninguno tapado por un nombre exportado (D1)."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(pages.__path__))
    assert names == ['backtest', 'casas', 'common', 'elecciones', 'escanos', 'filters', 'inicio', 'metodo', 'promedios', 'slugs']
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
    assert status == 200 and page.index('generales/2023-07') < page.index('generales/2019-11')
    assert 'href="/elecciones/es/generales/2023-07"' in page and data_href(body, 'es:2023-07-23') == '/elecciones/es/generales/2023-07'
    data = initial(body)
    assert 'sector' not in data and 'blocks' not in data
    assert [row['date'] for row in data['events']] == ['2023-07-23', '2019-11-10']
    assert 'id="sector-chart"' not in page and 'id="sector"' not in page
    table = re.search(r'<table id="events-table">(.*?)</table>', page, re.S).group(1)
    assert [re.sub(r'<.*?>', '', th) for th in re.findall(r'<th scope="col">(.*?)</th>', table)] == [
        'Elección', 'Sondeos', 'Encuestadoras', 'Error medio', 'Error de bloques', 'Sesgo bloques', 'Modelo']


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
    assert 'elecciones analizadas' not in page and 'id="errors-filter"' in page and 'id="errors-histogram"' in page
    assert 'id="errors-bullseye"' in page and 'id="weeks-table"' in page and 'Sesgo de bloques' not in page and 'Desviación hacia' not in page
    assert 'id="glosario"' in page and '¿Cómo calculamos el rating de las encuestadoras?' in page


def test_elecciones_without_analysis_keeps_the_list(events_site):
    page = html(call(events_site, '/elecciones')[2])
    assert 'Análisis no publicado todavía' in page and 'id="events-table"' in page and 'class="headline"' not in page


def test_eleccion_detail_has_the_error_histogram_and_the_glossary(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert 'id="poll-histogram"' in page and 'id="glosario"' in page and 'Sesgo de bloques' not in page
    assert initial(body)['poll_errors'][0].keys() >= {'error_avg', 'error_blocks_abs'}


def test_elecciones_history_context_filters(history_site):
    context = pages.elecciones_context(pages.parse_state({'scope': 'es'}), today=date(2026, 10, 15))
    assert not context['analysis_missing'] and len(context['errors']) == 2
    assert context['filter_events'] == [
        {'key': 'es:2023-07-23', 'date': '2023-07-23', 'name': 'Generales 2023', 'scope': 'es', 'scope_name': 'España',
         'href': '/elecciones/es/generales/2023-07'},
        {'key': 'es:2019-11-10', 'date': '2019-11-10', 'name': 'Generales 2023', 'scope': 'es', 'scope_name': 'España',
         'href': '/elecciones/es/generales/2019-11'}]
    assert context['years'] == [2023, 2024, 2025, 2026] and context['selected_scope'] == 'es'  # every year up to today
    assert context['filter_pollsters'] == ['GAD3', 'Sigma Dos']
    assert 'headline_text' not in context  # sin titular desde 2026-10-15
    assert all(row['error_blocks_abs'] == abs(row['error_blocks']) for row in context['errors'])
    assert set(context['initial']) == {'state', 'scopes', 'events', 'errors', 'error_blocks', 'years', 'selected_scope'}
    assert pages.error_rows('es', ['2019-11-10']) == [] and pages.error_rows('es-md', ['2023-07-23']) is None
    page = html(call(history_site, '/elecciones')[2])
    assert page.index('id="filters"') < page.index('id="figures"') < page.index('id="trend"') < page.index('id="histogram"') \
        < page.index('id="bullseye"') < page.index('id="weeks"') < page.index('id="events"') < page.index('id="glosario"') \
        < page.index('id="closing"')
    assert 'id="complements"' in page and 'id="trend-chart"' in page and 'id="sector-chart"' not in page
    assert 'class="headline"' not in page
    assert 'id="weeks-note"' in page and 'Elige un ámbito o unos años para ver las elecciones una a una.' in page
    assert '<option value="es:2019-11-10" data-scope="es">' in page and '<option value="Sigma Dos">' in page
    assert 'positivo, el sondeo daba más margen a Derecha del que hubo' in page
    assert 'Herding' not in page and 'Rebaño' not in page and 'La tabla necesita JavaScript' in page
    assert page.count('Análisis no publicado todavía') == 0


@pytest.fixture
def history_two_scopes(history_site, fresh_app):
    """`history_site` más Madrid: una elección (2021-05-04) con su análisis y un run publicado."""
    write_bundle(fresh_app.fs, scope='es-md', manifest=True)
    write_event(fresh_app.fs, scope='es-md', date='2021-05-04', model=False)
    write_analysis(fresh_app.fs, scope='es-md')
    errors = fixture_data('poll-errors')
    errors['rows'] = [dict(row, event_date='2021-05-04', event='2021-04M') for row in errors['rows']]
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es-md', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es-md')
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    return history_site


def test_elecciones_merges_every_scope_with_analysis(history_two_scopes):
    body = call(history_two_scopes, '/elecciones')[2]
    page = html(body).replace('\n', ' ')
    assert 'id="scope-filter"' not in page and 'Ámbito</th>' not in page  # el nombre de la elección basta
    assert '· Generales 2023</option>' in page and 'Generales 2023 · España' not in page and 'Madrid</option>' in page
    assert re.search(r'<select name="ambito">\s*<option value="" selected>Todos</option>', page)
    for name in ('eleccion', 'casa', 'desde', 'hasta'):
        assert 'name="{}"'.format(name) in page
    assert 'id="errors-measure"' in page and page.index('id="errors-measure"') < page.index('id="histogram"')
    assert re.search(r'<tr data-scope="es-md" data-date="2021-05-04" data-year="2021">', page)
    assert '<p class="muted" id="events-count">3 elecciones</p>' in page and 'href="/casas"' in page
    data = initial(body)
    assert [s['code'] for s in data['scopes']] == ['es', 'es-md'] and data['selected_scope'] is None
    assert {row['scope'] for row in data['errors']} == {'es', 'es-md'} and all(isinstance(row['year'], int) for row in data['errors'])
    assert set(data['errors'][0]) == set(pages.HISTORY_COLUMNS) | set(data['error_blocks'])
    assert [e['scope'] for e in data['events']][:3] == ['es', 'es-md', 'es'] and data['years'] == sorted(data['years'])
    assert data['years'] == list(range(2021, date.today().year + 1))  # desde la primera elección hasta hoy


def test_elecciones_marks_scopes_without_a_run_as_not_linkable(history_two_scopes, fresh_app):
    write_event(fresh_app.fs, scope='es-an', date='2022-06-19', model=False)  # elecciones y análisis, sin run
    write_analysis(fresh_app.fs, scope='es-an')
    errors = fixture_data('poll-errors')
    errors['rows'] = [dict(row, event_date='2022-06-19', event='2022-19J') for row in errors['rows']]
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es-an', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es-an')
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    scopes = pages.history_scopes()
    assert {s['code']: s['linkable'] for s in scopes} == {'es': True, 'es-an': False, 'es-md': True}
    hrefs = {(e['scope'], e['date']): e['href'] for e in pages.history_events(scopes)}
    assert hrefs[('es-an', '2022-06-19')] is None and hrefs[('es-md', '2021-05-04')] == '/elecciones/es/madrid/2021-05'
    body = call(history_two_scopes, '/elecciones')[2]
    row = re.search(r'<tr data-scope="es-an"[^>]*>\s*<td>(.*?)</td>', html(body), re.S).group(1)
    assert '<a' not in row and '2022' in row
    assert {s['code']: s['linkable'] for s in initial(body)['scopes']}['es-an'] is False


def test_error_blocks_come_from_the_first_scope_with_two_and_parts_are_read_once(history_two_scopes, fresh_app):
    errors = fixture_data('poll-errors')
    errors['blocks'] = []
    errors['rows'] = [dict(row, event_date='2021-05-04', event='2021-04M') for row in errors['rows']]
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es-md', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es-md')
    webapi.site().files.clear()
    assert initial(call(history_two_scopes, '/elecciones')[2])['error_blocks'] == ['Derecha', 'Izquierda']
    scopes = pages.history_scopes()
    assert pages.error_blocks(scopes, pages.poll_error_parts(scopes)) == ['Derecha', 'Izquierda']
    assert pages.error_blocks(scopes, {}) == []


def test_elecciones_scope_query_preselects_the_scope_or_falls_back(history_two_scopes, fresh_app):
    page = html(call(history_two_scopes, '/elecciones', query='scope=es-md')[2]).replace('\n', ' ')
    assert re.search(r'<option value="es-md" selected>Madrid</option>', page)
    assert initial(call(history_two_scopes, '/elecciones', query='scope=es-md')[2])['selected_scope'] == 'es-md'
    assert call(history_two_scopes, '/elecciones', query='scope=es-an')[0] == 404  # sin run publicado
    write_bundle(fresh_app.fs, scope='es-ar', manifest=True)  # con run y elecciones, pero sin análisis
    write_event(fresh_app.fs, scope='es-ar', date='2019-04-28', model=False)
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    body = call(history_two_scopes, '/elecciones', query='scope=es-ar')[2]
    assert initial(body)['selected_scope'] is None
    assert re.search(r'<option value="" selected>Todos</option>', html(body).replace('\n', ' '))


def test_history_helpers_skip_scopes_without_events_or_analysis(history_two_scopes, fresh_app):
    write_analysis(fresh_app.fs, scope='es-cb')  # análisis sin elecciones
    write_event(fresh_app.fs, scope='es-ar', date='2019-04-28', model=False)  # elecciones sin análisis
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    scopes = pages.history_scopes()
    assert [(s['code'], s['analysis']) for s in scopes] == [('es', True), ('es-ar', False), ('es-md', True)]
    events = pages.history_events(scopes)
    assert [e['scope'] for e in events] == ['es', 'es-md', 'es', 'es-ar']
    rows = pages.history_rows(scopes, events)
    assert {row['scope'] for row in rows} == {'es', 'es-md'}


def test_elecciones_without_events_shows_a_notice(api):
    status, _, body = call(api, '/elecciones', query='scope=es')
    assert status == 200 and 'Todavía no hay elecciones publicadas' in html(body) and 'initial-data' not in html(body)
    assert '/dist/js/pages/elecciones.js' in html(body)  # el módulo se carga y debe tolerar la falta del bloque


def test_eleccion_detail_renders_results_last_polls_bullseye_and_model(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and 'Generales 2023' in page
    assert page.count('/api/v1/events/es/2023-07-23/') == 7 + 2  # las descargas y el CSV de la tabla (control y noscript)
    table = re.search(r'<table id="last-table">(.*?)</table>', page, re.S).group(1)
    assert table.index('GAD3') < table.index('40dB') < table.index('CIS')  # de menor a mayor error de bloques
    assert [re.sub(r'<.*?>', '', th) for th in re.findall(r'<th scope="col">(.*?)</th>', table)] == [
        'Encuestadora', 'Muestra', 'Días antes', 'Error medio', 'Error de bloques', 'Derecha', 'Izquierda']
    assert 'href="/casas/gad3"' in table and 'href="/elecciones?scope=es"' in page
    assert 'id="pollsters-table"' not in page and 'Sesgo bloques' not in table  # ya no hay tabla de medias por casa
    assert 'Cs' in page and pages.OTHERS_COLOR in page  # partido fuera del catálogo del run
    # la tabla enseña los partidos de `parties.event` de data/params.json (PP, PSOE, VOX); Cs queda tras "Ver más"
    results = re.search(r'<table id="results-table">(.*?)</table>', page, re.S).group(1)
    assert results.count('<tr>') == 4 and results.count('<tr hidden>') == 1 and results.index('VOX') < results.index('<tr hidden>') < results.index('Cs')
    assert '<button type="button" class="more" id="results-more" hidden>Ver el partido restante</button>' in page
    # el selector Partidos/Bloques del hemiciclo y la tabla por bloques (oculta): Derecha = PP + VOX + Cs según los
    # parámetros, Izquierda = PSOE
    assert '<fieldset id="results-view" class="seg" hidden>' in page and 'name="rview"' in page
    blocks = re.search(r'<section class="card" id="results-blocks" hidden>(.*?)</section>', page, re.S).group(1)
    assert re.findall(r'<th scope="col">(.*?)</th>', blocks) == ['Bloque', '% Voto', 'Escaños']
    assert re.findall(r'</span>(\w+)</td>\s*<td>(.*?)</td>\s*<td><strong>(\d+)</strong>', blocks) == [
        ('Derecha', '45,5%', '200'), ('Izquierda', '31,4%', '150')]
    data = initial(body)
    assert set(data) == {'state', 'meta', 'results', 'last', 'blocks', 'split', 'block_colors', 'poll_errors', 'catalog', 'polls'}
    assert [row['pollster'] for row in data['last']] == ['GAD3', '40dB', 'CIS']
    assert data['last'][0] == {'pollster': 'GAD3', 'pollster_id': 2, 'days': 7, 'sample_size': 3000, 'error_avg': 0.99,
                               'error_blocks': 1.36, 'error_blocks_abs': 1.36, 'Derecha': 0.68, 'Izquierda': -0.68}
    # el hemiciclo del resultado oficial y el reparto del error por bloque (Izquierda a la izquierda)
    assert data['results'] == {'n_seats': 350, 'majority': 176, 'rows': [
        {'name': 'PP', 'seats': 160}, {'name': 'PSOE', 'seats': 150}, {'name': 'VOX', 'seats': 40}, {'name': 'Cs', 'seats': 0}],
        'blocks': [{'name': 'Derecha', 'seats': 200}, {'name': 'Izquierda', 'seats': 150}]}
    assert data['split'] == {'left': 'Izquierda', 'right': 'Derecha'}
    assert data['block_colors'] == {'Derecha': fixture_data('meta')['parties'][0]['color'], 'Izquierda': pages.OTHERS_COLOR}
    # primera fila como en Promedios: hemiciclo y tabla de resultados (% Voto y Escaños, sin P(primero))
    results = re.search(r'<table id="results-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col">(.*?)</th>', results) == ['Partido', '% Voto', '<span class="sr-only">Barra del voto</span>', 'Escaños']
    assert '<td>33,1%</td>' in results and '<td><strong>160</strong></td>' in results and 'P(primero)' not in page
    assert 'id="hemicycle"' in page and 'de los 350 de la cámara' in page
    assert 'id="error-chart"' in page and 'id="split-chart"' in page and 'class="chart big" id="bullseye-chart"' in page
    assert 'cuánto falló cada sondeo en Izquierda y a la derecha, en Derecha' in page
    assert page.index('id="event"') < page.index('class="hemi-grid"') < page.index('id="hemicycle-card"') < page.index('id="results"') \
        < page.index('class="split"') < page.index('id="last-errors"') < page.index('id="last-split"') < page.index('id="bullseye"') \
        < page.index('id="last"') < page.index('id="glosario"') < page.index('id="poll-histogram"') < page.index('id="polls"') \
        < page.index('id="model"') < page.index('id="downloads"')


def test_eleccion_detail_has_the_polls_table_of_the_cycle(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body).replace('\n', ' ')
    assert status == 200 and 'Todas las encuestas' in page and 'id="polls-controls" hidden' in page
    assert 'id="polls-mode"' in page and 'Escaños</label>' in page  # el fixture trae escaños
    assert polls_tbody(page).count('<tr>') == 3 and '16/07/23' in page and 'El País' in page
    assert '<span id="polls-count">3 sondeos</span>' in page and 'href="/api/v1/events/es/2023-07-23/cycle?format=csv"' in page
    assert 'Sondeos del ciclo, tal como se publicaron' in page and 'no están publicados todavía' not in page
    data = initial(body)
    assert data['polls']['parties'] == ['PP', 'PSOE', 'VOX'] and data['polls']['polls'][2]['seats'] == {'PP': [135, 140], 'PSOE': [120, 120]}
    # el catálogo del navegador: el del run primero (PP), luego los de la tabla y los de los resultados (Cs)
    assert [party['name'] for party in data['catalog']['parties']] == ['PP', 'PSOE', 'VOX', 'Cs']
    assert data['catalog']['parties'][0]['color'] != pages.OTHERS_COLOR and data['catalog']['parties'][2]['color'] == pages.OTHERS_COLOR
    assert data['catalog']['parties'][0]['block'] == 'Derecha' and data['catalog']['parties'][3]['block'] is None
    context = pages.eleccion_context(pages.parse_state({'scope': 'es'}), '2023-07-23')
    assert context['has_cycle'] and context['table_total'] == 3 and context['has_seats'] and context['months'][0]['value'] == '2023-07'


def test_eleccion_detail_without_cycle_part_shows_a_notice(events_site, monkeypatch):
    real = pages.elecciones.event_part
    monkeypatch.setattr(pages.elecciones, 'event_part', lambda scope, date, name: None if name == 'cycle' else real(scope, date, name))
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and 'Los sondeos de este ciclo no están publicados todavía' in page
    assert 'id="polls-table"' not in page and 'id="polls-controls"' not in page
    data = initial(body)
    assert 'polls' not in data and [party['name'] for party in data['catalog']['parties']] == ['PP', 'PSOE', 'VOX', 'Cs']


def test_eleccion_detail_hides_the_results_beyond_fifteen_without_params(events_site, monkeypatch):
    """Sin lista de partidos en los parámetros (o sin ninguno de sus partidos en ella), la tabla enseña los 15
    primeros; sin bloques en ninguna fuente, ni selector ni tabla por bloques."""
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'results':
            rows = [dict(data['rows'][0], party=f'P{i:02d}', seats=1) for i in range(23)]
            data = {**data, 'rows': rows}
        if name == 'meta':
            data = {**data, 'blocks': {}}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    for params in ({}, {'parties': {'event': ['PP', 'PSOE']}}):  # sin lista, o una lista sin ninguno de los partidos
        monkeypatch.setattr(pages.elecciones, 'event_params', lambda scope, date, params=params: params)
        status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
        page = html(body)
        table = re.search(r'<table id="results-table">(.*?)</table>', page, re.S).group(1)
        assert status == 200 and table.count('<tr>') == 16 and table.count('<tr hidden>') == 8  # cabecera + 15 visibles
        assert table.index('P14') < table.index('<tr hidden>') < table.index('P15')
        assert '<button type="button" class="more" id="results-more" hidden>Ver los 8 partidos restantes</button>' in page
        assert 'id="results-view"' not in page and 'id="results-blocks"' not in page
        assert initial(body)['results']['blocks'] == []
    context = pages.eleccion_context(pages.parse_state({'scope': 'es'}), '2023-07-23')
    assert context['results_hidden'] == 8 and context['block_result_rows'] == [] and pages.RESULTS_SHOWN == 15


def test_eleccion_detail_helpers_of_the_shown_parties_and_the_blocks():
    assert pages.shown_parties({'parties': {'event': ['PP', 'PSOE'], 'polls': ['PP']}}) == ['PP', 'PSOE']
    assert pages.shown_parties({}) is None and pages.shown_parties({'parties': {}}) is None and pages.shown_parties(None) is None
    rows = [{'party': 'PP', 'pct': 33.06, 'seats': 160}, {'party': 'PSOE', 'pct': 31.44, 'seats': 150},
            {'party': 'Cs', 'pct': 0.13, 'seats': 0}, {'party': 'ERC', 'pct': 1.9, 'seats': 7}]
    assert [row['shown'] for row in pages.mark_shown(rows, ['PSOE', 'ERC', 'UP'])] == [False, True, False, True]
    assert [row['shown'] for row in pages.mark_shown(rows, None, limit=2)] == [True, True, False, False]
    assert [row['shown'] for row in pages.mark_shown(rows, ['UP'], limit=3)] == [True, True, True, False]  # ninguno en la lista
    assert pages.mark_shown([], ['PP']) == []
    catalog = pages.Catalog({'PP': {'block': 'Derecha'}, 'ERC': {'block': 'Separatista'}})
    params = {'bmaps': {'blocks': {'Derecha': ['PP', 'Cs'], 'Izquierda': ['PSOE']}}}
    meta_blocks = {'Derecha': ['PP'], 'Izquierda': ['PSOE', 'UP']}
    names = ['PP', 'PSOE', 'Cs', 'ERC', 'UP', 'MM']
    # los parámetros primero, luego el meta de la elección, luego el catálogo del run; sin fuente, nada
    assert pages.party_blocks(names, params, meta_blocks, catalog) == {
        'PP': 'Derecha', 'PSOE': 'Izquierda', 'Cs': 'Derecha', 'ERC': 'Separatista', 'UP': 'Izquierda', 'MM': None}
    assert pages.party_blocks(names, {}, None, catalog) == {'PP': 'Derecha', 'PSOE': None, 'Cs': None, 'ERC': 'Separatista', 'UP': None, 'MM': None}
    blocks = pages.party_blocks(names, params, meta_blocks, catalog)
    assert pages.event_block_rows(rows, blocks, lambda block: block[0]) == [
        {'name': 'Derecha', 'pct': 33.19, 'seats': 160, 'color': 'D'}, {'name': 'Izquierda', 'pct': 31.44, 'seats': 150, 'color': 'I'},
        {'name': 'Separatista', 'pct': 1.9, 'seats': 7, 'color': 'S'}]
    assert pages.event_block_rows([{'party': 'X', 'pct': None, 'seats': None}], {'X': 'Otro'}, str) == [
        {'name': 'Otro', 'pct': None, 'seats': None, 'color': 'Otro'}]
    assert pages.event_block_rows(rows, {}, str) == []


def test_event_params_reads_the_data_file_through_a_cache(fresh_app, tmp_path):
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    webapi.checks._PARAMS.clear()
    params = webapi.event_params('es', '2023-07-23')
    assert params['parties']['event'][:3] == ['PP', 'PSOE', 'VOX'] and 'Cs' in params['bmaps']['blocks']['Derecha']
    assert webapi.event_params('es', '1900-01-01') == {} and webapi.event_params('xx', '2023-07-23') == {}
    # sin fichero, o sin directorio de datos, no hay parámetros
    fresh_app.set('data', FileSystem(str(tmp_path)))
    webapi.checks._PARAMS.clear()
    assert webapi.event_params('es', '2023-07-23') == {}
    fresh_app.set('data', None)
    assert webapi.event_params('es', '2023-07-23') == {}


def test_last_poll_rows_keep_the_closest_poll_of_each_house():
    rows = pages.with_abs([
        {'pollster': 'A', 'days': 3, 'error_blocks': -2.0}, {'pollster': 'B', 'days': 5, 'error_blocks': None},
        {'pollster': 'A', 'days': 9, 'error_blocks': 0.5}, {'pollster': None, 'days': 1, 'error_blocks': 0.1},
        {'pollster': 'C', 'days': 4, 'error_blocks': 1.0}, {'pollster': 'C', 'days': 4, 'error_blocks': 0.2},
    ])
    last = pages.last_poll_rows(rows)
    assert [(row['pollster'], row['days'], row['error_blocks_abs']) for row in last] == [('C', 4, 1.0), ('A', 3, 2.0), ('B', 5, None)]
    assert pages.last_poll_rows([]) == []


def test_eleccion_without_model_shows_a_notice(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2019-11')
    page = html(body)
    assert status == 200 and 'Sin backtest para esta elección' in page
    assert page.count('/api/v1/events/es/2019-11-10/') == 4 + 2


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
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and page.count('Escaños no proyectables en esta elección') == 1
    assert 'id="model-shares-30"' in page and 'id="model-shares-6"' in page
    assert 'id="model-seats-6"' in page and 'id="model-blocks-6"' in page
    assert 'id="model-seats-30"' not in page and 'id="model-blocks-30"' not in page


def test_eleccion_detail_orders_houses_by_the_absolute_block_error_of_their_last_poll(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'polls':
            rows = [dict(row, error_blocks=error) for row, error in zip(data['rows'], (-3.0, 1.0, 2.0))]
            data = {**data, 'rows': rows}
        if name == 'meta':
            data = {**data, 'summary': {**data['summary'], 'error_blocks': -2.5}}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    names = [row['pollster'] for row in real('es', '2023-07-23', 'polls')['rows']]
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    table = re.search(r'<table id="last-table">(.*?)</table>', page, re.S).group(1)
    assert status == 200 and '3,0' in table and '-3,0' not in table  # el error de bloques de la tabla va sin signo
    assert 'Error de bloques: 2,5 · Sesgo bloques: -2,5' in page
    shown = [re.sub(r'<.*?>', '', cell) for cell in re.findall(r'<tr>\s*<td>(.*?)</td>', table, re.S)]
    assert shown == [names[1], names[2], names[0]]
    data = initial(body)
    assert [row['error_blocks_abs'] for row in data['last']] == [1.0, 2.0, 3.0]
    assert [row['error_blocks'] for row in data['last']] == [1.0, 2.0, -3.0]


def test_last_rows_without_error_go_last_and_have_no_absolute_value(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'polls':
            errors = (None, -2.0, 1.0)
            data = {**data, 'rows': [dict(row, error_blocks=error) for row, error in zip(data['rows'], errors)]}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    context = pages.eleccion_context(pages.parse_state({'scope': 'es'}), '2023-07-23')
    assert [row['error_blocks_abs'] for row in context['last_rows']] == [1.0, 2.0, None]
    assert all('error_blocks_abs' in row for row in context['poll_rows'])


def test_last_row_without_id_is_plain_text(events_site, monkeypatch):
    real = pages.elecciones.event_part

    def part(scope, date, name):
        data = real(scope, date, name)
        if name == 'polls':
            data = {**data, 'rows': [dict(row, pollster_id=None) for row in data['rows']]}
        return data

    monkeypatch.setattr(pages.elecciones, 'event_part', part)
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    table = re.search(r'<table id="last-table">(.*?)</table>', html(body), re.S).group(1)
    assert status == 200 and 'GAD3' in table and '<a ' not in table


def test_eleccion_without_meta_is_a_404(events_site, fresh_app, monkeypatch):
    real = pages.elecciones.event_part
    monkeypatch.setattr(pages.elecciones, 'event_part',
                        lambda scope, date, name: None if name == 'meta' and date == '2023-07-23' else real(scope, date, name))
    st, headers, body = call(events_site, '/elecciones/es/generales/2023-07')
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
    assert '/dist/js/pages/metodo.js' in page and '<title>Demoscopio · Cuánto acierta el modelo</title>' in page
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
    assert [section_id for section_id, _ in sections] == ['intro', 'datos', 'publicacion', 'limitaciones']
    assert all('<p>' in body for _, body in sections)  # ninguna sección sin texto en la plantilla
    for section_id, title in pages.SECTIONS:
        assert 'id="{}"'.format(section_id) in page and title in page
    assert '10.000 elecciones' in page and 'contiene el' not in page  # sin backtest, sin frase de cobertura
    # el resumen antiguo se retiró al escribirse los textos de las cuatro preguntas
    assert 'id="como-funciona"' not in page and 'id="casas"' not in page and 'id="evaluacion"' not in page
    assert 'id="simulacion"' not in page and 'id="escanos"' not in page


def test_metodo_landing_quotes_the_coverage_of_the_general_backtest(api, fresh_app):
    write_backtest(fresh_app.fs)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'contiene el resultado el 96% de las veces a 6 días de la votación y el 100% a 30 días' in page
    assert 'En las generales: 2 elecciones, a 6 y 30 días de la votación.' in page
    # a 30 días el fixture cubre el 100%: sin la coletilla de los intervalos demasiado estrechos
    assert 'demasiado estrechos' not in page


def test_metodo_landing_says_the_intervals_fall_short_only_below_95(api, fresh_app, monkeypatch):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'demasiado estrechos' not in page  # sin backtest, nada que leer en la cobertura
    figures = {'n_sim': 10000, 'events': [], 'horizons': [6, 90],
               'coverage': [{'horizon': 6, 'cov95': 0.97}, {'horizon': 90, 'cov95': 0.88}]}
    monkeypatch.setattr(pages.metodo, 'national_figures', lambda: figures)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'el 97% de las veces a 6 días de la votación y el 88% a 90 días' in page
    assert 'Por debajo del 95%, los intervalos son demasiado estrechos' in page


def test_metodo_landing_names_a_single_backtest_election_in_singular(api, fresh_app):
    write_backtest(fresh_app.fs, meta={'events': ['2023-07-23'], 'horizons': [6]})
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'En las generales: 1 elección, a 6 días de la votación.' in page


def test_metodo_landing_dates_the_freeze_only_while_frozen(api, fresh_app):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'con el pronóstico definitivo. Los sondeos' in page and 'desde el' not in page
    assert 'no entran en el promedio porque su trabajo de campo acaba dentro de esos cinco días' in page
    assert 'el modelo no los usa' not in page
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda'},
                            today='2026-11-23')
    webapi.site().manifest_cache_clear()
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))
    assert 'con el pronóstico definitivo: desde el 23 nov 2026. Los sondeos' in page


def test_metodo_subpage_404_and_section_context(api, fresh_app):
    with pytest.raises(HttpError) as error:
        pages.metodo_section_context({'scope': None, 'mode': None, 'run': None}, 'nope')
    assert error.value.status == 404 and error.value.message == 'section not found'
    write_backtest(fresh_app.fs)
    context = pages.metodo_section_context({'scope': 'es', 'mode': None, 'run': None}, 'backtest')
    assert context['section'] == 'backtest' and context['published'] and context['subpages'][3]['active']
    assert context['figures']['n_sim'] == 10000 and len(context['thresholds']) == len(webapi.catalogue())


def test_metodo_landing_has_the_index_and_only_links_available_sections(api):
    page = html(call(api, '/metodo')[2])
    assert '¿Cómo promediamos las encuestas?' in page and 'href="/metodo/backtest"' in page
    assert 'href="/metodo/promedio"' in page and 'href="/metodo/escanos"' in page and 'href="/metodo/rating"' in page
    assert 'próximamente' not in page  # las cuatro preguntas tienen ya su texto
    assert 'Las preguntas del método' in page and 'Las cuatro preguntas' not in page
    text = re.sub(r'\s+', ' ', page)
    assert 'viendo solo los sondeos publicados hasta cada fecha' in text and 'lo que se sabía' not in text
    assert 'id="datos"' in page and 'id="publicacion"' in page and 'id="limitaciones"' in page
    assert 'se congela' in page and 'fuera de España' in page
    assert '1.000 simulaciones' not in page and 'id="scope-filter"' not in page


def test_metodo_promedio_text(api):
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and '<title>Demoscopio · ¿Cómo promediamos las encuestas?</title>' in page
    for anchor in ('que-entra', 'peso', 'efectos-de-casa', 'regresion', 'banda', 'voto-estimado'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'id="metodo-nav"' in page and '<math' in page and 'En resumen' in page
    assert 'ln(1 + <var>R</var>/100) / ln(1 + <var>R̄</var>/100)' in page  # el rating del sitio va de 0 a 100
    assert 'href="/metodo/promedio"' in html(call(api, '/metodo')[2])


def test_metodo_escanos_text(api):
    page = html(call(api, '/metodo/escanos')[2])
    for anchor in ('sorteo', 'hoy-y-eleccion', 'circunscripciones', 'dhondt', 'barreras', 'cifras'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'id="thresholds-table"' in page and 'Canarias' in page and 'Comunidad Valenciana' in page
    assert str(fixture_data('meta')['n_sim']) in page.replace('.', '') and '1.000 simulaciones' not in page
    assert '<math' in page and 'Ejemplo' in page
    text = re.sub(r'\s+', ' ', page)
    assert 'en un mes, uno al 30% puede moverse alrededor de un punto' in text and 'en unos meses' not in text
    assert 'suben y bajan a la vez' not in text and 'lo que gana un partido lo pierde otro' in text


def test_metodo_rating_text(api):
    page = html(call(api, '/metodo/rating')[2])
    for anchor in ('que-se-mide', 'calidad-previa', 'rating-final', 'ranking-global', 'herding'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'la mitad' in page or '50%' in page  # rating_weight 0,5 leído del catálogo
    assert '<math' in page
    landing = html(call(api, '/metodo')[2])
    assert 'Cómo funciona el modelo' not in landing and 'href="/metodo/rating"' in landing
    text = re.sub(r'\s+', ' ', page)
    assert 'la desviación de cada encuestadora apenas cambia' in text and 'no cambia:' not in text
    assert 'el denominador es la calidad previa media de las encuestadoras, no el rating medio' in text
    assert 'muestra multiplicada por <var>h</var>′<sup>2</sup>' in text
    assert 'viendo solo los sondeos publicados hasta cada fecha' in text and 'lo que se sabía' not in text
    assert 'Trabajo de campo' in text and 'Estándares' in text and '<table id="rubric-table">' in page
    assert '<strong>Diferencial</strong>' in text and 'Sesgo bloques' in text and 'Calidad</strong>' not in text
    assert 'cada punto vale 8 de calidad, con un mínimo de 5' in text and 'nadie pasa de 80 antes de los datos' in text
    assert 'los puntos que faltan se ganan acertando' in text and 'de la que apenas se sabe nada' in text
    assert 'En las tablas del sitio aparece como «Confianza»' in text and 'es la columna «Confianza»' in text
    assert 'entre 3 y 4' in text and 'en torno al 10' in text
    assert 'Por su cercanía a la votación' not in text and 'multiplica el peso por 0,7' not in text
    assert 'puesta a mano' not in text and 'rúbrica pública' in text
    assert 'calidad previa según una rúbrica pública' in re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))


def test_metodo_rating_reads_the_rubric_from_the_catalogue(api, monkeypatch):
    changed = pages.quality_rubric() | {'per_point': 10, 'floor': 1, 'cap': 100}
    monkeypatch.setattr(pages.metodo, 'quality_rubric', lambda: changed)
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'cada punto vale 10 de calidad, con un mínimo de 1' in text and 'nadie pasa de 100 antes de los datos' in text


def test_metodo_rating_reads_the_scope_weights_from_the_catalogue(api, monkeypatch):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'cada sondeo de unas autonómicas pesa la mitad que uno de unas generales' in page
    assert 'href="/metodo/promedio#peso"' in page and 'href="/casas?scope=es"' in page and 'href="/metodo/backtest"' in page
    assert 'id="metodo-nav"' in page and 'aria-current="page"' in page and 'En resumen' in page
    changed = webapi.catalogue().copy()
    changed.loc[changed.index != 'es', 'rating_weight'] = 0.25
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: changed)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'pesa el 25% de uno de unas generales' in page and 'la mitad' not in page
    mixed = changed.copy()
    mixed.loc['es-md', 'rating_weight'] = 0.5
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: mixed)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'pesa menos que uno de unas generales' in page and 'la mitad' not in page


def test_metodo_escanos_reads_the_barriers_from_the_catalogue(api, monkeypatch):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'en las generales, el 3% de los votos válidos' in page
    table = page[page.index('id="thresholds-table"'):]
    table = table[:table.index('</table>')]
    assert '<th scope="row">Canarias</th> <td>15%</td> <td>4%</td>' in table
    assert '<th scope="row">Comunidad Valenciana</th> <td>–</td> <td>5%</td>' in table
    assert 'href="/es/escanos"' in page and 'id="metodo-nav"' in page and 'aria-current="page"' in page
    changed = webapi.catalogue().copy()
    changed.loc['es', 'threshold'] = 4.5
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: changed)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'en las generales, el 4,5% de los votos válidos' in page and 'el 3% de los votos' not in page
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: changed.drop(index='es'))
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'en las generales, la que fija la ley electoral, de los votos válidos' in page


def test_metodo_escanos_quotes_the_coverage_only_with_a_backtest(api, fresh_app):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'contiene el resultado' not in page and 'se comprueba en <a href="/metodo/backtest">' in page
    write_backtest(fresh_app.fs)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'el intervalo del 95% del voto contiene el resultado el 96% de las veces a 6 días' in page


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
    assert page.count('>Método</a>') == 2 and '>Histórico Errores</a>' in page and '>Modelo</a>' not in page


def test_frozen_manifest_marks_every_page(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda electoral'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es')
    page = html(body)
    assert status == 200 and 'pronóstico definitivo para el' in page and 'congelado desde 23 nov 2026' in page
    assert 'freeze-banner' in page and 'href="/metodo#limitaciones"' in page
    status, _, body = call(api, '/es', query='mode=nowcast')
    assert 'estimación del' in html(body) and 'pronóstico definitivo' not in html(body)
    status, _, body = call(api, '/metodo', query='scope=es')
    assert 'freeze-banner' in html(body)
    publish.update_manifest(writer, freeze={'active': False})
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es', query='mode=forecast')
    assert 'freeze-banner' not in html(body) and 'pronóstico para el' in html(body) and 'congelado desde' not in html(body)


def test_window_options_run_from_the_whole_cycle_to_three_months(api):
    page = html(call(api, '/es')[2])
    assert page.index('value="all"') < page.index('value="1a"') < page.index('value="6m"') < page.index('value="3m"')
    assert 'value="all" checked' in page.replace('\n', ' ')








def test_every_page_uses_the_card_pattern_and_the_fonts(events_site):
    for path in ('/', '/es', '/es/escanos', '/elecciones', '/elecciones/es/generales/2023-07', '/metodo'):
        page = html(call(events_site, path.split('?')[0], query=path.partition('?')[2])[2])
        assert 'Qué es esto' not in page and '<meta name="description"' in page, path
        if path not in ('/', '/metodo'):
            assert 'class="how"' in page and 'class="lead"' in page, path


def test_houses_pages_use_the_card_pattern(houses):
    for path in ('/casas', '/casas/gad3'):
        page = html(call(houses, path, query='scope=es')[2])
        assert 'Qué es esto' not in page and '<meta name="description"' in page, path
        assert 'class="how"' in page and 'class="lead"' in page, path


def test_backtest_page_uses_the_card_pattern(api, fresh_app):
    write_backtest(fresh_app.fs)
    page = html(call(api, '/metodo/backtest', query='scope=es')[2])
    assert 'Qué es esto' not in page and '<meta name="description"' in page
    assert 'class="how"' in page and 'class="lead"' in page


def test_every_how_to_read_is_folded(api):
    for path in ('/es', '/es/escanos'):
        page = html(call(api, path)[2])
        main = page[page.index('<main'):]
        hows = re.findall(r'<details class="how"( open)?>', main)
        assert len(hows) > 1 and set(hows) == {''}, path




def test_fonts_are_vendored_and_declared():
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert 'DMSans.woff2' in css and 'Outfit.woff2' in css and 'Source' not in css
    assert 'font-weight: 400 700;' in css and 'font-weight: 700 800;' in css  # fuentes variables: negritas reales
    assert "url('/dist/vendor/fonts/" in css and 'font-display: swap' in css
    assert os.path.isfile(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'vendor', 'fonts', 'LICENSE'))


def test_site_css_has_the_grey_tokens_the_fluid_container_and_the_segments():
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    for token in ('--color-bg: #f4f5f8;', '--color-card: #fff;', '--color-soft: #eceef2;', '--radius: 18px;', '--shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05);', '--color-faint:'):
        assert token in css, token
    assert '--max-width' not in css and 'repeat(auto-fit, minmax(420px, 1fr))' in css
    assert '.seg {' in css and '.seg label.is-on' in css and 'max-width' not in css.split('/* Prose spans')[1][:80] and '.pill {' in css
    assert '.geo th:first-child' in css and '.geo tr.total td:first-child { background: var(--color-soft); }' in css
    hemi = css[css.index('.hemi-grid'):css.index('/* Escaños')]
    assert 'align-items: stretch' in css and '.hemi-card { display: flex; flex-direction: column; }' in css
    assert '.chart.hemi { height: auto; aspect-ratio: 2 / 1.1; max-width: 620px; margin: 0 auto; }' in css and 'grid-row' not in hemi


def test_controls_are_segments_on_every_page(api):
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    assert 'chart-window' not in page and re.search(r'<fieldset id="series-window" class="seg" hidden>', page)
    assert re.search(r'<fieldset id="pollster-window" class="seg">', page)
    assert 'chart-window' not in html(call(api, '/es/escanos')[2])


def test_casas_uses_the_glossary_and_drops_the_block_bias(houses):
    page = html(call(houses, '/casas')[2])
    assert 'class="headline"' not in page and 'id="glosario"' in page and 'Sesgo de bloques' not in page
    assert '¿Cómo calculamos el rating de las encuestadoras?' in page and 'id="scope-filter"' in page
    ranking = page[page.index('id="ranking-table"'):]
    thead = ranking[:ranking.index('</thead>')]
    heads = re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', thead)
    assert heads == ['Encuestadora', 'Rating', 'Confianza', 'Elecciones', 'Sondeos', 'Diferencial', 'Error medio',
                     'Sesgo bloques', 'Rebaño']
    assert 'class="help"' not in thead  # sin interrogante: el glosario explica las columnas (2026-10-16)
    assert 'Calidad' not in thead  # la palabra pública es «Confianza»
    assert 'Herding' not in page  # la palabra pública es «rebaño» (2026-10-15)


def test_casa_profile_shows_the_magnitude_and_the_signed_margin(houses):
    page = html(call(houses, '/casas/cis')[2])
    assert 'Error de bloques' not in page and 'Sesgo de bloques' not in page and 'miden lo mismo' not in page
    assert 'id="glosario"' in page and page.count('<td') == page.count('</td>')
    row = page[page.index('id="path-table"'):]
    row = row[row.index('<tbody>'):]
    assert row[:row.index('</tr>')].count('<td') == 7  # Elección, Sondeos, las tres medidas, Rating y Cambio (2026-10-15)


# Fase 7b, tarea 1: terminología pública, siglas, sin control de modo, fecha de hoy en el título, "Cómo leerlo"
# plegados y etiquetas del menú.
OLDER_RUN = '20261001-120000'


def visible_text(page):
    """Texto visible de una página: sin el bloque JSON, los scripts ni las etiquetas (los atributos fuera)."""
    page = re.sub(r'<script\b.*?</script>', ' ', page, flags=re.S)
    return re.sub(r'<[^>]+>', ' ', page)


# The one public "casa": the title Luis asked for in the profile (2026-10-15).
EFFECTS_TITLE = 'Efectos de casa por elección'
# The other public "casa": the column header Luis asked for in the weights example of the method (2026-10-16).
WEIGHTS_HOUSE_HEADER = '<th scope="col">Casa</th>'


def assert_public_wording(page, label):
    """Ni "casa/s" ni "patrocinador/es" en el texto visible (salvo el título `EFFECTS_TITLE` del perfil y la cabecera
    `WEIGHTS_HOUSE_HEADER` del ejemplo del promedio), y ningún "Cómo leerlo" abierto."""
    text = visible_text(page.replace(WEIGHTS_HOUSE_HEADER, ' ')).replace(EFFECTS_TITLE, ' ')
    assert not re.search(r'\b[Cc]asas?\b', text), (label, re.findall(r'.{40}\b[Cc]asas?\b.{20}', text)[:3])
    assert not re.search(r'\b[Pp]atrocinador(es)?\b', text), label
    assert '<details class="how" open' not in page, label


def test_public_wording_uses_encuestadora_and_medio_and_short_names(events_site):
    for path, query in (('/', ''), ('/es', ''), ('/es/escanos', ''), ('/elecciones', ''),
                        ('/elecciones', 'scope=es&date=2023-07-23'), ('/metodo', ''), ('/metodo/promedio', ''),
                        ('/metodo/escanos', ''), ('/metodo/rating', ''), ('/metodo/backtest', '')):
        assert_public_wording(html(call(events_site, path, query=query)[2]), path + '?' + query)
    for path in ('/es', '/es/escanos', '/'):
        page = html(call(events_site, path)[2])
        # the party cards of Promedios show the full name under the initials (phase 8); nothing else does
        without_cards = re.sub(r'<div class="fl"[^>]*>[^<]*</div>', '', page)
        assert 'Partido Popular' not in visible_text(without_cards) and 'title="Partido Popular"' in page, path
    # routes, ids and file names keep their names
    page = html(call(events_site, '/elecciones/es/generales/2023-07')[2])
    assert '/casas/gad3' in page
    source = open(os.path.join(pages.TEMPLATES_DIR, 'elecciones.html'), encoding='utf-8').read()
    assert '<select name="casa">' in source and "'/casas'" in source


def test_houses_pages_use_encuestadora(houses):
    for path in ('/casas', '/casas/gad3', '/casas/nueva'):
        page = html(call(houses, path, query='scope=es')[2])
        assert_public_wording(page, path)
        assert '/dist/js/pages/casas.js' in page
    page = html(call(houses, '/casas')[2])
    assert '<title>Demoscopio · Ranking Encuestadoras</title>' in page
    assert 'aria-current="page">Ranking Encuestadoras</a>' in page
    status, _, body = call(houses, '/casas', query='id=999')
    assert status == 404 and 'Encuestadora no encontrada' in html(body)


def test_menu_labels_and_titles(api):
    page = html(call(api, '/elecciones')[2])
    assert '>Histórico Errores</a>' in page and '>Ranking Encuestadoras</a>' in page
    assert '<title>Demoscopio · Histórico Errores</title>' in page
    assert [entry['label'] for entry in pages.PAGES] == ['Inicio', 'Histórico Errores', 'Ranking Encuestadoras',
                                                          'Método']


def test_event_bar_has_no_mode_control_and_the_title_shows_today(api, fresh_app):
    from datetime import datetime
    from mtpy.lib.pages.filters import _madrid_tz
    today = pages.fmt_date(datetime.now(_madrid_tz()).date().isoformat())
    for path in ('/es', '/es/escanos'):
        page = html(call(api, path)[2])
        bar = page[page.index('id="event-bar"'):page.index('<main')]
        # the district form of Escaños keeps `mode` hidden: no radio and nothing in the bar
        assert 'type="radio" id="mode-' not in page and 'name="mode"' not in bar, path
        assert 'mode-help' not in page and 'id="event-controls"' not in page, path
        assert '¿Qué cambia entre Hoy y Elección?' not in page and 'class="segmented"' not in page, path
        assert 'stimación a ' + today in page, path
        # without JS the bar still has the title, the date and the tabs
        assert 'Elecciones generales' in bar and '29 nov 2026' in bar and 'href="/es/escanos?mode=nowcast"' in bar
        # "Estimación a" is only ever followed by today; the footer names the publication date otherwise
        assert set(re.findall(r'[Ee]stimación a (\d+ \w+ \d{4})', page)) == {today}, path
        footer = page[page.index('<footer'):]
        assert 'Actualizado: {} (8 oct 2026)'.format(RUN) in footer and 'Estimación' not in footer, path
    write_bundle(fresh_app.fs, run_id=OLDER_RUN, manifest=False)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/es', query='run=' + OLDER_RUN)[2])
    assert 'estimación del 8 oct 2026' in page and 'Estimación a ' + today not in page  # run_at of the fixture
    bar = page[page.index('id="event-bar"'):page.index('<main')]
    assert '<span class="pill event-run">Run del 8 oct 2026</span>' in bar and '<form' not in bar
    # `?mode=` is still accepted in the query and kept in the tabs
    page = html(call(api, '/es', query='mode=forecast')[2])
    assert 'pronóstico para el 29 nov 2026' in page and 'href="/es/escanos?mode=forecast"' in page
    assert 'name="mode"' not in page


def test_frozen_site_has_no_control_and_shows_the_forecast(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    publish.update_manifest(writer, {'es': publish.latest_entry(fixture_data('headline') | {'run_id': RUN})},
                            freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    for path in ('/es', '/es/escanos'):
        page = html(call(api, path)[2])
        bar = page[page.index('id="event-bar"'):page.index('<main')]
        assert 'ronóstico definitivo para el 29 nov 2026' in page, path
        assert 'type="radio" id="mode-' not in page and 'name="mode"' not in bar, path
        assert 'id="event-controls"' not in page, path


def test_frozen_nowcast_link_keeps_the_run_date_in_the_title(api, fresh_app):
    from datetime import date
    writer = bundle.BundleWriter(fresh_app.fs)
    publish.update_manifest(writer, {'es': publish.latest_entry(fixture_data('headline') | {'run_id': RUN})},
                            freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/es', query='mode=nowcast')[2])
    assert 'estimación del' in page
    assert 'Estimación a {}'.format(pages.fmt_date(date.today().isoformat())) not in page


def test_title_of_shows_today_in_the_nowcast():
    from datetime import date
    meta = {'as_of': '2026-10-08', 'event_date': '2026-11-29', 'run_at': '2026-10-08T12:00:00Z'}
    today = date(2026, 10, 13)
    assert pages.title_of('nowcast', meta, False, False, True, today=today) == 'Estimación a 13 oct 2026'
    assert pages.title_of('nowcast', meta, False, True, True, today=today) == 'Estimación a 13 oct 2026'
    assert pages.title_of('nowcast', meta, False, True, False, today=today) == 'Estimación del 8 oct 2026'
    assert pages.title_of('forecast', meta, False, False, True, today=today) == 'Pronóstico para el 29 nov 2026'


def test_promedios_controls_in_two_rows_and_the_polls_table_by_name(api):
    page = html(call(api, '/es')[2])
    assert page.index('id="series-view"') < page.index('id="series-window"')
    before = page[:page.index('id="polls"')]
    assert '<label class="pill"' not in page and 'data-all' not in before and 'data-none' not in before
    assert '<th scope="col" class="f l" data-key="pollster"><button type="button" class="sort">Encuestadora</button></th>' in page and '<th scope="col">Casa</th>' not in page
    data = initial(call(api, '/es')[2])
    assert data['polls']['polls'][0]['name'] and 'sample_size' in data['polls']['polls'][0]


def test_promedios_legend_is_the_party_selector_and_the_house_chart_has_its_own_controls(api):
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    assert 'id="series-parties"' not in page and 'id="series-blocks"' not in page
    assert 'name="party"' not in page[:page.index('id="polls"')]  # the chips of the polls table are not the legend
    assert re.search(r'<fieldset id="pollster-view" class="[^"]*">', page) and re.search(r'<fieldset id="pollster-window" class="[^"]*">', page)
    assert 'name="pview" value="partidos" checked' in page and 'name="pwindow" value="all" checked' in page and 'name="pwindow" value="3m"' in page
    assert page.index('id="pollster-window"') < page.index('id="pollster-select"')
    assert 'Pulsa un partido de la leyenda' in page and 'Tiene su propia ventana y vista' in page


def test_poll_name_follows_the_loader_rule():
    assert pages.poll_name({'pollster': 'GAD3', 'sponsor': 'ABC'}) == 'GAD3 / ABC'
    assert pages.poll_name({'pollster': 'GAD3', 'sponsor': None}) == 'GAD3'
    assert pages.poll_name({'pollster': 'GAD3', 'sponsor': ''}) == 'GAD3'
    assert pages.poll_name({'pollster': 'GAD3'}) == 'GAD3'


def test_ink_color_is_dark_on_light_colours_and_white_on_dark_ones():
    assert pages.ink_color('#adcfef') == '#111' and pages.ink_color('#baf73e') == '#111'
    assert pages.ink_color('#ffffff') == '#111' and pages.ink_color('#fff') == '#111'
    assert pages.ink_color('#1d84ce') == '#fff' and pages.ink_color('#ef1c27') == '#fff'
    assert pages.ink_color(pages.OTHERS_COLOR) == '#fff' and pages.ink_color('#000') == '#fff'
    assert pages.ink_color('#adcfef80') == '#111'  # the alpha is ignored
    assert pages.ink_color('garbage') == '#fff' and pages.ink_color(None) == '#fff'
    assert pages.environment().filters['ink'] is pages.ink_color


def test_escanos_pills_carry_the_ink_of_their_colour(api):
    page = html(call(api, '/es/escanos')[2])
    for match in re.finditer(r'<label class="pill" style="--pill-color: (#[0-9a-fA-F]+); --pill-ink: (#[0-9a-f]+)"',
                             page):
        assert match.group(2) == pages.ink_color(match.group(1))
    assert len(re.findall(r'--pill-ink: ', page)) == page.count('<label class="pill"') > 0


def test_promedios_results_switch_between_parties_and_blocks_from_the_hemicycle_card(api):
    """Un solo par hemiciclo/tabla: la vista Partidos/Bloques vive en la tarjeta del hemiciclo y la tabla de bloques
    sale oculta del servidor (sin JS se ven los partidos)."""
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    hemi = page[page.index('id="hemicycle-card"'):page.index('id="parties-card"')]
    assert re.search(r'<fieldset id="results-view" class="seg" hidden>', hemi)
    assert 'name="rview" value="partidos" checked' in hemi and 'name="rview" value="bloques"' in hemi
    assert 'Con la vista Bloques' in hemi
    assert re.search(r'<section class="card" id="blocks-card" hidden>', page)
    assert not re.search(r'<section class="card" id="parties-card" hidden>', page)


def test_tables_show_probabilities_with_one_decimal_and_the_home_card_without(api):
    page = html(call(api, '/es/escanos')[2])
    table = page[page.index('id="parties-table"'):page.index('</table>', page.index('id="parties-table"'))]
    cells = re.findall(r'<td>([^<]*%)</td>', table)
    assert cells and all(re.fullmatch(r'(?:[<>] )?\d+,\d%', cell) for cell in cells), cells
    home = html(call(api, '/')[2])
    assert re.search(r'· Derecha (?:> )?\d+,\d%</small>', home)  # the majority chip of the hero, one decimal


def test_polls_table_offers_the_seats_view_only_when_a_poll_publishes_seats(api, fresh_app):
    page = html(call(api, '/es')[2])
    assert 'id="polls-mode"' not in page and 'name="pmode"' not in page  # el fixture no trae escaños
    polls = fixture_data('polls')
    polls['polls'][0]['seats'] = {'PP': [140, 142]}
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', polls, 'es', run_id=RUN)
    webapi.site().files.clear()
    body = call(api, '/es')[2]
    page = html(body).replace('\n', ' ')
    assert re.search(r'<fieldset id="polls-mode" class="seg">', page)
    assert 'name="pmode" value="pct" checked' in page and 'name="pmode" value="esc"' in page
    assert 'en las encuestadoras que los dan, los escaños' in page and 'En la vista Escaños' in page
    assert initial(body)['polls']['polls'][0]['seats'] == {'PP': [140, 142]}


def test_bar_and_shade_filters_follow_print_ratings():
    assert pages.bar_width(82.37) == 82.4 and pages.bar_width(120) == 100 and pages.bar_width(-3) == 0
    assert pages.bar_width(None) == 0 and pages.bar_width(5, 10, 10) == 0 and pages.bar_width(1.5, 1, 2) == 50
    assert pages.bar_sides(-1.0, 2) == {'left': 25.0, 'width': 25.0, 'side': 'neg'}
    assert pages.bar_sides(0.5, 2) == {'left': 50.0, 'width': 12.5, 'side': 'pos'}
    assert pages.bar_sides(3.0, 2) == {'left': 50.0, 'width': 50.0, 'side': 'pos'}  # saturada en el límite
    assert pages.bar_sides(-9, 2) == {'left': 0.0, 'width': 50.0, 'side': 'neg'}
    for value, limit in ((0, 2), (None, 2), (float('nan'), 2), (1, 0), (True, 2)):
        assert pages.bar_sides(value, limit) == {'left': 50.0, 'width': 0.0, 'side': 'zero'}
    assert pages.shade(2, 0, 4, '#d62728') == 'rgba(214, 39, 40, 0.275)'
    assert pages.shade(9, 0, 4, '#d62728') == 'rgba(214, 39, 40, 0.55)'  # recortado al tope
    assert pages.shade(0.5, 1, 0, '#d62728') == 'rgba(214, 39, 40, 0.275)'  # el rango puede ir hacia abajo
    assert pages.shade(1.5, 1, 2, '#7f7f7f', alpha=0.4) == 'rgba(127, 127, 127, 0.2)'
    assert pages.shade(0, 0, 4, '#d62728') == 'rgba(214, 39, 40, 0.0)'
    for value, start, end, color in ((None, 0, 4, '#d62728'), (1, 2, 2, '#d62728'), (1, 0, 4, 'red'), (True, 0, 4, '#d62728')):
        assert pages.shade(value, start, end, color) == 'transparent'
    env = pages.environment()
    assert env.filters['bar'] is pages.bar_width and env.filters['sides'] is pages.bar_sides and env.filters['shade'] is pages.shade


def test_event_detail_helpers_of_the_hemicycle_and_the_split():
    assert pages.majority_of(350) == 176 and pages.majority_of(135) == 68 and pages.majority_of(0) is None
    assert pages.majority_of(None) is None and pages.majority_of(True) is None and pages.majority_of(350.0) is None
    assert pages.split_sides(['Derecha', 'Izquierda']) == {'left': 'Izquierda', 'right': 'Derecha'}
    assert pages.split_sides(['Izquierda', 'Derecha', 'Otro']) == {'left': 'Izquierda', 'right': 'Derecha'}
    assert pages.split_sides(['A', 'B']) == {'left': 'B', 'right': 'A'}  # como el cuaderno: el primero a la derecha
    assert pages.split_sides(['Derecha']) is None and pages.split_sides([]) is None
    catalog = pages.Catalog({'PSOE': {}, 'PP': {}, 'VOX': {}})
    assert pages.chart_parties(catalog, ['PP', 'VOX', 'ERC'], ['PP', 'PSOE', 'Cs', 'ERC']) == ['PSOE', 'PP', 'VOX', 'ERC', 'Cs']
    assert pages.chart_parties(catalog) == [] and pages.chart_parties(pages.Catalog(), ['Cs']) == ['Cs']


def test_rng_cell_places_the_range_the_median_and_the_majority():
    assert pages.rng_cell(118, 160, 140, 350, 176) == {'left': 33.7, 'width': 12.0, 'mark': 40.0, 'line': 50.3}
    assert pages.rng_cell(0, 4, 0, 350) == {'left': 0.0, 'width': 1.1, 'mark': 0.0, 'line': None}
    assert pages.rng_cell(300, 420, 350, 350, 176)['width'] == 14.3  # recortado a 100
    assert pages.rng_cell(None, 160, 140, 350) is None and pages.rng_cell(118, 160, None, 350) is None
    assert pages.rng_cell(118, 160, 140, 0) is None and pages.rng_cell(160, 118, 140, 350) is None
    assert pages.environment().filters['rng'] is pages.rng_cell


def test_card_subtitle_is_a_pill(api):
    page = html(call(api, '/es')[2])
    assert '<span class="pill">estimación a ' in page and 'mode-subtitle' not in page


def test_tabbar_links_the_five_sections_and_marks_the_active_one():
    bar = pages.tabbar('es', None, '/escanos', True, '?mode=forecast')
    assert [row['href'] for row in bar] == ['/', '/es?mode=forecast', '/es/escanos?mode=forecast', '/elecciones?scope=es', '/casas']
    assert [row['active'] for row in bar] == [False, False, True, False, False]
    assert [row['icon'] for row in bar] == ['home', 'line', 'hemi', 'target', 'trophy']
    bar = pages.tabbar('es-md', '/casas', '', False, '')
    assert bar[1]['href'] == '/es-md' and bar[3]['href'] == '/elecciones?scope=es-md' and bar[4]['active']
    assert not any(row['active'] for row in pages.tabbar('es', '/metodo', '', False, ''))
    assert pages.chamber_text('es', {'n_seats': 350}) == 'Congreso, 350 escaños'
    assert pages.chamber_text('es-md', {'n_seats': 135}) == 'Parlamento, 135 escaños'
    assert pages.chamber_text('es', {}) is None


def test_header_is_compact_and_the_pages_carry_the_tabbar_and_the_icons(api):
    page = html(call(api, '/es')[2])
    assert 'brand-tagline' not in page and '<meta name="description"' in page
    assert '<svg hidden aria-hidden="true"' in page and page.index('<symbol id="i-home"') < page.index('<main')
    assert 'class="ico-btn" href="/metodo" aria-label="Método"' in page
    bar = re.search(r'<nav class="tabbar" aria-label="Secciones">(.*?)</nav>', page, re.S).group(1)
    assert bar.count('<a ') == 5 and 'href="/es/escanos?mode=nowcast"' in bar and '<use href="#i-trophy"' in bar
    assert re.search(r'<a class="on" href="/es\?mode=nowcast"[^>]*><svg[^>]*><use href="#i-line"', bar)
    assert page.index('</footer>') < page.index('<nav class="tabbar"')


def test_event_bar_names_the_chamber_and_the_publication(api):
    page = html(call(api, '/es')[2])
    assert '<h2>Elecciones generales<small>29 nov 2026 · ' in page and 'Congreso, 350 escaños</small></h2>' in page
    assert '<span class="pill brand">Estimación a ' in page
    assert 'Publicación del 8 oct 2026 · 120 sondeos' in page
    assert 'class="seg lg event-tabs"' in page and re.search(r'<a class="on" href="/es\?mode=nowcast" aria-current="page"><svg', page)


def test_error_page_has_no_tabbar(api):
    status, _, body = call(api, '/nope')
    assert status == 404 and 'class="tabbar"' not in html(body) and '<svg hidden aria-hidden="true"' not in html(body)


def test_promedios_opens_with_the_kpis_the_party_cards_and_the_lead_strip(events_site):
    page = html(call(events_site, '/es')[2])
    assert page.index('id="kpis"') < page.index('id="pcards"') < page.index('id="series"')
    kpis = re.findall(r'<div class="kpi"><div class="l">([^<]+)</div>', page)
    assert kpis == ['Primer partido', 'Escaños del líder', 'Mayoría absoluta', 'Sondeos del ciclo', 'Último sondeo']
    flat = page.replace('\n', '')
    assert '<div class="v" style="color: #1d84ce">PP<small>33,1% del voto</small></div>' in flat
    assert '135<small>de 350</small>' in flat and 'Rango 95%: 120–150' in page
    assert '<div class="s">PP: 135 escaños</div>' in page and '41,0%<small>Derecha</small>' in flat
    assert '120<small>de 15 encuestadoras</small>' in flat and '5 oct<small>Sigma Dos</small>' in flat
    card = re.search(r'<article class="pc" style="--c: #1d84ce" data-name="PP">(.*?)</article>', page, re.S).group(1)
    assert '<span class="rk">1.º</span>' in card and '<b>135</b><span>33,1%</span>' in card
    assert '−25 esc. <i>vs 2023</i>' in card and '+0,0 <i>pt</i>' in card  # 135 − 160; 33,1 − 33,06 = +0,04 → +0,0
    assert 'class="spark"' in card and 'Último año' in card
    assert 'Líder: <b>PP</b> · <b>135 escaños</b> · le faltan <b>41</b> para la mayoría · P(primero) <b>62,0%</b>' in flat


def test_promedios_party_table_draws_the_vote_bar_and_the_seat_range(api):
    page = html(call(api, '/es')[2])
    table = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    assert '<span class="bar"><i style="width: 100.0%; background: #1d84ce"></i></span>' in table
    assert 'class="rng"' in table and 'left: 34.3%; width: 8.6%' in table and 'calc(38.6% - 1px)' in table and '<u style="left: 50.3%"></u>' in table
    assert '120–150' in table and '<th scope="col">Rango 95%</th>' in table


def test_promedios_without_polls_part_and_with_an_unknown_party_still_renders(api, fresh_app, tmp_path):
    summary = fixture_data('summary')
    summary['parties'].append(dict(summary['parties'][0], name='XX', seats=3, pct=1.0, seats_lo=0, seats_hi=5, p_first=0))
    summary['totals']['XX'] = 3
    writer = write_bundle(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'summary', 'nowcast'), 'summary', summary, 'es', run_id=RUN, mode='nowcast')
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'polls')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es')
    page = html(body)
    assert status == 200 and '<div class="l">Último sondeo</div>' not in page and 'id="kpis"' in page  # the footer keeps the date
    assert 'data-name="XX"' not in page  # fuera de bmaps.main
    assert 'background: #9e9e9e' in re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)


def test_home_hero_is_the_national_scope_with_its_figures(api):
    page = html(call(api, '/')[2])
    flat = page.replace('\n', '')
    assert '<h1 class="hero-t">' in page and 'un solo <em>promedio</em>.' in page and pages.inicio.INTRO in page
    assert 'Generales · 29 nov 2026 · ' in page and 'id="hero-card"' in page and 'class="flag"' in page
    assert 'Congreso de los Diputados · 350 escaños · 29 nov 2026' in page
    assert '<b style="color: #1d84ce">135</b>' in flat and 'primer partido con el <strong>33,1%</strong>' in page
    bar = re.search(r'<div class="seatbar" id="hero-seatbar">(.*?)<div class="chips">', page, re.S).group(1)
    assert '<i style="width: 38.6%; background: #1d84ce" title="PP 135"></i>' in bar and 'left: 50.3%' in bar
    assert 'Derecha 135' in bar
    # con p < 0,5 el chip va en gris: "Sin mayoría clara · Derecha 41%" (D4); con p >= 0,5, `.chip.ok`
    assert '<span class="chip">Sin mayoría clara <small>· Derecha 41,0%</small></span>' in flat
    assert 'href="/es">Ver el promedio' in page and 'class="btn" href="/es/escanos">Escaños' in page
    assert 'Actualizado: 8 oct 2026 · 120 sondeos de 15 encuestadoras' in page
    assert 'id="hemicycle-mini"' in page and '/dist/js/pages/inicio.js' in page
    data = initial(call(api, '/')[2])
    assert data == {'hero': {'seats': {'PP': 135}, 'colors': {'PP': '#1d84ce'}, 'fullnames': {'PP': 'Partido Popular'},
                             'order': ['PP'], 'n_seats': 350, 'majority': 176}}
    polls = re.search(r'<div class="plist" id="latest-polls">(.*?)</div>\s*</section>', page, re.S).group(1)
    assert polls.count('class="pr"') == FIXTURE_POLLS and '<b>5</b>oct' in polls and 'Sigma Dos' in polls and '33,1%' in polls


def test_home_lists_the_regional_scopes_as_rows_and_names_the_unpublished_ones(api, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    rows = re.search(r'<div class="card rows" id="auto-rows">(.*?)</div>\s*</div>\s*<aside', page, re.S).group(1)
    assert rows.count('class="row"') == 1 and 'href="/es-md"' in rows and '<div class="n">Madrid</div>' in rows
    assert 'PP 33,1% · 135 esc.' in rows and 'Derecha: <b>41,0%</b> de mayoría' in rows
    assert 'Sin pronóstico todavía: Andalucía, Aragón' in rows and rows.count(',') >= 15 and 'Madrid,' not in rows
    assert 'card-muted' not in page and 'Pronóstico no publicado todavía' not in page
    assert page.index('href="/es"') < page.index('href="/es-md"')


def test_home_tolerates_a_regional_scope_whose_run_has_no_headline(api, fresh_app, tmp_path):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es-md', '20261009-120000', 'headline')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/')
    assert status == 200 and '<div class="n">Madrid</div>' in html(body)


def test_home_hero_falls_back_to_the_first_published_scope(api, fresh_app, tmp_path):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    path = tmp_path / bundle.PREFIX / bundle.path_manifest()
    manifest = json.loads(path.read_text())
    del manifest['data']['scopes']['es']
    path.write_text(json.dumps(manifest))
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert re.search(r'<span class="eyebrow"[^>]*>Madrid · \d+ \w+ \d{4} · ', page)
    assert 'class="flag"' not in page and 'Parlamento · 350 escaños · ' in page and 'Si hoy hubiera elecciones en Madrid' in page
    assert 'id="auto-rows"' not in page  # el único ámbito publicado es el hero
    assert 'href="/es-md/escanos"' in re.search(r'<nav class="tabbar".*?</nav>', page, re.S).group(0)


def test_home_without_summary_or_polls_keeps_the_hero_without_the_bar(api, fresh_app, tmp_path):
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'polls')))
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'summary', 'nowcast')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/')
    page = html(body)
    assert status == 200 and 'id="hero-card"' in page and 'id="hero-seatbar"' not in page
    assert 'id="latest-polls"' not in page and 'initial-data' not in page
    assert '<b style="color: #1d84ce">135</b>' in page.replace('\n', '')  # el líder sale del headline
    assert 'Sin mayoría clara <small>· Derecha 41,0%</small>' in page.replace('\n', '')


def test_home_hero_segments_read_the_chamber_from_the_summary():
    """La barra de escaños sale del resumen (`totals` y `n_seats`), no del `meta`, cuyo esquema exige `n_seats`."""
    catalog = pages.catalog(fixture_data('meta'))
    assert pages.inicio.hero_segments({'totals': {'PP': 135}, 'n_seats': 350}, catalog) == [
        {'name': 'PP', 'color': '#1d84ce', 'seats': 135, 'width': 38.6}]
    assert pages.inicio.hero_segments({'parties': [{'name': 'PP', 'seats': 135}]}, catalog)[0]['width'] == 100.0
    assert pages.inicio.hero_segments(None, catalog) == [] and pages.inicio.hero_segments({'totals': {}}, catalog) == []


def test_home_follows_the_default_mode_while_frozen(api, fresh_app):
    headline = fixture_data('headline')
    headline['forecast']['parties'] = [dict(party, name='PP', pct=41.5, seats=161) for party in headline['forecast']['parties']]
    writer = write_bundle(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'headline'), 'headline', headline, 'es', run_id=RUN)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert 'Si hoy hubiera elecciones' in page and '33,1' in page and '41,5' not in page
    publish.update_manifest(writer, {'es': publish.latest_entry(headline | {'run_id': RUN})},
                            freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert 'Pronóstico definitivo' in page and '41,5%' in page and '>161</b>' in page and 'Si hoy hubiera' not in page


def test_home_closing_sections_link_the_model_pages_and_the_past(events_site):
    page = html(call(events_site, '/')[2])
    more = page[page.index('id="more"'):page.index('id="how"')]
    for href, icon in (('/elecciones', 'i-target'), ('/casas', 'i-trophy'), ('/metodo', 'i-book')):
        assert re.search(r'<a class="card card-link" href="%s">\s*<span class="ic"><svg[^>]*><use href="#%s"' % (href, icon), more), href
    assert page.count('<div class="card step">') == 4 and 'Recoge los sondeos' in page
    past = page[page.index('id="past"'):]
    assert 'href="/elecciones/es/generales/2023-07"' in past and past.index('2023') < past.index('2019')
    assert 'PP −25 escaños <small>vs 2023</small>' in page.replace('\n', '')  # el chip de delta del hero


def test_escanos_table_draws_the_seat_range_and_the_ridge_is_compact(api):
    page = html(call(api, '/es/escanos')[2])
    table = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    assert 'class="rng"' in table and '<u style="left: 50.3%"></u>' in table and '120–150' in table
    assert '<fieldset class="pills lg" id="coalition-parties">' in page
    assert '<p class="chip" id="coalition-result">' in page


def test_history_figures_are_a_kpi_row(history_site):
    page = html(call(history_site, '/elecciones')[2])
    kpis = re.search(r'<div class="kpis" id="figures-list" hidden>(.*?)</div>\s*<ul class="complements"', page, re.S).group(1)
    assert kpis.count('<div class="kpi">') == 4
    for id_ in ('figure-polls', 'figure-events', 'figure-mean', 'figure-bias'):
        assert 'class="v" id="%s"' % id_ in kpis, id_
    assert 'id="figure-mean-label"' in kpis and '<dl class="figures"' not in page


def test_profile_opens_with_three_kpis_and_keeps_the_path_table(houses):
    page = html(call(houses, '/casas/gad3', query='scope=es')[2])
    profile = page[page.index('id="profile"'):page.index('id="path"')]
    assert re.findall(r'<div class="kpi"><div class="l">([^<]+)</div>', profile) == ['Rating', 'Confianza', 'Tipo']
    assert '<div class="v">82,4<small></small></div>' in profile.replace('\n', '') and 'entra así en el promedio' in profile
    assert 'Tipo: ' not in profile and 'id="path-table"' in page and 'id="rating-path"' in page


def test_election_detail_has_the_lead_strip_without_p_first(events_site):
    page = html(call(events_site, '/elecciones/es/generales/2023-07')[2])
    strip = re.search(r'<p class="lead-strip">(.*?)</p>', page, re.S).group(1).replace('\n', '')
    assert 'Líder: <b>PP</b> · <b>160 escaños</b>' in strip and 'P(primero)' not in strip
    assert 'le faltan <b>16</b> para la mayoría' in strip
    assert 'class="chart hemi"' in page
    table = re.search(r'<table id="results-table">(.*?)</table>', page, re.S).group(1)
    assert '<span class="bar"><i style="width: 100.0%;' in table


def test_method_nav_is_a_segmented_control(api):
    page = html(call(api, '/metodo/promedio')[2])
    nav = re.search(r'<nav id="metodo-nav" class="seg metodo-nav"[^>]*>(.*?)</nav>', page, re.S).group(1)
    assert re.search(r'<a class="on" href="/metodo/promedio"[^>]*aria-current="page"', nav)


def test_polls_csv_link_survives_a_single_page_and_the_legend_is_accessible(api):
    """Revisión final de la fase 8: el enlace CSV queda fuera de la paginación (que el navegador esconde con una sola
    página), la leyenda de píldoras lleva su grupo y sus tintas, y el texto apagado pasa AA."""
    page = html(call(api, '/es')[2])
    polls = page[page.index('id="polls"'):page.index('id="closing"')]
    pag = re.search(r'<div class="pag" id="polls-pag" hidden>(.*?)</div>', polls, re.S).group(1)
    assert 'Descargar CSV' not in pag and re.search(r'<p class="csv-link"><a href="/api/v1/forecast/es/polls\?[^"]*format=csv">Descargar CSV</a></p>', polls)
    assert '<div class="legend" id="series-legend" role="group" aria-label="Partidos y bloques del gráfico" hidden>' in page
    data = initial(call(api, '/es')[2])
    assert data['inks'] == {'PP': pages.ink_color('#1d84ce'), 'Derecha': pages.ink_color('#1d84ce')}
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert '.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(0, 1fr));' in css
    assert '.kpi:last-child:nth-child(odd) { grid-column: 1 / -1; }' in css
    assert '.mut { color: var(--color-muted); font-size: 12px; }' in css and '.evbar .run small { font-size: 11.5px; color: var(--color-muted); }' in css
    for gone in ('.event-title', '.brand-tagline', '.event-run,'):
        assert gone not in css, gone


def test_profile_effects_switch_between_result_and_cycle_deviations(houses):
    """El perfil enseña la desviación frente al resultado (solo las elecciones con tres sondeos en los 90 días
    previos) y, con el conmutador, la desviación frente al consenso del ciclo (todas); el "Cómo leerlo" lo explica."""
    page = html(call(houses, '/casas/gad3', query='scope=es')[2])
    card = page[page.index('id="effects"'):page.index('id="herding"')]
    assert '<fieldset id="effects-view" class="seg" hidden>' in card
    assert 'name="eview" value="resultado" checked' in card and 'name="eview" value="ciclo"' in card
    result = re.search(r'<table id="profile-effects">(.*?)</table>', card, re.S).group(1)
    cycle = re.search(r'<table id="profile-effects-cycle">(.*?)</table>', card, re.S).group(1)
    assert result.count('<tr>') == 1 + 2 and cycle.count('<tr>') == 1 + 3  # la cabecera y las filas
    assert '0,7' in cycle and '0,7' not in result  # 2026 solo tiene la desviación frente al consenso
    assert '<div class="table-wrap" id="profile-effects-cycle-wrap" hidden>' in card
    assert 'tres sondeos' in card and '90 días' in card and 'consenso' in card
    records = [{'event_date': '2023-07-23', 'event_scope': 'es', 'party': 'PP', 'dev_result_c': None, 'dev_cycle': 1.5},
               {'event_date': '2019-11-10', 'event_scope': 'es', 'party': 'PP', 'dev_result_c': 0.4, 'dev_cycle': None}]
    assert [row['event_date'] for row in pages.effects_table(records)['rows']] == ['2019-11-10']
    assert [row['event_date'] for row in pages.effects_table(records, key='dev_cycle')['rows']] == ['2023-07-23']


def test_home_wording_of_the_second_walkthrough(api):
    """Apuntes de Luis del 15 de octubre: la fecha de la estimación es la de hoy, "Actualizado:", sin la nota de las
    tres páginas, el subtítulo nuevo, diez encuestas, el pie con el perfil de X y aire bajo la barra de escaños."""
    from datetime import datetime
    page = html(call(api, '/')[2])
    today = pages.fmt_date(datetime.now(pages.filters._madrid_tz()).date().isoformat())
    side = page[page.index('id="side-hemicycle"'):page.index('id="side-polls"')]
    assert '<span class="pill">%s</span>' % today in side
    assert 'Actualizado: 8 oct 2026 · 120 sondeos de 15 encuestadoras' in page and 'Última publicación' not in page
    assert 'Tres páginas comunes' not in page
    assert 'Un proyecto transparente y divulgativo, con el método y modelo a la vista.' in page and 'Sin cookies' not in page
    assert pages.inicio.LATEST_POLLS == 10
    footer = page[page.index('<footer'):]
    assert 'Actualizado: {} (8 oct 2026)'.format(RUN) in footer and 'Publicación:' not in footer
    assert 'Modelo y curación: Luis Sancho (<a href="https://x.com/luis_sancho" rel="me">@luis_sancho</a>)' in footer
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert '.seatbar { position: relative; margin: 6px 0 34px; }' in css


def test_second_walkthrough_layout_rules_and_the_method_index(api):
    """Apuntes de Luis (2026-10-15, segunda tanda): filtros de los gráficos en una fila en escritorio, gráficos de
    series de 600 px, textos a todo el ancho de la tarjeta y las cuatro preguntas del método en una fila."""
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert '.series-controls { display: flex; flex-direction: row; flex-wrap: wrap; align-items: center;' in css
    assert '  .series-controls { flex-direction: column; align-items: flex-start; }' in css  # móvil
    assert '.chart.chart-tall { height: 600px; }' in css and '  .chart.chart-tall { height: 280px; }' in css
    assert 'max-width: 72ch' not in css and 'max-width: 90ch' not in css
    assert '.grid-four { grid-template-columns: repeat(4, minmax(0, 1fr)); }' in css
    page = html(call(api, '/metodo')[2])
    assert '<div class="grid grid-four">' in page


def test_profile_and_detail_canonicals_are_the_clean_urls(houses, fresh_app):
    page = html(call(houses, '/casas/gad3', query='scope=es')[2])
    assert '<link rel="canonical" href="https://demoscopio.es/casas/gad3">' in page
    assert '<meta property="og:url" content="https://demoscopio.es/casas/gad3">' in page
    ranking = html(call(houses, '/casas', query='scope=es')[2])
    assert '<link rel="canonical" href="https://demoscopio.es/casas?scope=es">' in ranking
    write_event(fresh_app.fs)
    detail = html(call(houses, '/elecciones/es/generales/2023-07-23', query='scope=es&mode=forecast')[2])
    assert '<link rel="canonical" href="https://demoscopio.es/elecciones/es/generales/2023-07">' in detail
    assert '<title>Demoscopio · Generales 2023</title>' in detail and 'og:title" content="Demoscopio · Generales 2023"' in detail


# ---------------------------------------------------------------------------------------------------------------------
# URLs limpias: /casas/{encuestadora} y /elecciones/{país}/{ámbito}/{año-mes} (Luis, 2026-10-15)


@pytest.mark.parametrize('text, slug', [
    ('GAD3', 'gad3'), ('Sigma Dos', 'sigma-dos'), ('Simple Lógica', 'simple-logica'), ('40dB', '40db'),
    ('Redondo & Asociados', 'redondo-asociados'), ('A+M', 'a-m'), ('Castilla y León', 'castilla-y-leon'),
    ('Castilla-La Mancha', 'castilla-la-mancha'), ('Tàstic', 'tastic'), ('  Obradoiro de Socioloxía ', 'obradoiro-de-socioloxia'),
    ('???', ''), (None, ''),
])
def test_slugify_lowers_to_ascii_and_joins_with_hyphens(text, slug):
    assert pages.slugify(text) == slug


def test_scope_slug_names_the_general_elections_and_the_regions_by_their_name():
    assert pages.scope_slug('es', 'España') == 'generales'
    assert pages.scope_slug('es-md', 'Madrid') == 'madrid'
    assert pages.scope_slug('es-cl', 'Castilla y León') == 'castilla-y-leon'


def test_event_href_uses_the_month_unless_another_election_of_the_scope_shares_it(history_two_scopes):
    assert pages.event_href('es', '2023-07-23') == '/elecciones/es/generales/2023-07'
    assert pages.event_href('es-md', '2021-05-04') == '/elecciones/es/madrid/2021-05'
    assert pages.event_href('es', '2023-07-23', dates=['2023-07-23', '2023-07-02']) == '/elecciones/es/generales/2023-07-23'


def test_resolve_event_path_finds_the_election_of_a_month_or_of_a_date(history_two_scopes, fresh_app):
    assert pages.resolve_event_path('es', 'generales', '2023-07') == ('es', '2023-07-23')
    assert pages.resolve_event_path('es', 'generales', '2023-07-23') == ('es', '2023-07-23')
    assert pages.resolve_event_path('es', 'madrid', '2021-05') == ('es-md', '2021-05-04')
    write_event(fresh_app.fs, date='2023-07-02', model=False)  # dos elecciones en el mismo mes: el mes ya no basta
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    assert pages.event_href('es', '2023-07-23') == '/elecciones/es/generales/2023-07-23'
    assert pages.resolve_event_path('es', 'generales', '2023-07-02') == ('es', '2023-07-02')
    with pytest.raises(HttpError) as ambiguous:
        pages.resolve_event_path('es', 'generales', '2023-07')
    assert ambiguous.value.status == 404 and str(ambiguous.value.message) == 'event not published'


@pytest.mark.parametrize('segments, status, message', [
    (('fr', 'generales', '2023-07'), 404, 'unknown scope'),
    (('es', 'marte', '2023-07'), 404, 'unknown scope'),
    (('es', 'generales', '2023-13'), 400, 'invalid date'),
    (('es', 'generales', '2023-7'), 400, 'invalid date'),
    (('es', 'generales', '2023-07-32'), 400, 'invalid date'),
    (('es', 'generales', '2020-01'), 404, 'event not published'),
    (('es', 'generales', '2023-07-24'), 404, 'event not published'),
    (('es', 'andalucia', '2022-06'), 404, 'scope not published'),
])
def test_resolve_event_path_errors(history_two_scopes, fresh_app, segments, status, message):
    write_event(fresh_app.fs, scope='es-an', date='2022-06-19', model=False)  # elecciones sin run: sin detalle
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    with pytest.raises(HttpError) as error:
        pages.resolve_event_path(*segments)
    assert error.value.status == status and str(error.value.message) == message


def test_pollster_href_slugs_the_name_and_falls_back_to_the_id():
    assert pages.pollster_href('GAD3', 2) == '/casas/gad3'
    assert pages.pollster_href('Simple Lógica', 5) == '/casas/simple-logica'
    assert pages.pollster_href('???', 9) == '/casas/9'


def test_resolve_pollster_finds_the_id_by_slug_or_by_number(houses, monkeypatch):
    assert pages.resolve_pollster('gad3') == 2 and pages.resolve_pollster('cis') == 1 and pages.resolve_pollster('3') == 3
    for slug in ('nadie', 'GAD3', '999', ''):
        with pytest.raises(HttpError) as missing:
            pages.resolve_pollster(slug)
        assert missing.value.status == 404 and str(missing.value.message) == 'pollster not found'
    # dos nombres con el mismo slug (ninguno en la base hoy): el slug lleva al id más bajo y el otro perfil sigue en su
    # URL numérica (los enlaces se construyen sin consultar la base, así que no pueden esquivar la colisión)
    twins = pd.DataFrame({'id': [7, 3], 'name': ['Opina', 'opina'], 'mtype': ['private'] * 2, 'quality': [50.0] * 2,
                          'rating': [None] * 2})
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: twins)
    webapi.queries._QUERIES.clear()
    assert pages.resolve_pollster('opina') == 3 and pages.resolve_pollster('7') == 7


def test_casa_profile_has_a_clean_url_and_the_old_one_redirects(houses):
    status, _, body = call(houses, '/casas/gad3')
    page = html(body)
    assert status == 200 and '<title>Demoscopio · GAD3</title>' in page and 'id="profile-effects"' in page
    assert initial(body)['path'][0]['event_date'] == '2021-05-04'  # el primer punto del camino
    status, headers, _ = call(houses, '/casas', query='id=2')
    assert status == 301 and headers['location'] == '/casas/gad3' and headers['cache-control'] == 'no-store'
    status, headers, _ = call(houses, '/casas', query='scope=es&id=2&mode=forecast')
    assert status == 301 and headers['location'] == '/casas/gad3?scope=es&mode=forecast'
    status, headers, body = call(houses, '/casas/nadie')
    assert status == 404 and 'Encuestadora no encontrada' in html(body) and headers['cache-control'] == 'no-store'


def test_casas_and_the_election_detail_link_the_profiles_by_slug(houses, fresh_app):
    page = html(call(houses, '/casas', query='scope=es')[2])
    ranking = re.search(r'<table[^>]*id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert '<td class="f l"><a href="/casas/gad3"><b>GAD3</b></a></td>' in ranking and '/casas?id=' not in page
    effects = page[page.index('id="house-effects"'):page.index('id="herding"')]
    assert re.search(r'href="/casas/[a-z0-9-]+"', effects) and '?id=' not in effects
    write_event(fresh_app.fs)
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    detail = html(call(houses, '/elecciones/es/generales/2023-07')[2])
    table = re.search(r'<table id="last-table">(.*?)</table>', detail, re.S).group(1)
    assert 'href="/casas/gad3"' in table and '/casas?id=' not in detail


def test_eleccion_detail_has_a_clean_url_and_the_old_one_redirects(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and '<title>Demoscopio · Generales 2023</title>' in page and 'id="last-table"' in page
    assert initial(body)['meta']['event_date'] == '2023-07-23'
    status, headers, _ = call(events_site, '/elecciones', query='scope=es&date=2023-07-23')
    assert status == 301 and headers['location'] == '/elecciones/es/generales/2023-07' and headers['cache-control'] == 'no-store'
    status, headers, _ = call(events_site, '/elecciones', query='date=2019-11-10&mode=forecast')
    assert status == 301 and headers['location'] == '/elecciones/es/generales/2019-11?mode=forecast'


@pytest.mark.parametrize('path, status, text', [
    ('/elecciones/es/marte/2023-07', 404, 'Ámbito desconocido.'),
    ('/elecciones/es/generales/2023-13', 400, 'Fecha no válida.'),
    ('/elecciones/es/generales/2020-01', 404, 'Elección no publicada.'),
    ('/elecciones/es/generales', 404, 'Página no encontrada.'),
])
def test_eleccion_detail_route_errors(events_site, path, status, text):
    got, headers, body = call(events_site, path)
    assert got == status and text in html(body) and headers['cache-control'] == 'no-store'


def test_history_and_home_link_the_elections_by_slug(history_two_scopes):
    status, _, body = call(history_two_scopes, '/elecciones')
    page = html(body)
    assert status == 200 and 'href="/elecciones/es/generales/2023-07"' in page and 'href="/elecciones/es/madrid/2021-05"' in page
    assert '/elecciones?scope=' not in page.replace('/elecciones?scope=es"', '')  # solo queda el enlace al histórico
    events = {event['key']: event['href'] for event in initial(body)['events']}
    assert events['es:2023-07-23'] == '/elecciones/es/generales/2023-07' and events['es-md:2021-05-04'] == '/elecciones/es/madrid/2021-05'
    home = html(call(history_two_scopes, '/')[2])
    past = home[home.index('id="past"'):]
    assert 'href="/elecciones/es/generales/2023-07"' in past and 'href="/elecciones/es/madrid/2021-05"' in past


# Seguimiento 2026-10-15 (recorrido de Luis): en el perfil, los cuatro listados van de la elección más nueva a la
# más antigua, la tarjeta de efectos se llama "Efectos de casa por elección" y encima va "Último sondeo por
# elección": cuánto se separó el último sondeo de la casa del resultado de cada partido o, con el conmutador, de
# la media de los últimos sondeos de las demás casas (los del histórico de errores).
LAST_ROWS = [
    {'scope': 'es', 'event_date': '2023-07-23', 'pollster': 'GAD3', 'days': 7},
    {'scope': 'es', 'event_date': '2023-07-23', 'pollster': 'Sigma Dos', 'days': 8},
    {'scope': 'es', 'event_date': '2023-07-23', 'pollster': 'CIS', 'days': 12},  # su sondeo del ciclo es del día 10
]
LAST_CYCLE = {'parties': ['PP', 'PSOE', 'VOX'], 'polls': [
    {'date': '2023-07-10', 'pollster': 'CIS', 'PP': 31.2, 'PSOE': 32.4, 'VOX': 14.0},
    {'date': '2023-07-15', 'pollster': '40dB', 'PP': 33.5, 'PSOE': 29.9, 'VOX': 13.1},  # sin fila en el análisis
    {'date': '2023-07-15', 'pollster': 'Sigma Dos', 'PP': 32.1, 'PSOE': 31.3, 'VOX': 12.0},
    {'date': '2023-07-16', 'pollster': 'GAD3', 'PP': 34.1, 'PSOE': 30.3, 'VOX': None},
    {'date': '2023-07-16', 'pollster': 'GAD3', 'PP': 36.0, 'PSOE': 28.0, 'VOX': 10.0},  # segundo del mismo día: no cuenta
]}
LAST_RESULTS = {'rows': [{'party': 'PP', 'pct': 33.06}, {'party': 'PSOE', 'pct': 31.44}, {'party': 'Cs', 'pct': 0.13}]}


def test_last_poll_records_measure_the_last_poll_against_the_result_and_the_other_houses():
    records = pages.last_poll_records('es', LAST_ROWS, LAST_CYCLE, LAST_RESULTS, 'GAD3')
    assert records == [
        {'event_scope': 'es', 'event_date': '2023-07-23', 'party': 'PP', 'result': 1.04, 'consensus': 2.0},
        {'event_scope': 'es', 'event_date': '2023-07-23', 'party': 'PSOE', 'result': -1.14, 'consensus': -1.0},
        {'event_scope': 'es', 'event_date': '2023-07-23', 'party': 'VOX', 'result': None, 'consensus': None},
    ]
    # para Sigma Dos la única otra casa con su último sondeo en el ciclo es GAD3, que no da VOX; VOX tampoco está en
    # el resultado publicado
    records = pages.last_poll_records('es', LAST_ROWS, LAST_CYCLE, LAST_RESULTS, 'Sigma Dos')
    assert [(row['party'], row['result'], row['consensus']) for row in records] == [
        ('PP', -0.96, -2.0), ('PSOE', -0.14, 1.0), ('VOX', None, None)]


def test_last_poll_records_without_the_poll_the_cycle_or_the_results():
    assert pages.last_poll_records('es', LAST_ROWS, LAST_CYCLE, LAST_RESULTS, 'CIS') == []  # no está en el ciclo
    assert pages.last_poll_records('es', LAST_ROWS, None, LAST_RESULTS, 'GAD3') == []
    assert pages.last_poll_records('es', [], LAST_CYCLE, LAST_RESULTS, 'GAD3') == []  # sin fila en el análisis
    records = pages.last_poll_records('es', LAST_ROWS, LAST_CYCLE, None, 'GAD3')
    assert [(row['result'], row['consensus']) for row in records] == [(None, 2.0), (None, -1.0), (None, None)]


def test_profile_rows_can_run_from_the_newest_event():
    records = [{'event_date': '2019-11-10', 'n': 1}, {'event_date': '2026-11-29', 'n': 2}, {'event_date': '2023-07-23', 'n': 3}]
    assert [row['n'] for row in pages.profile_rows(records, ('event_date', 'n'))] == [1, 3, 2]
    assert [row['n'] for row in pages.profile_rows(records, ('event_date', 'n'), newest_first=True)] == [2, 3, 1]


def test_effects_table_runs_from_the_newest_election():
    records = [{'event_date': '2019-11-10', 'event_scope': 'es', 'party': 'PP', 'dev_result_c': 0.4},
               {'event_date': '2023-07-23', 'event_scope': 'es-md', 'party': 'PP', 'dev_result_c': 0.2},
               {'event_date': '2023-07-23', 'event_scope': 'es', 'party': 'PP', 'dev_result_c': 1.5}]
    rows = pages.effects_table(records)['rows']
    assert [(row['event_date'], row['scope']) for row in rows] == [('2023-07-23', 'es'), ('2023-07-23', 'es-md'), ('2019-11-10', 'es')]


def test_casa_profile_lists_run_from_the_newest_election_to_the_oldest(houses, fresh_app):
    """Los cuatro listados del perfil van de la elección más nueva a la más antigua: el camino al rating (la próxima
    elección, la primera), la diana, los efectos (las dos vistas) y el rebaño; el gráfico del camino sigue en orden."""
    write_event(fresh_app.fs, date='2019-11-10')  # una segunda elección con su fila en el análisis: dos en la diana
    data = fixture_data('poll-errors')
    data['rows'].append(dict(data['rows'][0], event_date='2019-11-10', event='2019-10N', days=9))
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es', 'poll-errors'), 'poll-errors', data, 'es')
    status, _, body = call(houses, '/casas/gad3')
    page = html(body)
    assert status == 200
    path = re.search(r'<table id="path-table">(.*?)</table>', page, re.S).group(1)
    assert path.index('Generales 2023') < path.index('4 may 2021 · Madrid') and 'de la más nueva a la más antigua' in path
    assert 'Hoy' not in path  # el rating de hoy es el final de la última elección; lo marca el gráfico
    bullseye = re.search(r'<table id="bullseye-table">(.*?)</table>', page, re.S).group(1)
    assert bullseye.count('<tr>') == 3 and bullseye.index('23 jul 2023') < bullseye.index('10 nov 2019')
    effects = re.search(r'<table id="profile-effects">(.*?)</table>', page, re.S).group(1)
    assert effects.index('Generales 2023') < effects.index('4 may 2021 · Madrid')
    cycle = re.search(r'<table id="profile-effects-cycle">(.*?)</table>', page, re.S).group(1)
    assert cycle.index('29 nov 2026') < cycle.index('Generales 2023') < cycle.index('4 may 2021 · Madrid')
    herding = re.search(r'<table id="profile-herding">(.*?)</table>', page, re.S).group(1)
    assert herding.index('29 nov 2026') < herding.index('Generales 2023') < herding.index('4 may 2021 · Madrid')
    data = initial(body)
    assert [row['event_date'] for row in data['bullseye']] == ['2023-07-23', '2019-11-10']
    assert [row['event_date'] for row in data['path']] == ['2021-05-04', '2023-05-28', '2023-07-23']


def test_casa_profile_shows_the_last_poll_by_election_over_the_house_effects(houses, fresh_app):
    """Encima de "Efectos de casa por elección" (antes "de encuestadora"), la tarjeta del último sondeo: cuánto se
    separó de cada partido frente al resultado o, con el conmutador, frente a la media de los últimos sondeos de las
    demás casas; los sondeos son los del histórico de errores, leídos del ciclo publicado de cada elección."""
    data = fixture_data('event-cycle')
    data['polls'][1] = dict(data['polls'][1], pollster_id=4, pollster='Sigma Dos', PP=32.1, PSOE=31.3, VOX=12.0)
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_event('es', '2023-07-23', 'cycle'), 'event-cycle', data, 'es')
    page = html(call(houses, '/casas/gad3')[2])
    assert 'Efectos de casa por elección' in page and 'Efectos de encuestadora por elección' not in page
    assert page.index('id="bullseye"') < page.index('id="last"') < page.index('id="effects"') < page.index('id="herding"')
    card = page[page.index('id="last"'):page.index('id="effects"')]
    assert '<h2>Último sondeo por elección</h2>' in card
    assert '<fieldset id="last-view" class="seg" hidden>' in card
    assert 'name="lview" value="resultado" checked' in card and 'name="lview" value="consenso"' in card
    result = re.search(r'<table id="profile-last">(.*?)</table>', card, re.S).group(1)
    consensus = re.search(r'<table id="profile-last-consensus">(.*?)</table>', card, re.S).group(1)
    # VOX no tiene valor en ninguna fila: fuera de las columnas (2026-10-15)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', result) == ['Elección', 'PP', 'PSOE']
    assert re.findall(r'<td[^>]*>([^<]*)</td>', result) == ['Generales 2023', '1,0', '-1,1']
    assert re.findall(r'<td[^>]*>([^<]*)</td>', consensus) == ['Generales 2023', '2,0', '-1,0']
    # el fondo es más rojo cuanto mayor la desviación en cualquier sentido, hasta 8 puntos
    from mtpy.lib.pages.filters import shade
    assert '<td class="cell-shade" style="background-color: {}">-1,1</td>'.format(shade(1.14, 0, 8, '#d62728')) in result
    assert '<td class="cell-shade" style="background-color: {}">2,0</td>'.format(shade(2.0, 0, 8, '#d62728')) in consensus
    assert '<div class="table-wrap" id="profile-last-consensus-wrap" hidden>' in card
    assert 'sondeo menos resultado' in card and 'demás encuestadoras' in card and 'más rojo' in card
    # una casa sin sondeos en las elecciones analizadas: aviso en vez de tablas
    page = html(call(houses, '/casas/nueva')[2])
    card = page[page.index('id="last"'):page.index('id="effects"')]
    assert 'no tiene sondeos en las elecciones analizadas' in card and 'id="profile-last"' not in card


def test_casa_profile_last_poll_card_without_analysis(houses_without_analysis):
    page = html(call(houses_without_analysis, '/casas/gad3')[2])
    card = page[page.index('id="last"'):page.index('id="effects"')]
    assert 'Análisis no publicado todavía' in card and 'id="profile-last"' not in card


# Seguimiento 2026-10-15 (recorrido de Luis): "Camino al rating" rehecho. El gráfico es el rating con el que la casa
# llegó a cada elección que sondeó, de cualquier ámbito (las del mismo día, fundidas), de 0 a 100, hasta hoy; la
# tabla lleva las mismas filas, de la más nueva a la más antigua, con "Hoy" la primera, y «Qué mide cada columna»
# define justo sus columnas.
PATH_RATINGS = [
    {'event_scope': 'es', 'event_date': '2023-07-23', 'rating': 80.0, 'rating_adj': 80.5},
    {'event_scope': 'es', 'event_date': '2026-11-29', 'rating': 82.4, 'rating_adj': 83.1},
    {'event_scope': 'es-md', 'event_date': '2021-05-04', 'rating': 71.0, 'rating_adj': 70.2},
    {'event_scope': 'es-md', 'event_date': '2023-05-28', 'rating': 78.5, 'rating_adj': 77.9},
    {'event_scope': 'es-ar', 'event_date': '2023-05-28', 'rating': 78.5, 'rating_adj': 77.9},
    {'event_scope': 'es-md', 'event_date': '2027-05-23', 'rating': 82.4, 'rating_adj': 83.1},
]
PATH_EVENTS = [
    {'event_scope': 'es', 'event_date': '2023-07-23', 'n_polls': 14, 'error_avg': 1.3, 'error_blocks': 1.5, 'bias_dev_adj': -0.2},
    {'event_scope': 'es-md', 'event_date': '2021-05-04', 'n_polls': 7, 'error_avg': 2.1, 'error_blocks': -0.8, 'bias_dev_adj': -1.5},
    {'event_scope': 'es-md', 'event_date': '2023-05-28', 'n_polls': 3, 'error_avg': 1.8, 'error_blocks': 0.6, 'bias_dev_adj': -2.0},
    {'event_scope': 'es-ar', 'event_date': '2023-05-28', 'n_polls': 2, 'error_avg': 2.8, 'error_blocks': -0.4, 'bias_dev_adj': 0.5},
    {'event_scope': 'es-ct', 'event_date': '2024-05-12', 'n_polls': 4, 'error_avg': 1.1, 'error_blocks': None,
     'bias_dev_adj': None},  # sin rating
    {'event_scope': 'es-an', 'event_date': '2026-05-17', 'n_polls': 4, 'error_avg': 0.9, 'error_blocks': 0.1,
     'bias_dev_adj': -1.8},  # tras "hoy"
]
PATH_SCOPES = {'es': 'España', 'es-an': 'Andalucía', 'es-ar': 'Aragón', 'es-ct': 'Cataluña', 'es-md': 'Madrid'}


def path_label(scope, date):
    return 'Generales 2023' if (scope, date) == ('es', '2023-07-23') else '{} · {}'.format(pages.fmt_date(date), PATH_SCOPES[scope])


def test_rating_path_has_a_point_per_polled_election_date_up_to_today():
    path = pages.rating_path(PATH_RATINGS, PATH_EVENTS, path_label, PATH_SCOPES, list(PATH_SCOPES), date(2025, 1, 1))
    assert list(path) == ['points', 'today']
    # las medidas de una fecha con varios ámbitos: la media de sus sondeos, ponderada por ámbito; el rating es el que le
    # quedó tras ese evento (el del siguiente cálculo, que ya lo incluye) y el cambio, frente al que tenía al llegar (la
    # fila de ese evento o, sin ella, la última anterior)
    assert path['points'] == [
        {'event_date': '2021-05-04', 'scopes': ['es-md'], 'label': '4 may 2021 · Madrid', 'kind': 'regional', 'n_polls': 7,
         'error_avg': 2.1, 'error_blocks': -0.8, 'bias_dev_adj': -1.5, 'rating': 78.5, 'change': 7.5},
        {'event_date': '2023-05-28', 'scopes': ['es-ar', 'es-md'], 'label': '28 may 2023 · Aragón y Madrid', 'kind': 'regional',
         'n_polls': 5, 'error_avg': 2.2, 'error_blocks': 0.2, 'bias_dev_adj': -1.0, 'rating': 80.0, 'change': 1.5},
        {'event_date': '2023-07-23', 'scopes': ['es'], 'label': 'Generales 2023', 'kind': 'general', 'n_polls': 14,
         'error_avg': 1.3, 'error_blocks': 1.5, 'bias_dev_adj': -0.2, 'rating': 82.4, 'change': 2.4},
        {'event_date': '2024-05-12', 'scopes': ['es-ct'], 'label': '12 may 2024 · Cataluña', 'kind': 'regional', 'n_polls': 4,
         'error_avg': 1.1, 'error_blocks': None, 'bias_dev_adj': None, 'rating': 82.4, 'change': 2.4},
    ]
    # hoy: el rating de la próxima elección general, sin sondeos, medidas ni cambio
    assert path['today'] == {'event_date': '2025-01-01', 'scopes': [], 'label': 'Hoy', 'kind': 'today', 'n_polls': None,
                             'error_avg': None, 'error_blocks': None, 'bias_dev_adj': None, 'rating': 82.4, 'change': None}


def test_rating_path_names_many_scopes_and_falls_back_without_a_national_row():
    events = [{'event_scope': scope, 'event_date': '2015-05-24', 'n_polls': 1} for scope in ('es-an', 'es-ar', 'es-ct', 'es-md')]
    ratings = [{'event_scope': 'es-md', 'event_date': '2015-05-24', 'rating': 58.6, 'rating_adj': 39.8},
               {'event_scope': 'es-ct', 'event_date': '2028-06-26', 'rating': 70.0, 'rating_adj': 71.0}]
    path = pages.rating_path(ratings, events, path_label, PATH_SCOPES, list(PATH_SCOPES), date(2026, 10, 15))
    assert [point['label'] for point in path['points']] == ['24 may 2015 · Andalucía, Aragón y 2 más']
    assert path['points'][0]['n_polls'] == 4 and path['points'][0]['rating'] == 70.0  # el siguiente cálculo tras esa fecha
    assert path['points'][0]['change'] == 11.4  # frente al rating con el que llegó (58,6)
    assert path['today']['rating'] == 70.0 and path['today']['event_date'] == '2026-10-15'  # la última fila, sin nacional
    empty = pages.rating_path([], events, path_label, PATH_SCOPES, list(PATH_SCOPES), date(2026, 10, 15))
    assert empty['today'] is None and empty['points'][0]['rating'] is None and empty['points'][0]['change'] is None


def test_casa_profile_path_card_lists_every_polled_election_and_defines_its_columns(houses):
    status, _, body = call(houses, '/casas/gad3')
    page = html(body)
    assert status == 200 and 'id="rating-path"' in page
    card = page[page.index('id="path"'):page.index('id="bullseye"')]
    table = re.search(r'<table id="path-table">(.*?)</table>', card, re.S).group(1)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', table) == [
        'Evento', 'Sondeos', 'Diferencial', 'Error medio', 'Sesgo bloques', 'Rating', 'Cambio']
    cells = lambda row: [re.sub(r'<[^>]+>', '', cell).strip() for cell in re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)]
    trs = re.findall(r'<tr>(.*?)</tr>', table, re.S)[1:]
    assert [cells(row) for row in trs] == [['Generales 2023', '14', '-0,2', '1,3', '1,5', '82,4', '2,4'],
                                            ['28 may 2023 · Aragón y Madrid', '5', '-1,0', '2,2', '0,2', '80,0', '1,5'],
                                            ['4 may 2021 · Madrid', '7', '-1,5', '2,1', '-0,8', '78,5', '7,5']]
    # los formatos (2026-10-17): barra divergente del diferencial en ±2 (verde a la izquierda, roja a la derecha), rojos
    # del error medio de 0 a 8, barra divergente del sesgo de bloques en ±8 en el color del bloque al que favorece (la
    # fixture solo trae el primero, azul; el segundo, gris) y doble gradiente del cambio (rojo de -10 a 0, verde de 0 a 10)
    from mtpy.lib.pages.filters import bar_sides, shade
    first = trs[0]
    assert '<td class="cell-shade" style="background-color: {}">1,3</td>'.format(shade(1.3, 0, 8, '#d62728')) in first
    assert '<td class="cell-bar cell-diverging"><i class="fill pos" style="left: 45.0%; width: 5.0%"></i><span>-0,2</span></td>' in first
    bias = bar_sides(1.5, 8)
    assert '<td class="cell-bar cell-diverging"><i class="fill block" style="left: {}%; width: {}%; background: #1d84ce"></i><span>1,5</span></td>'.format(bias['left'], bias['width']) in first
    assert '<td class="cell-shade" style="background-color: {}">2,4</td>'.format(shade(2.37, 0, 10, '#2ca02c')) in first
    bias = bar_sides(-0.8, 8)
    assert '<i class="fill block" style="left: {}%; width: {}%; background: #9e9e9e"></i><span>-0,8</span>'.format(bias['left'], bias['width']) in trs[2]
    glossary = re.search(r'<details class="how" id="glosario">(.*?)</details>', card, re.S).group(1)
    assert re.findall(r'<dt>([^<]*)</dt>', glossary) == ['Evento', 'Sondeos', 'Diferencial', 'Error medio', 'Sesgo bloques', 'Rating', 'Cambio']
    assert 'a Derecha' in glossary and 'sube el rating' in glossary and 'Error de bloques' not in card and 'Rebaño' not in card
    assert header_tooltips(table) == glossary_entries(glossary)  # el tooltip de cada cabecera es su entrada del glosario
    assert 'de 0 a 100' in card and 'hasta hoy' in card
    data = initial(body)
    assert [(point['event_date'], point['kind'], point['n_polls']) for point in data['path']] == [
        ('2021-05-04', 'regional', 7), ('2023-05-28', 'regional', 5), ('2023-07-23', 'general', 14)]
    assert data['today']['rating'] == 82.37 and data['today']['label'] == 'Hoy'


def test_effects_table_drops_the_parties_without_any_value():
    records = [{'event_date': '2023-07-23', 'event_scope': 'es', 'party': 'PP', 'dev_result_c': 0.4, 'dev_cycle': 0.1},
               {'event_date': '2023-07-23', 'event_scope': 'es', 'party': 'VOX', 'dev_result_c': None, 'dev_cycle': -0.3},
               {'event_date': '2019-11-10', 'event_scope': 'es', 'party': 'VOX', 'dev_result_c': None, 'dev_cycle': None}]
    result = pages.effects_table(records)
    assert result['columns'] == ['PP'] and [row['cells'] for row in result['rows']] == [[0.4]]
    cycle = pages.effects_table(records, key='dev_cycle')
    assert cycle['columns'] == ['PP', 'VOX'] and [row['cells'] for row in cycle['rows']] == [[0.1, -0.3]]


def test_casas_glossary_explains_every_ranking_column_in_order(houses):
    """«Qué mide cada columna» del ranking: las siete columnas de la tabla, en su orden, con la regla de puntos del
    la confianza leída del catálogo; el glosario del perfil (camino al rating) no cambia."""
    page = html(call(houses, '/casas')[2])
    glossary = re.search(r'<details class="how" id="glosario">(.*?)</details>', page, re.S).group(1)
    assert re.findall(r'<dt>([^<]*)</dt>', glossary) == ['Encuestadora', 'Rating', 'Confianza', 'Elecciones', 'Sondeos',
                                                          'Diferencial', 'Error medio', 'Sesgo bloques', 'Rebaño']
    table = re.search(r'<table class="et et-rank" id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert header_tooltips(table) == glossary_entries(glossary)  # el tooltip de cada cabecera es su entrada del glosario
    assert 'entra en el promedio de la próxima elección' in glossary  # el rating del ranking es el de hoy
    # la fixture solo trae un bloque en `vs`: el segundo queda sin nombre (y su barra, gris)
    assert 'más margen a Derecha del que hubo; negativo, más margen al segundo' in glossary
    assert 'Cada punto vale 8 de confianza, con un mínimo de 5.' in glossary
    assert 'cada ámbito y fecha por separado' in glossary  # elecciones
    assert 'todas se explican en «Qué mide cada columna»' in re.sub(r'\s+', ' ', page)
    profile = html(call(houses, '/casas/gad3')[2])
    profile_glossary = re.search(r'<details class="how" id="glosario">(.*?)</details>', profile, re.S).group(1)
    assert re.findall(r'<dt>([^<]*)</dt>', profile_glossary) == ['Evento', 'Sondeos', 'Diferencial', 'Error medio',
                                                                  'Sesgo bloques', 'Rating', 'Cambio']


def test_casas_ranking_shows_the_differential_and_the_block_bias_with_their_bars(houses):
    """Ranking: el diferencial con la barra del perfil (negativo en verde, menos sesgo que las demás, en ±2) y el sesgo
    de bloques con la barra divergente en ±8 pintada con el color del bloque al que favorece (positivo, el primero:
    Derecha, azul; negativo, el segundo, que la fixture no define: gris)."""
    from mtpy.lib.pages.filters import bar_sides
    page = html(call(houses, '/casas')[2])
    start = page.index('id="ranking-table"')
    table = page[start:page.index('</table>', start)]
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.S)[1:]
    cis = next(row for row in rows if '>CIS<' in row)
    gad3 = next(row for row in rows if '>GAD3<' in row)
    assert '<i class="fill neg" style="left: 50.0%; width: 5.0%"></i><span>0,2</span>' in cis
    assert '<i class="fill pos" style="left: 25.0%; width: 25.0%"></i><span>-1,0</span>' in gad3
    bias = bar_sides(4.1, 8)
    assert '<i class="fill block" style="left: {}%; width: {}%; background: #1d84ce"></i><span>4,1</span>'.format(bias['left'], bias['width']) in cis
    bias = bar_sides(-3.0, 8)
    assert '<i class="fill block" style="left: {}%; width: {}%; background: #9e9e9e"></i><span>-3,0</span>'.format(bias['left'], bias['width']) in gad3
    assert 'class="help"' not in page


def test_block_bar_colors_follow_the_two_blocks_of_the_margin():
    """Los colores de la barra del sesgo de bloques: el del primer bloque de `vs` para el lado positivo, el del segundo
    para el negativo, gris cuando falta el bloque o la barra está vacía."""
    meta = {'bmaps': {'vs': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}},
            'parties': [{'name': 'PP', 'color': '#1d84ce'}, {'name': 'VOX', 'color': '#63be21'},
                        {'name': 'PSOE', 'color': '#ef1c27'}]}
    blocks, colors = pages.block_bar_colors(meta, pages.catalog(meta))
    assert blocks == ['Derecha', 'Izquierda']
    assert colors == {'pos': '#1d84ce', 'neg': '#ef1c27', 'zero': pages.OTHERS_COLOR}
    blocks, colors = pages.block_bar_colors({'bmaps': {'vs': {'Derecha': ['PP']}}}, pages.catalog(meta))
    assert blocks == ['Derecha'] and colors['pos'] == '#1d84ce' and colors['neg'] == pages.OTHERS_COLOR
    assert pages.block_bar_colors({}, pages.catalog(meta)) == ([], {'pos': pages.OTHERS_COLOR, 'neg': pages.OTHERS_COLOR,
                                                                   'zero': pages.OTHERS_COLOR})
