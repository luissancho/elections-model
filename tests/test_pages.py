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
    PARTIES, ROOT, RUN, clear_queries, fixture_data, use_database, write_analysis, write_backtest, write_bundle, write_event,
    write_parties, write_study
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
    """{etiqueta: tooltip} de las cabeceras de una tabla, con los espacios normalizados (la etiqueta sin su segunda
    línea, `th-sub`)."""
    import html as html_mod
    return {label: re.sub(r'\s+', ' ', html_mod.unescape(title)).strip()
            for title, label in re.findall(r'<th scope="col"[^>]*title="([^"]*)"[^>]*>([^<]*)(?:<span class="th-sub">[^<]*</span>)?</th>',
                                           table)}


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
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs', 'cards', 'inks', 'deltas', 'catalog'}
    assert data['catalog'] == {'parties': [dict(pages.catalog(data['meta'])['PP'], name='PP')], 'bmaps': data['meta']['bmaps']}
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
        assert {'date', 'pollster_id', 'pollster', 'name', 'sample_size', 'href'} | parties <= set(poll)
        extra = {'sponsor', 'start_date', 'end_date', 'seats', 'rating'}
        assert set(poll) <= {'date', 'pollster_id', 'pollster', 'name', 'sample_size', 'href'} | extra | parties
        assert poll['href'] == pages.pollster_href(poll['pollster'], poll['pollster_id'])


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
    assert list(data) == ['state', 'meta', 'summary', 'dist', 'fan', 'districts', 'scenario', 'runs', 'deltas']
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
    assert [l['label'] for l in links] == ['Resumen por partido y bloque', 'Escaños por simulación', 'Circunscripciones', 'Reparto central', 'Abanico por horizonte']
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
    assert page.count('type="hidden" name="run" value="{}"'.format(RUN)) == 1  # sólo el formulario de circunscripción
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
    assert pages.result_rows(summary) == [{'name': 'PP', 'pct': 32.66, 'seats': 139, 'seats_lo': None, 'seats_hi': None, 'p_first': 0.95,
                                           'prev_pct': None, 'prev_seats': None, 'delta_pct': None, 'delta_seats': None, 'prev_from': []}]
    rows = pages.block_result_rows(summary, fixture_data('meta'), pages.Catalog(fixture_data('meta')))
    assert [r['name'] for r in rows] == ['Derecha', 'Izquierda'] and rows[0]['p_majority'] == 0.995 and rows[0]['color'].startswith('#')
    assert pages.result_rows({}) == [] and pages.block_result_rows({}, fixture_data('meta'), pages.Catalog(fixture_data('meta'))) == []


def test_promedios_central_block_is_three_cards_with_ranges(api):
    page = html(call(api, '/es')[2])
    order = [page.index(m) for m in ('id="series"', 'id="hemicycle-card"', 'id="parties-card"', 'id="blocks-card"', 'id="evolution-card"')]
    assert order == sorted(order)
    heads = re.findall(r'<th scope="col">([^<]*)</th>', page[page.index('id="parties-table"'):page.index('id="blocks-result-table"')])
    assert heads == ['Partido', '% Voto', 'Escaños', 'Rango 95%']  # the probabilities live in Escaños (2026-10-17)
    assert re.findall(r'<th scope="col">([^<]*)</th>', page[page.index('id="blocks-result-table"'):page.index('id="evolution-card"')]) == ['Bloque', '% Voto', 'Escaños', 'Rango 95%']
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
    assert set(data) == {'state', 'meta', 'headline', 'series', 'polls', 'projection', 'vote', 'summary', 'runs', 'cards', 'inks', 'deltas', 'catalog'}
    assert all(abs(v * 10 - round(v * 10)) < 1e-9 for v in data['series']['mean']['PP'] if v is not None)
    assert 'pollster_id' in data['polls']['polls'][0]


def test_promedios_controls_are_hidden_without_js(api):
    """Sin JS los controles no hacen nada y no se muestran."""
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    for control in ('series-window', 'series-view', 'series-layers'):
        assert re.search(r'<fieldset id="{}" class="[^"]*" hidden>'.format(control), page), control
    # «Mostrar»: sólo los sondeos, sólo el promedio o ambos, marcado «Ambos» por defecto (Luis, 2026-10-23)
    layers = page[page.index('id="series-layers"'):]
    layers = layers[:layers.index('</fieldset>')]
    assert '<legend>Mostrar</legend>' in layers
    assert re.findall(r'<label><input type="radio" name="layers" value="(\w+)"( checked)?> (\w+)</label>', layers) == [
        ('sondeos', '', 'Sondeos'), ('promedio', '', 'Promedio'), ('ambos', ' checked', 'Ambos')]
    assert '«Sin sondeos»' not in page and 'Sondeos / Promedio / Ambos' in page
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
    # dos decimales: con uno, el eje estrecho del gráfico dibujaba la media en escalones (2026-10-19)
    assert out == {'dates': series['dates'], 'parties': ['PP'], 'mean': {'PP': [33.15, None]},
                   'lo': {'PP': [31.04, None]}, 'hi': {'PP': [35.26, 36.0]}}
    assert pages.round_series(series, 1)['mean'] == {'PP': [33.1, None]}
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
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': [name, 'GAD3', 'Hamalgama', 'Nueva', 'Cero', 'Agregador', 'Autonomica', 'Anonima'],
        'mtype': ['state', None, None, None, 'online', 'aggr', None, None],  # los valores de la base de datos; sin tipo, una casa privada
        'quality': [56.0, 80.0, 40.0, 5.0, 5.0, 16.0, 32.0, 24.0], 'rating': [55, 70, None, None, 50, None, None, None],  # rating nacional truncado: distinto del camino del ámbito
    })
    ratings = pd.DataFrame({
        'event_date': pd.to_datetime(['2019-11-10', '2023-07-23', '2023-07-23', '2026-11-29', '2026-11-29', '2026-11-29',
                                      '2026-11-29']),
        'pollster_id': [1, 1, 2, 1, 2, 3, 5], 'pollster': [name, name, 'GAD3', name, 'GAD3', 'Hamalgama', 'Cero'],
        'quality': [60.0, 60.0, 80.0, 60.0, 80.0, 40.0, 50.0], 'num_events': [1, 2, 1, 3, 2, 0, 0],
        'num_national_events': [1, 1, 1, 2, 3, 0, 0],
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
        'pollster_id': [1, 3, 6, 7, 8], 'pollster': [name, 'Hamalgama', 'Agregador', 'Autonomica', 'Anonima'],
        'fieldwork': [3, 2, 0, 2, 2],
        'fieldwork_source': ['https://www.cis.es/es/metodologia',
                             'https://hamalgama.example/ficha (ficha de 2026-10-01); https://hamalgama.example/metodologia', '',
                             'web', 'web'],
        'transparency': [3, 1, 1, 1, 1],
        'transparency_source': ['https://www.cis.es/es/microdatos', 'Ficha técnica del sondeo de 2026-10-01',
                                'https://agregador.example/metodo', 'web', 'web'],
        'independence': [0, 1, 0, 1, 1],
        'independence_source': ['', 'base de datos del proyecto, consulta de patrocinadores a 2026-10-16', '', 'web', 'web'],
        'identity': [1, 1, 1, 1, 0],
        'identity_source': ['https://www.cis.es', 'https://hamalgama.example', 'https://agregador.example', 'web', ''],
        'standards': [0, 0, 0, 0, 0], 'standards_source': ['', '', '', '', ''],
        'notes': ['Organismo público adscrito al Ministerio de la Presidencia', '', '', '', ''],
        'status': ['revisada', 'borrador', 'revisada', 'revisada', 'revisada'],
        'checked_at': pd.to_datetime(['2026-10-16', '2026-10-16', '2026-10-01', '2026-10-16', '2026-10-16']),
        'points': [7, 5, 2, 4, 3], 'quality': [56.0, 40.0, 16.0, 32.0, 24.0],
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
    (fixtures): la diana del perfil sólo cuenta las elecciones publicadas, como el histórico."""
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
    assert 'Cero' not in ranking and ranking.count('<tr') == 3 and 'id="ranking-more"' not in page
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
    assert ranking.count('<tr') == 61 and ranking.count(' hidden>') == 10  # cabecera + 50 visibles, 10 ocultas
    assert ranking.index('P50') < ranking.index(' hidden>') < ranking.index('P51')
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


def test_casas_with_analysis_without_house_effects_shows_the_notice(houses, fresh_app):
    """Un ámbito con análisis (`meta`, rebaño) pero sin `house-effects` (ciclo sin sondeos) da 200 y el aviso de la
    tarjeta de efectos, sin mapa de calor ni enlace al CSV (que daría 404)."""
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.remove(bundle.path_analysis('es', 'house-effects'))
    writer.remove(bundle.path_analysis_csv('es', 'house-effects'))
    status, _, body = call(houses, '/casas', query='scope=es')
    page = html(body)
    assert status == 200 and 'Análisis no publicado todavía' in page
    assert 'id="effects-heatmap"' not in page and 'id="herding-table"' in page
    assert initial(body)['grid'] is None
    assert 'Efectos de casa por partido y bloque' not in page and 'Rebaño por encuestadora' in page


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
    assert 'Hamalgama' not in ranking and ranking.count('<tr') == 3  # cabecera, GAD3 y CIS
    assert '82,4' in ranking and '0,74' in ranking  # rating (escala 0-100) y herding (análisis) de GAD3
    # los formatos (2026-10-17): barra verde del rating de 0 a 100, barra divergente del sesgo de bloques en ±8 en el color
    # del bloque al que favorece (la fixture sólo trae el primero: el segundo, gris), rojos del error medio de 0 a 8 y
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
    # los filtros de la diana: ámbito (sólo los que tienen sondeos suyos) y años, ocultos hasta que el JS los cablea
    form = re.search(r'<form id="bullseye-filter"[^>]*hidden>(.*?)</form>', page, re.S).group(1)
    years = [str(year) for year in range(2023, date.today().year + 1)]  # todos los años desde el primer evento hasta hoy (2026-10-17)
    assert re.findall(r'<option value="([^"]*)"', form) == ['', 'es'] + years + years and 'id="bullseye-count"' in form
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
    assert status == 200 and '<h2>Nueva' in page and page.replace('\n', '').count('<div class="v">–<small></small></div>') == 1  # sólo el rating sin valor
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
    assert list(rows[0]) == ['pollster_id', 'pollster', 'rating', 'quality', 'num_polls', 'num_events', 'num_national_events',
                             'error_avg', 'error_blocks', 'bias_blocks', 'bias_dev_adj', 'herding']
    assert rows[0]['num_events'] is None and rows[0]['num_national_events'] is None and pages.ranking_rows(pd.DataFrame(), None) == []
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
    assert '<div class="l">Tipo</div><div class="v">Privada<small></small></div>' in flat and '>70,' not in flat  # sin tipo: «Privada» (2026-10-17)
    assert '<div class="v">Pública<small></small></div>' in html(call(houses, '/casas/cis')[2]).replace('\n', '')  # `mtype` con su etiqueta en castellano


def test_casa_profile_quality_card_lists_the_rubric_with_sources(houses):
    page = html(call(houses, '/casas/cis')[2])
    kpis = page[page.index('id="profile"'):page.index('id="quality"')]
    assert '<div class="v">56,0<small></small></div>' in kpis.replace('\n', '') and '7 de 10 puntos' in kpis
    card = page[page.index('id="quality"'):page.index('id="path"')]
    rows = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>(\d) / (\d)</td>', card)
    assert [r[0] for r in rows] == ['Trabajo de campo', 'Transparencia', 'Independencia', 'Implantación']
    assert [(r[1], r[2]) for r in rows] == [('3', '3'), ('3', '3'), ('0', '2'), ('1', '2')]
    assert '<td>–</td>' in card  # Independencia, sin fuente
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
    assert 'rúbrica pública de cuatro criterios' in plain['quality']
    assert 'Cada punto vale' not in plain['quality'] and 'al primer bloque del que hubo; negativo, más margen al segundo' in plain['error_blocks']
    scaled = {column['key']: column['text']
              for column in pages.table_columns('ranking', ['Derecha', 'Izquierda'], {'per_point': 9, 'floor': 4})}
    assert scaled['quality'].endswith('Cada punto vale 9 de confianza, con un mínimo de 4.')
    assert 'más margen a Derecha del que hubo; negativo, más margen a Izquierda.' in scaled['error_blocks']
    assert [column['label'] for column in pages.table_columns('ranking')] == [
        'Encuestadora', 'Rating', 'Confianza', 'Elecciones', 'Sondeos', 'Diferencial', 'Error medio', 'Sesgo bloques', 'Rebaño']
    assert [column.get('sub') for column in pages.table_columns('ranking')][3] == '(generales)'
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
    assert 'href="https://hamalgama.example/ficha (' not in card


def test_casa_profile_quality_card_of_an_unrated_house(houses):
    page = html(call(houses, '/casas/nueva')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'Sin evaluar' in card and 'confianza mínima, 5' in card and 'href="/metodo/rating#calidad-previa"' in card
    assert '<th scope="row">' not in card and 'sin evaluar' in page[page.index('id="profile"'):page.index('id="quality"')]


def test_casa_profile_quality_card_of_an_unrated_house_with_national_events(houses, monkeypatch):
    """Sin evaluar pero con sondeos de generales: el punto de Implantación calculado da 8, no el mínimo 5, y el texto
    de la tarjeta y el KPI cuentan lo mismo."""
    pollsters = webapi.queries.get_pollsters().copy()
    pollsters.loc[pollsters['name'] == 'Nueva', 'quality'] = 8.0
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: pollsters)
    webapi.queries._QUERIES.clear()
    page = html(call(houses, '/casas/nueva')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'Sin evaluar' in card and 'alguna elección general' in card and 'confianza 8' in card
    assert 'confianza mínima' not in card and '<th scope="row">' not in card
    kpis = page[page.index('id="profile"'):page.index('id="quality"')]
    assert '>8,0<' in kpis and 'sin evaluar' in kpis


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
                                            ('Implantación', '1')]
    assert 'href="https://agregador.example/metodo"' in card and '2 de 10 puntos' in card
    assert 'la confianza es informativa' in card and 'Sin evaluar' not in card


def test_profile_confidence_card_shows_implantation_with_its_two_parts(houses):
    """Implantación: las generales se calculan de los sondeos (sin fuente) y los estándares llevan la suya."""
    from mtpy.lib.pages.casas import quality_card
    from mtpy.lib.quality import load_criteria
    page = html(call(houses, '/casas/cis')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'calculado a partir de los sondeos publicados' in card and 'Ha sondeado al menos una elección general.' in card
    criteria = load_criteria()
    record = {'fieldwork': 2, 'fieldwork_source': 'a', 'transparency': 2, 'transparency_source': 'b', 'independence': 1,
              'independence_source': 'c', 'identity': 1, 'identity_source': 'd', 'standards': 1,
              'standards_source': 'https://iaespana.example/socios', 'points': 7, 'quality': 56.0, 'status': 'revisada'}
    full = quality_card(record, {'mtype': 'private'}, criteria)
    row = full['rows'][-1]
    assert [r['name'] for r in full['rows']][-1] == 'Implantación' and (row['points'], row['max']) == (2, 2)
    assert row['national'] is True and row['standards'] is True and row['parts'][0]['url'] == 'https://iaespana.example/socios'
    none = quality_card(dict(record, points=5, standards_source=''), {'mtype': 'private'}, criteria)
    row = none['rows'][-1]
    assert (row['points'], row['national']) == (0, False) and row['level'] == criteria['criteria']['implantation']['levels']['0']
    assert none['support'] == '5 de 10 puntos'


def test_profile_page_card_shows_implantation_0_without_national_events(houses):
    """Una encuestadora evaluada sin generales: Implantación 0 de 2, con el nivel 0 y sin generales sondeadas."""
    page = html(call(houses, '/casas/autonomica')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    rows = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>(\d) / (\d)</td>', card)
    assert rows[-1] == ('Implantación', '0', '2') and 'Sólo ha sondeado elecciones autonómicas o locales.' in card
    assert 'ninguna elección general sondeada.' in card and 'calculado a partir de los sondeos publicados' not in card


def test_profile_page_card_voids_transparency_without_identity(houses):
    """Con identity = 0 y transparencia puntuada, la fila cuenta 0 de 3 y explica la condición."""
    page = html(call(houses, '/casas/anonima')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    rows = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>(\d) / (\d)</td>', card)
    assert ('Transparencia', '0', '3') in rows and 'Sin una entidad identificable detrás' in card


def test_profile_confidence_card_voids_transparency_without_identity(houses):
    """Sin entidad identificable la transparencia cuenta 0 y la fila explica la condición."""
    from mtpy.lib.pages.casas import quality_card
    from mtpy.lib.quality import load_criteria
    record = {'fieldwork': 2, 'fieldwork_source': 'a', 'transparency': 1, 'transparency_source': 'b', 'independence': 1,
              'independence_source': 'c', 'identity': 0, 'identity_source': '', 'standards': 0, 'standards_source': '',
              'points': 3, 'quality': 24.0, 'status': 'revisada'}
    rows = {r['key']: r for r in quality_card(record, {'mtype': 'private'}, load_criteria())['rows']}
    assert (rows['transparency']['points'], rows['transparency']['max'], rows['transparency']['void']) == (0, 3, True)
    assert rows['transparency']['level'].startswith('Sin una entidad identificable detrás')
    assert rows['fieldwork']['void'] is False


def test_metodo_rubric_lists_four_criteria_and_their_notes(api, fresh_app):
    """/metodo/rating lista los cuatro criterios del JSON con sus notas, sin «Identidad» ni «Estándares»."""
    from mtpy.lib.pages.metodo import quality_rubric
    rubric = quality_rubric()
    assert [r['name'] for r in rubric['rows']] == ['Trabajo de campo', 'Transparencia', 'Independencia', 'Implantación']
    notes = [r['note'] for r in rubric['rows']]
    assert notes[0] is None and notes[1].startswith('Sin una entidad identificable') and notes[3].startswith('Las elecciones generales')
    page = html(call(api, '/metodo/rating')[2])
    rubric_html = page[page.index('<th scope="col">Criterio</th>'):]
    assert 'Sin una entidad identificable detrás' in rubric_html and 'Las elecciones generales se cuentan con los sondeos' in rubric_html
    names = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>0-\d</td>', rubric_html)
    assert names == ['Trabajo de campo', 'Transparencia', 'Independencia', 'Implantación']
    assert 'Identidad' not in names and 'Estándares' not in names


def test_pages_package_layout():
    """Un módulo por página, `common` y `filters`; ninguno tapado por un nombre exportado (D1)."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(pages.__path__))
    assert names == ['analisis', 'backtest', 'casas', 'cera', 'common', 'elecciones', 'escanos', 'filters', 'inicio',
                     'metodo', 'promedios', 'slugs']
    for name in names:
        assert inspect.ismodule(getattr(pages, name)), name
    assert pages.fmt_pct.__module__ == 'mtpy.lib.pages.filters'
    assert pages.common_context.__module__ == 'mtpy.lib.pages.common'
    assert pages.casas_context.__module__ == 'mtpy.lib.pages.casas'
    assert pages.cera_context.__module__ == 'mtpy.lib.pages.analisis'
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
        'Elección', 'Sondeos', 'Encuestadoras', 'Error medio', 'Sesgo bloques']  # sin «Error de bloques» ni «Modelo» (2026-10-17)


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


def test_elecciones_links_the_elections_of_scopes_without_a_run(history_two_scopes, fresh_app):
    """Un ámbito con elecciones publicadas pero sin previsión (recién votado, o con pocos sondeos) también enlaza
    el detalle de sus elecciones (2026-10-20)."""
    write_event(fresh_app.fs, scope='es-an', date='2022-06-19', model=False)  # elecciones y análisis, sin run
    write_analysis(fresh_app.fs, scope='es-an')
    errors = fixture_data('poll-errors')
    errors['rows'] = [dict(row, event_date='2022-06-19', event='2022-19J') for row in errors['rows']]
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_analysis('es-an', 'poll-errors'),
                                                 bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es-an')
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    scopes = pages.history_scopes()
    assert [s['code'] for s in scopes] == ['es', 'es-an', 'es-md'] and all('linkable' not in s for s in scopes)
    hrefs = {(e['scope'], e['date']): e['href'] for e in pages.history_events(scopes)}
    assert hrefs[('es-an', '2022-06-19')] == '/elecciones/es/andalucia/2022-06'
    assert hrefs[('es-md', '2021-05-04')] == '/elecciones/es/madrid/2021-05'
    body = call(history_two_scopes, '/elecciones')[2]
    row = re.search(r'<tr data-scope="es-an"[^>]*>\s*<td>(.*?)</td>', html(body), re.S).group(1)
    assert 'href="/elecciones/es/andalucia/2022-06"' in row


def test_election_detail_of_a_scope_without_a_run_uses_the_national_run(history_two_scopes, fresh_app):
    """El detalle de una elección de un ámbito sin previsión se monta con el run nacional (catálogo, pie y
    menús); los datos y los bloques son los de la elección."""
    write_event(fresh_app.fs, scope='es-an', date='2022-06-19', model=False)
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    status, _, body = call(history_two_scopes, '/elecciones/es/andalucia/2022-06')
    assert status == 200
    page = html(body)
    assert re.search(r'<link rel="canonical" href="[^"]*/elecciones/es/andalucia/2022-06">', page)
    data = initial(body)
    assert data['meta']['scope'] == 'es-an' and data['meta']['event_date'] == '2022-06-19'
    assert data['state']['scope'] == 'es'  # la parte común es la nacional
    # las descargas son las de la elección, y los bloques llevan color aunque el ámbito no tenga run
    assert 'href="/api/v1/events/es-an/2022-06-19/cycle?format=csv"' in page
    assert 'href="/api/v1/events/es/' not in page
    with_run = initial(call(history_two_scopes, '/elecciones/es/madrid/2021-05')[2])
    assert data['block_colors'] == with_run['block_colors'] and data['block_colors']['Derecha'] != pages.OTHERS_COLOR
    # una fecha que el ámbito no tiene publicada sigue siendo 404
    assert call(history_two_scopes, '/elecciones/es/andalucia/2018-12')[0] == 404


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
    assert page.count('/api/v1/events/es/2023-07-23/') == 8 + 2  # las descargas (la serie incluida) y el CSV de la tabla (control y noscript)
    table = re.search(r'<table id="last-table">(.*?)</table>', page, re.S).group(1)
    assert table.index('GAD3') < table.index('40dB') < table.index('CIS')  # de menor a mayor error de bloques
    assert [re.sub(r'<.*?>', '', th) for th in re.findall(r'<th scope="col">(.*?)</th>', table)] == [
        'Encuestadora', 'Muestra', 'Días antes', 'Error medio', 'Sesgo bloques', 'Izquierda', 'Derecha']  # Izquierda a la izquierda (2026-10-17)
    assert 'href="/casas/gad3"' in table and 'href="/elecciones?scope=es"' in page
    assert 'id="pollsters-table"' not in page  # ya no hay tabla de medias por casa
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
    assert set(data) == {'state', 'meta', 'results', 'last', 'blocks', 'split', 'block_colors', 'poll_errors', 'catalog', 'polls',
                         'series', 'result', 'model_last', 'inks', 'model_point'}
    assert [row['pollster'] for row in data['last']] == ['GAD3', '40dB', 'CIS']
    assert data['last'][0] == {'pollster': 'GAD3', 'pollster_id': 2, 'days': 7, 'sample_size': 3000, 'error_avg': 0.99,
                               'error_blocks': 1.36, 'error_blocks_abs': 1.36, 'Derecha': 0.68, 'Izquierda': -0.68}
    # el hemiciclo del resultado oficial y el reparto del error por bloque (Izquierda a la izquierda)
    rows = data['results'].pop('rows')
    assert data['results'] == {'n_seats': 350, 'majority': 176, 'blocks': [{'name': 'Derecha', 'seats': 200}, {'name': 'Izquierda', 'seats': 150}]}
    assert rows[0] == {'name': 'PP', 'seats': 160, 'pct': 33.06} and all('pct' in row for row in rows)
    assert [(row['name'], row['seats']) for row in rows] == [('PP', 160), ('PSOE', 150), ('VOX', 40), ('Cs', 0)]
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
        < page.index('id="series"') < page.index('id="model"') < page.index('class="split"') < page.index('id="last-errors"') \
        < page.index('id="last-split"') < page.index('id="bullseye"') < page.index('id="last"') < page.index('id="glosario"') \
        < page.index('id="poll-histogram"') < page.index('id="polls"') < page.index('id="downloads"')


def test_catalog_adds_the_published_parties_behind_the_run():
    """El catálogo del run manda y conserva su orden; detrás van los partidos publicados que falten, con el gris de
    «otros» para un color malformado; un nombre ausente de ambos sigue cayendo en `__missing__`."""
    meta = {'parties': [{'name': 'PP', 'fullname': 'Partido Popular', 'color': '#1d84ce', 'block': 'Derecha'}]}
    rows = [{'id': 2, 'name': 'PP', 'fullname': 'PP (tabla)', 'color': '#000000', 'block': 'Otros'},
            {'id': 12, 'name': 'CUP', 'fullname': "Candidatura d'Unitat Popular", 'color': '#ffed00', 'block': 'Separatista'},
            {'id': 99, 'name': 'XX', 'fullname': None, 'color': 'amarillo', 'block': None}]
    catalog = pages.catalog(meta, rows)
    assert list(catalog) == ['PP', 'CUP', 'XX']
    assert catalog['PP'] == {'fullname': 'Partido Popular', 'color': '#1d84ce', 'block': 'Derecha'}
    assert catalog['CUP'] == {'fullname': "Candidatura d'Unitat Popular", 'color': '#ffed00', 'block': 'Separatista'}
    assert catalog['XX'] == {'fullname': 'XX', 'color': pages.OTHERS_COLOR, 'block': None}
    assert catalog['ZZ'] == {'fullname': 'ZZ', 'color': pages.OTHERS_COLOR, 'block': None} and 'ZZ' not in catalog
    assert pages.catalog(meta) == pages.catalog(meta, None) == pages.catalog(meta, []) == {'PP': catalog['PP']}


def test_eleccion_detail_colours_the_parties_outside_the_run_from_the_published_catalogue(events_site, fresh_app):
    """Con `parties.json` publicado, los partidos de la elección que el run no simula (PSOE, VOX, Cs) salen con su
    color, su nombre completo y su bloque, en la página y en el catálogo del navegador; el run sigue primero."""
    write_parties(fresh_app.fs)
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and 'Ciudadanos' in page and '#eb6109' in page and 'Partido Socialista Obrero Español' in page
    data = initial(body)
    assert [party['name'] for party in data['catalog']['parties']] == ['PP', 'PSOE', 'VOX', 'Cs']
    assert data['catalog']['parties'][0]['color'] == fixture_data('meta')['parties'][0]['color']
    assert data['catalog']['parties'][1] == {'name': 'PSOE', 'fullname': 'Partido Socialista Obrero Español', 'color': '#ef1c27', 'block': 'Izquierda'}
    assert data['catalog']['parties'][3] == {'name': 'Cs', 'fullname': 'Ciudadanos', 'color': '#eb6109', 'block': 'Derecha'}
    context = pages.eleccion_context(pages.parse_state({'scope': 'es'}), '2023-07-23')
    assert [row['color'] for row in context['results_rows']] == [fixture_data('meta')['parties'][0]['color'], '#ef1c27', '#63be21', '#eb6109']
    assert 'CUP' in context['catalog'] and 'CUP' not in [party['name'] for party in data['catalog']['parties']]


def test_promedios_browser_catalogue_covers_the_table_and_the_last_results(events_site, fresh_app):
    """El catálogo que Promedios manda al navegador lleva los partidos del run, los de la tabla de sondeos y los del
    último resultado, con el color del catálogo publicado para los que el run no simula, y los `bmaps` del run."""
    write_parties(fresh_app.fs)
    data = initial(call(events_site, '/es')[2])
    assert [party['name'] for party in data['catalog']['parties']] == ['PP', 'PSOE', 'VOX', 'Cs']
    assert data['catalog']['parties'][3] == {'name': 'Cs', 'fullname': 'Ciudadanos', 'color': '#eb6109', 'block': 'Derecha'}
    assert data['catalog']['bmaps'] == fixture_data('meta')['bmaps']
    assert [card['color'] for card in data['cards']] == [fixture_data('meta')['parties'][0]['color']]


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


def test_eleccion_detail_draws_the_average_with_the_result_and_what_the_model_said(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body).replace('\n', ' ')
    assert status == 200 and 'congelado a 6 días de la elección' in page
    assert 'id="series-chart"' in page and 'id="series-window"' in page and 'id="series-legend"' in page and 'name="view"' in page
    # «Sondeos / Promedio / Ambos», como en Promedios (2026-10-23): oculto sin JS, «Ambos» marcado
    layers = page[page.index('id="series-layers"'):]
    layers = layers[:layers.index('</fieldset>')]
    assert re.search(r'<fieldset id="series-layers" class="seg" hidden>', page)
    assert re.findall(r'name="layers" value="(\w+)"( checked)?>', layers) == [('sondeos', ''), ('promedio', ''), ('ambos', ' checked')]
    assert '«Sin sondeos»' not in page and 'Sondeos / Promedio / Ambos' in page
    assert 'href="/api/v1/events/es/2023-07-23/series?format=csv"' in page and 'Promedio de sondeos del modelo a 6 días' in page
    assert 'no está publicado todavía: hace falta' not in page
    # la tarjeta del modelo: selector de horizonte (6 marcado), una tabla por horizonte (la de 30 oculta) y la de bloques
    assert '<fieldset id="model-horizon" class="seg" hidden>' in page and '<input type="radio" name="horizon" value="6" checked>' in page
    assert 'data-horizon="6">' in page and 'data-horizon="30" hidden>' in page
    table = re.search(r'<table id="model-table-6">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col">(.*?)</th>', table) == ['Partido', 'Promedio (%)', 'Resultado (%)', 'Error', 'Escaños modelo', 'Escaños reales']
    assert table.index('<td>PP</td>') < table.index('<td>32,4</td>') < table.index('<td><strong>140</strong></td>')
    assert 'class="cell-shade"' in table and '0,6</td>' in table  # el error, 32,41 − 33,05, en su celda sombreada
    assert 'id="model-blocks-6"' in page and 'id="model-shares-6"' not in page and 'Lo que dijo el modelo' in page
    # las dos filas fijas del pie, tras la media simple
    foot = page[page.index('<tfoot>'):page.index('</tfoot>')]
    assert foot.index('Media simple') < foot.index('<b>Promedio de sondeos</b><small>modelo · a 6 días</small>') < foot.index('<b>Resultado</b><small>23 jul 2023</small>')
    assert foot.count('<tr class="fixed"') == 2 and 'Congelado el 17 jul 2023' in foot and '>32,4</td>' in foot  # el líder, sombreado
    data = initial(body)
    assert data['series']['horizon'] == 6 and data['series']['dates'][-1] == '2023-07-17' and data['series']['mean']['PP'][-1] == 32.41  # dos decimales
    assert data['result']['date'] == '2023-07-23' and data['result']['values']['PP'] == 33.06
    assert data['model_last'] == {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.41, 'PSOE': 30.99, 'VOX': 13.01},
                                  'seats': {'PP': 140, 'PSOE': 118, 'VOX': 30}, 'seats_valid': True}
    assert data['catalog']['bmaps'] == {'main': ['PP', 'PSOE', 'VOX'], 'blocks': {'Derecha': ['PP', 'VOX', 'Cs'], 'Izquierda': ['PSOE']}}
    assert set(data['inks']) >= {'PP', 'PSOE', 'VOX', 'Derecha', 'Izquierda'}


def test_eleccion_detail_with_model_but_without_series_shows_a_notice(events_site, monkeypatch):
    real = pages.elecciones.event_part
    monkeypatch.setattr(pages.elecciones, 'event_part', lambda scope, date, name: None if name == 'series' else real(scope, date, name))
    status, _, body = call(events_site, '/elecciones/es/generales/2023-07')
    page = html(body)
    assert status == 200 and 'El promedio de este ciclo no está publicado todavía' in page
    assert 'id="series"' in page and 'id="series-chart"' not in page and page.count('/api/v1/events/es/2023-07-23/') == 7 + 2
    assert 'series-download' not in page  # nothing to download without the series
    data = initial(body)
    assert 'series' not in data and data['model_last']['horizon'] == 6 and 'tr class="fixed"' in page


def test_promedios_polls_table_has_no_fixed_rows(api):
    page = html(call(api, '/es')[2])
    assert 'Media simple' in page and 'tr class="fixed"' not in page


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


NOT_ANALYSED = ('Esta elección tuvo menos de 10 sondeos útiles, así que no la usamos para el rating de las '
                'encuestadoras ni para el backtest.')


def test_eleccion_not_analysed_says_it_does_not_count_for_the_rating(api, fresh_app):
    write_event(fresh_app.fs, date='2019-11-10', model=False, featured=False)
    status, _, body = call(api, '/elecciones/es/generales/2019-11')
    page = html(body)
    assert status == 200 and NOT_ANALYSED in page and 'Sin backtest para esta elección' not in page


def test_eleccion_meta_without_featured_counts_as_analysed(api, fresh_app, tmp_path):
    """Una elección publicada antes de que la meta llevara `featured` se lee como analizada."""
    write_event(fresh_app.fs, date='2019-11-10', model=False)
    path = tmp_path / bundle.PREFIX / bundle.path_event('es', '2019-11-10', 'meta')
    meta = json.loads(path.read_text())
    del meta['data']['featured']
    path.write_text(json.dumps(meta))
    status, _, body = call(api, '/elecciones/es/generales/2019-11')
    page = html(body)
    assert status == 200 and 'Sin backtest para esta elección' in page and NOT_ANALYSED not in page


def test_eleccion_without_model_shows_a_notice(events_site):
    status, _, body = call(events_site, '/elecciones/es/generales/2019-11')
    page = html(body)
    assert status == 200 and 'Sin backtest para esta elección' in page
    assert page.count('/api/v1/events/es/2019-11-10/') == 4 + 2
    assert 'id="series"' not in page and 'id="model-horizon"' not in page and 'tr class="fixed"' not in page


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
    pp = tables[0]['rows'][0]
    assert pp == {'name': 'PP', 'official': 33.05, 'model': 32.41, 'error': round(32.41 - 33.05, 2), 'official_seats': 137, 'model_seats': 140}
    assert set(tables[0]['blocks'][0]) == {'block', 'official', 'model', 'error', 'official_seats', 'model_seats',
                                           'p_majority', 'majority'}
    # las medidas del modelo, con el margen de la elección (Derecha − Izquierda): PP+VOX frente a PSOE
    assert set(tables[0]) >= {'error_avg', 'error_blocks'} and tables[0]['error_avg'] is not None
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
    assert tables[1]['rows'][0]['error'] == -1.0  # el voto sigue ahí aunque los escaños no valgan


def test_model_last_takes_the_shortest_horizon_and_the_horizon_run_seats():
    model = {'horizons': [30, 6],
             'shares': [{'horizon': h, 'party': p, 'main': p != 'Cs', 'official': 33.0, 'model': m}
                        for h in (6, 30) for p, m in (('PP', 32.4 if h == 6 else 31.0), ('Cs', 1.2))],
             'seats': [{'horizon': h, 'party': 'PP', 'main': True, 'official': 137, 'model': 140, 'model_h': 142 if h == 6 else None,
                        'seats_valid': True} for h in (6, 30)],
             'blocks': []}
    assert pages.model_last(model, '2023-07-23') == {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.4, 'Cs': 1.2},
                                                     'seats': {'PP': 142}, 'seats_valid': True}
    nowcast = {**model, 'horizons': [30]}
    assert pages.model_last(nowcast, '2023-07-23')['seats'] == {'PP': 140} and pages.model_last(nowcast, '2023-07-23')['horizon'] == 30
    assert pages.model_last({'horizons': [], 'shares': [], 'seats': [], 'blocks': []}, '2023-07-23') is None


def test_fixed_rows_of_the_polls_table_footer():
    last = {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.4, 'PSOE': 31.0}, 'seats': {'PP': 140}, 'seats_valid': True}
    results = [{'party': 'PP', 'pct': 33.06, 'seats': 160}, {'party': 'PSOE', 'pct': 31.44, 'seats': 150}]
    colors = {'PP': '#1d84ce', 'PSOE': '#e4001c'}
    rows = pages.fixed_rows(last, results, ['PP', 'PSOE', 'VOX'], colors, '2023-07-23')
    assert [(row['label'], row['small']) for row in rows] == [('Promedio de sondeos', 'modelo · a 6 días'), ('Resultado', '23 jul 2023')]
    assert rows[0]['title'] == 'Congelado el 17 jul 2023: sólo los sondeos publicados hasta entonces' and rows[1]['title'] is None
    assert [cell['value'] for cell in rows[0]['cells']] == ['32,4', '31,0', '–'] and [cell['leader'] for cell in rows[0]['cells']] == [True, False, False]
    assert [cell['value'] for cell in rows[1]['cells']] == ['33,1', '31,4', '–'] and rows[1]['cells'][2]['color'] == pages.OTHERS_COLOR
    assert pages.fixed_rows(None, results, ['PP'], colors, '2023-07-23') == []


def test_curve_block_colors_follow_the_page_then_the_first_party():
    """Los bloques de la curva van del color que ya usa la página para el bloque (hemiciclo, resultados por bloque) y,
    si la página no lo tiene, del color de su primer partido: así las tres vistas Bloques cuadran."""
    catalog = pages.Catalog({'PP': {'fullname': 'PP', 'color': '#1d84ce', 'block': 'Derecha'},
                             'ERC': {'fullname': 'ERC', 'color': '#ffb232', 'block': 'Separatista'}})
    blocks_map = {'Derecha': ['PP', 'VOX'], 'Separatista': ['ERC']}
    colors = pages.curve_block_colors(blocks_map, {'Derecha': '#000080', 'Izquierda': '#e4001c'}, catalog)
    assert colors == {'Derecha': '#000080', 'Separatista': '#ffb232'}
    assert pages.curve_block_colors({}, {'Derecha': '#000080'}, catalog) == {}


def test_horizon_tables_merge_vote_and_seats_and_prefer_the_horizon_run():
    model = {'horizons': [6],
             'shares': [{'horizon': 6, 'party': 'PP', 'main': True, 'official': 33.05, 'model': 32.41},
                        {'horizon': 6, 'party': 'Cs', 'main': False, 'official': 1.0, 'model': 1.5}],
             'seats': [{'horizon': 6, 'party': 'PP', 'main': True, 'official': 137, 'model': 140, 'model_h': 142, 'seats_valid': True}],
             'blocks': [{'horizon': 6, 'block': 'Derecha', 'official_seats': 169, 'model_seats': 171, 'model_seats_h': 173,
                         'p_majority': 0.4, 'p_majority_h': 0.45, 'majority': False, 'seats_valid': True}]}
    [table] = pages.horizon_tables(model)
    assert table['rows'] == [{'name': 'PP', 'official': 33.05, 'model': 32.41, 'error': round(32.41 - 33.05, 2), 'official_seats': 137, 'model_seats': 142}]
    assert table['blocks'] == [{'block': 'Derecha', 'official': None, 'model': None, 'error': None, 'official_seats': 169,
                                'model_seats': 173, 'p_majority': 0.45, 'majority': False}]  # sin `parties`, sin voto
    model['seats'][0]['model_h'] = None
    del model['blocks'][0]['model_seats_h'], model['blocks'][0]['p_majority_h']
    [table] = pages.horizon_tables(model)
    assert table['rows'][0]['model_seats'] == 140 and table['blocks'][0]['model_seats'] == 171 and table['blocks'][0]['p_majority'] == 0.4


def test_horizon_tables_sum_the_vote_of_each_block_and_score_the_model():
    """El voto de cada bloque es la suma de sus partidos (`parties` del backtest), y la tabla lleva las medidas de
    las encuestadoras para el modelo: error medio (partidos principales, sin signo) y sesgo bloques (margen del
    primer bloque menos el del segundo, modelo menos resultado)."""
    shares = [('PP', True, 33.0, 32.0), ('VOX', True, 12.0, 13.0), ('PSOE', True, 31.0, 29.0), ('Cs', False, 1.0, 1.5)]
    model = {'horizons': [6],
             'shares': [{'horizon': 6, 'party': p, 'main': m, 'official': o, 'model': v} for p, m, o, v in shares],
             'seats': [],
             'blocks': [{'horizon': 6, 'block': 'Derecha', 'parties': 'PP+VOX', 'official_seats': 169, 'model_seats': 171,
                         'p_majority': 0.4, 'majority': False, 'seats_valid': True},
                        {'horizon': 6, 'block': 'Izquierda', 'parties': 'PSOE+UP', 'official_seats': 121, 'model_seats': 118,
                         'p_majority': 0.0, 'majority': False, 'seats_valid': True}]}
    [table] = pages.horizon_tables(model, ['Derecha', 'Izquierda'])
    derecha, izquierda = table['blocks']
    assert (derecha['official'], derecha['model'], derecha['error']) == (45.0, 45.0, 0.0)
    assert (izquierda['official'], izquierda['model'], izquierda['error']) == (None, None, None)  # UP sin voto: no se suma a medias
    assert table['error_avg'] == round((1 + 1 + 2) / 3, 2) and table['error_blocks'] is None
    model['blocks'][1]['parties'] = 'PSOE'
    [table] = pages.horizon_tables(model, ['Derecha', 'Izquierda'])
    assert table['error_blocks'] == round((45.0 - 29.0) - (45.0 - 31.0), 2)  # +2: el modelo dio 2 puntos más de margen a la derecha
    [table] = pages.horizon_tables(model, ['Izquierda', 'Derecha'])
    assert table['error_blocks'] == -2.0  # el signo sigue el orden del margen de la elección
    assert pages.horizon_tables(model)[0]['error_blocks'] is None  # sin margen, sin sesgo


def test_eleccion_detail_scores_the_model_like_the_pollsters(events_site):
    """«Lo que dijo el modelo»: su error medio y su sesgo bloques en cada horizonte, la tabla de bloques con los
    campos de la de partidos más la probabilidad de mayoría, y una fila «Modelo (promedio)» en la tabla de las
    encuestadoras, con el horizonte más corto (2026-10-21)."""
    page = html(call(events_site, '/elecciones/es/generales/2023-07')[2])
    model = page[page.index('id="model"'):page.index('id="last-errors"')]
    assert re.search(r'<p class="muted model-score">Error medio: [0-9,]+ · Sesgo bloques: -?[0-9,]+</p>', model)
    blocks = re.search(r'<table id="model-blocks-6">(.*?)</table>', model, re.S).group(1)
    assert re.findall(r'<th scope="col">([^<]*)</th>', blocks) == ['Bloque', 'Promedio (%)', 'Resultado (%)', 'Error',
                                                                  'Escaños modelo', 'Escaños reales', 'Prob. de mayoría']
    table = re.search(r'<table id="last-table">(.*?)</table>', page, re.S).group(1)
    row = re.search(r'<tr class="model-row">(.*?)</tr>', table, re.S).group(1)
    assert '<a href="/metodo/promedio">Modelo (promedio)</a>' in row and '<td>6</td>' in row and '<td class="na">—</td>' in row
    assert table.count('<tr class="model-row">') == 1
    # y en la diana, resaltado: el punto del modelo va aparte de los de las encuestadoras
    data = initial(call(events_site, '/elecciones/es/generales/2023-07')[2])
    point = data['model_point']
    assert point['pollster'] == 'Modelo (promedio)' and point['days'] == 6 and set(data['blocks']) <= set(point)
    assert all(row['pollster'] != 'Modelo (promedio)' for row in data['last'])
    assert 'El punto naranja, por encima de los demás, es el modelo' in page


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
    assert 'id="model-table-30"' in page and 'id="model-table-6"' in page and 'id="model-shares-6"' not in page
    # la tabla de bloques sigue (el voto vale) con los escaños en raya, como la de partidos
    assert 'id="model-blocks-6"' in page and 'id="model-blocks-30"' in page


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
    assert status == 200 and '-3,0' in table  # el sesgo bloques de la tabla va con signo (2026-10-17)
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
    houses = re.sub(r'<tr class="model-row">.*?</tr>', '', table, flags=re.S)  # la fila del modelo enlaza al método
    assert status == 200 and 'GAD3' in houses and '<a ' not in houses


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


def test_model_record_scores_the_model_in_every_election_with_a_backtest(history_site):
    """El modelo como una encuestadora: en cada elección con backtest, su estimación al horizonte más corto, medida
    con las definiciones de las encuestadoras y frente a sus últimos sondeos; la de 2019 no tiene backtest."""
    record = pages.model_record()
    [row] = record['rows']
    assert row['scope'] == 'es' and row['event_date'] == '2023-07-23' and row['days'] == 6 and row['year'] == 2023
    assert row['label'] == 'Generales 2023' and row['name'] == row['label']
    assert row['error_avg'] is not None and set(record['blocks']) <= set(row)
    assert 0.0 <= row['better_than'] <= 1.0 and row['n_pollsters'] == 3
    assert {r['party'] for r in row['parties']} >= {'PP'}
    kpis = {kpi['label']: kpi for kpi in record['kpis']}
    assert list(kpis) == ['Elecciones', 'Error medio', 'Error de bloques', 'Frente a las encuestadoras']
    assert kpis['Elecciones']['value'] == '1' and kpis['Frente a las encuestadoras']['unit'] == '%'


def test_backtest_page_starts_with_the_model_as_a_pollster(history_site):
    """/metodo/backtest abre con el modelo medido como una encuestadora: KPIs, camino (Evento, Error medio, Sesgo
    bloques), diana con filtros y su tabla (Evento y los bloques, la izquierda primero) y su último estimado por
    partido frente al resultado; después, las tarjetas por ámbito de siempre (2026-10-21)."""
    status, _, body = call(history_site, '/metodo/backtest', query='scope=es')
    page = html(body)
    assert status == 200
    order = [page.index('id="{}"'.format(card)) for card in ('model-profile', 'model-path', 'bullseye', 'model-last')]
    assert order == sorted(order) and order[-1] < page.index('id="scope-filter"')
    assert re.findall(r'<div class="kpi"><div class="l">([^<]+)</div>', page)[:4] == [
        'Elecciones', 'Error medio', 'Error de bloques', 'Frente a las encuestadoras']
    path = re.search(r'<table id="model-path-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', path) == ['Evento', 'Error medio', 'Sesgo bloques']
    assert '<td>Generales 2023</td>' in path
    table = re.search(r'<table id="bullseye-table" data-fixed="1">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col">([^<]*)</th>', table) == ['Evento', 'Izquierda', 'Derecha']
    assert '<form id="bullseye-filter" class="series-controls filters-bar" hidden>' in page
    last = re.search(r'<table id="model-last-table">(.*?)</table>', page, re.S).group(1)
    assert 'PP' in last and '<td>Generales 2023</td>' in last
    data = initial(body)
    assert [row['event_date'] for row in data['bullseye']] == ['2023-07-23'] and data['bullseye_blocks'] == ['Derecha', 'Izquierda']
    assert [s['code'] for s in data['bullseye_scopes']] == ['es']


def test_backtest_page_without_model_elections_has_no_model_cards(api):
    page = html(call(api, '/metodo/backtest', query='scope=es')[2])
    assert 'id="model-profile"' not in page and 'id="bullseye"' not in page


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
    assert 'viendo sólo los sondeos publicados hasta cada fecha' in text and 'lo que se sabía' not in text
    assert 'id="datos"' in page and 'id="publicacion"' in page and 'id="limitaciones"' in page
    assert 'se congela' in page and 'fuera de España' in page
    assert '1.000 simulaciones' not in page and 'id="scope-filter"' not in page


def test_metodo_promedio_text(api):
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and '<title>Demoscopio · ¿Cómo promediamos las encuestas?</title>' in page
    for anchor in ('que-entra', 'peso', 'efectos-de-casa', 'regresion', 'banda', 'voto-estimado'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'id="metodo-nav"' in page and '<math' in page and 'id="resumen"' in page
    assert page.index('id="rating"') < page.index('id="cuentas"')
    # the formula of the rating factor lives in the note of the weights (rating de 0 a 100)
    notes = page[page.index('id="notas"'):]
    assert re.search(r'<math display="block" aria-label="[^"]*logaritmo de uno más Q sub c entre cien', notes)
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
    for anchor in ('que-se-mide', 'calidad-previa', 'evidencia', 'rating-final', 'ranking-global', 'herding'):
        assert 'id="' + anchor + '"' in page, anchor
    assert 'la mitad' in page or '50%' in page  # rating_weight 0,5 leído del catálogo
    assert '<math' in page
    landing = html(call(api, '/metodo')[2])
    assert 'Cómo funciona el modelo' not in landing and 'href="/metodo/rating"' in landing
    text = re.sub(r'\s+', ' ', page)
    assert 'la desviación de cada encuestadora apenas cambia' in text and 'no cambia:' not in text
    assert 'el denominador es la calidad previa media de las encuestadoras, no el rating medio' in text
    assert 'muestra multiplicada por <var>h</var>′<sup>2</sup>' in text
    assert 'viendo sólo los sondeos publicados hasta cada fecha' in text and 'lo que se sabía' not in text
    assert 'Trabajo de campo' in text and 'Implantación' in text and '<table id="rubric-table">' in page
    assert '<strong>Diferencial</strong>' in text and 'Sesgo bloques' in text and 'Calidad</strong>' not in text
    assert 'cada punto vale 8 de calidad, con un mínimo de 5' in text and 'nadie pasa de 80 antes de los datos' in text
    assert 'los puntos que faltan se ganan acertando' in text and 'de la que apenas se sabe nada' in text
    assert 'En las tablas del sitio aparece como «Confianza»' in text and 'es la columna «Confianza»' in text
    assert 'cuentan como una elección' in text and 'pesa en torno al 10% de su rating' in text
    assert 'la calidad previa pesa dos tercios del rating' in text and 'entre 3 y 4' not in text
    assert 'Por su cercanía a la votación' not in text and 'multiplica el peso por 0,7' not in text
    assert 'puesta a mano' not in text and 'rúbrica pública' in text
    assert 'calidad previa según una rúbrica pública' in re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))


def test_metodo_rating_explains_evidence_by_election(api):
    """/metodo/rating counts the evidence in elections, quotes sigma, tau and k as the code sets them (k ~ 2
    elections, not 3 polls) and presents the prior quality as a prior of observable criteria."""
    from inspect import signature

    from mtpy.lib.computer import Computer

    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    params = signature(Computer.__init__).parameters
    sigma, tau = params['bias_dev_sigma'].default, params['bias_dev_tau'].default
    k = (sigma / tau) ** 2
    assert 'por elecciones' in text and '<var>k</var>' in text
    assert '{:.3f}'.format(sigma).replace('.', ',') in text and 'τ es la escala, {}'.format(
        '{:.2f}'.format(tau).replace('.', ',')) in text
    assert '{:.2f}'.format(k).replace('.', ',') in text and '2 elecciones' in text
    assert 'un prior de criterios observables' in text and 'no un predictor validado' in text
    assert 'id="criteria-test-table"' in text and 'id="difficulty-table"' in text and 'Implantación' in text
    assert 'la calidad previa pesa como tres sondeos' not in text and 'tres sondeos' not in text
    # sigma as measured by the rating procedure and with the degrees-of-freedom correction; the 0,006 spread is raw
    assert 'unos 0,014' in text and '0,0135 con el procedimiento de medida' in text
    assert 'corrección por grados de libertad' in text and 'de verdad' not in text
    assert 'menos aún si se descuenta el ruido de cada media' in text
    # Regionals are not easier, so not discounted further; their lower separation is already in their weight
    assert 'no hay motivo para descontarlas más' in text and 'ya lo recoge su peso' in text
    assert 'dos autonómicas cuentan en el rating como unas generales del mismo año' in text
    assert 'su peso <var>k</var> son unas 2 elecciones de evidencia' in text


def test_metodo_rating_reads_the_rubric_from_the_catalogue(api, monkeypatch):
    changed = pages.quality_rubric() | {'per_point': 10, 'floor': 1, 'cap': 100}
    monkeypatch.setattr(pages.metodo, 'quality_rubric', lambda: changed)
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'cada punto vale 10 de calidad, con un mínimo de 1' in text and 'nadie pasa de 100 antes de los datos' in text


def test_metodo_rating_reads_the_scope_weights_from_the_catalogue(api, monkeypatch):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'cada elección autonómica pesa la mitad que unas generales' in page
    assert 'href="/metodo/promedio#peso"' in page and 'href="/casas?scope=es"' in page and 'href="/metodo/backtest"' in page
    assert 'id="metodo-nav"' in page and 'aria-current="page"' in page and 'En resumen' in page
    changed = webapi.catalogue().copy()
    changed.loc[changed.index != 'es', 'rating_weight'] = 0.25
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: changed)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'pesa el 25% de unas generales' in page and 'la mitad' not in page
    mixed = changed.copy()
    mixed.loc['es-md', 'rating_weight'] = 0.5
    monkeypatch.setattr(pages.metodo, 'catalogue', lambda: mixed)
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'pesa menos que unas generales' in page and 'la mitad' not in page


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
    # the comparison of the shortest horizon and the years of the evaluated elections, for the method texts (2026-10-23)
    assert figures['backtest'] == {'horizon': 6, 'n_cases': 2, 'mae': 1.69, 'mae_last_poll': 1.74, 'mae_mean_4w': 1.86, 'cov95': 0.957}
    assert figures['years'] == ['2019', '2023']


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
    heads = re.findall(r'<th scope="col"[^>]*>([^<]*)', thead)
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




def assert_public_wording(page, label):
    """Redacción pública (reglas de Luis): ni "patrocinador/es" en el texto visible (se dice «medio»), ningún "Cómo
    leerlo" abierto y el adverbio «sólo» (solamente) siempre con tilde, su regla de oro. «Casa» está permitida como
    sinónimo secundario de «encuestadora» desde el 2026-10-23 (antes estaba prohibida); el efecto se llama «efecto de
    casa»."""
    text = visible_text(page)
    assert not re.search(r'\b[Pp]atrocinador(es)?\b', text), label
    assert '<details class="how" open' not in page, label
    # the adjective keeps the accent off («un solo sondeo», «por sí solo»); anything else is the adverb
    sólo = re.findall(r'(?<![Uu]n )(?<![Ee]l )(?<![Ss]í )\b[Ss]olo\b.{0,25}', text)
    assert not sólo, (label, sólo[:5])


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
                                                          'Análisis', 'Método']


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


def test_series_charts_say_the_axis_follows_what_is_shown(events_site):
    """El eje vertical del promedio ya no arranca en cero (Luis, 2026-10-19): lo dicen las tres tarjetas que lo dibujan."""
    promedios = re.sub(r'\s+', ' ', html(call(events_site, '/es')[2]))
    detail = re.sub(r'\s+', ' ', html(call(events_site, '/elecciones/es/generales/2023-07')[2]))
    assert promedios.count('El eje vertical no empieza en cero') == 1 and 'el eje vertical se ajusta a lo que queda a la vista' in promedios
    assert detail.count('El eje vertical no empieza en cero') == 1


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


def test_promedios_hemicycle_card_offers_a_download_button_hidden_without_js(api):
    """El hemiciclo se descarga como PNG firmado: el botón va en la cabecera de la tarjeta, oculto sin JS, y lleva
    en sus atributos el título (la elección y su fecha), el subtítulo (el gráfico y la fecha de la estimación), la
    firma (dominio del sitio y usuario de X del autor) y el nombre del fichero (sitio, gráfico, ámbito y fecha del
    run)."""
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    hemi = page[page.index('id="hemicycle-card"'):page.index('id="hemicycle"')]
    button = re.search(r'<header class="card-head"><div class="card-tools"><button type="button" class="tool-btn" '
                       r'id="hemicycle-download" hidden((?: data-\w+="[^"]*")+)>(.*?)</button></div><h2>Hemiciclo ', hemi)
    assert button, hemi
    attrs = dict(re.findall(r'data-(\w+)="([^"]*)"', button.group(1)))
    assert attrs['title'] == 'Elecciones generales · 29 nov 2026'
    assert re.fullmatch(r'Hemiciclo · Estimación a \d+ \w+ 2026', attrs['subtitle']), attrs
    assert attrs['sign'] == 'demoscopio.es · @luis_sancho'
    assert attrs['file'] == 'demoscopio-hemiciclo-es-2026-10-08.png'
    assert set(attrs) == {'title', 'subtitle', 'sign', 'file'}
    assert button.group(2) == '<svg aria-hidden="true"><use href="#i-download"/></svg>Descargar'
    assert 'id="i-download"' in page  # the icon is in the sprite
    # the subtitle follows the mode and the file the scope
    forecast = html(call(api, '/es', query='mode=forecast')[2]).replace('\n', ' ')
    assert 'data-subtitle="Hemiciclo · Pronóstico para el 29 nov 2026"' in forecast
    # the other hemicycles (home, election detail) have no button
    assert 'hemicycle-download' not in html(call(api, '/')[2])


def test_promedios_series_card_offers_a_download_button_hidden_without_js(api):
    """El promedio de sondeos se descarga como PNG firmado (Luis, 2026-10-21), con el mismo botón que el hemiciclo:
    el subtítulo nombra el gráfico (con la proyección en el modo Elección) y el fichero, el promedio."""
    page = html(call(api, '/es')[2]).replace('\n', ' ')
    card = page[page.index('id="series"'):page.index('id="series-chart"')]
    button = re.search(r'<header class="card-head"><div class="card-tools"><button type="button" class="tool-btn" '
                       r'id="series-download" hidden((?: data-\w+="[^"]*")+)>(.*?)</button></div><h2>Promedio de sondeos ', card)
    assert button, card
    attrs = dict(re.findall(r'data-(\w+)="([^"]*)"', button.group(1)))
    assert attrs['title'] == 'Elecciones generales · 29 nov 2026'
    assert re.fullmatch(r'Promedio de sondeos · Estimación a \d+ \w+ 2026', attrs['subtitle']), attrs
    assert attrs['sign'] == 'demoscopio.es · @luis_sancho'
    assert attrs['file'] == 'demoscopio-promedio-es-2026-10-08.png'
    assert button.group(2) == '<svg aria-hidden="true"><use href="#i-download"/></svg>Descargar'
    forecast = html(call(api, '/es', query='mode=forecast')[2]).replace('\n', ' ')
    assert 'data-subtitle="Promedio de sondeos y proyección · Pronóstico para el 29 nov 2026"' in forecast


def download_attrs(page, button_id, heading):
    """Atributos `data-*` del botón de descarga `button_id` de una cabecera de tarjeta cuyo título empieza por
    `heading`, o None si no está."""
    button = re.search(r'<header class="card-head"><div class="card-tools"><button type="button" class="tool-btn" '
                       r'id="%s" hidden((?: data-\w+="[^"]*")+)>.*?</button></div><h2>%s' % (button_id, re.escape(heading)), page)
    return dict(re.findall(r'data-(\w+)="([^"]*)"', button.group(1))) if button else None


def test_promedios_pollster_card_offers_a_download_button_named_after_the_house_in_js(events_site):
    """La encuestadora frente al promedio también se descarga (Luis, 2026-10-21): el servidor deja el subtítulo
    sin la casa, que el JS antepone con la elegida, y el JS le añade al fichero su nombre."""
    page = html(call(events_site, '/es')[2]).replace('\n', ' ')
    attrs = download_attrs(page, 'pollster-download', 'Una encuestadora frente al promedio')
    assert attrs, page[page.index('id="pollster"'):page.index('id="pollster"') + 800]
    assert attrs['title'] == 'Elecciones generales · 29 nov 2026'
    assert re.fullmatch(r'frente al promedio · Estimación a \d+ \w+ 2026', attrs['subtitle']), attrs
    assert attrs['file'] == 'demoscopio-encuestadora-es-2026-10-08.png' and attrs['sign'] == 'demoscopio.es · @luis_sancho'


def test_eleccion_series_card_offers_a_download_button(events_site):
    """El promedio del detalle de una elección pasada se descarga con el nombre y la fecha de la elección, el
    horizonte al que se congeló y el resultado (sin serie no hay botón: el test del aviso)."""
    page = html(call(events_site, '/elecciones/es/generales/2023-07')[2]).replace('\n', ' ')
    attrs = download_attrs(page, 'series-download', 'Promedio de sondeos')
    assert attrs == {'title': 'Generales 2023 · 23 jul 2023', 'subtitle': 'Promedio de sondeos a 6 días y resultado oficial',
                     'sign': 'demoscopio.es · @luis_sancho', 'file': 'demoscopio-promedio-es-2023-07-23.png'}


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


def test_promedios_opens_with_the_kpis_and_the_party_cards(events_site):
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
    assert '<span class="rk">1.º</span>' in card
    # la flecha junto a cada cifra, en una rejilla que alinea «escaños» y «voto» debajo: 135 − 160 = ▼ 25;
    # 33,1 − 33,06 = +0,04 → sin cambio, un guion (Luis, 2026-10-17)
    assert ('<div class="fig" title="Flechas: frente al resultado del 23 jul 2023"><div class="big"><b>135</b><span class="dlt down">▼ 25</span></div>'
            '<div class="big"><span>33,1%</span><span class="dlt flat">–</span></div><div class="lb">escaños</div><div class="lb">voto</div></div>') in card
    assert 'vs 2023' not in card and 'class="dl"' not in card
    css = open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    # la flecha va en su color, sin píldora, en las tablas; en blanco sobre el color del partido en las tarjetas
    assert '.dlt.up { color: #2ca02c; }' in css and '.dlt.down { color: #d62728; }' in css and '.pc .big .dlt {' in css
    assert 'border-radius' not in css[css.index('.dlt {'):css.index('.dlt.flat')]
    assert 'class="spark"' in card and 'Último año' in card
    # sin la banda gris del líder bajo el hemiciclo (Luis, 2026-10-19): las tarjetas y la tabla ya lo dicen
    assert 'class="lead-strip"' not in page


def test_promedios_party_table_draws_the_seat_range_without_a_vote_bar(api):
    page = html(call(api, '/es')[2])
    table = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    # la barra del voto era redundante con la cifra (Luis, 2026-10-19); tampoco en la tabla de bloques
    assert 'class="bar"' not in table and 'Barra del voto' not in page
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
                             'order': ['PP'], 'n_seats': 350, 'majority': 176, 'deltas': {}, 'results_year': None}}
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


def test_home_past_elections_include_scopes_without_a_run(history_two_scopes, fresh_app):
    """«Elecciones pasadas» de Inicio lista también las elecciones de un ámbito sin previsión publicada (recién
    votado): su detalle existe (2026-10-20)."""
    write_event(fresh_app.fs, scope='es-an', date='2022-06-19', model=False)
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    past = html(call(history_two_scopes, '/')[2])
    past = past[past.index('id="past"'):]
    assert 'href="/elecciones/es/andalucia/2022-06"' in past and 'href="/elecciones/es/madrid/2021-05"' in past


def test_escanos_table_draws_the_seat_range_and_the_ridge_is_compact(api):
    page = html(call(api, '/es/escanos')[2])
    table = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    assert 'class="rng"' in table and '<u style="left: 50.3%"></u>' in table and '120–150' in table
    assert '<fieldset class="pills lg" id="coalition-parties">' in page
    assert '<p class="chip" id="coalition-result">' in page


def test_escanos_majority_calculator_draws_the_bar_above_the_histogram(api):
    """La calculadora se llama «de mayorías» (Luis, 2026-10-21) y lleva encima del histograma la barra del reparto
    central, con los partidos marcados delante; Promedios la nombra igual."""
    page = html(call(api, '/es/escanos')[2]).replace('\n', ' ')
    card = page[page.index('id="coalition"'):page.index('id="districts"')]
    assert '<h2>Calculadora de mayorías ' in card and 'coaliciones' not in card
    assert card.index('id="coalition-result"') < card.index('id="coalition-bar"') < card.index('id="coalition-chart"')
    assert re.search(r'<div class="chart" id="coalition-bar" role="img" aria-label="[^"]+"></div>', card)
    assert 'los partidos marcados delante, en color, y el resto detrás, en gris' in re.sub(r'\s+', ' ', card)
    assert 'La franja de arriba y la línea del histograma son el reparto central' in re.sub(r'\s+', ' ', card)
    assert 'mediana' not in card.lower()
    promedios = re.sub(r'\s+', ' ', html(call(api, '/es')[2]))
    assert 'la calculadora de mayorías están en la pestaña Escaños' in promedios and 'calculadora de coaliciones' not in promedios


def test_escanos_texts_call_the_headline_the_central_allocation(api, fresh_app):
    write_districts(fresh_app.fs)
    page = re.sub(r'\s+', ' ', html(call(api, '/es/escanos')[2]))
    assert "Reparto central: D'Hondt en cada circunscripción sobre el voto central; cada fila suma sus escaños." in page
    assert 'la línea gruesa el reparto central' in page and 'la línea gruesa la mediana' not in page
    assert 'la simulación más cercana a la mediana' not in page and 'Escenario central' not in page
    assert 'la suma de las circunscripciones es el titular' in page
    assert 'Reparto central' in page  # etiqueta del CSV
    promedios = re.sub(r'\s+', ' ', html(call(api, '/es')[2]))
    assert 'escaños del reparto central' in promedios and 'escenario central' not in promedios


def test_metodo_escanos_explains_the_central_allocation(api):
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert '<strong>Los escaños de cada partido</strong> son su reparto central' in text
    assert 'el mismo titular en el nowcast y en el forecast' in text
    assert 'a veces saca escaño y a veces no' in text
    assert 'son su mediana' not in text and 'El escenario central' not in text


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


def test_election_detail_and_home_have_no_lead_strip(events_site):
    """Sin la banda gris del líder bajo el hemiciclo, ni en el detalle de una elección ni en Inicio (Luis,
    2026-10-21; en Promedios ya no estaba desde el 2026-10-19)."""
    page = html(call(events_site, '/elecciones/es/generales/2023-07')[2])
    assert 'lead-strip' not in page and 'Líder:' not in page and 'class="chart hemi"' in page
    home = html(call(events_site, '/')[2])
    assert 'lead-strip' not in home and 'Líder:' not in home and 'id="hemicycle-mini"' in home
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
    """El perfil enseña la desviación frente al resultado (sólo las elecciones con tres sondeos en los 90 días
    previos) y, con el conmutador, la desviación frente al consenso del ciclo (todas); el "Cómo leerlo" lo explica."""
    page = html(call(houses, '/casas/gad3', query='scope=es')[2])
    card = page[page.index('id="effects"'):page.index('id="herding"')]
    assert '<fieldset id="effects-view" class="seg" hidden>' in card
    assert 'name="eview" value="resultado" checked' in card and 'name="eview" value="ciclo"' in card
    result = re.search(r'<table id="profile-effects">(.*?)</table>', card, re.S).group(1)
    cycle = re.search(r'<table id="profile-effects-cycle">(.*?)</table>', card, re.S).group(1)
    assert result.count('<tr>') == 1 + 2 and cycle.count('<tr>') == 1 + 3  # la cabecera y las filas
    assert '0,7' in cycle and '0,7' not in result  # 2026 sólo tiene la desviación frente al consenso
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
    # 2026-10-19: la altura sigue al ancho (2 : 1) entre 480 px y un tope que cabe en un portátil; 280 px fijos en móvil
    assert '.chart.chart-tall { height: auto; aspect-ratio: 2 / 1; min-height: 480px; max-height: min(720px, 72vh); }' in css
    assert '  .chart.chart-tall { height: 280px; min-height: 0; max-height: none; aspect-ratio: auto; }' in css
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
    (('es', 'aragon', '2023-05'), 404, 'scope not published'),  # sin elecciones publicadas
])
def test_resolve_event_path_errors(history_two_scopes, fresh_app, segments, status, message):
    write_event(fresh_app.fs, scope='es-an', date='2022-06-19', model=False)  # elecciones sin run: con detalle
    webapi.site().manifest_cache_clear()
    webapi.site().files.clear()
    with pytest.raises(HttpError) as error:
        pages.resolve_event_path(*segments)
    assert error.value.status == status and str(error.value.message) == message
    assert pages.resolve_event_path('es', 'andalucia', '2022-06') == ('es-an', '2022-06-19')


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
    twins = pd.DataFrame({'id': [7, 3], 'name': ['Opina', 'opina'], 'mtype': [None] * 2, 'quality': [50.0] * 2,
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
    assert '/elecciones?scope=' not in page.replace('/elecciones?scope=es"', '')  # sólo queda el enlace al histórico
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
    assert 'Efectos de casa por elección' in page and 'de encuestadora por elección' not in page
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
    # el fondo es rojo si el sondeo se quedó corto y verde si se pasó, más intenso cuanto mayor, hasta 8 puntos (2026-10-17)
    from mtpy.lib.pages.filters import shade
    assert '<td class="cell-shade" style="background-color: {}">-1,1</td>'.format(shade(-1.14, 0, -8, '#d62728')) in result
    assert '<td class="cell-shade" style="background-color: {}">2,0</td>'.format(shade(2.0, 0, 8, '#2ca02c')) in consensus
    assert '<div class="table-wrap" id="profile-last-consensus-wrap" hidden>' in card
    assert 'sondeo menos resultado' in card and 'demás encuestadoras' in card and 'verde si se pasó' in card
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
    # fixture sólo trae el primero, azul; el segundo, gris) y doble gradiente del cambio (rojo de -10 a 0, verde de 0 a 10)
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
    # la fixture sólo trae un bloque en `vs`: el segundo queda sin nombre (y su barra, gris)
    assert 'más margen a Derecha del que hubo; negativo, más margen al segundo' in glossary
    assert 'Cada punto vale 8 de confianza, con un mínimo de 5.' in glossary
    assert 'cada ámbito y fecha por separado' in glossary  # elecciones
    assert ('Entre paréntesis, cuántas de ellas fueron elecciones generales: el dato con el que se calcula el criterio '
            '«Implantación» de la confianza.') in glossary
    assert 'todas se explican en «Qué mide cada columna»' in re.sub(r'\s+', ' ', page)
    profile = html(call(houses, '/casas/gad3')[2])
    profile_glossary = re.search(r'<details class="how" id="glosario">(.*?)</details>', profile, re.S).group(1)
    assert re.findall(r'<dt>([^<]*)</dt>', profile_glossary) == ['Evento', 'Sondeos', 'Diferencial', 'Error medio',
                                                                  'Sesgo bloques', 'Rating', 'Cambio']


def test_casas_ranking_shows_the_general_elections_in_brackets_in_elecciones(houses):
    """«Elecciones» lleva entre paréntesis las generales («28 (5)»), con «(generales)» bajo la cabecera; sin
    columna ni filtro aparte (2026-10-20)."""
    page = html(call(houses, '/casas')[2])
    table = re.search(r'<table class="et et-rank" id="ranking-table">(.*?)</table>', page, re.S).group(1)
    heads = re.findall(r'<th scope="col"[^>]*>([^<]*)', table)
    assert heads == ['Encuestadora', 'Rating', 'Confianza', 'Elecciones', 'Sondeos', 'Diferencial',
                     'Error medio', 'Sesgo bloques', 'Rebaño']
    assert '>Elecciones<span class="th-sub">(generales)</span></th>' in table
    assert table.count('<col ') == len(heads)
    assert '<td>2 (3)</td>' in table  # GAD3: 2 elecciones de la fixture, 3 generales
    assert 'ranking-national-events' not in page and 'data-national-events' not in page and 'Sólo con experiencia' not in page


def test_casas_ranking_with_a_known_zero_shows_zero_in_brackets(houses, monkeypatch):
    """A known count of 0 shows «(0)»; only an unknown count omits the brackets."""
    known = pd.DataFrame({
        'event_date': pd.to_datetime(['2026-11-29']), 'pollster_id': [1], 'pollster': ['CIS'], 'quality': [50.0],
        'num_events': [1], 'num_national_events': [0], 'num_polls': [3], 'error_avg': [1.0], 'error_blocks': [0.5],
        'bias_blocks': [0.5], 'rating': [60.0],
    })
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: known)
    webapi.queries._QUERIES.clear()
    page = html(call(houses, '/casas')[2])
    table = re.search(r'<table class="et et-rank" id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert '<td>1 (0)</td>' in table


def test_casas_ranking_without_num_national_events_shows_only_the_elections(houses, monkeypatch):
    """Old ratings rows (before the database column exists): the elections without brackets."""
    old = pd.DataFrame({
        'event_date': pd.to_datetime(['2026-11-29']), 'pollster_id': [1], 'pollster': ['CIS'], 'quality': [50.0],
        'num_events': [1], 'num_polls': [3], 'error_avg': [1.0], 'error_blocks': [0.5], 'bias_blocks': [0.5],
        'rating': [60.0],
    })
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: old)
    webapi.queries._QUERIES.clear()
    page = html(call(houses, '/casas')[2])
    table = re.search(r'<table class="et et-rank" id="ranking-table">(.*?)</table>', page, re.S).group(1)
    assert '<td>1</td>' in table and '<td>1 (' not in table


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


def test_casa_profile_bullseye_table_shades_the_block_errors(houses):
    """Diana de errores: las celdas de cada bloque llevan doble gradiente, rojo de -8 a 0 y verde de 0 a 8, sobre el
    valor publicado (el de la isla de datos)."""
    from mtpy.lib.pages.filters import fmt_num, shade
    body = call(houses, '/casas/gad3')[2]
    table = re.search(r'<table id="bullseye-table">(.*?)</table>', html(body), re.S).group(1)
    data = initial(body)
    assert data['bullseye'] and data['blocks']
    for row in data['bullseye']:
        for block in data['blocks']:
            value = row[block]
            colour = shade(value, 0, -8, '#d62728') if value is not None and value < 0 else shade(value, 0, 8, '#2ca02c')
            assert '<td class="cell-shade" style="background-color: {}">{}</td>'.format(colour, fmt_num(value, 1)) in table, (block, value)
    assert 'rojo si se quedó corto y verde si se pasó' in re.sub(r'\s+', ' ', html(body))


# Seguimiento 2026-10-17 (recorrido de Luis): degradados y barras en las tablas de Histórico Errores, del detalle de una
# elección, del perfil y de Promedios; el tipo de la encuestadora en castellano; el filtro de años de la diana hasta hoy.

def test_shade_around_filter_shades_both_ways_from_a_centre():
    """`shade_around`: el color de abajo crece del centro hacia centro - span y el de arriba, hacia centro + span."""
    from mtpy.lib.pages.filters import shade, shade_around
    assert shade_around(-4, 0, 8, '#d62728', '#2ca02c') == shade(-4, 0, -8, '#d62728') == 'rgba(214, 39, 40, 0.275)'
    assert shade_around(4, 0, 8, '#d62728', '#2ca02c') == shade(4, 0, 8, '#2ca02c') == 'rgba(44, 160, 44, 0.275)'
    assert shade_around(0, 0, 8, '#d62728', '#2ca02c') == 'rgba(44, 160, 44, 0.0)'  # el centro, transparente
    assert shade_around(0.5, 1, 1, '#d62728', '#7f7f7f') == 'rgba(214, 39, 40, 0.275)'  # el rebaño: rojo bajo 1
    assert shade_around(1.5, 1, 1, '#d62728', '#7f7f7f') == 'rgba(127, 127, 127, 0.275)'  # y gris sobre 1
    assert shade_around(-20, 0, 8, '#d62728', '#2ca02c') == 'rgba(214, 39, 40, 0.55)'  # saturado en el tope
    for value, span in ((None, 8), (float('nan'), 8), (True, 8), (3, 0), (3, -1)):
        assert shade_around(value, 0, span, '#d62728', '#2ca02c') == 'transparent'
    assert pages.environment().filters['shade_around'] is pages.shade_around


def test_year_range_runs_from_the_first_year_to_today():
    assert pages.year_range({2019, 2023}, date(2026, 10, 17)) == [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    assert pages.year_range([2023, 2019], date(2021, 1, 1)) == [2019, 2020, 2021, 2022, 2023]  # nunca antes del último año
    assert pages.year_range([], date(2026, 10, 17)) == []


def test_table_blocks_put_the_left_block_first_and_bias_colors_follow_the_blocks():
    """Las columnas de bloque de las tablas van con Izquierda a la izquierda (los ejes de la diana no cambian) y la barra
    del sesgo toma el color del bloque al que favorece: el primero a la derecha (positivo), el segundo a la izquierda."""
    assert pages.table_blocks(['Derecha', 'Izquierda']) == ['Izquierda', 'Derecha']
    assert pages.table_blocks(['Izquierda', 'Derecha']) == ['Izquierda', 'Derecha']
    assert pages.table_blocks(['Derecha', 'Izquierda', 'Otros']) == ['Izquierda', 'Derecha', 'Otros']
    assert pages.table_blocks(['Derecha']) == ['Derecha'] and pages.table_blocks([]) == []
    meta = fixture_data('meta')
    grey = pages.OTHERS_COLOR
    assert pages.bias_colors(['Derecha', 'Izquierda'], meta, pages.catalog(meta)) == {'pos': '#1d84ce', 'neg': grey, 'zero': grey}
    assert pages.bias_colors(['Derecha'], meta, pages.catalog(meta)) == {'pos': '#1d84ce', 'neg': grey, 'zero': grey}
    assert pages.bias_colors([], meta, pages.catalog(meta)) == {'pos': grey, 'neg': grey, 'zero': grey}


def test_mtype_label_names_the_type_of_the_pollster_in_spanish():
    assert pages.mtype_label('state') == 'Pública' and pages.mtype_label('aggr') == 'Agregador'  # el sustantivo, como la prosa del sitio
    assert pages.mtype_label('online') == 'Panel Online'  # con las iniciales en mayúscula (Luis, 2026-10-17)
    for missing in (None, '', '-', float('nan')):
        assert pages.mtype_label(missing) == 'Privada'  # sin tipo en el catálogo: una encuestadora privada corriente
    assert pages.mtype_label('otro') == 'otro'  # un valor desconocido sale tal cual


def test_elecciones_events_table_shades_the_mean_error_and_bars_the_block_bias(events_site):
    """Elecciones publicadas: sin «Error de bloques» ni «Modelo»; el error medio con fondo rojo de 0 a 8 y el sesgo
    bloques con barra divergente en ±8 en el color del bloque al que favorecieron los sondeos (el primero, azul, a la
    derecha; el segundo a la izquierda, gris en la fixture porque no tiene partido)."""
    from mtpy.lib.pages.filters import bar_sides, fmt_num, shade
    body = call(events_site, '/elecciones', query='scope=es')[2]
    page = html(body)
    table = re.search(r'<table id="events-table">(.*?)</table>', page, re.S).group(1)
    assert [re.sub(r'<.*?>', '', th) for th in re.findall(r'<th scope="col">(.*?)</th>', table)] == [
        'Elección', 'Sondeos', 'Encuestadoras', 'Error medio', 'Sesgo bloques']
    context = pages.elecciones_context(pages.parse_state({'scope': 'es'}))
    assert context['bias_colors'] == {'pos': '#1d84ce', 'neg': pages.OTHERS_COLOR, 'zero': pages.OTHERS_COLOR}
    assert len(context['events']) == 2
    for event in context['events']:
        bias = bar_sides(event['error_blocks'], 8)
        assert event['error_blocks'] > 0 and event['error_avg'] > 0
        assert '<td class="cell-shade" style="background-color: {}">{}</td>'.format(
            shade(event['error_avg'], 0, 8, '#d62728'), fmt_num(event['error_avg'], 1)) in table
        assert ('<td class="cell-bar cell-diverging"><i class="fill block" style="left: {}%; width: {}%; background: #1d84ce"></i>'
                '<span>{}</span></td>').format(bias['left'], bias['width'], fmt_num(event['error_blocks'], 1)) in table
    assert 'Modelo' not in table and '>sí<' not in table and table.count('<td') == 2 * 5
    card = re.sub(r'\s+', ' ', page[page.index('id="events"'):page.index('id="closing"')])
    assert 'hasta 8 puntos de margen' in card and 'más rojo cuanto mayor' in card and 'tiene backtest' not in card


def test_eleccion_last_table_shows_the_signed_bias_and_the_blocks_left_first(events_site):
    """Último sondeo de cada encuestadora: el error medio con fondo rojo de 0 a 8; «Sesgo bloques», con signo y barra
    divergente en ±8 en el color del bloque favorecido, en lugar del error de bloques sin signo; Izquierda antes que
    Derecha, ambas con doble gradiente rojo-verde en ±8. Los ejes de la diana siguen en el orden publicado."""
    from mtpy.lib.pages.filters import bar_sides, fmt_num, shade, shade_around
    body = call(events_site, '/elecciones/es/generales/2023-07')[2]
    page = html(body)
    table = re.search(r'<table id="last-table">(.*?)</table>', page, re.S).group(1)
    assert [re.sub(r'<.*?>', '', th) for th in re.findall(r'<th scope="col">(.*?)</th>', table)] == [
        'Encuestadora', 'Muestra', 'Días antes', 'Error medio', 'Sesgo bloques', 'Izquierda', 'Derecha']
    data = initial(body)
    assert data['blocks'] == ['Derecha', 'Izquierda'] and 'table_blocks' not in data
    rows = re.findall(r'<tr>(.*?)</tr>', table, re.S)[1:]
    assert len(rows) == len(data['last']) == 3
    colors = {'pos': data['block_colors']['Derecha'], 'neg': data['block_colors']['Izquierda'], 'zero': pages.OTHERS_COLOR}
    for row, rendered in zip(data['last'], rows):
        cells = re.findall(r'<td[^>]*>.*?</td>', rendered, re.S)
        assert len(cells) == 7
        bias = bar_sides(row['error_blocks'], 8)
        assert cells[3] == '<td class="cell-shade" style="background-color: {}">{}</td>'.format(
            shade(row['error_avg'], 0, 8, '#d62728'), fmt_num(row['error_avg'], 1))
        assert cells[4] == ('<td class="cell-bar cell-diverging"><i class="fill block" style="left: {}%; width: {}%; background: {}"></i>'
                            '<span>{}</span></td>').format(bias['left'], bias['width'], colors[bias['side']], fmt_num(row['error_blocks'], 1))
        for cell, block in zip(cells[5:], ('Izquierda', 'Derecha')):
            assert cell == '<td class="cell-shade" style="background-color: {}">{}</td>'.format(
                shade_around(row[block], 0, 8, '#d62728', '#2ca02c'), fmt_num(row[block], 1))
    assert table.index('GAD3') < table.index('40dB') < table.index('CIS')  # el orden sigue siendo el error de bloques sin signo
    card = re.sub(r'\s+', ' ', page[page.index('id="last"'):page.index('id="poll-histogram"')])
    assert 'Error de bloques' not in table and 'rojo si se quedó corto y verde si se pasó' in card and 'hasta 8 puntos de margen' in card
    assert 'Error de bloques: ' in page  # la línea resumen de la cabecera sigue dando la magnitud


def test_casa_profile_tables_shade_both_ways_and_the_bullseye_years_run_to_today(houses):
    """Perfil: la tabla de la diana pone Izquierda antes que Derecha; el último sondeo y los efectos de casa llevan
    doble gradiente rojo-verde en ±8 (ya no el rojo sin signo); el rebaño, rojo de 1 a 0 y gris de 1 a 2; el filtro
    de años ofrece todos los años desde el primer evento hasta hoy."""
    from mtpy.lib.pages.filters import fmt_num, shade_around
    body = call(houses, '/casas/gad3')[2]
    page = html(body)
    data = initial(body)
    context = pages.casa_context(pages.parse_state({}), 2)
    bullseye = re.search(r'<table id="bullseye-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col">([^<]*)</th>', bullseye) == ['Elección', 'Fecha', 'Días antes', 'Izquierda', 'Derecha']
    assert data['blocks'] == ['Derecha', 'Izquierda'] and context['table_blocks'] == ['Izquierda', 'Derecha']
    cells = re.findall(r'<td class="cell-shade" style="background-color: ([^"]*)">([^<]*)</td>', bullseye)
    assert cells == [(shade_around(row[block], 0, 8, '#d62728', '#2ca02c'), fmt_num(row[block], 1))
                     for row in data['bullseye'] for block in ('Izquierda', 'Derecha')]
    years = list(range(2023, date.today().year + 1))
    assert context['bullseye_years'] == years and len(years) > 1
    for table_id, table in (('profile-last', context['last_table']), ('profile-last-consensus', context['last_consensus_table']),
                            ('profile-effects', context['effects_table']), ('profile-effects-cycle', context['cycle_table'])):
        rendered = re.search(r'<table id="{}">(.*?)</table>'.format(table_id), page, re.S).group(1)
        expected = [(shade_around(value, 0, 8, '#d62728', '#2ca02c'), fmt_num(value, 1)) for row in table['rows'] for value in row['cells']]
        assert re.findall(r'<td class="cell-shade" style="background-color: ([^"]*)">([^<]*)</td>', rendered) == expected, table_id
    assert context['last_table']['rows'] and context['effects_table']['rows']  # las tablas comprobadas tienen filas
    herding = re.search(r'<table id="profile-herding">(.*?)</table>', page, re.S).group(1)
    expected = [(shade_around(row['ratio'], 1, 1, '#d62728', '#7f7f7f'), fmt_num(row['ratio'], 2)) for row in context['herding_table']]
    assert expected and re.findall(r'<td class="cell-shade" style="background-color: ([^"]*)">([^<]*)</td>', herding) == expected
    effects = page[page.index('id="effects"'):page.index('id="herding"')]
    assert 'rojo si le daba menos voto' in effects and 'verde si le daba más' in effects
    assert 'más rojo cuanto más por debajo de 1' in page[page.index('id="herding"'):]


def test_promedios_result_tables_drop_the_probabilities(api):
    """Resultados por partido y por bloque: sin P(primero) ni P(mayoría), que la pestaña Escaños detalla; el KPI
    «Primer partido» conserva su P(primero)."""
    page = html(call(api, '/es')[2])
    parties = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    blocks = re.search(r'<table id="blocks-result-table">(.*?)</table>', page, re.S).group(1)
    assert 'P(primero)' not in parties and 'P(mayoría)' not in blocks and '%</td>' in parties
    # Partido, % Voto, Escaños y Rango 95%: la barra del voto se fue el 2026-10-19
    assert all(row.count('<td') == 4 for row in re.findall(r'<tr>(.*?)</tr>', parties, re.S)[1:])
    assert all(row.count('<td') == 4 for row in re.findall(r'<tr>(.*?)</tr>', blocks, re.S)[1:])
    cards = page[page.index('id="parties-card"'):page.index('id="evolution-card"')]
    assert 'probabilidad de quedar primero' not in cards.split('<details')[0]  # the card lead no longer promises it
    assert 'pestaña Escaños' in cards and re.search(r'<div class="s">P\(primero\) [\d,]+%</div>', page)  # the KPI keeps it


# 2026-10-17 (segunda petición de Luis): Promedios compara con la última elección publicada del ámbito en las dos tablas
# de resultados, en puntos y en escaños, por partido y por bloque.

def test_last_results_sum_the_previous_blocks_with_the_catalogue(events_site):
    """`last_results` con catálogo añade los totales por bloque del resultado anterior, con cada partido de entonces en
    el bloque que le dan los parámetros de esa elección (Cs en Derecha aunque ya no exista)."""
    catalog = pages.catalog(fixture_data('meta'))
    results = pages.last_results('es', catalog)
    assert results['date'] == '2023-07-23' and results['parties']['PP'] == {'seats': 160, 'pct': 33.06}
    assert results['blocks'] == {'Derecha': {'pct': 45.48, 'seats': 200}, 'Izquierda': {'pct': 31.44, 'seats': 150}}
    assert 'blocks' not in pages.last_results('es')  # sin catálogo, como lo usa Inicio


def test_result_rows_carry_the_deltas_against_the_previous_election():
    summary = {'parties': [{'name': 'PP', 'pct': 33.1, 'seats': 135, 'p_first': 0.6},
                           {'name': 'Sumar', 'pct': 12.4, 'seats': 25, 'p_first': 0.0}],
               'blocks': [{'name': 'Derecha', 'pct': 45.2, 'seats': 178, 'p_majority': 0.4},
                          {'name': 'Izquierda', 'pct': 40.0, 'seats': 160, 'p_majority': 0.1}]}
    results = {'date': '2023-07-23', 'year': 2023, 'parties': {'PP': {'seats': 160, 'pct': 33.06}, 'UP': {'seats': 31, 'pct': 12.3}},
               'blocks': {'Derecha': {'pct': 45.48, 'seats': 200}, 'Izquierda': {'pct': 31.44, 'seats': 150}}}
    rows = pages.result_rows(summary, results)
    assert [(row['name'], row['prev_pct'], row['prev_seats'], row['delta_pct'], row['delta_seats']) for row in rows] == [
        ('PP', 33.06, 160, 0.04, -25), ('Sumar', None, None, None, None)]  # Sumar no concurrió: sin diferencia
    meta = fixture_data('meta')
    blocks = pages.block_result_rows(summary, meta, pages.Catalog(meta), results)
    assert [(row['name'], row['prev_pct'], row['prev_seats'], row['delta_pct'], row['delta_seats']) for row in blocks] == [
        ('Derecha', 45.48, 200, -0.28, -22), ('Izquierda', 31.44, 150, 8.56, 10)]
    # sin resultados anteriores, o sin bloques en ellos, las cuatro claves van a None
    assert all(row[key] is None for row in pages.result_rows(summary) for key in ('prev_pct', 'prev_seats', 'delta_pct', 'delta_seats'))
    assert all(row['delta_seats'] is None for row in pages.block_result_rows(summary, meta, pages.Catalog(meta), {'date': '2023-07-23', 'parties': {}}))
    # las diferencias de escaños que llevan los hemiciclos y la barra de Escaños: sólo las que existen, con el año
    assert pages.seat_deltas(summary, meta, pages.Catalog(meta), results) == {'parties': {'PP': -25}, 'blocks': {'Derecha': -22, 'Izquierda': 10}, 'year': 2023}
    assert pages.seat_deltas(summary, meta, pages.Catalog(meta), None) == {'parties': {}, 'blocks': {}, 'year': None}


def test_promedios_tables_compare_with_the_previous_election(events_site, fresh_app):
    """Con una elección publicada en el ámbito, las dos tablas llevan una columna «vs 2023» tras el voto y otra tras
    los escaños: una insignia con flecha verde si sube y roja si baja (sin degradados, Luis 2026-10-17) y un guion
    sin cambio o para un partido que no concurrió entonces."""
    summary = fixture_data('summary')
    summary['parties'].append(dict(summary['parties'][0], name='XX', seats=3, pct=1.0, seats_lo=0, seats_hi=5, p_first=0))
    summary['totals']['XX'] = 3
    writer = write_bundle(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'summary', 'nowcast'), 'summary', summary, 'es', run_id=RUN, mode='nowcast')
    webapi.site().manifest_cache_clear()
    page = html(call(events_site, '/es')[2])
    parties = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    blocks = re.search(r'<table id="blocks-result-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', parties) == ['Partido', '% Voto', 'vs 2023', 'Escaños', 'vs 2023', 'Rango 95%']
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', blocks) == ['Bloque', '% Voto', 'vs 2023', 'Escaños', 'vs 2023', 'Rango 95%']
    assert parties.count('title="Diferencia en puntos frente al resultado oficial del 23 jul 2023"') == 1
    assert parties.count('title="Diferencia en escaños frente al resultado oficial del 23 jul 2023"') == 1
    rows = re.findall(r'<tr>(.*?)</tr>', parties, re.S)[1:]
    cells = lambda row: re.findall(r'<td><span class="dlt (\w+)">([^<]*)</span></td>', row)
    # PP: 33,1 − 33,06 = +0,04 → sin cambio apreciable; 135 − 160 = ▼ 25
    assert cells(rows[0]) == [('flat', '–'), ('down', '▼ 25')]
    assert cells(rows[1]) == [('flat', '–'), ('flat', '–')]  # XX no concurrió en 2023
    assert 'cell-shade' not in parties and 'cell-shade' not in blocks  # sin degradados en estas columnas
    # Derecha: 33,1 − 45,48 = −12,38 → ▼ 12,4; 135 − 200 = ▼ 65
    rows = re.findall(r'<tr>(.*?)</tr>', blocks, re.S)[1:]
    assert cells(rows[0]) == [('down', '▼ 12,4'), ('down', '▼ 65')]
    cards = re.sub(r'\s+', ' ', page[page.index('id="hemicycle-card"'):page.index('id="evolution-card"')])
    assert 'frente al resultado oficial' in cards and 'un guion' in cards and 'ya no existen' in cards
    assert 'flecha verde si sube y roja si baja' in cards and 'cuánto cambia frente a 2023' in cards  # el hemiciclo también lo dice
    # el hemiciclo recibe sólo las diferencias que existen (XX no concurrió), con el año
    assert initial(call(events_site, '/es')[2])['deltas'] == {'parties': {'PP': -25}, 'blocks': {'Derecha': -65}, 'year': 2023}


def test_promedios_tables_without_a_published_election_have_no_comparison(api):
    page = html(call(api, '/es')[2])
    parties = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    assert 'vs ' not in parties and 'cell-shade' not in parties and 'frente al resultado oficial' not in page


# Recorrido de Luis (2026-10-17, tarde): insignias con flecha en vez de degradados en las diferencias, también en los
# hemiciclos de Inicio y Promedios y en la barra de Escaños.

def test_delta_badge_filter_gives_an_arrow_and_a_direction():
    from mtpy.lib.pages.filters import delta_badge
    assert delta_badge(6) == {'dir': 'up', 'text': '▲ 6'} and delta_badge(-1) == {'dir': 'down', 'text': '▼ 1'}
    assert delta_badge(0) == {'dir': 'flat', 'text': '–'}
    assert delta_badge(0.04, 1) == {'dir': 'flat', 'text': '–'}  # redondea a 0,0: sin cambio apreciable
    assert delta_badge(-12.38, 1) == {'dir': 'down', 'text': '▼ 12,4'} and delta_badge(1.0, 1) == {'dir': 'up', 'text': '▲ 1,0'}
    assert delta_badge(-0.05, 1) == {'dir': 'down', 'text': '▼ 0,1'}  # medio punto arriba, como `delta`
    for missing in (None, float('nan'), True, 'x'):
        assert delta_badge(missing) == {'dir': 'flat', 'text': '–'}
    assert pages.environment().filters['badge'] is pages.delta_badge


def test_home_and_seats_pages_embed_the_seat_deltas_for_their_hemicycle_and_bar(events_site):
    """El hemiciclo de Inicio y la barra de Escaños reciben las diferencias de escaños frente a la última elección."""
    hero = initial(call(events_site, '/')[2])['hero']
    assert hero['deltas'] == {'PP': -25} and hero['results_year'] == 2023
    data = initial(call(events_site, '/es/escanos')[2])
    assert data['deltas'] == {'parties': {'PP': -25}, 'blocks': {'Derecha': -65}, 'year': 2023}


def test_escanos_tables_compare_the_seats_with_the_previous_election(events_site):
    """Las dos tablas de Escaños llevan una columna «vs 2023» tras los escaños con la flecha de la diferencia (sólo
    escaños: no hay voto en estas tablas); sin elección publicada, no aparece."""
    page = html(call(events_site, '/es/escanos')[2])
    parties = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    blocks = re.search(r'<table id="blocks-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', parties) == ['Partido', 'Escaños', 'vs 2023', 'Rango 95%', 'P(escaño)', 'P(primero)', 'P(mayoría)']
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', blocks) == ['Bloque', 'Escaños', 'vs 2023', 'Rango 95%', 'P(mayoría)']
    assert parties.count('title="Diferencia en escaños frente al resultado oficial del 23 jul 2023"') == 1
    assert re.search(r'<td><strong>135</strong></td>\s*<td><span class="dlt down">▼ 25</span></td>', parties)
    assert re.search(r'<td><strong>135</strong></td>\s*<td><span class="dlt down">▼ 65</span></td>', blocks)
    assert 'frente al resultado oficial del 23 jul 2023' in re.sub(r'\s+', ' ', page[page.index('id="seats-bar"'):page.index('id="ridge-chart"') + 2000])


def test_escanos_tables_without_a_published_election_have_no_comparison(api):
    plain = html(call(api, '/es/escanos')[2])
    assert 'vs 2023' not in plain and 'class="dlt' not in plain and initial(call(api, '/es/escanos')[2])['deltas'] == {'parties': {}, 'blocks': {}, 'year': None}


# Seguimiento 2026-10-20 (Luis): un partido que absorbe a otro en el `smap` del evento (FA a Sumar, PP a UPN) se compara
# con lo que obtuvieron ambos en la última elección, en vez de salir sin diferencia.

def test_inherited_results_sum_the_parties_a_run_party_absorbs():
    """`inherited_results`: el resultado previo de un partido es el suyo más el de los que absorbe por reglas `agg`
    sin regiones (el linaje del promedio de encuestas, `Simulator.poll_blocks`); las reglas `sub`, las regionales y
    las fuentes que el run simula aparte no cuentan; los resultados originales no se tocan."""
    results = {'date': '2023-07-23', 'year': 2023,
               'parties': {'PP': {'seats': 137, 'pct': 33.06}, 'SUMAR': {'seats': 31, 'pct': 12.33},
                           'UPN': {'seats': 1, 'pct': 0.21}, 'VOX': {'seats': 33, 'pct': 12.38}},
               'blocks': {'Derecha': {'seats': 171, 'pct': 45.65}}}
    meta = {'parties': [{'name': name} for name in ('PP', 'VOX', 'FA', 'UP', 'SALF', 'COMPROMIS')],
            'smap': {'FA': [{'type': 'agg', 'names': ['SUMAR']},
                            {'type': 'agg', 'names': [], 'regions': {'exclude': ['3', '12', '46']}}],
                     'UP': [{'type': 'sub', 'names': ['FA']}],
                     'COMPROMIS': [{'type': 'agg', 'names': ['SUMAR'], 'regions': ['3', '12', '46']}],
                     'PP': {'type': 'agg', 'names': ['UPN']},  # una regla suelta, sin lista, también vale
                     'SALF': [{'type': 'agg', 'names': ['VOX']}],  # VOX se simula aparte: conserva su resultado
                     'VOX': 'VOX'}}  # el formato antiguo del fixture se ignora
    out = pages.inherited_results(results, meta)
    assert out['parties']['FA'] == {'seats': 31, 'pct': 12.33, 'inherited': ['SUMAR']}
    assert out['parties']['PP'] == {'seats': 138, 'pct': 33.27, 'inherited': ['PP', 'UPN']}
    assert 'UP' not in out['parties'] and 'COMPROMIS' not in out['parties'] and 'SALF' not in out['parties']
    assert out['parties']['VOX'] == {'seats': 33, 'pct': 12.38} and out['parties']['SUMAR'] == {'seats': 31, 'pct': 12.33}
    assert out['blocks'] == results['blocks'] and out['year'] == 2023 and out['date'] == '2023-07-23'
    assert 'FA' not in results['parties'] and results['parties']['PP'] == {'seats': 137, 'pct': 33.06}  # sin tocar
    assert pages.inherited_results(None, meta) is None
    assert pages.inherited_results(results, {})['parties'] == results['parties']
    # sin escaños ni voto en ninguna de las partes, la diferencia queda sin valor
    bare = pages.inherited_results({'parties': {'SUMAR': {'seats': None, 'pct': None}}}, meta)
    assert bare['parties']['FA'] == {'seats': None, 'pct': None, 'inherited': ['SUMAR']}


def test_result_rows_and_cards_carry_whom_a_party_is_compared_with():
    """Las filas y las tarjetas de Promedios llevan `prev_from`, los partidos cuyo resultado suma la comparación
    (vacío cuando es el propio), y las diferencias se calculan sobre esa suma."""
    summary = fixture_data('summary')  # PP: 135 escaños, 33,1 %
    meta = fixture_data('meta') | {'smap': {'PP': [{'type': 'agg', 'names': ['PSOE']}]}}
    results = {'date': '2023-07-23', 'year': 2023, 'blocks': {},
               'parties': {'PP': {'seats': 160, 'pct': 33.06}, 'PSOE': {'seats': 150, 'pct': 31.44}}}
    previous = pages.inherited_results(results, meta)
    row = pages.result_rows(summary, previous)[0]
    assert (row['prev_pct'], row['prev_seats'], row['delta_pct'], row['delta_seats']) == (64.5, 310, -31.4, -175)
    assert row['prev_from'] == ['PP', 'PSOE']
    assert pages.result_rows(summary, results)[0]['prev_from'] == []
    card = pages.party_cards(summary, meta, pages.Catalog(meta), previous)[0]
    assert (card['delta_seats'], card['delta_pct'], card['prev_from']) == (-175, -31.4, ['PP', 'PSOE'])
    assert pages.seat_deltas(summary, meta, pages.Catalog(meta), previous)['parties'] == {'PP': -175}


def test_promedios_compares_a_successor_with_the_parties_it_absorbs(events_site, fresh_app):
    """Con una regla `agg` en el `smap` del run (el PP absorbe al PSOE en el fixture), las columnas «vs 2023», las
    tarjetas y el hemiciclo de Promedios, las tablas y la barra de Escaños y el hemiciclo de Inicio comparan con la
    suma de ambos resultados, y la flecha dice con quién al pasar el ratón."""
    meta = fixture_data('meta') | {'smap': {'PP': [{'type': 'agg', 'names': ['PSOE']}]}}
    writer = write_bundle(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().manifest_cache_clear()
    page = html(call(events_site, '/es')[2])
    parties = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    row = re.findall(r'<tr>(.*?)</tr>', parties, re.S)[1]
    # PP: 33,1 − (33,06 + 31,44) = −31,4; 135 − (160 + 150) = −175
    assert re.findall(r'<td><span class="dlt (\w+)" title="([^"]*)">([^<]*)</span></td>', row) == [
        ('down', 'Frente a PP + PSOE en 2023', '▼ 31,4'), ('down', 'Frente a PP + PSOE en 2023', '▼ 175')]
    card = re.search(r'<article class="pc"[^>]*data-name="PP">(.*?)</article>', page, re.S).group(1)
    assert 'title="Flechas: frente al resultado de PP + PSOE el 23 jul 2023"' in card and '▼ 175' in card and '▼ 31,4' in card
    assert initial(call(events_site, '/es')[2])['deltas']['parties'] == {'PP': -175}
    assert 'suma lo que obtuvieron ambos' in re.sub(r'\s+', ' ', page)  # «Cómo leerlo» lo explica
    seats = html(call(events_site, '/es/escanos')[2])
    assert re.search(r'<td><strong>135</strong></td>\s*<td><span class="dlt down" title="Frente a PP \+ PSOE en 2023">▼ 175</span></td>', seats)
    assert initial(call(events_site, '/es/escanos')[2])['deltas']['parties'] == {'PP': -175}
    assert initial(call(events_site, '/')[2])['hero']['deltas'] == {'PP': -175}


# Seguimiento 2026-10-19: el rating de la encuestadora en «Todas las encuestas» (Promedios y detalle de una elección),
# como badge en degradado rojo (0) - gris (50) - verde (100) entre la encuestadora y la fecha; el nombre enlaza al perfil.

def test_gradient_filter_mixes_the_stops_spread_over_the_range():
    """`gradient`: el color de un valor en una escala de varios tramos, los colores repartidos por igual entre los
    extremos; recortado en los extremos; transparente sin valor, sin rango o sin dos colores hexadecimales."""
    from mtpy.lib.pages.filters import gradient
    stops = ['#d62728', '#7f7f7f', '#2ca02c']
    assert gradient(0, 0, 100, stops) == '#d62728' and gradient(50, 0, 100, stops) == '#7f7f7f'
    assert gradient(100, 0, 100, stops) == '#2ca02c'
    assert gradient(25, 0, 100, stops) == '#ab5354' and gradient(75, 0, 100, stops) == '#569056'
    assert gradient(5, 0, 100, stops) == '#cd3031'
    assert gradient(-10, 0, 100, stops) == '#d62728' and gradient(140, 0, 100, stops) == '#2ca02c'  # recortado
    assert gradient(75, 100, 0, stops) == '#ab5354'  # el rango puede ir hacia abajo
    assert gradient(0.5, 0, 1, ['#000000', '#ffffff']) == '#808080'  # el medio redondea hacia arriba, como en JS
    for value, start, end, colors in ((None, 0, 100, stops), (float('nan'), 0, 100, stops), (True, 0, 100, stops),
                                      (3, 1, 1, stops), (3, 0, 100, ['#d62728']), (3, 0, 100, ['red', '#7f7f7f']),
                                      (3, 0, 100, [])):
        assert gradient(value, start, end, colors) == 'transparent', (value, start, end, colors)
    assert pages.RATING_COLORS == ('#d62728', '#7f7f7f', '#2ca02c')
    assert pages.environment().filters['gradient'] is pages.gradient


def test_polls_table_rows_carry_the_rating_badge_and_the_profile_link():
    polls = [
        {'date': '2026-10-01', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'rating': 73.08, 'PP': 31.1},
        {'date': '2026-09-04', 'pollster': 'Ateneo del Dato', 'pollster_id': 9, 'rating': 5, 'PP': 25.5},
        {'date': '2026-09-02', 'pollster': 'CIS', 'PP': 30.0},
    ]
    rows = pages.polls_table(polls, ['PP'], {})['rows']
    assert [(row['rating'], row['rating_color']) for row in rows] == [
        ('73,1', pages.gradient(73.08, 0, 100, pages.RATING_COLORS)), ('5,0', '#cd3031'), ('–', None)]
    assert [row['href'] for row in rows] == ['/casas/sigma-dos', '/casas/ateneo-del-dato', None]


def test_chart_polls_keep_the_rating_and_add_the_profile_link():
    polls = {'parties': ['PP'], 'polls': [
        {'date': '2026-10-01', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'rating': 73.08, 'PP': 31.1},
        {'date': '2026-09-02', 'pollster': 'CIS', 'PP': 30.0},
    ]}
    out = pages.chart_polls(polls)['polls']
    assert out[0]['rating'] == 73.08 and out[0]['href'] == '/casas/sigma-dos'
    assert 'rating' not in out[1] and out[1]['href'] is None


def test_promedios_polls_table_shows_the_rating_badge_between_the_house_and_the_date_and_links_the_profile(api, fresh_app):
    polls = fixture_data('polls')
    base = polls['polls'][0]
    polls['polls'] = [base | {'date': '2026-10-01', 'rating': 73.08},
                      base | {'date': '2026-09-01', 'pollster': 'Ateneo del Dato', 'pollster_id': 9, 'rating': 5.0},
                      base | {'date': '2026-08-01', 'pollster': 'Sin casa', 'pollster_id': None}]
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', polls, 'es', run_id=RUN)
    webapi.site().files.clear()
    body = call(api, '/es')[2]
    page = html(body).replace('\n', ' ')
    start = page.index('id="polls-table"')
    head = page[start:page.index('<tbody>', start)]
    assert '<col class="c-f"><col class="c-r"><col class="c-d"><col class="c-s">' in head
    assert head.index('data-key="pollster"') < head.index('data-key="rating"') < head.index('data-key="date"')
    assert re.search(r'<th scope="col" data-key="rating" title="[^"]+"><button type="button" class="sort">Rating \(0-100\)</button></th>', head)
    rows = re.findall(r'<tr>(.*?)</tr>', polls_tbody(page))
    assert len(rows) == 3
    assert '<a href="/casas/sigma-dos"><b>Sigma Dos</b></a>' in rows[0]
    assert '<td><span class="rt" style="background: {}">73,1</span></td>'.format(pages.gradient(73.08, 0, 100, pages.RATING_COLORS)) in rows[0]
    assert rows[0].index('</a>') < rows[0].index('class="rt"') < rows[0].index('01/10/26')
    assert '<td><span class="rt" style="background: #cd3031">5,0</span></td>' in rows[1]
    assert '<b>Sin casa</b>' in rows[2] and '<a href' not in rows[2] and '<td class="na">—</td>' in rows[2]
    foot = page[page.index('<tfoot>', start):page.index('</tfoot>', start)]
    assert foot.count('<td></td>') == 3  # rating, fecha y muestra vacías bajo la media simple
    assert 'El rating (0-100)' in page and 'rojo (0)' in page and 'su perfil' in page


def test_eleccion_cycle_table_shows_the_rating_each_house_arrived_with_and_links_its_profile(events_site):
    body = call(events_site, '/elecciones/es/generales/2023-07')[2]
    page = html(body).replace('\n', ' ')
    tbody = polls_tbody(page)
    assert 'Rating (0-100)' in page and 'llegó a esta elección' in page
    assert '<a href="/casas/gad3"><b>GAD3</b></a>' in tbody
    assert '<span class="rt" style="background: {}">0,9</span>'.format(pages.gradient(0.9, 0, 100, pages.RATING_COLORS)) in tbody
    data = initial(body)['polls']['polls']
    assert data[0]['rating'] == 0.8 and data[0]['href'] == '/casas/cis'


# --- Análisis ------------------------------------------------------------------------------------------------------

STUDY_VERDICTS = [('q1', 'SÍ'), ('q2', 'SIN EVIDENCIA'), ('q3', 'NO')]


def publish_study(fs, meta=None, parts=None):
    """Publica el estudio de los fixtures y vacía las cachés del sitio."""
    write_study(fs, meta=meta, parts=parts)
    webapi.site().manifest_cache_clear()


def verdict_pills(page):
    """`(tono, etiqueta)` de las píldoras de veredicto de una página, en orden."""
    return re.findall(r'<span class="pill (q[123])"[^>]*>([^<]+)</span>', page)


def study_cards_html(page):
    """El HTML de la rejilla `#studies` de las tarjetas de los estudios (las tarjetas no anidan `div`)."""
    cards = page[page.index('<div class="studies" id="studies">'):]
    return cards[:cards.index('</div>')]


def test_menu_has_analisis_between_ranking_and_metodo(api):
    page = html(call(api, '/metodo')[2])
    nav = page[page.index('<nav aria-label="Secciones">'):]
    nav = nav[:nav.index('</nav>')]
    assert re.findall(r'href="([^"]+)"', nav) == ['/', '/elecciones', '/casas', '/analisis', '/metodo']
    assert '>Análisis</a>' in nav and 'aria-current' not in nav.split('/analisis')[1].split('</a>')[0]
    # the tab bar of the phones keeps its five entries: the Análisis pages light none
    assert [row['key'] for row in pages.TABBAR] == ['inicio', 'promedios', 'escanos', 'historico', 'ranking']
    assert '/analisis' not in pages.common.TABBAR_ACTIVE
    page = html(call(api, '/analisis')[2])
    tabbar = page[page.index('class="tabbar'):]
    assert 'aria-current' not in tabbar[:tabbar.index('</nav>')]


def test_analisis_index_lists_the_study_or_says_none(api, fresh_app):
    status, _, body = call(api, '/analisis')
    page = html(body)
    assert status == 200 and '<title>Demoscopio · Análisis</title>' in page
    assert 'href="/analisis" aria-current="page">Análisis</a>' in page
    assert 'Preguntas del debate electoral contestadas con datos oficiales, comprobables después de cada elección' in page
    assert 'Todavía no hay ningún análisis publicado.' in page and 'href="/analisis/voto-cera"' not in page
    publish_study(fresh_app.fs)
    page = html(call(api, '/analisis')[2])
    meta = fixture_data('study-meta')
    assert 'Todavía no hay ningún análisis publicado.' not in page
    # two cards per row on desktop, each with only the title, the subtitle and the link to the study
    cards = study_cards_html(page)
    assert meta['title'] in cards and meta['subtitle'] in cards
    assert '<a class="btn pri" href="/analisis/voto-cera">Ver el análisis completo' in cards
    assert verdict_pills(cards) == [] and 'Predicción registrada' not in cards and 'Comprobada el' not in cards


def test_analisis_unknown_study_is_404(api, fresh_app):
    for path in ('/analisis/nope', '/analisis/voto-cera'):
        status, headers, body = call(api, path)
        assert status == 404 and headers['cache-control'] == 'no-store', path
        assert 'Ese análisis no existe o no está publicado.' in html(body), path
    publish_study(fresh_app.fs)
    assert call(api, '/analisis/nope')[0] == 404
    status, _, body = call(api, '/analisis/voto-cera')
    page = html(body)
    assert status == 200
    assert '<title>Demoscopio · El voto CERA en España y el efecto de la «Ley de Nietos»</title>' in page
    assert '¿Es posible orquestar con éxito un fraude electoral significativo a través del censo exterior?' in page
    assert 'href="/analisis" aria-current="page">Análisis</a>' in page
    assert 'class="kpis"' in page and '2.769.645' in page
    assert '<span class="dlt up">▲ 19%</span> frente a 2023 (2.327.388)' in page  # the arrow is not escaped
    assert set(initial(body)) == {'state', 'census', 'origins', 'participation', 'ecological', 'slope', 'cannibal',
                                  'sensitive', 'gap', 'radar'}


def test_inicio_shows_the_studies_band_only_with_a_study(api, fresh_app):
    write_event(fresh_app.fs)  # a past election, so the band goes after «Elecciones pasadas»
    page = html(call(api, '/')[2])
    assert 'id="analisis"' not in page and 'href="/analisis/voto-cera"' not in page
    publish_study(fresh_app.fs)
    page = html(call(api, '/')[2])
    assert '<div class="sec-h" id="analisis"><h2>Análisis</h2><a href="/analisis">Ver los análisis ›</a></div>' in page
    band = page[page.index('id="analisis"'):]
    assert page.index('id="past"') < page.index('id="analisis"')
    # the same card as the index of the Análisis: title, subtitle and the button, no verdicts and no dates
    meta = fixture_data('study-meta')
    cards = study_cards_html(band)
    assert '<article class="card study-card">' in cards
    assert '<h3>{}</h3>'.format(meta['title']) in cards and meta['subtitle'] in cards
    assert '<a class="btn pri" href="/analisis/voto-cera">Ver el análisis completo' in cards
    assert verdict_pills(cards) == [] and 'Predicción registrada' not in cards and 'Comprobada el' not in cards


def test_api_url_study():
    assert pages.api_url('voto-cera', 'prediction', kind='study', fmt='csv', table='last-seat') == \
        '/api/v1/studies/voto-cera/prediction?format=csv&table=last-seat'
    assert pages.api_url('voto-cera', 'meta', kind='study') == '/api/v1/studies/voto-cera/meta'


def test_cera_context_figures_come_from_the_parts(api, fresh_app):
    publish_study(fresh_app.fs)
    context = pages.cera_context(pages.parse_state({}))
    meta, check = fixture_data('study-meta'), fixture_data('study-check')
    scenario = fixture_data('study-prediction')['scenarios']['convocatoria']
    previous = next(row for row in check['elections'] if row['date'] == '2023-07-23')
    figures = context['figures']
    assert figures['n_over_gap_2023'] == previous['n_over_gap']
    assert figures['decided_2023'] == [row['name'] for row in previous['decided']]
    assert figures['net_psoe_2023'] == previous['net_psoe']
    assert figures['census_total'] == 2769645 == meta['election']['census_total']
    assert figures['electors_2023'] == scenario['census'] - scenario['new_electors']
    # the base of the increment (the monthly close) and Infoelectoral's census of 2023, both from the parts
    assert figures['baseline_date'] == meta['election']['baseline_date'] == '2023-06-01'
    assert figures['electors_official'] == 2328260 and figures['electors_gap'] == 2328260 - 2327388
    assert figures['growth_abs'] == scenario['new_electors']
    assert figures['growth_pct'] == pytest.approx(100 * scenario['new_electors'] / figures['electors_2023'])
    assert figures['voters_projected'] == scenario['voters_projected']
    assert figures['votes_projected'] == scenario['votes_projected']
    assert figures['voters_2023'] == scenario['voters_2023'] and figures['changed'] is False
    assert figures['net_before'] == meta['regression']['net_before']
    assert figures['p_slope'] == meta['regression']['p_slope']
    # the page shows the periods updated to the last close (`now`); the article's window stays for «Cómo leerlo»
    assert (figures['g1'], figures['g2'], figures['ratio'], figures['periods_version']) == (600, 900, 1.5, 'now')
    assert (figures['g1_article'], figures['g2_article'], figures['ratio_article']) == (600, 1200, 2.0)
    assert figures['r'] == -0.743 and figures['n_ecological'] == 3
    assert figures['max_gain'] == 1 and figures['pool'] == 181032 and figures['grid_observed'] == 0
    assert figures['report']['n_max'] == 3 and figures['sensitive_all'] == {'stock_share': 25.0, 'growth_share': 18.1}
    assert figures['sensitive_countries'] == {'stock_share': 31.0, 'growth_share': 27.5}
    assert figures['sensitive_version'] == 'now'
    # the litigation and the participation rates come from `params`
    assert figures['litigation'] == 163020 and figures['latam_rate'] == 5 and figures['rest_rate'] == 10
    assert context['registered'] == {'at': '2026-10-18', 'claim': meta['registered']['claim']}
    assert context['checked'] is False and context['sources'] == meta['sources']
    assert [verdict['label'] for verdict in context['verdicts']] == ['SÍ', 'SIN EVIDENCIA', 'NO']
    assert [kpi['label'] for kpi in context['kpis']] == ['Censo CERA 29-N', 'Inscripciones en litigio',
                                                         'Votantes CERA esperados']
    assert context['kpis'][0]['value'] == '2.769.645' and context['kpis'][2]['value'] == '239.221'
    assert len(context['csv_links']) == 11
    assert {'label': 'Último escaño por provincia',
            'href': '/api/v1/studies/voto-cera/prediction?format=csv&table=last-seat'} in context['csv_links']
    assert context['article']['date'] == '2026-06-30' and context['article']['url'].startswith('https://medium.com/')
    assert [page['href'] for page in context['pages'] if page['active']] == ['/analisis']
    gap = context['initial']['gap']
    assert set(gap) == {'convocatoria', 'sin_litigio'} and gap['sin_litigio'][0]['net'] == 6.9
    assert context['initial']['radar']['max'] == [100, 100, 100]
    # the KPI rows quote the litigation and the rates of `params`
    kpis = context['kpis']
    assert '<span class="dlt up">▲ 19%</span> frente a 2023 (2.327.388)' in str(kpis[0]['support'])
    assert kpis[1]['value'] == '163.020'
    assert kpis[1]['support'] == ('suspendidas por el Supremo el 10-09, mantenido el 12-10 · '
                                  'censo vigente: cerrado el 1-10-2026')
    assert kpis[2]['support'] == '~5% Latam · ~10% resto · 2023: 208.431'
    # without them in `params` and without a registration: no error, `None`, and the KPI notes drop their pieces
    publish_study(fresh_app.fs, meta={'params': {'election': meta['params']['election']}, 'registered_at': None,
                                      'registered': None})
    context = pages.cera_context(pages.parse_state({}))
    assert context['registered'] is None and context['figures']['litigation'] is None
    assert context['figures']['latam_rate'] is None and context['figures']['rest_rate'] is None
    assert context['kpis'][1]['support'] == 'censo vigente: cerrado el 1-10-2026'
    assert context['kpis'][2]['support'] == '2023: 208.431'


def test_cera_page_loads_its_module_and_the_island_carries_what_the_charts_draw(api, fresh_app):
    publish_study(fresh_app.fs)
    body = call(api, '/analisis/voto-cera')[2]
    assert '<script type="module" src="/dist/js/pages/analisis-cera.js"></script>' in html(body)
    island = initial(body)
    census = fixture_data('study-census')
    assert set(island['census']) == {'series', 'marks', 'periods'}
    assert island['census']['periods'] == census['periods']['now']  # the updated periods, as the page
    assert island['origins'][0] == {'name': 'Latinoamérica', 'g1': 100, 'g2': 150, 'ratio': 1.5}
    assert island['sensitive'] == fixture_data('study-placement')['sensitive']['now']
    # the parties of the last seat label the rows of the gap chart
    for rows in island['gap'].values():
        assert all(set(row) == {'name', 'winner', 'runner_up', 'gap', 'net'} for row in rows)
    assert island['gap']['convocatoria'][0] == {'name': 'Madrid', 'winner': 'PP', 'runner_up': 'PSOE', 'gap': 120.5,
                                                'net': 7.1}


def cera_census(n):
    """La parte `census` del fixture con `n` provincias sintéticas (`P01`...): las altas se duplican de un periodo
    del artículo al otro (×1,5 en los periodos actualizados) y crecen con el número de la provincia del inicio del
    segundo al censo de la convocatoria."""
    census = fixture_data('study-census')
    provinces = [{'region_id': i, 'name': 'P{:02d}'.format(i), 'c1_start': 900, 'c1_end': 1000, 'c2_start': 1000,
                  'c2_end': 1200, 'g1': 100, 'g2': 200, 'ratio': 2.0, 'c1_start_now': 900, 'c1_end_now': 1000,
                  'c2_start_now': 1000, 'c2_end_now': 1150, 'g1_now': 100, 'g2_now': 150, 'ratio_now': 1.5}
                 for i in range(1, n + 1)]
    growth = [{'region_id': i, 'name': 'P{:02d}'.format(i), 'baseline': 1100, 'election': 1000 + 10 * i,
               'growth': 10 * i - 100, 'pct': None} for i in range(1, n + 1)]
    return census | {'provinces': provinces, 'election_provinces': growth}


def cera_page(api):
    """La página del estudio CERA con los espacios normalizados (las frases de las plantillas cortan líneas)."""
    return re.sub(r'\s+', ' ', html(call(api, '/analisis/voto-cera')[2]))


def test_cera_page_head_and_question_1(api, fresh_app):
    publish_study(fresh_app.fs, parts={'census': cera_census(12)})
    page = cera_page(api)
    assert '<title>Demoscopio · El voto CERA en España y el efecto de la «Ley de Nietos»</title>' in page
    assert ('<p class="study-subtitle">¿Es posible orquestar con éxito un fraude electoral significativo a través '
            'del censo exterior?</p>') in page
    head = page[page.index('id="study-head"'):page.index('class="study-title"')]
    assert re.findall(r'<span class="(pill[^"]*)">([^<]+)</span>', head) == [
        ('pill brand', 'Análisis'), ('pill', 'Generales del 29 de noviembre de 2026'),
        ('pill warn', 'Predicción registrada · se comprueba tras el escrutinio CERA')]
    toc = page[page.index('<nav class="section-nav toc"'):]
    toc = toc[:toc.index('</nav>')]
    assert re.findall(r'href="([^"]+)"', toc) == ['#resumen', '#p1', '#p2', '#p3', '#debate', '#comprobacion', '#datos']
    assert '<a href="#p1" class="q1">1 · ¿Se dispara el censo?</a>' in toc
    assert 'class="q2">2 · ¿Sesgo o colocación?</a>' in toc and 'class="q3">3 · ¿Mueven escaños?</a>' in toc
    assert '2.769.645' in page[page.index('class="kpis"'):page.index('id="resumen"')]
    resumen = page[page.index('id="resumen"'):page.index('id="p1"')]
    assert re.findall(r'<div class="vv">([^<]+)</div>', resumen) == ['SÍ', 'SIN EVIDENCIA', 'NO']
    assert re.findall(r'<a class="(q[123])" href="#(p[123])">', resumen) == [('q1', 'p1'), ('q2', 'p2'), ('q3', 'p3')]
    assert '(×1,5)' in resumen and '~5%' in resumen and '~10%' in resumen  # the pace of the updated periods
    assert 'el 18% de las altas con el 25% del censo' in resumen
    assert 'Con el censo real, la participación real y el reparto real: cero escaños.' in resumen
    assert 'class="note blue"' in resumen and 'los datos por encima de bandos y consignas.</p>' in resumen
    assert page[:page.index('id="p1"')].endswith('<section class="card qcard q1" ')
    p1 = page[page.index('id="p1"'):]
    p1 = p1[:p1.index('</section>')]
    assert '<div class="qn">1</div><h2 class="qt">¿Se dispara el censo de los españoles en el exterior?</h2>' in p1
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', p1) == ['SÍ']
    for chart in ('census-chart', 'periods-chart'):
        assert re.search(r'<div class="chart" id="{}" role="img" aria-label="[^"]+"></div>'.format(chart), p1)
    countries = p1[p1.index('id="countries-table"'):]
    countries = countries[:countries.index('</table>')]
    assert '<td><b>Venezuela</b></td><td>200</td><td><b>300</b></td><td>×1,5</td>' in countries
    assert countries.index('Venezuela') < countries.index('Argentina')  # by the additions after the law
    provinces = p1[p1.index('id="provinces-table"'):]
    provinces = provinces[:provinces.index('</table>')]
    assert '<th scope="col">Censo 01-2026</th>' in provinces  # the start of the second updated period
    rows = re.findall(r'<tr( hidden)?>\s*<td><b>(P\d\d)</b></td>', provinces)
    assert [name for _, name in rows] == ['P{:02d}'.format(i) for i in range(12, 0, -1)]  # by growth
    assert [bool(hidden) for hidden, _ in rows] == [False] * 8 + [True] * 4
    assert '<td><b>P12</b></td><td>1.000</td><td>1.120</td><td><b>+120</b></td><td>+12%</td>' in provinces
    assert '<button type="button" class="more" id="provinces-more" hidden>Ver las 12 provincias</button>' in p1
    summary = p1[p1.index('class="sum"'):]
    summary = summary[:summary.index('</div>')]
    assert '<b>Sí, y no tiene sentido negarlo.</b>' in summary and '(×1,5: de 1.200 a 1.800)' in summary
    assert 'suma 442.257 electores desde las generales de 2023 (+19%)' in summary
    assert '(Argentina ×1,5)' in summary and 'Cuba' not in summary  # no row for Cuba: its clause goes
    assert 'class="how"' in p1 and 'p &lt; 0,001' in p1
    how = p1[p1.index('class="how"'):]
    assert 'del 1 ene 2025 al 1 dic 2025, antes de la ley, y del 1 ene 2026 al 1 sept 2026, después' in how
    assert 'Los periodos se actualizan con cada cierre; el último publicado es el 1 sept 2026.' in how
    assert ('En la réplica del artículo con su ventana (enero 2022 – diciembre 2022 y enero 2023 – junio 2023), son '
            '1.200 altas antes de la ley y 2.400 después (×2,0).') in how
    assert '×2,0' not in p1[:p1.index('class="how"')]  # the article's pace only in «Cómo leerlo»
    assert 'aria-label="Altas netas: 1.200 antes de la ley y 1.800 después, ×1,5"></div>' in p1
    assert 'medium.com' not in page[:page.index('id="datos"')]  # the article appears only in «Datos y método» (task 17)
    # the default fixture has three provinces: no hidden rows and no button; a checked study says so
    publish_study(fresh_app.fs, meta={'checked_at': '2026-12-03'}, parts={'check': {'verdict': 'acierto'}})
    page = cera_page(api)
    provinces = page[page.index('id="provinces-table"'):]
    assert 'id="provinces-more"' not in page and ' hidden>' not in provinces[:provinces.index('</table>')]
    assert '<span class="pill ok">Comprobada con el escrutinio del 3 dic 2026</span>' in page
    assert 'Predicción registrada' not in page[:page.index('class="study-title"')]  # the head: the check's note has it
    publish_study(fresh_app.fs, meta={'registered_at': None, 'registered': None})
    page = cera_page(api)
    assert '<span class="pill">Borrador · predicción sin registrar</span>' in page
    assert 'Predicción registrada' not in page and 'medium.com' not in page[:page.index('id="datos"')]


def test_cera_question_1_helpers():
    analisis, cera = pages.analisis, pages.cera
    assert analisis._long_date('2026-11-29') == '29 de noviembre de 2026' and analisis._long_date(None) is None
    assert cera._months('2019-03-01', '2022-10-01') == 43 and cera._months(None, '2022-10-01') is None
    assert cera._country_name('ESTADOS UNIDOS DE AMÉRICA') == 'Estados Unidos de América'
    assert cera._country_name('MÉXICO') == 'México' and cera._country_name('Argentina') == 'Argentina'
    assert cera._p_value(0.000415) == '< 0,001' and cera._p_value(0.0421) == '= 0,042'
    assert [cera._pace_tone(x) for x in (5.1, 1.6, 0.7, None)] == ['pos', None, 'neg', None]
    # a country of the INE in capitals: the summary finds it, the table shows it in the case of the prose
    census = {'countries': [{'name': 'CUBA', 'g1': 15000, 'g2': 16463, 'ratio': 1.1}], 'periods': {'article': [
        ['2019-03-01', '2022-10-01'], ['2022-10-01', '2026-05-01']]}}
    q1 = cera._question_1(census, {'p_slope': 0.2})
    assert q1['exile']['name'] == 'Cuba' and q1['latam'] == [] and q1['countries'][0]['name'] == 'Cuba'
    assert q1['months'] == [43, 43] and q1['equal'] and q1['start_label'] == '10-2022'
    assert q1['provinces'] == [] and q1['provinces_hidden'] == 0 and q1['slope_significant'] is False
    assert q1['version'] == 'article' and q1['window'] == 'marzo 2019 – octubre 2022 – mayo 2026'  # no `now`
    assert cera._window([['2022-01-01', '2022-12-01'], ['2023-01-01', '2023-06-01']]) == (
        'enero 2022 – diciembre 2022 y enero 2023 – junio 2023')
    assert cera._window(None) is None and cera._window([['2022-01-01', None], ['2023-01-01', '2023-06-01']]) is None
    assert cera._version({'article': {'a': 1}, 'now': None}) == 'article' and cera._version({'now': {'a': 1}}) == 'now'


def cera_section(page, id):
    """La tarjeta `<section … id="{id}">` de la página del estudio, hasta su cierre."""
    section = page[page.index('id="{}"'.format(id)):]
    return section[:section.index('</section>')]


def cera_table(section, id, tables=1):
    """El bloque con `id` de una tarjeta, hasta el cierre de su tabla (o de la última de sus `tables` tablas)."""
    block = section[section.index('id="{}"'.format(id)):]
    end = -1
    for _ in range(tables):
        end = block.index('</table>', end + 1)
    return block[:end + len('</table>')]


def regional_vote(community, previous, last, d_psoe, d_other_left, d_block):
    """Dos filas `autonomicas` de `vote.processes` (comicio anterior y último) con su margen y cuota de izquierda."""
    base = fixture_data('study-vote')['processes'][1]
    return [base | {'community': community, 'step': 'previous', 'date': previous[0], 'margin': previous[1],
                    'left_share': previous[2]},
            base | {'community': community, 'step': 'last', 'date': last[0], 'margin': last[1], 'left_share': last[2],
                    'd_psoe': d_psoe, 'd_other_left': d_other_left, 'd_block': d_block}]


def test_cera_page_questions_2_and_3(api, fresh_app):
    publish_study(fresh_app.fs)
    page = cera_page(api)
    assert page[:page.index('id="p2"')].endswith('<section class="card qcard q2" ')
    p2 = cera_section(page, 'p2')
    assert ('<div class="qn">2</div><h2 class="qt">¿Sesgo de izquierdas o reparto en provincias «estratégicas»?</h2>'
            in p2)
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', p2) == ['SIN EVIDENCIA']
    for chart in ('origin-chart', 'ecological-chart', 'slope-chart', 'cannibal-chart', 'sensitive-chart'):
        assert re.search(r'<div class="chart" id="{}" role="img" aria-label="[^"]+"></div>'.format(chart), p2), chart
    assert '<p class="ctitle">Las nuevas altas participan la mitad</p>' in p2  # ~5% frente a ~10%
    assert '<p class="ctitle">A más censo latinoamericano, menos voto a la izquierda</p>' in p2
    assert '3 provincias, generales · r = −0,74' in p2
    assert '<p class="ctitle">Las provincias sensibles reciben menos altas que el peso que ya tenían</p>' in p2
    band = p2[p2.index('<div class="band">'):]  # the three regional elections of `vote.processes`, both steps
    band = band[:band.index('</div>')]
    assert '<b>No.</b>' in band and 'cae en las 3 comunidades' in band
    assert 'Castilla y León 6,0 → 1,3; Andalucía 5,0 → 2,5; Aragón 4,0 → 2,0' in band
    madrid = cera_table(p2, 'madrid-table')
    assert '<td><b>Madrid, Venezuela</b></td><td>8,5%</td><td>–</td><td>10%</td>' in madrid  # the `_now` shares
    assert '<td><b>Madrid, conjunto</b></td>' in madrid
    assert 'La adscripción de quien nunca vivió en España' in p2
    assert 'Madrid, con 3 escaños,' in p2 and 'el neto del CERA favorece al PSOE' in p2  # region 28 of last_seat
    assert 'el sitio más caro' not in p2  # Valencia has more seats in the fixture
    summary = p2[p2.index('class="sum"'):]
    summary = summary[:summary.index('</div>')]
    assert '<b>Ni concentración estratégica ni giro a la izquierda.</b>' in summary
    assert 'Las nuevas altas participan al ~5%, frente al ~10% del resto' in summary
    assert 'pasan del 25% del censo al 18% de las altas' in summary  # `sensitive.now`
    assert 'las altas de Argentina, Cuba y Venezuela pasan del 31% al 28%' in summary
    assert 'class="how"' in p2 and 'falacia ecológica' in p2
    how = p2[p2.index('class="how"'):]
    assert 'del 1 ene 2025 con su cuota de las altas hasta el 1 sept 2026' in how
    assert ('En la réplica del artículo con su ventana (enero 2022 – diciembre 2022 y enero 2023 – junio 2023), su '
            'cuota pasa del 24,1% del censo al 17,4% de las altas; con Argentina, Cuba y Venezuela, del 30,0% al '
            '26,7%.') in how
    assert '17,4%' not in p2[:p2.index('class="how"')]
    # question 3
    p3 = cera_section(page, 'p3')
    assert page[:page.index('id="p3"')].endswith('<section class="card qcard q3" ')
    assert '<div class="qn">3</div><h2 class="qt">¿Pueden alterar el reparto de escaños el 29-N?</h2>' in p3
    assert re.findall(r'<span class="verdict">([^<]+)<i></i></span>', p3) == ['NO']
    scenario = p3[p3.index('<fieldset class="seg" id="scenario">'):]
    scenario = scenario[:scenario.index('</fieldset>')]
    assert '<legend>Escenario</legend>' in scenario
    assert re.findall(r'<input type="radio" name="scenario" value="(\w+)"( checked)?>', scenario) == [
        ('convocatoria', ' checked'), ('sin_litigio', '')]
    prediction = cera_table(p3, 'prediction-table', tables=2)
    assert re.findall(r'<table data-scenario="(\w+)"( hidden)?>', prediction) == [('convocatoria', ''),
                                                                                  ('sin_litigio', ' hidden')]
    first = prediction[:prediction.index('data-scenario="sin_litigio"')]
    assert '<td>152</td><td>152</td>' in first and '<td>170</td><td>170</td>' in first
    assert '<td>2.327.388</td><td>2.769.645</td><td><span class="dlt up">▲ 442.257</span></td>' in first
    assert 'Otros' not in prediction  # the fixture has only the two blocks
    assert '<b>Sin las 163.020 inscripciones en litigio</b>' in p3
    assert '8.070 votos menos y, también, cero cambios' in p3
    for chart in ('gap-chart', 'radar-chart'):
        assert re.search(r'<div class="chart" id="{}" role="img" aria-label="[^"]+"></div>'.format(chart), p3), chart
    assert '<h3>¿Qué haría falta para que importara?</h3>' in p3
    grid = cera_table(p3, 'grid-table')
    rows = re.findall(r'<tr>\s*<td><b>([^<]+)</b></td>(.*?)</tr>', grid)
    assert [label for label, _ in rows] == ['9% (observada)', '30%']
    cells = [re.findall(r'<td[^>]*>', row) for _, row in rows]
    assert [len(row) for row in cells] == [2, 2] and all('style="background' in td for row in cells for td in row)
    assert '<td class="cell-shade" style="background-color: rgba(22, 163, 74, 0.55)">+2</td>' in grid  # the max
    assert '>0*</td>' in grid  # the observed cell, with the uniform split
    assert 'Reparto uniforme forzado' in p3
    scenarios = cera_table(p3, 'scenarios-table')
    assert 'real (×1,19)' in scenarios and 'empírico' in scenarios and '90% izq.' in scenarios
    assert len(re.findall(r'<tr>\s*<td><b>', scenarios)) == 2
    placement = cera_table(p3, 'placement-table')
    assert len(re.findall(r'<tr>\s*<td><b>', placement)) == 2
    assert 'Sin colocación dirigida, con el reparto real: 0.' in p3 and 'las 181.032 altas' in p3
    summary = p3[p3.index('class="sum"'):]
    summary = summary[:summary.index('</div>')]
    assert '<b>Con datos reales, cero escaños.</b>' in summary and 'ninguna de las 3 provincias' in summary
    assert 'class="how"' in p3 and '163.020' in p3[p3.index('class="how"'):]
    assert ('El censo de 2023 que se resta (2.327.388) es el cierre mensual del INE del 1 jun 2023, la base del '
            'incremento, porque da el desglose por provincia y país que el modelo necesita; no es el censo CERA '
            'oficial de esas generales en Infoelectoral, 2.328.260 electores, 872 más.') in p3
    assert 'medium.com' not in page[:page.index('id="datos"')]
    # richer parts: the gated sentences follow the data (margins that fall, a null cell, a scenario that changes)
    vote = fixture_data('study-vote')
    processes = (vote['processes'][:1] + regional_vote('Castilla y León', ('2022-02-13', 6.0, 45.0),
                                                       ('2026-03-15', 1.3, 40.0), 2.0, -2.5, -0.5)
                 + regional_vote('Aragón', ('2023-05-28', 4.0, 44.0), ('2026-02-08', 2.0, 42.0), 1.5, -1.9, -0.4))
    sensitivity = fixture_data('study-sensitivity')
    sensitivity['grid']['values'] = [[1, None], [1, 2]]
    prediction = fixture_data('study-prediction')
    prediction['scenarios']['sin_litigio']['changed'] = [{'region_id': 28, 'name': 'Madrid'}]
    params = fixture_data('study-meta')['params'] | {
        'sensitive_magnitude': 5, 'madrid_countries': ['MÉXICO', 'VENEZUELA'], 'split_source': '2023-07-23',
        'placement_countries': ['ARGENTINA', 'MÉXICO', 'CUBA', 'VENEZUELA']}
    publish_study(fresh_app.fs, meta={'params': params}, parts={
        'vote': {'processes': processes}, 'sensitivity': sensitivity, 'prediction': prediction})
    page = cera_page(api)
    p2, p3 = cera_section(page, 'p2'), cera_section(page, 'p3')
    band = p2[p2.index('class="band"'):]
    band = band[:band.index('</div>')]
    assert '¿Y la ventaja de la izquierda fuera no ha crecido en las últimas autonómicas?' in band
    assert '<b>No.</b>' in band and 'cae en las 2 comunidades' in band
    assert 'Castilla y León 6,0 → 1,3' in band and 'absorbe a SUMAR/Podemos' in band
    assert 'el bloque de izquierda apenas se mueve: −0,5 · −0,4' in p2 and 'de 2022 a 2026' in p2
    assert 'la cuota de izquierda del CERA baja en todos los procesos desde 2022' in p2
    assert 'Cuota de las provincias de 5 escaños o menos' in p2 and '3 provincias, generales 2023' in p2
    assert 'Cuota de Madrid en las altas de México y Venezuela' in p2 and 'Madrid, 2 países' in p2
    assert 'las 181.032 altas de Argentina, México, Cuba y Venezuela' in p3
    grid = cera_table(p3, 'grid-table')
    assert '<td class="cell-shade">–</td>' in grid and grid.count('style="background') == 3
    assert '+1*</td>' in grid and 'la celda observada da +1; con el reparto real por provincia, 0' in p3
    assert 'cero cambios' not in p3 and 'el reparto cambia en 1 provincia (Madrid)' in p3


def test_cera_page_question_1_copy_fixes(api, fresh_app):
    """Cuba «apenas se mueve» sólo bajo `PACE_FAST`; la frase de los periodos dice el último cierre publicado."""
    census = fixture_data('study-census')
    cuba = {'country_code': 'CUB', 'name': 'CUBA', 'g1': 100, 'g2': 110, 'ratio': 1.1, 'g1_now': 100, 'g2_now': 110,
            'ratio_now': 1.1}
    publish_study(fresh_app.fs, parts={'census': census | {'countries': census['countries'] + [cuba]}})
    p1 = cera_section(cera_page(api), 'p1')
    assert 'pero Cuba, el país símbolo del exilio, apenas se mueve (×1,1)' in p1
    assert 'Los periodos se actualizan con cada cierre; el último publicado es el 1 sept 2026.' in p1
    countries = census['countries'] + [cuba | {'ratio_now': 2.4}]
    publish_study(fresh_app.fs, parts={'census': census | {'countries': countries}})
    p1 = cera_section(cera_page(api), 'p1')
    assert 'apenas se mueve' not in p1 and 'Cuba ×2,4' in p1


def test_cera_page_falls_back_to_the_article_periods_and_says_so(api, fresh_app):
    """A census part without `now` (an older one): the page shows the article's periods and «Cómo leerlo» says so."""
    census = fixture_data('study-census')
    old = {key: [{col: value for col, value in row.items() if not col.endswith('_now')} for row in census[key]]
           for key in ('provinces', 'countries', 'origins', 'exposure')}
    placement = fixture_data('study-placement')
    publish_study(fresh_app.fs, parts={
        'census': census | old | {'periods': {'article': census['periods']['article']}},
        'placement': placement | {'sensitive': {'article': placement['sensitive']['article']},
                                  'correlations': {'article': placement['correlations']['article']}}})
    body = call(api, '/analisis/voto-cera')[2]
    page = re.sub(r'\s+', ' ', html(body))
    p1, p2 = cera_section(page, 'p1'), cera_section(page, 'p2')
    assert '(×2,0)' in page[page.index('id="resumen"'):page.index('id="p1"')]
    assert 'Los periodos se actualizan' not in p1
    assert ('Este cierre no trae los periodos actualizados: las cifras son las de la réplica del artículo con su '
            'ventana (enero 2022 – diciembre 2022 y enero 2023 – junio 2023).') in p1
    assert 'pasan del 24% del censo al 17% de las altas' in p2
    assert 'Este cierre no trae los periodos actualizados: las cuotas son las de la réplica del artículo' in p2
    assert 'r = 0,80 con el tamaño de la provincia' in cera_section(page, 'debate')
    island = initial(body)
    assert island['census']['periods'] == census['periods']['article']
    assert island['sensitive'] == placement['sensitive']['article']


def cera_rows(table):
    """`(provincia, celdas)` de cada fila de una tabla de la página del estudio, con `hidden` en las ocultas."""
    return re.findall(r'<tr( hidden)?>\s*<td><b>([^<]+)</b></td>(.*?)</tr>', table)


def check_row(region_id, name, gap, voters, decided=False):
    """Una fila de `check.rows` (el último escaño con y sin CERA de una provincia)."""
    return {'region_id': region_id, 'name': name, 'magnitude': 5, 'winner_with': 'PP',
            'winner_without': 'PSOE' if decided else 'PP', 'gap': gap, 'voters_cera': voters,
            'over_gap': voters > gap, 'decided': decided, 'crosses_block': decided}


def test_cera_page_debate_check_and_sources(api, fresh_app):
    publish_study(fresh_app.fs)
    page = cera_page(api)
    debate = cera_section(page, 'debate')
    assert page[:page.index('id="debate"')].endswith('<section class="card" ')
    assert '<h2>Lo que dice el debate, y por qué no se sostiene <span class="pill bad">' in debate
    assert 'Cada cifra mide algo; ninguna mide escaños.' in debate
    # the claim of the 23 seats, with the replication of the report from `prediction.report`
    claim = debate[debate.index('<article class="claim">'):]
    claim = claim[:claim.index('</article>')]
    assert '<h3>«Hasta 23 escaños»</h3>' in claim and 'The Objective, 4-9-2026' in claim
    assert 'la media desde 2019 o la máxima provincial' in claim  # the first general election of the parts
    assert 'con el 0,3% de participación, 12 votos y 1 provincia;' in claim  # n_mean 1,2: one province
    assert 'con el 0,9%, 40 votos y 3 provincias.' in claim
    assert '<dt>Problema 1 · no reparte entre partidos</dt>' in claim
    assert '<dt>Problema 3 · participación de otro censo</dt>' in claim
    # the parts carry no general election before the voto rogado: the qualitative sentence, no figure
    assert 'sobre todo de emigrantes europeos; el escenario «máximo» supone 40 votantes' in claim
    assert ('<div class="verd">Con el criterio del informe, en 2023 había 3 escaños en juego; cambió uno, en Madrid, '
            'y en contra de la hipótesis.</div>') in claim
    # 2023 already passed the test: the over-gap rows of `check.rows`, by gap, with the D'Hondt with and without CERA
    passed = cera_table(debate, 'passed-table')
    assert ('<p class="ctitle">2023 ya pasó la prueba: el voto CERA real superó la brecha en 3 provincias y cambió 1 '
            'escaño</p>') in debate
    rows = cera_rows(passed)
    assert [name for _, name, _ in rows] == ['Girona', 'Cantabria', 'Madrid']  # Valencia stays under its gap
    assert ('<td>PP</td><td>PSOE</td><td>1.339</td><td><b>44.807</b></td>'
            '<td class="pos">Sí, al PP, cruza de bloque</td>') in rows[2][2]
    assert rows[0][2].rstrip().endswith('<td class="mut">No</td>')
    assert 'Esta misma tabla se rellenará con el escrutinio del 29-N.' in debate
    assert debate.count('<article class="claim">') == 4
    for title in ('«16 provincias decisivas»', '«162 pueblos con más electores fuera que dentro»',
                  '«Un censo fabricado para la izquierda»'):
        assert '<h3>{}</h3>'.format(title) in debate, title
    assert 'la cuota de izquierda del CERA baja en todas' in debate  # the three regional elections of the fixture
    assert 'r = 0,82 con el tamaño de la provincia' in debate  # `correlations.now`
    assert 'cómo se adscribe a una provincia a quien nunca vivió en España' in debate
    assert 'class="how"' in debate
    # the check, before the scrutiny: the two elections and the registered row of the 29-N
    check = cera_section(page, 'comprobacion')
    assert ('<h2>Comprobación: ¿decidió el voto CERA algún escaño, y a favor de quién? '
            '<span class="pill warn">Pendiente del escrutinio CERA</span></h2>') in check
    table = cera_table(check, 'check-table')
    rows = cera_rows(table)
    assert [name for _, name, _ in rows] == ['Generales 2019', 'Generales 2023', 'Generales 29 nov 2026']
    assert ('<td>208.431</td><td>0,6%</td><td>3</td><td><b>1</b></td><td>Madrid: PSOE → PP</td><td>−1</td>'
            '<td>−1</td>') in rows[1][2]
    assert 'no dará al PSOE ni a la izquierda un saldo neto de más de 1 escaño' in rows[2][2]
    assert 'registrada el 18 oct 2026' in rows[2][2] and 'borrador' not in table
    assert 'id="check-rows"' not in page and 'id="check-more"' not in page
    note = check[check.index('<p class="note">'):]
    note = note[:note.index('</p>')]
    assert 'Lo que se registra: el saldo neto del PSOE y de la izquierda por el voto CERA no mayor que 1 escaño;' in note
    assert 'Madrid (PSOE frente a PP)' in note and 'se multiplicara por 1,5' in note
    assert 'class="how"' in check
    assert ('Los votantes CERA de la tabla son el «Total votantes CERA» de los resultados por provincia de '
            'Infoelectoral: 208.431 en 2023. La Oficina del Censo Electoral da 233.688 para esas generales, pero '
            'cuenta otra cosa: los sobres de voto recibidos, no los votantes del escrutinio.') in check
    # data and method: the sources, the CSV and, last, the article
    datos = cera_section(page, 'datos')
    assert '<ul class="src-list">' in datos and '<a href="https://infoelectoral.interior.gob.es/">' in datos
    assert datos.count('class="pill brand" href="/api/v1/studies/voto-cera/') == 11
    assert page.count('medium.com') == 1 and page.index('medium.com') > page.index('id="datos"')
    assert datos.rstrip().endswith('El voto CERA en España y el efecto de la «Ley de Nietos»</a>.</p>')
    assert 'se publicó en Medium el 30 jun 2026' in datos
    glossary = page[page.index('<details class="how" id="glosario">'):]
    glossary = glossary[:glossary.index('</details>')]
    assert page.index('id="glosario"') > page.index('id="datos"')
    assert '<summary>Glosario</summary>' in glossary and '<dt>Voto rogado</dt>' in glossary
    assert re.findall(r'<dt>([^<]+)</dt>', glossary) == [
        'CERA', 'Voto rogado', 'Ley de Nietos', 'Censo de la convocatoria', 'Brecha del último escaño', 'Efecto neto',
        'Inferencia ecológica', 'Provincias sensibles', 'Colocación óptima']
    assert not re.search(r'(?<![\d.,])1 (escaños|provincias)\b', page)  # a real singular everywhere
    # a draft: the row of the 29-N says so; with one seat at stake in 2023 and none changed, the verdict says it
    base = fixture_data('study-check')
    one = base['elections'][-1] | {'n_over_gap': 1, 'n_decided': 0, 'decided': []}
    publish_study(fresh_app.fs, meta={'registered_at': None, 'registered': None},
                  parts={'check': {'elections': base['elections'][:-1] + [one]}})
    page = cera_page(api)
    table = cera_table(cera_section(page, 'comprobacion'), 'check-table')
    assert 'borrador: predicción sin registrar' in table and 'registrada el' not in table
    assert 'la media desde 2019 o' in page
    publish_study(fresh_app.fs, parts={'participation': {'generales': []}})
    assert 'la media desde 1986 o' in cera_page(api)  # no general election in the parts: the typed fallback
    publish_study(fresh_app.fs, meta={'registered_at': None, 'registered': None},
                  parts={'check': {'elections': base['elections'][:-1] + [one]}})
    page = cera_page(api)
    assert ('<div class="verd">Con el criterio del informe, en 2023 había 1 escaño en juego; no cambió '
            'ninguno.</div>') in page
    assert not re.search(r'(?<![\d.,])1 (escaños|provincias)\b', page)
    # after the scrutiny: the row of the 29-N filled, the provinces of the 29-N and the verdict from the data
    rows = [check_row(100 + i, 'Provincia {:02d}'.format(i), 1000 + i, 500) for i in range(21)]
    rows[0] = check_row(28, 'Madrid', 900, 40000, decided=True)
    election = base['elections'][-1] | {'date': '2026-11-29', 'voters': 250000, 'rate': 9.0, 'n_over_gap': 1,
                                         'net_psoe': 0, 'net_left': 0}
    publish_study(fresh_app.fs, meta={'checked_at': '2026-12-03'}, parts={'check': {
        'elections': base['elections'] + [election], 'rows': rows, 'verdict': 'acierto'}})
    page = cera_page(api)
    check = cera_section(page, 'comprobacion')
    assert '<span class="pill ok">Escrutinio del 3 dic 2026 cargado</span>' in check
    assert [name for _, name, _ in cera_rows(cera_table(check, 'check-table'))] == [
        'Generales 2019', 'Generales 2023', 'Generales 2026']
    provinces = cera_rows(cera_table(check, 'check-rows'))
    assert len(provinces) == 21 and [hidden for hidden, _, _ in provinces].count(' hidden') == 1
    assert provinces[0][1] == 'Madrid' and '<td class="pos">Sí, cruza de bloque</td>' in provinces[0][2]
    assert '<button type="button" class="more" id="check-more" hidden>Ver más</button>' in check
    note = check[check.index('<p class="note ok">'):]
    note = note[:note.index('</p>')]
    assert 'Predicción registrada el 18 oct 2026' in note and 'acierto' in note
    assert 'Resultado: saldo del PSOE 0, de la izquierda 0' in note
    debate = cera_section(page, 'debate')
    assert 'Con el escrutinio del 29-N: el voto CERA real superó la brecha en 1 provincia y' in debate
    assert 'Esta misma tabla se rellenará' not in debate
    # a failure, the same way
    publish_study(fresh_app.fs, meta={'checked_at': '2026-12-03'}, parts={'check': {
        'elections': base['elections'] + [election | {'net_psoe': 2, 'net_left': 2}], 'rows': rows,
        'verdict': 'fallo'}})
    check = cera_section(cera_page(api), 'comprobacion')
    assert '<p class="note ok">' not in check and 'fallo' in check and 'saldo del PSOE +2' in check


def test_cera_page_names_the_candidatures_by_their_siglas(api, fresh_app):
    """The parts carry the full Infoelectoral names and their siglas: the page and its island print only the siglas."""
    publish_study(fresh_app.fs)
    body = call(api, '/analisis/voto-cera')[2]
    page = re.sub(r'\s+', ' ', html(body))
    for name in ('PARTIDO POPULAR', 'PARTIDO SOCIALISTA OBRERO ESPAÑOL', 'JUNTS PER CATALUNYA'):
        assert name not in page, name
    debate, check = cera_section(page, 'debate'), cera_section(page, 'comprobacion')
    assert '<td>JxCat</td><td>JxCat</td>' in cera_table(debate, 'passed-table')
    assert '<td class="pos">Sí, al PP, cruza de bloque</td>' in debate
    assert '<td>Madrid: PSOE → PP</td>' in cera_table(check, 'check-table')
    assert 'Madrid (PSOE frente a PP)' in check and 'el neto del CERA favorece al PSOE' in cera_section(page, 'p2')
    gap = initial(body)['gap']['convocatoria'][0]
    assert (gap['winner'], gap['runner_up']) == ('PP', 'PSOE')
    # an older part without the siglas: the name it has, as it comes
    prediction = fixture_data('study-prediction')
    prediction['last_seat'] = [{key: value for key, value in row.items() if not key.endswith('_sigla')}
                               for row in prediction['last_seat']]
    publish_study(fresh_app.fs, parts={'prediction': prediction})
    gap = initial(call(api, '/analisis/voto-cera')[2])['gap']['convocatoria'][0]
    assert (gap['winner'], gap['runner_up']) == ('PARTIDO POPULAR', 'PARTIDO SOCIALISTA OBRERO ESPAÑOL')


def test_shade_scale_filter_grows_green_from_lo_to_hi():
    """`shade_scale`: nada sin valor; el verde más claro en `lo`, el más intenso en `hi`, recortado a los extremos."""
    from mtpy.lib.pages.filters import shade_scale
    assert shade_scale(None, 0, 10) == '' and shade_scale(float('nan'), 0, 10) == '' and shade_scale(True, 0, 10) == ''
    assert shade_scale(0, 0, 10) == 'rgba(22, 163, 74, 0.08)'
    assert shade_scale(10, 0, 10) == 'rgba(22, 163, 74, 0.55)'
    assert shade_scale(5, 0, 10) == 'rgba(22, 163, 74, 0.315)'
    assert shade_scale(-3, 0, 10) == shade_scale(0, 0, 10) and shade_scale(40, 0, 10) == shade_scale(10, 0, 10)
    assert shade_scale(0, 0, 0) == shade_scale(0, 0, 10)  # an empty range: the lightest
    assert pages.environment().filters['shade_scale'] is pages.shade_scale


def test_run_less_colours_take_the_first_party_of_each_named_block():
    """Sin run del ámbito: el margen es el de la elección y los bloques del mapa nacional se colorean con su
    primer partido en los resultados; un bloque fuera del mapa (el resto) se queda sin color, como con run."""
    meta = {'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}}
    national = {'bmaps': {'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE'], 'Regionalista': ['PNV']}}}
    blocks_of = {'PP': 'Derecha', 'PSOE': 'Izquierda', 'ADELANTE': 'Regionalista', 'PA': 'Regionalista', 'X': 'Otros',
                 'Y': None}
    out = pages.run_less_colours(meta, national, blocks_of)
    assert out['bmaps']['vs'] == meta['blocks']
    assert out['bmaps']['blocks'] == {'Derecha': ['PP'], 'Izquierda': ['PSOE'], 'Regionalista': ['ADELANTE', 'PA']}
    catalog = pages.Catalog({name: {'fullname': name, 'color': color} for name, color in
                             [('PP', '#1d84ce'), ('PSOE', '#ef1c27'), ('ADELANTE', '#1eca7f'), ('PA', '#00aa00'),
                              ('X', '#123456')]})
    assert pages.block_color('Regionalista', out, catalog) == '#1eca7f'
    assert pages.block_color('Otros', out, catalog) == pages.OTHERS_COLOR
