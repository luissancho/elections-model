"""Tests de lo que la web dice a los rastreadores: `/robots.txt` y `/sitemap.xml`."""
import copy
import os
import re
import xml.etree.ElementTree as ET

from mtpy.core.app import Config
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, houses, houses_without_analysis, html, site_api  # noqa: F401
from tests.test_webapi_unit import ROOT, RUN, clear_queries, fixture_data, use_database, write_study  # noqa: F401

SITE = 'https://demoscopio.es'
NS = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
POLL_PATH = '/es/sondeos/2026-10-05/sigma-dos'  # the only poll of the fixture bundle


def write_polls(fs, changes):
    """Reescribe la parte `polls` del run del fixture con una copia de su único sondeo por cada elemento de
    `changes`, con esos campos cambiados (antes de la primera petición: los ficheros de un run se guardan)."""
    data = fixture_data('polls')
    data['polls'] = [copy.deepcopy(data['polls'][0]) | change for change in changes]
    bundle.BundleWriter(fs).write_json(bundle.path_part('es', RUN, 'polls'), 'polls', data, 'es', run_id=RUN)


def sitemap(api):
    """`{loc: lastmod o None}` del sitemap, en su orden, tras comprobar que responde como XML cacheable."""
    status, headers, body = call(api, '/sitemap.xml')
    assert status == 200 and headers['content-type'] == 'application/xml; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=3600' and headers['etag'].startswith('"')
    assert body.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n')
    root = ET.fromstring(body)
    assert root.tag == '{%s}urlset' % NS['s']
    return {url.findtext('s:loc', namespaces=NS): url.findtext('s:lastmod', namespaces=NS) for url in root}


def test_robots_txt_closes_the_api_and_names_the_sitemap(api, fresh_app):
    """`/robots.txt` es texto plano cacheable: todo abierto menos `/api/`, y el sitemap con el origen público."""
    status, headers, body = call(api, '/robots.txt')
    assert status == 200 and headers['content-type'] == 'text/plain; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=3600' and headers['etag'].startswith('"')
    assert html(body) == 'User-agent: *\nDisallow: /api/\n\nSitemap: https://demoscopio.es/sitemap.xml\n'
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/'}}))
    assert html(call(api, '/robots.txt')[2]).endswith('Sitemap: https://demoscopio.dev/sitemap.xml\n')


def test_robots_route_takes_the_dot_literally(api):
    """El punto de la ruta va escapado (los patrones son expresiones regulares): `/robotsxtxt` no es el fichero."""
    status, headers, _ = call(api, '/robotsxtxt')
    assert status == 404 and headers['content-type'].startswith('text/html')


def test_sitemap_lists_the_published_pages_in_order(houses, fresh_app):
    """El sitemap lleva las URL canónicas de lo publicado, por grupos: la portada y los ámbitos (con la fecha de su
    última actualización), las elecciones pasadas, las encuestadoras, el método, los análisis y los sondeos."""
    write_study(fresh_app.fs)
    urls = sitemap(houses)
    paths = [loc[len(SITE):] for loc in urls]
    assert all(loc.startswith(SITE + '/') for loc in urls) and len(set(paths)) == len(paths)
    assert not any('?' in path or path.startswith('/api/') or path == '/privacidad' for path in paths)
    assert paths[:3] == ['/', '/es', '/elecciones']
    # `write_bundle` dates the manifest when it writes it; the `run_at` of the run is the one of the fixture
    assert urls[SITE + '/'] == webapi.site().manifest_data()['updated_at']
    assert urls[SITE + '/es'] == '2026-10-08T12:00:00Z'
    assert all(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', urls[SITE + path]) for path in ('/', '/es'))
    assert urls[SITE + '/elecciones'] is None and urls[SITE + POLL_PATH] is None
    expected = ['/elecciones', '/elecciones/es/generales/2023-07', '/casas', '/casas/agregador', '/casas/gad3',
                '/metodo', '/metodo/promedio', '/metodo/escanos', '/metodo/rating', '/metodo/backtest', '/analisis',
                '/analisis/voto-cera', POLL_PATH]
    assert [path for path in paths if path in expected] == expected
    assert paths[-1] == POLL_PATH
    profiles = [path for path in paths if path.startswith('/casas/')]
    assert profiles == sorted(profiles) and len(profiles) == 8  # every house of the fake catalogue


def test_every_sitemap_url_is_a_live_page_and_its_own_canonical(houses, fresh_app):
    """Cada URL del sitemap responde 200 y se declara a sí misma como canónica, también con sondeos de una misma
    casa el mismo día: con patrocinadores distintos cada uno tiene su página; repetidos, ninguno (su página es un
    404 y no se lista)."""
    write_study(fresh_app.fs)
    write_polls(fresh_app.fs, [
        {},
        {'date': '2026-10-06', 'sponsor': 'El Mundo'}, {'date': '2026-10-06', 'sponsor': None},
        {'date': '2026-10-07', 'sponsor': 'Okdiario'}, {'date': '2026-10-07', 'sponsor': 'OKDiario'},
    ])
    urls = sitemap(houses)
    polls = [loc[len(SITE):] for loc in urls if '/sondeos/' in loc]
    assert polls == [POLL_PATH, '/es/sondeos/2026-10-06/sigma-dos/el-mundo', '/es/sondeos/2026-10-06/sigma-dos']
    assert call(houses, '/es/sondeos/2026-10-07/sigma-dos/okdiario')[0] == 404  # the page two records share
    for loc in urls:
        status, _, body = call(houses, loc[len(SITE):])
        assert status == 200, loc
        assert '<link rel="canonical" href="{}">'.format(loc) in html(body), loc


def head_of(page):
    """La cabecera (`<head>`) de una página."""
    return page[page.index('<head>'):page.index('</head>')]


def test_every_sitemap_page_declares_its_path_whatever_the_query(houses, fresh_app):
    """Ningún parámetro llega a la canónica ni a `og:url`: ni el ámbito de una página global ni los de las
    campañas y las redes (`utm_*`, `fbclid`, `gclid`). Cada página tiene una sola dirección, la del sitemap."""
    write_study(fresh_app.fs)
    tracking = 'utm_source=x&utm_medium=social&fbclid=abc&gclid=1'

    for loc in list(sitemap(houses)) + [SITE + '/privacidad']:
        # the home sends its old `?scope=` to the event of the scope (a 301): it is asked without it
        query = tracking if loc == SITE + '/' else 'scope=es&' + tracking
        status, _, body = call(houses, loc[len(SITE):], query=query)
        head = head_of(html(body)) if status == 200 else ''
        assert status == 200 and '<link rel="canonical" href="{}">'.format(loc) in head, loc
        assert '<meta property="og:url" content="{}">'.format(loc) in head, loc
        assert 'utm_' not in head and 'fbclid' not in head and 'gclid' not in head, loc


def test_no_query_value_reaches_the_head(houses, fresh_app):
    """Lo que venga en la consulta no se copia en la página, tampoco escapado."""
    for path in ('/', '/es', '/metodo', '/casas', '/elecciones', '/privacidad'):
        status, _, body = call(houses, path, query='zzq=%22%3E%3Cscript%3Ealert(1)%3C/script%3E&zzr=a%26b')
        page = html(body)
        assert status == 200 and '<link rel="canonical" href="{}{}">'.format(SITE, path) in head_of(page), path
        assert 'zzq' not in page and 'zzr' not in page and 'alert(1)' not in page, path


def test_sitemap_without_database_is_a_503_page(api):
    """Sin base de datos no hay perfiles de encuestadoras: 503 sin caché, nunca una lista a medias."""
    status, headers, body = call(api, '/sitemap.xml')
    assert status == 503 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'no-store' and 'etag' not in headers
    assert 'La base de datos no está disponible' in html(body)


def test_sitemap_without_bundle_is_a_503_and_robots_still_answers(fresh_app, tmp_path):
    """Sin nada publicado el sitemap es un 503; `robots.txt`, que no lee nada, responde igual."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    empty = site_api(fresh_app)
    status, headers, body = call(empty, '/sitemap.xml')
    assert status == 503 and headers['cache-control'] == 'no-store'
    assert 'Todavía no hay ninguna actualización del modelo' in html(body)
    assert call(empty, '/robots.txt')[0] == 200


def test_sitemap_skips_the_scopes_without_polls_part(houses, tmp_path):
    """Un ámbito cuya última actualización no tiene la parte `polls` sale sin páginas de sondeos."""
    removed = [path for path in tmp_path.rglob('polls.json') if 'runs' in path.parts]
    assert removed
    for path in removed:
        path.unlink()
    urls = sitemap(houses)
    assert SITE + '/es' in urls and not any('/sondeos/' in loc for loc in urls)


def test_poll_paths_list_only_the_polls_their_page_resolves(houses, monkeypatch):
    """Un sondeo entra cuando su página lo encuentra (`find_poll`: exactamente uno responde a la URL). Dos registros
    con el mismo día, casa y patrocinador no tienen página; tampoco uno sin fecha o sin casa."""
    polls = {'polls': [
        # the same day, house and sponsor twice: their page is a 404
        {'date': '2026-10-05', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'sponsor': 'El Mundo'},
        {'date': '2026-10-05', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'sponsor': 'El Mundo'},
        # one house, one day, with and without a sponsor: a page each
        {'date': '2026-10-06', 'pollster': 'GAD3', 'pollster_id': 2, 'sponsor': 'ABC'},
        {'date': '2026-10-06', 'pollster': 'GAD3', 'pollster_id': 2, 'sponsor': None},
        # a sponsor whose name gives no slug next to a poll without sponsor: the same URL for both
        {'date': '2026-10-07', 'pollster': 'NC Report', 'pollster_id': 3, 'sponsor': '¿?'},
        {'date': '2026-10-07', 'pollster': 'NC Report', 'pollster_id': 3, 'sponsor': None},
        # no valid date, no house
        {'date': None, 'pollster': 'GAD3', 'pollster_id': 2},
        {'date': '2026-10-08', 'pollster': None},
        # a house without slug whose id comes as a float: the page does not find it under its number
        {'date': '2026-10-08', 'pollster': '¿?', 'pollster_id': 9.0, 'sponsor': None},
        {'date': '2026-10-09', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'sponsor': None},
    ]}
    monkeypatch.setattr(pages.crawlers, 'bundle_part', lambda *args: polls)
    assert pages.crawlers.poll_paths() == ['/es/sondeos/2026-10-06/gad3/abc', '/es/sondeos/2026-10-06/gad3',
                                           '/es/sondeos/2026-10-09/sigma-dos']


def test_sitemap_xml_escapes_the_urls(houses, fresh_app):
    """Lo que XML reserva va escapado: un `&` en el origen no rompe el fichero."""
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/a&b'}}))
    use_database(fresh_app)  # the fake database hangs from the configuration just replaced
    text = pages.sitemap_xml()
    assert '<loc>https://demoscopio.dev/a&amp;b/es</loc>' in text and 'a&b' not in text
    ET.fromstring(text)
