"""Tests de lo que la web dice a los rastreadores: `/robots.txt` y `/sitemap.xml`."""
import os
import re
import xml.etree.ElementTree as ET

from mtpy.core.app import Config
from mtpy.core.io import FileSystem
from mtpy.lib import pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, houses, houses_without_analysis, html, site_api  # noqa: F401
from tests.test_webapi_unit import ROOT, use_database, write_study

SITE = 'https://demoscopio.es'
NS = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
POLL_PATH = '/es/sondeos/2026-10-05/sigma-dos'  # the only poll of the fixture bundle


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
    manifest = webapi.site().manifest_data()  # `write_bundle` dates the manifest and the run when it writes them
    assert urls[SITE + '/'] == manifest['updated_at'] and urls[SITE + '/es'] == manifest['scopes']['es']['run_at']
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
    """Cada URL del sitemap responde 200 y se declara a sí misma como canónica."""
    write_study(fresh_app.fs)
    for loc in sitemap(houses):
        status, _, body = call(houses, loc[len(SITE):])
        assert status == 200, loc
        assert '<link rel="canonical" href="{}">'.format(loc) in html(body), loc


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
    status, headers, _ = call(empty, '/sitemap.xml')
    assert status == 503 and headers['cache-control'] == 'no-store'
    assert call(empty, '/robots.txt')[0] == 200


def test_sitemap_skips_the_scopes_without_polls_part(houses, tmp_path):
    """Un ámbito cuya última actualización no tiene la parte `polls` sale sin páginas de sondeos."""
    removed = [path for path in tmp_path.rglob('polls.json') if 'runs' in path.parts]
    assert removed
    for path in removed:
        path.unlink()
    urls = sitemap(houses)
    assert SITE + '/es' in urls and not any('/sondeos/' in loc for loc in urls)


def test_poll_paths_are_unique_and_skip_the_polls_without_page(houses, monkeypatch):
    """Dos sondeos con el mismo día, casa y patrocinador son una URL; sin fecha o sin casa, ninguna."""
    poll = {'date': '2026-10-05', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'sponsor': 'El Mundo'}
    polls = {'polls': [poll, dict(poll), {'date': None, 'pollster': 'GAD3'}, {'date': '2026-10-06', 'pollster': None}]}
    monkeypatch.setattr(pages.crawlers, 'bundle_part', lambda *args: polls)
    assert pages.crawlers.poll_paths() == ['/es/sondeos/2026-10-05/sigma-dos/el-mundo']


def test_sitemap_xml_escapes_the_urls(houses, fresh_app):
    """Lo que XML reserva va escapado: un `&` en el origen no rompe el fichero."""
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/a&b'}}))
    use_database(fresh_app)  # the fake database hangs from the configuration just replaced
    text = pages.sitemap_xml()
    assert '<loc>https://demoscopio.dev/a&amp;b/es</loc>' in text and 'a&b' not in text
    ET.fromstring(text)
