"""Tests de lo que la web dice a los rastreadores: `/robots.txt` y `/sitemap.xml`."""
from mtpy.core.app import Config
from tests.test_api_asgi import call
from tests.test_pages import api, html  # noqa: F401


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
