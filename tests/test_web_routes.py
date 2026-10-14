"""Tests estáticos del frontend `web/`: literales de la API frente a `ROUTES`, recursos referenciados e imports."""
import os
import re

import pandas as pd
import pytest

from mtpy.core.api import Router
from mtpy.lib import webapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEB = os.path.join(ROOT, 'web')
DIST = os.path.join(WEB, 'dist')
TEMPLATES = os.path.join(WEB, 'templates')


def js_files():
    for folder, _, files in os.walk(os.path.join(DIST, 'js')):
        for name in files:
            if name.endswith('.js'):
                yield os.path.join(folder, name)


def html_files():
    """HTML de las plantillas si existe `web/templates` y, si no (transición), los de `web/`."""
    base = TEMPLATES if os.path.isdir(TEMPLATES) else WEB
    return [os.path.join(base, f) for f in os.listdir(base) if f.endswith('.html')]


def pages_routes():
    """Rutas de las páginas (`mtpy.lib.pages.ROUTES`); vacío mientras el módulo no exista."""
    try:
        from mtpy.lib import pages
    except ImportError:
        return []
    return list(pages.ROUTES)


SCOPE_CODES = set(pd.read_csv(os.path.join(ROOT, 'data', 'es-scopes.csv'))['scode'])


def is_page(ref):
    """Indica si una referencia sin extensión es una página: `/`, `web/<nombre>.html` o una ruta de `pages`."""
    if ref == '/' or os.path.isfile(os.path.join(WEB, ref.lstrip('/') + '.html')):
        return True
    page_router = Router()
    for route in pages_routes():
        if route[1] != 'redirect':  # /promedio, /escanos and /modelo only redirect: they are not pages
            page_router.add_route(*route)
    for route in page_router.routes:
        match = Router.parse_route(route, ref, 'GET')
        if match is False:
            continue
        # `/{scope}` takes any segment: only `x` (the template-variable placeholder) or a scope of the catalogue
        if 'scope' in match and match['scope'] not in SCOPE_CODES | {'x'}:
            continue
        return True
    return False


def router():
    r = Router()
    webapi.add_routes(r)
    return r


def matches_a_route(path):
    return any(Router.parse_route(route, path, 'GET') is not False for route in router().routes)


def test_templates_api_literals_match_a_route_and_the_js_has_none():
    literals = []
    for path in html_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for match in re.finditer(r"""['"`](/api/v1/[^'"`?]*)""", text):
            literal = re.sub(r'\{\{.*?\}\}', 'x', match.group(1)).rstrip('/')
            literals.append((os.path.relpath(path, ROOT), literal))
    assert literals, 'ningún literal /api/v1 en web/templates'
    bad = [(f, lit) for f, lit in literals if not matches_a_route(lit)]
    assert bad == []
    # el JS dibuja desde `initial-data`: no consulta la API
    with_api = []
    for path in js_files():
        with open(path, encoding='utf-8') as fh:
            if '/api/v1' in fh.read():
                with_api.append(os.path.relpath(path, ROOT))
    assert with_api == []


def test_referenced_assets_exist():
    missing = []
    for path in html_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for ref in re.findall(r'(?:src|href|action)="(/[^"]*)"', text):
            ref = re.sub(r'(?<=/)\{\{.*?\}\}', 'x', ref)  # un segmento de la ruta (`/{{ state.scope }}/escanos`)
            ref = re.sub(r'\{\{.*?\}\}', '', ref)  # la query de la plantilla (`{{ query }}`) no es parte de la ruta
            ref = ref.split('#')[0]  # ni el ancla (`#limitaciones`)
            ref = ref.split('&amp;')[0]  # ni los parámetros que se añaden a la query (`&amp;region=`)
            ref = ref.split('?')[0]  # ni una query literal (`/casas?id=`)
            if ref.startswith('/api/'):
                continue
            if ref.startswith('/dist/') or '.' in ref:
                if not os.path.isfile(os.path.join(WEB, ref.lstrip('/'))):
                    missing.append((os.path.basename(path), ref))
            elif not is_page(ref):
                missing.append((os.path.basename(path), ref))
        if ("{% extends 'base.html' %}" in text or os.path.dirname(path) == WEB) and os.path.basename(path) != 'index.html':
            # cada página carga su módulo, salvo el Inicio, que es solo HTML
            assert 'type="module"' in text and '<script' in text
        assert 'onclick=' not in text  # CSP: sin inline
        assert re.findall(r'<script(?![^>]*\bsrc=)(?![^>]*type="application/json")[^>]*>', text) == []
    assert missing == []


def test_css_urls_are_local_and_exist():
    """Las fuentes y demás `url()` de `site.css` se sirven desde `/dist/` (CSP `default-src 'self'`) y existen."""
    with open(os.path.join(DIST, 'css', 'site.css'), encoding='utf-8') as fh:
        css = fh.read()
    urls = re.findall(r"url\(['\"]?([^'\")]+)", css)
    assert urls and all(url.startswith('/dist/') for url in urls), urls
    assert [url for url in urls if not os.path.isfile(os.path.join(WEB, url.lstrip('/')))] == []


def page_target(path):
    """`(controlador, acción)` de la primera ruta de `pages.ROUTES` que casa con `path`, como `Router.handle`."""
    page_router = Router()
    for route in pages_routes():
        page_router.add_route(*route)
    for route in page_router.routes:
        if Router.parse_route(route, path, 'GET') is not False:
            return route['controller'], route['action']
    return None


def test_fixed_page_routes_win_over_the_scope_route():
    """Las rutas fijas van antes que `/{scope}`: `/casas` no es un ámbito ni `/metodo/backtest` un evento."""
    assert page_target('/promedio') == ('redirect', 'promedio')
    assert page_target('/escanos') == ('redirect', 'escanos')
    assert page_target('/modelo') == ('redirect', 'modelo')
    assert page_target('/casas') == ('casas', 'index')
    assert page_target('/elecciones') == ('elecciones', 'index')
    assert page_target('/metodo') == ('metodo', 'index')
    assert page_target('/metodo/backtest') == ('metodo', 'section')
    assert page_target('/es') == ('promedios', 'index')
    assert page_target('/es-md/escanos') == ('escanos', 'index')
    assert page_target('/ES') is None and page_target('/es/') is None and page_target('/es//escanos') is None


def test_js_imports_resolve():
    missing = []
    for path in js_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for ref in re.findall(r"""from\s+['"](\.{1,2}/[^'"]+)['"]""", text):
            if not os.path.isfile(os.path.normpath(os.path.join(os.path.dirname(path), ref))):
                missing.append((os.path.relpath(path, ROOT), ref))
    assert missing == []


def test_vendor_is_pinned_and_licensed():
    assert os.path.getsize(os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')) > 900_000
    with open(os.path.join(DIST, 'vendor', 'LICENSE-echarts.txt'), encoding='utf-8') as fh:
        assert 'Apache License' in fh.read(400)


def test_chart_containers_have_aria_labels():
    for path in html_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for tag in re.findall(r'<div[^>]*class="chart[^"]*"[^>]*>', text):
            assert 'aria-label="' in tag and 'role="img"' in tag, (os.path.basename(path), tag)
