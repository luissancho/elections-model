"""Tests estáticos del frontend `web/`: literales de la API frente a `ROUTES`, recursos referenciados e imports."""
import os
import re

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


def is_page(ref):
    """Indica si una referencia sin extensión es una página: `/`, `web/<nombre>.html` o una ruta de `pages`."""
    if ref == '/' or os.path.isfile(os.path.join(WEB, ref.lstrip('/') + '.html')):
        return True
    page_router = Router()
    for route in pages_routes():
        page_router.add_route(*route)
    return any(Router.parse_route(route, ref, 'GET') is not False for route in page_router.routes)


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
        for ref in re.findall(r'(?:src|href)="(/[^"]*)"', text):
            ref = re.sub(r'\{\{.*?\}\}', '', ref)  # la query de la plantilla (`{{ query }}`) no es parte de la ruta
            if ref.startswith('/api/'):
                continue
            if ref.startswith('/dist/') or '.' in ref:
                if not os.path.isfile(os.path.join(WEB, ref.lstrip('/'))):
                    missing.append((os.path.basename(path), ref))
            elif not is_page(ref):
                missing.append((os.path.basename(path), ref))
        if "{% extends 'base.html' %}" in text or os.path.dirname(path) == WEB:
            assert 'type="module"' in text and '<script' in text  # cada página carga su módulo
        assert 'onclick=' not in text  # CSP: sin inline
        assert re.findall(r'<script(?![^>]*\bsrc=)(?![^>]*type="application/json")[^>]*>', text) == []
    assert missing == []


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
