"""Tests estáticos del frontend `web/`: literales de la API frente a `ROUTES`, recursos referenciados e imports."""
import os
import re

import pytest

from mtpy.core.api import Router
from mtpy.lib import webapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEB = os.path.join(ROOT, 'web')


def js_files():
    for folder, _, files in os.walk(os.path.join(WEB, 'js')):
        for name in files:
            if name.endswith('.js'):
                yield os.path.join(folder, name)


def html_files():
    return [os.path.join(WEB, f) for f in os.listdir(WEB) if f.endswith('.html')]


def router():
    r = Router()
    webapi.add_routes(r)
    return r


def matches_a_route(path):
    return any(Router.parse_route(route, path, 'GET') is not False for route in router().routes)


def test_every_api_literal_in_the_js_matches_a_route():
    literals = []
    for path in js_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for match in re.finditer(r"""['"`](/api/v1/[^'"`?]*)""", text):
            literal = re.sub(r'\$\{[^}]*\}', 'x', match.group(1)).rstrip('/')
            literals.append((os.path.relpath(path, ROOT), literal))
    assert literals, 'ningún literal /api/v1 en web/js'
    bad = [(f, lit) for f, lit in literals if not matches_a_route(lit)]
    assert bad == []


def test_referenced_assets_exist():
    missing = []
    for path in html_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for ref in re.findall(r'(?:src|href)="(/[^"]*)"', text):
            if ref.startswith('/api/'):
                continue
            target = os.path.join(WEB, ref.lstrip('/'))
            page = ref == '/' and os.path.isfile(os.path.join(WEB, 'index.html'))
            # Página sin extensión (`/promedio` → `web/promedio.html`), como la sirve la ruta del sitio.
            page = page or ('.' not in ref and os.path.isfile(target + '.html'))
            if not page and not os.path.isfile(target):
                missing.append((os.path.basename(path), ref))
        assert 'type="module"' in text and '<script' in text
        assert 'onclick=' not in text and '<script>' not in text  # CSP: sin inline
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
    assert os.path.getsize(os.path.join(WEB, 'vendor', 'echarts-5.6.0.min.js')) > 900_000
    with open(os.path.join(WEB, 'vendor', 'LICENSE-echarts.txt'), encoding='utf-8') as fh:
        assert 'Apache License' in fh.read(400)
