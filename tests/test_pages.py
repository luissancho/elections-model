"""Tests de las páginas servidas por los controladores (`Page`, `Index`, `Promedio`) con el arnés ASGI."""
import json
import os
import re

import pytest

from mtpy import mtpy
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, pages, publish, webapi
from tests.test_api_asgi import call
from tests.test_webapi_unit import ROOT, RUN, fixture_data, write_bundle


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


def test_home_is_rendered_html_with_cache_headers(api):
    status, headers, body = call(api, '/')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=60' and headers['etag'].startswith('"')
    assert '<title>Pronóstico electoral · Portada</title>' in page
    assert 'Pronóstico para el' in page and 'aria-current="page"' in page
    assert '<option value="es" selected>' in page and 'name="mode" value="forecast" checked' in page.replace('\n', ' ')
    assert 'id="headline-table"' in page and '<th scope="col">Partido</th>' in page
    assert fixture_data('headline')['forecast']['parties'][0]['name'] in page
    assert '/dist/css/site.css' in page and '/dist/vendor/echarts-5.6.0.min.js' in page and '/dist/js/pages/index.js' in page
    assert 'onclick=' not in page and '<script>' not in page
    assert page.count('<script') == 3  # ECharts (defer), el módulo de la página y el bloque JSON


def test_home_initial_data_has_the_page_parts_with_one_run(api):
    data = initial(call(api, '/')[2])
    assert set(data) == {'state', 'meta', 'headline', 'vote', 'summary', 'runs'}
    assert data['state'] == {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    assert list(data['state']) == ['scope', 'mode', 'run', 'pinned']
    assert data['meta']['run_id'] == RUN and data['headline']['run_id'] == RUN
    assert data['runs']['runs'][0]['run_id'] == RUN


def test_nowcast_and_pinned_run_are_reflected(api):
    status, headers, body = call(api, '/', query='mode=nowcast&run=' + RUN)
    page = html(body)
    assert status == 200 and 'Estimación a' in page
    assert 'name="mode" value="nowcast" checked' in page.replace('\n', ' ')
    assert 'type="hidden" name="run" value="{}"'.format(RUN) in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': True}
    assert 'href="/promedio?scope=es&amp;mode=nowcast&amp;run={}"'.format(RUN) in page


def test_freeze_banner_and_attribution(api, fresh_app):
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda electoral'})
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert 'id="freeze-banner"' in page and 'Veda electoral' in page
    assert 'Sondeos: Wikipedia' in page and 'href="/api/v1/manifest"' in page


@pytest.mark.parametrize('path, query, status, text', [
    ('/', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
    ('/', 'scope=es-xx', 400, 'Ámbito no válido'),
    ('/', 'run=2026-10-08', 400, 'Run no válido'),
    ('/', 'run=20200101-000000', 404, 'Ese run no existe'),
    ('/', 'mode=tomorrow', 400, 'Modo no válido'),
    ('/nope', '', 404, 'Página no encontrada'),
    ('/promedio', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
])
def test_page_errors_are_html_and_not_cached(api, path, query, status, text):
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
    """Con `es` publicado, `?scope=` ausente lleva a `es`; un ámbito publicado adicional se lista en el selector."""
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert '<option value="es" selected>' in page and '<option value="es-md">Madrid</option>' in page


def test_templates_compile_with_strict_undefined():
    env = pages.environment()
    names = sorted(f for f in os.listdir(pages.TEMPLATES_DIR) if f.endswith('.html'))
    assert {'base.html', '_header.html', '_footer.html', 'error.html', 'index.html', 'promedio.html'} <= set(names)
    for name in names:
        env.get_template(name)


def test_filters_follow_the_js_rules():
    assert pages.fmt_pct(32.66) == '32,7 %' and pages.fmt_pct(None) == '–'
    assert pages.fmt_num(1234567) == '1.234.567' and pages.fmt_num(3.14159, 1) == '3,1'
    assert pages.fmt_prob(0.995) == '> 99 %' and pages.fmt_prob(0.003) == '< 1 %' and pages.fmt_prob(0) == '0 %'
    assert pages.fmt_prob(1) == '100 %' and pages.fmt_prob(0.968) == '97 %' and pages.fmt_prob(None) == '–'
    assert pages.fmt_range(27.31, 38.02) == '27,3–38,0' and pages.fmt_range(118.0, 159.5, 0) == '118–160'
    assert pages.fmt_date('2026-10-13') == '13 oct 2026' and pages.fmt_date('2026-09-05') == '5 sept 2026'
    assert pages.fmt_date('garbage') == '–' and pages.fmt_date(None) == '–'
    assert pages.fmt_datetime('2026-10-08T18:11:02Z') == '8 oct 2026, 20:11'
    assert pages.api_url('es', 'polls', run=RUN, fmt='csv') == '/api/v1/forecast/es/polls?run={}&format=csv'.format(RUN)
    assert pages.api_url('es', 'vote', mode='forecast') == '/api/v1/forecast/es/forecast/vote'


def test_promedio_is_rendered_with_the_polls_table_and_csv_link(api):
    status, headers, body = call(api, '/promedio', query='scope=es&mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Pronóstico electoral · Promedio de sondeos</title>' in page
    assert 'href="/promedio?scope=es&amp;mode=forecast" aria-current="page"' in page.replace('\n', ' ')
    assert 'id="series-chart"' in page and 'Sondeos del ciclo' in page
    assert 'href="/api/v1/forecast/es/polls?run={}&amp;format=csv"'.format(RUN) in page
    assert '<th scope="col">Fecha</th>' in page and '/dist/js/pages/promedio.js' in page
    data = initial(body)
    assert set(data) == {'state', 'meta', 'series', 'polls', 'projection', 'vote'}
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
