"""Tests de las páginas servidas por los controladores (`Page`, `Index`, `Promedio`) con el arnés ASGI."""
import json
import os
import re
from types import SimpleNamespace

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
    ('/escanos', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
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
    assert {'base.html', '_header.html', '_footer.html', 'error.html', 'index.html', 'promedio.html', 'escanos.html'} <= set(names)
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


def test_unexpected_error_is_a_generic_500_page_and_is_logged(api, fresh_app, monkeypatch):
    """Una excepción inesperada da un 500 HTML genérico, sin traza ni detalle, y se registra una vez."""
    def boom(state, active='/'):
        raise RuntimeError('boom')

    errors = []
    monkeypatch.setattr(pages, 'index_context', boom)
    fresh_app.logger = SimpleNamespace(error=errors.append)
    status, headers, body = call(api, '/')
    page = html(body)
    assert status == 500 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'no-store' and 'etag' not in headers
    assert 'Error interno' in page and 'Traceback' not in page and 'boom' not in page
    assert len(errors) == 1 and 'boom' in errors[0] and 'Traceback' in errors[0]


HOSTILE = '</script><script>alert(1)</script>"\'<b>&'


def test_bundle_strings_are_escaped_and_the_json_island_stays_safe(api, fresh_app):
    """Textos hostiles del paquete se escapan en el HTML y en el bloque JSON; un color no válido cae al gris."""
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
    for path in ('/', '/promedio', '/escanos'):
        body = call(api, path)[2]
        page = html(body)
        rest = block.sub('', page)
        assert '<script>alert' not in page and '&lt;script&gt;alert' in rest
        assert '<b>Hostil' not in page and 'url(' not in rest
        data = initial(body)
        assert data['meta']['parties'][0]['fullname'] == HOSTILE
    home = html(call(api, '/')[2])
    assert 'background-color: {}'.format(pages.OTHERS_COLOR) in home
    assert '&lt;b&gt;Hostil&lt;/b&gt;' in html(call(api, '/promedio')[2])


def test_promedio_nowcast_and_pinned_run(api):
    status, headers, body = call(api, '/promedio', query='scope=es&mode=nowcast')
    page = html(body)
    assert status == 200 and 'name="mode" value="nowcast" checked' in page.replace('\n', ' ')
    assert re.search(r'<span class="muted">a ', page)
    status, headers, body = call(api, '/promedio', query='scope=es&run=' + RUN)
    page = html(body)
    assert status == 200 and 'run={}&amp;format=csv'.format(RUN) in page
    assert '<input type="hidden" name="run" value="{}">'.format(RUN) in page


def test_promedio_initial_polls_carry_only_the_chart_columns(api):
    data = initial(call(api, '/promedio')[2])
    parties = set(data['polls']['parties'])
    assert set(data['polls']) == {'parties', 'polls'}
    for poll in data['polls']['polls']:
        assert set(poll) == {'date', 'pollster'} | parties


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


def test_promedio_title_mentions_the_projection_only_in_forecast(api):
    assert 'Promedio de sondeos y proyección' in html(call(api, '/promedio', query='mode=forecast')[2])
    page = html(call(api, '/promedio', query='mode=nowcast')[2])
    assert 'Promedio de sondeos' in page and 'y proyección' not in page


def test_promedio_caption_counts_the_polls(api, fresh_app):
    """The fixture has 1 poll of 120 used; with n_polls == shown polls the caption changes."""
    assert 'Últimos 1 de 120 sondeos' in html(call(api, '/promedio')[2])
    meta = fixture_data('meta') | {'n_polls': 1}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().files.clear()  # the first request cached the run files
    assert 'Todos los sondeos del ciclo (1)' in html(call(api, '/promedio')[2])


def test_escanos_is_rendered_with_tables_links_and_initial(api):
    status, headers, body = call(api, '/escanos', query='scope=es&mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Pronóstico electoral · Escaños</title>' in page
    assert 'href="/escanos?scope=es&amp;mode=forecast" aria-current="page"' in page.replace('\n', ' ')
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


def test_escanos_nowcast_and_pinned_run(api):
    page = html(call(api, '/escanos', query='mode=nowcast&run=' + RUN)[2])
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
    page = html(call(api, '/escanos', query='scope=es&mode=forecast')[2])
    assert 'id="districts"' in page and '<option value="28" selected>Madrid</option>' in page
    assert 'Madrid · 37 escaños' in page  # caption del detalle
    assert 'href="/escanos?scope=es&amp;mode=forecast&amp;region=8"' in page
    assert '<td title="6–8">7</td>' in page and '<td title="14–18">16</td>' in page
    assert '<td>Total</td>' in page and '140' in page
    page = html(call(api, '/escanos', query='scope=es&mode=forecast&region=8')[2])
    assert '<option value="8" selected>Barcelona</option>' in page and 'Barcelona · 32 escaños' in page


def test_escanos_district_form_keeps_the_pinned_run(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/escanos', query='scope=es&run=' + RUN)[2])
    assert page.count('type="hidden" name="run" value="{}"'.format(RUN)) == 2  # cabecera y formulario de circunscripción


@pytest.mark.parametrize('query, status, text', [
    ('region=abc', 400, 'Circunscripción no válida'),
    ('region=99', 404, 'Circunscripción no encontrada'),
])
def test_escanos_region_errors(api, fresh_app, query, status, text):
    write_districts(fresh_app.fs)
    got, headers, body = call(api, '/escanos', query=query)
    assert got == status and headers['cache-control'] == 'no-store' and text in html(body)


def test_escanos_without_districts_hides_the_section(api, fresh_app):
    write_districts(fresh_app.fs, districts={'parties': ['PP'], 'regions': [], 'rows': []},
                    scenario={'simulation': 0, 'parties': ['PP'], 'rows': [{'region_id': 0, 'region': 'es-md', 'seats': [70]}]})
    status, headers, body = call(api, '/escanos')
    assert status == 200 and 'id="districts"' not in html(body)
    assert call(api, '/escanos', query='region=28')[0] == 404


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
    for path in ('/', '/promedio', '/escanos'):
        status, headers, body = call(api, path, query='scope=es-md')
        assert status == 200 and '<option value="es-md" selected>Madrid</option>' in html(body), path
    assert 'name="coalition" value="PP" checked' in html(call(api, '/escanos', query='scope=es-md')[2])


def test_district_table_and_region_rows():
    table = pages.district_table(DISTRICTS, SCENARIO)
    assert table['parties'] == ['PP', 'PSOE'] and [r['name'] for r in table['rows']] == ['Barcelona', 'Madrid']
    assert table['rows'][1]['cells'] == [{'seats': 16, 'range': '14–18'}, {'seats': 9, 'range': '8–11'}]
    assert table['total'] == {'seats': 246, 'cells': [{'seats': 140, 'range': '–'}, {'seats': 106, 'range': '–'}]}
    assert pages.district_table({'parties': [], 'regions': [], 'rows': []}, SCENARIO) is None
    assert [r['name'] for r in pages.region_rows(DISTRICTS, 28)] == ['PP', 'PSOE']
    assert pages.resolve_region(None, DISTRICTS['regions'])['id'] == 28
