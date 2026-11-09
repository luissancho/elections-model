"""El ejemplo del promedio de `/metodo/promedio` en el navegador (`web/dist/js/weights-example.js`): carga de las partes de
un ámbito, partidos, vista de un partido, textos que reescriben las cifras del servidor y filas de la tabla (se salta sin
`node`)."""
import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODULE = os.path.join(ROOT, 'web', 'dist', 'js', 'weights-example.js')
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures', 'bundle')
NATIONAL = {'code': 'es', 'label': 'Generales', 'name': 'España', 'published': '2026-10-08'}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def fixture(name):
    with open(os.path.join(FIXTURES, name + '.json')) as f:
        return json.load(f)['data']


def run_js(expr, **data):
    """Evalúa `expr` (que puede usar `await`) con el módulo como `w` y cada `data` como constante; devuelve su JSON."""
    consts = ''.join('const {} = {};'.format(name, json.dumps(value)) for name, value in data.items())
    script = 'import * as w from "file://{}"; {} console.log(JSON.stringify(await (async () => ({}))()));'.format(
        MODULE, consts, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def three_parties():
    weights = fixture('weights')
    stats = weights['stats']['PP']
    weights['parties'] = ['A', 'B', 'C']
    weights['stats'] = {'A': {**stats, 'mean': 20.0}, 'B': {**stats, 'mean': 33.1}, 'C': {**stats, 'mean': None}}
    return weights


def test_party_options_and_default_order_by_mean():
    assert run_js('w.partyOptions(weights)', weights=three_parties()) == ['B', 'A', 'C']
    assert run_js('w.defaultParty(weights)', weights=three_parties()) == 'B'
    assert run_js('w.defaultParty({parties: []})') is None


def test_example_view_slices_and_sorts_the_party():
    view = run_js('w.exampleView(weights, meta, "PP")', weights=fixture('weights'), meta=fixture('meta'))
    assert view['name'] == 'PP' and view['color'] == '#1d84ce' and view['date'] == '2026-10-08'
    assert [row['date'] for row in view['rows']] == ['2026-10-05', '2026-10-01', '2026-09-24']   # los más nuevos arriba
    assert view['n_rows'] == 3 and view['stats']['nobs'] == 3
    assert view['line']['dates'][-1] == '2026-10-08' and len(view['kernel']['values']) == 15
    assert view['top'] == {'pollster': 'Sigma Dos', 'sponsor': None, 'date': '2026-10-01', 'weight': 1.138}
    assert run_js('w.exampleView(weights, meta, "ERC")', weights=fixture('weights'), meta=fixture('meta')) is None


def test_texts_match_the_served_page():
    """Las mismas cadenas que `test_promedio_page_shows_the_example_block` lee en la página servida: al volver a las
    generales y al líder, el texto queda como lo pintó el servidor."""
    texts = run_js('w.exampleTexts(w.exampleView(weights, meta, "PP"), scope)', weights=fixture('weights'),
                   meta=fixture('meta'), scope=NATIONAL)
    assert texts['published'] == texts['date'] == '8 oct 2026'
    assert (texts['mean'], texts['lo'], texts['hi']) == ('33,1%', '32,4%', '33,8%')
    assert (texts['nobs'], texts['bandwidth'], texts['neff'], texts['n_rows']) == ('3', '42', '2,6', '3')
    assert texts['subject'] == 'PP' and texts['dates_hidden'] is True
    assert texts['top'] == ' El sondeo que más pesa es el de Sigma Dos del 1 oct, con 1,14.'


def test_texts_explain_a_reading_day_after_the_run():
    texts = run_js('w.exampleTexts(w.exampleView(weights, meta, "PP"), scope)', weights=fixture('weights'),
                   meta=fixture('meta'), scope={**NATIONAL, 'published': '2026-10-01'})
    assert texts['dates_hidden'] is False and texts['published'] == '1 oct 2026'


def test_subject_names_the_scope_outside_the_generals():
    assert run_js('w.subjectOf("PSOE", {code: "es-an", name: "Andalucía"})') == 'PSOE en Andalucía'
    assert run_js('w.subjectOf("PP", {code: "es", name: "España"})') == 'PP'


def test_top_and_texts_without_rows():
    weights = fixture('weights')
    weights['rows'] = [{**row, 'weight': 0} for row in weights['rows'][:2]]
    texts = run_js('(() => { const view = w.exampleView(weights, meta, "PP"); '
                   'return {top: view.top, texts: w.exampleTexts(view, scope)}; })()',
                   weights=weights, meta=fixture('meta'), scope=NATIONAL)
    assert texts['top'] is None and texts['texts']['top'] == '' and texts['texts']['n_rows'] == '2'


def test_table_rows_match_the_old_markup():
    html = run_js('w.tableRowsHtml(w.exampleView(weights, meta, "PP").rows)', weights=fixture('weights'),
                  meta=fixture('meta'))
    assert html.count('<tr>') == 3 and html.index('<td class="f l">5 oct</td>') < html.index('<td class="f l">24 sept</td>')
    assert '<td>32,4%</td>' in html and '<td>32,9%</td>' in html                    # publicada y corregida (GAD3)
    assert '<b>GAD3</b> / ABC' in html and '<b>Sigma Dos</b></td>' in html          # sin patrocinador, sin barra
    assert '>–<' in html                                                              # muestra ausente → guion
    assert 'class="fill w" style="left: 0; width: 17.1%"' in html                    # peso 0,512 sobre 3
    assert html.count('class="cell-bar"') == 3 and '<span>0,51</span>' in html
    assert '<td>0,64</td>' in html and '<td>0,80</td>' in html and '<td>2.000</td>' in html


def test_table_rows_escape_names():
    rows = [{**fixture('weights')['rows'][0], 'pollster': 'A<b>&"', 'sponsor': "O'Neil"}]
    html = run_js('w.tableRowsHtml(rows)', rows=rows)
    assert '<b>A&lt;b&gt;&amp;&quot;</b> / O&#39;Neil' in html and '<b>A<b>' not in html


def fake_fetch(responses):
    """Un `fetch` de prueba en JS: la URL elige `[status, body]` en `responses`."""
    return ('async (url) => {{ const [status, body] = {}[url]; '
            'return {{ok: status === 200, status, json: async () => body}}; }}').format(json.dumps(responses))


def test_load_example_resolves_both_parts():
    scope = {'weights_href': '/w', 'meta_href': '/m'}
    responses = {'/w': [200, {'data': fixture('weights')}], '/m': [200, {'data': fixture('meta')}]}
    got = run_js('w.loadExample(scope, {})'.format(fake_fetch(responses)), scope=scope)
    assert got['weights'] == fixture('weights') and got['meta']['parties'] == fixture('meta')['parties']


def test_load_example_rejects_missing_parts():
    scope = {'weights_href': '/w', 'meta_href': '/m'}
    catch = 'w.loadExample(scope, {}).then(() => "ok", (error) => error.message)'
    missing = {'/w': [404, {}], '/m': [200, {'data': fixture('meta')}]}
    assert 'HTTP 404' in run_js(catch.format(fake_fetch(missing)), scope=scope)
    empty = {'/w': [200, {'data': {**fixture('weights'), 'parties': []}}], '/m': [200, {'data': fixture('meta')}]}
    assert run_js(catch.format(fake_fetch(empty)), scope=scope) != 'ok'
    no_data = {'/w': [200, {'data': fixture('weights')}], '/m': [200, {}]}
    assert run_js(catch.format(fake_fetch(no_data)), scope=scope) != 'ok'
