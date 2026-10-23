"""Series, sondeos y proyección de bloque y estado del hash del promedio (`web/dist/js/blocks.js`), comprobados con
`node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BLOCKS_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'blocks.js').as_uri()
SERIES = {'dates': ['2026-01-01', '2026-01-02'], 'parties': ['PP', 'VOX', 'PSOE'],
          'mean': {'PP': [30, 31], 'VOX': [10, None], 'PSOE': [28, 28]},
          'lo': {'PP': [28, 29], 'VOX': [8, None], 'PSOE': [26, 26]},
          'hi': {'PP': [32, 33], 'VOX': [12, None], 'PSOE': [30, 30]}}
BLOCKS = {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}
POLLS = {'parties': ['PP', 'VOX', 'PSOE'],
         'polls': [{'date': '2026-01-01', 'pollster': 'A', 'PP': 30, 'VOX': 10, 'PSOE': 28},
                   {'date': '2026-01-02', 'pollster': 'B', 'PSOE': 27}]}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = ('import * as b from "{}"; const series = {}; const blocks = {}; const polls = {}; '
              'console.log(JSON.stringify({}));').format(BLOCKS_JS, json.dumps(SERIES), json.dumps(BLOCKS),
                                                         json.dumps(POLLS), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_block_series_sums_means_and_adds_half_widths_in_quadrature():
    out = run_js('b.blockSeries(series, blocks)')
    assert out['parties'] == ['Derecha', 'Izquierda'] and out['mean']['Derecha'] == [40, 31]
    assert out['dates'] == SERIES['dates'] and out['mean']['Izquierda'] == [28, 28]
    assert out['lo']['Derecha'] == [40 - (8 ** 0.5), 29] and out['hi']['Derecha'][0] == 40 + (8 ** 0.5)
    assert out['lo']['Izquierda'] == [26, 26] and out['hi']['Izquierda'] == [30, 30]
    assert run_js('b.blockSeries({dates: [], parties: [], mean: {}, lo: {}, hi: {}}, blocks)')['mean']['Derecha'] == []


def test_block_series_without_any_party_of_the_block_is_null():
    out = run_js('b.blockSeries(series, {Otros: ["CUP"], Derecha: ["PP", "CUP"]})')
    assert out['mean']['Otros'] == [None, None] and out['lo']['Otros'] == [None, None]
    assert out['mean']['Derecha'] == [30, 31] and out['lo']['Derecha'] == [28, 29]


def test_block_polls_and_projection():
    out = run_js('b.blockPolls(polls, blocks)')
    assert out['parties'] == ['Derecha', 'Izquierda']
    assert out['polls'][0]['Derecha'] == 40 and 'Derecha' not in out['polls'][1] and out['polls'][1]['Izquierda'] == 27
    assert out['polls'][1]['date'] == '2026-01-02' and out['polls'][1]['pollster'] == 'B'
    assert run_js('b.blockPolls({parties: [], polls: [{date: "d", pollster: "A", pollster_id: 7, PP: 1}]}, blocks)'
                  )['polls'] == [{'date': 'd', 'pollster': 'A', 'pollster_id': 7, 'Derecha': 1}]
    # the poll box of the blocks view needs the poll name and its sample
    assert run_js('b.blockPolls({parties: [], polls: [{date: "d", pollster: "A", name: "A / M", sample_size: null, '
                  'PSOE: 2}]}, blocks)')['polls'] == [{'date': 'd', 'pollster': 'A', 'name': 'A / M',
                                                         'sample_size': None, 'Izquierda': 2}]
    projection = '{dates: ["d"], groups: {parties: {}, blocks: {names: ["Derecha"], mean: {}, lo: {}, hi: {}}}}'
    assert run_js('b.blockProjection({})'.format(projection))['groups']['parties']['names'] == ['Derecha']
    assert run_js('b.blockProjection({})'.format(projection))['dates'] == ['d']
    assert run_js('b.blockProjection({dates: ["d"], groups: {parties: {}}})') is None
    assert run_js('b.blockProjection(null)') is None


def test_block_result_sums_the_parties_of_each_block():
    assert run_js('b.blockResult({date: "2026-01-10", values: {PP: 30, VOX: 10, PSOE: 28}}, blocks)') == {'date': '2026-01-10', 'values': {'Derecha': 40, 'Izquierda': 28}}
    assert run_js('b.blockResult({date: "2026-01-10", values: {PSOE: 28}}, blocks)') == {'date': '2026-01-10', 'values': {'Izquierda': 28}}


def test_series_hash_round_trip_and_fallbacks():
    ctx = '{parties: ["PP", "PSOE"], blocks: ["Derecha"], windows: ["3m", "6m", "1a", "all"]}'
    assert run_js(f'b.parseSeriesHash("#ventana=6m&vista=bloques&sel=Derecha", {ctx})') == {'window': '6m', 'view': 'bloques', 'selected': ['Derecha'], 'layers': 'ambos'}
    assert run_js(f'b.parseSeriesHash("#ventana=9m&vista=bloques&sel=Nadie", {{parties: ["PP"], blocks: [], windows: ["all"]}})') == {'window': 'all', 'view': 'partidos', 'selected': None, 'layers': 'ambos'}
    assert run_js('b.seriesHash({window: "1a", view: "partidos", selected: ["PP", "PSOE"]})') == '#ventana=1a&vista=partidos&sel=PP,PSOE'
    assert run_js(f'b.parseSeriesHash("", {ctx})') == {'window': 'all', 'view': 'partidos', 'selected': None, 'layers': 'ambos'}
    # los nombres se leen en el orden de la lista de la vista; los de la otra vista no valen
    assert run_js(f'b.parseSeriesHash("#sel=PSOE,PP,Derecha", {ctx})')['selected'] == ['PP', 'PSOE']
    # los nombres que no son de la vista no llegan a la leyenda
    assert run_js(f'b.parseSeriesHash("#vista=partidos&sel=Derecha,PP", {ctx})')['selected'] == ['PP']
    assert run_js(f'b.parseSeriesHash("#vista=bloques&sel=Derecha", {ctx})')['selected'] == ['Derecha']
    # un nombre con coma o espacio vuelve igual; una codificación rota se ignora
    names = '["A,B", "C D"]'
    back = run_js(f'b.parseSeriesHash(b.seriesHash({{window: "3m", view: "partidos", selected: {names}}}), '
                  f'{{parties: {names}, blocks: [], windows: ["3m"]}})')
    assert back == {'window': '3m', 'view': 'partidos', 'selected': ['A,B', 'C D'], 'layers': 'ambos'}
    assert run_js(f'b.parseSeriesHash("#sel=%E0%A4%A,PP", {ctx})')['selected'] == ['PP']
    # una selección vacía explícita vuelve vacía; una clave ausente o con nombres desconocidos, `null`
    assert run_js('b.seriesHash({window: "all", view: "partidos", selected: []})') == '#ventana=all&vista=partidos&sel='
    assert run_js(f'b.parseSeriesHash("#ventana=all&vista=partidos&sel=", {ctx})')['selected'] == []
    assert run_js(f'b.parseSeriesHash("#sel=Nadie", {ctx})')['selected'] is None
    # sin selección explícita el hash no lleva `sel`
    assert run_js('b.seriesHash({window: "all", view: "bloques", selected: null})') == '#ventana=all&vista=bloques'


def test_series_hash_carries_the_layers_only_when_one_is_left():
    """«Sondeos / Promedio / Ambos» (Luis, 2026-10-23): las dos capas por defecto, y el hash sólo lleva la capa
    cuando se deja una (`mostrar=sondeos`, `mostrar=promedio`); un valor desconocido vuelve a las dos."""
    ctx = '{parties: ["PP", "PSOE"], blocks: ["Derecha"], windows: ["3m", "all"]}'
    assert run_js(f'b.parseSeriesHash("#ventana=3m&mostrar=sondeos", {ctx})')['layers'] == 'sondeos'
    assert run_js(f'b.parseSeriesHash("#mostrar=promedio", {ctx})')['layers'] == 'promedio'
    assert run_js(f'b.parseSeriesHash("#mostrar=ambos", {ctx})')['layers'] == 'ambos'
    assert run_js(f'b.parseSeriesHash("#mostrar=nada", {ctx})')['layers'] == 'ambos'
    assert run_js(f'b.parseSeriesHash("#ventana=3m", {ctx})')['layers'] == 'ambos'
    assert run_js('b.seriesHash({window: "3m", view: "partidos", selected: null, layers: "promedio"})') == '#ventana=3m&vista=partidos&mostrar=promedio'
    assert run_js('b.seriesHash({window: "3m", view: "partidos", selected: null, layers: "sondeos"})') == '#ventana=3m&vista=partidos&mostrar=sondeos'
    assert run_js('b.seriesHash({window: "3m", view: "partidos", selected: null, layers: "ambos"})') == '#ventana=3m&vista=partidos'
    assert run_js('b.seriesHash({window: "3m", view: "partidos", selected: null})') == '#ventana=3m&vista=partidos'
    back = run_js(f'b.parseSeriesHash(b.seriesHash({{window: "3m", view: "bloques", selected: ["Derecha"], layers: "sondeos"}}), {ctx})')
    assert back == {'window': '3m', 'view': 'bloques', 'selected': ['Derecha'], 'layers': 'sondeos'}
