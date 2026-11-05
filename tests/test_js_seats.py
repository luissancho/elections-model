"""Paridad entre `web/dist/js/seats.js` y la regla de cuantiles de `Stat` (se salta sin `node`)."""
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from mtpy.core.utils.stat import Stat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEATS_JS = os.path.join(ROOT, 'web', 'dist', 'js', 'seats.js')
DIST = {'n_seats': 350, 'parties': ['PP', 'PSOE', 'VOX'],
        'seats': [[140, 106, 65], [137, 110, 62], [145, 100, 70], [133, 112, 61], [150, 95, 72], [139, 108, 60], [141, 104, 66], [136, 109, 63]]}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as s from "file://{}"; const dist = {}; console.log(JSON.stringify({}));'.format(SEATS_JS, json.dumps(DIST), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_summarize_matches_stat_quantiles():
    values = np.array([row[0] + row[2] for row in DIST['seats']], dtype=float)
    stat = Stat(values, dropna=True)
    got = run_js('s.summarize(s.coalitionSeats(dist, ["PP", "VOX"]), 176)')
    assert got == {'n': 8, 'median': stat.median(), 'lo': stat.quantile(0.025), 'hi': stat.quantile(0.975),
                   'min': 194.0, 'max': 222.0, 'pMajority': 1.0}


def test_column_and_coalition_ignore_unknown_parties():
    assert run_js('s.column(dist, "PSOE")') == [row[1] for row in DIST['seats']]
    assert run_js('s.column(dist, "ERC")') == []
    assert run_js('s.coalitionSeats(dist, ["ERC"])') == []
    assert run_js('s.coalitionSeats(dist, ["PP", "ERC"])') == [row[0] for row in DIST['seats']]
    assert run_js('s.summarize([], 176)') is None


def test_histogram_bins_cover_the_range_without_gaps():
    assert run_js('s.binWidth([0, 0, 1], 40)') == 1 and run_js('s.binWidth([100, 180], 40)') == 3
    assert run_js('s.histogram([3, 4, 4, 9], 2)') == {'starts': [2, 4, 6, 8], 'counts': [1, 2, 0, 1]}
    assert run_js('s.histogram([0, 0, 0], 1)') == {'starts': [0], 'counts': [3]}


def test_coalition_hash_round_trip():
    assert run_js('s.parseCoalition("#coalition=VOX,PP,ERC", dist.parties)') == ['PP', 'VOX']
    assert run_js('s.parseCoalition("#other=1", dist.parties)') is None
    assert run_js('s.parseCoalition("#coalition=", dist.parties)') == []
    assert run_js('s.coalitionHash(["PP", "VOX"])') == '#coalition=PP,VOX' and run_js('s.coalitionHash([])') == ''


def test_coalition_text_gives_the_central_seats_and_the_probability():
    """La franja de la calculadora de mayorías: los partidos, sus escaños en el reparto central (la cifra de la barra
    y de los bloques) y la probabilidad de mayoría absoluta (Luis, 2026-10-21)."""
    text = run_js('s.coalitionText(3, 202, {median: 201, lo: 181, hi: 221, pMajority: 0.992})')
    assert text == '3 partidos · 202 escaños · P(mayoría absoluta): 99,2%'
    assert run_js('s.coalitionText(1, 140, {median: 139, lo: 118, hi: 160, pMajority: 0})') == '1 partido · 140 escaños · P(mayoría absoluta): 0,0%'


def test_coalition_caption_gives_the_simulations_the_central_seats_and_the_range():
    """Las tres líneas junto al histograma, como las de los KDE de «Reparto de escaños»: el reparto central de los
    partidos marcados (no la mediana de su suma) y el millar con punto."""
    caption = run_js('s.coalitionCaption({n: 1000, median: 201, lo: 181, hi: 220.5, pMajority: 0.992}, 202)')
    assert caption == [{'label': 'Simulaciones', 'value': '1.000'}, {'label': 'Reparto central', 'value': '202'},
                       {'label': 'Rango 95%', 'value': '181–221'}]


def test_load_dist():
    """`loadDist` resuelve con la parte (el `data` del sobre de la API) de una respuesta correcta y rechaza la promesa con
    una que no lo es."""
    ok = run_js('await s.loadDist("/x", async (href) => ({ok: true, json: () => ({schema: "dist@1", data: {parties: ["PP"], seats: [[140]], href}})}))')
    assert ok == {'parties': ['PP'], 'seats': [[140]], 'href': '/x'}
    failed = run_js('await s.loadDist("/x", async () => ({ok: false, status: 503})).then(() => "resolved", (error) => error.message)')
    assert failed != 'resolved' and '503' in failed
    # a network failure, or a body without the envelope, rejects too
    assert run_js('await s.loadDist("/x", async () => { throw new Error("offline"); }).then(() => "resolved", () => "rejected")') == 'rejected'
    assert run_js('await s.loadDist("/x", async () => ({ok: true, json: () => ({parties: ["PP"]})})).then(() => "resolved", () => "rejected")') == 'rejected'
    # a part without the columns the calculator reads (`parties` and the simulations, `seats`) rejects as well
    for data in ('{seats: [[140]]}', '{parties: "PP", seats: [[140]]}', '{parties: ["PP"]}', '{parties: ["PP"], seats: null}'):
        assert run_js('await s.loadDist("/x", async () => ({ok: true, json: () => ({data: %s})})).then(() => "resolved", () => "rejected")' % data) == 'rejected', data


WIRE_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'pages', 'seats-wire.js').as_uri()
CATALOG_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'catalog.js').as_uri()
ECHARTS = os.path.join(ROOT, 'web', 'dist', 'vendor', 'echarts-5.6.0.min.js')


def chip_after_load(summary_parties, fetch_body):
    """Texto de la franja de la calculadora y estado de sus píldoras tras la petición de las simulaciones de
    `wireSeatsPage`, con un documento mínimo (sólo la franja y sus píldoras; el resto de la página no está) y `fetch`
    sustituido por uno que responde `fetch_body`."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); '
        'const el = (extra = {{}}) => ({{hidden: false, disabled: false, textContent: "", querySelectorAll: () => [], '
        'addEventListener() {{}}, setAttribute() {{}}, classList: {{add() {{}}, remove() {{}}, toggle() {{}}}}, ...extra}}); '
        'const nodes = {{"coalition-parties": el(), "coalition-result": el()}}; '
        'globalThis.window = {{echarts, location: {{pathname: "/es", search: "", hash: ""}}, history: {{replaceState() {{}}}}, '
        'innerWidth: 1200, addEventListener() {{}}}}; '
        'globalThis.document = {{getElementById: (id) => nodes[id] || null, querySelector: () => null, '
        'querySelectorAll: () => [], addEventListener() {{}}}}; '
        'globalThis.fetch = async () => ({{ok: true, json: () => ({})}}); '
        'console.error = () => {{}}; process.on("unhandledRejection", () => {{}}); '
        'const w = await import("{}"); const {{Catalog}} = await import("{}"); '
        'const meta = {{majority: 176, n_seats: 350, parties: [{{name: "PP", color: "#1e5aa8"}}], bmaps: {{}}}}; '
        'w.wireSeatsPage({{state: {{scope: "es", mode: "forecast"}}, meta, summary: {{parties: {}}}, dist_href: "/x"}}, new Catalog(meta)); '
        'await new Promise((resolve) => setTimeout(resolve, 50)); '
        'const chip = nodes["coalition-result"]; '
        'console.log(JSON.stringify({{text: chip.textContent, disabled: nodes["coalition-parties"].disabled}})); process.exit(0);'
    ).format(json.dumps(ECHARTS), json.dumps(fetch_body), WIRE_JS, CATALOG_JS, json.dumps(summary_parties))
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def test_calculator_says_the_simulations_failed_when_the_success_path_throws():
    """Una excepción al preparar la calculadora con las simulaciones ya recibidas no deja la franja en «Cargando las
    simulaciones…» con las píldoras desactivadas: dice que no se han podido cargar, como un fallo de la petición."""
    body = {'data': {'parties': ['PP'], 'seats': [[140], [150]]}}
    # `summary.parties` that is not a list makes the preparation of the calculator throw
    assert chip_after_load({'PP': 140}, body) == {'text': 'No se han podido cargar las simulaciones.', 'disabled': True}
    # a part without the simulations is a failed load too
    assert chip_after_load([], {'data': {'parties': ['PP']}}) == {'text': 'No se han podido cargar las simulaciones.', 'disabled': True}
