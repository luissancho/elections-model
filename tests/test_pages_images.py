"""Tests de la imagen propia del enlace de un sondeo: la marca con la que dibuja el servidor, lo que la imagen dice
(como datos), el dibujo y su dirección."""
import copy
import inspect
import os
import re

import pytest
from PIL import Image

from mtpy.core.api import HttpError
from mtpy.lib import pages
from mtpy.lib.pages import sondeos
from tests.test_pages import api, houses, houses_without_analysis, html, site_api  # noqa: F401
from tests.test_webapi_unit import ROOT

IMG_DIR = os.path.join(ROOT, 'web', 'dist', 'img')
INK = (26, 28, 34)  # --color-text
ACCENT = (29, 79, 134)  # --color-accent
# A poll with thirteen parties, in the order of the columns of the polls table (GAD3 for Vocento, 2026-11-05)
PARTIES = ['PP', 'PSOE', 'VOX', 'FA', 'UP', 'SALF', 'ERC', 'JxCat', 'EHB', 'PNV', 'BNG', 'CC', 'AA']
COLORS = {name: '#{:02x}80{:02x}'.format(16 * index, 255 - 16 * index) for index, name in enumerate(PARTIES)}
GAD3 = {'date': '2026-11-05', 'pollster_id': 2, 'pollster': 'GAD3', 'sponsor': 'Vocento', 'start_date': '2026-11-03',
        'end_date': '2026-11-05', 'sample_size': 1502.0, 'rating': 71.74, 'weight': 2.157, 'url': None,
        'PP': 31.9, 'PSOE': 25.9, 'VOX': 18.6, 'FA': 7.5, 'UP': 1.4, 'SALF': 1.3, 'ERC': 2.7, 'JxCat': 1.4, 'EHB': 1.5,
        'PNV': 0.9, 'BNG': 0.9, 'CC': 0.4, 'AA': 1.6,
        'seats': {'PP': [136, 136], 'PSOE': [103, 103], 'VOX': [63, 63], 'FA': [12, 12], 'UP': [1, 1], 'SALF': [0, 0],
                  'ERC': [12, 12], 'JxCat': [5, 5], 'EHB': [8, 8], 'PNV': [4, 4], 'BNG': [2, 2], 'CC': [1, 1],
                  'AA': [3, 3]}}
ELECTION = 'Elecciones generales 2026'
HOST = 'demoscopio.es'


def near(pixel, colour, tolerance=2):
    """Si un píxel tiene ese color, con el margen del redondeo al reducir la imagen."""
    return all(abs(a - b) <= tolerance for a, b in zip(pixel[:3], colour))


def test_wordmark_is_a_transparent_image_of_the_logo():
    """La marca en PNG es el logotipo de `logo.svg` sobre fondo transparente: 800 px de ancho, la proporción de su
    viewBox y sólo sus dos colores (la tinta de las letras y el azul de la lente)."""
    with open(os.path.join(IMG_DIR, 'logo.svg'), encoding='utf-8') as file:
        box = [float(number) for number in re.search(r'viewBox="([^"]*)"', file.read()).group(1).split()]

    image = Image.open(os.path.join(IMG_DIR, 'wordmark.png'))
    width, height = image.size
    assert image.mode == 'RGBA' and width == 800
    assert abs(height - 800 * box[3] / box[2]) <= 1
    corners = ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
    assert all(image.getpixel(corner)[3] == 0 for corner in corners)
    opaque = {pixel[:3] for pixel in image.getdata() if pixel[3] == 255}
    assert any(near(pixel, INK) for pixel in opaque) and any(near(pixel, ACCENT) for pixel in opaque)
    assert all(near(pixel, INK) or near(pixel, ACCENT) for pixel in opaque)
    # the letters at the left and the lens at the right end, as in the logo
    assert near(max(opaque, key=lambda pixel: pixel[2]), ACCENT)
    left = {pixel[:3] for pixel in image.crop((0, 0, width // 2, height)).getdata() if pixel[3] == 255}
    assert left and all(near(pixel, INK) for pixel in left)


def poll_of(*drop, **change):
    """Una copia del sondeo de GAD3 sin los campos de `drop` y con los de `change`."""
    poll = copy.deepcopy(GAD3) | change
    return {key: value for key, value in poll.items() if key not in drop}


def preview(poll=None, previous=None, parties=None):
    """Lo que dice la imagen de un sondeo (el de GAD3 por defecto), con el catálogo de la prueba."""
    return sondeos.preview_of(GAD3 if poll is None else poll, previous, PARTIES if parties is None else parties,
                              COLORS, ELECTION, HOST)


def test_fieldwork_compact():
    """Las fechas de campo van compactas y con el año: sólo se repite lo que cambia entre el primer día y el
    último."""
    cases = [(('2026-11-03', '2026-11-05'), '3–5 nov 2026'), (('2026-10-29', '2026-11-02'), '29 oct–2 nov 2026'),
             (('2025-12-28', '2026-01-02'), '28 dic 2025–2 ene 2026'), (('2026-11-05', '2026-11-05'), '5 nov 2026'),
             ((None, '2026-11-05'), '5 nov 2026'), (('2026-11-03', None), '3 nov 2026'),
             (('2026-09-01', '2026-09-03'), '1–3 sept 2026'), ((None, None), None), (('ayer', ''), None)]

    for (start, end), text in cases:
        assert sondeos.fieldwork_compact({'start_date': start, 'end_date': end}) == text, (start, end)

    assert sondeos.fieldwork_compact({}) is None


def test_preview_shows_the_six_most_voted_by_vote():
    """Entran los seis partidos con más voto del sondeo, de mayor a menor (el orden de su página, no el de las
    columnas de la tabla), y el pie cuenta los que se quedan fuera."""
    data = preview()
    assert [row['name'] for row in data['rows']] == ['PP', 'PSOE', 'VOX', 'FA', 'ERC', 'AA']
    assert data['more'] == 'y 7 partidos más en demoscopio.es'
    assert preview(parties=PARTIES[:7])['more'] == 'y 1 partido más en demoscopio.es'
    six = preview(parties=PARTIES[:6])
    assert six['more'] is None and [row['name'] for row in six['rows']] == ['PP', 'PSOE', 'VOX', 'FA', 'UP', 'SALF']
    three = preview(parties=['VOX', 'PSOE', 'PP'])
    assert three['more'] is None and [row['name'] for row in three['rows']] == ['PP', 'PSOE', 'VOX']
    # a party of the catalogue the poll has no figure for is not a party of the poll
    assert preview(poll_of('AA', 'CC'))['more'] == 'y 5 partidos más en demoscopio.es'


def test_preview_rows_carry_share_change_and_seats():
    """Cada fila lleva lo que se pinta: las siglas, el color, el voto con el formato del sitio, el cambio frente a
    la entrega anterior y los escaños de la casa."""
    previous = poll_of('FA', date='2026-10-20', PP=31.5, PSOE=26.4)
    rows = preview(previous=previous)['rows']
    assert rows[0] == {'name': 'PP', 'color': COLORS['PP'], 'value': 31.9, 'share': '31,9%',
                       'change': {'direction': 'up', 'text': '0,4'}, 'seats': '136'}
    assert rows[1]['change'] == {'direction': 'down', 'text': '0,5'}
    assert rows[2]['change'] == {'direction': 'same', 'text': '='}
    assert rows[3]['name'] == 'FA' and rows[3]['change'] is None  # the previous delivery did not have it
    assert all(row['change'] is None for row in preview()['rows'])  # no previous delivery

    ranges = preview(poll_of(seats={'PP': [125, 136], 'PSOE': [100, 100], 'UP': [1, 2]}))
    assert [row['seats'] for row in ranges['rows']] == ['125–136', '100', None, None, None, None]
    assert ranges['has_seats'] is True and preview()['has_seats'] is True
    assert preview(poll_of('seats'))['has_seats'] is False
    assert all(row['seats'] is None for row in preview(poll_of('seats'))['rows'])
    # seats only for a party the image does not show: no column
    assert preview(poll_of(seats={'UP': [1, 2]}))['has_seats'] is False


def test_preview_texts():
    """El título, la línea gris y el texto alternativo dicen la casa y el medio, la elección, el campo y la muestra;
    lo que el sondeo no trae se omite."""
    data = preview()
    assert data['title'] == 'GAD3 para Vocento' and preview(poll_of('sponsor'))['title'] == 'GAD3'
    assert data['subtitle'] == 'Elecciones generales 2026 · Campo: 3–5 nov 2026 · 1.502 entrevistas'
    assert preview(poll_of('sample_size'))['subtitle'] == 'Elecciones generales 2026 · Campo: 3–5 nov 2026'
    assert preview(poll_of(sample_size=0))['subtitle'] == 'Elecciones generales 2026 · Campo: 3–5 nov 2026'
    no_field = preview(poll_of('start_date', 'end_date'))
    assert no_field['subtitle'] == 'Elecciones generales 2026 · 5 nov 2026 · 1.502 entrevistas'
    assert data['alt'] == ('Sondeo de GAD3 para Vocento (campo 3–5 nov 2026): PP 31,9%, PSOE 25,9%, VOX 18,6%, '
                           'FA 7,5%, ERC 2,7%, AA 1,6%.')
    assert no_field['alt'].startswith('Sondeo de GAD3 para Vocento (5 nov 2026): PP 31,9%, ')
    empty = preview(parties=[])
    assert empty['rows'] == [] and empty['alt'] == 'Sondeo de GAD3 para Vocento (campo 3–5 nov 2026).'
    assert empty['more'] is None and empty['has_seats'] is False


def test_preview_version_follows_what_is_drawn(monkeypatch):
    """La huella de la dirección cambia con lo que la imagen pinta (una cifra, el cambio frente a la entrega
    anterior, un color, el diseño) y con nada más."""
    version = preview()['version']
    assert re.fullmatch(r'[0-9a-f]{8}', version) and preview(poll_of())['version'] == version
    assert preview(poll_of(PP=32.0))['version'] != version
    assert preview(previous=poll_of(date='2026-10-20', PP=31.5))['version'] != version
    assert preview(poll_of(seats={'PP': [125, 136]}))['version'] != version
    assert preview(poll_of(rating=10.0, weight=0.5, url='https://example.org/'))['version'] == version
    monkeypatch.setitem(COLORS, 'PP', '#000000')
    assert preview()['version'] != version
    monkeypatch.undo()
    monkeypatch.setattr(sondeos, 'PREVIEW_DESIGN', sondeos.PREVIEW_DESIGN + 1)
    assert preview()['version'] != version
    # what the image says is composed as data: the module does not draw
    assert 'PIL' not in inspect.getsource(sondeos)


def test_poll_preview_reads_the_published_poll(houses, fresh_app):
    """Los datos de la imagen salen del sondeo publicado que nombra la dirección, por el nombre de la casa o por su
    identificador, y son los que lleva el contexto de su página."""
    state = pages.parse_state({}, 'es')
    data = pages.poll_preview(state, '2026-10-05', 'sigma-dos')
    assert data['title'] == 'Sigma Dos' and data['subtitle'] == 'Elecciones generales 2026 · 5 oct 2026'
    assert [(row['name'], row['share'], row['change'], row['seats']) for row in data['rows']] == [
        ('PP', '33,1%', None, None)]
    assert data['more'] is None and data['has_seats'] is False
    assert data['alt'] == 'Sondeo de Sigma Dos (5 oct 2026): PP 33,1%.'
    assert pages.poll_preview(state, '2026-10-05', '7') == data
    assert pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos') == data
    assert pages.sondeo_context(state, '2026-10-05', 'sigma-dos')['preview'] == data

    with pytest.raises(HttpError) as error:
        pages.poll_preview(state, '2026-10-05', 'nadie')

    assert error.value.status == 404 and error.value.message == 'poll not found'
    assert pages.latest_poll_preview('es', '2026-10-05', 'nadie') is None
    assert pages.latest_poll_preview('es', '2026-10-06', 'sigma-dos') is None
