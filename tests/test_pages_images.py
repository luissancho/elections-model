"""Tests de la imagen propia del enlace de un sondeo: la marca con la que dibuja el servidor, lo que la imagen dice
(como datos), el dibujo y su dirección."""
import copy
import hashlib
import html as html_escapes
import inspect
import io
import os
import re

import pytest
from PIL import Image, ImageChops, ImageColor, ImageFont

from mtpy.core.api import HttpError
from mtpy.lib import pages
from mtpy.lib.pages import sondeos
from tests.test_api_asgi import call
from tests.test_pages import api, houses, houses_without_analysis, html, site_api  # noqa: F401
from tests.test_pages_crawlers import POLL_PATH, head_of, sitemap, write_polls
from tests.test_pages_titles import site_paths
from tests.test_webapi_unit import ROOT, RUN, fixture_data, write_study

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
# Every character of the texts of an image: figures, signs and the letters of the names in the languages of Spain
CHARS = '0123456789%,.:=–·…-+()/&ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzáéíóúüçàèòÁÉÍÓÚ'
UP = (22, 163, 74)  # --color-ok
DOWN = (220, 38, 38)  # --color-bad
SITE = 'https://demoscopio.es'
IMAGE_PATH = POLL_PATH + '.png'  # the image of the only poll of the fixture bundle
BRAND_ALT = 'Demoscopio: promedio de encuestas, previsión de escaños y ranking de encuestadoras'
# Shrinking the drawing leaves a faint halo of 3 px around any ink: an empty area is checked from further
HALO = 4


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
    # each text of the picture moves it on its own: the title, the grey line, the footnote and the name of a row
    monkeypatch.setitem(COLORS, 'PPX', COLORS['PP'])
    renamed = poll_of('PP', PPX=GAD3['PP'], seats=GAD3['seats'] | {'PPX': GAD3['seats']['PP']})
    texts = ('title', 'subtitle', 'more', 'rows')
    others = [('title', preview(poll_of(sponsor='Colpisa'))),
              ('subtitle', preview(poll_of(sample_size=1200.0))),
              ('subtitle', sondeos.preview_of(GAD3, None, PARTIES, COLORS, 'Elecciones generales 2027', HOST)),
              ('more', preview(poll_of('CC'))),
              ('more', sondeos.preview_of(GAD3, None, PARTIES, COLORS, ELECTION, 'example.org')),
              ('rows', preview(renamed, parties=['PPX'] + PARTIES[1:]))]

    for text, other in others:
        assert [key for key in texts if other[key] != preview()[key]] == [text]
        assert other['version'] != version, text

    assert [row['name'] for row in others[-1][1]['rows']][:2] == ['PPX', 'PSOE']
    assert [dict(row, name='PP') for row in others[-1][1]['rows']][0] == preview()['rows'][0]  # only the name
    monkeypatch.setitem(COLORS, 'PP', '#000000')
    assert preview()['version'] != version
    monkeypatch.undo()
    monkeypatch.setattr(pages.previews, 'PREVIEW_DESIGN', pages.previews.PREVIEW_DESIGN + 1)
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


def row_of(name, value, color='#3366aa', change=None, seats=None):
    """Una fila de la imagen, como las de `preview_of`."""
    return {'name': name, 'color': color, 'value': float(value), 'share': pages.fmt_pct(value), 'change': change,
            'seats': seats}


def card(rows, title='GAD3 para Vocento', more=None,
         subtitle='Elecciones generales 2026 · Campo: 3–5 nov 2026 · 1.502 entrevistas'):
    """Los datos de una imagen con esas filas."""
    return {'title': title, 'subtitle': subtitle, 'rows': rows, 'more': more, 'alt': '', 'version': '',
            'has_seats': any(row['seats'] is not None for row in rows)}


def drawn(data):
    """La imagen que dibuja `poll_png`, abierta."""
    return Image.open(io.BytesIO(pages.images.poll_png(data)))


def blank(image, box):
    """Si un recuadro `(izquierda, arriba, derecha, abajo)` de la imagen es todo fondo."""
    area = image.convert('RGB').crop(box)
    return ImageChops.difference(area, Image.new('RGB', area.size, pages.images.BACKGROUND)).getbbox() is None


def count_near(image, box, colour, tolerance=40):
    """Cuántos píxeles de un recuadro tienen más o menos ese color."""
    return sum(near(pixel, colour, tolerance) for pixel in image.convert('RGB').crop(box).getdata())


def site_font(file, size, axes):
    """Una fuente del sitio abierta a mano, con sus ejes fijados, al tamaño al que se dibuja."""
    face = ImageFont.truetype(os.path.join(ROOT, 'web', 'dist', 'vendor', 'fonts', file), size * pages.images.SCALE,
                              layout_engine=ImageFont.Layout.BASIC)
    face.set_variation_by_axes(axes)
    return face


def row_band(index):
    """`(arriba, abajo)` de la franja de la barra de una fila."""
    top = pages.images.ROW_TOP + index * pages.images.ROW_STEP

    return top + 10, top + 10 + pages.images.BAR_HEIGHT


def test_fonts_are_the_site_fonts_at_their_weights():
    """El servidor dibuja con las fuentes que sirve el sitio (las woff2 variables), cada una con su peso, y tienen
    todas las letras y signos de los textos."""
    images = pages.images
    title, bold, body = images.font('title', 40), images.font('bold', 40), images.font('body', 24)
    assert title.getlength('Demoscopio 31,9%') > bold.getlength('Demoscopio 31,9%')  # the weight is applied
    assert images.font('title', 40) is title  # opened once
    assert title.size == 40 * images.SCALE and body.size == 24 * images.SCALE

    for face in (title, bold, body):
        missing = bytes(face.getmask('\U0010fffd'))
        assert [char for char in CHARS if bytes(face.getmask(char)) == missing] == []

    # each is the file the site serves with its axes pinned: the weight, and the optical size of the text font,
    # which follows the size of the text (DM Sans has the axes ``opsz`` and ``wght``, in that order)
    line = 'Elecciones generales 2026 · Campo: 3–5 nov 2026'
    assert title.getlength(line) == site_font('Outfit.woff2', 40, [800]).getlength(line)
    assert bold.getlength(line) == site_font('Outfit.woff2', 40, [700]).getlength(line)
    assert body.getlength(line) == site_font('DMSans.woff2', 24, [24, 500]).getlength(line)
    assert body.getlength(line) != site_font('DMSans.woff2', 24, [9, 500]).getlength(line)
    assert body.getlength(line) != site_font('DMSans.woff2', 24, [24, 700]).getlength(line)
    assert images.font('body', 60).getlength(line) == site_font('DMSans.woff2', 60, [40, 500]).getlength(line)
    assert abs(images.text_width('title', 'GAD3 para Vocento', 50) - 472) <= 2
    assert images.text_width('body', '', 20) == 0


def test_fit_text_shrinks_then_cuts():
    """Un texto que no cabe se encoge, de píxel en píxel, hasta su mínimo; si ni así cabe, se corta con «…»."""
    fit, width = pages.images.fit_text, pages.images.text_width
    assert fit('title', 'GAD3 para Vocento', 50, 34, 830) == (50, 'GAD3 para Vocento')
    size, text = fit('title', 'Ateneo del Dato para elDiario.es', 50, 34, 700)
    assert text == 'Ateneo del Dato para elDiario.es' and 34 <= size < 50
    assert width('title', text, size) <= 700 < width('title', text, size + 1)

    long = 'Instituto de Estudios Sociales Avanzados de Andalucía para la Radio y Televisión de Andalucía'
    size, text = fit('title', long, 50, 34, 830)
    assert size == 34 and text.endswith('…') and long.startswith(text[:-1]) and not text[:-1].endswith(' ')
    assert 700 < width('title', text, 34) <= 830

    column = pages.images.BAR_LEFT - pages.images.MARGIN - 12
    size, text = fit('bold', 'PODEMOS-IU-AV', 34, 22, column)
    assert size == 22 and text.endswith('…') and width('bold', text, size) <= column
    assert fit('title', 'Demoscopio', 50, 34, 1) == (34, '…')  # no room for a letter
    # a cut that falls after a space drops the space
    assert fit('title', 'Sigma Dos para El Mundo', 34, 34, width('title', 'Sigma …', 34)) == (34, 'Sigma…')
    assert fit('title', '', 50, 34, 100) == (50, '')


def test_name_column_grows_before_the_names_shrink():
    """Las siglas van todas al mismo tamaño: si alguna no cabe en su columna, la columna crece (y las barras
    empiezan más a la derecha) hasta un tope; pasado el tope, se encogen todas."""
    images = pages.images
    width = images.text_width
    assert images.name_column(['PP', 'PSOE', 'VOX', 'JxCat']) == (34, images.BAR_LEFT)
    assert images.name_column([]) == (34, images.BAR_LEFT)
    size, left = images.name_column(['PP', 'COMPROMIS'])
    assert size == 34 and images.BAR_LEFT < left <= images.BAR_LEFT_MAX
    assert left - images.MARGIN - width('bold', 'COMPROMIS', 34) >= 12
    size, left = images.name_column(['PP', 'W' * 9])
    assert left == images.BAR_LEFT_MAX and 22 <= size < 34
    assert left - images.MARGIN - width('bold', 'W' * 9, size) >= 12
    assert images.name_column(['W' * 30]) == (22, images.BAR_LEFT_MAX)
    # a name too long even then is cut before the bars
    image = drawn(card([row_of('W' * 30, 40), row_of('PP', 30)]))
    top, bottom = row_band(0)
    assert blank(image, (images.BAR_LEFT_MAX - 10, top, images.BAR_LEFT_MAX - HALO, bottom))
    assert not blank(image, (images.MARGIN, top, images.BAR_LEFT_MAX - 40, bottom))


def test_poll_png_is_a_1200x675_png_and_repeats():
    """La imagen es un PNG de 1200×675; los mismos datos dan los mismos bytes y otra cifra, otros."""
    data = sondeos.preview_of(GAD3, None, PARTIES, COLORS, ELECTION, HOST)
    png = pages.images.poll_png(data)
    image = Image.open(io.BytesIO(png))
    assert png.startswith(b'\x89PNG\r\n\x1a\n') and image.format == 'PNG'
    assert image.size == (1200, 675) == pages.images.SIZE and image.mode == 'RGB'
    assert pages.images.poll_png(copy.deepcopy(data)) == png
    assert pages.images.poll_png(sondeos.preview_of(poll_of(PP=32.0), None, PARTIES, COLORS, ELECTION, HOST)) != png
    assert len(png) < 500_000


def test_poll_png_draws_the_bars_in_the_party_colours():
    """La barra de cada partido es de su color y proporcional a su voto (la mayor ocupa el ancho máximo, y la de
    un cero no desaparece); un partido sin color válido la lleva gris."""
    images = pages.images
    colours = ['#1d84ce', '#ef1c27', '#63be21', None, 'nope', '#11223344']
    image = drawn(card([row_of('PP', 40, colours[0]), row_of('PSOE', 4, colours[1]), row_of('VOX', 0, colours[2]),
                        row_of('FA', 0, colours[3]), row_of('UP', 0, colours[4]), row_of('AA', 0, colours[5])]))
    left, end = images.BAR_LEFT, images.BAR_LEFT + images.BAR_MAX
    middle = [sum(row_band(index)) // 2 for index in range(6)]
    background = ImageColor.getrgb(images.BACKGROUND)
    assert near(image.getpixel((end - 20, middle[0])), ImageColor.getrgb(colours[0]))
    assert near(image.getpixel((left + 20, middle[0])), ImageColor.getrgb(colours[0]))
    assert image.getpixel((end + 4, middle[0])) == background  # the largest bar ends at the maximum width
    assert near(image.getpixel((left + 20, middle[1])), ImageColor.getrgb(colours[1]))  # 4 of 40: 47 px
    assert image.getpixel((left + 60, middle[1])) == background
    assert near(image.getpixel((left + 4, middle[2])), ImageColor.getrgb(colours[2]))  # a zero: 8 px
    assert image.getpixel((left + 14, middle[2])) == background
    neutral = ImageColor.getrgb(images.NEUTRAL)
    assert near(image.getpixel((left + 4, middle[3])), neutral) and near(image.getpixel((left + 4, middle[4])), neutral)
    assert near(image.getpixel((left + 4, middle[5])), (0x11, 0x22, 0x33))  # the alpha of a colour is dropped
    # nothing at the left of the bars but the names, and nothing between two rows
    assert blank(image, (left - 10, row_band(0)[0], left - HALO, row_band(5)[1]))
    assert blank(image, (left, row_band(0)[1] + 4, end, row_band(1)[0] - 4))


def test_poll_png_without_seats_leaves_the_column_empty():
    """La columna «Escaños» sólo sale si algún partido de la imagen trae escaños; la celda de un partido sin ellos
    queda vacía."""
    images = pages.images
    column = (images.RIGHT - 190, images.ROW_TOP - 30, images.RIGHT + 1, images.ROW_TOP + 6 * images.ROW_STEP)
    rows = [row_of('PP', 31.9), row_of('PSOE', 25.9), row_of('VOX', 18.6)]
    assert blank(drawn(card(rows)), column)
    seats = [row_of('PP', 31.9, seats='125–136'), row_of('PSOE', 25.9), row_of('VOX', 18.6, seats='63')]
    image = drawn(card(seats))
    assert not blank(image, column)
    assert not blank(image, (images.RIGHT - 190, images.ROW_TOP - 30, images.RIGHT + 1, images.ROW_TOP - 4))  # header
    cells = [(images.RIGHT - 190, row_band(index)[0], images.RIGHT + 1, row_band(index)[1]) for index in range(3)]
    assert not blank(image, cells[0]) and blank(image, cells[1]) and not blank(image, cells[2])
    # the column is one decision for the whole card (``has_seats``), not one per row
    assert blank(drawn(dict(card(seats), has_seats=False)), column)


def test_poll_png_marks_the_change_after_the_figure():
    """Detrás de la cifra, el cambio frente a la entrega anterior: verde si sube, rojo si baja, y nada de color si
    no hay entrega anterior."""
    images = pages.images
    image = drawn(card([row_of('PP', 31.9, change={'direction': 'up', 'text': '0,4'}),
                        row_of('PSOE', 25.9, change={'direction': 'down', 'text': '0,5'}),
                        row_of('VOX', 18.6, change={'direction': 'same', 'text': '='}), row_of('FA', 7.5)]))

    def after(index):
        """Lo que sigue a la barra de una fila, hasta la columna de los escaños."""
        return images.BAR_LEFT + 60, row_band(index)[0], images.RIGHT - 200, row_band(index)[1]

    assert count_near(image, after(0), UP) > 60 and count_near(image, after(0), DOWN) == 0
    assert count_near(image, after(1), DOWN) > 60 and count_near(image, after(1), UP) == 0
    assert [count_near(image, after(index), colour) for index in (2, 3) for colour in (UP, DOWN)] == [0, 0, 0, 0]
    plain = drawn(card([row_of('PP', 31.9), row_of('PSOE', 25.9), row_of('VOX', 18.6), row_of('FA', 7.5)]))
    # the «=» of the third row is ink the plain image does not have
    assert image.crop(after(2)).tobytes() != plain.crop(after(2)).tobytes()
    assert image.crop(after(3)).tobytes() == plain.crop(after(3)).tobytes()


def test_figures_never_reach_the_seats_column():
    """Con la barra más larga, la cifra más ancha, un cambio de dos cifras y la horquilla más ancha, queda aire
    entre lo que sigue a la barra y los escaños; también cuando unas siglas largas empujan las barras o cuando los
    escaños son un texto que no cabe en su columna."""
    images = pages.images
    change = {'direction': 'down', 'text': '10,0'}

    for name, seats in (('PP', '125–136'), ('COMPROMIS', '125–136'), ('W' * 12, '125–136'),
                        ('PP', '1.000.000–2.000.000')):
        image = drawn(card([row_of(name, 100, change=change, seats=seats), row_of('PSOE', 0, seats='0')]))
        top, bottom = row_band(0)
        columns = [blank(image, (x, top, x + 1, bottom)) for x in range(images.BAR_RIGHT + 1, images.RIGHT + 1)]
        inked = [index for index, empty in enumerate(columns) if not empty]
        # the figure right after the bar and the seats against the right edge
        assert inked and inked[0] < 20 and inked[-1] > len(columns) - 8, (name, seats)
        gaps = [second - first - 1 for first, second in zip(inked, inked[1:])]
        assert max(gaps) >= 24, (name, seats, max(gaps))


def test_poll_png_leaves_the_corner_of_the_link_title_free():
    """La esquina inferior izquierda queda libre (X superpone ahí el título del enlace), y también las franjas de
    arriba y de abajo que Facebook recorta; el pie va abajo a la derecha."""
    rows = [row_of(name, value, change={'direction': 'up', 'text': '0,4'}, seats='12')
            for name, value in zip(PARTIES, (31.9, 25.9, 18.6, 7.5, 2.7, 1.6, 1.5, 1.4))]
    image = drawn(card(rows, more='y 7 partidos más en demoscopio.es'))
    assert blank(image, (0, 611, 600, 675)) and blank(image, (0, 0, 1200, 24)) and blank(image, (0, 651, 1200, 675))
    assert not blank(image, (700, 611, 1200, 651))  # the footnote
    assert blank(drawn(card(rows)), (0, 611, 1200, 675))  # no footnote, and no seventh row
    assert blank(image, (0, row_band(5)[1] + 8, 1200, 611))


def test_poll_png_has_the_header_of_the_sketch():
    """Arriba, el título y la línea gris a la izquierda y la marca a la derecha, sin pisarse."""
    images = pages.images
    image = drawn(card([row_of('PP', 31.9)]))
    mark = (images.RIGHT - images.WORDMARK_WIDTH, 60, images.RIGHT + 1, 120)
    assert count_near(image, mark, ACCENT, 12) > 100 and count_near(image, mark, INK, 12) > 500  # lens and letters
    # pasted with its transparency: around the letters, inside its own rectangle, the background of the card
    own = (mark[0], images.WORDMARK_TOP, images.RIGHT, images.WORDMARK_TOP + images.wordmark().height // images.SCALE)
    paper = ImageColor.getrgb(images.BACKGROUND)
    assert count_near(image, own, paper, 0) > (own[2] - own[0]) * (own[3] - own[1]) // 4
    assert blank(image, (images.RIGHT + HALO, 0, 1200, 675)) and blank(image, (0, 0, images.MARGIN - HALO, 675))
    assert count_near(image, (images.MARGIN, 60, 600, 115), INK, 12) > 2000  # the title
    assert not blank(image, (images.MARGIN, 128, 900, 160))  # the grey line
    long = 'Instituto de Estudios Sociales Avanzados de Andalucía para la Radio y Televisión de Andalucía'
    wide = drawn(card([row_of('PP', 31.9)], title=long, subtitle=' · '.join([long] * 3)))
    gap = (images.RIGHT - images.WORDMARK_WIDTH - 30, 30, images.RIGHT - images.WORDMARK_WIDTH - 4, 125)
    # cut before the wordmark and before the right edge
    assert blank(wide, gap) and blank(wide, (images.RIGHT + HALO, 0, 1200, 675))
    assert wide.crop(mark).tobytes() == image.crop(mark).tobytes()


def test_poll_png_survives_odd_inputs():
    """Nombres con letras que las fuentes no tienen, marcas de HTML, saltos de línea o textos larguísimos, cifras
    extremas, ninguna fila o demasiadas: siempre sale una imagen de 1200×675."""
    odd = [card([]), card([row_of('PP', 33.1)]), card([row_of('PP', 33.1)], title='A' * 200, subtitle='B' * 400),
           card([row_of('<script>', 50), row_of('😀🗳', 30), row_of('漢字', 20), row_of('PODEMOS-IU-AV', 5)],
                title='<b>Casa</b> & "Medio" 😀', subtitle='línea\nrota\ty más'),
           card([row_of('A', 0), row_of('B', 0)]), card([row_of('A', 100, seats='350'), row_of('B', 0, seats='0')]),
           card([row_of('P%d' % index, 20 - index, seats=str(index)) for index in range(13)], more='x' * 300),
           card([row_of('PP', 33.1)], title='', subtitle='')]

    for data in odd:
        image = drawn(data)
        assert image.size == (1200, 675) and image.format == 'PNG', data['title'][:20]
        assert blank(image, (0, 0, 1200, 24)) and blank(image, (0, 651, 1200, 675)), data['title'][:20]
        assert blank(image, (0, 611, 600, 675)), data['title'][:20]

    # a line break in a name does not break the line of the image
    assert blank(drawn(odd[3]), (0, 160, 1200, pages.images.ROW_TOP + pages.images.BAR_TOP - HALO))


def page_of(api, path, query=''):
    """El HTML de una página que responde 200."""
    status, _, body = call(api, path, query=query)
    assert status == 200, (path, query, status)
    return html(body)


def meta_of(page, name):
    """El contenido de una etiqueta `og:` de la cabecera, tal como va escrito (y va una sola vez)."""
    found = re.findall(r'<meta property="{}" content="([^"]*)">'.format(re.escape(name)), head_of(page))
    assert len(found) == 1, (name, found)
    return found[0]


def test_poll_image_route_answers_a_cacheable_png(houses, fresh_app):
    """La imagen de un sondeo vive en la dirección de su página acabada en `.png`: un PNG de 1200×675 con un día
    de caché y su ETag, que dibuja el sondeo de lo último publicado."""
    status, headers, body = call(houses, IMAGE_PATH)
    assert status == 200 and headers['content-type'] == 'image/png'
    assert headers['cache-control'] == 'public, max-age=86400'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert body == pages.images.poll_png(pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos'))
    assert Image.open(io.BytesIO(body)).size == (1200, 675)
    # a HEAD answers the same headers without drawing a body
    status, head_headers, nothing = call(houses, IMAGE_PATH, method='HEAD')
    assert status == 200 and nothing == b'' and head_headers['content-type'] == 'image/png'
    assert head_headers['etag'] == headers['etag']


def test_poll_image_route_by_house_id_and_with_outlet(houses, fresh_app):
    """La imagen responde igual pedida por el nombre de la casa o por su identificador, y la de un sondeo con
    medio lleva el medio en su dirección, como su página."""
    write_polls(fresh_app.fs, [{}, {'sponsor': 'El Mundo', 'PP': 34.0}])
    status, _, plain = call(houses, IMAGE_PATH)
    assert status == 200 and call(houses, '/es/sondeos/2026-10-05/7.png')[2] == plain
    status, headers, outlet = call(houses, POLL_PATH + '/el-mundo.png')
    assert status == 200 and headers['content-type'] == 'image/png' and outlet != plain
    data = pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos', 'el-mundo')
    assert data['title'] == 'Sigma Dos para El Mundo' and data['rows'][0]['share'] == '34,0%'
    assert outlet == pages.images.poll_png(data)
    assert call(houses, '/es/sondeos/2026-10-05/7/el-mundo.png')[2] == outlet


def test_poll_image_route_refuses_what_the_page_refuses(houses, monkeypatch):
    """Donde la página del sondeo responde 404 o 400, su imagen también, con la página de error y sin dibujar
    nada; y sólo `.png` es una imagen."""
    def never(preview):
        raise AssertionError('an image was drawn')

    monkeypatch.setattr(pages.images, 'poll_png', never)
    refused = (('/es/sondeos/2026-10-05/nadie', 404), ('/es/sondeos/2026-10-06/sigma-dos', 404),
               ('/es/sondeos/2026-10-05/sigma-dos/nadie', 404), ('/es/sondeos/2026-13-45/sigma-dos', 400),
               ('/zz/sondeos/2026-10-05/sigma-dos', 404))

    for path, code in refused:
        assert call(houses, path)[0] == code, path
        status, headers, body = call(houses, path + '.png')
        assert status == code and headers['content-type'] == 'text/html; charset=utf-8', path
        assert headers['cache-control'] == 'no-store' and 'etag' not in headers and b'<html' in body, path

    for path in (POLL_PATH + '.jpg', POLL_PATH + '.PNG', POLL_PATH + '.png.png', POLL_PATH + 'xpng', '/es.png'):
        assert call(houses, path)[0] == 404, path


def test_poll_image_ignores_the_query(houses, fresh_app):
    """La imagen es siempre la de lo último publicado: ni `mode`, ni `run`, ni la huella `v`, ni un parámetro
    inválido la cambian o la hacen fallar."""
    body = call(houses, IMAGE_PATH)[2]

    for query in ('mode=forecast', 'mode=nowcast', 'mode=zz', 'run=' + RUN, 'run=20200101-000000', 'run=nope',
                  'scope=zz', 'v=abc', 'zz=1&utm_source=x'):
        status, headers, other = call(houses, IMAGE_PATH, query=query)
        assert status == 200 and headers['content-type'] == 'image/png' and other == body, query


def test_poll_page_declares_its_image(houses, fresh_app):
    """La página de un sondeo declara su imagen como `og:image`: la dirección limpia de la página acabada en
    `.png`, absoluta y con la huella de lo que pinta, con un texto alternativo que dice las cifras. Nada más la
    enlaza y no entra en el sitemap."""
    data = pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos')
    page = page_of(houses, POLL_PATH)
    address = SITE + IMAGE_PATH + '?v=' + data['version']
    assert meta_of(page, 'og:image') == address
    assert meta_of(page, 'og:image:alt') == 'Sondeo de Sigma Dos (5 oct 2026): PP 33,1%.' == data['alt']
    assert meta_of(page, 'og:image:type') == 'image/png'
    assert (meta_of(page, 'og:image:width'), meta_of(page, 'og:image:height')) == ('1200', '675')
    assert '<meta name="twitter:card" content="summary_large_image">' in page
    assert page.count(IMAGE_PATH) == 1 and 'og-image.png' not in page
    # asked by the id of the house or with parameters, the same address
    for odd, query in (('/es/sondeos/2026-10-05/7', ''), (POLL_PATH, 'mode=nowcast&utm_source=x')):
        assert meta_of(page_of(houses, odd, query=query), 'og:image') == address, (odd, query)
    # and the address answers the image
    status, headers, _ = call(houses, IMAGE_PATH, query='v=' + data['version'])
    assert status == 200 and headers['content-type'] == 'image/png'
    assert not [loc for loc in sitemap(houses) if '.png' in loc]


def test_poll_image_address_follows_the_configured_origin(houses, fresh_app):
    """La dirección de la imagen sale de `web.site_url`, sin barra doble aunque venga con barra final."""
    fresh_app.config.offset_set('web', {'site_url': 'https://demoscopio.dev/'})  # the database stays
    image = meta_of(page_of(houses, POLL_PATH), 'og:image')
    assert image.startswith('https://demoscopio.dev' + IMAGE_PATH + '?v=') and 'demoscopio.es' not in image


def test_poll_image_alt_is_escaped_once(houses, fresh_app):
    """El nombre de una casa con marcas de HTML, comillas y `&` llega al texto alternativo escapado una vez, sin
    abrir ninguna etiqueta, y su imagen se dibuja."""
    change = {'pollster': 'A <b> & "C"'}
    write_polls(fresh_app.fs, [change])
    path = pages.poll_detail_href('es', fixture_data('polls')['polls'][0] | change)
    page = page_of(houses, path)
    alt = meta_of(page, 'og:image:alt')
    assert alt == 'Sondeo de A &lt;b&gt; &amp; &#34;C&#34; (5 oct 2026): PP 33,1%.'
    assert html_escapes.unescape(alt) == 'Sondeo de A <b> & "C" (5 oct 2026): PP 33,1%.'
    assert '<b>' not in head_of(page) and meta_of(page, 'og:image').startswith(SITE + path + '.png?v=')
    status, headers, body = call(houses, path + '.png')
    assert status == 200 and headers['content-type'] == 'image/png' and body.startswith(b'\x89PNG')


def test_image_version_follows_the_figures(houses, fresh_app):
    """Si una cifra del sondeo cambia, cambia la huella de la dirección de su imagen, y las redes piden la nueva."""
    poll = fixture_data('polls')['polls'][0]
    before = sondeos.preview_of(poll, None, ['PP'], {'PP': '#1d84ce'}, ELECTION, HOST)
    write_polls(fresh_app.fs, [{'PP': 34.0}])
    after = pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos')
    assert after['rows'][0]['share'] == '34,0%' and after['version'] != before['version']
    assert after == sondeos.preview_of(poll | {'PP': 34.0}, None, ['PP'], {'PP': '#1d84ce'}, ELECTION, HOST)
    assert meta_of(page_of(houses, POLL_PATH), 'og:image') == SITE + IMAGE_PATH + '?v=' + after['version']


def test_pinned_run_page_declares_the_latest_image_or_the_brand_one(houses, fresh_app, monkeypatch):
    """La imagen siempre dibuja lo último publicado: una página con un run fijado declara la huella y el texto de
    ese último run, y la imagen de marca si el sondeo ya no está en él."""
    own = pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos')
    latest = dict(own, version='0badcafe', alt='Sondeo de Sigma Dos (5 oct 2026): PP 35,0%.')
    monkeypatch.setattr(pages, 'latest_poll_preview', lambda *args: latest)
    pinned = page_of(houses, POLL_PATH, query='run=' + RUN)
    assert meta_of(pinned, 'og:image') == SITE + IMAGE_PATH + '?v=0badcafe'
    assert meta_of(pinned, 'og:image:alt') == latest['alt']
    # without a pinned run the page already shows the latest: its own data, read once
    plain = page_of(houses, POLL_PATH)
    assert meta_of(plain, 'og:image') == SITE + IMAGE_PATH + '?v=' + own['version']
    assert meta_of(plain, 'og:image:alt') == own['alt']
    monkeypatch.setattr(pages, 'latest_poll_preview', lambda *args: None)
    gone = page_of(houses, POLL_PATH, query='run=' + RUN)
    assert meta_of(gone, 'og:image') == pages.og_image_url() and meta_of(gone, 'og:image:alt') == BRAND_ALT


@pytest.fixture
def drawing_caches():
    """Olvida, antes y después de la prueba, lo que el dibujo guarda por proceso (fuentes, marca y si puede
    dibujar)."""
    images = pages.images

    def forget():
        for cached in (images.font, images.wordmark, images.available):
            cached.cache_clear()

    forget()
    yield
    forget()


@pytest.mark.parametrize('missing', ['FONTS_DIR', 'WORDMARK_PATH'])
def test_poll_page_keeps_the_brand_image_when_nothing_can_be_drawn(houses, fresh_app, monkeypatch, tmp_path,
                                                                   drawing_caches, missing):
    """Si este proceso no puede dibujar (faltan las fuentes o la marca, o este Pillow no lee las fuentes), la página
    de un sondeo sigue con la imagen de marca en vez de declarar una imagen que respondería un error; se comprueba
    una vez por proceso y se avisa."""
    images = pages.images
    assert images.available() is True and images.available.cache_info().currsize == 1
    own = meta_of(page_of(houses, POLL_PATH), 'og:image')
    assert own.startswith(SITE + IMAGE_PATH + '?v=')
    images.available.cache_clear()
    images.font.cache_clear()
    images.wordmark.cache_clear()
    monkeypatch.setattr(images, missing, str(tmp_path / 'nothing-here'))

    with pytest.warns(UserWarning, match='cannot be drawn'):
        assert images.available() is False

    assert images.available() is False  # asked once: no second warning, no second try
    page = page_of(houses, POLL_PATH)
    assert meta_of(page, 'og:image') == pages.og_image_url() and meta_of(page, 'og:image:alt') == BRAND_ALT
    assert IMAGE_PATH not in page
    pinned = page_of(houses, POLL_PATH, query='run=' + RUN)
    assert meta_of(pinned, 'og:image') == pages.og_image_url()
    # the address itself answers an error, not a broken picture
    status, headers, _ = call(houses, IMAGE_PATH)
    assert status == 500 and headers['cache-control'] == 'no-store'


def test_other_pages_keep_the_brand_image(houses, fresh_app):
    """Sólo la página de un sondeo y el detalle de una elección pasada tienen imagen propia: el resto sigue con la
    de marca y su texto alternativo."""
    write_study(fresh_app.fs)
    paths = site_paths(houses)
    with_image = {POLL_PATH, '/elecciones/es/generales/2023-07'}
    assert with_image <= set(paths) and len(paths) > 10

    for path in paths:
        page = page_of(houses, path)
        own = path in with_image
        assert (meta_of(page, 'og:image') == pages.og_image_url()) is not own, path
        assert (meta_of(page, 'og:image:alt') == BRAND_ALT) is not own, path
        assert meta_of(page, 'og:image:type') == 'image/png', path
