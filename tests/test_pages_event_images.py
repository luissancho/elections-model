"""Tests de la imagen propia del enlace de una elección pasada: lo que dice (como datos, con el cambio frente a la
elección anterior), su dibujo (el de la tarjeta de un sondeo) y su dirección."""
import copy
import hashlib
import io
import os
import re
import shlex
import subprocess

import pytest
from PIL import Image

from mtpy.core.api import HttpError
from mtpy.lib import bundle, pages
from mtpy.lib.pages import elecciones
from mtpy.lib.pages.filters import SNIPPET_LENGTH
from tests.test_api_asgi import call
from tests.test_pages import api, houses, houses_without_analysis, site_api  # noqa: F401
from tests.test_pages_crawlers import POLL_PATH, sitemap
from tests.test_pages_images import BRAND_ALT, SITE, meta_of, page_of
from tests.test_webapi_unit import RUN, fixture_data, write_event

HOST = 'demoscopio.es'
TITLE = 'Elecciones Generales Julio 2023'
DATE = '2023-07-23'
# The result of a general election as published: by vote, with the rest of the candidacies («-») among the parties
ROWS = [
    {'party': 'PP', 'pct': 33.06, 'seats': 137, 'color': '#1d84ce'},
    {'party': 'PSOE', 'pct': 31.68, 'seats': 121, 'color': '#ef1c27'},
    {'party': 'VOX', 'pct': 12.38, 'seats': 33, 'color': '#63be21'},
    {'party': 'SUMAR', 'pct': 12.33, 'seats': 31, 'color': '#e51c55'},
    {'party': 'ERC', 'pct': 1.89, 'seats': 7, 'color': '#ffb232'},
    {'party': 'JxCat', 'pct': 1.6, 'seats': 7, 'color': '#ee5976'},
    {'party': 'EHB', 'pct': 1.36, 'seats': 6, 'color': '#b5cf18'},
    {'party': '-', 'pct': 1.2, 'seats': 0, 'color': '#9e9e9e'},
    {'party': 'PNV', 'pct': 1.12, 'seats': 5, 'color': '#4aae4a'},
    {'party': 'PACMA', 'pct': 0.67, 'seats': 0, 'color': '#adc02e'},
    {'party': 'BNG', 'pct': 0.62, 'seats': 1, 'color': '#adcfef'},
    {'party': 'CC', 'pct': 0.47, 'seats': 1, 'color': '#ffd700'},
    {'party': 'UPN', 'pct': 0.21, 'seats': 1, 'color': '#2a52be'},
]
# The election before: SUMAR did not stand under that name
BEFORE = {'date': '2019-11-10',
          'pct': {'PP': 20.82, 'PSOE': 28.0, 'VOX': 15.09, 'UP': 12.86, 'ERC': 3.61, 'JxCat': 2.19, '-': 1.0}}
LEGEND = 'Resultado oficial del 23 jul 2023 · Cambio frente a la elección anterior (10 nov 2019)'
EVENT_PATH = '/elecciones/es/generales/2023-07'  # the only election of the fixture bundle
EVENT_IMAGE = EVENT_PATH + '.png'
CHECK_API = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deploy', 'check-api.sh')


def preview(rows=None, previous=BEFORE, title=TITLE, date=DATE, host=HOST):
    """Lo que dice la imagen de una elección (las generales de 2023 por defecto)."""
    return elecciones.result_preview(title, date, ROWS if rows is None else rows, previous, host)


def rows_with(**changes):
    """Las filas del resultado con los campos de algunos partidos cambiados (`PP={'pct': 33.07}`)."""
    return [row | changes.get(row['party'], {}) for row in copy.deepcopy(ROWS)]


def test_result_preview_marks_the_change_against_the_previous_election():
    """La imagen de una elección enseña los seis partidos con más voto, por voto, con su resultado, sus escaños y
    su cambio frente a la elección anterior; un partido que no se presentó con esas siglas va sin cambio."""
    data = preview()
    assert data['title'] == TITLE and data['subtitle'] == LEGEND
    assert [row['name'] for row in data['rows']] == ['PP', 'PSOE', 'VOX', 'SUMAR', 'ERC', 'JxCat']
    assert [row['share'] for row in data['rows']] == ['33,1%', '31,7%', '12,4%', '12,3%', '1,9%', '1,6%']
    assert [row['value'] for row in data['rows']] == [33.06, 31.68, 12.38, 12.33, 1.89, 1.6]
    assert [row['color'] for row in data['rows']] == [row['color'] for row in ROWS[:6]]
    assert [row['seats'] for row in data['rows']] == ['137', '121', '33', '31', '7', '7'] and data['has_seats']
    up, down = (lambda text: {'direction': 'up', 'text': text}), (lambda text: {'direction': 'down', 'text': text})
    assert [row['change'] for row in data['rows']] == [up('12,2'), up('3,7'), down('2,7'), None, down('1,7'),
                                                      down('0,6')]
    # the same share as before is told with the equals sign, also when the difference rounds to nothing
    same = preview(previous=BEFORE | {'pct': BEFORE['pct'] | {'PP': 33.06, 'PSOE': 31.72}})
    assert [row['change'] for row in same['rows'][:2]] == [{'direction': 'same', 'text': '='}] * 2
    # the rows come by vote whatever the order they are published in
    assert [row['name'] for row in preview(list(reversed(ROWS)))['rows']] == [row['name'] for row in data['rows']]


def test_result_preview_without_a_previous_election():
    """La primera elección publicada de un ámbito, o una en la que ninguno de los seis se presentó la vez
    anterior, no lleva cambios, y la línea gris no promete ninguno."""
    first = preview(previous=None)
    assert first['subtitle'] == 'Resultado oficial del 23 jul 2023'
    assert [row['change'] for row in first['rows']] == [None] * 6
    assert [row['share'] for row in first['rows']] == [row['share'] for row in preview()['rows']]
    new = preview(previous={'date': '2019-11-10', 'pct': {'UP': 12.86, 'Cs': 6.8, 'PP': None}})
    assert new['subtitle'] == 'Resultado oficial del 23 jul 2023' and not any(row['change'] for row in new['rows'])
    # one party to compare is enough for the legend
    one = preview(previous={'date': '2019-11-10', 'pct': {'JxCat': 2.19}})
    assert one['subtitle'] == LEGEND and [bool(row['change']) for row in one['rows']] == [False] * 5 + [True]


def test_result_preview_skips_the_rest_row_and_counts_the_seated():
    """La fila «-» (el resto de candidaturas) no es un partido: no entra entre los seis aunque tenga más voto, ni
    en el pie, que cuenta sólo los partidos con escaño que no caben."""
    assert elecciones.REST_PARTY == '-'
    data = preview(rows_with(**{'-': {'pct': 13.0, 'seats': 0}}))
    assert [row['name'] for row in data['rows']] == ['PP', 'PSOE', 'VOX', 'SUMAR', 'ERC', 'JxCat']
    assert data['more'] == 'y 5 partidos más con escaño en demoscopio.es'  # EHB, PNV, BNG, CC and UPN; not PACMA
    assert preview(rows_with(**{'-': {'seats': 3}}))['more'] == data['more']
    assert preview(host='')['more'] == 'y 5 partidos más con escaño'
    one = preview(rows_with(PNV={'seats': 0}, BNG={'seats': 0}, CC={'seats': 0}, UPN={'seats': None}))
    assert one['more'] == 'y 1 partido más con escaño en demoscopio.es'
    assert preview(ROWS[:6] + ROWS[9:10])['more'] is None and preview(ROWS[:3])['more'] is None


def test_result_preview_texts():
    """El texto alternativo y la descripción dicen las cifras que se pintan; la descripción añade los escaños (la
    palabra, sólo en la primera cifra), cabe en una descripción y nunca corta una cifra."""
    data = preview()
    assert data['alt'] == ('Elecciones Generales Julio 2023, resultado oficial del 23 jul 2023: PP 33,1%, PSOE 31,7%, '
                           'VOX 12,4%, SUMAR 12,3%, ERC 1,9%, JxCat 1,6%.')
    assert data['description'] == ('Elecciones Generales Julio 2023: PP 33,1% (137 escaños), PSOE 31,7% (121), '
                                   'VOX 12,4% (33), SUMAR 12,3% (31), ERC 1,9% (7), JxCat 1,6% (7).')
    assert len(data['description']) <= SNIPPET_LENGTH and pages.snippet(data['description']) == data['description']
    # one seat is said in the singular; a party without seats published has no parenthesis and leaves the word to
    # the next one
    single = preview(rows_with(PP={'seats': 1}))['description']
    assert single.startswith('Elecciones Generales Julio 2023: PP 33,1% (1 escaño), PSOE 31,7% (121), ')
    unseated = preview(rows_with(PP={'seats': None}))
    assert unseated['description'].startswith('Elecciones Generales Julio 2023: PP 33,1%, PSOE 31,7% (121 escaños), ')
    assert unseated['rows'][0]['seats'] is None and unseated['has_seats']
    # a long name leaves room for fewer figures, the last one whole
    long_title = 'Elecciones al Parlamento de la Comunidad Valenciana de mayo de 2023, repetidas'
    fewer = preview(title=long_title)['description']
    assert fewer == long_title + ': PP 33,1% (137 escaños), PSOE 31,7% (121), VOX 12,4% (33), SUMAR 12,3% (31).'
    assert len(fewer) <= SNIPPET_LENGTH < len(fewer) + len(', ERC 1,9% (7)')


def test_result_preview_of_odd_elections():
    """Sin resultado publicado la imagen se queda en su cabecera y la página conserva su frase; una fila sin voto
    no se pinta, y una elección sin escaños publicados no tiene columna."""
    empty = preview([])
    assert empty['rows'] == [] and empty['more'] is None and not empty['has_seats']
    assert empty['description'] is None and empty['subtitle'] == 'Resultado oficial del 23 jul 2023'
    assert empty['alt'] == 'Elecciones Generales Julio 2023, resultado oficial del 23 jul 2023.'
    assert Image.open(io.BytesIO(pages.images.result_png(empty))).size == (1200, 675)
    assert preview([{'party': '-', 'pct': 100.0, 'seats': 0, 'color': '#9e9e9e'}])['rows'] == []
    voteless = preview(rows_with(PSOE={'pct': None}, VOX={'pct': float('nan')}))
    assert [row['name'] for row in voteless['rows']] == ['PP', 'SUMAR', 'ERC', 'JxCat', 'EHB', 'PNV']
    seatless = preview([dict(row, seats=None) for row in ROWS])
    assert not seatless['has_seats'] and seatless['more'] is None
    assert seatless['description'].startswith('Elecciones Generales Julio 2023: PP 33,1%, PSOE 31,7%, ')
    # seats published as floats are told as whole numbers
    assert preview(rows_with(PP={'seats': 137.0}))['rows'][0]['seats'] == '137'
    # an election without a name is told by its date
    assert elecciones.result_title({'name': None}, DATE) == 'Elección del 23 jul 2023'
    assert elecciones.result_title({'name': TITLE}, DATE) == TITLE


def test_result_preview_version_follows_what_is_drawn():
    """La huella de la dirección cambia con cualquier cosa que la imagen pinta (el título, la fecha, una cifra, un
    escaño, el resultado anterior, el pie) y sólo con eso."""
    base = preview()
    assert re.fullmatch(r'[0-9a-f]{8}', base['version']) and preview()['version'] == base['version']
    changed = {
        'title': preview(title='Generales 2023'),
        'date': preview(date='2023-07-24'),
        'share': preview(rows_with(PP={'pct': 33.16})),
        'seats': preview(rows_with(PP={'seats': 138})),
        'colour': preview(rows_with(PP={'color': '#000000'})),
        'previous share': preview(previous=BEFORE | {'pct': BEFORE['pct'] | {'PP': 21.82}}),
        'previous date': preview(previous=BEFORE | {'date': '2019-04-28'}),
        'footnote': preview(rows_with(UPN={'seats': 0})),
        'host': preview(host='demoscopio.dev'),
    }
    versions = {key: data['version'] for key, data in changed.items()}
    assert base['version'] not in versions.values() and len(set(versions.values())) == len(versions)
    # what is not drawn does not move it: a share that rounds to the same figure, a party that does not fit
    assert preview(rows_with(PP={'pct': 33.06}, PACMA={'pct': 0.7}))['version'] == base['version']
    assert preview(previous=BEFORE | {'pct': BEFORE['pct'] | {'UP': 1.0}})['version'] == base['version']


def test_result_png_is_the_card_of_a_poll():
    """La imagen de una elección es la tarjeta de un sondeo con otros datos: el mismo dibujo, con las barras del
    color de cada partido."""
    data = preview()
    png = pages.images.result_png(data)
    assert png == pages.images.poll_png(data) and png[:8] == b'\x89PNG\r\n\x1a\n'
    image = Image.open(io.BytesIO(png)).convert('RGB')
    assert image.size == (1200, 675)
    # inside the bar of the first row (ROW_TOP + BAR_TOP + half the bar), past its rounded corner
    assert image.getpixel((pages.images.BAR_LEFT + 60, 226)) == (0x1d, 0x84, 0xce)
    assert pages.images.result_png(preview(previous=None)) != png


def write_results(fs, date, rows, scope='es'):
    """Reescribe el resultado de una elección ya publicada con `write_event` (antes de la primera lectura)."""
    data = fixture_data('event-results') | {'rows': rows}
    bundle.BundleWriter(fs).write_json(bundle.path_event(scope, date, 'results'), 'event-results', data, scope)


def state_of(scope='es', **query):
    """Los parámetros de la página de una elección de ese ámbito."""
    return pages.parse_state(query, scope)


def test_previous_result_reads_the_election_before(houses, fresh_app):
    """La elección anterior es la publicada inmediatamente antes en el mismo ámbito; la primera no tiene."""
    write_event(fresh_app.fs, date='2019-11-10', model=False)
    write_event(fresh_app.fs, date='2015-12-20', model=False)
    write_results(fresh_app.fs, '2019-11-10', [{'party': 'PP', 'pct': 20.82, 'seats': 89},
                                               {'party': 'UP', 'pct': 12.86, 'seats': 35},
                                               {'party': '-', 'pct': 1.0, 'seats': 0}])
    assert elecciones.previous_result('es', '2023-07-23') == {
        'date': '2019-11-10', 'pct': {'PP': 20.82, 'UP': 12.86, '-': 1.0}}
    before = elecciones.previous_result('es', '2019-11-10')
    assert before['date'] == '2015-12-20' and before['pct']['PP'] == 33.06  # the result of the fixture
    assert elecciones.previous_result('es', '2015-12-20') is None
    assert elecciones.previous_result('es-md', '2023-05-28') is None


def test_event_preview_reads_the_published_election(houses, fresh_app):
    """Lo que dice la imagen sale de lo publicado, con los colores del catálogo, y es lo mismo que lleva el
    contexto de la página; la única elección publicada de un ámbito no tiene con qué compararse."""
    data = pages.event_preview(state_of(), '2023-07-23')
    context = pages.eleccion_context(state_of(), '2023-07-23')
    assert data == context['preview']
    assert data['title'] == 'Generales 2023' and data['subtitle'] == 'Resultado oficial del 23 jul 2023'
    assert [(row['name'], row['share'], row['seats']) for row in data['rows']] == [
        ('PP', '33,1%', '160'), ('PSOE', '31,4%', '150'), ('VOX', '12,3%', '40'), ('Cs', '0,1%', '0')]
    assert [row['color'] for row in data['rows']] == [row['color'] for row in context['results_rows']]
    assert [row['change'] for row in data['rows']] == [None] * 4 and data['more'] is None
    assert data['alt'] == 'Generales 2023, resultado oficial del 23 jul 2023: PP 33,1%, PSOE 31,4%, VOX 12,3%, Cs 0,1%.'
    assert data['description'] == ('Generales 2023: PP 33,1% (160 escaños), PSOE 31,4% (150), VOX 12,3% (40), '
                                   'Cs 0,1% (0).')
    assert pages.latest_event_preview('es', '2023-07-23') == data


def test_event_preview_compares_with_the_published_election_before(houses, fresh_app):
    """Con la elección anterior publicada, cada partido lleva su cambio frente a ella."""
    write_event(fresh_app.fs, date='2019-11-10', model=False)
    write_results(fresh_app.fs, '2019-11-10', [{'party': 'PSOE', 'pct': 28.0, 'seats': 120},
                                               {'party': 'PP', 'pct': 20.82, 'seats': 89},
                                               {'party': 'Cs', 'pct': 6.8, 'seats': 10}])
    data = pages.event_preview(state_of(), '2023-07-23')
    assert data['subtitle'] == LEGEND
    assert [row['change'] and (row['change']['direction'], row['change']['text']) for row in data['rows']] == [
        ('up', '12,2'), ('up', '3,4'), None, ('down', '6,7')]
    assert data == pages.eleccion_context(state_of(), '2023-07-23')['preview']
    # and the election before is the first of the scope
    first = pages.event_preview(state_of(), '2019-11-10')
    assert first['subtitle'] == 'Resultado oficial del 10 nov 2019' and not any(row['change'] for row in first['rows'])
    assert [row['name'] for row in first['rows']] == ['PSOE', 'PP', 'Cs']


def test_event_preview_refuses_what_the_page_refuses(houses, fresh_app):
    """Una elección sin publicar o una fecha inválida dan el error de la página; `latest_event_preview` calla sólo
    el 404."""
    with pytest.raises(HttpError) as error:
        pages.event_preview(state_of(), '2020-01-01')
    assert error.value.status == 404 and error.value.message == 'event not published'
    with pytest.raises(HttpError) as error:
        pages.event_preview(state_of(), '2023-13-45')
    assert error.value.status == 400
    assert pages.latest_event_preview('es', '2020-01-01') is None
    with pytest.raises(HttpError) as error:
        pages.latest_event_preview('es', '2023-13-45')
    assert error.value.status == 400


def test_event_image_route_answers_a_cacheable_png(houses, fresh_app):
    """La imagen de una elección vive en la dirección de su página acabada en `.png`: un PNG de 1200×675 con un
    día de caché (pedida con su huella) y su ETag, igual pedida por el mes o por la fecha y con cualquier
    consulta."""
    version = pages.latest_event_preview('es', '2023-07-23')['version']
    status, headers, body = call(houses, EVENT_IMAGE, query='v=' + version)
    assert status == 200 and headers['content-type'] == 'image/png'
    assert headers['cache-control'] == 'public, max-age=86400'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert body == pages.images.result_png(pages.latest_event_preview('es', '2023-07-23'))
    assert Image.open(io.BytesIO(body)).size == (1200, 675)
    assert call(houses, '/elecciones/es/generales/2023-07-23.png')[2] == body

    for query in ('mode=forecast', 'mode=zz', 'run=' + RUN, 'run=20200101-000000', 'run=nope', 'scope=zz', 'v=abc',
                  'zz=1&utm_source=x'):
        status, headers, other = call(houses, EVENT_IMAGE, query=query)
        assert status == 200 and headers['content-type'] == 'image/png' and other == body, query

    # a HEAD answers the same headers without a body
    status, head_headers, nothing = call(houses, EVENT_IMAGE, method='HEAD')
    assert status == 200 and nothing == b'' and head_headers['content-type'] == 'image/png'
    assert head_headers['etag'] == headers['etag']


def test_event_image_route_refuses_what_the_page_refuses(houses, monkeypatch):
    """Donde la página de la elección responde 404 o 400, su imagen también, con la página de error y sin dibujar
    nada; y sólo `.png` es una imagen."""
    def never(preview):
        raise AssertionError('an image was drawn')

    monkeypatch.setattr(pages.images, 'result_png', never)
    refused = (('/elecciones/es/generales/2020-01', 404), ('/elecciones/es/generales/2020-01-01', 404),
               ('/elecciones/es/generales/2023-13', 400), ('/elecciones/es/generales/abc', 400),
               ('/elecciones/es/nadie/2023-07', 404), ('/elecciones/es/madrid/2023-05', 404))

    for path, code in refused:
        assert call(houses, path)[0] == code, path
        status, headers, body = call(houses, path + '.png')
        assert status == code and headers['content-type'] == 'text/html; charset=utf-8', path
        assert headers['cache-control'] == 'no-store' and 'etag' not in headers and b'<html' in body, path

    for path in (EVENT_PATH + '.jpg', EVENT_PATH + '.PNG', EVENT_PATH + '.png.png', '/elecciones.png',
                 '/elecciones/es/generales.png'):
        assert call(houses, path)[0] == 404, path


def test_election_page_declares_its_image(houses, fresh_app):
    """La página de una elección declara su imagen como `og:image`: su dirección canónica acabada en `.png`,
    absoluta y con la huella de lo que pinta, con un texto alternativo que dice las cifras. Nada más la enlaza y
    no entra en el sitemap."""
    data = pages.latest_event_preview('es', '2023-07-23')
    page = page_of(houses, EVENT_PATH)
    address = SITE + EVENT_IMAGE + '?v=' + data['version']
    assert meta_of(page, 'og:image') == address
    assert meta_of(page, 'og:image:alt') == data['alt'] == (
        'Generales 2023, resultado oficial del 23 jul 2023: PP 33,1%, PSOE 31,4%, VOX 12,3%, Cs 0,1%.')
    assert meta_of(page, 'og:image:type') == 'image/png'
    assert (meta_of(page, 'og:image:width'), meta_of(page, 'og:image:height')) == ('1200', '675')
    assert page.count(EVENT_IMAGE) == 1 and 'og-image.png' not in page

    # asked by its full date or with parameters, the same address
    for odd, query in (('/elecciones/es/generales/2023-07-23', ''), (EVENT_PATH, 'mode=nowcast&utm_source=x')):
        assert meta_of(page_of(houses, odd, query=query), 'og:image') == address, (odd, query)

    # and the address answers the image
    status, headers, _ = call(houses, EVENT_IMAGE, query='v=' + data['version'])
    assert status == 200 and headers['content-type'] == 'image/png'
    assert not [loc for loc in sitemap(houses) if '.png' in loc]


def test_election_image_version_follows_the_result_before(houses, fresh_app):
    """Con la elección anterior publicada, la imagen gana los cambios y la página declara la huella de esa imagen,
    que no es la de la misma elección sin con qué compararse."""
    write_event(fresh_app.fs, date='2019-11-10', model=False)
    write_results(fresh_app.fs, '2019-11-10', [{'party': 'PP', 'pct': 20.82, 'seats': 89}])
    data = pages.latest_event_preview('es', '2023-07-23')
    assert data['rows'][0]['change'] == {'direction': 'up', 'text': '12,2'}
    assert meta_of(page_of(houses, EVENT_PATH), 'og:image') == SITE + EVENT_IMAGE + '?v=' + data['version']
    rows = [{'party': row['name'], 'pct': row['value'], 'seats': int(row['seats']), 'color': row['color']}
            for row in data['rows']]
    alone = elecciones.result_preview(data['title'], '2023-07-23', rows, None, HOST)
    assert alone['version'] != data['version'] and alone['alt'] == data['alt']
    assert call(houses, EVENT_IMAGE)[2] == pages.images.result_png(data) != pages.images.result_png(alone)


def test_pinned_election_page_declares_the_latest_image(houses, fresh_app, monkeypatch):
    """La imagen siempre dibuja lo último publicado: una página con un run fijado declara la huella y el texto de
    ese último run, y la imagen de marca si la elección ya no está."""
    own = pages.latest_event_preview('es', '2023-07-23')
    latest = dict(own, version='0badcafe', alt='Generales 2023, resultado oficial del 23 jul 2023: PP 35,0%.')
    asked = []
    monkeypatch.setattr(pages, 'latest_event_preview', lambda *args: asked.append(args) or latest)
    pinned = page_of(houses, EVENT_PATH, query='run=' + RUN)
    assert meta_of(pinned, 'og:image') == SITE + EVENT_IMAGE + '?v=0badcafe'
    assert meta_of(pinned, 'og:image:alt') == latest['alt'] and asked == [('es', '2023-07-23')]
    # without a pinned run the page already shows the latest: its own data, read once
    plain = page_of(houses, EVENT_PATH)
    assert meta_of(plain, 'og:image') == SITE + EVENT_IMAGE + '?v=' + own['version']
    assert meta_of(plain, 'og:image:alt') == own['alt'] and len(asked) == 1
    monkeypatch.setattr(pages, 'latest_event_preview', lambda *args: None)
    gone = page_of(houses, EVENT_PATH, query='run=' + RUN)
    assert meta_of(gone, 'og:image') == pages.og_image_url() and meta_of(gone, 'og:image:alt') == BRAND_ALT


def test_election_page_keeps_the_brand_image_when_nothing_can_be_drawn(houses, fresh_app, monkeypatch):
    """Si este proceso no puede dibujar, la página de una elección sigue con la imagen de marca en vez de declarar
    una imagen que respondería un error."""
    assert meta_of(page_of(houses, EVENT_PATH), 'og:image').startswith(SITE + EVENT_IMAGE + '?v=')
    monkeypatch.setattr(pages.images, 'available', lambda: False)
    page = page_of(houses, EVENT_PATH)
    assert meta_of(page, 'og:image') == pages.og_image_url() and meta_of(page, 'og:image:alt') == BRAND_ALT
    assert EVENT_IMAGE not in page
    assert meta_of(page_of(houses, EVENT_PATH, query='run=' + RUN), 'og:image') == pages.og_image_url()


def test_event_image_of_a_scope_without_forecast(houses, fresh_app):
    """Un ámbito sin previsión publicada tiene el detalle de sus elecciones, y con él su imagen: se dibuja con el
    catálogo del run nacional."""
    write_event(fresh_app.fs, scope='es-md', date='2023-05-28', model=False)
    path = pages.event_href('es-md', '2023-05-28')
    assert path == '/elecciones/es/madrid/2023-05'
    data = pages.latest_event_preview('es-md', '2023-05-28')
    assert [row['name'] for row in data['rows']] == ['PP', 'PSOE', 'VOX', 'Cs']
    assert data['subtitle'] == 'Resultado oficial del 28 may 2023'  # the national elections are not its previous one
    assert [row['color'] for row in data['rows']] == [
        row['color'] for row in pages.latest_event_preview('es', '2023-07-23')['rows']]
    page = page_of(houses, path)
    assert meta_of(page, 'og:image') == SITE + path + '.png?v=' + data['version']
    status, headers, body = call(houses, path + '.png')
    assert status == 200 and headers['content-type'] == 'image/png' and body == pages.images.result_png(data)


def test_declared_image_addresses_keep_a_day(houses, fresh_app):
    """La dirección que cada página declara en su `og:image` (la de un sondeo, la de una elección y la de una
    elección con un run fijado), pedida tal cual, responde la imagen con su día de caché."""
    for path, query in ((POLL_PATH, ''), (EVENT_PATH, ''), (EVENT_PATH, 'run=' + RUN)):
        declared = meta_of(page_of(houses, path, query), 'og:image')
        assert declared.startswith(SITE + path + '.png?v='), declared
        address, _, asked = declared[len(SITE):].partition('?')
        status, headers, _ = call(houses, address, query=asked)
        assert status == 200 and headers['content-type'] == 'image/png', declared
        assert headers['cache-control'] == 'public, max-age=86400', declared


def test_election_image_under_an_old_fingerprint_keeps_a_minute(houses, fresh_app):
    """Con la huella de un dibujo que ya no es el de ahora (un enlace antiguo, o un worker atrasado justo tras
    publicar), la dirección responde la imagen de ahora con el minuto de las páginas; con la de ahora, su día."""
    write_event(fresh_app.fs, date='2019-11-10', model=False)
    write_results(fresh_app.fs, '2019-11-10', [{'party': 'PP', 'pct': 20.82, 'seats': 89}])
    data = pages.latest_event_preview('es', '2023-07-23')
    rows = [{'party': row['name'], 'pct': row['value'], 'seats': int(row['seats']), 'color': row['color']}
            for row in data['rows']]
    # the fingerprint the same election had before the previous one was published
    old = elecciones.result_preview(data['title'], '2023-07-23', rows, None, HOST)['version']
    assert old != data['version']
    status, headers, body = call(houses, EVENT_IMAGE, query='v=' + old)
    assert status == 200 and body == pages.images.result_png(data)
    assert headers['cache-control'] == 'public, max-age=60'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    status, headers, same = call(houses, EVENT_IMAGE, query='v=' + data['version'])
    assert status == 200 and same == body and headers['cache-control'] == 'public, max-age=86400'
    assert call(houses, EVENT_IMAGE)[1]['cache-control'] == 'public, max-age=60'


def check_api_lines():
    """Las líneas de `deploy/check-api.sh` que piden una dirección de las elecciones, troceadas en sus argumentos,
    y el patrón con el que su `check_published` da un 404 por «sin publicar»."""
    with open(CHECK_API, encoding='utf-8') as file:
        script = file.read()

    pattern = re.search(r"""grep -qE '([^']+)' "\$body"; then\n +echo "n/a  \$path sin publicar\"""", script)
    lines = [shlex.split(line) for line in script.splitlines() if re.match(r'check(_published)? "?/elecciones', line)]

    return lines, pattern.group(1)


def check_api_verdicts(site, tmp_path):
    """Lo que `deploy/check-api.sh` diría de cada una de sus líneas de las elecciones contra `site`: `OK`, `n/a`
    (un 404 que su patrón da por «sin publicar», pasado por el `grep` de verdad) o `FAIL` con el código."""
    lines, pattern = check_api_lines()
    verdicts = {}

    for words in lines:
        address, _, query = words[1].partition('?')
        status, _, body = call(site, address, query=query)
        want = words[2]

        if words[0] == 'check_published':
            want = (words[4:5] or ['200'])[0]
            page = tmp_path / 'body'
            page.write_bytes(body)

            if status == 404 and subprocess.run(['grep', '-qE', pattern, str(page)]).returncode == 0:
                verdicts[words[1]] = 'n/a'
                continue

        verdicts[words[1]] = 'OK' if str(status) == want else 'FAIL {}'.format(status)

    return verdicts


def test_check_api_does_not_fail_on_a_package_without_published_elections(api, tmp_path):
    """En un paquete sin elecciones publicadas (el local casi siempre), `deploy/check-api.sh` da por «sin publicar»
    la página de una elección, su imagen y el 400 de una fecha no válida, que sólo se alcanza con elecciones."""
    assert check_api_verdicts(api, tmp_path) == {
        '/elecciones?scope=es': 'OK', '/elecciones?date=abc': 'OK', EVENT_PATH: 'n/a',
        '/elecciones/es/generales/abc.png': 'n/a', EVENT_IMAGE: 'n/a'}


def test_check_api_checks_a_published_election_and_its_image(houses, tmp_path):
    """Con la elección publicada, `deploy/check-api.sh` comprueba su página, su imagen y el 400 de una fecha no
    válida."""
    assert check_api_verdicts(houses, tmp_path) == {
        '/elecciones?scope=es': 'OK', '/elecciones?date=abc': 'OK', EVENT_PATH: 'OK',
        '/elecciones/es/generales/abc.png': 'OK', EVENT_IMAGE: 'OK'}
