"""Tests de la imagen propia del enlace de una elección pasada: lo que dice (como datos, con el cambio frente a la
elección anterior), su dibujo (el de la tarjeta de un sondeo) y su dirección."""
import copy
import io
import re

import pytest
from PIL import Image

from mtpy.core.api import HttpError
from mtpy.lib import bundle, pages
from mtpy.lib.pages import elecciones
from mtpy.lib.pages.filters import SNIPPET_LENGTH
from tests.test_pages import api, houses, houses_without_analysis, site_api  # noqa: F401
from tests.test_webapi_unit import fixture_data, write_event

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
