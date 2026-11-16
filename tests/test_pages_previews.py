"""Tests de las reglas que comparten las vistas previas de los enlaces (la de un sondeo y la de una elección
pasada): la marca del cambio, la huella de lo que se dibuja y las cifras que caben en una descripción."""
import re

from mtpy.lib import pages
from mtpy.lib.pages import previews
from mtpy.lib.pages.filters import SNIPPET_LENGTH


def test_preview_change():
    """El cambio de un partido se dice con su sentido y sus puntos sin signo; «=» si no cambia y nada si no hay
    con qué comparar."""
    assert previews.preview_change(None) is None
    assert previews.preview_change(0) == {'direction': 'same', 'text': '='}
    assert previews.preview_change(0.4) == {'direction': 'up', 'text': '0,4'}
    assert previews.preview_change(-12.3) == {'direction': 'down', 'text': '12,3'}


def test_preview_version_follows_the_design_and_the_data(monkeypatch):
    """La huella son ocho cifras hexadecimales de lo que se dibuja: la misma para los mismos datos, otra si cambia
    un dato o la versión del dibujo."""
    drawn = {'title': 'Sigma Dos', 'subtitle': 'Elecciones generales 2026', 'more': None,
             'rows': [{'name': 'PP', 'share': '33,1%'}]}
    version = previews.preview_version(drawn)
    assert re.fullmatch(r'[0-9a-f]{8}', version)
    assert previews.preview_version(dict(reversed(list(drawn.items())))) == version
    assert previews.preview_version(drawn | {'title': 'Sigma Dos para El Mundo'}) != version
    assert previews.preview_version(drawn | {'rows': [{'name': 'PP', 'share': '33,2%'}]}) != version
    assert previews.preview_version(drawn | {'more': 'y 1 partido más'}) != version
    monkeypatch.setattr(previews, 'PREVIEW_DESIGN', previews.PREVIEW_DESIGN + 1)
    assert previews.preview_version(drawn) != version


def test_with_figures_keeps_whole_figures_that_fit():
    """Una descripción lleva las cifras que caben enteras, por orden, y acaba en punto; nunca corta una cifra ni
    pasa del tope. Sin cifras, o si no cabe ninguna, se queda en su entrada."""
    assert SNIPPET_LENGTH == 160
    figures = ['PP 33,1%', 'PSOE 31,7%', 'VOX 12,4%']
    whole = 'Sondeo de Sigma Dos (5 oct 2026). PP 33,1%, PSOE 31,7%, VOX 12,4%.'
    assert previews.with_figures('Sondeo de Sigma Dos (5 oct 2026). ', figures) == whole
    # one character short of the last figure: it stays out, whole
    short = previews.with_figures('Sondeo de Sigma Dos (5 oct 2026). ', figures, limit=len(whole) - 1)
    assert short == 'Sondeo de Sigma Dos (5 oct 2026). PP 33,1%, PSOE 31,7%.'
    assert previews.with_figures('Sondeo de Sigma Dos (5 oct 2026). ', figures, limit=len(whole)) == whole
    # not even the first one fits, or there is none: the lead, without its trailing space
    assert previews.with_figures('Sondeo de Sigma Dos (5 oct 2026). ', figures, limit=40) == \
        'Sondeo de Sigma Dos (5 oct 2026).'
    assert previews.with_figures('Sondeo de Sigma Dos (5 oct 2026). ', []) == 'Sondeo de Sigma Dos (5 oct 2026).'
    # the default limit is the one of a description
    many = ['P{} 10,0%'.format(index) for index in range(40)]
    text = previews.with_figures('Elecciones generales 2026: ', many)
    assert len(text) <= SNIPPET_LENGTH < len(text) + len(', P16 10,0%') and text.endswith('%.')
    assert pages.snippet(text) == text


def test_the_package_exports_the_shared_rules():
    """Las constantes de las vistas previas viven en `previews` y la fachada de las páginas las sigue ofreciendo."""
    assert pages.PREVIEW_ROWS == previews.PREVIEW_ROWS == 6
    assert pages.PREVIEW_DESIGN == previews.PREVIEW_DESIGN
    assert pages.previews is previews
