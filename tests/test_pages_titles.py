"""Tests de los títulos de las páginas (`<title>`, `og:title`) y de su `<h1>`."""
import re

from mtpy.lib import pages
from tests.test_api_asgi import call
from tests.test_pages import api, houses, houses_without_analysis, html, site_api  # noqa: F401
from tests.test_pages_crawlers import POLL_PATH, SITE, sitemap
from tests.test_webapi_unit import ROOT, RUN, clear_queries, fixture_data, use_database, write_study  # noqa: F401

BRAND = ' · Demoscopio'
HOME_TITLE = 'Demoscopio · Promedio de encuestas electorales en España'
# the full `<title>` of one page of each kind of the fixture site
TITLES = {
    '/': HOME_TITLE,
    '/es': 'Encuestas elecciones generales 2026: promedio y escaños' + BRAND,
    POLL_PATH: 'Encuesta Sigma Dos · 5 oct 2026 · Elecciones generales 2026' + BRAND,
    '/elecciones': 'Errores de las encuestas en las elecciones pasadas' + BRAND,
    '/elecciones/es/generales/2023-07': 'Generales 2023: resultado y errores de las encuestas' + BRAND,
    '/casas': 'Ranking de encuestadoras: cuáles aciertan más' + BRAND,
    '/casas/gad3': 'GAD3: rating e historial de la encuestadora' + BRAND,
    '/casas/agregador': 'Agregador: historial de la encuestadora' + BRAND,  # an aggregator has no rating
    '/metodo': 'Método: cómo funciona el modelo' + BRAND,
    '/metodo/promedio': '¿Cómo promediamos las encuestas?' + BRAND,
    '/metodo/escanos': '¿Cómo simulamos los escaños?' + BRAND,
    '/metodo/rating': '¿Cómo calculamos el rating de las encuestadoras?' + BRAND,
    '/metodo/backtest': 'Cuánto acierta el modelo' + BRAND,
    '/analisis': 'Análisis electorales con datos oficiales' + BRAND,
    '/analisis/voto-cera': 'El voto CERA en España y el efecto de la «Ley de Nietos»' + BRAND,
    '/privacidad': 'Privacidad y cookies' + BRAND,
}


def title_of(page):
    """El texto del único `<title>` de una página."""
    titles = re.findall(r'<title>(.*?)</title>', page, re.S)
    assert len(titles) == 1
    return titles[0]


def og_title_of(page):
    """El contenido de la única etiqueta `og:title` de una página."""
    titles = re.findall(r'<meta property="og:title" content="([^"]*)">', page)
    assert len(titles) == 1
    return titles[0]


def site_paths(api):
    """Las rutas de todas las páginas del sitio del fixture: las del sitemap y la de privacidad, que no entra."""
    return [loc[len(SITE):] for loc in sitemap(api)] + ['/privacidad']


def test_titles_say_what_each_page_is(houses, fresh_app):
    """Cada tipo de página se titula con lo que es, y la marca va al final; en la portada, al principio."""
    write_study(fresh_app.fs)

    for path, title in TITLES.items():
        status, _, body = call(houses, path)
        assert status == 200 and title_of(html(body)) == title, path


def test_every_title_ends_with_the_brand_and_og_title_drops_it(houses, fresh_app):
    """Ninguna página se queda con el título de otra ni empieza por la marca (salvo la portada), y `og:title` es
    el título sin la marca: la vista previa de un enlace ya enseña el dominio y `og:site_name`."""
    write_study(fresh_app.fs)
    paths = site_paths(houses)
    titles = {}

    for path in paths:
        status, _, body = call(houses, path)
        page = html(body)
        assert status == 200, path
        titles[path] = title = title_of(page)

        if path == '/':
            assert title == HOME_TITLE and og_title_of(page) == HOME_TITLE
        else:
            assert title.endswith(BRAND) and not title.startswith('Demoscopio'), path
            assert og_title_of(page) == title[:-len(BRAND)], path

    assert set(TITLES) <= set(paths)
    assert len(set(titles.values())) == len(titles)


def test_event_phrase_names_the_election_in_running_text():
    """El evento en texto corrido, como se busca: sin mayúscula inicial ni el punto medio de la barra del evento."""
    scopes = [{'code': 'es', 'name': 'España'}, {'code': 'es-md', 'name': 'Madrid'}]
    assert pages.event_phrase('es', scopes) == 'elecciones generales'
    assert pages.event_phrase('es-md', scopes) == 'elecciones autonómicas Madrid'
    assert pages.event_phrase('es-xx', scopes) == 'elecciones autonómicas es-xx'
    state = {'scope': 'es-md', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    assert pages.event_info('es-md', {}, state, scopes=scopes)['phrase'] == 'elecciones autonómicas Madrid'


def test_event_title_without_a_year_does_not_print_none():
    """Sin año del evento (la fecha de la elección no se pudo leer), el título va sin él, nunca con «None». El
    paquete exige la fecha, así que se comprueba el bloque de la plantilla."""
    for name, expected in (('promedios.html', 'Encuestas elecciones generales: promedio y escaños'),
                           ('sondeo.html', 'Encuesta GAD3 · 5 oct 2026 · Elecciones generales')):
        template = pages.environment().get_template(name)
        context = template.new_context({'event_bar': {'phrase': 'elecciones generales', 'year': None},
                                        'poll_title': 'GAD3', 'poll_date': '5 oct 2026'})
        assert ''.join(template.blocks['title'](context)) == expected


def test_error_page_title_ends_with_the_brand(api):
    """La página de error también dice primero qué es."""
    status, _, body = call(api, '/nope')
    assert status == 404 and title_of(html(body)) == 'Error 404' + BRAND
