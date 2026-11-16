"""Tests de los títulos de las páginas (`<title>`, `og:title`) y de su `<h1>`."""
import os
import re

from mtpy.lib import pages
from tests.test_api_asgi import call
from tests.test_pages import api, houses, houses_without_analysis, html, site_api  # noqa: F401
from tests.test_pages_crawlers import POLL_PATH, SITE, sitemap, write_polls
from tests.test_webapi_unit import ROOT, RUN, clear_queries, fixture_data, use_database, write_study  # noqa: F401

BRAND = ' · Demoscopio'
HOME_TITLE = 'Demoscopio · Promedio de encuestas electorales en España'
# the wordmark of the header: a link home, not a heading
WORDMARK = ('<p class="wordmark"><a href="/"><img src="/dist/img/logo.svg" alt="Demoscopio" width="574" '
            'height="100"></a></p>')
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


def headings_of(page):
    """Los `<h1>` de una página, con sus etiquetas."""
    return re.findall(r'<h1[\s>].*?</h1>', page, re.S)


def test_every_page_has_a_single_h1_that_is_not_the_wordmark(houses, fresh_app):
    """Cada página tiene un solo `<h1>`, el suyo: el logotipo de la cabecera es un enlace a la portada."""
    write_study(fresh_app.fs)

    for path in site_paths(houses):
        page = html(call(houses, path)[2])
        assert len(headings_of(page)) == 1 and page.count('<h1') == 1, path
        assert WORDMARK in page and 'logo.svg' not in headings_of(page)[0], path


def test_hidden_heading_carries_the_title_before_the_event_bar(houses):
    """Una página sin titular visible lleva su título en un `<h1>` que sólo leen los lectores de pantalla y los
    buscadores, tras la cabecera: quien salta a él y sigue leyendo pasa por la barra del evento y por el contenido."""
    page = html(call(houses, '/es')[2])
    assert re.search(r'</header>\s*<h1 class="sr-only">Encuestas elecciones generales 2026: promedio y escaños</h1>'
                     r'\s*<div id="event-bar"', page)
    page = html(call(houses, '/casas')[2])
    assert re.search(r'</header>\s*<h1 class="sr-only">Ranking de encuestadoras: cuáles aciertan más</h1>'
                     r'\s*<main id="main">', page)


def test_pages_with_a_visible_heading_do_not_get_the_hidden_one(houses, fresh_app):
    """La portada, el índice de análisis, los estudios y las páginas del método con titular ya tienen su `<h1>`."""
    write_study(fresh_app.fs)
    study = '<h1 class="study-title">El voto CERA en España y el efecto de la «Ley de Nietos»</h1>'
    visible = {'/': '<h1 class="hero-t">', '/analisis': '<h1>Análisis</h1>', '/analisis/voto-cera': study,
               '/metodo/promedio': '<h1 class="headline">¿Cómo promediamos las encuestas?</h1>'}

    for path, heading in visible.items():
        page = html(call(houses, path)[2])
        assert heading in page and '<h1 class="sr-only">' not in page, path


def test_templates_empty_the_hidden_heading_exactly_when_they_show_one():
    """Una plantilla que extiende `base.html` vacía el bloque `heading` si y sólo si pinta su propio `<h1>`: ni dos
    `<h1>` ni ninguno. Los parciales no llevan `<h1>`."""
    pages_seen = 0

    for name in sorted(os.listdir(pages.TEMPLATES_DIR)):
        with open(os.path.join(pages.TEMPLATES_DIR, name), encoding='utf-8') as f:
            text = re.sub(r'\{#.*?#\}', '', f.read(), flags=re.S)  # the comments may name the tag

        if "{% extends 'base.html' %}" not in text:
            assert ('<h1' in text) == (name in ('base.html', 'error.html')), name
            continue

        pages_seen += 1
        shows_heading = '<h1' in text or "tag='h1'" in text
        assert ('{% block heading %}{% endblock %}' in text) == shows_heading, name

    assert pages_seen >= 17


def test_reserved_characters_are_escaped_once_in_title_og_and_heading(houses, fresh_app):
    """Un nombre con un carácter reservado de HTML llega escapado una sola vez al título, a `og:title` y al `<h1>`."""
    write_polls(fresh_app.fs, [{'pollster': 'Sondeos & Datos'}])
    status, _, body = call(houses, '/es/sondeos/2026-10-05/sondeos-datos')
    page = html(body)
    text = 'Encuesta Sondeos &amp; Datos · 5 oct 2026 · Elecciones generales 2026'
    assert status == 200 and title_of(page) == text + BRAND and og_title_of(page) == text
    assert headings_of(page) == ['<h1 class="sr-only">' + text + '</h1>']


def test_error_page_has_its_status_as_heading(api):
    """La página de error no hereda de `base.html`: lleva su propio `<h1>` oculto y el logotipo sin título."""
    status, _, body = call(api, '/nope')
    page = html(body)
    assert status == 404 and headings_of(page) == ['<h1 class="sr-only">Error 404</h1>'] and WORDMARK in page


def test_header_css_styles_the_wordmark_not_a_heading():
    """El CSS de la cabecera cuelga del logotipo (`.wordmark`), no de un `<h1>` que ya no existe, y le deja la
    tipografía de los títulos."""
    with open(os.path.join(pages.REPO_ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8') as f:
        css = f.read()

    assert '#site-header h1' not in css and css.count('#site-header .wordmark') == 4
    # the alt text of a logo that fails to load keeps the type it had as a heading
    rule = re.search(r'#site-header \.wordmark \{([^}]*)\}', css).group(1)
    assert 'font-family: var(--font-head);' in rule and 'font-weight: 700;' in rule
