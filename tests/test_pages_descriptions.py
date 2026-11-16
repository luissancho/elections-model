"""Tests de las descripciones de las páginas (`<meta name="description">` y `og:description`) y de su corte."""
import glob
import os
import re
from html import unescape

from markupsafe import Markup

from mtpy.lib import pages
from mtpy.lib.pages.filters import SNIPPET_LENGTH, snippet
from tests.test_api_asgi import call
from tests.test_pages import (api, houses, houses_without_analysis, html, publish_study, site_api,  # noqa: F401
                              write_event)
from tests.test_pages_crawlers import POLL_PATH, write_polls
from tests.test_pages_titles import site_paths
from tests.test_webapi_unit import ROOT, RUN, clear_queries, fixture_data, use_database, write_study  # noqa: F401

# the description of one page of each kind whose text is not the one it had before 2026-11-16
DESCRIPTIONS = {
    '/es': ('Elecciones generales 2026: promedio de las encuestas, escaños previstos por partido y circunscripción y '
            'probabilidad de mayoría absoluta.'),
    POLL_PATH: 'Elecciones generales 2026: sondeo de Sigma Dos (5 oct 2026). Voto por partido y peso en el promedio.',
    '/elecciones': ('Cuánto se equivocaron las encuestas frente al resultado oficial en cada elección general y '
                    'autonómica: su error medio y hacia qué bloque fallaron.'),
    '/elecciones/es/generales/2023-07': ('Generales 2023: el resultado oficial y cuánto se equivocaron frente a él '
                                         'los sondeos, cada encuestadora y el modelo.'),
    '/casas': ('Qué encuestadoras aciertan más: el ranking por rating, cuánto se separa cada una del promedio de '
               'todas (su efecto de casa) y cuáles se ciñen al resto.'),
    '/casas/gad3': ('Perfil de la encuestadora GAD3: su rating, lo que se sabe de cómo trabaja, sus errores en cada '
                    'elección y sus efectos de casa.'),
    # an aggregator has no rating and its polls stay out of the average
    '/casas/agregador': ('Perfil de Agregador: de qué tipo es, lo que se sabe de cómo trabaja y sus errores en cada '
                         'elección. Sus sondeos no entran en el promedio y no tiene rating.'),
    '/metodo': ('El método del modelo: el promedio de encuestas, la simulación de escaños, el rating de las '
                'encuestadoras, cuánto acierta y dónde están sus límites.'),
    '/metodo/promedio': ('Cómo calculamos el promedio de encuestas: qué sondeos entran, cuánto pesa cada uno, cómo '
                         'ajustamos las diferencias entre encuestadoras y cómo lo comprobamos.'),
    '/metodo/escanos': ("Cómo pasamos del promedio de encuestas a los escaños: el sorteo del voto, el reparto por "
                        "circunscripciones con la regla D'Hondt y qué significa cada cifra."),
    '/metodo/rating': ('Cómo calculamos el rating de cada encuestadora: qué error medimos, con qué sondeos lo '
                       'comparamos, su calidad previa y cuánto pesa en el promedio.'),
}


def description_of(page):
    """El texto de la única `<meta name="description">` de una página, sin escapar."""
    found = re.findall(r'<meta name="description" content="([^"]*)">', page)
    assert len(found) == 1
    return unescape(found[0])


def og_description_of(page):
    """El texto de la única `og:description` de una página, sin escapar."""
    found = re.findall(r'<meta property="og:description" content="([^"]*)">', page)
    assert len(found) == 1
    return unescape(found[0])


def test_snippet_keeps_a_text_that_fits():
    """Lo que cabe queda igual, con los espacios y saltos de línea reducidos a uno."""
    assert SNIPPET_LENGTH == 160
    assert snippet('Qué encuestadoras aciertan más.') == 'Qué encuestadoras aciertan más.'
    assert snippet('  Dos   líneas\n  de texto. ') == 'Dos líneas de texto.'
    assert snippet('a' * 160) == 'a' * 160 and snippet('') == ''


def test_snippet_keeps_the_leading_sentences_that_fit():
    """Un texto largo se queda en sus primeras frases completas, mientras quepan."""
    first = 'Los cinco días anteriores a la votación la ley prohíbe publicar sondeos en España.'
    second = 'Se publican fuera, y en las redes circulan como precios de la fruta.'
    third = 'En las 5 generales desde 2015, el consenso de esos sondeos estuvo más cerca del resultado.'
    assert snippet(' '.join((first, second, third))) == first + ' ' + second
    # a sentence may end with a question mark or with a closing quote
    question = '¿Es posible orquestar con éxito un fraude electoral significativo a través del censo exterior?'
    assert snippet(question + ' ' + 'No. ' * 30).startswith(question + ' No.')
    assert snippet('Dijo que «acertó». ' + 'Y sigue. ' * 30, limit=30) == 'Dijo que «acertó». Y sigue.'
    assert snippet('Lo llamó «un error.» ' + 'Y sigue. ' * 30, limit=30) == 'Lo llamó «un error.» Y sigue.'


def test_snippet_cuts_at_a_whole_word_when_no_sentence_fits():
    """Si la primera frase no cabe (o las que caben se quedan en menos de la mitad), corta en una palabra entera,
    sin dejar al final una palabra de tres letras o menos, y añade «…»; nunca pasa del tope."""
    long_sentence = ('Se dice que la gente de izquierdas contesta menos a las encuestas y que los votantes de Vox '
                     'esconden su voto, y que por eso los sondeos infravaloran a unos y a otros en cada elección.')
    cut = snippet(long_sentence)
    assert cut == ('Se dice que la gente de izquierdas contesta menos a las encuestas y que los votantes de Vox '
                   'esconden su voto, y que por eso los sondeos infravaloran a unos…')
    assert len(cut) <= SNIPPET_LENGTH
    # a short first sentence is not enough: the cut goes on into the second
    text = 'Sí. ' + 'Palabra ' * 40
    assert snippet(text).startswith('Sí. Palabra Palabra') and snippet(text).endswith('Palabra…')
    assert len(snippet(text)) <= SNIPPET_LENGTH
    # no space at all: a hard cut that still fits
    assert snippet('x' * 500) == 'x' * 159 + '…'
    assert snippet('uno dos tres cuatro, cinco seis', limit=21) == 'uno dos tres cuatro…'
    # neither a closing quote inside a sentence nor an abbreviation ends it
    assert snippet('A' * 90 + ' «rebaño» Sigue ' + 'palabra ' * 20).endswith('palabra…')
    assert snippet('B' * 90 + ' S.L. para el diario ' + 'palabra ' * 20).endswith('palabra…')


def test_snippet_cuts_escaped_text_without_splitting_an_entity():
    """El texto que llega ya escapado (el bloque de una plantilla) se corta por caracteres, no por entidades, y
    sale como texto llano: la plantilla lo escapa una sola vez."""
    text = Markup('La regla D&#39;Hondt &amp; las barreras. ') + Markup('Y más. ' * 40)
    cut = snippet(text, limit=41)
    assert cut == "La regla D'Hondt & las barreras. Y más." and not isinstance(cut, Markup)
    template = pages.environment().from_string('<meta content="{{ text|snippet(41) }}">')
    assert template.render(text=text) == '<meta content="La regla D&#39;Hondt &amp; las barreras. Y más.">'


def test_every_description_fits_a_snippet_and_matches_og(houses, fresh_app):
    """Todas las páginas tienen descripción, cabe en un resultado de búsqueda y es la misma en `og:description`."""
    write_study(fresh_app.fs)

    for path in site_paths(houses):
        page = html(call(houses, path)[2])
        description = description_of(page)
        assert description and len(description) <= SNIPPET_LENGTH, (path, len(description))
        assert og_description_of(page) == description and 'None' not in description, path


def test_fixed_descriptions_fit_without_the_cut():
    """El texto fijo de una plantilla (sin datos dentro) cabe entero: el corte es para lo que depende de los datos."""
    fixed = {}

    for path in sorted(glob.glob(os.path.join(pages.REPO_ROOT, 'web', 'templates', '*.html'))):
        with open(path, encoding='utf-8') as f:
            block = re.search(r'{% block description %}(.*?){% endblock %}', f.read(), re.S)
        if block and '{' not in block.group(1):
            fixed[os.path.basename(path)] = ' '.join(unescape(block.group(1)).split())

    assert {'index.html', 'analisis.html', 'analisis_voto_cera.html', 'privacidad.html', 'metodo.html',
            'metodo_backtest.html', 'metodo_promedio.html', 'metodo_escanos.html', 'metodo_rating.html',
            'casas.html', 'elecciones.html'} <= set(fixed)
    for name, text in fixed.items():
        assert text and len(text) <= SNIPPET_LENGTH and snippet(text) == text, (name, len(text))


def test_description_that_fits_is_escaped_once(houses, fresh_app):
    """Un nombre con un carácter reservado de HTML llega escapado una sola vez a las dos etiquetas también cuando
    la descripción cabe entera (la rama del corte que usa casi todas las páginas)."""
    write_polls(fresh_app.fs, [{'pollster': 'Sondeos & Datos'}])
    status, _, body = call(houses, '/es/sondeos/2026-10-05/sondeos-datos')
    page = html(body)
    text = ('Elecciones generales 2026: sondeo de Sondeos &amp; Datos (5 oct 2026). Voto por partido y peso en el '
            'promedio.')
    assert status == 200 and '<meta name="description" content="%s">' % text in page
    assert '<meta property="og:description" content="%s">' % text in page


def test_study_subtitle_is_cut_in_both_tags(houses, fresh_app):
    """El subtítulo publicado de un estudio, que no cabe, sale cortado igual en la descripción y en
    `og:description` (la vista previa de un enlace), y la página lo sigue enseñando entero."""
    publish_study(fresh_app.fs, study='sesgo-cis')
    subtitle = fixture_data('sesgo-cis-meta')['subtitle']
    status, _, body = call(houses, '/analisis/sesgo-cis')
    page = html(body)
    assert status == 200 and len(subtitle) > SNIPPET_LENGTH and unescape(page).count(subtitle) == 1
    assert description_of(page) == og_description_of(page) == (
        'Desde el nombramiento de José Félix Tezanos en 2018, el CIS ha dado al bloque de izquierdas más voto del '
        'que obtuvo en 5 de 6 elecciones generales…')


def test_descriptions_say_what_each_page_is(houses, fresh_app):
    """Cada tipo de página se describe con lo que tiene, en un texto que cabe (acaba en punto, no en «…»)."""
    for path, description in DESCRIPTIONS.items():
        status, _, body = call(houses, path)
        assert status == 200 and description_of(html(body)) == description, path
        assert description.endswith('.') and len(description) <= SNIPPET_LENGTH, path


def test_no_description_repeats_or_leads_with_a_menu_label(houses, fresh_app):
    """Ninguna página se queda con la descripción de otra, ni empieza por la etiqueta del menú, que fuera de la web
    no dice nada («Histórico Errores: …»)."""
    write_study(fresh_app.fs)
    seen = {}

    for path in site_paths(houses):
        description = description_of(html(call(houses, path)[2]))
        assert description not in seen, (path, seen.get(description))
        seen[description] = path
        assert not any(description.startswith(entry['label'] + ':') for entry in pages.PAGES), path


def test_election_description_only_names_the_model_when_the_page_has_it(houses, fresh_app):
    """Una elección sin la tarjeta «Lo que dijo el modelo» no dice «y el modelo» en su descripción."""
    write_event(fresh_app.fs, date='2019-11-10', model=False, featured=False)
    status, _, body = call(houses, '/elecciones/es/generales/2019-11')
    page = html(body)
    assert status == 200 and 'Lo que dijo el modelo' not in page
    description = description_of(page)
    assert description.endswith(': el resultado oficial y cuánto se equivocaron frente a él los sondeos y cada '
                                'encuestadora.')
    page = html(call(houses, '/elecciones/es/generales/2023-07')[2])
    assert 'Lo que dijo el modelo' in page
    assert description_of(page) == DESCRIPTIONS['/elecciones/es/generales/2023-07']


def test_election_without_a_name_is_described_by_its_date():
    """Sin nombre, la elección se describe por su fecha (bloque de la plantilla)."""
    template = pages.environment().get_template('eleccion.html')
    context = template.new_context({'event': {'name': None, 'event_date': '2023-07-23'}, 'horizon_tables': []})
    assert unescape(''.join(template.blocks['description'](context))) == (
        'Elección del 23 jul 2023: el resultado oficial y cuánto se equivocaron frente a él los sondeos y cada '
        'encuestadora.')


def poll_description(**context):
    """El bloque `description` de la página de un sondeo con ese contexto, sin escapar y sin cortar."""
    template = pages.environment().get_template('sondeo.html')
    return unescape(''.join(template.blocks['description'](template.new_context(context))))


def test_poll_description_says_what_the_page_holds_and_fits():
    """La descripción de un sondeo dice de quién es y de cuándo, y lo que la página tiene: escaños sólo si los
    publicó y el cambio sólo si hay una entrega anterior. El de las generales con el nombre más largo cabe entero."""
    event_bar = {'election': 'Elecciones generales 2026'}
    lead = 'Elecciones generales 2026: sondeo de Sigma Dos para El Mundo (5 oct 2026). '
    texts = {
        (True, True): lead + 'Voto y escaños, cambio frente al anterior y peso en el promedio.',
        (False, True): lead + 'Voto, cambio frente al anterior y peso en el promedio.',
        (True, False): lead + 'Voto y escaños por partido y peso en el promedio.',
        (False, False): lead + 'Voto por partido y peso en el promedio.',
    }
    for (has_seats, has_previous), text in texts.items():
        assert poll_description(event_bar=event_bar, poll_title='Sigma Dos para El Mundo', poll_date='5 oct 2026',
                                has_seats=has_seats, previous={'title': 'x'} if has_previous else None) == text
        assert snippet(text) == text
    # the longest house and outlet of the general election (32 characters) with the longest date: still whole
    longest = poll_description(event_bar=event_bar, poll_title='Ateneo del Dato para elDiario.es',
                               poll_date='28 sept 2026', has_seats=True, previous={'title': 'x'})
    assert snippet(longest) == longest and longest.endswith('y peso en el promedio.')
    # so does the usual poll of a regional election
    regional = poll_description(event_bar={'election': 'Elecciones autonómicas Madrid 2027'}, has_seats=True,
                                poll_title='Sigma Dos para El Mundo', poll_date='28 sept 2026', previous={'title': 'x'})
    assert snippet(regional) == regional


def test_long_poll_description_ends_at_its_first_sentence(houses, fresh_app):
    """Un sondeo con una encuestadora y un medio de nombre muy largo no cabe: la descripción se queda en su primera
    frase, la que dice de quién es y de cuándo, igual en `og:description`."""
    change = {'pollster': 'Instituto de Estudios Demoscópicos y de Opinión',
              'sponsor': 'El Periódico de la Mañana y Radio Norte'}
    write_polls(fresh_app.fs, [change])
    path = pages.poll_detail_href('es', fixture_data('polls')['polls'][0] | change)
    status, _, body = call(houses, path)
    page = html(body)
    assert status == 200 and description_of(page) == og_description_of(page) == (
        'Elecciones generales 2026: sondeo de Instituto de Estudios Demoscópicos y de Opinión para El Periódico de '
        'la Mañana y Radio Norte (5 oct 2026).')


def test_event_description_fits_for_the_longest_scope_name():
    """La descripción del evento cabe entera también en el ámbito autonómico de nombre más largo."""
    scopes = [{'code': 'es-vc', 'name': 'Comunidad Valenciana'}]
    state = {'scope': 'es-vc', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    info = pages.event_info('es-vc', {'event_date': '2027-05-23'}, state, scopes=scopes)
    assert info['election'] == 'Elecciones autonómicas Comunidad Valenciana 2027'
    undated = pages.event_info('es-vc', {}, state, scopes=scopes)
    assert undated['election'] == 'Elecciones autonómicas Comunidad Valenciana'
    template = pages.environment().get_template('promedios.html')
    text = unescape(''.join(template.blocks['description'](template.new_context({'event_bar': info}))))
    assert text == ('Elecciones autonómicas Comunidad Valenciana 2027: promedio de las encuestas, escaños previstos '
                    'por partido y circunscripción y probabilidad de mayoría absoluta.')
    assert snippet(text) == text
