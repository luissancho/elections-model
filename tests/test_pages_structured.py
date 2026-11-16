"""Tests de los datos estructurados de las páginas (bloques JSON-LD de schema.org en la cabecera)."""
import json
import os
import re
import struct

from mtpy.lib import bundle, pages, publish, webapi
from mtpy.lib.pages import structured
from tests.test_api_asgi import call
from tests.test_pages import (api, houses, houses_without_analysis, html, publish_study, site_api,  # noqa: F401
                              write_event)
from tests.test_pages_crawlers import POLL_PATH, canonical_of, write_polls
from tests.test_pages_descriptions import description_of
from tests.test_pages_titles import site_paths
from tests.test_webapi_unit import ROOT, RUN, clear_queries, fixture_data, use_database, write_study  # noqa: F401

ORIGIN = 'https://demoscopio.es'
CONTEXT = 'https://schema.org'
TAGLINE = 'Promedio de encuestas, previsión de escaños y ranking de encuestadoras'
LOGO = ORIGIN + '/dist/img/apple-touch-icon.png'
PERSON = {'@type': 'Person', 'name': 'Luis Sancho', 'url': 'https://x.com/luis_sancho'}
PUBLISHER = {'@type': 'Organization', 'name': 'Demoscopio', 'url': ORIGIN + '/', 'logo': LOGO}


def blocks_of(page):
    """Los bloques JSON-LD de una página, ya leídos, en su orden."""
    return [json.loads(text) for text in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page, re.S)]


def test_website_and_organization_blocks():
    """La portada dice cómo se llama el sitio y quién lo publica, con el lema y el autor que ya enseña la web."""
    assert structured.website(ORIGIN) == {
        '@context': CONTEXT, '@type': 'WebSite', 'name': 'Demoscopio', 'url': ORIGIN + '/', 'inLanguage': 'es',
        'description': TAGLINE}
    assert structured.organization(ORIGIN) == {
        '@context': CONTEXT, '@type': 'Organization', 'name': 'Demoscopio', 'url': ORIGIN + '/', 'logo': LOGO,
        'description': TAGLINE, 'founder': PERSON}
    assert structured.person() == PERSON
    # the logo is a file of the site, a PNG of at least 112x112 (what Google asks of a logo)
    with open(os.path.join(pages.REPO_ROOT, 'web', structured.LOGO_PATH.lstrip('/')), 'rb') as f:
        header = f.read(24)
    assert header[:8] == b'\x89PNG\r\n\x1a\n' and min(struct.unpack('>II', header[16:24])) >= 112


def test_article_block_keeps_only_what_it_is_given():
    """Un artículo lleva su titular, su dirección, su autor y quién lo publica; la descripción y la fecha, sólo si
    las hay."""
    full = structured.article(ORIGIN, '/analisis/sesgo-cis', 'El sesgo del CIS de Tezanos',
                              'Desde 2018, el CIS ha dado al bloque de izquierdas más voto del que obtuvo.',
                              '2026-11-06T10:00:00Z')
    assert full == {
        '@context': CONTEXT, '@type': 'Article', 'headline': 'El sesgo del CIS de Tezanos',
        'description': 'Desde 2018, el CIS ha dado al bloque de izquierdas más voto del que obtuvo.',
        'url': ORIGIN + '/analisis/sesgo-cis', 'mainEntityOfPage': ORIGIN + '/analisis/sesgo-cis', 'inLanguage': 'es',
        'dateModified': '2026-11-06T10:00:00Z', 'author': PERSON, 'publisher': PUBLISHER}
    bare = structured.article(ORIGIN, '/analisis/rebano', 'Rebaño', None, None)
    assert 'description' not in bare and 'dateModified' not in bare and bare['headline'] == 'Rebaño'
    assert structured.article(ORIGIN, '/analisis/rebano', 'Rebaño', '', '') == bare


def test_breadcrumbs_number_the_trail_and_leave_the_page_without_item():
    """Las migas numeran desde 1 y sólo enlazan lo que tiene dirección: la página misma va sin ella."""
    assert structured.breadcrumbs(ORIGIN, [('Análisis', '/analisis'), ('El sesgo del CIS de Tezanos', None)]) == {
        '@context': CONTEXT, '@type': 'BreadcrumbList', 'itemListElement': [
            {'@type': 'ListItem', 'position': 1, 'name': 'Análisis', 'item': ORIGIN + '/analisis'},
            {'@type': 'ListItem', 'position': 2, 'name': 'El sesgo del CIS de Tezanos'}]}


def test_block_text_reads_a_template_block():
    """El texto de un bloque de una plantilla, como lo vería el lector: sin escapar y en una línea."""
    assert structured.block_text('metodo_promedio.html', 'title', {}) == '¿Cómo promediamos las encuestas?'
    fallback = structured.block_text('analisis_sesgo_cis.html', 'title', {'study_meta': {}})
    assert fallback == 'El sesgo del CIS de Tezanos'
    context = {'study_meta': {'title': 'A & B\n «c» <d>'}}
    assert structured.block_text('analisis_sesgo_cis.html', 'title', context) == 'A & B «c» <d>'


def test_block_survives_reserved_characters():
    """Un nombre con `</script>`, comillas y `&` no cierra la etiqueta ni rompe el JSON: se lee de vuelta tal cual."""
    name = '</script><script>alert(1)</script> "A" & \'B\' <!-- c'
    template = pages.environment().from_string(
        '{% for item in structured %}<script type="application/ld+json">{{ item|tojson }}</script>{% endfor %}')
    page = template.render(structured=[structured.breadcrumbs(ORIGIN, [('Análisis', '/analisis'), (name, None)])])
    assert page.count('<script') == 1 and page.count('</script>') == 1 and 'alert(1)</script>' not in page
    assert '<!--' not in page
    assert blocks_of(page)[0]['itemListElement'][1]['name'] == name


def expected_types(path):
    """Los tipos de bloque que lleva la página de una ruta, en su orden (la tabla de la spec)."""
    if path == '/':
        return ['WebSite', 'Organization']
    if path.startswith('/analisis/'):
        return ['Article', 'BreadcrumbList']
    if path.startswith(('/metodo/', '/casas/', '/elecciones/')) or '/sondeos/' in path:
        return ['BreadcrumbList']
    return []


def trail_of(page):
    """`[(nombre, dirección o None), ...]` de las migas de una página."""
    trails = [block for block in blocks_of(page) if block['@type'] == 'BreadcrumbList']
    assert len(trails) == 1
    items = trails[0]['itemListElement']
    assert [item['position'] for item in items] == list(range(1, len(items) + 1))
    return [(item['name'], item.get('item')) for item in items]


def page_of(api, path, query=''):
    """El HTML de una página que responde 200."""
    status, _, body = call(api, path, query=query)
    assert status == 200, (path, query, status)
    return html(body)


def stamp_study(fs, study, published_at):
    """Apunta en el manifiesto la última publicación de un estudio, como hace `publish`."""
    entry = {'published_at': published_at, 'registered_at': None, 'checked': False}
    publish.update_manifest(bundle.BundleWriter(fs), studies={study: entry})
    webapi.site().manifest_cache_clear()


def test_every_page_carries_the_blocks_of_its_kind(houses, fresh_app):
    """Cada página lleva los bloques de su tipo y ninguno más, todos JSON válido de schema.org."""
    write_study(fresh_app.fs)
    seen = set()

    for path in site_paths(houses):
        blocks = blocks_of(page_of(houses, path))
        assert [block['@type'] for block in blocks] == expected_types(path), path
        assert all(block['@context'] == CONTEXT for block in blocks), path
        seen.update(block['@type'] for block in blocks)

    # the fixture has a page of every kind: the home, a study, method pages, profiles, an election and a poll
    assert seen == {'WebSite', 'Organization', 'Article', 'BreadcrumbList'}
    assert {path.split('/')[1] for path in site_paths(houses) if expected_types(path) == ['BreadcrumbList']} >= {
        'metodo', 'casas', 'elecciones', 'es'}


def test_home_names_the_site_and_its_publisher(houses, fresh_app):
    """La portada, y sólo ella, dice cómo se llama el sitio y quién lo publica."""
    assert blocks_of(page_of(houses, '/')) == [structured.website(ORIGIN), structured.organization(ORIGIN)]
    assert blocks_of(page_of(houses, '/es')) == [] and blocks_of(page_of(houses, '/casas')) == []


def test_study_is_an_article_with_what_the_page_shows(houses, fresh_app):
    """El artículo de un estudio repite lo que la página enseña: su titular, su descripción, su dirección, la fecha
    de «Actualizado el» y el autor del pie."""
    publish_study(fresh_app.fs, study='sesgo-cis')
    stamp_study(fresh_app.fs, 'sesgo-cis', '2026-11-06T10:00:00Z')
    page = page_of(houses, '/analisis/sesgo-cis', query='scope=es&utm_source=x')
    article = blocks_of(page)[0]
    headline = re.sub(r'<[^>]+>', '', re.search(r'<h1[^>]*>(.*?)</h1>', page, re.S).group(1)).strip()
    assert article['@type'] == 'Article' and article['headline'] == headline == 'El sesgo del CIS de Tezanos'
    assert article['description'] == description_of(page) and article['description'].endswith('…')
    assert article['url'] == article['mainEntityOfPage'] == ORIGIN + '/analisis/sesgo-cis'
    assert (article['url'],) * 2 == canonical_of(houses, '/analisis/sesgo-cis')
    assert article['dateModified'] == '2026-11-06T10:00:00Z' and 'Actualizado el 6 nov 2026' in page
    assert article['author'] == PERSON and article['publisher'] == PUBLISHER and article['inLanguage'] == 'es'
    assert 'Modelo y curación: Luis Sancho' in page  # the footer names the author the block names
    assert trail_of(page) == [('Análisis', ORIGIN + '/analisis'), ('El sesgo del CIS de Tezanos', None)]


def test_study_without_a_publication_date_has_no_date(houses, fresh_app):
    """Sin la fecha de publicación en el manifiesto, el artículo va sin fecha; el titular es el de la página aunque
    el paquete traiga otro título (el estudio del voto CERA)."""
    write_study(fresh_app.fs)
    page = page_of(houses, '/analisis/voto-cera')
    article = blocks_of(page)[0]
    assert 'dateModified' not in article and 'datePublished' not in article
    assert article['headline'] == 'El voto CERA en España y el efecto de la «Ley de Nietos»'
    assert article['description'] == description_of(page)


def test_study_date_is_the_one_of_the_manifest_even_when_the_page_does_not_show_it(houses, fresh_app):
    """La fecha del artículo es la de la última publicación del estudio, también en el que no la enseña (el del
    voto CERA no lleva «Actualizado el»)."""
    write_study(fresh_app.fs)
    stamp_study(fresh_app.fs, 'voto-cera', '2026-11-02T09:30:00Z')
    page = page_of(houses, '/analisis/voto-cera')
    assert blocks_of(page)[0]['dateModified'] == '2026-11-02T09:30:00Z' and '2 nov 2026' not in page


def test_every_study_and_method_page_has_the_blocks_its_structured_data_quotes():
    """El controlador cita el bloque `title` (y, en un estudio, `description`) de la plantilla de la página: toda
    plantilla de un estudio o de una subpágina del método los define, también las que el fixture no publica."""
    quoted = [(entry['template'], ('title', 'description')) for entry in pages.STUDIES.values()]
    quoted += [(entry['template'], ('title',)) for entry in pages.METODO_SECTIONS.values()]
    assert len(quoted) == 9

    for template, names in quoted:
        blocks = pages.environment().get_template(template).blocks
        assert set(names) <= set(blocks), template


def test_breadcrumbs_of_each_kind_of_page(houses, fresh_app):
    """Las migas de una página de segundo nivel: su sección, con la etiqueta del menú y su dirección, y la
    página."""
    assert trail_of(page_of(houses, '/metodo/promedio')) == [
        ('Método', ORIGIN + '/metodo'), ('¿Cómo promediamos las encuestas?', None)]
    assert trail_of(page_of(houses, '/metodo/backtest')) == [
        ('Método', ORIGIN + '/metodo'), ('Cuánto acierta el modelo', None)]
    assert trail_of(page_of(houses, '/casas/gad3')) == [('Ranking Encuestadoras', ORIGIN + '/casas'), ('GAD3', None)]
    assert trail_of(page_of(houses, '/elecciones/es/generales/2023-07')) == [
        ('Histórico Errores', ORIGIN + '/elecciones'), ('Generales 2023', None)]
    assert trail_of(page_of(houses, POLL_PATH)) == [
        ('Elecciones generales 2026', ORIGIN + '/es'), ('Sigma Dos · 5 oct 2026', None)]
    # an election without a name is named by its date, like its title
    write_event(fresh_app.fs, date='2019-11-10', model=False, featured=False)
    schema = bundle.EVENT_SCHEMAS['meta']
    meta = fixture_data(schema) | {'scope': 'es', 'event_date': '2019-11-10', 'has_model': False, 'featured': False,
                                   'name': None}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_event('es', '2019-11-10', 'meta'), schema, meta, 'es')
    webapi.site().manifest_cache_clear()
    assert trail_of(page_of(houses, '/elecciones/es/generales/2019-11')) == [
        ('Histórico Errores', ORIGIN + '/elecciones'), ('10 nov 2019', None)]


def test_blocks_follow_the_configured_origin(houses, fresh_app):
    """Las direcciones de los bloques salen de `web.site_url`, sin barra doble aunque venga con barra final."""
    fresh_app.config.offset_set('web', {'site_url': 'https://demoscopio.dev/'})  # the database stays
    origin = 'https://demoscopio.dev'
    assert blocks_of(page_of(houses, '/')) == [structured.website(origin), structured.organization(origin)]
    assert trail_of(page_of(houses, '/casas/gad3'))[0] == ('Ranking Encuestadoras', origin + '/casas')
    write_study(fresh_app.fs)
    for path in ('/', '/casas/gad3', '/analisis/voto-cera', POLL_PATH):
        text = json.dumps(blocks_of(page_of(houses, path)))
        assert 'demoscopio.es' not in text and 'dev//' not in text, path


def test_blocks_carry_canonical_urls_whatever_the_request(houses, fresh_app):
    """Una página pedida por un identificador, con la fecha completa o con parámetros lleva en sus bloques las
    mismas direcciones que pedida por su dirección limpia."""
    query = 'scope=es&utm_source=x&fbclid=abc'
    for odd, clean in (('/casas/2', '/casas/gad3'), ('/es/sondeos/2026-10-05/7', POLL_PATH),
                       ('/elecciones/es/generales/2023-07-23', '/elecciones/es/generales/2023-07'),
                       ('/metodo/rating', '/metodo/rating')):
        blocks = blocks_of(page_of(houses, odd, query=query))
        assert blocks and blocks == blocks_of(page_of(houses, clean)), odd
        assert 'utm_source' not in json.dumps(blocks) and 'scope=' not in json.dumps(blocks), odd


def test_hostile_name_reaches_the_block_of_a_page_intact(houses, fresh_app):
    """El nombre de una casa con `</script>`, comillas y `&` llega a las migas de su sondeo tal cual, sin cerrar la
    etiqueta ni dejar HTML suelto en la página."""
    name = 'Datos </script><script>alert(1)</script> "A" & B'
    change = {'pollster': name}
    write_polls(fresh_app.fs, [change])
    page = page_of(houses, pages.poll_detail_href('es', fixture_data('polls')['polls'][0] | change))
    assert trail_of(page)[1] == (name + ' · 5 oct 2026', None)
    assert '<script>alert(1)' not in page and page.count('application/ld+json') == 1
