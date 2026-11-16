"""Tests de los datos estructurados de las páginas (bloques JSON-LD de schema.org en la cabecera)."""
import json
import os
import re
import struct

from mtpy.lib import pages
from mtpy.lib.pages import structured

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
