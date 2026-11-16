"""La medición de visitas con consentimiento (Google Tag Manager, `WEB_GTM_ID`) y la página `/privacidad`: sin
contenedor configurado las páginas no llevan nada de la medición; con él, el aviso oculto y el módulo que lo gobierna,
sin scripts en línea ni nada de Google en el HTML; y la política de seguridad de nginx admite a Google sólo donde la
medición lo necesita."""
import os
import re
import warnings

import pytest

from mtpy.core.app import Config
from mtpy.lib import pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, html, visible_text  # noqa: F401  (fixtures y ayudas de las páginas)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GTM_ID = 'GTM-TEST123'
MODULE = '<script type="module" src="/dist/js/pages/consent.js"></script>'
NOTICE = ('<section id="consent" class="consent" aria-label="Aviso de cookies" data-gtm-id="GTM-TEST123" '
          'data-months="12" hidden>')
REJECT = '<button type="button" data-consent="denied">Rechazar</button>'
ACCEPT = '<button type="button" data-consent="granted">Aceptar</button>'
FOOTER_LINK = '<a href="/privacidad">Privacidad y cookies</a>'


def use_container(app, gtm_id=GTM_ID):
    """Configura el contenedor de Tag Manager (`web.gtm_id`) en `app`."""
    app.set('config', Config({'web': {'gtm_id': gtm_id}}))


def test_gtm_id_is_empty_by_default_and_only_a_container_id_is_taken(fresh_app):
    """`web.gtm_id` (`WEB_GTM_ID`) sale vacío sin configurar; un identificador de contenedor se toma sin los espacios
    de alrededor, y cualquier otra cosa (un ID de Analytics, minúsculas, HTML) se descarta con un aviso."""
    assert webapi.settings()['gtm_id'] == '' and pages.analytics_config() is None
    use_container(fresh_app, ' GTM-KFJ2W98T ')
    assert webapi.settings()['gtm_id'] == 'GTM-KFJ2W98T'
    assert pages.analytics_config() == {'gtm_id': 'GTM-KFJ2W98T', 'months': 12} and pages.CONSENT_MONTHS == 12
    for bad in ('G-ABC123', 'gtm-kfj2w98t', 'GTM-', '"><script>', 'GTM-ABC DEF'):
        use_container(fresh_app, bad)
        with pytest.warns(UserWarning, match='invalid web.gtm_id'):
            assert webapi.settings()['gtm_id'] == ''
    use_container(fresh_app, '')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert webapi.settings()['gtm_id'] == '' and pages.analytics_config() is None


def test_pages_carry_nothing_of_the_measurement_without_a_container(api):
    """Sin `WEB_GTM_ID` (local, tests) las páginas no llevan el aviso, ni su módulo, ni botones de elección; el pie
    enlaza la página de privacidad igualmente."""
    for path in ('/', '/es', '/metodo', '/analisis', '/privacidad'):
        status, _, body = call(api, path)
        page = html(body)
        assert status == 200, path
        assert 'id="consent"' not in page and MODULE not in page and 'data-consent' not in page, path
        assert 'googletagmanager' not in page, path
        assert FOOTER_LINK in page[page.index('<footer'):page.index('</footer>')], path


def test_pages_with_a_container_carry_the_hidden_notice_and_its_module(api, fresh_app):
    """Con el contenedor configurado toda página lleva el aviso, oculto hasta que el módulo decide enseñarlo, con el
    identificador y los meses de validez como datos, los dos botones con el mismo peso (rechazar es tan fácil como
    aceptar) y el enlace a la página de privacidad; el módulo va el primero de los scripts. Nada de Google está en el
    HTML (lo carga el módulo, y sólo tras aceptar) y no hay scripts en línea, que la política de seguridad bloquea."""
    use_container(fresh_app)
    for path in ('/', '/es', '/metodo', '/analisis', '/privacidad'):
        status, _, body = call(api, path)
        page = html(body)
        assert status == 200 and page.count(MODULE) == 1 and page.count('id="consent"') == 1, path
        assert page.index(MODULE) < page.index('/dist/vendor/echarts-5.6.0.min.js') < page.index('</head>'), path
        notice = page[page.index(NOTICE):]
        notice = notice[:notice.index('</section>')]
        assert page.index(NOTICE) < page.index('id="site-header"'), path  # the first thing a screen reader meets
        assert re.sub(r'\s+', ' ', visible_text(notice)).strip() == (
            'Usamos cookies de Google Analytics para medir las visitas: cuánta gente nos lee y qué páginas se ven. '
            'Sólo se activan si aceptas. Más información Rechazar Aceptar'), path
        assert '<a href="/privacidad">Más información</a>' in notice, path
        assert notice.index(REJECT) < notice.index(ACCEPT) and notice.count('<button') == 2, path
        assert '<script>' not in page and 'onclick=' not in page and 'googletagmanager' not in page, path
        assert_public_wording(page, path)


def test_event_page_adds_one_script_with_a_container(api, fresh_app):
    """La página del evento pasa de tres scripts a cuatro: el módulo del aviso, ECharts, el módulo de la página y el
    bloque JSON."""
    assert html(call(api, '/es')[2]).count('<script') == 3
    use_container(fresh_app)
    assert html(call(api, '/es')[2]).count('<script') == 4


def test_error_page_has_no_notice(api, fresh_app):
    """La página de error no mide: ni aviso ni módulo."""
    use_container(fresh_app)
    status, _, body = call(api, '/nope')
    assert status == 404 and 'consent' not in html(body)


def test_privacy_page_tells_what_is_measured_and_how_to_choose(api, fresh_app):
    """`/privacidad` cuenta quién responde, qué se guarda, las cookies de Google Analytics con su duración, los meses
    que vale la elección y los derechos del lector; no enciende ninguna entrada de la navegación y su canónica es la
    ruta limpia. Sin contenedor avisa de que la medición no está activa; con él ofrece los dos botones y el estado de
    la elección, que el módulo enseña."""
    status, headers, body = call(api, '/privacidad')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Demoscopio · Privacidad y cookies</title>' in page
    assert '<link rel="canonical" href="https://demoscopio.es/privacidad">' in page
    assert 'aria-current="page"' not in page
    for anchor in ('responsable', 'datos', 'cookies', 'eleccion', 'derechos'):
        assert '<section id="{}" class="card">'.format(anchor) in page, anchor
    text = re.sub(r'\s+', ' ', visible_text(page))
    for piece in ('Luis Sancho', 'Google Analytics', 'Google Ireland Limited', '_ga', '_ga_', '2 años', '12 meses',
                  'Agencia Española de Protección de Datos', 'No hay registro, formularios ni publicidad'):
        assert piece in text, piece
    assert '<a href="https://x.com/luis_sancho" rel="me">@luis_sancho</a>' in page
    choice = page[page.index('<section id="eleccion"'):]
    choice = choice[:choice.index('</section>')]
    assert 'La medición de visitas no está activa en este servidor.' in choice and 'data-consent' not in choice
    assert_public_wording(page, '/privacidad')

    use_container(fresh_app)
    page = html(call(api, '/privacidad')[2])
    choice = page[page.index('<section id="eleccion"'):]
    choice = choice[:choice.index('</section>')]
    assert REJECT in choice and ACCEPT in choice and 'no está activa' not in choice
    states = dict(re.findall(r'<span data-consent-when="(\w+)" hidden>([^<]+)</span>', choice))
    assert states == {'granted': 'Ahora mismo: has aceptado la medición.',
                      'denied': 'Ahora mismo: has rechazado la medición.',
                      'unset': 'Ahora mismo: todavía no has elegido.'}
    assert_public_wording(page, '/privacidad')


def test_privacy_route_goes_before_the_scope_route(api):
    """`/privacidad` es una ruta fija: no cae en `/{scope}` (que daría un 404 de ámbito desconocido), y admite el
    ámbito en la consulta como las demás páginas globales."""
    routes = [route[0] for route in pages.ROUTES]
    assert routes.index('/privacidad') < routes.index('/{scope}')
    assert call(api, '/privacidad', query='scope=es')[0] == 200


def csp_of(conf):
    """Las políticas `Content-Security-Policy` de la configuración de nginx, cada una como {directiva: [fuentes]}."""
    policies = re.findall(r'add_header Content-Security-Policy "([^"]+)" always;', conf)
    return [{part.split()[0]: part.split()[1:] for part in policy.split(';') if part.strip()} for policy in policies]


def test_nginx_csp_admits_google_only_where_the_measurement_needs_it():
    """La política de seguridad es la misma en los tres sitios donde nginx la envía y abre a Google lo justo: los
    scripts de Tag Manager (y su vista previa), las imágenes y las conexiones de Analytics sin funciones publicitarias
    y los estilos y fuentes de la vista previa. Los scripts en línea y `eval` siguen bloqueados, así que en el
    contenedor no funcionan las etiquetas de HTML personalizado ni las variables de JavaScript personalizado."""
    with open(os.path.join(ROOT, 'deploy', 'docker', 'nginx.conf'), encoding='utf-8') as file:
        policies = csp_of(file.read())
    assert len(policies) == 3 and policies[0] == policies[1] == policies[2]
    csp = policies[0]
    assert csp == {
        'default-src': ["'self'"],
        'script-src': ["'self'", 'https://www.googletagmanager.com', 'https://tagmanager.google.com'],
        'img-src': ["'self'", 'data:', 'https://www.googletagmanager.com', 'https://*.google-analytics.com',
                    'https://ssl.gstatic.com', 'https://www.gstatic.com'],
        'style-src': ["'self'", "'unsafe-inline'", 'https://www.googletagmanager.com', 'https://tagmanager.google.com',
                      'https://fonts.googleapis.com'],
        'font-src': ["'self'", 'data:', 'https://fonts.gstatic.com'],
        'connect-src': ["'self'", 'https://www.googletagmanager.com', 'https://*.google-analytics.com',
                        'https://*.google.com'],
        'frame-ancestors': ["'self'"],
        'base-uri': ["'self'"],
        'object-src': ["'none'"],
        'form-action': ["'self'"],
    }
    assert not {"'unsafe-inline'", "'unsafe-eval'"} & set(csp['script-src'])


def test_site_css_styles_the_notice_above_the_tab_bar():
    """El aviso es una tarjeta fija al pie; en el móvil sube por encima de la barra de pestañas y, oculto, no ocupa."""
    with open(os.path.join(ROOT, 'web', 'dist', 'css', 'site.css'), encoding='utf-8') as file:
        css = file.read()
    assert '.consent[hidden] { display: none; }' in css
    desktop = css[css.index('\n.consent {'):]
    assert 'position: fixed;' in desktop[:desktop.index('}')]
    phone = css[css.rindex('@media (max-width: 719px) {'):]
    assert '.consent { bottom: calc(68px + env(safe-area-inset-bottom)); }' in phone
