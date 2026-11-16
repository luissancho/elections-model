"""El consentimiento de la medición de visitas (`web/dist/js/consent.js`), comprobado con `node` (se salta sin él): la
elección guardada y su caducidad, las señales de Consent Mode, la URL del contenedor, las cookies que se borran y el
cableado del aviso sobre un documento mínimo. Nada de Google se carga antes de aceptar."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONSENT_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'consent.js').as_uri()
ENTRY_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'pages', 'consent.js')
GTM_SRC = 'https://www.googletagmanager.com/gtm.js?id=GTM-TEST123'
DENIED_ADS = {'ad_storage': 'denied', 'ad_user_data': 'denied', 'ad_personalization': 'denied'}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    """Evalúa `expr` con `c` (el módulo) definido y devuelve su JSON."""
    script = 'const c = await import("{}"); console.log(JSON.stringify({})); process.exit(0);'.format(CONSENT_JS, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


# A minimal document: the notice, its two buttons, the three state texts of the privacy page, a cookie jar that
# records what is written, a local storage and a location that counts the reloads. A step is a button to press
# (`granted`/`denied`) or `{restore: <stored choice>}`: the page shown again (from the back/forward cache unless
# `persisted` is false) after another page stored that choice
HARNESS = '''
const c = await import("{module}");
const node = (dataset, hidden) => ({{dataset, hidden, on: {{}}, addEventListener(type, fn) {{ this.on[type] = fn; }}}});
const banner = {banner};
const buttons = {{denied: node({{consent: 'denied'}}, false), granted: node({{consent: 'granted'}}, false)}};
const states = ['granted', 'denied', 'unset'].map((when) => node({{consentWhen: when}}, true));
const scripts = [], written = [];
let reloads = 0;
const store = {store};
const doc = {{
  getElementById: (id) => (id === 'consent' ? banner : null),
  querySelectorAll: (selector) => (selector === '[data-consent]' ? Object.values(buttons)
    : selector === '[data-consent-when]' ? states : []),
  createElement: (tag) => ({{tag}}),
  head: {{appendChild: (script) => scripts.push(script)}},
  get cookie() {{ return {cookie}; }},
  set cookie(value) {{ written.push(value); }},
}};
const storage = {storage};
const win = {{localStorage: storage, location: {{hostname: {hostname}, reload() {{ reloads += 1; }}}}, on: {{}},
  addEventListener(type, fn) {{ this.on[type] = fn; }}}};
c.wireConsent({{doc, win, clock: () => new Date({now})}});
for (const step of {steps}) {{
  if (typeof step === 'string') {{
    buttons[step].on.click();
  }} else {{
    store.consent = JSON.stringify(step.restore);
    win.on.pageshow({{persisted: step.persisted !== false}});
  }}
}}
console.log(JSON.stringify({{
  hidden: banner ? banner.hidden : null,
  scripts: scripts.map((script) => [script.tag, script.src, script.async]),
  layer: (win.dataLayer || []).map((args) => Array.from(args).map((arg) => (arg instanceof Date ? 'date' : arg))),
  store, written, reloads,
  shown: states.filter((state) => !state.hidden).map((state) => state.dataset.consentWhen),
}}));
process.exit(0);
'''
BANNER = "node({gtmId: 'GTM-TEST123', months: '12'}, true)"
STORAGE = '{getItem: (key) => (key in store ? store[key] : null), setItem: (key, value) => { store[key] = value; }}'
NOW = '2026-11-16T10:00:00Z'


def wire(steps=(), stored=None, cookie='', hostname='demoscopio.es', banner=BANNER, storage=STORAGE, now=NOW):
    """Monta el aviso sobre el documento mínimo, da los pasos de `steps` (pulsar `granted`/`denied`, o volver a la
    página con otra elección guardada) y devuelve lo
    observado: si el aviso queda oculto, los scripts añadidos, el `dataLayer`, lo guardado, las cookies escritas, las
    recargas y el texto de estado visible."""
    store = {'consent': json.dumps(stored)} if stored is not None else {}
    script = HARNESS.format(module=CONSENT_JS, banner=banner, store=json.dumps(store), cookie=json.dumps(cookie),
                            storage=storage, hostname=json.dumps(hostname), now=json.dumps(now), steps=json.dumps(list(steps)))
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def test_a_stored_choice_is_read_back_until_it_expires():
    """La elección se guarda con su fecha y vale los meses dados: al cumplirlos se vuelve a preguntar. Lo que no se
    entiende (nada guardado, otro formato, otro valor, una fecha rota o futura) es como no haber elegido."""
    saved = run_js('c.storedChoice("granted", new Date("2026-11-16T10:00:00Z"))')
    assert json.loads(saved) == {'analytics': 'granted', 'at': '2026-11-16T10:00:00.000Z'}
    read = 'c.readChoice({}, new Date("{}"), 12)'
    assert run_js(read.format(json.dumps(saved), '2026-11-17T00:00:00Z')) == 'granted'
    assert run_js(read.format(json.dumps(saved), '2027-11-16T09:59:59Z')) == 'granted'
    assert run_js(read.format(json.dumps(saved), '2027-11-16T10:00:00Z')) is None
    denied = json.dumps(json.dumps({'analytics': 'denied', 'at': '2026-11-16T10:00:00.000Z'}))
    assert run_js(read.format(denied, '2026-12-01T00:00:00Z')) == 'denied'
    for raw in ('null', '""', '"granted"', '"{"', '"[]"', '"null"',
                json.dumps(json.dumps({'analytics': 'maybe', 'at': '2026-11-16T10:00:00.000Z'})),
                json.dumps(json.dumps({'analytics': 'granted'})),
                json.dumps(json.dumps({'analytics': 'granted', 'at': 'ayer'})),
                json.dumps(json.dumps({'analytics': 'granted', 'at': '2026-12-25T10:00:00.000Z'}))):
        assert run_js(read.format(raw, '2026-11-17T00:00:00Z')) is None, raw
    # without a valid number of months nothing lasts
    assert run_js('c.readChoice({}, new Date("2026-11-17T00:00:00Z"), NaN)'.format(json.dumps(saved))) is None


def test_consent_signals_never_grant_advertising():
    """Las señales de Consent Mode v2: la publicidad siempre denegada (la web no tiene) y la medición según la
    elección; cualquier otra cosa es «denegado»."""
    assert run_js('c.consentSignals("granted")') == {**DENIED_ADS, 'analytics_storage': 'granted'}
    assert run_js('c.consentSignals("denied")') == {**DENIED_ADS, 'analytics_storage': 'denied'}
    assert run_js('c.consentSignals(null)') == {**DENIED_ADS, 'analytics_storage': 'denied'}


def test_gtm_url_only_for_a_container_id():
    """La URL del contenedor sólo sale de un identificador `GTM-…`: nada más llega a la dirección del script."""
    assert run_js('c.gtmUrl("GTM-TEST123")') == GTM_SRC
    for bad in ('', 'G-ABC123', 'GTM-', 'gtm-test123', 'GTM-TEST123&l=x', 'GTM-TEST 123'):
        assert run_js('c.gtmUrl({})'.format(json.dumps(bad))) is None, bad
    assert run_js('c.gtmUrl(undefined)') is None


def test_analytics_cookies_and_the_domains_they_may_live_on():
    """Las cookies de Google que se borran al rechazar (`_ga`, `_ga_<ID>` y las antiguas `_gid`, `_gat`, `_gcl_…`) y
    los dominios en los que pueden estar: el del sitio y sus padres, con punto delante."""
    cookie = '_ga=GA1.1.12.34; theme=dark; _ga_ABC123=GS1.1.1; _gid=GA1.2.3; _gat=1; _gcl_au=1.1.2; _gaother=1; x_ga=2'
    assert run_js('c.analyticsCookies({})'.format(json.dumps(cookie))) == ['_ga', '_ga_ABC123', '_gid', '_gat', '_gcl_au']
    assert run_js('c.analyticsCookies("")') == []
    assert run_js('c.cookieDomains("demoscopio.es")') == ['.demoscopio.es']
    assert run_js('c.cookieDomains("www.demoscopio.es")') == ['.www.demoscopio.es', '.demoscopio.es']
    assert run_js('c.cookieDomains("localhost")') == []


def test_without_a_choice_the_notice_shows_and_nothing_of_google_loads():
    """La primera visita: el aviso se ve, no se añade ningún script, no hay `dataLayer` y no se guarda nada."""
    seen = wire()
    assert seen['hidden'] is False and seen['scripts'] == [] and seen['layer'] == []
    assert seen['store'] == {} and seen['written'] == [] and seen['reloads'] == 0
    assert seen['shown'] == ['unset']


def test_accepting_stores_the_choice_and_loads_the_container_with_the_consent_first():
    """Al aceptar: se guarda la elección con su fecha, el aviso se oculta y el contenedor se carga una sola vez, con
    las señales de consentimiento por delante de `js` en el `dataLayer` (el fragmento de Google, con el permiso antes)."""
    seen = wire(['granted', 'granted'])
    assert seen['hidden'] is True and seen['shown'] == ['granted']
    # the date kept is the one of the click, read from the clock
    assert json.loads(seen['store']['consent']) == {'analytics': 'granted', 'at': '2026-11-16T10:00:00.000Z'}
    assert seen['scripts'] == [['script', GTM_SRC, True]]
    assert seen['layer'] == [['consent', 'default', {**DENIED_ADS, 'analytics_storage': 'granted'}], ['js', 'date']]
    assert seen['written'] == [] and seen['reloads'] == 0


def test_rejecting_stores_the_choice_and_loads_nothing():
    """Al rechazar: se guarda, el aviso se oculta y no se carga nada de Google ni se recarga la página."""
    seen = wire(['denied'])
    assert seen['hidden'] is True and seen['shown'] == ['denied']
    assert json.loads(seen['store']['consent']) == {'analytics': 'denied', 'at': '2026-11-16T10:00:00.000Z'}
    assert seen['scripts'] == [] and seen['layer'] == [] and seen['reloads'] == 0


def test_a_returning_reader_is_not_asked_again():
    """Con la elección guardada el aviso no sale: quien aceptó carga el contenedor al entrar; quien rechazó, nada."""
    granted = wire(stored={'analytics': 'granted', 'at': '2026-06-01T08:00:00.000Z'})
    assert granted['hidden'] is True and granted['scripts'] == [['script', GTM_SRC, True]] and granted['shown'] == ['granted']
    # reading does not renew the date: the consent expires 12 months after it was given
    assert json.loads(granted['store']['consent'])['at'] == '2026-06-01T08:00:00.000Z'
    denied = wire(stored={'analytics': 'denied', 'at': '2026-06-01T08:00:00.000Z'})
    assert denied['hidden'] is True and denied['scripts'] == [] and denied['layer'] == [] and denied['shown'] == ['denied']


def test_an_expired_choice_asks_again_without_loading():
    """Pasados los meses de validez se vuelve a preguntar, y mientras tanto no se carga nada."""
    seen = wire(stored={'analytics': 'granted', 'at': '2025-11-16T09:00:00.000Z'})
    assert seen['hidden'] is False and seen['scripts'] == [] and seen['shown'] == ['unset']


def test_withdrawing_deletes_the_cookies_and_reloads():
    """Quien había aceptado y rechaza (desde la página de privacidad): se guarda el rechazo, se borran las cookies de
    Google en el dominio del sitio y sus padres, y la página se recarga para que el contenedor deje de estar cargado."""
    seen = wire(['denied'], stored={'analytics': 'granted', 'at': '2026-06-01T08:00:00.000Z'},
                cookie='_ga=GA1.1.1; _ga_ABC=GS1.1; other=1', hostname='www.demoscopio.es')
    assert json.loads(seen['store']['consent']) == {'analytics': 'denied', 'at': '2026-11-16T10:00:00.000Z'}
    assert seen['written'] == [
        '_ga=; Max-Age=0; path=/',
        '_ga=; Max-Age=0; path=/; domain=.www.demoscopio.es',
        '_ga=; Max-Age=0; path=/; domain=.demoscopio.es',
        '_ga_ABC=; Max-Age=0; path=/',
        '_ga_ABC=; Max-Age=0; path=/; domain=.www.demoscopio.es',
        '_ga_ABC=; Max-Age=0; path=/; domain=.demoscopio.es',
    ]
    assert seen['reloads'] == 1


def test_rejecting_deletes_leftover_cookies_without_reloading():
    """Quien rechaza sin el contenedor cargado (una elección caducada dejó cookies) las pierde igual, sin recarga."""
    seen = wire(['denied'], cookie='_ga=GA1.1.1')
    assert seen['written'] == ['_ga=; Max-Age=0; path=/', '_ga=; Max-Age=0; path=/; domain=.demoscopio.es']
    assert seen['reloads'] == 0 and seen['scripts'] == []


def test_accepting_after_rejecting_loads_the_container_at_once():
    """Quien había rechazado y acepta (desde la página de privacidad) carga el contenedor sin recargar."""
    seen = wire(['granted'], stored={'analytics': 'denied', 'at': '2026-06-01T08:00:00.000Z'})
    assert seen['scripts'] == [['script', GTM_SRC, True]] and seen['reloads'] == 0 and seen['shown'] == ['granted']


def test_a_page_restored_from_the_back_cache_reads_the_choice_again():
    """Al volver atrás el navegador puede restaurar la página tal como quedó, con el aviso a la vista aunque ya se
    haya elegido en otra: se vuelve a leer la elección. Quien aceptó carga el contenedor; quien rechazó después de
    aceptar se queda sin él (recarga); una carga normal no hace nada de esto."""
    granted = {'analytics': 'granted', 'at': '2026-11-16T09:00:00.000Z'}
    denied = {'analytics': 'denied', 'at': '2026-11-16T09:00:00.000Z'}
    seen = wire([{'restore': granted}])
    assert seen['hidden'] is True and seen['scripts'] == [['script', GTM_SRC, True]] and seen['shown'] == ['granted']
    seen = wire([{'restore': denied}])
    assert seen['hidden'] is True and seen['scripts'] == [] and seen['shown'] == ['denied'] and seen['reloads'] == 0
    seen = wire([{'restore': denied}], stored=granted)
    assert seen['reloads'] == 1 and seen['shown'] == ['denied']
    seen = wire([{'restore': granted, 'persisted': False}])
    assert seen['hidden'] is False and seen['scripts'] == []


def test_without_the_notice_or_with_a_bad_container_nothing_happens():
    """Sin el aviso en la página (medición sin configurar) o con un identificador que no es de contenedor, el módulo
    no hace nada: ni carga ni enseña el aviso."""
    missing = wire(banner='null')
    assert missing['scripts'] == [] and missing['layer'] == [] and missing['shown'] == []
    bad = wire(banner="node({gtmId: 'G-ABC123', months: '12'}, true)")
    assert bad['hidden'] is True and bad['scripts'] == [] and bad['shown'] == []


def test_a_browser_without_storage_still_works_for_the_page():
    """Si el navegador no deja guardar (modo privado, almacenamiento bloqueado), aceptar vale para esa página y no
    rompe nada; en la siguiente se vuelve a preguntar."""
    broken = '{getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }}'
    seen = wire(['granted'], storage=broken)
    assert seen['hidden'] is True and seen['scripts'] == [['script', GTM_SRC, True]] and seen['store'] == {}
    assert wire(storage=broken)['hidden'] is False


def test_the_entry_module_only_wires_the_notice():
    """El módulo que cargan las páginas (`pages/consent.js`) sólo importa `consent.js` y cablea el aviso."""
    source = ENTRY_JS.read_text(encoding='utf-8')
    assert "import {wireConsent} from '../consent.js';" in source and 'wireConsent();' in source
    assert 'googletagmanager' not in source
