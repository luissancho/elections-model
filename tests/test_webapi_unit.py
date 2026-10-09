"""Tests de `mtpy/lib/webapi.py` sin servidor ni base de datos: validadores, cachés y lector del paquete."""
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from mtpy.core.api import HttpError
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish, webapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures', 'bundle')
RUN = '20261008-120000'


def fixture_data(name):
    with open(os.path.join(FIXTURES, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)['data']


def write_bundle(fs, scope='es', run_id=RUN, manifest=True):
    """Paquete mínimo a partir de los fixtures: un run completo, su history y el manifest."""
    writer = bundle.BundleWriter(fs)
    for part in bundle.RUN_PARTS:
        writer.write_json(bundle.path_part(scope, run_id, part), part, fixture_data(part), scope, run_id=run_id)
    for mode in bundle.MODES:
        for part in bundle.MODE_PARTS:
            writer.write_json(bundle.path_part(scope, run_id, part, mode), part, fixture_data(part), scope,
                              run_id=run_id, mode=mode)
    writer.write_csv(bundle.path_csv(scope, run_id, 'series'), pd.DataFrame({'date': ['2026-10-05'], 'name': ['PP']}))
    writer.write_csv(bundle.path_csv(scope, run_id, 'vote', 'forecast'), pd.DataFrame({'name': ['PP'], 'pct': [40.0]}))
    publish.rebuild_history(writer, scope)
    if manifest:
        publish.update_manifest(writer, {scope: publish.latest_entry(fixture_data('headline') | {'run_id': run_id})})
    return writer


@pytest.fixture
def site_app(fresh_app, tmp_path):
    """`App` con `fs` local, el catálogo de `data/` y un paquete mínimo publicado para `es`."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return fresh_app


def test_validators_accept_the_catalogue_and_reject_the_rest(site_app):
    assert webapi.check_scope('es') == 'es' and webapi.check_scope('es-md') == 'es-md'
    for bad in ("es'", 'es;', 'es-xx', 'ES', 'es/..', ''):
        with pytest.raises(HttpError) as err:
            webapi.check_scope(bad)
        assert err.value.status == 400
    assert webapi.check_run(None) is None and webapi.check_run('') is None
    assert webapi.check_run(RUN) == RUN
    for bad in ('2026-10-08', RUN + '\n', RUN + '/..', 'latest'):
        with pytest.raises(HttpError) as err:
            webapi.check_run(bad)
        assert err.value.status == 400
    assert webapi.check_mode('nowcast') == 'nowcast'
    with pytest.raises(HttpError) as err:
        webapi.check_mode('tomorrow')
    assert err.value.status == 400
    assert webapi.check_part('series', webapi.RUN_PARTS) == 'series'
    with pytest.raises(HttpError) as err:
        webapi.check_part('nope', webapi.RUN_PARTS)
    assert err.value.status == 404
    assert webapi.check_format(None) == 'json' and webapi.check_format('csv') == 'csv'
    with pytest.raises(HttpError) as err:
        webapi.check_format('xml')
    assert err.value.status == 400


def test_ttl_cache_with_an_injected_clock():
    now = [0.0]
    calls = []
    cache = webapi.TTLCache(ttl=60, clock=lambda: now[0])

    def loader():
        calls.append(1)
        return len(calls)
    assert cache.get('k', loader) == 1 and cache.get('k', loader) == 1
    now[0] = 59.9
    assert cache.get('k', loader) == 1
    now[0] = 60.0
    assert cache.get('k', loader) == 2

    def boom():
        raise FileNotFoundError('x')
    with pytest.raises(FileNotFoundError):
        cache.get('other', boom)
    assert cache.get('other', lambda: 'ok') == 'ok'


def test_lru_cache_evicts_the_oldest():
    cache = webapi.LRUCache(maxsize=2)
    assert cache.get('a', lambda: 1) == 1 and cache.get('b', lambda: 2) == 2
    assert cache.get('a', lambda: 99) == 1          # acierto: `a` pasa a ser el más reciente
    assert cache.get('c', lambda: 3) == 3           # expulsa `b`
    assert cache.get('b', lambda: 20) == 20 and len(cache) == 2


def test_settings_defaults_without_config(fresh_app):
    assert webapi.settings() == {'prefix': 'site/v1', 'cache_ttl': 60.0}
    from mtpy.core.app import Config
    fresh_app.set('config', Config({'web': {'prefix': 'site-dry/v1', 'cache_ttl': '5'}}))
    assert webapi.settings() == {'prefix': 'site-dry/v1', 'cache_ttl': 5.0}
    fresh_app.set('config', Config({'web': {'prefix': '', 'cache_ttl': ''}}))
    assert webapi.settings() == {'prefix': 'site/v1', 'cache_ttl': 60.0}


def test_site_reads_the_bundle_and_resolves_latest(site_app):
    site = webapi.site()
    assert site is webapi.site()
    manifest = bundle.loads(site.manifest())
    assert manifest['schema'] == 'manifest@1' and site.latest_run('es') == RUN
    assert bundle.loads(site.history('es'))['data']['runs'][0]['run_id'] == RUN
    assert bundle.loads(site.run_file('es', RUN, 'meta'))['schema'] == 'meta@1'
    assert bundle.loads(site.run_file('es', RUN, 'vote', 'forecast'))['mode'] == 'forecast'
    assert site.run_csv('es', RUN, 'series') == b'date,name\n2026-10-05,PP\n'
    assert site.run_csv('es', RUN, 'vote', 'forecast').startswith(b'name,pct\n')
    assert site.freeze() == {'active': False, 'message': None}
    for call, status in (
        (lambda: site.latest_run('es-md'), 404), (lambda: site.history('es-md'), 404),
        (lambda: site.run_file('es', '20200101-000000', 'meta'), 404), (lambda: site.run_csv('es', RUN, 'meta'), 404),
    ):
        with pytest.raises(HttpError) as err:
            call()
        assert err.value.status == status


def test_site_scopes_crosses_the_catalogue_with_the_manifest(site_app):
    data = webapi.site().scopes()
    codes = [row['code'] for row in data['scopes']]
    assert codes[:2] == ['es', 'es-an'] and len(codes) == 18 and data['contract'] == 1
    es = data['scopes'][0]
    assert es['simulable'] is True and es['latest'] == RUN and es['seats'] == 350 and es['parent'] is None
    md = [row for row in data['scopes'] if row['code'] == 'es-md'][0]
    assert md['simulable'] is False and md['latest'] is None and md['parent'] == 'es' and md['name'] == 'Madrid'


def test_site_without_manifest_is_a_503_and_caches_nothing(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    with pytest.raises(HttpError) as err:
        webapi.site().manifest()
    assert err.value.status == 503
    write_bundle(fresh_app.fs)
    assert bundle.loads(webapi.site().manifest())['data']['scopes']['es']['latest'] == RUN


def test_site_pointers_expire_with_the_ttl(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    writer = write_bundle(fresh_app.fs)
    now = [0.0]
    site = webapi.Site(bundle.BundleReader(fresh_app.fs), ttl=60, clock=lambda: now[0])
    assert site.latest_run('es') == RUN
    publish.update_manifest(writer, {'es': {'latest': '20261009-120000'}})
    assert site.latest_run('es') == RUN
    now[0] = 61.0
    assert site.latest_run('es') == '20261009-120000'


def test_site_requires_a_file_system(fresh_app):
    with pytest.raises(HttpError) as err:
        webapi.site()
    assert err.value.status == 503


def test_webapi_import_stays_light():
    """Los workers de la web no deben cargar el modelo: `webapi` solo importa `bundle`.

    scipy y statsmodels quedan fuera de la comprobación: `mtpy.core.io` (y con él `bundle`) ya los
    arrastra a través de `mtpy.core.utils.helpers`, ajeno a `webapi`.
    """
    import subprocess
    import sys
    code = ("import sys, mtpy.lib.webapi; "
            "print(sorted(m for m in ('mtpy.lib.data', 'mtpy.lib.publish', 'mtpy.lib.simulator') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == '[]'


def test_settings_fall_back_to_the_default_ttl_on_garbage(monkeypatch):
    fake = SimpleNamespace(config=SimpleNamespace(web=SimpleNamespace(prefix='', cache_ttl='abc')))
    monkeypatch.setattr(webapi.App, 'get_', staticmethod(lambda: fake))
    with pytest.warns(UserWarning):
        assert webapi.settings() == {'prefix': bundle.PREFIX, 'cache_ttl': 60.0}


def test_check_region_accepts_integers_only():
    assert webapi.check_region(None) is None and webapi.check_region('') is None
    assert webapi.check_region('28') == 28
    for bad in ('abc', '-1', '28.0', '1' * 7, '28;'):
        with pytest.raises(HttpError) as err:
            webapi.check_region(bad)
        assert err.value.status == 400 and err.value.message == 'invalid region'
