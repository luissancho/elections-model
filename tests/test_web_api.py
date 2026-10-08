"""Tests de la API `/api/v1` (controladores `Base` y `Forecast`) con el arnés ASGI, sin servidor ni base."""
import hashlib
import json
import os

import pytest

from mtpy.core.api import Api, Router
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish, webapi
from tests.test_api_asgi import call
from tests.test_webapi_unit import ROOT, RUN, fixture_data, write_bundle


def make_web_api(app):
    """Router real de la API (namespace `mtpy.controllers`) registrado en `app`; devuelve el ASGI."""
    router = Router()
    webapi.add_routes(router)
    app.set('router', router)
    return Api()


@pytest.fixture
def api(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return make_web_api(fresh_app)


def test_health_needs_nothing_and_is_not_cached(fresh_app):
    status, headers, body = call(make_web_api(fresh_app), '/api/v1/health')
    assert status == 200 and json.loads(body) == {'status': 'ok', 'contract': 1}
    assert headers['cache-control'] == 'no-store' and headers['access-control-allow-origin'] == '*'


def test_manifest_is_served_verbatim_with_pointer_cache(api, tmp_path):
    status, headers, body = call(api, '/api/v1/manifest')
    assert status == 200 and headers['content-type'] == 'application/json; charset=utf-8'
    assert body == (tmp_path / 'site' / 'v1' / 'manifest.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=60'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert 'x-freeze' not in headers


def test_without_bundle_the_api_answers_503(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    status, headers, body = call(make_web_api(fresh_app), '/api/v1/manifest')
    assert status == 503 and json.loads(body)['message'] == 'no bundle published yet'
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(make_web_api(fresh_app), '/api/v1/forecast/es/headline')
    assert status == 503


def test_scopes_cross_catalogue_and_manifest(api):
    status, headers, body = call(api, '/api/v1/scopes')
    data = json.loads(body)
    assert status == 200 and data['scopes'][0]['code'] == 'es' and data['scopes'][0]['latest'] == RUN
    assert headers['cache-control'] == 'public, max-age=60'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert headers['content-type'] == 'application/json; charset=utf-8'


def test_meta_and_parts_resolve_latest_or_an_explicit_run(api, tmp_path):
    run_dir = tmp_path / 'site' / 'v1' / 'runs' / 'es' / RUN
    status, headers, body = call(api, '/api/v1/forecast/es')
    assert status == 200 and body == (run_dir / 'meta.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(api, '/api/v1/forecast/es/series', query='run=' + RUN)
    assert status == 200 and body == (run_dir / 'series.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=31536000, immutable'
    status, headers, body = call(api, '/api/v1/forecast/es/runs')
    assert status == 200 and json.loads(body)['schema'] == 'history@1'
    status, headers, body = call(api, '/api/v1/forecast/es/forecast/vote')
    assert status == 200 and json.loads(body)['mode'] == 'forecast' and headers['cache-control'] == 'public, max-age=60'


def test_csv_twins(api):
    status, headers, body = call(api, '/api/v1/forecast/es/series', query='format=csv')
    assert status == 200 and headers['content-type'] == 'text/csv; charset=utf-8'
    assert headers['content-disposition'] == 'attachment; filename="es-{}-series.csv"'.format(RUN)
    assert body == b'date,name\n2026-10-05,PP\n' and headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(api, '/api/v1/forecast/es/forecast/vote', query='format=csv&run=' + RUN)
    assert status == 200 and headers['content-disposition'].endswith('-forecast-vote.csv"')
    assert headers['cache-control'] == 'public, max-age=31536000, immutable'
    status, headers, body = call(api, '/api/v1/forecast/es/meta', query='format=csv')
    assert status == 404 and json.loads(body)['message'] == 'no csv for this part'
    status, headers, body = call(api, '/api/v1/forecast/es/series', query='format=xml')
    assert status == 400 and json.loads(body)['message'] == 'invalid format'


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/forecast/es-xx', '', 400, 'invalid scope'),
    ('/api/v1/forecast/es-md', '', 404, 'scope not published'),
    ('/api/v1/forecast/es-md/runs', '', 404, 'no runs published'),
    ('/api/v1/forecast/es', 'run=2026-10-08', 400, 'invalid run'),
    ('/api/v1/forecast/es', 'run=20200101-000000', 404, 'run not found'),
    ('/api/v1/forecast/es/nope', '', 404, 'unknown part'),
    ('/api/v1/forecast/es/tomorrow/vote', '', 400, 'invalid mode'),
    ('/api/v1/forecast/es/forecast/nope', '', 404, 'unknown part'),
    ('/api/v1/forecast/es/forecast/vote/extra', '', 404, '404 Not Found'),
    ("/api/v1/forecast/es'/meta", '', 404, '404 Not Found'),
    ('/api/v1/nope', '', 404, '404 Not Found'),
])
def test_errors_are_json_and_not_cached(api, path, query, status, message):
    got, headers, body = call(api, path, query=query)
    assert got == status and json.loads(body) == {'status': 'error', 'message': message}
    assert headers['cache-control'] == 'no-store' and headers['content-type'] == 'application/json; charset=utf-8'


def test_post_is_not_allowed(api):
    status, headers, body = call(api, '/api/v1/manifest', method='POST')
    assert status == 404


def test_freeze_header_when_the_manifest_is_frozen(api, fresh_app):
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda'})
    webapi.site().manifest_cache_clear()
    status, headers, body = call(api, '/api/v1/forecast/es/headline')
    assert status == 200 and headers['x-freeze'] == 'active'
    status, headers, body = call(api, '/api/v1/health')
    assert 'x-freeze' not in headers


def test_mtpy_api_registers_the_web_routes(fresh_app):
    from mtpy import mtpy
    api = mtpy.api(routes=webapi.ROUTES)
    patterns = [route['pattern'] for route in fresh_app.router.routes]
    assert patterns[0] == '/' and len(patterns) == 1 + len(webapi.ROUTES)
    assert all(route['methods'] == {'GET'} for route in fresh_app.router.routes[1:])
