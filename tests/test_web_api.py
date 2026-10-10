"""Tests de la API `/api/v1` (controladores `Base`, `Forecast`, `Data` y `Analysis`) con el arnés ASGI.

Sin servidor ni base: los lectores de `data.py` se parchean en `webapi`.
"""
import hashlib
import json
import os

import pandas as pd
import pytest

from mtpy.core.api import Api, Router
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish, webapi
from tests.test_api_asgi import call
from tests.test_webapi_unit import ROOT, RUN, clear_queries, fixture_data, write_analysis, write_bundle  # noqa: F401


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
    ('/api/v1/forecast/es-md/headline', 'format=xml', 400, 'invalid format'),
    ('/api/v1/forecast/es-md/forecast/vote', 'format=xml', 400, 'invalid format'),
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


def fake_readers(monkeypatch):
    """Lectores de `data.py` parcheados en `webapi` con tablas pequeñas; devuelve las llamadas recibidas."""
    seen = []
    pollsters = pd.DataFrame({
        'id': [2, 3], 'name': ['GAD3', 'CIS'], 'mtype': ['private', 'public'], 'quality': [1.0, 0.8],
    })
    ratings = pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2023-07-23', '2026-11-29', '2026-11-29']),
        'pollster_id': [2, 3, 2, 3], 'pollster': ['GAD3', 'CIS', 'GAD3', 'CIS'], 'rating': [1.4, 0.6, 1.5, 0.7],
    })
    polls = pd.DataFrame({
        'event_date': pd.to_datetime(['2026-11-29'] * 3),
        'date': pd.to_datetime(['2026-09-01', '2026-10-01', '2026-09-15']),
        'pollster_id': [2, 3, 2], 'PP': [33.0, 34.0, 35.0], 'PSOE': [28.0, float('nan'), 27.5],
    })
    monkeypatch.setattr(webapi, 'get_parties', lambda: pd.DataFrame({'id': [1, 2], 'name': ['PP', 'PSOE']}))
    monkeypatch.setattr(webapi, 'get_pollsters', lambda: pollsters)
    monkeypatch.setattr(webapi, 'get_ratings', lambda scope: ratings)
    monkeypatch.setattr(webapi, 'get_next_event_date', lambda scope: '2026-11-29')

    def poll_series(scope, event):
        seen.append(('polls', scope, event))
        return polls
    monkeypatch.setattr(webapi, 'get_poll_series', poll_series)

    def results(scope, date, region):
        seen.append((scope, date, region))
        return pd.DataFrame({'date': pd.to_datetime([date]), 'region_id': [region], 'party': ['PP'], 'pct': [33.1]})
    monkeypatch.setattr(webapi, 'get_event_results', results)
    monkeypatch.setattr(webapi, 'get_house_effects', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23']), 'pollster_id': [3], 'party': ['PP'], 'dev_result_c': [-1.2]}))
    monkeypatch.setattr(webapi, 'get_herding', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23']), 'pollster_id': [2], 'ratio': [0.78]}))
    monkeypatch.setattr(webapi, 'get_drift', lambda scope: pd.DataFrame({
        'party': ['PP'], 'horizon': [4], 'rms': [1.1]}))
    return seen


@pytest.fixture
def db_api(api, fresh_app, monkeypatch):
    """API con base simulada (`app.db` cualquiera) y los lectores parcheados."""
    fresh_app.db = object()
    fresh_app.seen = fake_readers(monkeypatch)
    return api


def test_ratings_json_and_csv(db_api):
    status, headers, body = call(db_api, '/api/v1/ratings', query='scope=es')
    data = json.loads(body)
    assert status == 200 and data['contract'] == 1 and data['scope'] == 'es'
    assert [row['event_date'] for row in data['rows']] == ['2026-11-29', '2026-11-29']
    assert headers['cache-control'] == 'public, max-age=60' and headers['access-control-allow-origin'] == '*'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert headers['content-type'] == 'application/json; charset=utf-8'
    status, headers, body = call(db_api, '/api/v1/ratings', query='scope=es&format=csv')
    assert status == 200 and headers['content-type'] == 'text/csv; charset=utf-8'
    assert headers['content-disposition'] == 'attachment; filename="ratings-es.csv"'
    assert body.startswith(b'event_date,pollster_id,pollster,rating\n2026-11-29,2,GAD3,1.5\n')
    assert headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(db_api, '/api/v1/ratings', query='event=2023-07-23')
    assert json.loads(body)['rows'][0]['rating'] == 1.4 and json.loads(body)['scope'] == 'es'


def test_catalogues_and_scope_tables(db_api):
    assert json.loads(call(db_api, '/api/v1/parties')[2])['rows'][0] == {'id': 1, 'name': 'PP'}
    assert json.loads(call(db_api, '/api/v1/pollsters')[2])['rows'][1]['name'] == 'CIS'
    status, headers, body = call(db_api, '/api/v1/parties', query='format=csv')
    assert headers['content-disposition'] == 'attachment; filename="parties.csv"' and body == b'id,name\n1,PP\n2,PSOE\n'
    for path, column in (('house-effects', 'dev_result_c'), ('herding', 'ratio'), ('drift', 'rms')):
        status, headers, body = call(db_api, '/api/v1/' + path, query='scope=es')
        assert status == 200 and column in json.loads(body)['rows'][0]
        status, headers, body = call(db_api, '/api/v1/' + path, query='format=csv')
        assert headers['content-disposition'] == 'attachment; filename="{}-es.csv"'.format(path)


def test_pollster_profile(db_api):
    status, headers, body = call(db_api, '/api/v1/pollsters/3')
    data = json.loads(body)
    assert status == 200 and data['pollster']['name'] == 'CIS' and data['scope'] == 'es'
    assert [row['rating'] for row in data['ratings']] == [0.6, 0.7] and data['house_effects'][0]['party'] == 'PP'
    assert data['herding'] == [] and headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(db_api, '/api/v1/pollsters/999')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'pollster not found'}
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(db_api, '/api/v1/pollsters/abc')
    assert status == 404  # el alias `{id:num}` no casa: ruta inexistente


def test_polls_default_event_limit_and_pollster(db_api):
    data = json.loads(call(db_api, '/api/v1/polls', query='limit=2')[2])
    assert [row['date'] for row in data['rows']] == ['2026-10-01', '2026-09-15'] and data['rows'][0]['PSOE'] is None
    data = json.loads(call(db_api, '/api/v1/polls', query='pollster=2&event=2026-11-29')[2])
    assert [row['pollster_id'] for row in data['rows']] == [2, 2]
    status, headers, body = call(db_api, '/api/v1/polls', query='pollster=2&event=2026-11-29&format=csv')
    assert headers['content-disposition'] == 'attachment; filename="polls-es-2026-11-29-2.csv"'


def test_polls_filters_share_one_cached_read(db_api, fresh_app):
    call(db_api, '/api/v1/polls', query='limit=5&pollster=2')
    call(db_api, '/api/v1/polls', query='limit=50')
    assert fresh_app.seen == [('polls', 'es', '2026-11-29')]
    status, headers, body = call(db_api, '/api/v1/polls', query='event=2020-01-01')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'event not found'}
    assert headers['cache-control'] == 'no-store' and fresh_app.seen == [('polls', 'es', '2026-11-29')]
    status, headers, body = call(db_api, '/api/v1/ratings', query='event=2020-01-01')
    assert status == 404 and json.loads(body)['message'] == 'event not found'


def test_results_use_region_zero_by_default(db_api, fresh_app):
    data = json.loads(call(db_api, '/api/v1/results/es/2023-07-23')[2])
    assert data['rows'][0]['pct'] == 33.1 and data['rows'][0]['date'] == '2023-07-23'
    call(db_api, '/api/v1/results/es/2023-07-23', query='region=28')
    assert fresh_app.seen == [('es', '2023-07-23', 0), ('es', '2023-07-23', 28)]


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/polls', 'limit=9999', 400, 'invalid limit'),
    ('/api/v1/polls', 'pollster=abc', 400, 'invalid pollster'),
    ('/api/v1/polls', 'event=2026-13-01', 400, 'invalid date'),
    ('/api/v1/ratings', 'event=29/11/2026', 400, 'invalid date'),
    ('/api/v1/results/es/2026-13-01', '', 400, 'invalid date'),
    ('/api/v1/results/es/2023-07-23', 'region=x', 400, 'invalid region'),
    ('/api/v1/results/es-xx/2023-07-23', '', 400, 'invalid scope'),
    ('/api/v1/house-effects', 'scope=es-xx', 400, 'invalid scope'),
    ('/api/v1/herding', "scope=es'", 400, 'invalid scope'),
    ('/api/v1/drift', 'format=xml', 400, 'invalid format'),
    ('/api/v1/pollsters/3', 'format=csv', 404, 'no csv for this part'),
])
def test_data_errors_are_json_and_not_cached(db_api, path, query, status, message):
    got, headers, body = call(db_api, path, query=query)
    assert got == status and json.loads(body) == {'status': 'error', 'message': message}
    assert headers['cache-control'] == 'no-store' and headers['access-control-allow-origin'] == '*'


def test_data_without_database_is_a_503(api):
    status, headers, body = call(api, '/api/v1/parties')
    assert status == 503 and json.loads(body) == {'status': 'error', 'message': 'no database configured'}
    assert headers['cache-control'] == 'no-store'


def test_database_failure_is_a_503(db_api, fresh_app, monkeypatch):
    errors = []
    fresh_app.set('logger', type('Logger', (), {'error': staticmethod(errors.append)})())

    def boom(scope):
        raise RuntimeError('connection refused')
    monkeypatch.setattr(webapi, 'get_drift', boom)
    status, headers, body = call(db_api, '/api/v1/drift')
    assert status == 503 and json.loads(body)['message'] == 'database unavailable' and len(errors) == 1
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(db_api, '/api/v1/drift')
    assert status == 503 and len(errors) == 2  # el fallo no se cachea: el lector se vuelve a llamar
    monkeypatch.setattr(webapi, 'get_drift', lambda scope: pd.DataFrame({'party': ['PP']}))
    assert call(db_api, '/api/v1/drift')[0] == 200


def test_analysis_parts_and_csv(api, fresh_app, tmp_path):
    status, headers, body = call(api, '/api/v1/analysis/es/herding')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'analysis not published'}
    assert headers['cache-control'] == 'no-store'
    write_analysis(fresh_app.fs)
    webapi.site().manifest_cache_clear()
    status, headers, body = call(api, '/api/v1/analysis/es/herding')
    assert status == 200 and body == (tmp_path / 'site' / 'v1' / 'analysis' / 'es' / 'herding.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=60' and headers['access-control-allow-origin'] == '*'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    status, headers, body = call(api, '/api/v1/analysis/es/herding', query='format=csv')
    assert status == 200 and headers['content-disposition'] == 'attachment; filename="es-herding.csv"'
    assert body.startswith(b'pollster,herding\n')
    status, headers, body = call(api, '/api/v1/analysis/es/herding', query='format=csv&table=events')
    assert status == 200 and headers['content-disposition'] == 'attachment; filename="es-herding-events.csv"'
    assert body.startswith(b'event_date,pollster,ratio\n') and headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(api, '/api/v1/analysis/es/meta')
    assert status == 200 and json.loads(body)['schema'] == 'analysis-meta@1'


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/analysis/es/meta', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/analysis/es/house-effects', 'format=csv&table=events', 404, 'no csv for this part'),
    ('/api/v1/analysis/es/herding', 'format=csv&table=nope', 404, 'no csv for this part'),
    ('/api/v1/analysis/es/nope', '', 404, 'unknown part'),
    ('/api/v1/analysis/es-xx/herding', '', 400, 'invalid scope'),
    ('/api/v1/analysis/es/herding', 'format=xml', 400, 'invalid format'),
    ('/api/v1/analysis/es-md/herding', '', 404, 'analysis not published'),
])
def test_analysis_errors(api, fresh_app, path, query, status, message):
    write_analysis(fresh_app.fs)
    got, headers, body = call(api, path, query=query)
    assert got == status and json.loads(body) == {'status': 'error', 'message': message}
    assert headers['cache-control'] == 'no-store'
