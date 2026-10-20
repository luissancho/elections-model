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
from tests.fakes import synthetic_event
from tests.test_api_asgi import call
from tests.test_webapi_unit import (  # noqa: F401
    ROOT, RUN, clear_queries, fixture_data, use_database, write_analysis, write_backtest, write_bundle, write_event,
    write_study
)


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


def test_weights_part_is_served_with_the_run(api):
    status, headers, body = call(api, '/api/v1/forecast/es/weights')
    assert status == 200 and headers['cache-control'] == 'public, max-age=60'
    data = json.loads(body)
    assert data['schema'] == 'weights@1' and data['data']['parties'] == ['PP'] and data['data']['date'] == '2026-10-08'
    assert data['data']['stats']['PP']['nobs'] == 3 == len(data['data']['rows'])


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
    assert all(route['methods'] == {'GET', 'HEAD'} for route in fresh_app.router.routes[1:])


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
    monkeypatch.setattr(webapi.queries, 'get_parties', lambda: pd.DataFrame({'id': [1, 2], 'name': ['PP', 'PSOE']}))
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: pollsters)
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings)
    monkeypatch.setattr(webapi.queries, 'get_pollsters_quality', lambda: pd.DataFrame({
            'pollster_id': [3], 'fieldwork': [2], 'fieldwork_source': ['web'], 'transparency': [2],
            'transparency_source': ['web'], 'independence': [1], 'independence_source': ['web'], 'identity': [1],
            'identity_source': ['web'], 'standards': [1], 'standards_source': ['web'], 'notes': ['ok'],
            'status': ['revisada'], 'checked_at': pd.to_datetime(['2026-10-16']), 'points': [7], 'quality': [56.0],
        }))
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: '2026-11-29')

    def poll_series(scope, event, drop_contexts=None):
        seen.append(('polls', scope, event))
        return polls
    monkeypatch.setattr(webapi.queries, 'get_poll_series', poll_series)

    def results(scope, date, region):
        seen.append((scope, date, region))
        return pd.DataFrame({'date': pd.to_datetime([date] * 2), 'region_id': [0, 28], 'party': ['PP'] * 2,
                             'pct': [33.1, 30.2]})
    monkeypatch.setattr(webapi.queries, 'get_event_results', results)
    monkeypatch.setattr(webapi.queries, 'get_event_dates', lambda scope: ['2019-11-10', '2023-07-23', '2026-11-29'])
    monkeypatch.setattr(webapi.queries, 'get_house_effects', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23']), 'pollster_id': [3], 'party': ['PP'], 'dev_result_c': [-1.2]}))
    monkeypatch.setattr(webapi.queries, 'get_herding', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23']), 'pollster_id': [2], 'ratio': [0.78]}))
    monkeypatch.setattr(webapi.queries, 'get_drift', lambda scope: pd.DataFrame({
        'party': ['PP'], 'horizon': [4], 'rms': [1.1]}))
    monkeypatch.setattr(webapi.queries, 'get_events', lambda scope: synthetic_event().events)
    return seen


@pytest.fixture
def db_api(api, fresh_app, monkeypatch):
    """API con base simulada (`app.db` cualquiera) y los lectores parcheados."""
    use_database(fresh_app)
    fresh_app.seen = fake_readers(monkeypatch)
    return api


def test_ratings_json_and_csv(db_api):
    status, headers, body = call(db_api, '/api/v1/ratings', query='scope=es')
    data = json.loads(body)
    assert status == 200 and data['contract'] == 1 and data['scope'] == 'es'
    assert data['event'] == '2026-11-29'  # la fecha resuelta, no `null`
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
    assert json.loads(body)['event'] == '2023-07-23'


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
    assert data['quality']['status'] == 'revisada' and data['quality']['quality'] == 56.0
    assert data['quality']['implantation'] == 2 and data['quality']['national'] is True
    assert json.loads(call(db_api, '/api/v1/pollsters/2')[2])['quality'] is None
    status, headers, body = call(db_api, '/api/v1/pollsters/999')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'pollster not found'}
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(db_api, '/api/v1/pollsters/abc')
    assert status == 404  # el alias `{id:num}` no casa: ruta inexistente


def test_polls_default_event_limit_and_pollster(db_api):
    data = json.loads(call(db_api, '/api/v1/polls', query='limit=2')[2])
    assert [row['date'] for row in data['rows']] == ['2026-10-01', '2026-09-15'] and data['rows'][0]['PSOE'] is None
    assert data['event'] == '2026-11-29'  # la fecha resuelta, no `null`
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
    assert data['rows'] == [{'date': '2023-07-23', 'region_id': 0, 'party': 'PP', 'pct': 33.1}]
    assert data['region'] == 0
    data = json.loads(call(db_api, '/api/v1/results/es/2023-07-23', query='region=28')[2])
    assert [row['pct'] for row in data['rows']] == [30.2] and data['region'] == 28
    # una sola lectura por elección, con todas las regiones; la región se filtra fuera de la caché
    assert fresh_app.seen == [('es', '2023-07-23', None)]


def test_results_of_an_unknown_election_are_a_404_without_reading(db_api, fresh_app):
    status, headers, body = call(db_api, '/api/v1/results/es/2020-01-01')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'event not found'}
    assert headers['cache-control'] == 'no-store' and fresh_app.seen == []


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
    monkeypatch.setattr(webapi.queries, 'get_drift', boom)
    status, headers, body = call(db_api, '/api/v1/drift')
    assert status == 503 and json.loads(body)['message'] == 'database unavailable' and len(errors) == 1
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(db_api, '/api/v1/drift')
    assert status == 503 and len(errors) == 2  # el fallo no se cachea: el lector se vuelve a llamar
    monkeypatch.setattr(webapi.queries, 'get_drift', lambda scope: pd.DataFrame({'party': ['PP']}))
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


def test_studies_index_and_parts_and_csv(api, fresh_app, tmp_path):
    status, headers, body = call(api, '/api/v1/studies/voto-cera/prediction')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'study not published'}
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(api, '/api/v1/studies')
    assert status == 200 and json.loads(body) == {'contract': bundle.CONTRACT, 'studies': []}
    write_study(fresh_app.fs)
    webapi.site().manifest_cache_clear()
    status, headers, body = call(api, '/api/v1/studies')
    body = json.loads(body)
    meta = fixture_data('study-meta')
    assert status == 200 and body['contract'] == bundle.CONTRACT
    assert body['studies'] == [{'study': 'voto-cera', 'title': meta['title'], 'subtitle': meta['subtitle'],
                                'registered_at': '2026-10-18', 'checked_at': None}]
    assert headers['cache-control'] == 'public, max-age=60' and headers['access-control-allow-origin'] == '*'
    status, headers, body = call(api, '/api/v1/studies/voto-cera/prediction')
    assert status == 200
    assert body == (tmp_path / 'site' / 'v1' / 'studies' / 'voto-cera' / 'prediction.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=60' and headers['access-control-allow-origin'] == '*'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    status, headers, body = call(api, '/api/v1/studies/voto-cera/prediction', query='format=csv')
    assert status == 200 and headers['content-disposition'] == 'attachment; filename="voto-cera-last-seat.csv"'
    assert body.startswith(b'name,value\n') and headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(api, '/api/v1/studies/voto-cera/census', query='format=csv&table=census-countries')
    assert status == 200 and headers['content-disposition'] == 'attachment; filename="voto-cera-census-countries.csv"'
    status, headers, body = call(api, '/api/v1/studies/voto-cera/census', query='format=csv')
    assert headers['content-disposition'] == 'attachment; filename="voto-cera-census-series.csv"'
    status, headers, body = call(api, '/api/v1/studies/voto-cera/meta')
    assert status == 200 and json.loads(body)['schema'] == 'study-meta@1'


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/studies/voto-cera/meta', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/studies/voto-cera/vote', 'format=csv&table=last-seat', 404, 'no csv for this part'),
    ('/api/v1/studies/voto-cera/placement', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/studies/voto-cera/nope', '', 404, 'unknown part'),
    ('/api/v1/studies/nope/census', '', 404, 'unknown study'),
    ('/api/v1/studies/voto-cera/census', 'format=xml', 400, 'invalid format'),
])
def test_studies_errors(api, fresh_app, path, query, status, message):
    write_study(fresh_app.fs)
    got, headers, body = call(api, path, query=query)
    assert got == status and json.loads(body) == {'status': 'error', 'message': message}
    assert headers['cache-control'] == 'no-store'


def test_events_list_crosses_the_database_with_the_bundle(db_api, fresh_app):
    write_event(fresh_app.fs)
    status, headers, body = call(db_api, '/api/v1/events/es')
    body = json.loads(body)
    assert status == 200 and [r['date'] for r in body['rows']] == ['2019-11-10', '2023-07-23', '2026-11-29']
    assert [r['has_page'] for r in body['rows']] == [False, True, False] and body['rows'][2]['featured'] is False
    status, headers, _ = call(db_api, '/api/v1/events/es', query='format=csv')
    assert headers['content-disposition'] == 'attachment; filename="events-es.csv"'


def test_events_list_without_database_is_a_503(api):
    status, _, body = call(api, '/api/v1/events/es')
    assert status == 503 and json.loads(body)['message'] == 'no database configured'


def test_event_parts_and_csv(api, fresh_app):
    write_event(fresh_app.fs)
    status, headers, body = call(api, '/api/v1/events/es/2023-07-23/pollsters')
    assert status == 200 and json.loads(body)['schema'] == 'event-pollsters@1' and 'immutable' not in headers['cache-control']
    status, headers, _ = call(api, '/api/v1/events/es/2023-07-23/model', query='format=csv&table=seats')
    assert headers['content-type'].startswith('text/csv') and 'es-2023-07-23-model-seats.csv' in headers['content-disposition']
    status, headers, _ = call(api, '/api/v1/events/es/2023-07-23/polls', query='format=csv')
    assert 'es-2023-07-23-polls.csv' in headers['content-disposition']


def test_model_csv_of_an_event_without_backtest_is_404(api, fresh_app):
    write_event(fresh_app.fs, model=False)
    status, headers, body = call(api, '/api/v1/events/es/2023-07-23/model', query='format=csv&table=seats')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': 'event not published'}
    assert headers['cache-control'] == 'no-store'


def test_backtest_parts_and_csv(api, fresh_app):
    write_backtest(fresh_app.fs)
    status, headers, body = call(api, '/api/v1/backtest/es/metrics')
    assert status == 200 and json.loads(body)['schema'] == 'backtest-metrics@1' and 'immutable' not in headers['cache-control']
    status, headers, body = call(api, '/api/v1/backtest/es/metrics', query='format=csv')
    assert status == 200 and 'es-backtest-metrics.csv' in headers['content-disposition']
    assert headers['content-type'].startswith('text/csv')


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/events/es/2023-13-01/meta', '', 400, 'invalid date'),
    ('/api/v1/events/es/2019-11-10/meta', '', 404, 'event not published'),
    ('/api/v1/events/es/2023-07-23/nope', '', 404, 'unknown part'),
    ('/api/v1/events/es/2023-07-23/meta', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/events/es/2023-07-23/model', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/events/es/2023-07-23/polls', 'format=csv&table=seats', 404, 'no csv for this part'),
    ('/api/v1/events/es-xx', '', 400, 'invalid scope'),
    ('/api/v1/backtest/es-cb/metrics', '', 404, 'backtest not published'),
    ('/api/v1/backtest/es/meta', 'format=csv', 404, 'no csv for this part'),
    ('/api/v1/backtest/es/nope', '', 404, 'unknown part'),
])
def test_events_and_backtest_errors_are_json_and_not_cached(api, fresh_app, path, query, status, message):
    write_event(fresh_app.fs)
    write_backtest(fresh_app.fs)
    st, headers, body = call(api, path, query=query)
    assert st == status and json.loads(body) == {'status': 'error', 'message': message}
    assert headers['cache-control'] == 'no-store'


def test_polls_route_never_serves_banned_contexts(db_api, fresh_app, monkeypatch):
    seen = []
    def poll_series(scope, event, drop_contexts=None):
        seen.append(drop_contexts)
        return pd.DataFrame({'date': ['2026-10-01'], 'pollster_id': [1], 'ctype': [None], 'PP': [30.0]})
    monkeypatch.setattr(webapi.queries, 'get_poll_series', poll_series)
    webapi.queries._QUERIES.clear()
    status, _, body = call(db_api, '/api/v1/polls', query='scope=es&limit=5')
    assert status == 200 and seen == [['wban', 'exit']]
