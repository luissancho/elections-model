"""Tests del núcleo HTTP de mtpy (`mtpy/core/api.py`) con un arnés ASGI sin servidor ni base de datos."""
import asyncio
import json
import types

import numpy as np
import pandas as pd

from mtpy.core.api import Api, Controller, HttpError, Router


class Echo(Controller):
    """Controlador de pruebas: una acción por comportamiento verificado."""

    async def list_action(self):
        return [1, 2, 3]

    async def types_action(self):
        return {
            'n': np.int64(3), 'f': np.float64(1.5), 'b': np.bool_(True), 'd': pd.Timestamp('2026-10-07'),
            'nat': pd.NaT, 'arr': np.array([1.0, np.nan]), 'none': None,
        }

    async def nan_action(self):
        return {'x': float('nan')}

    async def text_action(self):
        return 'hola'

    async def csv_action(self):
        self.response.set_content_type('text/csv')
        return 'a,b\n1,2\n'

    async def query_action(self):
        return self.request.query

    async def item_action(self, id):
        return {'id': id}

    async def teapot_int_action(self):
        return 418

    async def bool_action(self):
        return True

    async def boom_action(self):
        raise RuntimeError('kaboom')

    async def cached_action(self):
        self.response.set_cache(60)
        return {}

    async def immutable_action(self):
        self.response.set_cache(31536000, immutable=True).set_etag('abc')
        return {}

    async def dt64_action(self):
        return {'d': np.datetime64('2026-10-07', 'ns'), 'arr': pd.Series(pd.to_datetime(['2026-10-07', None])).values}

    async def td64_action(self):
        return {'t': np.timedelta64(5, 'D')}

    async def cached_nan_action(self):
        self.response.set_cache(60).set_etag('x')
        return {'x': float('nan')}

    async def cached_404_action(self):
        self.response.set_cache(60).set_etag('x')
        return 404

    async def cached_crash_action(self):
        self.response.set_cache(60).set_etag('x')
        raise RuntimeError('crash')

    async def http_error_action(self):
        raise HttpError(404, 'no such thing')

    async def teapot_action(self):
        raise HttpError(418)

    async def cached_boom_action(self):
        self.response.set_cache(60).set_etag('x')
        raise HttpError(404, 'gone')


class Guarded(Echo):
    """Controlador cuyo `before_dispatch` rechaza toda petición."""

    def before_dispatch(self):
        raise HttpError(401, 'login required')


ROUTES = [
    ('/list', 'echo', 'list'), ('/types', 'echo', 'types'), ('/nan', 'echo', 'nan'), ('/text', 'echo', 'text'),
    ('/csv', 'echo', 'csv'), ('/query', 'echo', 'query'), ('/item/{id}', 'echo', 'item'),
    ('/teapot-int', 'echo', 'teapot_int'), ('/bool', 'echo', 'bool'), ('/boom', 'echo', 'boom'),
    ('/cached', 'echo', 'cached'), ('/immutable', 'echo', 'immutable'),
    ('/dt64', 'echo', 'dt64'), ('/td64', 'echo', 'td64'), ('/cached-nan', 'echo', 'cached_nan'),
    ('/http-error', 'echo', 'http_error'), ('/teapot', 'echo', 'teapot'), ('/guarded', 'guarded', 'list'),
    ('/cached-boom', 'echo', 'cached_boom'), ('/cached-404', 'echo', 'cached_404'),
    ('/cached-crash', 'echo', 'cached_crash'),
]


def make_api(app, routes=ROUTES, not_found=()):
    """Router con los controladores de este módulo registrado en `app`; devuelve el callable ASGI."""
    router = Router().set_namespace(types.SimpleNamespace(Echo=Echo, Guarded=Guarded))
    for pattern, controller, action in routes:
        router.add_route(pattern, controller, action, methods=['GET'])
    for pattern, controller in not_found:
        router.add_not_found(pattern, controller)
    app.set('router', router)
    return Api()


def call(api, path, query='', method='GET', scope_type='http', incoming=None):
    """Una petición ASGI; devuelve (status, cabeceras, cuerpo) en `http` y los mensajes enviados en otros scopes.

    Los scopes distintos de `http` solo llevan `type` y `headers`, como los que envía uvicorn.
    """
    scope = {'type': scope_type, 'headers': []}
    if scope_type == 'http':
        scope.update({'path': path, 'method': method, 'query_string': query.encode('latin-1')})
    pending = list(incoming) if incoming is not None else [{'type': 'http.request', 'body': b'', 'more_body': False}]
    sent = []

    async def receive():
        return pending.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(api(scope, receive, send))
    if scope_type != 'http':
        return sent
    start = [m for m in sent if m['type'] == 'http.response.start'][0]
    body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
    headers = {k.decode('latin-1'): v.decode('latin-1') for k, v in start['headers']}
    return start['status'], headers, body


def test_query_returns_first_value_of_each_key_and_keeps_blank_values(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/query', query='a=1&b=2&a=3&c=')
    assert status == 200
    assert json.loads(body) == {'a': '1', 'b': '2', 'c': ''}


def test_query_is_empty_without_query_string(fresh_app):
    assert json.loads(call(make_api(fresh_app), '/query')[2]) == {}


def test_list_is_serialised_as_json(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/list')
    assert status == 200
    assert headers['content-type'] == 'application/json; charset=utf-8'
    assert json.loads(body) == [1, 2, 3]


def test_numpy_and_pandas_values_are_converted(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/types')
    assert status == 200
    assert json.loads(body) == {
        'n': 3, 'f': 1.5, 'b': True, 'd': '2026-10-07', 'nat': None, 'arr': [1.0, None], 'none': None,
    }


def test_nan_in_native_float_gives_a_json_500_not_an_invalid_body(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/nan')
    assert status == 500
    assert headers['content-type'] == 'application/json; charset=utf-8'
    assert json.loads(body) == {'status': 'error', 'message': 'Invalid content'}


def test_unknown_int_status_and_bool_do_not_raise(fresh_app):
    api = make_api(fresh_app)
    status, headers, body = call(api, '/teapot-int')
    assert status == 418
    assert json.loads(body) == {'status': 'error', 'message': '418 Error'}
    status, headers, body = call(api, '/bool')
    assert status == 200
    assert json.loads(body) is True


def test_str_without_explicit_type_is_text_plain(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/text')
    assert headers['content-type'] == 'text/plain; charset=utf-8'
    assert body == b'hola'


def test_explicit_content_type_is_kept_for_str(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/csv')
    assert status == 200
    assert headers['content-type'] == 'text/csv; charset=utf-8'
    assert body == b'a,b\n1,2\n'


def test_header_keys_are_case_insensitive(fresh_app):
    from mtpy.core.api import Response

    response = Response(None).set_header('Content-Type', 'text/csv')
    assert response.headers == {'content-type': 'text/csv'}
    assert response.get_header('CONTENT-TYPE') == 'text/csv'


def test_cache_headers(fresh_app):
    api = make_api(fresh_app)
    assert call(api, '/cached')[1]['cache-control'] == 'public, max-age=60'
    headers = call(api, '/immutable')[1]
    assert headers['cache-control'] == 'public, max-age=31536000, immutable'
    assert headers['etag'] == '"abc"'


def test_status_codes_include_422_and_503():
    from mtpy.core.api import Response

    assert Response.status_codes[422] == 'Unprocessable Entity'
    assert Response.status_codes[503] == 'Service Unavailable'


def test_datetime64_values_are_serialised_as_dates(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/dt64')
    assert status == 200
    assert json.loads(body) == {'d': '2026-10-07', 'arr': ['2026-10-07', None]}


def test_timedelta64_is_rejected_with_a_json_500(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/td64')
    assert status == 500
    assert json.loads(body) == {'status': 'error', 'message': 'Invalid content'}


def test_serialisation_failure_is_not_cacheable(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/cached-nan')
    assert status == 500
    assert headers['cache-control'] == 'no-store'
    assert 'etag' not in headers


def test_http_error_becomes_a_json_error_with_its_status(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/http-error')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': 'no such thing'}


def test_http_error_with_unknown_status_and_no_message(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/teapot')
    assert status == 418
    assert json.loads(body) == {'status': 'error', 'message': '418 Error'}


def test_http_error_in_before_dispatch_is_handled(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/guarded')
    assert status == 401
    assert json.loads(body) == {'status': 'error', 'message': 'login required'}


def test_unhandled_exception_is_logged_and_returns_500(fresh_app):
    errors = []
    fresh_app.set('logger', types.SimpleNamespace(error=errors.append))
    status, headers, body = call(make_api(fresh_app), '/boom')
    assert status == 500
    assert json.loads(body) == {'status': 'error', 'message': '500 Internal Server Error'}
    assert len(errors) == 1 and 'kaboom' in errors[0]


def test_unhandled_exception_without_logger_still_returns_500(fresh_app):
    assert call(make_api(fresh_app), '/boom')[0] == 500


def test_error_responses_are_not_cacheable(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/cached-boom')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': 'gone'}
    assert headers['cache-control'] == 'no-store'
    assert 'etag' not in headers


def test_prefix_not_found_does_not_reuse_previous_params(fresh_app):
    api = make_api(fresh_app, not_found=[('/x', 'echo')])
    assert json.loads(call(api, '/item/42')[2]) == {'id': '42'}
    status, headers, body = call(api, '/x/anything')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': '404 Not Found'}


def test_unknown_route_is_a_json_404(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/nope')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': '404 Not Found'}


def test_lifespan_scope_is_acknowledged(fresh_app):
    sent = call(make_api(fresh_app), '', scope_type='lifespan',
                incoming=[{'type': 'lifespan.startup'}, {'type': 'lifespan.shutdown'}])
    assert sent == [{'type': 'lifespan.startup.complete'}, {'type': 'lifespan.shutdown.complete'}]


def test_other_scopes_are_ignored(fresh_app):
    assert call(make_api(fresh_app), '', scope_type='websocket', incoming=[]) == []


def test_mtpy_api_registers_extra_routes(fresh_app):
    from mtpy import mtpy

    api = mtpy.api(routes=[('/api/v1/ping', 'index', 'index', ['GET'])])
    assert isinstance(api, Api)
    assert [route['pattern'] for route in fresh_app.router.routes] == ['/', '/api/v1/ping']
    assert fresh_app.router.routes[1]['methods'] == {'GET'}


def test_mtpy_api_without_app_returns_none():
    from mtpy import mtpy
    from mtpy.core.app import App

    saved = App._app
    App._app = None
    try:
        assert mtpy.api() is None
    finally:
        App._app = saved


def test_int_error_result_is_not_cacheable(fresh_app):
    """Un error devuelto como entero no hereda la cache ni el etag de la acción."""
    status, headers, body = call(make_api(fresh_app), '/cached-404')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': '404 Not Found'}
    assert headers['cache-control'] == 'no-store'
    assert 'etag' not in headers


def test_unhandled_exception_response_is_not_cacheable(fresh_app):
    """Una excepción no controlada responde 500 sin cache ni etag."""
    status, headers, body = call(make_api(fresh_app), '/cached-crash')
    assert status == 500
    assert headers['cache-control'] == 'no-store'
    assert 'etag' not in headers


def test_serialisation_failure_is_logged(fresh_app):
    """Un contenido no serializable se registra en el log antes de responder 500."""
    errors = []
    fresh_app.set('logger', types.SimpleNamespace(error=errors.append))
    status, headers, body = call(make_api(fresh_app), '/nan')
    assert status == 500
    assert len(errors) == 1 and 'not JSON compliant' in errors[0]


def test_index_route_returns_api_home(fresh_app):
    """La ruta raíz de `mtpy.api()` devuelve la portada de la API."""
    from mtpy import mtpy

    api = mtpy.api()
    status, headers, body = call(api, '/')
    assert status == 200
    assert json.loads(body) == {'status': 'ok', 'message': 'API Home'}
