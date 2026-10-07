"""Tests del núcleo HTTP de mtpy (`mtpy/core/api.py`) con un arnés ASGI sin servidor ni base de datos."""
import asyncio
import json
import types

import numpy as np
import pandas as pd

from mtpy.core.api import Api, Controller, Router


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


ROUTES = [
    ('/list', 'echo', 'list'), ('/types', 'echo', 'types'), ('/nan', 'echo', 'nan'), ('/text', 'echo', 'text'),
    ('/csv', 'echo', 'csv'), ('/query', 'echo', 'query'), ('/item/{id}', 'echo', 'item'),
    ('/teapot-int', 'echo', 'teapot_int'), ('/bool', 'echo', 'bool'), ('/boom', 'echo', 'boom'),
    ('/cached', 'echo', 'cached'), ('/immutable', 'echo', 'immutable'),
]


def make_api(app, routes=ROUTES, not_found=()):
    """Router con los controladores de este módulo registrado en `app`; devuelve el callable ASGI."""
    router = Router().set_namespace(types.SimpleNamespace(Echo=Echo))
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
