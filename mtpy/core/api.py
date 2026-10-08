import datetime
from importlib import import_module
import json
import math
import re
from urllib.parse import parse_qsl

import numpy as np
import pandas as pd

from .app import Core
from .utils.strings import to_camel


def _nan_to_none(value):
    """
    Replace the ``NaN`` floats of a value by ``None``, descending into nested lists.

    Parameters
    ----------
    value : Any
        A scalar or a (possibly nested) list, such as the output of ``numpy.ndarray.tolist``.

    Returns
    -------
    Any
        The same value with every ``float`` NaN replaced by ``None``.
    """
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def json_default(obj):
    """
    Convert the objects that ``json`` cannot serialise natively (``default`` of ``json.dumps``).

    Missing values (``pd.NaT``, ``pd.NA`` and the NaN inside numpy arrays and scalars) become
    ``None``, i.e. JSON ``null``. The order of the checks matters: ``pd.NaT`` is a ``datetime``
    subclass, so it is tested before the dates. A NaN in a native ``float`` (``np.float64``
    included) never reaches this function; with ``allow_nan=False`` ``json.dumps`` raises
    ``ValueError`` for it.

    Parameters
    ----------
    obj : Any
        The object that ``json`` does not know how to serialise.

    Returns
    -------
    Any
        A JSON-compatible value:

        - ``None`` for ``pd.NaT`` and ``pd.NA``.
        - A list, with the NaN replaced by ``None`` at any depth, for a ``numpy.ndarray``.
          ``datetime64`` arrays are cast to microseconds first, so their elements are dates
          (``NaT`` is ``None``) instead of nanosecond integers.
        - The equivalent Python scalar, ``None`` if it is a NaN, for a ``numpy.generic``
          (``datetime64`` is cast to microseconds first, as in arrays).
        - A ``YYYY-MM-DD`` string for a ``datetime`` (``pd.Timestamp`` included) at midnight
          without timezone, an ISO 8601 string for any other.
        - An ISO 8601 string for a ``date``.
        - A list for a ``set``.

    Raises
    ------
    TypeError
        If the type of the object is not supported, ``timedelta64`` (scalars and arrays)
        included.
    """
    if obj is pd.NaT or obj is pd.NA:
        return None

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == 'm':
            raise TypeError('timedelta64 values are not JSON serializable')

        if obj.dtype.kind == 'M':
            obj = obj.astype('datetime64[us]')

        return _nan_to_none(obj.tolist())

    if isinstance(obj, np.generic):
        if isinstance(obj, np.timedelta64):
            raise TypeError('timedelta64 values are not JSON serializable')

        if isinstance(obj, np.datetime64):
            obj = obj.astype('datetime64[us]')

        return _nan_to_none(obj.item())

    if isinstance(obj, datetime.datetime):
        if obj.time() == datetime.time() and obj.tzinfo is None:
            return obj.strftime('%Y-%m-%d')

        return obj.isoformat()

    if isinstance(obj, datetime.date):
        return obj.isoformat()

    if isinstance(obj, set):
        return list(obj)

    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


class Api(Core):

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive)
        response = Response(send)

        self.app.set('request', request)
        self.app.set('response', response)

        router = self.app.router

        await router.handle()
        await router.dispatch()


class Request(Core):

    def __init__(self, scope, receive):
        super().__init__()

        self.scope = scope
        self.receive = receive

        self.uri = scope['path'].lstrip('/')
        self.method = scope['method'].upper()
        self.request = self.get_request()

    def get_request(self):
        """
        Parse the query string of the request into a list of key-value pairs.

        Blank values are kept (``?c=`` yields ``('c', '')``) and repeated keys are preserved
        in order of appearance.

        Returns
        -------
        list of tuple
            The decoded ``(key, value)`` pairs of the query string.
        """
        qs = self.scope['query_string']

        if isinstance(qs, bytes):
            qs = qs.decode('latin-1')

        return parse_qsl(qs, keep_blank_values=True)

    @property
    def query(self) -> dict[str, str]:
        """
        Query string parameters as a plain dictionary.

        When a key is repeated, the first value wins. Keys with a blank value are kept.

        Returns
        -------
        dict
            The ``{key: value}`` mapping of the query string, empty if there is none.
        """
        query = {}

        for key, value in self.request:
            query.setdefault(key, value)

        return query

    async def get_body(self):
        body = b''
        more_body = True

        while more_body:
            message = await self.receive()
            body += message.get('body', b'')
            more_body = message.get('more_body', False)

        return body

    async def get_raw_body(self):
        if not hasattr(self, '_raw'):
            body = await self.get_body()
            self._raw = body.decode('utf-8')

        return self._raw

    async def get_json_body(self):
        if not hasattr(self, '_json'):
            body = await self.get_raw_body()
            self._json = json.loads(body) if body != '' else {}

        return self._json


class Response(Core):

    status_codes = {
        200: 'OK',
        301: 'Moved Permanently',
        302: 'Found',
        400: 'Bad Request',
        401: 'Unauthorized',
        403: 'Forbidden',
        404: 'Not Found',
        422: 'Unprocessable Entity',
        500: 'Internal Server Error',
        503: 'Service Unavailable'
    }

    def __init__(self, send):
        super().__init__()

        self.send = send

        self.status_code = 200
        self.headers = {}
        self.content = b''

    @property
    def raw_headers(self):
        return [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in self.headers.items()]

    def get_header(self, key):
        """
        Get the value of a response header.

        Parameters
        ----------
        key : str
            The name of the header, case insensitive.

        Returns
        -------
        str
            The value of the header.

        Raises
        ------
        KeyError
            If the header has not been set.
        """
        return self.headers[key.lower()]

    def set_header(self, key, value):
        """
        Set a response header, or remove it when the value is ``None``.

        Parameters
        ----------
        key : str
            The name of the header, case insensitive (it is stored in lower case).
        value : str or None
            The value of the header. ``None`` removes the header if it was set.

        Returns
        -------
        Response
            The response itself, to allow chaining.
        """
        key = key.lower()

        if value is not None:
            self.headers[key] = value
        elif key in self.headers:
            del self.headers[key]

        return self

    def set_content_type(self, content_type, charset='utf-8'):
        if charset is not None:
            self.set_header('content-type', content_type + '; charset=' + charset)
        else:
            self.set_header('content-type', content_type)

        return self

    def set_cache(self, seconds: int, immutable: bool = False) -> 'Response':
        """
        Allow any cache to reuse the response for a time (``cache-control`` header).

        Parameters
        ----------
        seconds : int
            Number of seconds the response stays fresh (``max-age``).
        immutable : bool, optional
            Whether to add the ``immutable`` directive, for responses that do not change
            while they are fresh.

        Returns
        -------
        Response
            The response itself, to allow chaining.
        """
        value = f'public, max-age={seconds}'

        if immutable:
            value += ', immutable'

        return self.set_header('cache-control', value)

    def set_etag(self, value: str) -> 'Response':
        """
        Set the entity tag of the response (``etag`` header).

        Parameters
        ----------
        value : str
            The opaque tag without quotes; it is sent as a strong validator (``"<value>"``).

        Returns
        -------
        Response
            The response itself, to allow chaining.
        """
        return self.set_header('etag', f'"{value}"')

    def set_status_code(self, status_code):
        self.status_code = status_code

        return self

    def set_content(self, content):
        """
        Set the body of the response from the value returned by a controller action.

        The type of ``content`` decides the body and the content type:

        - ``None``: empty ``text/plain`` body.
        - ``bool``: JSON ``true`` or ``false``.
        - ``int``: error response whose status code is the integer, with the JSON body
          ``{"status": "error", "message": "<code> <reason>"}``.
        - ``dict``, ``list`` or ``tuple``: JSON (see ``json_default`` for the conversions). A
          content that cannot be serialised, NaN in a native ``float`` included, gives a 500
          with the JSON body ``{"status": "error", "message": "Invalid content"}``.
        - ``str`` or ``bytes``: sent as is, ``text/plain`` unless the content type has already
          been set.
        - Anything else: 500 with the JSON body
          ``{"status": "error", "message": "Invalid content-type"}``.

        Parameters
        ----------
        content : Any
            The value to send.

        Returns
        -------
        Response
            The response itself, to allow chaining.
        """
        if content is None:
            self.set_content_type('text/plain')
            self.content = b''
        elif isinstance(content, bool):
            self.set_content_type('application/json')
            self.content = json.dumps(content).encode('utf-8')
        elif isinstance(content, int):
            self.set_content_type('application/json')
            self.set_status_code(content)
            self.content = json.dumps({
                'status': 'error',
                'message': str(content) + ' ' + Response.status_codes.get(content, 'Error')
            }).encode('utf-8')
        elif isinstance(content, (dict, list, tuple)):
            self.set_content_type('application/json')

            try:
                self.content = json.dumps(
                    content, ensure_ascii=False, allow_nan=False, default=json_default
                ).encode('utf-8')
            except (TypeError, ValueError):
                self.set_header('cache-control', 'no-store').set_header('etag', None)
                self.set_status_code(500)
                self.content = json.dumps({
                    'status': 'error',
                    'message': 'Invalid content'
                }).encode('utf-8')
        elif isinstance(content, str):
            if 'content-type' not in self.headers:
                self.set_content_type('text/plain')

            self.content = content.encode('utf-8')
        elif isinstance(content, bytes):
            if 'content-type' not in self.headers:
                self.set_content_type('text/plain')

            self.content = content
        else:
            self.set_header('cache-control', 'no-store').set_header('etag', None)
            self.set_content_type('application/json')
            self.set_status_code(500)
            self.content = json.dumps({
                'status': 'error',
                'message': 'Invalid content-type'
            }).encode('utf-8')

        self.set_header('content-length', str(len(self.content)))

        return self

    async def send_content(self, content):
        self.set_content(content)

        await self.send({
            'type': 'http.response.start',
            'status': self.status_code,
            'headers': self.raw_headers,
        })
        await self.send({
            'type': 'http.response.body',
            'body': self.content
        })


class Router(Core):

    def __init__(self):
        super().__init__()

        self.controller = None
        self.action = None
        self.params = {}

        self.routes = []
        self.not_found = []

        self.regex = r'{(([0-9a-z_\-]+)(:[0-9a-z_\-]+)?)}'
        self.alias_map = {
            '': r'[0-9a-z_\-]+',
            'str': r'[a-z]+',
            'num': r'[0-9]+',
            'loc': r'[a-z]{2}',
            'uri': r'.+'
        }

        self.namespace = import_module('...controllers', package=__name__)

    def set_namespace(self, namespace):
        if isinstance(namespace, str):
            self.namespace = import_module(namespace)
        else:
            self.namespace = namespace

        return self

    def add_route(self, pattern, controller, action='index', methods=None):
        methods = methods or ['GET', 'POST']

        params = []

        for match in re.compile(self.regex).finditer(pattern):
            group, name, ctype = match.groups('')
            ctype = ctype.lstrip(':')
            alias = self.alias_map[ctype]

            pattern = pattern.replace('{' + group + '}', '(?P<' + name + '>' + alias + ')')
            params.append(name)

        methods = set(method.upper() for method in methods)

        self.routes.append({
            'pattern': pattern,
            'params': params,
            'controller': controller,
            'action': action,
            'methods': methods
        })

        return self

    def add_not_found(self, pattern, controller, action='not_found'):
        self.not_found.append({
            'pattern': pattern,
            'controller': controller,
            'action': action
        })

        return self

    @staticmethod
    def parse_route(route, uri, method):
        if not method.upper() in route['methods']:
            return False

        pattern = re.compile('^' + route['pattern'] + '$')
        match = pattern.match(uri)

        if not match:
            return False

        params = match.groupdict()

        return params

    @staticmethod
    def check_not_found(route, uri):
        return uri.startswith(route['pattern'])

    async def handle(self, uri=None):
        method = self.app.request.method
        if uri is None:
            uri = '/' + self.app.request.uri

        for route in self.routes:
            params = Router.parse_route(route, uri, method)

            if params is not False:
                name = to_camel(route['controller'])
                self.controller = getattr(self.namespace, name)()
                self.action = route['action']
                self.params = params

                return self

        for route in self.not_found:
            if Router.check_not_found(route, uri):
                name = to_camel(route['controller'])
                self.controller = getattr(self.namespace, name)()
                self.action = route['action']

                return self

        self.controller = Controller()
        self.action = 'not_found'
        self.params = {}

    async def dispatch(self):
        if self.loaded():
            await self.controller.dispatch(self.action, **self.params)

    def loaded(self):
        return True if self.controller is not None else False


class Controller(Core):

    def __init__(self):
        super().__init__()

        self.request = self.app.request
        self.response = self.app.response

        self.action = None
        self.params = {}

        self.result = None

    async def dispatch(self, action, **kwargs):
        self.action = action
        self.params = dict(kwargs)

        self.before_dispatch()

        self.result = await getattr(self, action + '_action')(**self.params)

        self.after_dispatch()

        await self.send()

    def before_dispatch(self):
        pass

    def after_dispatch(self):
        pass

    async def send(self):
        await self.response.send_content(self.result)

    async def not_found_action(self):
        return 404
