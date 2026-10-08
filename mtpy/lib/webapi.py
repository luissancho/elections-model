"""
Service layer of the web API.

Phase 2 serves the published bundle (``site/v1``): request validators, small in-process
caches and a reader that returns the bundle files as bytes, untouched. The database routes
arrive in phase 4. This module imports only ``bundle``, so the web workers do not load the
model nor the scientific stack.
"""
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from ..core.api import HttpError
from ..core.app import App
from . import bundle
from .bundle import BundleReader

DEFAULT_TTL = 60
LRU_SIZE = 200
IMMUTABLE_MAX_AGE = 31536000
FORMATS = ('json', 'csv')
CSV_PARTS = bundle.RUN_PARTS[2:]
MODE_PARTS = bundle.MODE_PARTS
RUN_PARTS = bundle.RUN_PARTS
MODES = bundle.MODES


def settings() -> dict:
    """
    Read the ``web`` section of the configuration.

    Returns
    -------
    dict
        ``{'prefix': str, 'cache_ttl': float}``. Missing or empty values fall back to
        ``bundle.PREFIX`` and ``DEFAULT_TTL``.
    """
    config = getattr(App.get_(), 'config', None)
    web = getattr(config, 'web', None) if config is not None else None
    prefix = getattr(web, 'prefix', None)
    ttl = getattr(web, 'cache_ttl', None)

    return {
        'prefix': str(prefix) if prefix not in (None, '') else bundle.PREFIX,
        'cache_ttl': float(ttl) if ttl not in (None, '') else float(DEFAULT_TTL),
    }


class TTLCache:
    """
    Cache whose entries expire a fixed time after being stored.

    Parameters
    ----------
    ttl : float
        Seconds an entry stays valid.
    clock : callable, optional
        Function returning the current time in seconds. Defaults to ``time.monotonic``.
    """

    def __init__(self, ttl: float, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._items = {}
        self._lock = Lock()

    def get(self, key: Any, loader: Callable[[], Any]) -> Any:
        """
        Return the cached value or load, store and return a fresh one.

        Parameters
        ----------
        key : hashable
            Cache key.
        loader : callable
            Function without arguments that produces the value on a miss.

        Returns
        -------
        object
            The cached or freshly loaded value.

        Raises
        ------
        Exception
            Whatever ``loader`` raises; nothing is stored in that case.
        """
        with self._lock:
            hit = self._items.get(key)

        if hit is not None and self.clock() - hit[0] < self.ttl:
            return hit[1]

        value = loader()

        with self._lock:
            self._items[key] = (self.clock(), value)

        return value

    def clear(self) -> None:
        """
        Drop every entry.

        Returns
        -------
        None
        """
        with self._lock:
            self._items.clear()


class LRUCache:
    """
    Cache that keeps the most recently used entries.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of entries; the least recently used is evicted beyond it.
    """

    def __init__(self, maxsize: int = LRU_SIZE):
        self.maxsize = maxsize
        self._items = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        """
        Number of cached entries.

        Returns
        -------
        int
            Entries currently stored.
        """
        return len(self._items)

    def get(self, key: Any, loader: Callable[[], Any]) -> Any:
        """
        Return the cached value or load, store and return a fresh one.

        Parameters
        ----------
        key : hashable
            Cache key.
        loader : callable
            Function without arguments that produces the value on a miss.

        Returns
        -------
        object
            The cached or freshly loaded value.

        Raises
        ------
        Exception
            Whatever ``loader`` raises; nothing is stored in that case.
        """
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)

                return self._items[key]

        value = loader()

        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)

            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

        return value

    def clear(self) -> None:
        """
        Drop every entry.

        Returns
        -------
        None
        """
        with self._lock:
            self._items.clear()


_CATALOGUE = TTLCache(DEFAULT_TTL)


def catalogue() -> pd.DataFrame:
    """
    Scopes catalogue (``data/es-scopes.csv``) indexed by ``scode``, cached with a TTL.

    Returns
    -------
    pandas.DataFrame
        One row per scope, in file order.

    Raises
    ------
    HttpError
        503 when the app has no data directory.
    """
    data = App.get_().data

    if data is None:
        raise HttpError(503, 'no data directory configured')

    _CATALOGUE.ttl = settings()['cache_ttl']

    return _CATALOGUE.get('catalogue', lambda: data.read_csv('es-scopes.csv').set_index('scode'))


def scope_codes() -> list:
    """
    Codes of every scope in the catalogue.

    Returns
    -------
    list of str
        Scope codes in catalogue order.
    """
    return catalogue().index.tolist()


def check_scope(scope: str) -> str:
    """
    Validate a scope code.

    Parameters
    ----------
    scope : str
        Scope code from the request.

    Returns
    -------
    str
        The same code.

    Raises
    ------
    HttpError
        400 when it is malformed or not in the catalogue.
    """
    if not isinstance(scope, str) or not bundle.ROUTE_ALIAS_RE.fullmatch(scope) or scope not in scope_codes():
        raise HttpError(400, 'invalid scope')

    return scope


def check_run(run: Optional[str]) -> Optional[str]:
    """
    Validate a run id.

    Parameters
    ----------
    run : str or None
        Run id from the request.

    Returns
    -------
    str or None
        The run id, or ``None`` when it is missing or empty.

    Raises
    ------
    HttpError
        400 when it does not look like ``YYYYMMDD-HHMMSS``.
    """
    if run is None or run == '':
        return None

    if not isinstance(run, str) or not bundle.RUN_ID_RE.fullmatch(run):
        raise HttpError(400, 'invalid run')

    return run


def check_mode(mode: str) -> str:
    """
    Validate a simulation mode.

    Parameters
    ----------
    mode : str
        Mode from the request.

    Returns
    -------
    str
        The same mode.

    Raises
    ------
    HttpError
        400 when it is not in ``MODES``.
    """
    if mode not in MODES:
        raise HttpError(400, 'invalid mode')

    return mode


def check_part(part: str, allowed: Sequence[str]) -> str:
    """
    Validate a bundle part name.

    Parameters
    ----------
    part : str
        Part name from the request.
    allowed : sequence of str
        Part names accepted by the route.

    Returns
    -------
    str
        The same part.

    Raises
    ------
    HttpError
        404 when it is not allowed.
    """
    if part not in allowed:
        raise HttpError(404, 'unknown part')

    return part


def check_format(fmt: Optional[str]) -> str:
    """
    Validate an output format.

    Parameters
    ----------
    fmt : str or None
        Format from the request.

    Returns
    -------
    str
        ``'json'`` when missing or empty, else the same format.

    Raises
    ------
    HttpError
        400 when it is not in ``FORMATS``.
    """
    if fmt is None or fmt == '':
        return 'json'

    if fmt not in FORMATS:
        raise HttpError(400, 'invalid format')

    return fmt


class Site:
    """
    Service over the published bundle: pointers under a TTL, run files under an LRU.

    Files are returned as the bytes stored in the bundle, never re-serialised.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    ttl : float, optional
        Seconds the manifest and history pointers stay cached.
    lru : int, optional
        Number of run files kept in memory.
    clock : callable, optional
        Time source of the TTL cache.
    """

    def __init__(self, reader: BundleReader, ttl: float = DEFAULT_TTL, lru: int = LRU_SIZE,
                 clock: Callable[[], float] = time.monotonic):
        self.reader = reader
        self.pointers = TTLCache(ttl, clock)
        self.files = LRUCache(lru)

    def _read(self, name: str) -> bytes:
        """
        Read a bundle file verbatim.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        """
        return self.reader.fs.read_bytes(self.reader.path(name))

    def manifest_cache_clear(self) -> None:
        """
        Drop the cached pointers so the next read goes to the file system.

        Returns
        -------
        None
        """
        self.pointers.clear()

    def manifest(self) -> bytes:
        """
        Bytes of ``manifest.json``.

        Returns
        -------
        bytes
            The manifest, cached under the TTL.

        Raises
        ------
        HttpError
            503 when no bundle has been published yet.
        """
        def load():
            try:
                return self._read(bundle.path_manifest())
            except FileNotFoundError:
                raise HttpError(503, 'no bundle published yet')

        return self.pointers.get('manifest', load)

    def manifest_data(self) -> dict:
        """
        Payload of the manifest.

        Returns
        -------
        dict
            The ``data`` member of the manifest envelope.
        """
        return bundle.loads(self.manifest())['data']

    def freeze(self) -> dict:
        """
        Freeze state published in the manifest.

        Returns
        -------
        dict
            The ``freeze`` member of the manifest data.
        """
        return self.manifest_data()['freeze']

    def latest_run(self, scope: str) -> str:
        """
        Run id of the latest published run of a scope.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        str
            The run id.

        Raises
        ------
        HttpError
            404 when the scope has no entry in the manifest.
        """
        entry = self.manifest_data()['scopes'].get(scope)

        if entry is None:
            raise HttpError(404, 'scope not published')

        return entry['latest']

    def history(self, scope: str) -> bytes:
        """
        Bytes of the history file of a scope.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        bytes
            The history, cached under the TTL.

        Raises
        ------
        HttpError
            404 when the scope has no published runs.
        """
        def load():
            try:
                return self._read(bundle.path_history(scope))
            except FileNotFoundError:
                raise HttpError(404, 'no runs published')

        return self.pointers.get(('history', scope), load)

    def _run_bytes(self, path: str) -> bytes:
        """
        Read a run file through the LRU cache.

        Parameters
        ----------
        path : str
            Path relative to the bundle prefix.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        HttpError
            404 when the file does not exist.
        """
        def load():
            try:
                return self._read(path)
            except FileNotFoundError:
                raise HttpError(404, 'run not found')

        return self.files.get(path, load)

    def run_file(self, scope: str, run_id: str, part: str, mode: Optional[str] = None) -> bytes:
        """
        Bytes of a JSON part of a run.

        Parameters
        ----------
        scope : str
            Scope code.
        run_id : str
            Run id.
        part : str
            Part name.
        mode : str, optional
            Simulation mode, for the mode parts.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 when the run or the part does not exist.
        """
        return self._run_bytes(bundle.path_part(scope, run_id, part, mode))

    def run_csv(self, scope: str, run_id: str, name: str, mode: Optional[str] = None) -> bytes:
        """
        Bytes of a CSV twin of a part of a run.

        Parameters
        ----------
        scope : str
            Scope code.
        run_id : str
            Run id.
        name : str
            Part name.
        mode : str, optional
            Simulation mode, for the mode parts.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 when the run or the CSV does not exist.
        """
        return self._run_bytes(bundle.path_csv(scope, run_id, name, mode))

    def scopes(self) -> dict:
        """
        Scopes catalogue crossed with the manifest.

        Returns
        -------
        dict
            ``{'contract': int, 'scopes': [row, ...]}`` with one row per catalogue scope, in
            catalogue order. Rows of scopes without a manifest entry are not simulable and
            their run fields are ``None``.
        """
        published = self.manifest_data()['scopes']
        fields = ('run_at', 'event_date', 'as_of', 'date_last', 'n_polls')
        rows = []

        for code, scope in catalogue().iterrows():
            entry = published.get(code)
            row = {
                'code': code,
                'name': scope['name'],
                'parent': None if pd.isna(scope['parent']) else scope['parent'],
                'seats': int(scope['seats']),
                'simulable': entry is not None,
                'latest': entry['latest'] if entry is not None else None,
            }

            for field in fields:
                row[field] = entry.get(field) if entry is not None else None

            rows.append(row)

        return {'contract': bundle.CONTRACT, 'scopes': rows}


def site() -> Site:
    """
    Site service of the process, created on first use and kept in the app.

    Returns
    -------
    Site
        The shared instance.

    Raises
    ------
    HttpError
        503 when the app has no file system.
    """
    app = App.get_()
    instance = app.site

    if instance is None:
        if app.fs is None:
            raise HttpError(503, 'no file system configured')

        conf = settings()
        instance = Site(BundleReader(app.fs, prefix=conf['prefix']), ttl=conf['cache_ttl'])
        app.set('site', instance)

    return instance


ROUTES = [
    ('/api/v1/health', 'forecast', 'health', ['GET']),
    ('/api/v1/manifest', 'forecast', 'manifest', ['GET']),
    ('/api/v1/scopes', 'forecast', 'scopes', ['GET']),
    ('/api/v1/forecast/{scope}', 'forecast', 'meta', ['GET']),
    ('/api/v1/forecast/{scope}/runs', 'forecast', 'runs', ['GET']),
    ('/api/v1/forecast/{scope}/{part}', 'forecast', 'part', ['GET']),
    ('/api/v1/forecast/{scope}/{mode:str}/{part}', 'forecast', 'mode_part', ['GET']),
]


def add_routes(router) -> None:
    """
    Register the web API routes in a router.

    Parameters
    ----------
    router : Router
        Router that receives every route of ``ROUTES``, in order.

    Returns
    -------
    None
    """
    for route in ROUTES:
        router.add_route(*route)
