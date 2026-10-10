"""
Service layer of the web API.

It serves the published bundle (``site/v1``: forecast runs and the houses analysis) and the
live database tables: request validators, small in-process caches, a reader that returns the
bundle files as bytes, untouched, and the ``query_*`` functions over the readers of
``mtpy/lib/data.py``. This module imports only ``bundle`` at load time; ``data`` (and with it
the database models) is imported on the first query, so the web workers that never touch the
database do not load it.
"""
import datetime
import re
import time
import traceback
import warnings
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
QUERY_CACHE_SIZE = 128
DEFAULT_LIMIT = 1000
MAX_LIMIT = 5000
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
INT_RE = re.compile(r'\d{1,6}')
FORMATS = ('json', 'csv')
ANALYSIS_PARTS = bundle.ANALYSIS_PARTS
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
        ``bundle.PREFIX`` and ``DEFAULT_TTL``; a ``cache_ttl`` that is not a number
        also falls back, with a warning.
    """
    config = getattr(App.get_(), 'config', None)
    web = getattr(config, 'web', None) if config is not None else None
    prefix = getattr(web, 'prefix', None)
    ttl = getattr(web, 'cache_ttl', None)

    try:
        cache_ttl = float(ttl) if ttl not in (None, '') else float(DEFAULT_TTL)
    except (TypeError, ValueError):
        warnings.warn('invalid web.cache_ttl {!r}; using {}'.format(ttl, DEFAULT_TTL))
        cache_ttl = float(DEFAULT_TTL)

    return {
        'prefix': str(prefix) if prefix not in (None, '') else bundle.PREFIX,
        'cache_ttl': cache_ttl,
    }


class TTLCache:
    """
    Cache whose entries expire a fixed time after being stored, optionally capped in size.

    An expired entry is replaced when its key is asked again; with ``maxsize`` the least
    recently used entry is evicted on insert, so keys shaped by requests cannot grow the
    cache without bound.

    Parameters
    ----------
    ttl : float
        Seconds an entry stays valid.
    clock : callable, optional
        Function returning the current time in seconds. Defaults to ``time.monotonic``.
    maxsize : int, optional
        Maximum number of entries; ``None`` (default) for no cap.
    """

    def __init__(self, ttl: float, clock: Callable[[], float] = time.monotonic, maxsize: Optional[int] = None):
        self.ttl = ttl
        self.clock = clock
        self.maxsize = maxsize
        self._items = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        """
        Number of stored entries, expired ones included.

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
            hit = self._items.get(key)

            if hit is not None and self.clock() - hit[0] < self.ttl:
                self._items.move_to_end(key)

                return hit[1]

        value = loader()

        with self._lock:
            self._items[key] = (self.clock(), value)
            self._items.move_to_end(key)

            while self.maxsize is not None and len(self._items) > self.maxsize:
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


def check_region(region: Optional[str]) -> Optional[int]:
    """
    Validate a district id.

    Parameters
    ----------
    region : str, optional
        ``region`` query parameter.

    Returns
    -------
    int or None
        The id as an integer, or ``None`` when it is missing or empty.

    Raises
    ------
    HttpError
        400 when it is not a number of one to six digits.
    """
    if region is None or region == '':
        return None

    if not isinstance(region, str) or not re.fullmatch(r'\d{1,6}', region):
        raise HttpError(400, 'invalid region')

    return int(region)


def check_date(value: Optional[str]) -> str:
    """
    Validate an ISO calendar date.

    Parameters
    ----------
    value : str
        Date from the request.

    Returns
    -------
    str
        The same date, as ``YYYY-MM-DD``.

    Raises
    ------
    HttpError
        400 when it is not a real date written as ``YYYY-MM-DD``.
    """
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        raise HttpError(400, 'invalid date')

    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        raise HttpError(400, 'invalid date')

    return value


def check_pollster(value: Optional[str]) -> int:
    """
    Validate a pollster id.

    Parameters
    ----------
    value : str
        Pollster id from the request.

    Returns
    -------
    int
        The id as an integer.

    Raises
    ------
    HttpError
        400 when it is not a number of one to six digits.
    """
    if not isinstance(value, str) or not INT_RE.fullmatch(value):
        raise HttpError(400, 'invalid pollster')

    return int(value)


def check_limit(value: Optional[str], default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    """
    Validate a maximum number of rows.

    Parameters
    ----------
    value : str or None
        ``limit`` query parameter.
    default : int, optional
        Limit used when the parameter is missing or empty.
    maximum : int, optional
        Largest limit accepted.

    Returns
    -------
    int
        The limit.

    Raises
    ------
    HttpError
        400 when it is not an integer between 1 and ``maximum``.
    """
    if value is None or value == '':
        return default

    if not isinstance(value, str) or not INT_RE.fullmatch(value) or not 1 <= int(value) <= maximum:
        raise HttpError(400, 'invalid limit')

    return int(value)


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

    def _analysis_bytes(self, path: str) -> bytes:
        """
        Read a file of a published analysis through the TTL cache.

        The analysis is overwritten on every publication, so it is cached like the pointers,
        not like the immutable run files.

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
                raise HttpError(404, 'analysis not published')

        return self.pointers.get(('analysis', path), load)

    def analysis(self, scope: str, part: str) -> bytes:
        """
        Bytes of a JSON part of the houses analysis of a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.
        part : str
            Part name, one of ``ANALYSIS_PARTS``.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 when the part is not published.
        """
        return self._analysis_bytes(bundle.path_analysis(scope, part))

    def analysis_csv(self, scope: str, name: str) -> bytes:
        """
        Bytes of a CSV twin of the houses analysis of a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.
        name : str
            CSV name, one of ``bundle.ANALYSIS_CSV``.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 when the CSV is not published.
        """
        return self._analysis_bytes(bundle.path_analysis_csv(scope, name))

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


def _reader(name: str) -> Callable:
    """
    Proxy of a reader of ``mtpy/lib/data.py`` that imports the module on its first call.

    The proxies are module attributes of ``webapi`` (``get_ratings``, ...), looked up at call
    time by the ``query_*`` functions, so tests replace them with ``monkeypatch.setattr``.

    Parameters
    ----------
    name : str
        Name of the reader in ``data``.

    Returns
    -------
    callable
        Function that forwards its arguments to ``data.<name>``.
    """
    def proxy(*args, **kwargs):
        from . import data

        return getattr(data, name)(*args, **kwargs)

    proxy.__name__ = name
    proxy.__doc__ = 'Lazy proxy of ``mtpy.lib.data.{}``.'.format(name)

    return proxy


get_parties = _reader('get_parties')
get_pollsters = _reader('get_pollsters')
get_ratings = _reader('get_ratings')
get_poll_series = _reader('get_poll_series')
get_event_results = _reader('get_event_results')
get_house_effects = _reader('get_house_effects')
get_herding = _reader('get_herding')
get_drift = _reader('get_drift')
get_next_event_date = _reader('get_next_event_date')

_QUERIES = TTLCache(DEFAULT_TTL, maxsize=QUERY_CACHE_SIZE)


def database() -> Any:
    """
    Database connection of the app.

    Returns
    -------
    object
        ``App.get_().db``.

    Raises
    ------
    HttpError
        503 when the app has no database.
    """
    db = App.get_().db

    if db is None:
        raise HttpError(503, 'no database configured')

    return db


def _query(key: tuple, loader: Callable[[], Any]) -> Any:
    """
    Run a database read through the query cache.

    Only bounded reads are cached: whole per-scope tables, the catalogues, the polls of an
    event that exists in the scope and the results of an election. Row filters chosen by the
    request (``event`` of the ratings, ``pollster``, ``limit``) are applied on the cached
    frame, outside the cache. The cache has a TTL and an LRU cap of ``QUERY_CACHE_SIZE``.

    Parameters
    ----------
    key : tuple
        Cache key: the read name followed by its validated arguments.
    loader : callable
        Function without arguments that reads the database.

    Returns
    -------
    object
        The cached or freshly read result.

    Raises
    ------
    HttpError
        503 ``no database configured`` before reading when the app has no database; 503
        ``database unavailable`` when the reader fails (the trace goes to the app logger and
        nothing is cached).
    """
    database()
    _QUERIES.ttl = settings()['cache_ttl']

    def load():
        try:
            return loader()
        except Exception as error:
            logger = App.get_().logger
            if logger is not None:
                logger.error('Database query {!r} failed: {!r}\n{}'.format(key, error, traceback.format_exc()))
            raise HttpError(503, 'database unavailable')

    return _QUERIES.get(key, load)


def _rows_of(frame: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """
    Rows of a frame whose column equals a value, with a fresh index.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table.
    column : str
        Column to filter on.
    value : object
        Value to keep.

    Returns
    -------
    pandas.DataFrame
        The matching rows; empty when the column is missing.
    """
    if column not in frame.columns:
        return frame.iloc[0:0]

    return frame.loc[frame[column] == value].reset_index(drop=True)


def _ratings(scope: str) -> pd.DataFrame:
    """
    Whole ratings table of a scope (every event, the next one included), cached.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    pandas.DataFrame
        ``data.get_ratings(scope)``.
    """
    return _query(('ratings', scope), lambda: get_ratings(scope))


def _next_event(scope: str) -> Optional[str]:
    """
    Date of the next election of a scope, cached.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    str or None
        ``data.get_next_event_date(scope)``.
    """
    return _query(('next_event', scope), lambda: get_next_event_date(scope))


def _event_dates(frame: pd.DataFrame) -> pd.Series:
    """
    Event dates of a table as timestamps.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table with an ``event_date`` column.

    Returns
    -------
    pandas.Series
        The dates, aligned with the rows.
    """
    return pd.to_datetime(frame['event_date'])


def check_event(scope: str, event: str) -> str:
    """
    Check that a validated date is an event of a scope: one in its ratings or its next election.

    The set of events comes from bounded cached reads, so an arbitrary date never creates a
    cache key nor a database read of its own.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str
        Date validated by ``check_date``.

    Returns
    -------
    str
        The same date.

    Raises
    ------
    HttpError
        404 ``event not found`` when the scope has no such event.
    """
    ratings = _ratings(scope)

    if 'event_date' in ratings.columns and (_event_dates(ratings) == pd.Timestamp(event)).any():
        return event

    if event == _next_event(scope):
        return event

    raise HttpError(404, 'event not found')


def query_parties() -> pd.DataFrame:
    """
    Parties catalogue.

    Returns
    -------
    pandas.DataFrame
        ``data.get_parties()``.
    """
    return _query(('parties',), lambda: get_parties())


def query_pollsters() -> pd.DataFrame:
    """
    Pollsters catalogue.

    Returns
    -------
    pandas.DataFrame
        ``data.get_pollsters()``: ``id``, ``name``, ``mtype``, ``quality``, ``rating``.
    """
    return _query(('pollsters',), lambda: get_pollsters())


def query_pollster(pollster_id: int, scope: str) -> dict:
    """
    Profile of a pollster in a scope, composed from the cached per-scope tables.

    Parameters
    ----------
    pollster_id : int
        Validated pollster id.
    scope : str
        Validated scope code.

    Returns
    -------
    dict
        ``{'pollster': record, 'ratings': [...], 'house_effects': [...], 'herding': [...]}``:
        the catalogue row of the pollster and its rows of the ratings (every event, the next
        one included, by date), house effects and herding tables of the scope, as JSON records.

    Raises
    ------
    HttpError
        404 ``pollster not found`` when the id is not in the catalogue.
    """
    pollster = _rows_of(query_pollsters(), 'id', pollster_id)

    if pollster.empty:
        raise HttpError(404, 'pollster not found')

    ratings = _rows_of(_ratings(scope), 'pollster_id', pollster_id)
    if 'event_date' in ratings.columns:
        ratings = ratings.sort_values('event_date', ignore_index=True)

    return {
        'pollster': frame_to_records(pollster)[0],
        'ratings': frame_to_records(ratings),
        'house_effects': frame_to_records(_rows_of(query_house_effects(scope), 'pollster_id', pollster_id)),
        'herding': frame_to_records(_rows_of(query_herding(scope), 'pollster_id', pollster_id)),
    }


def query_ratings(scope: str, event: Optional[str] = None) -> pd.DataFrame:
    """
    Pollster ratings of an event of a scope.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str, optional
        Validated event date; by default the latest event of the ratings, which is the next
        election.

    Returns
    -------
    pandas.DataFrame
        One row per pollster of the event, with a fresh index; empty when there is none.

    Raises
    ------
    HttpError
        404 ``event not found`` when ``event`` is not an event of the scope.
    """
    ratings = _ratings(scope)

    if event is not None:
        check_event(scope, event)

    if ratings.empty:
        return ratings

    dates = _event_dates(ratings)
    target = dates.max() if event is None else pd.Timestamp(event)

    return ratings.loc[dates == target].reset_index(drop=True)


def query_polls(scope: str, event: Optional[str] = None, pollster: Optional[int] = None,
                limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """
    Polls of an event of a scope, one row per poll with a column per party.

    The reader is ``data.get_poll_series`` (the poll fields, its weights and errors, and the
    ``pct`` of each party as a column); ``get_poll_results`` returns one row per poll and
    party instead. The polls of the event are cached once; ``pollster`` and ``limit`` are
    applied on that frame.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str, optional
        Validated event date; by default the next election of the scope.
    pollster : int, optional
        Validated pollster id; only its polls when given.
    limit : int, optional
        Validated maximum number of polls; the most recent ones are kept.

    Returns
    -------
    pandas.DataFrame
        The polls, most recent first; empty when the scope has no next election.

    Raises
    ------
    HttpError
        404 ``event not found`` when ``event`` is not an event of the scope.
    """
    date = _next_event(scope) if event is None else check_event(scope, event)

    if date is None:
        return pd.DataFrame()

    polls = _query(('polls', scope, date), lambda: get_poll_series(scope, date))

    if pollster is not None:
        polls = _rows_of(polls, 'pollster_id', pollster)
    if 'date' in polls.columns:
        polls = polls.sort_values('date', ascending=False, kind='stable')

    return polls.head(limit).reset_index(drop=True)


def query_results(scope: str, date: str, region: int = 0) -> pd.DataFrame:
    """
    Results of an election, by party, in a region of a scope.

    Parameters
    ----------
    scope : str
        Validated scope code.
    date : str
        Validated election date.
    region : int, optional
        Validated region id; 0 is the whole scope.

    Returns
    -------
    pandas.DataFrame
        ``data.get_event_results(scope, date, region)``.
    """
    return _query(('results', scope, date, region), lambda: get_event_results(scope, date, region))


def query_house_effects(scope: str) -> pd.DataFrame:
    """
    House effects table (``pollsters_parties``) of a scope.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    pandas.DataFrame
        ``data.get_house_effects(scope)``: one row per event, pollster and party.
    """
    return _query(('house_effects', scope), lambda: get_house_effects(scope))


def query_herding(scope: str) -> pd.DataFrame:
    """
    Herding table (``pollsters_herding``) of a scope.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    pandas.DataFrame
        ``data.get_herding(scope)``: one row per election cycle and pollster.
    """
    return _query(('herding', scope), lambda: get_herding(scope))


def query_drift(scope: str) -> pd.DataFrame:
    """
    Drift table of a scope.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    pandas.DataFrame
        ``data.get_drift(scope)``: one row per past election, party and horizon.
    """
    return _query(('drift', scope), lambda: get_drift(scope))


def frame_to_records(frame: pd.DataFrame) -> list:
    """
    Rows of a frame as JSON records.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table. A named index becomes leading columns; a default one is dropped.

    Returns
    -------
    list of dict
        One dict per row, with dates as ISO strings and missing values as ``None``.
    """
    unnamed = all(name is None for name in frame.index.names)
    flat = frame.reset_index(drop=unnamed)

    return bundle.jsonable(flat.to_dict(orient='records'))


def frame_to_csv(frame: pd.DataFrame) -> bytes:
    """
    A frame as a CSV download.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table.

    Returns
    -------
    bytes
        UTF-8 CSV with ``\\n`` line endings and no index.
    """
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8')


ROUTES = [
    ('/api/v1/health', 'forecast', 'health', ['GET']),
    ('/api/v1/manifest', 'forecast', 'manifest', ['GET']),
    ('/api/v1/scopes', 'forecast', 'scopes', ['GET']),
    ('/api/v1/parties', 'data', 'parties', ['GET']),
    ('/api/v1/pollsters', 'data', 'pollsters', ['GET']),
    ('/api/v1/pollsters/{id:num}', 'data', 'pollster', ['GET']),
    ('/api/v1/ratings', 'data', 'ratings', ['GET']),
    ('/api/v1/polls', 'data', 'polls', ['GET']),
    ('/api/v1/results/{scope}/{date}', 'data', 'results', ['GET']),
    ('/api/v1/house-effects', 'data', 'house_effects', ['GET']),
    ('/api/v1/herding', 'data', 'herding', ['GET']),
    ('/api/v1/drift', 'data', 'drift', ['GET']),
    ('/api/v1/analysis/{scope}/{part}', 'analysis', 'part', ['GET']),
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
