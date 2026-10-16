"""Database queries of the web API: lazy readers over ``mtpy/lib/data.py`` and the ``query_*`` functions."""
import traceback
from typing import Any, Callable, Optional

import pandas as pd

from ...core.api import HttpError
from ...core.app import App
from .. import bundle
from .caches import DEFAULT_TTL, TTLCache, settings
from .checks import DEFAULT_LIMIT

QUERY_CACHE_SIZE = 128
WEB_DROP_CTYPES = ('wban', 'exit')  # polls published during the ban and exit polls never reach the web


def _reader(name: str) -> Callable:
    """
    Proxy of a reader of ``mtpy/lib/data.py`` that imports the module on its first call.

    The proxies are module attributes of ``webapi.queries`` (``get_ratings``, ...), looked up at call
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
        from .. import data

        return getattr(data, name)(*args, **kwargs)

    proxy.__name__ = name
    proxy.__doc__ = 'Lazy proxy of ``mtpy.lib.data.{}``.'.format(name)

    return proxy


get_parties = _reader('get_parties')
get_pollsters = _reader('get_pollsters')
get_ratings = _reader('get_ratings')
get_poll_series = _reader('get_poll_series')
get_event_results = _reader('get_event_results')
get_event_dates = _reader('get_event_dates')
get_house_effects = _reader('get_house_effects')
get_pollster_ratings = _reader('get_pollster_ratings')
get_pollster_events = _reader('get_pollster_events')
get_herding = _reader('get_herding')
get_drift = _reader('get_drift')
get_next_event_date = _reader('get_next_event_date')
get_events = _reader('get_events')
get_pollsters_quality = _reader('get_pollsters_quality')

_QUERIES = TTLCache(DEFAULT_TTL, maxsize=QUERY_CACHE_SIZE)


def database_host() -> Optional[str]:
    """
    Host of the configured database adapter (``config.db.params.host``).

    ``mtpy.run()`` always builds the adapter, so an empty ``DB_HOST`` is what tells a process
    without a database apart from one whose database is down.

    Returns
    -------
    str or None
        The host, or ``None`` when the configuration, its ``db`` section or the host is
        missing or empty.
    """
    config = getattr(App.get_(), 'config', None)
    section = getattr(config, 'db', None) if config is not None else None
    params = getattr(section, 'params', None) if section is not None else None
    host = getattr(params, 'host', None) if params is not None else None

    return str(host) if host not in (None, '') else None


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
        503 ``no database configured`` when the app has no database adapter or the adapter has
        no host (``database_host``).
    """
    db = App.get_().db

    if db is None or database_host() is None:
        raise HttpError(503, 'no database configured')

    return db


def _query(key: tuple, loader: Callable[[], Any]) -> Any:
    """
    Run a database read through the query cache.

    Only bounded reads are cached: whole per-scope tables, the catalogues, the polls of an
    event that exists in the scope and the results of an election (every region). Row filters
    chosen by the request (``event`` of the ratings, ``pollster``, ``limit``, ``region``) are
    applied on the cached frame, outside the cache. The cache has a TTL and an LRU cap of
    ``QUERY_CACHE_SIZE``.

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


def query_pollsters_quality() -> pd.DataFrame:
    """
    Replica of the public quality rubric of the pollsters.

    Returns
    -------
    pandas.DataFrame
        ``data.get_pollsters_quality()``: one row per evaluated pollster (empty before the replication).
    """
    return _query(('pollsters-quality',), lambda: get_pollsters_quality())


def query_pollster(pollster_id: int, scope: str, every_scope: bool = False) -> dict:
    """
    Profile of a pollster, composed from the cached tables.

    Parameters
    ----------
    pollster_id : int
        Validated pollster id.
    scope : str
        Validated scope code of the ratings.
    every_scope : bool, optional
        Take the house effects and the herding of every scope (the global profile) instead of those of
        ``scope``.

    Returns
    -------
    dict
        ``{'pollster': record, 'ratings': [...], 'house_effects': [...], 'herding': [...],
        'quality': record or None}``: the catalogue row of the pollster and its rows of the ratings (every
        event of the scope, the next one included, by date), house effects and herding tables, as JSON
        records; ``quality`` is its evaluation of the rubric, or ``None`` when it is not evaluated.

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

    tables = None if every_scope else scope
    quality = _rows_of(query_pollsters_quality(), 'pollster_id', pollster_id)

    return {
        'pollster': frame_to_records(pollster)[0],
        'ratings': frame_to_records(ratings),
        'house_effects': frame_to_records(_rows_of(query_house_effects(tables), 'pollster_id', pollster_id)),
        'herding': frame_to_records(_rows_of(query_herding(tables), 'pollster_id', pollster_id)),
        'quality': frame_to_records(quality)[0] if not quality.empty else None,
    }


def query_pollster_path(pollster_id: int) -> dict:
    """
    The path of a pollster's rating: its ratings in every scope and the elections it polled.

    Each table is read once per pollster and cached under the LRU cap (a bounded read: one pollster).

    Parameters
    ----------
    pollster_id : int
        Validated pollster id.

    Returns
    -------
    dict
        ``{'ratings': [...], 'events': [...]}``: the JSON records of ``data.get_pollster_ratings`` (every event of
        every scope, the next ones included) and of ``data.get_pollster_events`` (``event_scope``, ``event_date``
        and ``n_polls``).
    """
    ratings = _query(('pollster-ratings', pollster_id), lambda: get_pollster_ratings(pollster_id))
    events = _query(('pollster-events', pollster_id), lambda: get_pollster_events(pollster_id))

    return {'ratings': frame_to_records(ratings), 'events': frame_to_records(events)}


def ratings_event(scope: str, event: Optional[str] = None) -> Optional[str]:
    """
    Event whose ratings are served: the requested one or, by default, the latest event of the
    ratings, which is the next election.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str, optional
        Validated event date.

    Returns
    -------
    str or None
        The date, ``YYYY-MM-DD``; ``None`` by default when the scope has no ratings.

    Raises
    ------
    HttpError
        404 ``event not found`` when ``event`` is not an event of the scope.
    """
    if event is not None:
        return check_event(scope, event)

    ratings = _ratings(scope)

    if ratings.empty or 'event_date' not in ratings.columns:
        return None

    latest = _event_dates(ratings).max()

    return None if pd.isna(latest) else latest.strftime('%Y-%m-%d')


def query_ratings(scope: str, event: Optional[str] = None) -> pd.DataFrame:
    """
    Pollster ratings of an event of a scope.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str, optional
        Validated event date; by default the latest event of the ratings, which is the next
        election (``ratings_event``).

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
    target = ratings_event(scope, event)

    if ratings.empty or target is None:
        return ratings

    return ratings.loc[_event_dates(ratings) == pd.Timestamp(target)].reset_index(drop=True)


def polls_event(scope: str, event: Optional[str] = None) -> Optional[str]:
    """
    Event whose polls are served: the requested one or, by default, the next election.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str, optional
        Validated event date.

    Returns
    -------
    str or None
        The date; ``None`` by default when the scope has no next election.

    Raises
    ------
    HttpError
        404 ``event not found`` when ``event`` is not an event of the scope.
    """
    return _next_event(scope) if event is None else check_event(scope, event)


def query_polls(scope: str, event: Optional[str] = None, pollster: Optional[int] = None,
                limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """
    Polls of an event of a scope, one row per poll with a column per party.

    The reader is ``data.get_poll_series`` (the poll fields, its weights and errors, and the
    ``pct`` of each party as a column); ``get_poll_results`` returns one row per poll and
    party instead. The polls of the event are cached once; ``pollster`` and ``limit`` are
    applied on that frame. Polls tagged with a ``WEB_DROP_CTYPES`` context (published during
    the ban, exit polls) are dropped at the source.

    Parameters
    ----------
    scope : str
        Validated scope code.
    event : str, optional
        Validated event date; by default the next election of the scope (``polls_event``).
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
    date = polls_event(scope, event)

    if date is None:
        return pd.DataFrame()

    polls = _query(('polls', scope, date), lambda: get_poll_series(scope, date, drop_contexts=list(WEB_DROP_CTYPES)))

    if pollster is not None:
        polls = _rows_of(polls, 'pollster_id', pollster)
    if 'date' in polls.columns:
        polls = polls.sort_values('date', ascending=False, kind='stable')

    return polls.head(limit).reset_index(drop=True)


def _election_dates(scope: str) -> list:
    """
    Dates of the elections of a scope, cached.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    list of str
        ``data.get_event_dates(scope)``.
    """
    return _query(('events', scope), lambda: get_event_dates(scope))


def query_results(scope: str, date: str, region: int = 0) -> pd.DataFrame:
    """
    Results of an election, by party, in a region of a scope.

    The results of the election are read once for every region (``get_event_results`` without
    a region) and cached per election; ``region`` is applied on that frame, outside the cache.
    The date is checked first against the cached elections of the scope, so an arbitrary date
    never reads the results.

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
        The rows of ``data.get_event_results(scope, date)`` of the region, with a fresh index;
        empty when the region has none.

    Raises
    ------
    HttpError
        404 ``event not found`` when ``date`` is not an election of the scope.
    """
    if date not in _election_dates(scope):
        raise HttpError(404, 'event not found')

    results = _query(('results', scope, date), lambda: get_event_results(scope, date, None))

    return _rows_of(results, 'region_id', region)


def query_house_effects(scope: Optional[str]) -> pd.DataFrame:
    """
    House effects table (``pollsters_parties``) of a scope, or of every scope.

    Parameters
    ----------
    scope : str or None
        Validated scope code; ``None`` for the rows of every scope (``event_scope`` tells them apart).

    Returns
    -------
    pandas.DataFrame
        ``data.get_house_effects(scope)``: one row per event, pollster and party.
    """
    return _query(('house_effects', scope), lambda: get_house_effects(scope))


def query_herding(scope: Optional[str]) -> pd.DataFrame:
    """
    Herding table (``pollsters_herding``) of a scope, or of every scope.

    Parameters
    ----------
    scope : str or None
        Validated scope code; ``None`` for the rows of every scope (``event_scope`` tells them apart).

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


def query_events(scope: str) -> pd.DataFrame:
    """
    Elections table of a scope (``date``, ``scope``, ``name``, ``featured``), cached.

    Parameters
    ----------
    scope : str
        Validated scope code.

    Returns
    -------
    pandas.DataFrame
        ``data.get_events(scope)``.
    """
    return _query(('events_table', scope), lambda: get_events(scope))


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
