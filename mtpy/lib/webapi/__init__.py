"""
Service layer of the web API.

It serves the published bundle (``site/v1``: forecast runs and the houses analysis) and the
live database tables. The package has four submodules: ``caches`` (TTL and LRU caches and the
``web`` settings), ``checks`` (request validators and the scope catalogue), ``published`` (the
reader that returns the bundle files as bytes, untouched) and ``queries`` (the ``query_*``
functions over the readers of ``mtpy/lib/data.py``). Loading the package imports
only ``bundle``; ``data`` (and with it the database models) is imported on the first query,
so the web workers that never touch the database do not load it.
"""
from .caches import (
    DEFAULT_SITE_URL, DEFAULT_TTL, LRU_SIZE, LRUCache, TTLCache, settings,
)
from .checks import (
    ANALYSIS_PARTS, BACKTEST_PARTS, CSV_PARTS, DATE_RE, DEFAULT_LIMIT, EVENT_PARTS, FORMATS, INT_RE, MAX_LIMIT,
    MODE_PARTS, MODES, RUN_PARTS, catalogue, check_date, check_format, check_limit, check_mode, check_part,
    check_pollster, check_region, check_run, check_scope, scope_codes,
)
from .published import (
    IMMUTABLE_MAX_AGE, Site, site,
)
from .queries import (
    QUERY_CACHE_SIZE, check_event, database, database_host, frame_to_csv, frame_to_records, get_drift,
    get_event_dates, get_event_results, get_events, get_herding, get_house_effects, get_next_event_date,
    get_parties, get_poll_series, get_pollsters, get_ratings, polls_event, query_drift, query_events,
    query_herding, query_house_effects, query_parties, query_polls, query_pollster, query_pollsters,
    query_ratings, query_results, ratings_event,
)


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
    ('/api/v1/events/{scope}', 'events', 'index', ['GET']),
    ('/api/v1/events/{scope}/{date}/{part}', 'events', 'part', ['GET']),
    ('/api/v1/backtest/{scope}/{part}', 'backtest', 'part', ['GET']),
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
