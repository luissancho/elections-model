"""Request validators of the web API, the scope catalogue they rely on and the parameters of the elections."""
import datetime
import json
import re
from typing import Optional, Sequence

import pandas as pd

from ...core.api import HttpError
from ...core.app import App
from .. import bundle
from .caches import DEFAULT_TTL, TTLCache, settings

DEFAULT_LIMIT = 1000
MAX_LIMIT = 5000
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
INT_RE = re.compile(r'\d{1,6}')
FORMATS = ('json', 'csv')
ANALYSIS_PARTS = bundle.ANALYSIS_PARTS
BACKTEST_PARTS = bundle.BACKTEST_PARTS
EVENT_PARTS = bundle.EVENT_PARTS
CSV_PARTS = bundle.RUN_PARTS[2:]
MODE_PARTS = bundle.MODE_PARTS
RUN_PARTS = bundle.RUN_PARTS
MODES = bundle.MODES


_CATALOGUE = TTLCache(DEFAULT_TTL)
_PARAMS = TTLCache(DEFAULT_TTL)


def event_params(scope: str, date: str) -> dict:
    """
    Parameters of an election in ``data/params.json``, the file ``data.get_event_params`` reads.

    The file is read through a TTL cache; without a data directory or the file, no election has parameters.

    Parameters
    ----------
    scope : str
        Validated scope code.
    date : str
        Validated election date.

    Returns
    -------
    dict
        The entry of the election (``parties``, ``bmaps``, ``smap``...), or ``{}`` when the file or the
        entry is missing.
    """
    data = App.get_().data

    if data is None:
        return {}

    _PARAMS.ttl = settings()['cache_ttl']

    def load():
        return json.loads(data.read('params.json')) if data.exists('params.json') else {}

    params = _PARAMS.get('params', load)
    entry = (params.get(scope) or {}).get(date) if isinstance(params, dict) else None

    return entry if isinstance(entry, dict) else {}


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
