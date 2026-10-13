"""Shared pieces of the publication package: constants, provenance, the publication guards and scope helpers."""
import os
import platform
import re
import subprocess
import warnings
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from ...models.elections import Polls
from ..data import get_next_event_date, get_scopes


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


PCT_DECIMALS = 2


PROB_DECIMALS = 3


SEATS_DECIMALS = 1


PROVENANCE_PACKAGES = ('numpy', 'pandas', 'scipy', 'statsmodels')


ISO_DAY_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def records(frame, index_name='name'):
    """Convert a frame into a list of row dicts, with its index as a leading column.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to convert.
    index_name : str
        Name given to the index column.

    Returns
    -------
    list of dict
        One dict per row; NaN values are kept (the bundle writer turns them into null).
    """
    return frame.rename_axis(index_name).reset_index().to_dict(orient='records')


def round_cols(frame, pct=(), prob=(), seats=()):
    """Round the present columns of each group to its number of decimals.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to round (not modified).
    pct, prob, seats : iterable of str
        Column names rounded to ``PCT_DECIMALS``, ``PROB_DECIMALS`` and ``SEATS_DECIMALS``.

    Returns
    -------
    pandas.DataFrame
        A rounded copy.
    """
    out = frame.copy()
    for cols, decimals in ((pct, PCT_DECIMALS), (prob, PROB_DECIMALS), (seats, SEATS_DECIMALS)):
        present = [c for c in cols if c in out.columns]
        if present:
            out[present] = out[present].round(decimals)
    return out


def _git(args, cwd):
    """Run a git command and return its decoded output, or ``None`` if it fails."""
    try:
        return subprocess.check_output(['git', *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def provenance(cwd: Optional[str] = None) -> dict:
    """Collect the code version and the library versions of the run.

    Parameters
    ----------
    cwd : str, optional
        Directory where git commands run.

    Returns
    -------
    dict
        ``commit`` (short SHA, ``GIT_COMMIT`` env var or ``None``), ``dirty`` (uncommitted changes;
        ``False`` if git is unavailable) and ``versions`` (Python and main libraries).
    """
    sha, porcelain = _git(['rev-parse', '--short', 'HEAD'], cwd), _git(['status', '--porcelain'], cwd)
    versions = {'python': platform.python_version()}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    return {'commit': sha or os.getenv('GIT_COMMIT') or None, 'dirty': bool(porcelain),
            'versions': versions}


def _iso_dates(values) -> pd.Series:
    """Return dates (strings, timestamps or datetimes) as ``YYYY-MM-DD`` strings."""
    return pd.to_datetime(pd.Series(values)).dt.strftime('%Y-%m-%d')


ATTRIBUTION = {'polls': 'Sondeos: Wikipedia, CC BY-SA 4.0', 'results': 'Origen de los datos: Ministerio del Interior',
               'model': 'Modelo y curación: Luis Sancho'}


class PublishRefused(Exception):
    """The publication was refused by a guard (for instance the LOREG window)."""


class NothingToPublish(ValueError):
    """The scope has no polls or its average cannot be fitted, so there is nothing to publish."""


def _madrid_tz():
    """Return the Europe/Madrid time zone, or a fixed UTC+1 offset (with a warning) without tzdata.

    Returns
    -------
    datetime.tzinfo
        ``ZoneInfo('Europe/Madrid')``, or ``timezone(timedelta(hours=1))`` when the time zone
        database is missing (slim Docker images may lack ``tzdata``).
    """
    try:
        return ZoneInfo('Europe/Madrid')
    except ZoneInfoNotFoundError:
        warnings.warn('Europe/Madrid time zone not found (install tzdata); using a fixed UTC+1 offset',
                      stacklevel=2)
        return timezone(timedelta(hours=1))


def _today(today: Optional[str | date | datetime] = None) -> date:
    """Normalise the date taken as today; ``None`` is today in Europe/Madrid.

    Parameters
    ----------
    today : str, datetime.date or datetime.datetime, optional
        ``YYYY-MM-DD`` string, date, or datetime reduced with ``.date()``.

    Returns
    -------
    datetime.date
        The date.
    """
    if today is None:
        return datetime.now(_madrid_tz()).date()
    if isinstance(today, datetime):
        return today.date()
    if isinstance(today, str):
        return date.fromisoformat(today)
    return today


def loreg_guard(scope: str, event_date: str, today: Optional[str | date | datetime] = None,
                force: bool = False, national: Optional[str] = None) -> bool:
    """Refuse to publish polls in the five days before an election (LOREG art. 69.7).

    The window covers the day of the election and the five days before it, for the own election of
    the scope and for the national one, since the ban on polls is general in a national election.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str
        Election date of the scope, ``YYYY-MM-DD``.
    today : str, datetime.date or datetime.datetime, optional
        Current date (``YYYY-MM-DD`` string, date, or datetime reduced with ``.date()``); defaults
        to today in Europe/Madrid.
    force : bool
        Publish even inside the window.
    national : str, optional
        Date (``YYYY-MM-DD``) of the next election of ``es``, or ``None`` when there is none.

    Returns
    -------
    bool
        Whether the date falls inside the window of the own election or of the national one.

    Raises
    ------
    PublishRefused
        Inside the window and not forced; the message names the election that closes the window
        (the own one first).
    """
    today = _today(today)
    guard = None
    for guard_scope, guard_date in ((scope, event_date), ('es', national)):
        if guard_date is not None and 0 <= (date.fromisoformat(guard_date) - today).days <= 5:
            guard = (guard_scope, guard_date)
            break
    if guard is not None and not force:
        raise PublishRefused('{} {}: inside the LOREG window (art. 69.7) of {} {}; pass force=true to publish'.format(
            scope, event_date, *guard))
    return guard is not None


def resolve_scopes(scopes, catalogue: Optional[pd.DataFrame] = None) -> list:
    """Expand the requested scopes into a list of scope codes.

    Parameters
    ----------
    scopes : str or sequence of str
        ``'all'`` (``es`` plus every scope with a parent), one scope code or a sequence of them.
    catalogue : pandas.DataFrame, optional
        Scope catalogue indexed by ``scode`` with a ``parent`` column; defaults to ``get_scopes()``.

    Returns
    -------
    list of str
        Scope codes to publish.

    Raises
    ------
    ValueError
        Some scope is not in the catalogue.
    """
    catalogue = get_scopes() if catalogue is None else catalogue
    if isinstance(scopes, str):
        if scopes == 'all':
            scopes = ['es'] + [s for s in catalogue.index if s != 'es' and pd.notnull(catalogue.loc[s, 'parent'])]
        else:
            scopes = [scopes]
    else:
        scopes = list(scopes)
    unknown = [s for s in scopes if s not in catalogue.index]
    if unknown:
        raise ValueError('publish: unknown scopes {}'.format(sorted(unknown)))
    return scopes


def next_event_date(scope: str) -> Optional[str]:
    """Return the date of the next event of the scope, or ``None`` when there is none.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    str or None
        ``YYYY-MM-DD`` date.
    """
    return get_next_event_date(scope)


def db_stats(scope: str, event_date: str) -> tuple:
    """Count the polls of the event in the database and find the date of the latest one.

    Parameters
    ----------
    scope, event_date : str
        Event scope and date.

    Returns
    -------
    tuple
        ``(number of polls, 'YYYY-MM-DD' of the latest)``; ``(0, None)`` when there are none.
    """
    polls = Polls().get_results(query=dict(filters=["event_scope = '{}'".format(scope),
                                                    "event_date = '{}'".format(event_date)]), formatted=True)
    if polls.shape[0] == 0:
        return 0, None
    return int(polls.shape[0]), polls['date'].max().strftime('%Y-%m-%d')


def _weighted_mean(values, weights) -> Optional[float]:
    """Mean of ``values`` weighted by ``weights`` rounded to ``PCT_DECIMALS``; ``None`` when nothing weighs."""
    values = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    weights = np.asarray(pd.to_numeric(weights, errors='coerce'), dtype=float)
    ok = ~np.isnan(values) & ~np.isnan(weights)
    if not ok.any() or weights[ok].sum() <= 0:
        return None

    return round(float(np.average(values[ok], weights=weights[ok])), PCT_DECIMALS)


def _iso_day(value) -> str:
    """Return a ``YYYY-MM-DD`` string as such, raising ``ValueError`` if it is not a real date."""
    if not isinstance(value, str) or not ISO_DAY_RE.fullmatch(value):
        raise ValueError(f'invalid event date {value!r} (expected YYYY-MM-DD)')

    return date.fromisoformat(value).isoformat()


def latest_entry(headline: dict) -> dict:
    """Build the manifest entry of a scope from the headline of its latest run.

    Parameters
    ----------
    headline : dict
        Data of ``headline.json``.

    Returns
    -------
    dict
        Entry with ``latest``, ``run_at``, ``event_date``, ``as_of``, ``date_last`` and ``n_polls``.
    """
    return {'latest': headline['run_id'], 'run_at': headline['run_at'], 'event_date': headline['event_date'],
            'as_of': headline['as_of'], 'date_last': headline['date_last'], 'n_polls': headline['n_polls']}
