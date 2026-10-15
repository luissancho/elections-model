"""
Clean URLs of the profile of a pollster (``/casas/{slug}``) and of the detail of an election
(``/elecciones/{country}/{scope}/{month}``): the slugs of the names, the links built from them and the resolution
of a path back to the pollster id or to the scope and date. The old query URLs (``/casas?id=``,
``/elecciones?scope=&date=``) redirect to these (Luis, 2026-10-15).
"""
import re
import unicodedata
from typing import Optional

from ...core.api import HttpError
from ..webapi import check_date, query_pollsters, site
from .common import resolve_scope

# Segment of the national scope: the general elections, rather than the name of the country
NATIONAL_SLUG = 'generales'
PROFILE_PATH = '/casas'
DETAIL_PATH = '/elecciones'
_MONTH_RE = re.compile(r'[0-9]{4}-[0-9]{2}')
_NON_SLUG_RE = re.compile(r'[^a-z0-9]+')


def slugify(text: Optional[str]) -> str:
    """
    URL slug of a name: ASCII letters and digits in lower case, joined with hyphens.

    Parameters
    ----------
    text : str or None
        A pollster or scope name (``'Simple Lógica'``, ``'Castilla y León'``).

    Returns
    -------
    str
        The slug (``'simple-logica'``, ``'castilla-y-leon'``): accents dropped, every other run of characters
        turned into one hyphen, none at the ends; ``''`` for a name without letters or digits, or not a string.
    """
    if not isinstance(text, str):
        return ''

    ascii_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    return _NON_SLUG_RE.sub('-', ascii_text.lower()).strip('-')


def scope_slug(code: str, name: str) -> str:
    """
    Segment of a scope in the URL of an election.

    Parameters
    ----------
    code : str
        Scope code (``es``, ``es-md``).
    name : str
        Its catalogue name.

    Returns
    -------
    str
        ``NATIONAL_SLUG`` for a country (a code without region), the slug of the name otherwise (``madrid``).
    """
    return NATIONAL_SLUG if '-' not in code else slugify(name)


def scope_paths() -> dict:
    """
    Path of the elections of every catalogue scope.

    Returns
    -------
    dict
        ``{code: '/elecciones/{country}/{slug}'}`` (``{'es': '/elecciones/es/generales', 'es-md':
        '/elecciones/es/madrid'}``), the country being the first part of the code.
    """
    return {row['code']: '{}/{}/{}'.format(DETAIL_PATH, row['code'].split('-')[0], scope_slug(row['code'], row['name']))
            for row in site().scopes()['scopes']}


def event_segment(date: str, dates: list) -> str:
    """
    Last segment of the URL of an election.

    Parameters
    ----------
    date : str
        Date of the election (``YYYY-MM-DD``).
    dates : list of str
        Dates of every published election of its scope.

    Returns
    -------
    str
        The month (``YYYY-MM``) or, when another published election of the scope falls in the same month, the
        whole date.
    """
    month = date[:7]

    return month if sum(1 for other in dates if other[:7] == month) <= 1 else date


def event_href(scope: str, date: str, dates: Optional[list] = None) -> str:
    """
    Link to the detail of a published election.

    Parameters
    ----------
    scope : str
        Scope code.
    date : str
        Date of the election.
    dates : list of str, optional
        Published dates of the scope (``site().events(scope)`` by default).

    Returns
    -------
    str
        ``'/elecciones/es/generales/2023-07'`` (``event_segment``).
    """
    dates = site().events(scope) if dates is None else dates

    return '{}/{}'.format(scope_paths()[scope], event_segment(date, dates))


def resolve_event_path(country: str, slug: str, segment: str) -> tuple:
    """
    Election of a detail route.

    Parameters
    ----------
    country : str
        First segment of the route (``es``).
    slug : str
        Scope segment (``generales``, ``madrid``).
    segment : str
        Month (``YYYY-MM``) or date (``YYYY-MM-DD``) of the election.

    Returns
    -------
    tuple of str
        ``(scope, date)``: the scope code and the published date.

    Raises
    ------
    HttpError
        404 ``unknown scope`` when no catalogue scope has that path; 404 ``scope not published`` and 503 as
        ``resolve_scope``; 400 ``invalid date`` for a segment that is not a month or a date; 404 ``event not
        published`` when no published election matches, or when two fall in the month asked for.
    """
    prefix = '{}/{}/{}'.format(DETAIL_PATH, country, slug)
    codes = [code for code, path in scope_paths().items() if path == prefix]

    if not codes:
        raise HttpError(404, 'unknown scope')

    scope = resolve_scope(codes[0])
    dates = site().events(scope)

    if _MONTH_RE.fullmatch(segment):
        check_date(segment + '-01')
        matches = [date for date in dates if date.startswith(segment)]
    else:
        date = check_date(segment)
        matches = [date] if date in dates else []

    if len(matches) != 1:
        raise HttpError(404, 'event not published')

    return scope, matches[0]


def pollster_href(name: Optional[str], pollster_id: int) -> str:
    """
    Link to the profile of a pollster.

    Parameters
    ----------
    name : str or None
        Its name.
    pollster_id : int
        Its id, the fallback of a name without a slug.

    Returns
    -------
    str
        ``'/casas/gad3'``, or ``'/casas/{id}'`` when the name gives no slug (``resolve_pollster`` accepts both).
    """
    return '{}/{}'.format(PROFILE_PATH, slugify(name) or pollster_id)


def pollster_slugs() -> dict:
    """
    Slug of every pollster of the catalogue.

    Returns
    -------
    dict
        ``{slug: id}``; two names with the same slug keep the lowest id (the other profile stays reachable by its
        number), a name without a slug has none.
    """
    frame = query_pollsters()
    slugs = {}

    for pollster_id, name in sorted(zip(frame['id'].astype(int), frame['name']), key=lambda pair: pair[0]):
        slug = slugify(name)

        if slug:
            slugs.setdefault(slug, int(pollster_id))

    return slugs


def resolve_pollster(segment: str) -> int:
    """
    Pollster of a profile route.

    Parameters
    ----------
    segment : str
        Slug of the pollster (``gad3``) or its id.

    Returns
    -------
    int
        The pollster id.

    Raises
    ------
    HttpError
        404 ``pollster not found`` for an unknown slug or number; 503 without a database.
    """
    if segment.isdigit():
        pollster_id = int(segment)

        if pollster_id in set(query_pollsters()['id'].astype(int)):
            return pollster_id
    else:
        pollster_id = pollster_slugs().get(segment) if segment else None

        if pollster_id is not None:
            return pollster_id

    raise HttpError(404, 'pollster not found')
