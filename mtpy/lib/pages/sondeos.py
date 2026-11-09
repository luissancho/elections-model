"""
The page of a poll: the lookup of the poll named by the URL and the average of the day it was published.
"""
from typing import Optional

from ...core.api import HttpError
from .slugs import slugify


def find_poll(polls: list, date: str, pollster: str, sponsor: Optional[str] = None) -> dict:
    """
    Resolve the poll that a detail URL names.

    Parameters
    ----------
    polls : list
        Records of the ``polls`` part of a scope.
    date : str
        Day of the poll, ``YYYY-MM-DD``.
    pollster : str
        House segment of the URL: the slug of its name or its id.
    sponsor : str, optional
        Sponsor segment of the URL (the slug of the sponsor). Without it, the polls of the house that day without a
        sponsor slug are preferred, and if there are none, all the polls of the house that day.

    Returns
    -------
    dict
        The poll record, when exactly one poll matches.

    Raises
    ------
    HttpError
        404 ``poll not found`` when no poll or more than one matches.
    """
    day_polls = [poll for poll in polls
                 if str(poll.get('date'))[:10] == date
                 and pollster in {slugify(poll.get('pollster')), str(poll.get('pollster_id'))}]

    if sponsor is not None:
        candidates = [poll for poll in day_polls if slugify(poll.get('sponsor') or '') == sponsor]
    else:
        candidates = [poll for poll in day_polls if not slugify(poll.get('sponsor') or '')] or day_polls

    if len(candidates) != 1:
        raise HttpError(404, 'poll not found')

    return candidates[0]


def series_at(series: dict, date: str) -> dict:
    """
    Average of the last day of a series up to a date.

    Parameters
    ----------
    series : dict
        A ``series`` part: ``dates`` (ISO, ascending), ``parties`` and ``mean`` (``{party: [values]}``).
    date : str
        Day to look up, ``YYYY-MM-DD``.

    Returns
    -------
    dict
        ``{party: mean}`` of the last day of the series that is not after ``date`` (``None`` for a party without a
        value that day); ``{}`` when the series has no such day.
    """
    index = None

    for i, day in enumerate(series.get('dates') or []):
        if day <= date:
            index = i

    if index is None:
        return {}

    mean = series.get('mean') or {}

    return {name: mean[name][index] if name in mean and index < len(mean[name]) else None
            for name in series.get('parties') or []}
