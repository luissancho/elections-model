"""
Home page context: the index of the published events and the past elections.
"""
from datetime import date
from typing import Optional

from ..webapi import site
from .common import (block_color, bundle_part, catalog, common_context, days_left, days_text, default_mode,
                     event_title, published_scopes)
from .elecciones import published_events

INTRO = ('Un modelo que promedia las encuestas y simula los escaños de cada elección: elige una para ver en qué '
         'situación está.')


def card_headline(row: dict, mode: str = 'nowcast') -> tuple:
    """
    Leading party and leading majority of the latest run of a scope.

    Parameters
    ----------
    row : dict
        Row of ``published_scopes()`` (``code``, ``latest``).
    mode : str, optional
        ``'nowcast'`` (default) or ``'forecast'``: the estimate of the headline the card reads.

    Returns
    -------
    tuple
        ``(leader, majority)``: ``leader`` is ``{'name', 'fullname', 'color', 'pct', 'seats'}`` for the party
        with the highest ``pct`` of the estimate of ``mode``, ``majority`` is ``{'block', 'p', 'color'}`` for the
        block with the highest ``p_majority``; each one is ``None`` when the run has no headline (or no data for it).
    """
    code, run = row['code'], row['latest']
    headline = bundle_part(site().run_file, code, run, 'headline')

    if not headline:
        return None, None

    estimate = headline.get(mode) or {}
    meta = bundle_part(site().run_file, code, run, 'meta') or {}
    parties = catalog(meta)
    leader = majority = None
    rows = estimate.get('parties') or []

    if rows:
        top = max(rows, key=lambda party: party.get('pct') or 0)
        entry = parties[top['name']]
        leader = {'name': top['name'], 'fullname': entry['fullname'], 'color': entry['color'],
                  'pct': top.get('pct'), 'seats': top.get('seats')}

    probs = estimate.get('p_majority') or {}

    if probs:
        block = max(probs, key=probs.get)
        majority = {'block': block, 'p': probs[block], 'color': block_color(block, meta, parties)}

    return leader, majority


def event_cards(scopes: list, today: Optional[date] = None) -> list:
    """
    One card per catalogue scope: the published events with their headline, the rest as "not published".

    Parameters
    ----------
    scopes : list of dict
        Every row of ``site().scopes()['scopes']`` (published or not).
    today : datetime.date, optional
        Reference day for the days left.

    Returns
    -------
    list of dict
        ``es`` first, then the published scopes by ascending ``event_date``, then the unpublished ones by
        name; each with ``code``, ``title``, ``published``, ``event_date``, ``days_text``, ``run_at``,
        ``frozen``, ``mode``, ``href``, ``href_seats``, ``leader`` and ``majority`` (``card_headline``).
        The unpublished ones have no date, link or figures.
    """
    frozen = bool((site().manifest_data().get('freeze') or {}).get('active'))
    mode = default_mode(frozen)
    ordered = sorted(scopes, key=lambda row: (row['code'] != 'es', not row['simulable'],
                                              (row.get('event_date') or '') if row['simulable'] else row['name']))
    cards = []

    for row in ordered:
        code = row['code']
        published = bool(row['simulable'])
        leader, majority = card_headline(row, mode) if published else (None, None)
        cards.append({
            'code': code,
            'title': event_title(code, scopes),
            'published': published,
            'event_date': row.get('event_date') if published else None,
            'days_text': days_text(days_left(row.get('event_date'), today)) if published else '',
            'run_at': row.get('run_at') if published else None,
            'frozen': frozen,
            'mode': mode,
            'href': '/' + code if published else None,
            'href_seats': '/' + code + '/escanos' if published else None,
            'leader': leader,
            'majority': majority,
        })

    return cards


def past_events(scopes: list) -> list:
    """
    Past elections of every published scope, newest first.

    Parameters
    ----------
    scopes : list of dict
        Rows of ``published_scopes()``.

    Returns
    -------
    list of dict
        ``scope``, ``scope_name``, ``date``, ``name`` and ``href`` (the detail in ``/elecciones``) per election.
    """
    events = []

    for row in scopes:
        for event in published_events(row['code']):
            events.append({
                'scope': row['code'],
                'scope_name': row['name'],
                'date': event['date'],
                'name': event['name'],
                'href': '/elecciones?scope={}&date={}'.format(row['code'], event['date']),
            })

    return sorted(events, key=lambda event: event['date'], reverse=True)


def inicio_context(state: dict) -> dict:
    """
    Template context of the home page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).

    Returns
    -------
    dict
        ``common_context`` of the home (default scope) plus ``cards`` (``event_cards``), ``past``
        (``past_events``) and ``intro``.

    Raises
    ------
    HttpError
        503 without a bundle.
    """
    context = common_context(state, active='/')
    context.update(cards=event_cards(site().scopes()['scopes']), past=past_events(context['scopes']), intro=INTRO)

    return context
