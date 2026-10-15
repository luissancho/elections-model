"""
Home page context: the hero of the next election (the first published scope), the rows of the other published
scopes, the latest polls and the past elections.
"""
from datetime import date, datetime
from typing import Optional

from ..webapi import site
from .common import (CHAMBERS, block_color, bundle_part, catalog, common_context, days_left, days_text, default_mode,
                     event_title, hemicycle_order, published_scopes)
from .elecciones import last_results, published_events
from .filters import _madrid_tz, delta_text, fmt_date_short
from .promedios import lead_strip, leader_of, poll_sample
from .slugs import event_href

INTRO = ('Demoscopio promedia los sondeos de cada elección, simula el reparto de escaños y puntúa a las encuestadoras '
         'según lo que acertaron. Un proyecto transparente y divulgativo, con el método y modelo a la vista.')
# Long names of the chambers, for the featured card
CHAMBER_NAMES = {'es': 'Congreso de los Diputados'}
DEFAULT_CHAMBER_NAME = 'Parlamento'
# Past elections shown as chips before "+ N más"
PAST_SHOWN = 12
# Rows of "Últimas encuestas" and the parties whose figures they show
LATEST_POLLS = 10
TOP_PARTIES = 3


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
        name; each with ``code``, ``name`` (of the catalogue), ``run`` (the latest run), ``title``,
        ``published``, ``event_date``, ``days_text``, ``run_at``, ``frozen``, ``mode``, ``href``, ``href_seats``,
        ``leader`` and ``majority`` (``card_headline``). The unpublished ones have no date, run, link or figures.
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
            'name': row['name'],
            'run': row.get('latest') if published else None,
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
        ``scope``, ``scope_name``, ``date``, ``name`` and ``href`` (``event_href``: the detail of the election) per
        election.
    """
    events = []

    for row in scopes:
        dates = site().events(row['code'])

        for event in published_events(row['code']):
            events.append({
                'scope': row['code'],
                'scope_name': row['name'],
                'date': event['date'],
                'name': event['name'],
                'href': event_href(row['code'], event['date'], dates),
            })

    return sorted(events, key=lambda event: event['date'], reverse=True)


def hero_segments(summary: Optional[dict], parties) -> list:
    """
    Segments of the seat bar of the hero: the parties with seats in the order of the hemicycle.

    Parameters
    ----------
    summary : dict or None
        Data of the summary part of the mode shown (``totals`` or ``parties``).
    parties : Catalog
        Party catalogue of the run.

    Returns
    -------
    list of dict
        ``{'name', 'color', 'seats', 'width'}`` per party with seats, ``width`` as a percentage of the chamber
        with one decimal; empty without a summary or without seats.
    """
    if not summary:
        return []

    totals = summary.get('totals') or {row['name']: row.get('seats') for row in summary.get('parties') or []}
    n_seats = summary.get('n_seats') or sum(seats for seats in totals.values() if seats)

    if not n_seats:
        return []

    names = hemicycle_order([name for name, seats in totals.items() if seats], parties)

    return [{'name': name, 'color': parties[name]['color'], 'seats': totals[name],
             'width': round(100 * totals[name] / n_seats, 1)} for name in names]


def majority_chip(probs: dict, summary: Optional[dict]) -> Optional[dict]:
    """
    The majority chip of the hero: the block most likely to reach the absolute majority.

    Parameters
    ----------
    probs : dict
        ``p_majority`` by block (of the summary or of the headline).
    summary : dict or None
        Data of the summary part, for the seats of the block.

    Returns
    -------
    dict or None
        ``{'block', 'p', 'seats', 'reached'}``: ``reached`` when the probability reaches 0.5 (the chip is green and
        names the seats), otherwise the grey "Sin mayoría clara"; ``None`` without probabilities.
    """
    if not probs:
        return None

    block = max(probs, key=probs.get)
    rows = (summary or {}).get('vs') or []
    rows = rows + ((summary or {}).get('blocks') or [])
    seats = next((row.get('seats') for row in rows if row.get('name') == block), None)

    return {'block': block, 'p': probs[block], 'seats': seats, 'reached': probs[block] >= 0.5}


def hero_context(card: dict, today: Optional[date] = None) -> Optional[dict]:
    """
    Data of the featured card of the home: the next election of the first published scope.

    Parameters
    ----------
    card : dict
        Row of ``event_cards`` (``code``, ``run``, ``mode``, ``frozen``...); unpublished gives ``None``.
    today : datetime.date, optional
        Reference day for the days left.

    Returns
    -------
    dict or None
        The texts and figures of the card: ``code``, ``name``, ``href``, ``href_seats``, ``title``, ``flag`` (only
        ``es``), ``chamber`` (long name), ``n_seats``, ``majority``, ``date``, ``days_text``, ``eyebrow``,
        ``mode_label``, ``side_title``, ``leader`` (``name``, ``fullname``, ``color``, ``seats``, ``pct``, from the
        headline of the mode), ``rows`` (the parties of the headline), ``segments`` (``hero_segments``),
        ``majority_pct``, ``ends`` (seats of the ``vs`` blocks, Izquierda first), ``chip`` (``majority_chip``),
        ``deltas`` (the two parties with most seats present in the last results, ``{'name', 'text'}``),
        ``results_year``, ``lead`` (``lead_strip``), ``run_at``, ``as_of``, ``estimated`` (the date of the estimate
        shown: today for a live nowcast, the publication date otherwise), ``n_polls``, ``n_pollsters``, ``top``
        (the names of the ``TOP_PARTIES`` parties by vote), ``top_seats`` (the four parties with most seats, for
        the legend of the small hemicycle), ``others`` (the seats of the rest) and ``catalog``.
    """
    if not card.get('published') or not card.get('run'):
        return None

    code, run, mode = card['code'], card['run'], card['mode']
    meta = bundle_part(site().run_file, code, run, 'meta') or {}
    headline = bundle_part(site().run_file, code, run, 'headline') or {}
    estimate = headline.get(mode) or {}
    summary = bundle_part(site().run_file, code, run, 'summary', mode)
    parties = catalog(meta)
    rows = estimate.get('parties') or []
    leader = leader_of(rows)
    n_seats = (summary or {}).get('n_seats') or meta.get('n_seats')
    majority = (summary or {}).get('majority') or meta.get('majority')
    results = last_results(code)
    previous = (results or {}).get('parties') or {}
    deltas = [{'name': row['name'], 'text': delta_text((row.get('seats') or 0) - (previous[row['name']].get('seats') or 0))}
              for row in sorted(rows, key=lambda row: -(row.get('seats') or 0)) if row['name'] in previous][:2]
    forecast = mode == 'forecast'
    national = code == 'es'
    name = card['name']
    top_seats = [{'name': row['name'], 'color': parties[row['name']]['color'], 'seats': row.get('seats'), 'pct': row.get('pct')}
                 for row in sorted(rows, key=lambda row: -(row.get('seats') or 0))[:4]]
    probs = (summary or {}).get('p_majority') or estimate.get('p_majority') or {}

    return {
        'code': code,
        'name': name,
        'href': card['href'],
        'href_seats': card['href_seats'],
        'title': card['title'],
        'flag': national,
        'chamber': CHAMBER_NAMES.get(code, DEFAULT_CHAMBER_NAME),
        'n_seats': n_seats,
        'majority': majority,
        'date': card['event_date'],
        'days_text': card['days_text'],
        'eyebrow': 'Generales' if national else name,
        'mode_label': ('Pronóstico definitivo' if card['frozen'] else 'Pronóstico') if forecast else 'Si hoy hubiera elecciones',
        'side_title': (('Pronóstico para las generales' if national else 'Pronóstico para ' + name) if forecast
                       else ('Si hoy hubiera generales' if national else 'Si hoy hubiera elecciones en ' + name)),
        'leader': {'name': leader['name'], 'fullname': parties[leader['name']]['fullname'],
                   'color': parties[leader['name']]['color'], 'seats': leader.get('seats'),
                   'pct': leader.get('pct')} if leader else None,
        'rows': rows,
        'segments': hero_segments(summary, parties) if n_seats else [],
        'majority_pct': round(100 * majority / n_seats, 1) if majority and n_seats else None,
        'ends': sorted([{'name': row['name'], 'seats': row.get('seats')} for row in (summary or {}).get('vs') or []],
                       key=lambda row: row['name'] != 'Izquierda'),
        'chip': majority_chip(probs, summary),
        'deltas': deltas,
        'results_year': results['year'] if results else None,
        'lead': lead_strip(rows, majority),
        'run_at': card['run_at'],
        'as_of': headline.get('as_of') or meta.get('as_of'),
        # the date of the estimate shown: today for the live nowcast (as the event bar says), the publication
        # date once frozen or in forecast mode
        'estimated': (today or datetime.now(_madrid_tz()).date()).isoformat() if not forecast and not card['frozen'] else card['run_at'],
        'n_polls': meta.get('n_polls'),
        'n_pollsters': meta.get('n_pollsters'),
        'top': [row['name'] for row in sorted(rows, key=lambda row: -(row.get('pct') or 0))[:TOP_PARTIES]],
        'top_seats': top_seats,
        'others': (n_seats - sum(row.get('seats') or 0 for row in top_seats)) if n_seats and top_seats else None,
        'catalog': parties,
    }


def latest_polls(code: str, run: str, parties, top: list, n: int = LATEST_POLLS) -> list:
    """
    The latest polls of a run, for the side card of the home.

    Parameters
    ----------
    code : str
        Scope code.
    run : str
        Run id.
    parties : Catalog
        Party catalogue of the run.
    top : list of str
        Parties whose figures each row shows.
    n : int, optional
        Rows.

    Returns
    -------
    list of dict
        ``{'date', 'day', 'month', 'pollster', 'sponsor', 'sample', 'figures': [{'name', 'color', 'pct'}]}`` of the
        ``n`` most recent polls (ties by pollster name); empty without the polls part.
    """
    polls = bundle_part(site().run_file, code, run, 'polls') or {}
    rows = sorted((poll for poll in polls.get('polls') or [] if poll.get('date')),
                  key=lambda poll: (poll['date'], poll.get('pollster') or ''), reverse=True)
    out = []

    for poll in rows[:n]:
        day, month = fmt_date_short(poll['date']).split(' ')
        out.append({
            'date': poll['date'],
            'day': day,
            'month': month,
            'pollster': poll.get('pollster') or '',
            'sponsor': poll.get('sponsor') or None,
            'sample': poll_sample(poll),
            'figures': [{'name': name, 'color': parties[name]['color'], 'pct': poll.get(name)} for name in top],
        })

    return out


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
        ``common_context`` of the home (default scope) plus ``intro``, ``hero`` (``hero_context`` of the first
        published scope, or ``None``), ``autonomics`` (the other published cards), ``unpublished`` (the names of
        the scopes without a forecast), ``latest_polls``, ``past_shown`` and ``past_more`` (``past_events`` cut at
        ``PAST_SHOWN``) and, with a seat bar, ``initial`` (the hemicycle data: ``hero.seats``, ``colors``,
        ``fullnames``, ``order``, ``n_seats``, ``majority``).

    Raises
    ------
    HttpError
        503 without a bundle.
    """
    context = common_context(state, active='/')
    cards = event_cards(site().scopes()['scopes'])
    featured = next((card for card in cards if card['published']), None)
    hero = hero_context(featured) if featured else None
    past = past_events(context['scopes'])
    context.update(
        intro=INTRO,
        hero=hero,
        autonomics=[card for card in cards if card['published'] and (featured is None or card['code'] != featured['code'])],
        unpublished=[card['name'] for card in cards if not card['published']],
        latest_polls=latest_polls(featured['code'], featured['run'], hero['catalog'], hero['top']) if hero else [],
        past_shown=past[:PAST_SHOWN],
        past_more=max(0, len(past) - PAST_SHOWN),
    )

    if hero and hero['segments']:
        context['initial'] = {'hero': {
            'seats': {segment['name']: segment['seats'] for segment in hero['segments']},
            'colors': {segment['name']: segment['color'] for segment in hero['segments']},
            'fullnames': {segment['name']: hero['catalog'][segment['name']]['fullname'] for segment in hero['segments']},
            'order': [segment['name'] for segment in hero['segments']],
            'n_seats': hero['n_seats'],
            'majority': hero['majority'],
        }}

    return context
