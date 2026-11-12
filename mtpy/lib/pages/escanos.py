"""
The seats of an event: ``seats_context``, the seats part of the event page (``/{scope}``), and its pieces (calculator,
districts, downloads).
"""
import unicodedata
from typing import Optional

from ...core.api import HttpError
from ..webapi import check_region
from .common import api_url, part
from .filters import DASH, fmt_range

CSV_PARTS = (
    ('summary', 'Resumen por partido y bloque', True),
    ('dist', 'Escaños por simulación', True),
    ('districts', 'Circunscripciones', True),
    ('scenario', 'Reparto central', True),
    ('seat-odds', 'Escaños uno a uno por circunscripción', True),
    ('fan', 'Abanico por horizonte', False),
)


def default_coalition(meta: dict, summary: dict, names: list) -> list:
    """
    Parties checked by default in the coalition calculator.

    Parameters
    ----------
    meta : dict
        Data of the meta part (``bmaps``).
    summary : dict
        Data of the summary part (``vs`` and ``blocks``).
    names : list of str
        Parties the calculator can add up (``dist['parties']``).

    Returns
    -------
    list of str
        The parties of the first block of ``summary['vs']`` (or of ``summary['blocks']`` when
        ``vs`` is empty) according to ``meta['bmaps']``, in the order of ``names`` and limited to
        them; empty without blocks.
    """
    bmaps = meta.get('bmaps') or {}

    for key in ('vs', 'blocks'):
        rows = summary.get(key) or []

        if rows:
            groups = bmaps.get(key)
            members = (groups.get(rows[0]['name']) if isinstance(groups, dict) else None) or []

            return [name for name in names if name in members]

    return []


def csv_links(scope: str, mode: str, run: str, skip: tuple = ()) -> list:
    """
    Download links of the seats of the event page.

    Parameters
    ----------
    scope : str
        Scope code.
    mode : str
        Simulation mode of the mode parts.
    run : str
        Run id the page shows.
    skip : tuple of str, default ()
        Parts left out (the run has no ``seat-odds``).

    Returns
    -------
    list of dict
        ``{'label', 'href'}`` for ``summary``, ``dist``, ``districts``, ``scenario``, ``seat-odds`` (with
        ``mode``) and ``fan`` (without it), all CSV and pinned to ``run``, but those of ``skip``.
    """
    return [
        {'label': label, 'href': api_url(scope, name, mode=mode if by_mode else None, run=run, fmt='csv')}
        for name, label, by_mode in CSV_PARTS
        if name not in skip
    ]


def resolve_region(region: Optional[int], regions: list) -> Optional[dict]:
    """
    Pick the district the detail table shows.

    Parameters
    ----------
    region : int, optional
        Requested district id.
    regions : list of dict
        ``districts['regions']`` (``id``, ``name``, ``seats``).

    Returns
    -------
    dict or None
        The requested district; the one with most seats (the first on ties) when ``region`` is
        ``None``; ``None`` when there are no districts and no request.

    Raises
    ------
    HttpError
        404 when ``region`` is not among the districts.
    """
    if region is None:
        return max(regions, key=lambda item: item['seats']) if regions else None

    for item in regions:
        if item['id'] == region:
            return item

    raise HttpError(404, 'region not found')


def _sort_name(name: str) -> str:
    """
    Sort key of a territory name: accents folded and case ignored, as the table headers sort it.

    Parameters
    ----------
    name : str
        Territory name.

    Returns
    -------
    str
        The name without combining marks, casefolded.
    """
    decomposed = unicodedata.normalize('NFKD', str(name))

    return ''.join(char for char in decomposed if not unicodedata.combining(char)).casefold()


def district_table(districts: dict, scenario: dict, summary: Optional[dict] = None) -> Optional[dict]:
    """
    Overview table of the central allocation by district and party.

    Parameters
    ----------
    districts : dict
        Data of the ``districts`` part.
    scenario : dict
        Data of the ``scenario`` part.
    summary : dict, optional
        Data of the ``summary`` part: the vote (``pct``) of its ``parties`` breaks the ties in seats.

    Returns
    -------
    dict or None
        ``None`` without districts. Otherwise ``parties`` (every party of the run, ``districts['parties']``,
        with or without seats, by central total descending, then by vote descending, then in that
        order; the script keeps this order in the vote view too), ``rows`` (``id``,
        ``name``, ``seats`` and ``cells`` per district) and ``total`` (the central row of the
        whole scope, or ``None``). Each cell is ``{'seats': int or None, 'range': str}``: the
        central seats (``None`` when the district is not in the ``scenario`` part) and the 95 % interval.
    """
    regions = districts.get('regions') or []

    if not regions:
        return None

    scenario_parties = list(scenario.get('parties') or [])
    scenario_rows = {item['region_id']: item['seats'] for item in scenario.get('rows') or []}
    intervals = {(item['region_id'], item['name']): item for item in districts.get('rows') or []}
    names = list(districts.get('parties') or [])

    def seats_of(region_id, name):
        values = scenario_rows.get(region_id)

        if values is None or name not in scenario_parties:
            return None

        return values[scenario_parties.index(name)]

    def total_of(name):
        if 0 in scenario_rows:
            return seats_of(0, name) or 0

        return sum((seats_of(region_id, name) or 0) for region_id in scenario_rows)

    # every party of the run, with or without seats (2026-10-30): a column of dashes says it wins none;
    # the same order in both views of the table, so the vote breaks the ties among the seatless
    votes = {row['name']: row.get('pct') for row in (summary or {}).get('parties') or []}
    shown = sorted(names, key=lambda name: (-total_of(name), -(votes.get(name) or 0)))

    def cells(region_id):
        out = []

        for name in shown:
            item = intervals.get((region_id, name))
            out.append({
                'seats': seats_of(region_id, name),
                'range': fmt_range(item['seats_lo'], item['seats_hi'], 0) if item else DASH,
            })

        return out

    total = None

    if 0 in scenario_rows:
        total = {'seats': sum(scenario_rows[0]), 'cells': cells(0)}

    return {
        'parties': shown,
        'rows': [
            {'id': item['id'], 'name': item['name'], 'seats': item['seats'], 'cells': cells(item['id'])}
            for item in sorted(regions, key=lambda item: _sort_name(item['name']))
        ],
        'total': total,
    }


def has_communities(regions: list) -> bool:
    """
    Whether the districts can be grouped by autonomous community.

    Parameters
    ----------
    regions : list of dict
        ``districts['regions']`` (older runs have no ``community`` or ``votes``).

    Returns
    -------
    bool
        True when every region has a ``community`` and a numeric ``votes`` and there are at least two
        different community codes (the rule of ``canGroup`` in ``territory.js``).
    """
    if not regions:
        return False

    for item in regions:
        votes = item.get('votes')

        if not item.get('community') or isinstance(votes, bool) or not isinstance(votes, (int, float)):
            return False

    return len({item['community'].get('code') for item in regions}) > 1


def region_rows(districts: dict, region_id: int) -> list:
    """
    Party rows of one district.

    Parameters
    ----------
    districts : dict
        Data of the ``districts`` part.
    region_id : int
        District id.

    Returns
    -------
    list of dict
        The rows of that district ordered by ``seats`` and then ``pct``, both descending (stable),
        unformatted. Rows without a vote (``pct`` is ``None``: the party does not stand there) are
        dropped.
    """
    rows = [item for item in districts.get('rows') or []
            if item['region_id'] == region_id and item['pct'] is not None]

    return sorted(rows, key=lambda item: (-(item['seats'] or 0), -(item['pct'] or 0)))


def seats_context(context: dict, region=None) -> dict:
    """
    The seats part of an event page, read over its resolved common context: the summary, the coalition calculator,
    the downloads and the districts of the run and mode shown.

    Parameters
    ----------
    context : dict
        ``common_context`` of the event page (``state``, ``meta``, ``catalog``).
    region : str, optional
        ``region`` query parameter: id of the district the detail table shows (the largest by default).

    Returns
    -------
    dict
        The keys the seats add to the page: ``summary``, ``calculator`` (``name``, ``fullname``, ``color`` and
        ``checked`` per party of the ``dist`` part, the pills the server renders), ``csv_links``, the districts keys
        (``district_table``, ``regions``, ``region`` and ``region_rows``), ``territory_levels``
        (``has_communities``) and ``initial_extra``, the members the seats add to the initial data: ``dist_href``
        (the JSON route of the ``dist`` part of the run and mode shown, which the browser asks for after the first
        drawing: the simulations are too heavy for the page, Luis 2026-11-04), ``fan``, ``districts``,
        ``scenario`` and ``odds_hrefs`` (district id to the href of its card); ``odds`` (``seat_odds_links``, ``None``
        without the page «Escaños que bailan»).

    Raises
    ------
    HttpError
        404 for a missing part or an unknown district; 400 for an invalid ``region``.
    """
    # imported here: ``circunscripciones`` imports this module
    from .circunscripciones import seat_odds_links

    region_id = check_region(region)
    full, meta, parties = context['state'], context['meta'], context['catalog']
    scope, mode, run = full['scope'], full['mode'], full['run']
    summary = part(scope, run, 'summary', mode=mode)
    # the simulations stay out of the page: only the names of the calculator are read from them
    names = list(part(scope, run, 'dist', mode=mode).get('parties') or [])
    districts = part(scope, run, 'districts', mode=mode)
    scenario = part(scope, run, 'scenario', mode=mode)
    regions = districts.get('regions') or []
    current = resolve_region(region_id, regions)
    checked = default_coalition(meta, summary, names)
    odds = seat_odds_links(scope, run, mode, context['query'])

    return {
        'summary': summary,
        'calculator': [
            {'name': name, 'fullname': parties[name]['fullname'], 'color': parties[name]['color'],
             'checked': name in checked}
            for name in names
        ],
        'csv_links': csv_links(scope, mode, run, skip=() if odds else ('seat-odds',)),
        'odds': odds,
        'district_table': district_table(districts, scenario, summary),
        'regions': regions,
        'territory_levels': has_communities(regions),
        'region': current,
        'region_rows': region_rows(districts, current['id']) if current else [],
        'initial_extra': {
            'dist_href': api_url(scope, 'dist', mode=mode, run=run),
            'fan': part(scope, run, 'fan'),
            'districts': districts,
            'scenario': scenario,
            'odds_hrefs': odds['districts'] if odds else {},
        },
    }
