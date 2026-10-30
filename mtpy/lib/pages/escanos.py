"""
Seats page context.
"""
import unicodedata
from typing import Optional

from ...core.api import HttpError
from .. import bundle
from ..webapi import check_region, site
from .common import Catalog, api_url, block_color, common_context, ink_color, part
from .elecciones import last_results
from .filters import DASH, _is_missing, fmt_int, fmt_prob, fmt_range
from .promedios import inherited_results, seat_deltas

CSV_PARTS = (
    ('summary', 'Resumen por partido y bloque', True),
    ('dist', 'Escaños por simulación', True),
    ('districts', 'Circunscripciones', True),
    ('scenario', 'Reparto central', True),
    ('fan', 'Abanico por horizonte', False),
)


def seat_rows(rows: list) -> list:
    """
    Party rows of the summary sorted by seats.

    Parameters
    ----------
    rows : list of dict
        Records of ``summary['parties']`` (``seats`` is an integer or ``None``).

    Returns
    -------
    list of dict
        The same records by ``seats`` descending, ``None`` last; ties keep the input order.
    """
    return sorted(rows, key=lambda row: -1 if row.get('seats') is None else row['seats'], reverse=True)


def block_rows(rows: list, meta: dict, parties: Catalog) -> list:
    """
    Block rows of the summary with their colour.

    Parameters
    ----------
    rows : list of dict
        Records of ``summary['vs']`` or ``summary['blocks']``.
    meta : dict
        Data of the meta part.
    parties : Catalog
        Party catalogue of the run.

    Returns
    -------
    list of dict
        Copies of the records, in the input order, with ``color`` (``block_color``).
    """
    return [dict(row, color=block_color(row['name'], meta, parties)) for row in rows]


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


def csv_links(scope: str, mode: str, run: str) -> list:
    """
    Download links of the seats page.

    Parameters
    ----------
    scope : str
        Scope code.
    mode : str
        Simulation mode of the mode parts.
    run : str
        Run id the page shows.

    Returns
    -------
    list of dict
        ``{'label', 'href'}`` for ``summary``, ``dist``, ``districts``, ``scenario`` (with
        ``mode``) and ``fan`` (without it), all CSV and pinned to ``run``.
    """
    return [
        {'label': label, 'href': api_url(scope, name, mode=mode if by_mode else None, run=run, fmt='csv')}
        for name, label, by_mode in CSV_PARTS
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


def escanos_context(state: dict, region=None, active: Optional[str] = None, event: bool = False,
                    active_suffix: str = '') -> dict:
    """
    Template context of the seats page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    region : str, optional
        ``region`` query parameter: id of the district the detail table shows (the largest by
        default).
    active : str or None, optional
        ``href`` of the page in the navigation (``None``: the seats are an event page).
    event : bool, optional
        Whether it is rendered as an event page (see ``common_context``).
    active_suffix : str, optional
        Suffix of the current tab of the event bar (``'/escanos'`` on the event page).

    Returns
    -------
    dict
        The common context plus ``summary``, ``party_rows`` (``seat_rows``), ``block_rows``, ``calculator`` (``name``,
        ``fullname``, ``color`` and ``checked`` per party of ``dist``), ``csv_links``, ``subtitle``, the districts keys
        (``district_table``, ``regions``, ``region`` and ``region_rows``), ``territory_levels`` (``has_communities``),
        ``deltas`` (``seat_deltas`` against the last published election, for the «vs {year}» column of the tables),
        ``results_year`` and ``results_date`` (of that election; ``None`` without one), ``blocks`` (the names of
        ``meta['bmaps']['blocks']``, the Bloques view of the evolution; ``[]`` without a block map) and ``initial``
        (``state``, ``meta``, ``summary``, ``dist``, ``fan``, ``districts``, ``scenario``, ``runs``, ``deltas``
        (``seat_deltas`` against the last published election, for the seats bar) and ``inks`` (the text colour over
        the colour of each party and block, for the pills of the evolution legend), the data the charts, the
        calculator and the territory table are drawn from).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope, a missing run, a missing part or an
        unknown district; 400 for an invalid ``region``.
    """
    region_id = check_region(region)
    context = common_context(state, active, event=event, active_suffix=active_suffix)
    full, meta, parties = context['state'], context['meta'], context['catalog']
    scope, mode, run = full['scope'], full['mode'], full['run']
    summary = part(scope, run, 'summary', mode=mode)
    dist = part(scope, run, 'dist', mode=mode)
    names = list(dist.get('parties') or [])
    districts = part(scope, run, 'districts', mode=mode)
    scenario = part(scope, run, 'scenario', mode=mode)
    current = resolve_region(region_id, districts.get('regions') or [])
    checked = default_coalition(meta, summary, names)
    suffix = context['title_suffix']
    results = inherited_results(last_results(scope, parties), meta)
    deltas = seat_deltas(summary, meta, parties, results)
    # whom each party is compared with, for the tooltip of its arrow (``inherited_results``)
    inherited = {name: entry['inherited'] for name, entry in ((results or {}).get('parties') or {}).items()
                 if entry.get('inherited')}

    context['summary'] = summary
    context['inherited'] = inherited
    context['party_rows'] = seat_rows(summary.get('parties') or [])
    context['block_rows'] = block_rows(summary.get('blocks') or [], meta, parties)
    context['calculator'] = [
        {'name': name, 'fullname': parties[name]['fullname'], 'color': parties[name]['color'],
         'checked': name in checked}
        for name in names
    ]
    context['csv_links'] = csv_links(scope, mode, run)
    context['subtitle'] = '· ' + suffix[:1].lower() + suffix[1:]
    context['district_table'] = district_table(districts, scenario, summary)
    context['regions'] = districts.get('regions') or []
    context['territory_levels'] = has_communities(context['regions'])
    context['region'] = current
    context['region_rows'] = region_rows(districts, current['id']) if current else []
    context['deltas'] = deltas
    context['results_year'] = results['year'] if results else None
    context['results_date'] = results['date'] if results else None
    # the Bloques view of the evolution, summed in the browser from the block map
    blocks = (meta.get('bmaps') or {}).get('blocks')
    context['blocks'] = list(blocks) if isinstance(blocks, dict) else []
    context['initial'] = {
        'state': full,
        'meta': meta,
        'summary': summary,
        'dist': dist,
        'fan': part(scope, run, 'fan'),
        'districts': districts,
        'scenario': scenario,
        'runs': bundle.loads(site().history(scope))['data'],
        'deltas': deltas,
        # ink of every party and block of the evolution, for the pills of its legend when they are on
        'inks': {**{party['name']: ink_color(party.get('color')) for party in meta.get('parties') or []},
                 **{block: ink_color(block_color(block, meta, parties)) for block in context['blocks']}},
    }

    return context
