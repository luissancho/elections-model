"""
Seats page context.
"""
from typing import Optional

from ...core.api import HttpError
from .. import bundle
from ..webapi import check_region, site
from .common import Catalog, api_url, block_color, common_context, part
from .filters import DASH, _is_missing, fmt_int, fmt_prob, fmt_range

CSV_PARTS = (
    ('summary', 'Resumen por partido y bloque', True),
    ('dist', 'Escaños por simulación', True),
    ('districts', 'Circunscripciones', True),
    ('scenario', 'Escenario central', True),
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


def escanos_headline_text(summary: dict, meta: dict) -> str:
    """
    One-line headline of the seats page in plain words.

    Parameters
    ----------
    summary : dict
        Data of the summary part (``blocks`` with ``name`` and ``p_majority``; ``majority`` and
        ``n_seats``).
    meta : dict
        Data of the meta part (``majority`` and ``n_seats``, which take precedence).

    Returns
    -------
    str
        ``'{block} es el bloque con más probabilidad de mayoría absoluta ({p}); la mayoría está en
        {majority} escaños de {n_seats}.'``, the block being that with the highest ``p_majority`` (the
        first one on ties); without blocks with a probability, ``'La mayoría absoluta está en
        {majority} escaños de {n_seats}.'``.
    """
    majority = fmt_int(meta.get('majority') if meta.get('majority') is not None else summary.get('majority'))
    n_seats = fmt_int(meta.get('n_seats') if meta.get('n_seats') is not None else summary.get('n_seats'))
    rows = [row for row in summary.get('blocks') or [] if not _is_missing(row.get('p_majority'))]

    if not rows:
        return 'La mayoría absoluta está en {} escaños de {}.'.format(majority, n_seats)

    top = max(rows, key=lambda row: row['p_majority'])

    return ('{} es el bloque con más probabilidad de mayoría absoluta ({}); '
            'la mayoría está en {} escaños de {}.').format(
        top['name'], fmt_prob(top['p_majority']), majority, n_seats)


def district_table(districts: dict, scenario: dict) -> Optional[dict]:
    """
    Overview table of the central scenario by district and party.

    Parameters
    ----------
    districts : dict
        Data of the ``districts`` part.
    scenario : dict
        Data of the ``scenario`` part.

    Returns
    -------
    dict or None
        ``None`` without districts. Otherwise ``parties`` (names with any seat in the scenario or
        ``seats_hi`` above zero, in the order of ``districts['parties']``), ``rows`` (``id``,
        ``name``, ``seats`` and ``cells`` per district) and ``total`` (the scenario row of the
        whole scope, or ``None``). Each cell is ``{'seats': int or None, 'range': str}``: the
        scenario seats (``None`` when the district is not in the scenario) and the 95 % interval.
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

    def has_seats(name):
        if any((seats_of(region_id, name) or 0) > 0 for region_id in scenario_rows):
            return True

        return any((item['seats_hi'] or 0) > 0 for (_, party), item in intervals.items() if party == name)

    shown = [name for name in names if has_seats(name)]

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
            for item in regions
        ],
        'total': total,
    }


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
        The common context plus ``summary``, ``party_rows`` (``seat_rows``), ``block_rows`` and
        ``headline_text`` (``escanos_headline_text``), ``calculator`` (``name``,
        ``fullname``, ``color`` and ``checked`` per party of ``dist``), ``csv_links``, ``subtitle``, the districts keys
        (``district_table``, ``regions``, ``region`` and ``region_rows``) and
        ``initial`` (``state``, ``meta``, ``summary``, ``dist``, ``fan`` and ``runs``, the data
        the charts and the calculator are drawn from).

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

    context['summary'] = summary
    context['headline_text'] = escanos_headline_text(summary, meta)
    context['party_rows'] = seat_rows(summary.get('parties') or [])
    context['block_rows'] = block_rows(summary.get('blocks') or [], meta, parties)
    context['calculator'] = [
        {'name': name, 'fullname': parties[name]['fullname'], 'color': parties[name]['color'],
         'checked': name in checked}
        for name in names
    ]
    context['csv_links'] = csv_links(scope, mode, run)
    context['subtitle'] = '· ' + suffix[:1].lower() + suffix[1:]
    context['district_table'] = district_table(districts, scenario)
    context['regions'] = districts.get('regions') or []
    context['region'] = current
    context['region_rows'] = region_rows(districts, current['id']) if current else []
    context['initial'] = {
        'state': full,
        'meta': meta,
        'summary': summary,
        'dist': dist,
        'fan': part(scope, run, 'fan'),
        'runs': bundle.loads(site().history(scope))['data'],
    }

    return context
