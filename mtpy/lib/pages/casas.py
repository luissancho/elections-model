"""
Houses page context.
"""
import re

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Callable, Optional, Sequence

from .. import webapi
from ..quality import level_text, load_criteria, max_points, quality_breakdown
from ..webapi import query_pollster, query_pollster_path, query_ratings, site
from .common import OTHERS_COLOR, analysis_part, api_url, block_color, common_context
from .elecciones import (error_blocks, event_part, history_events, history_rows, history_scopes, poll_error_parts,
                         table_blocks, year_range)
from .filters import _madrid_tz, fmt_num, fmt_date

RATING_SCOPE = 'es'
# Rows of the ranking the page shows before "Cargar más".
RANKING_SHOWN = 50
# The type of a pollster in the catalogue (``mtype``) and its public label; a pollster without one (null, or ``-`` as
# the catalogue prints it) is an ordinary private house (``MTYPE_DEFAULT``).
MTYPE_LABELS = {'state': 'Pública', 'aggr': 'Agregador', 'online': 'Panel Online'}
MTYPE_DEFAULT = 'Privada'
RANKING_KEYS = ('pollster_id', 'pollster', 'rating', 'quality', 'num_polls', 'num_events', 'num_national_events',
                'error_avg', 'error_blocks', 'bias_blocks', 'bias_dev_adj')
HERDING_KEYS = ('event_scope', 'event_date', 'n', 'ratio', 'p_value')
# The links of a quality source cell and the punctuation that may close the sentence after one.
URL_PATTERN = re.compile(r'(https?://\S+)')
URL_TRAILING = ').;,:'
# Threads that read the `cycle` and `results` parts of the bullseye elections at once (one S3 read each).
PART_READERS = 8
ANALYSIS_CSV = (
    ('house-effects', None, 'Efectos de encuestadora por partido y bloque'),
    ('herding', None, 'Rebaño por encuestadora'),
    ('herding', 'events', 'Rebaño por encuestadora y ciclo'),
    ('poll-errors', None, 'Error de los sondeos en cada elección'),
)



def mtype_label(mtype) -> str:
    """
    Public label of the type of a pollster.

    Parameters
    ----------
    mtype : str or None
        The ``mtype`` of the catalogue (``'state'``, ``'aggr'``, ``'online'``); ``None``, NaN, ``''`` or ``'-'`` when
        it has none.

    Returns
    -------
    str
        The label of ``MTYPE_LABELS``; ``MTYPE_DEFAULT`` without a type; an unknown value as it comes.
    """
    if mtype is None or (isinstance(mtype, float) and mtype != mtype) or mtype in ('', '-'):
        return MTYPE_DEFAULT

    return MTYPE_LABELS.get(mtype, mtype)


def _count_or_none(value) -> Optional[int]:
    """
    A count of a ratings row as an integer, ``None`` when it is unknown.

    Parameters
    ----------
    value : int, float or None
        The value of the row; ``None`` or NaN when the column is missing or null.

    Returns
    -------
    int or None
        The count (``0`` included), or ``None`` without one.
    """
    if value is None or value != value:
        return None

    return int(value)


def ranking_rows(ratings, herding: Optional[dict]) -> list:
    """
    Rows of the pollsters ranking.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Ratings of one event (``webapi.query_ratings``).
    herding : dict or None
        Data of the ``herding`` analysis part, or ``None`` when it is not published.

    Returns
    -------
    list of dict
        The pollsters with a rating and at least one poll (``num_polls``; a pollster without polls has its
        prior quality as rating, nothing to rank), by ``rating`` descending (ties keep the input order), with
        the keys of ``RANKING_KEYS`` (``None`` when a column is missing, ``num_national_events`` too: old ratings rows
        without the column) plus ``herding``, the pooled herding of
        the analysis for that pollster name or ``None``. ``pollster_id`` is an integer, for the profile links.
    """
    if ratings.empty or 'rating' not in ratings.columns:
        return []

    pooled = {row['pollster']: row.get('herding') for row in (herding or {}).get('rows') or []}
    rows = []

    for record in webapi.frame_to_records(ratings):
        if record.get('rating') is None or not record.get('num_polls'):
            continue

        row = {key: record.get(key) for key in RANKING_KEYS}
        row['pollster_id'] = int(row['pollster_id'])
        row['num_national_events'] = _count_or_none(row['num_national_events'])
        row['herding'] = pooled.get(row['pollster'])
        rows.append(row)

    return sorted(rows, key=lambda row: row['rating'], reverse=True)


# The columns of the two tables of the houses, in order: the ranking (``/casas``) and the rating path of a profile.
# Each column has a public label and one description, which the table header carries as tooltip and the glossary
# «Qué mide cada columna» lists (``table_columns``).
RANKING_COLUMNS = ('pollster', 'rating', 'quality', 'num_events', 'num_polls', 'bias_dev_adj', 'error_avg',
                   'error_blocks', 'herding')
PATH_COLUMNS = ('event', 'n_polls', 'bias_dev_adj', 'error_avg', 'error_blocks', 'rating', 'change')
COLUMN_LABELS = {'pollster': 'Encuestadora', 'rating': 'Rating', 'quality': 'Confianza', 'num_events': 'Elecciones',
                 'num_polls': 'Sondeos', 'bias_dev_adj': 'Diferencial',
                 'error_avg': 'Error medio', 'error_blocks': 'Sesgo bloques', 'herding': 'Rebaño', 'event': 'Evento',
                 'n_polls': 'Sondeos', 'change': 'Cambio'}
# A second, smaller line under the label of a column: «Elecciones» carries the general elections in brackets
COLUMN_SUBLABELS = {'num_events': '(generales)'}


def table_columns(kind: str, blocks: Sequence[str] = (), scale: Optional[dict] = None) -> list:
    """
    The columns of a table of the houses with their public label and description.

    Parameters
    ----------
    kind : str
        ``'ranking'`` (``RANKING_COLUMNS``) or ``'path'`` (``PATH_COLUMNS``, the rating path of a profile).
    blocks : sequence of str, optional
        Names of the two blocks of the margin (first, second), for the sign of «Sesgo bloques»; a missing name
        is written as «el primer bloque» / «el segundo».
    scale : dict, optional
        ``{'per_point', 'floor'}`` of the rubric, for the points rule of «Confianza»; left out without it.

    Returns
    -------
    list of dict
        ``{'key', 'label', 'sub', 'text'}`` per column, in the order of the table; ``sub`` is the smaller second
        line of the header (``COLUMN_SUBLABELS``), ``None`` for most columns.
    """
    first = 'a ' + blocks[0] if len(blocks) > 0 else 'al primer bloque'
    second = 'a ' + blocks[1] if len(blocks) > 1 else 'al segundo'
    rule = (' Cada punto vale {} de confianza, con un mínimo de {}.'.format(scale['per_point'], scale['floor'])
            if scale and 'per_point' in scale else '')
    that = ' de esa elección' if kind == 'path' else ''
    texts = {
        'pollster': 'La encuestadora que firma los sondeos. Pulsa en su nombre para ver su perfil.',
        'rating': (
            'La nota, de 0 a 100, con la que quedó tras evaluar esa elección con todo su historial anterior, mezclada '
            'con su confianza: la que se le aplica hasta la siguiente. La de la primera fila es la de hoy, la que entra '
            'en el promedio.' if kind == 'path' else
            'La nota, de 0 a 100, con la que la encuestadora entra en el promedio de la próxima elección: mezcla el '
            'sesgo de sus sondeos pasados, comparado con el de los sondeos de las demás en las mismas fechas, con su '
            'confianza, que pesa más cuantas menos elecciones ha sondeado. Cuanto más alto, más pesan sus sondeos.'),
        'quality': (
            'Lo que se supone de la encuestadora antes de mirar sus sondeos, de 0 a 100, según una rúbrica pública de '
            'cuatro criterios sobre cómo trabaja (la calidad previa del método); pesa en el rating mientras ha '
            'sondeado pocas elecciones.' + rule),
        'num_events': ('En cuántas elecciones publicó sondeos, contando cada ámbito y fecha por separado: dos '
                       'autonómicas del mismo día cuentan dos. Entre paréntesis, cuántas de ellas fueron elecciones '
                       'generales: el dato con el que se calcula el criterio «Implantación» de la confianza.'),
        'num_polls': ('Cuántos sondeos suyos entran en el cálculo del rating, de todas las elecciones anteriores y de '
                      'todos los ámbitos, tras descartar los de peso ínfimo.'),
        'event': 'La elección, o las elecciones del mismo día, en que publicó sondeos; de la más nueva a la más antigua.',
        'n_polls': ('Cuántos sondeos publicó para esa elección de los que entran en el modelo; el diferencial, el error '
                    'medio y el sesgo de bloques son la media de esos sondeos.'),
        'bias_dev_adj': (
            'Cuánto se separó el sesgo de sus sondeos del de los sondeos comparables{} (misma fecha y tamaño), en la '
            'escala del sesgo del modelo, no en puntos: negativo, menos sesgo que los demás, y eso sube el rating; '
            'positivo, más. Un 2 ya es una diferencia grande.'.format(that)),
        'error_avg': ('Distancia media, en puntos, entre sus sondeos{} y el resultado de cada partido principal. Sin '
                      'signo: cuanto más rojo, mayor.'.format(that)),
        'error_blocks': (
            'Cuánto se equivocaron sus sondeos{} en el margen entre los dos bloques principales, en puntos y con signo: '
            'positivo, daban más margen {} del que hubo; negativo, más margen {}.'.format(that, first, second)),
        'herding': ('Cuánto varían sus sondeos frente a lo que permite el azar de sus muestras: por debajo de 1, menos '
                    'de lo esperable, como si se acercaran a los demás.'),
        'change': 'Cuánto subió o bajó el rating tras esa elección frente al que tenía al llegar: verde si subió, rojo si bajó.',
    }
    keys = PATH_COLUMNS if kind == 'path' else RANKING_COLUMNS

    return [{'key': key, 'label': COLUMN_LABELS[key], 'sub': COLUMN_SUBLABELS.get(key), 'text': texts[key]} for key in keys]


def block_bar_colors(meta: dict, parties) -> tuple:
    """
    The two blocks of the margin («Sesgo bloques») and the colour of each side of its bar.

    Parameters
    ----------
    meta : dict
        Data of the meta part of the run (``bmaps.vs`` names the blocks, the first one first).
    parties : Catalog
        Party catalogue of the run, for the colour of each block (``block_color``).

    Returns
    -------
    tuple
        ``(blocks, colors)``: the names of the first two blocks of ``vs`` (fewer when the map has fewer) and
        ``{'pos', 'neg', 'zero'}`` with the colour of the first block for a positive bias (the poll favoured it),
        the colour of the second for a negative one and ``OTHERS_COLOR`` for a missing block or an empty bar.
    """
    blocks = list(((meta or {}).get('bmaps') or {}).get('vs') or [])[:2]
    colors = {
        'pos': block_color(blocks[0], meta, parties) if blocks else OTHERS_COLOR,
        'neg': block_color(blocks[1], meta, parties) if len(blocks) > 1 else OTHERS_COLOR,
        'zero': OTHERS_COLOR,
    }

    return blocks, colors


def house_effects_grid(part: dict) -> dict:
    """
    Grid of the house effects of the current cycle: one row per pollster, one cell per column.

    Parameters
    ----------
    part : dict
        Data of the ``house-effects`` analysis part (``parties``, ``blocks``, ``rows``).

    Returns
    -------
    dict
        ``{'columns': parties + blocks, 'rows': [{'pollster', 'cells'}]}``; each cell is
        ``{'cycle', 'hist', 'trend'}`` (all ``None`` when the analysis has no row for it). Only
        pollsters with some ``cycle`` are kept, in the order of the analysis, then sorted by the
        ``cycle`` of the first block descending (of the first column without blocks), ``None``
        last.
    """
    blocks = list(part.get('blocks') or [])
    columns = list(part.get('parties') or []) + blocks
    values = {}

    for row in part.get('rows') or []:
        values.setdefault(row['pollster'], {})[row['name']] = row

    rows = []

    for pollster, by_name in values.items():
        cells = [{key: (by_name.get(name) or {}).get(key) for key in ('cycle', 'hist', 'trend')}
                 for name in columns]

        if any(cell['cycle'] is not None for cell in cells):
            rows.append({'pollster': pollster, 'cells': cells})

    if columns:
        index = columns.index(blocks[0]) if blocks else 0
        rows.sort(key=lambda row: (row['cells'][index]['cycle'] is None, -(row['cells'][index]['cycle'] or 0)))

    return {'columns': columns, 'rows': rows}


def herding_rows(part: dict) -> list:
    """
    Rows of the herding table.

    Parameters
    ----------
    part : dict
        Data of the ``herding`` analysis part.

    Returns
    -------
    list of dict
        ``part['rows']`` by ``herding`` ascending (the most herded first), ``None`` last.
    """
    rows = part.get('rows') or []

    return sorted(rows, key=lambda row: (row.get('herding') is None, row.get('herding') or 0))


def profile_rows(records: list, keys: tuple, newest_first: bool = False) -> list:
    """
    Rows of a profile table, by event date.

    Parameters
    ----------
    records : list of dict
        JSON records of a table of the pollster (``webapi.query_pollster``).
    keys : tuple of str
        Keys to keep, in order (``None`` when a record lacks one).
    newest_first : bool, optional
        Sort from the newest event to the oldest instead of ascending.

    Returns
    -------
    list of dict
        The records reduced to ``keys``, sorted by ``event_date`` (stable).
    """
    rows = [{key: record.get(key) for key in keys} for record in records]

    return sorted(rows, key=lambda row: row.get('event_date') or '', reverse=newest_first)


def election_label(scope: str, date: str, names: dict, scope_names: dict) -> str:
    """
    Name of an election in the tables of a profile.

    Parameters
    ----------
    scope : str
        Scope code of the election.
    date : str
        Its ISO date.
    names : dict
        ``{(scope, date): name}`` of the published elections.
    scope_names : dict
        ``{code: name}`` of the catalogue scopes.

    Returns
    -------
    str
        The published name or, for an election without a page (the current cycle, an old election), its date
        and the name of its scope, which the date alone would not tell apart.
    """
    return names.get((scope, date)) or '{} · {}'.format(fmt_date(date), scope_names.get(scope, scope))


def effects_table(records: list, label: Optional[Callable] = None, order: Sequence[str] = (),
                  key: str = 'dev_result_c') -> dict:
    """
    House effects of a pollster by election and party.

    Parameters
    ----------
    records : list of dict
        Its rows of the house effects table (``event_scope``, ``event_date``, ``party``, ``dev_result_c``,
        ``dev_cycle``...); a row without ``event_scope`` belongs to ``RATING_SCOPE``.
    label : callable, optional
        ``(scope, date) -> str``, the name of each election; its date by default.
    order : sequence of str, optional
        Party names that go first, in this order (the catalogue of the run); the rest follow by first
        appearance.
    key : str, optional
        Column shown: ``dev_result_c`` (the deviation from the result, centred; only the elections where the
        pollster had enough polls in the final window) or ``dev_cycle`` (the deviation from the consensus of the
        whole cycle).

    Returns
    -------
    dict
        ``{'columns': parties, 'rows': [{'scope', 'event_date', 'label', 'cells'}]}``: one row per election from
        the newest to the oldest (the same date, by scope code) with the ``key`` of each party (``None`` when
        missing). An election without any value (the next one, which has no result yet) has no row, and a party
        without any value in the rows kept has no column.
    """
    seen = []
    values = {}

    for record in records:
        party = record.get('party')

        if party not in seen:
            seen.append(party)

        election = (record.get('event_scope') or RATING_SCOPE, record.get('event_date'))
        values.setdefault(election, {})[party] = record.get(key)

    columns = [name for name in order if name in seen] + [name for name in seen if name not in order]
    rows = [
        {'scope': scope, 'event_date': date, 'label': label(scope, date) if label else fmt_date(date),
         'cells': [values[(scope, date)].get(party) for party in columns]}
        for scope, date in sorted(sorted(values, key=lambda key: key[0]), key=lambda key: key[1] or '', reverse=True)
    ]

    rows = [row for row in rows if any(cell is not None for cell in row['cells'])]
    kept = [index for index, _ in enumerate(columns) if any(row['cells'][index] is not None for row in rows)]

    return {'columns': [columns[index] for index in kept],
            'rows': [dict(row, cells=[row['cells'][index] for index in kept]) for row in rows]}


def rating_path(ratings: list, events: list, label: Callable, scope_names: dict, order: Sequence[str],
                today: date) -> dict:
    """
    The path of a pollster's rating: how its polls did at each election it polled, in every scope, and the rating
    it was left with after each one, up to today.

    Parameters
    ----------
    ratings : list of dict
        Its ratings records of every scope (``event_scope``, ``event_date``, ``rating``, ``rating_adj``), the next
        elections included (``webapi.query_pollster_path``).
    events : list of dict
        The elections it polled (``event_scope``, ``event_date``, ``n_polls``, ``error_avg``, ``error_blocks``,
        ``bias_dev_adj``).
    label : callable
        ``(scope, date) -> str``, the name of a single election.
    scope_names : dict
        ``{code: name}`` of the catalogue scopes.
    order : sequence of str
        Scope codes in catalogue order.
    today : datetime.date
        The date the path ends at.

    Returns
    -------
    dict
        ``points``: one per election date before ``today`` with polls, ascending, with ``event_date``, ``scopes``
        (the scopes polled that day, in catalogue order), ``label`` (the name of the election for one scope; the
        date and the scope names, up to three and then "y N más", for several), ``kind`` (``general`` when the
        national scope is among them, else ``regional``), ``n_polls`` (the sum), ``error_avg``, ``error_blocks``
        and ``bias_dev_adj`` (the means of its polls of that date: the error over the main parties and the signed
        error of the margin, in points, and the deviation of their bias from the comparable polls, in the bias
        scale of the rating; weighted by the polls of each scope that has a value, rounded to two decimals;
        ``None`` without any), ``rating`` (the rating it was left with after that date: the one of its first
        rated event after it, computed with every election up to that one included, the one applied until the
        next; ``None`` without a later row: the chart skips the point and the table shows a dash) and ``change``
        (that rating minus the one it arrived with: its row of that date or, without one, its last row before;
        ``None`` without either). ``today``: the same keys for today (``label`` "Hoy", ``kind`` ``today``, no
        scopes, polls, measures nor change) with the rating of the next national election or, without one, of
        the latest rated event; ``None`` without ratings.
    """
    rank = {code: index for index, code in enumerate(order)}
    by_event = {(row.get('event_scope'), row.get('event_date')): row for row in ratings}
    rated = sorted((row for row in ratings if row.get('event_date')), key=lambda row: row['event_date'])
    polled = {}
    measures = {}

    for row in events:
        polled.setdefault(row.get('event_date'), {})[row.get('event_scope')] = row.get('n_polls') or 0
        measures[(row.get('event_scope'), row.get('event_date'))] = row

    def mean_of(event_date, scopes, key):
        """Mean of a measure over the scopes of a date, weighted by their polls; ``None`` without a value."""
        pairs = [(measures[(code, event_date)].get(key), polled[event_date][code]) for code in scopes
                 if measures.get((code, event_date), {}).get(key) is not None]
        weight = sum(max(n, 1) for _, n in pairs)

        return round(sum(value * max(n, 1) for value, n in pairs) / weight, 2) if pairs else None

    limit = today.isoformat()
    points = []

    for event_date in sorted(day for day in polled if day and day < limit):
        scopes = sorted(polled[event_date], key=lambda code: (rank.get(code, len(rank)), code))
        # the rating it arrived with (its row of that date, else the last one before) and the one it was left
        # with: the next evaluation, which already counts that date
        before = next((by_event[(code, event_date)] for code in scopes if (code, event_date) in by_event), None)
        before = before if before is not None else next((row for row in reversed(rated) if row['event_date'] < event_date), {})
        after = next((row for row in rated if row['event_date'] > event_date), {})
        rating = after.get('rating')
        arrived = before.get('rating')

        if len(scopes) == 1:
            text = label(scopes[0], event_date)
        else:
            names = [scope_names.get(code, code) for code in scopes]
            shown = names if len(names) <= 3 else names[:2] + ['{} más'.format(len(names) - 2)]
            text = '{} · {} y {}'.format(fmt_date(event_date), ', '.join(shown[:-1]), shown[-1])

        points.append({
            'event_date': event_date, 'scopes': scopes, 'label': text,
            'kind': 'general' if RATING_SCOPE in scopes else 'regional', 'n_polls': sum(polled[event_date].values()),
            'error_avg': mean_of(event_date, scopes, 'error_avg'), 'error_blocks': mean_of(event_date, scopes, 'error_blocks'),
            'bias_dev_adj': mean_of(event_date, scopes, 'bias_dev_adj'), 'rating': rating,
            'change': None if rating is None or arrived is None else round(rating - arrived, 2),
        })

    national = [row for row in ratings if row.get('event_scope') == RATING_SCOPE and (row.get('event_date') or '') >= limit]

    if national:
        current = min(national, key=lambda row: row['event_date'])
    else:
        current = max(ratings, key=lambda row: row.get('event_date') or '') if ratings else None

    today_point = None if current is None else {
        'event_date': limit, 'scopes': [], 'label': 'Hoy', 'kind': 'today', 'n_polls': None, 'error_avg': None,
        'error_blocks': None, 'bias_dev_adj': None, 'rating': current.get('rating'), 'change': None,
    }

    return {'points': points, 'today': today_point}


def poll_date(event_date: Optional[str], days) -> Optional[str]:
    """
    Date of a poll published some days before an election.

    Parameters
    ----------
    event_date : str or None
        ISO date of the election.
    days : int or None
        Days between the poll and the election (``days`` of the ``poll-errors`` rows).

    Returns
    -------
    str or None
        The ISO date of the poll; ``None`` without a date or without days.
    """
    if not event_date or days is None:
        return None

    return (date.fromisoformat(event_date) - timedelta(days=int(days))).isoformat()


def last_poll_records(scope: str, rows: list, cycle: Optional[dict], results: Optional[dict], name: str) -> list:
    """
    Deviations of the last poll of a pollster in one election, by party.

    The polls are those of the published error analysis (``poll-errors``: the last poll of each pollster in the
    final window of the election, the ones of the bullseye) read, with their shares, from the ``cycle`` part of
    the election: the poll of each pollster published ``days`` days before it (the first one of the part when it
    published several that day).

    Parameters
    ----------
    scope : str
        Scope code of the election.
    rows : list of dict
        Its ``poll-errors`` rows (``pollster``, ``event_date``, ``days``), one per pollster.
    cycle : dict or None
        Its ``cycle`` part (``parties`` and ``polls`` with ``date``, ``pollster`` and one share per party), or
        ``None`` when it is not published.
    results : dict or None
        Its ``results`` part (``rows`` with ``party`` and ``pct``), or ``None``.
    name : str
        Name of the pollster.

    Returns
    -------
    list of dict
        One ``{'event_scope', 'event_date', 'party', 'result', 'consensus'}`` per party of the cycle, in its
        order (records for ``effects_table``): ``result`` is the share of the poll minus the official share
        (``None`` when the poll or the results lack the party) and ``consensus`` the share minus the plain mean of
        the last polls of the other pollsters of the analysis that are in the cycle and report the party (``None``
        without any); both in points, rounded to two decimals. Empty when the pollster has no row, or its poll is
        not in the cycle.
    """
    own = next((row for row in rows if row.get('pollster') == name), None)

    if own is None or cycle is None:
        return []

    dates = {row['pollster']: poll_date(row.get('event_date'), row.get('days')) for row in rows if row.get('pollster')}
    polls = {}

    for poll in cycle.get('polls') or []:
        pollster = poll.get('pollster')

        if pollster in dates and pollster not in polls and poll.get('date') == dates[pollster]:
            polls[pollster] = poll

    if name not in polls:
        return []

    official = {row['party']: row.get('pct') for row in (results or {}).get('rows') or []}
    others = [poll for pollster, poll in polls.items() if pollster != name]
    records = []

    for party in cycle.get('parties') or []:
        share = polls[name].get(party)
        result = official.get(party)
        reported = [poll[party] for poll in others if poll.get(party) is not None]
        records.append({
            'event_scope': scope, 'event_date': own.get('event_date'), 'party': party,
            'result': None if share is None or result is None else round(share - result, 2) or 0.0,
            'consensus': None if share is None or not reported else round(share - sum(reported) / len(reported), 2) or 0.0,
        })

    return records


def casas_context(state: dict, active: str = '/casas') -> dict:
    """
    Template context of the houses page.

    The ranking is always the national one (``RATING_SCOPE``), whatever the scope: the rating of a
    pollster does not depend on it, and its herding column comes from the national analysis for the same
    reason. The page renders every row and hides those beyond ``RANKING_SHOWN`` behind "Cargar más". The
    ratings come from the database; the house effects and the herding cards from the houses analysis of the
    scope (chosen in the house-effects card), whose ``meta`` part marks it as complete.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``ranking`` (``ranking_rows``), ``ranking_shown`` (``RANKING_SHOWN``) and
        ``ranking_hidden`` (rows beyond them, behind "Cargar más"), ``ranking_columns`` (``table_columns``), ``block_colors`` (``block_bar_colors``), ``error_max`` (the largest ``error_avg`` of
        the ranking, the fullest shade of that column; 0 without one), ``grid`` (``house_effects_grid`` or
        ``None``), ``herding`` (``herding_rows`` or ``None``), ``analysis_missing``, ``pollster_ids``
        (``{name: id}`` of the pollsters rated in the event, for the links), ``ranking_event`` (ISO date of the
        election the ranking is for, ``None`` without ratings; the card is dated at the ``run_at`` of the common
        ``meta``, like the footer), ``csv_links`` and ``initial`` (``state``, ``meta`` and ``grid``).

    Raises
    ------
    HttpError
        503 without a bundle or without a database; 404 for an unpublished scope or a missing
        run.
    """
    context = common_context(state, active)
    full = context['state']
    scope = full['scope']
    ratings = query_ratings(RATING_SCOPE)
    missing = analysis_part(scope, 'meta') is None
    effects = None if missing else analysis_part(scope, 'house-effects')
    herding = None if missing else analysis_part(scope, 'herding')
    if scope == RATING_SCOPE:
        national_herding = herding
    else:
        national_herding = None if analysis_part(RATING_SCOPE, 'meta') is None else analysis_part(RATING_SCOPE, 'herding')
    ranking = ranking_rows(ratings, national_herding)
    grid = house_effects_grid(effects) if effects is not None else None
    links = [{'label': 'Ratings de la próxima elección general', 'href': api_url(RATING_SCOPE, 'ratings', fmt='csv', kind='data')}]

    if not missing:
        links += [
            {'label': label, 'href': api_url(scope, name, fmt='csv', kind='analysis', table=table)}
            for name, table, label in ANALYSIS_CSV
            if name != 'house-effects' or effects is not None  # not published: the CSV would be a 404
        ]

    context['ranking'] = ranking
    context['ranking_shown'] = RANKING_SHOWN
    context['ranking_hidden'] = max(0, len(ranking) - RANKING_SHOWN)
    context['grid'] = grid
    context['herding'] = herding_rows(herding) if herding is not None else None
    # the columns of the ranking with their descriptions (header tooltips and glossary): the two blocks of the
    # margin name the sign of «Sesgo bloques» and colour its bar; the scale of the rubric gives the points rule
    # of «Confianza»
    blocks, context['block_colors'] = block_bar_colors(context['meta'], context['catalog'])
    context['ranking_columns'] = table_columns('ranking', blocks, load_criteria()['scale'])
    context['analysis_missing'] = missing
    context['pollster_ids'] = {
        record['pollster']: int(record['pollster_id'])
        for record in webapi.frame_to_records(ratings)
        if record.get('pollster') is not None and record.get('pollster_id') is not None
    }
    context['ranking_event'] = (
        str(ratings['event_date'].max())[:10] if 'event_date' in ratings.columns and not ratings.empty else None)
    context['csv_links'] = links
    context['initial'] = {'state': full, 'meta': context['meta'], 'grid': grid}

    return context


def profile_kpis(current: dict, mtype: Optional[str], quality_support: str) -> list:
    """
    The KPI row of a pollster profile: its rating, its prior quality and its type.

    Parameters
    ----------
    current : dict
        The current values of the pollster (``rating``, ``quality``).
    mtype : str or None
        Label of the type of the pollster.
    quality_support : str
        Text under the prior quality: the points of its evaluation, or that it is not evaluated.

    Returns
    -------
    list of dict
        Three ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` rows for ``_macros.kpis``.
    """
    return [
        {'label': 'Rating', 'value': fmt_num(current.get('rating'), 1), 'unit': '', 'support': 'entra así en el promedio',
         'ok': False, 'color': None},
        {'label': 'Confianza', 'value': fmt_num(current.get('quality'), 1), 'unit': '', 'support': quality_support,
         'ok': False, 'color': None},
        {'label': 'Tipo', 'value': mtype or '–', 'unit': '', 'support': '', 'ok': False, 'color': None},
    ]


def source_parts(source: str) -> list:
    """
    Split a source cell of the quality rubric into its text and its links.

    A cell may hold several URLs mixed with text (``https://a/ficha (ficha de 2026-10-01); https://b``). Each URL
    loses the punctuation that closes the sentence around it (``)``, ``.``, ``;``, ``,`` or ``:``; a ``)`` only
    when the URL has no matching ``(``), which goes back to the text that follows.

    Parameters
    ----------
    source : str
        The source cell, possibly empty.

    Returns
    -------
    list
        ``{'text': str, 'url': str or None}`` in the order of the cell: ``url`` is set (and equals ``text``) on the
        links and ``None`` on the plain text; no part has an empty ``text``.
    """
    parts = []
    for token in URL_PATTERN.split(source or ''):
        if URL_PATTERN.fullmatch(token):
            url, tail = token, ''
            while url[-1] in URL_TRAILING and (url[-1] != ')' or url.count('(') < url.count(')')):
                url, tail = url[:-1], url[-1] + tail
            parts.append({'text': url, 'url': url})
            token = tail
        if token:
            if parts and parts[-1]['url'] is None:
                parts[-1]['text'] += token
            else:
                parts.append({'text': token, 'url': None})
    return parts


def quality_card(record: Optional[dict], pollster: dict, criteria: dict) -> dict:
    """
    The card of the prior quality of a pollster: its evaluation against the public rubric.

    Parameters
    ----------
    record : dict or None
        The evaluation of the pollster (``webapi.query_pollster()['quality']``); ``None`` when it is not evaluated.
    pollster : dict
        The catalogue row of the pollster (``mtype`` is read).
    criteria : dict
        The catalogue of criteria (``quality.load_criteria``).

    Returns
    -------
    dict
        ``evaluated``, ``rows`` (one ``{'key', 'name', 'points', 'max', 'level', 'source', 'parts', 'void', 'national',
        'standards'}`` per criterion, built from ``quality.quality_breakdown`` and followed by the implantation;
        ``parts`` is the source split into text and links by ``source_parts``; ``void`` is True on a transparency
        that counts 0 for lack of an identifiable entity, whose ``level`` then is the note of the criterion;
        ``national`` and ``standards`` are set on the implantation row only, ``source`` and ``parts`` being those
        of the standards; empty when not evaluated), ``points``, ``max_points``, ``quality``,
        ``status``, ``draft`` (status ``borrador``), ``checked_at`` (ISO string), ``notes``, ``informative``
        (aggregators and online panels, whose polls do not enter the average), ``support`` (the text under the KPI),
        the ``floor`` and ``per_point`` of the scale and ``national`` (set when not evaluated only: whether the
        pollster has polled a general election, computed from the quality of today, which is above the floor only
        with that implantation point).
    """
    total = max_points(criteria)
    scale = criteria['scale']
    rows = []
    if record:
        breakdown = quality_breakdown(record, criteria)
        for row in breakdown['rows']:
            row = dict(row, parts=source_parts(row['source']), void=False, national=None, standards=None)
            if row['key'] == 'transparency' and breakdown['transparency_void']:
                row.update(void=True, level=criteria['criteria']['transparency'].get('note') or row['level'])
            rows.append(row)
        standards = record.get('standards_source') or ''
        implantation = breakdown['implantation']
        rows.append({'key': 'implantation', 'name': criteria['criteria']['implantation']['name'],
                     'points': implantation, 'max': criteria['criteria']['implantation']['max'],
                     'level': level_text(criteria, 'implantation', implantation), 'source': standards,
                     'parts': source_parts(standards), 'void': False, 'national': breakdown['national'],
                     'standards': int(record.get('standards') or 0) == 1})
    points = int(record['points']) if record else None
    status = record.get('status') if record else None
    return {
        'evaluated': bool(record), 'rows': rows, 'points': points, 'max_points': total,
        'quality': record.get('quality') if record else None, 'status': status, 'draft': status == 'borrador',
        'checked_at': record.get('checked_at') if record else None, 'notes': (record.get('notes') or None) if record else None,
        'informative': pollster.get('mtype') in ('aggr', 'online'),
        'support': f'{points} de {total} puntos' if record else 'sin evaluar',
        'floor': scale['floor'], 'per_point': scale['per_point'],
        'national': None if record else bool((pollster.get('quality') or 0) > scale['floor']),
    }


def casa_context(state: dict, pollster_id: int, active: str = '/casas', today: Optional[date] = None) -> dict:
    """
    Template context of the profile of a pollster.

    The profile is global, whatever the scope of the request: the header takes the national ratings
    (``RATING_SCOPE``; the rating is the same number in every scope), the rating path runs through every election
    of every scope the pollster polled, the bullseye has its last poll in every published election of every scope
    with analysis (the same elections as the error history) and the house effects and the herding tables cover
    the elections and cycles of every scope.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    pollster_id : int
        Pollster id validated by ``webapi.check_pollster``.
    active : str, optional
        ``href`` of the page in the navigation.
    today : datetime.date, optional
        The date the rating path ends at; today in Madrid by default.

    Returns
    -------
    dict
        The common context plus ``kpis`` (``profile_kpis``), ``profile`` (``webapi.query_pollster`` of every
        scope), ``mtype`` (the public label of the type of the pollster, ``mtype_label``), ``current`` (the
        ``rating`` of its last national ratings row, the next general election, ``None`` without ratings; and its
        ``quality`` of today, the catalogue one), ``quality`` (``quality_card``), ``path`` (the rows of the rating
        path table: the
        ``points`` of ``rating_path`` from the newest), ``path_points`` (the points, ascending, for the chart),
        ``today`` (``rating_path``, the end of the chart), ``last_table`` and
        ``last_consensus_table`` (``effects_table`` of the ``result`` and ``consensus`` of ``last_poll_records``:
        the deviations of its last poll in every election of the bullseye, read from the ``cycle`` and
        ``results`` parts of the election), ``effects_table`` and ``cycle_table`` (``effects_table`` of
        ``dev_result_c`` and ``dev_cycle``, with the elections named by ``election_label``), ``herding_table``
        (its herding by cycle, ``HERDING_KEYS`` plus ``label``), ``bullseye`` (its rows of the ``poll-errors``
        parts of the published elections: ``event``, ``event_date``, ``year``, ``days``, ``scope``, ``name`` and
        the error of each block; empty without analysis), ``bullseye_scopes`` (``{code, name}`` of the scopes
        with a bullseye row, in catalogue order) and ``bullseye_years`` (every year from the first row to today,
        ``year_range``), the options of the bullseye filters, ``blocks`` (the pair of ``error_blocks``, the axes of
        the bullseye), ``table_blocks`` (the same pair in the order of the table columns, ``table_blocks``),
        ``analysis_missing`` (no scope
        has a published analysis) and ``initial`` (``state``, ``meta``, ``path`` = ``path_points``, ``today``,
        ``bullseye``, ``blocks`` and ``scopes`` = ``bullseye_scopes``). Every table runs from the newest election
        to the oldest.

    Raises
    ------
    HttpError
        503 without a bundle or without a database; 404 for an unpublished scope, a missing run
        or an unknown pollster.
    """
    context = common_context(state, active)
    full = context['state']
    profile = query_pollster(pollster_id, RATING_SCOPE, every_scope=True)
    name = profile['pollster'].get('name')
    scopes = history_scopes()
    parts = poll_error_parts(scopes)
    events = history_events(scopes)
    names = {(event['scope'], event['date']): event['name'] for event in events if event.get('name')}
    catalogue = site().scopes()['scopes']
    scope_names = {row['code']: row['name'] for row in catalogue}
    blocks = error_blocks(scopes, parts)

    def label(scope, date):
        """Name of an election of the tables."""
        return election_label(scope, date, names, scope_names)

    errors = history_rows(scopes, events, parts)
    bullseye = sorted(
        (dict({key: row.get(key) for key in ('event', 'event_date', 'year', 'days', 'scope') + tuple(blocks)},
              name=label(row['scope'], row['event_date']))
         for row in errors if row.get('pollster') == name),
        key=lambda row: row['event_date'] or '', reverse=True)
    by_election = {}

    for row in errors:
        by_election.setdefault((row['scope'], row['event_date']), []).append(row)

    last_records = []
    wanted = [(row['scope'], row['event_date'], part) for row in bullseye for part in ('cycle', 'results')]

    # The parts of every election at once: each is one read of the bundle (cached under the TTL afterwards).
    with ThreadPoolExecutor(max_workers=PART_READERS) as pool:
        parts_of = dict(zip(wanted, pool.map(lambda key: event_part(*key), wanted)))

    for row in bullseye:
        scope, event_date = row['scope'], row['event_date']
        last_records += last_poll_records(scope, by_election.get((scope, event_date)) or [],
                                          parts_of[(scope, event_date, 'cycle')],
                                          parts_of[(scope, event_date, 'results')], name)

    polled = {row['scope'] for row in bullseye}
    bullseye_scopes = [{'code': scope['code'], 'name': scope['name']} for scope in scopes if scope['code'] in polled]
    day = today or datetime.now(_madrid_tz()).date()
    bullseye_years = year_range({row['year'] for row in bullseye if row.get('year') is not None}, day)
    path_data = query_pollster_path(pollster_id)
    path = rating_path(path_data['ratings'], path_data['events'], label, scope_names, [row['code'] for row in catalogue], day)
    latest = profile_rows(profile['ratings'], ('rating', 'quality', 'event_date'))
    mtype = profile['pollster'].get('mtype')
    order = list(context['catalog'])

    context['profile'] = profile
    context['mtype'] = mtype_label(mtype)
    context['current'] = {'rating': latest[-1]['rating'] if latest else None,
                          'quality': profile['pollster'].get('quality'),
                          'event_date': latest[-1]['event_date'] if latest else None}
    criteria = load_criteria()
    context['quality'] = quality_card(profile.get('quality'), profile['pollster'], criteria)
    context['kpis'] = profile_kpis(context['current'], context['mtype'], context['quality']['support'])
    # the columns of the rating path (header tooltips and glossary) and the colours of the bar of «Sesgo bloques»
    context['path_columns'] = table_columns('path', blocks, criteria['scale'])
    context['block_colors'] = block_bar_colors(context['meta'], context['catalog'])[1]
    context['path'] = path['points'][::-1]
    context['path_points'] = path['points']
    context['today'] = path['today']
    context['last_table'] = effects_table(last_records, label, order=order, key='result')
    context['last_consensus_table'] = effects_table(last_records, label, order=order, key='consensus')
    context['effects_table'] = effects_table(profile['house_effects'], label, order=order)
    context['cycle_table'] = effects_table(profile['house_effects'], label, order=order, key='dev_cycle')
    context['herding_table'] = [dict(row, label=label(row.get('event_scope') or RATING_SCOPE, row['event_date']))
                                for row in profile_rows(profile['herding'], HERDING_KEYS, newest_first=True)]
    context['bullseye'] = bullseye
    context['bullseye_scopes'] = bullseye_scopes
    context['bullseye_years'] = bullseye_years
    context['blocks'] = blocks
    context['table_blocks'] = table_blocks(blocks)
    context['analysis_missing'] = not any(scope['analysis'] for scope in scopes)
    context['initial'] = {'state': full, 'meta': context['meta'], 'path': path['points'], 'today': path['today'],
                          'bullseye': bullseye, 'blocks': blocks, 'scopes': bullseye_scopes}

    return context
