"""
View models of the page of the CIS bias study (``/analisis/sesgo-cis``): the verdicts, the three KPI rows, the three
tests of chance, the technical note, the two comparison boxes and the ``initial`` island with what the charts draw.

``analisis.lean_context`` assembles them from the published parts; each function takes the data it reads and returns
what the template renders, with ``None`` or an empty structure for what the parts do not have, never an error.
"""
from typing import Optional

from ..studies.lean.stats import sign_chance
from .cera import _get
from .filters import delta_text, fmt_int, fmt_num
from .slugs import PROFILE_PATH, slugify

LEAN = 'sesgo-cis'
# Verdict of each of the three questions of the study and the colour class of its card
VERDICTS = [('SÍ', 'q1'), ('NO', 'q3'), ('NO', 'q3')]
QUESTIONS = ('¿Se inclina el CIS hacia la izquierda desde que lo preside Tezanos?',
             '¿Puede ser azar, o un fallo general de las encuestas?',
             '¿Fue el CIS «el único que acertó» en 2023?')
# The two blocks of the study, in the order the bullseye draws them
LEFT, RIGHT = 'Izquierda', 'Derecha'
BLOCKS = [RIGHT, LEFT]
PERIODS = ('with', 'without')
TEST_KEYS = ('coin', 'dates', 'consensus')
TEST_LABELS = {'coin': '1 · Si fuera una moneda',
               'dates': '2 · Una prueba por fecha electoral',
               'consensus': '3 · Frente a las demás casas, elección a elección'}


def verdict_rows() -> list:
    """
    The three questions of the study with their verdict.

    Returns
    -------
    list of dict
        ``{'label', 'tone', 'question'}`` per question, in order.
    """
    return [{'label': label, 'tone': tone, 'question': question}
            for (label, tone), question in zip(VERDICTS, QUESTIONS)]


def _num(x, digits: int = 1) -> str:
    """``fmt_num`` with a dash for a missing figure (``fmt_num`` already returns it)."""
    return fmt_num(x, digits)


def _count(k, n) -> Optional[str]:
    """``'{k} de {n}'``, or ``None`` when one of them is missing."""
    return None if k is None or n is None else '{} de {}'.format(fmt_int(k), fmt_int(n))


def kpis(summary: Optional[dict], periods: Optional[dict]) -> list:
    """
    The three rows of the ``m.kpis`` macro.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    periods : dict or None
        ``meta.periods``, for the label of the period without Tezanos.

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` per KPI; a figure the part lacks gives a ``None`` value
        and dashes in the support.
    """
    with_, without = _get(summary, 'with') or {}, _get(summary, 'without') or {}
    row = {'unit': '', 'ok': False, 'color': None}
    mean_right = with_.get('mean_right')
    mean_lean = with_.get('mean_lean')
    right = _count(with_.get('k_right'), with_.get('n_right'))
    before = _count(_get(without, 'consensus', 'k'), _get(without, 'consensus', 'n'))
    support = ['último sondeo del CIS antes de cada elección analizada desde que lo preside Tezanos']

    if right:
        support.append('{} con la derecha subestimada'.format(right))

    return [
        dict(row, label='Elecciones con la izquierda sobreestimada', value=_count(with_.get('k_left'), with_.get('n')),
             support=' · '.join(support)),
        dict(row, label='Cuánto sobreestima a la izquierda frente a la derecha',
             value=None if mean_lean is None else _num(mean_lean) + ' puntos',
             support='de media en cada elección: {} de más al bloque de izquierdas y {} de menos al de derechas'
                     ' · sin Tezanos ({}), el mismo CIS: {}'.format(
                         _num(with_.get('mean_left')), _num(None if mean_right is None else abs(mean_right)),
                         _get(periods, 'without', 'label') or '–', delta_text(without.get('mean_lean'), 1))),
        dict(row, label='A la izquierda del consenso de las demás encuestadoras',
             value=_count(_get(with_, 'consensus', 'k'), _get(with_, 'consensus', 'n')),
             support='{} puntos más a la izquierda que la media de las otras casas en la misma elección{}'.format(
                 _num(_get(with_, 'consensus', 'mean_diff')), ' · antes, ' + before if before else '')),
    ]


def tests(summary: Optional[dict]) -> list:
    """
    The three boxes of the tests of chance.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.

    Returns
    -------
    list of dict
        ``{'key', 'label', 'value', 'text', 'highlight'}`` for ``coin``, ``dates`` and ``consensus``, in order; ``value``
        and ``text`` are ``None`` when the part lacks the test; only the third is highlighted.
    """
    with_ = _get(summary, 'with') or {}
    boxes = []

    for key in TEST_KEYS:
        chance = _get(with_, 'chance', key)
        box = {'key': key, 'label': TEST_LABELS[key], 'value': None, 'text': None, 'highlight': key == 'consensus'}
        boxes.append(box)

        if not chance:
            continue

        n, k = fmt_int(chance['n']), fmt_int(chance['k'])

        if key == 'coin':
            box['value'] = '1 entre ' + fmt_int(chance['odds'])
            box['text'] = ('Probabilidad de sobreestimar a la izquierda en {} o más de {} elecciones si cada una fuera '
                           'cara o cruz. Pero varias autonómicas comparten fecha y contexto: no son {} pruebas '
                           'independientes.').format(k, n, n)
        elif key == 'dates':
            box['value'] = '1 entre ' + fmt_int(chance['odds'])
            box['text'] = ('Agrupando las elecciones del mismo día en una sola prueba quedan {} fechas; en {} el CIS '
                           'sobreestimó a la izquierda de media.').format(n, k)
        else:
            box['value'] = '{} de {}'.format(k, n)
            box['text'] = ('Elecciones en que el CIS quedó más a la izquierda que la media de las otras encuestadoras, '
                           '{} puntos de media. Si el error fuera general, el CIS estaría unas veces a un lado y otras '
                           'al otro. Como moneda: 1 entre {}.').format(_num(_get(with_, 'consensus', 'mean_diff')),
                                                                       fmt_int(chance['odds']))

    return boxes


def tech_note(summary: Optional[dict]) -> dict:
    """
    The Student's t of the three tests in both periods, for the technical note.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.

    Returns
    -------
    dict
        ``{'with': {'all', 'dates', 'consensus', 'mean', 'sd'}, 'without': {'all', 'dates', 'consensus'}}`` with every t
        formatted to one decimal (a dash when missing); ``mean`` and ``sd`` are those of ``with.all``.
    """
    note = {period: {key: _num(_get(summary, period, 't', key, 't')) for key in ('all', 'dates', 'consensus')}
            for period in PERIODS}
    note['with'].update(mean=_num(_get(summary, 'with', 't', 'all', 'mean')),
                        sd=_num(_get(summary, 'with', 't', 'all', 'sd')))

    return note


def compare_boxes(summary: Optional[dict], periods: Optional[dict]) -> list:
    """
    The two boxes that compare the period without Tezanos with the one with him.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    periods : dict or None
        ``meta.periods``.

    Returns
    -------
    list of dict
        ``{'label', 'k', 'n', 'share', 'mean_left', 'mean_right', 'consensus_k', 'consensus_n', 'consensus_share',
        'mean_diff', 'after'}`` for ``without`` and then ``with`` (``after`` True); the shares are fractions (0-1) or
        ``None``.
    """
    boxes = []

    for period in reversed(PERIODS):
        data = _get(summary, period) or {}
        n, k = data.get('n'), data.get('k_left')
        cn, ck = _get(data, 'consensus', 'n'), _get(data, 'consensus', 'k')
        boxes.append({
            'label': _get(periods, period, 'label'),
            'k': k, 'n': n,
            'share': k / n if n and k is not None else None,
            'mean_left': data.get('mean_left'), 'mean_right': data.get('mean_right'),
            'consensus_k': ck, 'consensus_n': cn,
            'consensus_share': ck / cn if cn and ck is not None else None,
            'mean_diff': _get(data, 'consensus', 'mean_diff'),
            'after': period == 'with',
        })

    return boxes


def house_href(name: Optional[str]) -> Optional[str]:
    """Profile link of a house by its name (``/casas/{slug}``), or ``None`` without a name."""
    return '{}/{}'.format(PROFILE_PATH, slugify(name)) if name else None


def election_rows(parts: dict, period: str = 'with') -> list:
    """The rows of the ``elections`` part of a period, in the order of the part (newest first)."""
    return [row for row in _get(parts.get('elections'), 'rows') or [] if row.get('period') == period]


def house_rows(parts: dict) -> list:
    """The rows of the ``houses`` part with the ``href`` of the profile of each house."""
    return [dict(row, href=house_href(row.get('pollster'))) for row in _get(parts.get('houses'), 'rows') or []]


def _cis_share(rows: list) -> Optional[float]:
    """``share_left`` of the CIS row of the ranking, or ``None`` without it."""
    share = next((row.get('share_left') for row in rows if row.get('cis')), None)

    return share if isinstance(share, (int, float)) and not isinstance(share, bool) else None


def _other_shares(rows: list) -> list:
    """``share_left`` of every house of the ranking but the CIS that has one."""
    return [row['share_left'] for row in rows if not row.get('cis')
            and isinstance(row.get('share_left'), (int, float)) and not isinstance(row.get('share_left'), bool)]


def cis_leads(rows: list) -> bool:
    """
    Whether the CIS has the strictly highest share of elections with the left overestimated in the ranking: the
    condition of the strong titles of question 2 («el CIS es la única casa prácticamente monocolor», «Ninguna otra
    encuestadora se acerca»).

    Parameters
    ----------
    rows : list of dict
        The rows of the ``houses`` part.

    Returns
    -------
    bool
        ``True`` when the CIS has a share above every other house's (also when it is the only house with one);
        ``False`` without the CIS row or its share.
    """
    share = _cis_share(rows)

    return share is not None and all(other < share for other in _other_shares(rows))


def runner_up(rows: list) -> Optional[dict]:
    """
    The house that follows the CIS in the ranking, for the summary of question 2: the one with the highest share of
    elections with the left overestimated (then the highest lean), and how likely its count is as a coin. It only
    *follows* the CIS when its share is strictly below the CIS's; otherwise there is none.

    Parameters
    ----------
    rows : list of dict
        The rows of the ``houses`` part (``house_rows``).

    Returns
    -------
    dict or None
        ``{'name', 'href', 'k', 'n', 'odds'}`` (``odds`` formatted, ``None`` without a chance), or ``None`` without
        the CIS row, without another house with elections or when that house's share is not below the CIS's.
    """
    share = _cis_share(rows)
    others = [row for row in rows if not row.get('cis') and row.get('n') and row.get('k_left') is not None]

    if share is None or not others:
        return None

    best = max(others, key=lambda row: (row.get('share_left') or 0, row.get('mean_lean') or 0))

    if not (best.get('share_left') or 0) < share:
        return None

    chance = sign_chance(best['n'], best['k_left'])

    return {'name': best.get('pollster'), 'href': house_href(best.get('pollster')), 'k': best['k_left'],
            'n': best['n'], 'odds': fmt_int(chance['odds']) if chance else None}


def _bull(rows: list) -> list:
    """Rows of the bullseye: where the last poll of the CIS fell, per block."""
    return [{'event': row.get('event'), 'event_date': row.get('event_date'), 'days': row.get('days'),
             'scope': row.get('scope'), 'name': row.get('label'), RIGHT: row.get(RIGHT), LEFT: row.get(LEFT)}
            for row in rows]


def _distance(election: dict) -> list:
    """Houses by distance to the result, with the consensus inserted at its position."""
    rows = [{'name': house.get('pollster'), 'dist': house.get('dist'), 'cis': house.get('pollster') == 'CIS',
             'consensus': False} for house in election.get('houses') or []]
    consensus = election.get('consensus')

    if consensus:
        rows.append({'name': 'Consenso ({} casas)'.format(fmt_int(consensus.get('n'))), 'dist': consensus.get('dist'),
                     'cis': False, 'consensus': True})

    return sorted(rows, key=lambda row: (row['dist'] is None, row['dist'] or 0))


def island(parts: dict) -> dict:
    """
    What the charts of the page draw (the page script reads it).

    Parameters
    ----------
    parts : dict
        The published parts by name (``{}`` for a missing one).

    Returns
    -------
    dict
        ``bull`` (``{'blocks', 'with', 'without'}``), ``matrix`` (the part as is), ``consensus`` (one point per
        election with the CIS against the consensus), ``ranking`` (one point per house), ``election`` (the last polls
        of the election of the study by block) and ``distance`` (houses and consensus by distance to the result,
        ``[]`` without the ``election`` part).
    """
    election = parts.get('election') or {}
    houses = [{'event': house.get('pollster'), 'name': house.get('pollster'), 'event_date': election.get('date'),
               'days': house.get('days'), RIGHT: house.get('e_right'), LEFT: house.get('e_left'),
               'cis': house.get('pollster') == 'CIS'} for house in election.get('houses') or []]
    consensus = election.get('consensus')

    return {
        'bull': {'blocks': BLOCKS, 'with': _bull(election_rows(parts, 'with')),
                 'without': _bull(election_rows(parts, 'without'))},
        'matrix': parts.get('matrix') or {},
        'consensus': [{'x': row['consensus'], 'y': row.get(LEFT), 'label': row.get('label'),
                       'national': row.get('scope') == 'es'}
                      for row in election_rows(parts, 'with') if row.get('consensus') is not None],
        'ranking': [{'x': None if row.get('share_left') is None else row['share_left'] * 100,
                     'y': row.get('mean_lean'), 'name': row.get('pollster'), 'k': row.get('k_left'), 'n': row.get('n'),
                     'cis': row.get('cis')} for row in _get(parts.get('houses'), 'rows') or []],
        'election': {'blocks': BLOCKS, 'houses': houses,
                     'consensus': {'event': 'Consenso', 'name': 'Consenso de las {} casas'.format(fmt_int(consensus.get('n'))),
                                   RIGHT: consensus.get('e_right'), LEFT: consensus.get('e_left')}
                     if consensus else None},
        'distance': _distance(election) if election else [],
    }
