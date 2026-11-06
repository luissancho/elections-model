"""
View models of the page of the clandestine polls study (``/analisis/andorra``): the verdicts, the lead, the three KPI
rows, the three boxes of question 2, the tables (by election, cell by cell, by election with the model, houses and big
polls), the conditioned sentences of each question, the card of the election in progress and the ``initial`` island
with what the dumbbell draws.

``analisis.ban_context`` assembles them from the published parts; each function takes the data it reads and returns
what the template renders, with ``None`` or an empty structure for what the parts do not have, never an error.

Every strong sentence carries its condition (spec D7): the thresholds are the constants below. The card of the
election in progress (``latest_card``) is the legal guard of the page: while ``meta.next.open`` is false it carries
nothing of the ``latest`` part.
"""
from datetime import timedelta
from typing import Optional

from .cera import MONTHS_LONG, _get
from .filters import DASH, MONTHS, delta_text, fmt_int, fmt_num
from .herd import _date, _is_num, day_month, names, short_day
from .lean import house_href

BAN = 'andorra'
QUESTIONS = ('¿Aciertan más que el último sondeo legal?',
             '¿Habrían afinado el promedio?',
             '¿Quién afina en el mercado de la fruta?')
# Question 1: «SÍ» from this share of closer pairs with a binomial p below ``P_CHANCE``, «NO» below ``NO_SHARE``
YES_SHARE, NO_SHARE, P_CHANCE = 0.6, 0.5, 0.05
# Below this p the coin is «menos de 1 vez entre 10.000» (the published p has three significant figures)
P_FLOOR = 1e-4
# Question 3: «LOS GRANDES» from this many big polls with this share of them closer
BIG_MIN, BIG_SHARE = 5, 0.75
# A cell nobody saw: the result at least ``BLIND_GAP`` points from the frozen average and the consensus moving less
# than ``BLIND_MOVE`` (the sentence «Lo que no vio nadie, tampoco lo vio la fruta»)
BLIND_GAP, BLIND_MOVE = 3, 1
# Below this distance (points) the consensus «nailed» the result, and a move this small is «apenas»
NAILED = 0.05
# Errors closer than this (points per party) are a tie of the consensus and the frozen average («empate»)
TIE = 0.1
# A cell that moved away this much or more is red, not pale
STRONG_MOVE = 1
# Pairs a house needs to be named in the texts of question 3, and a scope among the best of question 1
HOUSE_MIN, SCOPE_MIN = 10, 5
# Days between the start of the LOREG ban on polls and the election
VEDA_DAYS = 6
# The parties that take the article in prose («al PSOE», «el PP»); the others go bare («a Vox», «Sumar»)
WITH_ARTICLE = ('PP', 'PSOE', 'PNV', 'PSC', 'BNG', 'PRC', 'PAR')
WORDS = ('ninguna', 'una', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez')


def _num(x, digits: int = 2) -> str:
    """An error or a difference with two decimals (``fmt_num``; a dash when missing)."""
    return fmt_num(x, digits)


def _count(k, n) -> Optional[str]:
    """``'{k} de {n}'``, or ``None`` when one of them is missing."""
    return None if k is None or n is None else '{} de {}'.format(fmt_int(k), fmt_int(n))


def words(n) -> str:
    """A small count in words, feminine («cinco», «una»; «ninguna» for 0); the figure from eleven on."""
    return WORDS[n] if isinstance(n, int) and 0 <= n < len(WORDS) else fmt_int(n)


def the(party: Optional[str]) -> str:
    """A party in prose with its article when it takes one: «el PSOE», «Vox»."""
    return 'el ' + party if party in WITH_ARTICLE else party or DASH


def to(party: Optional[str]) -> str:
    """A party after «a»: «al PSOE», «a Vox»."""
    return 'al ' + party if party in WITH_ARTICLE else 'a ' + (party or DASH)


def long_label(label: Optional[str]) -> str:
    """The label of a general election in prose: «abril de 2019» for «2019 abr.»; any other label as it is."""
    year, _, month = (label or '').partition(' ')

    if year.isdigit() and month.rstrip('.') in MONTHS:
        return '{} de {}'.format(MONTHS_LONG[MONTHS.index(month.rstrip('.'))], year)

    return label or DASH


def coin_text(p) -> Optional[str]:
    """
    How often a fair coin gets the pairs: «menos de 1 vez entre 10.000» below ``P_FLOOR``, «1 vez entre 37» below
    ``P_CHANCE``, ``None`` from ``P_CHANCE`` on (the texts then say chance cannot be ruled out) or without ``p``.
    """
    if not _is_num(p) or p >= P_CHANCE:
        return None

    return 'menos de 1 vez entre 10.000' if p < P_FLOOR else '1 vez entre ' + fmt_int(round(1 / p))


def _gain(mean_d, unit: str = 'puntos por partido') -> str:
    """The mean change of the error in words: «mejora media de 0,36 puntos por partido» (a dash without it)."""
    if not _is_num(mean_d):
        return 'mejora media ' + DASH

    if mean_d < 0:
        return 'mejora media de {} {}'.format(_num(abs(mean_d)), unit)

    return 'sin mejora media ({} {})'.format(delta_text(mean_d, 2), unit)


def _model_rows(rows: list) -> list:
    """The national rows of the ``elections`` part with a ``model`` block, oldest first."""
    return sorted((row for row in rows if row.get('national') and row.get('model')),
                  key=lambda row: row.get('event_date') or '')


def _not_beaten(models: list) -> list:
    """The rows of ``_model_rows`` where the consensus did not beat the frozen average."""
    return [row for row in models if not _get(row, 'model', 'beaten')]


def _tied(row: dict) -> bool:
    """Whether the consensus and the frozen average of an election are level (``TIE``)."""
    frozen, consensus = _get(row, 'model', 'error_frozen'), _get(row, 'model', 'error_consensus')

    return _is_num(frozen) and _is_num(consensus) and abs(consensus - frozen) < TIE


def _arrow(row: dict, key: str = 'error_consensus') -> str:
    """«2,03 → 0,84»: the frozen error and another of an election."""
    return '{} → {}'.format(_num(_get(row, 'model', 'error_frozen')), _num(_get(row, 'model', key)))


def blind_cell(cells: list, last: bool = False) -> Optional[dict]:
    """
    The cell nobody saw: the result at least ``BLIND_GAP`` points from the frozen average and the consensus moving
    less than ``BLIND_MOVE``; the one farthest from the result when several are.

    Parameters
    ----------
    cells : list of dict
        The rows of the ``cells`` part.
    last : bool, default False
        Look only at the last election of the part.

    Returns
    -------
    dict or None
        The cell, or ``None`` without one.
    """
    if last and cells:
        newest = max(row.get('event_date') or '' for row in cells)
        cells = [row for row in cells if row.get('event_date') == newest]

    blind = [row for row in cells if _is_num(row.get('gap')) and _is_num(row.get('move'))
             and _is_num(row.get('official')) and _is_num(row.get('consensus'))
             and abs(row['gap']) >= BLIND_GAP and abs(row['move']) < BLIND_MOVE]

    return max(blind, key=lambda row: abs(row['official'] - row['consensus'])) if blind else None


def verdicts(summary: Optional[dict], houses: list = ()) -> list:
    """
    The three questions of the study with their verdict (spec D7).

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    houses : list of dict, optional
        The rows of the ``houses`` part (by share descending), for the verdict of question 3 without big polls.

    Returns
    -------
    list of dict
        ``{'label', 'tone', 'question'}`` per question: P1 «SÍ» (``q1``) with ``share >= YES_SHARE`` and ``p <
        P_CHANCE``, «NO» (``q3``) with ``share < NO_SHARE``, else «A MEDIAS» (``q2``); P2 «EN k DE n» (``q2``), «SÍ»
        (``q1``) when every election was beaten, «NO» (``q3``) when none, and a ``None`` label without a model; P3
        «LOS GRANDES» (``qb``) from ``BIG_MIN`` big polls with ``BIG_SHARE`` of them closer, else «k DE n GRANDES»,
        and without big polls the name of the best house (of those with ``HOUSE_MIN`` pairs when there are; ``None``
        without houses).
    """
    share, p = _get(summary, 'pairs', 'share'), _get(summary, 'pairs', 'p')

    if _is_num(share) and _is_num(p) and share >= YES_SHARE and p < P_CHANCE:
        first = ('SÍ', 'q1')
    elif _is_num(share) and share < NO_SHARE:
        first = ('NO', 'q3')
    else:
        first = ('A MEDIAS', 'q2')

    n, k = _get(summary, 'elections', 'n'), _get(summary, 'elections', 'k_beaten')

    if not n or k is None:
        second = (None, 'q2')
    elif k == n:
        second = ('SÍ', 'q1')
    elif k == 0:
        second = ('NO', 'q3')
    else:
        second = ('EN {} DE {}'.format(k, n), 'q2')

    big_n, big_k = _get(summary, 'big', 'n') or 0, _get(summary, 'big', 'k') or 0

    if big_n >= BIG_MIN and big_k / big_n >= BIG_SHARE:
        third = 'LOS GRANDES'
    elif big_n:
        third = '{} DE {} GRANDES'.format(big_k, big_n)
    else:
        best = (_ranked(list(houses)) or list(houses))[:1]
        third = best[0].get('pollster') if best else None

    return [{'label': label, 'tone': tone, 'question': question}
            for (label, tone), question in zip((first, second, (third, 'qb')), QUESTIONS)]


def lead_text(summary: Optional[dict], cells: list, params: dict) -> str:
    """
    The live part of the lead, after its two fixed sentences.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    cells : list of dict
        The rows of the ``cells`` part.
    params : dict
        ``meta.params`` (``since``, ``chart_since``).

    Returns
    -------
    str
        With a model, the elections where the consensus beat the frozen average and the party figures closer to the
        result, then «Lo que no vio nadie, tampoco lo vio la fruta…» when the last election has a ``blind_cell``
        (otherwise a neutral sentence); without a model, the pairs.
    """
    elections, pairs = _get(summary, 'elections'), _get(summary, 'pairs') or {}

    if not elections:
        return ('En {} pares casa a casa desde {}, el sondeo clandestino estuvo más cerca del resultado que el último '
                'legal de la misma casa en {}.').format(fmt_int(pairs.get('n')),
                                                         (params.get('since') or '')[:4] or DASH, fmt_int(pairs.get('k')))

    n, k = elections.get('n'), elections.get('k_beaten')
    where = 'en todas' if k == n else 'en ninguna' if k == 0 else 'en ' + words(k)
    text = ('En las {} generales desde {}, el consenso de esos sondeos estuvo más cerca del resultado que el promedio '
            'congelado {}, y quedó más cerca del resultado en {} de las {} cifras de partido.').format(
        words(n), (params.get('chart_since') or '')[:4] or DASH, where, fmt_int(_get(summary, 'cells', 'k_closer')),
        fmt_int(_get(summary, 'cells', 'n')))
    blind = blind_cell(cells, last=True)

    if blind:
        return text + (' Lo que no vio nadie, tampoco lo vio la fruta: en {} dejó {} {} puntos por {} de lo que '
                       'sacó.').format(long_label(blind.get('label')), to(blind.get('party')),
                                       fmt_num(abs(blind['official'] - blind['consensus']), 1),
                                       'debajo' if blind['consensus'] < blind['official'] else 'encima')

    return text + ' Aquí está cada cifra, elección a elección.'


def _missed_text(missed: list) -> Optional[str]:
    """«el PSOE en 2015, el PSOE y Vox en abril de 2019»: the cells that were not closer, grouped by election."""
    groups = {}

    for cell in missed:
        groups.setdefault(cell.get('label'), []).append(the(cell.get('party')))

    return ', '.join('{} en {}'.format(names(parties), long_label(label)) for label, parties in groups.items()) or None


def kpis(summary: Optional[dict], elections: list, cells: list, params: dict) -> list:
    """
    The three rows of the ``m.kpis`` macro (spec D7).

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    elections : list of dict
        The rows of the ``elections`` part.
    cells : list of dict
        The rows of the ``cells`` part.
    params : dict
        ``meta.params``.

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` per KPI: with a model, the general elections where
        the consensus beat the frozen average, the party figures closer to the result and the pairs; without a
        model, the elections with clandestine polls, the big polls (the pairs of the general elections without big
        polls) and the pairs.
    """
    row = {'unit': '', 'ok': False, 'color': None}
    pairs = _get(summary, 'pairs') or {}
    generals, regional = pairs.get('generals') or {}, pairs.get('regional') or {}
    since = (params.get('since') or '')[:4] or DASH
    el, cs = _get(summary, 'elections'), _get(summary, 'cells') or {}
    support = ['{} pares en generales desde {} ({} más cerca) y {} en autonómicas ({})'.format(
        fmt_int(generals.get('n')), since, fmt_int(generals.get('k')), fmt_int(regional.get('n')),
        fmt_int(regional.get('k'))), _gain(pairs.get('mean_d'))]
    coin = coin_text(pairs.get('p'))

    if coin:
        support.append('una moneda lo hace ' + coin)

    third = dict(row, label='Sondeos clandestinos más cerca que el último legal de su casa',
                 value=_count(pairs.get('k'), pairs.get('n')), support=' · '.join(support))

    if not el:
        big = _get(summary, 'big') or {}
        second = (dict(row, label='Sondeos de la jornada más cerca que el último legal',
                       value=_count(big.get('k'), big.get('n')),
                       support='sondeos clandestinos con {} entrevistas o más · {}'.format(
                           fmt_int(big.get('threshold')), _gain(big.get('mean_d'))))
                  if big.get('n') else
                  dict(row, label='Sondeos clandestinos de las generales más cerca que el último legal',
                       value=_count(generals.get('k'), generals.get('n')),
                       support='{} generales desde {} · {}'.format(fmt_int(generals.get('n_events')), since,
                                                                   _gain(generals.get('mean_d')))))
        return [
            dict(row, label='Elecciones con sondeos clandestinos',
                 value=fmt_int((generals.get('n_events') or 0) + (regional.get('n_events') or 0)),
                 support='{} generales y {} autonómicas desde {}, con un último sondeo legal de la misma casa con el '
                         'que comparar'.format(fmt_int(generals.get('n_events')), fmt_int(regional.get('n_events')),
                                               since)),
            second,
            third,
        ]

    models = _model_rows(elections)
    first = ['error medio en los partidos principales: del {} del promedio congelado al {} del consenso '
             'clandestino'.format(_num(el.get('mean_frozen')), _num(el.get('mean_consensus')))]
    beaten = [model for model in models if _get(model, 'model', 'beaten')]

    if beaten:
        best = max(beaten, key=lambda model: model['model']['error_frozen'] - model['model']['error_consensus'])
        first.append('en {}, de {} a {}'.format(long_label(best.get('label')), _num(best['model']['error_frozen']),
                                                _num(best['model']['error_consensus'])))

    missed = _not_beaten(models)

    if missed:
        first.append(', '.join('en {}, de {} a {}'.format(long_label(model.get('label')),
                                                           _num(_get(model, 'model', 'error_frozen')),
                                                           _num(_get(model, 'model', 'error_consensus')))
                               for model in missed) + (', la única que no' if len(missed) == 1 else ', las que no'))

    parties = {}

    for cell in cells:
        parties[cell.get('event_date')] = parties.get(cell.get('event_date'), 0) + 1

    second = []

    if parties:
        lo, hi = min(parties.values()), max(parties.values())
        count = lambda x: words(x) if x >= 2 else fmt_int(x)  # noqa: E731 (masculine: «un partido» stays a figure)
        second.append('{} generales, {} partidos principales cada una'.format(
            words(len(parties)), 'con ' + count(lo) if lo == hi else 'de {} a {}'.format(count(lo), count(hi))))

    second.append('{} apuntaron en la dirección correcta y {} pasaron de largo'.format(
        fmt_int(cs.get('k_direction')), fmt_int(cs.get('k_crossed'))))
    missed_cells = _missed_text(cs.get('missed') or [])

    if missed_cells:
        second.append('las que no: ' + missed_cells)

    return [
        dict(row, label='Generales en que la fruta afinó el promedio congelado',
             value=_count(el.get('k_beaten'), el.get('n')), support=' · '.join(first)),
        dict(row, label='Cifras de partido que quedaron más cerca del resultado',
             value=_count(cs.get('k_closer'), cs.get('n')), support=' · '.join(second)),
        third,
    ]


def _seen(cells: list, count: int = 3) -> list:
    """The cells that moved the most toward the result without crossing it, largest move first."""
    return sorted((cell for cell in cells if cell.get('closer') and not cell.get('crossed')
                   and _is_num(cell.get('move')) and _is_num(cell.get('gap'))),
                  key=lambda cell: -abs(cell['move']))[:count]


def _seen_text(cells: list, count: int = 3) -> Optional[str]:
    """«Podemos y Cs en 2015, el PP en abril de 2019»: the ``_seen`` cells, by election."""
    return _missed_text(sorted(_seen(cells, count), key=lambda cell: cell.get('event_date') or ''))


def tests(summary: Optional[dict], elections: list, cells: list) -> list:
    """
    The three boxes of question 2 (``[]`` without a model).

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    elections : list of dict
        The rows of the ``elections`` part.
    cells : list of dict
        The rows of the ``cells`` part.

    Returns
    -------
    list of dict
        ``{'key', 'label', 'value', 'note', 'text', 'tone'}`` for ``beaten``, ``mean`` and ``cells``; ``tone`` is
        ``'ok'`` for the first when the consensus beat the frozen average in most elections, ``''`` otherwise.
    """
    el = _get(summary, 'elections')

    if not el:
        return []

    cs = _get(summary, 'cells') or {}
    models = _model_rows(elections)
    beaten = [model for model in models if _get(model, 'model', 'beaten')]
    first = []

    if beaten:
        first.append(names(['{} ({})'.format(long_label(model.get('label')), _arrow(model)) for model in beaten]) + '.')

    for model in _not_beaten(models):
        first.append(('En {}, empate: {} frente a {}.' if _tied(model) else 'En {}, no: {} frente a {}.').format(
            long_label(model.get('label')), _num(_get(model, 'model', 'error_frozen')),
            _num(_get(model, 'model', 'error_consensus'))))

    frozen, consensus, legal = el.get('mean_frozen'), el.get('mean_consensus'), el.get('mean_legal')
    second = None

    if _is_num(frozen) and _is_num(consensus):
        second = '{} puntos {} por partido.'.format(_num(abs(frozen - consensus)),
                                                   'menos' if consensus < frozen else 'más')

        if _is_num(legal) and frozen < legal:
            second += (' Para situarlo: el promedio congelado ya bate al último sondeo legal medio ({} frente a '
                       '{}).').format(_num(frozen), _num(legal))

    half = cs.get('k_half')
    third = ['En {} de ellas recorrió más de la mitad del camino.'.format(fmt_int(half)) if half else
             'En ninguna recorrió más de la mitad del camino.']
    seen, blind = _seen_text(cells), blind_cell(cells)

    if seen:
        third.append('Lo más visto: {}.'.format(seen))

    if blind:
        third.append('Lo que nadie vio: {} de {}.'.format(the(blind.get('party')), long_label(blind.get('label'))))

    n, k = el.get('n'), el.get('k_beaten')

    return [
        {'key': 'beaten', 'label': '1 · Elecciones en que el consenso clandestino batió al promedio congelado',
         'value': _count(k, n), 'note': None, 'text': ' '.join(first) or None,
         'tone': 'ok' if n and k is not None and k * 2 > n else ''},
        {'key': 'mean', 'label': '2 · Error medio de las {}'.format(words(n)), 'value': _num(consensus),
         'note': 'desde ' + _num(frozen) if _is_num(frozen) else None, 'text': second, 'tone': ''},
        {'key': 'cells', 'label': '3 · Cifras que quedaron más cerca del resultado',
         'value': _count(cs.get('k_closer'), cs.get('n')), 'note': None, 'text': ' '.join(third), 'tone': ''},
    ]


def answers(summary: Optional[dict], elections: list, houses: list) -> list:
    """
    The sentence of each card-link of the summary («Tres preguntas, tres respuestas»).

    Returns
    -------
    list of str
        One per question: the pairs (and those of the general elections); the elections beaten by the consensus and
        the closer cells, «aunque se quede corto» when fewer than half of them went half the way (without a model,
        that there is none); the big polls and the two best houses with ``HOUSE_MIN`` pairs or more.
    """
    pairs = _get(summary, 'pairs') or {}
    first = ('En {} de {} pares el sondeo clandestino estuvo más cerca del resultado que el último que la misma casa '
             'publicó antes de la veda. En generales, {}.').format(
        fmt_int(pairs.get('k')), fmt_int(pairs.get('n')),
        _count(_get(pairs, 'generals', 'k'), _get(pairs, 'generals', 'n')) or DASH)
    el, cs = _get(summary, 'elections'), _get(summary, 'cells') or {}

    if not el:
        second = 'No hay generales con modelo publicado: sin promedio congelado no hay con qué comparar.'
    else:
        beaten = [long_label(row.get('label')) for row in _model_rows(elections) if _get(row, 'model', 'beaten')]
        frozen, consensus = el.get('mean_frozen'), el.get('mean_consensus')
        margin = (', por {} puntos de media'.format(_num(frozen - consensus))
                  if _is_num(frozen) and _is_num(consensus) and consensus < frozen else '')
        second = ('El consenso clandestino bate al promedio congelado en {}{}.'.format(names(beaten), margin) if beaten
                  else 'El consenso clandestino no bate al promedio congelado en ninguna de las {}.'.format(
                      words(el.get('n'))))
        short = cs.get('k_half') is not None and cs.get('k_closer') and cs['k_half'] * 2 < cs['k_closer']
        second += ' Y queda más cerca del resultado en {} cifras{}.'.format(
            _count(cs.get('k_closer'), cs.get('n')) or DASH, ', aunque se quede corto' if short else '')

    big = _get(summary, 'big') or {}
    third = ('Los sondeos de la jornada con {} entrevistas o más afinaron en {}.'.format(
        fmt_int(big.get('threshold')), _count(big.get('k'), big.get('n'))) if big.get('n') else
        'Ningún sondeo clandestino llegó a {} entrevistas.'.format(fmt_int(big.get('threshold'))))
    top = _ranked(houses)[:2]

    if top:
        third += ' {} afina en {} de sus {} vedas'.format(top[0].get('pollster'), fmt_int(top[0].get('k')),
                                                         fmt_int(top[0].get('n')))
        third += '; {}, en {}.'.format(top[1].get('pollster'), _count(top[1].get('k'), top[1].get('n'))) \
            if len(top) > 1 else '.'

    return [first, second, third]


def _ranked(houses: list) -> list:
    """The houses with ``HOUSE_MIN`` pairs or more, by share and then pairs descending."""
    return sorted((row for row in houses if (row.get('n') or 0) >= HOUSE_MIN and _is_num(row.get('share'))),
                  key=lambda row: (-row['share'], -row['n']))


def p1_sum(summary: Optional[dict]) -> dict:
    """
    The summary sentence of question 1.

    Returns
    -------
    dict
        ``{'strong', 'rest'}``: the share of closer pairs in tenths with the word of the verdict («Sí:», «A
        medias:», «No: sólo»); the mean improvement, the general elections, the coin only with ``p < P_CHANCE``
        (otherwise that chance cannot be ruled out) and «La última semana mueve voto…» only with «SÍ».
    """
    pairs = _get(summary, 'pairs') or {}
    label = verdicts(summary)[0]['label']
    share = pairs.get('share')
    tenths = fmt_int(round(share * 10)) if _is_num(share) else DASH
    lead = {'SÍ': 'Sí: en {}', 'NO': 'No: sólo en {}'}.get(label, 'A medias: en {}').format(tenths)
    strong = lead + (' de cada 10 pares el sondeo clandestino estuvo más cerca del resultado que el último legal de '
                     'su casa')
    mean_d = pairs.get('mean_d')
    rest = [', con una {}.'.format(_gain(mean_d)) if _is_num(mean_d) and mean_d < 0 else '; {}.'.format(_gain(mean_d))]
    generals = _count(_get(pairs, 'generals', 'k'), _get(pairs, 'generals', 'n'))

    if generals:
        rest.append(' En generales, {}.'.format(generals))

    coin = coin_text(pairs.get('p'))
    k, n = pairs.get('k'), (pairs.get('n') or 0) - (pairs.get('ties') or 0)

    if coin:
        rest.append(' Que una moneda acierte {} veces de {} pasa {}: no es azar.'.format(fmt_int(k), fmt_int(n), coin))
    else:
        rest.append(' Con estas cifras no se puede descartar el azar.')

    if label == 'SÍ':
        rest.append(' La última semana mueve voto, y los sondeos clandestinos lo recogen.')

    return {'strong': strong, 'rest': ''.join(rest)}


def p2_sum(summary: Optional[dict], elections: list, cells: list) -> Optional[dict]:
    """
    The summary sentence of question 2 (``None`` without a model).

    Returns
    -------
    dict or None
        ``{'strong', 'rest'}``: the elections where the consensus would have refined the average («y en la otra no lo
        habría estropeado» only when every other one is a tie), the closer cells («lo que se mueve en la última
        semana, la fruta lo ve» from ``YES_SHARE`` of them), the ``blind_cell`` («Lo que no ve es la sorpresa»)
        and the counterfactual only with ``mean_with``.
    """
    el = _get(summary, 'elections')

    if not el:
        return None

    n, k = el.get('n'), el.get('k_beaten')
    models = _model_rows(elections)
    strong = ('En las {} generales' if k == n else 'En ninguna de las {} generales' if k == 0 else
              'En {} de {{}} generales'.format(fmt_int(k))).format(fmt_int(n))
    strong += ' los sondeos clandestinos habrían afinado el promedio'
    others = _not_beaten(models)
    rest = []

    if k and others and all(_tied(row) for row in others):
        rest.append(', y en {} no lo habría estropeado'.format('la otra' if len(others) == 1 else
                                                              'las otras ' + words(len(others))))

    rest.append('.')
    cs = _get(summary, 'cells') or {}
    kc, nc = cs.get('k_closer'), cs.get('n')

    if nc:
        rest.append(' El consenso clandestino queda más cerca del resultado en {} cifras{}'.format(
            _count(kc, nc), ': lo que se mueve en la última semana, la fruta lo ve.' if kc >= YES_SHARE * nc else '.'))

    blind = blind_cell(cells)

    if blind:
        rest.append(' Lo que no ve es la sorpresa: en {} {} {} {} cuando le faltaban {}.'.format(
            long_label(blind.get('label')), 'subió' if blind['move'] > 0 else 'bajó', to(blind.get('party')),
            fmt_num(abs(blind['move']), 1), fmt_num(abs(blind['gap']), 1)))

    frozen, with_ = el.get('mean_frozen'), el.get('mean_with')

    if _is_num(frozen) and _is_num(with_):
        with_rows = [row for row in models if _is_num(_get(row, 'model', 'error_with'))]
        better = [row.get('label') for row in with_rows if _get(row, 'model', 'beaten_with')]
        same = better == [row.get('label') for row in models if _get(row, 'model', 'beaten')]
        text = (' Y el propio modelo, recalculado el día de la votación con los sondeos clandestinos dentro, habría {} '
                'su error de {} a {}').format('bajado' if with_ < frozen else 'subido', _num(frozen), _num(with_))

        if better:
            text += ': mejor en las mismas {} elecciones'.format(words(len(better))) if same else \
                ': mejor en {} de {}'.format(fmt_int(len(better)), fmt_int(len(with_rows)))

        worse = [row for row in with_rows if not _get(row, 'model', 'beaten_with')]

        if worse:
            text += ', peor en ' + names(['{} (de {} a {})'.format(long_label(row.get('label')),
                                                                     _num(row['model']['error_frozen']),
                                                                     _num(row['model']['error_with']))
                                           for row in worse])

        rest.append(text + '.')

    return {'strong': strong, 'rest': ''.join(rest)}


def moves_text(cells: list, count: int = 3) -> Optional[str]:
    """
    The technical note «Lo que se movió y lo que faltaba» of question 2 (``None`` without cells).

    Returns
    -------
    str or None
        The ``count`` cells that moved the most toward the result without crossing it (``_seen``), the one left
        farthest from the result and, only with a ``blind_cell``, «La fruta corrige la última semana; no corrige el
        sesgo del ciclo».
    """
    rows = [cell for cell in cells if _is_num(cell.get('official')) and _is_num(cell.get('consensus'))]
    moved = _seen(cells, count)

    if not rows or not moved:
        return None

    far = max(rows, key=lambda cell: abs(cell['official'] - cell['consensus']))
    text = 'lo que más se movió, {}; lo que más faltó, {} en {}, que se quedó a {} puntos del resultado.'.format(
        '; '.join('{} en {} ({} de {})'.format(the(cell.get('party')), long_label(cell.get('label')),
                                               delta_text(cell['move'], 1), delta_text(cell['gap'], 1))
                  for cell in moved),
        the(far.get('party')), long_label(far.get('label')), fmt_num(abs(far['official'] - far['consensus']), 1))

    if blind_cell(cells):
        text += ' La fruta corrige la última semana; no corrige el sesgo del ciclo.'

    return text


def p3_sum(summary: Optional[dict], houses: list) -> dict:
    """
    The summary sentence of question 3.

    Returns
    -------
    dict
        ``{'strong', 'rest'}``: the big polls («Los grandes afinan:» only with the verdict «LOS GRANDES»; without
        big polls, that none reached the threshold) and, when there are, the two best houses with ``HOUSE_MIN`` pairs
        or more and the worst of them («sin mejora media» when its mean change is below ``NAILED``).
    """
    big = _get(summary, 'big') or {}

    if big.get('n'):
        strong = ('{} de los {} sondeos de la jornada con {} entrevistas o más estuvieron más cerca que el último '
                  'legal de su casa').format(fmt_int(big.get('k')), fmt_int(big.get('n')),
                                             fmt_int(big.get('threshold')))

        if verdicts(summary)[2]['label'] == 'LOS GRANDES':
            strong = 'Los grandes afinan: ' + strong

        mean_d = big.get('mean_d')
        rest = [', con una mejora de {} puntos por partido.'.format(_num(abs(mean_d)))
                if _is_num(mean_d) and mean_d < 0 else '.']
    else:
        strong = 'Ningún sondeo clandestino llegó a {} entrevistas.'.format(fmt_int(big.get('threshold')))
        rest = []

    ranked = _ranked(houses)

    if ranked:
        top = ranked[:2]
        rest.append(' Entre las casas con {} vedas o más, {} {}'.format(
            words(HOUSE_MIN), names(['{} ({})'.format(row.get('pollster'), _count(row.get('k'), row.get('n')))
                                     for row in top]),
            'son las que más afinan' if len(top) > 1 else 'es la que más afina'))

        if len(ranked) > 2:
            low = ranked[-1]
            flat = _is_num(low.get('mean_d')) and abs(low['mean_d']) < NAILED
            rest.append('; la que menos, {} ({}{})'.format(low.get('pollster'), _count(low.get('k'), low.get('n')),
                                                           ', sin mejora media' if flat else ''))

        rest.append('.')

    return {'strong': strong, 'rest': ''.join(rest)}


def election_rows(elections: Optional[dict], summary: Optional[dict], params: dict) -> list:
    """
    The rows of the table «Por elección» of question 1.

    Parameters
    ----------
    elections : dict or None
        The ``elections`` part (``rows`` and ``period``).
    summary : dict or None
        The ``summary`` part, for the three sums.
    params : dict
        ``meta.params`` (``since``, ``chart_since``).

    Returns
    -------
    list of dict
        ``{'label', 'note', 'n', 'k', 'share', 'mean_d', 'sum', 'sep'}``: the ``period`` row («1996 – 2011 (5
        generales)», with the last year and the count of the general elections before ``chart_since`` that the rows
        have, else the year before ``chart_since`` and the general elections of the summary not in the rows), the
        general elections since ``chart_since`` oldest first (``k`` and ``mean_d`` ``None`` without pairs), then
        Generales, Autonómicas and Todas (``sum``; the first with ``sep``).
    """
    rows = _get(elections, 'rows') or []
    chart_since = params.get('chart_since') or ''
    national = sorted((row for row in rows if row.get('national') and (row.get('event_date') or '') >= chart_since),
                      key=lambda row: row.get('event_date') or '')
    early = [row for row in rows if row.get('national') and (row.get('event_date') or '') < chart_since
             and row.get('n_pairs')]
    pairs = _get(summary, 'pairs') or {}
    period = _get(elections, 'period') or {}
    out = []

    if period.get('n'):
        count = len(early) if early else max((_get(pairs, 'generals', 'n_events') or 0)
                                             - sum(1 for row in national if row.get('n_pairs')), 0)
        end = max(row['event_date'][:4] for row in early) if early else \
            str(int(chart_since[:4]) - 1) if chart_since[:4].isdigit() else DASH
        out.append({'label': '{} – {}'.format((period.get('since') or params.get('since') or '')[:4] or DASH, end),
                    'note': '({} {})'.format(fmt_int(count), 'general' if count == 1 else 'generales'),
                    'n': period.get('n'), 'k': period.get('k'), 'share': None, 'mean_d': period.get('mean_d'),
                    'sum': False, 'sep': False})

    for row in national:
        has = bool(row.get('n_pairs'))
        out.append({'label': row.get('label'), 'note': None, 'n': row.get('n_pairs') or 0,
                    'k': row.get('k_closer') if has else None, 'share': None,
                    'mean_d': row.get('mean_d') if has else None, 'sum': False, 'sep': False})

    regional_events = _get(pairs, 'regional', 'n_events')

    for label, block, note in (('Generales', pairs.get('generals'), None),
                               ('Autonómicas', pairs.get('regional'),
                                '({} {})'.format(fmt_int(regional_events), 'elección' if regional_events == 1 else
                                                 'elecciones') if regional_events is not None else None),
                               ('Todas', pairs, None)):
        block = block or {}
        n, k = block.get('n'), block.get('k')
        out.append({'label': label, 'note': note, 'n': n, 'k': k, 'share': k / n if n and k is not None else None,
                    'mean_d': block.get('mean_d'), 'sum': True, 'sep': label == 'Generales'})

    return out


def exception_text(summary: Optional[dict], scope_names: dict) -> Optional[str]:
    """
    The note under the table «Por elección»: the scope that fine-tunes the least and those that do the most.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part (``exception``, ``pairs.by_scope``, ``pairs.share``).
    scope_names : dict
        Scope code → name.

    Returns
    -------
    str or None
        «Cataluña es la excepción: 17 pares y sólo 6 más cerca.» when its share is below ``NO_SHARE`` («La que menos
        afina es…» otherwise), then the regional scopes with ``SCOPE_MIN`` pairs or more and a share above the
        whole one, best first (up to three); ``None`` without an exception.
    """
    exception = _get(summary, 'exception') or {}
    n, k = exception.get('n'), exception.get('k')

    if not n or k is None:
        return None

    name = scope_names.get(exception.get('scope'), exception.get('scope'))
    text = ('{} es la excepción: {} pares y sólo {} más cerca.' if k / n < NO_SHARE else
            'La que menos afina es {}: {} pares, {} más cerca.').format(name, fmt_int(n), fmt_int(k))
    share = _get(summary, 'pairs', 'share')
    best = sorted((row for row in _get(summary, 'pairs', 'by_scope') or []
                   if row.get('scope') not in ('es', exception.get('scope')) and (row.get('n') or 0) >= SCOPE_MIN
                   and row.get('k') is not None and _is_num(share) and row['k'] / row['n'] > share),
                  key=lambda row: -row['k'] / row['n'])[:3]

    if best:
        text += ' {}, {}.'.format(names(['{} ({})'.format(scope_names.get(row['scope'], row['scope']),
                                                          _count(row['k'], row['n'])) for row in best]),
                                  'la que más afina' if len(best) == 1 else 'las que más afinan')

    return text


def movement(cell: dict) -> dict:
    """
    What the consensus did with a party figure, for the last column of the cells table.

    Returns
    -------
    dict
        ``{'text', 'tone'}``: «clavado · +3,1 de +3,1» (``ok``) within ``NAILED`` of the result; «pasó de largo ·
        −2,3 cuando faltaban −0,7» (``kol``) or «pasó de largo, pero más cerca · …» (``okl``) when it crossed the
        result; «hacia el resultado · +0,3 de +1,0» (``ok`` past half the way, else ``okl``; «apenas (−0,05)» for a
        move within ``NAILED``); «se alejó · −0,4» (``ko`` from ``STRONG_MOVE``, else ``kol``); «no se movió»
        (``mut``); a dash (``mut``) without the figures.
    """
    move, gap = cell.get('move'), cell.get('gap')
    consensus, official = cell.get('consensus'), cell.get('official')

    if not all(_is_num(x) for x in (move, gap, consensus, official)):
        return {'text': DASH, 'tone': 'mut'}

    if abs(consensus - official) < NAILED:
        return {'text': 'clavado · {} de {}'.format(delta_text(move, 1), delta_text(gap, 1)), 'tone': 'ok'}

    if cell.get('crossed'):
        verb = 'faltaba' if fmt_num(abs(gap), 1) == '1,0' else 'faltaban'
        text = '{} · {} cuando {} {}'.format('pasó de largo, pero más cerca' if cell.get('closer') else 'pasó de largo',
                                             delta_text(move, 1), verb, delta_text(gap, 1))
        return {'text': text, 'tone': 'okl' if cell.get('closer') else 'kol'}

    if cell.get('closer'):
        if abs(move) <= NAILED:
            return {'text': 'hacia el resultado · apenas ({})'.format(delta_text(move, 2)), 'tone': 'okl'}

        return {'text': 'hacia el resultado · {} de {}'.format(delta_text(move, 1), delta_text(gap, 1)),
                'tone': 'ok' if cell.get('half') else 'okl'}

    if abs(move) <= NAILED:
        return {'text': 'no se movió', 'tone': 'mut'}

    return {'text': 'se alejó · ' + delta_text(move, 1), 'tone': 'ko' if abs(move) >= STRONG_MOVE else 'kol'}


def cell_groups(cells: list) -> list:
    """
    The cells table of question 2, grouped by election in the order of the part.

    Returns
    -------
    list of dict
        ``{'label', 'event_date', 'rows'}`` per election; each row is the cell with its ``movement`` (``text``,
        ``tone``).
    """
    groups = []

    for cell in cells:
        if not groups or groups[-1]['event_date'] != cell.get('event_date'):
            groups.append({'label': cell.get('label'), 'event_date': cell.get('event_date'), 'rows': []})

        groups[-1]['rows'].append(dict(cell, **movement(cell)))

    return groups


def model_rows(elections: list, summary: Optional[dict]) -> list:
    """
    The table «Por elección: error medio en los partidos principales» of question 2.

    Returns
    -------
    list of dict
        ``{'label', 'n_ban', 'legal', 'frozen', 'consensus', 'with', 'beaten', 'beaten_with', 'mean'}`` per general
        election with a model, oldest first, then the «Media» row of ``summary.elections`` (``mean`` true, the
        flags ``None``); ``[]`` without a model.
    """
    el = _get(summary, 'elections')

    if not el:
        return []

    rows = [{'label': row.get('label'), 'n_ban': row.get('n_ban'), 'legal': row['model'].get('error_legal'),
             'frozen': row['model'].get('error_frozen'), 'consensus': row['model'].get('error_consensus'),
             'with': row['model'].get('error_with'), 'beaten': row['model'].get('beaten'),
             'beaten_with': row['model'].get('beaten_with'), 'mean': False} for row in _model_rows(elections)]
    rows.append({'label': 'Media', 'n_ban': None, 'legal': el.get('mean_legal'), 'frozen': el.get('mean_frozen'),
                 'consensus': el.get('mean_consensus'), 'with': el.get('mean_with'), 'beaten': None,
                 'beaten_with': None, 'mean': True})

    return rows


def house_rows(houses: list) -> list:
    """The rows of the houses table with the ``href`` of the profile of each house (``lean.house_href``)."""
    return [dict(row, href=house_href(row.get('pollster'))) for row in houses]


def big_rows(houses: Optional[dict]) -> list:
    """The big polls of the ``houses`` part (``big_rows``), in its order (sample size descending)."""
    return list(_get(houses, 'big_rows') or [])


def panel_names(houses: list, pairs: list, params: dict) -> list:
    """
    The houses of the table that are open panels: out of the consensus but compared with themselves in the pairs
    (``exclude_mtypes_consensus`` minus ``exclude_mtypes_pairs``), for the note under the houses table.
    """
    panels = set(params.get('exclude_mtypes_consensus') or []) - set(params.get('exclude_mtypes_pairs') or [])
    ids = {pair.get('pollster_id') for pair in pairs if pair.get('mtype') in panels}

    return [row.get('pollster') for row in houses if row.get('pollster_id') in ids]


def latest_card(latest: Optional[dict], next_: Optional[dict]) -> dict:
    """
    The card «El 29-N»: closed until the election in progress is published with its result (the legal guard).

    Parameters
    ----------
    latest : dict or None
        The ``latest`` part.
    next_ : dict or None
        ``meta.next`` (``{event_date, open}``).

    Returns
    -------
    dict
        ``open`` (``next.open`` and ``latest.event_date == next.event_date``), ``event_date``, ``year``,
        ``day_month`` («29 de noviembre»), ``short`` («29-N»), ``veda_from`` (the day the veda starts, ``VEDA_DAYS``
        before) and, only when open, ``parties``, ``polls`` (with the ``href`` of each house), ``rows`` (consensus,
        frozen average and result: ``{'label', 'pct', 'error'}``) and ``error``; closed, those are empty and nothing of
        ``latest`` is in the card.
    """
    event_date = _get(next_, 'event_date')
    day = _date(event_date)
    is_open = bool(_get(next_, 'open')) and bool(latest) and _get(latest, 'event_date') == event_date
    card = {'open': is_open, 'event_date': event_date, 'year': str(day.year) if day else None,
            'day_month': day_month(event_date), 'short': short_day(event_date),
            'veda_from': day_month((day - timedelta(days=VEDA_DAYS)).isoformat()) if day else None,
            'parties': [], 'polls': [], 'rows': [], 'error': None}

    if not is_open:
        return card

    card.update(
        parties=list(latest.get('parties') or []),
        polls=[dict(poll, href=house_href(poll.get('pollster'))) for poll in latest.get('polls') or []],
        rows=[{'label': label, 'pct': latest.get(key) or {}, 'error': _get(latest, 'error', error)}
              for label, key, error in (('Consenso clandestino', 'consensus', 'consensus'),
                                        ('Promedio congelado', 'frozen', 'frozen'), ('Resultado', 'official', None))],
        error=latest.get('error') or {},
    )

    return card


def chart_label(row: dict) -> str:
    """The label of a pair in the dumbbell: «GESOP · 2015», «GAD3 · 19 abr.» (the year cut when the month is in)."""
    label = row.get('label') or ''
    year, _, month = label.partition(' ')

    return '{} · {}'.format(row.get('pollster'), year[2:] + ' ' + month if year.isdigit() and month else label)


def island(pairs: Optional[dict], params: dict) -> dict:
    """
    What the dumbbell draws (the page script reads it).

    Parameters
    ----------
    pairs : dict or None
        The ``pairs`` part.
    params : dict
        ``meta.params`` (``chart_since``).

    Returns
    -------
    dict
        ``{'pairs': [{'label', 'error_legal', 'error_ban'}]}`` for the pairs of the general elections since
        ``chart_since``, oldest election first and by ``d`` ascending inside each.
    """
    since = params.get('chart_since') or ''
    rows = [row for row in _get(pairs, 'rows') or [] if row.get('national') and (row.get('event_date') or '') >= since]
    rows.sort(key=lambda row: (row.get('event_date') or '', row['d'] if _is_num(row.get('d')) else 0))

    return {'pairs': [{'label': chart_label(row), 'error_legal': row.get('error_legal'),
                       'error_ban': row.get('error_ban')} for row in rows]}


def pills(summary: Optional[dict], params: dict, card: dict) -> dict:
    """
    The pills of the head.

    Returns
    -------
    dict
        ``{'evidence', 'state', 'state_tone'}``: «Evidencia histórica · 5 generales con modelo · 116 sondeos
        clandestinos desde 1996, con las autonómicas» (without the model clause without a model); «Se completa con
        2026 tras el cierre de urnas del 29-N» (``warn``) while the card is closed, «Completada con 2026» (``ok``)
        once open, ``None`` without the election in progress.
    """
    n = _get(summary, 'elections', 'n')
    evidence = ['Evidencia histórica']

    if n:
        evidence.append('{} generales con modelo'.format(fmt_int(n)))

    evidence.append('{} sondeos clandestinos desde {}, con las autonómicas'.format(
        fmt_int(_get(summary, 'pairs', 'n')), (params.get('since') or '')[:4] or DASH))

    if not card.get('year'):
        state = None
    elif card['open']:
        state = 'Completada con {}'.format(card['year'])
    else:
        state = 'Se completa con {} tras el cierre de urnas del {}'.format(card['year'], card['short'])

    return {'evidence': ' · '.join(evidence), 'state': state, 'state_tone': 'ok' if card.get('open') else 'warn'}


def counts(summary: Optional[dict], params: dict) -> dict:
    """
    The small counts the fixed texts say in words.

    Returns
    -------
    dict
        ``model`` (the general elections with a model, «cinco»), ``horizon`` (the days of the frozen average,
        «seis»), ``min_pairs`` (the pairs a house needs for its table, «tres») and ``house_min`` (those it needs to be
        named in the texts, «diez»).
    """
    return {'model': words(_get(summary, 'elections', 'n') or 0), 'horizon': words(params.get('frozen_horizon')),
            'min_pairs': words(params.get('min_pairs')), 'house_min': words(HOUSE_MIN)}
