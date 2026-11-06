"""
View models of the page of the shy-vote study (``/analisis/voto-oculto``): the verdicts, the three KPI rows, the
table of question 1, the three boxes of question 2, the table of question 3, the conditions and the prose fragments of
the texts, the card «Después del 29-N» and the ``initial`` island with what the three charts draw.

``analisis.shy_context`` assembles them from the published parts; each function takes the data it reads and returns
what the template renders, with ``None`` or an empty structure for what the parts do not have, never an error. Errors
are polls minus result: negative means the party got more than the polls said (underestimated).
"""
import re
from datetime import date, datetime
from typing import Optional

from .cera import MONTHS_LONG, _get
from .common import days_left as days_to
from .filters import DASH, MONTHS, _madrid_tz, delta_text, fmt_int, fmt_num
from .herd import _date, day_month, names, short_day

SHY = 'voto-oculto'
QUESTIONS = ('¿Subestiman las encuestas a la izquierda?',
             '¿Hay voto oculto de Vox?',
             '¿De quién es el voto oculto, entonces?')
# Colour class of each verdict; any other «DE …» (a party with hidden vote) is green
TONES = {'NO': 'q3', 'SÍ': 'q1', 'DE NADIE': 'qb', 'SIN DATOS': 'q2'}
BLOCKS = ('Izquierda', 'Derecha')
LINEAGES = ('PSOE', 'PP', 'VOX', 'UP')
KEYS = BLOCKS + LINEAGES
# Short public names of the blocks and lineages (the charts and the row of the election once published)
LABELS = {'Izquierda': 'Izquierda', 'Derecha': 'Derecha', 'PSOE': 'PSOE', 'PP': 'PP', 'VOX': 'Vox',
          'UP': 'IU / Podemos / Sumar'}
# Rows of the chart of question 3, bottom up
STRIP_ORDER = ('Derecha', 'Izquierda', 'UP', 'VOX', 'PP', 'PSOE')
# Rows of ``summary.parties`` of the party in government
INCUMBENTS = ('incumbent_general', 'incumbent_regional')
# Below this p value an effect is not chance (the verdicts of ``studies.shy.study.verdicts``)
SIGNIFICANCE = 0.05
# A mean at or beyond this many points from zero is «far from zero»
FAR = 1
# Defaults of ``meta.params`` (spec D4)
WINDOW_DAYS, RECENT_FROM, SIZE_THRESHOLD, BIG_THRESHOLD = 14, '2018-09-01', 2, -3
PP_EARLY = ('1982-01-01', '1996-01-01')
# The example of a falling party of the note of Podemos: the PP in April 2019
PP_EXAMPLE = ('es', '2019-04-28')
# The keys of a figures block (spec D3)
FIGURE_KEYS = ('n', 'under', 'over', 'mean', 'median', 'p_t', 'p_sign', 'size', 'size_over')
# Small counts in words (feminine: elecciones, veces, semanas)
WORDS = ('ninguna', 'una', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez')


def _is_num(x) -> bool:
    """Whether ``x`` is a real number (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def today() -> date:
    """Today in Europe/Madrid, the day the texts of the page count from (the tests replace it)."""
    return datetime.now(_madrid_tz()).date()


def params(meta: Optional[dict]) -> dict:
    """``meta.params`` with the defaults of the keys the texts read."""
    found = _get(meta, 'params') or {}

    return {'window_days': found.get('window_days') or WINDOW_DAYS,
            'recent_from': found.get('recent_from') or RECENT_FROM,
            'size_threshold': found.get('size_threshold') or SIZE_THRESHOLD,
            'big_threshold': found.get('big_threshold') or BIG_THRESHOLD,
            'pp_early': found.get('pp_early') or PP_EARLY}


def count_word(k) -> str:
    """A small count in words («seis»), larger ones in figures; a dash without a count."""
    if not _is_num(k):
        return DASH

    return WORDS[int(k)] if 0 <= k < len(WORDS) else fmt_int(k)


def times(k) -> str:
    """«una vez», «dos veces», «12 veces»."""
    return 'una vez' if k == 1 else count_word(k) + ' veces'


def figure_blocks(summary: Optional[dict]) -> dict:
    """
    The figures blocks the texts quote, each a dict with every key of ``FIGURE_KEYS`` (``None`` when the part lacks
    it), so the template reads them without guards.

    Returns
    -------
    dict
        ``left_all``, ``left_general``, ``left_regional``, ``right_regional``, ``lean_all``, ``lean_regional``,
        ``vox``, ``psoe_general``, ``pp_late``, ``pp_all``, ``podemos_before``, ``podemos_general``,
        ``podemos_regional``, ``incumbent_general``, ``incumbent_regional`` and ``size`` (``n_pairs``, ``mean``,
        ``share_over``).
    """
    paths = {'left_all': ('left', 'all'), 'left_general': ('left', 'general'), 'left_regional': ('left', 'regional'),
             'right_regional': ('right', 'regional'), 'lean_all': ('lean', 'all'),
             'lean_regional': ('lean', 'regional'), 'vox': ('vox',), 'psoe_general': ('psoe_general',),
             'pp_late': ('pp', 'late'), 'pp_all': ('pp', 'all'), 'podemos_before': ('podemos', 'before'),
             'podemos_general': ('podemos', 'general_recent'), 'podemos_regional': ('podemos', 'regional_recent')}
    blocks = {name: _get(summary, *path) for name, path in paths.items()}
    blocks |= {key: _party(summary, key) for key in INCUMBENTS}
    out = {name: {key: (found or {}).get(key) for key in FIGURE_KEYS} for name, found in blocks.items()}
    size = _get(summary, 'size') or {}
    out['size'] = {key: size.get(key) for key in ('n_pairs', 'mean', 'share_over')}

    return out


def pct(p) -> str:
    """
    A probability as a percentage for the prose.

    Returns
    -------
    str
        «26 %» from 10 % up, one decimal below («4,2 %»), two significant figures below 1 % («0,04 %»), «menos del
        0,01 %» below that; a dash without a number.
    """
    if not _is_num(p):
        return DASH

    value = p * 100

    if value >= 10:
        return fmt_num(value, 0) + ' %'

    if value >= 1:
        return fmt_num(value, 1) + ' %'

    if value >= 0.01:
        return '{:.2g}'.format(value).replace('.', ',') + ' %'

    return 'menos del 0,01 %'


def pct_with(p, article: str = 'del') -> str:
    """
    ``pct`` after its article: «del 26 %» (or «de menos del 0,01 %») with ``'del'``, «el 23 %» (or «menos del 0,01
    %») with ``'el'``.
    """
    text = pct(p)

    if text.startswith('menos'):
        return ('de ' if article == 'del' else '') + text

    return article + ' ' + text


def coin(n) -> str:
    """What a fair coin gives in ``n`` throws: «19 ó 20» for an odd ``n``, «20» for an even one."""
    if not _is_num(n):
        return DASH

    half = int(n) // 2

    return fmt_int(half) if n % 2 == 0 else '{} ó {}'.format(fmt_int(half), fmt_int(half + 1))


def p_text(p) -> str:
    """A p value with two decimals, «< 0,01» below that."""
    return '< 0,01' if _is_num(p) and p < 0.005 else fmt_num(p, 2)


def _when(label: Optional[str]) -> str:
    """The date part of a general election label in prose: «1977», «abril de 2019» (from «Generales abr. 2019»)."""
    text = (label or '').removeprefix('Generales ').strip()
    match = re.fullmatch(r'(\w+)\.? (\d{4})', text)

    if match and match.group(1) in MONTHS:
        return '{} de {}'.format(MONTHS_LONG[MONTHS.index(match.group(1))], match.group(2))

    return text


def verdict_rows(summary: Optional[dict]) -> list:
    """
    The three questions of the study with their verdict, as ``summary.verdicts`` gives them (spec D6).

    Returns
    -------
    list of dict
        ``{'label', 'tone', 'question'}`` per question: the verdict, its colour class (``TONES``; any «DE …» other
        than «DE NADIE» is ``q1``) and the question; ``'–'`` and ``q2`` without the verdict.
    """
    verdicts = _get(summary, 'verdicts') or {}
    rows = []

    for key, question in zip(('p1', 'p2', 'p3'), QUESTIONS):
        label = verdicts.get(key)

        if not label:
            rows.append({'label': DASH, 'tone': 'q2', 'question': question})
            continue

        tone = TONES.get(label) or ('q1' if label.startswith('DE ') else 'q2')
        rows.append({'label': label, 'tone': tone, 'question': question})

    return rows


def verdict_names(summary: Optional[dict]) -> list:
    """
    The parties named by the third verdict (``summary.verdicts.p3``, «DE PSOE Y DE VOX»), with the casing of the
    labels of ``summary.parties`` («PSOE», «Vox»); ``[]`` with «DE NADIE», «SIN DATOS» or without the verdict.
    """
    label = _get(summary, 'verdicts', 'p3') or ''

    if not label.startswith('DE ') or label == 'DE NADIE':
        return []

    cased = {text.upper(): text for text in LABELS.values()}
    cased |= {row['label'].upper(): row['label'] for row in _get(summary, 'parties') or [] if row.get('label')}

    return [cased.get(name, name) for name in (part.removeprefix('DE ').strip()
                                                for part in re.split(r' Y (?=DE )', label))]


def _under(figures: Optional[dict]) -> bool:
    """Whether a figures block is underestimated beyond chance (mean below 0, ``p_t`` below ``SIGNIFICANCE``)."""
    return bool(figures) and _is_num(figures.get('mean')) and figures['mean'] < 0 and \
        _is_num(figures.get('p_t')) and figures['p_t'] < SIGNIFICANCE


def _more(figures: Optional[dict], first: str, second: str) -> bool:
    """Whether the count ``first`` of a figures block is larger than ``second`` (``False`` without them)."""
    a, b = _get(figures, first), _get(figures, second)

    return _is_num(a) and _is_num(b) and a > b


def _chance(figures: Optional[dict]) -> bool:
    """Whether the split of the signs of a figures block is what chance gives (``p_sign`` at least 0,05)."""
    p = _get(figures, 'p_sign')

    return _is_num(p) and p >= SIGNIFICANCE


def _tail(under, over) -> str:
    """
    The count the chance of a sign split speaks of: «9 o más» (the underestimates) or, when the overestimates are
    more, «30 o más de más», so that the reader does not attach the chance to the count shown before it.
    """
    under, over = under or 0, over or 0

    return '{} o más de más'.format(fmt_int(over)) if over > under else '{} o más'.format(fmt_int(under))


def _party(summary: Optional[dict], key: str) -> Optional[dict]:
    """The figures of the row ``key`` of ``summary.parties``, or ``None``."""
    row = next((row for row in _get(summary, 'parties') or [] if row.get('key') == key), None)

    return _get(row, 'figures')


def flags(summary: Optional[dict], meta: Optional[dict]) -> dict:
    """
    The conditions and the dates of the texts of the page (spec D6).

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    meta : dict or None
        The ``meta`` part.

    Returns
    -------
    dict
        ``has_run`` (``summary.current`` is set), ``held`` (the date of ``meta.election`` is today or past, in
        Madrid), ``election_day`` («29 de noviembre»), ``election_short`` («29-N»), ``election_year``, ``days_left``
        (from ``today()``); ``p1_yes``, ``p2_yes`` (verdicts «SÍ»), ``p3_someone`` (a party with hidden vote);
        ``over_more`` (the left overestimated more often than underestimated), ``under_more`` (the reverse),
        ``vox_under_more`` (Vox underestimated more often than overestimated),
        ``general_under`` (the same in the generals, underestimated), ``psoe_streak`` (the PSOE underestimated more
        often in the generals, with elections beyond the big threshold), ``regional_both_over`` (the left
        overestimated more often in the regionals and both blocks with a positive regional mean), ``lean_chance``
        (the sign of left − right split as chance does), ``lean_regional_chance`` (the same in the regionals),
        ``vox_chance``, ``pp_early_under`` (every early PP error
        negative), ``pp_late_against`` (the PP overestimated more often since the end of the early period),
        ``podemos_all`` (overestimated in every recent regional), ``podemos_over``, ``podemos_chance`` (the split
        of the recent regionals is what chance gives), ``podemos_general_under``,
        ``podemos_before_chance``, ``incumbent_vanishes`` (the regional incumbent split as chance does),
        ``incumbent_general_under``, ``incumbent_hidden`` (an incumbent row underestimated beyond chance),
        ``generals_fail`` (the PSOE in the generals and the party in government do not pass the test: neither
        underestimated beyond chance, the incumbent underestimated more often in the generals and not in the
        regionals, the PSOE figures present),
        ``far_from_zero`` (rows of ``parties`` with ``|mean| >= FAR``), ``coin`` and the parameters of the texts
        (``window_weeks``, ``window_days``, ``recent_year``, ``pp_late_year``, ``big_points``, ``size_threshold``).
    """
    election = _get(meta, 'election') or {}
    day = _date(election.get('date'))
    verdicts = _get(summary, 'verdicts') or {}
    p3 = verdicts.get('p3') or ''
    left = _get(summary, 'left') or {}
    vox = _get(summary, 'vox')
    pp = _get(summary, 'pp') or {}
    podemos = _get(summary, 'podemos') or {}
    psoe = _get(summary, 'psoe_general')
    recent = _get(podemos, 'regional_recent')
    regional_means = (_get(left, 'regional', 'mean'), _get(summary, 'right', 'regional', 'mean'))
    early = [row.get('error') for row in _get(pp, 'early', 'errors') or []]
    p = params(meta)
    means = [_get(row, 'figures', 'mean') for row in _get(summary, 'parties') or []]

    return {
        'has_run': _get(summary, 'current') is not None,
        'held': bool(day) and day <= today(),
        'election_day': day_month(election.get('date')),
        'election_short': short_day(election.get('date')),
        'election_year': str(day.year) if day else None,
        'days_left': days_to(election.get('date'), today()),
        'p1_yes': verdicts.get('p1') == 'SÍ',
        'p2_yes': verdicts.get('p2') == 'SÍ',
        'p3_someone': p3.startswith('DE ') and p3 != 'DE NADIE',
        'over_more': _more(left.get('all'), 'over', 'under'),
        'under_more': _more(left.get('all'), 'under', 'over'),
        'general_under': _more(left.get('general'), 'under', 'over'),
        'psoe_streak': _more(psoe, 'under', 'over') and bool(_get(psoe, 'big')),
        'regional_both_over': _more(left.get('regional'), 'over', 'under')
        and all(_is_num(mean) and mean > 0 for mean in regional_means),
        'lean_chance': _chance(_get(summary, 'lean', 'all')),
        'lean_regional_chance': _chance(_get(summary, 'lean', 'regional')),
        'vox_chance': _chance(vox),
        'vox_under_more': _more(vox, 'under', 'over'),
        'pp_early_under': bool(early) and all(_is_num(error) and error < 0 for error in early),
        'pp_late_against': _more(pp.get('late'), 'over', 'under'),
        'podemos_all': _is_num(_get(recent, 'n')) and _get(recent, 'over') == _get(recent, 'n'),
        'podemos_over': _more(recent, 'over', 'under'),
        'podemos_chance': _chance(recent),
        'podemos_general_under': _more(podemos.get('general_recent'), 'under', 'over'),
        'podemos_before_chance': _chance(podemos.get('before')),
        'incumbent_vanishes': _chance(_party(summary, 'incumbent_regional')),
        'incumbent_general_under': _more(_party(summary, 'incumbent_general'), 'under', 'over'),
        'incumbent_hidden': any(_under(_party(summary, key)) for key in INCUMBENTS),
        'generals_fail': _chance(_party(summary, 'incumbent_regional'))
        and _more(_party(summary, 'incumbent_general'), 'under', 'over') and _is_num(_get(psoe, 'n'))
        and not _under(psoe) and not _under(_party(summary, 'incumbent_general')),
        'far_from_zero': sum(1 for mean in means if _is_num(mean) and abs(mean) >= FAR),
        'coin': coin(_get(vox, 'n')),
        'window_weeks': count_word(p['window_days'] // 7),
        'window_days': p['window_days'],
        'recent_year': p['recent_from'][:4],
        'pp_late_year': str(p['pp_early'][1])[:4],
        'big_points': fmt_num(abs(p['big_threshold']), 0),
        'size_threshold': fmt_num(p['size_threshold'], 0),
    }


def kpis(summary: Optional[dict], meta: Optional[dict] = None) -> list:
    """
    The three rows of the ``m.kpis`` macro.

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` (the last two, of the macro, always off): the
        elections in which the polls underestimated the left, those in which they underestimated Vox and the size of
        the error with a big party.
    """
    left = _get(summary, 'left', 'all') or {}
    vox = _get(summary, 'vox')
    size = _get(summary, 'size') or {}
    f = flags(summary, meta)
    support = 'y la sobreestimaron en {} · sesgo medio {} puntos'.format(fmt_int(left.get('over')),
                                                                        delta_text(left.get('mean'), 1))

    if f['over_more']:
        support += ': cuando se equivocan, más veces le dan de más que de menos'
    elif f['under_more']:
        support += ': cuando se equivocan, más veces de menos que de más'

    if vox:
        vox_row = {'label': 'Elecciones en que subestimaron a Vox',
                   'value': '{} de {}'.format(fmt_int(vox.get('under')), fmt_int(vox.get('n'))), 'unit': '',
                   'support': 'sesgo medio {} puntos · con una moneda saldrían {}: la probabilidad de {} por '
                              'azar es {}'.format(delta_text(vox.get('mean'), 1), f['coin'],
                                                  _tail(vox.get('under'), vox.get('over')),
                                                  pct_with(vox.get('p_sign')))}
    else:
        vox_row = {'label': 'Elecciones en que subestimaron a Vox', 'value': DASH, 'unit': '', 'support': None}

    rows = [
        {'label': 'Elecciones en que los sondeos subestimaron a la izquierda',
         'value': '{} de {}'.format(fmt_int(left.get('under')), fmt_int(left.get('n'))), 'unit': '',
         'support': support},
        vox_row,
        {'label': 'Error de los sondeos finales con un partido grande', 'value': fmt_num(size.get('mean'), 1),
         'unit': ' puntos',
         'support': 'en cualquier sentido · PSOE, PP, Vox e IU/Podemos/Sumar en {} pares elección-partido · en el {} '
                    '% de los casos, más de {} puntos'.format(
                        fmt_int(size.get('n_pairs')),
                        fmt_num(size['share_over'] * 100, 0) if _is_num(size.get('share_over')) else DASH,
                        f['size_threshold'])},
    ]

    return [dict(row, ok=False, color=None) for row in rows]


def _figures_row(label: str, figures: Optional[dict], bold: bool = False) -> dict:
    """A row of the table of question 1: the counts, the mean and the p value of a figures block."""
    figures = figures or {}

    return {'label': label, 'n': figures.get('n'), 'under': figures.get('under'), 'over': figures.get('over'),
            'mean': figures.get('mean'), 'p_t': figures.get('p_t'), 'p': p_text(figures.get('p_t')), 'bold': bold}


def last_general_year(elections: list) -> Optional[int]:
    """The year of the newest general election of the ``elections`` rows, or ``None``."""
    years = [int(row['event_date'][:4]) for row in elections if row.get('general') and row.get('event_date')]

    return max(years) if years else None


def left_rows(summary: Optional[dict], elections: list = (), meta: Optional[dict] = None) -> list:
    """
    The five rows of the table of question 1.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    elections : list of dict, optional
        The rows of the ``elections`` part, for the year of the last general election (``summary.last_year``
        without them).
    meta : dict or None, optional
        The ``meta`` part, for the start of the recent period.

    Returns
    -------
    list of dict
        ``{'label', 'n', 'under', 'over', 'mean', 'p_t', 'p', 'bold'}`` for the generals, the regionals, the recent
        elections, all of them (bold) and the left minus the right over all of them.
    """
    left = _get(summary, 'left') or {}
    last = _get(summary, 'last_year')
    general_last = last_general_year(list(elections)) or last
    since = _date(params(meta)['recent_from'])
    recent = 'Desde {} {}'.format(MONTHS[since.month - 1] + '.', since.year) if since else 'Recientes'

    return [
        _figures_row('Generales {}-{}'.format(_get(summary, 'first_year') or DASH, general_last or DASH),
                     left.get('general')),
        _figures_row('Autonómicas {}-{}'.format(_get(summary, 'first_year_regional') or DASH, last or DASH),
                     left.get('regional')),
        _figures_row(recent, left.get('recent')),
        _figures_row('Todas', left.get('all'), bold=True),
        _figures_row('Izquierda − Derecha, todas', _get(summary, 'lean', 'all')),
    ]


def _extremes_text(rows: list) -> str:
    """«Andalucía 2018 (−6,0) y Castilla y León 2022 (−3,9)» of the first two extremes of a side."""
    return names(['{} ({})'.format(row.get('label'), delta_text(row.get('error'), 1)) for row in rows[:2]])


def vox_tests(summary: Optional[dict]) -> list:
    """
    The three boxes of question 2.

    Returns
    -------
    list of dict
        ``{'key', 'label', 'value', 'unit', 'text'}`` for ``under`` (the count and the coin), ``mean`` (the mean and
        the median) and ``extremes`` (the largest error of each side); ``[]`` without figures of Vox.
    """
    vox = _get(summary, 'vox')

    if not vox:
        return []

    f = flags(summary, None)
    under, over = vox.get('under') or 0, vox.get('over') or 0
    extremes = vox.get('extremes') or {}
    low, high = extremes.get('under') or [], extremes.get('over') or []
    median = 'Mediana {}'.format(delta_text(vox.get('median'), 1))
    sides = []

    if low:
        sides.append('Sacó más de lo que decían en ' + _extremes_text(low))
    if high:
        sides.append(('menos' if low else 'Sacó menos de lo que decían') + ' en ' + _extremes_text(high))

    return [
        {'key': 'under', 'label': '1 · Elecciones en que Vox sacó más de lo que decían',
         'value': '{} de {}'.format(fmt_int(vox.get('under')), fmt_int(vox.get('n'))), 'unit': None,
         'text': 'Y menos en {}. Si los sondeos no tuvieran sesgo saldrían {}: la probabilidad de {} por azar '
                 'es {}.'.format(fmt_int(over), f['coin'], _tail(under, over), pct_with(vox.get('p_sign')))},
        {'key': 'mean', 'label': '2 · Sesgo medio', 'value': delta_text(vox.get('mean'), 1), 'unit': 'puntos',
         'text': median + (': un error que no se distingue de cero.' if f['vox_chance'] and not f['p2_yes']
                           else '.')},
        {'key': 'extremes',
         'label': '3 · Las sorpresas, en los dos sentidos' if low and high else '3 · Las mayores sorpresas',
         'value': '{} y {}'.format(delta_text(low[0].get('error'), 1) if low else DASH,
                                   delta_text(high[0].get('error'), 1) if high else DASH),
         'unit': None, 'text': '; '.join(sides) + '.' if sides else None},
    ]


def party_rows(summary: Optional[dict]) -> list:
    """
    The rows of the table of question 3, in the order of ``summary.parties``.

    Returns
    -------
    list of dict
        ``{'key', 'label', 'n', 'under', 'over', 'mean', 'p_t', 'p', 'size'}``; a row without figures has every
        figure ``None`` (the template writes dashes).
    """
    rows = []

    for row in _get(summary, 'parties') or []:
        figures = row.get('figures') or {}
        rows.append({'key': row.get('key'), 'label': row.get('label'), 'n': figures.get('n'),
                     'under': figures.get('under'), 'over': figures.get('over'), 'mean': figures.get('mean'),
                     'p_t': figures.get('p_t'), 'p': p_text(figures.get('p_t')) if figures else DASH,
                     'size': figures.get('size')})

    return rows


def big_years(summary: Optional[dict]) -> str:
    """The general elections in which the PSOE beat its polls by more than the big threshold, in prose: «1977, 1986,
    1993, 1996, 2004 y 2023» (``''`` without them)."""
    return names([_when(label) for label in _get(summary, 'psoe_general', 'big') or []])


def texts(summary: Optional[dict], elections: list, meta: Optional[dict]) -> dict:
    """
    The prose fragments of the conditioned texts that need more than a filter.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    elections : list of dict
        The rows of the ``elections`` part.
    meta : dict or None
        The ``meta`` part.

    Returns
    -------
    dict
        ``big_years`` (``big_years``), ``big_elections`` («seis elecciones»), ``general_odds`` («el 23 %», the
        chance of the split of the left in the generals), ``last_general`` (year), ``vox_since`` (the year of the
        first election with Vox), ``vox_general`` (``{'n', 'k', 'years'}``: the general elections with Vox, in
        figures and in words, and their years in prose) and ``vox_regional`` (the regional ones), ``vox_up`` and
        ``vox_down`` («Andalucía 2018 (−6,0)»: the largest error of each side), ``pp_early`` (``{'span', 'errors',
        'times'}``: «1982 y 1993», the signed errors in prose and «cuatro veces seguidas»), ``pp_examples``
        («+4,0 en abril de 2019»: the general elections since the early period in which the PP got less than its polls
        by more than the big threshold), ``pp_april`` (the error of the PP in April 2019, ``None`` without that
        row), ``hidden`` (the parties named by the third verdict, ``verdict_names``, in prose; ``hidden_n`` their
        number), ``podemos_general`` (the recent generals in words), ``podemos_contrary`` (the times they were
        underestimated, «dos veces») and ``n_communities`` (``n_scopes`` − 1).
    """
    p = params(meta)
    vox_rows = sorted((row for row in elections if _is_num(row.get('VOX'))), key=lambda row: row['event_date'])
    vox_general = [row for row in vox_rows if row.get('general')]
    general_years = list(dict.fromkeys(row['event_date'][:4] for row in vox_general))
    n_vox = _get(summary, 'vox', 'n')
    early = _get(summary, 'pp', 'early') or {}
    early_errors = early.get('errors') or []
    early_years = [_when(row.get('label')) for row in early_errors]
    late_from = str(p['pp_early'][1])
    examples = ['{} en {}'.format(delta_text(row['PP'], 1), _when(row.get('label'))) for row in elections
                if row.get('general') and (row.get('event_date') or '') >= late_from and _is_num(row.get('PP'))
                and row['PP'] > -p['big_threshold']]
    april = next((row.get('PP') for row in elections
                  if (row.get('scope'), row.get('event_date')) == PP_EXAMPLE and _is_num(row.get('PP'))), None)
    hidden = verdict_names(summary)
    n_scopes = _get(summary, 'n_scopes')
    n_big = len(_get(summary, 'psoe_general', 'big') or [])
    extremes = _get(summary, 'vox', 'extremes') or {}
    up, down = ((extremes.get(side) or [None])[0] for side in ('under', 'over'))

    return {
        'big_years': big_years(summary),
        'big_elections': 'una elección' if n_big == 1 else count_word(n_big) + ' elecciones',
        'general_odds': pct_with(_get(summary, 'left', 'general', 'p_sign'), 'el'),
        'last_general': last_general_year(elections) or _get(summary, 'last_year'),
        'vox_since': vox_rows[0]['event_date'][:4] if vox_rows else None,
        'vox_general': {'n': len(vox_general), 'k': count_word(len(vox_general)), 'years': names(general_years)},
        'vox_up': _extremes_text([up]) if up else None,
        'vox_down': _extremes_text([down]) if down else None,
        'vox_regional': n_vox - len(vox_general) if _is_num(n_vox) else None,
        'pp_early': {'span': '{} y {}'.format(early_years[0], early_years[-1]) if early_years else None,
                     'errors': ', '.join(delta_text(row.get('error'), 1) for row in early_errors),
                     'times': '{} seguidas'.format(times(len(early_errors))) if len(early_errors) > 1 else
                     times(len(early_errors))},
        'pp_examples': ', '.join(examples),
        'pp_april': april,
        'hidden': names(hidden),
        'hidden_n': len(hidden),
        'podemos_general': count_word(_get(summary, 'podemos', 'general_recent', 'n')),
        'podemos_contrary': times(_get(summary, 'podemos', 'general_recent', 'under')),
        'n_communities': n_scopes - 1 if _is_num(n_scopes) else None,
    }


def _period(current: dict) -> Optional[str]:
    """«del 22 de octubre al 5 de noviembre (12)», the window of the polls of the run; ``None`` without dates."""
    first, last = day_month(current.get('from')), day_month(current.get('to'))

    if not first or not last:
        return None

    return 'del {} al {} ({})'.format(first, last, fmt_int(current.get('n_polls')))


def after_rows(summary: Optional[dict], elections: list, meta: Optional[dict]) -> dict:
    """
    The card «Después del 29-N»: what the final polls say today or, once published, the row of the election.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    elections : list of dict
        The rows of the ``elections`` part.
    meta : dict or None
        The ``meta`` part.

    Returns
    -------
    dict
        ``columns`` (``[{'key', 'label'}]``), ``label``, ``date``, ``day`` («29-N»: of ``meta.election`` or of the
        row), ``now``, ``polls``, ``results``, ``error``, ``period`` and ``belief``. With ``summary.current``: the
        blocks present in its ``values`` and then the parties of its ``names`` present in them (labelled with the
        name of the run), ``now`` the values, ``period`` (``_period``) and ``belief`` («la izquierda sacaría más de
        35,4 y Vox más de 19,5», with the values present of the two groups of the beliefs, ``None`` without
        them). Without it and without ``meta.election``: the newest row of ``elections`` with scope ``es``,
        only when its year is ``summary.last_year`` (the election just published), with its six keys with an error
        (``LABELS``), its ``label``, ``date`` and dicts. Otherwise every value ``None`` and no columns.
    """
    out = {'columns': [], 'label': None, 'date': None, 'day': short_day(_get(meta, 'election', 'date')), 'now': None,
           'polls': None, 'results': None, 'error': None, 'period': None, 'belief': None}
    current = _get(summary, 'current')

    if current is not None:
        values = current.get('values') or {}
        keys = [key for key in BLOCKS if key in values]
        keys += [name for name in current.get('names') or [] if name in values and name not in keys]
        belief = []

        if _is_num(values.get('Izquierda')):
            belief.append('la izquierda sacaría más de ' + fmt_num(values['Izquierda'], 1))
        if _is_num(values.get('VOX')):
            belief.append('Vox {}más de {}'.format('' if belief else 'sacaría ', fmt_num(values['VOX'], 1)))

        return out | {'columns': [{'key': key, 'label': key} for key in keys], 'now': values,
                      'period': _period(current), 'belief': ' y '.join(belief) or None}

    if _get(meta, 'election'):
        return out

    rows = [row for row in elections if row.get('scope') == 'es' and row.get('event_date')]
    newest = max(rows, key=lambda row: row['event_date']) if rows else None

    if newest is None or str(newest['event_date'][:4]) != str(_get(summary, 'last_year')):
        return out

    keys = [key for key in KEYS if _is_num(newest.get(key))]

    return out | {'columns': [{'key': key, 'label': LABELS[key]} for key in keys], 'label': newest.get('label'),
                  'date': newest['event_date'], 'day': short_day(newest['event_date']),
                  'polls': newest.get('polls') or {},
                  'results': newest.get('results') or {}, 'error': {key: newest[key] for key in keys}}


def _bars(rows: list, key: str) -> list:
    """The bars of the error of ``key``: ``[{short, label, value, general}]`` of the rows with it, by date."""
    rows = sorted((row for row in rows if _is_num(row.get(key))), key=lambda row: row.get('event_date') or '')

    return [{'short': row.get('short'), 'label': row.get('label'), 'value': row[key],
             'general': bool(row.get('general'))} for row in rows]


def island(parts: dict) -> dict:
    """
    What the three charts of the page draw (the page script reads it).

    Parameters
    ----------
    parts : dict
        The published parts by name (``{}`` for a missing one).

    Returns
    -------
    dict
        ``left_bars`` and ``left_mean`` (the error of the left by election and its mean), ``vox_bars`` and
        ``vox_mean`` (the same for Vox), and ``party_strip`` (``{'rows': [{party, label, value}], 'means': {party:
        mean}, 'order': [...]}``: every error of the six groups, their means and the rows of the chart bottom up).
    """
    summary = parts.get('summary') or {}
    rows = _get(parts.get('elections'), 'rows') or []
    means = {row.get('key'): _get(row, 'figures', 'mean') for row in summary.get('parties') or []}

    return {
        'left_bars': _bars(rows, 'Izquierda'),
        'left_mean': _get(summary, 'left', 'all', 'mean'),
        'vox_bars': _bars(rows, 'VOX'),
        'vox_mean': _get(summary, 'vox', 'mean'),
        'party_strip': {
            'rows': [{'party': LABELS[key], 'label': row.get('label'), 'value': row[key]}
                     for row in rows for key in STRIP_ORDER if _is_num(row.get(key))],
            'means': {LABELS[key]: means.get(key) for key in STRIP_ORDER},
            'order': [LABELS[key] for key in STRIP_ORDER],
        },
    }
