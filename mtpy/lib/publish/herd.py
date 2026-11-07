"""Exporters of the herd study (``rebano``) and its computation for ``publish_study``.

The study package (``mtpy.lib.studies.herd``) is imported inside ``compute_for_publish``, never at module level: the
web imports ``publish`` and must stay light.
"""
from typing import Optional

import pandas as pd

from .. import bundle
from .studies import _rounded, STUDY_TITLES
from ...core.io import FileSystem

HERD_SOURCES = [
    {'label': 'Histórico de elecciones', 'url': '/elecciones'},
    {'label': 'Análisis', 'url': '/analisis'},
]

# A comparison is «significant» when its sign test is below this level and most elections go the same way
ALPHA = 0.05
# Below this probability the chance is written «menos de 1 entre 1.000», as on the page
RARE = 0.001

ELECTION_COLUMNS = ['scope', 'event_date', 'label', 'kind', 'comparison', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b',
                    'change', 'falls', 'p']

ROBUSTNESS_COLUMNS = ['pair_days', 'min_rows', 'default', 'comparison', 'n', 'k', 'median_change', 'change', 'p']

WEEK_COLUMNS = ['event_date', 'label', 'current', 'lo', 'hi', 'pairs', 'comparisons', 'ratio']


def compute_for_publish(data: FileSystem, writer: bundle.BundleWriter):
    """Compute the study from the published parts of the live bundle (even if the writer is a dry run).

    Parameters
    ----------
    data : FileSystem
        The data folder (the parameters of the study).
    writer : BundleWriter
        The writer of the publication; only its file system is used.

    Returns
    -------
    HerdStudy
        The computed study.
    """
    from ..studies.herd.study import compute_herd

    return compute_herd(bundle.BundleReader(writer.fs, prefix=bundle.PREFIX), data)


def _clean(obj):
    """Plain numbers (``None`` for NaN) without rounding: the study rounds its figures, and a rounding to 2 decimals
    would publish a ``p`` of ``5e-06`` as ``0.0``."""
    return _rounded(obj, None)


def export_herd_summary(study) -> tuple:
    """Export the headline figures of the study.

    Returns
    -------
    tuple
        ``(data, {'robustness': frame})``: ``data`` for ``rebano-summary``; the frame has the rows of the robustness
        grid, with ``n`` and ``k`` as integers (0 in the empty cells) and ``None`` as NaN.
    """
    data = _clean(study.summary)
    frame = pd.DataFrame(data['robustness']['rows'], columns=ROBUSTNESS_COLUMNS)
    frame[['n', 'k']] = frame[['n', 'k']].astype('int64')

    return data, {'robustness': frame}


def export_herd_elections(study) -> tuple:
    """Export one row per election and comparison (the same pairs of houses in both stretches).

    Returns
    -------
    tuple
        ``(data, {'elections': frame})``: ``data`` for ``rebano-elections`` with ``rows``; the frame has the same rows.
    """
    rows = _clean(study.elections)
    return {'rows': rows}, {'elections': pd.DataFrame(rows, columns=ELECTION_COLUMNS)}


def export_herd_current(study) -> tuple:
    """Export the election in progress (its comparisons so far), or ``None`` when there is none.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``rebano-current``.
    """
    return {'current': _clean(study.current)}, {}


def export_herd_weeks(study) -> tuple:
    """Export the dispersion between houses by bucket of weeks to the election, per election and on average.

    Returns
    -------
    tuple
        ``(data, {'weeks': frame})``: ``data`` for ``rebano-weeks`` (``buckets``, ``rows``, ``mean``); the frame is
        long: the rows plus one row per bucket of the mean (label «Media», no date, pairs or comparisons).
    """
    data = _clean(study.weeks)
    mean = [{'event_date': None, 'label': 'Media', 'current': False, 'lo': m['lo'], 'hi': m['hi'], 'pairs': None,
             'comparisons': None, 'ratio': m['ratio']} for m in data['mean']]
    frame = pd.DataFrame(data['rows'] + mean, columns=WEEK_COLUMNS)
    frame[['pairs', 'comparisons']] = frame[['pairs', 'comparisons']].astype('Int64')

    return data, {'weeks': frame}


def _significant(comparison: dict) -> bool:
    """Whether the sign test of a comparison is below ``ALPHA`` with most elections going down."""
    p = comparison.get('sign_p')

    return p is not None and p < ALPHA and comparison['k'] * 2 > comparison['n']


def _times(value: float) -> str:
    """A count of times with the thousands separator of the site (``1.234``)."""
    return f'{int(round(value)):,}'.replace(',', '.')


def _chance(p: float) -> str:
    """A sign-test probability as the page writes it: «1 entre 769», «menos de 1 entre 1.000»."""
    return 'menos de 1 entre 1.000' if p < RARE else f'1 entre {_times(1 / p)}'


def _subtitle(summary: dict) -> str:
    """Write the subtitle from the figures of the summary: what holds in the last three weeks, then, if there is
    data, what holds before them. The year is the first of the elections of ``far_final`` (falling back on the
    ``since_year`` of the summary)."""
    comparisons = summary['comparisons']
    final, far_mid, mid_final = comparisons['far_final'], comparisons['far_mid'], comparisons['mid_final']
    n, k, since = final['n'], final['k'], final.get('since_year') or summary.get('since_year')
    elections = f'elecciones con sondeos lejos y cerca de la votación, desde {since}'

    if n == 0:
        return ('Todavía no hay elecciones con sondeos suficientes para comparar las mismas encuestadoras lejos y '
                'cerca de la votación.')

    if k * 2 <= n:
        return (f'En sólo {k} de las {n} {elections}, las mismas encuestadoras se parecen más entre sí en las tres '
                f'últimas semanas que lejos de ella; en las demás, no.')

    head = (f'En {k} de las {n} {elections}, las mismas encuestadoras se parecen más entre sí en las tres últimas '
            f'semanas que lejos de ella')

    if not _significant(final):
        return head + ', una proporción que podría deberse al azar.'

    text = f"{head}: por azar, {_chance(final['sign_p'])}."

    if far_mid['n'] > 0:
        count = f"{far_mid['k']} de {far_mid['n']}"

        if _significant(far_mid):
            text += f' Y también hasta tres semanas antes: {count}.'
        elif _significant(mid_final):
            text += f' Hasta tres semanas antes, no: {count}, como una moneda.'

    return text


def export_herd_meta(study, study_id: str = 'rebano', registration: Optional[dict] = None,
                     checked_at: Optional[str] = None, prov: Optional[dict] = None,
                     seconds: Optional[dict] = None) -> tuple:
    """Export the description of the study: titles, election in progress, parameters, inputs, sources, provenance.

    ``registration`` and ``checked_at`` are accepted for the signature of the other studies and ignored: this
    study has nothing to register or check.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``rebano-meta``. No CSV twin.
    """
    title = STUDY_TITLES[study_id][0]
    prov = prov or {'commit': None, 'dirty': False, 'versions': {}}
    data = {
        'study': study_id, 'title': title, 'subtitle': _subtitle(study.summary),
        'scope': study.params['live_scope'],
        'election': study.election, 'params': study.params, 'inputs': study.inputs,
        'sources': [dict(s) for s in HERD_SOURCES],
        'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'], 'seconds': seconds or {},
    }
    return data, {}
