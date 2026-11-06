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
    {'label': 'Método del promedio', 'url': '/metodo/promedio'},
    {'label': 'Análisis', 'url': '/analisis'},
]

# A comparison is «significant» when its sign test is below this level and most elections go the same way
ALPHA = 0.05

ELECTION_COLUMNS = ['scope', 'event_date', 'label', 'kind', 'comparison', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b',
                    'change', 'falls', 'p']


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
        ``(data, {})``: ``data`` for ``rebano-summary``.
    """
    return _clean(study.summary), {}


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


def _significant(comparison: dict) -> bool:
    """Whether the sign test of a comparison is below ``ALPHA`` with most elections going down."""
    p = comparison.get('sign_p')

    return p is not None and p < ALPHA and comparison['k'] * 2 > comparison['n']


def _times(value: float) -> str:
    """A count of times with the thousands separator of the site (``1.234``)."""
    return f'{int(round(value)):,}'.replace(',', '.')


def _subtitle(summary: dict) -> str:
    """Write the subtitle from the figures of the summary: what holds in the last three weeks, then, if there is
    data, what holds before them."""
    comparisons = summary['comparisons']
    final, far_mid = comparisons['far_final'], comparisons['far_mid']
    head = (f"En {final['k']} de las {final['n']} elecciones con datos desde {summary['since_year']}, las mismas "
            f"encuestadoras se parecen más entre sí en las tres últimas semanas que lejos de la votación")

    if not _significant(final):
        return head + ', una proporción que podría deberse al azar.'

    text = f"{head}: por azar pasaría una vez de cada {_times(1 / final['sign_p'])}."

    if far_mid['n'] > 0:
        count = f"{far_mid['k']} de {far_mid['n']}"

        if _significant(far_mid):
            text += f' Y también hasta tres semanas antes: {count}.'
        else:
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
