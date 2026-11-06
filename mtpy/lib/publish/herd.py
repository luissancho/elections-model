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
    {'label': 'Histórico de errores de las encuestas por elección', 'url': '/elecciones'},
    {'label': 'Ranking Encuestadoras y rebaño por casa', 'url': '/casas'},
    {'label': 'Método: el rebaño, aparte', 'url': '/metodo/rating#herding'},
    {'label': 'El sesgo del CIS de Tezanos', 'url': '/analisis/sesgo-cis'},
    {'label': 'Análisis', 'url': '/analisis'},
]

# The final window counts as «one urn» up to this mean ratio of the past cycles
URN_RATIO = 1.1

CYCLE_WINDOWS = ('before', 'last', 'final')

WEEK_COLUMNS = ['event_date', 'label', 'current', 'lo', 'hi', 'k', 'ratio']


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


def _flat_cycle(row: dict) -> dict:
    """A cycle row with its nested measures flattened to scalar columns (``names`` joined by ``/``)."""
    flat = {key: value for key, value in row.items() if key not in (*CYCLE_WINDOWS, 'cis', 'names')}
    flat['names'] = '/'.join(row['names'])

    for window in CYCLE_WINDOWS:
        flat[f'{window}_k'] = row[window]['k']
        flat[f'{window}_ratio'] = row[window]['ratio']

    for key in ('last', 'cycle'):
        flat[f'cis_{key}_k'] = row['cis'][key]['k']
        flat[f'cis_{key}_ratio'] = row['cis'][key]['ratio']

    for key in ('k', 'h', 'p'):
        flat[f'cis_own_{key}'] = row['cis']['own'][key]

    return flat


def export_herd_cycles(study) -> tuple:
    """Export one row per election cycle, past ascending then the one in progress.

    Returns
    -------
    tuple
        ``(data, {'cycles': frame})``: ``data`` for ``rebano-cycles`` with ``rows``; the frame is the flat form.
    """
    rows = _clean(study.cycles)
    return {'rows': rows}, {'cycles': pd.DataFrame([_flat_cycle(row) for row in rows])}


def export_herd_weeks(study) -> tuple:
    """Export the dispersion by weeks to the election.

    Returns
    -------
    tuple
        ``(data, {'weeks': frame})``: ``data`` for ``rebano-weeks`` with ``buckets``, ``rows`` and ``mean``; the frame
        is the long form, the rows plus the ``mean`` rows (``event_date`` null, label ``Media``).
    """
    weeks = _clean(study.weeks)
    mean = [{'event_date': None, 'label': 'Media', 'current': False, **row} for row in weeks['mean']]
    return weeks, {'weeks': pd.DataFrame([*weeks['rows'], *mean], columns=WEEK_COLUMNS)}


def export_herd_houses(study) -> tuple:
    """Export one row per cycle and house with enough polls in the final window.

    Returns
    -------
    tuple
        ``(data, {'houses': frame})``: ``data`` for ``rebano-houses`` with ``rows``.
    """
    rows = _clean(study.houses)
    return {'rows': rows}, {'houses': pd.DataFrame(rows)}


def _subtitle(template: str, study) -> str:
    """Render the subtitle template from the summary: the clauses that depend on the figures appear only if they
    hold."""
    summary = study.summary
    k, n = summary['k_narrowed'], summary['n_cycles']
    mean_final = summary['mean_final']
    final_clause = ''

    if mean_final is not None and mean_final <= URN_RATIO:
        final_clause = (': en las últimas tres semanas, tanto como si todas las casas sacaran sus muestras de la '
                        'misma urna')

    current = ''
    now, houses = summary['current'], summary['houses_now']

    if now and now['like_final'] and houses and houses['n'] > 0 and houses['k'] > 0:
        k_now, n_now = houses['k'], houses['n']

        if k_now == n_now == 1:
            who = 'la única casa con sondeos suficientes publica'
        elif k_now == n_now:
            who = f'las {n_now} casas con sondeos suficientes publican'
        else:
            who = f'{k_now} de las {n_now} casas con sondeos suficientes publican'

        current = (f" En {study.election['date'][:4]} ya van así desde el principio, y {who} cifras que oscilan "
                   f"menos de lo que cabría esperar sólo por el azar de sus muestras.")

    return template.format(n_words=str(n) if k == n else f'{k} de las {n}', since_year=study.params['since'][:4],
                           final_clause=final_clause, current=current)


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
    title, template = STUDY_TITLES[study_id]
    prov = prov or {'commit': None, 'dirty': False, 'versions': {}}
    data = {
        'study': study_id, 'title': title, 'subtitle': _subtitle(template, study), 'scope': study.params['scope'],
        'election': study.election, 'params': study.params, 'inputs': study.inputs,
        'sources': [dict(s) for s in HERD_SOURCES],
        'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'], 'seconds': seconds or {},
    }
    return data, {}
