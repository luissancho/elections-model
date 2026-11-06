"""Exporters of the ban study (``andorra``) and its computation for ``publish_study``.

The study package (``mtpy.lib.studies.ban``) is imported inside ``compute_for_publish``, never at module level: the
web imports ``publish`` and must stay light.
"""
from typing import Optional

import pandas as pd

from .. import bundle
from .studies import _rounded, NothingToPublish, STUDY_TITLES, BAN_INTRO
from ...core.io import FileSystem

# Decimals of the vote shares; errors, differences and means keep the 2 of ``_rounded``
SHARE_DECIMALS = 1

# Significant figures of the p value of the summary
P_FIGURES = 3

URL_CONFIDENCIAL = ('https://www.elconfidencial.com/espana/cataluna/elecciones-catalanas/2017-12-18/'
                    'encuesta-sondeo-electoral-prohibido-periodic-andorra_1494464/')

URL_ONDA_CERO = ('https://www.ondacero.es/noticias/espana/'
                 'encuestas-electorales-andorra-ascenso-brocoli-descalabro-naranja_201911065dc2b2650cf2767d148c38e5.html')

URL_HERALDO = ('https://www.heraldo.es/noticias/aragon/2026/02/05/'
               'las-encuestas-de-las-elecciones-en-aragon-a-traves-de-ppollo-brocolis-y-frutos-rojos-1891137.html')

BAN_SOURCES = [
    {'label': '«La \'frutería de Andorra\' y las encuestas prohibidas: por qué es ilegal difundirlas», '
              'El Confidencial, 18-12-2017', 'url': URL_CONFIDENCIAL},
    {'label': '«Resultados del sondeo electoral de Andorra: notable ascenso del \'brócoli\' y descalabro de la '
              '\'naranja\'», Onda Cero, 6-11-2019', 'url': URL_ONDA_CERO},
    {'label': '«Cómo saltarse la prohibición de publicar encuestas con el \'ppollo\', el brócoli y los frutos '
              'rojos», Heraldo de Aragón, 5-2-2026', 'url': URL_HERALDO},
    {'label': 'Histórico de errores de las encuestas por elección', 'url': '/elecciones'},
    {'label': 'Método', 'url': '/metodo'},
    {'label': 'Análisis', 'url': '/analisis'},
]

MODEL_COLUMNS = ('error_legal', 'error_frozen', 'error_consensus', 'error_with', 'beaten', 'beaten_with')


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
    BanStudy
        The computed study.

    Raises
    ------
    NothingToPublish
        If no national election has clandestine polls (nothing is written).
    """
    from ..studies.ban.study import compute_ban

    study = compute_ban(bundle.BundleReader(writer.fs, prefix=bundle.PREFIX), data)

    if study.latest is None:
        # Before any part is written: a failure in the last part would leave new parts beside the old ``meta``
        raise NothingToPublish('andorra: no national election with clandestine polls')

    return study


def _significant(p, figures: int = P_FIGURES):
    """A probability with ``figures`` significant figures (``None`` for a missing one)."""
    return None if p is None else float(f'{p:.{figures}g}')


def _frame(rows: list, columns: Optional[list] = None) -> pd.DataFrame:
    """The CSV twin of a list of rows."""
    return pd.DataFrame(rows, columns=columns)


def export_ban_summary(study) -> tuple:
    """Export the headline figures: pairs, elections with a model, cells, big samples and the exception.

    Every figure is rounded to 2 decimals but the ``p``, which keeps three significant figures.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``andorra-summary``.
    """
    data = _rounded(study.summary)
    data['pairs']['p'] = _significant(_rounded(study.summary['pairs']['p'], None))
    return data, {}


def export_ban_pairs(study) -> tuple:
    """Export every pair of a clandestine poll with the last legal one of its house.

    Returns
    -------
    tuple
        ``(data, {'pairs': frame})``: ``data`` for ``andorra-pairs`` with ``rows``.
    """
    rows = _rounded(study.pairs)
    return {'rows': rows}, {'pairs': _frame(rows)}


def _flat_election(row: dict) -> dict:
    """An election row with its ``model`` flattened to ``model_*`` columns (null without a model)."""
    flat = {key: value for key, value in row.items() if key != 'model'}
    model = row['model'] or {}

    for key in MODEL_COLUMNS:
        flat[f'model_{key}'] = model.get(key)

    return flat


def export_ban_elections(study) -> tuple:
    """Export one row per election with clandestine polls and the period before the chart.

    Returns
    -------
    tuple
        ``(data, {'elections': frame})``: ``data`` for ``andorra-elections`` with ``rows`` and ``period``; the frame
        has the ``model`` flattened.
    """
    rows = _rounded(study.elections)
    return {'rows': rows, 'period': _rounded(study.period)}, {'elections': _frame([_flat_election(r) for r in rows])}


def export_ban_cells(study) -> tuple:
    """Export one row per election and party of the general elections with a model; shares to 1 decimal.

    Returns
    -------
    tuple
        ``(data, {'cells': frame})``: ``data`` for ``andorra-cells`` with ``rows``.
    """
    rows = _rounded(study.cells, SHARE_DECIMALS)
    return {'rows': rows}, {'cells': _frame(rows)}


def export_ban_houses(study) -> tuple:
    """Export one row per house with enough pairs and the pairs of the big samples.

    Returns
    -------
    tuple
        ``(data, {'houses': frame})``: ``data`` for ``andorra-houses`` with ``rows`` and ``big_rows``; the frame has
        the houses.
    """
    rows = _rounded(study.houses)
    return {'rows': rows, 'big_rows': _rounded(study.big)}, {'houses': _frame(rows)}


def export_ban_latest(study) -> tuple:
    """Export the clandestine polls of the last national election and the figures against its result.

    Shares to 1 decimal, the errors to 2.

    Returns
    -------
    tuple
        ``(data, {'latest-polls': frame})``: ``data`` for ``andorra-latest``; the frame has one row per poll with the
        share of each party as a column.

    Raises
    ------
    NothingToPublish
        If no national election has clandestine polls.
    """
    latest = study.latest

    if latest is None:
        raise NothingToPublish('andorra: no national election with clandestine polls')

    data = {**_rounded({k: v for k, v in latest.items() if k != 'error'}, SHARE_DECIMALS),
            'error': _rounded(latest['error'])}
    data['polls'] = [{**poll, 'error_avg': _rounded(poll['error_avg'])} for poll in data['polls']]
    rows = [{**{k: v for k, v in poll.items() if k != 'pct'}, **poll['pct']} for poll in data['polls']]
    return data, {'latest-polls': _frame(rows)}


def _subtitle(template: str, study) -> str:
    """Render the subtitle from the summary: with a model the elections and cells, without it the pairs."""
    elections = study.summary['elections']

    if elections is None:
        pairs = study.summary['pairs']
        return (f"{BAN_INTRO} En {pairs['n']} pares casa a casa desde {study.params['since'][:4]}, el sondeo "
                f"clandestino estuvo más cerca del resultado que el último legal en {pairs['k']}.")

    cells = study.summary['cells']
    return template.format(year=study.params['chart_since'][:4], n=elections['n'], k=elections['k_beaten'], kc=cells['k_closer'], nc=cells['n'])


def export_ban_meta(study, study_id: str = 'andorra', registration: Optional[dict] = None,
                    checked_at: Optional[str] = None, prov: Optional[dict] = None,
                    seconds: Optional[dict] = None) -> tuple:
    """Export the description of the study: titles, election in progress, parameters, inputs, sources, provenance.

    ``registration`` and ``checked_at`` are accepted for the signature of the other studies and ignored: this
    study has nothing to register or check.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``andorra-meta``. No CSV twin.
    """
    title, template = STUDY_TITLES[study_id]
    prov = prov or {'commit': None, 'dirty': False, 'versions': {}}
    data = {
        'study': study_id, 'title': title, 'subtitle': _subtitle(template, study),
        'params': _rounded(study.params, None), 'inputs': study.inputs, 'next': study.next,
        'sources': [dict(s) for s in BAN_SOURCES],
        'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'], 'seconds': seconds or {},
    }
    return data, {}
