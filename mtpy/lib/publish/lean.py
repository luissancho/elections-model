"""Exporters of the CIS lean study (``sesgo-cis``) and its computation for ``publish_study``.

The study package (``mtpy.lib.studies.lean``, with its statistics) is imported inside ``compute_for_publish``, never at
module level: the web imports ``publish`` and must stay light.
"""
from typing import Optional

import pandas as pd

from .. import bundle
from .studies import _rounded, STUDY_TITLES
from ...core.io import FileSystem

URL_INDEPENDIENTE_2018_07 = ('https://www.elindependiente.com/politica/2018/07/09/'
                             'jose-felix-tezanos-presidente-del-cis-abandona-cargo-la-ejecutiva-federal-del-psoe/')

URL_INDEPENDIENTE_2018_09 = ('https://www.elindependiente.com/politica/2018/09/25/'
                             'los-datos-sorprendentes-cocina-cis-tezanos/')

URL_UNED = ('http://portal.uned.es/portal/page?_pageid=93%2C63197807'
            '&_dad=portal&_schema=PORTAL')

LEAN_SOURCES = [
    {'label': 'Histórico de errores de las encuestas por elección', 'url': '/elecciones'},
    {'label': 'Perfil del CIS', 'url': '/casas/cis'},
    {'label': 'Método: el ejemplo del CIS desde 2018', 'url': '/metodo/promedio#ejemplo-cis'},
    {'label': 'Análisis', 'url': '/analisis'},
    {'label': '«José Félix Tezanos, presidente del CIS, abandona su cargo en la Ejecutiva Federal del PSOE», '
              'El Independiente, 9-7-2018', 'url': URL_INDEPENDIENTE_2018_07},
    {'label': '«Los datos sorprendentes de la cocina del CIS de Tezanos», El Independiente, 25-9-2018',
     'url': URL_INDEPENDIENTE_2018_09},
    {'label': 'Perfil de José Félix Tezanos en la UNED', 'url': URL_UNED},
]


def compute_for_publish(data: FileSystem, writer: bundle.BundleWriter):
    """Compute the study from the published parts of the live bundle (even if the writer is a dry run).

    Parameters
    ----------
    data : FileSystem
        The data folder (the parameters and the catalogues of the study).
    writer : BundleWriter
        The writer of the publication; only its file system is used.

    Returns
    -------
    LeanStudy
        The computed study.
    """
    from ..studies.lean.study import compute_lean

    return compute_lean(bundle.BundleReader(writer.fs, prefix=bundle.PREFIX), data)


def export_lean_summary(study) -> tuple:
    """Export the headline figures of both periods and the 2023 comparison.

    The internal ``rows`` of each period are left out (they are the ``elections`` part).

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``sesgo-cis-summary`` with ``with``, ``without`` and ``election``.
    """
    periods = {key: {k: v for k, v in study.summary[key].items() if k != 'rows'} for key in ('with', 'without')}
    return _rounded({**periods, 'election': study.summary['election']}), {}


def export_lean_elections(study) -> tuple:
    """Export one row per election and period of the pollster, newest first.

    Returns
    -------
    tuple
        ``(data, {'elections': frame})``: ``data`` for ``sesgo-cis-elections`` with ``rows``.
    """
    rows = _rounded(study.elections)
    return {'rows': rows}, {'elections': pd.DataFrame(rows)}


def export_lean_houses(study) -> tuple:
    """Export one row per house with its lean toward the left bloc.

    Returns
    -------
    tuple
        ``(data, {'houses': frame})``: ``data`` for ``sesgo-cis-houses`` with ``rows``.
    """
    rows = _rounded(study.houses)
    return {'rows': rows}, {'houses': pd.DataFrame(rows)}


def export_lean_matrix(study) -> tuple:
    """Export the house by election matrix of the error of the left bloc.

    Returns
    -------
    tuple
        ``(data, {'matrix': frame})``: ``data`` for ``sesgo-cis-matrix`` with ``events``, ``houses`` and ``cells``
        (``[event index, house index, error]``); the frame is the long form
        ``pollster, scope, event_date, Izquierda``.
    """
    matrix = _rounded(study.matrix)
    long = [{'pollster': matrix['houses'][house], 'scope': matrix['events'][event]['scope'],
             'event_date': matrix['events'][event]['event_date'], 'Izquierda': value}
            for event, house, value in matrix['cells']]
    return matrix, {'matrix': pd.DataFrame(long, columns=['pollster', 'scope', 'event_date', 'Izquierda'])}


def export_lean_election(study) -> tuple:
    """Export the study election (2023): result, CIS, consensus and every house.

    Returns
    -------
    tuple
        ``(data, {'election-houses': frame})``: ``data`` for ``sesgo-cis-election``.
    """
    election = _rounded(study.election)
    return election, {'election-houses': pd.DataFrame(election['houses'])}


def export_lean_meta(study, study_id: str = 'sesgo-cis', registration: Optional[dict] = None,
                     checked_at: Optional[str] = None, prov: Optional[dict] = None,
                     seconds: Optional[dict] = None) -> tuple:
    """Export the description of the study: titles, periods, parameters, inputs, sources and provenance.

    ``registration`` and ``checked_at`` are accepted for the signature of the other studies and ignored: this
    study has nothing to register or check.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``sesgo-cis-meta``. No CSV twin.
    """
    title, subtitle = STUDY_TITLES[study_id]
    prov = prov or {'commit': None, 'dirty': False, 'versions': {}}
    data = {
        'study': study_id, 'title': title,
        'subtitle': subtitle.format(k=study.summary['with']['k_left'], n=study.summary['with']['n']),
        'pollster': study.params['pollster'], 'periods': study.periods, 'blocks': study.params['blocks'],
        'params': study.params, 'inputs': study.inputs, 'sources': [dict(s) for s in LEAN_SOURCES],
        'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'], 'seconds': seconds or {},
    }
    return data, {}
