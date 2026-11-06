"""Exporters of the shy-vote study (``voto-oculto``) and its computation for ``publish_study``.

The study package (``mtpy.lib.studies.shy``) is imported inside ``compute_for_publish``, never at module level: the
web imports ``publish`` and must stay light.
"""
from typing import Optional

import pandas as pd

from .. import bundle
from .studies import _rounded, STUDY_TITLES
from ...core.io import FileSystem

SHY_SOURCES = [
    {'label': 'Tezanos reconoce errores al estimar el voto de Vox en algunos sondeos (The Objective, 16 de marzo '
              'de 2022)',
     'url': 'https://theobjective.com/espana/2022-03-16/tezanos-reconoce-errores-al-estimar-el-voto-de-vox-en-'
            'algunos-sondeos-y-anuncia-cambios-en-los-modelos-de-estimacion/'},
    {'label': '¿Son fiables las encuestas electorales? El voto oculto del PP (El Independiente, 28 de febrero de '
              '2019)',
     'url': 'https://www.elindependiente.com/opinion/2019/02/28/encuestas-electorales-son-fiables/'},
    {'label': 'Alaminos y Alaminos-Fernández, «Recuerdo de voto y distorsiones analíticas: el caso español tras las '
              'elecciones generales de 2023», REIS 192 (2025)',
     'url': 'https://reis.cis.es/index.php/reis/article/view/2549'},
    {'label': 'Pavía y Larraz, «Sesgo de no-respuesta y modelos de superpoblación en encuestas electorales», '
              'REIS 137 (2012)',
     'url': 'https://reis.cis.es/index.php/reis/article/view/656'},
]

# The lineage keys of an election row, as the columns ``name_{key}``, ``polls_{key}`` and ``results_{key}``
LINEAGES = ('PSOE', 'PP', 'VOX', 'UP')
KEYS = ('Izquierda', 'Derecha') + LINEAGES


def compute_for_publish(data: FileSystem, writer: bundle.BundleWriter):
    """Compute the study from the published parts of the live bundle (even if the writer is a dry run).

    Parameters
    ----------
    data : FileSystem
        The data folder (the parameters, the party in government and the catalogue).
    writer : BundleWriter
        The writer of the publication; only its file system is used.

    Returns
    -------
    ShyStudy
        The computed study.
    """
    from ..studies.shy.study import compute_shy

    return compute_shy(bundle.BundleReader(writer.fs, prefix=bundle.PREFIX), data)


def _clean(obj):
    """Plain numbers (``None`` for NaN) without rounding: the study rounds its figures."""
    return _rounded(obj, None)


def export_shy_summary(study) -> tuple:
    """Export the headline figures of the study.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``voto-oculto-summary``.
    """
    return _clean(study.summary), {}


def _flat_election(row: dict) -> dict:
    """An election row with its nested values flattened to scalar columns: ``blocks`` as ``block_left`` and
    ``block_right`` (members joined by ``+``), ``names`` as ``name_{key}`` and ``polls`` and ``results`` as
    ``polls_{key}`` and ``results_{key}`` in lower case."""
    flat = {key: value for key, value in row.items() if key not in ('blocks', 'names', 'polls', 'results')}
    blocks = row['blocks']
    flat['block_left'] = '+'.join(blocks['Izquierda']) if blocks else None
    flat['block_right'] = '+'.join(blocks['Derecha']) if blocks else None

    for key in LINEAGES:
        flat[f'name_{key.lower()}'] = (row['names'] or {}).get(key)

    for part in ('polls', 'results'):
        for key in KEYS:
            flat[f'{part}_{key.lower()}'] = row[part].get(key)

    return flat


def export_shy_elections(study) -> tuple:
    """Export one row per election, chronological.

    Returns
    -------
    tuple
        ``(data, {'elections': frame})``: ``data`` for ``voto-oculto-elections`` with ``rows``; the frame is the
        flat form.
    """
    rows = _clean(study.elections)
    return {'rows': rows}, {'elections': pd.DataFrame([_flat_election(row) for row in rows])}


def export_shy_groups(study) -> tuple:
    """Export the figures blocks of the summary as long rows.

    Returns
    -------
    tuple
        ``(data, {'groups': frame})``: ``data`` for ``voto-oculto-groups`` with ``rows``.
    """
    rows = _clean(study.groups)
    return {'rows': rows}, {'groups': pd.DataFrame(rows)}


def _tail(summary: dict, labels: dict) -> str:
    """The last clause of the subtitle, from the verdicts: the direction of the error cannot be guessed, leans to
    the left or to the parties named by the third verdict."""
    verdicts = summary['verdicts']

    if verdicts['p1'] == 'SÍ':
        return ', y el sentido del error se inclina hacia la izquierda'

    if verdicts['p3'] == 'DE NADIE':
        return ', y el sentido del error no se puede adivinar'

    cased = {text.upper(): text for text in labels.values()}
    names = [part.removeprefix('DE ') for part in verdicts['p3'].split(' Y DE ')]
    names = [cased.get(name, name) for name in names]

    return f", y el sentido del error se inclina hacia {' y '.join(names)}"


def _subtitle(template: str, study) -> str:
    """Render the subtitle template from the summary: the clause about Vox appears only if there is data."""
    summary = study.summary
    left = summary['left']['all']
    vox = summary['vox']
    vox_clause = '' if vox is None else f"; a Vox, menos en {vox['under']} de {vox['n']}"
    size = f"{summary['size']['mean']:.1f}".replace('.', ',')

    return template.format(n=summary['n_elections'], k_under=left['under'], k_over=left['over'],
                           vox_clause=vox_clause, size=size, tail=_tail(summary, study.params['labels']))


def export_shy_meta(study, study_id: str = 'voto-oculto', registration: Optional[dict] = None,
                    checked_at: Optional[str] = None, prov: Optional[dict] = None,
                    seconds: Optional[dict] = None) -> tuple:
    """Export the description of the study: titles, election in progress, parameters, inputs, sources, provenance.

    ``registration`` and ``checked_at`` are accepted for the signature of the other studies and ignored: this
    study has nothing to register or check.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``voto-oculto-meta``. No CSV twin.
    """
    title, template = STUDY_TITLES[study_id]
    prov = prov or {'commit': None, 'dirty': False, 'versions': {}}
    data = {
        'study': study_id, 'title': title, 'subtitle': _subtitle(template, study), 'scopes': study.scopes,
        'election': study.election, 'params': study.params, 'inputs': study.inputs,
        'sources': [dict(s) for s in SHY_SOURCES],
        'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'], 'seconds': seconds or {},
    }
    return data, {}
