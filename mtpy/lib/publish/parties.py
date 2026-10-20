"""Exporter and ``publish_parties`` for the party catalogue of the bundle (``parties.json``)."""
from typing import Callable, Optional

import pandas as pd

from .. import bundle
from ..bundle import BundleWriter
from ..data import get_parties


PARTIES_COLUMNS = ['id', 'name', 'fullname', 'parent_id', 'color', 'block', 'regional']


def _text(value) -> Optional[str]:
    """Return ``value`` as text, or ``None`` when it is missing (``None``, NaN or NA)."""
    return None if pd.isna(value) else str(value)


def export_parties(parties: pd.DataFrame) -> dict:
    """Export the party catalogue (``data.get_parties``), one row per party in the order of the table.

    Parameters
    ----------
    parties : pandas.DataFrame
        Parties table with ``PARTIES_COLUMNS`` (``block`` may be categorical, ``parent_id`` and ``regional``
        may hold NaN).

    Returns
    -------
    dict
        Data for the ``parties`` schema: ``{'rows': [...]}`` with ``id`` (int), ``name``, ``fullname``,
        ``parent_id`` (int or null), ``color``, ``block`` and ``regional`` (bool or null) per party; missing
        texts are null.
    """
    rows = []

    for record in parties.reindex(columns=PARTIES_COLUMNS).to_dict(orient='records'):
        rows.append({
            'id': int(record['id']),
            'name': str(record['name']),
            'fullname': _text(record['fullname']),
            'parent_id': None if pd.isna(record['parent_id']) else int(record['parent_id']),
            'color': _text(record['color']),
            'block': _text(record['block']),
            'regional': None if pd.isna(record['regional']) else bool(record['regional']),
        })

    return {'rows': rows}


def publish_parties(writer: BundleWriter, parties: Optional[Callable[[], pd.DataFrame]] = None) -> dict:
    """Write the party catalogue of the database into the bundle, overwriting the previous one.

    The file lives next to the manifest (``bundle.path_parties``), outside the runs: it carries no ``scope``
    nor ``run_id`` and is rewritten on every publication of a forecast or an analysis, so the pages can colour
    and name every party, simulated or not.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to write into.
    parties : callable, optional
        Source of the parties table; ``data.get_parties`` by default (tests inject a frame without a database).

    Returns
    -------
    dict
        ``{'n_parties': number of rows written}``.
    """
    data = export_parties((parties or get_parties)())
    writer.write_json(bundle.path_parties(), 'parties', data, None)

    return {'n_parties': len(data['rows'])}
