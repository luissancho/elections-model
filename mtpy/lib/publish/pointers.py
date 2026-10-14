"""Manifest and run pointers: ``rebuild_history``, ``update_manifest``, ``point`` and ``unpublish``."""
from datetime import date, datetime
from typing import Optional

from .. import bundle
from ..bundle import BundleReader, BundleWriter
from .common import ATTRIBUTION, latest_entry, _today


def rebuild_history(writer: BundleWriter, scope: str) -> dict:
    """Rebuild ``history.json`` of a scope from the headlines of its complete runs.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to read and write.
    scope : str
        Scope code.

    Returns
    -------
    dict
        The history data, ``{'scope', 'runs'}``.
    """
    runs = [writer.read_json(bundle.path_part(scope, rid, 'headline'))['data'] for rid in writer.list_runs(scope)]
    data = {'scope': scope, 'runs': runs}
    writer.write_json(bundle.path_history(scope), 'history', data, scope, run_id=None)
    return data


def read_manifest(reader: BundleReader) -> dict:
    """Read the manifest data, or the default one when the bundle has none yet.

    The decision rests on ``FileNotFoundError`` from the read, not on ``exists``: fsspec's ``exists``
    returns ``False`` on any error, so a transient S3 failure would replace the real manifest.

    Parameters
    ----------
    reader : BundleReader
        Bundle to read.

    Returns
    -------
    dict
        Manifest data.

    Raises
    ------
    Exception
        Any read error other than a missing manifest (``FileNotFoundError``) propagates.
    """
    try:
        data = reader.read_json(bundle.path_manifest())['data']
    except FileNotFoundError:  # only a missing manifest; any other read error must not be overwritten
        data = None
    if data is not None:
        data['freeze'] = {'since': None, **(data.get('freeze') or {})}
        return data
    return {'contract': bundle.CONTRACT, 'updated_at': None, 'scopes': {},
            'freeze': {'active': False, 'message': None, 'since': None}, 'attribution': dict(ATTRIBUTION)}


def _write_manifest(writer: BundleWriter, data: dict) -> dict:
    """Stamp the manifest data (time, contract and current ``ATTRIBUTION``), write it and return it."""
    data['updated_at'] = bundle.iso_utc(writer.now())
    data['contract'] = bundle.CONTRACT
    data['attribution'] = dict(ATTRIBUTION)
    writer.write_json(bundle.path_manifest(), 'manifest', data, scope=None)
    return data


def update_manifest(writer: BundleWriter, scopes: Optional[dict] = None, freeze: Optional[dict] = None,
                    today: Optional[str | date | datetime] = None) -> dict:
    """Merge scope entries and/or the freeze switch into the manifest and write it.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to update.
    scopes : dict, optional
        Entries by scope code; each replaces the existing entry of its scope.
    freeze : dict, optional
        ``{'active': bool, 'message': str | None}``. The manifest stores it with ``since``: the date
        (Europe/Madrid) it went from inactive to active, kept while it stays active and ``None`` when
        it is switched off.
    today : str, datetime.date or datetime.datetime, optional
        Date taken as today for ``since``, as in ``loreg_guard``.

    Returns
    -------
    dict
        The manifest data written.

    Raises
    ------
    ValueError
        ``freeze`` has no boolean ``active``.
    """
    if freeze is not None and not (isinstance(freeze, dict) and isinstance(freeze.get('active'), bool)):
        raise ValueError('publish: freeze needs a boolean "active"')
    data = read_manifest(writer)
    data['scopes'].update(scopes or {})
    if freeze is not None:
        previous = data['freeze']
        since = None
        if freeze['active']:
            since = previous['since'] if previous.get('active') and previous.get('since') else _today(today).isoformat()
        data['freeze'] = {'active': freeze['active'], 'message': freeze.get('message'), 'since': since}
    return _write_manifest(writer, data)


def _check_run(scope: str, run_id: str) -> None:
    """Validate a scope and a run id before any bundle path is built from them.

    Parameters
    ----------
    scope, run_id : str
        Scope code (``ROUTE_ALIAS_RE``) and run id (``RUN_ID_RE``).

    Raises
    ------
    ValueError
        The scope or the run id is malformed (``'/'``, ``'..'``, a date...).
    """
    if not (isinstance(scope, str) and bundle.ROUTE_ALIAS_RE.fullmatch(scope)):
        raise ValueError('invalid scope {!r}'.format(scope))
    if not (isinstance(run_id, str) and bundle.RUN_ID_RE.fullmatch(run_id)):
        raise ValueError('{}: invalid run id {!r}'.format(scope, run_id))


def point(writer: BundleWriter, scope: str, run_id: str) -> dict:
    """Point the manifest of a scope to an existing run.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to update.
    scope, run_id : str
        Scope and run to publish as the latest.

    Returns
    -------
    dict
        The manifest data written.

    Raises
    ------
    ValueError
        The scope or run id is malformed, or the run does not exist or is incomplete.
    """
    _check_run(scope, run_id)
    name = bundle.path_part(scope, run_id, 'headline')
    if not writer.exists(name):
        raise ValueError('{}: run {} not found'.format(scope, run_id))
    return update_manifest(writer, {scope: latest_entry(writer.read_json(name)['data'])})


def unpublish(writer: BundleWriter, scope: str, run_id: str) -> dict:
    """Delete a run, rebuild the history and repoint the manifest if it pointed to that run.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to update.
    scope, run_id : str
        Run to delete.

    Returns
    -------
    dict
        ``{'removed': run_id, 'latest': run now published for the scope or None}``.

    Raises
    ------
    ValueError
        The scope or run id is malformed, or the run does not exist.
    """
    _check_run(scope, run_id)
    if not writer.exists(bundle.path_run(scope, run_id)):
        raise ValueError('{}: run {} not found'.format(scope, run_id))
    writer.remove(bundle.path_run(scope, run_id))
    rebuild_history(writer, scope)
    manifest = read_manifest(writer)
    latest = manifest['scopes'].get(scope, {}).get('latest')
    if latest == run_id:
        runs = writer.list_runs(scope)
        if runs:
            point(writer, scope, runs[-1])
            latest = runs[-1]
        else:
            del manifest['scopes'][scope]
            _write_manifest(writer, manifest)
            latest = None
    return {'removed': run_id, 'latest': latest}
