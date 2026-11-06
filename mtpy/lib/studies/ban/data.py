"""
Readers of the inputs of the ban study: the parts already published in the bundle.

The study does not touch the database. It reads, for every published election of every scope of the manifest, the
``meta``, ``ban``, ``results``, ``pollsters`` and ``model`` parts of its event folder, and the ``meta`` of the latest
national run (only its ``event_date``), all through a ``BundleReader``, plus the small parameters file in ``data/``.

The legal guard: an election folder without ``results`` (the election in progress, whose ``ban`` part may exist) is
skipped before any of its parts is read, and the polls of a run are never read. This module keeps independent of the
web layer.
"""
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...bundle import BundleReader, path_event, path_manifest, path_part
from ....core.io import FileSystem
from ..labels import election_label, short_year_label

PARAMS_FILE = 'andorra/params.json'
# The poll columns the study keeps, before one column per party
POLL_COLUMNS = ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'sample_size', 'mtype',
                'days', 'error_avg']


class MissingInput(Exception):
    """
    An input the study needs is not published.

    Parameters
    ----------
    part : str
        Name of the missing input (``pairs`` when no election has a pair of polls).

    Attributes
    ----------
    part : str
        The same name.
    """

    def __init__(self, part: str):
        super().__init__(f'missing input: {part}')
        self.part = part


@dataclass
class Event:
    """
    One published election with its clandestine polls and what the study reads with them.

    Attributes
    ----------
    scope : str
        Scope code.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    label : str
        Public label: the year (``short_year_label``) in the national scope, ``{name} {year}``
        (``election_label``) elsewhere; with the month when another election of the scope shares the year.
    name : str or None
        Name of the election from its ``meta``.
    national : bool
        Whether the scope is ``params['national']``.
    main : list of str
        The main parties of the election (``meta.main``).
    polls : pandas.DataFrame
        The clandestine polls of the ``ban`` part, in its order: the ``POLL_COLUMNS`` and one float column per
        party of ``main`` (NaN for a missing figure); ``sample_size``, ``days`` and ``error_avg`` float with NaN for
        null.
    legal : pandas.DataFrame
        The last legal poll of each house of ``polls`` (``ban.legal``), with the same columns.
    pollsters : pandas.DataFrame or None
        The rows of the ``pollsters`` part (``last_days``, ``last_error_avg``…); ``None`` without it.
    official : pandas.Series or None
        The ``pct`` of the ``results`` part by party, restricted to ``main``; ``None`` when no main party is in it.
    frozen : pandas.DataFrame or None
        The ``shares`` rows of the ``model`` part at ``params['frozen_horizon']`` for the parties of ``main``,
        indexed by ``party`` (in the order of ``main``) with ``model`` and ``official``; ``None`` without the part
        or without rows at that horizon.
    with_ban : pandas.DataFrame or None
        The ``ban`` rows of the ``model`` part (the backtest with the clandestine polls in), shaped as ``frozen``;
        ``None`` when ``frozen`` is ``None`` or the part lacks them.
    """
    scope: str
    event_date: str
    label: str
    name: Optional[str]
    national: bool
    main: list
    polls: pd.DataFrame
    legal: pd.DataFrame
    pollsters: Optional[pd.DataFrame]
    official: Optional[pd.Series]
    frozen: Optional[pd.DataFrame]
    with_ban: Optional[pd.DataFrame]


def load_params(data: FileSystem) -> dict:
    """
    Parameters of the study (``data/andorra/params.json``).

    Parameters
    ----------
    data : FileSystem
        The ``data/`` directory of the repository.

    Returns
    -------
    dict
        The JSON as written: ``national``, ``since``, ``chart_since``, ``max_days``, ``frozen_horizon``,
        ``min_pairs``, ``big_sample``, ``exclude_mtypes_pairs``, ``exclude_mtypes_consensus`` and ``half``.
    """
    return json.loads(data.read(PARAMS_FILE))


def list_scopes(reader: BundleReader, national: str = 'es') -> list:
    """
    The scopes of the manifest, the national one first.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    national : str, default 'es'
        Code of the national scope.

    Returns
    -------
    list of str
        The keys of ``manifest.scopes`` in their order with ``national`` moved first; empty without a manifest.
    """
    if not reader.exists(path_manifest()):
        return []

    scopes = list(reader.read_json(path_manifest())['data']['scopes'])

    return sorted(scopes, key=lambda scope: scope != national)


def _polls_frame(records: list, parties: list) -> pd.DataFrame:
    """The polls of a ``ban`` part as a table: the ``POLL_COLUMNS`` and one float column per party (NaN for a
    missing figure), ``sample_size``, ``days`` and ``error_avg`` float with NaN for null."""
    frame = pd.DataFrame(records).reindex(columns=POLL_COLUMNS + list(parties))

    for column in ['sample_size', 'days', 'error_avg'] + list(parties):
        frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(float)

    return frame


def _model_rows(rows: list, main: list, horizon: Optional[int] = None) -> Optional[pd.DataFrame]:
    """The rows of a ``model`` table (``shares`` or ``ban``) of the parties of ``main``, at ``horizon`` when given,
    indexed by ``party`` in the order of ``main`` with float ``model`` and ``official``; ``None`` without rows."""
    frame = pd.DataFrame(rows, columns=['horizon', 'party', 'model', 'official'])

    if horizon is not None:
        frame = frame[frame['horizon'] == horizon]

    frame = frame[frame['party'].isin(main)].drop_duplicates('party').set_index('party')
    frame = frame.reindex([party for party in main if party in frame.index])[['model', 'official']].astype(float)

    return frame if len(frame) else None


def _read(reader: BundleReader, scope: str, date: str, part: str) -> Optional[dict]:
    """The ``data`` of an event part, ``None`` when it is not published."""
    name = path_event(scope, date, part)

    return reader.read_json(name)['data'] if reader.exists(name) else None


def _check_event(reader: BundleReader, scope: str, date: str) -> Optional[str]:
    """Why an election is skipped (``'no results'`` or ``'no ban'``), ``None`` when it can be read. Only checks
    existence: nothing of an election without results is read."""
    if not reader.exists(path_event(scope, date, 'results')):
        return 'no results'

    if not reader.exists(path_event(scope, date, 'ban')):
        return 'no ban'

    return None


def read_event(reader: BundleReader, scope: str, date: str, params: dict, repeated: bool) -> Optional[Event]:
    """
    One published election with its clandestine polls.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    scope : str
        Scope code.
    date : str
        Election date, ``YYYY-MM-DD``.
    params : dict
        Parameters of the study (``load_params``); uses ``national`` and ``frozen_horizon``.
    repeated : bool
        Whether another published election of the scope shares the year (the label then carries the month).

    Returns
    -------
    Event or None
        The election; ``None`` when its ``results`` or its ``ban`` part is missing (``read_events`` says which),
        in which case none of its parts is read.
    """
    if _check_event(reader, scope, date) is not None:
        return None

    meta = _read(reader, scope, date, 'meta')
    ban = _read(reader, scope, date, 'ban')
    results = _read(reader, scope, date, 'results')
    pollsters = _read(reader, scope, date, 'pollsters')
    model = _read(reader, scope, date, 'model')

    main = list(meta.get('main') or ban['parties'])
    national = scope == params['national']
    name = meta.get('name')
    label = short_year_label(date, repeated) if national else election_label(scope, name, date, repeated)

    pct = pd.Series({row['party']: row['pct'] for row in results['rows']}, dtype=float)
    official = pct.reindex([party for party in main if party in pct.index])

    frozen = with_ban = None
    if model is not None:
        frozen = _model_rows(model.get('shares') or [], main, params['frozen_horizon'])
        if frozen is not None and model.get('ban'):
            with_ban = _model_rows(model['ban'], main)

    return Event(scope=scope, event_date=date, label=label, name=name, national=national, main=main,
                 polls=_polls_frame(ban['polls'], main), legal=_polls_frame(ban['legal'], main),
                 pollsters=pd.DataFrame(pollsters['rows']) if pollsters is not None else None,
                 official=official if len(official) else None, frozen=frozen, with_ban=with_ban)


def _parts_found(reader: BundleReader, scope: str, date: str) -> list:
    """The parts of an election the study reads that are published, in reading order."""
    return [part for part in ('meta', 'ban', 'results', 'pollsters', 'model')
            if reader.exists(path_event(scope, date, part))]


def read_events(reader: BundleReader, params: dict) -> tuple[list, dict]:
    """
    Every published election with clandestine polls of every scope of the manifest since ``params['since']``.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``national``, ``since`` and ``frozen_horizon``.

    Returns
    -------
    events : list of Event
        The elections, ascending by date (the national scope first on the same date).
    inputs : dict
        ``{scope: {'read': [{'event_date', 'parts'}], 'skipped': {date: reason}}}`` with reason ``'no results'``
        (the election in progress: none of its parts is read) or ``'no ban'``.
    """
    scopes = list_scopes(reader, params['national'])
    events = []
    inputs = {}

    for scope in scopes:
        dates = [date for date in reader.list_events(scope) if date >= params['since']]
        years = pd.Series([date[:4] for date in dates], dtype=object).value_counts()
        inputs[scope] = {'read': [], 'skipped': {}}

        for date in dates:
            reason = _check_event(reader, scope, date)

            if reason is not None:
                inputs[scope]['skipped'][date] = reason
                continue

            events.append(read_event(reader, scope, date, params, bool(years[date[:4]] > 1)))
            inputs[scope]['read'].append({'event_date': date, 'parts': _parts_found(reader, scope, date)})

    order = {scope: index for index, scope in enumerate(scopes)}
    events.sort(key=lambda event: (event.event_date, order[event.scope]))

    return events, inputs


def read_next(reader: BundleReader, params: dict) -> Optional[dict]:
    """
    The election of the latest national run and whether it is published with results.

    Only the ``meta`` of the run is read, for its ``event_date``.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``national``.

    Returns
    -------
    dict or None
        ``{'event_date', 'open'}`` with ``open`` true when ``events/{national}/{event_date}/results`` exists;
        ``None`` without a manifest, without a national run or without its ``meta``.
    """
    if not reader.exists(path_manifest()):
        return None

    national = params['national']
    run_id = reader.read_json(path_manifest())['data']['scopes'].get(national, {}).get('latest')
    meta = path_part(national, run_id, 'meta') if run_id else None

    if meta is None or not reader.exists(meta):
        return None

    event_date = reader.read_json(meta)['data']['event_date']

    return {'event_date': event_date, 'open': reader.exists(path_event(national, event_date, 'results'))}
