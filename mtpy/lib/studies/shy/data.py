"""
Readers of the inputs of the shy vote study: the parts already published in the bundle.

The study does not touch the database. It reads, for every published election of every scope of the catalogue, the
``meta``, ``cycle`` and ``results`` parts, and the ``meta`` and ``polls`` of the latest national run while its
election has no published results, all through a ``BundleReader``, plus the small files in ``data/`` (the parameters,
the party in government on every election day and the scopes catalogue). This module keeps independent of the web
layer.
"""
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...bundle import BundleReader, path_event, path_manifest, path_part
from ....core.io import FileSystem
from ..lean.data import NATIONAL, _label, _short

PARAMS_FILE = 'voto-oculto/params.json'
INCUMBENTS_FILE = 'voto-oculto/incumbents.json'
CATALOGUE_FILE = 'es-scopes.csv'
# The poll columns the study keeps, before one column per party
POLL_COLUMNS = ['date', 'pollster_id', 'pollster', 'start_date', 'end_date', 'sample_size', 'mtype', 'weight']
# The parts an election needs to be read
ELECTION_PARTS = ('meta', 'cycle', 'results')


class MissingInput(Exception):
    """
    A part the study needs is not published.

    Parameters
    ----------
    part : str
        Name of the missing part, relative to the bundle (``events/es/2023-07-23/cycle``).

    Attributes
    ----------
    part : str
        The same name.
    """

    def __init__(self, part: str):
        super().__init__(f'missing input: {part}')
        self.part = part


@dataclass
class Election:
    """
    A published election with its final polls and its result.

    Attributes
    ----------
    scope : str
        Scope code.
    scope_name : str
        Name of the scope in the catalogue.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    general : bool
        Whether it is a national election.
    label : str
        Public label (``Generales 2023``).
    short : str
        Short label (``GEN 23``).
    blocks : dict or None
        ``{block: [parties]}`` of the election, ``None`` if it has none.
    parties : list of str
        The parties of the polls, in the order of the ``cycle`` part.
    results : dict
        ``{party: percentage}`` of the result.
    n_cycle : int
        Number of polls of the whole cycle, before keeping the final ones.
    polls : pandas.DataFrame
        The final polls (``final_polls``) with the ``POLL_COLUMNS`` and one float column per party.
    """
    scope: str
    scope_name: str
    event_date: str
    general: bool
    label: str
    short: str
    blocks: Optional[dict]
    parties: list
    results: dict
    n_cycle: int
    polls: pd.DataFrame


def load_params(data: FileSystem) -> dict:
    """
    Parameters of the study (``data/voto-oculto/params.json``).

    Parameters
    ----------
    data : FileSystem
        The ``data/`` directory of the repository.

    Returns
    -------
    dict
        The JSON as written: ``window_days``, ``recent_from``, ``blocks``, ``lineages``, ``labels``,
        ``size_parties``, ``size_threshold``, ``big_threshold``, ``pp_early`` and ``extremes``.
    """
    return json.loads(data.read(PARAMS_FILE))


def load_incumbents(data: FileSystem) -> dict:
    """
    The party in government on every election day (``data/voto-oculto/incumbents.json``).

    Parameters
    ----------
    data : FileSystem
        The ``data/`` directory of the repository.

    Returns
    -------
    dict
        ``{scope: {date: [names]}}``: the names under which the party can appear in the results, the first one
        present counting.
    """
    return json.loads(data.read(INCUMBENTS_FILE))


def load_catalogue(data: FileSystem) -> list:
    """
    The scopes of the catalogue (``data/es-scopes.csv``), the national one and the regional ones.

    Parameters
    ----------
    data : FileSystem
        The ``data/`` directory of the repository.

    Returns
    -------
    list of tuple
        ``(scode, name)`` of the rows whose ``scode`` is ``es`` or starts with ``es-``, in file order.
    """
    frame = data.read_csv(CATALOGUE_FILE)

    return [(code, name) for code, name in zip(frame['scode'], frame['name'])
            if code == NATIONAL or code.startswith(f'{NATIONAL}-')]


def final_polls(polls: pd.DataFrame, until: str, window_days: int) -> pd.DataFrame:
    """
    The final polls: those with weight whose fieldwork ended in the last days before a date.

    Parameters
    ----------
    polls : pandas.DataFrame
        Polls with ``date``, ``end_date`` (may be null) and ``weight``.
    until : str
        The date the window ends at, ``YYYY-MM-DD`` (the election or the ``as_of`` of a run).
    window_days : int
        Length of the window in days.

    Returns
    -------
    pandas.DataFrame
        The rows with ``weight > 0`` whose ``end_date``, or ``date`` without it, is between 0 and ``window_days``
        days before ``until``, in their order.
    """
    end = pd.to_datetime(polls['end_date'].where(polls['end_date'].notna(), polls['date']))
    days = (pd.Timestamp(until) - end).dt.days

    return polls[(polls['weight'] > 0) & days.between(0, window_days)]


def _polls_frame(records: list, parties: list) -> pd.DataFrame:
    """The polls of a ``cycle`` or run ``polls`` part as a table: the ``POLL_COLUMNS`` and one float column per
    party (NaN for a missing figure), ``sample_size`` float with NaN for null."""
    frame = pd.DataFrame(records).reindex(columns=POLL_COLUMNS + list(parties))
    frame['sample_size'] = pd.to_numeric(frame['sample_size'], errors='coerce').astype(float)
    frame['weight'] = frame['weight'].astype(float)
    frame[list(parties)] = frame[list(parties)].astype(float)

    return frame


def is_current(reader: BundleReader, scope: str, date: str) -> bool:
    """
    Whether an election is still to come: its event folder has no ``results`` part.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    scope : str
        Scope code.
    date : str
        Election date, ``YYYY-MM-DD``.

    Returns
    -------
    bool
        ``True`` if ``events/{scope}/{date}/results.json`` does not exist.
    """
    return not reader.exists(path_event(scope, date, 'results'))


def _read_election(reader: BundleReader, scope: str, name: str, date: str, month: bool, params: dict):
    """The election of a scope and date, or the reason it is skipped (``'parts'``, ``'no final polls'``)."""
    if not all(reader.exists(path_event(scope, date, part)) for part in ELECTION_PARTS):
        return 'parts'

    meta = reader.read_json(path_event(scope, date, 'meta'))['data']
    cycle = reader.read_json(path_event(scope, date, 'cycle'))['data']
    rows = reader.read_json(path_event(scope, date, 'results'))['data']['rows']
    parties = list(cycle['parties'])
    polls = final_polls(_polls_frame(cycle['polls'], parties), date, params['window_days'])

    if polls.empty:
        return 'no final polls'

    return Election(scope=scope, scope_name=name, event_date=date, general=scope == NATIONAL,
                    label=_label(scope, name, date, month), short=_short(scope, date, month),
                    blocks=meta.get('blocks') or None, parties=parties,
                    results={row['party']: float(row['pct']) for row in rows}, n_cycle=len(cycle['polls']),
                    polls=polls.reset_index(drop=True))


def read_elections(reader: BundleReader, catalogue: list, params: dict) -> tuple[list[Election], dict]:
    """
    The published elections of the scopes of the catalogue with a result and final polls.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    catalogue : list of tuple
        ``(scode, name)`` of the scopes (``load_catalogue``).
    params : dict
        Parameters of the study (``load_params``); uses ``window_days``.

    Returns
    -------
    elections : list of Election
        Ascending by date, ties in the order of the catalogue. The label carries the month when two published
        elections of the scope share the year.
    inputs : dict
        By scope: ``{'events': n, 'skipped': [{'date', 'reason'}]}`` with ``n`` the elections read and ``reason``
        ``'parts'`` (a part is missing) or ``'no final polls'``; or ``{'skipped': 'no events'}`` for a scope without
        events.
    """
    found = []
    inputs = {}

    for order, (scope, name) in enumerate(catalogue):
        try:
            dates = reader.list_events(scope)
        except Exception:
            dates = []

        if not dates:
            inputs[scope] = {'skipped': 'no events'}
            continue

        skipped = []
        count = 0
        years = [date[:4] for date in dates]

        for date in dates:
            election = _read_election(reader, scope, name, date, years.count(date[:4]) > 1, params)

            if isinstance(election, str):
                skipped.append({'date': date, 'reason': election})
                continue

            found.append((date, order, election))
            count += 1

        inputs[scope] = {'events': count, 'skipped': skipped}

    return [election for _, _, election in sorted(found, key=lambda item: item[:2])], inputs


def read_current(reader: BundleReader, params: dict) -> tuple[Optional[dict], dict]:
    """
    The election in progress: the latest national run while its election has no published results.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``window_days``.

    Returns
    -------
    current : dict or None
        ``{'run_id', 'as_of', 'event_date', 'blocks', 'main', 'polls'}`` with ``blocks`` and ``main`` from the
        ``bmaps`` of the run (``None`` and an empty list without them) and ``polls`` the final polls of the run
        (``final_polls`` at ``as_of``); ``None`` without a manifest, without a run or once its election is
        published.
    inputs : dict
        ``{'run_id', 'as_of'}`` of the run read, or ``{'skipped': reason}`` with reason ``'no manifest'``,
        ``'no run'`` or ``'election published'``.
    """
    if not reader.exists(path_manifest()):
        return None, {'skipped': 'no manifest'}

    run_id = reader.read_json(path_manifest())['data']['scopes'].get(NATIONAL, {}).get('latest')

    if not run_id:
        return None, {'skipped': 'no run'}

    meta = reader.read_json(path_part(NATIONAL, run_id, 'meta'))['data']

    if not is_current(reader, NATIONAL, meta['event_date']):
        return None, {'skipped': 'election published'}

    polls = reader.read_json(path_part(NATIONAL, run_id, 'polls'))['data']
    bmaps = meta.get('bmaps') or {}
    current = {'run_id': run_id, 'as_of': meta['as_of'], 'event_date': meta['event_date'],
               'blocks': bmaps.get('blocks') or None, 'main': list(bmaps.get('main') or []),
               'polls': final_polls(_polls_frame(polls['polls'], polls['parties']), meta['as_of'],
                                    params['window_days']).reset_index(drop=True)}

    return current, {'run_id': run_id, 'as_of': meta['as_of']}
