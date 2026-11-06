"""
Readers of the inputs of the herd study: the parts already published in the bundle.

The study does not touch the database. It reads, through a ``BundleReader``, the ``cycle`` part (the polls of the whole
cycle) of every published election of every scope of the manifest (the ones with a ``results`` part), the ``meta`` and
``polls`` of the latest run of the live scope while its election has no published results, and the small parameters
file in ``data/``. This module keeps independent of the web layer.
"""
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...bundle import BundleReader, path_event, path_manifest, path_part
from ...mtypes import DROP_MTYPES
from ....core.io import FileSystem
from ..ban.data import list_scopes
from ..labels import election_label
from .stats import _is_number

PARAMS_FILE = 'rebano/params.json'
NATIONAL = 'es'


class MissingInput(Exception):
    """
    A part the study needs is not published.

    Parameters
    ----------
    part : str
        Name of the missing part (``events``).

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
    One election with the polls the study compares.

    Attributes
    ----------
    scope : str
        Scope code.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    label : str
        Public label (``Generales 2019``, ``Madrid 2021``).
    kind : str
        ``'generales'`` for the national scope, ``'autonómicas'`` for the rest.
    names : list of str
        The series measured (``main_series``).
    polls : pandas.DataFrame
        The polls kept (``prepare_polls``): ``poll``, ``pollster``, ``days``, ``n`` and one column per series.
    current : bool
        Whether it is the election in progress (read from the latest run) rather than a past one.
    as_of : str or None
        The ``as_of`` of the run for the election in progress; ``None`` for a past one.
    """
    scope: str
    event_date: str
    label: str
    kind: str
    names: list
    polls: pd.DataFrame
    current: bool
    as_of: Optional[str]


def load_params(data: FileSystem) -> dict:
    """
    Parameters of the study (``data/rebano/params.json``).

    Parameters
    ----------
    data : FileSystem
        The ``data/`` directory of the repository.

    Returns
    -------
    dict
        The JSON as written: ``scopes``, ``since``, ``live_scope``, ``max_days``, ``pair_days``, ``periods``,
        ``min_rows``, ``n_series``, ``exclude``, ``permutations`` and ``seed``.
    """
    return json.loads(data.read(PARAMS_FILE))


def main_series(polls: pd.DataFrame, parties: list[str], n_series: int) -> list[str]:
    """
    The series the study measures: the parties with the highest mean share over the polls.

    Parameters
    ----------
    polls : pandas.DataFrame
        The polls of a cycle, one float column per party.
    parties : list of str
        The candidate parties, in the order that breaks ties.
    n_series : int
        How many series to keep.

    Returns
    -------
    list of str
        The ``n_series`` parties with the highest mean share, descending, ties in the order of ``parties``.
    """
    means = [(-polls[name].mean(), position, name) for position, name in enumerate(parties)
             if name in polls and _is_number(polls[name].mean())]

    return [name for _, _, name in sorted(means)[:n_series]]


def prepare_polls(records: list, parties: list, event_date: str, params: dict) -> tuple[pd.DataFrame, list]:
    """
    The polls of a cycle the study compares, and the series it measures.

    Keeps the polls whose fieldwork ended (``end_date``, or ``date`` without it) 0 to ``max_days`` days before the
    election and whose ``mtype`` is not in ``DROP_MTYPES``; the series are the ``n_series`` parties with the highest
    mean share over those polls (the excluded houses and the polls without a sample count here); then keeps those
    with ``sample_size > 0``, ``weight > 0`` and a house not in ``exclude``.

    Parameters
    ----------
    records : list of dict
        The polls of a ``cycle`` or run ``polls`` part.
    parties : list of str
        The parties of the polls, in the order of the part.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    params : dict
        Parameters of the study (``load_params``); uses ``max_days``, ``n_series`` and ``exclude``.

    Returns
    -------
    polls : pandas.DataFrame
        Columns ``poll`` (index of the poll in ``records``, as str), ``pollster``, ``days`` (int), ``n`` (float, the
        sample) and one float column per series.
    names : list of str
        The series.
    """
    frame = pd.DataFrame(records).reindex(columns=['end_date', 'date', 'pollster', 'sample_size', 'weight', 'mtype']
                                          + list(parties))
    frame['poll'] = [str(index) for index in range(len(frame))]
    end = pd.to_datetime(frame['end_date'].where(frame['end_date'].notna(), frame['date']))
    frame['days'] = (pd.Timestamp(event_date) - end).dt.days
    frame['n'] = pd.to_numeric(frame['sample_size'], errors='coerce').astype(float)
    frame['weight'] = pd.to_numeric(frame['weight'], errors='coerce').astype(float)

    for name in parties:
        frame[name] = pd.to_numeric(frame[name], errors='coerce').astype(float)

    frame = frame[frame['days'].between(0, params['max_days']) & ~frame['mtype'].isin(DROP_MTYPES)]
    names = main_series(frame, list(parties), params['n_series'])
    frame = frame[(frame['n'] > 0) & (frame['weight'] > 0) & ~frame['pollster'].isin(params['exclude'])]
    frame = frame[['poll', 'pollster', 'days', 'n'] + names].reset_index(drop=True)
    frame['days'] = frame['days'].astype(int)

    return frame, names


def _label(scope: str, date: str, dates: list, names: dict) -> str:
    """Public label of an election; with the month when another date of ``dates`` shares its year."""
    month = sum(other[:4] == date[:4] for other in set(dates) | {date}) > 1

    return election_label(scope, names.get(scope, scope), date, month=month)


def _kind(scope: str) -> str:
    """``'generales'`` for the national scope, ``'autonómicas'`` for the rest."""
    return 'generales' if scope == NATIONAL else 'autonómicas'


def read_elections(reader: BundleReader, params: dict, names: dict) -> tuple[list[Election], dict]:
    """
    The past elections: every date of every scope with a ``cycle`` and ``results`` part.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``scopes`` (``'all'`` or a list of codes), ``since`` and what
        ``prepare_polls`` uses.
    names : dict
        ``{scope: name}`` of the scopes catalogue, for the labels.

    Returns
    -------
    elections : list of Election
        Ascending by date, the national scope first on the same date.
    inputs : dict
        ``{scope: {'read': [dates], 'skipped': {date: reason}}}`` with reason ``'no cycle'`` or ``'no results'``.
    """
    scopes = list_scopes(reader, NATIONAL)

    if params.get('scopes', 'all') != 'all':
        scopes = [scope for scope in scopes if scope in params['scopes']]

    elections = []
    inputs = {}

    for scope in scopes:
        dates = [date for date in reader.list_events(scope) if params.get('since') is None or date >= params['since']]
        inputs[scope] = {'read': [], 'skipped': {}}

        for date in dates:
            if not reader.exists(path_event(scope, date, 'cycle')):
                inputs[scope]['skipped'][date] = 'no cycle'
                continue

            if not reader.exists(path_event(scope, date, 'results')):
                inputs[scope]['skipped'][date] = 'no results'
                continue

            cycle = reader.read_json(path_event(scope, date, 'cycle'))['data']
            polls, series = prepare_polls(cycle['polls'], list(cycle['parties']), date, params)
            elections.append(Election(scope=scope, event_date=date, label=_label(scope, date, dates, names),
                                      kind=_kind(scope), names=series, polls=polls, current=False, as_of=None))
            inputs[scope]['read'].append(date)

    order = {scope: index for index, scope in enumerate(scopes)}
    elections.sort(key=lambda election: (election.event_date, order[election.scope]))

    return elections, inputs


def read_current(reader: BundleReader, params: dict, names: dict) -> tuple[Optional[Election], dict]:
    """
    The election in progress: the latest run of the live scope while its election has no published results.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``live_scope``, ``since`` and what ``prepare_polls`` uses.
    names : dict
        ``{scope: name}`` of the scopes catalogue, for the labels.

    Returns
    -------
    current : Election or None
        The election with ``current`` true, the polls of the run ``polls`` part and the ``as_of`` of its ``meta``;
        ``None`` without a manifest, without a run of the scope or once its election has ``results``.
    inputs : dict
        ``{'run_id', 'as_of', 'event_date'}`` of the run read, or ``{'skipped': reason}`` with reason
        ``'no manifest'``, ``'no run'`` or ``'election published'``.
    """
    scope = params['live_scope']

    if not reader.exists(path_manifest()):
        return None, {'skipped': 'no manifest'}

    run_id = reader.read_json(path_manifest())['data']['scopes'].get(scope, {}).get('latest')

    if not run_id:
        return None, {'skipped': 'no run'}

    meta = reader.read_json(path_part(scope, run_id, 'meta'))['data']
    event_date = meta['event_date']

    if reader.exists(path_event(scope, event_date, 'results')):
        return None, {'skipped': 'election published'}

    polls = reader.read_json(path_part(scope, run_id, 'polls'))['data']
    prepared, series = prepare_polls(polls['polls'], list(polls['parties']), event_date, params)
    dates = [date for date in reader.list_events(scope) if params.get('since') is None or date >= params['since']]
    current = Election(scope=scope, event_date=event_date, label=_label(scope, event_date, dates, names),
                       kind=_kind(scope), names=series, polls=prepared, current=True, as_of=meta['as_of'])

    return current, {'run_id': run_id, 'as_of': meta['as_of'], 'event_date': event_date}
