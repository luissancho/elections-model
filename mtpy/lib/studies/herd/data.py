"""
Readers of the inputs of the herd study: the parts already published in the bundle.

The study does not touch the database. It reads, for every published election of the scope, the ``cycle`` part (the
polls of the whole cycle) with the ``series`` (the curve of the average) and ``pollsters`` (the error of the last poll
of every house) parts when they exist; the ``meta``, ``polls`` and ``series`` of the latest run while its election has
no published results; and the ``herding`` analysis of the scope, all through a ``BundleReader``, plus the small
parameters file in ``data/``. This module keeps independent of the web layer.
"""
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...bundle import BundleReader, path_analysis, path_event, path_manifest, path_part
from ....core.io import FileSystem
from ..lean.data import MONTHS

PARAMS_FILE = 'rebano/params.json'
# The poll columns the study keeps, before one column per party
POLL_COLUMNS = ['date', 'pollster_id', 'pollster', 'start_date', 'end_date', 'sample_size', 'weight']


class MissingInput(Exception):
    """
    A part the study needs is not published.

    Parameters
    ----------
    part : str
        Name of the missing part, relative to the bundle (``events/es``).

    Attributes
    ----------
    part : str
        The same name.
    """

    def __init__(self, part: str):
        super().__init__(f'missing input: {part}')
        self.part = part


@dataclass
class Cycle:
    """
    The polls of one electoral cycle and what the study reads with them.

    Attributes
    ----------
    event_date : str
        Election date, ``YYYY-MM-DD``.
    label : str
        Public label of the cycle (``cycle_label``).
    current : bool
        Whether it is the cycle in progress (read from the latest run) rather than a past one.
    polls : pandas.DataFrame
        One row per poll with the ``POLL_COLUMNS`` (dates as the strings of the part, ``sample_size`` float with
        NaN for null) and one float column per party.
    parties : list of str
        The parties of the polls, in the order of the part.
    curve : pandas.DataFrame or None
        The average published with the polls, indexed by ``Timestamp`` with one column per party; ``None`` without
        a ``series`` part.
    errors : dict
        ``{pollster: {'last_error_avg', 'last_error_blocks'}}`` from the ``pollsters`` part; empty without it and in
        the current cycle.
    inputs : dict
        The parts read for the cycle (``read_cycles`` and ``read_current`` say which).
    """
    event_date: str
    label: str
    current: bool
    polls: pd.DataFrame
    parties: list
    curve: Optional[pd.DataFrame]
    errors: dict
    inputs: dict


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
        The JSON as written: ``scope``, ``since``, ``max_days``, ``window_days``, ``final_days``, ``local_window``,
        ``n_series``, ``min_polls``, ``exclude`` and ``buckets``.
    """
    return json.loads(data.read(PARAMS_FILE))


def cycle_label(event_date: str, dates: list) -> str:
    """
    Public label of a cycle: the year of its election, followed by the abbreviated month when another date of
    ``dates`` shares the year (``2019 abr.`` and ``2019 nov.``).

    Parameters
    ----------
    event_date : str
        Election date, ``YYYY-MM-DD``.
    dates : list of str
        The election dates labelled together.

    Returns
    -------
    str
        The label.
    """
    year = event_date[:4]

    if sum(date[:4] == year for date in set(dates) | {event_date}) > 1:
        return f'{year} {MONTHS[int(event_date[5:7]) - 1]}.'

    return year


def is_published(reader: BundleReader, scope: str, date: str) -> bool:
    """
    Whether an election is published with its results (its event folder has a ``results`` part).

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
        ``True`` if ``events/{scope}/{date}/results.json`` exists.
    """
    return reader.exists(path_event(scope, date, 'results'))


def _polls_frame(records: list, parties: list) -> pd.DataFrame:
    """The polls of a ``cycle`` or run ``polls`` part as a table: the ``POLL_COLUMNS`` and one float column per
    party (NaN for a missing figure), ``sample_size`` float with NaN for null."""
    frame = pd.DataFrame(records).reindex(columns=POLL_COLUMNS + list(parties))
    frame['sample_size'] = pd.to_numeric(frame['sample_size'], errors='coerce').astype(float)
    frame['weight'] = frame['weight'].astype(float)
    frame[list(parties)] = frame[list(parties)].astype(float)

    return frame


def _curve(series: dict) -> pd.DataFrame:
    """The ``mean`` of a ``series`` part, indexed by ``Timestamp`` with one float column per party."""
    index = pd.DatetimeIndex(pd.to_datetime(series['dates']), name='date')

    return pd.DataFrame({name: series['mean'][name] for name in series['parties']}, index=index).astype(float)


def read_cycles(reader: BundleReader, params: dict) -> tuple[list[Cycle], dict]:
    """
    The past cycles of the scope: every published election since ``params['since']`` with a ``cycle`` part.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``scope`` and ``since``.

    Returns
    -------
    cycles : list of Cycle
        The cycles ascending by date, with ``current`` false, ``curve`` from the ``series`` part and ``errors``
        from the ``pollsters`` part when they exist.
    inputs : dict
        By election date: ``{'cycle', 'series', 'pollsters'}`` with the ``generated_at`` of each part read
        (``None`` for ``series`` and ``pollsters`` when missing), or ``{'skipped': 'no cycle part'}``.
    """
    scope = params['scope']
    dates = [date for date in reader.list_events(scope) if date >= params['since']]
    cycles = []
    inputs = {}

    for date in dates:
        if not reader.exists(path_event(scope, date, 'cycle')):
            inputs[date] = {'skipped': 'no cycle part'}
            continue

        cycle = reader.read_json(path_event(scope, date, 'cycle'))
        series = pollsters = None

        if reader.exists(path_event(scope, date, 'series')):
            series = reader.read_json(path_event(scope, date, 'series'))

        if reader.exists(path_event(scope, date, 'pollsters')):
            pollsters = reader.read_json(path_event(scope, date, 'pollsters'))

        inputs[date] = {'cycle': cycle['generated_at'],
                        'series': series['generated_at'] if series else None,
                        'pollsters': pollsters['generated_at'] if pollsters else None}
        errors = {
            row['pollster']: {'last_error_avg': row.get('last_error_avg'),
                              'last_error_blocks': row.get('last_error_blocks')}
            for row in (pollsters['data']['rows'] if pollsters else [])
        }
        parties = list(cycle['data']['parties'])
        cycles.append(Cycle(event_date=date, label=cycle_label(date, dates), current=False,
                            polls=_polls_frame(cycle['data']['polls'], parties), parties=parties,
                            curve=_curve(series['data']) if series else None, errors=errors, inputs=inputs[date]))

    return cycles, inputs


def read_current(reader: BundleReader, params: dict) -> tuple[Optional[Cycle], dict]:
    """
    The cycle in progress: the latest run of the scope while its election has no published results.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    params : dict
        Parameters of the study (``load_params``); uses ``scope`` and ``since``.

    Returns
    -------
    current : Cycle or None
        The cycle with ``current`` true, the polls of the run ``polls`` part, ``curve`` from its ``series`` part
        (published cut at ``as_of``) and ``errors`` empty; ``None`` without a manifest, without a run of the scope
        or once its election is published with results.
    inputs : dict
        ``{'run_id', 'as_of', 'event_date'}`` of the run read, or ``{'skipped': reason}`` with reason
        ``'no manifest'``, ``'no run'`` or ``'election published'``.
    """
    scope = params['scope']

    if not reader.exists(path_manifest()):
        return None, {'skipped': 'no manifest'}

    run_id = reader.read_json(path_manifest())['data']['scopes'].get(scope, {}).get('latest')

    if not run_id:
        return None, {'skipped': 'no run'}

    meta = reader.read_json(path_part(scope, run_id, 'meta'))['data']
    event_date = meta['event_date']

    if is_published(reader, scope, event_date):
        return None, {'skipped': 'election published'}

    polls = reader.read_json(path_part(scope, run_id, 'polls'))['data']
    series = reader.read_json(path_part(scope, run_id, 'series'))['data']
    inputs = {'run_id': run_id, 'as_of': meta['as_of'], 'event_date': event_date}
    dates = [date for date in reader.list_events(scope) if date >= params['since']] + [event_date]
    parties = list(polls['parties'])
    current = Cycle(event_date=event_date, label=cycle_label(event_date, dates), current=True,
                    polls=_polls_frame(polls['polls'], parties), parties=parties, curve=_curve(series), errors={},
                    inputs=inputs)

    return current, inputs


def read_herding(reader: BundleReader, scope: str) -> tuple[dict, dict]:
    """
    The herding of every house over each whole cycle, from the ``herding`` analysis of the scope.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    scope : str
        Scope code.

    Returns
    -------
    rows : dict
        ``{(event_date, pollster): {'n', 'ratio', 'p_value'}}`` from the ``events`` of the part; empty without it.
    inputs : dict
        ``{'generated_at'}`` of the part, or ``{'skipped': 'no herding part'}``.
    """
    part = path_analysis(scope, 'herding')

    if not reader.exists(part):
        return {}, {'skipped': 'no herding part'}

    envelope = reader.read_json(part)
    rows = {(row['event_date'], row['pollster']): {'n': row['n'], 'ratio': row['ratio'], 'p_value': row['p_value']}
            for row in envelope['data']['events']}

    return rows, {'generated_at': envelope['generated_at']}
