"""
Readers of the inputs of the lean study: the parts already published in the bundle.

The study does not touch the database. It reads the ``poll-errors`` analysis of every scope, the parts of one
election (``meta``, ``polls`` and ``results``) and the polls of the latest national run, all through a
``BundleReader``, and the small parameters file in ``data/``. This module keeps independent of the web layer.
"""
import json

import pandas as pd

from ...bundle import BundleReader, path_analysis, path_event, path_manifest, path_part
from ....core.io import FileSystem

PARAMS_FILE = 'sesgo-cis/params.json'
NATIONAL = 'es'


class MissingInput(Exception):
    """
    A part the study needs is not published.

    Parameters
    ----------
    part : str
        Name of the missing part, relative to the bundle (``events/es/2023-07-23/polls``).

    Attributes
    ----------
    part : str
        The same name.
    """

    def __init__(self, part: str):
        super().__init__(f'missing input: {part}')
        self.part = part


def load_params(data: FileSystem) -> dict:
    """
    Parameters of the study (``data/sesgo-cis/params.json``).

    Parameters
    ----------
    data : FileSystem
        The ``data/`` directory of the repository.

    Returns
    -------
    dict
        The JSON as written: ``pollster``, ``since``, ``control``, ``blocks``, ``min_elections``, ``max_days``,
        ``active_from``, ``election`` and ``labels``.
    """
    return json.loads(data.read(PARAMS_FILE))


def _is_number(value) -> bool:
    """Whether ``value`` is a real number (``None``, booleans and NaN are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _short(scope: str, event_date: str) -> str:
    """Short label of an election: ``GEN 23`` for the national scope, ``AN 18`` for ``es-an`` 2018."""
    code = 'GEN' if scope == NATIONAL else scope.upper().removeprefix('ES-')

    return f'{code} {event_date[2:4]}'


def read_error_rows(reader: BundleReader, catalogue: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Rows of the ``poll-errors`` analysis of every scope, restricted to the published elections.

    The analysis has every election of the scope with results and polls, the site only the published ones (those
    with an event folder), so the study speaks of the same elections as the pages.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    catalogue : pandas.DataFrame
        Scopes catalogue indexed by ``scode`` with a ``name`` column.

    Returns
    -------
    rows : list of dict
        The fields of the ``poll-errors`` rows plus ``scope``, ``scope_name``, ``label``, ``short``, ``year``
        and ``lean`` (``Izquierda - Derecha`` rounded to 2 decimals; ``None`` when either is missing), in
        catalogue order then ``event_date``.
    inputs : dict
        ``{scope: {'run_id', 'generated_at'}}`` for the scopes read, ``{scope: {'skipped': reason}}`` for the
        rest.
    """
    rows = []
    inputs = {}

    for code, name in catalogue['name'].items():
        part = path_analysis(code, 'poll-errors')

        if not (reader.analysis_exists(code) and reader.exists(part)):
            inputs[code] = {'skipped': 'no analysis'}
            continue

        envelope = reader.read_json(part)
        inputs[code] = {'run_id': envelope.get('run_id'), 'generated_at': envelope.get('generated_at')}
        published = set(reader.list_events(code))

        for row in sorted(envelope['data'].get('rows') or [], key=lambda item: item['event_date']):
            event_date = row['event_date']

            if event_date not in published:
                continue

            left, right = row.get('Izquierda'), row.get('Derecha')
            year = int(event_date[:4])
            rows.append(row | {
                'Derecha': right,
                'scope': code,
                'scope_name': name,
                'label': f'Generales {year}' if code == NATIONAL else f'{name} {year}',
                'short': _short(code, event_date),
                'year': year,
                'lean': round(left - right, 2) if _is_number(left) and _is_number(right) else None,
            })

    return rows, inputs


def read_election(reader: BundleReader, scope: str, date: str) -> dict:
    """
    The three published parts of an election the study reads.

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
    dict
        ``{'meta', 'polls', 'results'}``, each the ``data`` of its part.

    Raises
    ------
    MissingInput
        When any of the parts is not published; ``part`` names it.
    """
    election = {}

    for part in ('meta', 'polls', 'results'):
        name = path_event(scope, date, part)

        if not reader.exists(name):
            raise MissingInput(name.removesuffix('.json'))

        election[part] = reader.read_json(name)['data']

    return election


def active_pollsters(reader: BundleReader, rows: list[dict], active_from: str) -> tuple[set[str], str]:
    """
    Pollsters still publishing: with a poll in the latest national run or an error row in a recent election.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    rows : list of dict
        Rows from ``read_error_rows``.
    active_from : str
        First date that counts as recent, ``YYYY-MM-DD``.

    Returns
    -------
    names : set of str
        The pollsters.
    source : str
        ``'run+events'`` when the national run was read, ``'events'`` when there is no manifest or no run.
    """
    names = {row['pollster'] for row in rows if row['event_date'] >= active_from}
    source = 'events'

    if reader.exists(path_manifest()):
        run_id = reader.read_json(path_manifest())['data']['scopes'].get(NATIONAL, {}).get('latest')
        part = path_part(NATIONAL, run_id, 'polls') if run_id else None

        if part and reader.exists(part):
            polls = reader.read_json(part)['data']['polls']
            names |= {poll['pollster'] for poll in polls if poll['date'] >= active_from}
            source = 'run+events'

    return names, source
