"""
The lean study in one call: `compute_lean` reads the published parts through the bundle reader and chains the
computations of `stats` for the period with the pollster's lean, the control period, the ranking of houses, the
matrix of elections and the detail of one election. `LeanStudy` carries the results; this module only reads and
wires, every figure comes from `data` and `stats`.
"""
from dataclasses import dataclass
from typing import Optional

from ...bundle import BundleReader, path_event
from ....core.io import FileSystem
from .data import NATIONAL, MissingInput, active_pollsters, load_params, read_election, read_error_rows
from .stats import election_table, house_table, matrix_table, summarize_period


@dataclass
class LeanStudy:
    """
    Results of the lean study.

    Attributes
    ----------
    params : dict
        Parameters of the study, ``data/sesgo-cis/params.json`` updated with the overrides.
    periods : dict
        ``{'with': {...}, 'without': {...}}``, each ``from``, ``to``, ``label``, ``first_event``, ``last_event``
        (oldest and newest election of the pollster in the period, ``None`` without rows) and ``n``.
    rows : list of dict
        Rows of ``read_error_rows`` of every house, both periods.
    summary : dict
        ``with`` and ``without`` (``summarize_period`` of the pollster, rows included) and ``election``
        (``cis_dist``, ``consensus_dist``, ``closer``, ``rank``, ``n``).
    elections : list of dict
        Rows of the pollster in both periods with ``consensus``, ``n_others``, ``others_left`` and ``period``,
        newest first.
    houses : list of dict
        Output of ``house_table`` over the period with the lean.
    matrix : dict
        Output of ``matrix_table``.
    election : dict
        Output of ``election_table`` plus ``scope``, ``date`` and ``name``.
    inputs : dict
        ``scopes`` (inputs of ``read_error_rows``), ``election`` (``generated_at`` of its polls), ``active_source``
        and ``run_id`` (national run of the active houses, ``None`` without one).
    """

    params: dict
    periods: dict
    rows: list
    summary: dict
    elections: list
    houses: list
    matrix: dict
    election: dict
    inputs: dict


def _period(rows: list, start: str, end: Optional[str], label: str) -> dict:
    """Description of a period from the pollster's rows inside it."""
    dates = sorted(row['event_date'] for row in rows)

    return {'from': start, 'to': end, 'label': label, 'first_event': dates[0] if dates else None,
            'last_event': dates[-1] if dates else None, 'n': len(dates)}


def compute_lean(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> LeanStudy:
    """
    Compute the lean study from the published bundle.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    data : FileSystem
        The ``data/`` directory of the repository (parameters and scopes catalogue).
    params : dict, optional
        Overrides of the parameters file.

    Returns
    -------
    LeanStudy
        The results.

    Raises
    ------
    MissingInput
        When the national scope has no analysis or the parts of the election are not published.
    """
    params = load_params(data) | (params or {})
    pollster = params['pollster']
    catalogue = data.read_csv('es-scopes.csv').set_index('scode')

    rows, scopes = read_error_rows(reader, catalogue)

    if 'skipped' in scopes.get(NATIONAL, {'skipped': True}):
        raise MissingInput('analysis/es')

    since, (control_from, control_to) = params['since'], params['control']
    with_rows = [row for row in rows if row['event_date'] >= since]
    without_rows = [row for row in rows if control_from <= row['event_date'] < control_to]

    summary = {'with': summarize_period(with_rows, pollster), 'without': summarize_period(without_rows, pollster)}
    own = {'with': [row for row in with_rows if row['pollster'] == pollster],
           'without': [row for row in without_rows if row['pollster'] == pollster]}
    periods = {'with': _period(own['with'], since, None, params['labels']['with']),
               'without': _period(own['without'], control_from, control_to, params['labels']['without'])}

    elections = []
    for period in ('with', 'without'):
        consensus = {(row['scope'], row['event_date']): row for row in summary[period]['rows']}
        for row in own[period]:
            other = consensus.get((row['scope'], row['event_date']), {})
            elections.append(row | {'consensus': other.get('consensus'), 'n_others': other.get('n_others', 0),
                                    'others_left': other.get('others_left', 0), 'period': period})
    elections.sort(key=lambda row: row['event_date'], reverse=True)

    active, active_source, run_id = active_pollsters(reader, rows, params['active_from'])
    houses = house_table(with_rows, pollster, params['min_elections'], active)
    matrix = matrix_table(with_rows, houses)

    scope, date = params['election']['scope'], params['election']['date']
    parts = read_election(reader, scope, date)
    election = election_table(parts, pollster, params['max_days'], params['blocks']) | {
        'scope': scope, 'date': date, 'name': parts['meta'].get('name')}
    polls = reader.read_json(path_event(scope, date, 'polls'))

    own_dist = (election['cis'] or {}).get('dist')
    ranking = [house['pollster'] for house in election['houses']]
    summary['election'] = {
        'cis_dist': own_dist,
        'consensus_dist': (election['consensus'] or {}).get('dist'),
        'closer': sum(1 for house in election['houses'] if house['pollster'] != pollster and own_dist is not None
                      and house['dist'] < own_dist),
        'rank': ranking.index(pollster) + 1 if pollster in ranking else None,
        'n': len(ranking),
    }

    inputs = {'scopes': scopes, 'election': {'generated_at': polls.get('generated_at')},
              'active_source': active_source, 'run_id': run_id}

    return LeanStudy(params=params, periods=periods, rows=rows, summary=summary, elections=elections, houses=houses,
                     matrix=matrix, election=election, inputs=inputs)
