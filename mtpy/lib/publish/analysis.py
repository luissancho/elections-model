"""Exporters and ``publish_analysis`` for the houses analysis of a scope."""
import time
from typing import Callable, Optional, Sequence

import pandas as pd

from .. import bundle
from ..bundle import BundleWriter
from ..computer import Computer
from .common import db_stats, NothingToPublish, provenance, REPO_ROOT, round_cols, _iso_dates


ANALYSIS_HE_COLUMNS = ['pollster', 'name', 'n_events', 'hist', 'trend', 'cycle', 'n_cycle']


ANALYSIS_HERDING_COLUMNS = ['pollster', 'herding', 'herding_last', 'herding_p', 'herding_n', 'herding_events']


ANALYSIS_EVENTS_COLUMNS = ['event_date', 'pollster', 'n', 'ratio', 'p_value']


POLL_ERRORS_COLUMNS = ['event_date', 'event', 'pollster', 'days', 'sample_size', 'weight', 'error_avg', 'bias_avg',
                       'error_blocks', 'bias_blocks', 'bias']


ERROR_DATA_COLUMNS = ['date', 'pollster', 'party', 'regional', 'weeks', 'weight', 'pct', 'error']


POLL_ERRORS_PARAMS = {'bmap': 'vs', 'drange': (6, 42), 'n_last': 1}


def export_analysis_house_effects(table: pd.DataFrame, parties: Sequence[str], blocks: Sequence[str]) -> tuple:
    """Export the house effects of each pollster by party and block (``Computer.print_house_effects``).

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by ``(pollster, name)`` with ``n_events``, ``hist``, ``trend``, ``cycle`` and ``n_cycle``.
    parties : sequence of str
        Parties of the table (the ``main`` parties of the event); those absent from it are left out.
    blocks : sequence of str
        Names of the blocks of the table (the ``vs`` blocks of the event).

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``analysis-house-effects`` schema and its CSV twin. Effects are rounded to
        ``PCT_DECIMALS``; ``n_events`` is an integer and ``n_cycle`` an integer or null.
    """
    frame = table.reset_index().reindex(columns=ANALYSIS_HE_COLUMNS)
    frame = round_cols(frame, pct=('hist', 'trend', 'cycle'))
    frame['n_events'] = frame['n_events'].fillna(0).astype(int)
    frame['n_cycle'] = frame['n_cycle'].astype('Int64')
    names = set(frame['name'])
    data = {'parties': [p for p in parties if p in names], 'blocks': list(blocks),
            'rows': frame.to_dict(orient='records')}
    return data, frame


def export_analysis_herding(summary: pd.DataFrame, events: pd.DataFrame) -> tuple:
    """Export the herding of each pollster (``Computer.herding_summary``) and of each cycle (``get_herding``).

    Parameters
    ----------
    summary : pandas.DataFrame
        Indexed by ``pollster`` with ``herding``, ``herding_last``, ``herding_p``, ``herding_n`` and
        ``herding_events``.
    events : pandas.DataFrame
        Rows of the herding table: ``event_date``, ``pollster``, ``n``, ``ratio`` and ``p_value`` at least.

    Returns
    -------
    tuple of (dict, pandas.DataFrame, pandas.DataFrame)
        Data for the ``analysis-herding`` schema and the CSV twins of its ``rows`` and its ``events``
        (sorted by ``event_date`` and ``pollster``). Ratios and p-values are rounded to
        ``PROB_DECIMALS``, counts are integers and dates ISO.
    """
    rows = summary.rename_axis('pollster').reset_index().reindex(columns=ANALYSIS_HERDING_COLUMNS)
    rows = round_cols(rows, prob=('herding', 'herding_last', 'herding_p'))
    rows[['herding_n', 'herding_events']] = rows[['herding_n', 'herding_events']].astype('Int64')
    cycles = events.reindex(columns=ANALYSIS_EVENTS_COLUMNS).copy()
    cycles['event_date'] = _iso_dates(cycles['event_date']).to_numpy()
    cycles['pollster'] = cycles['pollster'].astype(str)
    cycles = round_cols(cycles, prob=('ratio', 'p_value'))
    cycles['n'] = cycles['n'].astype('Int64')
    cycles = cycles.sort_values(['event_date', 'pollster'], ignore_index=True)
    data = {'rows': rows.to_dict(orient='records'), 'events': cycles.to_dict(orient='records')}
    return data, rows, cycles


def export_poll_errors(frame: pd.DataFrame, blocks: Sequence[str]) -> tuple:
    """Export the errors of the polls in past elections (``Computer.get_polls_metric('error', bmap='vs')``).

    Parameters
    ----------
    frame : pandas.DataFrame
        Indexed by ``(event, pollster, days)``, with ``event_date`` (ISO date of ``event``),
        ``sample_size``, ``weight``, the error and bias columns and one column per block.
    blocks : sequence of str
        Names of the blocks; those absent from the frame are left out.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``poll-errors`` schema and its CSV twin. ``weight`` is rounded to ``PROB_DECIMALS``,
        errors, biases and blocks to ``PCT_DECIMALS``; ``days`` is an integer and ``sample_size`` an
        integer or null (the frame gives ``0`` for an unknown sample).
    """
    present = [b for b in blocks if b in frame.columns]
    table = frame.reset_index().reindex(columns=POLL_ERRORS_COLUMNS + present)
    table['event_date'] = _iso_dates(table['event_date']).to_numpy()
    table = round_cols(table, pct=['error_avg', 'bias_avg', 'error_blocks', 'bias_blocks', 'bias'] + present,
                       prob=('weight',))
    table['days'] = table['days'].astype(int)
    table['sample_size'] = table['sample_size'].where(table['sample_size'] > 0).astype('Int64')
    return {'blocks': present, 'rows': table.to_dict(orient='records')}, table


def export_error_data(frame: pd.DataFrame) -> tuple:
    """Export the errors of the polls by party (``Computer.get_error_estimator_data``).

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows with ``date``, ``pollster``, ``party``, ``regional``, ``weeks``, ``weight``, ``pct`` and
        ``error`` (other columns, such as ``color``, are left out).

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``error-data`` schema and its CSV twin, in the order of the frame. ``weight`` is
        rounded to ``PROB_DECIMALS``, ``pct`` and ``error`` to ``PCT_DECIMALS``; dates are ISO.
    """
    table = frame.reindex(columns=ERROR_DATA_COLUMNS).copy()
    table['date'] = _iso_dates(table['date']).to_numpy()
    table = round_cols(table, pct=('pct', 'error'), prob=('weight',))
    table[['regional', 'weeks']] = table[['regional', 'weeks']].astype(int)
    return {'rows': table.to_dict(orient='records')}, table


def export_analysis_meta(scope: str, event_date: str, events: Sequence[str], n_pollsters: int, params: dict,
                         prov: dict, seconds: dict) -> dict:
    """Build the ``analysis-meta`` document describing how and when the analysis of a scope was produced.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str
        Date (``YYYY-MM-DD``) of the current event, whose cycle the analysis measures.
    events : sequence of str
        Dates of the elections of the scope the analysis is computed on.
    n_pollsters : int
        Number of pollsters in the analysis.
    params : dict
        Parameters of the analysis.
    prov : dict
        Result of ``provenance``.
    seconds : dict
        Seconds spent in each step.

    Returns
    -------
    dict
        Data for the ``analysis-meta`` schema.
    """
    return {'scope': scope, 'event_date': event_date, 'events': list(events), 'n_pollsters': int(n_pollsters),
            'params': params, 'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'],
            'seconds': seconds}


def publish_analysis(scope: str, writer: BundleWriter, computer: Optional[Callable] = None,
                     stats: Optional[Callable] = None, prov: Optional[Callable] = None, verbose: int = 0) -> dict:
    """Compute the houses analysis of a scope and write it into the bundle, overwriting the previous one.

    The parts are written in the order ``house-effects``, ``herding``, ``poll-errors`` and ``error-data``
    (each JSON with its CSV twins) and ``meta`` last, so a failure halfway leaves the previous ``meta.json``
    (and the analysis it describes, partly overwritten) in place. Neither the manifest nor the runs are
    touched.

    The current event is the last election of the computer (``event_dates[-1]``, the next event when the
    scope has one, as in ``Computer.print_house_effects``), and its ``vs`` blocks and ``main`` parties
    come from ``event_params`` of the computer (``get_event_params``).

    Parameters
    ----------
    scope : str
        Scope code.
    writer : BundleWriter
        Bundle destination.
    computer, stats, prov : callable, optional
        Injectable replacements of ``Computer``, ``db_stats`` and ``provenance``. ``stats`` records in
        ``meta.params`` the polls of the current event in the database and the date of the latest one.
    verbose : int
        Computer verbosity.

    Returns
    -------
    dict
        ``{'scope', 'event_date', 'n_pollsters', 'seconds'}``.

    Raises
    ------
    NothingToPublish
        Building the computer or its series, or ``print_house_effects`` (whose average of the current
        event needs its polls), raised ``ValueError``; nothing is written then. Any later error,
        ``ValueError`` included, propagates as is.
    """
    start = mark = time.perf_counter()

    def lap():
        """Seconds since the previous lap (or the start); restart the lap."""
        nonlocal mark
        now = time.perf_counter()
        elapsed, mark = now - mark, now
        return elapsed

    try:
        comp = (computer or Computer)(scope=scope, path='.', verbose=verbose).build_series()
        seconds = {'build': lap()}
        event_date = comp.event_dates[-1]
        bmaps = comp.event_params[event_date]['bmaps']
        blocks, main = bmaps['vs'], list(bmaps.get('main', []))
        table = comp.print_house_effects()  # fits an average of the current event: no polls is a ValueError
    except ValueError as e:
        raise NothingToPublish(str(e)) from e

    def write(part, data, frames):
        """Write the JSON of a part and its CSV twins (``{csv name: frame}``)."""
        writer.write_json(bundle.path_analysis(scope, part), bundle.ANALYSIS_SCHEMAS[part], data, scope)
        for name, frame in frames.items():
            writer.write_csv(bundle.path_analysis_csv(scope, name), frame)

    effects, frame = export_analysis_house_effects(table, main, list(blocks))
    write('house-effects', effects, {'house-effects': frame})
    seconds['house_effects'] = lap()
    summary = comp.herding_summary()
    herding = comp.herding if comp.herding is not None else comp.load_herding()
    data, rows, cycles = export_analysis_herding(summary, herding)
    write('herding', data, {'herding': rows, 'herding-events': cycles})
    seconds['herding'] = lap()
    data, frame = export_poll_errors(comp.get_polls_metric('error', **POLL_ERRORS_PARAMS), list(blocks))
    write('poll-errors', data, {'poll-errors': frame})
    seconds['poll_errors'] = lap()
    data, frame = export_error_data(comp.get_error_estimator_data())
    write('error-data', data, {'error-data': frame})
    seconds['error_data'] = lap()

    db_polls, db_last_poll = (stats or db_stats)(scope, event_date)
    params = {**POLL_ERRORS_PARAMS, 'drange': list(POLL_ERRORS_PARAMS['drange']), 'main': main,
              'blocks': dict(blocks), 'db_polls': db_polls, 'db_last_poll': db_last_poll}
    n_pollsters = len({row['pollster'] for row in effects['rows']} | set(summary.index.astype(str)))
    seconds['total'] = time.perf_counter() - start
    seconds = {step: round(value, 1) for step, value in seconds.items()}
    prov = prov or (lambda: provenance(cwd=REPO_ROOT))
    meta = export_analysis_meta(scope, event_date, comp.event_dates, n_pollsters, params, prov(), seconds)
    write('meta', meta, {})
    return {'scope': scope, 'event_date': event_date, 'n_pollsters': n_pollsters, 'seconds': seconds}
