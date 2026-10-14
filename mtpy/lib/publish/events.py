"""Exporters and ``publish_event`` for one past election of a scope."""
import os
import time
from datetime import date, datetime
from typing import Callable, Optional, Sequence

import pandas as pd

from .. import bundle
from ..bundle import BundleWriter
from ..computer import Computer
from ..data import get_event_results, get_events, get_pollsters
from ..forecaster import published_seats
from .analysis import export_poll_errors
from .backtest import BACKTEST_FILES, backtest_source, export_backtest_table
from .common import NothingToPublish, provenance, REPO_ROOT, round_cols, _iso_dates, _iso_day, _today, _weighted_mean
from .forecast import export_poll_rows, POLL_COLUMNS


# ---------------------------------------------------------------------------------------------------------------
# Events: one past election of a scope (poll errors by house, official results and the model backtest of the event)
# ---------------------------------------------------------------------------------------------------------------

EVENT_POLLSTER_COLUMNS = ['pollster_id', 'pollster', 'n_polls', 'sample_size', 'error_avg', 'error_blocks',
                          'bias_blocks', 'last_days', 'last_error_avg', 'last_error_blocks', 'last_bias_blocks']


EVENT_MODEL_PARTS = ('shares', 'seats', 'blocks')


def _records(table: pd.DataFrame) -> list:
    """Rows of a table as dicts, with ``None`` for missing values."""
    return [{key: None if value is pd.NA else value for key, value in row.items()}
            for row in table.to_dict(orient='records')]


def _present_blocks(frame: pd.DataFrame, blocks: Sequence[str]) -> list:
    """The blocks that are columns of a table of poll errors, in the order of ``blocks``."""
    return [b for b in blocks if b in frame.columns]


def export_event_polls(frame: pd.DataFrame, blocks: Sequence[str], ids: dict) -> tuple:
    """Export the polls of a past election with their errors, the most recent first.

    Parameters
    ----------
    frame : pandas.DataFrame
        Result of ``Computer.get_polls_metric('error', bmap='vs')`` (see ``export_poll_errors``).
    blocks : sequence of str
        Names of the blocks.
    ids : dict
        Pollster id by pollster name.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``event-polls`` schema and its CSV twin: the rows of ``export_poll_errors`` plus
        ``pollster_id`` (integer, or null for a pollster missing from ``ids``), ascending by ``days``.
    """
    data, table = export_poll_errors(frame, blocks)
    table = table.copy()
    table['pollster'] = table['pollster'].astype(str)
    table.insert(table.columns.get_loc('pollster') + 1, 'pollster_id', table['pollster'].map(ids).astype('Int64'))
    table = table.sort_values('days', kind='stable', ignore_index=True)

    return {'blocks': data['blocks'], 'rows': _records(table)}, table


def export_event_cycle(frame: pd.DataFrame, names: Sequence[str], seats: Optional[dict] = None) -> tuple:
    """Export the polls of the cycle of a past election as they were published, the oldest first.

    Parameters
    ----------
    frame : pandas.DataFrame
        Polls of the election, as ``Computer.filter_polls(featured=True)`` gives them (indexed by
        ``event_date``, ``date``, ``pollster_id`` and ``sponsor_id``, with ``pollster``, ``sponsor``,
        ``start_date``, ``end_date``, ``sample_size``, ``mtype``, ``rating``, ``weight_sample``,
        ``weight_over`` and one share per party).
    names : sequence of str
        Party names to export, in column order; one missing from the frame is skipped.
    seats : dict, optional
        Seat projections by ``(date, pollster_id, sponsor_id)`` (``published_seats``); none by default.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``event-cycle`` schema (``parties``, ``columns`` and ``polls``, the same shape as the
        ``polls`` part of a run so the pages draw both alike) and its CSV twin: ascending by date and pollster,
        an unknown ``sample_size`` (zero) null and ``weight`` the product of the overall and sample weights.
    """
    table = frame.reset_index()
    names = [name for name in names if name in table.columns]
    for column in ('pollster', 'sponsor'):
        if column in table.columns:
            table[column] = table[column].astype(object).where(table[column].notna(), None)
    if 'sample_size' in table.columns:
        table['sample_size'] = table['sample_size'].where(table['sample_size'] > 0).astype('Int64')
    if 'weight' not in table.columns and {'weight_over', 'weight_sample'} <= set(table.columns):
        table['weight'] = table['weight_over'].astype(float) * table['weight_sample'].astype(float)
    table = table.sort_values(['date', 'pollster_id'], kind='stable', ignore_index=True)
    records, polls = export_poll_rows(table, names, seats)

    return {'parties': list(names), 'columns': list(POLL_COLUMNS), 'polls': records}, polls


def export_event_pollsters(frame: pd.DataFrame, blocks: Sequence[str], ids: dict) -> tuple:
    """Export the accuracy of each pollster in a past election.

    Parameters
    ----------
    frame : pandas.DataFrame
        Result of ``Computer.get_polls_metric('error', bmap='vs')`` (see ``export_poll_errors``).
    blocks : sequence of str
        Names of the blocks.
    ids : dict
        Pollster id by pollster name.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``event-pollsters`` schema and its CSV twin: one row per pollster with ``n_polls``,
        the median ``sample_size`` (null when unknown), the mean ``error_avg``, ``error_blocks`` and
        ``bias_blocks`` and the ``last_*`` values (days, errors and signed deviation of each block) of its
        poll with the fewest ``days``. Ascending by ``error_blocks``, nulls last.
    """
    present = _present_blocks(frame, blocks)
    table = frame.reset_index()
    table['pollster'] = table['pollster'].astype(str)  # the index level is categorical in the real frame
    table['sample_size'] = table['sample_size'].where(table['sample_size'] > 0)
    summary = table.groupby('pollster', sort=False, observed=True).agg(
        n_polls=('days', 'size'), sample_size=('sample_size', 'median'), error_avg=('error_avg', 'mean'),
        error_blocks=('error_blocks', 'mean'), bias_blocks=('bias_blocks', 'mean'))
    cols = ['days', 'error_avg', 'error_blocks', 'bias_blocks'] + present
    last = table.sort_values('days', kind='stable').drop_duplicates('pollster').set_index('pollster')[cols]
    last.columns = ['last_' + col for col in cols]
    out = summary.join(last).reset_index()
    out['pollster_id'] = out['pollster'].map(ids).astype('Int64')
    out['sample_size'] = out['sample_size'].round().astype('Int64')
    out['last_days'] = out['last_days'].astype(int)
    out = out.sort_values('error_blocks', na_position='last', kind='stable', ignore_index=True)
    out = out.reindex(columns=EVENT_POLLSTER_COLUMNS + ['last_' + b for b in present])
    out = round_cols(out, pct=[col for col in out.columns if col.endswith(('error_avg', 'error_blocks', 'bias_blocks'))
                               or col.startswith('last_') and col != 'last_days'])

    return {'blocks': present, 'rows': _records(out)}, out


def export_event_results(frame: pd.DataFrame) -> tuple:
    """Export the official national results of a past election.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows of ``data.get_event_results`` with ``party_id``, ``party``, ``votes``, ``pct`` and ``seats``.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``event-results`` schema and its CSV twin: rows descending by ``votes``, with integer
        ``votes`` and ``seats`` and ``pct`` rounded to ``PCT_DECIMALS``; ``n_seats`` is the sum of ``seats``.
    """
    table = frame.reindex(columns=['party_id', 'party', 'votes', 'pct', 'seats']).copy()
    table[['party_id', 'votes', 'seats']] = table[['party_id', 'votes', 'seats']].astype(int)
    table = round_cols(table, pct=('pct',)).sort_values('votes', ascending=False, kind='stable', ignore_index=True)

    return {'n_seats': int(table['seats'].sum()), 'rows': _records(table)}, table


def export_event_model(source: Optional[str], event_date: str) -> tuple:
    """Slice the model backtest (``shares``, ``seats`` and ``blocks``) to one election.

    Parameters
    ----------
    source : str, optional
        Folder of the backtest tables (see ``backtest_source``). A table whose file is missing, or a missing
        folder, gives an empty slice.
    event_date : str
        Election date, ``YYYY-MM-DD``.

    Returns
    -------
    tuple of (dict, dict of pandas.DataFrame)
        Data for the ``event-model`` schema (``horizons``, ascending, and the rows of each slice) and the CSV
        twins ``model-shares``, ``model-seats`` and ``model-blocks`` of the tables found. Rounded as
        ``export_backtest_table``.
    """
    data = {'horizons': [], **{part: [] for part in EVENT_MODEL_PARTS}}
    frames = {}
    horizons = set()
    for part in EVENT_MODEL_PARTS:
        path = os.path.join(source, BACKTEST_FILES[part]) if source else None
        if path is None or not os.path.isfile(path):
            continue
        table = pd.read_csv(path)
        table = table.loc[table['event_date'].astype(str) == event_date].reset_index(drop=True)
        slice_, frame = export_backtest_table(table)
        data[part] = slice_['rows']
        frames['model-' + part] = frame
        horizons.update(int(h) for h in table['horizon'].unique())
    data['horizons'] = sorted(horizons)

    return data, frames


def event_summary(frame: pd.DataFrame, blocks: Sequence[str]) -> dict:
    """Summarise the polls of a past election.

    Parameters
    ----------
    frame : pandas.DataFrame
        Result of ``Computer.get_polls_metric('error', bmap='vs')`` (see ``export_poll_errors``).
    blocks : sequence of str
        Names of the blocks.

    Returns
    -------
    dict
        ``n_polls``, ``n_pollsters`` and the means weighted by ``weight`` of ``error_avg``, ``error_blocks`` and
        ``bias_blocks`` (``PCT_DECIMALS``), plus ``bias``: ``{block: weighted mean of its signed deviation}``.
    """
    table = frame.reset_index()
    table['pollster'] = table['pollster'].astype(str)
    weights = table['weight']
    summary = {'n_polls': int(len(table)), 'n_pollsters': int(table['pollster'].nunique())}
    for col in ('error_avg', 'error_blocks', 'bias_blocks'):
        summary[col] = _weighted_mean(table[col], weights)
    summary['bias'] = {b: _weighted_mean(table[b], weights) for b in _present_blocks(table, blocks)}

    return summary


def export_event_meta(scope: str, event_date: str, name: Optional[str], blocks: dict, main: Sequence[str],
                      summary: dict, model: dict, prov: dict, seconds: dict) -> dict:
    """Build the ``event-meta`` document describing a published past election.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    name : str, optional
        Name of the election.
    blocks : dict
        Parties of each ``vs`` block of the election.
    main : sequence of str
        Main parties of the election.
    summary : dict
        Result of ``event_summary``.
    model : dict
        Data of ``export_event_model``; ``has_model`` is whether it has ``horizons`` (so it agrees with them).
    prov : dict
        Result of ``provenance``.
    seconds : dict
        Seconds spent in each step.

    Returns
    -------
    dict
        Data for the ``event-meta`` schema.
    """
    return {'scope': scope, 'event_date': event_date, 'name': name, 'n_polls': summary['n_polls'],
            'n_pollsters': summary['n_pollsters'], 'blocks': {k: list(v) for k, v in blocks.items()},
            'main': list(main), 'summary': summary, 'has_model': bool(model['horizons']),
            'horizons': list(model['horizons']), 'commit': prov['commit'], 'dirty': bool(prov['dirty']),
            'versions': prov['versions'], 'seconds': seconds}


def event_dates_to_publish(scope: str, event_date, events: Optional[Callable] = None,
                           today: Optional[str | date | datetime] = None) -> list:
    """Resolve the ``event_date`` argument of ``publish event`` into election dates.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str or list of str
        ``'all'`` (the featured elections of the scope before today), one date or a list of dates.
    events : callable, optional
        Injectable replacement of ``data.get_events``.
    today : str, datetime.date or datetime.datetime, optional
        Date taken as today (see ``_today``).

    Returns
    -------
    list of str
        ISO dates: ascending for ``'all'``, in the given order otherwise.

    Raises
    ------
    ValueError
        A date is not a real ISO date.
    """
    if event_date == 'all':
        table = (events or get_events)(scope)
        days = _iso_dates(table['date'])
        past = table['featured'].astype(bool).to_numpy() & (days < _today(today).isoformat()).to_numpy()

        return sorted(days[past].unique())

    return [_iso_day(day) for day in ([event_date] if isinstance(event_date, str) else list(event_date))]


def publish_event(scope: str, event_date: str, writer: BundleWriter, computer: Optional[Callable] = None,
                  events: Optional[Callable] = None, results: Optional[Callable] = None,
                  pollsters: Optional[Callable] = None, seats: Optional[Callable] = None,
                  source: Optional[str] = None, prov: Optional[Callable] = None, verbose: int = 0) -> dict:
    """Write one past election of a scope into the bundle, overwriting the previous publication.

    The parts are written in the order ``pollsters``, ``polls``, ``cycle``, ``results`` and ``model`` (each JSON
    with its CSV twins) and ``meta`` last, so a failure halfway leaves the previous ``meta.json`` in place. The
    polls errors come from a ``Computer`` of the election alone (``get_polls_metric('error', bmap='vs')``), the
    polls of the cycle as published from the same polls (``filter_polls(featured=True)``, so every poll of the
    table has its error) with the seat projections of the loader, the results from the database and the model
    from the backtest tables (see ``export_event_model``). The manifest and the runs are not touched.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    writer : BundleWriter
        Bundle destination.
    computer : callable, optional
        Injectable replacement of ``Computer``.
    events, results, pollsters : callable, optional
        Injectable replacements of ``data.get_events(scope)``, ``data.get_event_results(scope, date, region_id)``
        and ``data.get_pollsters()``.
    seats : callable, optional
        Injectable replacement of ``forecaster.published_seats(scope, event_date, names)``.
    source : str, optional
        Results folder of the backtest (see ``backtest_source``).
    prov : callable, optional
        Injectable replacement of ``provenance``.
    verbose : int
        Computer verbosity.

    Returns
    -------
    dict
        ``{'scope', 'event_date', 'n_polls', 'n_pollsters', 'has_model', 'seconds'}``.

    Raises
    ------
    NothingToPublish
        The date is not a featured election of the scope, it has no official results, or the ``Computer`` raised
        ``ValueError`` or gave no polls; nothing is written then.
    ValueError
        The date is not a real ISO date.
    """
    start = mark = time.perf_counter()

    def lap():
        """Seconds since the previous lap (or the start); restart the lap."""
        nonlocal mark
        now = time.perf_counter()
        elapsed, mark = now - mark, now
        return elapsed

    event_date = _iso_day(event_date)
    table = (events or get_events)(scope)
    row = table.loc[(_iso_dates(table['date']) == event_date).to_numpy() & table['featured'].astype(bool).to_numpy()]
    if row.empty:
        raise NothingToPublish(f'{event_date} is not a featured election of {scope}')
    name = row['name'].iloc[0]
    name = None if pd.isna(name) else str(name)
    official = (results or get_event_results)(scope, event_date, 0)
    if official.empty:
        raise NothingToPublish(f'no official results for {scope} {event_date}')

    try:
        comp = (computer or Computer)(scope=scope, event_dates=[event_date], path='.', verbose=verbose).build_series()
        bmaps = comp.event_params[event_date]['bmaps']
        blocks, main = bmaps['vs'], list(bmaps.get('main', []))
        errors = comp.get_polls_metric('error', bmap='vs')
        cycle = comp.filter_polls(featured=True)
    except ValueError as e:
        raise NothingToPublish(str(e)) from e
    if errors.empty:
        raise NothingToPublish(f'no polls for {scope} {event_date}')
    seconds = {'build': lap()}

    houses = (pollsters or get_pollsters)()
    ids = {str(pname): int(pid) for pid, pname in zip(houses['id'], houses['name'])}

    def write(part, data, frames):
        """Write the JSON of a part and its CSV twins (``{csv name: frame}``)."""
        writer.write_json(bundle.path_event(scope, event_date, part), bundle.EVENT_SCHEMAS[part], data, scope)
        for csv_name, frame in frames.items():
            writer.write_csv(bundle.path_event_csv(scope, event_date, csv_name), frame)

    data, frame = export_event_pollsters(errors, list(blocks), ids)
    write('pollsters', data, {'pollsters': frame})
    data, frame = export_event_polls(errors, list(blocks), ids)
    write('polls', data, {'polls': frame})
    names = [name for name in comp.names if name in cycle.columns]
    data, frame = export_event_cycle(cycle, names, (seats or published_seats)(scope, event_date, names))
    write('cycle', data, {'cycle': frame})
    seconds['polls'] = lap()
    data, frame = export_event_results(official)
    write('results', data, {'results': frame})
    seconds['results'] = lap()
    model, frames = export_event_model(backtest_source(scope, source), event_date)
    write('model', model, frames)
    for csv_name in bundle.EVENT_CSV:  # twins of tables this publication lacks would describe an older one
        stale = bundle.path_event_csv(scope, event_date, csv_name)
        if csv_name.startswith('model-') and csv_name not in frames and writer.exists(stale):
            writer.remove(stale)
    seconds['model'] = lap()

    summary = event_summary(errors, list(blocks))
    seconds['total'] = time.perf_counter() - start
    seconds = {step: round(value, 1) for step, value in seconds.items()}
    prov = prov or (lambda: provenance(cwd=REPO_ROOT))
    meta = export_event_meta(scope, event_date, name, blocks, main, summary, model, prov(), seconds)
    write('meta', meta, {})

    return {'scope': scope, 'event_date': event_date, 'n_polls': summary['n_polls'],
            'n_pollsters': summary['n_pollsters'], 'has_model': meta['has_model'], 'seconds': seconds}
