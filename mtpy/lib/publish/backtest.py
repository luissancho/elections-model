"""Converter of the model backtest tables and ``publish_backtest``."""
import json
import os
import time
import types
from typing import Callable, Optional

import pandas as pd

from .. import bundle
from ..bundle import BundleWriter
from .common import NothingToPublish, provenance, REPO_ROOT, round_cols


BACKTEST_FILES = {'metrics': 'metrics.csv', 'by-horizon': 'by_horizon.csv', 'blocks': 'blocks.csv',
                  'shares': 'shares.csv', 'seats': 'seats.csv'}


PROB_PREFIXES = ('cov_', 'brier', 'log_score', 'p_majority')


BACKTEST_BOOL_COLS = ('seats_valid', 'majority', 'main')


BACKTEST_RUN_EXCLUDED = ('run_at', 'commit', 'scope', 'events', 'horizons', 'note')


def backtest_source(scope: str, root: Optional[str] = None) -> tuple:
    """File system and folder of the backtest tables of a scope, as the ``backtest`` job writes them.

    Parameters
    ----------
    scope : str
        Scope code.
    root : str, optional
        Local results folder. Defaults to ``backtest/results`` in the file system of the app.

    Returns
    -------
    tuple of (FileSystem, str)
        See ``backtest.results_fs``.
    """
    from ..backtest import results_fs  # lazy: the backtest library loads the simulator

    return results_fs(root, scope)


def export_backtest_table(frame: pd.DataFrame) -> tuple:
    """Convert a table of the backtest into the ``columns``/``rows`` form of the ``backtest-*`` schemas.

    A float column with some value, all of them integral and a name not starting with one of
    ``PROB_PREFIXES`` becomes nullable integer (``Int64``: ``int`` in the JSON, ``null`` for NaN, no ``.0`` in
    the CSV); the other float columns are rounded to ``PCT_DECIMALS`` (``PROB_DECIMALS`` those starting with
    one of ``PROB_PREFIXES``), integer columns stay ``int`` and ``seats_valid``, ``majority`` and ``main`` are
    nullable booleans (a missing value stays missing, never ``True``). Missing values are ``None`` in the rows.

    Parameters
    ----------
    frame : pandas.DataFrame
        A table read from ``backtest/results`` (not modified).

    Returns
    -------
    tuple
        ``(data, table)``: the data of the schema, ``{'columns', 'rows'}``, and the rounded table, its CSV twin.
    """
    table = frame.copy()
    floats = [col for col in table.columns if pd.api.types.is_float_dtype(table[col])]
    probs = [col for col in floats if str(col).startswith(PROB_PREFIXES)]
    for col in floats:
        values = table[col].dropna()
        if col not in probs and len(values) and (values % 1 == 0).all():
            table[col] = table[col].astype('Int64')
    floats = [col for col in floats if pd.api.types.is_float_dtype(table[col])]
    table = round_cols(table, pct=[col for col in floats if col not in probs], prob=probs)
    for col in BACKTEST_BOOL_COLS:
        if col in table.columns:
            table[col] = table[col].astype('boolean')

    rows = [{key: None if value is pd.NA else value for key, value in row.items()}
            for row in table.to_dict(orient='records')]

    return {'columns': [str(col) for col in table.columns], 'rows': rows}, table


def export_backtest_meta(scope: str, run: dict, tables, source: str, prov: dict, seconds: dict) -> dict:
    """Build the ``backtest-meta`` document describing how and when the backtest of a scope was produced.

    Parameters
    ----------
    scope : str
        Scope code.
    run : dict
        Content of the ``run.json`` of the backtest, or ``{}`` when there is none (then ``events`` and
        ``horizons`` are the sorted values of ``tables.metrics``).
    tables : object
        Has the ``metrics`` table (with ``event_date``, ``horizon`` and ``seats_valid``).
    source : str
        Where the tables come from (``backtest.public_source``: the folder relative to the repository in local,
        the prefix in the file system of the app for S3, never the bucket), written as is.
    prov : dict
        Result of ``provenance`` at publication.
    seconds : dict
        Seconds spent in each step.

    Returns
    -------
    dict
        Data for the ``backtest-meta`` schema.
    """
    metrics = tables.metrics
    events = run.get('events', sorted(metrics['event_date'].astype(str).unique()))
    horizons = run.get('horizons', sorted(int(h) for h in metrics['horizon'].unique()))
    params = {key: value for key, value in run.items() if key not in BACKTEST_RUN_EXCLUDED}
    return {'scope': scope, 'run_at': run.get('run_at'), 'commit': run.get('commit'), 'params': params,
            'events': list(events), 'horizons': list(horizons), 'n_cases': int(len(metrics)),
            'n_seats_cases': int(metrics['seats_valid'].sum()), 'source': source, 'published': prov,
            'seconds': seconds}


def publish_backtest(scope: str, writer: BundleWriter, source: Optional[str] = None,
                     prov: Optional[Callable] = None) -> dict:
    """Convert the backtest tables of a scope into the bundle, overwriting the previous publication.

    Reads ``metrics.csv``, ``by_horizon.csv``, ``blocks.csv``, ``shares.csv``, ``seats.csv`` and (when it
    exists) ``run.json`` from ``backtest_source(scope, source)``. The five tables are written in the order of
    ``bundle.BACKTEST_CSV`` (each JSON with its CSV twin) and ``meta`` last, so a failure halfway leaves the
    previous ``meta.json`` in place. It reads no database; the manifest and the runs are not touched.

    Parameters
    ----------
    scope : str
        Scope code.
    writer : BundleWriter
        Bundle destination.
    source : str, optional
        Local results folder (see ``backtest_source``); the app file system by default.
    prov : callable, optional
        Injectable replacement of ``provenance``.

    Returns
    -------
    dict
        ``{'scope', 'n_cases', 'n_events', 'seconds'}``.

    Raises
    ------
    NothingToPublish
        There is no ``metrics.csv`` for the scope; nothing is written then.
    """
    start = time.perf_counter()
    from ..backtest import public_source, results_path  # lazy, as in ``backtest_source``

    fs, folder = backtest_source(scope, source)
    if not fs.exists(results_path(folder, BACKTEST_FILES['metrics'])):
        raise NothingToPublish(f'no backtest results for {scope}')

    run = {}
    run_file = results_path(folder, 'run.json')
    if fs.exists(run_file):
        run = json.loads(fs.read_bytes(run_file).decode('utf-8'))
    frames = {part: fs.read_csv(results_path(folder, BACKTEST_FILES[part])) for part in bundle.BACKTEST_CSV}
    tables = types.SimpleNamespace(run=run, **{part.replace('-', '_'): frame for part, frame in frames.items()})
    read = time.perf_counter() - start

    for part in bundle.BACKTEST_CSV:
        data, frame = export_backtest_table(frames[part])
        writer.write_json(bundle.path_backtest(scope, part), bundle.BACKTEST_SCHEMAS[part], data, scope)
        writer.write_csv(bundle.path_backtest_csv(scope, part), frame)

    seconds = {'read': read, 'total': time.perf_counter() - start}
    seconds = {step: round(value, 1) for step, value in seconds.items()}
    prov = prov or (lambda: provenance(cwd=REPO_ROOT))
    meta = export_backtest_meta(scope, run, tables, public_source(fs, folder), prov(), seconds)
    writer.write_json(bundle.path_backtest(scope, 'meta'), bundle.BACKTEST_SCHEMAS['meta'], meta, scope)
    n_events = len(meta['events'])

    return {'scope': scope, 'n_cases': meta['n_cases'], 'n_events': n_events, 'seconds': seconds}
