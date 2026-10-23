"""
Layout, JSON envelope and schemas of the published bundle.

This is the only module that knows the paths of the bundle files, how Python, numpy and pandas
values become JSON (``null`` for NaN) and the keys every published file must have. The bundle
lives under ``PREFIX`` (``DRY_PREFIX`` for dry runs) of ``app.fs`` and is immutable once written::

    site/v1/
        manifest.json                           latest run of every scope
        runs/{scope}/history.json               headline of every run, ascending by run_id
        runs/{scope}/{run_id}/
            meta.json headline.json series.json polls.json fan.json
            house-effects.json dispersion.json
            {nowcast,forecast}/
                vote.json summary.json dist.json districts.json scenario.json projection.json
            csv/
                series.csv ...                  {mode}-{name}.csv for the per-mode files
        analysis/{scope}/                       houses analysis of a scope, overwritten on each publication
            meta.json house-effects.json herding.json poll-errors.json error-data.json
            csv/
                house-effects.csv herding.csv herding-events.csv poll-errors.csv error-data.csv
        backtest/{scope}/                       model backtest of a scope, overwritten on each publication
            meta.json metrics.json by-horizon.json blocks.json shares.json seats.json
            csv/
                metrics.csv by-horizon.csv blocks.csv shares.csv seats.csv
        events/{scope}/{date}/                  one past election of a scope, overwritten on each publication
            meta.json pollsters.json polls.json results.json model.json
            csv/
                pollsters.csv polls.csv results.csv model-shares.csv model-seats.csv model-blocks.csv

Every JSON file is an envelope::

    {schema: 'name@1', contract: 1, scope, run_id, mode, generated_at, data: {...}}

where ``schema`` is ``{name}@{CONTRACT}``, ``scope`` and ``run_id`` are ``null`` when the file
does not belong to one (``manifest``, ``history``; ``run_id`` for the analysis), ``mode`` is only set
for ``MODE_PARTS`` and ``data`` has the keys listed in ``SCHEMAS``. The analysis parts are validated
against the schema named in ``ANALYSIS_SCHEMAS``, the backtest ones against ``BACKTEST_SCHEMAS`` and the event
ones against ``EVENT_SCHEMAS``; ``meta.json`` is written last, so an analysis, a backtest or an event without it is
incomplete. The contract is described in ``docs/web/contrato.md``.

``BundleReader`` and ``BundleWriter`` read and write the bundle through a file system (``app.fs``),
validating every JSON file against its schema before it is written.
"""
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from ..core.io import FileSystem
from ..core.utils.serialize import json_default

PREFIX = 'site/v1'
DRY_PREFIX = 'site-dry/v1'
CONTRACT = 1
MODES = ('nowcast', 'forecast')
RUN_PARTS = ('meta', 'headline', 'series', 'polls', 'fan', 'house-effects', 'dispersion', 'weights')
MODE_PARTS = ('vote', 'summary', 'dist', 'districts', 'scenario', 'projection')
ANALYSIS_PARTS = ('meta', 'house-effects', 'herding', 'poll-errors', 'error-data')
ANALYSIS_SCHEMAS = {'meta': 'analysis-meta', 'house-effects': 'analysis-house-effects', 'herding': 'analysis-herding',
                    'poll-errors': 'poll-errors', 'error-data': 'error-data'}
ANALYSIS_CSV = ('house-effects', 'herding', 'herding-events', 'poll-errors', 'error-data')
BACKTEST_PARTS = ('meta', 'metrics', 'by-horizon', 'blocks', 'shares', 'seats')
BACKTEST_SCHEMAS = {part: 'backtest-' + part for part in BACKTEST_PARTS}
BACKTEST_CSV = BACKTEST_PARTS[1:]
EVENT_PARTS = ('meta', 'pollsters', 'polls', 'cycle', 'results', 'model', 'series')
EVENT_SCHEMAS = {part: 'event-' + part for part in EVENT_PARTS}
EVENT_CSV = ('pollsters', 'polls', 'cycle', 'results', 'model-shares', 'model-seats', 'model-blocks', 'series')
STUDIES = ('voto-cera',)
STUDY_PARTS = ('meta', 'census', 'participation', 'vote', 'placement', 'prediction', 'sensitivity', 'check')
STUDY_SCHEMAS = {part: 'study-' + part for part in STUDY_PARTS}
STUDY_CSV = ('census-series', 'census-provinces', 'census-countries', 'participation', 'vote', 'ecological',
             'last-seat', 'sensitivity-grid', 'sensitivity-scenarios', 'placement-grid', 'check')
STUDY_CSV_OF = {'census': ('census-series', 'census-provinces', 'census-countries'),
                'participation': ('participation',), 'vote': ('vote', 'ecological'), 'prediction': ('last-seat',),
                'sensitivity': ('sensitivity-grid', 'sensitivity-scenarios', 'placement-grid'), 'check': ('check',)}
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
RUN_ID_RE = re.compile(r'^\d{8}-\d{6}$')
ROUTE_ALIAS_RE = re.compile(r'^[0-9a-z_\-]+$')

NULLABLE_STR = (str, type(None))
NULLABLE_INT = (int, type(None))
SCHEMAS = {
    'manifest': {'contract': int, 'updated_at': NULLABLE_STR, 'scopes': dict, 'studies': dict, 'freeze': dict,
                 'attribution': dict},
    'history': {'scope': str, 'runs': list},
    'parties': {'rows': list},
    'meta': {
        'run_id': str, 'run_at': str, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'scope': str,
        'event_date': str, 'as_of': str, 'date_last': str, 'date_fit_last': NULLABLE_STR, 'horizon_max': int,
        'n_sim': int, 'seed': NULLABLE_INT, 'drange': list, 'max_fc': int, 'alpha': float, 'correctors': dict,
        'n_polls': int, 'n_pollsters': int, 'db_polls': NULLABLE_INT, 'db_last_poll': NULLABLE_STR,
        'n_seats': int, 'majority': int, 'parties': list, 'bmaps': dict, 'smap': dict, 'regions': list,
        'diagnostics': dict, 'seconds': dict, 'freeze': bool, 'limit_date': NULLABLE_STR,
    },
    'headline': {'run_id': str, 'run_at': str, 'event_date': str, 'as_of': str, 'date_last': str, 'n_polls': int,
                 'backfill': bool, 'nowcast': dict, 'forecast': dict},
    'series': {'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict},
    'polls': {'parties': list, 'columns': list, 'polls': list, 'results': list},
    'fan': {'horizons': list, 'rows': list},
    'house-effects': {'rows': list},
    'dispersion': {'rows': list},
    'weights': {'date': str, 'parties': list, 'rows': list, 'line': dict, 'kernel': dict, 'stats': dict},
    'vote': {'horizon': int, 'when': str, 'rows': list},
    'summary': {'n_seats': int, 'majority': int, 'parties': list, 'vs': list, 'blocks': list, 'p_majority': dict,
                'totals': dict},
    'dist': {'n_seats': int, 'parties': list, 'seats': list},
    'districts': {'parties': list, 'regions': list, 'rows': list},
    'scenario': {'parties': list, 'rows': list},
    'projection': {'dates': list, 'groups': dict},
    'analysis-meta': {'scope': str, 'event_date': str, 'events': list, 'n_pollsters': int, 'params': dict,
                      'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'seconds': dict},
    'analysis-house-effects': {'parties': list, 'blocks': list, 'rows': list},
    'analysis-herding': {'rows': list, 'events': list},
    'poll-errors': {'blocks': list, 'rows': list},
    'error-data': {'rows': list},
    'backtest-meta': {'scope': str, 'run_at': NULLABLE_STR, 'commit': NULLABLE_STR, 'params': dict, 'events': list,
                      'horizons': list, 'n_cases': int, 'n_seats_cases': int, 'source': str, 'published': dict,
                      'seconds': dict},
    **{schema: {'columns': list, 'rows': list} for part, schema in BACKTEST_SCHEMAS.items() if part != 'meta'},
    'event-meta': {'scope': str, 'event_date': str, 'name': NULLABLE_STR, 'n_polls': int, 'n_pollsters': int,
                   'blocks': dict, 'main': list, 'summary': dict, 'has_model': bool, 'featured': bool,
                   'horizons': list,
                   'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'seconds': dict},
    'event-pollsters': {'blocks': list, 'rows': list},
    'event-polls': {'blocks': list, 'rows': list},
    'event-cycle': {'parties': list, 'columns': list, 'polls': list},
    'event-results': {'n_seats': int, 'rows': list},
    'event-model': {'horizons': list, 'shares': list, 'seats': list, 'blocks': list},
    'event-series': {'horizon': int, 'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict},
    'study-meta': {'study': str, 'title': str, 'subtitle': str, 'election': dict, 'registered_at': NULLABLE_STR,
                   'registered_commit': NULLABLE_STR, 'registered': (dict, type(None)),
                   'registered_changed_at': NULLABLE_STR, 'checked_at': NULLABLE_STR, 'params': dict, 'sources': list,
                   'series_last': str, 'regression': dict, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict,
                   'seconds': dict},
    'study-census': {'series': dict, 'marks': list, 'periods': dict, 'provinces': list, 'countries': list,
                     'origins': list, 'exposure': list, 'election_provinces': list, 'election_countries': list},
    'study-participation': {'origin': list, 'generales': list, 'autonomicas': list},
    'study-vote': {'processes': list, 'ratios': list, 'ecological': dict},
    'study-placement': {'sensitive': dict, 'acceleration': list, 'correlations': dict, 'madrid': list},
    'study-prediction': {'scenarios': dict, 'last_seat': list, 'report': dict},
    'study-sensitivity': {'grid': dict, 'scenarios': list, 'placement': dict},
    'study-check': {'elections': list, 'rows': list, 'verdict': NULLABLE_STR},
}


def run_id(now: Optional[datetime] = None) -> str:
    """
    Build the identifier of a run from its moment, as ``YYYYMMDD-HHMMSS`` in UTC.

    Parameters
    ----------
    now : datetime, optional
        Moment of the run. A naive value is taken as UTC. Defaults to the current time.

    Returns
    -------
    str
        The run id, which matches ``RUN_ID_RE`` and ``ROUTE_ALIAS_RE``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    return now.astimezone(timezone.utc).strftime('%Y%m%d-%H%M%S')


def iso_utc(moment: datetime) -> str:
    """
    Format a moment as ``YYYY-MM-DDTHH:MM:SSZ`` in UTC.

    Parameters
    ----------
    moment : datetime
        The moment to format. A naive value is taken as UTC.

    Returns
    -------
    str
        The ISO 8601 string, converted to UTC.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    return moment.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def jsonable(obj: Any) -> Any:
    """
    Convert a value into plain JSON types (``dict``, ``list``, ``str``, ``int``, ``float``, ``bool``, ``None``).

    Parameters
    ----------
    obj : Any
        The value to convert, at any depth. Non-string dict keys are converted to ``str`` (numpy
        keys go through ``json_default`` first). NaN and infinite floats become ``None``.

    Returns
    -------
    Any
        The JSON-compatible value.

    Raises
    ------
    TypeError
        If the value is a DataFrame or a Series, or its type is not supported by ``json_default``.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                if not isinstance(key, (bool, int, float)):
                    key = json_default(key)
                key = str(key)
            result[key] = jsonable(value)

        return result

    if isinstance(obj, (list, tuple)):
        return [jsonable(item) for item in obj]

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError('export frames as records or columns before serialising')

    return jsonable(json_default(obj))


def dumps(obj: Any) -> bytes:
    """
    Serialise a value to compact UTF-8 JSON bytes, with ``null`` for NaN.

    Parameters
    ----------
    obj : Any
        The value to serialise (see ``jsonable``).

    Returns
    -------
    bytes
        The JSON document.
    """
    return json.dumps(jsonable(obj), ensure_ascii=False, allow_nan=False, separators=(',', ':'),
                      default=json_default).encode('utf-8')


def loads(raw: bytes) -> Any:
    """
    Parse JSON bytes.

    Parameters
    ----------
    raw : bytes
        The JSON document.

    Returns
    -------
    Any
        The parsed value.
    """
    return json.loads(raw)


def envelope(schema: str, data: Any, scope: Optional[str], run_id: Optional[str] = None, mode: Optional[str] = None,
             generated_at: Optional[str] = None) -> dict:
    """
    Wrap the data of a file in the common envelope.

    Parameters
    ----------
    schema : str
        Name of the schema (a key of ``SCHEMAS``), without the version.
    data : Any
        The content of the file.
    scope : str, optional
        Scope of the run, ``None`` if the file does not belong to one.
    run_id : str, optional
        Identifier of the run, ``None`` if the file does not belong to one.
    mode : str, optional
        ``'nowcast'`` or ``'forecast'`` for the per-mode files, ``None`` otherwise.
    generated_at : str, optional
        ISO 8601 UTC moment. Defaults to now.

    Returns
    -------
    dict
        The envelope, with the keys ``schema``, ``contract``, ``scope``, ``run_id``, ``mode``,
        ``generated_at`` and ``data``.
    """
    if generated_at is None:
        generated_at = iso_utc(datetime.now(timezone.utc))

    return {
        'schema': f'{schema}@{CONTRACT}',
        'contract': CONTRACT,
        'scope': scope,
        'run_id': run_id,
        'mode': mode,
        'generated_at': generated_at,
        'data': data,
    }


def path_manifest() -> str:
    """
    Path of the manifest, relative to the bundle prefix.

    Returns
    -------
    str
        ``'manifest.json'``.
    """
    return 'manifest.json'


def path_parties() -> str:
    """
    Path of the party catalogue, relative to the bundle prefix.

    Returns
    -------
    str
        ``'parties.json'``.
    """
    return 'parties.json'


def path_runs(scope: str) -> str:
    """
    Path of the folder with the runs of a scope.

    Parameters
    ----------
    scope : str
        The scope.

    Returns
    -------
    str
        ``'runs/{scope}'``.
    """
    return f'runs/{scope}'


def path_history(scope: str) -> str:
    """
    Path of the history file of a scope.

    Parameters
    ----------
    scope : str
        The scope.

    Returns
    -------
    str
        ``'runs/{scope}/history.json'``.
    """
    return f'{path_runs(scope)}/history.json'


def path_run(scope: str, run_id: str) -> str:
    """
    Path of the folder of a run.

    Parameters
    ----------
    scope : str
        The scope.
    run_id : str
        The run identifier.

    Returns
    -------
    str
        ``'runs/{scope}/{run_id}'``.
    """
    return f'{path_runs(scope)}/{run_id}'


def path_part(scope: str, run_id: str, part: str, mode: Optional[str] = None) -> str:
    """
    Path of a JSON file of a run.

    Parameters
    ----------
    scope : str
        The scope.
    run_id : str
        The run identifier.
    part : str
        The file name without extension (one of ``RUN_PARTS`` or ``MODE_PARTS``).
    mode : str, optional
        Mode folder for the per-mode files.

    Returns
    -------
    str
        ``'runs/{scope}/{run_id}/{part}.json'`` or ``'runs/{scope}/{run_id}/{mode}/{part}.json'``.
    """
    base = path_run(scope, run_id)
    if mode is None:
        return f'{base}/{part}.json'

    return f'{base}/{mode}/{part}.json'


def path_csv(scope: str, run_id: str, name: str, mode: Optional[str] = None) -> str:
    """
    Path of a CSV file of a run.

    Parameters
    ----------
    scope : str
        The scope.
    run_id : str
        The run identifier.
    name : str
        The file name without extension.
    mode : str, optional
        Mode, prepended to the name for the per-mode files.

    Returns
    -------
    str
        ``'runs/{scope}/{run_id}/csv/{name}.csv'`` or ``'.../csv/{mode}-{name}.csv'``.
    """
    base = f'{path_run(scope, run_id)}/csv'
    if mode is None:
        return f'{base}/{name}.csv'

    return f'{base}/{mode}-{name}.csv'


def _check_file(scope: str, name: str, names: tuple, kind: str) -> None:
    """Check the scope and the file name of an analysis or backtest path, raising ``ValueError`` if invalid."""
    if not isinstance(scope, str) or not ROUTE_ALIAS_RE.fullmatch(scope):
        raise ValueError(f'invalid scope {scope!r}')
    if name not in names:
        raise ValueError(f'unknown {kind} file {name!r} (expected one of {", ".join(names)})')


def _check_study(study: str) -> None:
    """Check the slug of a study path, raising ``ValueError`` if it is not one of ``STUDIES``."""
    if not isinstance(study, str) or not ROUTE_ALIAS_RE.fullmatch(study) or study not in STUDIES:
        raise ValueError(f'unknown study {study!r} (expected one of {", ".join(STUDIES)})')


def path_analysis(scope: str, part: str) -> str:
    """
    Path of a JSON file of the analysis of a scope.

    Parameters
    ----------
    scope : str
        The scope; it must match ``ROUTE_ALIAS_RE``.
    part : str
        The file name without extension (one of ``ANALYSIS_PARTS``).

    Returns
    -------
    str
        ``'analysis/{scope}/{part}.json'``.

    Raises
    ------
    ValueError
        If the scope or the part is invalid.
    """
    _check_file(scope, part, ANALYSIS_PARTS, 'analysis')

    return f'analysis/{scope}/{part}.json'


def path_analysis_csv(scope: str, name: str) -> str:
    """
    Path of a CSV file of the analysis of a scope.

    Parameters
    ----------
    scope : str
        The scope; it must match ``ROUTE_ALIAS_RE``.
    name : str
        The file name without extension (one of ``ANALYSIS_CSV``).

    Returns
    -------
    str
        ``'analysis/{scope}/csv/{name}.csv'``.

    Raises
    ------
    ValueError
        If the scope or the name is invalid.
    """
    _check_file(scope, name, ANALYSIS_CSV, 'analysis')

    return f'analysis/{scope}/csv/{name}.csv'


def path_study(study: str, part: str) -> str:
    """
    Path of a JSON file of a study.

    Parameters
    ----------
    study : str
        The study slug (one of ``STUDIES``).
    part : str
        The file name without extension (one of ``STUDY_PARTS``).

    Returns
    -------
    str
        ``'studies/{study}/{part}.json'``.

    Raises
    ------
    ValueError
        If the study or the part is invalid.
    """
    _check_study(study)
    _check_file(study, part, STUDY_PARTS, 'study')

    return f'studies/{study}/{part}.json'


def path_study_csv(study: str, name: str) -> str:
    """
    Path of a CSV file of a study.

    Parameters
    ----------
    study : str
        The study slug (one of ``STUDIES``).
    name : str
        The file name without extension (one of ``STUDY_CSV``).

    Returns
    -------
    str
        ``'studies/{study}/csv/{name}.csv'``.

    Raises
    ------
    ValueError
        If the study or the name is invalid.
    """
    _check_study(study)
    _check_file(study, name, STUDY_CSV, 'study')

    return f'studies/{study}/csv/{name}.csv'


def path_backtest(scope: str, part: str) -> str:
    """
    Path of a JSON file of the backtest of a scope.

    Parameters
    ----------
    scope : str
        The scope; it must match ``ROUTE_ALIAS_RE``.
    part : str
        The file name without extension (one of ``BACKTEST_PARTS``).

    Returns
    -------
    str
        ``'backtest/{scope}/{part}.json'``.

    Raises
    ------
    ValueError
        If the scope or the part is invalid.
    """
    _check_file(scope, part, BACKTEST_PARTS, 'backtest')

    return f'backtest/{scope}/{part}.json'


def path_backtest_csv(scope: str, name: str) -> str:
    """
    Path of a CSV file of the backtest of a scope.

    Parameters
    ----------
    scope : str
        The scope; it must match ``ROUTE_ALIAS_RE``.
    name : str
        The file name without extension (one of ``BACKTEST_CSV``).

    Returns
    -------
    str
        ``'backtest/{scope}/csv/{name}.csv'``.

    Raises
    ------
    ValueError
        If the scope or the name is invalid.
    """
    _check_file(scope, name, BACKTEST_CSV, 'backtest')

    return f'backtest/{scope}/csv/{name}.csv'


def _check_event(scope: str, date: str, name: str, names: tuple) -> None:
    """Check the scope, the date and the file name of an event path, raising ``ValueError`` if invalid."""
    _check_file(scope, name, names, 'event')
    if not isinstance(date, str) or not DATE_RE.fullmatch(date):
        raise ValueError(f'invalid event date {date!r}')


def path_event(scope: str, date: str, part: str) -> str:
    """
    Path of a JSON file of a past election of a scope.

    Parameters
    ----------
    scope : str
        The scope; it must match ``ROUTE_ALIAS_RE``.
    date : str
        Election date, ``YYYY-MM-DD`` (``DATE_RE``).
    part : str
        The file name without extension (one of ``EVENT_PARTS``).

    Returns
    -------
    str
        ``'events/{scope}/{date}/{part}.json'``.

    Raises
    ------
    ValueError
        If the scope, the date or the part is invalid.
    """
    _check_event(scope, date, part, EVENT_PARTS)

    return f'events/{scope}/{date}/{part}.json'


def path_event_csv(scope: str, date: str, name: str) -> str:
    """
    Path of a CSV file of a past election of a scope.

    Parameters
    ----------
    scope : str
        The scope; it must match ``ROUTE_ALIAS_RE``.
    date : str
        Election date, ``YYYY-MM-DD`` (``DATE_RE``).
    name : str
        The file name without extension (one of ``EVENT_CSV``).

    Returns
    -------
    str
        ``'events/{scope}/{date}/csv/{name}.csv'``.

    Raises
    ------
    ValueError
        If the scope, the date or the name is invalid.
    """
    _check_event(scope, date, name, EVENT_CSV)

    return f'events/{scope}/{date}/csv/{name}.csv'


def validate(name: str, obj: dict) -> dict:
    """
    Check that an envelope has the shape of a schema.

    Parameters
    ----------
    name : str
        Name of the schema (a key of ``SCHEMAS``).
    obj : dict
        The envelope to check.

    Returns
    -------
    dict
        The same envelope.

    Raises
    ------
    ValueError
        If the schema is unknown, the envelope is malformed or the ``data`` lacks a required key
        or has a value of the wrong type. The message starts with ``'{name}: '``.
    """
    if name not in SCHEMAS:
        raise ValueError(f'{name}: unknown schema')

    expected = f'{name}@{CONTRACT}'
    got = obj.get('schema')
    if got != expected:
        raise ValueError(f'{name}: schema {got} != {expected}')

    if obj.get('contract') != CONTRACT:
        raise ValueError(f'{name}: contract {obj.get("contract")} != {CONTRACT}')

    if not isinstance(obj.get('scope'), NULLABLE_STR):
        raise ValueError(f'{name}: invalid scope')

    rid = obj.get('run_id')
    if rid is not None and not (isinstance(rid, str) and RUN_ID_RE.match(rid)):
        raise ValueError(f'{name}: invalid run_id')

    mode = obj.get('mode')
    if (mode is None and name in MODE_PARTS) or (mode is not None and mode not in MODES):
        raise ValueError(f'{name}: invalid mode')

    if not isinstance(obj.get('generated_at'), str):
        raise ValueError(f'{name}: invalid generated_at')

    data = obj.get('data')
    if not isinstance(data, dict):
        raise ValueError(f'{name}: data must be an object')

    schema = SCHEMAS[name]
    missing = sorted(key for key in schema if key not in data)
    if missing:
        raise ValueError(f'{name}: missing keys {missing}')

    for key, types in schema.items():
        value = data[key]
        if not isinstance(value, types) or (types is float and isinstance(value, bool)):
            raise ValueError(f'{name}: wrong type for {key}')

    return obj


class BundleReader:
    """
    Read-only access to a bundle stored in a file system.

    Parameters
    ----------
    fs : FileSystem
        The file system (local or S3) that holds the bundle.
    prefix : str, optional
        Root of the bundle in the file system, without trailing slash. Defaults to ``PREFIX``.
    """

    def __init__(self, fs: FileSystem, prefix: str = PREFIX):
        self.fs = fs
        self.prefix = prefix

    def path(self, name: str) -> str:
        """
        Full path of a bundle file in the file system.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.

        Returns
        -------
        str
            ``'{prefix}/{name}'``.
        """
        return f'{self.prefix}/{name}'

    def exists(self, name: str) -> bool:
        """
        Check whether a file or folder of the bundle exists.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.

        Returns
        -------
        bool
            ``True`` if it exists.
        """
        return self.fs.exists(self.path(name))

    def read_json(self, name: str) -> dict:
        """
        Read and parse a JSON file of the bundle.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.

        Returns
        -------
        dict
            The parsed envelope.
        """
        return loads(self.fs.read_bytes(self.path(name)))

    def list_runs(self, scope: str) -> list:
        """
        List the complete runs of a scope, a run being complete when it has its ``headline.json``.

        Parameters
        ----------
        scope : str
            The scope.

        Returns
        -------
        list of str
            The run ids, ascending. Empty if the scope has no runs folder.
        """
        try:
            names = self.fs.listdir(self.path(path_runs(scope)))
        except FileNotFoundError:
            return []

        return sorted(
            name for name in names
            if RUN_ID_RE.match(name) and self.exists(path_part(scope, name, 'headline'))
        )

    def analysis_exists(self, scope: str) -> bool:
        """
        Check whether the analysis of a scope is published, which is when it has its ``meta.json``.

        Parameters
        ----------
        scope : str
            The scope; it must match ``ROUTE_ALIAS_RE``.

        Returns
        -------
        bool
            ``True`` if ``analysis/{scope}/meta.json`` exists.

        Raises
        ------
        ValueError
            If the scope is invalid.
        """
        return self.exists(path_analysis(scope, 'meta'))

    def study_exists(self, study: str) -> bool:
        """
        Check whether a study is published, which is when it has its ``meta.json``.

        Parameters
        ----------
        study : str
            The study slug (one of ``STUDIES``).

        Returns
        -------
        bool
            ``True`` if ``studies/{study}/meta.json`` exists.

        Raises
        ------
        ValueError
            If the study is invalid.
        """
        return self.exists(path_study(study, 'meta'))

    def backtest_exists(self, scope: str) -> bool:
        """
        Check whether the backtest of a scope is published, which is when it has its ``meta.json``.

        Parameters
        ----------
        scope : str
            The scope; it must match ``ROUTE_ALIAS_RE``.

        Returns
        -------
        bool
            ``True`` if ``backtest/{scope}/meta.json`` exists.

        Raises
        ------
        ValueError
            If the scope is invalid.
        """
        return self.exists(path_backtest(scope, 'meta'))

    def list_events(self, scope: str) -> list:
        """
        List the published past elections of a scope, an election being published when it has its ``meta.json``.

        Parameters
        ----------
        scope : str
            The scope; it must match ``ROUTE_ALIAS_RE``.

        Returns
        -------
        list of str
            The election dates, ascending. Empty if the scope has no events folder.

        Raises
        ------
        ValueError
            If the scope is invalid.
        """
        _check_file(scope, 'meta', EVENT_PARTS, 'event')
        try:
            names = self.fs.listdir(self.path(f'events/{scope}'))
        except FileNotFoundError:
            return []

        return sorted(name for name in names if DATE_RE.fullmatch(name) and self.event_exists(scope, name))

    def event_exists(self, scope: str, date: str) -> bool:
        """
        Check whether a past election of a scope is published, which is when it has its ``meta.json``.

        Parameters
        ----------
        scope : str
            The scope; it must match ``ROUTE_ALIAS_RE``.
        date : str
            Election date, ``YYYY-MM-DD``.

        Returns
        -------
        bool
            ``True`` if ``events/{scope}/{date}/meta.json`` exists.

        Raises
        ------
        ValueError
            If the scope or the date is invalid.
        """
        return self.exists(path_event(scope, date, 'meta'))


class BundleWriter(BundleReader):
    """
    Read and write access to a bundle stored in a file system.

    Parameters
    ----------
    fs : FileSystem
        The file system (local or S3) that holds the bundle.
    prefix : str, optional
        Root of the bundle in the file system, without trailing slash. Defaults to ``PREFIX``.
    clock : callable, optional
        Function returning the current moment as a ``datetime``. Defaults to the current UTC time.
    """

    def __init__(self, fs: FileSystem, prefix: str = PREFIX, clock: Optional[Callable[[], datetime]] = None):
        super().__init__(fs, prefix)
        self.clock = clock

    def now(self) -> datetime:
        """
        Current moment according to the clock.

        Returns
        -------
        datetime
            The value of the clock, or the current UTC time if there is none.
        """
        if self.clock is not None:
            return self.clock()

        return datetime.now(timezone.utc)

    def begin_run(self, scope: str, run_id: str) -> None:
        """
        Check that a run can be written, as runs are immutable. Nothing is created.

        Parameters
        ----------
        scope : str
            The scope.
        run_id : str
            The run identifier.

        Raises
        ------
        FileExistsError
            If the run already exists.
        """
        if self.exists(path_run(scope, run_id)):
            raise FileExistsError(f'run {run_id} of scope {scope} already exists')

    def write_json(self, name: str, schema: str, data: Any, scope: Optional[str], run_id: Optional[str] = None,
                   mode: Optional[str] = None) -> str:
        """
        Wrap data in an envelope, validate it and write it as JSON.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.
        schema : str
            Name of the schema (a key of ``SCHEMAS``).
        data : Any
            The content of the file.
        scope : str, optional
            Scope of the run, ``None`` if the file does not belong to one.
        run_id : str, optional
            Identifier of the run, ``None`` if the file does not belong to one.
        mode : str, optional
            ``'nowcast'`` or ``'forecast'`` for the per-mode files, ``None`` otherwise.

        Returns
        -------
        str
            The ``name`` that was written.

        Raises
        ------
        ValueError
            If the envelope does not validate; nothing is written.
        """
        env = jsonable(envelope(schema, data, scope, run_id, mode, iso_utc(self.now())))
        validate(schema, env)
        self.fs.write_bytes(dumps(env), self.path(name))

        return name

    def write_csv(self, name: str, frame: pd.DataFrame) -> str:
        """
        Write a DataFrame as UTF-8 CSV with ``\\n`` line endings and no index.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.
        frame : pd.DataFrame
            The table to write.

        Returns
        -------
        str
            The ``name`` that was written.
        """
        self.fs.write_bytes(frame.to_csv(index=False, lineterminator='\n').encode('utf-8'), self.path(name))

        return name

    def remove(self, name: str) -> None:
        """
        Remove a file or folder (recursively) of the bundle.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.
        """
        self.fs.remove(self.path(name))
