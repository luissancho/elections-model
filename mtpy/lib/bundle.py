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

Every JSON file is an envelope::

    {schema: 'name@1', contract: 1, scope, run_id, mode, generated_at, data: {...}}

where ``schema`` is ``{name}@{CONTRACT}``, ``scope`` and ``run_id`` are ``null`` when the file
does not belong to one (``manifest``, ``history``), ``mode`` is only set for ``MODE_PARTS`` and
``data`` has the keys listed in ``SCHEMAS``. The contract is described in ``docs/web/contrato.md``.

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
RUN_PARTS = ('meta', 'headline', 'series', 'polls', 'fan', 'house-effects', 'dispersion')
MODE_PARTS = ('vote', 'summary', 'dist', 'districts', 'scenario', 'projection')
RUN_ID_RE = re.compile(r'^\d{8}-\d{6}$')
ROUTE_ALIAS_RE = re.compile(r'^[0-9a-z_\-]+$')

NULLABLE_STR = (str, type(None))
NULLABLE_INT = (int, type(None))
SCHEMAS = {
    'manifest': {'contract': int, 'updated_at': NULLABLE_STR, 'scopes': dict, 'freeze': dict, 'attribution': dict},
    'history': {'scope': str, 'runs': list},
    'meta': {
        'run_id': str, 'run_at': str, 'commit': NULLABLE_STR, 'dirty': bool, 'versions': dict, 'scope': str,
        'event_date': str, 'as_of': str, 'date_last': str, 'date_fit_last': NULLABLE_STR, 'horizon_max': int,
        'n_sim': int, 'seed': NULLABLE_INT, 'drange': list, 'max_fc': int, 'alpha': float, 'correctors': dict,
        'n_polls': int, 'n_pollsters': int, 'db_polls': NULLABLE_INT, 'db_last_poll': NULLABLE_STR,
        'n_seats': int, 'majority': int, 'parties': list, 'bmaps': dict, 'smap': dict, 'regions': list,
        'diagnostics': dict, 'seconds': dict, 'freeze': bool,
    },
    'headline': {'run_id': str, 'run_at': str, 'event_date': str, 'as_of': str, 'date_last': str, 'n_polls': int,
                 'nowcast': dict, 'forecast': dict},
    'series': {'dates': list, 'parties': list, 'mean': dict, 'lo': dict, 'hi': dict},
    'polls': {'parties': list, 'columns': list, 'polls': list, 'results': list},
    'fan': {'horizons': list, 'rows': list},
    'house-effects': {'rows': list},
    'dispersion': {'rows': list},
    'vote': {'horizon': int, 'when': str, 'rows': list},
    'summary': {'n_seats': int, 'majority': int, 'parties': list, 'vs': list, 'blocks': list, 'p_majority': dict,
                'totals': dict},
    'dist': {'n_seats': int, 'parties': list, 'seats': list},
    'districts': {'parties': list, 'regions': list, 'rows': list},
    'scenario': {'simulation': int, 'parties': list, 'rows': list},
    'projection': {'dates': list, 'groups': dict},
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
