"""Reader of the published bundle: the ``Site`` service and the ``site()`` accessor."""
import time
from typing import Callable, Optional

import pandas as pd

from ...core.api import HttpError
from ...core.app import App
from .. import bundle
from ..bundle import BundleReader
from .caches import DEFAULT_TTL, LRU_SIZE, LRUCache, TTLCache, settings
from .checks import catalogue

IMMUTABLE_MAX_AGE = 31536000


class Site:
    """
    Service over the published bundle: pointers under a TTL, run files under an LRU.

    Files are returned as the bytes stored in the bundle, never re-serialised.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    ttl : float, optional
        Seconds the manifest and history pointers stay cached.
    lru : int, optional
        Number of run files kept in memory.
    clock : callable, optional
        Time source of the TTL cache.
    """

    def __init__(self, reader: BundleReader, ttl: float = DEFAULT_TTL, lru: int = LRU_SIZE,
                 clock: Callable[[], float] = time.monotonic):
        self.reader = reader
        self.pointers = TTLCache(ttl, clock)
        self.files = LRUCache(lru)

    def _read(self, name: str) -> bytes:
        """
        Read a bundle file verbatim.

        Parameters
        ----------
        name : str
            Path relative to the bundle prefix.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        """
        return self.reader.fs.read_bytes(self.reader.path(name))

    def manifest_cache_clear(self) -> None:
        """
        Drop the cached pointers so the next read goes to the file system.

        Returns
        -------
        None
        """
        self.pointers.clear()

    def _manifest_entry(self) -> tuple:
        """
        Raw bytes and parsed payload of ``manifest.json``, cached together under the TTL.

        Returns
        -------
        tuple
            ``(bytes, data)``: the file content and the ``data`` member of its envelope.

        Raises
        ------
        HttpError
            503 when no bundle has been published yet.
        """
        def load():
            try:
                raw = self._read(bundle.path_manifest())
            except FileNotFoundError:
                raise HttpError(503, 'no bundle published yet')

            return raw, bundle.loads(raw)['data']

        return self.pointers.get('manifest', load)

    def manifest(self) -> bytes:
        """
        Bytes of ``manifest.json``.

        Returns
        -------
        bytes
            The manifest, cached under the TTL.

        Raises
        ------
        HttpError
            503 when no bundle has been published yet.
        """
        return self._manifest_entry()[0]

    def manifest_data(self) -> dict:
        """
        Payload of the manifest, parsed once per TTL.

        Returns
        -------
        dict
            The ``data`` member of the manifest envelope.

        Raises
        ------
        HttpError
            503 when no bundle has been published yet.
        """
        return self._manifest_entry()[1]

    def freeze(self) -> dict:
        """
        Freeze state published in the manifest.

        Returns
        -------
        dict
            The ``freeze`` member of the manifest data.
        """
        return self.manifest_data()['freeze']

    def latest_run(self, scope: str) -> str:
        """
        Run id of the latest published run of a scope.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        str
            The run id.

        Raises
        ------
        HttpError
            404 when the scope has no entry in the manifest.
        """
        entry = self.manifest_data()['scopes'].get(scope)

        if entry is None:
            raise HttpError(404, 'scope not published')

        return entry['latest']

    def history(self, scope: str) -> bytes:
        """
        Bytes of the history file of a scope.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        bytes
            The history, cached under the TTL.

        Raises
        ------
        HttpError
            404 when the scope has no published runs.
        """
        def load():
            try:
                return self._read(bundle.path_history(scope))
            except FileNotFoundError:
                raise HttpError(404, 'no runs published')

        return self.pointers.get(('history', scope), load)

    def _run_bytes(self, path: str) -> bytes:
        """
        Read a run file through the LRU cache.

        Parameters
        ----------
        path : str
            Path relative to the bundle prefix.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        HttpError
            404 when the file does not exist.
        """
        def load():
            try:
                return self._read(path)
            except FileNotFoundError:
                raise HttpError(404, 'run not found')

        return self.files.get(path, load)

    def run_file(self, scope: str, run_id: str, part: str, mode: Optional[str] = None) -> bytes:
        """
        Bytes of a JSON part of a run.

        Parameters
        ----------
        scope : str
            Scope code.
        run_id : str
            Run id.
        part : str
            Part name.
        mode : str, optional
            Simulation mode, for the mode parts.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 when the run or the part does not exist.
        """
        return self._run_bytes(bundle.path_part(scope, run_id, part, mode))

    def run_csv(self, scope: str, run_id: str, name: str, mode: Optional[str] = None) -> bytes:
        """
        Bytes of a CSV twin of a part of a run.

        Parameters
        ----------
        scope : str
            Scope code.
        run_id : str
            Run id.
        name : str
            Part name.
        mode : str, optional
            Simulation mode, for the mode parts.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 when the run or the CSV does not exist.
        """
        return self._run_bytes(bundle.path_csv(scope, run_id, name, mode))

    def _mutable_bytes(self, kind: str, path: str, message: str) -> bytes:
        """
        Read an overwritable file of the bundle (analysis, events, backtest) through the TTL cache.

        These files are overwritten on every publication, so they are cached like the pointers,
        not like the immutable run files.

        Parameters
        ----------
        kind : str
            Kind of file, the first member of the cache key.
        path : str
            Path relative to the bundle prefix.
        message : str
            Message of the 404 raised when the file does not exist.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        HttpError
            404 with ``message`` when the file does not exist.
        """
        def load():
            try:
                return self._read(path)
            except FileNotFoundError:
                raise HttpError(404, message)

        return self.pointers.get((kind, path), load)

    def analysis(self, scope: str, part: str) -> bytes:
        """
        Bytes of a JSON part of the houses analysis of a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.
        part : str
            Part name, one of ``ANALYSIS_PARTS``.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 when the part is not published.
        """
        return self._mutable_bytes('analysis', bundle.path_analysis(scope, part), 'analysis not published')

    def analysis_csv(self, scope: str, name: str) -> bytes:
        """
        Bytes of a CSV twin of the houses analysis of a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.
        name : str
            CSV name, one of ``bundle.ANALYSIS_CSV``.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 when the CSV is not published.
        """
        return self._mutable_bytes('analysis', bundle.path_analysis_csv(scope, name), 'analysis not published')

    def events(self, scope: str) -> list:
        """
        Dates of the past elections published for a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.

        Returns
        -------
        list of str
            The dates, ascending, cached under the TTL. Empty when the scope has none.
        """
        return self.pointers.get(('events', scope), lambda: self.reader.list_events(scope))

    def event(self, scope: str, date: str, part: str) -> bytes:
        """
        Bytes of a JSON part of a past election.

        Parameters
        ----------
        scope : str
            Validated scope code.
        date : str
            Validated election date.
        part : str
            Part name, one of ``EVENT_PARTS``.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 ``event not published`` when the file does not exist.
        """
        return self._mutable_bytes('event', bundle.path_event(scope, date, part), 'event not published')

    def event_csv(self, scope: str, date: str, name: str) -> bytes:
        """
        Bytes of a CSV twin of a past election.

        Parameters
        ----------
        scope : str
            Validated scope code.
        date : str
            Validated election date.
        name : str
            CSV name, one of ``bundle.EVENT_CSV``.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 ``event not published`` when the file does not exist.
        """
        return self._mutable_bytes('event', bundle.path_event_csv(scope, date, name), 'event not published')

    def backtest(self, scope: str, part: str) -> bytes:
        """
        Bytes of a JSON part of the model backtest of a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.
        part : str
            Part name, one of ``BACKTEST_PARTS``.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 ``backtest not published`` when the file does not exist.
        """
        return self._mutable_bytes('backtest', bundle.path_backtest(scope, part), 'backtest not published')

    def backtest_csv(self, scope: str, name: str) -> bytes:
        """
        Bytes of a CSV twin of the model backtest of a scope.

        Parameters
        ----------
        scope : str
            Validated scope code.
        name : str
            CSV name, one of ``bundle.BACKTEST_CSV``.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 ``backtest not published`` when the file does not exist.
        """
        return self._mutable_bytes('backtest', bundle.path_backtest_csv(scope, name), 'backtest not published')

    def scopes(self) -> dict:
        """
        Scopes catalogue crossed with the manifest.

        Returns
        -------
        dict
            ``{'contract': int, 'scopes': [row, ...]}`` with one row per catalogue scope, in
            catalogue order. Rows of scopes without a manifest entry are not simulable and
            their run fields are ``None``.
        """
        published = self.manifest_data()['scopes']
        fields = ('run_at', 'event_date', 'as_of', 'date_last', 'n_polls')
        rows = []

        for code, scope in catalogue().iterrows():
            entry = published.get(code)
            row = {
                'code': code,
                'name': scope['name'],
                'parent': None if pd.isna(scope['parent']) else scope['parent'],
                'seats': int(scope['seats']),
                'simulable': entry is not None,
                'latest': entry['latest'] if entry is not None else None,
            }

            for field in fields:
                row[field] = entry.get(field) if entry is not None else None

            rows.append(row)

        return {'contract': bundle.CONTRACT, 'scopes': rows}


def site() -> Site:
    """
    Site service of the process, created on first use and kept in the app.

    Returns
    -------
    Site
        The shared instance.

    Raises
    ------
    HttpError
        503 when the app has no file system.
    """
    app = App.get_()
    instance = app.site

    if instance is None:
        if app.fs is None:
            raise HttpError(503, 'no file system configured')

        conf = settings()
        instance = Site(BundleReader(app.fs, prefix=conf['prefix']), ttl=conf['cache_ttl'])
        app.set('site', instance)

    return instance
