"""Reader of the published bundle: the ``Site`` service and the ``site()`` accessor."""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from ...core.api import HttpError
from ...core.app import App
from .. import bundle
from ..bundle import BundleReader
from .caches import DEFAULT_TTL, LRU_SIZE, LRUCache, TTLCache, settings
from .checks import catalogue

IMMUTABLE_MAX_AGE = 31536000
# Threads that read the small files of the bundle side by side. Measured on 2026-11-11 with the 18 listings and the 94
# metas of the past elections, from a laptop: 18.4 s with 1 thread, 2.4 s with 8, 1.5 s with 16 and 1.05 s with 32
POOL_SIZE = 16


class Site:
    """
    Service over the published bundle: pointers under a TTL, run files under an LRU.

    Files are returned as the bytes stored in the bundle, never re-serialised.

    The files that a page needs by the dozen (the metas of the past elections, the analysis of every scope) are read
    in waves over a thread pool, always started from the thread of the request: a task of the pool never waits for
    another one, so the pool cannot block itself whatever its size.

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
    workers : int, optional
        Threads of the pool that reads several files at once.
    """

    def __init__(self, reader: BundleReader, ttl: float = DEFAULT_TTL, lru: int = LRU_SIZE,
                 clock: Callable[[], float] = time.monotonic, workers: int = POOL_SIZE):
        self.reader = reader
        self.pointers = TTLCache(ttl, clock)
        self.files = LRUCache(lru)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='site-read')

    def _each(self, call: Callable, items: Sequence) -> list:
        """
        Results of a call for every item, run side by side in the pool.

        Only the thread of the request may use it: ``call`` must not come back here (nor to anything that waits for
        the pool), or the tasks could end up waiting for each other.

        Parameters
        ----------
        call : callable
            Function of one item.
        items : sequence
            Items to process.

        Returns
        -------
        list
            ``call(item)`` per item, in the order of ``items``; fewer than two items run in the calling thread.

        Raises
        ------
        Exception
            The first error of ``call``, by the order of the items.
        """
        items = list(items)

        if len(items) < 2:
            return [call(item) for item in items]

        return list(self.pool.map(call, items))

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

    def parties(self) -> bytes:
        """
        Bytes of ``parties.json``, the party catalogue published next to the manifest.

        Returns
        -------
        bytes
            The file content, cached under the TTL like the other overwritable files.

        Raises
        ------
        HttpError
            404 when no catalogue has been published yet.
        """
        return self._mutable_bytes('parties', bundle.path_parties(), 'parties not published')

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

    def study(self, study: str, part: str) -> bytes:
        """
        Bytes of a JSON part of a published study.

        Parameters
        ----------
        study : str
            Validated study slug.
        part : str
            Part name, one of ``STUDY_PARTS[study]``.

        Returns
        -------
        bytes
            The JSON file.

        Raises
        ------
        HttpError
            404 when the part is not published.
        """
        return self._mutable_bytes('study', bundle.path_study(study, part), 'study not published')

    def study_csv(self, study: str, name: str) -> bytes:
        """
        Bytes of a CSV twin of a published study.

        Parameters
        ----------
        study : str
            Validated study slug.
        name : str
            CSV name, one of ``bundle.STUDY_CSV[study]``.

        Returns
        -------
        bytes
            The CSV file.

        Raises
        ------
        HttpError
            404 when the CSV is not published.
        """
        return self._mutable_bytes('study', bundle.path_study_csv(study, name), 'study not published')

    def studies(self) -> list:
        """
        Slugs of the studies that are published, which is when they have their ``meta.json``.

        Returns
        -------
        list of str
            The slugs of ``bundle.STUDIES`` with a published ``meta``, in that order; the metas are read side by
            side.
        """
        def published(study):
            try:
                self.study(study, 'meta')
            except HttpError:
                return False

            return True

        return [study for study, found in zip(bundle.STUDIES, self._each(published, bundle.STUDIES)) if found]

    def _event_meta(self, scope: str, date: str) -> Optional[bytes]:
        """
        Bytes of the ``meta`` part of a past election, or ``None`` when it is not published.

        Parameters
        ----------
        scope : str
            Validated scope code.
        date : str
            Election date.

        Returns
        -------
        bytes or None
            The JSON file through the TTL cache (``event``), ``None`` on its 404.

        Raises
        ------
        Exception
            Any error of the read other than the 404 of an unpublished election.
        """
        try:
            return self.event(scope, date, 'meta')
        except HttpError as error:
            if error.status == 404:
                return None

            raise

    def _load_events(self, scopes: Sequence[str]) -> dict:
        """
        Published election dates of several scopes, read in two waves: the listings, then every ``meta``.

        An election is published when it has its ``meta`` (the rule of ``BundleReader.list_events``); reading it
        answers that and leaves it in the TTL cache for whoever needs it next, so nothing asks ``exists``.

        Parameters
        ----------
        scopes : sequence of str
            Validated scope codes.

        Returns
        -------
        dict
            ``{scope: [date, ...]}``, the dates ascending; an empty list for a scope without elections.
        """
        scopes = list(scopes)
        listings = self._each(self.reader.list_event_dates, scopes)
        pairs = [(scope, date) for scope, dates in zip(scopes, listings) for date in dates]
        metas = self._each(lambda pair: self._event_meta(*pair), pairs)
        published = {scope: [] for scope in scopes}

        for (scope, date), meta in zip(pairs, metas):
            if meta is not None:
                published[scope].append(date)

        return published

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
            The dates, ascending, cached under the TTL (``_load_events``: one listing and the metas side by side).
            Empty when the scope has none.
        """
        return self.pointers.get(('events', scope), lambda: self._load_events([scope])[scope])

    def warm_events(self, scopes: Sequence[str]) -> None:
        """
        Read at once the published elections of several scopes, for the pages that go over all of them.

        The scopes whose list of dates is still cached are left alone; the rest are loaded together (``_load_events``:
        every listing side by side, then every meta), instead of scope by scope as ``events`` would.

        Parameters
        ----------
        scopes : sequence of str
            Validated scope codes.

        Returns
        -------
        None
        """
        stale = [scope for scope in scopes if not self.pointers.fresh(('events', scope))]

        if not stale:
            return

        loaded = self._load_events(stale)

        for scope in stale:
            self.pointers.get(('events', scope), lambda scope=scope: loaded[scope])

    def warm_analysis(self, scopes: Sequence[str], parts: Sequence[str]) -> None:
        """
        Read at once some parts of the houses analysis of several scopes, so the page finds them in the TTL cache.

        Parameters
        ----------
        scopes : sequence of str
            Validated scope codes.
        parts : sequence of str
            Part names, of ``ANALYSIS_PARTS``.

        Returns
        -------
        None

        Raises
        ------
        HttpError
            Any error of a read other than the 404 of an unpublished part, which is ignored.
        """
        def read(pair):
            try:
                self.analysis(*pair)
            except HttpError as error:
                if error.status != 404:
                    raise

        self._each(read, [(scope, part) for scope in scopes for part in parts])

    def event_metas(self, scope: str) -> dict:
        """
        The ``meta`` part of every published past election of a scope.

        The metas whose cache entry has expired while the list of dates has not are read again side by side, so the
        caller never falls back to reading them one by one.

        Parameters
        ----------
        scope : str
            Validated scope code.

        Returns
        -------
        dict
            ``{date: bytes}`` in the order of ``events`` (ascending); a meta that has disappeared since the list was
            read is left out.
        """
        dates = self.events(scope)
        stale = [date for date in dates
                 if not self.pointers.fresh(('event', bundle.path_event(scope, date, 'meta')))]
        self._each(lambda date: self._event_meta(scope, date), stale)
        metas = {}

        for date in dates:
            raw = self._event_meta(scope, date)

            if raw is not None:
                metas[date] = raw

        return metas

    def backtests(self) -> list:
        """
        Scopes with a published backtest.

        Returns
        -------
        list of str
            The scope codes, sorted, cached under the TTL (``BundleReader.list_backtests``).
        """
        return self.pointers.get('backtests', self.reader.list_backtests)

    def cached(self, key: tuple, loader: Callable[[], Any]) -> Any:
        """
        A value derived from several files of the bundle, cached under the TTL of the pointers.

        For the pages that combine parts of many scopes: within the TTL the parts are not read again, missing ones
        included (the read of a missing part raises, so the TTL never stores it); ``manifest_cache_clear`` drops the
        value with the pointers.

        Parameters
        ----------
        key : tuple
            Cache key; its first member names the value (``('reliability', scopes)``).
        loader : callable
            Function without arguments that computes the value on a miss.

        Returns
        -------
        object
            The cached or freshly computed value (``None`` is cached too).
        """
        return self.pointers.get(('derived',) + tuple(key), loader)

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
