"""Small in-process caches of the web API (TTL and LRU) and the ``web`` settings that size them."""
import time
import warnings
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Optional

from ...core.app import App
from .. import bundle

DEFAULT_TTL = 60
# Public origin of the site (``WEB_SITE_URL``: demoscopio.es in production, the local host in development),
# for the canonical links of the pages
DEFAULT_SITE_URL = 'https://demoscopio.es'
LRU_SIZE = 200


def settings() -> dict:
    """
    Read the ``web`` section of the configuration.

    Returns
    -------
    dict
        ``{'prefix': str, 'cache_ttl': float, 'site_url': str}``. Missing or empty values fall back
        to ``bundle.PREFIX``, ``DEFAULT_TTL`` and ``DEFAULT_SITE_URL``; a ``cache_ttl`` that is not a
        number also falls back, with a warning. ``site_url`` (the public origin of the site, for
        the canonical links) loses its trailing slash.
    """
    config = getattr(App.get_(), 'config', None)
    web = getattr(config, 'web', None) if config is not None else None
    prefix = getattr(web, 'prefix', None)
    ttl = getattr(web, 'cache_ttl', None)
    site_url = getattr(web, 'site_url', None)

    try:
        cache_ttl = float(ttl) if ttl not in (None, '') else float(DEFAULT_TTL)
    except (TypeError, ValueError):
        warnings.warn('invalid web.cache_ttl {!r}; using {}'.format(ttl, DEFAULT_TTL))
        cache_ttl = float(DEFAULT_TTL)

    return {
        'prefix': str(prefix) if prefix not in (None, '') else bundle.PREFIX,
        'cache_ttl': cache_ttl,
        'site_url': str(site_url).rstrip('/') if site_url not in (None, '') else DEFAULT_SITE_URL,
    }


class TTLCache:
    """
    Cache whose entries expire a fixed time after being stored, optionally capped in size.

    An expired entry is replaced when its key is asked again; with ``maxsize`` the least
    recently used entry is evicted on insert, so keys shaped by requests cannot grow the
    cache without bound.

    Parameters
    ----------
    ttl : float
        Seconds an entry stays valid.
    clock : callable, optional
        Function returning the current time in seconds. Defaults to ``time.monotonic``.
    maxsize : int, optional
        Maximum number of entries; ``None`` (default) for no cap.
    """

    def __init__(self, ttl: float, clock: Callable[[], float] = time.monotonic, maxsize: Optional[int] = None):
        self.ttl = ttl
        self.clock = clock
        self.maxsize = maxsize
        self._items = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        """
        Number of stored entries, expired ones included.

        Returns
        -------
        int
            Entries currently stored.
        """
        return len(self._items)

    def get(self, key: Any, loader: Callable[[], Any]) -> Any:
        """
        Return the cached value or load, store and return a fresh one.

        Parameters
        ----------
        key : hashable
            Cache key.
        loader : callable
            Function without arguments that produces the value on a miss.

        Returns
        -------
        object
            The cached or freshly loaded value.

        Raises
        ------
        Exception
            Whatever ``loader`` raises; nothing is stored in that case.
        """
        with self._lock:
            hit = self._items.get(key)

            if hit is not None and self.clock() - hit[0] < self.ttl:
                self._items.move_to_end(key)

                return hit[1]

        value = loader()

        with self._lock:
            self._items[key] = (self.clock(), value)
            self._items.move_to_end(key)

            while self.maxsize is not None and len(self._items) > self.maxsize:
                self._items.popitem(last=False)

        return value

    def clear(self) -> None:
        """
        Drop every entry.

        Returns
        -------
        None
        """
        with self._lock:
            self._items.clear()


class LRUCache:
    """
    Cache that keeps the most recently used entries.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of entries; the least recently used is evicted beyond it.
    """

    def __init__(self, maxsize: int = LRU_SIZE):
        self.maxsize = maxsize
        self._items = OrderedDict()
        self._lock = Lock()

    def __len__(self) -> int:
        """
        Number of cached entries.

        Returns
        -------
        int
            Entries currently stored.
        """
        return len(self._items)

    def get(self, key: Any, loader: Callable[[], Any]) -> Any:
        """
        Return the cached value or load, store and return a fresh one.

        Parameters
        ----------
        key : hashable
            Cache key.
        loader : callable
            Function without arguments that produces the value on a miss.

        Returns
        -------
        object
            The cached or freshly loaded value.

        Raises
        ------
        Exception
            Whatever ``loader`` raises; nothing is stored in that case.
        """
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)

                return self._items[key]

        value = loader()

        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)

            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

        return value

    def clear(self) -> None:
        """
        Drop every entry.

        Returns
        -------
        None
        """
        with self._lock:
            self._items.clear()
