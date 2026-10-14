import hashlib

from ..core.api import Controller, HttpError
from ..lib.webapi import IMMUTABLE_MAX_AGE, settings, site


class Base(Controller):
    """
    Common behaviour of the ``/api/v1`` controllers: CORS, cache headers and JSON/CSV bytes.
    """

    def before_dispatch(self) -> None:
        """
        Allow cross-origin reads on every response, errors included.

        Returns
        -------
        None
        """
        self.response.set_header('access-control-allow-origin', '*')

    @property
    def query(self) -> dict:
        """
        Query string parameters of the request.

        Returns
        -------
        dict
            Parameter names and values.
        """
        return self.request.query

    def cache(self, immutable: bool = False) -> None:
        """
        Set the cache headers of the response.

        Parameters
        ----------
        immutable : bool, optional
            True for content addressed by an explicit run, cacheable for a year.

        Returns
        -------
        None
        """
        if immutable:
            self.response.set_cache(IMMUTABLE_MAX_AGE, immutable=True)
        else:
            self.response.set_cache(int(settings()['cache_ttl']))

    def no_store(self) -> None:
        """
        Forbid caching of the response.

        Returns
        -------
        None
        """
        self.response.set_header('cache-control', 'no-store')

    def json_bytes(self, content: bytes, immutable: bool = False) -> bytes:
        """
        Serve JSON bytes untouched, with content type, ETag and cache headers.

        Parameters
        ----------
        content : bytes
            Serialised JSON.
        immutable : bool, optional
            Whether the content is addressed by an explicit run.

        Returns
        -------
        bytes
            The same content.
        """
        self.response.set_content_type('application/json')
        self.response.set_etag(hashlib.md5(content).hexdigest())
        self.cache(immutable)

        return content

    def csv_bytes(self, content: bytes, filename: str, immutable: bool = False) -> bytes:
        """
        Serve CSV bytes as a download, with ETag and cache headers.

        Parameters
        ----------
        content : bytes
            CSV file.
        filename : str
            Name proposed to the client.
        immutable : bool, optional
            Whether the content is addressed by an explicit run.

        Returns
        -------
        bytes
            The same content.
        """
        self.response.set_content_type('text/csv')
        self.response.set_header('content-disposition', 'attachment; filename="{}"'.format(filename))
        self.response.set_etag(hashlib.md5(content).hexdigest())
        self.cache(immutable)

        return content

    def notice_freeze(self) -> None:
        """
        Flag the response with ``x-freeze: active`` while the publication is frozen.

        Without a manifest there is nothing to flag, so its error is ignored here.

        Returns
        -------
        None
        """
        try:
            active = site().freeze().get('active')
        except HttpError:
            return

        if active:
            self.response.set_header('x-freeze', 'active')
