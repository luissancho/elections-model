"""
Base controller of the server-rendered web pages (Jinja2 templates of ``web/templates``).
"""
import hashlib
from typing import Optional
from urllib.parse import urlencode

from ..core.api import Controller
from ..lib import pages, webapi


class Page(Controller):
    """
    Common behaviour of the web pages: Jinja2 rendering, cache headers and HTML errors.

    Attributes
    ----------
    active : str or None
        ``href`` of the page in the navigation; subclasses set it (``None`` for the event pages).
    """

    active = '/'

    def state(self, scope: Optional[str] = None) -> dict:
        """
        Validated page parameters of the request.

        Parameters
        ----------
        scope : str, optional
            Scope of the route of an event page, which wins over the ``scope`` of the query.

        Returns
        -------
        dict
            ``{'scope', 'mode', 'run'}`` (see ``pages.parse_state``).

        Raises
        ------
        HttpError
            400 when a given parameter is invalid.
        """
        return pages.parse_state(self.request.query, scope)

    def query_without(self, *names: str) -> str:
        """
        Query string of the request without some parameters, for a redirection.

        Parameters
        ----------
        *names : str
            Parameters to drop.

        Returns
        -------
        str
            ``'?key=value&...'`` with the other parameters in their order, or ``''`` when none is left.
        """
        params = [(key, value) for key, value in self.request.query.items() if key not in names]

        return '?' + urlencode(params) if params else ''

    def redirect(self, location: str) -> str:
        """
        Answer a permanent redirection.

        Sets the status 301, the ``location`` header and ``no-store`` (the target of an old URL
        may change with the published scopes); the body is empty.

        Parameters
        ----------
        location : str
            Target path, with its query string.

        Returns
        -------
        str
            The empty body.
        """
        self.response.set_status_code(301)
        self.response.set_header('location', location)
        self.response.set_header('cache-control', 'no-store')

        return ''

    def render(self, template: str, **context) -> str:
        """
        Render a page template as a cacheable HTML response.

        Sets ``text/html``, an ETag (MD5 of the UTF-8 bytes) and ``public, max-age=<WEB_CACHE_TTL>`` (60 by default).

        Parameters
        ----------
        template : str
            Template name in ``web/templates``.
        **context
            Template variables.

        Returns
        -------
        str
            The rendered HTML.
        """
        html = pages.environment().get_template(template).render(**context)

        self.response.set_content_type('text/html')
        self.response.set_etag(hashlib.md5(html.encode('utf-8')).hexdigest())
        self.response.set_cache(int(webapi.settings()['cache_ttl']))

        return html

    def error_page(self, status: int, message: str) -> str:
        """
        Render the error page as an uncacheable HTML response.

        Sets the status code, ``text/html`` and ``no-store``, and drops the ETag and the
        ``content-disposition`` that the action may have set.

        Parameters
        ----------
        status : int
            HTTP status code.
        message : str
            Error message; a key of ``pages.MESSAGES`` is shown in Spanish, an unknown
            message of a 5xx error as a generic one, any other as is.

        Returns
        -------
        str
            The rendered HTML.
        """
        if message in pages.MESSAGES:
            text = pages.MESSAGES[message]
        elif status >= 500:
            text = 'Error interno.'
        else:
            text = message

        self.response.set_status_code(status)
        self.response.set_content_type('text/html')
        self.response.set_header('cache-control', 'no-store')
        self.response.set_header('etag', None)
        self.response.set_header('content-disposition', None)

        return pages.environment().get_template('error.html').render(
            site_title=pages.SITE_TITLE, status=status, message=text
        )

    def error_result(self, status: int, message: str) -> str:
        """
        Render an error of an action (``HttpError`` or unexpected exception) as an HTML page.

        ``Controller.dispatch`` calls it; an unexpected exception arrives as a 500 with a
        generic message, which the error page shows as ``Error interno.``.

        Parameters
        ----------
        status : int
            HTTP status code.
        message : str
            Error message (see ``error_page``).

        Returns
        -------
        str
            The rendered error page.
        """
        return self.error_page(status, message)

    async def not_found_action(self) -> str:
        """
        Page for an unknown path.

        Returns
        -------
        str
            The 404 error page.
        """
        return self.error_page(404, 'Página no encontrada.')
