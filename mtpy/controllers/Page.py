"""
Base controller of the server-rendered web pages (Jinja2 templates of ``web/templates``).
"""
import hashlib
import traceback

from ..core.api import Controller, HttpError
from ..lib import pages, webapi


class Page(Controller):
    """
    Common behaviour of the web pages: Jinja2 rendering, cache headers and HTML errors.

    Attributes
    ----------
    active : str
        ``href`` of the page in the navigation; subclasses set it.
    """

    active = '/'

    def state(self) -> dict:
        """
        Validated page parameters of the request.

        Returns
        -------
        dict
            ``{'scope', 'mode', 'run'}`` (see ``pages.parse_state``).

        Raises
        ------
        HttpError
            400 when a given parameter is invalid.
        """
        return pages.parse_state(self.request.query)

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

    async def dispatch(self, action, **kwargs):
        """
        Run an action and send its HTML; errors become HTML error pages.

        An ``HttpError`` gives the error page with its status and message; any other
        exception is logged with its traceback and gives a 500 error page.

        Parameters
        ----------
        action : str
            Action name, without the ``_action`` suffix.
        **kwargs
            Route parameters passed to the action.

        Returns
        -------
        None
        """
        self.action = action
        self.params = dict(kwargs)

        try:
            self.before_dispatch()

            self.result = await getattr(self, action + '_action')(**self.params)

            self.after_dispatch()
        except HttpError as error:
            self.result = self.error_page(error.status, error.message)
        except Exception as error:
            if self.app.logger is not None:
                self.app.logger.error('Unhandled error in {}.{}: {!r}\n{}'.format(
                    type(self).__name__, action, error, traceback.format_exc()
                ))
            self.result = self.error_page(500, 'Error interno')

        await self.send()

    async def not_found_action(self) -> str:
        """
        Page for an unknown path.

        Returns
        -------
        str
            The 404 error page.
        """
        return self.error_page(404, 'Página no encontrada.')
