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
    IMAGE_TTL : int
        Seconds a cache may keep the image of the link previews of a page (``png``) under the address the page
        declares: the pages that have one do not change, and when what they draw is corrected the page declares
        another address (the ``?v=`` of its ``og:image``). Under any other address the image is kept for the time
        of the pages.
    """

    active = '/'
    IMAGE_TTL = 86400

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

    def canonical_url(self) -> str:
        """
        Canonical URL of the page of the request: the public origin and the path, without the query.

        No parameter gives a page an address of its own: ``mode``, ``run``, ``region`` and ``seccion`` change the
        view of an event; ``scope`` sets the filter and the links of a global page and, in ``/casas`` and
        ``/metodo/backtest``, the cards of that scope, which stay views of the national page; the rest come from
        outside (``utm_*``, ``fbclid``, ``gclid``). One page, one address.

        Returns
        -------
        str
            ``'<site_url>/<path>'``, for ``<link rel="canonical">`` and ``og:url``.
        """
        return '{}/{}'.format(webapi.settings()['site_url'], self.request.uri)

    def section_trail(self, href: str, name: str) -> dict:
        """
        Breadcrumbs of a page of the second level: its section of the menu, then the page.

        Parameters
        ----------
        href : str
            ``href`` of the section in ``pages.PAGES`` (``'/metodo'``).
        name : str
            Name of the page, as the page shows it.

        Returns
        -------
        dict
            The ``BreadcrumbList`` block (``pages.structured.breadcrumbs``): the label of the section in the menu
            with its URL, and ``name`` without one (the page itself).
        """
        label = next(page['label'] for page in pages.PAGES if page['href'] == href)

        return pages.structured.breadcrumbs(webapi.settings()['site_url'], [(label, href), (name, None)])

    def render(self, template: str, **context) -> str:
        """
        Render a page template as a cacheable HTML response.

        Sets ``text/html``, an ETag (MD5 of the UTF-8 bytes) and ``public, max-age=<WEB_CACHE_TTL>`` (60 by default).

        Parameters
        ----------
        template : str
            Template name in ``web/templates``.
        **context
            Template variables; ``canonical`` (``canonical_url``), ``og_image`` (``pages.og_image_url``, the
            brand image of the link previews), ``og_image_alt`` (``None``: the text alternative of the brand
            image; a page with an image of its own passes both), ``analytics`` (``pages.analytics_config``, the
            settings of the cookie notice, ``None`` when the visits are not measured) and ``structured`` (the
            schema.org blocks of the page, ``pages.structured``; none by default) are added when missing.

        Returns
        -------
        str
            The rendered HTML.
        """
        context.setdefault('canonical', self.canonical_url())
        context.setdefault('og_image', pages.og_image_url())
        context.setdefault('og_image_alt', None)
        context.setdefault('analytics', pages.analytics_config())
        context.setdefault('structured', [])
        html = pages.environment().get_template(template).render(**context)

        self.response.set_content_type('text/html')
        self.response.set_etag(hashlib.md5(html.encode('utf-8')).hexdigest())
        self.response.set_cache(int(webapi.settings()['cache_ttl']))

        return html

    def preview_image(self, href: str, preview: Optional[dict]) -> dict:
        """
        The image of its own a page declares for its link previews.

        Parameters
        ----------
        href : str
            Canonical path of the page (``'/es/sondeos/2026-11-05/gad3'``); its image is that address ending in
            ``.png``.
        preview : dict or None
            What the image says (``pages.preview_of``, ``pages.result_preview``), with its ``version`` and its
            ``alt``; ``None`` when there is nothing to draw.

        Returns
        -------
        dict
            The ``og_image`` (absolute, with the fingerprint of what it draws in ``?v=``: the networks keep an
            image by its address) and ``og_image_alt`` of ``render``. Empty without a preview or when this process
            cannot draw (``pages.images.available``): the page keeps the brand image.
        """
        if preview is None or not pages.images.available():
            return {}

        return {'og_image': '{}{}.png?v={}'.format(webapi.settings()['site_url'], href, preview['version']),
                'og_image_alt': preview['alt']}

    def png(self, image: bytes, version: str) -> bytes:
        """
        Answer an image the server draws for the link previews of a page.

        Sets ``image/png``, an ETag (MD5 of the bytes) and how long a cache may keep it: ``IMAGE_TTL`` only when
        the ``v`` of the request is the fingerprint of what was drawn, which is the address the page declares.
        Any other address (another ``v``, an empty one or none) does not name this drawing: for a minute after a
        publication a worker may still draw the previous run under the new address, and an old link keeps its old
        ``v`` after a correction. Those get the time of the pages (``cache_ttl``), so a drawing never outlives a
        page in a cache under an address that names another one.

        Parameters
        ----------
        image : bytes
            The PNG (``pages.images``).
        version : str
            Fingerprint of what the image draws (the ``version`` of its preview).

        Returns
        -------
        bytes
            The same bytes.
        """
        named = self.request.query.get('v') == version
        self.response.set_content_type('image/png', charset=None)
        self.response.set_etag(hashlib.md5(image).hexdigest())
        self.response.set_cache(self.IMAGE_TTL if named else int(webapi.settings()['cache_ttl']))

        return image

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
            site_title=pages.SITE_TITLE, site_tagline=pages.SITE_TAGLINE, status=status, message=text
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
