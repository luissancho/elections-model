"""
Controller of the past elections page (``/elecciones``), of the detail of one
(``/elecciones/{country}/{scope}/{month}``) and of the image of its link previews (the same address ending in
``.png``).
"""
from ..lib import pages, webapi
from .Page import Page


class Elecciones(Page):
    """
    Past elections page: the error history of the published elections; ``/elecciones/es/generales/2023-07``, one
    election, and ``/elecciones/es/generales/2023-07.png`` the image a network shows when that page is shared
    (``image_action``), which the page declares as its ``og:image``.
    """

    active = '/elecciones'

    async def index_action(self) -> str:
        """
        Render the list of elections; with the old ``date`` parameter, redirect to the detail of that election.

        Returns
        -------
        str
            The HTML of ``elecciones.html``, or the empty body of a 301 to ``pages.event_href`` (the rest of the
            query kept) when ``date`` is given: the election of the ``scope`` of the query, or of the first
            published scope, validated by ``pages.resolve_event`` (400 invalid, 404 unpublished).
        """
        date = self.request.query.get('date')

        if date is not None:
            scope = pages.resolve_scope(self.state()['scope'])
            date = pages.resolve_event(scope, date)

            return self.redirect(pages.event_href(scope, date) + self.query_without('scope', 'date'))

        return self.render('elecciones.html', **pages.elecciones_context(self.state(), active=self.active))

    async def detail_action(self, country: str, scope: str, date: str) -> str:
        """
        Render the detail of a past election.

        Parameters
        ----------
        country : str
            Country segment of the route (``es``).
        scope : str
            Scope segment (``generales``, ``madrid``).
        date : str
            Month (``2023-07``) or date of the election (``pages.resolve_event_path``).

        Returns
        -------
        str
            The HTML of ``eleccion.html``, with its breadcrumbs (the election by its name or, without one, by its
            date, like its title); its canonical is the month URL (``pages.event_href``), whatever the segment and
            the query. Its ``og:image`` is the image of the election (``image_action``, ``preview_image``): the
            canonical ending in ``.png``, with the fingerprint of what it draws and a text alternative with its
            figures. The image always draws with the latest run, so a page with a pinned run declares the
            fingerprint of the latest one.
        """
        code, date = pages.resolve_event_path(country, scope, date)
        context = pages.eleccion_context(self.state(code), date, active=self.active)
        href = pages.event_href(code, date)
        event = context['event']
        trail = self.section_trail(self.active, event.get('name') or pages.fmt_date(event['event_date']))
        preview = context['preview']

        if context['state']['pinned']:
            preview = pages.latest_event_preview(code, date)

        return self.render('eleccion.html', canonical=webapi.settings()['site_url'] + href, structured=[trail],
                           **self.preview_image(href, preview), **context)

    async def image_action(self, country: str, scope: str, date: str) -> bytes:
        """
        Draw the image of the link previews of a past election.

        It is always drawn with the latest published run (the colours of the parties): the query (``mode``,
        ``run``, the ``v`` of the address the page declares, anything else) is not read.

        Parameters
        ----------
        country : str
            Country segment of the route (``es``).
        scope : str
            Scope segment (``generales``, ``madrid``).
        date : str
            Month (``2023-07``) or date of the election (``pages.resolve_event_path``).

        Returns
        -------
        bytes
            The PNG (``pages.images.result_png`` of ``pages.event_preview``), answered as a cacheable image
            (``png``: a day under the fingerprint of what it draws, the time of the pages under any other address).

        Raises
        ------
        HttpError
            404 for an unknown scope or election and 400 for an invalid date, as the page; nothing is drawn then.
        """
        code, date = pages.resolve_event_path(country, scope, date)

        preview = pages.event_preview(pages.parse_state({}, code), date)

        return self.png(pages.images.result_png(preview), preview['version'])
