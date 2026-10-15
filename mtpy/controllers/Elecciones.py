"""
Controller of the past elections page (``/elecciones``) and of the detail of one
(``/elecciones/{country}/{scope}/{month}``).
"""
from ..lib import pages, webapi
from .Page import Page


class Elecciones(Page):
    """
    Past elections page: the error history of the published elections; ``/elecciones/es/generales/2023-07``, one
    election.
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
            The HTML of ``eleccion.html``; its canonical is the month URL (``pages.event_href``), whatever the
            segment and the query.
        """
        code, date = pages.resolve_event_path(country, scope, date)
        context = pages.eleccion_context(self.state(code), date, active=self.active)

        canonical = webapi.settings()['site_url'] + pages.event_href(code, date)

        return self.render('eleccion.html', canonical=canonical, **context)
