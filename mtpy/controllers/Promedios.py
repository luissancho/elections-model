"""
Controller of the event page of a scope (``/{scope}``, «Encuestas»): the Promedios and the Escaños in one page.
"""
from ..lib import pages
from .Page import Page


class Promedios(Page):
    """
    Event page of a scope: the poll average, the hemicycle and the results table, the coalition calculator, the
    evolution of the updates, one house against the average, the districts and the polls of the cycle, by sections.
    """

    active = None

    async def index_action(self, scope: str) -> str:
        """
        Render the event page of a scope.

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).

        Returns
        -------
        str
            The HTML of ``promedios.html``; the ``region`` query parameter chooses the district of the detail table
            (400 when invalid, 404 when unknown).
        """
        scope = pages.resolve_event_scope(scope)
        context = pages.promedios_context(self.state(scope), region=self.request.query.get('region'))

        return self.render('promedios.html', **context)
