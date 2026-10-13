"""
Controller of the Escaños tab of an event (``/{scope}/escanos``).
"""
from ..lib import pages
from .Page import Page


class Escanos(Page):
    """
    Seats page: party and block tables, seat distributions, coalition calculator, fan and downloads.
    """

    active = None

    async def index_action(self, scope: str) -> str:
        """
        Render the Escaños tab of the event of a scope.

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).

        Returns
        -------
        str
            The HTML of ``escanos.html``.
        """
        scope = pages.resolve_event_scope(scope)
        context = pages.escanos_context(self.state(scope), region=self.request.query.get('region'),
                                        active=self.active, event=True, active_suffix='/escanos')

        return self.render('escanos.html', **context)
