"""
Controller of the Promedios tab of an event (``/{scope}``).
"""
from ..lib import pages
from .Page import Page


class Promedios(Page):
    """
    Promedios tab of the event of a scope: the poll average, headline, vote, seats, majorities, evolution, one
    house against the average and the polls of the cycle.
    """

    active = None

    async def index_action(self, scope: str) -> str:
        """
        Render the Promedios tab of the event of a scope.

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).

        Returns
        -------
        str
            The HTML of ``promedios.html``.
        """
        scope = pages.resolve_event_scope(scope)
        context = pages.promedios_context(self.state(scope))

        return self.render('promedios.html', **context)
