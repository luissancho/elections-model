"""
Controller of the Promedios tab of an event (``/{scope}``).
"""
from ..lib import pages
from .Page import Page


class Promedios(Page):
    """
    Promedios tab of the event of a scope: headline, vote, seats, majorities and evolution.
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
        context = pages.index_context(self.state(scope), active=self.active, event=True)

        return self.render('promedios.html', **context)
