"""
Controller of the poll average page (``/promedio``).
"""
from ..lib import pages
from .Page import Page


class Promedio(Page):
    """
    Poll average page: polls, average and projection chart, and the table of the latest polls.
    """

    active = '/promedio'

    async def index_action(self) -> str:
        """
        Render the poll average page.

        Returns
        -------
        str
            The HTML of ``promedio.html``.
        """
        return self.render('promedio.html', **pages.promedio_context(self.state(), active=self.active))
