"""
Controller of the seats page (``/escanos``).
"""
from ..lib import pages
from .Page import Page


class Escanos(Page):
    """
    Seats page: party and block tables, seat distributions, coalition calculator, fan and downloads.
    """

    active = '/escanos'

    async def index_action(self) -> str:
        """
        Render the seats page.

        Returns
        -------
        str
            The HTML of ``escanos.html``.
        """
        context = pages.escanos_context(self.state(), region=self.request.query.get('region'),
                                        active=self.active)

        return self.render('escanos.html', **context)
