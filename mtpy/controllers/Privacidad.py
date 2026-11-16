"""
Controller of the privacy page (``/privacidad``).
"""
from ..lib import pages
from .Page import Page


class Privacidad(Page):
    """
    Privacy page: who answers for the site, what it keeps of its readers, the cookies of the measurement of visits
    and how to accept or reject them.
    """

    active = None

    async def index_action(self) -> str:
        """
        Render the privacy page.

        Returns
        -------
        str
            The HTML of ``privacidad.html``.
        """
        return self.render('privacidad.html', **pages.privacidad_context(self.state()))
