"""
Controller of the home page (``/``).
"""
from ..lib import pages
from .Page import Page


class Index(Page):
    """
    Home page: headline table, charts of the vote, the seats and the majorities, and evolution.
    """

    active = '/'

    async def index_action(self) -> str:
        """
        Render the home page.

        Returns
        -------
        str
            The HTML of ``index.html``.
        """
        return self.render('index.html', **pages.index_context(self.state(), active=self.active))
