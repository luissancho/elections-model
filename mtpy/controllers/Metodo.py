"""
Controller of the method page (``/metodo``).
"""
from ..lib import pages
from .Page import Page


class Metodo(Page):
    """
    Method page: a static, non-specialist explanation of how the model works.
    """

    active = '/metodo'

    async def index_action(self) -> str:
        """
        Render the explanation of the method.

        Returns
        -------
        str
            The HTML of ``metodo.html``.
        """
        return self.render('metodo.html', **pages.metodo_context(self.state(), active=self.active))
