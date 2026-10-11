"""
Controller of the model accuracy page (``/modelo``).
"""
from ..lib import pages
from .Page import Page


class Modelo(Page):
    """
    Model page: the accuracy of the model by horizon, from the published backtest.
    """

    active = '/modelo'

    async def index_action(self) -> str:
        """
        Render the backtest of the scope.

        Returns
        -------
        str
            The HTML of ``modelo.html``; a notice when the scope has no published backtest.
        """
        return self.render('modelo.html', **pages.modelo_context(self.state(), active=self.active))
