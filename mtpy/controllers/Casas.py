"""
Controller of the houses page (``/casas``) and of the profile of a pollster (``/casas?id=N``).
"""
from ..lib import pages, webapi
from .Page import Page


class Casas(Page):
    """
    Houses page: pollsters ranking, house effects and herding; with ``?id=``, the profile of a pollster.
    """

    active = '/casas'

    async def index_action(self) -> str:
        """
        Render the houses page, or the profile of a pollster when ``id`` is given.

        Returns
        -------
        str
            The HTML of ``casa.html`` with ``id`` (validated by ``webapi.check_pollster``),
            otherwise of ``casas.html``.
        """
        pollster = self.request.query.get('id')

        if pollster is not None:
            pollster_id = webapi.check_pollster(pollster)
            context = pages.casa_context(self.state(), pollster_id, active=self.active)

            return self.render('casa.html', **context)

        return self.render('casas.html', **pages.casas_context(self.state(), active=self.active))
