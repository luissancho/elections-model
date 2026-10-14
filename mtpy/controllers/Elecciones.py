"""
Controller of the past elections page (``/elecciones``) and of the detail of one (``/elecciones?date=``).
"""
from ..lib import pages
from .Page import Page


class Elecciones(Page):
    """
    Past elections page: the error history of the published elections; with ``?date=``, one election.
    """

    active = '/elecciones'

    async def index_action(self) -> str:
        """
        Render the list of elections, or the detail of one when ``date`` is given.

        Returns
        -------
        str
            The HTML of ``eleccion.html`` with ``date`` (validated by ``pages.resolve_event``),
            otherwise of ``elecciones.html``.
        """
        date = self.request.query.get('date')

        if date is not None:
            context = pages.eleccion_context(self.state(), date, active=self.active)

            return self.render('eleccion.html', **context)

        return self.render('elecciones.html', **pages.elecciones_context(self.state(), active=self.active))
