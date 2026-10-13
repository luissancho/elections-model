"""
Controller of the method page (``/metodo``) and of its subpages (``/metodo/{section}``).
"""
from ..core.api import HttpError
from ..lib import pages
from .Page import Page


class Metodo(Page):
    """
    Method page: a static, non-specialist explanation of how the model works, and its subpages.
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

    async def section_action(self, section: str) -> str:
        """
        Render a subpage of the method.

        Parameters
        ----------
        section : str
            Id of the subpage (a key of ``pages.METODO_SECTIONS``).

        Returns
        -------
        str
            The HTML of the template of the subpage (``metodo_backtest.html`` for ``backtest``).

        Raises
        ------
        HttpError
            404 ``section not found`` for an unknown subpage or one without text yet.
        """
        entry = pages.METODO_SECTIONS.get(section)

        if entry is None or entry['template'] is None:
            raise HttpError(404, 'section not found')

        # only the backtest has a text so far: the other subpages bring their context with their texts
        return self.render(entry['template'], **pages.backtest_context(self.state(), active=self.active))
