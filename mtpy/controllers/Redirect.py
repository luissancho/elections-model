"""
Controller of the URLs of the pages before phase 7 (``/promedio``, ``/escanos``, ``/modelo``): permanent redirections.
"""
from ..lib import pages, webapi
from .Page import Page


class Redirect(Page):
    """
    Old page URLs: 301 to the event pages of a scope and to the backtest subpage of the method.
    """

    active = None

    def event_location(self, suffix: str) -> str:
        """
        Target of an old event page: the route of its scope, keeping the rest of the query.

        Parameters
        ----------
        suffix : str
            Suffix of the event tab (``''`` or ``'/escanos'``).

        Returns
        -------
        str
            ``'/{scope}{suffix}'`` plus the query without ``scope``; the first published scope
            when the query has none.

        Raises
        ------
        HttpError
            400 for an invalid scope; 404 for an unpublished one; 503 without a bundle.
        """
        scope = self.request.query.get('scope') or None
        scope = pages.resolve_scope(webapi.check_scope(scope) if scope is not None else None)

        return '/' + scope + suffix + self.query_without('scope')

    async def promedio_action(self) -> str:
        """
        Redirect ``/promedio`` to ``/{scope}``.

        Returns
        -------
        str
            The empty body of the 301.
        """
        return self.redirect(self.event_location(''))

    async def escanos_action(self) -> str:
        """
        Redirect ``/escanos`` to ``/{scope}/escanos`` (``region`` included).

        Returns
        -------
        str
            The empty body of the 301.
        """
        return self.redirect(self.event_location('/escanos'))

    async def modelo_action(self) -> str:
        """
        Redirect ``/modelo`` to ``/metodo/backtest``, keeping ``scope`` and dropping ``mode`` and ``run``.

        Returns
        -------
        str
            The empty body of the 301.
        """
        return self.redirect('/metodo/backtest' + self.query_without('mode', 'run'))
