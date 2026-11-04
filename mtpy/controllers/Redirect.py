"""
Controller of the old page URLs: those before phase 7 (``/promedio``, ``/escanos``, ``/modelo``) and the seats page of
a scope (``/{scope}/escanos``, merged into the event page on 2026-11-04). Permanent redirections.
"""
from urllib.parse import urlencode

from ..lib import pages, webapi
from .Page import Page


class Redirect(Page):
    """
    Old page URLs: 301 to the event page of a scope (to its seats sections for the seats pages) and to the backtest
    subpage of the method.
    """

    active = None

    def query_scope(self) -> str:
        """
        Scope of an old URL that carries it in the query.

        Returns
        -------
        str
            The ``scope`` of the query; the first published scope when the query has none.

        Raises
        ------
        HttpError
            400 for an invalid scope; 404 for an unpublished one; 503 without a bundle.
        """
        scope = self.request.query.get('scope') or None

        return pages.resolve_scope(webapi.check_scope(scope) if scope is not None else None)

    def seats_location(self, scope: str) -> str:
        """
        Target of an old seats page: the event page of ``scope``, scrolled to its seats.

        The query is kept but ``scope`` and ``seccion``, and ``seccion`` goes last: ``territorio`` with a ``region``
        (the detail of a district), ``escanos`` otherwise. The location has no fragment, so the browser keeps the one
        of the link (``#coalition=PP,VOX``); the page scrolls to ``seccion`` and drops it (``url-state.js``).

        Parameters
        ----------
        scope : str
            Scope of the event page, already resolved.

        Returns
        -------
        str
            ``'/{scope}?...&seccion=...'``.
        """
        section = 'territorio' if self.request.query.get('region') else 'escanos'
        params = [(key, value) for key, value in self.request.query.items() if key not in ('scope', 'seccion')]

        return '/' + scope + '?' + urlencode(params + [('seccion', section)])

    async def promedio_action(self) -> str:
        """
        Redirect ``/promedio`` to ``/{scope}``.

        Returns
        -------
        str
            The empty body of the 301.
        """
        return self.redirect('/' + self.query_scope() + self.query_without('scope'))

    async def escanos_action(self) -> str:
        """
        Redirect ``/escanos`` straight to the seats of the event page (``seats_location``), not through
        ``/{scope}/escanos``: one 301, not two.

        Returns
        -------
        str
            The empty body of the 301.
        """
        return self.redirect(self.seats_location(self.query_scope()))

    async def seats_action(self, scope: str) -> str:
        """
        Redirect the seats page of a scope (``/{scope}/escanos``) to the seats of its event page (``seats_location``).

        Parameters
        ----------
        scope : str
            Scope of the route.

        Returns
        -------
        str
            The empty body of the 301.

        Raises
        ------
        HttpError
            404 for a scope outside the catalogue or without a published forecast (``pages.resolve_event_scope``);
            503 without a bundle.
        """
        return self.redirect(self.seats_location(pages.resolve_event_scope(scope)))

    async def modelo_action(self) -> str:
        """
        Redirect ``/modelo`` to ``/metodo/backtest``, keeping ``scope`` and dropping ``mode`` and ``run``.

        Returns
        -------
        str
            The empty body of the 301.
        """
        return self.redirect('/metodo/backtest' + self.query_without('mode', 'run'))
