"""
Controller of the addresses of the page «Escaños que bailan» that there was
(``/{scope}/circunscripciones[/{district}]``, retired on 2026-11-13): they lead to its card on the event page.
"""
from ..lib import pages
from .Page import Page

# the parameters of the page that was, which the event page does not read: the party of its card of a party, its tabs
# and its dropdown of districts
RETIRED_PARAMS = ('partido', 'vista', 'circunscripcion')


class Circunscripciones(Page):
    """
    The addresses of the page «Escaños que bailan», which the card of the same name of the event page replaced: a 301
    to that card, keeping the mode and the pinned run.
    """

    active = None

    def card_location(self, scope: str) -> str:
        """
        Target of an address of the page that was: the card «Escaños que bailan» of the event page of ``scope``.

        The card opens on its own district (``default_district``): the district of the address is not kept, and neither
        are the parameters of ``RETIRED_PARAMS``; the rest of the query (``mode``, ``run``) is. The anchor is the
        card's own (``seat-odds``), not the one of its section, which opens with the territory table. The target does
        not depend on whether the run shown has the card: the event page says what it has.

        Parameters
        ----------
        scope : str
            Scope of the event page, already resolved.

        Returns
        -------
        str
            ``'/{scope}{query}#seat-odds'``.
        """
        return '/{}{}#seat-odds'.format(scope, self.query_without(*RETIRED_PARAMS))

    async def index_action(self, scope: str) -> str:
        """
        Redirect ``/{scope}/circunscripciones`` to the card of the event page (``card_location``).

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
        return self.redirect(self.card_location(pages.resolve_event_scope(scope)))

    async def detail_action(self, scope: str, district: str) -> str:
        """
        Redirect ``/{scope}/circunscripciones/{district}`` to the card of the event page (``card_location``), whatever
        the district.

        Parameters
        ----------
        scope : str
            Scope of the route.
        district : str
            Slug of the district of the address; not kept.

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
        return self.redirect(self.card_location(pages.resolve_event_scope(scope)))
