"""
Controller of the page «Escaños que bailan» (``/{scope}/circunscripciones[/{district}]``).
"""
from typing import Optional
from urllib.parse import quote

from ..lib import pages
from .Page import Page


class Circunscripciones(Page):
    """
    The seats of every district, one by one: which ones each party has safe and which ones are open, with the share
    of the simulations in which it wins them. A subpage of the event: it follows its mode and its pinned run.
    """

    active = None

    def page(self, scope: str, district: Optional[str] = None) -> str:
        """
        Render the page for a scope already resolved.

        Parameters
        ----------
        scope : str
            Scope code.
        district : str, optional
            Slug of the district shown; the first of the list without it.

        Returns
        -------
        str
            The HTML of ``circunscripciones.html``; its canonical drops ``mode`` and ``run`` (``canonical_url``).
        """
        context = pages.seat_odds_context(self.state(scope), district)

        return self.render('circunscripciones.html', **context)

    async def index_action(self, scope: str) -> str:
        """
        Render the page on the district with most open seats, or send the dropdown of districts to the address of
        the one chosen.

        The dropdown (``#district-pick``) submits ``circunscripcion`` with the slug of a district: without
        JavaScript the browser lands here, and the answer is a 301 to ``/{scope}/circunscripciones/{slug}`` with the
        rest of the query (``mode``, ``run``).

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).

        Returns
        -------
        str
            The HTML of the page, or the empty body of the redirection.

        Raises
        ------
        HttpError
            404 for an unknown scope or a run without the ``seat-odds`` part (or with a single district); 400 for an
            invalid ``mode`` or ``run``.
        """
        code = pages.resolve_event_scope(scope)
        wanted = self.request.query.get('circunscripcion')

        if wanted:
            return self.redirect('/{}/circunscripciones/{}{}'.format(code, quote(wanted, safe=''),
                                                                     self.query_without('circunscripcion')))

        return self.page(code)

    async def detail_action(self, scope: str, district: str) -> str:
        """
        Render the page on one district.

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).
        district : str
            Slug of the district.

        Returns
        -------
        str
            The HTML of the page.

        Raises
        ------
        HttpError
            404 for an unknown scope or district, or a run without the ``seat-odds`` part (or with a single
            district); 400 for an invalid ``mode`` or ``run``.
        """
        return self.page(pages.resolve_event_scope(scope), district)
