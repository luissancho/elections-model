"""
Controller of the page «Escaños que bailan» (``/{scope}/circunscripciones[/{district}]``).
"""
from typing import Optional
from urllib.parse import quote

from ..lib import pages
from .Page import Page


class Circunscripciones(Page):
    """
    The seats of every district, one by one: which ones each party has safe, which ones are likely and which ones
    are in play, with the share of the simulations in which it wins them. A subpage of the event: it follows its mode
    and its pinned run.
    """

    active = None

    def page(self, scope: str, district: Optional[str] = None, party: Optional[str] = None,
             view: Optional[str] = None) -> str:
        """
        Render the page for a scope already resolved.

        Parameters
        ----------
        scope : str
            Scope code.
        district : str, optional
            Slug of the district shown; it wins over ``party`` and ``view``.
        party : str, optional
            Slug of the party of the view by party (``partido``); it wins over ``view``.
        view : str, optional
            ``partidos`` for the chart of the parties without the card of a party, ``renidos`` for the list of the
            closest seats (``vista``).

        Returns
        -------
        str
            The HTML of ``circunscripciones.html``; its canonical drops ``mode`` and ``run`` (``canonical_url``) and
            the parameters the page ignores: ``circunscripcion`` always, ``partido`` and ``vista`` on the route of a
            district, ``vista`` when ``partido`` wins (and either when empty).
        """
        context = pages.seat_odds_context(self.state(scope), district, party, view)
        ignored = ['circunscripcion']

        if district is not None or party is None:
            ignored.append('partido')

        if district is not None or party is not None or view is None:
            ignored.append('vista')

        return self.render('circunscripciones.html', canonical=self.canonical_url(*ignored), **context)

    async def index_action(self, scope: str) -> str:
        """
        Render the page on the district with most options in play (or on the chart of the parties and the card of the
        party of ``partido``, the chart of the parties of ``vista=partidos`` or the closest seats of
        ``vista=renidos``), or send the dropdown of districts to the address of the one chosen.

        The dropdown (``#district-pick``) submits ``circunscripcion`` with the slug of a district (the browser lands
        here with or without JavaScript), and the answer is a 301 to ``/{scope}/circunscripciones/{slug}`` with the
        rest of the query (``mode``, ``run``) and the fragment of the card (``pages.district_href``).

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
            404 for an unknown scope or party, or a run without the ``seat-odds`` part (or with a single district);
            400 for an invalid ``mode``, ``run`` or ``vista``.
        """
        code = pages.resolve_event_scope(scope)
        wanted = self.request.query.get('circunscripcion')

        if wanted:
            # the address of `district_href`, with the fragment of the card: on a phone the page lands on it
            rest = self.query_without('circunscripcion')

            return self.redirect(pages.district_href(code, quote(wanted, safe=''), rest))

        return self.page(code, party=self.request.query.get('partido') or None,
                         view=self.request.query.get('vista') or None)

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
