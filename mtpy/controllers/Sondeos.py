"""
Controller of the page of a poll (``/{scope}/sondeos/{date}/{pollster}[/{sponsor}]``) and of the image of its link
previews (the same address ending in ``.png``).
"""
from typing import Optional

from ..lib import pages, webapi
from ..lib.webapi import check_date
from .Page import Page


class Sondeos(Page):
    """
    Page of a poll: its card to share (the vote against the average of its day, the seats of the house, its rating
    and weight), the detail of its vote and of its weight, the chart of its house in the cycle and its sheet. And the
    image a network shows when the page is shared (``image_action``), which the page declares as its ``og:image``.
    """

    active = None

    async def detail_action(self, scope: str, date: str, pollster: str, sponsor: Optional[str] = None) -> str:
        """
        Render the page of a poll.

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).
        date : str
            Day of the poll, the last day of its fieldwork (``YYYY-MM-DD``).
        pollster : str
            Slug of the house, or its id.
        sponsor : str, optional
            Slug of the sponsor, when the poll has one.

        Returns
        -------
        str
            The HTML of ``sondeo.html``, with its breadcrumbs (the election, linking to the page of the event, and
            the poll); its canonical is the URL of the poll (``pages.poll_detail_href``), without ``mode`` nor
            ``run``. Its ``og:image`` is the image of the poll (``image_action``, ``preview_image``): the canonical
            ending in ``.png``, with the fingerprint of what it draws and a text alternative with its figures. The
            image always draws the latest run, so a page with a pinned run declares the fingerprint of the latest
            one, and the brand image when the poll is not in it.

        Raises
        ------
        HttpError
            404 for an unknown scope or poll; 400 for an invalid date.
        """
        code = pages.resolve_event_scope(scope)
        check_date(date)
        context = pages.sondeo_context(self.state(code), date, pollster, sponsor)
        origin = webapi.settings()['site_url']
        trail = pages.structured.breadcrumbs(origin, [
            (context['event_bar']['election'], '/' + code),
            ('{} · {}'.format(context['poll_title'], context['poll_date']), None)])
        preview = context['preview']

        if context['state']['pinned']:
            preview = pages.latest_poll_preview(code, date, pollster, sponsor)

        return self.render('sondeo.html', canonical=origin + context['detail_href'], structured=[trail],
                           **self.preview_image(context['detail_href'], preview), **context)

    async def image_action(self, scope: str, date: str, pollster: str, sponsor: Optional[str] = None) -> bytes:
        """
        Draw the image of the link previews of a poll.

        It is always the poll of the latest published run: the query (``mode``, ``run``, the ``v`` of the address
        the page declares, anything else) is not read.

        Parameters
        ----------
        scope : str
            Scope of the route (``pages.resolve_event_scope``).
        date : str
            Day of the poll, the last day of its fieldwork (``YYYY-MM-DD``).
        pollster : str
            Slug of the house, or its id.
        sponsor : str, optional
            Slug of the sponsor, when the poll has one.

        Returns
        -------
        bytes
            The PNG (``pages.images.poll_png`` of ``pages.poll_preview``), answered as a cacheable image (``png``).

        Raises
        ------
        HttpError
            404 for an unknown scope or poll and 400 for an invalid date, as the page; nothing is drawn then.
        """
        code = pages.resolve_event_scope(scope)
        check_date(date)
        return self.png(pages.images.poll_png(pages.poll_preview(pages.parse_state({}, code), date, pollster, sponsor)))
