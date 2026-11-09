"""
Controller of the page of a poll (``/{scope}/sondeos/{date}/{pollster}[/{sponsor}]``).
"""
from typing import Optional

from ..lib import pages, webapi
from ..lib.webapi import check_date
from .Page import Page


class Sondeos(Page):
    """
    Page of a poll: its card to share (the vote against the average of its day, the seats of the house, its rating
    and weight), the detail of its vote and of its weight, the chart of its house in the cycle and its sheet.
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
            The HTML of ``sondeo.html``; its canonical is the URL of the poll (``pages.poll_detail_href``), without
            ``mode`` nor ``run``.

        Raises
        ------
        HttpError
            404 for an unknown scope or poll; 400 for an invalid date.
        """
        code = pages.resolve_event_scope(scope)
        check_date(date)
        context = pages.sondeo_context(self.state(code), date, pollster, sponsor)

        return self.render('sondeo.html', canonical=webapi.settings()['site_url'] + context['detail_href'], **context)
