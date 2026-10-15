"""
Controller of the houses page (``/casas``) and of the profile of a pollster (``/casas/{slug}``).
"""
from ..lib import pages, webapi
from .Page import Page


class Casas(Page):
    """
    Houses page: pollsters ranking, house effects and herding; ``/casas/{slug}``, the profile of a pollster.
    """

    active = '/casas'

    async def index_action(self) -> str:
        """
        Render the houses page; with the old ``id`` parameter, redirect to the profile of that pollster.

        Returns
        -------
        str
            The HTML of ``casas.html``, or the empty body of a 301 to ``pages.pollster_href`` (the rest of the
            query kept) when ``id`` (validated by ``webapi.check_pollster``, 404 when unknown) is given.
        """
        pollster = self.request.query.get('id')

        if pollster is not None:
            pollster_id = webapi.check_pollster(pollster)
            name = webapi.query_pollster(pollster_id, pages.casas.RATING_SCOPE)['pollster'].get('name')

            return self.redirect(pages.pollster_href(name, pollster_id) + self.query_without('id'))

        return self.render('casas.html', **pages.casas_context(self.state(), active=self.active))

    async def profile_action(self, pollster: str) -> str:
        """
        Render the profile of a pollster.

        Parameters
        ----------
        pollster : str
            Slug of the pollster, or its id (``pages.resolve_pollster``).

        Returns
        -------
        str
            The HTML of ``casa.html``; its canonical is the slug URL, whatever the query (the profile is
            global: the scope of the ranking page does not change it).
        """
        pollster_id = pages.resolve_pollster(pollster)
        context = pages.casa_context(self.state(), pollster_id, active=self.active)
        href = pages.pollster_href(context['profile']['pollster'].get('name'), pollster_id)

        return self.render('casa.html', canonical=webapi.settings()['site_url'] + href, **context)
