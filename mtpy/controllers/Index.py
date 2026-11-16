"""
Controller of the home page (``/``).
"""
from ..lib import pages, webapi
from .Page import Page


class Index(Page):
    """
    Home page; a request with ``?scope=`` (the URL of the home before phase 7) goes to the event of the scope.
    """

    active = '/'

    async def index_action(self) -> str:
        """
        Render the home page, or redirect ``/?scope=X`` to ``/X`` keeping the rest of the query.

        Returns
        -------
        str
            The HTML of ``index.html`` (the index of events), with the blocks that name the site and its publisher
            (``pages.structured``), or the empty body of the 301.
        """
        scope = self.request.query.get('scope')

        if scope:
            scope = pages.resolve_scope(webapi.check_scope(scope))

            return self.redirect('/' + scope + self.query_without('scope'))

        origin = webapi.settings()['site_url']
        structured = [pages.structured.website(origin), pages.structured.organization(origin)]

        return self.render('index.html', structured=structured, **pages.inicio_context(self.state()))
