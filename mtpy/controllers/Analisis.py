"""
Controller of the Análisis section: the index of the studies (``/analisis``) and the page of each one
(``/analisis/{study}``).
"""
from ..lib import pages, webapi
from .Page import Page


class Analisis(Page):
    """
    Análisis pages: questions of the electoral debate answered with official data, one page per published study.
    """

    active = '/analisis'

    async def index_action(self) -> str:
        """
        Render the index of the published studies.

        Returns
        -------
        str
            The HTML of ``analisis.html``.
        """
        return self.render('analisis.html', **pages.analisis_context(self.state()))

    async def study_action(self, study: str) -> str:
        """
        Render the page of a study.

        Parameters
        ----------
        study : str
            Study slug (a key of ``pages.STUDIES``).

        Returns
        -------
        str
            The HTML of the template of the study (``analisis_voto_cera.html`` for ``voto-cera``), with its
            ``Article`` block (the headline and the description of the page, its URL and the time of its last
            publication) and its breadcrumbs.

        Raises
        ------
        HttpError
            404 ``study not found`` for a study without a page or not published.
        """
        context = pages.study_context(self.state(), study)
        template = pages.STUDIES[study]['template']
        headline = pages.structured.block_text(template, 'title', context)
        description = pages.snippet(pages.structured.block_text(template, 'description', context))
        article = pages.structured.article(webapi.settings()['site_url'], '/' + self.request.uri, headline,
                                           description, pages.study_published_at(study))

        return self.render(template, structured=[article, self.section_trail(self.active, headline)], **context)
