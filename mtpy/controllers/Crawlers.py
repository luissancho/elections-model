"""
Controller of the files for the crawlers: ``/robots.txt`` and ``/sitemap.xml``.
"""
import hashlib

from ..lib import pages
from .Page import Page


class Crawlers(Page):
    """
    Files the crawlers ask for; their errors are the HTML error page of every page.
    """

    active = None

    def respond(self, body: str, content_type: str) -> str:
        """
        Answer a text file as a cacheable response.

        Sets the content type, an ETag (MD5 of the UTF-8 bytes) and ``public, max-age=<pages.CRAWLERS_TTL>``.

        Parameters
        ----------
        body : str
            Content of the file.
        content_type : str
            Its media type, without the charset.

        Returns
        -------
        str
            ``body``.
        """
        self.response.set_content_type(content_type)
        self.response.set_etag(hashlib.md5(body.encode('utf-8')).hexdigest())
        self.response.set_cache(pages.CRAWLERS_TTL)

        return body

    async def robots_action(self) -> str:
        """
        Answer ``/robots.txt``.

        Returns
        -------
        str
            The text of ``pages.robots_txt``, as ``text/plain``.
        """
        return self.respond(pages.robots_txt(), 'text/plain')
