"""
What the site tells the crawlers: ``/robots.txt`` (what not to crawl and where the sitemap is) and ``/sitemap.xml``
(every canonical page that is published). Both are computed per request from the same links the pages use, so they
never drift from what is published.
"""
from ..webapi import settings

# Seconds both answers stay fresh: a new page takes at most this long to reach the sitemap
CRAWLERS_TTL = 3600
SITEMAP_PATH = '/sitemap.xml'
# The API serves the data of the pages as JSON and CSV: crawling those links only loads S3 and the database
DISALLOWED = ('/api/',)


def robots_txt() -> str:
    """
    Text of ``/robots.txt``.

    Returns
    -------
    str
        The rules for every crawler (everything allowed but ``DISALLOWED``), an empty line and the absolute URL of
        the sitemap (``web.site_url``), each line ended by a line feed.
    """
    lines = ['User-agent: *']
    lines += ['Disallow: ' + path for path in DISALLOWED]
    lines += ['', 'Sitemap: {}{}'.format(settings()['site_url'], SITEMAP_PATH)]

    return '\n'.join(lines) + '\n'
