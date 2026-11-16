"""
What the site tells the crawlers: ``/robots.txt`` (what not to crawl and where the sitemap is) and ``/sitemap.xml``
(every canonical page that is published). Both are computed per request from the same links the pages use, so they
never drift from what is published.
"""
from xml.sax.saxutils import escape

from ...core.api import HttpError
from ..webapi import settings, site
from .analisis import study_cards
from .common import bundle_part, published_scopes
from .metodo import METODO_SECTIONS
from .slugs import DETAIL_PATH, PROFILE_PATH, event_href, poll_detail_href, pollster_slugs
from .sondeos import find_poll

# Seconds both answers stay fresh (``max-age``): a new page takes at most this long to reach a cached sitemap
CRAWLERS_TTL = 3600
SITEMAP_PATH = '/sitemap.xml'
# The API serves the data of the pages as JSON and CSV: crawling those links only loads S3 and the database
DISALLOWED = ('/api/',)
SITEMAP_NS = 'http://www.sitemaps.org/schemas/sitemap/0.9'


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


def scope_entries() -> list:
    """
    Sitemap entries of the home and of the event page of every published scope.

    Returns
    -------
    list of dict
        ``{'path', 'lastmod'}``: ``/`` with the ``updated_at`` of the manifest, then ``/{scope}`` with the ``run_at``
        of its latest run, in catalogue order.

    Raises
    ------
    HttpError
        503 when no bundle has been published yet.
    """
    entries = [{'path': '/', 'lastmod': site().manifest_data().get('updated_at')}]
    entries += [{'path': '/' + row['code'], 'lastmod': row['run_at']} for row in published_scopes()]

    return entries


def election_paths() -> list:
    """
    Paths of the history of elections and of the detail of every published election.

    Returns
    -------
    list of str
        ``/elecciones`` and one ``event_href`` per published election, by scope in catalogue order.
    """
    codes = [row['code'] for row in site().scopes()['scopes']]
    site().warm_events(codes)
    paths = [DETAIL_PATH]

    for code in codes:
        dates = site().events(code)
        paths += [event_href(code, date, dates) for date in dates]

    return paths


def pollster_paths() -> list:
    """
    Paths of the houses page and of the profile of every pollster with a slug.

    Returns
    -------
    list of str
        ``/casas`` and ``/casas/{slug}`` by slug; a pollster whose name gives no slug (its profile lives under its
        id) is left out.

    Raises
    ------
    HttpError
        503 without a database.
    """
    return [PROFILE_PATH] + ['{}/{}'.format(PROFILE_PATH, slug) for slug in sorted(pollster_slugs())]


def method_paths() -> list:
    """
    Paths of the method and of its subpages with a text.

    Returns
    -------
    list of str
        ``/metodo`` and ``/metodo/{id}`` for the entries of ``METODO_SECTIONS`` with a template, in menu order.
    """
    return ['/metodo'] + ['/metodo/' + section for section, entry in METODO_SECTIONS.items()
                          if entry['template'] is not None]


def study_paths() -> list:
    """
    Paths of the index of studies and of every published study with a page.

    Returns
    -------
    list of str
        ``/analisis`` and the ``href`` of each card of ``study_cards``.
    """
    return ['/analisis'] + [card['href'] for card in study_cards()]


def page_finds_poll(polls: list, href: str) -> bool:
    """
    Whether the page of a poll answers: its route finds exactly one poll for the URL.

    Parameters
    ----------
    polls : list of dict
        Records of the ``polls`` part of the scope (those of the day of the poll are enough).
    href : str
        Link to the page (``poll_detail_href``): ``'/{scope}/sondeos/{date}/{house}'`` plus the sponsor, when it
        has one.

    Returns
    -------
    bool
        False when ``find_poll`` answers 404: two records share the day, the house and the sponsor, or the page
        cannot tell the poll from another of the same house and day.
    """
    date, house, *sponsor = href.split('/')[3:]

    try:
        find_poll(polls, date, house, sponsor[0] if sponsor else None)
    except HttpError:
        return False

    return True


def poll_paths() -> list:
    """
    Paths of the pages of the polls of every published scope.

    Returns
    -------
    list of str
        ``poll_detail_href`` of each poll of the ``polls`` part of the latest run of every published scope, in
        that order. The polls without a link (no valid date or no house), those whose page does not answer
        (``page_finds_poll``) and the scopes without the part are left out.
    """
    paths = {}

    for row in published_scopes():
        polls = (bundle_part(site().run_file, row['code'], row['latest'], 'polls') or {}).get('polls') or []
        by_day = {}

        for poll in polls:
            by_day.setdefault(str(poll.get('date'))[:10], []).append(poll)

        for poll in polls:
            href = poll_detail_href(row['code'], poll)

            if href is not None and page_finds_poll(by_day[str(poll.get('date'))[:10]], href):
                paths.setdefault(href)

    return list(paths)


def sitemap_entries() -> list:
    """
    Every page of the sitemap: the canonical ones that are published, without query variants.

    Returns
    -------
    list of dict
        ``{'path': str, 'lastmod': str or None}`` in this order: the home and the published scopes (the only ones
        with a ``lastmod``, the ISO time of their last update), the elections, the pollsters, the method, the
        studies and the polls.

    Raises
    ------
    HttpError
        503 without a bundle or without a database (the profiles of the pollsters come from it): never a list
        with a group missing.
    """
    paths = election_paths() + pollster_paths() + method_paths() + study_paths() + poll_paths()

    return scope_entries() + [{'path': path, 'lastmod': None} for path in paths]


def sitemap_xml() -> str:
    """
    Text of ``/sitemap.xml``, a ``urlset`` of the sitemaps.org protocol.

    Returns
    -------
    str
        The XML with one ``<url>`` per line: its ``<loc>`` (``web.site_url`` plus the path, escaped) and, when the
        entry has one, its ``<lastmod>``.

    Raises
    ------
    HttpError
        Those of ``sitemap_entries``.
    """
    origin = settings()['site_url']
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="{}">'.format(SITEMAP_NS)]

    for entry in sitemap_entries():
        lastmod = '<lastmod>{}</lastmod>'.format(escape(entry['lastmod'])) if entry['lastmod'] else ''
        lines.append('<url><loc>{}</loc>{}</url>'.format(escape(origin + entry['path']), lastmod))

    lines.append('</urlset>')

    return '\n'.join(lines) + '\n'
