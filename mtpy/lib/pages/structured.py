"""
Structured data of the pages: the schema.org blocks (JSON-LD) that tell a search engine what a reader already sees.
The home says what the site is called and who publishes it, a study that it is an article with its author and its
date, and a second-level page which section it belongs to. Each builder returns a plain dict; the controller of the
page passes them as ``structured`` and ``base.html`` prints one ``<script type="application/ld+json">`` per block.
"""
from html import unescape
from typing import Optional

from .common import AUTHOR_NAME, AUTHOR_URL, SITE_TAGLINE, SITE_TITLE, environment

SCHEMA_CONTEXT = 'https://schema.org'
LANGUAGE = 'es'
# The logo of the publisher: the lens of the brand, a square PNG (a search engine asks for 112x112 at least)
LOGO_PATH = '/dist/img/apple-touch-icon.png'


def person() -> dict:
    """
    The author of the site, as the footer of every page names it.

    Returns
    -------
    dict
        A ``Person`` node (``name``, ``url``), without ``@context``: it goes inside another block.
    """
    return {'@type': 'Person', 'name': AUTHOR_NAME, 'url': AUTHOR_URL}


def publisher(origin: str) -> dict:
    """
    The site as the publisher of a page.

    Parameters
    ----------
    origin : str
        Public origin of the site, without a trailing slash (``settings()['site_url']``).

    Returns
    -------
    dict
        An ``Organization`` node (``name``, ``url``, ``logo``), without ``@context``.
    """
    return {'@type': 'Organization', 'name': SITE_TITLE, 'url': origin + '/', 'logo': origin + LOGO_PATH}


def website(origin: str) -> dict:
    """
    Block of the home that names the site (the name a search engine shows over each result).

    Parameters
    ----------
    origin : str
        Public origin of the site, without a trailing slash.

    Returns
    -------
    dict
        A ``WebSite`` block: ``name``, ``url`` (the home), ``inLanguage`` and ``description`` (the tagline).
    """
    return {'@context': SCHEMA_CONTEXT, '@type': 'WebSite', 'name': SITE_TITLE, 'url': origin + '/',
            'inLanguage': LANGUAGE, 'description': SITE_TAGLINE}


def organization(origin: str) -> dict:
    """
    Block of the home that says who publishes the site.

    Parameters
    ----------
    origin : str
        Public origin of the site, without a trailing slash.

    Returns
    -------
    dict
        An ``Organization`` block: what ``publisher`` gives plus ``description`` (the tagline) and ``founder``
        (``person``).
    """
    return {'@context': SCHEMA_CONTEXT, **publisher(origin), 'description': SITE_TAGLINE, 'founder': person()}


def article(origin: str, href: str, headline: str, description: Optional[str] = None,
            modified: Optional[str] = None) -> dict:
    """
    Block of a page that is an article (a study).

    Parameters
    ----------
    origin : str
        Public origin of the site, without a trailing slash.
    href : str
        Canonical path of the page (``'/analisis/sesgo-cis'``).
    headline : str
        The headline the page shows.
    description : str, optional
        The description of the page; left out when empty.
    modified : str, optional
        ISO date or time of its last publication; left out when empty.

    Returns
    -------
    dict
        An ``Article`` block: ``headline``, ``url`` and ``mainEntityOfPage`` (the canonical URL), ``inLanguage``,
        ``author`` (``person``) and ``publisher`` (``publisher``), plus ``description`` and ``dateModified`` when
        given.
    """
    url = origin + href
    block = {'@context': SCHEMA_CONTEXT, '@type': 'Article', 'headline': headline, 'url': url, 'mainEntityOfPage': url,
             'inLanguage': LANGUAGE, 'author': person(), 'publisher': publisher(origin)}

    if description:
        block['description'] = description

    if modified:
        block['dateModified'] = modified

    return block


def breadcrumbs(origin: str, trail: list) -> dict:
    """
    Block with the place of a page in the site.

    Parameters
    ----------
    origin : str
        Public origin of the site, without a trailing slash.
    trail : list of tuple
        ``(name, href)`` from the section down to the page; ``href`` is the canonical path, or ``None`` for the page
        itself (a search engine takes its URL).

    Returns
    -------
    dict
        A ``BreadcrumbList`` block whose ``itemListElement`` numbers the trail from 1; each ``ListItem`` has its
        ``name`` and, when the href is given, its ``item`` (the absolute URL).
    """
    items = []

    for position, (name, href) in enumerate(trail, start=1):
        item = {'@type': 'ListItem', 'position': position, 'name': name}

        if href:
            item['item'] = origin + href

        items.append(item)

    return {'@context': SCHEMA_CONTEXT, '@type': 'BreadcrumbList', 'itemListElement': items}


def block_text(template: str, block: str, context: dict) -> str:
    """
    Text of a block of a page template, as the reader sees it.

    Lets a structured block quote the page (its headline, its description) instead of repeating the text.

    Parameters
    ----------
    template : str
        Template name in ``web/templates``.
    block : str
        Name of the block (``'title'``, ``'description'``).
    context : dict
        Template variables the block uses.

    Returns
    -------
    str
        The rendered block without HTML escaping and with its whitespace collapsed.
    """
    loaded = environment().get_template(template)
    text = ''.join(loaded.blocks[block](loaded.new_context(context)))

    return ' '.join(unescape(text).split())
