#!/usr/bin/env python
"""
Build the trimmed Wikipedia fixtures used by the loader tests: only the headings (h2-h4) and the
`wikitable` tables of each article, in document order. Content from the English Wikipedia, CC BY-SA 4.0.

Usage (from the repository root):
    python tests/fixtures/wikipedia/build_fixtures.py [--source DIR]

With `--source`, the full pages are read from `DIR/{title}.html` instead of being downloaded. The counts
asserted in `tests/test_loader_unit.py` belong to the snapshot of 2026-10-01: rebuilding the fixtures
from the live pages may change them.
"""
import argparse
import os

import requests
from lxml import html

HERE = os.path.abspath(os.path.dirname(__file__))
HEADERS = {'User-Agent': 'ManyThings/1.0 (https://manythings.pro/; info@manythings.pro) mtpy/elections/1.0'}
PAGES = [
    '2023_Madrilenian_regional_election',
    'Next_Madrilenian_regional_election',
    '2022_Castilian-Leonese_regional_election',
    '2019_Asturian_regional_election',
    '2023_Canarian_regional_election',
    '2023_Asturian_regional_election'
]


def trim(content: bytes) -> bytes:
    """
    Reduce a full article to its headings and `wikitable` tables.

    Parameters
    ----------
    content : bytes
        HTML of the full article.

    Returns
    -------
    bytes
        HTML document with the kept elements as children of `body`.
    """
    doc = html.fromstring(content)
    for el in doc.xpath('//style | //script | //link | //img | //sup[contains(@class, "reference")]'):
        el.drop_tree()

    body = html.Element('body')
    nodes = doc.xpath("//*[self::h2 or self::h3 or self::h4] | //table[contains(@class, 'wikitable')]")
    for node in nodes:
        if node.tag == 'table' and node.xpath("ancestor::table[contains(@class, 'wikitable')]"):
            continue
        node.tail = '\n'
        body.append(node)

    # Explicit charset: without it lxml reads the trimmed file as latin-1
    head = html.Element('head')
    head.append(html.Element('meta', charset='utf-8'))

    root = html.Element('html')
    root.append(head)
    root.append(body)

    return html.tostring(root, encoding='utf-8')


def main() -> None:
    """Download (or read) every page of `PAGES` and write its trimmed version next to this script."""
    parser = argparse.ArgumentParser(description='Build the trimmed Wikipedia fixtures')
    parser.add_argument('--source', default=None, help='directory with the full pages already downloaded')
    args = parser.parse_args()

    for page in PAGES:
        if args.source:
            with open(os.path.join(args.source, page + '.html'), 'rb') as fh:
                content = fh.read()
        else:
            content = requests.get('https://en.wikipedia.org/wiki/' + page, headers=HEADERS, timeout=60).content

        with open(os.path.join(HERE, page + '.html'), 'wb') as fh:
            fh.write(trim(content))


if __name__ == '__main__':
    main()
