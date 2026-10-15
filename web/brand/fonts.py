"""
Fetch the web fonts of the site from Google Fonts, as variable woff2 files subset to the latin range.

The site serves DM Sans (text, axes ``opsz`` and ``wght``) and Outfit (headings, axis ``wght``) from
``web/dist/vendor/fonts/`` (CSP ``default-src 'self'``: no external fonts). Both are SIL OFL; their notices live
in ``LICENSE`` next to the files. Run once from the repository root and commit the files::

    python web/brand/fonts.py

The Google Fonts CSS API answers a modern browser with one ``@font-face`` per subset, each with the variable font
of the whole axes; this takes the ``latin`` block and cuts the axes down to the ranges the site uses. Needs
``fonttools`` with the woff2 extra (``brotli``). Not a runtime dependency of the web.
"""
import os
import re
import urllib.request

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'web', 'dist', 'vendor', 'fonts')
# Family: (axis ranges of the request, file name, axis ranges kept by the partial instancing)
FAMILIES = {
    'DM Sans': ('opsz,wght@9..40,400..700', 'DMSans.woff2', {'wght': (400, 700), 'opsz': (9, 40)}),
    'Outfit': ('wght@700..800', 'Outfit.woff2', {'wght': (700, 800)}),
}
CSS_URL = 'https://fonts.googleapis.com/css2?family={family}:{axes}&display=swap'
# A browser that gets woff2: without it the API serves TrueType
USER_AGENT = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/130.0.0.0 Safari/537.36')


def fetch(out_dir: str = OUT) -> list:
    """
    Download the latin variable woff2 of every family of ``FAMILIES``.

    Parameters
    ----------
    out_dir : str, optional
        Directory of the files (created when missing).

    Returns
    -------
    list of str
        File names written (the names of ``FAMILIES``).
    """
    os.makedirs(out_dir, exist_ok=True)
    names = []

    for family, (axes, name, kept) in FAMILIES.items():
        css = _get(CSS_URL.format(family=family.replace(' ', '+'), axes=axes)).decode('utf-8')
        path = os.path.join(out_dir, name)

        with open(path, 'wb') as file:
            file.write(_get(_latin_url(css)))

        _restrict(path, kept)
        names.append(name)

    return names


def _restrict(path: str, axes: dict) -> None:
    """
    Cut the variable font of ``path`` down to the axis ranges the site uses (partial instancing), in place.

    Parameters
    ----------
    path : str
        The woff2 file.
    axes : dict
        ``{tag: (min, max)}`` of the ranges to keep; the API serves the whole axis.
    """
    font = instancer.instantiateVariableFont(TTFont(path), axes, inplace=False, updateFontNames=False)
    font.flavor = 'woff2'
    font.save(path)


def _latin_url(css: str) -> str:
    """
    URL of the latin woff2 in the CSS the API returned.

    Parameters
    ----------
    css : str
        The ``@font-face`` rules.

    Returns
    -------
    str
        The ``src`` URL of the ``/* latin */`` block.

    Raises
    ------
    ValueError
        When the CSS has no such block.
    """
    for subset, rules in re.findall(r'/\* (\w+) \*/\s*@font-face\s*{([^}]*)}', css):
        if subset == 'latin':
            match = re.search(r"src:\s*url\(([^)]+)\)\s*format\('woff2'\)", rules)

            if match:
                return match.group(1)

    raise ValueError('no latin woff2 in the CSS')


def _get(url: str) -> bytes:
    """Body of a GET of ``url`` as a browser."""
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})

    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


if __name__ == '__main__':
    for written in fetch():
        print(written, os.path.getsize(os.path.join(OUT, written)), 'bytes')
