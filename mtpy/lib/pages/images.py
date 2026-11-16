"""
Images the server draws for the link previews (``og:image``) of the pages that do not change: the card of a poll.

A network that reads a link runs no JavaScript, so the card the browser draws on the page of a poll cannot be the
image of its link: the server draws its own with Pillow, 1200x675, from plain data (``sondeos.preview_of``). The
texts are set in the fonts the site serves (the variable woff2 of ``web/dist/vendor/fonts``, which the FreeType of
the Pillow wheels reads as they are) with the basic layout, so an image comes out the same whether or not this
Pillow has raqm; the wordmark is pasted from ``web/dist/img/wordmark.png``, which ``web/brand/build.py`` rasterises
from the logo.

Everything is laid out in pixels of the final image and drawn ``SCALE`` times larger, then shrunk with Lanczos, for
smooth edges. The pieces (``canvas``, ``write``, ``fit_text``, ``box``, ``header``, ``encode``) know nothing about
polls: another card (a past election) can be built with them.
"""
import io
import math
import os
from functools import lru_cache
from typing import Optional

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .common import REPO_ROOT

SIZE = (1200, 675)
SCALE = 2
FONTS_DIR = os.path.join(REPO_ROOT, 'web', 'dist', 'vendor', 'fonts')
WORDMARK_PATH = os.path.join(REPO_ROOT, 'web', 'dist', 'img', 'wordmark.png')
# Font of each kind of text: the file and its weight (axis ``wght``); the optical size (axis ``opsz``) of a font
# that has it follows the size of the text, as in the browser
FACES = {'title': ('Outfit.woff2', 800), 'bold': ('Outfit.woff2', 700), 'body': ('DMSans.woff2', 500)}
ELLIPSIS = '…'

BACKGROUND = '#f4f5f8'  # --color-bg
INK = '#1a1c22'  # --color-text
MUTED = '#5a5f72'  # --color-muted
FAINT = '#9298ac'  # --color-faint
UP = '#16a34a'  # --color-ok
DOWN = '#dc2626'  # --color-bad
NEUTRAL = '#9298ac'  # the bar of a party without a valid colour

# Layout, in pixels of the image. X lays the title of the link over the bottom left corner and Facebook cuts some
# 24 px at the top and at the bottom: nothing goes there
MARGIN = 64  # left edge of the texts
RIGHT = 1136  # right edge: the wordmark, the seats and the footnote end here
WORDMARK_WIDTH = 200
WORDMARK_TOP = 70
WORDMARK_GAP = 42  # air between the title and the wordmark
TITLE_BASELINE = 108
TITLE_SIZES = (50, 34)  # the title starts at the first and shrinks down to the second before it is cut
SUBTITLE_BASELINE = 150
SUBTITLE_SIZES = (25, 20)
ROWS = 6  # the rows that fit
ROW_TOP = 196
ROW_STEP = 66
NAME_SIZES = (34, 22)
NAME_GAP = 12  # air between the names and the bars
BAR_LEFT = 200  # where the bars start, when the names fit at the left of it
BAR_LEFT_MAX = 290  # the furthest a long name pushes the bars
BAR_RIGHT = 670  # where the bar of the largest party ends
BAR_MAX = BAR_RIGHT - BAR_LEFT
BAR_MIN = 8  # the bar of a zero
BAR_TOP = 10  # from the top of the row
BAR_HEIGHT = 40
BAR_RADIUS = 8
SHARE_SIZE = 38
CHANGE_SIZE = 24
SEATS_SIZES = (36, 22)
SEATS_WIDTH = 190  # the room of the seats column
SEATS_LABEL = 'Escaños'
LABEL_SIZE = 20  # the header of the seats column and the footnote
FOOT_BASELINE = 632
FOOT_LEFT = 624  # the footnote never starts at the left of this: the corner of the link title stays free


@lru_cache(maxsize=128)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Font of a kind of text at a size, opened once per process.

    Parameters
    ----------
    name : str
        Kind of text, a key of ``FACES``: ``'title'`` (Outfit, extra bold), ``'bold'`` (Outfit, bold) or ``'body'``
        (DM Sans, medium).
    size : int
        Size of the text, in pixels of the image.

    Returns
    -------
    PIL.ImageFont.FreeTypeFont
        The variable font of the site at ``size * SCALE`` pixels, with the basic layout and its axes pinned: the
        weight of ``FACES`` and, where the font has the axis, the optical size of ``size``; each within its range.
    """
    file, weight = FACES[name]
    face = ImageFont.truetype(os.path.join(FONTS_DIR, file), size * SCALE, layout_engine=ImageFont.Layout.BASIC)

    try:
        axes = face.get_variation_axes()
    except OSError:  # a static font: nothing to pin
        axes = []

    values = []

    for axis in axes:
        label = axis['name'].decode('utf-8', 'replace') if isinstance(axis['name'], bytes) else str(axis['name'])
        value = {'Weight': weight, 'Optical size': size}.get(label, axis['default'])
        values.append(min(max(value, axis['minimum']), axis['maximum']))

    if values:
        face.set_variation_by_axes(values)

    return face


def text_width(name: str, text: str, size: int) -> float:
    """
    Width of a line of text.

    Parameters
    ----------
    name : str
        Kind of text (``font``).
    text : str
        The text.
    size : int
        Size of the text, in pixels of the image.

    Returns
    -------
    float
        The advance of the text, in pixels of the image, measured with the font it is drawn with.
    """
    return font(name, size).getlength(text) / SCALE


def fit_text(name: str, text: str, size: int, min_size: int, max_width: float) -> tuple:
    """
    Fit a line of text into a width: first it shrinks, then it is cut.

    Parameters
    ----------
    name : str
        Kind of text (``font``).
    text : str
        The text.
    size : int
        The size it has when it fits, in pixels of the image.
    min_size : int
        The smallest size it may shrink to.
    max_width : float
        The room, in pixels of the image.

    Returns
    -------
    tuple
        ``(size, text)``: the largest whole size between ``size`` and ``min_size`` at which the text fits, with the
        text as it came; when it fits at none, ``min_size`` and the longest head of the text that fits followed by
        ``ELLIPSIS`` (the ellipsis alone when not even a letter does).
    """
    for candidate in range(size, min_size - 1, -1):
        if text_width(name, text, candidate) <= max_width:
            return candidate, text

    low, high = 0, len(text)

    while low < high:
        middle = (low + high + 1) // 2

        if text_width(name, text[:middle].rstrip() + ELLIPSIS, min_size) <= max_width:
            low = middle
        else:
            high = middle - 1

    return min_size, text[:low].rstrip() + ELLIPSIS


def clean(text) -> str:
    """
    A text as one line.

    Parameters
    ----------
    text : object
        The text; ``None`` counts as empty.

    Returns
    -------
    str
        The text with every run of white space (line breaks, tabs, no-break spaces) as one space, and none at
        the ends.
    """
    return ' '.join(str(text if text is not None else '').split())


def colour_of(value, fallback: str = NEUTRAL) -> tuple:
    """
    A colour for Pillow.

    Parameters
    ----------
    value : object
        A CSS colour (``'#1d84ce'``).
    fallback : str, optional
        The colour used when ``value`` is not one.

    Returns
    -------
    tuple
        ``(red, green, blue)``; the opacity of a colour that has one is dropped.
    """
    try:
        return ImageColor.getrgb(value)[:3]
    except (ValueError, TypeError, AttributeError):
        return ImageColor.getrgb(fallback)[:3]


def canvas() -> Image.Image:
    """
    An empty image to draw on.

    Returns
    -------
    PIL.Image.Image
        RGB image of ``SIZE`` times ``SCALE``, filled with ``BACKGROUND``.
    """
    return Image.new('RGB', (SIZE[0] * SCALE, SIZE[1] * SCALE), BACKGROUND)


def write(draw: ImageDraw.ImageDraw, name: str, text: str, x: float, baseline: float, size: int, fill,
          right: bool = False) -> float:
    """
    Draw a line of text.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Where to draw (a ``canvas``).
    name : str
        Kind of text (``font``).
    text : str
        The text, one line.
    x : float
        Where the text starts, or where it ends with ``right``, in pixels of the image.
    baseline : float
        Baseline of the text, in pixels of the image.
    size : int
        Size of the text, in pixels of the image.
    fill : str or tuple
        Colour.
    right : bool, optional
        Whether the text ends at ``x`` instead of starting there.

    Returns
    -------
    float
        The width of the text (``text_width``).
    """
    draw.text((x * SCALE, baseline * SCALE), text, font=font(name, size), fill=fill, anchor='rs' if right else 'ls')

    return text_width(name, text, size)


def box(draw: ImageDraw.ImageDraw, left: float, top: float, right: float, bottom: float, radius: float,
        fill) -> None:
    """
    Draw a rectangle with round corners.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Where to draw (a ``canvas``).
    left, top, right, bottom : float
        The rectangle, in pixels of the image.
    radius : float
        Radius of the corners, in pixels of the image.
    fill : str or tuple
        Colour.
    """
    draw.rounded_rectangle((left * SCALE, top * SCALE, right * SCALE, bottom * SCALE), radius * SCALE, fill=fill)


@lru_cache(maxsize=1)
def wordmark() -> Image.Image:
    """
    The wordmark of the site at the size the images show it, read once per process.

    Returns
    -------
    PIL.Image.Image
        RGBA image, ``WORDMARK_WIDTH * SCALE`` wide, from ``WORDMARK_PATH``.
    """
    with Image.open(WORDMARK_PATH) as source:
        mark = source.convert('RGBA')

    width = WORDMARK_WIDTH * SCALE

    return mark.resize((width, round(width * mark.height / mark.width)), Image.LANCZOS)


def header(image: Image.Image, title: str, subtitle: str) -> None:
    """
    Draw the header of a card: its title and the grey line under it at the left, the wordmark at the right.

    Parameters
    ----------
    image : PIL.Image.Image
        The card (a ``canvas``).
    title : str
        The title; it shrinks (``TITLE_SIZES``) and then is cut so that it ends ``WORDMARK_GAP`` before the
        wordmark.
    subtitle : str
        The grey line, its parts joined with ``' · '``; it shrinks (``SUBTITLE_SIZES``) and then is cut at the
        right edge.
    """
    draw = ImageDraw.Draw(image)
    mark = wordmark()
    image.paste(mark, (RIGHT * SCALE - mark.width, WORDMARK_TOP * SCALE), mark)
    size, text = fit_text('title', clean(title), *TITLE_SIZES, RIGHT - WORDMARK_WIDTH - WORDMARK_GAP - MARGIN)
    write(draw, 'title', text, MARGIN, TITLE_BASELINE, size, INK)
    # a little more air around the dots than a plain space gives
    size, text = fit_text('body', clean(subtitle).replace(' · ', '  ·  '), *SUBTITLE_SIZES, RIGHT - MARGIN)
    write(draw, 'body', text, MARGIN, SUBTITLE_BASELINE, size, MUTED)


def encode(image: Image.Image) -> bytes:
    """
    The finished card as a PNG.

    Parameters
    ----------
    image : PIL.Image.Image
        The card (a ``canvas``, drawn).

    Returns
    -------
    bytes
        The PNG of the card shrunk to ``SIZE`` with Lanczos; the same card gives the same bytes.
    """
    buffer = io.BytesIO()
    image.resize(SIZE, Image.LANCZOS).save(buffer, 'PNG', compress_level=6)

    return buffer.getvalue()


def name_column(names: list) -> tuple:
    """
    Size of the names of the rows of a card and where its bars start.

    Every name has the same size. When one does not fit at the left of ``BAR_LEFT``, the column grows and the bars
    start further right, up to ``BAR_LEFT_MAX``; past that the names shrink together.

    Parameters
    ----------
    names : list of str
        The names of the rows.

    Returns
    -------
    tuple
        ``(size, bar_left)``: the size of the names (``NAME_SIZES``) and the left edge of the bars, ``NAME_GAP``
        after the widest name. With a name too wide for the smallest size, that size and ``BAR_LEFT_MAX`` (the
        name is then cut).
    """
    def widest(size: int) -> float:
        return max((text_width('bold', name, size) for name in names), default=0)

    full, smallest = NAME_SIZES
    room = BAR_LEFT_MAX - MARGIN - NAME_GAP

    if widest(full) <= room:
        return full, max(BAR_LEFT, math.ceil(MARGIN + widest(full) + NAME_GAP))

    for size in range(full - 1, smallest, -1):
        if widest(size) <= room:
            return size, BAR_LEFT_MAX

    return smallest, BAR_LEFT_MAX


def mark_change(draw: ImageDraw.ImageDraw, change: Optional[dict], x: float, baseline: float) -> None:
    """
    Draw the change of a party against the previous delivery: a triangle and the points, or the equals sign.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Where to draw (a ``canvas``).
    change : dict or None
        ``direction`` and ``text`` (``sondeos.preview_of``); nothing is drawn without it.
    x : float
        Where the mark starts, in pixels of the image.
    baseline : float
        Baseline of its text, in pixels of the image.
    """
    if not change:
        return

    direction, text, size = change.get('direction'), clean(change.get('text')), CHANGE_SIZE

    if direction not in ('up', 'down'):
        write(draw, 'body', text, x, baseline, size, FAINT)

        return

    # a triangle drawn as such: the fonts of the site have no arrows
    fill = UP if direction == 'up' else DOWN
    half, foot = size * .32, baseline - size * .08
    head = foot - size * .56
    flat, tip = (foot, head) if direction == 'up' else (head, foot)
    corners = ((x, flat), (x + 2 * half, flat), (x + half, tip))
    draw.polygon([(point * SCALE, level * SCALE) for point, level in corners], fill=fill)
    write(draw, 'body', text, x + 2 * half + size * .22, baseline, size, fill)


def poll_card(preview: dict) -> Image.Image:
    """
    Draw the card of a poll.

    Parameters
    ----------
    preview : dict
        What the card says (``sondeos.preview_of``): ``title``, ``subtitle``, ``rows``, ``has_seats`` and ``more``.

    Returns
    -------
    PIL.Image.Image
        The card, ``SCALE`` times its size: the header (``header``); a row per party, ``ROWS`` at most, with its
        name, a bar of its colour (``NEUTRAL`` without a valid one) as long as its share against the largest of the
        card (``BAR_MIN`` at least), its share and, after it, its change (``mark_change``); the seats of each party
        in a column at the right, under ``SEATS_LABEL``, only with ``has_seats``; and the footnote ``more`` at the
        bottom right.
    """
    image = canvas()
    draw = ImageDraw.Draw(image)
    header(image, preview.get('title'), preview.get('subtitle'))
    rows = list(preview.get('rows') or [])[:ROWS]
    names = [clean(row.get('name')) for row in rows]
    size, bar_left = name_column(names)
    largest = max((row['value'] for row in rows), default=0)
    has_seats = bool(preview.get('has_seats')) and bool(rows)

    if has_seats:
        write(draw, 'body', SEATS_LABEL, RIGHT, ROW_TOP - 8, LABEL_SIZE, FAINT, right=True)

    for index, (row, name) in enumerate(zip(rows, names)):
        top = ROW_TOP + index * ROW_STEP
        middle = top + BAR_TOP + BAR_HEIGHT / 2
        _, text = fit_text('bold', name, size, size, bar_left - MARGIN - NAME_GAP)
        # the capitals of the name centred on the bar: they are some 0.7 of the size tall
        write(draw, 'bold', text, MARGIN, middle + round(.35 * size), size, INK)
        share = max(row['value'], 0) / largest if largest > 0 else 0
        end = bar_left + max(BAR_MIN, (BAR_RIGHT - bar_left) * share)
        box(draw, bar_left, top + BAR_TOP, end, top + BAR_TOP + BAR_HEIGHT, BAR_RADIUS, colour_of(row.get('color')))
        width = write(draw, 'title', clean(row.get('share')), end + 16, top + 44, SHARE_SIZE, INK)
        mark_change(draw, row.get('change'), end + 16 + width + 18, top + 42)

        if has_seats and row.get('seats') is not None:
            seats_size, text = fit_text('bold', clean(row['seats']), *SEATS_SIZES, SEATS_WIDTH)
            write(draw, 'bold', text, RIGHT, top + 44, seats_size, INK, right=True)

    if preview.get('more'):
        _, text = fit_text('body', clean(preview['more']), LABEL_SIZE, LABEL_SIZE, RIGHT - FOOT_LEFT)
        write(draw, 'body', text, RIGHT, FOOT_BASELINE, LABEL_SIZE, FAINT, right=True)

    return image


def poll_png(preview: dict) -> bytes:
    """
    The image of the link previews of a poll.

    Parameters
    ----------
    preview : dict
        What the image says (``sondeos.preview_of``).

    Returns
    -------
    bytes
        A PNG of ``SIZE`` (``poll_card``, ``encode``).
    """
    return encode(poll_card(preview))
