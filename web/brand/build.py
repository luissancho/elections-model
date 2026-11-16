"""
Build the brand assets of the web from the vendored Outfit (the headings font), instanced at its extra bold weight.

The wordmark is "Demoscopi" set in the font, with its glyphs converted to SVG paths (so it needs no webfont), and
the last "o" replaced by the lens: a ring with a dot in the middle, in the accent colour. The lens alone is the
favicon, so the header and the browser tab share one mark. The image of the link previews (``og:image``) sets the
same wordmark over the tagline and the domain, next to the lens enlarged over a cloud of polls and their average: it
carries no figures, so it never goes stale in the caches of the social networks.

Outputs, in ``web/dist/img/``:

- ``logo.svg``: the wordmark, used by the site header (``_brand.html``).
- ``favicon.svg``: the lens, for the browsers that take an SVG icon.
- ``favicon.ico``: the lens in 16, 32 and 48 px, for the rest (and for ``/favicon.ico``).
- ``apple-touch-icon.png``: the lens on the page background, 180 px, for the iOS home screen.
- ``og-image.png``: the image of the shared links, 1200x675 (``base.html`` declares it as ``og:image``).
- ``wordmark.png``: the wordmark on a transparent background, for the images the server draws
  (``mtpy.lib.pages.images``), which cannot read the SVG.

Run from the repository root::

    python web/brand/build.py

Needs ``fonttools`` with the woff2 extra (``brotli``) and ``Pillow``; neither is a runtime dependency of the web.
"""
import io
import math
import os
import random

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from PIL import Image, ImageColor, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# The variable font the site serves (axis wght 700-800), instanced at the weight of the wordmark
FONT = os.path.join(ROOT, 'web', 'dist', 'vendor', 'fonts', 'Outfit.woff2')
WEIGHT = 800
OUT = os.path.join(ROOT, 'web', 'dist', 'img')

NAME = 'Demoscopio'
WORD = NAME[:-1]  # the glyphs set in the font; the last "o" is the lens
INK = '#1a1c22'  # --color-text
ACCENT = '#1d4f86'  # --color-accent
BACKGROUND = '#f4f5f8'  # --color-bg
# Lens, in font units (1000 per em), relative to the origin of the "o" it replaces: the ring takes the height of
# the "o" of the font, its stroke is a share of that height and the dot a share of the radius (``lens_geometry``)
LENS_STROKE_SHARE = 0.19
LENS_DOT_SHARE = 0.28
PAD = 30  # padding of the wordmark viewBox, in font units
# Favicon box, in SVG user units: the lens fills it leaving a little air
ICON_BOX = 600
ICON_RADIUS = 250
ICON_STROKE = 90
ICON_DOT = 70
# Image of the link previews (og:image), in pixels of the image. X and Facebook show it at about 1.91:1 and cut
# some 23 px at the top and at the bottom: nothing but background goes there
BODY_FONT = os.path.join(ROOT, 'web', 'dist', 'vendor', 'fonts', 'DMSans.woff2')
MUTED = '#5a5f72'  # --color-muted
GLASS = '#ffffff'
OG_FILE = 'og-image.png'
OG_SIZE = (1200, 675)
OG_SUPERSAMPLE = 4
OG_WORDMARK = (84, 226, 580)  # left, top and width of the box of the wordmark (the viewBox of logo.svg)
OG_TAGLINE = ('Promedio de encuestas, previsión de escaños', 'y ranking de encuestadoras')  # SITE_TAGLINE
OG_TAGLINE_AT = (88, (398, 438), 28)  # x, baselines and size: DM Sans, wght 500, opsz 28
OG_DOMAIN = 'demoscopio.es'
OG_DOMAIN_AT = (88, 590, 24)  # x, baseline and size: Outfit, wght 700
OG_LENS = (958, 338, 218, 46)  # centre, outer radius and width of the ring
OG_DOTS = (150, 7, 0.22)  # polls of the cloud, seed and opacity
# The wordmark as an image: file and width in pixels (four times what the image of a poll shows it at)
WORDMARK_PNG = ('wordmark.png', 800)
WORDMARK_SUPERSAMPLE = 4


def number(value: float) -> str:
    """One decimal at most, without a trailing ``.0``: keeps the SVG small."""
    text = '{:.1f}'.format(value)
    return text[:-2] if text.endswith('.0') else text


def instance(path: str, **axes) -> TTFont:
    """
    A font of the site pinned at some values of its axes (a static font is returned as is).

    Parameters
    ----------
    path : str
        The woff2 file.
    **axes
        Value of each axis (``wght``, ``opsz``).

    Returns
    -------
    TTFont
        The font, without its woff2 flavour: it is only saved, if at all, as the plain TTF Pillow reads.
    """
    font = TTFont(path)

    if 'fvar' in font:
        font = instancer.instantiateVariableFont(font, axes, inplace=False)
    font.flavor = None

    return font


def load_font() -> TTFont:
    """
    The wordmark font: ``FONT`` instanced at ``WEIGHT``.

    Returns
    -------
    TTFont
        The font.
    """
    return instance(FONT, wght=WEIGHT)


def glyph_box(font: TTFont, char: str) -> tuple:
    """
    Bounding box of the glyph of a character, in font units.

    Parameters
    ----------
    font : TTFont
        The font.
    char : str
        The character.

    Returns
    -------
    tuple
        ``(xMin, yMin, xMax, yMax)``.
    """
    glyphs = font.getGlyphSet()
    pen = BoundsPen(glyphs)
    glyphs[font.getBestCmap()[ord(char)]].draw(pen)

    return pen.bounds


def lens_geometry(font: TTFont) -> dict:
    """
    Geometry of the lens, from the "o" of the font.

    Parameters
    ----------
    font : TTFont
        The font.

    Returns
    -------
    dict
        ``center`` (relative to the origin of the "o"), ``radius`` (outer), ``stroke`` and ``dot`` (radius of the
        dot), in font units.
    """
    x_min, y_min, x_max, y_max = glyph_box(font, 'o')
    height = y_max - y_min

    return {
        'center': ((x_min + x_max) / 2, (y_min + y_max) / 2),
        'radius': height / 2,
        'stroke': round(LENS_STROKE_SHARE * height),
        'dot': round(LENS_DOT_SHARE * height / 2),
    }


def kerning(font: TTFont) -> dict:
    """
    Pair adjustments of the ``kern`` feature of the GPOS table, flattened.

    Parameters
    ----------
    font : TTFont
        The font.

    Returns
    -------
    dict
        ``{(left, right): x_advance}`` for the pairs the feature adjusts, by glyph name. Format 1 subtables
        give the pairs directly; format 2 subtables are expanded over the glyphs of each class.
    """
    gpos = font['GPOS'].table
    lookups = set()
    for record in gpos.FeatureList.FeatureRecord:
        if record.FeatureTag == 'kern':
            lookups.update(record.Feature.LookupListIndex)
    pairs = {}
    for index in sorted(lookups):
        lookup = gpos.LookupList.Lookup[index]
        for subtable in lookup.SubTable:
            if subtable.LookupType == 9:  # extension: unwrap
                subtable = subtable.ExtSubTable
            if subtable.LookupType != 2:
                continue
            if subtable.Format == 1:
                for left, pair_set in zip(subtable.Coverage.glyphs, subtable.PairSet):
                    for record in pair_set.PairValueRecord:
                        advance = getattr(record.Value1, 'XAdvance', 0) or 0
                        if advance:
                            pairs.setdefault((left, record.SecondGlyph), advance)
            elif subtable.Format == 2:
                class1 = subtable.ClassDef1.classDefs
                class2 = subtable.ClassDef2.classDefs
                rights = {}
                for glyph, klass in class2.items():
                    rights.setdefault(klass, []).append(glyph)
                for left in subtable.Coverage.glyphs:
                    record1 = subtable.Class1Record[class1.get(left, 0)]
                    for klass, record2 in enumerate(record1.Class2Record):
                        advance = getattr(record2.Value1, 'XAdvance', 0) or 0
                        if advance:
                            for right in rights.get(klass, []):
                                pairs.setdefault((left, right), advance)
    return pairs


def glyph_run(font: TTFont, text: str) -> list:
    """
    Positions of the glyphs of a line of text, as the font spaces them.

    Parameters
    ----------
    font : TTFont
        The font.
    text : str
        The text.

    Returns
    -------
    list
        ``[(char, x), ...]``: the origin of each glyph in font units, from the advances of ``hmtx`` and the pair
        adjustments of ``kerning``.
    """
    cmap = font.getBestCmap()
    hmtx = font['hmtx']
    pairs = kerning(font)
    run = []
    x = 0
    previous = None
    for char in text:
        name = cmap[ord(char)]
        if previous is not None:
            x += pairs.get((previous, name), 0)
        run.append((char, x))
        x += hmtx[name][0]
        previous = name
    return run


def wordmark_layout(font: TTFont) -> dict:
    """
    Geometry of the wordmark, on the SVG axis (y grows downwards, the baseline is ``y = 0``), in font units.

    Parameters
    ----------
    font : TTFont
        The wordmark font.

    Returns
    -------
    dict
        ``glyphs`` (``glyph_run`` of ``WORD``, spaced as if its last "o" followed), ``lens`` (``cx``, ``cy``,
        ``radius`` of the middle of the ring, ``stroke`` and ``dot``, in the place of that "o") and ``box``
        (``(left, top, width, height)``: the ink plus ``PAD``).
    """
    lens = lens_geometry(font)
    run = glyph_run(font, WORD + 'o')
    lens_x = run[-1][1]
    left = font['hmtx'][font.getBestCmap()[ord(WORD[0])]][1] - PAD
    right = lens_x + lens['center'][0] + lens['radius'] + PAD
    # the tallest ink of the word (the dot of the "i" or the cap of the "D") and the descender of the "p"
    top = -(max(glyph_box(font, char)[3] for char in set(WORD)) + PAD)
    bottom = -min(glyph_box(font, char)[1] for char in set(WORD)) + PAD
    return {
        'glyphs': run[:-1],
        'lens': {'cx': lens_x + lens['center'][0], 'cy': -lens['center'][1],
                 'radius': lens['radius'] - lens['stroke'] / 2, 'stroke': lens['stroke'], 'dot': lens['dot']},
        'box': (left, top, right - left, bottom - top),
    }


def wordmark() -> str:
    """
    The wordmark as an SVG document.

    Returns
    -------
    str
        SVG with the glyph paths of ``WORD`` and the lens in the place of the last "o" (``wordmark_layout``), y
        flipped to the SVG axis and the viewBox fitted to the ink plus ``PAD``. Its intrinsic size is a tenth of
        the font units (about 100 px tall), which ``_brand.html`` repeats in ``width``/``height`` and the CSS
        scales down.
    """
    font = load_font()
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    layout = wordmark_layout(font)
    lens = layout['lens']
    left, top, width, height = layout['box']

    paths = []
    for char, x in layout['glyphs']:
        pen = SVGPathPen(glyphs, ntos=number)
        glyphs[cmap[ord(char)]].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
        paths.append('<path d="{}"/>'.format(pen.getCommands()))

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="{l} {t} {w} {h}" width="{pw}" height="{ph}" '
        'role="img" aria-labelledby="title">\n'
        '  <title id="title">{name}</title>\n'
        '  <g fill="{ink}">\n    {paths}\n  </g>\n'
        '  <g fill="none" stroke="{accent}" stroke-width="{stroke}">\n'
        '    <circle cx="{cx}" cy="{cy}" r="{r}"/>\n  </g>\n'
        '  <circle fill="{accent}" cx="{cx}" cy="{cy}" r="{dot}"/>\n'
        '</svg>\n'
    ).format(
        l=number(left), t=number(top), w=number(width), h=number(height), pw=number(width / 10),
        ph=number(height / 10), name=NAME, ink=INK,
        paths='\n    '.join(paths), accent=ACCENT, stroke=lens['stroke'], cx=number(lens['cx']),
        cy=number(lens['cy']), r=number(lens['radius']), dot=lens['dot']
    )


def lens_svg() -> str:
    """The lens alone, as a square SVG document (the favicon)."""
    center = ICON_BOX / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {box} {box}">\n'
        '  <circle fill="none" stroke="{accent}" stroke-width="{stroke}" cx="{c}" cy="{c}" r="{r}"/>\n'
        '  <circle fill="{accent}" cx="{c}" cy="{c}" r="{dot}"/>\n'
        '</svg>\n'
    ).format(box=ICON_BOX, accent=ACCENT, stroke=ICON_STROKE, c=number(center),
             r=number(ICON_RADIUS - ICON_STROKE / 2), dot=ICON_DOT)


def lens_png(size: int, background: str = None, scale: float = 1.0, supersample: int = 16) -> Image.Image:
    """
    The lens rasterised.

    Parameters
    ----------
    size : int
        Side of the square image, in pixels.
    background : str, optional
        Background colour; transparent when ``None``.
    scale : float, optional
        Size of the lens box relative to the image (1 fills it like ``lens_svg``).
    supersample : int, optional
        The lens is drawn this many times larger and shrunk with Lanczos, for smooth edges.

    Returns
    -------
    PIL.Image.Image
        RGBA image.
    """
    big = size * supersample
    image = Image.new('RGBA', (big, big), background or (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = big * scale / ICON_BOX  # pixels per SVG unit
    center = big / 2
    stroke = ICON_STROKE * unit
    outer = ICON_RADIUS * unit
    draw.ellipse([center - outer, center - outer, center + outer, center + outer], outline=ACCENT,
                 width=max(1, round(stroke)))
    dot = ICON_DOT * unit
    draw.ellipse([center - dot, center - dot, center + dot, center + dot], fill=ACCENT)
    return image.resize((size, size), Image.LANCZOS)


def raster_font(font: TTFont, size: float) -> ImageFont.FreeTypeFont:
    """
    A font for Pillow, from its TTF in memory.

    Parameters
    ----------
    font : TTFont
        The font (``instance``).
    size : float
        Size of the em, in pixels.

    Returns
    -------
    PIL.ImageFont.FreeTypeFont
        The font with the basic layout: ``draw_text`` places each glyph itself, so the spacing does not depend on
        whether this Pillow has raqm.
    """
    buffer = io.BytesIO()
    font.save(buffer)
    buffer.seek(0)
    return ImageFont.truetype(buffer, size, layout_engine=ImageFont.Layout.BASIC)


def draw_text(draw: ImageDraw.ImageDraw, font: TTFont, text: str, x: float, baseline: float, size: float,
              fill: str) -> None:
    """
    Draw a line of text glyph by glyph, spaced as the font says (``glyph_run``).

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Where to draw.
    font : TTFont
        The font (``instance``).
    text : str
        The text.
    x : float
        Origin of the first glyph, in pixels.
    baseline : float
        Baseline of the text, in pixels.
    size : float
        Size of the em, in pixels.
    fill : str
        Colour.
    """
    raster = raster_font(font, size)
    unit = size / font['head'].unitsPerEm
    for char, offset in glyph_run(font, text):
        draw.text((x + offset * unit, baseline), char, font=raster, fill=fill, anchor='ls')


def draw_wordmark(letters: ImageDraw.ImageDraw, lens: ImageDraw.ImageDraw, font: TTFont, layout: dict, unit: float,
                  origin: float, baseline: float, ink, accent) -> None:
    """
    Draw the wordmark: its letters, glyph by glyph where ``logo.svg`` has them, and the lens in the place of the
    last "o".

    Parameters
    ----------
    letters : PIL.ImageDraw.ImageDraw
        Where the letters go.
    lens : PIL.ImageDraw.ImageDraw
        Where the lens goes (the same canvas, or another when each colour is drawn on its own).
    font : TTFont
        The wordmark font (``load_font``).
    layout : dict
        ``wordmark_layout(font)``.
    unit : float
        Pixels per font unit.
    origin : float
        Pixel of ``x = 0`` of the layout.
    baseline : float
        Pixel of the baseline (``y = 0`` of the layout).
    ink : str or int
        Colour of the letters.
    accent : str or int
        Colour of the lens.
    """
    raster = raster_font(font, font['head'].unitsPerEm * unit)

    for char, x in layout['glyphs']:
        letters.text((origin + x * unit, baseline), char, font=raster, fill=ink, anchor='ls')

    mark = layout['lens']
    cx, cy = origin + mark['cx'] * unit, baseline + mark['cy'] * unit
    outer = (mark['radius'] + mark['stroke'] / 2) * unit
    lens.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], outline=accent, width=round(mark['stroke'] * unit))
    dot = mark['dot'] * unit
    lens.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=accent)


def wordmark_png(width: int) -> Image.Image:
    """
    The wordmark rasterised on a transparent background, with the geometry of ``logo.svg``.

    The letters and the lens are drawn as two masks, ``WORDMARK_SUPERSAMPLE`` times larger and shrunk with
    Lanczos, and each gets its flat colour: the edges fade in opacity only, so the image leaves no dark fringe on
    any background.

    Parameters
    ----------
    width : int
        Width of the image, in pixels: the box of the viewBox of ``logo.svg`` (the ink plus ``PAD``).

    Returns
    -------
    PIL.Image.Image
        RGBA image, ``width`` wide and as tall as the ratio of the box says.
    """
    font = load_font()
    layout = wordmark_layout(font)
    left, top, box_width, box_height = layout['box']
    scale = WORDMARK_SUPERSAMPLE
    unit = width * scale / box_width
    size = (width * scale, round(box_height * unit))
    letters, lens = Image.new('L', size, 0), Image.new('L', size, 0)
    draw_wordmark(ImageDraw.Draw(letters), ImageDraw.Draw(lens), font, layout, unit, -left * unit, -top * unit,
                  255, 255)
    size = (width, round(width * box_height / box_width))
    image = Image.new('RGBA', size, (0, 0, 0, 0))

    for mask, colour in ((letters, INK), (lens, ACCENT)):
        layer = Image.new('RGBA', size, ImageColor.getrgb(colour) + (0,))
        layer.putalpha(mask.resize(size, Image.LANCZOS))
        image = Image.alpha_composite(image, layer)

    return image


def polls_cloud() -> tuple:
    """
    The cloud of polls and their average under the big lens of the share image, in pixels of the image.

    The polls are drawn around the average with a fixed seed, so the image is the same on every build. They are
    no real figures: a slope with a wave, as a share of the radius of the glass.

    Returns
    -------
    tuple
        ``(dots, curve)``: the centres ``(x, y)`` of the ``OG_DOTS`` polls and the points of the average, from
        the left to the right edge of the glass.
    """
    cx, cy, radius, ring = OG_LENS
    inner = radius - ring
    rng = random.Random(OG_DOTS[1])

    def trend(x: float) -> float:
        """The average at ``x``."""
        t = (x - (cx - inner)) / (2 * inner)
        return cy + inner * (.33 - .68 * t + .19 * math.sin(2.6 * math.pi * t))

    dots = []
    for _ in range(OG_DOTS[0]):
        x = rng.uniform(cx - inner, cx + inner)
        dots.append((x, trend(x) + rng.gauss(0, .26 * inner)))
    curve = [(x, trend(x)) for x in (cx - inner + i * 2 * inner / 80 for i in range(81))]

    return dots, curve


def draw_lens(image: Image.Image, scale: int) -> None:
    """
    Draw the big lens of the share image: the ring, and inside it the glass with the cloud of polls and their
    average (``polls_cloud``), clipped to the glass.

    Parameters
    ----------
    image : PIL.Image.Image
        The canvas, ``scale`` times the image.
    scale : int
        Pixels of the canvas per pixel of the image.
    """
    cx, cy, radius, ring = OG_LENS
    inner = radius - ring
    left, top = (cx - inner) * scale, (cy - inner) * scale
    side = 2 * inner * scale
    dots, curve = polls_cloud()

    glass = Image.new('RGB', (side, side), GLASS)
    pen = ImageDraw.Draw(glass, 'RGBA')  # the dots are translucent: where two overlap the blue adds up
    dot = .037 * inner * scale
    colour = ImageColor.getrgb(ACCENT) + (round(255 * OG_DOTS[2]),)
    for x, y in dots:
        x, y = x * scale - left, y * scale - top
        pen.ellipse([x - dot, y - dot, x + dot, y + dot], fill=colour)
    pen.line([(x * scale - left, y * scale - top) for x, y in curve], fill=ACCENT, width=round(.051 * inner * scale),
             joint='curve')
    mask = Image.new('L', (side, side), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, side - 1, side - 1], fill=255)

    # the ring is what the glass leaves of a disc of the outer radius: no seam between them
    outer = radius * scale
    ImageDraw.Draw(image).ellipse([cx * scale - outer, cy * scale - outer, cx * scale + outer, cy * scale + outer],
                                  fill=ACCENT)
    image.paste(glass, (left, top), mask)


def og_image() -> Image.Image:
    """
    The image of the link previews (``og:image``): the wordmark, the tagline and the domain at the left and the
    lens over a cloud of polls at the right, on the page background.

    It is drawn ``OG_SUPERSAMPLE`` times larger and shrunk with Lanczos, for smooth edges. The texts are set in
    the fonts of the site (``draw_text``) and the wordmark has the geometry of ``logo.svg`` (``wordmark_layout``).

    Returns
    -------
    PIL.Image.Image
        RGB image of ``OG_SIZE``.
    """
    scale = OG_SUPERSAMPLE
    image = Image.new('RGB', (OG_SIZE[0] * scale, OG_SIZE[1] * scale), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # wordmark: a point (x, y) of ``wordmark_layout`` falls at (origin + x * unit, baseline + y * unit)
    font = load_font()
    layout = wordmark_layout(font)
    box_left, box_top, box_width, _ = layout['box']
    mark_left, mark_top, mark_width = OG_WORDMARK
    unit = mark_width * scale / box_width  # pixels of the canvas per font unit
    origin, baseline = mark_left * scale - box_left * unit, mark_top * scale - box_top * unit
    draw_wordmark(draw, draw, font, layout, unit, origin, baseline, INK, ACCENT)

    x, baselines, size = OG_TAGLINE_AT
    body = instance(BODY_FONT, wght=500, opsz=size)
    for line, line_baseline in zip(OG_TAGLINE, baselines):
        draw_text(draw, body, line, x * scale, line_baseline * scale, size * scale, MUTED)
    x, domain_baseline, size = OG_DOMAIN_AT
    draw_text(draw, instance(FONT, wght=700), OG_DOMAIN, x * scale, domain_baseline * scale, size * scale, ACCENT)

    draw_lens(image, scale)

    return image.resize(OG_SIZE, Image.LANCZOS)


def build() -> None:
    """Write the six assets to ``OUT``."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'logo.svg'), 'w', encoding='utf-8') as file:
        file.write(wordmark())
    with open(os.path.join(OUT, 'favicon.svg'), 'w', encoding='utf-8') as file:
        file.write(lens_svg())
    icons = {size: lens_png(size) for size in (48, 32, 16)}
    icons[48].save(os.path.join(OUT, 'favicon.ico'), sizes=[(48, 48), (32, 32), (16, 16)],
                   append_images=[icons[32], icons[16]])
    lens_png(180, background=BACKGROUND, scale=0.72).save(os.path.join(OUT, 'apple-touch-icon.png'), optimize=True)
    og_image().save(os.path.join(OUT, OG_FILE), optimize=True)
    wordmark_png(WORDMARK_PNG[1]).save(os.path.join(OUT, WORDMARK_PNG[0]), optimize=True)


if __name__ == '__main__':
    build()
    print('brand assets written to', os.path.relpath(OUT, os.getcwd()))
