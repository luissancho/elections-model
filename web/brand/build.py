"""
Build the brand assets of the web from the vendored Source Serif 4 Semibold (the headings font).

The wordmark is "Demoscopi" set in the font, with its glyphs converted to SVG paths (so it needs no webfont), and
the last "o" replaced by the lens: a ring with a dot in the middle, in the accent colour. The lens alone is the
favicon, so the header and the browser tab share one mark.

Outputs, in ``web/dist/img/``:

- ``logo.svg``: the wordmark, used by the site header (``_brand.html``).
- ``favicon.svg``: the lens, for the browsers that take an SVG icon.
- ``favicon.ico``: the lens in 16, 32 and 48 px, for the rest (and for ``/favicon.ico``).
- ``apple-touch-icon.png``: the lens on the page background, 180 px, for the iOS home screen.

Run from the repository root::

    python web/brand/build.py

Needs ``fonttools`` with the woff2 extra (``brotli``) and ``Pillow``; neither is a runtime dependency of the web.
"""
import os

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FONT = os.path.join(ROOT, 'web', 'dist', 'vendor', 'fonts', 'SourceSerif4-4.005-Semibold.woff2')
OUT = os.path.join(ROOT, 'web', 'dist', 'img')

NAME = 'Demoscopio'
WORD = NAME[:-1]  # the glyphs set in the font; the last "o" is the lens
INK = '#1a1c22'  # --color-text
ACCENT = '#1d4f86'  # --color-accent
BACKGROUND = '#f4f5f8'  # --color-bg
# Lens, in font units (1000 per em), relative to the origin of the "o" it replaces. The "o" of the font is
# 482 wide and 507 tall (from -14 to 493), with 130-unit sides and 48-unit top and bottom: the ring takes its
# height and a uniform stroke in between.
LENS_CENTER = (278, 240)
LENS_RADIUS = 253  # outer radius
LENS_STROKE = 90
LENS_DOT = 70  # radius of the dot
PAD = 30  # padding of the wordmark viewBox, in font units
# Favicon box, in SVG user units: the lens fills it leaving a little air
ICON_BOX = 600
ICON_RADIUS = 250
ICON_STROKE = 90
ICON_DOT = 70


def number(value: float) -> str:
    """One decimal at most, without a trailing ``.0``: keeps the SVG small."""
    text = '{:.1f}'.format(value)
    return text[:-2] if text.endswith('.0') else text


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


def wordmark() -> str:
    """
    The wordmark as an SVG document.

    Returns
    -------
    str
        SVG with the glyph paths of ``WORD`` and the lens in the place of the last "o", y flipped to the SVG
        axis and the viewBox fitted to the ink plus ``PAD``. Its intrinsic size is a tenth of the font units
        (about 100 px tall), which ``_brand.html`` repeats in ``width``/``height`` and the CSS scales down.
    """
    font = TTFont(FONT)
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    hmtx = font['hmtx']
    pairs = kerning(font)
    names = [cmap[ord(char)] for char in WORD + 'o']

    paths = []
    x = 0
    for position, name in enumerate(names):
        if position:
            x += pairs.get((names[position - 1], name), 0)
        if position < len(names) - 1:
            pen = SVGPathPen(glyphs, ntos=number)
            glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
            paths.append('<path d="{}"/>'.format(pen.getCommands()))
        else:
            lens_x = x
        x += hmtx[name][0]

    cx, cy = lens_x + LENS_CENTER[0], -LENS_CENTER[1]
    left = hmtx[names[0]][1] - PAD
    right = lens_x + LENS_CENTER[0] + LENS_RADIUS + PAD
    top = -(723 + PAD)  # the dot of the "i" is the tallest ink of the word
    bottom = 234 + PAD  # the descender of the "p"
    width, height = right - left, bottom - top
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
        paths='\n    '.join(paths), accent=ACCENT, stroke=LENS_STROKE, cx=number(cx), cy=number(cy),
        r=number(LENS_RADIUS - LENS_STROKE / 2), dot=LENS_DOT
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


def build() -> None:
    """Write the four assets to ``OUT``."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'logo.svg'), 'w', encoding='utf-8') as file:
        file.write(wordmark())
    with open(os.path.join(OUT, 'favicon.svg'), 'w', encoding='utf-8') as file:
        file.write(lens_svg())
    icons = {size: lens_png(size) for size in (48, 32, 16)}
    icons[48].save(os.path.join(OUT, 'favicon.ico'), sizes=[(48, 48), (32, 32), (16, 16)],
                   append_images=[icons[32], icons[16]])
    lens_png(180, background=BACKGROUND, scale=0.72).save(os.path.join(OUT, 'apple-touch-icon.png'), optimize=True)


if __name__ == '__main__':
    build()
    print('brand assets written to', os.path.relpath(OUT, os.getcwd()))
