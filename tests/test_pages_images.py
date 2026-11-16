"""Tests de la imagen propia del enlace de un sondeo: la marca con la que dibuja el servidor, lo que la imagen dice
(como datos), el dibujo y su dirección."""
import os
import re

from PIL import Image

from tests.test_webapi_unit import ROOT

IMG_DIR = os.path.join(ROOT, 'web', 'dist', 'img')
INK = (26, 28, 34)  # --color-text
ACCENT = (29, 79, 134)  # --color-accent


def near(pixel, colour, tolerance=2):
    """Si un píxel tiene ese color, con el margen del redondeo al reducir la imagen."""
    return all(abs(a - b) <= tolerance for a, b in zip(pixel[:3], colour))


def test_wordmark_is_a_transparent_image_of_the_logo():
    """La marca en PNG es el logotipo de `logo.svg` sobre fondo transparente: 800 px de ancho, la proporción de su
    viewBox y sólo sus dos colores (la tinta de las letras y el azul de la lente)."""
    with open(os.path.join(IMG_DIR, 'logo.svg'), encoding='utf-8') as file:
        box = [float(number) for number in re.search(r'viewBox="([^"]*)"', file.read()).group(1).split()]

    image = Image.open(os.path.join(IMG_DIR, 'wordmark.png'))
    width, height = image.size
    assert image.mode == 'RGBA' and width == 800
    assert abs(height - 800 * box[3] / box[2]) <= 1
    corners = ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
    assert all(image.getpixel(corner)[3] == 0 for corner in corners)
    opaque = {pixel[:3] for pixel in image.getdata() if pixel[3] == 255}
    assert any(near(pixel, INK) for pixel in opaque) and any(near(pixel, ACCENT) for pixel in opaque)
    assert all(near(pixel, INK) or near(pixel, ACCENT) for pixel in opaque)
    # the letters at the left and the lens at the right end, as in the logo
    assert near(max(opaque, key=lambda pixel: pixel[2]), ACCENT)
    left = {pixel[:3] for pixel in image.crop((0, 0, width // 2, height)).getdata() if pixel[3] == 255}
    assert left and all(near(pixel, INK) for pixel in left)
