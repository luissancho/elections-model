# Web: imagen de marca para los enlaces compartidos — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que cualquier enlace de la web compartido en X, Facebook o mensajería salga con una imagen de marca de 16:9.

**Architecture:** `web/brand/build.py` gana una salida, `web/dist/img/og-image.png`, dibujada con Pillow a partir del mismo logotipo, colores y fuentes que el resto de la marca. `mtpy/lib/pages/common.py` da su URL absoluta con una versión (hash del fichero); `Page.render` la pasa a todas las plantillas como `og_image` y `base.html` la declara junto a la tarjeta grande de X.

**Tech Stack:** Python 3.11 (`python` es el 3.11.9 de pyenv), Pillow 10.3, fontTools (con brotli, sólo para `build.py`), Jinja2, pytest. Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-11-16-web-imagen-enlaces-design.md`

## Global Constraints

- La imagen es un PNG RGB de 1200×675, de menos de 300 kB, sin nada más que fondo en las franjas de 24 px de arriba y de abajo.
- Colores: fondo `#f4f5f8`, tinta `#1a1c22`, acento `#1d4f86`, lema `#5a5f72`, cristal `#ffffff`.
- Sin cifras ni colores de partido en la imagen.
- `requirements.txt` no cambia; `build.py` sigue sin ser dependencia de la web (no lo importa nada de `mtpy`).
- `logo.svg`, `favicon.svg`, `favicon.ico` y `apple-touch-icon.png` no cambian.
- `error.html` no cambia: la página de error sigue sin etiquetas `og:`.
- Identificadores en inglés; textos públicos en español. PEP8, docstrings numpydoc en todas las funciones, líneas ≤ 120.
- Las pruebas no tocan la RDS ni escriben en `backtest/results`.
- Luis hace commits en paralelo en el mismo árbol: antes de cada commit, `git status` y `git log -1`; añadir sólo los ficheros de la tarea, nunca `git commit -a` ni reescribir sus commits.

## Review Focus

1. **Falta el fichero de la imagen en el servidor** (despliegue incompleto): las páginas siguen respondiendo 200 y la URL sale sin `?v=`. → Tarea 2, `test_share_image_url_without_the_file_has_no_version`.
2. **`site_url` con barra final o con otro host** (`demoscopio.dev` en local): URL absoluta sin doble barra. → Tarea 2, `test_pages_carry_the_share_image_and_the_large_card`.
3. **Contenido en lo que recortan X y Facebook**: las dos franjas son sólo fondo. → Tarea 1, `test_og_image_is_the_16_9_brand_image`.
4. **Regenerar la marca cambia el logotipo o los favicons**: tras ejecutar `build.py`, `git status` sólo enseña el fichero nuevo. → Tarea 1, paso 4.
5. **Otra máquina, otro motor de composición de texto** (Pillow con o sin raqm): las posiciones de las letras no dependen de él. → Tarea 1, decisión de dibujo (composición `BASIC` y kerning propio).

---

### Task 1: la imagen (`web/brand/build.py` → `web/dist/img/og-image.png`)

**Files:**
- Modify: `web/brand/build.py`
- Create: `web/dist/img/og-image.png` (generado por el script)
- Modify: `deploy/README.md:238-245` (párrafo «Marca»)
- Test: `tests/test_deploy_files.py`

**Interfaces:**
- Produces: el fichero `web/dist/img/og-image.png`, PNG RGB de 1200×675. La Tarea 2 sólo usa el fichero.

- [ ] **Step 1: escribir la prueba que falla**, al final de los tests de marca de `tests/test_deploy_files.py`:

```python
def test_og_image_is_the_16_9_brand_image():
    """La imagen de los enlaces compartidos existe, es un PNG de 1200×675 ligero, deja sólo fondo en las franjas que
    recortan X y Facebook, lleva el logotipo y la lente donde dice la spec y repite el lema del sitio."""
    path = ROOT / 'web' / 'dist' / 'img' / 'og-image.png'
    assert path.stat().st_size < 300_000
    background, ink, accent = (244, 245, 248), (26, 28, 34), (29, 79, 134)
    with Image.open(path) as png:
        assert (png.format, png.mode, png.size) == ('PNG', 'RGB', (1200, 675))
        for band in ((0, 0, 1200, 24), (0, 651, 1200, 675)):
            assert png.crop(band).getcolors() == [(1200 * 24, background)]
        assert png.getpixel((96, 267)) == ink  # the stem of the "D" of the wordmark
        assert png.getpixel((763, 338)) == accent  # the ring of the lens, at its left
        assert png.getpixel((1190, 338)) == background  # nothing bleeds off the right edge
    source = read('web/brand/build.py')
    lines = re.search(r"OG_TAGLINE = \('([^']+)', '([^']+)'\)", source).groups()
    assert "SITE_TAGLINE = '{}'".format(' '.join(lines)) in read('mtpy/lib/pages/common.py')
    assert "OG_DOMAIN = 'demoscopio.es'" in source and "'og-image.png'" in source
```

- [ ] **Step 2: ejecutarla y verla fallar**

Run: `python -m pytest tests/test_deploy_files.py::test_og_image_is_the_16_9_brand_image -q`
Expected: FAIL con `FileNotFoundError` (no existe `og-image.png`).

- [ ] **Step 3: implementar la imagen en `web/brand/build.py`**

Constantes nuevas (valores de la spec):

```python
BODY_FONT = os.path.join(ROOT, 'web', 'dist', 'vendor', 'fonts', 'DMSans.woff2')
MUTED = '#5a5f72'  # --color-muted
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
```

Funciones (los cuerpos, a criterio del implementador salvo lo indicado):

- `glyph_run(font: TTFont, text: str) -> list`: `[(char, x), ...]`, el origen de cada glifo en unidades de la fuente, con los avances de `hmtx` y los pares de `kerning(font)`. Es el bucle que hoy vive dentro de `wordmark()`.
- `wordmark_layout(font: TTFont) -> dict`: `{'glyphs': glyph_run de WORD + 'o', 'lens': {'cx', 'cy', 'radius', 'stroke', 'dot'}, 'box': (left, top, width, height)}` con los valores que `wordmark()` calcula hoy (eje y del SVG; `radius` es el del centro del trazo). `wordmark()` pasa a escribirse a partir de él y **su salida no cambia ni en un byte**.
- `og_image() -> Image.Image`: la imagen RGB de 1200×675.
- `build()` escribe además `og_image().save(os.path.join(OUT, OG_FILE), optimize=True)`; el docstring del módulo y su lista «Outputs» lo recogen.

Decisiones de dibujo de `og_image()`:

- Se dibuja en un lienzo `OG_SUPERSAMPLE` veces mayor y se reduce con `Image.LANCZOS`, como `lens_png`.
- Textos con las fuentes de la web instanciadas con `instancer.instantiateVariableFont` (Outfit `wght` 800 para el logotipo y 700 para el dominio; DM Sans `wght` 500 y `opsz` 28 para el lema), guardadas como TTF en memoria (`font.flavor = None`, `BytesIO`) y cargadas con `ImageFont.truetype(buffer, size, layout_engine=ImageFont.Layout.BASIC)`.
- Cada glifo se dibuja por separado en la posición de `glyph_run` (`anchor='ls'`), para que el espaciado sea el de la fuente (el del navegador) y no dependa de que Pillow tenga raqm.
- Logotipo: con `k = 580 / box.width`, un punto `(x, y)` del SVG de `wordmark_layout` cae en el píxel `(84 + (x - box.left)·k, 226 + (y - box.top)·k)`; los glifos miden `1000·k` px y su línea base es `y = 0`. La lente del logotipo, con `ACCENT`, a partir de `lens`.
- Lente grande: el cristal y su contenido se dibujan en una capa y se pegan con una máscara circular de radio `218 − 46 = 172`; el aro se pinta después encima. La nube y la curva son exactamente éstas (coordenadas de la imagen, antes de escalar):

```python
cx, cy, radius, ring = OG_LENS
inner = radius - ring
rng = random.Random(OG_DOTS[1])


def trend(x):
    """The average under the lens: a slope with a wave, as a share of the radius of the glass."""
    t = (x - (cx - inner)) / (2 * inner)
    return cy + inner * (.33 - .68 * t + .19 * math.sin(2.6 * math.pi * t))


dots = []
for _ in range(OG_DOTS[0]):
    x = rng.uniform(cx - inner, cx + inner)
    dots.append((x, trend(x) + rng.gauss(0, .26 * inner)))  # each a disc of radius .037 * inner
curve = [(x, trend(x)) for x in (cx - inner + i * 2 * inner / 80 for i in range(81))]  # width .051 * inner
```

  Los discos, en `ACCENT` con opacidad 0,22 que se acumula donde se pisan (`ImageDraw.Draw(layer, 'RGBA')`); la curva, en `ACCENT` opaco con `joint='curve'`.

- [ ] **Step 4: generar y comprobar que sólo aparece el fichero nuevo**

Run: `python web/brand/build.py && git status --short web/dist/img`
Expected: una sola línea, `?? web/dist/img/og-image.png`. Si `logo.svg` o `favicon.svg` salen modificados, el cambio de `wordmark()` no es neutro: corregirlo. Si sólo cambian los bytes de `favicon.ico` o `apple-touch-icon.png` (otra versión del codificador), `git restore` de esos dos.

- [ ] **Step 5: pasar las pruebas de marca**

Run: `python -m pytest tests/test_deploy_files.py -q`
Expected: PASS (12 tests).

- [ ] **Step 6: mirar la imagen**

Abrir `web/dist/img/og-image.png` y repasarla contra la lista «La imagen» de la spec: logotipo nítido y con la lente en su sitio, lema en dos líneas, dominio, lente con la nube y la curva recortadas al cristal. Referencia visual local, no versionada: la tarjeta «D1» de `.superpowers/brainstorm/26542-1794823103/content/brand-image-v3.html`.

- [ ] **Step 7: documentar** en el párrafo «Marca» de `deploy/README.md`: son cinco ficheros; `og-image.png` es la imagen de 1200×675 de los enlaces compartidos (logotipo, lema, dominio y la lente sobre una nube de sondeos), sin cifras para que no caduque en la caché de las redes.

- [ ] **Step 8: commit**

```bash
git add web/brand/build.py web/dist/img/og-image.png tests/test_deploy_files.py deploy/README.md
git commit -m "Marca: build.py genera og-image.png, la imagen de 1200×675 de los enlaces compartidos (logotipo, lema, dominio y la lente sobre una nube de sondeos y su promedio, sin cifras)"
```

---

### Task 2: las etiquetas (`og:image` y la tarjeta grande de X)

**Files:**
- Modify: `mtpy/lib/pages/common.py` (constantes y dos funciones, junto a `SITE_TITLE`)
- Modify: `mtpy/lib/pages/__init__.py` (reexportar los nombres nuevos)
- Modify: `mtpy/controllers/Page.py:107-126` (`render`)
- Modify: `web/templates/base.html:14-15`
- Modify: `deploy/README.md:244-245` (frase de la canónica y `og:url`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `web/dist/img/og-image.png` (Tarea 1).
- Produces: `pages.OG_IMAGE = 'og-image.png'`; `pages.common.OG_IMAGE_PATH` (ruta absoluta del fichero bajo `REPO_ROOT/web/dist/img`); `pages.og_image_version() -> str` (8 primeros hex del MD5 del fichero, `''` si no existe; `functools.lru_cache`); `pages.og_image_url() -> str`; variable de plantilla `og_image` en toda página que pinta `Page.render`.

- [ ] **Step 1: escribir las pruebas que fallan**, en `tests/test_pages.py` tras `test_canonical_drops_the_view_parameters_and_keeps_the_content_ones` (añadir `import hashlib` arriba):

```python
def test_pages_carry_the_share_image_and_the_large_card(api, fresh_app):
    """Toda página declara la imagen de marca de los enlaces compartidos (`og:image` absoluta según `web.site_url`,
    con la versión del fichero, su tamaño y su texto alternativo) y la tarjeta grande de X; la de error, no."""
    status, headers, body = call(api, '/es')
    page = html(body)
    image = re.search(r'<meta property="og:image" content="([^"]+)">', page).group(1)
    with open(os.path.join(ROOT, 'web', 'dist', 'img', 'og-image.png'), 'rb') as file:
        version = hashlib.md5(file.read()).hexdigest()[:8]
    assert status == 200 and image == 'https://demoscopio.es/dist/img/og-image.png?v=' + version
    assert '<meta property="og:image:type" content="image/png">' in page
    assert '<meta property="og:image:width" content="1200">' in page
    assert '<meta property="og:image:height" content="675">' in page
    assert ('<meta property="og:image:alt" content="Demoscopio: promedio de encuestas, previsión de escaños y ranking '
            'de encuestadoras">') in page
    assert '<meta name="twitter:card" content="summary_large_image">' in page and 'content="summary"' not in page
    for path, query in (('/', ''), ('/metodo', 'scope=es')):
        assert '<meta property="og:image" content="{}">'.format(image) in html(call(api, path, query=query)[2])
    assert 'og:image' not in html(call(api, '/nope')[2])
    from mtpy.core.app import Config
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/'}}))
    assert ('<meta property="og:image" content="https://demoscopio.dev/dist/img/og-image.png?v={}">'.format(version)
            in html(call(api, '/es')[2]))


def test_share_image_url_without_the_file_has_no_version(api, monkeypatch, tmp_path):
    """Sin el fichero de la imagen (un despliegue incompleto) las páginas no fallan: la URL sale sin versión."""
    monkeypatch.setattr(pages.common, 'OG_IMAGE_PATH', str(tmp_path / 'missing.png'))
    pages.og_image_version.cache_clear()
    try:
        assert pages.og_image_url() == 'https://demoscopio.es/dist/img/og-image.png'
        assert call(api, '/es')[0] == 200
    finally:
        pages.og_image_version.cache_clear()
```

- [ ] **Step 2: ejecutarlas y verlas fallar**

Run: `python -m pytest tests/test_pages.py -q -k "share_image"`
Expected: FAIL (2 tests): `AttributeError` en `re.search(...).group` y en `pages.common.OG_IMAGE_PATH`.

- [ ] **Step 3: implementar**

- `mtpy/lib/pages/common.py`: `OG_IMAGE`, `OG_IMAGE_PATH`, `og_image_version()` y `og_image_url() -> str`, que devuelve `'<site_url>/dist/img/og-image.png?v=<version>'` (sin `?v=` cuando la versión es `''`); `site_url` sale de `settings()`, ya sin barra final. `og_image_version` lee `OG_IMAGE_PATH` en cada llamada no cacheada (la prueba lo cambia con `monkeypatch`).
- `mtpy/lib/pages/__init__.py`: añadir `OG_IMAGE`, `og_image_url` y `og_image_version` a lo que se importa de `.common`.
- `Page.render`: `context.setdefault('og_image', pages.og_image_url())` junto al de `canonical`, y su docstring.
- `web/templates/base.html`, tras `og:locale`, sustituyendo la línea de `twitter:card`:

```html
  <meta property="og:image" content="{{ og_image }}">
  <meta property="og:image:type" content="image/png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="675">
  <meta property="og:image:alt" content="{{ site_title }}: {{ site_tagline|lower }}">
  <meta name="twitter:card" content="summary_large_image">
```

- [ ] **Step 4: pasar las pruebas**

Run: `python -m pytest tests/test_pages.py tests/test_deploy_files.py -q`
Expected: PASS, sin ningún fallo nuevo en el resto de `test_pages.py`.

- [ ] **Step 5: documentar** en `deploy/README.md`, tras la frase de la canónica: cada página declara `og:image` (`pages.og_image_url`: la imagen de marca con `?v=` y el hash del fichero, para que las redes pidan la nueva cuando cambie) y la tarjeta grande de X; un controlador puede pasar otra imagen a `render` con `og_image=`.

- [ ] **Step 6: commit**

```bash
git add mtpy/lib/pages/common.py mtpy/lib/pages/__init__.py mtpy/controllers/Page.py web/templates/base.html tests/test_pages.py deploy/README.md
git commit -m "Web: las páginas declaran la imagen de marca de los enlaces compartidos (og:image absoluta con la versión del fichero, tamaño y texto alternativo) y la tarjeta grande de X"
```

---

## Cierre

- [ ] Run: `python -m pytest -q` — Expected: PASS (todo el conjunto, sin los de integración).
- [ ] Tras el despliegue (lo hace Luis): los tres pasos de «Comprobación tras desplegar» de la spec.
