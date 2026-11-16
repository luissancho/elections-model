# Web: datos estructurados

Cambio pequeño, sexto paso de la ronda de SEO y redes sociales. Luis dio paso a seguir con la ronda el 2026-11-16
(«Vale, continúa»); el diseño no lo ha revisado: lo que no viene de él está en «Decisiones mías, por revisar».
Plan: `docs/superpowers/plans/2026-11-16-web-datos-estructurados.md`.

## Intención

Decirle a Google, en su propio formato, lo que un lector ya ve en la página: cómo se llama el sitio y quién lo
publica, que un estudio es un artículo con su autor y su fecha, y en qué sección está cada página. Es lo que decide
el nombre que sale sobre cada resultado («Demoscopio» en vez de «demoscopio.es»), la fecha y el autor de un
estudio y la línea de la sección bajo el título.

## Hechos de partida (2026-11-16, documentación de Google leída ese día)

- Ninguna página lleva datos estructurados (`application/ld+json`). Sí `og:site_name`.
- **Nombre del sitio**: Google lo toma sobre todo de un `WebSite` (`name`, `url`) en la portada, y sólo en la
  portada.
- **`Organization`**: sin propiedades obligatorias; recomendadas `name`, `url`, `logo` (112×112 px como mínimo,
  que se vea bien sobre blanco), `description`. En la portada.
- **`Article`**: sin obligatorias; recomendadas `headline`, `author` (`name` y `url`), `datePublished`,
  `dateModified` e `image` (una imagen del artículo, no un logotipo).
- **`BreadcrumbList`**: al menos dos elementos con `position` y `name`; el último puede ir sin `item` (Google usa
  la dirección de la página). Sólo se enseña en ordenador.
- **`Dataset`**: es para Google Dataset Search; pide `name` y `description`, y recomienda `license`.
- El pie de todas las páginas ya dice «Modelo y curación: Luis Sancho (@luis_sancho)».
- Cuatro estudios enseñan «Página actualizada el …» con el `published_at` del manifiesto. La fecha de la primera
  publicación de un estudio no está guardada en ningún sitio.
- `web/dist/img/apple-touch-icon.png` es la lente de la marca, 180×180 sobre fondo casi blanco.

## Enfoques

1. **JSON-LD compuesto en Python y pintado por `base.html`** (elegido): `mtpy/lib/pages/structured.py` construye
   cada bloque como un diccionario, el controlador de la página los pasa en `structured`, y la plantilla base los
   escribe con `tojson`, que escapa lo que rompería la etiqueta. Los textos salen de los que la página ya enseña.
2. Bloques escritos a mano en cada plantilla. Descartado: un nombre con comillas o `</script>` rompe el JSON.
3. Microdatos en el HTML (`itemscope`, `itemprop`). Descartado: toca el marcado visible de todas las tarjetas.

## Diseño

| Página | Bloques |
|---|---|
| `/` | `WebSite` y `Organization` |
| un estudio (`/analisis/…`) | `Article` y `BreadcrumbList` |
| una página del método (`/metodo/…`) | `BreadcrumbList` |
| el perfil de una encuestadora | `BreadcrumbList` |
| una elección pasada | `BreadcrumbList` |
| un sondeo | `BreadcrumbList` |

Las demás (el evento, `/elecciones`, `/casas`, `/metodo`, `/analisis`, `/privacidad`) no llevan ninguno: son el
primer nivel y no son artículos.

- **`WebSite`**: `name` «Demoscopio», `url` la portada, `inLanguage` «es», `description` el lema del sitio.
- **`Organization`**: `name`, `url`, `logo` (la lente, `apple-touch-icon.png`), `description` (el lema) y
  `founder`, una `Person` con `name` «Luis Sancho» y `url` su cuenta de X (la del pie).
- **`Article`** de un estudio: `headline` (el titular visible de la página), `description` (la misma que
  `<meta name="description">`), `url` y `mainEntityOfPage` (la canónica), `inLanguage`, `author` (la misma
  `Person`), `publisher` (la organización, con su logotipo) y `dateModified` (el `published_at` del manifiesto,
  la fecha que la página enseña como «actualizada»; sin ella, sin la propiedad). Sin `datePublished` ni `image`:
  no hay fecha de primera publicación guardada, y la imagen de marca no es una imagen del artículo.
- **`BreadcrumbList`**, dos elementos: la sección y la página.

  | Página | Primer elemento | Segundo |
  |---|---|---|
  | un estudio | Análisis → `/analisis` | el titular |
  | una página del método | Método → `/metodo` | su titular |
  | una encuestadora | Ranking Encuestadoras → `/casas` | su nombre |
  | una elección pasada | Histórico Errores → `/elecciones` | su nombre, o su fecha |
  | un sondeo | la elección («Elecciones generales 2026») → `/es` | «Sigma Dos para El Mundo · 5 oct 2026» |

  Los nombres del primer elemento son las etiquetas del menú; todas las direcciones son las canónicas.

### Código

- `mtpy/lib/pages/structured.py`: `website`, `organization`, `article`, `breadcrumbs` (devuelven diccionarios) y
  `block_text` (el texto de un bloque de una plantilla con el contexto de la página, para el titular).
- `mtpy/lib/pages/common.py`: `AUTHOR_NAME`.
- `Page.render` añade `structured` (lista vacía por defecto); los controladores `Index`, `Analisis`, `Metodo`,
  `Casas`, `Elecciones` y `Sondeos` pasan los suyos.
- `web/templates/base.html`: un `<script type="application/ld+json">` por bloque, tras las etiquetas `og:`.

### Pruebas

`tests/test_pages_structured.py`:

- Cada página del sitemap, y `/privacidad`: todo bloque es JSON válido con `@context` `https://schema.org`, y cada
  página lleva exactamente los tipos de la tabla.
- La portada: `WebSite` y `Organization` con sus campos; el logotipo existe y mide al menos 112×112.
- Un estudio: `headline` igual a su `<h1>` visible, `description` igual a su `<meta name="description">`, `url`
  igual a su canónica, `dateModified` del manifiesto (y sin ella si el manifiesto no la trae).
- Las migas de cada tipo de página: nombres, direcciones canónicas y posiciones.
- Un nombre con `</script>`, comillas y `&`: el JSON se lee de vuelta con el nombre exacto y la etiqueta no se
  cierra antes de tiempo.

## Fuera de este paso

- **`Dataset`** para las descargas: pide decidir la licencia de los datos del modelo (los sondeos vienen de
  Wikipedia, CC BY-SA 4.0).
- `datePublished` de los estudios: haría falta guardar la fecha de su primera publicación.
- Migas visibles en la página, `FAQPage` (Google sólo la enseña en webs de salud y de gobiernos) y `Event` (es
  para eventos a los que se asiste).
- Resúmenes con cifras para las páginas que no cambian: sigue en la ronda.

## Decisiones mías, por revisar

- **Demoscopio como `Organization`** con Luis Sancho como `founder` y como autor de los estudios, con su nombre y
  su cuenta de X (lo que ya dice el pie).
- **El logotipo de la organización es la lente** (el icono cuadrado), no la marca con el nombre.
- **Qué páginas llevan qué** (la tabla), y que las migas usen las etiquetas del menú.
- **Los estudios sin `datePublished`.**
