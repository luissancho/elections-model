# Web: títulos de las páginas y `<h1>`

Cambio pequeño, tercer paso de la ronda de SEO y redes sociales. Luis dio paso a la tarea el 2026-11-16 («Sigue con
esto»); el diseño y los textos no los ha revisado: todo lo que no viene de él está en «Decisiones mías, por
revisar». Plan: `docs/superpowers/plans/2026-11-16-web-titulos-h1.md`.

## Intención

Que el título de cada página diga qué hay en ella con las palabras que la gente busca, y que cada página tenga un
solo `<h1>`, el suyo. El título es lo que Google enseña como enlace y lo que queda en la pestaña del navegador; el
`<h1>` es por donde un lector de pantalla y un buscador entienden de qué va la página.

## Hechos de partida (2026-11-16)

- `base.html` compone `<title>` y `og:title` como «Demoscopio · {título}»: la marca ocupa el principio, lo primero
  que se corta en un resultado de búsqueda (unos 60 caracteres) y en una pestaña.
- Siete títulos son la etiqueta del menú o un nombre a secas: «Inicio», «Histórico Errores», «Ranking Encuestadoras»,
  «Método», «Análisis», «GAD3» (el perfil de una encuestadora) y «Generales 2023» (una elección pasada).
- `_brand.html` envuelve el logotipo en un `<h1>` en todas las páginas, también en la de error. Siete tipos de
  página tienen además un `<h1>` propio (la portada, `/analisis`, los cinco estudios y tres páginas del método):
  dos `<h1>`. Los otros nueve sólo tienen el del logotipo: su `<h1>` dice «Demoscopio».
- Ningún JavaScript lee `<h1>`, `<h2>` ni el título. El CSS de la cabecera cuelga de `#site-header h1`. Ya existe
  la clase `.sr-only` (texto que sólo leen los lectores de pantalla).
- Los textos de los títulos salen del bloque `title` de cada plantilla; tres páginas del método lo reutilizan como
  titular visible (`m.headline(self.title(), tag='h1')`).

## Enfoques

Para el `<h1>` de las páginas que hoy no tienen uno propio:

1. **Un `<h1>` oculto a la vista con el título de la página, puesto por `base.html`** (elegido). Las páginas con
   un `<h1>` visible lo desactivan. No cambia nada de lo que se ve, una página nueva lo recibe sin acordarse, y
   el texto es el del título. Google excluye expresamente de su política de texto oculto el que es para lectores
   de pantalla.
2. **Subir a `<h1>` lo que ya hace de título visible**: el nombre del evento de la barra, el título de la primera
   tarjeta. Descartado en este paso: esos textos nombran peor la página («GAD3», «23 jul 2023 · Generales 2023»),
   cuatro páginas no tienen ninguno que sirva (`/elecciones` empieza por «Filtros») y obliga a tocar el CSS de las
   tarjetas.
3. **Un título visible nuevo en cada página.** Es un cambio de diseño: lo decide Luis.

Para el orden del título, uno solo razonable: la página primero y la marca al final; la portada, al revés.

## Diseño

### `<title>` y `og:title`

- `<title>`: «{título} · Demoscopio». La portada es la excepción: «Demoscopio · Promedio de encuestas electorales
  en España».
- `og:title`: el título sin la marca (la vista previa ya enseña el dominio y `og:site_name`). La portada, el
  mismo texto que su `<title>`.
- Página de error: «Error 404 · Demoscopio».

| Página | Hoy (tras «Demoscopio · ») | Título nuevo (antes de « · Demoscopio») |
|---|---|---|
| `/` | Inicio | Demoscopio · Promedio de encuestas electorales en España |
| `/es` | Encuestas, promedios y escaños · Elecciones generales 2026 | Encuestas elecciones generales 2026: promedio y escaños |
| `/es-md` | Encuestas, promedios y escaños · Elecciones autonómicas · Madrid 2027 | Encuestas elecciones autonómicas Madrid 2027: promedio y escaños |
| un sondeo | Sigma Dos para El Mundo · 5 oct 2026 · Elecciones generales 2026 | Encuesta Sigma Dos para El Mundo · 5 oct 2026 · Elecciones generales 2026 |
| `/elecciones` | Histórico Errores | Errores de las encuestas en las elecciones pasadas |
| una elección pasada | Generales 2023 | Generales 2023: resultado y errores de las encuestas |
| `/casas` | Ranking Encuestadoras | Ranking de encuestadoras: cuáles aciertan más |
| una encuestadora | GAD3 | GAD3: rating e historial de la encuestadora |
| una casa sin rating | Agregador | Agregador: historial de la encuestadora |
| `/metodo` | Método | Método: cómo funciona el modelo |
| `/analisis` | Análisis | Análisis electorales con datos oficiales |

Sin cambios: las cuatro subpáginas del método, los estudios y `/privacidad`. Sin año conocido, el evento va sin él
(«Encuestas elecciones generales: promedio y escaños»). El menú conserva sus etiquetas.

### `<h1>`

- El logotipo de la cabecera deja de ser un título: `<p class="wordmark">` con el mismo enlace y la misma imagen.
- `base.html` pone, tras la cabecera y antes de la barra del evento, `<h1 class="sr-only">` con el título de la
  página (bloque `heading`). Ahí, quien salta al `<h1>` y sigue leyendo pasa por la barra del evento y luego por el
  contenido.
- Las páginas con `<h1>` visible vacían el bloque: la portada, `/analisis`, los cinco estudios y las tres páginas
  del método con titular.
- Página de error: `<h1 class="sr-only">Error {estado}</h1>`.

Nada de lo que se ve cambia: se comprueba con capturas de antes y después.

### Código

- `web/templates/base.html`: bloques `title`, `title_brand` (la marca al final; la portada lo vacía) y `heading`.
- `mtpy/lib/pages/common.py`: `event_phrase(scope, scopes)` («elecciones generales», «elecciones autonómicas
  Madrid») y la clave `phrase` de `event_info`.
- Plantillas: los títulos de la tabla; `_brand.html`; `error.html`.
- `web/dist/css/site.css`: las reglas de `#site-header h1` pasan a `.wordmark`.

### Pruebas

`tests/test_pages_titles.py`:

- Cada página del sitemap, y `/privacidad`: un solo `<h1>`, que no es el logotipo; el título acaba en
  « · Demoscopio» (la portada empieza por la marca) y no se repite entre páginas; `og:title` es el título sin la
  marca.
- El título exacto de cada tipo de página de la tabla.
- El `<h1>` oculto va entre la cabecera y la barra del evento; las páginas con titular visible no lo llevan.
- La página de error: título y `<h1>`.
- `event_phrase` para el ámbito nacional, uno autonómico y uno fuera del catálogo.

## Fuera de este paso

- Las descripciones: tres del método pasan de 300 caracteres (Google enseña unos 160), dos empiezan por la
  etiqueta del menú («Histórico Errores: …») y la de `/es` dice «de Elecciones generales».
- Un título visible en las páginas que no lo tienen (enfoque 3).
- Datos estructurados, `twitter:site`, las canónicas con `?scope=`: siguen en la ronda.

## Decisiones mías, por revisar

- **Todos los textos de la tabla.** En particular: el título de la portada (no usa el lema «Todas las encuestas,
  un solo promedio» porque no lleva la palabra «electorales»); «Encuestas elecciones … 2026» en estilo de
  titular, sin preposiciones, que es como se busca; «autonómicas» en los ámbitos autonómicos (más largo, pero
  «elecciones Madrid» también son las municipales); «Encuesta» delante en la página de un sondeo.
- **El `<h1>` oculto** en vez de uno visible, y su sitio antes de la barra del evento (fuera de `<main>`).
- **`og:title` sin la marca.**
- Las descripciones no se tocan en este paso.
