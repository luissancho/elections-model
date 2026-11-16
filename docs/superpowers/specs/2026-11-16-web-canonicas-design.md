# Web: la URL canónica de cada página

Cambio pequeño, quinto paso de la ronda de SEO y redes sociales. Luis dio paso a seguir con la ronda el 2026-11-16
(«Vale, continúa»); el diseño no lo ha revisado: lo que no viene de él está en «Decisiones mías, por revisar».
Plan: `docs/superpowers/plans/2026-11-16-web-canonicas.md`.

## Intención

Que cada página tenga una sola dirección ante un buscador y ante una red social: la suya, sin nada detrás. Hoy una
misma página se declara canónica con varias direcciones distintas, y eso reparte entre ellas lo que debería sumar
una sola (la posición en Google, las veces que se comparte en Facebook).

## Hechos de partida (2026-11-16, comprobados en la web de las pruebas)

- `Page.canonical_url` compone la canónica y `og:url` con la ruta y **toda** la consulta menos `mode` y `run` (la
  página del evento quita además `region` y `seccion`). Era lo correcto cuando el contenido se elegía por la
  consulta (`?scope=`, `?date=`, `?id=`); desde las URL limpias ya no se elige así en ninguna página.
- Así que cualquier parámetro se queda en la canónica:
  - `/es?fbclid=abc` (lo que Facebook añade a todo enlace que se pulsa) se declara canónica con el `fbclid`, y lo
    mismo `utm_source` y compañía, que son los parámetros con los que se miden las campañas en GA4.
  - `/casas?scope=es` se declara canónica con `?scope=es`, y también `/elecciones`, `/metodo` y sus cuatro
    páginas, `/analisis` y sus estudios y `/privacidad`.
- Esos enlaces con `?scope=` existen: `/casas?scope=es` está enlazada desde la página del evento, desde la de cada
  sondeo y desde dos del método; `/elecciones?scope=es`, desde el evento, el ranking, el método y cada elección
  pasada («← Todas las elecciones»). El sitemap lista `/casas` y `/elecciones`: dos direcciones para lo mismo, y
  cada una dice ser la buena.
- Las páginas con dirección propia (el perfil de una encuestadora, una elección pasada, un sondeo) ya fijan su
  canónica limpia en el controlador. Ningún JavaScript toca la canónica.
- Una dirección con la barra final (`/casas/`) o con otra capitalización es un 404: no hay variantes de la ruta.
- `base.html` envía `twitter:card`, pero no dice de quién es el contenido.

## Enfoques

1. **La canónica es la ruta, sin consulta, en todas las páginas** (elegido). Una regla, sin lista que mantener, y
   un parámetro nuevo o ajeno nunca llega a la cabecera.
2. Añadir `scope` a los parámetros que se quitan en las páginas globales. Descartado: deja dentro `utm_*`,
   `fbclid`, `gclid` y cualquier otro.
3. Una lista de parámetros permitidos por página. Descartado: hoy estaría vacía en todas.

## Diseño

- `Page.canonical_url()` devuelve `WEB_SITE_URL` + la ruta, sin parámetros; deja de aceptar la lista de parámetros
  que quitar, y la página del evento usa la canónica por defecto.
- `og:url` sigue siendo la misma dirección.
- `/casas?scope=es-md` pasa a declarar `/casas`. Con un ámbito autonómico esa página cambia dos tarjetas (los
  efectos de casa y el rebaño son los del ámbito); el ranking, el título y la descripción son los mismos. Queda
  como una vista de `/casas`, igual que `mode` o `region` lo son del evento.
- `base.html` añade `<meta name="twitter:creator" content="@luis_sancho">` (`author_handle`, la misma cuenta que
  ya firma el pie de todas las páginas y las imágenes descargables).

### Pruebas

- Cada página del sitemap, y `/privacidad`, pedida con `?scope=es&utm_source=x&utm_medium=social&fbclid=abc&gclid=1`:
  responde 200 y su canónica y su `og:url` son su dirección limpia.
- La página del evento con `mode`, `run`, `region` y `seccion`; `/metodo?scope=es` y `/casas?scope=es` (las dos
  pruebas que fijaban la consulta en la canónica cambian de signo).
- `twitter:creator` en la cabecera.

## Fuera de este paso

- Los enlaces internos a `/casas` y `/elecciones` conservan `?scope=`: sirven para que el filtro de ámbito llegue
  puesto. Con la canónica limpia ya no cuentan como otra página.
- `twitter:site` (la cuenta del sitio en X): no hay una cuenta propia de Demoscopio en el código.
- Datos estructurados y resúmenes con cifras: siguen en la ronda.

## Decisiones mías, por revisar

- **`/casas?scope=X` y `/elecciones?scope=X` dejan de ser páginas propias para un buscador** (su canónica es
  `/casas` y `/elecciones`), también con un ámbito autonómico.
- **`twitter:creator` con la cuenta personal del autor**, y `twitter:site` sin poner hasta que el sitio tenga la
  suya.
