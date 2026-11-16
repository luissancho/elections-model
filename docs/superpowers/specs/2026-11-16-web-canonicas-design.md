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
  consulta (`?scope=`, `?date=`, `?id=`); desde las URL limpias, el evento, la elección, la encuestadora y el
  sondeo van en la ruta.
- De las 17 páginas globales, dos cambian parte de su contenido con `?scope=`: `/casas` (las tarjetas de
  efectos de casa y de rebaño, y cuatro descargas; el ranking es el mismo) y `/metodo/backtest` (siete tarjetas
  con el backtest del ámbito, bajo el filtro; las cuatro de arriba son las mismas). En las dos, el título, la
  descripción y el `<h1>` no cambian con el ámbito. En las demás sólo cambian los enlaces (en `/elecciones`,
  además, el filtro que llega puesto).
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
- `/casas?scope=es-md` pasa a declarar `/casas`, y `/metodo/backtest?scope=es-md`, `/metodo/backtest`. Con un
  ámbito autonómico esas dos páginas cambian tarjetas, pero no su título ni su descripción: quedan como vistas
  de la página nacional, igual que `mode` o `region` lo son del evento. La alternativa (que cada ámbito sea una
  página propia) pediría además un título y una descripción por ámbito, o Google vería 18 páginas con el mismo
  título.
- `base.html` añade `<meta name="twitter:creator" content="@luis_sancho">` (`author_handle`, la misma cuenta que
  ya firma el pie de todas las páginas y las imágenes descargables).

### Pruebas

- Cada página del sitemap, y `/privacidad`, pedida con `?scope=es&utm_source=x&utm_medium=social&fbclid=abc&gclid=1`:
  responde 200 y su canónica y su `og:url` son su dirección limpia.
- La página del evento con `mode`, `run`, `region` y `seccion`; `/metodo?scope=es` y `/casas?scope=es` (las dos
  pruebas que fijaban la consulta en la canónica cambian de signo).
- `twitter:creator` en la cabecera.
- Con un ámbito autonómico publicado, las páginas globales pedidas con su `?scope=` declaran su ruta; el perfil
  de una encuestadora y un sondeo pedidos por el identificador de la casa declaran la dirección con su nombre.

## Corregido tras la revisión

- Esta spec decía que ninguna página elige su contenido por la consulta. No es así en `/metodo/backtest`, cuyas
  tablas son las del ámbito del filtro: queda dicho arriba y entre las decisiones por revisar.
- Tres comportamientos sin prueba: el ámbito autonómico en una página global, y el perfil y el sondeo pedidos
  por un identificador (el cambio había dejado sin efecto la prueba que había del perfil).

## Fuera de este paso

- Los enlaces internos a `/casas` y `/elecciones` conservan `?scope=`: sirven para que el filtro de ámbito llegue
  puesto. Con la canónica limpia ya no cuentan como otra página.
- `twitter:site` (la cuenta del sitio en X): no hay una cuenta propia de Demoscopio en el código.
- Datos estructurados y resúmenes con cifras: siguen en la ronda.

## Decisiones mías, por revisar

- **`/casas?scope=X` y `/metodo/backtest?scope=X` dejan de ser páginas propias para un buscador** (su canónica
  es la página nacional), también con un ámbito autonómico: las tarjetas de efectos de casa y de rebaño de un
  ámbito, y las tablas de su backtest, se siguen viendo pero no tienen una dirección que Google pueda indexar
  aparte. Volver atrás es pasar la canónica con `?scope=` en esos dos controladores.
- `/elecciones?scope=X`, igual; ahí el contenido es el mismo y sólo cambia el filtro que llega puesto.
- **`twitter:creator` con la cuenta personal del autor**, y `twitter:site` sin poner hasta que el sitio tenga la
  suya.
