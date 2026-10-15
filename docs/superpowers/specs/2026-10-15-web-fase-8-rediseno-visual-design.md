# Web, fase 8: rediseño visual al estilo de Objetivo176 y Election Simulator

Fecha: 2026-10-15. Estado: aprobada por Luis el 15 de octubre ("Por lo demás, adelante") con dos apuntes ya
incorporados: las autonómicas en la columna principal de la portada (D4.2) y aire entre los KPIs y las tarjetas
de partido en móvil (D3.6).
Maqueta aprobada por Luis el 15 de octubre ("Maravillosos, gran trabajo"), con un único apunte, el contenedor
fluido a todo el ancho: `docs/superpowers/specs/2026-10-15-web-fase-8-mockups.html` (copia de
`.superpowers/brainstorm/67833-1792054449/content/mockups.html`; se abre en el navegador sin servidor, con las
tipografías de Google Fonts solo para la maqueta). Sigue a la fase 7f
(`docs/superpowers/specs/2026-10-15-web-fase-7f-dianas-evento-ranking-design.md`). Esta spec solo describe lo
que cambia respecto de lo que hay en `dev`.

## Intención

Luis quiere "otra pasada al diseño": que la web copie los estilos, tamaños de fuente y tablas de Objetivo176
(objetivo176.es/generales) y de Election Simulator (electionenjoyer.com/spain), en escritorio y en móvil; que la
portada se inspire en la de Objetivo176 y sea "mucho más atractiva"; y que el conjunto sea "atractivo a la vista,
colorido y fácil de navegar", espectacular pero sencillo de entender tanto para expertos como para profanos.

Lo que comparten las dos referencias, y la maqueta copia: una sans geométrica (DM Sans) a tamaños pequeños, una
display (Outfit) para títulos y cifras grandes, fondo gris muy claro, tarjetas blancas de radio 18 px con sombra
suave en capas, tablas densas con punto de color, nombre en negrita, cifras tabulares y barras dentro de las
celdas. Election Simulator aporta el color: tarjetas de partido en su color con la cifra enorme, y una fila de
indicadores con el titular del pronóstico. Objetivo176 aporta la navegación de app en móvil (barra inferior de
iconos) y la composición de la portada: titular grande con una palabra en color, tarjeta destacada de las
Generales, lista compacta de autonómicas y, al lado, "si hoy hubiera generales" y las últimas encuestas.

Lo que falla hoy, según el reconocimiento del 15 de octubre: tipografía de informe (serif a 17 px), ninguna
cifra destacada, nada de color fuera de los gráficos, una portada que es un muro de veinte tarjetas de texto
iguales y, en móvil, el titular de la portada que se sale por la derecha.

## Decisiones de Luis (2026-10-15)

1. Tipografía y marca: DM Sans + Outfit, y rehacer el wordmark con la nueva tipografía.
2. Navegación móvil: barra inferior de iconos, como Objetivo176.
3. Modo oscuro: no, queda para una fase posterior.
4. Contenedor fluido a todo el ancho en escritorio, no una caja fija (apunte sobre la maqueta).

## Objetivos

1. Sistema visual nuevo en todo el sitio: tipografías vendidas DM Sans y Outfit, escala de tamaños de las
   referencias, tokens nuevos, tarjetas de radio 18 px, controles en tinta oscura, tablas con barras y rangos,
   fila de indicadores (KPI) y tarjetas de partido en su color.
2. Portada nueva: hero con titular, tarjeta destacada de las Generales (cifra grande, barra de escaños con la
   marca de la mayoría, chips, botones), hemiciclo y últimas encuestas al lado, autonómicas en filas compactas,
   "Para entender el modelo", "¿Cómo funciona Demoscopio?" y las elecciones pasadas como fichas.
3. Navegación: cabecera compacta y fija arriba; en móvil, barra inferior de cinco entradas y un icono en la
   cabecera para la sexta.
4. Promedios: fila de KPIs y tarjetas de partido encima del promedio; hemiciclo con los números dentro de los
   arcos y el total en el centro; tabla de partidos con barra de voto y barra de rango; el resto de tarjetas
   heredan el estilo.
5. El resto de páginas (Escaños, Histórico Errores, Ranking Encuestadoras, perfil de encuestadora, detalle de
   una elección, Método) heredan el sistema sin cambiar de estructura, con los retoques que se listan en D6.
6. Los gráficos ECharts adoptan las tipografías y los colores del sistema.
7. El wordmark se rehace con Outfit; el favicon (la lente) no cambia.

## No objetivos

- Modo oscuro (decisión 3). Los tokens se definen de forma que añadirlo después sea cambiar valores, no
  selectores, pero no se implementa.
- Ningún cambio en el modelo, en los jobs de publicación ni en el paquete publicado: las páginas leen lo que ya
  está (`headline`, `meta`, `series`, `polls`, `summary`, los eventos publicados).
- Ninguna ruta nueva ni cambio en la API. Nada de nginx ni Docker: las fuentes se sirven desde `/dist/vendor/`
  como hasta ahora, y la CSP (`default-src 'self'`) no cambia.
- Ni fotos de líderes ni logotipos de partido (no hay imágenes ni derechos): el color del partido hace su papel.
- Los colores de los partidos y de los bloques siguen saliendo del catálogo del run; no se tocan.
- Los textos del método, el glosario y los "Cómo leerlo" existentes no se reescriben (solo los de las tarjetas
  nuevas o que cambian).

## Anti-objetivos (lo que haría fracasar la fase aunque funcionara)

- Perder la lectura sin JavaScript: la portada y las páginas siguen saliendo del servidor con sus cifras y sus
  tablas; lo único que dibuja el navegador son los gráficos (como hoy) y las tarjetas de partido llevan sus
  cifras en el HTML (solo el sparkline es JS). Nada de `script` inline ni de recursos externos (CSP).
- Páginas mucho más pesadas: las fuentes son cuatro ficheros woff2 subconjuntados al latín (unos 120 KB en
  total, frente a los tres actuales); la isla `#initial-data` de la portada lleva solo lo que dibuja el hemiciclo
  (los escaños por partido del ámbito nacional).
- Un móvil inservible: barra inferior con zonas de toque de 44 px, tablas con desplazamiento horizontal y la
  primera columna fija, tarjetas de partido en carrusel horizontal, nada que se salga por la derecha (el titular
  de hoy sí lo hace: queda arreglado).
- Perder contraste AA: los textos apagados siguen pasando 4,5:1 sobre fondo y tarjeta; el texto blanco de las
  tarjetas de partido va sobre el color del partido oscurecido (D3.7), no sobre el color puro.
- Confundir la selección con un partido: los controles seleccionados van en tinta oscura, nunca en un azul que se
  parezca al del PP.
- Romper los enlaces guardados: los `hash` de los gráficos (`#ventana=&vista=&sel=`, filtros del histórico)
  siguen significando lo mismo.
- Romper las pruebas: `python -m pytest -m "not integration" -q` en verde (758 al empezar), sin avisos nuevos.

## Restricciones

- Patrón de siempre: controlador `Page` + Jinja2 (`StrictUndefined`, autoescape) + isla JSON `#initial-data` +
  módulos JS que solo dibujan (ECharts 5.6 vendido). Nada de build step ni de dependencias nuevas en tiempo de
  ejecución. Módulos puros con tests de paridad en `node` (`tests/test_js_*.py`).
- Textos en español, "encuestadora/s" y "medio/s", siglas de partido siempre, el mismo tono que hoy.
- Responsive con media queries (la maqueta usa container queries por ser un visor; el sitio usa el ancho de la
  ventana): móvil hasta 719 px, tableta de 720 a 1099 px, escritorio desde 1100 px. Contenedor fluido: `main`
  sin `max-width`, relleno lateral de 32 px (16 px en móvil), como ya está.
- Las fuentes son OFL (DM Sans y Outfit, de Google Fonts): se vendorizan como woff2 estáticos en
  `web/dist/vendor/fonts/` con su licencia en `LICENSE`, y se retiran Source Serif 4 y Source Sans 3 (el
  wordmark deja de depender de ellas).
- `tests/test_pages.py` fija los textos y el marcado que cambian; `tests/test_deploy_files.py` o uno nuevo
  comprueba que las fuentes referenciadas en `site.css` existen y están en `LICENSE`.

## Diseño

### D1. Tipografía y tokens

Tipografías vendidas (woff2, subconjunto latín con ñ, acentos y comillas españolas; `font-display: swap`):

| Familia | Pesos | Uso |
| --- | --- | --- |
| DM Sans | 400, 500, 600, 700 | todo el texto, controles, tablas, ejes de los gráficos |
| Outfit | 700, 800 | títulos, cifras grandes, wordmark |

Escala (desktop; en móvil solo cambian las filas marcadas):

| Elemento | Tamaño | Peso / familia |
| --- | --- | --- |
| Cuerpo (`body`) | 15 px / 1,5 | DM Sans 400 |
| Entradillas (`.lead`), notas | 13,5 px | DM Sans 400, `--color-muted` |
| Controles, chips, píldoras | 12,5 px / 11 px | DM Sans 600 |
| Tablas | 13 px (12,5 px en móvil) | DM Sans 400; nombres 700 |
| Cabeceras de tabla | 10,5 px, mayúsculas, espaciado .04em | DM Sans 600, `--color-muted` |
| Título de tarjeta | 16 px | Outfit 700 |
| Título de sección (portada) | 22 px (20 px en móvil) | Outfit 800 |
| Título de evento (barra de evento) | 26 px (22 px en móvil) | Outfit 800 |
| Titular de la portada | 46 px (32 px en móvil), interlineado 1,05, espaciado -.025em | Outfit 800 |
| Cifra de KPI | 22 px | Outfit 700 |
| Cifra de tarjeta de partido | 38 px | Outfit 800 |
| Cifra del hero | 60 px (48 px en móvil) | Outfit 800 |

Tokens de `web/dist/css/site.css` (los que cambian o se añaden; el resto sigue):

| Token | Valor | Uso |
| --- | --- | --- |
| `--color-bg` | `#f4f5f8` | sin cambio |
| `--color-soft` / `--color-soft2` | `#eceef2` / `#e3e6ec` | fondos de controles; pistas de las barras |
| `--color-text` | `#1a1c22` | tinta; también los controles seleccionados |
| `--color-muted` | `#5a5f72` | sin cambio |
| `--color-faint` | `#9298ac` | solo elementos decorativos (chevrones, puntos de ejes); ya no para texto, que pasa a `--color-muted` |
| `--color-accent` | `#1d4f86` | marca (lente), enlaces, barra de título de las tarjetas, entrada activa del menú, iconos de "Para entender el modelo" |
| `--color-accent-soft` | `#e7eef8` | fondo de las píldoras de marca y de los iconos |
| `--color-hover` | `#eff4ff` | fila de tabla al pasar el ratón |
| `--color-ok` / `--color-ok-soft` | `#16a34a` / `#ecfdf3` | chips y cifras positivas (mayoría, delta positivo) |
| `--color-bad` / `--color-bad-soft` | `#dc2626` / `#fef2f2` | cifras negativas, punto del contador de días |
| `--font-head` | `'Outfit', 'DM Sans', sans-serif` | |
| `--font-body` | `'DM Sans', system-ui, sans-serif` | |
| `--radius` / `--radius-sm` | `18px` / `10px` | tarjetas / chips, botones, celdas |
| `--shadow` | `0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05)` | tarjetas |
| `--shadow-up` | `0 2px 4px rgba(16,24,40,.05), 0 14px 32px rgba(16,24,40,.09)` | tarjeta enlazada al pasar el ratón |

`font-variant-numeric: tabular-nums` en `body`. Los textos siguen limitados a `max-width: 72ch` donde ya lo
están (prosa, "Cómo leerlo", glosario) y a 90ch en las entradillas de tarjeta.

### D2. Cabecera, navegación y barra inferior

- Cabecera (`#site-header`): fija arriba (`position: sticky; top: 0`), 56 px de alto (52 px en móvil), blanca,
  línea inferior `--color-line`. A la izquierda el wordmark (`logo.svg`, 24 px de alto, rehecho en D8) sin la
  frase de debajo (la `brand-tagline` desaparece de la cabecera; `SITE_TAGLINE` sigue en la `description`).
  A la derecha el menú de siempre (Inicio, Histórico Errores, Ranking Encuestadoras, Método) a 13,5 px / 500,
  `--color-muted`; la entrada activa en tinta, 700 y una línea de 3 px en `--color-accent` por debajo (como
  hoy). La pancarta de la veda (`#freeze-banner`) queda encima de la cabecera y también fija.
- En móvil (≤ 719 px) la cabecera muestra solo el wordmark y, a la derecha, un botón redondo de 36 px con el
  icono de interrogación que enlaza a `/metodo` (`aria-label="Método"`). El menú de la cabecera se oculta.
- Barra inferior (`nav.tabbar`, nueva en `base.html`, solo visible en móvil): fija abajo (`position: sticky;
  bottom: 0` al final del `body`, con `padding-bottom: env(safe-area-inset-bottom)`), blanca, línea superior,
  cinco entradas de icono (22 px) y etiqueta (10,5 px / 600), zona de toque de 44 px, la activa en
  `--color-accent`:
  1. Inicio → `/`.
  2. Promedios → `/{scope}` del evento actual; fuera de un evento, `/es` (el ámbito por defecto de
     `resolve_scope`).
  3. Escaños → `/{scope}/escanos`, misma regla.
  4. Histórico → `/elecciones` (con `?scope=` del evento actual cuando lo hay, como ya enlazan las tarjetas
     de cierre).
  5. Ranking → `/casas`.
  Así las seis páginas se alcanzan en móvil: cinco en la barra y Método en la cabecera. La barra reemplaza al
  menú desplazable horizontal de hoy. `main` lleva 92 px de relleno inferior en móvil para que la barra no tape
  el pie. Las entradas salen de una constante nueva `TABBAR` de `pages/common.py` al lado de `PAGES`, y el
  controlador marca la activa (`active`) con la misma regla que el menú (la de los eventos, por el sufijo).
- Los iconos son SVG inline en una plantilla parcial nueva `_svg.html` (un `<svg>` con `<symbol>` por icono,
  incluido una vez al principio del `body`, y `<use href="#i-...">` donde se usan): inicio (casa), promedios
  (línea), escaños (semicírculo), histórico (diana), ranking (trofeo), método (libro), interrogación, chevron.
- Barra de evento (`#event-bar`): título del evento a 26 px Outfit 800 y debajo, a 13 px, la fecha, los días
  que faltan y la cámara ("Domingo 29 nov 2026 · faltan 45 días · Congreso, 350 escaños"; la cámara y sus
  escaños salen de `meta.n_seats` y del nombre del ámbito: "Congreso" para `es`, "Parlamento" para los
  autonómicos). Las pestañas Promedios | Escaños como segmentado grande (D3.3) con su icono. A la derecha, la
  píldora de marca "Estimación a {as_of}" y, debajo en gris, "Publicación del {run_at} · {n_polls} sondeos"
  (la píldora "Run del…" de hoy se funde en esto; con un run fijado, la píldora dice "Run del {run_at}" como
  hoy). En móvil se apila: título, pestañas, píldora.

### D3. Componentes

1. **Tarjeta** (`.card`): blanca, borde `--color-line`, radio 18, `--shadow`, relleno 16×18 px (14×16 en
   móvil). Cabecera (`.card-head`) en una fila con el título a la izquierda (barra vertical de 4×18 px en
   `--color-accent` delante, como hoy) y, a la derecha, lo que la tarjeta lleve (píldora de fecha, segmentado,
   enlace "Ver todas ›"). El subtítulo de modo ("· estimación a 13 oct 2026") pasa a ser una píldora
   (`.pill`) al lado del título. "Cómo leerlo" sigue siendo `<details>`, con el "+" dentro de un círculo de 18 px
   en `--color-accent-soft`.
2. **Píldora** (`.pill`): 11 px / 600, relleno 3×9, radio 999, fondo `--color-soft`; variantes `.ok`
   (verde), `.brand` (acento claro) y con punto de color (`.dot` de 8 px). **Chip** (`.chip`): 13 px / 600,
   relleno 6×11, radio 10, para datos con etiqueta secundaria (`<small>`); variante `.ok`.
3. **Segmentado** (`.seg`): como hoy, pero la opción activa en tinta (`--color-text`) con texto blanco, no en
   acento; 12,5 px / 600; variante `.lg` (13 px, relleno 7×14) para las pestañas del evento. Mismo trato para
   `.event-tabs`. Los `.pills` de selección múltiple (leyenda del promedio, calculadora de coaliciones, partidos
   de la diana) pasan a `.lg`: borde de 1,5 px del color del partido, texto en tinta; activo, relleno del color
   del partido y texto `ink_color`.
4. **Botones** (`.btn`): 13 px / 600, relleno 9×16, radio 10; `.pri` en tinta con texto blanco (el "Ver el
   promedio →" del hero), el resto blanco con borde `--color-line-strong`. Los `button`/`select` de los
   formularios pasan a este estilo (radio 10, no 999).
5. **Fila de KPIs** (`.kpis` > `.kpi`): una tarjeta partida en N columnas iguales con un separador de 1 px
   entre ellas (del 20 % al 80 % de la altura); cada KPI con etiqueta (12 px, muted), valor (22 px Outfit 700,
   con un `<small>` de 12 px al lado) y una línea de apoyo (11,5 px / 600, muted; `.ok` en verde). Tableta: tres
   columnas; móvil: dos, y el último KPI a todo el ancho.
6. **Tarjeta de partido** (`.pcards` > `.pc`): rejilla de seis columnas (tres en tableta; en móvil, carrusel
   horizontal con `scroll-snap`, tarjetas de 210 px, sin barra de desplazamiento visible, sangrado hasta los
   bordes de la pantalla y con los mismos 16 px de separación respecto de la fila de KPIs que el resto de
   bloques: el carrusel no puede perder el margen superior del `.stack`; apunte de Luis sobre la maqueta).
   Cada tarjeta:
   `--c` = color del partido; fondo `linear-gradient(135deg, color-mix(in srgb, var(--c) 88%, #000),
   color-mix(in srgb, var(--c) 60%, #000))`; texto blanco; puesto ("1.º") en una píldora translúcida arriba a la
   derecha; siglas (Outfit 700, 15 px) y nombre completo (11 px, 78 % de opacidad, con elipsis); escaños
   (Outfit 800, 38 px) y % de voto (Outfit 600, 17 px) con sus rótulos en mayúsculas de 10,5 px; deltas frente a
   la última elección del ámbito ("+3 esc. vs 2023 · −0,4 pt"; "nuevo" si el partido no concurrió); y abajo,
   a todo el ancho, el sparkline del último año (SVG de 28 px de alto, trazo blanco de 2 px, punto final;
   `preserveAspectRatio="none"` y `vector-effect="non-scaling-stroke"`) con el rótulo "Último año". El
   sparkline lo dibuja JS a partir de la isla; sin JS la tarjeta queda igual sin la línea.
7. **Filas** (`.rows` > `.row`): lista de enlaces en una tarjeta sin relleno, cada fila con nombre (14 px / 700)
   y fecha (12 px, muted) a la izquierda, una píldora con punto de color en el centro ("PSOE 33,5 % · 17 esc."),
   un texto a la derecha ("Derecha: **82 %** de mayoría") y un chevron; fondo `--color-hover` al pasar el
   ratón. Variante `.rows-2`: dos columnas con separador vertical (una columna en tableta y móvil). En móvil, la
   fila se apila en tres líneas con el chevron a la derecha.
8. **Tablas** (`table`): cabeceras a 10,5 px en mayúsculas `--color-muted` con línea de 1,5 px; celdas de 7×8
   px con línea `--color-line`; números a la derecha, texto a la izquierda; fila `--color-hover` al pasar el
   ratón; el nombre del partido con su punto (9 px) y en negrita (`.pn`). Celdas nuevas, calculadas en el
   servidor con filtros de `pages.filters` (como `bar`, `sides` y `shade` de la fase 7f):
   - `.bar`: barra horizontal de 8 px sobre pista `--color-soft2` (90 px de ancho; 84 en móvil), relleno con el
     color del partido y anchura proporcional al máximo de la columna (filtro `bar(valor, máximo)`, ya existe).
   - `.rng`: pista de 12 px (110 px de ancho) con el intervalo del 95 % como una banda del color del partido al
     45 % de opacidad entre `lo` y `hi` (escala de 0 al total de escaños), una marca de 3×16 px del color en la
     mediana y una línea discontinua en la mayoría absoluta; a su lado el texto "118–160" en gris (filtro nuevo
     `rng(lo, hi, mediana, total, mayoría)` que devuelve los tres porcentajes; la plantilla pinta el marcado).
   - En las cabeceras de las tablas de sondeos, el punto del partido encima de sus siglas (`.pd`, ya existe).
   En móvil, `.table-wrap`/`.et-wrap` siguen con desplazamiento horizontal y la primera columna fija.
9. **Hemiciclo** (`charts/hemicycle.js`): los partidos con al menos el 5 % de los escaños llevan la etiqueta
   dentro del arco (`label.position: 'inside'`, siglas en Outfit 700 a 15 px y escaños en Outfit 800 a 24 px,
   blanco); los demás no llevan etiqueta (el tooltip la da); desaparecen las líneas de etiqueta. En el centro, el
   total ("350", Outfit 800, 34 px) y debajo "ESCAÑOS · MAYORÍA ABSOLUTA 176" (10,5 px, mayúsculas, muted), en
   vez del texto de dos líneas de hoy. La mayoría se marca con una línea discontinua desde el hueco hasta fuera
   del arco en el ángulo de la mayoría y una píldora negra con el número encima del arco (elementos `graphic`
   calculados con el mismo centro y radio que la serie). Radio `['52%', '100%']` y `center: ['50%', '92%']`
   para que el semicírculo llene la tarjeta; el contenedor limita su ancho a 620 px centrado (520 px el pequeño de
   la portada). Debajo del hemiciclo, en Promedios y en el detalle de una elección, la tira "Líder: **PP** ·
   **140 escaños** · le faltan **36** para la mayoría · P(primero) **97 %**" (`.lead-strip`, fondo
   `--color-soft`, radio 12; sale del servidor con las cifras del resumen; en el detalle de una elección, sin
   P(primero) y con "superó la mayoría en N" cuando proceda).
10. **Leyenda de píldoras** de los gráficos de series (`.legend` > `.lg`): una leyenda HTML nueva encima del
    gráfico, una píldora por partido o bloque (borde de 1,5 px del color, activo relleno del color con tinta
    `ink_color`), a 12 px / 600, relleno 3×10, con el punto de 9 px. Solo la llevan los dos gráficos de series
    de Promedios ("Promedio de sondeos" y "Una encuestadora frente al promedio"), que esconden la leyenda
    interna de ECharts (`legend.show: false`) y la gobiernan con `dispatchAction({type: 'legendToggleSelect'})`
    (y el evento `legendselectchanged` sigue escribiendo `sel=` en el `hash`, con el mismo significado que hoy;
    al cargar, las píldoras reflejan la selección del `hash`). Los demás gráficos (evolución, abanico, tendencia
    del histórico, barras) conservan la leyenda de ECharts con el tema de D7. Decisión del autor tras la revisión
    del constructor: la maqueta aprobada lleva las píldoras, y hoy no existe una leyenda HTML (la fase 7c la
    puso dentro del gráfico).
11. **Tarjeta enlazada** (`.card-link`, portada): icono de 40 px en un cuadrado de radio 12 en
    `--color-accent-soft`, título Outfit 700 de 17 px, texto a 13,5 px muted y una línea final "Ver el histórico
    →" en acento; sube 2 px con `--shadow-up` al pasar el ratón (sin movimiento con `prefers-reduced-motion`).
12. **Pie** (`#site-footer`): 12,5 px muted, tres líneas como hoy; enlaces en acento.

### D4. Portada

`index.html` se reescribe (y `pages/inicio.py` amplía su contexto). Estructura, de arriba abajo:

1. **Hero** (`.hero`): dos columnas (1,3 fr / 0,85 fr) alineadas arriba; la columna principal lleva el
   titular, la tarjeta destacada y las autonómicas (punto 2), y la lateral las dos tarjetas de apoyo, más corta
   que la principal (el hueco bajo ella se rellenará en otra fase). Una columna en tableta y móvil, con la
   columna lateral después de la tarjeta destacada y antes de las autonómicas; en tableta las dos tarjetas
   laterales van en una fila de dos columnas.
   - Columna principal: la línea de contexto (`.eyebrow`: punto rojo y "GENERALES · 29 NOV 2026 · FALTAN 45
     DÍAS" en mayúsculas de 11,5 px; en veda, "GENERALES · 29 NOV 2026 · PRONÓSTICO DEFINITIVO"), el titular
     "Todas las encuestas, un solo **promedio**." (con `promedio` en `--color-accent`), el párrafo "Demoscopio
     promedia los sondeos de cada elección, simula el reparto de escaños y puntúa a las encuestadoras según lo
     que acertaron. Sin cookies y con el método a la vista." (`INTRO` de `inicio.py` cambia a este texto; el
     titular es una constante nueva `HEADLINE`, en dos partes para el énfasis) y la **tarjeta destacada**
     (`.hcard`) del ámbito nacional:
     - Cabecera: bandera (SVG inline de tres franjas, 42×28 px, radio 6; para los ámbitos autonómicos no hay
       bandera), "Elecciones generales" (Outfit 800, 22 px), debajo "Congreso de los Diputados · 350 escaños ·
       29 nov 2026" (12,5 px), y a la derecha la píldora "Si hoy hubiera elecciones" (o "Pronóstico" /
       "Pronóstico definitivo" según `mode` y veda, como las tarjetas de hoy).
     - Cifra grande: los escaños del primer partido en su color (60 px) y la frase "de 350 escaños para el
       **PP**, primer partido con el **32,7 %** del voto".
     - Barra de escaños (`.seatbar`): HTML (un `<i>` por partido con escaños, anchura en % del total, en el
       orden del hemiciclo: bloques Izquierda, Separatista, Regionalista, Derecha y dentro de cada bloque por
       escaños), 16 px de alto, radio 6; marca vertical de la mayoría con el rótulo "176 · mayoría absoluta"
       debajo; en los extremos "Izquierda 117" y "Derecha 203" (los escaños de los dos bloques de `bmaps.vs`).
     - Chips: en verde, "**Derecha 203 escaños** · > 99 % de mayoría absoluta" (el bloque con mayor
       `p_majority`; si ninguno pasa del 50 %, el chip va en gris: "Sin mayoría clara · Derecha 48 %"); y hasta
       dos chips grises con los deltas de escaños de los dos primeros partidos frente a la última elección
       ("**PP +3** escaños vs 2023") cuando hay resultado publicado.
     - Botones "Ver el promedio →" (`.btn.pri`, a `/es`) y "Escaños" (`.btn`, a `/es/escanos`), y a la derecha
       en gris "Última publicación: 8 oct 2026 · 315 sondeos de 23 encuestadoras".
   - Columna lateral (`.side`): la tarjeta "Si hoy hubiera generales" (píldora con la fecha de la estimación;
     hemiciclo pequeño, D3.9, dibujado por `pages/inicio.js` nuevo a partir de la isla; leyenda de los cuatro
     primeros partidos con escaños y % y "Otros N"; la tira "Líder: …") y la tarjeta "Últimas encuestas" (enlace
     "Ver todas ›" a `/es#polls`; cinco filas: fecha en un recuadro gris (día en Outfit 800 de 15 px, mes en
     mayúsculas), encuestadora en negrita y "para {medio} · {n} entrevistas" debajo, y a la derecha el % de los
     tres primeros partidos del promedio en su color, Outfit 700 de 12,5 px).
2. **Autonómicas**: dentro de la columna principal del hero, debajo de la tarjeta destacada (decisión de Luis,
   2026-10-15: "aunque quede un poco más largo, no me importa el espacio vacío que quedaría entonces a la
   derecha, ya lo rellenaremos"). Título de sección (`.sec-h` "Autonómicas" + "Once comunidades con
   pronóstico publicado, por fecha de elección.") y una tarjeta `.card.rows` de una sola columna: una fila por
   ámbito publicado, por fecha de elección (el orden de `event_cards`), con nombre de la comunidad, "23 may
   2027 · faltan 220 días", la píldora del primer partido ("PSOE 33,5 % · 17 esc.") y "Derecha: **82 %** de
   mayoría"; al final, una línea gris "Sin pronóstico todavía: Andalucía, Aragón, …" con los ámbitos no
   publicados (las tarjetas grises de hoy desaparecen). Sin ámbitos autonómicos publicados, la sección no sale.
   La variante `.rows-2` de D3.7 queda disponible pero la portada no la usa. En tableta y móvil (una columna),
   la columna lateral va entre la tarjeta destacada y las autonómicas.
3. **Para entender el modelo**: título de sección y las tres tarjetas enlazadas de hoy con icono (D3.11) y textos:
   Histórico Errores, "Cuánto se equivocaron los sondeos en cada elección pasada, encuestadora a encuestadora,
   con su diana de errores."; Ranking Encuestadoras, "Qué encuestadoras aciertan más, cuáles se desvían y cuánto
   pesan sus sondeos en el promedio."; Método, "Cómo se promedian los sondeos, cómo se simulan los escaños y
   cómo se comprueba todo contra las elecciones pasadas.".
4. **¿Cómo funciona Demoscopio?**: cuatro tarjetas numeradas (círculo negro con el número en Outfit 800): "Recoge
   los sondeos", "Los promedia con pesos", "Simula los escaños", "Se comprueba a sí mismo", con los textos de la
   maqueta (son estáticos, en la plantilla).
5. **Elecciones pasadas**: título con "Ver el histórico ›" (a `/elecciones`) y las fichas (`.past`): las doce
   más recientes como "Cataluña **12 may 2024**" (nombre corto del ámbito y la fecha; enlazan al detalle) y una
   última ficha "+ N más" al histórico. La lista `past_events` ya existe; la plantilla la corta.
6. Pie.

Qué ámbito es el hero: el primero de `event_cards` que esté publicado (`es` cuando lo está, que es el caso
normal). Si `es` no estuviera publicado, el hero es el primer ámbito publicado con los textos derivados del
ámbito: título de `event_title`, línea de contexto con el nombre corto del ámbito ("MADRID · 23 MAY 2027 ·
FALTAN 220 DÍAS"), tarjeta lateral "Si hoy hubiera elecciones en Madrid", sin bandera (solo `es` la lleva) y
"Parlamento" en vez de "Congreso de los Diputados"; ese ámbito no se repite en las filas de "Autonómicas". Sin
ningún ámbito publicado, la portada no tiene hero ni filas (solo las secciones estáticas), igual que hoy
muestra las tarjetas grises. El "Promedios"/"Escaños" de la barra inferior fuera de un evento apunta al ámbito
del hero (`resolve_scope(None)`, el primer publicado). Decisión del autor tras la revisión del constructor.

La isla de la portada (`initial`) lleva `hero` con `scope`, `n_seats`, `majority`, `seats` (`{siglas: escaños}`
del hemiciclo), `colors`, `order` (las siglas en el orden del hemiciclo) y `series` (las medias semanales del
último año de los partidos de `bmaps.main`, para los sparklines si la portada los llevara; no los lleva, así que
`series` se omite: decisión del constructor si lo reutiliza). `inicio_context` añade `hero` (las cifras de la
tarjeta destacada), `latest_polls` (las cinco últimas del `polls` del run nacional, por fecha descendente, con
`date`, `pollster`, `sponsor`, `sample_size` y los % de los tres primeros partidos), `autonomics` (las tarjetas
publicadas no nacionales, `event_cards` ya lo da) y `unpublished` (los nombres sin pronóstico). Los deltas frente
a la última elección salen de los resultados del evento publicado más reciente del ámbito
(`pages.elecciones.published_events` y su parte `results`); sin él, no hay chips de delta.

### D5. Promedios

Orden de la página: barra de evento (D2), **fila de KPIs**, **tarjetas de partido**, "Promedio de sondeos",
`.hemi-grid` (Hemiciclo | Resultados por partido, y Resultados por bloque como hoy), "Evolución de las
publicaciones", "Una encuestadora frente al promedio", "Todas las encuestas", pie.

- KPIs (cinco): "Primer partido" (siglas en su color, "32,7 % del voto"; apoyo "P(primero) 97 %"); "Escaños
  del líder" ("140 de 350"; apoyo "Rango 95 %: 118–160"); "Mayoría absoluta" ("> 99 % Derecha"; apoyo en verde
  "PP + VOX + SALF: 203 escaños", los partidos del bloque según `bmaps.vs`); "Sondeos del ciclo" ("315 de 23
  encuestadoras"; apoyo "Desde el 7 ago 2023", la fecha del primer sondeo); "Último sondeo" ("3 oct
  SocioMétrica"; apoyo "para El Español · 1000 entrevistas"). En modo Elección, los dos primeros leen el
  pronóstico. Todo sale del servidor (`summary`, `meta`, `polls`).
- Tarjetas de partido: los partidos de `bmaps.main` en ese orden (D3.6), con el puesto por escaños (empates por
  voto) y los deltas frente a la última elección del ámbito (como en D4). Los sparklines los dibuja
  `pages/promedios.js` con la `series` de la isla (últimos 365 días, una muestra por semana).
- "Promedio de sondeos": la fila de controles (Partidos/Bloques, ventana) y la leyenda de píldoras como hoy,
  con el estilo nuevo; el gráfico, con el tema de D7.
- Hemiciclo: D3.9, con la tira del líder debajo. "Resultados por partido": columnas Partido, % Voto (cifra y
  `.bar`), Escaños (negrita), Rango 95 % (`.rng` y el texto), P(primero). "Resultados por bloque": Bloque, %
  Voto con `.bar` del color del bloque, Escaños, Rango 95 % con `.rng`, P(mayoría).
- "Todas las encuestas": la misma tabla de hoy con el estilo nuevo (encuestadora en negrita y "para {medio}"
  debajo, fecha, muestra, los partidos con su punto en la cabecera, el mejor dato de cada fila en negrita como
  ya hace). La paginación de 15 filas ("‹ Más recientes" / "Más antiguos ›", `#polls-pag`) se queda tal cual,
  con sus botones en `.btn`; el "Ver más (307)" de la maqueta era una aproximación, no un cambio de
  interacción (decisión del autor: el parcial `_polls_table.html` y sus módulos se comparten con el detalle de
  una elección y tienen tests de paridad). "Descargar CSV" pasa a enlace gris al lado de la paginación.
- La tarjeta de cierre ("¿Cómo promediamos las encuestas?", `m.closing`) se queda al final, después de "Todas
  las encuestas", como en todas las páginas; la maqueta no dibujó el final de la página.

### D6. El resto de páginas

Heredan el sistema por las clases comunes. Lo específico:

- **Escaños**: la tabla "Reparto de escaños" adopta `.rng` en "Rango 95 %" (los escaños siguen en cifra, en
  negrita); el gráfico de barras apiladas (`#seats-bar`) usa el tema nuevo; las distribuciones (`ridge`)
  bajan a 64 px por partido (hoy unos 120) para que la tarjeta no ocupe dos pantallas; la "Calculadora de
  coaliciones" usa las píldoras `.lg` y muestra el resultado como un chip grande (verde si llega a la mayoría,
  gris si no). "Resultados por territorio" no cambia salvo el estilo. El orden de las tarjetas y la tarjeta de
  cierre no cambian.
- **Histórico Errores**: la barra de filtros con chips/segmentados nuevos; "Cifras de la selección" (`.figures`,
  cuatro cifras que rellena `history-dom.js`) pasa a una fila de cuatro KPIs (D3.5) con los mismos `id`; el
  resto, estilo.
- **Ranking Encuestadoras**: la tabla ya tiene el estilo de barras de la fase 7f; adopta la cabecera y las
  celdas de D3.8. "Efectos de encuestadora" y "Rebaño", estilo.
- **Perfil de encuestadora**: la línea "Tipo: … · Calidad: … · Rating: …" de la tarjeta `profile` pasa a una
  fila de tres KPIs (Rating, con el apoyo "entra así en el promedio"; Calidad previa; Tipo). "Camino al rating"
  conserva su gráfico y su tabla, solo con el estilo nuevo (decisión del autor tras la revisión del
  constructor; el borrador decía "como fila de KPIs" sin haber mirado la tarjeta).
- **Detalle de una elección**: hemiciclo (D3.9) con la tira del líder (sin P(primero)); tablas con `.bar`; el
  resto, estilo.
- **Método** (las cuatro subpáginas): prosa a 15 px con `max-width: 72ch`, títulos Outfit; la subnavegación del
  método como segmentado; las tarjetas del índice como tarjetas enlazadas (D3.11) sin icono.
- **Error** (`error.html`): hereda.

### D7. Gráficos (ECharts)

`charts/base.js`: `THEME.fontFamily` pasa a `'DM Sans', system-ui, sans-serif`; `textColor` `#1a1c22`;
`gridColor` `rgba(0,0,0,.08)` (las líneas de la rejilla, más suaves) y el eje en `rgba(0,0,0,.15)`; etiquetas
de ejes a 10,5 px en `#5a5f72`; tooltip con fondo blanco, borde `--color-line`, radio 10, sombra `--shadow` y
texto en tinta; `barColor` sigue `#3a6ea5`. Las leyendas internas de ECharts se mantienen donde no hay leyenda
de píldoras. Las cifras de las etiquetas de los gráficos (valores al final de las barras, etiquetas del
hemiciclo) van en Outfit cuando son grandes (hemiciclo) y en DM Sans el resto. `resize` en `window.resize`,
como hoy. Los gráficos de la portada se dibujan con `pages/inicio.js` (módulo nuevo que importa
`charts/hemicycle.js`).

### D8. Marca

`web/brand/build.py` pasa a leer Outfit ExtraBold (800) de `web/dist/vendor/fonts/` (el mismo fichero woff2
que sirve la web) y recalcula las constantes de la lente para la "o" de Outfit (anchura, altura y grosor de su
"o": el anillo ocupa la altura de la "o" y el grosor del trazo de la fuente; el punto central, el 28 % del
radio); el wordmark se regenera (`logo.svg`, mismo `viewBox` de unos 100 px de alto, misma marca). El favicon
(`favicon.svg`, `favicon.ico`, `apple-touch-icon.png`) sale del mismo script y no cambia de forma. `INK`,
`ACCENT` y `BACKGROUND` siguen. El script sigue sin ser dependencia de la web.

### D9. Lo que no cambia

Rutas, API, publicación, los `hash` de los gráficos, los textos del método y del glosario, los "Cómo leerlo" de
las tarjetas que no cambian, los colores de partidos y bloques, la CSP, nginx, Docker, la veda y su pancarta
(solo su posición, fija arriba).

## Lo que decide el constructor

- Cómo subconjuntar las fuentes (`pyftsubset` con `--unicodes` del latín básico y extendido, o los ficheros
  estáticos de Google Fonts tal cual si pesan menos de 30 KB cada uno) y los nombres de fichero.
- Los `symbol` exactos de los iconos (trazo de 2 px, 24×24, `currentColor`), mientras sean reconocibles y del
  mismo estilo.
- Los detalles de ECharts para la línea y la píldora de la mayoría del hemiciclo (`graphic` o una serie
  auxiliar), mientras el resultado sea el de la maqueta.
- Cómo partir los módulos JS (un `pages/inicio.js` nuevo; lo demás, en los módulos de cada página).
- Las clases y la organización de `site.css` (una sección por componente, como hoy), y si los sparklines se
  dibujan en SVG a mano (como la maqueta) o con ECharts (SVG a mano pesa menos y basta).
- Los nombres de las pruebas; los umbrales de las capturas.

## Pruebas que fijan esta fase

- `tests/test_pages.py`: cabecera sin frase, menú y barra inferior (entradas, enlaces del evento actual, la
  activa), portada (titular, tarjeta destacada con la cifra y la barra de escaños, chips, autonómicas en filas,
  línea de no publicados, últimas encuestas, fichas de pasadas), barra de evento (cámara, píldora de
  estimación), Promedios (KPIs, tarjetas de partido con deltas y "nuevo", tira del líder, celdas `.bar`/`.rng`),
  filtros `rng` y los nuevos ayudantes del contexto (`hero`, `latest_polls`, deltas con y sin resultado
  publicado); páginas restantes sin regresión de textos.
- `tests/test_js_charts.py`: opción del hemiciclo (etiquetas dentro a partir del 5 %, sin líneas, centro,
  marca de la mayoría).
- Un test de ficheros: las fuentes referenciadas por `site.css` existen en `web/dist/vendor/fonts/`, las
  antiguas no se referencian, y `LICENSE` nombra las nuevas.
- `python web/brand/build.py` regenera los cuatro ficheros de marca sin error (comprobación manual del
  constructor, con captura).
- Capturas con Chrome sin cabeza (comprobación manual del constructor, como en las fases 7c y 7d: uvicorn con
  el paquete local y la base, sin herramienta de navegador nueva en el repositorio) de `/`, `/es`,
  `/es/escanos`, `/casas`, `/casas?id=`, `/elecciones`, `/elecciones?date=` y `/metodo` a 390, 1000, 1440 y
  1920 px, sin errores de consola y sin desplazamiento horizontal de la página
  (`document.documentElement.scrollWidth <= innerWidth`, leído con `--dump-dom` o desde la consola). Las
  capturas se adjuntan al informe del último paso del plan.
- `python -m pytest -m "not integration" -q` en verde.

## Decisiones del autor tras la revisión del constructor (2026-10-15)

Un constructor leyó el borrador sin poder preguntar y devolvió seis dudas; ninguna cambia lo que Luis aprobó,
así que las resolvió el autor y quedan marcadas en el texto:

1. Leyenda de los gráficos de series de Promedios: píldoras HTML (como la maqueta) gobernando a ECharts; los
   demás gráficos conservan su leyenda interna (D3.10).
2. "Todas las encuestas": se conserva la paginación de 15 filas; solo cambia el estilo (D5).
3. Portada sin `es` publicado: el hero es el primer ámbito publicado, con textos derivados del ámbito (D4).
4. Las tarjetas de cierre se quedan en todas las páginas, también en Promedios y Escaños (D5, D6).
5. Las capturas son una comprobación manual del constructor, sin herramienta de navegador nueva (Pruebas).
6. Perfil de encuestadora: los KPIs son los tres de la cabecera; "Camino al rating" conserva gráfico y tabla
   (D6).

## Estado

Aprobada el 15 de octubre de 2026 tras la revisión del constructor y los dos apuntes de Luis. Siguiente paso: el
plan de implementación (`docs/superpowers/plans/`).
