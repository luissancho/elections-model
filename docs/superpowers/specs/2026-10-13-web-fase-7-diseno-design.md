# Fase 7 de la web: diseño, narración y método (2026-10-13)

Diseño acordado con Luis el 2026-10-13 en el chat de la fase 7. Las fases 0-6 están integradas en `master`
(spec general: `2026-10-07-web-publicacion-design.md`, con el estado al cierre de la fase 6 al final). Esta
fase no toca el paquete publicado, el contrato de datos, nginx ni la imagen Docker: cambia las páginas.

## Intención

Que cualquiera entre en la web y vea en qué situación estamos, con el modelo partido en dos: el promedio de
encuestas y la simulación de escaños. Hoy la web está completa pero es "muy simple, sin detalles" (Luis): siete
páginas con tarjetas sin más explicación que un párrafo "Qué es esto", una cabecera que da el mismo peso a
los selectores que a la navegación y un método en una sola página sin fórmulas. Luis quiere un diseño
profesional al estilo de Objetivo 176 (bloques en tarjetas con título y descripción, fondo pastel de periódico
económico), una historia por partes que siga la estructura de sus notebooks (PollsForecast, PollsSimulations,
PollsErrors + PollstersEvent, PollstersRatings) y un método con el detalle de las páginas de 538, en prosa con
alguna fórmula suelta.

## Objetivos

1. Un Inicio que elige el evento; dos páginas por evento (Promedios y Escaños) y tres globales (histórico,
   rating y método) que cuentan la historia en orden, cada una cerrada con un enlace a su texto del método.
2. Cada gráfico y cada tabla con un título, una descripción de qué muestra y un "Cómo leerlo" con su contexto.
3. Un método en cuatro subpáginas: tres textos (promedio, escaños, rating) y el backtest, al final.
4. Una sola convención para las medidas de error en toda la web, escrita una vez.
5. Los puntos de diseño que quedaron abiertos en las fases 5 y 6, resueltos (sección "Decisiones").
6. Desplegado antes de la semana del 2026-11-17 (ensayo de la veda); la veda congela el 2026-11-23.

## No objetivos

- nginx, Docker, ECS, credenciales, el paquete publicado, `publish`, el contrato (`docs/web/contrato.md`) y
  la API `/api/v1`: no cambian. Si una página necesita un dato que el paquete no tiene, se para y se pregunta.
- Modo oscuro, selector de idioma, nombre nuevo, logo y dominio: más adelante.
- Simuladores o calculadoras nuevas (la calculadora de coaliciones existente se queda).
- El histograma de error por partido del detalle de una elección (el paquete no lleva el error por partido de
  cada sondeo) y el gráfico de error frente a tamaño del partido de PollsErrors.
- Un build step de JS o CSS, frameworks o librerías nuevas (KaTeX incluido: las fórmulas van en HTML y MathML).

## Anti-objetivos

- Paredes de texto que escondan los gráficos: la descripción es una frase o dos; el detalle va en "Cómo
  leerlo", plegado.
- Jerga en las páginas públicas: los términos técnicos viven en Método; las páginas hablan en castellano llano.
- Cifras escritas a mano que el paquete puede dar (simulaciones, coberturas, elecciones del backtest).
- Una página que hoy funciona sin JavaScript y mañana no (formularios GET con "Ver", tablas servidas).
- URLs antiguas que den 404: `/promedio` y `/modelo` redirigen.
- Un texto del método que afirme algo que el código no hace.

## Restricciones

- Patrón de página intacto: controlador `Page` + plantilla Jinja2 + isla JSON `#initial-data` + JS que solo
  dibuja (ECharts 5.6 vendorizado); nginx sirve solo `/dist/`; sin `fetch` ni literal `/api/v1` en el JS (test).
- `StrictUndefined` en las plantillas; filtros de `pages/filters.py`; textos en español; `fmt_num` agrupa
  miles.
- CSP actual (`default-src 'self'`, `style-src 'self' 'unsafe-inline'`): las fuentes se sirven desde
  `/dist/fonts/` (permitido por `default-src`), sin Google Fonts ni CDN.
- Las páginas de evento exigen un pronóstico publicado del ámbito y son las únicas que entienden `mode` y
  `run`; las globales llevan el ámbito como filtro propio (`?scope=`, `es` por defecto) y, como hoy, exigen un
  pronóstico publicado del ámbito filtrado (catálogo del run para nombres y colores).
- Presupuesto orientativo, no duro: `/` para `es` en torno a 500 KB de HTML sin comprimir (hoy `/promedio`
  ≈ 460 KB, de los que `series` son 249 KB: 1.164 fechas × 13 partidos × media, lo y hi con dos decimales;
  nginx comprime). Recortes autorizados en la isla de datos, por este orden: las series con un decimal (el
  gráfico no distingue más; el CSV conserva la precisión), `history.runs` y `summary` reducidos a los campos
  que la página dibuja, y las filas de la tabla de sondeos. No se recortan partidos ni bandas de `series`: la
  vista Bloques necesita la banda de todos los partidos del bloque. Fuentes ≤ 2 ficheros woff2 y ≤ 150 KB en
  total.
- Pruebas: `python -m pytest -m "not integration" -q` en verde (568 hoy), `node --check` y los tests de
  paridad con node donde haya lógica pura nueva; `deploy/check-api.sh` y `deploy/smoke.sh` al día.

## Decisiones de Luis (2026-10-13)

1. Mapa: Promedios (página de llegada, mezcla de Portada y Promedio), Escaños, Histórico de errores, Rating de
   encuestadoras y Método con cuatro subpáginas: "¿Cómo promediamos las encuestas?", "¿Cómo simulamos los
   escaños?", "¿Cómo calculamos el rating de las encuestadoras?" y, al final, el backtest (hoy `/modelo`).
2. Promedios: primero el gráfico del promedio de sondeos, por defecto todo el ciclo, con selector de rango, de
   partidos y vista por bloques; después los gráficos de la Portada actual; al final el listado de encuestas y el
   enlace al primer texto. Además, el último gráfico de PollsForecast: los sondeos de una casa frente al promedio.
3. Escaños: lo que tiene, con mejor explicación de cada bloque, sobre todo el abanico por horizonte y la
   evolución de las publicaciones; cierra con el enlace al segundo texto.
4. Histórico de errores y Rating de encuestadoras cierran con el enlace al tercer texto. El histórico añade la
   diana (bullseye) de errores.
5. Los bloques conservan sus nombres del catálogo (Izquierda, Derecha, Regionalista, Separatista).
6. Los selectores de ámbito y de Hoy / Elección se mantienen, a la derecha y menos prominentes (superada por
   12-14: el ámbito se elige en Inicio o en el filtro de cada página global; el modo, en la barra de evento).
7. Las páginas de detalle (una elección, una casa) siguen bajo su pestaña.
8. Diseño: el de Objetivo 176 con fondo pastel de periódico económico; bloques profesionales en tarjetas con
   título y descripción. Nombre y logo, más adelante.
9. Método: prosa con alguna fórmula suelta, al estilo 538.
10. Prioridad si no da tiempo a todo: reordenar la historia, luego las explicaciones, luego el Método.
11. Puntos abiertos de las fases 5 y 6, todos aceptados como se proponen en "Decisiones de los puntos abiertos".
12. (Más tarde el mismo día.) Promedios y Escaños son relativas a un evento, la próxima elección de un ámbito;
    las demás son globales. Hay una página principal de inicio donde se elige el evento. Las globales viven
    solo en la navegación principal, con enlaces desde las páginas del evento (mi recomendación, aceptada).
13. Rutas naturales por evento con el código del ámbito (`/es`, `/es/escanos`, `/es-md`, …) y redirecciones
    desde las URL actuales.
14. El selector Hoy / Elección queda con **Hoy** por defecto y con una explicación clara de qué implica cada
    opción.

## Diseño

### Mapa del sitio y rutas

Dos clases de página. Las **de evento** hablan de la próxima elección de un ámbito (Promedios y Escaños) y
viven bajo la ruta del ámbito. Las **globales** (Inicio, Histórico de errores, Rating de encuestadoras y
Método) viven en la navegación principal y, donde su contenido depende del ámbito, llevan un filtro propio.

| Página | Ruta | Contenido | Cierre |
|---|---|---|---|
| Inicio | `/` | índice de eventos: los pronósticos publicados (uno por ámbito) y las elecciones pasadas con detalle | — |
| Promedios (evento) | `/{scope}` | promedio de sondeos, titular, voto, escaños, hemiciclo, mayoría, evolución, una casa frente al promedio, sondeos del ciclo | "¿Cómo promediamos las encuestas?" + enlaces al histórico y al rating del ámbito |
| Escaños (evento) | `/{scope}/escanos` (+ `?region=`) | lo actual, reexplicado | "¿Cómo simulamos los escaños?" + los mismos enlaces |
| Histórico de errores | `/elecciones` (+ `?scope=`, `?date=`) | filtros, histograma, diana, error por semanas, error del sector, lista; detalle por elección | "¿Cómo calculamos el rating de las encuestadoras?" |
| Rating de encuestadoras | `/casas` (+ `?scope=`, `?id=`) | lo actual, reexplicado; perfil por casa | el mismo tercer texto |
| Método | `/metodo` | portada del método: intro, datos, publicación, limitaciones e índice de las cuatro subpáginas | — |
| | `/metodo/promedio` | "¿Cómo promediamos las encuestas?" | |
| | `/metodo/escanos` | "¿Cómo simulamos los escaños?" | |
| | `/metodo/rating` | "¿Cómo calculamos el rating de las encuestadoras?" | |
| | `/metodo/backtest` (+ `?scope=`) | "¿Cuánto acierta el modelo?": el contenido de `/modelo` más un gráfico de error de escaños | |

`{scope}` es el código del ámbito del catálogo (`es`, `es-md`, …), validado con `webapi.check_scope`
(`fullmatch`); un ámbito desconocido o sin pronóstico publicado da la página 404 de hoy. Las rutas con
parámetro se registran después de las fijas para que `/escanos`, `/casas`, `/elecciones` y `/metodo` no las
capture el parámetro; el constructor fija el orden de resolución del `Router` con un test. Las páginas de
evento aceptan `?mode=` y `?run=`; las globales, `?scope=` (`es` por defecto); ninguna mezcla las dos familias.

Redirecciones 301, conservando el resto de la query:

| Antes | Después |
|---|---|
| `/?scope=X` (con `scope`) | `/X` |
| `/promedio` (con `?scope=X` o sin él) | `/X` (`/es` sin `scope`) |
| `/escanos` (con `?scope=X` o sin él) | `/X/escanos` |
| `/modelo` | `/metodo/backtest` (con `?scope=` si venía) |

`/` sin `scope` es el Inicio y no redirige. `/elecciones` y `/casas` no cambian. El núcleo no tiene ayuda de
redirección: el controlador pone `set_status_code(301)` y la cabecera `Location`, sin cuerpo útil; nginx no
cachea los 301 (`proxy_cache_valid 200 60s`).

Etiquetas del menú principal (mi decisión, con "de"; Luis escribió "Histórico Errores" y "Rating
Encuestadoras"): Inicio, Histórico de errores, Rating de encuestadoras, Método. `PAGES` pasa a esas cuatro
entradas; Promedios y Escaños son las dos pestañas de la barra de evento y las subpáginas del método tienen
su subnavegación propia. Los tests de rutas existentes (`tests/test_web_routes.py`: todo enlace de una
plantilla resuelve) cubren los enlaces nuevos.

### Cabecera, barra de evento y pie

- **Cabecera general** (todas las páginas): el nombre del sitio (h1, enlace a `/`) a la izquierda y el menú
  de cuatro entradas a la derecha. Sin selectores: el ámbito se elige en Inicio (eligiendo un evento) o en el
  filtro de cada página global; el modo, en la barra de evento. El banner de veda se mantiene encima.
- **Barra de evento** (solo Promedios y Escaños), debajo de la cabecera: el nombre del evento ("Elecciones
  generales · 29 de noviembre de 2026": nombre del ámbito del catálogo y `event_date` del `meta`), los días
  que faltan, las dos pestañas Promedios | Escaños y, a la derecha, discreto, el control Hoy / Elección como
  dos botones segmentados (formulario GET con "Ver" sin JS, como hoy) y, con un run fijado, el chip "Run del
  {fecha}" (mi decisión). Debajo del control, una línea que dice qué significa la opción activa (ver "Hoy y
  Elección, títulos y veda").
- **Filtro de ámbito** (Histórico, Rating y la subpágina del backtest), en la cabecera de la propia página: un
  desplegable "Ámbito" con los ámbitos publicados (GET + "Ver" sin JS; envío al cambiar con JS), `es` por
  defecto. La portada y los textos del método no lo llevan: sus cifras vivas son las de las generales y lo
  dicen.
- Móvil (< 720 px): nombre en la primera fila y menú en una fila con desplazamiento horizontal; la barra de
  evento se apila (nombre y fecha; pestañas; control a la derecha). Nada de menú hamburguesa (sin JS para
  navegar).
- **Pie**: como hoy (run, estimación, último sondeo, commit, "congelado desde", atribución, manifiesto) más el
  enlace a `/metodo`; en las páginas globales, el run del pie es el del ámbito filtrado.

### Lenguaje visual

- Fondo de página pastel cálido, de periódico económico; tarjetas blancas con borde de 1 px, radio pequeño,
  sin sombras o con una sombra mínima; líneas y texto apagado con contraste AA sobre el fondo (WCAG 1.4.3).
- Tipografía propia servida desde `/dist/vendor/fonts/` (nginx ya sirve `/dist/vendor/` con caché de un
  año; el resto de `/dist/` se revalida en cada vista) con el nombre de fichero versionado, en woff2 con
  `font-display: swap` y su licencia junto a los ficheros (`/dist/vendor/fonts/LICENSE`): una serif para los
  títulos y una sans para el texto y las tablas, ambas de licencia libre (SIL OFL). Mi decisión es el
  emparejamiento serif + sans; la elección concreta es del constructor dentro del presupuesto. Fallback a la
  pila del sistema.
- Tokens en `:root` (`--color-bg`, `--color-card`, `--color-text`, `--color-muted`, `--color-line`,
  `--color-accent`, `--font-head`, `--font-body`, espaciados); los colores de partido salen del catálogo del run
  y los de bloque de `THEME` (`charts/base.js`), como hoy. Sin modo oscuro.
- Anchura máxima 1.100 px (como hoy) con márgenes de 16 px en móvil; rejilla de dos columnas en escritorio
  que se apila en móvil; gráficos de 360 px de alto en escritorio y 240 px en móvil, como hoy.
- Las tablas conservan `table-wrap` con desplazamiento horizontal; en móvil, la primera columna fija (mi
  decisión; `position: sticky`).
- Sin animaciones de entrada; ECharts con animación desactivada bajo `prefers-reduced-motion`.

### El patrón de tarjeta

Toda sección es una tarjeta con la misma anatomía:

1. **Título** (h2) con, cuando aplica, el subtítulo del modo ("· estimación a …" / "· pronóstico para el …").
2. **Descripción**: una o dos frases, siempre visibles, que dicen qué muestra el bloque y de dónde sale.
3. **Cuerpo**: el gráfico o la tabla, con sus controles.
4. **Cómo leerlo**: un `<details class="how">` plegado por defecto (abierto en la primera tarjeta de cada
   página) con tres partes cortas: qué es cada elemento (línea, banda, punto, color), cómo interpretarlo y qué
   no dice. Como mucho 120 palabras. Sin JS funciona (`<details>` es nativo).

Los párrafos "Qué es esto:" actuales desaparecen: su contenido se reparte entre la descripción y el "Cómo
leerlo". El constructor redacta ambos para cada tarjeta a partir del texto actual, de los notebooks y de los
documentos de método, en castellano llano, sin cifras a mano y sin jerga (ver "Glosario" para los términos
permitidos).

Cada página abre con un **titular**: una línea, fuera de tarjeta, con la situación en palabras llanas y las
cifras vivas del paquete. Promedios: "Si hoy hubiera elecciones, {primero} ganaría con el {pct} % del voto y
{escaños} escaños; {bloque} tiene un {p} % de probabilidad de mayoría absoluta" en modo Hoy, y "El {fecha}, …"
en modo Elección (con la regla de "definitivo" de abajo). Escaños: la mayoría y el bloque más probable.
Histórico: las elecciones analizadas y el error medio del último sondeo de cada casa. Rating: las casas con
rating y la mejor valorada. Método: una frase. (Mi decisión: el titular; los textos exactos, del constructor.)

Cada pestaña cierra con una **tarjeta de enlace** al texto del método que le corresponde: la pregunta como
título, dos frases de adelanto y el enlace.

### Glosario de medidas de error

Una sola definición, en una plantilla parcial `web/templates/_glosario.html` incluida (como `<details>` al pie
de la tarjeta que las usa) en el histórico, el detalle de una elección, el ranking y el perfil de casa:

- **Error medio**: distancia media, en puntos, entre el sondeo y el resultado de cada partido principal. Sin
  signo. (`error_avg`.)
- **Error de bloques**: cuánto se equivocó el sondeo en el margen entre los dos bloques principales, en
  puntos. Sin signo. (`|error_blocks|`, calculado en las páginas como hoy, `error_blocks_abs`.)
- **Sesgo del margen**: el mismo error con signo: positivo, el sondeo daba más margen a {primer bloque} del que
  hubo; negativo, menos. (`error_blocks`.)
- **Herding**: solo donde aparece, con la frase de hoy ("por debajo de 1, sus sondeos varían menos de lo que
  permite el azar").

"Sesgo de bloques" (`bias_blocks`, magnitud sin signo) y "Desviación hacia {bloque}" (`summary.bias`) salen de
todas las tablas públicas y quedan en los CSV y en la API (decisión de Luis sobre lo primero; lo segundo es mi
decisión por la misma razón: para el lector son duplicados del error de bloques y del sesgo del margen).
Toda columna que muestre una de estas medidas usa su nombre del glosario y nada más; las frases "en puntos" y
"miden lo mismo" desaparecen. Antes de etiquetar una columna, el constructor comprueba en la fuente el signo
de cada campo (`error_blocks` del camino al rating de `casa.html` y del ranking vienen de la base, no del
paquete; la convención es la misma: magnitud bajo "Error de bloques", valor con signo bajo "Sesgo del margen").

### Inicio (`/`)

El índice de eventos. Orden:

1. Una línea de presentación del sitio (qué es: un modelo que promedia las encuestas y simula los escaños de
   cada elección) y nada más de texto.
2. **Próximas elecciones**: una tarjeta por ámbito con pronóstico publicado en el manifest, las generales
   primero y en grande, las autonómicas después ordenadas por fecha de elección. Cada tarjeta: nombre y fecha
   del evento, días que faltan, el partido que va primero con su voto y sus escaños (modo Hoy), el bloque con
   más probabilidad de mayoría absoluta y esa probabilidad, el estado de la publicación (fecha del último run;
   "congelado" en veda) y el enlace a su Promedios, con un segundo enlace pequeño a Escaños. Datos: `meta` y
   `headline` del último run de cada ámbito y la probabilidad de mayoría de la misma parte que hoy usa la
   Portada; lecturas con la caché de 60 s de `Site` (dos por ámbito; con ~18 ámbitos, el primer render tras
   caducar la caché puede tardar un par de segundos contra S3, aceptable).
3. **Elecciones pasadas**: una lista discreta de las elecciones publicadas con detalle (`events` de cada
   ámbito del manifest), de la más reciente a la más antigua, con nombre, fecha y enlace a
   `/elecciones?scope=…&date=…`, y un enlace al histórico completo.
4. Tres enlaces de cierre a Histórico, Rating y Método con una frase cada uno (mi decisión: el Inicio es
   también la puerta a lo global).

Sin gráficos (mi decisión: cifras, no miniaturas), así que funciona entero sin JS.

### Promedios (`/{scope}`)

Primera pestaña del evento. Sustituye a `index.html` y `promedio.html` (un módulo `pages/promedios.py` que
reúne `index_context` y `promedio_context`; `index.py` y `promedio.py` desaparecen o quedan como helpers, a
criterio del constructor). Orden de las tarjetas, bajo la barra de evento:

1. **Promedio de sondeos** (gráfico alto, `renderSeries`). Controles encima del gráfico, en una fila:
   - *Ventana*: 3 meses, 6 meses, 1 año, Todo el ciclo; por defecto **Todo el ciclo** (cambia el valor por
     defecto de la fase 3b). `range.js` gana las ventanas nuevas (`WINDOW_DAYS`) y su test de node.
   - *Vista*: Partidos / Bloques.
   - *Partidos*: casillas con el punto de color de cada partido de `series.parties`, en el orden del
     catálogo; por defecto marcados los partidos `main` del run (`meta.bmaps.main`), el resto sin marcar; un
     enlace "Todos" / "Ninguno". En la vista Bloques, las casillas son los cuatro bloques de
     `meta.bmaps.blocks`, todos marcados.
   - Las series de bloque se calculan en el navegador, en un módulo puro `web/dist/js/blocks.js` con test de
     paridad en node: media = suma de las medias de sus partidos; banda = media ∓ raíz de la suma de los
     cuadrados de las semianchuras de cada partido (sorteos independientes, como el notebook: "los bloques
     suman las varianzas de sus partidos"); sondeos = suma de los partidos del bloque presentes en cada
     sondeo (un sondeo sin ninguno de ellos no se dibuja). La proyección del modo Elección **no se
     recalcula**: se dibuja la publicada en `projection.groups.blocks` (la del simulador, la misma verdad que
     `/escanos`), igual que la de partidos usa `projection.groups.parties`. El salto entre la banda del
     promedio y la de la proyección en `as_of` es el mismo que ya existe para los partidos (consenso de las
     casas frente a voto previsto) y lo explica el "Cómo leerlo". Colores de bloque de `THEME`.
   - El estado de los controles vive en el hash (`#ventana=…&vista=…&partidos=…`) y se escribe solo al
     cambiar, como `#coalition=` en Escaños; sin JS, el gráfico no se dibuja (como hoy) y el resto se lee.
2. **Titular**: la tabla de hoy (Partido, Voto e intervalo, Escaños e intervalo, Primero), con el título del modo.
3. **Voto** y **Escaños** (rejilla de dos; barras con intervalo, como hoy).
4. **Hemiciclo** y **Mayoría absoluta** (rejilla de dos, como hoy).
5. **Evolución de las publicaciones** (como hoy, con los puntos huecos de las retrospectivas).
6. **Una casa frente al promedio**: un desplegable de casa (las casas con sondeos en el ciclo, por número de
   sondeos; por defecto la que más tiene) y el gráfico `renderSeries` con la línea y la banda del promedio y
   solo los sondeos de esa casa; sigue la ventana, la vista y los partidos del gráfico 1. Datos: `polls` y
   `series` del run, ya en la página; sin publicación nueva. Réplica del último gráfico de PollsForecast.
7. **Sondeos del ciclo**: la tabla de hoy (últimas 40 filas + "Descargar CSV").
8. Tarjeta de cierre: **¿Cómo promediamos las encuestas?** y los dos enlaces globales filtrados al ámbito:
   "Cómo acertaron las encuestas en las elecciones anteriores" (histórico) y "Las casas que sondean este
   ámbito" (rating).

### Escaños (`/{scope}/escanos`)

Segunda pestaña del evento. Mismas tarjetas y mismo orden que hoy (partido con histogramas, bloques, calculadora de coaliciones,
circunscripciones, abanico por horizonte, evolución de las publicaciones, descargas) con el patrón de tarjeta
aplicado. Cuidado especial con:

- **Abanico por horizonte**: la descripción dice que cada columna es "a cuántos días de la elección" y que el
  horizonte 0 es hoy; el "Cómo leerlo" explica por qué el abanico se abre (la deriva de la opinión crece con
  los días), que la columna de la derecha es el pronóstico del modo Elección y que no es una predicción de
  trayectoria sino de incertidumbre.
- **Evolución de las publicaciones**: qué es un run, qué es un punto hueco, por qué los puntos cambian (nuevos
  sondeos y no cambios del modelo) y que no es una serie de sondeos.
- Cierra con "¿Cómo simulamos los escaños?" y los mismos dos enlaces globales filtrados al ámbito.

### Histórico de errores (`/elecciones`) y detalle (`?date=`)

Página global con filtro de ámbito propio (`?scope=`, `es` por defecto). La lista pasa a contar la historia de PollsErrors y la diana de PollstersRatings; el detalle sigue contando
PollstersEvent. Datos nuevos para la lista: `analysis/{scope}/poll-errors` (una fila por sondeo: el último de
cada casa en cada elección, entre 6 y 42 días antes, con `event_date`, `event`, `pollster`, `days`,
`sample_size`, `weight`, `error_avg`, `error_blocks`, `bias_blocks` y una columna con signo por bloque `vs`;
`blocks` dice cuáles). **Solo cuentan las elecciones publicadas** (decisión de Luis): el contexto filtra las
filas de `poll-errors` por las fechas de `events/{scope}` antes de embeberlas, de modo que el filtro de
elección, el histograma, la diana y la tabla por semanas hablan de las mismas elecciones que la lista; el
titular cuenta las publicadas y el tooltip de cada punto de la diana enlaza a su detalle. Las elecciones con
sondeos y resultado pero sin página (en `es`, las de 1977 a 1989, con menos de 10 sondeos útiles y por tanto
no `featured`; en las autonómicas, casos como Cantabria 2019) quedan fuera. Sin análisis publicado, las tarjetas que dependen de él muestran "Análisis no publicado
todavía", como en `/casas`, y el resto de la página se sirve. Orden:

1. Titular.
2. **Filtros**: Elección (Todas / una, de la más reciente a la más antigua) y Casa (Todas / una). Solo
   JavaScript: gobiernan el histograma, la diana y la tabla por semanas; estado en el hash
   (`#eleccion=…&casa=…`); sin JS, los gráficos no se dibujan y la tabla sale sin filtrar. Un contador "N
   sondeos de M" junto a los filtros.
3. **Histograma del error por sondeo** (`renderHistogram`): medida elegible con dos botones, Error de bloques
   (por defecto) y Error medio; barras de recuento sin ponderar (mi decisión: el peso del notebook no se
   explica al lector), con la media marcada; bins de un punto.
4. **Diana** (`renderBullseye`): un punto por sondeo filtrado, en los ejes de los dos bloques `vs` (las
   columnas con signo de `poll-errors`), color por elección (una escala de colores neutra, no la de los
   partidos; leyenda por elección), tooltip con casa, elección y días antes; anillos de 2, 5 y 10 puntos como
   en el detalle.
5. **Error por semanas antes de la elección**: tabla (elección × semana: sondeos y error de bloques medio) a
   partir de las filas filtradas, semana = `(days + 1) // 7`, como PollsErrors; filas por elección y una
   fila "Todas". Se recalcula con el filtro (lógica pura en `web/dist/js/errors.js` con test de node).
6. **Error del sector por elección** (el gráfico de hoy).
7. **Elecciones publicadas** (la lista de hoy) con columnas Elección, Sondeos, Casas, Error medio, Error de
   bloques, Sesgo del margen y Modelo.
8. Glosario y "¿Cómo calculamos el rating de las encuestadoras?".

Detalle (`eleccion.html`): mismas tarjetas que hoy más un **Histograma del error por sondeo** de la elección
(sus `polls.rows`, misma medida elegible, misma componente) después de "Casas"; columnas con la convención
del glosario y sin "Sesgo de bloques"; la diana del último sondeo de cada casa se queda.

### Rating de encuestadoras (`/casas`) y perfil (`?id=`)

Página global con filtro de ámbito propio (`?scope=`, `es` por defecto). Mismas tarjetas que hoy (ranking con gráfico y tabla, efectos de casa, herding, descargas) con el patrón de
tarjeta y el glosario; la tabla del ranking queda con Casa, Rating, Calidad, Sondeos, Elecciones, Error medio,
Error de bloques y Herding. Perfil: cabecera, camino al rating (tabla con Rating, Rating ajustado, Error de
bloques (magnitud), Sesgo del margen (con signo), Sondeos), diana, efectos de casa por elección, herding por
ciclo; sin "Sesgo de bloques". Cierra con "¿Cómo calculamos el rating de las encuestadoras?".

### Método (`/metodo` y subpáginas)

`/metodo` es la portada del método: una introducción de dos párrafos (qué es el modelo y sus dos partes), un
índice en cuatro tarjetas con las tres preguntas y "¿Cuánto acierta el modelo?", y las secciones transversales
que hoy existen y no pertenecen a ninguna pregunta: **Datos** (fuentes y curación), **Publicación** (runs,
retrospectivos, API) y **Limitaciones** (con el punto de la veda reescrito: la publicación se congela el 23 de
noviembre con el pronóstico definitivo; los sondeos publicados fuera de España durante la veda se cargan en la
base pero el modelo no los usa ni la web los muestra). `metodo.html` se reparte entre esta portada y los textos.

La portada y los tres textos no llevan filtro de ámbito: sus cifras vivas (simulaciones, coberturas) son las de
`es` y el texto lo dice ("en las generales"); la subpágina del backtest sí lo lleva, como las demás páginas
globales (mi decisión).

Las cuatro subpáginas comparten plantilla: subnavegación de las cuatro entradas arriba, índice de secciones
(anclas) al principio, prosa en secciones h2/h3, un "En resumen" de tres o cuatro líneas al inicio, y alguna
fórmula suelta donde aclare más que las palabras, al estilo 538: inline con HTML (`<var>`, `<sub>`, `<sup>`) y
en bloque con MathML (soportado por los navegadores actuales; sin librerías), nunca más de un puñado por
texto. Entre 1.200 y 2.000 palabras por texto. Las cifras que el paquete o el backtest publican se leen de ahí
en la plantilla (`meta.n_sim`, horizontes y elecciones del backtest, coberturas por horizonte, barreras del
catálogo donde estén parametrizadas); las que no, se escriben con su fuente al lado.

Regla de contraste: toda afirmación sobre lo que el modelo hace se comprueba en el código o en los
documentos de método antes de escribirla; el informe de la tarea lista las afirmaciones comprobadas y dónde.
Fuentes por texto (todas en `docs/analisis-2026-09/`):

- **¿Cómo promediamos las encuestas?** (`/metodo/promedio`): `forecaster.md`, `computer-weights.md`,
  `metodo-6-house-effects.md`, `metodo-10-error-estandar-cluster.md`, `metodo-13-herding.md` (incluida la
  muestra efectiva M12 y su variante simétrica, activa por defecto: una casa menos dispersa de lo que permite
  su muestra también pesa menos), `metodo-4-sucesion-encuestas.md`. Secciones orientativas: qué sondeos
  entran; el peso de cada sondeo (muestra, sobrerrepresentación, rating, dispersión); los efectos de casa;
  la regresión local; la banda y por qué se agrupa por casa; de las encuestas al voto estimado.
- **¿Cómo simulamos los escaños?** (`/metodo/escanos`): `simulator.md`, `stat.md`, `metodo-1-otros-barrera.md`,
  `metodo-2-resumenes.md`, `metodo-5-nowcast-horizonte.md`, `metodo-7-composicion.md` (desactivado, y por
  qué), `metodo-8-edad-partido.md`, `metodo-9-oscilaciones-autonomicas.md`. Secciones orientativas: del
  promedio al sorteo del voto (error del promedio, error histórico, deriva); Hoy frente a Elección; de España
  a las circunscripciones (swing proporcional, herencias, oscilaciones); el reparto D'Hondt con un ejemplo
  resuelto; la **tabla de barreras por ámbito** con una nota de las simplificaciones (Canarias y su doble
  barrera, Comunidad Valenciana, Extremadura); qué es cada cifra publicada (mediana, intervalo, probabilidades).
  La tabla de barreras es la **vigente** por ámbito y sale del catálogo `data/es-scopes.csv` (columnas
  `threshold` y `threshold_scope`), que el proceso web ya lee con `webapi.checks.catalogue()`:
  `metodo_context` expone esas columnas (y `rating_weight`, para el texto del rating) con un test; no es un
  dato del paquete y no cambia el contrato. Las excepciones por elección pasada de `data/params.json` (por
  ejemplo Murcia hasta 2015) no entran en la tabla: una nota dice que el modelo aplica a cada elección pasada
  la barrera de su momento.
- **¿Cómo calculamos el rating de las encuestadoras?** (`/metodo/rating`): `computer-ratings.md`,
  `computer-estimators.md`, `metodo-6-house-effects.md` (M6b: el rating con el error descompuesto),
  `metodo-11-ambitos-autonomicos.md` (ratings transversales: las autonómicas entran con el `rating_weight` de
  su ámbito, menor que el de las generales; el valor se lee del catálogo, no se escribe a mano),
  `metodo-13-herding.md`. Secciones orientativas: qué se mide
  (sesgo entre bloques y dentro del bloque, frente al sector de la misma semana); la calidad previa y cómo
  pesa; el rating adjustado y el rating final; el ranking global; el herding como información.
- **¿Cuánto acierta el modelo?** (`/metodo/backtest`): el contenido de `modelo.html` (error del promedio por
  horizonte frente a las líneas base, coberturas, escaños y mayorías, por elección, bloques, "Qué es esto",
  parámetros, descargas) con el patrón de tarjeta, más un **gráfico de error de escaños por horizonte**
  (`renderLines`: error absoluto medio de escaños de la ejecución nowcast y de la ejecución a horizonte) junto
  a la tabla "Escaños y mayorías". `metodo-3-backtest.md` como fuente del texto.

Los tres pasajes revisados de la fase 5 quedan así: el herding sí resta peso por defecto (comprobado en
`Simulator`/`Forecaster`, M13); las autonómicas sí pesan menos (comprobado en `metodo-11`, ratings
transversales); la veda, como arriba. Las "1.000 simulaciones" y la frase de calibración salen del paquete.

### Hoy y Elección, títulos y veda

- El modo por defecto de las páginas de evento es **Hoy** (`nowcast`; hoy el valor por defecto es `forecast`):
  la primera pregunta del lector es qué pasaría hoy. Excepción (mi decisión): con la web congelada, el modo por
  defecto es Elección, porque el pronóstico definitivo es lo que hay que ver y "Hoy" ya no recibe sondeos.
- La línea bajo el control explica la opción activa con las fechas vivas. Hoy: "Si las elecciones fueran hoy:
  el promedio de sondeos leído el {as_of}, con los sondeos publicados hasta el {date_last}". Elección: "Lo que
  puede pasar el {event_date}: añade la deriva de la opinión en los {n} días que faltan, así que los
  intervalos son más anchos". Un `<details>` "¿Qué cambia entre Hoy y Elección?" junto al control da el
  párrafo completo (qué comparten, qué añade cada uno, cuál mirar para qué) y enlaza a "¿Cómo simulamos los
  escaños?". Los subtítulos de las tarjetas ("· estimación a …" / "· pronóstico para el …") se mantienen.
- "Pronóstico definitivo para el {fecha}" solo cuando la web está congelada (`manifest.freeze.active`) **y** el
  run mostrado es el último del ámbito en el manifest. Un run anterior fijado con `?run=` se titula "Pronóstico
  publicado el {run_at}" en modo Elección y "Estimación a {as_of}" en modo Hoy; el banner de veda sigue.
- El titular de cada página respeta la misma regla.

### Datos que lee cada página (sin cambios de contrato)

| Página | Partes |
|---|---|
| Inicio | manifest; `meta` y `headline` del último run de cada ámbito publicado (y la parte de la que hoy sale la probabilidad de mayoría de la Portada); `events` de cada ámbito |
| Promedios | run: `meta`, `headline`, `series`, `polls` (recortado para el gráfico, completo para la tabla hasta 40 filas), `projection`, `summary`; `history` |
| Escaños | como hoy |
| Histórico | eventos del paquete (`events`, `meta` de cada una) + `analysis/{scope}/poll-errors` y `meta` |
| Detalle de elección | como hoy (`polls.rows` alimenta además el histograma) |
| Rating y perfil | como hoy (base + análisis) |
| Método | `meta` del run (cabecera, `n_sim`, `bmaps`), `backtest/{scope}` (coberturas, parámetros) y el catálogo de ámbitos (`webapi.checks.catalogue()`: barreras y `rating_weight`) |

Todo existe en el paquete o en el catálogo que la web ya lee; si una tarea descubre que no, se detiene y se
consulta.

### JavaScript y gráficos

- Módulos puros nuevos con test de paridad en node (`tests/test_js_*.py`, saltados sin node):
  `web/dist/js/blocks.js` (series de bloque), `web/dist/js/errors.js` (filtros y tabla por semanas),
  `range.js` ampliado (ventanas nuevas).
- `charts/series.js`: `renderSeries` ya acepta `parties` y `selected`; la vista Bloques le pasa las series de
  bloque como si fueran partidos, con sus colores y nombres completos; la casa frente al promedio le pasa los
  sondeos filtrados. Sin cambios de firma salvo lo imprescindible.
- `charts/histogram.js` y `charts/bullseye.js` se reutilizan; `bullseye` gana color por grupo (elección) y
  leyenda; `line.js` sirve para el gráfico de error de escaños.
- Páginas: `pages/promedios.js` (sustituye a `index.js` y `promedio.js`), `elecciones.js` (filtros,
  histograma, diana, tabla), `modelo.js` pasa a `metodo.js` (backtest) o se renombra; `casas.js` y
  `escanos.js` apenas cambian; `pages/index.js` pasa a no existir (el Inicio no dibuja). `common.js`:
  `wireControls()` pasa a cablear la barra de evento y el filtro de ámbito (dos formularios GET distintos) y
  gana el cableado de controles con estado en el hash si no está ya factorizado (`#coalition` lo hace en
  `escanos.js`). El `query` que las plantillas propagan en los enlaces se parte en dos: `?mode=&run=` dentro
  del evento y `?scope=` hacia las globales.

### Accesibilidad, móvil y rendimiento

- Como hoy: `role="img"` con `aria-label` en cada gráfico; formularios GET con "Ver" sin JS; sin `fetch`.
- A 360 px de ancho ninguna página desplaza horizontalmente salvo dentro de `table-wrap`; los objetivos
  táctiles miden al menos 40 px; los controles del promedio caben en dos filas.
- Presupuesto de la sección "Restricciones", medido por el constructor con `curl -s | wc -c` sobre `es`.
- Cada página lleva `<meta name="description">` con una frase (mi decisión).

### Pruebas

- `tests/test_pages.py`: rutas nuevas (`/` índice, `/{scope}`, `/{scope}/escanos`, `/metodo/*`), 404 de
  ámbito desconocido y sin pronóstico, orden de resolución del router (las rutas fijas antes que `/{scope}`),
  la tabla de redirecciones 301 con `Location` y query, modo por defecto Hoy (y Elección con la web
  congelada), barra de evento solo en las páginas de evento y filtro de ámbito solo en las globales, etiquetas
  del menú, `_glosario.html` presente en las cuatro plantillas que lo usan, regla de "definitivo" con run
  fijado, isla de datos del histórico (`poll-errors` presente / aviso sin análisis), histograma del detalle,
  ausencia de "Sesgo de bloques" en el HTML público, Inicio con un ámbito publicado y con varios.
- `tests/test_web_routes.py` (enlaces de plantilla resuelven) y el test de "sin `fetch` ni `/api/v1` en el JS"
  siguen en pie.
- Tests de node de `blocks.js`, `errors.js` y `range.js`.
- `deploy/check-api.sh` y `deploy/smoke.sh`: rutas nuevas y redirecciones; `bash -n` limpio.

### Documentación

- `deploy/README.md` "La web" y "Cómo añadir una página": el mapa nuevo, las subpáginas del método, las
  redirecciones, el patrón de tarjeta y el glosario.
- `docs/web/contrato.md`: solo si lista las rutas de páginas (las de la API no cambian).
- Esta spec: "Estado al cierre de la fase 7" al terminar, como en las fases anteriores.

## Lo que queda al constructor

- Los tokens exactos de color y tipografía, el emparejamiento concreto de fuentes dentro del presupuesto y la
  licencia, el radio y la sombra de las tarjetas.
- Los textos: descripciones y "Cómo leerlo" de cada tarjeta, los titulares, las tarjetas de enlace, los tres
  textos del método y la portada del método, con la regla de contraste.
- La estructura interna de `pages/promedios.py`, de la plantilla compartida de las subpáginas del método y del
  cableado de los filtros; si `index.py`/`promedio.py` se funden o se mantienen como helpers.
- Cómo se implementan las rutas con parámetro (`/{scope}`, `/{scope}/escanos`, `/metodo/{seccion}`: una ruta
  con parámetro validado, o rutas explícitas), el orden de registro frente a las fijas y el 404 de un valor
  desconocido con la página de error de hoy.
- El nombre de cada evento en Inicio y en la barra de evento a partir del catálogo ("Elecciones generales",
  "Elecciones a la Asamblea de Madrid" o el nombre del ámbito), y el cálculo de los días que faltan en
  Europe/Madrid.
- La escala de colores por elección de la diana y la leyenda.
- El orden de las tareas dentro de la prioridad de Luis (historia, explicaciones, método) y el reparto en
  commits.

## Orden de ejecución y plazo

Prioridad de Luis: primero la historia (mapa, cabecera, Promedios, redirecciones, Método con sus rutas y el
backtest movido), luego las explicaciones (patrón de tarjeta en todas las páginas, glosario, convención,
histórico con filtros, diana e histogramas), por último los tres textos del método. Si el plazo aprieta, los
textos pueden entrar uno a uno: una subpágina sin texto no se enlaza (ni en el índice ni en la tarjeta de
cierre de su pestaña) y la portada del método conserva mientras tanto el texto actual de `metodo.html` en una
sección "Cómo funciona el modelo". Nada de enlaces a páginas vacías.

Hito: desplegado por Luis antes de la semana del 2026-11-17. Ejecución como en las fases anteriores (plan con
`superpowers:writing-plans`; Luis elige el método de ejecución; trabajo en `dev`; Luis integra).

## Decisiones de los puntos abiertos (fases 5 y 6)

Propuestas el 2026-10-13 y aceptadas por Luis:

1. Histograma del detalle de una elección: sí, el mismo histograma del histórico fijado a esa elección, con la
   media del sector; sin la rejilla por partido.
2. "Pronóstico definitivo" con `?run=`: solo si el run fijado es el último y la web está congelada; un run
   anterior se titula "Pronóstico publicado el {fecha}".
3. Valores con signo y "miden lo mismo": una convención y un glosario compartido (arriba); el perfil de casa
   pasa a ella; "Sesgo de bloques" sale de las tablas públicas y queda en CSV y API.
4. Los tres pasajes de `/metodo` y las cifras a mano: se resuelven al escribir los textos con la regla de
   contraste; herding y autonómicas comprobados y correctos en el fondo; la veda reescrita; las cifras del
   paquete.
5. Barreras autonómicas: tabla por ámbito en el texto de escaños, con nota de simplificaciones; lo
   parametrizado en el catálogo sale de él, el resto con fuente.
6. Backtest: un gráfico de error de escaños por horizonte junto a su tabla, con `renderLines`.

## Revisión del constructor (2026-10-13)

Un revisor independiente leyó el documento como constructor y levantó cinco preguntas. Cuatro se resuelven
aquí como decisiones mías (ya incorporadas arriba); la quinta es de Luis.

1. Presupuesto de `/`: no era alcanzable como estaba escrito; pasa a orientativo con los recortes autorizados
   de "Restricciones".
2. Tabla de barreras y `rating_weight`: del catálogo de ámbitos que la web ya lee, solo la barrera vigente.
3. Vista Bloques en modo Elección: proyección publicada (`projection.groups.blocks`), no recalculada.
4. Fuentes: en `/dist/vendor/fonts/` con nombre versionado para la caché de un año sin tocar nginx.
5. Qué elecciones gobiernan el histórico (filtros, histograma, diana y tabla por semanas): las del análisis
   (`analysis/{scope}/poll-errors`: todas las elecciones del ámbito con resultado y sondeos; en `es`, 16 desde
   1977) o solo las publicadas con página (`events/{scope}`: las `featured`, con al menos 10 sondeos útiles,
   que Luis publica; en `es`, 11 desde 1993). **Decisión de Luis: solo las publicadas**; el contexto filtra las
   filas del análisis por las fechas publicadas (ver "Histórico de errores").

## Preguntas abiertas

Ninguna.

## Estado al cierre de la fase 7 (2026-10-13)

Hecho en `dev` (plan en `c57aab0`, `docs/superpowers/plans/2026-10-13-web-fase-7-diseno.md`; código en
`c57aab0..b2a31c6`, 21 commits, y los tres de la Task 12), con 644 tests unitarios en verde (`python -m pytest -m "not
integration" -q`: 568 antes de la fase y 76 nuevos). Sin tests de integración nuevos; ninguno se ejecutó en la fase.
Ledger de la ejecución: `.superpowers/sdd/2026-10-13-web-fase-7-diseno/progress.md` (informes por tarea en la misma
carpeta).

- `a41de49`, rutas de evento (Task 1, 594 tests): `/{scope}` y `/{scope}/escanos` al final de `pages.ROUTES`,
  validadas con `resolve_event_scope` (404 "Ámbito desconocido." fuera del catálogo); modo Hoy por defecto
  (`parse_state` deja `mode` en `None` y `common_context` lo resuelve con `default_mode`); controlador `Redirect` y
  `Page.redirect` (301 de `/promedio`, `/escanos` y `/modelo`; `Index` redirige `/?scope=`); `PAGES` con las cuatro
  entradas globales, `EVENT_TABS`, la barra de evento (`_event_bar.html`), el filtro de ámbito
  (`_scope_filter.html`), `METODO_SECTIONS`, `subpages` y `method_href`; `check-api.sh` y `smoke.sh` con las rutas
  de evento.
- `0719066` y `93a778b`, Inicio (Task 2, 597 tests): `pages/inicio.py`, `index.html` reescrita y
  `_event_card.html` (pronósticos publicados, uno por ámbito, y elecciones pasadas); los cuatro flecos de la Task 1
  (comprobación de enlaces más estricta en `test_web_routes.py`, `title_of` con el run mostrado explícito, clave
  `event_bar` en vez de `event`, aserción exacta del subtítulo de Escaños).
- `54c7872`, Promedios (Task 3, 608 tests): `pages/promedios.py` funde `index.py` y `promedio.py`; gráfico del
  promedio con ventana (todo el ciclo por defecto), selección de partidos y vista Bloques (`blocks.js`, proyección
  publicada en modo Elección, E8), estado en el hash (E7), gráfico de una casa frente al promedio y series a un
  decimal en la isla de datos.
- `4ad1e78`, lenguaje visual (Task 4, 615 tests): tokens en `:root`, Source Serif 4 y Source Sans 3 en
  `/dist/vendor/fonts/` (E5), `_macros.html` (`card_start`, `card_end`, `how`, `headline`, `closing`), barra de evento
  restilizada; contraste AA calculado (texto apagado 6,05:1 sobre el fondo).
- `16fc4fd` y `1e48483`, Escaños (Task 5, 618 tests): descripción y "Cómo leerlo" de cada tarjeta y cierre con el
  enlace al método; los once flecos de Promedios y de la barra de evento (controles de casa ocultos sin JS, selección
  vacía en el hash, `region` en el formulario del modo, redacción).
- `8ea69a1`, `bd75301` y `df0af16`, Histórico de errores (Task 6, 628 tests): `errors.js`, opciones nuevas de
  `renderHistogram` y `renderBullseye` (E6), filtros, histograma, diana, tabla por semanas y glosario
  (`_glosario.html`); en la ronda de corrección, el histograma cuenta sondeos y la diana da un tono a cada elección
  publicada (paleta de 16).
- `a6ad106`, `1e52e99` y `1a7d40c`, Rating de encuestadoras (Task 7, 630 tests): ranking y perfil con el glosario y
  textos por tarjeta; el valor de la base, una media con signo, se muestra una sola vez como "Sesgo del margen"; un
  `</td>` sobrante del camino al rating, quitado.
- `aa87c57` y `6dbe705`, portada del método y backtest (Task 8, 638 tests): introducción, índice de las cuatro
  preguntas, datos, publicación y limitaciones con cifras vivas de las generales (`national_figures`), subnavegación
  `_metodo_nav.html`, `/metodo/backtest` con el gráfico de error de escaños por horizonte; la barrera de las generales
  sale del catálogo.
- `c478a21` y `7f73d4d`, "¿Cómo promediamos las encuestas?" (Task 9, 639 tests; 1.882 palabras): fórmula del rating
  en la escala publicada de 0 a 100 y la frase de la veda como se comporta el código.
- `ea8659b` y `35b5b64`, "¿Cómo simulamos los escaños?" (Task 10, 641 tests; 1.986 palabras): tabla de barreras por
  ámbito del catálogo, deriva con su escala mensual y sin afirmar sorteos correlacionados.
- `5e6add6` y `b2a31c6`, "¿Cómo calculamos el rating de las encuestadoras?" (Task 11, 643 tests; 1.980 palabras): las
  cuatro subpáginas con texto, sin la sección provisional "Cómo funciona el modelo"; la frase de la calculadora de
  coaliciones de Escaños corregida.
- `940c3a8` (Task 12, 644 tests): los menores de redacción aplazados de las tareas 8-11 (abajo); `830cd4b`:
  `check-api.sh` y `smoke.sh` con las rutas del método y las fuentes, `deploy/README.md` ("La web", "Cómo añadir una
  página") y el párrafo de rutas de páginas de `docs/web/contrato.md`; este commit: este estado de cierre y la fila 7
  de la spec general.

Decisiones de ejecución (E1-E12 del plan), tal como quedaron:

1. E1. El Inicio es `Index` con `index.html` reescrita; Promedios es el controlador `Promedios`; `pages/index.py` y
   `pages/promedio.py` se fundieron en `pages/promedios.py`; `pages/modelo.py` pasó a `pages/backtest.py`; `Promedio.py`
   y `Modelo.py` desaparecieron y `Redirect.py` atiende las URL antiguas.
2. E2. `/{scope}` y `/{scope}/escanos` van al final de `ROUTES`; desconocido, 404 `unknown scope`; del catálogo sin
   pronóstico, 404 `scope not published`; `/metodo/{section}` valida contra `METODO_SECTIONS` (404 `section not
   found`).
3. E3. `parse_state` deja `mode` en `None` y `common_context` lo resuelve; las plantillas reciben siempre el modo
   resuelto.
4. E4. `Page.redirect`: 301, `location`, `cache-control: no-store`, cuerpo vacío; la query pierde solo `scope` (y
   `mode`/`run` hacia `/metodo/backtest`).
5. E5. Source Serif 4 Semibold y Source Sans 3 Regular, originales OFL sin modificar, con `LICENSE`; la negrita del
   texto se sintetiza.
6. E6. `renderHistogram` (`unit`, `label`, `markLabel`) y `renderBullseye` (`groupOf`, `colors`, `legend`, `labels`)
   con los valores por defecto de antes.
7. E7. Filtros del histórico y controles del promedio en el hash con `history.replaceState`; lógica pura en
   `errors.js` y `blocks.js`, con tests de paridad en node.
8. E8. La vista Bloques dibuja `projection.groups.blocks` y calcula solo la banda histórica.
9. E9. Textos del método en HTML con `<section>`, `<nav>` de anclas y fórmulas con `<var>`/`<sub>`/`<sup>` y `<math>`;
   cada texto activó su entrada de `METODO_SECTIONS` al entrar.
10. E10. `/metodo` conserva datos, publicación y limitaciones; "Cómo funciona el modelo" vivió en la portada hasta la
    Task 11, que lo retiró.
11. E11. Orden de ejecución = orden de las tareas; entraron las doce, también los tres textos.
12. E12. "Elecciones generales" para `es` y "Elecciones autonómicas · {name}" para los demás; los días que faltan, en
    Europe/Madrid.

Resoluciones tomadas durante la ejecución (las líneas `Ruling:` del ledger, en orden):

1. Se trabaja en `dev` en el checkout principal, sin worktree, como en las fases 1-6.
2. R1: `_event_bar.html` toma el destino del formulario de `event.form_action` (hoy `event_bar.form_action`) en vez de
   la cadena de filtros Jinja del plan.
3. R2: el Inicio saca el bloque líder de `headline[mode].p_majority`; `majority_rows` se queda en Promedios.
4. R3: los literales de fecha de los tests se adaptan a `fmt_date` ("29 nov 2026").
5. Task 4: las fuentes pesan 188 KB (80 + 108) frente a los "≤ 150 KB" orientativos; se aceptan (originales OFL,
   caché de un año, la serif es parte del diseño): unos 40 KB más en la primera visita. Los ficheros llevan la
   versión en el nombre (`SourceSerif4-4.005-Semibold.woff2`, `SourceSans3-3.052-Regular.woff2`) para la caché de un año.
6. Task 6: el plan pedía una escala neutra fija de 8 tonos y la spec "color por elección"; manda la spec: tantos
   tonos distinguibles como elecciones publicadas (paleta de 16; `es` tiene 11).
7. Task 7: donde el valor de la base es una media con signo (fila del ranking, camino al rating) se muestra una vez
   como "Sesgo del margen" y no hay columna "Error de bloques" (no se toma el valor absoluto de una media);
   "Error de bloques" queda donde hay datos por sondeo (histórico, detalle); `_glosario.html` gana
   `with_block_error`.
8. Task 9: la frase de la veda de «Limitaciones» dice lo que hace el código (esos sondeos no entran porque su trabajo
   de campo acaba dentro de la veda; la API no los sirve), no un filtro por etiqueta que el modelo no tiene (ver
   "Pendiente de Luis").

Desviaciones de los implementadores aceptadas en la revisión (no son `Ruling:`; constan en el ledger y los
informes):

- Task 1: test del menú limitado a la navegación principal; `title_of` compara el run mostrado con el último; `/nope`
  da 404 "Ámbito desconocido." (E2).
- Task 2: `bundle.PREFIX` en la ruta de un test del brief; "Partido Popular (PP) va primero" (nombre corto entre
  paréntesis); `index.html` exenta de la comprobación de módulo; parcial `_event_card.html`.
- Task 3: `pollster_id` en el fixture de sondeos; regla CSS de los `fieldset` ocultos; clave del hash `sel`; el
  selector de casa filtra por nombre; "Pronóstico definitivo: el …".
- Task 4: `metodo.html` quedó con el marcado viejo hasta la Task 8; descripciones nuevas cortas donde no había
  texto; Hoy/Elección como pares `input` + `label`; parámetro `level` de `card_start`.
- Task 5: titular sin bloques "La mayoría absoluta está en {mayoría} escaños de {n}."; `region` oculto siempre en
  Escaños.
- Task 8: la frase de calibración vive en «Limitaciones»; el índice enlaza `/metodo/backtest` sin `?scope=`; la
  portada no lleva subnavegación (las tarjetas del índice hacen ese papel); sufijo "(modo Elección)" en la tabla de
  escaños.
- Task 10: cabeceras "Barrera de circunscripción" (`threshold`) y "Barrera de todo el ámbito" (`threshold_scope`),
  comprobadas en `resolve_thresholds` y `alloc_seats`; constantes reestimadas y `max_fc` descritos sin cifra.
- Task 11: descripción de la portada del método reescrita; la tarjeta de cierre del rating enlaza `/casas` y
  `/elecciones`.
- Task 12: la tercera aparición de "viendo solo lo que se sabía en cada fecha" (la descripción de "Cómo se hace el
  backtest" en `metodo_backtest.html`) también se corrigió; el título del índice de la portada pasa a "Las preguntas
  del método"; la línea larga de la veda de `metodo.html` se partió y su test compara con los espacios normalizados;
  `check-api.sh` comprueba además una fuente detrás de nginx (`font/woff2`) y `smoke.sh` su caché de un año.

Verificado en la sesión (2026-10-13, Task 12):

- Suite unitaria: `python -m pytest -m "not integration" -q` → 644 passed, 75 deselected, 54 warnings (los mismos 54
  de antes de la fase).
- `node --check` de los 23 ficheros de `web/dist/js/` versionados → sin errores. `bash -n` de `deploy/check-api.sh` y
  `deploy/smoke.sh` → limpio. `pycodestyle --select=E30 tests/test_pages.py` → limpio.
- `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8001` y `bash deploy/check-api.sh http://127.0.0.1:8001` →
  salida 0, 32 OK, 13 `n/a` (sin nginx, sin base, sin publicar y `congelación: inactiva`), `todas las rutas OK`.
- Tamaño de las páginas (paquete local, run `20261008-181100`): `/es` 387.883 bytes en modo Hoy y 405.436 con
  `?mode=forecast` (presupuesto orientativo ≈ 500 KB); `/es/escanos` 134.802; `/metodo/backtest` 41.967; las demás
  por debajo de 22 KB.
- Recorrido con `curl`: `/`, `/es`, `/es/escanos`, `/elecciones`, `/metodo` y sus cuatro subpáginas → 200; `/casas` →
  503 ("La base de datos no está disponible.", esperado sin base); `/promedio?scope=es&mode=forecast` → 301 a
  `/es?mode=forecast`; `/modelo?scope=es&mode=nowcast` → 301 a `/metodo/backtest?scope=es`; `/escanos?region=28` → 301
  a `/es/escanos?region=28`; `/?scope=es` → 301 a `/es`; `/es-zz` y `/metodo/nope` → 404. El paquete local no tiene
  `analysis` ni `events`: `/elecciones` muestra el aviso de elecciones no publicadas.
- Chrome sin cabeza (proxy desechable en el 8002 que sirve `/dist/` e inyecta una sonda; 360 px dentro de un iframe,
  porque la ventana sin cabeza no baja de 500 px), las once vistas (`/`, `/es`, `/es?mode=forecast`, `/es/escanos`,
  `/elecciones`, `/casas`, `/metodo` y sus cuatro subpáginas) a 1280 y a 360 px: `scrollWidth` igual al ancho, ningún
  elemento fuera fuera de tablas y menú, las dos fuentes `loaded`, sin errores de JS ni de consola, gráficos montados
  (6 en `/es`, 17 en Escaños, 3 en el backtest). Capturas de `/metodo/rating` a 1280 (fórmulas `<math>` bien
  compuestas) y de `/es` a 360 (barra de evento apilada). Uvicorn y el proxy, parados; el nginx del 8080 no se tocó.
- `bash deploy/nginx-check.sh` → `syntax is ok` y `test is successful` (nginx sin cambios en la fase).

No verificado:

- `docker build` y `bash deploy/smoke.sh` (sin Docker en la sesión): la caché de un año y el tipo `font/woff2` de las
  fuentes detrás del nginx de la imagen.
- Producción y S3: nada se publicó ni desplegó; las páginas con `analysis` y `events` publicados (histórico con
  datos, diana, detalle de una elección, ranking y perfil de casa con base) solo se han visto en los tests y, en la
  Task 6, en navegador con datos de prueba.
- Un navegador real en un móvil (solo Chrome sin cabeza a 360 px) y la navegación sin JS fuera de los tests.
- Que las cifras de los textos del método sigan siendo ciertas tras cambios del modelo: las que no salen del paquete
  o del catálogo están escritas a mano con su fuente (informes de las tareas 9-11).

Menores aplazados (las líneas `minor (deferred)` del ledger que siguen abiertas, agrupadas):

- Ya resueltos en la Task 12: la coletilla de cobertura de «Limitaciones» (ahora solo si la cobertura al 95 % del
  horizonte más largo baja de 0,95), "Las cuatro preguntas" escrito a mano, la línea de la veda de ~330 columnas, los
  teasers del backtest ("viendo solo los sondeos publicados hasta cada fecha"), "apenas cambia", el denominador de la
  calidad previa media, el `h` del herding acercado (`h′`), las líneas en blanco PEP8 de `tests/test_pages.py` y las
  dos líneas sin partir de `metodo_rating.html`.
- Código: `common.py` creció de 471 a 771 líneas (los ayudantes de la barra de evento podrían mudarse a
  `pages/event.py`); `_is_missing` importado como privado por `promedios.py`, `escanos.py` y `metodo.py` (exponerlo);
  `poll-errors` se lee dos veces en `elecciones_context`; sin regla CSS `.toc` (el índice usa `section-nav toc`);
  comentario de cabecera torpe en `_glosario.html`; cinco reglas CSS de la barra de evento donde bastaba una.
- Textos y marcado: "95 %" sigue en cabeceras, leyendas y `aria-label` (el nivel del intervalo es una constante del
  método); el "Cómo leerlo" de la diana dice que el punto lleva a su elección y el enlace está en el tooltip; sin aviso
  cuando el análisis existe pero no tiene filas de las elecciones publicadas ("0 sondeos de 0") ni en el histograma
  del detalle vacío; tres tonos azules y verdes cercanos en la paleta de 16; una línea de JSDoc larga en
  `histogram.js`.
- Tests: ninguno comprueba "congelado" en el Inicio ni los guiones de una tarjeta sin titular; el orden de las
  elecciones pasadas se comprueba de forma débil; no hay test de `national_figures` sin un run de `es`; líneas de
  test de más de 120 columnas (algunas del brief).
- De fases anteriores siguen abiertos los menores que esta fase no recogió (estado al cierre de la fase 6, "Menores
  aplazados").

Pendiente de Luis:

- Desplegar antes de la semana del 17 de noviembre (con Docker antes, si puede: `docker build` y `bash
  deploy/smoke.sh`, con el paquete local y con `deploy/elections.env`), y comprobar en producción que las fuentes
  cargan (`/dist/vendor/fonts/`, tipo `font/woff2` y caché de un año: `check-api.sh` y `smoke.sh` lo miran) y que las
  URL antiguas redirigen.
- Publicar `analysis` y `event` de los ámbitos que falten para que el histórico, la diana y el rating tengan datos
  (en local no hay ninguno).
- Nombre y logo del sitio (decisión 8: "más adelante").
- Modelo (hallazgo de la Task 9, fuera de la fase): `Forecaster.load_polls` no pasa `drop_contexts`, así que el
  promedio en vivo excluye los sondeos `wban`/`exit` solo por la ventana de fechas (`drange=6`), no por etiqueta; el
  `Computer` sí los quita por etiqueta. Una línea (`drop_contexts=['wban', 'exit']`) lo haría explícito; si cambia,
  la frase de la veda de «Limitaciones» puede volver a decir que el modelo no los usa. La nota del cierre de la fase 6
  que decía lo contrario queda enmendada en la spec general.
- Histórico sin JS: el brief de la Task 6 sustituyó el "sin JS la tabla sale sin filtrar" de la spec por una tabla
  vacía con `<noscript>`; decidir si basta.
- Recorrer las páginas en un móvil real y releer los tres textos del método (1.882, 1.986 y 1.980 palabras) con la
  lista de afirmaciones de cada informe.
- Los pendientes de la fase 6 (ECS, ensayo de la veda la semana del 17, congelación el 23, descongelación el 29) siguen
  en pie.

## Ajustes tras el recorrido de Luis (fase 7b, 2026-10-13)

Plan `docs/superpowers/plans/2026-10-13-web-fase-7b-ajustes.md` (notas de Luis reproducidas allí), commits
`33fd90f..` sobre `dev` (`75f2e30`, `e2bdaf1`, `6c55df8`, `5b681fb`, `f91c783`, `be60bd8`, `0921698` y el de este
cierre); 666 tests unitarios (645 al empezar). El README (`deploy/README.md`, "La web", "Componentes de la fase 7b")
describe el estado resultante.

### Qué se hizo con cada nota

- **Portada con tarjetas**: cada ámbito es una tarjeta entera enlazada (`<a class="card card-link">`, la de las
  generales grande, `.tile-hero`), con "Ver escaños" debajo y fuera del enlace; "Para entender el modelo" son tres
  tarjetas clicables: Histórico Errores, Ranking Encuestadoras y Método.
- **Selector de partidos como el de "Una encuestadora frente al promedio"**: píldoras con el color de cada partido
  (o bloque) al estilo de la leyenda de ECharts, sin título ni Todos/Ninguno; la leyenda de ECharts de Promedios se
  oculta. La píldora marcada toma el color del partido y una tinta legible calculada en el servidor.
- **Tres selectores en tres líneas (Vista, Ventana, Partidos)**: hecho, en ese orden; Partidos sin título visible
  (la `legend` queda solo para lectores de pantalla).
- **Recuadro por sondeo en lugar del día**: el tooltip del promedio es por elemento; en un sondeo enseña su nombre,
  fecha, muestra y las siglas en su color con su valor, de mayor a menor.
- **Quitar el selector Hoy / Elección**: fuera el formulario y el `<details>` que lo explicaba; Hoy por defecto y
  Elección mientras la publicación está congelada (`default_mode`); `?mode=` sigue aceptado en la query.
- **"Sondeos del ciclo": cargar más y columna "Sondeo"**: el servidor pinta 40 filas; la isla lleva todas y "Cargar
  más" añade 40 cada vez hasta que no quedan, y entonces desaparece y el foco pasa al pie de la tabla. La columna
  "Sondeo" (`encuestadora / medio`, o la encuestadora sola) sustituye a Casa y Patrocinador.
- **Escaños: sin "Mín–máx" ni "Media", "Rango 95 %"**: tabla de partidos Partido, Escaños, Rango 95 %, P(escaño),
  P(primero), P(mayoría).
- **Histogramas con líneas de mediana (gruesa) y 95 % (finas), sin el valor**: hecho en los de partido y en el de la
  calculadora.
- **Solo "Bloques ideológicos, con todos los partidos"**: queda una sola tabla de bloques (fuera `#vs-table`).
- **Calculadora con el mismo selector de partidos**: píldoras; el cálculo se rehace al marcar.
- **"Histórico Errores"** y **"Ranking Encuestadoras"**: etiquetas del menú, de las tarjetas del Inicio y de los
  enlaces internos.
- **La diana más grande y centrada**: `#errors-bullseye` cuadrada, hasta 640 px y centrada en su tarjeta; la del
  detalle de una elección y la del perfil (`#bullseye-chart`) siguen en 480 px.
- **"Estimación a …" con la fecha de hoy en modo Hoy y la del evento en Elección**: "Estimación a {hoy}"
  (Europe/Madrid) con el último run, "Estimación del {run_at}" con un run fijado anterior; "Pronóstico para el
  {fecha}" / "Pronóstico definitivo para el {fecha}" / "Pronóstico publicado el {run_at}" en Elección. El pie dice
  "promedio leído el {as_of}" para no repetir otra fecha junto al título.
- **Todos los "Cómo leerlo" plegados**: la macro `how()` ya no tiene `open`.
- **Siempre siglas**: en tablas, leyendas, tooltips y tarjetas; el nombre completo solo en `title=`.
- **Barras por defecto en azul neutro**: `THEME.barColor`.
- **"casa/s" → "encuestadora/s", "patrocinador/es" → "medio/s"**: en todo el texto visible, incluido el método
  ("efectos de encuestadora"); rutas, ids, clases y claves de datos (`/casas`, `casa.html`) intactos.

### Decisiones (rulings del registro de ejecución)

- "Igual que el selector de Una casa frente al promedio" se interpretó como píldoras coloreadas al estilo de la
  leyenda de ECharts (sin título ni botones), en Promedios y en la calculadora; las leyendas de ECharts de Promedios se
  ocultan. Si estaba mal: rehacer un control.
- El "recuadro por punto del promedio" es el tooltip por sondeo (`trigger: 'item'`) con nombre, fecha, muestra y
  siglas en color con su valor en orden descendente. Si estaba mal: volver al tooltip por eje.
- El nombre del sondeo se compone en la página como en el cargador (`pollster / sponsor`), sin republicar. Si estaba
  mal: añadir `name` a `POLL_COLUMNS` y republicar.
- "casa" → "encuestadora" se aplica también a "efectos de casa" (→ "efectos de encuestadora") y a los textos del
  método, como pide "todos los sitios". Si estaba mal: deshacer el término en el método.
- Sin control de modo, el `<details>` "¿Qué cambia entre Hoy y Elección?" desaparece de la barra; queda la línea
  `mode_text`. Si estaba mal: reponer el `<details>`.
- Las píldoras marcadas llevan una tinta calculada en el servidor (texto oscuro sobre colores claros, blanco sobre
  oscuros): el "texto blanco" del brief no se leía sobre BNG o CC. Si estaba mal: una variable CSS.

### Menores aplicados en el cierre (`0921698`)

Etiquetas nuevas y la nota "sin selector de modo" en `docs/web/contrato.md`; fuera las reglas CSS muertas
`.card-hero` y `.coalition-parties` (`.card h3 a` se queda: la usan las tarjetas de `/metodo`); los tests del Inicio
comprueban que el `<h3>` está dentro del enlace de la tarjeta grande y que los tres enlaces de sección son tarjetas;
fuera la clave `n_polls` del contexto de Promedios (ninguna plantilla la leía); importación de `pages/__init__.py`
ordenada.

### Menores que quedan

- Veda: con la publicación congelada, `?mode=nowcast` (sin camino en la interfaz) titula "Estimación a {hoy}" sobre
  datos congelados; comprobarlo en el ensayo de la semana del 17-11.
- Inicio: el color del borde de las tarjetas al pasar el ratón es una decisión visual de Luis.
- Promedios: unos ~15.000 símbolos invisibles de "día" por gráfico para el tooltip del promedio (fluidez en un móvil;
  alternativa: un scatter invisible por línea); la isla repite las 40 primeras filas de la tabla; el tooltip escribe
  "muestra 1000" (`fmtInt` de JS) y la tabla "1.000" (`fmt_int` de Python), anterior a la fase.
- Escaños: aserción de píldoras laxa en los tests; objeto de etiquetas duplicado en `escanos.js`; cuando un extremo
  del rango cae en el bin de la mediana, las dos líneas se solapan (se ve en SALF).
- Vistos en la verificación de este cierre (no pedidos en las notas, para que Luis decida): la tabla de bloques, la de
  la región y la frase de la calculadora siguen diciendo "Intervalo 95 %" (la nota solo cambiaba la tabla de
  partidos); la línea discontinua de la mayoría del histograma de la calculadora perdió su etiqueta "Mayoría: 176"
  con las demás; a 360 px la fila de Ventana ocupa dos líneas; la columna "Sondeo" va alineada a la derecha como las
  numéricas; el sitio no tiene favicon (`/favicon.ico` da 404, anterior a la fase).

### Verificación (2026-10-13, en local)

- `python -m pytest -m "not integration" -q`: 666 passed, 75 deselected. `node --check` de los 25 ficheros de
  `web/dist/js`: sin errores. `bash -n` de `deploy/check-api.sh` y `deploy/smoke.sh`: sin errores (ninguno nombra las
  etiquetas cambiadas; `?mode=` de `check-api.sh` sigue siendo válido).
- `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8001` sobre el paquete local (run 20261008-181100) y `bash
  deploy/check-api.sh http://127.0.0.1:8001`: "todas las rutas OK" (lo de nginx, base y eventos, `n/a`). `/es` pesa
  433.659 bytes (34.057 en gzip), `/es?mode=forecast` 451.212 (35.902), `/es/escanos` 131.715. Títulos: "Estimación a
  13 oct 2026" y "Pronóstico para el 29 nov 2026".
- Chrome sin cabeza (155) por el protocolo DevTools, detrás de un proxy desechable que sirve `/dist/`, a 1280 y
  360 px (con emulación de móvil) en `/`, `/es`, `/es/escanos`, `/elecciones`, `/metodo` y `/metodo/backtest`: sin
  desbordamiento horizontal (`scrollWidth` = ancho) y sin errores de consola (salvo el 404 de `/favicon.ico`); fuentes
  cargadas; la barra de evento sin formulario. Inicio: cuatro tarjetas enlazadas con su `<h3>`; un clic en el texto de
  la tarjeta grande lleva a `/es`. Promedios: Vista, Ventana y Partidos en tres filas, 13 píldoras, leyenda de ECharts
  oculta; con el foco en la píldora de PP (llegando con Tab, anillo de foco visible), Espacio la desmarca, quita PP
  del gráfico y del de la encuestadora y del hash, y otro Espacio la repone; el tooltip del último sondeo dice
  "SocioMétrica / El Español · 3 oct 2026 · muestra 1000" y las 13 siglas en color de mayor a menor; "Cargar más"
  pasa de 40 a 315 filas en siete clics, el pie dice "Todos los sondeos del ciclo (315)" y el botón desaparece; sin
  "None" ni "undefined". Escaños: 17 gráficos; los histogramas llevan tres líneas sólidas (1, 3 y 1 px) sin
  etiqueta; Espacio sobre la píldora del PSOE en la calculadora la marca y el resultado pasa de "3 partidos · mediana
  202" a "4 partidos · mediana 309". Diana: en una página de prueba con `site.css`, `#errors-bullseye` mide 640 × 640
  y queda centrada (214 px a cada lado) a 1280 y 294 × 294 a 360, y `#bullseye-chart` sigue en 480 × 480.
- `bash deploy/nginx-check.sh`: "syntax is ok" / "test is successful" (nginx sin cambios).

### Lo que no se verificó

- La diana con datos: el paquete local no tiene elecciones ni análisis publicados (`/elecciones` enseña "Todavía no
  hay elecciones publicadas en este ámbito."); la regla CSS se midió en una página de prueba. Un ámbito congelado en
  el navegador (lo cubren los tests). `/casas` (sin base). Docker, un móvil real y producción.

Pendiente de Luis: recorrer las páginas con estos cambios (en el navegador y en un móvil real), decidir los menores
de arriba y, con la diana publicada (`publish analysis` y `publish event`), mirar que el tamaño le encaja.
