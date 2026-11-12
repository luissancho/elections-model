# «Escaños que bailan»: los escaños de cada circunscripción, uno a uno, con su probabilidad — diseño

Fecha: 2026-11-12. Página nueva de un ámbito con varias circunscripciones (`/{scope}/circunscripciones`) y parte
nueva del run (`seat-odds`). Maqueta aprobada por Luis en lo que se indica abajo:
`.superpowers/brainstorm/53417-1794493344/content/pagina-v3.html` del checkout principal (fuera de git; se abre
como fichero suelto, lleva los datos y las fuentes dentro).

## Intención

Luis quiere que quien visita la web **vea dónde están bailando los escaños**: en cada circunscripción, qué
escaños tiene cada partido asegurados y cuáles están en el aire. Hoy la web da, por circunscripción y partido, el
reparto central, el «Rango 95%» y la probabilidad de sacar al menos un escaño. El rango no basta: en 140 de las
202 casillas con opciones del run del 12-11 son dos valores («2–3») y no dice si es un 90/10 o un cara o cruz.

La idea de Luis: usar las simulaciones del run para decir **en qué porcentaje de ellas** un partido acaba con cada
número de escaños en cada circunscripción, y enseñarlo en dos vistas: una circunscripción con todos sus partidos y
un partido en todas sus circunscripciones. Distribución discreta, no continua.

Cifras del 12-11 (10.000 simulaciones hechas en local con el código de dev, modo Hoy; orientativas, no son las de
un run publicado, y la página abrirá en modo Elección): 206 escaños seguros, 144 en el aire, 318 aspirantes a esos
144 y 51 de 52 circunscripciones con algo en juego.

## Decisiones de Luis (2026-11-12)

1. El propósito: ver dónde bailan los escaños.
2. Dos vistas: por circunscripción (todos los partidos) y por partido (todas las circunscripciones).
3. El dibujo **«escaño a escaño»** (la opción C de la primera maqueta), preferido a la tira de probabilidades por
   número de escaños (A) y a las barras por partido (B).
4. Un escaño es **«seguro» a partir del 99%** de las simulaciones (frente al 90 y al 95%: con el 95% el propio
   modelo espera perder 1,2 de sus «seguros» y sólo en el 39% de las simulaciones no falla ninguno; con el 99%,
   0,2 y el 85%).

Todo lo demás de este documento son decisiones mías que Luis ha visto en la maqueta sin corregirlas o que no ha
visto; están listadas en «Decisiones mías» para que las revise.

## Vocabulario

- **Escaño k-ésimo de un partido en una circunscripción**: lo consigue una simulación si el partido saca ahí
  `k` escaños o más. Su **probabilidad** es la parte de las simulaciones en que lo consigue.
- **Seguro**: probabilidad ≥ 99% (`SAFE_SHARE = 0.99`).
- **Posible**: probabilidad ≥ 1% y < 99% (`OPEN_FLOOR = 0.01`); lleva su porcentaje.
- **Descartado**: probabilidad < 1%. No se dibuja ni se cuenta.
- **Escaños en el aire de una circunscripción**: sus escaños menos los seguros (nunca menos de 0), es decir, los que
  no son seguros de nadie. Con 8 escaños y 5 seguros hay 3 en el aire: se los disputan los partidos con algún
  escaño posible.
- Tres palabras, cada una con un solo significado (Luis, 2026-11-12, tras la fusión). Hasta entonces el escaño
  posible se llamaba también «en el aire» y cada uno era un «aspirante»: el resto de este documento, anterior al
  cambio, usa todavía esas dos palabras (ver «Estado al cierre»).
- **Reparto central**: el titular de escaños de la web (M13), D'Hondt sobre el voto central. No siempre es lo más
  probable circunscripción a circunscripción: el 12-11, 13 escaños del reparto central salían en menos de la
  mitad de las simulaciones (el primero de Vox en La Coruña, en el 37%). La página lo enseña, no lo esconde.

Los dos umbrales son constantes del módulo de la página, no de la parte publicada: cambiarlos no exige republicar.

## Objetivos

1. **Parte nueva `seat-odds`** por modo en cada run (ver «Datos»).
2. **Página** `/{scope}/circunscripciones` con tres vistas: por circunscripción, por partido y «Los más reñidos».
   Sigue el modo y el run de la página del evento: la misma barra del evento, el modo de `?mode=` o, sin él, el
   de `default_mode` (ahora Elección: faltan menos de 60 días) y el mismo `run` fijado. Sin selector de modo, como
   el resto de la web desde la fase 7b.
3. **Cabecera** con cuatro cifras del ámbito: escaños seguros (de los del ámbito), escaños en el aire,
   aspirantes y circunscripciones con algo en juego (de las que hay).
4. **Vista por circunscripción.** Lista de circunscripciones y, al lado, la ficha de la elegida.
   - Lista: nombre, escaños y «N en el aire» (o «decidida» si no hay ninguno). Orden por defecto: más escaños en
     el aire primero; a igualdad, más escaños y después el nombre. Conmutador «Más en el aire / A–Z». En pantallas
     estrechas la lista es un desplegable.
   - Ficha: nombre, escaños, «N seguros», «N en el aire»; una frase («De los 8 escaños, 5 son seguros (PP 2,
     PSOE 2, VOX 1). Los otros 3 se los disputan 6 aspirantes.»); una fila por partido con sus escaños del reparto
     central, sus fichas y un resumen («2 seguros · 2 en el aire», que se omite en pantallas estrechas); leyenda y
     «Cómo leerlo».
   - Filas: los partidos con alguna ficha o con escaños en el reparto central, por escaños centrales descendentes
     y, a igualdad, por escaños esperados (la suma de sus probabilidades).
5. **Vista por partido.** Un botón por partido con alguna ficha en el ámbito. Ficha del partido: nombre completo,
   escaños del reparto central, «N seguros», «N en el aire»; una fila por circunscripción donde tiene algún
   escaño en el aire (nombre enlazado a su ficha, escaños de la circunscripción, escaños centrales, fichas,
   resumen), de más a menos dudosa: primero aquella cuyo escaño en el aire esté más cerca del 50%. Debajo, una
   línea con las circunscripciones donde sólo tiene seguros («Sin nada en el aire (1): Melilla 1»); la línea no
   sale si no hay ninguna.
6. **Vista «Los más reñidos».** Los 40 escaños en el aire más cercanos al 50%: circunscripción (enlazada),
   «3.º de PP», barra con la raya del 50% y su porcentaje, y si el reparto central se lo da o no.
7. **El dibujo** (ver «Fichas»).
8. **Direcciones propias**: cada circunscripción y cada partido tienen la suya (ver «Rutas») para compartirlas.
9. **Imagen descargable** de la ficha de una circunscripción y de la de un partido, con el botón de tarjeta que
   ya tiene la web (`m.card_image_button`, `card-image.js`).
10. **Entradas** desde la página del evento, en «Resultados por territorio»: un enlace a la página nueva y, en el
    detalle de una circunscripción, otro a su ficha. Sólo si el run que se muestra tiene la parte y el ámbito tiene
    dos circunscripciones o más. Conservan el modo y el run fijado. El detalle lo redibuja el navegador
    (`renderRegion` de `territory-dom.js`), así que su enlace tiene que seguir al selector: un cambio pequeño en el
    JavaScript y en los datos de esa página.
11. **Método**: un apartado en `/metodo/escanos` que explique qué es un escaño seguro y uno en el aire.
12. **Contrato**: enmienda de `docs/web/contrato.md` con la parte, su CSV y las rutas de la página.

## No objetivos

- La tira de probabilidades por número de escaños y las barras (opciones A y B): descartadas por Luis.
- Los **repartos completos** más repetidos de una circunscripción (quién le quita el escaño a quién). En las
  pequeñas funcionan (Cantabria: dos repartos suman el 92%), en Madrid hay 1.715 distintos y el central sale en el
  1%. Queda para otra versión.
- Un mapa.
- Cambiar de circunscripción o de partido sin recargar la página.
- La evolución de un escaño en el tiempo, y rellenar la parte en los runs ya publicados.
- **Comprobar los «seguros» contra el histórico** (cuántos de los escaños que el modelo habría dado por seguros en
  elecciones pasadas se cumplieron) y el cotejo con el resultado del 29-N. Son la continuación natural; aquí no.
- Tocar el modelo, el reparto central, la tabla «Resultados por territorio» o su detalle (salvo los dos enlaces).
- Una entrada en la barra inferior del móvil o en la cabecera.

## Anti-objetivos

- Que un escaño que el modelo pierde en más de 1 de cada 100 simulaciones se pinte como seguro.
- Que la página contradiga al reparto central sin explicarlo: toda ficha en el aire dice si el central la da.
- Que una ficha en el aire lleve la etiqueta «99%» o «100%» (parecería segura), o «0%».
- Que la identidad del partido o el estado de una ficha dependan sólo del color.
- Que la página exista, o se enlace, en un ámbito de una sola circunscripción o en un run sin la parte.

## Restricciones

- **Fecha.** `loreg_guard` rechaza publicar desde el 24-N (el día de la elección y los cinco anteriores): el último
  run que puede llevar la parte es el del 23-N. Un `update` normal con la imagen de jobs reconstruida se la da a
  cada ámbito, así que el exportador va primero en el plan.
- La web y la API viven en `mtpy/controllers`, `mtpy/lib` y los jobs; plantillas Jinja2 pintadas en el
  controlador; nada de FastAPI ni de scripts sueltos.
- CSP `default-src 'self'`: el JavaScript, sólo en módulos de `web/dist/js` (nada en línea). Los estilos en línea
  están permitidos (`style-src 'unsafe-inline'`) y son los que dan el color y la intensidad de cada ficha.
- Identificadores en inglés neutro; los textos públicos, en español, con «sólo» con tilde y sin «empresa»
  (`assert_public_wording`); ningún «Cómo leerlo» abierto.
- Los nombres de módulo de una página siguen a su URL, como `promedios`, `casas` o `sondeos`.
- La página debe poder leerse sin JavaScript (salvo la imagen, el orden A–Z y el salto automático del desplegable).

## Enfoques

Para los datos:

1. **Parte nueva por modo, `seat-odds`** (elegido, decisión mía). El simulador ya guarda los escaños de cada
   simulación por circunscripción y partido (`Simulator.results`); sólo falta resumirlos y escribirlos. Pesa unos
   8 KB en `es` (196 filas, 646 números). No toca ninguna parte existente y la API la sirve sin código nuevo.
2. **Ampliar `districts`** con una lista por fila. Descartado: su CSV son 11 columnas planas («las mismas filas»)
   y una lista lo rompe; además la leen la tabla de territorio, su detalle y el detalle de un sondeo.
3. **Publicar la distribución exacta** (probabilidad de exactamente `k`). Es la misma información, pero la página
   pone el umbral sobre «al menos `k`» y sumar probabilidades redondeadas puede cruzarlo en falso. Descartado.

Para la página:

1. **Pintada en el servidor, cada vista con su dirección** (elegido, decisión mía). El dibujo son cajas HTML, no
   un gráfico: no hace falta ECharts ni JavaScript para verlo. Cada circunscripción y cada partido son un enlace
   que se puede compartir e indexar, y las reglas (umbrales, recuentos, orden) viven en un solo sitio, en Python.
   Coste: cambiar de circunscripción recarga la página.
2. **Dibujada en el navegador desde el JSON**, como la maqueta. Cambio instantáneo, pero sin JavaScript no hay
   página, las reglas se duplican en JS y los enlaces compartidos dependen del estado de la URL. Puede añadirse
   después encima del enfoque 1.
3. **Tarjetas dentro de la página del evento.** Descartado: la vista por partido llega a 51 filas (PP y PSOE
   tienen algún escaño en el aire en 51 de las 52 circunscripciones con el umbral del 99%) y esa página ya tiene
   cinco secciones.

## Datos: la parte `seat-odds`

Parte por modo (`nowcast` y `forecast`), junto a `districts` y `scenario`.

- **Fuente.** `Simulator.results` (simulaciones × regiones × partidos, enteros) y `Simulator.central_result()`.
  Un método nuevo del simulador, vecino de `unit_summary`, da para una región y cada partido en cuántas
  simulaciones saca al menos `k` escaños, para `k` de 1 al máximo que saca en alguna.
- **`data`**:
  - `n_sim`: simulaciones del modo.
  - `parties`: los partidos del run, en su orden.
  - `regions`: `{id, name, seats}` de cada circunscripción, sin el total del ámbito, en el orden de
    `districts.regions`.
  - `rows`: una por circunscripción y partido con algún escaño en alguna simulación o en el reparto central:
    `{region_id, name, seats, reach}`. `seats` es el escaño entero del reparto central (el de `scenario`);
    `reach` es una lista de **enteros**: `reach[k-1]` son las simulaciones en que el partido saca `k` escaños o
    más. Es no creciente, sin ceros al final, y puede estar vacía (el central da un escaño que ninguna simulación
    da).
- Se publican recuentos, no proporciones, para que el umbral no dependa de un redondeo: la probabilidad es
  `reach[k-1] / n_sim`.
- **Invariante**: en cada circunscripción la suma de todos los `reach` es como mucho `n_sim × escaños`, y
  exactamente eso cuando todas las simulaciones reparten todos los escaños (`alloc_seats` no reparte ninguno si
  ninguna candidatura pasa la barrera). Se comprueba en una prueba con datos sintéticos, no al publicar: no debe
  poder abortar una publicación.
- **CSV**: largo, una fila por circunscripción, partido y `k`: `region_id, region, name, seats, k, sims, p`.
- `publish_forecast` simula siempre con `split=True`: los ámbitos de una sola circunscripción (es-md, es-cb,
  es-nc, es-ri) llevan la parte con esa única región, y su página responde 404. `regions` y `rows` van vacíos
  sólo cuando `central_result()` es `None`.
- Los runs anteriores no la tienen: su página y su ruta de la API responden 404, como cualquier parte que falta.
- Hay que darla de alta en `MODE_PARTS`, `SCHEMAS`, `MODE_EXPORTERS` y las descargas CSV de la página del evento.
  La API sirve sin más el JSON y el CSV de cualquier parte de `MODE_PARTS` (`controllers/Forecast.py`), guion
  incluido.
- Los bundles de prueba son `tests/fixtures/bundle/*.json`, que escribe `write_bundle` de
  `tests/test_webapi_unit.py` recorriendo `MODE_PARTS`: hace falta un `seat-odds.json` en cuanto la parte esté
  dada de alta. El bundle actual tiene una circunscripción y un partido; las pruebas de la página necesitan uno
  con varias. `tests/fakes.py` construye un `Simulator` real sobre matrices sintéticas, así que el exportador
  corre en las pruebas de publicación sin tocarlo.

Ejemplo (Cádiz, 8 escaños, 10.000 simulaciones):

```json
{"region_id": 11, "name": "PP",   "seats": 3, "reach": [10000, 10000, 8010, 730, 10]}
{"region_id": 11, "name": "PSOE", "seats": 2, "reach": [10000, 9950, 4110, 80]}
{"region_id": 11, "name": "VOX",  "seats": 2, "reach": [10000, 8530, 570]}
{"region_id": 11, "name": "AA",   "seats": 1, "reach": [7580]}
```

Con los umbrales: PP 2 seguros, el 3.º al 80% (el central se lo da) y el 4.º al 7% (no se lo da); PSOE 2
seguros (el 2.º, al 99,5%) y el 3.º al 41%; Vox 1 seguro, el 2.º al 85% y el 3.º al 6%; AA, el 1.º al 76%.
Cinco seguros, tres en el aire, seis aspirantes.

## Fichas (el dibujo)

Cada escaño posible de un partido en una circunscripción es una ficha cuadrada (22 px, esquinas de 6 px, 4 px
entre fichas), en orden: 1.º, 2.º, 3.º… Las fichas de una fila saltan de línea si no caben.

| Estado | Relleno | Borde | Etiqueta debajo |
|---|---|---|---|
| Seguro | color del partido, sólido | ninguno | ninguna |
| En el aire y el central se lo da (`k ≤ seats`) | color del partido muy tenue, más intenso cuanto más probable | 2 px continuo del color | porcentaje, en negrita |
| En el aire y el central no se lo da | igual | 2 px discontinuo del color | porcentaje, en gris |
| Descartado | no se dibuja | | |

- **Etiqueta**: porcentaje entero sin decimales, pegado («80%»), limitado a entre 1% y 98%: un escaño en el aire
  nunca se rotula 99%, 100% ni 0%.
- El texto va siempre en la tinta de la página, nunca en el color del partido. El partido se identifica por su
  nombre con el punto de color de las tablas de la web.
- Cada ficha lleva un texto alternativo («3.º de PP: lo consigue en el 80% de las simulaciones», «2.º de PP:
  seguro») y la fila se entiende leída en voz alta.
- La leyenda explica los tres estados y que por debajo del 1% no se dibuja.
- La intensidad del relleno de las fichas en el aire en la maqueta es `0,08 + 0,42 × p` de opacidad.

## Rutas

- `/{scope}/circunscripciones`: vista por circunscripción con la primera de la lista (la que más escaños tiene en
  el aire).
- `/{scope}/circunscripciones/{slug}`: la ficha de esa circunscripción; `slug` es `slugify` de su nombre
  (`cadiz`, `la-coruna`). 404 si no existe.
- `/{scope}/circunscripciones?partido={slug}`: vista por partido (`slugify` de sus siglas). 404 si no existe.
- `/{scope}/circunscripciones?vista=renidos`: «Los más reñidos».
- `mode` y `run`, como en la página del evento. La canónica conserva `partido` y `vista`, y quita `mode` y `run`.
- 404 si el ámbito tiene menos de dos circunscripciones en la parte o el run no la tiene.
- Título: «Escaños que bailan en Cádiz · Elecciones generales 2026» (y sus variantes por vista).
- En la barra inferior del móvil enciende «Encuestas», como la página del evento.

Apoyos que ya existen: la página de un sondeo (`pages/sondeos.py`) es una subpágina del evento con su barra
(`common_context(event=True)`, que también enciende «Encuestas», y el índice de secciones reenlazado a
`/{scope}{query}#id`); `bundle_part(site().run_file, ...)` lee una parte que un run puede no tener; los `slug` de
`data/es-districts.csv` coinciden con `slugify(name)` y no chocan dentro de ningún ámbito, ni los de las siglas de
`data/params.json`; `nginx.conf` no necesita cambios.

## Textos (provisionales: los revisa Luis)

- Título de la página: «Escaños que bailan». Entradilla: «Qué escaños tiene seguros cada partido en cada
  circunscripción y cuáles están en el aire, según las 10.000 simulaciones del modelo.» (con el `n_sim` real).
- «Cómo leerlo»: cada ficha es un escaño que el partido puede conseguir en esa circunscripción. Es seguro si lo
  consigue en el 99% o más de las simulaciones; si lo consigue en menos del 1%, no se dibuja; entre medias está
  en el aire y lleva su porcentaje. La columna «Central» es el reparto central de la página de encuestas: no
  siempre coincide con lo más probable, y por eso hay fichas en el aire que el reparto central da y otras que no.
  «Seguro» no es una garantía: es lo que dicen las simulaciones.
- Enlace de entrada: «Escaños que bailan: los seguros y los que están en el aire ›».

## Riesgos y límites

- **El 99% es del modelo, no del histórico.** Las colas provinciales dependen del ruido del swing (método 18) y
  no se han cotejado con elecciones pasadas escaño a escaño. Por eso el texto dice qué significa «seguro» y por
  eso la comprobación histórica queda apuntada como continuación.
- **Baile en el umbral.** Un escaño en torno al 99% puede pasar de seguro a «98%» de un run al siguiente por el
  azar de las simulaciones (con 10.000, ±0,2 puntos; con 1.000, ±0,6). Los ámbitos publicados con 1.000
  simulaciones lo notarán más.
- En teoría los seguros de una circunscripción podrían sumar más que sus escaños; por eso «en el aire» nunca baja
  de 0. En los datos del 12-11 no ocurre.

## Pruebas

- Simulador: el recuento por región con un `results` pequeño hecho a mano; el invariante de la suma.
- Publicación: la parte en un bundle de prueba, su esquema, su CSV y el ámbito sin circunscripciones.
- Página: las funciones puras (estado de cada ficha en los umbrales exactos, recuentos, etiqueta limitada, orden
  de la lista, de las filas y de «Los más reñidos», frase de la ficha en singular y plural); las tres vistas
  pintadas; los 404; los enlaces de entrada presentes y ausentes; `assert_public_wording`.
- Rutas y API: las rutas nuevas, la parte por modo y su CSV.
- Comprobación en el navegador con un run real en local, en escritorio y a 390 px.

## Decisiones mías (para que Luis las revise)

1. Página propia en vez de tarjetas, con una tercera vista, «Los más reñidos», y las cuatro cifras de cabecera.
2. El nombre «Escaños que bailan» y la ruta `/{scope}/circunscripciones` (y no «provincias»: Ceuta, Melilla y las
   circunscripciones insulares o comarcales de algunas autonómicas no lo son).
3. El partido por parámetro (`?partido=psoe`) y no con ruta propia.
4. El suelo del 1% para dibujar un escaño, simétrico al 99%.
5. El reparto central a la vista: borde continuo o discontinuo.
6. La etiqueta limitada a 1%–98%.
7. Parte nueva con recuentos enteros y página pintada en el servidor, con recarga al cambiar de circunscripción.
8. Fuera de esta versión: repartos completos, mapa, comprobación histórica y cotejo con el 29-N.
9. Todos los textos públicos.

## Lo que queda al constructor

Nombres de funciones, clases y ficheros dentro de las convenciones del repositorio; el CSS exacto a partir de la
maqueta y de los componentes que ya existen (tarjetas, `.seg`, `.pill`, `.dot`, KPIs); cómo se resuelve el
desplegable sin JavaScript (como el formulario de circunscripción de la página del evento); el detalle de la
imagen descargada; el reparto del trabajo en tareas.

## Estado al cierre (2026-11-12)

### Lo construido

Rama `feat/escanos-que-bailan` (`git log --oneline dev..HEAD`):

- Plan en seis tareas: `60507db`.
- Parte `seat-odds` por modo, con su CSV (`Simulator.seat_reach`, `export_seat_odds`): `0fede09`.
- Funciones puras de la página (fichas al 99% y desde el 1%, recuentos, órdenes, frases): `88b6cc5`, y la frase que
  concuerda con uno o ningún aspirante: `6e7988f`.
- Página `/{scope}/circunscripciones` con la cabecera y la ficha de cada circunscripción: `152c2c7`.
- Barra del evento de una subpágina en un solo sitio (sondeo y circunscripciones): `dc83d28`.
- Vista por partido y «Los más reñidos»: `b529437`.
- Entradas desde la página del evento («Resultados por territorio») y descarga de la parte: `2352595`.
- Apartado «Escaños seguros y en el aire» de `/metodo/escanos` y rutas de la página en `docs/web/contrato.md`:
  `86a1dad`.
- Este cierre y la regla de CSS de los enlaces de entrada (abajo): `88f8cec`.
- Ola de arreglos de la revisión de la rama:
  - móvil y tableta («Los más reñidos» en dos líneas, anclas `#ficha` y `#vistas`, la ficha entre 720 y 1099 px,
    pestañas y botones de partido, la unidad de los escaños de la lista): `2096167`;
  - enlaces de la página del evento sin calcular fichas, parte sin simulaciones y ausencia de la parte recordada:
    `1ec1ac2`;
  - línea de la imagen con la elección y la fecha, título de la dirección sin circunscripción, canónica, redondeo de
    las mitades y líneas largas: `50af29b`;
  - contrato y esta sección: el commit que la actualiza.

### Lo que difiere de este documento

- El título lleva delante el «Demoscopio · » que pone `base.html` a todas las páginas: «Demoscopio · Escaños que
  bailan en Madrid · Elecciones generales 2026».
- La imagen descargada se llama `demoscopio-escanos-{scope}-{slug}.png`, con el `slug` de la circunscripción o el de
  las siglas del partido (`demoscopio-escanos-es-madrid.png`, `demoscopio-escanos-es-psoe.png`), como el resto de
  descargas del sitio.
- Las frases con un solo aspirante o con ninguno: el verbo concuerda («disputa 1 aspirante» frente a «disputan 6
  aspirantes») y, sin aspirantes, la parte que los cuenta no sale. Las variantes en singular y vacías de la frase del partido y de la de «Los
  más reñidos» las escribió el constructor; todas tienen su test y son provisionales, como el resto de textos.
- El enlace «Escaño a escaño en …» del detalle de la circunscripción no desaparece cuando la circunscripción
  mostrada no tiene ficha: se pinta oculto y sin `href`, para que el JavaScript lo enseñe si el selector pasa a una
  que sí la tiene.
- La barra del evento de una subpágina (el índice de secciones reenlazado a `/{scope}{query}#id`) la arma una sola
  función, `subpage_event_bar`, que comparten la página de un sondeo y esta.
- Los botones de partido de la vista por partido usan la clase `so-party-btn` (`so-party` ya era la celda del
  partido en la ficha).
- Los dos enlaces de entrada de «Resultados por territorio» (`p.more > a.more-link`) salían como enlaces corrientes
  (15 px, subrayados), porque `.more-link` sólo tenía estilo dentro de `.card-head`. Una regla de `site.css` les da
  el aspecto de los demás «Ver más» del sitio (13 px, seminegrita, sin subrayado salvo al pasar el ratón).
- En el método, los umbrales están escritos («99%», «1%»): el contexto de `/metodo/escanos` no recibe las
  constantes `SAFE_SHARE` y `OPEN_FLOOR`, que sí usan la leyenda y el «Cómo leerlo» de la página. Si cambian, hay
  que cambiar ese párrafo a mano.
- «Los más reñidos» por debajo de 720 px: cada fila en dos líneas («3.º de PP» con su barra; la circunscripción y
  «Central: se lo da» / «Central: no se lo da»), con la cabecera de la tabla oculta a la vista pero no a los
  lectores de pantalla (la tabla declara sus roles ARIA). Así toda ficha en el aire dice si el central la da en
  cualquier ancho (antes la columna desaparecía en el móvil, contra el anti-objetivo).
- Los enlaces caen por debajo de la cabecera fija: la ficha (`#ficha`) deja 68 px arriba (114 px con el aviso de
  congelación; 116 px en el móvil, para que el desplegable se siga viendo encima) y las pestañas (`#vistas`), 68 px.
  El 301 del desplegable lleva `#ficha`, y las tres pestañas y los botones de partido enlazan a `…#vistas`.
- Entre 720 y 1099 px (tableta en vertical) la ficha junto a la lista no lleva el resumen de cada fila («2 seguros ·
  2 en el aire», como en el móvil) para dejar sitio a las fichas, y las cuatro cifras de cabecera van en una fila.
- En el móvil las tres pestañas van en una línea que se desliza de lado (como el índice de la barra del evento), y
  los botones de partido forman un bloque con las esquinas de las tarjetas, separado de las pestañas.
- La imagen descargada de la ficha de una circunscripción y de la de un partido lleva, sólo en la imagen, una línea
  con la elección, la estimación y el día del run: «Elecciones generales 2026 · Pronóstico para el 29 nov 2026 ·
  Actualizado: 8 oct 2026» (texto provisional).
- La dirección sin circunscripción, partido ni vista (`/{scope}/circunscripciones`) se titula «Escaños que bailan ·
  Elecciones generales 2026» y se describe como la página («Qué escaños tiene seguros cada partido en cada
  circunscripción y cuáles están en el aire, según las simulaciones del modelo para …»), no como su primera
  circunscripción, que cambia de un run a otro; su ficha y el nombre de su imagen siguen siendo los de esa primera.
- La canónica quita además lo que la página no usa: `circunscripcion` siempre, `partido` y `vista` en la ruta de una
  circunscripción y `vista` cuando gana `partido`.
- Los escaños de cada circunscripción de la lista se leen con su unidad («8 escaños», «1 escaño»: el singular lo
  añadió el constructor).
- Una parte sin simulaciones (`n_sim` 0) cuenta como ausente: la página responde 404 y la del evento no la enlaza.
  La página del evento recuerda durante el TTL de los punteros si el run tiene la parte, para no releer el paquete
  en cada visita (ningún run publicado hasta hoy la lleva), y arma sus enlaces con las circunscripciones de la parte,
  sin calcular fichas.
- La etiqueta redondea hacia arriba las mitades exactas que la coma flotante deja un pelo por debajo (145 de 1.000
  simulaciones: «15%», no «14%»).
- Tras la fusión, a petición de Luis (2026-11-12): el enlace a la página va también al pie de las tablas de la
  tarjeta «Resultados» de la página del evento (sobre «Ver las distribuciones»), además de al pie de «Resultados por
  territorio». Es el mismo enlace en los dos sitios (macro `m.seat_odds_link`) y con la misma condición.
- Tras la fusión, a petición de Luis (2026-11-12, «no entiendo lo de aspirantes»): tres palabras con un solo
  significado cada una. **Seguro** (99% o más), **posible** (entre el 1% y el 99%, con su porcentaje) y **en el
  aire** (los escaños de una circunscripción o del ámbito que no son seguros de nadie). «En el aire» nombraba antes
  dos cosas (144 escaños en la cabecera, 328 fichas en «Los más reñidos») y «aspirantes» contaba fichas, no
  partidos. Lo que cambia en la página:
  - la cabecera lleva tres cifras (seguros, en el aire, circunscripciones con algo en juego), sin «Aspirantes»;
  - la frase de la ficha nombra a los partidos: «Los otros 3 se los disputan PP, PSOE, VOX y AA.»; con un solo
    partido, «sólo están al alcance de PP»; sin ninguno, la frase acaba en los seguros;
  - la nota de cada fila dice «2 seguros · opta a 2 más» («opta a 1» sin seguros);
  - el partido: pastilla «opta a 85 más», frase «Tiene 66 escaños seguros y opta a otros 85 en 51
    circunscripciones.» («No tiene ningún escaño seguro y opta a…», «…y no opta a ninguno más.») y la línea «Sólo
    seguros (N): …»;
  - «Los más reñidos»: «Los 40 escaños más cercanos al cara o cruz.», sin el recuento de fichas;
  - la leyenda dice «posible, y el reparto central se lo da / no se lo da», y el «Cómo leerlo» y el apartado del
    método («Escaños seguros, posibles y en el aire») definen las tres palabras.
  En el código, `district_views` y `scope_totals` ya no llevan `contenders`; en las filas, `open` sigue siendo el
  número de escaños posibles del partido y, en la circunscripción, los escaños en el aire.

### Comprobación en el navegador

Datos reales de `es`, sólo leídos: un `Simulator` construido como en `publish_forecast` (nowcast, semilla 42, hoy
2026-11-12, 1.000 simulaciones, 52 circunscripciones), del que salieron `seat-odds`, `districts`, `scenario` y el
catálogo de partidos. Se escribieron en un paquete de prueba en una carpeta temporal fuera del repositorio (el del
fixture de los tests con esas partes en los dos modos), servido en local con uvicorn, y se miró con Chrome headless
por el protocolo de DevTools. La página abrió en Elección (el run del fixture está a menos de 60 días); las cifras de
cabecera: 208 seguros de 350, 142 en el aire, 328 aspirantes, 51 de 52 circunscripciones con algo en juego. Nada se
publicó ni se escribió en S3 ni en la base.

La primera comprobación (tarea 6) fue a 1280 y 390 px. Tras la ola de arreglos se repitió con los mismos datos a 360,
390, 768, 1024 y 1280 px (emulación de móvil por debajo de 720), con estos resultados:

| Criterio | Resultado | Cómo se vio |
|---|---|---|
| Madrid (38 escaños) sin desbordar la tarjeta ni pisar las etiquetas | pasa | a los cinco anchos, el ancho de desplazamiento de la tarjeta es su ancho y el de la página, el de la ventana. Columna de fichas: 169, 199, 239, 495 y 604 px; alto de la rejilla: 582, 545, 462, 354 y 313 px (antes, a 720 px, la columna de fichas medía 11 px y la rejilla 1.715) |
| Entre 720 y 1099 px, la ficha junto a la lista y las cifras de cabecera | pasa | a 768 y 1024 px la lista se ve al lado, la ficha no lleva el resumen de cada fila y las cuatro cifras van en una fila; a 1280 px vuelve el resumen |
| Un enlace a `…#ficha` deja ver la cabecera de la ficha | pasa | `/es/circunscripciones/cadiz?mode=forecast#ficha`: a 360 y 390 px la ficha empieza en 116 px, su título en 131 y la cabecera fija acaba en 52 (el desplegable, entre 64 y 100, a la vista); a 768, 1024 y 1280 px la ficha empieza en 68, su título en 85 y la cabecera acaba en 57 |
| Un cambio del desplegable del móvil cae en la ficha | pasa | a 360 y 390 px, de Madrid a Soria: la dirección final es `/es/circunscripciones/soria?mode=forecast#ficha`, el título «Soria» en 131 px, bajo la cabecera (52) y el desplegable (64–100) |
| Las pestañas y los botones de partido caen en `#vistas` | pasa | `…?partido=psoe#vistas`: las pestañas empiezan en 68 px a los cinco anchos |
| Vista por partido de un partido grande sin desplazamiento horizontal | pasa | PP, 51 filas: ni la página ni la tarjeta se desplazan a los cinco anchos |
| «Los más reñidos» sin desplazamiento horizontal y con el estado del reparto central | pasa | a 360 y 390 px la tabla mide 294 y 324 px en tarjetas de 326 y 356 (antes 379 en 324), la página no se desplaza, y las 40 filas enseñan «Central: se lo da» o «Central: no se lo da» dentro de la tarjeta (filas de 57 px, en dos líneas); a 768, 1024 y 1280 px, la columna «Reparto central» con «se lo da» / «no se lo da» y sin el prefijo |
| Pestañas y botones de partido en el móvil | pasa, con una salvedad | las pestañas van en una línea (48 px de alto, antes una píldora de 94), pero su contenido mide 372 px y caben 326 (360) o 356 (390): «Los más reñidos» se corta en el borde y se llega a ella deslizando de lado, como el índice de la barra del evento. Los botones de partido: esquinas de 18 px y 16 px de separación de las pestañas |
| La imagen dice la elección y la fecha | pasa | Madrid a 1280 y 390 px y PSOE a 1280: el PNG lleva «Elecciones generales 2026 · Pronóstico para el 29 nov 2026 · Actualizado: 8 oct 2026» bajo la cabecera, y en la página esa línea no se ve (`display: none`); el resto de la imagen, como en la tarea 6 |
| En `/es`, los enlaces de entrada | pasa | «Escaño a escaño en Madrid ›» con `/es/circunscripciones/madrid?mode=forecast#ficha` y el enlace a la página; en los datos del script, `odds_hrefs` con claves de texto (`"1"`: Álava) |
| «Más en el aire / A–Z» reordena sin recargar | NO COMPROBADO de nuevo | pasó en la tarea 6; la ola no tocó ese CSS ni su marcado |
| Ninguna ficha en el aire dice 99%, 100% ni 0% | NO COMPROBADO de nuevo en el navegador | pasó en la tarea 6; la ola cambió el redondeo de las mitades, cubierto por los tests (0,145 → 15%, 0,575 → 58%, y los extremos 98% y 1%) |
| El caso oculto de «Escaño a escaño en…» (circunscripción sin ficha) | NO COMPROBADO en el navegador | con los datos reales las 52 tienen ficha; lo cubre un test |

Visto fuera de los criterios: a 320 px (fuera de los anchos pedidos) la vista por partido de PP se desplaza de lado
18 px dentro de la tarjeta y la página 1 px; «Los más reñidos» y la ficha de Madrid caben.

### Límites conocidos

- La envoltura de una ficha en el aire mide 31 px (el cuadro de 22 px centrado, para que quepa su porcentaje): entre
  una ficha segura y una en el aire quedan unos 8 px, no los 4 px de «Fichas».
- Con «A–Z» el orden visual de la lista (CSS `order`) no es el del foco del teclado ni el de lectura, que siguen el
  orden «Más en el aire» del marcado.
- La elección «A–Z» no se conserva al pasar a otra circunscripción (cada una es una página).
- El desplegable del móvil está ordenado por escaños en el aire, como dice este documento; ordenarlo por nombre
  queda como sugerencia para Luis.

### Lo que queda (Luis)

- Revisar los textos públicos: entradilla, frases de las fichas, «Cómo leerlo», enlaces de entrada y el apartado
  del método.
- Revisar los textos nuevos de la ola: la línea de la imagen («… · Actualizado: …»), el título y la descripción de la
  dirección sin circunscripción y el prefijo «Central: » del móvil.
- Decidir si las pestañas del móvil (360–390 px) se quedan deslizándose de lado con la tercera cortada o se acortan.
- Integrar la rama en `dev`, push, imágenes nuevas de web y de jobs, y un `update` de cada ámbito antes del 24-N (el
  último run que puede llevar la parte es el del 23-N). No hace falta relleno.
- Al limpiar el worktree, quitar el enlace `.env` que se le añadió para la comprobación.
