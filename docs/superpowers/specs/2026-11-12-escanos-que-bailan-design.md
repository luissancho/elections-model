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
- **En el aire**: probabilidad ≥ 1% y < 99% (`OPEN_FLOOR = 0.01`). Cada uno es un **aspirante**.
- **Descartado**: probabilidad < 1%. No se dibuja ni se cuenta.
- **Escaños en el aire de una circunscripción**: sus escaños menos los seguros (nunca menos de 0). Con 8 escaños y
  5 seguros hay 3 en el aire, aunque se los disputen 6 aspirantes.
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
