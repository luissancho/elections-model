# «Escaños que bailan», segunda versión: tres estados, mapa de provincias y hemiciclo — diseño

Fecha: 2026-11-13. Rehace la página `/{scope}/circunscripciones` construida el 12-11
(`docs/superpowers/specs/2026-11-12-escanos-que-bailan-design.md`, en adelante «la primera versión»). Maqueta
aprobada por Luis («queda muy bien»), con los datos del run 20261113-094116:
`.superpowers/brainstorm/16860-1794563945/content/mapa-partidos-columnas-bloques.html` del checkout principal
(fuera de git; lleva datos y fuentes dentro: leer su `<style>` y su `<script>`, no los blobs base64).

Donde este documento y la primera versión no coincidan, manda éste.

## Intención

Luis quiere que la página se lea de un vistazo y se pueda compartir: un mapa de las provincias y un gráfico de los
partidos, con cada escaño dibujado según su probabilidad. Trajo como referencia dos gráficos de prensa (un mapa de
tarjetas por provincia y unas columnas de cuadrados por partido) y pidió expresamente **no copiarlos**: buscar una
forma propia y mejor.

Tres ideas suyas ordenan todo:

- **Sólo probabilidades.** La página deja de decir qué da o no da el reparto central («olvida lo que el reparto
  central da o no da, céntrate en probabilidades»).
- **Tres estados, y por debajo del 25% nada.**
- **Círculos y anillos** como signo propio, y un **hemiciclo** para diferenciarse.

## Decisiones de Luis (2026-11-13)

1. Estados por probabilidad: **seguro** (99% o más), **probable** (del 75% al 99%), **en juego** (del 25% al 75%).
   Por debajo del 25% no se dibuja ni se cuenta.
2. El escaño es un **círculo**: sólido si es seguro, más claro y con su borde si es probable, con el **borde
   discontinuo** si está en juego. Nada de rayados (sería copiar la referencia). La opacidad va en función del
   porcentaje.
3. Nada del reparto central en esta página: ni menciones, ni líneas, ni marcas.
4. **Mapa de provincias** como vista principal, con un selector de **todos los partidos en liza**.
5. **Partidos**: un **hemiciclo** (lo que más diferencia) y también las **columnas**, con una columna para cada
   partido de los bloques Izquierda y Derecha y, en «Otros», sólo los del bloque Otros.
6. Sin el texto «Por debajo del 25% no se dibuja».

## Vocabulario

Sólo estas palabras, cada una con un significado:

- **Opción de escaño**: el escaño k-ésimo de un partido en una circunscripción (lo consigue una simulación si el
  partido saca ahí `k` o más), cuando su probabilidad llega al 25%. Es la unidad de todos los dibujos y recuentos.
- **Seguro**: probabilidad ≥ 99% (`SAFE_SHARE = 0.99`, como hasta ahora).
- **Probable**: ≥ 75% y < 99% (`LIKELY_SHARE = 0.75`).
- **En juego**: ≥ 25% y < 75% (`PLAY_SHARE = 0.25`).

La probabilidad es `reach[k-1] / n_sim`, comparada sin redondear. Las constantes son de la página.

Desaparecen de la página, de sus textos y de sus tests: «posible», «en el aire», «aspirantes», «reparto central» y
«Central». Se refiere a los textos de la página, no a los nombres: Asturias tiene una circunscripción que se llama
«Central». `OPEN_FLOOR` (1%) deja de existir: el suelo es `PLAY_SHARE`.

Las opciones suman más que los escaños (390 para 350 en el run del 13-11: 203 seguras, 103 probables, 84 en
juego): a un mismo escaño pueden optar varios partidos. La página lo dice una vez, al pie de cada gráfico, con una
frase que depende de las cifras: «Hay 390 opciones para 350 escaños: a un mismo escaño pueden optar varios
partidos.» si las opciones superan a los escaños; si no, el pie no lleva esa frase. (En la maqueta decía «optan dos
o tres partidos»: el 13-11 son 187 opciones no seguras para 147 escaños no seguros, así que era decir de más.)

## El signo: el círculo

Un solo signo en el mapa, las fichas, el hemiciclo y las columnas. El color es el del partido.

| Estado | Relleno | Borde | Etiqueta (sólo en las fichas) |
|---|---|---|---|
| Seguro | sólido | del mismo color | ninguna |
| Probable | opacidad de 0,42 a 0,68 según el porcentaje (`0,42 + 0,26 × (p − 0,75) / 0,24`) | continuo | porcentaje, en negrita |
| En juego | opacidad de 0,06 a 0,26 (`0,06 + 0,20 × (p − 0,25) / 0,50`) | discontinuo | porcentaje, en gris |

- Borde de 1 px en los círculos pequeños (mapa, columnas, hemiciclo) y de 2 px en los de las fichas (22 px).
- La etiqueta es un porcentaje entero pegado («78%»), con el redondeo de ahora, entre 25% y 98%: nunca 99% ni 100%.
- Cada círculo lleva su texto alternativo («3.º de PP: lo consigue en el 78% de las simulaciones», «2.º de PP:
  seguro»). El estado nunca depende sólo del color: sólido, borde continuo o borde discontinuo.
- El texto va siempre en la tinta de la página; el partido se nombra con sus siglas y su punto.

## La página

Título «Escaños que bailan». Entradilla: «Qué escaños tiene seguros cada partido, cuáles son probables y cuáles
están en juego, según las 10.000 simulaciones del modelo.» (con el `n_sim` real). Tres pestañas: **Provincias**,
**Partidos** y **Los más reñidos**. Desaparecen la tira de cifras de la cabecera, la lista de circunscripciones y
su conmutador «Más en el aire / A–Z».

### Provincias (la entrada)

Una tarjeta con el mapa y, debajo, la ficha de la circunscripción elegida.

- **Título** «Dónde bailan los escaños, provincia a provincia». **Entradilla** «Cada círculo es una opción de
  escaño: sólido si es seguro, más claro si es probable y con el borde discontinuo si está en juego. Toca una
  provincia para ver su detalle.»
- **Selector de partido**: «Todos» y un botón por cada partido del run con alguna opción dibujada, en el orden del
  run, con su punto. **Selector de estado**: «Todos», «Seguros», «Probables», «En juego», cada uno con su recuento
  del ámbito (390 · 203 · 103 · 84). Al elegir, los círculos que no cumplen se atenúan (opacidad 0,08); los dos
  selectores se combinan. Los recuentos no cambian con el partido elegido.
- **Mapa**: una rejilla de 10 columnas. Cada circunscripción es una casilla con su abreviatura y sus opciones,
  ordenadas por partido (orden del run) y, dentro, por estado y por `k`. Las casillas crecen hacia abajo según lo
  que contengan. La elegida lleva borde oscuro. Al tocar una casilla cambia la ficha de debajo **sin recargar la
  página**, como en la maqueta (ver «La ficha, en el sitio»).
  - Círculos de 11 px con 2 px de separación; en pantallas estrechas, 7 px y 1,5 px. En escritorio el mapa no pasa
    de 760 px de ancho.
  - La posición y la abreviatura de cada circunscripción salen del catálogo (ver «Datos»). Un ámbito sin posiciones
    (los autonómicos) pinta las casillas seguidas, por nombre, con el **nombre completo** de la circunscripción como
    rótulo y en una rejilla de casillas más anchas (las que quepan con un mínimo de 150 px), no en las diez
    columnas.
- **Leyenda** de los tres estados con sus tramos. **Pie**: la frase de las opciones y los escaños (ver
  «Vocabulario»).
- **Desplegable** de circunscripciones en pantallas estrechas, como hoy, pero ordenado por nombre y con el texto
  «Cádiz · 8 escaños».
- **La ficha, en el sitio.** La página lleva pintadas las fichas de todas las circunscripciones y enseña sólo la
  elegida. Cada casilla del mapa es la etiqueta de un radio; el servidor marca el de la circunscripción de la
  dirección (o la elegida por defecto) y el CSS enseña su ficha y resalta su casilla (radios y `:has()`, sin
  JavaScript). Con JavaScript, además, la dirección cambia a la de la circunscripción tocada sin recargar, y el
  desplegable hace lo mismo; sin él, la dirección no cambia al tocar y el desplegable sigue enviando su formulario
  (el 301 de hoy). Las direcciones de cada circunscripción siguen existiendo y abren la página con su ficha.
  El título y la descripción son los de la circunscripción de la dirección.
- **Ficha** de la circunscripción: nombre, escaños y una pastilla por estado con opciones («5 seguros», «3
  probables», «1 en juego»); la frase; una fila por partido con sus círculos de 22 px y sus porcentajes, y a la
  derecha el resumen («2 seguros · 1 probable»; se omite en pantallas estrechas); la leyenda y el «Cómo leerlo».
  - Filas: los partidos con alguna opción, de más a menos escaños esperados entre las opciones dibujadas (la suma
    de sus probabilidades).
  - Frase: la de hoy, con los partidos que tienen alguna opción probable o en juego: «De los 8 escaños, 5 son
    seguros (PP 2, PSOE 2, VOX 1). Los otros 3 se los disputan PP, PSOE, VOX y AA.» Conserva todas sus variantes
    (un solo partido, ninguno, ninguno seguro, un solo escaño).
  - Sin columna «Central» y sin distinguir opciones por el reparto central.

### Partidos

Una tarjeta con un conmutador **Hemiciclo / Columnas** (se abre en Hemiciclo) y, debajo, el selector de partido y
la ficha del partido elegido.

**Hemiciclo.** Título «El hemiciclo que baila». Entradilla: «Cada círculo es una opción de escaño de ese partido en
alguna provincia. En el núcleo de cada partido, los seguros; hacia el borde, los probables y los que están en
juego.»

- Dibuja **todas las opciones** del ámbito (390 el 13-11), no los escaños de la cámara. Por eso no lleva el total
  de escaños, ni la raya de la mayoría, ni nada que lo haga pasar por un reparto.
- Filas concéntricas (10 con 390 opciones), con las plazas de cada fila en proporción a su radio. Las plazas se
  ordenan por ángulo, de izquierda a derecha, y se reparten a los partidos en el orden del hemiciclo de la web
  (`hemicycle_order`: por bloque y, dentro, por el orden del catálogo). Dentro del trozo de un partido, las plazas
  se llenan de dentro afuera: primero las seguras, luego las probables y por último las que están en juego (de más
  a menos probabilidad).
- En el hueco central, los tres recuentos con su signo: «203 seguros», «103 probables», «84 en juego».
- Debajo, una pastilla por partido con sus tres recuentos («PSOE 66 · 30 · 25»).
- Pie: «Tras cada partido: seguros · probables · en juego.» y la frase de las opciones y los escaños.

**Columnas.** Título «Las opciones de escaño de cada partido». Entradilla: «Cada círculo es una opción de escaño en
alguna provincia. Abajo, los seguros; encima, los probables y los que están en juego.»

- Una columna por cada partido de los bloques **Izquierda** y **Derecha** del run (`meta.bmaps.blocks`), en ese
  orden y, dentro, en el del bloque, y una columna **«Otros»** con los partidos del bloque Otros, cada círculo en
  el color de su partido. El 13-11: PSOE, FA, UP, PP, VOX, SALF y Otros.
- Los círculos se apilan de abajo arriba, 7 por fila (5 en pantallas estrechas): seguros, probables y en juego. Un
  eje con una raya cada 50 opciones.
- Bajo cada columna, las siglas y los tres recuentos. Pie: «Bajo cada partido: seguros · probables · en juego.
  Otros: ERC, JxCat…» (con su punto).

**Ficha del partido.** La de hoy, con los tres estados: un botón por partido con alguna opción dibujada; nombre
completo, una pastilla por estado; una fila por circunscripción donde tiene alguna opción probable o en juego
(nombre enlazado a su ficha, círculos y resumen), de más a menos dudosa (primero aquella cuya opción no segura esté
más cerca del 50%), y la línea «Sólo seguros (N): …». Frase: «Tiene 66 escaños seguros, 30 probables y 25 en
juego.» (sólo los estados con alguna opción; sin ninguna no segura, «Tiene 2 escaños seguros.»).

### Los más reñidos

Como hoy: las 40 opciones más cercanas al 50%, con su circunscripción enlazada, «3.º de PP», la barra con la raya
del 50% y el porcentaje. Sin la columna «Reparto central» ni su línea «Central: …» del móvil.

### Lo que no cambia

Las direcciones (`/{scope}/circunscripciones`, `/{slug}`, `?partido=`, `?vista=renidos`), el 301 del desplegable,
la canónica, los 404 y el 400, el modo y el run de la página del evento, la barra del evento, la línea de la
imagen con la elección y la fecha, y el botón de imagen de cada tarjeta. Se añade `?vista=partidos` (la pestaña
Partidos sin ficha de partido). Sin circunscripción en la dirección se elige la que más opciones en juego tiene (a
igualdad, más escaños y después el nombre).

Llevan botón de imagen la tarjeta del mapa, la del hemiciclo o las columnas, la ficha de la circunscripción y la
del partido. **La imagen recoge lo que está a la vista** (como en las demás tarjetas de la web): el partido y el
estado elegidos en el mapa, el Hemiciclo o las Columnas, la ficha visible. Sale **sin las filas de selectores** y,
cuando el partido o el estado no son «Todos», con una línea que lo dice («PSOE · En juego») y con ello en el nombre
del fichero.

## Textos (provisionales: los revisa Luis)

- «Cómo leerlo» de las fichas: «Cada círculo es un escaño que el partido puede conseguir en esa circunscripción. Es
  seguro si lo consigue en el 99% o más de las simulaciones, probable si lo consigue entre el 75% y el 99% y está
  en juego entre el 25% y el 75%. Los que no son seguros llevan su porcentaje. "Seguro" no es una garantía: es lo
  que dicen las simulaciones.» (con los umbrales de las constantes y comillas angulares).
- Enlace de la página del evento: «Escaños que bailan: seguros, probables y en juego ›».
- Método (`/metodo/escanos`, apartado `#seguros`): título «Escaños seguros, probables y en juego». «En cada
  simulación repartimos los escaños de cada circunscripción. Al terminar contamos, para cada partido y cada escaño
  (el primero, el segundo, el tercero…), en cuántas simulaciones lo consigue. Llamamos seguro al escaño que sale
  en el 99% o más de las simulaciones, probable al que sale entre el 75% y el 99% y decimos que está en juego el
  que sale entre el 25% y el 75%. Como a un escaño en juego suelen optar dos o tres partidos, las opciones suman
  más que los escaños.» El segundo párrafo se queda sólo con: «"Seguro" no es una garantía: es lo que dicen
  nuestras simulaciones, que heredan los errores del promedio y los de la proyección del voto a las
  circunscripciones.»

## Objetivos

1. Los tres estados, el círculo y el vocabulario en toda la página, el enlace del evento y el método.
2. La pestaña Provincias: mapa con sus dos selectores, leyenda, pie, desplegable y ficha.
3. La pestaña Partidos: hemiciclo y columnas con su conmutador, selector y ficha de partido.
4. «Los más reñidos» sin el reparto central.
5. Posición y abreviatura de las 52 circunscripciones de `es` en el catálogo.
6. Imagen descargable de las cuatro tarjetas.
7. Contrato y spec al día.

## No objetivos

- Tocar el modelo, la parte `seat-odds` (su `seats` del reparto central se queda en la parte, sin uso en esta
  página), la API o republicar: basta con desplegar la web.
- Posiciones de mapa para los ámbitos autonómicos (pintan la rejilla seguida).
- Un mapa con las formas reales de las provincias.
- Que los recuentos del selector de estado cambien con el partido elegido.
- Guardar en la dirección el partido o el estado elegidos en el mapa, o el conmutador Hemiciclo / Columnas.
- Dibujar o contar nada por debajo del 25%.

## Anti-objetivos

- Que el hemiciclo se lea como el reparto de los 350 escaños.
- Que aparezca en la página cualquiera de las palabras retiradas, o el reparto central.
- Que un círculo en juego lleve rayado, o que su borde discontinuo no se distinga del continuo en el tamaño en que
  se pinta.
- Que un círculo por debajo del 99% se pinte sólido.
- Parecerse a la referencia en lo que la define: cuadrados, rayado y columnas como única vista de partidos.

## Restricciones

- Como la primera versión: todo en `mtpy/controllers` y `mtpy/lib`, pintado en el servidor con Jinja2; la página se
  lee sin JavaScript; nada de scripts en línea (CSP `default-src 'self'`); los estilos en línea están permitidos.
- Los dos selectores del mapa y el conmutador Hemiciclo / Columnas funcionan sin JavaScript: radios y `:has()`,
  como el orden A–Z de la primera versión, con un bloque `<style>` generado para los partidos del run.
- Ninguna librería de gráficos: mapa, hemiciclo y columnas son HTML y CSS (o SVG en línea) con el mismo signo.
- Identificadores en inglés neutro; textos públicos en español, «sólo» con tilde, sin «empresa» ni «patrocinador»,
  ningún «Cómo leerlo» abierto; porcentajes pegados.
- No hay fecha de publicación que cumplir (no cambia ningún dato), pero interesa desplegarlo antes del 24-N, con
  runs recientes a la vista.

## Enfoques

Para el mapa:

1. **Rejilla de casillas con posiciones del catálogo** (elegido, decisión mía). Diez columnas, cada circunscripción
   en su celda; las casillas crecen con su contenido y nunca se pisan. Es HTML, se lee en el móvil y sale tal cual
   en la imagen.
2. Tarjetas colocadas a mano en coordenadas libres, como la referencia. Descartado: se pisan en cuanto cambia el
   número de opciones de una provincia, y es justo lo que no queremos copiar.
3. Mapa con las formas de las provincias. Descartado: las pequeñas no dan sitio a sus círculos y pide geometría y
   una librería.

Para el gráfico de partidos:

1. **Hemiciclo pintado en el servidor** (elegido por Luis frente a quedarse sólo con las columnas; pintarlo en el
   servidor es decisión mía). Las posiciones se calculan en Python y cada plaza es un círculo con el mismo signo.
2. El hemiciclo de ECharts que ya tiene la web. Descartado: sólo existe con JavaScript, su marca no admite el
   borde discontinuo sin rehacerla y dibuja escaños, no opciones.
3. Sólo columnas. Descartado por Luis como vista única: es lo que hace la referencia. Se quedan como segunda vista.

## Datos

- `data/es-districts.csv` gana cinco columnas: `abbr`, `map_col`, `map_row` (desde 0), `map_w`, `map_h` (1 si
  vacías). Sólo se rellenan en el ámbito `es`; el resto de ámbitos las dejan vacías.
- Colocación de `es` (abreviatura, columna, fila; Barcelona y Madrid ocupan 2 × 2 y Valencia 2 × 1), por código
  INE:

  | Fila | Casillas (columna) |
  |---|---|
  | 0 | 15 C (0), 27 LU (1), 33 AS (2), 39 S (3), 48 BI (4), 20 SS (5), 31 NA (6), 22 HU (7), 25 L (8), 17 GI (9) |
  | 1 | 36 PO (0), 32 OU (1), 24 LE (2), 34 P (3), 9 BU (4), 1 VI (5), 26 LO (6), 50 Z (7), 8 B (8, 2 × 2) |
  | 2 | 49 ZA (2), 47 VA (3), 40 SG (4), 42 SO (5) |
  | 3 | 37 SA (2), 5 AV (3), 28 M (4, 2 × 2), 19 GU (6), 44 TE (7), 43 T (8) |
  | 4 | 10 CC (2), 16 CU (6), 12 CS (8) |
  | 5 | 6 BA (2), 45 TO (4), 13 CR (5), 2 AB (6), 46 V (7, 2 × 1), 7 IB (9) |
  | 6 | 14 CO (4), 23 J (5), 30 MU (7), 3 A (8) |
  | 7 | 21 H (2), 41 SE (3), 29 MA (4), 18 GR (5), 4 AL (6) |
  | 8 | 38 TF (0), 35 GC (1), 11 CA (3), 51 CE (5), 52 ML (6) |

- Los bloques de las columnas salen de `meta.bmaps.blocks` del run: o trae `Izquierda`, `Derecha` y `Otros` (con
  todos los demás partidos simulados), o no existe (`web_blocks` de `mtpy/lib/utils.py`). Si no existe, cada partido
  con opciones lleva su columna y no hay «Otros».
- El orden del hemiciclo sale del `block` de cada partido en el catálogo (Izquierda, Separatista, Regionalista,
  Derecha), no de `bmaps`.
- La web ya lee los CSV del catálogo en `data/` al servir (`mtpy/lib/webapi/checks.py`) y la imagen los lleva.
  Añadir columnas al CSV no afecta a la copia del catálogo en la base: `save_catalogues` se queda sólo con las
  columnas que conoce el modelo, así que no hace falta ningún ALTER.
- Peso: la página de `es` lleva las 52 fichas. Se mide en gzip en la comprobación final y se anota (la página del
  evento pesa unos 84 KB).

## Pruebas

- Funciones puras: el estado en los bordes exactos (99%, 98,99%, 75%, 74,99%, 25%, 24,99%); la etiqueta entre 25%
  y 98%; la opacidad en los extremos de cada tramo; el orden de las opciones de una casilla; la frase de la ficha
  y la del partido; los recuentos por estado de un ámbito, una circunscripción y un partido.
- Mapa: cada circunscripción en su celda con su tamaño; un ámbito sin posiciones en rejilla seguida; la elegida
  marcada; los selectores con todos los partidos con opciones y los recuentos del ámbito.
- Hemiciclo: tantas plazas como opciones, sin repetir posición; los partidos en el orden del hemiciclo de la web;
  dentro de cada partido ninguna opción menos probable queda en una fila más interior que otra más probable; con
  pocas opciones (un ámbito autonómico) sigue saliendo un hemiciclo.
- Columnas: los partidos de Izquierda y Derecha con su columna, el resto en «Otros»; el caso sin bloques.
- Página: las tres pestañas pintadas; la ficha de la circunscripción de la dirección marcada y las demás
  presentes y ocultas; ninguna de las palabras retiradas en los textos de la página (los nombres de circunscripción
  no cuentan) en las tres;
  `assert_public_wording`; los 404 y el 400 de siempre; el enlace del evento con su texto nuevo; el método.
- Navegador, con un run real, a 360, 390, 768, 1024 y 1280 px: el borde discontinuo se distingue en el mapa y en
  el hemiciclo; nada se desborda; los selectores, el conmutador y el cambio de ficha al tocar una casilla
  funcionan sin recargar y también con JavaScript desactivado; las cuatro imágenes se descargan enteras, con los
  bordes discontinuos, sin selectores y con la línea del partido y el estado elegidos; un ámbito autonómico con
  varias circunscripciones se ve bien sin posiciones de mapa.

## Decisiones mías (para que Luis las revise)

1. El nombre de la unidad, «opción de escaño», y la frase del pie que explica por qué suman más que los escaños.
2. La opacidad por tramos (el porcentaje la mueve dentro de cada estado) y no como porcentaje literal: un probable
   al 90% sería casi igual que un seguro.
3. Fuera la tira de cifras de la cabecera y la lista de circunscripciones: los recuentos viven en el selector de
   estado y en el centro del hemiciclo, y el mapa hace de lista.
4. El selector de partido lleva los partidos con alguna opción dibujada (hoy, los 13 del run).
5. La colocación y las abreviaturas de las provincias (la de la maqueta).
6. Partidos se abre en Hemiciclo; el orden de las columnas es el de los bloques de la web (Izquierda, Derecha,
   Otros); 7 círculos por fila.
7. En el hemiciclo, los partidos en el orden del hemiciclo de la web y los estados de dentro afuera; en el centro,
   los tres recuentos.
8. Se conservan la ficha del partido y «Los más reñidos», sin el reparto central.
9. El desplegable del móvil, por nombre.
10. Los selectores y el conmutador sin JavaScript, con recuentos fijos.
11. Los ámbitos autonómicos, sin posiciones de mapa por ahora.
12. Todos los textos, incluida la frase del pie, que en la maqueta decía «optan dos o tres partidos» y aquí dice
    «a un mismo escaño pueden optar varios partidos» (y no sale si las opciones no superan a los escaños).
13. La ficha cambia en el sitio al tocar una casilla (todas las fichas van en la página), en vez de recargar con un
    enlace como en la primera versión.
14. La imagen descargada recoge lo que está a la vista, sin selectores y diciendo el partido y el estado elegidos.
15. En los ámbitos autonómicos las casillas llevan el nombre completo y son más anchas.

## Lo que queda al constructor

Los nombres de funciones y clases; si el hemiciclo es HTML posicionado o SVG en línea (lo que mejor conserve el
borde discontinuo en la imagen descargada y en el móvil); el CSS exacto a partir de la maqueta; cómo se resuelve
el tamaño de los círculos del hemiciclo en pantallas estrechas; qué se borra del código de la primera versión y
qué se adapta; el reparto del trabajo en tareas.
