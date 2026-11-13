# «Escaños que bailan» dentro de la página del evento — diseño

Fecha: 2026-11-13. Sustituye a la página aparte `/{scope}/circunscripciones` construida hoy
(`docs/superpowers/specs/2026-11-13-escanos-que-bailan-mapas-design.md`, en adelante «la página aparte»). Bocetos con
el run 20261113-094116, hechos con las piezas de verdad: `mock-lado-1280.png` (elegido) y `mock-apilado-1280.png`, en
la carpeta temporal de la sesión.

## Intención

Luis quiere que todo esté en una sola página, la del evento (`/{scope}`), con su enlace en el índice de la cabecera, y
que lo que se traiga sea sólo lo que informa: el mapa con la ficha de la circunscripción y las columnas de partidos.

## Decisiones de Luis (2026-11-13)

1. Una sola página: «Escaños que bailan» pasa a ser una tarjeta de la página del evento. La página aparte desaparece.
2. La misma tarjeta, con el selector **Provincias / Partidos**.
3. **Provincias**: el mapa y la ficha de la circunscripción, que se elige desde el mapa, en la misma tarjeta. En
   escritorio, **lado a lado** (mapa a la izquierda, ficha a la derecha).
4. **Partidos**: sólo las columnas. Sin hemiciclo y sin el listado por partido.
5. En el índice de la cabecera, la sección de hoy «Escaños» (Hemiciclo, Resultados, Calculadora de mayorías) pasa a
   llamarse **«Resultados»**, y la nueva se llama **«Escaños»** e incluye lo que hoy es «Territorio».
6. **Los anclajes se renombran** con las secciones, «da igual las consecuencias»: `#resultados` y `#escanos`;
   `#territorio` deja de existir.

## Qué cambia en la página del evento

- **Índice**: Promedio · Resultados · Escaños · Evolución · Tabla.
- **Orden**: la sección «Escaños» va detrás de «Resultados» y antes de «Evolución». Lleva dos tarjetas: «Escaños que
  bailan» y, debajo, «Resultados por territorio», que no cambia salvo que pierde su enlace «Escaño a escaño en…».
- **Anclajes**: «Resultados» es `#resultados` y «Escaños» es `#escanos`. Todo lo de dentro de la web que apuntaba a
  los de antes se cambia para que siga cayendo en su sitio: la portada y el método (a `#resultados`, donde están el
  Hemiciclo y la tabla), los enlaces que comparte la calculadora (`#resultados&coalition=…`), la tabla y el formulario
  del territorio (`seccion=escanos`), las redirecciones de las rutas viejas de escaños y la barra de las subpáginas.
- **Consecuencias aceptadas** para los enlaces ya compartidos fuera: uno a `#escanos` cae ahora en la sección nueva
  (el mapa) y no en el Hemiciclo; uno de la calculadora (`#escanos&coalition=PP,VOX`) sigue marcando su coalición pero
  cae en la sección nueva; uno a `#territorio` abre la página arriba del todo. No se añade ningún apaño para ellos.

## La tarjeta «Escaños que bailan»

Título «Escaños que bailan» y la entradilla de la página aparte («Qué escaños tiene seguros cada partido, cuáles son
probables y cuáles están en juego, según las 10.000 simulaciones del modelo.», con el `n_sim` real). Debajo, el
selector Provincias / Partidos (se abre en Provincias; dice «Circunscripciones» donde alguna no es una provincia, como
hoy).

**Provincias.** Lo que hoy lleva la tarjeta del mapa (selector de partido, selector de estado con sus recuentos,
mapa, leyenda y pie de las opciones) más la ficha de la circunscripción elegida, dentro de la misma tarjeta.

- Desde 1.200 px de ventana: el mapa a la izquierda (hasta 760 px) y la ficha a la derecha, separadas por una raya. En
  esa disposición las filas de la ficha no llevan el resumen de la derecha («12 seguros · 3 probables…»): no cabe, y
  las pastillas de la cabecera ya dan los totales.
- Por debajo de 1.200 px: la ficha debajo del mapa, con su resumen como hoy (y sin él por debajo de 720 px).
- La ficha: nombre, escaños, pastillas por estado, la frase y una fila por partido con sus círculos y porcentajes. No
  repite la leyenda (la tarjeta ya la lleva) y el «Cómo leerlo» va una vez, al pie de la tarjeta.
- Se abre en la circunscripción con más opciones en juego, como hoy.

**Partidos.** Sólo las columnas, con su entradilla, su eje, su leyenda y su pie, como están hoy (orden de la tabla
«Resultados», «Otros» al final, raya de encima de la columna más alta).

**Sin JavaScript** funciona todo lo anterior: el selector de vista, los dos selectores del mapa y el cambio de ficha
al tocar una casilla van con radios y `:has()`, y sin `:has()` se ven todas las fichas. El bloque `<style>` generado
(una regla por partido y otra por circunscripción) pasa a la cabecera de la página del evento, sólo cuando hay tarjeta.

**Imagen.** Un botón, el de la tarjeta, con lo que esté a la vista: en Provincias, el mapa con la ficha elegida al
lado, sin las filas de selectores y con la línea del partido y el estado elegidos; en Partidos, las columnas. El
nombre del fichero lo dice: `demoscopio-escanos-{scope}-{circunscripción}[-{partido}][-{estado}].png` y
`demoscopio-escanos-{scope}-columnas.png`.

**Runs sin el dato** (los fijados anteriores al 13-11 y los ámbitos de una sola circunscripción): la tarjeta no se
pinta. Si el run tiene tabla de territorio, la sección «Escaños» sale sólo con ella; si tampoco, no hay sección ni
entrada en el índice, como hoy.

## Qué desaparece

- La página aparte, su plantilla, sus pestañas, sus títulos y descripciones y su canónica.
- El hemiciclo de las opciones (su cálculo, su dibujo, sus estilos y lo que se añadió a la imagen para él).
- La ficha del partido y su selector (`?partido=`), y `?vista=`.
- Los dos enlaces «Escaños que bailan: seguros, probables y en juego ›» al pie de las tablas y el enlace «Escaño a
  escaño en…» bajo el detalle del territorio: la tarjeta está en la misma página y en el índice.
- Que la dirección, el título y la canónica sigan a la casilla tocada: las circunscripciones dejan de tener dirección
  propia.

## Direcciones antiguas

`/{scope}/circunscripciones` y `/{scope}/circunscripciones/{circunscripción}`, con la query que traigan, responden con
un 301 a la página del evento en la sección «Escaños» (`/{scope}` con su `mode` y su `run`, y `#escanos`). No se
conserva la circunscripción: la tarjeta se abre en la suya por defecto.

## Decisiones mías (para que Luis las revise)

1. A dónde apunta ahora cada enlace interno que usaba los anclajes de antes (arriba).
2. La sección «Escaños» sube: hoy «Territorio» va detrás de «Evolución»; pasa a ir detrás de «Resultados».
3. El punto de corte del lado a lado, 1.200 px (a 1.024 no cabe).
4. En el móvil y en tabletas, con JavaScript, al tocar una casilla la página baja hasta la ficha si no está a la
   vista (la ficha queda debajo del mapa y, si no, parece que no pasa nada). En escritorio no hace falta. Era una de
   las decisiones que quedaron abiertas en la página aparte.
5. El desplegable de circunscripciones del móvil se queda, pero sólo con JavaScript (elige la casilla sin recargar).
   Sin JavaScript no sale: las casillas hacen de selector. Antes enviaba un formulario a la dirección de la
   circunscripción, que ya no existe.
6. La tarjeta del mapa y «Resultados por territorio» eligen circunscripción cada una por su lado. Unirlas es otro
   trabajo.
7. Los nombres de los ficheros de imagen.

## No objetivos

- Tocar el modelo, la parte `seat-odds` o republicar: basta con desplegar la web.
- Cambiar «Resultados por territorio», el Hemiciclo del reparto central o la Calculadora.
- Sincronizar la circunscripción elegida entre las dos tarjetas de la sección.
- Rehacer los selectores del mapa en el móvil (siguen ocupando seis filas; queda pendiente).
- La lista de escaños disputados entre partidos: pide publicar los repartos completos.

## Anti-objetivos

- Que algún enlace de dentro de la web siga apuntando a un anclaje de antes, o caiga en la sección que no es.
- Que la página del evento deje de leerse o de manejarse sin JavaScript.
- Que la tarjeta se pinte, o deje un hueco o un enlace en el índice, en un run sin el dato.
- Que vuelva a aparecer en la tarjeta el reparto central o cualquiera de las palabras retiradas.

## Enfoque

Uno solo con sentido: llevar a la plantilla del evento las piezas ya hechas y probadas (las macros del mapa, la ficha
y las columnas, las funciones puras y sus estilos), pintadas en el servidor, y borrar lo que sobra. La alternativa,
pintarlo en el navegador desde el JSON como los gráficos de ECharts, rompería la lectura sin JavaScript y obligaría a
reescribir lo que ya funciona.

## Peso

La página del evento pesa hoy unos 107 KB en gzip. Las 52 fichas, el mapa y las columnas añaden unos 20 KB y unos
4.000 elementos. Se mide al terminar y se anota.

## Pruebas

- Página del evento: el índice con sus etiquetas y su orden; la sección con sus dos tarjetas; el selector de vista,
  los del mapa y las casillas como radios, con la circunscripción por defecto marcada; todas las fichas presentes; en
  Partidos sólo las columnas; ni hemiciclo de opciones, ni ficha de partido, ni enlaces a la página aparte; un run sin
  el dato sin tarjeta, y uno sin territorio sin sección; las palabras retiradas y la redacción pública.
- Rutas antiguas: 301 con `mode` y `run` desde la portada de la página aparte, desde una circunscripción y con
  `partido`, `vista` o `circunscripcion` en la query.
- Los anclajes: `#resultados` en la sección del Hemiciclo y `#escanos` en la nueva; ningún `#territorio` ni
  `seccion=territorio` en la página ni en sus scripts; cada enlace interno (portada, método, calculadora, tabla y
  formulario del territorio, rutas viejas, barra de las subpáginas) apunta al que le toca.
- Navegador, con el run real, a 360, 390, 768, 1024, 1200, 1280 y 1440 px: lado a lado desde 1.200 y apilado por
  debajo, sin desbordes (Madrid y Barcelona elegidas); el cambio de ficha, de vista y de filtros sin recargar y sin
  JavaScript; la bajada a la ficha en el móvil; las dos imágenes; un ámbito sin posiciones de mapa; el peso.

## Lo que queda al constructor

Dónde viven las piezas (si el módulo de la página aparte se queda como módulo de la tarjeta o se reparte), el ancla y
el `id` de la tarjeta, cómo se resuelve el lado a lado en la copia de la imagen (se dibuja a 1.200 px), qué se borra
de los tests de la página aparte y qué se muda a los de la página del evento, y el reparto del trabajo en tareas.

## Estado al cierre (2026-11-13)

### Lo construido

Rama `escanos-evento`, desde `dev` (`git log --oneline dev..HEAD`):

- El índice pasa a Promedio · Resultados · Escaños · Evolución · Tabla, los anclajes se renombran (`#resultados`,
  `#escanos`) y la sección del territorio sube detrás de Resultados: `d76a1bb`.
- La tarjeta «Escaños que bailan» en la sección Escaños, pintada en el servidor, con el selector Provincias /
  Partidos, el mapa y la ficha lado a lado y las columnas: `fec38fd`.
- Su script y su imagen (el nombre del fichero y la imagen según lo que está a la vista, el desplegable del móvil y
  la bajada a la ficha al tocar una casilla en apilado): `ccf142c`.
- Fuera la página aparte (sus direcciones responden con un 301 a la tarjeta), el hemiciclo de las opciones y la
  ficha de partido: `65f67f7` (sólo cambia de nombre el módulo y sus tests) y `b534d87`.
- Arreglos de la comprobación en el navegador (abajo): `439e1fa` y `dd35986`.
- Este cierre: el commit que lo añade.

### Comprobación en el navegador

Datos reales, sólo leídos: el run 20261113-094116 de `es` (10.000 simulaciones), con todas sus partes de los dos
modos, su historial, el manifest y el catálogo de partidos bajados de la API pública de demoscopio.es (GET sin
credenciales). Se escribieron, tal cual, en el paquete de prueba de los tests, en una carpeta temporal fuera del
repositorio y con el `data/` del worktree, y el arnés de los tests pintó la página del evento: en Pronóstico y en
Hoy, sin la parte `seat-odds`, con el aviso de congelación y con la parte reducida a las ocho provincias andaluzas y
sin posiciones de mapa (66 opciones para 60 escaños: 38, 17 y 11). Las páginas se sirvieron en local como ficheros,
con `/dist` del worktree, la API contestando con los ficheros bajados y la política de seguridad de contenidos de
`deploy/docker/nginx.conf` en cada respuesta. Se miraron con Chrome 157 headless por el protocolo de DevTools a 360,
390, 768, 1024, 1199, 1200, 1280 y 1440 px (emulación de móvil y táctil por debajo de 720), con clics, toques y
teclas de verdad. Sin JavaScript: `setScriptExecutionDisabled`. Las cifras de la tarjeta salen como deben: 390
opciones para 350 escaños (203 seguras, 103 probables y 84 en juego); en Hoy, 390 (204, 102 y 84). Nada se publicó
ni se escribió en S3 ni en la base.

| Criterio | Resultado | Cómo se vio |
|---|---|---|
| El índice dice Promedio · Resultados · Escaños · Evolución · Tabla y cada entrada baja a su sección | pasa | a 1280 y 390 px, con JavaScript y sin él: cada sección queda bajo la cabecera (a 56 px con la cabecera de 57, a 52 en el móvil). `?seccion=escanos` y `?seccion=resultados` caen en la suya y la dirección pierde `seccion` |
| Lado a lado desde 1.200 px, apilado por debajo | pasa | a 1.199 px la ficha va debajo del mapa, con el resumen de cada fila (que se va por debajo de 720); a 1.200, al lado y sin él. Igual con JavaScript y sin él, en `es` y en el ámbito sin posiciones |
| Sin desbordes | pasa | a los ocho anchos, con Madrid y con Barcelona elegidas (Sevilla y Málaga en el ámbito sin posiciones), el ancho de la página es el de la ventana y ningún elemento de la tarjeta pasa de su borde |
| Vista, filtros y ficha sin recargar, con JavaScript y sin él | pasa | a 1280 y 390 px: Partidos enseña las 7 columnas y esconde el mapa; PSOE deja 121 círculos enteros y 269 atenuados; PSOE y En juego, 25 y 365; Cádiz enseña sólo su ficha; la dirección no cambia |
| El resto de la página sigue funcionando | pasa | el Hemiciclo, el promedio, la evolución, la calculadora, el abanico y el gráfico de una encuestadora se dibujan a 1280 y 390 px; las tablas tienen sus filas (13 partidos, 3 bloques, el territorio y su detalle, 15 encuestas). La calculadora escribe `#resultados&coalition=…` y un enlace así abre en Resultados con sus partidos marcados, como antes con `#escanos`. Los enlaces del territorio llevan `seccion=escanos#escanos` y su formulario va a `/es#escanos`. Sin JavaScript, la página de `dev` y la de la rama tienen las mismas tablas con las mismas filas: sólo cambian los anclajes. La consola, sin errores ni avisos (salvo los de html2canvas al hacer una imagen) |
| Al tocar una casilla del norte, la página baja hasta la ficha; con el teclado, no | pasa tras un arreglo | Lugo, Madrid y Barcelona tocadas al pie de la ventana a 360, 390, 768, 1024 y 1199 px: la página baja hasta que la ficha se ve; una ficha más alta que la ventana queda bajo la cabecera (16 px por debajo en el móvil, 11 con el aviso de congelación desde 720). A 1280 no se mueve. La flecha pasa de Lugo a Asturias sin desplazar la página; el desplegable (360 y 390) elige Sevilla sin desplazarla. En el móvil, la ficha acababa debajo de la barra de pestañas: arreglado (`439e1fa`) |
| Las dos imágenes | pasa | descargadas con el botón y miradas: `demoscopio-escanos-es-barcelona-psoe-en-juego.png` (mapa y ficha lado a lado, sin selectores, con la línea «PSOE · En juego», la de la elección y la firma), `demoscopio-escanos-es-columnas.png` y, desde una ventana de 390 px, `demoscopio-escanos-es-madrid-pp-probables.png`, también lado a lado |
| Tras ir a otra página y volver, la línea y el nombre coinciden con los radios | pasa | VOX y Seguros, otra página y atrás: Chrome devolvió la página de la caché de ida y vuelta, con los radios, la línea «VOX · Seguros» y `…-madrid-vox-seguros.png` |
| Las direcciones viejas acaban en la tarjeta | pasa | con el arnés de los tests y el run real: `/es/circunscripciones` → 301 a `/es#escanos`; `/es/circunscripciones/cadiz?mode=nowcast` → `/es?mode=nowcast#escanos`; con `run`, `partido`, `vista` o `circunscripcion` en la query, igual, con `mode` y `run` y nada más. En el navegador, esas direcciones de destino caen en la sección Escaños |
| Un run sin la parte `seat-odds` | pasa | sin tarjeta, sin ninguna regla suya en la cabecera, con «Escaños» en el índice y sólo «Resultados por territorio» en la sección; el Hemiciclo y la tabla de encuestas funcionan y la consola está limpia |

### Peso

HTML de la página del evento `/es` con el run real, en Pronóstico (KB de 1.000 bytes, gzip de nivel 6, elementos
contados por el navegador sin JavaScript). «Antes» es el código de `dev` con los mismos datos.

| | Sin comprimir | gzip | Elementos |
|---|---|---|---|
| Antes | 1.035,0 KB | 107,2 KB | 3.380 |
| Después | 1.298,9 KB | 126,3 KB | 6.995 |
| Diferencia | +263,8 KB | +19,0 KB | +3.615 |

En Hoy, igual: de 106,4 a 125,6 KB en gzip. La tarjeta tiene 3.620 elementos. Este documento calculaba unos 20 KB
y unos 4.000 elementos; queda por debajo del tope de 30 KB que se puso el plan. Sin la parte, la página pesa
106,6 KB en gzip.

### Arreglos de la comprobación

- `439e1fa`: en el móvil, la barra de pestañas está pegada abajo y la ficha a la que baja la página al tocar una
  casilla acababa debajo de ella (en Barcelona a 360 px, la última fila y los porcentajes de FA). La ficha guarda
  ahora un margen abajo del alto de la barra, como guarda arriba el de la cabecera.
- `dd35986`: sólo comentarios. Decían que en el móvil el aviso de congelación no se queda pegado arriba; sí se queda,
  encima de la cabecera.

### Lo que difiere de este documento

Decisiones del plan, tal como quedaron:

1. La tarjeta se llama `seat-odds` por dentro y vive bajo una sola clave del contexto, `seat_odds`.
2. El módulo de la página aparte pasa a llamarse `seat_odds.py`, y sus tests, `test_pages_seat_odds.py`.
3. La bajada a la ficha al tocar una casilla sólo ocurre con puntero o dedo (no con el teclado), sólo en apilado y
   sólo si la ficha no está a la vista: si asoma, la página no se mueve.
4. La imagen de Provincias se llama `demoscopio-escanos-{scope}-{circunscripción}[-{partido}][-{estado}].png`.

Decidido durante la construcción: la línea de la imagen con la elección, la estimación y la fecha va una vez, encima
de las dos vistas, para que la lleve también la imagen de las columnas.

Diferencias visibles con la maqueta elegida (a 1280 px) que este documento no recoge:

- «Toca una provincia para ver su detalle.» no cierra la entradilla: va al final de una línea que la maqueta no
  tiene, bajo el selector de vista y ya de la página aparte, «Cada círculo es una opción de escaño: sólido si es
  seguro, más claro si es probable y con el borde discontinuo si está en juego.».
- La leyenda y el pie de las opciones van bajo el mapa, a su ancho, y no a todo el ancho de la tarjeta bajo el mapa y
  la ficha; la raya vertical entre los dos acaba donde acaba la ficha y no donde acaba el mapa.

### Lo que no se pudo comprobar

- La web detrás de nginx: las páginas se sirvieron como ficheros, aunque con la misma política de seguridad de
  contenidos. Tampoco otros navegadores (Safari, Firefox) ni móviles de verdad: sólo Chrome headless con emulación.
- Los 301 en el navegador: con ficheros no hay servidor; se vieron con el arnés de los tests.
- La salida sin `:has()`: sólo la cubren los tests.
- Un ámbito autonómico con su parte publicada: ninguno la tiene aún; se usó la de `es` reducida a Andalucía.

### Lo que queda (Luis)

- Los selectores del mapa en el móvil siguen ocupando seis filas (cuatro de partidos y dos de estados), como en la
  página aparte.
- La página del evento responde con un error 500 si un bloque de `meta.bmaps.blocks` no tiene fila en el `summary`
  (`promedios.html`, la leyenda de la tabla de bloques). Ya pasa en `dev`: se reprodujo en los dos quitando la fila
  de Derecha. El run real lleva las tres filas (Izquierda, Derecha y Otros) en los dos modos, así que hoy no puede
  pasar.
- En el móvil, el aviso de congelación se queda pegado arriba y tapa la cabecera (ya pasaba antes de esta rama). Con
  un aviso de una línea (48 px) la ficha queda bajo él; con uno de dos líneas a 360 px (70 px) la ficha empieza 2 px
  por debajo de su borde y su nombre se sigue leyendo.
- Las dos diferencias con la maqueta, arriba, si hay que cambiarlas.
- Después de desplegar, una comprobación en la web de verdad: abrir `/es#escanos` en un móvil, tocar una provincia y
  descargar la imagen de la tarjeta. Nadie pudo probarlo detrás de nginx, en otros navegadores ni en móviles de
  verdad.
- Fusionar en `dev`, push e imagen nueva de web. No hay que republicar nada.

### Pendientes menores

- Los comentarios de `_seat_odds.html` siguen llamando «card» a la ficha de la circunscripción.
- `docs/web/contrato.md` conserva, encima de la enmienda que las corrige, las frases viejas sobre `/{scope}/escanos`.
- El test de la barra del evento de las subpáginas vive en `tests/test_pages_seat_odds.py`.
- Las redirecciones de las direcciones viejas conservan un `seccion` que traiga la query (`?seccion=tabla` cae en
  `#tabla`).
- `Circunscripciones.active = None` ya no se usa.
