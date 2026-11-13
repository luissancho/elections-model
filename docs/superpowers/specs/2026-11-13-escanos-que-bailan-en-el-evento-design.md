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

## Qué cambia en la página del evento

- **Índice**: Promedio · Resultados · Escaños · Evolución · Tabla.
- **Orden**: la sección «Escaños» va detrás de «Resultados» y antes de «Evolución». Lleva dos tarjetas: «Escaños que
  bailan» y, debajo, «Resultados por territorio», que no cambia salvo que pierde su enlace «Escaño a escaño en…».
- **Los anclajes no cambian**: «Resultados» sigue siendo `#escanos` y «Escaños» es la sección `#territorio`. Hay
  enlaces repartidos por la web y compartidos fuera que los usan (la portada, el método, los de la calculadora con
  `#escanos&coalition=…`, `seccion=territorio`, las redirecciones de las rutas viejas) y así ninguno se rompe.

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
un 301 a la página del evento en la tarjeta (`/{scope}` con su `mode` y su `run`, y el ancla de la tarjeta). No se
conserva la circunscripción: la tarjeta se abre en la suya por defecto.

## Decisiones mías (para que Luis las revise)

1. Los anclajes de las secciones no se renombran (arriba). El coste es que la dirección dice `#territorio` donde el
   índice dice «Escaños», y `#escanos` donde dice «Resultados».
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

- Que algún enlace existente a `#escanos` o `#territorio` deje de caer en su sitio.
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
- Los anclajes `#escanos` y `#territorio` siguen existiendo y en las secciones que les tocan.
- Navegador, con el run real, a 360, 390, 768, 1024, 1200, 1280 y 1440 px: lado a lado desde 1.200 y apilado por
  debajo, sin desbordes (Madrid y Barcelona elegidas); el cambio de ficha, de vista y de filtros sin recargar y sin
  JavaScript; la bajada a la ficha en el móvil; las dos imágenes; un ámbito sin posiciones de mapa; el peso.

## Lo que queda al constructor

Dónde viven las piezas (si el módulo de la página aparte se queda como módulo de la tarjeta o se reparte), el ancla y
el `id` de la tarjeta, cómo se resuelve el lado a lado en la copia de la imagen (se dibuja a 1.200 px), qué se borra
de los tests de la página aparte y qué se muda a los de la página del evento, y el reparto del trabajo en tareas.
