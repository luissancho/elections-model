# Web: fechas clave de una elección («Convocatoria», «Campaña»…) en las series temporales

Cambio pequeño. Luis lo pidió el 2026-11-23 («en todas las series temporales (promedios y evoluciones) dos líneas
verticales marcando la fecha de convocatoria de elecciones y la de comienzo de campaña») y, en la conversación,
decidió que las fechas se guarden por elección, con sitio para otras fechas clave (el 11-M en 2004), propuso
`data/params.json`, fijó que la convocatoria es el día anterior a la publicación del decreto y pidió que la tabla
de la evolución compare con esas fechas. Lo que no viene de él está en «Decisiones mías, por revisar».
Plan: `docs/superpowers/plans/2026-11-23-web-marcas-convocatoria-campana.md`.

## Intención

Que quien mira un promedio o una evolución vea de un vistazo dónde empezó la carrera (la convocatoria), dónde la
campaña y qué otros días pesaron, y pueda leer el movimiento de las curvas frente a ellos. (El para qué es lectura
mía: Luis pidió las líneas, no lo dijo.)

## Hechos de partida (2026-11-23)

- El promedio se dibuja con `seriesOption` (`web/dist/js/charts/series.js`) en cuatro gráficos: el de la página
  del ámbito, su «casa frente al promedio», el del detalle de una elección pasada y el del detalle de un sondeo.
  Ya lleva líneas discontinuas: «Estimación» y «Elección», con la etiqueta arriba, y en la vista de bloques
  «Entra UP», «Entra SALF», con la etiqueta escrita a lo largo de la línea (`marks`).
- Esas entradas las calcula el modelo (el primer sondeo de cada partido nacido en el ciclo) y llegan con cada run
  (`series.first_polls`).
- La evolución se dibuja con `evolutionOption` (`web/dist/js/charts/evolution.js`), sin líneas de marca. Su
  imagen (`evolutionImageOption`) se queda sólo con las series de los nombres mostrados.
- La tarjeta «Evolución de las actualizaciones» se pinta siempre, en la sección `#evolucion` de la página del
  ámbito, que comparte con «Una encuestadora frente al promedio». Madrid y Cataluña, sin convocar, la enseñan
  con un solo punto (todos sus runs tienen el mismo `as_of`).
- Su tabla («Ver los datos», `web/dist/js/evolution-changes.js`) da la cifra del último día y el cambio frente a
  tres días, cada columna con su fecha: el anterior, una semana atrás y el primero de la serie.
- Las imágenes descargables salen de esas mismas opciones.
- No hay fecha de convocatoria ni de campaña guardada: `elections.events` sólo tiene el día de la elección.
- `data/params.json` ya va por ámbito y elección (`{scope: {fecha: entrada}}`, 112 elecciones, las próximas
  incluidas). La web lo lee con caché (`webapi.checks.event_params`). La acción `params` del job `load` reescribe
  `parties` y `bmaps` y conserva las demás claves (`merge_event_params`), y `format_params` escribe un
  diccionario desconocido con una clave por línea.
- Plazos legales desde la reforma de 1994: el decreto de convocatoria se firma 55 días antes de la elección y se
  publica al día siguiente, y la campaña empieza 16 días antes. No valen para las dos repeticiones (26-J de 2016
  y 10-N de 2019, ésta con 47 días y campaña de 8) ni para las seis generales anteriores (1977 a 1993). Las
  autonómicas del fichero empiezan en 2009.

## Enfoques

1. **Una clave `dates` en la entrada de cada elección de `data/params.json`** (elegido; propuesto por Luis). El
   fichero ya tiene la clave de la tabla (ámbito y elección) y la web ya lo lee: ni fichero ni lector nuevos.
2. Un CSV propio en `data/`. Lo mismo con un fichero y un lector más.
3. Una tabla `elections.events_dates` en la base. Pide crear la tabla, un modelo, una carga y publicarla en el
   bundle para que llegue a la web; el modelo no necesita hoy estas fechas en SQL.
4. Sin datos: «Convocatoria» en el primer run publicado y «Campaña» 16 días antes (la primera idea de Luis).
   Descartado al verlo sobre lo publicado: Madrid y Cataluña tienen runs sin estar convocadas y las elecciones
   pasadas no tienen runs.

## Diseño

- **Datos.** Cada elección puede llevar en `params.json`:

  ```json
  "dates": {
      "call": "2004-01-19",
      "campaign": "2004-02-27",
      "key": {
          "2004-03-11": "11-M"
      }
  }
  ```

  `call` es el día de la convocatoria (el anterior a la publicación del decreto) y `campaign` el primero de la
  campaña; `key`, un diccionario de fecha a etiqueta. Las tres son opcionales.
- **La web pinta sólo lo que hay en el fichero.** Una elección sin `dates` (las próximas, mientras no se
  convoquen) no lleva marcas.
- **Lectura.** Una función de `mtpy/lib/pages` convierte las `dates` de la elección en la lista de marcas
  (`[{date, label}]`, por fecha): «Convocatoria», «Campaña» y la etiqueta de cada fecha clave. La página del
  ámbito, la del sondeo y la de una elección pasada la pasan en sus datos iniciales. Nada que republicar.
- **Dónde:** los cuatro gráficos del promedio y la evolución; con ellos, sus imágenes descargables.
- **Cómo se ven:** la misma línea discontinua gris de las demás marcas, con la etiqueta escrita a lo largo de la
  línea, dentro del gráfico, como «Entra SALF». Una marca sólo se dibuja si su día cae dentro del eje.
- En el promedio se suman a `marks` en las dos vistas (partidos y bloques), junto a las entradas de partidos, que
  siguen saliendo del run y sólo en Bloques. `evolutionOption` gana `marks` y su imagen las conserva.
- **Tarjeta de la evolución, sólo con la elección convocada.** Si la elección del run no tiene `dates.call`, la
  tarjeta no se pinta. La sección `#evolucion` y su entrada del índice siguen mientras quede en ella «Una
  encuestadora frente al promedio»; sin ninguna de las dos tarjetas, desaparecen.
- **Tabla de la evolución.** Tras la cifra del último día, las columnas de cambio pasan a ser, por este orden:
  frente al día anterior, frente al inicio de la campaña y frente a la convocatoria. Las columnas «una semana
  atrás» y «primer día de la serie» desaparecen.
  - La referencia de una fecha es el último día con run en esa fecha o antes.
  - La columna de una fecha sólo sale cuando el último día es posterior a ella: antes de que empiece la campaña
    quedan dos columnas.
  - Cada columna sigue encabezada por su fecha («vs 13-nov»), con lo que es en el `title`.
  - El texto de «Cómo se lee» de la tarjeta se reescribe con las columnas nuevas.
- **Relleno inicial**, con un guion de un solo uso que escribe el fichero con `write_params`:
  - Elecciones celebradas desde 1994: `call` 55 días antes y `campaign` 16 días antes.
  - 26-J de 2016: `call` 2016-05-03. 10-N de 2019: `call` 2019-09-24, `campaign` 2019-11-01.
  - 29-N: `call` 2026-10-05, `campaign` 2026-11-13.
  - 14-M de 2004: `key` con el 11-M.
  - Generales de 1977 a 1993: sin `dates` hasta comprobar cada fecha.

## Decisiones mías, por revisar

- La forma de `dates` (`call`, `campaign`, `key`) y sus nombres.
- Las fechas del relleno salen del plazo legal, sin comprobar elección por elección.
- En las dos repeticiones el decreto se firmó y se publicó el mismo día (disolución automática): pongo ese día
  como convocatoria, no el anterior. A comprobar al construir.
- En la tabla, si dos columnas caen en la misma fecha (el día siguiente a la convocatoria o al inicio de la
  campaña), queda una sola.
- Si la serie de runs empieza después de una fecha, su columna no sale.
- Las cabeceras de la tabla siguen siendo fechas, como Luis decidió el 2026-11-04, y no «vs campaña».
- Sin convocatoria desaparece la tarjeta entera de la evolución, no sólo la tabla.
- El gráfico de la casa de la página de un sondeo gana también, en Bloques, las entradas de partidos que ya
  llevan los otros tres gráficos del promedio: las marcas pasan a salir de un solo sitio (`seriesViews`).
- Los textos de «Cómo se lee» sólo nombran las líneas de las fechas clave en las páginas que las tienen.
- Las entradas de partidos no pasan al fichero: las sigue calculando el modelo.
- El modelo recibe `dates` con el resto de la entrada (`get_event_params`) y no la usa.
- Fuera: el ejemplo vivo de `/metodo/promedio`, las tarjetas con chispa de los partidos y las imágenes de enlace.

## Pruebas

- Python: la función de las marcas (las tres claves, el orden, una elección sin `dates`), los datos iniciales de
  las tres páginas, la sección de la evolución con y sin `dates.call`, y el fichero (cada fecha de `dates` es una fecha válida; `call` antes que `campaign` y ésta
  antes que la elección; `format_params` lo deja como está; `load params` conserva `dates`).
- JS (`tests/test_js_charts.py`): las marcas en `seriesOption` junto a «Estimación», «Elección» y las entradas;
  `marks` en `evolutionOption` y en su imagen; nada fuera del eje; la tabla de cambios con tres columnas, con
  dos antes de la campaña y con una cuando dos fechas coinciden.
- En el navegador, con los scripts de comprobación: `/es` a 3 meses y con todo el ciclo, la elección de 2004 con
  el 11-M, en escritorio y en móvil, que las marcas del final no se pisen.

## Cambios tras la implementación (2026-11-23)

Lo que quedó distinto de lo escrito arriba, y por qué.

- **Dónde va cada etiqueta.** El diseño las ponía todas a lo largo de su línea, arriba, como «Entra SALF». En
  el móvil (390 px) con todo el ciclo, que es la vista con la que abre la página, las líneas de la convocatoria y
  de la campaña quedan a 9 px y las dos etiquetas se escribían una encima de la otra; en la evolución, la
  etiqueta caía sobre las líneas de PP y PSOE. Queda así:
  - En los gráficos del promedio, las fechas clave escriben su etiqueta por turnos en los dos extremos de su
    línea (`KEY_MARK_ENDS`): la convocatoria arriba, la campaña abajo, una tercera fecha arriba. Las entradas de
    partidos siguen arriba.
  - En la evolución, las etiquetas van en horizontal por encima del gráfico, donde no pasa ninguna línea de
    partido, cada una hacia dentro: en la primera mitad de los días el texto empieza en su línea y en la segunda
    acaba en ella.
- **La marca lleva su tipo.** `pages.key_dates` da `{date, label, kind}` (`call`, `campaign`, `key`): la tabla de
  la evolución necesita saber cuál es la convocatoria y cuál la campaña. `pages.run_key_dates` da las de la
  elección de un run.
- **El «Cómo se lee».** La frase de las líneas discontinuas va también en «Una encuestadora frente al promedio»,
  que las dibuja, y en el detalle de una elección pasada sólo cuando su serie está publicada.
- **El fichero no se reescribió entero.** `params.json` tiene dos listas de la entrada del 29-N partidas a mano
  (`event` y `polls`), que `format_params` volvería a unir. El relleno añadió `dates` sin tocar eso, y el test
  comprueba cómo escribe `format_params` las fechas, no que el fichero entero coincida con su salida.
- **Las dos repeticiones**, comprobadas en el BOE: el decreto se firmó y se publicó el mismo día (3 de mayo de
  2016 y 24 de septiembre de 2019), y ese día es su `call`.
- **Lo que no se pudo ver en el navegador.** La elección de 2004 no tiene la serie del promedio publicada, así
  que su página no tiene gráfico y el 11-M no se pinta hasta que la tenga; una tercera fecha clave se probó sobre
  la página de 2023. La comprobación se hizo sobre las páginas publicadas, con las fechas clave añadidas a sus
  datos y el JavaScript de la rama: los gráficos, la tabla y las imágenes descargadas se vieron con datos reales
  en escritorio y en móvil; los textos de las plantillas, sólo por sus tests.
