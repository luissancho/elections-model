# Web, fase 7c: rediseño visual, bloque central, tabla de encuestas y tabla por territorio

Fecha: 2026-10-14. Estado: aprobada por Luis en el brainstorming del 14 de octubre (maqueta en
`.superpowers/brainstorm/65129-1791976803/content/fase7c-maqueta-v3.html`, "me encanta el diseño").
Sigue a la fase 7b (`docs/superpowers/specs/2026-10-13-web-fase-7-diseno-design.md`, apartado "Ajustes tras el
recorrido de Luis"). Esta spec no repite lo que ya está en producción: solo describe lo que cambia.

## Intención

Tras recorrer la web publicada, Luis quiere que se parezca más a un sitio de datos moderno (la referencia es
Objetivo176, objetivo176.es): fondo gris muy claro en vez del salmón, tarjetas blancas con sombra suave, el
contenedor a todo el ancho de la pantalla para que quepa más información, y dos tablas grandes y filtrables como
las suyas ("Todas las encuestas" en /encuestas y "Resultados por territorio" en /generales). Además, el bloque
central de Promedios se simplifica a tres tarjetas (hemiciclo, partidos, bloques) sin rangos, y hay varias
correcciones pequeñas de texto y de controles.

## Objetivos

1. Estilo visual nuevo en todo el sitio: fondo `#f4f5f8`, tarjetas blancas, controles segmentados redondeados,
   contenedor fluido.
2. Pie: "Publicación: {run_id} ({run_at}) · Último sondeo: {date_last}"; cabecera del evento sin la frase de modo.
3. Promedios: sin el titular "Si hoy hubiera elecciones, PP ganaría…"; selector de partidos del "Promedio de
   sondeos" en la leyenda del gráfico (como en "Evolución de las publicaciones"); "Una encuestadora frente al
   promedio" con los mismos filtros que el promedio más el de encuestadora.
4. Promedios: las tarjetas Estimación, Voto, Escaños, Hemiciclo y Mayoría absoluta se sustituyen por tres:
   "Hemiciclo" (grande), "Resultados por partido" y "Resultados por bloque".
5. Promedios: "Sondeos del ciclo" se sustituye por "Todas las encuestas", una tabla filtrable, ordenable y
   paginada en el navegador, al estilo de Objetivo176.
6. Escaños: la tabla de "Circunscripciones" se sustituye por "Resultados por territorio", con conmutadores
   CCAA/Provincia y % Voto/Escaños, celdas teñidas y ganador marcado, al estilo de Objetivo176.

## No objetivos

- Ningún cambio en el modelo ni en el cálculo: la publicación solo añade dos campos a `districts.regions` (D6).
- Ninguna página nueva, ninguna ruta nueva ni cambios en la API. Nada de nginx ni Docker.
- Los textos del método, el glosario y los "Cómo leerlo" existentes no se reescriben (solo los de las tarjetas que
  cambian).
- Ni modo oscuro ni cambio de tipografías (siguen Source Serif 4 y Source Sans 3) ni de color de acento.
- La marca (cabecera, favicon, canonical) es trabajo aparte de Luis, en curso en su árbol; esta fase no lo toca.

## Anti-objetivos (lo que haría fracasar la fase aunque funcionara)

- Perder la lectura sin JavaScript: las tablas nuevas tienen que salir del servidor con contenido (las filas
  iniciales), y los controles solo se muestran cuando hay JS. Nada de `script` inline (CSP `default-src 'self'`).
- Páginas mucho más pesadas: los sondeos del ciclo ya viajan en la isla `#initial-data` para el gráfico; la tabla
  los reutiliza. Escaños añade `districts` y `scenario` a la isla (unos 80 KB en `es`), y nada más.
- Tablas inservibles en el móvil: desplazamiento horizontal con la primera columna fija y la pista
  "Desliza para ver más →"; el bloque de tres tarjetas se apila en una columna.
- Perder contraste AA en textos apagados y en las celdas teñidas (el texto de una celda teñida es siempre el color
  de texto normal sobre un tinte de ≤ 18 % de alfa).
- Romper los enlaces guardados: el hash `#ventana=&vista=&sel=` del promedio sigue significando lo mismo.

## Restricciones

- Patrón de siempre: controlador `Page` + Jinja2 (`StrictUndefined`, autoescape) + isla JSON `#initial-data` +
  módulos JS que solo dibujan (ECharts 5.6 vendido). Nada de build step. Módulos puros con tests de paridad en
  node (`tests/js/`) como `range.js`, `blocks.js`, `errors.js`, `polls-tooltip.js`.
- Textos en español, "encuestadora/s" y "medio/s", siglas de partido siempre.
- `python -m pytest -m "not integration" -q` en verde (697 al empezar) sin avisos nuevos; los tests de páginas
  (`tests/test_pages.py`) fijan los textos y el marcado que cambian.
- Un ancho de página fluido no debe romper los gráficos ECharts: siguen con `resize` en `window.resize`.

## Diseño

### D1. Estilo visual y contenedor fluido

Tokens de `web/dist/css/site.css` (valores de la maqueta aprobada):

| Token | Valor |
| --- | --- |
| `--color-bg` | `#f4f5f8` |
| `--color-card` | `#fff` |
| `--color-text` | `#1a1c22` |
| `--color-muted` | `#5a5f72` (AA sobre fondo y tarjeta) |
| `--color-faint` | el gris más claro que pase AA (4,5:1) sobre blanco con el tono de `#9298ac` (unos `#6b7186`); solo cabeceras de tabla en mayúsculas y notas pequeñas, nunca texto corrido |
| `--color-line` | `rgba(0,0,0,.08)` |
| `--color-line-strong` | `rgba(0,0,0,.15)` |
| `--color-soft` | `#eceef2` |
| `--color-accent` | `#1d4f86` (se mantiene) |
| `--radius` | `12px` |
| `--shadow` | `0 2px 8px rgba(0,0,0,.06)` |

- `.card`: fondo blanco, borde `--color-line`, radio 12, sombra, relleno 20×22 px. El título de tarjeta lleva una
  barra vertical de 5×20 px con el color de acento a su izquierda (`::before`), como en la maqueta.
- Controles segmentados (`.seg`): fondo `--color-soft`, radio 999 px, opción activa en acento con texto blanco.
  Se usan en todos los grupos de radios de la web (Vista, Ventana, CCAA/Provincia, % Voto/Escaños y los filtros
  de las páginas globales). Los `.pills` de selección múltiple (calculadora de coaliciones, partidos de la
  diana) pasan a chips redondeados con el mismo lenguaje (borde de 1 px, activo con el color del partido, tinta
  `ink_color`).
- Contenedor: `main` deja de tener `max-width`; relleno lateral de 32 px en escritorio y 16 px por debajo de
  720 px; `--band-pad` pasa a ser el mismo relleno. Las rejillas de tarjetas usan
  `repeat(auto-fit, minmax(420px, 1fr))` para que una pantalla ancha gane columnas. Solo la prosa se limita: los
  "Cómo leerlo", el glosario, los textos del método y las tarjetas de texto de la portada tienen
  `max-width: 72ch` (decisión del controlador, para que se lean bien).
- Tablas: cabeceras en mayúsculas pequeñas (`--color-faint`, 11 px, espaciado .06em), filas con línea inferior
  fina, números tabulares, fila al pasar el ratón con fondo `#f6f8fb`.
- Todo el sitio hereda el estilo (Inicio, Histórico Errores, Ranking Encuestadoras, Método, detalle de elección y
  de encuestadora): sus tarjetas, tablas y filtros usan las mismas clases; no se rediseñan sus contenidos.
- El tema de ECharts (`web/dist/js/charts/base.js`) ajusta el fondo transparente, el color de los ejes
  (`--color-line-strong`) y el color de las barras (`#3a6ea5` se mantiene).

### D2. Cabecera del evento y pie

- `_event_bar.html`: desaparece `<p class="mode-text">`. `mode_text` deja de existir en el contexto si ninguna
  otra plantilla lo usa (el constructor lo comprueba y lo retira junto con sus tests).
- `_footer.html`: `Publicación: {{ meta.run_id }} ({{ meta.run_at|date }}) · Último sondeo: {{ meta.date_last|date }}`
  y, si la web está congelada, ` · congelado desde {{ freeze.since|date }}` (decisión del controlador). Se
  retiran el commit, "(sucio)" y "promedio leído el". Ejemplo: `Publicación: 20261014-080707 (14 oct 2026) ·
  Último sondeo: 11 oct 2026`.

### D3. Promedios: titular y bloque central

- Se retira `{{ m.headline(headline_text) }}` de `promedios.html`. `headline_text` de `pages/promedios.py` se
  conserva solo si la portada u otra página lo usa; si no, se retira con sus tests.
- Las tarjetas `headline`, `vote`, `seats`, `hemicycle-card` y `majority-card` (y sus "Cómo leerlo") se
  sustituyen por una rejilla `.hemi-grid` de tres tarjetas, en el mismo sitio de la página que ocupan hoy, es
  decir, debajo de la tarjeta "Promedio de sondeos" (decisión de Luis, 2026-10-14: la página sigue empezando
  por el promedio), en este orden del DOM:
  1. `hemicycle-card`, título "Hemiciclo", entradilla "El reparto central de los escaños si hoy hubiera
     elecciones, dibujado como la cámara." (en modo Elección: "…en la elección, dibujado como la cámara."). El
     gráfico es el hemiciclo continuo actual, tal cual (`charts/hemicycle.js`, medio donut de ECharts con el
     texto "176 escaños para la mayoría" en el hueco; Luis lo prefirió al de puntos de la maqueta), más grande:
     altura mínima 380 px, y ocupa las dos filas de la rejilla (columna izquierda, ~57 % del ancho). Sin pie
     debajo: el hueco ya dice la mayoría (decisión del controlador, para no repetir el 176).
  2. `parties-card`, título "Resultados por partido", entradilla "Voto y escaños previstos de cada partido y su
     probabilidad de quedar primero." Tabla con columnas Partido (punto de color + sigla, `title` con el nombre
     completo), % Voto (`pct`, 1 decimal, coma), Escaños (`seats`, el titular entero, en negrita) y P(primero)
     (`p_first`, porcentaje entero). Todos los partidos de `summary.parties`, en su orden. Sin rangos.
  3. `blocks-card`, título "Resultados por bloque", entradilla "Voto y escaños de cada bloque y su probabilidad
     de mayoría absoluta." Columnas Bloque (punto con el color del bloque, el de `block_color()`: el del primer
     partido del bloque en `bmaps.blocks`, que en `es` da `#1d84ce`, `#ef1c27`, `#4aae4a` y `#ffb232`), % Voto,
     Escaños y P(mayoría) (`p_majority`).
     Los cuatro bloques del catálogo, de `summary.blocks`, en el orden Derecha, Izquierda, Regionalista,
     Separatista; un bloque sin partidos no aparece.
  Las dos tablas salen del servidor (sin JS). En pantallas de menos de 900 px la rejilla es una columna
  (hemiciclo, partidos, bloques).
- "Cómo leerlo" de cada tarjeta (plegados): hemiciclo, "Cada sector es un partido con los escaños del escenario
  central; la línea marca la mayoría absoluta. En cada simulación el reparto cambia: la pestaña Escaños enseña
  cuánto."; partidos, "Primero es la probabilidad de sacar más escaños que nadie, no la de gobernar. Los rangos
  de cada partido están en la pestaña Escaños."; bloques, "La probabilidad mide la fracción de simulaciones en
  que los partidos del bloque suman la mayoría absoluta. Los pactos entre bloques no cuentan aquí: la calculadora
  de coaliciones de la pestaña Escaños los prueba."
- Se retiran de `promedios.js` el dibujo de las barras de voto y escaños y las barras de mayoría; `charts/bars.js`
  se borra si ninguna otra página lo usa (el constructor lo comprueba).

### D4. Promedios: controles de los dos gráficos de series

- "Promedio de sondeos": `#series-view` (Partidos/Bloques) y `#series-window` (Todo el ciclo/1 año/6 meses/3
  meses) siguen, con el estilo `.seg`. Desaparecen los fieldsets `#series-parties` y `#series-blocks`: el
  selector de partidos es la leyenda del gráfico, `legend: {type: 'scroll', top: 0}`, pulsable, como en
  "Evolución de las publicaciones". La selección de la leyenda es el `sel` del hash (`#ventana=&vista=&sel=`),
  en los dos sentidos: al cargar, el hash fija qué series están seleccionadas; al pulsar la leyenda
  (`legendselectchanged`), se reescribe el hash. `blocks.js` (`parseSeriesHash`, `seriesHash`) no cambia de
  formato.
- "Una encuestadora frente al promedio": sus propios `#pollster-view` y `#pollster-window` (`.seg`, mismas
  opciones) delante del selector `#pollster-select`, y su leyenda pasa de solo lectura a pulsable. Su estado es
  independiente del gráfico de arriba, arranca con los mismos valores que el promedio tenía al cargar y no va en
  el hash (decisiones del controlador). Los puntos del gráfico siguen con el tooltip por sondeo.

### D5. Promedios: "Todas las encuestas"

Sustituye a la tarjeta `polls` ("Sondeos del ciclo"). Título "Todas las encuestas"; entradilla "Estimación de
voto (%) de cada sondeo del ciclo, tal como se publicó. Sombreado, el partido más votado. Pulsa una cabecera para
ordenar."

- Datos: `initial.polls` (ya en la isla para el gráfico: `parties` y `polls` con `date`, `pollster`,
  `pollster_id`, `sample_size` y el % de cada partido). `chart_polls` (`pages/promedios.py`) pasa a incluir
  también `sponsor`, `start_date` y `end_date`, que hoy quita (se adapta
  `test_promedios_initial_polls_carry_only_the_chart_columns`). Se retiran `initial.table`, el botón
  `#polls-more` y `nextRows` de `polls-table.js`: toda la tabla se calcula en el navegador.
- Partidos marcados al cargar: todos los del run (como en la maqueta aprobada, "Partidos (13/13)"), también en
  la versión del servidor; `table_parties` deja de limitarse a los `main` (decisión del controlador: el filtro
  permite quitar columnas, y en el móvil la primera columna fija y el desplazamiento horizontal las hacen
  manejables).
- Barra de controles (`.bar`, oculta sin JS): botón "Partidos (n/N)" y botón "Encuestadoras (n/N)", cada uno
  abre un desplegable con chips (los de partido en su color, los de encuestadora neutros) y los botones
  "Seleccionar todos/as" y "Quitar todos/as"; siempre queda al menos uno marcado. "Desde" y "Hasta" son `select`
  de meses (`YYYY-MM`, etiqueta "Ago 2023"), con todos los meses que tienen sondeos; por defecto, el primero y el
  último. A la derecha, "N sondeos · Descargar CSV" (el enlace CSV de hoy).
- Tabla `.et` (`table-layout: fixed`, `colgroup`): Encuestadora (fija a la izquierda al desplazar, en negrita, con
  el medio debajo en pequeño; "—" sin medio), Fecha (`dd/mm/aa`; `title` "Trabajo de campo: inicio – fin" cuando
  hay fechas de campo), Muestra (miles con punto; "—" sin muestra) y una columna por partido seleccionado (punto
  de color encima de la sigla; `title` con el nombre completo). Valores con 1 decimal y coma; "—" cuando el
  sondeo no da dato; la celda del valor máximo de la fila lleva el color del partido al 14 % de alfa y negrita.
- Orden: pulsar una cabecera ordena por esa columna (`aria-sort`, flecha ▾/▴); por defecto Fecha descendente;
  desempate, fecha descendente. Un sondeo sin dato en la columna ordenada va al final.
- Pie de tabla: "Media simple · N sondeos" con la media sin ponderar de los sondeos filtrados (no solo los de la
  página) en cada partido, el máximo sombreado igual.
- Paginación de 15: "‹ Más recientes · 1–15 de N · Más antiguos ›" (con otro orden, "Anteriores/Siguientes");
  los botones se desactivan en los extremos; cualquier cambio de filtro vuelve a la página 1.
- Sin JS: el servidor dibuja las 15 más recientes con el mismo marcado y la fila de media de esas 15; el resto
  no se ve (el CSV sigue teniendo todo).
- El estado de la tabla no va en el hash (decisión del controlador).
- Módulo puro `web/dist/js/polls-table.js` (sustituye al actual, donde viven los módulos puros):
  `monthsOf(polls)`, `filterPolls(polls, {parties, pollsters, from, to})`, `sortPolls(rows, key, asc)`,
  `pageOf(rows, page, size)`, `meanOf(rows, parties)` y `leaderOf(values)`; tests de paridad en node
  (`tests/test_js_*.py`, por subproceso, como los existentes).
- "Cómo leerlo": "Cada fila es un sondeo: la encuestadora y el medio o quien lo encargó, la fecha de
  publicación, la muestra y el voto de cada partido, sin ninguna corrección del modelo. La media simple del pie
  es la de los sondeos filtrados, todos con el mismo peso: no es el promedio del modelo, que pondera por
  calidad, muestra y fecha."

### D6. Escaños: "Resultados por territorio"

Sustituye a la tabla `#district-table` de la tarjeta `districts`, que pasa a titularse "Resultados por
territorio". El selector de circunscripción (`#district-form`) y su tabla de detalle (`#region-table`, con los
intervalos y P(escaño)) se mantienen debajo tal cual (decisión del controlador).

- Controles (`.seg`, ocultos sin JS), a la derecha del título: nivel "CCAA | Provincia" (solo cuando el run
  trae `community` en más de una comunidad; por defecto CCAA si existe, si no Provincia) y modo
  "% Voto | Escaños" (por defecto Escaños).
- Datos en la isla de Escaños: `districts` (filas `region_id, region, name, pct, …, seats`) y `scenario`
  (escaños del escenario central por circunscripción), más `districts.regions` con dos campos nuevos (abajo).
  Escaños de una provincia = escenario central (como la tabla actual, de modo que las filas suman los escaños
  de la circunscripción y el TOTAL suma 350); % Voto = `pct` de `districts`. Una comunidad suma los escaños de sus
  provincias y pondera el % de voto por los `votes` de cada provincia.
- Columnas: territorio (con un punto del color del ganador tras el nombre; empate: un punto por empatado), una
  por partido ordenadas por su total descendente (en Escaños, solo partidos con algún escaño en el escenario; en
  % Voto, todos los del run), y Total (escaños del territorio, o "100 %"). Fila TOTAL al final (escaños del
  escenario en todo el ámbito; en % Voto, el `vote` del run). Pulsar una cabecera ordena (territorio, cada
  partido, total); por defecto, territorio alfabético.
- Celdas: tinte con el color del partido cuya intensidad va de 0 a 18 % de alfa entre el mínimo y el máximo de
  la columna; la celda del ganador en negrita y con un anillo interior fino; "—" cuando vale 0 o no hay dato. En
  Escaños, cada celda conserva el `title` con el rango del 95 % de hoy. En nivel Provincia el nombre enlaza al
  detalle (`?region=`), como ahora; en CCAA no.
- Sin JS: el servidor dibuja la tabla en nivel Provincia y modo Escaños con el mismo marcado (es la tabla actual
  con el estilo nuevo); los controles no se ven.
- Módulo puro `web/dist/js/territory.js`: `territoryRows(districts, scenario, {level, mode})`,
  `groupByCommunity(rows, regions)`, `tintAlpha(value, min, max)`, `winnerOf(row)`, `sortTerritory(rows, key,
  asc)`; tests de paridad en node (incluida la ponderación por votos y un empate de ganador).
- Fila TOTAL en % Voto: el `pct` de `summary.parties` (mismo estimador que `vote` y ya en la isla; `vote` no
  se añade).
- Publicación (enmienda al contrato, `docs/web/contrato.md`, apartado `districts`): `districts.regions` pasa a
  llevar `community` (`{code, name}` del catálogo `data/es-districts.csv` (`reg_code`) + `data/es-regions.csv`;
  `null` cuando la comunidad es el propio ámbito, es decir, cuando `reg_code` coincide con el `ine_code` del
  ámbito en `es-scopes.csv`, como en los ámbitos autonómicos) y `votes` (votos totales de la
  elección anterior en la circunscripción, de `sim.reg_totals['votes']`, entero). Es compatible hacia atrás:
  los runs antiguos no los tienen y la página oculta el nivel CCAA. `CONTRACT` no cambia.
- "Cómo leerlo": el texto actual de la tarjeta, más una frase: "En CCAA los escaños son la suma de sus
  provincias y el voto, la media ponderada por los votos de la elección anterior."

### D7. Lo que no cambia

Rutas, API, publicación (salvo D6), textos del método, la portada (solo el estilo), las páginas globales (solo el
estilo y los `.seg`), la calculadora de coaliciones, histogramas, abanico, evoluciones, descargas, el titular de
Escaños (`escanos_headline_text`) y los subtítulos de modo de las tarjetas ("· estimación a …").

## Lo que decide el constructor

- Organización del CSS (un solo `site.css`), nombres exactos de clases nuevas (`.seg`, `.bar`, `.et`, `.geo`,
  `.hemi-grid` son los de la maqueta y se pueden usar tal cual) y los valores de espaciado.
- Cómo se implementan los desplegables de Partidos/Encuestadoras (un `<details>` por botón o un panel propio),
  siempre accesibles con teclado y cerrables al pulsar fuera.
- Si `charts/bars.js` y `headline_text` se retiran o se quedan (según otros usos).
- Qué tests de página se adaptan y cuáles son nuevos; los de paridad en node son obligatorios para los módulos
  puros nuevos.

## Pruebas que fijan esta fase

- `tests/test_pages.py`: pie con "Publicación: … (…) · Último sondeo: …" y sin "commit"; `_event_bar` sin
  "Si las elecciones fueran hoy"; Promedios sin el titular, con las tres tarjetas en ese orden y sus cuatro
  columnas; sin `#series-parties`; `#pollster-view` y `#pollster-window` presentes; "Todas las encuestas" con 15
  filas y la fila "Media simple"; Escaños con "Resultados por territorio", la tabla de provincias y los controles
  ocultos (`hidden`).
- `tests/js/`: paridad de `polls-table.js` y `territory.js`.
- `tests/test_publish_unit.py`: `districts.regions` con `community` y `votes` en `es` y `community: null` en un
  autonómico.
- `tests/test_css.py` o equivalente: `--color-bg: #f4f5f8`, ausencia de `--max-width` en `main`.

## Estado al cierre de la fase 7c

Cerrada el 2026-10-14 en `dev`, commits `6f97137..a822ef0` (plan `68c8551`; entre medias, dos commits ajenos de Luis,
`80e5d17` y `f548c56`). Suite: 719 tests en verde (697 al empezar), 54 avisos preexistentes. Ejecutada en modo
subagent-driven: ocho tareas con revisión por tarea, revisión final de toda la rama y una ola de correcciones
(`591e4a5`, `ca78b33`, `2530399`, `a822ef0`). Los iconos de marca se regeneraron con los colores nuevos a petición de Luis (commit siguiente al cierre). Ajuste posterior de Luis (2026-10-14, `9db413a`): el bloque central pasa a un solo par hemiciclo/tabla a la misma altura, con un selector Partidos/Bloques en la tarjeta del hemiciclo que cambia el hemiciclo (por partido o por bloque) y muestra la tabla correspondiente; sin JS se ve la de partidos. Segundo ajuste (`e8ba327`): en Escaños, los histogramas pequeños por partido se sustituyen por un solo gráfico de crestas (`web/dist/js/kde.js`, densidad gaussiana con el ancho de banda de Scott comprobada contra `scipy.stats.gaussian_kde`; `charts/ridge.js`), una fila por partido sobre el mismo eje de 0 a 200 escaños (o el tamaño de la cámara si es menor), cada curva a su propia escala, con la mediana, el rango y la mayoría marcados, como `Simulator.plot_dist_kde` en el notebook. Tercer ajuste (commit siguiente): Escaños pierde su titular, y las tarjetas "Escaños por partido" y "Bloques" se funden en "Reparto de escaños" con un selector Partidos/Bloques (barra apilada, tabla y crestas de la vista; las crestas de bloques suman los partidos de cada bloque); Cuarto ajuste: las crestas tienen el doble de alto (144 px por fila) y un eje común por vista, de 0 al 60 % de la cámara para los partidos y al 80 % para los bloques (210 y 280 en `es`); `ridgeRows` admite `range` y conserva el eje propio por fila (`seatRange`) cuando no se le pasa.

Hecho, por apartado: D1 (tokens grises, tarjetas con barra de acento, `.seg`, contenedor fluido, prosa a 72ch, tema
de ECharts), D2 (pie "Publicación: … · Último sondeo: …", barra del evento sin la frase de modo, Promedios sin
titular), D3 (tres tarjetas: hemiciclo continuo, "Resultados por partido", "Resultados por bloque"), D4 (leyenda
como selector, controles propios del gráfico de encuestadora), D5 ("Todas las encuestas": `polls-table.js` +
`pages/polls-dom.js`, 15 filas del servidor, media simple sobre todos los sondeos), D6 ("Resultados por
territorio": `territory.js` + `pages/territory-dom.js`, `community` y `votes` en `districts.regions`, enmienda del
contrato). Desviaciones registradas: la media del servidor cubre todos los sondeos (no "esas 15"); la tabla del
servidor de territorios alinea partidos y orden con el primer dibujo del JS (tintes y puntos del ganador solo con
JS); el hemiciclo no se estira a la altura de las tablas (`align-items: start`, 460 px en escritorio).

Pendiente de Luis: recorrido visual en escritorio y móvil; publicar `forecast` de `es` (y de los autonómicos) para
que los runs lleven `community`/`votes` y aparezca el nivel CCAA; pregunta de
modelo: VOX y SALF salen con `pct` nulo en S. C. Tenerife en `districts.rows` del run local (¿listas de candidatos?).

Menores aplazados (sin efecto visible): `publish_mode` trata `export_districts` como caso especial; redondeo
bancario de Python frente a `Math.round` en medias exactas .x5; empate de totales iguales ordenado distinto en
servidor y JS; el botón de paginación desactivado pierde el foco; `markSegments` replica `markPills`;
`--gutter` sin uso; el test del CSS no fija todos los tokens.
