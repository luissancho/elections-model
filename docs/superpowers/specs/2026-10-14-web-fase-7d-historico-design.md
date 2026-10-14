# Web, fase 7d: Histórico Errores de todos los ámbitos y ranking único

Fecha: 2026-10-14. Estado: aprobada por Luis ("Perfecto, adelante") tras la propuesta en el chat. Sigue a la fase 7c
(`docs/superpowers/specs/2026-10-14-web-fase-7c-rediseno-design.md`). Solo describe lo que cambia.

## Intención

Luis quiere que "Histórico Errores" deje de ser la página de un ámbito y pase a ser el histórico de todas las
elecciones (generales y autonómicas), filtrable por ámbito, por elección, por encuestadora y por años, con una
comparación que responda a preguntas como "¿los sondeos de los últimos cinco años fallan menos que los
anteriores?" o "¿el CIS desde 2019 falla más que antes?". Y que "Ranking Encuestadoras" sea un ranking único: hoy
el rating de una encuestadora es el mismo número en todos los ámbitos (se calcula con sus sondeos de todas las
elecciones anteriores a la próxima, y hoy la próxima general y la próxima autonómica ven las mismas elecciones
pasadas); solo cambia qué encuestadoras aparecen. Comprobado el 2026-10-14 contra la API local: 109 encuestadoras
comunes con rating idéntico, 20 solo en las generales.

## Objetivos

1. `/elecciones`: un solo histórico con las elecciones publicadas de todos los ámbitos con análisis, y una barra de
   filtros encadenados: Ámbito, Elección, Encuestadora, Años (desde/hasta) y Medida.
2. Cifras de la selección con su comparación frente al complemento de cada filtro activo.
3. Un gráfico nuevo, "Evolución del error por elección", que sustituye a "Error del sector".
4. El histograma, la diana, la tabla por semanas y la tabla de elecciones existentes trabajan sobre la selección.
5. `/casas`: el ranking es el de la próxima elección general, para todas las encuestadoras puntuadas, sea cual sea
   el ámbito elegido; el filtro de ámbito queda para los efectos de encuestadora y el herding (el ciclo).

## No objetivos

- Ningún cambio en el modelo, en el cálculo de ratings ni en la publicación: todo sale de lo ya publicado
  (`analysis/{ámbito}/poll-errors`, `events/{ámbito}/{fecha}/meta`) y de la tabla de ratings de la base.
- El detalle de una elección (`/elecciones?scope=X&date=Y`) y el perfil de una encuestadora no cambian.
- Sin páginas ni rutas nuevas; `?scope=` sigue aceptándose en `/elecciones` y preselecciona el ámbito.

## Anti-objetivos

- Que la página pese: solo viajan las filas de `poll-errors` de las elecciones publicadas (una por elección y
  encuestadora; unas 166 en `es`, menos de 1 500 en total) con las columnas que usan los gráficos. Nada de
  `error-data` (miles de filas por ámbito).
- Que una selección estrecha engañe: el contador y las comparaciones muestran siempre cuántos sondeos hay detrás.
- Perder la lectura sin JavaScript: la tabla de elecciones completa sale del servidor; los filtros y los gráficos
  necesitan JS, como hoy.
- Comparar lo incomparable: el error de bloques se calcula sobre los bloques `vs` (Izquierda y Derecha), que
  existen en todos los ámbitos; una fila sin esos dos bloques no entra en la diana.

## Restricciones

Las de la fase 7c (patrón `Page` + Jinja2 + isla + módulos JS puros con tests de paridad en node; textos en
español; `.seg` para radios; suite `python -m pytest -m "not integration" -q` en verde, 730 al empezar; commits en
inglés con el trailer de siempre).

## Diseño

### D1. Datos del histórico (`mtpy/lib/pages/elecciones.py`)

- `history_scopes() -> list[dict]`: los ámbitos del catálogo (`site().scopes()`, en su orden) con al menos una
  elección publicada; `{'code', 'name', 'analysis'}` con el nombre del catálogo ("España", "Madrid", …) y si tienen
  análisis publicado (`analysis_part(code, 'meta')`); ver la decisión 4.
- `history_events(scopes) -> list[dict]`: `published_events(code)` de cada ámbito con `scope` y `scope_name`
  añadidos, todas juntas por fecha descendente; cada una con su `href` al detalle.
- `history_rows(scopes, events) -> list[dict]`: `error_rows(code, fechas publicadas)` de cada ámbito, cada fila
  con `scope`, `scope_name` y `year` (el año de `event_date`), y solo las columnas que usa la página: `scope`,
  `scope_name`, `year`, `event_date`, `event`, `pollster`, `days`, `error_avg`, `error_blocks`,
  `error_blocks_abs`, `bias`, y las dos columnas de bloque (`Derecha`, `Izquierda`).
- `elecciones_context`: `events` (todas, con ámbito), `scopes` (`history_scopes`), `errors` (todas las filas),
  `filter_events` (fecha, nombre y ámbito de cada elección), `filter_pollsters` (nombres, alfabético),
  `years` (los años presentes, ascendente), `headline_text` (`history_headline` sobre todas: "En las N elecciones
  analizadas (G generales y A autonómicas)…"), `analysis_missing` (sin ningún ámbito con análisis) e `initial`
  (`state`, `scopes`, `events`, `errors`, `error_blocks`). La página deja de incluir `_scope_filter.html`; el
  `?scope=` de la URL, si viene, es el ámbito preseleccionado en el filtro (`initial.state.scope`), y sin él el
  filtro arranca en "Todos".

### D2. Filtros y estado (`web/dist/js/errors.js`, `web/templates/elecciones.html`)

Tarjeta "Filtros" con, en este orden: Ámbito (`select`: "Todos" y los de `scopes`), Elección (`select`: "Todas" y
las elecciones del ámbito elegido, o de todos, etiquetadas "{fecha} · {nombre}"; ajuste 2026-10-14: sin el ámbito,
el nombre de la elección ya lo dice), Encuestadora
(`select`: "Todas" y las encuestadoras con filas en la selección de ámbito y años), Años (dos `select`, "Desde" y
"Hasta", con los años de `years`; por defecto el primero y el último) y Medida (`.seg`, error de bloques / error
medio, que hoy vive en la tarjeta del histograma y se muda aquí). Cambiar el ámbito o los años vacía la
elección y la encuestadora si dejan de estar entre las opciones. Contador "N sondeos de M elecciones".

Estado en el hash, `#ambito=&eleccion=&casa=&desde=&hasta=&medida=` (solo las claves fijadas; `medida` solo si no
es la de por defecto); `parseErrorsHash`/`errorsHash` se amplían manteniendo compatibles los enlaces de hoy
(`#eleccion=&casa=&medida=`). `filterRows(rows, {scope, event, pollster, from, to})` filtra por las cinco.

### D3. Cifras de la selección (tarjeta nueva, `#figures`)

Cuatro cifras: Sondeos, Elecciones, la medida elegida (media) y "Sesgo del margen" (media de `error_blocks` con
signo, con la convención del glosario). Debajo, una línea por filtro activo con su complemento, calculado por
`complements(rows, state, measure)`: las filas que pasan todos los demás filtros pero no ese; etiquetas "Otros
ámbitos", "Otras elecciones", "Otras encuestadoras" y "Otros años", cada una con "media · N sondeos". Sin filtro
activo no hay líneas. Es lo que responde "el CIS desde 2019 frente a las demás en esos años" (línea Otras
encuestadoras) y "frente al CIS de antes" (línea Otros años).

### D4. Evolución del error por elección (tarjeta nueva, `#trend`, sustituye a `#sector`)

`charts/trend.js`: eje de tiempo; un punto por elección de la selección (la media de la medida de sus filas; tamaño
según el número de sondeos, de 6 a 18 px), una serie por ámbito (color: acento para `es`, los 16 tonos neutros del
histórico para los autonómicos, por orden de aparición) con leyenda; una línea de tendencia, la media móvil de las
últimas cinco elecciones de la selección ponderada por sondeos (`trendPoints(rows, measure)` en `errors.js`); y,
cuando hay encuestadora elegida, una segunda línea gris "Resto de encuestadoras" con la media de las demás en
esas mismas elecciones. El aviso de cada punto dice elección, ámbito, media y sondeos, y enlaza al detalle.

### D5. Histograma, diana, semanas y tabla

- Histograma: sin cambios salvo que la medida viene de los filtros.
- Diana: con un ámbito elegido, un color por elección (como hoy); con "Todos", un color por ámbito, con la
  leyenda de ámbitos. Solo entran las filas con los dos bloques `vs`.
- Semanas: una fila por elección si la selección tiene 15 elecciones o menos; con más, solo la fila "Todas" y una
  nota "Elige un ámbito o unos años para ver las elecciones una a una".
- Tabla de elecciones (`#events-table`, del servidor): todas las de todos los ámbitos, sin columna "Ámbito" (ajuste
  2026-10-14: el nombre de la elección basta; el `data-scope` de cada fila sigue sirviendo al filtro); el JS oculta las filas que no pasan el filtro de ámbito, elección y años (la encuestadora
  no aplica: son elecciones) y actualiza el pie "M elecciones".
- La tarjeta de cierre enlaza a "Ranking Encuestadoras" (sin "de este ámbito").

### D6. Ranking único (`mtpy/lib/pages/casas.py`, `web/templates/casas.html`)

- `casas_context` lee el ranking de las generales siempre: `query_ratings('es')` (y `pollster_ids`, el titular y
  el CSV "Ratings de la próxima elección general" con `api_url('es', 'ratings')`). Los efectos de encuestadora y el
  herding siguen siendo los del ámbito elegido; la columna "Herding" de la tabla del ranking también.
- Entradilla de la tarjeta: "La valoración de cada encuestadora de cara a la próxima elección general, calculada
  con sus sondeos de todas las elecciones anteriores, en todos los ámbitos; no depende del ámbito elegido arriba:
  cuanto más alta, más pesan sus sondeos en el promedio." Subtítulo con la fecha de la elección general.
- El perfil (`/casas?scope=X&id=N`) no cambia.

## Pruebas que fijan la fase

- `tests/test_pages.py`: fixture con análisis y elecciones de dos ámbitos (`es` y `es-md`); `/elecciones` lista las
  elecciones de los dos con la columna Ámbito, sin `#scope-filter`, con los `select` de ámbito/elección/
  encuestadora/años, `initial.scopes` y `initial.errors` con `scope` y `year`; `/elecciones?scope=es-md`
  preselecciona Madrid; `/casas?scope=es-md` muestra el ranking de `es`.
- `tests/test_js_errors.py`: `filterRows` con ámbito y años, `complements`, `trendPoints` (media móvil ponderada),
  hash nuevo y viejo.
- Unit: `history_scopes`, `history_events`, `history_rows`, `history_headline` con generales y autonómicas.

## Decisiones del controlador tras la revisión del constructor (2026-10-14)

1. **Identidad de una elección.** La fecha no basta entre ámbitos (varias autonómicas caen el mismo día). La clave de
   una elección es `{ámbito}:{fecha}` (`es:2023-07-23`): es el valor del `select` de Elección, de `eleccion=` en el
   hash y la clave de agrupación de la diana y de la tabla por semanas; el JS la deriva de `scope` y `event_date`
   de cada fila (`eventKey(row)`). Un hash antiguo `#eleccion=2023-07-23` (sin ámbito) se resuelve a la única
   elección de esa fecha dentro del ámbito elegido (o de `es` sin ámbito) y, si no hay una sola, se ignora.
2. **`?scope=` y `common_context`.** Las páginas globales siguen pasando por `common_context`, que exige un
   pronóstico publicado para `?scope=X` (404 si no lo hay), como hoy. El filtro de ámbito del histórico vive en el
   hash (`#ambito=`), no en `?scope=`; `?scope=` solo preselecciona el filtro cuando llega y el ámbito está entre
   los del histórico, y `selected_scope` sale del valor crudo de la consulta (`None` sin él), no del `state`
   resuelto. Los tests que piden `?scope=es-md` escriben antes un run de `es-md`.
3. **Enlaces del ranking único.** Las filas del ranking enlazan al perfil con `scope=es` (los ratings son
   nacionales); las tablas de efectos y herding enlazan con el ámbito elegido, como hoy.
4. **Ámbitos con elecciones pero sin análisis, o al revés.** `history_scopes` son los ámbitos con al menos una
   elección publicada; el análisis es opcional por ámbito: sin él aportan elecciones a la tabla y al titular, pero
   ninguna fila de errores (no aparecen en el filtro de ámbito de los sondeos). `analysis_missing` es "ningún
   ámbito con análisis": entonces se ocultan filtros y gráficos y queda la tabla, como hoy. Un ámbito con análisis
   pero sin elecciones publicadas no aporta nada. Una elección sin filas en `poll-errors` sigue en la tabla con sus
   cifras del `meta`; el contador de sondeos solo cuenta filas.
5. **Complementos.** "Años" cuenta como filtro activo solo cuando `desde`/`hasta` difieren de los extremos de los
   años presentes; "Otros años" son las filas que pasan los demás filtros con año fuera del rango; un complemento
   sin filas (p. ej. "Otros ámbitos" con una elección elegida) se omite.
6. **Generales y autonómicas.** General es `scope == 'es'` (el único sin `parent` en el catálogo); el resto,
   autonómicas.

## Lo que decide el constructor

El marcado exacto de la tarjeta de cifras, los anchos de los `select`, la implementación de la media móvil (la
fórmula está fijada), y si `charts/line.js` se puede retirar cuando "Error del sector" desaparezca (comprobar
otros usos: el perfil de encuestadora dibuja su rating con `renderLine`).

## Estado al cierre

Cerrada el 2026-10-14 en `dev`, commits `6044b85..5fa356d` (spec y plan en `337a313`). Suite: 745 tests en verde (730 al
empezar), 54 avisos preexistentes. Ejecutada en modo subagent-driven: cuatro tareas con revisión por tarea, revisión
final de toda la rama y una ola de correcciones (`e2c2a56`, `5fa356d`).

Hecho: D1 (datos de todos los ámbitos con elecciones publicadas; análisis opcional por ámbito; `poll-errors` leído
una vez por ámbito), D2 (filtros encadenados en una barra, estado en el hash con la clave `{ámbito}:{fecha}` y
`ambito=` explícito para "Todos"; los enlaces antiguos por fecha se resuelven dentro del ámbito de la página o de
`es`), D3 (cifras y complementos por filtro activo), D4 (`charts/trend.js`, evolución por elección con tendencia
ponderada y "resto de encuestadoras"), D5 (histograma, diana por elección o por ámbito, semanas con tope de 15,
tabla de elecciones de todos los ámbitos filtrada en el navegador) y D6 (ranking único de `es`, enlaces del ranking
con `scope=es`). Desviación registrada: las elecciones de ámbitos sin pronóstico publicado salen en el histórico
sin enlace al detalle (el detalle sigue necesitando un run, decisión 2); publicar `forecast` en esos ámbitos activa
los enlaces.

Pendiente de Luis: recorrido visual en escritorio y móvil; publicar `analysis` en los ámbitos que no lo tienen
(Andalucía) para que entren en el histórico, y `forecast` en los autonómicos para que sus elecciones enlacen al
detalle.

Menores aplazados: las lecturas de un análisis ausente no se cachean (un GET a S3 por ámbito sin análisis y
petición; nginx cachea la página 60 s); la lógica de años se repite entre `errors.js` y el pegamento de la tabla;
con más de 15 ámbitos autonómicos los colores se repiten; las opciones de Elección del servidor incluyen ámbitos
sin análisis (el JS las rehace); `trendPoints` descarta las elecciones sin sondeos antes de la ventana.

## Ajuste 2026-10-14: las elecciones se nombran sin el ámbito

Luis: "A la hora de listar los eventos, no hace falta que añadas el ámbito, con el nombre del evento es suficiente".
Quitado el sufijo " · {ámbito}" de la lista de elecciones pasadas de Inicio, del `select` de elección del histórico,
del tooltip de la evolución y de la columna "Ámbito" de la tabla de elecciones. En la tabla por semanas y en el
tooltip de la diana, donde las filas solo traían el código corto (`2023-23J`), ahora va el nombre completo de la
elección (`name`, que `wireHistory` añade a cada fila desde `events`); sin nombre se conserva "{código} · {ámbito}",
porque el código solo es ambiguo entre ámbitos que votaron el mismo día.

## Ajuste 2026-10-15: sin titular

Luis: "Elimina el texto de cabecera en 'Histórico Errores' y 'Ranking Encuestadoras'". Quitados el titular del
histórico (`history_headline`, `headline_text`), el del ranking y el del perfil de una encuestadora (los tres
constructores y sus pruebas); `m.headline` queda para Inicio y las páginas del método. En la misma tanda: los textos
de tarjeta pasan de 72 a 100 ch y la diana del perfil toma la clase `big` (720 px, centrada).

## Ajuste 2026-10-15: el selector de ámbito de `/casas` baja a los paneles

Luis: "Quita también el selector de ámbito de 'Ranking Encuestadoras', ya que quiero un ranking global. Si hace falta
en algún panel, déjalo en el panel". El ranking no depende del ámbito, pero los efectos de encuestadora y el herding
son el análisis del ciclo de un ámbito, y el perfil (camino, diana, efectos y herding por elección) también. Así que
el `_scope_filter.html` deja de ir sobre las tarjetas: en `/casas` vive dentro de "Efectos de encuestadora" y manda en
esa tarjeta, en Herding y en las descargas del análisis; la columna Herding del ranking pasa a salir del análisis
nacional (`RATING_SCOPE`), para que el ranking sea idéntico sea cual sea el ámbito del panel. En el perfil vive dentro
de la tarjeta de cabecera, cuyo rating y calidad son los nacionales (`query_pollster(id, 'es')`); las tarjetas de
abajo siguen al ámbito elegido. `/metodo/backtest` conserva el filtro sobre las tarjetas.

Y, el mismo día, Luis aceptó la alternativa de hacer el perfil global ("Ok, hazlo así"): el perfil no lleva selector.
El camino al rating es el nacional (el rating es el mismo número en todos los ámbitos), la diana lleva su último
sondeo en cada elección publicada de cada ámbito con análisis (las mismas elecciones que el histórico:
`history_scopes`/`history_events`/`history_rows`, que ahora importa `casas.py`; `analysis_part` se muda a `common.py`
para evitar el ciclo de imports), y los efectos por elección y el herding por ciclo salen de las tablas de todos los
ámbitos (`query_pollster(id, 'es', every_scope=True)` lee `get_house_effects(None)` y `get_herding(None)`, cacheados
aparte). Las columnas de los efectos son la unión de partidos, el catálogo del run primero y el resto por aparición;
cada elección se nombra por su nombre publicado o, sin página (el ciclo en curso, elecciones antiguas), por su fecha y
su ámbito (`election_label`). Los enlaces al perfil van sin ámbito (`/casas?id=N`).

## Ajuste 2026-10-15: «rebaño» y los filtros de la diana del perfil

Luis: "Cambia la palabra 'herding' por 'rebaño' en todas partes" y "en el perfil de encuestadora, unos filtros en el
gráfico de diana para elegir un ámbito (o todos) y un rango de años, parecido al del Histórico Errores, pero sólo
esos dos filtros". La palabra pública es «rebaño» en todas las plantillas (tablas, tarjetas, glosario, textos del
método y etiquetas de los CSV); los nombres técnicos no cambian: parte `herding` del análisis, rutas
`/api/v1/analysis/{scope}/herding`, columnas de los CSV, claves de los contextos e ids de las tarjetas (`#herding`).
La diana del perfil lleva un formulario `#bullseye-filter` (Ámbito: Todos o los ámbitos con sondeos suyos; Desde y
Hasta: los años de sus elecciones) que el servidor pinta oculto y `casas.js` enseña y cablea con `filterRows` y
`yearsOf` de `errors.js`; redibuja la diana y la tabla y escribe "N elecciones". Sin hash: el perfil no tiene otro
estado. Las filas de la diana llevan `year` y `scope`; `initial.scopes` son los ámbitos ofrecidos.

Y, en la misma diana (Luis, 2026-10-15): las generales en el azul del sitio y las autonómicas en naranja (`#c0672e`),
con leyenda; cada punto etiquetado "{código}-{ámbito}" ("2023-23J-es", "2023-28M-es-md"); y los puntos más tenues
cuanto más antigua la elección (`errors.js` `fadeByDate`: de 0,3 la más antigua a 1 la más reciente, lineal en días,
fijado sobre todas las elecciones de la encuestadora para que el filtro no cambie el tono). `renderBullseye` gana
`labelOf` y `alphaOf` (punto y etiqueta se atenúan juntos).
