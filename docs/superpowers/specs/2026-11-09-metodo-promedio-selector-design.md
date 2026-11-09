# /metodo/promedio: selector de ámbito y partido en «Un cálculo completo» (2026-11-09)

## Intención

La tarjeta `#ejemplo` de `/metodo/promedio` («Un cálculo completo, con datos publicados») enseña hoy un único caso:
los sondeos que pesan en el promedio del partido líder del último run nacional. Luis quiere que sea **divulgativa sin
ceñirse a un solo partido**: que el lector pueda ver cómo se comporta el método con otras nubes de puntos. Cambiar de
partido y de ámbito enseña contrastes reales: un partido grande con muchos sondeos frente a uno pequeño, o las
generales frente a una autonómica con pocos sondeos (campana más ancha, tamaño efectivo menor, banda más abierta).

## Objetivos

- Dos desplegables en la tarjeta: **ámbito** (Generales + los ámbitos autonómicos con run en vivo) y **partido** (los
  del `weights` de ese ámbito).
- Al cambiar cualquiera de los dos se actualizan el gráfico, la tabla de sondeos, las cifras del texto, el enlace de
  descarga del CSV y el conmutador Publicada/Corregida sigue funcionando.
- La página carga más ligera: el gráfico y la tabla dejan de ir en el HTML y se cargan por JS después.

## No objetivos

- Elecciones pasadas: no publican `weights` (`EVENT_PARTS` no lo incluye); generarlos desde el backtest es otro
  proyecto.
- Enlaces profundos (`?ambito=…&partido=…`) ni estado en la URL.
- Cambios en la publicación (`export_weights`, partes del run) o en la API.
- Mover el ejemplo a la página del evento.

## Anti-objetivos

- Que quien lea sin JS o un buscador pierda las cifras del texto: son lo más divulgativo de la tarjeta.
- Que la página salte al cargar el gráfico y la tabla.
- Que el texto y el gráfico muestren runs distintos si se publica uno nuevo entre la carga y el clic.

## Enfoques considerados

- **A. Sólo partido**: un desplegable de partido sobre el `weights` nacional. Barato, pero pierde el contraste con
  las autonómicas.
- **B. Ámbito + partido** (elegido): A más un desplegable de ámbito que lee el `weights` y el `meta` de otro ámbito.
- Carga: **todo por JS** (descartado: el texto quedaría sin cifras sin JS y para los buscadores) frente a **término
  medio** (elegido): el servidor pinta el texto con las cifras del líder nacional; gráfico, tabla y selectores llegan
  por JS.

Luis eligió B con el término medio (2026-11-09).

## Diseño

### Servidor (`mtpy/lib/pages/metodo.py`, `web/templates/metodo_promedio.html`)

- `weights_example()` sigue leyendo el `weights` nacional para el texto por defecto (líder por `headline`, como hoy),
  pero ya no devuelve `rows`, `line` ni `kernel` a la isla: sólo lo que pinta el texto (`name`, `date`, `published`,
  `stats`, `n_rows`, `top`, `run`).
- Nueva lista de opciones de ámbito desde `site().scopes()` (manifiesto, sin leer partes): los ámbitos `simulable`,
  Generales primero y luego en el orden del catálogo, cada uno con `code`, `name`, `run` (`latest`), `published`
  (`run_at`), `weights_href` y `meta_href` **fijados al run** (`api_url(..., run=run)`) y `csv_href`.
- La isla `initial.example` pasa a llevar `{scope, party, scopes: [...]}`; nada de filas.
- Plantilla:
  - Las cifras del texto van en `<span data-w="mean">…</span>` (y `lo`, `hi`, `nobs`, `bandwidth`, `neff`, `date`,
    `published`, `name`, `n_rows`, `top`…), renderizadas por el servidor con el caso por defecto.
  - El párrafo de «Las fechas describen dos cosas distintas…» se muestra u oculta según `date > published`.
  - Los desplegables, el `<details>` de la tabla y el gráfico van `hidden` sin JS (como `#weights-field`); la tabla
    sale con `<thead>` y `<tbody>` vacío.
  - El contenedor del gráfico conserva su alto (`chart-tall`), así no salta al dibujarse.

### Navegador

- Módulo nuevo `web/dist/js/weights-example.js` con funciones puras y testeables: elegir el partido por defecto de un
  ámbito, filtrar y ordenar las filas de un partido (más recientes primero), calcular el sondeo que más pesa, formatear
  las cifras del texto y construir las filas de la tabla.
- Carga con `fetch` como `loadDist` de `seats.js`: `weights` + `meta` del ámbito en paralelo, en caché por ámbito en
  memoria. El ámbito por defecto se carga nada más montar la página.
- `metodo.js` monta el ejemplo: revela los controles, rellena el desplegable de partidos (ordenados por media, de mayor
  a menor), redibuja con `renderWeights` y reescribe los `data-w`, la tabla, el `aria-label`, el resumen «Ver los N
  sondeos» y el enlace CSV. Publicada/Corregida conserva su valor al cambiar de partido o de ámbito.
- Si una parte falla (por ejemplo, un run anterior a la parte `weights`), la tarjeta muestra el aviso con
  `showError` y deja el texto del caso anterior; los selectores siguen activos.

### Decisiones que tomo yo (corregibles)

- Partido por defecto al cambiar de ámbito: el de mayor `stats.mean` en su `weights` (en el nacional inicial sigue
  siendo el líder del `headline`).
- Texto fuera de las generales: «el promedio del {partido} en {ámbito}»; en las generales, como hoy.
- Etiquetas: «Ámbito» y «Partido»; los desplegables reutilizan el estilo de los `<select>` que ya hay
  (`elecciones.html`). Si hace falta CSS nuevo, mínimo, porque `site.css` tiene cambios de Luis sin commit.
- El título de la tarjeta y su subtítulo no cambian de estructura; el subtítulo también lleva `data-w` para el partido
  y la fecha.

## Pruebas

- Python: el contexto de la página trae las opciones de ámbito con sus `href` fijados al run, la isla ya no lleva filas,
  y el texto sale con las cifras del caso por defecto (ajustar `tests/test_pages_ejemplo.py`).
- JS (estilo `tests/test_js_*.py`): las funciones puras del módulo nuevo, la carga con un `fetch` de prueba y el error.
- Comprobación en navegador de la página local: cambiar de partido y de ámbito, Publicada/Corregida, sin JS.

## Lo que queda al constructor

- Nombres exactos de las funciones y de los atributos `data-w`.
- Cómo escribir las filas de la tabla en JS siguiendo los formatos de `format.js` (`fmtNum`, porcentajes, fechas
  cortas) para que coincidan con los filtros de Jinja.
