# Página del evento unificada (Promedios + Escaños) — diseño

Fecha: 2026-11-04. Páginas afectadas: `/{scope}` (Promedios: `controllers/Promedios.py`, `lib/pages/promedios.py`,
`web/templates/promedios.html`, `web/dist/js/pages/promedios.js`) y `/{scope}/escanos` (Escaños:
`controllers/Escanos.py`, `lib/pages/escanos.py`, `web/templates/escanos.html`, `web/dist/js/pages/escanos.js`).

## Intención

Luis quiere unir las dos pestañas del evento en una sola página, «la página de la elección en curso». Hoy los
datos están mezclados: Promedios enseña escaños (hemiciclo, tabla con escaños y rango) y Escaños enseña
porcentajes (abanico de voto, voto por circunscripción). Para el lector son dos mitades de la misma pregunta,
y no tiene sentido obligarle a cambiar de pestaña para responderla.

## Objetivos

1. **Una sola página por ámbito en `/{scope}`**, con todo lo que hoy reparten las dos pestañas y sin
   duplicados.
2. **Orden** (decidido por Luis: el promedio va primero y después todo lo demás en el orden propuesto):

   ```
   [barra del evento: Generales 2026 · 29-nov · 25 días]   índice: Promedio · Escaños · Evolución · Territorio · Tabla
   KPIs + carrusel de partidos
   ── Promedio ──    Promedio de sondeos (y proyección)
   ── Escaños ──     Hemiciclo | tabla única de resultados (Partidos/Bloques)
                     «Ver las distribuciones» (ridge)
                     Calculadora de mayorías
   ── Evolución ──   Evolución de las actualizaciones (selector Voto / Escaños)
                     Una encuestadora frente al promedio
   ── Territorio ──  Resultados por territorio
   ── Tabla ──       Todas las encuestas
                     Abanico de voto por horizonte · Descargas
   cierre: los dos enlaces al método (promedio y escaños)
   ```

3. **Un índice de secciones** sustituye a las dos pestañas (`EVENT_TABS`) de la barra del evento. Son enlaces
   a anclas de la misma página, y la barra se queda fija al hacer scroll, como ahora. **Las secciones no llevan
   título visible** (decisión de Luis): son envoltorios con `id`, y los separadores «── … ──» del esquema de
   arriba sólo indican dónde empieza cada ancla. Cada tarjeta mantiene su propio título.
3b. **Nombre de la página** (decisión de Luis): **«Encuestas»** en la barra inferior del móvil y donde haga
   falta una etiqueta corta. El `<title>` es **«Encuestas, promedios y escaños · {título del evento}»**
   («Encuestas, promedios y escaños · Generales 2026»), y la descripción sigue la misma idea. La clave interna
   de `TABBAR` puede seguir siendo la actual o cambiar a una neutra en inglés; eso lo decide quien lo
   construya.
4. **Sin duplicados**:
   - **Una sola tabla de resultados**, que sustituye a «Resultados por partido» de Promedios y a la tabla de
     «Reparto de escaños». Columnas: Partido · % Voto · vs {año} · Escaños · vs {año} · Rango 95% ·
     P(escaño) · P(primero) · P(mayoría). Su gemela por bloques tiene: Bloque · % Voto · vs · Escaños · vs ·
     Rango 95% · P(mayoría). Esto revierte a propósito la decisión del 17-10 de quitar las P(*) de Promedios:
     ahora es la única tabla. P(escaño) se queda (Luis, 2026-11-04): responde a «¿entra en el Congreso?» en los
     partidos pequeños. La tabla se desliza en el móvil dentro de `table-wrap`, como hoy la de Escaños.
   - **Un solo dibujo del reparto central: el hemiciclo.** La barra apilada de «Reparto de escaños»
     (`seats-bar`) y su botón de descarga desaparecen. La calculadora mantiene su propia barra.
   - **Un solo selector Partidos / Bloques** para el hemiciclo, la tabla y las distribuciones (hoy hay dos:
     `results-view` en Promedios y `seats-view` en Escaños).
   - **Una sola tarjeta de Evolución**, con un selector «Voto / Escaños» que cambia el gráfico y la tabla de
     cambios (`evolution-wire.js` ya sabe dibujar las dos métricas). Por defecto, Voto.
5. **Las URL antiguas no se rompen.** `/{scope}/escanos` devuelve un 301 a `/{scope}` con la misma query más
   `seccion=escanos` (o `seccion=territorio` si trae `region=N`), **sin fragmento** en el `Location`. Así el
   navegador conserva el fragmento original (`#coalition=PP,VOX`), cosa que no haría si el `Location` llevase su
   propio `#escanos`. Al cargar, el JS hace scroll a la sección de `seccion` y quita el parámetro con
   `replaceState`. Sin JS, la página se abre arriba. El `/escanos` antiguo (anterior a la fase 7) redirige
   directamente al destino final, sin encadenar dos 301. *Decisión mía*, a partir de la revisión: un
   `Location` con `#escanos` rompía los enlaces a la calculadora que ya circulan.
6. **Los textos «Cómo leerlo» se reescriben** para que no mencionen pestañas: «la pestaña Escaños enseña
   cuánto» pasa a «más abajo, en Escaños…», y «No incluyen los sondeos: están en la pestaña Promedios»
   desaparece porque ya están en la misma página. Las reglas de redacción de siempre siguen vigentes («sólo»
   con tilde, encuestadora o casa y nunca empresa, «actualización» para los runs; `assert_public_wording`).

## No objetivos

- No cambian los datos publicados, la API ni los CSV. No se toca el modelo.
- No se rediseñan las tarjetas por dentro: se mueven, se fusionan o se borran como se dice arriba.
- No hay pestañas internas con JS ni contenido oculto según la sección (descartado, ver Enfoques).
- No hay carga diferida de partes por la API, salvo que la medición del punto «Peso» lo exija.

## Antiobjetivos

- Que el lector pierda algo que hoy ve en alguna de las dos pestañas, salvo lo que se quita a propósito (la
  barra apilada).
- Que un enlace compartido a `/es/escanos` o a una calculadora (`#coalition=PP,VOX`) deje de llevar a donde
  llevaba.
- Que los controles se pisen el estado en la URL (ver «Estado en el hash»).

## Restricciones

- **Estado en el hash.** Hoy escriben el `location.hash` entero tanto los controles del promedio
  (`seriesHash` en `blocks.js`: `#ventana=…&vista=…&sel=…&mostrar=…`, vía `series-wire.js`) como la calculadora
  (`coalitionHash` en `seats.js`: `#coalition=…`). En una sola página se borrarían el uno al otro y también el
  ancla de la sección. Hace falta un pequeño módulo común que lea los pares `clave=valor` del hash, cambie sólo
  los de su dueño y conserve el resto, incluido un ancla suelta (`#escanos&coalition=PP,VOX`). Los lectores
  actuales ya separan por `&` y buscan su clave, así que siguen sirviendo. Como el navegador no hace scroll a
  `#escanos&…` por sí mismo, al cargar la página el JS hace scroll a la sección del token suelto, si lo hay.
  Los enlaces que ya circulan (`#coalition=…`, `#ventana=…`) tienen que seguir funcionando.
- **Circunscripción elegida.** Hoy el formulario `district-form` recarga `/{scope}/escanos?region=N#districts`
  (sin JS) y el JS construye esa misma URL (`escanos.js:181`). *Decisión mía:* con JS, elegir una
  circunscripción dibuja su detalle en el navegador a partir de `initial.districts` (que ya trae todas las filas),
  sin recargar una página que ahora pesa más y sin perder el estado del hash. Sin JS, el formulario envía a
  `/{scope}?…&region=N&seccion=territorio`. El controlador de `/{scope}` acepta `region` (con los mismos 400/404
  que hoy Escaños) para el primer dibujo del servidor.
- **Peso.** El contexto inicial (`initial`) junta los de las dos páginas: series, polls, projection, vote,
  summary, dist, districts, scenario, fan, el histórico de runs y los deltas. `runs`, `summary`, `meta`,
  `deltas` e `inks` van una sola vez. *Regla (decisión mía):* se mide el HTML de `/es` con gzip contra la
  Promedios de hoy sola, y sobre un run de producción de 10.000 simulaciones (leído del bucket publicado), no
  sobre el run local del 8-10 (1.000 simulaciones: `dist` de 33 KB, cuando con 10.000 rondará los ~340 KB sin
  comprimir). Si el `/es` unido pesa más del doble que la Promedios de hoy con gzip, o el dibujo inicial se nota
  más lento en el móvil, se le cuenta a Luis antes de seguir. **Saltó (2026-11-04):** con el run
  `20261104-120225` (10.000 simulaciones) la Promedios de hoy pesa 68 KB con gzip y la unida rondaría los
  ~170 KB (2,5×). Casi todo es `dist` (86 KB con gzip). **Decisión de Luis:** `dist` no va en el JSON inicial. La
  página la pide a la API (`/api/v1/forecast/{scope}/{mode}/dist?run=…`) justo después del primer dibujo. La
  calculadora y las distribuciones muestran «cargando» mientras llega y un aviso si la petición falla. Sin
  `dist` la página queda en ~85 KB (1,25×).
- **Trabajo en paralelo de Luis:** `promedios.html` y `escanos.html` han cambiado en disco mientras se escribía
  este diseño. El plan se hace sobre su estado en el momento de empezar.
- **Barra inferior del móvil** (`TABBAR`): Promedios y Escaños se convierten en una sola entrada, así que quedan
  4: Inicio · Encuestas · Histórico · Ranking.
- **Navegación:** Inicio (botón «Escaños» del hero, `href_seats` en `inicio.py`) apunta a `/{scope}#escanos`.
  `metodo_escanos.html:764` enlaza `/es#escanos`. Se actualizan `deploy/check-api.sh`, `deploy/smoke.sh` y
  `deploy/README.md`: `/es/escanos` pasa a esperar un 301.
- **El modo** (`?mode=`, `?run=`) sigue siendo uno para toda la página, como ahora.
- **Tests:** `test_pages.py`, `test_web_routes.py` y `test_pages_ejemplo.py` cubren hoy las dos páginas y se
  adaptan. Los tests nunca usan `out=None` (incidente del 31-10).

## Enfoques

- **A. Una página larga con un índice de secciones fijo (elegido por Luis).** Todo está a la vista y es
  indexable, cada sección tiene su ancla para compartirla y funciona sin JS. El coste es una página larga y
  más pesada.
- B. Una página con pestañas internas por JS. Esconde contenido, sin JS no funciona y las anclas compartidas
  dependen del JS. Descartado.
- C. Mantener dos URL, pero cortadas por tema (Pronóstico / Sondeos). Sólo cambia de sitio la mezcla.
  Descartado.

## Lo decidido sobre el cómo

- Enfoque A, con el orden del punto 2 (promedio primero).
- El 301 de `/{scope}/escanos` conserva la query y añade `seccion=escanos`, sin fragmento (§5).
- La barra apilada sale y el hemiciclo es el único dibujo del reparto central.
- La tabla de resultados es única y lleva P(primero) y P(mayoría).

## Para quien lo construya

- **Organización del código** (*propuesta mía*): `promedios.html` absorbe las tarjetas de `escanos.html`, que se
  borra. `promedios_context` compone el contexto con las piezas de `escanos.py` (`seat_rows`, `block_rows`,
  `district_table`, `region_rows`, `csv_links`, `default_coalition`…), sin duplicarlas. El controlador
  `Escanos` pasa a ser una acción de `Redirect`. En JS, el cableado de escaños de `escanos.js` (calculadora,
  distribuciones, territorio, abanico) se convierte en un módulo `pages/seats-wire.js` que importa
  `promedios.js`, y la página carga un solo script.
- La maquetación de `hemi-grid`, ahora que la tabla es más ancha (¿hemiciclo arriba y tabla debajo en
  pantallas medianas?), se resuelve mirándola en el navegador, en escritorio y en móvil.
- Los nombres de los ids que hoy se repiten entre las dos plantillas (`parties-table`, `evolution-*`) se
  ordenan sin cambiar los ids públicos de las anclas: `#promedio`, `#escanos`, `#evolucion`, `#territorio`,
  `#tabla`.
- Los subtítulos y los nombres de fichero de las descargas siguen el patrón actual.

## Abierto para Luis

Nada. El nombre de la página y las secciones sin título los decidió Luis el 2026-11-04.

## Cierre

Hecho el 2026-11-05 en la rama `feat/pagina-evento`. Commits desde `1384172`:

```
9cf4e6f web: /{scope}/escanos redirige a la página del evento; barra inferior de 4 entradas («Encuestas»)
008f5ef web: los botones «Ver las distribuciones» y «Ver los datos» se ocultan de verdad con hidden; píldoras de la calculadora atenuadas mientras no hay simulaciones
3974aef web: el JS de escaños (seats-wire.js) se carga en la página del evento, con la evolución Voto/Escaños y el detalle de circunscripción sin recarga
aba9edd web: la página del evento junta Promedios y Escaños por secciones (plantilla y contexto)
7eee9a6 docs: dist se pide aparte a la API (decisión de Luis por el peso)
2cd72f5 web: estado compartido del fragmento de la URL (url-state.js) para los controles de la página del evento
```

**Peso** (run `20261104-120225` de `es`, `mode=forecast`, producción vía servidor local):

| Página | Sin comprimir | gzip -9 |
|---|---:|---:|
| `/es` antes (Promedios) | 683.862 B | 67.909 B |
| `/es/escanos` antes | 820.826 B | 121.495 B |
| `/es` ahora (unificada) | 904.624 B | 84.132 B |
| `dist` pedido aparte (`dist_href`) | 338.156 B | 85.755 B |

La página unificada pesa 1,24 veces la Promedios de partida (el límite era 2 veces) y 31 % menos en gzip que las dos
páginas por separado. La petición diferida de `dist` añade 85,8 KB tras el primer pintado (total 169,9 KB si se
consume todo, frente a los 121,5 KB de Escaños: ese coste extra sólo lo paga quien abre la calculadora o las
distribuciones, y la primera pintura no lo espera).

**Maquetación:** `hemi-grid` queda en `minmax(0, 1fr) minmax(0, 1.4fr)` con `align-items: start`; la tabla de
resultados cabe entera desde ~1200 px y por debajo de 1100 px las dos tarjetas se apilan (medidas en el ledger).

**Revisión en navegador:** el índice de 5 enlaces, `/es/escanos#coalition=PP,VOX`, el formulario sin JavaScript y la
consola, en `/es` y `/es-md`, en los dos modos, a 1280 y 390 px, sin fallos. Sin JavaScript, enviar el formulario
muestra la circunscripción elegida pero la página se queda arriba (sin JS no hay quien desplace a la sección).
