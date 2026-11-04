# Evolución de las actualizaciones: donuts de cambio y tabla de cambios — diseño corto

Fecha: 2026-11-04. Tarjeta «Evolución de las actualizaciones» de Promedios (voto, `pct`) y Escaños (`seats`),
código compartido en `web/dist/js/charts/evolution.js` y `web/dist/js/pages/evolution-wire.js`.

## Intención

Luis quiere que la tarjeta cuente cómo se ha movido cada partido: desde el inicio, en la última semana y si ha
cambiado hoy mismo. La distinción entre runs en vivo y de relleno (`backfill`) deja de importar al lector.

## Objetivos

1. **Un punto por día, sin distinguir el tipo de run.** De cada día se queda el run con el `run_at` más tardío,
   sea en vivo o de relleno. Todos los puntos son sólidos.
2. **Donut = cambio.** Un punto se dibuja hueco (el donut actual: relleno blanco, borde del color, 8 px) cuando la
   cifra que se ve cambia respecto al día anterior con datos de ese mismo nombre: el voto redondeado a un decimal
   (`fmtPct`) o el escaño. El primer día no lleva donut.
3. **Tabla de cambios bajo el gráfico** (enfoque A, elegido por Luis): una fila por nombre de la vista actual,
   con la cifra del último día y tres columnas de cambio con las flechas `.dlt`.

```
                 4 nov    Hoy      7 días    Desde 5 oct
PP              31,1 %   ▼ 0,2    ▼ 0,5     ▼ 1,6
PSOE            26,7 %   ▲ 0,1    ▼ 0,1     ▲ 0,3
VOX             19,3 %   ▲ 0,1    ▲ 0,4     ▲ 1,4
```

## No objetivos

- No se toca el histórico publicado (`history.json`) ni el back: todo sale de `initial.runs`.
- No cambia qué se publica: los runs de relleno siguen marcados `backfill` en los datos, sólo deja de usarse en la
  tarjeta.
- No hay plazo «1 mes»: los tres plazos son los que pidió Luis.

## Antiobjetivos

- Que el gráfico y la tabla se contradigan (un donut sin cambio en la tabla o al revés).
- Que la deriva diaria de 0,01 sin sondeos nuevos llene el gráfico de donuts: por eso la regla usa la cifra
  redondeada, no la diferencia en bruto.

## Enfoques considerados

- **A. Tabla en el navegador** con los runs diarios de la tarjeta (elegido). Mismos datos, mismo redondeo y misma
  vista Partidos/Bloques que el gráfico. Requiere un espejo JS de `delta_badge`.
- B. Tabla en el servidor con la macro `badge`: duplicaría en Python la elección del run diario y la suma por
  bloques; el cambio a Bloques obligaría a dos tablas. Descartado.
- C. Cambios dentro de las pastillas de la leyenda: ilegible con 10+ partidos y en móvil. Descartado.

## Decisiones de detalle (mías, para revisar)

- **Días de comparación.** «Hoy» compara el último día del histórico con el día anterior con datos. «7 días»
  compara con el último día con datos en o antes de (último día − 7). «Desde {fecha}» compara con el primer día
  del histórico. Sin día de comparación, o sin el nombre en ese día, la celda lleva «–».
- **Cambios entre cifras redondeadas.** Δ = cifra mostrada hoy − cifra mostrada entonces (a un decimal o en
  escaños), para que la tabla cuadre con lo que se lee y con los donuts. «–» cuando el Δ es 0.
- **Cabeceras.** La columna de la cifra lleva la fecha del último día («4 nov»). La primera columna de cambio se
  llama «Hoy» si el último día es hoy (fecha local del navegador) y, si no, lleva la fecha de ese día. La última
  columna es «Desde {fecha corta del primer día}».
- **Filas.** Los nombres de la vista (Partidos o Bloques) en el orden de la leyenda. La tabla sigue la selección
  de la leyenda: un nombre oculto en el gráfico se oculta en la tabla. Así la descarga, que incluye la tabla
  debajo (`share.js`, `readTable`), muestra lo mismo que el gráfico.
- **Tooltip.** La cabecera pasa a ser siempre la fecha del día (`fmtDate`), sin hora ni «estimación
  retrospectiva». Cada línea añade la flecha del cambio frente al día anterior cuando lo hay.
- **Espejo de `delta_badge`.** Nueva `fmtBadge(x, digits)` en `format.js`, con el mismo contrato que la de
  Python (`{dir, text}`: `▲ 6`, `▼ 1,0` o «–»), y una prueba que compare ambas.
- **Texto «Cómo leerlo».** Se quita la frase de los puntos huecos como estimaciones retrospectivas y se explica
  que el punto hueco marca un día en que la cifra cambió, y qué compara cada columna de la tabla. Se mantiene
  «actualización» como palabra pública y «sólo» con tilde.

## Queda para quien lo construya

- El marcado de la tabla (clases de las tablas del sitio, `.dlt`), su ancho en móvil y cómo se rellena en
  `wireEvolution` al cambiar de vista o de selección.
- Las pruebas en `tests/test_js_evolution.py` (elección del run diario sin `backfill`, donuts con deriva por
  debajo del redondeo, días de comparación con huecos, nombres ausentes) y el ajuste de las que hoy esperan
  puntos huecos por `backfill`.
- Comprobación en navegador de las dos páginas y de la descarga con la tabla debajo.
