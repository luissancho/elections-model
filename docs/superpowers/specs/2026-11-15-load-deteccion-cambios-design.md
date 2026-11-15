# Job `load`: detectar cambios en los sondeos ya cargados — diseño corto

Fecha: 2026-11-15.

## Intención

Wikipedia añade a veces un sondeo incompleto (sin muestra, sin escaños, sin algún partido) y lo completa después.
Si `load` lo cargó incompleto, se queda así: la carga sólo mira si el sondeo existe, no si ha cambiado. Hoy la única
salida es darse cuenta a ojo y lanzar un `load` con `overwrite`, que borra y reescribe todo el ciclo.

Luis quiere que `load` detecte esos cambios y, más adelante, vuelva a meter el sondeo completo. **Este diseño cubre
sólo la detección**; la actualización es el paso siguiente y queda esbozada al final para que la detección le sirva
tal cual.

## Objetivos

1. En cada `load` (con y sin `overwrite`, con y sin `dry_run`), comparar lo leído del artículo con lo que hay en la
   base y decir, sondeo a sondeo, qué ha cambiado.
2. Que el aviso diga qué cambió (campo, antes → ahora), no sólo que algo cambió.
3. Que salga donde ya salen los sondeos nuevos: bajo la línea de resultado del ámbito, en el resumen final y en la
   alerta, y por tanto también en `update`.

## No objetivos

- Actualizar los sondeos cambiados (paso siguiente).
- Cambiar qué se guarda: `load` escribe exactamente lo mismo que hoy, con y sin `overwrite`.
- Cambios de esquema en la base.

## Lo que haría fracasar el cambio

- Avisos falsos en cada carga (por redondeos, por columnas calculadas o por las correcciones a mano): si la lista
  no está vacía la mayoría de los días, nadie la leerá.
- Que un fallo de la comparación tumbe una carga que hoy funciona.

## Enfoques

1. **Comparar sondeo a sondeo (elegido por Luis).** Cruzar cada sondeo leído con el de la base por su clave y
   comparar los datos que vienen de Wikipedia. Dice qué cambió, no añade riesgo a la carga diaria y deja la
   actualización acotada a los sondeos afectados.
2. **`overwrite` siempre y contar la diferencia.** No hay segundo paso, pero cada carga borra el ciclo entero antes
   de reescribirlo: un corte de la conexión con la RDS a mitad lo deja vacío, y obliga a un compute completo.
3. **Huella por sondeo.** Una columna nueva con un resumen de lo leído. Exige un ALTER y sólo dice «ha cambiado».

## Diseño

### Qué se compara

La clave de un sondeo es `(event_date, event_scope, date, pollster_id, sponsor_id)`, con `date` el fin de campo.
Se compara lo leído **después** de `build_series` (con las correcciones de `data/polls/{scope}/{fecha}.csv` ya
aplicadas y los partidos excluidos ya fuera) con las filas de `polls` y `polls_results` del ciclo en la base.

- **Del sondeo:** `start_date`, `sample_size` y `ctype`. Nada calculado (`days`, `parties`, pesos, rating, errores,
  sesgos) ni `mtype`, que viene del catálogo y no del artículo.
- **De cada partido:** `pct`, `seats_min` y `seats_max` (`seats` es su media, no se compara aparte). Un partido que
  está en un lado y no en el otro es un partido añadido o quitado.
- Dos nulos son iguales; `pct` se compara redondeado a dos decimales y los enteros como enteros.

### Los cuatro casos

| Caso | Condición | Línea |
|---|---|---|
| Nuevo | Clave leída que la base no tiene | `es: + 40dB / Prisa, 2026-11-10 → 2026-11-14` (como hoy) |
| Cambiado | Misma clave, algún dato distinto | `es: ~ 40dB / Prisa, 2026-11-10 → 2026-11-14: muestra — → 2000; + SALF 1,8; PP escaños — → 125–136` |
| Sustituye a | Uno nuevo y uno retirado emparejados | `es: + SW Demoscopia / La Séptima, 2026-10-26 → 2026-11-13 (sustituye a 2026-10-26 → 2026-11-06)` |
| Retirado | Clave de la base que el artículo ya no trae | `es: - Hamalgama / OKdiario, 2026-10-20 → 2026-10-22` |

Emparejado de «sustituye a»: un nuevo y un retirado de la misma casa, y además o bien el mismo fin de campo (cambió
el medio) o bien el mismo medio y el mismo inicio de campo (cambió el fin: un tracking que se alarga). Sólo si el
emparejado es uno a uno; si hay duda, quedan como un `+` y un `-` sueltos. El retirado emparejado no sale además como
`-`.

Detalle de la línea de un sondeo cambiado, separado por `; `: `muestra a → b`, `inicio a → b`, `tipo a → b`,
`+ PARTIDO pct` (añadido), `- PARTIDO` (quitado), `PARTIDO a → b` (cambia el %), `PARTIDO escaños a → b`. Un nulo
se escribe `—`. Son líneas de consola, en el mismo registro que las `+` actuales.

### Con `overwrite`, `dry_run` y `years`

- La comparación se hace siempre antes de guardar. Con `overwrite` las líneas dicen lo que la reescritura ha
  cambiado; sin él, lo que queda pendiente en la base; en un `dry_run`, lo que cambiaría.
- Con `years` el artículo se lee a medias: no se informa de retirados ni de sustituciones (los nuevos y los
  cambiados sí).

### Dónde vive

- Una función pura en `mtpy/lib/load.py`, `poll_changes(read_polls, read_results, cur_polls, cur_results,
  removed=True)`, que devuelve las tres listas (`changed`, `replaced`, `removed`) como dicts planos, ordenadas como
  `new_poll_list`. Es la pieza que se prueba a fondo.
- `WikipediaLoader.select_series` ya lee los sondeos de la base; lee además sus resultados (una consulta más por
  carga) y deja lo necesario en el cargador, como hace hoy con `new_polls`.
- `load_polls` añade al resultado `changed`, `replaced` y `removed` junto a `new`; los sustitutos siguen contando
  en `new`.
- `pipelines/load/Polls.py` pinta las líneas con formateadores hermanos de `poll_line`; la línea de resultado añade
  los recuentos cuando no son cero (`· 2 changed, 1 removed`).
- Si la comparación falla, se avisa con un `warnings.warn` (como `parties checks failed`) y la carga sigue con las
  tres listas vacías.

### Pruebas

- Unitarias de `poll_changes`: sin cambios; cada campo del sondeo; partido añadido, quitado y cambiado; escaños;
  nulos y redondeo; nuevo, retirado y los dos emparejados; emparejado ambiguo; `removed=False`.
- De las líneas (`test_pipelines_load.py`) y del resultado de `load_polls` con cargador de pega
  (`test_load_unit.py`), incluido el fallo de la comparación que no tumba la carga.

## Decisiones mías, por revisar

- La lista de campos comparados (sin `mtype`, con `ctype`).
- La regla de emparejado y que los dudosos queden sueltos.
- Que con `years` no haya retirados.
- Los símbolos `~` y `-` y el detalle en español dentro de una línea por lo demás en inglés.

## Paso siguiente (fuera de este diseño)

Con la lista `changed` en la mano, actualizar es reescribir sólo esos sondeos: borrar sus filas de `polls` y
`polls_results` por clave, volver a insertarlas, marcar el evento como cargado y dejar que el compute recalcule los
pesos. Los `replaced` añadirían el borrado del sondeo viejo. Queda por decidir si es automático o tras una opción
(`refresh`), y qué hacer con los retirados sin pareja.
