# Job `load`: actualizar los sondeos que han cambiado en Wikipedia — diseño corto

Fecha: 2026-11-15. Continúa `2026-11-15-load-deteccion-cambios-design.md` (la detección, ya hecha).

## Intención

Que un sondeo que entró incompleto acabe completo sin que Luis tenga que verlo en el log ni lanzar un `overwrite`
de todo el ciclo: `load` ya detecta los cambios; ahora los aplica él solo.

## Objetivos

1. Un `load` normal (sin `overwrite`) reescribe los sondeos cambiados: borra sus filas de `polls` y
   `polls_results` por clave y las vuelve a insertar con lo leído. El resto del ciclo no se toca.
2. Un sondeo nuevo que sustituye a otro entra como hoy y el viejo se borra.
3. `update` rehace el compute y la publicación ese día aunque no haya sondeos nuevos.

## No objetivos

- Borrar los retirados sin pareja: sólo se avisan (línea `-`). Para borrarlos sigue estando `overwrite`.
- Una opción aparte (`refresh`): obligaría a seguir mirando el log y lanzando algo a mano.

## Lo que haría fracasar el cambio

- Que una lectura defectuosa del artículo reescriba o borre en masa sondeos buenos.
- Que un `dry_run` escriba algo.

## Enfoques

1. **Automático dentro de `load` (aprobado por Luis).** Los cambiados se suman a los sondeos que se guardan y sus
   filas viejas se borran antes por clave.
2. **Opción `refresh`.** Descartado: no quita el trabajo manual.
3. **`overwrite` siempre.** Descartado en el diseño de la detección (borra el ciclo entero en cada carga).

## Diseño

- **Qué se aplica:** las listas `changed` y `replaced` de `load.poll_changes`. `removed` nunca.
- **Freno:** si `changed` + `replaced` suman más de `MAX_UPDATES` = 10, no se aplica ninguno: se avisa con un
  `warnings.warn` (cuenta en los avisos del ámbito) y las líneas siguen saliendo. Los nuevos se guardan como hoy,
  también los que sustituyen a otro (el viejo se queda).
- **Orden de escritura:** primero se borran por clave las filas de `polls_results` y `polls` de los cambiados y de
  los viejos sustituidos; después se guardan los nuevos y los cambiados juntos, por el camino de siempre
  (`save_polls`, `save_results`, pesos). Si la conexión cae entre el borrado y el guardado, el sondeo falta en la
  base y la siguiente carga lo ve como nuevo y lo mete: se repara solo.
- **Con `overwrite`:** nada nuevo que hacer, la reescritura ya aplica los cambios. **Con `dry_run`:** no se
  escribe ni se borra nada.
- **Si la comparación falla:** como hoy, se avisa y la carga sigue sin actualizar nada.
- **`update`:** `save_polls` marca el evento como cargado (`events.loaded_at`), así que el compute lo ve pendiente.
- **Líneas:** las mismas. La línea de resultado dice `N updated` cuando los cambios se han escrito (carga normal
  bajo el freno, o `overwrite`) y `N changed` cuando no (`dry_run`, freno). `n_saved` cuenta nuevos y cambiados.

### Dónde vive

- `load.poll_changes` devuelve además `keys`: las claves `(date, pollster_id, sponsor_id)` de los cambiados y de
  los viejos sustituidos. No va al resultado del job.
- `WikipediaLoader.select_changed(keys)` suma a `polls`/`results` las filas leídas de esas claves;
  `WikipediaLoader.delete_polls(keys)` borra de la base las filas de esas claves.
- `load_polls` decide (freno, `overwrite`, `save`) y añade `updated` (bool) al resultado.

## Decisiones mías, por revisar

- El freno cuenta cambiados y sustituidos juntos, y al saltar no aplica ninguno.
- Un fallo al borrar o guardar tumba la carga del ámbito, como cualquier fallo de escritura de hoy.
