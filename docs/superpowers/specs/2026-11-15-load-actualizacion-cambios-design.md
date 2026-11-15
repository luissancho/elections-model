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

## Tras la revisión final (2026-11-15)

- **Redondeo como la base:** el % se compara redondeado a dos decimales «mitad hacia arriba» sobre la cifra
  decimal (31,125 es 31,13, como lo guarda la base). Con el redondeo del real binario (31,12) un sondeo así se
  habría reescrito en cada carga, sin converger nunca.
- **Freno relativo (decisión mía, por revisar):** además del tope de 10, nunca más de la mitad de los sondeos del
  ciclo en la base, con un suelo de 2 (`load.update_limit`). Un ciclo autonómico de 8 sondeos con una columna
  corrida en el artículo daba 8 cambiados, por debajo de 10, y se habría reescrito entero.
- **Prueba en seco:** la línea dice cuántos son nuevos de verdad (antes contaba también los cambiados).
- **Prueba en seco de todos los ámbitos:** 8 comprobados (es, es-as, es-ib, es-cb, es-vc, es-ga, es-md, es-pv,
  es-ri) sin ningún cambiado ni retirado; 5 fallan antes de comparar por nombres sin mapear (es-cn, es-cm, es-ct,
  es-mc, es-nc) y 4 no tienen artículo.
- Pendientes menores: una carga con sólo sustituidos no dice `updated` en la línea de resultado; tras saltar el
  freno, el viejo de un sustituido queda como `-` hasta un `overwrite`; el borrado va antes del guardado (ventana
  corta en la que el sondeo falta); `notes` o `url` puestos a mano en un sondeo cambiado se pierden al
  reescribirlo; faltan pruebas de `delete_polls` con claves nulas y de `years` con un cambiado.
