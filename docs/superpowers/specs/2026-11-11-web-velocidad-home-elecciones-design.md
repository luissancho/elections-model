# Velocidad de la home y de `/elecciones`: fuera la tira de elecciones pasadas, lecturas en paralelo y refresco por detrás — diseño corto

Fecha: 2026-11-11.

## Intención

Luis quiere que la home cargue rápido. Suponía que mostraba demasiada información; la medición dice otra cosa: el
navegador recibe poco (HTML 7 KB comprimidos, CSS 23 KB, ECharts diferido y guardado un año) y la espera está en el
servidor. Cada vez que caduca la copia de 60 s, montar la página exige leer de S3 muchos ficheros pequeños, uno detrás
de otro, y con visitas separadas más de un minuto casi todas pagan ese camino.

Medido en demoscopio.es el 2026-11-11 (tiempo hasta recibir la página) y en local (lecturas a S3 por render con los
punteros caducados):

| Página | Copia vigente | Copia caducada | Lecturas a S3 |
|---|---|---|---|
| Home | 0,16–0,23 s | 5,6–6,0 s | 214 (188 de la tira «Elecciones pasadas») |
| `/elecciones` | – | 5,7–6,8 s | 244 |
| Home sin la tira (simulada) | – | ≈ 1 s (estimado) | 41 |

Las 188 lecturas de la tira son dos por cada una de las 94 elecciones publicadas (un `exists` y la lectura de su
`meta`), para mostrar 12 nombres y un «+ 82 más».

## Objetivos

1. **Home:** quitar el bloque «Elecciones pasadas».
2. **Lector compartido:** las elecciones pasadas se leen en paralelo y sin la comprobación duplicada. Vive en `Site`,
   así que lo aprovechan `/elecciones`, la home (que sigue leyendo las generales para el «vs 2023» de los escaños) y
   cualquier página que liste elecciones.
3. **nginx, sólo `/elecciones`:** servir la copia guardada al instante y refrescarla por detrás.

Resultado esperado con la copia caducada, a medir en producción tras el despliegue con el mismo procedimiento
(estimaciones a partir de ≈ 27 ms por lectura desde ECS): home ≈ 0,2–0,3 s (25 lecturas, la mayoría en paralelo);
`/elecciones` ≈ 0,5 s (150 lecturas en tres oleadas) y, con visitas a menos de 10 minutos, sin espera.

## No objetivos

- Un índice de elecciones en el bundle (una lectura en vez de 112): cambia el contrato y exige republicar. Después del
  29-N, si aún hace falta.
- Refresco por detrás en la home, las páginas de evento o el resto: siguen como hoy.
- El peso de `/es` (152 KB comprimidos) y el de ECharts.
- `BundleReader.list_events`: lo usan los jobs de los estudios y no cambia.

## Anti-objetivos

- Que alguna página muestre algo distinto de lo de hoy: mismas elecciones, mismo orden, mismos 404.
- Tocar el contrato del bundle u obligar a republicar.
- Que la home o una página de evento sirvan cifras viejas tras una publicación.

## Restricciones

- Faltan 18 días para el 29-N: cambio pequeño y reversible; el TTL de 60 s no se toca.
- Cuatro workers de gunicorn, cada uno con su caché en memoria.
- Luis trabaja en paralelo en el mismo árbol (el 2026-11-11 tenía cambios sin commit en `tests/test_pages.py` y
  `web/templates/metodo_escanos.html`): no revertir ni reescribir nada suyo.

## Enfoques

1. **Leer en paralelo y sin la comprobación duplicada.** No toca el bundle; en el portátil las lecturas de la tira
   bajan de 18,4 s a 1,5 s con 16 hilos. Elegido por Luis.
2. **nginx sirve la copia guardada y la refresca por detrás.** Nadie espera, a cambio de que una visita vea una copia
   algo vieja. Elegido por Luis sólo para `/elecciones`, cuyo contenido es el que menos cambia.
3. **Índice de elecciones en el bundle.** El más limpio, pero cambia el contrato y exige republicar. Aplazado.
4. **Cargar la tira después con JavaScript** (la idea inicial). Descartado: la espera no desaparece, se traslada a la
   tira.

Además, decisión de Luis (2026-11-11): la tira de la home se quita en vez de acelerarse.

## Diseño

### Home

- `web/templates/index.html`: fuera el bloque `#past` (cabecera, chips y «+ N más»). La banda «Análisis» queda detrás
  de «¿Cómo funciona Demoscopio?».
- `mtpy/lib/pages/inicio.py`: fuera `past_events`, `PAST_SHOWN` y las claves `past_shown` y `past_more`.
- `web/dist/css/site.css`: fuera las reglas `.past`, que sólo usa ese bloque.
- **Enlace a `/elecciones` (decisión mía, a confirmar por Luis):** no se añade uno nuevo. La home ya lo tiene en la
  tarjeta «Histórico Errores» («Ver el histórico →») de «Para entender el modelo», dos bloques más arriba, y en el
  menú. Alternativa si Luis lo prefiere: una línea con el enlace donde estaba la tira.

### Lector (`mtpy/lib/webapi/published.py`)

- `Site.events(scope)`: un listado de `events/{scope}` y la lectura **en paralelo** de la `meta` de cada fecha a
  través de la caché de 60 s (la misma entrada que luego usa `published_events`). Una elección está publicada si su
  `meta` existe, la regla de `list_events`; desaparecen los 94 `exists`. Devuelve lo mismo que hoy: las fechas
  publicadas, ascendentes.
- Una precarga para las páginas que recorren todos los ámbitos: los listados de todos en paralelo y, después, todas
  las metas en paralelo (dos oleadas planas, sin pools anidados). `elecciones_context` la llama al empezar y lee
  también en paralelo las partes `meta` y `poll-errors` del análisis de cada ámbito (36 lecturas).
- Las metas de los estudios (`Site.studies`, 5 lecturas) pasan por el mismo pool.
- Un pool de hilos del `Site`, de tamaño acorde con las conexiones del cliente de S3.
- Errores como hoy: el 404 de una `meta` significa «no publicada»; cualquier otro se propaga.

### nginx (`deploy/docker/nginx.conf`)

- `location = /elecciones` con la misma configuración que `location /` más `proxy_cache_background_update on;`
  (`proxy_cache_use_stale updating` ya está). Cubre también `/elecciones?scope=…`; el detalle de una elección
  (`/elecciones/es/…`) sigue como hoy.
- La copia servida tiene como mucho unos 10 minutos: `inactive=10m` borra la que nadie ha pedido en ese tiempo, y
  entonces la visita recibe un render nuevo (≈ 0,5 s con el objetivo 2).

## Pruebas

- `tests/test_pages.py`: los tests de la tira de la home se retiran o se adaptan (cuatro sitios que buscan
  `id="past"`).
- `Site.events`: mismas fechas que `BundleReader.list_events` (también con una carpeta de fecha sin `meta`), ningún
  `exists`, y el número de lecturas de un render de la home y de `/elecciones` con los punteros caducados.
- `tests/test_deploy_files.py`: el bloque nuevo de nginx y que `location /` no lleva el refresco por detrás.
- Con el contenedor real: que la respuesta de `/elecciones` con la copia caducada no espera al refresco (depende de
  la versión de nginx de la imagen).
- Medición antes y después: lecturas por render en local y `curl` en producción tras el despliegue.

## Despliegue

Imagen nueva y `update-service` (pasos de Luis). No hay que republicar nada.

## Lo que queda al constructor

Los nombres de los métodos nuevos, el tamaño del pool y cómo guardar bajo el TTL el «no existe» de una carpeta de
fecha sin `meta`, para que no se relea en cada petición.
