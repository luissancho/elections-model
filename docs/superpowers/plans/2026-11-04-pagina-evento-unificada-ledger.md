# Página del evento unificada — mediciones

## Peso de partida (tarea 0, 2026-11-04)

Run `20261104-120225` de `es` (producción, leído del bucket desde la web local), `mode=forecast`.

| Página | HTML sin comprimir | gzip -9 |
|---|---:|---:|
| `/es` (Promedios) | 683.862 B | 67.909 B |
| `/es/escanos` | 820.826 B | 121.495 B |

Partes del JSON inicial con gzip: series 25,8 KB, polls 16,4 KB, runs 4,9 KB. Sólo en Escaños: dist 85,6 KB
(338 KB sin comprimir), districts 9,1 KB, fan 0,9 KB, scenario 0,8 KB.

## Maquetación de `hemi-grid` (tarea 3, 2026-11-04)

Medido en Chrome headless sobre un paquete sintético de 8 partidos (tabla de resultados de 9 columnas: necesita
613 px).

- Con el reparto de antes (`1fr 1.15fr`) la tabla se deslizaba hasta ~1340 px de ventana (604 de 613 px a 1280).
  **Decidido:** `minmax(0, 1fr) minmax(0, 1.4fr)`. La tabla cabe entera desde ~1200 px (662 px a 1280, 755 a
  1440) y el hemiciclo sigue legible (462 px de ancho a 1280). Entre 1100 y 1200 px se desliza un poco dentro de
  `table-wrap` (557 de 613 px a 1100). Por debajo de 1100 px las dos tarjetas ya se apilaban (hemiciclo arriba, tabla
  debajo) y así se quedan.
- `align-items: start` en lugar de `stretch`: al abrir «Ver las distribuciones» la tarjeta de Resultados crece
  ~900 px y, con `stretch`, el hemiciclo quedaba en una tarjeta vacía igual de alta. Ahora cada tarjeta tiene su
  altura (a 1280: 459 y 524 px cerradas).
- Móvil (390 px): todo apilado, sin desbordar el documento. La tabla se desliza en su `table-wrap` y su caption se
  corta con ella (ya pasaba en Escaños).

## Peso de llegada (tarea 5, 2026-11-05)

Mismo run `20261104-120225` de `es`, `mode=forecast`, rama `feat/pagina-evento` servida en local con el `.env` real
(`curl` a `/es?mode=forecast&run=20261104-120225`, `wc -c` y `gzip -9c | wc -c`).

| Página | HTML sin comprimir | gzip -9 |
|---|---:|---:|
| `/es` unificada | 904.624 B | 84.132 B |
| `/api/v1/forecast/es/forecast/dist?run=…` (diferido) | 338.156 B | 85.755 B |

Relación con la Promedios de partida: 84.132 / 67.909 = 1,24 (límite 2,0 = 135.818 B). Con `dist` incluido:
169.887 B, frente a los 121.495 B de la antigua Escaños. `/es/escanos` responde 301 a `/es?seccion=escanos`.

Revisión en navegador (Chrome headless, 1280 y 390 px, `/es` y `/es-md`, forecast y nowcast): los 5 enlaces del índice
dejan la sección en el borde inferior de la cabecera fija (56 px a 1280, 52 px a 390; el `scroll-margin-top` de la
sección coincide con la cabecera, 1 px de solape en escritorio por el borde); `/es/escanos#coalition=PP,VOX` acaba en
`/es#coalition=PP,VOX` con PP y VOX marcados y el desplazamiento en Escaños; sin JavaScript el formulario envía
`seccion=territorio&region=4` y muestra «Almería · 6 escaños»; ningún error de consola.
