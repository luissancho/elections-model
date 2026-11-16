# Web: imagen propia de una elección pasada y descripciones con cifras

Octavo paso de la ronda de SEO y redes sociales, pedido por Luis el 2026-11-16 al cerrar el de la imagen de un
sondeo (`2026-11-16-web-imagen-sondeo-design.md`). Diseño corto: casi todo se reutiliza de aquel paso.

## Intención

Las páginas que no cambian pueden enseñar cifras en la vista previa de su enlace sin chocar con la caché de X. El
sondeo ya tiene su imagen. Faltan dos cosas: la imagen de una **elección pasada** y que la **descripción** (el
texto bajo el título en Google y en la tarjeta del enlace) de un sondeo y de una elección diga sus cifras.

## Lo que decidió Luis

- «Yo seguiría la misma lógica que en las de sondeos»: el servidor dibuja al pedir, la misma tarjeta (aspecto A),
  los seis partidos con más voto por orden de voto, un día de caché, huella `?v=`, respaldo a la imagen de marca.
- «Para el triángulo, prefiero vs la elección anterior»: el triángulo de cada fila es el cambio del voto frente a
  la elección anterior, no la diferencia con el promedio de sondeos que yo proponía.

## Hechos medidos (2026-11-16, datos públicos de las 94 elecciones con página, 18 ámbitos)

- 18 son la primera elección publicada de su ámbito: no tienen anterior.
- En las otras 76, de las filas que entrarían en la imagen, 71 son de un partido que no figura con esas siglas en
  la elección anterior (UP y Cs en 2015, VOX en 2019, SUMAR en 2023, AP → PP en 1989…). En total, 384 de 557
  filas llevarían triángulo.
- El `prev_result` que la parte `model` da a cada partido coincide siempre con buscar las siglas en el resultado
  de la elección anterior publicada (0 diferencias en las 34 elecciones con modelo).
- 93 de 94 resultados traen una fila «-» (el resto de candidaturas), que en 9 elecciones estaría entre las seis.
- Un resultado trae entre 5 y 97 filas; con escaño, entre 2 y 16 partidos.
- Dibujadas las 94 con la tarjeta del sondeo: 27 títulos se encogen, ninguno se corta; ninguna línea gris se
  encoge; unas siglas largas empujan las barras en 6.
- Descripciones de 160 caracteres: caben las seis cifras en 372 de los 487 sondeos publicados (en el resto, de
  tres a cinco) y en 89 de las 94 elecciones.

## Diseño

### Imagen de una elección pasada

- **Dirección:** la de la página acabada en `.png` (`/elecciones/es/generales/2023-07.png`), con el mes o con la
  fecha completa, como la página. Mismas cabeceras que la del sondeo (un día de caché, ETag) y mismos errores que
  la página (400, 404), sin dibujar nada.
- **Tarjeta:** la del sondeo, sin tocar el dibujo.
  - Título: el nombre de la elección (o «Elección del 23 jul 2023» si no tiene).
  - Línea gris: «Resultado oficial del 23 jul 2023 · Cambio frente a la elección anterior (10 nov 2019)»; la
    segunda parte sólo si alguna fila lleva triángulo.
  - Filas: los seis partidos con más voto, por voto; barra y cifra con el resultado oficial.
  - Triángulo: voto de ahora menos voto en la elección anterior publicada del mismo ámbito, en puntos con un
    decimal; verde si sube, rojo si baja, «=» si no cambia. Un partido que no figura con esas siglas en la
    anterior va sin triángulo.
  - Columna «Escaños»: los escaños obtenidos.
  - Pie: «y 5 partidos más con escaño en demoscopio.es», si los hay.
- **Página:** declara la imagen en `og:image` (dirección canónica + `.png?v=<huella>`) con un texto alternativo
  que dice las cifras. Con un run fijado declara la huella del último run; si el proceso no puede dibujar, sigue
  con la imagen de marca (`pages.images.available`).

### Descripciones con cifras

- **Sondeo:** «Elecciones generales 2026: sondeo de Sigma Dos para El Mundo (24 ago 2023). PP 34,4%, PSOE 31,7%,
  VOX 12,7%, FA 12,4%, ERC 1,8%, JxCat 1,5%.» Las cifras son las de la imagen, tantas como quepan enteras en 160
  caracteres. Sustituye a «Voto y escaños, cambio frente al anterior y peso en el promedio».
- **Elección:** «Elecciones Generales Julio 2023: PP 33,1% (137 escaños), PSOE 31,7% (121), VOX 12,4% (33), SUMAR
  12,3% (31), ERC 1,9% (7), JxCat 1,6% (7).» Sin resultado publicado se queda la frase de hoy.

## Decisiones mías (por revisar)

- Elección anterior = la publicada inmediatamente antes en el mismo ámbito (abril de 2019 para noviembre de 2019).
- Los partidos se casan por sus siglas, como hace el modelo; no se suman ni se heredan coaliciones (SUMAR 2023 no
  se compara con UP 2019).
- La fila «-» no cuenta entre las seis ni en el pie.
- El pie cuenta sólo partidos con escaño que no caben.
- Las descripciones pierden la frase que decía qué hay en la página: con las cifras no cabe, y el título ya lo
  dice («…: resultado y errores de las encuestas»).
- En la descripción de una elección, la palabra «escaños» va sólo en la primera cifra.
- Las reglas comunes a las dos vistas previas (seis filas, versión del dibujo, huella, cifras que caben) pasan a
  un módulo propio, `pages/previews.py`.

## Fuera

La imagen de una actualización fijada con `?run=`; el `no-store` cuando la huella pedida no coincide y la memoria
de imágenes ya dibujadas (las dos preguntas abiertas del paso anterior valen igual para esta imagen).
