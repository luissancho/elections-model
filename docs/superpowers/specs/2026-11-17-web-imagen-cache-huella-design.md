# Web: el día de caché de la imagen de un enlace sólo vale para la dirección que la página declara

Cierra una de las dos preguntas abiertas de las imágenes de enlace (`2026-11-16-web-imagen-sondeo-design.md`,
`2026-11-16-web-imagen-eleccion-design.md`): qué pasa justo después de publicar. Diseño corto, aprobado por Luis el
2026-11-17 («ok» a la propuesta de la regla del minuto).

## Intención

Durante el primer minuto tras una publicación, un worker puede seguir con el run anterior (`cache_ttl`). Si una red
pide la dirección nueva de una imagen (`….png?v=<huella nueva>`) y le responde ese worker, recibe el dibujo viejo
con un día de caché, y nginx lo conserva bajo la dirección nueva mientras se siga pidiendo. Lo mismo pasa con un
enlace antiguo (`?v=<huella vieja>`) después de una corrección: la dirección no nombra lo que se dibuja.

## Hechos leídos (2026-11-17)

- nginx guarda en `api_cache` lo que responde la aplicación, por dirección completa (con su consulta), el tiempo
  que diga `cache-control` (`deploy/docker/nginx.conf`, `location /`; `inactive=10m` borra lo que nadie pide en 10
  minutos). No hay `proxy_ignore_headers`.
- Las páginas salen con `public, max-age=<cache_ttl>` (60 s por defecto); las imágenes, con
  `public, max-age=86400` (`Page.IMAGE_TTL`) sea cual sea su consulta.
- La huella `v` es la de lo que el servidor dibuja (`previews.preview_version`); la página la declara en
  `og:image`.

## Diseño

- El día de caché (`Page.IMAGE_TTL`) sólo se da cuando la `v` pedida es exactamente la huella de lo que el
  servidor acaba de dibujar. Con otra `v`, con `v` vacía o sin `v`, la misma imagen sale con el tiempo de las
  páginas (`cache_ttl`, un minuto): `public, max-age=60`.
- El resto no cambia: el PNG, su ETag, que la imagen es siempre la de lo último publicado, los 400 y 404 sin caché
  y que la consulta no cambia el dibujo. Otros parámetros junto a la `v` correcta no quitan el día.
- Vale para las dos imágenes (sondeo y elección pasada): la regla vive en `Page.png`.

## Enfoques considerados

- **Un minuto si la huella no coincide (elegido).** Un dibujo viejo dura como mucho un minuto en nginx desde que
  se guarda, igual que una página (hasta dos minutos tras la publicación: uno de worker atrasado y otro de
  nginx), y nginx sigue absorbiendo las ráfagas de un enlace antiguo muy compartido.
- `no-store` si no coincide: un enlace antiguo muy compartido obligaría a dibujar en cada petición.
- Que los workers se enteren de la publicación al momento: arregla también el 404 de un sondeo recién publicado,
  pero toca el flujo de publicación; fuera, y no antes del 29-N.

## Decisiones mías (por revisar)

- Sin `v` también es un minuto: esa dirección no nombra ningún dibujo, así que su contenido puede cambiar.
- El tiempo corto es `cache_ttl` (el de las páginas), no una constante nueva.

## Fuera

La copia que una red ya se haya llevado en esos dos minutos (X la guarda unos siete días: sólo lo evita esperar
tres minutos antes de compartir); la copia de un día que nginx ya tuviera de una dirección antigua antes de una
corrección; el 404 de un sondeo recién publicado en un worker atrasado; la memoria de imágenes
ya dibujadas (descartada: nginx ya las guarda por dirección); cambios en nginx.
