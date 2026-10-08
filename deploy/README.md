# Despliegue de la web (fase 0)

Guía breve para construir y probar la imagen `elections-web`. Sustituye `<tag>` por la etiqueta que uses.

## Fichero de configuración

- Toda la configuración (web, publicación y jobs) sale de `deploy/elections.env`: el usuario de PostgreSQL y
  las claves de AWS/S3 que ya usa el proyecto. Está ignorado por git y excluido de la imagen por
  `.dockerignore`; se inyecta en tiempo de ejecución con `--env-file`.
- Plantilla con todas las claves (valores vacíos): `.env.example` en la raíz del repositorio.

## Construir la imagen

Desde la raíz del repositorio:

```
docker build --platform linux/amd64 -t elections-web:<tag> .
```

La imagen copia el repositorio completo (`COPY . .`) salvo lo excluido en `.dockerignore`; `deploy/elections.env`,
`.git` y el material de claves nunca entran en ella. Servicios según variables de entorno: `APP_API` arranca
nginx + gunicorn, `APP_CRONTAB` arranca supercronic con `deploy/docker/crontab`, `APP_QUEUE` el worker;
`RUN_JOB` ejecuta un job y termina.

## Arrancar el contenedor

```
docker run -d --restart unless-stopped -p 127.0.0.1:8042:8042 --env-file deploy/elections.env elections-web:<tag>
```

## Comprobar S3

Este job escribe, lee, lista y borra un fichero en `site/v1/_smoke` para detectar pronto si `s3fs==0.4.2` sigue funcionando con el bucket.

Desde el portátil:

```
set -a; . deploy/elections.env; set +a; python job.py check_s3
```

`load_dotenv` no pisa las variables ya exportadas, así que el `.env` del repositorio no interfiere.
El job se niega a probar un `files/` local (sería un falso visto bueno); para forzarlo:
`python job.py check_s3 '{"allow_local": true}'`.

Desde la imagen:

```
docker run --rm --env-file deploy/elections.env -e RUN_JOB=check_s3 elections-web:<tag>
```

Cada paso imprime `<paso> OK` o `<paso> FAIL: <error>`; si alguno falla, el job termina con error.

## Si `check_s3` falla

1. Subir `s3fs` a una versión de 2024 o posterior (con un `aiobotocore` compatible) y repetir.
2. Si sigue fallando, la fase 1 reimplementa `read_bytes`, `write_bytes`, `exists`, `listdir` y `remove` de `mtpy/core/services/s3.py` sobre `boto3`.

El primer `FAIL` es la causa raíz (los pasos posteriores fallan en cascada). Un `AccessDenied`, o un `gone FAIL` tras `remove OK`, indica que al usuario de AWS le falta `s3:DeleteObject` sobre el bucket, no un problema de `s3fs`.
