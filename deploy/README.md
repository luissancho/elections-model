# Despliegue de la web (fase 0)

Guía breve para construir y probar la imagen `elections-web`. Sustituye `<tag>` por la etiqueta que uses.

## Construir la imagen

Desde la raíz del repositorio:

```
docker build --platform linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t elections-web:<tag> .
```

## Arrancar el contenedor

```
docker run -d --restart unless-stopped -p 127.0.0.1:8042:8042 --env-file ~/.config/elections-model/web.env elections-web:<tag>
```

## Ficheros de entorno

- `deploy/web.env.example` y `deploy/publish.env.example` son plantillas sin secretos (valores vacíos).
- Los `.env` reales viven en `~/.config/elections-model/` (`web.env` y `publish.env`), con modo 600, y nunca en el repositorio.
- `GIT_COMMIT` no debe aparecer en los `.env`: lo fija la imagen en el build y un `--env-file` lo pisaría.
- IAM: `elections-web` solo lee (`s3:GetObject` y `s3:ListBucket` sobre `site/*`); `elections-publish` añade `s3:PutObject` y `s3:DeleteObject` sobre `site/*` (los necesitan `unpublish` y `check_s3`).
- `deploy/sql/web_reader.sql` crea el rol de solo lectura de la base; se ejecuta con el usuario maestro de la RDS.

## Comprobar S3

Este job escribe, lee, lista y borra un fichero en `site/v1/_smoke` para detectar pronto si `s3fs==0.4.2` sigue funcionando con el bucket.

Desde el portátil, con un `.env` que apunte al bucket:

```
set -a; . ~/.config/elections-model/publish.env; set +a; python job.py check_s3
```

`load_dotenv` no pisa las variables ya exportadas, así que el `.env` del repositorio no interfiere.
El job se niega a probar un `files/` local (sería un falso visto bueno); para forzarlo:
`python job.py check_s3 '{"allow_local": true}'`.

Desde la imagen:

```
docker run --rm --env-file ~/.config/elections-model/publish.env -e RUN_JOB=check_s3 elections-web:<tag>
```

Cada paso imprime `<paso> OK` o `<paso> FAIL: <error>`; si alguno falla, el job termina con error.

## Si `check_s3` falla

1. Subir `s3fs` a una versión de 2024 o posterior (con un `aiobotocore` compatible) y repetir.
2. Si sigue fallando, la fase 1 reimplementa `read_bytes`, `write_bytes`, `exists`, `listdir` y `remove` de `mtpy/core/services/s3.py` sobre `boto3`.

El primer `FAIL` es la causa raíz (los pasos posteriores fallan en cascada). Un `AccessDenied`, o un `gone FAIL` tras `remove OK`, indica que al usuario IAM le falta `s3:DeleteObject`, no un problema de `s3fs`.
