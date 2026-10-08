FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_PATH=/app

# Runtime only: nginx + supervisord serve the API, curl runs the healthcheck (every pin ships wheels;
# psycopg2-binary bundles its own libpq)
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl nginx supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /bin/bash /bin/sh

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY deploy/docker/nginx.conf /etc/nginx/nginx.conf
COPY deploy/docker/supervisord.conf /etc/supervisord.conf
COPY deploy/docker/init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

RUN addgroup --gid 10001 docker \
    && adduser --disabled-password --uid 10001 --ingroup docker docker \
    && mkdir -p /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor \
    && chown -R docker:docker /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor \
    && chmod -R g=u /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor

WORKDIR ${APP_PATH}

# Explicit list: credentials, git history, notebooks and runtime files never enter the image
COPY --chown=docker:docker api.py job.py worker.py pipeline.py config.json ./
COPY --chown=docker:docker mtpy/ mtpy/
COPY --chown=docker:docker data/ data/
RUN mkdir -p files log && chown -R docker:docker files log

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

USER docker
EXPOSE 8042
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8042/ || exit 1

CMD ["/bin/bash", "-c", "init.sh"]
