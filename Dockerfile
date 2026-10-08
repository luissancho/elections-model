FROM python:3.11-slim

# Build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        bash \
        build-essential \
        cmake \
        curl \
        dnsutils \
        gcc \
        libpq-dev \
        nginx \
        supervisor \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSLO "https://github.com/aptible/supercronic/releases/download/v0.2.29/supercronic-linux-amd64" \
    && echo "cd48d45c4b10f3f0bfdd3a57d054cd05ac96812b supercronic-linux-amd64" | sha1sum -c - \
    && chmod +x supercronic-linux-amd64 \
    && mv supercronic-linux-amd64 /usr/local/bin/supercronic-linux-amd64 \
    && ln -s /usr/local/bin/supercronic-linux-amd64 /usr/local/bin/supercronic \
    && ln -sf /bin/bash /bin/sh

# Python requirements
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

# Configure crontab schedules
COPY ./deploy/docker/crontab /etc/crontab

# Configure NGINX
COPY ./deploy/docker/nginx.conf /etc/nginx/nginx.conf

# Configure Supervisord
COPY ./deploy/docker/supervisord.conf /etc/supervisord.conf

# Init script
COPY ./deploy/docker/init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

RUN addgroup --gid 10001 docker \
    && adduser --disabled-password --uid 10001 --ingroup docker docker \
    && mkdir -p /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor \
    && chown -R docker:docker /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor \
    && chmod -R g=u /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor

# Setup application root
ENV APP_PATH /app
RUN mkdir -p ${APP_PATH}
WORKDIR ${APP_PATH}

# App source code
COPY . .
RUN mkdir -p files log

# Set permissions
RUN chown -R docker:docker ${APP_PATH} \
    && chmod -R g=u ${APP_PATH}

USER docker
EXPOSE 8042

# Init script
CMD ["/bin/bash", "-c", "init.sh"]
