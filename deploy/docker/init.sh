#!/bin/bash

if [ -n "$RUN_JOB" ] ; then
    echo "Run job [$RUN_JOB]..."
    # the job name is the first word; everything after the first space is one argument (JSON with spaces)
    read -r job_name job_params <<< "$RUN_JOB"
    python /app/job.py "$job_name" ${job_params:+"$job_params"}
else
    SERVICES=()
    APP_CRONTAB=$(echo "$APP_CRONTAB" | xargs)
    if [ -n "$APP_CRONTAB" ] && [ -s /etc/crontab ] && [ -n "$(grep '[^[:space:]]' /etc/crontab)" ] ; then
        SERVICES+=("crontab")
    fi
    APP_API=$(echo "$APP_API" | xargs)
    if [ -n "$APP_API" ] ; then
        SERVICES+=("nginx" "gunicorn")
    fi
    APP_QUEUE=$(echo "$APP_QUEUE" | xargs)
    if [ -n "$APP_QUEUE" ] ; then
        SERVICES+=("worker")
    fi

    if [ ${#SERVICES[@]} -gt 0 ] ; then
        export GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"
        echo "Run supervisord..."
        supervisord -c /etc/supervisord.conf

        for SERVICE in ${SERVICES[@]}; do
            echo "Start [$SERVICE]..."
            supervisorctl start $SERVICE
        done
    fi

    echo "App ready..."
    tail -f /dev/null
fi
