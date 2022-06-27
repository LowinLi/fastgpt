#!/bin/sh
gunicorn -w ${GUNICORN_WORKER} --threads ${GUNICORN_THREADS} -b 0.0.0.0:${PORT} --timeout 1800 web:app