#!/bin/bash
set -e

if [ "$1" = "gunicorn" ]; then
  # Wait for the database to be ready
  /app/wait-for-it.sh db:5432 -t 30
  # Run Django database migrations
  python manage.py makemigrations
  python manage.py migrate
  # Collect static files
  python manage.py collectstatic --noinput
fi

# Execute the container's main command
exec "$@"