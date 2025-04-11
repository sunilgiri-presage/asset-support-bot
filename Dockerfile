# Dockerfile (for web service)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if needed (e.g., gcc, libpq-dev)
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Copy and set permissions for wait-for-it.sh (if still used for migrations)
COPY wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh

# Copy and set permissions for entrypoint script (if needed for initial tasks)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Collect static files (optional here; you might run this as part of deployment)
RUN python manage.py collectstatic --noinput

# Expose Gunicorn's port
EXPOSE 8000

# Run Gunicorn to serve your Django app
CMD ["gunicorn", "asset_support_bot.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
