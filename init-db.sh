#!/bin/bash
set -e

# Ensure the database and user are created
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER asset_support_user WITH PASSWORD 'Sunilgiri@1#';
    GRANT ALL PRIVILEGES ON DATABASE asset_support_db TO asset_support_user;
EOSQL
