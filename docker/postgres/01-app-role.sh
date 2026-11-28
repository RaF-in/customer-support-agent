#!/bin/sh
# Creates the application's everyday Postgres role.
#
# Why: the compose POSTGRES_USER (`app`) is a superuser, and superusers
# ignore row-level security — so the application pool must connect as an
# ordinary role for the tenant-isolation policies to apply.
#
# When: the postgres image runs this file ONCE, while it initialises an
# empty data volume. An existing volume never re-runs it (see the
# "Two Postgres roles" section of the README for how to add the role to an
# existing database, or how to reset the volume).
#
# Inputs (set on the postgres service in docker-compose.yml):
#   APP_RW_USER, APP_RW_PASSWORD, POSTGRES_USER, POSTGRES_DB
set -eu

# --set + :"var" / :'var' make psql quote the role name and the password.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --set app_rw_user="$APP_RW_USER" --set app_rw_password="$APP_RW_PASSWORD" <<'SQL'
CREATE ROLE :"app_rw_user"
    LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS
    PASSWORD :'app_rw_password';
SQL
