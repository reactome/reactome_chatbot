#!/bin/sh
set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

# Create "postgres-app" role
vault policy write postgres-app /config/postgres-app-policy.hcl
vault token create -field=token -policy=postgres-app | vault login -token-only - > /tokens/postgres-app/token

# Create "postgres-admin" vault token and login with it
vault policy write postgres-admin /config/postgres-admin-policy.hcl
export VAULT_TOKEN=$(
    vault token create -field=token -policy=postgres-admin | vault login -token-only -
)

# Enable database secrets engine with Postgres DB
vault secrets enable database
vault write database/config/postgres \
    plugin_name=postgresql-database-plugin \
    connection_url="postgresql://{{username}}:{{password}}@postgres?host=/sockets/postgres" \
    allowed_roles="*" \
    username="postgres" \
    password=$POSTGRES_PASSWORD

# Rotate $POSTGRES_PASSWORD out of use ASAP
vault write -force database/rotate-root/postgres

# Create the database role for postgres-app
vault write database/roles/postgres-app \
    db_name=postgres \
    creation_statements=@/config/app-roles.sql \
    default_ttl=1h \
    max_ttl=24h
