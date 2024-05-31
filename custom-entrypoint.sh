#!/bin/bash

# Source the environment variables from the .env file
set -a
. /docker-entrypoint-initdb.d/.env
set +a

# Execute the original entrypoint script with all arguments passed to this script
exec docker-entrypoint.sh "$@"
