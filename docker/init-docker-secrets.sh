#!/bin/bash

set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

# Check if a file path is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <compose.yaml>"
  exit 1
fi

yaml_file="$1"

# Parse compose.yaml to initialize blank secrets
awk '
BEGIN { inside_secrets = 0; parent_key = ""; }
/^secrets:/ { inside_secrets = 1; next; }
inside_secrets && /^[[:space:]]{2}[A-Za-z0-9_]+:/ {
  parent_key = $1;
  sub(/:$/, "", parent_key);
}
inside_secrets && /^[[:space:]]{4}external:[[:space:]]*true/ {
  print parent_key;
}
' "$yaml_file" | while read -r secret_name; do
  echo "Creating blank Docker Secret: $secret_name"
  echo -n " " | docker secret create "$secret_name" -
done
