#!/bin/sh
set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

# Check if $1 and $2 are provided, if not, prompt the user
if [ -z "${1:-}" ]; then
  read -p "Enter the number of key shares (total # of keys generated): " key_shares
else
  key_shares="$1"
fi

if [ -z "${2:-}" ]; then
  read -p "Enter the key threshold (minimum # of keys to unseal): " key_threshold
else
  key_threshold="$2"
fi

# Run the vault operator init command and pipe the output to less, capturing errors
vault operator init -key-shares="$key_shares" -key-threshold="$key_threshold" 2>&1 | less
