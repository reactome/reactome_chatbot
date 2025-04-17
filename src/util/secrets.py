import os
import urllib.parse
from pathlib import Path
from time import sleep
from typing import Iterable

import hvac
import hvac.exceptions
import requests_unixsocket

from util.logging import logging


DOCKER_SECRETS = Path("/run/secrets")
POSTGRES_SOCKET = "/sockets/postgres/"

VAULT_SOCKET = "/sockets/vault/vault.sock"
VAULT_TOKEN_FILE = Path("/tokens/postgres-app/token")
VAULT_URI = "http+unix://{encoded_socket}".format(
    encoded_socket=urllib.parse.quote(VAULT_SOCKET, safe=""),
)

socket_session = requests_unixsocket.Session()


def get_secret(
    secret_name: str, default: str | None = None
) -> str | None:
    """Get a secret by name from Docker Secrets or an environment variable."""

    secret_path: Path = DOCKER_SECRETS / secret_name
    secret: str | None = None

    if secret_path.exists():
        with open(secret_path, "r") as secret_file:
            secret = secret_file.read().strip()

    if secret is None:
        secret = os.getenv(secret_name, default)

    return secret


def load_secrets_to_environ(secret_names: Iterable[str]) -> None:
    """Load secrets into os.environ"""

    for secret_name in secret_names:
        secret: str | None = get_secret(secret_name)
        if secret:
            os.environ[secret_name] = secret

def get_db_uri(db_name: str | None) -> str | None:
    if not db_name:
        return None

    username: str
    password: str | None

    if VAULT_TOKEN_FILE.exists():
        with VAULT_TOKEN_FILE.open("r") as token_file:
            token: str = token_file.read().strip()
        vault_client = hvac.Client(
            url=VAULT_URI,
            token=token,
            session=socket_session,
        )
        while True:
            try:
                credentials = vault_client.secrets.database.generate_credentials(
                    name="postgres-app",
                )
                break
            except hvac.exceptions.VaultDown:
                logging.warning("Waiting for Vault unseal...")
            except Exception as e:
                logging.error("Vault error. Waiting for Vault unseal...", e)
            sleep(10)
        response = credentials
        logging.warning(response)
        username, password = response["data"].values()
    else:
        username = os.getenv("POSTGRES_USER", "postgres")
        password = get_secret("POSTGRES_PASSWORD", "postgres")

    if password is None:
        return None

    return f"postgresql://{username}:{password}@localhost/{db_name}?host={POSTGRES_SOCKET}"
