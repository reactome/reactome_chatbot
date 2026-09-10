"""Read secrets from Docker Secrets, falling back to the environment.

Harvested from the `security-improvements` branch (Adam Wright), which pairs it
with a Vault deployment. Only the Docker Secrets half is taken here: it works
with the compose file that exists today, needs no new services, and changes
nothing for a deployment that does not use secrets.

The Vault half of that branch is deliberately not included -- it reaches Postgres
over a unix socket the current compose does not mount, waits forever on an
unsealed Vault, and logs the credentials it is issued.

Why bother: an environment variable is visible in `docker inspect`, in the
process environment of every child, and in a `.env` file on disk that has to be
chmod'ed by hand -- which is what was done to this host's config on 2026-09-04.
A Docker secret is a file mounted only into the containers that declare it.
"""

import os
import urllib.parse
from collections.abc import Iterable
from pathlib import Path
from time import sleep

from util.logging import logging

# Where the Docker engine mounts secrets inside a container. Absent outside one,
# which is the whole fallback path.
DOCKER_SECRETS = Path("/run/secrets")

# Postgres is reached over a unix socket, never TCP: the server is started with
# `--auth-host reject`, so there is no network path to the database at all.
POSTGRES_SOCKET = "/sockets/postgres/"

# Vault, when deployed, issues short-lived Postgres credentials and writes a
# token for this application to a shared volume. Absent means "no Vault", which
# is the development path.
VAULT_SOCKET = "/sockets/vault/vault.sock"
VAULT_TOKEN_FILE = Path("/tokens/postgres-app/token")
VAULT_ROLE = "postgres-app"
VAULT_URI = "http+unix://" + urllib.parse.quote(VAULT_SOCKET, safe="")

# Vault starts sealed and is unsealed by an operator. Waiting is correct;
# waiting forever is not -- the original of this looped on VaultDown with no
# bound, so a Vault that never came up left the container running and silent
# instead of failing. Roughly five minutes, then say so and stop.
VAULT_UNSEAL_ATTEMPTS = 30
VAULT_UNSEAL_INTERVAL_SECONDS = 10


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret, preferring the mounted file over the environment.

    The file wins because a deployment that has gone to the trouble of mounting
    a secret means it; silently preferring a stale environment variable would
    make the secret look applied when it was not.

    An empty or whitespace-only file is treated as absent. `init-docker-secrets`
    on the security-improvements branch creates secrets containing a single
    space as placeholders, and a blank password that reached Postgres would fail
    somewhere far less obvious than here.
    """
    secret_path = DOCKER_SECRETS / name
    try:
        contents = secret_path.read_text().strip()
    except OSError:
        # Not mounted, not readable, or not in a container at all.
        contents = ""
    return contents or os.getenv(name, default)


def mounted_secrets(names: Iterable[str]) -> list[str]:
    """Which of `names` have a non-empty file mounted. Reads no contents.

    Deliberately separate from `load_secrets_to_environ`, and deliberately
    stat-only. The caller wants to log which secrets a deployment supplied, and
    a function that has read the secret bodies cannot return anything a reader
    -- human or static analyser -- can be sure is safe to log. CodeQL flagged
    exactly that on the first version of this: "Clear-text logging of sensitive
    information", on a line that logged only names. It was right to; the names
    were derived from a value the file contents had flowed into.

    Size rather than contents, so a placeholder of one space still counts as
    absent for `get_secret` while showing here as present-but-blank would not:
    a whitespace-only file has non-zero size, so it is compared after stripping
    is impossible -- and that is the point. This answers "was a file mounted",
    not "is it usable".
    """
    present: list[str] = []
    for name in names:
        try:
            if (DOCKER_SECRETS / name).stat().st_size > 0:
                present.append(name)
        except OSError:
            continue
    return present


def load_secrets_to_environ(names: Iterable[str]) -> None:
    """Copy mounted secrets into os.environ.

    Everything downstream -- chainlit, psycopg, the OpenAI client -- reads
    os.environ, so the mounted files are placed there once at startup rather
    than teaching each consumer about /run/secrets.

    Returns nothing on purpose: see `mounted_secrets`. Only file-backed values
    are written, so a name already in the environment and not mounted is left
    exactly as it is.
    """
    for name in names:
        secret_path = DOCKER_SECRETS / name
        try:
            contents = secret_path.read_text().strip()
        except OSError:
            continue
        if contents:
            os.environ[name] = contents


# The values worth mounting rather than passing as environment variables. Names
# match env_template, so a deployment can move one across without renaming it.
SECRET_NAMES = (
    "OPENAI_API_KEY",
    "POSTGRES_PASSWORD",
    "PGADMIN_DEFAULT_PASSWORD",
    "CLOUDFLARE_SECRET_KEY",
    "TAVILY_API_KEY",
    "CHAINLIT_AUTH_SECRET",
    "LITERAL_API_KEY",
)


def _vault_credentials() -> tuple[str, str]:
    """Ask Vault for a fresh Postgres username and password.

    Never logs the response. The original of this called
    `logging.warning(response)` with Vault's reply, which contains exactly the
    credentials it has just issued.
    """
    import hvac
    import hvac.exceptions
    import requests_unixsocket

    token = VAULT_TOKEN_FILE.read_text().strip()
    client = hvac.Client(
        url=VAULT_URI, token=token, session=requests_unixsocket.Session()
    )

    for attempt in range(1, VAULT_UNSEAL_ATTEMPTS + 1):
        try:
            issued = client.secrets.database.generate_credentials(name=VAULT_ROLE)
            return issued["data"]["username"], issued["data"]["password"]
        except hvac.exceptions.VaultDown:
            logging.warning(
                f"Vault is sealed; waiting for an operator to unseal it "
                f"({attempt}/{VAULT_UNSEAL_ATTEMPTS})"
            )
        except Exception as exc:
            # The message, never the response body.
            logging.error(
                f"Vault error while issuing credentials: {type(exc).__name__}"
            )
        sleep(VAULT_UNSEAL_INTERVAL_SECONDS)

    raise RuntimeError(
        f"Vault did not issue Postgres credentials after "
        f"{VAULT_UNSEAL_ATTEMPTS} attempts. It is configured "
        f"({VAULT_TOKEN_FILE} exists) but not usable -- unseal it, or remove the "
        "token to fall back to POSTGRES_PASSWORD."
    )


def get_db_uri(db_name: str | None, *, driver: str | None = None) -> str | None:
    """Build a Postgres URI over the unix socket, preferring Vault credentials.

    `driver` selects a SQLAlchemy dialect (`psycopg`); omit it for a plain libpq
    URI, which is what psycopg and langgraph want.

    Returns None when `db_name` is empty or no password can be found, which is
    how every caller decides whether database-backed features are configured at
    all.
    """
    if not db_name:
        return None

    if VAULT_TOKEN_FILE.exists():
        username, password = _vault_credentials()
    else:
        username = os.getenv("POSTGRES_USER", "postgres")
        found = get_secret("POSTGRES_PASSWORD")
        if found is None:
            return None
        password = found

    scheme = f"postgresql+{driver}" if driver else "postgresql"
    return (
        # safe="" so a "/" is escaped too. quote() leaves it alone by default,
        # and Vault-issued passwords are random punctuation -- an unescaped
        # slash silently truncates the URI at the database name.
        f"{scheme}://{urllib.parse.quote(username, safe='')}"
        f":{urllib.parse.quote(password, safe='')}"
        f"@/{db_name}?host={POSTGRES_SOCKET}"
    )
