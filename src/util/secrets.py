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
from collections.abc import Iterable
from pathlib import Path

# Where the Docker engine mounts secrets inside a container. Absent outside one,
# which is the whole fallback path.
DOCKER_SECRETS = Path("/run/secrets")


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


def load_secrets_to_environ(names: Iterable[str]) -> list[str]:
    """Copy mounted secrets into os.environ; return the names that came from a file.

    Everything downstream -- chainlit, psycopg, the OpenAI client -- reads
    os.environ, so the mounted files are placed there once at startup rather
    than teaching each consumer about /run/secrets.

    Only file-backed values are written. A name already in the environment and
    not mounted is left exactly as it is.
    """
    loaded: list[str] = []
    for name in names:
        secret_path = DOCKER_SECRETS / name
        try:
            contents = secret_path.read_text().strip()
        except OSError:
            continue
        if contents:
            os.environ[name] = contents
            loaded.append(name)
    return loaded


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
