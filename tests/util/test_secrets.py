"""Docker secrets, and the precedence between a mounted file and the environment."""

from pathlib import Path

import pytest

import util.secrets as secrets
from util.secrets import (
    SECRET_NAMES,
    get_secret,
    load_secrets_to_environ,
    mounted_secrets,
)


@pytest.fixture
def mounted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand in for /run/secrets, which exists only inside a container."""
    monkeypatch.setattr(secrets, "DOCKER_SECRETS", tmp_path)
    return tmp_path


def test_a_mounted_secret_wins_over_the_environment(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment that mounted a secret meant it.

    Preferring a stale environment variable would make the secret look applied
    when it was not -- the failure would surface as an auth error somewhere else
    entirely.
    """
    (mounted / "POSTGRES_PASSWORD").write_text("from-the-file\n")
    monkeypatch.setenv("POSTGRES_PASSWORD", "from-the-environment")
    assert get_secret("POSTGRES_PASSWORD") == "from-the-file"


def test_the_environment_is_used_when_nothing_is_mounted(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path every deployment that does not use secrets takes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert get_secret("OPENAI_API_KEY") == "sk-from-env"


def test_a_blank_secret_counts_as_absent(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`init-docker-secrets` creates placeholders containing a single space.

    A blank password reaching Postgres fails somewhere far less obvious than
    here, so an empty file falls through to the environment rather than
    overriding it with nothing.
    """
    (mounted / "POSTGRES_PASSWORD").write_text("   \n")
    monkeypatch.setenv("POSTGRES_PASSWORD", "real-password")
    assert get_secret("POSTGRES_PASSWORD") == "real-password"


def test_a_missing_secret_falls_back_to_the_default(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert get_secret("NOT_SET_ANYWHERE", "fallback") == "fallback"
    assert get_secret("NOT_SET_ANYWHERE") is None


def test_loading_writes_only_file_backed_values(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    (mounted / "TAVILY_API_KEY").write_text("tvly-secret")
    (mounted / "CLOUDFLARE_SECRET_KEY").write_text("  ")  # placeholder
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    load_secrets_to_environ(
        ["TAVILY_API_KEY", "CLOUDFLARE_SECRET_KEY", "OPENAI_API_KEY"]
    )

    assert os.environ["TAVILY_API_KEY"] == "tvly-secret"
    assert os.environ["OPENAI_API_KEY"] == "sk-from-env", "left exactly as it was"
    assert "CLOUDFLARE_SECRET_KEY" not in os.environ, "a blank file writes nothing"


def test_mounted_secrets_lists_names_without_reading_them(mounted: Path) -> None:
    """What the startup log is built from.

    Separate from loading because a function that has read secret bodies cannot
    return anything provably safe to log -- CodeQL flagged the first version of
    this as "Clear-text logging of sensitive information" on a line that logged
    only names, and it was right to: those names were derived from a value the
    contents had flowed into.
    """
    (mounted / "TAVILY_API_KEY").write_text("tvly-secret")
    assert mounted_secrets(["TAVILY_API_KEY", "OPENAI_API_KEY"]) == ["TAVILY_API_KEY"]


def test_mounted_secrets_is_empty_outside_a_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(secrets, "DOCKER_SECRETS", tmp_path / "nope")
    assert mounted_secrets(SECRET_NAMES) == []


def test_loading_does_not_blank_an_environment_variable(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name with no mounted file must not be cleared."""
    import os

    value = "real-password"
    monkeypatch.setenv("POSTGRES_PASSWORD", value)
    load_secrets_to_environ(["POSTGRES_PASSWORD"])
    assert os.environ["POSTGRES_PASSWORD"] == value


def test_missing_secrets_directory_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a container /run/secrets does not exist at all."""
    monkeypatch.setattr(secrets, "DOCKER_SECRETS", tmp_path / "nope")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    load_secrets_to_environ(SECRET_NAMES)
    assert get_secret("OPENAI_API_KEY") == "sk-from-env"


def test_the_secret_names_are_ones_the_deployment_actually_uses() -> None:
    """A name here that no template mentions would never be mounted."""
    template = Path(__file__).parent.parent.parent / "env_template"
    declared = template.read_text()
    unknown = [
        n
        for n in SECRET_NAMES
        if n not in declared and n not in {"CHAINLIT_AUTH_SECRET", "LITERAL_API_KEY"}
    ]
    assert not unknown, f"not in env_template: {unknown}"


def test_db_uri_uses_the_socket_and_never_tcp(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postgres is started with no TCP listener at all, so a host:port URI
    cannot connect. Verified against a real container: `psql -h 127.0.0.1`
    gives "Connection refused" while the socket URI below returns a row."""
    monkeypatch.setattr(secrets, "VAULT_TOKEN_FILE", Path("/nonexistent"))
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")

    uri = secrets.get_db_uri("chainlit")

    assert uri == "postgresql://postgres:pw@/chainlit?host=/sockets/postgres/"
    assert "5432" not in uri
    assert "@postgres:" not in uri, "no host:port form anywhere"


def test_db_uri_names_the_dialect_when_asked(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLAlchemy needs postgresql+psycopg; psycopg and langgraph need it absent."""
    monkeypatch.setattr(secrets, "VAULT_TOKEN_FILE", Path("/nonexistent"))
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    uri = secrets.get_db_uri("chainlit", driver="psycopg")
    assert uri is not None
    assert uri.startswith("postgresql+psycopg://")


def test_db_uri_escapes_credentials(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vault-issued passwords contain punctuation; an unescaped one silently
    truncates the URI at the first '@' or '/'."""
    monkeypatch.setattr(secrets, "VAULT_TOKEN_FILE", Path("/nonexistent"))
    monkeypatch.setenv("POSTGRES_USER", "user@host")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word")

    uri = secrets.get_db_uri("chainlit")
    assert uri is not None
    assert "user%40host" in uri
    assert "p%40ss%2Fword" in uri


@pytest.mark.parametrize("db_name", [None, ""])
def test_no_database_configured_returns_none(
    db_name: str | None, mounted: Path
) -> None:
    """How every caller decides whether database-backed features exist."""
    assert secrets.get_db_uri(db_name) is None


def test_no_password_anywhere_returns_none(
    mounted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(secrets, "VAULT_TOKEN_FILE", Path("/nonexistent"))
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    assert secrets.get_db_uri("chainlit") is None
