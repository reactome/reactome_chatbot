"""The captcha must read its secret from the same place everything else does.

`get_secret` prefers a mounted Docker secret file over the environment, and says
why: "a deployment that has gone to the trouble of mounting a secret means it".
`bin/chat-fastapi.py` loads CLOUDFLARE_SECRET_KEY that way -- and then decides
whether the captcha is enabled, and what secret to send to Cloudflare, by reading
`os.environ` directly.

So a deployment that mounts the secret rather than exporting it gets the captcha
silently switched off. That is Principle IV: configuration that cannot be honoured
must stop the process, not substitute something plausible.
"""

from pathlib import Path

import pytest

from util.secrets import get_secret


def test_get_secret_prefers_a_mounted_file_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("util.secrets.DOCKER_SECRETS", tmp_path)
    (tmp_path / "CLOUDFLARE_SECRET_KEY").write_text("from-the-mounted-file\n")
    monkeypatch.setenv("CLOUDFLARE_SECRET_KEY", "from-the-environment")
    assert get_secret("CLOUDFLARE_SECRET_KEY") == "from-the-mounted-file"


def test_a_mounted_secret_is_invisible_to_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The divergence itself, stated plainly.

    With the secret mounted and not exported, `get_secret` finds it and
    `os.environ` does not. Any code that asks os.environ whether the captcha is
    configured concludes it is not, and skips it.
    """
    import os

    monkeypatch.setattr("util.secrets.DOCKER_SECRETS", tmp_path)
    (tmp_path / "CLOUDFLARE_SECRET_KEY").write_text("mounted-only\n")
    monkeypatch.delenv("CLOUDFLARE_SECRET_KEY", raising=False)

    assert get_secret("CLOUDFLARE_SECRET_KEY") == "mounted-only"
    assert os.getenv("CLOUDFLARE_SECRET_KEY") is None


def test_chat_fastapi_never_reads_the_captcha_secret_from_os_environ() -> None:
    """A source-level guard, because the script cannot be imported.

    `bin/chat-fastapi.py` is hyphenated, so it is not importable as a module and
    the middleware cannot be exercised directly here. What can be checked is that
    it never reaches around `get_secret` for this particular secret, which is the
    mistake being prevented.
    """
    source = Path("bin/chat-fastapi.py").read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "CLOUDFLARE_SECRET_KEY" in line
        and ("os.getenv" in line or "os.environ" in line)
    ]
    assert not offenders, (
        "the captcha secret must come from get_secret, which prefers a mounted "
        f"Docker secret file over the environment: {offenders}"
    )
