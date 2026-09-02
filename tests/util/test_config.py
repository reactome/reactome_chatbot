"""Config loading, with an emphasis on what happens when config.yml is wrong.

`Config.from_yaml` returning None switches off every config-driven feature,
including the message quota, so the fallback path matters as much as the happy
one: a typo in config.yml must not quietly remove rate limiting.
"""

from pathlib import Path

import pytest

from util.config_yml import CONFIG_DEFAULT_YML, Config

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID = """
profiles: ["React-to-Me"]
features:
  postprocessing:
    enabled: true
    user_group: all
usage_limits:
  message_rates:
    - users: ["all"]
      max_messages: 5
      interval: 1h
messages:
  hello:
    message: hi
    trigger:
      event: on_chat_start
"""


@pytest.fixture(autouse=True)
def _in_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """CONFIG_DEFAULT_YML is a relative path, so the fallback needs the repo root."""
    monkeypatch.chdir(REPO_ROOT)


def test_the_shipped_default_config_is_valid() -> None:
    """If this fails, every fallback below lands on None and features silently die."""
    config = Config.from_yaml(CONFIG_DEFAULT_YML)
    assert config is not None
    assert config.usage_limits.message_rates, "defaults must carry a message quota"


def test_valid_config_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(VALID)
    config = Config.from_yaml(path)
    assert config is not None
    assert config.usage_limits.message_rates[0].max_messages == 5


def test_missing_config_falls_back_to_defaults(tmp_path: Path) -> None:
    config = Config.from_yaml(tmp_path / "nope.yml")
    assert config is not None
    assert config.usage_limits.message_rates


def test_invalid_config_falls_back_to_defaults_not_to_none(tmp_path: Path) -> None:
    """The important one: a bad interval must not disable the limiter.

    Before, a ValidationError returned None, and `message_rate_limited(None)`
    reports "not limited" -- so one typo removed the quota for everybody.
    """
    path = tmp_path / "config.yml"
    path.write_text(VALID.replace("interval: 1h", "interval: 1hr"))
    config = Config.from_yaml(path)
    assert config is not None, "must fall back to defaults, not to None"
    assert config.usage_limits.message_rates, "the fallback still carries a quota"


def test_config_yml_as_a_directory_falls_back(tmp_path: Path) -> None:
    """docker-compose bind-mounts ./config.yml; if it is absent on the host,
    Docker creates a *directory* there and open() raises IsADirectoryError."""
    path = tmp_path / "config.yml"
    path.mkdir()
    assert Config.from_yaml(path) is not None


@pytest.mark.parametrize("content", ["", "\n", "just a string", "[1, 2, 3]"])
def test_non_mapping_config_falls_back(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yml"
    path.write_text(content)
    assert Config.from_yaml(path) is not None


def test_unknown_profile_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(VALID.replace('["React-to-Me"]', '["Nonexistent Profile"]'))
    config = Config.from_yaml(path)
    # falls back rather than starting with a profile the agent cannot build
    assert config is not None
    assert "Nonexistent Profile" not in [str(p) for p in config.profiles]
